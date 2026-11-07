//! Python bindings: `axtb._axtb`, a thin layer over `axtb-driver`.
//!
//! Keyword arguments fill an `axtb_driver::Request`; the driver checks and runs it, and its
//! `Response` comes back as a dict with the keys of the CLI's JSON (arrays as numpy arrays).
//! Errors: `axtb.InputError` (a `ValueError`) for input, unsupported elements and GFN-FF
//! topologies, `axtb.ScfNotConvergedError` (a `RuntimeError`, with `iterations`,
//! `energy_change` and `residual`), `RuntimeError` for numerical failures and panics; each has
//! `kind`, the driver's error kind.

use axtb_driver::{Method, Request, Response, Task, Temperatures};
use numpy::{AllowTypeChange, PyArray1, PyArray2, PyArrayLike2, PyArrayMethods};
use pyo3::exceptions::{PyRuntimeError, PyRuntimeWarning, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyList};

pyo3::create_exception!(
    axtb,
    InputError,
    PyValueError,
    "Invalid input: structure, charge and spin, settings, names, settings that do not apply to the method, \
     elements without parameters, a structure GFN-FF cannot set up. `kind` says which."
);
pyo3::create_exception!(
    axtb,
    ScfNotConvergedError,
    PyRuntimeError,
    "The SCF did not converge: `iterations`, `energy_change` (Eh) and `residual` of its last attempt."
);

/// The Python exception of a driver error, with the error's `kind` (and `iterations`,
/// `energy_change`, `residual` of an SCF that did not converge; `z` of an unsupported element)
/// as attributes.
fn py_err(py: Python<'_>, e: axtb_driver::Error) -> PyErr {
    let msg = e.to_string();
    let err = match &e {
        axtb_driver::Error::ScfNotConverged { .. } => ScfNotConvergedError::new_err(msg),
        e if e.is_input() => InputError::new_err(msg),
        _ => PyRuntimeError::new_err(msg),
    };
    let value = err.value(py);
    // setting attributes of a new exception instance cannot fail
    let _ = value.setattr("kind", e.kind());
    match e {
        axtb_driver::Error::ScfNotConverged { iterations, energy_change, residual } => {
            let _ = value.setattr("iterations", iterations);
            let _ = value.setattr("energy_change", energy_change);
            let _ = value.setattr("residual", residual);
        }
        axtb_driver::Error::UnsupportedElement { z, .. } => {
            let _ = value.setattr("z", z);
        }
        _ => {}
    }
    err
}

/// `f` without the GIL; its error as the Python exception, a panic as `RuntimeError`.
fn guarded<T: Send>(py: Python<'_>, f: impl FnOnce() -> axtb_driver::Result<T> + Send) -> PyResult<T> {
    match py.allow_threads(|| std::panic::catch_unwind(std::panic::AssertUnwindSafe(f))) {
        Ok(Ok(v)) => Ok(v),
        Ok(Err(e)) => Err(py_err(py, e)),
        Err(panic) => {
            let what = panic
                .downcast_ref::<&str>()
                .map(|s| s.to_string())
                .or_else(|| panic.downcast_ref::<String>().cloned())
                .unwrap_or_default();
            Err(PyRuntimeError::new_err(format!("internal error (a bug in axtb; please report it): {what}")))
        }
    }
}

/// Rows of three (positions, dipole derivatives) from any array-like of shape (`rows`, 3).
fn rows3(a: &PyArrayLike2<'_, f64, AllowTypeChange>, rows: usize, what: &str) -> PyResult<Vec<[f64; 3]>> {
    let v = a.as_array();
    if v.shape() != [rows, 3] {
        return Err(InputError::new_err(format!("{what} must have shape ({rows}, 3), not {:?}", v.shape())));
    }
    Ok(v.rows().into_iter().map(|r| [r[0], r[1], r[2]]).collect())
}

/// A Python warning of `category` with `msg`.
fn warn(py: Python<'_>, msg: &str) -> PyResult<()> {
    let msg = std::ffi::CString::new(msg.replace('\0', " ")).unwrap_or_default();
    PyErr::warn(py, py.get_type::<PyRuntimeWarning>().as_any(), &msg, 1)
}

/// The JSON value as Python objects (null, bool, number, string, list, dict).
fn to_py(py: Python<'_>, v: &serde_json::Value) -> PyResult<PyObject> {
    use serde_json::Value;
    Ok(match v {
        Value::Null => py.None(),
        Value::Bool(b) => PyBool::new(py, *b).to_owned().into_any().unbind(),
        Value::Number(n) => match (n.as_u64(), n.as_i64(), n.as_f64()) {
            (Some(u), _, _) => u.into_pyobject(py)?.into_any().unbind(),
            (_, Some(i), _) => i.into_pyobject(py)?.into_any().unbind(),
            (_, _, Some(f)) => f.into_pyobject(py)?.into_any().unbind(),
            _ => py.None(),
        },
        Value::String(s) => s.into_pyobject(py)?.into_any().unbind(),
        Value::Array(a) => {
            let l = PyList::empty(py);
            for x in a {
                l.append(to_py(py, x)?)?;
            }
            l.into_any().unbind()
        }
        Value::Object(m) => {
            let d = PyDict::new(py);
            for (k, x) in m {
                d.set_item(k, to_py(py, x)?)?;
            }
            d.into_any().unbind()
        }
    })
}

fn array1<'py>(py: Python<'py>, v: &[f64]) -> Bound<'py, PyArray1<f64>> {
    PyArray1::from_slice(py, v)
}

/// An (n, K) array of rows.
fn array2<'py, const K: usize>(py: Python<'py>, rows: &[[f64; K]]) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let flat: Vec<f64> = rows.iter().flatten().copied().collect();
    PyArray1::from_vec(py, flat).reshape([rows.len(), K])
}

fn matrix(py: Python<'_>, m: axtb_driver::Matrix) -> PyResult<Bound<'_, PyArray2<f64>>> {
    PyArray1::from_vec(py, m.data).reshape([m.rows, m.cols])
}

/// The response as a dict: the keys of the CLI's JSON, arrays as numpy arrays (`gradient`
/// (N, 3), `hessian` (3N, 3N), `normal_modes` (modes, N, 3), ...).
fn response_dict(py: Python<'_>, mut r: Response) -> PyResult<Bound<'_, PyDict>> {
    // the large arrays go straight into numpy, not through JSON values
    let hessian = r.hessian.as_mut().map(|h| std::mem::take(&mut h.hessian));
    let modes = r.vibrations.as_mut().map(|v| std::mem::take(&mut v.normal_modes));
    let bond_orders = r.scf.as_mut().and_then(|s| s.bond_orders.take());
    let value = serde_json::to_value(&r).map_err(|e| PyRuntimeError::new_err(format!("response: {e}")))?;
    let out = to_py(py, &value)?.into_bound(py).downcast_into::<PyDict>()?;
    let nat = r.charges.as_ref().map(Vec::len).or_else(|| r.vibrations.as_ref().map(|v| v.masses.len())).unwrap_or(0);
    let set1 = |k: &str, v: &Option<Vec<f64>>| -> PyResult<()> {
        if let Some(v) = v {
            out.set_item(k, array1(py, v))?;
        }
        Ok(())
    };
    set1("charges", &r.charges)?;
    if let Some(d) = &r.dipole {
        out.set_item("dipole", array1(py, d))?;
    }
    if let Some(g) = &r.gradient {
        out.set_item("gradient", array2(py, g)?)?;
    }
    if let Some(s) = &r.scf {
        out.set_item("quadrupole", array1(py, &s.quadrupole))?;
        out.set_item("shell_charges", array1(py, &s.shell_charges))?;
        out.set_item("atomic_dipoles", array2(py, &s.atomic_dipoles)?)?;
        out.set_item("atomic_quadrupoles", array2(py, &s.atomic_quadrupoles)?)?;
        out.set_item("orbital_energies_alpha", array1(py, &s.orbital_energies_alpha))?;
        out.set_item("orbital_energies_beta", array1(py, &s.orbital_energies_beta))?;
        out.set_item("occupations_alpha", array1(py, &s.occupations_alpha))?;
        out.set_item("occupations_beta", array1(py, &s.occupations_beta))?;
    }
    if let Some(b) = bond_orders {
        out.set_item("bond_orders", matrix(py, b)?)?;
    }
    if let Some(h) = hessian {
        out.set_item("hessian", matrix(py, h)?)?;
    }
    if let Some(d) = r.hessian.as_ref().and_then(|h| h.dipole_derivatives.as_ref()) {
        out.set_item("dipole_derivatives", array2(py, d)?)?;
    }
    if let Some(v) = &r.vibrations {
        out.set_item("frequencies", array1(py, &v.frequencies))?;
        out.set_item("reduced_masses", array1(py, &v.reduced_masses))?;
        out.set_item("force_constants", array1(py, &v.force_constants))?;
        out.set_item("masses", array1(py, &v.masses))?;
        set1("ir_intensities", &v.ir_intensities)?;
    }
    if let Some(m) = modes {
        let flat: Vec<f64> = m.iter().flatten().flatten().copied().collect();
        out.set_item("normal_modes", PyArray1::from_vec(py, flat).reshape([m.len(), nat, 3])?)?;
    }
    if let Some(g) = &r.gfnff {
        let t = &g.topology;
        let d = PyDict::new(py);
        let bonds: Vec<i64> = t.bonds.iter().flatten().map(|&i| i as i64).collect();
        d.set_item("bonds", PyArray1::from_vec(py, bonds).reshape([t.bonds.len(), 2])?)?;
        d.set_item("pi_bond_orders", array1(py, &t.pi_bond_orders))?;
        d.set_item("bond_types", PyArray1::from_vec(py, t.bond_types.iter().map(|&b| b as i64).collect()))?;
        d.set_item("hybridization", PyArray1::from_vec(py, t.hybridization.iter().map(|&h| h as i64).collect()))?;
        d.set_item("pi_atoms", PyArray1::from_slice(py, &t.pi_atoms))?;
        d.set_item("fragments", PyArray1::from_vec(py, t.fragments.iter().map(|&f| f as i64).collect()))?;
        d.set_item("fragment_charges", array1(py, &t.fragment_charges))?;
        d.set_item("topology_charges", array1(py, &t.topology_charges))?;
        d.set_item("n_angles", t.n_angles)?;
        d.set_item("n_torsions", t.n_torsions)?;
        out.set_item("topology", d)?;
    }
    Ok(out)
}

/// Run on a driver calculator and warn about a run-time LAPACK that was not used.
fn finish(py: Python<'_>, r: Response) -> PyResult<Bound<'_, PyDict>> {
    if let Some(w) = axtb_driver::eigensolver_warning(&r) {
        warn(py, &w)?;
    }
    response_dict(py, r)
}

/// The Molden request of a `molden` argument: `None`/`False` none, `True` or `"spherical"`
/// spherical functions, `"cartesian"` Cartesian ones (the text under `"molden"`).
fn molden_into(req: &mut Request, molden: Option<&Bound<'_, PyAny>>) -> PyResult<()> {
    let Some(v) = molden.filter(|v| !v.is_none()) else { return Ok(()) };
    if let Ok(b) = v.downcast::<PyBool>() {
        req.molden = b.is_true();
        return Ok(());
    }
    if let Ok(s) = v.extract::<String>() {
        let f = axtb_driver::names::molden_functions(&s).map_err(|e| py_err(v.py(), e))?;
        req.molden = true;
        req.molden_cartesian = f == axtb::MoldenFunctions::Cartesian;
        return Ok(());
    }
    Err(InputError::new_err("molden must be True, False, \"spherical\" or \"cartesian\""))
}

/// One temperature or a sequence of them.
fn temperatures(t: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Temperatures>> {
    let Some(t) = t.filter(|t| !t.is_none()) else { return Ok(None) };
    if let Ok(x) = t.extract::<f64>() {
        return Ok(Some(Temperatures::One(x)));
    }
    t.extract::<Vec<f64>>()
        .map(|v| Some(Temperatures::Many(v)))
        .map_err(|_| InputError::new_err("temperature must be a number or a sequence of numbers (K)"))
}

/// The method's settings of `Calculator` and `singlepoint`; `None` means the method's default.
struct Settings<'a> {
    method: &'a str,
    charge: f64,
    uhf: Option<i64>,
    accuracy: Option<f64>,
    etemp: Option<f64>,
    max_iter: Option<usize>,
    threads: usize,
    dispersion: Option<bool>,
    conventions: Option<String>,
    alpb: Option<String>,
    gbsa: Option<String>,
    solvation_state: Option<String>,
    mixer: Option<String>,
    solvation_correction: Option<String>,
    correction_conventions: Option<String>,
    fragment_charges: Option<Vec<f64>>,
    topology_positions: Option<PyArrayLike2<'a, f64, AllowTypeChange>>,
    guess: Option<String>,
    scf_trace: bool,
}

impl Settings<'_> {
    /// The request of these settings for `numbers` at `positions`.
    fn request(self, numbers: Vec<usize>, positions: &PyArrayLike2<'_, f64, AllowTypeChange>) -> PyResult<Request> {
        let nat = numbers.len();
        let positions = rows3(positions, nat, "positions")?;
        let topology_positions = self.topology_positions.map(|t| rows3(&t, nat, "topology_positions")).transpose()?;
        Ok(Request {
            numbers,
            positions,
            charge: self.charge,
            uhf: self.uhf,
            method: Some(self.method.to_string()),
            conventions: self.conventions,
            accuracy: self.accuracy,
            etemp: self.etemp,
            max_iter: self.max_iter,
            threads: Some(self.threads),
            dispersion: self.dispersion,
            mixer: self.mixer,
            scf_trace: self.scf_trace,
            alpb: self.alpb,
            gbsa: self.gbsa,
            solvation_state: self.solvation_state,
            solvation_correction: self.solvation_correction,
            correction_conventions: self.correction_conventions,
            guess: self.guess,
            fragment_charges: self.fragment_charges,
            topology_positions,
            ..Request::default()
        })
    }
}

/// Warn about more SCF threads than help for one single point.
fn warn_threads(py: Python<'_>, method: Method, threads: usize) -> PyResult<()> {
    match axtb_driver::threads_warning(method, threads, false) {
        Some(w) => warn(py, &w),
        None => Ok(()),
    }
}

/// A converged SCF wavefunction kept in memory: `Calculator.guess_state()` returns it,
/// `Calculator.set_guess_state(state)` starts another calculator's next SCF from it (Rust:
/// `axtb_driver::Calculator::guess` and `set_guess`). GFN2-xTB: shell charges and atomic
/// moments; g-xTB: what its program's restart file holds (v1.1 the density matrices; 2.0.1 shell
/// charges, atomic moments, magnetizations and densities); the composite: those of its three
/// parts. Opaque; `method` and `natoms` say what it belongs to.
#[pyclass(module = "axtb", frozen)]
struct ScfState {
    inner: axtb_driver::ScfState,
}

#[pymethods]
impl ScfState {
    /// The method of the calculation the wavefunction belongs to.
    #[getter]
    fn method(&self) -> &'static str {
        self.inner.method().name()
    }

    /// Atoms of the molecule.
    #[getter]
    fn natoms(&self) -> usize {
        self.inner.natoms()
    }

    fn __repr__(&self) -> String {
        format!("ScfState(method={:?}, natoms={})", self.inner.method().name(), self.inner.natoms())
    }
}

/// A method bound to one molecule (atomic numbers, charge, unpaired electrons); see the
/// package docs and the README for every argument. Reuses the setup and the previous geometry's
/// wavefunction between calls. Positions in bohr (any array-like of shape (N, 3)); results in
/// Hartree and bohr, as dicts with the keys of the CLI's JSON.
///
/// `method`: `"gfn2"` (GFN2-xTB), `"gfnff"` (GFN-FF), `"gxtb-v1.1"` or `"gxtb-v2.0.1"` (g-xTB);
/// names ignore case and anything but letters and digits. Every other argument is keyword-only
/// and `None` means the method's default; a setting that does not apply to the method raises
/// `axtb.InputError`. `guess`: where each SCF starts (`None`: the conventions' choice;
/// `"previous"`, or the method's own start `"eeq-bc"`, `"eeq"`, `"fresh"`). GFN-FF sets its
/// topology up from `positions` (or `topology_positions`) once.
#[pyclass(module = "axtb")]
struct Calculator {
    inner: axtb_driver::Calculator,
}

#[pymethods]
impl Calculator {
    #[new]
    #[pyo3(signature = (method, numbers, positions, *, charge=0.0, uhf=None, accuracy=None, etemp=None, max_iter=None,
                        threads=1, dispersion=None, conventions=None, alpb=None, gbsa=None, solvation_state=None,
                        mixer=None, solvation_correction=None, correction_conventions=None, fragment_charges=None,
                        topology_positions=None, guess=None, scf_trace=false))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        method: &str,
        numbers: Vec<usize>,
        positions: PyArrayLike2<'_, f64, AllowTypeChange>,
        charge: f64,
        uhf: Option<i64>,
        accuracy: Option<f64>,
        etemp: Option<f64>,
        max_iter: Option<usize>,
        threads: usize,
        dispersion: Option<bool>,
        conventions: Option<String>,
        alpb: Option<String>,
        gbsa: Option<String>,
        solvation_state: Option<String>,
        mixer: Option<String>,
        solvation_correction: Option<String>,
        correction_conventions: Option<String>,
        fragment_charges: Option<Vec<f64>>,
        topology_positions: Option<PyArrayLike2<'_, f64, AllowTypeChange>>,
        guess: Option<String>,
        scf_trace: bool,
    ) -> PyResult<Self> {
        let req = Settings {
            method,
            charge,
            uhf,
            accuracy,
            etemp,
            max_iter,
            threads,
            dispersion,
            conventions,
            alpb,
            gbsa,
            solvation_state,
            mixer,
            solvation_correction,
            correction_conventions,
            fragment_charges,
            topology_positions,
            guess,
            scf_trace,
        }
        .request(numbers, &positions)?;
        let inner = guarded(py, || axtb_driver::Calculator::new(&req))?;
        Ok(Self { inner })
    }

    /// The method (canonical name).
    #[getter]
    fn method(&self) -> &'static str {
        self.inner.method().name()
    }

    /// The method's conventions (canonical name).
    #[getter]
    fn conventions(&self) -> &'static str {
        self.inner.conventions()
    }

    /// Energy (and with `gradient=True` the gradient) at `positions` (bohr). `molden`: `True`
    /// (or `"spherical"`) or `"cartesian"` adds the text of a Molden file of the converged
    /// orbitals (`"molden"`); `bond_orders=True` adds the Mayer-Wiberg bond orders (N, N).
    #[pyo3(signature = (positions, gradient=false, molden=None, bond_orders=false))]
    fn singlepoint<'py>(
        &mut self,
        py: Python<'py>,
        positions: PyArrayLike2<'py, f64, AllowTypeChange>,
        gradient: bool,
        molden: Option<&Bound<'py, PyAny>>,
        bond_orders: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let pos = rows3(&positions, self.inner.numbers().len(), "positions")?;
        let mut req = Request { gradient, bond_orders, ..Request::default() };
        molden_into(&mut req, molden)?;
        let task = Task::from_request(&req, self.inner.method()).map_err(|e| py_err(py, e))?;
        warn_threads(py, self.inner.method(), self.inner.threads())?;
        let inner = &mut self.inner;
        let r = guarded(py, || inner.compute(&pos, &task))?;
        finish(py, r)
    }

    /// Hessian at `positions` (bohr) with the single point there and the vibrational analysis:
    /// the response of the CLI's `--hess`. `method`: `"analytic"` (default; g-xTB v1.1
    /// `"numerical"`), `"response"` or `"numerical"`; `step` (bohr) of the finite-difference
    /// methods; `accuracy` the SCF accuracy of the displacements (numerical) or the reference.
    /// The analysis settings are `vibrational_analysis`'s (its `conventions` is
    /// `analysis_conventions` here; default: the calculation's).
    #[pyo3(signature = (positions, step=None, accuracy=None, method=None, *, vibrations=true, temperature=None,
                        pressure=None, rotor_cutoff=None, imaginary_cutoff=None, frequency_scale=None,
                        symmetry_number=None, symmetry_threshold=None, masses=None, thermo=true,
                        analysis_conventions=None, molden=None, bond_orders=false))]
    #[allow(clippy::too_many_arguments)]
    fn hessian<'py>(
        &mut self,
        py: Python<'py>,
        positions: PyArrayLike2<'py, f64, AllowTypeChange>,
        step: Option<f64>,
        accuracy: Option<f64>,
        method: Option<String>,
        vibrations: bool,
        temperature: Option<&Bound<'py, PyAny>>,
        pressure: Option<f64>,
        rotor_cutoff: Option<f64>,
        imaginary_cutoff: Option<f64>,
        frequency_scale: Option<f64>,
        symmetry_number: Option<u32>,
        symmetry_threshold: Option<f64>,
        masses: Option<Vec<f64>>,
        thermo: bool,
        analysis_conventions: Option<String>,
        molden: Option<&Bound<'py, PyAny>>,
        bond_orders: bool,
    ) -> PyResult<Bound<'py, PyDict>> {
        let pos = rows3(&positions, self.inner.numbers().len(), "positions")?;
        let mut req = Request {
            hessian: true,
            hessian_method: method,
            hessian_step: step,
            hessian_accuracy: accuracy,
            vibrations: (!vibrations).then_some(false),
            thermo: (!thermo).then_some(false),
            temperature: temperatures(temperature)?,
            pressure,
            rotor_cutoff,
            imaginary_cutoff,
            frequency_scale,
            symmetry_number,
            symmetry_threshold,
            masses,
            analysis_conventions,
            bond_orders,
            ..Request::default()
        };
        molden_into(&mut req, molden)?;
        let task = Task::from_request(&req, self.inner.method()).map_err(|e| py_err(py, e))?;
        let inner = &mut self.inner;
        let r = guarded(py, || inner.compute(&pos, &task))?;
        finish(py, r)
    }

    /// The converged wavefunction of the last SCF (`None` before the first, and for GFN-FF),
    /// to start another calculator for the same method and molecule from it
    /// (`set_guess_state`). Rust: `axtb_driver::Calculator::guess`.
    fn guess_state(&self) -> Option<ScfState> {
        self.inner.guess().map(|inner| ScfState { inner })
    }

    /// Start the next SCF from `state` (the `guess_state()` of a calculator for the same
    /// method and molecule), whatever `guess` says. Rust: `axtb_driver::Calculator::set_guess`.
    fn set_guess_state(&mut self, py: Python<'_>, state: &ScfState) -> PyResult<()> {
        self.inner.set_guess(&state.inner).map_err(|e| py_err(py, e))
    }

    /// Forget the previous wavefunction (the next SCF starts from the method's own guess);
    /// `gxtb_only`: of the composite only the g-xTB part's.
    #[pyo3(signature = (gxtb_only=false))]
    fn reset_guess(&mut self, gxtb_only: bool) {
        self.inner.reset_guess(gxtb_only);
    }

    /// GFN2-xTB: start the next SCF from given moments: shell charges (nsh,), atomic dipoles
    /// (N, 3) and atomic quadrupoles (N, 6), e.g. the `shell_charges`, `atomic_dipoles` and
    /// `atomic_quadrupoles` of a previous result or another program's converged moments. Rust:
    /// `axtb_driver::Calculator::set_moments`.
    fn set_guess(
        &mut self,
        py: Python<'_>,
        shell_charges: Vec<f64>,
        atomic_dipoles: PyArrayLike2<'_, f64, AllowTypeChange>,
        atomic_quadrupoles: PyArrayLike2<'_, f64, AllowTypeChange>,
    ) -> PyResult<()> {
        let nat = self.inner.numbers().len();
        let dp = rows3(&atomic_dipoles, nat, "atomic_dipoles")?;
        let qp = atomic_quadrupoles.as_array();
        if qp.shape() != [nat, 6] {
            return Err(InputError::new_err(format!("atomic_quadrupoles must have shape ({nat}, 6)")));
        }
        let qp: Vec<[f64; 6]> = qp.rows().into_iter().map(|r| std::array::from_fn(|k| r[k])).collect();
        self.inner.set_moments(shell_charges, dp, qp).map_err(|e| py_err(py, e))
    }
}

/// One stateless calculation; see `Calculator` for the arguments. `gradient=True` adds the
/// gradient; `molden`, `bond_orders` as for `Calculator.singlepoint`; GFN-FF:
/// `topology_positions` (N, 3; bohr) sets the topology up from another structure.
#[pyfunction]
#[pyo3(signature = (numbers, positions, *, charge=0.0, uhf=None, method="gfn2", gradient=false, accuracy=None,
                    etemp=None, max_iter=None, threads=1, dispersion=None, conventions=None, alpb=None, gbsa=None,
                    solvation_state=None, mixer=None, molden=None, solvation_correction=None,
                    correction_conventions=None, fragment_charges=None, topology_positions=None, bond_orders=false,
                    scf_trace=false))]
#[allow(clippy::too_many_arguments)]
fn singlepoint<'py>(
    py: Python<'py>,
    numbers: Vec<usize>,
    positions: PyArrayLike2<'py, f64, AllowTypeChange>,
    charge: f64,
    uhf: Option<i64>,
    method: &str,
    gradient: bool,
    accuracy: Option<f64>,
    etemp: Option<f64>,
    max_iter: Option<usize>,
    threads: usize,
    dispersion: Option<bool>,
    conventions: Option<String>,
    alpb: Option<String>,
    gbsa: Option<String>,
    solvation_state: Option<String>,
    mixer: Option<String>,
    molden: Option<&Bound<'py, PyAny>>,
    solvation_correction: Option<String>,
    correction_conventions: Option<String>,
    fragment_charges: Option<Vec<f64>>,
    topology_positions: Option<PyArrayLike2<'py, f64, AllowTypeChange>>,
    bond_orders: bool,
    scf_trace: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let mut req = Settings {
        method,
        charge,
        uhf,
        accuracy,
        etemp,
        max_iter,
        threads,
        dispersion,
        conventions,
        alpb,
        gbsa,
        solvation_state,
        mixer,
        solvation_correction,
        correction_conventions,
        fragment_charges,
        topology_positions,
        guess: None,
        scf_trace,
    }
    .request(numbers, &positions)?;
    req.gradient = gradient;
    req.bond_orders = bond_orders;
    molden_into(&mut req, molden)?;
    if let Ok(m) = Method::parse(method) {
        warn_threads(py, m, threads)?;
    }
    let r = guarded(py, || axtb_driver::run(&req))?;
    finish(py, r)
}

/// Vibrational analysis of a Hessian from any source (method-agnostic): `numbers` (N),
/// `positions` (N, 3) in bohr, `hessian` (3N, 3N) in Eh/bohr² (any array-likes). Optional:
/// `dipole_derivatives` (3N, 3; e) for IR intensities, `energy` (Eh) for total enthalpies and
/// free energies, `masses` (N, u). `conventions` (the request's `analysis_conventions`):
/// `"axtb"` (default), `"xtb"` (xtb 6.8.0) or `"xtb-6.7.1"` (also `"gxtb"`). Thermochemistry
/// (`thermo=True`): `temperature` (K, a number or a sequence; results at the last), `pressure`
/// (Pa), `rotor_cutoff` and `imaginary_cutoff` (cm⁻¹), `frequency_scale`, `symmetry_number`,
/// `symmetry_threshold` (bohr); `None` means the default. Returns the analysis keys of a
/// Hessian's response.
#[pyfunction]
#[pyo3(signature = (numbers, positions, hessian, *, dipole_derivatives=None, energy=None, conventions=None,
                    masses=None, temperature=None, pressure=None, rotor_cutoff=None, imaginary_cutoff=None,
                    frequency_scale=None, symmetry_number=None, symmetry_threshold=None, thermo=true))]
#[allow(clippy::too_many_arguments)]
fn vibrational_analysis<'py>(
    py: Python<'py>,
    numbers: Vec<usize>,
    positions: PyArrayLike2<'py, f64, AllowTypeChange>,
    hessian: PyArrayLike2<'py, f64, AllowTypeChange>,
    dipole_derivatives: Option<PyArrayLike2<'py, f64, AllowTypeChange>>,
    energy: Option<f64>,
    conventions: Option<String>,
    masses: Option<Vec<f64>>,
    temperature: Option<&Bound<'py, PyAny>>,
    pressure: Option<f64>,
    rotor_cutoff: Option<f64>,
    imaginary_cutoff: Option<f64>,
    frequency_scale: Option<f64>,
    symmetry_number: Option<u32>,
    symmetry_threshold: Option<f64>,
    thermo: bool,
) -> PyResult<Bound<'py, PyDict>> {
    let nat = numbers.len();
    let n3 = 3 * nat;
    let positions = rows3(&positions, nat, "positions")?;
    let h = hessian.as_array();
    if h.shape() != [n3, n3] {
        return Err(InputError::new_err(format!("hessian must have shape ({n3}, {n3})")));
    }
    let h: Vec<Vec<f64>> = h.rows().into_iter().map(|r| r.to_vec()).collect();
    let dipole_derivatives = dipole_derivatives.map(|d| rows3(&d, n3, "dipole_derivatives")).transpose()?;
    let req = Request {
        numbers,
        positions,
        analyze_hessian: Some(h),
        dipole_derivatives,
        energy,
        analysis_conventions: conventions,
        masses,
        thermo: (!thermo).then_some(false),
        temperature: temperatures(temperature)?,
        pressure,
        rotor_cutoff,
        imaginary_cutoff,
        frequency_scale,
        symmetry_number,
        symmetry_threshold,
        ..Request::default()
    };
    let r = guarded(py, || axtb_driver::run(&req))?;
    response_dict(py, r)
}

/// The canonical name of a method name (`"GFN2-xTB"` -> `"gfn2"`, `"g-xTB 2.0"` ->
/// `"gxtb-v2.0.1"`); `axtb.InputError` for an unknown one.
#[pyfunction]
fn canonical_method(py: Python<'_>, name: &str) -> PyResult<&'static str> {
    axtb_driver::canonical_method(name).map_err(|e| py_err(py, e))
}

/// Why g-xTB alone refuses an implicit solvent, and the composite to use instead.
#[pyfunction]
fn gxtb_solvent_hint(py: Python<'_>, method: &str) -> PyResult<String> {
    let m = Method::parse(method).map_err(|e| py_err(py, e))?;
    Ok(axtb_driver::gxtb_solvent_hint(m))
}

/// axtb's extension module. `axtb/__init__.py` checks the CPU before it imports this module (an
/// x86-64-v3 build runs AVX instructions as soon as the module loads).
#[pymodule]
fn _axtb(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add("InputError", py.get_type::<InputError>())?;
    m.add("ScfNotConvergedError", py.get_type::<ScfNotConvergedError>())?;
    m.add_function(wrap_pyfunction!(singlepoint, m)?)?;
    m.add_function(wrap_pyfunction!(vibrational_analysis, m)?)?;
    m.add_function(wrap_pyfunction!(canonical_method, m)?)?;
    m.add_function(wrap_pyfunction!(gxtb_solvent_hint, m)?)?;
    m.add_class::<Calculator>()?;
    m.add_class::<ScfState>()?;
    Ok(())
}
