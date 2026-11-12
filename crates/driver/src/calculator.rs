//! A method bound to one molecule: dispatch to GFN2-xTB, g-xTB (and its solvation composite)
//! and GFN-FF, and the response of each.

use crate::error::{Error, Result};
use crate::names::{self, Method};
use crate::request::Request;
use crate::response::{
    self, Attempt, Composite, Details, Gfnff, HessianPart, Matrix, Named, Part, Parts, Response, Scf, Topology,
    Vibrations,
};
use crate::task::Task;
use axtb::method::{Fingerprint, Method as _, Restart, ScfData, Want, Wavefunction, WavefunctionData};
use axtb::vibrations::Conventions as VibConventions;
use axtb::ToMolden;
use std::time::Instant;

/// The calculator of a method.
enum Engine {
    Gfn2(Box<axtb::Calculator>),
    #[cfg(feature = "gxtb")]
    Gxtb(Box<axtb_gxtb::Calculator>),
    /// g-xTB in implicit solvent (the solvation composite).
    #[cfg(feature = "gxtb")]
    Composite(Box<axtb_gxtb::CompositeCalculator>),
    Gfnff(Box<axtb_gfnff::Calculator>),
}

/// A converged wavefunction of one calculator, to start another calculator's SCF from it
/// ([`Calculator::guess`], [`Calculator::set_guess`]). Opaque.
#[derive(Clone)]
pub struct ScfState {
    inner: StateInner,
    method: Method,
    natoms: usize,
}

#[derive(Clone)]
#[allow(clippy::large_enum_variant)] // (until the driver holds the trait object)
enum StateInner {
    Gfn2(Wavefunction),
    #[cfg(feature = "gxtb")]
    Gxtb(Wavefunction),
    /// the composite's parts': g-xTB, GFN2 gas phase, GFN2 solvated
    #[cfg(feature = "gxtb")]
    Composite(Wavefunction),
}

impl ScfState {
    /// The method of the calculation the wavefunction belongs to.
    pub fn method(&self) -> Method {
        self.method
    }

    /// Atoms of the molecule.
    pub fn natoms(&self) -> usize {
        self.natoms
    }
}

/// A method bound to one molecule (atomic numbers, charge, unpaired electrons) with its
/// settings, built from a [`Request`]: [`Calculator::compute`] runs a [`Task`] at any
/// geometry and starts each SCF from the previous geometry's wavefunction (as the method's
/// conventions or the request's `guess` say).
pub struct Calculator {
    engine: Engine,
    method: Method,
    conventions: &'static str,
    numbers: Vec<usize>,
    charge: f64,
    uhf: Option<usize>,
    threads: usize,
    /// The analysis conventions of the calculation's conventions.
    analysis: VibConventions,
}

fn input(msg: impl Into<String>) -> Error {
    Error::Input(msg.into())
}

/// Refuse the settings in `bad` for `method`.
fn refuse(method: Method, bad: &[&str]) -> Result<()> {
    if bad.is_empty() {
        Ok(())
    } else {
        Err(input(format!("{method} does not support {}", bad.join(", "))))
    }
}

/// The implicit-solvent setting of a request, if any.
fn solvation(req: &Request) -> Result<Option<axtb::SolvationSpec>> {
    let state = req.solvation_state.as_deref().map(names::solvation_state).transpose()?;
    let (model, solvent) = match (&req.alpb, &req.gbsa) {
        (Some(_), Some(_)) => return Err(input("give either alpb or gbsa, not both")),
        (Some(s), None) => (axtb::SolvationModel::Alpb, s),
        (None, Some(s)) => (axtb::SolvationModel::Gbsa, s),
        (None, None) => {
            if state.is_some() {
                return Err(input("solvation_state is a setting of the implicit solvent (alpb or gbsa)"));
            }
            return Ok(None);
        }
    };
    Ok(Some(axtb::SolvationSpec { model, solvent: solvent.clone(), state: state.unwrap_or_default() }))
}

/// The hint for a g-xTB method given an implicit solvent alone: g-xTB has no solvent
/// parameters; the solvation composite is the way.
pub fn gxtb_solvent_hint(method: Method) -> String {
    format!(
        "{method} has no implicit-solvent parameters (the g-xTB program stops with --alpb/--gbsa). For g-xTB in \
         solvent use the solvation composite E = E(g-xTB) + [E(GFN2-xTB + ALPB/GBSA) - E(GFN2-xTB)]: add \
         solvation_correction \"gfn2\" (CLI: --solvation-correction gfn2) to the solvent (alpb or gbsa)"
    )
}

impl Calculator {
    /// Set up the method of `req` for its molecule at `req.positions` (bohr): its settings are
    /// checked (unknown names, settings of other methods and impossible values are
    /// [`Error::Input`]); the fields of what to compute are not read (see
    /// [`Task::from_request`]).
    pub fn new(req: &Request) -> Result<Self> {
        let method = Method::parse(req.method.as_deref().unwrap_or("gfn2"))?;
        let nat = req.numbers.len();
        if req.positions.len() != nat {
            return Err(input(format!("positions must have one row per atom ({nat}), not {}", req.positions.len())));
        }
        if req.uhf.is_some_and(|u| u < 0) {
            return Err(input(format!("uhf (unpaired electrons) must not be negative, not {}", req.uhf.unwrap())));
        }
        if req.threads == Some(0) {
            return Err(input("threads must be at least 1"));
        }
        if req.max_iter == Some(0) {
            return Err(input("max_iter must be at least 1"));
        }
        if method != Method::Gfnff && (req.fragment_charges.is_some() || req.topology_positions.is_some()) {
            return Err(input("fragment_charges and topology_positions are GFN-FF settings"));
        }
        if !method.is_gxtb() && (req.solvation_correction.is_some() || req.correction_conventions.is_some()) {
            return Err(input(
                "solvation_correction and correction_conventions are settings of g-xTB in solvent; GFN2-xTB has its \
                 own implicit solvent (alpb or gbsa alone)",
            ));
        }
        let threads = req.threads.unwrap_or(1);
        let uhf = req.uhf.map(|u| u as usize);
        let guess = req.guess.as_deref().map(names::guess).transpose()?;
        let (engine, conventions, analysis) = match method {
            Method::Gfn2 => {
                let conventions =
                    req.conventions.as_deref().map(names::gfn2_conventions).transpose()?.unwrap_or_default();
                let mut opts = axtb::Options::default();
                if let Some(a) = req.accuracy {
                    opts.scf.accuracy = a;
                }
                opts.scf.etemp = req.etemp;
                opts.scf.max_iter = req.max_iter;
                opts.mixing.mixer = req.mixer.as_deref().map(names::mixer).transpose()?;
                match guess {
                    Some(names::Guess::Sad) => opts.initial_guess = axtb::InitialGuess::Sad,
                    Some(names::Guess::EeqBc) => {
                        return Err(input(
                            "GFN2-xTB starts from EEQ charges or neutral atoms: guess \"eeq\" or \"sad\"",
                        ))
                    }
                    _ => {}
                }
                // a named start at every geometry is the method's own start at every geometry
                opts.scf.restart = guess.map(|g| match g {
                    names::Guess::Previous => Restart::Previous,
                    _ => Restart::Fresh,
                });
                opts.threads = threads;
                opts.trace = req.scf_trace;
                opts.dispersion = req.dispersion.unwrap_or(true);
                opts.conventions = conventions;
                opts.solvation = solvation(req)?;
                let calc = axtb::Calculator::new(
                    axtb::Hamiltonian::Gfn2,
                    &req.numbers,
                    &req.positions,
                    req.charge,
                    uhf,
                    opts,
                )?;
                (Engine::Gfn2(Box::new(calc)), conventions.label(), conventions.into())
            }
            Method::Gfnff => {
                let mut bad = Vec::new();
                if req.alpb.is_some() || req.gbsa.is_some() || req.solvation_state.is_some() {
                    bad.push("implicit solvation");
                }
                if req.dispersion == Some(false) {
                    bad.push("switching off dispersion");
                }
                if req.accuracy.is_some()
                    || req.etemp.is_some()
                    || req.max_iter.is_some()
                    || req.mixer.is_some()
                    || req.guess.is_some()
                    || req.scf_trace
                {
                    bad.push("SCF settings (accuracy, etemp, max_iter, mixer, guess, scf_trace)");
                }
                refuse(method, &bad)?;
                let (ff, analysis, label) = names::gfnff_conventions(req.conventions.as_deref().unwrap_or("axtb"))?;
                let setup = match &req.topology_positions {
                    Some(t) if t.len() != nat => {
                        return Err(input(format!("topology_positions must have one row per atom ({nat})")))
                    }
                    Some(t) => t.as_slice(),
                    None => &req.positions,
                };
                let opts = axtb_gfnff::Options {
                    threads,
                    fragment_charges: req.fragment_charges.clone(),
                    conventions: ff,
                    ..Default::default()
                };
                let calc = axtb_gfnff::Calculator::new(&req.numbers, setup, req.charge, opts)?;
                (Engine::Gfnff(Box::new(calc)), label, analysis)
            }
            Method::GxtbV1_1 | Method::GxtbV2_0_1 => {
                #[cfg(feature = "gxtb")]
                {
                    let (engine, conventions) = gxtb_engine(req, method, threads, uhf, guess)?;
                    // the g-xTB 2.0.1 program is a 6.7.1-based xtb build
                    let analysis = match conventions {
                        axtb_gxtb::Conventions::Gxtb => VibConventions::Xtb671,
                        axtb_gxtb::Conventions::Axtb => VibConventions::Axtb,
                    };
                    (engine, names::gxtb_conventions_label(conventions), analysis)
                }
                #[cfg(not(feature = "gxtb"))]
                {
                    let _ = (uhf, guess);
                    return Err(input(format!("{method}: this axtb was built without g-xTB (feature `gxtb`)")));
                }
            }
        };
        Ok(Self {
            engine,
            method,
            conventions,
            numbers: req.numbers.clone(),
            charge: req.charge,
            uhf,
            threads,
            analysis,
        })
    }

    /// The method.
    pub fn method(&self) -> Method {
        self.method
    }

    /// The canonical name of the method's conventions.
    pub fn conventions(&self) -> &'static str {
        self.conventions
    }

    /// Atomic numbers of the molecule.
    pub fn numbers(&self) -> &[usize] {
        &self.numbers
    }

    /// Threads of the calculator.
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Run `task` at `positions` (bohr). With a Hessian, the response is that of the single
    /// point at `positions` (energy, gradient, charges, orbitals; g-xTB's SCF from its own
    /// start, as without the Hessian) plus the Hessian, its dipole derivatives and its
    /// vibrational analysis.
    pub fn compute(&mut self, positions: &[[f64; 3]], task: &Task) -> Result<Response> {
        let t0 = Instant::now();
        let nat = self.numbers.len();
        if positions.len() != nat {
            return Err(input(format!("positions must have one row per atom ({nat}), not {}", positions.len())));
        }
        let mut out = Response {
            program: response::PROGRAM,
            version: response::VERSION,
            method: Some(self.method.name()),
            conventions: self.conventions,
            ..Default::default()
        };
        let molden = task.molden.is_some();
        let orbitals = match &mut self.engine {
            Engine::Gfn2(calc) => {
                let hess = match &task.hessian {
                    Some(h) => {
                        let o = axtb::HessianOptions { method: h.method, step: h.step, accuracy: h.accuracy };
                        Some(calc.hessian(positions, &o)?)
                    }
                    None => None,
                };
                let want =
                    Want { gradient: task.gradient || hess.is_some(), orbitals: molden, bond_orders: task.bond_orders };
                let sp = calc.singlepoint(positions, &want)?;
                out.hessian = hess.map(hessian_part);
                report_into(&mut out, sp)
            }
            #[cfg(feature = "gxtb")]
            Engine::Gxtb(calc) => {
                let hess = match &task.hessian {
                    Some(h) => {
                        // afterwards g-xTB holds no wavefunction: the single point below starts
                        // from its own guess, as without the Hessian
                        let o = axtb::HessianOptions { method: h.method, step: h.step, accuracy: h.accuracy };
                        Some(calc.hessian(positions, &o)?)
                    }
                    None => None,
                };
                let want =
                    Want { gradient: task.gradient || hess.is_some(), orbitals: molden, bond_orders: task.bond_orders };
                let sp = calc.singlepoint(positions, &want)?;
                out.hessian = hess.map(hessian_part);
                report_into(&mut out, sp)
            }
            #[cfg(feature = "gxtb")]
            Engine::Composite(calc) => {
                let hess = match &task.hessian {
                    // afterwards the g-xTB part holds no wavefunction: its single point below
                    // starts from its own guess, as without the Hessian
                    Some(h) => {
                        let o = axtb::HessianOptions { method: h.method, step: h.step, accuracy: h.accuracy };
                        Some(calc.hessian(positions, &o)?)
                    }
                    None => None,
                };
                let want =
                    Want { gradient: task.gradient || hess.is_some(), orbitals: molden, bond_orders: task.bond_orders };
                let mut sp = calc.singlepoint(positions, &want)?;
                let report = sp.extras.composite.take().expect("the composite's report");
                out.composite = Some(composite_response(*report, hess.as_ref().map(|h| h.parts.as_slice())));
                out.hessian = hess.map(hessian_part);
                report_into(&mut out, sp)
            }
            Engine::Gfnff(calc) => {
                let hess = match &task.hessian {
                    Some(h) => {
                        let o = axtb::HessianOptions { method: h.method, step: h.step, accuracy: h.accuracy };
                        Some(hessian_part(calc.hessian(positions, &o)?))
                    }
                    None => None,
                };
                let r = calc.singlepoint(positions, &Want::gradient(task.gradient || hess.is_some()))?;
                let g = r.extras.gfnff.expect("GFN-FF reports its topology");
                let t = g.topology;
                out.energy = Some(r.energy);
                out.gradient = r.gradient;
                out.charges = Some(r.charges);
                out.dipole = Some(r.dipole);
                out.terms = Some(Named::new(r.terms.iter().map(|t| (t.name, t.value)).collect()));
                out.timings = Some(Named::new(r.timings));
                out.gfnff = Some(Gfnff {
                    n_hbonds: g.n_hbonds[0] + g.n_hbonds[1],
                    n_xbonds: g.n_xbonds,
                    topology: Topology {
                        bonds: t.bonds,
                        pi_bond_orders: t.pi_bond_orders,
                        bond_types: t.bond_types,
                        hybridization: t.hybridization,
                        pi_atoms: t.pi_atoms,
                        fragments: t.fragments,
                        fragment_charges: t.fragment_charges,
                        topology_charges: t.topology_charges,
                        n_angles: t.n_angles,
                        n_torsions: t.n_torsions,
                    },
                });
                out.hessian = hess;
                None
            }
        };
        check_finite(&out)?;
        if let Some(m) = &task.molden {
            let orbitals = orbitals.ok_or_else(|| Error::Numerical("the calculation returned no orbitals".into()))?;
            let title = format!("axtb {} {}", response::VERSION, self.method);
            let text = orbitals.to_molden(m.functions, &title)?;
            if let Some(path) = &m.file {
                std::fs::write(path, &text).map_err(|e| input(format!("cannot write the Molden file {path}: {e}")))?;
            }
            out.molden_file = m.file.clone();
            out.molden = m.text.then_some(text);
        }
        if let (Some(h), Some(hp)) = (&task.hessian, &out.hessian) {
            if let Some(v) = &h.vibrations {
                let opts = v.options(self.analysis, self.method == Method::Gfnff);
                let a = axtb::vibrations::analyze(
                    &self.numbers,
                    positions,
                    &hp.hessian.data,
                    hp.dipole_derivatives.as_deref(),
                    &opts,
                )?;
                out.vibrations = Some(Vibrations::new(a, out.energy));
            }
        }
        out.seconds = Some(t0.elapsed().as_secs_f64());
        Ok(out)
    }

    /// The converged wavefunction of the last SCF (`None` before the first, and for GFN-FF),
    /// for another calculator of the same method and molecule ([`Calculator::set_guess`]).
    pub fn guess(&self) -> Option<ScfState> {
        let inner = match &self.engine {
            Engine::Gfn2(c) => c.wavefunction().map(StateInner::Gfn2),
            #[cfg(feature = "gxtb")]
            Engine::Gxtb(c) => c.wavefunction().map(StateInner::Gxtb),
            #[cfg(feature = "gxtb")]
            Engine::Composite(c) => c.wavefunction().map(StateInner::Composite),
            Engine::Gfnff(_) => None,
        }?;
        Some(ScfState { inner, method: self.method, natoms: self.numbers.len() })
    }

    /// Start the next SCF from `state` (from [`Calculator::guess`] of a calculator for the same
    /// method and molecule), whatever the `guess` setting says.
    pub fn set_guess(&mut self, state: &ScfState) -> Result<()> {
        if state.method != self.method || state.natoms != self.numbers.len() {
            return Err(input(format!(
                "a {} wavefunction of {} atoms cannot start a {} SCF of {} atoms",
                state.method,
                state.natoms,
                self.method,
                self.numbers.len()
            )));
        }
        match (&mut self.engine, &state.inner) {
            (Engine::Gfn2(c), StateInner::Gfn2(g)) => c.set_wavefunction(g)?,
            #[cfg(feature = "gxtb")]
            (Engine::Gxtb(c), StateInner::Gxtb(g)) => c.set_wavefunction(g)?,
            #[cfg(feature = "gxtb")]
            (Engine::Composite(c), StateInner::Composite(w)) => c.set_wavefunction(w)?,
            _ => return Err(input("the wavefunction belongs to another kind of calculation")),
        }
        Ok(())
    }

    /// Start the next GFN2-xTB SCF from given moments: shell charges, atomic dipoles and
    /// quadrupoles (e.g. those of a previous response, or another program's).
    pub fn set_moments(
        &mut self,
        shell_charges: Vec<f64>,
        dipoles: Vec<[f64; 3]>,
        quadrupoles: Vec<[f64; 6]>,
    ) -> Result<()> {
        match &mut self.engine {
            Engine::Gfn2(c) => {
                let data = ScfData {
                    shell_charges,
                    atomic_dipoles: dipoles,
                    atomic_quadrupoles: quadrupoles,
                    ..ScfData::default()
                };
                let fingerprint = Fingerprint::new(c.info().name, &self.numbers, self.charge, self.uhf);
                Ok(c.set_wavefunction(&Wavefunction::new(fingerprint, WavefunctionData::Scf(data)))?)
            }
            Engine::Gfnff(_) => Err(input("gfnff has no SCF; starting moments are GFN2-only")),
            #[cfg(feature = "gxtb")]
            _ => Err(input(
                "starting moments are GFN2-only; g-xTB starts from a whole wavefunction (the guess of another \
                 calculator)",
            )),
        }
    }

    /// Forget the previous wavefunction(s): the next SCF starts from the method's own guess.
    /// `gxtb_only`: of the composite only the g-xTB part's (for every other method nothing
    /// but g-xTB's).
    pub fn reset_guess(&mut self, gxtb_only: bool) {
        match (&mut self.engine, gxtb_only) {
            (Engine::Gfn2(c), false) => c.reset_wavefunction(),
            #[cfg(feature = "gxtb")]
            (Engine::Gxtb(c), _) => c.reset_wavefunction(),
            #[cfg(feature = "gxtb")]
            (Engine::Composite(c), true) => c.composite_mut().base_mut().method.reset_wavefunction(),
            #[cfg(feature = "gxtb")]
            (Engine::Composite(c), false) => c.reset_wavefunction(),
            _ => {}
        }
    }
}

/// The g-xTB calculator of a request: plain, or the solvation composite with
/// `solvation_correction`.
#[cfg(feature = "gxtb")]
fn gxtb_engine(
    req: &Request,
    method: Method,
    threads: usize,
    uhf: Option<usize>,
    guess: Option<names::Guess>,
) -> Result<(Engine, axtb_gxtb::Conventions)> {
    let version = method.gxtb_version().expect("a g-xTB method");
    let conventions = req.conventions.as_deref().map(names::gxtb_conventions).transpose()?.unwrap_or_default();
    if req.scf_trace {
        return Err(input(format!("{method} does not support scf_trace (GFN2-xTB only)")));
    }
    if let Some(c) = &req.solvation_correction {
        if req.dispersion == Some(false) {
            return Err(input("the g-xTB solvation composite does not support switching off dispersion"));
        }
        let correction = names::correction(c)?;
        let solvation = solvation(req)?
            .ok_or_else(|| input("solvation_correction needs a solvent: alpb or gbsa (e.g. alpb \"water\")"))?;
        let mut o = axtb_gxtb::CompositeOptions {
            version,
            correction,
            solvation,
            accuracy: req.accuracy.unwrap_or(1.0),
            etemp: req.etemp,
            max_iter: req.max_iter,
            threads,
            conventions,
            reuse_guess: match guess {
                None => None,
                Some(names::Guess::Previous) => Some(true),
                Some(names::Guess::Fresh) => Some(false),
                Some(_) => return Err(input(
                    "the g-xTB solvation composite starts each part from its method's own start: guess \"previous\" \
                         or \"fresh\"",
                )),
            },
            ..Default::default()
        };
        o.correction_conventions = req.correction_conventions.as_deref().map(names::gfn2_conventions).transpose()?;
        o.mixer = req.mixer.as_deref().map(names::mixer).transpose()?;
        let calc = axtb_gxtb::CompositeCalculator::new(&req.numbers, &req.positions, req.charge, uhf, o)?;
        return Ok((Engine::Composite(Box::new(calc)), conventions));
    }
    if req.alpb.is_some() || req.gbsa.is_some() || req.solvation_state.is_some() || req.correction_conventions.is_some()
    {
        return Err(input(gxtb_solvent_hint(method)));
    }
    if req.dispersion == Some(false) {
        return Err(input(format!("{method} does not support switching off dispersion")));
    }
    if req.mixer.is_some() {
        return Err(input(format!("{method} does not support choosing the SCF mixer")));
    }
    let mut o = axtb_gxtb::Options { threads, conventions, ..Default::default() };
    if let Some(a) = req.accuracy {
        o.scf.accuracy = a;
    }
    o.scf.etemp = req.etemp;
    o.scf.max_iter = req.max_iter;
    o.scf.restart = match guess {
        None => None,
        Some(names::Guess::Previous) => Some(Restart::Previous),
        Some(names::Guess::Fresh | names::Guess::EeqBc) => Some(Restart::Fresh),
        Some(names::Guess::Eeq | names::Guess::Sad) => {
            return Err(input(format!("{method} starts from the program's EEQ-BC charges: guess \"eeq-bc\"")))
        }
    };
    let calc = axtb_gxtb::Calculator::new(version, &req.numbers, &req.positions, req.charge, uhf, o)?;
    Ok((Engine::Gxtb(Box::new(calc)), conventions))
}

/// A Hessian of an SCF method in the response.
fn hessian_part(h: axtb::Hessian) -> HessianPart {
    HessianPart {
        hessian: Matrix::square(h.hessian),
        hessian_method: names::hessian_method_label(h.method),
        hessian_max_asymmetry: h.max_asymmetry,
        hessian_timings: Named::new(h.timings),
        dipole_derivatives: h.dipole_derivatives,
    }
}

/// Fill the single point of an SCF method into `out`; returns its orbitals (when kept).
fn report_into(out: &mut Response, sp: axtb::method::SinglePoint) -> Option<axtb::Orbitals> {
    let e = sp.electronic.expect("an SCF method's electronic structure");
    let [a, b] = axtb::properties::alpha_beta(&e.levels);
    let nat = sp.charges.len();
    let s = e.scf;
    out.eigensolver = Some(s.eigensolver);
    out.energy = Some(sp.energy);
    out.gradient = sp.gradient;
    out.dipole = Some(sp.dipole);
    out.terms = Some(Named::new(sp.terms.iter().map(|t| (t.name, t.value)).collect()));
    out.timings = Some(Named::new(sp.timings));
    out.scf = Some(Scf {
        iterations: s.iterations,
        guess: s.start,
        mixer_used: s.mixer_used,
        scf_attempts: (!s.attempts.is_empty()).then(|| {
            s.attempts
                .iter()
                .map(|a| Attempt { mixer: a.mixer, iterations: a.iterations, converged: a.converged })
                .collect()
        }),
        homo_lumo_gap: e.homo_lumo_gap,
        quadrupole: e.quadrupole,
        shell_charges: e.shell_charges,
        atomic_dipoles: e.atomic_dipoles,
        atomic_quadrupoles: e.atomic_quadrupoles,
        orbital_energies_alpha: a.energies,
        orbital_energies_beta: b.energies,
        occupations_alpha: a.occupations,
        occupations_beta: b.occupations,
        bond_orders: e.bond_orders.map(|bo| Matrix { rows: nat, cols: nat, data: bo }),
    });
    out.charges = Some(sp.charges);
    e.orbitals
}

/// Refuse a non-finite energy, gradient or Hessian (JSON cannot hold them; they would be
/// `null`).
fn check_finite(out: &Response) -> Result<()> {
    let bad = |what: &str| Err(Error::Numerical(format!("the calculation gave a non-finite {what}")));
    if out.energy.is_some_and(|e| !e.is_finite()) {
        return bad("energy");
    }
    if out.gradient.as_ref().is_some_and(|g| g.iter().flatten().any(|x| !x.is_finite())) {
        return bad("gradient");
    }
    if out.hessian.as_ref().is_some_and(|h| h.hessian.data.iter().any(|x| !x.is_finite())) {
        return bad("Hessian");
    }
    Ok(())
}

/// A composite's report as the response's `composite` group (`hessian`: the parts of its
/// Hessian).
fn composite_response(r: axtb::method::CompositeReport, hessian: Option<&[axtb::method::PartReport]>) -> Composite {
    let part = |p: &axtb::method::PartReport| {
        let scf = p.scf.as_ref();
        let attempts = scf.map_or(&[][..], |s| &s.attempts[..]);
        let part = Part {
            method: p.label.clone(),
            conventions: p.conventions,
            energy: p.energy,
            iterations: scf.map_or(0, |s| s.iterations),
            mixer_used: scf.and_then(|s| s.mixer_used),
            scf_attempts: attempts
                .iter()
                .map(|a| Attempt { mixer: a.mixer, iterations: a.iterations, converged: a.converged })
                .collect(),
            etemp: p.etemp,
            max_iter: p.max_iter,
            retry: p.retry.clone(),
            seconds: p.seconds,
            hessian_method: p.hessian_method.map(names::hessian_method_label),
        };
        (p.key, part)
    };
    Composite {
        composite: Details(r.description),
        components: Named(r.components),
        parts: Parts(r.parts.iter().map(part).collect()),
        hessian_parts: hessian.map(|h| Parts(h.iter().map(part).collect())),
    }
}
