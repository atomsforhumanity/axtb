//! The response: what the CLI writes as JSON and Python returns as a dict, with the same
//! fields. A field is present when it was computed (`skip` when `None`), except
//! `homo_lumo_gap`, which is `null` without a LUMO.

use serde::ser::{SerializeMap, SerializeSeq};
use serde::{Serialize, Serializer};

/// Program name of every response.
pub const PROGRAM: &str = "axtb";
/// axtb's version, of every response.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// A dense row-major matrix, written as a list of rows.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    /// Row-major entries (`rows * cols`).
    pub data: Vec<f64>,
}

impl Matrix {
    /// The square matrix of row-major `data` (`n * n` entries).
    pub fn square(data: Vec<f64>) -> Self {
        let n = (data.len() as f64).sqrt().round() as usize;
        debug_assert_eq!(n * n, data.len());
        Self { rows: n, cols: n, data }
    }
}

impl Serialize for Matrix {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut seq = s.serialize_seq(Some(self.rows))?;
        if self.cols > 0 {
            for row in self.data.chunks(self.cols) {
                seq.serialize_element(row)?;
            }
        } else {
            for _ in 0..self.rows {
                seq.serialize_element(&[] as &[f64])?;
            }
        }
        seq.end()
    }
}

/// Named values in their order (energy terms, timings), written as an object. A name that
/// occurs twice is written once, with the values summed.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Named(pub Vec<(&'static str, f64)>);

impl Named {
    /// The values with repeated names summed, in the order of first occurrence.
    pub fn new(values: Vec<(&'static str, f64)>) -> Self {
        let mut out: Vec<(&'static str, f64)> = Vec::with_capacity(values.len());
        for (k, v) in values {
            match out.iter_mut().find(|(n, _)| *n == k) {
                Some(slot) => slot.1 += v,
                None => out.push((k, v)),
            }
        }
        Self(out)
    }

    /// The value of a name.
    pub fn get(&self, name: &str) -> Option<f64> {
        self.0.iter().find(|(n, _)| *n == name).map(|(_, v)| *v)
    }
}

impl Serialize for Named {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut m = s.serialize_map(Some(self.0.len()))?;
        for (k, v) in &self.0 {
            m.serialize_entry(k, v)?;
        }
        m.end()
    }
}

/// One SCF attempt (GFN2-xTB: mixer, iterations, converged).
#[derive(Serialize, Clone, Debug, PartialEq)]
pub struct Attempt {
    pub mixer: &'static str,
    pub iterations: usize,
    pub converged: bool,
}

/// What every SCF method reports besides energy and charges.
#[derive(Serialize, Clone, Debug, Default, PartialEq)]
pub struct Scf {
    /// Iterations of the attempt that converged.
    pub iterations: usize,
    /// Where the SCF started: `eeq`, `sad` (GFN2-xTB), `eeq-bc` (g-xTB), `previous`.
    pub guess: &'static str,
    /// GFN2-xTB: the mixer that produced the result (`broyden`, `diis`, `diis-fallback`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mixer_used: Option<&'static str>,
    /// GFN2-xTB: every SCF attempt.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scf_attempts: Option<Vec<Attempt>>,
    /// HOMO-LUMO gap (Eh); `null` without a LUMO.
    pub homo_lumo_gap: Option<f64>,
    /// Molecular quadrupole about the origin (xx, xy, yy, xz, yz, zz; e bohr², traceless;
    /// tblite's form in the `tblite` and g-xTB 2.0.1 `gxtb` conventions).
    pub quadrupole: [f64; 6],
    /// Converged Mulliken shell charges, atomic dipoles (N x 3) and quadrupoles (N x 6), e.g.
    /// to restart another program.
    pub shell_charges: Vec<f64>,
    pub atomic_dipoles: Vec<[f64; 3]>,
    pub atomic_quadrupoles: Vec<[f64; 6]>,
    /// Orbital energies (Eh) and occupations of the alpha and beta electrons (a closed shell:
    /// its orbitals for both spins with half the occupations).
    pub orbital_energies_alpha: Vec<f64>,
    pub orbital_energies_beta: Vec<f64>,
    pub occupations_alpha: Vec<f64>,
    pub occupations_beta: Vec<f64>,
    /// Mayer-Wiberg bond orders (N x N), when asked for.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bond_orders: Option<Matrix>,
}

/// The GFN-FF topology (0-based atom indices).
#[derive(Serialize, Clone, Debug, Default, PartialEq)]
pub struct Topology {
    pub bonds: Vec<[usize; 2]>,
    pub pi_bond_orders: Vec<f64>,
    pub bond_types: Vec<i32>,
    pub hybridization: Vec<i32>,
    pub pi_atoms: Vec<bool>,
    pub fragments: Vec<usize>,
    pub fragment_charges: Vec<f64>,
    pub topology_charges: Vec<f64>,
    pub n_angles: usize,
    pub n_torsions: usize,
}

/// What GFN-FF reports besides energy and charges.
#[derive(Serialize, Clone, Debug, Default, PartialEq)]
pub struct Gfnff {
    /// Hydrogen and halogen bonds of the evaluation.
    pub n_hbonds: usize,
    pub n_xbonds: usize,
    pub topology: Topology,
}

/// A Hessian and how it was computed.
#[derive(Serialize, Clone, Debug, Default, PartialEq)]
pub struct HessianPart {
    /// 3N x 3N, Eh/bohr², symmetrized.
    pub hessian: Matrix,
    /// `analytic`, `response` or `numerical`.
    pub hessian_method: &'static str,
    /// Largest |H_ij - H_ji| before the symmetrization.
    pub hessian_max_asymmetry: f64,
    /// Wall time per phase (s).
    pub hessian_timings: Named,
    /// ∂μ/∂R_x (3N x 3, e), for IR intensities.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dipole_derivatives: Option<Vec<[f64; 3]>>,
}

/// Enthalpy, entropy and heat capacity of one degree of freedom.
#[derive(Serialize, Clone, Copy, Debug, Default, PartialEq)]
pub struct Contribution {
    pub enthalpy: f64,
    pub entropy: f64,
    pub heat_capacity: f64,
}

impl From<&axtb::vibrations::Contribution> for Contribution {
    fn from(c: &axtb::vibrations::Contribution) -> Self {
        Self { enthalpy: c.enthalpy, entropy: c.entropy, heat_capacity: c.heat_capacity }
    }
}

/// The thermochemistry at one temperature (Eh, Eh/K).
#[derive(Serialize, Clone, Debug, Default, PartialEq)]
pub struct ThermoPoint {
    pub temperature: f64,
    /// H(T) including the ZPVE.
    pub enthalpy: f64,
    pub thermal_enthalpy: f64,
    pub entropy: f64,
    pub ts: f64,
    /// xtb's G(RRHO) contribution.
    pub g_rrho: f64,
    pub heat_capacity: f64,
    pub translation: Contribution,
    pub rotation: Contribution,
    pub vibration: Contribution,
    /// energy + g_rrho, with an energy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub free_energy: Option<f64>,
    /// energy + enthalpy, with an energy.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_enthalpy: Option<f64>,
}

impl ThermoPoint {
    fn new(p: &axtb::vibrations::ThermoPoint, energy: Option<f64>) -> Self {
        Self {
            temperature: p.temperature,
            enthalpy: p.enthalpy,
            thermal_enthalpy: p.thermal_enthalpy,
            entropy: p.entropy,
            ts: p.ts,
            g_rrho: p.g_rrho,
            heat_capacity: p.heat_capacity,
            translation: (&p.translation).into(),
            rotation: (&p.rotation).into(),
            vibration: (&p.vibration).into(),
            free_energy: energy.map(|e| e + p.g_rrho),
            total_enthalpy: energy.map(|e| e + p.enthalpy),
        }
    }
}

/// Quasi-RRHO thermochemistry: the last temperature's values, the settings and every
/// temperature in `points`.
#[derive(Serialize, Clone, Debug, Default, PartialEq)]
pub struct Thermo {
    #[serde(flatten)]
    pub last: ThermoPoint,
    pub point_group: String,
    pub symmetry_number: u32,
    pub linear: bool,
    /// cm⁻¹.
    pub rotational_constants: [f64; 3],
    /// u.
    pub mass: f64,
    pub n_modes: usize,
    pub n_inverted: usize,
    pub n_dropped: usize,
    pub zpve: f64,
    pub pressure: f64,
    pub rotor_cutoff: f64,
    pub imaginary_cutoff: f64,
    pub frequency_scale: f64,
    pub points: Vec<ThermoPoint>,
}

/// The vibrational analysis of a Hessian.
#[derive(Serialize, Clone, Debug, Default, PartialEq)]
pub struct Vibrations {
    /// `axtb`, `xtb` or `xtb-6.7.1`.
    pub vibrational_conventions: &'static str,
    /// cm⁻¹, ascending, imaginary negative.
    pub frequencies: Vec<f64>,
    /// Per mode and atom: Cartesian displacements of unit length.
    pub normal_modes: Vec<Vec<[f64; 3]>>,
    /// u.
    pub reduced_masses: Vec<f64>,
    /// mdyn/Å.
    pub force_constants: Vec<f64>,
    /// km/mol, with dipole derivatives.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ir_intensities: Option<Vec<f64>>,
    pub n_imaginary: usize,
    pub linear: bool,
    /// The masses of the analysis (u).
    pub masses: Vec<f64>,
    /// Without `thermo: false`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub thermo: Option<Thermo>,
}

impl Vibrations {
    /// The analysis `v`; `energy` adds total enthalpies and free energies.
    pub fn new(v: axtb::vibrations::VibAnalysis, energy: Option<f64>) -> Self {
        let thermo = v.thermo.as_ref().map(|t| Thermo {
            last: ThermoPoint::new(t.last(), energy),
            point_group: t.point_group.clone(),
            symmetry_number: t.symmetry_number,
            linear: t.linear,
            rotational_constants: t.rotational_constants,
            mass: t.mass,
            n_modes: t.n_modes,
            n_inverted: t.n_inverted,
            n_dropped: t.n_dropped,
            zpve: t.zpve,
            pressure: t.pressure,
            rotor_cutoff: t.rotor_cutoff,
            imaginary_cutoff: t.imaginary_cutoff,
            frequency_scale: t.frequency_scale,
            points: t.points.iter().map(|p| ThermoPoint::new(p, energy)).collect(),
        });
        Self {
            vibrational_conventions: v.conventions.label(),
            frequencies: v.frequencies,
            normal_modes: v.normal_modes,
            reduced_masses: v.reduced_masses,
            force_constants: v.force_constants,
            ir_intensities: v.ir_intensities,
            n_imaginary: v.n_imaginary,
            linear: v.linear,
            masses: v.masses,
            thermo,
        }
    }
}

/// Named values of a composite's description (its definition and settings), written as an
/// object in their order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Details(pub Vec<(&'static str, axtb::method::Detail)>);

impl Serialize for Details {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut m = s.serialize_map(Some(self.0.len()))?;
        for (k, v) in &self.0 {
            m.serialize_entry(k, &DetailRef(v))?;
        }
        m.end()
    }
}

/// One value of [`Details`] as JSON: text and names as strings, numbers, flags, groups as
/// objects.
struct DetailRef<'a>(&'a axtb::method::Detail);

impl Serialize for DetailRef<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use axtb::method::Detail;
        match self.0 {
            Detail::Text(t) => s.serialize_str(t),
            Detail::Name(n) => s.serialize_str(n),
            Detail::Number(x) => s.serialize_f64(*x),
            Detail::Flag(b) => s.serialize_bool(*b),
            Detail::Group(g) => {
                let mut m = s.serialize_map(Some(g.len()))?;
                for (k, v) in g {
                    m.serialize_entry(k, &DetailRef(v))?;
                }
                m.end()
            }
        }
    }
}

/// How one part of the composite converged.
#[derive(Serialize, Clone, Debug, Default, PartialEq)]
pub struct Part {
    pub method: String,
    pub conventions: &'static str,
    pub energy: f64,
    pub iterations: usize,
    pub mixer_used: Option<&'static str>,
    pub scf_attempts: Vec<Attempt>,
    pub etemp: f64,
    pub max_iter: usize,
    /// The retry that converged it, if any.
    pub retry: Option<String>,
    pub seconds: f64,
    /// The part's Hessian method (in `hessian_parts`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hessian_method: Option<&'static str>,
}

/// A composite's parts by key, written as an object in their order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Parts(pub Vec<(&'static str, Part)>);

impl Parts {
    /// The part `key`.
    pub fn get(&self, key: &str) -> Option<&Part> {
        self.0.iter().find(|(k, _)| *k == key).map(|(_, p)| p)
    }
}

impl Serialize for Parts {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let mut m = s.serialize_map(Some(self.0.len()))?;
        for (k, v) in &self.0 {
            m.serialize_entry(k, v)?;
        }
        m.end()
    }
}

/// What a composite adds (the g-xTB solvation composite: its definition and settings, the
/// energies of g-xTB, the GFN2 halves and the correction, how each part converged).
#[derive(Serialize, Clone, Debug, Default, PartialEq)]
pub struct Composite {
    pub composite: Details,
    /// The energies of the single point's parts and corrections (Eh).
    pub components: Named,
    pub parts: Parts,
    /// The parts of the Hessian, with their Hessian methods.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hessian_parts: Option<Parts>,
}

/// The response to a [`crate::Request`], the same for every front end.
///
/// Every response has `program`, `version`, `method` (canonical; `null` for an analysis of a
/// given Hessian) and `conventions` (the method's, canonical; the analysis conventions for an
/// analysis); SCF methods add `eigensolver` (`SinglePoint::eigensolver`: `faer`, or the run-time
/// LAPACK library). A calculation has `energy`, `charges`, `dipole`, `terms`, `timings` and
/// `seconds`, and `gradient` when asked for or with a Hessian; the groups below are present
/// when they apply.
#[derive(Serialize, Clone, Debug, Default, PartialEq)]
pub struct Response {
    pub program: &'static str,
    pub version: &'static str,
    pub method: Option<&'static str>,
    pub conventions: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eigensolver: Option<String>,
    /// Eh (an analysis: the given energy).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub energy: Option<f64>,
    /// Eh/bohr, N x 3.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gradient: Option<Vec<[f64; 3]>>,
    /// Atomic charges (e): Mulliken (SCF methods) or EEQ (GFN-FF).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub charges: Option<Vec<f64>>,
    /// Dipole moment about the origin (e bohr).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub dipole: Option<[f64; 3]>,
    /// Energy terms (Eh), named as the method's program reports them.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub terms: Option<Named>,
    #[serde(flatten)]
    pub scf: Option<Scf>,
    #[serde(flatten)]
    pub gfnff: Option<Gfnff>,
    /// Text of the Molden file (request `molden`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub molden: Option<String>,
    /// Where the Molden file was written (request `molden_file`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub molden_file: Option<String>,
    #[serde(flatten)]
    pub hessian: Option<HessianPart>,
    #[serde(flatten)]
    pub vibrations: Option<Vibrations>,
    #[serde(flatten)]
    pub composite: Option<Composite>,
    /// Wall time per phase of the single point (s).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub timings: Option<Named>,
    /// Wall time of the request (s; the CLI and `axtb.singlepoint`: with the setup).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seconds: Option<f64>,
}
