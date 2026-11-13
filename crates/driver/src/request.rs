// SPDX-License-Identifier: LGPL-3.0-or-later
//! The request: what the CLI reads from argv or JSON and Python from keyword arguments.

use serde::Deserialize;

/// One temperature or a list (K).
#[derive(Deserialize, Clone, Debug, PartialEq)]
#[serde(untagged)]
pub enum Temperatures {
    One(f64),
    Many(Vec<f64>),
}

impl Temperatures {
    /// The temperatures as a list.
    pub fn to_vec(&self) -> Vec<f64> {
        match self {
            Temperatures::One(t) => vec![*t],
            Temperatures::Many(v) => v.clone(),
        }
    }
}

/// A calculation: the molecule, the method and its settings, and what to compute. Unknown
/// fields are refused, and so are settings that do not apply to the method (see
/// [`crate::run`]); `None` (or a missing field) means the method's default. Units: bohr,
/// Hartree, kelvin, pascal.
///
/// With `analyze_hessian` the request is a vibrational analysis of a given Hessian instead
/// (`numbers`, `positions`, the analysis settings, `dipole_derivatives` and `energy`); the
/// calculation's fields are refused then.
#[derive(Deserialize, Clone, Debug, Default)]
#[serde(deny_unknown_fields)]
pub struct Request {
    /// Atomic numbers.
    pub numbers: Vec<usize>,
    /// Positions (bohr), one row per atom.
    pub positions: Vec<[f64; 3]>,
    /// Total charge (default 0).
    #[serde(default)]
    pub charge: f64,
    /// Unpaired electrons (default: 0 or 1 by the electron count); GFN-FF has no electrons
    /// and ignores it.
    pub uhf: Option<i64>,

    /// `gfn2` (default), `gfnff`, `gxtb-v1.1` or `gxtb-v2.0.1` (see [`crate::Method::parse`]).
    pub method: Option<String>,
    /// The method's conventions (default `axtb`): GFN2-xTB `axtb`, `xtb`, `xtb-6.7.1`,
    /// `tblite`; g-xTB `axtb`, `gxtb`; GFN-FF `axtb`, `xtb`, `xtb-6.7.1`.
    pub conventions: Option<String>,
    /// SCF accuracy multiplier (SCF methods; default 1).
    pub accuracy: Option<f64>,
    /// Electronic temperature (K; SCF methods; default GFN2-xTB 300, g-xTB 0).
    pub etemp: Option<f64>,
    /// SCF iteration limit (SCF methods; default: GFN2-xTB and g-xTB 2.0.1 250, g-xTB v1.1 1000).
    pub max_iter: Option<usize>,
    /// Threads (default 1; at least 1).
    pub threads: Option<usize>,
    /// `false` leaves out the dispersion (GFN2-xTB only, for validation).
    pub dispersion: Option<bool>,
    /// SCF mixer `auto`, `broyden` or `diis` (GFN2-xTB and the composite's GFN2 halves).
    pub mixer: Option<String>,
    /// Print every SCF iteration to stderr (GFN2-xTB).
    #[serde(default)]
    pub scf_trace: bool,
    /// Implicit solvent, ALPB or GBSA (GFN2-xTB, the g-xTB composite).
    pub alpb: Option<String>,
    pub gbsa: Option<String>,
    /// Reference state of the solvation energy: `gsolv` (default), `bar1M`, `reference`.
    pub solvation_state: Option<String>,
    /// g-xTB in solvent: the composite E_gxtb + [E_GFN2+solv - E_GFN2] (`gfn2`).
    pub solvation_correction: Option<String>,
    /// Conventions of the composite's GFN2 halves (default: `axtb` for g-xTB's `axtb`,
    /// `xtb-6.7.1` for `gxtb`).
    pub correction_conventions: Option<String>,
    /// Where the SCFs of a calculator start: `previous` (the previous geometry's state), `fresh`
    /// (the method's own start at every geometry), or a named start at every geometry: `eeq` or
    /// `sad` (GFN2-xTB), `eeq-bc` (g-xTB); default: the conventions' choice.
    pub guess: Option<String>,
    /// GFN-FF: charges of the molecular fragments, in the order of their first atoms (one per
    /// fragment, summing to the total charge).
    pub fragment_charges: Option<Vec<f64>>,
    /// GFN-FF: positions (bohr) to set the topology up from (default: `positions`).
    pub topology_positions: Option<Vec<[f64; 3]>>,

    /// Compute the gradient.
    #[serde(default)]
    pub gradient: bool,
    /// Compute the Hessian (and the gradient), with its vibrational analysis.
    #[serde(default)]
    pub hessian: bool,
    /// `analytic`, `response` or `numerical` (default: the method's).
    pub hessian_method: Option<String>,
    /// Finite-difference step (bohr) of numerical and response Hessians.
    pub hessian_step: Option<f64>,
    /// SCF accuracy of the Hessian's displacements (numerical) or reference (analytic,
    /// response).
    pub hessian_accuracy: Option<f64>,
    /// Put the text of a Molden file of the converged orbitals into the response.
    #[serde(default)]
    pub molden: bool,
    /// Write the Molden file there.
    pub molden_file: Option<String>,
    /// Cartesian instead of spherical functions in the Molden file.
    #[serde(default)]
    pub molden_cartesian: bool,
    /// Add the Mayer-Wiberg bond orders (N x N) to the response (SCF methods).
    #[serde(default)]
    pub bond_orders: bool,

    /// `false` skips the vibrational analysis of the Hessian.
    pub vibrations: Option<bool>,
    /// Conventions of the analysis: `axtb`, `xtb`, `xtb-6.7.1` (default: the calculation's).
    pub analysis_conventions: Option<String>,
    /// Atomic masses (u; default: the analysis conventions' table).
    pub masses: Option<Vec<f64>>,
    /// `false` skips the thermochemistry.
    pub thermo: Option<bool>,
    /// Temperatures of the thermochemistry (K; default 298.15; the results at the last).
    pub temperature: Option<Temperatures>,
    /// Pressure (Pa; default 101325).
    pub pressure: Option<f64>,
    /// Rotor cutoff (cm⁻¹, xtb's `sthr`; default 50).
    pub rotor_cutoff: Option<f64>,
    /// Imaginary-mode cutoff (cm⁻¹ magnitude, xtb's `ithr`; default 20).
    pub imaginary_cutoff: Option<f64>,
    /// Frequency scale of the thermochemistry (default 1; GFN-FF in xtb's conventions 1.03).
    pub frequency_scale: Option<f64>,
    /// Rotational symmetry number (default: from the point group).
    pub symmetry_number: Option<u32>,
    /// Symmetry threshold (bohr, xtb's `desy`; default 0.1).
    pub symmetry_threshold: Option<f64>,

    /// Analysis only: the Hessian (3N x 3N, Eh/bohr²) to analyze.
    pub analyze_hessian: Option<Vec<Vec<f64>>>,
    /// Analysis only: dipole derivatives (3N rows, e) for IR intensities.
    pub dipole_derivatives: Option<Vec<[f64; 3]>>,
    /// Analysis only: the electronic energy (Eh) for total enthalpies and free energies.
    pub energy: Option<f64>,
}

impl Request {
    /// A request for `numbers` at `positions` (bohr) with every setting at its default.
    pub fn new(numbers: Vec<usize>, positions: Vec<[f64; 3]>) -> Self {
        Self { numbers, positions, ..Default::default() }
    }

    /// Whether a Molden file is asked for (text or file).
    pub(crate) fn wants_molden(&self) -> bool {
        self.molden || self.molden_file.is_some()
    }

    /// Whether any vibrational-analysis setting is given.
    pub(crate) fn has_analysis_settings(&self) -> bool {
        self.analysis_conventions.is_some()
            || self.masses.is_some()
            || self.thermo.is_some()
            || self.has_thermo_settings()
    }

    /// Whether any thermochemistry setting is given.
    pub(crate) fn has_thermo_settings(&self) -> bool {
        self.temperature.is_some()
            || self.pressure.is_some()
            || self.rotor_cutoff.is_some()
            || self.imaginary_cutoff.is_some()
            || self.frequency_scale.is_some()
            || self.symmetry_number.is_some()
            || self.symmetry_threshold.is_some()
    }
}
