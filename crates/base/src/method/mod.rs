// SPDX-License-Identifier: LGPL-3.0-or-later
//! What a method is: the [`Method`] trait that every calculator of the axtb crates implements
//! (GFN2-xTB, g-xTB, GFN-FF and composites of them), and the types it speaks in.
//!
//! A method is bound to one molecule (atomic numbers, charge, unpaired electrons) when it is
//! built, and then computes single points ([`Method::singlepoint`], what [`Want`] asks for) and
//! Hessians ([`Method::hessian`]) at any positions. An SCF method keeps the converged
//! [`Wavefunction`] of its last SCF and starts the next one from it as its restart policy says
//! ([`ScfSettings::restart`]); a wavefunction can be read, given to another calculator of the same
//! method and molecule, or forgotten. [`MethodInfo`] says what the method is and offers (its
//! names, its electrons, its Hessian methods and their defaults, the conventions of the
//! vibrational analysis of its Hessians, the form of its molecular quadrupole).
//!
//! The results are one [`SinglePoint`] and one [`Hessian`] for every method: the energy terms
//! as the method's reference program reports them ([`Term`]), the atomic charges and dipole, the
//! electronic structure of an SCF method ([`Electronic`]) and what only some methods have
//! ([`Extras`]).
//!
//! [`RetryLadder`] is the one retry policy: a list of tries ([`Rung`]: a fresh start, another
//! electronic temperature or iteration limit) climbed until one succeeds.
//!
//! **Unstable until 1.0**: the trait and its types may change in any minor release.

mod composite;
mod results;
mod retry;
mod settings;
mod wavefunction;

pub use composite::{CompositeReport, Detail, PartReport};
pub use results::{
    Attempt, Electronic, Extras, GfnffReport, GfnffTopology, Hessian, ScfReport, SinglePoint, Term, TermKind,
};
pub use retry::{RetryLadder, RetryOn, Rung};
pub use settings::{Restart, ScfSettings};
pub use wavefunction::{Fingerprint, ScfData, Wavefunction, WavefunctionData};

use crate::{Error, Result};

/// A method bound to one molecule (see the module documentation).
///
/// After an error of [`singlepoint`](Method::singlepoint) or [`hessian`](Method::hessian) the
/// method holds no wavefunction: its next SCF starts from the method's own start unless one is
/// set. (A session that keeps a wavefunction per lineage keeps the last good one itself.)
pub trait Method: Send {
    /// What the method is and offers (fixed when it is built).
    fn info(&self) -> &MethodInfo;

    /// Refuse a calculation the method cannot do, before anything runs: a single point with
    /// `want`, and a Hessian with `hessian` before it. Default: everything is accepted.
    fn check(&self, _want: &Want, _hessian: Option<&HessianOptions>) -> Result<()> {
        Ok(())
    }

    /// Energy and what `want` asks for at `positions` (bohr). The SCF starts as the restart
    /// policy says (a wavefunction given with [`set_wavefunction`](Method::set_wavefunction)
    /// always); an SCF from a previous wavefunction that does not converge is run once more from
    /// the method's own start.
    fn singlepoint(&mut self, positions: &[[f64; 3]], want: &Want) -> Result<SinglePoint>;

    /// Hessian at `positions` (bohr); `None` fields of `opts` are the method's defaults
    /// ([`MethodInfo::hessian`]). Afterwards the method holds the wavefunction its next single
    /// point at these positions should start from (each method documents which).
    fn hessian(&mut self, positions: &[[f64; 3]], opts: &HessianOptions) -> Result<Hessian>;

    /// The converged wavefunction of the last SCF; `None` before the first, after an error and
    /// for a method without an SCF.
    fn wavefunction(&self) -> Option<Wavefunction> {
        None
    }

    /// Start the next SCF from `wf`, whatever the restart policy says. Refused (and nothing
    /// changes) unless it belongs to this method and molecule ([`Wavefunction::check`]) and fits
    /// it.
    fn set_wavefunction(&mut self, _wf: &Wavefunction) -> Result<()> {
        Err(Error::Input(format!("{} has no SCF to start from a wavefunction", self.info().name)))
    }

    /// Forget the wavefunction: the next SCF starts from the method's own start.
    fn reset_wavefunction(&mut self) {}

    /// The SCF settings of later calls; `None`: no SCF.
    fn scf_settings(&self) -> Option<&ScfSettings> {
        None
    }

    /// Replace the SCF settings of later calls (checked). What the method built when it was
    /// set up stays (GFN2-xTB's integral screening keeps the accuracy it was built with).
    fn set_scf_settings(&mut self, _settings: ScfSettings) -> Result<()> {
        Err(Error::Input(format!("{} has no SCF settings", self.info().name)))
    }
}

/// What a method is and offers.
#[derive(Clone, Debug, PartialEq)]
pub struct MethodInfo {
    /// Canonical method name: `gfn2`, `gxtb-v2.0.1`, `gxtb-v1.1`, `gfnff` (a composite: its
    /// base's).
    pub name: &'static str,
    /// Canonical name of the conventions: `axtb`, `xtb`, `xtb-6.7.1`, `tblite`, `gxtb`.
    pub conventions: &'static str,
    /// The electrons the method models.
    pub electrons: Electrons,
    /// Its Hessian methods and their defaults.
    pub hessian: HessianDefaults,
    /// The vibrational analysis of its Hessians.
    pub analysis: AnalysisDefaults,
    /// The form of [`Electronic::quadrupole`]; `None`: no molecular quadrupole.
    pub quadrupole: Option<QuadrupoleForm>,
}

/// The electrons a method models.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Electrons {
    /// None (a force field): no orbitals, no SCF.
    None,
    /// One set of orbitals with alpha and beta occupations (GFN2-xTB).
    OneSet,
    /// Orbitals per spin channel (g-xTB: one channel for a closed shell, two for an open shell).
    PerSpin,
}

/// The Hessian methods a method offers, each with its defaults, and the method used when none
/// is given.
#[derive(Clone, Debug, PartialEq)]
pub struct HessianDefaults {
    pub default: HessianMethod,
    pub offered: Vec<HessianDefault>,
}

impl HessianDefaults {
    /// The defaults of `method`, if offered.
    pub fn get(&self, method: HessianMethod) -> Option<&HessianDefault> {
        self.offered.iter().find(|d| d.method == method)
    }

    /// The method of `opts`, or the default.
    pub fn resolve(&self, opts: &HessianOptions) -> HessianMethod {
        opts.method.unwrap_or(self.default)
    }
}

/// The defaults of one Hessian method: the finite-difference step (bohr; `None`: no step) and
/// the SCF accuracy of its SCFs (`None`: no SCF).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HessianDefault {
    pub method: HessianMethod,
    pub step: Option<f64>,
    pub accuracy: Option<f64>,
}

/// The vibrational analysis of a method's Hessians: the conventions of the calculation's
/// conventions, and a frequency scale of the thermochemistry in the xtb programs' conventions
/// (GFN-FF: 1.03, as xtb scales GFN-FF frequencies).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AnalysisDefaults {
    pub conventions: AnalysisConventions,
    pub xtb_frequency_scale: Option<f64>,
}

impl AnalysisDefaults {
    /// The default frequency scale in the analysis conventions `c`; `None`: the analysis'
    /// own default.
    pub fn frequency_scale(&self, c: AnalysisConventions) -> Option<f64> {
        self.xtb_frequency_scale.filter(|_| c.is_xtb())
    }
}

/// Constants, masses and projection of a vibrational analysis (`axtb::vibrations`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum AnalysisConventions {
    /// CODATA 2022 constants, standard atomic masses, mass-weighted projection.
    #[default]
    Axtb,
    /// The xtb 6.8.0 program.
    Xtb,
    /// The xtb 6.7.1 program (also what the g-xTB 2.0.1 program, a 6.7.1-based xtb build,
    /// does).
    Xtb671,
}

impl AnalysisConventions {
    /// `axtb`, `xtb` or `xtb-6.7.1`, as parsed.
    pub fn label(self) -> &'static str {
        match self {
            AnalysisConventions::Axtb => "axtb",
            AnalysisConventions::Xtb => "xtb",
            AnalysisConventions::Xtb671 => "xtb-6.7.1",
        }
    }

    /// Whether these are one of the xtb programs' conventions.
    pub fn is_xtb(self) -> bool {
        self != AnalysisConventions::Axtb
    }
}

impl std::fmt::Display for AnalysisConventions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

impl std::str::FromStr for AnalysisConventions {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "axtb" | "default" => Ok(AnalysisConventions::Axtb),
            "xtb" | "xtb-6.8.0" => Ok(AnalysisConventions::Xtb),
            // the g-xTB 2.0.1 program is a 6.7.1-based xtb build
            "xtb-6.7.1" | "gxtb" => Ok(AnalysisConventions::Xtb671),
            other => Err(Error::Input(format!(
                "unknown vibrational-analysis conventions {other:?} (axtb, xtb or xtb-6.7.1)"
            ))),
        }
    }
}

/// How the molecular quadrupole is assembled from the atomic moments (`axtb::properties`).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum QuadrupoleForm {
    /// Θ = Σ_A [θ_A + 3/2 (q r r + r d + d r) − 1/2 (q r² + 2 r·d) 1]: the traceless
    /// quadrupole in the convention of the atomic quadrupoles (xtb's `molmom`).
    #[default]
    Traceless,
    /// tblite's `get_molecular_quadrupole_moment` (the g-xTB 2.0.1 program's `--quadrupole`):
    /// the off-diagonal elements of the charge and dipole parts doubled.
    Tblite,
}

/// What a single point computes besides the energy, charges and dipole (and an SCF method's
/// electronic structure).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Want {
    /// The analytic gradient.
    pub gradient: bool,
    /// The converged orbitals with their basis ([`Electronic::orbitals`]).
    pub orbitals: bool,
    /// The Mayer–Wiberg bond orders ([`Electronic::bond_orders`]).
    pub bond_orders: bool,
}

impl Want {
    /// The energy alone.
    pub const ENERGY: Want = Want { gradient: false, orbitals: false, bond_orders: false };
    /// The energy and the gradient.
    pub const GRADIENT: Want = Want { gradient: true, orbitals: false, bond_orders: false };

    /// The energy, and the gradient when `gradient`.
    pub fn gradient(gradient: bool) -> Want {
        Want { gradient, ..Want::ENERGY }
    }
}

/// How a Hessian is computed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum HessianMethod {
    /// Central differences of analytic gradients, one SCF per displacement.
    Numerical,
    /// Central differences of the gradient functional with the wavefunction extrapolated
    /// by the coupled-perturbed response (no SCF per displacement).
    Response,
    /// Analytic second derivatives (with the coupled-perturbed response of an SCF method).
    #[default]
    Analytic,
}

impl HessianMethod {
    /// `numerical`, `response` or `analytic`.
    pub fn label(self) -> &'static str {
        match self {
            HessianMethod::Numerical => "numerical",
            HessianMethod::Response => "response",
            HessianMethod::Analytic => "analytic",
        }
    }
}

impl std::str::FromStr for HessianMethod {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "numerical" | "numeric" | "fd" => Ok(HessianMethod::Numerical),
            "response" => Ok(HessianMethod::Response),
            "analytic" | "analytical" => Ok(HessianMethod::Analytic),
            other => Err(Error::Input(format!("unknown Hessian method {other:?} (numerical, response, analytic)"))),
        }
    }
}

/// Settings of a Hessian; `None`: the method's default ([`MethodInfo::hessian`]).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct HessianOptions {
    pub method: Option<HessianMethod>,
    /// Finite-difference step (bohr) of the numerical and response Hessians.
    pub step: Option<f64>,
    /// SCF accuracy: of every displacement (numerical), of the reference wavefunction
    /// (response, analytic). Never looser than the method's own.
    pub accuracy: Option<f64>,
}
