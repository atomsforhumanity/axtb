//! axtb: extended tight-binding quantum chemistry. This crate holds GFN2-xTB and what the
//! methods of the workspace share (basis, integrals, Molden files, properties, vibrational
//! analysis; the molecule and the eigensolver from `axtb-base`, re-exported); g-xTB is
//! `axtb-gxtb`, GFN-FF `axtb-gfnff`.
//!
//! Design rules:
//! - no implicit threading: every parallel region runs on the calculator's own rayon pool of
//!   the thread count in the options (default 1), so many single-threaded calculations can
//!   run side by side without oversubscription. The only process-wide state is the optional
//!   run-time LAPACK library (`AXTB_LAPACK`, read once per process; see
//!   [`linalg::GenEigSolver`]);
//! - atomic units throughout (bohr, Hartree);
//! - every parameter table is generated from the reference sources (`scripts/gen_data.py`).

pub(crate) mod analytic;
pub mod basis;
mod calculator;
pub mod conventions;
pub(crate) mod coulomb;
pub(crate) mod dispersion;
pub(crate) mod eeq;
pub(crate) mod gfn2;
pub(crate) mod hamiltonian;
mod hessian;
pub mod integrals;
pub mod io;
mod mixer;
pub mod molden;
pub(crate) mod ncoord;
pub mod param;
pub mod properties;
pub(crate) mod repulsion;
pub(crate) mod response;
mod solvation;
#[cfg(test)]
mod tests_response;
pub mod threads;
pub mod vibrations;
mod xtb;

pub use calculator::{Calculator, Hessian};
pub use conventions::{ConventionParams, Conventions};
pub use hessian::{HessianMethod, HessianOptions};
pub use mixer::{MixerKind, ScfAttempt};
pub use molden::{MoldenFunctions, Orbitals};
pub use properties::{GapRule, OrbitalLevels, QuadrupoleForm};
pub use solvation::{ReferenceState, SolvationModel, SolvationSpec};
pub use xtb::{EnergyTerms, InitialGuess, ScfGuess, ScfOptions, SinglePoint};

pub use axtb_base::linalg;
pub use axtb_base::units;
pub use axtb_base::{Error, Molecule, Result, MIN_ATOM_DISTANCE};

/// The molecule and the 3-vector helpers (from `axtb-base`).
pub mod molecule {
    pub use axtb_base::math::{dot3, norm2_sq, sub3};
    pub use axtb_base::{Molecule, MIN_ATOM_DISTANCE};
}

/// Available methods.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    Gfn2,
}

impl std::str::FromStr for Method {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().replace(['-', '_'], "").as_str() {
            "gfn2" | "gfn2xtb" => Ok(Method::Gfn2),
            other => Err(Error::Input(format!("unknown method {other:?}"))),
        }
    }
}

/// Options of a calculation.
#[derive(Clone, Debug)]
pub struct Options {
    pub scf: ScfOptions,
    /// Include dispersion (always on for production use; off only for validation).
    pub dispersion: bool,
    /// Numerical conventions (axtb's corrected default, or exact xtb/tblite reproduction).
    pub conventions: Conventions,
    /// The settings of the conventions; `None` (the default) takes `conventions.params()`. A
    /// modified table runs a variant of the conventions, e.g. one setting of xtb 6.7.1 at a
    /// time in xtb 6.8.0's (`examples/conventions_ablation`); `conventions` still chooses
    /// xtb's bond-order cutoff.
    pub params: Option<ConventionParams>,
    /// Implicit solvation (ALPB or GBSA), off by default.
    pub solvation: Option<SolvationSpec>,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            scf: ScfOptions::default(),
            dispersion: true,
            conventions: Conventions::Axtb,
            params: None,
            solvation: None,
        }
    }
}

/// Build the Hamiltonian of `method` for `mol` and its charge-dependent extra terms
/// (dispersion, solvation).
pub(crate) fn build_method(
    method: Method,
    mol: &Molecule,
    opts: &Options,
) -> Result<(xtb::XtbModel, Vec<Box<dyn xtb::ChargeDependentTerm>>)> {
    match method {
        Method::Gfn2 => {
            let params = opts.params.unwrap_or_else(|| opts.conventions.params());
            let model = gfn2::build(mol, opts.scf.accuracy, &params)?;
            let mut extra: Vec<Box<dyn xtb::ChargeDependentTerm>> = Vec::new();
            if opts.dispersion {
                extra.push(Box::new(dispersion::D4Term::gfn2(mol, &params)?));
            }
            if let Some(spec) = &opts.solvation {
                extra.push(Box::new(solvation::SolvationTerm::new(mol, spec, params.solvation)?));
            }
            Ok((model, extra))
        }
    }
}

/// Energy (and optionally gradient) of `mol` with `method`, without keeping state.
pub fn singlepoint(method: Method, mol: &Molecule, opts: &Options, gradient: bool) -> Result<SinglePoint> {
    let mut calc = Calculator::new(method, &mol.numbers, &mol.positions, mol.charge, mol.uhf, opts.clone())?;
    calc.singlepoint(&mol.positions, gradient)
}

/// GFN2-xTB's core Hamiltonian H0 and overlap S of `mol` (dense, basis functions x basis
/// functions) in `conventions` at accuracy 1, as the SCF builds them. For the eigensolver
/// benchmark's test matrices (`bench/linalg/dump`). Only with the off-by-default
/// `bench-internals` feature, and not part of the supported API: it may change in any release.
#[cfg(feature = "bench-internals")]
pub fn gfn2_core_matrices(mol: &Molecule, conventions: Conventions) -> Result<(faer::Mat<f64>, faer::Mat<f64>)> {
    let model = gfn2::build(mol, 1.0, &conventions.params())?;
    let cn = model.ncoord.compute(mol);
    let (se, _) = model.h0.selfenergy(mol, &model.basis, &cn);
    let ints = model.h0.build(mol, &model.basis, &se, model.pair_cutoff, false);
    Ok((ints.h0, ints.overlap))
}
