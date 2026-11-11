//! Implicit solvation (ALPB, GBSA; crate `axtb-solvation`) as a charge-dependent SCF term:
//! the GFN2-xTB models of the xtb 6.8.0 and 6.7.1 programs, and the corrected model.

use crate::molecule::Molecule;
use crate::xtb::ChargeDependentTerm;
use crate::{Error, Result};
use axtb_solvation::{Conventions as SolvConventions, SolvCache, SolvOptions, Solvation};
pub use axtb_solvation::{Model as SolvationModel, ReferenceState};
use std::any::Any;

/// Which implicit solvation model to add.
#[derive(Clone, Debug, PartialEq)]
pub struct SolvationSpec {
    /// ALPB (`xtb --alpb`) or GBSA (`xtb --gbsa`).
    pub model: SolvationModel,
    /// Solvent name as xtb accepts it (case-insensitive, with xtb's aliases).
    pub solvent: String,
    /// Reference state of the solvation free energy (xtb default `gsolv`).
    pub state: ReferenceState,
}

impl SolvationSpec {
    /// Parse a reference state name (`gsolv`, `bar1M`, `reference`).
    pub fn parse_state(s: &str) -> Result<ReferenceState> {
        match s.to_ascii_lowercase().as_str() {
            "gsolv" => Ok(ReferenceState::Gsolv),
            "bar1m" => Ok(ReferenceState::Bar1M),
            "reference" => Ok(ReferenceState::Reference),
            other => {
                Err(Error::Input(format!("unknown solvation reference state {other:?} (gsolv, bar1M, reference)")))
            }
        }
    }
}

/// The solvation model bound to one molecule.
pub struct SolvationTerm(Solvation);

impl SolvationTerm {
    /// `variant` ([`ConventionParams::solvation`](crate::ConventionParams::solvation)): xtb's
    /// solvation (quirks included) for xtb conventions; the corrected solvation for axtb and
    /// tblite conventions (tblite's own ALPB/GBSA has neither xtb bug and agrees with the
    /// corrected model to ~1e-9 Eh).
    pub fn new(mol: &Molecule, spec: &SolvationSpec, variant: SolvConventions) -> Result<Self> {
        let opts = SolvOptions { conventions: variant, ..SolvOptions::default() };
        Solvation::with_options(&mol.numbers, &spec.solvent, spec.model, spec.state, &opts)
            .map(SolvationTerm)
            .map_err(|e| Error::Input(format!("solvation: {e}")))
    }
}

fn cache(c: &(dyn Any + Send + Sync)) -> &SolvCache {
    c.downcast_ref::<SolvCache>().expect("solvation cache")
}

impl ChargeDependentTerm for SolvationTerm {
    fn name(&self) -> &'static str {
        "solvation"
    }

    fn update(&self, mol: &Molecule) -> Box<dyn Any + Send + Sync> {
        Box::new(self.0.update(&mol.positions))
    }

    fn fixed_energy(
        &self,
        _mol: &Molecule,
        c: &(dyn Any + Send + Sync),
        energies: &mut [f64],
        grad: Option<&mut [[f64; 3]]>,
        _parallel: bool,
    ) {
        self.0.fixed_energy(cache(c), energies, grad);
    }

    fn energy(&self, _mol: &Molecule, c: &(dyn Any + Send + Sync), qat: &[f64], energies: &mut [f64], _parallel: bool) {
        self.0.energy(cache(c), qat, energies);
    }

    fn potential(&self, _mol: &Molecule, c: &(dyn Any + Send + Sync), qat: &[f64], vat: &mut [f64], _parallel: bool) {
        self.0.potential(cache(c), qat, vat);
    }

    fn gradient(&self, _mol: &Molecule, c: &(dyn Any + Send + Sync), qat: &[f64], grad: &mut [[f64; 3]]) {
        self.0.gradient(cache(c), qat, grad);
    }

    /// Analytic for the corrected Born integrator (axtb and tblite conventions); xtb's
    /// stale-`r1` integrator (xtb conventions) has no consistent derivatives and uses central
    /// differences.
    fn second_derivatives(
        &self,
        mol: &Molecule,
        c: &(dyn Any + Send + Sync),
        qat: &[f64],
        dv: &mut [f64],
        hess: Option<&mut [f64]>,
        parallel: bool,
    ) {
        let mut hess = hess;
        if !self.0.second_derivatives(cache(c), qat, dv, hess.as_deref_mut()) {
            crate::xtb::fd_second_derivatives(self, mol, qat, dv, hess, parallel);
        }
    }

    /// The charge-dependent energy is `q^T B q / 2`: its kernel is the Born matrix.
    fn kernel(&self, _mol: &Molecule, c: &(dyn Any + Send + Sync), _qat: &[f64], k: &mut [f64]) {
        for (x, b) in k.iter_mut().zip(cache(c).born_matrix()) {
            *x += b;
        }
    }
}
