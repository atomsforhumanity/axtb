//! D4 dispersion (crate `axtb-d4`) as a self-consistent, charge-dependent SCF term.

use crate::error::{Error, Result};
use crate::molecule::Molecule;
use crate::xtb::ChargeDependentTerm;
use crate::ConventionParams;
use axtb_d4::{ChargeModel, D4Cache, D4Dispersion, D4Model, RationalDamping};
use std::any::Any;

/// Self-consistent D4: two-body energy with the SCF Mulliken charges, three-body (ATM) at
/// zero charges, both with GFN2-xTB damping parameters.
pub(crate) struct D4Term(D4Dispersion);

impl D4Term {
    /// GFN2-xTB D4 with the coordination-number tables and real-space cutoffs of a
    /// convention ([`ConventionParams::d4_rcov`], [`d4_en`](ConventionParams::d4_en),
    /// [`d4_cutoff`](ConventionParams::d4_cutoff)).
    pub(crate) fn gfn2(mol: &Molecule, params: &ConventionParams) -> Result<Self> {
        let model = D4Model::new(&mol.numbers, ChargeModel::Gfn2)
            .map_err(|e| Error::Input(format!("D4: {e}")))?
            .with_cn_parameters(params.d4_rcov, params.d4_en);
        Ok(D4Term(
            D4Dispersion::new(model, RationalDamping::GFN2, params.d4_cutoff).with_pair_sums(params.d4_atm_pair_sums),
        ))
    }
}

fn cache(c: &(dyn Any + Send + Sync)) -> &D4Cache {
    c.downcast_ref::<D4Cache>().expect("D4 cache")
}

impl ChargeDependentTerm for D4Term {
    fn name(&self) -> &'static str {
        "dispersion"
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
        parallel: bool,
    ) {
        self.0.three_body(cache(c), energies, grad, parallel);
    }

    fn energy(&self, _mol: &Molecule, c: &(dyn Any + Send + Sync), qat: &[f64], energies: &mut [f64], parallel: bool) {
        self.0.energy(cache(c), qat, energies, parallel);
    }

    fn potential(&self, _mol: &Molecule, c: &(dyn Any + Send + Sync), qat: &[f64], vat: &mut [f64], parallel: bool) {
        self.0.potential(cache(c), qat, vat, parallel);
    }

    fn gradient(&self, _mol: &Molecule, c: &(dyn Any + Send + Sync), qat: &[f64], grad: &mut [[f64; 3]]) {
        self.0.gradient(cache(c), qat, grad);
    }

    fn kernel(&self, _mol: &Molecule, c: &(dyn Any + Send + Sync), qat: &[f64], k: &mut [f64]) {
        self.0.charge_hessian(cache(c), qat, k);
    }

    fn second_derivatives(
        &self,
        _mol: &Molecule,
        c: &(dyn Any + Send + Sync),
        qat: &[f64],
        dv: &mut [f64],
        hess: Option<&mut [f64]>,
        parallel: bool,
    ) {
        self.0.potential_derivative(cache(c), qat, dv);
        if let Some(h) = hess {
            self.0.hessian(cache(c), qat, h, parallel);
        }
    }
}
