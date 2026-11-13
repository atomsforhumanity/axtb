//! The intermediates of the g-xTB 2.0.1 gradient, as core [`Intermediate`]s of the setup's chain
//! layout ([`axtb::terms::ChainLayout`]): the g-xTB, basis and revD4 coordination numbers, the
//! EEQ-BC charges, and the q-vSZP effective charges of the SCF and H0 bases, which depend on
//! the charges and the basis CN.
//!
//! Registered in the order EEQ-BC, revD4 CN, basis CN, g-xTB CN, effective charges, so
//! [`Chain::finish`] applies the effective charges first (into the charges and the basis CN)
//! and then the g-xTB, basis and revD4 CNs and the charges, the order of the gradient's chain
//! rules before.

use crate::basis::{effective_charge_deriv, effective_charges, qeff_h0_weights, qeff_weights};
use crate::cn::{basis_cn, basis_cn_gradient, gxtb_cn, gxtb_cn_gradient};
use crate::eeqbc::EeqBcSolution;
use crate::eeqbc::{eeqbc, eeqbc_charge_gradient};
use crate::model::Model;
use axtb::terms::{Chain, ChainLayout, Geometry, Intermediate, Jacobians, Slot, Values};
use axtb::{Molecule, Result};
use std::sync::Arc;

/// The slots of the setup's intermediates.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Slots {
    /// Effective charges of the SCF basis and of the H0 basis.
    pub qeff: Slot,
    pub qeff0: Slot,
    /// The g-xTB, basis and revD4 coordination numbers.
    pub cn: Slot,
    pub cnb: Slot,
    pub cnd4: Slot,
    /// The EEQ-BC charges.
    pub eeq: Slot,
}

/// The chain layout of a setup of `mol` in `model`, whose EEQ-BC equations were solved as
/// `solution` with the kernel of the gradient (2.0.1; v1.1's gradient factorizes another
/// matrix, `None`).
pub(crate) fn layout(model: &Model, mol: &Molecule, solution: Option<Arc<EeqBcSolution>>) -> (ChainLayout, Slots) {
    let mut l = ChainLayout::new();
    let eeq = l.register(Arc::new(EeqBcCharges { model: *model, positions: mol.positions.clone(), solution }))[0];
    let cnd4 = l.register(Arc::new(RevD4Cn(*model)))[0];
    let cnb = l.register(Arc::new(BasisCn(*model)))[0];
    let cn = l.register(Arc::new(GxtbCn(*model)))[0];
    let q = l.register(Arc::new(EffectiveCharges { model: *model, eeq, cnb }));
    (l, Slots { qeff: q[0], qeff0: q[1], cn, cnb, cnd4, eeq })
}

/// The g-xTB coordination number ([`gxtb_cn`]).
struct GxtbCn(Model);

impl Intermediate for GxtbCn {
    fn name(&self) -> &'static str {
        "gxtb cn"
    }

    fn compute(&self, mol: &Molecule, _: &Values) -> Result<Vec<Vec<f64>>> {
        Ok(vec![gxtb_cn(&self.0, mol)])
    }

    fn chain(&self, geo: &Geometry<'_>, w: &[&[f64]], chain: &mut Chain) -> Result<()> {
        gxtb_cn_gradient(&self.0, geo.mol, w[0], &mut chain.g);
        Ok(())
    }
}

/// The q-vSZP basis coordination number ([`basis_cn`]).
struct BasisCn(Model);

impl Intermediate for BasisCn {
    fn name(&self) -> &'static str {
        "basis cn"
    }

    fn compute(&self, mol: &Molecule, _: &Values) -> Result<Vec<Vec<f64>>> {
        Ok(vec![basis_cn(&self.0, mol)])
    }

    fn chain(&self, geo: &Geometry<'_>, w: &[&[f64]], chain: &mut Chain) -> Result<()> {
        basis_cn_gradient(&self.0, geo.mol, w[0], &mut chain.g);
        Ok(())
    }
}

/// The revD4 coordination number ([`crate::revd4::coordination_numbers`]).
struct RevD4Cn(Model);

impl Intermediate for RevD4Cn {
    fn name(&self) -> &'static str {
        "revd4 cn"
    }

    fn compute(&self, mol: &Molecule, _: &Values) -> Result<Vec<Vec<f64>>> {
        Ok(vec![crate::revd4::coordination_numbers(&self.0, mol)])
    }

    fn chain(&self, geo: &Geometry<'_>, w: &[&[f64]], chain: &mut Chain) -> Result<()> {
        crate::revd4::add_cn_gradient(&self.0, geo.mol, w[0], &mut chain.g);
        Ok(())
    }
}

/// The EEQ-BC charges, with the factorization of the setup's geometry (`positions`), which the
/// gradient at that geometry reuses; at any other geometry, or without it (v1.1), the equations
/// are solved again.
struct EeqBcCharges {
    model: Model,
    positions: Vec<[f64; 3]>,
    solution: Option<Arc<EeqBcSolution>>,
}

impl Intermediate for EeqBcCharges {
    fn name(&self) -> &'static str {
        "eeq-bc"
    }

    fn compute(&self, mol: &Molecule, _: &Values) -> Result<Vec<Vec<f64>>> {
        Ok(vec![crate::eeqbc::eeqbc_charges(&self.model, mol)?.charges])
    }

    fn chain(&self, geo: &Geometry<'_>, w: &[&[f64]], chain: &mut Chain) -> Result<()> {
        match &self.solution {
            Some(solution) if geo.mol.positions == self.positions => {
                eeqbc(&self.model, geo.mol)?.add_charge_gradient(solution, w[0], &mut chain.g);
                Ok(())
            }
            _ => eeqbc_charge_gradient(&self.model, geo.mol, w[0], &mut chain.g),
        }
    }
}

/// The effective charges q_eff of the SCF basis and q_eff0 of the H0 basis
/// ([`effective_charge`]) of the EEQ-BC charges and the basis CN.
struct EffectiveCharges {
    model: Model,
    eeq: Slot,
    cnb: Slot,
}

impl Intermediate for EffectiveCharges {
    fn name(&self) -> &'static str {
        "qvszp q_eff"
    }

    fn outputs(&self) -> usize {
        2
    }

    fn compute(&self, mol: &Molecule, values: &Values) -> Result<Vec<Vec<f64>>> {
        let (q, cn) = (values.get(self.eeq), values.get(self.cnb));
        let z = &mol.numbers;
        let (qeff, qeff0) =
            (0..mol.nat()).map(|i| effective_charges(&self.model, z[i], q[i], cn[i])).map(|(_, a, b)| (a, b)).unzip();
        Ok(vec![qeff, qeff0])
    }

    /// dE/dq += dE/dq_eff a + dE/dq_eff0 a0, dE/dCN_b += dE/dq_eff b + dE/dq_eff0 b0 (both bases in
    /// one expression).
    fn chain(&self, geo: &Geometry<'_>, w: &[&[f64]], chain: &mut Chain) -> Result<()> {
        let mol = geo.mol;
        let (q, cnb) = (geo.value(self.eeq), geo.value(self.cnb));
        for i in 0..mol.nat() {
            let z = mol.numbers[i];
            let (a, b) = effective_charge_deriv(&self.model, qeff_weights(z), q[i], cnb[i]);
            let (a0, b0) = effective_charge_deriv(&self.model, qeff_h0_weights(&self.model, z), q[i], cnb[i]);
            chain.slot(self.eeq)[i] += w[0][i] * a + w[1][i] * a0;
            chain.slot(self.cnb)[i] += w[0][i] * b + w[1][i] * b0;
        }
        Ok(())
    }

    /// dq_eff/dR = (∂q_eff/∂q) dq/dR + (∂q_eff/∂CN_b) dCN_b/dR, the same for q_eff0.
    fn jacobian(&self, geo: &Geometry<'_>, deps: &Jacobians) -> Result<Vec<Vec<f64>>> {
        let mol = geo.mol;
        let nat = mol.nat();
        let n3 = 3 * nat;
        let (q, cnb) = (geo.value(self.eeq), geo.value(self.cnb));
        let (eeq, jcnb) = (deps.get(self.eeq), deps.get(self.cnb));
        let mut qeff = vec![0.0; nat * n3];
        let mut qeff0 = vec![0.0; nat * n3];
        for a in 0..nat {
            let z = mol.numbers[a];
            let (fq, fc) = effective_charge_deriv(&self.model, qeff_weights(z), q[a], cnb[a]);
            let (gq, gc) = effective_charge_deriv(&self.model, qeff_h0_weights(&self.model, z), q[a], cnb[a]);
            for x in 0..n3 {
                let (e, b) = (eeq[a * n3 + x], jcnb[a * n3 + x]);
                qeff[a * n3 + x] = fq * e + fc * b;
                qeff0[a * n3 + x] = gq * e + gc * b;
            }
        }
        Ok(vec![qeff, qeff0])
    }
}
