//! EEQ-BC charges of g-xTB 2.0.1 (multicharge `eeqbc2025` as built into the 2.0.1 program):
//! the model of [`crate::eeqbc`] with refitted parameters ([`crate::data_v2`] `EEQBC_*`)
//! and its 2.0.1 form ([`form`]; spec §4.1):
//!
//! - no local-charge hardness term: A_AA = (eta_A + sqrt(2/pi)/w_A) C_AA + 1;
//! - w_A = rad_A (1 - 0.3 CN_A / CNavg_A^0.75);
//! - c_AB = sqrt(cap_A cap_B) 1/2 [1 + erf(-0.75 (R - Rv_AB)/Rv_AB)] with
//!   Rv_AB = R^D3_AB (bohr) / (2 aatoau) (s_A + s_B), s = `eeqbc_rvdw_scale`.
//!
//! CN_A = sum_B 1/2 [1 + erf(-2 (R - Rc)/Rc^0.75)], Rc = r_A + r_B (`eeqbc_cov_radii`), and
//! the local charges q^loc (Pauling EN / 3.98, multicharge's actinide values) use the 25-bohr
//! CN cutoff of the program ([`super::Model::cut`]).

use super::Model;
use crate::data_v2::{
    EEQBC_AVG_CN, EEQBC_CAP, EEQBC_CHI, EEQBC_COV_RADII, EEQBC_ETA, EEQBC_KCNCHI, EEQBC_KQCHI, EEQBC_RAD,
    EEQBC_RVDW_SCALE,
};
use crate::eeqbc::{EeqBc, EeqBcAtom, EeqBcCharges, EeqBcForm, Kernel, RcRule, RvdwRule};
use axtb::{Molecule, Result};

/// sqrt(pi) as the 2.0.1 program writes it.
const SQRTPI: f64 = 1.772_453_850_905_516;

/// The 2.0.1 form of EEQ-BC in the conventions of `model` (its Ångström-to-bohr factor and
/// its CN cutoff).
pub fn form(model: &Model) -> EeqBcForm {
    EeqBcForm {
        kbc: 0.75,
        kcnrad: 0.3,
        rc: RcRule::Sum,
        kqeta: false,
        rvdw: RvdwRule::ScaledD3(*model),
        cutoff: Some(*model),
        sqrtpi: SQRTPI,
        gradient_kernel: Kernel::RootGamma,
    }
}

/// The 2.0.1 EEQ-BC model of `mol`.
pub fn eeqbc<'m>(model: &Model, mol: &'m Molecule) -> Result<EeqBc<'m>> {
    super::check_elements(&mol.numbers)?;
    Ok(EeqBc::new(form(model), mol, |z| {
        let k = z - 1;
        EeqBcAtom {
            chi: EEQBC_CHI[k],
            eta: EEQBC_ETA[k],
            rad: EEQBC_RAD[k],
            kcnchi: EEQBC_KCNCHI[k],
            kqchi: EEQBC_KQCHI[k],
            kqeta: 0.0,
            cap: EEQBC_CAP[k],
            rcov: EEQBC_COV_RADII[k],
            cn_avg: EEQBC_AVG_CN[k],
            rvdw_scale: EEQBC_RVDW_SCALE[k],
        }
    }))
}

/// EEQ-BC partial charges of `mol` (positions in bohr; total charge `mol.charge`).
pub fn eeqbc_charges(model: &Model, mol: &Molecule) -> Result<EeqBcCharges> {
    eeqbc(model, mol)?.charges()
}

/// Adds sum_A dE/dq_A dq_A/dR to `grad` for a function E of the EEQ-BC charges (adjoint of
/// the linear charge equations A y = b: dE/dR = w^T (db/dR - dA/dR y), A w = (dE/dq, 0)).
/// For several `dedq` at one geometry, factorize once: [`eeqbc`], [`EeqBc::gradient_system`].
pub fn eeqbc_charge_gradient(model: &Model, mol: &Molecule, dedq: &[f64], grad: &mut [[f64; 3]]) -> Result<()> {
    eeqbc(model, mol)?.gradient_system().add_charge_gradient(dedq, grad);
    Ok(())
}
