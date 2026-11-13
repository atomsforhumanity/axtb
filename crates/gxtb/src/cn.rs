// SPDX-License-Identifier: LGPL-3.0-or-later
//! Error-function coordination numbers of g-xTB 2.0.1 (mctc-lib `erf` counting with a pair
//! cutoff): CN_A = sum_B 1/2 [1 + erf(-k (R_AB - Rc_AB) / Rc_AB^e)] * cut(R_AB), with the cutoff
//! factor of [`Model::cut`] (25 bohr in the program).
//!
//! - g-xTB CN ([`gxtb_cn`]): k = 2.068, Rc = rcov_A + rcov_B (`pa_cn_rcov`), e = 1; used raw by
//!   the H0 self-energies and through the smoothed square root by the repulsion, first and
//!   second order, the AES dipole scaling.
//! - basis CN ([`basis_cn`]): k = 3.75, Rc = R_A + R_B (q-vSZP radii), e = 1; q_eff.
//!
//! v1.1 (the 1.1.0 program's `ncoord_erf`, `ncoord_basq`): the g-xTB CN with k = -glob(11) and
//! Rc = (rad_A + rad_B)/2 of its `gxtb_parameters`, the same basis CN, both over every pair
//! (no cutoff; [`Model::cn_cutoff`]).

use crate::model::Model;
use crate::options::Version;
use axtb::Molecule;
use axtb_base::ncoord::{CnModel, Cutoff, Erf, Smooth, Unweighted};

/// Cutoff of the program's CNs (mctc-lib default), bohr.
pub const CN_CUTOFF: f64 = 25.0;
/// Steepness of the g-xTB CN.
pub const KCN_GXTB: f64 = 2.068;
/// Steepness of the basis CN.
pub const KCN_BASIS: f64 = 3.75;

/// The erf CN of `model` over the pair radii `rc` (`axtb-base`'s [`CnModel`] with the count
/// [`Erf`] of steepness `k`, 1/2 [1 + erf(-k (R - Rc)/Rc)]): 2.0.1 skips the pairs beyond
/// [`Model::reach`] of [`CN_CUTOFF`] and multiplies by the cutoff factor [`Model::cut`]; v1.1
/// counts every pair (factor 1). Pairs with R^2 < `f64::EPSILON` are skipped, and in the
/// gradient the pairs whose dE/dCN sum to zero.
#[allow(clippy::type_complexity)] // the parts are closures, which have no nameable type
fn erf_cn<'a, R: Fn(usize, usize) -> f64 + 'a>(
    model: &'a Model,
    k: f64,
    rc: R,
) -> CnModel<Erf, R, Unweighted, Smooth<impl Fn(f64) -> (f64, f64) + 'a>> {
    let cutoff = model.cn_cutoff();
    let reach = cutoff.map_or(Cutoff::None, |c| Cutoff::Squared(model.reach(c).powi(2)));
    CnModel::new(Erf { k }, rc)
        .with_cutoff(reach)
        .with_min_r2(f64::EPSILON)
        .with_switch(Smooth(move |r: f64| cutoff.map_or((1.0, 0.0), |c| model.cut(r, c))))
        .skipping_zero_weights()
}

fn gxtb_rc(mol: &Molecule) -> impl Fn(usize, usize) -> f64 + '_ {
    use crate::data::v2_0_1::CN_RCOV;
    move |i, j| CN_RCOV[mol.numbers[i] - 1] + CN_RCOV[mol.numbers[j] - 1]
}

/// v1.1's radii of the g-xTB CN (`rad` of `gxtb_parameters`, bohr) and its steepness (the
/// program writes k (R - Rc)/Rc with k = glob(11) < 0, which is [`Erf`]'s -k' (R - Rc)/Rc with
/// k' = -k bit for bit).
fn gxtb_v1_1(mol: &Molecule) -> (f64, Vec<f64>) {
    let p = crate::params::gxtb_v1_1();
    let rad = mol.numbers.iter().map(|&z| p.element(z).expect("g-xTB v1.1 parameters of a checked element").rad);
    (-p.glob[10], rad.collect())
}

fn basis_rc(mol: &Molecule) -> impl Fn(usize, usize) -> f64 + '_ {
    use crate::data::v1_1::RCOV_BASIS_CN;
    move |i, j| RCOV_BASIS_CN[mol.numbers[i] - 1] + RCOV_BASIS_CN[mol.numbers[j] - 1]
}

/// The g-xTB coordination number.
pub fn gxtb_cn(model: &Model, mol: &Molecule) -> Vec<f64> {
    match model.version {
        Version::V2_0_1 => erf_cn(model, KCN_GXTB, gxtb_rc(mol)).compute(&mol.positions),
        Version::V1_1 => {
            let (k, rad) = gxtb_v1_1(mol);
            erf_cn(model, k, move |i, j| 0.5 * (rad[i] + rad[j])).compute(&mol.positions)
        }
    }
}

/// dE/dCN of the g-xTB CN to the geometry.
pub fn gxtb_cn_gradient(model: &Model, mol: &Molecule, dedcn: &[f64], g: &mut [[f64; 3]]) {
    match model.version {
        Version::V2_0_1 => erf_cn(model, KCN_GXTB, gxtb_rc(mol)).add_gradient(&mol.positions, dedcn, g),
        Version::V1_1 => {
            let (k, rad) = gxtb_v1_1(mol);
            erf_cn(model, k, move |i, j| 0.5 * (rad[i] + rad[j])).add_gradient(&mol.positions, dedcn, g)
        }
    }
}

/// The q-vSZP basis coordination number.
pub fn basis_cn(model: &Model, mol: &Molecule) -> Vec<f64> {
    erf_cn(model, KCN_BASIS, basis_rc(mol)).compute(&mol.positions)
}

/// dE/dCN of the basis CN to the geometry.
pub fn basis_cn_gradient(model: &Model, mol: &Molecule, dedcn: &[f64], g: &mut [[f64; 3]]) {
    erf_cn(model, KCN_BASIS, basis_rc(mol)).add_gradient(&mol.positions, dedcn, g)
}
