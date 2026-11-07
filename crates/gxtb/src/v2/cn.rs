//! Error-function coordination numbers of g-xTB 2.0.1 (mctc-lib `erf` counting with a pair
//! cutoff): CN_A = sum_B 1/2 [1 + erf(-k (R_AB - Rc_AB) / Rc_AB^e)] * cut(R_AB), with the cutoff
//! factor of [`Model::cut`] (25 bohr in the program).
//!
//! - g-xTB CN ([`gxtb_cn`]): k = 2.068, Rc = rcov_A + rcov_B (`pa_cn_rcov`), e = 1; used raw by
//!   the H0 self-energies and through the smoothed square root by the repulsion, first and
//!   second order, the AES dipole scaling.
//! - basis CN ([`basis_cn`]): k = 3.75, Rc = R_A + R_B (q-vSZP radii), e = 1; q_eff.

use super::Model;
use crate::eeqbc::erf_f64;
use axtb::molecule::{norm2_sq, sub3};
use axtb::Molecule;

/// Cutoff of the program's CNs (mctc-lib default), bohr.
pub const CN_CUTOFF: f64 = 25.0;
/// Steepness of the g-xTB CN.
pub const KCN_GXTB: f64 = 2.068;
/// Steepness of the basis CN.
pub const KCN_BASIS: f64 = 3.75;

const SQRTPI: f64 = 1.772_453_850_905_516;

/// One counting term and its derivative with respect to R.
#[inline]
pub fn erf_count(k: f64, r: f64, rc: f64, den: f64) -> (f64, f64) {
    let arg = -k * (r - rc) / den;
    (0.5 * (1.0 + erf_f64(arg)), -k / den * (-arg * arg).exp() / SQRTPI)
}

/// Generic erf CN with radii sums `rc(i, j)` and normalization `den(rc)`.
pub fn erf_cn(
    model: &Model,
    mol: &Molecule,
    k: f64,
    rc: impl Fn(usize, usize) -> f64,
    den: impl Fn(f64) -> f64,
) -> Vec<f64> {
    let n = mol.nat();
    let reach2 = model.reach(CN_CUTOFF).powi(2);
    let mut cn = vec![0.0; n];
    for i in 0..n {
        for j in 0..i {
            let r2 = norm2_sq(sub3(mol.positions[i], mol.positions[j]));
            if r2 > reach2 || r2 < f64::EPSILON {
                continue;
            }
            let r = r2.sqrt();
            let c = rc(i, j);
            let (cnt, _) = erf_count(k, r, c, den(c));
            let (sw, _) = model.cut(r, CN_CUTOFF);
            let v = cnt * sw;
            cn[i] += v;
            cn[j] += v;
        }
    }
    cn
}

/// Adds sum_A dE/dCN_A dCN_A/dR of an erf CN to `g`.
pub fn erf_cn_gradient(
    model: &Model,
    mol: &Molecule,
    k: f64,
    rc: impl Fn(usize, usize) -> f64,
    den: impl Fn(f64) -> f64,
    dedcn: &[f64],
    g: &mut [[f64; 3]],
) {
    let n = mol.nat();
    let reach2 = model.reach(CN_CUTOFF).powi(2);
    for i in 0..n {
        for j in 0..i {
            let w = dedcn[i] + dedcn[j];
            if w == 0.0 {
                continue;
            }
            let v = sub3(mol.positions[i], mol.positions[j]);
            let r2 = norm2_sq(v);
            if r2 > reach2 || r2 < f64::EPSILON {
                continue;
            }
            let r = r2.sqrt();
            let c = rc(i, j);
            let (cnt, dcnt) = erf_count(k, r, c, den(c));
            let (sw, dsw) = model.cut(r, CN_CUTOFF);
            let f = w * (dcnt * sw + cnt * dsw) / r;
            for x in 0..3 {
                g[i][x] += f * v[x];
                g[j][x] -= f * v[x];
            }
        }
    }
}

fn gxtb_rc(mol: &Molecule) -> impl Fn(usize, usize) -> f64 + '_ {
    use crate::data_v2::CN_RCOV;
    move |i, j| CN_RCOV[mol.numbers[i] - 1] + CN_RCOV[mol.numbers[j] - 1]
}

fn basis_rc(mol: &Molecule) -> impl Fn(usize, usize) -> f64 + '_ {
    use crate::data::RCOV_BASIS_CN;
    move |i, j| RCOV_BASIS_CN[mol.numbers[i] - 1] + RCOV_BASIS_CN[mol.numbers[j] - 1]
}

/// The g-xTB coordination number.
pub fn gxtb_cn(model: &Model, mol: &Molecule) -> Vec<f64> {
    erf_cn(model, mol, KCN_GXTB, gxtb_rc(mol), |c| c)
}

/// dE/dCN of the g-xTB CN to the geometry.
pub fn gxtb_cn_gradient(model: &Model, mol: &Molecule, dedcn: &[f64], g: &mut [[f64; 3]]) {
    erf_cn_gradient(model, mol, KCN_GXTB, gxtb_rc(mol), |c| c, dedcn, g)
}

/// The q-vSZP basis coordination number.
pub fn basis_cn(model: &Model, mol: &Molecule) -> Vec<f64> {
    erf_cn(model, mol, KCN_BASIS, basis_rc(mol), |c| c)
}

/// dE/dCN of the basis CN to the geometry.
pub fn basis_cn_gradient(model: &Model, mol: &Molecule, dedcn: &[f64], g: &mut [[f64; 3]]) {
    erf_cn_gradient(model, mol, KCN_BASIS, basis_rc(mol), |c| c, dedcn, g)
}
