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

use super::Model;
use crate::eeqbc::erf_f64;
use crate::options::Version;
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
    let mut cn = vec![0.0; n];
    let Some(cutoff) = model.cn_cutoff() else {
        // every pair, no cutoff factor (v1.1)
        for i in 0..n {
            for j in 0..i {
                let r = norm2_sq(sub3(mol.positions[i], mol.positions[j])).sqrt();
                let c = rc(i, j);
                let v = erf_count(k, r, c, den(c)).0;
                cn[i] += v;
                cn[j] += v;
            }
        }
        return cn;
    };
    let reach2 = model.reach(cutoff).powi(2);
    for i in 0..n {
        for j in 0..i {
            let r2 = norm2_sq(sub3(mol.positions[i], mol.positions[j]));
            if r2 > reach2 || r2 < f64::EPSILON {
                continue;
            }
            let r = r2.sqrt();
            let c = rc(i, j);
            let (cnt, _) = erf_count(k, r, c, den(c));
            let (sw, _) = model.cut(r, cutoff);
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
    let cutoff = model.cn_cutoff();
    let reach2 = cutoff.map_or(f64::INFINITY, |c| model.reach(c).powi(2));
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
            let f = match cutoff {
                Some(cutoff) => {
                    let (sw, dsw) = model.cut(r, cutoff);
                    w * (dcnt * sw + cnt * dsw) / r
                }
                None => w * dcnt / r,
            };
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

/// v1.1's radii of the g-xTB CN (`rad` of `gxtb_parameters`, bohr) and its steepness (the
/// program writes k (R - Rc)/Rc with k = glob(11) < 0, which is [`erf_count`]'s -k' (R - Rc)/Rc
/// with k' = -k bit for bit).
fn gxtb_v1_1(mol: &Molecule) -> (f64, Vec<f64>) {
    let p = crate::params::gxtb_v1_1();
    let rad = mol.numbers.iter().map(|&z| p.element(z).expect("g-xTB v1.1 parameters of a checked element").rad);
    (-p.glob[10], rad.collect())
}

fn basis_rc(mol: &Molecule) -> impl Fn(usize, usize) -> f64 + '_ {
    use crate::data::RCOV_BASIS_CN;
    move |i, j| RCOV_BASIS_CN[mol.numbers[i] - 1] + RCOV_BASIS_CN[mol.numbers[j] - 1]
}

/// The g-xTB coordination number.
pub fn gxtb_cn(model: &Model, mol: &Molecule) -> Vec<f64> {
    match model.version {
        Version::V2_0_1 => erf_cn(model, mol, KCN_GXTB, gxtb_rc(mol), |c| c),
        Version::V1_1 => {
            let (k, rad) = gxtb_v1_1(mol);
            erf_cn(model, mol, k, |i, j| 0.5 * (rad[i] + rad[j]), |c| c)
        }
    }
}

/// dE/dCN of the g-xTB CN to the geometry.
pub fn gxtb_cn_gradient(model: &Model, mol: &Molecule, dedcn: &[f64], g: &mut [[f64; 3]]) {
    match model.version {
        Version::V2_0_1 => erf_cn_gradient(model, mol, KCN_GXTB, gxtb_rc(mol), |c| c, dedcn, g),
        Version::V1_1 => {
            let (k, rad) = gxtb_v1_1(mol);
            erf_cn_gradient(model, mol, k, |i, j| 0.5 * (rad[i] + rad[j]), |c| c, dedcn, g)
        }
    }
}

/// The q-vSZP basis coordination number.
pub fn basis_cn(model: &Model, mol: &Molecule) -> Vec<f64> {
    erf_cn(model, mol, KCN_BASIS, basis_rc(mol), |c| c)
}

/// dE/dCN of the basis CN to the geometry.
pub fn basis_cn_gradient(model: &Model, mol: &Molecule, dedcn: &[f64], g: &mut [[f64; 3]]) {
    erf_cn_gradient(model, mol, KCN_BASIS, basis_rc(mol), |c| c, dedcn, g)
}
