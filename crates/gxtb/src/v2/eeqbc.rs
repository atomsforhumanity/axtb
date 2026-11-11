//! EEQ-BC charges of g-xTB 2.0.1 (multicharge `eeqbc2025` as built into the 2.0.1 program):
//! the v1.1 model ([`crate::eeqbc`]) with refitted parameters ([`crate::data_v2`] `EEQBC_*`)
//! and three changes (spec §4.1):
//!
//! - no local-charge hardness term: A_AA = (eta_A + sqrt(2/pi)/w_A) C_AA + 1;
//! - w_A = rad_A (1 - 0.3 CN_A / CNavg_A^0.75);
//! - c_AB = sqrt(cap_A cap_B) 1/2 [1 + erf(-0.75 (R - Rv_AB)/Rv_AB)] with
//!   Rv_AB = R^D3_AB (bohr) / (2 aatoau) (s_A + s_B), s = `eeqbc_rvdw_scale`.
//!
//! CN_A = sum_B 1/2 [1 + erf(-2 (R - Rc)/Rc^0.75)], Rc = r_A + r_B (`eeqbc_cov_radii`), and
//! the local charges q^loc (Pauling EN / 3.98, multicharge's actinide values) use the 25-bohr
//! CN cutoff of the program ([`super::Model::cut`]).

use super::cn::{erf_count, CN_CUTOFF};
use super::Model;
use crate::data::PAULING_EN;
use crate::data_v2::{
    EEQBC_AVG_CN, EEQBC_CAP, EEQBC_CHI, EEQBC_COV_RADII, EEQBC_ETA, EEQBC_KCNCHI, EEQBC_KQCHI, EEQBC_RAD,
    EEQBC_RVDW_SCALE,
};
use crate::eeqbc::{erf_f64, solve_in_place, EeqBcCharges};
use axtb::molecule::{norm2_sq, sub3};
use axtb::{Error, Molecule, Result};
use faer::Mat;

const KCN: f64 = 2.0;
const NORM_EXP: f64 = 0.75;
const KBC: f64 = 0.75;
const KCNRAD: f64 = 0.3;
const EN_NORM: f64 = 3.98;
const SQRTPI: f64 = 1.772_453_850_905_516;

/// Geometry-dependent quantities of the charge equations.
struct Geometry {
    n: usize,
    dist: Vec<f64>,
    cn: Vec<f64>,
    qloc: Vec<f64>,
    /// Maxwell capacitance matrix (n+1)^2 and the pair capacitances' R derivatives.
    cmat: Vec<f64>,
    x: Vec<f64>,
    width: Vec<f64>,
    hdiag: Vec<f64>,
}

fn z(mol: &Molecule, i: usize) -> usize {
    mol.numbers[i] - 1
}

fn rvdw(model: &Model, mol: &Molecule, i: usize, j: usize) -> f64 {
    model.vdw_pair(mol.numbers[i], mol.numbers[j])
        * (0.5 / model.aatoau)
        * (EEQBC_RVDW_SCALE[z(mol, i)] + EEQBC_RVDW_SCALE[z(mol, j)])
}

fn capacitance(model: &Model, mol: &Molecule, i: usize, j: usize, r: f64) -> (f64, f64) {
    let rv = rvdw(model, mol, i, j);
    let pre = (EEQBC_CAP[z(mol, i)] * EEQBC_CAP[z(mol, j)]).sqrt();
    let arg = -KBC * (r - rv) / rv;
    (pre * 0.5 * (1.0 + erf_f64(arg)), pre * (-KBC / rv) * (-arg * arg).exp() / SQRTPI)
}

/// Count of the CN and the local charges, with the cutoff factor, and its R derivative.
fn count(model: &Model, mol: &Molecule, i: usize, j: usize, r: f64) -> (f64, f64) {
    if r > model.reach(CN_CUTOFF) {
        return (0.0, 0.0);
    }
    let rc = EEQBC_COV_RADII[z(mol, i)] + EEQBC_COV_RADII[z(mol, j)];
    let (c, dc) = erf_count(KCN, r, rc, rc.powf(NORM_EXP));
    let (sw, dsw) = model.cut(r, CN_CUTOFF);
    (c * sw, dc * sw + c * dsw)
}

fn geometry(model: &Model, mol: &Molecule) -> Result<Geometry> {
    super::check_elements(&mol.numbers)?;
    let n = mol.nat();
    let mut cn = vec![0.0; n];
    let mut qloc = vec![mol.charge / n as f64; n];
    let mut dist = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..i {
            let r = norm2_sq(sub3(mol.positions[i], mol.positions[j])).sqrt();
            dist[i * n + j] = r;
            dist[j * n + i] = r;
            let (c, _) = count(model, mol, i, j, r);
            cn[i] += c;
            cn[j] += c;
            let den = (PAULING_EN[z(mol, j)] - PAULING_EN[z(mol, i)]) / EN_NORM;
            qloc[i] += den * c;
            qloc[j] -= den * c;
        }
    }
    let m = n + 1;
    let mut cmat = vec![0.0; m * m];
    for i in 0..n {
        for j in 0..i {
            let (c, _) = capacitance(model, mol, i, j, dist[i * n + j]);
            cmat[i * m + j] = -c;
            cmat[j * m + i] = -c;
            cmat[i * m + i] += c;
            cmat[j * m + j] += c;
        }
    }
    cmat[n * m + n] = 1.0;
    let mut x = vec![0.0; m];
    for i in 0..n {
        let k = z(mol, i);
        x[i] = -EEQBC_CHI[k] + EEQBC_KCNCHI[k] * cn[i] + EEQBC_KQCHI[k] * qloc[i];
    }
    x[n] = mol.charge;
    let width: Vec<f64> = (0..n)
        .map(|i| EEQBC_RAD[z(mol, i)] * (1.0 - KCNRAD * cn[i] / EEQBC_AVG_CN[z(mol, i)].powf(NORM_EXP)))
        .collect();
    let sqrt2pi = (2.0 / std::f64::consts::PI).sqrt();
    let hdiag: Vec<f64> = (0..n).map(|i| EEQBC_ETA[z(mol, i)] + sqrt2pi / width[i]).collect();
    Ok(Geometry { n, dist, cn, qloc, cmat, x, width, hdiag })
}

fn matrices(g: &Geometry) -> (Mat<f64>, Mat<f64>) {
    let (n, m) = (g.n, g.n + 1);
    let mut amat = Mat::<f64>::zeros(m, m);
    let mut rhs = Mat::<f64>::zeros(m, 1);
    for i in 0..m {
        rhs[(i, 0)] = (0..m).map(|k| g.cmat[i * m + k] * g.x[k]).sum();
    }
    for i in 0..n {
        for j in 0..i {
            let r = g.dist[i * n + j];
            let gam2 = 1.0 / (g.width[i] * g.width[i] + g.width[j] * g.width[j]);
            let a = erf_f64((r * r * gam2).sqrt()) / r * g.cmat[j * m + i];
            amat[(i, j)] = a;
            amat[(j, i)] = a;
        }
        amat[(i, i)] = g.hdiag[i] * g.cmat[i * m + i] + 1.0;
        amat[(i, n)] = 1.0;
        amat[(n, i)] = 1.0;
    }
    (amat, rhs)
}

/// EEQ-BC partial charges of `mol` (positions in bohr; total charge `mol.charge`).
pub fn eeqbc_charges(model: &Model, mol: &Molecule) -> Result<EeqBcCharges> {
    let g = geometry(model, mol)?;
    let (amat, mut rhs) = matrices(&g);
    solve_in_place(amat, &mut rhs);
    let charges: Vec<f64> = (0..g.n).map(|i| rhs[(i, 0)]).collect();
    if charges.iter().any(|q| !q.is_finite()) {
        return Err(Error::Numerical("EEQ-BC: singular charge equations".into()));
    }
    Ok(EeqBcCharges { cn: g.cn, qloc: g.qloc, charges })
}

/// Adds sum_A dE/dq_A dq_A/dR to `grad` for a function E of the EEQ-BC charges (adjoint of
/// the linear charge equations A y = b: dE/dR = w^T (db/dR - dA/dR y), A w = (dE/dq, 0)).
pub fn eeqbc_charge_gradient(model: &Model, mol: &Molecule, dedq: &[f64], grad: &mut [[f64; 3]]) -> Result<()> {
    let g = geometry(model, mol)?;
    let (n, m) = (g.n, g.n + 1);
    let (amat, rhs) = matrices(&g);
    let mut y = rhs;
    solve_in_place(amat.clone(), &mut y);
    let mut w = Mat::<f64>::zeros(m, 1);
    for i in 0..n {
        w[(i, 0)] = dedq[i];
    }
    solve_in_place(amat, &mut w);
    let (w, y): (Vec<f64>, Vec<f64>) = ((0..m).map(|i| w[(i, 0)]).collect(), (0..m).map(|i| y[(i, 0)]).collect());
    let cmat = &g.cmat;
    let ctw: Vec<f64> = (0..m).map(|i| (0..m).map(|k| cmat[k * m + i] * w[k]).sum()).collect();
    let sqrt2pi = (2.0 / std::f64::consts::PI).sqrt();
    let mut decn = vec![0.0; n];
    let mut deql = vec![0.0; n];
    for i in 0..n {
        let k = z(mol, i);
        decn[i] += ctw[i] * EEQBC_KCNCHI[k];
        deql[i] += ctw[i] * EEQBC_KQCHI[k];
        let dwidth = -EEQBC_RAD[k] * KCNRAD / EEQBC_AVG_CN[k].powf(NORM_EXP);
        decn[i] -= w[i] * y[i] * cmat[i * m + i] * (-sqrt2pi / (g.width[i] * g.width[i])) * dwidth;
    }
    let gfun = |i: usize, j: usize| -> (f64, f64, f64) {
        let r = g.dist[i * n + j];
        let sigma = (g.width[i] * g.width[i] + g.width[j] * g.width[j]).sqrt();
        let t = r / sigma;
        let e = erf_f64(t);
        let gauss = 2.0 / SQRTPI * (-t * t).exp();
        (e / r, gauss / (sigma * r) - e / (r * r), -gauss / (sigma * sigma))
    };
    for i in 0..n {
        for j in 0..i {
            let r = g.dist[i * n + j];
            let (gg, dgdr, dgds) = gfun(i, j);
            let (_, dcdr) = capacitance(model, mol, i, j, r);
            let cij = cmat[i * m + j];
            let sym = w[i] * y[j] + w[j] * y[i];
            let sigma = (g.width[i] * g.width[i] + g.width[j] * g.width[j]).sqrt();
            let (ki, kj) = (z(mol, i), z(mol, j));
            let dwi = -EEQBC_RAD[ki] * KCNRAD / EEQBC_AVG_CN[ki].powf(NORM_EXP);
            let dwj = -EEQBC_RAD[kj] * KCNRAD / EEQBC_AVG_CN[kj].powf(NORM_EXP);
            decn[i] -= sym * cij * dgds * g.width[i] / sigma * dwi;
            decn[j] -= sym * cij * dgds * g.width[j] / sigma * dwj;
            let t1 = dcdr * (w[i] - w[j]) * (g.x[i] - g.x[j]);
            let t2 = sym * (dgdr * cij - gg * dcdr) + w[i] * g.hdiag[i] * dcdr * y[i] + w[j] * g.hdiag[j] * dcdr * y[j];
            let dedr = t1 - t2;
            let v = sub3(mol.positions[i], mol.positions[j]);
            for k in 0..3 {
                grad[i][k] += dedr * v[k] / r;
                grad[j][k] -= dedr * v[k] / r;
            }
        }
    }
    for i in 0..n {
        for j in 0..i {
            let r = g.dist[i * n + j];
            let (_, dcount) = count(model, mol, i, j, r);
            if dcount == 0.0 {
                continue;
            }
            let den = (PAULING_EN[z(mol, j)] - PAULING_EN[z(mol, i)]) / EN_NORM;
            let dedr = (decn[i] + decn[j]) * dcount + (deql[i] - deql[j]) * den * dcount;
            let v = sub3(mol.positions[i], mol.positions[j]);
            for k in 0..3 {
                grad[i][k] += dedr * v[k] / r;
                grad[j][k] -= dedr * v[k] / r;
            }
        }
    }
    Ok(())
}
