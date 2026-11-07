//! Bond-capacitor electronegativity equilibration (EEQ-BC) charges as used by g-xTB v1.1.
//!
//! Model: T. Froitzheim, M. Müller, A. Hansen, S. Grimme, J. Chem. Phys. 162, 214109 (2025).
//! The g-xTB 1.1.0 program (`eeq_bc_`) uses the same model and the same parameters as
//! multicharge's `eeqbc2025` (verified column by column, and its global constants
//! kcn = 2, norm_exp = 0.75, kbc = 0.6, kcnrad = 0.14 appear as literals in the binary);
//! the charges below reproduce the program's to ~1e-15 (see the reference tests).
//!
//! With r_AB in bohr:
//! - CN_A = sum_B erfc-count(r_AB), count = 1/2 [1 + erf(-k_cn (r - R_AB) / R_AB^0.75)],
//!   R_AB = (rcov2_A + rcov2_B) / 2;
//! - local charge q^loc_A = sum_B (EN_B - EN_A)/3.98 * count(r_AB) + Q/N;
//! - capacitance c_AB = sqrt(cap_A cap_B) 1/2 [1 + erf(-k_bc (r_AB - R^vdW_AB) / R^vdW_AB)] with the
//!   D3 pair radii R^vdW_AB in **Ångström** (as in multicharge, which converts them back);
//! - Maxwell matrix C: C_AB = -c_AB, C_AA = sum_B c_AB, C_{N+1,N+1} = 1;
//! - x_A = -chi_A + kcnchi_A CN_A + kqchi_A q^loc_A, x_{N+1} = Q; right-hand side b = C x;
//! - charge width w_A = rad_A (1 - k_cnrad CN_A / CNavg_A^0.75);
//! - A_AB = erf(r_AB / sqrt(w_A^2 + w_B^2)) / r_AB * C_AB (A != B),
//!   A_AA = (eta_A + kqeta_A q^loc_A + sqrt(2/pi)/w_A) C_AA + 1, Lagrange row/column of ones;
//! - (q, lambda) = A^-1 b.

use crate::data::{CN_AVG, PAULING_EN, VDW_RAD_PACKED};
use crate::params::{EeqBcParams, MAX_ELEM};
use axtb::molecule::{norm2_sq, sub3};
use axtb::{Error, Molecule, Result};
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::lu::partial_pivoting::{factor, solve};
use faer::{Mat, Par};

/// libm's error function (gfortran's `ERF`), the workspace's one wrapper.
pub(crate) use axtb_d4::erf as erf_f64;

/// Steepness of the EEQ-BC counting function.
pub const KCN: f64 = 2.0;
/// Exponent of the distance (and average-CN) normalization.
pub const NORM_EXP: f64 = 0.75;
/// Steepness of the bond capacitance switch.
pub const KBC: f64 = 0.6;
/// CN scaling of the charge width.
pub const KCNRAD: f64 = 0.14;
/// Electronegativity normalization (fluorine).
pub const EN_NORM: f64 = 3.98;

/// D3 pairwise van der Waals radius (Ångström) of atomic numbers `za`, `zb` (1-based).
pub fn vdw_rad_pair(za: usize, zb: usize) -> f64 {
    let (hi, lo) = if za >= zb { (za, zb) } else { (zb, za) };
    VDW_RAD_PACKED[hi * (hi - 1) / 2 + lo - 1]
}

/// Solve `a x = b` in place (`b` becomes `x`; `a` is overwritten by its LU factors) with a
/// sequential partially pivoted LU. faer's high-level `a.partial_piv_lu()` would run on faer's
/// global parallelism, i.e. start rayon's global pool (one thread per CPU) in every process.
pub(crate) fn solve_in_place(a: &mut Mat<f64>, b: &mut Mat<f64>) {
    let m = a.nrows();
    let par = Par::Seq;
    let (mut perm, mut perm_inv) = (vec![0usize; m], vec![0usize; m]);
    let mut mem = MemBuffer::new(StackReq::any_of(&[
        factor::lu_in_place_scratch::<usize, f64>(m, m, par, Default::default()),
        solve::solve_in_place_scratch::<usize, f64>(m, b.ncols(), par),
    ]));
    let (_, p) =
        factor::lu_in_place(a.as_mut(), &mut perm, &mut perm_inv, par, MemStack::new(&mut mem), Default::default());
    solve::solve_in_place(a.as_ref(), a.as_ref(), p, b.as_mut(), par, MemStack::new(&mut mem));
}

/// EEQ-BC results.
#[derive(Clone, Debug)]
pub struct EeqBcCharges {
    /// Coordination numbers (printed as "CN" by the program).
    pub cn: Vec<f64>,
    /// Electronegativity-weighted local charges (printed as "q_CN").
    pub qloc: Vec<f64>,
    /// Partial charges (printed as "q").
    pub charges: Vec<f64>,
}

/// EEQ-BC partial charges of `mol` (positions in bohr; total charge `mol.charge`).
pub fn eeqbc_charges(param: &EeqBcParams, mol: &Molecule) -> Result<EeqBcCharges> {
    let n = mol.nat();
    if let Some(&z) = mol.numbers.iter().find(|&&z| z > MAX_ELEM) {
        return Err(Error::UnsupportedElement { method: "EEQ-BC", z });
    }
    let el = |i: usize| &param.elements[mol.numbers[i]];
    let mut cn = vec![0.0; n];
    let mut qloc = vec![mol.charge / n as f64; n];
    let mut dist = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..i {
            let r = norm2_sq(sub3(mol.positions[i], mol.positions[j])).sqrt();
            dist[i * n + j] = r;
            dist[j * n + i] = r;
            let rc = 0.5 * (el(i).rcov2 + el(j).rcov2);
            let count = 0.5 * (1.0 + erf_f64(-KCN * (r - rc) / rc.powf(NORM_EXP)));
            cn[i] += count;
            cn[j] += count;
            let den = (PAULING_EN[mol.numbers[j] - 1] - PAULING_EN[mol.numbers[i] - 1]) / EN_NORM;
            qloc[i] += den * count;
            qloc[j] -= den * count;
        }
    }
    // Maxwell capacitance matrix
    let m = n + 1;
    let mut cmat = vec![0.0; m * m];
    for i in 0..n {
        for j in 0..i {
            let rvdw = vdw_rad_pair(mol.numbers[i], mol.numbers[j]);
            let c = (el(i).cap * el(j).cap).sqrt() * 0.5 * (1.0 + erf_f64(-KBC * (dist[i * n + j] - rvdw) / rvdw));
            cmat[i * m + j] = -c;
            cmat[j * m + i] = -c;
            cmat[i * m + i] += c;
            cmat[j * m + j] += c;
        }
    }
    cmat[n * m + n] = 1.0;
    let mut x = vec![0.0; m];
    for i in 0..n {
        x[i] = -el(i).chi + el(i).kcnchi * cn[i] + el(i).kqchi * qloc[i];
    }
    x[n] = mol.charge;
    let width: Vec<f64> =
        (0..n).map(|i| el(i).rad * (1.0 - KCNRAD * cn[i] / CN_AVG[mol.numbers[i] - 1].powf(NORM_EXP))).collect();
    let sqrt2pi = (2.0 / std::f64::consts::PI).sqrt();
    let mut amat = Mat::<f64>::zeros(m, m);
    let mut rhs = Mat::<f64>::zeros(m, 1);
    for i in 0..m {
        rhs[(i, 0)] = (0..m).map(|k| cmat[i * m + k] * x[k]).sum();
    }
    for i in 0..n {
        for j in 0..i {
            let r = dist[i * n + j];
            let gam2 = 1.0 / (width[i] * width[i] + width[j] * width[j]);
            let a = erf_f64((r * r * gam2).sqrt()) / r * cmat[j * m + i];
            amat[(i, j)] = a;
            amat[(j, i)] = a;
        }
        amat[(i, i)] = (el(i).eta + el(i).kqeta * qloc[i] + sqrt2pi / width[i]) * cmat[i * m + i] + 1.0;
        amat[(i, n)] = 1.0;
        amat[(n, i)] = 1.0;
    }
    solve_in_place(&mut amat, &mut rhs);
    let charges: Vec<f64> = (0..n).map(|i| rhs[(i, 0)]).collect();
    if charges.iter().any(|q| !q.is_finite()) {
        return Err(Error::Numerical("EEQ-BC: singular charge equations".into()));
    }
    Ok(EeqBcCharges { cn, qloc, charges })
}

/// Gradient of a function of the EEQ-BC charges: with `dedq[A]` = dE/dq_A, adds
/// sum_A dE/dq_A dq_A/dR to `grad` (positions in bohr). Adjoint of the linear charge
/// equations A(R) y = b(R), y = (q, lambda): dE/dR = w^T (db/dR - dA/dR y) with A w = (dE/dq, 0)
/// (A is symmetric). R enters through the distances (Coulomb kernel, bond capacitances), the
/// coordination numbers (charge widths, electronegativities) and the local charges.
pub fn eeqbc_charge_gradient(param: &EeqBcParams, mol: &Molecule, dedq: &[f64], grad: &mut [[f64; 3]]) -> Result<()> {
    let n = mol.nat();
    let m = n + 1;
    let el = |i: usize| &param.elements[mol.numbers[i]];
    let sqrtpi = std::f64::consts::PI.sqrt();
    // geometry-dependent quantities, as in eeqbc_charges
    let mut cn = vec![0.0; n];
    let mut qloc = vec![mol.charge / n as f64; n];
    let mut dist = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..i {
            let r = norm2_sq(sub3(mol.positions[i], mol.positions[j])).sqrt();
            dist[i * n + j] = r;
            dist[j * n + i] = r;
            let rc = 0.5 * (el(i).rcov2 + el(j).rcov2);
            let count = 0.5 * (1.0 + erf_f64(-KCN * (r - rc) / rc.powf(NORM_EXP)));
            cn[i] += count;
            cn[j] += count;
            let den = (PAULING_EN[mol.numbers[j] - 1] - PAULING_EN[mol.numbers[i] - 1]) / EN_NORM;
            qloc[i] += den * count;
            qloc[j] -= den * count;
        }
    }
    let cap = |i: usize, j: usize| -> (f64, f64) {
        let rvdw = vdw_rad_pair(mol.numbers[i], mol.numbers[j]);
        let pre = (el(i).cap * el(j).cap).sqrt();
        let r = dist[i * n + j];
        let arg = -KBC * (r - rvdw) / rvdw;
        let c = pre * 0.5 * (1.0 + erf_f64(arg));
        let dc = pre * (-KBC / rvdw) * (-arg * arg).exp() / sqrtpi;
        (c, dc)
    };
    let mut cmat = vec![0.0; m * m];
    for i in 0..n {
        for j in 0..i {
            let c = cap(i, j).0;
            cmat[i * m + j] = -c;
            cmat[j * m + i] = -c;
            cmat[i * m + i] += c;
            cmat[j * m + j] += c;
        }
    }
    cmat[n * m + n] = 1.0;
    let mut x = vec![0.0; m];
    for i in 0..n {
        x[i] = -el(i).chi + el(i).kcnchi * cn[i] + el(i).kqchi * qloc[i];
    }
    x[n] = mol.charge;
    let width: Vec<f64> =
        (0..n).map(|i| el(i).rad * (1.0 - KCNRAD * cn[i] / CN_AVG[mol.numbers[i] - 1].powf(NORM_EXP))).collect();
    let dwidth: Vec<f64> = (0..n).map(|i| -el(i).rad * KCNRAD / CN_AVG[mol.numbers[i] - 1].powf(NORM_EXP)).collect();
    let sqrt2pi = (2.0 / std::f64::consts::PI).sqrt();
    let hdiag: Vec<f64> = (0..n).map(|i| el(i).eta + el(i).kqeta * qloc[i] + sqrt2pi / width[i]).collect();
    let mut amat = Mat::<f64>::zeros(m, m);
    let mut rhs = Mat::<f64>::zeros(m, 1);
    for i in 0..m {
        rhs[(i, 0)] = (0..m).map(|k| cmat[i * m + k] * x[k]).sum();
    }
    let gfun = |i: usize, j: usize| -> (f64, f64, f64) {
        // g = erf(r/sigma)/r and its derivatives with respect to r and sigma
        let r = dist[i * n + j];
        let sigma = (width[i] * width[i] + width[j] * width[j]).sqrt();
        let t = r / sigma;
        let e = erf_f64(t);
        let gauss = 2.0 / sqrtpi * (-t * t).exp();
        (e / r, gauss / (sigma * r) - e / (r * r), -gauss / (sigma * sigma))
    };
    for i in 0..n {
        for j in 0..i {
            let a = gfun(i, j).0 * cmat[j * m + i];
            amat[(i, j)] = a;
            amat[(j, i)] = a;
        }
        amat[(i, i)] = hdiag[i] * cmat[i * m + i] + 1.0;
        amat[(i, n)] = 1.0;
        amat[(n, i)] = 1.0;
    }
    let mut lu = amat.clone();
    let mut y = rhs.clone();
    solve_in_place(&mut lu, &mut y);
    let mut w = Mat::<f64>::zeros(m, 1);
    for i in 0..n {
        w[(i, 0)] = dedq[i];
    }
    let mut lu = amat;
    solve_in_place(&mut lu, &mut w);
    let (w, y): (Vec<f64>, Vec<f64>) = ((0..m).map(|i| w[(i, 0)]).collect(), (0..m).map(|i| y[(i, 0)]).collect());
    // C^T w (C symmetric)
    let ctw: Vec<f64> = (0..m).map(|i| (0..m).map(|k| cmat[k * m + i] * w[k]).sum()).collect();
    // dE/dCN and dE/dqloc per atom; dE/dr per pair
    let mut decn = vec![0.0; n];
    let mut deql = vec![0.0; n];
    for i in 0..n {
        // x_i: T1 = w^T C dx
        decn[i] += ctw[i] * el(i).kcnchi;
        deql[i] += ctw[i] * el(i).kqchi;
        // A_ii = h_i C_ii + 1: -w_i y_i C_ii dh_i
        deql[i] -= w[i] * y[i] * cmat[i * m + i] * el(i).kqeta;
        decn[i] -= w[i] * y[i] * cmat[i * m + i] * (-sqrt2pi / (width[i] * width[i])) * dwidth[i];
    }
    for i in 0..n {
        for j in 0..i {
            let (g, dgdr, dgds) = gfun(i, j);
            let (_, dcdr) = cap(i, j);
            let cij = cmat[i * m + j]; // = -c_ij
            let sym = w[i] * y[j] + w[j] * y[i];
            // widths: sigma = sqrt(w_i^2 + w_j^2)
            let sigma = (width[i] * width[i] + width[j] * width[j]).sqrt();
            decn[i] -= sym * cij * dgds * width[i] / sigma * dwidth[i];
            decn[j] -= sym * cij * dgds * width[j] / sigma * dwidth[j];
            // distance: T1 via C, T2 via g and C
            let t1 = dcdr * (w[i] - w[j]) * (x[i] - x[j]);
            let t2 = sym * (dgdr * cij - g * dcdr) + w[i] * hdiag[i] * dcdr * y[i] + w[j] * hdiag[j] * dcdr * y[j];
            let dedr = t1 - t2;
            let r = dist[i * n + j];
            let v = sub3(mol.positions[i], mol.positions[j]);
            for k in 0..3 {
                grad[i][k] += dedr * v[k] / r;
                grad[j][k] -= dedr * v[k] / r;
            }
        }
    }
    // chain through the counting function of CN and qloc
    for i in 0..n {
        for j in 0..i {
            let r = dist[i * n + j];
            let rc = 0.5 * (el(i).rcov2 + el(j).rcov2);
            let rcn = rc.powf(NORM_EXP);
            let arg = -KCN * (r - rc) / rcn;
            let dcount = -KCN / rcn * (-arg * arg).exp() / sqrtpi;
            let den = (PAULING_EN[mol.numbers[j] - 1] - PAULING_EN[mol.numbers[i] - 1]) / EN_NORM;
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
