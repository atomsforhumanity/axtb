//! Fock matrices of g-xTB v1.1: the derivatives of the energy terms with respect to the
//! spin densities, as the 1.1.0 program builds them in its SCF (`set1espot_`, `setespot_`,
//! `set3espot_`, `setvsdq_` + `addanisotropich1_`, `setospot_`, `setfock_mat_`,
//! `setfock_indo_`, `tblite_disp_d4_mp_get_potential_`), verified against the program's
//! Fock matrices (reference tests).
//!
//! Conventions: full row-major `n x n` matrices in axtb AO order; densities per spin (closed
//! shell: one matrix, the total density); `F^s = dE/dP^s` (closed shell: `dE/dP`).
//!
//! - Mulliken charge potentials: with V_l = dE/dq_l for the shell charges
//!   q_l = n^ref_l - sum_{mu in l} (P S)_{mu mu}, F_{mu nu} += -1/2 (V_l(mu) + V_l(nu)) S_{mu nu}
//!   (both spins). Atomic potentials (revD4, AES charge part) enter every shell of the atom.
//! - Multipole potentials (AES): F_{mu nu} += -1/2 [v_d(A).D_{mu nu} + v_d(B).D_{nu mu} +
//!   v_Q(A).Q_{mu nu} + v_Q(B).Q_{nu mu}] with the dipole/quadrupole integrals around the
//!   atom of the first index. Only the charge-dipole, charge-quadrupole and dipole-dipole parts
//!   of E_AES are differentiated (the program's `setvsdq_`); the dipole-quadrupole and
//!   quadrupole-quadrupole energies have no potential.
//! - Spin polarization: w_l = dE/dm_l, F^alpha += 1/2 (w_l(mu) + w_l(nu)) S, F^beta -= the same.
//! - Mulliken exchange: F^s = -1/4 [(g o SP) S + S (g o PS) + g o SPS + S (g o P) S]
//!   (P = P^s; closed shell -1/8 with the total density), `o` the elementwise product with the
//!   AO kernel g.
//! - Onsite exchange: F^s = 1/2 (Y S + S Y) + D with Y = dE/dPt (dual density Pt) for the
//!   exchange-type pairs and, for the population part y_k = dE/dPt_kk, D_kk = y_k and
//!   D_kl = (y_k + y_l) S_kl (k != l): the program applies the latter without the factor 1/2,
//!   i.e. at twice its derivative off the diagonal.
//!
//! Apart from the two deviations noted (AES, onsite exchange) the Fock matrices are the exact
//! derivatives of the energy (checked by finite differences).

use crate::data::{GAM3, GMUNU, SPIN_W0};
use crate::eeqbc::erf_f64;
use crate::es::{atom_charges, reference_drho, spin_index, K1_DIS, K1_S, K4};
use crate::exchange::{gmunu_index, OFX_SCALE};
use crate::params::GxtbParams;
use axtb::{Molecule, Result};
use faer::{MatMut, MatRef, Par};

/// Shell list entry: (atom, shell index on the atom, l).
pub type Shell = (usize, usize, usize);

/// The total density of spin densities `p` (closed shell: the one matrix, copied), for both
/// versions.
pub fn total_density(p: &[Vec<f64>]) -> Vec<f64> {
    if p.len() == 1 {
        p[0].clone()
    } else {
        p[0].iter().zip(&p[1]).map(|(a, b)| a + b).collect()
    }
}

/// `a b` for row-major square matrices (faer on row-major views, no copies), on `par`
/// (explicit; never faer's global setting).
pub fn matmul(n: usize, a: &[f64], b: &[f64], par: Par) -> Vec<f64> {
    let mut out = vec![0.0; n * n];
    faer::linalg::matmul::matmul(
        MatMut::from_row_major_slice_mut(&mut out, n, n),
        faer::Accum::Replace,
        MatRef::from_row_major_slice(a, n, n),
        MatRef::from_row_major_slice(b, n, n),
        1.0,
        par,
    );
    out
}

/// Shell potentials V_l = d(E1 + E2 + E3 + E4)/dq_l.
pub fn isotropic_shell_potential(
    param: &GxtbParams,
    mol: &Molecule,
    shells: &[Shell],
    cn: &[f64],
    q: &[f64],
    gamma2: &[f64],
    tau: &[f64],
) -> Result<Vec<f64>> {
    let n = shells.len();
    let qa = atom_charges(mol.nat(), shells, q);
    let drho = reference_drho(mol, shells);
    let two_over_sqrtpi = 2.0 / std::f64::consts::PI.sqrt();
    // per-atom sums: sum_{l in A} mu_l q_l, and the third-order sum_{l in A} q_l (tau q)_l
    let mut mu = vec![0.0; n];
    let mut muq = vec![0.0; mol.nat()];
    let tq: Vec<f64> = (0..n).map(|i| (0..n).map(|j| tau[i * n + j] * q[j]).sum()).collect();
    let mut qtq = vec![0.0; mol.nat()];
    for (i, &(iat, ish, _)) in shells.iter().enumerate() {
        let el = param.element(mol.numbers[iat])?;
        mu[i] = el.gam1[ish] * (1.0 + el.e1cn * cn[iat].sqrt());
        muq[iat] += mu[i] * q[i];
        qtq[iat] += q[i] * tq[i];
    }
    let mut v = vec![0.0; n];
    for (i, &(iat, _, _)) in shells.iter().enumerate() {
        let x = qa[iat];
        let f = 1.0 + K1_DIS * (erf_f64(x - K1_S) + erf_f64(x + K1_S));
        let df = K1_DIS * two_over_sqrtpi * ((-(x - K1_S) * (x - K1_S)).exp() + (-(x + K1_S) * (x + K1_S)).exp());
        let mut vi = mu[i] * f + muq[iat] * df;
        let gam = GAM3[mol.numbers[iat] - 1];
        let mut v2 = 0.0;
        let mut v3 = 0.0;
        for (j, &(jat, _, _)) in shells.iter().enumerate() {
            if jat != iat {
                vi -= drho[j] * gamma2[i * n + j];
            }
            v2 += gamma2[i * n + j] * q[j];
            let gj = qa[jat] * GAM3[mol.numbers[jat] - 1];
            v3 += gj * q[j] * tau[j * n + i];
        }
        // E3 = 1/3 sum_{l l'} g_l q_l q_l' tau_ll', g_l = q_A(l) Gamma_A(l)
        v3 += qa[iat] * gam * tq[i] + gam * qtq[iat];
        v[i] = vi + v2 + v3 / 3.0 + K4 * q[i] * q[i] * q[i];
    }
    Ok(v)
}

/// Potentials of the anisotropic electrostatics: (v_q, v_d, v_Q) per atom = dE/d(q, d, Q)
/// with the off-diagonal quadrupole components counted twice. Without `full` only the
/// charge-dipole, charge-quadrupole and dipole-dipole parts (the program's `setvsdq_`); with
/// `full` the exact derivative of [`crate::aes::aes_energy`].
#[allow(clippy::type_complexity)]
pub fn aes_potentials(
    mol: &Molecule,
    q: &[f64],
    dip: &[[f64; 3]],
    qp: &[[f64; 6]],
    k: &[Vec<f64>; 4],
    full: bool,
) -> (Vec<f64>, Vec<[f64; 3]>, Vec<[f64; 6]>) {
    let n = mol.nat();
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let mut vs = vec![0.0; n];
    let mut vd = vec![[0.0; 3]; n];
    let mut vq = vec![[0.0; 6]; n];
    for i in 0..n {
        for j in 0..i {
            let ij = i * n + j;
            let (g3, g5, g7, g9) = (k[0][ij], k[1][ij], k[2][ij], k[3][ij]);
            let p = &mol.positions;
            let r = [p[j][0] - p[i][0], p[j][1] - p[i][1], p[j][2] - p[i][2]];
            let r2 = dot(r, r);
            let rr = [r[0] * r[0], 2.0 * r[0] * r[1], r[1] * r[1], 2.0 * r[0] * r[2], 2.0 * r[1] * r[2], r[2] * r[2]];
            let qmat = |q: &[f64; 6]| [[q[0], q[1], q[3]], [q[1], q[2], q[4]], [q[3], q[4], q[5]]];
            let (qi, qj) = (qmat(&qp[i]), qmat(&qp[j]));
            let mv = |m: &[[f64; 3]; 3], v: [f64; 3]| [dot(m[0], v), dot(m[1], v), dot(m[2], v)];
            let (qir, qjr) = (mv(&qi, r), mv(&qj, r));
            let (rqir, rqjr) = (dot(r, qir), dot(r, qjr));
            let (di, dj) = (dip[i], dip[j]);
            let (dir, djr) = (dot(di, r), dot(dj, r));
            // e_qd g3
            vs[j] += dir * g3;
            vs[i] -= djr * g3;
            for x in 0..3 {
                vd[i][x] += q[j] * r[x] * g3;
                vd[j][x] -= q[i] * r[x] * g3;
            }
            // e_qQ g5
            vs[i] += rqjr * g5;
            vs[j] += rqir * g5;
            for c in 0..6 {
                vq[i][c] += q[j] * rr[c] * g5;
                vq[j][c] += q[i] * rr[c] * g5;
            }
            // e_dd g5
            for x in 0..3 {
                vd[i][x] += (dj[x] * r2 - 3.0 * r[x] * djr) * g5;
                vd[j][x] += (di[x] * r2 - 3.0 * r[x] * dir) * g5;
            }
            if !full {
                continue;
            }
            // - e_dQ g7, e_dQ = 5 (rQ_i r)(d_j.r) - 2 (r.Q_i d_j) r^2 - [5 (d_i.r)(rQ_j r) - 2 (r.Q_j d_i) r^2]
            for x in 0..3 {
                vd[j][x] -= (5.0 * rqir * r[x] - 2.0 * r2 * qir[x]) * g7;
                vd[i][x] += (5.0 * r[x] * rqjr - 2.0 * r2 * qjr[x]) * g7;
            }
            let sym = |a: [f64; 3], b: [f64; 3]| {
                [
                    a[0] * b[0],
                    a[0] * b[1] + a[1] * b[0],
                    a[1] * b[1],
                    a[0] * b[2] + a[2] * b[0],
                    a[1] * b[2] + a[2] * b[1],
                    a[2] * b[2],
                ]
            };
            let (rdj, rdi) = (sym(r, dj), sym(r, di));
            for c in 0..6 {
                vq[i][c] -= (5.0 * rr[c] * djr - 2.0 * r2 * rdj[c]) * g7;
                vq[j][c] += (5.0 * rr[c] * dir - 2.0 * r2 * rdi[c]) * g7;
            }
            // 1/3 e_QQ g9, e_QQ = 35 (rQ_i r)(rQ_j r) - 20 (Q_i r).(Q_j r) r^2 + 3 Tr(Q_i Q_j) r^4
            let (rqj, rqi) = (sym(r, qjr), sym(r, qir));
            let w = [1.0, 2.0, 1.0, 2.0, 2.0, 1.0];
            for c in 0..6 {
                vq[i][c] += (35.0 * rr[c] * rqjr - 20.0 * r2 * rqj[c] + 3.0 * r2 * r2 * w[c] * qp[j][c]) * g9 / 3.0;
                vq[j][c] += (35.0 * rr[c] * rqir - 20.0 * r2 * rqi[c] + 3.0 * r2 * r2 * w[c] * qp[i][c]) * g9 / 3.0;
            }
        }
    }
    (vs, vd, vq)
}

/// Spin potentials w_l = dE_spin/dm_l (m_l = alpha - beta shell population).
pub fn spin_shell_potential(param: &GxtbParams, mol: &Molecule, shells: &[Shell], mag: &[f64]) -> Result<Vec<f64>> {
    let mut w = vec![0.0; shells.len()];
    for (i, &(ia, _, li)) in shells.iter().enumerate() {
        let z = mol.numbers[ia];
        let k = param.element(z)?.atspin;
        for (j, &(ja, _, lj)) in shells.iter().enumerate() {
            if ja == ia {
                w[i] += k * SPIN_W0[(z - 1) * 10 + spin_index(li, lj)] * mag[j];
            }
        }
    }
    Ok(w)
}

/// Add -1/2 (v(mu) + v(nu)) S_{mu nu} for an AO potential `v`.
pub fn add_charge_potential(n: usize, s: &[f64], v: &[f64], f: &mut [f64], scale: f64) {
    for i in 0..n {
        for j in 0..n {
            f[i * n + j] -= scale * 0.5 * (v[i] + v[j]) * s[i * n + j];
        }
    }
}

/// Add the multipole potentials: -1/2 [vd(A(mu)).D_{mu nu} + vd(A(nu)).D_{nu mu} + (Q likewise)],
/// with `dpint[x][mu*n + nu]` the dipole integrals around the atom of mu (same for `qpint`).
pub fn add_multipole_potential(
    n: usize,
    ao_atom: &[usize],
    dpint: &[Vec<f64>; 3],
    qpint: &[Vec<f64>; 6],
    vd: &[[f64; 3]],
    vq: &[[f64; 6]],
    f: &mut [f64],
) {
    for i in 0..n {
        let a = ao_atom[i];
        for j in 0..n {
            let b = ao_atom[j];
            let (ij, ji) = (i * n + j, j * n + i);
            let mut t = 0.0;
            for x in 0..3 {
                t += vd[a][x] * dpint[x][ij] + vd[b][x] * dpint[x][ji];
            }
            for c in 0..6 {
                t += vq[a][c] * qpint[c][ij] + vq[b][c] * qpint[c][ji];
            }
            f[ij] -= 0.5 * t;
        }
    }
}

/// Mulliken-exchange Fock matrix of one spin density `p` (closed shell: total density with
/// `closed = true`).
pub fn mfx_fock(n: usize, s: &[f64], p: &[f64], gamma: &[f64], closed: bool, par: Par) -> Vec<f64> {
    let sp = matmul(n, s, p, par);
    let sps = matmul(n, &sp, s, par);
    let mut gsp = vec![0.0; n * n];
    let mut gp = vec![0.0; n * n];
    for k in 0..n * n {
        gsp[k] = gamma[k] * sp[k];
        gp[k] = gamma[k] * p[k];
    }
    let a = matmul(n, &gsp, s, par); // (g o SP) S; its transpose is S (g o PS)
    let b = matmul(n, &matmul(n, s, &gp, par), s, par);
    let scale = if closed { -0.125 } else { -0.25 };
    let mut f = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let ij = i * n + j;
            f[ij] = scale * (a[ij] + a[j * n + i] + gamma[ij] * sps[ij] + b[ij]);
        }
    }
    f
}

/// Onsite-exchange Fock matrix of one spin density `p` (closed shell: total density with
/// `closed = true`); `ao` lists (atom, shell, l) per AO.
/// `exact = false` applies the population part off the diagonal at twice its derivative, as
/// the program does; `exact = true` gives the exact derivative of the OFX energy.
pub fn ofx_fock(
    numbers: &[usize],
    ao: &[Shell],
    s: &[f64],
    p: &[f64],
    closed: bool,
    exact: bool,
    par: Par,
) -> Vec<f64> {
    let n = ao.len();
    let scale = if closed { 0.5 } else { 1.0 };
    let ps = matmul(n, p, s, par);
    let mut pt = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            if ao[i].0 == ao[j].0 {
                pt[i * n + j] = 0.5 * scale * (ps[i * n + j] + ps[j * n + i]);
            }
        }
    }
    let mut y = vec![0.0; n * n];
    // diagonal (population) part of Y, applied onsite only
    let mut ydiag = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            let (ai, si, li) = ao[i];
            let (aj, sj, lj) = ao[j];
            if ai != aj {
                continue;
            }
            let Some(k) = gmunu_index(li, lj) else { continue };
            let lam = OFX_SCALE * GMUNU[(k - 1) + 9 * (numbers[ai] - 1)];
            if i != j {
                y[i * n + j] -= 2.0 * lam * pt[i * n + j];
            }
            if si == sj && li >= 1 {
                let f = (if i == j { (2 * li + 1) as f64 } else { 0.0 } - 1.0) / (2 * li + 1) as f64;
                ydiag[i] -= 2.0 * f * lam * pt[j * n + j];
            }
        }
    }
    let ys = matmul(n, &y, s, par);
    let mut f = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            // the program applies the population (diagonal) part without the factor 1/2 off
            // the diagonal: (y_i + y_j) S_ij, i != j (twice the derivative)
            let off = if exact { 0.5 } else { 1.0 };
            let d = if i == j { ydiag[i] } else { off * (ydiag[i] + ydiag[j]) * s[i * n + j] };
            f[i * n + j] = 0.5 * (ys[i * n + j] + ys[j * n + i]) + d;
        }
    }
    f
}
