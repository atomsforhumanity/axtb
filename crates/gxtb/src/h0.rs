//! Extended-Hückel Hamiltonian H0 of g-xTB v1.1 (P-Eqs. 18-21), decoded from the 1.1.0
//! program (`scf_`, `modbas_`, `sintm_diat_`) and verified element by element against the
//! matrices it builds (reference tests).
//!
//! - **H0 basis** (`modbas_`): per atom the q-vSZP shells with exponents scaled by
//!   `pln(l)` and contraction coefficients c0 + c1 q'_eff, where
//!   q'_eff = k̃0 (q - k_q2 q^2) + k̃2 k_cn sqrt(CN) + k̃3 k_qcn q CN, i.e. the three terms
//!   of the SCF-basis effective charge reweighted by `kqvszp` = (k̃0, k̃2, k̃3).
//! - **Scaled overlap** (`sintm_diat_`): for every atom pair A != B the overlap block in the
//!   H0 basis is rotated to the diatomic frame (bond along z), its sigma (m = 0), pi (|m| = 1)
//!   and delta (|m| = 2) parts are scaled by k_sigma = 2/(1/h0k1_A + 1/h0k1_B),
//!   k_pi = 2/(1/h0k2_A + 1/h0k2_B), k_delta = 5, and rotated back. Blocks with f functions
//!   stay unscaled (as in the program's `scale_diatomic_frame`; not exercised by the reference
//!   set).
//! - **Levels**: H_i = -(lev_l - lev_cn_l CN_A) for AO i in shell l of atom A (g-xTB CN).
//! - **Off-diagonal**: H_ij = 1/4 (H_i + H_j)(K_li + K_lj) pi_i pi_j S~_ij with
//!   pi_i = 1 + poly_A F_li R_AB / Rc_AB, Rc_AB = (rcov_A + rcov_B)/2 (`com::rcov`, bohr),
//!   K = (glob(1), glob(2), 1.4, 2.85) and F = (1, glob(3), 2, 2) for l = s, p, d, f
//!   (`scf_$KLL`, `scf_$RFL`); same-atom off-diagonal elements are zero (orthogonal AOs).

use crate::data::RCOV_COM;
use crate::params::{GxtbParams, QvszpParams};
use crate::qvszp::{EffectiveCharge, QvszpBasis};
use axtb::basis::{Cgto, MAXPRIM};
use axtb::integrals::{overlap, MBLK};
use axtb::molecule::{norm2_sq, sub3};
use axtb::{Error, Molecule, Result};
use std::f64::consts::PI;

/// Wolfsberg-type shell factors K_l (s, p, d, f) of the off-diagonal H0 elements.
pub fn k_wolfsberg(param: &GxtbParams) -> [f64; 4] {
    [param.glob[0], param.glob[1], 1.4, 2.85]
}

/// Shell factors F_l (s, p, d, f) of the distance polynomial.
pub fn k_poly(param: &GxtbParams) -> [f64; 4] {
    [1.0, param.glob[2], 2.0, 2.0]
}

/// Primitive-pair screening of the program's overlap code (`xtb_intgrad::get_overlap`, as
/// called from `sintm_diat_`): pairs with a_i a_j R^2/(a_i + a_j) > 20 are skipped.
pub const INTCUT: f64 = 20.0;

/// Delta-bond scaling of the diatomic-frame overlap.
pub const K_DELTA: f64 = 5.0;

const DFACTORIAL: [f64; 6] = [1.0, 1.0, 3.0, 15.0, 105.0, 945.0];

/// Effective charge of the H0 basis.
pub fn h0_effective_charge(param: &GxtbParams, z: usize, e: &EffectiveCharge) -> Result<f64> {
    let k = param.element(z)?.kqvszp;
    Ok(k[0] * e.q_term + k[1] * e.cn_term + k[2] * e.qcn_term)
}

/// Normalized shells of the H0 basis of one atom.
pub fn h0_atom_shells(param: &GxtbParams, basis: &QvszpParams, z: usize, qeff_h0: f64) -> Result<Vec<Cgto>> {
    let el = basis.element(z)?;
    let pln = param.element(z)?.pln;
    el.shells
        .iter()
        .enumerate()
        .map(|(ish, sh)| {
            let np = sh.alpha.len();
            if np > MAXPRIM {
                return Err(Error::Input(format!("H0 basis: {np} primitives exceed MAXPRIM")));
            }
            let l = sh.l;
            let alpha: Vec<f64> = sh.alpha.iter().map(|a| pln[ish] * a).collect();
            let raw: Vec<f64> = (0..np).map(|i| sh.c0[i] + sh.c1[i] * qeff_h0).collect();
            let mut s = 0.0;
            for i in 0..np {
                for j in 0..np {
                    let (a, b) = (alpha[i], alpha[j]);
                    s += raw[i] * raw[j] * (2.0 * (a * b).sqrt() / (a + b)).powf(l as f64 + 1.5);
                }
            }
            let scale = 1.0 / s.sqrt();
            let mut cgto = Cgto { l, nprim: np, alpha: [0.0; MAXPRIM], coeff: [0.0; MAXPRIM] };
            for (i, (&a, &c)) in alpha.iter().zip(&raw).enumerate() {
                let norm = (2.0 * a / PI).powf(0.75) * (4.0 * a).sqrt().powi(l as i32) / DFACTORIAL[l].sqrt();
                cgto.alpha[i] = a;
                cgto.coeff[i] = c * norm * scale;
            }
            Ok(cgto)
        })
        .collect()
}

/// d/dq'_eff of the shells of [`h0_atom_shells`].
pub fn h0_atom_shells_dq(param: &GxtbParams, basis: &QvszpParams, z: usize, qeff_h0: f64) -> Result<Vec<Cgto>> {
    let el = basis.element(z)?;
    let pln = param.element(z)?.pln;
    Ok(el
        .shells
        .iter()
        .enumerate()
        .map(|(ish, sh)| {
            let alpha: Vec<f64> = sh.alpha.iter().map(|a| pln[ish] * a).collect();
            crate::qvszp::shell_q_derivative(sh.l, &alpha, &sh.c0, &sh.c1, qeff_h0)
        })
        .collect())
}

/// Projectors onto the bond-frame |m| = 0, 1, 2 subspaces of a shell with angular momentum
/// `l` <= 2, in axtb's real spherical-harmonic order, for bond direction `n` (unit vector).
/// Returned as `proj[mu][a * nsph + b]`.
fn bond_projectors(l: usize, n: [f64; 3]) -> [Vec<f64>; 3] {
    let ns = 2 * l + 1;
    let mut proj = [vec![0.0; ns * ns], vec![0.0; ns * ns], vec![0.0; ns * ns]];
    if l == 0 {
        proj[0][0] = 1.0;
        return proj;
    }
    // orthonormal frame (e1, e2, n)
    let t = if n[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let d = t[0] * n[0] + t[1] * n[1] + t[2] * n[2];
    let mut e1 = [t[0] - d * n[0], t[1] - d * n[1], t[2] - d * n[2]];
    let ne = norm2_sq(e1).sqrt();
    e1 = [e1[0] / ne, e1[1] / ne, e1[2] / ne];
    let e2 = [n[1] * e1[2] - n[2] * e1[1], n[2] * e1[0] - n[0] * e1[2], n[0] * e1[1] - n[1] * e1[0]];
    let frame = [e1, e2, n]; // bond-frame x, y, z axes in lab coordinates
                             // u[m_bond][m_lab]: lab components of the bond-frame functions
    let mut u = vec![0.0; ns * ns];
    let mu_of: Vec<usize>;
    match l {
        1 => {
            // axtb order y, z, x -> axis index 1, 2, 0
            let axis = [1usize, 2, 0];
            for (mb, &ab) in axis.iter().enumerate() {
                for (ml, &al) in axis.iter().enumerate() {
                    u[mb * ns + ml] = frame[ab][al];
                }
            }
            mu_of = vec![1, 0, 1];
        }
        2 => {
            let s3 = 3f64.sqrt() / 2.0;
            // quadratic forms of the axtb d functions (m = -2..2), Frobenius norm^2 = 3/2
            let mut q = [[[0.0; 3]; 3]; 5];
            q[0][0][1] = s3;
            q[0][1][0] = s3;
            q[1][1][2] = s3;
            q[1][2][1] = s3;
            q[2][0][0] = -0.5;
            q[2][1][1] = -0.5;
            q[2][2][2] = 1.0;
            q[3][0][2] = s3;
            q[3][2][0] = s3;
            q[4][0][0] = s3;
            q[4][1][1] = -s3;
            for mb in 0..5 {
                // bond-frame form in lab coordinates: sum_ab q[a][b] frame[a] (x) frame[b]
                let mut lab = [[0.0; 3]; 3];
                for a in 0..3 {
                    for b in 0..3 {
                        if q[mb][a][b] != 0.0 {
                            for x in 0..3 {
                                for y in 0..3 {
                                    lab[x][y] += q[mb][a][b] * frame[a][x] * frame[b][y];
                                }
                            }
                        }
                    }
                }
                for ml in 0..5 {
                    let mut dot = 0.0;
                    for x in 0..3 {
                        for y in 0..3 {
                            dot += q[ml][x][y] * lab[x][y];
                        }
                    }
                    u[mb * ns + ml] = dot / 1.5;
                }
            }
            mu_of = vec![2, 1, 0, 1, 2];
        }
        _ => unreachable!(),
    }
    for (mb, &mu) in mu_of.iter().enumerate() {
        for a in 0..ns {
            for b in 0..ns {
                proj[mu][a * ns + b] += u[mb * ns + a] * u[mb * ns + b];
            }
        }
    }
    proj
}

/// Quadratic forms of axtb's real d functions (m = -2..2: xy, yz, z2, xz, x2-y2), Frobenius
/// norm^2 3/2.
fn d_forms() -> [[[f64; 3]; 3]; 5] {
    let s3 = 3f64.sqrt() / 2.0;
    let mut q = [[[0.0; 3]; 3]; 5];
    q[0][0][1] = s3;
    q[0][1][0] = s3;
    q[1][1][2] = s3;
    q[1][2][1] = s3;
    q[2][0][0] = -0.5;
    q[2][1][1] = -0.5;
    q[2][2][2] = 1.0;
    q[3][0][2] = s3;
    q[3][2][0] = s3;
    q[4][0][0] = s3;
    q[4][1][1] = -s3;
    q
}

/// Bond-frame projectors P_mu (|m| = mu = 0, 1, 2) of a shell with l <= 2 for the unit bond
/// vector `n` in closed form, and their derivatives dP_mu/dn_k (`n` treated as a free vector;
/// contract with the tangential dn). Row-major `(2l+1) x (2l+1)`, axtb order.
/// p: P_0 = n n^T, P_1 = 1 - P_0. d (forms q_m): P_0 = a a^T with a_m = n^T q_m n,
/// (P_1)_{mm'} = (2 (q_m n).(q_m' n) - 2 a_m a_m')/(3/2), P_2 = 1 - P_0 - P_1.
#[allow(clippy::type_complexity)]
pub fn bond_projectors_deriv(l: usize, n: [f64; 3]) -> ([Vec<f64>; 3], [[Vec<f64>; 3]; 3]) {
    let ns = 2 * l + 1;
    let z = || vec![0.0; ns * ns];
    let mut p = [z(), z(), z()];
    let mut dp = [[z(), z(), z()], [z(), z(), z()], [z(), z(), z()]];
    match l {
        0 => p[0][0] = 1.0,
        1 => {
            let ax = [1usize, 2, 0];
            for a in 0..3 {
                for b in 0..3 {
                    let v = n[ax[a]] * n[ax[b]];
                    p[0][a * 3 + b] = v;
                    p[1][a * 3 + b] = if a == b { 1.0 } else { 0.0 } - v;
                    for k in 0..3 {
                        let d = if ax[a] == k { n[ax[b]] } else { 0.0 } + if ax[b] == k { n[ax[a]] } else { 0.0 };
                        dp[0][k][a * 3 + b] = d;
                        dp[1][k][a * 3 + b] = -d;
                    }
                }
            }
        }
        2 => {
            let q = d_forms();
            let mv = |m: &[[f64; 3]; 3], v: [f64; 3]| -> [f64; 3] {
                [0, 1, 2].map(|x| m[x][0] * v[0] + m[x][1] * v[1] + m[x][2] * v[2])
            };
            let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
            let b: Vec<[f64; 3]> = (0..5).map(|m| mv(&q[m], n)).collect();
            let a: Vec<f64> = (0..5).map(|m| dot(n, b[m])).collect();
            // q_m b_m' (for the derivative of b_m . b_m')
            let qb: Vec<Vec<[f64; 3]>> = (0..5).map(|m| (0..5).map(|mp| mv(&q[m], b[mp])).collect()).collect();
            for m in 0..5 {
                for mp in 0..5 {
                    let k0 = a[m] * a[mp];
                    let k1 = (2.0 * dot(b[m], b[mp]) - 2.0 * k0) / 1.5;
                    let idx = m * 5 + mp;
                    p[0][idx] = k0;
                    p[1][idx] = k1;
                    p[2][idx] = if m == mp { 1.0 } else { 0.0 } - k0 - k1;
                    for k in 0..3 {
                        let (da, dap) = (2.0 * b[m][k], 2.0 * b[mp][k]);
                        let d0 = da * a[mp] + a[m] * dap;
                        let d1 = (2.0 * (qb[m][mp][k] + qb[mp][m][k]) - 2.0 * d0) / 1.5;
                        dp[0][k][idx] = d0;
                        dp[1][k][idx] = d1;
                        dp[2][k][idx] = -d0 - d1;
                    }
                }
            }
        }
        _ => panic!("bond projectors for l = {l}"),
    }
    (p, dp)
}

/// Scale an overlap block `s` (rows: shell of angular momentum `la` on A, columns: `lb` on
/// B, row-major) in the diatomic frame of the bond A -> B.
pub fn scale_diatomic_block(la: usize, lb: usize, n: [f64; 3], k: [f64; 3], s: &mut [f64]) {
    if la > 2 || lb > 2 {
        return;
    }
    let (na, nb) = (2 * la + 1, 2 * lb + 1);
    let pa = bond_projectors(la, n);
    let pb = bond_projectors(lb, n);
    let mut out = vec![0.0; na * nb];
    for mu in 0..=la.min(lb) {
        // out += k_mu * Pa S Pb
        let mut tmp = vec![0.0; na * nb];
        for i in 0..na {
            for j in 0..nb {
                tmp[i * nb + j] = (0..na).map(|a| pa[mu][i * na + a] * s[a * nb + j]).sum();
            }
        }
        for i in 0..na {
            for j in 0..nb {
                out[i * nb + j] += k[mu] * (0..nb).map(|b| tmp[i * nb + b] * pb[mu][b * nb + j]).sum::<f64>();
            }
        }
    }
    s.copy_from_slice(&out);
}

/// H0 of a molecule and the scaled overlap it is built from (full nao x nao matrices in
/// axtb AO order: atom, shell, spherical component m = -l..l).
#[derive(Clone, Debug)]
pub struct H0 {
    pub nao: usize,
    /// AO -> (atom, shell, l).
    pub ao: Vec<(usize, usize, usize)>,
    /// Diatomic-frame scaled overlap in the H0 basis (unit diagonal).
    pub stilde: Vec<f64>,
    /// Hamiltonian H0.
    pub h: Vec<f64>,
    /// AO levels H_i (diagonal of H0 is -... see module docs; stored as H0_ii).
    pub level: Vec<f64>,
}

/// Build H0 from the q-vSZP basis data (`QvszpBasis`, for the effective-charge terms) and the
/// g-xTB coordination numbers `cn` (not square-rooted).
pub fn build_h0(param: &GxtbParams, basis: &QvszpParams, mol: &Molecule, qb: &QvszpBasis, cn: &[f64]) -> Result<H0> {
    let nat = mol.nat();
    let mut shells = Vec::with_capacity(nat);
    for i in 0..nat {
        let q = h0_effective_charge(param, mol.numbers[i], &qb.qeff[i])?;
        shells.push(h0_atom_shells(param, basis, mol.numbers[i], q)?);
    }
    let kw = k_wolfsberg(param);
    let kp = k_poly(param);
    let mut ao = Vec::new();
    let mut first = Vec::with_capacity(nat);
    for (iat, sh) in shells.iter().enumerate() {
        let mut v = Vec::new();
        for (ish, s) in sh.iter().enumerate() {
            v.push(ao.len());
            for _ in 0..2 * s.l + 1 {
                ao.push((iat, ish, s.l));
            }
        }
        first.push(v);
    }
    let nao = ao.len();
    let mut level = vec![0.0; nao];
    for (k, &(iat, ish, _)) in ao.iter().enumerate() {
        let e = param.element(mol.numbers[iat])?;
        level[k] = -(e.lev[ish] - e.lev_cn[ish] * cn[iat]);
    }
    let mut stilde = vec![0.0; nao * nao];
    let mut h = vec![0.0; nao * nao];
    for k in 0..nao {
        stilde[k * nao + k] = 1.0;
        h[k * nao + k] = level[k];
    }
    let mut blk = [0.0; MBLK];
    for ia in 0..nat {
        let za = mol.numbers[ia];
        let ea = param.element(za)?;
        for ib in 0..ia {
            let zb = mol.numbers[ib];
            let eb = param.element(zb)?;
            let rab = sub3(mol.positions[ib], mol.positions[ia]);
            let r = norm2_sq(rab).sqrt();
            let n = [rab[0] / r, rab[1] / r, rab[2] / r];
            let ksig = 2.0 / (1.0 / ea.h0k[0] + 1.0 / eb.h0k[0]);
            let kpi = 2.0 / (1.0 / ea.h0k[1] + 1.0 / eb.h0k[1]);
            let rc = 0.5 * (RCOV_COM[za - 1] + RCOV_COM[zb - 1]);
            for (isa, sa) in shells[ia].iter().enumerate() {
                for (isb, sb) in shells[ib].iter().enumerate() {
                    // block rows: shell on A, columns: shell on B
                    let vec = sub3(mol.positions[ia], mol.positions[ib]);
                    overlap(sb, sa, vec, INTCUT, &mut blk);
                    let (na, nb) = (2 * sa.l + 1, 2 * sb.l + 1);
                    let mut s: Vec<f64> = (0..na * nb).map(|x| blk[(x % nb) + nb * (x / nb)]).collect();
                    scale_diatomic_block(sa.l, sb.l, n, [ksig, kpi, K_DELTA], &mut s);
                    let pia = 1.0 + ea.poly * kp[sa.l] * r / rc;
                    let pib = 1.0 + eb.poly * kp[sb.l] * r / rc;
                    for a in 0..na {
                        for b in 0..nb {
                            let i = first[ia][isa] + a;
                            let j = first[ib][isb] + b;
                            let st = s[a * nb + b];
                            stilde[i * nao + j] = st;
                            stilde[j * nao + i] = st;
                            let hij = 0.25 * (level[i] + level[j]) * (kw[sa.l] + kw[sb.l]) * pia * pib * st;
                            h[i * nao + j] = hij;
                            h[j * nao + i] = hij;
                        }
                    }
                }
            }
        }
    }
    Ok(H0 { nao, ao, stilde, h, level })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn closed_form_projectors_match_frame_projectors() {
        for n in [[0.0, 0.0, 1.0], [0.6, -0.48, 0.64], [-0.36, 0.48, 0.8], [1.0, 0.0, 0.0]] {
            for l in 0..=2 {
                let a = bond_projectors(l, n);
                let (b, db) = bond_projectors_deriv(l, n);
                for mu in 0..=l {
                    for k in 0..a[mu].len() {
                        assert!((a[mu][k] - b[mu][k]).abs() < 1e-14, "l={l} mu={mu}");
                    }
                }
                // derivatives against central differences along a tangent direction
                let t = [n[1] - n[2], n[2] - n[0], n[0] - n[1]];
                let h = 1e-6;
                let move_n = |s: f64| {
                    let v = [n[0] + s * t[0], n[1] + s * t[1], n[2] + s * t[2]];
                    let r = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                    [v[0] / r, v[1] / r, v[2] / r]
                };
                let (pp, _) = bond_projectors_deriv(l, move_n(h));
                let (pm, _) = bond_projectors_deriv(l, move_n(-h));
                let tdot = t[0] * n[0] + t[1] * n[1] + t[2] * n[2];
                let tt = [t[0] - tdot * n[0], t[1] - tdot * n[1], t[2] - tdot * n[2]];
                for mu in 0..=l {
                    for k in 0..pp[mu].len() {
                        let fd = (pp[mu][k] - pm[mu][k]) / (2.0 * h);
                        let an: f64 = (0..3).map(|c| db[mu][c][k] * tt[c]).sum();
                        assert!((fd - an).abs() < 1e-8, "l={l} mu={mu} {fd} {an}");
                    }
                }
            }
        }
    }
}
