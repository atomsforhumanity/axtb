//! Anisotropic electrostatics of g-xTB v1.1 (P-Eqs. 32-33), as implemented by the 1.1.0
//! program (`xtb_aespot::mmompop`, `mmomgaberf`, `aniso_electro`), verified at the program's
//! converged density (reference tests).
//!
//! - Atomic moments (Mulliken-partitioned, GFN2-xTB style): for every AO mu on atom A,
//!   d_A -= sum_nu P_{mu nu} <mu|(r - R_A)|nu>, and the traceless quadrupole
//!   Q_A -= sum_nu P_{mu nu} [3/2 <mu|(r-R_A)(r-R_A)|nu> - 1/2 <mu|(r-R_A)^2|nu> 1]
//!   (components xx, xy, yy, xz, yz, zz; total density).
//! - Damped kernels for every atom pair (R in bohr, R0_AB the D3 pair radius in Ångström,
//!   pairs beyond R = 200 bohr skipped):
//!   g3 = k3/2 [1 - erf(-0.45 (R - R0))]/R^3,
//!   g_n = k_n/2 [1 - erf(-1.05 (R - R0))]/R^n for n = 5, 7, 9,
//!   k3 = glob(16), k5 = glob(17), k7 = glob(18), k9 = -0.003.
//! - Energy (pairs A > B, r = R_B - R_A):
//!   E = sum [ e_qd g3 + (e_qQ + e_dd) g5 - e_dQ g7 + 1/3 e_QQ g9 ],
//!   e_qd = q_B (d_A.r) - q_A (d_B.r), e_qQ = q_B r^T Q_A r + q_A r^T Q_B r,
//!   e_dd = (d_A.d_B) r^2 - 3 (d_A.r)(d_B.r),
//!   e_dQ = 5 (r^T Q_A r)(d_B.r) - 2 (r^T Q_A d_B) r^2 - [5 (d_A.r)(r^T Q_B r) - 2 (r^T Q_B d_A) r^2],
//!   e_QQ = 35 (r^T Q_A r)(r^T Q_B r) - 20 (Q_A r).(Q_B r) r^2 + 3 Tr(Q_A Q_B) r^4,
//!   with the atomic Mulliken charges q (no onsite terms).

use crate::eeqbc::{erf_f64, vdw_rad_pair};
use crate::params::GxtbParams;
use axtb::basis::{ao_count, ao_offsets, Cgto};
use axtb::integrals::{multipole, Scratch, MBLK};
use axtb::molecule::{norm2_sq, sub3};
use axtb::Molecule;

/// Primitive screening of the multipole integrals (as for the SCF overlap).
pub const INTCUT: f64 = 25.0;
/// Erf steepness of the q-d kernel and of the higher kernels.
pub const KDMP3: f64 = 0.45;
pub const KDMP5: f64 = 1.05;
/// Prefactor of the quadrupole-quadrupole kernel.
pub const K9: f64 = -0.003;
/// Pair distance beyond which the kernels are zero (bohr).
pub const R_MAX: f64 = 200.0;

/// Atomic dipoles (3 per atom) and traceless quadrupoles (6 per atom) from the spin-summed
/// density `p` (full nao x nao in the AO order of `shells`).
pub fn atomic_moments(mol: &Molecule, shells: &[Vec<Cgto>], p: &[f64]) -> (Vec<[f64; 3]>, Vec<[f64; 6]>) {
    let nat = mol.nat();
    let first = ao_offsets(shells);
    let nao = ao_count(shells);
    let mut dip = vec![[0.0; 3]; nat];
    let mut qp = vec![[0.0; 6]; nat];
    let mut s = [0.0; MBLK];
    let mut d = [[0.0; 3]; MBLK];
    let mut q = [[0.0; 6]; MBLK];
    let mut scratch = Scratch::default();
    for ia in 0..nat {
        for ib in 0..nat {
            // moments around R_A (ket = shell on A)
            let vec = sub3(mol.positions[ia], mol.positions[ib]);
            for (isa, sa) in shells[ia].iter().enumerate() {
                for (isb, sb) in shells[ib].iter().enumerate() {
                    multipole(sb, sa, vec, INTCUT, &mut s, &mut d, &mut q, &mut scratch);
                    let (na, nb) = (2 * sa.l + 1, 2 * sb.l + 1);
                    for a in 0..na {
                        let mu = first[ia][isa] + a;
                        for b in 0..nb {
                            let nu = first[ib][isb] + b;
                            let pmn = p[mu * nao + nu];
                            let k = b + nb * a;
                            for x in 0..3 {
                                dip[ia][x] -= pmn * d[k][x];
                            }
                            for x in 0..6 {
                                qp[ia][x] -= pmn * q[k][x];
                            }
                        }
                    }
                }
            }
        }
    }
    (dip, qp)
}

/// Damped kernels g3, g5, g7, g9 (full nat x nat, zero diagonal).
pub fn aes_kernels(param: &GxtbParams, mol: &Molecule) -> [Vec<f64>; 4] {
    let n = mol.nat();
    let g = &param.glob;
    let mut out = [vec![0.0; n * n], vec![0.0; n * n], vec![0.0; n * n], vec![0.0; n * n]];
    for i in 0..n {
        for j in 0..i {
            let r = norm2_sq(sub3(mol.positions[i], mol.positions[j])).sqrt();
            if r > R_MAX {
                continue;
            }
            let r0 = vdw_rad_pair(mol.numbers[i], mol.numbers[j]);
            let (r2, r3) = (r * r, r * r * r);
            let (r5, r7, r9) = (r2 * r3, r2 * r2 * r3, r2 * r2 * r2 * r3);
            let f3 = (1.0 - erf_f64(-KDMP3 * (r - r0))) * (0.5 * g[15]);
            let f5 = (1.0 - erf_f64(-KDMP5 * (r - r0))) * 0.5;
            let v = [f3 / r3, g[16] * f5 / r5, g[17] * f5 / r7, f5 * K9 / r9];
            for (k, val) in v.iter().enumerate() {
                out[k][i * n + j] = *val;
                out[k][j * n + i] = *val;
            }
        }
    }
    out
}

fn qmat(q: &[f64; 6]) -> [[f64; 3]; 3] {
    [[q[0], q[1], q[3]], [q[1], q[2], q[4]], [q[3], q[4], q[5]]]
}

/// AES energy from atomic charges, moments and kernels.
pub fn aes_energy(mol: &Molecule, q: &[f64], dip: &[[f64; 3]], qp: &[[f64; 6]], k: &[Vec<f64>; 4]) -> f64 {
    aes_energy_rows(mol, q, dip, qp, k, 0..mol.nat())
}

/// The pair terms (i, j < i) of [`aes_energy`] for the atoms i in `rows`.
pub fn aes_energy_rows(
    mol: &Molecule,
    q: &[f64],
    dip: &[[f64; 3]],
    qp: &[[f64; 6]],
    k: &[Vec<f64>; 4],
    rows: std::ops::Range<usize>,
) -> f64 {
    let n = mol.nat();
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let mv = |m: &[[f64; 3]; 3], v: [f64; 3]| [dot(m[0], v), dot(m[1], v), dot(m[2], v)];
    let mut e = 0.0;
    for i in rows {
        let qi = qmat(&qp[i]);
        for j in 0..i {
            let qj = qmat(&qp[j]);
            let r = sub3(mol.positions[j], mol.positions[i]);
            let r2 = dot(r, r);
            let (di, dj) = (dip[i], dip[j]);
            let (qir, qjr) = (mv(&qi, r), mv(&qj, r));
            let (rqir, rqjr) = (dot(r, qir), dot(r, qjr));
            let e_qd = q[j] * dot(di, r) - q[i] * dot(dj, r);
            let e_qq = q[j] * rqir + q[i] * rqjr;
            let e_dd = dot(dj, di) * r2 - 3.0 * dot(dj, r) * dot(di, r);
            let e_dq = (5.0 * rqir * dot(dj, r) - 2.0 * dot(r, mv(&qi, dj)) * r2)
                - (5.0 * dot(di, r) * rqjr - 2.0 * dot(r, mv(&qj, di)) * r2);
            let mut tr = 0.0;
            for a in 0..3 {
                for b in 0..3 {
                    tr += qi[a][b] * qj[a][b];
                }
            }
            let e_q2 = 35.0 * rqir * rqjr - 20.0 * dot(qir, qjr) * r2 + 3.0 * tr * r2 * r2;
            let ij = i * n + j;
            e += e_qd * k[0][ij] + (e_qq + e_dd) * k[1][ij] - e_dq * k[2][ij] + e_q2 * k[3][ij] / 3.0;
        }
    }
    e
}
