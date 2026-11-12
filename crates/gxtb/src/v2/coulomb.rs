//! Charge- and multipole-dependent electrostatics and spin polarization of g-xTB 2.0.1
//! (spec §4.7-§4.12). Shell charges q_l = n0_l - Mulliken population, atomic charges
//! q_A = sum_l q_l, f(CN) the smoothed square root of the g-xTB CN.
//!
//! - First order: E1 = sum_l mu_l [1 + k_A f(CN_A)] g(q_A) q_l,
//!   g(q) = 1/2 {2 + 0.025 [erf(q - 0.666666666) + erf(q + 0.666666666)]}.
//! - Second order: E2 = 1/2 sum q_i gamma_ij q_j + sum_i q_i sum_{j not on A(i)} gamma_ij r_j,
//!   gamma_ij = 1/(R + exp(-0.294621155 R)/eta_ij), eta_ij = 2/(1/U_i + 1/U_j),
//!   U_l = `ps_tb2_shell_hubbard` `pa_hubbard_parameter` [1 + `pa_tb2_hubbard_cn` f(CN)],
//!   r_l = n0_l - Z_l.
//! - Third order: E3 = 1/3 sum_i q_i^2 Gamma_i sum_j tau_ij q_j, Gamma_l = `pa_tb3_hubbard_derivs`
//!   k_l, tau = -0.2093327496 eta^2 within an atom, 2.3 x (1 - 1.3 x/2) exp(-1.3 x) with
//!   x = R/eta between atoms.
//! - Fourth order: E4 = 1/24 sum_l 0.036 k_l q_l^4.
//! - Anisotropic electrostatics: the v1.1 pair formula ([`crate::aes::aes_energy`]) with
//!   kernels g_n = 1/2 k_n [1 - erf(-s_n (R - R̄_AB))]/R^n and atomic dipoles scaled by
//!   c_A = 1 + `pa_aes_dip_scale` f(CN_A).
//! - Spin: E = 1/2 sum_A sum_{l,l' in A} m_l W_ll' m_l', W = `pa_wll_scale` W0.
//!
//! All potentials are exact derivatives (the program's Fock matrix is exact for these terms).

use super::{sqrt_cn, Model};
use crate::data_v2::{
    AES_DIP_SCALE, HUBBARD, HUBBARD_CN, HUBBARD_DERIV, IPEA, IPEA_CN, PG_TB3_KSHELL, PG_TB4_KSHELL, REF_OCC,
    SHELL_HUBBARD, SPIN_W0, WLL_SCALE, ZEFF_SHELL,
};
use crate::eeqbc::erf_f64;
use crate::es::spin_index;
use axtb::molecule::{norm2_sq, sub3};
use axtb::Molecule;

/// Shell list entry: (atom, shell index on the atom, l).
pub type Shell = (usize, usize, usize);

pub const K1_DIS: f64 = 0.025;
pub const K1_S: f64 = 0.666666666;
pub const K2_EXP: f64 = 0.294621155;
pub const K3_ONSITE: f64 = 0.2093327496;
pub const K3_A: f64 = 1.3;
pub const K3_B: f64 = 2.3;
pub const K4: f64 = 0.036;
/// AES kernel prefactors and steepnesses for n = 3, 5, 7, 9.
pub const AES_K: [f64; 4] = [0.3405910191, 0.1691310614, 0.074034339, -0.02];
pub const AES_S: [f64; 4] = [0.5, 1.0, 1.0, 1.0];
const SQRTPI: f64 = 1.772_453_850_905_516;

/// First-order charge function g(q) and its derivative.
#[inline]
pub fn first_order_g(q: f64) -> (f64, f64) {
    let (a, b) = (q - K1_S, q + K1_S);
    (0.5 * (2.0 + K1_DIS * (erf_f64(a) + erf_f64(b))), 0.5 * K1_DIS * 2.0 / SQRTPI * ((-a * a).exp() + (-b * b).exp()))
}

/// Second derivative of the first-order charge function g(q).
#[inline]
pub fn first_order_g2(q: f64) -> f64 {
    let (a, b) = (q - K1_S, q + K1_S);
    -0.5 * K1_DIS * 4.0 / SQRTPI * (a * (-a * a).exp() + b * (-b * b).exp())
}

/// gamma(R, eta) and its derivatives with respect to R and eta.
#[inline]
pub fn gamma_kernel(r: f64, eta: f64) -> (f64, f64, f64) {
    let ex = (-K2_EXP * r).exp();
    let den = r + ex / eta;
    let g = 1.0 / den;
    (g, -(1.0 - K2_EXP * ex / eta) * g * g, ex / (eta * eta) * g * g)
}

/// tau(R, eta) between atoms (`same = false`) or within an atom, and its derivatives with
/// respect to R and eta.
#[inline]
pub fn tau_kernel(r: f64, eta: f64, same: bool) -> (f64, f64, f64) {
    if same {
        (-K3_ONSITE * eta * eta, 0.0, -2.0 * K3_ONSITE * eta)
    } else {
        let x = r / eta;
        let e = (-K3_A * x).exp();
        let t = K3_B * x * (1.0 - K3_A * x / 2.0) * e;
        // dt/dx = 2.3 e [1 - 1.3 x - 1.3 x (1 - 1.3 x/2)]
        let dtdx = K3_B * e * (1.0 - K3_A * x - K3_A * x * (1.0 - K3_A * x / 2.0));
        (t, dtdx / eta, -dtdx * r / (eta * eta))
    }
}

/// AES kernel g_n at distance `r` with pair radius `r0` and its R derivative (n index 0..4
/// for n = 3, 5, 7, 9).
#[inline]
pub fn aes_kernel(n: usize, r: f64, r0: f64) -> (f64, f64) {
    let p = (3 + 2 * n) as i32;
    let a = AES_S[n];
    let f = 1.0 - erf_f64(-a * (r - r0));
    let df = 2.0 * a / SQRTPI * (-(a * (r - r0)).powi(2)).exp();
    let c = 0.5 * AES_K[n];
    (c * f / r.powi(p), c * (df / r.powi(p) - p as f64 * f / r.powi(p + 1)))
}

/// Shell-resolved electrostatics data of one geometry.
#[derive(Clone, Debug)]
pub struct Coulomb {
    pub shells: Vec<Shell>,
    pub nat: usize,
    /// First-order mu_l [1 + k f(CN)] per shell.
    pub mu: Vec<f64>,
    /// Hubbard parameters U'_l.
    pub u: Vec<f64>,
    /// Second- and third-order kernels (nsh x nsh).
    pub gamma: Vec<f64>,
    pub tau: Vec<f64>,
    /// Gamma_l (third order) and 0.036 k_l (fourth order) per shell.
    pub gam3: Vec<f64>,
    pub gam4: Vec<f64>,
    /// Reference shell charge r_l = n0_l - Z_l.
    pub rref: Vec<f64>,
    /// AES kernels g3, g5, g7, g9 (nat x nat).
    pub aes: [Vec<f64>; 4],
    /// Dipole scaling c_A.
    pub dip_scale: Vec<f64>,
}

/// The shells of a molecule: (atom, shell index, l).
pub fn shell_list(mol: &Molecule) -> Vec<Shell> {
    let mut v = Vec::new();
    for (iat, &z) in mol.numbers.iter().enumerate() {
        for (ish, sh) in super::element_shells(z).iter().enumerate() {
            v.push((iat, ish, sh.l));
        }
    }
    v
}

impl Coulomb {
    pub fn new(model: &Model, mol: &Molecule, cn: &[f64]) -> Self {
        Self::new_par(model, mol, cn, false)
    }

    /// [`Coulomb::new`], kernel rows in parallel when `parallel` (same result).
    pub fn new_par(model: &Model, mol: &Molecule, cn: &[f64], parallel: bool) -> Self {
        let shells = shell_list(mol);
        let nsh = shells.len();
        let nat = mol.nat();
        let mut mu = Vec::with_capacity(nsh);
        let mut u = Vec::with_capacity(nsh);
        let mut gam3 = Vec::with_capacity(nsh);
        let mut gam4 = Vec::with_capacity(nsh);
        let mut rref = Vec::with_capacity(nsh);
        for &(iat, ish, l) in &shells {
            let z = mol.numbers[iat] - 1;
            let f = sqrt_cn(cn[iat]).0;
            mu.push(IPEA[z][ish] * (1.0 + IPEA_CN[z] * f));
            u.push(SHELL_HUBBARD[z][ish] * HUBBARD[z] * (1.0 + HUBBARD_CN[z] * f));
            gam3.push(HUBBARD_DERIV[z] * PG_TB3_KSHELL[l]);
            gam4.push(K4 * PG_TB4_KSHELL[l]);
            rref.push(REF_OCC[z][ish] - ZEFF_SHELL[z][ish]);
        }
        let [gamma, tau] = crate::parallel::symmetric(nsh, parallel, |i, j| {
            let (ia, ja) = (shells[i].0, shells[j].0);
            let same = ia == ja;
            let r = if same { 0.0 } else { norm2_sq(sub3(mol.positions[ia], mol.positions[ja])).sqrt() };
            let eta = 2.0 / (1.0 / u[i] + 1.0 / u[j]);
            [gamma_kernel(r, eta).0, tau_kernel(r, eta, same).0]
        });
        let aes = crate::parallel::symmetric(nat, parallel, |i, j| {
            if i == j {
                return [0.0; 4];
            }
            let r = norm2_sq(sub3(mol.positions[i], mol.positions[j])).sqrt();
            let r0 = model.rvdw(mol.numbers[i], mol.numbers[j]);
            [0, 1, 2, 3].map(|n| aes_kernel(n, r, r0).0)
        });
        let dip_scale = (0..nat).map(|i| 1.0 + AES_DIP_SCALE[mol.numbers[i] - 1] * sqrt_cn(cn[i]).0).collect();
        Self { shells, nat, mu, u, gamma, tau, gam3, gam4, rref, aes, dip_scale }
    }

    /// Atomic charges from shell charges.
    pub fn atom_charges(&self, q: &[f64]) -> Vec<f64> {
        let mut qa = vec![0.0; self.nat];
        for (k, &(iat, _, _)) in self.shells.iter().enumerate() {
            qa[iat] += q[k];
        }
        qa
    }

    /// (E1, E2, E3, E4), shells in fixed chunks (in parallel when `parallel`; same result).
    pub fn isotropic_energies(&self, q: &[f64], parallel: bool) -> [f64; 4] {
        let n = self.shells.len();
        let qa = self.atom_charges(q);
        let parts = axtb_base::threads::map_chunks(n, parallel, |rows| {
            let (mut e1, mut e2, mut e3, mut e4) = (0.0, 0.0, 0.0, 0.0);
            for i in rows {
                let ia = self.shells[i].0;
                e1 += self.mu[i] * first_order_g(qa[ia]).0 * q[i];
                let (mut gq, mut gr, mut tq) = (0.0, 0.0, 0.0);
                for j in 0..n {
                    gq += self.gamma[i * n + j] * q[j];
                    if self.shells[j].0 != ia {
                        gr += self.gamma[i * n + j] * self.rref[j];
                    }
                    tq += self.tau[i * n + j] * q[j];
                }
                e2 += 0.5 * q[i] * gq + q[i] * gr;
                e3 += q[i] * q[i] * self.gam3[i] * tq / 3.0;
                e4 += self.gam4[i] * q[i].powi(4) / 24.0;
            }
            [e1, e2, e3, e4]
        });
        let mut e = [0.0; 4];
        for p in parts {
            for k in 0..4 {
                e[k] += p[k];
            }
        }
        e
    }

    /// Shell potentials dE1..4/dq_l, shells in parallel when `parallel` (same result).
    pub fn isotropic_potential(&self, q: &[f64], parallel: bool) -> Vec<f64> {
        let n = self.shells.len();
        let qa = self.atom_charges(q);
        let mut muq = vec![0.0; self.nat];
        for i in 0..n {
            muq[self.shells[i].0] += self.mu[i] * q[i];
        }
        let tq: Vec<f64> = axtb_base::threads::map(n, parallel, |i| (0..n).map(|j| self.tau[i * n + j] * q[j]).sum());
        let w3: Vec<f64> = (0..n).map(|i| q[i] * q[i] * self.gam3[i]).collect();
        axtb_base::threads::map(n, parallel, |i| {
            let ia = self.shells[i].0;
            let (g, dg) = first_order_g(qa[ia]);
            let mut vi = self.mu[i] * g + muq[ia] * dg;
            for j in 0..n {
                vi += self.gamma[i * n + j] * q[j];
                if self.shells[j].0 != ia {
                    vi += self.gamma[i * n + j] * self.rref[j];
                }
                // tau is symmetric: its column i is its row i
                vi += w3[j] * self.tau[i * n + j] / 3.0;
            }
            vi += 2.0 * q[i] * self.gam3[i] * tq[i] / 3.0;
            vi += self.gam4[i] * q[i].powi(3) / 6.0;
            vi
        })
    }

    /// Change of the shell potentials dE1..4/dq_l for shell charge changes `dq` at shell
    /// charges `q` (the Hessian of E1 + E2 + E3 + E4 times `dq`).
    pub fn isotropic_hessian_vec(&self, q: &[f64], dq: &[f64]) -> Vec<f64> {
        let n = self.shells.len();
        let qa = self.atom_charges(q);
        let dqa = self.atom_charges(dq);
        let mut muq = vec![0.0; self.nat];
        let mut mudq = vec![0.0; self.nat];
        for i in 0..n {
            muq[self.shells[i].0] += self.mu[i] * q[i];
            mudq[self.shells[i].0] += self.mu[i] * dq[i];
        }
        let tq: Vec<f64> = (0..n).map(|i| (0..n).map(|j| self.tau[i * n + j] * q[j]).sum()).collect();
        let tdq: Vec<f64> = (0..n).map(|i| (0..n).map(|j| self.tau[i * n + j] * dq[j]).sum()).collect();
        // sum_l tau_kl a_l q_l dq_l
        let w: Vec<f64> = (0..n).map(|l| self.gam3[l] * q[l] * dq[l]).collect();
        let mut v = vec![0.0; n];
        for i in 0..n {
            let ia = self.shells[i].0;
            let (_, dg) = first_order_g(qa[ia]);
            let d2g = first_order_g2(qa[ia]);
            let mut vi = self.mu[i] * dg * dqa[ia] + mudq[ia] * dg + muq[ia] * d2g * dqa[ia];
            let mut gd = 0.0;
            let mut tw = 0.0;
            for j in 0..n {
                gd += self.gamma[i * n + j] * dq[j];
                tw += self.tau[i * n + j] * w[j];
            }
            vi += gd;
            vi += 2.0 / 3.0 * (self.gam3[i] * tq[i] * dq[i] + self.gam3[i] * q[i] * tdq[i] + tw);
            vi += 0.5 * self.gam4[i] * q[i] * q[i] * dq[i];
            v[i] = vi;
        }
        v
    }

    /// Scaled dipoles c_A d_A (also the dipole potential dE/dd_A = c_A dE/d(c d)_A of a
    /// potential of the scaled dipoles).
    pub fn scaled_dipoles(&self, dip: &[[f64; 3]]) -> Vec<[f64; 3]> {
        dip.iter().zip(&self.dip_scale).map(|(d, c)| [c * d[0], c * d[1], c * d[2]]).collect()
    }

    /// AES energy ([`crate::aes::aes_energy`] with the scaled dipoles), atom rows in fixed chunks
    /// (in parallel when `parallel`; same result).
    pub fn aes_energy(&self, mol: &Molecule, qa: &[f64], dip: &[[f64; 3]], qp: &[[f64; 6]], parallel: bool) -> f64 {
        let d = self.scaled_dipoles(dip);
        axtb_base::threads::map_chunks(mol.nat(), parallel, |rows| {
            crate::aes::aes_energy_rows(mol, qa, &d, qp, &self.aes, rows)
        })
        .into_iter()
        .sum()
    }

    /// AES potentials dE/dq_A, dE/dd_A and dE/dQ_A (the off-diagonal quadrupole components
    /// counted twice), exact; atom rows in fixed chunks (in parallel when `parallel`; same
    /// result).
    #[allow(clippy::type_complexity)]
    pub fn aes_potentials(
        &self,
        mol: &Molecule,
        qa: &[f64],
        dip: &[[f64; 3]],
        qp: &[[f64; 6]],
        parallel: bool,
    ) -> (Vec<f64>, Vec<[f64; 3]>, Vec<[f64; 6]>) {
        let (vs, vd_scaled, vq) = self.aes_potentials_raw(mol, qa, dip, qp, parallel);
        (vs, self.scaled_dipoles(&vd_scaled), vq)
    }

    /// [`Coulomb::aes_potentials`] with the dipole potential dE/d(c d) of the scaled dipoles.
    #[allow(clippy::type_complexity)]
    pub fn aes_potentials_raw(
        &self,
        mol: &Molecule,
        qa: &[f64],
        dip: &[[f64; 3]],
        qp: &[[f64; 6]],
        parallel: bool,
    ) -> (Vec<f64>, Vec<[f64; 3]>, Vec<[f64; 6]>) {
        let n = mol.nat();
        let d = self.scaled_dipoles(dip);
        crate::parallel::reduce_rows(
            n,
            parallel,
            || (vec![0.0; n], vec![[0.0; 3]; n], vec![[0.0; 6]; n]),
            |i, acc| aes_potentials_row(mol, qa, &d, qp, &self.aes, i, &mut acc.0, &mut acc.1, &mut acc.2),
            |a, b| {
                crate::parallel::add_into(&mut a.0, &b.0);
                crate::parallel::add_into(a.1.as_flattened_mut(), b.1.as_flattened());
                crate::parallel::add_into(a.2.as_flattened_mut(), b.2.as_flattened());
            },
        )
    }
}

/// The AES potential terms (exact derivatives, as [`crate::fock::aes_potentials`] with `full`)
/// of atom `i` and its partners j < i, added to `vs`, `vd`, `vq`; the potentials of atom i are
/// summed in registers.
pub fn aes_potentials_row(
    mol: &Molecule,
    q: &[f64],
    dip: &[[f64; 3]],
    qp: &[[f64; 6]],
    k: &[Vec<f64>; 4],
    i: usize,
    vs: &mut [f64],
    vd: &mut [[f64; 3]],
    vq: &mut [[f64; 6]],
) {
    let n = mol.nat();
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let qmat = |q: &[f64; 6]| [[q[0], q[1], q[3]], [q[1], q[2], q[4]], [q[3], q[4], q[5]]];
    let mv = |m: &[[f64; 3]; 3], v: [f64; 3]| [dot(m[0], v), dot(m[1], v), dot(m[2], v)];
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
    let w = [1.0, 2.0, 1.0, 2.0, 2.0, 1.0];
    let p = &mol.positions;
    let (k3, k5, k7, k9) =
        (&k[0][i * n..i * n + i], &k[1][i * n..i * n + i], &k[2][i * n..i * n + i], &k[3][i * n..i * n + i]);
    let qi = qmat(&qp[i]);
    let (pi, di, qpi, chi) = (p[i], dip[i], qp[i], q[i]);
    let mut vsi = 0.0;
    let mut vdi = [0.0; 3];
    let mut vqi = [0.0; 6];
    for j in 0..i {
        // (g9 carries the 1/3 of the QQ term)
        let (g3, g5, g7, g9) = (k3[j], k5[j], k7[j], k9[j] / 3.0);
        let r = [p[j][0] - pi[0], p[j][1] - pi[1], p[j][2] - pi[2]];
        let r2 = dot(r, r);
        let rr = [r[0] * r[0], 2.0 * r[0] * r[1], r[1] * r[1], 2.0 * r[0] * r[2], 2.0 * r[1] * r[2], r[2] * r[2]];
        let qj = qmat(&qp[j]);
        let (qir, qjr) = (mv(&qi, r), mv(&qj, r));
        let (rqir, rqjr) = (dot(r, qir), dot(r, qjr));
        let dj = dip[j];
        let (dir, djr) = (dot(di, r), dot(dj, r));
        let chj = q[j];
        let (rdj, rdi) = (sym(r, dj), sym(r, di));
        let (rqj, rqi) = (sym(r, qjr), sym(r, qir));
        // e_qd g3, e_qQ g5
        vs[j] += dir * g3 + rqir * g5;
        vsi += -djr * g3 + rqjr * g5;
        let (vdj, vqj, qpj) = (&mut vd[j], &mut vq[j], &qp[j]);
        for x in 0..3 {
            // e_qd g3, e_dd g5, - e_dQ g7
            vdi[x] +=
                chj * r[x] * g3 + (dj[x] * r2 - 3.0 * r[x] * djr) * g5 + (5.0 * r[x] * rqjr - 2.0 * r2 * qjr[x]) * g7;
            vdj[x] +=
                -chi * r[x] * g3 + (di[x] * r2 - 3.0 * r[x] * dir) * g5 - (5.0 * rqir * r[x] - 2.0 * r2 * qir[x]) * g7;
        }
        for c in 0..6 {
            // e_qQ g5, - e_dQ g7, 1/3 e_QQ g9
            vqi[c] += chj * rr[c] * g5 - (5.0 * rr[c] * djr - 2.0 * r2 * rdj[c]) * g7
                + (35.0 * rr[c] * rqjr - 20.0 * r2 * rqj[c] + 3.0 * r2 * r2 * w[c] * qpj[c]) * g9;
            vqj[c] += chi * rr[c] * g5
                + (5.0 * rr[c] * dir - 2.0 * r2 * rdi[c]) * g7
                + (35.0 * rr[c] * rqir - 20.0 * r2 * rqi[c] + 3.0 * r2 * r2 * w[c] * qpi[c]) * g9;
        }
    }
    vs[i] += vsi;
    for x in 0..3 {
        vd[i][x] += vdi[x];
    }
    for c in 0..6 {
        vq[i][c] += vqi[c];
    }
}

/// Spin constant W_ll' of an atom of element `z`.
#[inline]
pub fn spin_w(z: usize, la: usize, lb: usize) -> f64 {
    WLL_SCALE[z - 1] * SPIN_W0[z - 1][spin_index(la, lb)]
}

/// Spin-polarization energy of shell magnetizations `m`.
pub fn spin_energy(mol: &Molecule, shells: &[Shell], m: &[f64]) -> f64 {
    let mut e = 0.0;
    for (i, &(ia, _, li)) in shells.iter().enumerate() {
        for (j, &(ja, _, lj)) in shells.iter().enumerate() {
            if ja == ia {
                e += 0.5 * m[i] * spin_w(mol.numbers[ia], li, lj) * m[j];
            }
        }
    }
    e
}

/// Spin potentials dE/dm_l.
pub fn spin_potential(mol: &Molecule, shells: &[Shell], m: &[f64]) -> Vec<f64> {
    let mut w = vec![0.0; shells.len()];
    for (i, &(ia, _, li)) in shells.iter().enumerate() {
        for (j, &(ja, _, lj)) in shells.iter().enumerate() {
            if ja == ia {
                w[i] += spin_w(mol.numbers[ia], li, lj) * m[j];
            }
        }
    }
    w
}
