//! Self-consistent repulsion of g-xTB 2.0.1 (spec §4.4):
//!
//! E = sum_{A<B} z_A z_B M_AB, z_A = Z0_A (1 - kq_A q_A) with the SCF Mulliken atomic charges q,
//! M_AB = [k_s exp(-k_a a_AB (R + R0_AB)^2) + exp(-a_AB (R + R0_AB)^{3/2})]
//!        [1 + k1_AB/R + kpen_AB (Rv/R)^2 + k3 (Rv/R)^3 + k4 (Rv/R)^4] * cut(R),
//! a_A = alpha0_A (1 + kcn_A f(CN_A)), a_AB = a_A a_B/(a_A + a_B), R0_AB = sqrt(R0_A R0_B),
//! k1_AB and kpen_AB arithmetic means, Rv the shared vdW pair radius; pairs beyond 25 bohr are
//! cut ([`Model::cut`]). The geometry part M is computed once per geometry; the potential
//! dE/dq_A enters the Fock matrix as an atomic Mulliken potential.

use super::{sqrt_cn, Model};
use crate::data_v2::{REP_ALPHA, REP_CN, REP_K1, REP_Q, REP_ROFFSET, REP_ZEFF};
use axtb::molecule::{norm2_sq, sub3};
use axtb::Molecule;

pub const CUTOFF: f64 = 25.0;
pub const K_S: f64 = 0.0046511298;
pub const K_A: f64 = 0.73;
pub const K3: f64 = 0.011095539524126988;
pub const K4: f64 = 0.011607795128002491;
pub const KPEN_LIGHT: f64 = 0.012098131381864387;
pub const KPEN_HEAVY: f64 = 0.008544252691968662;

fn kpen(z: usize) -> f64 {
    if z <= 2 {
        KPEN_LIGHT
    } else {
        KPEN_HEAVY
    }
}

/// Scaled exponent a_A and its CN derivative.
pub fn alpha(z: usize, cn: f64) -> (f64, f64) {
    let (f, df) = sqrt_cn(cn);
    (REP_ALPHA[z - 1] * (1.0 + REP_CN[z - 1] * f), REP_ALPHA[z - 1] * REP_CN[z - 1] * df)
}

/// M_AB and its derivatives with respect to R and a_AB.
pub fn kernel(model: &Model, za: usize, zb: usize, r: f64, a: f64) -> (f64, f64, f64) {
    let (sw, dsw) = model.cut(r, CUTOFF);
    if sw == 0.0 {
        return (0.0, 0.0, 0.0);
    }
    let x = r + (REP_ROFFSET[za - 1] * REP_ROFFSET[zb - 1]).sqrt();
    let e1 = K_S * (-K_A * a * x * x).exp();
    let x15 = x.sqrt() * x;
    let e2 = (-a * x15).exp();
    let damp = e1 + e2;
    let ddamp_dr = -2.0 * K_A * a * x * e1 - 1.5 * a * x.sqrt() * e2;
    let ddamp_da = -K_A * x * x * e1 - x15 * e2;
    let k1 = 0.5 * (REP_K1[za - 1] + REP_K1[zb - 1]);
    let kp = 0.5 * (kpen(za) + kpen(zb));
    let rv = model.rvdw(za, zb);
    let t = rv / r;
    let poly = 1.0 + k1 / r + kp * t * t + K3 * t * t * t + K4 * t * t * t * t;
    let dpoly = -k1 / (r * r) - (2.0 * kp * t * t + 3.0 * K3 * t * t * t + 4.0 * K4 * t * t * t * t) / r;
    let m = damp * poly;
    (m * sw, (ddamp_dr * poly + damp * dpoly) * sw + m * dsw, ddamp_da * poly * sw)
}

/// Geometry part of the repulsion.
#[derive(Clone, Debug)]
pub struct Repulsion {
    pub nat: usize,
    /// Z0_A and kq_A per atom.
    pub z0: Vec<f64>,
    pub kq: Vec<f64>,
    /// M_AB (nat x nat, zero diagonal).
    pub m: Vec<f64>,
}

impl Repulsion {
    pub fn new(model: &Model, mol: &Molecule, cn: &[f64]) -> Self {
        Self::new_par(model, mol, cn, false)
    }

    /// [`Repulsion::new`], rows in parallel when `parallel` (same result).
    pub fn new_par(model: &Model, mol: &Molecule, cn: &[f64], parallel: bool) -> Self {
        let n = mol.nat();
        let al: Vec<f64> = (0..n).map(|i| alpha(mol.numbers[i], cn[i]).0).collect();
        let reach2 = model.reach(CUTOFF).powi(2);
        let [m] = crate::parallel::symmetric(n, parallel, |i, j| {
            let r2 = norm2_sq(sub3(mol.positions[i], mol.positions[j]));
            if i == j || r2 > reach2 || r2 < 1e-12 {
                return [0.0];
            }
            let a = al[i] * al[j] / (al[i] + al[j]);
            [kernel(model, mol.numbers[i], mol.numbers[j], r2.sqrt(), a).0]
        });
        let z0 = mol.numbers.iter().map(|&z| REP_ZEFF[z - 1]).collect();
        let kq = mol.numbers.iter().map(|&z| REP_Q[z - 1]).collect();
        Self { nat: n, z0, kq, m }
    }

    /// Effective charges z_A.
    pub fn zeff(&self, q: &[f64]) -> Vec<f64> {
        (0..self.nat).map(|i| self.z0[i] * (1.0 - self.kq[i] * q[i])).collect()
    }

    /// Energy at atomic charges `q`.
    pub fn energy(&self, q: &[f64]) -> f64 {
        self.energy_par(q, false)
    }

    /// [`Repulsion::energy`], atom rows in fixed chunks (in parallel when `parallel`).
    pub fn energy_par(&self, q: &[f64], parallel: bool) -> f64 {
        let z = self.zeff(q);
        let n = self.nat;
        crate::parallel::sum(n, parallel, |i| (0..i).map(|j| z[i] * z[j] * self.m[i * n + j]).sum::<f64>())
    }

    /// dE/dq_A.
    pub fn potential(&self, q: &[f64]) -> Vec<f64> {
        self.potential_par(q, false)
    }

    /// [`Repulsion::potential`], atoms in parallel when `parallel` (same result).
    pub fn potential_par(&self, q: &[f64], parallel: bool) -> Vec<f64> {
        let z = self.zeff(q);
        let n = self.nat;
        crate::parallel::map_items(n, parallel, |i| {
            -self.kq[i] * self.z0[i] * (0..n).map(|j| self.m[i * n + j] * z[j]).sum::<f64>()
        })
    }

    /// Change of dE/dq_A for atomic charge changes `dq` (E is quadratic in q).
    pub fn potential_delta(&self, dq: &[f64]) -> Vec<f64> {
        let n = self.nat;
        let dz: Vec<f64> = (0..n).map(|j| -self.z0[j] * self.kq[j] * dq[j]).collect();
        (0..n).map(|i| -self.kq[i] * self.z0[i] * (0..n).map(|j| self.m[i * n + j] * dz[j]).sum::<f64>()).collect()
    }

    /// Explicit gradient at fixed charges `q`: geometry to `g`, dE/dCN (g-xTB CN) to `dcn`.
    pub fn gradient(&self, model: &Model, mol: &Molecule, cn: &[f64], q: &[f64], g: &mut [[f64; 3]], dcn: &mut [f64]) {
        self.gradient_par(model, mol, cn, q, g, dcn, false);
    }

    /// [`Repulsion::gradient`], atom rows in fixed chunks (in parallel when `parallel`).
    pub fn gradient_par(
        &self,
        model: &Model,
        mol: &Molecule,
        cn: &[f64],
        q: &[f64],
        g_out: &mut [[f64; 3]],
        dcn_out: &mut [f64],
        parallel: bool,
    ) {
        let n = self.nat;
        let z = self.zeff(q);
        let al: Vec<(f64, f64)> = (0..n).map(|i| alpha(mol.numbers[i], cn[i])).collect();
        let reach2 = model.reach(CUTOFF).powi(2);
        let (gs, ds) = crate::parallel::reduce_rows(
            n,
            parallel,
            || (vec![[0.0; 3]; n], vec![0.0; n]),
            |i, (g, dcn)| {
                for j in 0..i {
                    let v = sub3(mol.positions[i], mol.positions[j]);
                    let r2 = norm2_sq(v);
                    if r2 > reach2 || r2 < 1e-12 {
                        continue;
                    }
                    let r = r2.sqrt();
                    let (ai, aj) = (al[i].0, al[j].0);
                    let s = ai + aj;
                    let a = ai * aj / s;
                    let (_, dmdr, dmda) = kernel(model, mol.numbers[i], mol.numbers[j], r, a);
                    let zz = z[i] * z[j];
                    let f = zz * dmdr / r;
                    for x in 0..3 {
                        g[i][x] += f * v[x];
                        g[j][x] -= f * v[x];
                    }
                    // da/dai = aj^2/s^2
                    dcn[i] += zz * dmda * aj * aj / (s * s) * al[i].1;
                    dcn[j] += zz * dmda * ai * ai / (s * s) * al[j].1;
                }
            },
            |a, b| {
                crate::parallel::add_into(a.0.as_flattened_mut(), b.0.as_flattened());
                crate::parallel::add_into(&mut a.1, &b.1);
            },
        );
        crate::parallel::add_into(g_out.as_flattened_mut(), gs.as_flattened());
        crate::parallel::add_into(dcn_out, &ds);
    }
}
