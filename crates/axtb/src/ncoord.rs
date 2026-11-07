//! Coordination numbers of GFN2-xTB: the double exponential counting function (ka = 10,
//! kb = 20, shift 2 bohr).
//!
//! Derivatives are never stored as a dense `3 x nat x nat` array: callers pass dE/dCN and
//! the contribution to the Cartesian gradient is accumulated directly (O(N^2) time, O(N)
//! memory).

use crate::molecule::{norm2_sq, sub3, Molecule};

const DEXP_KA: f64 = 10.0;
const DEXP_KB: f64 = 20.0;
const DEXP_SHIFT: f64 = 2.0;

#[inline]
fn exp_count(k: f64, r: f64, r0: f64) -> f64 {
    1.0 / (1.0 + (-k * (r0 / r - 1.0)).exp())
}

#[inline]
fn exp_dcount(k: f64, r: f64, r0: f64) -> f64 {
    let e = (-k * (r0 / r - 1.0)).exp();
    (-k * r0 * e) / (r.powf(2.0) * (e + 1.0).powf(2.0))
}

/// Second derivative of [`exp_count`] with respect to r.
#[inline]
fn exp_d2count(k: f64, r: f64, r0: f64) -> f64 {
    let f = exp_count(k, r, r0);
    let du = k * r0 / (r * r);
    let d2u = -2.0 * k * r0 / (r * r * r);
    let df = -f * (1.0 - f) * du;
    -((1.0 - 2.0 * f) * df * du + f * (1.0 - f) * d2u)
}

/// The double exponential counting function and its first and second derivatives in r.
#[inline]
fn count(r: f64, rc: f64) -> f64 {
    exp_count(DEXP_KA, r, rc) * exp_count(DEXP_KB, r, rc + DEXP_SHIFT)
}

#[inline]
fn dcount(r: f64, rc: f64) -> f64 {
    exp_dcount(DEXP_KA, r, rc) * exp_count(DEXP_KB, r, rc + DEXP_SHIFT)
        + exp_count(DEXP_KA, r, rc) * exp_dcount(DEXP_KB, r, rc + DEXP_SHIFT)
}

#[inline]
fn d2count(r: f64, rc: f64) -> f64 {
    let (a, da, d2a) = (exp_count(DEXP_KA, r, rc), exp_dcount(DEXP_KA, r, rc), exp_d2count(DEXP_KA, r, rc));
    let rb = rc + DEXP_SHIFT;
    let (b, db, d2b) = (exp_count(DEXP_KB, r, rb), exp_dcount(DEXP_KB, r, rb), exp_d2count(DEXP_KB, r, rb));
    d2a * b + 2.0 * da * db + a * d2b
}

/// A coordination number definition for one molecule.
#[derive(Clone, Debug)]
pub struct Ncoord {
    /// Covalent radius of each species (bohr).
    rcov: Vec<f64>,
    cutoff: f64,
}

impl Ncoord {
    /// Coordination number with per-element radii `rcov(z)` (bohr) and a cutoff (bohr).
    pub fn with_radii(mol: &Molecule, rcov: impl Fn(usize) -> f64, cutoff: f64) -> Self {
        Self { rcov: mol.species.iter().map(|&z| rcov(z)).collect(), cutoff }
    }

    /// Coordination numbers of all atoms.
    pub fn compute(&self, mol: &Molecule) -> Vec<f64> {
        let nat = mol.nat();
        let mut cn = vec![0.0; nat];
        let cutoff2 = self.cutoff * self.cutoff;
        for i in 0..nat {
            let ri = mol.positions[i];
            let rci = self.rcov[mol.id[i]];
            for j in 0..i {
                let rij = sub3(ri, mol.positions[j]);
                let r2 = norm2_sq(rij);
                if r2 > cutoff2 || r2 < 1.0e-12 {
                    continue;
                }
                let c = count(r2.sqrt(), rci + self.rcov[mol.id[j]]);
                cn[i] += c;
                cn[j] += c;
            }
        }
        cn
    }

    /// Jacobian `dCN_i/dR_x` (row-major `nat x 3 nat`).
    pub fn jacobian(&self, mol: &Molecule) -> Vec<f64> {
        let nat = mol.nat();
        let n3 = 3 * nat;
        let mut jac = vec![0.0; nat * n3];
        let cutoff2 = self.cutoff * self.cutoff;
        for i in 0..nat {
            let ri = mol.positions[i];
            let rci = self.rcov[mol.id[i]];
            for j in 0..i {
                let rij = sub3(ri, mol.positions[j]);
                let r2 = norm2_sq(rij);
                if r2 > cutoff2 || r2 < 1.0e-12 {
                    continue;
                }
                let r = r2.sqrt();
                let dc = dcount(r, rci + self.rcov[mol.id[j]]) / r;
                for k in 0..3 {
                    let d = dc * rij[k];
                    jac[i * n3 + 3 * i + k] += d;
                    jac[i * n3 + 3 * j + k] -= d;
                    jac[j * n3 + 3 * i + k] += d;
                    jac[j * n3 + 3 * j + k] -= d;
                }
            }
        }
        jac
    }

    /// Add `sum_i w_i d²CN_i/dR dR` to `hess` (row-major `3 nat x 3 nat`).
    pub fn add_hessian(&self, mol: &Molecule, w: &[f64], hess: &mut [f64]) {
        let nat = mol.nat();
        let n3 = 3 * nat;
        let cutoff2 = self.cutoff * self.cutoff;
        for i in 0..nat {
            let ri = mol.positions[i];
            let rci = self.rcov[mol.id[i]];
            for j in 0..i {
                let rij = sub3(ri, mol.positions[j]);
                let r2 = norm2_sq(rij);
                if r2 > cutoff2 || r2 < 1.0e-12 {
                    continue;
                }
                let r = r2.sqrt();
                let rc = rci + self.rcov[mol.id[j]];
                let wij = w[i] + w[j];
                crate::coulomb::add_radial_hessian(hess, n3, i, j, rij, r, wij * dcount(r, rc), wij * d2count(r, rc));
            }
        }
    }

    /// Add `sum_i dE/dCN_i * dCN_i/dR` to `grad`.
    pub fn add_gradient(&self, mol: &Molecule, dedcn: &[f64], grad: &mut [[f64; 3]]) {
        let nat = mol.nat();
        let cutoff2 = self.cutoff * self.cutoff;
        for i in 0..nat {
            let ri = mol.positions[i];
            let rci = self.rcov[mol.id[i]];
            for j in 0..i {
                let rij = sub3(ri, mol.positions[j]);
                let r2 = norm2_sq(rij);
                if r2 > cutoff2 || r2 < 1.0e-12 {
                    continue;
                }
                let r = r2.sqrt();
                let dc = dcount(r, rci + self.rcov[mol.id[j]]) * (dedcn[i] + dedcn[j]) / r;
                for k in 0..3 {
                    grad[i][k] += dc * rij[k];
                    grad[j][k] -= dc * rij[k];
                }
            }
        }
    }
}
