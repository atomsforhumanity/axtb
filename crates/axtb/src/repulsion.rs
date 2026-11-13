// SPDX-License-Identifier: LGPL-3.0-or-later
//! Classical repulsion of the GFN-xTB methods:
//! `E = sum_{A<B} Z_A Z_B exp(-sqrt(α_A α_B) R^k) / R^r`.

use crate::molecule::{norm2_sq, sub3, Molecule};
use crate::terms::{Chain, Chain2, Geometry, GeometryTerm, TermEnergy};
use crate::Result;

#[derive(Clone, Debug)]
pub struct Repulsion {
    /// Pair exponent, effective charge product and distance exponents, `[isp][jsp]`.
    alpha: Vec<Vec<f64>>,
    zeff: Vec<Vec<f64>>,
    kexp: Vec<Vec<f64>>,
    rexp: f64,
    cutoff: f64,
}

impl Repulsion {
    /// `alpha`, `zeff` per species; `kexp` applies when either atom is heavier than He,
    /// `kexp_light` to H/He pairs.
    pub fn new(mol: &Molecule, alpha: &[f64], zeff: &[f64], kexp: f64, kexp_light: f64, rexp: f64) -> Self {
        let n = mol.nid();
        let mut a = vec![vec![0.0; n]; n];
        let mut z = vec![vec![0.0; n]; n];
        let mut k = vec![vec![0.0; n]; n];
        for i in 0..n {
            for j in 0..n {
                a[i][j] = (alpha[i] * alpha[j]).sqrt();
                z[i][j] = zeff[i] * zeff[j];
                k[i][j] = if mol.species[i] > 2 || mol.species[j] > 2 { kexp } else { kexp_light };
            }
        }
        Self { alpha: a, zeff: z, kexp: k, rexp, cutoff: 25.0 }
    }

    /// Change the real-space cutoff (bohr).
    pub fn with_cutoff(mut self, cutoff: f64) -> Self {
        self.cutoff = cutoff;
        self
    }

    /// Energy per atom; with `grad` also the gradient.
    pub fn compute(&self, mol: &Molecule, energies: &mut [f64], mut grad: Option<&mut [[f64; 3]]>) {
        let cutoff2 = self.cutoff * self.cutoff;
        for i in 0..mol.nat() {
            let isp = mol.id[i];
            for j in 0..i {
                let jsp = mol.id[j];
                let rij = sub3(mol.positions[i], mol.positions[j]);
                let r2 = norm2_sq(rij);
                if r2 > cutoff2 || r2 < 1.0e-12 {
                    continue;
                }
                let r1 = r2.sqrt();
                let kexp = self.kexp[jsp][isp];
                let alpha = self.alpha[jsp][isp];
                let r1k = r1.powf(kexp);
                let exa = (-alpha * r1k).exp();
                let r1r = r1.powf(self.rexp);
                let de = self.zeff[jsp][isp] * exa / r1r;
                energies[i] += 0.5 * de;
                energies[j] += 0.5 * de;
                if let Some(g) = grad.as_deref_mut() {
                    let f = -(alpha * r1k * kexp + self.rexp) * de / r2;
                    for k in 0..3 {
                        g[i][k] += f * rij[k];
                        g[j][k] -= f * rij[k];
                    }
                }
            }
        }
    }

    /// Second derivatives, added to `hess` (row-major `3 nat x 3 nat`).
    fn hessian(&self, mol: &Molecule, hess: &mut [f64]) {
        let n3 = 3 * mol.nat();
        let cutoff2 = self.cutoff * self.cutoff;
        for i in 0..mol.nat() {
            let isp = mol.id[i];
            for j in 0..i {
                let jsp = mol.id[j];
                let rij = sub3(mol.positions[i], mol.positions[j]);
                let r2 = norm2_sq(rij);
                if r2 > cutoff2 || r2 < 1.0e-12 {
                    continue;
                }
                let r1 = r2.sqrt();
                let kexp = self.kexp[jsp][isp];
                let alpha = self.alpha[jsp][isp];
                let e = self.zeff[jsp][isp] * (-alpha * r1.powf(kexp)).exp() / r1.powf(self.rexp);
                let t = -alpha * kexp * r1.powf(kexp - 1.0) - self.rexp / r1;
                let dt = -alpha * kexp * (kexp - 1.0) * r1.powf(kexp - 2.0) + self.rexp / r2;
                axtb_base::math::add_radial_hessian(hess, n3, i, j, rij, r1, e * t, e * (t * t + dt));
            }
        }
    }
}

/// The repulsion as a term: energy per atom (each pair split evenly), gradient and analytic
/// Hessian.
impl GeometryTerm for Repulsion {
    type Cache = ();

    fn names(&self) -> &[&'static str] {
        &["repulsion"]
    }

    fn partitions_energy(&self) -> bool {
        true
    }

    fn update(&self, _geo: &Geometry<'_>) -> Result<()> {
        Ok(())
    }

    fn energy(&self, geo: &Geometry<'_>, _c: &(), e: &mut TermEnergy<'_>, chain: Option<&mut Chain>) {
        let grad = chain.map(|c| &mut c.g[..]);
        let mut own = Vec::new();
        let atoms = match e.atoms.as_deref_mut() {
            Some(row) => row,
            None => {
                own.resize(geo.mol.nat(), 0.0);
                &mut own[..]
            }
        };
        self.compute(geo.mol, atoms, grad);
        e.pieces[0] = atoms.iter().sum();
    }

    fn add_hessian(&self, geo: &Geometry<'_>, _c: &(), h: &mut Chain2<'_>) -> Result<()> {
        self.hessian(geo.mol, h.h);
        Ok(())
    }
}
