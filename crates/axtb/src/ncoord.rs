//! Coordination numbers of GFN2-xTB: `axtb-base`'s [`CnModel`] with the double exponential
//! counting function ([`DoubleExp::GFN2`]: ka = 10, kb = 20, shift 2 bohr), the conventions'
//! covalent radii and a hard cutoff (r² > cutoff² and r² < 1e-12 bohr² skipped, as xtb and
//! tblite do).
//!
//! Derivatives are never stored as a dense `3 x nat x nat` array: callers pass dE/dCN and
//! the contribution to the Cartesian gradient is accumulated directly (O(N^2) time, O(N)
//! memory).

use crate::molecule::Molecule;
use crate::terms::{Chain, Geometry, Intermediate, Jacobians, Values};
use crate::Result;
use axtb_base::ncoord::{CnModel, Cutoff, DoubleExp};

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

    /// The model of `mol`'s atoms.
    #[inline]
    fn model<'a>(&'a self, mol: &'a Molecule) -> CnModel<DoubleExp, impl Fn(usize, usize) -> f64 + 'a> {
        CnModel::new(DoubleExp::GFN2, move |i: usize, j: usize| self.rcov[mol.id[i]] + self.rcov[mol.id[j]])
            .with_cutoff(Cutoff::Squared(self.cutoff * self.cutoff))
            .with_min_r2(1.0e-12)
    }

    /// Coordination numbers of all atoms.
    pub fn compute(&self, mol: &Molecule) -> Vec<f64> {
        self.model(mol).compute(&mol.positions)
    }

    /// Jacobian `dCN_i/dR_x` (row-major `nat x 3 nat`).
    pub fn jacobian(&self, mol: &Molecule) -> Vec<f64> {
        self.model(mol).jacobian(&mol.positions)
    }

    /// Add `sum_i w_i d²CN_i/dR dR` to `hess` (row-major `3 nat x 3 nat`).
    pub fn add_hessian(&self, mol: &Molecule, w: &[f64], hess: &mut [f64]) {
        self.model(mol).add_hessian(&mol.positions, w, hess);
    }

    /// Add `sum_i dE/dCN_i * dCN_i/dR` to `grad`.
    pub fn add_gradient(&self, mol: &Molecule, dedcn: &[f64], grad: &mut [[f64; 3]]) {
        self.model(mol).add_gradient(&mol.positions, dedcn, grad);
    }
}

/// The coordination number of GFN2-xTB (see the [module](self)) as an intermediate of the
/// GFN2-xTB model's chain layout, named [`Gfn2Cn::NAME`]: the H0 self-energies and the damping
/// radii of the anisotropic electrostatics depend on it.
pub struct Gfn2Cn(Ncoord);

impl Gfn2Cn {
    /// Its name in a chain layout.
    pub const NAME: &'static str = "gfn2 cn";

    pub(crate) fn new(ncoord: Ncoord) -> Self {
        Self(ncoord)
    }
}

impl Intermediate for Gfn2Cn {
    fn name(&self) -> &'static str {
        Self::NAME
    }

    fn compute(&self, mol: &Molecule, _values: &Values) -> Result<Vec<Vec<f64>>> {
        Ok(vec![self.0.compute(mol)])
    }

    fn chain(&self, geo: &Geometry<'_>, w: &[&[f64]], chain: &mut Chain) -> Result<()> {
        self.0.add_gradient(geo.mol, w[0], &mut chain.g);
        Ok(())
    }

    fn jacobian(&self, geo: &Geometry<'_>, _deps: &Jacobians) -> Result<Vec<Vec<f64>>> {
        Ok(vec![self.0.jacobian(geo.mol)])
    }
}
