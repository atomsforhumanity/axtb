// SPDX-License-Identifier: LGPL-3.0-or-later
//! DFT-D4 coordination number (`cn_count%dftd4` of mctc-lib 0.7.0): error
//! function counting with the electronegativity-dependent prefactor, hard
//! real-space cutoff, no CN cap (`cut < 0`), molecular (non-periodic) systems.
//!
//! The pair loops are `axtb-base`'s [`CnModel`]: the counting function [`Erf::D4`]
//! (k = [`KCN`]), [`en_factor`] as the pair weight, dftd4's pair test (r² > cutoff² and
//! r² < 1e-12 bohr² skipped) and its gradient contraction ([`GradientOrder::Jacobian`],
//! pairs whose dE/dCN sum to zero skipped).

use crate::model::D4Model;
use axtb_base::ncoord::{CnModel, Count, Cutoff, Erf, GradientOrder, Weight};

/// Steepness of the D4 counting function.
pub const KCN: f64 = Erf::D4.k;
const K4: f64 = 4.10451;
const K5: f64 = 19.08857;
const K6: f64 = 2.0 * (11.28174 * 11.28174);

/// The error function of the reference programs (from `axtb-base`).
pub use axtb_base::math::erf;

/// Electronegativity factor `k4 * exp(-(|en_i - en_j| + k5)^2 / k6)`.
#[inline]
pub fn en_factor(en_i: f64, en_j: f64) -> f64 {
    let x = (en_i - en_j).abs() + K5;
    K4 * (-((x * x) / K6)).exp()
}

/// Counting function `0.5 * (1 + erf(-kcn (r - rc) / rc))` ([`Erf::D4`]).
#[inline]
pub fn count(rc: f64, r: f64) -> f64 {
    Erf::D4.count(r, rc)
}

/// Radial derivative of [`count`].
#[inline]
pub fn dcount(rc: f64, r: f64) -> f64 {
    Erf::D4.dcount(r, rc)
}

impl D4Model {
    /// The D4 coordination number of the model's atoms with a hard real-space cutoff
    /// `cutoff` (bohr).
    pub(crate) fn cn_model(&self, cutoff: f64) -> CnModel<Erf, impl Fn(usize, usize) -> f64 + '_, impl Weight + '_> {
        CnModel::new(Erf::D4, move |i: usize, j: usize| self.rcov[self.id[i]] + self.rcov[self.id[j]])
            .with_weight(move |i: usize, j: usize| en_factor(self.en[self.id[i]], self.en[self.id[j]]))
            .with_cutoff(Cutoff::Squared(cutoff * cutoff))
            .with_min_r2(1.0e-12)
            .with_order(GradientOrder::Jacobian)
            .skipping_zero_weights()
    }

    /// D4 coordination numbers (written to `cn`, length `nat`) with a hard
    /// real-space cutoff `cutoff` (bohr; dftd4/tblite use 30).
    pub fn coordination_numbers(&self, positions: &[[f64; 3]], cutoff: f64, cn: &mut [f64]) {
        let nat = self.natoms();
        assert!(positions.len() >= nat && cn.len() >= nat);
        self.cn_model(cutoff).compute_into(&positions[..nat], &mut cn[..nat]);
    }

    /// Contract `dE/dCN` with the CN geometry derivative and accumulate into
    /// `gradient` (`add_coordination_number_derivs`).
    pub fn add_coordination_number_derivs(
        &self,
        positions: &[[f64; 3]],
        cutoff: f64,
        dedcn: &[f64],
        gradient: &mut [[f64; 3]],
    ) {
        let nat = self.natoms();
        assert!(positions.len() >= nat && dedcn.len() >= nat && gradient.len() >= nat);
        self.cn_model(cutoff).add_gradient(&positions[..nat], dedcn, gradient);
    }
}
