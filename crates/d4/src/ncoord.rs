// SPDX-License-Identifier: LGPL-3.0-or-later
//! DFT-D4 coordination number (`cn_count%dftd4` of mctc-lib 0.7.0): error
//! function counting with the electronegativity-dependent prefactor, hard
//! real-space cutoff, no CN cap (`cut < 0`), molecular (non-periodic) systems.

use crate::model::D4Model;

/// Steepness of the D4 counting function.
pub const KCN: f64 = 7.5;
const K4: f64 = 4.10451;
const K5: f64 = 19.08857;
const K6: f64 = 2.0 * (11.28174 * 11.28174);
const SQRTPI: f64 = 1.7724538509055159; // sqrt(pi) of the f64 pi, correctly rounded

/// The error function of the reference programs (from `axtb-base`).
pub use axtb_base::math::erf;

/// Electronegativity factor `k4 * exp(-(|en_i - en_j| + k5)^2 / k6)`.
#[inline]
pub fn en_factor(en_i: f64, en_j: f64) -> f64 {
    let x = (en_i - en_j).abs() + K5;
    K4 * (-((x * x) / K6)).exp()
}

/// Counting function `0.5 * (1 + erf(-kcn (r - rc) / rc))`.
#[inline]
pub fn count(rc: f64, r: f64) -> f64 {
    0.5 * (1.0 + erf(-((KCN * (r - rc)) / rc)))
}

/// Radial derivative of [`count`].
#[inline]
pub fn dcount(rc: f64, r: f64) -> f64 {
    let e = KCN * (r - rc) / rc;
    let expterm = (-(e * e)).exp();
    -(KCN * expterm) / (SQRTPI * rc)
}

impl D4Model {
    /// D4 coordination numbers (written to `cn`, length `nat`) with a hard
    /// real-space cutoff `cutoff` (bohr; dftd4/tblite use 30).
    pub fn coordination_numbers(&self, positions: &[[f64; 3]], cutoff: f64, cn: &mut [f64]) {
        let nat = self.natoms();
        assert!(positions.len() >= nat && cn.len() >= nat);
        let cutoff2 = cutoff * cutoff;
        cn[..nat].fill(0.0);
        for iat in 0..nat {
            let isp = self.id[iat];
            let xi = positions[iat];
            for jat in 0..iat {
                let jsp = self.id[jat];
                let xj = positions[jat];
                let v = [xi[0] - xj[0], xi[1] - xj[1], xi[2] - xj[2]];
                let r2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
                if r2 > cutoff2 || r2 < 1.0e-12 {
                    continue;
                }
                let r = r2.sqrt();
                let den = en_factor(self.en[isp], self.en[jsp]);
                let rc = self.rcov[isp] + self.rcov[jsp];
                let c = den * count(rc, r);
                cn[iat] += c;
                cn[jat] += c;
            }
        }
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
        let cutoff2 = cutoff * cutoff;
        for iat in 0..nat {
            let isp = self.id[iat];
            let xi = positions[iat];
            for jat in 0..iat {
                let jsp = self.id[jat];
                let xj = positions[jat];
                let v = [xi[0] - xj[0], xi[1] - xj[1], xi[2] - xj[2]];
                let r2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
                if r2 > cutoff2 || r2 < 1.0e-12 {
                    continue;
                }
                let dedcnij = dedcn[iat] + dedcn[jat];
                if dedcnij == 0.0 {
                    continue;
                }
                let r = r2.sqrt();
                let den = en_factor(self.en[isp], self.en[jsp]);
                let rc = self.rcov[isp] + self.rcov[jsp];
                let dc = den * dcount(rc, r);
                for k in 0..3 {
                    let g = dc * v[k] / r * dedcnij;
                    gradient[iat][k] += g;
                    gradient[jat][k] -= g;
                }
            }
        }
    }
}
