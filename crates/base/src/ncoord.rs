// SPDX-License-Identifier: LGPL-3.0-or-later
//! Coordination numbers, written once.
//!
//! CN_A = Σ_B w_AB f(R_AB; Rc_AB) s(R_AB), summed over the pairs that count, and its geometry
//! derivatives: the contraction Σ_A (dE/dCN_A) dCN_A/dR into a gradient, the Jacobian dCN/dR
//! and the weighted second derivative Σ_A w_A d²CN_A/dR². A [`CnModel`] is put together from
//! explicit parts, each a convention of the program it reproduces:
//!
//! | part | type | choices |
//! |---|---|---|
//! | counting function f(r; rc) with its r-derivative and, where one exists, its second | [`Count`], [`Count2`] | [`DoubleExp`], [`Erf`], [`XtbErf`] |
//! | pair radius Rc_AB | a closure `(i, j) -> rc` (bohr) | the caller's: sums or means of element radii |
//! | pair weight w_AB | [`Weight`] | [`Unweighted`], or a closure (D4's electronegativity factor) |
//! | which pairs count | [`Cutoff`], [`CnModel::min_r2`] | r² ≤ c² (dftd4, tblite, xtb), r ≤ c (xtb's GFN-FF) or every pair; pairs with r² below `min_r2` skipped (1e-12 bohr² in dftd4, tblite and xtb) |
//! | cutoff factor s(r) | [`Switch`] | [`NoSwitch`], or [`Smooth`] with the caller's s(r) and ds/dr |
//! | cap | [`log_cap`], [`dlog_cap`] | applied by the caller to the sums (multicharge's EEQ, xtb's GFN-FF) |
//! | gradient contraction | [`GradientOrder`], [`CnModel::skip_zero_weight`] | the order of the chain-rule products; whether pairs whose weights sum to zero are skipped |
//!
//! The counting functions:
//!
//! | function | f(r; rc) | from | used by |
//! |---|---|---|---|
//! | [`DoubleExp::GFN2`] | 1/(1 + e^(−10 (rc/r − 1))) · 1/(1 + e^(−20 ((rc + 2)/r − 1))) | mctc-lib's `dexp` counting function as xtb and tblite evaluate it | GFN2-xTB |
//! | [`Erf`] | ½ (1 + erf(−k (r − rc)/rc)) | dftd4's `erf_count` (mctc-lib) | D4 (k = 7.5, with the electronegativity factor as weight), the EEQ guess of GFN2-xTB; in `axtb-gxtb` the revD4 CNs and g-xTB 1.1's own CNs |
//! | [`XtbErf::GFNFF`] | ½ (1 + erf(−7.5 (r − rc)/rc)) with the relative distance formed first; its derivative uses xtb's truncated √π | xtb's `gfnff_dlogcoord` | GFN-FF |
//!
//! Pairs are visited for i ascending and j < i ascending, and each count is added to CN_i and
//! CN_j, so every CN sums its partners in ascending order, as the reference programs' loops do.
//! The parts are generic parameters (closures and zero-sized types), so each model's pair loop is
//! compiled for its parts: no indirect calls and no allocations in the loops. Every expression
//! keeps the operation order of the code it replaced (`axtb`'s GFN2-xTB CN, dftd4's CN in
//! `axtb-d4`, GFN-FF's), so the results are those of the programs bit for bit; where two
//! programs order the chain rule differently, both orders are here as named variants.

use crate::math::{add_radial_hessian, erf, norm2_sq, sub3};
use std::ops::{Add, Div, Mul, Neg, Sub};

/// √π of the f64 π, correctly rounded (dftd4).
const SQRTPI: f64 = 1.7724538509055159;

/// A counting function f(r; rc) of the distance r and the pair radius rc (bohr), and its first
/// r-derivative.
pub trait Count {
    /// f(r; rc).
    fn count(&self, r: f64, rc: f64) -> f64;
    /// df/dr.
    fn dcount(&self, r: f64, rc: f64) -> f64;
}

/// A counting function with an analytic second r-derivative.
pub trait Count2: Count {
    /// d²f/dr².
    fn d2count(&self, r: f64, rc: f64) -> f64;
}

/// The double exponential counting function of GFN2-xTB (mctc-lib's `dexp` CN, as xtb and tblite
/// evaluate it): f = e(ka; r, rc) · e(kb; r, rc + shift) with e(k; r, r0) = 1/(1 + exp(−k (r0/r − 1))).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct DoubleExp {
    /// Steepness of the first factor.
    pub ka: f64,
    /// Steepness of the second factor.
    pub kb: f64,
    /// Offset of the second factor's radius (bohr).
    pub shift: f64,
}

impl DoubleExp {
    /// GFN2-xTB: ka = 10, kb = 20, shift 2 bohr.
    pub const GFN2: DoubleExp = DoubleExp { ka: 10.0, kb: 20.0, shift: 2.0 };
}

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

impl Count for DoubleExp {
    #[inline]
    fn count(&self, r: f64, rc: f64) -> f64 {
        exp_count(self.ka, r, rc) * exp_count(self.kb, r, rc + self.shift)
    }

    #[inline]
    fn dcount(&self, r: f64, rc: f64) -> f64 {
        exp_dcount(self.ka, r, rc) * exp_count(self.kb, r, rc + self.shift)
            + exp_count(self.ka, r, rc) * exp_dcount(self.kb, r, rc + self.shift)
    }
}

impl Count2 for DoubleExp {
    #[inline]
    fn d2count(&self, r: f64, rc: f64) -> f64 {
        let (a, da, d2a) = (exp_count(self.ka, r, rc), exp_dcount(self.ka, r, rc), exp_d2count(self.ka, r, rc));
        let rb = rc + self.shift;
        let (b, db, d2b) = (exp_count(self.kb, r, rb), exp_dcount(self.kb, r, rb), exp_d2count(self.kb, r, rb));
        d2a * b + 2.0 * da * db + a * d2b
    }
}

/// The error-function counting function of dftd4 (mctc-lib's `erf_count`):
/// f = ½ (1 + erf(−k (r − rc)/rc)) with the steepness k > 0.
///
/// A program that writes the argument as `(−k)(r − rc)/rc`, or as `k′ (r − rc)/rc` with a
/// negative k′, gets the same bits with k = −k′: IEEE rounding is symmetric in sign, so
/// (−a)·b = −(a·b) and (−a)/b = −(a/b) exactly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Erf {
    /// Steepness k.
    pub k: f64,
}

impl Erf {
    /// k = 7.5: dftd4's D4 CN, multicharge's EEQ CN and g-xTB's revD4 CN.
    pub const D4: Erf = Erf { k: 7.5 };
}

impl Count for Erf {
    #[inline]
    fn count(&self, r: f64, rc: f64) -> f64 {
        0.5 * (1.0 + erf(-((self.k * (r - rc)) / rc)))
    }

    #[inline]
    fn dcount(&self, r: f64, rc: f64) -> f64 {
        let e = self.k * (r - rc) / rc;
        let expterm = (-(e * e)).exp();
        -(self.k * expterm) / (SQRTPI * rc)
    }
}

impl Count2 for Erf {
    #[inline]
    fn d2count(&self, r: f64, rc: f64) -> f64 {
        let e = self.k * (r - rc) / rc;
        2.0 * e * self.k * self.k * (-(e * e)).exp() / (SQRTPI * rc * rc)
    }
}

/// The arithmetic of [`XtbErf`] and of the cap: `f64`, or a caller's forward-mode dual number
/// (GFN-FF builds its analytic Hessian by evaluating its CN with dual numbers).
pub trait Scalar:
    Copy
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Add<f64, Output = Self>
    + Sub<f64, Output = Self>
    + Mul<f64, Output = Self>
    + Div<f64, Output = Self>
{
    /// The constant `v`.
    fn cst(v: f64) -> Self;
    /// e^x.
    fn exp(self) -> Self;
    /// ln x.
    fn ln(self) -> Self;
    /// The error function ([`erf`] for `f64`).
    fn erf(self) -> Self;
}

impl Scalar for f64 {
    #[inline]
    fn cst(v: f64) -> Self {
        v
    }
    #[inline]
    fn exp(self) -> Self {
        f64::exp(self)
    }
    #[inline]
    fn ln(self) -> Self {
        f64::ln(self)
    }
    #[inline]
    fn erf(self) -> Self {
        erf(self)
    }
}

/// The error-function counting function of xtb's GFN-FF (`gfnff_dlogcoord`, `create_erfCN` and
/// `create_derfCN`): f = ½ (1 + erf(kn (r − rc)/rc)) with kn < 0 and the relative distance
/// (r − rc)/rc formed first (its own rounding, not [`Erf`]'s); the derivative uses xtb's
/// truncated √π, part of xtb's gradient. Generic over [`Scalar`] for GFN-FF's dual numbers.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct XtbErf {
    /// Steepness, negative.
    pub kn: f64,
    /// The √π of the derivative.
    pub sqrtpi: f64,
}

impl XtbErf {
    /// GFN-FF: kn = −7.5 and xtb's √π = 1.77245385091.
    pub const GFNFF: XtbErf = XtbErf { kn: -7.5, sqrtpi: 1.77245385091 };

    /// f(r; rc) (`create_erfCN`).
    #[inline]
    pub fn count_of<T: Scalar>(&self, r: T, rc: f64) -> T {
        let dr = (r - rc) / rc;
        ((dr * self.kn).erf() + 1.0) * 0.5
    }

    /// xtb's df/dr (`create_derfCN`, with the truncated √π).
    #[inline]
    pub fn dcount_of<T: Scalar>(&self, r: T, rc: f64) -> T {
        let dr = (r - rc) / rc;
        let e = (-(dr * (self.kn * self.kn)) * dr).exp();
        e * (self.kn / self.sqrtpi) / rc
    }
}

impl Count for XtbErf {
    #[inline]
    fn count(&self, r: f64, rc: f64) -> f64 {
        self.count_of(r, rc)
    }

    #[inline]
    fn dcount(&self, r: f64, rc: f64) -> f64 {
        self.dcount_of(r, rc)
    }
}

/// The logarithmic cap of a coordination number at `cnmax` (multicharge's EEQ CN, xtb's GFN-FF
/// `create_logCN`): ln(1 + e^cnmax) − ln(1 + e^(cnmax − cn)).
#[inline]
pub fn log_cap<T: Scalar>(cn: T, cnmax: f64) -> T {
    let a = (1.0 + cnmax.exp()).ln();
    T::cst(a) - ((T::cst(cnmax) - cn).exp() + 1.0).ln()
}

/// d[`log_cap`]/dcn = e^cnmax / (e^cn + e^cnmax) (xtb's `create_dlogCN`).
#[inline]
pub fn dlog_cap<T: Scalar>(cn: T, cnmax: f64) -> T {
    let e = cnmax.exp();
    T::cst(e) / (cn.exp() + e)
}

/// Which pairs count, by distance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Cutoff {
    /// Every pair.
    None,
    /// Pairs with r² ≤ the given squared cutoff (bohr²), tested on r² as dftd4, tblite and xtb
    /// do. The caller forms the square as its program does.
    Squared(f64),
    /// Pairs with r ≤ the given cutoff (bohr), tested on r = √(r²) as xtb's GFN-FF does.
    Distance(f64),
}

impl Cutoff {
    /// True if a pair at squared distance `r2` lies beyond the cutoff.
    #[inline]
    pub fn excludes(self, r2: f64) -> bool {
        match self {
            Cutoff::None => false,
            Cutoff::Squared(c2) => r2 > c2,
            Cutoff::Distance(c) => r2.sqrt() > c,
        }
    }
}

/// A weight w_AB of each pair's count.
pub trait Weight {
    /// w_ij.
    fn weight(&self, i: usize, j: usize) -> f64;
}

/// No pair weight (w = 1; multiplying by 1 is exact, so the counts are the counting function's
/// bits).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Unweighted;

impl Weight for Unweighted {
    #[inline]
    fn weight(&self, _: usize, _: usize) -> f64 {
        1.0
    }
}

impl<F: Fn(usize, usize) -> f64> Weight for F {
    #[inline]
    fn weight(&self, i: usize, j: usize) -> f64 {
        self(i, j)
    }
}

/// A cutoff factor s(r) multiplying each count.
pub trait Switch {
    /// The count `c` at distance `r` times s(r).
    fn value(&self, r: f64, c: f64) -> f64;
    /// d(c s)/dr from dc/dr and the count `c` (evaluated only when s is not constant).
    fn derivative(&self, r: f64, dc: f64, c: impl FnOnce() -> f64) -> f64;
}

/// No cutoff factor (s = 1).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct NoSwitch;

impl Switch for NoSwitch {
    #[inline]
    fn value(&self, _: f64, c: f64) -> f64 {
        c
    }
    #[inline]
    fn derivative(&self, _: f64, dc: f64, _: impl FnOnce() -> f64) -> f64 {
        dc
    }
}

/// A smooth cutoff factor: the closure returns s(r) and ds/dr; the count is c·s and its
/// derivative dc·s + c·ds.
#[derive(Clone, Copy, Debug)]
pub struct Smooth<F>(pub F);

impl<F: Fn(f64) -> (f64, f64)> Switch for Smooth<F> {
    #[inline]
    fn value(&self, r: f64, c: f64) -> f64 {
        c * (self.0)(r).0
    }
    #[inline]
    fn derivative(&self, r: f64, dc: f64, c: impl FnOnce() -> f64) -> f64 {
        let (s, ds) = (self.0)(r);
        dc * s + c() * ds
    }
}

/// The order of the chain-rule products in the gradient contraction Σ_A (dE/dCN_A) dCN_A/dR.
/// Per pair, with d = d(w f s)/dr, the pair vector R_ij = R_i − R_j, its length r and
/// σ = dE/dCN_i + dE/dCN_j:
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GradientOrder {
    /// t = (d σ)/r, then g_i += t R_ij and g_j −= t R_ij (xtb and axtb's GFN2-xTB, g-xTB).
    #[default]
    Scaled,
    /// g_i += ((d R_ij,k)/r) σ per component: the Jacobian entry first, as dftd4 and tblite
    /// contract their `dcndr` with dE/dCN.
    Jacobian,
}

/// A coordination number: counting function `C`, pair radius `R`, pair weight `W` and cutoff
/// factor `S`, with the pair test and the gradient conventions (see the [module](self)).
#[derive(Clone, Copy, Debug)]
pub struct CnModel<C, R, W = Unweighted, S = NoSwitch> {
    /// The counting function.
    pub count: C,
    /// The pair radius rc(i, j) (bohr).
    pub radius: R,
    /// The pair weight.
    pub weight: W,
    /// The cutoff factor.
    pub switch: S,
    /// Which pairs count by distance.
    pub cutoff: Cutoff,
    /// Pairs with r² below this are skipped (bohr²; dftd4, tblite and xtb: 1e-12; 0: none).
    pub min_r2: f64,
    /// The order of the gradient contraction.
    pub order: GradientOrder,
    /// Skip pairs whose weights dE/dCN_i + dE/dCN_j (gradient) or w_i + w_j (Hessian) sum to
    /// exactly zero, as dftd4 and g-xTB 2.0.1 do.
    pub skip_zero_weight: bool,
}

impl<C, R> CnModel<C, R> {
    /// The counting function `count` with pair radii `radius`, over every pair, without weight,
    /// cutoff factor or minimum distance, contracted in the [`GradientOrder::Scaled`] order.
    pub fn new(count: C, radius: R) -> Self {
        CnModel {
            count,
            radius,
            weight: Unweighted,
            switch: NoSwitch,
            cutoff: Cutoff::None,
            min_r2: 0.0,
            order: GradientOrder::Scaled,
            skip_zero_weight: false,
        }
    }
}

impl<C, R, W, S> CnModel<C, R, W, S> {
    /// With pair weights `weight`.
    pub fn with_weight<W2>(self, weight: W2) -> CnModel<C, R, W2, S> {
        let CnModel { count, radius, switch, cutoff, min_r2, order, skip_zero_weight, .. } = self;
        CnModel { count, radius, weight, switch, cutoff, min_r2, order, skip_zero_weight }
    }

    /// With the cutoff factor `switch`.
    pub fn with_switch<S2>(self, switch: S2) -> CnModel<C, R, W, S2> {
        let CnModel { count, radius, weight, cutoff, min_r2, order, skip_zero_weight, .. } = self;
        CnModel { count, radius, weight, switch, cutoff, min_r2, order, skip_zero_weight }
    }

    /// With the cutoff `cutoff`.
    pub fn with_cutoff(mut self, cutoff: Cutoff) -> Self {
        self.cutoff = cutoff;
        self
    }

    /// Skipping pairs with r² < `min_r2` (bohr²).
    pub fn with_min_r2(mut self, min_r2: f64) -> Self {
        self.min_r2 = min_r2;
        self
    }

    /// With the gradient contraction order `order`.
    pub fn with_order(mut self, order: GradientOrder) -> Self {
        self.order = order;
        self
    }

    /// Skipping pairs whose weights sum to zero in the derivatives.
    pub fn skipping_zero_weights(mut self) -> Self {
        self.skip_zero_weight = true;
        self
    }
}

impl<C: Count, R: Fn(usize, usize) -> f64, W: Weight, S: Switch> CnModel<C, R, W, S> {
    /// The pair (i, j) of `pos` (bohr) if it counts: R_i − R_j and r².
    #[inline]
    pub fn pair(&self, pos: &[[f64; 3]], i: usize, j: usize) -> Option<([f64; 3], f64)> {
        let v = sub3(pos[i], pos[j]);
        let r2 = norm2_sq(v);
        if self.cutoff.excludes(r2) || r2 < self.min_r2 {
            return None;
        }
        Some((v, r2))
    }

    /// The count w f s of a counted pair (i, j) at distance `r`.
    #[inline]
    pub fn pair_count(&self, i: usize, j: usize, r: f64) -> f64 {
        let c = self.weight.weight(i, j) * self.count.count(r, (self.radius)(i, j));
        self.switch.value(r, c)
    }

    /// d(w f s)/dr of a counted pair (i, j) at distance `r`.
    #[inline]
    pub fn pair_dcount(&self, i: usize, j: usize, r: f64) -> f64 {
        let (w, rc) = (self.weight.weight(i, j), (self.radius)(i, j));
        self.switch.derivative(r, w * self.count.dcount(r, rc), || w * self.count.count(r, rc))
    }

    /// The coordination numbers of the atoms at `pos` (bohr).
    pub fn compute(&self, pos: &[[f64; 3]]) -> Vec<f64> {
        let mut cn = vec![0.0; pos.len()];
        self.compute_into(pos, &mut cn);
        cn
    }

    /// The coordination numbers of the atoms at `pos` (bohr), written to `cn` (one per atom).
    pub fn compute_into(&self, pos: &[[f64; 3]], cn: &mut [f64]) {
        assert_eq!(cn.len(), pos.len());
        cn.fill(0.0);
        for i in 0..pos.len() {
            for j in 0..i {
                if let Some((_, r2)) = self.pair(pos, i, j) {
                    let c = self.pair_count(i, j, r2.sqrt());
                    cn[i] += c;
                    cn[j] += c;
                }
            }
        }
    }

    /// Add Σ_A dedcn_A dCN_A/dR to `g`, in the order [`CnModel::order`].
    pub fn add_gradient(&self, pos: &[[f64; 3]], dedcn: &[f64], g: &mut [[f64; 3]]) {
        let n = pos.len();
        assert!(dedcn.len() >= n && g.len() >= n);
        for i in 0..n {
            for j in 0..i {
                let s = dedcn[i] + dedcn[j];
                if self.skip_zero_weight && s == 0.0 {
                    continue;
                }
                let Some((v, r2)) = self.pair(pos, i, j) else {
                    continue;
                };
                let r = r2.sqrt();
                let d = self.pair_dcount(i, j, r);
                match self.order {
                    GradientOrder::Scaled => {
                        let t = d * s / r;
                        for k in 0..3 {
                            g[i][k] += t * v[k];
                            g[j][k] -= t * v[k];
                        }
                    }
                    GradientOrder::Jacobian => {
                        for k in 0..3 {
                            let t = d * v[k] / r * s;
                            g[i][k] += t;
                            g[j][k] -= t;
                        }
                    }
                }
            }
        }
    }

    /// The Jacobian dCN_A/dR_x (row-major `nat x 3 nat`).
    pub fn jacobian(&self, pos: &[[f64; 3]]) -> Vec<f64> {
        let n = pos.len();
        let n3 = 3 * n;
        let mut jac = vec![0.0; n * n3];
        for i in 0..n {
            for j in 0..i {
                let Some((v, r2)) = self.pair(pos, i, j) else {
                    continue;
                };
                let r = r2.sqrt();
                let dc = self.pair_dcount(i, j, r) / r;
                for k in 0..3 {
                    let d = dc * v[k];
                    jac[i * n3 + 3 * i + k] += d;
                    jac[i * n3 + 3 * j + k] -= d;
                    jac[j * n3 + 3 * i + k] += d;
                    jac[j * n3 + 3 * j + k] -= d;
                }
            }
        }
        jac
    }
}

impl<C: Count2, R: Fn(usize, usize) -> f64, W: Weight> CnModel<C, R, W, NoSwitch> {
    /// Add Σ_A w_A d²CN_A/dR dR to `hess` (row-major `3 nat x 3 nat`): per pair the radial
    /// Hessian with dE/dr = (w_i + w_j) w_ij f′ and d²E/dr² = (w_i + w_j) w_ij f″.
    pub fn add_hessian(&self, pos: &[[f64; 3]], w: &[f64], hess: &mut [f64]) {
        let n = pos.len();
        let n3 = 3 * n;
        assert!(w.len() >= n && hess.len() >= n3 * n3);
        for i in 0..n {
            for j in 0..i {
                let wij = w[i] + w[j];
                if self.skip_zero_weight && wij == 0.0 {
                    continue;
                }
                let Some((v, r2)) = self.pair(pos, i, j) else {
                    continue;
                };
                let r = r2.sqrt();
                let rc = (self.radius)(i, j);
                let s = wij * self.weight.weight(i, j);
                add_radial_hessian(hess, n3, i, j, v, r, s * self.count.dcount(r, rc), s * self.count.d2count(r, rc));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn positions() -> Vec<[f64; 3]> {
        vec![[0.0, 0.0, 0.0], [1.9, 0.3, -0.2], [-1.1, 1.7, 0.4], [0.4, -1.2, 1.8], [3.6, 1.1, 0.9], [-2.9, -0.8, -1.3]]
    }

    #[test]
    fn counting_derivatives_match_finite_differences() {
        let h = 1e-5;
        for (r, rc) in [(2.0, 2.4), (3.1, 2.4), (1.4, 3.0), (5.0, 3.3)] {
            let fd = |f: &dyn Fn(f64) -> f64| (f(r + h) - f(r - h)) / (2.0 * h);
            let c = DoubleExp::GFN2;
            assert!((fd(&|x| c.count(x, rc)) - c.dcount(r, rc)).abs() < 1e-8);
            assert!((fd(&|x| c.dcount(x, rc)) - c.d2count(r, rc)).abs() < 1e-7);
            let c = Erf::D4;
            assert!((fd(&|x| c.count(x, rc)) - c.dcount(r, rc)).abs() < 1e-8);
            assert!((fd(&|x| c.dcount(x, rc)) - c.d2count(r, rc)).abs() < 1e-7);
            // xtb's derivative carries its truncated sqrt(pi): relative error ~2e-12
            let c = XtbErf::GFNFF;
            let d = c.dcount(r, rc);
            assert!((fd(&|x| c.count(x, rc)) - d).abs() < 1e-8 * (1.0 + d.abs()));
        }
    }

    #[test]
    fn erf_sign_conventions_give_the_same_bits() {
        // the EEQ guess and g-xTB 1.1 write the argument as (-k)(r - rc)/rc
        for k in [7.5, 3.75, 1.8217544049] {
            let c = Erf { k };
            for i in 0..400 {
                let r = 0.3 + 0.05 * i as f64;
                for rc in [1.1, 2.4, 3.7] {
                    let other = 0.5 * (1.0 + erf(-k * (r - rc) / rc));
                    assert_eq!(c.count(r, rc).to_bits(), other.to_bits());
                }
            }
        }
    }

    #[test]
    fn log_cap_matches_its_derivative() {
        for cn in [0.0, 1.3, 4.0, 7.9, 12.0] {
            let h = 1e-6;
            let fd = (log_cap(cn + h, 4.4) - log_cap(cn - h, 4.4)) / (2.0 * h);
            assert!((fd - dlog_cap(cn, 4.4)).abs() < 1e-9);
            // multicharge's form
            let mc = (1.0 + 8.0f64.exp()).ln() - (1.0 + (8.0 - cn).exp()).ln();
            assert_eq!(log_cap(cn, 8.0).to_bits(), mc.to_bits());
        }
    }

    #[test]
    fn derivatives_of_a_model_agree() {
        let pos = positions();
        let n = pos.len();
        let rad = [1.2, 2.0, 1.6, 1.4, 2.4, 1.0];
        let rc = |i: usize, j: usize| rad[i] + rad[j];
        let en = |i: usize, j: usize| 1.0 + 0.1 * (i as f64 - j as f64).abs();
        let dedcn = [0.3, -0.2, 0.0, 0.7, 0.1, -0.4];
        for order in [GradientOrder::Scaled, GradientOrder::Jacobian] {
            let model = CnModel::new(Erf::D4, rc)
                .with_weight(en)
                .with_cutoff(Cutoff::Squared(16.0))
                .with_min_r2(1e-12)
                .with_order(order);
            let jac = model.jacobian(&pos);
            let mut g = vec![[0.0; 3]; n];
            model.add_gradient(&pos, &dedcn, &mut g);
            let mut hess = vec![0.0; 9 * n * n];
            model.add_hessian(&pos, &dedcn, &mut hess);
            let step = 1e-5;
            for x in 0..3 * n {
                let shifted = |s: f64| {
                    let mut p = pos.clone();
                    p[x / 3][x % 3] += s;
                    p
                };
                let (cp, cm) = (model.compute(&shifted(step)), model.compute(&shifted(-step)));
                let mut gx = 0.0;
                for a in 0..n {
                    let d = (cp[a] - cm[a]) / (2.0 * step);
                    assert!((d - jac[a * 3 * n + x]).abs() < 1e-8, "jacobian");
                    gx += dedcn[a] * jac[a * 3 * n + x];
                }
                assert!((gx - g[x / 3][x % 3]).abs() < 1e-12, "gradient");
                let (mut gp, mut gm) = (vec![[0.0; 3]; n], vec![[0.0; 3]; n]);
                model.add_gradient(&shifted(step), &dedcn, &mut gp);
                model.add_gradient(&shifted(-step), &dedcn, &mut gm);
                for y in 0..3 * n {
                    let d = (gp[y / 3][y % 3] - gm[y / 3][y % 3]) / (2.0 * step);
                    assert!((d - hess[x * 3 * n + y]).abs() < 1e-7, "hessian");
                }
            }
        }
    }

    #[test]
    fn smooth_cutoff_factor_enters_value_and_derivative() {
        let pos = positions();
        let n = pos.len();
        let rc = |i: usize, j: usize| 1.5 + 0.1 * (i + j) as f64;
        let s = |r: f64| (1.0 / (1.0 + r * r), -2.0 * r / ((1.0 + r * r) * (1.0 + r * r)));
        let model = CnModel::new(DoubleExp::GFN2, rc).with_switch(Smooth(s)).with_cutoff(Cutoff::Distance(5.0));
        let jac = model.jacobian(&pos);
        let step = 1e-5;
        for x in 0..3 * n {
            let mut p = pos.clone();
            p[x / 3][x % 3] += step;
            let cp = model.compute(&p);
            p[x / 3][x % 3] -= 2.0 * step;
            let cm = model.compute(&p);
            for a in 0..n {
                assert!(((cp[a] - cm[a]) / (2.0 * step) - jac[a * 3 * n + x]).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn cutoffs_select_pairs() {
        let pos = [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 1e-7]];
        let m = CnModel::new(Erf::D4, |_: usize, _: usize| 2.0);
        assert!(m.pair(&pos, 1, 0).is_some() && m.pair(&pos, 2, 0).is_some());
        let m = m.with_min_r2(1e-12).with_cutoff(Cutoff::Squared(4.0));
        assert!(m.pair(&pos, 1, 0).is_none() && m.pair(&pos, 2, 0).is_none() && m.pair(&pos, 2, 1).is_none());
        let m = m.with_cutoff(Cutoff::Distance(3.0));
        assert!(m.pair(&pos, 1, 0).is_some());
    }
}
