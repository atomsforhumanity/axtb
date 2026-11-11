//! Scalar type of the energy and gradient kernels.
//!
//! The kernels are written once, generic over [`Real`]: with `f64` they are the energy
//! and analytic gradient (a literal port of xtb's formulas, operation by operation); with
//! [`Dual`] (forward-mode automatic differentiation) the same code gives the exact
//! derivative of xtb's analytic gradient, which is how the analytic Hessian is built.
//! Multiplication and addition are commutative in IEEE arithmetic, so writing `x * 2.0`
//! for the Fortran `2*x` keeps every result bit-identical; association order is kept.

use std::fmt::Debug;
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

/// Error function of the reference programs (libm; from `axtb-base`).
pub use axtb_base::math::erf;

pub trait Real:
    Copy
    + Debug
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + Add<f64, Output = Self>
    + Sub<f64, Output = Self>
    + Mul<f64, Output = Self>
    + Div<f64, Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + Send
    + Sync
{
    /// True for dual numbers: kernels then evaluate a few factors in an algebraically equal
    /// form whose derivative stays accurate where xtb's form is 0/0 (see `terms`); the f64
    /// path keeps xtb's operations.
    const DUAL: bool;
    fn cst(v: f64) -> Self;
    fn val(self) -> f64;
    fn sqrt(self) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    fn powf(self, e: f64) -> Self;
    fn powi(self, e: i32) -> Self;
    fn sin(self) -> Self;
    fn cos(self) -> Self;
    fn acos(self) -> Self;
    fn asin(self) -> Self;
    fn erf(self) -> Self;
    /// `c - self`
    #[inline]
    fn rsub(self, c: f64) -> Self {
        Self::cst(c) - self
    }
    /// `c / self`
    #[inline]
    fn rdiv(self, c: f64) -> Self {
        Self::cst(c) / self
    }
}

impl Real for f64 {
    const DUAL: bool = false;
    #[inline]
    fn cst(v: f64) -> Self {
        v
    }
    #[inline]
    fn val(self) -> f64 {
        self
    }
    #[inline]
    fn sqrt(self) -> Self {
        f64::sqrt(self)
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
    fn powf(self, e: f64) -> Self {
        f64::powf(self, e)
    }
    #[inline]
    fn powi(self, e: i32) -> Self {
        f64::powi(self, e)
    }
    #[inline]
    fn sin(self) -> Self {
        f64::sin(self)
    }
    #[inline]
    fn cos(self) -> Self {
        f64::cos(self)
    }
    #[inline]
    fn acos(self) -> Self {
        f64::acos(self)
    }
    #[inline]
    fn asin(self) -> Self {
        f64::asin(self)
    }
    #[inline]
    fn erf(self) -> Self {
        erf(self)
    }
}

/// Forward-mode dual number `v + d ε` (one derivative direction).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Dual {
    pub v: f64,
    pub d: f64,
}

impl Dual {
    #[inline]
    pub fn new(v: f64, d: f64) -> Self {
        Dual { v, d }
    }
}

impl Add for Dual {
    type Output = Dual;
    #[inline]
    fn add(self, o: Dual) -> Dual {
        Dual::new(self.v + o.v, self.d + o.d)
    }
}
impl Sub for Dual {
    type Output = Dual;
    #[inline]
    fn sub(self, o: Dual) -> Dual {
        Dual::new(self.v - o.v, self.d - o.d)
    }
}
impl Mul for Dual {
    type Output = Dual;
    #[inline]
    fn mul(self, o: Dual) -> Dual {
        Dual::new(self.v * o.v, self.d * o.v + self.v * o.d)
    }
}
impl Div for Dual {
    type Output = Dual;
    #[inline]
    fn div(self, o: Dual) -> Dual {
        let q = self.v / o.v;
        Dual::new(q, (self.d - q * o.d) / o.v)
    }
}
impl Neg for Dual {
    type Output = Dual;
    #[inline]
    fn neg(self) -> Dual {
        Dual::new(-self.v, -self.d)
    }
}
impl Add<f64> for Dual {
    type Output = Dual;
    #[inline]
    fn add(self, o: f64) -> Dual {
        Dual::new(self.v + o, self.d)
    }
}
impl Sub<f64> for Dual {
    type Output = Dual;
    #[inline]
    fn sub(self, o: f64) -> Dual {
        Dual::new(self.v - o, self.d)
    }
}
impl Mul<f64> for Dual {
    type Output = Dual;
    #[inline]
    fn mul(self, o: f64) -> Dual {
        Dual::new(self.v * o, self.d * o)
    }
}
impl Div<f64> for Dual {
    type Output = Dual;
    #[inline]
    fn div(self, o: f64) -> Dual {
        Dual::new(self.v / o, self.d / o)
    }
}
impl AddAssign for Dual {
    #[inline]
    fn add_assign(&mut self, o: Dual) {
        *self = *self + o;
    }
}
impl SubAssign for Dual {
    #[inline]
    fn sub_assign(&mut self, o: Dual) {
        *self = *self - o;
    }
}
impl MulAssign for Dual {
    #[inline]
    fn mul_assign(&mut self, o: Dual) {
        *self = *self * o;
    }
}

impl Real for Dual {
    const DUAL: bool = true;
    #[inline]
    fn cst(v: f64) -> Self {
        Dual::new(v, 0.0)
    }
    #[inline]
    fn val(self) -> f64 {
        self.v
    }
    #[inline]
    fn sqrt(self) -> Self {
        let s = self.v.sqrt();
        Dual::new(s, if s != 0.0 { self.d / (2.0 * s) } else { 0.0 })
    }
    #[inline]
    fn exp(self) -> Self {
        let e = self.v.exp();
        Dual::new(e, self.d * e)
    }
    #[inline]
    fn ln(self) -> Self {
        Dual::new(self.v.ln(), self.d / self.v)
    }
    #[inline]
    fn powf(self, e: f64) -> Self {
        if self.v == 0.0 {
            return Dual::new(0.0f64.powf(e), 0.0);
        }
        let p = self.v.powf(e);
        Dual::new(p, self.d * e * p / self.v)
    }
    #[inline]
    fn powi(self, e: i32) -> Self {
        let p = self.v.powi(e);
        let dp = if e == 0 { 0.0 } else { e as f64 * self.v.powi(e - 1) };
        Dual::new(p, self.d * dp)
    }
    #[inline]
    fn sin(self) -> Self {
        Dual::new(self.v.sin(), self.d * self.v.cos())
    }
    #[inline]
    fn cos(self) -> Self {
        Dual::new(self.v.cos(), -self.d * self.v.sin())
    }
    #[inline]
    fn acos(self) -> Self {
        let s = (1.0 - self.v * self.v).sqrt();
        Dual::new(self.v.acos(), if s > 0.0 { -self.d / s } else { 0.0 })
    }
    #[inline]
    fn asin(self) -> Self {
        let s = (1.0 - self.v * self.v).sqrt();
        Dual::new(self.v.asin(), if s > 0.0 { self.d / s } else { 0.0 })
    }
    #[inline]
    fn erf(self) -> Self {
        let two_over_sqrtpi = 2.0 / std::f64::consts::PI.sqrt();
        Dual::new(erf(self.v), self.d * two_over_sqrtpi * (-self.v * self.v).exp())
    }
}

/// 3-vector helpers on generic scalars.
pub type V3<T> = [T; 3];

#[inline]
pub fn sub<T: Real>(a: V3<T>, b: V3<T>) -> V3<T> {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
#[inline]
pub fn scale<T: Real>(a: V3<T>, s: T) -> V3<T> {
    [a[0] * s, a[1] * s, a[2] * s]
}
/// `sum(a**2)` in index order
#[inline]
pub fn dot<T: Real>(a: V3<T>, b: V3<T>) -> T {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
#[inline]
pub fn sq<T: Real>(a: V3<T>) -> T {
    a[0] * a[0] + a[1] * a[1] + a[2] * a[2]
}
/// Fortran `NORM2` (gfortran/ifort compute sqrt of the sum of squares for 3 elements)
#[inline]
pub fn norm2<T: Real>(a: V3<T>) -> T {
    sq(a).sqrt()
}
#[inline]
pub fn cross<T: Real>(a: V3<T>, b: V3<T>) -> V3<T> {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
