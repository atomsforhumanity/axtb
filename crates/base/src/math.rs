//! The error function of the reference programs and 3-vector helpers.

extern "C" {
    // C99 libm error function (the routine gfortran's ERF intrinsic calls).
    #[link_name = "erf"]
    fn libm_erf(x: f64) -> f64;
}

/// The error function of the C library (libm), the routine gfortran's `ERF` intrinsic calls, so
/// that terms with an error function keep the reference programs' bits. Every crate of the
/// workspace uses this one.
#[inline]
pub fn erf(x: f64) -> f64 {
    // SAFETY: a pure libm function of a double, without side effects
    unsafe { libm_erf(x) }
}

/// Difference vector `a - b`.
#[inline]
pub fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Squared norm, summed in index order.
#[inline]
pub fn norm2_sq(v: [f64; 3]) -> f64 {
    v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
}

/// Dot product, summed in index order.
#[inline]
pub fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
