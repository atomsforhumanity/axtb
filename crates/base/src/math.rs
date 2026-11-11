//! The error function of the reference programs, 3-vector helpers and the Cartesian Hessian
//! block of a pair term.

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

/// Add the Cartesian Hessian of a pair term `E(r)` with `dE/dr = e1` and `d²E/dr² = e2` to `h`
/// (row-major `n3 x n3`): with `u = v/r`, the 3x3 block `e2 u uᵀ + (e1/r) (1 - u uᵀ)` is added to
/// the (i, i) and (j, j) blocks and subtracted from the (i, j) and (j, i) blocks.
/// `v = R_i - R_j`, `r = |v|`.
#[inline]
pub fn add_radial_hessian(h: &mut [f64], n3: usize, i: usize, j: usize, v: [f64; 3], r: f64, e1: f64, e2: f64) {
    let u = [v[0] / r, v[1] / r, v[2] / r];
    for a in 0..3 {
        for b in 0..3 {
            let d = if a == b { 1.0 } else { 0.0 };
            let blk = e2 * u[a] * u[b] + e1 / r * (d - u[a] * u[b]);
            h[(3 * i + a) * n3 + 3 * i + b] += blk;
            h[(3 * j + a) * n3 + 3 * j + b] += blk;
            h[(3 * i + a) * n3 + 3 * j + b] -= blk;
            h[(3 * j + a) * n3 + 3 * i + b] -= blk;
        }
    }
}
