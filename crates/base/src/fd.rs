// SPDX-License-Identifier: LGPL-3.0-or-later
//! Finite differences, written once for every method: the central-difference stencils, the
//! columns of a derivative over many coordinates, symmetrization, and the numerical Hessian
//! from central differences of an analytic gradient.
//!
//! The pieces keep the arithmetic of the code they replaced, so that results stay the same to
//! the last bit:
//!
//! - [`derivative`] calls `f` at the offsets of a [`Stencil`] in a fixed order (h, −h, and for
//!   [`Stencil::Central4`] 2h, −2h) and combines the values element by element with
//!   [`central2`] or [`central4`]; the caller moves its own coordinate by the offset
//!   (`x + d`), so the displaced coordinate is formed as before;
//! - [`columns`] runs one closure per coordinate, sequentially or on the current rayon pool,
//!   each call on one thread and its result at its index: the output is the same for every
//!   thread count as long as the closure itself runs sequentially;
//! - [`symmetrize`] replaces a matrix by `(H + Hᵀ)/2` with [`symmetric_part`] and reports the
//!   largest asymmetry; since `a + b = b + a` exactly, it does not matter whether a caller
//!   stored its columns as rows or as columns;
//! - [`numerical_hessian`] is the semi-numerical Hessian of all methods: central differences of
//!   the analytic gradient and dipole over the 3N coordinates, parallel over coordinates. The
//!   two ways the methods place their points are the variants of [`Displacement`].
//!
//! Sources: the textbook two- and four-point central differences, and the finite-difference
//! code of the LGPL crates `axtb` (GFN2-xTB) and `axtb-gfnff` that moved here. The GPL g-xTB
//! crate calls these routines from its own code; nothing here comes from it.

use rayon::prelude::*;
use std::convert::Infallible;

/// A central-difference stencil along one coordinate with step h.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stencil {
    /// Values at x + h and x − h, in that order; f′ ≈ [`central2`], error O(h²).
    Central2,
    /// Values at x + h, x − h, x + 2h and x − 2h, in that order; f′ ≈ [`central4`], error O(h⁴).
    Central4,
}

/// The two-point central difference `(f(x+h) − f(x−h)) / (2h)` of the values at x + h and
/// x − h, element by element.
#[inline]
pub fn central2<V: Values>(plus: &V, minus: &V, h: f64) -> V {
    V::zip_with([plus, minus], &|[p, m]| (p - m) / (2.0 * h))
}

/// The four-point central difference `(8 (f(x+h) − f(x−h)) − (f(x+2h) − f(x−2h))) / (12h)`,
/// element by element.
#[inline]
pub fn central4<V: Values>(plus1: &V, minus1: &V, plus2: &V, minus2: &V, h: f64) -> V {
    V::zip_with([plus1, minus1, plus2, minus2], &|[a, b, c, d]| (8.0 * (a - b) - (c - d)) / (12.0 * h))
}

/// Values that finite differences combine element by element: numbers, arrays and vectors of
/// them, and pairs (e.g. a gradient with a dipole).
pub trait Values: Sized {
    /// `f` applied to the corresponding elements of `K` values of the same shape.
    fn zip_with<const K: usize>(values: [&Self; K], f: &impl Fn([f64; K]) -> f64) -> Self;
}

impl Values for f64 {
    #[inline]
    fn zip_with<const K: usize>(values: [&Self; K], f: &impl Fn([f64; K]) -> f64) -> Self {
        f(values.map(|v| *v))
    }
}

impl<T: Values, const N: usize> Values for [T; N] {
    fn zip_with<const K: usize>(values: [&Self; K], f: &impl Fn([f64; K]) -> f64) -> Self {
        std::array::from_fn(|i| T::zip_with(values.map(|v| &v[i]), f))
    }
}

impl<T: Values> Values for Vec<T> {
    fn zip_with<const K: usize>(values: [&Self; K], f: &impl Fn([f64; K]) -> f64) -> Self {
        let n = values.first().map_or(0, |v| v.len());
        debug_assert!(values.iter().all(|v| v.len() == n), "finite differences of values of different lengths");
        (0..n).map(|i| T::zip_with(values.map(|v| &v[i]), f)).collect()
    }
}

impl<A: Values, B: Values> Values for (A, B) {
    fn zip_with<const K: usize>(values: [&Self; K], f: &impl Fn([f64; K]) -> f64) -> Self {
        (A::zip_with(values.map(|v| &v.0), f), B::zip_with(values.map(|v| &v.1), f))
    }
}

/// The derivative of `f` along one coordinate by `stencil` with step `h`: `f(d)` is the value
/// with the coordinate moved by the offset `d` (h, −h, then for [`Stencil::Central4`] 2h, −2h,
/// in that order), and the values are combined element by element.
pub fn derivative<V: Values>(stencil: Stencil, h: f64, mut f: impl FnMut(f64) -> V) -> V {
    let Ok(v) = try_derivative(stencil, h, |d| Ok::<V, Infallible>(f(d)));
    v
}

/// [`derivative`] of a fallible `f`: the first error in the order of the offsets.
pub fn try_derivative<V: Values, E>(stencil: Stencil, h: f64, mut f: impl FnMut(f64) -> Result<V, E>) -> Result<V, E> {
    Ok(match stencil {
        Stencil::Central2 => {
            let plus = f(h)?;
            let minus = f(-h)?;
            central2(&plus, &minus, h)
        }
        Stencil::Central4 => {
            let (p1, m1, p2, m2) = (f(h)?, f(-h)?, f(2.0 * h)?, f(-2.0 * h)?);
            central4(&p1, &m1, &p2, &m2, h)
        }
    })
}

/// `f(y)` for every coordinate `y` in `0..n`, collected in index order. With `parallel` the
/// calls are spread over the current rayon pool (the caller wraps the call in its pool's
/// `install`); without it they run in order on the calling thread. Each call runs entirely on
/// one thread, so if `f` does its own work sequentially the results are identical for any
/// thread count.
pub fn columns<T: Send>(n: usize, parallel: bool, f: impl Fn(usize) -> T + Sync + Send) -> Vec<T> {
    if parallel {
        (0..n).into_par_iter().map(f).collect()
    } else {
        (0..n).map(f).collect()
    }
}

/// The row-major `rows x cols.len()` matrix whose column y is `cols[y]` (of length `rows`).
pub fn from_columns<C: AsRef<[f64]>>(cols: &[C], rows: usize) -> Vec<f64> {
    let n = cols.len();
    let mut out = vec![0.0; rows * n];
    for (y, col) in cols.iter().enumerate() {
        let col = col.as_ref();
        for x in 0..rows {
            out[x * n + y] = col[x];
        }
    }
    out
}

/// Element (i, j) of the symmetric part `(H + Hᵀ)/2` of a row-major n x n matrix `h`.
#[inline]
pub fn symmetric_part(h: &[f64], n: usize, i: usize, j: usize) -> f64 {
    0.5 * (h[i * n + j] + h[j * n + i])
}

/// Replace a row-major n x n matrix by its symmetric part (the diagonal is kept); returns the
/// largest |H_ij − H_ji| before.
pub fn symmetrize(h: &mut [f64], n: usize) -> f64 {
    let mut asym: f64 = 0.0;
    for i in 0..n {
        for j in 0..i {
            let (a, b) = (h[i * n + j], h[j * n + i]);
            asym = asym.max((a - b).abs());
            let m = symmetric_part(h, n, i, j);
            h[i * n + j] = m;
            h[j * n + i] = m;
        }
    }
    asym
}

/// Where [`numerical_hessian`] places the two points of a coordinate x.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Displacement {
    /// x + h and x − h, each formed from x (the GFN-FF calculator).
    FromReference,
    /// x + h, then (x + h) − 2h: forward by the step and back by twice the step (the GFN2-xTB
    /// and g-xTB calculators). The second point can differ from x − h in the last bit.
    ForwardBack,
}

/// A Hessian from [`numerical_hessian`].
#[derive(Clone, Debug)]
pub struct NumericalHessian {
    /// Row-major 3N x 3N matrix of second derivatives, symmetrized ([`symmetrize`]).
    pub hessian: Vec<f64>,
    /// Largest |H_ij − H_ji| before symmetrization.
    pub max_asymmetry: f64,
    /// Derivatives ∂μ/∂x of the dipole, one row per coordinate x = 3·atom + k.
    pub dipole_derivatives: Vec<[f64; 3]>,
}

/// Semi-numerical Hessian: column y is the central difference ([`central2`], step `step`) of
/// the gradient that `point` returns for the positions with coordinate y displaced (see
/// [`Displacement`]), and the dipole derivative is the same difference of its dipoles. Every
/// other coordinate is as in `positions`. Columns run in parallel on the current rayon pool
/// when `parallel`, each one on a single thread and written straight into the result, so the
/// Hessian is the same for every thread count if `point` runs sequentially. Errors: the first
/// failed displacement in coordinate order.
pub fn numerical_hessian<E: Send>(
    positions: &[[f64; 3]],
    step: f64,
    displacement: Displacement,
    parallel: bool,
    point: impl Fn(&[[f64; 3]]) -> Result<(Vec<[f64; 3]>, [f64; 3]), E> + Sync,
) -> Result<NumericalHessian, E> {
    let n3 = 3 * positions.len();
    let mut hessian = vec![0.0; n3 * n3];
    // column y of the Hessian into row y of the buffer: its transpose, the same once symmetrized
    let column = |(y, row): (usize, &mut [f64])| -> Result<[f64; 3], E> {
        let (a, k) = (y / 3, y % 3);
        let mut x = positions.to_vec();
        let x0 = x[a][k];
        x[a][k] = x0 + step;
        let (gp, mup) = point(&x)?;
        x[a][k] = match displacement {
            Displacement::FromReference => x0 - step,
            Displacement::ForwardBack => x[a][k] - 2.0 * step,
        };
        let (gm, mum) = point(&x)?;
        let (gp, gm) = (gp.as_flattened(), gm.as_flattened());
        assert!(
            gp.len() == n3 && gm.len() == n3,
            "a gradient of {} or {} entries for {n3} coordinates",
            gp.len(),
            gm.len()
        );
        for (r, h) in row.iter_mut().enumerate() {
            *h = central2(&gp[r], &gm[r], step);
        }
        Ok(central2(&mup, &mum, step))
    };
    let chunk = n3.max(1);
    let dipd: Vec<Result<[f64; 3], E>> = if parallel {
        hessian.par_chunks_mut(chunk).enumerate().map(column).collect()
    } else {
        hessian.chunks_mut(chunk).enumerate().map(column).collect()
    };
    let dipole_derivatives = dipd.into_iter().collect::<Result<Vec<_>, E>>()?;
    let max_asymmetry = symmetrize(&mut hessian, n3);
    Ok(NumericalHessian { hessian, max_asymmetry, dipole_derivatives })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stencils_keep_their_expressions() {
        let f = |x: f64| x.sin() * x.exp();
        let (x0, h) = (0.3, 1e-3);
        let d2 = derivative(Stencil::Central2, h, |d| f(x0 + d));
        assert_eq!(d2.to_bits(), ((f(x0 + h) - f(x0 - h)) / (2.0 * h)).to_bits());
        let d4 = derivative(Stencil::Central4, h, |d| f(x0 + d));
        let expect = (8.0 * (f(x0 + h) - f(x0 - h)) - (f(x0 + 2.0 * h) - f(x0 - 2.0 * h))) / (12.0 * h);
        assert_eq!(d4.to_bits(), expect.to_bits());
        let exact = x0.exp() * (x0.sin() + x0.cos());
        assert!((d4 - exact).abs() < 1e-11 && (d2 - exact).abs() < 1e-6);
    }

    #[test]
    fn offsets_in_order_and_first_error() {
        let mut seen = Vec::new();
        let v: Vec<[f64; 2]> = derivative(Stencil::Central4, 0.5, |d| {
            seen.push(d);
            vec![[d, 2.0 * d]]
        });
        assert_eq!(seen, [0.5, -0.5, 1.0, -1.0]);
        assert_eq!(v, vec![[1.0, 2.0]]);
        let mut calls = 0;
        let r: Result<f64, String> = try_derivative(Stencil::Central4, 0.5, |d| {
            calls += 1;
            if d < 0.0 {
                Err(format!("at {d}"))
            } else {
                Ok(d)
            }
        });
        assert_eq!((r, calls), (Err("at -0.5".to_string()), 2));
    }

    #[test]
    fn symmetrize_and_columns() {
        let mut h = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(symmetrize(&mut h, 2), 1.0);
        assert_eq!(h, [1.0, 2.5, 2.5, 4.0]);
        let cols = columns(3, false, |y| vec![y as f64, 10.0 + y as f64]);
        assert_eq!(cols, columns(3, true, |y| vec![y as f64, 10.0 + y as f64]));
        assert_eq!(from_columns(&cols, 2), [0.0, 1.0, 2.0, 10.0, 11.0, 12.0]);
    }

    #[test]
    fn numerical_hessian_of_a_quadratic() {
        // E = sum_ij A_ij x_i x_j / 2 with a symmetric A: the Hessian is A; dipole = x
        let pos = [[0.1, -0.2, 0.3], [1.0, 0.5, -0.7]];
        let a: Vec<f64> = (0..36).map(|k| ((k / 6 + 1) * (k % 6 + 1)) as f64 / 7.0).collect();
        let point = |x: &[[f64; 3]]| -> Result<(Vec<[f64; 3]>, [f64; 3]), ()> {
            let xf = x.as_flattened();
            let g: Vec<f64> = (0..6).map(|i| (0..6).map(|j| a[i * 6 + j] * xf[j]).sum()).collect();
            Ok((g.chunks(3).map(|c| [c[0], c[1], c[2]]).collect(), [xf[0] + xf[3], xf[1] + xf[4], xf[2] + xf[5]]))
        };
        for disp in [Displacement::FromReference, Displacement::ForwardBack] {
            let seq = numerical_hessian(&pos, 1e-3, disp, false, point).unwrap();
            let par = numerical_hessian(&pos, 1e-3, disp, true, point).unwrap();
            assert_eq!(seq.hessian, par.hessian);
            for (x, y) in seq.hessian.iter().zip(&a) {
                assert!((x - y).abs() < 1e-9);
            }
            for (k, d) in seq.dipole_derivatives.iter().enumerate() {
                for e in 0..3 {
                    assert!((d[e] - if e == k % 3 { 1.0 } else { 0.0 }).abs() < 1e-9);
                }
            }
        }
        // coordinates 2 and 3 fail: the error of coordinate 2, at its first point
        let err = numerical_hessian(&pos, 1e-3, Displacement::FromReference, true, |x: &[[f64; 3]]| {
            if x[0][2] != 0.3 || x[1][0] != 1.0 {
                Err(format!("{x:?}"))
            } else {
                point(x).map_err(|_| String::new())
            }
        });
        assert_eq!(err.unwrap_err(), format!("{:?}", [[0.1, -0.2, 0.3 + 1e-3], [1.0, 0.5, -0.7]]));
    }
}
