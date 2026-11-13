// SPDX-License-Identifier: LGPL-3.0-or-later
//! Dense linear systems A X = B, as named strategies. Partial pivoting chooses the row of the
//! largest |a_ik|; the strategies differ in which row wins a tie and in their arithmetic, so
//! each caller keeps the one it was validated with:
//!
//! - [`Lu`]: faer's LU factorization with partial pivoting (the first row of the largest
//!   |a_ik|, as LAPACK's; multipliers by the pivot's reciprocal; recursive blocking with
//!   matrix-product updates), kept for further solves. The EEQ charge equations
//!   ([`crate::eeq`]: the GFN2-xTB guess, GFN-FF's two charge models and the response of their
//!   charges, g-xTB's EEQ-BC charges and their adjoint), GFN2-xTB's SCC response, and g-xTB
//!   v1.1's Pulay coefficients.
//! - [`gauss_last_max`]: Gaussian elimination with partial pivoting on a matrix stored by rows
//!   (the last row of the largest |a_ik|, as `Iterator::max_by` picks; multipliers by
//!   division; the right-hand side eliminated along). The modified Broyden mixer of GFN2-xTB.
//!
//! g-xTB 2.0.1's DIIS has a third, LAPACK's unblocked `getf2` (in `axtb-gxtb`).

use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::lu::partial_pivoting::{factor, solve};
use faer::perm::PermRef;
use faer::{Mat, MatMut, Par};

/// faer's LU factorization with partial pivoting of a square matrix, kept for solves: the pivot
/// is the first row of the largest |a_ik| (as LAPACK's), the multipliers are taken with the
/// pivot's reciprocal, and the factorization is blocked recursively with matrix-product
/// updates (unlike [`gauss_last_max`]). On an explicit parallelism: faer's high-level
/// `partial_piv_lu` would run on faer's global parallelism.
pub struct Lu {
    lu: Mat<f64>,
    fwd: Vec<usize>,
    bwd: Vec<usize>,
    par: Par,
}

impl Lu {
    /// Factorize `a` (its storage keeps the factors) on the parallelism `par`, which the solves
    /// use too. A singular `a` is factorized as well; its solves give infinities or NaN.
    pub fn new(mut a: Mat<f64>, par: Par) -> Self {
        let n = a.nrows();
        let mut fwd = vec![0usize; n];
        let mut bwd = vec![0usize; n];
        let mut mem = MemBuffer::new(factor::lu_in_place_scratch::<usize, f64>(n, n, par, Default::default()));
        factor::lu_in_place(a.as_mut(), &mut fwd, &mut bwd, par, MemStack::new(&mut mem), Default::default());
        Lu { lu: a, fwd, bwd, par }
    }

    /// [`Lu::new`], or `None` when `a` is singular to working precision (a pivot is zero or not
    /// finite).
    pub fn try_new(a: Mat<f64>, par: Par) -> Option<Self> {
        let lu = Self::new(a, par);
        let n = lu.lu.nrows();
        if (0..n).any(|i| !lu.lu[(i, i)].is_finite() || lu.lu[(i, i)] == 0.0) {
            return None;
        }
        Some(lu)
    }

    /// Solve A X = B in place (`b` becomes X).
    pub fn solve(&self, b: MatMut<'_, f64>) {
        let n = self.lu.nrows();
        let mut mem = MemBuffer::new(solve::solve_in_place_scratch::<usize, f64>(n, b.ncols(), self.par));
        let perm = PermRef::new_checked(&self.fwd, &self.bwd, n);
        solve::solve_in_place(self.lu.as_ref(), self.lu.as_ref(), perm, b, self.par, MemStack::new(&mut mem));
    }

    /// The solution x of A x = `b`.
    pub fn solve_vec(&self, b: &[f64]) -> Vec<f64> {
        let mut m = Mat::<f64>::zeros(b.len(), 1);
        for (i, v) in b.iter().enumerate() {
            m[(i, 0)] = *v;
        }
        self.solve(m.as_mut());
        (0..b.len()).map(|i| m[(i, 0)]).collect()
    }
}

/// Solve a x = b in place (`b` becomes x; `a`, given by rows, is overwritten) by Gaussian
/// elimination with partial pivoting: the pivot is the last row of the largest |a_ik| (as
/// `Iterator::max_by` picks), the multipliers are quotients, and the right-hand side is
/// eliminated along (unlike [`Lu`]). Returns `false` when a pivot is zero (`a` and `b` are then
/// partly eliminated).
pub fn gauss_last_max(a: &mut [Vec<f64>], b: &mut [f64]) -> bool {
    let n = b.len();
    for col in 0..n {
        let piv = (col..n).max_by(|&i, &j| a[i][col].abs().total_cmp(&a[j][col].abs())).unwrap();
        if a[piv][col] == 0.0 {
            return false;
        }
        a.swap(col, piv);
        b.swap(col, piv);
        for row in col + 1..n {
            let f = a[row][col] / a[col][col];
            if f != 0.0 {
                for k in col..n {
                    a[row][k] -= f * a[col][k];
                }
                b[row] -= f * b[col];
            }
        }
    }
    for col in (0..n).rev() {
        let mut s = b[col];
        for k in col + 1..n {
            s -= a[col][k] * b[k];
        }
        b[col] = s / a[col][col];
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A system whose first column has a tie of largest magnitude: both strategies solve it,
    /// and they pivot on different rows (faer on the first, `gauss_last_max` on the last).
    #[test]
    fn strategies_solve_and_break_ties_differently() {
        let rows = [[1.0, 2.0, 0.5], [-3.0, 1.0, 2.0], [3.0, -1.0, 1.0]];
        let x = [0.25, -1.5, 2.0];
        let b: Vec<f64> = rows.iter().map(|r| r.iter().zip(&x).map(|(a, x)| a * x).sum()).collect();
        let a = Mat::<f64>::from_fn(3, 3, |i, j| rows[i][j]);
        let lu = Lu::try_new(a, Par::Seq).unwrap();
        // the first row of the largest |a_i0| (row 1, -3) is the first pivot
        assert_eq!(lu.lu[(0, 0)], -3.0);
        let mut g: Vec<Vec<f64>> = rows.iter().map(|r| r.to_vec()).collect();
        let mut y = b.clone();
        assert!(gauss_last_max(&mut g, &mut y));
        // the last (row 2, 3)
        assert_eq!(g[0], vec![3.0, -1.0, 1.0]);
        for (v, w) in [lu.solve_vec(&b), y].iter().flat_map(|s| s.iter().zip(&x)) {
            assert!((v - w).abs() < 1e-14, "{v} {w}");
        }
    }

    /// Singular to the last bit: a column of zeros stays zero through the elimination.
    #[test]
    fn singular_systems() {
        let a = Mat::<f64>::from_fn(3, 3, |i, j| if j == 1 { 0.0 } else { (i + 2 * j + 1) as f64 });
        assert!(Lu::try_new(a.clone(), Par::Seq).is_none());
        assert!(Lu::new(a, Par::Seq).solve_vec(&[1.0, 2.0, 3.0]).iter().any(|v| !v.is_finite()));
        let mut g = vec![vec![1.0, 0.0], vec![2.0, 0.0]];
        assert!(!gauss_last_max(&mut g, &mut [1.0, 1.0]));
    }
}
