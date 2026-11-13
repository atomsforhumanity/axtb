// SPDX-License-Identifier: LGPL-3.0-or-later
//! The MO basis of a coupled-perturbed SCF and the transforms between it and the AO basis.
//!
//! A response is described in the orbitals `C` (columns) of the converged SCF: an AO matrix `X`
//! becomes `X̃ = Cᵀ X C`. The occupation response couples two orbitals only when one of them
//! carries electrons ([`super::OccupationResponse`]), so a symmetric response matrix has no
//! virtual-virtual block and is kept by its rows `[0, n_o)` (n_o x n). With the
//! occupied-occupied block of these rows `A` halved, the AO matrix of the symmetric MO matrix
//! is `C_o A Cᵀ + (C_o A Cᵀ)ᵀ`, `C_o` the first `n_o` columns of `C` ([`MoSpace::to_ao`]).
//!
//! [`MoSpace::to_mo`] forms `(C_oᵀ X) C` (2 n_o n² multiply-adds; the order of the two
//! products decides the rounding). Products run on faer's kernels with the caller's `Par`.

use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatRef, Par};

/// The orbitals of a converged SCF and the number of rows `n_o` of its response matrices.
#[derive(Clone, Copy, Debug)]
pub struct MoSpace<'a> {
    c: MatRef<'a, f64>,
    n_o: usize,
}

impl<'a> MoSpace<'a> {
    /// Orbitals `c` (n x n, one per column) and the rows `[0, n_o)` of the response.
    pub fn new(c: MatRef<'a, f64>, n_o: usize) -> Self {
        assert!(c.nrows() == c.ncols() && n_o <= c.ncols());
        Self { c, n_o }
    }

    /// The orbital coefficients.
    pub fn coefficients(&self) -> MatRef<'a, f64> {
        self.c
    }

    /// The number of rows of the response matrices.
    pub fn n_o(&self) -> usize {
        self.n_o
    }

    /// Rows `[0, n_o)` of `Cᵀ X C`, formed as `(C_oᵀ X) C`.
    pub fn to_mo(&self, x: MatRef<'_, f64>, par: Par) -> Mat<f64> {
        let n = self.c.nrows();
        let mut t = Mat::<f64>::zeros(self.n_o, n);
        matmul(t.as_mut(), Accum::Replace, self.c.subcols(0, self.n_o).transpose(), x, 1.0, par);
        let mut out = Mat::<f64>::zeros(self.n_o, n);
        matmul(out.as_mut(), Accum::Replace, t.as_ref(), self.c, 1.0, par);
        out
    }

    /// [`MoSpace::to_mo`] in the columns `[v0, n)` only (zero before them).
    pub fn to_mo_from(&self, x: MatRef<'_, f64>, v0: usize, par: Par) -> Mat<f64> {
        let n = self.c.nrows();
        let mut t = Mat::<f64>::zeros(self.n_o, n);
        matmul(t.as_mut(), Accum::Replace, self.c.subcols(0, self.n_o).transpose(), x, 1.0, par);
        let mut out = Mat::<f64>::zeros(self.n_o, n);
        matmul(out.as_mut().subcols_mut(v0, n - v0), Accum::Replace, t.as_ref(), self.c.subcols(v0, n - v0), 1.0, par);
        out
    }

    /// Rows `[0, n_o)` of `Cᵀ M C` for a matrix `M = L Eᵀ + E Lᵀ` local to the AOs
    /// `[off, off + na)`: `L` (n x na) holds its columns there and `E` the unit columns of those
    /// AOs. With `Y = Cᵀ L` and `C_A` the rows of `C` at these AOs, the rows are
    /// `Y_o C_A + C_A,oᵀ Yᵀ`.
    pub fn local_to_mo(&self, l: MatRef<'_, f64>, off: usize, par: Par) -> Mat<f64> {
        let n = self.c.nrows();
        let na = l.ncols();
        let mut y = Mat::<f64>::zeros(n, na);
        matmul(y.as_mut(), Accum::Replace, self.c.transpose(), l, 1.0, par);
        let ca = self.c.subrows(off, na);
        let mut out = Mat::<f64>::zeros(self.n_o, n);
        matmul(out.as_mut(), Accum::Replace, y.as_ref().subrows(0, self.n_o), ca, 1.0, par);
        matmul(out.as_mut(), Accum::Add, ca.subcols(0, self.n_o).transpose(), y.transpose(), 1.0, par);
        out
    }

    /// AO matrix `C_o (A Cᵀ) + transpose` of the symmetric MO matrix with rows `a`
    /// (`[0, n_o)`, occupied-occupied block halved).
    pub fn to_ao(&self, a: MatRef<'_, f64>, par: Par) -> Mat<f64> {
        let n = self.c.nrows();
        debug_assert_eq!(a.nrows(), self.n_o);
        let mut t = Mat::<f64>::zeros(self.n_o, n);
        matmul(t.as_mut(), Accum::Replace, a, self.c.transpose(), 1.0, par);
        let mut x = Mat::<f64>::zeros(n, n);
        matmul(x.as_mut(), Accum::Replace, self.c.subcols(0, self.n_o), t.as_ref(), 1.0, par);
        add_transpose(&x)
    }

    /// [`MoSpace::to_ao`] of rows `a` that vanish in the columns before `v0`.
    pub fn to_ao_from(&self, a: MatRef<'_, f64>, v0: usize, par: Par) -> Mat<f64> {
        let n = self.c.nrows();
        let nv = n - v0;
        debug_assert_eq!(a.nrows(), self.n_o);
        let mut t = Mat::<f64>::zeros(self.n_o, n);
        matmul(t.as_mut(), Accum::Replace, a.subcols(v0, nv), self.c.subcols(v0, nv).transpose(), 1.0, par);
        let mut x = Mat::<f64>::zeros(n, n);
        matmul(x.as_mut(), Accum::Replace, self.c.subcols(0, self.n_o), t.as_ref(), 1.0, par);
        add_transpose(&x)
    }

    /// Halve the occupied-occupied block of rows `a` (the form [`MoSpace::to_ao`] takes).
    pub fn halve_occupied(&self, a: &mut Mat<f64>) {
        for q in 0..self.n_o {
            for p in 0..self.n_o {
                a[(p, q)] *= 0.5;
            }
        }
    }
}

/// `X + Xᵀ` of a square matrix. (A loop over 32 x 32 tiles was measured slower for 150 to
/// 1500 rows except near 700.)
fn add_transpose(x: &Mat<f64>) -> Mat<f64> {
    let n = x.nrows();
    let mut out = Mat::<f64>::zeros(n, n);
    for j in 0..n {
        for i in 0..n {
            out[(i, j)] = x[(i, j)] + x[(j, i)];
        }
    }
    out
}
