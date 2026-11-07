//! Dense linear algebra with explicit parallelism (faer's low-level kernels only, never
//! its high-level solvers, which would use a global thread pool).

use crate::error::{Error, Result};
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::evd::{self_adjoint_evd, self_adjoint_evd_scratch, ComputeEigenvectors};
use faer::linalg::lu::partial_pivoting::{factor, solve};
use faer::perm::PermRef;
use faer::{Mat, MatRef, Par};

/// LU factorization with partial pivoting of a square matrix, kept for further solves.
pub struct Lu {
    lu: Mat<f64>,
    fwd: Vec<usize>,
    bwd: Vec<usize>,
    par: Par,
}

impl Lu {
    pub fn new(mut a: Mat<f64>, par: Par) -> Result<Self> {
        let n = a.nrows();
        let mut fwd = vec![0usize; n];
        let mut bwd = vec![0usize; n];
        let mut mem = MemBuffer::new(factor::lu_in_place_scratch::<usize, f64>(n, n, par, Default::default()));
        factor::lu_in_place(a.as_mut(), &mut fwd, &mut bwd, par, MemStack::new(&mut mem), Default::default());
        if (0..n).any(|i| !a[(i, i)].is_finite() || a[(i, i)] == 0.0) {
            return Err(Error::Numerical("singular EEQ matrix".into()));
        }
        Ok(Lu { lu: a, fwd, bwd, par })
    }

    /// Solve `A X = B` in place.
    pub fn solve(&self, b: &mut Mat<f64>) {
        let n = self.lu.nrows();
        let mut mem = MemBuffer::new(solve::solve_in_place_scratch::<usize, f64>(n, b.ncols(), self.par));
        let perm = PermRef::new_checked(&self.fwd, &self.bwd, n);
        solve::solve_in_place(self.lu.as_ref(), self.lu.as_ref(), perm, b.as_mut(), self.par, MemStack::new(&mut mem));
    }

    pub fn solve_vec(&self, b: &[f64]) -> Vec<f64> {
        let mut m = Mat::<f64>::zeros(b.len(), 1);
        for (i, v) in b.iter().enumerate() {
            m[(i, 0)] = *v;
        }
        self.solve(&mut m);
        (0..b.len()).map(|i| m[(i, 0)]).collect()
    }
}

/// Eigenvalues (ascending) and eigenvectors (columns) of a symmetric matrix.
pub fn symmetric_eigen(a: MatRef<'_, f64>, par: Par) -> Result<(Vec<f64>, Mat<f64>)> {
    let n = a.nrows();
    let mut u = Mat::<f64>::zeros(n, n);
    let mut s = faer::diag::Diag::<f64>::zeros(n);
    let mut mem = MemBuffer::new(StackReq::any_of(&[self_adjoint_evd_scratch::<f64>(
        n,
        ComputeEigenvectors::Yes,
        par,
        Default::default(),
    )]));
    self_adjoint_evd(a, s.as_mut(), Some(u.as_mut()), par, MemStack::new(&mut mem), Default::default())
        .map_err(|_| Error::Numerical("Hueckel eigensolver failed to converge".into()))?;
    Ok(((0..n).map(|i| s[i]).collect(), u))
}
