//! Dense linear solves.

use crate::{Error, Result};
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::{Mat, Par};

/// Solve `A X = B` in place (`b` becomes X) by LU decomposition with partial pivoting.
pub fn lu_solve(mut a: Mat<f64>, b: &mut Mat<f64>, par: Par) -> Result<()> {
    use faer::linalg::lu::partial_pivoting::{factor, solve};
    use faer::perm::PermRef;
    let n = a.nrows();
    let mut fwd = vec![0usize; n];
    let mut bwd = vec![0usize; n];
    let req = StackReq::any_of(&[
        factor::lu_in_place_scratch::<usize, f64>(n, n, par, Default::default()),
        solve::solve_in_place_scratch::<usize, f64>(n, b.ncols(), par),
    ]);
    let mut mem = MemBuffer::new(req);
    factor::lu_in_place(a.as_mut(), &mut fwd, &mut bwd, par, MemStack::new(&mut mem), Default::default());
    if (0..n).any(|i| !a[(i, i)].is_finite() || a[(i, i)] == 0.0) {
        return Err(Error::Numerical("singular response matrix".into()));
    }
    let perm = PermRef::new_checked(&fwd, &bwd, n);
    solve::solve_in_place(a.as_ref(), a.as_ref(), perm, b.as_mut(), par, MemStack::new(&mut mem));
    Ok(())
}
