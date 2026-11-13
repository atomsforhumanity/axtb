// SPDX-License-Identifier: LGPL-3.0-or-later
//! Symmetric eigenproblems: the generalized one of the SCF, F C = S C ε ([`GenEigSolver`], with
//! ports of LAPACK's `dsygst` here and of `dsterf`/`dstein` in `tridiag`, and the run-time
//! library of `lapack`; the method is described in the documentation of [`crate::linalg`]),
//! and the standard one A U = U Λ ([`sym_eig`], faer's).

use super::lapack::{Lapack, LapackEig};
use super::tridiag;
use crate::{Error, Result};
use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::cholesky::llt::factor::{cholesky_in_place, cholesky_in_place_scratch, LltRegularization};
use faer::linalg::evd::{self_adjoint_evd, self_adjoint_evd_scratch, ComputeEigenvectors};
use faer::linalg::triangular_solve::{solve_lower_triangular_in_place, solve_upper_triangular_in_place};
use faer::{Mat, MatMut, MatRef, Par};

/// The run-time LAPACK library of the SCF eigensolver: `Ok(None)` when `AXTB_LAPACK` is not
/// set, `Ok(Some(name and version))` when it is loaded, `Err(reason)` when `AXTB_LAPACK` names
/// a library that cannot be used (faer runs instead). The variable is read and the library
/// loaded once per process, by this call or the first SCF of 200 or more basis functions.
pub fn runtime_lapack() -> std::result::Result<Option<String>, String> {
    match Lapack::from_env() {
        Ok(lib) => Ok(lib.map(Lapack::describe)),
        Err(e) => Err(e.to_string()),
    }
}

/// Rows from which the run-time LAPACK library solves the eigenproblems (below, faer is as
/// fast and the library's start-up, dlopen and its first call, ~5 ms, would cost more than it
/// saves) and [`GenEigSolver::solve_lowest`] reduces to standard form by [`sygst`] (below, the
/// reduction takes under a millisecond either way, and two triangular solves keep the
/// results of small molecules as they were).
const LARGE: usize = 200;

/// Thread count of a faer parallelism setting.
fn threads_of(par: Par) -> usize {
    match par {
        Par::Seq => 1,
        Par::Rayon(n) => n.get(),
    }
}

/// Solver for F C = S C ε with a fixed overlap matrix.
pub struct GenEigSolver {
    /// Lower Cholesky factor of S.
    l: Mat<f64>,
    work: Mat<f64>,
    mem: MemBuffer,
    par: Par,
    /// Work space of [`GenEigSolver::solve_lowest`] (allocated on first use).
    lowest: Option<Lowest>,
    /// `solve_lowest` uses the full solver: the last spectrum had clusters too large for
    /// inverse iteration ([`cluster_cost`](tridiag::cluster_cost) above 1, or above 1/2
    /// while on the full solver).
    full_next: bool,
    /// The last `solve_lowest` call used the full solver.
    last_full: bool,
    /// The run-time LAPACK eigensolver (`AXTB_LAPACK`), when loaded.
    lapack: Option<(&'static Lapack, LapackEig)>,
    /// Why faer runs although `AXTB_LAPACK` is set: the library was refused, or failed.
    lapack_not_used: Option<String>,
}

/// Tridiagonal reduction and its Householder factors, and the work space of their use.
struct Lowest {
    trid: Mat<f64>,
    householder: Mat<f64>,
    mem: MemBuffer,
}

impl GenEigSolver {
    /// Factorize the overlap matrix.
    pub fn new(s: MatRef<'_, f64>, par: Par) -> Result<Self> {
        let n = s.nrows();
        // (in parallel, the copy goes into a matrix the pool's threads zeroed: they take its page
        // faults)
        let mut l = if matches!(par, Par::Seq) {
            s.to_owned()
        } else {
            let mut l = super::zeros(n, n, par);
            l.copy_from(s);
            l
        };
        let req = StackReq::any_of(&[
            cholesky_in_place_scratch::<f64>(n, par, Default::default()),
            self_adjoint_evd_scratch::<f64>(n, ComputeEigenvectors::Yes, par, Default::default()),
        ]);
        let mut mem = MemBuffer::new(req);
        cholesky_in_place(l.as_mut(), LltRegularization::default(), par, MemStack::new(&mut mem), Default::default())
            .map_err(|_| Error::Numerical("overlap matrix is not positive definite".into()))?;
        // zero the strict upper triangle so `l` is a clean lower-triangular factor
        for j in 0..n {
            for i in 0..j {
                l[(i, j)] = 0.0;
            }
        }
        // a run-time LAPACK library from LARGE rows on
        let (lapack, lapack_not_used) = match (n >= LARGE).then(Lapack::from_env) {
            Some(Ok(Some(lib))) => (Some((lib, LapackEig::new(n))), None),
            Some(Err(e)) => (None, Some(format!("AXTB_LAPACK not used: {e}"))),
            _ => (None, None),
        };
        Ok(Self {
            l,
            work: super::zeros(n, n, par),
            mem,
            par,
            lowest: None,
            full_next: false,
            last_full: false,
            lapack,
            lapack_not_used,
        })
    }

    /// The eigensolver of this solver's calls: `"faer"`, or the run-time library's name and
    /// version (`"mkl (Intel(R) oneAPI Math Kernel Library ...)"`); with the reason when
    /// `AXTB_LAPACK` is set but faer ran (`"faer (AXTB_LAPACK not used: ...)"`, or after the
    /// library failed in a call, `"faer (after mkl failed: dstedc info 3)"`).
    pub fn backend(&self) -> String {
        match (&self.lapack, &self.lapack_not_used) {
            (Some((lib, _)), _) => lib.describe(),
            (None, Some(why)) => format!("faer ({why})"),
            (None, None) => "faer".into(),
        }
    }

    /// Forget the spectra of earlier [`solve_lowest`](Self::solve_lowest) calls, for a new SCF
    /// with the same overlap: its calls then choose their paths as those of a new solver.
    #[inline]
    pub fn restart(&mut self) {
        self.full_next = false;
    }

    /// Solve for eigenvalues (ascending) and S-orthonormal eigenvectors (columns of `c`).
    /// `f` must be symmetric; it is not modified. The reduction to standard form is two
    /// triangular solves with n right-hand sides (the program conventions' solver, kept bit for
    /// bit; [`solve_lowest`](Self::solve_lowest) reduces large matrices in half the flops).
    pub fn solve(&mut self, f: MatRef<'_, f64>, eps: &mut [f64], c: &mut Mat<f64>) -> Result<()> {
        let n = f.nrows();
        // work = L⁻¹ F
        self.work.copy_from(f);
        solve_lower_triangular_in_place(self.l.as_ref(), self.work.as_mut(), self.par);
        // c = L⁻¹ (L⁻¹ F)ᵀ = L⁻¹ F L⁻ᵀ
        c.copy_from(self.work.transpose());
        solve_lower_triangular_in_place(self.l.as_ref(), c.as_mut(), self.par);
        if let Some((lib, eig)) = self.lapack.as_mut() {
            // the library's dsytrd + dstedc + dormtr on A (lower triangle), then C = L⁻ᵀ U
            let threads = threads_of(self.par);
            let mut a = vec![0.0; n * n];
            for j in 0..n {
                for i in j..n {
                    a[j * n + i] = c[(i, j)];
                }
            }
            let mut v = Vec::new();
            let name = lib.name;
            match eig.solve(lib, threads, &mut a, n, eps, &mut v, |e| e.len()) {
                Ok(_) => {
                    c.copy_from(faer::MatRef::from_column_major_slice(&v, n, n));
                    solve_upper_triangular_in_place(self.l.transpose(), c.as_mut(), self.par);
                    return Ok(());
                }
                // faer from now on, on the transformed matrix still in `c` (the following
                // eigenpairs of the SCF differ from the library's by round-off)
                Err(e) => {
                    self.lapack = None;
                    self.lapack_not_used = Some(format!("after {name} failed: {e}"));
                }
            }
        }
        // eigendecomposition of the transformed matrix (lower triangle is read)
        let mut s = faer::diag::Diag::<f64>::zeros(n);
        self_adjoint_evd(
            c.as_ref(),
            s.as_mut(),
            Some(self.work.as_mut()),
            self.par,
            MemStack::new(&mut self.mem),
            Default::default(),
        )
        .map_err(|_| Error::Numerical("eigensolver failed to converge".into()))?;
        for i in 0..n {
            eps[i] = s[i];
        }
        // back-transform: C = L⁻ᵀ U
        c.copy_from(&self.work);
        solve_upper_triangular_in_place(self.l.transpose(), c.as_mut(), self.par);
        Ok(())
    }
}

impl GenEigSolver {
    /// All eigenvalues (ascending) of F C = S C ε and the S-orthonormal eigenvectors of the
    /// lowest `k(eps)` of them (columns 0..k of `c`; the other columns are not set). Returns
    /// k, or n when the full solver computed every eigenvector. Cheaper than
    /// [`GenEigSolver::solve`] when only the occupied orbitals are needed.
    ///
    /// From 200 rows on the reduction to standard form is LAPACK's two-sided `dsygst`
    /// (half the flops of `solve`'s two triangular solves, and as accurate); below, it is
    /// `solve`'s. With a run-time LAPACK library (`AXTB_LAPACK`, 200 rows or more) the
    /// library does everything: the reduction (`dsygst`), the eigenvalues (`dsytrd` +
    /// `dstedc`), the k eigenvectors alone (`dormtr`) and their back-transformation to
    /// C = L⁻ᵀ U (`dtrsm`). Otherwise the transformed matrix is reduced to tridiagonal form
    /// (faer), its eigenvalues computed by root-free QL/QR and the k eigenvectors by inverse
    /// iteration (LAPACK's `dsterf` and `dstein`, ported), which are back-transformed alone,
    /// unless the eigenvalues form large clusters, where divide and conquer is faster: after a
    /// spectrum whose cluster cost (inverse iteration's relative to divide and conquer's, from
    /// the clusters' sizes) is above 1 the calls use the full solver until the cost drops
    /// below 1/2 (SCF spectra change little between iterations; the hysteresis keeps a
    /// spectrum near the threshold on one path). Given the library (or none), the choice
    /// depends on the eigenvalues alone, not on the thread count. With inverse iteration the
    /// first k eigenvectors of a call with a larger k are the same.
    pub fn solve_lowest(
        &mut self,
        f: MatRef<'_, f64>,
        eps: &mut [f64],
        c: &mut Mat<f64>,
        k: impl Fn(&[f64]) -> usize,
    ) -> Result<usize> {
        use faer::linalg::evd::tridiag::tridiag_in_place;
        use faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj;
        let n = f.nrows();
        self.last_full = true;
        if n < 2 {
            self.solve(f, eps, c)?;
            return Ok(n);
        }
        if self.full_next && self.lapack.is_none() {
            self.solve_full_reduced(f, eps, c)?;
            // ||T||_1 of the tridiagonal form is not at hand: its lower bound max |eps|
            let tnorm = eps[0].abs().max(eps[n - 1].abs());
            self.full_next = tridiag::cluster_cost(&eps[..n], k(&eps[..n]), tnorm) > 0.5;
            return Ok(n);
        }
        let par = self.par;
        if let Some((lib, eig)) = self.lapack.as_mut() {
            // all in the library: dsygst, dsytrd + dstedc, dormtr on the k needed columns, and
            // C = L⁻ᵀ U by dtrsm
            let name = lib.name;
            match eig.solve_generalized(lib, threads_of(par), f, self.l.as_ref(), eps, &k) {
                Ok(k) => {
                    self.last_full = false;
                    if k > 0 {
                        c.as_mut().subcols_mut(0, k).copy_from(MatRef::from_column_major_slice(eig.vectors(k), n, k));
                    }
                    return Ok(k);
                }
                // faer's path below, from F again
                Err(e) => {
                    self.lapack = None;
                    self.lapack_not_used = Some(format!("after {name} failed: {e}"));
                }
            }
        }
        // A = L⁻¹ F L⁻ᵀ (lower triangle), then tridiagonal form
        let lw = Self::lowest_work(&mut self.lowest, n, par);
        reduce(&mut lw.trid, f, self.l.as_ref(), &mut self.work, c, par);
        tridiag_in_place(
            lw.trid.as_mut(),
            lw.householder.as_mut(),
            par,
            MemStack::new(&mut lw.mem),
            Default::default(),
        );
        let d: Vec<f64> = (0..n).map(|i| lw.trid[(i, i)]).collect();
        let e: Vec<f64> = (0..n - 1).map(|i| lw.trid[(i + 1, i)]).collect();
        eps[..n].copy_from_slice(&d);
        let mut e2 = e.clone();
        tridiag::eigenvalues(&mut eps[..n], &mut e2)?;
        let k = k(&eps[..n]).min(n);
        self.last_full = false;
        if k == 0 {
            return Ok(0);
        }
        self.full_next = tridiag::cluster_cost(&eps[..n], k, tridiag::one_norm(&d, &e)) > 1.0;
        let mut z = vec![0.0; n * k];
        if tridiag::eigenvectors(&d, &e, &eps[..k], &mut z, !matches!(par, Par::Seq)).is_err() {
            // inverse iteration did not converge: every eigenpair from the full solver
            self.last_full = true;
            self.solve_full_reduced(f, eps, c)?;
            return Ok(n);
        }
        let mut zm = faer::MatMut::from_column_major_slice_mut(&mut z, n, k);
        apply_block_householder_sequence_on_the_left_in_place_with_conj(
            lw.trid.as_ref().submatrix(1, 0, n - 1, n - 1),
            lw.householder.as_ref(),
            faer::Conj::No,
            zm.as_mut().subrows_mut(1, n - 1),
            par,
            MemStack::new(&mut lw.mem),
        );
        // C = L⁻ᵀ U for the k columns
        solve_upper_triangular_in_place(self.l.transpose(), zm.as_mut(), par);
        c.as_mut().subcols_mut(0, k).copy_from(zm.as_ref());
        Ok(k)
    }

    /// The work space of `solve_lowest` for n rows (allocated on first use).
    fn lowest_work(lowest: &mut Option<Lowest>, n: usize, par: Par) -> &mut Lowest {
        use faer::linalg::evd::tridiag::tridiag_in_place_scratch;
        use faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch;
        lowest.get_or_insert_with(|| {
            let bs = faer::linalg::qr::no_pivoting::factor::recommended_blocksize::<f64>(n, n);
            let req = StackReq::any_of(&[
                tridiag_in_place_scratch::<f64>(n, par, Default::default()),
                apply_block_householder_sequence_on_the_left_in_place_scratch::<f64>(n - 1, bs, n),
            ]);
            Lowest { trid: super::zeros(n, n, par), householder: Mat::zeros(bs, n - 1), mem: MemBuffer::new(req) }
        })
    }

    /// Every eigenpair by the full solver, the full-solver path of
    /// [`solve_lowest`](Self::solve_lowest) and [`solve_all_with`](Self::solve_all_with): below
    /// [`LARGE`] rows [`solve`](Self::solve), from [`LARGE`] rows on faer's on the reduction by
    /// [`sygst`].
    fn solve_full_reduced(&mut self, f: MatRef<'_, f64>, eps: &mut [f64], c: &mut Mat<f64>) -> Result<()> {
        let n = f.nrows();
        if n < LARGE {
            return self.solve(f, eps, c);
        }
        let par = self.par;
        let lw = Self::lowest_work(&mut self.lowest, n, par);
        reduce(&mut lw.trid, f, self.l.as_ref(), &mut self.work, c, par);
        let mut s = faer::diag::Diag::<f64>::zeros(n);
        self_adjoint_evd(
            lw.trid.as_ref(),
            s.as_mut(),
            Some(self.work.as_mut()),
            par,
            MemStack::new(&mut self.mem),
            Default::default(),
        )
        .map_err(|_| Error::Numerical("eigensolver failed to converge".into()))?;
        for i in 0..n {
            eps[i] = s[i];
        }
        // back-transform: C = L⁻ᵀ U
        c.copy_from(&self.work);
        solve_upper_triangular_in_place(self.l.transpose(), c.as_mut(), par);
        Ok(())
    }

    /// Whether the last [`solve_lowest`](Self::solve_lowest) call used the full solver.
    #[inline]
    pub fn used_full(&self) -> bool {
        self.last_full
    }

    /// Every eigenpair, by the full solver (`full`) or by the path of `solve_lowest`: with the
    /// method of an earlier [`solve_lowest`](Self::solve_lowest) call ([`used_full`](Self::used_full))
    /// and the same `f`, the eigenvectors that call computed are reproduced (the orbitals of a
    /// converged SCF with its virtual ones): bit for bit with faer; with a run-time library to
    /// round-off (~1e-16), since `dormtr` blocks the back-transformation of n columns
    /// differently from that of k (OpenBLAS, and MKL 2025 at 240 rows on one thread, as before
    /// `dsygst`). Leaves the choice of the next `solve_lowest` call as it was.
    pub fn solve_all_with(&mut self, f: MatRef<'_, f64>, eps: &mut [f64], c: &mut Mat<f64>, full: bool) -> Result<()> {
        if full {
            return self.solve_full_reduced(f, eps, c);
        }
        let (full_next, last_full) = (std::mem::replace(&mut self.full_next, false), self.last_full);
        let r = self.solve_lowest(f, eps, c, |e| e.len());
        (self.full_next, self.last_full) = (full_next, last_full);
        r.map(|_| ())
    }
}

/// Eigenvalues (ascending) and orthonormal eigenvectors (columns) of the symmetric matrix `a`
/// (its lower triangle is read): faer's `self_adjoint_evd` (tridiagonalization, divide and
/// conquer) on the parallelism `par`. For the small eigenproblems outside an SCF (GFN-FF's
/// Hückel systems, the vibrational analysis): the results and the work space are allocated
/// in every call.
pub fn sym_eig(a: MatRef<'_, f64>, par: Par) -> Result<(Vec<f64>, Mat<f64>)> {
    let n = a.nrows();
    let mut s = faer::diag::Diag::<f64>::zeros(n);
    let mut u = Mat::<f64>::zeros(n, n);
    let mut mem = MemBuffer::new(self_adjoint_evd_scratch::<f64>(n, ComputeEigenvectors::Yes, par, Default::default()));
    self_adjoint_evd(a, s.as_mut(), Some(u.as_mut()), par, MemStack::new(&mut mem), Default::default())
        .map_err(|_| Error::Numerical("eigensolver failed to converge".into()))?;
    Ok(((0..n).map(|i| s[i]).collect(), u))
}

/// A = L⁻¹ F L⁻ᵀ, its lower triangle into `a`: the reduction of
/// [`GenEigSolver::solve_lowest`]. From [`LARGE`] rows on by [`sygst`]; below, as in
/// [`GenEigSolver::solve`], by two triangular solves (`work` and `c` are scratch).
fn reduce(a: &mut Mat<f64>, f: MatRef<'_, f64>, l: MatRef<'_, f64>, work: &mut Mat<f64>, c: &mut Mat<f64>, par: Par) {
    if f.nrows() >= LARGE {
        a.copy_from_triangular_lower(f);
        sygst(a.as_mut(), l, par);
    } else {
        work.copy_from(f);
        solve_lower_triangular_in_place(l, work.as_mut(), par);
        c.copy_from(work.transpose());
        solve_lower_triangular_in_place(l, c.as_mut(), par);
        a.copy_from_triangular_lower(c.as_ref());
    }
}

/// Block size of [`sygst`]'s outer level, its diagonal blocks' inner level, and the row chunks
/// of its panel updates. Fixed (not chosen by the thread count), so that the results do not
/// depend on it. Measured on the i9-11900H (n = 300, 600, 1200; 1, 2, 4 and 8 threads): 48 to
/// 96 within a few per cent of each other, 64 best or tied; 16 for the diagonal blocks halves
/// their cost against the unblocked reduction of 64 rows.
const SYGST_BLOCK: usize = 64;
const SYGST_INNER: usize = 16;
const SYGST_ROWS: usize = 64;

/// A = L⁻¹ A L⁻ᵀ in place on the lower triangle of `a` (the strict upper triangle is neither
/// read nor written), with the lower triangular `l`: the two-sided reduction of LAPACK's
/// `dsygst` (itype 1, lower; n³ flops, half those of two triangular solves with n right-hand
/// sides, and like them free of an explicit L⁻¹). Per block column of [`SYGST_BLOCK`]: the
/// diagonal block (recursively, in blocks of [`SYGST_INNER`], then [`sygs2`]), the panel below
/// it (A21 L11⁻ᵀ - ½ L21 A11, in independent row chunks), the symmetric rank-2k update of the
/// trailing lower triangle (faer's triangular products) and the second ½ L21 A11. LAPACK's
/// last step per block column, A21 := L22⁻¹ A21, reads nothing that later block columns write
/// and nothing later reads it, so it is deferred and the panels are solved together,
/// independently. Each parallel part works on fixed chunks with sequential kernels, or is a
/// faer matrix product (which splits its output, never a sum), so the result is the same for
/// every thread count.
fn sygst(a: MatMut<'_, f64>, l: MatRef<'_, f64>, par: Par) {
    sygst_blocked(a, l, SYGST_BLOCK, par);
}

/// [`sygst`] in blocks of `nb` columns.
fn sygst_blocked(mut a: MatMut<'_, f64>, l: MatRef<'_, f64>, nb: usize, par: Par) {
    use faer::linalg::matmul::matmul;
    use faer::linalg::matmul::triangular::{matmul as triangular_matmul, BlockStructure};
    use faer::reborrow::{Reborrow, ReborrowMut};
    use rayon::prelude::*;
    let n = a.nrows();
    if n <= SYGST_INNER {
        sygs2(a, l);
        return;
    }
    let seq = matches!(par, Par::Seq);
    // the reduced diagonal block, both triangles (the operand of the two ½ L21 A11 products)
    let mut sym = Mat::<f64>::zeros(nb, nb);
    for k in (0..n).step_by(nb) {
        let kb = nb.min(n - k);
        let (mut a11, _, mut a21, mut a22) = a.rb_mut().submatrix_mut(k, k, n - k, n - k).split_at_mut(kb, kb);
        let l11 = l.submatrix(k, k, kb, kb);
        sygst_blocked(a11.rb_mut(), l11, SYGST_INNER, Par::Seq);
        if k + kb == n {
            break;
        }
        let mut sym = sym.as_mut().submatrix_mut(0, 0, kb, kb);
        for j in 0..kb {
            for i in j..kb {
                sym[(i, j)] = a11[(i, j)];
                sym[(j, i)] = a11[(i, j)];
            }
        }
        let sym = sym.rb();
        let m = n - k - kb;
        let l21 = l.submatrix(k + kb, k, m, kb);
        // A21 := A21 L11⁻ᵀ - ½ L21 A11, row by row
        let rows = |(mut a21, l21): (MatMut<'_, f64>, MatRef<'_, f64>)| {
            solve_lower_triangular_in_place(l11, a21.rb_mut().transpose_mut(), Par::Seq);
            matmul(a21, faer::Accum::Add, l21, sym, -0.5, Par::Seq);
        };
        if seq {
            for r in (0..m).step_by(SYGST_ROWS) {
                let h = SYGST_ROWS.min(m - r);
                rows((a21.rb_mut().subrows_mut(r, h), l21.subrows(r, h)));
            }
        } else {
            a21.rb_mut().par_row_chunks_mut(SYGST_ROWS).zip(l21.par_row_chunks(SYGST_ROWS)).for_each(rows);
        }
        // A22 -= A21 L21ᵀ + L21 A21ᵀ (lower triangle)
        let (add, low, rect) = (faer::Accum::Add, BlockStructure::TriangularLower, BlockStructure::Rectangular);
        triangular_matmul(a22.rb_mut(), low, add, a21.rb(), rect, l21.transpose(), rect, -1.0, par);
        triangular_matmul(a22.rb_mut(), low, add, l21, rect, a21.rb().transpose(), rect, -1.0, par);
        // A21 -= ½ L21 A11
        matmul(a21.rb_mut(), add, l21, sym, -0.5, par);
    }
    // the deferred solves: block column p below its diagonal block, A21 := L22⁻¹ A21
    let panel = |(p, a): (usize, MatMut<'_, f64>)| {
        let r = ((p + 1) * nb).min(n);
        solve_lower_triangular_in_place(l.submatrix(r, r, n - r, n - r), a.subrows_mut(r, n - r), Par::Seq);
    };
    if seq {
        for p in 0..n.div_ceil(nb) {
            let c = p * nb;
            panel((p, a.rb_mut().subcols_mut(c, nb.min(n - c))));
        }
    } else {
        a.par_col_chunks_mut(nb).enumerate().for_each(panel);
    }
}

/// [`sygst`] unblocked (LAPACK `dsygs2`, itype 1, lower), for n <= [`SYGST_INNER`]: on dense
/// copies of `a` and `l`.
fn sygs2(mut a: MatMut<'_, f64>, l: MatRef<'_, f64>) {
    const M: usize = SYGST_INNER;
    let n = a.nrows();
    debug_assert!(n <= M);
    let (mut x, mut y) = ([0.0; M * M], [0.0; M * M]);
    for j in 0..n {
        for i in j..n {
            x[i + j * n] = a[(i, j)];
            y[i + j * n] = l[(i, j)];
        }
    }
    let (x, y) = (&mut x[..n * n], &y[..n * n]);
    for k in 0..n {
        let (left, right) = x.split_at_mut((k + 1) * n);
        let ak = &mut left[k * n..];
        let lk = &y[k * n..(k + 1) * n];
        let akk = ak[k] / (lk[k] * lk[k]);
        ak[k] = akk;
        // a(k+1:, k) / l_kk - ½ akk l(k+1:, k)
        let (r, ct) = (1.0 / lk[k], -0.5 * akk);
        for (v, &w) in ak[k + 1..].iter_mut().zip(&lk[k + 1..]) {
            *v = *v * r + ct * w;
        }
        // A(k+1:, k+1:) -= a lᵀ + l aᵀ (lower triangle)
        for (jj, col) in right.chunks_exact_mut(n).enumerate() {
            let j = k + 1 + jj;
            let (t1, t2) = (-lk[j], -ak[j]);
            for ((v, &p), &q) in col[j..].iter_mut().zip(&ak[j..]).zip(&lk[j..]) {
                *v += p * t1 + q * t2;
            }
        }
        for (v, &w) in ak[k + 1..].iter_mut().zip(&lk[k + 1..]) {
            *v += ct * w;
        }
        // a(k+1:, k) := L(k+1:, k+1:)⁻¹ a(k+1:, k)
        for j in k + 1..n {
            let lj = &y[j * n..(j + 1) * n];
            let v = ak[j] / lj[j];
            ak[j] = v;
            for (p, &q) in ak[j + 1..].iter_mut().zip(&lj[j + 1..]) {
                *p -= v * q;
            }
        }
    }
    for j in 0..n {
        for i in j..n {
            a[(i, j)] = x[i + j * n];
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `solve_lowest` against `solve`: eigenvalues, residuals and S-orthonormality of the lowest
    /// eigenvectors, and the same first columns for a larger k.
    #[test]
    fn lowest_eigenpairs_match_the_full_solver() {
        let mut seed = 11u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        };
        for &n in &[3usize, 40, 150, 260] {
            let mut f = Mat::<f64>::zeros(n, n);
            let mut s = Mat::<f64>::zeros(n, n);
            for i in 0..n {
                for j in 0..=i {
                    let v = rnd();
                    f[(i, j)] = v;
                    f[(j, i)] = v;
                    let w = 0.1 * rnd() / (1.0 + (i as f64 - j as f64).abs());
                    s[(i, j)] = w;
                    s[(j, i)] = w;
                }
                s[(i, i)] = 1.0;
            }
            let mut solver = GenEigSolver::new(s.as_ref(), Par::Seq).unwrap();
            // faer's paths also when AXTB_LAPACK names a library (`runtime_lapack_matches_faer`)
            solver.lapack = None;
            let (mut e1, mut c1) = (vec![0.0; n], Mat::<f64>::zeros(n, n));
            solver.solve(f.as_ref(), &mut e1, &mut c1).unwrap();
            let (mut e2, mut c2) = (vec![0.0; n], Mat::<f64>::zeros(n, n));
            let k = solver.solve_lowest(f.as_ref(), &mut e2, &mut c2, |_| n / 2 + 1).unwrap();
            for (a, b) in e1.iter().zip(&e2) {
                assert!((a - b).abs() < 1e-12, "{a} {b}");
            }
            let fc = &f * c2.as_ref().subcols(0, k);
            let sc = &s * c2.as_ref().subcols(0, k);
            for j in 0..k {
                for i in 0..n {
                    assert!((fc[(i, j)] - e2[j] * sc[(i, j)]).abs() < 1e-11);
                }
                for l in 0..k {
                    let dot: f64 = (0..n).map(|i| c2[(i, l)] * sc[(i, j)]).sum();
                    assert!((dot - if l == j { 1.0 } else { 0.0 }).abs() < 1e-11);
                }
            }
            let mut c3 = Mat::<f64>::zeros(n, n);
            let full = solver.used_full();
            solver.solve_all_with(f.as_ref(), &mut e2, &mut c3, full).unwrap();
            assert_eq!(c2.as_ref().subcols(0, k), c3.as_ref().subcols(0, k));
        }
    }

    /// For a matrix large enough for faer's parallel tridiagonalization, both solvers repeat
    /// bit for bit on 3 threads, and 1 and 3 threads agree to round-off.
    #[test]
    fn eigenpairs_on_several_threads() {
        let n = 400;
        let mut seed = 5u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        };
        let mut f = Mat::<f64>::zeros(n, n);
        let mut s = Mat::<f64>::identity(n, n);
        for i in 0..n {
            for j in 0..i {
                let v = rnd();
                f[(i, j)] = v;
                f[(j, i)] = v;
                let w = 0.01 * rnd();
                s[(i, j)] = w;
                s[(j, i)] = w;
            }
        }
        let run = |par: Par| {
            let mut solver = GenEigSolver::new(s.as_ref(), par).unwrap();
            let (mut e1, mut c1) = (vec![0.0; n], Mat::<f64>::zeros(n, n));
            solver.solve(f.as_ref(), &mut e1, &mut c1).unwrap();
            let (mut e2, mut c2) = (vec![0.0; n], Mat::<f64>::zeros(n, n));
            let k = solver.solve_lowest(f.as_ref(), &mut e2, &mut c2, |_| n / 2).unwrap();
            (e1, c1, e2, c2.as_ref().subcols(0, k).to_owned())
        };
        let seq = run(Par::Seq);
        let pool = rayon::ThreadPoolBuilder::new().num_threads(3).build().unwrap();
        let par = pool.install(|| run(Par::rayon(3)));
        assert_eq!(pool.install(|| run(Par::rayon(3))), par);
        // eigenvalues, and eigenvectors up to their signs
        let close = |a: &[f64], b: &[f64]| a.iter().zip(b).all(|(x, y)| (x - y).abs() < 1e-12);
        let cols = |a: &Mat<f64>, b: &Mat<f64>| {
            (0..b.ncols()).all(|j| {
                let s = if a[(0, j)] * b[(0, j)] < 0.0 { -1.0 } else { 1.0 };
                (0..n).all(|i| (a[(i, j)] - s * b[(i, j)]).abs() < 1e-9)
            })
        };
        assert!(close(&seq.0, &par.0) && close(&seq.2, &par.2));
        assert!(cols(&seq.1, &par.1) && cols(&seq.3, &par.3));
    }

    /// A library named by `AXTB_LAPACK`: both solver paths against faer's eigenvalues, and
    /// residuals and S-orthonormality of the eigenvectors. Needs the library, so it runs on
    /// request only, e.g.
    /// `AXTB_LAPACK=libmkl_rt.so.2 cargo test --release -p axtb-base --features linalg runtime_lapack -- --ignored`.
    #[test]
    #[ignore = "needs AXTB_LAPACK naming a LAPACK library"]
    fn runtime_lapack_matches_faer() {
        let lib = Lapack::from_env().unwrap().expect("AXTB_LAPACK names no library");
        eprintln!("library: {}", lib.describe());
        let n = 240;
        let mut seed = 7u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        };
        let mut f = Mat::<f64>::zeros(n, n);
        let mut s = Mat::<f64>::identity(n, n);
        for i in 0..n {
            for j in 0..i {
                let v = rnd();
                f[(i, j)] = v;
                f[(j, i)] = v;
                let w = 0.02 * rnd();
                s[(i, j)] = w;
                s[(j, i)] = w;
            }
        }
        let mut solver = GenEigSolver::new(s.as_ref(), Par::Seq).unwrap();
        assert!(solver.lapack.is_some());
        let (mut e1, mut c1) = (vec![0.0; n], Mat::<f64>::zeros(n, n));
        solver.solve(f.as_ref(), &mut e1, &mut c1).unwrap();
        let (mut e2, mut c2) = (vec![0.0; n], Mat::<f64>::zeros(n, n));
        let k = solver.solve_lowest(f.as_ref(), &mut e2, &mut c2, |_| n / 2).unwrap();
        assert_eq!(k, n / 2);
        // every eigenvector: the first k as solve_lowest's, to round-off (the library blocks the
        // back-transformation of n columns differently from that of k)
        let (mut e4, mut c4) = (vec![0.0; n], Mat::<f64>::zeros(n, n));
        solver.solve_all_with(f.as_ref(), &mut e4, &mut c4, false).unwrap();
        let d4 = (0..k)
            .flat_map(|j| (0..n).map(move |i| (i, j)))
            .fold(0.0f64, |m, (i, j)| m.max((c2[(i, j)] - c4[(i, j)]).abs()));
        eprintln!("solve_all_with against solve_lowest: max |dC| {d4:e}");
        assert!(d4 < 1e-12, "solve_all_with: {d4:e}");
        solver.lapack = None;
        let (mut e3, mut c3) = (vec![0.0; n], Mat::<f64>::zeros(n, n));
        solver.solve(f.as_ref(), &mut e3, &mut c3).unwrap();
        for i in 0..n {
            assert!((e1[i] - e3[i]).abs() < 1e-12, "full {i}: {} {}", e1[i], e3[i]);
            assert!((e2[i] - e3[i]).abs() < 1e-12, "lowest {i}: {} {}", e2[i], e3[i]);
        }
        for (c, k) in [(&c1, n), (&c2, n / 2)] {
            let fc = &f * c.as_ref().subcols(0, k);
            let sc = &s * c.as_ref().subcols(0, k);
            for j in 0..k {
                for i in 0..n {
                    assert!((fc[(i, j)] - e3[j] * sc[(i, j)]).abs() < 1e-11, "residual {j}");
                }
                let dot: f64 = (0..n).map(|i| c[(i, j)] * sc[(i, j)]).sum();
                assert!((dot - 1.0).abs() < 1e-11, "norm {j}");
            }
        }
    }

    /// `sygst` against two triangular solves, at sizes around its block edges: the lower
    /// triangle agrees to round-off, the strict upper triangle is not touched, and 1 and 3
    /// threads give the same bits.
    #[test]
    fn two_sided_reduction_matches_triangular_solves() {
        let mut seed = 17u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
        };
        let pool = rayon::ThreadPoolBuilder::new().num_threads(3).build().unwrap();
        for &n in &[1usize, 2, 15, 16, 17, 40, 63, 64, 65, 129, 200, 333] {
            let mut f = Mat::<f64>::zeros(n, n);
            let mut s = Mat::<f64>::identity(n, n);
            for i in 0..n {
                for j in 0..i {
                    let v = rnd();
                    f[(i, j)] = v;
                    f[(j, i)] = v;
                    let w = 0.4 * rnd() / (1.0 + 0.1 * (i - j) as f64);
                    s[(i, j)] = w;
                    s[(j, i)] = w;
                }
                f[(i, i)] = rnd();
            }
            let l = s.as_ref().llt(faer::Side::Lower).unwrap().L().to_owned();
            let mut x = f.clone();
            solve_lower_triangular_in_place(l.as_ref(), x.as_mut(), Par::Seq);
            let mut want = x.transpose().to_owned();
            solve_lower_triangular_in_place(l.as_ref(), want.as_mut(), Par::Seq);
            let scale =
                (0..n).flat_map(|j| (j..n).map(move |i| (i, j))).fold(0.0f64, |m, (i, j)| m.max(want[(i, j)].abs()));
            let reduce = |par: Par| {
                let mut a = Mat::<f64>::from_fn(n, n, |i, j| if i >= j { f[(i, j)] } else { 7.0 });
                sygst(a.as_mut(), l.as_ref(), par);
                a
            };
            let one = reduce(Par::Seq);
            assert_eq!(pool.install(|| reduce(Par::rayon(3))), one, "n = {n}: 1 and 3 threads");
            for j in 0..n {
                for i in 0..n {
                    if i < j {
                        assert_eq!(one[(i, j)], 7.0, "n = {n}: upper triangle written");
                    } else {
                        let d = (one[(i, j)] - want[(i, j)]).abs();
                        assert!(d < 1e-14 * scale, "n = {n} ({i}, {j}): {} vs {}", one[(i, j)], want[(i, j)]);
                    }
                }
            }
        }
    }

    /// A spectrum with one large cluster (60 of 100 eigenvalues within 1e-6) moves the calls
    /// after it to the full solver.
    #[test]
    fn large_clusters_use_the_full_solver() {
        // 240 rows: the full solver on the two-sided reduction, which with S = 1 is exact,
        // so the eigenpairs are `solve`'s bit for bit there too
        for n in [100, 240] {
            let mut f = Mat::<f64>::zeros(n, n);
            for i in 0..n {
                f[(i, i)] = if i < 60 { -1.0 + 1e-8 * i as f64 } else { i as f64 };
                if i > 0 {
                    f[(i, i - 1)] = 1e-9;
                    f[(i - 1, i)] = 1e-9;
                }
            }
            let s = Mat::<f64>::identity(n, n);
            let mut solver = GenEigSolver::new(s.as_ref(), Par::Seq).unwrap();
            // faer's path also when AXTB_LAPACK names a library (which has no full-solver path)
            solver.lapack = None;
            let (mut e1, mut c1) = (vec![0.0; n], Mat::<f64>::zeros(n, n));
            solver.solve(f.as_ref(), &mut e1, &mut c1).unwrap();
            let (mut e2, mut c2) = (vec![0.0; n], Mat::<f64>::zeros(n, n));
            assert_eq!(solver.solve_lowest(f.as_ref(), &mut e2, &mut c2, |_| 60).unwrap(), 60);
            for _ in 0..2 {
                let (mut e2, mut c2) = (vec![0.0; n], Mat::<f64>::zeros(n, n));
                assert_eq!(solver.solve_lowest(f.as_ref(), &mut e2, &mut c2, |_| 60).unwrap(), n);
                assert_eq!(e1, e2);
                assert_eq!(c1, c2);
            }
        }
    }
}
