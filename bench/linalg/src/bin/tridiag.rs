// SPDX-License-Identifier: LGPL-3.0-or-later
//! Tridiagonal reductions for axtb's SCF eigensolver (perf phase, P1): speed and accuracy of the
//! candidates on reduced SCF matrices A = L⁻¹ F L⁻ᵀ dumped from real runs.
//!
//!   tridiag <libmkl_rt.so> <matrix.bin>... [--threads 1,4,8] [--reps 7] [--variants v,...]
//!
//! Variants:
//!   faer22[:THR]   faer 0.22's `tridiag_in_place` (axtb's today), optional `par_threshold`
//!   faer24[:THR]   faer 0.24's (the latest release)
//!   blocked:NB     LAPACK's blocked algorithm (`dsytrd`/`dlatrd`/`dsytd2`, lower), written
//!                  fresh: panels of NB columns whose symmetric matrix-vector products run in
//!                  fixed column chunks summed in a fixed order, the rank-2k update of the trailing
//!                  matrix by faer's triangular matrix product (results independent of the thread
//!                  count)
//!   mkl            MKL's `dsytrd` (the yardstick)
//!
//! Timing: the reduction alone, median over reps, all variants interleaved per rep, inside a
//! rayon pool of the thread count (MKL: its own threads). Accuracy (the same downstream for every
//! variant: MKL's `dstedc` on T, V = Q Z): the largest eigenvalue error against MKL's `dsyevd`
//! of A, the orthogonality max|VᵀV − I|, the backward error ‖AV − VΛ‖_F / ‖A‖_F and the
//! reduction's ‖QᵀAQ − T‖_F / ‖A‖_F. Matrix files: n x n f64, column-major, little endian.
//! `TRIDIAG_PROFILE=1` adds the blocked variant's phases (symmetric products, the rest of the
//! panels, rank-2k updates, unblocked tail). Run with `KMP_BLOCKTIME=0` for the rayon variants:
//! MKL's OpenMP threads otherwise spin into their timings (and MKL itself is timed in a process of
//! its own, with its default block time, which its speed depends on).
//!
//! Results (laptop, i9-11900H; matrices of GFN2-xTB and g-xTB 2.0.1 taxol, C50H102, (H2O)100,
//! (H2O)200 and Cu(NH3)4 2+): `results/tridiag/` (`p1-tables.md` summarizes `p1-final*.tsv`).

use faer22 as faer;
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::reborrow::{Reborrow, ReborrowMut};
use faer::{Accum, Mat, MatMut, MatRef, Par};
use rayon::prelude::*;
use std::ffi::c_int;
use std::time::Instant;

type FInt = c_int;
type L = usize;

// ------------------------------------------------------------------------------- MKL (dlopen)

#[allow(non_camel_case_types)]
mod sig {
    use super::*;
    pub type dsyevd = unsafe extern "C" fn(*const u8, *const u8, *const FInt, *mut f64, *const FInt, *mut f64, *mut f64, *const FInt, *mut FInt, *const FInt, *mut FInt, L, L);
    pub type dstedc = unsafe extern "C" fn(*const u8, *const FInt, *mut f64, *mut f64, *mut f64, *const FInt, *mut f64, *const FInt, *mut FInt, *const FInt, *mut FInt, L);
    pub type dsytrd = unsafe extern "C" fn(*const u8, *const FInt, *mut f64, *const FInt, *mut f64, *mut f64, *mut f64, *mut f64, *const FInt, *mut FInt, L);
    pub type dorgtr = unsafe extern "C" fn(*const u8, *const FInt, *mut f64, *const FInt, *const f64, *mut f64, *const FInt, *mut FInt, L);
    pub type dgemm = unsafe extern "C" fn(*const u8, *const u8, *const FInt, *const FInt, *const FInt, *const f64, *const f64, *const FInt, *const f64, *const FInt, *const f64, *mut f64, *const FInt, L, L);
}

struct Mkl {
    _lib: libloading::Library,
    set_threads: unsafe extern "C" fn(c_int),
    dsyevd: sig::dsyevd,
    dstedc: sig::dstedc,
    dsytrd: sig::dsytrd,
    dorgtr: sig::dorgtr,
    dgemm: sig::dgemm,
}

impl Mkl {
    unsafe fn load(path: &str) -> Self {
        let lib = libloading::Library::new(path).unwrap_or_else(|e| panic!("dlopen {path}: {e}"));
        macro_rules! sym {
            ($t:ty, $n:expr) => {
                *lib.get::<$t>(concat!($n, "\0").as_bytes()).unwrap_or_else(|e| panic!("{}: {e}", $n))
            };
        }
        let set_iface: unsafe extern "C" fn(c_int) -> c_int = sym!(unsafe extern "C" fn(c_int) -> c_int, "MKL_Set_Interface_Layer");
        set_iface(0);
        let set_layer: unsafe extern "C" fn(c_int) -> c_int = sym!(unsafe extern "C" fn(c_int) -> c_int, "MKL_Set_Threading_Layer");
        set_layer(0); // INTEL (libiomp5)
        Mkl {
            set_threads: sym!(unsafe extern "C" fn(c_int), "MKL_Set_Num_Threads"),
            dsyevd: sym!(sig::dsyevd, "dsyevd_"),
            dstedc: sym!(sig::dstedc, "dstedc_"),
            dsytrd: sym!(sig::dsytrd, "dsytrd_"),
            dorgtr: sym!(sig::dorgtr, "dorgtr_"),
            dgemm: sym!(sig::dgemm, "dgemm_"),
            _lib: lib,
        }
    }

    fn threads(&self, t: usize) {
        unsafe { (self.set_threads)(t as c_int) }
    }

    /// Eigenvalues of the symmetric `a` (n x n, column-major).
    fn eigenvalues(&self, a: &[f64], n: usize) -> Vec<f64> {
        let mut a = a.to_vec();
        let mut w = vec![0.0; n];
        let ni = n as FInt;
        let (mut wq, mut iwq, mut info) = ([0.0], [0 as FInt], 0);
        let m1: FInt = -1;
        unsafe {
            (self.dsyevd)(b"N".as_ptr(), b"L".as_ptr(), &ni, a.as_mut_ptr(), &ni, w.as_mut_ptr(), wq.as_mut_ptr(), &m1, iwq.as_mut_ptr(), &m1, &mut info, 1, 1);
            let (lw, liw) = (wq[0] as FInt, iwq[0]);
            let mut work = vec![0.0; lw as usize];
            let mut iwork = vec![0 as FInt; liw as usize];
            (self.dsyevd)(b"N".as_ptr(), b"L".as_ptr(), &ni, a.as_mut_ptr(), &ni, w.as_mut_ptr(), work.as_mut_ptr(), &lw, iwork.as_mut_ptr(), &liw, &mut info, 1, 1);
        }
        assert_eq!(info, 0, "dsyevd");
        w
    }

    /// Every eigenpair of the tridiagonal (d, e): (eigenvalues, eigenvectors column-major).
    fn tridiagonal_eigen(&self, d: &[f64], e: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = d.len();
        let (mut d, mut e) = (d.to_vec(), e.to_vec());
        e.push(0.0);
        let mut z = vec![0.0; n * n];
        let ni = n as FInt;
        let (mut wq, mut iwq, mut info) = ([0.0], [0 as FInt], 0);
        let m1: FInt = -1;
        unsafe {
            (self.dstedc)(b"I".as_ptr(), &ni, d.as_mut_ptr(), e.as_mut_ptr(), z.as_mut_ptr(), &ni, wq.as_mut_ptr(), &m1, iwq.as_mut_ptr(), &m1, &mut info, 1);
            let (lw, liw) = (wq[0] as FInt, iwq[0]);
            let mut work = vec![0.0; lw as usize];
            let mut iwork = vec![0 as FInt; liw as usize];
            (self.dstedc)(b"I".as_ptr(), &ni, d.as_mut_ptr(), e.as_mut_ptr(), z.as_mut_ptr(), &ni, work.as_mut_ptr(), &lw, iwork.as_mut_ptr(), &liw, &mut info, 1);
        }
        assert_eq!(info, 0, "dstedc");
        (d, z)
    }

    /// LAPACK's `dsytrd` (lower) in place: (d, e, tau).
    fn sytrd(&self, a: &mut [f64], n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let ni = n as FInt;
        let (mut d, mut e, mut tau) = (vec![0.0; n], vec![0.0; n.saturating_sub(1)], vec![0.0; n.saturating_sub(1)]);
        let (mut wq, mut info) = ([0.0], 0);
        let m1: FInt = -1;
        unsafe {
            (self.dsytrd)(b"L".as_ptr(), &ni, a.as_mut_ptr(), &ni, d.as_mut_ptr(), e.as_mut_ptr(), tau.as_mut_ptr(), wq.as_mut_ptr(), &m1, &mut info, 1);
            let lw = wq[0] as FInt;
            let mut work = vec![0.0; lw as usize];
            (self.dsytrd)(b"L".as_ptr(), &ni, a.as_mut_ptr(), &ni, d.as_mut_ptr(), e.as_mut_ptr(), tau.as_mut_ptr(), work.as_mut_ptr(), &lw, &mut info, 1);
        }
        assert_eq!(info, 0, "dsytrd");
        (d, e, tau)
    }

    /// Q of a lower `dsytrd` factorization (reflectors below the subdiagonal of `a`, `tau`).
    fn orgtr(&self, a: &[f64], n: usize, tau: &[f64]) -> Vec<f64> {
        let mut q = a.to_vec();
        let ni = n as FInt;
        let (mut wq, mut info) = ([0.0], 0);
        let m1: FInt = -1;
        unsafe {
            (self.dorgtr)(b"L".as_ptr(), &ni, q.as_mut_ptr(), &ni, tau.as_ptr(), wq.as_mut_ptr(), &m1, &mut info, 1);
            let lw = wq[0] as FInt;
            let mut work = vec![0.0; lw as usize];
            (self.dorgtr)(b"L".as_ptr(), &ni, q.as_mut_ptr(), &ni, tau.as_ptr(), work.as_mut_ptr(), &lw, &mut info, 1);
        }
        assert_eq!(info, 0, "dorgtr");
        q
    }

    /// C = op(A) op(B) for n x n column-major matrices.
    fn gemm(&self, ta: u8, tb: u8, n: usize, a: &[f64], b: &[f64]) -> Vec<f64> {
        let mut c = vec![0.0; n * n];
        let ni = n as FInt;
        unsafe { (self.dgemm)(&ta, &tb, &ni, &ni, &ni, &1.0, a.as_ptr(), &ni, b.as_ptr(), &ni, &0.0, c.as_mut_ptr(), &ni, 1, 1) };
        c
    }
}

// ------------------------------------------------------------------------------- faer variants

macro_rules! faer_variant {
    ($modname:ident, $faer:ident, $bs:ident) => {
        mod $modname {
            use $faer::dyn_stack::{MemBuffer, MemStack};
            use $faer::linalg::evd::tridiag::{tridiag_in_place, tridiag_in_place_scratch, TridiagParams};
            use $faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_with_conj;
            use $faer::{Auto, Conj, Mat, MatRef, Par, Spec};

            pub struct Out {
                pub a: Mat<f64>,
                pub h: Mat<f64>,
            }

            pub fn params(thr: Option<usize>) -> Spec<TridiagParams, f64> {
                let mut p = <TridiagParams as Auto<f64>>::auto();
                if let Some(t) = thr {
                    p.par_threshold = t;
                }
                Spec::new(p)
            }

            /// The reduction (timed part): `a` (lower) becomes T and the reflectors.
            pub fn run(src: &[f64], n: usize, threads: usize, thr: Option<usize>) -> (Out, f64) {
                let par = if threads > 1 { Par::rayon(threads) } else { Par::Seq };
                let mut a = Mat::<f64>::from_fn(n, n, |i, j| src[i + j * n]);
                let bs = $faer::linalg::qr::no_pivoting::factor::$bs::<f64>(n, n);
                let mut h = Mat::<f64>::zeros(bs, n - 1);
                let mut mem = MemBuffer::new(tridiag_in_place_scratch::<f64>(n, par, params(thr)));
                let t = std::time::Instant::now();
                tridiag_in_place(a.as_mut(), h.as_mut(), par, MemStack::new(&mut mem), params(thr));
                (Out { a, h }, t.elapsed().as_secs_f64())
            }

            /// (d, e, Q column-major).
            pub fn result(out: &Out, n: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
                let d = (0..n).map(|i| out.a[(i, i)]).collect();
                let e = (0..n - 1).map(|i| out.a[(i + 1, i)]).collect();
                let mut q = Mat::<f64>::identity(n, n);
                let bs = out.h.nrows();
                let mut mem = MemBuffer::new(
                    $faer::linalg::householder::apply_block_householder_sequence_on_the_left_in_place_scratch::<f64>(n - 1, bs, n),
                );
                let ess: MatRef<'_, f64> = out.a.as_ref().submatrix(1, 0, n - 1, n - 1);
                apply_block_householder_sequence_on_the_left_in_place_with_conj(
                    ess,
                    out.h.as_ref(),
                    Conj::No,
                    q.as_mut().submatrix_mut(1, 0, n - 1, n),
                    Par::Seq,
                    MemStack::new(&mut mem),
                );
                let qv = (0..n * n).map(|k| q[(k % n, k / n)]).collect();
                (d, e, qv)
            }
        }
    };
}

faer_variant!(f22, faer22, recommended_blocksize);
faer_variant!(f24, faer24, recommended_block_size);

// ------------------------------------------------------------------------------- blocked

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
/// Profile of the blocked variant (ns): symv, latrd total, syr2k, sytd2.
static PROF: [AtomicU64; 4] = [AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0)];
fn prof(k: usize, t: Instant) {
    PROF[k].fetch_add(t.elapsed().as_nanos() as u64, Relaxed);
}

/// Column parts of the symmetric matrix-vector product, of about equal lower-triangle area
/// (fixed: the sums do not depend on the thread count).
const SYMV_PARTS: usize = 16;

/// Rows from which the symmetric product runs in parallel, and the row loops (fixed sizes: the
/// parallel and sequential paths compute the same sums).
const PAR_SYMV: usize = 256;
const PAR_ROWS: usize = 512;

/// LAPACK's `dlarfg`: the reflector H = I − τ (1, v)(1, v)ᵀ with H (alpha, x) = (beta, 0); x
/// becomes v. Returns (beta, tau).
fn larfg(alpha: f64, x: &mut [f64]) -> (f64, f64) {
    if x.is_empty() {
        return (alpha, 0.0);
    }
    let mut xnorm = nrm2(x);
    if xnorm == 0.0 {
        return (alpha, 0.0);
    }
    let mut alpha = alpha;
    let mut beta = -alpha.hypot(xnorm).copysign(alpha);
    let safmin = f64::MIN_POSITIVE / f64::EPSILON;
    let mut knt = 0;
    if beta.abs() < safmin {
        let rsafmn = 1.0 / safmin;
        loop {
            knt += 1;
            x.iter_mut().for_each(|v| *v *= rsafmn);
            beta *= rsafmn;
            alpha *= rsafmn;
            if beta.abs() >= safmin || knt >= 20 {
                break;
            }
        }
        xnorm = nrm2(x);
        beta = -alpha.hypot(xnorm).copysign(alpha);
    }
    let tau = (beta - alpha) / beta;
    let s = 1.0 / (alpha - beta);
    x.iter_mut().for_each(|v| *v *= s);
    for _ in 0..knt {
        beta *= safmin;
    }
    (beta, tau)
}

/// The 2-norm with scaling (LAPACK's `dnrm2` semantics for the ranges that occur).
fn nrm2(x: &[f64]) -> f64 {
    let scale = x.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    if scale == 0.0 || !scale.is_finite() {
        return scale;
    }
    let s: f64 = x.iter().map(|v| (v / scale) * (v / scale)).sum();
    scale * s.sqrt()
}

/// Dot product with eight partial sums (a fixed order).
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = [0.0; 8];
    let (ac, bc) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ar, br) = (ac.remainder(), bc.remainder());
    for (x, y) in ac.zip(bc) {
        for l in 0..8 {
            acc[l] += x[l] * y[l];
        }
    }
    let mut r = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
    for (x, y) in ar.iter().zip(br) {
        r += x * y;
    }
    r
}

/// y += s x.
#[inline]
fn axpy(s: f64, x: &[f64], y: &mut [f64]) {
    for (yv, xv) in y.iter_mut().zip(x) {
        *yv += s * xv;
    }
}

/// The first column of each of the SYMV_PARTS parts of an m x m lower triangle (equal areas).
fn symv_bounds(m: usize) -> [usize; SYMV_PARTS + 1] {
    let mut b = [0; SYMV_PARTS + 1];
    for k in 0..=SYMV_PARTS {
        let f = 1.0 - (1.0 - k as f64 / SYMV_PARTS as f64).sqrt();
        b[k] = ((f * m as f64) as usize).min(m);
    }
    b[SYMV_PARTS] = m;
    b
}

/// Columns c0..c1 of y = A v into `yc` (rows c0..m; the rows of `yc` before c0 are not touched).
fn symv_part(m: usize, a: &[f64], lda: usize, v: &[f64], c0: usize, c1: usize, yc: &mut [f64]) {
    yc[c0..m].iter_mut().for_each(|x| *x = 0.0);
    for j in c0..c1 {
        let col = &a[j * lda + j..j * lda + m];
        let vj = v[j];
        let (colr, vr) = (&col[1..], &v[j + 1..m]);
        let yr = &mut yc[j + 1..m];
        let mut acc = [0.0; 8];
        let (cc, vc) = (colr.chunks_exact(8), vr.chunks_exact(8));
        let (crm, vrm) = (cc.remainder(), vc.remainder());
        let mut yc8 = yr.chunks_exact_mut(8);
        for ((c, vv), y) in cc.zip(vc).zip(&mut yc8) {
            for l in 0..8 {
                y[l] += c[l] * vj;
                acc[l] += c[l] * vv[l];
            }
        }
        let mut t = ((acc[0] + acc[1]) + (acc[2] + acc[3])) + ((acc[4] + acc[5]) + (acc[6] + acc[7]));
        for ((c, vv), y) in crm.iter().zip(vrm).zip(yc8.into_remainder()) {
            *y += c * vj;
            t += c * vv;
        }
        yc[j] += col[0] * vj + t;
    }
}

/// y = A v for the symmetric m x m `A` (lower triangle, column-major, column stride `lda`,
/// starting at `a[0]`): SYMV_PARTS fixed column parts, each into its own row of `part`, then the
/// parts summed in order (row r: the parts that start at or before r). The parts in parallel on
/// the current pool when `par`.
fn symv_lower(m: usize, a: &[f64], lda: usize, v: &[f64], y: &mut [f64], part: &mut [f64], par: bool) {
    let b = symv_bounds(m);
    let part = &mut part[..SYMV_PARTS * m];
    if par && m >= PAR_SYMV {
        part.par_chunks_mut(m).enumerate().for_each(|(k, yc)| symv_part(m, a, lda, v, b[k], b[k + 1], yc));
    } else {
        for (k, yc) in part.chunks_mut(m).enumerate() {
            symv_part(m, a, lda, v, b[k], b[k + 1], yc);
        }
    }
    y[..m].iter_mut().for_each(|x| *x = 0.0);
    for k in 0..SYMV_PARTS {
        if b[k] < b[k + 1] {
            for (yv, p) in y[b[k]..m].iter_mut().zip(&part[k * m + b[k]..(k + 1) * m]) {
                *yv += p;
            }
        }
    }
}


/// y = Xᵀ x for the column-major `rows x cols` X (stride ldx).
fn gemv_t(rows: usize, cols: usize, x: &[f64], ldx: usize, xv: &[f64], y: &mut [f64]) {
    for j in 0..cols {
        y[j] = dot(&x[j * ldx..j * ldx + rows], &xv[..rows]);
    }
}

/// LAPACK's `dlatrd` (lower) on the m x m matrix at `a` (stride lda): reduces its first `nb`
/// columns; W (m x nb, stride m) the update's second factor.
fn latrd_lower(m: usize, nb: usize, a: &mut [f64], lda: usize, e: &mut [f64], tau: &mut [f64], w: &mut [f64], part: &mut [f64], par: bool) {
    let ldw = m;
    let mut tmp1 = vec![0.0; nb];
    let mut tmp2 = vec![0.0; nb];
    let mut wrow = vec![0.0; nb];
    let mut arow = vec![0.0; nb];
    let mut v = vec![0.0; m];
    let mut y = vec![0.0; m];
    // row blocks of the row-parallel loops (each row's sum in the order of the sequential gemvs)
    const RB: usize = 64;
    let rpar = |rows: usize| par && rows >= PAR_ROWS;
    for i in 0..nb {
        // A(i:m, i) -= A(i:m, 0:i) W(i, 0:i)ᵀ + W(i:m, 0:i) A(i, 0:i)ᵀ
        if i > 0 {
            for k in 0..i {
                wrow[k] = -w[i + k * ldw];
                arow[k] = -a[i + k * lda];
            }
            let rows = m - i;
            let (aread, acol) = a.split_at_mut(i * lda);
            let acol = &mut acol[i..m];
            let (wr, ar) = (&wrow[..i], &arow[..i]);
            let upd = |r0: usize, out: &mut [f64]| {
                let (r, len) = (i + r0, out.len());
                for c in 0..i {
                    axpy(wr[c], &aread[r + c * lda..r + c * lda + len], out);
                }
                for c in 0..i {
                    axpy(ar[c], &w[r + c * ldw..r + c * ldw + len], out);
                }
            };
            if rpar(rows) {
                acol.par_chunks_mut(RB).enumerate().for_each(|(b, out)| upd(b * RB, out));
            } else {
                upd(0, acol);
            }
        }
        if i + 1 < m {
            // the reflector of A(i+1:m, i)
            let (beta, t) = {
                let c = &mut a[i * lda + i + 1..i * lda + m];
                let (head, tail) = c.split_at_mut(1);
                let r = larfg(head[0], tail);
                head[0] = 1.0;
                r
            };
            e[i] = beta;
            tau[i] = t;
            let rows = m - i - 1;
            v[..rows].copy_from_slice(&a[i * lda + i + 1..i * lda + m]);
            let vv = &v[..rows];
            // tmp1 = W(i+1:m, 0:i)ᵀ v, tmp2 = A(i+1:m, 0:i)ᵀ v, and the parts of A22 v, together
            let tp = Instant::now();
            let sub = &a[(i + 1) * lda + i + 1..];
            let b = symv_bounds(rows);
            let part = &mut part[..SYMV_PARTS * rows];
            let (wv, av) = (&w[i + 1..], &a[i + 1..]);
            let dots = |t1: &mut [f64], t2: &mut [f64]| {
                gemv_t(rows, i, wv, ldw, vv, t1);
                gemv_t(rows, i, av, lda, vv, t2);
            };
            if par && rows >= PAR_SYMV {
                rayon::join(
                    || part.par_chunks_mut(rows).enumerate().for_each(|(k, yc)| symv_part(rows, sub, lda, vv, b[k], b[k + 1], yc)),
                    || dots(&mut tmp1[..i], &mut tmp2[..i]),
                );
            } else {
                for (k, yc) in part.chunks_mut(rows).enumerate() {
                    symv_part(rows, sub, lda, vv, b[k], b[k + 1], yc);
                }
                dots(&mut tmp1[..i], &mut tmp2[..i]);
            }
            prof(0, tp);
            // y = (A22 v − A(:, 0:i) tmp1 − W(:, 0:i) tmp2) t, per row in that order
            let (t1, t2, part) = (&tmp1[..i], &tmp2[..i], &*part);
            let fin = |r0: usize, out: &mut [f64]| {
                let len = out.len();
                out.iter_mut().for_each(|x| *x = 0.0);
                for p in 0..SYMV_PARTS {
                    // the part's rows at or after its first column
                    if b[p] < b[p + 1] && b[p] < r0 + len {
                        let from = b[p].max(r0);
                        for (x, q) in out[from - r0..].iter_mut().zip(&part[p * rows + from..p * rows + r0 + len]) {
                            *x += q;
                        }
                    }
                }
                for c in 0..i {
                    axpy(-t1[c], &av[r0 + c * lda..r0 + c * lda + len], out);
                }
                for c in 0..i {
                    axpy(-t2[c], &wv[r0 + c * ldw..r0 + c * ldw + len], out);
                }
                out.iter_mut().for_each(|x| *x *= t);
            };
            if rpar(rows) {
                y[..rows].par_chunks_mut(RB).enumerate().for_each(|(bk, out)| fin(bk * RB, out));
            } else {
                fin(0, &mut y[..rows]);
            }
            let alpha = -0.5 * t * dot(&y[..rows], vv);
            for k in 0..rows {
                y[k] += alpha * vv[k];
            }
            w[i * ldw + i + 1..i * ldw + m].copy_from_slice(&y[..rows]);
        }
    }
}

/// LAPACK's unblocked `dsytd2` (lower) on the m x m matrix at `a`.
fn sytd2_lower(m: usize, a: &mut [f64], lda: usize, d: &mut [f64], e: &mut [f64], tau: &mut [f64], part: &mut [f64], par: bool) {
    for i in 0..m.saturating_sub(1) {
        let (beta, t) = {
            let c = &mut a[i * lda + i + 1..i * lda + m];
            let (head, tail) = c.split_at_mut(1);
            larfg(head[0], tail)
        };
        e[i] = beta;
        if t != 0.0 {
            a[i * lda + i + 1] = 1.0;
            let rows = m - i - 1;
            let v: Vec<f64> = a[i * lda + i + 1..i * lda + m].to_vec();
            let mut w = vec![0.0; rows];
            symv_lower(rows, &a[(i + 1) * lda + i + 1..], lda, &v, &mut w, part, par);
            w.iter_mut().for_each(|x| *x *= t);
            let alpha = -0.5 * t * dot(&w, &v);
            for k in 0..rows {
                w[k] += alpha * v[k];
            }
            // A22 -= v wᵀ + w vᵀ (lower)
            for j in 0..rows {
                let (vj, wj) = (v[j], w[j]);
                let c = &mut a[(i + 1 + j) * lda + i + 1 + j..(i + 1 + j) * lda + m];
                for (k, x) in c.iter_mut().enumerate() {
                    *x -= v[j + k] * wj + w[j + k] * vj;
                }
            }
            a[i * lda + i + 1] = beta;
        }
        d[i] = a[i * lda + i];
        tau[i] = t;
    }
    if m > 0 {
        d[m - 1] = a[(m - 1) * lda + m - 1];
    }
}

/// LAPACK's blocked `dsytrd` (lower) in place on the n x n column-major `a`: (d, e, tau), the
/// reflectors below the subdiagonal (LAPACK's layout).
fn sytrd_blocked(a: &mut [f64], n: usize, nb: usize, nx: usize, threads: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let par = threads > 1;
    let fpar = if par { Par::rayon(threads) } else { Par::Seq };
    let (mut d, mut e, mut tau) = (vec![0.0; n], vec![0.0; n.saturating_sub(1)], vec![0.0; n.saturating_sub(1)]);
    let mut part = vec![0.0; SYMV_PARTS * n];
    let mut w = vec![0.0; n * nb];
    let nx = nx.max(nb);
    let mut i = 0;
    while n > nx && i < n - nx {
        let m = n - i;
        {
            let sub = &mut a[i * n + i..];
            let tp = Instant::now();
            latrd_lower(m, nb, sub, n, &mut e[i..], &mut tau[i..], &mut w[..m * nb], &mut part, par);
            prof(1, tp);
        }
        // A(i+nb:n, i+nb:n) -= V Wᵀ + W Vᵀ, V = A(i+nb:n, i:i+nb), W = W(nb:m, 0:nb)
        let mt = m - nb;
        let tp = Instant::now();
        {
            let full = MatMut::from_column_major_slice_with_stride_mut(&mut a[..], n, n, n);
            let (left, right) = full.split_at_col_mut(i + nb);
            let v = left.rb().submatrix(i + nb, i, mt, nb);
            let wm = MatRef::from_column_major_slice_with_stride(&w[nb..], mt, nb, m);
            let mut a22 = right.submatrix_mut(i + nb, 0, mt, mt);
            use faer::linalg::matmul::triangular::{matmul, BlockStructure};
            matmul(a22.rb_mut(), BlockStructure::TriangularLower, Accum::Add, v, BlockStructure::Rectangular, wm.transpose(), BlockStructure::Rectangular, -1.0, fpar);
            matmul(a22.rb_mut(), BlockStructure::TriangularLower, Accum::Add, wm, BlockStructure::Rectangular, v.transpose(), BlockStructure::Rectangular, -1.0, fpar);
        }
        prof(2, tp);
        for j in i..i + nb {
            a[j * n + j + 1] = e[j];
            d[j] = a[j * n + j];
        }
        i += nb;
    }
    {
        let m = n - i;
        let sub = &mut a[i * n + i..];
        let tp = Instant::now();
        sytd2_lower(m, sub, n, &mut d[i..], &mut e[i..], &mut tau[i..], &mut part, par);
        prof(3, tp);
    }
    (d, e, tau)
}

// ------------------------------------------------------------------------------- measurements

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

struct Acc {
    eig: f64,
    orth: f64,
    resid: f64,
    red: f64,
}

/// Accuracy of (d, e, Q) for A (all column-major n x n).
fn accuracy(mkl: &Mkl, a: &[f64], n: usize, lref: &[f64], d: &[f64], e: &[f64], q: &[f64]) -> Acc {
    let anorm = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let (lam, z) = mkl.tridiagonal_eigen(d, e);
    let eig = lam.iter().zip(lref).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max);
    let v = mkl.gemm(b'N', b'N', n, q, &z);
    let vtv = mkl.gemm(b'T', b'N', n, &v, &v);
    let orth = (0..n * n).map(|k| (vtv[k] - if k % n == k / n { 1.0 } else { 0.0 }).abs()).fold(0.0, f64::max);
    let av = mkl.gemm(b'N', b'N', n, a, &v);
    let resid = (0..n * n).map(|k| av[k] - v[k] * lam[k / n]).map(|x| x * x).sum::<f64>().sqrt() / anorm;
    // ‖QᵀAQ − T‖_F / ‖A‖_F
    let aq = mkl.gemm(b'N', b'N', n, a, q);
    let qaq = mkl.gemm(b'T', b'N', n, q, &aq);
    let mut s = 0.0;
    for j in 0..n {
        for i in 0..n {
            let t = if i == j {
                d[i]
            } else if i == j + 1 {
                e[j]
            } else if j == i + 1 {
                e[i]
            } else {
                0.0
            };
            s += (qaq[i + j * n] - t).powi(2);
        }
    }
    Acc { eig, orth, resid, red: s.sqrt() / anorm }
}

#[derive(Clone)]
enum Variant {
    Faer22(Option<usize>),
    Faer24(Option<usize>),
    Blocked(usize, usize),
    Mkl,
}

impl Variant {
    fn name(&self) -> String {
        match self {
            Variant::Faer22(None) => "faer22".into(),
            Variant::Faer22(Some(t)) => format!("faer22:thr{t}"),
            Variant::Faer24(None) => "faer24".into(),
            Variant::Faer24(Some(t)) => format!("faer24:thr{t}"),
            Variant::Blocked(nb, nx) => format!("blocked:nb{nb}:nx{nx}"),
            Variant::Mkl => "mkl".into(),
        }
    }
    fn parse(s: &str) -> Self {
        let mut it = s.split(':');
        match it.next().unwrap() {
            "faer22" => Variant::Faer22(it.next().map(|t| t.parse().unwrap())),
            "faer24" => Variant::Faer24(it.next().map(|t| t.parse().unwrap())),
            "blocked" => Variant::Blocked(it.next().unwrap().parse().unwrap(), it.next().map_or(64, |t| t.parse().unwrap())),
            "mkl" => Variant::Mkl,
            v => panic!("unknown variant {v}"),
        }
    }
    /// Timed reduction; returns (seconds, d, e, Q) (Q only when `want_q`).
    fn run(&self, mkl: &Mkl, src: &[f64], n: usize, threads: usize, want_q: bool) -> (f64, Vec<f64>, Vec<f64>, Vec<f64>) {
        match self {
            Variant::Faer22(thr) => {
                let (o, s) = f22::run(src, n, threads, *thr);
                let (d, e, q) = if want_q { f22::result(&o, n) } else { (vec![], vec![], vec![]) };
                (s, d, e, q)
            }
            Variant::Faer24(thr) => {
                let (o, s) = f24::run(src, n, threads, *thr);
                let (d, e, q) = if want_q { f24::result(&o, n) } else { (vec![], vec![], vec![]) };
                (s, d, e, q)
            }
            Variant::Blocked(nb, nx) => {
                let mut a = src.to_vec();
                let t = Instant::now();
                let (d, e, tau) = sytrd_blocked(&mut a, n, *nb, *nx, threads);
                let s = t.elapsed().as_secs_f64();
                let q = if want_q { mkl.orgtr(&a, n, &tau) } else { vec![] };
                (s, d, e, q)
            }
            Variant::Mkl => {
                mkl.threads(threads);
                let mut a = src.to_vec();
                let t = Instant::now();
                let (d, e, tau) = mkl.sytrd(&mut a, n);
                let s = t.elapsed().as_secs_f64();
                let q = if want_q { mkl.orgtr(&a, n, &tau) } else { vec![] };
                (s, d, e, q)
            }
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mkl = unsafe { Mkl::load(&args[1]) };
    let mut files = Vec::new();
    let mut threads = vec![1, 4, 8];
    let mut reps = 7;
    let mut variants: Vec<Variant> = ["faer22", "faer22:0", "faer24", "blocked:32", "blocked:64", "mkl"].iter().map(|s| Variant::parse(s)).collect();
    let mut acc_only = false;
    let mut it = args[2..].iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--threads" => threads = it.next().unwrap().split(',').map(|t| t.parse().unwrap()).collect(),
            "--reps" => reps = it.next().unwrap().parse().unwrap(),
            "--variants" => variants = it.next().unwrap().split(',').map(Variant::parse).collect(),
            "--accuracy-only" => acc_only = true,
            f => files.push(f.to_string()),
        }
    }
    println!("# file\tn\tvariant\tthreads\tmedian_ms\tmin_ms\teig_err\torth\tresid\tred\tbits_vs_1t");
    for f in &files {
        let bytes = std::fs::read(f).unwrap();
        let src: Vec<f64> = bytes.chunks_exact(8).map(|c| f64::from_le_bytes(c.try_into().unwrap())).collect();
        let n = (src.len() as f64).sqrt().round() as usize;
        assert_eq!(n * n, src.len());
        let name = std::path::Path::new(f).file_stem().unwrap().to_string_lossy().to_string();
        mkl.threads(8);
        let lref = mkl.eigenvalues(&src, n);
        // accuracy (and the bits of each thread count against 1 thread)
        for v in &variants {
            let mut first: Option<(Vec<u64>, Vec<u64>)> = None;
            for &t in &threads {
                let pool = rayon::ThreadPoolBuilder::new().num_threads(t).build().unwrap();
                let (_, d, e, q) = pool.install(|| v.run(&mkl, &src, n, t, true));
                let bits = (d.iter().map(|x| x.to_bits()).collect::<Vec<_>>(), e.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
                let same = match &first {
                    None => {
                        first = Some(bits);
                        "-".to_string()
                    }
                    Some(b) => (if *b == bits { "same" } else { "differ" }).to_string(),
                };
                mkl.threads(8);
                let acc = accuracy(&mkl, &src, n, &lref, &d, &e, &q);
                if acc_only || t == threads[0] || same == "differ" {
                    println!("{name}\t{n}\t{}\t{t}\t-\t-\t{:.2e}\t{:.2e}\t{:.2e}\t{:.2e}\t{same}", v.name(), acc.eig, acc.orth, acc.resid, acc.red);
                } else {
                    println!("{name}\t{n}\t{}\t{t}\t-\t-\t-\t-\t-\t-\t{same}", v.name());
                }
            }
        }
        if acc_only {
            continue;
        }
        // timing: per thread count, every variant once per rep, interleaved
        for &t in &threads {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(t).build().unwrap();
            let mut times: Vec<Vec<f64>> = vec![Vec::new(); variants.len()];
            for v in &variants {
                pool.install(|| v.run(&mkl, &src, n, t, false)); // warm-up
            }
            for r in 0..reps {
                for k in 0..variants.len() {
                    let k = if r % 2 == 0 { k } else { variants.len() - 1 - k };
                    let (s, ..) = pool.install(|| variants[k].run(&mkl, &src, n, t, false));
                    times[k].push(s);
                }
            }
            for (v, ts) in variants.iter().zip(times.iter_mut()) {
                if std::env::var_os("TRIDIAG_PROFILE").is_some() {
                    if let Variant::Blocked(..) = v {
                        for p in PROF.iter() {
                            p.store(0, Relaxed);
                        }
                        let k = 3;
                        for _ in 0..k {
                            pool.install(|| v.run(&mkl, &src, n, t, false));
                        }
                        let f = |i: usize| PROF[i].load(Relaxed) as f64 / 1e6 / k as f64;
                        println!("# profile {name} {} {t}t: symv {:.2} latrd {:.2} (other {:.2}) syr2k {:.2} sytd2 {:.2} ms", v.name(), f(0), f(1), f(1) - f(0), f(2), f(3));
                    }
                }
                let min = ts.iter().cloned().fold(f64::INFINITY, f64::min);
                println!("{name}\t{n}\t{}\t{t}\t{:.3}\t{:.3}\t-\t-\t-\t-\t-", v.name(), 1e3 * median(ts), 1e3 * min);
            }
        }
    }
    let _ = (MemBuffer::new(faer::dyn_stack::StackReq::EMPTY), MemStack::new);
    let _ = Mat::<f64>::zeros(0, 0);
}
