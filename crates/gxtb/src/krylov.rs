//! Krylov solvers of the g-xTB response equations and their vector operations:
//!
//! - [`cg`]: conjugate gradients for the symmetric positive definite coupled-perturbed system of
//!   2.0.1 (`crate::response`), on block vectors (one faer matrix per spin channel) with the
//!   eight-partial-sum dot product ([`block_dot`]);
//! - [`gmres`]: right-preconditioned restarted GMRES for v1.1's non-symmetric Z-vector equations
//!   (the response of the program's Fock matrix), on plain vectors with the sequential dot product.
//!
//! Both start from x = 0 and stop at |b - A x| <= tol |b|; the convergence settings and the
//! messages of a failure belong to the callers.

use crate::dense::{dot, dot_seq};
use axtb::Error;
use faer::Mat;

/// `y += a x` elementwise.
#[inline]
pub fn add_scaled(a: f64, x: &[f64], y: &mut [f64]) {
    for (u, v) in y.iter_mut().zip(x) {
        *u += a * v;
    }
}

/// `y -= a x` elementwise.
#[inline]
pub fn sub_scaled(a: f64, x: &[f64], y: &mut [f64]) {
    for (u, v) in y.iter_mut().zip(x) {
        *u -= a * v;
    }
}

/// The inner product of two block vectors: the eight-partial-sum dot products of the columns
/// of every block, summed in order.
pub fn block_dot(a: &[Mat<f64>], b: &[Mat<f64>]) -> f64 {
    let mut s = 0.0;
    for (x, y) in a.iter().zip(b) {
        for j in 0..x.ncols() {
            s += dot(x.col_as_slice(j), y.col_as_slice(j));
        }
    }
    s
}

/// The Euclidean norm with the sequential dot product.
fn norm(a: &[f64]) -> f64 {
    dot_seq(a, a).sqrt()
}

/// A conjugate-gradient solution.
pub struct CgSolution {
    pub x: Vec<Mat<f64>>,
    /// The final residual b - A x (`None` for a zero right-hand side, where x = 0).
    pub residual: Option<Vec<Mat<f64>>>,
    pub iterations: usize,
}

/// Why [`cg`] stopped without a solution.
#[derive(Clone, Copy, Debug)]
pub enum CgError {
    /// `max_iter` iterations without convergence; the relative residual |r| / |b| reached.
    NotConverged { iterations: usize, residual: f64 },
    /// A direction with p.Ap <= 0: the operator is not positive definite.
    Indefinite,
}

/// Conjugate gradients for `apply(x) = b` with a symmetric positive definite operator on block
/// vectors, until |r| <= `tol` |b| (at most `max_iter` operator applications).
pub fn cg(
    apply: impl Fn(&[Mat<f64>]) -> Vec<Mat<f64>>,
    b: &[Mat<f64>],
    tol: f64,
    max_iter: usize,
) -> Result<CgSolution, CgError> {
    let bnorm = block_dot(b, b).sqrt();
    let mut x: Vec<Mat<f64>> = b.iter().map(|r| Mat::<f64>::zeros(r.nrows(), r.ncols())).collect();
    let mut iterations = 0;
    let mut residual = None;
    if bnorm > 0.0 {
        let mut r = b.to_vec();
        let mut p = r.clone();
        let mut rr = block_dot(&r, &r);
        loop {
            if rr.sqrt() <= tol * bnorm {
                break;
            }
            if iterations >= max_iter {
                return Err(CgError::NotConverged { iterations, residual: rr.sqrt() / bnorm });
            }
            iterations += 1;
            let ap = apply(&p);
            let pap = block_dot(&p, &ap);
            if pap <= 0.0 {
                return Err(CgError::Indefinite);
            }
            let alpha = rr / pap;
            for k in 0..x.len() {
                for j in 0..x[k].ncols() {
                    add_scaled(alpha, p[k].col_as_slice(j), x[k].col_as_slice_mut(j));
                    sub_scaled(alpha, ap[k].col_as_slice(j), r[k].col_as_slice_mut(j));
                }
            }
            let rr_new = block_dot(&r, &r);
            let beta = rr_new / rr;
            rr = rr_new;
            for k in 0..p.len() {
                for j in 0..p[k].ncols() {
                    for (u, v) in p[k].col_as_slice_mut(j).iter_mut().zip(r[k].col_as_slice(j)) {
                        *u = v + beta * *u;
                    }
                }
            }
        }
        residual = Some(r);
    }
    Ok(CgSolution { x, residual, iterations })
}

/// Why [`gmres`] stopped without a solution.
#[derive(Debug)]
pub enum GmresError {
    /// The operator failed.
    Operator(Error),
    /// `max_iter` operator applications without convergence; the relative residual reached.
    NotConverged { residual: f64 },
}

impl From<Error> for GmresError {
    fn from(e: Error) -> Self {
        GmresError::Operator(e)
    }
}

/// Right-preconditioned restarted GMRES for `op(x) = b` with the diagonal preconditioner `diag`
/// (x = u / diag where |diag| > 1e-12), restarted every `restart` steps, until
/// |b - op(x)| <= `tol` |b| (at most `max_iter` preconditioned steps; an inner cycle also ends
/// once its estimated residual is below 0.1 `tol` |b|).
pub fn gmres(
    mut op: impl FnMut(&[f64]) -> axtb::Result<Vec<f64>>,
    diag: &[f64],
    b: &[f64],
    tol: f64,
    max_iter: usize,
    restart: usize,
) -> Result<Vec<f64>, GmresError> {
    let dim = b.len();
    let mut x = vec![0.0; dim];
    let bnorm = norm(b);
    if bnorm == 0.0 {
        return Ok(x);
    }
    let pre =
        |v: &[f64]| -> Vec<f64> { v.iter().zip(diag).map(|(a, d)| if d.abs() > 1e-12 { a / d } else { *a }).collect() };
    let mut iters = 0;
    loop {
        let ax = op(&x)?;
        let r: Vec<f64> = b.iter().zip(&ax).map(|(a, c)| a - c).collect();
        let beta = norm(&r);
        if beta <= tol * bnorm {
            return Ok(x);
        }
        if iters >= max_iter {
            return Err(GmresError::NotConverged { residual: beta / bnorm });
        }
        let m = restart.min(dim);
        let mut v: Vec<Vec<f64>> = vec![r.iter().map(|a| a / beta).collect()];
        let mut h = vec![vec![0.0; m]; m + 1];
        let (mut cs, mut sn) = (vec![0.0; m], vec![0.0; m]);
        let mut e1 = vec![0.0; m + 1];
        e1[0] = beta;
        let mut k_used = 0;
        for k in 0..m {
            iters += 1;
            let mut w = op(&pre(&v[k]))?;
            for j in 0..=k {
                h[j][k] = dot_seq(&w, &v[j]);
                sub_scaled(h[j][k], &v[j], &mut w);
            }
            h[k + 1][k] = norm(&w);
            for j in 0..k {
                let t = cs[j] * h[j][k] + sn[j] * h[j + 1][k];
                h[j + 1][k] = -sn[j] * h[j][k] + cs[j] * h[j + 1][k];
                h[j][k] = t;
            }
            let rr = (h[k][k] * h[k][k] + h[k + 1][k] * h[k + 1][k]).sqrt();
            cs[k] = h[k][k] / rr;
            sn[k] = h[k + 1][k] / rr;
            h[k][k] = rr;
            let hk1 = h[k + 1][k];
            h[k + 1][k] = 0.0;
            e1[k + 1] = -sn[k] * e1[k];
            e1[k] *= cs[k];
            k_used = k + 1;
            if e1[k + 1].abs() <= 0.1 * tol * bnorm || hk1 == 0.0 || iters >= max_iter {
                break;
            }
            v.push(w.iter().map(|a| a / hk1).collect());
        }
        // solve the triangular system and update x
        let mut yv = vec![0.0; k_used];
        for i in (0..k_used).rev() {
            let mut t = e1[i];
            for j in i + 1..k_used {
                t -= h[i][j] * yv[j];
            }
            yv[i] = t / h[i][i];
        }
        let mut u = vec![0.0; dim];
        for (j, yj) in yv.iter().enumerate() {
            add_scaled(*yj, &v[j], &mut u);
        }
        let du = pre(&u);
        for t in 0..dim {
            x[t] += du[t];
        }
    }
}
