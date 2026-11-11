//! Mixing of the SCF variables between SCC iterations: modified Broyden mixing
//! (D. D. Johnson, PRB 38, 12807 (1988)) with the weights and damping used by xtb/tblite,
//! and Pulay (DIIS) mixing with restarts.

use crate::{Error, Result};
use std::collections::VecDeque;

/// Which accelerator mixes the SCC variables between SCF iterations.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MixerKind {
    /// Modified Broyden mixing with xtb's and tblite's weights, damping and full history:
    /// the iterations of those programs.
    Broyden,
    /// Pulay (DIIS) mixing of the SCC variables: the next input is the
    /// combination of the last inputs whose residuals cancel best (least squares,
    /// coefficients summing to 1) plus the damped combined residual, with a restart from
    /// the best iterate when the residual stalls or diverges.
    Diis,
    /// Broyden first; if it does not converge within `max_iter`, DIIS from the initial
    /// guess with its own `max_iter` budget (the default of the axtb conventions). Results
    /// Broyden converges are exactly Broyden's.
    Auto,
}

impl MixerKind {
    /// Lower-case name (`"broyden"`, `"diis"`, `"auto"`).
    pub fn label(self) -> &'static str {
        match self {
            MixerKind::Broyden => "broyden",
            MixerKind::Diis => "diis",
            MixerKind::Auto => "auto",
        }
    }
}

/// One SCF attempt of a single point (see [`MixerKind::Auto`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScfAttempt {
    pub mixer: MixerKind,
    pub iterations: usize,
    pub converged: bool,
}

impl std::str::FromStr for MixerKind {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "broyden" => Ok(MixerKind::Broyden),
            "diis" | "pulay" => Ok(MixerKind::Diis),
            "auto" => Ok(MixerKind::Auto),
            other => Err(Error::Input(format!("unknown SCF mixer {other:?} (auto, broyden or diis)"))),
        }
    }
}

/// Broyden mixer over a flat vector of SCF variables.
pub struct Broyden {
    ndim: usize,
    memory: usize,
    iter: usize,
    damp: f64,
    /// History slots, `memory` at most, allocated as the history fills (`memory` is the SCF's
    /// `max_iter`, which may be large).
    df: Vec<Vec<f64>>,
    u: Vec<Vec<f64>>,
    a: Vec<Vec<f64>>,
    omega: Vec<f64>,
    /// Current input vector.
    pub q_in: Vec<f64>,
    /// Current residual (output - input).
    pub dq: Vec<f64>,
    qlast: Vec<f64>,
    dqlast: Vec<f64>,
}

impl Broyden {
    pub fn new(ndim: usize, memory: usize, damp: f64) -> Self {
        let memory = memory.max(1);
        Self {
            ndim,
            memory,
            iter: 0,
            damp,
            df: Vec::new(),
            u: Vec::new(),
            a: Vec::new(),
            omega: Vec::new(),
            q_in: vec![0.0; ndim],
            dq: vec![0.0; ndim],
            qlast: vec![0.0; ndim],
            dqlast: vec![0.0; ndim],
        }
    }

    /// Root mean square of the residual.
    pub fn error(&self) -> f64 {
        let n = self.dq.len() as f64;
        self.dq.iter().map(|x| x * x / n).sum::<f64>().sqrt()
    }

    /// Produce the next input vector from `q_in` and the residual `dq`.
    pub fn next(&mut self) -> bool {
        self.iter += 1;
        let iter = self.iter;
        let n = self.ndim;
        let alpha = self.damp;
        if iter == 1 {
            self.dqlast.copy_from_slice(&self.dq);
            self.qlast.copy_from_slice(&self.q_in);
            for i in 0..n {
                self.q_in[i] += alpha * self.dq[i];
            }
            return true;
        }
        let memory = self.memory;
        let itn = iter - 1;
        let it1 = (itn - 1) % memory;
        let nhist = memory.min(itn);
        let first = if itn + 1 > memory { itn + 1 - memory } else { 1 };
        let hist: Vec<usize> = (0..nhist).map(|ih| (first + ih - 1) % memory).collect();
        const OMEGA0: f64 = 0.01;
        const MINW: f64 = 1.0;
        const MAXW: f64 = 100000.0;
        const WFAC: f64 = 0.01;
        if it1 == self.df.len() {
            // a new history slot (every entry of a slot is set below before it is read)
            self.df.push(vec![0.0; n]);
            self.u.push(vec![0.0; n]);
            self.omega.push(0.0);
            self.a.iter_mut().for_each(|row| row.push(0.0));
            self.a.push(vec![0.0; self.df.len()]);
        }

        let norm = dot(&self.dq, &self.dq).sqrt();
        self.omega[it1] = if norm > WFAC / MAXW { WFAC / norm } else { MAXW };
        if self.omega[it1] < MINW {
            self.omega[it1] = MINW;
        }
        for i in 0..n {
            self.df[it1][i] = self.dq[i] - self.dqlast[i];
        }
        let inv = 1.0 / dot(&self.df[it1], &self.df[it1]).sqrt().max(f64::EPSILON);
        for x in self.df[it1].iter_mut() {
            *x *= inv;
        }
        let mut c = vec![0.0; nhist];
        for (ih, &i) in hist.iter().enumerate() {
            c[ih] = self.omega[i] * dot(&self.df[i], &self.dq);
            for &j in &hist {
                self.a[i][j] = dot(&self.df[i], &self.df[j]);
            }
        }
        let mut beta = vec![vec![0.0; nhist]; nhist];
        for (ih, &i) in hist.iter().enumerate() {
            for (jh, &j) in hist.iter().enumerate() {
                beta[ih][jh] = self.omega[i] * self.omega[j] * self.a[i][j];
            }
            beta[ih][ih] += OMEGA0 * OMEGA0;
        }
        // beta is symmetric, so solving beta^T x = c is solving beta x = c
        if !solve_dense(&mut beta, &mut c) {
            return false;
        }
        for k in 0..n {
            self.u[it1][k] = alpha * self.df[it1][k] + inv * (self.q_in[k] - self.qlast[k]);
        }
        self.dqlast.copy_from_slice(&self.dq);
        self.qlast.copy_from_slice(&self.q_in);
        for k in 0..n {
            self.q_in[k] += alpha * self.dq[k];
        }
        for (ih, &i) in hist.iter().enumerate() {
            let f = self.omega[i] * c[ih];
            for k in 0..n {
                self.q_in[k] -= f * self.u[i][k];
            }
        }
        true
    }
}

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Solve `a x = b` in place (Gaussian elimination with partial pivoting); false if singular.
fn solve_dense(a: &mut [Vec<f64>], b: &mut [f64]) -> bool {
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

/// Number of past iterations the DIIS mixer combines.
pub const DIIS_SIZE: usize = 30;
/// A residual difference whose part not explained by the newer ones is smaller than this
/// fraction of its norm is (nearly) linearly dependent on them: it and all older ones are
/// left out, which keeps the least-squares problem well conditioned and the extrapolation
/// tame.
pub const DIIS_DEPENDENCE: f64 = 1e-2;
/// Restart when the iteration stalls: after this many iterations without a residual smaller
/// than the best one so far, the mixer restarts from the best iterate with half the damping.
pub const DIIS_STALL: usize = 12;
/// Restart when the iteration diverges: a residual norm above this multiple of the smallest
/// one so far (and above [`DIIS_RESTART_FLOOR`]) restarts the mixer the same way at once.
pub const DIIS_RESTART_GROWTH: f64 = 1e3;
/// Residual norm below which the divergence guard never acts.
pub const DIIS_RESTART_FLOOR: f64 = 0.1;

/// Pulay (DIIS) mixer over a flat vector of SCF variables ([`MixerKind::Diis`]), also known
/// as Anderson mixing.
///
/// With inputs `q_i` and residuals `r_i = F(q_i) - q_i` of the last [`DIIS_SIZE`]
/// iterations, the coefficients `c` minimize `|sum_i c_i r_i|` subject to
/// `sum_i c_i = 1`, and the next input is `sum_i c_i (q_i + damp r_i)`; the first step is
/// plain damped mixing, as in Broyden mixing. The coefficients come from the equivalent
/// unconstrained problem in the differences `r_n - r_i` to the newest residual, solved by
/// a QR factorization newest first (see [`DIIS_DEPENDENCE`]). When the residual stops
/// improving ([`DIIS_STALL`]) or grows by [`DIIS_RESTART_GROWTH`], the history is dropped
/// and the iteration restarts from the best input so far with half the damping.
pub struct Diis {
    damp: f64,
    hist_q: VecDeque<Vec<f64>>,
    hist_r: VecDeque<Vec<f64>>,
    /// Residual norm, input and residual of the best iterate so far.
    best: Option<(f64, Vec<f64>, Vec<f64>)>,
    /// Iterations since the best iterate.
    since_best: usize,
    /// Current input vector.
    pub q_in: Vec<f64>,
    /// Current residual (output - input).
    pub dq: Vec<f64>,
}

impl Diis {
    pub fn new(ndim: usize, damp: f64) -> Self {
        Self {
            damp,
            hist_q: VecDeque::new(),
            hist_r: VecDeque::new(),
            best: None,
            since_best: 0,
            q_in: vec![0.0; ndim],
            dq: vec![0.0; ndim],
        }
    }

    /// Root mean square of the residual.
    pub fn error(&self) -> f64 {
        let n = self.dq.len() as f64;
        self.dq.iter().map(|x| x * x / n).sum::<f64>().sqrt()
    }

    /// Produce the next input vector from `q_in` and the residual `dq`.
    pub fn next(&mut self) -> bool {
        let norm = dot(&self.dq, &self.dq).sqrt();
        if !norm.is_finite() {
            return false;
        }
        if let Some((best, q, r)) = &self.best {
            let diverging = norm > DIIS_RESTART_GROWTH * best && norm > DIIS_RESTART_FLOOR;
            if diverging || self.since_best >= DIIS_STALL {
                self.since_best = 0;
                self.hist_q.clear();
                self.hist_r.clear();
                self.damp *= 0.5;
                for k in 0..q.len() {
                    self.q_in[k] = q[k] + self.damp * r[k];
                }
                return true;
            }
        }
        if self.best.as_ref().is_none_or(|b| norm < b.0) {
            self.best = Some((norm, self.q_in.clone(), self.dq.clone()));
            self.since_best = 0;
        } else {
            self.since_best += 1;
        }
        if self.hist_q.len() == DIIS_SIZE {
            self.hist_q.pop_front();
            self.hist_r.pop_front();
        }
        self.hist_q.push_back(self.q_in.clone());
        self.hist_r.push_back(self.dq.clone());
        let c = diis_coefficients(&self.hist_r);
        for k in 0..self.q_in.len() {
            let mut x = 0.0;
            for (i, ci) in c.iter().enumerate() {
                if *ci != 0.0 {
                    x += ci * (self.hist_q[i][k] + self.damp * self.hist_r[i][k]);
                }
            }
            self.q_in[k] = x;
        }
        true
    }
}

/// Coefficients `c` minimizing `|sum_i c_i r_i|` with `sum_i c_i = 1` (Pulay), from the
/// equivalent problem `min_g |r_n - sum_j g_j (r_n - r_j)|` (Anderson) with
/// `c_j = g_j`, `c_n = 1 - sum_j g_j`. The differences are orthogonalized newest first
/// (modified Gram-Schmidt); a difference that is (nearly) dependent on newer ones ends the
/// basis ([`DIIS_DEPENDENCE`]) and it and the older ones get zero coefficients.
fn diis_coefficients(r: &VecDeque<Vec<f64>>) -> Vec<f64> {
    let m = r.len();
    let last = &r[m - 1];
    let n = last.len();
    // orthonormal basis of the kept differences and the upper-triangular factor (column l
    // of R in rcol[l])
    let mut basis: Vec<Vec<f64>> = Vec::new();
    let mut rcol: Vec<Vec<f64>> = Vec::new();
    let mut kept: Vec<usize> = Vec::new();
    for j in (0..m - 1).rev() {
        let mut d: Vec<f64> = (0..n).map(|k| last[k] - r[j][k]).collect();
        let norm0 = dot(&d, &d).sqrt();
        if norm0.is_nan() || norm0 <= 0.0 {
            break;
        }
        let mut col = vec![0.0; basis.len() + 1];
        for (l, b) in basis.iter().enumerate() {
            let h = dot(b, &d);
            col[l] = h;
            for k in 0..n {
                d[k] -= h * b[k];
            }
        }
        let nr = dot(&d, &d).sqrt();
        if nr < DIIS_DEPENDENCE * norm0 {
            break;
        }
        col[basis.len()] = nr;
        d.iter_mut().for_each(|x| *x /= nr);
        basis.push(d);
        rcol.push(col);
        kept.push(j);
    }
    // R g = Q^T r_n by back substitution
    let p = basis.len();
    let rhs: Vec<f64> = basis.iter().map(|b| dot(b, last)).collect();
    let mut g = vec![0.0; p];
    for l in (0..p).rev() {
        let mut v = rhs[l];
        for k in l + 1..p {
            v -= rcol[k][l] * g[k];
        }
        g[l] = v / rcol[l][l];
    }
    let mut c = vec![0.0; m];
    c[m - 1] = 1.0;
    for (l, &j) in kept.iter().enumerate() {
        c[j] += g[l];
        c[m - 1] -= g[l];
    }
    c
}

/// The SCF mixer of one SCF ([`MixerKind`]).
pub enum Mixer {
    Broyden(Broyden),
    Diis(Diis),
}

impl Mixer {
    /// A mixer for `ndim` variables; `max_iter` is Broyden's memory, `damp` the damping.
    /// `Auto` is a sequence of mixers handled by the SCF; here it means its first, Broyden.
    pub fn new(kind: MixerKind, ndim: usize, max_iter: usize, damp: f64) -> Self {
        match kind {
            MixerKind::Broyden | MixerKind::Auto => Mixer::Broyden(Broyden::new(ndim, max_iter, damp)),
            MixerKind::Diis => Mixer::Diis(Diis::new(ndim, damp)),
        }
    }

    /// Current input vector and residual (output - input).
    pub fn vectors(&mut self) -> (&mut Vec<f64>, &mut Vec<f64>) {
        match self {
            Mixer::Broyden(m) => (&mut m.q_in, &mut m.dq),
            Mixer::Diis(m) => (&mut m.q_in, &mut m.dq),
        }
    }

    /// Root mean square of the residual.
    pub fn error(&self) -> f64 {
        match self {
            Mixer::Broyden(m) => m.error(),
            Mixer::Diis(m) => m.error(),
        }
    }

    /// Produce the next input vector; false if the mixing failed.
    pub fn next(&mut self) -> bool {
        match self {
            Mixer::Broyden(m) => m.next(),
            Mixer::Diis(m) => m.next(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// On a linear fixed-point problem q = A q + b DIIS behaves like GMRES: it converges in
    /// about as many steps as A has distinct eigenvalues, also where plain damped mixing
    /// diverges (eigenvalue -4).
    #[test]
    fn diis_solves_a_linear_fixed_point_problem() {
        let eig = [0.9, 0.5, -0.5, -0.95, -4.0, 0.2];
        let b = [1.0, -2.0, 0.5, 3.0, -1.0, 0.25];
        let n = eig.len();
        let mut m = Diis::new(n, 0.4);
        let mut it = 0;
        loop {
            for k in 0..n {
                m.dq[k] = eig[k] * m.q_in[k] + b[k] - m.q_in[k];
            }
            if m.error() < 1e-12 {
                break;
            }
            assert!(m.next());
            it += 1;
            assert!(it <= 2 * n, "not converged after {it} steps (residual {})", m.error());
        }
        for k in 0..n {
            assert!((m.q_in[k] - b[k] / (1.0 - eig[k])).abs() < 1e-10);
        }
    }

    #[test]
    fn pulay_coefficients_cancel_the_residuals() {
        let r: VecDeque<Vec<f64>> = [vec![1.0, 0.0], vec![-3.0, 0.0]].into_iter().collect();
        let c = diis_coefficients(&r);
        assert!((c[0] + c[1] - 1.0).abs() < 1e-14);
        assert!((c[0] * 1.0 + c[1] * -3.0).abs() < 1e-14);
        // a residual parallel to the newest one adds nothing and gets no weight
        let r: VecDeque<Vec<f64>> = [vec![2.0, 2.0], vec![1.0, 1.0], vec![1.0, 1.0]].into_iter().collect();
        let c = diis_coefficients(&r);
        assert_eq!(c[1], 0.0);
        assert!((c.iter().sum::<f64>() - 1.0).abs() < 1e-14);
    }
}
