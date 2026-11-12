//! Mixing between SCF iterations. The accelerators come in two kinds, by what they mix:
//!
//! - [`VectorMixer`]: the input vector of the self-consistent variables, before the Fock matrix
//!   is built from it (GFN2-xTB: shell charges, atomic dipoles and quadrupoles, packed by
//!   [`ScfVariables`](super::ScfVariables)). Its energy depends on the density only through
//!   `Tr(H0 P)` and these moments, so mixing them is enough.
//! - [`FockMixer`]: the Fock matrices of all spin channels, after they are built (g-xTB, whose
//!   exchange depends on the whole density). g-xTB's Fock mixers reproduce its programs and are
//!   in `axtb-gxtb`.
//!
//! The vector mixers here are modified Broyden mixing (D. D. Johnson, PRB 38, 12807 (1988)) with
//! the weights and damping used by xtb/tblite, and Pulay (DIIS) mixing with restarts; GFN2-xTB
//! runs them as the attempts of a plan of mixers ([`MixerKind`]).

use super::convergence::Iteration;
use crate::linalg::gauss_last_max;
use crate::{Error, Result};
use std::collections::VecDeque;

/// A mixer of the input vector of an SCF's self-consistent variables (see the module
/// documentation). In each iteration the SCF builds the Fock matrix of
/// [`input`](Self::input), finds the output vector of the new density, hands it to
/// [`set_residual`](Self::set_residual) and calls [`next`](Self::next) for the input of the
/// next iteration. Implemented by GFN2-xTB's Broyden and DIIS mixers.
pub trait VectorMixer {
    /// The current input vector.
    fn input(&self) -> &[f64];
    /// The current input vector, to write (the start before the first iteration).
    fn input_mut(&mut self) -> &mut [f64];
    /// The residual (output − input) of the current input.
    fn residual(&self) -> &[f64];
    /// Set the residual to `out − input` for the output vector `out`.
    fn set_residual(&mut self, out: &[f64]);
    /// Root mean square of the residual.
    fn rms(&self) -> f64 {
        rms(self.residual())
    }
    /// Replace the input by the next one, from the inputs and residuals so far; `false` if the
    /// mixing failed (the SCF cannot go on with this mixer).
    fn next(&mut self) -> bool;
}

/// A mixer of the Fock matrices of an SCF (see the module documentation). In each iteration the
/// SCF hands over the Fock matrices it built, one per spin channel, and diagonalizes the mixed
/// ones. Implemented by g-xTB's program mixers (`axtb-gxtb`).
///
/// The mixer takes the built matrices over (it keeps some of them in its history) and holds
/// the mixed ones, which the SCF reads through [`mixed`](Self::mixed): no matrix is copied for
/// the handover.
pub trait FockMixer {
    /// Mix the Fock matrices `fock` built in SCF iteration `iter` (counted from 1; one row-major
    /// n x n matrix per spin channel) with those of the earlier iterations. Returns the
    /// residual the mixer measures: the RMS of the change of the Fock matrices against the last
    /// mixed ones (0 before anything was mixed), or NaN for a mixer that measures none.
    fn mix(&mut self, iter: usize, fock: Vec<Vec<f64>>) -> Result<f64>;
    /// The mixed Fock matrix of spin channel `s` (row-major n x n) of the last
    /// [`mix`](Self::mix), to diagonalize.
    fn mixed(&self, s: usize) -> &[f64];
    /// The record of an iteration once its energy is known, for mixers whose schedule depends
    /// on it (by default ignored).
    fn observe(&mut self, _it: &Iteration) {}
}

/// Root mean square of `r`, sqrt(Σ r_k² / n) (each square divided by n, then summed).
pub(crate) fn rms(r: &[f64]) -> f64 {
    let n = r.len() as f64;
    r.iter().map(|x| x * x / n).sum::<f64>().sqrt()
}

/// `dq = out − q_in` (the residual of a vector mixer).
fn set_residual(out: &[f64], q_in: &[f64], dq: &mut [f64]) {
    for k in 0..dq.len() {
        dq[k] = out[k] - q_in[k];
    }
}

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

impl MixerKind {
    /// A fresh mixer of this kind for `ndim` variables with damping `damp` (Broyden keeps a
    /// history of `memory` iterations, the SCF's `max_iter`). `Auto` is a plan of mixers
    /// ([`MixerPlan`]); here it means its first, Broyden.
    pub(crate) fn mixer(self, ndim: usize, memory: usize, damp: f64) -> Box<dyn VectorMixer> {
        match self {
            MixerKind::Broyden | MixerKind::Auto => Box::new(Broyden::new(ndim, memory, damp)),
            MixerKind::Diis => Box::new(Diis::new(ndim, damp)),
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

/// GFN2-xTB's SCF attempts for a [`MixerKind`]: one attempt per mixer, in order, each from the
/// start guess with its own `max_iter` iterations, until one converges ([`MixerKind::Auto`]:
/// Broyden, then DIIS). The plan records the attempts and names the mixer of the result.
pub(crate) struct MixerPlan {
    kind: MixerKind,
    attempts: Vec<ScfAttempt>,
}

impl MixerPlan {
    pub(crate) fn new(kind: MixerKind) -> Self {
        let mut plan = Self { kind, attempts: Vec::new() };
        plan.attempts.reserve_exact(plan.mixers().len());
        plan
    }

    /// The mixers of the attempts, in order.
    pub(crate) fn mixers(&self) -> &'static [MixerKind] {
        match self.kind {
            MixerKind::Auto => &[MixerKind::Broyden, MixerKind::Diis],
            MixerKind::Broyden => &[MixerKind::Broyden],
            MixerKind::Diis => &[MixerKind::Diis],
        }
    }

    /// Record an attempt that ended.
    pub(crate) fn record(&mut self, attempt: ScfAttempt) {
        self.attempts.push(attempt);
    }

    /// SCF iterations of all attempts together.
    pub(crate) fn iterations(&self) -> usize {
        self.attempts.iter().map(|a| a.iterations).sum()
    }

    /// The mixer that produced the result: `"broyden"`, `"diis"`, or `"diis-fallback"` (`auto`
    /// after Broyden did not converge).
    pub(crate) fn used(&self) -> &'static str {
        match (self.kind, self.attempts.len()) {
            (MixerKind::Auto, n) if n > 1 => "diis-fallback",
            _ => self.attempts.last().map_or("broyden", |a| a.mixer.label()),
        }
    }

    /// The attempts, in order.
    pub(crate) fn into_attempts(self) -> Vec<ScfAttempt> {
        self.attempts
    }
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
    q_in: Vec<f64>,
    /// Current residual (output - input).
    dq: Vec<f64>,
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
}

impl VectorMixer for Broyden {
    fn input(&self) -> &[f64] {
        &self.q_in
    }

    fn input_mut(&mut self) -> &mut [f64] {
        &mut self.q_in
    }

    fn residual(&self) -> &[f64] {
        &self.dq
    }

    fn set_residual(&mut self, out: &[f64]) {
        set_residual(out, &self.q_in, &mut self.dq);
    }

    /// The next input vector from `q_in` and the residual `dq`.
    fn next(&mut self) -> bool {
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
        if !gauss_last_max(&mut beta, &mut c) {
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
    q_in: Vec<f64>,
    /// Current residual (output - input).
    dq: Vec<f64>,
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
}

impl VectorMixer for Diis {
    fn input(&self) -> &[f64] {
        &self.q_in
    }

    fn input_mut(&mut self) -> &mut [f64] {
        &mut self.q_in
    }

    fn residual(&self) -> &[f64] {
        &self.dq
    }

    fn set_residual(&mut self, out: &[f64]) {
        set_residual(out, &self.q_in, &mut self.dq);
    }

    /// The next input vector from `q_in` and the residual `dq`.
    fn next(&mut self) -> bool {
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
        let mut out = vec![0.0; n];
        loop {
            for k in 0..n {
                out[k] = eig[k] * m.input()[k] + b[k];
            }
            m.set_residual(&out);
            if m.rms() < 1e-12 {
                break;
            }
            assert!(m.next());
            it += 1;
            assert!(it <= 2 * n, "not converged after {it} steps (residual {})", m.rms());
        }
        for k in 0..n {
            assert!((m.input()[k] - b[k] / (1.0 - eig[k])).abs() < 1e-10);
        }
    }

    #[test]
    fn the_plan_names_the_mixer_of_the_result() {
        let attempt = |mixer, iterations, converged| ScfAttempt { mixer, iterations, converged };
        let mut plan = MixerPlan::new(MixerKind::Auto);
        assert_eq!(plan.mixers(), [MixerKind::Broyden, MixerKind::Diis]);
        plan.record(attempt(MixerKind::Broyden, 250, false));
        plan.record(attempt(MixerKind::Diis, 60, true));
        assert_eq!((plan.used(), plan.iterations()), ("diis-fallback", 310));
        let mut plan = MixerPlan::new(MixerKind::Auto);
        plan.record(attempt(MixerKind::Broyden, 12, true));
        assert_eq!(plan.used(), "broyden");
        let mut plan = MixerPlan::new(MixerKind::Diis);
        assert_eq!(plan.mixers(), [MixerKind::Diis]);
        plan.record(attempt(MixerKind::Diis, 12, true));
        assert_eq!(plan.used(), "diis");
        assert_eq!(plan.into_attempts().len(), 1);
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
