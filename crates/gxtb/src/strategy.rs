//! The SCF strategy of the g-xTB 1.1.0 program (v1.1): its Fock mixing (`diisfock_`),
//! [`Gxtb110FockMixer`], and its convergence test with the final unmixed iteration,
//! [`Gxtb110Rule`], for the core's SCF traits (`axtb::scf`). [`crate::scf::Setup::scf_from`]
//! runs them.

use axtb::scf::{ConvergenceRule, FockMixer, Iteration, Verdict};
use axtb::Result;
use faer::{Mat, Par};

/// The 1.1.0 program's Fock mixing (`diisfock_`), all matrices per spin as full row-major
/// vectors here (closed-shell iteration numbers; open shell one later, see
/// [`crate::ScfOptions::transition_iteration`]): iterations 1 and 2 damp,
/// Fs = Fin + alpha (F - Fin) (Fin: the previous mixed Fock matrix, zero at the start);
/// iteration 3 stores its pair (F, F - Fin) and returns 2 Fin + alpha (F - Fin); from
/// iteration 4 on the stored pairs (at most `size`, oldest dropped) are combined,
/// Fs = sum_i c_i F_i with the coefficients minimizing |sum_i c_i r_i|^2, r_i = F_i - Fin_i,
/// sum_i c_i = 1 (dot products over the packed lower triangle, separately for each spin; the
/// bordered equations solved by faer's LU). Only the direction of Fs matters (it is
/// diagonalized). An SCF restarted from a given density (`restart`, the program's
/// `gxtbrestart`) has no transition step: iteration 3 (4) already combines the stored pairs, its
/// own only, so Fs = F (found by comparing every iteration of the program's restarted runs).
///
/// It measures no residual (the 1.1.0 SCF converges on the density change): `mix` returns NaN.
pub struct Gxtb110FockMixer {
    /// AOs per channel.
    n: usize,
    alpha: f64,
    ndamp: usize,
    restart: bool,
    size: usize,
    /// The mixed matrices of the last iteration (Fin of the next; empty before the first).
    fin: Vec<Vec<f64>>,
    hist: Vec<(Vec<Vec<f64>>, Vec<f64>)>,
}

impl Gxtb110FockMixer {
    /// The mixer for `n` AOs per channel with the damping `alpha`, the transition step in
    /// iteration `ndamp` (closed shell 3, open shell 4), at most `size` stored pairs (closed
    /// shell 7, open shell 8), for an SCF restarted from a given density or not.
    pub fn new(n: usize, alpha: f64, ndamp: usize, size: usize, restart: bool) -> Self {
        Self { n, alpha, ndamp, restart, size, fin: Vec::new(), hist: Vec::new() }
    }
}

impl FockMixer for Gxtb110FockMixer {
    fn mix(&mut self, iter: usize, f: Vec<Vec<f64>>) -> Result<f64> {
        let n = self.n;
        let ns = f.len();
        let fin = match std::mem::take(&mut self.fin) {
            fin if fin.is_empty() => vec![vec![0.0; n * n]; ns],
            fin => fin,
        };
        let out: Vec<Vec<f64>> = if iter < self.ndamp {
            (0..ns).map(|s| (0..n * n).map(|k| fin[s][k] + self.alpha * (f[s][k] - fin[s][k])).collect()).collect()
        } else {
            // residual over the packed lower triangles of all spins
            let mut r = Vec::with_capacity(ns * n * (n + 1) / 2);
            for s in 0..ns {
                for i in 0..n {
                    for j in 0..=i {
                        r.push(f[s][i * n + j] - fin[s][i * n + j]);
                    }
                }
            }
            let transition = iter == self.ndamp && !self.restart;
            let out = transition.then(|| {
                (0..ns)
                    .map(|s| (0..n * n).map(|k| 2.0 * fin[s][k] + self.alpha * (f[s][k] - fin[s][k])).collect())
                    .collect()
            });
            self.hist.push((f, r));
            if self.hist.len() > self.size {
                self.hist.remove(0);
            }
            out.unwrap_or_else(|| {
                (0..ns)
                    .map(|s| {
                        let c = diis_coefficients(&self.hist, s, n * (n + 1) / 2);
                        let mut v = vec![0.0; n * n];
                        for (ci, (fi, _)) in c.iter().zip(&self.hist) {
                            for k in 0..n * n {
                                v[k] += ci * fi[s][k];
                            }
                        }
                        v
                    })
                    .collect()
            })
        };
        self.fin = out;
        Ok(f64::NAN)
    }

    fn mixed(&self, s: usize) -> &[f64] {
        &self.fin[s]
    }
}

/// Pulay coefficients of spin `s` from the residuals of `hist` (`npk` packed elements per spin).
fn diis_coefficients(hist: &[(Vec<Vec<f64>>, Vec<f64>)], s: usize, npk: usize) -> Vec<f64> {
    let m = hist.len();
    let mut b = Mat::<f64>::zeros(m + 1, m + 1);
    let r = |i: usize| &hist[i].1[s * npk..(s + 1) * npk];
    for i in 0..m {
        for j in 0..=i {
            let v: f64 = r(i).iter().zip(r(j)).map(|(a, b)| a * b).sum();
            b[(i, j)] = v;
            b[(j, i)] = v;
        }
        b[(i, m)] = -1.0;
        b[(m, i)] = -1.0;
    }
    let mut rhs = Mat::<f64>::zeros(m + 1, 1);
    rhs[(m, 0)] = -1.0;
    // sequential: faer's high-level `partial_piv_lu` would start rayon's global pool
    axtb_base::linalg::Lu::new(b, Par::Seq).solve(rhs.as_mut());
    if (0..m).any(|i| !rhs[(i, 0)].is_finite()) {
        let mut c = vec![0.0; m];
        c[m - 1] = 1.0;
        return c;
    }
    (0..m).map(|i| rhs[(i, 0)]).collect()
}

/// The 1.1.0 program's convergence test: converged when the energy changed by less than
/// `ethr` and the RMS change of the packed density matrices is below `pthr` in the same
/// iteration. The program then diagonalizes the unmixed Fock matrix of the converged density
/// once more ([`Verdict::OneMore`]), and that iteration is the result; an SCF whose iteration
/// limit comes first is not converged.
pub struct Gxtb110Rule {
    ethr: f64,
    pthr: f64,
    /// The convergence test was met; the next iteration is the last.
    met: bool,
    /// The last, unmixed iteration ran.
    done: bool,
}

impl Gxtb110Rule {
    /// The test with the energy threshold `ethr` (Eh) and the density threshold `pthr`.
    pub fn new(ethr: f64, pthr: f64) -> Self {
        Self { ethr, pthr, met: false, done: false }
    }
}

impl ConvergenceRule for Gxtb110Rule {
    fn update(&mut self, it: &Iteration) -> Verdict {
        if self.met {
            self.done = true;
            return Verdict::Done;
        }
        if it.energy_change.abs() < self.ethr && it.density_change < self.pthr {
            self.met = true;
            return Verdict::OneMore { unmixed: true };
        }
        Verdict::Continue
    }

    fn converged(&self) -> bool {
        self.done
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn it(iteration: usize, energy_change: f64, density_change: f64) -> Iteration {
        Iteration { iteration, energy: 0.0, energy_change, residual: f64::NAN, density_change }
    }

    /// Converged, then one unmixed iteration whatever its changes; converged only after it.
    #[test]
    fn one_unmixed_iteration_after_convergence() {
        let mut r = Gxtb110Rule::new(1e-8, 2e-5);
        assert_eq!(r.update(&it(1, 1e-3, 1e-3)), Verdict::Continue);
        assert_eq!(r.update(&it(2, 1e-9, 1e-6)), Verdict::OneMore { unmixed: true });
        assert!(!r.converged());
        assert_eq!(r.update(&it(3, 1.0, 1.0)), Verdict::Done);
        assert!(r.converged());
    }

    /// Damping from a zero Fin, then the transition step 2 Fin + alpha (F - Fin), then Pulay
    /// over the stored pairs; a restarted SCF skips the transition step (one stored pair: F).
    #[test]
    fn damping_transition_and_pulay() {
        let mut m = Gxtb110FockMixer::new(1, 0.5, 3, 7, false);
        assert!(m.mix(1, vec![vec![2.0]]).unwrap().is_nan());
        assert_eq!(m.mixed(0), [1.0]);
        m.mix(2, vec![vec![2.0]]).unwrap();
        assert_eq!(m.mixed(0), [1.5]);
        m.mix(3, vec![vec![2.0]]).unwrap();
        assert_eq!(m.mixed(0), [2.0 * 1.5 + 0.5 * 0.5]);
        let mut m = Gxtb110FockMixer::new(1, 0.5, 3, 7, true);
        m.mix(1, vec![vec![2.0]]).unwrap();
        m.mix(2, vec![vec![2.0]]).unwrap();
        m.mix(3, vec![vec![4.0]]).unwrap();
        assert!((m.mixed(0)[0] - 4.0).abs() < 1e-14);
    }
}
