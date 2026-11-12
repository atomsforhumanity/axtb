//! The SCF strategy of the 2.0.1 program (tblite's SCF as the program runs it, spec §4.17): its
//! Fock mixing, [`Gxtb201FockDiis`], and its convergence test, [`Gxtb201Rule`], for the core's
//! SCF traits (`axtb::scf`). [`super::scf::Setup::scf_from`] runs them.

use axtb::scf::{ConvergenceRule, FockMixer, Iteration, Verdict};
use axtb::Result;

/// Energy change (Eh) after which the program's SCF switches from damping to DIIS before
/// iteration 4 (the threshold of tblite's DIIS mixer stage; independent of the accuracy).
const DIIS_SWITCH_ENERGY: f64 = 1e-12;

/// The 2.0.1 program's Fock mixing: tblite's damping and DIIS stages on the full Fock vector
/// (the matrices of all spin channels one after the other).
///
/// - Iteration 1 is not mixed: F̃₁ = F₁.
/// - Damping, F̃_k = F̃_{k−1} + damping · r_k with the residual r_k = F_k − F̃_{k−1}, until
///   DIIS starts: after iteration 3, or after an iteration whose energy changed by less than
///   1e-12 Eh ([`FockMixer::observe`]; tblite switches mixer stages at the stage's start
///   iteration or energy threshold).
/// - From iteration 3 on the pairs (F_k, r_k) are stored, at most `size` (the oldest dropped),
///   with the DIIS matrix B_ij = r_i · r_j.
/// - DIIS: F̃_k = Σ_i c_i F_i with the coefficients of the bordered B matrix
///   ([`diis_coefficients`], LAPACK `getf2` order); F̃_k = F_k when those equations are
///   singular.
///
/// The residual of iteration k is the RMS of r_k over the full vector (0 in iteration 1). The
/// element loops and dot products run in the fixed chunks of [`crate::parallel`] (the same bits
/// for every thread count).
pub(crate) struct Gxtb201FockDiis {
    /// AOs per channel.
    n: usize,
    damping: f64,
    size: usize,
    parallel: bool,
    /// DIIS has replaced the damping.
    diis: bool,
    /// The mixed Fock vector of the last iteration.
    fmix: Vec<f64>,
    /// The stored (F_k, r_k), oldest first.
    hist: Vec<(Vec<f64>, Vec<f64>)>,
    /// B_ij = r_i · r_j of the stored residuals.
    bmat: Vec<Vec<f64>>,
}

impl Gxtb201FockDiis {
    /// The mixer for `n` AOs per channel with the damping factor `damping` (the program's 0.2)
    /// and at most `size` stored iterations (the program's 7); `parallel`: chunks on the
    /// current rayon pool.
    pub(crate) fn new(n: usize, damping: f64, size: usize, parallel: bool) -> Self {
        Self { n, damping, size, parallel, diis: false, fmix: Vec::new(), hist: Vec::new(), bmat: Vec::new() }
    }
}

impl FockMixer for Gxtb201FockDiis {
    fn mix(&mut self, iter: usize, mut fock: Vec<Vec<f64>>) -> Result<f64> {
        let par = self.parallel;
        let dot = |a: &[f64], b: &[f64]| crate::parallel::sum(a.len(), par, |k| a[k] * b[k]);
        let fvec: Vec<f64> = if fock.len() == 1 { std::mem::take(&mut fock[0]) } else { fock.concat() };
        let len = fvec.len();
        if iter == 1 {
            self.fmix = fvec;
            return Ok(0.0);
        }
        let mut r = vec![0.0; len];
        let fmix = &self.fmix;
        crate::parallel::for_each_mut(&mut r, par, |k, x| *x = fvec[k] - fmix[k]);
        let err = (dot(&r, &r) / len as f64).sqrt();
        if !self.diis {
            let d = self.damping;
            crate::parallel::for_each_mut(&mut self.fmix, par, |k, x| *x += d * r[k]);
        }
        let mut fvec = Some(fvec);
        let (hist, bmat) = (&mut self.hist, &mut self.bmat);
        if iter >= 3 {
            let row: Vec<f64> = hist.iter().map(|(_, rj)| dot(&r, rj)).chain(std::iter::once(dot(&r, &r))).collect();
            for (bj, &v) in bmat.iter_mut().zip(&row) {
                bj.push(v);
            }
            bmat.push(row);
            hist.push((fvec.take().unwrap(), r));
            if hist.len() > self.size {
                hist.remove(0);
                bmat.remove(0);
                bmat.iter_mut().for_each(|b| {
                    b.remove(0);
                });
            }
        }
        if self.diis {
            match diis_coefficients(bmat) {
                Some(cf) => {
                    let mut v = vec![0.0; len];
                    crate::parallel::for_each_mut(&mut v, par, |k, x| {
                        for (ci, (fi, _)) in cf.iter().zip(hist.iter()) {
                            *x += ci * fi[k];
                        }
                    });
                    self.fmix = v;
                }
                None => self.fmix = fvec.unwrap_or_else(|| hist.last().unwrap().0.clone()),
            }
        }
        Ok(err)
    }

    fn mixed(&self, s: usize) -> &[f64] {
        let nn = self.n * self.n;
        &self.fmix[s * nn..(s + 1) * nn]
    }

    fn observe(&mut self, it: &Iteration) {
        self.diis |= it.iteration >= 3 || it.energy_change.abs() < DIIS_SWITCH_ENERGY;
    }
}

/// Solve the bordered DIIS equations of the matrix `b` (B_ij = r_i · r_j) with an unblocked LU
/// with partial pivoting (LAPACK `getf2` order: the first maximum on ties) and return the
/// coefficients; `None` if the equations are singular.
pub(crate) fn diis_coefficients(b: &[Vec<f64>]) -> Option<Vec<f64>> {
    let m = b.len();
    let n = m + 1;
    let mut a = vec![0.0; n * n];
    for i in 0..m {
        for j in 0..m {
            a[i * n + j] = b[i][j];
        }
        a[i * n + m] = 1.0;
        a[m * n + i] = 1.0;
    }
    let mut rhs = vec![0.0; n];
    rhs[m] = 1.0;
    let mut piv = vec![0usize; n];
    for k in 0..n {
        let mut p = k;
        let mut amax = a[k * n + k].abs();
        for i in k + 1..n {
            if a[i * n + k].abs() > amax {
                amax = a[i * n + k].abs();
                p = i;
            }
        }
        piv[k] = p;
        if a[p * n + k] == 0.0 {
            return None;
        }
        if p != k {
            for j in 0..n {
                a.swap(k * n + j, p * n + j);
            }
        }
        let inv = 1.0 / a[k * n + k];
        for i in k + 1..n {
            a[i * n + k] *= inv;
        }
        for i in k + 1..n {
            let l = a[i * n + k];
            if l != 0.0 {
                for j in k + 1..n {
                    a[i * n + j] -= l * a[k * n + j];
                }
            }
        }
    }
    for k in 0..n {
        if piv[k] != k {
            rhs.swap(k, piv[k]);
        }
    }
    for i in 0..n {
        let mut s = rhs[i];
        for j in 0..i {
            s -= a[i * n + j] * rhs[j];
        }
        rhs[i] = s;
    }
    for i in (0..n).rev() {
        let mut s = rhs[i];
        for j in i + 1..n {
            s -= a[i * n + j] * rhs[j];
        }
        rhs[i] = s / a[i * n + i];
    }
    if rhs.iter().any(|x| !x.is_finite()) {
        return None;
    }
    Some(rhs[..m].to_vec())
}

/// The 2.0.1 program's convergence test (tblite's `check_convergence`): converged after
/// iteration k once |E_j − E_{j−1}| < `ethr` in some iteration j ≤ k and the Fock residual is
/// below `pthr` in some iteration 1 < j ≤ k. Each criterion counts once met (tblite only ever
/// sets its flags); no extra iteration.
pub(crate) struct Gxtb201Rule {
    ethr: f64,
    pthr: f64,
    econverged: bool,
    pconverged: bool,
}

impl Gxtb201Rule {
    /// The test with the energy threshold `ethr` (Eh; the program's 1e-6·acc) and the residual
    /// threshold `pthr` (the program's 2e-5·acc).
    pub(crate) fn new(ethr: f64, pthr: f64) -> Self {
        Self { ethr, pthr, econverged: false, pconverged: false }
    }
}

impl ConvergenceRule for Gxtb201Rule {
    fn update(&mut self, it: &Iteration) -> Verdict {
        self.econverged |= it.energy_change.abs() < self.ethr;
        self.pconverged |= it.iteration > 1 && it.residual < self.pthr;
        if self.converged() {
            Verdict::Done
        } else {
            Verdict::Continue
        }
    }

    fn converged(&self) -> bool {
        self.econverged && self.pconverged
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn it(iteration: usize, energy_change: f64, residual: f64) -> Iteration {
        Iteration { iteration, energy: 0.0, energy_change, residual, density_change: f64::NAN }
    }

    /// Each criterion counts once met, in any iteration; the residual of iteration 1 does not.
    #[test]
    fn the_flags_are_sticky() {
        let mut r = Gxtb201Rule::new(1e-6, 2e-5);
        assert_eq!(r.update(&it(1, 1e-7, 0.0)), Verdict::Continue);
        assert_eq!(r.update(&it(2, 1e-3, 1e-6)), Verdict::Done);
        assert!(r.converged());
    }

    /// Damping in iteration 2, DIIS (here two stored vectors) once the energy change was tiny.
    #[test]
    fn damping_then_diis() {
        let mut m = Gxtb201FockDiis::new(1, 0.2, 7, false);
        assert_eq!(m.mix(1, vec![vec![1.0]]).unwrap(), 0.0);
        assert_eq!(m.mixed(0), [1.0]);
        // F~2 = 1 + 0.2 (2 - 1)
        assert_eq!(m.mix(2, vec![vec![2.0]]).unwrap(), 1.0);
        assert_eq!(m.mixed(0), [1.2]);
        m.observe(&it(2, 1e-13, 1.0));
        // DIIS with one stored pair: F~3 = F3
        m.mix(3, vec![vec![3.0]]).unwrap();
        assert!((m.mixed(0)[0] - 3.0).abs() < 1e-14);
    }

    /// Pulay's coefficients are proportional to 1/B_ii for orthogonal residuals; equal (zero)
    /// residuals give singular equations.
    #[test]
    fn getf2_coefficients() {
        let c = diis_coefficients(&[vec![1.0, 0.0], vec![0.0, 4.0]]).unwrap();
        assert!((c[0] - 0.8).abs() < 1e-14 && (c[1] - 0.2).abs() < 1e-14, "{c:?}");
        assert!(diis_coefficients(&[vec![0.0, 0.0], vec![0.0, 0.0]]).is_none());
    }
}
