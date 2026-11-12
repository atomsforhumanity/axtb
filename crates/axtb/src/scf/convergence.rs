//! When an SCF stops: after every iteration a [`ConvergenceRule`] reads the [`Iteration`]
//! record and gives its [`Verdict`]. Each method's rule reproduces its programs' test:
//! GFN2-xTB's is here (xtb's and tblite's test, with xtb 6.7.1's per-atom residual and extra
//! iteration as parameters), g-xTB's two are in `axtb-gxtb`.

use super::mixing::rms;
use crate::conventions::{ConventionParams, SccResidual};

/// What a convergence rule reads of one SCF iteration.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Iteration {
    /// The iteration, counted from 1.
    pub iteration: usize,
    /// The energy of the iteration (Eh).
    pub energy: f64,
    /// Its change from the previous iteration (in the first iteration: the energy itself).
    pub energy_change: f64,
    /// The residual the SCF measures: GFN2-xTB the norm of the residual of the mixed SCC vector
    /// (RMS or per atom, by the conventions); g-xTB 2.0.1 the RMS of the Fock residual (0 in the
    /// first iteration); NaN where none is measured.
    pub residual: f64,
    /// RMS change of the density matrices (g-xTB v1.1: over the packed lower triangles of all
    /// spin channels); NaN where it is not measured.
    pub density_change: f64,
}

/// What an SCF does after an iteration.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Not converged: iterate on.
    Continue,
    /// Converged: this iteration is the result.
    Done,
    /// Converged, but a later iteration is the result: iterate on and ask the rule again
    /// (`unmixed`: diagonalize the next iteration's Fock matrix as it is built, not mixed).
    OneMore { unmixed: bool },
}

/// A convergence test of an SCF (see the module documentation).
pub trait ConvergenceRule {
    /// The verdict after the iteration `it` (called for every iteration, in order).
    fn update(&mut self, it: &Iteration) -> Verdict;
    /// Whether the SCF counts as converged if it ends after the last iteration passed to
    /// [`update`](Self::update) (at its iteration limit, or because the mixer failed): after
    /// [`Verdict::Done`], and for a rule that counts its first converged iteration (xtb 6.7.1)
    /// from then on.
    fn converged(&self) -> bool;
}

/// GFN2-xTB's convergence test (xtb, tblite; points 2 and 3 of
/// [`Conventions`](crate::Conventions)): an energy change below 1e-6·acc Eh and a residual
/// norm of the SCC vector below 2e-5·acc ([`SccResidual::Rms`]), or with
/// [`SccResidual::PerAtom`] (xtb 6.7.1) sqrt(Σ dq² / N_atoms) below 1e-4·acc when the vector
/// holds multipoles. With the extra iteration (xtb 6.7.1 `lastdiag`) the SCF counts as converged
/// from its first converged iteration and stops at the next converged one.
pub(crate) struct XtbRule {
    residual: SccResidual,
    nat: usize,
    extra_iteration: bool,
    econv: f64,
    pconv: f64,
    converged: bool,
    lastdiag: bool,
}

impl XtbRule {
    /// The rule of the conventions `params` at the accuracy `acc` for an SCC vector of `nat`
    /// atoms, with atomic multipoles or without.
    pub(crate) fn new(params: &ConventionParams, acc: f64, nat: usize, multipoles: bool) -> Self {
        let per_atom = params.scc_residual == SccResidual::PerAtom;
        let pconv = if per_atom && multipoles { 1.0e-4 * acc } else { 2.0e-5 * acc };
        Self {
            residual: params.scc_residual,
            nat,
            extra_iteration: params.scc_extra_iteration,
            econv: 1.0e-6 * acc,
            pconv,
            converged: false,
            lastdiag: false,
        }
    }

    /// The residual norm the rule tests, of the residual `r` of the mixed vector.
    pub(crate) fn residual(&self, r: &[f64]) -> f64 {
        match self.residual {
            SccResidual::Rms => rms(r),
            SccResidual::PerAtom => (r.iter().map(|x| x * x).sum::<f64>() / self.nat as f64).sqrt(),
        }
    }
}

impl ConvergenceRule for XtbRule {
    fn update(&mut self, it: &Iteration) -> Verdict {
        let ok = it.energy_change.abs() < self.econv && it.residual < self.pconv;
        if !self.extra_iteration {
            self.converged = ok;
            return if ok { Verdict::Done } else { Verdict::Continue };
        }
        // xtb 6.7.1: converged once counts as converged; the next converged iteration ends it
        self.converged |= ok;
        if ok && self.lastdiag {
            return Verdict::Done;
        }
        self.lastdiag |= ok;
        if self.converged {
            Verdict::OneMore { unmixed: false }
        } else {
            Verdict::Continue
        }
    }

    fn converged(&self) -> bool {
        self.converged
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Conventions;

    fn it(iteration: usize, energy_change: f64, residual: f64) -> Iteration {
        Iteration { iteration, energy: 0.0, energy_change, residual, density_change: f64::NAN }
    }

    #[test]
    fn xtb_rule_stops_at_the_first_converged_iteration() {
        let mut r = XtbRule::new(&Conventions::Tblite.params(), 1.0, 3, true);
        assert_eq!(r.update(&it(1, 1e-3, 1e-6)), Verdict::Continue);
        assert!(!r.converged());
        assert_eq!(r.update(&it(2, 1e-7, 1e-6)), Verdict::Done);
        assert!(r.converged());
        // RMS over the vector
        assert_eq!(r.residual(&[0.5; 4]), 0.5);
    }

    /// xtb 6.7.1: the iteration after the first converged one must converge too; in between
    /// the SCF counts as converged (its iteration limit then ends it converged).
    #[test]
    fn xtb671_rule_needs_a_second_converged_iteration() {
        let mut r = XtbRule::new(&Conventions::Xtb671.params(), 1.0, 4, true);
        assert_eq!(r.update(&it(1, 1e-7, 1e-5)), Verdict::OneMore { unmixed: false });
        assert!(r.converged());
        assert_eq!(r.update(&it(2, 1e-3, 1e-5)), Verdict::OneMore { unmixed: false });
        assert_eq!(r.update(&it(3, 1e-7, 1e-5)), Verdict::Done);
        // per atom: sqrt(Σ r² / N_atoms) = sqrt(8 / 4)
        assert_eq!(r.residual(&[1.0; 8]), 2f64.sqrt());
    }
}
