//! The SCF loop, once for every self-consistent tight-binding method: [`run_scf`] iterates a
//! method's [`ScfModel`] with an [`ScfStrategy`] (how the iterations are mixed and when they
//! stop) from a given start state, and returns the [`ScfOutcome`].
//!
//! An iteration diagonalizes Fock matrices, occupies the orbitals, builds the density, replaces
//! the state by the state of that density (the Mulliken analysis) and evaluates its energy; the
//! [`ConvergenceRule`] reads the [`Iteration`] record. Where the Fock matrices come from is the
//! accelerator's choice ([`Accelerator`]):
//!
//! - [`Accelerator::Moments`]: a [`VectorMixer`] mixes the state as a vector (GFN2-xTB's SCC
//!   moments, laid out by [`ScfModel::pack`]); the Fock matrices of iteration k are built from
//!   the mixed input ([`ScfModel::build_fock`]) and diagonalized as built, and the mixer's
//!   residual is the output state minus that input.
//! - [`Accelerator::Fock`]: a [`FockMixer`] mixes Fock matrices (g-xTB, whose exchange depends
//!   on the whole density); the matrices of iteration k are those [`ScfModel::evaluate`] built
//!   with the energy of iteration k − 1 (iteration 1: of the start), mixed, then diagonalized.
//!
//! Each method keeps its arithmetic in its model (its Fock build, eigensolver calls,
//! occupations, density builder, Mulliken analysis and energy) and its programs' conventions in
//! its strategy types; the loop has no method-specific branch. A rule's
//! [`Verdict::OneMore`] runs one more iteration whose result counts, with `unmixed` its Fock
//! matrices as built ([`FockSource::Unmixed`]; the g-xTB 1.1.0 program's last iteration). Retries
//! with another start or mixer are separate SCFs, one [`run_scf`] each (GFN2-xTB's attempts).
//!
//! The model is a type parameter (its calls are static); the mixer and the rule are trait
//! objects, a few dynamic calls per iteration.

use super::convergence::{ConvergenceRule, Iteration, Verdict};
use super::mixing::{FockMixer, VectorMixer};
use crate::{Error, Result};
use axtb_base::timing::Laps;
use std::ops::ControlFlow;

/// The Fock matrices an SCF iteration diagonalizes ([`ScfModel::diagonalize`]).
#[derive(Clone, Copy)]
pub enum FockSource<'a> {
    /// Those the model built from the iteration's input state ([`ScfModel::build_fock`]; with
    /// an [`Accelerator::Moments`] mixer). The model holds them.
    Built,
    /// The mixed matrices of a Fock mixer ([`FockMixer::mixed`]).
    Mixed(&'a dyn FockMixer),
    /// The matrices built with the previous iteration's energy, not mixed (after
    /// [`Verdict::OneMore`] with `unmixed`): one row-major n x n matrix per spin channel.
    Unmixed(&'a [Vec<f64>]),
}

impl<'a> FockSource<'a> {
    /// The row-major n x n matrix of spin channel `s`; `None` for [`FockSource::Built`].
    pub fn matrix(&self, s: usize) -> Option<&'a [f64]> {
        match *self {
            FockSource::Built => None,
            FockSource::Mixed(m) => Some(m.mixed(s)),
            FockSource::Unmixed(f) => Some(&f[s]),
        }
    }
}

/// What mixes the iterations of an SCF (see the module documentation).
pub enum Accelerator<'a> {
    /// The input state as a vector, before the Fock matrices are built from it.
    Moments(&'a mut dyn VectorMixer),
    /// The Fock matrices, after they are built.
    Fock(&'a mut dyn FockMixer),
}

/// How an SCF iterates: its accelerator, its convergence rule and its iteration limit. The
/// mixer and the rule are borrowed: after [`run_scf`] the caller reads them (e.g. GFN2-xTB's
/// gradient moments from the mixer's last input).
pub struct ScfStrategy<'a> {
    pub accelerator: Accelerator<'a>,
    pub rule: &'a mut dyn ConvergenceRule,
    /// Iterations at most (at least 1).
    pub max_iter: usize,
}

/// Why an SCF ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stop {
    /// The rule's [`Verdict::Done`].
    Converged,
    /// `max_iter` iterations without [`Verdict::Done`].
    IterationLimit,
    /// The vector mixer could not give the next input ([`VectorMixer::next`]).
    MixerFailed,
    /// The model broke off in [`ScfModel::diagonalize`] (it keeps why).
    Model,
}

/// The end of an SCF.
#[derive(Clone, Debug)]
pub struct ScfOutcome<S> {
    /// The state of the last iteration (its output; after [`Stop::MixerFailed`] the last
    /// complete iteration's, after [`Stop::Model`] the input of the iteration that broke off).
    pub state: S,
    /// Iterations run (the one that broke off included, a failed mixer's next one not).
    pub iterations: usize,
    /// Whether the SCF counts as converged: [`ConvergenceRule::converged`] at its end.
    pub converged: bool,
    pub stop: Stop,
    /// The record of the last complete iteration (before the first: energy, energy change and
    /// residual 0, density change NaN).
    pub last: Iteration,
}

/// One SCF of a method (see the module documentation): its state, how it builds and
/// diagonalizes Fock matrices, its density, Mulliken analysis and energy. The model holds the
/// work arrays of the SCF (eigensolver, orbitals, occupations, density), which its caller reads
/// after [`run_scf`] to package the result.
pub trait ScfModel {
    /// The self-consistent state an iteration ends with (GFN2-xTB: the SCC moments; g-xTB: the
    /// density matrices, with their moments).
    type State;

    /// Length of the state as a vector for an [`Accelerator::Moments`] mixer; 0 (the default):
    /// the model is not mixed as a vector.
    fn vector_len(&self) -> usize {
        0
    }

    /// Write `state` as a vector into `v` ([`vector_len`](Self::vector_len) long).
    fn pack(&self, _state: &Self::State, _v: &mut [f64]) {}

    /// Set `state` from the vector `v`.
    fn unpack(&self, _v: &[f64], _state: &mut Self::State) {}

    /// Build the Fock matrices of iteration `iter` from its input `state`, to diagonalize them
    /// as built ([`FockSource::Built`]; a model mixed as a vector). A model mixed by its Fock
    /// matrices builds them in [`evaluate`](Self::evaluate); the default refuses.
    fn build_fock(&mut self, _iter: usize, _state: &Self::State, _laps: &mut Laps) -> Result<()> {
        Err(Error::Input("this SCF model builds its Fock matrices with its energy: mix them with a Fock mixer".into()))
    }

    /// Diagonalize the Fock matrices `fock` and occupy the orbitals.
    /// [`ControlFlow::Break`]: the SCF cannot go on (the model keeps why, e.g. GFN2-xTB's lost
    /// Fermi level).
    fn diagonalize(&mut self, fock: FockSource<'_>) -> Result<ControlFlow<()>>;

    /// Build the density of the occupied orbitals.
    fn density(&mut self);

    /// Replace `state` by the state of the new density (the Mulliken analysis).
    fn update_state(&mut self, state: &mut Self::State);

    /// The energy (Eh) of `state` in iteration `iter` (0: the start), with the entropy term of
    /// the iteration's occupations, and with `fock` the Fock matrices of `state` for a Fock
    /// mixer (one row-major n x n matrix per spin channel; a model mixed as a vector returns
    /// none). Times its own phases in `laps`.
    fn evaluate(
        &mut self,
        iter: usize,
        state: &Self::State,
        fock: bool,
        laps: &mut Laps,
    ) -> Result<(f64, Option<Vec<Vec<f64>>>)>;

    /// The RMS change of the density in the last [`update_state`](Self::update_state), for
    /// rules that test it; NaN (the default): not measured.
    fn density_change(&self) -> f64 {
        f64::NAN
    }

    /// The record of an iteration, after the rule read it (to keep a history or trace the
    /// SCF; ignored by default).
    fn record(&mut self, _it: &Iteration) {}
}

/// Run an SCF of `model` from `start` with `strategy` (see the module documentation). Errors of
/// the model (and of a Fock mixer) end it at once; ending without convergence is an outcome
/// ([`ScfOutcome::stop`]), for the caller to report or retry.
///
/// Phases in `laps`: `scf: mixing` (a Fock mixer), `scf: diagonalization`, `scf: density`,
/// `scf: mulliken`, `scf: energy+mixing` (a vector mixer: the energy, the residual and the
/// test), and the model's own in [`ScfModel::build_fock`] and [`ScfModel::evaluate`].
pub fn run_scf<M: ScfModel>(
    model: &mut M,
    strategy: ScfStrategy<'_>,
    start: M::State,
    laps: &mut Laps,
) -> Result<ScfOutcome<M::State>> {
    let ScfStrategy { mut accelerator, rule, max_iter } = strategy;
    let mut state = start;
    let mut last = Iteration { iteration: 0, energy: 0.0, energy_change: 0.0, residual: 0.0, density_change: f64::NAN };
    let mut iterations = 0;
    let mut stop = Stop::IterationLimit;
    // a vector mixer's output vector; a Fock mixer's next matrices (built with the energy)
    let mut packed = Vec::new();
    let mut fock = None;
    match &accelerator {
        Accelerator::Moments(mixer) => {
            let n = model.vector_len();
            if n == 0 || mixer.input().len() != n {
                return Err(Error::Input(format!(
                    "a vector mixer of {} variables for an SCF state of {n}",
                    mixer.input().len()
                )));
            }
            packed = vec![0.0; n];
        }
        // the start's Fock matrices (its energy is no iteration's)
        Accelerator::Fock(_) => fock = model.evaluate(0, &state, true, laps)?.1,
    }
    // the rule asked for an iteration with the Fock matrices as built
    let mut unmixed = false;
    for iter in 1..=max_iter {
        let (source, residual) = match &mut accelerator {
            Accelerator::Moments(mixer) => {
                if iter > 1 && !unmixed {
                    if !mixer.next() {
                        stop = Stop::MixerFailed;
                        break;
                    }
                    model.unpack(mixer.input(), &mut state);
                }
                iterations = iter;
                model.build_fock(iter, &state, laps)?;
                model.pack(&state, mixer.input_mut());
                (None, f64::NAN)
            }
            Accelerator::Fock(mixer) => {
                laps.skip();
                // (built with the previous energy, unless that iteration was the unmixed one)
                let f = match fock.take() {
                    Some(f) => f,
                    None => model.evaluate(iter - 1, &state, true, laps)?.1.ok_or_else(|| {
                        Error::Numerical("the SCF model built no Fock matrices for its Fock mixer".into())
                    })?,
                };
                let (f, residual) = if unmixed { (Some(f), f64::NAN) } else { (None, mixer.mix(iter, f)?) };
                laps.lap("scf: mixing");
                iterations = iter;
                (f, residual)
            }
        };
        let fock_source = match (&accelerator, &source) {
            (Accelerator::Moments(_), _) => FockSource::Built,
            (Accelerator::Fock(_), Some(f)) => FockSource::Unmixed(f),
            (Accelerator::Fock(mixer), None) => FockSource::Mixed(&**mixer),
        };
        if model.diagonalize(fock_source)?.is_break() {
            stop = Stop::Model;
            break;
        }
        drop(source);
        laps.lap("scf: diagonalization");
        model.density();
        laps.lap("scf: density");
        model.update_state(&mut state);
        laps.lap("scf: mulliken");
        let vector = match &mut accelerator {
            Accelerator::Moments(mixer) => {
                model.pack(&state, &mut packed);
                mixer.set_residual(&packed);
                true
            }
            Accelerator::Fock(_) => false,
        };
        let (energy, next) = model.evaluate(iter, &state, !vector && !unmixed, laps)?;
        let residual = match &accelerator {
            Accelerator::Moments(mixer) => rule.residual_norm(mixer.residual()),
            Accelerator::Fock(_) => residual,
        };
        let it = Iteration {
            iteration: iter,
            energy,
            energy_change: energy - last.energy,
            residual,
            density_change: model.density_change(),
        };
        let verdict = rule.update(&it);
        if let Accelerator::Fock(mixer) = &mut accelerator {
            mixer.observe(&it);
        }
        model.record(&it);
        if vector {
            laps.lap("scf: energy+mixing");
        }
        last = it;
        if verdict == Verdict::Done {
            stop = Stop::Converged;
            break;
        }
        unmixed = verdict == Verdict::OneMore { unmixed: true };
        fock = next;
    }
    Ok(ScfOutcome { state, iterations, converged: rule.converged(), stop, last })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A one-function model: the state is p, its Fock "matrix" f(p) = p/2 + 1/4 (fixed point
    /// p = 1/2), its density the diagonalized value; it logs what the driver asks of it.
    #[derive(Default)]
    struct Toy {
        diagonalized: f64,
        log: Vec<String>,
    }

    impl ScfModel for Toy {
        type State = f64;

        fn diagonalize(&mut self, fock: FockSource<'_>) -> Result<ControlFlow<()>> {
            let kind = if matches!(fock, FockSource::Unmixed(_)) { "unmixed" } else { "mixed" };
            self.diagonalized = fock.matrix(0).expect("a Fock mixer's matrices")[0];
            self.log.push(format!("diagonalize {kind}"));
            Ok(ControlFlow::Continue(()))
        }

        fn density(&mut self) {}

        fn update_state(&mut self, p: &mut f64) {
            *p = self.diagonalized;
        }

        fn evaluate(&mut self, iter: usize, p: &f64, fock: bool, _: &mut Laps) -> Result<(f64, Option<Vec<Vec<f64>>>)> {
            self.log.push(format!("evaluate {iter} {fock}"));
            Ok(((p - 0.5).powi(2), fock.then(|| vec![vec![0.5 * p + 0.25]])))
        }
    }

    /// Damping by 1/2 between the built and the last mixed matrix; counts its calls.
    #[derive(Default)]
    struct Damp(Vec<f64>, usize);

    impl FockMixer for Damp {
        fn mix(&mut self, _iter: usize, f: Vec<Vec<f64>>) -> Result<f64> {
            self.1 += 1;
            let r = self.0.first().map_or(0.0, |m| f[0][0] - m);
            self.0 = vec![self.0.first().map_or(f[0][0], |m| m + 0.5 * r)];
            Ok(r.abs())
        }
        fn mixed(&self, _s: usize) -> &[f64] {
            &self.0
        }
    }

    /// Converged at iteration `.0`, then one unmixed iteration that ends it.
    struct Script(usize, bool);

    impl ConvergenceRule for Script {
        fn update(&mut self, it: &Iteration) -> Verdict {
            match it.iteration {
                i if i < self.0 => Verdict::Continue,
                i if i == self.0 => Verdict::OneMore { unmixed: true },
                _ => {
                    self.1 = true;
                    Verdict::Done
                }
            }
        }
        fn converged(&self) -> bool {
            self.1
        }
    }

    /// The start's Fock matrices come with an energy that is no iteration's; the unmixed
    /// iteration diagonalizes the matrices built with the previous energy, does not mix them
    /// and builds none.
    #[test]
    fn fock_mixing_with_a_last_unmixed_iteration() {
        let (mut model, mut mixer, mut rule) = (Toy::default(), Damp::default(), Script(3, false));
        let strategy = ScfStrategy { accelerator: Accelerator::Fock(&mut mixer), rule: &mut rule, max_iter: 10 };
        let out = run_scf(&mut model, strategy, 0.0, &mut Laps::new()).unwrap();
        assert_eq!((out.stop, out.iterations, out.converged), (Stop::Converged, 4, true));
        assert_eq!(mixer.1, 3);
        let expected = [
            "evaluate 0 true",
            "diagonalize mixed",
            "evaluate 1 true",
            "diagonalize mixed",
            "evaluate 2 true",
            "diagonalize mixed",
            "evaluate 3 true",
            "diagonalize unmixed",
            "evaluate 4 false",
        ];
        assert_eq!(model.log, expected);
        // the unmixed iteration's density is f(p_3) as built: p_1 = 1/4 (no mixing in
        // iteration 1), p_2 = 5/16 (damped), p_3 = 23/64
        assert_eq!(out.state, 0.5 * (23.0 / 64.0) + 0.25);
        assert!(out.last.residual.is_nan() && out.last.density_change.is_nan());
        assert_eq!(out.last.energy, (out.state - 0.5).powi(2));
    }

    /// At the iteration limit after its OneMore the rule has not converged; a vector mixer is
    /// refused for a model without a vector layout.
    #[test]
    fn limits_and_refusals() {
        let (mut model, mut mixer, mut rule) = (Toy::default(), Damp::default(), Script(3, false));
        let strategy = ScfStrategy { accelerator: Accelerator::Fock(&mut mixer), rule: &mut rule, max_iter: 3 };
        let out = run_scf(&mut model, strategy, 0.0, &mut Laps::new()).unwrap();
        assert_eq!((out.stop, out.iterations, out.converged), (Stop::IterationLimit, 3, false));
        let mut diis = super::super::mixing::Diis::new(1, 0.5);
        let strategy = ScfStrategy { accelerator: Accelerator::Moments(&mut diis), rule: &mut rule, max_iter: 3 };
        assert!(matches!(run_scf(&mut Toy::default(), strategy, 0.0, &mut Laps::new()), Err(Error::Input(_))));
    }
}
