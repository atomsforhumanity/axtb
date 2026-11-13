// SPDX-License-Identifier: LGPL-3.0-or-later
//! Stateful calculator: build the method once for a molecule, then evaluate energies,
//! gradients and Hessians at any number of geometries through the [`Method`] trait.
//!
//! Threading: a calculator created with `threads = n > 1` owns one rayon pool of exactly
//! `n` threads ([`Threads`]) and runs all of its parallel work inside it. Work that is
//! parallel at the outer level (Hessian displacements) runs each inner single point
//! sequentially, so the thread count never multiplies. With `threads = 1` no thread is ever
//! spawned.

use crate::hessian::{HessianMethod, HessianOptions};
use crate::method::{
    AnalysisConventions, AnalysisDefaults, Attempt, Electronic, Electrons, Extras, Fingerprint, HessianDefault,
    HessianDefaults, Method, MethodInfo, QuadrupoleForm, Restart, RetryLadder, ScfData, ScfReport, ScfSettings, Term,
    Want, Wavefunction, WavefunctionData,
};
use crate::molecule::Molecule;
use crate::terms::{AnyMomentTerm, ChainLayout, MomentTerm};
use crate::xtb::{check_scf_options, Gfn2Result, ScfGuess, ScfOptions, XtbModel};
use crate::{Error, Result};
use crate::{Hamiltonian, Options};
use axtb_base::fd;
use axtb_base::threads::Threads;

pub use axtb_base::method::Hessian;

/// The canonical name of GFN2-xTB.
pub const METHOD: &str = "gfn2";

/// GFN2-xTB and its settings bound to one molecule (atomic numbers, charge, spin).
pub struct Calculator {
    hamiltonian: Hamiltonian,
    mol: Molecule,
    model: XtbModel,
    /// The terms added to the model, in order: dispersion, solvation, the user's
    /// ([`Calculator::push_term`]).
    extra: Vec<std::sync::Arc<dyn AnyMomentTerm>>,
    /// The settings. The SCF settings ([`Options::scf`], [`Options::mixing`],
    /// [`Options::initial_guess`], [`Options::trace`]) may change between calculations; the
    /// method's setup (conventions, dispersion, solvation) and the thread count (the
    /// calculator's pool) are fixed when the calculator is built.
    pub options: Options,
    /// The converged moments of the last SCF (or given ones), and whether they were given
    /// (they start the next SCF whatever the restart policy says).
    start: Option<ScfGuess>,
    given: bool,
    threads: Threads,
    info: MethodInfo,
    fingerprint: Fingerprint,
}

/// What GFN2-xTB is and offers in `conventions`.
fn info(conventions: crate::Conventions) -> MethodInfo {
    MethodInfo {
        name: METHOD,
        conventions: conventions.label(),
        electrons: Electrons::OneSet,
        hessian: HessianDefaults {
            default: HessianMethod::Analytic,
            offered: vec![
                HessianDefault { method: HessianMethod::Analytic, step: None, accuracy: Some(ANALYTIC_ACCURACY) },
                HessianDefault {
                    method: HessianMethod::Response,
                    step: Some(RESPONSE_STEP),
                    accuracy: Some(ANALYTIC_ACCURACY),
                },
                HessianDefault {
                    method: HessianMethod::Numerical,
                    step: Some(NUMERICAL_STEP),
                    accuracy: Some(NUMERICAL_ACCURACY),
                },
            ],
        },
        analysis: AnalysisDefaults { conventions: AnalysisConventions::from(conventions), xtb_frequency_scale: None },
        quadrupole: Some(QuadrupoleForm::from(conventions)),
    }
}

/// Default step (bohr) of the numerical Hessian.
pub const NUMERICAL_STEP: f64 = 0.005;
/// Default step (bohr) of the response Hessian.
pub const RESPONSE_STEP: f64 = 5e-4;
/// Default SCF accuracy of the numerical Hessian's displacements.
pub const NUMERICAL_ACCURACY: f64 = 0.01;
/// Default SCF accuracy of the analytic and response Hessians' reference.
pub const ANALYTIC_ACCURACY: f64 = 1e-3;

impl Calculator {
    /// Set up `hamiltonian` for a molecule; `positions` (bohr) give the initial geometry.
    pub fn new(
        hamiltonian: Hamiltonian,
        numbers: &[usize],
        positions: &[[f64; 3]],
        charge: f64,
        uhf: Option<usize>,
        options: Options,
    ) -> Result<Self> {
        check_scf_options(&options.run(options.threads, &Want::ENERGY))?;
        let threads = Threads::new(options.threads)?;
        Self::with_threads(hamiltonian, numbers, positions, charge, uhf, options, threads)
    }

    /// As [`Calculator::new`], but running on `threads` (a pool shared with other
    /// calculators, e.g. the parts of a composite method, so that they use one pool) instead
    /// of a pool of its own. It must have `options.threads` threads.
    pub fn with_threads(
        hamiltonian: Hamiltonian,
        numbers: &[usize],
        positions: &[[f64; 3]],
        charge: f64,
        uhf: Option<usize>,
        options: Options,
        threads: Threads,
    ) -> Result<Self> {
        check_scf_options(&options.run(options.threads, &Want::ENERGY))?;
        let n = threads.count();
        if n != options.threads {
            return Err(Error::Input(format!("a pool of {n} threads for a calculation on {}", options.threads)));
        }
        let mol = Molecule::new(numbers, positions, charge, uhf)?;
        let (model, extra) = crate::build_method(hamiltonian, &mol, &options)?;
        let info = info(options.conventions);
        let fingerprint = Fingerprint::of(METHOD, &mol);
        Ok(Self { hamiltonian, mol, model, extra, options, start: None, given: false, threads, info, fingerprint })
    }

    /// The Hamiltonian of this calculator.
    pub fn hamiltonian(&self) -> Hamiltonian {
        self.hamiltonian
    }

    /// The molecule (positions of the last evaluation).
    pub fn molecule(&self) -> &Molecule {
        &self.mol
    }

    /// Add an energy term to the model, after its dispersion and solvation terms and the terms
    /// added before: every later single point, gradient and Hessian includes it, and its pieces
    /// appear in the single point's terms after GFN2-xTB's own. Terms that read GFN2-xTB's
    /// coordination number find its slot in [`Calculator::layout`]. GFN2-xTB has no shell
    /// magnetizations, so a term that needs them is refused.
    pub fn push_term<T>(&mut self, term: T) -> Result<()>
    where
        T: MomentTerm + 'static,
        T::Cache: 'static,
    {
        if term.needs().magnetization {
            return Err(Error::Input("GFN2-xTB has no shell magnetizations for a term to read".into()));
        }
        self.extra.push(std::sync::Arc::new(term));
        Ok(())
    }

    /// The model's intermediates: GFN2-xTB's coordination number
    /// ([`Gfn2Cn::NAME`](crate::terms::Gfn2Cn::NAME)).
    pub fn layout(&self) -> &ChainLayout {
        &self.model.layout
    }

    fn run(&self, mol: &Molecule, scf: &ScfOptions, guess: Option<&ScfGuess>, gradient: bool) -> Result<Gfn2Result> {
        let mut sp = self.model.singlepoint(mol, &self.extra, scf, guess, gradient)?;
        // xtb (6.7.1 and 6.8.0, `get_wiberg`) leaves out pairs more than 10 bohr apart
        if matches!(self.options.conventions, crate::Conventions::Xtb | crate::Conventions::Xtb671) {
            if let Some(bo) = sp.bond_orders.as_mut() {
                crate::properties::cut_bond_orders(bo, &mol.positions, 10.0);
            }
        }
        Ok(sp)
    }

    /// `run` at the calculator's geometry inside its thread pool (outside it, parallel kernels
    /// would run on rayon's global pool, i.e. on every core instead of the threads asked for).
    fn run_pooled(&self, scf: &ScfOptions, guess: Option<&ScfGuess>, gradient: bool) -> Result<Gfn2Result> {
        self.threads.install(|| self.run(&self.mol, scf, guess, gradient))
    }

    /// The SCF settings of a calculation, with the thread count of the calculator's pool.
    fn scf_options(&self, want: &Want) -> ScfOptions {
        self.options.run(self.threads.count(), want)
    }

    /// The moments the next SCF starts from (`None`: the initial guess), taken: a given
    /// wavefunction always, the previous one when the restart policy reuses it.
    fn take_start(&mut self) -> Option<ScfGuess> {
        let reuse = self.given || self.options.scf.reuses(Restart::Previous);
        self.given = false;
        let start = self.start.take();
        start.filter(|_| reuse)
    }

    fn singlepoint_inner(&mut self, positions: &[[f64; 3]], want: &Want) -> Result<Gfn2Result> {
        let start = self.take_start();
        self.mol.set_positions(positions)?;
        let scf = self.scf_options(want);
        // a guess from a distant geometry can hurt; retry once from scratch (inside the pool:
        // outside it the parallel kernels would run on rayon's global pool, i.e. on every core
        // instead of the `threads` asked for)
        let ladder = RetryLadder::warm_then_fresh(start.is_some());
        let (res, _) = ladder
            .climb(|rung| self.run_pooled(&scf, if rung.fresh { None } else { start.as_ref() }, want.gradient))?;
        self.start = Some(res.guess());
        Ok(res)
    }

    fn hessian_inner(&mut self, positions: &[[f64; 3]], opts: &HessianOptions) -> Result<Hessian> {
        if let Some(step) = opts.step.filter(|s| !(*s > 0.0 && s.is_finite())) {
            return Err(Error::Input(format!("the Hessian step must be a positive number, got {step}")));
        }
        if let Some(acc) = opts.accuracy.filter(|a| !(*a > 0.0 && a.is_finite())) {
            return Err(Error::Input(format!("the Hessian's SCF accuracy must be a positive number, got {acc}")));
        }
        let start = self.take_start();
        self.mol.set_positions(positions)?;
        let t0 = std::time::Instant::now();
        let mut out = match opts.method.unwrap_or(HessianMethod::Analytic) {
            HessianMethod::Numerical => {
                self.numerical_hessian(start, opts.step.unwrap_or(NUMERICAL_STEP), opts.accuracy)
            }
            m @ (HessianMethod::Response | HessianMethod::Analytic) => self.response_hessian(start, m, opts),
        }?;
        out.timings.push(("total", t0.elapsed().as_secs_f64()));
        // the reference's moments start the next SCF (the single point at these positions)
        self.given = true;
        Ok(out)
    }

    fn response_hessian(
        &mut self,
        start: Option<ScfGuess>,
        method: HessianMethod,
        opts: &HessianOptions,
    ) -> Result<Hessian> {
        let mut scf = self.scf_options(&Want::ENERGY);
        scf.accuracy = opts.accuracy.unwrap_or(ANALYTIC_ACCURACY).min(scf.accuracy);
        let threads = scf.threads;
        let run = || -> Result<(Hessian, ScfGuess)> {
            let extra = &self.extra;
            let t0 = std::time::Instant::now();
            // a guess from a distant geometry can hurt; retry once from scratch
            let ladder = RetryLadder::warm_then_fresh(start.is_some());
            let ((reference, st), _) = ladder.climb(|rung| {
                let guess = if rung.fresh { None } else { start.as_ref() };
                self.model.singlepoint_state(&self.mol, extra, &scf, guess, true, true)
            })?;
            let st = st.expect("state");
            let mut timings = vec![("reference scf", t0.elapsed().as_secs_f64())];
            let (mut h, dipd, t) = match method {
                HessianMethod::Response => {
                    let step = opts.step.unwrap_or(RESPONSE_STEP);
                    crate::hessian::response_hessian(&self.model, extra, &self.mol, &st, threads, step)?
                }
                _ => crate::analytic::analytic_hessian(&self.model, &self.mol, &st, threads)?,
            };
            timings.extend(t);
            let max_asymmetry = fd::symmetrize(&mut h, 3 * self.mol.nat());
            let guess = reference.guess();
            Ok((finish(reference, h, max_asymmetry, dipd, method, timings), guess))
        };
        let (res, guess) = self.threads.install(run)?;
        self.start = Some(guess);
        Ok(res)
    }

    /// Semi-numerical Hessian from central differences of analytic gradients
    /// (step in bohr; 0.005 is a good default), [`fd::numerical_hessian`] with every atom moved
    /// forward by the step and back by twice the step. The SCF of every displacement is
    /// converged to `accuracy` (default 0.01 when `None`) starting from the reference
    /// wavefunction. Every SCF, the reference's too, runs sequentially, the displacements in
    /// parallel.
    fn numerical_hessian(&mut self, start: Option<ScfGuess>, step: f64, accuracy: Option<f64>) -> Result<Hessian> {
        let mut scf = self.scf_options(&Want::ENERGY);
        scf.accuracy = accuracy.unwrap_or(NUMERICAL_ACCURACY).min(scf.accuracy);
        scf.threads = 1;
        let t0 = std::time::Instant::now();
        let reference = self.threads.install(|| self.run(&self.mol, &scf, start.as_ref(), true))?;
        let mut timings = vec![("reference scf", t0.elapsed().as_secs_f64())];
        let guess = reference.guess();
        let point = |x: &[[f64; 3]]| -> Result<(Vec<[f64; 3]>, [f64; 3])> {
            let mut mol = self.mol.clone();
            mol.positions.copy_from_slice(x);
            let r = self.run(&mol, &scf, Some(&guess), true)?;
            Ok((r.gradient.unwrap(), r.dipole))
        };
        let t0 = std::time::Instant::now();
        let parallel = self.threads.parallel();
        let nh = self.threads.install(|| {
            fd::numerical_hessian(&self.mol.positions, step, fd::Displacement::ForwardBack, parallel, point)
        })?;
        timings.push(("displaced gradients", t0.elapsed().as_secs_f64()));
        self.start = Some(guess);
        let dipd = Some(nh.dipole_derivatives);
        Ok(finish(reference, nh.hessian, nh.max_asymmetry, dipd, HessianMethod::Numerical, timings))
    }

    /// The single point as the trait reports it: GFN2-xTB's terms (the eight xtb prints, then
    /// those of added terms), the molecular quadrupole in the conventions' form.
    fn report(&self, sp: Gfn2Result) -> crate::method::SinglePoint {
        let t = &sp.terms;
        let mut terms = vec![
            Term::piece("repulsion", t.repulsion),
            Term::piece("dispersion", t.dispersion),
            // the band energy Tr(H0 P) is only inside the electronic energy
            Term::subtotal("electronic", t.electronic),
            Term::piece("isotropic_es", t.isotropic_es),
            Term::piece("third_order", t.third_order),
            Term::piece("anisotropic", t.anisotropic),
            Term::piece("entropy", t.entropy),
            Term::piece("solvation", t.solvation),
        ];
        terms.extend(t.other.iter().map(|&(n, v)| Term::piece(n, v)));
        let form = self.info.quadrupole.unwrap_or_default();
        let quadrupole =
            crate::properties::molecular_quadrupole(&self.mol.positions, &sp.qat, &sp.dpat, &sp.qpat, form);
        let attempts = sp.scf_attempts.iter().map(|a| Attempt {
            mixer: a.mixer.label(),
            iterations: a.iterations,
            converged: a.converged,
        });
        crate::method::SinglePoint {
            energy: sp.energy,
            terms,
            gradient: sp.gradient,
            charges: sp.qat,
            dipole: sp.dipole,
            electronic: Some(Electronic {
                shell_charges: sp.qsh,
                atomic_dipoles: sp.dpat,
                atomic_quadrupoles: sp.qpat,
                quadrupole,
                levels: sp.levels,
                homo_lumo_gap: sp.homo_lumo_gap,
                orbitals: sp.orbitals,
                bond_orders: sp.bond_orders,
                scf: ScfReport {
                    iterations: sp.iterations,
                    start: sp.guess,
                    mixer_used: sp.mixer_used,
                    attempts: attempts.collect(),
                    eigensolver: sp.eigensolver,
                },
            }),
            extras: Extras::default(),
            timings: sp.timings,
        }
    }
}

impl Method for Calculator {
    fn info(&self) -> &MethodInfo {
        &self.info
    }

    /// Energy and what `want` asks for at `positions`. The SCF starts from the previous
    /// geometry's moments when the restart policy reuses them ([`ScfSettings::restart`]; `None`:
    /// always) or from given ones; if that SCF does not converge, it runs once more from the
    /// initial guess.
    fn singlepoint(&mut self, positions: &[[f64; 3]], want: &Want) -> Result<crate::method::SinglePoint> {
        match self.singlepoint_inner(positions, want) {
            Ok(sp) => Ok(self.report(sp)),
            Err(e) => {
                self.reset_wavefunction();
                Err(e)
            }
        }
    }

    /// Hessian at `positions` (see [`HessianMethod`] and [`HessianOptions`]): analytic (the
    /// default) or response with the reference converged to `accuracy` (default 1e-3), or
    /// numerical with every displacement converged to `accuracy` (default 0.01) from the
    /// reference's moments. The reference's moments start the next SCF (whatever the restart
    /// policy says), so that the single point at these positions starts from them.
    ///
    /// All methods run on the calculator's thread pool, parallel over displacements or
    /// perturbations with every inner piece sequential, so thread counts never multiply.
    fn hessian(&mut self, positions: &[[f64; 3]], opts: &HessianOptions) -> Result<Hessian> {
        let r = self.hessian_inner(positions, opts);
        if r.is_err() {
            self.reset_wavefunction();
        }
        r
    }

    fn wavefunction(&self) -> Option<Wavefunction> {
        let m = self.start.as_ref()?;
        let data = ScfData {
            shell_charges: m.qsh.clone(),
            atomic_dipoles: m.dpat.clone(),
            atomic_quadrupoles: m.qpat.clone(),
            ..ScfData::default()
        };
        Some(Wavefunction::new(self.fingerprint.clone(), WavefunctionData::Scf(data)))
    }

    /// Start the next SCF from the shell charges and atomic multipoles of `wf` (e.g. another
    /// GFN2-xTB calculator's, or another program's converged moments).
    fn set_wavefunction(&mut self, wf: &Wavefunction) -> Result<()> {
        wf.check(&self.fingerprint)?;
        let WavefunctionData::Scf(d) = wf.data() else {
            return Err(Error::Input("the wavefunction belongs to another kind of calculation".into()));
        };
        let nat = self.mol.nat();
        let nsh = self.model.basis.nsh;
        if d.shell_charges.len() != nsh || d.atomic_dipoles.len() != nat || d.atomic_quadrupoles.len() != nat {
            return Err(Error::Input(format!(
                "guess needs {nsh} shell charges and {nat} atomic dipoles/quadrupoles, got {}/{}/{}",
                d.shell_charges.len(),
                d.atomic_dipoles.len(),
                d.atomic_quadrupoles.len()
            )));
        }
        let mut values = d
            .shell_charges
            .iter()
            .chain(d.atomic_dipoles.iter().flatten())
            .chain(d.atomic_quadrupoles.iter().flatten());
        if values.any(|x| !x.is_finite()) {
            return Err(Error::Input("guess with non-finite charges or multipoles".into()));
        }
        self.start = Some(ScfGuess {
            qsh: d.shell_charges.clone(),
            dpat: d.atomic_dipoles.clone(),
            qpat: d.atomic_quadrupoles.clone(),
        });
        self.given = true;
        Ok(())
    }

    fn reset_wavefunction(&mut self) {
        self.start = None;
        self.given = false;
    }

    fn scf_settings(&self) -> Option<&ScfSettings> {
        Some(&self.options.scf)
    }

    fn set_scf_settings(&mut self, settings: ScfSettings) -> Result<()> {
        let mut o = self.options.clone();
        o.scf = settings;
        check_scf_options(&o.run(self.threads.count(), &Want::ENERGY))?;
        self.options.scf = o.scf;
        Ok(())
    }
}

/// Package a symmetrized Hessian `h` ([`fd::symmetrize`]; `max_asymmetry` from before).
fn finish(
    reference: Gfn2Result,
    h: Vec<f64>,
    max_asymmetry: f64,
    dipole_derivatives: Option<Vec<[f64; 3]>>,
    method: HessianMethod,
    timings: Vec<(&'static str, f64)>,
) -> Hessian {
    Hessian {
        energy: reference.energy,
        gradient: reference.gradient.unwrap(),
        hessian: h,
        max_asymmetry,
        dipole_derivatives,
        method,
        timings,
        parts: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::method::TermKind;

    /// GFN2-xTB's pieces (repulsion, dispersion, the electrostatic and entropy terms,
    /// solvation) sum to the energy minus the band energy Tr(H0 P), which xtb reports only
    /// inside the electronic energy (a subtotal); with D4 and ALPB, at 300 K.
    #[test]
    fn pieces_sum_to_the_energy_less_the_band_energy() {
        let numbers = [8, 1, 1, 7, 1, 1, 1];
        let positions = [
            [0.0, 0.0, 0.22],
            [0.0, 1.43, -0.89],
            [0.0, -1.43, -0.89],
            [0.0, 0.0, 5.6],
            [0.0, 1.77, 6.2],
            [1.53, -0.88, 6.2],
            [-1.53, -0.88, 6.2],
        ];
        let solvation = crate::SolvationSpec {
            model: crate::SolvationModel::Alpb,
            solvent: "water".into(),
            state: Default::default(),
        };
        let o = Options { solvation: Some(solvation), ..Default::default() };
        let mut calc = Calculator::new(Hamiltonian::Gfn2, &numbers, &positions, 0.0, None, o.clone()).unwrap();
        let sp = calc.singlepoint(&positions, &Want::ENERGY).unwrap();
        let kinds: Vec<TermKind> = sp.terms.iter().map(|t| t.kind).collect();
        assert_eq!(kinds.iter().filter(|k| **k == TermKind::Subtotal).count(), 1);
        assert_eq!(sp.terms[2].name, "electronic");
        let scf = o.run(1, &Want::ENERGY);
        let (_, st) = calc.model.singlepoint_state(&calc.mol, &calc.extra, &scf, None, false, true).unwrap();
        let st = st.unwrap();
        let (h0, p) = (&st.ints.h0, &st.density);
        let n = h0.nrows();
        let band: f64 = (0..n).map(|i| (0..n).map(|j| h0[(i, j)] * p[(j, i)]).sum::<f64>()).sum();
        let gap = sp.energy - sp.pieces();
        assert!((gap - band).abs() < 1e-10, "energy - pieces {gap} vs Tr(H0 P) {band}");
        assert!(band.abs() > 1.0, "the band energy is a large part of the energy");
    }
}
