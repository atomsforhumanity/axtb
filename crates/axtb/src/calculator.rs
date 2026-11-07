//! Stateful calculator: build the method once for a molecule, then evaluate energies,
//! gradients and Hessians at any number of geometries.
//!
//! Threading: a calculator created with `threads = n > 1` owns one rayon pool of exactly
//! `n` threads and runs all of its parallel work inside it. Work that is parallel at the
//! outer level (Hessian displacements) runs each inner single point sequentially, so the
//! thread count never multiplies. With `threads = 1` no thread is ever spawned.

use crate::error::{Error, Result};
use crate::hessian::{HessianMethod, HessianOptions};
use crate::molecule::Molecule;
use crate::xtb::{check_scf_options, ChargeDependentTerm, ScfGuess, ScfOptions, SinglePoint, XtbModel};
use crate::{Method, Options};
use rayon::prelude::*;
use std::sync::Arc;

/// A nuclear Hessian.
#[derive(Clone, Debug)]
pub struct Hessian {
    /// Energy and gradient at the reference geometry.
    pub energy: f64,
    pub gradient: Vec<[f64; 3]>,
    /// Row-major `3N x 3N` matrix of second derivatives (Eh/bohr²), symmetrized.
    pub hessian: Vec<f64>,
    /// Largest asymmetry |H_ij - H_ji| before symmetrization.
    pub max_asymmetry: f64,
    /// Dipole derivatives ∂μ/∂R_x (e; one row per Cartesian coordinate x = 3·atom + k), for
    /// IR intensities: from the SCC response (analytic and response Hessians) or central
    /// differences of the displaced dipoles (numerical); `None` when not available.
    pub dipole_derivatives: Option<Vec<[f64; 3]>>,
    /// How it was computed.
    pub method: HessianMethod,
    /// Wall time per phase in seconds.
    pub timings: Vec<(&'static str, f64)>,
}

/// Method and settings bound to one molecule (atomic numbers, charge, spin).
pub struct Calculator {
    method: Method,
    mol: Molecule,
    model: XtbModel,
    extra: Vec<Box<dyn ChargeDependentTerm>>,
    /// The settings. The SCF settings may change between calculations; the method's setup
    /// (conventions, dispersion, solvation) and the thread count (the calculator's pool) are
    /// fixed when the calculator is built.
    pub options: Options,
    guess: Option<ScfGuess>,
    pool: Option<Arc<rayon::ThreadPool>>,
}

impl Calculator {
    /// Set up `method` for a molecule; `positions` (bohr) give the initial geometry.
    pub fn new(
        method: Method,
        numbers: &[usize],
        positions: &[[f64; 3]],
        charge: f64,
        uhf: Option<usize>,
        options: Options,
    ) -> Result<Self> {
        check_scf_options(&options.scf)?;
        let pool = if options.scf.threads > 1 {
            Some(Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(options.scf.threads)
                    .build()
                    .map_err(|e| Error::Input(format!("cannot create thread pool: {e}")))?,
            ))
        } else {
            None
        };
        Self::with_pool(method, numbers, positions, charge, uhf, options, pool)
    }

    /// As [`Calculator::new`], but running in `pool` (shared with other calculators, e.g. the
    /// parts of a composite method, so that they use one pool) instead of a pool of its own.
    /// The pool must have `options.scf.threads` threads; `None` requires `threads = 1`.
    pub fn with_pool(
        method: Method,
        numbers: &[usize],
        positions: &[[f64; 3]],
        charge: f64,
        uhf: Option<usize>,
        options: Options,
        pool: Option<Arc<rayon::ThreadPool>>,
    ) -> Result<Self> {
        check_scf_options(&options.scf)?;
        let threads = pool.as_ref().map_or(1, |p| p.current_num_threads());
        if threads != options.scf.threads {
            return Err(Error::Input(format!(
                "a pool of {threads} threads for a calculation on {}",
                options.scf.threads
            )));
        }
        let mol = Molecule::new(numbers, positions, charge, uhf)?;
        let (model, extra) = crate::build_method(method, &mol, &options)?;
        Ok(Self { method, mol, model, extra, options, guess: None, pool })
    }

    /// The method of this calculator.
    pub fn method(&self) -> Method {
        self.method
    }

    /// The molecule (positions of the last evaluation).
    pub fn molecule(&self) -> &Molecule {
        &self.mol
    }

    /// Forget the wavefunction of the previous geometry.
    pub fn reset_guess(&mut self) {
        self.guess = None;
    }

    /// The wavefunction the next SCF starts from (the previous geometry's, or one given with
    /// [`Calculator::set_guess`]).
    pub fn guess(&self) -> Option<&ScfGuess> {
        self.guess.as_ref()
    }

    /// Start the next SCF from given shell charges and atomic multipoles (e.g. another
    /// program's converged moments, to check whether two programs found different SCF
    /// solutions of the same molecule).
    pub fn set_guess(&mut self, guess: ScfGuess) -> Result<()> {
        let nat = self.mol.nat();
        let nsh = self.model.basis.nsh;
        if guess.qsh.len() != nsh || guess.dpat.len() != nat || guess.qpat.len() != nat {
            return Err(Error::Input(format!(
                "guess needs {nsh} shell charges and {nat} atomic dipoles/quadrupoles, got {}/{}/{}",
                guess.qsh.len(),
                guess.dpat.len(),
                guess.qpat.len()
            )));
        }
        let mut values = guess.qsh.iter().chain(guess.dpat.iter().flatten()).chain(guess.qpat.iter().flatten());
        if values.any(|x| !x.is_finite()) {
            return Err(Error::Input("guess with non-finite charges or multipoles".into()));
        }
        self.guess = Some(guess);
        Ok(())
    }

    fn run(&self, mol: &Molecule, scf: &ScfOptions, guess: Option<&ScfGuess>, gradient: bool) -> Result<SinglePoint> {
        let extra: Vec<&dyn ChargeDependentTerm> = self.extra.iter().map(|b| b.as_ref()).collect();
        let mut sp = self.model.singlepoint(mol, &extra, scf, guess, gradient)?;
        // xtb (6.7.1 and 6.8.0, `get_wiberg`) leaves out pairs more than 10 bohr apart
        if matches!(self.options.conventions, crate::Conventions::Xtb | crate::Conventions::Xtb671) {
            if let Some(bo) = sp.bond_orders.as_mut() {
                crate::properties::cut_bond_orders(bo, &mol.positions, 10.0);
            }
        }
        Ok(sp)
    }

    /// `f` inside this calculator's thread pool (directly without one): outside it, parallel
    /// kernels would run on rayon's global pool, i.e. on every core instead of the threads
    /// asked for.
    fn in_pool<T: Send>(&self, f: impl FnOnce() -> T + Send) -> T {
        match &self.pool {
            Some(pool) => pool.install(f),
            None => f(),
        }
    }

    /// `run` at the calculator's geometry inside its thread pool.
    fn run_pooled(&self, scf: &ScfOptions, guess: Option<&ScfGuess>, gradient: bool) -> Result<SinglePoint> {
        self.in_pool(|| self.run(&self.mol, scf, guess, gradient))
    }

    /// The SCF settings of a calculation, with the thread count of the calculator's pool.
    fn scf_options(&self) -> ScfOptions {
        let threads = self.pool.as_ref().map_or(1, |p| p.current_num_threads());
        ScfOptions { threads, ..self.options.scf.clone() }
    }

    /// Energy and optionally gradient at `positions`; starts from the previous geometry's
    /// wavefunction when there is one.
    pub fn singlepoint(&mut self, positions: &[[f64; 3]], gradient: bool) -> Result<SinglePoint> {
        self.mol.set_positions(positions)?;
        let scf = self.scf_options();
        let res = self.run_pooled(&scf, self.guess.as_ref(), gradient);
        let res = match res {
            // a guess from a distant geometry can hurt; retry once from scratch (inside the
            // pool: outside it the parallel kernels would run on rayon's global pool, i.e.
            // on every core instead of the `threads` asked for)
            Err(Error::ScfNotConverged { .. }) if self.guess.is_some() => {
                self.guess = None;
                self.run_pooled(&scf, None, gradient)?
            }
            other => other?,
        };
        self.guess = Some(res.guess());
        Ok(res)
    }

    /// Hessian at `positions` (see [`HessianMethod`] and [`HessianOptions`]).
    ///
    /// All methods run on the calculator's thread pool, parallel over displacements or
    /// perturbations with every inner piece sequential, so thread counts never multiply.
    pub fn hessian(&mut self, positions: &[[f64; 3]], opts: &HessianOptions) -> Result<Hessian> {
        if let Some(step) = opts.step.filter(|s| !(*s > 0.0 && s.is_finite())) {
            return Err(Error::Input(format!("the Hessian step must be a positive number, got {step}")));
        }
        if let Some(acc) = opts.accuracy.filter(|a| !(*a > 0.0 && a.is_finite())) {
            return Err(Error::Input(format!("the Hessian's SCF accuracy must be a positive number, got {acc}")));
        }
        self.mol.set_positions(positions)?;
        let t0 = std::time::Instant::now();
        let mut out = match opts.method {
            HessianMethod::Numerical => self.numerical_hessian(opts.step.unwrap_or(0.005), opts.accuracy),
            HessianMethod::Response | HessianMethod::Analytic => self.response_hessian(opts),
        }?;
        out.timings.push(("total", t0.elapsed().as_secs_f64()));
        Ok(out)
    }

    fn response_hessian(&mut self, opts: &HessianOptions) -> Result<Hessian> {
        let mut scf = self.scf_options();
        scf.accuracy = opts.accuracy.unwrap_or(1e-3).min(scf.accuracy);
        let threads = scf.threads;
        let run = || -> Result<(Hessian, ScfGuess)> {
            let extra: Vec<&dyn ChargeDependentTerm> = self.extra.iter().map(|b| b.as_ref()).collect();
            let t0 = std::time::Instant::now();
            let (reference, st) =
                match self.model.singlepoint_state(&self.mol, &extra, &scf, self.guess.as_ref(), true, true) {
                    // a guess from a distant geometry can hurt; retry once from scratch
                    Err(Error::ScfNotConverged { .. }) if self.guess.is_some() => {
                        self.model.singlepoint_state(&self.mol, &extra, &scf, None, true, true)?
                    }
                    other => other?,
                };
            let st = st.expect("state");
            let mut timings = vec![("reference scf", t0.elapsed().as_secs_f64())];
            let (h, dipd, t) = match opts.method {
                HessianMethod::Response => {
                    let step = opts.step.unwrap_or(5e-4);
                    crate::hessian::response_hessian(&self.model, &extra, &self.mol, &st, threads, step)?
                }
                _ => crate::analytic::analytic_hessian(&self.model, &extra, &self.mol, &st, threads)?,
            };
            timings.extend(t);
            let guess = reference.guess();
            Ok((finish(reference, h, dipd, opts.method, timings), guess))
        };
        let (res, guess) = self.in_pool(run)?;
        self.guess = Some(guess);
        Ok(res)
    }

    /// Semi-numerical Hessian from central differences of analytic gradients
    /// (step in bohr; 0.005 is a good default). The SCF of every displacement is converged
    /// to `accuracy` (default 0.01 when `None`) starting from the reference wavefunction.
    /// Every SCF, the reference's too, runs sequentially, the displacements in parallel.
    fn numerical_hessian(&mut self, step: f64, accuracy: Option<f64>) -> Result<Hessian> {
        let mut scf = self.scf_options();
        scf.accuracy = accuracy.unwrap_or(0.01).min(scf.accuracy);
        scf.threads = 1;
        let t0 = std::time::Instant::now();
        let reference = self.in_pool(|| self.run(&self.mol, &scf, self.guess.as_ref(), true))?;
        let mut timings = vec![("reference scf", t0.elapsed().as_secs_f64())];
        let guess = reference.guess();
        let n3 = 3 * self.mol.nat();
        let column = |idx: usize| -> Result<(Vec<f64>, [f64; 3])> {
            let (atom, dir) = (idx / 3, idx % 3);
            let mut mol = self.mol.clone();
            mol.positions[atom][dir] += step;
            let rp = self.run(&mol, &scf, Some(&guess), true)?;
            mol.positions[atom][dir] -= 2.0 * step;
            let rm = self.run(&mol, &scf, Some(&guess), true)?;
            let (gp, gm) = (rp.gradient.unwrap(), rm.gradient.unwrap());
            let dmu = [0, 1, 2].map(|k| (rp.dipole[k] - rm.dipole[k]) / (2.0 * step));
            Ok(((0..n3).map(|j| (gp[j / 3][j % 3] - gm[j / 3][j % 3]) / (2.0 * step)).collect(), dmu))
        };
        let t0 = std::time::Instant::now();
        let cols: Vec<Result<(Vec<f64>, [f64; 3])>> = if self.pool.is_some() {
            self.in_pool(|| (0..n3).into_par_iter().map(column).collect())
        } else {
            (0..n3).map(column).collect()
        };
        let mut h = vec![0.0; n3 * n3];
        let mut dipd = Vec::with_capacity(n3);
        for (i, col) in cols.into_iter().enumerate() {
            let (col, dmu) = col?;
            for j in 0..n3 {
                h[j * n3 + i] = col[j];
            }
            dipd.push(dmu);
        }
        timings.push(("displaced gradients", t0.elapsed().as_secs_f64()));
        self.guess = Some(guess);
        Ok(finish(reference, h, Some(dipd), HessianMethod::Numerical, timings))
    }
}

/// Symmetrize and package a Hessian.
fn finish(
    reference: SinglePoint,
    mut h: Vec<f64>,
    dipole_derivatives: Option<Vec<[f64; 3]>>,
    method: HessianMethod,
    timings: Vec<(&'static str, f64)>,
) -> Hessian {
    let n3 = 3 * reference.qat.len();
    let mut max_asym: f64 = 0.0;
    for i in 0..n3 {
        for j in 0..i {
            let (a, b) = (h[i * n3 + j], h[j * n3 + i]);
            max_asym = max_asym.max((a - b).abs());
            let m = 0.5 * (a + b);
            h[i * n3 + j] = m;
            h[j * n3 + i] = m;
        }
    }
    Hessian {
        energy: reference.energy,
        gradient: reference.gradient.unwrap(),
        hessian: h,
        max_asymmetry: max_asym,
        dipole_derivatives,
        method,
        timings,
    }
}
