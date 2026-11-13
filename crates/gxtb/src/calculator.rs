//! Calculator of g-xTB (2.0.1 and v1.1): axtb-base's [`Method`], so that the axtb command-line
//! driver, Python package and qccodec codec select it by method name (`"gxtb-v2.0.1"`,
//! `"gxtb-v1.1"`).
//!
//! g-xTB lives in this GPL-3.0 crate, which builds on axtb (LGPL-3.0); the method trait is
//! axtb-base's, so the front ends hold it like every other method.
//!
//! Energies, analytic gradients (2.0.1: [`crate::v2::gradient`]; v1.1: [`crate::response`])
//! and Hessians ([`Method::hessian`]: for 2.0.1 analytic (the default) or response,
//! [`crate::v2::hessian`]; for both versions numerical, central differences of analytic
//! gradients as GFN2-xTB's), with the settings of [`Options`] (refused when out of range or
//! meant for the other version, [`ScfOptions::check`]). Where an SCF starts is the restart
//! policy [`ScfSettings::restart`]: in the default conventions from the previous geometry's
//! converged state (the programs' restart files `xtbrestart` and `gxtbrestart` carry the same),
//! in the program conventions from the program's EEQ-BC start, so that every result is that of
//! a fresh program run.
//!
//! Threading (as in axtb): with `threads > 1` the calculator owns one rayon pool of exactly
//! that size (`axtb_base::threads::Threads`) and runs everything inside it; with `threads = 1`
//! no thread is spawned and faer's global parallelism is never touched.

pub use crate::options::Version;
use crate::scf::{Conventions, ScfOptions, ScfResult, Setup};
use crate::timing::Timings;
use crate::v2;
use axtb::method::{
    AnalysisConventions, AnalysisDefaults, Electronic, Electrons, Extras, Fingerprint, HessianDefault, HessianDefaults,
    Method, MethodInfo, QuadrupoleForm, Restart, RetryLadder, ScfData, ScfReport, ScfSettings, SinglePoint, Term, Want,
    Wavefunction, WavefunctionData,
};
use axtb::molden::{MoSet, Orbitals, Spin};
use axtb::properties::{homo_lumo_gap, mayer_bond_orders, molecular_quadrupole, GapRule, OrbitalLevels};
use axtb::{Error, HessianMethod, HessianOptions, Molecule, Result};
use axtb_base::fd;
use axtb_base::threads::Threads;
use faer::Par;
use std::time::Instant;

impl v2::scf::Energies {
    /// The terms with their names as the axtb front ends report them for g-xTB 2.0.1: every
    /// term a piece of the energy but `electronic`, the SCF energy (a subtotal).
    pub fn terms(&self) -> Vec<Term> {
        vec![
            Term::piece("one_electron", self.one_electron),
            Term::piece("first_order", self.first_order),
            Term::piece("second_order", self.second_order),
            Term::piece("third_order", self.third_order),
            Term::piece("fourth_order", self.fourth_order),
            Term::piece("es_multipole", self.aes),
            Term::piece("exchange", self.exchange),
            Term::piece("spin_polarization", self.spin),
            Term::piece("dispersion", self.dispersion),
            Term::piece("repulsion", self.repulsion),
            Term::piece("entropy", self.entropy),
            Term::subtotal("electronic", self.electronic),
            Term::piece("increments", self.increments),
            Term::piece("dispersion_three_body", self.dispersion_three_body),
        ]
    }
}

impl crate::scf::Energies {
    /// The terms with their names as the axtb front ends report them for g-xTB v1.1: every term
    /// a piece of the energy but `electronic`, the SCF energy (a subtotal). v1.1 has `es1`
    /// (first order) and `es2_3` (second to fourth order) and splits the exchange into
    /// `exchange_mulliken` and `exchange_onsite`, as the 1.1.0 program prints them.
    pub fn terms(&self) -> Vec<Term> {
        vec![
            Term::piece("one_electron", self.one_electron),
            Term::piece("es1", self.es1),
            Term::piece("es2_3", self.es23),
            Term::piece("es_multipole", self.aes),
            Term::piece("exchange_mulliken", self.mfx),
            Term::piece("exchange_onsite", self.ofx),
            Term::piece("spin_polarization", self.spin),
            Term::piece("entropy", self.entropy),
            Term::subtotal("electronic", self.electronic),
            Term::piece("increments", self.increments),
            Term::piece("dispersion", self.disp2),
            Term::piece("dispersion_three_body", self.disp3),
            Term::piece("repulsion", self.repulsion),
        ]
    }
}

/// The converger settings of the programs' SCFs; `None`: the program's (see [`ScfOptions`]).
/// Some belong to one version: `density_tol` and `transition_iteration` to v1.1, `residual_tol`
/// to 2.0.1; the other version refuses them.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ConvergerSettings {
    /// Explicit threshold of the energy change (Eh).
    pub energy_tol: Option<f64>,
    /// v1.1 only: explicit threshold of the RMS change of the packed spin densities.
    pub density_tol: Option<f64>,
    /// 2.0.1 only: explicit threshold of the RMS Fock residual.
    pub residual_tol: Option<f64>,
    /// Damping factor in (0, 1] of the first iterations; `None`: v1.1 [`crate::scf::DAMPING`],
    /// 2.0.1 0.2.
    pub damping: Option<f64>,
    /// v1.1 only: iteration of the transition step; `None`: 3 (open shell: 4).
    pub transition_iteration: Option<usize>,
    /// Largest number of stored Fock matrices of the Pulay (DIIS) mixing, at least 1; `None`:
    /// v1.1 7 (open shell: 8), 2.0.1 7.
    pub diis_size: Option<usize>,
}

/// Settings of a g-xTB calculator.
#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    /// The SCF settings every SCF method has: accuracy, iteration limit (`None`: the program's,
    /// [`crate::scf::MAX_ITER`] for v1.1, [`crate::v2::scf::MAX_ITER`] for 2.0.1), electronic
    /// temperature (`None`: 0 K, the programs' default: Aufbau occupations), restart policy
    /// (`None`: previous in the axtb conventions, the program's start in the program's).
    pub scf: ScfSettings,
    /// The programs' converger settings.
    pub converger: ConvergerSettings,
    /// v1.1: the Fock matrix of the SCF, the exact derivative of the energy or the program's;
    /// 2.0.1: units, electron count and cutoffs (the program's or the default ones).
    pub conventions: Conventions,
    /// Threads of the calculator's pool (1: no thread is spawned), fixed when it is built.
    pub threads: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            scf: ScfSettings::default(),
            converger: ConvergerSettings::default(),
            conventions: Conventions::default(),
            threads: 1,
        }
    }
}

impl Options {
    /// The settings of one SCF; `all_orbitals`: every orbital.
    pub fn run(&self, all_orbitals: bool) -> ScfOptions {
        let c = &self.converger;
        ScfOptions {
            max_iter: self.scf.max_iter.unwrap_or(0),
            accuracy: self.scf.accuracy,
            energy_tol: c.energy_tol,
            density_tol: c.density_tol,
            residual_tol: c.residual_tol,
            damping: c.damping,
            transition_iteration: c.transition_iteration,
            diis_size: c.diis_size,
            threads: self.threads,
            etemp: self.scf.etemp.unwrap_or(ETEMP),
            conventions: self.conventions,
            all_orbitals,
        }
    }

    /// Whether the SCFs start from the previous geometry's converged state.
    pub fn reuses(&self) -> bool {
        let default = if self.conventions == Conventions::Axtb { Restart::Previous } else { Restart::Fresh };
        self.scf.reuses(default)
    }

    /// The SCF iteration limit for `version`.
    pub fn iteration_limit(&self, version: Version) -> usize {
        self.run(false).iteration_limit(version)
    }

    /// Refuse settings that are out of range or belong to the other version.
    pub fn check(&self, version: Version) -> Result<()> {
        self.run(false).check(version)
    }
}

/// g-xTB's default electronic temperature (K), the programs' (Aufbau occupations).
pub const ETEMP: f64 = 0.0;

/// The geometry-dependent setup of either version.
#[derive(Clone, Debug)]
enum AnySetup {
    V2(Box<v2::scf::Setup>),
    V1(Box<Setup>),
}

/// The converged SCF state a later SCF of the same molecule can start from, as the programs'
/// restart files carry it.
#[derive(Clone, Debug)]
enum Start {
    /// 2.0.1: shell charges, atomic dipoles and quadrupoles, shell magnetizations and spin
    /// densities, as `xtbrestart`.
    V2_0_1(Box<v2::scf::State>),
    /// v1.1: the spin densities (closed shell: the total density), as `gxtbrestart`.
    V1_1(Vec<Vec<f64>>),
}

/// g-xTB bound to one molecule (atomic numbers, charge, unpaired electrons).
pub struct Calculator {
    mol: Molecule,
    /// Settings; checked by every calculation ([`Options::check`]). `threads` must stay the
    /// number the calculator was built with.
    pub options: Options,
    version: Version,
    threads: Threads,
    /// Setup of the current geometry (built in `new`, reused while the positions are equal).
    setup: Option<AnySetup>,
    /// The last converged state (or one given with [`Method::set_wavefunction`]).
    state: Option<Start>,
    /// Start the next SCF from `state` even if the options do not reuse it (it was given).
    given: bool,
    info: MethodInfo,
    fingerprint: Fingerprint,
}

/// The setup of `mol` for `version` in the conventions of `opts`, on `opts.threads` threads
/// (inside the caller's pool).
fn new_setup(version: Version, mol: &Molecule, opts: &ScfOptions) -> Result<AnySetup> {
    Ok(match version {
        Version::V2_0_1 => AnySetup::V2(Box::new(v2::scf::Setup::with_threads(
            mol,
            crate::v2::Model::v2_0_1(opts.conventions),
            opts.threads,
        )?)),
        Version::V1_1 => {
            let mut s = Setup::new(mol)?;
            s.set_threads(opts.threads)?;
            AnySetup::V1(Box::new(s))
        }
    })
}

/// Factor on the 2.0.1 convergence thresholds of an SCF that starts from a previous state in
/// the default conventions. A fresh 2.0.1 SCF ends with a DIIS step that overshoots the
/// program's thresholds (ΔE < 1e-6·acc Eh, Fock residual RMS < 2e-5·acc); a restarted one meets
/// them in 2-5 iterations and stops right there, with errors twice those of a fresh SCF that
/// depend on the history (caffeine: gradients 1.4e-4 vs 7e-5 Eh/bohr from the converged ones),
/// which made geomeTRIC take 48 instead of 23 steps. With 0.1 the errors are half a fresh SCF's
/// and restarts still take fewer iterations (`tests/gxtb_parity_features.py`). The program
/// conventions keep the program's thresholds, so that its restarts are reproduced.
pub const RESTART_ACCURACY: f64 = 0.1;

/// The 2.0.1 SCF options of a calculation from `start` (`opts`, with the thresholds scaled by
/// [`RESTART_ACCURACY`] for a restart in the default conventions).
fn v2_options(opts: &ScfOptions, restart: bool) -> ScfOptions {
    let mut o = opts.clone();
    if restart && opts.conventions == Conventions::Axtb {
        o.accuracy *= RESTART_ACCURACY;
    }
    o
}

/// Default step (bohr) of the numerical Hessian.
pub const NUMERICAL_STEP: f64 = 0.005;
/// Default SCF accuracy of every displacement of the numerical Hessian.
pub const NUMERICAL_ACCURACY: f64 = 0.01;
/// Default step (bohr) of the 2.0.1 response Hessian.
pub const RESPONSE_STEP: f64 = 2e-4;
/// Default SCF accuracy of the reference of the 2.0.1 analytic and response Hessians (1e-11 Eh,
/// residual 2e-10: the error of both 2.0.1 methods is first order in the reference residual, and
/// 1e-3 leaves ~2e-7 Eh/bohr^2 on caffeine for no measurable saving).
pub const ANALYTIC_ACCURACY: f64 = 1e-5;

/// What g-xTB `version` is and offers in `conventions`.
fn info(version: Version, conventions: Conventions) -> MethodInfo {
    let numerical = HessianDefault {
        method: HessianMethod::Numerical,
        step: Some(NUMERICAL_STEP),
        accuracy: Some(NUMERICAL_ACCURACY),
    };
    let hessian = match version {
        Version::V2_0_1 => HessianDefaults {
            default: HessianMethod::Analytic,
            offered: vec![
                HessianDefault { method: HessianMethod::Analytic, step: None, accuracy: Some(ANALYTIC_ACCURACY) },
                HessianDefault {
                    method: HessianMethod::Response,
                    step: Some(RESPONSE_STEP),
                    accuracy: Some(ANALYTIC_ACCURACY),
                },
                numerical,
            ],
        },
        Version::V1_1 => HessianDefaults { default: HessianMethod::Numerical, offered: vec![numerical] },
    };
    MethodInfo {
        name: version.method(),
        conventions: conventions.label(),
        electrons: Electrons::PerSpin,
        hessian,
        // the g-xTB 2.0.1 program is a 6.7.1-based xtb build
        analysis: AnalysisDefaults {
            conventions: match conventions {
                Conventions::Gxtb => AnalysisConventions::Xtb671,
                Conventions::Axtb => AnalysisConventions::Axtb,
            },
            xtb_frequency_scale: None,
        },
        // the 2.0.1 program's quadrupole (`xtb --gxtb --quadrupole`, tblite's form) in its
        // conventions, the traceless tensor otherwise (the 1.1.0 program prints none)
        quadrupole: Some(match (version, conventions) {
            (Version::V2_0_1, Conventions::Gxtb) => QuadrupoleForm::Tblite,
            _ => QuadrupoleForm::Traceless,
        }),
    }
}

impl Calculator {
    /// Set up g-xTB `version` for a molecule: atomic numbers, initial positions (bohr), integer
    /// total charge and `uhf` unpaired electrons (`None`: 0 or 1 by electron count). Fails on
    /// elements without parameters, impossible charge and spin, and invalid `options`.
    pub fn new(
        version: Version,
        numbers: &[usize],
        positions: &[[f64; 3]],
        charge: f64,
        uhf: Option<usize>,
        options: Options,
    ) -> Result<Self> {
        let threads = Threads::new(options.threads)?;
        Self::with_threads(version, numbers, positions, charge, uhf, options, threads)
    }

    /// As [`Calculator::new`], but running on `threads` (a pool shared with other calculators,
    /// e.g. the parts of a composite method) instead of a pool of its own. It must have
    /// `options.threads` threads.
    pub fn with_threads(
        version: Version,
        numbers: &[usize],
        positions: &[[f64; 3]],
        charge: f64,
        uhf: Option<usize>,
        options: Options,
        threads: Threads,
    ) -> Result<Self> {
        options.check(version)?;
        let n = threads.count();
        if n != options.threads {
            return Err(Error::Input(format!("a pool of {n} threads for a calculation on {}", options.threads)));
        }
        let mol = Molecule::new(numbers, positions, charge, uhf)?;
        // fails early on elements without parameters and impossible charge/spin (built in the
        // pool: its loops use the calculator's threads)
        let setup = threads.install(|| new_setup(version, &mol, &options.run(false)))?;
        let info = info(version, options.conventions);
        let fingerprint = Fingerprint::of(version.method(), &mol);
        Ok(Self { mol, options, version, threads, setup: Some(setup), state: None, given: false, info, fingerprint })
    }

    /// The g-xTB version.
    pub fn version(&self) -> Version {
        self.version
    }

    /// The molecule (positions of the last evaluation).
    pub fn molecule(&self) -> &Molecule {
        &self.mol
    }

    /// Whether the SCFs start from the previous converged state ([`ScfSettings::restart`]).
    pub fn reuses_guess(&self) -> bool {
        self.options.reuses()
    }

    /// Check the options of the next calculation (and follow a change of conventions).
    fn check_options(&mut self) -> Result<()> {
        self.options.check(self.version)?;
        let threads = self.threads.count();
        if self.options.threads != threads {
            return Err(Error::Input(format!(
                "threads changed from {threads} to {} after the calculator was built",
                self.options.threads
            )));
        }
        if self.info.conventions != self.options.conventions.label() {
            self.info = info(self.version, self.options.conventions);
        }
        Ok(())
    }

    /// The state the next SCF starts from (`None`: the program's start), taken: a calculation
    /// that fails leaves no state behind.
    fn take_start(&mut self) -> Option<Start> {
        let start = if self.given || self.options.reuses() { self.state.take() } else { None };
        self.given = false;
        start
    }

    /// Move to `positions` (bohr): the setup is rebuilt when they or the conventions changed.
    fn move_to(&mut self, positions: &[[f64; 3]]) -> Result<()> {
        if self.mol.positions.as_slice() != positions {
            self.mol.set_positions(positions)?;
            self.setup = None;
        }
        if matches!(&self.setup, Some(AnySetup::V2(s)) if s.model.conventions != self.options.conventions) {
            self.setup = None;
        }
        Ok(())
    }

    fn singlepoint_inner(&mut self, positions: &[[f64; 3]], want: &Want) -> Result<SinglePoint> {
        self.check_options()?;
        self.move_to(positions)?;
        let start = self.take_start();
        let mut setup = self.setup.take();
        let opts = self.options.run(want.orbitals);
        let (mol, version, threads, form) = (&self.mol, self.version, &self.threads, self.info.quadrupole);
        // a state from a distant geometry can hurt; once more from the program's start
        let ladder = RetryLadder::warm_then_fresh(start.is_some());
        let res = ladder.climb(|rung| {
            let s = if rung.fresh { None } else { start.as_ref() };
            let (r, back) = threads.install(|| run(version, mol, &opts, setup.take(), s, want, form));
            setup = back;
            r
        });
        self.setup = setup;
        let ((r, state), _) = res?;
        self.state = Some(state);
        Ok(r)
    }

    fn hessian_inner(&mut self, positions: &[[f64; 3]], opts: &HessianOptions) -> Result<axtb::Hessian> {
        let t0 = Instant::now();
        let method = opts.method.unwrap_or(self.info.hessian.default);
        let mut out = match (method, self.version) {
            (HessianMethod::Numerical, _) => {
                self.numerical(positions, opts.step.unwrap_or(NUMERICAL_STEP), opts.accuracy)
            }
            (m @ (HessianMethod::Response | HessianMethod::Analytic), Version::V2_0_1) => {
                self.response_hessian(positions, opts, m)
            }
            (m, v) => Err(unavailable(m, v)),
        }?;
        out.timings.push(("total", t0.elapsed().as_secs_f64()));
        Ok(out)
    }

    /// The SCF accuracy of a Hessian: `accuracy` (default `default`), never looser than the
    /// calculator's own.
    fn hessian_accuracy(&self, accuracy: Option<f64>, default: f64) -> Result<f64> {
        let accuracy = accuracy.unwrap_or(default);
        if !(accuracy > 0.0 && accuracy.is_finite()) {
            return Err(Error::Input(format!("accuracy must be positive and finite, got {accuracy}")));
        }
        Ok(accuracy.min(self.options.scf.accuracy))
    }

    fn numerical(&mut self, positions: &[[f64; 3]], step: f64, accuracy: Option<f64>) -> Result<axtb::Hessian> {
        if !(step > 0.0 && step.is_finite()) {
            return Err(Error::Input(format!("Hessian step must be positive and finite, got {step}")));
        }
        let accuracy = self.hessian_accuracy(accuracy, NUMERICAL_ACCURACY)?;
        self.check_options()?;
        self.move_to(positions)?;
        let opts = ScfOptions { accuracy, threads: 1, ..self.options.run(false) };
        let mol = &self.mol;
        let version = self.version;
        // energy, gradient and dipole of one geometry
        let point = |m: &Molecule| -> Result<(f64, Vec<[f64; 3]>, [f64; 3])> {
            let (energy, g, qa, dip) = match version {
                Version::V2_0_1 => {
                    let setup = v2::scf::Setup::new(m, crate::v2::Model::v2_0_1(opts.conventions))?;
                    let r = setup.scf(&opts)?;
                    let g = setup.gradient(&r)?;
                    (r.energies.total, g, r.atom_charges.clone(), r.state.dip.clone())
                }
                Version::V1_1 => {
                    let setup = Setup::new(m)?;
                    let r = setup.scf(&opts)?;
                    let g = setup.gradient(&r)?;
                    let (dpat, _) = setup.atomic_moments(&crate::fock::total_density(&r.density));
                    (r.energies.total, g, r.atom_charges.clone(), dpat)
                }
            };
            let mut mu = [0.0; 3];
            for (a, p) in m.positions.iter().enumerate() {
                for k in 0..3 {
                    mu[k] += qa[a] * p[k] + dip[a][k];
                }
            }
            Ok((energy, g, mu))
        };
        let (energy, gradient, _) = point(mol)?;
        // every atom moved forward by the step and back by twice the step, as GFN2-xTB does
        let displaced = |x: &[[f64; 3]]| -> Result<(Vec<[f64; 3]>, [f64; 3])> {
            let mut m = mol.clone();
            m.positions.copy_from_slice(x);
            let (_, g, mu) = point(&m)?;
            Ok((g, mu))
        };
        let parallel = self.threads.parallel();
        let nh = self.threads.install(|| {
            fd::numerical_hessian(&mol.positions, step, fd::Displacement::ForwardBack, parallel, displaced)
        })?;
        Ok(axtb::Hessian {
            energy,
            gradient,
            hessian: nh.hessian,
            max_asymmetry: nh.max_asymmetry,
            dipole_derivatives: Some(nh.dipole_derivatives),
            method: HessianMethod::Numerical,
            timings: Vec::new(),
            parts: Vec::new(),
        })
    }

    fn response_hessian(
        &mut self,
        positions: &[[f64; 3]],
        opts: &HessianOptions,
        method: HessianMethod,
    ) -> Result<axtb::Hessian> {
        let step = opts.step.unwrap_or(RESPONSE_STEP);
        let accuracy = self.hessian_accuracy(opts.accuracy, ANALYTIC_ACCURACY)?;
        self.check_options()?;
        self.move_to(positions)?;
        // the reference SCF starts where a single point would
        let start = match self.take_start() {
            Some(Start::V2_0_1(s)) => Some(*s),
            _ => None,
        };
        // the response needs every orbital
        let sopts = ScfOptions { accuracy, ..self.options.run(true) };
        // the setup of this geometry, when the calculator holds one (move_to dropped it otherwise)
        let cached = match self.setup.take() {
            Some(AnySetup::V2(s)) => Some(*s),
            _ => None,
        };
        let mol = &self.mol;
        let run = || -> Result<(axtb::Hessian, v2::scf::Setup)> {
            let t0 = Instant::now();
            let mut setup = match cached {
                Some(s) => s,
                None => v2::scf::Setup::with_threads(mol, crate::v2::Model::v2_0_1(sopts.conventions), sopts.threads)?,
            };
            let ladder = RetryLadder::warm_then_fresh(start.is_some());
            let (r, _) = ladder.climb(|rung| match rung.fresh {
                false => setup.scf_from(&v2_options(&sopts, start.is_some()), start.as_ref()),
                true => setup.scf(&sopts),
            })?;
            let gradient = setup.gradient(&r)?;
            let mut timings = vec![("reference scf", t0.elapsed().as_secs_f64())];
            // the analytic Hessian's defaults: the symmetric response terms at tolerance 1e-6
            // (v2::hessian::AnalyticOptions); the response Hessian solves to 1e-10
            let (mut h, dipd, t) = if method == HessianMethod::Analytic {
                setup.analytic_hessian(&r, &v2::hessian::AnalyticOptions::default())
            } else {
                setup.response_hessian(&r, step, v2::response::CpksOptions::default())
            }?;
            timings.extend(t);
            let max_asymmetry = fd::symmetrize(&mut h, 3 * mol.nat());
            let hess = axtb::Hessian {
                energy: r.energies.total,
                gradient,
                hessian: h,
                max_asymmetry,
                dipole_derivatives: Some(dipd),
                method,
                timings,
                parts: Vec::new(),
            };
            Ok((hess, setup))
        };
        let (hess, setup) = self.threads.install(run)?;
        self.setup = Some(AnySetup::V2(Box::new(setup)));
        Ok(hess)
    }
}

/// The refusal of a Hessian method `version` does not have.
fn unavailable(m: HessianMethod, v: Version) -> Error {
    Error::Input(format!("{} Hessians: the {m:?} method is not available (use numerical)", v.method()))
}

impl Method for Calculator {
    fn info(&self) -> &MethodInfo {
        &self.info
    }

    /// v1.1 has numerical Hessians only.
    fn check(&self, _want: &Want, hessian: Option<&HessianOptions>) -> Result<()> {
        if let Some(h) = hessian {
            let m = h.method.unwrap_or(self.info.hessian.default);
            if self.info.hessian.get(m).is_none() {
                return Err(unavailable(m, self.version));
            }
        }
        Ok(())
    }

    /// Energy and what `want` asks for at `positions` (bohr). The SCF starts from the previous
    /// converged state when the restart policy reuses it (or from a given one); if that SCF
    /// does not converge, it runs once more from the program's start. The SCF report's `start`
    /// says which start gave the result.
    fn singlepoint(&mut self, positions: &[[f64; 3]], want: &Want) -> Result<SinglePoint> {
        let r = self.singlepoint_inner(positions, want);
        if r.is_err() {
            self.reset_wavefunction();
        }
        r
    }

    /// Hessian at `positions` (bohr) with the method and settings of `opts`:
    ///
    /// - `Analytic` (2.0.1, its default): analytic second derivatives with the coupled-perturbed
    ///   response ([`v2::analytic`]), SCF accuracy of the reference default [`ANALYTIC_ACCURACY`];
    /// - `Response` (2.0.1): central differences of the analytic gradient functional with the
    ///   density and energy-weighted density extrapolated by the coupled-perturbed response
    ///   ([`v2::hessian`]; no SCF per displacement), step default [`RESPONSE_STEP`], SCF accuracy
    ///   of the reference default [`ANALYTIC_ACCURACY`];
    /// - `Numerical` (both versions; v1.1's default): central differences of analytic gradients,
    ///   step default [`NUMERICAL_STEP`], the SCF of every displacement from the program's start
    ///   with the thresholds scaled by the accuracy (default [`NUMERICAL_ACCURACY`]).
    ///
    /// The accuracy is never looser than the calculator's own. Displacements run in parallel
    /// on the calculator's threads, each sequentially, so the result is the same for every
    /// thread count. Afterwards the calculator holds no wavefunction: the single point at these
    /// positions starts from the program's start, as without the Hessian.
    fn hessian(&mut self, positions: &[[f64; 3]], opts: &HessianOptions) -> Result<axtb::Hessian> {
        let r = self.hessian_inner(positions, opts);
        self.reset_wavefunction();
        r
    }

    fn wavefunction(&self) -> Option<Wavefunction> {
        let data = match self.state.as_ref()? {
            Start::V2_0_1(s) => ScfData {
                shell_charges: s.q.clone(),
                atomic_dipoles: s.dip.clone(),
                atomic_quadrupoles: s.qp.clone(),
                shell_magnetizations: s.mag.clone(),
                densities: s.p.clone(),
            },
            Start::V1_1(p) => ScfData { densities: p.clone(), ..ScfData::default() },
        };
        Some(Wavefunction::new(self.fingerprint.clone(), WavefunctionData::Scf(data)))
    }

    /// Start the next SCF from `wf` (another calculator's of this version and molecule): its
    /// densities (v1.1), with its moments and magnetizations (2.0.1), checked against the setup
    /// when the SCF starts.
    fn set_wavefunction(&mut self, wf: &Wavefunction) -> Result<()> {
        wf.check(&self.fingerprint)?;
        let WavefunctionData::Scf(d) = wf.data() else {
            return Err(Error::Input("the wavefunction belongs to another kind of calculation".into()));
        };
        if d.densities.is_empty() {
            return Err(Error::Input(
                "starting moments are GFN2-only; g-xTB starts from a whole wavefunction (the guess of another \
                 calculator)"
                    .into(),
            ));
        }
        self.state = Some(match self.version {
            Version::V2_0_1 => Start::V2_0_1(Box::new(v2::scf::State {
                p: d.densities.clone(),
                q: d.shell_charges.clone(),
                // recomputed from the shell charges when the SCF starts (`restart_state`)
                qa: Vec::new(),
                dip: d.atomic_dipoles.clone(),
                qp: d.atomic_quadrupoles.clone(),
                mag: d.shell_magnetizations.clone(),
            })),
            Version::V1_1 => Start::V1_1(d.densities.clone()),
        });
        self.given = true;
        Ok(())
    }

    fn reset_wavefunction(&mut self) {
        self.state = None;
        self.given = false;
    }

    fn scf_settings(&self) -> Option<&ScfSettings> {
        Some(&self.options.scf)
    }

    fn set_scf_settings(&mut self, settings: ScfSettings) -> Result<()> {
        let o = Options { scf: settings, ..self.options.clone() };
        o.check(self.version)?;
        self.options = o;
        Ok(())
    }
}

/// One single point inside the caller's pool, from `start` (`None`: the program's start);
/// returns the results with the converged state, and the setup for reuse.
#[allow(clippy::type_complexity)]
fn run(
    version: Version,
    mol: &Molecule,
    opts: &ScfOptions,
    setup: Option<AnySetup>,
    start: Option<&Start>,
    want: &Want,
    form: Option<QuadrupoleForm>,
) -> (Result<(SinglePoint, Start)>, Option<AnySetup>) {
    let t0 = Instant::now();
    let mut setup = match setup {
        Some(s) => s,
        None => match new_setup(version, mol, opts) {
            Ok(s) => s,
            Err(e) => return (Err(e), None),
        },
    };
    let t_setup = t0.elapsed().as_secs_f64();
    let t1 = Instant::now();
    let label = if start.is_some() { "previous" } else { "eeq-bc" };
    let form = form.unwrap_or_default();
    let res = match &mut setup {
        AnySetup::V1(setup) => {
            let p0 = match start {
                Some(Start::V1_1(p)) => Ok(Some(p.as_slice())),
                Some(_) => Err(Error::Input("a g-xTB 2.0.1 wavefunction cannot start a v1.1 SCF".into())),
                None => Ok(None),
            };
            p0.and_then(|p0| setup.scf_from(opts, p0)).and_then(|mut r| {
                let mut timings = vec![("setup", t_setup), ("scf", t1.elapsed().as_secs_f64())];
                timings.append(&mut r.timings);
                let g = if want.gradient {
                    let t2 = Instant::now();
                    let g = setup.gradient(&r)?;
                    timings.push(("gradient", t2.elapsed().as_secs_f64()));
                    Some(g)
                } else {
                    None
                };
                Ok(results(setup, r, g, timings, opts, want, label, form))
            })
        }
        AnySetup::V2(setup) => {
            let s0 = match start {
                Some(Start::V2_0_1(s)) => Ok(Some(s.as_ref())),
                Some(_) => Err(Error::Input("a g-xTB v1.1 wavefunction cannot start a 2.0.1 SCF".into())),
                None => Ok(None),
            };
            let vo = v2_options(opts, start.is_some());
            // the setup's phases (once: a reused setup reports none)
            let setup_laps = std::mem::take(&mut setup.timings);
            let t_setup =
                if setup_laps.is_empty() { t_setup } else { t_setup.max(setup_laps.iter().map(|x| x.1).sum()) };
            s0.and_then(|s0| setup.scf_from(&vo, s0)).and_then(|mut r| {
                let mut timings = vec![("setup", t_setup)];
                timings.extend(setup_laps);
                timings.push(("scf", t1.elapsed().as_secs_f64()));
                timings.append(&mut r.timings);
                let g = if want.gradient {
                    let t2 = Instant::now();
                    let (g, laps) = setup.gradient_timed(&r)?;
                    timings.push(("gradient", t2.elapsed().as_secs_f64()));
                    timings.extend(laps);
                    Some(g)
                } else {
                    None
                };
                Ok(results_v2(setup, r, g, timings, opts, want, label, form))
            })
        }
    };
    (res, Some(setup))
}

/// The gap rule of the conventions: the programs print the alpha channel's gap.
fn gap_rule(conventions: Conventions) -> GapRule {
    match conventions {
        Conventions::Gxtb => GapRule::Alpha,
        Conventions::Axtb => GapRule::Overall,
    }
}

/// Orbital levels per channel from the per-channel energies and occupations.
fn levels(energies: &[Vec<f64>], occupations: &[Vec<f64>]) -> Vec<OrbitalLevels> {
    energies
        .iter()
        .zip(occupations)
        .map(|(e, o)| OrbitalLevels { energies: e.clone(), occupations: o.clone() })
        .collect()
}

/// Mayer-Wiberg bond orders of the spin densities `p` (closed shell: the total density).
fn bond_orders(nat: usize, ao_atom: &[usize], s: &[f64], p: &[Vec<f64>], par: Par) -> Vec<f64> {
    let n = ao_atom.len();
    let s = faer::MatRef::from_row_major_slice(s, n, n);
    let p: Vec<faer::MatRef<'_, f64>> = p.iter().map(|x| faer::MatRef::from_row_major_slice(x, n, n)).collect();
    mayer_bond_orders(nat, ao_atom, s, &p, par)
}

/// The orbitals of a g-xTB SCF in the per-atom basis `shells`: one set (closed shell) or the
/// alpha and beta channels; `coefficients` row-major n x n with column k = orbital k.
fn collect_orbitals(
    mol: &Molecule,
    shells: &[Vec<axtb::basis::Cgto>],
    energies: &[Vec<f64>],
    occupations: &[Vec<f64>],
    coefficients: &[Vec<f64>],
) -> Orbitals {
    let n = energies[0].len();
    let sets = (0..energies.len())
        .map(|s| MoSet {
            spin: if s == 0 { Spin::Alpha } else { Spin::Beta },
            energies: energies[s].clone(),
            occupations: occupations[s].clone(),
            coefficients: (0..n).flat_map(|k| (0..n).map(move |mu| coefficients[s][mu * n + k])).collect(),
        })
        .collect();
    Orbitals { numbers: mol.numbers.clone(), positions: mol.positions.clone(), shells: shells.to_vec(), sets }
}

/// The dipole moment about the origin of atomic charges `q` and dipoles `d` at `positions`.
fn dipole(positions: &[[f64; 3]], q: &[f64], d: &[[f64; 3]]) -> [f64; 3] {
    let mut dipole = [0.0; 3];
    for (a, (q, d)) in q.iter().zip(d).enumerate() {
        for x in 0..3 {
            dipole[x] += q * positions[a][x] + d[x];
        }
    }
    dipole
}

#[allow(clippy::too_many_arguments)]
fn results(
    setup: &Setup,
    mut r: ScfResult,
    gradient: Option<Vec<[f64; 3]>>,
    timings: Timings,
    opts: &ScfOptions,
    want: &Want,
    start: &'static str,
    form: QuadrupoleForm,
) -> (SinglePoint, Start) {
    let orbitals = want.orbitals.then(|| {
        collect_orbitals(&setup.mol, &setup.basis.scf.cgto, &r.orbital_energies, &r.occupations, &r.coefficients)
    });
    let levels = levels(&r.orbital_energies, &r.occupations);
    let homo_lumo_gap = homo_lumo_gap(&levels, setup.nalpha as f64, setup.nbeta as f64, gap_rule(opts.conventions));
    let ao_atom = &setup.basis.scf.ao2at;
    let bond_orders = want.bond_orders.then(|| bond_orders(setup.mol.nat(), ao_atom, &setup.s, &r.density, setup.par));
    let ptot = crate::fock::total_density(&r.density);
    let (dpat, qpat) = setup.atomic_moments(&ptot);
    let positions = &setup.mol.positions;
    let sp = SinglePoint {
        energy: r.energies.total,
        terms: r.energies.terms(),
        gradient,
        dipole: dipole(positions, &r.atom_charges, &dpat),
        electronic: Some(Electronic {
            shell_charges: std::mem::take(&mut r.shell_charges),
            quadrupole: molecular_quadrupole(positions, &r.atom_charges, &dpat, &qpat, form),
            atomic_dipoles: dpat,
            atomic_quadrupoles: qpat,
            levels,
            homo_lumo_gap,
            orbitals,
            bond_orders,
            scf: ScfReport {
                iterations: r.iterations,
                start,
                mixer_used: None,
                attempts: Vec::new(),
                eigensolver: r.eigensolver.clone(),
            },
        }),
        charges: std::mem::take(&mut r.atom_charges),
        extras: Extras::default(),
        timings,
    };
    (sp, Start::V1_1(std::mem::take(&mut r.density)))
}

#[allow(clippy::too_many_arguments)]
fn results_v2(
    setup: &v2::scf::Setup,
    mut r: v2::scf::ScfResult,
    gradient: Option<Vec<[f64; 3]>>,
    timings: Timings,
    opts: &ScfOptions,
    want: &Want,
    start: &'static str,
    form: QuadrupoleForm,
) -> (SinglePoint, Start) {
    let orbitals = want.orbitals.then(|| {
        collect_orbitals(&setup.mol, &setup.basis.scf.cgto, &r.orbital_energies, &r.occupations, &r.coefficients)
    });
    let levels = levels(&r.orbital_energies, &r.occupations);
    let homo_lumo_gap = homo_lumo_gap(&levels, setup.nalpha, setup.nbeta, gap_rule(opts.conventions));
    let bond_orders =
        want.bond_orders.then(|| bond_orders(setup.mol.nat(), &setup.basis.scf.ao2at, &setup.s, &r.state.p, setup.par));
    let (dpat, qpat) = (r.state.dip.clone(), r.state.qp.clone());
    let positions = &setup.mol.positions;
    let sp = SinglePoint {
        energy: r.energies.total,
        terms: r.energies.terms(),
        gradient,
        dipole: dipole(positions, &r.atom_charges, &dpat),
        electronic: Some(Electronic {
            shell_charges: std::mem::take(&mut r.shell_charges),
            quadrupole: molecular_quadrupole(positions, &r.atom_charges, &dpat, &qpat, form),
            atomic_dipoles: dpat,
            atomic_quadrupoles: qpat,
            levels,
            homo_lumo_gap,
            orbitals,
            bond_orders,
            scf: ScfReport {
                iterations: r.iterations,
                start,
                mixer_used: None,
                attempts: Vec::new(),
                eigensolver: r.eigensolver.clone(),
            },
        }),
        charges: std::mem::take(&mut r.atom_charges),
        extras: Extras::default(),
        timings,
    };
    (sp, Start::V2_0_1(Box::new(r.state)))
}
