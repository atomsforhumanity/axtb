//! Calculator API of g-xTB (2.0.1 and v1.1), shaped like `axtb::Calculator` and returning
//! `axtb::SinglePoint`, so that the axtb command-line driver, Python package and qccodec codec
//! select it by method name (`"gxtb-v2.0.1"`, `"gxtb-v1.1"`).
//!
//! g-xTB lives in this GPL-3.0 crate, which builds on axtb (LGPL-3.0); `axtb::Calculator`
//! cannot construct it without making the core depend on GPL code, hence the separate type.
//!
//! Energies, analytic gradients (2.0.1: [`crate::v2::gradient`]; v1.1: [`crate::response`])
//! and Hessians ([`Calculator::hessian`]: for 2.0.1 analytic (the default) or response,
//! [`crate::v2::hessian`]; for both versions numerical, central differences of analytic
//! gradients as `axtb::Calculator::hessian`), with the settings of [`ScfOptions`] (refused when
//! out of range or meant for the other version, [`ScfOptions::check`]). Where an SCF starts is
//! [`ScfOptions::guess`]: in the default conventions from the previous geometry's converged
//! state ([`GuessState`]; the programs' restart files `xtbrestart` and `gxtbrestart` carry the
//! same), in the program conventions from the program's EEQ-BC start, so that every result is
//! that of a fresh program run.
//!
//! Threading (as in axtb): with `threads > 1` the calculator owns one rayon pool of exactly
//! that size (`axtb_base::threads::Threads`) and runs everything inside it; with `threads = 1`
//! no thread is spawned and faer's global parallelism is never touched.

use crate::scf::{Conventions, Energies, Guess, ScfOptions, ScfResult, Setup};
use crate::timing::Timings;
use crate::v2;
use axtb::molden::{MoSet, Orbitals, Spin};
use axtb::properties::{homo_lumo_gap, mayer_bond_orders, GapRule, OrbitalLevels};
use axtb::{EnergyTerms, Error, HessianMethod, HessianOptions, Molecule, Result, SinglePoint};
use axtb_base::fd;
use axtb_base::threads::Threads;
use faer::Par;
use std::time::Instant;

/// The g-xTB versions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Version {
    /// g-xTB 2.0.1 (the grimme-lab/g-xtb 2.0.1 release, `xtb --gxtb`), the default.
    #[default]
    V2_0_1,
    /// g-xTB v1.1 (public preprint version, the 1.1.0 program).
    V1_1,
}

impl Version {
    /// The version a method name selects: `gxtb-v2.0.1` (also `gxtb-2.0.1`, `gxtbv201`,
    /// `g-xTB 2.0`) or `gxtb-v1.1` (also `gxtb_v1.1`, `gxtbv11`, `g-xTB 1.1`), case-insensitive.
    pub fn from_method(name: &str) -> Option<Self> {
        let key: String = name.to_ascii_lowercase().chars().filter(|c| c.is_ascii_alphanumeric()).collect();
        match key.as_str() {
            "gxtbv201" | "gxtb201" | "gxtbv20" | "gxtb20" => Some(Self::V2_0_1),
            "gxtbv11" | "gxtb11" => Some(Self::V1_1),
            _ => None,
        }
    }

    /// The method name: `"gxtb-v2.0.1"` or `"gxtb-v1.1"`.
    pub fn method(&self) -> &'static str {
        match self {
            Self::V2_0_1 => "gxtb-v2.0.1",
            Self::V1_1 => "gxtb-v1.1",
        }
    }
}

impl v2::scf::Energies {
    /// The terms with their names as the axtb front ends report them for g-xTB 2.0.1.
    pub fn named(&self) -> Vec<(&'static str, f64)> {
        vec![
            ("one_electron", self.one_electron),
            ("first_order", self.first_order),
            ("second_order", self.second_order),
            ("third_order", self.third_order),
            ("fourth_order", self.fourth_order),
            ("es_multipole", self.aes),
            ("exchange", self.exchange),
            ("spin_polarization", self.spin),
            ("dispersion", self.dispersion),
            ("repulsion", self.repulsion),
            ("entropy", self.entropy),
            ("electronic", self.electronic),
            ("increments", self.increments),
            ("dispersion_three_body", self.dispersion_three_body),
        ]
    }
}

/// The energy terms of either version. Each version reports its program's own decomposition
/// ([`Terms::named`]), so the names differ: 2.0.1 has `first_order` to `fourth_order` and one
/// `exchange` (Mulliken, bond-order and onsite exchange), as tblite's containers; v1.1 has
/// `es1` (first order) and `es2_3` (second to fourth order) and splits the exchange into
/// `exchange_mulliken` and `exchange_onsite`, as the 1.1.0 program prints them.
#[derive(Clone, Debug)]
pub enum Terms {
    V2_0_1(v2::scf::Energies),
    V1_1(Energies),
}

impl Terms {
    /// The terms with the names the axtb front ends report.
    pub fn named(&self) -> Vec<(&'static str, f64)> {
        match self {
            Self::V2_0_1(e) => e.named(),
            Self::V1_1(e) => e.named(),
        }
    }

    /// Total energy.
    pub fn total(&self) -> f64 {
        match self {
            Self::V2_0_1(e) => e.total,
            Self::V1_1(e) => e.total,
        }
    }
}

/// Results of a g-xTB single point: the method-independent part as `axtb::SinglePoint` and
/// all g-xTB energy terms.
#[derive(Clone, Debug)]
pub struct Results {
    /// `terms` maps the g-xTB terms onto axtb's GFN2-shaped fields: `repulsion`, `dispersion`
    /// (revD4 two- plus three-body), `electronic` (everything else, incl. the atomic
    /// increments, so that energy = repulsion + dispersion + electronic), `isotropic_es`
    /// (first to fourth order), `anisotropic` (AES). Open shells: orbital energies and
    /// occupations of the alpha orbitals followed by the beta orbitals.
    pub singlepoint: SinglePoint,
    pub energies: Terms,
}

impl Energies {
    /// The terms with their names as the axtb front ends report them for g-xTB v1.1.
    pub fn named(&self) -> Vec<(&'static str, f64)> {
        vec![
            ("one_electron", self.one_electron),
            ("es1", self.es1),
            ("es2_3", self.es23),
            ("es_multipole", self.aes),
            ("exchange_mulliken", self.mfx),
            ("exchange_onsite", self.ofx),
            ("spin_polarization", self.spin),
            ("entropy", self.entropy),
            ("electronic", self.electronic),
            ("increments", self.increments),
            ("dispersion", self.disp2),
            ("dispersion_three_body", self.disp3),
            ("repulsion", self.repulsion),
        ]
    }
}

/// The geometry-dependent setup of either version.
#[derive(Clone, Debug)]
enum AnySetup {
    V2(Box<v2::scf::Setup>),
    V1(Box<Setup>),
}

/// The converged SCF state a later SCF of the same molecule can start from
/// ([`ScfOptions::guess`]), as the programs' restart files carry it.
#[derive(Clone, Debug)]
pub enum GuessState {
    /// 2.0.1: shell charges, atomic dipoles and quadrupoles, shell magnetizations and spin
    /// densities, as `xtbrestart`.
    V2_0_1(Box<v2::scf::State>),
    /// v1.1: the spin densities (closed shell: the total density), as `gxtbrestart`.
    V1_1(Vec<Vec<f64>>),
}

impl GuessState {
    /// The version whose SCF the state belongs to.
    pub fn version(&self) -> Version {
        match self {
            GuessState::V2_0_1(_) => Version::V2_0_1,
            GuessState::V1_1(_) => Version::V1_1,
        }
    }
}

/// g-xTB bound to one molecule (atomic numbers, charge, unpaired electrons).
pub struct Calculator {
    mol: Molecule,
    /// Settings; checked by every calculation ([`ScfOptions::check`]). `threads` must stay the
    /// number the calculator was built with.
    pub options: ScfOptions,
    version: Version,
    threads: Threads,
    /// Setup of the current geometry (built in `new`, reused while the positions are equal).
    setup: Option<AnySetup>,
    /// The last converged state (or one given with [`Calculator::set_guess`]).
    state: Option<GuessState>,
    /// Start the next SCF from `state` even if the options do not reuse it (it was given).
    given: bool,
}

/// The setup of `mol` for `version` in the conventions of `opts`, on `opts.threads` threads
/// (inside the caller's pool).
fn new_setup(version: Version, mol: &Molecule, opts: &ScfOptions) -> Result<AnySetup> {
    Ok(match version {
        Version::V2_0_1 => AnySetup::V2(Box::new(v2::scf::Setup::with_threads(mol, opts.conventions, opts.threads)?)),
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
        options: ScfOptions,
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
        options: ScfOptions,
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
        let setup = threads.install(|| new_setup(version, &mol, &options))?;
        Ok(Self { mol, options, version, threads, setup: Some(setup), state: None, given: false })
    }

    /// The g-xTB version.
    pub fn version(&self) -> Version {
        self.version
    }

    /// The molecule (positions of the last evaluation).
    pub fn molecule(&self) -> &Molecule {
        &self.mol
    }

    /// Whether the SCFs start from the previous converged state ([`ScfOptions::guess`]).
    pub fn reuses_guess(&self) -> bool {
        self.options.reuses_guess()
    }

    /// How the molecular quadrupole is printed in these conventions: the 2.0.1 program's
    /// (`xtb --gxtb --quadrupole`, tblite's form) in its conventions, the traceless tensor
    /// otherwise (the 1.1.0 program prints none).
    pub fn quadrupole_form(&self) -> axtb::QuadrupoleForm {
        match (self.version, self.options.conventions) {
            (Version::V2_0_1, Conventions::Gxtb) => axtb::QuadrupoleForm::Tblite,
            _ => axtb::QuadrupoleForm::Traceless,
        }
    }

    /// Forget the previous converged state: the next SCF starts from the program's start.
    pub fn reset_guess(&mut self) {
        self.state = None;
        self.given = false;
    }

    /// The last converged state (the start of the next SCF when the calculator reuses it).
    pub fn guess(&self) -> Option<&GuessState> {
        self.state.as_ref()
    }

    /// Start the next SCF from `state` (e.g. the [`Calculator::guess`] of another calculator for
    /// the same molecule), whatever [`ScfOptions::guess`] says. It must belong to this version
    /// and molecule (checked when the SCF starts).
    pub fn set_guess(&mut self, state: GuessState) -> Result<()> {
        if state.version() != self.version {
            return Err(Error::Input(format!(
                "a {} wavefunction cannot start a {} SCF",
                state.version().method(),
                self.version.method()
            )));
        }
        self.state = Some(state);
        self.given = true;
        Ok(())
    }

    /// Check the options of the next calculation.
    fn check_options(&self) -> Result<()> {
        self.options.check(self.version)?;
        let threads = self.threads.count();
        if self.options.threads != threads {
            return Err(Error::Input(format!(
                "threads changed from {threads} to {} after the calculator was built",
                self.options.threads
            )));
        }
        Ok(())
    }

    /// The state the next SCF starts from (`None`: the program's start), taken: a calculation
    /// that fails leaves no state behind.
    fn take_start(&mut self) -> Option<GuessState> {
        let start = if self.given || self.options.reuses_guess() { self.state.take() } else { None };
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

    /// Energy (and with `gradient` the analytic gradient, Eh/bohr) at `positions` (bohr).
    ///
    /// The SCF starts from the previous converged state when the calculator reuses it
    /// ([`ScfOptions::guess`]); if that SCF does not converge, it runs once more from the
    /// program's start. `SinglePoint::guess` says which start gave the result.
    pub fn singlepoint(&mut self, positions: &[[f64; 3]], gradient: bool) -> Result<Results> {
        self.check_options()?;
        self.move_to(positions)?;
        let start = self.take_start();
        let mut setup = self.setup.take();
        let (mol, opts, version) = (&self.mol, &self.options, self.version);
        let go = |setup: Option<AnySetup>, start: Option<&GuessState>| {
            self.threads.install(|| run(version, mol, opts, setup, start, gradient))
        };
        let (mut res, s) = go(setup.take(), start.as_ref());
        setup = s;
        if start.is_some() && matches!(res, Err(Error::ScfNotConverged { .. })) {
            // a state from a distant geometry can hurt; once more from the program's start
            let (r, s) = go(setup.take(), None);
            res = r;
            setup = s;
        }
        self.setup = setup;
        let (r, state) = res?;
        self.state = Some(state);
        Ok(r)
    }

    /// The default Hessian method of this version: analytic for 2.0.1, numerical for v1.1.
    pub fn default_hessian_method(&self) -> HessianMethod {
        match self.version {
            Version::V2_0_1 => HessianMethod::Analytic,
            Version::V1_1 => HessianMethod::Numerical,
        }
    }

    /// Hessian at `positions` (bohr) with the method and settings of `opts`:
    ///
    /// - `Analytic` (2.0.1, its default): analytic second derivatives with the coupled-perturbed
    ///   response ([`v2::analytic`]), SCF accuracy of the reference default 1e-5 (1e-11 Eh,
    ///   residual 2e-10: the error of both 2.0.1 methods is first order in the reference
    ///   residual, and 1e-3 leaves ~2e-7 Eh/bohr^2 on caffeine for no measurable saving);
    /// - `Response` (2.0.1): central differences of the analytic gradient functional with the
    ///   density and energy-weighted density extrapolated by the coupled-perturbed response
    ///   ([`v2::hessian`]; no SCF per displacement), step default 2e-4 bohr, SCF accuracy of
    ///   the reference default 1e-5;
    /// - `Numerical` (both versions; v1.1's default): central differences of analytic gradients
    ///   as `axtb::Calculator::hessian`, step default 0.005 bohr, the SCF of every displacement
    ///   from the program's start with the thresholds scaled by the accuracy (default 0.01).
    ///
    /// The accuracy is never looser than the calculator's own. Displacements run in parallel
    /// on the calculator's threads, each sequentially, so the result is the same for every
    /// thread count.
    pub fn hessian(&mut self, positions: &[[f64; 3]], opts: &HessianOptions) -> Result<axtb::Hessian> {
        let t0 = Instant::now();
        let mut out = match (opts.method, self.version) {
            (HessianMethod::Numerical, _) => self.numerical(positions, opts.step.unwrap_or(0.005), opts.accuracy),
            (m @ (HessianMethod::Response | HessianMethod::Analytic), Version::V2_0_1) => {
                self.response_hessian(positions, opts, m)
            }
            (m, v) => {
                Err(Error::Input(format!("{} Hessians: the {m:?} method is not available (use numerical)", v.method())))
            }
        }?;
        out.timings.push(("total", t0.elapsed().as_secs_f64()));
        Ok(out)
    }

    /// Numerical Hessian with step `step` (bohr) and SCF accuracy `accuracy` of every
    /// displacement (default 0.01): [`Calculator::hessian`] with [`HessianMethod::Numerical`].
    pub fn numerical_hessian(
        &mut self,
        positions: &[[f64; 3]],
        step: f64,
        accuracy: Option<f64>,
    ) -> Result<axtb::Hessian> {
        self.hessian(positions, &HessianOptions { method: HessianMethod::Numerical, step: Some(step), accuracy })
    }

    /// The SCF accuracy of a Hessian: `accuracy` (default `default`), never looser than the
    /// calculator's own.
    fn hessian_accuracy(&self, accuracy: Option<f64>, default: f64) -> Result<f64> {
        let accuracy = accuracy.unwrap_or(default);
        if !(accuracy > 0.0 && accuracy.is_finite()) {
            return Err(Error::Input(format!("accuracy must be positive and finite, got {accuracy}")));
        }
        Ok(accuracy.min(self.options.accuracy))
    }

    fn numerical(&mut self, positions: &[[f64; 3]], step: f64, accuracy: Option<f64>) -> Result<axtb::Hessian> {
        if !(step > 0.0 && step.is_finite()) {
            return Err(Error::Input(format!("Hessian step must be positive and finite, got {step}")));
        }
        let accuracy = self.hessian_accuracy(accuracy, 0.01)?;
        self.check_options()?;
        self.move_to(positions)?;
        let opts =
            ScfOptions { accuracy, threads: 1, keep_orbitals: false, bond_orders: false, ..self.options.clone() };
        let mol = &self.mol;
        let version = self.version;
        // energy, gradient and dipole of one geometry
        let point = |m: &Molecule| -> Result<(f64, Vec<[f64; 3]>, [f64; 3])> {
            let (energy, g, qa, dip) = match version {
                Version::V2_0_1 => {
                    let setup = v2::scf::Setup::new(m, opts.conventions)?;
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
        // every atom moved forward by the step and back by twice the step, as axtb::Calculator does
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
        })
    }

    fn response_hessian(
        &mut self,
        positions: &[[f64; 3]],
        opts: &HessianOptions,
        method: HessianMethod,
    ) -> Result<axtb::Hessian> {
        let step = opts.step.unwrap_or(2e-4);
        let accuracy = self.hessian_accuracy(opts.accuracy, 1e-5)?;
        self.check_options()?;
        self.move_to(positions)?;
        // the reference SCF starts where a single point would
        let start = match self.take_start() {
            Some(GuessState::V2_0_1(s)) => Some(*s),
            _ => None,
        };
        // the response needs every orbital
        let sopts = ScfOptions { accuracy, keep_orbitals: true, ..self.options.clone() };
        // the setup of this geometry, when the calculator holds one (move_to dropped it otherwise)
        let cached = match self.setup.take() {
            Some(AnySetup::V2(s)) => Some(*s),
            _ => None,
        };
        let mol = &self.mol;
        let run = || -> Result<(axtb::Hessian, v2::scf::Setup, GuessState)> {
            let t0 = Instant::now();
            let mut setup = match cached {
                Some(s) => s,
                None => v2::scf::Setup::with_threads(mol, sopts.conventions, sopts.threads)?,
            };
            let r = match setup.scf_from(&v2_options(&sopts, start.is_some()), start.as_ref()) {
                Err(Error::ScfNotConverged { .. }) if start.is_some() => setup.scf(&sopts)?,
                other => other?,
            };
            let state = GuessState::V2_0_1(Box::new(r.state.clone()));
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
            };
            Ok((hess, setup, state))
        };
        let (hess, setup, state) = self.threads.install(run)?;
        self.setup = Some(AnySetup::V2(Box::new(setup)));
        self.state = Some(state);
        Ok(hess)
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
    start: Option<&GuessState>,
    gradient: bool,
) -> (Result<(Results, GuessState)>, Option<AnySetup>) {
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
    let label = if start.is_some() { Guess::Previous.name() } else { Guess::EeqBc.name() };
    let res = match &mut setup {
        AnySetup::V1(setup) => {
            let p0 = match start {
                Some(GuessState::V1_1(p)) => Ok(Some(p.as_slice())),
                Some(_) => Err(Error::Input("a g-xTB 2.0.1 wavefunction cannot start a v1.1 SCF".into())),
                None => Ok(None),
            };
            p0.and_then(|p0| setup.scf_from(opts, p0)).and_then(|mut r| {
                let mut timings = vec![("setup", t_setup), ("scf", t1.elapsed().as_secs_f64())];
                timings.append(&mut r.timings);
                let g = if gradient {
                    let t2 = Instant::now();
                    let g = setup.gradient(&r)?;
                    timings.push(("gradient", t2.elapsed().as_secs_f64()));
                    Some(g)
                } else {
                    None
                };
                Ok(results(setup, r, g, timings, opts, label))
            })
        }
        AnySetup::V2(setup) => {
            let s0 = match start {
                Some(GuessState::V2_0_1(s)) => Ok(Some(s.as_ref())),
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
                let g = if gradient {
                    let t2 = Instant::now();
                    let (g, laps) = setup.gradient_timed(&r)?;
                    timings.push(("gradient", t2.elapsed().as_secs_f64()));
                    timings.extend(laps);
                    Some(g)
                } else {
                    None
                };
                Ok(results_v2(setup, r, g, timings, opts, label))
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

fn results(
    setup: &Setup,
    mut r: ScfResult,
    gradient: Option<Vec<[f64; 3]>>,
    timings: Timings,
    opts: &ScfOptions,
    guess: &'static str,
) -> (Results, GuessState) {
    let e = &r.energies;
    let orbitals = opts.keep_orbitals.then(|| {
        collect_orbitals(&setup.mol, &setup.basis.scf.cgto, &r.orbital_energies, &r.occupations, &r.coefficients)
    });
    let levels = levels(&r.orbital_energies, &r.occupations);
    let homo_lumo_gap = homo_lumo_gap(&levels, setup.nalpha as f64, setup.nbeta as f64, gap_rule(opts.conventions));
    let ao_atom = &setup.basis.scf.ao2at;
    let bond_orders = opts.bond_orders.then(|| bond_orders(setup.mol.nat(), ao_atom, &setup.s, &r.density, setup.par));
    let ptot = crate::fock::total_density(&r.density);
    let (dpat, qpat) = setup.atomic_moments(&ptot);
    let mut dipole = [0.0; 3];
    for (a, (q, d)) in r.atom_charges.iter().zip(&dpat).enumerate() {
        for x in 0..3 {
            dipole[x] += q * setup.mol.positions[a][x] + d[x];
        }
    }
    let terms = EnergyTerms {
        repulsion: e.repulsion,
        dispersion: e.disp2 + e.disp3,
        electronic: e.electronic + e.increments,
        isotropic_es: e.es1 + e.es23,
        anisotropic: e.aes,
        entropy: e.entropy,
        ..Default::default()
    };
    let singlepoint = SinglePoint {
        energy: e.total,
        terms,
        gradient,
        qat: r.atom_charges.clone(),
        qsh: r.shell_charges.clone(),
        dpat,
        qpat,
        dipole,
        orbital_energies: r.orbital_energies.concat(),
        occupations: r.occupations.concat(),
        iterations: r.iterations,
        mixer_used: None,
        scf_attempts: Vec::new(),
        timings,
        orbitals,
        levels,
        homo_lumo_gap,
        bond_orders,
        guess,
        eigensolver: r.eigensolver.clone(),
    };
    let state = GuessState::V1_1(std::mem::take(&mut r.density));
    (Results { singlepoint, energies: Terms::V1_1(r.energies) }, state)
}

fn results_v2(
    setup: &v2::scf::Setup,
    r: v2::scf::ScfResult,
    gradient: Option<Vec<[f64; 3]>>,
    timings: Timings,
    opts: &ScfOptions,
    guess: &'static str,
) -> (Results, GuessState) {
    let e = &r.energies;
    let orbitals = opts.keep_orbitals.then(|| {
        collect_orbitals(&setup.mol, &setup.basis.scf.cgto, &r.orbital_energies, &r.occupations, &r.coefficients)
    });
    let levels = levels(&r.orbital_energies, &r.occupations);
    let homo_lumo_gap = homo_lumo_gap(&levels, setup.nalpha, setup.nbeta, gap_rule(opts.conventions));
    let bond_orders =
        opts.bond_orders.then(|| bond_orders(setup.mol.nat(), &setup.basis.scf.ao2at, &setup.s, &r.state.p, setup.par));
    let (dpat, qpat) = (r.state.dip.clone(), r.state.qp.clone());
    let mut dipole = [0.0; 3];
    for (a, (q, d)) in r.atom_charges.iter().zip(&dpat).enumerate() {
        for x in 0..3 {
            dipole[x] += q * setup.mol.positions[a][x] + d[x];
        }
    }
    let terms = EnergyTerms {
        repulsion: e.repulsion,
        dispersion: e.dispersion + e.dispersion_three_body,
        electronic: e.total - e.repulsion - e.dispersion - e.dispersion_three_body,
        isotropic_es: e.first_order + e.second_order + e.third_order + e.fourth_order,
        anisotropic: e.aes,
        entropy: e.entropy,
        ..Default::default()
    };
    // closed shell: one channel with doubled occupations
    let singlepoint = SinglePoint {
        energy: e.total,
        terms,
        gradient,
        qat: r.atom_charges.clone(),
        qsh: r.shell_charges.clone(),
        dpat,
        qpat,
        dipole,
        orbital_energies: r.orbital_energies.concat(),
        occupations: r.occupations.concat(),
        iterations: r.iterations,
        mixer_used: None,
        scf_attempts: Vec::new(),
        timings,
        orbitals,
        levels,
        homo_lumo_gap,
        bond_orders,
        guess,
        eigensolver: r.eigensolver.clone(),
    };
    (Results { singlepoint, energies: Terms::V2_0_1(r.energies) }, GuessState::V2_0_1(Box::new(r.state)))
}
