//! g-xTB in implicit solvent: a solvation composite, in one process.
//!
//! g-xTB has no implicit-solvent parametrization (the g-xTB 2.0.1 program stops with
//! `--alpb`/`--gbsa`). The composite puts it in solvent by adding GFN2-xTB's solvation free
//! energy:
//!
//! ```text
//! E_solv(x) = E_gxtb(x) + [E_GFN2+solv(x) - E_GFN2(x)]
//! ```
//!
//! and the same for gradients and Hessians, so optimizations, TS searches and IRCs run on the
//! solvated surface. ΔG_solv(x) = E_GFN2+solv(x) - E_GFN2(x) is the difference of two GFN2-xTB
//! calculations on the same geometry with the same settings, one with ALPB or GBSA. The
//! motivating workflow, a reaction-path search, ran the three parts as three xtb program calls per
//! gradient; [`CompositeCalculator`] holds the three calculators for one molecule, keeps each
//! part's setup between geometries (and, in axtb conventions, starts each SCF from the previous
//! geometry's converged state) and runs the parts one after the other on one thread pool, so
//! its results do not depend on the thread count beyond the last bit.
//!
//! - **The GFN2 halves.** xtb's GFN2-xTB defaults (accuracy 1, 300 K, 250 iterations, D4), the
//!   charge and unpaired electrons of the molecule, and the solvent model's default reference
//!   state `gsolv` (1 M in the gas phase to 1 M in solution). Both halves always run with the
//!   same settings; only the solvent model differs.
//! - **Electronic temperatures.** By default g-xTB runs at its program's default of 0 K and
//!   the GFN2 halves at xtb's default of 300 K ([`CompositeOptions::etemp`] `None`); an explicit
//!   temperature applies to all three parts.
//! - **The SCF retry ladders** ([`PAIR_RETRIES`], [`GXTB_RETRIES`]). A GFN2 pair whose SCF fails
//!   (no convergence, or a numerical breakdown) is run again, both halves at the same settings,
//!   with 1000 K and 1000 iterations, then 1000 iterations, then 5000 K and 1000 iterations. A
//!   g-xTB SCF that fails is run again at 1000 K, then 3000 K, each with 1000 iterations,
//!   unless the temperature was set explicitly. The rung that converged is reported per part
//!   ([`PartInfo::retry`]), so a result from a smeared rung is never mistaken for a 0 K one.
//! - **The zero check.** The first geometry's ΔG_solv must not be zero (|ΔG| < 1e-9 Eh), or the
//!   calculation stops instead of reporting gas-phase energies as solvated (e.g. a solvent
//!   model that did not apply).
//!
//! Conventions: [`CompositeOptions::conventions`] are the g-xTB part's (as for g-xTB alone);
//! the GFN2 halves follow [`CompositeOptions::correction_conventions`], by default `axtb` for
//! `axtb` (corrected solvation) and `xtb-6.7.1` for `gxtb`. `conventions = Gxtb` therefore
//! reproduces the three programs `xtb --gxtb`, `xtb --alpb water` and `xtb` of the g-xTB 2.0.1
//! build of xtb 6.7.1: every SCF starts from the program's guess, as each program call does.
use crate::calculator::{Calculator, Options, Version};
use crate::scf::Conventions;
use axtb::method::SinglePoint;
use axtb::method::{Attempt, Method as _, Restart, ScfSettings, Term, Want, Wavefunction};
use axtb::{Error, HessianMethod, HessianOptions, MixerKind, Result, SolvationModel, SolvationSpec};
use axtb_base::threads::{self, Threads};
use std::time::Instant;

/// The method that supplies the solvation correction.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Correction {
    /// GFN2-xTB with ALPB or GBSA: ΔG_solv = E_GFN2+solv - E_GFN2.
    #[default]
    Gfn2,
}

impl Correction {
    /// Name as the front ends accept it.
    pub fn name(self) -> &'static str {
        match self {
            Correction::Gfn2 => "gfn2",
        }
    }
}

impl std::str::FromStr for Correction {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        let key: String = s.to_ascii_lowercase().chars().filter(|c| c.is_ascii_alphanumeric()).collect();
        match key.as_str() {
            "gfn2" | "gfn2xtb" => Ok(Correction::Gfn2),
            _ => Err(Error::Input(format!(
                "unknown solvation correction {s:?}; the only one is \"gfn2\" (GFN2-xTB with ALPB/GBSA)"
            ))),
        }
    }
}

/// Settings of a composite calculation. `Default`: g-xTB 2.0.1 with GFN2-xTB + ALPB(water),
/// each program's default SCF settings, the retry ladders, axtb conventions.
#[derive(Clone, Debug)]
pub struct CompositeOptions {
    /// g-xTB version of the gas-phase energy.
    pub version: Version,
    /// Method of the solvation correction.
    pub correction: Correction,
    /// Solvent model, solvent and reference state of the solvated GFN2 half.
    pub solvation: SolvationSpec,
    /// SCF accuracy of all three parts (xtb's `--acc`: multiplier of the convergence
    /// thresholds; default 1).
    pub accuracy: f64,
    /// Electronic temperature (K) of all three parts; `None`: each method's default, g-xTB 0 K
    /// and GFN2-xTB 300 K, the programs' defaults.
    pub etemp: Option<f64>,
    /// SCF iteration limit of all three parts; `None`: each method's default (250 for g-xTB
    /// 2.0.1 and GFN2-xTB).
    pub max_iter: Option<usize>,
    /// Threads of the composite's one pool; the parts run one after the other inside it.
    pub threads: usize,
    /// Conventions of the g-xTB part (`Axtb` or `Gxtb`, the program's).
    pub conventions: Conventions,
    /// Conventions of the GFN2 halves; `None`: `Axtb` for `conventions = Axtb`, `Xtb671` for
    /// `Gxtb` (the g-xTB 2.0.1 build of xtb is xtb 6.7.1 for GFN2-xTB).
    pub correction_conventions: Option<axtb::Conventions>,
    /// SCF mixer of the GFN2 halves; `None`: their conventions' choice.
    pub mixer: Option<MixerKind>,
    /// Retry failed SCFs with the ladders [`PAIR_RETRIES`] and [`GXTB_RETRIES`] (default true).
    pub retries: bool,
    /// Start each SCF from the previous geometry's converged state; `None`: each part as its
    /// conventions say, the GFN2 halves in their axtb conventions and the g-xTB part in its axtb
    /// conventions (the programs start every call from their own guess).
    pub reuse_guess: Option<bool>,
}

impl Default for CompositeOptions {
    fn default() -> Self {
        Self {
            version: Version::V2_0_1,
            correction: Correction::Gfn2,
            solvation: SolvationSpec {
                model: SolvationModel::Alpb,
                solvent: "water".into(),
                state: Default::default(),
            },
            accuracy: 1.0,
            etemp: None,
            max_iter: None,
            threads: 1,
            conventions: Conventions::Axtb,
            correction_conventions: None,
            mixer: None,
            retries: true,
            reuse_guess: None,
        }
    }
}

impl CompositeOptions {
    /// Conventions of the GFN2 halves (explicit, or following the g-xTB part's).
    pub fn gfn2_conventions(&self) -> axtb::Conventions {
        self.correction_conventions.unwrap_or(match self.conventions {
            Conventions::Axtb => axtb::Conventions::Axtb,
            Conventions::Gxtb => axtb::Conventions::Xtb671,
        })
    }
}

/// Default electronic temperature (K) of the GFN2 halves (xtb's GFN2-xTB default).
pub const GFN2_ETEMP: f64 = 300.0;
/// Default electronic temperature (K) of g-xTB (the 2.0.1 program's default).
pub const GXTB_ETEMP: f64 = 0.0;

/// Retry ladder of a GFN2 pair: electronic temperature (`None`: unchanged) and iteration
/// limit of each try after the first, both halves at the same rung.
pub const PAIR_RETRIES: [(Option<f64>, usize); 3] = [(Some(1000.0), 1000), (None, 1000), (Some(5000.0), 1000)];
/// Retry ladder of g-xTB (electronic temperature, iteration limit), only without an explicit
/// electronic temperature.
pub const GXTB_RETRIES: [(f64, usize); 2] = [(1000.0, 1000), (3000.0, 1000)];

/// How one part of the composite converged.
#[derive(Clone, Debug)]
pub struct PartInfo {
    /// `"gxtb-v2.0.1"` (or v1.1), `"gfn2+alpb"`/`"gfn2+gbsa"`, `"gfn2"`.
    pub method: String,
    /// Conventions of the part.
    pub conventions: &'static str,
    /// Energy of the part (Eh).
    pub energy: f64,
    /// SCF iterations of the converging attempt.
    pub iterations: usize,
    /// GFN2-xTB: mixer of the converging attempt (`"broyden"`, `"diis"`, `"diis-fallback"`);
    /// `None` for g-xTB.
    pub mixer_used: Option<&'static str>,
    /// GFN2-xTB: every attempt of the converging rung.
    pub scf_attempts: Vec<Attempt>,
    /// Electronic temperature (K) and iteration limit of the converging rung.
    pub etemp: f64,
    pub max_iter: usize,
    /// The retry that converged (`None`: the first try), e.g. `"etemp 1000 K, 1000 iterations"`.
    pub retry: Option<String>,
    /// Wall time of the part (all rungs), seconds.
    pub seconds: f64,
    /// Hessians: how the part's Hessian was computed.
    pub hessian_method: Option<HessianMethod>,
}

/// Results of a composite single point.
#[derive(Clone, Debug)]
pub struct CompositeResults {
    /// The composite: `energy` and `gradient` are E_gxtb + ΔG_solv and its gradient, the terms
    /// g-xTB's and `solvation_correction` (ΔG_solv); charges, moments, orbitals and iterations
    /// are g-xTB's.
    pub singlepoint: SinglePoint,
    /// The g-xTB part.
    pub gxtb: SinglePoint,
    /// The GFN2 halves with and without the solvent model.
    pub gfn2_solvated: SinglePoint,
    pub gfn2_gas: SinglePoint,
    /// ΔG_solv = E_GFN2+solv - E_GFN2 (Eh) and its gradient.
    pub solvation_energy: f64,
    pub solvation_gradient: Option<Vec<[f64; 3]>>,
    /// g-xTB, GFN2 solvated, GFN2 gas.
    pub parts: [PartInfo; 3],
}

/// A composite Hessian: total and parts.
#[derive(Clone, Debug)]
pub struct CompositeHessian {
    /// H_gxtb + H_GFN2+solv - H_GFN2, with energy and gradient alike (`method`: g-xTB's).
    pub total: axtb::Hessian,
    pub gxtb: axtb::Hessian,
    pub gfn2_solvated: axtb::Hessian,
    pub gfn2_gas: axtb::Hessian,
    /// ΔG_solv at the reference geometry (Eh).
    pub solvation_energy: f64,
    /// Electronic temperature, iteration limit and retry of each part, as in [`PartInfo`]
    /// (`energy` is the part's at the Hessian's reference SCF; no iteration count).
    pub parts: [PartInfo; 3],
}

/// The composite bound to one molecule: g-xTB plus two GFN2-xTB calculators.
pub struct CompositeCalculator {
    options: CompositeOptions,
    gxtb: Calculator,
    solvated: axtb::Calculator,
    gas: axtb::Calculator,
    /// Base temperature and iteration limits (before retries).
    gxtb_base: (f64, usize),
    gfn2_base: (f64, usize),
    reuse_guess: bool,
    /// The zero check of ΔG_solv has passed.
    checked: bool,
}

fn gfn2_label(c: axtb::Conventions) -> &'static str {
    c.label()
}

fn gxtb_label(c: Conventions) -> &'static str {
    match c {
        Conventions::Axtb => "axtb",
        Conventions::Gxtb => "gxtb",
    }
}

fn model_label(m: SolvationModel) -> &'static str {
    match m {
        SolvationModel::Alpb => "alpb",
        SolvationModel::Gbsa => "gbsa",
    }
}

fn state_label(s: axtb::ReferenceState) -> &'static str {
    match s {
        axtb::ReferenceState::Gsolv => "gsolv",
        axtb::ReferenceState::Bar1M => "bar1M",
        axtb::ReferenceState::Reference => "reference",
    }
}

fn retry_label(etemp: Option<f64>, max_iter: usize) -> String {
    match etemp {
        Some(t) => format!("etemp {t} K, {max_iter} iterations"),
        None => format!("{max_iter} iterations"),
    }
}

/// An SCF that failed: not converged, or broken down (Broyden failure, Fermi level lost).
/// Retried, as a failed program call would be; input errors are not.
fn not_converged(e: &Error) -> bool {
    matches!(e, Error::ScfNotConverged { .. } | Error::Numerical(_))
}

/// The SCF report of a part's single point: iterations, mixer, attempts.
fn new_scf(sp: &SinglePoint) -> (usize, Option<&'static str>, Vec<Attempt>) {
    let s = sp.scf().expect("every part has an SCF");
    (s.iterations, s.mixer_used, s.attempts.clone())
}

fn add(a: &[[f64; 3]], b: &[[f64; 3]], sign: f64) -> Vec<[f64; 3]> {
    a.iter().zip(b).map(|(x, y)| [x[0] + sign * y[0], x[1] + sign * y[1], x[2] + sign * y[2]]).collect()
}

/// Which retry a part converged on.
#[derive(Clone, Debug)]
struct Rung {
    etemp: f64,
    max_iter: usize,
    retry: Option<String>,
    seconds: f64,
}

impl CompositeCalculator {
    /// Set up the composite for a molecule (`positions` in bohr, `uhf` unpaired electrons;
    /// `None`: 0 or 1 by electron count).
    pub fn new(
        numbers: &[usize],
        positions: &[[f64; 3]],
        charge: f64,
        uhf: Option<usize>,
        options: CompositeOptions,
    ) -> Result<Self> {
        threads::check(options.threads)?;
        if !(options.accuracy > 0.0 && options.accuracy.is_finite()) {
            return Err(Error::Input(format!("accuracy must be positive and finite, got {}", options.accuracy)));
        }
        if let Some(t) = options.etemp {
            if !(t >= 0.0 && t.is_finite()) {
                return Err(Error::Input(format!("electronic temperature must be >= 0 K and finite, got {t}")));
            }
        }
        let threads = Threads::new(options.threads)?;

        // g-xTB part
        let gxtb_etemp = options.etemp.unwrap_or(GXTB_ETEMP);
        let gopts = Options {
            scf: ScfSettings {
                accuracy: options.accuracy,
                max_iter: options.max_iter,
                etemp: Some(gxtb_etemp),
                restart: options.reuse_guess.map(|r| if r { Restart::Previous } else { Restart::Fresh }),
            },
            threads: options.threads,
            conventions: options.conventions,
            ..Default::default()
        };
        let gxtb_max_iter = gopts.iteration_limit(options.version);
        let gxtb = Calculator::with_threads(options.version, numbers, positions, charge, uhf, gopts, threads.clone())?;

        // the GFN2 halves: one set of options, the solvated half adds the solvent model
        let gfn2_conv = options.gfn2_conventions();
        let mut o = axtb::Options { conventions: gfn2_conv, ..Default::default() };
        o.scf.accuracy = options.accuracy;
        o.threads = options.threads;
        o.mixing.mixer = options.mixer;
        o.scf.etemp = Some(options.etemp.unwrap_or(GFN2_ETEMP));
        o.scf.max_iter = options.max_iter;
        let gfn2_base = (o.scf.etemp.unwrap_or(GFN2_ETEMP), o.scf.max_iter.unwrap_or(axtb::MAX_ITER));
        let gas = axtb::Calculator::with_threads(
            axtb::Hamiltonian::Gfn2,
            numbers,
            positions,
            charge,
            uhf,
            o.clone(),
            threads.clone(),
        )?;
        let so = axtb::Options { solvation: Some(options.solvation.clone()), ..o };
        let solvated =
            axtb::Calculator::with_threads(axtb::Hamiltonian::Gfn2, numbers, positions, charge, uhf, so, threads)?;
        let reuse_guess = options.reuse_guess.unwrap_or(gfn2_conv == axtb::Conventions::Axtb);
        Ok(Self {
            options,
            gxtb,
            solvated,
            gas,
            gxtb_base: (gxtb_etemp, gxtb_max_iter),
            gfn2_base,
            reuse_guess,
            checked: false,
        })
    }

    /// The settings.
    pub fn options(&self) -> &CompositeOptions {
        &self.options
    }

    /// The molecule (positions of the last evaluation).
    pub fn molecule(&self) -> &axtb::Molecule {
        self.gxtb.molecule()
    }

    /// The composite's definition, e.g.
    /// `E = E[gxtb-v2.0.1] + (E[gfn2 + alpb(water, gsolv)] - E[gfn2])`.
    pub fn definition(&self) -> String {
        let o = &self.options;
        format!(
            "E = E[{}] + (E[{} + {}({}, {})] - E[{}])",
            o.version.method(),
            o.correction.name(),
            model_label(o.solvation.model),
            o.solvation.solvent.to_ascii_lowercase(),
            state_label(o.solvation.state),
            o.correction.name()
        )
    }

    /// Solvent model name (`"alpb"`, `"gbsa"`) and reference state name (`"gsolv"`, ...).
    pub fn solvation_labels(&self) -> (&'static str, &'static str) {
        (model_label(self.options.solvation.model), state_label(self.options.solvation.state))
    }

    /// Conventions of the g-xTB part and of the GFN2 halves.
    pub fn conventions_labels(&self) -> (&'static str, &'static str) {
        (gxtb_label(self.options.conventions), gfn2_label(self.options.gfn2_conventions()))
    }

    /// Base electronic temperatures (K) of the g-xTB part and the GFN2 halves.
    pub fn etemps(&self) -> (f64, f64) {
        (self.gxtb_base.0, self.gfn2_base.0)
    }

    /// Whether the GFN2 halves start from the previous geometry's wavefunction.
    pub fn reuses_guess(&self) -> bool {
        self.reuse_guess
    }

    /// Whether the g-xTB part starts from the previous geometry's converged state.
    pub fn gxtb_reuses_guess(&self) -> bool {
        self.gxtb.reuses_guess()
    }

    /// The default Hessian method of the g-xTB part (analytic for 2.0.1, numerical for v1.1).
    pub fn default_hessian_method(&self) -> HessianMethod {
        self.gxtb.info().hessian.default
    }

    /// Forget the previous converged states of all parts.
    pub fn reset_guess(&mut self) {
        self.gxtb.reset_wavefunction();
        self.solvated.reset_wavefunction();
        self.gas.reset_wavefunction();
    }

    /// The g-xTB part's calculator (e.g. for its wavefunction).
    pub fn gxtb(&self) -> &Calculator {
        &self.gxtb
    }

    /// The g-xTB part's calculator, mutable (e.g. to set its wavefunction).
    pub fn gxtb_mut(&mut self) -> &mut Calculator {
        &mut self.gxtb
    }

    /// The GFN2 halves' calculators (gas phase, solvated).
    pub fn gfn2_mut(&mut self) -> (&mut axtb::Calculator, &mut axtb::Calculator) {
        (&mut self.gas, &mut self.solvated)
    }

    /// The GFN2 halves' last wavefunctions (gas phase, solvated).
    pub fn gfn2_guesses(&self) -> (Option<Wavefunction>, Option<Wavefunction>) {
        (self.gas.wavefunction(), self.solvated.wavefunction())
    }

    /// g-xTB with its retries (only without an explicit electronic temperature).
    fn run_gxtb<T>(&mut self, mut f: impl FnMut(&mut Calculator) -> Result<T>) -> Result<(T, Rung)> {
        let t0 = Instant::now();
        let (etemp, max_iter) = self.gxtb_base;
        let first = f(&mut self.gxtb);
        let err = match first {
            Ok(v) => return Ok((v, Rung { etemp, max_iter, retry: None, seconds: t0.elapsed().as_secs_f64() })),
            Err(e) if not_converged(&e) && self.options.retries && self.options.etemp.is_none() => e,
            Err(e) => return Err(e),
        };
        let base = self.gxtb.options.clone();
        let mut out = Err(err.clone());
        for (t, m) in GXTB_RETRIES {
            let mut o = base.clone();
            let m = m.max(max_iter);
            o.scf.max_iter = Some(m);
            o.scf.etemp = Some(t);
            self.gxtb.options = o;
            match f(&mut self.gxtb) {
                Ok(v) => {
                    out = Ok((v, Rung { etemp: t, max_iter: m, retry: Some(retry_label(Some(t), m)), seconds: 0.0 }));
                    break;
                }
                Err(e) if not_converged(&e) => out = Err(e),
                Err(e) => {
                    out = Err(e);
                    break;
                }
            }
        }
        self.gxtb.options = base;
        out.map(|(v, mut r)| {
            r.seconds = t0.elapsed().as_secs_f64();
            (v, r)
        })
    }

    /// The GFN2 pair (gas, then solvated) with its retries: both
    /// halves always at the same rung.
    fn run_pair<T>(&mut self, mut f: impl FnMut(&mut axtb::Calculator) -> Result<T>) -> Result<(T, T, Rung, [f64; 2])> {
        let (etemp0, iter0) = self.gfn2_base;
        let mut rungs: Vec<(f64, usize, Option<String>)> = vec![(etemp0, iter0, None)];
        if self.options.retries {
            for (t, m) in PAIR_RETRIES {
                let m = m.max(iter0);
                rungs.push((t.unwrap_or(etemp0), m, Some(retry_label(t, m))));
            }
        }
        let mut seconds = [0.0; 2];
        let mut last = None;
        for (etemp, max_iter, retry) in rungs {
            for c in [&mut self.gas, &mut self.solvated] {
                c.options.scf.etemp = Some(etemp);
                c.options.scf.max_iter = Some(max_iter);
                if !self.reuse_guess {
                    c.reset_wavefunction();
                }
            }
            let t0 = Instant::now();
            let gas = f(&mut self.gas);
            seconds[0] += t0.elapsed().as_secs_f64();
            let gas = match gas {
                Ok(v) => v,
                Err(e) if not_converged(&e) => {
                    last = Some(e);
                    continue;
                }
                Err(e) => {
                    last = Some(e);
                    break;
                }
            };
            let t0 = Instant::now();
            let solv = f(&mut self.solvated);
            seconds[1] += t0.elapsed().as_secs_f64();
            match solv {
                Ok(solv) => {
                    self.restore_pair();
                    return Ok((gas, solv, Rung { etemp, max_iter, retry, seconds: 0.0 }, seconds));
                }
                Err(e) if not_converged(&e) => last = Some(e),
                Err(e) => {
                    last = Some(e);
                    break;
                }
            }
        }
        self.restore_pair();
        Err(last.expect("at least one rung"))
    }

    fn restore_pair(&mut self) {
        let (etemp, max_iter) = self.gfn2_base;
        for c in [&mut self.gas, &mut self.solvated] {
            c.options.scf.etemp = Some(etemp);
            c.options.scf.max_iter = Some(max_iter);
        }
    }

    /// The zero check on the first geometry.
    fn check(&mut self, dg: f64) -> Result<()> {
        if self.checked {
            return Ok(());
        }
        if !dg.is_finite() || dg.abs() < 1e-9 {
            return Err(Error::Numerical(format!(
                "the {} model for {} did not change the GFN2-xTB energy (dG_solv = {dg:.3e} Eh): refusing to report \
                 gas-phase energies as solvated",
                model_label(self.options.solvation.model).to_ascii_uppercase(),
                self.options.solvation.solvent
            )));
        }
        self.checked = true;
        Ok(())
    }

    fn part(
        &self,
        k: usize,
        energy: f64,
        scf: Option<(usize, Option<&'static str>, Vec<Attempt>)>,
        rung: &Rung,
        seconds: f64,
    ) -> PartInfo {
        let (gconv, fconv) = self.conventions_labels();
        let (method, conventions) = match k {
            0 => (self.options.version.method().to_string(), gconv),
            1 => (format!("{}+{}", self.options.correction.name(), model_label(self.options.solvation.model)), fconv),
            _ => (self.options.correction.name().to_string(), fconv),
        };
        PartInfo {
            method,
            conventions,
            energy,
            iterations: scf.as_ref().map_or(0, |s| s.0),
            mixer_used: scf.as_ref().and_then(|s| s.1),
            scf_attempts: scf.map_or_else(Vec::new, |s| s.2),
            etemp: rung.etemp,
            max_iter: rung.max_iter,
            retry: rung.retry.clone(),
            seconds,
            hessian_method: None,
        }
    }

    /// Energy and what `want` asks for (the gradient of all parts; orbitals and bond orders of
    /// the g-xTB part) of the composite at `positions` (bohr).
    pub fn singlepoint(&mut self, positions: &[[f64; 3]], want: &Want) -> Result<CompositeResults> {
        let t0 = Instant::now();
        let gradient = want.gradient;
        let (gx, grung) = self.run_gxtb(|c| c.singlepoint(positions, want))?;
        let (gas, solv, prung, secs) = self.run_pair(|c| c.singlepoint(positions, &Want::gradient(gradient)))?;
        let dg = solv.energy - gas.energy;
        self.check(dg)?;
        let dgrad = match (&solv.gradient, &gas.gradient) {
            (Some(s), Some(g)) => Some(add(s, g, -1.0)),
            _ => None,
        };
        let mut sp = gx.clone();
        sp.energy = gx.energy + dg;
        sp.terms.push(Term::piece("solvation_correction", dg));
        sp.gradient = match (&gx.gradient, &dgrad) {
            (Some(g), Some(d)) => Some(add(g, d, 1.0)),
            _ => None,
        };
        sp.timings = vec![
            ("gxtb", grung.seconds),
            ("gfn2 gas", secs[0]),
            ("gfn2 solvated", secs[1]),
            ("total", t0.elapsed().as_secs_f64()),
        ];
        let parts = [
            self.part(0, gx.energy, Some(new_scf(&gx)), &grung, grung.seconds),
            self.part(1, solv.energy, Some(new_scf(&solv)), &prung, secs[1]),
            self.part(2, gas.energy, Some(new_scf(&gas)), &prung, secs[0]),
        ];
        Ok(CompositeResults {
            singlepoint: sp,
            gxtb: gx,
            gfn2_solvated: solv,
            gfn2_gas: gas,
            solvation_energy: dg,
            solvation_gradient: dgrad,
            parts,
        })
    }

    /// Hessian of the composite at `positions` (bohr): H_gxtb + H_GFN2+solv - H_GFN2.
    ///
    /// `method` applies to all parts; `None`: each part's default (g-xTB 2.0.1 and GFN2-xTB
    /// analytic, g-xTB v1.1 numerical). `step` and `accuracy` apply to all parts when given,
    /// else each part's defaults (see [`Calculator::hessian`] and `axtb::Calculator::hessian`).
    pub fn hessian(
        &mut self,
        positions: &[[f64; 3]],
        method: Option<HessianMethod>,
        step: Option<f64>,
        accuracy: Option<f64>,
    ) -> Result<CompositeHessian> {
        let t0 = Instant::now();
        let gm = method.unwrap_or(self.gxtb.info().hessian.default);
        let gopts = HessianOptions { method: Some(gm), step, accuracy };
        let (hg, grung) = self.run_gxtb(|c| c.hessian(positions, &gopts))?;
        let fopts = HessianOptions { method: Some(method.unwrap_or(HessianMethod::Analytic)), step, accuracy };
        let (hgas, hsolv, prung, secs) = self.run_pair(|c| c.hessian(positions, &fopts))?;
        let dg = hsolv.energy - hgas.energy;
        self.check(dg)?;
        let n = hg.hessian.len();
        let mut h = Vec::with_capacity(n);
        for i in 0..n {
            h.push(hg.hessian[i] + (hsolv.hessian[i] - hgas.hessian[i]));
        }
        let gradient = add(&hg.gradient, &add(&hsolv.gradient, &hgas.gradient, -1.0), 1.0);
        let total = axtb::Hessian {
            energy: hg.energy + dg,
            gradient,
            hessian: h,
            max_asymmetry: hg.max_asymmetry.max(hsolv.max_asymmetry).max(hgas.max_asymmetry),
            method: hg.method,
            // the dipole is the g-xTB part's (IR intensities of the composite Hessian)
            dipole_derivatives: hg.dipole_derivatives.clone(),
            timings: vec![
                ("gxtb", grung.seconds),
                ("gfn2 gas", secs[0]),
                ("gfn2 solvated", secs[1]),
                ("total", t0.elapsed().as_secs_f64()),
            ],
            parts: Vec::new(),
        };
        let mut parts = [
            self.part(0, hg.energy, None, &grung, grung.seconds),
            self.part(1, hsolv.energy, None, &prung, secs[1]),
            self.part(2, hgas.energy, None, &prung, secs[0]),
        ];
        for (p, m) in parts.iter_mut().zip([hg.method, hsolv.method, hgas.method]) {
            p.hessian_method = Some(m);
        }
        Ok(CompositeHessian { total, gxtb: hg, gfn2_solvated: hsolv, gfn2_gas: hgas, solvation_energy: dg, parts })
    }
}
