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
//! - **The SCF retry ladders** ([`PAIR_RETRIES`], [`GXTB_RETRIES`], axtb-base's `RetryLadder`). A GFN2 pair whose SCF fails
//!   (no convergence, or a numerical breakdown) is run again, both halves at the same settings,
//!   with 1000 K and 1000 iterations, then 1000 iterations, then 5000 K and 1000 iterations. A
//!   g-xTB SCF that fails is run again at 1000 K, then 3000 K, each with 1000 iterations,
//!   unless the temperature was set explicitly. The rung that converged is reported per part
//!   (the part's `retry` in the composite's report), so a result from a smeared rung is never
//!   mistaken for a 0 K one.
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
use axtb::method::{
    Composite, Correction as CompositeCorrection, Detail, Fingerprint, Guard, HessianOptions, Method, MethodInfo, Part,
    Restart, RetryLadder, RetryOn, Rung, ScfSettings, SinglePoint, Want, Wavefunction, WavefunctionData,
};
use axtb::{Error, MixerKind, Result, SolvationModel, SolvationSpec};
use axtb_base::threads::{self, Threads};

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

/// The keys of the composite's parts in its reports, in their order.
pub const PART_KEYS: [&str; 3] = ["gxtb", "gfn2_solvated", "gfn2_gas"];

/// The composite bound to one molecule: g-xTB plus two GFN2-xTB calculators, on axtb-base's
/// generic [`Composite`] (this type is its g-xTB policy: which methods pair, the temperatures,
/// the retry ladders, the zero check and the labels).
pub struct CompositeCalculator {
    options: CompositeOptions,
    inner: Composite,
    /// Base temperature and iteration limits (before retries).
    gxtb_base: (f64, usize),
    gfn2_base: (f64, usize),
    reuse_guess: bool,
    gxtb_reuse_guess: bool,
}

fn gfn2_label(c: axtb::Conventions) -> &'static str {
    c.label()
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

/// The rungs of a ladder: the first try at `base` (temperature, iteration limit), then each of
/// `retries` (temperature, `None`: the base's; iteration limit, at least the base's); `fresh`:
/// every try from the method's own start.
fn rungs(base: (f64, usize), retries: &[(Option<f64>, usize)], fresh: bool) -> Vec<Rung> {
    let mut out = vec![Rung { fresh, etemp: Some(base.0), max_iter: Some(base.1), label: None }];
    for &(t, m) in retries {
        let m = m.max(base.1);
        out.push(Rung { fresh, etemp: Some(t.unwrap_or(base.0)), max_iter: Some(m), label: Some(retry_label(t, m)) });
    }
    out
}

/// What the composite is and offers: g-xTB's information, with the Hessian methods every part
/// offers.
fn info(gxtb: &MethodInfo, gfn2: &MethodInfo) -> MethodInfo {
    let mut info = gxtb.clone();
    info.hessian.offered.retain(|d| gfn2.hessian.get(d.method).is_some());
    info
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
        let gxtb_reuse_guess = gopts.reuses();
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

        // the ladders: g-xTB's only without an explicit temperature; the pair's every rung from
        // the method's own start when the halves do not reuse wavefunctions
        let gxtb_base = (gxtb_etemp, gxtb_max_iter);
        let gxtb_retries: Vec<(Option<f64>, usize)> = if options.retries && options.etemp.is_none() {
            GXTB_RETRIES.map(|(t, m)| (Some(t), m)).to_vec()
        } else {
            vec![]
        };
        let gxtb_ladder = RetryLadder { rungs: rungs(gxtb_base, &gxtb_retries, false), on: RetryOn::Failed };
        let pair_retries = if options.retries { PAIR_RETRIES.to_vec() } else { vec![] };
        let pair_ladder = RetryLadder { rungs: rungs(gfn2_base, &pair_retries, !reuse_guess), on: RetryOn::Failed };

        let info = info(gxtb.info(), gas.info());
        let fingerprint = Fingerprint::new(info.name, numbers, charge, uhf);
        let gxtb_conventions = options.conventions.label();
        let base = Part {
            key: PART_KEYS[0],
            label: options.version.method().to_string(),
            conventions: gxtb_conventions,
            timing: "gxtb",
            method: Box::new(gxtb),
        };
        let correction = options.correction.name();
        let model = model_label(options.solvation.model);
        let gas = Part {
            key: PART_KEYS[2],
            label: correction.to_string(),
            conventions: gfn2_label(gfn2_conv),
            timing: "gfn2 gas",
            method: Box::new(gas),
        };
        let solvated = Part {
            key: PART_KEYS[1],
            label: format!("{correction}+{model}"),
            conventions: gfn2_label(gfn2_conv),
            timing: "gfn2 solvated",
            method: Box::new(solvated),
        };
        // the zero check: the first geometry's dG_solv must not vanish
        let (model_name, solvent) = (model.to_ascii_uppercase(), options.solvation.solvent.clone());
        let guard: Guard = Box::new(move |dg: f64| {
            if !dg.is_finite() || dg.abs() < 1e-9 {
                return Err(Error::Numerical(format!(
                    "the {model_name} model for {solvent} did not change the GFN2-xTB energy (dG_solv = {dg:.3e} \
                     Eh): refusing to report gas-phase energies as solvated"
                )));
            }
            Ok(())
        });
        let solvation = CompositeCorrection {
            name: "solvation_correction",
            // gas, then solvated: dG_solv = E_GFN2+solv - E_GFN2
            parts: vec![(-1.0, gas), (1.0, solvated)],
            ladder: pair_ladder,
            guard: Some(guard),
        };
        let mut c = Self {
            options,
            inner: Composite::new(info, fingerprint, base, gxtb_ladder, vec![solvation]),
            gxtb_base,
            gfn2_base,
            reuse_guess,
            gxtb_reuse_guess,
        };
        let description = c.describe();
        c.inner = c.inner.with_description(description, PART_KEYS.to_vec());
        Ok(c)
    }

    /// The settings.
    pub fn options(&self) -> &CompositeOptions {
        &self.options
    }

    /// The generic composite (its parts: g-xTB, then the GFN2 halves gas and solvated).
    pub fn composite(&self) -> &Composite {
        &self.inner
    }

    /// The generic composite, mutable.
    pub fn composite_mut(&mut self) -> &mut Composite {
        &mut self.inner
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
        (self.options.conventions.label(), gfn2_label(self.options.gfn2_conventions()))
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
        self.gxtb_reuse_guess
    }

    /// The composite's definition and settings, as its reports carry them.
    fn describe(&self) -> Vec<(&'static str, Detail)> {
        let o = &self.options;
        let (model, state) = self.solvation_labels();
        let (conventions, correction_conventions) = self.conventions_labels();
        let (gxtb, gfn2) = self.etemps();
        vec![
            ("definition", Detail::Text(self.definition())),
            ("method", Detail::Name(o.version.method())),
            ("solvation_correction", Detail::Name(o.correction.name())),
            ("solvation_model", Detail::Name(model)),
            ("solvent", Detail::Text(o.solvation.solvent.clone())),
            ("solvation_state", Detail::Name(state)),
            ("conventions", Detail::Name(conventions)),
            ("correction_conventions", Detail::Name(correction_conventions)),
            ("etemp", Detail::Group(vec![("gxtb", Detail::Number(gxtb)), ("gfn2", Detail::Number(gfn2))])),
            ("reuse_guess", Detail::Flag(self.reuses_guess())),
            ("gxtb_reuse_guess", Detail::Flag(self.gxtb_reuses_guess())),
            ("retries", Detail::Flag(o.retries)),
        ]
    }
}

impl Method for CompositeCalculator {
    fn info(&self) -> &MethodInfo {
        self.inner.info()
    }

    fn check(&self, want: &Want, hessian: Option<&HessianOptions>) -> Result<()> {
        self.inner.check(want, hessian)
    }

    /// E_gxtb + ΔG_solv and its gradient; the charges, electronic structure and terms are
    /// g-xTB's, with `solvation_correction` (ΔG_solv) after them; the orbitals and bond orders
    /// `want` asks for are g-xTB's. The report ([`Extras::composite`](axtb::method::Extras))
    /// has the definition and settings, the components and how each part converged.
    fn singlepoint(&mut self, positions: &[[f64; 3]], want: &Want) -> Result<SinglePoint> {
        self.inner.singlepoint(positions, want)
    }

    /// H_gxtb + H_GFN2+solv - H_GFN2. A Hessian method given applies to all parts; `None`: each
    /// part's default (g-xTB 2.0.1 and GFN2-xTB analytic, g-xTB v1.1 numerical); so do the step
    /// and the accuracy. The method and the dipole derivatives (IR intensities) are g-xTB's.
    fn hessian(&mut self, positions: &[[f64; 3]], opts: &HessianOptions) -> Result<axtb::Hessian> {
        self.inner.hessian(positions, opts)
    }

    fn wavefunction(&self) -> Option<Wavefunction> {
        self.inner.wavefunction()
    }

    /// The parts' wavefunctions of another composite of this method and molecule; starting
    /// moments are refused as for g-xTB alone.
    fn set_wavefunction(&mut self, wf: &Wavefunction) -> Result<()> {
        if matches!(wf.data(), WavefunctionData::Scf(d) if d.densities.is_empty()) {
            return Err(Error::Input(
                "starting moments are GFN2-only; g-xTB starts from a whole wavefunction (the guess of another \
                 calculator)"
                    .into(),
            ));
        }
        self.inner.set_wavefunction(wf)
    }

    fn reset_wavefunction(&mut self) {
        self.inner.reset_wavefunction()
    }
}
