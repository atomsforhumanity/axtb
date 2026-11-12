// SPDX-License-Identifier: LGPL-3.0-or-later
//! Methods made of methods: [`Composite`], a base method plus weighted corrections, each part a
//! [`Method`] and each group of parts with its [`RetryLadder`].
//!
//! E = E_base + Σ_j C_j with C_j = Σ_i w_ji E_ji, summed in the order of the parts (the first
//! term w_j0 E_j0, then + w_ji E_ji); gradients and Hessians the same way element by element.
//! A two-part difference (w = -1 then +1) is the difference of the two energies bit for bit
//! (-1·x and 1·x are exact, and IEEE addition is commutative), so E_base + (E_plus - E_minus)
//! keeps the association of a hand-written difference.

use super::{Fingerprint, Hessian, WavefunctionData};
use super::{
    HessianMethod, HessianOptions, Method, MethodInfo, RetryLadder, Rung, ScfReport, SinglePoint, Term, Want,
    Wavefunction,
};
use crate::{Error, Result};
use std::time::Instant;

/// A named value of a composite's description.
#[derive(Clone, Debug, PartialEq)]
pub enum Detail {
    Text(String),
    Name(&'static str),
    Number(f64),
    Flag(bool),
    Group(Vec<(&'static str, Detail)>),
}

/// What a composite reports with a single point: its description (what its builder says about
/// it), the energies of its parts and corrections, and how each part converged.
#[derive(Clone, Debug, PartialEq)]
pub struct CompositeReport {
    pub description: Vec<(&'static str, Detail)>,
    /// The parts' energies (in the report order) and the corrections (Eh).
    pub components: Vec<(&'static str, f64)>,
    pub parts: Vec<PartReport>,
}

/// How one part of a composite converged.
#[derive(Clone, Debug, PartialEq)]
pub struct PartReport {
    /// The part's key in the reports (e.g. `gfn2_solvated`).
    pub key: &'static str,
    /// What the part is (e.g. `gfn2+alpb`).
    pub label: String,
    /// The part's conventions.
    pub conventions: &'static str,
    /// The part's energy (Eh).
    pub energy: f64,
    /// Its SCF (a single point's part); `None` for a Hessian's parts.
    pub scf: Option<ScfReport>,
    /// Electronic temperature (K) and iteration limit of the try that converged.
    pub etemp: f64,
    pub max_iter: usize,
    /// The retry that converged (`None`: the first try).
    pub retry: Option<String>,
    /// Wall time of the part, all tries (s).
    pub seconds: f64,
    /// A Hessian's part: how its Hessian was computed.
    pub hessian_method: Option<HessianMethod>,
}

/// One part of a composite.
pub struct Part {
    /// Its key in the reports.
    pub key: &'static str,
    /// What it is, for the reports.
    pub label: String,
    /// Its conventions, for the reports.
    pub conventions: &'static str,
    /// Its name in the timings.
    pub timing: &'static str,
    pub method: Box<dyn Method>,
}

/// The check of a correction's first value (e.g. that a solvent model changed the energy); it
/// runs until it has passed once.
pub type Guard = Box<dyn FnMut(f64) -> Result<()> + Send + Sync>;

/// A correction: weighted parts tried together on one ladder (every part at the same rung).
pub struct Correction {
    /// Its term in the single point and its component in the reports.
    pub name: &'static str,
    /// Weights and parts, in the order they run and are summed.
    pub parts: Vec<(f64, Part)>,
    /// Its tries. Each rung carries the effective electronic temperature and iteration limit of
    /// its try (for the reports); `fresh` forgets the parts' wavefunctions before the try.
    pub ladder: RetryLadder,
    pub guard: Option<Guard>,
}

/// A method made of methods (see the module documentation).
pub struct Composite {
    info: MethodInfo,
    fingerprint: Fingerprint,
    base: Part,
    base_ladder: RetryLadder,
    corrections: Vec<Correction>,
    description: Vec<(&'static str, Detail)>,
    report_order: Vec<&'static str>,
}

/// The outcome of one group of parts on its ladder: each part's result, its time over all
/// tries, and the rung that converged.
struct Climbed<T> {
    results: Vec<T>,
    seconds: Vec<f64>,
    rung: Rung,
}

impl Composite {
    /// The composite of `base` (tried on `base_ladder`) and `corrections`. `info` says what it
    /// is (its name is its base's), `fingerprint` what its wavefunctions belong to.
    pub fn new(
        info: MethodInfo,
        fingerprint: Fingerprint,
        base: Part,
        base_ladder: RetryLadder,
        corrections: Vec<Correction>,
    ) -> Self {
        let mut report_order = vec![base.key];
        report_order.extend(corrections.iter().flat_map(|c| c.parts.iter().map(|(_, p)| p.key)));
        Self { info, fingerprint, base, base_ladder, corrections, description: Vec::new(), report_order }
    }

    /// The description the reports carry, and the order of the parts in them (by key; every
    /// part once).
    pub fn with_description(
        mut self,
        description: Vec<(&'static str, Detail)>,
        report_order: Vec<&'static str>,
    ) -> Self {
        self.description = description;
        self.report_order = report_order;
        self
    }

    /// The base part.
    pub fn base(&self) -> &Part {
        &self.base
    }

    /// The base part, mutable.
    pub fn base_mut(&mut self) -> &mut Part {
        &mut self.base
    }

    /// The corrections.
    pub fn corrections(&self) -> &[Correction] {
        &self.corrections
    }

    /// Every part: the base, then the corrections' parts in order.
    pub fn parts_mut(&mut self) -> Vec<&mut Part> {
        let mut out = vec![&mut self.base];
        out.extend(self.corrections.iter_mut().flat_map(|c| c.parts.iter_mut().map(|(_, p)| p)));
        out
    }

    /// The description of the reports.
    pub fn description(&self) -> &[(&'static str, Detail)] {
        &self.description
    }

    /// `parts` sorted into the report order.
    fn ordered(&self, mut parts: Vec<PartReport>) -> Vec<PartReport> {
        parts.sort_by_key(|p| self.report_order.iter().position(|k| *k == p.key).unwrap_or(usize::MAX));
        parts
    }

    fn singlepoint_inner(&mut self, positions: &[[f64; 3]], want: &Want) -> Result<SinglePoint> {
        let t0 = Instant::now();
        let base = climb(std::slice::from_mut(&mut self.base), &self.base_ladder, |m| m.singlepoint(positions, want))?;
        let mut sp = base.results.into_iter().next().expect("the base's result");
        let mut reports = vec![part_report(&self.base, sp.energy, sp.scf().cloned(), &base.rung, base.seconds[0])];
        let mut timings = vec![(self.base.timing, base.seconds[0])];
        let (mut energy, mut gradient) = (sp.energy, sp.gradient.take());
        let part_want = Want::gradient(want.gradient);
        let mut components = Vec::new();
        for c in &mut self.corrections {
            let (weights, mut parts): (Vec<f64>, Vec<Part>) = std::mem::take(&mut c.parts).into_iter().unzip();
            let climbed = climb(&mut parts, &c.ladder, |m| m.singlepoint(positions, &part_want));
            c.parts = weights.iter().copied().zip(parts).collect();
            let climbed = climbed?;
            let value = weighted(&weights, climbed.results.iter().map(|r| r.energy));
            if let Some(guard) = c.guard.as_mut() {
                guard(value)?;
                c.guard = None;
            }
            for (((_, part), r), s) in c.parts.iter().zip(&climbed.results).zip(&climbed.seconds) {
                reports.push(part_report(part, r.energy, r.scf().cloned(), &climbed.rung, *s));
                timings.push((part.timing, *s));
            }
            if let Some(g) = gradient.as_mut() {
                let gs: Vec<&Vec<[f64; 3]>> =
                    climbed.results.iter().map(|r| r.gradient.as_ref().expect("the parts' gradients")).collect();
                let cg = weighted_rows(&weights, &gs);
                for (a, b) in g.iter_mut().zip(&cg) {
                    for k in 0..3 {
                        a[k] += b[k];
                    }
                }
            }
            energy += value;
            sp.terms.push(Term::piece(c.name, value));
            components.push((c.name, value));
        }
        timings.push(("total", t0.elapsed().as_secs_f64()));
        let parts = self.ordered(reports);
        let mut all: Vec<(&'static str, f64)> = parts.iter().map(|p| (p.key, p.energy)).collect();
        all.extend(components);
        sp.energy = energy;
        sp.gradient = gradient;
        sp.timings = timings;
        sp.extras.composite =
            Some(Box::new(CompositeReport { description: self.description.clone(), components: all, parts }));
        Ok(sp)
    }

    fn hessian_inner(&mut self, positions: &[[f64; 3]], opts: &HessianOptions) -> Result<Hessian> {
        let t0 = Instant::now();
        let base = climb(std::slice::from_mut(&mut self.base), &self.base_ladder, |m| m.hessian(positions, opts))?;
        let mut h = base.results.into_iter().next().expect("the base's Hessian");
        let mut reports = vec![PartReport {
            hessian_method: Some(h.method),
            ..part_report(&self.base, h.energy, None, &base.rung, base.seconds[0])
        }];
        let mut timings = vec![(self.base.timing, base.seconds[0])];
        for c in &mut self.corrections {
            let (weights, mut parts): (Vec<f64>, Vec<Part>) = std::mem::take(&mut c.parts).into_iter().unzip();
            let climbed = climb(&mut parts, &c.ladder, |m| m.hessian(positions, opts));
            c.parts = weights.iter().copied().zip(parts).collect();
            let climbed = climbed?;
            let value = weighted(&weights, climbed.results.iter().map(|r| r.energy));
            if let Some(guard) = c.guard.as_mut() {
                guard(value)?;
                c.guard = None;
            }
            for (((_, part), r), s) in c.parts.iter().zip(&climbed.results).zip(&climbed.seconds) {
                reports.push(PartReport {
                    hessian_method: Some(r.method),
                    ..part_report(part, r.energy, None, &climbed.rung, *s)
                });
                timings.push((part.timing, *s));
                h.max_asymmetry = h.max_asymmetry.max(r.max_asymmetry);
            }
            let gs: Vec<&Vec<[f64; 3]>> = climbed.results.iter().map(|r| &r.gradient).collect();
            let cg = weighted_rows(&weights, &gs);
            for (a, b) in h.gradient.iter_mut().zip(&cg) {
                for k in 0..3 {
                    a[k] += b[k];
                }
            }
            let hs: Vec<&[f64]> = climbed.results.iter().map(|r| r.hessian.as_slice()).collect();
            let ch = weighted_values(&weights, &hs);
            for (a, b) in h.hessian.iter_mut().zip(&ch) {
                *a += b;
            }
            h.energy += value;
        }
        timings.push(("total", t0.elapsed().as_secs_f64()));
        h.timings = timings;
        h.parts = self.ordered(reports);
        Ok(h)
    }
}

/// Run `f` on every part of a group in order, rung by rung of `ladder`, until every part
/// succeeds at one rung; the parts' settings are those of the rung for its try and their own
/// again afterwards.
fn climb<T>(
    parts: &mut [Part],
    ladder: &RetryLadder,
    mut f: impl FnMut(&mut dyn Method) -> Result<T>,
) -> Result<Climbed<T>> {
    let base: Vec<_> = parts.iter().map(|p| p.method.scf_settings().cloned()).collect();
    let mut seconds = vec![0.0; parts.len()];
    let r = ladder.climb(|rung| {
        let mut results = Vec::with_capacity(parts.len());
        for ((part, b), s) in parts.iter_mut().zip(&base).zip(seconds.iter_mut()) {
            if let Some(b) = b {
                let mut settings = b.clone();
                settings.etemp = rung.etemp.or(b.etemp);
                settings.max_iter = rung.max_iter.or(b.max_iter);
                part.method.set_scf_settings(settings)?;
            }
            if rung.fresh {
                part.method.reset_wavefunction();
            }
            let t0 = Instant::now();
            let r = f(part.method.as_mut());
            *s += t0.elapsed().as_secs_f64();
            results.push(r?);
        }
        Ok(results)
    });
    for (part, b) in parts.iter_mut().zip(base) {
        if let Some(b) = b {
            part.method.set_scf_settings(b)?;
        }
    }
    let (results, i) = r?;
    Ok(Climbed { results, seconds, rung: ladder.rungs[i].clone() })
}

/// The report of `part` with `energy` and `scf`, converged at `rung`.
fn part_report(part: &Part, energy: f64, scf: Option<ScfReport>, rung: &Rung, seconds: f64) -> PartReport {
    PartReport {
        key: part.key,
        label: part.label.clone(),
        conventions: part.conventions,
        energy,
        scf,
        etemp: rung.etemp.unwrap_or(f64::NAN),
        max_iter: rung.max_iter.unwrap_or(0),
        retry: rung.label.clone(),
        seconds,
        hessian_method: None,
    }
}

/// Σ_i w_i x_i from the first term.
fn weighted(w: &[f64], x: impl Iterator<Item = f64>) -> f64 {
    let mut terms = w.iter().zip(x).map(|(w, x)| w * x);
    let first = terms.next().unwrap_or(0.0);
    terms.fold(first, |acc, t| acc + t)
}

/// Σ_i w_i x_i of rows of 3-vectors, element by element.
fn weighted_rows(w: &[f64], x: &[&Vec<[f64; 3]>]) -> Vec<[f64; 3]> {
    let n = x.first().map_or(0, |r| r.len());
    (0..n).map(|a| std::array::from_fn(|k| weighted(w, x.iter().map(|r| r[a][k])))).collect()
}

/// Σ_i w_i x_i of vectors, element by element.
fn weighted_values(w: &[f64], x: &[&[f64]]) -> Vec<f64> {
    let n = x.first().map_or(0, |r| r.len());
    (0..n).map(|e| weighted(w, x.iter().map(|r| r[e]))).collect()
}

impl Method for Composite {
    fn info(&self) -> &MethodInfo {
        &self.info
    }

    /// The base's checks with `want`, every part's with the gradient alone; every part's for a
    /// Hessian.
    fn check(&self, want: &Want, hessian: Option<&HessianOptions>) -> Result<()> {
        self.base.method.check(want, hessian)?;
        for c in &self.corrections {
            for (_, p) in &c.parts {
                p.method.check(&Want::gradient(want.gradient), hessian)?;
            }
        }
        Ok(())
    }

    /// The base's single point (its charges, electronic structure and terms) with the
    /// composite's energy and gradient, one term per correction, the timings of the parts and
    /// [`Extras::composite`](super::Extras::composite). The orbitals and bond orders `want` asks
    /// for are the base's.
    fn singlepoint(&mut self, positions: &[[f64; 3]], want: &Want) -> Result<SinglePoint> {
        let r = self.singlepoint_inner(positions, want);
        if r.is_err() {
            self.reset_wavefunction();
        }
        r
    }

    /// The parts' Hessians (each with `opts`, `None` fields its own defaults) combined; the
    /// method and dipole derivatives are the base's, the largest asymmetry over the parts.
    /// Afterwards each part holds the wavefunction its own Hessian leaves.
    fn hessian(&mut self, positions: &[[f64; 3]], opts: &HessianOptions) -> Result<Hessian> {
        let r = self.hessian_inner(positions, opts);
        if r.is_err() {
            self.reset_wavefunction();
        }
        r
    }

    /// Every part's wavefunction (`None` where a part has none); `None` when no part has one.
    fn wavefunction(&self) -> Option<Wavefunction> {
        let mut parts = vec![self.base.method.wavefunction()];
        parts.extend(self.corrections.iter().flat_map(|c| c.parts.iter().map(|(_, p)| p.method.wavefunction())));
        parts
            .iter()
            .any(Option::is_some)
            .then(|| Wavefunction::new(self.fingerprint.clone(), WavefunctionData::Parts(parts)))
    }

    /// Every part's next SCF from its wavefunction in `wf` (parts without one keep theirs).
    fn set_wavefunction(&mut self, wf: &Wavefunction) -> Result<()> {
        wf.check(&self.fingerprint)?;
        let WavefunctionData::Parts(given) = wf.data() else {
            return Err(Error::Input("the wavefunction belongs to another kind of calculation".into()));
        };
        let mut parts = self.parts_mut();
        if given.len() != parts.len() {
            return Err(Error::Input("the wavefunction belongs to another kind of calculation".into()));
        }
        for (part, w) in parts.iter_mut().zip(given) {
            if let Some(w) = w {
                part.method.set_wavefunction(w)?;
            }
        }
        Ok(())
    }

    fn reset_wavefunction(&mut self) {
        for part in self.parts_mut() {
            part.method.reset_wavefunction();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::{
        AnalysisConventions, AnalysisDefaults, Electrons, Extras, HessianDefaults, RetryOn, ScfSettings,
    };
    use super::*;

    /// A one-atom "method" with energy `e` at temperatures of at least `needs` K (below: an SCF
    /// that does not converge) and gradient `e` along x.
    struct Toy {
        e: f64,
        needs: f64,
        scf: ScfSettings,
        info: MethodInfo,
        tries: usize,
    }

    fn toy(e: f64, needs: f64) -> Box<dyn Method> {
        let info = MethodInfo {
            name: "toy",
            conventions: "axtb",
            electrons: Electrons::None,
            hessian: HessianDefaults { default: HessianMethod::Numerical, offered: vec![] },
            analysis: AnalysisDefaults { conventions: AnalysisConventions::Axtb, xtb_frequency_scale: None },
            quadrupole: None,
        };
        let scf = ScfSettings { etemp: Some(0.0), ..ScfSettings::default() };
        Box::new(Toy { e, needs, scf, info, tries: 0 })
    }

    impl Method for Toy {
        fn info(&self) -> &MethodInfo {
            &self.info
        }
        fn singlepoint(&mut self, _: &[[f64; 3]], want: &Want) -> Result<SinglePoint> {
            self.tries += 1;
            if self.scf.etemp.unwrap() < self.needs {
                return Err(Error::ScfNotConverged { iterations: 1, energy_change: 0.0, residual: 0.0 });
            }
            Ok(SinglePoint {
                energy: self.e,
                terms: vec![Term::piece("toy", self.e)],
                gradient: want.gradient.then(|| vec![[self.e, 0.0, 0.0]]),
                charges: vec![0.0],
                dipole: [0.0; 3],
                electronic: None,
                extras: Extras::default(),
                timings: vec![],
            })
        }
        fn hessian(&mut self, _: &[[f64; 3]], _: &HessianOptions) -> Result<Hessian> {
            Err(Error::Input("no Hessian".into()))
        }
        fn scf_settings(&self) -> Option<&ScfSettings> {
            Some(&self.scf)
        }
        fn set_scf_settings(&mut self, s: ScfSettings) -> Result<()> {
            self.scf = s;
            Ok(())
        }
    }

    fn part(key: &'static str, method: Box<dyn Method>) -> Part {
        Part { key, label: key.into(), conventions: "axtb", timing: key, method }
    }

    /// E = E_base + (E_b - E_a) bit for bit; the correction's parts climb together to the rung
    /// both converge on; the settings are restored; the guard sees the correction once.
    #[test]
    fn weighted_sum_and_ladders() {
        let info = toy(0.0, 0.0).info().clone();
        let fingerprint = Fingerprint::new("toy", &[1], 0.0, None);
        let rung = |t: f64| Rung { fresh: false, etemp: Some(t), max_iter: Some(1), label: Some(format!("{t} K")) };
        let ladder = RetryLadder { rungs: vec![rung(0.0), rung(500.0), rung(1000.0)], on: RetryOn::NotConverged };
        let (ea, eb) = (0.1 + 0.2, 0.3_f64.sqrt());
        let seen = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        let s2 = seen.clone();
        let correction = Correction {
            name: "delta",
            parts: vec![(-1.0, part("a", toy(ea, 0.0))), (1.0, part("b", toy(eb, 800.0)))],
            ladder,
            guard: Some(Box::new(move |c| {
                s2.lock().unwrap().push(c);
                Ok(())
            })),
        };
        let base = part("base", toy(-1.5, 0.0));
        let mut c = Composite::new(info, fingerprint, base, RetryLadder::once(), vec![correction]);
        for _ in 0..2 {
            let sp = c.singlepoint(&[[0.0; 3]], &Want::GRADIENT).unwrap();
            assert_eq!(sp.energy.to_bits(), (-1.5 + (eb - ea)).to_bits());
            assert_eq!(sp.gradient.as_ref().unwrap()[0][0].to_bits(), (-1.5 + (eb - ea)).to_bits());
            assert_eq!(sp.term("delta"), Some(eb - ea));
            let r = sp.extras.composite.unwrap();
            assert_eq!(r.parts[1].retry.as_deref(), Some("1000 K"));
            assert_eq!(r.components.last(), Some(&("delta", eb - ea)));
        }
        assert_eq!(*seen.lock().unwrap(), vec![eb - ea]);
        for p in c.parts_mut() {
            assert_eq!(p.method.scf_settings().unwrap().etemp, Some(0.0));
        }
    }
}
