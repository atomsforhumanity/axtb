//! Broken-symmetry unrestricted singlets: an axtb extension of g-xTB; neither g-xTB program has
//! it ([`crate::ScfOptions::broken_symmetry`]).
//!
//! Both programs run a molecule without unpaired electrons with one spin channel, so a singlet
//! that dissociates into two radicals stays closed-shell and never reaches the radical-pair
//! limit (H2 at 3 Å: the 2.0.1 singlet lies 58 kcal/mol above the triplet). g-xTB's spin
//! polarization energy ½ Σ m W m over the shell magnetizations is defined for any spin density,
//! so an unrestricted singlet (two spin channels, N_alpha = N_beta) is a state of the same
//! energy expression; it reduces exactly to the restricted one at zero magnetization. g-xTB
//! was fitted on restricted singlets, and a broken-symmetry energy mixes singlet and triplet
//! character.
//!
//! At one geometry ([`scf_v2`], [`scf_v1`]):
//! 1. the restricted SCF converges (from the previous geometry's restricted state or the
//!    program's start, as without the setting), with every orbital;
//! 2. the start of the unrestricted SCF: the restricted orbitals with each frontier pair
//!    (HOMO − k, LUMO + k) rotated by +θ for alpha and −θ for beta, θ = 45° ([`THETA`]), one
//!    pair for each orbital of the smaller of the degenerate HOMO and LUMO sets (orbital
//!    energies within [`DEGENERATE`] of the HOMO's, the LUMO's), each occupied once
//!    ([`mixed_start`]); its densities and Mulliken moments start the SCF as a restart does;
//!    at a finite electronic temperature the start is the same integer-occupied determinant
//!    and the SCF smears its own occupations;
//! 3. the unrestricted SCF (two spin channels, N_alpha = N_beta) from there, to the thresholds
//!    of a restart;
//! 4. the unrestricted solution is the result if it lies more than [`ENERGY_TOL`] (times the
//!    SCF accuracy when that exceeds 1) below the restricted one; otherwise, or when it does not
//!    converge, the restricted solution is ([`Outcome`]).
//!
//! When the restricted SCF of a fresh start does not converge (stretched bonds are where it
//! struggles), the orbitals of a restricted SCF at [`FALLBACK_ETEMPS`] make the start of step 2
//! instead; an unrestricted solution with ⟨S²⟩ ≥ [`S2_CHECK`] is then the result without a
//! restricted energy to compare with ([`BrokenSymmetry::start_etemp`]).
//!
//! At the next geometry a calculator whose previous state is unrestricted runs only the
//! unrestricted SCF from that state. When its ⟨S²⟩ falls below [`S2_CHECK`] (the magnetization
//! is collapsing onto the restricted solution), the restricted SCF runs too, from the total
//! density, then steps 2-4 from its orbitals, and the lowest solution below the restricted one
//! by the tolerance is the result; a restricted result sends the next geometry back to step 1.
//! A restricted previous state always runs steps 1-4. Every SCF of a geometry is listed in
//! [`BrokenSymmetry::scfs`].
//!
//! ⟨S²⟩ of a single determinant (or of smeared occupations): S_z(S_z + 1) + N_beta −
//! tr(P^alpha S P^beta S) ([`s2`]); the 1.1.0 program prints ⟨S²⟩ − S_z(S_z + 1) of open
//! shells as "spin-contamination".

use crate::calculator::v2_options;
use crate::scf::{ScfOptions, ScfResult, Setup};
use crate::v2;
use axtb::{Error, Result};
use faer::linalg::matmul::matmul;
use faer::{Accum, MatMut, MatRef, Par};

/// Rotation angle (rad) of the frontier orbital pairs in the start: 45°, the alpha HOMO
/// (φ_H + φ_L)/√2 and the beta HOMO (φ_H − φ_L)/√2.
pub const THETA: f64 = std::f64::consts::FRAC_PI_4;

/// Orbitals within this energy (Eh) of the HOMO (LUMO) belong to its degenerate set.
pub const DEGENERATE: f64 = 1e-6;

/// The unrestricted solution is kept when it lies more than this (Eh) below the restricted
/// one; with an SCF accuracy above 1, this times the accuracy.
pub const ENERGY_TOL: f64 = 1e-6;

/// An unrestricted solution with at least this ⟨S²⟩ is kept without a restricted energy to
/// compare with (restarted from the previous geometry's broken-symmetry state, or started from
/// smeared orbitals); below, the restricted solution is computed and compared.
pub const S2_CHECK: f64 = 0.05;

/// Electronic temperatures (K) of the restricted SCFs, tried in turn, whose orbitals start the
/// unrestricted SCF when the restricted SCF of a fresh start does not converge (those above
/// the calculation's own temperature).
pub const FALLBACK_ETEMPS: [f64; 2] = [1000.0, 3000.0];

/// Which solution a broken-symmetry calculation returned, and why.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// The unrestricted (broken-symmetry) solution: it lies lower than the restricted one, or
    /// it kept a magnetization (⟨S²⟩ ≥ [`S2_CHECK`]) without a restricted SCF to compare with.
    Broken,
    /// The restricted solution: the unrestricted SCF from the HOMO-LUMO start ended within
    /// [`ENERGY_TOL`] of it (the symmetry did not break) or above it.
    NotLower,
    /// The restricted solution: the unrestricted SCF restarted from the previous geometry's
    /// broken-symmetry state collapsed onto it, and a new HOMO-LUMO start did not end lower.
    Collapsed,
    /// The restricted solution: the unrestricted SCF from the HOMO-LUMO start did not converge.
    NotConverged,
    /// The restricted solution: there is no HOMO-LUMO pair to mix (no electrons, or every
    /// orbital occupied).
    NoFrontier,
}

impl Outcome {
    /// Name as the front ends report it: `broken`, `not-lower`, `collapsed`,
    /// `not-converged`, `no-frontier`.
    pub fn name(self) -> &'static str {
        match self {
            Outcome::Broken => "broken",
            Outcome::NotLower => "not-lower",
            Outcome::Collapsed => "collapsed",
            Outcome::NotConverged => "not-converged",
            Outcome::NoFrontier => "no-frontier",
        }
    }
}

/// One SCF of a broken-symmetry calculation.
#[derive(Clone, Debug, PartialEq)]
pub struct ScfRun {
    /// Spin channels: 1 (restricted) or 2 (unrestricted).
    pub channels: usize,
    /// Where it started: `"previous"` (the previous geometry's state), `"eeq-bc"` (the program's
    /// start), `"homo-lumo"` (the mixed restricted orbitals), `"total-density"` (the total
    /// density of the unrestricted solution of this geometry).
    pub start: &'static str,
    /// Electronic temperature (K).
    pub etemp: f64,
    /// Iterations (the limit when it did not converge).
    pub iterations: usize,
    pub converged: bool,
    /// Energy (Eh) and ⟨S²⟩ when converged.
    pub energy: Option<f64>,
    pub s2: Option<f64>,
}

/// What the broken-symmetry treatment did at one geometry.
#[derive(Clone, Debug, PartialEq)]
pub struct BrokenSymmetry {
    pub outcome: Outcome,
    /// ⟨S²⟩ of the returned solution (0 for the restricted one).
    pub s2: f64,
    /// Start of the reported unrestricted solution (`unrestricted_energy`): `"homo-lumo"` or
    /// `"previous"`; `None` when no unrestricted SCF converged.
    pub start: Option<&'static str>,
    /// Frontier orbital pairs rotated in the HOMO-LUMO start (0 without one).
    pub pairs: usize,
    /// The restricted solution's energy (Eh) at the calculation's temperature, when computed.
    pub restricted_energy: Option<f64>,
    /// Energy (Eh) and ⟨S²⟩ of the lowest converged unrestricted solution.
    pub unrestricted_energy: Option<f64>,
    pub unrestricted_s2: Option<f64>,
    /// SCF iterations of all restricted and of all unrestricted SCFs at this geometry (`None`:
    /// none ran).
    pub restricted_iterations: Option<usize>,
    pub unrestricted_iterations: Option<usize>,
    /// Electronic temperature (K) of the restricted SCF whose orbitals made the HOMO-LUMO
    /// start when the restricted SCF did not converge at the calculation's own ([`FALLBACK_ETEMPS`]).
    pub start_etemp: Option<f64>,
    /// Every SCF of this geometry, in order.
    pub scfs: Vec<ScfRun>,
}

impl BrokenSymmetry {
    /// Whether the result is the unrestricted (broken-symmetry) solution.
    pub fn broken(&self) -> bool {
        self.outcome == Outcome::Broken
    }

    fn new() -> Self {
        Self {
            outcome: Outcome::NotLower,
            s2: 0.0,
            start: None,
            pairs: 0,
            restricted_energy: None,
            unrestricted_energy: None,
            unrestricted_s2: None,
            restricted_iterations: None,
            unrestricted_iterations: None,
            start_etemp: None,
            scfs: Vec::new(),
        }
    }
}

/// ⟨S²⟩ of the spin densities `pa`, `pb` (row-major n x n) with the overlap `s`:
/// S_z(S_z + 1) + N_beta − tr(P^alpha S P^beta S), S_z = (N_alpha − N_beta)/2.
pub fn s2(n: usize, s: &[f64], pa: &[f64], pb: &[f64], par: Par) -> f64 {
    let ps = |p: &[f64]| {
        let mut out = vec![0.0; n * n];
        matmul(
            MatMut::from_row_major_slice_mut(&mut out, n, n),
            Accum::Replace,
            MatRef::from_row_major_slice(p, n, n),
            MatRef::from_row_major_slice(s, n, n),
            1.0,
            par,
        );
        out
    };
    let (a, b) = (ps(pa), ps(pb));
    let trace = |m: &[f64]| (0..n).map(|i| m[i * n + i]).sum::<f64>();
    let (na, nb) = (trace(&a), trace(&b));
    // tr(A B) = sum_ij A_ij B_ji
    let overlap: f64 = (0..n).map(|i| (0..n).map(|j| a[i * n + j] * b[j * n + i]).sum::<f64>()).sum();
    let sz = 0.5 * (na - nb);
    sz * (sz + 1.0) + nb - overlap
}

/// The densities (alpha, beta) of the broken-symmetry start from the restricted orbitals `c`
/// (row-major n x n, column k = orbital k, every orbital) with energies `eps` and `nocc`
/// doubly occupied orbitals: the lowest `nocc` orbitals once per spin, with each frontier pair
/// (HOMO − k, LUMO + k), k < pairs, rotated: φ_H ← cos θ φ_H ± sin θ φ_L (+ alpha, − beta).
/// `pairs` is the smaller size of the degenerate HOMO and LUMO sets ([`DEGENERATE`]). `None`
/// without a HOMO or a LUMO.
pub fn mixed_start(n: usize, nocc: usize, eps: &[f64], c: &[f64], par: Par) -> Option<([Vec<f64>; 2], usize)> {
    if nocc == 0 || nocc >= n {
        return None;
    }
    let (homo, lumo) = (nocc - 1, nocc);
    let gh = (0..=homo).rev().take_while(|&i| eps[homo] - eps[i] <= DEGENERATE).count();
    let gl = (lumo..n).take_while(|&j| eps[j] - eps[lumo] <= DEGENERATE).count();
    let pairs = gh.min(gl);
    let (cs, sn) = (THETA.cos(), THETA.sin());
    let density = |sign: f64| {
        let mut co = vec![0.0; n * nocc];
        for i in 0..n {
            co[i * nocc..(i + 1) * nocc].copy_from_slice(&c[i * n..i * n + nocc]);
            for k in 0..pairs {
                let (h, l) = (homo - k, lumo + k);
                co[i * nocc + h] = cs * c[i * n + h] + sign * sn * c[i * n + l];
            }
        }
        let mut p = vec![0.0; n * n];
        let m = MatRef::from_row_major_slice(&co, n, nocc);
        matmul(MatMut::from_row_major_slice_mut(&mut p, n, n), Accum::Replace, m, m.transpose(), 1.0, par);
        p
    };
    Some(([density(1.0), density(-1.0)], pairs))
}

/// ⟨S²⟩ of a 2.0.1 result (0 with one channel).
pub fn s2_v2(setup: &v2::scf::Setup, r: &v2::scf::ScfResult) -> f64 {
    match r.state.p.as_slice() {
        [pa, pb] => s2(setup.nao, &setup.s, pa, pb, setup.par),
        _ => 0.0,
    }
}

/// ⟨S²⟩ of a v1.1 result (0 with one channel).
pub fn s2_v1(setup: &Setup, r: &ScfResult) -> f64 {
    match r.density.as_slice() {
        [pa, pb] => s2(setup.nao, &setup.s, pa, pb, setup.par),
        _ => 0.0,
    }
}

/// The restricted state of an unrestricted 2.0.1 state: its total density and moments.
pub(crate) fn restricted_v2(st: &v2::scf::State) -> v2::scf::State {
    let p = match st.p.as_slice() {
        [pa, pb] => vec![pa.iter().zip(pb).map(|(a, b)| a + b).collect()],
        _ => st.p.clone(),
    };
    v2::scf::State { p, mag: None, ..st.clone() }
}

/// The SCF of one version as the procedure uses it.
trait Engine {
    type Res;
    type Start;
    fn set_unrestricted(&mut self, on: bool) -> Result<()>;
    /// The SCF from `start` (a restart: the restart's thresholds), every orbital with `all`.
    fn run(&self, opts: &ScfOptions, start: Option<&Self::Start>, all: bool) -> Result<Self::Res>;
    fn energy(r: &Self::Res) -> f64;
    fn iterations(r: &Self::Res) -> usize;
    fn s2(&self, r: &Self::Res) -> f64;
    /// The HOMO-LUMO mixed start of a restricted result with every orbital, and its pairs.
    fn mixed(&self, r: &Self::Res) -> Option<(Self::Start, usize)>;
    /// The restricted start of an unrestricted result (its total density).
    fn total(&self, r: &Self::Res) -> Self::Start;
    fn channels(s: &Self::Start) -> usize;
}

impl Engine for v2::scf::Setup {
    type Res = v2::scf::ScfResult;
    type Start = v2::scf::State;
    fn set_unrestricted(&mut self, on: bool) -> Result<()> {
        v2::scf::Setup::set_unrestricted(self, on)
    }
    fn run(&self, opts: &ScfOptions, start: Option<&Self::Start>, all: bool) -> Result<Self::Res> {
        let mut o = v2_options(opts, start.is_some());
        o.keep_orbitals |= all;
        self.scf_from(&o, start)
    }
    fn energy(r: &Self::Res) -> f64 {
        r.energies.total
    }
    fn iterations(r: &Self::Res) -> usize {
        r.iterations
    }
    fn s2(&self, r: &Self::Res) -> f64 {
        s2_v2(self, r)
    }
    fn mixed(&self, r: &Self::Res) -> Option<(Self::Start, usize)> {
        if !(r.all_orbitals && r.coefficients.len() == 1) {
            return None;
        }
        let nocc = self.nalpha.round() as usize;
        let ([pa, pb], pairs) = mixed_start(self.nao, nocc, &r.orbital_energies[0], &r.coefficients[0], self.par)?;
        Some((self.state_of(vec![pa, pb]), pairs))
    }
    fn total(&self, r: &Self::Res) -> Self::Start {
        restricted_v2(&r.state)
    }
    fn channels(s: &Self::Start) -> usize {
        s.p.len()
    }
}

impl Engine for Setup {
    type Res = ScfResult;
    type Start = Vec<Vec<f64>>;
    fn set_unrestricted(&mut self, on: bool) -> Result<()> {
        Setup::set_unrestricted(self, on)
    }
    fn run(&self, opts: &ScfOptions, start: Option<&Self::Start>, _all: bool) -> Result<Self::Res> {
        // v1.1 computes every orbital, and its restarts keep the program's thresholds
        self.scf_from(opts, start.map(Vec::as_slice))
    }
    fn energy(r: &Self::Res) -> f64 {
        r.energies.total
    }
    fn iterations(r: &Self::Res) -> usize {
        r.iterations
    }
    fn s2(&self, r: &Self::Res) -> f64 {
        s2_v1(self, r)
    }
    fn mixed(&self, r: &Self::Res) -> Option<(Self::Start, usize)> {
        let ([pa, pb], pairs) =
            mixed_start(self.nao, self.nalpha, &r.orbital_energies[0], &r.coefficients[0], self.par)?;
        Some((vec![pa, pb], pairs))
    }
    fn total(&self, r: &Self::Res) -> Self::Start {
        match r.density.as_slice() {
            [pa, pb] => vec![pa.iter().zip(pb).map(|(a, b)| a + b).collect()],
            p => p.to_vec(),
        }
    }
    fn channels(s: &Self::Start) -> usize {
        s.len()
    }
}

/// Whether an SCF error is a failure to converge.
fn not_converged(e: &Error) -> bool {
    matches!(e, Error::ScfNotConverged { .. })
}

/// The procedure at one geometry (module documentation).
struct Procedure<'a> {
    opts: &'a ScfOptions,
    tol: f64,
    info: BrokenSymmetry,
}

impl<'a> Procedure<'a> {
    fn new(opts: &'a ScfOptions) -> Self {
        Self { opts, tol: ENERGY_TOL * opts.accuracy.max(1.0), info: BrokenSymmetry::new() }
    }

    /// One SCF with `channels` spin channels at `etemp`, recorded.
    fn scf<E: Engine>(
        &mut self,
        e: &mut E,
        channels: usize,
        start: Option<&E::Start>,
        label: &'static str,
        etemp: f64,
    ) -> Result<E::Res> {
        e.set_unrestricted(channels == 2)?;
        let o = ScfOptions { etemp, ..self.opts.clone() };
        let r = e.run(&o, start, channels == 1);
        let (iterations, converged, energy, s2) = match &r {
            Ok(r) => (E::iterations(r), true, Some(E::energy(r)), (channels == 2).then(|| e.s2(r))),
            Err(Error::ScfNotConverged { iterations, .. }) => (*iterations, false, None, None),
            Err(_) => (0, false, None, None),
        };
        self.info.scfs.push(ScfRun { channels, start: label, etemp, iterations, converged, energy, s2 });
        if let (Some(en), Some(s2)) = (energy, s2) {
            if self.info.unrestricted_energy.is_none_or(|u| en < u) {
                self.info.unrestricted_energy = Some(en);
                self.info.unrestricted_s2 = Some(s2);
                self.info.start = Some(label);
            }
        }
        if channels == 1 && etemp == self.opts.etemp {
            if let Some(en) = energy {
                self.info.restricted_energy = Some(en);
            }
        }
        r
    }

    /// The result and the information, with the iteration sums.
    fn finish<R>(mut self, r: R, outcome: Outcome, s2: f64) -> (R, BrokenSymmetry) {
        self.info.outcome = outcome;
        self.info.s2 = s2;
        let sum = |channels: usize| {
            let it: Vec<usize> =
                self.info.scfs.iter().filter(|s| s.channels == channels).map(|s| s.iterations).collect();
            (!it.is_empty()).then(|| it.iter().sum())
        };
        self.info.restricted_iterations = sum(1);
        self.info.unrestricted_iterations = sum(2);
        (r, self.info)
    }

    fn run<E: Engine>(mut self, e: &mut E, start: Option<&E::Start>) -> Result<(E::Res, BrokenSymmetry)> {
        let t = self.opts.etemp;
        if let Some(st) = start.filter(|s| E::channels(s) == 2) {
            let u = self.scf(e, 2, Some(st), "previous", t)?;
            let s2u = e.s2(&u);
            if s2u >= S2_CHECK {
                return Ok(self.finish(u, Outcome::Broken, s2u));
            }
            // the magnetization is collapsing: the restricted solution from the total density,
            // and from its orbitals a new HOMO-LUMO start (another broken solution may exist)
            let r = self.scf(e, 1, Some(&e.total(&u)), "total-density", t)?;
            let prior = (E::energy(&u) < E::energy(&r) - self.tol).then_some((u, s2u));
            return self.mix_and_compare(e, r, prior, Outcome::Collapsed);
        }
        let label = if start.is_some() { "previous" } else { "eeq-bc" };
        match self.scf(e, 1, start, label, t) {
            Ok(r) => self.mix_and_compare(e, r, None, Outcome::NotLower),
            Err(err) if not_converged(&err) && start.is_none() => self.fallback(e, err),
            Err(err) => Err(err),
        }
    }

    /// Steps 2-4 from the restricted result `r`; `prior` an unrestricted solution already below
    /// it, `otherwise` the outcome when neither ends lower.
    fn mix_and_compare<E: Engine>(
        mut self,
        e: &mut E,
        r: E::Res,
        prior: Option<(E::Res, f64)>,
        otherwise: Outcome,
    ) -> Result<(E::Res, BrokenSymmetry)> {
        let mut best = prior;
        let mut outcome = otherwise;
        match e.mixed(&r) {
            None if best.is_none() && otherwise == Outcome::NotLower => outcome = Outcome::NoFrontier,
            None => {}
            Some((st, pairs)) => {
                self.info.pairs = pairs;
                match self.scf(e, 2, Some(&st), "homo-lumo", self.opts.etemp) {
                    Ok(u) => {
                        let s2u = e.s2(&u);
                        let below = E::energy(&u) < E::energy(&r) - self.tol;
                        if below && best.as_ref().is_none_or(|(b, _)| E::energy(&u) < E::energy(b)) {
                            best = Some((u, s2u));
                        }
                    }
                    Err(err) if not_converged(&err) => {
                        if otherwise == Outcome::NotLower {
                            outcome = Outcome::NotConverged;
                        }
                    }
                    Err(err) => return Err(err),
                }
            }
        }
        Ok(match best {
            Some((u, s2u)) => self.finish(u, Outcome::Broken, s2u),
            None => self.finish(r, outcome, 0.0),
        })
    }

    /// The restricted SCF of a fresh start did not converge: the HOMO-LUMO start from the
    /// orbitals of a smeared restricted SCF ([`FALLBACK_ETEMPS`]); `err` if nothing converges.
    fn fallback<E: Engine>(mut self, e: &mut E, err: Error) -> Result<(E::Res, BrokenSymmetry)> {
        let t = self.opts.etemp;
        for ts in FALLBACK_ETEMPS.into_iter().filter(|&ts| ts > t) {
            let rs = match self.scf(e, 1, None, "eeq-bc", ts) {
                Ok(r) => r,
                Err(x) if not_converged(&x) => continue,
                Err(x) => return Err(x),
            };
            let Some((st, pairs)) = e.mixed(&rs) else { break };
            let u = match self.scf(e, 2, Some(&st), "homo-lumo", t) {
                Ok(u) => u,
                Err(x) if not_converged(&x) => continue,
                Err(x) => return Err(x),
            };
            self.info.pairs = pairs;
            self.info.start_etemp = Some(ts);
            let s2u = e.s2(&u);
            if s2u >= S2_CHECK {
                return Ok(self.finish(u, Outcome::Broken, s2u));
            }
            // little magnetization: compare with the restricted solution from its total density
            return match self.scf(e, 1, Some(&e.total(&u)), "total-density", t) {
                Ok(r) if E::energy(&u) < E::energy(&r) - self.tol => Ok(self.finish(u, Outcome::Broken, s2u)),
                Ok(r) => Ok(self.finish(r, Outcome::NotLower, 0.0)),
                Err(x) if not_converged(&x) => Err(err),
                Err(x) => Err(x),
            };
        }
        Err(err)
    }
}

/// The SCF of a 2.0.1 setup of a closed-shell molecule with the broken-symmetry treatment (see
/// the module documentation) from `start` (the previous geometry's state, restricted or
/// unrestricted; `None`: the program's start). `opts` are the calculator's (their
/// `keep_orbitals` applies to the returned solution). Errors: those of the restricted SCF from
/// a given start, and of an unrestricted SCF restarted from an unrestricted `start` (the caller
/// then starts over from the program's start). The setup is left with one spin channel.
pub fn scf_v2(
    setup: &mut v2::scf::Setup,
    opts: &ScfOptions,
    start: Option<&v2::scf::State>,
) -> Result<(v2::scf::ScfResult, BrokenSymmetry)> {
    let out = Procedure::new(opts).run(setup, start);
    v2::scf::Setup::set_unrestricted(setup, false)?;
    out
}

/// [`scf_v2`] for a v1.1 setup: states are spin densities (closed shell: the total density),
/// and every SCF keeps the program's thresholds (v1.1 restarts do). The setup is left with
/// one spin channel.
pub fn scf_v1(setup: &mut Setup, opts: &ScfOptions, start: Option<&[Vec<f64>]>) -> Result<(ScfResult, BrokenSymmetry)> {
    let start = start.map(<[Vec<f64>]>::to_vec);
    let out = Procedure::new(opts).run(setup, start.as_ref());
    Setup::set_unrestricted(setup, false)?;
    out
}
