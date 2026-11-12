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
//! At the next geometry a calculator whose previous state is unrestricted runs only the
//! unrestricted SCF from that state. When its ⟨S²⟩ falls below [`S2_CHECK`] (the magnetization
//! is collapsing onto the restricted solution), the restricted SCF runs too, from the total
//! density, and step 4 decides; a restricted result sends the next geometry back to step 1.
//! A restricted previous state always runs steps 1-4.
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

/// An unrestricted SCF restarted from the previous geometry's broken-symmetry state is
/// compared with the restricted solution when its ⟨S²⟩ is below this; above, it is kept.
pub const S2_CHECK: f64 = 0.05;

/// Which solution a broken-symmetry calculation returned, and why.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// The unrestricted (broken-symmetry) solution: it lies lower than the restricted one, or
    /// it was restarted from a broken-symmetry state and kept its magnetization.
    Broken,
    /// The restricted solution: the unrestricted SCF from the HOMO-LUMO start ended within
    /// [`ENERGY_TOL`] of it (the symmetry did not break) or above it.
    NotLower,
    /// The restricted solution: the unrestricted SCF restarted from the previous geometry's
    /// broken-symmetry state collapsed onto it.
    Collapsed,
    /// The restricted solution: the unrestricted SCF did not converge.
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

/// What the broken-symmetry treatment did at one geometry.
#[derive(Clone, Debug, PartialEq)]
pub struct BrokenSymmetry {
    pub outcome: Outcome,
    /// ⟨S²⟩ of the returned solution (0 for the restricted one).
    pub s2: f64,
    /// Start of the unrestricted SCF: `"homo-lumo"` (the mixed restricted orbitals) or
    /// `"previous"` (the previous geometry's broken-symmetry state); `None`: none ran.
    pub start: Option<&'static str>,
    /// Frontier orbital pairs rotated in the HOMO-LUMO start (0 without one).
    pub pairs: usize,
    /// Energies (Eh), ⟨S²⟩ and SCF iterations of the solutions computed at this geometry
    /// (`None`: not computed; the unrestricted ones also when the SCF did not converge).
    pub restricted_energy: Option<f64>,
    pub unrestricted_energy: Option<f64>,
    pub unrestricted_s2: Option<f64>,
    pub restricted_iterations: Option<usize>,
    pub unrestricted_iterations: Option<usize>,
}

impl BrokenSymmetry {
    /// Whether the result is the unrestricted (broken-symmetry) solution.
    pub fn broken(&self) -> bool {
        self.outcome == Outcome::Broken
    }

    fn new(outcome: Outcome) -> Self {
        Self {
            outcome,
            s2: 0.0,
            start: None,
            pairs: 0,
            restricted_energy: None,
            unrestricted_energy: None,
            unrestricted_s2: None,
            restricted_iterations: None,
            unrestricted_iterations: None,
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

/// The energy tolerance of the acceptance rule for SCF accuracy `accuracy`.
fn tolerance(accuracy: f64) -> f64 {
    ENERGY_TOL * accuracy.max(1.0)
}

/// Whether an SCF error is a failure to converge.
fn not_converged(e: &Error) -> bool {
    matches!(e, Error::ScfNotConverged { .. })
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

/// The SCF of a 2.0.1 setup of a closed-shell molecule with the broken-symmetry treatment (see
/// the module documentation) from `start` (the previous geometry's state, restricted or
/// unrestricted; `None`: the program's start). `opts` are the calculator's (their
/// `keep_orbitals` applies to the returned solution). Errors: those of the restricted SCF, and
/// of an unrestricted SCF restarted from an unrestricted `start` (the caller then starts over
/// from the program's start). The setup is left with one spin channel.
pub fn scf_v2(
    setup: &mut v2::scf::Setup,
    opts: &ScfOptions,
    start: Option<&v2::scf::State>,
) -> Result<(v2::scf::ScfResult, BrokenSymmetry)> {
    let out = scf_v2_inner(setup, opts, start);
    setup.set_unrestricted(false)?;
    out
}

fn scf_v2_inner(
    setup: &mut v2::scf::Setup,
    opts: &ScfOptions,
    start: Option<&v2::scf::State>,
) -> Result<(v2::scf::ScfResult, BrokenSymmetry)> {
    let restart = v2_options(opts, true);
    let tol = tolerance(opts.accuracy);
    if let Some(st) = start.filter(|s| s.p.len() == 2) {
        setup.set_unrestricted(true)?;
        let u = setup.scf_from(&restart, Some(st))?;
        let s2u = s2_v2(setup, &u);
        let mut info = BrokenSymmetry {
            start: Some("previous"),
            unrestricted_energy: Some(u.energies.total),
            unrestricted_s2: Some(s2u),
            unrestricted_iterations: Some(u.iterations),
            ..BrokenSymmetry::new(Outcome::Broken)
        };
        if s2u >= S2_CHECK {
            info.s2 = s2u;
            return Ok((u, info));
        }
        // the magnetization is collapsing: the restricted solution from the total density
        setup.set_unrestricted(false)?;
        let r = setup.scf_from(&restart, Some(&restricted_v2(&u.state)))?;
        info.restricted_energy = Some(r.energies.total);
        info.restricted_iterations = Some(r.iterations);
        if u.energies.total < r.energies.total - tol {
            info.s2 = s2u;
            return Ok((u, info));
        }
        info.outcome = Outcome::Collapsed;
        return Ok((r, info));
    }
    setup.set_unrestricted(false)?;
    // every orbital of the restricted solution, for the LUMO
    let ropts = ScfOptions { keep_orbitals: true, ..if start.is_some() { restart.clone() } else { opts.clone() } };
    let r = setup.scf_from(&ropts, start)?;
    let mut info = BrokenSymmetry {
        restricted_energy: Some(r.energies.total),
        restricted_iterations: Some(r.iterations),
        ..BrokenSymmetry::new(Outcome::NoFrontier)
    };
    let nocc = setup.nalpha.round() as usize;
    let mixed = if r.all_orbitals && r.coefficients.len() == 1 {
        mixed_start(setup.nao, nocc, &r.orbital_energies[0], &r.coefficients[0], setup.par)
    } else {
        None
    };
    let Some(([pa, pb], pairs)) = mixed else { return Ok((r, info)) };
    info.start = Some("homo-lumo");
    info.pairs = pairs;
    let st = setup.state_of(vec![pa, pb]);
    setup.set_unrestricted(true)?;
    let u = setup.scf_from(&restart, Some(&st));
    let u = match u {
        Ok(u) => u,
        Err(e) if not_converged(&e) => {
            if let Error::ScfNotConverged { iterations, .. } = e {
                info.unrestricted_iterations = Some(iterations);
            }
            info.outcome = Outcome::NotConverged;
            return Ok((r, info));
        }
        Err(e) => return Err(e),
    };
    let s2u = s2_v2(setup, &u);
    info.unrestricted_energy = Some(u.energies.total);
    info.unrestricted_s2 = Some(s2u);
    info.unrestricted_iterations = Some(u.iterations);
    if u.energies.total < r.energies.total - tol {
        info.outcome = Outcome::Broken;
        info.s2 = s2u;
        Ok((u, info))
    } else {
        info.outcome = Outcome::NotLower;
        Ok((r, info))
    }
}

/// [`scf_v2`] for a v1.1 setup: states are spin densities (closed shell: the total density),
/// and every SCF keeps the program's thresholds (v1.1 restarts do). The setup is left with
/// one spin channel.
pub fn scf_v1(setup: &mut Setup, opts: &ScfOptions, start: Option<&[Vec<f64>]>) -> Result<(ScfResult, BrokenSymmetry)> {
    let out = scf_v1_inner(setup, opts, start);
    setup.set_unrestricted(false)?;
    out
}

fn scf_v1_inner(
    setup: &mut Setup,
    opts: &ScfOptions,
    start: Option<&[Vec<f64>]>,
) -> Result<(ScfResult, BrokenSymmetry)> {
    let tol = tolerance(opts.accuracy);
    if let Some(p) = start.filter(|p| p.len() == 2) {
        setup.set_unrestricted(true)?;
        let u = setup.scf_from(opts, Some(p))?;
        let s2u = s2_v1(setup, &u);
        let mut info = BrokenSymmetry {
            start: Some("previous"),
            unrestricted_energy: Some(u.energies.total),
            unrestricted_s2: Some(s2u),
            unrestricted_iterations: Some(u.iterations),
            ..BrokenSymmetry::new(Outcome::Broken)
        };
        if s2u >= S2_CHECK {
            info.s2 = s2u;
            return Ok((u, info));
        }
        setup.set_unrestricted(false)?;
        let ptot: Vec<f64> = u.density[0].iter().zip(&u.density[1]).map(|(a, b)| a + b).collect();
        let r = setup.scf_from(opts, Some(&[ptot]))?;
        info.restricted_energy = Some(r.energies.total);
        info.restricted_iterations = Some(r.iterations);
        if u.energies.total < r.energies.total - tol {
            info.s2 = s2u;
            return Ok((u, info));
        }
        info.outcome = Outcome::Collapsed;
        return Ok((r, info));
    }
    setup.set_unrestricted(false)?;
    let r = setup.scf_from(opts, start)?;
    let mut info = BrokenSymmetry {
        restricted_energy: Some(r.energies.total),
        restricted_iterations: Some(r.iterations),
        ..BrokenSymmetry::new(Outcome::NoFrontier)
    };
    let Some(([pa, pb], pairs)) =
        mixed_start(setup.nao, setup.nalpha, &r.orbital_energies[0], &r.coefficients[0], setup.par)
    else {
        return Ok((r, info));
    };
    info.start = Some("homo-lumo");
    info.pairs = pairs;
    setup.set_unrestricted(true)?;
    let u = setup.scf_from(opts, Some(&[pa, pb]));
    let u = match u {
        Ok(u) => u,
        Err(e) if not_converged(&e) => {
            if let Error::ScfNotConverged { iterations, .. } = e {
                info.unrestricted_iterations = Some(iterations);
            }
            info.outcome = Outcome::NotConverged;
            return Ok((r, info));
        }
        Err(e) => return Err(e),
    };
    let s2u = s2_v1(setup, &u);
    info.unrestricted_energy = Some(u.energies.total);
    info.unrestricted_s2 = Some(s2u);
    info.unrestricted_iterations = Some(u.iterations);
    if u.energies.total < r.energies.total - tol {
        info.outcome = Outcome::Broken;
        info.s2 = s2u;
        Ok((u, info))
    } else {
        info.outcome = Outcome::NotLower;
        Ok((r, info))
    }
}
