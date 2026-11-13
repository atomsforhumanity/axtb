// SPDX-License-Identifier: LGPL-3.0-or-later
//! Orbital occupations of g-xTB at a finite electronic temperature, shared by both versions:
//! the g-xTB programs' Fermi smearing and the exact filling of the default conventions, with
//! their entropy terms. tblite's and xtb's rules, the Aufbau filling and the Newton iteration of
//! xtb's `fermismear` come from [`axtb_base::occupations`]; [`crate::fermi`] chooses among these per
//! version and conventions.
//!
//! | rule | function and parameters | used by | from |
//! |---|---|---|---|
//! | g-xTB 1.1.0 | [`fill_gxtb110`], [`Fermismear::GXTB110`], kB [`KB_GXTB110`] | v1.1, program conventions | the 1.1.0 program's `fermismear_` (decoded from the binary; `docs/gxtb-v1.1-spec.md` §4.17) |
//! | g-xTB 2.0.1 | [`fill_gxtb201`], [`NEWTON_GXTB201`] | 2.0.1, program conventions | tblite's `get_fermi_filling` as compiled into the 2.0.1 program (`docs/gxtb-v2.0.1-spec.md` §4.17.1) |
//! | exact | [`fill_exact`], [`EXACT_MAX_STEPS`] | both versions, default conventions | axtb: the Fermi level to round-off, so that E − TS is variational in the occupations |
//!
//! Entropy terms: [`entropy_thresholded`] (the 1.1.0 program), [`entropy_exact`] (the exact
//! filling); the 2.0.1 program's is tblite's ([`axtb_base::occupations::entropy_tblite`]).
//!
//! The exact filling is AFH's own; only g-xTB's default conventions use it.

use axtb_base::occupations::{aufbau, fermismear_iterate, Newton};

/// Boltzmann constant (Eh/K) of the g-xTB 1.1.0 program: xtb's kB as a single-precision
/// literal (`real(sp)`, read from its `fermismear_`), 3.1668084830016596e-6.
pub const KB_GXTB110: f64 = 3.166_808_5e-6_f32 as f64;

/// √ε of `f64` (2⁻²⁶), the electron-count tolerance of tblite as compiled into g-xTB 2.0.1.
const SQRT_EPSILON: f64 = 1.490_116_119_384_765_6e-8;

/// The Newton search of the g-xTB 2.0.1 program: √ε = 2⁻²⁶ ≈ 1.49e-8 electrons, 200 steps.
pub const NEWTON_GXTB201: Newton = Newton { tol: SQRT_EPSILON, max_steps: 200 };

/// Steps of the exact search ([`fill_exact`]); it reaches round-off long before.
pub const EXACT_MAX_STEPS: usize = 500;

/// The g-xTB 1.1.0 program's `fermismear_` ([`fill_gxtb110`]): its stopping rule and the guard
/// it adds to Σ f' in the step's denominator.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Fermismear {
    pub newton: Newton,
    /// Added to Σ f' in the denominator of every step.
    pub guard: f64,
}

impl Fermismear {
    /// The program's values: xtb's 1e-9 electrons and 200 steps, guard 1e-14.
    pub const GXTB110: Fermismear = Fermismear { newton: Newton::XTB671, guard: 1e-14 };
}

/// The g-xTB 1.1.0 program's Fermi smearing (`fermismear_`) of `nel` electrons of one spin
/// over the levels `emo` at `kt`, for an integer count with 1 ≤ `nel` < `emo.len()`: the
/// occupations `occ` (as long as `emo`, overwritten) and the Fermi level they were computed
/// with. xtb's Newton iteration ([`fermismear_iterate`]) with the binary's arithmetic: two SSE2
/// lanes over the first 2⌊n/2⌋ levels (even and odd levels summed separately, then added), the
/// remainder in order; f = 1/(1 + eˣ) and f' = eˣ/((kT (1 + eˣ))(1 + eˣ)) for x = (ε − μ)/kT < 50,
/// else 0; μ += (nel − Σ f)/(Σ f' + `rule.guard`).
pub fn fill_gxtb110(nel: usize, kt: f64, emo: &[f64], occ: &mut [f64], rule: Fermismear) -> f64 {
    fermismear_iterate(nel, emo, rule.newton, |ef| {
        let (total, dtotal) = fermismear_sums_gxtb110(ef, kt, emo, occ);
        (total, dtotal + rule.guard)
    })
}

/// Occupations at the Fermi level `ef`, Σ f and Σ f' of the g-xTB 1.1.0 program's
/// `fermismear_` (two SSE2 lanes over the first 2⌊n/2⌋ levels, then the remainder).
#[inline]
fn fermismear_sums_gxtb110(ef: f64, kt: f64, emo: &[f64], occ: &mut [f64]) -> (f64, f64) {
    let n = emo.len();
    let n2 = n & !1;
    let (mut t0, mut t1, mut d0, mut d1) = (0.0, 0.0, 0.0, 0.0);
    let fermi = |e: f64| -> (f64, f64) {
        let x = (e - ef) / kt;
        if x < 50.0 {
            let ex = x.exp();
            (1.0 / (1.0 + ex), ex / (kt * (1.0 + ex) * (1.0 + ex)))
        } else {
            (0.0, 0.0)
        }
    };
    for i in (0..n2).step_by(2) {
        let (fa, da) = fermi(emo[i]);
        let (fb, db) = fermi(emo[i + 1]);
        occ[i] = fa;
        occ[i + 1] = fb;
        t0 += fa;
        t1 += fb;
        d0 += da;
        d1 += db;
    }
    let mut total = t0 + t1;
    let mut dtotal = d0 + d1;
    for i in n2..n {
        let (f, d) = fermi(emo[i]);
        occ[i] = f;
        total += f;
        dtotal += d;
    }
    (total, dtotal)
}

/// The g-xTB 1.1.0 program's entropy term of one spin's occupations at `kt`:
/// kT Σ [f ln f + (1 − f) ln(1 − f)] over the levels with f > `thr` and 1 − f > `thr`
/// (the program: 1e-9; ≤ 0).
pub fn entropy_thresholded(occ: &[f64], kt: f64, thr: f64) -> f64 {
    let mut s = 0.0;
    for &f in occ {
        if f > thr && 1.0 - f > thr {
            let g = 1.0 - f;
            s += f * f.ln() + g * g.ln();
        }
    }
    kt * s
}

/// One Fermi function evaluation of the g-xTB 2.0.1 program: occupation and its derivative
/// with respect to the level, `vector` selecting the vectorized (AVX, separate multiplications)
/// or the scalar remainder (fused multiply-adds) arithmetic of the binary.
#[inline]
fn fermi_gxtb201(arg: f64, kt: f64, vector: bool) -> (f64, f64) {
    if arg < 50.0 {
        let ex = arg.exp();
        let f = 1.0 / (ex + 1.0);
        let df = if vector {
            let e1 = ex + 1.0;
            ex / (e1 * (e1 * kt))
        } else {
            let k = kt.mul_add(ex, kt);
            ex / ex.mul_add(k, k)
        };
        (f, df)
    } else {
        (0.0, 0.0)
    }
}

/// The g-xTB 2.0.1 program's filling of `nel` (real) electrons of one spin over the levels
/// `emo` at `kt` (tblite's `get_fermi_filling` as compiled into it). `occ` (as long as `emo`)
/// receives the Aufbau occupations, which fix the HOMO index h ([`aufbau`]); when h > 0 and
/// `kt` > 0, Newton steps for the Fermi level with the target **h** (not `nel`: the program's
/// fractional count is rounded) from ½ (ε_h + ε_min(h+1, n)): f = f' = 0 for x = (ε − μ)/kT ≥ 50,
/// μ += (h − Σ f)/Σ f' when Σ f' > √tiny, then stop when |h − Σ f| ≤ `newton.tol` (tested after
/// the step, so the occupations are those of the level before it) or after `newton.max_steps`.
/// The binary sums the first 4⌊n/4⌋ levels in four AVX lanes, reduced (l0 + l2) + (l1 + l3),
/// and the remainder in order with fused multiply-adds in f'. Returns the Fermi level of the
/// occupations (`None` for the Aufbau occupations).
pub fn fill_gxtb201(nel: f64, kt: f64, emo: &[f64], occ: &mut [f64], newton: Newton) -> Option<f64> {
    let n = emo.len();
    let h = aufbau(nel, occ);
    if h == 0 || kt <= 0.0 || n == 0 {
        return None;
    }
    let occt = h as f64;
    let sqrttiny = f64::MIN_POSITIVE.sqrt();
    let mut ef = 0.5 * (emo[h.max(1) - 1] + emo[(h + 1).min(n) - 1]);
    let n4 = n & !3;
    let mut ef_used = ef;
    for _ in 0..newton.max_steps {
        ef_used = ef;
        let mut lt = [0.0; 4];
        let mut ld = [0.0; 4];
        for i in 0..n4 {
            let (f, df) = fermi_gxtb201((emo[i] - ef) / kt, kt, true);
            occ[i] = f;
            lt[i % 4] += f;
            ld[i % 4] += df;
        }
        let mut total = (lt[0] + lt[2]) + (lt[1] + lt[3]);
        let mut dtotal = (ld[0] + ld[2]) + (ld[1] + ld[3]);
        for i in n4..n {
            let (f, df) = fermi_gxtb201((emo[i] - ef) / kt, kt, false);
            occ[i] = f;
            total += f;
            dtotal += df;
        }
        let diff = occt - total;
        if dtotal > sqrttiny {
            ef += diff / dtotal;
        }
        if diff.abs() <= newton.tol {
            break;
        }
    }
    Some(ef_used)
}

/// Fermi occupation f = 1/(1 + eˣ) and 1 − f = eˣ/(1 + eˣ), both without cancellation
/// (the exact filling).
#[inline]
pub fn fermi_pair(x: f64) -> (f64, f64) {
    if x > 0.0 {
        let e = (-x).exp();
        (e / (1.0 + e), 1.0 / (1.0 + e))
    } else {
        let e = x.exp();
        (1.0 / (1.0 + e), e / (1.0 + e))
    }
}

/// Exact filling of `nel` (real) electrons of one spin over the levels `emo` at `kt` > 0:
/// the occupations `occ` (as long as `emo`) at the Fermi level μ with Σ f = nel to round-off,
/// and μ. Σ f increases monotonically with μ: a bracket around the solution, then Newton steps
/// from the midpoint of the levels ⌈nel⌉ and ⌈nel⌉ + 1, replaced by bisection when they leave
/// the bracket; stop when |Σ f − nel| ≤ 4 ε max(nel, 1), when the step leaves μ unchanged or the
/// bracket is narrower than 2 ε max(|μ|, kT), or after `max_steps` ([`EXACT_MAX_STEPS`]).
/// No electrons: empty levels and μ 50 kT below the first; every level full: μ 50 kT above the
/// last.
pub fn fill_exact(nel: f64, kt: f64, emo: &[f64], occ: &mut [f64], max_steps: usize) -> f64 {
    let n = emo.len();
    let count = |mu: f64| -> (f64, f64) {
        let mut s = 0.0;
        let mut ds = 0.0;
        for &e in emo {
            let (f, g) = fermi_pair((e - mu) / kt);
            s += f;
            ds += f * g / kt;
        }
        (s, ds)
    };
    if n == 0 || nel <= 0.0 {
        occ.fill(0.0);
        return emo.first().copied().unwrap_or(0.0) - 50.0 * kt;
    }
    if nel >= n as f64 {
        occ.fill(1.0);
        return emo[n - 1] + 50.0 * kt;
    }
    let emin = emo.iter().copied().fold(f64::INFINITY, f64::min);
    let emax = emo.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let (mut lo, mut hi) = (emin - kt, emax + kt);
    while count(lo).0 > nel {
        lo -= (hi - lo).max(kt);
    }
    while count(hi).0 < nel {
        hi += (hi - lo).max(kt);
    }
    let k = nel.ceil() as usize;
    let mut mu = if k >= 1 && k < n { 0.5 * (emo[k - 1] + emo[k]) } else { 0.5 * (lo + hi) };
    if !(mu > lo && mu < hi) {
        mu = 0.5 * (lo + hi);
    }
    let tol = 4.0 * f64::EPSILON * nel.max(1.0);
    for _ in 0..max_steps {
        let (s, ds) = count(mu);
        let r = s - nel;
        if r.abs() <= tol {
            break;
        }
        if r > 0.0 {
            hi = mu;
        } else {
            lo = mu;
        }
        let newton = if ds > 0.0 { mu - r / ds } else { f64::NAN };
        let next = if newton > lo && newton < hi { newton } else { 0.5 * (lo + hi) };
        if next == mu || hi - lo <= 2.0 * f64::EPSILON * mu.abs().max(kt) {
            mu = next;
            break;
        }
        mu = next;
    }
    for (o, &e) in occ.iter_mut().zip(emo) {
        *o = fermi_pair((e - mu) / kt).0;
    }
    mu
}

/// Entropy term kT Σ [f ln f + (1 − f) ln(1 − f)] of the levels `emo` filled at the Fermi level
/// `mu` (the exact filling; f and 1 − f from [`fermi_pair`]; ≤ 0).
pub fn entropy_exact(emo: &[f64], mu: f64, kt: f64) -> f64 {
    let mut s = 0.0;
    for &e in emo {
        let (f, g) = fermi_pair((e - mu) / kt);
        if f > 0.0 {
            s += f * f.ln();
        }
        if g > 0.0 {
            s += g * g.ln();
        }
    }
    kt * s
}

#[cfg(test)]
mod tests {
    use super::*;
    use axtb::units::reference::KB_XTB_TBLITE;
    use axtb_base::occupations::entropy_tblite;

    #[test]
    fn gxtb110_boltzmann_constant() {
        assert_eq!(KB_GXTB110, 3.1668084830016596e-6);
    }

    #[test]
    fn gxtb201_tolerance_is_sqrt_epsilon() {
        assert_eq!(NEWTON_GXTB201.tol.to_bits(), f64::EPSILON.sqrt().to_bits());
    }

    #[test]
    fn exact_filling_places_the_electrons() {
        let emo = [-0.9, -0.5, -0.31, -0.30, -0.29, 0.1, 0.2];
        let mut occ = [0.0; 7];
        for kt in [1e-4, 3e-3, 1.6e-2, 0.2] {
            for nel in [1.0, 3.0, 4.0, 6.0] {
                let mu = fill_exact(nel, kt, &emo, &mut occ, EXACT_MAX_STEPS);
                let s: f64 = occ.iter().sum();
                assert!((s - nel).abs() < 1e-13, "kt {kt} nel {nel}: {s}");
                assert!(entropy_exact(&emo, mu, kt) <= 0.0);
            }
        }
    }

    #[test]
    fn gxtb110_filling_places_the_electrons() {
        let emo = [-1.07, -0.54, -0.48, -0.47, 0.29, 0.32, 0.4];
        let mut occ = [0.0; 7];
        fill_gxtb110(3, 3000.0 * KB_GXTB110, &emo, &mut occ, Fermismear::GXTB110);
        assert!((occ.iter().sum::<f64>() - 3.0).abs() <= 1e-9);
        assert!(occ[2] < 1.0 && occ[3] > 0.0);
    }

    #[test]
    fn gxtb201_filling_rounds_the_electron_count() {
        // the program's fractional count (water: 3.99999999997510 per channel) fills 4 electrons
        let emo = [-1.07, -0.54, -0.48, -0.42, 0.29, 0.32];
        let mut occ = [0.0; 6];
        let mu = fill_gxtb201(3.9999999999751, 3000.0 * KB_XTB_TBLITE, &emo, &mut occ, NEWTON_GXTB201);
        assert!(mu.is_some());
        assert_eq!(occ.iter().sum::<f64>(), 4.0);
        // 1.4 electrons: HOMO 1, one electron
        fill_gxtb201(1.4, 0.01, &emo, &mut occ, NEWTON_GXTB201);
        assert!((occ.iter().sum::<f64>() - 1.0).abs() < 1e-8);
        // no electrons in the channel: Aufbau kept
        let mu = fill_gxtb201(0.3, 0.01, &emo, &mut occ, NEWTON_GXTB201);
        assert!(mu.is_none() && occ[0] == 0.3);
    }

    #[test]
    fn entropies_agree_where_no_level_is_cut() {
        let occ = [1.0 - 1e-3, 0.7, 0.3, 1e-3];
        let kt = 0.01;
        let a = entropy_tblite(&occ, kt);
        let b = entropy_thresholded(&occ, kt, 1e-9);
        assert!(a < 0.0 && (a - b).abs() < 1e-15, "{a} {b}");
    }
}
