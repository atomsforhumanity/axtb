//! Orbital occupations of g-xTB 2.0.1 at a finite electronic temperature (Fermi smearing,
//! the program's `--etemp`), per spin channel, and the electronic entropy term −TS.
//!
//! - Program mode reproduces the 2.0.1 program (tblite's `get_fermi_filling` as compiled into
//!   it; decoded from the binary, spec §4.17): the 0 K Aufbau filling fixes the HOMO index
//!   h = ⌊n_σ⌋ (+1 when the fractional part exceeds ½), and the Fermi level is found by Newton
//!   steps for **h electrons**, not n_σ: the program's fractional electron count (Σ n⁰ − Q) is
//!   rounded per channel at any temperature above 0 K. Start at the midpoint of orbitals h and
//!   h+1; f = 0 for (ε − μ)/kT ≥ 50; Newton steps while Σ f' > √tiny; stop once |h − Σ f| ≤
//!   2⁻²⁶ (the occupations are those of the level before the last step) or after 200 steps.
//!   Channels with h = 0 keep the Aufbau occupations. kB = 3.166808578545117e-6 Eh/K (xtb).
//! - The default conventions fill the integer electron count of each channel with the exact
//!   Fermi level (Newton steps safeguarded by bisection, to round-off), so that the free energy
//!   E − TS is variational in the occupations and its analytic derivatives need no occupation
//!   response; kB of CODATA 2022.
//!
//! The entropy term is −TS = kT Σ_σ Σ_i [f ln f + (1 − f) ln(1 − f)] (≤ 0), added to the
//! electronic energy (the program spreads it over the atoms' electronic energies, so it enters
//! the SCF energy, its convergence test and the total).

use axtb::units::{reference::KB_XTB_TBLITE, KB_HARTREE};

/// Boltzmann constant (Eh/K): the program's in program mode, CODATA 2022 otherwise.
pub fn boltzmann(program: bool) -> f64 {
    if program {
        KB_XTB_TBLITE
    } else {
        KB_HARTREE
    }
}

/// Occupations of one spin channel with `nel` (real) electrons over the levels `emo` at 0 K:
/// 1 for the first ⌊nel⌋, the remainder for the next; and tblite's HOMO index (1-based:
/// ⌊nel⌋, plus 1 when the remainder exceeds ½).
pub fn aufbau(nel: f64, n: usize) -> (Vec<f64>, i64) {
    let mut occ = vec![0.0; n];
    let full = nel.floor() as usize;
    for o in occ.iter_mut().take(full.min(n)) {
        *o = 1.0;
    }
    if full < n {
        occ[full] = nel - full as f64;
    }
    let homo = nel.floor() as i64 + i64::from(nel % 1.0 > 0.5);
    (occ, homo)
}

/// One Fermi function evaluation of the program: occupation and its derivative with respect
/// to the level, `vector` selecting the vectorized (AVX, separate multiplications) or the scalar
/// remainder (fused multiply-adds) arithmetic of the binary.
#[inline]
fn program_fermi(arg: f64, kt: f64, vector: bool) -> (f64, f64) {
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

/// The 2.0.1 program's filling of one channel (see the module documentation): occupations and
/// the Fermi level they were computed with (`None` when the Aufbau occupations are kept).
pub fn program_filling(nel: f64, kt: f64, emo: &[f64]) -> (Vec<f64>, Option<f64>) {
    let n = emo.len();
    let (mut occ, homo) = aufbau(nel, n);
    if homo <= 0 || kt <= 0.0 || n == 0 {
        return (occ, None);
    }
    let h = homo as usize;
    let occt = homo as f64;
    let thr = f64::EPSILON.sqrt();
    let sqrttiny = f64::MIN_POSITIVE.sqrt();
    let mut ef = 0.5 * (emo[h.max(1) - 1] + emo[(h + 1).min(n) - 1]);
    let n4 = n & !3;
    let mut ef_used = ef;
    for _ in 0..200 {
        ef_used = ef;
        // the binary sums the first 4⌊n/4⌋ levels in four lanes, reduced as (l0 + l2) + (l1 + l3),
        // then the remainder in order
        let mut lt = [0.0; 4];
        let mut ld = [0.0; 4];
        for i in 0..n4 {
            let (f, df) = program_fermi((emo[i] - ef) / kt, kt, true);
            occ[i] = f;
            lt[i % 4] += f;
            ld[i % 4] += df;
        }
        let mut total = (lt[0] + lt[2]) + (lt[1] + lt[3]);
        let mut dtotal = (ld[0] + ld[2]) + (ld[1] + ld[3]);
        for i in n4..n {
            let (f, df) = program_fermi((emo[i] - ef) / kt, kt, false);
            occ[i] = f;
            total += f;
            dtotal += df;
        }
        let diff = occt - total;
        if dtotal > sqrttiny {
            ef += diff / dtotal;
        }
        if diff.abs() <= thr {
            break;
        }
    }
    (occ, Some(ef_used))
}

/// The program's entropy term of one channel: kT Σ ln(f^f (1 − f)^(1 − f)).
pub fn program_entropy(occ: &[f64], kt: f64) -> f64 {
    occ.iter().map(|&o| (o.powf(o) * (1.0 - o).powf(1.0 - o)).ln()).sum::<f64>() * kt
}

/// Fermi occupation 1/(1 + e^x) and 1 − f = e^x/(1 + e^x), both without cancellation.
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

/// Exact filling of one channel with `nel` electrons at `kt` > 0 (default conventions):
/// occupations and the Fermi level μ with Σ f = nel to round-off.
pub fn exact_filling(nel: f64, kt: f64, emo: &[f64]) -> (Vec<f64>, f64) {
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
    let fill = |mu: f64| emo.iter().map(|&e| fermi_pair((e - mu) / kt).0).collect::<Vec<f64>>();
    if n == 0 || nel <= 0.0 {
        let mu = emo.first().copied().unwrap_or(0.0) - 50.0 * kt;
        return (vec![0.0; n], mu);
    }
    if nel >= n as f64 {
        let mu = emo[n - 1] + 50.0 * kt;
        return (vec![1.0; n], mu);
    }
    // bracket: Σ f increases monotonically with μ
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
    for _ in 0..500 {
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
    (fill(mu), mu)
}

/// Entropy term kT Σ [f ln f + (1 − f) ln(1 − f)] of levels `emo` filled at the Fermi level
/// `mu` (default conventions; ≤ 0).
pub fn exact_entropy(emo: &[f64], mu: f64, kt: f64) -> f64 {
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

/// Occupations, Fermi level and entropy term of one channel with `nel` electrons over the
/// levels `emo` at `kt` (Eh; 0: Aufbau) in the given conventions.
pub fn fill_channel(nel: f64, kt: f64, emo: &[f64], program: bool) -> (Vec<f64>, Option<f64>, f64) {
    if kt <= 0.0 {
        return (aufbau(nel, emo.len()).0, None, 0.0);
    }
    if program {
        let (occ, mu) = program_filling(nel, kt, emo);
        let ts = program_entropy(&occ, kt);
        (occ, mu, ts)
    } else {
        let (occ, mu) = exact_filling(nel, kt, emo);
        let ts = exact_entropy(emo, mu, kt);
        (occ, Some(mu), ts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_filling_places_the_electrons() {
        let emo = [-0.9, -0.5, -0.31, -0.30, -0.29, 0.1, 0.2];
        for kt in [1e-4, 3e-3, 1.6e-2, 0.2] {
            for nel in [1.0, 3.0, 4.0, 6.0] {
                let (occ, mu) = exact_filling(nel, kt, &emo);
                let s: f64 = occ.iter().sum();
                assert!((s - nel).abs() < 1e-13, "kt {kt} nel {nel}: {s}");
                // free energy stationary: d/dmu of the grand potential term vanishes at the solution
                let ts = exact_entropy(&emo, mu, kt);
                assert!(ts <= 0.0);
            }
        }
    }

    #[test]
    fn program_filling_rounds_the_electron_count() {
        // the program's fractional count (water: 3.99999999997510 per channel) fills 4 electrons
        let emo = [-1.07, -0.54, -0.48, -0.42, 0.29, 0.32];
        let (occ, mu) = program_filling(3.9999999999751, 3000.0 * KB_XTB_TBLITE, &emo);
        assert!(mu.is_some());
        assert_eq!(occ.iter().sum::<f64>(), 4.0);
        // 1.4 electrons: HOMO 1, one electron
        let (occ, _) = program_filling(1.4, 0.01, &emo);
        assert!((occ.iter().sum::<f64>() - 1.0).abs() < 1e-8);
        // no electrons in the channel: Aufbau kept
        let (occ, mu) = program_filling(0.3, 0.01, &emo);
        assert!(mu.is_none() && occ[0] == 0.3);
    }
}
