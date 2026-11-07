//! Orbital occupations of g-xTB v1.1 at a finite electronic temperature (the 1.1.0 program's
//! `-tel <K>`, Fermi smearing) and the entropy term −TS.
//!
//! Program mode reproduces the 1.1.0 program (`solve_` → `occu_`, `fermismear_`; decoded from
//! the binary, spec §4.17): smearing when T > 1e-6 K, per spin channel with at least one
//! occupied and one virtual orbital, on that channel's levels (closed shell: both electron
//! spins on the one set of levels, occupations summed). Newton steps for the Fermi level with
//! the integer electron count N_σ: start at ½ (ε_N + ε_N+1); f = 1/(eˣ + 1) and
//! f' = eˣ/(kT (1 + eˣ)²) for x = (ε − μ)/kT < 50, else 0; μ += (N − Σf)/(Σf' + 1e-14); stop
//! when |N − Σf| ≤ 1e-9 (the occupations are those before the last step) or after 200 steps.
//! kB: xtb's 3.166808578545117e-6 Eh/K rounded to single precision (3.1668084830016596e-6). The entropy
//! term is kT Σ [f ln f + (1 − f) ln(1 − f)] over the levels with f > 1e-9 and 1 − f > 1e-9.
//! The default conventions fill exactly ([`crate::v2::fermi::exact_filling`], CODATA 2022 kB).

/// The 1.1.0 program's Boltzmann constant (Eh/K): xtb's kB (3.166808578545117e-6) as a
/// single-precision value (read from `fermismear_`).
pub const KB_PROGRAM: f64 = 3.166_808_5e-6_f32 as f64;

/// Electronic temperature (K) above which the program smears.
pub const TEL_MIN: f64 = 1e-6;

/// The program's Fermi smearing (`fermismear_`) of `nel` electrons over the levels `emo`
/// (1 <= nel < emo.len()): occupations and the Fermi level they were computed with (the
/// program returns the level after one more step).
pub fn program_filling(nel: usize, kt: f64, emo: &[f64]) -> (Vec<f64>, f64) {
    let n = emo.len();
    let mut occ = vec![0.0; n];
    let occt = nel as f64;
    let mut ef = 0.5 * (emo[nel - 1] + emo[nel]);
    let mut ef_used = ef;
    let n2 = n & !1;
    for _ in 0..200 {
        ef_used = ef;
        // two SSE2 lanes over the first 2⌊n/2⌋ levels, then the remainder
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
        let diff = occt - total;
        ef += diff / (dtotal + 1e-14);
        if diff.abs() <= 1e-9 {
            break;
        }
    }
    (occ, ef_used)
}

/// The program's entropy term of one channel (≤ 0).
pub fn program_entropy(occ: &[f64], kt: f64) -> f64 {
    let thr = 1e-9;
    let mut s = 0.0;
    for &f in occ {
        if f > thr && 1.0 - f > thr {
            let g = 1.0 - f;
            s += f * f.ln() + g * g.ln();
        }
    }
    kt * s
}

/// Occupations (per electron spin, 0/1 scale), Fermi level and entropy term of one channel
/// with `nel` electrons over `emo` at `kt` (0: Aufbau).
pub fn fill_channel(nel: usize, kt: f64, emo: &[f64], program: bool) -> (Vec<f64>, Option<f64>, f64) {
    let n = emo.len();
    if kt <= 0.0 || nel == 0 || nel >= n {
        return ((0..n).map(|i| if i < nel { 1.0 } else { 0.0 }).collect(), None, 0.0);
    }
    if program {
        let (occ, ef) = program_filling(nel, kt, emo);
        let ts = program_entropy(&occ, kt);
        (occ, Some(ef), ts)
    } else {
        let (occ, mu) = crate::v2::fermi::exact_filling(nel as f64, kt, emo);
        let ts = crate::v2::fermi::exact_entropy(emo, mu, kt);
        (occ, Some(mu), ts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn program_constant() {
        assert_eq!(KB_PROGRAM, 3.1668084830016596e-6);
    }
}
