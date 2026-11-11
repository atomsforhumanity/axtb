//! Orbital occupations of g-xTB v1.1 at a finite electronic temperature (the 1.1.0 program's
//! `-tel <K>`, Fermi smearing) and the entropy term −TS, on the rules of [`crate::occupations`].
//!
//! Program mode reproduces the 1.1.0 program (`solve_` → `occu_`, `fermismear_`; decoded from
//! the binary, spec §4.17): smearing when T > 1e-6 K, per spin channel with at least one
//! occupied and one virtual orbital, on that channel's levels (closed shell: both electron
//! spins on the one set of levels, occupations summed), with the integer electron count N_σ
//! ([`occupations::fill_gxtb110`] with [`Fermismear::GXTB110`]: Newton steps from
//! ½ (ε_N + ε_N+1), f' = eˣ/(kT (1 + eˣ)²), μ += (N − Σf)/(Σf' + 1e-14), stop when
//! |N − Σf| ≤ 1e-9 or after 200 steps, two SSE2 lanes). kB: xtb's 3.166808578545117e-6 Eh/K
//! rounded to single precision ([`occupations::KB_GXTB110`]). The entropy term is
//! kT Σ [f ln f + (1 − f) ln(1 − f)] over the levels with f > 1e-9 and 1 − f > 1e-9
//! ([`occupations::entropy_thresholded`]). The default conventions fill exactly
//! ([`occupations::fill_exact`], CODATA 2022 kB).

use crate::occupations::{self, Fermismear};
use axtb_base::occupations as base;

/// Electronic temperature (K) above which the program smears.
pub const TEL_MIN: f64 = 1e-6;

/// Levels with f or 1 − f at or below this are left out of the program's entropy term.
const ENTROPY_THR: f64 = 1e-9;

/// Occupations (per electron spin, 0/1 scale), Fermi level and entropy term of one channel
/// with `nel` electrons over `emo` at `kt` (0: Aufbau).
pub fn fill_channel(nel: usize, kt: f64, emo: &[f64], program: bool) -> (Vec<f64>, Option<f64>, f64) {
    let n = emo.len();
    let mut occ = vec![0.0; n];
    if kt <= 0.0 || nel == 0 || nel >= n {
        base::aufbau(nel as f64, &mut occ);
        return (occ, None, 0.0);
    }
    if program {
        let ef = occupations::fill_gxtb110(nel, kt, emo, &mut occ, Fermismear::GXTB110);
        let ts = occupations::entropy_thresholded(&occ, kt, ENTROPY_THR);
        (occ, Some(ef), ts)
    } else {
        let mu = occupations::fill_exact(nel as f64, kt, emo, &mut occ, occupations::EXACT_MAX_STEPS);
        let ts = occupations::entropy_exact(emo, mu, kt);
        (occ, Some(mu), ts)
    }
}
