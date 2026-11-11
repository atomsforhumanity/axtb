//! Orbital occupations of g-xTB 2.0.1 at a finite electronic temperature (Fermi smearing,
//! the program's `--etemp`), per spin channel, and the electronic entropy term −TS, on the rules
//! of [`crate::occupations`].
//!
//! - Program mode reproduces the 2.0.1 program (tblite's `get_fermi_filling` as compiled into
//!   it; decoded from the binary, spec §4.17.1; [`occupations::fill_gxtb201`]): the 0 K Aufbau
//!   filling fixes the HOMO index h = ⌊n_σ⌋ (+1 when the fractional part exceeds ½), and the
//!   Fermi level is found by Newton steps for **h electrons**, not n_σ: the program's fractional
//!   electron count (Σ n⁰ − Q) is rounded per channel at any temperature above 0 K. Start at the
//!   midpoint of orbitals h and h+1; f = 0 for (ε − μ)/kT ≥ 50; Newton steps while Σ f' > √tiny;
//!   stop once |h − Σ f| ≤ 2⁻²⁶ (the occupations are those of the level before the last step) or
//!   after 200 steps. Channels with h = 0 keep the Aufbau occupations. The entropy term is
//!   tblite's ([`base::entropy_tblite`]). kB = 3.166808578545117e-6 Eh/K (xtb).
//! - The default conventions fill the integer electron count of each channel with the exact
//!   Fermi level (Newton steps safeguarded by bisection, to round-off;
//!   [`occupations::fill_exact`]), so that the free energy E − TS is variational in the
//!   occupations and its analytic derivatives need no occupation response; kB of CODATA 2022.
//!
//! The entropy term is −TS = kT Σ_σ Σ_i [f ln f + (1 − f) ln(1 − f)] (≤ 0), added to the
//! electronic energy (the program spreads it over the atoms' electronic energies, so it enters
//! the SCF energy, its convergence test and the total).

use crate::occupations::{self, EXACT_MAX_STEPS, NEWTON_GXTB201};
use axtb::units::{reference::KB_XTB_TBLITE, KB_HARTREE};
use axtb_base::occupations as base;

/// Boltzmann constant (Eh/K): the program's in program mode, CODATA 2022 otherwise.
pub fn boltzmann(program: bool) -> f64 {
    if program {
        KB_XTB_TBLITE
    } else {
        KB_HARTREE
    }
}

/// Occupations of one spin channel with `nel` (real) electrons over `n` levels at 0 K
/// ([`base::aufbau`]).
pub fn aufbau(nel: f64, n: usize) -> Vec<f64> {
    let mut occ = vec![0.0; n];
    base::aufbau(nel, &mut occ);
    occ
}

/// Occupations, Fermi level and entropy term of one channel with `nel` electrons over the
/// levels `emo` at `kt` (Eh; 0: Aufbau) in the given conventions.
pub fn fill_channel(nel: f64, kt: f64, emo: &[f64], program: bool) -> (Vec<f64>, Option<f64>, f64) {
    let mut occ = vec![0.0; emo.len()];
    if kt <= 0.0 {
        base::aufbau(nel, &mut occ);
        return (occ, None, 0.0);
    }
    if program {
        let mu = occupations::fill_gxtb201(nel, kt, emo, &mut occ, NEWTON_GXTB201);
        let ts = base::entropy_tblite(&occ, kt);
        (occ, mu, ts)
    } else {
        let mu = occupations::fill_exact(nel, kt, emo, &mut occ, EXACT_MAX_STEPS);
        let ts = occupations::entropy_exact(emo, mu, kt);
        (occ, Some(mu), ts)
    }
}
