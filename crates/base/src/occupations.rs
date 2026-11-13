// SPDX-License-Identifier: LGPL-3.0-or-later
//! Orbital occupations: the 0 K Aufbau filling, the Fermi smearing of tblite and xtb at a finite
//! electronic temperature and tblite's electronic entropy term −TS, written once for the
//! methods that use them.
//!
//! The rules differ in their tolerance, in whether the electron count is tested before or after
//! the Newton step and in the derivative's operand order; they are reproduction conventions,
//! so each is its own function here, with the parameters that differ explicit and its
//! arithmetic exactly as the program computes it:
//!
//! | rule | function and parameters | used by | from |
//! |---|---|---|---|
//! | tblite | [`fill_tblite`], [`Newton::XTB680`] | GFN2-xTB (every convention); the Hückel step of GFN-FF as xtb 6.8.0 computes it | tblite `get_fermi_filling`, xtb 6.8.0 `fermismear` (`src/scc_core.f90`) |
//! | xtb `fermismear` | [`fill_fermismear`], [`Newton::XTB671`] | the Hückel step of GFN-FF as xtb 6.7.1 computes it | xtb 6.7.1 `fermismear` |
//!
//! [`aufbau`] is the 0 K filling the searches start from (and tblite's HOMO index);
//! [`entropy_tblite`] is tblite's entropy term kT Σ ln(f^f (1 − f)^(1 − f)) (≤ 0);
//! [`fermismear_iterate`] is the Newton iteration of xtb's `fermismear` for other programs'
//! arithmetic. The g-xTB programs' rules and the exact filling of g-xTB's default conventions
//! live in `axtb-gxtb` (its `occupations` module), on these pieces.
//!
//! Which rule a method uses in which conventions is the method's choice, as is its Boltzmann
//! constant: the levels may be in any unit (GFN-FF's Hückel levels are in eV), and kT is in the
//! same unit ([`crate::units::KB_HARTREE`], [`crate::units::reference::KB_XTB_TBLITE`]).

/// When a Newton search for the Fermi level stops.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Newton {
    /// The search stops once the occupations hold the target electron count to within `tol`.
    pub tol: f64,
    /// At most this many evaluations of the occupations (each with its Newton step).
    pub max_steps: usize,
}

impl Newton {
    /// xtb 6.8.0's `fermismear` and tblite's `get_fermi_filling` as GFN2-xTB runs it:
    /// min(√ε, 1e5 ε) = 1e5 ε ≈ 2.2e-11 electrons, 200 steps.
    pub const XTB680: Newton = Newton { tol: 1e5 * f64::EPSILON, max_steps: 200 };
    /// xtb 6.7.1's `fermismear`: 1e-9 electrons, 200 steps.
    pub const XTB671: Newton = Newton { tol: 1e-9, max_steps: 200 };
}

/// The 0 K filling of `nel` electrons of one spin (`nel` > −1) over the levels of `occ`
/// (lowest first): 1 for the first ⌊nel⌋, the remainder for the next, 0 above. Returns
/// tblite's HOMO index (1-based): ⌊nel⌋, plus 1 when the remainder exceeds ½ (0: no HOMO).
pub fn aufbau(nel: f64, occ: &mut [f64]) -> usize {
    let n = occ.len();
    let full = nel.floor() as usize;
    for (i, o) in occ.iter_mut().enumerate() {
        *o = if i < full { 1.0 } else { 0.0 };
    }
    // nel % 1 = nel − ⌊nel⌋ exactly for nel > −1
    let rem = nel % 1.0;
    if full < n {
        occ[full] = rem;
    }
    full + usize::from(rem > 0.5)
}

/// tblite's and xtb 6.8.0's Fermi function at `arg = (ε − ε_F)/kT`: the occupation and its
/// derivative with respect to ε, −f (1 − f)/kT (exactly 0 or 1 with zero derivative beyond
/// |arg| > 50).
#[inline]
pub fn fermi(arg: f64, kt: f64) -> (f64, f64) {
    if arg > 50.0 {
        (0.0, 0.0)
    } else if arg < -50.0 {
        (1.0, 0.0)
    } else {
        let f = 1.0 / (arg.exp() + 1.0);
        (f, -f * (1.0 - f) / kt)
    }
}

/// tblite's Fermi filling of `nel` (real) electrons of one spin over the levels `emo` at `kt`
/// (GFN2-xTB; xtb 6.8.0's `fermismear`). `occ` (as long as `emo`) receives the Aufbau
/// occupations, and when `nel` > 0 and `kt` > 0 the Fermi-smeared ones. Newton steps for the
/// Fermi level with the target `nel`, from the midpoint of the HOMO and the level above it
/// ([`aufbau`]'s index): the Fermi function [`fermi`]; stop when |nel − Σ f| ≤ `newton.tol`
/// (tested before the step), when Σ f' ≤ √tiny, or after `newton.max_steps`. Returns the Fermi
/// level of the occupations, the last one evaluated (`None` for the Aufbau occupations).
pub fn fill_tblite(nel: f64, kt: f64, emo: &[f64], occ: &mut [f64], newton: Newton) -> Option<f64> {
    let homo = aufbau(nel, occ);
    if !(nel > 0.0 && kt > 0.0) {
        return None;
    }
    let nao = emo.len();
    let sqrttiny = f64::MIN_POSITIVE.sqrt();
    let i0 = homo.max(1) - 1;
    let i1 = (homo + 1).min(nao) - 1;
    let mut ef = 0.5 * (emo[i0] + emo[i1]);
    let mut ef_used = ef;
    for _ in 0..newton.max_steps {
        let mut total = 0.0;
        let mut dtotal = 0.0;
        ef_used = ef;
        for i in 0..nao {
            let (f, df) = fermi((emo[i] - ef) / kt, kt);
            occ[i] = f;
            total += f;
            dtotal -= df;
        }
        if (nel - total).abs() <= newton.tol {
            break;
        }
        if dtotal > sqrttiny {
            ef += (nel - total) / dtotal;
        } else {
            break;
        }
    }
    Some(ef_used)
}

/// xtb 6.7.1's `fermismear`: the Fermi-smeared occupations `occ` (as long as `emo`,
/// overwritten) of `nel` electrons of one spin over the levels `emo` at `kt`, for an integer
/// count with 1 ≤ `nel` < `emo.len()` (the callers keep the Aufbau occupations otherwise).
/// [`fermismear_iterate`] with the levels summed in order, f = 1/(eˣ + 1) and
/// f' = eˣ/(kT ((eˣ + 1)(eˣ + 1))) for x = (ε − μ)/kT < 50, else f = f' = 0 (no lower cutoff), and
/// the step μ += (nel − Σ f)/Σ f'. Returns the Fermi level the occupations were computed with.
pub fn fill_fermismear(nel: usize, kt: f64, emo: &[f64], occ: &mut [f64], newton: Newton) -> f64 {
    fermismear_iterate(nel, emo, newton, |ef| {
        let mut total = 0.0;
        let mut dtotal = 0.0;
        for i in 0..emo.len() {
            let x = (emo[i] - ef) / kt;
            let (f, df) = if x < 50.0 {
                let ex = x.exp();
                (1.0 / (ex + 1.0), ex / (kt * (ex + 1.0).powi(2)))
            } else {
                (0.0, 0.0)
            };
            occ[i] = f;
            total += f;
            dtotal += df;
        }
        (total, dtotal)
    })
}

/// The Newton iteration of xtb's `fermismear` for an integer count `nel` (1 ≤ `nel` <
/// `emo.len()`) over the levels `emo`, with one program's arithmetic in `count`: from
/// μ = ½ (ε_nel + ε_nel+1), `count(μ)` writes the occupations at μ and returns Σ f and the
/// denominator of the step (xtb: Σ f'); μ += (nel − Σ f)/denominator, then stop when
/// |nel − Σ f| ≤ `newton.tol` (tested after the step, so the occupations are those before it)
/// or after `newton.max_steps`. Returns the Fermi level the last occupations were computed with
/// (the programs keep the level after one more step).
#[inline]
pub fn fermismear_iterate(nel: usize, emo: &[f64], newton: Newton, mut count: impl FnMut(f64) -> (f64, f64)) -> f64 {
    let occt = nel as f64;
    let mut ef = 0.5 * (emo[nel - 1] + emo[nel]);
    let mut ef_used = ef;
    for _ in 0..newton.max_steps {
        ef_used = ef;
        let (total, denominator) = count(ef);
        let diff = occt - total;
        ef += diff / denominator;
        if diff.abs() <= newton.tol {
            break;
        }
    }
    ef_used
}

/// tblite's entropy term of one spin's occupations at `kt`: kT Σ ln(f^f (1 − f)^(1 − f)) (≤ 0;
/// GFN2-xTB, the g-xTB 2.0.1 program).
pub fn entropy_tblite(occ: &[f64], kt: f64) -> f64 {
    occ.iter().map(|&o| (o.powf(o) * (1.0 - o).powf(1.0 - o)).ln()).sum::<f64>() * kt
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::reference::KB_XTB_TBLITE;

    #[test]
    fn tolerance_is_the_programs_expression() {
        assert_eq!(Newton::XTB680.tol.to_bits(), f64::EPSILON.sqrt().min(1e5 * f64::EPSILON).to_bits());
    }

    #[test]
    fn aufbau_fills_and_finds_the_homo() {
        let mut occ = [9.0; 5];
        assert_eq!(aufbau(2.75, &mut occ), 3);
        assert_eq!(occ, [1.0, 1.0, 0.75, 0.0, 0.0]);
        assert_eq!(aufbau(2.25, &mut occ), 2);
        assert_eq!(aufbau(7.0, &mut occ), 7);
        assert_eq!(occ, [1.0; 5]);
        assert_eq!(aufbau(0.0, &mut occ), 0);
        assert_eq!(occ, [0.0; 5]);
        // a count just below zero (g-xTB 2.0.1's program count with every electron unpaired)
        assert_eq!(aufbau(-2e-11, &mut occ), 0);
        assert_eq!(occ[0], -2e-11);
    }

    #[test]
    fn newton_searches_place_the_electrons() {
        let emo = [-1.07, -0.54, -0.48, -0.47, 0.29, 0.32];
        let kt = 3000.0 * KB_XTB_TBLITE;
        let mut occ = [0.0; 6];
        assert!(fill_tblite(3.0, kt, &emo, &mut occ, Newton::XTB680).is_some());
        assert!((occ.iter().sum::<f64>() - 3.0).abs() <= Newton::XTB680.tol);
        assert!(occ[2] < 1.0 && occ[3] > 0.0);
        assert!(entropy_tblite(&occ, kt) < 0.0);
        fill_fermismear(3, kt, &emo, &mut occ, Newton::XTB671);
        assert!((occ.iter().sum::<f64>() - 3.0).abs() <= Newton::XTB671.tol);
        // no smearing without electrons or temperature: the Aufbau occupations
        assert!(fill_tblite(0.0, kt, &emo, &mut occ, Newton::XTB680).is_none());
        assert!(fill_tblite(3.5, 0.0, &emo, &mut occ, Newton::XTB680).is_none());
        assert_eq!(occ, [1.0, 1.0, 1.0, 0.5, 0.0, 0.0]);
        // integer occupations carry no entropy
        aufbau(3.0, &mut occ);
        assert_eq!(entropy_tblite(&occ, kt), 0.0);
    }
}
