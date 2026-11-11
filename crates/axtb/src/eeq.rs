//! Electronegativity equilibration (EEQ) charges, EEQ 2019 model of D4
//! (E. Caldeweyher et al., J. Chem. Phys. 150, 154122 (2019); parameters from the
//! multicharge library). Used as the SCF starting guess, as the xtb program does.

use crate::linalg::Lu;
use crate::molecule::{norm2_sq, sub3, Molecule};
use crate::param::covalent_rad_d3;
use crate::param::eeq_data::{EEQ_CHI, EEQ_ETA, EEQ_KCNCHI, EEQ_RAD};
use crate::{Error, Result};
use axtb_d4::erf;
use faer::{Mat, Par};
use std::f64::consts::PI;

const CN_EXP: f64 = 7.5;

/// The coordination number of the EEQ model: error-function counting with covalent radii
/// `rcov` (bohr) up to `cutoff` (bohr), optionally capped smoothly at `cap`.
#[derive(Clone, Copy, Debug)]
pub struct EeqCn {
    pub rcov: fn(usize) -> f64,
    pub cutoff: f64,
    pub cap: Option<f64>,
}

impl EeqCn {
    /// multicharge's EEQ 2019 (dftd4, tblite): D3 radii, 25 bohr, capped at 8.
    pub const MULTICHARGE: EeqCn = EeqCn { rcov: covalent_rad_d3, cutoff: 25.0, cap: Some(8.0) };
}

/// EEQ 2019 partial charges (summing to the molecular charge) with the coordination number
/// `cnm` (multicharge's [`EeqCn::MULTICHARGE`]; the xtb program's guess, `goedecker_chrgeq`
/// with `ncoord_erf`, uses xtb's radii, 40 bohr and no cap).
pub fn eeq_charges_with(mol: &Molecule, cnm: &EeqCn) -> Result<Vec<f64>> {
    let n = mol.nat();
    if let Some(&z) = mol.numbers.iter().find(|&&z| z > EEQ_CHI.len()) {
        return Err(Error::UnsupportedElement { method: "EEQ", z });
    }
    // error-function coordination number, optionally capped smoothly
    let mut cn = vec![0.0; n];
    for i in 0..n {
        for j in 0..i {
            let r2 = norm2_sq(sub3(mol.positions[i], mol.positions[j]));
            if r2 > cnm.cutoff * cnm.cutoff || r2 < 1e-12 {
                continue;
            }
            let r = r2.sqrt();
            let rc = (cnm.rcov)(mol.numbers[i]) + (cnm.rcov)(mol.numbers[j]);
            let c = 0.5 * (1.0 + erf(-CN_EXP * (r - rc) / rc));
            cn[i] += c;
            cn[j] += c;
        }
    }
    if let Some(cap) = cnm.cap {
        for c in cn.iter_mut() {
            *c = (1.0 + cap.exp()).ln() - (1.0 + (cap - *c).exp()).ln();
        }
    }
    let sqrt2pi = (2.0 / PI).sqrt();
    let mut a = Mat::<f64>::zeros(n + 1, n + 1);
    let mut b = Mat::<f64>::zeros(n + 1, 1);
    for i in 0..n {
        let zi = mol.numbers[i] - 1;
        a[(i, i)] = EEQ_ETA[zi] + sqrt2pi / EEQ_RAD[zi];
        for j in 0..i {
            let zj = mol.numbers[j] - 1;
            let r2 = norm2_sq(sub3(mol.positions[i], mol.positions[j]));
            let gam = 1.0 / (EEQ_RAD[zi].powi(2) + EEQ_RAD[zj].powi(2));
            let v = erf((r2 * gam).sqrt()) / r2.sqrt();
            a[(i, j)] = v;
            a[(j, i)] = v;
        }
        a[(i, n)] = 1.0;
        a[(n, i)] = 1.0;
        b[(i, 0)] = -EEQ_CHI[zi] + EEQ_KCNCHI[zi] / (cn[i] + 1e-14).sqrt() * cn[i];
    }
    b[(n, 0)] = mol.charge;
    // Sequential LU on purpose. faer's high-level `a.partial_piv_lu().solve(..)` runs on faer's
    // *global* parallelism (default: rayon's global pool, one thread per visible CPU), so every
    // process started rayon's pool even at threads=1: 126 threads in a 126-CPU cluster pod. A
    // singular matrix gives charges that are not finite, the error below.
    Lu::new(a, Par::Seq).solve(b.as_mut());
    let q: Vec<f64> = (0..n).map(|i| b[(i, 0)]).collect();
    if q.iter().any(|v| !v.is_finite()) {
        return Err(Error::Numerical("EEQ charges are not finite".into()));
    }
    Ok(q)
}
