//! Classical (density-independent) g-xTB v1.1 terms: the g-xTB coordination number, the
//! semi-classical repulsion and the atomic energy increments.
//!
//! Decoded from the 1.1.0 program (`scf_`, `repuls_`) and verified against its printed
//! "nuclear repulsion" and "atomic core increments" (reference tests).

use crate::params::GxtbParams;
use axtb::molecule::{norm2_sq, sub3};
use axtb::{Molecule, Result};
use axtb_base::ncoord::{CnModel, Erf};

/// The g-xTB coordination number (used by H0, repulsion, Hubbard parameters, ...):
/// CN_A = sum_B 1/2 [1 + erf(k (R_AB - Rc_AB) / Rc_AB)], Rc_AB = (rad_A + rad_B)/2 with the
/// element radii `rad` (bohr) and k = glob(11) (negative), all pairs, no cutoff. The pair loop
/// and the counting function are `axtb-base`'s ([`CnModel`], [`Erf`] with steepness −k: IEEE
/// rounding is symmetric in sign, so its −((−k)(R − Rc))/Rc is the program's k (R − Rc)/Rc
/// bit for bit).
pub fn gxtb_cn(param: &GxtbParams, mol: &Molecule) -> Result<Vec<f64>> {
    let k = param.glob[10];
    let rad: Vec<f64> = mol.numbers.iter().map(|&z| param.element(z).map(|e| e.rad)).collect::<Result<_>>()?;
    Ok(CnModel::new(Erf { k: -k }, |i: usize, j: usize| 0.5 * (rad[i] + rad[j])).compute(&mol.positions))
}

/// Semi-classical repulsion energy (Eh):
/// E = sum_{A<B} Z_A Z_B exp(-a_AB (R + R0_AB)^{3/2}) [1 + (k1_A + k1_B)/(2R) + k2/R^2 + k3/R^3 + k4/R^4]
/// with Z_A = rep3 (1 - rep5 q_A - rep6 q_A^2) (EEQ-BC charges q), a_A = rep1 (1 + rep4 sqrt(CN_A)),
/// a_AB = 2 a_A a_B / (a_A + a_B), R0_AB = sqrt(rep2_A rep2_B), k1 = glob(9) for H and He and
/// glob(4) otherwise, (k2, k3, k4) = glob(13..15).
pub fn repulsion_energy(param: &GxtbParams, mol: &Molecule, q: &[f64], cn: &[f64]) -> Result<f64> {
    let n = mol.nat();
    let g = &param.glob;
    let mut zeff = vec![0.0; n];
    let mut alpha = vec![0.0; n];
    let mut k1 = vec![0.0; n];
    for i in 0..n {
        let z = mol.numbers[i];
        let r = &param.element(z)?.rep;
        zeff[i] = (1.0 - r[4] * q[i] - r[5] * q[i] * q[i]) * r[2];
        alpha[i] = (r[3] * cn[i].sqrt() + 1.0) * r[0];
        k1[i] = if z <= 2 { g[8] } else { g[3] };
    }
    let mut e = 0.0;
    for i in 0..n {
        let ri = &param.element(mol.numbers[i])?.rep;
        for j in i + 1..n {
            let rj = &param.element(mol.numbers[j])?.rep;
            let r = norm2_sq(sub3(mol.positions[i], mol.positions[j])).sqrt();
            let a = 2.0 / (1.0 / alpha[j] + 1.0 / alpha[i]);
            let x = (ri[1] * rj[1]).sqrt() + r;
            let damp = (-(x.sqrt() * x * a)).exp();
            let r2 = r * r;
            let r3 = r * r2;
            let poly = (k1[j] + k1[i]) * 0.5 / r + 1.0 + g[12] / r2 + g[13] / r3 + g[14] / (r * r3);
            e += damp * zeff[i] * zeff[j] * poly;
        }
    }
    Ok(e)
}

/// Sum of the atomic energy increments (Eh), "atomic core increments" of the program.
pub fn increment_energy(param: &GxtbParams, mol: &Molecule) -> Result<f64> {
    mol.numbers.iter().map(|&z| param.element(z).map(|e| e.increment)).sum()
}
