//! Interaction kernels of the charge-dependent terms (`setgab_lrao_` of the 1.1.0 program).
//!
//! Shell Hubbard parameters U_l = gam2_l gam_Z (1 + g2cn_Z sqrt(CN)) (`com::gam` element
//! hardness, g-xTB CN), Mulliken-exchange Hubbard parameters U^x_l = fam_l gam_Z.
//!
//! - Second order (P-Eq. 30): gamma2_ij = 1/(R + exp(-glob(12) R)/eta), eta = 2/(1/U_i + 1/U_j);
//!   onsite (same atom) the program evaluates this at R = 1e-12 bohr.
//! - Mulliken-approximated range-separated exchange (P-Eq. 42), per AO pair of shells i, j:
//!   gamma^x = (alpha + (1 - alpha) erf(omega R)) / (R + exp(-(glob(7) + glob(8) R0_AB) R)/(eta~ g)),
//!   alpha = 0.15, omega = glob(19), R0_AB the D3 pair radius in Ångström,
//!   eta~ = 2^(xi-1) (U^x_i U^x_j)^(xi/2) / (U^x_i + U^x_j)^(xi-1) with xi = 2 if either shell is
//!   a polarization shell (`bascom::bas_pol`) and 1 otherwise, and g = 1.39 for the same shell
//!   of the same atom, else g = sqrt(G_li G_lj), G = (glob(5), glob(6), glob(6), glob(6)).

use crate::data::{BAS_POL, GAM};
use crate::eeqbc::{erf_f64, vdw_rad_pair};
use crate::params::GxtbParams;
use axtb::molecule::{norm2_sq, sub3};
use axtb::{Molecule, Result};

/// Short-range exact-exchange admixture of the MFX kernel.
pub const ALPHA_FX: f64 = 0.15;
/// Distance the program stores for an atom with itself (diagonal of its packed distance
/// array); it enters the onsite second-order kernel.
pub const R_SAME_ATOM: f64 = 1e-12;

/// Onsite same-shell factor of the MFX kernel.
pub const G_ONSITE: f64 = 1.39;

/// Shell Hubbard parameters of the second-order term.
pub fn hubbard_u(param: &GxtbParams, mol: &Molecule, shells: &[(usize, usize, usize)], cn: &[f64]) -> Result<Vec<f64>> {
    shells
        .iter()
        .map(|&(iat, ish, _)| {
            let z = mol.numbers[iat];
            let e = param.element(z)?;
            Ok(e.gam2[ish] * ((e.g2cn * cn[iat].sqrt() + 1.0) * GAM[z - 1]))
        })
        .collect()
}

/// Second-order shell kernel (full nsh x nsh).
pub fn gamma2(param: &GxtbParams, mol: &Molecule, shells: &[(usize, usize, usize)], u: &[f64]) -> Vec<f64> {
    let n = shells.len();
    let k = param.glob[11];
    let mut g = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let (ia, ja) = (shells[i].0, shells[j].0);
            let r = if ia == ja { R_SAME_ATOM } else { norm2_sq(sub3(mol.positions[ia], mol.positions[ja])).sqrt() };
            let inv_eta = 1.0 / (2.0 / (1.0 / u[i] + 1.0 / u[j]));
            let v = 1.0 / (r + inv_eta * (-k * r).exp());
            g[i * n + j] = v;
            g[j * n + i] = v;
        }
    }
    g
}

/// Third-order kernel tau (`setgab_lrao_`, array next to gamma2): within an atom
/// tau = -1/2 gamma2^2 (= dgamma2/d(1/eta) at R -> 0), between atoms
/// tau = 1/2 R e^{-1.15 x} (3.2 - 1.84 x)/eta with x = R/eta and eta as in [`gamma2`].
pub fn tau3(mol: &Molecule, shells: &[(usize, usize, usize)], u: &[f64], gamma2: &[f64]) -> Vec<f64> {
    let n = shells.len();
    let mut t = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let (ia, ja) = (shells[i].0, shells[j].0);
            let v = if ia == ja {
                let g = gamma2[i * n + j];
                -0.5 * g * g
            } else {
                let r = norm2_sq(sub3(mol.positions[ia], mol.positions[ja])).sqrt();
                let inv_eta = 1.0 / (2.0 / (1.0 / u[i] + 1.0 / u[j]));
                let e = (-1.15 * r * inv_eta).exp();
                0.5 * e * r * ((-1.84 * r * inv_eta + 3.2) * inv_eta)
            };
            t[i * n + j] = v;
            t[j * n + i] = v;
        }
    }
    t
}

fn uwe_av(a: f64, b: f64, xi: f64) -> f64 {
    (xi - 1.0).exp2() * (a * b).powf(0.5 * xi) / (a + b).powf(xi - 1.0)
}

/// Mulliken-exchange kernel per shell pair (full nsh x nsh; AOs of a shell share the value).
pub fn gamma_mfx(param: &GxtbParams, mol: &Molecule, shells: &[(usize, usize, usize)]) -> Result<Vec<f64>> {
    let n = shells.len();
    let g = &param.glob;
    let gl = [g[4], g[5], g[5], g[5]];
    let omega = g[18];
    let mut ux = Vec::with_capacity(n);
    let mut pol = Vec::with_capacity(n);
    for &(iat, ish, _) in shells {
        let z = mol.numbers[iat];
        ux.push(param.element(z)?.fam[ish] * GAM[z - 1]);
        pol.push(BAS_POL[ish * 103 + z - 1]);
    }
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let (ia, ja) = (shells[i].0, shells[j].0);
            let xi = (pol[i].max(pol[j]) + 1) as f64;
            let eta = uwe_av(ux[i], ux[j], xi);
            let gg = if i == j { G_ONSITE } else { (gl[shells[i].2] * gl[shells[j].2]).sqrt() };
            let r = if ia == ja { R_SAME_ATOM } else { norm2_sq(sub3(mol.positions[ia], mol.positions[ja])).sqrt() };
            let r0 = vdw_rad_pair(mol.numbers[ia], mol.numbers[ja]);
            let damp = (-(g[6] + g[7] * r0) * r).exp();
            let v = (ALPHA_FX + (1.0 - ALPHA_FX) * erf_f64(omega * r)) / (r + damp / (eta * gg));
            out[i * n + j] = v;
            out[j * n + i] = v;
        }
    }
    Ok(out)
}
