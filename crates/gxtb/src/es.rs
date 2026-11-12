//! Isotropic charge-dependent energies of g-xTB v1.1: first to fourth order in the shell
//! charges (program routine `electro_`), verified against the program's ES1 and ES2+3 at its
//! converged density to ~1e-16 Eh (reference tests).
//!
//! Shell charges q_l = n^ref_l - Mulliken population, atomic charges q_A = sum_l q_l, with
//! the shell list in basis order (`fock::shell_list`), the g-xTB CN and the second-order
//! kernel gamma2 (full nsh x nsh).
//!
//! - First order ("ES1", P-Eqs. 25-28):
//!   E1 = sum_l mu_l q_l f(q_A) - sum_{l in A} sum_{l' in B != A} drho_{l'} gamma2_{l l'} q_l,
//!   mu_l = gam1_l (1 + e1cn_A sqrt(CN_A)),
//!   f(q) = 1 + 0.012 [erf(q - 2/3) + erf(q + 2/3)] (k_dis = 0.012, k_x = 1, k_s = 2/3),
//!   drho_l = Z_l - n^ref_l (Aufbau valence electrons `com::zeff_shell` minus the reference
//!   occupation).
//! - Second order: E2 = 1/2 sum q_l gamma2_{l l'} q_l'.
//! - Third order: E3 = 1/3 sum_{l, l'} q_l q_l' q_A(l) Gamma_A(l) tau_{l l'} with the kernel
//!   [`crate::kernels::tau3`], Gamma_A = `gp3_gam3` (compiled in; no shell scaling).
//! - Fourth order: E4 = k4/4 sum_l q_l^4 with k4 = 0.021 (shell-resolved, constant).
//!   "ES2+3" of the program is E2 + E3 + E4.

use crate::data::{GAM3, ZEFF_SHELL};
use crate::data_runtime::REF_SHELL_OCC;
use crate::eeqbc::erf_f64;
use crate::params::GxtbParams;
use axtb::{Molecule, Result};

/// Fourth-order constant (per shell).
pub const K4: f64 = 0.021;
/// First-order charge switch: amplitude and offset.
pub const K1_DIS: f64 = 0.012;
pub const K1_S: f64 = 0.666666666;

/// Atomic charges from shell charges ([`axtb_base::math::atom_charges`]).
pub fn atom_charges(nat: usize, shells: &[(usize, usize, usize)], q: &[f64]) -> Vec<f64> {
    axtb_base::math::atom_charges(nat, shells.iter().map(|s| s.0), q)
}

/// Reference density change drho_l = Z_l - n^ref_l per shell.
pub fn reference_drho(mol: &Molecule, shells: &[(usize, usize, usize)]) -> Vec<f64> {
    shells
        .iter()
        .map(|&(iat, ish, _)| {
            let z = mol.numbers[iat];
            ZEFF_SHELL[ish * 103 + z - 1] as f64 - REF_SHELL_OCC[z - 1][ish]
        })
        .collect()
}

/// First-order energy.
pub fn first_order_energy(
    param: &GxtbParams,
    mol: &Molecule,
    shells: &[(usize, usize, usize)],
    cn: &[f64],
    q: &[f64],
    gamma2: &[f64],
) -> Result<f64> {
    let n = shells.len();
    let qa = atom_charges(mol.nat(), shells, q);
    let drho = reference_drho(mol, shells);
    let mut e = 0.0;
    for (i, &(iat, ish, _)) in shells.iter().enumerate() {
        let el = param.element(mol.numbers[iat])?;
        let mu = el.gam1[ish] * (1.0 + el.e1cn * cn[iat].sqrt());
        let f = 1.0 + K1_DIS * (erf_f64(qa[iat] - K1_S) + erf_f64(qa[iat] + K1_S));
        e += mu * q[i] * f;
        for (j, &(jat, _, _)) in shells.iter().enumerate() {
            if jat != iat {
                e -= drho[j] * gamma2[i * n + j] * q[i];
            }
        }
    }
    Ok(e)
}

/// Second-, third- and fourth-order energies (E2, E3, E4).
pub fn higher_order_energies(
    mol: &Molecule,
    shells: &[(usize, usize, usize)],
    q: &[f64],
    gamma2: &[f64],
    tau: &[f64],
) -> (f64, f64, f64) {
    let n = shells.len();
    let qa = atom_charges(mol.nat(), shells, q);
    let (mut e2, mut e3, mut e4) = (0.0, 0.0, 0.0);
    for i in 0..n {
        let iat = shells[i].0;
        let gam = GAM3[mol.numbers[iat] - 1];
        for j in 0..n {
            let g = gamma2[i * n + j];
            e2 += 0.5 * q[i] * g * q[j];
            e3 += q[i] * q[j] * qa[iat] * gam * tau[i * n + j] / 3.0;
        }
        e4 += 0.25 * K4 * q[i].powi(4);
    }
    (e2, e3, e4)
}

/// Index (0-based) into the 10 spin constants of an element for a shell pair (l, l').
pub fn spin_index(la: usize, lb: usize) -> usize {
    const INDX: [usize; 16] = [1, 2, 3, 4, 2, 5, 6, 7, 3, 6, 8, 9, 4, 7, 9, 10];
    INDX[la * 4 + lb] - 1
}

/// Onsite spin polarization (P-Eqs. 34-35, `energy_os_`):
/// E = 1/2 sum_A sum_{l, l' in A} m_l m_l' atspin_A W0_{l l'}, with the Mulliken shell
/// magnetizations m_l (alpha minus beta populations) and the compiled-in W0
/// ([`crate::data::SPIN_W0`]).
pub fn spin_energy(param: &GxtbParams, mol: &Molecule, shells: &[(usize, usize, usize)], mag: &[f64]) -> Result<f64> {
    let mut e = 0.0;
    for (i, &(ia, _, li)) in shells.iter().enumerate() {
        let z = mol.numbers[ia];
        let k = param.element(z)?.atspin;
        for (j, &(ja, _, lj)) in shells.iter().enumerate() {
            if ja == ia {
                e += 0.5 * mag[i] * mag[j] * k * crate::data::SPIN_W0[(z - 1) * 10 + spin_index(li, lj)];
            }
        }
    }
    Ok(e)
}
