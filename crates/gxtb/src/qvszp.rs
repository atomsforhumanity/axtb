//! The q-vSZP atomic orbital basis as used by g-xTB v1.1.
//!
//! q-vSZP: M. Müller, A. Hansen, S. Grimme, J. Chem. Phys. 158, 014103 (2023), in the variant
//! shipped with g-xTB 1.1.0 (`basisq`). Every atom gets its own contracted Gaussians, whose
//! contraction coefficients depend linearly on an effective charge:
//!
//! - CN_A = sum_B 1/2 [1 + erf(-3.75 (r_AB - R_AB) / R_AB)], R_AB = R_A + R_B from the
//!   program's own covalent radii (`ncoord_basq`; [`RCOV_BASIS_CN`], bohr), all pairs;
//! - q_eff,A = q_A - k_q2 q_A^2 + k_cn sqrt(CN_A) + k_qcn q_A CN_A with the EEQ-BC charges q_A
//!   and the element constants of the `basisq` header (k_cn, k_q2, k_qcn);
//! - c_i = c0_i + c1_i q_eff for each primitive exp(-alpha_i r^2) of a shell; the program
//!   stores c_i times the primitive normalization and normalizes the contracted function
//!   (its `molden.input` prints exactly these normalized coefficients).
//!
//! All of this is verified against the 1.1.0 program to ~1e-15 (reference tests).

use crate::data::RCOV_BASIS_CN;
use crate::params::QvszpParams;
use axtb::basis::{Cgto, MAXPRIM};
use axtb::{Error, Molecule, Result};
use axtb_base::ncoord::{CnModel, Erf};
use std::f64::consts::PI;

/// Steepness of the basis coordination number.
pub const KCN_BASIS: f64 = 3.75;

/// Coordination number of the q-vSZP effective charge (printed as "CN(basis)"):
/// CN_A = sum_B 1/2 [1 + erf(-3.75 (R_AB - Rc_AB) / Rc_AB)], Rc_AB = R_A + R_B, all pairs, no
/// cutoff (`axtb-base`'s [`CnModel`] with [`Erf`]).
pub fn basis_cn(mol: &Molecule) -> Vec<f64> {
    let rc = |i: usize, j: usize| RCOV_BASIS_CN[mol.numbers[i] - 1] + RCOV_BASIS_CN[mol.numbers[j] - 1];
    CnModel::new(Erf { k: KCN_BASIS }, rc).compute(&mol.positions)
}

/// The three contributions to the effective charge and their sum (the program prints them
/// as "q change", "CN change", "qCN change" and "total").
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EffectiveCharge {
    pub q_term: f64,
    pub cn_term: f64,
    pub qcn_term: f64,
    pub qeff: f64,
}

/// Effective charges q_eff of all atoms from charges `q` and basis CNs `cn`.
pub fn effective_charges(basis: &QvszpParams, mol: &Molecule, q: &[f64], cn: &[f64]) -> Result<Vec<EffectiveCharge>> {
    (0..mol.nat())
        .map(|i| {
            let el = basis.element(mol.numbers[i])?;
            let q_term = q[i] - el.k_q2 * q[i] * q[i];
            let cn_term = el.k_cn * cn[i].sqrt();
            let qcn_term = el.k_qcn * q[i] * cn[i];
            Ok(EffectiveCharge { q_term, cn_term, qcn_term, qeff: q_term + cn_term + qcn_term })
        })
        .collect()
}

const DFACTORIAL: [f64; 6] = [1.0, 1.0, 3.0, 15.0, 105.0, 945.0];

/// Contracted shells of an atom of element `z` with effective charge `qeff`, normalized,
/// in the [`axtb::basis::Cgto`] convention (coefficients include the normalization of the
/// x^l primitive component).
pub fn atom_shells(basis: &QvszpParams, z: usize, qeff: f64) -> Result<Vec<Cgto>> {
    let el = basis.element(z)?;
    el.shells
        .iter()
        .map(|sh| {
            let np = sh.alpha.len();
            if np > MAXPRIM {
                return Err(Error::Input(format!("q-vSZP: {np} primitives exceed MAXPRIM")));
            }
            let l = sh.l;
            let raw: Vec<f64> = (0..np).map(|i| sh.c0[i] + sh.c1[i] * qeff).collect();
            // self-overlap of the contraction of normalized primitives
            let mut s = 0.0;
            for i in 0..np {
                for j in 0..np {
                    let (a, b) = (sh.alpha[i], sh.alpha[j]);
                    s += raw[i] * raw[j] * (2.0 * (a * b).sqrt() / (a + b)).powf(l as f64 + 1.5);
                }
            }
            let scale = 1.0 / s.sqrt();
            let mut cgto = Cgto { l, nprim: np, alpha: [0.0; MAXPRIM], coeff: [0.0; MAXPRIM] };
            for (i, (&a, &c)) in sh.alpha.iter().zip(&raw).enumerate() {
                let norm = (2.0 * a / PI).powf(0.75) * (4.0 * a).sqrt().powi(l as i32) / DFACTORIAL[l].sqrt();
                cgto.alpha[i] = a;
                cgto.coeff[i] = c * norm * scale;
            }
            Ok(cgto)
        })
        .collect()
}

/// Derivative with respect to q of the normalized shell with exponents `alpha` and raw
/// coefficients c0 + c1 q (as [`atom_shells`] builds it): a shell with the same primitives and
/// coefficients dc_i/dq (integrals are linear in the coefficients of each shell).
pub fn shell_q_derivative(l: usize, alpha: &[f64], c0: &[f64], c1: &[f64], q: f64) -> Cgto {
    let np = alpha.len();
    let raw: Vec<f64> = (0..np).map(|i| c0[i] + c1[i] * q).collect();
    let (mut s, mut ds) = (0.0, 0.0);
    for i in 0..np {
        for j in 0..np {
            let (a, b) = (alpha[i], alpha[j]);
            let o = (2.0 * (a * b).sqrt() / (a + b)).powf(l as f64 + 1.5);
            s += raw[i] * raw[j] * o;
            ds += 2.0 * c1[i] * raw[j] * o;
        }
    }
    let mut cgto = Cgto { l, nprim: np, alpha: [0.0; MAXPRIM], coeff: [0.0; MAXPRIM] };
    for i in 0..np {
        let a = alpha[i];
        let norm = (2.0 * a / PI).powf(0.75) * (4.0 * a).sqrt().powi(l as i32) / DFACTORIAL[l].sqrt();
        cgto.alpha[i] = a;
        cgto.coeff[i] = norm * (c1[i] / s.sqrt() - raw[i] * ds / (2.0 * s * s.sqrt()));
    }
    cgto
}

/// d/dq_eff of the shells of [`atom_shells`].
pub fn atom_shells_dq(basis: &QvszpParams, z: usize, qeff: f64) -> Result<Vec<Cgto>> {
    Ok(basis.element(z)?.shells.iter().map(|sh| shell_q_derivative(sh.l, &sh.alpha, &sh.c0, &sh.c1, qeff)).collect())
}

/// The program's internal (unnormalized-contraction) coefficients of a shell:
/// (c0 + c1 q_eff) times the primitive normalization of the x^l component.
pub fn raw_shell_coefficients(basis: &QvszpParams, z: usize, qeff: f64) -> Result<Vec<Vec<f64>>> {
    let el = basis.element(z)?;
    Ok(el
        .shells
        .iter()
        .map(|sh| {
            (0..sh.alpha.len())
                .map(|i| {
                    let a = sh.alpha[i];
                    let norm = (2.0 * a / PI).powf(0.75) * (4.0 * a).sqrt().powi(sh.l as i32) / DFACTORIAL[sh.l].sqrt();
                    (sh.c0[i] + sh.c1[i] * qeff) * norm
                })
                .collect()
        })
        .collect())
}

/// The q-vSZP basis of a molecule: per-atom shells plus the effective charges.
#[derive(Clone, Debug)]
pub struct QvszpBasis {
    pub cn: Vec<f64>,
    pub qeff: Vec<EffectiveCharge>,
    /// Shells of each atom.
    pub shells: Vec<Vec<Cgto>>,
}

impl QvszpBasis {
    /// Build the basis of `mol` from EEQ-BC charges `q`.
    pub fn new(basis: &QvszpParams, mol: &Molecule, q: &[f64]) -> Result<Self> {
        let cn = basis_cn(mol);
        let qeff = effective_charges(basis, mol, q, &cn)?;
        let shells = (0..mol.nat()).map(|i| atom_shells(basis, mol.numbers[i], qeff[i].qeff)).collect::<Result<_>>()?;
        Ok(Self { cn, qeff, shells })
    }
}
