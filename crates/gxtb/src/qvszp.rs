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
//! All of this is verified against the 1.1.0 program to ~1e-15 (reference tests). The
//! contraction ([`contracted`], [`contracted_dq`], with tabulated or scaled [`Exponents`]) is
//! the one of every q-vSZP shell of both versions: the SCF bases and the H0 bases of v1.1
//! ([`crate::h0`]) and 2.0.1 ([`crate::v2::basis`]).

use crate::data::RCOV_BASIS_CN;
use crate::params::{QvszpParams, QvszpShell};
use axtb::basis::{Basis, Cgto, MAXPRIM};
use axtb::{Error, Molecule, Result};
use axtb_base::math::primitive_norm;
use axtb_base::ncoord::{CnModel, Erf};

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

/// The exponents of a q-vSZP shell: as tabulated (the SCF basis of both versions) or scaled
/// by a factor (the H0 bases: `pln` of v1.1, `ps_h0_qvszp_exp_scal` of 2.0.1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Exponents {
    Tabulated,
    Scaled(f64),
}

impl Exponents {
    #[inline]
    fn of(self, alpha: f64) -> f64 {
        match self {
            Exponents::Tabulated => alpha,
            Exponents::Scaled(f) => f * alpha,
        }
    }
}

/// The exponents `exps` of shell `sh` and its raw coefficients c0 + c1 q.
#[inline]
fn primitives(sh: &QvszpShell, exps: Exponents, q: f64) -> Result<([f64; MAXPRIM], [f64; MAXPRIM])> {
    let np = sh.alpha.len();
    if np > MAXPRIM {
        return Err(Error::Input(format!("q-vSZP: {np} primitives exceed MAXPRIM")));
    }
    let (mut alpha, mut raw) = ([0.0; MAXPRIM], [0.0; MAXPRIM]);
    for i in 0..np {
        alpha[i] = exps.of(sh.alpha[i]);
        raw[i] = sh.c0[i] + sh.c1[i] * q;
    }
    Ok((alpha, raw))
}

/// The overlaps (2 sqrt(a b)/(a + b))^(l + 3/2) of the normalized primitives with exponents
/// `alpha[..np]`, `o[i * MAXPRIM + j]`. Each pair is evaluated once: the expression is
/// symmetric in a and b to the last bit (products and sums commute exactly).
#[inline]
fn primitive_overlaps(l: usize, alpha: &[f64; MAXPRIM], np: usize) -> [f64; MAXPRIM * MAXPRIM] {
    let mut o = [0.0; MAXPRIM * MAXPRIM];
    for i in 0..np {
        for j in 0..=i {
            let (a, b) = (alpha[i], alpha[j]);
            let v = (2.0 * (a * b).sqrt() / (a + b)).powf(l as f64 + 1.5);
            o[i * MAXPRIM + j] = v;
            o[j * MAXPRIM + i] = v;
        }
    }
    o
}

/// The normalized contracted shell `sh` with exponents `exps` and coefficients c0 + c1 q, in the
/// [`axtb::basis::Cgto`] convention (coefficients include the normalization of the x^l
/// primitive component).
pub fn contracted(sh: &QvszpShell, exps: Exponents, q: f64) -> Result<Cgto> {
    let (alpha, raw) = primitives(sh, exps, q)?;
    let (l, np) = (sh.l, sh.alpha.len());
    let o = primitive_overlaps(l, &alpha, np);
    // self-overlap of the contraction of normalized primitives
    let mut s = 0.0;
    for i in 0..np {
        for j in 0..np {
            s += raw[i] * raw[j] * o[i * MAXPRIM + j];
        }
    }
    let scale = 1.0 / s.sqrt();
    let mut cgto = Cgto { l, nprim: np, alpha, coeff: [0.0; MAXPRIM] };
    for i in 0..np {
        let norm = primitive_norm(l, alpha[i]);
        cgto.coeff[i] = raw[i] * norm * scale;
    }
    Ok(cgto)
}

/// Derivative with respect to q of the shell of [`contracted`]: a shell with the same
/// primitives and coefficients dc_i/dq (integrals are linear in the coefficients of each
/// shell).
pub fn contracted_dq(sh: &QvszpShell, exps: Exponents, q: f64) -> Result<Cgto> {
    let (alpha, raw) = primitives(sh, exps, q)?;
    let (l, np) = (sh.l, sh.alpha.len());
    let o = primitive_overlaps(l, &alpha, np);
    let (mut s, mut ds) = (0.0, 0.0);
    for i in 0..np {
        for j in 0..np {
            let o = o[i * MAXPRIM + j];
            s += raw[i] * raw[j] * o;
            ds += 2.0 * sh.c1[i] * raw[j] * o;
        }
    }
    let mut cgto = Cgto { l, nprim: np, alpha, coeff: [0.0; MAXPRIM] };
    for i in 0..np {
        let norm = primitive_norm(l, alpha[i]);
        cgto.coeff[i] = norm * (sh.c1[i] / s.sqrt() - raw[i] * ds / (2.0 * s * s.sqrt()));
    }
    Ok(cgto)
}

/// The shells `shells` of one atom at effective charge `q`, shell `ish` with exponents
/// `exps(ish)` ([`contracted`]), or with `dq` their q derivatives ([`contracted_dq`]).
pub fn contracted_shells(
    shells: &[QvszpShell],
    exps: impl Fn(usize) -> Exponents,
    q: f64,
    dq: bool,
) -> Result<Vec<Cgto>> {
    let shell = |(ish, sh): (usize, &QvszpShell)| {
        if dq {
            contracted_dq(sh, exps(ish), q)
        } else {
            contracted(sh, exps(ish), q)
        }
    };
    shells.iter().enumerate().map(shell).collect()
}

/// Contracted shells of an atom of element `z` with effective charge `qeff`, normalized
/// ([`contracted`] with the tabulated exponents).
pub fn atom_shells(basis: &QvszpParams, z: usize, qeff: f64) -> Result<Vec<Cgto>> {
    contracted_shells(&basis.element(z)?.shells, |_| Exponents::Tabulated, qeff, false)
}

/// d/dq_eff of the shells of [`atom_shells`].
pub fn atom_shells_dq(basis: &QvszpParams, z: usize, qeff: f64) -> Result<Vec<Cgto>> {
    contracted_shells(&basis.element(z)?.shells, |_| Exponents::Tabulated, qeff, true)
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
                    let norm = primitive_norm(sh.l, a);
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
    /// The shells of each atom (`scf.cgto[atom]`) and their AO bookkeeping, with the primitive
    /// screening of the SCF overlap ([`crate::ints::INTCUT`]).
    pub scf: Basis,
}

impl QvszpBasis {
    /// Build the basis of `mol` from EEQ-BC charges `q`.
    pub fn new(basis: &QvszpParams, mol: &Molecule, q: &[f64]) -> Result<Self> {
        let cn = basis_cn(mol);
        let qeff = effective_charges(basis, mol, q, &cn)?;
        let shells = (0..mol.nat()).map(|i| atom_shells(basis, mol.numbers[i], qeff[i].qeff)).collect::<Result<_>>()?;
        Ok(Self { cn, qeff, scf: Basis::from_atom_shells(mol, shells, crate::ints::INTCUT) })
    }
}
