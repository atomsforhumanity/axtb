//! The q-vSZP atomic orbital basis as used by g-xTB v1.1.
//!
//! q-vSZP: M. Müller, A. Hansen, S. Grimme, J. Chem. Phys. 158, 014103 (2023), in the variant
//! shipped with g-xTB 1.1.0 (`basisq`). Every atom gets its own contracted Gaussians, whose
//! contraction coefficients depend linearly on an effective charge:
//!
//! - CN_A = sum_B 1/2 [1 + erf(-3.75 (r_AB - R_AB) / R_AB)], R_AB = R_A + R_B from the
//!   program's own covalent radii (`ncoord_basq`; `RCOV_BASIS_CN`, bohr), all pairs;
//! - q_eff,A = q_A - k_q2 q_A^2 + k_cn sqrt(CN_A) + k_qcn q_A CN_A with the EEQ-BC charges q_A
//!   and the element constants of the `basisq` header (k_cn, k_q2, k_qcn);
//! - c_i = c0_i + c1_i q_eff for each primitive exp(-alpha_i r^2) of a shell; the program
//!   stores c_i times the primitive normalization and normalizes the contracted function
//!   (its `molden.input` prints exactly these normalized coefficients).
//!
//! All of this is verified against the 1.1.0 program to ~1e-15 (reference tests). The
//! contraction ([`contracted`], [`contracted_dq`], with tabulated or scaled [`Exponents`]) is
//! the one of every q-vSZP shell of both versions: the SCF and H0 bases of both versions
//! ([`crate::basis`]).

use crate::params::QvszpShell;
use axtb::basis::{Cgto, MAXPRIM};
use axtb::{Error, Result};
use axtb_base::math::primitive_norm;

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
