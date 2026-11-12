//! Atomic correction potentials (P-Eqs. 22-24) as implemented by the g-xTB 1.1.0 program
//! (`setupcbas0_`, `setupcbas_`, `calcvacp_`), verified element by element against the
//! H0 + ACP matrix it builds (reference tests).
//!
//! Every atom carries one normalized single-Gaussian projector shell per l = s, p, d, f
//! with exponent `acp_exp(l)` (all 2l+1 spherical components). With the overlaps
//! S_{mu,r} between the SCF q-vSZP AOs and the projectors,
//! H^ACP_{mu nu} = sum_r w_r S_{mu r} S_{nu r},
//! w_{A,l} = acp_coeff(l) (1 + acpcn_A sqrt(CN_A)) with the g-xTB CN (signed, linear in the
//! coefficient; the preprint's CN/CN^avg form is not what the program does).
//! Atom pairs with R^2 > 1600 bohr^2 are skipped.

use crate::params::GxtbParams;
use axtb::basis::{Cgto, MAXPRIM};
use axtb::integrals::{overlap, MBLK};
use axtb::molecule::{norm2_sq, sub3};
use axtb::{Molecule, Result};
use axtb_base::math::primitive_norm;

/// Squared-distance cutoff of the projector overlaps (bohr^2).
pub const R2_CUTOFF: f64 = 1600.0;
/// Primitive screening of the projector overlaps.
pub const INTCUT: f64 = 20.0;

/// Normalized projector shell.
pub fn projector(l: usize, alpha: f64) -> Cgto {
    let mut c = Cgto { l, nprim: 1, alpha: [0.0; MAXPRIM], coeff: [0.0; MAXPRIM] };
    c.alpha[0] = alpha;
    c.coeff[0] = primitive_norm(l, alpha);
    c
}

/// Add H^ACP to `h` (full nao x nao, axtb AO order of `shells`). `cn` is the g-xTB CN
/// (not square-rooted).
pub fn add_acp(param: &GxtbParams, mol: &Molecule, shells: &[Vec<Cgto>], cn: &[f64], h: &mut [f64]) -> Result<()> {
    let nat = mol.nat();
    let mut first = Vec::with_capacity(nat);
    let mut nao = 0;
    for sh in shells {
        first.push(nao);
        nao += sh.iter().map(|s| 2 * s.l + 1).sum::<usize>();
    }
    let mut blk = [0.0; MBLK];
    let mut col = vec![0.0; nao];
    for ip in 0..nat {
        let e = param.element(mol.numbers[ip])?;
        let f = 1.0 + e.acpcn * cn[ip].sqrt();
        for l in 0..4 {
            let w = e.acp_coeff[l] * f;
            if w == 0.0 {
                continue;
            }
            let proj = projector(l, e.acp_exp[l]);
            for m in 0..2 * l + 1 {
                col.iter_mut().for_each(|x| *x = 0.0);
                for (ia, sh) in shells.iter().enumerate() {
                    let vec = sub3(mol.positions[ia], mol.positions[ip]);
                    if norm2_sq(vec) > R2_CUTOFF {
                        continue;
                    }
                    let mut k = first[ia];
                    for s in sh {
                        overlap(&proj, s, vec, INTCUT, &mut blk);
                        let ns = 2 * s.l + 1;
                        let np = 2 * l + 1;
                        for a in 0..ns {
                            col[k + a] = blk[m + np * a];
                        }
                        k += ns;
                    }
                }
                for i in 0..nao {
                    if col[i] == 0.0 {
                        continue;
                    }
                    let wi = w * col[i];
                    for j in 0..nao {
                        h[i * nao + j] += wi * col[j];
                    }
                }
            }
        }
    }
    Ok(())
}
