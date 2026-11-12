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
use axtb::ao::ShellPairs;
use axtb::basis::{Basis, Cgto, MAXPRIM};
use axtb::integrals::{overlap, MBLK};
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

/// Add H^ACP to `h` (full nao x nao, axtb AO order of the SCF basis `bas`). `cn` is the g-xTB
/// CN (not square-rooted).
pub fn add_acp(param: &GxtbParams, mol: &Molecule, bas: &Basis, cn: &[f64], h: &mut [f64]) -> Result<()> {
    let nat = mol.nat();
    let nao = bas.nao;
    let mut blk = [0.0; MBLK];
    let mut col = vec![0.0; nao];
    // R^2 > R2_CUTOFF skipped (sqrt(1600) = 40 exactly), no exponent screening
    let pairs = ShellPairs::unscreened(mol, bas, R2_CUTOFF.sqrt());
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
                for ia in 0..nat {
                    let Some(p) = pairs.pair(ia, ip) else { continue };
                    pairs.for_each_against(&p, &proj, |sp| {
                        overlap(&proj, sp.ci, p.vec, INTCUT, &mut blk);
                        let np = 2 * l + 1;
                        for a in 0..2 * sp.ci.l + 1 {
                            col[sp.ii + a] = blk[m + np * a];
                        }
                    });
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
