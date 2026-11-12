//! One-electron integrals in a per-atom basis ([`Basis::from_atom_shells`]), with the
//! program's primitive screening (a_i a_j R^2/(a_i + a_j) > INTCUT skipped) and no atom pair
//! cutoff or shell-pair screening (the 1.1.0 program evaluates every pair).

use axtb::ao::ShellPairs;
use axtb::basis::{ao_count, Basis, Cgto};
use axtb::integrals::{overlap, MBLK};
use axtb::Molecule;

/// Primitive-pair screening of the program's SCF-basis overlap (25; the H0 and ACP overlaps
/// use 20, see [`crate::h0::INTCUT`], [`crate::acp::INTCUT`]).
pub const INTCUT: f64 = 25.0;

/// Full overlap matrix (row-major nao x nao): the blocks of atom pairs B <= A, each written
/// in both orientations from the one evaluation.
pub fn overlap_matrix(mol: &Molecule, bas: &Basis) -> Vec<f64> {
    let nao = bas.nao;
    let mut s = vec![0.0; nao * nao];
    let mut blk = [0.0; MBLK];
    let pairs = ShellPairs::unscreened(mol, bas, f64::INFINITY);
    for ia in 0..mol.nat() {
        for ib in 0..=ia {
            let Some(p) = pairs.pair(ia, ib) else { continue };
            pairs.for_each(&p, |sp| {
                overlap(sp.cj, sp.ci, p.vec, INTCUT, &mut blk);
                let (na, nb) = (2 * sp.ci.l + 1, 2 * sp.cj.l + 1);
                for a in 0..na {
                    for b in 0..nb {
                        let (i, j) = (sp.ii + a, sp.jj + b);
                        s[i * nao + j] = blk[b + nb * a];
                        s[j * nao + i] = blk[b + nb * a];
                    }
                }
            });
        }
    }
    s
}

/// Mulliken shell charges q_l = n^ref_l - sum_{mu in l} (P S)_{mu mu} (spin densities summed),
/// with the program's reference shell occupations
/// ([`crate::data_runtime::REF_SHELL_OCC`]); shells in the order of `shells`.
pub fn mulliken_shell_charges(numbers: &[usize], shells: &[Vec<Cgto>], s: &[f64], p: &[Vec<f64>]) -> Vec<f64> {
    let n = ao_count(shells);
    let mut q = Vec::new();
    let mut k = 0;
    for (iat, sh) in shells.iter().enumerate() {
        for (ish, c) in sh.iter().enumerate() {
            let mut pop = 0.0;
            for mu in k..k + 2 * c.l + 1 {
                for ps in p {
                    for nu in 0..n {
                        pop += ps[mu * n + nu] * s[nu * n + mu];
                    }
                }
            }
            q.push(crate::data_runtime::REF_SHELL_OCC[numbers[iat] - 1][ish] - pop);
            k += 2 * c.l + 1;
        }
    }
    q
}
