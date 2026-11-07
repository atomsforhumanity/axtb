//! One-electron integrals in a per-atom basis (`Vec<Vec<Cgto>>`), with the program's
//! primitive screening (a_i a_j R^2/(a_i + a_j) > INTCUT skipped).

use axtb::basis::Cgto;
use axtb::integrals::{overlap, MBLK};
use axtb::molecule::sub3;
use axtb::Molecule;

/// Primitive-pair screening of the program's SCF-basis overlap (25; the H0 and ACP overlaps
/// use 20, see [`crate::h0::INTCUT`], [`crate::acp::INTCUT`]).
pub const INTCUT: f64 = 25.0;

/// AO bookkeeping: (atom, shell, l) per AO, in axtb order (spherical m = -l..l).
pub fn ao_list(shells: &[Vec<Cgto>]) -> Vec<(usize, usize, usize)> {
    let mut ao = Vec::new();
    for (iat, sh) in shells.iter().enumerate() {
        for (ish, s) in sh.iter().enumerate() {
            for _ in 0..2 * s.l + 1 {
                ao.push((iat, ish, s.l));
            }
        }
    }
    ao
}

/// Full overlap matrix (row-major nao x nao).
pub fn overlap_matrix(mol: &Molecule, shells: &[Vec<Cgto>]) -> Vec<f64> {
    let mut first = Vec::new();
    let mut nao = 0;
    for sh in shells {
        let mut v = Vec::new();
        for s in sh {
            v.push(nao);
            nao += 2 * s.l + 1;
        }
        first.push(v);
    }
    let mut s = vec![0.0; nao * nao];
    let mut blk = [0.0; MBLK];
    for ia in 0..shells.len() {
        for ib in 0..=ia {
            let vec = sub3(mol.positions[ia], mol.positions[ib]);
            for (isa, sa) in shells[ia].iter().enumerate() {
                for (isb, sb) in shells[ib].iter().enumerate() {
                    overlap(sb, sa, vec, INTCUT, &mut blk);
                    let (na, nb) = (2 * sa.l + 1, 2 * sb.l + 1);
                    for a in 0..na {
                        for b in 0..nb {
                            let (i, j) = (first[ia][isa] + a, first[ib][isb] + b);
                            s[i * nao + j] = blk[b + nb * a];
                            s[j * nao + i] = blk[b + nb * a];
                        }
                    }
                }
            }
        }
    }
    s
}

/// Mulliken shell charges q_l = n^ref_l - sum_{mu in l} (P S)_{mu mu} (spin densities summed),
/// with the program's reference shell occupations
/// ([`crate::data_runtime::REF_SHELL_OCC`]); shells in the order of `shells`.
pub fn mulliken_shell_charges(numbers: &[usize], shells: &[Vec<Cgto>], s: &[f64], p: &[Vec<f64>]) -> Vec<f64> {
    let n = ao_list(shells).len();
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
