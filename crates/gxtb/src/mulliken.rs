//! Mulliken analysis of g-xTB densities, both versions: shell populations
//! `sum_{mu in l} (P S)_{mu mu}`, the shell charges `n0_l` minus them, shell magnetizations
//! (alpha minus beta populations) and the atomic dipoles and traceless quadrupoles
//! `-sum_{mu in A} (P D)_{mu mu}`, `-sum_{mu in A} (P Q)_{mu mu}` with the multipole integrals
//! around the atom of the first (row) AO index.
//!
//! Row-major n x n matrices in the AO order of the SCF basis ([`Basis`]). The overlap is
//! exactly symmetric (both orientations are written from one block), so its rows are its
//! columns.
//!
//! Rounding orders, each written once:
//! - shell populations (both versions): per AO row the sequential sum of P S ([`dot_seq`]),
//!   added to the shells in AO order;
//! - atomic multipoles (both versions, [`moments_by_rows`]): per AO row the negated sequential
//!   sums of P D and P Q, added to the atoms in AO order (rows in parallel when asked).
//!
//! The 2.0.1 response analyses density changes with eight-accumulator row dots instead
//! (`v2::scf::Setup::delta_state`); GFN2-xTB's analysis (`axtb::scf`) sums columns with four
//! accumulators.

use crate::dense::dot_seq;
use axtb::basis::Basis;
use axtb_base::threads;

/// Mulliken shell populations of a (spin) density `p` with the overlap `s`, rows in parallel
/// with `parallel` (the result does not depend on it).
pub fn shell_populations(bas: &Basis, s: &[f64], p: &[f64], parallel: bool) -> Vec<f64> {
    let n = bas.nao;
    let row = |mu: usize| dot_seq(&p[mu * n..(mu + 1) * n], &s[mu * n..(mu + 1) * n]);
    let mut pop = vec![0.0; bas.nsh];
    if parallel {
        for (mu, v) in threads::map(n, true, row).into_iter().enumerate() {
            pop[bas.ao2sh[mu]] += v;
        }
    } else {
        for mu in 0..n {
            pop[bas.ao2sh[mu]] += row(mu);
        }
    }
    pop
}

/// Mulliken shell charges `n0 - pop` of the total density `p` for the reference occupations
/// `n0`.
pub fn shell_charges(bas: &Basis, s: &[f64], n0: &[f64], p: &[f64], parallel: bool) -> Vec<f64> {
    let pop = shell_populations(bas, s, p, parallel);
    n0.iter().zip(&pop).map(|(r, x)| r - x).collect()
}

/// Shell magnetizations (alpha minus beta populations) of spin densities `p`; `None` for a
/// closed shell (one matrix).
pub fn magnetizations(bas: &Basis, s: &[f64], p: &[Vec<f64>], parallel: bool) -> Option<Vec<f64>> {
    (p.len() == 2).then(|| {
        let (pa, pb) = (shell_populations(bas, s, &p[0], parallel), shell_populations(bas, s, &p[1], parallel));
        pa.iter().zip(&pb).map(|(a, b)| a - b).collect()
    })
}

/// Shell populations, atomic dipoles and quadrupoles of the total density `p` in 2.0.1's order
/// (row sums; rows in parallel with `parallel`).
#[allow(clippy::type_complexity)]
pub fn moments_by_rows(
    bas: &Basis,
    s: &[f64],
    dpint: &[Vec<f64>; 3],
    qpint: &[Vec<f64>; 6],
    p: &[f64],
    parallel: bool,
) -> (Vec<f64>, Vec<[f64; 3]>, Vec<[f64; 6]>) {
    let n = bas.nao;
    let nat = bas.nsh_at.len();
    let rows = threads::map(n, parallel, |mu| {
        let r = mu * n..(mu + 1) * n;
        let pr = &p[r.clone()];
        let mut out = [0.0; 10];
        out[0] = dot_seq(pr, &s[r.clone()]);
        for x in 0..3 {
            out[1 + x] = -dot_seq(pr, &dpint[x][r.clone()]);
        }
        for c in 0..6 {
            out[4 + c] = -dot_seq(pr, &qpint[c][r.clone()]);
        }
        out
    });
    let mut pop = vec![0.0; bas.nsh];
    let mut dip = vec![[0.0; 3]; nat];
    let mut qp = vec![[0.0; 6]; nat];
    for (mu, v) in rows.iter().enumerate() {
        let a = bas.ao2at[mu];
        pop[bas.ao2sh[mu]] += v[0];
        for x in 0..3 {
            dip[a][x] += v[1 + x];
        }
        for c in 0..6 {
            qp[a][c] += v[4 + c];
        }
    }
    (pop, dip, qp)
}
