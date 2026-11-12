//! q-vSZP basis of g-xTB 2.0.1: the v1.1 tables (bit-identical in the 2.0.1 program) with the
//! effective charge q_eff = k0 (q - k1 q^2) + k2 f(CN) + k3 q CN (f = smoothed square root,
//! k0 = 1, k1 = k_q2, k2 = k_cn, k3 = k_qcn of the `basisq` header; EEQ-BC charges q and basis
//! CN), and the H0 basis with exponents scaled by `ps_h0_qvszp_exp_scal` and k0, k2, k3 scaled
//! by `pa_h0_qvszp_k{0,2,3}_scal` (k1 unscaled). Overlap, dipole and quadrupole integrals of the
//! SCF basis with tblite's screening: primitive pairs with a_i a_j R^2/(a_i + a_j) > 25 are
//! skipped and, in program mode, atom pairs beyond tblite's integral cutoff
//! min(sqrt(2 * 25 / alpha_min), 40) bohr.

use super::cn::basis_cn;
use super::{element_shells, qvszp, sqrt_cn, Model};
use crate::data_v2::{H0_EXP_SCALE, H0_K0_SCALE, H0_K2_SCALE, H0_K3_SCALE};
use crate::qvszp::{contracted_shells, Exponents};
use axtb::basis::{ao_offsets, Cgto};
use axtb::integrals::{multipole, Scratch, MBLK};
use axtb::molecule::{norm2_sq, sub3};
use axtb::{Molecule, Result};

/// tblite's primitive screening threshold at accuracy 1 (and below).
pub const INTCUT: f64 = 25.0;
/// Largest integral pair cutoff (bohr).
pub const MAX_CUTOFF: f64 = 40.0;

/// The q-vSZP basis of a molecule and its effective charges.
#[derive(Clone, Debug)]
pub struct Basis {
    /// Basis coordination numbers.
    pub cn: Vec<f64>,
    /// The three q_eff terms per atom: (q - k1 q^2, f(CN), q CN).
    pub terms: Vec<[f64; 3]>,
    /// Effective charges of the SCF and of the H0 basis.
    pub qeff: Vec<f64>,
    pub qeff_h0: Vec<f64>,
    /// Normalized shells of every atom (SCF and H0 basis).
    pub shells: Vec<Vec<Cgto>>,
    pub h0_shells: Vec<Vec<Cgto>>,
    /// Smallest primitive exponent of the SCF basis.
    pub alpha_min: f64,
}

/// SCF-basis weights (k0, k1, k2, k3) of the effective charge of element `z`.
pub fn qeff_weights(z: usize) -> [f64; 4] {
    let el = qvszp().element(z).expect("q-vSZP covers H-Lr");
    [1.0, el.k_q2, el.k_cn, el.k_qcn]
}

/// H0-basis weights (k0, k1, k2, k3) of element `z`.
pub fn qeff_h0_weights(z: usize) -> [f64; 4] {
    let k = qeff_weights(z);
    [H0_K0_SCALE[z - 1] * k[0], k[1], H0_K2_SCALE[z - 1] * k[2], H0_K3_SCALE[z - 1] * k[3]]
}

/// Effective charge and its terms for weights `k`, charge `q` and basis CN `cn`.
pub fn effective_charge(k: [f64; 4], q: f64, cn: f64) -> ([f64; 3], f64) {
    let t = [q - k[1] * q * q, sqrt_cn(cn).0, q * cn];
    (t, k[0] * t[0] + k[2] * t[1] + k[3] * t[2])
}

/// d q_eff / d q and d q_eff / d CN.
pub fn effective_charge_deriv(k: [f64; 4], q: f64, cn: f64) -> (f64, f64) {
    (k[0] * (1.0 - 2.0 * k[1] * q) + k[3] * cn, k[2] * sqrt_cn(cn).1 + k[3] * q)
}

/// SCF-basis shells of element `z` at effective charge `q` (`dq`: their q derivatives).
pub fn scf_shells(z: usize, q: f64, dq: bool) -> Result<Vec<Cgto>> {
    contracted_shells(element_shells(z), |_| Exponents::Tabulated, q, dq)
}

/// H0-basis shells of element `z` at effective charge `q` (`dq`: their q derivatives):
/// exponents scaled by `ps_h0_qvszp_exp_scal`.
pub fn h0_shells(z: usize, q: f64, dq: bool) -> Result<Vec<Cgto>> {
    contracted_shells(element_shells(z), |ish| Exponents::Scaled(H0_EXP_SCALE[z - 1][ish]), q, dq)
}

impl Basis {
    /// The basis of `mol` from the EEQ-BC charges `q`.
    pub fn new(model: &Model, mol: &Molecule, q: &[f64]) -> Result<Self> {
        let n = mol.nat();
        let cn = basis_cn(model, mol);
        let mut terms = Vec::with_capacity(n);
        let mut qeff = Vec::with_capacity(n);
        let mut qeff_h0 = Vec::with_capacity(n);
        let mut shells = Vec::with_capacity(n);
        let mut hshells = Vec::with_capacity(n);
        let mut alpha_min = f64::INFINITY;
        for i in 0..n {
            let z = mol.numbers[i];
            let (t, qe) = effective_charge(qeff_weights(z), q[i], cn[i]);
            let (_, qh) = effective_charge(qeff_h0_weights(z), q[i], cn[i]);
            let sh = scf_shells(z, qe, false)?;
            for s in &sh {
                for p in 0..s.nprim {
                    alpha_min = alpha_min.min(s.alpha[p]);
                }
            }
            shells.push(sh);
            hshells.push(h0_shells(z, qh, false)?);
            terms.push(t);
            qeff.push(qe);
            qeff_h0.push(qh);
        }
        Ok(Self { cn, terms, qeff, qeff_h0, shells, h0_shells: hshells, alpha_min })
    }

    /// Integral pair cutoff (bohr): tblite's `get_cutoff` in program mode, none otherwise.
    pub fn pair_cutoff(&self, model: &Model) -> f64 {
        if model.program() {
            (2.0 * INTCUT / self.alpha_min).sqrt().min(MAX_CUTOFF)
        } else {
            f64::INFINITY
        }
    }
}

/// Smallest primitive exponent of a shell.
pub fn min_exponent(c: &Cgto) -> f64 {
    c.alpha[..c.nprim].iter().copied().fold(f64::INFINITY, f64::min)
}

/// Smallest primitive exponent of each atom's shells.
pub fn atom_min_exponents(shells: &[Vec<Cgto>]) -> Vec<f64> {
    shells.iter().map(|sh| sh.iter().map(min_exponent).fold(f64::INFINITY, f64::min)).collect()
}

/// Whether every primitive pair of two shells (or atoms) with smallest exponents `a` and `b` at
/// squared distance `r2` is beyond the primitive screening (a_i a_j R^2/(a_i + a_j) > INTCUT;
/// smallest for the smallest exponents): their integrals and derivatives are exactly zero, and
/// skipping them changes no result. The margin covers the rounding of the kernels' test.
#[inline]
pub fn screened(a: f64, b: f64, r2: f64) -> bool {
    a * b * r2 / (a + b) > INTCUT * (1.0 + 1e-10)
}

/// Overlap S (full, row-major) and the dipole and traceless quadrupole integrals of all AO
/// pairs around the atom of the first index (`dp[x][mu*n + nu]` = <mu|(r - R_A(mu))_x|nu>),
/// for atom pairs within `cutoff` (bohr). The rows of each atom are one task (in parallel when
/// `parallel`); blocks that are exactly zero are skipped.
#[allow(clippy::type_complexity)]
pub fn integrals(
    mol: &Molecule,
    shells: &[Vec<Cgto>],
    cutoff: f64,
    parallel: bool,
) -> (Vec<f64>, [Vec<f64>; 3], [Vec<f64>; 6]) {
    let first = ao_offsets(shells);
    let nao: usize = shells.iter().flatten().map(|s| 2 * s.l + 1).sum();
    let nat = mol.nat();
    let amin = atom_min_exponents(shells);
    let mut bounds: Vec<usize> = (0..nat).map(|a| first[a].first().copied().unwrap_or(0)).collect();
    bounds.push(nao);
    let z = || vec![0.0; nao * nao];
    let mut sm = z();
    let (mut d0, mut d1, mut d2) = (z(), z(), z());
    let (mut q0, mut q1, mut q2, mut q3, mut q4, mut q5) = (z(), z(), z(), z(), z(), z());
    let cut2 = cutoff * cutoff;
    {
        let mut mats: Vec<&mut [f64]> =
            vec![&mut sm, &mut d0, &mut d1, &mut d2, &mut q0, &mut q1, &mut q2, &mut q3, &mut q4, &mut q5];
        crate::parallel::for_row_blocks(&mut mats, nao, &bounds, parallel, |ia, rows| {
            let mut s = [0.0; MBLK];
            let mut d = [[0.0; 3]; MBLK];
            let mut q = [[0.0; 6]; MBLK];
            let mut scratch = Scratch::default();
            let row0 = bounds[ia];
            for ib in 0..nat {
                let vec = sub3(mol.positions[ia], mol.positions[ib]);
                let r2 = norm2_sq(vec);
                if r2 > cut2 || (ia != ib && screened(amin[ia], amin[ib], r2)) {
                    continue;
                }
                for (isa, sa) in shells[ia].iter().enumerate() {
                    for (isb, sb) in shells[ib].iter().enumerate() {
                        if ia != ib && screened(min_exponent(sa), min_exponent(sb), r2) {
                            continue;
                        }
                        multipole(sb, sa, vec, INTCUT, &mut s, &mut d, &mut q, &mut scratch);
                        let (na, nb) = (2 * sa.l + 1, 2 * sb.l + 1);
                        for a in 0..na {
                            let mu = first[ia][isa] + a - row0;
                            for b in 0..nb {
                                let nu = first[ib][isb] + b;
                                let k = b + nb * a;
                                let o = mu * nao + nu;
                                rows[0][o] = s[k];
                                for x in 0..3 {
                                    rows[1 + x][o] = d[k][x];
                                }
                                for c in 0..6 {
                                    rows[4 + c][o] = q[k][c];
                                }
                            }
                        }
                    }
                }
            }
        });
    }
    // the overlap is symmetric; average the two evaluations
    let s0 = sm.clone();
    crate::parallel::fill_tiled(&mut sm, nao, parallel, |i, j| {
        if i == j {
            s0[i * nao + i]
        } else {
            0.5 * (s0[i * nao + j] + s0[j * nao + i])
        }
    });
    (sm, [d0, d1, d2], [q0, q1, q2, q3, q4, q5])
}
