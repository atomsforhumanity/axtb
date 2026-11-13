//! What reproduces the g-xTB 1.1.0 program beyond the v1.1 model (`Version::V1_1` of the
//! engine, [`crate::v2`]): the parts of its SCF that are program behaviour rather than model.
//!
//! - **Start** ([`guess_density`]): the program's start density (`guess_psh_`, `guess_p_`),
//!   diagonal, with the reference shell occupations of every atom scaled to its EEQ-BC electron
//!   count, n_l (Z_A - q_A)/Z_A, spread evenly over the 2l+1 AOs of the shell (open shell: half
//!   per spin). v1.1 starts from it in both conventions, as its mixer
//!   (`crate::strategy::Gxtb110FockMixer` and `Gxtb110Rule`, the 1.1.0 program's `diisfock_`)
//!   does.
//! - **The program's Fock matrix** ([`ProgramFock`], conventions `Gxtb`): the 1.1.0 program
//!   leaves out the AES potentials of the dipole-quadrupole and quadrupole-quadrupole energies
//!   (its `setvsdq_` differentiates E_AES' of the charge-dipole, charge-quadrupole and
//!   dipole-dipole parts only), and applies the population part of the onsite exchange at twice
//!   its derivative off the diagonal (`setfock_indo_`). The engine's terms are exact
//!   derivatives; this module turns their Fock matrix into the program's.
//! - **The eigenvalue cap** ([`cap`]): the program computes the lowest min(N_alpha + 10, n)
//!   eigenpairs only (DSYGVX) and sets the other eigenvalues to 100 Eh, so those levels stay
//!   empty at any electronic temperature.
//!
//! - **The gradient** ([`zvector`]): the program's Fock matrix is not the derivative of its
//!   energy, so its analytic gradient needs the response of the density (the Z-vector method).
//!
//! The program's Fermi smearing (`fermismear_`) is one of the shared rules of
//! [`crate::occupations`].

pub(crate) mod zvector;

use crate::v2::coulomb::{potentials_with, AesCache};
use crate::v2::exchange::Products;
use crate::v2::scf::Setup;
use axtb::Molecule;

/// The program's start density of a v1.1 setup (spin densities; closed shell: the total).
pub fn guess_density(setup: &Setup) -> Vec<Vec<f64>> {
    let n = setup.nao;
    let ns = setup.nspin;
    let fac = if ns == 2 { 0.5 } else { 1.0 };
    let mol = &setup.mol;
    let mut zval = vec![0.0; mol.nat()];
    for &(iat, ish, _) in &setup.shells {
        zval[iat] += crate::data::ZEFF_SHELL[ish * 103 + mol.numbers[iat] - 1] as f64;
    }
    let mut p = vec![0.0; n * n];
    for mu in 0..n {
        let ish = setup.basis.scf.ao2sh[mu];
        let (iat, _, l) = setup.shells[ish];
        let z = zval[iat];
        let psh = (z - setup.eeq.charges[iat]) * setup.ref_occ[ish] / z;
        p[mu * n + mu] = fac * psh / (2 * l + 1) as f64;
    }
    vec![p; ns]
}

/// The program's eigenvalue cap: the levels beyond min(N_alpha + 10, n) at 100 Eh.
pub fn cap(eps: &[f64], nalpha: usize) -> Vec<f64> {
    let nev = (nalpha + 10).min(eps.len());
    eps.iter().enumerate().map(|(k, &x)| if k < nev { x } else { 100.0 }).collect()
}

/// The 1.1.0 program's Fock matrix from the engine's exact one: the AES kernels of E_AES' (g3,
/// g5; g7 = g9 = 0) for the AES potentials, and the extra off-diagonal half of the onsite
/// exchange's population part.
#[derive(Clone, Debug)]
pub struct ProgramFock {
    /// AES kernels g3, g5, 0, 0 (nat x nat).
    pub subset: [Vec<f64>; 4],
}

impl ProgramFock {
    /// The program's AES kernels of `aes`.
    pub fn new(aes: &AesCache) -> Self {
        let z = vec![0.0; aes.aes[0].len()];
        Self { subset: [aes.aes[0].clone(), aes.aes[1].clone(), z.clone(), z] }
    }

    /// The AES potentials of the program's Fock matrix (those of E_AES'; v1.1 scales no
    /// dipoles).
    #[allow(clippy::type_complexity)]
    pub fn aes_potentials(
        &self,
        mol: &Molecule,
        qa: &[f64],
        dip: &[[f64; 3]],
        qp: &[[f64; 6]],
        parallel: bool,
    ) -> (Vec<f64>, Vec<[f64; 3]>, Vec<[f64; 6]>) {
        potentials_with(mol, qa, dip, qp, &self.subset, parallel)
    }

    /// Add the program's extra half of the onsite exchange's population part to the Fock
    /// matrices `fm` of the spin densities with products `pr` (closed shell: one, the total
    /// density): w (-(y_m + y_k) S_mk / 16) for m != k with y = m2 diag(P S) per channel
    /// ([`crate::v2::exchange::Dual`]), w = 1 (closed) or 2.
    pub fn add_ofx_extra(&self, setup: &Setup, pr: &[Products], fm: &mut [Vec<f64>]) {
        let Some(d) = &setup.exch.dual else { return };
        let n = setup.nao;
        let w = if pr.len() == 1 { 1.0 } else { 2.0 };
        let s = &setup.s;
        for (prm, f) in pr.iter().zip(fm.iter_mut()) {
            let ps = &prm.ps;
            let mut y = vec![0.0; n];
            for &(i0, i1) in &setup.exch.atom_ao {
                for m in i0..i1 {
                    for k in i0..i1 {
                        let l2 = d.m2[m * n + k];
                        if l2 != 0.0 {
                            y[m] += l2 * ps[k * n + k];
                        }
                    }
                }
            }
            for m in 0..n {
                for k in 0..n {
                    if m != k {
                        f[m * n + k] -= w * (y[m] + y[k]) * s[m * n + k] / 16.0;
                    }
                }
            }
        }
    }
}
