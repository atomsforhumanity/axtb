//! Mulliken analysis of a density: shell charges `q_l = n0_l - Σ_{μ∈l} (P S)_μμ` and atomic
//! dipoles and traceless quadrupoles `μ_A = -Σ_{μ∈A} (P D)_μμ`, `Θ_A = -Σ_{μ∈A} (P Q)_μμ` with the
//! multipole integrals around the atom of the column AO (tblite's
//! `get_mulliken_shell_charges`, `get_mulliken_atomic_multipoles`).
//!
//! Each column of the products is a four-accumulator dot product ([`super::dot`]), computed in
//! parallel when asked; the columns are then summed into the shells and atoms in AO order, so
//! the result does not depend on the thread count.

use super::dot;
use crate::basis::Basis;
use crate::hamiltonian::Integrals;
use axtb_base::threads;
use faer::Mat;

/// Mulliken shell charges (`n0sh` minus the shell populations of `p`) and, with `multipoles`,
/// atomic dipoles and quadrupoles (zeros without).
pub(crate) fn moments(
    bas: &Basis,
    ints: &Integrals,
    p: &Mat<f64>,
    n0sh: &[f64],
    qsh: &mut [f64],
    dpat: &mut [[f64; 3]],
    qpat: &mut [[f64; 6]],
    multipoles: bool,
    parallel: bool,
) {
    let nao = bas.nao;
    let column = |nu: usize| -> [f64; 10] {
        let pc = p.col_as_slice(nu);
        let mut r = [0.0; 10];
        r[0] = dot(pc, ints.overlap.col_as_slice(nu));
        if multipoles {
            for k in 0..3 {
                r[1 + k] = dot(pc, ints.dip[k].col_as_slice(nu));
            }
            for k in 0..6 {
                r[4 + k] = dot(pc, ints.quad[k].col_as_slice(nu));
            }
        }
        r
    };
    let cols = threads::map_with_min_len(nao, parallel, 16, column);
    qsh.copy_from_slice(n0sh);
    dpat.iter_mut().for_each(|x| *x = [0.0; 3]);
    qpat.iter_mut().for_each(|x| *x = [0.0; 6]);
    for (nu, r) in cols.iter().enumerate() {
        let iat = bas.ao2at[nu];
        qsh[bas.ao2sh[nu]] -= r[0];
        for k in 0..3 {
            dpat[iat][k] -= r[1 + k];
        }
        for k in 0..6 {
            qpat[iat][k] -= r[4 + k];
        }
    }
}
