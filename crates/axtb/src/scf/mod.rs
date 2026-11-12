//! Building blocks of a self-consistent-charge tight-binding SCF that do not depend on the
//! method: the layout of the self-consistent variables ([`ScfVariables`]), the Mulliken
//! analysis that gives them from a density (`mulliken`, crate-private), the Fock matrix of
//! their potentials (`fock`, crate-private), the mixers between iterations ([`VectorMixer`],
//! [`FockMixer`]), the convergence tests ([`ConvergenceRule`]) and the SCF loop that runs a
//! method's [`ScfModel`] with them ([`run_scf`]).
//!
//! The SCC variables of a GFN-type method are Mulliken shell charges, atomic dipoles and
//! traceless atomic quadrupoles (and, for spin-polarized methods, shell magnetizations). Each is
//! `s_t = s0_t - Tr(P O_t)` for a symmetric AO operator `O_t`: the overlap restricted to the AOs
//! of one shell, or the dipole and quadrupole integrals around one atom restricted to its AOs.
//! The energy depends on the density through `Tr(H0 P)` and the variables, so the Fock matrix is
//! `F = H0 - Σ_t V_t O_t` with the potentials `V_t = ∂E/∂s_t` (GFN2-xTB: C. Bannwarth,
//! S. Ehlert, S. Grimme, J. Chem. Theory Comput. 15, 1652 (2019); the same forms as tblite's
//! Mulliken analysis and Fock assembly).

pub(crate) mod convergence;
mod driver;
pub(crate) mod fock;
pub(crate) mod mixing;
pub(crate) mod mulliken;
mod variables;

pub use convergence::{ConvergenceRule, Iteration, Verdict};
pub use driver::{run_scf, Accelerator, FockSource, ScfModel, ScfOutcome, ScfStrategy, Stop};
pub use mixing::{FockMixer, VectorMixer};
pub use variables::{ScfVariables, QUAD_OFFDIAG};

use faer::Mat;
use rayon::prelude::*;

/// Dot product with four independent accumulators (lets the compiler vectorize).
#[inline]
pub(crate) fn dot(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = [0.0; 4];
    let chunks = n / 4;
    for c in 0..chunks {
        for k in 0..4 {
            acc[k] += a[4 * c + k] * b[4 * c + k];
        }
    }
    let mut s = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    for i in 4 * chunks..n {
        s += a[i] * b[i];
    }
    s
}

/// Run `f(j, column j)` for every column of `m`, in parallel when asked (each column by one
/// task).
fn for_columns(m: &mut Mat<f64>, parallel: bool, f: impl Fn(usize, &mut [f64]) + Sync + Send) {
    let col = |(j, c): (usize, faer::ColMut<'_, f64>)| {
        f(j, c.try_as_col_major_mut().expect("the columns of a Mat are contiguous").as_slice_mut())
    };
    if parallel {
        m.par_col_iter_mut().enumerate().with_min_len(16).for_each(col);
    } else {
        m.col_iter_mut().enumerate().for_each(col);
    }
}
