//! Dense linear algebra of the methods, on faer's kernels (feature `linalg`; faer is a large
//! dependency that `axtb-d4` and `axtb-solvation` do without):
//!
//! - [`GenEigSolver`]: the SCF eigensolver of the tight-binding methods (GFN2-xTB, g-xTB v1.1
//!   and 2.0.1), F C = S C ε, with the optional LAPACK library loaded at run time
//!   ([`runtime_lapack`], `AXTB_LAPACK`);
//! - [`sym_eig`]: the standard symmetric eigenproblem, for the small ones outside an SCF
//!   (GFN-FF's Hückel systems, the vibrational analysis);
//! - [`weighted_outer`] and [`weighted_outer_sym`]: density matrices C diag(w) Cᵀ;
//! - [`Lu`] and [`gauss_last_max`]: dense linear systems, two strategies whose pivot choices
//!   differ on ties (each caller keeps the one it was validated with);
//! - [`zeros`]: zero matrices written by the pool's threads (large matrices' first-touch page
//!   faults spread over the threads).
//!
//! Parallelism is always explicit (faer's `Par`, on the caller's rayon pool): faer's high-level
//! solvers would run on its global parallelism, which starts rayon's global pool (one thread
//! per CPU) in every process.
//!
//! # The SCF eigensolver
//!
//! The overlap is Cholesky-factorized once per geometry (S = L Lᵀ); every SCF iteration
//! then reduces F to standard form, A = L⁻¹ F L⁻ᵀ, solves the symmetric eigenproblem of A and
//! back-transforms its eigenvectors, C = L⁻ᵀ U. The reduction and the back-transformation
//! never form L⁻¹: they are triangular solves, or (in [`GenEigSolver::solve_lowest`], from
//! 200 rows on) the two-sided reduction of LAPACK's `dsygst` (ported to faer's kernels, or the
//! run-time library's own), which takes half the flops of two triangular solves with n
//! right-hand sides. faer's parallel tridiagonalization (matrices above ~313 rows) sums one
//! partial matrix-vector product per thread of the pool: the results repeat bit for bit on a
//! given thread count and differ between thread counts by round-off, as xtb's with MKL do; the
//! rest of the faer path gives the same bits on every thread count.
//!
//! The symmetric eigenproblems of 200 or more rows run in a LAPACK library loaded at run
//! time when the environment variable `AXTB_LAPACK` names one (see [`runtime_lapack`] and the
//! README's Threads section); faer solves them otherwise, and when the library fails.

mod density;
mod eigen;
mod lapack;
mod lu;
mod tridiag;
mod zeros;

pub use density::{weighted_outer, weighted_outer_sym};
pub use eigen::{runtime_lapack, sym_eig, GenEigSolver};
pub use lu::{gauss_last_max, Lu};
pub use zeros::zeros;
