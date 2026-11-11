//! Dense linear algebra of the methods, on faer's kernels (feature `linalg`; faer is a large
//! dependency that `axtb-d4` and `axtb-solvation` do without):
//!
//! - [`GenEigSolver`]: the SCF eigensolver of the tight-binding methods, F C = S C ε, with the
//!   optional LAPACK library loaded at run time ([`runtime_lapack`], `AXTB_LAPACK`) and ports
//!   of LAPACK's `dsygst`, `dsterf` and `dstein`;
//! - [`weighted_outer`] and [`weighted_outer_sym`]: density matrices C diag(w) Cᵀ;
//! - [`lu_solve`]: dense linear systems.
//!
//! Parallelism is always explicit (faer's `Par`, on the caller's rayon pool): faer's high-level
//! solvers would run on its global parallelism, which starts rayon's global pool (one thread
//! per CPU) in every process.

mod density;
mod eigen;
mod lapack;
mod lu;
mod tridiag;

pub use density::{weighted_outer, weighted_outer_sym};
pub use eigen::{runtime_lapack, GenEigSolver};
pub use lu::lu_solve;
