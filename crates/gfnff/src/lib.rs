//! GFN-FF: the generic force field of Spicher and Grimme (Angew. Chem. Int. Ed. 59,
//! 15665 (2020), doi:10.1002/anie.202004239), ported from xtb's implementation
//! (`src/gfnff`, LGPL-3.0-or-later) and validated against the xtb 6.8.0 program (default) and
//! the xtb 6.7.1 program ([`Conventions`]).
//!
//! * [`Calculator::new`] sets up the [`Topology`] (neighbors, hybridization, rings, pi systems
//!   with Hueckel bond orders, topology charges, bonded terms, HB/XB candidates) from a
//!   structure and keeps it for every later geometry (as xtb does within an optimization);
//! * [`Calculator::singlepoint`] gives energies, analytic gradients and EEQ charges
//!   ([`Results`]), [`Calculator::hessian`] and [`Calculator::numerical_hessian`] a
//!   [`Hessian`] with dipole derivatives;
//! * molecules only (no periodic boundary conditions).
//!
//! Units: bohr and Hartree. Errors: [`Error::Input`] and [`Error::UnsupportedElement`] for
//! input the force field does not take, [`Error::Topology`] for a structure its setup cannot
//! handle, [`Error::Numerical`] for a failed or non-finite calculation; no NaN or infinity is
//! returned.
//!
//! Threads: a calculator owns one rayon pool of exactly the requested size; with one thread
//! nothing is spawned. The pair loops use fixed chunks summed in a fixed order and the dense
//! linear algebra runs faer's kernels with the calculator's thread count, so results repeat
//! bit for bit on a given thread count. Between thread counts they can differ by round-off,
//! from the LU factorizations of the EEQ systems (faer's parallel kernels split them by
//! thread count above ~100 atoms): C50H102, (H2O)200, (H2O)333 and crambin on 1, 2, 4 and 8
//! threads gave topology charges ≤ 3e-12 e, energies ≤ 1.2e-12 Eh, gradients ≤ 9e-14 Eh/bohr
//! and analytic Hessians ≤ 3e-13 Eh/bohr² apart; the 24 reference structures (up to 113
//! atoms) gave identical bits.

mod calculator;
mod cells;
mod cn;
mod conventions;
#[rustfmt::skip] // parameter tables keep their compact layout
mod data;
mod energy;
mod error;
mod geom;
mod hessian;
mod linalg;
mod param;
mod real;
mod setup;
mod setup_util;
mod terms;
mod topology;

pub use calculator::{Calculator, Options, Results};
pub use conventions::Conventions;
pub use energy::Energies;
pub use error::{Error, Result};
pub use hessian::Hessian;
pub use topology::{Angle, Bond, BondDistance, BondHb, Topology, Torsion};

/// Method name used by the CLI, Python and qccodec wiring.
pub const METHOD: &str = "gfnff";

/// True for the names that select GFN-FF.
pub fn is_method(name: &str) -> bool {
    let key: String = name.to_ascii_lowercase().chars().filter(|c| c.is_ascii_alphanumeric()).collect();
    key == "gfnff"
}
