// SPDX-License-Identifier: LGPL-3.0-or-later
//! The foundation of the axtb crates: what every method crate needs and none should copy.
//!
//! - [`Molecule`]: atomic numbers, positions (bohr), charge and unpaired electrons, checked once
//!   for every method ([`MIN_ATOM_DISTANCE`]);
//! - [`Error`] and [`Result`]: one error type with the kinds every front end reports;
//! - [`units`]: CODATA 2022 constants, and in [`units::reference`] the factors the reference
//!   programs' parametrizations were tabulated with (never to be updated);
//! - [`math`]: the C library's error function (the reference programs' bits) and 3-vector
//!   helpers;
//! - [`timing`]: wall time per named phase;
//! - [`fd`]: finite differences (stencils, columns over coordinates, symmetrization and the
//!   numerical Hessian of every method).
//!
//! Its only dependency is rayon (the parallel loops of [`fd`], on the caller's pool), so that
//! `axtb-d4`, `axtb-solvation` and `axtb-gfnff` can use it as well as the tight-binding
//! framework `axtb`.

mod error;
pub mod fd;
pub mod math;
mod molecule;
pub mod timing;
pub mod units;

pub use error::{Error, Result};
pub use molecule::{check_positions, Molecule, MIN_ATOM_DISTANCE};
