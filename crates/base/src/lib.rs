// SPDX-License-Identifier: LGPL-3.0-or-later
//! The foundation of the axtb crates: what every method crate needs and none should copy.
//!
//! - [`Molecule`]: atomic numbers, positions (bohr), charge and unpaired electrons, checked once
//!   for every method ([`MIN_ATOM_DISTANCE`]);
//! - [`Error`] and [`Result`]: one error type with the kinds every front end reports;
//! - [`units`]: CODATA 2022 constants, and in [`units::reference`] the factors the reference
//!   programs' parametrizations were tabulated with (never to be updated);
//! - [`elements`]: the element tables the method crates share (Pauling electronegativities,
//!   covalent radii);
//! - [`math`]: the C library's error function (the reference programs' bits), 3-vector
//!   helpers and the Cartesian Hessian block of a pair term;
//! - [`ncoord`]: coordination numbers (the counting functions of GFN2-xTB, dftd4 and GFN-FF,
//!   pair tests, cutoff factors and the logarithmic cap) with their gradients, Jacobians and
//!   second derivatives;
//! - [`occupations`]: the Aufbau filling, tblite's and xtb's Fermi smearing and tblite's
//!   electronic entropy term;
//! - [`timing`]: wall time per named phase;
//! - [`fd`]: finite differences (stencils, columns over coordinates, symmetrization and the
//!   numerical Hessian of every method);
//! - `linalg`, with the feature `linalg`: dense linear algebra on faer (the SCF eigensolver
//!   with its optional run-time LAPACK library, density matrices, linear solves);
//! - `eeq`, with the feature `linalg`: the bordered linear system of the electronegativity
//!   equilibration charge models (EEQ 2019, GFN-FF's two EEQs, g-xTB's EEQ-BC), its constraint
//!   rows and solves.
//!
//! Without features its only dependency is rayon (the parallel loops of [`fd`], on the
//! caller's pool), so that `axtb-d4` and `axtb-solvation` can use it as well as the
//! tight-binding framework `axtb`; the feature `linalg` adds faer, for `axtb`, `axtb-gfnff`
//! and `axtb-gxtb`.

/// Element tables of mctc-lib that several crates use (axtb, axtb-d4, axtb-gfnff), generated
/// from the Fortran sources by `crates/axtb/scripts/gen_data.py`: the literals are mctc-lib's
/// digits, so every value is bit-identical to the compiled Fortran parameter. Index `Z - 1`.
#[rustfmt::skip] // generated tables keep their compact layout
pub mod elements;
#[cfg(feature = "linalg")]
pub mod eeq;
mod error;
pub mod fd;
#[cfg(feature = "linalg")]
pub mod linalg;
pub mod math;
mod molecule;
pub mod ncoord;
pub mod occupations;
pub mod timing;
pub mod units;

pub use error::{Error, Result};
pub use molecule::{check_positions, Molecule, MIN_ATOM_DISTANCE};
