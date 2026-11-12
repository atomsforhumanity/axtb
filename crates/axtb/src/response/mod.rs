//! Coupled-perturbed SCF: the linear response of a converged SCF to a perturbation of its Fock
//! and overlap matrices (nuclear displacements for Hessians and dipole derivatives).
//!
//! The pieces every method shares are written once here:
//! - [`OccupationResponse`] (`occupation`): the occupation algebra, i.e. the divided
//!   differences of the occupations between orbitals (with the degenerate-level rule, the
//!   spins summed into one response and the rows as parameters), the Fermi-level shifts and the
//!   resulting responses of the density and the energy-weighted density in the MO basis;
//! - [`MoSpace`] (`mospace`): the transforms between the AO basis and the rows of MO response
//!   matrices.
//!
//! Each method keeps its own solver for the coupled equations: GFN2-xTB's Fock matrix depends on
//! the density only through the SCC variables, so its response is a small dense system in their
//! space (`scc`, crate-private); g-xTB's exchange depends on the whole density matrix, so
//! g-xTB 2.0.1 solves in the space of orbital pairs (conjugate gradients, in axtb-gxtb).

pub mod mospace;
pub mod occupation;
pub(crate) mod scc;

pub use mospace::MoSpace;
pub use occupation::{occupied_count, within_kt, MoResponse, OccupationResponse};
pub(crate) use scc::Response;
