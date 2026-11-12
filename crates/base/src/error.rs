//! The one error type of the axtb crates.

use std::fmt;

/// Errors returned by axtb.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// Invalid input (structure, charge/spin, settings).
    Input(String),
    /// An element is not parametrized for the chosen method.
    UnsupportedElement { method: &'static str, z: usize },
    /// A force field's topology setup failed for this structure (the reference program stops
    /// too).
    Topology(String),
    /// The self-consistent field did not converge.
    ScfNotConverged { iterations: usize, energy_change: f64, residual: f64 },
    /// A numerical failure (e.g. overlap matrix not positive definite).
    Numerical(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Input(msg) => write!(f, "invalid input: {msg}"),
            Error::UnsupportedElement { method, z } => {
                write!(f, "{method} has no parameters for element Z={z}")
            }
            Error::Topology(msg) => write!(f, "topology setup failed: {msg}"),
            Error::ScfNotConverged { iterations, energy_change, residual } => write!(
                f,
                "SCF not converged in {iterations} iterations (last energy change {energy_change:.3e} Eh, residual {residual:.3e})"
            ),
            Error::Numerical(msg) => write!(f, "numerical failure: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

/// Result alias.
pub type Result<T> = std::result::Result<T, Error>;
