//! The one error type of the front ends.

use serde::ser::SerializeMap;
use serde::{Serialize, Serializer};
use std::fmt;

/// Why a request failed. Every method's errors map to these kinds, so that the CLI (`kind` of
/// its JSON error) and Python (the exception class) tell the same story.
#[derive(Debug, Clone)]
pub enum Error {
    /// Invalid request: structure, charge and spin, settings, names, options that do not apply
    /// to the method.
    Input(String),
    /// An element the method has no parameters for (`z`, its atomic number).
    UnsupportedElement { z: usize, message: String },
    /// GFN-FF could not set up the topology of the structure (an input problem: xtb stops too).
    Topology(String),
    /// The SCF did not converge: the last attempt's iterations, energy change (Eh) and
    /// residual.
    ScfNotConverged { iterations: usize, energy_change: f64, residual: f64 },
    /// A numerical failure (singular matrices, non-finite results).
    Numerical(String),
}

impl Error {
    /// The kind as the front ends name it: `input`, `unsupported_element`, `topology`,
    /// `scf_not_converged` or `numerical`.
    pub fn kind(&self) -> &'static str {
        match self {
            Error::Input(_) => "input",
            Error::UnsupportedElement { .. } => "unsupported_element",
            Error::Topology(_) => "topology",
            Error::ScfNotConverged { .. } => "scf_not_converged",
            Error::Numerical(_) => "numerical",
        }
    }

    /// Whether the request is at fault (input, unsupported element, topology): Python raises
    /// `axtb.InputError`, a `ValueError`.
    pub fn is_input(&self) -> bool {
        matches!(self, Error::Input(_) | Error::UnsupportedElement { .. } | Error::Topology(_))
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Input(msg) => write!(f, "invalid input: {msg}"),
            Error::UnsupportedElement { message, .. } => f.write_str(message),
            Error::Topology(msg) => write!(f, "GFN-FF topology setup failed: {msg}"),
            Error::ScfNotConverged { iterations, energy_change, residual } => write!(
                f,
                "SCF not converged in {iterations} iterations (last energy change {energy_change:.3e} Eh, residual {residual:.3e})"
            ),
            Error::Numerical(msg) => write!(f, "numerical failure: {msg}"),
        }
    }
}

impl std::error::Error for Error {}

/// The error as the CLI writes it: `kind`, `message` and the kind's own fields (`z`;
/// `iterations`, `energy_change`, `residual`).
impl Serialize for Error {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        let mut m = s.serialize_map(None)?;
        m.serialize_entry("kind", self.kind())?;
        m.serialize_entry("message", &self.to_string())?;
        match self {
            Error::UnsupportedElement { z, .. } => m.serialize_entry("z", z)?,
            Error::ScfNotConverged { iterations, energy_change, residual } => {
                m.serialize_entry("iterations", iterations)?;
                m.serialize_entry("energy_change", energy_change)?;
                m.serialize_entry("residual", residual)?;
            }
            _ => {}
        }
        m.end()
    }
}

impl From<axtb::Error> for Error {
    fn from(e: axtb::Error) -> Self {
        match e {
            axtb::Error::Input(msg) => Error::Input(msg),
            axtb::Error::UnsupportedElement { z, .. } => Error::UnsupportedElement { z, message: e.to_string() },
            axtb::Error::ScfNotConverged { iterations, energy_change, residual } => {
                Error::ScfNotConverged { iterations, energy_change, residual }
            }
            axtb::Error::Numerical(msg) => Error::Numerical(msg),
        }
    }
}

impl From<axtb_gfnff::Error> for Error {
    fn from(e: axtb_gfnff::Error) -> Self {
        match e {
            axtb_gfnff::Error::Input(msg) => Error::Input(msg),
            axtb_gfnff::Error::UnsupportedElement(z) => Error::UnsupportedElement { z, message: e.to_string() },
            axtb_gfnff::Error::Topology(msg) => Error::Topology(msg),
            axtb_gfnff::Error::Numerical(msg) => Error::Numerical(msg),
        }
    }
}

/// Result alias.
pub type Result<T> = std::result::Result<T, Error>;
