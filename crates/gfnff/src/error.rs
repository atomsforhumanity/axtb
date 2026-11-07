//! Errors of the GFN-FF crate.

use std::fmt;

#[derive(Debug, Clone)]
pub enum Error {
    /// Invalid input (atoms, charges, options).
    Input(String),
    /// An element without GFN-FF parameters.
    UnsupportedElement(usize),
    /// The topology setup failed (xtb would stop too).
    Topology(String),
    /// A numerical failure (singular EEQ matrix, eigensolver).
    Numerical(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Input(s) => write!(f, "invalid input: {s}"),
            Error::UnsupportedElement(z) => write!(f, "GFN-FF has no parameters for element Z={z} (H-Lr only)"),
            Error::Topology(s) => write!(f, "GFN-FF topology setup failed: {s}"),
            Error::Numerical(s) => write!(f, "numerical failure: {s}"),
        }
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
