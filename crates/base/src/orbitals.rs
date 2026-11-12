// SPDX-License-Identifier: LGPL-3.0-or-later
//! Molecular orbitals as data: the contracted Gaussian shells they are expanded in ([`Cgto`]),
//! the orbital sets of a converged SCF ([`Orbitals`], [`MoSet`]) and the orbital energies and
//! occupations of each spin channel ([`OrbitalLevels`]). The method crates fill them; `axtb`
//! writes them as Molden files (`axtb::molden::ToMolden`).

use crate::{Error, Result};

/// Highest angular momentum supported by the integral code.
pub const MAXL: usize = 4;
/// Maximum number of primitives in a contracted Gaussian.
pub const MAXPRIM: usize = 16;

/// Number of spherical components for angular momentum `l`.
#[inline]
pub const fn nsph(l: usize) -> usize {
    2 * l + 1
}

/// Number of Cartesian components for angular momentum `l`.
#[inline]
pub const fn ncart(l: usize) -> usize {
    (l + 1) * (l + 2) / 2
}

/// A contracted Cartesian Gaussian shell; the coefficients include the normalization of
/// the x^l component (the spherical transformation completes the normalization).
#[derive(Clone, Copy, Debug)]
pub struct Cgto {
    /// Angular momentum.
    pub l: usize,
    /// Number of primitives.
    pub nprim: usize,
    /// Primitive exponents.
    pub alpha: [f64; MAXPRIM],
    /// Contraction coefficients.
    pub coeff: [f64; MAXPRIM],
}

impl Cgto {
    /// Smallest primitive exponent.
    #[inline]
    pub fn min_alpha(&self) -> f64 {
        self.alpha[..self.nprim].iter().copied().fold(f64::INFINITY, f64::min)
    }
}

/// Spin of a set of orbitals.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Spin {
    Alpha,
    Beta,
}

impl Spin {
    /// The label of the Molden `Spin=` field.
    pub fn label(&self) -> &'static str {
        match self {
            Spin::Alpha => "Alpha",
            Spin::Beta => "Beta",
        }
    }
}

/// One set of molecular orbitals.
#[derive(Clone, Debug)]
pub struct MoSet {
    pub spin: Spin,
    /// Orbital energies (Eh).
    pub energies: Vec<f64>,
    /// Occupations: 0 to 2 in the single set of a closed shell, 0 to 1 in an alpha or beta set.
    pub occupations: Vec<f64>,
    /// Coefficients orbital by orbital: `coefficients[k * nao + mu]` is AO `mu` in orbital
    /// `k`, in axtb's AO order (atom by atom, shell by shell, m = -l..l; p: y, z, x).
    pub coefficients: Vec<f64>,
}

/// Converged molecular orbitals with the basis they are expanded in.
#[derive(Clone, Debug)]
pub struct Orbitals {
    /// Atomic numbers and positions (bohr).
    pub numbers: Vec<usize>,
    pub positions: Vec<[f64; 3]>,
    /// Contracted shells of every atom ([`Cgto`] convention: coefficients include the
    /// normalization of the x^l primitive).
    pub shells: Vec<Vec<Cgto>>,
    /// One set (closed shell) or an alpha and a beta set (open shell).
    pub sets: Vec<MoSet>,
}

impl Orbitals {
    /// Number of (spherical) basis functions.
    pub fn nao(&self) -> usize {
        self.shells.iter().flatten().map(|s| nsph(s.l)).sum()
    }
}

/// Which functions the `[MO]` coefficients of a Molden file refer to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MoldenFunctions {
    /// Spherical functions (`[5D]`, `[7F]`), as the calculation uses them (default).
    #[default]
    Spherical,
    /// Cartesian functions (`[6D]`, `[10F]`), each component normalized.
    Cartesian,
}

impl std::str::FromStr for MoldenFunctions {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "spherical" | "pure" | "5d" => Ok(Self::Spherical),
            "cartesian" | "6d" => Ok(Self::Cartesian),
            other => Err(Error::Input(format!("unknown Molden functions {other:?} (spherical or cartesian)"))),
        }
    }
}

/// Orbital energies (Eh, ascending) and occupations of one spin channel.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct OrbitalLevels {
    pub energies: Vec<f64>,
    pub occupations: Vec<f64>,
}

impl OrbitalLevels {
    /// Electrons in the channel.
    pub fn electrons(&self) -> f64 {
        self.occupations.iter().sum()
    }
}
