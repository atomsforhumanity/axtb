// SPDX-License-Identifier: LGPL-3.0-or-later
//! The results of every method: [`SinglePoint`] and [`Hessian`].

use super::composite::{CompositeReport, PartReport};
use super::HessianMethod;
use crate::orbitals::{OrbitalLevels, Orbitals};

/// Whether an energy term is a piece of the partition of the energy or a subtotal of pieces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TermKind {
    /// A piece: the pieces of a single point sum to its energy (to rounding), unless the method
    /// documents a part that is only inside a subtotal.
    Piece,
    /// A sum of other contributions (e.g. the electronic energy).
    Subtotal,
}

/// One energy term (Eh) of a single point, named as the method's reference program reports it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Term {
    pub name: &'static str,
    pub value: f64,
    pub kind: TermKind,
}

impl Term {
    /// A piece of the partition.
    pub fn piece(name: &'static str, value: f64) -> Self {
        Self { name, value, kind: TermKind::Piece }
    }

    /// A subtotal.
    pub fn subtotal(name: &'static str, value: f64) -> Self {
        Self { name, value, kind: TermKind::Subtotal }
    }
}

/// The results of a single point.
#[derive(Clone, Debug)]
pub struct SinglePoint {
    /// Total energy (Eh).
    pub energy: f64,
    /// The method's energy terms in its reference program's order, pieces and subtotals
    /// together ([`TermKind`]).
    pub terms: Vec<Term>,
    /// Eh/bohr, N x 3 (when asked for).
    pub gradient: Option<Vec<[f64; 3]>>,
    /// Atomic charges (e): Mulliken (SCF methods) or the force field's.
    pub charges: Vec<f64>,
    /// Dipole moment about the origin (e bohr).
    pub dipole: [f64; 3],
    /// The electronic structure of an SCF method; `None` without electrons.
    pub electronic: Option<Electronic>,
    /// What only some methods report.
    pub extras: Extras,
    /// Wall time per phase (s).
    pub timings: Vec<(&'static str, f64)>,
}

impl SinglePoint {
    /// The value of the term `name`.
    pub fn term(&self, name: &str) -> Option<f64> {
        self.terms.iter().find(|t| t.name == name).map(|t| t.value)
    }

    /// The sum of the pieces, in order.
    pub fn pieces(&self) -> f64 {
        self.terms.iter().filter(|t| t.kind == TermKind::Piece).map(|t| t.value).sum()
    }

    /// How the SCF went (`None` without an SCF).
    pub fn scf(&self) -> Option<&ScfReport> {
        self.electronic.as_ref().map(|e| &e.scf)
    }
}

/// The electronic structure of an SCF single point.
#[derive(Clone, Debug)]
pub struct Electronic {
    /// Mulliken shell charges, atomic dipoles (N x 3) and quadrupoles (N x 6: xx, xy, yy, xz,
    /// yz, zz, traceless).
    pub shell_charges: Vec<f64>,
    pub atomic_dipoles: Vec<[f64; 3]>,
    pub atomic_quadrupoles: Vec<[f64; 6]>,
    /// The molecular quadrupole about the origin (e bohr²), in the form of
    /// [`MethodInfo::quadrupole`](super::MethodInfo::quadrupole).
    pub quadrupole: [f64; 6],
    /// Orbital energies and occupations per spin channel: one for a closed shell (occupations
    /// up to 2), alpha and beta for an open shell.
    pub levels: Vec<OrbitalLevels>,
    /// HOMO–LUMO gap (Eh, the method's and conventions' rule); `None` without a LUMO.
    pub homo_lumo_gap: Option<f64>,
    /// The converged orbitals with their basis ([`Want::orbitals`](super::Want::orbitals)).
    pub orbitals: Option<Orbitals>,
    /// Mayer–Wiberg bond orders, N x N row-major ([`Want::bond_orders`](super::Want::bond_orders)).
    pub bond_orders: Option<Vec<f64>>,
    pub scf: ScfReport,
}

/// How the SCF of a single point went.
#[derive(Clone, Debug, PartialEq)]
pub struct ScfReport {
    /// Iterations of the attempt that converged.
    pub iterations: usize,
    /// Where it started: `previous` (a given or the previous geometry's wavefunction) or the
    /// method's own start (`eeq`, `sad`, `eeq-bc`).
    pub start: &'static str,
    /// The mixer that produced the result (GFN2-xTB: `broyden`, `diis`, `diis-fallback`).
    pub mixer_used: Option<&'static str>,
    /// Every attempt of the method's own plan (GFN2-xTB's mixers); empty without one.
    pub attempts: Vec<Attempt>,
    /// The eigensolver: `faer`, or the run-time LAPACK library.
    pub eigensolver: String,
}

/// One SCF attempt.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Attempt {
    pub mixer: &'static str,
    pub iterations: usize,
    pub converged: bool,
}

/// What only some methods report.
#[derive(Clone, Debug, Default)]
pub struct Extras {
    /// GFN-FF: its topology and the hydrogen and halogen bonds of this geometry.
    pub gfnff: Option<GfnffReport>,
    /// A composite: its definition, components and parts.
    pub composite: Option<Box<CompositeReport>>,
}

/// GFN-FF's report of a single point.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GfnffReport {
    /// Hydrogen bonds (case 1, then cases 2 and 3) and halogen bonds at this geometry.
    pub n_hbonds: [usize; 2],
    pub n_xbonds: usize,
    pub topology: GfnffTopology,
}

/// GFN-FF's topology (set up once; 0-based atom indices).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct GfnffTopology {
    pub bonds: Vec<[usize; 2]>,
    pub pi_bond_orders: Vec<f64>,
    pub bond_types: Vec<i32>,
    pub hybridization: Vec<i32>,
    pub pi_atoms: Vec<bool>,
    pub fragments: Vec<usize>,
    pub fragment_charges: Vec<f64>,
    pub topology_charges: Vec<f64>,
    pub n_angles: usize,
    pub n_torsions: usize,
}

/// A nuclear Hessian.
#[derive(Clone, Debug)]
pub struct Hessian {
    /// Energy and gradient at the reference geometry.
    pub energy: f64,
    pub gradient: Vec<[f64; 3]>,
    /// Row-major `3N x 3N` matrix of second derivatives (Eh/bohr²), symmetrized.
    pub hessian: Vec<f64>,
    /// Largest asymmetry |H_ij - H_ji| before symmetrization.
    pub max_asymmetry: f64,
    /// Dipole derivatives ∂μ/∂R_x (e; one row per Cartesian coordinate x = 3·atom + k), for
    /// IR intensities; `None` when not available.
    pub dipole_derivatives: Option<Vec<[f64; 3]>>,
    /// How it was computed.
    pub method: HessianMethod,
    /// Wall time per phase in seconds.
    pub timings: Vec<(&'static str, f64)>,
    /// A composite's parts (with their Hessian methods); empty otherwise.
    pub parts: Vec<PartReport>,
}
