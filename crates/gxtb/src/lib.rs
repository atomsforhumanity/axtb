//! g-xTB (T. Froitzheim, M. Müller, A. Hansen, S. Grimme, ChemRxiv 2025,
//! doi:10.26434/chemrxiv-2025-bjxvt) in two versions on one engine ([`model`]):
//!
//! - **2.0.1**, the version of the grimme-lab/g-xtb 2.0.1 release (`xtb --gxtb`), decoded from
//!   that program, method `"gxtb-v2.0.1"` and the default [`Version`], with its own parameters
//!   ([`data_v2`]);
//! - **v1.1**, the public, preprint version as implemented by the g-xTB 1.1.0 program
//!   (grimme-lab/g-xtb release v1.1.0), method `"gxtb-v1.1"`: the same engine with v1.1's
//!   parameters ([`params`], [`data`], [`data_runtime`]) and its forms of the terms, selected by
//!   the version in the [`model::Model`].
//!
//! **Use.** [`Calculator`] implements axtb-base's [`Method`] (the axtb CLI, Python package and
//! qccodec codec select it by method name): `Calculator::new(Version::V2_0_1, numbers,
//! positions, charge, uhf, Options::default())`, then [`Method::singlepoint`] (energy, analytic
//! gradient) and [`Method::hessian`] (2.0.1: analytic by default, response or numerical; v1.1:
//! numerical). [`Options`] holds the settings of both versions and refuses those of the other
//! one. Lower down, [`scf::Setup`] (`Setup::new(mol, Model::new(version, conventions))`) and
//! [`scf::Setup::scf`], with the settings of one SCF ([`ScfOptions`]).
//!
//! **Conventions** ([`Conventions`]). [`Conventions::Gxtb`] reproduces the programs:
//! - 2.0.1: total energies to ≤ 1.8e-12 Eh (9.1e-12 Eh for (H2O)100), the same SCF iterations,
//!   per-term energies and analytic gradients (`tests/reference_v2.rs`, `tests/gradient_v2.rs`,
//!   84 molecules H–Lr; `docs/gxtb-v2.0.1-spec.md` §10);
//! - v1.1: total energies, energy terms, Mulliken charges and orbital energies to ~1e-12 Eh
//!   with the program's Fock matrix, its start density, Fock mixing and convergence test
//!   (`tests/reference.rs`); what is the program's behaviour rather than v1.1's model (the start
//!   density, the eigenvalue cap, the Fock matrix that is not the derivative of the energy, its
//!   SCF strategy and the Z-vector gradient that Fock matrix needs) lives in one crate-private
//!   module.
//!
//! The default [`Conventions::Axtb`]: for 2.0.1 CODATA 2022 units, an integer electron count
//! and smooth cutoffs (the same energy expression and SCF); for v1.1 the exact derivative of
//! the energy as Fock matrix, so the SCF is stationary (energies up to ~4e-5 Eh lower, energy
//! differences within ~1e-5 Eh).
//!
//! **Restarts.** [`Calculator`] starts each SCF from the previous geometry's converged state
//! (its [`Wavefunction`](axtb::method::Wavefunction): what the programs' `xtbrestart` and
//! `gxtbrestart` hold), by default in the axtb conventions (the restart policy
//! `Options::scf.restart`).
//!
//! **Finite electronic temperature** (`Options::scf.etemp`; the programs' `--etemp` and `-tel`):
//! Fermi smearing with the entropy term in both versions ([`fermi`], on the shared rules of
//! [`occupations`]).
//!
//! **Implicit solvent:** [`composite`], E_gxtb + [E_GFN2+ALPB/GBSA - E_GFN2] for energies,
//! gradients and Hessians in one process ([`CompositeCalculator`]).
//!
//! **Threads:** a calculator with `threads > 1` owns one rayon pool of that size
//! (`axtb_base::threads::Threads`); the setups' loops and, from
//! `axtb_base::threads::DENSE_MIN_AO` basis functions on, their dense linear algebra run in
//! it, in fixed chunks summed in order (`axtb_base::threads` and [`parallel`]).
//!
//! Shared pieces: the EEQ-BC charge model ([`eeqbc`]), the q-vSZP contraction ([`qvszp`]), the
//! revD4 reference data ([`disp`]), the Fock builders ([`fock`]) and Mulliken analysis
//! ([`mulliken`]), row-major dense matrices, their conversions to faer and the dot products in
//! the crate's two rounding orders ([`dense`]), and the Krylov solvers of the response equations
//! ([`krylov`]: conjugate gradients for 2.0.1's, GMRES for v1.1's Z-vector equations).
//!
//! The energy expressions, the parameter-file layouts and the status of every term are
//! documented in `docs/gxtb-v2.0.1-spec.md` (a term-by-term delta to v1.1) and
//! `docs/gxtb-v1.1-spec.md`.

pub mod calculator;
pub mod composite;
#[rustfmt::skip] // parameter tables keep their compact layout
pub mod data;
#[rustfmt::skip] // parameter tables keep their compact layout
pub mod data_runtime;
#[rustfmt::skip] // parameter tables keep their compact layout
pub mod data_v2;
pub mod analytic;
pub mod basis;
pub mod cn;
pub mod coulomb;
pub mod dense;
pub mod disp;
pub mod eeqbc;
pub mod eeqbc_model;
pub mod exchange;
pub mod explicit;
pub mod fermi;
pub mod fock;
pub mod gradient;
pub mod h0;
pub mod hessian;
pub(crate) mod intermediates;
pub mod krylov;
pub mod model;
pub mod mulliken;
pub mod occupations;
pub mod options;
pub mod parallel;
pub mod params;
pub(crate) mod program110;
pub mod qvszp;
pub mod repulsion;
pub mod response;
pub mod revd4;
pub mod scf;
pub(crate) mod strategy;
pub mod tangent;
pub(crate) mod terms;
pub mod timing;

pub use axtb::{method, Method};
pub use calculator::{Calculator, ConvergerSettings, Options, Version, RESTART_ACCURACY};
pub use composite::{CompositeCalculator, CompositeOptions, Correction};
pub use options::{Conventions, ScfOptions};
pub use scf::{Energies, ScfResult, Setup};

use axtb::{Error, Molecule, Result};

/// Refuse a total charge that is not an integer (or not finite): both programs take integer
/// charges, and the electron count follows from it.
pub(crate) fn check_charge(mol: &Molecule) -> Result<()> {
    if mol.charge.fract() == 0.0 {
        Ok(())
    } else {
        Err(Error::Input(format!("g-xTB: the charge must be an integer, got {}", mol.charge)))
    }
}

/// Ångström per bohr as the g-xTB 1.1.0 program uses it: the single-precision literal
/// `0.52917726` widened to double (coordinates are divided by it on input).
pub const BOHR_ANGSTROM: f64 = 0.52917726_f32 as f64;

/// Read an XYZ file (Ångström) the way the g-xTB 1.1.0 program does (coordinates divided by
/// [`BOHR_ANGSTROM`]); [`model::read_xyz`] reads it for 2.0.1.
pub fn read_xyz(text: &str, charge: f64, uhf: Option<usize>) -> Result<Molecule> {
    let (numbers, positions) = axtb::io::read_xyz_with(text, |x| x / BOHR_ANGSTROM)?;
    Molecule::new(&numbers, &positions, charge, uhf)
}
