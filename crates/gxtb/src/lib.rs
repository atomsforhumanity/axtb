//! g-xTB (T. Froitzheim, M. Müller, A. Hansen, S. Grimme, ChemRxiv 2025,
//! doi:10.26434/chemrxiv-2025-bjxvt) in two versions:
//!
//! - **2.0.1**, the version of the grimme-lab/g-xtb 2.0.1 release (`xtb --gxtb`), decoded from
//!   that program, method `"gxtb-v2.0.1"` and the default [`Version`]: module [`v2`] with its
//!   own parameters ([`data_v2`]);
//! - **v1.1**, the public, preprint version as implemented by the g-xTB 1.1.0 program
//!   (grimme-lab/g-xtb release v1.1.0), method `"gxtb-v1.1"`: the other modules
//!   ([`scf`] and the terms it uses).
//!
//! **Use.** [`Calculator`] (the interface of `axtb::Calculator`; the axtb CLI, Python package
//! and qccodec codec select it by method name): `Calculator::new(Version::V2_0_1, numbers,
//! positions, charge, uhf, ScfOptions::default())`, then [`Calculator::singlepoint`] (energy,
//! analytic gradient) and [`Calculator::hessian`] (2.0.1: analytic by default, response or
//! numerical; v1.1: numerical). [`ScfOptions`] holds the settings of both versions and refuses
//! those of the other one. Lower down, [`v2::scf::Setup`] + [`v2::scf::Setup::scf`] (2.0.1) and
//! [`scf::Setup`] + [`scf::Setup::scf`] (v1.1).
//!
//! **Conventions** ([`Conventions`]). [`Conventions::Gxtb`] reproduces the programs:
//! - 2.0.1: total energies to ≤ 1.8e-12 Eh (9.1e-12 Eh for (H2O)100), the same SCF iterations,
//!   per-term energies and analytic gradients (`tests/reference_v2.rs`, `tests/gradient_v2.rs`,
//!   84 molecules H–Lr; `docs/gxtb-v2.0.1-spec.md` §10);
//! - v1.1: total energies, energy terms, Mulliken charges and orbital energies to ~1e-12 Eh
//!   with the program's Fock matrix (`tests/reference.rs`).
//!
//! The default [`Conventions::Axtb`]: for 2.0.1 CODATA 2022 units, an integer electron count
//! and smooth cutoffs (the same energy expression and SCF); for v1.1 the exact derivative of
//! the energy as Fock matrix, so the SCF is stationary (energies up to ~4e-5 Eh lower, energy
//! differences within ~1e-5 Eh).
//!
//! **Restarts.** [`Calculator`] starts each SCF from the previous geometry's converged state
//! ([`ScfOptions::guess`], [`GuessState`]; what the programs' `xtbrestart` and `gxtbrestart`
//! hold), by default in the axtb conventions.
//!
//! **Finite electronic temperature** (`ScfOptions::etemp`; the programs' `--etemp` and `-tel`):
//! Fermi smearing with the entropy term in both versions ([`v2::fermi`], [`fermi`], on the
//! shared rules of [`occupations`]).
//!
//! **Implicit solvent:** [`composite`], E_gxtb + [E_GFN2+ALPB/GBSA - E_GFN2] for energies,
//! gradients and Hessians in one process ([`CompositeCalculator`]).
//!
//! **Threads:** a calculator with `threads > 1` owns one rayon pool of that size; the setups'
//! loops and, from [`parallel::DENSE_MIN_AO`] basis functions on, their dense linear algebra
//! run in it, in fixed chunks summed in order ([`parallel`]).
//!
//! **The v1.1 modules**, each verified against the 1.1.0 program:
//! - parsers for the three parameter files ([`params`]) and the compiled-in tables
//!   ([`data`], [`data_runtime`]);
//! - the EEQ-BC charge model ([`eeqbc`]);
//! - the charge- and CN-adaptive q-vSZP basis ([`qvszp`]), its overlap matrix and Mulliken
//!   shell charges ([`ints`]);
//! - the g-xTB coordination number, the semi-classical repulsion and the atomic increments
//!   ([`classical`]);
//! - the extended-Hückel Hamiltonian with the diatomic-frame scaled overlap ([`h0`]) and the
//!   atomic correction potentials ([`acp`]);
//! - the second-order, third-order and Mulliken-exchange kernels ([`kernels`]); the
//!   first- to fourth-order and spin-polarization energies ([`es`]), the anisotropic
//!   electrostatics ([`aes`]); the Mulliken-exchange and onsite-exchange energies
//!   ([`exchange`]);
//! - the revD4 dispersion ([`disp`], built on `axtb_d4`);
//! - the Fock matrices of all terms ([`fock`]) and the SCF driver with the program's start
//!   density, Fock mixing and convergence criteria ([`scf`]);
//! - analytic nuclear gradients ([`scf::Setup::gradient`]): the explicit geometry derivative
//!   of every term ([`gradient`]) and the Z-vector response of the program's non-variational
//!   Fock matrix ([`response`]), checked against finite differences of the energy (≤ 6e-10
//!   Eh/bohr) and the program's numerical gradients (`tests/gradient.rs`);
//! - Fermi smearing ([`fermi`]) with the Z-vector response with fractional occupations and
//!   Fermi-level shifts for program-mode gradients (`tests/etemp.rs`).
//!
//! The energy expressions, the parameter-file layouts and the status of every term are
//! documented in `docs/gxtb-v2.0.1-spec.md` (a term-by-term delta to v1.1) and
//! `docs/gxtb-v1.1-spec.md`.

pub mod acp;
pub mod aes;
pub mod broken_symmetry;
pub mod calculator;
pub mod classical;
pub mod composite;
#[rustfmt::skip] // parameter tables keep their compact layout
pub mod data;
#[rustfmt::skip] // parameter tables keep their compact layout
pub mod data_runtime;
#[rustfmt::skip] // parameter tables keep their compact layout
pub mod data_v2;
pub mod disp;
pub mod eeqbc;
pub mod es;
pub mod exchange;
pub mod fermi;
pub mod fock;
pub mod gradient;
pub mod h0;
pub mod ints;
pub mod kernels;
pub mod occupations;
pub mod parallel;
pub mod params;
pub mod qvszp;
pub mod response;
pub mod scf;
pub mod timing;
pub mod v2;

pub use broken_symmetry::{BrokenSymmetry, Outcome};
pub use calculator::{Calculator, GuessState, Results, Terms, Version};
pub use composite::{CompositeCalculator, CompositeHessian, CompositeOptions, CompositeResults, Correction, PartInfo};
pub use scf::{singlepoint, Conventions, Energies, Guess, ScfOptions, ScfResult, Setup};

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
/// [`BOHR_ANGSTROM`]); [`v2::read_xyz`] reads it for 2.0.1.
pub fn read_xyz(text: &str, charge: f64, uhf: Option<usize>) -> Result<Molecule> {
    let (numbers, positions) = axtb::io::read_xyz_with(text, |x| x / BOHR_ANGSTROM)?;
    Molecule::new(&numbers, &positions, charge, uhf)
}
