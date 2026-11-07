// SPDX-License-Identifier: LGPL-3.0-or-later
//! DFT-D4 dispersion for GFN2-xTB, ported from dftd4 4.3.1, mctc-lib 0.7.0 and
//! tblite 0.8.0 (`tblite_disp_d4`).
//!
//! Conventions: positions in bohr (`&[[f64; 3]]`), atomic numbers 1-based,
//! energies in Hartree, gradients in Eh/bohr accumulated with `+=`, charges are
//! atomic partial charges (positive = electron deficient, tblite `wfn%qat`).
//! Molecular (non-periodic) systems only. No global state and no threads of its own: the
//! methods with a `parallel` argument run on the caller's rayon pool when it is true, with
//! the same results (see [`D4Dispersion`]).
//!
//! Typical SCF use (what tblite does for GFN2-xTB):
//!
//! ```no_run
//! use axtb_d4::D4Dispersion;
//! # let numbers = [8usize, 1, 1];
//! # let positions = [[0.0f64; 3]; 3];
//! # let q = [0.0f64; 3];
//! let disp = D4Dispersion::gfn2(&numbers).unwrap();    // once per molecule
//! let cache = disp.update(&positions);                // once per geometry
//! let mut e_atom = vec![0.0; 3];
//! let mut grad = vec![[0.0; 3]; 3];
//! disp.three_body(&cache, &mut e_atom, Some(&mut grad), false); // non-SCC, q = 0
//! // every SCF iteration:
//! let mut e2 = vec![0.0; 3];
//! let mut vat = vec![0.0; 3];                         // dE/dq_A
//! disp.energy_and_potential(&cache, &q, &mut e2, &mut vat);
//! // after convergence:
//! disp.gradient(&cache, &q, &mut grad);               // fixed-q geometry gradient
//! ```

#[rustfmt::skip] // parameter tables keep their compact layout
pub mod data;
mod disp;
mod hessian;
mod model;
mod ncoord;

pub use data::MAX_REF;
pub use disp::{smooth_cutoff, D4Cache, D4Dispersion, RationalDamping, RealspaceCutoff};
pub use model::{
    covalent_radius, r4r2_value, trapzd, zeta, zeta_dzeta, AtomicC6, ChargeModel, D4Model, ElementReferences,
    RefWeights, AATOAU, DEFAULT_GA, DEFAULT_GC, DEFAULT_WF,
};
pub use ncoord::{count as cn_count, dcount as cn_dcount, en_factor, erf, KCN};

/// Errors of the D4 model setup.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum D4Error {
    /// Element without D4 reference data (Z = 0, Z > 118, or 104 <= Z <= 111).
    UnsupportedElement { atom: usize, number: usize },
}

impl std::fmt::Display for D4Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            D4Error::UnsupportedElement { atom, number } => {
                write!(f, "D4: unsupported element Z={number} (atom {atom})")
            }
        }
    }
}

impl std::error::Error for D4Error {}

#[cfg(test)]
mod tests;
