//! The xtb versions whose GFN-FF the port reproduces. Everything that differs between them
//! is one table, [`ConventionParams`], resolved once when the topology is set up; the rest of
//! the code reads the table instead of testing the version.
//!
//! xtb 6.8.0's GFN-FF differs from 6.7.1's (the g-xTB 2.0.1 build of xtb, `26dd68d`) in four
//! places that change results for molecules (`docs/gfnff-spec.md` §11), two in `src/gfnff` and
//! two in code it shares with the rest of xtb:
//!
//! 1. the covalent radii (`covalentRadD3 * aatoau * 4/3`, `src/param/covalentradd3.f90`) are
//!    converted with mctc-lib's CODATA 2018 bohr (`src/mctc/convert.f90`, `autoaa =
//!    0.529177210904492434`) instead of 0.52917726: every coordination number, the HB
//!    coordination number and the bend/torsion damping move by ~1e-7 relative, energies by
//!    ~1e-9 Eh (6.8.0 changed the constant for all of xtb; the GFN-FF code is unchanged);
//! 2. a pi bond counts as one in the bond and torsion setup only if its Hückel bond order is
//!    above `pibo_thr = 1e3 eps` (6.7.1: above 0), so round-off bond orders (~1e-15) of
//!    bonds next to a pi system no longer give them pi-bond parameters (`gfnff_ini.f90`);
//! 3. `nbondmat_pbc` (`gfnff_ini2.F90`) drops every unpaired neighbor (one that does not have
//!    the atom as its neighbor, e.g. of a metal) from the reduced neighbor list before
//!    counting bonds between atom pairs; 6.7.1 shifted the original list once per unpaired
//!    neighbor and so removed the wrong entries when an atom had several;
//! 4. the Fermi smearing of the Hückel step (`fermismear`, `src/scc_core.f90`, 4000 K) stops
//!    its Newton search for the Fermi level when the electron count is within
//!    min(√ε, 1e5 ε) = 2.2e-11 (6.7.1: 1e-9), takes the derivative of the Fermi function as
//!    f(1 - f)/kT and clamps it beyond |ε - ε_F| > 50 kT; the pi bond orders move by up to
//!    ~1e-9, energies by ~1e-11 Eh ([`FermiSmearing`]).
//!
//! A fifth difference is the compiler's, not the source's: the 6.8.0 release is built with
//! Intel's LLVM compiler (ifx, oneAPI 2025), the 6.7.1 build with the classic ifort, and the two
//! inline `NORM2` of a 3-vector with different fused multiply-adds. That decides the order of
//! exactly equidistant neighbors in symmetric structures ([`TieNorms`]; the out-of-plane term
//! of NH3-like centres, the jth nearest neighbor of the setup and the HB lists).
//!
//! The rest of the 6.8.0 changes to `src/gfnff` do not change results for molecules: setup
//! printout switches, heap instead of stack arrays, OpenMP reductions written as critical
//! sections, periodic-only code (`get_damat_3d`, `ncoordLatP`), the bond list of the
//! harmonic 2020 variant (not GFN-FF 1.0.3) and the fragmentation of the optimizer's model
//! Hessian (`frag_hess.f90`).

/// Version of xtb's GFN-FF. xtb 6.8.0's differs from 6.7.1's in the bohr of the covalent
/// radii, the pi-bond threshold of the setup, the removal of one-sided neighbors before the
/// bond counts between atom pairs, the Fermi smearing of the Hückel step, and the compiler's
/// rounding of the distances that decide between equidistant neighbors
/// (`docs/gfnff-spec.md` §11).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Conventions {
    /// xtb 6.8.0 (`xtb --gfnff`, 4b8c987). axtb's default.
    #[default]
    Xtb680,
    /// xtb 6.7.1 as the g-xTB 2.0.1 build of xtb has it (`26dd68d`).
    Xtb671,
}

impl Conventions {
    /// `xtb-6.8.0` or `xtb-6.7.1`.
    pub fn label(self) -> &'static str {
        match self {
            Conventions::Xtb680 => "xtb-6.8.0",
            Conventions::Xtb671 => "xtb-6.7.1",
        }
    }

    /// The conventions of a calculation's conventions name: `axtb` (default), `xtb` and
    /// `xtb-6.8.0` select 6.8.0's GFN-FF (the force field is the same in axtb's and xtb's
    /// conventions), `xtb-6.7.1` 6.7.1's; `None` for any other name.
    pub fn from_calculation(name: &str) -> Option<Self> {
        match name.to_ascii_lowercase().as_str() {
            "axtb" | "default" | "xtb" | "xtb-6.8.0" => Some(Conventions::Xtb680),
            "xtb-6.7.1" => Some(Conventions::Xtb671),
            _ => None,
        }
    }

    /// The settings of these conventions.
    pub(crate) fn params(self) -> ConventionParams {
        match self {
            Conventions::Xtb680 => ConventionParams {
                rcov_aatoau: XTB680_AATOAU,
                pibo_threshold: 1.0e3 * f64::EPSILON,
                unpaired_neighbors: UnpairedNeighbors::Removed,
                fermi_smearing: FermiSmearing::Xtb680,
                tie_norms: TieNorms::Ifx,
            },
            Conventions::Xtb671 => ConventionParams {
                rcov_aatoau: XTB671_AATOAU,
                pibo_threshold: 0.0,
                unpaired_neighbors: UnpairedNeighbors::ShiftedOriginal,
                fermi_smearing: FermiSmearing::Xtb671,
                tie_norms: TieNorms::Ifort,
            },
        }
    }
}

impl std::fmt::Display for Conventions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// Ångström to bohr of xtb 6.7.1's and xtb 6.8.0's tables (from `axtb-base`).
pub use axtb_base::units::reference::{XTB671_AATOAU, XTB680_AATOAU};

/// How `nbondmat` removes the unpaired neighbors of an atom before counting the bonds between
/// atom pairs (point 3 of the module documentation).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum UnpairedNeighbors {
    /// Every unpaired neighbor is removed (xtb 6.8.0).
    Removed,
    /// For every unpaired neighbor, the entries of the original list after its position are
    /// shifted down by one into the reduced list (xtb 6.7.1), which removes the wrong entries
    /// when an atom has more than one.
    ShiftedOriginal,
}

/// The Fermi smearing of the Hückel pi systems (xtb's `fermismear`; point 4 of the module
/// documentation).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FermiSmearing {
    /// Newton search for the Fermi level to 1e-9 electrons; derivative e^x/(kT (e^x + 1)²);
    /// the occupations of the last evaluation (xtb 6.7.1;
    /// [`axtb_base::occupations::fill_fermismear`]).
    Xtb671,
    /// To min(√ε, 1e5 ε) electrons; derivative f(1 - f)/kT, occupations 0 and 1 beyond
    /// |x| > 50 (xtb 6.8.0; tblite's filling, [`axtb_base::occupations::fill_tblite`]).
    Xtb680,
}

/// How the reference binary rounds the distances whose order decides between equivalent
/// neighbors: Fortran `NORM2` of a 3-vector as the compiler inlines it (found in the
/// binaries' disassembly; the AVX2 code path, which the programs run on CPUs with AVX2).
/// Only these discrete decisions use them; every other distance is the plain square root of
/// the sum of squares.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TieNorms {
    /// ifort (xtb 6.7.1, the g-xTB 2.0.1 build): `jth_nb` and `nn_nearest_noM`
    /// √fma(z, z, fma(x, x, y²)); the out-of-plane sort of `gfnff_ini` √fma(z, z, fma(y, y, x²)).
    Ifort,
    /// ifx 2025 (xtb 6.8.0): `jth_nb` √fma(z, z, fma(y, y, x²)) for the neighbors it takes four
    /// at a time and √(fma(x, x, y²) + z²) for the remaining ones; `nn_nearest_noM`
    /// √fma(z, z, x² + y²); the out-of-plane sort √((x² + y²) + z²).
    Ifx,
}

impl TieNorms {
    /// Distance from atom i to its neighbor at position `k` of a list of `n` (`jth_nb`).
    #[inline]
    pub fn jth_nb(self, d: [f64; 3], k: usize, n: usize) -> f64 {
        let [x, y, z] = d;
        match self {
            TieNorms::Ifort => z.mul_add(z, x.mul_add(x, y * y)).sqrt(),
            TieNorms::Ifx if k < n - n % 4 => z.mul_add(z, y.mul_add(y, x * x)).sqrt(),
            TieNorms::Ifx => (x.mul_add(x, y * y) + z * z).sqrt(),
        }
    }

    /// Distance of `nn_nearest_noM` (the closest non-metal neighbor).
    #[inline]
    pub fn nearest(self, d: [f64; 3]) -> f64 {
        let [x, y, z] = d;
        match self {
            TieNorms::Ifort => z.mul_add(z, x.mul_add(x, y * y)).sqrt(),
            TieNorms::Ifx => z.mul_add(z, x * x + y * y).sqrt(),
        }
    }

    /// Distance of the out-of-plane neighbor sort of `gfnff_ini`.
    #[inline]
    pub fn out_of_plane(self, d: [f64; 3]) -> f64 {
        let [x, y, z] = d;
        match self {
            TieNorms::Ifort => z.mul_add(z, y.mul_add(y, x * x)).sqrt(),
            TieNorms::Ifx => ((x * x + y * y) + z * z).sqrt(),
        }
    }
}

/// The settings that differ between [`Conventions`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConventionParams {
    /// Ångström-to-bohr factor of the covalent radii (`rcov = r_D3 * aatoau * 4/3`; point 1).
    pub rcov_aatoau: f64,
    /// Hückel bond orders above this make a bond a pi bond in the bond and torsion setup
    /// (point 2; 6.7.1: 0, i.e. any positive bond order).
    pub pibo_threshold: f64,
    /// Removal of unpaired neighbors in `nbondmat` (point 3).
    pub unpaired_neighbors: UnpairedNeighbors,
    /// Fermi smearing of the Hückel step (point 4).
    pub fermi_smearing: FermiSmearing,
    /// Rounding of the distances that decide between equidistant neighbors.
    pub tie_norms: TieNorms,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names() {
        for (s, c) in [
            ("axtb", Conventions::Xtb680),
            ("xtb", Conventions::Xtb680),
            ("XTB-6.8.0", Conventions::Xtb680),
            ("xtb-6.7.1", Conventions::Xtb671),
        ] {
            assert_eq!(Conventions::from_calculation(s), Some(c));
        }
        assert_eq!(Conventions::from_calculation("tblite"), None);
        assert_eq!(Conventions::default(), Conventions::Xtb680);
    }

    #[test]
    fn bohr_constants() {
        // mctc-lib derives its bohr from CODATA 2018: hbar / (m_e c alpha)
        let h = 6.62607015e-34;
        let me = 9.1093837015e-31;
        let c = 299792458.0;
        let alpha = 7.2973525693e-3;
        let bohr = h / (2.0 * std::f64::consts::PI) / (me * c * alpha);
        assert_eq!(XTB680_AATOAU, 1.0 / (bohr * 1e10));
        assert!((XTB680_AATOAU / XTB671_AATOAU - 1.0).abs() < 2e-7);
    }
}
