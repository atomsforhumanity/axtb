//! g-xTB 2.0.1, the version of g-xTB distributed with the grimme-lab/g-xtb 2.0.1 release
//! (`xtb --gxtb`: xtb 6.7.1 with a g-xTB build of tblite), decoded from that program
//! (`docs/gxtb-v2.0.1-spec.md`, a term-by-term delta to `docs/gxtb-v1.1-spec.md`).
//!
//! Same method family as v1.1 (q-vSZP basis, EEQ-BC charges, extended-Hückel H0 with
//! atomic correction potentials, first- to fourth-order and anisotropic electrostatics, spin
//! polarization, Mulliken-approximated exchange, revD4) with every parameter refitted, all 103
//! elements parametrised, and new or changed forms: an H0 anisotropy (atomic dipole shift)
//! term, a self-consistent repulsion, a bond-order correction and charge-dependent onsite
//! integrals in the exchange, screened revD4 damping, and tblite's SCF. Its Fock matrix is the
//! exact derivative of its energy, so the converged SCF is stationary and the analytic gradient
//! ([`scf::Setup::gradient`]) needs no response term.
//!
//! Conventions ([`crate::Conventions`]):
//! - `Gxtb` (program mode) reproduces the 2.0.1 program: mctc-lib's Ångström-to-bohr factor for
//!   every Ångström table, the program's fractional electron count (sum of the stored reference
//!   occupations), its hard cutoffs (CNs and repulsion 25 bohr, revD4 30/50/25 bohr, integral
//!   pair cutoff of tblite) and its SCF (no mixing in iteration 1, damping 0.2 in iterations 2-3,
//!   then DIIS on the full Fock matrix of both spins, 7 vectors).
//! - `Axtb` (default): the same energy expression, Fock matrix and SCF with an integer electron
//!   count, CODATA 2022 units for the Ångström tables, and the hard cutoffs replaced by smooth
//!   switches (quintic smoothstep from the program's cutoff to 1 bohr beyond it; no integral
//!   pair cutoff), so energy and gradient are continuous for any system size.

pub mod analytic;
pub mod basis;
pub mod cn;
pub mod coulomb;
pub mod disp;
pub mod eeqbc;
pub mod exchange;
pub mod explicit;
pub mod fermi;
pub mod gradient;
pub mod h0;
pub mod hessian;
pub(crate) mod intermediates;
pub mod repulsion;
pub mod response;
pub mod scf;
pub(crate) mod strategy;
pub mod tangent;
pub(crate) mod terms;

use crate::options::{Conventions, Version};
use axtb::units::reference::MCTC_AATOAU;

/// Highest atomic number with g-xTB 2.0.1 parameters.
pub const MAX_ELEM: usize = 103;

/// Width (bohr) of the smooth switches that replace the program's hard cutoffs in the default
/// conventions.
pub const SWITCH_WIDTH: f64 = 1.0;

/// The smoothed square root of a coordination number used by every CN-dependent term of
/// 2.0.1, f(CN) = sqrt(CN + 1e-12) - 1e-6, and its derivative.
#[inline]
pub fn sqrt_cn(cn: f64) -> (f64, f64) {
    let s = (cn + 1e-12).sqrt();
    (s - 1e-6, 0.5 / s)
}

/// What a g-xTB calculation is: the version (its parameters and forms) and the conventions, with
/// the constants that follow from them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Model {
    pub version: Version,
    pub conventions: Conventions,
    /// Ångström to bohr for the tables given in Ångström (D3 pair radii, Pyykkö radii) and
    /// the program's unit literals (mctc-lib's CODATA 2018 value in program mode, CODATA 2022
    /// otherwise).
    pub aatoau: f64,
}

impl Model {
    /// The model of `version` in `conventions`.
    pub fn new(version: Version, conventions: Conventions) -> Self {
        let aatoau = match conventions {
            Conventions::Gxtb => MCTC_AATOAU,
            Conventions::Axtb => axtb::units::AATOAU,
        };
        Self { version, conventions, aatoau }
    }

    /// g-xTB 2.0.1 in `conventions`.
    pub fn v2_0_1(conventions: Conventions) -> Self {
        Self::new(Version::V2_0_1, conventions)
    }

    /// g-xTB v1.1 in `conventions`.
    pub fn v1_1(conventions: Conventions) -> Self {
        Self::new(Version::V1_1, conventions)
    }

    /// Program mode (hard cutoffs, fractional electron count).
    #[inline]
    pub fn program(&self) -> bool {
        self.conventions == Conventions::Gxtb
    }

    /// D3 van der Waals pair radius (bohr) of atomic numbers `za`, `zb`.
    #[inline]
    pub fn vdw_pair(&self, za: usize, zb: usize) -> f64 {
        crate::eeqbc::vdw_rad_pair(za, zb) * self.aatoau
    }

    /// The shared van der Waals pair radius R̄_AB = 1/2 (s_A + s_B) R^D3_AB (bohr).
    #[inline]
    pub fn rvdw(&self, za: usize, zb: usize) -> f64 {
        use crate::data_v2::RVDW_SCALE;
        self.vdw_pair(za, zb) * (0.5 * (RVDW_SCALE[za - 1] + RVDW_SCALE[zb - 1]))
    }

    /// dftd4/mctc-lib `covalent_rad_d3` (4/3 Pyykkö, bohr): revD4 CN radii and the H0 radii.
    #[inline]
    pub fn covalent_rad_d3(&self, z: usize) -> f64 {
        if self.program() {
            crate::data_v2::D4_RCOV[z - 1]
        } else {
            axtb::param::covalent_rad_d3_codata2022(z)
        }
    }

    /// Radius of the H0 distance polynomial: 2.0.1's `get_rad` (the Pyykkö radius in Ångström
    /// times 1.889725949, with fixed values for H, F and Ne); v1.1's `com::rcov` (bohr; the
    /// program divides by their mean, [`super::h0::shell_poly`] by their sum).
    pub fn h0_rad(&self, z: usize) -> f64 {
        if self.version == Version::V1_1 {
            return crate::data::RCOV_COM[z - 1];
        }
        match z {
            1 => 0.7615595574470001,
            9 => 0.9448629745,
            10 => 1.15273282889,
            _ => self.covalent_rad_d3(z) * 0.75 / self.aatoau * 1.889725949,
        }
    }

    /// f(CN) and df/dCN of the CN-dependent parameters: 2.0.1's smoothed square root
    /// ([`sqrt_cn`]); v1.1's plain square root, whose derivative the 1.1.0 program sets to 0 at
    /// CN = 0.
    #[inline]
    pub fn cn_root(&self, cn: f64) -> (f64, f64) {
        match self.version {
            Version::V2_0_1 => sqrt_cn(cn),
            Version::V1_1 => {
                let s = cn.sqrt();
                (s, if s > 0.0 { 0.5 / s } else { 0.0 })
            }
        }
    }

    /// The cutoff (bohr) of the g-xTB and basis coordination numbers ([`Model::cut`]): 25 bohr
    /// in 2.0.1, none in v1.1.
    #[inline]
    pub fn cn_cutoff(&self) -> Option<f64> {
        match self.version {
            Version::V2_0_1 => Some(cn::CN_CUTOFF),
            Version::V1_1 => None,
        }
    }

    /// The primitive screening of the H0-basis overlaps and of the ACP projector overlaps
    /// (2.0.1: tblite's 25; v1.1: the 1.1.0 program's 20); the SCF-basis integrals use
    /// [`basis::INTCUT`] (25) in both versions.
    #[inline]
    pub fn intcut_h0(&self) -> f64 {
        match self.version {
            Version::V2_0_1 => basis::INTCUT,
            Version::V1_1 => crate::h0::INTCUT,
        }
    }

    /// Whether the H0 has 2.0.1's anisotropy term (the dipole shift a_A).
    #[inline]
    pub fn anisotropy(&self) -> bool {
        self.version == Version::V2_0_1
    }

    /// Cutoff factor of a pair at distance `r` for a cutoff radius `cutoff` and its derivative:
    /// program mode 1 up to the cutoff, 0 beyond; default conventions a quintic smoothstep
    /// from 1 at `cutoff` to 0 at `cutoff + SWITCH_WIDTH`.
    #[inline]
    pub fn cut(&self, r: f64, cutoff: f64) -> (f64, f64) {
        if self.program() {
            if r <= cutoff {
                (1.0, 0.0)
            } else {
                (0.0, 0.0)
            }
        } else {
            axtb_d4::smooth_cutoff(r, cutoff + SWITCH_WIDTH, SWITCH_WIDTH)
        }
    }

    /// Largest distance with a nonzero cutoff factor.
    #[inline]
    pub fn reach(&self, cutoff: f64) -> f64 {
        if self.program() {
            cutoff
        } else {
            cutoff + SWITCH_WIDTH
        }
    }
}

/// Real spherical shells of an element in q-vSZP order: angular momenta of the first
/// `NSHELL` shells.
pub fn element_shells(z: usize) -> &'static [crate::params::QvszpShell] {
    let b = qvszp();
    let el = b.element(z).expect("q-vSZP covers H-Lr");
    &el.shells[..crate::data_v2::NSHELL[z - 1]]
}

/// The q-vSZP shells of an element in `model`: 2.0.1's first `NSHELL` ([`element_shells`]),
/// every shell of the `basisq` file in v1.1.
pub fn model_shells(model: &Model, z: usize) -> &'static [crate::params::QvszpShell] {
    match model.version {
        Version::V2_0_1 => element_shells(z),
        Version::V1_1 => &qvszp().element(z).expect("q-vSZP covers H-Lr").shells,
    }
}

/// The q-vSZP basis tables (bit-identical in 1.1.0 and 2.0.1), parsed once.
pub fn qvszp() -> &'static crate::params::QvszpParams {
    static B: std::sync::OnceLock<crate::params::QvszpParams> = std::sync::OnceLock::new();
    B.get_or_init(crate::params::QvszpParams::v1_1)
}

/// Element check for g-xTB 2.0.1.
pub fn check_elements(numbers: &[usize]) -> axtb::Result<()> {
    if let Some(&z) = numbers.iter().find(|&&z| z == 0 || z > MAX_ELEM) {
        return Err(axtb::Error::UnsupportedElement { method: "g-xTB 2.0.1", z });
    }
    Ok(())
}

/// Element check for the version of `model` (both cover H-Lr).
pub fn check_model_elements(model: &Model, numbers: &[usize]) -> axtb::Result<()> {
    match model.version {
        Version::V2_0_1 => check_elements(numbers),
        Version::V1_1 => {
            let p = crate::params::gxtb_v1_1();
            for &z in numbers {
                if z == 0 || z > MAX_ELEM {
                    return Err(axtb::Error::UnsupportedElement { method: "g-xTB v1.1", z });
                }
                p.element(z)?;
            }
            Ok(())
        }
    }
}

/// Read an XYZ file (Ångström) for g-xTB 2.0.1: coordinates multiplied by the conventions'
/// Ångström-to-bohr factor (program mode: mctc-lib's, as the 2.0.1 program reads them).
pub fn read_xyz(text: &str, charge: f64, uhf: Option<usize>, conventions: Conventions) -> axtb::Result<axtb::Molecule> {
    let aatoau = Model::v2_0_1(conventions).aatoau;
    let (numbers, positions) = axtb::io::read_xyz(text, aatoau)?;
    axtb::Molecule::new(&numbers, &positions, charge, uhf)
}
