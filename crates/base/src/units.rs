// SPDX-License-Identifier: LGPL-3.0-or-later
//! Physical constants and unit conversions.
//!
//! Physical constants are NIST CODATA 2022 values as curated in AFH's qcref package
//! (`qcref/src/qcref/data/codata/2022.csv`, from
//! <https://physics.nist.gov/cuu/Constants/Table/allascii.txt>; E. Tiesinga, P. J. Mohr,
//! D. B. Newell, B. N. Taylor, CODATA Internationally Recommended 2022 Values of the
//! Fundamental Physical Constants, NIST SRD 121). Derived conversions follow qcref's
//! `data/constants/constants.csv` expressions.
//!
//! Constants that are *part of a reference parametrization* (the values the original
//! programs used when their parameters were fitted or tabulated) are not physical
//! constants to be updated: changing them changes the method's results. They live in
//! [`reference`](mod@reference) with the reason for each.

/// Bohr radius in Ångström. CODATA 2022 "Bohr radius" 5.291 772 105 44 e-11 m
/// (qcref `BOHR_RADIUS`; `BOHR_TO_ANGSTROM = BOHR_RADIUS * 1e10`).
pub const AUTOAA: f64 = 5.29177210544e-11 * 1e10;
/// Ångström to bohr (qcref `ANGSTROM_TO_BOHR = 1 / BOHR_TO_ANGSTROM`).
pub const AATOAU: f64 = 1.0 / AUTOAA;

/// Hartree per kelvin. CODATA 2022 "kelvin-hartree relationship" 3.166 811 563 4564 e-6 Eh
/// (qcref `KELVIN_HARTREE_RELATIONSHIP`); converts the electronic temperature.
pub const KB_HARTREE: f64 = 3.1668115634564e-6;

/// Hartree in eV. CODATA 2022 "Hartree energy in eV" 27.211 386 245 981 eV
/// (qcref `HARTREE_TO_EV`).
pub const HARTREE_TO_EV: f64 = 27.211386245981;

/// Hartree in joule. CODATA 2022 "hartree-joule relationship" 4.359 744 722 2060 e-18 J.
pub const HARTREE_TO_JOULE: f64 = 4.3597447222060e-18;
/// Avogadro constant, CODATA 2022 (exact) 6.022 140 76 e23 mol⁻¹.
pub const AVOGADRO: f64 = 6.02214076e23;
/// Thermochemical kilocalorie in joule, 4184 J (NIST SP 811 Appendix B.8; qcref `KCAL_TO_JOULE`).
pub const KCAL_TO_JOULE: f64 = 4184.0;
/// Hartree in kcal/mol (qcref `HARTREE_TO_KCAL_PER_MOL = HARTREE_TO_JOULE * AVOGADRO_NUMBER / KCAL_TO_JOULE`).
pub const HARTREE_TO_KCAL_PER_MOL: f64 = HARTREE_TO_JOULE * AVOGADRO / KCAL_TO_JOULE;

/// Electron mass in unified atomic mass units. CODATA 2022 "electron mass in u"
/// 5.485 799 090 441 e-4 u (qcref `ELECTRON_MASS_IN_U`); for mass-weighted Hessians.
pub const ELECTRON_MASS_IN_U: f64 = 5.485799090441e-4;
/// Hartree in inverse centimeters. CODATA 2022 "hartree-inverse meter relationship"
/// 2.194 746 313 6314 e7 m⁻¹, divided by 100.
pub const HARTREE_TO_WAVENUMBER: f64 = 2.1947463136314e7 / 100.0;
/// Fine-structure constant α, CODATA 2022 7.297 352 5643 e-3 (qcref `FINE_STRUCTURE_CONSTANT`).
pub const FINE_STRUCTURE: f64 = 7.2973525643e-3;
/// Atomic unit of pressure Eh/a0³ in pascal, from CODATA 2022 "hartree-joule relationship"
/// and "Bohr radius".
pub const AU_PRESSURE_PA: f64 = HARTREE_TO_JOULE / (AUTOAA * 1e-10 * AUTOAA * 1e-10 * AUTOAA * 1e-10);
/// Standard atmosphere in pascal (exact; CODATA 2022 "standard atmosphere").
pub const STANDARD_ATMOSPHERE: f64 = 101325.0;
/// IR intensity in km/mol per squared dipole derivative with respect to a mass-weighted
/// normal coordinate in atomic units (e² / mₑ): N_A π α² a0 / 3, from the CODATA 2022 values
/// above (A = N_A π / (3 c²) |∂μ/∂Q|² in Gaussian atomic units, c = 1/α).
pub const AU_TO_KM_PER_MOL: f64 =
    AVOGADRO * std::f64::consts::PI * FINE_STRUCTURE * FINE_STRUCTURE * AUTOAA * 1e-10 / 3.0 / 1000.0;

/// Constants that define the reference parametrizations. Do not replace them with newer
/// CODATA values: the parameters were tabulated with these, and the reference programs'
/// results depend on them.
pub mod reference {
    use std::f64::consts::PI;

    // mctc-lib (used by tblite) derives its bohr radius from CODATA 2018 constants
    // (mctc-lib src/mctc/io/codata2018.f90, convert.f90); tblite's radius tables
    // (covalent radii, atomic radii) are converted to bohr with it.
    const PLANCK_2018: f64 = 6.62607015e-34;
    const ELECTRON_MASS_2018: f64 = 9.1093837015e-31;
    const SPEED_OF_LIGHT: f64 = 299792458.0;
    const FINE_STRUCTURE_2018: f64 = 7.2973525693e-3;
    const HBAR_2018: f64 = PLANCK_2018 / (2.0 * PI);
    const BOHR_METER_2018: f64 = HBAR_2018 / (ELECTRON_MASS_2018 * SPEED_OF_LIGHT * FINE_STRUCTURE_2018);
    /// mctc-lib's Ångström-to-bohr factor (CODATA 2018 derived), used in tblite's tables.
    pub const MCTC_AATOAU: f64 = 1.0 / (BOHR_METER_2018 * 1e10);

    /// Boltzmann constant (Eh/K) used by xtb (`xtb_mctc_constants::kB`) and tblite for
    /// the electronic temperature.
    pub const KB_XTB_TBLITE: f64 = 3.166808578545117e-06;

    /// Ångström to bohr of xtb 6.7.1's tables (`autoaa = 0.52917726`, not CODATA).
    pub const XTB671_AATOAU: f64 = 1.0 / 0.52917726;

    /// Ångström to bohr of xtb 6.8.0's tables: mctc-lib's CODATA 2018 bohr as xtb writes it
    /// (src/mctc/convert.f90 `autoaa = 5.29177210904492434e-1_wp`).
    pub const XTB680_AATOAU: f64 = 1.0 / 5.29177210904492434e-1;
}
