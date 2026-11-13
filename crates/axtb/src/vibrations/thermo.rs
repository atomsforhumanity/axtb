// SPDX-License-Identifier: LGPL-3.0-or-later
//! Quasi-RRHO thermochemistry of an ideal gas: translation (ideal gas at pressure p),
//! rigid rotor (rotational symmetry number from the point group) and harmonic vibrations
//! whose entropy is interpolated to that of a free rotor below the rotor cutoff (Grimme's
//! modified RRHO, Chem. Eur. J. 2012, 18, 9955: Head-Gordon weights
//! w = 1 / (1 + (ω0/ω)^4), free-rotor moment μ' = μ B_av / (μ + B_av) with μ = ħ/(2ω) and B_av
//! the average molecular moment of inertia), as the xtb program computes it (`thermodyn` in
//! xtb's `src/thermo.f90`, driven by `print_thermo` in `src/main/property.F90`):
//!
//! - frequencies are scaled by `frequency_scale`; imaginary modes above `-imaginary_cutoff`
//!   are inverted and the others dropped (counted in [`Thermo::n_dropped`]);
//! - the zero-point energy is ½ Σ ω over the modes kept; the vibrational enthalpy is the
//!   harmonic one, the entropy and heat capacity are interpolated;
//! - H(T) = ZPVE + H_tr + H_rot + H_vib with H_tr = 5/2 kT (including pV); G = H − TS.
//!
//! In the `xtb` and `xtb-6.7.1` conventions everything is computed with the xtb program's
//! constants and unit conversions (cal/mol and back, its moment-of-inertia routine `axis2`,
//! frequencies below 1 cm⁻¹ in magnitude ignored, the symmetry number from the
//! three-character point-group label) so that the results equal the program's to its printed
//! digits. xtb 6.8.0 differs from 6.7.1 only in its bohr radius (mctc-lib's CODATA 2018 value
//! instead of 0.52917726 Å, in `axis2` and the free-rotor moment) and its linearity test
//! (`is_linear`: a unit-mass moment about the centroid below 1e-4 bohr², instead of a
//! rotational constant from `axis2` below 1e-10 cm⁻¹); `thermodyn` is the same. The `axtb`
//! conventions compute in atomic units with CODATA 2022 constants and use every vibrational
//! mode of the projected Hessian and the point group's true symmetry number.

use super::symmetry::{point_group, xtb_symmetry_number};
use super::{xtbc, Conventions, LinearTest};
use crate::units;
use crate::{Error, Result};

/// Settings of the thermochemistry.
#[derive(Clone, Debug)]
pub struct ThermoOptions {
    /// Temperatures in K (default 298.15). xtb reports the last one as the result.
    pub temperatures: Vec<f64>,
    /// Pressure of the ideal gas in Pa (default 101325, 1 atm, as xtb).
    pub pressure: f64,
    /// Rotor cutoff ω0 in cm⁻¹ of the entropy interpolation (xtb `sthr`, default 50); 0 or
    /// negative: harmonic oscillators only.
    pub rotor_cutoff: f64,
    /// Imaginary modes with |ω| below this (cm⁻¹, magnitude; xtb `ithr`, default 20) are
    /// treated as real; larger ones are left out.
    pub imaginary_cutoff: f64,
    /// Frequency scale factor (default 1; xtb's default is 1.03 for GFN-FF).
    pub frequency_scale: f64,
    /// Rotational symmetry number; `None` detects the point group.
    pub symmetry_number: Option<u32>,
    /// Tolerance (bohr) of the point-group detection (xtb `desy`, default 0.1).
    pub symmetry_threshold: f64,
    /// Above this many atoms the point group is not detected (C1 assumed; xtb `maxat`, 200).
    pub max_symmetry_atoms: usize,
}

impl Default for ThermoOptions {
    fn default() -> Self {
        Self {
            temperatures: vec![298.15],
            pressure: units::STANDARD_ATMOSPHERE,
            rotor_cutoff: 50.0,
            imaginary_cutoff: 20.0,
            frequency_scale: 1.0,
            symmetry_number: None,
            symmetry_threshold: 0.1,
            max_symmetry_atoms: 200,
        }
    }
}

/// Enthalpy (Eh), entropy (Eh/K) and heat capacity Cp (Eh/K) of one kind of motion.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Contribution {
    pub enthalpy: f64,
    pub entropy: f64,
    pub heat_capacity: f64,
}

/// Thermochemistry at one temperature (energies in Eh, entropies and heat capacities in Eh/K).
#[derive(Clone, Debug, PartialEq)]
pub struct ThermoPoint {
    pub temperature: f64,
    /// H(T) = ZPVE + H(T) − H(0): xtb's `H(T)/Eh`; add the electronic energy for the
    /// total enthalpy.
    pub enthalpy: f64,
    /// H(T) − H(0) (thermal enthalpy without the ZPVE; xtb's `H(0)-H(T)+PV` column).
    pub thermal_enthalpy: f64,
    pub entropy: f64,
    /// T·S.
    pub ts: f64,
    /// G(RRHO) = H(T) − T·S (xtb's `G(RRHO) contrib.`); add the electronic energy for the total
    /// free energy.
    pub g_rrho: f64,
    pub heat_capacity: f64,
    pub translation: Contribution,
    pub rotation: Contribution,
    pub vibration: Contribution,
}

/// Thermochemistry of a molecule.
#[derive(Clone, Debug, PartialEq)]
pub struct Thermo {
    /// Detected point group (`C1` above `max_symmetry_atoms`), even when the symmetry number
    /// was given.
    pub point_group: String,
    pub symmetry_number: u32,
    pub linear: bool,
    pub atom: bool,
    /// Rotational constants A >= B >= C in cm⁻¹; a vanishing moment of inertia (the axis of a
    /// linear molecule, below 3e-4 u Å²) gives 0.
    pub rotational_constants: [f64; 3],
    /// Molecular mass in u.
    pub mass: f64,
    /// Modes in the vibrational partition function, imaginary modes inverted, imaginary modes
    /// left out (xtb's `# imaginary freq.`).
    pub n_modes: usize,
    pub n_inverted: usize,
    pub n_dropped: usize,
    pub zpve: f64,
    pub pressure: f64,
    pub rotor_cutoff: f64,
    pub imaginary_cutoff: f64,
    pub frequency_scale: f64,
    /// One entry per temperature, in the order given.
    pub points: Vec<ThermoPoint>,
}

impl Thermo {
    /// The result at the last temperature (the one xtb reports).
    pub fn last(&self) -> &ThermoPoint {
        self.points.last().expect("at least one temperature")
    }
}

/// Moments of inertia as xtb's `axis2` computes them: rotational constants in cm⁻¹ from the
/// largest moment to the smallest (zero below 3e-4 u Å²), the average moment (kg m²) and the
/// mass (u). Positions are converted with xtb's bohr `autoaa` (Å), the inertia tensor
/// carries xtb's offsets of i·1e-10 on its packed elements, and the constants are its
/// `const1 = 1.66053`, `const2 = 16.8576522`.
fn axis2_xtb(masses: &[f64], positions: &[[f64; 3]], autoaa: f64) -> ([f64; 3], f64, f64) {
    let coord: Vec<[f64; 3]> = positions.iter().map(|p| [p[0] * autoaa, p[1] * autoaa, p[2] * autoaa]).collect();
    let mut sumw = 0.0;
    let mut sw = [0.0; 3];
    for (m, c) in masses.iter().zip(&coord) {
        sumw += m;
        for k in 0..3 {
            sw[k] += m * c[k];
        }
    }
    for s in &mut sw {
        *s /= sumw;
    }
    let mut t = [1e-10, 2e-10, 3e-10, 4e-10, 5e-10, 6e-10];
    for (m, c) in masses.iter().zip(&coord) {
        let (x, y, z) = (c[0] - sw[0], c[1] - sw[1], c[2] - sw[2]);
        t[0] += m * (y * y + z * z);
        t[1] -= m * x * y;
        t[2] += m * (z * z + x * x);
        t[3] -= m * z * x;
        t[4] -= m * y * z;
        t[5] += m * (x * x + y * y);
    }
    // packed lower triangle: (1,1) (2,1) (2,2) (3,1) (3,2) (3,3)
    let a = [[t[0], t[1], t[3]], [t[1], t[2], t[4]], [t[3], t[4], t[5]]];
    let (mut eig, _) = super::symmetry::eigh3(&a);
    let mut rot = [0.0; 3];
    let mut xyzmom = [0.0; 3];
    for i in 0..3 {
        if eig[i] < 3e-4 {
            eig[i] = 0.0;
            rot[i] = 0.0;
        } else {
            rot[i] = 2.9979245e4 * xtbc::CONST2 / eig[i];
        }
        xyzmom[i] = eig[i] * xtbc::CONST1;
    }
    let abc = [rot[2] / 2.9979245e4, rot[1] / 2.9979245e4, rot[0] / 2.9979245e4];
    let avmom = 1e-47 * (xyzmom[0] + xyzmom[1] + xyzmom[2]) / 3.0;
    (abc, avmom, sumw)
}

/// Principal moments of inertia (ascending, mₑ bohr²) of masses in u.
pub(crate) fn principal_moments(masses: &[f64], positions: &[[f64; 3]]) -> ([f64; 3], [[f64; 3]; 3], [f64; 3]) {
    let mtot: f64 = masses.iter().sum();
    let mut com = [0.0; 3];
    for (m, p) in masses.iter().zip(positions) {
        for k in 0..3 {
            com[k] += m * p[k] / mtot;
        }
    }
    let mut t = [[0.0; 3]; 3];
    for (m, p) in masses.iter().zip(positions) {
        let me = m / units::ELECTRON_MASS_IN_U;
        let r = [p[0] - com[0], p[1] - com[1], p[2] - com[2]];
        let r2 = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
        for i in 0..3 {
            for j in 0..3 {
                t[i][j] += me * (if i == j { r2 } else { 0.0 } - r[i] * r[j]);
            }
        }
    }
    let (w, v) = super::symmetry::eigh3(&t);
    (w, v, com)
}

/// Smallest principal moment (u Å²) below which a molecule counts as linear (xtb `axis2`).
pub(crate) const LINEAR_MOMENT_U_AA2: f64 = 3e-4;

/// Moment of inertia threshold in mₑ bohr² equivalent to [`LINEAR_MOMENT_U_AA2`].
pub(crate) fn linear_moment_au() -> f64 {
    LINEAR_MOMENT_U_AA2 / units::ELECTRON_MASS_IN_U * units::AATOAU * units::AATOAU
}

fn check(opts: &ThermoOptions) -> Result<()> {
    if opts.temperatures.is_empty() || opts.temperatures.iter().any(|t| !(t.is_finite() && *t > 0.0)) {
        return Err(Error::Input("thermochemistry needs at least one positive temperature".into()));
    }
    if !(opts.pressure.is_finite() && opts.pressure > 0.0) {
        return Err(Error::Input(format!("pressure must be positive, got {}", opts.pressure)));
    }
    if !(opts.frequency_scale.is_finite() && opts.frequency_scale > 0.0) {
        return Err(Error::Input(format!("frequency scale must be positive, got {}", opts.frequency_scale)));
    }
    if !opts.rotor_cutoff.is_finite() || !opts.imaginary_cutoff.is_finite() || !opts.symmetry_threshold.is_finite() {
        return Err(Error::Input("rotor cutoff, imaginary cutoff and symmetry threshold must be finite".into()));
    }
    if opts.symmetry_number == Some(0) {
        return Err(Error::Input("symmetry number must be at least 1".into()));
    }
    Ok(())
}

/// Thermochemistry from the vibrational frequencies (cm⁻¹, imaginary negative) of a molecule
/// (atomic numbers, positions in bohr, masses in u).
pub fn thermochemistry(
    numbers: &[usize],
    positions: &[[f64; 3]],
    masses: &[f64],
    frequencies: &[f64],
    conventions: Conventions,
    opts: &ThermoOptions,
) -> Result<Thermo> {
    check(opts)?;
    let nat = numbers.len();
    let ithr = -opts.imaginary_cutoff.abs();
    // symmetry
    let pg = if nat > opts.max_symmetry_atoms {
        "C1".to_string()
    } else {
        point_group(numbers, positions, opts.symmetry_threshold).name
    };
    let params = super::params(conventions);
    match params.thermo_bohr {
        Some(autoaa) => {
            let (abc, avmom, wt) = axis2_xtb(masses, positions, autoaa);
            let linear = match params.linear {
                LinearTest::UnitMassMoments => super::is_linear_xtb680(positions),
                _ => abc[2] < 1e-10,
            };
            let atom = abc[0] + abc[1] + abc[2] < 1e-6;
            let sym = opts.symmetry_number.unwrap_or_else(|| xtb_symmetry_number(&pg, linear));
            // modes: |ω| > 1 cm⁻¹, scaled, small imaginary inverted, the rest dropped
            let mut vibs: Vec<f64> =
                if atom { Vec::new() } else { frequencies.iter().copied().filter(|f| f.abs() > 1.0).collect() };
            for v in &mut vibs {
                *v *= opts.frequency_scale;
            }
            let mut n_inverted = 0;
            for v in &mut vibs {
                if *v < 0.0 && *v > ithr {
                    *v = -*v;
                    n_inverted += 1;
                }
            }
            let kept: Vec<f64> = vibs.iter().filter(|v| **v > 0.0).map(|v| v * xtbc::RCMTOAU).collect();
            let n_dropped = vibs.len() - kept.len();
            let zp = 0.5 * kept.iter().sum::<f64>();
            let points = opts
                .temperatures
                .iter()
                .map(|&t| {
                    thermodyn_xtb(
                        abc,
                        avmom,
                        linear,
                        atom,
                        sym as f64,
                        wt,
                        &kept,
                        t,
                        opts.rotor_cutoff,
                        opts.pressure,
                        zp,
                        autoaa,
                    )
                })
                .collect();
            Ok(Thermo {
                point_group: pg,
                symmetry_number: sym,
                linear,
                atom,
                rotational_constants: [abc[2], abc[1], abc[0]],
                mass: wt,
                n_modes: kept.len(),
                n_inverted,
                n_dropped,
                zpve: zp,
                pressure: opts.pressure,
                rotor_cutoff: opts.rotor_cutoff,
                imaginary_cutoff: opts.imaginary_cutoff.abs(),
                frequency_scale: opts.frequency_scale,
                points,
            })
        }
        None => {
            let (mom, _, _) = principal_moments(masses, positions);
            let thr = linear_moment_au();
            let atom = nat == 1 || mom[2] < thr;
            let linear = !atom && mom[0] < thr;
            let mom: Vec<f64> = mom.iter().map(|&m| if m < thr { 0.0 } else { m }).collect();
            // rotational constants 1/(2I) in Eh, from the largest moment to the smallest (as axis2)
            let b: Vec<f64> = mom.iter().rev().map(|&m| if m > 0.0 { 0.5 / m } else { 0.0 }).collect();
            let abc = [b[0], b[1], b[2]];
            let avmom = (mom[0] + mom[1] + mom[2]) / 3.0;
            let mass: f64 = masses.iter().sum();
            let sym = opts.symmetry_number.unwrap_or_else(|| super::symmetry::rotational_symmetry_number(&pg));
            let mut vibs: Vec<f64> =
                if atom { Vec::new() } else { frequencies.iter().map(|f| f * opts.frequency_scale).collect() };
            let mut n_inverted = 0;
            for v in &mut vibs {
                if *v < 0.0 && *v > ithr {
                    *v = -*v;
                    n_inverted += 1;
                }
            }
            let kept: Vec<f64> = vibs.iter().filter(|v| **v > 0.0).map(|v| v / units::HARTREE_TO_WAVENUMBER).collect();
            let n_dropped = vibs.len() - kept.len();
            let zp = 0.5 * kept.iter().sum::<f64>();
            let sthr = opts.rotor_cutoff / units::HARTREE_TO_WAVENUMBER;
            let p_au = opts.pressure / units::AU_PRESSURE_PA;
            let points = opts
                .temperatures
                .iter()
                .map(|&t| {
                    thermodyn_au(
                        abc,
                        avmom,
                        linear,
                        atom,
                        sym as f64,
                        mass / units::ELECTRON_MASS_IN_U,
                        &kept,
                        t,
                        sthr,
                        p_au,
                        zp,
                    )
                })
                .collect();
            Ok(Thermo {
                point_group: pg,
                symmetry_number: sym,
                linear,
                atom,
                rotational_constants: [abc[2], abc[1], abc[0]].map(|x| x * units::HARTREE_TO_WAVENUMBER),
                mass,
                n_modes: kept.len(),
                n_inverted,
                n_dropped,
                zpve: zp,
                pressure: opts.pressure,
                rotor_cutoff: opts.rotor_cutoff,
                imaginary_cutoff: opts.imaginary_cutoff.abs(),
                frequency_scale: opts.frequency_scale,
                points,
            })
        }
    }
}

/// Head-Gordon weight of the harmonic oscillator, 1 / (1 + (ω0/ω)^4) (1 for ω0 < 0).
fn chg_switching(omega: f64, sthr: f64) -> f64 {
    if sthr >= 0.0 {
        1.0 / (1.0 + (sthr / omega).powi(4))
    } else {
        1.0
    }
}

/// xtb's `thermodyn`, literally: vibrational sums in Eh, then R in cal/(mol K) and back to Eh
/// with its kcal/mol factor. `vibs` in Eh; `sthr_rcm` in cm⁻¹; `pressure` in Pa; `autoaa` the
/// program's bohr radius in Å.
fn thermodyn_xtb(
    abc_rcm: [f64; 3],
    avmom_si: f64,
    linear: bool,
    atom: bool,
    sym: f64,
    molmass: f64,
    vibs: &[f64],
    t: f64,
    sthr_rcm: f64,
    pressure: f64,
    zp: f64,
    autoaa: f64,
) -> ThermoPoint {
    let r = xtbc::MOLAR_GAS * xtbc::JTOCAL;
    let pi = xtbc::PI;
    let beta = 1.0 / xtbc::KB / t;
    let sthr = sthr_rcm * xtbc::RCMTOAU;
    let aatoau = 1.0 / autoaa;
    let avmom = avmom_si * xtbc::KGTOME * aatoau * aatoau * 1.0e20;
    let (a, b, c) = (abc_rcm[0] * xtbc::RCMTOAU, abc_rcm[1] * xtbc::RCMTOAU, abc_rcm[2] * xtbc::RCMTOAU);
    let mut h_vib = 0.0;
    let mut cpvib = 0.0;
    let mut s_vib = 0.0;
    let cp_rr = 0.5 / (beta * beta);
    for &omega in vibs {
        let ewj = (-omega * beta).exp();
        h_vib += omega * ewj / (1.0 - ewj);
        let cp_ho = omega * omega * ewj / (1.0 - ewj) / (1.0 - ewj);
        let mut mu = 0.5 / (omega + 1.0e-14);
        mu = mu * avmom / (mu + avmom);
        let (sv_ho, sv_rr) = if omega > 0.0 {
            (omega * beta * ewj / (1.0 - ewj) - (1.0 - ewj).ln(), 0.5 + (pi / beta * 2.0 * mu).sqrt().ln())
        } else {
            (0.0, 0.0)
        };
        let wofrot = 1.0 - chg_switching(omega, sthr);
        cpvib += (1.0 - wofrot) * cp_ho + wofrot * cp_rr;
        s_vib += r * ((1.0 - wofrot) * sv_ho + wofrot * sv_rr);
    }
    let h_vib = h_vib * r * beta * t;
    let cpvib = cpvib * r * beta * beta;
    let (mut h_rot, mut cprot, mut s_rot) = (0.0, 0.0, 0.0);
    if !atom {
        if linear {
            h_rot = r * t;
            cprot = r;
            s_rot = r * ((1.0 / (beta * a * sym)).ln() + 1.0);
        } else {
            h_rot = 3.0 * r * t / 2.0;
            cprot = 3.0 * r / 2.0;
            s_rot = 0.5 * r * (-2.0 * sym.ln() + (pi / (a * b * c * (beta * beta * beta))).ln() + 3.0);
        }
    }
    let h_tr = 5.0 * r * t / 2.0;
    let cptr = 5.0 * r / 2.0;
    let p_au = if pressure == units::STANDARD_ATMOSPHERE { xtbc::ATMTOAU } else { pressure * xtbc::PATOAU };
    let s_tr = r * (2.5 * (t * xtbc::KB).ln() + 1.5 * (xtbc::AMUTOAU * molmass / (2.0 * pi)).ln() - p_au.ln() + 2.5);
    let cptot = cptr + cpvib + cprot;
    let s_tot = s_tr + s_vib + s_rot;
    let h_tot = h_tr + h_vib + h_rot;
    // cal/mol -> Eh
    let e = |x: f64| x / 1000.0 * xtbc::KCALTOAU;
    let ht = e(h_tot);
    let et = ht + zp;
    let ts = s_tot * t / 1000.0 * xtbc::KCALTOAU;
    ThermoPoint {
        temperature: t,
        enthalpy: et,
        thermal_enthalpy: ht,
        entropy: e(s_tot),
        ts,
        g_rrho: et - ts,
        heat_capacity: e(cptot),
        translation: Contribution { enthalpy: e(h_tr), entropy: e(s_tr), heat_capacity: e(cptr) },
        rotation: Contribution { enthalpy: e(h_rot), entropy: e(s_rot), heat_capacity: e(cprot) },
        vibration: Contribution { enthalpy: e(h_vib), entropy: e(s_vib), heat_capacity: e(cpvib) },
    }
}

/// The same model in atomic units with CODATA 2022 constants. `abc`, `vibs`, `sthr` in Eh,
/// `avmom` in mₑ bohr², `mass` in mₑ, `p` in Eh/bohr³.
fn thermodyn_au(
    abc: [f64; 3],
    avmom: f64,
    linear: bool,
    atom: bool,
    sym: f64,
    mass: f64,
    vibs: &[f64],
    t: f64,
    sthr: f64,
    p: f64,
    zp: f64,
) -> ThermoPoint {
    let pi = std::f64::consts::PI;
    let kb = units::KB_HARTREE;
    let kt = kb * t;
    let beta = 1.0 / kt;
    // vibrations (entropies and heat capacities in units of k)
    let (mut h_vib, mut s_vib, mut c_vib) = (0.0, 0.0, 0.0);
    for &omega in vibs {
        let x = omega * beta;
        let em1 = -(-x).exp_m1(); // 1 - e^-x
        let n = (-x).exp() / em1; // e^-x / (1 - e^-x)
        h_vib += omega * n;
        let c_ho = x * x * n / em1;
        let sv_ho = x * n - em1.ln();
        let mu = 0.5 / omega;
        let mu = if avmom > 0.0 { mu * avmom / (mu + avmom) } else { mu };
        let sv_rr = 0.5 + (2.0 * pi * mu * kt).sqrt().ln();
        let w = 1.0 - chg_switching(omega, sthr);
        s_vib += (1.0 - w) * sv_ho + w * sv_rr;
        c_vib += (1.0 - w) * c_ho + w * 0.5;
    }
    let (mut h_rot, mut s_rot, mut c_rot) = (0.0, 0.0, 0.0);
    if !atom {
        if linear {
            h_rot = kt;
            c_rot = 1.0;
            s_rot = (kt / (abc[0] * sym)).ln() + 1.0;
        } else {
            h_rot = 1.5 * kt;
            c_rot = 1.5;
            s_rot = 0.5 * (-2.0 * sym.ln() + (pi * kt * kt * kt / (abc[0] * abc[1] * abc[2])).ln() + 3.0);
        }
    }
    let h_tr = 2.5 * kt;
    let c_tr = 2.5;
    let s_tr = 2.5 * kt.ln() + 1.5 * (mass / (2.0 * pi)).ln() - p.ln() + 2.5;
    let ht = h_tr + h_rot + h_vib;
    let s = kb * (s_tr + s_rot + s_vib);
    let et = ht + zp;
    let ts = t * s;
    ThermoPoint {
        temperature: t,
        enthalpy: et,
        thermal_enthalpy: ht,
        entropy: s,
        ts,
        g_rrho: et - ts,
        heat_capacity: kb * (c_tr + c_rot + c_vib),
        translation: Contribution { enthalpy: h_tr, entropy: kb * s_tr, heat_capacity: kb * c_tr },
        rotation: Contribution { enthalpy: h_rot, entropy: kb * s_rot, heat_capacity: kb * c_rot },
        vibration: Contribution { enthalpy: h_vib, entropy: kb * s_vib, heat_capacity: kb * c_vib },
    }
}
