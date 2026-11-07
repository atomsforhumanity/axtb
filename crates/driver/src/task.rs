//! What a call computes besides the energy, checked against the method.

use crate::error::{Error, Result};
use crate::names::{self, Method};
use crate::request::Request;
use axtb::vibrations::{Conventions as VibConventions, ThermoOptions, VibOptions};

/// Settings of the thermochemistry; `None`: the default.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ThermoSettings {
    pub temperatures: Option<Vec<f64>>,
    pub pressure: Option<f64>,
    pub rotor_cutoff: Option<f64>,
    pub imaginary_cutoff: Option<f64>,
    pub frequency_scale: Option<f64>,
    pub symmetry_number: Option<u32>,
    pub symmetry_threshold: Option<f64>,
}

/// Settings of the vibrational analysis; `None`: the default.
#[derive(Clone, Debug, PartialEq)]
pub struct VibSettings {
    /// Default: the calculation's (an analysis of a given Hessian: `axtb`).
    pub conventions: Option<VibConventions>,
    pub masses: Option<Vec<f64>>,
    /// `None`: no thermochemistry.
    pub thermo: Option<ThermoSettings>,
}

impl VibSettings {
    /// The analysis settings of a request (its `vibrations` aside).
    pub(crate) fn from_request(req: &Request) -> Result<Self> {
        let conventions = req.analysis_conventions.as_deref().map(names::analysis_conventions).transpose()?;
        let thermo = if req.thermo == Some(false) {
            if req.has_thermo_settings() {
                return Err(Error::Input(
                    "thermochemistry settings (temperature, pressure, rotor_cutoff, imaginary_cutoff, \
                     frequency_scale, symmetry_number, symmetry_threshold) with thermo off"
                        .into(),
                ));
            }
            None
        } else {
            Some(ThermoSettings {
                temperatures: req.temperature.as_ref().map(|t| t.to_vec()),
                pressure: req.pressure,
                rotor_cutoff: req.rotor_cutoff,
                imaginary_cutoff: req.imaginary_cutoff,
                frequency_scale: req.frequency_scale,
                symmetry_number: req.symmetry_number,
                symmetry_threshold: req.symmetry_threshold,
            })
        };
        Ok(Self { conventions, masses: req.masses.clone(), thermo })
    }

    /// The options of `axtb::vibrations::analyze`; `default` are the calculation's
    /// conventions. GFN-FF analyzed in an xtb program's conventions takes xtb's frequency
    /// scale 1.03 unless one is given.
    pub(crate) fn options(&self, default: VibConventions, gfnff: bool) -> VibOptions {
        let conventions = self.conventions.unwrap_or(default);
        let thermo = self.thermo.as_ref().map(|t| {
            let d = ThermoOptions::default();
            ThermoOptions {
                temperatures: t.temperatures.clone().unwrap_or(d.temperatures),
                pressure: t.pressure.unwrap_or(d.pressure),
                rotor_cutoff: t.rotor_cutoff.unwrap_or(d.rotor_cutoff),
                imaginary_cutoff: t.imaginary_cutoff.unwrap_or(d.imaginary_cutoff),
                frequency_scale: t.frequency_scale.unwrap_or(if gfnff && conventions.is_xtb() {
                    1.03
                } else {
                    d.frequency_scale
                }),
                symmetry_number: t.symmetry_number,
                symmetry_threshold: t.symmetry_threshold.unwrap_or(d.symmetry_threshold),
                ..d
            }
        });
        VibOptions { conventions, masses: self.masses.clone(), thermo }
    }
}

/// A Hessian and its vibrational analysis.
#[derive(Clone, Debug, PartialEq)]
pub struct HessianTask {
    /// `None`: the method's default (analytic; g-xTB v1.1 numerical; the composite each
    /// part's).
    pub method: Option<axtb::HessianMethod>,
    pub step: Option<f64>,
    pub accuracy: Option<f64>,
    /// `None`: no vibrational analysis.
    pub vibrations: Option<VibSettings>,
}

/// A Molden file of the converged orbitals.
#[derive(Clone, Debug, PartialEq)]
pub struct MoldenTask {
    pub functions: axtb::MoldenFunctions,
    /// Put the text into the response.
    pub text: bool,
    /// Write the file there.
    pub file: Option<String>,
}

/// What a call computes besides the energy (and the SCF properties): the gradient, a Hessian
/// with its analysis, a Molden file, the bond orders.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Task {
    pub gradient: bool,
    pub hessian: Option<HessianTask>,
    pub molden: Option<MoldenTask>,
    pub bond_orders: bool,
}

impl Task {
    /// The task of a request for `method`; refuses settings that do not apply (a Hessian's
    /// settings without a Hessian, the analysis' without an analysis, orbitals of GFN-FF, a
    /// step of an analytic Hessian, ...).
    pub fn from_request(req: &Request, method: Method) -> Result<Self> {
        if req.analyze_hessian.is_some() || req.dipole_derivatives.is_some() || req.energy.is_some() {
            return Err(Error::Input(
                "analyze_hessian, dipole_derivatives and energy are for the analysis of a given Hessian, \
                 not for a calculation"
                    .into(),
            ));
        }
        let mut bad = Vec::new();
        if !req.hessian {
            if req.hessian_method.is_some() || req.hessian_step.is_some() || req.hessian_accuracy.is_some() {
                bad.push("hessian_method, hessian_step and hessian_accuracy");
            }
            if req.vibrations.is_some() || req.has_analysis_settings() {
                bad.push("the vibrational-analysis settings");
            }
            if !bad.is_empty() {
                return Err(Error::Input(format!("{} need a Hessian (hessian: true)", bad.join(" and "))));
            }
        }
        let hessian = if req.hessian {
            let hm = req.hessian_method.as_deref().map(names::hessian_method).transpose()?;
            let vibrations = if req.vibrations == Some(false) {
                if req.has_analysis_settings() {
                    return Err(Error::Input("vibrational-analysis settings with vibrations off".into()));
                }
                None
            } else {
                Some(VibSettings::from_request(req)?)
            };
            let resolved = hm.unwrap_or(match method {
                Method::GxtbV1_1 => axtb::HessianMethod::Numerical,
                _ => axtb::HessianMethod::Analytic,
            });
            if req.hessian_step.is_some() && resolved == axtb::HessianMethod::Analytic {
                return Err(Error::Input(
                    "hessian_step is a setting of the numerical and response Hessians, not of the analytic one".into(),
                ));
            }
            if method == Method::Gfnff {
                if hm == Some(axtb::HessianMethod::Response) {
                    return Err(Error::Input("gfnff Hessians are analytic or numerical (no SCF response)".into()));
                }
                if req.hessian_accuracy.is_some() {
                    return Err(Error::Input("gfnff Hessians take no hessian_accuracy (no SCF)".into()));
                }
            }
            Some(HessianTask { method: hm, step: req.hessian_step, accuracy: req.hessian_accuracy, vibrations })
        } else {
            None
        };
        if req.molden_cartesian && !req.wants_molden() {
            return Err(Error::Input("molden_cartesian is a setting of a Molden file (molden or molden_file)".into()));
        }
        let molden = req.wants_molden().then(|| MoldenTask {
            functions: if req.molden_cartesian {
                axtb::MoldenFunctions::Cartesian
            } else {
                axtb::MoldenFunctions::Spherical
            },
            text: req.molden,
            file: req.molden_file.clone(),
        });
        if method == Method::Gfnff && (molden.is_some() || req.bond_orders) {
            return Err(Error::Input("gfnff has no orbitals (no Molden file, no bond orders; see topology)".into()));
        }
        Ok(Self { gradient: req.gradient, hessian, molden, bond_orders: req.bond_orders })
    }
}
