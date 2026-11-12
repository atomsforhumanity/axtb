//! Names of methods, conventions and settings as every front end accepts them.
//!
//! One alias rule for all of them: case and everything but ASCII letters and digits are
//! ignored, so `"GFN2-xTB"`, `"gfn2_xtb"` and `"gfn2xtb"` are one name, as are `"xtb-6.7.1"`
//! and `"XTB 6.7.1"`. The responses report the canonical spelling.

use crate::error::{Error, Result};
use axtb::vibrations::Conventions as VibConventions;

/// The key of a name under the alias rule: its ASCII letters and digits, lower case.
pub fn key(name: &str) -> String {
    name.chars().filter(char::is_ascii_alphanumeric).map(|c| c.to_ascii_lowercase()).collect()
}

/// The methods of the front ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Method {
    /// GFN2-xTB (`gfn2`).
    Gfn2,
    /// GFN-FF (`gfnff`).
    Gfnff,
    /// g-xTB v1.1 (`gxtb-v1.1`), the public preprint version.
    GxtbV1_1,
    /// g-xTB 2.0.1 (`gxtb-v2.0.1`).
    GxtbV2_0_1,
}

impl Method {
    /// The method a name selects: `gfn2` (also `GFN2-xTB`), `gfnff`, `gxtb-v1.1` (also
    /// `gxtb-1.1`, `g-xTB 1.1`) or `gxtb-v2.0.1` (also `gxtb-2.0.1`, `gxtb-v2.0`, `g-xTB 2.0`).
    pub fn parse(name: &str) -> Result<Self> {
        match key(name).as_str() {
            "gfn2" | "gfn2xtb" => Ok(Method::Gfn2),
            "gfnff" => Ok(Method::Gfnff),
            "gxtbv11" | "gxtb11" => Ok(Method::GxtbV1_1),
            "gxtbv201" | "gxtb201" | "gxtbv20" | "gxtb20" => Ok(Method::GxtbV2_0_1),
            _ => Err(Error::Input(format!(
                "unknown method {name:?}; use \"gfn2\", \"gfnff\", \"gxtb-v1.1\" or \"gxtb-v2.0.1\""
            ))),
        }
    }

    /// The canonical name: `gfn2`, `gfnff`, `gxtb-v1.1` or `gxtb-v2.0.1`.
    pub fn name(self) -> &'static str {
        match self {
            Method::Gfn2 => "gfn2",
            Method::Gfnff => "gfnff",
            Method::GxtbV1_1 => "gxtb-v1.1",
            Method::GxtbV2_0_1 => "gxtb-v2.0.1",
        }
    }

    /// Whether the method has an SCF (every method but GFN-FF).
    pub fn is_scf(self) -> bool {
        self != Method::Gfnff
    }

    /// Whether the method is a g-xTB version.
    pub fn is_gxtb(self) -> bool {
        matches!(self, Method::GxtbV1_1 | Method::GxtbV2_0_1)
    }

    /// The g-xTB version of a g-xTB method.
    #[cfg(feature = "gxtb")]
    pub(crate) fn gxtb_version(self) -> Option<axtb_gxtb::Version> {
        match self {
            Method::GxtbV1_1 => Some(axtb_gxtb::Version::V1_1),
            Method::GxtbV2_0_1 => Some(axtb_gxtb::Version::V2_0_1),
            _ => None,
        }
    }
}

impl std::fmt::Display for Method {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// The canonical name of a method name (see [`Method::parse`]); the codec and Python use it.
pub fn canonical_method(name: &str) -> Result<&'static str> {
    Method::parse(name).map(Method::name)
}

/// GFN2-xTB conventions: `axtb` (also `default`), `xtb` (also `xtb-6.8.0`), `xtb-6.7.1` or
/// `tblite`.
pub fn gfn2_conventions(name: &str) -> Result<axtb::Conventions> {
    match key(name).as_str() {
        "axtb" | "default" => Ok(axtb::Conventions::Axtb),
        "xtb" | "xtb680" => Ok(axtb::Conventions::Xtb),
        "xtb671" => Ok(axtb::Conventions::Xtb671),
        "tblite" => Ok(axtb::Conventions::Tblite),
        _ => Err(Error::Input(format!(
            "unknown gfn2 conventions {name:?}; use \"axtb\", \"xtb\" (xtb 6.8.0), \"xtb-6.7.1\" or \"tblite\""
        ))),
    }
}

/// g-xTB conventions: `axtb` (also `default`) or `gxtb` (the program's).
#[cfg(feature = "gxtb")]
pub(crate) fn gxtb_conventions(name: &str) -> Result<axtb_gxtb::Conventions> {
    match key(name).as_str() {
        "axtb" | "default" => Ok(axtb_gxtb::Conventions::Axtb),
        "gxtb" | "gxtbprogram" => Ok(axtb_gxtb::Conventions::Gxtb),
        _ => Err(Error::Input(format!(
            "unknown g-xTB conventions {name:?}; use \"axtb\" or \"gxtb\" (the g-xTB program's)"
        ))),
    }
}

/// The canonical name of g-xTB conventions.
#[cfg(feature = "gxtb")]
pub(crate) fn gxtb_conventions_label(c: axtb_gxtb::Conventions) -> &'static str {
    match c {
        axtb_gxtb::Conventions::Axtb => "axtb",
        axtb_gxtb::Conventions::Gxtb => "gxtb",
    }
}

/// GFN-FF conventions: `axtb` (also `default`) and `xtb` (also `xtb-6.8.0`) run xtb 6.8.0's
/// force field, `xtb-6.7.1` xtb 6.7.1's. Returns the force field, the analysis conventions
/// of its Hessians (`axtb`, or that xtb version's) and the canonical name.
pub(crate) fn gfnff_conventions(name: &str) -> Result<(axtb_gfnff::Conventions, VibConventions, &'static str)> {
    use axtb_gfnff::Conventions as Ff;
    match key(name).as_str() {
        "axtb" | "default" => Ok((Ff::Axtb, VibConventions::Axtb, "axtb")),
        "xtb" | "xtb680" => Ok((Ff::Xtb680, VibConventions::Xtb, "xtb")),
        "xtb671" => Ok((Ff::Xtb671, VibConventions::Xtb671, "xtb-6.7.1")),
        _ => Err(Error::Input(format!(
            "unknown gfnff conventions {name:?}; use \"axtb\", \"xtb\" (xtb 6.8.0) or \"xtb-6.7.1\""
        ))),
    }
}

/// Conventions of the vibrational analysis: `axtb` (also `default`), `xtb` (also
/// `xtb-6.8.0`), `xtb-6.7.1` (also `gxtb`: the g-xTB 2.0.1 program is a 6.7.1-based xtb).
pub fn analysis_conventions(name: &str) -> Result<VibConventions> {
    match key(name).as_str() {
        "axtb" | "default" => Ok(VibConventions::Axtb),
        "xtb" | "xtb680" => Ok(VibConventions::Xtb),
        "xtb671" | "gxtb" => Ok(VibConventions::Xtb671),
        _ => Err(Error::Input(format!(
            "unknown vibrational-analysis conventions {name:?}; use \"axtb\", \"xtb\" (xtb 6.8.0) or \"xtb-6.7.1\""
        ))),
    }
}

/// Hessian method: `analytic`, `response` or `numerical` (also `numeric`, `fd`).
pub fn hessian_method(name: &str) -> Result<axtb::HessianMethod> {
    match key(name).as_str() {
        "analytic" | "analytical" => Ok(axtb::HessianMethod::Analytic),
        "response" => Ok(axtb::HessianMethod::Response),
        "numerical" | "numeric" | "fd" => Ok(axtb::HessianMethod::Numerical),
        _ => Err(Error::Input(format!(
            "unknown Hessian method {name:?}; use \"analytic\", \"response\" or \"numerical\""
        ))),
    }
}

/// The canonical name of a Hessian method.
pub fn hessian_method_label(m: axtb::HessianMethod) -> &'static str {
    match m {
        axtb::HessianMethod::Analytic => "analytic",
        axtb::HessianMethod::Response => "response",
        axtb::HessianMethod::Numerical => "numerical",
    }
}

/// GFN2-xTB SCF mixer: `auto`, `broyden` or `diis` (also `pulay`).
pub fn mixer(name: &str) -> Result<axtb::MixerKind> {
    match key(name).as_str() {
        "auto" => Ok(axtb::MixerKind::Auto),
        "broyden" => Ok(axtb::MixerKind::Broyden),
        "diis" | "pulay" => Ok(axtb::MixerKind::Diis),
        _ => Err(Error::Input(format!("unknown SCF mixer {name:?}; use \"auto\", \"broyden\" or \"diis\""))),
    }
}

/// Reference state of the solvation free energy: `gsolv`, `bar1M` or `reference`.
pub fn solvation_state(name: &str) -> Result<axtb::ReferenceState> {
    match key(name).as_str() {
        "gsolv" => Ok(axtb::ReferenceState::Gsolv),
        "bar1m" => Ok(axtb::ReferenceState::Bar1M),
        "reference" => Ok(axtb::ReferenceState::Reference),
        _ => Err(Error::Input(format!(
            "unknown solvation reference state {name:?}; use \"gsolv\", \"bar1M\" or \"reference\""
        ))),
    }
}

/// Functions of a Molden file: `spherical` (also `pure`, `5d`) or `cartesian` (also `6d`).
pub fn molden_functions(name: &str) -> Result<axtb::MoldenFunctions> {
    match key(name).as_str() {
        "spherical" | "pure" | "5d" => Ok(axtb::MoldenFunctions::Spherical),
        "cartesian" | "6d" => Ok(axtb::MoldenFunctions::Cartesian),
        _ => Err(Error::Input(format!("unknown Molden functions {name:?}; use \"spherical\" or \"cartesian\""))),
    }
}

/// Solvation correction of the g-xTB composite: `gfn2` (also `GFN2-xTB`).
#[cfg(feature = "gxtb")]
pub(crate) fn correction(name: &str) -> Result<axtb_gxtb::Correction> {
    match key(name).as_str() {
        "gfn2" | "gfn2xtb" => Ok(axtb_gxtb::Correction::Gfn2),
        _ => Err(Error::Input(format!(
            "unknown solvation correction {name:?}; the only one is \"gfn2\" (GFN2-xTB with ALPB/GBSA)"
        ))),
    }
}

/// Where a calculator's SCFs start ([`guess`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Guess {
    /// The previous geometry's converged state (`previous`, also `restart`).
    Previous,
    /// The method's own start at every geometry (`fresh`).
    Fresh,
    /// Every geometry from EEQ charges (`eeq`; GFN2-xTB's own start).
    Eeq,
    /// Every geometry from neutral atoms (`sad`; GFN2-xTB).
    Sad,
    /// Every geometry from the program's EEQ-BC start (`eeq-bc`; g-xTB's own start).
    EeqBc,
}

/// Where a calculator's SCFs start: `previous` (also `restart`), the previous geometry's
/// converged state; `fresh`, the method's own start; or a named start, `eeq` or `sad`
/// (GFN2-xTB) or `eeq-bc` (g-xTB), which the calculator checks against its method.
pub fn guess(name: &str) -> Result<Guess> {
    match key(name).as_str() {
        "previous" | "restart" => Ok(Guess::Previous),
        "fresh" => Ok(Guess::Fresh),
        "eeq" => Ok(Guess::Eeq),
        "sad" => Ok(Guess::Sad),
        "eeqbc" => Ok(Guess::EeqBc),
        _ => Err(Error::Input(format!(
            "unknown guess {name:?}; use \"previous\", \"fresh\" (the method's own start), \"eeq\" or \"sad\" \
             (GFN2-xTB) or \"eeq-bc\" (g-xTB)"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_alias_rule() {
        for (name, method) in [
            ("gfn2", "gfn2"),
            ("GFN2-xTB", "gfn2"),
            ("GFN2 xTB", "gfn2"),
            ("gfn-ff", "gfnff"),
            ("g-xTB 1.1", "gxtb-v1.1"),
            ("gxtb_v1.1", "gxtb-v1.1"),
            ("gxtb-2.0.1", "gxtb-v2.0.1"),
            ("g-xTB 2.0", "gxtb-v2.0.1"),
        ] {
            assert_eq!(canonical_method(name).unwrap(), method, "{name}");
        }
        assert!(matches!(canonical_method("gfn1"), Err(Error::Input(_))));
        assert_eq!(gfn2_conventions("XTB 6.7.1").unwrap(), axtb::Conventions::Xtb671);
        assert_eq!(gfn2_conventions("default").unwrap(), axtb::Conventions::Axtb);
        assert_eq!(gfn2_conventions("xtb-6.8.0").unwrap(), axtb::Conventions::Xtb);
        assert_eq!(gfnff_conventions("xtb_6.7.1").unwrap().2, "xtb-6.7.1");
        assert_eq!(analysis_conventions("gxtb").unwrap(), VibConventions::Xtb671);
        assert_eq!(solvation_state("BAR 1M").unwrap(), axtb::ReferenceState::Bar1M);
        assert_eq!(guess("eeq-bc").unwrap(), Guess::EeqBc);
        assert_eq!(guess("Restart").unwrap(), Guess::Previous);
        assert_eq!(guess("SAD").unwrap(), Guess::Sad);
        assert!(guess("huckel").is_err());
        assert!(hessian_method("finite").is_err());
    }
}
