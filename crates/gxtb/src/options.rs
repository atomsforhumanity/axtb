//! What a g-xTB calculation computes: the version ([`Version`]), the conventions
//! ([`Conventions`]) and the settings of one SCF ([`ScfOptions`]), for both versions.

use axtb::{Error, Result};

/// The g-xTB versions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Version {
    /// g-xTB 2.0.1 (the grimme-lab/g-xtb 2.0.1 release, `xtb --gxtb`), the default.
    #[default]
    V2_0_1,
    /// g-xTB v1.1 (public preprint version, the 1.1.0 program).
    V1_1,
}

impl Version {
    /// The version a method name selects: `gxtb-v2.0.1` (also `gxtb-2.0.1`, `gxtbv201`,
    /// `g-xTB 2.0`) or `gxtb-v1.1` (also `gxtb_v1.1`, `gxtbv11`, `g-xTB 1.1`), case-insensitive.
    pub fn from_method(name: &str) -> Option<Self> {
        let key: String = name.to_ascii_lowercase().chars().filter(|c| c.is_ascii_alphanumeric()).collect();
        match key.as_str() {
            "gxtbv201" | "gxtb201" | "gxtbv20" | "gxtb20" => Some(Self::V2_0_1),
            "gxtbv11" | "gxtb11" => Some(Self::V1_1),
            _ => None,
        }
    }

    /// The method name: `"gxtb-v2.0.1"` or `"gxtb-v1.1"`.
    pub fn method(&self) -> &'static str {
        match self {
            Self::V2_0_1 => "gxtb-v2.0.1",
            Self::V1_1 => "gxtb-v1.1",
        }
    }
}

/// The settings of one SCF of either version ([`crate::scf::Setup::scf`], [`crate::v2::scf::Setup::scf`]; a
/// [`crate::Calculator`] assembles them from its [`crate::Options`]). The defaults (`None`, 0)
/// are the programs':
///
/// - **v1.1** (the 1.1.0 program's `scf_`, `diisfock_`; [`crate::scf::Setup::scf`]): Fock damping with
///   [`crate::scf::DAMPING`] in iterations 1-2 (open shell: 1-3), a transition step in iteration 3 (4), Pulay
///   mixing of at most 7 (8) Fock matrices from iteration 4 (5) on, convergence when
///   |dE| < 1e-8 Eh and the RMS change of the packed spin densities < 2e-5 (open shell: 4e-7 Eh
///   and 5e-5), each times `accuracy`, followed by one more iteration with the unmixed Fock
///   matrix whose energy is the result; at most [`crate::scf::MAX_ITER`] iterations.
/// - **2.0.1** (tblite's SCF as the 2.0.1 program runs it; [`crate::v2::scf`]): no mixing in
///   iteration 1, damping 0.2 in iterations 2-3, then DIIS with at most 7 Fock matrices,
///   convergence when |dE| < 1e-6 Eh and the RMS Fock residual < 2e-5, each times `accuracy`;
///   at most [`crate::v2::scf::MAX_ITER`] iterations.
///
/// With `conventions: Conventions::Gxtb` the SCF reproduces the program's iterations and
/// energies; with the default [`Conventions::Axtb`] v1.1 diagonalizes the exact derivative of
/// the energy instead, and 2.0.1 uses CODATA 2022 units, an integer electron count and smooth
/// cutoffs. Settings of one version only are refused for the other ([`ScfOptions::check`]).
#[derive(Clone, Debug)]
pub struct ScfOptions {
    /// SCF iteration limit; 0 (the default): the program's, [`crate::scf::MAX_ITER`] for v1.1 and
    /// [`crate::v2::scf::MAX_ITER`] for 2.0.1.
    pub max_iter: usize,
    /// Multiplier of the program's convergence thresholds (those not given explicitly).
    pub accuracy: f64,
    /// Explicit threshold of the energy change (Eh).
    pub energy_tol: Option<f64>,
    /// v1.1 only: explicit threshold of the RMS change of the packed spin densities.
    pub density_tol: Option<f64>,
    /// 2.0.1 only: explicit threshold of the RMS Fock residual (the Fock matrix of the new
    /// density minus the mixed one it came from, over both spin channels).
    pub residual_tol: Option<f64>,
    /// Damping factor in (0, 1] of the first iterations; `None`: v1.1 [`DAMPING`], 2.0.1 0.2.
    pub damping: Option<f64>,
    /// v1.1 only: iteration of the transition step (damping before it, Pulay mixing after);
    /// `None`: 3 (open shell: 4).
    pub transition_iteration: Option<usize>,
    /// Largest number of stored Fock matrices of the Pulay (DIIS) mixing, at least 1; `None`:
    /// v1.1 7 (open shell: 8), 2.0.1 7.
    pub diis_size: Option<usize>,
    /// Threads of a [`crate::Calculator`] (1: no thread is spawned).
    pub threads: usize,
    /// Electronic temperature (K) of the Fermi smearing; 0 (the programs' default for
    /// g-xTB): Aufbau occupations.
    pub etemp: f64,
    /// v1.1: the Fock matrix of the SCF, the exact derivative of the energy or the program's;
    /// 2.0.1: units, electron count and cutoffs (the program's or the default ones).
    pub conventions: Conventions,
    /// 2.0.1: every orbital's coefficients in the [`crate::v2::scf::ScfResult`] (its Hessians and
    /// the orbitals of a Molden file need them; without, only the occupied orbitals are
    /// computed). v1.1 computes every orbital anyway.
    pub all_orbitals: bool,
}

impl Default for ScfOptions {
    fn default() -> Self {
        Self {
            max_iter: 0,
            accuracy: 1.0,
            energy_tol: None,
            density_tol: None,
            residual_tol: None,
            damping: None,
            transition_iteration: None,
            diis_size: None,
            threads: 1,
            etemp: 0.0,
            conventions: Conventions::default(),
            all_orbitals: false,
        }
    }
}

impl ScfOptions {
    /// The SCF iteration limit for `version` (`max_iter`, 0: the program's).
    pub fn iteration_limit(&self, version: Version) -> usize {
        match (self.max_iter, version) {
            (0, Version::V1_1) => crate::scf::MAX_ITER,
            (0, Version::V2_0_1) => crate::v2::scf::MAX_ITER,
            (m, _) => m,
        }
    }

    /// Refuse settings that are out of range or belong to the other version.
    pub fn check(&self, version: Version) -> Result<()> {
        let positive = |name: &str, x: f64| {
            if x > 0.0 && x.is_finite() {
                Ok(())
            } else {
                Err(Error::Input(format!("{name} must be positive and finite, got {x}")))
            }
        };
        axtb_base::threads::check(self.threads)?;
        positive("accuracy", self.accuracy)?;
        for (name, tol) in
            [("energy_tol", self.energy_tol), ("density_tol", self.density_tol), ("residual_tol", self.residual_tol)]
        {
            tol.map_or(Ok(()), |x| positive(name, x))?;
        }
        if !(self.etemp >= 0.0 && self.etemp.is_finite()) {
            return Err(Error::Input(format!("electronic temperature must be >= 0 K and finite, got {}", self.etemp)));
        }
        if let Some(d) = self.damping {
            if !(d > 0.0 && d <= 1.0) {
                return Err(Error::Input(format!("damping must be in (0, 1], got {d}")));
            }
        }
        if self.diis_size == Some(0) {
            return Err(Error::Input("diis_size must be at least 1".into()));
        }
        if self.transition_iteration == Some(0) {
            return Err(Error::Input("transition_iteration must be at least 1 (iterations count from 1)".into()));
        }
        let other_version = match version {
            Version::V1_1 if self.residual_tol.is_some() => {
                "residual_tol is a g-xTB 2.0.1 setting (the Fock residual); v1.1 converges on density_tol (the \
                 density change)"
            }
            Version::V2_0_1 if self.density_tol.is_some() => {
                "density_tol is a g-xTB v1.1 setting (the density change); 2.0.1 converges on residual_tol (the \
                 Fock residual)"
            }
            Version::V2_0_1 if self.transition_iteration.is_some() => {
                "transition_iteration is a g-xTB v1.1 setting; the 2.0.1 SCF has no transition step"
            }
            _ => return Ok(()),
        };
        Err(Error::Input(other_version.into()))
    }
}

/// Which Fock matrix the SCF diagonalizes (the energy expression is the same).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Conventions {
    /// The exact derivative of the g-xTB energy (default): the converged SCF is stationary
    /// and the gradient needs no response term. Energies differ from the program's by
    /// -1e-8 to -2.3e-4 Eh (lower), energy differences by up to ~1e-5 Eh (spec §4.18).
    #[default]
    Axtb,
    /// The g-xTB 1.1.0 program's Fock matrix, which leaves out the AES potentials of the
    /// dipole-quadrupole and quadrupole-quadrupole energies and applies the population part
    /// of the onsite exchange at twice its derivative off the diagonal (§4.16 of the spec):
    /// reproduces the program's energies; its gradients need a response term.
    Gxtb,
}

impl Conventions {
    /// `axtb` or `gxtb`, the names the front ends report.
    pub fn label(self) -> &'static str {
        match self {
            Conventions::Axtb => "axtb",
            Conventions::Gxtb => "gxtb",
        }
    }
}

impl std::str::FromStr for Conventions {
    type Err = Error;
    /// `"axtb"` (also `"default"`) or `"gxtb"` (also `"gxtb-program"`); case and anything but
    /// letters and digits are ignored, as for the other names of the axtb front ends.
    fn from_str(s: &str) -> Result<Self> {
        let key: String = s.to_ascii_lowercase().chars().filter(|c| c.is_ascii_alphanumeric()).collect();
        match key.as_str() {
            "axtb" | "default" => Ok(Self::Axtb),
            "gxtb" | "gxtbprogram" => Ok(Self::Gxtb),
            _ => Err(Error::Input(format!("unknown g-xTB conventions {s:?}; use \"axtb\" or \"gxtb\""))),
        }
    }
}
