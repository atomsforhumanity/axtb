//! Numerical conventions: axtb's corrected default, or exact reproduction of the xtb 6.8.0,
//! xtb 6.7.1 and tblite 0.8.0 programs. Everything that differs between them is one table,
//! [`ConventionParams`], resolved once when a calculation is set up; the rest of the code
//! reads the table instead of testing the mode.

use crate::basis::{integral_cutoff, integral_cutoff_xtb, Basis};
pub use crate::eeq::EeqCn;
use crate::param::xtb_data::{
    XTB671_ATOMIC_RAD, XTB671_COVALENT_RAD_D3, XTB671_D4_EN, XTB671_D4_RCOV, XTB671_PAULING_EN, XTB_ATOMIC_RAD,
    XTB_COVALENT_RAD_D3, XTB_D4_EN, XTB_D4_RCOV, XTB_PAULING_EN,
};
use crate::param::{atomic_rad, atomic_rad_codata2022, covalent_rad_d3, covalent_rad_d3_codata2022, pauling_en};
use crate::scf::mixing::MixerKind;
use crate::units::{reference::KB_XTB_TBLITE, KB_HARTREE};
use crate::{Error, Result};
use axtb_d4::RealspaceCutoff;
use axtb_solvation::Conventions as SolvConventions;

/// Numerical conventions of a calculation.
///
/// `Axtb` (default) is axtb's own corrected and self-consistent choice: CODATA 2022
/// constants (via qcref) for radii and the electronic temperature, double-precision
/// tables, one coordination number for energy and gradient, xtb's longer real-space
/// cutoffs with smooth switching, and the solvation fixes for grimme-lab/xtb#1504 and
/// #1505. Constants that define the published parametrization (e.g. GFN2-xTB's eV-to-Hartree
/// factor) are kept in every mode.
///
/// `Xtb` reproduces the xtb 6.8.0 program, `Xtb671` the xtb 6.7.1 program (and the g-xTB
/// 2.0.1 build of xtb, which is 6.7.1 for GFN2-xTB) and `Tblite` tblite 0.8.0, to the last
/// digits, including their numerical quirks. xtb 6.8.0 adopted tblite's numerics for
/// GFN2-xTB (mctc-lib's bohr constant in its radius tables, double-precision D4 literals,
/// tblite's cutoffs, smooth D4 cutoffs, integral screening, mixed vector and convergence
/// test) and fixed the two solvation bugs (#1504, #1505); what is left of xtb in it is the
/// EEQ start, aufbau occupations up to 0.1 K, xtb's solvation constants and its gradient
/// potentials (see below). xtb 6.7.1 has the older bohr constant in its radius tables,
/// single-precision literals in its D4 coordination number, its own cutoffs and screening,
/// both solvation bugs and its own SCC loop. With solvation, `Tblite` and `Axtb` use the
/// corrected solvation, which tblite's own ALPB/GBSA matches to ~1e-9 Eh. Differences
/// between modes are ~1e-8 Eh for ordinary molecules (more for molecules larger than 25 bohr
/// between 6.7.1 and the others, and for the solvation cases the fixes address).
///
/// `Axtb` and `Tblite` run tblite's SCF loop, `Xtb` xtb 6.8.0's, which is tblite's except
/// for its start and its gradient, and `Xtb671` xtb 6.7.1's
/// ([`ConventionParams::eeq_guess_cn`] and the `scc_*` settings). The loops differ in six
/// places that only matter while the SCF is not converged, so at xtb's default accuracy 1
/// (thresholds 1e-6 Eh and 2e-5 e, 6.7.1: 1e-4 e): with tblite's loop, the 6.7.1
/// conventions reproduced the program's gradients only to ~4e-6 Eh/bohr there, with its
/// own to ≤ 3.3e-10 (`tests/solvation_composite.py`, `tests/reference_matrix`):
/// 1. the start: xtb (both versions) starts from `goedecker_chrgeq` EEQ charges on
///    `ncoord_erf`'s CN (xtb's D4 radii, 40 bohr, no cap; 6.7.1's radii single-precision
///    with the older bohr constant), not multicharge's capped CN; tblite from zero charges;
/// 2. the residual test ([`ConventionParams::scc_residual`]): 6.7.1 tests
///    sqrt(Σ dq² / N_atoms) < 1e-4·acc over the mixed vector with multipoles
///    (`scf_module.F90` `qconv`), tblite and 6.8.0 the RMS over the vector < 2e-5·acc;
/// 3. 6.7.1 runs one more iteration after the first converged one (`scc_core.f90`
///    `lastdiag`) and once converged counts the SCF as converged
///    ([`ConventionParams::scc_extra_iteration`]); 6.8.0 and tblite stop at the converged one;
/// 4. 6.7.1's mixed vector holds the quadrupole off-diagonals as they are; tblite and 6.8.0
///    scale them by sqrt 2, which changes Broyden's metric and so its iterates
///    ([`ConventionParams::scc_quadrupole_scale`]);
/// 5. 6.7.1's Fock matrix leaves out the dipole (quadrupole) term of an AO pair whose moment
///    integrals around the coordinate origin are small, Σ_k d_k² (Σ_diag M² + 2 Σ_off M²)
///    ≤ 1e-10·acc (`aespot.F90` `setdqlist`, `buildIsoAnisotropicH1`); the overlap part of
///    the potentials and the Mulliken moments use every integral
///    ([`ConventionParams::scc_multipole_neglect`]; 6.8.0 sets the threshold to 0);
/// 6. the moments and potentials of the gradient ([`ConventionParams::scc_gradient`]):
///    tblite takes the converged (output) moments and the last iteration's input
///    potentials; 6.8.0 the output moments, with the multipole (AES) potentials of the
///    integral derivatives recomputed from them (`setdvsdq`) and the isotropic shell shifts
///    the last input's (`ves`); 6.7.1 the moments as mixed once more after the last
///    iteration (`wfn%q`, `qsh`, `dipm`, `qp`) for the ES2, AES, D4 and solvation terms and
///    the AES potentials, and the isotropic shell shifts the last input's. The energy is
///    always the last iteration's, from its output moments.
///
/// [`Conventions::params`] lists every setting that differs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Conventions {
    #[default]
    Axtb,
    /// The xtb 6.8.0 program.
    Xtb,
    /// The xtb 6.7.1 program (also the g-xTB 2.0.1 build of xtb for GFN2-xTB).
    Xtb671,
    Tblite,
}

impl Conventions {
    /// The name of the conventions (`axtb`, `xtb`, `xtb-6.7.1`, `tblite`), as parsed.
    pub fn label(self) -> &'static str {
        match self {
            Conventions::Axtb => "axtb",
            Conventions::Xtb => "xtb",
            Conventions::Xtb671 => "xtb-6.7.1",
            Conventions::Tblite => "tblite",
        }
    }
}

impl std::str::FromStr for Conventions {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "axtb" | "default" => Ok(Conventions::Axtb),
            "xtb" | "xtb-6.8.0" => Ok(Conventions::Xtb),
            "xtb-6.7.1" => Ok(Conventions::Xtb671),
            "tblite" => Ok(Conventions::Tblite),
            other => Err(Error::Input(format!("unknown conventions {other:?} (axtb, xtb, xtb-6.7.1 or tblite)"))),
        }
    }
}

impl std::fmt::Display for Conventions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// A per-element value indexed by the atomic number (1-based).
pub type ElementTable = fn(usize) -> f64;

/// The SCC convergence test on the residual dq of the mixed vector (point 2 of
/// [`Conventions`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SccResidual {
    /// sqrt(Σ dq² / n) over the n mixed variables below 2e-5·acc (tblite, xtb 6.8.0).
    Rms,
    /// sqrt(Σ dq² / N_atoms) below 1e-4·acc when the vector holds multipoles, 2e-5·acc
    /// otherwise (xtb 6.7.1 `scf_module.F90` `qconv`, `scc_core.f90` `rmsq`).
    PerAtom,
}

/// The moments and potentials the gradient takes after the SCF (point 6 of [`Conventions`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SccGradient {
    /// The converged (output) moments; the potentials of the integral derivatives are the
    /// last iteration's input potentials (tblite).
    Output,
    /// The output moments; the multipole (AES) potentials of the integral derivatives from
    /// them, the isotropic shell shifts of the overlap term the last iteration's input
    /// (xtb 6.8.0: `setdvsdq` on `wfn%q`, `dipm`, `qp`, and `ves`).
    OutputAes,
    /// The moments as mixed once more after the last iteration, for the ES2, AES, D4 and
    /// solvation terms and the AES potentials; the isotropic shell shifts of the overlap term
    /// the last iteration's input (xtb 6.7.1).
    Mixed,
}

/// The settings that differ between [`Conventions`], resolved once at setup.
#[derive(Clone, Copy, Debug)]
pub struct ConventionParams {
    /// Boltzmann constant (Eh/K) of the electronic temperature.
    pub kb: f64,
    /// Electronic temperatures (K) up to which the occupations stay aufbau (no Fermi
    /// smearing). xtb smears only above 0.1 K (scc_core.f90: `if (et.gt.0.1_wp)`, unchanged in
    /// 6.8.0); with degenerate frontier orbitals (singlet O2 at 0.1 K) the two differ by
    /// 2.5e-3 Eh. 0 means: smear at every temperature above 0 K.
    pub aufbau_max_etemp: f64,
    /// Whether [`InitialGuess::Eeq`](crate::InitialGuess::Eeq) is honoured; tblite always
    /// starts from zero charges.
    pub eeq_guess: bool,
    /// Coordination number of the EEQ guess: multicharge's (capped at 8, 25 bohr) or the xtb
    /// program's (`goedecker_chrgeq` on `ncoord_erf`: its D4 radii, 40 bohr, no cap; point 1
    /// of [`Conventions`]).
    pub eeq_guess_cn: EeqCn,
    /// Pauling electronegativities of the core Hamiltonian's scaling.
    pub pauling_en: ElementTable,
    /// Atomic radii (bohr) of the core Hamiltonian's shell polynomials.
    pub atomic_rad: ElementTable,
    /// Covalent radii (bohr) of the GFN2-xTB coordination number.
    pub cn_rcov: ElementTable,
    /// Real-space cutoff (bohr) of the coordination number.
    pub cn_cutoff: f64,
    /// Real-space cutoff (bohr) of the repulsion.
    pub repulsion_cutoff: f64,
    /// Primitive-pair screening threshold (on the Gaussian product exponent) for an SCF
    /// accuracy. xtb 6.7.1: max(20, 25 - 10 log10 acc); xtb 6.8.0: tblite's
    /// 25 - 10 log10 acc within [5, 25] (acc clamped to [1e-4, 1e3]); tblite builds its basis
    /// at accuracy 1.
    pub primitive_cutoff: fn(f64) -> f64,
    /// Atom-pair distance cutoff (bohr) of the integrals for a basis and SCF accuracy;
    /// infinite where only primitive pairs are screened (xtb).
    pub pair_cutoff: fn(&Basis, f64) -> f64,
    /// Covalent radii (bohr) of the D4 coordination number.
    pub d4_rcov: ElementTable,
    /// Electronegativities of the D4 coordination number.
    pub d4_en: ElementTable,
    /// D4 real-space cutoffs and widths of their smooth switching.
    pub d4_cutoff: RealspaceCutoff,
    /// Implicit solvation variant (xtb 6.8.0's, xtb 6.7.1's with its two bugs, or the
    /// corrected model).
    pub solvation: SolvConventions,
    /// Mixer of the SCC variables when [`Mixing::mixer`](crate::Mixing::mixer) does
    /// not choose one. xtb and tblite: their modified Broyden mixing alone, so the
    /// iterations, the SCF solution where there are several, and the failures are theirs;
    /// axtb: [`MixerKind::Auto`], Broyden with a DIIS fallback for SCFs Broyden does not
    /// converge (`tests/scf_hard_set.py`).
    pub mixer: MixerKind,
    /// The SCC convergence test (point 2 of [`Conventions`]).
    pub scc_residual: SccResidual,
    /// One more iteration after the first converged one, and the SCF counts as converged
    /// once it was (xtb 6.7.1 `lastdiag`; point 3).
    pub scc_extra_iteration: bool,
    /// Factor of the quadrupole off-diagonals in the mixed vector (tblite and xtb 6.8.0:
    /// sqrt 2, so that the residual norm counts them twice; xtb 6.7.1: 1; point 4).
    pub scc_quadrupole_scale: f64,
    /// Leave small multipole integrals out of the Fock matrix (xtb 6.7.1,
    /// `hamiltonian::MultipoleNeglect`; point 5).
    pub scc_multipole_neglect: bool,
    /// Moments and potentials of the gradient (point 6).
    pub scc_gradient: SccGradient,
    /// Diagonalize the SCF's Fock matrices for the orbitals that enter the density only (those
    /// with an occupation above [`OCCUPATION_CUTOFF`], as xtb 6.8.0 builds its density,
    /// grimme-lab/xtb#1466; [`GenEigSolver::solve_lowest`](crate::linalg::GenEigSolver::solve_lowest))
    /// unless all orbitals are needed (Molden files, Hessians, open-shell bond orders), and
    /// build the density matrices as triangular products (`linalg::weighted_outer_sym`, the
    /// syrk of grimme-lab/xtb#1466: half the flops, rounded differently). The program
    /// conventions keep the full eigensolver and the full products, so that their results
    /// stay bit for bit.
    pub occupied_orbitals_only: bool,
    /// D4 three-body energy and gradient by pair sums (xtb 6.8.0, grimme-lab/xtb#1464,
    /// [`D4Dispersion::with_pair_sums`](axtb_d4::D4Dispersion::with_pair_sums)); the program
    /// conventions keep dftd4's per-triple order, bit for bit.
    pub d4_atm_pair_sums: bool,
}

/// Occupations below this leave an orbital out of the density in
/// [`ConventionParams::occupied_orbitals_only`].
pub const OCCUPATION_CUTOFF: f64 = 1e-16;

fn no_pair_cutoff(_: &Basis, _: f64) -> f64 {
    f64::INFINITY
}

impl Conventions {
    /// The settings of these conventions.
    pub fn params(self) -> ConventionParams {
        match self {
            // CODATA 2022 radii and kB, double-precision tables, xtb's screening and longer
            // cutoffs, D4 with a smooth 1 bohr switch (energy and gradient consistent and
            // continuous), corrected solvation, Broyden mixing with a DIIS fallback
            Conventions::Axtb => ConventionParams {
                kb: KB_HARTREE,
                aufbau_max_etemp: 0.0,
                eeq_guess: true,
                eeq_guess_cn: EeqCn::MULTICHARGE,
                pauling_en,
                atomic_rad: atomic_rad_codata2022,
                cn_rcov: covalent_rad_d3_codata2022,
                cn_cutoff: 40.0,
                repulsion_cutoff: 40.0,
                primitive_cutoff: integral_cutoff_xtb,
                pair_cutoff: no_pair_cutoff,
                d4_rcov: covalent_rad_d3_codata2022,
                d4_en: pauling_en,
                d4_cutoff: RealspaceCutoff { cn: 40.0, disp2: 72.0, disp3: 40.0, width2: 1.0, width3: 1.0 },
                solvation: SolvConventions::Corrected,
                mixer: MixerKind::Auto,
                scc_residual: SccResidual::Rms,
                scc_extra_iteration: false,
                scc_quadrupole_scale: std::f64::consts::SQRT_2,
                scc_multipole_neglect: false,
                scc_gradient: SccGradient::Output,
                occupied_orbitals_only: true,
                d4_atm_pair_sums: true,
            },
            // the xtb 6.8.0 program: mctc-lib's bohr constant and double-precision literals in
            // its tables (bit for bit with its operation order), tblite's cutoffs (CN and
            // repulsion 25 bohr, D4 CN 30, D4 50/25 bohr with 0.05 bohr switches), tblite's
            // primitive screening, aufbau occupations up to 0.1 K, its solvation (both bugs
            // fixed), tblite's SCC loop with xtb's EEQ start and gradient potentials
            Conventions::Xtb => ConventionParams {
                kb: KB_XTB_TBLITE,
                aufbau_max_etemp: 0.1,
                eeq_guess: true,
                eeq_guess_cn: EeqCn { rcov: |z| XTB_D4_RCOV[z - 1], cutoff: 40.0, cap: None },
                pauling_en: |z| XTB_PAULING_EN[z - 1],
                atomic_rad: |z| XTB_ATOMIC_RAD[z - 1],
                cn_rcov: |z| XTB_COVALENT_RAD_D3[z - 1],
                cn_cutoff: 25.0,
                repulsion_cutoff: 25.0,
                primitive_cutoff: integral_cutoff,
                pair_cutoff: no_pair_cutoff,
                d4_rcov: |z| XTB_D4_RCOV[z - 1],
                d4_en: |z| XTB_D4_EN[z - 1],
                d4_cutoff: RealspaceCutoff::TBLITE_GFN2,
                solvation: SolvConventions::Xtb,
                mixer: MixerKind::Broyden,
                scc_residual: SccResidual::Rms,
                scc_extra_iteration: false,
                scc_quadrupole_scale: std::f64::consts::SQRT_2,
                scc_multipole_neglect: false,
                scc_gradient: SccGradient::OutputAes,
                occupied_orbitals_only: false,
                d4_atm_pair_sums: false,
            },
            // the xtb 6.7.1 program: its tables (older bohr constant, single-precision D4
            // CN literals), cutoffs and screening, aufbau occupations up to 0.1 K, its
            // solvation, its SCC loop
            Conventions::Xtb671 => ConventionParams {
                kb: KB_XTB_TBLITE,
                aufbau_max_etemp: 0.1,
                eeq_guess: true,
                eeq_guess_cn: EeqCn { rcov: |z| XTB671_D4_RCOV[z - 1], cutoff: 40.0, cap: None },
                pauling_en: |z| XTB671_PAULING_EN[z - 1],
                atomic_rad: |z| XTB671_ATOMIC_RAD[z - 1],
                cn_rcov: |z| XTB671_COVALENT_RAD_D3[z - 1],
                cn_cutoff: 40.0,
                repulsion_cutoff: 40.0,
                primitive_cutoff: integral_cutoff_xtb,
                pair_cutoff: no_pair_cutoff,
                d4_rcov: |z| XTB671_D4_RCOV[z - 1],
                d4_en: |z| XTB671_D4_EN[z - 1],
                d4_cutoff: RealspaceCutoff::XTB_GFN2,
                solvation: SolvConventions::Xtb671,
                mixer: MixerKind::Broyden,
                scc_residual: SccResidual::PerAtom,
                scc_extra_iteration: true,
                scc_quadrupole_scale: 1.0,
                scc_multipole_neglect: true,
                scc_gradient: SccGradient::Mixed,
                occupied_orbitals_only: false,
                d4_atm_pair_sums: false,
            },
            // tblite 0.8.0: mctc-lib radii, 25 bohr cutoffs, atom-pair integral screening,
            // dftd4's CN tables and tblite's D4 cutoffs, zero-charge start
            Conventions::Tblite => ConventionParams {
                kb: KB_XTB_TBLITE,
                aufbau_max_etemp: 0.0,
                eeq_guess: false,
                eeq_guess_cn: EeqCn::MULTICHARGE,
                pauling_en,
                atomic_rad,
                cn_rcov: covalent_rad_d3,
                cn_cutoff: 25.0,
                repulsion_cutoff: 25.0,
                primitive_cutoff: |_| integral_cutoff(1.0),
                pair_cutoff: |bas, acc| bas.cutoff(acc),
                d4_rcov: axtb_d4::covalent_radius,
                d4_en: |z| axtb_d4::data::PAULING_EN[z - 1],
                d4_cutoff: RealspaceCutoff::TBLITE_GFN2,
                solvation: SolvConventions::Corrected,
                mixer: MixerKind::Broyden,
                scc_residual: SccResidual::Rms,
                scc_extra_iteration: false,
                scc_quadrupole_scale: std::f64::consts::SQRT_2,
                scc_multipole_neglect: false,
                scc_gradient: SccGradient::Output,
                occupied_orbitals_only: false,
                d4_atm_pair_sums: false,
            },
        }
    }
}
