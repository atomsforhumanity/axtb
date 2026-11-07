// SPDX-License-Identifier: LGPL-3.0-or-later
//! Implicit solvation (ALPB and GBSA) for GFN2-xTB, reproducing xtb 6.8.0 and 6.7.1.
//!
//! This is a port of xtb's own solvation code (`src/solv/{model,gbsa,born,sasa,
//! kernel,state,lebedev}.f90`), which is what `xtb mol.xyz --alpb water` (and
//! `--gbsa`) uses for GFN2-xTB unless `--tblite` is given.  The solvation free
//! energy is
//!
//! ```text
//! G_solv = 1/2 q^T B q            (Gelec + Ghb: polar ALPB/GB term, H-bond term)
//!        + sum_i gamma_i S_i      (Gsasa: surface tension times smoothed SASA)
//!        + G_shift                (empirical shift + reference-state shift)
//! ```
//!
//! where `q` are the Mulliken atomic partial charges (electrons negative), and
//! `B` is the Born matrix: the P16 (ALPB) or Still (GBSA) kernel, plus
//! `2 * hbw_i` on the diagonal (H-bond correction, `hbw_i = hb_i * S_i/R_i^2`),
//! plus the ALPB shape term `keps * alpha/eps / A_det` in every element.
//!
//! Usage in an SCF: build [`Solvation`] once per molecule, call
//! [`Solvation::update`] once per geometry, then every iteration
//! [`Solvation::potential`] (adds `dE/dq = B q` to the atomic potential) and
//! [`Solvation::energy`] (or both at once with
//! [`Solvation::energy_and_potential`]); [`Solvation::fixed_energy`] provides
//! the charge-independent part and its gradient, [`Solvation::gradient`] the
//! explicit nuclear gradient of the charge-dependent part at fixed charges.
//! As in xtb, the full nuclear gradient also needs the solvation potential in
//! the atomic potential that multiplies dS/dR in the Hamiltonian gradient
//! (exactly like the isotropic Coulomb potential), since the Mulliken charges
//! depend on the overlap.
//!
//! Units: positions in bohr, energies in Hartree, gradients in Hartree/bohr.
//! Nothing here spawns threads or uses global state; [`Solvation`] is
//! immutable after construction and may be shared between threads.
//!
//! xtb quirks reproduced on purpose ([`Conventions`]): xtb 6.7.1's stale `r1` in one
//! branch of the Born radius integral (see `born.rs`; grimme-lab/xtb#1504), its density
//! term of the `reference` state (see [`state_shift`]; grimme-lab/xtb#1505) and its
//! Angstrom-to-bohr factor 1/0.52917726 for all radii ([`AATOAU`]); xtb 6.8.0 fixed the
//! two bugs and converts with mctc-lib's factor ([`XTB680_AATOAU`]). Both use xtb's own
//! constants and Lebedev grid selection (see [`SolvOptions`]).

#![allow(clippy::neg_cmp_op_on_partial_ord, clippy::manual_clamp)]

mod born;
mod hessian;
mod kernel;
pub mod lebedev;
// parameter tables keep their compact layout
#[rustfmt::skip]
mod lebedev_data;
#[rustfmt::skip]
mod params;
mod sasa;

use std::fmt;

pub use crate::Conventions as SolvConventions;
pub use kernel::Kernel;

/// Solvation model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Model {
    /// Analytical linearized Poisson-Boltzmann model (`xtb --alpb`): P16 kernel
    /// plus the ALPB shape correction.
    Alpb,
    /// Generalized Born with surface area (`xtb --gbsa`): Still kernel.
    Gbsa,
}

/// Reference state of the solvation free energy (`xtb --alpb <solvent> [state]`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default)]
pub enum ReferenceState {
    /// `gsolv` (xtb default): 1 L ideal gas / 1 L solution, no state shift.
    #[default]
    Gsolv,
    /// `bar1M`: 1 bar ideal gas / 1 mol/L solution.
    Bar1M,
    /// `reference`: `bar1M` plus a solvent-density dependent term (see
    /// [`state_shift`]; xtb 6.7.1 uses the wrong density, grimme-lab/xtb#1505).
    Reference,
}

/// Optional settings (defaults as in xtb).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SolvOptions {
    /// Temperature in K for the reference-state shift (xtb default 298.15 K;
    /// xtb only changes it through `$md temp=`).
    pub temperature: f64,
    /// Number of Lebedev grid points per atom for the SASA (xtb default 230;
    /// `$gbsa gbsagrid=`).  Mapped to a grid like xtb's `bisectSearch`: the
    /// largest grid not larger than the request, except that 5810 gives 5294.
    pub n_ang: usize,
    /// Numerical conventions: `Xtb` (default here) reproduces the xtb 6.8.0 program,
    /// `Xtb671` xtb 6.7.1 including its two bugs; `Corrected` uses CODATA 2022 constants,
    /// the exact ideal-gas molar volume, the solvent mass density in the `reference` state,
    /// and a fresh `r1` for every pair of the Born-radius integral (fixes for
    /// grimme-lab/xtb#1504, #1505, as in xtb 6.8.0 and tblite).
    pub conventions: Conventions,
}

/// Numerical conventions of the solvation model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Conventions {
    /// Reproduce the xtb 6.8.0 program: xtb's constants, mctc-lib's Ångström-to-bohr factor
    /// ([`XTB680_AATOAU`]), the solvent mass density in the `reference` state and a fresh
    /// `r1` for every pair of the Born-radius integral (6.8.0 fixed grimme-lab/xtb#1504 and
    /// #1505).
    #[default]
    Xtb,
    /// Reproduce the xtb 6.7.1 program exactly, quirks included: its Ångström-to-bohr factor
    /// 1/0.52917726, its `rho/M` term in the `reference` state (#1505) and the stale `r1`
    /// (#1504).
    Xtb671,
    /// Corrected and self-consistent (see [`SolvOptions::conventions`]).
    Corrected,
}

// The CODATA 2022 values of the corrected conventions. This crate cannot depend on axtb, whose
// `units` module holds the same values (with their sources) and checks that they agree.
/// CODATA 2022 Bohr radius in Ångström (NIST, via AFH's qcref).
pub const CODATA2022_AUTOAA: f64 = 0.529177210544;
/// CODATA 2022 Boltzmann constant in Hartree per kelvin ("kelvin-hartree relationship").
pub const CODATA2022_KB: f64 = 3.1668115634564e-6;
/// Hartree in kcal/mol from CODATA 2022 (hartree-joule relationship × Avogadro / 4184 J).
pub const CODATA2022_AUTOKCAL: f64 = 4.3597447222060e-18 * 6.02214076e23 / 4184.0;
/// Ideal-gas molar volume (L) at 298.15 K and 1 bar from the CODATA 2022 molar gas
/// constant R = N_A k_B (exact): R T / p.
const CODATA2022_IDEAL_GAS_MOL_VOLUME: f64 = 6.02214076e23 * 1.380649e-23 * 298.15 / 1.0e5 * 1.0e3;

/// The settings that differ between [`Conventions`], resolved once in
/// [`Solvation::with_options`].
#[derive(Clone, Copy, Debug)]
struct ConventionParams {
    /// Bohr per Ångström for every radius.
    aatoau: f64,
    /// Solvent free-energy shift (Eh) of a parameter set in a reference state at a
    /// temperature (K): the empirical `gshift` plus the reference-state term.
    gshift: fn(&params::RawParam, ReferenceState, f64) -> f64,
    /// Recompute `r1` for every pair of the Born-radius integral (grimme-lab/xtb#1504).
    fresh_r1: bool,
}

impl Conventions {
    fn params(self) -> ConventionParams {
        match self {
            Conventions::Xtb => ConventionParams { aatoau: XTB680_AATOAU, gshift: gshift_xtb680, fresh_r1: true },
            Conventions::Xtb671 => ConventionParams { aatoau: AATOAU, gshift: gshift_xtb, fresh_r1: false },
            Conventions::Corrected => {
                ConventionParams { aatoau: 1.0 / CODATA2022_AUTOAA, gshift: gshift_corrected, fresh_r1: true }
            }
        }
    }
}

/// xtb 6.7.1's shift: its constants and its number-density `rho/M` term in the `reference`
/// state.
fn gshift_xtb(p: &params::RawParam, state: ReferenceState, temperature: f64) -> f64 {
    let density = p.rhos / p.smass * params::MOLCM3TOAU;
    p.gshift * KCALTOAU + state_shift(state, temperature, density, p.smass)
}

/// xtb 6.8.0's shift: its constants and the solvent's mass density (kg/L = g/cm³) in the
/// `reference` state (`model.f90`: `self%density = param%rhos`).
fn gshift_xtb680(p: &params::RawParam, state: ReferenceState, temperature: f64) -> f64 {
    p.gshift * KCALTOAU + state_shift(state, temperature, p.rhos, p.smass)
}

/// CODATA 2022 constants and the solvent's mass density (kg/L = g/cm³) in the `reference`
/// state, as tblite does (grimme-lab/xtb#1505).
fn gshift_corrected(p: &params::RawParam, state: ReferenceState, temperature: f64) -> f64 {
    p.gshift / CODATA2022_AUTOKCAL
        + state_shift_with(state, temperature, p.rhos, p.smass, CODATA2022_KB, CODATA2022_IDEAL_GAS_MOL_VOLUME)
}

impl Default for SolvOptions {
    fn default() -> Self {
        Self { temperature: params::DEFAULT_TEMPERATURE, n_ang: params::DEFAULT_NANG, conventions: Conventions::Xtb }
    }
}

/// Errors of the solvation setup.
#[derive(Clone, Debug, PartialEq)]
pub enum SolvError {
    /// The solvent has no GFN2-xTB parameters for this model.
    UnknownSolvent { solvent: String, model: Model },
    /// Atomic number outside 1..=94.
    UnsupportedElement { atom: usize, number: usize },
    /// No Lebedev grid for the requested number of points.
    InvalidGrid(usize),
}

impl fmt::Display for SolvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolvError::UnknownSolvent { solvent, model } => {
                let m = match model {
                    Model::Alpb => "ALPB",
                    Model::Gbsa => "GBSA",
                };
                write!(f, "solvent '{solvent}' is not parametrized for GFN2-xTB/{m}")
            }
            SolvError::UnsupportedElement { atom, number } => {
                write!(f, "atom {atom}: no solvation parameters for atomic number {number}")
            }
            SolvError::InvalidGrid(n) => write!(f, "no Lebedev grid for {n} points"),
        }
    }
}

impl std::error::Error for SolvError {}

/// Solvation free energy components, as printed by xtb (`-> Gsolv`, ...).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct SolvEnergies {
    /// Total solvation free energy `gelec + gsasa + ghb + gshift`.
    pub gsolv: f64,
    /// Polar (ALPB/GB) part, without the H-bond term.
    pub gelec: f64,
    /// Non-polar surface term.
    pub gsasa: f64,
    /// H-bond correction.
    pub ghb: f64,
    /// Empirical shift plus reference-state shift.
    pub gshift: f64,
}

/// Names a solvent can be given with, per model (all aliases xtb accepts).
pub fn solvent_names(model: Model) -> Vec<&'static str> {
    let table = match model {
        Model::Alpb => params::GFN2_ALPB,
        Model::Gbsa => params::GFN2_GBSA,
    };
    let mut names: Vec<&'static str> = Vec::new();
    for (alias, target) in params::NORMALIZE {
        if table.iter().any(|(n, _)| n == target) && !names.contains(alias) {
            names.push(alias);
        }
    }
    for (n, _) in table {
        if !names.contains(n) {
            names.push(n);
        }
    }
    names
}

fn lookup(solvent: &str, model: Model) -> Result<&'static params::RawParam, SolvError> {
    // xtb: lowercase, then normalizeSolventName, then the GFN2 selection table
    let lower = solvent.trim().to_ascii_lowercase();
    let norm = params::NORMALIZE.iter().find(|(a, _)| *a == lower).map(|(_, t)| (*t).to_string()).unwrap_or(lower);
    let table = match model {
        Model::Alpb => params::GFN2_ALPB,
        Model::Gbsa => params::GFN2_GBSA,
    };
    table
        .iter()
        .find(|(n, _)| *n == norm)
        .map(|(_, p)| *p)
        .ok_or_else(|| SolvError::UnknownSolvent { solvent: solvent.to_string(), model })
}

/// Reference-state shift in Hartree (xtb `getStateShift`), with xtb's constants.
///
/// `density` is the solvent density xtb passes: 6.7.1 its internal `rho/M * 8.92388e-2`
/// (rho in g/cm^3, M in g/mol), so that its `reference` state uses
/// `ln(rho * 89.2388 / M^2)` where the physical solvent concentration term is
/// `ln(1000 rho / M)` (grimme-lab/xtb#1505); 6.8.0 the mass density rho.
/// `molar_mass` in g/mol.
pub fn state_shift(state: ReferenceState, temperature: f64, density: f64, molar_mass: f64) -> f64 {
    state_shift_with(state, temperature, density, molar_mass, params::KB, params::IDEAL_GAS_MOL_VOLUME)
}

/// [`state_shift`] with explicit Boltzmann constant (Eh/K) and ideal-gas molar volume (L).
pub fn state_shift_with(
    state: ReferenceState,
    temperature: f64,
    density: f64,
    molar_mass: f64,
    kb: f64,
    ideal_gas_mol_volume: f64,
) -> f64 {
    use params::{AMBIENT_TEMPERATURE, REF_DENSITY, REF_MOLECULAR_MASS};
    match state {
        ReferenceState::Gsolv => 0.0,
        ReferenceState::Reference => {
            temperature
                * kb
                * ((ideal_gas_mol_volume * temperature / AMBIENT_TEMPERATURE).ln()
                    + (density / REF_DENSITY * REF_MOLECULAR_MASS / molar_mass).ln())
        }
        ReferenceState::Bar1M => temperature * kb * (ideal_gas_mol_volume * temperature / AMBIENT_TEMPERATURE).ln(),
    }
}

/// Bohr per Angstrom as used by xtb 6.7.1 (1 / 0.52917726, not CODATA).
pub const AATOAU: f64 = 1.0 / params::AUTOAA;
/// Bohr per Angstrom as used by xtb 6.8.0: mctc-lib's factor, derived from CODATA 2018
/// (xtb src/mctc/convert.f90 `autoaa = 5.29177210904492434e-1_wp`).
pub const XTB680_AATOAU: f64 = 1.0 / 5.29177210904492434e-1;
const KCALTOAU: f64 = 1.0 / params::AUTOKCAL;
const FOURPI: f64 = 4.0 * params::PI;

/// ALPB/GBSA solvation model for one molecule (atom types fixed).
#[derive(Clone, Debug)]
pub struct Solvation {
    nat: usize,
    model: Model,
    kernel: Kernel,
    state: ReferenceState,
    param_name: &'static str,
    dielectric: f64,
    temperature: f64,
    keps: f64,
    alpbet: f64,
    born_scale: f64,
    born_offset: f64,
    probe_rad: f64,
    gshift: f64,
    lrcut: f64,
    srcut: f64,
    smooth_w: f64,
    smooth: sasa::Smooth,
    vdwr: Vec<f64>,
    rho: Vec<f64>,
    svdw: Vec<f64>,
    vdwsa: Vec<f64>,
    trj2: Vec<[f64; 2]>,
    wrp: Vec<f64>,
    gamsasa: Vec<f64>,
    hbmag: Vec<f64>,
    lhb: bool,
    ang_grid: Vec<[f64; 3]>,
    ang_weight: Vec<f64>,
    /// Recompute `r1` for every pair of the Born-radius integral (xtb 6.8.0 and corrected
    /// conventions).
    fresh_r1: bool,
}

/// Geometry-dependent intermediates of [`Solvation`] (from [`Solvation::update`]).
#[derive(Clone, Debug, Default)]
pub struct SolvCache {
    nat: usize,
    xyz: Vec<[f64; 3]>,
    brad: Vec<f64>,
    dbrad: Vec<f64>,
    sasa: Vec<f64>,
    deriv: sasa::SurfaceDeriv,
    scratch: sasa::SasaScratch,
    dsdr: Vec<[f64; 3]>,
    gsasa: f64,
    hbw: Vec<f64>,
    dhbdw: Vec<f64>,
    adet: f64,
    born_mat: Vec<f64>,
}

impl SolvCache {
    /// Born radii (bohr).
    pub fn born_radii(&self) -> &[f64] {
        &self.brad
    }

    /// Smoothed solvent accessible surface area per atom in bohr^2.
    pub fn surface_area(&self) -> Vec<f64> {
        self.sasa.iter().map(|s| s * FOURPI).collect()
    }

    /// Born interaction matrix `B` (row-major `nat x nat`, symmetric), such
    /// that the charge-dependent energy is `1/2 q^T B q` and the potential
    /// `B q`.  Includes the H-bond diagonal and the ALPB shape term.
    pub fn born_matrix(&self) -> &[f64] {
        &self.born_mat
    }

    /// ALPB shape descriptor `A_det` (bohr); zero for GBSA.
    pub fn adet(&self) -> f64 {
        self.adet
    }
}

impl Solvation {
    /// GFN2-xTB solvation model with xtb's default options.
    ///
    /// `solvent` is matched case-insensitively with xtb's aliases
    /// (e.g. `h2o`, `dichloromethane`, `n-hexane`); see [`solvent_names`].
    pub fn new(numbers: &[usize], solvent: &str, model: Model, state: ReferenceState) -> Result<Self, SolvError> {
        Self::with_options(numbers, solvent, model, state, &SolvOptions::default())
    }

    /// As [`Solvation::new`] with explicit options.
    pub fn with_options(
        numbers: &[usize],
        solvent: &str,
        model: Model,
        state: ReferenceState,
        opts: &SolvOptions,
    ) -> Result<Self, SolvError> {
        use params::*;
        for (atom, &z) in numbers.iter().enumerate() {
            if z == 0 || z > NELEM {
                return Err(SolvError::UnsupportedElement { atom, number: z });
            }
        }
        let p = lookup(solvent, model)?;
        let (kernel, alpb) = match model {
            Model::Alpb => (Kernel::P16, true),
            Model::Gbsa => (Kernel::Still, false),
        };

        // xtb paramToModel
        let conv = opts.conventions.params();
        let aatoau = conv.aatoau;
        let dielectric = p.epsv;
        let born_scale = p.c1;
        let born_offset = p.soset * 0.1 * aatoau;
        let probe_rad = p.rprobe * aatoau;
        let any_hb = p.tmp.iter().any(|t| t.abs() > 1.0e-3);
        let gshift = (conv.gshift)(p, state, opts.temperature);

        // xtb initBorn
        let alpbet = if alpb { ALPB_ALPHA / dielectric } else { 0.0 };
        let keps = (1.0 / dielectric - 1.0) / (1.0 + alpbet);
        let lrcut = LRCUT_AA * aatoau;
        let w = SMOOTH_W_AA * aatoau;
        let w3 = w * (w * w);
        let smooth = sasa::Smooth { ah0: 0.5, ah1: 3.0 / (4.0 * w), ah3: -1.0 / (4.0 * w3) };

        let nat = numbers.len();
        let mut vdwr = Vec::with_capacity(nat);
        let mut rho = Vec::with_capacity(nat);
        let mut svdw = Vec::with_capacity(nat);
        let mut vdwsa = Vec::with_capacity(nat);
        let mut trj2 = Vec::with_capacity(nat);
        let mut wrp = Vec::with_capacity(nat);
        let mut gamsasa = Vec::with_capacity(nat);
        let mut hbmag = Vec::with_capacity(nat);
        for &z in numbers {
            let iz = z - 1;
            let rvdw = aatoau * VDW_RAD_D3_AA[iz];
            vdwr.push(rvdw);
            rho.push(rvdw * p.sx[iz]);
            svdw.push(rvdw - born_offset);
            let rsa = rvdw + probe_rad;
            vdwsa.push(rsa);
            trj2.push([(rsa - w) * (rsa - w), (rsa + w) * (rsa + w)]);
            let ah3 = smooth.ah3;
            let r = rsa + w;
            let mut wr = (0.25 / w + 3.0 * ah3 * (0.2 * r * r - 0.5 * r * rsa + rsa * rsa / 3.0)) * r * r * r;
            let r = rsa - w;
            wr -= (0.25 / w + 3.0 * ah3 * (0.2 * r * r - 0.5 * r * rsa + rsa * rsa / 3.0)) * r * r * r;
            wrp.push(wr);
            gamsasa.push(p.gamscale[iz] * FOURPI * SURFACE_TENSION);
            hbmag.push(if any_hb { -(p.tmp[iz] * p.tmp[iz]) * KCALTOAU } else { 0.0 });
        }
        let maxsa = vdwsa.iter().fold(f64::NEG_INFINITY, |m: f64, &x| m.max(x));
        let srcut = 2.0 * (w + maxsa) + SRCUT_AA * aatoau;
        let lhb = hbmag.iter().any(|&h| h < 0.0);

        let iang = lebedev::bisect_grid(opts.n_ang);
        let (ang_grid, ang_weight) = lebedev::angular_grid(iang).ok_or(SolvError::InvalidGrid(opts.n_ang))?;

        Ok(Self {
            nat,
            fresh_r1: conv.fresh_r1,
            model,
            kernel,
            state,
            param_name: p.name,
            dielectric,
            temperature: opts.temperature,
            keps,
            alpbet,
            born_scale,
            born_offset,
            probe_rad,
            gshift,
            lrcut,
            srcut,
            smooth_w: w,
            smooth,
            vdwr,
            rho,
            svdw,
            vdwsa,
            trj2,
            wrp,
            gamsasa,
            hbmag,
            lhb,
            ang_grid,
            ang_weight,
        })
    }

    /// Number of atoms.
    pub fn natoms(&self) -> usize {
        self.nat
    }
    /// Solvation model.
    pub fn model(&self) -> Model {
        self.model
    }
    /// Generalized Born kernel (P16 for ALPB, Still for GBSA).
    pub fn kernel(&self) -> Kernel {
        self.kernel
    }
    /// Reference state.
    pub fn reference_state(&self) -> ReferenceState {
        self.state
    }
    /// Name of the xtb parameter set, e.g. `gfn2_alpb_water`.
    pub fn parameter_set(&self) -> &'static str {
        self.param_name
    }
    /// Dielectric constant of the solvent.
    pub fn dielectric_constant(&self) -> f64 {
        self.dielectric
    }
    /// Temperature used for the reference-state shift (K).
    pub fn temperature(&self) -> f64 {
        self.temperature
    }
    /// Total free energy shift (empirical + reference state), Hartree.
    pub fn free_energy_shift(&self) -> f64 {
        self.gshift
    }
    /// Born radius scaling factor c1.
    pub fn born_scale(&self) -> f64 {
        self.born_scale
    }
    /// Born offset (bohr).
    pub fn born_offset(&self) -> f64 {
        self.born_offset
    }
    /// Solvent probe radius for the SASA (bohr).
    pub fn probe_radius(&self) -> f64 {
        self.probe_rad
    }
    /// Number of Lebedev points per atom actually used.
    pub fn grid_points(&self) -> usize {
        self.ang_grid.len()
    }
    /// Whether the H-bond correction is active for this molecule.
    pub fn has_hbond(&self) -> bool {
        self.lhb
    }

    /// Compute the geometry-dependent intermediates (Born radii, SASA,
    /// Born matrix, ...).
    pub fn update(&self, positions: &[[f64; 3]]) -> SolvCache {
        let mut cache = SolvCache::default();
        self.update_into(positions, &mut cache);
        cache
    }

    /// As [`Solvation::update`], reusing the buffers of `cache`.
    pub fn update_into(&self, positions: &[[f64; 3]], cache: &mut SolvCache) {
        let nat = self.nat;
        assert_eq!(positions.len(), nat, "number of positions does not match the number of atoms");
        let c = cache;
        c.nat = nat;
        c.xyz.clear();
        c.xyz.extend_from_slice(positions);
        resize(&mut c.brad, nat);
        resize(&mut c.dbrad, nat);
        resize(&mut c.sasa, nat);
        resize(&mut c.hbw, nat);
        resize(&mut c.dhbdw, nat);
        c.dsdr.clear();
        c.dsdr.resize(nat, [0.0; 3]);
        resize(&mut c.born_mat, nat * nat);

        born::born_radii(
            &c.xyz,
            &self.vdwr,
            &self.rho,
            &self.svdw,
            self.born_scale,
            self.lrcut,
            &mut c.brad,
            &mut c.dbrad,
            self.fresh_r1,
        );

        sasa::neighbour_list(&c.xyz, self.srcut, &self.vdwsa, self.smooth_w, &mut c.deriv);
        sasa::compute_numsa(
            &c.xyz,
            &self.vdwsa,
            &self.wrp,
            &self.trj2,
            self.smooth,
            params::TOLSESP,
            &self.ang_grid,
            &self.ang_weight,
            &mut c.sasa,
            &mut c.deriv,
            &mut c.scratch,
        );
        c.deriv.contract(&self.gamsasa, &mut c.dsdr);
        c.gsasa = c.sasa.iter().zip(&self.gamsasa).fold(0.0, |s, (a, g)| s + a * g);

        if self.lhb {
            for i in 0..nat {
                let smaxd = 1.0 / (self.vdwsa[i] * self.vdwsa[i]);
                let sasad = c.sasa[i] * smaxd;
                c.hbw[i] = self.hbmag[i] * sasad;
                c.dhbdw[i] = self.hbmag[i] * smaxd;
            }
        }

        c.adet = if self.alpbet > 0.0 { kernel::adet(&c.xyz, &self.vdwr) } else { 0.0 };

        let b = &mut c.born_mat;
        kernel::born_matrix(self.kernel, &c.xyz, self.keps, &c.brad, b);
        if self.lhb {
            for i in 0..nat {
                b[i * nat + i] += 2.0 * c.hbw[i];
            }
        }
        if self.alpbet > 0.0 {
            let shift = self.keps * self.alpbet / c.adet;
            b.iter_mut().for_each(|x| *x += shift);
        }
    }

    fn check(&self, cache: &SolvCache) {
        assert_eq!(cache.nat, self.nat, "solvation cache was not computed for this model/molecule");
    }

    /// Charge-independent part: surface term `gamma_i S_i` and the free energy
    /// shift (distributed evenly over the atoms), added to `energies`; its
    /// gradient is added to `grad` if given.
    pub fn fixed_energy(&self, cache: &SolvCache, energies: &mut [f64], grad: Option<&mut [[f64; 3]]>) {
        self.check(cache);
        let shift = self.gshift / self.nat as f64;
        for i in 0..self.nat {
            energies[i] += cache.sasa[i] * self.gamsasa[i];
            energies[i] += shift;
        }
        if let Some(g) = grad {
            for (gi, d) in g.iter_mut().zip(&cache.dsdr) {
                for k in 0..3 {
                    gi[k] += d[k];
                }
            }
        }
    }

    /// Charge-dependent part (polar ALPB/GB term and H-bond term),
    /// `E_i = 1/2 q_i (B q)_i`, added to `energies`.
    pub fn energy(&self, cache: &SolvCache, q: &[f64], energies: &mut [f64]) {
        self.check(cache);
        let nat = self.nat;
        for i in 0..nat {
            let row = &cache.born_mat[i * nat..(i + 1) * nat];
            let s = 0.5 * dot(row, q);
            energies[i] += q[i] * s;
        }
    }

    /// Potential `dE/dq_i = (B q)_i` added to `vat` (same convention as a
    /// Coulomb potential: E = 1/2 q^T B q with q the partial charges).
    pub fn potential(&self, cache: &SolvCache, q: &[f64], vat: &mut [f64]) {
        self.check(cache);
        let nat = self.nat;
        for i in 0..nat {
            let row = &cache.born_mat[i * nat..(i + 1) * nat];
            vat[i] += dot(row, q);
        }
    }

    /// [`Solvation::energy`] and [`Solvation::potential`] in one pass over
    /// the Born matrix (`E_i = 1/2 q_i V_i` with `V = B q`).
    pub fn energy_and_potential(&self, cache: &SolvCache, q: &[f64], energies: &mut [f64], vat: &mut [f64]) {
        self.check(cache);
        let nat = self.nat;
        for i in 0..nat {
            let row = &cache.born_mat[i * nat..(i + 1) * nat];
            let v = dot(row, q);
            vat[i] += v;
            energies[i] += q[i] * (0.5 * v);
        }
    }

    /// Explicit nuclear gradient of the charge-dependent part at fixed
    /// charges (kernel, Born radii, H-bond surface dependence, ALPB shape
    /// term), added to `grad`.  The surface-tension gradient is part of
    /// [`Solvation::fixed_energy`].
    pub fn gradient(&self, cache: &SolvCache, q: &[f64], grad: &mut [[f64; 3]]) {
        self.check(cache);
        let nat = self.nat;
        let mut work = vec![0.0f64; nat];
        kernel::add_gradient(self.kernel, &cache.xyz, q, self.keps, &cache.brad, grad, &mut work);
        for i in 0..nat {
            work[i] *= cache.dbrad[i];
        }
        born::add_psi_gradient(&cache.xyz, &self.vdwr, &self.rho, self.lrcut, &work, grad, self.fresh_r1);
        if self.lhb {
            for i in 0..nat {
                work[i] = cache.dhbdw[i] * q[i] * q[i];
            }
            cache.deriv.contract(&work, grad);
        }
        if self.alpbet > 0.0 {
            kernel::add_adet_deriv(&cache.xyz, &self.vdwr, self.keps * self.alpbet, q, grad);
        }
    }

    /// Energy components as xtb reports them (`-> Gsolv`, `Gelec`, `Gsasa`,
    /// `Ghb`, `Gshift`), computed like xtb's `getEnergyParts`.
    pub fn energy_parts(&self, cache: &SolvCache, q: &[f64]) -> SolvEnergies {
        self.check(cache);
        let nat = self.nat;
        let mut ghb = 0.0;
        if self.lhb {
            for i in 0..nat {
                ghb += cache.hbw[i] * (q[i] * q[i]);
            }
        }
        let mut epol = 0.0;
        for i in 0..nat {
            let row = &cache.born_mat[i * nat..(i + 1) * nat];
            epol += q[i] * (0.5 * dot(row, q));
        }
        let gelec = epol - ghb;
        let gsasa = cache.gsasa;
        let gshift = self.gshift;
        SolvEnergies { gsolv: gelec + gsasa + ghb + gshift, gelec, gsasa, ghb, gshift }
    }

    /// Total solvation free energy (all parts) at charges `q`.
    pub fn total_energy(&self, cache: &SolvCache, q: &[f64]) -> f64 {
        self.energy_parts(cache, q).gsolv
    }
}

/// Dot product with four interleaved partial sums (vectorisable, fixed order).
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let n = a.len().min(b.len());
    let (a, b) = (&a[..n], &b[..n]);
    let mut acc = [0.0f64; 4];
    let (ca, ra) = a.as_chunks::<4>();
    let (cb, rb) = b.as_chunks::<4>();
    for (x, y) in ca.iter().zip(cb) {
        for k in 0..4 {
            acc[k] += x[k] * y[k];
        }
    }
    let mut s = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

fn resize(v: &mut Vec<f64>, n: usize) {
    v.clear();
    v.resize(n, 0.0);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn read_ref(name: &str) -> (Vec<usize>, Vec<[f64; 3]>) {
        let path = format!("{}/tests/data/{name}.ref", env!("CARGO_MANIFEST_DIR"));
        let (mut z, mut x) = (vec![], vec![]);
        for line in std::fs::read_to_string(path).unwrap().lines() {
            let t: Vec<&str> = line.split_whitespace().collect();
            if t.first() == Some(&"atom") {
                z.push(t[1].parse().unwrap());
                x.push([t[2].parse().unwrap(), t[3].parse().unwrap(), t[4].parse().unwrap()]);
            }
        }
        (z, x)
    }

    /// The optimised SASA (pruned neighbour list, burial pre-pass, sparse
    /// derivative) is bitwise identical to the plain port of xtb's code.
    #[test]
    fn optimised_sasa_is_bitwise_identical_to_plain_port() {
        for (name, solvent, model) in [
            ("taxol_alpb_water", "water", Model::Alpb),
            ("caffeine_alpb_water", "hexane", Model::Gbsa),
            ("zncl4_alpb_water", "toluene", Model::Alpb),
            ("ferrocene_alpb_water", "dmso", Model::Alpb),
        ] {
            let (z, x) = read_ref(name);
            let s = Solvation::new(&z, solvent, model, ReferenceState::Gsolv).unwrap();
            let c = s.update(&x);
            let (sasa, dsdrt) = sasa::compute_numsa_reference(
                &x,
                s.srcut,
                &s.vdwsa,
                &s.wrp,
                &s.trj2,
                s.smooth,
                params::TOLSESP,
                &s.ang_grid,
                &s.ang_weight,
            );
            assert_eq!(c.sasa, sasa, "{name}");
            let nat = z.len();
            let mut dense = vec![vec![[0.0f64; 3]; nat]; nat];
            for i in 0..nat {
                dense[i][i] = c.deriv.self_grad[i];
                for p in c.deriv.nb_ptr[i]..c.deriv.nb_ptr[i + 1] {
                    dense[i][c.deriv.nb_idx[p]] = c.deriv.nb_grad[p];
                }
            }
            assert_eq!(dense, dsdrt, "{name}");
        }
    }
}
