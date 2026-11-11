//! Vibrational analysis of a nuclear Hessian: harmonic frequencies, normal modes, reduced
//! masses, IR intensities and quasi-RRHO thermochemistry ([`thermo`]).
//!
//! The analysis is independent of the method: [`analyze`] takes atomic numbers, positions
//! (bohr), any Cartesian Hessian (Eh/bohr², row-major 3N x 3N) and optionally the dipole
//! derivatives ∂μ/∂R (for IR intensities), so it serves GFN2-xTB, g-xTB and any other method
//! that provides a Hessian.
//!
//! Three conventions ([`Conventions`]; what differs between them is one table,
//! [`AnalysisParams`]):
//!
//! - `Xtb` reproduces the xtb 6.8.0 program (`xtb --hess`, and `xtb thermo` on its `hessian`
//!   file): xtb's masses and constants; the molecule is linear when its unit-mass moments of
//!   inertia about the centroid have one below 1e-4 bohr² (`is_linear`); translations and the
//!   three (linear: two) rotations about the center of mass projected in mass-weighted
//!   coordinates (`projectHessian`) and the projected mass-weighted Hessian diagonalized
//!   whole; its first modes below 0.01 cm⁻¹, as many as were projected, are the rigid-body
//!   modes; IR intensities with its factor 1.7770969e6 km/mol per au; its thermochemistry
//!   (see [`thermo`]) with mctc-lib's CODATA 2018 bohr in the moments of inertia.
//! - `Xtb671` reproduces the xtb 6.7.1 program (the g-xTB 2.0.1 build of xtb, `26dd68d`):
//!   the same masses and constants, but translations and rotations (about the geometric
//!   center) projected from the Cartesian Hessian before mass weighting (`trproj`), no
//!   projection for linear molecules (`axis`: smallest moment below 3e-4 u Å²; the five lowest
//!   modes are dropped instead), the six modes below 0.01 cm⁻¹ taken as the rigid-body modes,
//!   and the bohr 0.52917726 Å in the thermochemistry's moments of inertia.
//! - `Axtb` (default): standard atomic masses ([`masses::STANDARD`], from qcref) and CODATA
//!   2022 constants; translations and rotations about the center of mass (five for linear
//!   molecules) projected in mass-weighted coordinates, and the mass-weighted Hessian
//!   diagonalized in their orthogonal complement, so there are always exactly 3N-6 (3N-5)
//!   vibrations whatever their size; the point group's exact symmetry number.
//!
//! xtb 6.7.1 also rotated the molecule by (1, 2, 3)·1e-4 degrees before its numerical Hessian
//! (`rotmol`), so its printout belongs to the rotated structure; 6.8.0 does not rotate.
//! The analysis takes the positions it is given in all conventions.
//!
//! At a stationary point with an exact Hessian both projections give the same frequencies;
//! away from one (or with a noisy Hessian) the mass-weighted projection is the consistent
//! one (vibrations orthogonal to the rigid-body motions in the kinetic-energy metric).
//!
//! Normal modes are returned as Cartesian displacements scaled to unit length (qcdata's
//! `normal_modes_cartesian`); reduced masses are 1/Σ_k l_k²/m_k (u) for the normalized
//! mass-weighted eigenvector l; IR intensities are N_A π/(3c²) |∂μ/∂Q|² (km/mol).


#[rustfmt::skip] // parameter tables keep their compact layout
pub mod masses;
pub mod symmetry;
pub mod thermo;

use crate::units;
use crate::{Error, Result};
use faer::dyn_stack::{MemBuffer, MemStack};
use faer::linalg::evd::{self_adjoint_evd, self_adjoint_evd_scratch, ComputeEigenvectors};
use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatRef, Par};

pub use symmetry::{point_group, PointGroup};
pub use thermo::{thermochemistry, Contribution, Thermo, ThermoOptions, ThermoPoint};

/// Constants and unit conversions of the xtb 6.7.1 and 6.8.0 programs (`src/mctc/convert.f90`,
/// `src/mctc/mctc_constants.f90`, `src/axis_trafo.f90`), evaluated in the same order. The two
/// versions differ only in the bohr radius of `convert.f90` ([`AUTOAA`], [`AUTOAA_680`]).
pub(crate) mod xtbc {
    pub const PI: f64 = std::f64::consts::PI;
    /// xtb 6.7.1's bohr radius in Å.
    pub const AUTOAA: f64 = 0.52917726;
    /// xtb 6.8.0's bohr radius in Å: mctc-lib's, derived from CODATA 2018.
    pub const AUTOAA_680: f64 = 5.29177210904492434e-1;
    /// `PARAMETER (BOHR=0.52917726)` in `axis` (a default-real literal: single precision).
    pub const AXIS_BOHR: f64 = 0.52917726_f32 as f64;
    pub const AUTORCM: f64 = 219474.63067;
    pub const RCMTOAU: f64 = 1.0 / AUTORCM;
    pub const AUTOKCAL: f64 = 627.50947428;
    pub const KCALTOAU: f64 = 1.0 / AUTOKCAL;
    pub const AMUTOKG: f64 = 1.660539040e-27;
    pub const METOKG: f64 = 9.10938356e-31;
    pub const KGTOME: f64 = 1.0 / METOKG;
    pub const AMUTOAU: f64 = AMUTOKG * KGTOME;
    pub const KB: f64 = crate::units::reference::KB_XTB_TBLITE;
    pub const MOLAR_GAS: f64 = 8.314462618;
    pub const CALTOJ: f64 = 4.184;
    pub const JTOCAL: f64 = 1.0 / CALTOJ;
    const BOHR_RADIUS: f64 = 5.29177210903e-11;
    const AU_TIME: f64 = 2.4188843265857e-17;
    const AUTOPA: f64 = METOKG / (BOHR_RADIUS * (AU_TIME * AU_TIME));
    pub const PATOAU: f64 = 1.0 / AUTOPA;
    pub const ATMTOAU: f64 = 101325.0 * PATOAU;
    pub const AUTOKMMOL: f64 = 1.7770969e+6;
    /// `axis`/`axis2`: amu Å² to 1e-47 kg m², and amu Å² to cm⁻¹.
    pub const CONST1: f64 = 1.66053;
    pub const CONST2: f64 = 16.8576522;
}

/// Constants, masses and projection of the analysis (see the module documentation).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum Conventions {
    /// CODATA 2022 constants, standard atomic masses, mass-weighted projection.
    #[default]
    Axtb,
    /// The xtb 6.8.0 program.
    Xtb,
    /// The xtb 6.7.1 program (also what the g-xTB 2.0.1 program, a 6.7.1-based xtb build,
    /// does).
    Xtb671,
}

impl Conventions {
    /// `axtb`, `xtb` or `xtb-6.7.1`, as parsed.
    pub fn label(self) -> &'static str {
        match self {
            Conventions::Axtb => "axtb",
            Conventions::Xtb => "xtb",
            Conventions::Xtb671 => "xtb-6.7.1",
        }
    }

    /// Whether these are one of the xtb programs' conventions.
    pub fn is_xtb(self) -> bool {
        self != Conventions::Axtb
    }

    /// The settings of these conventions.
    pub fn params(self) -> AnalysisParams {
        match self {
            Conventions::Axtb => AnalysisParams {
                masses: &masses::STANDARD,
                projection: Projection::Complement,
                linear: LinearTest::PrincipalMoments,
                thermo_bohr: None,
                ir_factor: units::AU_TO_KM_PER_MOL,
            },
            Conventions::Xtb => AnalysisParams {
                masses: &masses::XTB,
                projection: Projection::MassWeighted,
                linear: LinearTest::UnitMassMoments,
                thermo_bohr: Some(xtbc::AUTOAA_680),
                ir_factor: xtbc::AUTOKMMOL,
            },
            Conventions::Xtb671 => AnalysisParams {
                masses: &masses::XTB,
                projection: Projection::Cartesian,
                linear: LinearTest::Axis,
                thermo_bohr: Some(xtbc::AUTOAA),
                ir_factor: xtbc::AUTOKMMOL,
            },
        }
    }
}

impl std::fmt::Display for Conventions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

impl std::str::FromStr for Conventions {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "axtb" | "default" => Ok(Conventions::Axtb),
            "xtb" | "xtb-6.8.0" => Ok(Conventions::Xtb),
            // the g-xTB 2.0.1 program is a 6.7.1-based xtb build
            "xtb-6.7.1" | "gxtb" => Ok(Conventions::Xtb671),
            other => Err(Error::Input(format!(
                "unknown vibrational-analysis conventions {other:?} (axtb, xtb or xtb-6.7.1)"
            ))),
        }
    }
}

impl From<crate::Conventions> for Conventions {
    /// The analysis conventions of a GFN2-xTB calculation: `xtb` for `xtb` (6.8.0),
    /// `xtb-6.7.1` for `xtb-6.7.1`, `axtb` otherwise (tblite has no vibrational analysis).
    fn from(c: crate::Conventions) -> Self {
        match c {
            crate::Conventions::Xtb => Conventions::Xtb,
            crate::Conventions::Xtb671 => Conventions::Xtb671,
            _ => Conventions::Axtb,
        }
    }
}

/// How translations and rotations are removed from the Hessian.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Projection {
    /// About the center of mass in mass-weighted coordinates (five for linear molecules, from
    /// the principal moments), the Hessian diagonalized in their orthogonal complement (axtb).
    Complement,
    /// About the center of mass in mass-weighted coordinates, three translations and three
    /// rotations (two for a molecule that `is_linear`), the projected Hessian diagonalized
    /// whole and its first modes below 0.01 cm⁻¹, as many as were projected, dropped
    /// (xtb 6.8.0 `projectHessian`).
    MassWeighted,
    /// About the geometric center from the Cartesian Hessian with unit masses, nothing for
    /// linear molecules (`axis`), the first six modes below 0.01 cm⁻¹ (linear: the five
    /// lowest) dropped (xtb 6.7.1 `trproj`).
    Cartesian,
}

/// When a molecule counts as linear (no rotation about its axis: 3N-5 vibrations, two
/// rotational degrees of freedom).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinearTest {
    /// Smallest principal moment of inertia below 3e-4 u Å² (axtb, CODATA 2022 bohr).
    PrincipalMoments,
    /// Smallest principal moment below 3e-4 u Å² as xtb 6.7.1's `axis` (Hessian, with its
    /// single-precision bohr) and `axis2` (thermochemistry: rotational constant below 1e-10)
    /// compute it.
    Axis,
    /// One of the unit-mass moments of inertia about the centroid below 1e-4 bohr²
    /// (xtb 6.8.0 `is_linear`, Hessian and thermochemistry).
    UnitMassMoments,
}

/// The settings that differ between [`Conventions`].
#[derive(Clone, Copy, Debug)]
pub struct AnalysisParams {
    /// Atomic masses (u) unless given.
    pub masses: &'static [f64; 118],
    pub projection: Projection,
    pub linear: LinearTest,
    /// Bohr radius (Å) of xtb's thermochemistry (`axis2`, `thermodyn`), or `None` for axtb's
    /// thermochemistry in atomic units with CODATA 2022 constants.
    pub thermo_bohr: Option<f64>,
    /// IR intensity in km/mol of |∂μ/∂Q|² = 1 au.
    pub ir_factor: f64,
}

/// Settings of [`analyze`].
#[derive(Clone, Debug, Default)]
pub struct VibOptions {
    pub conventions: Conventions,
    /// Atomic masses in u (e.g. isotopes); `None` for the conventions' table.
    pub masses: Option<Vec<f64>>,
    /// Thermochemistry; `None` for frequencies, modes and intensities only.
    pub thermo: Option<ThermoOptions>,
}

/// Result of [`analyze`].
#[derive(Clone, Debug)]
pub struct VibAnalysis {
    pub conventions: Conventions,
    /// Masses used (u).
    pub masses: Vec<f64>,
    pub linear: bool,
    /// Harmonic frequencies in cm⁻¹, ascending, imaginary ones negative: 3N-6 (3N-5 linear).
    pub frequencies: Vec<f64>,
    /// Normal modes: per mode, the Cartesian displacement of every atom, scaled to unit length.
    pub normal_modes: Vec<Vec<[f64; 3]>>,
    /// Reduced masses (u).
    pub reduced_masses: Vec<f64>,
    /// Force constants μ ω² in mdyn/Å (as Gaussian prints them).
    pub force_constants: Vec<f64>,
    /// IR intensities (km/mol), when dipole derivatives were given.
    pub ir_intensities: Option<Vec<f64>>,
    /// Number of imaginary (negative) frequencies.
    pub n_imaginary: usize,
    pub thermo: Option<Thermo>,
}

/// Symmetric eigendecomposition (ascending) of an n x n matrix.
fn eigh(a: &Mat<f64>) -> Result<(Vec<f64>, Mat<f64>)> {
    let n = a.nrows();
    let par = Par::Seq;
    let mut s = faer::diag::Diag::<f64>::zeros(n);
    let mut u = Mat::<f64>::zeros(n, n);
    let mut mem = MemBuffer::new(self_adjoint_evd_scratch::<f64>(n, ComputeEigenvectors::Yes, par, Default::default()));
    self_adjoint_evd(a.as_ref(), s.as_mut(), Some(u.as_mut()), par, MemStack::new(&mut mem), Default::default())
        .map_err(|_| Error::Numerical("eigensolver failed to converge".into()))?;
    Ok(((0..n).map(|i| s[i]).collect(), u))
}

/// The product `a b` on this thread. faer's `*` runs on its global parallelism, which is every
/// CPU unless the program changes it, and this analysis runs outside the calculator's pool.
fn mul(a: MatRef<'_, f64>, b: MatRef<'_, f64>) -> Mat<f64> {
    let mut c = Mat::zeros(a.nrows(), b.ncols());
    matmul(c.as_mut(), Accum::Replace, a, b, 1.0, Par::Seq);
    c
}

/// Mass-weighted eigenvectors of the vibrations (columns of `l`, orthonormal), their
/// eigenvalues (Eh/(bohr² mₑ)) and the factors 1/√m (mₑ) of every coordinate.
struct Modes {
    linear: bool,
    eigenvalues: Vec<f64>,
    l: Mat<f64>,
    inv_sqrt_m: Vec<f64>,
    to_wavenumber: f64,
}

/// The symmetric part of a row-major Hessian as a matrix.
fn symmetrized(h: &[f64], n3: usize) -> Mat<f64> {
    Mat::from_fn(n3, n3, |i, j| axtb_base::fd::symmetric_part(h, n3, i, j))
}

/// Gram-Schmidt orthonormalization (vectors below `eps` in squared norm after projection
/// are dropped).
fn orthonormalize(vecs: Vec<Vec<f64>>, eps: f64) -> Vec<Vec<f64>> {
    let mut out: Vec<Vec<f64>> = Vec::new();
    for mut v in vecs {
        for q in &out {
            let d: f64 = v.iter().zip(q).map(|(a, b)| a * b).sum();
            v.iter_mut().zip(q).for_each(|(a, b)| *a -= d * b);
        }
        let nn: f64 = v.iter().map(|x| x * x).sum();
        if nn < eps {
            continue;
        }
        let s = 1.0 / nn.sqrt();
        v.iter_mut().for_each(|x| *x *= s);
        out.push(v);
    }
    out
}

/// `axtb` conventions: projection in mass-weighted coordinates, diagonalization in the
/// complement of the rigid-body motions.
fn modes_axtb(positions: &[[f64; 3]], masses: &[f64], hessian: &[f64]) -> Result<Modes> {
    let nat = positions.len();
    let n3 = 3 * nat;
    let m_au: Vec<f64> = masses.iter().map(|m| m / units::ELECTRON_MASS_IN_U).collect();
    let inv_sqrt_m: Vec<f64> = (0..n3).map(|x| 1.0 / m_au[x / 3].sqrt()).collect();
    let hs = symmetrized(hessian, n3);
    let k = Mat::from_fn(n3, n3, |i, j| hs[(i, j)] * inv_sqrt_m[i] * inv_sqrt_m[j]);
    let (mom, axes, com) = thermo::principal_moments(masses, positions);
    let thr = thermo::linear_moment_au();
    let atom = nat == 1 || mom[2] < thr;
    let linear = !atom && mom[0] < thr;
    // rigid-body motions in mass-weighted coordinates
    let mut vecs: Vec<Vec<f64>> = Vec::new();
    for c in 0..3 {
        vecs.push((0..n3).map(|x| if x % 3 == c { m_au[x / 3].sqrt() } else { 0.0 }).collect());
    }
    for p in 0..3 {
        if mom[p] < thr {
            continue;
        }
        let a = [axes[0][p], axes[1][p], axes[2][p]];
        let mut v = vec![0.0; n3];
        for (i, r) in positions.iter().enumerate() {
            let d = [r[0] - com[0], r[1] - com[1], r[2] - com[2]];
            let cr = [a[1] * d[2] - a[2] * d[1], a[2] * d[0] - a[0] * d[2], a[0] * d[1] - a[1] * d[0]];
            for c in 0..3 {
                v[3 * i + c] = m_au[i].sqrt() * cr[c];
            }
        }
        vecs.push(v);
    }
    let d = orthonormalize(vecs, 1e-20);
    let nvib = n3 - d.len();
    // orthonormal basis of the complement: eigenvectors of P = 1 - D Dᵀ with eigenvalue 1
    let mut p = Mat::<f64>::identity(n3, n3);
    for v in &d {
        for i in 0..n3 {
            for j in 0..n3 {
                p[(i, j)] -= v[i] * v[j];
            }
        }
    }
    let (pw, pv) = eigh(&p)?;
    let cols: Vec<usize> = (0..n3).filter(|&i| pw[i] > 0.5).collect();
    if cols.len() != nvib {
        return Err(Error::Numerical(format!(
            "rigid-body projector has {} vibrational directions, expected {nvib}",
            cols.len()
        )));
    }
    let b = Mat::from_fn(n3, nvib, |i, j| pv[(i, cols[j])]);
    let kb = mul(k.as_ref(), b.as_ref());
    let kint = mul(b.transpose(), kb.as_ref());
    let kint = Mat::from_fn(nvib, nvib, |i, j| 0.5 * (kint[(i, j)] + kint[(j, i)]));
    let (w, y) = eigh(&kint)?;
    let l = mul(b.as_ref(), y.as_ref());
    Ok(Modes { linear, eigenvalues: w, l, inv_sqrt_m, to_wavenumber: units::HARTREE_TO_WAVENUMBER })
}

/// Linear molecule as xtb's `axis` decides it (smallest principal moment below 3e-4 u Å²,
/// with xtb's single-precision bohr and tensor offsets).
fn linear_xtb(masses: &[f64], positions: &[[f64; 3]]) -> bool {
    let coord: Vec<[f64; 3]> = positions.iter().map(|p| p.map(|x| x * xtbc::AXIS_BOHR)).collect();
    let mut sumw = 1e-20;
    let mut sw = [0.0; 3];
    for (m, c) in masses.iter().zip(&coord) {
        sumw += m;
        for k in 0..3 {
            sw[k] += m * c[k];
        }
    }
    let sw = sw.map(|s| s / sumw);
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
    let a = [[t[0], t[1], t[3]], [t[1], t[2], t[4]], [t[3], t[4], t[5]]];
    let (eig, _) = symmetry::eigh3(&a);
    // cc = const2 / eig(1), zero when eig(1) < 3e-4
    let cc = if eig[0] < 3e-4 { 0.0 } else { 2.9979245e4 * xtbc::CONST2 / eig[0] / 2.9979245e4 };
    cc < 1e-10
}

/// xtb's `trproj`: `P H P` with `P = 1 - F Fᵀ` and F the translations and rotations about the
/// geometric center (unit masses), orthonormalized by Gram-Schmidt.
fn trproj(positions: &[[f64; 3]], hs: &Mat<f64>) -> Mat<f64> {
    let nat = positions.len();
    let n3 = 3 * nat;
    let mut c = [0.0; 3];
    for p in positions {
        for k in 0..3 {
            c[k] += p[k];
        }
    }
    let c = c.map(|x| x / nat as f64);
    let mut f: Vec<Vec<f64>> = vec![vec![0.0; n3]; 6];
    for (i, p) in positions.iter().enumerate() {
        let (x, y, z) = (p[0] - c[0], p[1] - c[1], p[2] - c[2]);
        for k in 0..3 {
            f[k][3 * i + k] = 1.0;
        }
        f[3][3 * i + 1] = -z;
        f[3][3 * i + 2] = y;
        f[4][3 * i] = z;
        f[4][3 * i + 2] = -x;
        f[5][3 * i] = -y;
        f[5][3 * i + 1] = x;
    }
    let f = orthonormalize(f, f64::EPSILON);
    let fm = Mat::from_fn(n3, f.len(), |i, j| f[j][i]);
    let p = Mat::<f64>::identity(n3, n3) - mul(fm.as_ref(), fm.transpose());
    mul(mul(p.as_ref(), hs.as_ref()).as_ref(), p.as_ref())
}

/// xtb 6.8.0's `is_linear` (`axis_trafo.f90`): the unit-mass moments of inertia about the
/// centroid (`eigval3x3`), linear when one is below 1e-4 bohr²; a single atom counts as
/// linear.
pub(crate) fn is_linear_xtb680(positions: &[[f64; 3]]) -> bool {
    let n = positions.len();
    if n == 1 {
        return true;
    }
    let mut c = [0.0; 3];
    for p in positions {
        for k in 0..3 {
            c[k] += p[k];
        }
    }
    let c = c.map(|x| x / n as f64);
    let mut a = [[0.0; 3]; 3];
    for p in positions {
        let v = [p[0] - c[0], p[1] - c[1], p[2] - c[2]];
        let vv = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        for k in 0..3 {
            a[k][k] += vv;
        }
        a[0][0] -= v[0] * v[0];
        a[1][1] -= v[1] * v[1];
        a[2][2] -= v[2] * v[2];
        a[0][1] -= v[0] * v[1];
        a[1][2] -= v[1] * v[2];
        a[0][2] -= v[0] * v[2];
    }
    eigval3x3(&a).iter().any(|&m| m < 1.0e-4)
}

/// Eigenvalues of a symmetric 3 x 3 matrix (upper triangle) by xtb's trigonometric formula
/// (`xtb_mctc_math::eigval3x3`).
fn eigval3x3(a: &[[f64; 3]; 3]) -> [f64; 3] {
    let twothirdpi = 2.0 * xtbc::PI / 3.0;
    let r = a[0][1] * a[0][1] + a[0][2] * a[0][2] + a[1][2] * a[1][2];
    let q = (a[0][0] + a[1][1] + a[2][2]) / 3.0;
    let w = [a[0][0] - q, a[1][1] - q, a[2][2] - q];
    let p = ((w[0] * w[0] + w[1] * w[1] + w[2] * w[2] + 2.0 * r) / 6.0).sqrt();
    let mut r = (w[0] * (w[1] * w[2] - a[1][2] * a[1][2]) - a[0][1] * (a[0][1] * w[2] - a[1][2] * a[0][2])
        + a[0][2] * (a[0][1] * a[1][2] - w[1] * a[0][2]))
        / (p * p * p)
        * 0.5;
    if r <= -1.0 {
        r = 0.5 * twothirdpi;
    } else if r >= 1.0 {
        r = 0.0;
    } else {
        r = r.acos() / 3.0;
    }
    let w3 = q + 2.0 * p * r.cos();
    let w1 = q + 2.0 * p * (r + twothirdpi).cos();
    let w2 = 3.0 * q - w1 - w3;
    [w1, w2, w3]
}

/// Indices of the vibrations of xtb's frequency analysis, ascending: all but the `nproj`
/// rigid-body modes, which are the first modes below 0.01 cm⁻¹ (`first_below`) or the
/// `nproj` lowest; if fewer than `nproj` are below 0.01 cm⁻¹ (xtb 6.8.0 stops there, 6.7.1
/// leaves it undefined), the rest are the smallest in magnitude.
fn vibration_indices(freq: &[f64], nproj: usize, first_below: bool) -> Vec<usize> {
    let n3 = freq.len();
    let mut zero: Vec<usize> = if first_below {
        (0..n3).filter(|&i| freq[i].abs() < 0.01).take(nproj).collect()
    } else {
        (0..nproj.min(n3)).collect()
    };
    if zero.len() < nproj {
        let mut rest: Vec<usize> = (0..n3).filter(|i| !zero.contains(i)).collect();
        rest.sort_by(|&a, &b| freq[a].abs().total_cmp(&freq[b].abs()));
        zero.extend(rest.into_iter().take(nproj - zero.len()));
    }
    (0..n3).filter(|i| !zero.contains(i)).collect()
}

/// xtb 6.8.0's projector `1 - N Nᵀ` (`projectHessian`/`getNullSpace`) in mass-weighted
/// coordinates for masses in u: three translations and three rotations about the principal
/// axes through the center of mass (for a molecule that [`is_linear_xtb680`] not the axis of
/// the smallest moment; axes with a vanishing moment skipped), mass weighted and
/// orthonormalized, linearly dependent vectors dropped. Returns the projector and the number
/// of modes it removes.
fn projector_xtb680(positions: &[[f64; 3]], masses: &[f64]) -> (Mat<f64>, usize) {
    let nat = positions.len();
    let n3 = 3 * nat;
    let linear = is_linear_xtb680(positions);
    let mut vecs: Vec<Vec<f64>> = Vec::new();
    for c in 0..3 {
        vecs.push((0..n3).map(|x| if x % 3 == c { 1.0 } else { 0.0 }).collect());
    }
    let (mom, axes, com) = thermo::principal_moments(masses, positions);
    for p in 0..3 {
        let skip = if linear { p == 0 } else { mom[p] < f64::EPSILON };
        if skip {
            continue;
        }
        let a = [axes[0][p], axes[1][p], axes[2][p]];
        let mut v = vec![0.0; n3];
        for (i, r) in positions.iter().enumerate() {
            let d = [r[0] - com[0], r[1] - com[1], r[2] - com[2]];
            v[3 * i] = a[1] * d[2] - a[2] * d[1];
            v[3 * i + 1] = a[2] * d[0] - a[0] * d[2];
            v[3 * i + 2] = a[0] * d[1] - a[1] * d[0];
        }
        vecs.push(v);
    }
    // xtb weights with its masses in atomic units (`mol%atmass`)
    for v in &mut vecs {
        for x in 0..n3 {
            v[x] *= (masses[x / 3] * xtbc::AMUTOAU).sqrt();
        }
    }
    let d = orthonormalize(vecs, f64::EPSILON);
    let dm = Mat::from_fn(n3, d.len(), |i, j| d[j][i]);
    (Mat::<f64>::identity(n3, n3) - mul(dm.as_ref(), dm.transpose()), d.len())
}

/// `xtb` conventions (xtb 6.8.0, `hessian.F90`): the Hessian mass weighted, projected with
/// [`projector_xtb680`] and diagonalized whole; its first modes below 0.01 cm⁻¹, as many as
/// were projected, are the rigid-body modes.
fn modes_xtb680(positions: &[[f64; 3]], masses: &[f64], hessian: &[f64]) -> Result<Modes> {
    let nat = positions.len();
    let n3 = 3 * nat;
    let linear = nat > 1 && is_linear_xtb680(positions);
    let hs = symmetrized(hessian, n3);
    let inv_sqrt_m: Vec<f64> = (0..n3).map(|x| (1.0 / masses[x / 3].sqrt()) / xtbc::AMUTOAU.sqrt()).collect();
    if nat == 1 {
        return Ok(Modes {
            linear,
            eigenvalues: Vec::new(),
            l: Mat::zeros(n3, 0),
            inv_sqrt_m,
            to_wavenumber: xtbc::AUTORCM,
        });
    }
    let k = Mat::from_fn(n3, n3, |i, j| hs[(i, j)] * inv_sqrt_m[i] * inv_sqrt_m[j]);
    let (p, nproj) = projector_xtb680(positions, masses);
    let k = mul(mul(p.as_ref(), k.as_ref()).as_ref(), p.as_ref());
    let k = Mat::from_fn(n3, n3, |i, j| 0.5 * (k[(i, j)] + k[(j, i)]));
    let (w, v) = eigh(&k)?;
    let freq: Vec<f64> = w.iter().map(|x| xtbc::AUTORCM * x.abs().sqrt().copysign(*x)).collect();
    let keep = vibration_indices(&freq, nproj, true);
    let l = Mat::from_fn(n3, keep.len(), |i, j| v[(i, keep[j])]);
    Ok(Modes { linear, eigenvalues: keep.iter().map(|&i| w[i]).collect(), l, inv_sqrt_m, to_wavenumber: xtbc::AUTORCM })
}

/// `xtb-6.7.1` conventions: xtb 6.7.1's `numhess` after the Hessian.
fn modes_xtb671(positions: &[[f64; 3]], masses: &[f64], hessian: &[f64]) -> Result<Modes> {
    let nat = positions.len();
    let n3 = 3 * nat;
    let linear = nat > 1 && linear_xtb(masses, positions);
    let mut hs = symmetrized(hessian, n3);
    if !linear && nat > 1 {
        hs = trproj(positions, &hs);
    }
    let inv_sqrt_m: Vec<f64> = (0..n3).map(|x| (1.0 / masses[x / 3].sqrt()) / xtbc::AMUTOAU.sqrt()).collect();
    let k = Mat::from_fn(n3, n3, |i, j| hs[(i, j)] * inv_sqrt_m[i] * inv_sqrt_m[j]);
    let (w, v) = eigh(&k)?;
    let freq: Vec<f64> = w.iter().map(|x| xtbc::AUTORCM * x.abs().sqrt().copysign(*x)).collect();
    // rigid-body modes: the five lowest (linear) or the first six below 0.01 cm⁻¹
    let nzero = if nat == 1 {
        3
    } else if linear {
        5
    } else {
        6
    }
    .min(n3);
    let keep = vibration_indices(&freq, nzero, !(linear || nat == 1));
    let l = Mat::from_fn(n3, keep.len(), |i, j| v[(i, keep[j])]);
    Ok(Modes { linear, eigenvalues: keep.iter().map(|&i| w[i]).collect(), l, inv_sqrt_m, to_wavenumber: xtbc::AUTORCM })
}

/// Vibrational analysis of `hessian` (row-major 3N x 3N, Eh/bohr²) for atoms `numbers` at
/// `positions` (bohr). `dipole_derivatives` (3N rows ∂μ/∂R_x, e) give IR intensities.
pub fn analyze(
    numbers: &[usize],
    positions: &[[f64; 3]],
    hessian: &[f64],
    dipole_derivatives: Option<&[[f64; 3]]>,
    opts: &VibOptions,
) -> Result<VibAnalysis> {
    let nat = numbers.len();
    let n3 = 3 * nat;
    if nat == 0 || positions.len() != nat {
        return Err(Error::Input(format!("{} positions for {nat} atoms", positions.len())));
    }
    if hessian.len() != n3 * n3 {
        return Err(Error::Input(format!("Hessian has {} elements, expected {n3} x {n3}", hessian.len())));
    }
    if hessian.iter().any(|x| !x.is_finite()) || positions.iter().flatten().any(|x| !x.is_finite()) {
        return Err(Error::Input("Hessian and positions must be finite".into()));
    }
    if let Some(d) = dipole_derivatives {
        if d.len() != n3 {
            return Err(Error::Input(format!("{} dipole derivatives, expected {n3} (one per coordinate)", d.len())));
        }
    }
    let params = opts.conventions.params();
    let masses = match &opts.masses {
        Some(m) => {
            if m.len() != nat || m.iter().any(|x| !(x.is_finite() && *x > 0.0)) {
                return Err(Error::Input(format!("masses: need {nat} positive values")));
            }
            m.clone()
        }
        None => masses::masses(numbers, params.masses)?,
    };
    let modes = match params.projection {
        Projection::Complement => modes_axtb(positions, &masses, hessian)?,
        Projection::MassWeighted => modes_xtb680(positions, &masses, hessian)?,
        Projection::Cartesian => modes_xtb671(positions, &masses, hessian)?,
    };
    let nvib = modes.eigenvalues.len();
    let frequencies: Vec<f64> =
        modes.eigenvalues.iter().map(|x| modes.to_wavenumber * x.abs().sqrt().copysign(*x)).collect();
    let mut normal_modes = Vec::with_capacity(nvib);
    let mut reduced_masses = Vec::with_capacity(nvib);
    let mut force_constants = Vec::with_capacity(nvib);
    let ir_factor = params.ir_factor;
    let mut ir = dipole_derivatives.map(|_| Vec::with_capacity(nvib));
    for i in 0..nvib {
        let cart: Vec<f64> = (0..n3).map(|x| modes.l[(x, i)] * modes.inv_sqrt_m[x]).collect();
        let norm = cart.iter().map(|x| x * x).sum::<f64>().sqrt();
        normal_modes
            .push((0..nat).map(|a| [cart[3 * a] / norm, cart[3 * a + 1] / norm, cart[3 * a + 2] / norm]).collect());
        let ln: f64 = (0..n3).map(|x| modes.l[(x, i)] * modes.l[(x, i)]).sum();
        let mu = ln / (0..n3).map(|x| modes.l[(x, i)] * modes.l[(x, i)] / masses[x / 3]).sum::<f64>();
        reduced_masses.push(mu);
        // k = μ ω² in mdyn/Å: ω² (Eh/(bohr² mₑ)) μ (mₑ) -> Eh/bohr² -> mdyn/Å
        let k_au = modes.eigenvalues[i] * mu / units::ELECTRON_MASS_IN_U;
        // 1 mdyn/Å = 100 N/m
        force_constants.push(k_au * units::HARTREE_TO_JOULE / (units::AUTOAA * 1e-10).powi(2) / 100.0);
        if let (Some(d), Some(ir)) = (dipole_derivatives, ir.as_mut()) {
            let mut t = [0.0; 3];
            for (x, dx) in d.iter().enumerate() {
                for k in 0..3 {
                    t[k] += dx[k] * cart[x];
                }
            }
            ir.push(ir_factor * (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]));
        }
    }
    let n_imaginary = frequencies.iter().filter(|f| **f < 0.0).count();
    let thermo = match &opts.thermo {
        Some(t) => Some(thermochemistry(numbers, positions, &masses, &frequencies, opts.conventions, t)?),
        None => None,
    };
    Ok(VibAnalysis {
        conventions: opts.conventions,
        masses,
        linear: modes.linear,
        frequencies,
        normal_modes,
        reduced_masses,
        force_constants,
        ir_intensities: ir,
        n_imaginary,
        thermo,
    })
}

/// Project translations and rotations out of a Cartesian Hessian as the xtb program of
/// `conventions` does before writing its `hessian` file: xtb 6.8.0 (`Xtb`) in mass-weighted
/// coordinates with xtb's masses and transformed back; xtb 6.7.1
/// (`Xtb671`) with `trproj` (rotations about the geometric center, unit masses; nothing for
/// linear molecules). `xtb thermo` does not project a Turbomole Hessian, so a Hessian written
/// for it should be projected with this first.
pub fn project_xtb(
    numbers: &[usize],
    positions: &[[f64; 3]],
    hessian: &[f64],
    conventions: Conventions,
) -> Result<Vec<f64>> {
    let nat = numbers.len();
    let n3 = 3 * nat;
    if positions.len() != nat || hessian.len() != n3 * n3 {
        return Err(Error::Input("project_xtb: inconsistent dimensions".into()));
    }
    let masses = masses::masses(numbers, &masses::XTB)?;
    let mut hs = symmetrized(hessian, n3);
    match conventions.params().projection {
        Projection::Cartesian => {
            if nat > 1 && !linear_xtb(&masses, positions) {
                hs = trproj(positions, &hs);
            }
        }
        Projection::MassWeighted => {
            // xtb's amass_au = 1/sqrt(m amutoau): weight, project, transform back
            let a: Vec<f64> = (0..n3).map(|x| (1.0 / masses[x / 3].sqrt()) / xtbc::AMUTOAU.sqrt()).collect();
            let k = Mat::from_fn(n3, n3, |i, j| hs[(i, j)] * a[i] * a[j]);
            let (p, _) = projector_xtb680(positions, &masses);
            let k = mul(mul(p.as_ref(), k.as_ref()).as_ref(), p.as_ref());
            hs = Mat::from_fn(n3, n3, |i, j| 0.5 * (k[(i, j)] + k[(j, i)]) / (a[i] * a[j]));
        }
        Projection::Complement => {
            return Err(Error::Input("project_xtb: give the conventions of an xtb program".into()))
        }
    }
    Ok((0..n3 * n3).map(|e| hs[(e / n3, e % n3)]).collect())
}

#[cfg(test)]
mod tests;
