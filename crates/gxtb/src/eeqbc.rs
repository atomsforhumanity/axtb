//! Bond-capacitor electronegativity equilibration (EEQ-BC) charges of g-xTB: one model
//! ([`EeqBc`]) for both programs, each with its form ([`EeqBcForm`]) and element parameters.
//!
//! Model: T. Froitzheim, M. Müller, A. Hansen, S. Grimme, J. Chem. Phys. 162, 214109 (2025).
//! The g-xTB 1.1.0 program (`eeq_bc_`) uses the same model and the same parameters as
//! multicharge's `eeqbc2025` (verified column by column, and its global constants
//! kcn = 2, norm_exp = 0.75, kbc = 0.6, kcnrad = 0.14 appear as literals in the binary);
//! the charges below reproduce the program's to ~1e-15 (see the reference tests).
//!
//! With r_AB in bohr (v1.1, [`EeqBcForm::V1_1`] and the parsed `eeq` file):
//! - CN_A = sum_B erfc-count(r_AB), count = 1/2 [1 + erf(-k_cn (r - R_AB) / R_AB^0.75)],
//!   R_AB = (rcov2_A + rcov2_B) / 2;
//! - local charge q^loc_A = sum_B (EN_B - EN_A)/3.98 * count(r_AB) + Q/N;
//! - capacitance c_AB = sqrt(cap_A cap_B) 1/2 [1 + erf(-k_bc (r_AB - R^vdW_AB) / R^vdW_AB)] with the
//!   D3 pair radii R^vdW_AB in **Ångström** (as in multicharge, which converts them back);
//! - Maxwell matrix C: C_AB = -c_AB, C_AA = sum_B c_AB, C_{N+1,N+1} = 1;
//! - x_A = -chi_A + kcnchi_A CN_A + kqchi_A q^loc_A, x_{N+1} = Q; right-hand side b = C x;
//! - charge width w_A = rad_A (1 - k_cnrad CN_A / CNavg_A^0.75);
//! - A_AB = erf(r_AB / sqrt(w_A^2 + w_B^2)) / r_AB * C_AB (A != B),
//!   A_AA = (eta_A + kqeta_A q^loc_A + sqrt(2/pi)/w_A) C_AA + 1, Lagrange row/column of ones;
//! - (q, lambda) = A^-1 b.
//!
//! g-xTB 2.0.1 (multicharge `eeqbc2025` as built into the 2.0.1 program, [`EeqBcForm::of`] and
//! the `EEQBC_*` tables) refits the parameters and changes the form in four places (spec §4.1):
//! - no local-charge hardness term: A_AA = (eta_A + sqrt(2/pi)/w_A) C_AA + 1;
//! - R_AB = r_A + r_B (`eeqbc_cov_radii`) in CN_A = sum_B 1/2 [1 + erf(-2 (R - R_AB)/R_AB^0.75)],
//!   with the 25-bohr cutoff of the program's CNs ([`crate::model::Model::cut`]);
//! - w_A = rad_A (1 - 0.3 CN_A / CNavg_A^0.75), k_bc = 0.75;
//! - c_AB = sqrt(cap_A cap_B) 1/2 [1 + erf(-0.75 (R - Rv_AB)/Rv_AB)] with
//!   Rv_AB = R^D3_AB (bohr) / (2 aatoau) (s_A + s_B), s = `eeqbc_rvdw_scale`.
//!
//! [`eeqbc`] builds the model of a molecule for the version of a [`Model`], [`eeqbc_charges`] and
//! [`eeqbc_charge_gradient`] are its charges and their adjoint gradient.
//!
//! The bordered linear system is [`axtb_base::eeq`]'s (one total-charge constraint) with the
//! capacitance-scaled matrix assembled here. [`EeqBc::solve`] factorizes it once per geometry
//! and keeps it ([`EeqBcSolution`], owned, so that a setup can store it): the charges, and any
//! number of adjoint gradients ([`EeqBc::add_charge_gradient`]) from the same factorization.
//! The setup's factorization serves the gradients at its geometry
//! ([`EeqBc::charges_and_solution`]).
//!
//! The Coulomb kernel of two Gaussian charges, erf(r/sigma)/r with sigma² = w_A² + w_B², is
//! evaluated as multicharge does, erf(sqrt(r r gamma²))/r with gamma² = 1/sigma², for the charges
//! and the gradients of both versions (the 1.1.0 program's gradient writes erf(r/sigma)/r: the
//! same matrix to rounding).

use crate::cn::CN_CUTOFF;
use crate::data::v1_1::{CN_AVG, PAULING_EN, VDW_RAD_PACKED};
use crate::data::v2_0_1::{
    EEQBC_AVG_CN, EEQBC_CAP, EEQBC_CHI, EEQBC_COV_RADII, EEQBC_ETA, EEQBC_KCNCHI, EEQBC_KQCHI, EEQBC_RAD,
    EEQBC_RVDW_SCALE,
};
use crate::model::{Model, SQRTPI};
use crate::params::EeqBcParams;
use axtb::molecule::{norm2_sq, sub3};
use axtb::{Error, Molecule, Result};
use axtb_base::eeq::{ChargeEquations, ChargeSolver, Constraints};
use faer::Par;

/// libm's error function (gfortran's `ERF`), the workspace's one wrapper.
pub(crate) use axtb_d4::erf as erf_f64;

/// Steepness of the EEQ-BC counting function.
pub const KCN: f64 = 2.0;
/// Exponent of the distance (and average-CN) normalization.
pub const NORM_EXP: f64 = 0.75;
/// Electronegativity normalization (fluorine).
pub const EN_NORM: f64 = 3.98;

/// D3 pairwise van der Waals radius (Ångström) of atomic numbers `za`, `zb` (1-based).
pub fn vdw_rad_pair(za: usize, zb: usize) -> f64 {
    let (hi, lo) = if za >= zb { (za, zb) } else { (zb, za) };
    VDW_RAD_PACKED[hi * (hi - 1) / 2 + lo - 1]
}

/// The pair radius R_AB of the counting function from the element radii r_A, r_B.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RcRule {
    /// R_AB = 0.5 (r_A + r_B) (v1.1: the `eeq` file tabulates twice the radius).
    Mean,
    /// R_AB = r_A + r_B (2.0.1).
    Sum,
}

/// The van der Waals radius R^vdW_AB of the bond capacitance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum RvdwRule {
    /// The D3 pair radius in Ångström, used as bohr ([`vdw_rad_pair`]; v1.1).
    D3,
    /// The D3 pair radius in bohr times (s_A + s_B) / (2 aatoau), with the element scales s
    /// (2.0.1, in the conventions of the model).
    ScaledD3(Model),
}

/// The form of an EEQ-BC model: what differs between the programs besides the element
/// parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EeqBcForm {
    /// Steepness of the bond capacitance switch.
    pub kbc: f64,
    /// CN scaling of the charge width.
    pub kcnrad: f64,
    /// Pair radius of the counting function.
    pub rc: RcRule,
    /// Local-charge hardness term kqeta_A q^loc_A on the diagonal (v1.1); without it the
    /// diagonal is eta_A + sqrt(2/pi)/w_A as 2.0.1 writes it.
    pub kqeta: bool,
    /// Van der Waals radius of the capacitance.
    pub rvdw: RvdwRule,
    /// The 2.0.1 program's CN cutoff ([`Model::cut`] at [`CN_CUTOFF`]) on the counting
    /// function; pairs beyond it (zero count derivative) are skipped in the gradient. None: no
    /// cutoff (v1.1).
    pub cutoff: Option<Model>,
}

impl EeqBcForm {
    /// g-xTB v1.1 (multicharge's `eeqbc2025`).
    pub const V1_1: EeqBcForm =
        EeqBcForm { kbc: 0.6, kcnrad: 0.14, rc: RcRule::Mean, kqeta: true, rvdw: RvdwRule::D3, cutoff: None };
}

/// The EEQ-BC parameters of one atom (by its element).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct EeqBcAtom {
    /// Electronegativity.
    pub chi: f64,
    /// Chemical hardness.
    pub eta: f64,
    /// Gaussian charge width.
    pub rad: f64,
    /// CN scaling of chi.
    pub kcnchi: f64,
    /// Local-charge scaling of chi.
    pub kqchi: f64,
    /// Local-charge scaling of eta (used with [`EeqBcForm::kqeta`]).
    pub kqeta: f64,
    /// Bond capacitance.
    pub cap: f64,
    /// Radius of the counting function ([`RcRule`]).
    pub rcov: f64,
    /// Average coordination number of the charge width.
    pub cn_avg: f64,
    /// Scale of the van der Waals radius ([`RvdwRule::ScaledD3`]).
    pub rvdw_scale: f64,
}

/// EEQ-BC results.
#[derive(Clone, Debug)]
pub struct EeqBcCharges {
    /// Coordination numbers (printed as "CN" by the program).
    pub cn: Vec<f64>,
    /// Electronegativity-weighted local charges (printed as "q_CN").
    pub qloc: Vec<f64>,
    /// Partial charges (printed as "q").
    pub charges: Vec<f64>,
}

/// The EEQ-BC model of one molecule: its form and the parameters of its atoms.
pub struct EeqBc<'m> {
    form: EeqBcForm,
    mol: &'m Molecule,
    atoms: Vec<EeqBcAtom>,
}

/// Geometry-dependent quantities of the charge equations.
struct Geometry {
    dist: Vec<f64>,
    cn: Vec<f64>,
    qloc: Vec<f64>,
    /// Maxwell capacitance matrix, (n+1)², by rows.
    cmat: Vec<f64>,
    x: Vec<f64>,
    width: Vec<f64>,
    hdiag: Vec<f64>,
}

/// One counting term 1/2 [1 + erf(-k_cn (r - rc) / rc^0.75)] and its derivative with respect
/// to r.
#[inline]
fn erf_count(r: f64, rc: f64) -> (f64, f64) {
    let den = rc.powf(NORM_EXP);
    let arg = -KCN * (r - rc) / den;
    (0.5 * (1.0 + erf_f64(arg)), -KCN / den * (-arg * arg).exp() / SQRTPI)
}

impl<'m> EeqBc<'m> {
    /// The model of `form` for `mol`, with the parameters `atom(Z)` of each element.
    pub fn new(form: EeqBcForm, mol: &'m Molecule, atom: impl Fn(usize) -> EeqBcAtom) -> Self {
        let atoms = mol.numbers.iter().map(|&z| atom(z)).collect();
        EeqBc { form, mol, atoms }
    }

    /// g-xTB v1.1: [`EeqBcForm::V1_1`] with the parameters of the `eeq` file `param` and the
    /// program's average CNs.
    pub fn v1_1(param: &EeqBcParams, mol: &'m Molecule) -> Self {
        Self::new(EeqBcForm::V1_1, mol, |z| {
            let e = &param.elements[z];
            EeqBcAtom {
                chi: e.chi,
                eta: e.eta,
                rad: e.rad,
                kcnchi: e.kcnchi,
                kqchi: e.kqchi,
                kqeta: e.kqeta,
                cap: e.cap,
                rcov: e.rcov2,
                cn_avg: CN_AVG[z - 1],
                rvdw_scale: 0.0,
            }
        })
    }

    fn n(&self) -> usize {
        self.mol.nat()
    }

    /// (EN_j - EN_i) / 3.98: the local-charge weight of the pair (i, j).
    #[inline]
    fn en_weight(&self, i: usize, j: usize) -> f64 {
        let z = &self.mol.numbers;
        (PAULING_EN[z[j] - 1] - PAULING_EN[z[i] - 1]) / EN_NORM
    }

    /// Count of the CN and the local charges of the pair (i, j) at distance `r`, with the
    /// cutoff factor of the form, and its r derivative.
    #[inline]
    fn count(&self, i: usize, j: usize, r: f64) -> (f64, f64) {
        let (a, b) = (&self.atoms[i], &self.atoms[j]);
        let rc = match self.form.rc {
            RcRule::Mean => 0.5 * (a.rcov + b.rcov),
            RcRule::Sum => a.rcov + b.rcov,
        };
        match self.form.cutoff {
            None => erf_count(r, rc),
            Some(model) => {
                if r > model.reach(CN_CUTOFF) {
                    return (0.0, 0.0);
                }
                let (c, dc) = erf_count(r, rc);
                let (sw, dsw) = model.cut(r, CN_CUTOFF);
                (c * sw, dc * sw + c * dsw)
            }
        }
    }

    /// Bond capacitance c_ij at distance `r` and its r derivative.
    #[inline]
    fn capacitance(&self, i: usize, j: usize, r: f64) -> (f64, f64) {
        let (a, b) = (&self.atoms[i], &self.atoms[j]);
        let (za, zb) = (self.mol.numbers[i], self.mol.numbers[j]);
        let rv = match self.form.rvdw {
            RvdwRule::D3 => vdw_rad_pair(za, zb),
            RvdwRule::ScaledD3(model) => model.vdw_pair(za, zb) * (0.5 / model.aatoau) * (a.rvdw_scale + b.rvdw_scale),
        };
        let pre = (a.cap * b.cap).sqrt();
        let kbc = self.form.kbc;
        let arg = -kbc * (r - rv) / rv;
        (pre * 0.5 * (1.0 + erf_f64(arg)), pre * (-kbc / rv) * (-arg * arg).exp() / SQRTPI)
    }

    fn geometry(&self) -> Geometry {
        let mol = self.mol;
        let (n, m) = (self.n(), self.n() + 1);
        let p = &self.atoms;
        let mut cn = vec![0.0; n];
        let mut qloc = vec![mol.charge / n as f64; n];
        let mut dist = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..i {
                let r = norm2_sq(sub3(mol.positions[i], mol.positions[j])).sqrt();
                dist[i * n + j] = r;
                dist[j * n + i] = r;
                let (c, _) = self.count(i, j, r);
                cn[i] += c;
                cn[j] += c;
                let den = self.en_weight(i, j);
                qloc[i] += den * c;
                qloc[j] -= den * c;
            }
        }
        // Maxwell capacitance matrix
        let mut cmat = vec![0.0; m * m];
        for i in 0..n {
            for j in 0..i {
                let (c, _) = self.capacitance(i, j, dist[i * n + j]);
                cmat[i * m + j] = -c;
                cmat[j * m + i] = -c;
                cmat[i * m + i] += c;
                cmat[j * m + j] += c;
            }
        }
        cmat[n * m + n] = 1.0;
        let mut x = vec![0.0; m];
        for i in 0..n {
            x[i] = -p[i].chi + p[i].kcnchi * cn[i] + p[i].kqchi * qloc[i];
        }
        x[n] = mol.charge;
        let kcnrad = self.form.kcnrad;
        let width: Vec<f64> = (0..n).map(|i| p[i].rad * (1.0 - kcnrad * cn[i] / p[i].cn_avg.powf(NORM_EXP))).collect();
        let sqrt2pi = (2.0 / std::f64::consts::PI).sqrt();
        let hdiag: Vec<f64> = if self.form.kqeta {
            (0..n).map(|i| p[i].eta + p[i].kqeta * qloc[i] + sqrt2pi / width[i]).collect()
        } else {
            (0..n).map(|i| p[i].eta + sqrt2pi / width[i]).collect()
        };
        Geometry { dist, cn, qloc, cmat, x, width, hdiag }
    }

    /// The charge equations A y = b of `g`: A and b = C x.
    fn equations(&self, g: &Geometry) -> (ChargeEquations, Vec<f64>) {
        let (n, m) = (self.n(), self.n() + 1);
        let rhs: Vec<f64> = (0..m).map(|i| (0..m).map(|k| g.cmat[i * m + k] * g.x[k]).sum()).collect();
        let a = ChargeEquations::new(
            n,
            Constraints::Total,
            |i| g.hdiag[i] * g.cmat[i * m + i] + 1.0,
            |i, j| {
                let r = g.dist[i * n + j];
                let (wi, wj) = (g.width[i], g.width[j]);
                let gam2 = 1.0 / (wi * wi + wj * wj);
                erf_f64((r * r * gam2).sqrt()) / r * g.cmat[j * m + i]
            },
        );
        (a, rhs)
    }

    /// The charge equations at the geometry of the molecule, factorized once (sequential LU: faer's high-level `partial_piv_lu` would run
    /// on faer's global parallelism, i.e. start rayon's global pool in every process), and
    /// their solution. A singular matrix gives a solution that is not finite.
    pub fn solve(&self) -> EeqBcSolution {
        let g = self.geometry();
        let (a, rhs) = self.equations(&g);
        let solver = a.factorize(Par::Seq);
        let y = solver.solve(&rhs);
        EeqBcSolution { g, solver, y }
    }

    /// [`EeqBc::solve`], kept together with the model.
    pub fn system(&self) -> EeqBcSystem<'_, 'm> {
        EeqBcSystem { eeq: self, solution: self.solve() }
    }

    /// The EEQ-BC partial charges (positions in bohr; total charge `mol.charge`), with the
    /// coordination numbers and local charges.
    pub fn charges(&self) -> Result<EeqBcCharges> {
        Ok(self.charges_and_solution()?.0)
    }

    /// [`EeqBc::charges`] and the factorized equations they were solved from, for the
    /// gradients at this geometry.
    pub fn charges_and_solution(&self) -> Result<(EeqBcCharges, EeqBcSolution)> {
        let n = self.n();
        let s = self.solve();
        let charges = s.y[..n].to_vec();
        if charges.iter().any(|q| !q.is_finite()) {
            return Err(Error::Numerical("EEQ-BC: singular charge equations".into()));
        }
        Ok((EeqBcCharges { cn: s.g.cn.clone(), qloc: s.g.qloc.clone(), charges }, s))
    }

    /// Gradient of a function of the EEQ-BC charges: with `dedq[A]` = dE/dq_A, adds
    /// sum_A dE/dq_A dq_A/dR to `grad` (positions in bohr), with the factorized equations
    /// `solution` of this model at this geometry. Adjoint of the linear charge equations: dE/dR = w^T (db/dR -
    /// dA/dR y) with A w = (dE/dq, 0) (A is symmetric), one pair of triangular solves with the
    /// kept factorization. R enters through the distances (Coulomb kernel, bond
    /// capacitances), the coordination numbers (charge widths, electronegativities) and the
    /// local charges.
    pub fn add_charge_gradient(&self, solution: &EeqBcSolution, dedq: &[f64], grad: &mut [[f64; 3]]) {
        assert_eq!(solution.y.len(), self.n() + 1, "EEQ-BC gradient from equations of another molecule");
        let e = self;
        let (g, y) = (&solution.g, &solution.y);
        let mol = e.mol;
        let p = &e.atoms;
        let (n, m) = (e.n(), e.n() + 1);
        let kcnrad = e.form.kcnrad;
        let w = solution.solver.adjoint(&dedq[..n]);
        let cmat = &g.cmat;
        // C^T w (C symmetric)
        let ctw: Vec<f64> = (0..m).map(|i| (0..m).map(|k| cmat[k * m + i] * w[k]).sum()).collect();
        let dwidth: Vec<f64> = (0..n).map(|i| -p[i].rad * kcnrad / p[i].cn_avg.powf(NORM_EXP)).collect();
        let sqrt2pi = (2.0 / std::f64::consts::PI).sqrt();
        // dE/dCN and dE/dqloc per atom; dE/dr per pair
        let mut decn = vec![0.0; n];
        let mut deql = vec![0.0; n];
        for i in 0..n {
            // x_i: T1 = w^T C dx
            decn[i] += ctw[i] * p[i].kcnchi;
            deql[i] += ctw[i] * p[i].kqchi;
            // A_ii = h_i C_ii + 1: -w_i y_i C_ii dh_i
            if e.form.kqeta {
                deql[i] -= w[i] * y[i] * cmat[i * m + i] * p[i].kqeta;
            }
            decn[i] -= w[i] * y[i] * cmat[i * m + i] * (-sqrt2pi / (g.width[i] * g.width[i])) * dwidth[i];
        }
        let gfun = |i: usize, j: usize| -> (f64, f64, f64) {
            // gamma = erf(r/sigma)/r and its derivatives with respect to r and sigma
            let r = g.dist[i * n + j];
            let sigma = (g.width[i] * g.width[i] + g.width[j] * g.width[j]).sqrt();
            let t = r / sigma;
            let ef = erf_f64(t);
            let gauss = 2.0 / SQRTPI * (-t * t).exp();
            (ef / r, gauss / (sigma * r) - ef / (r * r), -gauss / (sigma * sigma))
        };
        for i in 0..n {
            for j in 0..i {
                let r = g.dist[i * n + j];
                let (gg, dgdr, dgds) = gfun(i, j);
                let (_, dcdr) = e.capacitance(i, j, r);
                let cij = cmat[i * m + j]; // = -c_ij
                let sym = w[i] * y[j] + w[j] * y[i];
                // widths: sigma = sqrt(w_i^2 + w_j^2)
                let sigma = (g.width[i] * g.width[i] + g.width[j] * g.width[j]).sqrt();
                decn[i] -= sym * cij * dgds * g.width[i] / sigma * dwidth[i];
                decn[j] -= sym * cij * dgds * g.width[j] / sigma * dwidth[j];
                // distance: T1 via C, T2 via gamma and C
                let t1 = dcdr * (w[i] - w[j]) * (g.x[i] - g.x[j]);
                let t2 =
                    sym * (dgdr * cij - gg * dcdr) + w[i] * g.hdiag[i] * dcdr * y[i] + w[j] * g.hdiag[j] * dcdr * y[j];
                let dedr = t1 - t2;
                let v = sub3(mol.positions[i], mol.positions[j]);
                for k in 0..3 {
                    grad[i][k] += dedr * v[k] / r;
                    grad[j][k] -= dedr * v[k] / r;
                }
            }
        }
        // chain through the counting function of CN and qloc
        let cutoff = e.form.cutoff.is_some();
        for i in 0..n {
            for j in 0..i {
                let r = g.dist[i * n + j];
                let (_, dcount) = e.count(i, j, r);
                if cutoff && dcount == 0.0 {
                    continue;
                }
                let den = e.en_weight(i, j);
                let dedr = (decn[i] + decn[j]) * dcount + (deql[i] - deql[j]) * den * dcount;
                let v = sub3(mol.positions[i], mol.positions[j]);
                for k in 0..3 {
                    grad[i][k] += dedr * v[k] / r;
                    grad[j][k] -= dedr * v[k] / r;
                }
            }
        }
    }
}

/// The EEQ-BC charge equations A(R) y = b(R) of one geometry, factorized, with their solution y = (q, lambda) ([`EeqBc::solve`]). It does not borrow
/// the molecule, so that a setup can keep it for the gradients at its geometry.
pub struct EeqBcSolution {
    g: Geometry,
    solver: ChargeSolver,
    y: Vec<f64>,
}

impl std::fmt::Debug for EeqBcSolution {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EeqBcSolution").field("atoms", &self.solver.atoms()).finish()
    }
}

/// An EEQ-BC model with the factorized equations of its geometry ([`EeqBc::system`]).
pub struct EeqBcSystem<'a, 'm> {
    eeq: &'a EeqBc<'m>,
    solution: EeqBcSolution,
}

impl EeqBcSystem<'_, '_> {
    /// [`EeqBc::add_charge_gradient`] with the kept factorization.
    pub fn add_charge_gradient(&self, dedq: &[f64], grad: &mut [[f64; 3]]) {
        self.eeq.add_charge_gradient(&self.solution, dedq, grad);
    }
}

// ---- the versions' models of a molecule ----

impl EeqBcForm {
    /// The form of `model`: v1.1's ([`EeqBcForm::V1_1`]) or 2.0.1's in the conventions of `model`
    /// (its Ångström-to-bohr factor and its CN cutoff).
    pub fn of(model: &Model) -> EeqBcForm {
        if model.version == crate::options::Version::V1_1 {
            return Self::V1_1;
        }
        EeqBcForm {
            kbc: 0.75,
            kcnrad: 0.3,
            rc: RcRule::Sum,
            kqeta: false,
            rvdw: RvdwRule::ScaledD3(*model),
            cutoff: Some(*model),
        }
    }
}

/// The EEQ-BC model of `mol` in `model` (v1.1: [`EeqBc::v1_1`]).
pub fn eeqbc<'m>(model: &Model, mol: &'m Molecule) -> Result<EeqBc<'m>> {
    crate::model::check_model_elements(model, &mol.numbers)?;
    if model.version == crate::options::Version::V1_1 {
        return Ok(EeqBc::v1_1(crate::params::eeqbc_v1_1(), mol));
    }
    Ok(EeqBc::new(EeqBcForm::of(model), mol, |z| {
        let k = z - 1;
        EeqBcAtom {
            chi: EEQBC_CHI[k],
            eta: EEQBC_ETA[k],
            rad: EEQBC_RAD[k],
            kcnchi: EEQBC_KCNCHI[k],
            kqchi: EEQBC_KQCHI[k],
            kqeta: 0.0,
            cap: EEQBC_CAP[k],
            rcov: EEQBC_COV_RADII[k],
            cn_avg: EEQBC_AVG_CN[k],
            rvdw_scale: EEQBC_RVDW_SCALE[k],
        }
    }))
}

/// EEQ-BC partial charges of `mol` (positions in bohr; total charge `mol.charge`).
pub fn eeqbc_charges(model: &Model, mol: &Molecule) -> Result<EeqBcCharges> {
    eeqbc(model, mol)?.charges()
}

/// Adds sum_A dE/dq_A dq_A/dR to `grad` for a function E of the EEQ-BC charges (adjoint of
/// the linear charge equations A y = b: dE/dR = w^T (db/dR - dA/dR y), A w = (dE/dq, 0)).
/// For several `dedq` at one geometry, factorize once: [`eeqbc`], [`EeqBc::system`]; at the
/// geometry of a [`crate::scf::Setup`], its gradients use the setup's factorization
/// ([`EeqBc::charges_and_solution`]).
pub fn eeqbc_charge_gradient(model: &Model, mol: &Molecule, dedq: &[f64], grad: &mut [[f64; 3]]) -> Result<()> {
    eeqbc(model, mol)?.system().add_charge_gradient(dedq, grad);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The error functions' sqrt(pi) of both versions is the 2.0.1 program's literal, one ulp
    /// above `PI.sqrt()`.
    #[test]
    fn sqrtpi_is_the_programs_literal() {
        assert_eq!(SQRTPI.to_bits(), std::f64::consts::PI.sqrt().to_bits() + 1);
    }
}
