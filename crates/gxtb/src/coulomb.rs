//! Charge- and multipole-dependent electrostatics and spin polarization of g-xTB 2.0.1
//! (spec §4.7-§4.12). Shell charges q_l = n0_l - Mulliken population, atomic charges
//! q_A = sum_l q_l, f(CN) the smoothed square root of the g-xTB CN.
//!
//! - First order: E1 = sum_l mu_l [1 + k_A f(CN_A)] g(q_A) q_l,
//!   g(q) = 1/2 {2 + 0.025 [erf(q - 0.666666666) + erf(q + 0.666666666)]}.
//! - Second order: E2 = 1/2 sum q_i gamma_ij q_j + sum_i q_i sum_{j not on A(i)} gamma_ij r_j,
//!   gamma_ij = 1/(R + exp(-0.294621155 R)/eta_ij), eta_ij = 2/(1/U_i + 1/U_j),
//!   U_l = `ps_tb2_shell_hubbard` `pa_hubbard_parameter` [1 + `pa_tb2_hubbard_cn` f(CN)],
//!   r_l = n0_l - Z_l.
//! - Third order: E3 = 1/3 sum_i q_i^2 Gamma_i sum_j tau_ij q_j, Gamma_l = `pa_tb3_hubbard_derivs`
//!   k_l, tau = -0.2093327496 eta^2 within an atom, 2.3 x (1 - 1.3 x/2) exp(-1.3 x) with
//!   x = R/eta between atoms.
//! - Fourth order: E4 = 1/24 sum_l 0.036 k_l q_l^4.
//! - Anisotropic electrostatics: the v1.1 pair formula ([`aes_energy_rows`]) with
//!   kernels g_n = 1/2 k_n [1 - erf(-s_n (R - R̄_AB))]/R^n and atomic dipoles scaled by
//!   c_A = 1 + `pa_aes_dip_scale` f(CN_A).
//! - Spin: E = 1/2 sum_A sum_{l,l' in A} m_l W_ll' m_l', W = `pa_wll_scale` W0.
//!
//! All potentials are exact derivatives (the program's Fock matrix is exact for these terms).
//!
//! g-xTB v1.1 ([`Version::V1_1`], the 1.1.0 program's `setgab_lrao_`, `electro_`,
//! `xtb_aespot`) has the same terms with its own parameters and, in a few places, its own forms
//! ([`IsoForm`], [`AesCache`]): the plain square root of the CN; the reference-charge cross
//! term reported with the first order ("ES1"); gamma = 1/(R + exp(-glob(12) R)/eta) evaluated at
//! R = 1e-12 bohr within an atom; tau = 1/2 R exp(-1.15 x) (3.2 - 1.84 x)/eta between atoms and
//! -gamma²/2 within an atom; the third order with the atomic charge,
//! E3 = 1/3 sum_{l,l'} q_l q_l' q_A(l) Gamma_A(l) tau_ll' (Gamma the element's `gam3`); a constant
//! fourth order 0.021/4 q^4; AES kernels k_n/2 [1 - erf(-a_n (R - R0))]/R^n with the D3 pair
//! radius in Ångström used as bohr and pairs beyond 200 bohr left out; no dipole scaling; the
//! spin constants `atspin` W0.

use crate::data::runtime::REF_SHELL_OCC;
use crate::data::v1_1::{GAM, GAM3};
use crate::data::v2_0_1::{
    AES_DIP_SCALE, HUBBARD, HUBBARD_CN, HUBBARD_DERIV, IPEA, IPEA_CN, PG_TB3_KSHELL, PG_TB4_KSHELL, REF_OCC,
    SHELL_HUBBARD, SPIN_W0, WLL_SCALE, ZEFF_SHELL,
};
use crate::eeqbc::{erf_f64, vdw_rad_pair};
use crate::model::sqrt_cn;
use crate::model::{Model, SQRTPI};
use crate::options::Version;
use axtb::molecule::{norm2_sq, sub3};
use axtb::Molecule;

/// Shell list entry: (atom, shell index on the atom, l).
pub type Shell = (usize, usize, usize);

pub const K1_DIS: f64 = 0.025;
pub const K1_S: f64 = 0.666666666;
pub const K2_EXP: f64 = 0.294621155;
pub const K3_ONSITE: f64 = 0.2093327496;
pub const K3_A: f64 = 1.3;
pub const K3_B: f64 = 2.3;
pub const K4: f64 = 0.036;
/// AES kernel prefactors and steepnesses for n = 3, 5, 7, 9.
pub const AES_K: [f64; 4] = [0.3405910191, 0.1691310614, 0.074034339, -0.02];
pub const AES_S: [f64; 4] = [0.5, 1.0, 1.0, 1.0];

/// v1.1: the first-order amplitude (the program's 0.012 [...] is k/2 with k = 0.024), the
/// distance of an atom from itself in gamma (the program's packed distance array), the fourth
/// order constant per shell, the AES erf steepnesses (g3; g5, g7, g9), the quadrupole-
/// quadrupole AES prefactor and the AES pair cutoff (bohr).
pub const K1_DIS_V1_1: f64 = 0.024;
pub const R_SAME_ATOM_V1_1: f64 = 1e-12;
pub const K4_V1_1: f64 = 0.021;
pub const AES_A_V1_1: [f64; 2] = [0.45, 1.05];
pub const AES_K9_V1_1: f64 = -0.003;
pub const AES_RMAX_V1_1: f64 = 200.0;

/// The first-order charge function g(q) = 1/2 {2 + k [erf(q - 0.666666666) + erf(q + 0.666666666)]}
/// and its derivative (2.0.1: k = 0.025; v1.1: k = 0.024, which is the program's
/// 1 + 0.012 [...] bit for bit).
#[inline]
pub fn first_order_g(q: f64, k: f64) -> (f64, f64) {
    let (a, b) = (q - K1_S, q + K1_S);
    (0.5 * (2.0 + k * (erf_f64(a) + erf_f64(b))), 0.5 * k * 2.0 / SQRTPI * ((-a * a).exp() + (-b * b).exp()))
}

/// Second derivative of the first-order charge function of [`first_order_g`].
#[inline]
pub fn first_order_g2(q: f64, k: f64) -> f64 {
    let (a, b) = (q - K1_S, q + K1_S);
    -0.5 * k * 4.0 / SQRTPI * (a * (-a * a).exp() + b * (-b * b).exp())
}

/// gamma(R, eta) = 1/(R + exp(-k R)/eta) with the exponent `k`, and its derivatives with
/// respect to R and eta.
#[inline]
pub fn gamma_kernel_k(r: f64, eta: f64, k: f64) -> (f64, f64, f64) {
    let ex = (-k * r).exp();
    let den = r + ex / eta;
    let g = 1.0 / den;
    (g, -(1.0 - k * ex / eta) * g * g, ex / (eta * eta) * g * g)
}

/// gamma(R, eta) of 2.0.1 ([`gamma_kernel_k`] with k = 0.294621155) and its derivatives.
#[inline]
pub fn gamma_kernel(r: f64, eta: f64) -> (f64, f64, f64) {
    gamma_kernel_k(r, eta, K2_EXP)
}

/// tau(R, eta) between atoms (`same = false`) or within an atom, and its derivatives with
/// respect to R and eta.
#[inline]
pub fn tau_kernel(r: f64, eta: f64, same: bool) -> (f64, f64, f64) {
    if same {
        (-K3_ONSITE * eta * eta, 0.0, -2.0 * K3_ONSITE * eta)
    } else {
        let x = r / eta;
        let e = (-K3_A * x).exp();
        let t = K3_B * x * (1.0 - K3_A * x / 2.0) * e;
        // dt/dx = 2.3 e [1 - 1.3 x - 1.3 x (1 - 1.3 x/2)]
        let dtdx = K3_B * e * (1.0 - K3_A * x - K3_A * x * (1.0 - K3_A * x / 2.0));
        (t, dtdx / eta, -dtdx * r / (eta * eta))
    }
}

/// AES kernel g_n at distance `r` with pair radius `r0` and its R derivative (n index 0..4
/// for n = 3, 5, 7, 9).
#[inline]
pub fn aes_kernel(n: usize, r: f64, r0: f64) -> (f64, f64) {
    aes_kernel_with(0.5 * AES_K[n], AES_S[n], (3 + 2 * n) as i32, r, r0)
}

/// The AES kernel c [1 - erf(-a (R - R0))]/R^p and its R derivative.
#[inline]
pub fn aes_kernel_with(c: f64, a: f64, p: i32, r: f64, r0: f64) -> (f64, f64) {
    let f = 1.0 - erf_f64(-a * (r - r0));
    let df = 2.0 * a / SQRTPI * (-(a * (r - r0)).powi(2)).exp();
    (c * f / r.powi(p), c * (df / r.powi(p) - p as f64 * f / r.powi(p + 1)))
}

/// Which charge multiplies the shell charge in the third order: E3 = 1/3 sum_l c_l q_l (tau q)_l
/// with c_l = q_l Gamma_l (2.0.1) or c_l = q_A(l) Gamma_l (v1.1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ThirdOrder {
    Shell,
    Atomic,
}

/// The form of the isotropic electrostatics of a version (what is not a parameter).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IsoForm {
    pub version: Version,
    pub third: ThirdOrder,
    /// The reference-charge cross term of the second order counts as first order (v1.1's ES1).
    pub cross_first: bool,
    /// The amplitude k of the first-order charge function ([`first_order_g`]).
    pub k1: f64,
    /// The exponent of gamma and the distance of an atom from itself.
    pub kgamma: f64,
    pub onsite_r: f64,
}

impl IsoForm {
    /// The form of `version`.
    pub fn of(version: Version) -> Self {
        match version {
            Version::V2_0_1 => IsoForm {
                version,
                third: ThirdOrder::Shell,
                cross_first: false,
                k1: K1_DIS,
                kgamma: K2_EXP,
                onsite_r: 0.0,
            },
            Version::V1_1 => IsoForm {
                version,
                third: ThirdOrder::Atomic,
                cross_first: true,
                k1: K1_DIS_V1_1,
                kgamma: crate::params::gxtb_v1_1().glob[11],
                onsite_r: R_SAME_ATOM_V1_1,
            },
        }
    }

    /// gamma and tau of a shell pair at distance `r` (atoms `same` or not; `r` is replaced by the
    /// form's onsite distance within an atom) with the average `eta`, and their derivatives with
    /// respect to R and eta: ((gamma, dgamma/dR, dgamma/deta), (tau, dtau/dR, dtau/deta)).
    #[inline]
    pub fn kernels(&self, r: f64, eta: f64, same: bool) -> ((f64, f64, f64), (f64, f64, f64)) {
        match self.version {
            Version::V2_0_1 => (gamma_kernel(r, eta), tau_kernel(r, eta, same)),
            Version::V1_1 => {
                let r = if same { self.onsite_r } else { r };
                let g = gamma_kernel_k(r, eta, self.kgamma);
                let t = if same {
                    (-0.5 * g.0 * g.0, 0.0, -g.0 * g.2)
                } else {
                    // tau = 1/2 e x (3.2 - 1.84 x), x = R/eta, e = exp(-1.15 x)
                    let s = r / eta;
                    let e = (-1.15 * s).exp();
                    let dphi = 0.5 * e * ((3.2 - 1.84 * s) - 1.84 * s - 1.15 * s * (3.2 - 1.84 * s));
                    (0.5 * e * s * (3.2 - 1.84 * s), dphi / eta, -dphi * r / (eta * eta))
                };
                (g, t)
            }
        }
    }
}

/// The isotropic electrostatics (first to fourth order) of one geometry: per-shell parameters
/// at the g-xTB CN, their CN derivatives, and the gamma and tau kernels.
#[derive(Clone, Debug)]
pub struct IsotropicCache {
    pub shells: Vec<Shell>,
    pub nat: usize,
    /// First-order mu_l [1 + k f(CN)] per shell and its CN derivative.
    pub mu: Vec<f64>,
    pub dmu: Vec<f64>,
    /// Hubbard parameters U'_l and their CN derivatives.
    pub u: Vec<f64>,
    pub du: Vec<f64>,
    /// Second- and third-order kernels (nsh x nsh).
    pub gamma: Vec<f64>,
    pub tau: Vec<f64>,
    /// Gamma_l (third order) and the fourth-order constant (E4 = gam4 q^4 / 24) per shell.
    pub gam3: Vec<f64>,
    pub gam4: Vec<f64>,
    /// Reference shell charge r_l = n0_l - Z_l.
    pub rref: Vec<f64>,
    pub form: IsoForm,
}

impl IsotropicCache {
    /// The isotropic electrostatics of `mol` in `model` with the SCF-basis shells `shells`
    /// ([`crate::fock::shell_list`]) and the g-xTB CN `cn`.
    pub fn new(model: &Model, mol: &Molecule, shells: &[Shell], cn: &[f64]) -> Self {
        Self::new_par(model, mol, shells, cn, false)
    }

    /// [`IsotropicCache::new`], kernel rows in parallel when `parallel` (same result).
    pub fn new_par(model: &Model, mol: &Molecule, shells: &[Shell], cn: &[f64], parallel: bool) -> Self {
        let shells = shells.to_vec();
        let nsh = shells.len();
        let nat = mol.nat();
        let form = IsoForm::of(model.version);
        let (mut mu, mut dmu, mut u, mut du) =
            (Vec::with_capacity(nsh), Vec::with_capacity(nsh), Vec::with_capacity(nsh), Vec::with_capacity(nsh));
        let (mut gam3, mut gam4, mut rref) =
            (Vec::with_capacity(nsh), Vec::with_capacity(nsh), Vec::with_capacity(nsh));
        match model.version {
            Version::V2_0_1 => {
                for &(iat, ish, l) in &shells {
                    let z = mol.numbers[iat] - 1;
                    let (f, df) = sqrt_cn(cn[iat]);
                    mu.push(IPEA[z][ish] * (1.0 + IPEA_CN[z] * f));
                    dmu.push(IPEA[z][ish] * IPEA_CN[z] * df);
                    u.push(SHELL_HUBBARD[z][ish] * HUBBARD[z] * (1.0 + HUBBARD_CN[z] * f));
                    du.push(SHELL_HUBBARD[z][ish] * HUBBARD[z] * HUBBARD_CN[z] * df);
                    gam3.push(HUBBARD_DERIV[z] * PG_TB3_KSHELL[l]);
                    gam4.push(K4 * PG_TB4_KSHELL[l]);
                    rref.push(REF_OCC[z][ish] - ZEFF_SHELL[z][ish]);
                }
            }
            Version::V1_1 => {
                let p = crate::params::gxtb_v1_1();
                for &(iat, ish, _) in &shells {
                    let z = mol.numbers[iat];
                    let e = p.element(z).expect("g-xTB v1.1 parameters of a checked element");
                    let sq = cn[iat].sqrt();
                    mu.push(e.gam1[ish] * (1.0 + e.e1cn * sq));
                    dmu.push(if sq > 0.0 { e.gam1[ish] * e.e1cn / (2.0 * sq) } else { 0.0 });
                    u.push(e.gam2[ish] * ((e.g2cn * sq + 1.0) * GAM[z - 1]));
                    du.push(if sq > 0.0 { e.gam2[ish] * GAM[z - 1] * e.g2cn / (2.0 * sq) } else { 0.0 });
                    gam3.push(GAM3[z - 1]);
                    gam4.push(6.0 * K4_V1_1);
                    rref.push(REF_SHELL_OCC[z - 1][ish] - crate::data::v1_1::ZEFF_SHELL[ish * 103 + z - 1] as f64);
                }
            }
        }
        let [gamma, tau] = match model.version {
            Version::V2_0_1 => crate::parallel::symmetric(nsh, parallel, |i, j| {
                let (ia, ja) = (shells[i].0, shells[j].0);
                let same = ia == ja;
                let r = if same { 0.0 } else { norm2_sq(sub3(mol.positions[ia], mol.positions[ja])).sqrt() };
                let eta = 2.0 / (1.0 / u[i] + 1.0 / u[j]);
                [gamma_kernel(r, eta).0, tau_kernel(r, eta, same).0]
            }),
            Version::V1_1 => {
                // `setgab_lrao_`: gamma = 1/(R + exp(-k R)/eta) with 1/eta formed first, R = 1e-12 bohr
                // within an atom; tau = -gamma²/2 within an atom, 1/2 R exp(-1.15 R/eta)
                // (3.2 - 1.84 R/eta)/eta between atoms
                let k = form.kgamma;
                crate::parallel::symmetric(nsh, parallel, |i, j| {
                    let (ia, ja) = (shells[i].0, shells[j].0);
                    let same = ia == ja;
                    let r = if same {
                        R_SAME_ATOM_V1_1
                    } else {
                        norm2_sq(sub3(mol.positions[ia], mol.positions[ja])).sqrt()
                    };
                    let inv_eta = 1.0 / (2.0 / (1.0 / u[i] + 1.0 / u[j]));
                    let g = 1.0 / (r + inv_eta * (-k * r).exp());
                    let t = if same {
                        -0.5 * g * g
                    } else {
                        let e = (-1.15 * r * inv_eta).exp();
                        0.5 * e * r * ((-1.84 * r * inv_eta + 3.2) * inv_eta)
                    };
                    [g, t]
                })
            }
        };
        Self { shells, nat, mu, dmu, u, du, gamma, tau, gam3, gam4, rref, form }
    }

    /// Atomic charges from shell charges ([`axtb_base::math::atom_charges`]).
    pub fn atom_charges(&self, q: &[f64]) -> Vec<f64> {
        atom_charges(self.nat, &self.shells, q)
    }

    /// The four pieces of the energy, shells in fixed chunks (in parallel when `parallel`; same
    /// result): (E1, E2, E3, E4); v1.1's first piece includes the cross term (its ES1).
    pub fn energies(&self, q: &[f64], parallel: bool) -> [f64; 4] {
        let n = self.shells.len();
        let qa = self.atom_charges(q);
        let (k1, third, cross_first) = (self.form.k1, self.form.third, self.form.cross_first);
        let parts = axtb_base::threads::map_chunks(n, parallel, |rows| {
            let (mut e1, mut e2, mut e3, mut e4) = (0.0, 0.0, 0.0, 0.0);
            for i in rows {
                let ia = self.shells[i].0;
                e1 += self.mu[i] * first_order_g(qa[ia], k1).0 * q[i];
                let (mut gq, mut gr, mut tq) = (0.0, 0.0, 0.0);
                for j in 0..n {
                    gq += self.gamma[i * n + j] * q[j];
                    if self.shells[j].0 != ia {
                        gr += self.gamma[i * n + j] * self.rref[j];
                    }
                    tq += self.tau[i * n + j] * q[j];
                }
                if cross_first {
                    e1 += q[i] * gr;
                    e2 += 0.5 * q[i] * gq;
                } else {
                    e2 += 0.5 * q[i] * gq + q[i] * gr;
                }
                e3 += match third {
                    ThirdOrder::Shell => q[i] * q[i] * self.gam3[i] * tq / 3.0,
                    ThirdOrder::Atomic => q[i] * qa[ia] * self.gam3[i] * tq / 3.0,
                };
                e4 += self.gam4[i] * q[i].powi(4) / 24.0;
            }
            [e1, e2, e3, e4]
        });
        let mut e = [0.0; 4];
        for p in parts {
            for k in 0..4 {
                e[k] += p[k];
            }
        }
        e
    }

    /// Shell potentials dE1..4/dq_l, shells in parallel when `parallel` (same result).
    pub fn potential(&self, q: &[f64], parallel: bool) -> Vec<f64> {
        let n = self.shells.len();
        let qa = self.atom_charges(q);
        let k1 = self.form.k1;
        let mut muq = vec![0.0; self.nat];
        for i in 0..n {
            muq[self.shells[i].0] += self.mu[i] * q[i];
        }
        let tq: Vec<f64> = axtb_base::threads::map(n, parallel, |i| (0..n).map(|j| self.tau[i * n + j] * q[j]).sum());
        // E3 = 1/3 sum_j w3_j-like terms: d/dq_i of sum_j c_j q_j (tau q)_j
        let (w3, atom3): (Vec<f64>, Vec<f64>) = match self.form.third {
            ThirdOrder::Shell => ((0..n).map(|i| q[i] * q[i] * self.gam3[i]).collect(), Vec::new()),
            ThirdOrder::Atomic => {
                // c_j = q_A(j) Gamma_j; dc_j/dq_i = Gamma_j for j on the atom of i
                let mut atom3 = vec![0.0; self.nat];
                for i in 0..n {
                    atom3[self.shells[i].0] += self.gam3[i] * q[i] * tq[i];
                }
                ((0..n).map(|i| qa[self.shells[i].0] * self.gam3[i] * q[i]).collect(), atom3)
            }
        };
        let third = self.form.third;
        axtb_base::threads::map(n, parallel, |i| {
            let ia = self.shells[i].0;
            let (g, dg) = first_order_g(qa[ia], k1);
            let mut vi = self.mu[i] * g + muq[ia] * dg;
            for j in 0..n {
                vi += self.gamma[i * n + j] * q[j];
                if self.shells[j].0 != ia {
                    vi += self.gamma[i * n + j] * self.rref[j];
                }
                // tau is symmetric: its column i is its row i
                vi += w3[j] * self.tau[i * n + j] / 3.0;
            }
            match third {
                ThirdOrder::Shell => vi += 2.0 * q[i] * self.gam3[i] * tq[i] / 3.0,
                ThirdOrder::Atomic => vi += (qa[ia] * self.gam3[i] * tq[i] + atom3[ia]) / 3.0,
            }
            vi += self.gam4[i] * q[i].powi(3) / 6.0;
            vi
        })
    }

    /// Change of the shell potentials dE1..4/dq_l for shell charge changes `dq` at shell
    /// charges `q` (the Hessian of E1 + E2 + E3 + E4 times `dq`).
    pub fn hessian_vec(&self, q: &[f64], dq: &[f64]) -> Vec<f64> {
        let n = self.shells.len();
        let k1 = self.form.k1;
        let qa = self.atom_charges(q);
        let dqa = self.atom_charges(dq);
        let mut muq = vec![0.0; self.nat];
        let mut mudq = vec![0.0; self.nat];
        for i in 0..n {
            muq[self.shells[i].0] += self.mu[i] * q[i];
            mudq[self.shells[i].0] += self.mu[i] * dq[i];
        }
        let tq: Vec<f64> = (0..n).map(|i| (0..n).map(|j| self.tau[i * n + j] * q[j]).sum()).collect();
        let tdq: Vec<f64> = (0..n).map(|i| (0..n).map(|j| self.tau[i * n + j] * dq[j]).sum()).collect();
        let mut v = vec![0.0; n];
        match self.form.third {
            ThirdOrder::Shell => {
                // sum_l tau_kl a_l q_l dq_l
                let w: Vec<f64> = (0..n).map(|l| self.gam3[l] * q[l] * dq[l]).collect();
                for i in 0..n {
                    let ia = self.shells[i].0;
                    let (_, dg) = first_order_g(qa[ia], k1);
                    let d2g = first_order_g2(qa[ia], k1);
                    let mut vi = self.mu[i] * dg * dqa[ia] + mudq[ia] * dg + muq[ia] * d2g * dqa[ia];
                    let mut gd = 0.0;
                    let mut tw = 0.0;
                    for j in 0..n {
                        gd += self.gamma[i * n + j] * dq[j];
                        tw += self.tau[i * n + j] * w[j];
                    }
                    vi += gd;
                    vi += 2.0 / 3.0 * (self.gam3[i] * tq[i] * dq[i] + self.gam3[i] * q[i] * tdq[i] + tw);
                    vi += 0.5 * self.gam4[i] * q[i] * q[i] * dq[i];
                    v[i] = vi;
                }
            }
            ThirdOrder::Atomic => {
                // v_k = 1/3 [sum_i c_i q_i tau_ik + c_k t_k + sum_{i on A(k)} Gamma_i q_i t_i],
                // c_i = q_A(i) Gamma_i, t = tau q: its change for dq
                let w: Vec<f64> = (0..n)
                    .map(|i| {
                        let ia = self.shells[i].0;
                        self.gam3[i] * (dqa[ia] * q[i] + qa[ia] * dq[i])
                    })
                    .collect();
                let mut atom = vec![0.0; self.nat];
                for i in 0..n {
                    atom[self.shells[i].0] += self.gam3[i] * (dq[i] * tq[i] + q[i] * tdq[i]);
                }
                for i in 0..n {
                    let ia = self.shells[i].0;
                    let (_, dg) = first_order_g(qa[ia], k1);
                    let d2g = first_order_g2(qa[ia], k1);
                    let mut vi = self.mu[i] * dg * dqa[ia] + mudq[ia] * dg + muq[ia] * d2g * dqa[ia];
                    let mut gd = 0.0;
                    let mut tw = 0.0;
                    for j in 0..n {
                        gd += self.gamma[i * n + j] * dq[j];
                        tw += self.tau[i * n + j] * w[j];
                    }
                    vi += gd;
                    let c = self.gam3[i] * (dqa[ia] * tq[i] + qa[ia] * tdq[i]);
                    vi += (tw + c + atom[ia]) / 3.0;
                    vi += 0.5 * self.gam4[i] * q[i] * q[i] * dq[i];
                    v[i] = vi;
                }
            }
        }
        v
    }
}

/// Anisotropic electrostatics of one geometry: the damped kernels g3, g5, g7, g9 (nat x nat) and
/// the dipole scaling (2.0.1) at the g-xTB CN.
#[derive(Clone, Debug)]
pub struct AesCache {
    pub model: Model,
    pub nat: usize,
    /// AES kernels g3, g5, g7, g9 (nat x nat).
    pub aes: [Vec<f64>; 4],
    /// Dipole scaling c_A and its CN derivative (2.0.1); v1.1 scales nothing (both empty).
    pub dip_scale: Vec<f64>,
    pub ddip_scale: Vec<f64>,
}

impl AesCache {
    /// The anisotropic electrostatics of `mol` in `model` at the g-xTB CN `cn`.
    pub fn new(model: &Model, mol: &Molecule, cn: &[f64]) -> Self {
        Self::new_par(model, mol, cn, false)
    }

    /// [`AesCache::new`], kernel rows in parallel when `parallel` (same result).
    pub fn new_par(model: &Model, mol: &Molecule, cn: &[f64], parallel: bool) -> Self {
        let nat = mol.nat();
        match model.version {
            Version::V2_0_1 => {
                let aes = crate::parallel::symmetric(nat, parallel, |i, j| {
                    if i == j {
                        return [0.0; 4];
                    }
                    let r = norm2_sq(sub3(mol.positions[i], mol.positions[j])).sqrt();
                    let r0 = model.rvdw(mol.numbers[i], mol.numbers[j]);
                    [0, 1, 2, 3].map(|n| aes_kernel(n, r, r0).0)
                });
                let dip_scale = (0..nat).map(|i| 1.0 + AES_DIP_SCALE[mol.numbers[i] - 1] * sqrt_cn(cn[i]).0).collect();
                let ddip_scale = (0..nat).map(|i| AES_DIP_SCALE[mol.numbers[i] - 1] * sqrt_cn(cn[i]).1).collect();
                Self { model: *model, nat, aes, dip_scale, ddip_scale }
            }
            Version::V1_1 => {
                // `mmomgaberf`: the program's kernels (pairs beyond 200 bohr zero)
                let g = &crate::params::gxtb_v1_1().glob;
                let aes = crate::parallel::symmetric(nat, parallel, |i, j| {
                    let r = norm2_sq(sub3(mol.positions[i], mol.positions[j])).sqrt();
                    if i == j || r > AES_RMAX_V1_1 {
                        return [0.0; 4];
                    }
                    let r0 = vdw_rad_pair(mol.numbers[i], mol.numbers[j]);
                    let (r2, r3) = (r * r, r * r * r);
                    let (r5, r7, r9) = (r2 * r3, r2 * r2 * r3, r2 * r2 * r2 * r3);
                    let f3 = (1.0 - erf_f64(-AES_A_V1_1[0] * (r - r0))) * (0.5 * g[15]);
                    let f5 = (1.0 - erf_f64(-AES_A_V1_1[1] * (r - r0))) * 0.5;
                    [f3 / r3, g[16] * f5 / r5, g[17] * f5 / r7, f5 * AES_K9_V1_1 / r9]
                });
                Self { model: *model, nat, aes, dip_scale: Vec::new(), ddip_scale: Vec::new() }
            }
        }
    }

    /// The four kernels of an atom pair at distance `r` with their R derivatives (the
    /// gradient's; zero beyond the version's pair cutoff).
    #[inline]
    pub fn pair_kernels(&self, za: usize, zb: usize, r: f64) -> [(f64, f64); 4] {
        match self.model.version {
            Version::V2_0_1 => {
                let r0 = self.model.rvdw(za, zb);
                [0, 1, 2, 3].map(|n| aes_kernel(n, r, r0))
            }
            Version::V1_1 => {
                if r > AES_RMAX_V1_1 {
                    return [(0.0, 0.0); 4];
                }
                let g = &crate::params::gxtb_v1_1().glob;
                let r0 = vdw_rad_pair(za, zb);
                [
                    aes_kernel_with(0.5 * g[15], AES_A_V1_1[0], 3, r, r0),
                    aes_kernel_with(0.5 * g[16], AES_A_V1_1[1], 5, r, r0),
                    aes_kernel_with(0.5 * g[17], AES_A_V1_1[1], 7, r, r0),
                    aes_kernel_with(0.5 * AES_K9_V1_1, AES_A_V1_1[1], 9, r, r0),
                ]
            }
        }
    }

    /// Whether the atomic dipoles are scaled (2.0.1).
    #[inline]
    pub fn scales_dipoles(&self) -> bool {
        !self.dip_scale.is_empty()
    }

    /// Scaled dipoles c_A d_A (also the dipole potential dE/dd_A = c_A dE/d(c d)_A of a
    /// potential of the scaled dipoles); the dipoles themselves without scaling.
    pub fn scaled_dipoles(&self, dip: &[[f64; 3]]) -> Vec<[f64; 3]> {
        if !self.scales_dipoles() {
            return dip.to_vec();
        }
        dip.iter().zip(&self.dip_scale).map(|(d, c)| [c * d[0], c * d[1], c * d[2]]).collect()
    }

    /// AES energy ([`aes_energy_rows`] with the scaled dipoles), atom rows in fixed chunks
    /// (in parallel when `parallel`; same result).
    pub fn energy(&self, mol: &Molecule, qa: &[f64], dip: &[[f64; 3]], qp: &[[f64; 6]], parallel: bool) -> f64 {
        let d = self.scaled_dipoles(dip);
        axtb_base::threads::map_chunks(mol.nat(), parallel, |rows| aes_energy_rows(mol, qa, &d, qp, &self.aes, rows))
            .into_iter()
            .sum()
    }

    /// AES potentials dE/dq_A, dE/dd_A and dE/dQ_A (the off-diagonal quadrupole components
    /// counted twice), exact; atom rows in fixed chunks (in parallel when `parallel`; same
    /// result).
    #[allow(clippy::type_complexity)]
    pub fn potentials(
        &self,
        mol: &Molecule,
        qa: &[f64],
        dip: &[[f64; 3]],
        qp: &[[f64; 6]],
        parallel: bool,
    ) -> (Vec<f64>, Vec<[f64; 3]>, Vec<[f64; 6]>) {
        let (vs, vd_scaled, vq) = self.potentials_raw(mol, qa, dip, qp, parallel);
        (vs, self.scaled_dipoles(&vd_scaled), vq)
    }

    /// [`AesCache::potentials`] with the dipole potential dE/d(c d) of the scaled dipoles.
    #[allow(clippy::type_complexity)]
    pub fn potentials_raw(
        &self,
        mol: &Molecule,
        qa: &[f64],
        dip: &[[f64; 3]],
        qp: &[[f64; 6]],
        parallel: bool,
    ) -> (Vec<f64>, Vec<[f64; 3]>, Vec<[f64; 6]>) {
        potentials_with(mol, qa, &self.scaled_dipoles(dip), qp, &self.aes, parallel)
    }
}

/// The AES potentials of the (scaled) dipoles `d` with the kernels `k`, atom rows in fixed
/// chunks (in parallel when `parallel`; same result).
#[allow(clippy::type_complexity)]
pub fn potentials_with(
    mol: &Molecule,
    qa: &[f64],
    d: &[[f64; 3]],
    qp: &[[f64; 6]],
    k: &[Vec<f64>; 4],
    parallel: bool,
) -> (Vec<f64>, Vec<[f64; 3]>, Vec<[f64; 6]>) {
    let n = mol.nat();
    crate::parallel::reduce_rows(
        n,
        parallel,
        || (vec![0.0; n], vec![[0.0; 3]; n], vec![[0.0; 6]; n]),
        |i, acc| aes_potentials_row(mol, qa, d, qp, k, i, &mut acc.0, &mut acc.1, &mut acc.2),
        |a, b| {
            crate::parallel::add_into(&mut a.0, &b.0);
            crate::parallel::add_into(a.1.as_flattened_mut(), b.1.as_flattened());
            crate::parallel::add_into(a.2.as_flattened_mut(), b.2.as_flattened());
        },
    )
}

/// The AES potential terms (exact derivatives)
/// of atom `i` and its partners j < i, added to `vs`, `vd`, `vq`; the potentials of atom i are
/// summed in registers.
pub fn aes_potentials_row(
    mol: &Molecule,
    q: &[f64],
    dip: &[[f64; 3]],
    qp: &[[f64; 6]],
    k: &[Vec<f64>; 4],
    i: usize,
    vs: &mut [f64],
    vd: &mut [[f64; 3]],
    vq: &mut [[f64; 6]],
) {
    let n = mol.nat();
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let qmat = |q: &[f64; 6]| [[q[0], q[1], q[3]], [q[1], q[2], q[4]], [q[3], q[4], q[5]]];
    let mv = |m: &[[f64; 3]; 3], v: [f64; 3]| [dot(m[0], v), dot(m[1], v), dot(m[2], v)];
    let sym = |a: [f64; 3], b: [f64; 3]| {
        [
            a[0] * b[0],
            a[0] * b[1] + a[1] * b[0],
            a[1] * b[1],
            a[0] * b[2] + a[2] * b[0],
            a[1] * b[2] + a[2] * b[1],
            a[2] * b[2],
        ]
    };
    let w = [1.0, 2.0, 1.0, 2.0, 2.0, 1.0];
    let p = &mol.positions;
    let (k3, k5, k7, k9) =
        (&k[0][i * n..i * n + i], &k[1][i * n..i * n + i], &k[2][i * n..i * n + i], &k[3][i * n..i * n + i]);
    let qi = qmat(&qp[i]);
    let (pi, di, qpi, chi) = (p[i], dip[i], qp[i], q[i]);
    let mut vsi = 0.0;
    let mut vdi = [0.0; 3];
    let mut vqi = [0.0; 6];
    for j in 0..i {
        // (g9 carries the 1/3 of the QQ term)
        let (g3, g5, g7, g9) = (k3[j], k5[j], k7[j], k9[j] / 3.0);
        let r = [p[j][0] - pi[0], p[j][1] - pi[1], p[j][2] - pi[2]];
        let r2 = dot(r, r);
        let rr = [r[0] * r[0], 2.0 * r[0] * r[1], r[1] * r[1], 2.0 * r[0] * r[2], 2.0 * r[1] * r[2], r[2] * r[2]];
        let qj = qmat(&qp[j]);
        let (qir, qjr) = (mv(&qi, r), mv(&qj, r));
        let (rqir, rqjr) = (dot(r, qir), dot(r, qjr));
        let dj = dip[j];
        let (dir, djr) = (dot(di, r), dot(dj, r));
        let chj = q[j];
        let (rdj, rdi) = (sym(r, dj), sym(r, di));
        let (rqj, rqi) = (sym(r, qjr), sym(r, qir));
        // e_qd g3, e_qQ g5
        vs[j] += dir * g3 + rqir * g5;
        vsi += -djr * g3 + rqjr * g5;
        let (vdj, vqj, qpj) = (&mut vd[j], &mut vq[j], &qp[j]);
        for x in 0..3 {
            // e_qd g3, e_dd g5, - e_dQ g7
            vdi[x] +=
                chj * r[x] * g3 + (dj[x] * r2 - 3.0 * r[x] * djr) * g5 + (5.0 * r[x] * rqjr - 2.0 * r2 * qjr[x]) * g7;
            vdj[x] +=
                -chi * r[x] * g3 + (di[x] * r2 - 3.0 * r[x] * dir) * g5 - (5.0 * rqir * r[x] - 2.0 * r2 * qir[x]) * g7;
        }
        for c in 0..6 {
            // e_qQ g5, - e_dQ g7, 1/3 e_QQ g9
            vqi[c] += chj * rr[c] * g5 - (5.0 * rr[c] * djr - 2.0 * r2 * rdj[c]) * g7
                + (35.0 * rr[c] * rqjr - 20.0 * r2 * rqj[c] + 3.0 * r2 * r2 * w[c] * qpj[c]) * g9;
            vqj[c] += chi * rr[c] * g5
                + (5.0 * rr[c] * dir - 2.0 * r2 * rdi[c]) * g7
                + (35.0 * rr[c] * rqir - 20.0 * r2 * rqi[c] + 3.0 * r2 * r2 * w[c] * qpi[c]) * g9;
        }
    }
    vs[i] += vsi;
    for x in 0..3 {
        vd[i][x] += vdi[x];
    }
    for c in 0..6 {
        vq[i][c] += vqi[c];
    }
}

/// Spin constant W_ll' of an atom of element `z` in `model`.
#[inline]
pub fn spin_w(model: &Model, z: usize, la: usize, lb: usize) -> f64 {
    match model.version {
        Version::V2_0_1 => WLL_SCALE[z - 1] * SPIN_W0[z - 1][spin_index(la, lb)],
        Version::V1_1 => {
            let k = crate::params::gxtb_v1_1().element(z).expect("g-xTB v1.1 parameters of a checked element").atspin;
            k * crate::data::v1_1::SPIN_W0[(z - 1) * 10 + spin_index(la, lb)]
        }
    }
}

/// Spin-polarization energy of shell magnetizations `m`.
pub fn spin_energy(model: &Model, mol: &Molecule, shells: &[Shell], m: &[f64]) -> f64 {
    let mut e = 0.0;
    for (i, &(ia, _, li)) in shells.iter().enumerate() {
        for (j, &(ja, _, lj)) in shells.iter().enumerate() {
            if ja == ia {
                e += 0.5 * m[i] * spin_w(model, mol.numbers[ia], li, lj) * m[j];
            }
        }
    }
    e
}

/// Spin potentials dE/dm_l.
pub fn spin_potential(model: &Model, mol: &Molecule, shells: &[Shell], m: &[f64]) -> Vec<f64> {
    let mut w = vec![0.0; shells.len()];
    for (i, &(ia, _, li)) in shells.iter().enumerate() {
        for (j, &(ja, _, lj)) in shells.iter().enumerate() {
            if ja == ia {
                w[i] += spin_w(model, mol.numbers[ia], li, lj) * m[j];
            }
        }
    }
    w
}

// ---- the AES energy and the shell bookkeeping (from the old tree's aes.rs and es.rs) ----

fn qmat(q: &[f64; 6]) -> [[f64; 3]; 3] {
    [[q[0], q[1], q[3]], [q[1], q[2], q[4]], [q[3], q[4], q[5]]]
}

/// The pair terms (i, j < i) of the AES energy for the atoms i in `rows`.
pub fn aes_energy_rows(
    mol: &Molecule,
    q: &[f64],
    dip: &[[f64; 3]],
    qp: &[[f64; 6]],
    k: &[Vec<f64>; 4],
    rows: std::ops::Range<usize>,
) -> f64 {
    let n = mol.nat();
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let mv = |m: &[[f64; 3]; 3], v: [f64; 3]| [dot(m[0], v), dot(m[1], v), dot(m[2], v)];
    let mut e = 0.0;
    for i in rows {
        let qi = qmat(&qp[i]);
        for j in 0..i {
            let qj = qmat(&qp[j]);
            let r = sub3(mol.positions[j], mol.positions[i]);
            let r2 = dot(r, r);
            let (di, dj) = (dip[i], dip[j]);
            let (qir, qjr) = (mv(&qi, r), mv(&qj, r));
            let (rqir, rqjr) = (dot(r, qir), dot(r, qjr));
            let e_qd = q[j] * dot(di, r) - q[i] * dot(dj, r);
            let e_qq = q[j] * rqir + q[i] * rqjr;
            let e_dd = dot(dj, di) * r2 - 3.0 * dot(dj, r) * dot(di, r);
            let e_dq = (5.0 * rqir * dot(dj, r) - 2.0 * dot(r, mv(&qi, dj)) * r2)
                - (5.0 * dot(di, r) * rqjr - 2.0 * dot(r, mv(&qj, di)) * r2);
            let mut tr = 0.0;
            for a in 0..3 {
                for b in 0..3 {
                    tr += qi[a][b] * qj[a][b];
                }
            }
            let e_q2 = 35.0 * rqir * rqjr - 20.0 * dot(qir, qjr) * r2 + 3.0 * tr * r2 * r2;
            let ij = i * n + j;
            e += e_qd * k[0][ij] + (e_qq + e_dd) * k[1][ij] - e_dq * k[2][ij] + e_q2 * k[3][ij] / 3.0;
        }
    }
    e
}

/// Atomic charges from shell charges ([`axtb_base::math::atom_charges`]).
pub fn atom_charges(nat: usize, shells: &[(usize, usize, usize)], q: &[f64]) -> Vec<f64> {
    axtb_base::math::atom_charges(nat, shells.iter().map(|s| s.0), q)
}

/// Index (0-based) into the 10 spin constants of an element for a shell pair (l, l').
pub fn spin_index(la: usize, lb: usize) -> usize {
    const INDX: [usize; 16] = [1, 2, 3, 4, 2, 5, 6, 7, 3, 6, 8, 9, 4, 7, 9, 10];
    INDX[la * 4 + lb] - 1
}
