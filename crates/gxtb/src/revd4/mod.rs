//! revD4 dispersion of g-xTB 2.0.1 (spec §4.15): the v1.1 C6 model ([`mod@reference`]: revD4 CN,
//! pair-specific Gaussian weights, tanh charge scaling n(q)/n(qref), hydrogen-subtracted
//! reference polarizabilities, K matrices) with the 2.0.1 reference charges ([`CHARGES`]:
//! `gxtbq`, `gxtbh`) and CN radii, and the screened rational damping of the 2.0.1 dftd4 build:
//!
//! - rho_AB = sqrt(3 r4r2_A r4r2_B), reff_AB = a2 1/2 [1 + erf(-a3 (R - a4 rho_AB))] (a1 = 0);
//! - two-body (SCF Mulliken charges, every iteration): E2 = -sum_{A<B} C6_AB(q) s6/(R + reff)^6,
//!   pairs within 50 bohr (s8 = 0: no C8 term);
//! - three-body (C6 at q = 0, once): E3 = sum_{A<B<C} s9 sqrt(C6_AB C6_AC C6_BC)
//!   (3 cos a cos b cos c + 1) prod_pairs 1/(R + reff)^3, triples with all distances within
//!   25 bohr.
//!
//! Cutoffs (CN 30 bohr, 50, 25) are hard in program mode and smooth otherwise
//! ([`Model::cut`]).
//!
//! v1.1 ([`mod@reference`]: the 1.1.0 program's embedded dftd4) has the same C6 model with its
//! reference charges and CN radii and dftd4's dampings: the rational two-body damping with
//! C8, s6 = 1, s8 = glob(20), a1 = glob(10), a2 = 0, E2 = -sum C6_AB [s6/(R^6 + R0^6) +
//! s8 rr/(R^8 + R0^8)], rr = 3 r4r2_A r4r2_B, R0 = a1 sqrt(rr) + a2, pairs within 60 bohr; the
//! three-body term with dftd4's zero damping 1/(1 + 6 prod (R0/R)^(16/3)), s9 = 1.75, triples within
//! 20 bohr ([`Damping`]); every cutoff hard and the CN's (30 bohr) too, in both conventions.

pub mod reference;

use crate::data::v1_1::D4_RCOV as D4_RCOV_V1_1;
use crate::data::v2_0_1::{D4_REFH, D4_REFQ};
use crate::model::{Model, SQRTPI};
use crate::options::Version;
use axtb::molecule::{norm2_sq, sub3};
use axtb::{Molecule, Result};
use axtb_base::math::erf;
use axtb_base::ncoord::{CnModel, Cutoff, Erf, Smooth, Unweighted};
use reference::{k_matrices, revd4_model, ReferenceCharges, CHARGES_V1_1};

pub const S6: f64 = 1.0;
pub const S9: f64 = 1.826410691;
pub const A1: f64 = 0.0;
pub const A2: f64 = 8.5302496868;
pub const A3: f64 = 0.5395219242;
pub const A4: f64 = 0.654;
/// Cutoffs (bohr): CN, two-body, three-body.
pub const CUTOFF_CN: f64 = 30.0;
pub const CUTOFF_2B: f64 = 50.0;
pub const CUTOFF_3B: f64 = 25.0;
/// The reference charges of 2.0.1 (`gxtbq`, `gxtbh`).
pub const CHARGES: ReferenceCharges = ReferenceCharges { refq: &D4_REFQ, refh: &D4_REFH };

/// tanh charge function n(q) = a + b tanh(c q + d) of element `z` and its derivative.
pub fn charge_function(z: usize, q: f64) -> (f64, f64) {
    reference::charge_function(z, q)
}

/// Second derivative of the tanh charge function n(q) = a + b tanh(c q + d).
pub fn charge_function_d2(z: usize, q: f64) -> f64 {
    // n' = b c (1 - t^2) => n'' = -2 c t n'
    let (_, dn) = charge_function(z, q);
    let p = &crate::data::v1_1::D4_TANH[4 * (z - 1)..4 * z];
    let t = (q * p[2] + p[3]).tanh();
    -2.0 * p[2] * t * dn
}

/// The revD4 coordination number of `mol` (`axtb-base`'s [`CnModel`]): dftd4's count
/// ([`Erf::D4`]: erf, 7.5) without the electronegativity factor, Rc = R_A + R_B, times a cutoff
/// factor (product rule in the gradient); pairs with R beyond the cutoff or R² < 1e-12 bohr²
/// skipped, and in the gradient the pairs whose dE/dCN sum to zero. 2.0.1: the radii of
/// `covalent_rad_d3` and the factor [`Model::cut`] up to [`Model::reach`]; v1.1: the 1.1.0
/// program's radii and a hard 30-bohr cutoff (factor 1) in both conventions.
#[allow(clippy::type_complexity)] // the parts are closures, which have no nameable type
fn revd4_cn<'a>(
    model: &'a Model,
    mol: &'a Molecule,
) -> CnModel<Erf, impl Fn(usize, usize) -> f64 + 'a, Unweighted, Smooth<impl Fn(f64) -> (f64, f64) + 'a>> {
    let v1 = model.version == Version::V1_1;
    let rad = move |z: usize| if v1 { D4_RCOV_V1_1[z - 1] } else { model.covalent_rad_d3(z) };
    let reach = if v1 { CUTOFF_CN } else { model.reach(CUTOFF_CN) };
    CnModel::new(Erf::D4, move |i: usize, j: usize| rad(mol.numbers[i]) + rad(mol.numbers[j]))
        .with_cutoff(Cutoff::Squared(reach.powi(2)))
        .with_min_r2(1e-12)
        .with_switch(Smooth(move |r: f64| if v1 { (1.0, 0.0) } else { model.cut(r, CUTOFF_CN) }))
        .skipping_zero_weights()
}

/// revD4 coordination numbers (erf, 7.5, Rc = R_A + R_B, cutoff factor; [`revd4_cn`]).
pub fn coordination_numbers(model: &Model, mol: &Molecule) -> Vec<f64> {
    revd4_cn(model, mol).compute(&mol.positions)
}

/// Adds sum_A dE/dCN_A dCN_A/dR (revD4 CN) to `g`.
pub fn add_cn_gradient(model: &Model, mol: &Molecule, dedcn: &[f64], g: &mut [[f64; 3]]) {
    revd4_cn(model, mol).add_gradient(&mol.positions, dedcn, g);
}

/// The dampings of a version: 2.0.1's screened ones, or v1.1's rational two-body damping (s8,
/// a1, a2) and zero-damped three-body term (s9, alpha/3).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Damping {
    Screened,
    Dftd4 { s8: f64, a1: f64, a2: f64, s9: f64, alp3: f64 },
}

/// v1.1's cutoffs (bohr): two-body and three-body.
pub const CUTOFF_2B_V1_1: f64 = 60.0;
pub const CUTOFF_3B_V1_1: f64 = 20.0;

#[inline]
fn pow3(x: f64) -> f64 {
    (x * x) * x
}
#[inline]
fn pow4(x: f64) -> f64 {
    let x2 = x * x;
    x2 * x2
}
#[inline]
fn pow6(x: f64) -> f64 {
    let x3 = pow3(x);
    x3 * x3
}
#[inline]
fn pow8(x: f64) -> f64 {
    let x4 = pow4(x);
    x4 * x4
}

/// Screened damping radius of a pair and its R derivative.
#[inline]
pub fn reff(r: f64, rho: f64) -> (f64, f64) {
    let arg = -A3 * (r - A4 * rho);
    let a = A1 * rho + A2;
    (a * 0.5 * (1.0 + erf(arg)), a * (-A3) * (-arg * arg).exp() / SQRTPI)
}

/// revD4 of one geometry.
#[derive(Clone, Debug)]
pub struct RevD4 {
    pub numbers: Vec<usize>,
    pub cn: Vec<f64>,
    /// K_AB (C6_AB = n_A(q_A) n_B(q_B) K_AB) and dK_AB/dCN_A (row-major nat x nat).
    pub kmat: Vec<f64>,
    pub dkmat: Vec<f64>,
    /// sqrt(3 r4r2_A r4r2_B) ingredients: r4r2 per atom.
    pub r4r2: Vec<f64>,
    /// Two-body damping factors s6/(R + reff)^6 * cut (nat x nat; v1.1: s6/(R^6 + R0^6) +
    /// s8 rr/(R^8 + R0^8) of dftd4).
    pub f6: Vec<f64>,
    pub damping: Damping,
}

impl RevD4 {
    pub fn new(model: &Model, mol: &Molecule) -> Result<Self> {
        Self::with_cn(model, mol, coordination_numbers(model, mol))
    }

    /// [`RevD4::new`], pair rows in parallel when `parallel` (same result).
    pub fn new_par(model: &Model, mol: &Molecule, parallel: bool) -> Result<Self> {
        Self::with_cn_par(model, mol, coordination_numbers(model, mol), parallel)
    }

    /// revD4 with given coordination numbers `cn` (instead of those of the geometry).
    pub fn with_cn(model: &Model, mol: &Molecule, cn: Vec<f64>) -> Result<Self> {
        Self::with_cn_par(model, mol, cn, false)
    }

    /// [`RevD4::with_cn`], pair rows in parallel when `parallel` (same result).
    pub fn with_cn_par(model: &Model, mol: &Molecule, cn: Vec<f64>, parallel: bool) -> Result<Self> {
        let nat = mol.nat();
        if model.version == Version::V1_1 {
            // dftd4's rational damping as `tblite_disp_d4` evaluates its pair energies
            let model4 = revd4_model(&mol.numbers, CHARGES_V1_1)?;
            let (kmat, dkmat) = k_matrices(&model4, &cn, CHARGES_V1_1.refq, parallel);
            let r4r2: Vec<f64> = (0..nat).map(|i| model4.r4r2_atom(i)).collect();
            let g = &crate::params::gxtb_v1_1().glob;
            let (s6, s8, a1, a2) = (1.0, g[19], g[9], 0.0);
            let c2 = CUTOFF_2B_V1_1 * CUTOFF_2B_V1_1;
            let [f6] = crate::parallel::symmetric(nat, parallel, |i, j| {
                let r2 = norm2_sq(sub3(mol.positions[i], mol.positions[j]));
                if i == j || r2 > c2 || r2 < f64::EPSILON {
                    return [0.0];
                }
                let rr = 3.0 * r4r2[i] * r4r2[j];
                let r0 = a1 * rr.sqrt() + a2;
                let t6 = 1.0 / (pow3(r2) + pow6(r0));
                let t8 = 1.0 / (pow4(r2) + pow8(r0));
                [s6 * t6 + s8 * rr * t8]
            });
            let damping = Damping::Dftd4 { s8, a1, a2, s9: reference::S9, alp3: reference::ALP / 3.0 };
            return Ok(Self { numbers: mol.numbers.clone(), cn, kmat, dkmat, r4r2, f6, damping });
        }
        let model4 = revd4_model(&mol.numbers, CHARGES)?;
        let (kmat, dkmat) = k_matrices(&model4, &cn, CHARGES.refq, parallel);
        let r4r2: Vec<f64> = (0..nat).map(|i| model4.r4r2_atom(i)).collect();
        let reach2 = model.reach(CUTOFF_2B).powi(2);
        let [f6] = crate::parallel::symmetric(nat, parallel, |i, j| {
            let r2 = norm2_sq(sub3(mol.positions[i], mol.positions[j]));
            if i == j || r2 > reach2 || r2 < 1e-12 {
                return [0.0];
            }
            let r = r2.sqrt();
            let rho = (3.0 * r4r2[i] * r4r2[j]).sqrt();
            let re = reff(r, rho).0;
            [S6 / (r + re).powi(6) * model.cut(r, CUTOFF_2B).0]
        });
        Ok(Self { numbers: mol.numbers.clone(), cn, kmat, dkmat, r4r2, f6, damping: Damping::Screened })
    }

    /// Two-body energy and dE/dq_A at atomic charges `q`.
    pub fn two_body(&self, q: &[f64]) -> (f64, Vec<f64>) {
        self.two_body_par(q, false)
    }

    /// [`RevD4::two_body`], atom rows in fixed chunks (in parallel when `parallel`).
    pub fn two_body_par(&self, q: &[f64], parallel: bool) -> (f64, Vec<f64>) {
        let nat = self.numbers.len();
        let nd: Vec<(f64, f64)> = (0..nat).map(|i| charge_function(self.numbers[i], q[i])).collect();
        crate::parallel::reduce_rows(
            nat,
            parallel,
            || (0.0, vec![0.0; nat]),
            |i, (e, v)| {
                for j in 0..i {
                    let kf = self.kmat[i * nat + j] * self.f6[i * nat + j];
                    if kf == 0.0 {
                        continue;
                    }
                    *e -= nd[i].0 * nd[j].0 * kf;
                    v[i] -= nd[i].1 * nd[j].0 * kf;
                    v[j] -= nd[i].0 * nd[j].1 * kf;
                }
            },
            |a, b| {
                a.0 += b.0;
                crate::parallel::add_into(&mut a.1, &b.1);
            },
        )
    }

    /// Change of the two-body potential dE/dq_A for atomic charge changes `dq` at charges `q`.
    pub fn two_body_hessian_vec(&self, q: &[f64], dq: &[f64]) -> Vec<f64> {
        let nat = self.numbers.len();
        let nd: Vec<(f64, f64, f64)> = (0..nat)
            .map(|i| {
                let (n, dn) = charge_function(self.numbers[i], q[i]);
                (n, dn, charge_function_d2(self.numbers[i], q[i]))
            })
            .collect();
        let mut v = vec![0.0; nat];
        for i in 0..nat {
            for j in 0..i {
                let kf = self.kmat[i * nat + j] * self.f6[i * nat + j];
                if kf == 0.0 {
                    continue;
                }
                v[i] -= (nd[i].2 * nd[j].0 * dq[i] + nd[i].1 * nd[j].1 * dq[j]) * kf;
                v[j] -= (nd[i].0 * nd[j].2 * dq[j] + nd[i].1 * nd[j].1 * dq[i]) * kf;
            }
        }
        v
    }

    /// Explicit two-body gradient at fixed charges: geometry to `g`, returns dE/dCN (revD4).
    pub fn two_body_gradient(&self, model: &Model, mol: &Molecule, q: &[f64], g: &mut [[f64; 3]]) -> Vec<f64> {
        self.two_body_gradient_par(model, mol, q, g, false)
    }

    /// [`RevD4::two_body_gradient`], atom rows in fixed chunks (in parallel when `parallel`).
    pub fn two_body_gradient_par(
        &self,
        model: &Model,
        mol: &Molecule,
        q: &[f64],
        g_out: &mut [[f64; 3]],
        parallel: bool,
    ) -> Vec<f64> {
        let nat = self.numbers.len();
        let nq: Vec<f64> = (0..nat).map(|i| charge_function(self.numbers[i], q[i]).0).collect();
        if let Damping::Dftd4 { s8, a1, a2, .. } = self.damping {
            return self.two_body_gradient_dftd4(mol, &nq, (s8, a1, a2), g_out, parallel);
        }
        let reach2 = model.reach(CUTOFF_2B).powi(2);
        let (gs, dedcn) = crate::parallel::reduce_rows(
            nat,
            parallel,
            || (vec![[0.0; 3]; nat], vec![0.0; nat]),
            |i, (g, dedcn)| {
                for j in 0..i {
                    let v = sub3(mol.positions[i], mol.positions[j]);
                    let r2 = norm2_sq(v);
                    if r2 > reach2 || r2 < 1e-12 {
                        continue;
                    }
                    let nn = nq[i] * nq[j];
                    let f6 = self.f6[i * nat + j];
                    dedcn[i] -= nn * self.dkmat[i * nat + j] * f6;
                    dedcn[j] -= nn * self.dkmat[j * nat + i] * f6;
                    let r = r2.sqrt();
                    let rho = (3.0 * self.r4r2[i] * self.r4r2[j]).sqrt();
                    let (re, dre) = reff(r, rho);
                    let (sw, dsw) = model.cut(r, CUTOFF_2B);
                    let base = S6 / (r + re).powi(6);
                    let dbase = -6.0 * S6 / (r + re).powi(7) * (1.0 + dre);
                    let df = dbase * sw + base * dsw;
                    let f = -nn * self.kmat[i * nat + j] * df / r;
                    for x in 0..3 {
                        g[i][x] += f * v[x];
                        g[j][x] -= f * v[x];
                    }
                }
            },
            |a, b| {
                crate::parallel::add_into(a.0.as_flattened_mut(), b.0.as_flattened());
                crate::parallel::add_into(&mut a.1, &b.1);
            },
        );
        crate::parallel::add_into(g_out.as_flattened_mut(), gs.as_flattened());
        dedcn
    }

    /// v1.1's two-body gradient at fixed charges (the 1.1.0 program's rational damping in its
    /// order): geometry to `g_out`, returns dE/dCN (revD4).
    fn two_body_gradient_dftd4(
        &self,
        mol: &Molecule,
        nq: &[f64],
        (s8, a1, a2): (f64, f64, f64),
        g_out: &mut [[f64; 3]],
        parallel: bool,
    ) -> Vec<f64> {
        let nat = self.numbers.len();
        let s6 = 1.0;
        let (gs, dedcn) = crate::parallel::reduce_rows(
            nat,
            parallel,
            || (vec![[0.0; 3]; nat], vec![0.0; nat]),
            |i, (g, dedcn)| {
                for j in 0..i {
                    let f = self.f6[i * nat + j];
                    if f == 0.0 {
                        continue;
                    }
                    let nn = nq[i] * nq[j];
                    dedcn[i] -= nn * self.dkmat[i * nat + j] * f;
                    dedcn[j] -= nn * self.dkmat[j * nat + i] * f;
                    let v = sub3(mol.positions[i], mol.positions[j]);
                    let r2 = norm2_sq(v);
                    let rr = 3.0 * self.r4r2[i] * self.r4r2[j];
                    let r0 = a1 * rr.sqrt() + a2;
                    let (r6, r8) = (r2 * r2 * r2, r2 * r2 * r2 * r2);
                    let (t6, t8) = (1.0 / (r6 + r0.powi(6)), 1.0 / (r8 + r0.powi(8)));
                    // d f / d r divided by r
                    let dfr = -6.0 * s6 * r2 * r2 * t6 * t6 - 8.0 * s8 * rr * r6 * t8 * t8;
                    let gg = -nn * self.kmat[i * nat + j] * dfr;
                    for x in 0..3 {
                        g[i][x] += gg * v[x];
                        g[j][x] -= gg * v[x];
                    }
                }
            },
            |a, b| {
                crate::parallel::add_into(a.0.as_flattened_mut(), b.0.as_flattened());
                crate::parallel::add_into(&mut a.1, &b.1);
            },
        );
        crate::parallel::add_into(g_out.as_flattened_mut(), gs.as_flattened());
        dedcn
    }

    /// Three-body energy (C6 at q = 0) and, with `grad`, its geometry gradient (added) and
    /// dE/dCN (returned).
    pub fn three_body(&self, model: &Model, mol: &Molecule, grad: Option<&mut [[f64; 3]]>) -> (f64, Vec<f64>) {
        self.three_body_par(model, mol, grad, false)
    }

    /// [`RevD4::three_body`] with the triples in fixed chunks of the first atom, in parallel on
    /// the current rayon pool when `parallel` (chunk results summed in order).
    pub fn three_body_par(
        &self,
        model: &Model,
        mol: &Molecule,
        grad: Option<&mut [[f64; 3]]>,
        parallel: bool,
    ) -> (f64, Vec<f64>) {
        match self.damping {
            Damping::Screened => self.three_body_with(model, mol, grad, parallel, triple),
            Damping::Dftd4 { s9, alp3, .. } => {
                self.three_body_with(model, mol, grad, parallel, |a, b, c, w| triple_zero(a, b, c, w, s9, alp3))
            }
        }
    }

    /// [`RevD4::three_body_par`] with the triple kernel `triple`.
    fn three_body_with(
        &self,
        model: &Model,
        mol: &Molecule,
        grad: Option<&mut [[f64; 3]]>,
        parallel: bool,
        triple: impl Fn(&Pair3, &Pair3, &Pair3, bool) -> Triple + Sync + Send,
    ) -> (f64, Vec<f64>) {
        let nat = self.numbers.len();
        let tab = self.pair_table(model, mol, parallel);
        let want = grad.is_some();
        let parts = axtb_base::threads::map_chunks(nat, parallel, |rows| {
            let mut e = 0.0;
            let mut dedcn = if want { vec![0.0; nat] } else { Vec::new() };
            let mut gl = if want { vec![[0.0; 3]; nat] } else { Vec::new() };
            // per row i: dE/dR_ik and dE/dC6_ik summed over j
            let mut acc_r = vec![0.0; nat];
            let mut acc_h = vec![0.0; nat];
            for i in rows {
                let ri = tri(i);
                if want {
                    acc_r[..i].iter_mut().for_each(|x| *x = 0.0);
                    acc_h[..i].iter_mut().for_each(|x| *x = 0.0);
                }
                for j in 0..i {
                    let pij = &tab[ri + j];
                    if pij.t == 0.0 {
                        continue;
                    }
                    let rj = tri(j);
                    let (mut sr_ij, mut sh_ij) = (0.0, 0.0);
                    for k in 0..j {
                        let (pik, pjk) = (&tab[ri + k], &tab[rj + k]);
                        if pik.t == 0.0 || pjk.t == 0.0 {
                            continue;
                        }
                        let t = triple(pij, pik, pjk, want);
                        e += t.e;
                        if !want {
                            continue;
                        }
                        sr_ij += t.dr[0];
                        sh_ij += t.h[0];
                        acc_r[k] += t.dr[1];
                        acc_h[k] += t.h[1];
                        for x in 0..3 {
                            gl[j][x] += t.dr[2] * pjk.u[x];
                            gl[k][x] -= t.dr[2] * pjk.u[x];
                        }
                        dedcn[j] += t.h[2] * pjk.dci;
                        dedcn[k] += t.h[2] * pjk.dcj;
                    }
                    if want {
                        for x in 0..3 {
                            gl[i][x] += sr_ij * pij.u[x];
                            gl[j][x] -= sr_ij * pij.u[x];
                        }
                        dedcn[i] += sh_ij * pij.dci;
                        dedcn[j] += sh_ij * pij.dcj;
                    }
                }
                if want {
                    for k in 0..i {
                        let pik = &tab[ri + k];
                        for x in 0..3 {
                            gl[i][x] += acc_r[k] * pik.u[x];
                            gl[k][x] -= acc_r[k] * pik.u[x];
                        }
                        dedcn[i] += acc_h[k] * pik.dci;
                        dedcn[k] += acc_h[k] * pik.dcj;
                    }
                }
            }
            (e, dedcn, gl)
        });
        let mut e = 0.0;
        let mut dedcn = vec![0.0; nat];
        let mut gsum = vec![[0.0; 3]; nat];
        for (pe, pd, pg) in &parts {
            e += pe;
            if want {
                for a in 0..nat {
                    dedcn[a] += pd[a];
                    for x in 0..3 {
                        gsum[a][x] += pg[a][x];
                    }
                }
            }
        }
        if let Some(g) = grad {
            for (a, b) in g.iter_mut().zip(&gsum) {
                for x in 0..3 {
                    a[x] += b[x];
                }
            }
        }
        (e, dedcn)
    }

    /// [`RevD4::three_body`] restricted to the triples that contain atom `only` (all with
    /// `None`).
    pub fn three_body_of(
        &self,
        model: &Model,
        mol: &Molecule,
        grad: Option<&mut [[f64; 3]]>,
        only: Option<usize>,
    ) -> (f64, Vec<f64>) {
        let Some(a) = only else {
            return self.three_body(model, mol, grad);
        };
        let nat = self.numbers.len();
        let tab = self.pair_table(model, mol, false);
        let want = grad.is_some();
        let mut e = 0.0;
        let mut dedcn = vec![0.0; nat];
        let mut gl = vec![[0.0; 3]; nat];
        for p in 0..nat {
            for q in 0..p {
                if p == a || q == a {
                    continue;
                }
                // the triple (i > j > k) of {a, p, q}
                let (i, j, k) = if a > p {
                    (a, p, q)
                } else if a > q {
                    (p, a, q)
                } else {
                    (p, q, a)
                };
                let (pij, pik, pjk) = (&tab[tri(i) + j], &tab[tri(i) + k], &tab[tri(j) + k]);
                if pij.t == 0.0 || pik.t == 0.0 || pjk.t == 0.0 {
                    continue;
                }
                let t = match self.damping {
                    Damping::Screened => triple(pij, pik, pjk, want),
                    Damping::Dftd4 { s9, alp3, .. } => triple_zero(pij, pik, pjk, want, s9, alp3),
                };
                e += t.e;
                if !want {
                    continue;
                }
                for (m, (x, y, pr)) in [(i, j, pij), (i, k, pik), (j, k, pjk)].into_iter().enumerate() {
                    for c in 0..3 {
                        gl[x][c] += t.dr[m] * pr.u[c];
                        gl[y][c] -= t.dr[m] * pr.u[c];
                    }
                    dedcn[x] += t.h[m] * pr.dci;
                    dedcn[y] += t.h[m] * pr.dcj;
                }
            }
        }
        if let Some(g) = grad {
            for (x, y) in g.iter_mut().zip(&gl) {
                for c in 0..3 {
                    x[c] += y[c];
                }
            }
        }
        (e, dedcn)
    }

    /// The pair data of the three-body term for every pair j < i (index `tri(i) + j`), rows in
    /// parallel when `parallel`.
    fn pair_table(&self, model: &Model, mol: &Molecule, parallel: bool) -> Vec<Pair3> {
        let nat = self.numbers.len();
        let n0: Vec<f64> = (0..nat).map(|i| charge_function(self.numbers[i], 0.0).0).collect();
        if let Damping::Dftd4 { a1, a2, alp3, .. } = self.damping {
            // v1.1: undamped 1/R^3 per pair, dftd4's zero damping per triple, (R0/R)^(alpha/3)
            let c2 = CUTOFF_3B_V1_1 * CUTOFF_3B_V1_1;
            let rows = axtb_base::threads::map(nat, parallel, |i| {
                (0..i)
                    .map(|j| {
                        let v = sub3(mol.positions[i], mol.positions[j]);
                        let r2 = norm2_sq(v);
                        if r2 > c2 || r2 < f64::EPSILON {
                            return Pair3::default();
                        }
                        let r = r2.sqrt();
                        let r0 = a1 * (3.0 * self.r4r2[i] * self.r4r2[j]).sqrt() + a2;
                        let nn = n0[i] * n0[j];
                        let c6 = nn * self.kmat[i * nat + j];
                        Pair3 {
                            r,
                            t: 1.0 / (r2 * r),
                            dt: -3.0 / (r2 * r2),
                            s: c6.abs().sqrt(),
                            ic: if c6 != 0.0 { 1.0 / c6 } else { 0.0 },
                            dci: nn * self.dkmat[i * nat + j],
                            dcj: nn * self.dkmat[j * nat + i],
                            u: [v[0] / r, v[1] / r, v[2] / r],
                            x: (r0 / r).powf(alp3),
                        }
                    })
                    .collect::<Vec<_>>()
            });
            return rows.concat();
        }
        let reach = model.reach(CUTOFF_3B);
        let rows = axtb_base::threads::map(nat, parallel, |i| {
            (0..i)
                .map(|j| {
                    let v = sub3(mol.positions[i], mol.positions[j]);
                    let r = norm2_sq(v).sqrt();
                    if r > reach {
                        return Pair3::default();
                    }
                    let rho = (3.0 * self.r4r2[i] * self.r4r2[j]).sqrt();
                    let (re, dre) = reff(r, rho);
                    let (sw, dsw) = model.cut(r, CUTOFF_3B);
                    let t = 1.0 / (r + re).powi(3);
                    let dt = -3.0 / (r + re).powi(4) * (1.0 + dre);
                    let nn = n0[i] * n0[j];
                    let c6 = nn * self.kmat[i * nat + j];
                    Pair3 {
                        r,
                        t: t * sw,
                        dt: dt * sw + t * dsw,
                        s: c6.abs().sqrt(),
                        ic: if c6 != 0.0 { 1.0 / c6 } else { 0.0 },
                        dci: nn * self.dkmat[i * nat + j],
                        dcj: nn * self.dkmat[j * nat + i],
                        u: [v[0] / r, v[1] / r, v[2] / r],
                        x: 0.0,
                    }
                })
                .collect::<Vec<_>>()
        });
        rows.concat()
    }
}

/// Start of row `i` of the lower-triangle pair table.
#[inline]
fn tri(i: usize) -> usize {
    i * (i.max(1) - 1) / 2
}

/// Pair data of the three-body term: distance, damping 1/(R + reff)^3 times the cutoff factor
/// and its R derivative, sqrt|C6| (q = 0), 1/C6, dC6/dCN of the first and the second atom, and
/// the unit vector from the second atom to the first.
#[derive(Clone, Copy, Debug, Default)]
struct Pair3 {
    r: f64,
    t: f64,
    dt: f64,
    s: f64,
    ic: f64,
    dci: f64,
    dcj: f64,
    u: [f64; 3],
    /// (R0/R)^(alpha/3) of dftd4's zero damping (v1.1).
    x: f64,
}

/// One triple's energy and (with `want`) dE/dR and dE/dC6 of its pairs (ij, ik, jk).
struct Triple {
    e: f64,
    dr: [f64; 3],
    h: [f64; 3],
}

#[inline]
fn triple(pij: &Pair3, pik: &Pair3, pjk: &Pair3, want: bool) -> Triple {
    let c9 = S9 * pij.s * pik.s * pjk.s;
    let (rij, rik, rjk) = (pij.r, pik.r, pjk.r);
    let (a, b, c) = (rij * rij, rik * rik, rjk * rjk);
    let (p1, p2, p3) = (a + b - c, a + c - b, b + c - a);
    let inv = 1.0 / (a * b * c);
    let pp = p1 * p2 * p3;
    let ang = 0.375 * pp * inv + 1.0;
    let ttt = pij.t * pik.t * pjk.t;
    let e = c9 * ttt * ang;
    if !want {
        return Triple { e, dr: [0.0; 3], h: [0.0; 3] };
    }
    // angular derivatives with respect to a, b, c (squared distances)
    let dpa = p2 * p3 + p1 * p3 - p1 * p2;
    let dpb = p2 * p3 - p1 * p3 + p1 * p2;
    let dpc = -p2 * p3 + p1 * p3 + p1 * p2;
    let da = 0.375 * inv * (dpa - pp / a);
    let db = 0.375 * inv * (dpb - pp / b);
    let dc = 0.375 * inv * (dpc - pp / c);
    let dr = [
        c9 * (pij.dt * pik.t * pjk.t * ang + ttt * da * 2.0 * rij),
        c9 * (pij.t * pik.dt * pjk.t * ang + ttt * db * 2.0 * rik),
        c9 * (pij.t * pik.t * pjk.dt * ang + ttt * dc * 2.0 * rjk),
    ];
    // dc9/dC6_xy = c9/(2 C6_xy)
    let he = 0.5 * c9 * ttt * ang;
    Triple { e, dr, h: [he * pij.ic, he * pik.ic, he * pjk.ic] }
}

/// One triple of v1.1's three-body term: the undamped 1/R^3 of the pairs with dftd4's zero
/// damping f = 1/(1 + 6 prod_pairs (R0/R)^(alpha/3)) of the triple.
#[inline]
fn triple_zero(pij: &Pair3, pik: &Pair3, pjk: &Pair3, want: bool, s9: f64, alp3: f64) -> Triple {
    let c9 = s9 * pij.s * pik.s * pjk.s;
    let (rij, rik, rjk) = (pij.r, pik.r, pjk.r);
    let (a, b, c) = (rij * rij, rik * rik, rjk * rjk);
    let (p1, p2, p3) = (a + b - c, a + c - b, b + c - a);
    let inv = 1.0 / (a * b * c);
    let pp = p1 * p2 * p3;
    let ang = 0.375 * pp * inv + 1.0;
    let ttt = pij.t * pik.t * pjk.t;
    let rpow = pij.x * pik.x * pjk.x;
    let fdmp = 1.0 / (1.0 + 6.0 * rpow);
    let e = c9 * ttt * ang * fdmp;
    if !want {
        return Triple { e, dr: [0.0; 3], h: [0.0; 3] };
    }
    let dpa = p2 * p3 + p1 * p3 - p1 * p2;
    let dpb = p2 * p3 - p1 * p3 + p1 * p2;
    let dpc = -p2 * p3 + p1 * p3 + p1 * p2;
    let da = 0.375 * inv * (dpa - pp / a);
    let db = 0.375 * inv * (dpb - pp / b);
    let dc = 0.375 * inv * (dpc - pp / c);
    // d fdmp / dR_xy = 6 (alpha/3) rpow fdmp^2 / R_xy
    let df = 6.0 * alp3 * rpow * fdmp * fdmp;
    let dr = [
        c9 * ((pij.dt * pik.t * pjk.t * ang + ttt * da * 2.0 * rij) * fdmp + ttt * ang * df / rij),
        c9 * ((pij.t * pik.dt * pjk.t * ang + ttt * db * 2.0 * rik) * fdmp + ttt * ang * df / rik),
        c9 * ((pij.t * pik.t * pjk.dt * ang + ttt * dc * 2.0 * rjk) * fdmp + ttt * ang * df / rjk),
    ];
    let he = 0.5 * c9 * ttt * ang * fdmp;
    Triple { e, dr, h: [he * pij.ic, he * pik.ic, he * pjk.ic] }
}
