//! revD4 dispersion of g-xTB v1.1 (P-Eq. 44), as implemented by the 1.1.0 program (its
//! embedded dftd4 with the `gxtb` reference set, `tblite_disp_d4`), built on
//! [`axtb_d4`]. Verified against the program's two-body and three-body energies to ~1e-18 Eh
//! (reference tests).
//!
//! The model is DFT-D4S-like with g-xTB modifications:
//!
//! - **CN** (`ncoord_erf`): CN_A = sum_B 1/2 [1 + erf(-7.5 (R - Rc)/Rc)], Rc = R_A + R_B with
//!   the dftd4 3.x covalent radii ([`D4_RCOV`]); no electronegativity factor, pairs beyond
//!   30 bohr skipped.
//! - **Reference systems**: dftd4's polarizabilities and secondary systems, but the program's
//!   own reference CNs ([`D4_REFCOVCN`] for the weights, [`D4_REFCN`] counted for the number of
//!   Gaussians), reference charges [`D4_REFQ`] and secondary-system charges [`D4_REFH`]. The
//!   hydrogen subtraction scales the secondary polarizability with the tanh charge function
//!   below, alpha_ref = max(0, ascale (alpha - hcount sscale secaiw zeta(h))), zeta(h) =
//!   (a + b tanh(c h + d))/(a + b tanh(d)) with the parameters of the secondary element.
//! - **Pair-specific Gaussian weights** (D4S, `weight_references_pair`): the weight of
//!   reference r of atom A in the pair AB is
//!   w_r(A, B) = n_A(q_A)/n_A(qref_r) * sum_g exp(-g wf_AB (CN_A - CN_r)^2) / norm,
//!   g = 1..ngw_r, with the pair factors wf_AB = [`K3_PAIR`](Z_A, Z_B) and the charge
//!   function n_A(q) = a + b tanh(c q + d) of element A ([`D4_TANH`]); q are the Mulliken
//!   atomic charges of the SCF.
//! - **C6**: C6_AB = sum_rs w_r(A, B) w_s(B, A) C6ref_rs (Casimir-Polder integration of the
//!   reference polarizabilities by `axtb_d4`).
//! - **Two-body**: rational damping with s6 = 1, s8 = glob(20), a1 = glob(10), a2 = 0,
//!   C8 from r4/r2 as in D4, pairs beyond 60 bohr skipped; evaluated with the SCF charges.
//! - **Three-body**: ATM with zero damping (alpha = 16), s9 = 1.75, C6 at q = 0, triples
//!   with all distances within 20 bohr; evaluated once (not self-consistent).
//!
//! Since the charge enters only through n_A(q_A), C6_AB(q) = n_A(q_A) n_B(q_B) K_AB with a
//! geometry-only K_AB, which [`RevD4`] precomputes.

use crate::data::{D4_RCOV, D4_REFCN, D4_REFCOVCN, D4_REFH, D4_REFQ, D4_TANH, K3_PAIR};
use crate::params::GxtbParams;
use axtb::molecule::{norm2_sq, sub3};
use axtb::{Error, Molecule, Result};
use axtb_d4::data::{ALPHAIW, ASCALE, HCOUNT, NFREQ, REFN, REFSYS, SECAIW, SSCALE};
use axtb_d4::{
    cn_count, ChargeModel, D4Cache, D4Dispersion, D4Model, ElementReferences, RationalDamping, RealspaceCutoff, MAX_REF,
};

/// Three-body scaling s9.
pub const S9: f64 = 1.75;
/// Exponent of the three-body zero damping.
pub const ALP: f64 = 16.0;
/// Real-space cutoffs (bohr): CN 30, two-body 60, three-body 20, all hard.
pub const CUTOFF: RealspaceCutoff = RealspaceCutoff { cn: 30.0, disp2: 60.0, disp3: 20.0, width2: 0.0, width3: 0.0 };
/// Largest element of the program's tables of the CN weighting factors.
const MAX_Z: usize = 103;

/// Damping parameters of the program (`new_d4_dispersion` called from `scf_`).
pub fn damping(param: &GxtbParams) -> RationalDamping {
    RationalDamping { s6: 1.0, s8: param.glob[19], s9: S9, a1: param.glob[9], a2: 0.0, alp: ALP }
}

/// tanh charge function n(q) = a + b tanh(c q + d) of element `z` and its derivative.
pub fn charge_function(z: usize, q: f64) -> (f64, f64) {
    let p = &D4_TANH[4 * (z - 1)..4 * z];
    let t = (q * p[2] + p[3]).tanh();
    (p[0] + p[1] * t, p[1] * p[2] * (1.0 - t * t))
}

pub(crate) fn nref(z: usize) -> usize {
    REFN[z - 1] as usize
}

/// Number of Gaussian functions per reference (`set_refgw`): with icn = nint(refcn) capped at
/// 19 and c(icn) the number of references with that icn (plus one for icn = 0),
/// ngw = c (c + 1)/2.
pub fn reference_ngw(z: usize) -> Vec<u32> {
    let n = nref(z);
    let icn = |r: usize| (D4_REFCN[7 * (z - 1) + r].round() as i64).clamp(0, 19) as usize;
    let mut cnc = [0u32; 20];
    cnc[0] = 1;
    for r in 0..n {
        cnc[icn(r)] += 1;
    }
    (0..n).map(|r| cnc[icn(r)] * (cnc[icn(r)] + 1) / 2).collect()
}

/// Reference polarizabilities (`set_refalpha_gxtb`): dftd4's with the hydrogen contributions
/// removed, the secondary system scaled by zeta(h) = n_s(h)/n_s(0) of its element s.
pub fn reference_alpha(z: usize) -> Vec<[f64; NFREQ]> {
    (0..nref(z))
        .map(|r| {
            let k = 7 * (z - 1) + r;
            let is = REFSYS[z - 1][r] as usize;
            let mut alpha = [0.0; NFREQ];
            if is == 0 {
                return alpha;
            }
            let zeta = charge_function(is, D4_REFH[k]).0 / charge_function(is, 0.0).0;
            let f = SSCALE[is - 1] * zeta;
            for (w, a) in alpha.iter_mut().enumerate() {
                let aiw = SECAIW[is - 1][w] * f;
                *a = (ASCALE[z - 1][r] * (ALPHAIW[z - 1][r][w] - HCOUNT[z - 1][r] * aiw)).max(0.0);
            }
            alpha
        })
        .collect()
}

/// The revD4 reference systems of element `z` for [`D4Model::with_reference_data`].
pub fn element_references(z: usize) -> ElementReferences {
    let n = nref(z);
    let k = 7 * (z - 1);
    ElementReferences {
        cn: D4_REFCOVCN[k..k + n].to_vec(),
        ngw: reference_ngw(z),
        q: D4_REFQ[k..k + n].to_vec(),
        alpha: reference_alpha(z),
    }
}

/// revD4 coordination numbers (no electronegativity factor, 30 bohr cutoff).
pub fn coordination_numbers(mol: &Molecule) -> Vec<f64> {
    let n = mol.nat();
    let cut2 = CUTOFF.cn * CUTOFF.cn;
    let mut cn = vec![0.0; n];
    for i in 0..n {
        for j in 0..i {
            let r2 = norm2_sq(sub3(mol.positions[i], mol.positions[j]));
            if r2 > cut2 || r2 < 1e-12 {
                continue;
            }
            let c = cn_count(D4_RCOV[mol.numbers[i] - 1] + D4_RCOV[mol.numbers[j] - 1], r2.sqrt());
            cn[i] += c;
            cn[j] += c;
        }
    }
    cn
}

/// Charge-independent Gaussian CN weights of the references of an atom (element `z`,
/// coordination number `cn`) for pair factor `wf`, divided by n(qref_r)
/// (`weight_references_pair` without the charge numerator).
pub(crate) fn scaled_cn_weights(z: usize, cn: f64, wf: f64, refq: &[f64]) -> ([f64; MAX_REF], [f64; MAX_REF]) {
    let n = nref(z);
    let k = 7 * (z - 1);
    let ngw = reference_ngw(z);
    // sum_g exp(-g wf d^2) and its derivative with respect to CN, d = CN - CN_r
    let gauss = |r: usize| {
        let d = cn - D4_REFCOVCN[k + r];
        (1..=ngw[r]).fold((0.0, 0.0), |(e, de), g| {
            let x = (-(g as f64 * (wf * d * d))).exp();
            (e + x, de - 2.0 * g as f64 * wf * d * x)
        })
    };
    let (norm, dnorm) = (0..n).map(gauss).fold((0.0, 0.0), |(a, b), (x, y)| (a + x, b + y));
    let inv = 1.0 / norm;
    let maxcn = D4_REFCOVCN[k..k + n].iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut w = [0.0; MAX_REF];
    let mut dw = [0.0; MAX_REF];
    for r in 0..n {
        let (e, de) = gauss(r);
        let mut gwk = e * inv;
        let mut dgwk = (de - e * dnorm * inv) * inv;
        if gwk.is_nan() || gwk.abs() > f64::MAX {
            gwk = if (maxcn - D4_REFCOVCN[k + r]).abs() < 1e-12 { 1.0 } else { 0.0 };
            dgwk = 0.0;
        }
        if !dgwk.is_finite() {
            dgwk = 0.0;
        }
        let den = charge_function(z, refq[k + r]).0;
        w[r] = gwk / den;
        dw[r] = dgwk / den;
    }
    (w, dw)
}

/// revD4 for one molecule and geometry.
#[derive(Clone, Debug)]
pub struct RevD4 {
    disp: D4Dispersion,
    cache: D4Cache,
    numbers: Vec<usize>,
    cn: Vec<f64>,
    /// K_AB (row-major nat x nat): C6_AB = n_A(q_A) n_B(q_B) K_AB.
    kmat: Vec<f64>,
    /// dK_AB/dCN_A (row-major nat x nat).
    dkmat: Vec<f64>,
    /// Damping: s6, s8, a1, a2 and the r4/r2 values per atom.
    damp: RationalDamping,
    r4r2: Vec<f64>,
}

impl RevD4 {
    /// Set up the model for `mol` (positions in bohr).
    pub fn new(param: &GxtbParams, mol: &Molecule) -> Result<Self> {
        let nat = mol.nat();
        if let Some(&z) = mol.numbers.iter().find(|&&z| z == 0 || z > MAX_Z) {
            return Err(Error::UnsupportedElement { method: "revD4", z });
        }
        let model = D4Model::new(&mol.numbers, ChargeModel::Eeq)
            .map_err(|e| Error::Input(e.to_string()))?
            .with_reference_data(element_references);
        let disp = D4Dispersion::new(model, damping(param), CUTOFF);
        let cache = disp.update(&mol.positions);
        let cn = coordination_numbers(mol);
        // pair-specific weights of atom i towards the elements present
        let mut species: Vec<usize> = mol.numbers.clone();
        species.sort_unstable();
        species.dedup();
        let sp: Vec<usize> = mol.numbers.iter().map(|z| species.binary_search(z).unwrap()).collect();
        let w: Vec<Vec<([f64; MAX_REF], [f64; MAX_REF])>> = (0..nat)
            .map(|i| {
                let zi = mol.numbers[i];
                species
                    .iter()
                    .map(|&zj| scaled_cn_weights(zi, cn[i], K3_PAIR[(zj - 1) * MAX_Z + zi - 1], &D4_REFQ))
                    .collect()
            })
            .collect();
        let model = disp.model();
        let mut kmat = vec![0.0; nat * nat];
        let mut dkmat = vec![0.0; nat * nat];
        for i in 0..nat {
            for j in 0..i {
                let ((wi, dwi), (wj, dwj)) = (&w[i][sp[j]], &w[j][sp[i]]);
                let (mut k, mut dki, mut dkj) = (0.0, 0.0, 0.0);
                for a in 0..model.nref_atom(i) {
                    for b in 0..model.nref_atom(j) {
                        let c = model.reference_c6(i, j, a, b);
                        k += wi[a] * wj[b] * c;
                        dki += dwi[a] * wj[b] * c;
                        dkj += wi[a] * dwj[b] * c;
                    }
                }
                kmat[i * nat + j] = k;
                kmat[j * nat + i] = k;
                dkmat[i * nat + j] = dki;
                dkmat[j * nat + i] = dkj;
            }
        }
        let r4r2 = (0..nat).map(|i| model.r4r2_atom(i)).collect();
        Ok(Self { disp, cache, numbers: mol.numbers.clone(), cn, kmat, dkmat, damp: damping(param), r4r2 })
    }

    /// The revD4 coordination numbers.
    pub fn cn(&self) -> &[f64] {
        &self.cn
    }

    /// Pair C6 coefficients (row-major nat x nat, zero diagonal) at atomic charges `q`.
    pub fn c6_matrix(&self, q: &[f64]) -> Vec<f64> {
        let nat = self.numbers.len();
        let n: Vec<f64> = (0..nat).map(|i| charge_function(self.numbers[i], q[i]).0).collect();
        let mut c6 = self.kmat.clone();
        for i in 0..nat {
            for j in 0..nat {
                c6[i * nat + j] *= n[i] * n[j];
            }
        }
        c6
    }

    /// Two-body energy at atomic charges `q` (positive = electron deficient).
    pub fn two_body_energy(&self, q: &[f64]) -> f64 {
        self.two_body_potential(q).0
    }

    /// Two-body energy and its derivative dE/dq_A at atomic charges `q`.
    pub fn two_body_potential(&self, q: &[f64]) -> (f64, Vec<f64>) {
        let nat = self.numbers.len();
        let nd: Vec<(f64, f64)> = (0..nat).map(|i| charge_function(self.numbers[i], q[i])).collect();
        let mut e = 0.0;
        let mut v = vec![0.0; nat];
        for (i, j, f) in self.cache.pairs() {
            let kf = self.kmat[i * nat + j] * f;
            e -= nd[i].0 * nd[j].0 * kf;
            v[i] -= nd[i].1 * nd[j].0 * kf;
            v[j] -= nd[i].0 * nd[j].1 * kf;
        }
        (e, v)
    }

    /// Gradient of the two-body energy at fixed atomic charges `q`: adds the explicit
    /// geometry derivative to `grad` and returns dE/dCN_A (revD4 CN); dE/dq is
    /// [`two_body_potential`](Self::two_body_potential).
    pub fn two_body_gradient(&self, positions: &[[f64; 3]], q: &[f64], grad: &mut [[f64; 3]]) -> Vec<f64> {
        let nat = self.numbers.len();
        let nq: Vec<f64> = (0..nat).map(|i| charge_function(self.numbers[i], q[i]).0).collect();
        let mut dedcn = vec![0.0; nat];
        let d = &self.damp;
        for (i, j, f) in self.cache.pairs() {
            let nn = nq[i] * nq[j];
            dedcn[i] -= nn * self.dkmat[i * nat + j] * f;
            dedcn[j] -= nn * self.dkmat[j * nat + i] * f;
            let v = [
                positions[i][0] - positions[j][0],
                positions[i][1] - positions[j][1],
                positions[i][2] - positions[j][2],
            ];
            let r2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            let rr = 3.0 * self.r4r2[i] * self.r4r2[j];
            let r0 = d.a1 * rr.sqrt() + d.a2;
            let (r6, r8) = (r2 * r2 * r2, r2 * r2 * r2 * r2);
            let (t6, t8) = (1.0 / (r6 + r0.powi(6)), 1.0 / (r8 + r0.powi(8)));
            // d f / d r divided by r
            let dfr = -6.0 * d.s6 * r2 * r2 * t6 * t6 - 8.0 * d.s8 * rr * r6 * t8 * t8;
            let g = -nn * self.kmat[i * nat + j] * dfr;
            for k in 0..3 {
                grad[i][k] += g * v[k];
                grad[j][k] -= g * v[k];
            }
        }
        dedcn
    }

    /// Gradient of the three-body energy (C6 at q = 0): adds the geometric part to `grad`
    /// and returns dE/dCN_A (revD4 CN).
    pub fn three_body_gradient(&self, grad: &mut [[f64; 3]]) -> Vec<f64> {
        let nat = self.numbers.len();
        let c6 = self.c6_matrix(&vec![0.0; nat]);
        let n0: Vec<f64> = (0..nat).map(|i| charge_function(self.numbers[i], 0.0).0).collect();
        let mut dc6 = self.dkmat.clone();
        for i in 0..nat {
            for j in 0..nat {
                dc6[i * nat + j] *= n0[i] * n0[j];
            }
        }
        let mut e = vec![0.0; nat];
        let mut dedcn = vec![0.0; nat];
        self.disp.three_body_c6_gradient(&self.cache, &c6, &dc6, &mut e, grad, &mut dedcn);
        dedcn
    }

    /// Adds sum_A dE/dCN_A dCN_A/dR (revD4 CN) to `grad`.
    pub fn add_cn_gradient(mol: &Molecule, dedcn: &[f64], grad: &mut [[f64; 3]]) {
        let n = mol.nat();
        let cut2 = CUTOFF.cn * CUTOFF.cn;
        for i in 0..n {
            for j in 0..i {
                let v = sub3(mol.positions[i], mol.positions[j]);
                let r2 = norm2_sq(v);
                if r2 > cut2 || r2 < 1e-12 {
                    continue;
                }
                let r = r2.sqrt();
                let dc = axtb_d4::cn_dcount(D4_RCOV[mol.numbers[i] - 1] + D4_RCOV[mol.numbers[j] - 1], r);
                let g = (dedcn[i] + dedcn[j]) * dc / r;
                for k in 0..3 {
                    grad[i][k] += g * v[k];
                    grad[j][k] -= g * v[k];
                }
            }
        }
    }

    /// Three-body (ATM) energy, C6 at q = 0.
    pub fn three_body_energy(&self) -> f64 {
        let nat = self.numbers.len();
        let c6 = self.c6_matrix(&vec![0.0; nat]);
        let mut e = vec![0.0; nat];
        self.disp.three_body_c6(&self.cache, &c6, &mut e);
        e.iter().sum()
    }
}
