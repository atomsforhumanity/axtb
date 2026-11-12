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
//!
//! The mechanisms that are dftd4's come from [`axtb_d4`]: the hydrogen-subtracted reference
//! polarizabilities ([`axtb_d4::reference_alpha`], with the tanh ratio as the secondary
//! scaling), the Gaussian counts ([`axtb_d4::reference_ngw`]), the reference C6 and their
//! contraction with the weights ([`D4Model::pair_c6_from_weights`]), the two-body pair list and
//! the three-body kernel ([`D4Dispersion`]). The tables, the CN, the pair-specific weights and
//! the two-body gradient (its damping derivative in its own order, which differs from
//! `axtb_d4`'s in the last bits) are g-xTB's and stay here; 2.0.1 ([`crate::v2::disp`]) uses
//! the same reference data with its own [`ReferenceCharges`].

use crate::data::{D4_RCOV, D4_REFCN, D4_REFCOVCN, D4_REFH, D4_REFQ, D4_TANH, K3_PAIR};
use crate::params::GxtbParams;
use axtb::{Error, Molecule, Result};
use axtb_base::ncoord::{CnModel, Cutoff, Erf};
use axtb_d4::data::{NFREQ, REFN};
use axtb_d4::{
    ChargeModel, D4Cache, D4Dispersion, D4Model, ElementReferences, RationalDamping, RealspaceCutoff, MAX_REF,
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

/// The reference charges of a revD4 parametrization (the program's tables, 7 entries per
/// element): `refq` of the reference systems (charge scaling of the weights), `refh` of their
/// secondary systems (hydrogen subtraction of the polarizabilities). The rest of the reference
/// data (CNs, polarizabilities, secondary systems) is the same in v1.1 and 2.0.1.
#[derive(Clone, Copy, Debug)]
pub struct ReferenceCharges {
    pub refq: &'static [f64; 826],
    pub refh: &'static [f64; 826],
}

/// The reference charges of g-xTB v1.1 ([`D4_REFQ`], [`D4_REFH`]).
pub const CHARGES_V1_1: ReferenceCharges = ReferenceCharges { refq: &D4_REFQ, refh: &D4_REFH };

/// Number of Gaussian functions per reference (`set_refgw`, [`axtb_d4::reference_ngw`]) from
/// the program's counted reference CNs [`D4_REFCN`].
pub fn reference_ngw(z: usize) -> Vec<u32> {
    let k = 7 * (z - 1);
    axtb_d4::reference_ngw(&D4_REFCN[k..k + nref(z)])
}

/// Reference polarizabilities (`set_refalpha_gxtb`): dftd4's with the hydrogen contributions
/// removed ([`axtb_d4::reference_alpha`]), the secondary system scaled by
/// zeta(h) = n_s(h)/n_s(0) of its element s, h from `refh`. (The product
/// sscale · secaiw · zeta is the same in any order here: the secondary scalings in use are
/// 0, 1/2 and 1.)
pub fn reference_alpha(z: usize, refh: &[f64; 826]) -> Vec<[f64; NFREQ]> {
    let k = 7 * (z - 1);
    let alpha = axtb_d4::reference_alpha(z, |is, r| charge_function(is, refh[k + r]).0 / charge_function(is, 0.0).0);
    alpha[..nref(z)].to_vec()
}

/// The revD4 reference systems of element `z` with the reference charges `charges`, for
/// [`D4Model::with_reference_data`].
pub fn element_references(z: usize, charges: ReferenceCharges) -> ElementReferences {
    let n = nref(z);
    let k = 7 * (z - 1);
    ElementReferences {
        cn: D4_REFCOVCN[k..k + n].to_vec(),
        ngw: reference_ngw(z),
        q: charges.refq[k..k + n].to_vec(),
        alpha: reference_alpha(z, charges.refh),
    }
}

/// The D4 model of `numbers` with the revD4 reference systems ([`element_references`]), for
/// its reference C6 coefficients and r4/r2 values; elements beyond the program's tables
/// (Z > 103) are refused.
pub(crate) fn revd4_model(numbers: &[usize], charges: ReferenceCharges) -> Result<D4Model> {
    if let Some(&z) = numbers.iter().find(|&&z| z == 0 || z > MAX_Z) {
        return Err(Error::UnsupportedElement { method: "revD4", z });
    }
    Ok(D4Model::new(numbers, ChargeModel::Eeq)
        .map_err(|e| Error::Input(e.to_string()))?
        .with_reference_data(|z| element_references(z, charges)))
}

/// K_AB and dK_AB/dCN_A (row-major nat x nat, zero diagonals) of the atoms of `model` at the
/// revD4 coordination numbers `cn`: the C6 contraction ([`D4Model::pair_c6_from_weights`]) of
/// the pair-specific weights of A towards the element of B ([`scaled_cn_weights`], pair factor
/// [`K3_PAIR`], reference charges `refq`) and of B towards A. Rows in parallel when `parallel`
/// (the same result).
pub(crate) fn k_matrices(model: &D4Model, cn: &[f64], refq: &[f64; 826], parallel: bool) -> (Vec<f64>, Vec<f64>) {
    let numbers = model.numbers();
    let nat = numbers.len();
    let mut species: Vec<usize> = numbers.to_vec();
    species.sort_unstable();
    species.dedup();
    let sp: Vec<usize> = numbers.iter().map(|z| species.binary_search(z).unwrap()).collect();
    // the weights of atom i towards each element present
    let w: Vec<Vec<([f64; MAX_REF], [f64; MAX_REF])>> = axtb_base::threads::map(nat, parallel, |i| {
        let zi = numbers[i];
        species.iter().map(|&zj| scaled_cn_weights(zi, cn[i], K3_PAIR[(zj - 1) * MAX_Z + zi - 1], refq)).collect()
    });
    // K_AB and dK_AB/dCN_A, dK_AB/dCN_B of the pairs B < A
    let [mut kmat, dk_low, dk_up] = crate::parallel::lower(nat, parallel, |i, j| {
        if j == i {
            return [0.0; 3];
        }
        let (wi, wj) = (&w[i][sp[j]], &w[j][sp[i]]);
        let (k, dki, dkj) = model.pair_c6_from_weights(i, j, (&wi.0, &wi.1), (&wj.0, &wj.1));
        [k, dki, dkj]
    });
    crate::parallel::mirror_lower(&mut kmat, nat, parallel);
    let mut dkmat = vec![0.0; nat * nat];
    crate::parallel::fill_tiled(&mut dkmat, nat, parallel, |i, j| match j.cmp(&i) {
        std::cmp::Ordering::Less => dk_low[i * nat + j],
        std::cmp::Ordering::Greater => dk_up[j * nat + i],
        std::cmp::Ordering::Equal => 0.0,
    });
    (kmat, dkmat)
}

/// The revD4 coordination number of `mol`: dftd4's count ([`Erf::D4`]) without the
/// electronegativity factor, radii [`D4_RCOV`], pairs with R² > 30² or R² < 1e-12 bohr² skipped
/// (`axtb-base`'s [`CnModel`]).
fn revd4_cn(mol: &Molecule) -> CnModel<Erf, impl Fn(usize, usize) -> f64 + '_> {
    CnModel::new(Erf::D4, |i: usize, j: usize| D4_RCOV[mol.numbers[i] - 1] + D4_RCOV[mol.numbers[j] - 1])
        .with_cutoff(Cutoff::Squared(CUTOFF.cn * CUTOFF.cn))
        .with_min_r2(1e-12)
}

/// revD4 coordination numbers (no electronegativity factor, 30 bohr cutoff).
pub fn coordination_numbers(mol: &Molecule) -> Vec<f64> {
    revd4_cn(mol).compute(&mol.positions)
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
        let model = revd4_model(&mol.numbers, CHARGES_V1_1)?;
        let disp = D4Dispersion::new(model, damping(param), CUTOFF);
        let cache = disp.update(&mol.positions);
        let cn = coordination_numbers(mol);
        let model = disp.model();
        let (kmat, dkmat) = k_matrices(model, &cn, CHARGES_V1_1.refq, false);
        let r4r2 = (0..mol.nat()).map(|i| model.r4r2_atom(i)).collect();
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
        revd4_cn(mol).add_gradient(&mol.positions, dedcn, grad);
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
