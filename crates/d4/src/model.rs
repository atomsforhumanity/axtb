// SPDX-License-Identifier: LGPL-3.0-or-later
//! The D4 dispersion model: per-species reference data, reference C6
//! coefficients, Gaussian/zeta reference weights, atomic C6 and polarizabilities.
//!
//! Port of `dftd4_model_d4` / `dftd4_reference` / `dftd4_model_utils` (dftd4 4.3.1).
//! Floating-point expressions keep the Fortran evaluation order.

use crate::data::{self, MAX_REF, NFREQ};
use crate::D4Error;

/// Which reference charges (and hydrogen-subtracted reference polarizabilities)
/// the model uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ChargeModel {
    /// Standard dftd4: EEQ (2019) reference charges (`d4_qmod%eeq`).
    Eeq,
    /// GFN2-xTB Mulliken reference charges (`d4_qmod%gfn2`), used by tblite/xtb.
    Gfn2,
}

/// Default charge scaling height `ga` (dftd4 `new_d4_model`).
pub const DEFAULT_GA: f64 = 3.0;
/// Default charge scaling steepness `gc` (dftd4 `new_d4_model`).
pub const DEFAULT_GC: f64 = 2.0;
/// Default CN interpolation weighting factor `wf` (dftd4 `new_d4_model`).
pub const DEFAULT_WF: f64 = 6.0;

/// Reference weights per atom (`[iat][iref]`, padded to [`MAX_REF`]) and their
/// derivatives with respect to the atom's own coordination number and charge.
#[derive(Clone, Debug, Default)]
pub struct RefWeights {
    /// zeta-scaled Gaussian weights `gwvec`
    pub gw: Vec<[f64; MAX_REF]>,
    /// d gw / d CN_i (`gwdcn`), empty if not requested
    pub dgwdcn: Vec<[f64; MAX_REF]>,
    /// d gw / d q_i (`gwdq`), empty if not requested
    pub dgwdq: Vec<[f64; MAX_REF]>,
}

/// Atomic C6 coefficients and derivatives as dense row-major `nat x nat` matrices.
#[derive(Clone, Debug, Default)]
pub struct AtomicC6 {
    pub nat: usize,
    /// `c6[i*nat + j]` = C6_ij (symmetric)
    pub c6: Vec<f64>,
    /// `dc6dcn[i*nat + j]` = dC6_ij / dCN_i
    pub dc6dcn: Vec<f64>,
    /// `dc6dq[i*nat + j]` = dC6_ij / dq_i
    pub dc6dq: Vec<f64>,
}

/// D4 dispersion model for one molecule (fixed atomic numbers).
#[derive(Clone, Debug)]
pub struct D4Model {
    pub(crate) numbers: Vec<usize>,
    /// species index of every atom
    pub(crate) id: Vec<usize>,
    /// atomic number of every species
    pub(crate) species: Vec<usize>,
    pub(crate) charge_model: ChargeModel,
    pub(crate) ga: f64,
    pub(crate) gc: f64,
    pub(crate) wf: f64,
    // ---- per species ----
    pub(crate) rcov: Vec<f64>,
    pub(crate) en: Vec<f64>,
    pub(crate) zeff: Vec<f64>,
    pub(crate) eta: Vec<f64>,
    pub(crate) r4r2: Vec<f64>,
    pub(crate) nref: Vec<usize>,
    pub(crate) ngw: Vec<[u32; MAX_REF]>,
    pub(crate) refcn: Vec<[f64; MAX_REF]>,
    pub(crate) refq: Vec<[f64; MAX_REF]>,
    pub(crate) aiw: Vec<[[f64; NFREQ]; MAX_REF]>,
    /// reference C6, block `[isp*nsp + jsp][iref][jref]`
    pub(crate) c6ref: Vec<[[f64; MAX_REF]; MAX_REF]>,
    pub(crate) mref: usize,
}

/// Conversion factor Angstrom -> bohr exactly as mctc-lib evaluates it (CODATA 2018 derived;
/// `axtb_base::units::reference::MCTC_AATOAU`).
pub const AATOAU: f64 = axtb_base::units::reference::MCTC_AATOAU;

/// D3/D4 covalent radius (bohr): 4/3 * Pyykko radius.
#[inline]
pub fn covalent_radius(z: usize) -> f64 {
    if z == 0 || z > data::MAX_ELEM {
        return 0.0;
    }
    4.0 / 3.0 * (AATOAU * data::COVALENT_RAD_2009[z - 1])
}

/// `sqrt(0.5 * <r4>/<r2> * sqrt(Z))` as used for the C8 extrapolation.
#[inline]
pub fn r4r2_value(z: usize) -> f64 {
    if z == 0 || z > data::MAX_ELEM {
        return 0.0;
    }
    (0.5 * (data::R4_OVER_R2[z - 1] * (z as f64).sqrt())).sqrt()
}

#[inline]
fn tab(table: &[f64; data::MAX_ELEM], z: usize) -> f64 {
    if z == 0 || z > data::MAX_ELEM {
        0.0
    } else {
        table[z - 1]
    }
}

/// `zeta` charge scaling function of dftd4.
#[inline]
pub fn zeta(a: f64, c: f64, qref: f64, qmod: f64) -> f64 {
    if qmod < 0.0 {
        a.exp()
    } else {
        (a * (1.0 - (c * (1.0 - qref / qmod)).exp())).exp()
    }
}

/// zeta and its derivative with respect to `qmod` (dftd4 `zeta`, `dzeta`).
#[inline]
pub fn zeta_dzeta(a: f64, c: f64, qref: f64, qmod: f64) -> (f64, f64) {
    if qmod < 0.0 {
        (a.exp(), 0.0)
    } else {
        let e = (c * (1.0 - qref / qmod)).exp();
        let z = (a * (1.0 - e)).exp();
        let dz = -((a * c * e * z * qref) / (qmod * qmod));
        (z, dz)
    }
}

/// Imaginary frequencies of the Casimir-Polder integration grid.
const FREQ: [f64; NFREQ] = [
    0.000001, 0.050000, 0.100000, 0.200000, 0.300000, 0.400000, 0.500000, 0.600000, 0.700000, 0.800000, 0.900000,
    1.000000, 1.200000, 1.400000, 1.600000, 1.800000, 2.000000, 2.500000, 3.000000, 4.000000, 5.000000, 7.500000,
    10.00000,
];

/// Trapezoid weights, evaluated like the Fortran parameter expression.
const fn trapezoid_weights() -> [f64; NFREQ] {
    let mut w = [0.0; NFREQ];
    w[0] = 0.5 * (FREQ[1] - FREQ[0]);
    let mut k = 1;
    while k < NFREQ - 1 {
        w[k] = 0.5 * ((FREQ[k] - FREQ[k - 1]) + (FREQ[k + 1] - FREQ[k]));
        k += 1;
    }
    w[NFREQ - 1] = 0.5 * (FREQ[NFREQ - 1] - FREQ[NFREQ - 2]);
    w
}
const TRAPZD_WEIGHTS: [f64; NFREQ] = trapezoid_weights();

/// Casimir-Polder trapezoid integration `sum(pol * weights)`.
#[inline]
pub fn trapzd(pol: &[f64; NFREQ]) -> f64 {
    let mut s = 0.0;
    for k in 0..NFREQ {
        s += pol[k] * TRAPZD_WEIGHTS[k];
    }
    s
}

/// Number of Gaussian weights per reference (`set_refgw`).
fn reference_ngw(z: usize) -> [u32; MAX_REF] {
    const MAX_CN: usize = 19;
    let mut ngw = [1u32; MAX_REF];
    let nref = data::REFN[z - 1] as usize;
    let mut cnc = [0u32; MAX_CN + 1];
    cnc[0] = 1;
    // nint: round half away from zero
    let icn = |x: f64| -> usize { (x.round() as i64).clamp(0, MAX_CN as i64) as usize };
    for ir in 0..nref {
        cnc[icn(data::REFCN[z - 1][ir])] += 1;
    }
    for ir in 0..nref {
        let c = cnc[icn(data::REFCN[z - 1][ir])];
        ngw[ir] = c * (c + 1) / 2;
    }
    ngw
}

/// Reference polarizabilities of element `z` (1-based) on the imaginary frequency grid, with
/// the hydrogen contributions of the secondary systems removed (the loop of dftd4's
/// `set_refalpha_eeq` / `set_refalpha_gfn2`):
/// `max(0, ascale (alpha_iw - hcount sscale secaiw s))` for each of the `data::REFN[z - 1]`
/// references, where `s = secondary_scale(is, ir)` is the charge scaling of the secondary
/// system `is` (1-based) of reference `ir`: dftd4's `zeta` for D4 ([`D4Model`]), 1 for the
/// D3-type dispersion of GFN-FF (whose `s · x` is then `x` exactly). Entries of references
/// without a secondary system and beyond `REFN` are zero.
pub fn reference_alpha(z: usize, secondary_scale: impl Fn(usize, usize) -> f64) -> [[f64; NFREQ]; MAX_REF] {
    let mut alpha = [[0.0; NFREQ]; MAX_REF];
    let nref = data::REFN[z - 1] as usize;
    for ir in 0..nref {
        let is = data::REFSYS[z - 1][ir] as usize;
        if is == 0 {
            continue;
        }
        let zv = secondary_scale(is, ir);
        let ascale = data::ASCALE[z - 1][ir];
        let hcount = data::HCOUNT[z - 1][ir];
        for k in 0..NFREQ {
            let aiw = data::SSCALE[is - 1] * data::SECAIW[is - 1][k] * zv;
            let v = ascale * (data::ALPHAIW[z - 1][ir][k] - hcount * aiw);
            alpha[ir][k] = if v > 0.0 { v } else { 0.0 };
        }
    }
    alpha
}

/// dftd4's reference polarizabilities: [`reference_alpha`] with the `zeta` charge scaling of
/// the secondary systems (`set_refalpha_eeq` / `set_refalpha_gfn2`).
fn dftd4_reference_alpha(z: usize, ga: f64, gc: f64, model: ChargeModel) -> [[f64; NFREQ]; MAX_REF] {
    let refh = match model {
        ChargeModel::Eeq => &data::REFH_EEQ,
        ChargeModel::Gfn2 => &data::REFH_GFN2,
    };
    reference_alpha(z, |is, ir| {
        // NB: dftd4 evaluates zeff and hardness of the *secondary system index*
        let iz = tab(&data::EFFECTIVE_NUCLEAR_CHARGE, is);
        let gam = tab(&data::CHEMICAL_HARDNESS, is) * gc;
        zeta(ga, gam, iz, refh[z - 1][ir] + iz)
    })
}

/// Reference C6 coefficients of all pairs of species by Casimir-Polder integration of their
/// reference polarizabilities `aiw[isp][iref]` (e.g. from [`reference_alpha`]):
/// `3/pi * trapzd(alpha_A * alpha_B)` on the 23-point grid ([`trapzd`]), each product formed
/// with the species of the larger index first. `nref[isp]` references per species; returns the
/// blocks `[isp * nsp + jsp][iref][jref]`, zero beyond `nref`.
pub fn reference_c6_table(nref: &[usize], aiw: &[[[f64; NFREQ]; MAX_REF]]) -> Vec<[[f64; MAX_REF]; MAX_REF]> {
    let nsp = nref.len();
    let thopi = 3.0 / data::PI;
    let mut c6ref = vec![[[0.0; MAX_REF]; MAX_REF]; nsp * nsp];
    let mut prod = [0.0; NFREQ];
    for isp in 0..nsp {
        for jsp in 0..=isp {
            for iref in 0..nref[isp] {
                for jref in 0..nref[jsp] {
                    for k in 0..NFREQ {
                        prod[k] = aiw[isp][iref][k] * aiw[jsp][jref][k];
                    }
                    let c6 = thopi * trapzd(&prod);
                    c6ref[jsp * nsp + isp][jref][iref] = c6;
                    c6ref[isp * nsp + jsp][iref][jref] = c6;
                }
            }
        }
    }
    c6ref
}

/// Reference data of one element supplied by the caller ([`D4Model::with_reference_data`]),
/// e.g. another program's modified dftd4 reference set. All vectors have one entry per
/// reference system (at most [`MAX_REF`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ElementReferences {
    /// Reference coordination numbers of the Gaussian weights (`refcovcn`).
    pub cn: Vec<f64>,
    /// Number of Gaussian functions per reference (`set_refgw`).
    pub ngw: Vec<u32>,
    /// Reference charges.
    pub q: Vec<f64>,
    /// Reference dynamic polarizabilities on the 23-point imaginary frequency grid, with the
    /// hydrogen contributions already removed (`set_refalpha_*`).
    pub alpha: Vec<[f64; NFREQ]>,
}

impl D4Model {
    /// New D4 model for the given atomic numbers with default `ga`, `gc`, `wf`.
    pub fn new(numbers: &[usize], charge_model: ChargeModel) -> Result<Self, D4Error> {
        Self::with_scaling(numbers, charge_model, DEFAULT_GA, DEFAULT_GC, DEFAULT_WF)
    }

    /// New D4 model with custom charge scaling height/steepness and CN weighting factor.
    pub fn with_scaling(
        numbers: &[usize],
        charge_model: ChargeModel,
        ga: f64,
        gc: f64,
        wf: f64,
    ) -> Result<Self, D4Error> {
        let mut species: Vec<usize> = Vec::new();
        let mut id = Vec::with_capacity(numbers.len());
        for (iat, &z) in numbers.iter().enumerate() {
            if z == 0 || z > data::MAX_ELEM || (104..=111).contains(&z) {
                return Err(D4Error::UnsupportedElement { atom: iat, number: z });
            }
            let isp = match species.iter().position(|&s| s == z) {
                Some(p) => p,
                None => {
                    species.push(z);
                    species.len() - 1
                }
            };
            id.push(isp);
        }
        let refq_table = match charge_model {
            ChargeModel::Eeq => &data::REFQ_EEQ,
            ChargeModel::Gfn2 => &data::REFQ_GFN2,
        };

        let rcov = species.iter().map(|&z| covalent_radius(z)).collect();
        let en = species.iter().map(|&z| tab(&data::PAULING_EN, z)).collect();
        let zeff = species.iter().map(|&z| tab(&data::EFFECTIVE_NUCLEAR_CHARGE, z)).collect();
        let eta = species.iter().map(|&z| tab(&data::CHEMICAL_HARDNESS, z)).collect();
        let r4r2 = species.iter().map(|&z| r4r2_value(z)).collect();
        let nref: Vec<usize> = species.iter().map(|&z| data::REFN[z - 1] as usize).collect();
        let ngw = species.iter().map(|&z| reference_ngw(z)).collect();
        let refcn = species
            .iter()
            .zip(&nref)
            .map(|(&z, &n)| {
                let mut c = [0.0; MAX_REF];
                c[..n].copy_from_slice(&data::REFCOVCN[z - 1][..n]);
                c
            })
            .collect();
        let refq = species
            .iter()
            .zip(&nref)
            .map(|(&z, &n)| {
                let mut c = [0.0; MAX_REF];
                c[..n].copy_from_slice(&refq_table[z - 1][..n]);
                c
            })
            .collect();
        let aiw: Vec<[[f64; NFREQ]; MAX_REF]> =
            species.iter().map(|&z| dftd4_reference_alpha(z, ga, gc, charge_model)).collect();

        let c6ref = reference_c6_table(&nref, &aiw);
        let mref = nref.iter().copied().max().unwrap_or(0);

        Ok(Self {
            numbers: numbers.to_vec(),
            id,
            species,
            charge_model,
            ga,
            gc,
            wf,
            rcov,
            en,
            zeff,
            eta,
            r4r2,
            nref,
            ngw,
            refcn,
            refq,
            aiw,
            c6ref,
            mref,
        })
    }

    /// Number of atoms.
    #[inline]
    pub fn natoms(&self) -> usize {
        self.numbers.len()
    }

    /// Atomic numbers the model was built for.
    #[inline]
    pub fn numbers(&self) -> &[usize] {
        &self.numbers
    }

    /// Reference charge model.
    #[inline]
    pub fn charge_model(&self) -> ChargeModel {
        self.charge_model
    }

    /// Maximum number of reference systems over all species.
    #[inline]
    pub fn max_ref(&self) -> usize {
        self.mref
    }

    /// Number of reference systems of atom `iat`.
    #[inline]
    pub fn nref_atom(&self, iat: usize) -> usize {
        self.nref[self.id[iat]]
    }

    /// `sqrt(0.5 <r4>/<r2> sqrt(Z))` of atom `iat`.
    #[inline]
    pub fn r4r2_atom(&self, iat: usize) -> f64 {
        self.r4r2[self.id[iat]]
    }

    /// Replace the covalent radii (bohr) and electronegativities used by the D4
    /// coordination number, per element (`f(z)`), e.g. to reproduce another program's
    /// tables exactly.
    pub fn with_cn_parameters(mut self, rcov: impl Fn(usize) -> f64, en: impl Fn(usize) -> f64) -> Self {
        self.rcov = self.species.iter().map(|&z| rcov(z)).collect();
        self.en = self.species.iter().map(|&z| en(z)).collect();
        self
    }

    /// Replace the reference systems of every element (reference CNs, Gaussian counts,
    /// reference charges and polarizabilities, `refs(z)`) and recompute the reference C6
    /// coefficients, e.g. to reproduce another program's modified reference set. Everything
    /// else (charge model, `ga`, `gc`, `wf`, CN parameters, r4/r2) is kept.
    ///
    /// # Panics
    /// If the vectors of an element differ in length or hold more than [`MAX_REF`] entries.
    pub fn with_reference_data(mut self, refs: impl Fn(usize) -> ElementReferences) -> Self {
        let nsp = self.species.len();
        for isp in 0..nsp {
            let r = refs(self.species[isp]);
            let n = r.cn.len();
            assert!(
                n <= MAX_REF && r.ngw.len() == n && r.q.len() == n && r.alpha.len() == n,
                "D4 reference data of Z={}: inconsistent lengths",
                self.species[isp]
            );
            self.nref[isp] = n;
            self.refcn[isp] = [0.0; MAX_REF];
            self.refcn[isp][..n].copy_from_slice(&r.cn);
            self.ngw[isp] = [1; MAX_REF];
            self.ngw[isp][..n].copy_from_slice(&r.ngw);
            self.refq[isp] = [0.0; MAX_REF];
            self.refq[isp][..n].copy_from_slice(&r.q);
            self.aiw[isp] = [[0.0; NFREQ]; MAX_REF];
            self.aiw[isp][..n].copy_from_slice(&r.alpha);
        }
        self.c6ref = reference_c6_table(&self.nref, &self.aiw);
        self.mref = self.nref.iter().copied().max().unwrap_or(0);
        self
    }

    /// Covalent radius (bohr) used in the D4 coordination number for atom `iat`.
    #[inline]
    pub fn rcov_atom(&self, iat: usize) -> f64 {
        self.rcov[self.id[iat]]
    }

    /// Pauling electronegativity used in the D4 coordination number for atom `iat`.
    #[inline]
    pub fn en_atom(&self, iat: usize) -> f64 {
        self.en[self.id[iat]]
    }

    /// Reference C6 between reference `iref` of atom `iat` and `jref` of atom `jat`.
    #[inline]
    pub fn reference_c6(&self, iat: usize, jat: usize, iref: usize, jref: usize) -> f64 {
        let nsp = self.species.len();
        self.c6ref[self.id[iat] * nsp + self.id[jat]][iref][jref]
    }

    /// Reference static polarizabilities of atom `iat` (frequency 1 of the grid).
    pub fn reference_alpha0(&self, iat: usize) -> Vec<f64> {
        let isp = self.id[iat];
        (0..self.nref[isp]).map(|r| self.aiw[isp][r][0]).collect()
    }

    #[inline]
    pub(crate) fn c6_block(&self, isp: usize, jsp: usize) -> &[[f64; MAX_REF]; MAX_REF] {
        &self.c6ref[isp * self.species.len() + jsp]
    }

    /// Reference weights `gw` (and optionally `d gw/d CN`, `d gw/d q`) for every
    /// atom, port of `d4_model%weight_references`.
    ///
    /// `q` are atomic partial charges (positive = electron deficient).
    pub fn weight_references_into(
        &self,
        cn: &[f64],
        q: &[f64],
        gw: &mut [[f64; MAX_REF]],
        mut dgwdcn: Option<&mut [[f64; MAX_REF]]>,
        mut dgwdq: Option<&mut [[f64; MAX_REF]]>,
    ) {
        let nat = self.natoms();
        assert!(cn.len() >= nat && q.len() >= nat && gw.len() >= nat);
        let eps_norm = f64::MIN_POSITIVE.sqrt();
        let want_dcn = dgwdcn.is_some();
        for iat in 0..nat {
            let isp = self.id[iat];
            let nref = self.nref[isp];
            let zi = self.zeff[isp];
            let gi = self.eta[isp] * self.gc;
            let cni = cn[iat];
            let refcn = &self.refcn[isp];
            let ngw = &self.ngw[isp];

            let mut norm = 0.0;
            let mut dnorm = 0.0;
            for iref in 0..nref {
                for igw in 1..=ngw[iref] {
                    let wf = igw as f64 * self.wf;
                    let d = cni - refcn[iref];
                    let g = (-(wf * (d * d))).exp();
                    norm += g;
                    if want_dcn {
                        dnorm += 2.0 * wf * (refcn[iref] - cni) * g;
                    }
                }
            }
            let norm = if norm.abs() > eps_norm { 1.0 / norm } else { 0.0 };

            let mut maxcn = f64::NEG_INFINITY;
            for iref in 0..nref {
                if refcn[iref] > maxcn {
                    maxcn = refcn[iref];
                }
            }

            let gwi = &mut gw[iat];
            *gwi = [0.0; MAX_REF];
            let mut dcn_row = [0.0; MAX_REF];
            let mut dq_row = [0.0; MAX_REF];
            for iref in 0..nref {
                let mut expw = 0.0;
                let mut expd = 0.0;
                for igw in 1..=ngw[iref] {
                    let wf = igw as f64 * self.wf;
                    let d = cni - refcn[iref];
                    let g = (-(wf * (d * d))).exp();
                    expw += g;
                    if want_dcn {
                        expd += 2.0 * wf * (refcn[iref] - cni) * g;
                    }
                }
                let mut gwk = expw * norm;
                if !gwk.is_finite() || norm == 0.0 {
                    gwk = if (maxcn - refcn[iref]).abs() < 1e-12 { 1.0 } else { 0.0 };
                }
                let qref = self.refq[isp][iref] + zi;
                let qmod = q[iat] + zi;
                let (z, dz) = if dgwdq.is_some() {
                    zeta_dzeta(self.ga, gi, qref, qmod)
                } else {
                    (zeta(self.ga, gi, qref, qmod), 0.0)
                };
                gwi[iref] = gwk * z;
                dq_row[iref] = gwk * dz;
                if want_dcn {
                    let mut dgwk = norm * (expd - expw * dnorm * norm);
                    if !dgwk.is_finite() || norm == 0.0 {
                        dgwk = 0.0;
                    }
                    dcn_row[iref] = dgwk * z;
                }
            }
            if let Some(d) = dgwdcn.as_deref_mut() {
                d[iat] = dcn_row;
            }
            if let Some(d) = dgwdq.as_deref_mut() {
                d[iat] = dq_row;
            }
        }
    }

    /// Reference weights with both derivatives (allocating convenience wrapper).
    pub fn weight_references(&self, cn: &[f64], q: &[f64]) -> RefWeights {
        let nat = self.natoms();
        let mut w = RefWeights {
            gw: vec![[0.0; MAX_REF]; nat],
            dgwdcn: vec![[0.0; MAX_REF]; nat],
            dgwdq: vec![[0.0; MAX_REF]; nat],
        };
        self.weight_references_into(cn, q, &mut w.gw, Some(&mut w.dgwdcn), Some(&mut w.dgwdq));
        w
    }

    /// C6 of one atom pair from reference weights.
    #[inline]
    pub fn pair_c6(&self, iat: usize, jat: usize, gw: &[[f64; MAX_REF]]) -> f64 {
        let (isp, jsp) = (self.id[iat], self.id[jat]);
        let blk = self.c6_block(isp, jsp);
        let (wi, wj) = (&gw[iat], &gw[jat]);
        let mut c6 = 0.0;
        for iref in 0..self.nref[isp] {
            for jref in 0..self.nref[jsp] {
                c6 += wi[iref] * wj[jref] * blk[iref][jref];
            }
        }
        c6
    }

    /// C6 of a pair and its derivatives given generic left derivative arrays:
    /// returns `(C6_ij, sum dl_i C w_j, sum w_i C dl_j)`.
    #[inline]
    pub(crate) fn pair_c6_deriv(
        &self,
        iat: usize,
        jat: usize,
        gw: &[[f64; MAX_REF]],
        dgw: &[[f64; MAX_REF]],
    ) -> (f64, f64, f64) {
        let (isp, jsp) = (self.id[iat], self.id[jat]);
        let blk = self.c6_block(isp, jsp);
        let (wi, wj) = (&gw[iat], &gw[jat]);
        let (di, dj) = (&dgw[iat], &dgw[jat]);
        let (mut c6, mut dci, mut dcj) = (0.0, 0.0, 0.0);
        for iref in 0..self.nref[isp] {
            for jref in 0..self.nref[jsp] {
                let r = blk[iref][jref];
                c6 += wi[iref] * wj[jref] * r;
                dci += di[iref] * wj[jref] * r;
                dcj += wi[iref] * dj[jref] * r;
            }
        }
        (c6, dci, dcj)
    }

    /// Dense atomic C6 matrix with CN and charge derivatives (`get_atomic_c6`).
    /// Derivative matrices are left empty if the corresponding weight
    /// derivatives in `w` are empty.
    pub fn atomic_c6(&self, w: &RefWeights) -> AtomicC6 {
        let nat = self.natoms();
        let mut out = AtomicC6 { nat, c6: vec![0.0; nat * nat], ..Default::default() };
        let dcn = w.dgwdcn.len() >= nat;
        let dq = w.dgwdq.len() >= nat;
        if dcn {
            out.dc6dcn = vec![0.0; nat * nat];
        }
        if dq {
            out.dc6dq = vec![0.0; nat * nat];
        }
        for iat in 0..nat {
            for jat in 0..=iat {
                let c6 = if dcn {
                    let (c6, di, dj) = self.pair_c6_deriv(iat, jat, &w.gw, &w.dgwdcn);
                    out.dc6dcn[iat * nat + jat] = di;
                    out.dc6dcn[jat * nat + iat] = dj;
                    c6
                } else {
                    self.pair_c6(iat, jat, &w.gw)
                };
                if dq {
                    let (_, di, dj) = self.pair_c6_deriv(iat, jat, &w.gw, &w.dgwdq);
                    out.dc6dq[iat * nat + jat] = di;
                    out.dc6dq[jat * nat + iat] = dj;
                }
                out.c6[iat * nat + jat] = c6;
                out.c6[jat * nat + iat] = c6;
            }
        }
        out
    }

    /// Static atomic polarizabilities and their CN / charge derivatives
    /// (`get_polarizabilities`). Derivatives are zero if not present in `w`.
    pub fn polarizabilities(&self, w: &RefWeights) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let nat = self.natoms();
        let mut alpha = vec![0.0; nat];
        let mut dadcn = vec![0.0; nat];
        let mut dadq = vec![0.0; nat];
        for iat in 0..nat {
            let isp = self.id[iat];
            for iref in 0..self.nref[isp] {
                let refa = self.aiw[isp][iref][0];
                alpha[iat] += w.gw[iat][iref] * refa;
                if let Some(d) = w.dgwdcn.get(iat) {
                    dadcn[iat] += d[iref] * refa;
                }
                if let Some(d) = w.dgwdq.get(iat) {
                    dadq[iat] += d[iref] * refa;
                }
            }
        }
        (alpha, dadcn, dadq)
    }
}
