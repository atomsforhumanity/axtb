// SPDX-License-Identifier: LGPL-3.0-or-later
//! Self-consistent D4 dispersion container (port of `tblite_disp_d4` with the
//! rational damping two-body and ATM three-body kernels of
//! `dftd4_damping_rational` / `dftd4_damping_atm`), molecular systems only.

use crate::data::MAX_REF;
use crate::model::{ChargeModel, D4Model};
use crate::D4Error;

/// Rational (Becke-Johnson) damping parameters with ATM three-body scaling.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RationalDamping {
    pub s6: f64,
    pub s8: f64,
    pub s9: f64,
    pub a1: f64,
    pub a2: f64,
    /// exponent of the zero-damping in the ATM term (dftd4 default 16)
    pub alp: f64,
}

impl RationalDamping {
    /// GFN2-xTB parameters (tblite `xtb/gfn2.f90`).
    pub const GFN2: Self = Self { s6: 1.0, s8: 2.7, s9: 5.0, a1: 0.52, a2: 5.0, alp: 16.0 };
}

/// Real-space cutoffs (bohr) and smoothing widths of the switching function.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RealspaceCutoff {
    /// coordination number cutoff (hard)
    pub cn: f64,
    /// two-body cutoff
    pub disp2: f64,
    /// three-body cutoff
    pub disp3: f64,
    /// width of the smooth two-body cutoff (0 = hard)
    pub width2: f64,
    /// width of the smooth three-body cutoff (0 = hard)
    pub width3: f64,
}

impl RealspaceCutoff {
    /// dftd4 defaults (`realspace_cutoff()`): cn 30, disp2 60, disp3 40, hard.
    pub const DFTD4: Self = Self { cn: 30.0, disp2: 60.0, disp3: 40.0, width2: 0.0, width3: 0.0 };
    /// tblite 0.8.0 GFN2-xTB: `realspace_cutoff(disp3=25, disp2=50)` (cn 30) with the
    /// default `xtb_config%smooth_cutoff = 0.05` bohr used for both widths.
    pub const TBLITE_GFN2: Self = Self { cn: 30.0, disp2: 50.0, disp3: 25.0, width2: 0.05, width3: 0.05 };
    /// The xtb 6.7.1 program's own GFN2-xTB D4 (`src/disp/dftd4.F90`, `scf_module.F90`):
    /// CN cutoff 40 bohr (`thr = 1600` bohr²), two-body 72 bohr (`rthr` of the SCC
    /// dispersion matrix; xtb uses 60 bohr in its gradient, which is inconsistent with its
    /// energy beyond 60 bohr, so 72 is used for both here), three-body 40 bohr, all hard.
    pub const XTB_GFN2: Self = Self { cn: 40.0, disp2: 72.0, disp3: 40.0, width2: 0.0, width3: 0.0 };
}

const SMOOTHSTEP3: f64 = 10.0;
const SMOOTHSTEP4: f64 = -15.0;
const SMOOTHSTEP5: f64 = 6.0;
const SMOOTHSTEP_DERIV: f64 = 30.0;

/// Quintic smoothstep switching function of dftd4 (`smooth_cutoff`): returns
/// `(sw, dsw/dr)`; `sw = 1` for `r <= cutoff - width`, `0` for `r >= cutoff`.
#[inline]
pub fn smooth_cutoff(r: f64, cutoff: f64, width: f64) -> (f64, f64) {
    if width <= 0.0 || cutoff <= 0.0 {
        return (1.0, 0.0);
    }
    let w = width.min(cutoff);
    let inner = cutoff - w;
    if r <= inner {
        (1.0, 0.0)
    } else if r >= cutoff {
        (0.0, 0.0)
    } else {
        let x = (cutoff - r) / w;
        let x2 = x * x;
        let sw = (x2 * x) * (SMOOTHSTEP3 + x * (SMOOTHSTEP4 + SMOOTHSTEP5 * x));
        let omx = 1.0 - x;
        let dsw = -((SMOOTHSTEP_DERIV * x2 * (omx * omx)) / w);
        (sw, dsw)
    }
}

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

/// Per-geometry data of the D4 container (tblite `dispersion_cache`): the
/// coordination numbers and, instead of tblite's reference-pair dispersion
/// matrix `dE_ij * C6ref`, only the scalar damped radial factors `dE_ij` of all
/// pairs inside the two-body cutoff (the reference C6 block is looked up per
/// species pair).
#[derive(Clone, Debug, Default)]
pub struct D4Cache {
    positions: Vec<[f64; 3]>,
    /// D4 coordination numbers
    pub cn: Vec<f64>,
    /// CSR offsets into the pair arrays (row = larger atom index)
    pub(crate) pair_offset: Vec<usize>,
    pub(crate) pair_j: Vec<u32>,
    /// `sw * (s6 t6 + s8 rrij t8)` (pair energy = -C6_ij * pair_e)
    pub(crate) pair_e: Vec<f64>,
    /// `sw * d(s6 t6 + s8 rrij t8)/(r dr) + dsw/dr * (..)/r` (gradient factor)
    pub(crate) pair_g: Vec<f64>,
    /// The pairs of each atom as the second (smaller) atom: offsets and pair indices, in
    /// ascending order of the first atom (the transpose of the rows)
    pair_t_offset: Vec<usize>,
    pair_t: Vec<u32>,
}

impl D4Cache {
    /// Number of atom pairs inside the two-body cutoff.
    pub fn npairs(&self) -> usize {
        self.pair_j.len()
    }
    /// Positions the cache was built for.
    pub fn positions(&self) -> &[[f64; 3]] {
        &self.positions
    }
    /// Atom pairs inside the two-body cutoff as `(i, j, f)` with `i > j` and the damped radial
    /// factor `f = sw (s6 t6 + s8 rr t8)`: the two-body energy of the pair is `-C6_ij f`.
    pub fn pairs(&self) -> impl Iterator<Item = (usize, usize, f64)> + '_ {
        (0..self.pair_offset.len().saturating_sub(1)).flat_map(move |i| {
            (self.pair_offset[i]..self.pair_offset[i + 1]).map(move |p| (i, self.pair_j[p] as usize, self.pair_e[p]))
        })
    }
}

/// Source of the pair C6 coefficients of the three-body kernel.
#[derive(Clone, Copy)]
enum PairC6<'a> {
    /// From reference weights (and their CN derivatives when gradients are requested).
    Weights { gw: &'a [[f64; MAX_REF]], dgwdcn: &'a [[f64; MAX_REF]] },
    /// A dense row-major `nat x nat` C6 matrix supplied by the caller (energy only).
    Matrix(&'a [f64]),
    /// A C6 matrix with its CN derivatives (`dc6dcn[i*nat + j]` = dC6_ij/dCN_i).
    MatrixDeriv { c6: &'a [f64], dc6dcn: &'a [f64] },
}

/// Half neighbour list of the three-body kernel (structure of arrays).
#[derive(Default)]
struct PairList {
    j: Vec<u32>,
    v: Vec<[f64; 3]>,
    r2: Vec<f64>,
    /// 1 / r^2
    ir2: Vec<f64>,
    /// 1 / r
    ir: Vec<f64>,
    sw: Vec<f64>,
    /// (dsw/dr) / r
    dswr: Vec<f64>,
    /// `(r0 / r)^(alp/3)`
    t: Vec<f64>,
    c6: Vec<f64>,
    /// (dC6/dCN_a) / C6 of the row atom a
    qa: Vec<f64>,
    /// (dC6/dCN_b) / C6 of the neighbour b
    qb: Vec<f64>,
}

/// Self-consistent D4 dispersion for one molecule: model + damping + cutoffs.
///
/// Threads: the methods with a `parallel` argument run their pair or triple loops on the
/// current rayon pool when it is true (the caller's pool: call them inside
/// `ThreadPool::install`), with fixed chunks summed in a fixed order, so the results are the
/// same bits either way. Everything else runs on the calling thread.
#[derive(Clone, Debug)]
pub struct D4Dispersion {
    /// Three-body gradient by pair sums ([`D4Dispersion::with_pair_sums`]).
    pair_sums: bool,
    model: D4Model,
    param: RationalDamping,
    cutoff: RealspaceCutoff,
    /// `3 * r4r2[a] * r4r2[b]` per species pair `[a*nsp + b]` (order matters bitwise)
    rr: Vec<f64>,
    /// `a1 * sqrt(rr) + a2`
    r0: Vec<f64>,
}

impl D4Dispersion {
    /// Combine a model with damping parameters and cutoffs.
    pub fn new(model: D4Model, param: RationalDamping, cutoff: RealspaceCutoff) -> Self {
        let nsp = model.species.len();
        let mut rr = vec![0.0; nsp * nsp];
        let mut r0 = vec![0.0; nsp * nsp];
        for a in 0..nsp {
            for b in 0..nsp {
                let v = 3.0 * model.r4r2[a] * model.r4r2[b];
                rr[a * nsp + b] = v;
                r0[a * nsp + b] = param.a1 * v.sqrt() + param.a2;
            }
        }
        Self { pair_sums: false, model, param, cutoff, rr, r0 }
    }

    /// Three-body energy and gradient summed per atom pair (xtb 6.8.0, grimme-lab/xtb#1464):
    /// for every pair, one branch-free, vectorized loop over all third atoms sums the triple
    /// energies and dE/dr of that pair (each triple is visited once per pair), and one pass
    /// over the pairs turns the sums into atomic energies, Cartesian gradients and dE/dCN.
    /// Several times faster than dftd4's loop over the triples (`false`, the default), whose
    /// summation order it does not keep: the results differ by round-off. Each pair's sums
    /// have a fixed order, so the results do not depend on the thread count.
    pub fn with_pair_sums(mut self, pair_sums: bool) -> Self {
        self.pair_sums = pair_sums;
        self
    }

    /// The D4 setup of tblite 0.8.0 for GFN2-xTB: GFN2 reference charges,
    /// s6=1, s8=2.7, a1=0.52, a2=5.0, s9=5.0, cutoffs [`RealspaceCutoff::TBLITE_GFN2`].
    pub fn gfn2(numbers: &[usize]) -> Result<Self, D4Error> {
        Ok(Self::new(D4Model::new(numbers, ChargeModel::Gfn2)?, RationalDamping::GFN2, RealspaceCutoff::TBLITE_GFN2))
    }

    /// The underlying D4 model.
    pub fn model(&self) -> &D4Model {
        &self.model
    }

    /// Damping parameters.
    pub fn param(&self) -> &RationalDamping {
        &self.param
    }

    /// Real-space cutoffs.
    pub fn cutoff(&self) -> &RealspaceCutoff {
        &self.cutoff
    }

    /// Number of atoms.
    pub fn natoms(&self) -> usize {
        self.model.natoms()
    }

    /// `3 r4r2_a r4r2_b` per species pair.
    pub(crate) fn rr_table(&self) -> &[f64] {
        &self.rr
    }

    /// `a1 sqrt(rr) + a2` per species pair.
    pub(crate) fn r0_table(&self) -> &[f64] {
        &self.r0
    }

    #[inline]
    fn sp_pair(&self, a: usize, b: usize) -> usize {
        a * self.model.species.len() + b
    }

    /// Build the per-geometry cache (tblite `update`): coordination numbers and
    /// the damped two-body radial factors of all pairs.
    pub fn update(&self, positions: &[[f64; 3]]) -> D4Cache {
        let mut cache = D4Cache::default();
        self.update_into(positions, &mut cache);
        cache
    }

    /// Like [`update`](Self::update) but reusing the buffers of `cache`.
    pub fn update_into(&self, positions: &[[f64; 3]], cache: &mut D4Cache) {
        let nat = self.natoms();
        assert!(positions.len() >= nat);
        cache.positions.clear();
        cache.positions.extend_from_slice(&positions[..nat]);
        cache.cn.resize(nat, 0.0);
        self.model.coordination_numbers(positions, self.cutoff.cn, &mut cache.cn);

        cache.pair_offset.clear();
        cache.pair_j.clear();
        cache.pair_e.clear();
        cache.pair_g.clear();
        cache.pair_offset.push(0);
        let p = &self.param;
        let (cutoff, width) = (self.cutoff.disp2, self.cutoff.width2);
        let cutoff2 = cutoff * cutoff;
        let id = &self.model.id;
        for iat in 0..nat {
            let xi = positions[iat];
            for jat in 0..iat {
                let xj = positions[jat];
                let v = [xi[0] - xj[0], xi[1] - xj[1], xi[2] - xj[2]];
                let r2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
                if r2 > cutoff2 || r2 < f64::EPSILON {
                    continue;
                }
                let r = r2.sqrt();
                let (sw, dswdr) = smooth_cutoff(r, cutoff, width);
                if sw <= 0.0 {
                    continue;
                }
                let k = self.sp_pair(id[iat], id[jat]);
                let (rrij, r0ij) = (self.rr[k], self.r0[k]);
                let t6 = 1.0 / (pow3(r2) + pow6(r0ij));
                let t8 = 1.0 / (pow4(r2) + pow8(r0ij));
                let d6 = -(6.0 * (r2 * r2) * (t6 * t6));
                let d8 = -(8.0 * pow3(r2) * (t8 * t8));
                let edisp0 = p.s6 * t6 + p.s8 * rrij * t8;
                let gdisp0 = p.s6 * d6 + p.s8 * rrij * d8;
                cache.pair_j.push(jat as u32);
                cache.pair_e.push(sw * edisp0);
                cache.pair_g.push(sw * gdisp0 + dswdr * edisp0 / r);
            }
            cache.pair_offset.push(cache.pair_j.len());
        }
        cache.pair_t_offset.clear();
        cache.pair_t_offset.resize(nat + 1, 0);
        for &j in &cache.pair_j {
            cache.pair_t_offset[j as usize + 1] += 1;
        }
        for a in 0..nat {
            cache.pair_t_offset[a + 1] += cache.pair_t_offset[a];
        }
        let mut fill = cache.pair_t_offset[..nat].to_vec();
        cache.pair_t.clear();
        cache.pair_t.resize(cache.pair_j.len(), 0);
        for (p, &j) in cache.pair_j.iter().enumerate() {
            cache.pair_t[fill[j as usize]] = p as u32;
            fill[j as usize] += 1;
        }
    }

    /// Per-pair terms `(to the first atom, to the second)` computed on the current rayon pool,
    /// then added to `out` atom by atom in the order of the sequential pair loop (each atom's
    /// own pairs, then the later atoms' pairs with it): the same bits as that loop.
    fn pair_terms_par(
        &self,
        cache: &D4Cache,
        out: &mut [f64],
        term: impl Fn(usize, usize, usize) -> (f64, f64) + Sync,
    ) {
        use rayon::prelude::*;
        let nat = self.natoms();
        let off = &cache.pair_offset;
        let t: Vec<(f64, f64)> = (0..nat)
            .into_par_iter()
            .flat_map_iter(|i| (off[i]..off[i + 1]).map(move |p| (i, p)))
            .map(|(i, p)| term(i, cache.pair_j[p] as usize, p))
            .collect();
        let sums: Vec<f64> = (0..nat)
            .into_par_iter()
            .with_min_len(16)
            .map(|i| {
                let mut x = out[i];
                for p in off[i]..off[i + 1] {
                    x += t[p].0;
                }
                for &p in &cache.pair_t[cache.pair_t_offset[i]..cache.pair_t_offset[i + 1]] {
                    x += t[p as usize].1;
                }
                x
            })
            .collect();
        out[..nat].copy_from_slice(&sums);
    }

    fn check_q(&self, cache: &D4Cache, q: &[f64]) -> usize {
        let nat = self.natoms();
        assert_eq!(cache.cn.len(), nat, "cache was built for a different molecule");
        assert!(q.len() >= nat, "need one charge per atom");
        nat
    }

    /// Self-consistent two-body energy at charges `q`, accumulated per atom into
    /// `energies` (tblite `get_energy`). `q` = atomic partial charges. With `parallel`, on the
    /// current rayon pool (the same bits).
    pub fn energy(&self, cache: &D4Cache, q: &[f64], energies: &mut [f64], parallel: bool) {
        let nat = self.check_q(cache, q);
        let mut gw = vec![[0.0; MAX_REF]; nat];
        self.model.weight_references_into(&cache.cn, q, &mut gw, None, None);
        if parallel {
            return self.pair_terms_par(cache, energies, |iat, jat, p| {
                let de = -(self.model.pair_c6(iat, jat, &gw) * cache.pair_e[p] * 0.5);
                (de, de)
            });
        }
        for iat in 0..nat {
            for p in cache.pair_offset[iat]..cache.pair_offset[iat + 1] {
                let jat = cache.pair_j[p] as usize;
                let c6 = self.model.pair_c6(iat, jat, &gw);
                let de = -(c6 * cache.pair_e[p] * 0.5);
                energies[iat] += de;
                energies[jat] += de;
            }
        }
    }

    /// Total self-consistent two-body energy at charges `q`.
    pub fn energy_total(&self, cache: &D4Cache, q: &[f64]) -> f64 {
        let mut e = vec![0.0; self.natoms()];
        self.energy(cache, q, &mut e, false);
        e.iter().sum()
    }

    /// Charge potential `dE_2b/dq_i`, accumulated into `vat` (tblite
    /// `get_potential`). With `parallel`, on the current rayon pool (the same bits).
    pub fn potential(&self, cache: &D4Cache, q: &[f64], vat: &mut [f64], parallel: bool) {
        self.energy_potential_impl(cache, q, None, vat, parallel);
    }

    /// Two-body energy (per atom, accumulated) and charge potential
    /// (accumulated) in one pass over the pairs.
    pub fn energy_and_potential(&self, cache: &D4Cache, q: &[f64], energies: &mut [f64], vat: &mut [f64]) {
        self.energy_potential_impl(cache, q, Some(energies), vat, false);
    }

    fn energy_potential_impl(
        &self,
        cache: &D4Cache,
        q: &[f64],
        mut energies: Option<&mut [f64]>,
        vat: &mut [f64],
        parallel: bool,
    ) {
        let nat = self.check_q(cache, q);
        assert!(vat.len() >= nat);
        let mut gw = vec![[0.0; MAX_REF]; nat];
        let mut dgwdq = vec![[0.0; MAX_REF]; nat];
        self.model.weight_references_into(&cache.cn, q, &mut gw, None, Some(&mut dgwdq));
        if parallel && energies.is_none() {
            // x - a = x + (-a) exactly: the same bits as the loop below
            return self.pair_terms_par(cache, vat, |iat, jat, p| {
                let (_, dqi, dqj) = self.model.pair_c6_deriv(iat, jat, &gw, &dgwdq);
                let e = cache.pair_e[p];
                (-(dqi * e), -(dqj * e))
            });
        }
        for iat in 0..nat {
            for p in cache.pair_offset[iat]..cache.pair_offset[iat + 1] {
                let jat = cache.pair_j[p] as usize;
                let (c6, dqi, dqj) = self.model.pair_c6_deriv(iat, jat, &gw, &dgwdq);
                let e = cache.pair_e[p];
                vat[iat] -= dqi * e;
                vat[jat] -= dqj * e;
                if let Some(en) = energies.as_deref_mut() {
                    let de = -(c6 * e * 0.5);
                    en[iat] += de;
                    en[jat] += de;
                }
            }
        }
    }

    /// Explicit geometry gradient of the two-body energy at fixed charges `q`
    /// (incl. the dC6/dCN chain through the coordination numbers), accumulated
    /// into `gradient` (tblite `get_gradient`; no dq/dR term).
    pub fn gradient(&self, cache: &D4Cache, q: &[f64], gradient: &mut [[f64; 3]]) {
        let nat = self.check_q(cache, q);
        assert!(gradient.len() >= nat);
        let mut gw = vec![[0.0; MAX_REF]; nat];
        let mut dgwdcn = vec![[0.0; MAX_REF]; nat];
        self.model.weight_references_into(&cache.cn, q, &mut gw, Some(&mut dgwdcn), None);
        let mut dedcn = vec![0.0; nat];
        let pos = &cache.positions;
        for iat in 0..nat {
            for p in cache.pair_offset[iat]..cache.pair_offset[iat + 1] {
                let jat = cache.pair_j[p] as usize;
                let (c6, dcni, dcnj) = self.model.pair_c6_deriv(iat, jat, &gw, &dgwdcn);
                let e = cache.pair_e[p];
                dedcn[iat] -= dcni * e;
                dedcn[jat] -= dcnj * e;
                let g = cache.pair_g[p];
                for k in 0..3 {
                    let dg = -c6 * g * (pos[iat][k] - pos[jat][k]);
                    gradient[iat][k] += dg;
                    gradient[jat][k] -= dg;
                }
            }
        }
        self.model.add_coordination_number_derivs(pos, self.cutoff.cn, &dedcn, gradient);
    }

    /// Non-self-consistent ATM three-body energy evaluated at q = 0, accumulated
    /// per atom into `energies`, with optional gradient (incl. CN chain rule),
    /// accumulated into `gradient` (tblite `get_engrad`). With `parallel`, on the current
    /// rayon pool (the same bits).
    pub fn three_body(&self, cache: &D4Cache, energies: &mut [f64], gradient: Option<&mut [[f64; 3]]>, parallel: bool) {
        let nat = self.natoms();
        assert_eq!(cache.cn.len(), nat, "cache was built for a different molecule");
        if self.param.s9.abs() < f64::EPSILON {
            return;
        }
        let q0 = vec![0.0; nat];
        let mut gw = vec![[0.0; MAX_REF]; nat];
        let mut dgwdcn = vec![[0.0; MAX_REF]; nat];
        let grad = gradient.is_some();
        self.model.weight_references_into(&cache.cn, &q0, &mut gw, if grad { Some(&mut dgwdcn) } else { None }, None);
        let mut dedcn = if grad { vec![0.0; nat] } else { Vec::new() };
        let src = PairC6::Weights { gw: &gw, dgwdcn: &dgwdcn };
        match gradient {
            Some(g) => {
                self.atm_kernel::<true>(cache, src, energies, &mut dedcn, g, parallel);
                self.model.add_coordination_number_derivs(&cache.positions, self.cutoff.cn, &dedcn, g);
            }
            None => {
                self.atm_kernel::<false>(cache, src, energies, &mut dedcn, &mut [], parallel);
            }
        }
    }

    /// ATM three-body energy with caller-supplied pair C6 coefficients (`c6[i*nat + j]`,
    /// symmetric), accumulated per atom into `energies`; same damping, cutoff and kernel as
    /// [`three_body`](Self::three_body), which uses the model's C6 at q = 0. For dispersion
    /// models whose C6 do not follow from per-atom reference weights (e.g. pair-specific
    /// weights). On the calling thread.
    pub fn three_body_c6(&self, cache: &D4Cache, c6: &[f64], energies: &mut [f64]) {
        let nat = self.natoms();
        assert_eq!(cache.cn.len(), nat, "cache was built for a different molecule");
        assert!(c6.len() >= nat * nat, "need a nat x nat C6 matrix");
        if self.param.s9.abs() < f64::EPSILON {
            return;
        }
        self.atm_kernel::<false>(cache, PairC6::Matrix(c6), energies, &mut [], &mut [], false);
    }

    /// [`three_body_c6`](Self::three_body_c6) with gradient: adds the geometric gradient at
    /// fixed C6 to `gradient` and dE/dCN_i (through `dc6dcn[i*nat + j]` = dC6_ij/dCN_i) to
    /// `dedcn`, without the CN derivative of this model (for C6 that depend on another CN).
    /// On the calling thread.
    pub fn three_body_c6_gradient(
        &self,
        cache: &D4Cache,
        c6: &[f64],
        dc6dcn: &[f64],
        energies: &mut [f64],
        gradient: &mut [[f64; 3]],
        dedcn: &mut [f64],
    ) {
        let nat = self.natoms();
        assert_eq!(cache.cn.len(), nat, "cache was built for a different molecule");
        assert!(c6.len() >= nat * nat && dc6dcn.len() >= nat * nat);
        assert!(gradient.len() >= nat && dedcn.len() >= nat);
        if self.param.s9.abs() < f64::EPSILON {
            return;
        }
        self.atm_kernel::<true>(cache, PairC6::MatrixDeriv { c6, dc6dcn }, energies, dedcn, gradient, false);
    }

    /// Half neighbour list of the three-body term: for atom a all b < a inside the cutoff,
    /// ascending, with the per-pair quantities of [`PairList`]; row offsets.
    fn atm_pairs<const GRAD: bool>(&self, cache: &D4Cache, src: PairC6) -> (Vec<usize>, PairList) {
        let nat = self.natoms();
        let pos = &cache.positions;
        let p = &self.param;
        let (cutoff, width) = (self.cutoff.disp3, self.cutoff.width3);
        let cutoff2 = cutoff * cutoff;
        let alp3 = p.alp / 3.0;
        let id = &self.model.id;
        let mut offs = Vec::with_capacity(nat + 1);
        let mut nb = PairList::default();
        offs.push(0usize);
        for a in 0..nat {
            for b in 0..a {
                let v = [pos[b][0] - pos[a][0], pos[b][1] - pos[a][1], pos[b][2] - pos[a][2]];
                let r2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
                if r2 > cutoff2 || r2 < f64::EPSILON {
                    continue;
                }
                let r = r2.sqrt();
                let (sw, dsw) = smooth_cutoff(r, cutoff, width);
                // r0 of the pair in dftd4's species order (smaller index first)
                let r0 = self.r0[self.sp_pair(id[b], id[a])];
                let (c6, dca, dcb) = match src {
                    PairC6::Weights { gw, dgwdcn } if GRAD => self.model.pair_c6_deriv(a, b, gw, dgwdcn),
                    PairC6::Weights { gw, .. } => (self.model.pair_c6(a, b, gw), 0.0, 0.0),
                    PairC6::Matrix(m) => {
                        debug_assert!(!GRAD, "C6 matrix without derivatives");
                        (m[a * nat + b], 0.0, 0.0)
                    }
                    PairC6::MatrixDeriv { c6, dc6dcn } => (c6[a * nat + b], dc6dcn[a * nat + b], dc6dcn[b * nat + a]),
                };
                nb.j.push(b as u32);
                nb.v.push(v);
                nb.r2.push(r2);
                nb.ir2.push(1.0 / r2);
                nb.ir.push(1.0 / r);
                nb.sw.push(sw);
                nb.dswr.push(dsw / r);
                nb.t.push((r0 / r).powf(alp3));
                nb.c6.push(c6);
                nb.qa.push(dca / c6);
                nb.qb.push(dcb / c6);
            }
            offs.push(nb.j.len());
        }

        (offs, nb)
    }

    /// Three-body energy (and gradient) by pair sums ([`D4Dispersion::with_pair_sums`], xtb
    /// 6.8.0's `atm_gradient_latp`). Every atom a keeps its neighbours' pair quantities in a
    /// dense window over the atom indices `lo[a]..hi[a]` (zeros where a pair is outside the
    /// cutoff, so those terms vanish): r², 1/r, (r0/r)^(α/3) and √C6·sw. For a pair (a, b) the
    /// loop over the third atoms k of both windows sums the triple energies divided by sw_ab
    /// (`d`) and their r_ab derivatives at fixed sw_ab (`g`); sw_ab and its derivative enter
    /// once per pair. Per pair, de = sw d (all its triples) gives de/6 to the energy of each
    /// atom and de/2 to dE/dC6_ab; (dE/dr)/r = sw g/r² − (dsw/dr)/r d.
    fn atm_pair_sums<const GRAD: bool>(
        &self,
        cache: &D4Cache,
        src: PairC6,
        energies: &mut [f64],
        dedcn: &mut [f64],
        gradient: &mut [[f64; 3]],
        parallel: bool,
    ) {
        let nat = self.natoms();
        let (offs, nb) = self.atm_pairs::<GRAD>(cache, src);
        // pairs inside the cutoff with a nonzero switch, and each one's row atom
        let pairs: Vec<(u32, u32)> = (0..nat)
            .flat_map(|a| (offs[a]..offs[a + 1]).map(move |e| (a as u32, e as u32)))
            .filter(|&(_, e)| nb.sw[e as usize] > 0.0)
            .collect();
        // windows: every neighbour k of a lies in lo[a]..hi[a]
        let (mut lo, mut hi) = (vec![usize::MAX; nat], vec![0usize; nat]);
        for &(a, e) in &pairs {
            let (a, b) = (a as usize, nb.j[e as usize] as usize);
            lo[a] = lo[a].min(b);
            hi[a] = hi[a].max(b + 1);
            lo[b] = lo[b].min(a);
            hi[b] = hi[b].max(a + 1);
        }
        let mut base = vec![0usize; nat + 1];
        for a in 0..nat {
            if hi[a] == 0 {
                lo[a] = 0;
            }
            base[a + 1] = base[a] + (hi[a] - lo[a]);
        }
        let len = base[nat];
        let (mut wr2, mut wir, mut wt, mut ww) = (vec![0.0; len], vec![0.0; len], vec![0.0; len], vec![0.0; len]);
        for &(a, e) in &pairs {
            let (a, e) = (a as usize, e as usize);
            let b = nb.j[e] as usize;
            let w = nb.c6[e].abs().sqrt() * nb.sw[e];
            for x in [base[a] + b - lo[a], base[b] + a - lo[b]] {
                wr2[x] = nb.r2[e];
                wir[x] = nb.ir[e];
                wt[x] = nb.t[e];
                ww[x] = w;
            }
        }
        let alp3 = self.param.alp / 3.0;
        // (g, d) of one pair, summed over the third atoms in LANES fixed lanes
        let pair_sum = |&(a, e): &(u32, u32)| -> (f64, f64) {
            let (a, e) = (a as usize, e as usize);
            let b = nb.j[e] as usize;
            let (k0, k1) = (lo[a].max(lo[b]), hi[a].min(hi[b]));
            if k1 <= k0 {
                return (0.0, 0.0);
            }
            let n = k1 - k0;
            let (ia, ib) = (base[a] + k0 - lo[a], base[b] + k0 - lo[b]);
            let (ar2, air, at, aw) = (&wr2[ia..ia + n], &wir[ia..ia + n], &wt[ia..ia + n], &ww[ia..ia + n]);
            let (br2, bir, bt, bw) = (&wr2[ib..ib + n], &wir[ib..ib + n], &wt[ib..ib + n], &ww[ib..ib + n]);
            let (mut gl, mut dl) = ([0.0; LANES], [0.0; LANES]);
            let cols = [ar2, br2, air, bir, at, bt, aw, bw];
            let pair = [nb.r2[e], nb.ir[e], 6.0 * nb.t[e], alp3];
            // blocks of BLOCK third atoms: the terms (a vectorized loop), then their lane sums
            // (a short last block padded with zeros)
            let (mut tg, mut td) = ([0.0; BLOCK], [0.0; BLOCK]);
            for k0 in (0..n).step_by(BLOCK) {
                let m = (n - k0).min(BLOCK);
                if m < BLOCK {
                    tg[m..].fill(0.0);
                    td[m..].fill(0.0);
                }
                atm_terms::<GRAD>(pair, cols.map(|s| &s[k0..k0 + m]), &mut tg[..m], &mut td[..m]);
                for c in 0..BLOCK / LANES {
                    for l in 0..LANES {
                        gl[l] += tg[c * LANES + l];
                        dl[l] += td[c * LANES + l];
                    }
                }
            }
            let cab = -(self.param.s9 * nb.c6[e].abs().sqrt());
            let (mut g, mut d) = (0.0, 0.0);
            for l in 0..LANES {
                g += gl[l];
                d += dl[l];
            }
            (g * cab, d * cab)
        };
        let sums: Vec<(f64, f64)> = if parallel {
            use rayon::prelude::*;
            pairs.par_iter().with_min_len(64).map(pair_sum).collect()
        } else {
            pairs.iter().map(pair_sum).collect()
        };
        const SIXTH: f64 = 1.0 / 6.0;
        for (&(a, e), &(g, d)) in pairs.iter().zip(&sums) {
            let (a, e) = (a as usize, e as usize);
            let b = nb.j[e] as usize;
            let de = nb.sw[e] * d;
            energies[a] -= de * SIXTH;
            energies[b] -= de * SIXTH;
            if GRAD {
                // pair (a, b < a): dE/dR_a = -(dE/dr)/r (R_b - R_a), dE/dR_b the opposite
                let f = nb.sw[e] * g * nb.ir2[e] - nb.dswr[e] * d;
                let v = nb.v[e];
                for x in 0..3 {
                    gradient[a][x] -= f * v[x];
                    gradient[b][x] += f * v[x];
                }
                let h = 0.5 * de;
                dedcn[a] -= h * nb.qa[e];
                dedcn[b] -= h * nb.qb[e];
            }
        }
    }

    /// ATM kernel over all triples i > j > k with every distance inside the
    /// three-body cutoff (same enumeration order as dftd4). Per-pair quantities
    /// (distance, switching function, zero-damping power, C6 and dC6/dCN) are
    /// precomputed on a half neighbour list; the pair (i, k) of a triple is
    /// looked up through a scatter array of row i while k runs over the list of
    /// j, so the triple loop does no pow/smoothing and never touches pairs
    /// outside the cutoff.
    fn atm_kernel<const GRAD: bool>(
        &self,
        cache: &D4Cache,
        src: PairC6,
        energies: &mut [f64],
        dedcn: &mut [f64],
        gradient: &mut [[f64; 3]],
        parallel: bool,
    ) {
        if self.pair_sums {
            return self.atm_pair_sums::<GRAD>(cache, src, energies, dedcn, gradient, parallel);
        }
        let nat = self.natoms();
        let p = &self.param;
        const THIRD: f64 = 1.0 / 3.0;
        let (offs, nb) = self.atm_pairs::<GRAD>(cache, src);

        // slot[k] = entry (i, k) of the current row i, or NONE
        const NONE: u32 = u32::MAX;
        let triples =
            |rows: std::ops::Range<usize>, energies: &mut [f64], dedcn: &mut [f64], gradient: &mut [[f64; 3]]| {
                let mut slot = vec![NONE; nat];
                for iat in rows {
                    let (i0, i1) = (offs[iat], offs[iat + 1]);
                    for e in i0..i1 {
                        slot[nb.j[e] as usize] = e as u32;
                    }
                    for eij in i0..i1 {
                        let jat = nb.j[eij] as usize;
                        let (vij, r2ij, ir2ij, irij) = (nb.v[eij], nb.r2[eij], nb.ir2[eij], nb.ir[eij]);
                        let (swij, dswij, tij, c6ij) = (nb.sw[eij], nb.dswr[eij], nb.t[eij], nb.c6[eij]);
                        // k runs over the neighbours of j (k < j, ascending) that are
                        // also neighbours of i
                        for e_jk in offs[jat]..offs[jat + 1] {
                            let kat = nb.j[e_jk] as usize;
                            let e_ik = slot[kat];
                            if e_ik == NONE {
                                continue;
                            }
                            let e_ik = e_ik as usize;

                            let (swik, swjk) = (nb.sw[e_ik], nb.sw[e_jk]);
                            let sw = swij * swik * swjk;
                            if sw <= 0.0 {
                                continue;
                            }
                            let (vik, r2ik, ir2ik, dswik) = (nb.v[e_ik], nb.r2[e_ik], nb.ir2[e_ik], nb.dswr[e_ik]);
                            let (vjk, r2jk, ir2jk, dswjk) = (nb.v[e_jk], nb.r2[e_jk], nb.ir2[e_jk], nb.dswr[e_jk]);
                            let (c6ik, c6jk) = (nb.c6[e_ik], nb.c6[e_jk]);
                            let c9 = -(p.s9 * (c6ij * c6ik * c6jk).abs().sqrt());
                            // i > j > k: all distinct, triple scale 1

                            // 1/r^3 and 1/r^5 with r = rij rik rjk
                            let ir2 = ir2ij * ir2ik * ir2jk;
                            let ir1 = irij * nb.ir[e_ik] * nb.ir[e_jk];
                            let ir3 = ir2 * ir1;
                            let ir5 = ir3 * ir2;
                            // (r0ij r0ik r0jk / (rij rik rjk))^(alp/3)
                            let rpow = tij * nb.t[e_ik] * nb.t[e_jk];
                            let fdmp = 1.0 / (1.0 + 6.0 * rpow);
                            let ang =
                                0.375 * (r2ij + r2jk - r2ik) * (r2ij - r2jk + r2ik) * (-r2ij + r2jk + r2ik) * ir5 + ir3;
                            let rr = ang * fdmp;

                            if !GRAD {
                                let de = rr * c9 * THIRD * sw;
                                energies[iat] -= de;
                                energies[jat] -= de;
                                energies[kat] -= de;
                                continue;
                            }

                            let dfdmp = -(2.0 * p.alp * rpow * (fdmp * fdmp));
                            let de0 = rr * c9;
                            let swc9 = sw * c9;

                            let dang = -((0.375
                                * (pow3(r2ij)
                                    + (r2ij * r2ij) * (r2jk + r2ik)
                                    + r2ij * (3.0 * (r2jk * r2jk) + 2.0 * r2jk * r2ik + 3.0 * (r2ik * r2ik))
                                    - 5.0 * ((r2jk - r2ik) * (r2jk - r2ik)) * (r2jk + r2ik)))
                                * ir5);
                            let fij = swc9 * (-(dang * fdmp) + ang * dfdmp) * ir2ij - de0 * dswij * swik * swjk;
                            let dang = -((0.375
                                * (pow3(r2ik)
                                    + (r2ik * r2ik) * (r2jk + r2ij)
                                    + r2ik * (3.0 * (r2jk * r2jk) + 2.0 * r2jk * r2ij + 3.0 * (r2ij * r2ij))
                                    - 5.0 * ((r2jk - r2ij) * (r2jk - r2ij)) * (r2jk + r2ij)))
                                * ir5);
                            let fik = swc9 * (-(dang * fdmp) + ang * dfdmp) * ir2ik - de0 * dswik * swij * swjk;
                            let dang = -((0.375
                                * (pow3(r2jk)
                                    + (r2jk * r2jk) * (r2ik + r2ij)
                                    + r2jk * (3.0 * (r2ik * r2ik) + 2.0 * r2ik * r2ij + 3.0 * (r2ij * r2ij))
                                    - 5.0 * ((r2ik - r2ij) * (r2ik - r2ij)) * (r2ik + r2ij)))
                                * ir5);
                            let fjk = swc9 * (-(dang * fdmp) + ang * dfdmp) * ir2jk - de0 * dswjk * swij * swik;

                            let de = de0 * sw;
                            let de_third = de * THIRD;
                            energies[iat] -= de_third;
                            energies[jat] -= de_third;
                            energies[kat] -= de_third;
                            for x in 0..3 {
                                let dgij = fij * vij[x];
                                let dgik = fik * vik[x];
                                let dgjk = fjk * vjk[x];
                                gradient[iat][x] -= dgij + dgik;
                                gradient[jat][x] += dgij - dgjk;
                                gradient[kat][x] += dgik + dgjk;
                            }
                            // qa = (dC6_ab/dCN_a)/C6_ab for the entry (a, b < a)
                            let h = de * 0.5;
                            dedcn[iat] -= h * (nb.qa[eij] + nb.qa[e_ik]);
                            dedcn[jat] -= h * (nb.qb[eij] + nb.qa[e_jk]);
                            dedcn[kat] -= h * (nb.qb[e_ik] + nb.qb[e_jk]);
                        }
                    }
                    for e in i0..i1 {
                        slot[nb.j[e] as usize] = NONE;
                    }
                }
            };

        // Rows in fixed chunks of about equal pair count, each accumulated separately
        // and added in chunk order. Serial and parallel runs use the same chunks, so the
        // result does not depend on the thread count.
        const NCHUNK: usize = 64;
        let npairs = offs[nat].max(1);
        let mut bounds = Vec::with_capacity(NCHUNK);
        let mut start = 0;
        for c in 1..=NCHUNK {
            let target = npairs * c / NCHUNK;
            let mut end = start;
            while end < nat && (offs[end + 1] <= target || c == NCHUNK) {
                end += 1;
            }
            if end > start {
                bounds.push((start, end));
                start = end;
            }
        }
        if start < nat {
            bounds.push((start, nat));
        }
        let ng = if GRAD { nat } else { 0 };
        let run = |&(a, b): &(usize, usize)| {
            let mut e = vec![0.0; nat];
            let mut d = vec![0.0; ng];
            let mut g = vec![[0.0; 3]; ng];
            triples(a..b, &mut e, &mut d, &mut g);
            (e, d, g)
        };
        let mut add = |(e, d, g): (Vec<f64>, Vec<f64>, Vec<[f64; 3]>)| {
            for i in 0..nat {
                energies[i] += e[i];
            }
            if GRAD {
                for i in 0..nat {
                    dedcn[i] += d[i];
                    for x in 0..3 {
                        gradient[i][x] += g[i][x];
                    }
                }
            }
        };
        if parallel {
            use rayon::prelude::*;
            let parts: Vec<_> = bounds.par_iter().map(run).collect();
            parts.into_iter().for_each(&mut add);
        } else {
            bounds.iter().map(run).for_each(&mut add);
        }
    }

    /// Full non-self-consistent D4 energy (two-body at fixed charges `q` plus
    /// three-body at q = 0) per atom, accumulated into `energies`, and optionally
    /// the gradient at fixed charges accumulated into `gradient`. Returns the
    /// total energy contributed by this call. On the calling thread.
    pub fn engrad_fixed_charges(
        &self,
        positions: &[[f64; 3]],
        q: &[f64],
        energies: &mut [f64],
        gradient: Option<&mut [[f64; 3]]>,
    ) -> f64 {
        let nat = self.natoms();
        let cache = self.update(positions);
        let mut e = vec![0.0; nat];
        self.energy(&cache, q, &mut e, false);
        match gradient {
            Some(g) => {
                self.gradient(&cache, q, g);
                self.three_body(&cache, &mut e, Some(g), false);
            }
            None => self.three_body(&cache, &mut e, None, false),
        }
        let mut total = 0.0;
        for i in 0..nat {
            energies[i] += e[i];
            total += e[i];
        }
        total
    }
}

/// Lane sums and block length of the three-body pair sums ([`D4Dispersion::with_pair_sums`]).
const LANES: usize = 4;
const BLOCK: usize = 64;

/// The three-body terms of the pair (a, b) for a block of third atoms k (see
/// `atm_pair_sums`): `d[k]` = triple energy / (s9 √C6_ab sw_ab), `g[k]` = dftd4's
/// `(−dang fdmp + dfdmp ang) c9` with the same factor (gradient only). `pair` = r_ab², 1/r_ab,
/// 6 (r0/r)_ab^(α/3), α/3; `x` = r², 1/r and (r0/r)^(α/3) of (a, k) and (b, k), then √C6·sw of
/// (a, k) and (b, k), all of the length of `d`. A plain loop over k, which vectorizes.
#[inline(always)]
fn atm_terms<const GRAD: bool>(pair: [f64; 4], x: [&[f64]; 8], g: &mut [f64], d: &mut [f64]) {
    let [r2, ir, t6, alp3] = pair;
    let r2_2 = r2 * r2;
    let m = d.len();
    let [r2ak, r2bk, irak, irbk, tak, tbk, wak, wbk] = x.map(|s| &s[..m]);
    let g = &mut g[..m];
    for k in 0..m {
        let c = wak[k] * wbk[k];
        let cralp = t6 * tak[k] * tbk[k];
        let fdmp = 1.0 / (1.0 + cralp);
        let ir1 = ir * irak[k] * irbk[k];
        let rrr2 = ir1 * ir1;
        let rrr3 = rrr2 * ir1;
        let angf = 0.375 * rrr2 * rrr3;
        let s1 = r2ak[k] + r2bk[k];
        let d1 = (r2bk[k] - r2ak[k]) * (r2bk[k] - r2ak[k]);
        let ang = angf * ((r2_2 - d1) * (s1 - r2)) + rrr3;
        let cf = c * fdmp;
        d[k] = ang * cf;
        if GRAD {
            // dftd4's dang = -angf p and dfdmp = -(α/3) cralp fdmp²
            let p = r2 * (r2 * (r2 + s1) + 2.0 * (s1 * s1) + d1) - 5.0 * (d1 * s1);
            g[k] = cf * (angf * p - alp3 * (cralp * fdmp) * ang);
        }
    }
}
