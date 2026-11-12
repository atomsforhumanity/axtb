// SPDX-License-Identifier: LGPL-3.0-or-later
//! Second derivatives of the self-consistent D4 dispersion: the charge kernel
//! `d²E/dq dq`, the geometry derivative of the charge potential at fixed charges, and
//! the nuclear Hessian at fixed charges (two-body term with the coordination-number and
//! charge dependence of the C6 coefficients, three-body ATM term at zero charges).
//!
//! Conventions as in the rest of the crate: positions in bohr, row-major dense
//! matrices, results accumulated with `+=`.

use crate::data::MAX_REF;
use crate::disp::{smooth_cutoff, D4Cache, D4Dispersion, RationalDamping};
use crate::model::D4Model;
use axtb_base::math::add_radial_hessian;

/// `zeta`, `dzeta/dq` and `d²zeta/dq²`.
#[inline]
fn zeta2(a: f64, c: f64, qref: f64, qmod: f64) -> (f64, f64, f64) {
    if qmod < 0.0 {
        (a.exp(), 0.0, 0.0)
    } else {
        let e = (c * (1.0 - qref / qmod)).exp();
        let z = (a * (1.0 - e)).exp();
        let u = c * qref / (qmod * qmod);
        let du = -2.0 * c * qref / (qmod * qmod * qmod);
        (z, -a * z * e * u, a * z * e * ((a * e - 1.0) * u * u - du))
    }
}

/// Smooth cutoff with its second derivative.
#[inline]
fn smooth2(r: f64, cutoff: f64, width: f64) -> (f64, f64, f64) {
    let (sw, dsw) = smooth_cutoff(r, cutoff, width);
    if width <= 0.0 || cutoff <= 0.0 {
        return (sw, dsw, 0.0);
    }
    let w = width.min(cutoff);
    if r <= cutoff - w || r >= cutoff {
        return (sw, dsw, 0.0);
    }
    let x = (cutoff - r) / w;
    (sw, dsw, 60.0 * x * (1.0 - x) * (1.0 - 2.0 * x) / (w * w))
}

/// Reference weights with their first and second derivatives.
pub(crate) struct Weights2 {
    pub gw: Vec<[f64; MAX_REF]>,
    pub dcn: Vec<[f64; MAX_REF]>,
    pub dq: Vec<[f64; MAX_REF]>,
    pub dcn2: Vec<[f64; MAX_REF]>,
    pub dq2: Vec<[f64; MAX_REF]>,
    pub dcnq: Vec<[f64; MAX_REF]>,
}

impl D4Model {
    /// Reference weights with first and second CN and charge derivatives (same edge
    /// cases as [`weight_references_into`](Self::weight_references_into)).
    pub(crate) fn weights2(&self, cn: &[f64], q: &[f64]) -> Weights2 {
        let nat = self.natoms();
        let z = [0.0; MAX_REF];
        let mut w = Weights2 {
            gw: vec![z; nat],
            dcn: vec![z; nat],
            dq: vec![z; nat],
            dcn2: vec![z; nat],
            dq2: vec![z; nat],
            dcnq: vec![z; nat],
        };
        let eps_norm = f64::MIN_POSITIVE.sqrt();
        for iat in 0..nat {
            let isp = self.id[iat];
            let nref = self.nref[isp];
            let zi = self.zeff[isp];
            let gi = self.eta[isp] * self.gc;
            let cni = cn[iat];
            let refcn = &self.refcn[isp];
            let ngw = &self.ngw[isp];
            let gauss = |iref: usize| -> (f64, f64, f64) {
                let (mut g0, mut g1, mut g2) = (0.0, 0.0, 0.0);
                for igw in 1..=ngw[iref] {
                    let wf = igw as f64 * self.wf;
                    let d = cni - refcn[iref];
                    let g = (-(wf * (d * d))).exp();
                    let t = 2.0 * wf * (refcn[iref] - cni);
                    g0 += g;
                    g1 += t * g;
                    g2 += (t * t - 2.0 * wf) * g;
                }
                (g0, g1, g2)
            };
            let (mut norm, mut dnorm, mut d2norm) = (0.0, 0.0, 0.0);
            for iref in 0..nref {
                let (a, b, c) = gauss(iref);
                norm += a;
                dnorm += b;
                d2norm += c;
            }
            let ni = if norm.abs() > eps_norm { 1.0 / norm } else { 0.0 };
            let maxcn = (0..nref).map(|r| refcn[r]).fold(f64::NEG_INFINITY, f64::max);
            for iref in 0..nref {
                let (expw, expd, expdd) = gauss(iref);
                let mut gwk = expw * ni;
                let mut d1 = ni * (expd - expw * dnorm * ni);
                let mut d2 = expdd * ni - 2.0 * expd * dnorm * ni * ni - expw * d2norm * ni * ni
                    + 2.0 * expw * dnorm * dnorm * ni * ni * ni;
                if !gwk.is_finite() || ni == 0.0 {
                    gwk = if (maxcn - refcn[iref]).abs() < 1e-12 { 1.0 } else { 0.0 };
                }
                if !d1.is_finite() || ni == 0.0 {
                    d1 = 0.0;
                }
                if !d2.is_finite() || ni == 0.0 {
                    d2 = 0.0;
                }
                let qref = self.refq[isp][iref] + zi;
                let qmod = q[iat] + zi;
                let (zz, dz, d2z) = zeta2(self.ga, gi, qref, qmod);
                w.gw[iat][iref] = gwk * zz;
                w.dcn[iat][iref] = d1 * zz;
                w.dq[iat][iref] = gwk * dz;
                w.dcn2[iat][iref] = d2 * zz;
                w.dq2[iat][iref] = gwk * d2z;
                w.dcnq[iat][iref] = d1 * dz;
            }
        }
        w
    }

    /// `sum_ab wi_a wj_b C6ref_ab` for the pair (i, j).
    #[inline]
    pub(crate) fn c6x(&self, iat: usize, jat: usize, wi: &[f64; MAX_REF], wj: &[f64; MAX_REF]) -> f64 {
        let (isp, jsp) = (self.id[iat], self.id[jat]);
        let blk = self.c6_block(isp, jsp);
        let mut c6 = 0.0;
        for iref in 0..self.nref[isp] {
            let mut acc = 0.0;
            for jref in 0..self.nref[jsp] {
                acc += wj[jref] * blk[iref][jref];
            }
            c6 += wi[iref] * acc;
        }
        c6
    }

    /// Jacobian `dCN_i/dR_x` of the D4 coordination numbers (row-major `nat x 3 nat`).
    pub fn coordination_number_jacobian(&self, positions: &[[f64; 3]], cutoff: f64) -> Vec<f64> {
        self.cn_model(cutoff).jacobian(&positions[..self.natoms()])
    }

    /// Add `sum_i w_i d²CN_i/dR dR` of the D4 coordination numbers to `hess`.
    pub fn add_coordination_number_hessian(&self, positions: &[[f64; 3]], cutoff: f64, w: &[f64], hess: &mut [f64]) {
        self.cn_model(cutoff).add_hessian(&positions[..self.natoms()], w, hess);
    }
}

#[inline]
fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Second-order forward-mode dual number in three variables (value, gradient, packed
/// symmetric Hessian `[00, 01, 02, 11, 12, 22]`).
#[derive(Clone, Copy, Debug, Default)]
struct D2 {
    v: f64,
    g: [f64; 3],
    h: [f64; 6],
}

const PK: [(usize, usize); 6] = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)];

impl D2 {
    fn var(v: f64, i: usize) -> Self {
        let mut g = [0.0; 3];
        g[i] = 1.0;
        D2 { v, g, h: [0.0; 6] }
    }
    fn cst(v: f64) -> Self {
        D2 { v, g: [0.0; 3], h: [0.0; 6] }
    }
    fn add(self, o: Self) -> Self {
        D2 {
            v: self.v + o.v,
            g: std::array::from_fn(|k| self.g[k] + o.g[k]),
            h: std::array::from_fn(|k| self.h[k] + o.h[k]),
        }
    }
    fn sub(self, o: Self) -> Self {
        D2 {
            v: self.v - o.v,
            g: std::array::from_fn(|k| self.g[k] - o.g[k]),
            h: std::array::from_fn(|k| self.h[k] - o.h[k]),
        }
    }
    fn scale(self, s: f64) -> Self {
        D2 { v: self.v * s, g: self.g.map(|x| x * s), h: self.h.map(|x| x * s) }
    }
    fn mul(self, o: Self) -> Self {
        D2 {
            v: self.v * o.v,
            g: std::array::from_fn(|k| self.g[k] * o.v + self.v * o.g[k]),
            h: std::array::from_fn(|q| {
                let (k, l) = PK[q];
                self.h[q] * o.v + self.g[k] * o.g[l] + self.g[l] * o.g[k] + self.v * o.h[q]
            }),
        }
    }
    /// `f(self)` given `f`, `f'`, `f''` at the value.
    fn map(self, f: f64, f1: f64, f2: f64) -> Self {
        D2 {
            v: f,
            g: self.g.map(|x| f1 * x),
            h: std::array::from_fn(|q| {
                let (k, l) = PK[q];
                f2 * self.g[k] * self.g[l] + f1 * self.h[q]
            }),
        }
    }
    fn powf(self, p: f64) -> Self {
        let x = self.v;
        self.map(x.powf(p), p * x.powf(p - 1.0), p * (p - 1.0) * x.powf(p - 2.0))
    }
    fn recip(self) -> Self {
        let x = self.v;
        self.map(1.0 / x, -1.0 / (x * x), 2.0 / (x * x * x))
    }
}

/// Per-pair data of the three-body term (pair (a, b) with b < a inside the cutoff).
#[derive(Clone, Copy)]
struct AtmPair {
    r2: f64,
    /// switching function of r and its first and second derivatives in r²
    sw: [f64; 3],
    /// (r0)^(alp/3)
    r0p: f64,
    /// C6 at q = 0 and its CN derivatives: d/dCN_a, d/dCN_b, d²/dCN_a², d²/dCN_b², d²/dCN_a dCN_b
    c6: f64,
    d: [f64; 5],
}

/// Half neighbour list of the three-body Hessian: for every atom a its pairs (a, b < a) inside
/// the cutoff, b ascending (entries `off[a]..off[a + 1]`).
struct AtmPairs {
    off: Vec<usize>,
    j: Vec<u32>,
    p: Vec<AtmPair>,
}

/// One triple (i, j, k), i > j > k, of the three-body Hessian as its accumulation needs it,
/// with Γ the damped angular term times the three switching functions, its derivatives Γ_u,
/// Γ_uv in the squared distances u, v of (r_ij², r_ik², r_jk²), c9 = s9 √|C6_ij C6_ik C6_jk|
/// and the first (`lm`) and second (`dl`) derivatives of ln c9 in the CNs of (i, j, k).
#[derive(Clone, Copy, Default)]
struct AtmTriple {
    /// Γ_uv c9 (packed like [`D2`]) and 2 Γ_u c9: the factors of ∇u ∇vᵀ and ∇²u
    guv: [f64; 6],
    gv: [f64; 3],
    /// Γ_u and Γ c9
    g: [f64; 3],
    vc9: f64,
    c9: f64,
    lm: [f64; 3],
    /// lower triangle, row by row
    dl: [f64; 6],
}

/// The three-body Hessian of one chunk of triples: Cartesian part (3N x 3N, only the blocks
/// (a, b) with b >= a), X = d²E/dR dCN (3N x N), F = d²E/dCN dCN (N x N, b >= a) and
/// f = dE/dCN.
struct AtmAcc {
    hess: Vec<f64>,
    x: Vec<f64>,
    fcc: Vec<f64>,
    fc: Vec<f64>,
}

/// How the three-body Hessian divides its work ([`ATM_LIMITS`]; smaller in tests). The
/// results do not depend on it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct AtmLimits {
    /// Memory (bytes) of chunk accumulators that may be held at once: below it the 16 chunks
    /// run in parallel, each with its own accumulator.
    pub small: usize,
    /// Capacity (possible triples) of a batch.
    pub batch: usize,
    /// Batches with fewer possible triples are evaluated and added on the calling thread.
    pub par_min: usize,
}

/// The limits of [`D4Dispersion::hessian`]: 32 MB of chunk accumulators, batches of about
/// 3 MB of [`AtmTriple`].
pub(crate) const ATM_LIMITS: AtmLimits = AtmLimits { small: 32 << 20, batch: 1 << 14, par_min: 1 << 12 };
/// Index pairs (local atoms) of the three pair distances of a triple (i, j, k).
const ATM_LAP: [(usize, usize); 3] = [(0, 1), (0, 2), (1, 2)];

impl D4Dispersion {
    /// Second derivatives of the two-body energy with respect to the atomic charges,
    /// added to `k` (row-major `nat x nat`).
    pub fn charge_hessian(&self, cache: &D4Cache, q: &[f64], k: &mut [f64]) {
        let m = self.model();
        let nat = m.natoms();
        let w = m.weights2(&cache.cn, q);
        for iat in 0..nat {
            for p in cache.pair_offset[iat]..cache.pair_offset[iat + 1] {
                let jat = cache.pair_j[p] as usize;
                let e = cache.pair_e[p];
                let kij = -e * m.c6x(iat, jat, &w.dq[iat], &w.dq[jat]);
                k[iat * nat + jat] += kij;
                k[jat * nat + iat] += kij;
                k[iat * nat + iat] -= e * m.c6x(iat, jat, &w.dq2[iat], &w.gw[jat]);
                k[jat * nat + jat] -= e * m.c6x(iat, jat, &w.gw[iat], &w.dq2[jat]);
            }
        }
    }

    /// Derivatives of the charge potential `dE/dq_i` with respect to the nuclear
    /// positions at fixed charges (including the CN dependence), added to `dv`
    /// (row-major `nat x 3 nat`).
    pub fn potential_derivative(&self, cache: &D4Cache, q: &[f64], dv: &mut [f64]) {
        let m = self.model();
        let nat = m.natoms();
        let n3 = 3 * nat;
        let pos = cache.positions();
        let w = m.weights2(&cache.cn, q);
        let mut dvcn = vec![0.0; nat * nat];
        for iat in 0..nat {
            for p in cache.pair_offset[iat]..cache.pair_offset[iat + 1] {
                let jat = cache.pair_j[p] as usize;
                let (e, g) = (cache.pair_e[p], cache.pair_g[p]);
                let v = sub(pos[iat], pos[jat]);
                let ai = m.c6x(iat, jat, &w.dq[iat], &w.gw[jat]);
                let aj = m.c6x(iat, jat, &w.gw[iat], &w.dq[jat]);
                for k in 0..3 {
                    let gv = g * v[k];
                    dv[iat * n3 + 3 * iat + k] -= gv * ai;
                    dv[iat * n3 + 3 * jat + k] += gv * ai;
                    dv[jat * n3 + 3 * iat + k] -= gv * aj;
                    dv[jat * n3 + 3 * jat + k] += gv * aj;
                }
                dvcn[iat * nat + iat] -= e * m.c6x(iat, jat, &w.dcnq[iat], &w.gw[jat]);
                dvcn[iat * nat + jat] -= e * m.c6x(iat, jat, &w.dq[iat], &w.dcn[jat]);
                dvcn[jat * nat + jat] -= e * m.c6x(iat, jat, &w.gw[iat], &w.dcnq[jat]);
                dvcn[jat * nat + iat] -= e * m.c6x(iat, jat, &w.dcn[iat], &w.dq[jat]);
            }
        }
        let jac = m.coordination_number_jacobian(pos, self.cutoff().cn);
        for i in 0..nat {
            for c in 0..nat {
                let x = dvcn[i * nat + c];
                if x != 0.0 {
                    for y in 0..n3 {
                        dv[i * n3 + y] += x * jac[c * n3 + y];
                    }
                }
            }
        }
    }

    /// Radial factor of the two-body pair energy `E = -C6 e(r)` with its first and
    /// second derivatives.
    fn pair_radial(&self, iat: usize, jat: usize, r: f64) -> (f64, f64, f64) {
        let m = self.model();
        let p = self.param();
        let k = m.id[iat] * m.species.len() + m.id[jat];
        let (rr, r0) = (self.rr_table()[k], self.r0_table()[k]);
        let (cutoff, width) = (self.cutoff().disp2, self.cutoff().width2);
        let (sw, sw1, sw2) = smooth2(r, cutoff, width);
        let r2 = r * r;
        let r4 = r2 * r2;
        let r6 = r4 * r2;
        let t6 = 1.0 / (r6 + r0.powi(6));
        let t8 = 1.0 / (r6 * r2 + r0.powi(8));
        let t6_1 = -6.0 * r4 * r * t6 * t6;
        let t6_2 = -30.0 * r4 * t6 * t6 + 72.0 * r6 * r4 * t6 * t6 * t6;
        let t8_1 = -8.0 * r6 * r * t8 * t8;
        let t8_2 = -56.0 * r6 * t8 * t8 + 128.0 * r6 * r6 * r2 * t8 * t8 * t8;
        let e0 = p.s6 * t6 + p.s8 * rr * t8;
        let e1 = p.s6 * t6_1 + p.s8 * rr * t8_1;
        let e2 = p.s6 * t6_2 + p.s8 * rr * t8_2;
        (sw * e0, sw1 * e0 + sw * e1, sw2 * e0 + 2.0 * sw1 * e1 + sw * e2)
    }

    /// Nuclear Hessian of the two-body energy at fixed charges `q` and of the
    /// three-body energy (q = 0), including the CN dependence of the C6 coefficients,
    /// added to `hess` (row-major `3 nat x 3 nat`). With `parallel`, the three-body part runs
    /// on the current rayon pool (the same bits).
    pub fn hessian(&self, cache: &D4Cache, q: &[f64], hess: &mut [f64], parallel: bool) {
        self.hessian_with(cache, q, hess, parallel, ATM_LIMITS);
    }

    /// [`hessian`](Self::hessian) with the three-body work divided by `lim`.
    pub(crate) fn hessian_with(&self, cache: &D4Cache, q: &[f64], hess: &mut [f64], parallel: bool, lim: AtmLimits) {
        let m = self.model();
        let nat = m.natoms();
        let n3 = 3 * nat;
        let pos = cache.positions();
        // X = d²E/dR dCN (n3 x nat), F = d²E/dCN dCN, f = dE/dCN
        let mut x = vec![0.0; n3 * nat];
        let mut fcc = vec![0.0; nat * nat];
        let mut fc = vec![0.0; nat];
        let w = m.weights2(&cache.cn, q);
        for iat in 0..nat {
            for p in cache.pair_offset[iat]..cache.pair_offset[iat + 1] {
                let jat = cache.pair_j[p] as usize;
                let v = sub(pos[iat], pos[jat]);
                let r = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                let (e, e1, e2) = self.pair_radial(iat, jat, r);
                let c6 = m.c6x(iat, jat, &w.gw[iat], &w.gw[jat]);
                let dci = m.c6x(iat, jat, &w.dcn[iat], &w.gw[jat]);
                let dcj = m.c6x(iat, jat, &w.gw[iat], &w.dcn[jat]);
                add_radial_hessian(hess, n3, iat, jat, v, r, -c6 * e1, -c6 * e2);
                for k in 0..3 {
                    let de = e1 * v[k] / r;
                    x[(3 * iat + k) * nat + iat] -= dci * de;
                    x[(3 * iat + k) * nat + jat] -= dcj * de;
                    x[(3 * jat + k) * nat + iat] += dci * de;
                    x[(3 * jat + k) * nat + jat] += dcj * de;
                }
                fcc[iat * nat + iat] -= e * m.c6x(iat, jat, &w.dcn2[iat], &w.gw[jat]);
                fcc[jat * nat + jat] -= e * m.c6x(iat, jat, &w.gw[iat], &w.dcn2[jat]);
                let cross = e * m.c6x(iat, jat, &w.dcn[iat], &w.dcn[jat]);
                fcc[iat * nat + jat] -= cross;
                fcc[jat * nat + iat] -= cross;
                fc[iat] -= e * dci;
                fc[jat] -= e * dcj;
            }
        }
        self.atm_hessian(cache, &mut x, &mut fcc, &mut fc, hess, parallel, lim);
        // chain rule through the D4 coordination numbers
        let jac = m.coordination_number_jacobian(pos, self.cutoff().cn);
        let mut xj = vec![0.0; n3 * n3];
        for r in 0..n3 {
            for c in 0..nat {
                let v = x[r * nat + c];
                if v != 0.0 {
                    for y in 0..n3 {
                        xj[r * n3 + y] += v * jac[c * n3 + y];
                    }
                }
            }
        }
        let mut fj = vec![0.0; nat * n3];
        for c in 0..nat {
            for d in 0..nat {
                let v = fcc[c * nat + d];
                if v != 0.0 {
                    for y in 0..n3 {
                        fj[c * n3 + y] += v * jac[d * n3 + y];
                    }
                }
            }
        }
        for a in 0..n3 {
            for b in 0..n3 {
                let mut jfj = 0.0;
                for c in 0..nat {
                    jfj += jac[c * n3 + a] * fj[c * n3 + b];
                }
                hess[a * n3 + b] += xj[a * n3 + b] + xj[b * n3 + a] + jfj;
            }
        }
        m.add_coordination_number_hessian(pos, self.cutoff().cn, &fc, hess);
    }

    /// ATM three-body second derivatives (q = 0): geometric part added to `hess`, the
    /// CN couplings to `x`, `fcc`, `fc`.
    ///
    /// The triples i > j > k (all three pairs inside the cutoff, nonzero switching product)
    /// are summed in 16 fixed chunks of rows i: each chunk from zero, every element taking its
    /// triples in order, and the chunk sums added to the outputs in chunk order, so that the
    /// bits do not depend on `parallel` or the thread count. A triple's Hessian block is
    /// symmetric to the bit (the same products and sums for (s, t) and (t, s)), and so is a
    /// chunk's sum, so only the atom blocks (a, b >= a) of the Cartesian part and of F are
    /// accumulated and the others read transposed.
    ///
    /// Serial runs hold one chunk accumulator. Parallel runs hold all 16 when they fit in
    /// `lim.small` bytes (one chunk per thread at a time), and otherwise one: a chunk is
    /// then worked through in batches of consecutive triples, each batch's per-triple
    /// quantities ([`AtmTriple`]) evaluated in parallel (overlapping the previous batch's
    /// accumulation) and added to the accumulator by rows, the rows split among the threads.
    /// So the memory does not grow with the thread count beyond 16 small accumulators.
    fn atm_hessian(
        &self,
        cache: &D4Cache,
        x_out: &mut [f64],
        fcc_out: &mut [f64],
        fc_out: &mut [f64],
        hess_out: &mut [f64],
        parallel: bool,
        lim: AtmLimits,
    ) {
        let p = *self.param();
        if p.s9.abs() < f64::EPSILON {
            return;
        }
        let m = self.model();
        let nat = m.natoms();
        let n3 = 3 * nat;
        let pos = cache.positions();
        let w = m.weights2(&cache.cn, &vec![0.0; nat]);
        let (cutoff, width) = (self.cutoff().disp3, self.cutoff().width3);
        let cutoff2 = cutoff * cutoff;
        let alp3 = p.alp / 3.0;
        // half neighbour list (a > b)
        let mut nb = AtmPairs { off: Vec::with_capacity(nat + 1), j: Vec::new(), p: Vec::new() };
        nb.off.push(0);
        for a in 0..nat {
            for b in 0..a {
                let v = sub(pos[b], pos[a]);
                let r2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
                if r2 > cutoff2 || r2 < f64::EPSILON {
                    continue;
                }
                let r = r2.sqrt();
                let (s0, s1, s2) = smooth2(r, cutoff, width);
                // derivatives with respect to r²
                let su = s1 / (2.0 * r);
                let suu = s2 / (4.0 * r2) - s1 / (4.0 * r2 * r);
                let r0 = self.r0_table()[m.id[b] * m.species.len() + m.id[a]];
                nb.j.push(b as u32);
                nb.p.push(AtmPair {
                    r2,
                    sw: [s0, su, suu],
                    r0p: r0.powf(alp3),
                    c6: m.c6x(a, b, &w.gw[a], &w.gw[b]),
                    d: [
                        m.c6x(a, b, &w.dcn[a], &w.gw[b]),
                        m.c6x(a, b, &w.gw[a], &w.dcn[b]),
                        m.c6x(a, b, &w.dcn2[a], &w.gw[b]),
                        m.c6x(a, b, &w.gw[a], &w.dcn2[b]),
                        m.c6x(a, b, &w.dcn[a], &w.dcn[b]),
                    ],
                });
            }
            nb.off.push(nb.j.len());
        }
        // fixed chunks of rows with about equal triple counts (∝ i²), summed in chunk order
        // (serial and parallel runs give identical results)
        const NCHUNK: usize = 16;
        let total: f64 = (0..nat).map(|i| (i * i) as f64).sum::<f64>().max(1.0);
        let mut bounds = Vec::new();
        let (mut start, mut acc) = (0usize, 0.0f64);
        for i in 0..nat {
            acc += (i * i) as f64;
            if bounds.len() + 1 < NCHUNK && acc >= total * (bounds.len() + 1) as f64 / NCHUNK as f64 {
                bounds.push((start, i + 1));
                start = i + 1;
            }
        }
        bounds.push((start, nat));
        let nb = &nb;
        // the pairs (i, j) of rows r0..r1 that have triples, in triple order
        let pairs_of = |(r0, r1): (usize, usize)| {
            (r0..r1)
                .flat_map(move |i| (nb.off[i]..nb.off[i + 1]).map(move |e| (i, e)))
                .filter(|&(_, e)| nb.off[nb.j[e] as usize + 1] > nb.off[nb.j[e] as usize])
        };
        // one chunk on the calling thread, pair by pair
        let chunk = |rows: (usize, usize), acc: &mut AtmAcc, buf: &mut AtmBuf| {
            for slot in pairs_of(rows) {
                let slots = [slot];
                buf.eval(nb, &slots, &p, usize::MAX);
                AtmBatch { nb, slots: &slots, buf, pos, nat }.add(acc, usize::MAX);
            }
        };
        let acc_bytes = std::mem::size_of::<f64>() * (n3 * n3 + n3 * nat + nat * nat + nat);
        if !parallel {
            let (mut acc, mut buf) = (AtmAcc::new(nat), AtmBuf::default());
            for &rows in &bounds {
                chunk(rows, &mut acc, &mut buf);
                acc.add_to(hess_out, x_out, fcc_out, fc_out, false);
                acc.clear(false);
            }
            return;
        }
        if bounds.len() * acc_bytes <= lim.small {
            // small: the chunks in parallel, each with its own accumulator
            use rayon::prelude::*;
            let parts: Vec<AtmAcc> = bounds
                .par_iter()
                .map(|&rows| {
                    let (mut acc, mut buf) = (AtmAcc::new(nat), AtmBuf::default());
                    chunk(rows, &mut acc, &mut buf);
                    acc
                })
                .collect();
            for acc in &parts {
                acc.add_to(hess_out, x_out, fcc_out, fc_out, false);
            }
            return;
        }
        // large: one accumulator; per chunk, batches of consecutive pairs (at most lim.batch
        // possible triples, the neighbours of j, each), then the chunk's sum added to the outputs
        let mut slots = Vec::new();
        let mut steps = Vec::new();
        for &rows in &bounds {
            let (mut first, mut cap) = (slots.len(), 0);
            for (i, e) in pairs_of(rows) {
                let j = nb.j[e] as usize;
                let c = nb.off[j + 1] - nb.off[j];
                if cap + c > lim.batch && slots.len() > first {
                    steps.push(AtmStep::Batch(first..slots.len()));
                    (first, cap) = (slots.len(), 0);
                }
                slots.push((i, e));
                cap += c;
            }
            if slots.len() > first {
                steps.push(AtmStep::Batch(first..slots.len()));
            }
            steps.push(AtmStep::Add);
        }
        let batches: Vec<&[(usize, usize)]> = steps
            .iter()
            .filter_map(|s| match s {
                AtmStep::Batch(r) => Some(&slots[r.clone()]),
                AtmStep::Add => None,
            })
            .collect();
        let mut acc = AtmAcc::new(nat);
        // batch b is evaluated into `cur` while batch b - 1 is added to the accumulator (from
        // `prev`)
        let (mut cur, mut prev) = (AtmBuf::default(), AtmBuf::default());
        let mut b = 0;
        if let Some(s) = batches.first() {
            cur.eval(nb, s, &p, lim.par_min);
        }
        for step in &steps {
            match step {
                AtmStep::Add => {
                    acc.add_to(hess_out, x_out, fcc_out, fc_out, true);
                    acc.clear(true);
                }
                AtmStep::Batch(_) => {
                    std::mem::swap(&mut cur, &mut prev);
                    let batch = AtmBatch { nb, slots: batches[b], buf: &prev, pos, nat };
                    match batches.get(b + 1) {
                        Some(next) => {
                            rayon::join(|| batch.add(&mut acc, lim.par_min), || cur.eval(nb, next, &p, lim.par_min));
                        }
                        None => batch.add(&mut acc, lim.par_min),
                    }
                    b += 1;
                }
            }
        }
    }
}

/// A step of the three-body Hessian: a batch of pairs (i, j) (a range of the pair list), or
/// adding the chunk accumulator to the outputs.
enum AtmStep {
    Batch(std::ops::Range<usize>),
    Add,
}

/// The triples (i, j, k) of the pair (i, j) (`e`: its entry in the list of i), k ascending,
/// written to `out` and their k to `ks`; returns their number.
fn atm_pair_triples(
    nb: &AtmPairs,
    i: usize,
    e: usize,
    p: &RationalDamping,
    out: &mut [AtmTriple],
    ks: &mut [u32],
) -> usize {
    let j = nb.j[e] as usize;
    let pij = nb.p[e];
    // k runs over the neighbours of j that are also neighbours of i (both lists ascending)
    let (mut e_ik, end_i) = (nb.off[i], nb.off[i + 1]);
    let mut n = 0;
    for e_jk in nb.off[j]..nb.off[j + 1] {
        let k = nb.j[e_jk];
        while e_ik < end_i && nb.j[e_ik] < k {
            e_ik += 1;
        }
        if e_ik == end_i {
            break;
        }
        if nb.j[e_ik] != k {
            continue;
        }
        let (pik, pjk) = (nb.p[e_ik], nb.p[e_jk]);
        if pij.sw[0] * pik.sw[0] * pjk.sw[0] <= 0.0 {
            continue;
        }
        // Γ(a, b, c) with a = r_ij², b = r_ik², c = r_jk²
        let a = D2::var(pij.r2, 0);
        let b = D2::var(pik.r2, 1);
        let c = D2::var(pjk.r2, 2);
        let prod = a.mul(b).mul(c);
        let ang = a
            .add(c)
            .sub(b)
            .mul(a.sub(c).add(b))
            .mul(c.add(b).sub(a))
            .mul(prod.powf(-2.5))
            .scale(0.375)
            .add(prod.powf(-1.5));
        let tt = pij.r0p * pik.r0p * pjk.r0p;
        let fd = D2::cst(1.0).add(prod.powf(-p.alp / 6.0).scale(6.0 * tt)).recip();
        let sw = |pp: &AtmPair, u: &D2| u.map(pp.sw[0], pp.sw[1], pp.sw[2]);
        let gam = ang.mul(fd).mul(sw(&pij, &a)).mul(sw(&pik, &b)).mul(sw(&pjk, &c));
        // c9 = s9 sqrt(C6ij C6ik C6jk) and its CN derivatives
        let c9 = p.s9 * (pij.c6 * pik.c6 * pjk.c6).abs().sqrt();
        // pairs as (pair data, local index of the higher atom, of the lower atom)
        let pl: [(&AtmPair, usize, usize); 3] = [(&pij, 0, 1), (&pik, 0, 2), (&pjk, 1, 2)];
        let mut lm = [0.0; 3];
        let mut dl = [[0.0; 3]; 3];
        for (pp, hi, lo) in pl {
            let (dh, dlo) = (pp.d[0] / pp.c6, pp.d[1] / pp.c6);
            lm[hi] += 0.5 * dh;
            lm[lo] += 0.5 * dlo;
            dl[hi][hi] += 0.5 * (pp.d[2] / pp.c6 - dh * dh);
            dl[lo][lo] += 0.5 * (pp.d[3] / pp.c6 - dlo * dlo);
            let cr = 0.5 * (pp.d[4] / pp.c6 - dh * dlo);
            dl[hi][lo] += cr;
            dl[lo][hi] += cr;
        }
        out[n] = AtmTriple {
            guv: std::array::from_fn(|qq| gam.h[qq] * c9),
            gv: std::array::from_fn(|u| 2.0 * gam.g[u] * c9),
            g: gam.g,
            vc9: gam.v * c9,
            c9,
            lm,
            dl: [dl[0][0], dl[1][0], dl[1][1], dl[2][0], dl[2][1], dl[2][2]],
        };
        ks[n] = k;
        n += 1;
    }
    n
}

/// The evaluated triples of a batch: those of pair q in `t[off[q]..off[q] + counts[q]]`, their
/// third atoms in `k`.
#[derive(Default)]
struct AtmBuf {
    t: Vec<AtmTriple>,
    k: Vec<u32>,
    off: Vec<usize>,
    counts: Vec<usize>,
}

impl AtmBuf {
    /// Evaluate the triples of the pairs `slots` ((i, entry of (i, j))), pairs in parallel when
    /// they have at least `par_min` possible triples.
    fn eval(&mut self, nb: &AtmPairs, slots: &[(usize, usize)], p: &RationalDamping, par_min: usize) {
        use rayon::prelude::*;
        // the region of every pair: room for each neighbour k of j
        self.off.clear();
        self.off.push(0);
        for &(_, e) in slots {
            let j = nb.j[e] as usize;
            self.off.push(self.off[self.off.len() - 1] + nb.off[j + 1] - nb.off[j]);
        }
        let size = self.off[slots.len()];
        if self.t.len() < size {
            self.t.resize(size, AtmTriple::default());
            self.k.resize(size, 0);
        }
        self.counts.clear();
        if size < par_min {
            for (q, &(i, e)) in slots.iter().enumerate() {
                let (a, b) = (self.off[q], self.off[q + 1]);
                self.counts.push(atm_pair_triples(nb, i, e, p, &mut self.t[a..b], &mut self.k[a..b]));
            }
            return;
        }
        let mut regions = Vec::with_capacity(slots.len());
        let (mut rt, mut rk) = (&mut self.t[..size], &mut self.k[..size]);
        for q in 0..slots.len() {
            let n = self.off[q + 1] - self.off[q];
            let (t, tr) = std::mem::take(&mut rt).split_at_mut(n);
            let (k, kr) = std::mem::take(&mut rk).split_at_mut(n);
            (rt, rk) = (tr, kr);
            regions.push((t, k));
        }
        slots
            .par_iter()
            .zip(regions)
            .map(|(&(i, e), (t, k))| atm_pair_triples(nb, i, e, p, t, k))
            .collect_into_vec(&mut self.counts);
    }
}

/// An evaluated batch: the pairs (i, entry of (i, j)) in triple order and their triples.
struct AtmBatch<'a> {
    nb: &'a AtmPairs,
    slots: &'a [(usize, usize)],
    buf: &'a AtmBuf,
    pos: &'a [[f64; 3]],
    nat: usize,
}

/// Accumulator rows `s0..s1` of [`AtmAcc`] (`hess`, `x`), with F and f of the atoms from `a0`
/// (those whose first row is among them).
struct AtmRows<'a> {
    s0: usize,
    s1: usize,
    a0: usize,
    nat: usize,
    hess: &'a mut [f64],
    x: &'a mut [f64],
    fcc: &'a mut [f64],
    fc: &'a mut [f64],
}

impl AtmBatch<'_> {
    /// Add the batch to `acc`, in parallel when it has at least `par_min` possible triples:
    /// the rows of the accumulator split into ranges of about equal work, each range taking
    /// the triples in order.
    fn add(&self, acc: &mut AtmAcc, par_min: usize) {
        use rayon::prelude::*;
        let (nat, nb, buf) = (self.nat, self.nb, self.buf);
        let n3 = 3 * nat;
        if buf.off[self.slots.len()] < par_min {
            return self.add_rows(0, n3, &mut acc.hess, &mut acc.x, &mut acc.fcc, &mut acc.fc);
        }
        // an atom's rows cost about its number of triples times the columns it adds (3, 6, 9
        // as atom i, j, k of the triple)
        let mut work = vec![0usize; nat];
        for (q, &(i, e)) in self.slots.iter().enumerate() {
            let n = buf.counts[q];
            work[i] += n;
            work[nb.j[e] as usize] += 2 * n;
            for &k in &buf.k[buf.off[q]..buf.off[q] + n] {
                work[k as usize] += 3;
            }
        }
        let smax = 3 * (self.slots[self.slots.len() - 1].0 + 1);
        let target = (3 * work.iter().sum::<usize>() / (4 * rayon::current_num_threads())).max(1);
        let mut ranges = Vec::new();
        let (mut s0, mut w) = (0, 0);
        for s in 0..smax {
            w += work[s / 3];
            if w >= target || s + 1 == smax {
                ranges.push((s0, s + 1));
                (s0, w) = (s + 1, 0);
            }
        }
        // the accumulators split by those rows (F and f by the atoms whose first row is in them)
        let mut parts = Vec::with_capacity(ranges.len());
        let (mut h, mut x, mut f2, mut f1) = (&mut acc.hess[..], &mut acc.x[..], &mut acc.fcc[..], &mut acc.fc[..]);
        for &(s0, s1) in &ranges {
            let na = s1.div_ceil(3) - s0.div_ceil(3);
            let (hh, hr) = std::mem::take(&mut h).split_at_mut((s1 - s0) * n3);
            let (xh, xr) = std::mem::take(&mut x).split_at_mut((s1 - s0) * nat);
            let (fh, fr) = std::mem::take(&mut f2).split_at_mut(na * nat);
            let (ch, cr) = std::mem::take(&mut f1).split_at_mut(na);
            (h, x, f2, f1) = (hr, xr, fr, cr);
            parts.push((s0, s1, hh, xh, fh, ch));
        }
        parts.into_par_iter().for_each(|(s0, s1, hh, xh, fh, ch)| self.add_rows(s0, s1, hh, xh, fh, ch));
    }

    /// Add the batch's triples to the accumulator rows `s0..s1` (`hess`, `x`: those rows; `fcc`,
    /// `fc`: the atoms whose first row is among them), every element taking its triples in
    /// order.
    fn add_rows(&self, s0: usize, s1: usize, hess: &mut [f64], x: &mut [f64], fcc: &mut [f64], fc: &mut [f64]) {
        // atoms with rows here
        let (xa, xb) = (s0 / 3, (s1 - 1) / 3);
        let here = |a: usize| xa <= a && a <= xb;
        let mut rows = AtmRows { s0, s1, a0: s0.div_ceil(3), nat: self.nat, hess, x, fcc, fc };
        let buf = self.buf;
        for (q, &(i, e)) in self.slots.iter().enumerate() {
            let j = self.nb.j[e] as usize;
            let (o, ks) = (buf.off[q], &buf.k[buf.off[q]..buf.off[q] + buf.counts[q]]);
            // the triples with an atom here (k ascending)
            let (lo, hi) = if here(i) || here(j) {
                (0, ks.len())
            } else {
                (ks.partition_point(|&k| (k as usize) < xa), ks.partition_point(|&k| (k as usize) <= xb))
            };
            for m in lo..hi {
                let k = ks[m] as usize;
                let t = AtmTerms::new(&buf.t[o + m], [i, j, k], self.pos, here(j) || here(k));
                if here(i) {
                    t.add_atom::<3>(&mut rows);
                }
                if here(j) {
                    t.add_atom::<6>(&mut rows);
                }
                if here(k) {
                    t.add_atom::<9>(&mut rows);
                }
            }
        }
    }
}

/// One triple's factors for its rows of the accumulators.
struct AtmTerms<'a> {
    t: &'a AtmTriple,
    at: [usize; 3],
    /// Cartesian gradients of r_ij², r_ik², r_jk² over the atoms (i, j, k)
    gu: [[f64; 9]; 3],
}

impl<'a> AtmTerms<'a> {
    /// The factors of the triple `t` of the atoms `at`; without `full` only those of the rows
    /// of its first atom (`gu` in the columns of that atom, where r_jk² has none).
    #[inline]
    fn new(t: &'a AtmTriple, at: [usize; 3], pos: &[[f64; 3]], full: bool) -> Self {
        let [i, j, k] = at;
        let dij = sub(pos[i], pos[j]);
        let dik = sub(pos[i], pos[k]);
        let mut gu = [[0.0; 9]; 3];
        for d in 0..3 {
            gu[0][d] = 2.0 * dij[d];
            gu[1][d] = 2.0 * dik[d];
        }
        if full {
            let djk = sub(pos[j], pos[k]);
            for d in 0..3 {
                gu[0][3 + d] = -2.0 * dij[d];
                gu[1][6 + d] = -2.0 * dik[d];
                gu[2][3 + d] = 2.0 * djk[d];
                gu[2][6 + d] = -2.0 * djk[d];
            }
        }
        AtmTerms { t, at, gu }
    }

    /// Add the rows of the triple's atom with `NC` / 3 - 1 as its local index (0 = i, 1 = j,
    /// 2 = k) that lie in `r`: its `NC` columns of the triple's 9 x 9 Hessian block (those of
    /// the atoms not below it), its row of X and, with its first row, its F and f.
    #[inline]
    fn add_atom<const NC: usize>(&self, r: &mut AtmRows) {
        let la = NC / 3 - 1;
        let ma = self.at[la];
        let (nat, gu, t) = (r.nat, &self.gu, self.t);
        let n3 = 3 * nat;
        for c in 0..3 {
            let row = 3 * ma + c;
            if row < r.s0 || row >= r.s1 {
                continue;
            }
            // row s of the 9 x 9 block, as the full block is built: Σ_uv Γ_uv ∇u ∇v^T, then
            // Γ_u ∇²u with ∇²(|R_p - R_q|²) = 2 (I, -I; -I, I) on atoms (p, q)
            let s = 3 * la + c;
            let mut h9 = [0.0; 9];
            for (qq, &(u, v)) in PK.iter().enumerate() {
                let (au, av) = (gu[u][s], gu[v][s]);
                if au == 0.0 && av == 0.0 {
                    continue;
                }
                let guv = t.guv[qq];
                if u == v {
                    for col in 0..NC {
                        h9[col] += guv * (au * gu[v][col]);
                    }
                } else {
                    for col in 0..NC {
                        h9[col] += guv * (au * gu[v][col] + av * gu[u][col]);
                    }
                }
            }
            for (u, &(pa, pb)) in ATM_LAP.iter().enumerate() {
                if la == pa {
                    h9[3 * pa + c] += t.gv[u];
                    h9[3 * pb + c] -= t.gv[u];
                } else if la == pb {
                    h9[3 * pb + c] += t.gv[u];
                    h9[3 * pa + c] -= t.gv[u];
                }
            }
            let rr = row - r.s0;
            let hrow = &mut r.hess[rr * n3..(rr + 1) * n3];
            for lb in 0..=la {
                for d in 0..3 {
                    hrow[3 * self.at[lb] + d] += h9[3 * lb + d];
                }
            }
            // dΓ/dR · dc9/dCN
            let gs = (0..3).map(|u| t.g[u] * gu[u][s]).sum::<f64>();
            let xrow = &mut r.x[rr * nat..(rr + 1) * nat];
            for (loc, &mm) in self.at.iter().enumerate() {
                xrow[mm] += gs * t.c9 * t.lm[loc];
            }
            if c == 0 {
                let ra = ma - r.a0;
                r.fc[ra] += t.vc9 * t.lm[la];
                for lb in 0..=la {
                    r.fcc[ra * nat + self.at[lb]] += t.vc9 * (t.lm[la] * t.lm[lb] + t.dl[la * (la + 1) / 2 + lb]);
                }
            }
        }
    }
}

impl AtmAcc {
    fn new(nat: usize) -> Self {
        let n3 = 3 * nat;
        AtmAcc { hess: vec![0.0; n3 * n3], x: vec![0.0; n3 * nat], fcc: vec![0.0; nat * nat], fc: vec![0.0; nat] }
    }

    /// Add the chunk to the outputs (the Cartesian part and F below the atom diagonal from
    /// their transposes: element by element the same values as adding the full matrices); on
    /// the current rayon pool when `parallel`.
    fn add_to(&self, hess: &mut [f64], x: &mut [f64], fcc: &mut [f64], fc: &mut [f64], parallel: bool) {
        use rayon::prelude::*;
        let nat = self.fc.len();
        let n3 = 3 * nat;
        let parallel = parallel && nat >= 64;
        add_upper_blocks(&self.hess, &mut hess[..n3 * n3], n3, 3, parallel);
        add_upper_blocks(&self.fcc, &mut fcc[..nat * nat], nat, 1, parallel);
        let add = |(a, b): (&mut [f64], &[f64])| a.iter_mut().zip(b).for_each(|(a, b)| *a += b);
        let x = &mut x[..n3 * nat];
        if parallel {
            x.par_chunks_mut(1 << 14).zip(self.x.par_chunks(1 << 14)).for_each(add);
        } else {
            add((x, &self.x));
        }
        add((&mut fc[..nat], &self.fc));
    }

    /// Zero what a chunk writes: the upper blocks, all of X and f.
    fn clear(&mut self, parallel: bool) {
        use rayon::prelude::*;
        let nat = self.fc.len();
        let n3 = 3 * nat;
        let parallel = parallel && nat >= 64;
        zero_upper_blocks(&mut self.hess, n3, 3, parallel);
        zero_upper_blocks(&mut self.fcc, nat, 1, parallel);
        if parallel {
            self.x.par_chunks_mut(1 << 14).for_each(|c| c.fill(0.0));
        } else {
            self.x.fill(0.0);
        }
        self.fc.fill(0.0);
    }
}

/// Rows of the tiles of [`add_upper_blocks`].
const TILE: usize = 32;

/// `out[r][c] += src[r][c]` if block(c) >= block(r) (blocks of `bs` rows and columns), else
/// `src[c][r]`, for n x n row-major matrices: the part below the block diagonal read from the
/// transpose in tiles (so that the reads stay in cache); rows in parallel when `parallel`.
fn add_upper_blocks(src: &[f64], out: &mut [f64], n: usize, bs: usize, parallel: bool) {
    use rayon::prelude::*;
    if n == 0 {
        return;
    }
    let tile_rows = |(t, rows): (usize, &mut [f64])| {
        let r0 = t * TILE;
        let nr = rows.len() / n;
        // below the block diagonal: columns before the row's block, tile by tile
        let cmax = bs * ((r0 + nr - 1) / bs);
        for c0 in (0..cmax).step_by(TILE) {
            for (dr, row) in rows.chunks_mut(n).enumerate() {
                let r = r0 + dr;
                for c in c0..(c0 + TILE).min(bs * (r / bs)) {
                    row[c] += src[c * n + r];
                }
            }
        }
        for (dr, row) in rows.chunks_mut(n).enumerate() {
            let r = r0 + dr;
            let cb = bs * (r / bs);
            for (o, s) in row[cb..].iter_mut().zip(&src[r * n + cb..(r + 1) * n]) {
                *o += s;
            }
        }
    };
    if parallel {
        out.par_chunks_mut(TILE * n).enumerate().for_each(tile_rows);
    } else {
        out.chunks_mut(TILE * n).enumerate().for_each(tile_rows);
    }
}

/// Zero the blocks (r, c >= r) of an n x n row-major matrix (blocks of `bs`), rows in parallel
/// when `parallel`.
fn zero_upper_blocks(m: &mut [f64], n: usize, bs: usize, parallel: bool) {
    use rayon::prelude::*;
    let zero = |(r, row): (usize, &mut [f64])| row[bs * (r / bs)..].fill(0.0);
    if parallel {
        m.par_chunks_mut(n.max(1)).enumerate().with_min_len(16).for_each(zero);
    } else {
        m.chunks_mut(n.max(1)).enumerate().for_each(zero);
    }
}
