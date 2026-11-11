// SPDX-License-Identifier: LGPL-3.0-or-later
//! Second derivatives of the self-consistent D4 dispersion: the charge kernel
//! `d²E/dq dq`, the geometry derivative of the charge potential at fixed charges, and
//! the nuclear Hessian at fixed charges (two-body term with the coordination-number and
//! charge dependence of the C6 coefficients, three-body ATM term at zero charges).
//!
//! Conventions as in the rest of the crate: positions in bohr, row-major dense
//! matrices, results accumulated with `+=`.

use crate::data::MAX_REF;
use crate::disp::{smooth_cutoff, D4Cache, D4Dispersion};
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
#[derive(Clone, Copy, Debug)]
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

/// Per-pair data of the three-body term (pair (a, b) with b < a).
#[derive(Clone, Copy, Default)]
struct AtmPair {
    ok: bool,
    r2: f64,
    /// switching function of r and its first and second derivatives in r²
    sw: [f64; 3],
    /// (r0)^(alp/3)
    r0p: f64,
    /// C6 at q = 0 and its CN derivatives: d/dCN_a, d/dCN_b, d²/dCN_a², d²/dCN_b², d²/dCN_a dCN_b
    c6: f64,
    d: [f64; 5],
}

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
        self.atm_hessian(cache, &mut x, &mut fcc, &mut fc, hess, parallel);
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
    fn atm_hessian(
        &self,
        cache: &D4Cache,
        x_out: &mut [f64],
        fcc_out: &mut [f64],
        fc_out: &mut [f64],
        hess_out: &mut [f64],
        parallel: bool,
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
        // pair table (a > b)
        let mut pairs = vec![AtmPair::default(); nat * nat];
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
                pairs[a * nat + b] = AtmPair {
                    ok: true,
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
                };
            }
        }
        let triples = |rows: std::ops::Range<usize>| -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
            let mut hess = vec![0.0; n3 * n3];
            let mut x = vec![0.0; n3 * nat];
            let mut fcc = vec![0.0; nat * nat];
            let mut fc = vec![0.0; nat];
            for i in rows {
                for j in 0..i {
                    let pij = pairs[i * nat + j];
                    if !pij.ok {
                        continue;
                    }
                    for k in 0..j {
                        let (pik, pjk) = (pairs[i * nat + k], pairs[j * nat + k]);
                        if !pik.ok || !pjk.ok {
                            continue;
                        }
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
                        let at = [i, j, k];
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
                        // Cartesian gradients of a, b, c over the atoms (i, j, k)
                        let dij = sub(pos[i], pos[j]);
                        let dik = sub(pos[i], pos[k]);
                        let djk = sub(pos[j], pos[k]);
                        let mut gu = [[0.0; 9]; 3];
                        for t in 0..3 {
                            gu[0][t] = 2.0 * dij[t];
                            gu[0][3 + t] = -2.0 * dij[t];
                            gu[1][t] = 2.0 * dik[t];
                            gu[1][6 + t] = -2.0 * dik[t];
                            gu[2][3 + t] = 2.0 * djk[t];
                            gu[2][6 + t] = -2.0 * djk[t];
                        }
                        let mut h9 = [[0.0; 9]; 9];
                        for (qq, &(u, v)) in PK.iter().enumerate() {
                            let guv = gam.h[qq] * c9;
                            for s in 0..9 {
                                let (au, av) = (gu[u][s], gu[v][s]);
                                if au == 0.0 && av == 0.0 {
                                    continue;
                                }
                                for t in 0..9 {
                                    let mut val = au * gu[v][t];
                                    if u != v {
                                        val += av * gu[u][t];
                                    }
                                    h9[s][t] += guv * val;
                                }
                            }
                        }
                        // Γ_u ∇²u: ∇²(|R_p - R_q|²) = 2 (I, -I; -I, I) on atoms (p, q)
                        let lap: [(usize, usize); 3] = [(0, 1), (0, 2), (1, 2)];
                        for (u, &(pa, pb)) in lap.iter().enumerate() {
                            let gv = 2.0 * gam.g[u] * c9;
                            for t in 0..3 {
                                h9[3 * pa + t][3 * pa + t] += gv;
                                h9[3 * pb + t][3 * pb + t] += gv;
                                h9[3 * pa + t][3 * pb + t] -= gv;
                                h9[3 * pb + t][3 * pa + t] -= gv;
                            }
                        }
                        for s in 0..9 {
                            let xs = 3 * at[s / 3] + s % 3;
                            for t in 0..9 {
                                hess[xs * n3 + 3 * at[t / 3] + t % 3] += h9[s][t];
                            }
                            // dΓ/dR · dc9/dCN
                            let gs = (0..3).map(|u| gam.g[u] * gu[u][s]).sum::<f64>();
                            for (loc, &mm) in at.iter().enumerate() {
                                x[xs * nat + mm] += gs * c9 * lm[loc];
                            }
                        }
                        for (la, &ma) in at.iter().enumerate() {
                            fc[ma] += gam.v * c9 * lm[la];
                            for (lb, &mb) in at.iter().enumerate() {
                                fcc[ma * nat + mb] += gam.v * c9 * (lm[la] * lm[lb] + dl[la][lb]);
                            }
                        }
                    }
                }
            }
            (hess, x, fcc, fc)
        };
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
        let mut add = |(h, xx, f2, f1): (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>)| {
            for (a, b) in hess_out.iter_mut().zip(&h) {
                *a += b;
            }
            for (a, b) in x_out.iter_mut().zip(&xx) {
                *a += b;
            }
            for (a, b) in fcc_out.iter_mut().zip(&f2) {
                *a += b;
            }
            for (a, b) in fc_out.iter_mut().zip(&f1) {
                *a += b;
            }
        };
        if parallel {
            use rayon::prelude::*;
            let parts: Vec<_> = bounds.par_iter().map(|&(a, b)| triples(a..b)).collect();
            parts.into_iter().for_each(&mut add);
        } else {
            bounds.iter().for_each(|&(a, b)| add(triples(a..b)));
        }
    }
}
