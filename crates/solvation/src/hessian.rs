// SPDX-License-Identifier: LGPL-3.0-or-later
//! Second derivatives of the solvation free energy at fixed charges: the geometry
//! derivative of the charge potential `(B q)_i` and the nuclear Hessian of the whole model
//! (generalized Born polar term with the Born radii's geometry dependence, ALPB shape term,
//! hydrogen-bond and surface-tension terms through the smoothed SASA).
//!
//! Smooth one-, three- and six-variable functions (Born-radius pair integrals and
//! transform, GB kernels, the ALPB shape descriptor) are differentiated with small
//! second-order forward-mode dual numbers; the SASA second derivatives are assembled per
//! grid point from the products of the switching functions. Only the corrected Born-radius
//! integrator (fresh `r1`) is differentiated: xtb's stale-`r1` quirk makes its energy and
//! gradient inconsistent, so [`Solvation::second_derivatives`] returns `false` there.

use crate::kernel::Kernel;
use crate::params::{OBC_ALPHA, OBC_BETA, OBC_GAMMA, TOLSESP, ZETA_P16};
use crate::{SolvCache, Solvation};
use axtb_base::math::{add_radial_hessian, sub3};

/// Value with first and second derivative in one variable.
#[derive(Clone, Copy, Debug)]
struct D1 {
    v: f64,
    d: f64,
    dd: f64,
}

/// A pair (i, j) of the Born-radius integral: its distance vector and distance, and the
/// contributions to ψ_i and ψ_j (`None` outside the integral's range).
type PsiPair = (usize, usize, [f64; 3], f64, Option<D1>, Option<D1>);

impl D1 {
    fn var(v: f64) -> Self {
        D1 { v, d: 1.0, dd: 0.0 }
    }
    fn cst(v: f64) -> Self {
        D1 { v, d: 0.0, dd: 0.0 }
    }
    fn map(self, f: f64, f1: f64, f2: f64) -> Self {
        D1 { v: f, d: f1 * self.d, dd: f2 * self.d * self.d + f1 * self.dd }
    }
    fn recip(self) -> Self {
        let x = self.v;
        self.map(1.0 / x, -1.0 / (x * x), 2.0 / (x * x * x))
    }
    fn ln(self) -> Self {
        let x = self.v;
        self.map(x.ln(), 1.0 / x, -1.0 / (x * x))
    }
    fn tanh(self) -> Self {
        let t = self.v.tanh();
        self.map(t, 1.0 - t * t, -2.0 * t * (1.0 - t * t))
    }
}

impl std::ops::Add for D1 {
    type Output = D1;
    fn add(self, o: D1) -> D1 {
        D1 { v: self.v + o.v, d: self.d + o.d, dd: self.dd + o.dd }
    }
}
impl std::ops::Sub for D1 {
    type Output = D1;
    fn sub(self, o: D1) -> D1 {
        D1 { v: self.v - o.v, d: self.d - o.d, dd: self.dd - o.dd }
    }
}
impl std::ops::Mul for D1 {
    type Output = D1;
    fn mul(self, o: D1) -> D1 {
        D1 { v: self.v * o.v, d: self.d * o.v + self.v * o.d, dd: self.dd * o.v + 2.0 * self.d * o.d + self.v * o.dd }
    }
}
impl std::ops::Mul<f64> for D1 {
    type Output = D1;
    fn mul(self, s: f64) -> D1 {
        D1 { v: self.v * s, d: self.d * s, dd: self.dd * s }
    }
}

/// Value, gradient and Hessian in `N` variables.
#[derive(Clone, Copy, Debug)]
struct Dn<const N: usize> {
    v: f64,
    g: [f64; N],
    h: [[f64; N]; N],
}

impl<const N: usize> Dn<N> {
    fn var(v: f64, i: usize) -> Self {
        let mut g = [0.0; N];
        g[i] = 1.0;
        Dn { v, g, h: [[0.0; N]; N] }
    }
    fn add(self, o: Self) -> Self {
        Dn {
            v: self.v + o.v,
            g: std::array::from_fn(|k| self.g[k] + o.g[k]),
            h: std::array::from_fn(|k| std::array::from_fn(|l| self.h[k][l] + o.h[k][l])),
        }
    }
    fn sub(self, o: Self) -> Self {
        self.add(o.scale(-1.0))
    }
    fn scale(self, s: f64) -> Self {
        Dn { v: self.v * s, g: self.g.map(|x| x * s), h: self.h.map(|r| r.map(|x| x * s)) }
    }
    fn mul(self, o: Self) -> Self {
        Dn {
            v: self.v * o.v,
            g: std::array::from_fn(|k| self.g[k] * o.v + self.v * o.g[k]),
            h: std::array::from_fn(|k| {
                std::array::from_fn(|l| {
                    self.h[k][l] * o.v + self.g[k] * o.g[l] + self.g[l] * o.g[k] + self.v * o.h[k][l]
                })
            }),
        }
    }
    fn map(self, f: f64, f1: f64, f2: f64) -> Self {
        Dn {
            v: f,
            g: self.g.map(|x| f1 * x),
            h: std::array::from_fn(|k| std::array::from_fn(|l| f2 * self.g[k] * self.g[l] + f1 * self.h[k][l])),
        }
    }
    fn recip(self) -> Self {
        let x = self.v;
        self.map(1.0 / x, -1.0 / (x * x), 2.0 / (x * x * x))
    }
    fn sqrt(self) -> Self {
        let s = self.v.sqrt();
        self.map(s, 0.5 / s, -0.25 / (s * self.v))
    }
    fn exp(self) -> Self {
        let e = self.v.exp();
        self.map(e, e, e)
    }
    fn powf(self, p: f64) -> Self {
        let x = self.v;
        self.map(x.powf(p), p * x.powf(p - 1.0), p * (p - 1.0) * x.powf(p - 2.0))
    }
}

/// Born-radius pair integral of the non-overlapping case (fresh `r1 = 1/r`).
fn nonoverlap(r: D1, rho: f64) -> D1 {
    let ap = r + D1::cst(rho);
    let am = r - D1::cst(rho);
    let ab = ap * am;
    let r1 = r.recip();
    ab.recip() * rho + (am * ap.recip()).ln() * r1 * 0.5
}

/// Born-radius pair integral of the overlapping case.
fn overlap(r: D1, rho: f64, rvdw: f64) -> D1 {
    let r1 = r.recip();
    let r12 = r1 * 0.5;
    let ap = r + D1::cst(rho);
    let am = r - D1::cst(rho);
    let rh1 = 1.0 / rvdw;
    let rhr1 = ap.recip();
    let aprh1 = ap * rh1;
    let lnab = aprh1.ln();
    D1::cst(rh1) - rhr1 + r12 * ((am * (rhr1 - aprh1 * rh1)) * 0.5 - lnab)
}

/// Contributions of the pair (i, j) to psi_i and psi_j with their radial derivatives
/// (the case logic of `born::pair` with a fresh `r1`).
fn psi_pair(r: f64, rhoi: f64, rhoj: f64, rvdwi: f64, rvdwj: f64) -> (Option<D1>, Option<D1>) {
    let x = D1::var(r);
    let ovij = !(r >= rvdwi + rhoj);
    let ovji = !(r >= rhoi + rvdwj);
    let ci = if !ovij {
        Some(nonoverlap(x, rhoj))
    } else if r + rhoj > rvdwi {
        Some(overlap(x, rhoj, rvdwi))
    } else {
        None
    };
    let cj = if !ovji {
        Some(nonoverlap(x, rhoi))
    } else if r + rhoi > rvdwj {
        Some(overlap(x, rhoi, rvdwj))
    } else {
        None
    };
    (ci, cj)
}

impl Solvation {
    /// GB pair kernel `keps / f_GB(r, a_i, a_j)` with derivatives in (r, a_i, a_j).
    fn gb_pair(&self, r: f64, ai: f64, aj: f64) -> Dn<3> {
        let (x, a, b) = (Dn::<3>::var(r, 0), Dn::<3>::var(ai, 1), Dn::<3>::var(aj, 2));
        let f = match self.kernel {
            Kernel::P16 => {
                let ab = a.mul(b).sqrt();
                let arg = ab.mul(ab.add(x.scale(ZETA_P16 / 16.0)).recip());
                let mut a16 = arg;
                for _ in 0..4 {
                    a16 = a16.mul(a16);
                }
                x.add(ab.mul(a16))
            }
            Kernel::Still => {
                let aa = a.mul(b);
                let r2 = x.mul(x);
                let dd = r2.scale(0.25).mul(aa.recip());
                r2.add(aa.mul(dd.scale(-1.0).exp())).sqrt()
            }
        };
        f.recip().scale(self.keps)
    }

    /// Born radius as a function of psi, with its first and second derivative.
    fn born_transform(&self, i: usize, psi: f64) -> D1 {
        let s1 = 1.0 / self.svdw[i];
        let v1 = 1.0 / self.vdwr[i];
        let s2 = 0.5 * self.svdw[i];
        let br = D1::var(psi) * s2;
        let arg = br * (D1::cst(OBC_ALPHA) + br * (br * OBC_GAMMA - D1::cst(OBC_BETA)));
        (D1::cst(s1) - arg.tanh() * v1).recip() * self.born_scale
    }

    /// `1 / A_det` of the ALPB shape term with its gradient and Hessian (dense, 3 nat).
    fn inverse_adet(&self, xyz: &[[f64; 3]]) -> (f64, Vec<f64>, Vec<f64>) {
        const TOF: f64 = 2.0 / 5.0;
        let nat = xyz.len();
        let n3 = 3 * nat;
        let w: Vec<f64> = self.vdwr.iter().map(|r| r * r * r).collect();
        let wt: f64 = w.iter().sum();
        let mut c = [0.0; 3];
        for (x, &wi) in xyz.iter().zip(&w) {
            for k in 0..3 {
                c[k] += x[k] * wi;
            }
        }
        for v in c.iter_mut() {
            *v /= wt;
        }
        // inertia tensor, unique components (xx, yy, zz, xy, xz, yz)
        const IDX: [(usize, usize); 6] = [(0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2)];
        let mut inert = [0.0; 6];
        // dI_u / dx_{i,k} (6 x n3)
        let mut gi = vec![[0.0; 6]; n3];
        for i in 0..nat {
            let r2v = self.vdwr[i] * self.vdwr[i];
            let v = sub3(xyz[i], c);
            let vv = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            for (u, &(a, b)) in IDX.iter().enumerate() {
                let d = if a == b { vv + TOF * r2v } else { 0.0 };
                inert[u] += w[i] * (d - v[a] * v[b]);
                for k in 0..3 {
                    let dab = if a == b { 2.0 * v[k] } else { 0.0 };
                    let t = dab - if a == k { v[b] } else { 0.0 } - if b == k { v[a] } else { 0.0 };
                    gi[3 * i + k][u] += w[i] * t;
                }
            }
        }
        let iv: [Dn<6>; 6] = std::array::from_fn(|u| Dn::<6>::var(inert[u], u));
        let (xx, yy, zz, xy, xz, yz) = (iv[0], iv[1], iv[2], iv[3], iv[4], iv[5]);
        let det = xx
            .mul(yy.mul(zz).sub(yz.mul(yz)))
            .sub(xy.mul(xy.mul(zz).sub(yz.mul(xz))))
            .add(xz.mul(xy.mul(yz).sub(yy.mul(xz))));
        let f = det.powf(-1.0 / 6.0).scale((TOF * wt).sqrt());
        let mut grad = vec![0.0; n3];
        for x in 0..n3 {
            grad[x] = (0..6).map(|u| f.g[u] * gi[x][u]).sum();
        }
        let mut hess = vec![0.0; n3 * n3];
        for x in 0..n3 {
            for y in 0..n3 {
                let mut s = 0.0;
                for u in 0..6 {
                    for v in 0..6 {
                        s += f.h[u][v] * gi[x][u] * gi[y][v];
                    }
                }
                hess[x * n3 + y] = s;
            }
        }
        // second derivatives of the inertia tensor:
        // d²I_ab / dx_ik dx_jl = (δ_ij w_i - w_i w_j / W)(2 δ_kl δ_ab - δ_ak δ_bl - δ_al δ_bk)
        let mut t = [[0.0; 3]; 3];
        for k in 0..3 {
            for l in 0..3 {
                let mut s = 0.0;
                for (u, &(a, b)) in IDX.iter().enumerate() {
                    let dkl = if k == l { 1.0 } else { 0.0 };
                    let dab = if a == b { 1.0 } else { 0.0 };
                    let mut tt = 2.0 * dkl * dab;
                    if a == k && b == l {
                        tt -= 1.0;
                    }
                    if a == l && b == k {
                        tt -= 1.0;
                    }
                    s += f.g[u] * tt;
                }
                t[k][l] = s;
            }
        }
        for i in 0..nat {
            for j in 0..nat {
                let s = if i == j { w[i] } else { 0.0 } - w[i] * w[j] / wt;
                for k in 0..3 {
                    for l in 0..3 {
                        hess[(3 * i + k) * n3 + 3 * j + l] += s * t[k][l];
                    }
                }
            }
        }
        (f.v, grad, hess)
    }

    /// Second derivatives at fixed charges `q`: the derivative of the potential
    /// `(B q)_i` with respect to the nuclear positions, added to `dv` (row-major
    /// `nat x 3 nat`), and with `hess` the nuclear Hessian of the whole solvation free
    /// energy (row-major `3 nat x 3 nat`). Returns `false` without touching the outputs
    /// for xtb's stale-`r1` Born integrator, whose energy has no consistent derivatives.
    pub fn second_derivatives(&self, cache: &SolvCache, q: &[f64], dv: &mut [f64], hess: Option<&mut [f64]>) -> bool {
        if !self.fresh_r1 {
            return false;
        }
        self.check(cache);
        let nat = self.nat;
        let n3 = 3 * nat;
        let xyz = &cache.xyz;
        let lrcut2 = self.lrcut * self.lrcut;
        let mut hess = hess;
        // psi and its Jacobian; second-derivative pair data kept for the contraction
        let mut psi = vec![0.0; nat];
        let mut jpsi = vec![0.0; nat * n3];
        let mut pairs: Vec<PsiPair> = Vec::new();
        for i in 0..nat {
            for j in 0..i {
                let v = sub3(xyz[i], xyz[j]);
                let r2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
                if !(r2 < lrcut2) {
                    continue;
                }
                let r = r2.sqrt();
                let (ci, cj) = psi_pair(r, self.rho[i], self.rho[j], self.vdwr[i], self.vdwr[j]);
                for (a, c) in [(i, ci), (j, cj)] {
                    if let Some(c) = c {
                        psi[a] += c.v;
                        for k in 0..3 {
                            jpsi[a * n3 + 3 * i + k] += c.d * v[k] / r;
                            jpsi[a * n3 + 3 * j + k] -= c.d * v[k] / r;
                        }
                    }
                }
                pairs.push((i, j, v, r, ci, cj));
            }
        }
        let bt: Vec<D1> = (0..nat).map(|i| self.born_transform(i, psi[i])).collect();
        let a: Vec<f64> = bt.iter().map(|b| b.v).collect();
        // A = da/dR (nat x n3)
        let mut amat = vec![0.0; nat * n3];
        for i in 0..nat {
            for x in 0..n3 {
                amat[i * n3 + x] = bt[i].d * jpsi[i * n3 + x];
            }
        }
        // GB polar term E = sum_{i>j} q_i q_j e_ij + 1/2 sum_i q_i² keps / a_i
        let mut ea = vec![0.0; nat]; // dE/da
        let mut eaa = vec![0.0; nat * nat]; // d²E/da da
        let mut era = vec![0.0; n3 * nat]; // d²E/dR da
        let mut dva = vec![0.0; nat * nat]; // dV_i/da_m
        for i in 0..nat {
            let self_e = self.keps / a[i];
            ea[i] -= 0.5 * q[i] * q[i] * self_e / a[i];
            eaa[i * nat + i] += q[i] * q[i] * self_e / (a[i] * a[i]);
            dva[i * nat + i] -= q[i] * self_e / a[i];
        }
        for i in 0..nat {
            for j in 0..i {
                let v = sub3(xyz[i], xyz[j]);
                let r = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                let e = self.gb_pair(r, a[i], a[j]);
                let qq = q[i] * q[j];
                if let Some(h) = hess.as_deref_mut() {
                    add_radial_hessian(h, n3, i, j, v, r, qq * e.g[0], qq * e.h[0][0]);
                }
                ea[i] += qq * e.g[1];
                ea[j] += qq * e.g[2];
                eaa[i * nat + i] += qq * e.h[1][1];
                eaa[j * nat + j] += qq * e.h[2][2];
                eaa[i * nat + j] += qq * e.h[1][2];
                eaa[j * nat + i] += qq * e.h[1][2];
                for k in 0..3 {
                    let u = v[k] / r;
                    era[(3 * i + k) * nat + i] += qq * e.h[0][1] * u;
                    era[(3 * i + k) * nat + j] += qq * e.h[0][2] * u;
                    era[(3 * j + k) * nat + i] -= qq * e.h[0][1] * u;
                    era[(3 * j + k) * nat + j] -= qq * e.h[0][2] * u;
                    // explicit potential derivative
                    dv[i * n3 + 3 * i + k] += q[j] * e.g[0] * u;
                    dv[i * n3 + 3 * j + k] -= q[j] * e.g[0] * u;
                    dv[j * n3 + 3 * i + k] += q[i] * e.g[0] * u;
                    dv[j * n3 + 3 * j + k] -= q[i] * e.g[0] * u;
                }
                dva[i * nat + i] += q[j] * e.g[1];
                dva[i * nat + j] += q[j] * e.g[2];
                dva[j * nat + j] += q[i] * e.g[2];
                dva[j * nat + i] += q[i] * e.g[1];
            }
        }
        // potential: through the Born radii
        for i in 0..nat {
            for m in 0..nat {
                let w = dva[i * nat + m];
                if w != 0.0 {
                    for x in 0..n3 {
                        dv[i * n3 + x] += w * amat[m * n3 + x];
                    }
                }
            }
        }
        // ALPB shape term: E = 1/2 keps alpbet Q² / A_det, V_i = keps alpbet Q / A_det
        let qtot: f64 = q.iter().sum();
        let shape = if self.alpbet > 0.0 { Some(self.inverse_adet(xyz)) } else { None };
        if let Some((_, g, _)) = &shape {
            let f = self.keps * self.alpbet * qtot;
            for i in 0..nat {
                for x in 0..n3 {
                    dv[i * n3 + x] += f * g[x];
                }
            }
        }
        // hydrogen-bond term: V_i += 2 dhbdw_i S_i q_i
        let mut jsasa = vec![0.0; nat * n3];
        for i in 0..nat {
            let g = cache.deriv.self_grad[i];
            for k in 0..3 {
                jsasa[i * n3 + 3 * i + k] += g[k];
            }
            for p in cache.deriv.nb_ptr[i]..cache.deriv.nb_ptr[i + 1] {
                let j = cache.deriv.nb_idx[p];
                let g = cache.deriv.nb_grad[p];
                for k in 0..3 {
                    jsasa[i * n3 + 3 * j + k] += g[k];
                }
            }
        }
        if self.lhb {
            for i in 0..nat {
                let f = 2.0 * cache.dhbdw[i] * q[i];
                if f != 0.0 {
                    for x in 0..n3 {
                        dv[i * n3 + x] += f * jsasa[i * n3 + x];
                    }
                }
            }
        }
        let Some(h) = hess else { return true };
        // GB chain rule through a(psi(R)):
        // H += X A + (X A)ᵀ + Aᵀ E_aa A + Σ_i E_a,i a''_i ψ'_i ψ'_iᵀ + Σ_i E_a,i a'_i ψ''_i
        let mut xa = vec![0.0; n3 * n3];
        for x in 0..n3 {
            for m in 0..nat {
                let w = era[x * nat + m];
                if w != 0.0 {
                    for y in 0..n3 {
                        xa[x * n3 + y] += w * amat[m * n3 + y];
                    }
                }
            }
        }
        let mut b = eaa.clone();
        // a''-term in psi variables: E_a,i a''_i ψ'ψ'ᵀ = (E_a,i a''_i / a'_i²) A_i A_iᵀ
        for i in 0..nat {
            if bt[i].d != 0.0 {
                b[i * nat + i] += ea[i] * bt[i].dd / (bt[i].d * bt[i].d);
            }
        }
        let mut ba = vec![0.0; nat * n3];
        for i in 0..nat {
            for m in 0..nat {
                let w = b[i * nat + m];
                if w != 0.0 {
                    for y in 0..n3 {
                        ba[i * n3 + y] += w * amat[m * n3 + y];
                    }
                }
            }
        }
        for x in 0..n3 {
            for y in 0..n3 {
                let mut s = xa[x * n3 + y] + xa[y * n3 + x];
                for i in 0..nat {
                    s += amat[i * n3 + x] * ba[i * n3 + y];
                }
                h[x * n3 + y] += s;
            }
        }
        for &(i, j, v, r, ci, cj) in &pairs {
            for (a_, c) in [(i, ci), (j, cj)] {
                if let Some(c) = c {
                    let w = ea[a_] * bt[a_].d;
                    add_radial_hessian(h, n3, i, j, v, r, w * c.d, w * c.dd);
                }
            }
        }
        if let Some((_, _, hs)) = &shape {
            let f = 0.5 * self.keps * self.alpbet * qtot * qtot;
            for (x, y) in h.iter_mut().zip(hs) {
                *x += f * y;
            }
        }
        // surface tension and hydrogen-bond terms: Σ_i W_i S_i
        let wsurf: Vec<f64> =
            (0..nat).map(|i| self.gamsasa[i] + if self.lhb { cache.dhbdw[i] * q[i] * q[i] } else { 0.0 }).collect();
        self.add_sasa_hessian(cache, &wsurf, h);
        true
    }

    /// Add `Σ_i w_i d²S_i/dR dR` of the smoothed SASA to `hess`.
    fn add_sasa_hessian(&self, cache: &SolvCache, w: &[f64], hess: &mut [f64]) {
        let nat = self.nat;
        let n3 = 3 * nat;
        let xyz = &cache.xyz;
        let sm = self.smooth;
        let d = &cache.deriv;
        // per grid point: atoms (iat, neighbours), unit vectors, a = s'/s, c = s''/s - a², |t|
        let mut nb: Vec<usize> = Vec::new();
        let mut tu: Vec<[f64; 3]> = Vec::new();
        let mut aa: Vec<f64> = Vec::new();
        let mut cc: Vec<f64> = Vec::new();
        let mut tn: Vec<f64> = Vec::new();
        for iat in 0..nat {
            if w[iat] == 0.0 {
                continue;
            }
            let (p0, p1) = (d.nb_ptr[iat], d.nb_ptr[iat + 1]);
            let rsas = self.vdwsa[iat];
            let wr = self.wrp[iat];
            let xa = xyz[iat];
            for (ip, p) in self.ang_grid.iter().enumerate() {
                let xp = [xa[0] + rsas * p[0], xa[1] + rsas * p[1], xa[2] + rsas * p[2]];
                nb.clear();
                tu.clear();
                aa.clear();
                cc.clear();
                tn.clear();
                let mut buried = false;
                let mut sasap = 1.0;
                for &ja in &d.nb_idx[p0..p1] {
                    let t = sub3(xp, xyz[ja]);
                    let tj2 = t[0] * t[0] + t[1] * t[1] + t[2] * t[2];
                    if tj2 <= self.trj2[ja][0] {
                        buried = true;
                        break;
                    }
                    if tj2 < self.trj2[ja][1] {
                        let tj = tj2.sqrt();
                        let uj = tj - self.vdwsa[ja];
                        let s = (sm.ah0 + (sm.ah1 + sm.ah3 * uj * uj) * uj).clamp(0.0, 1.0);
                        if s < f64::MIN_POSITIVE {
                            sasap = 0.0;
                            break;
                        }
                        let s1 = sm.ah1 + 3.0 * sm.ah3 * uj * uj;
                        let s2 = 6.0 * sm.ah3 * uj;
                        sasap *= s;
                        nb.push(ja);
                        tu.push([t[0] / tj, t[1] / tj, t[2] / tj]);
                        aa.push(s1 / s);
                        cc.push(s2 / s - (s1 / s) * (s1 / s));
                        tn.push(tj);
                    }
                }
                if buried || !(sasap > TOLSESP) {
                    continue;
                }
                let f = w[iat] * self.ang_weight[ip] * wr * sasap;
                let n = nb.len();
                // gradient of ln P per atom block: block 0 = iat, block 1 + m = nb[m]
                let mut g = vec![[0.0; 3]; n + 1];
                for m in 0..n {
                    for k in 0..3 {
                        g[0][k] += aa[m] * tu[m][k];
                        g[1 + m][k] = -aa[m] * tu[m][k];
                    }
                }
                let atom = |b: usize| if b == 0 { iat } else { nb[b - 1] };
                for bx in 0..=n {
                    for by in 0..=n {
                        let (ax, ay) = (atom(bx), atom(by));
                        for k in 0..3 {
                            for l in 0..3 {
                                hess[(3 * ax + k) * n3 + 3 * ay + l] += f * g[bx][k] * g[by][l];
                            }
                        }
                    }
                }
                for m in 0..n {
                    let ja = nb[m];
                    let u = tu[m];
                    for k in 0..3 {
                        for l in 0..3 {
                            let dkl = if k == l { 1.0 } else { 0.0 };
                            let blk = f * (cc[m] * u[k] * u[l] + aa[m] * (dkl - u[k] * u[l]) / tn[m]);
                            hess[(3 * iat + k) * n3 + 3 * iat + l] += blk;
                            hess[(3 * ja + k) * n3 + 3 * ja + l] += blk;
                            hess[(3 * iat + k) * n3 + 3 * ja + l] -= blk;
                            hess[(3 * ja + k) * n3 + 3 * iat + l] -= blk;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::{Model, ReferenceState, SolvOptions, Solvation};

    fn molecule() -> (Vec<usize>, Vec<[f64; 3]>) {
        // water dimer + methanol + a chloride, bohr
        let z = vec![8, 1, 1, 8, 1, 1, 6, 8, 1, 1, 1, 1, 17];
        let x = vec![
            [0.0, 0.0, 0.0],
            [1.80, 0.10, 0.0],
            [-0.45, 1.75, 0.10],
            [5.40, 0.30, 0.20],
            [6.00, 1.90, -0.30],
            [6.10, -1.10, -0.70],
            [2.0, -5.0, 1.0],
            [4.6, -5.3, 1.4],
            [1.3, -6.8, 0.4],
            [1.4, -4.5, 2.9],
            [1.5, -3.7, -0.4],
            [5.3, -4.0, 2.5],
            [-3.5, -3.0, 2.0],
        ];
        (z, x)
    }

    fn check(model: Model, solvent: &str) {
        let (z, x) = molecule();
        let nat = z.len();
        let n3 = 3 * nat;
        let opts = SolvOptions { conventions: crate::Conventions::Corrected, ..SolvOptions::default() };
        let s = Solvation::with_options(&z, solvent, model, ReferenceState::Gsolv, &opts).unwrap();
        let q: Vec<f64> = (0..nat).map(|i| 0.4 * ((i as f64) * 1.3).sin() - 0.05).collect();
        let c = s.update(&x);
        let mut dv = vec![0.0; nat * n3];
        let mut h = vec![0.0; n3 * n3];
        assert!(s.second_derivatives(&c, &q, &mut dv, Some(&mut h)));
        let step = 1e-5;
        let (mut edv, mut sdv, mut eh, mut sh) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
        for y in 0..n3 {
            let mut xp = x.clone();
            xp[y / 3][y % 3] += step;
            let mut xm = x.clone();
            xm[y / 3][y % 3] -= step;
            let (cp, cm) = (s.update(&xp), s.update(&xm));
            let (mut vp, mut vm) = (vec![0.0; nat], vec![0.0; nat]);
            s.potential(&cp, &q, &mut vp);
            s.potential(&cm, &q, &mut vm);
            for i in 0..nat {
                let fd = (vp[i] - vm[i]) / (2.0 * step);
                edv = edv.max((fd - dv[i * n3 + y]).abs());
                sdv = sdv.max(fd.abs());
            }
            let (mut gp, mut gm) = (vec![[0.0; 3]; nat], vec![[0.0; 3]; nat]);
            let mut e = vec![0.0; nat];
            s.fixed_energy(&cp, &mut e, Some(&mut gp));
            s.gradient(&cp, &q, &mut gp);
            s.fixed_energy(&cm, &mut e, Some(&mut gm));
            s.gradient(&cm, &q, &mut gm);
            for xx in 0..n3 {
                let fd = (gp[xx / 3][xx % 3] - gm[xx / 3][xx % 3]) / (2.0 * step);
                eh = eh.max((fd - h[xx * n3 + y]).abs());
                sh = sh.max(fd.abs());
            }
        }
        eprintln!("{model:?} {solvent}: potential derivative error {edv:.2e} (max {sdv:.2e}), Hessian error {eh:.2e} (max {sh:.2e})");
        assert!(edv < 1e-8 * sdv.max(1e-3), "potential derivative {edv:.3e}");
        assert!(eh < 1e-8 * sh.max(1e-3), "hessian {eh:.3e}");
    }

    /// Diagnostic: FD Hessians of the surface (fixed) and charge-dependent parts at large
    /// and small steps; a part whose difference is far above O(h²) is not C².
    #[test]
    #[ignore]
    fn smoothness_of_parts() {
        let (z, x) = molecule();
        let nat = z.len();
        let n3 = 3 * nat;
        let opts = SolvOptions { conventions: crate::Conventions::Corrected, ..SolvOptions::default() };
        let s = Solvation::with_options(&z, "water", Model::Alpb, ReferenceState::Gsolv, &opts).unwrap();
        let q: Vec<f64> = (0..nat).map(|i| 0.4 * ((i as f64) * 1.3).sin() - 0.05).collect();
        let fd = |step: f64, part: usize| -> Vec<f64> {
            let mut h = vec![0.0; n3 * n3];
            for y in 0..n3 {
                let mut g = [vec![[0.0; 3]; nat], vec![[0.0; 3]; nat]];
                for (k, sign) in [1.0, -1.0].into_iter().enumerate() {
                    let mut xp = x.clone();
                    xp[y / 3][y % 3] += sign * step;
                    let c = s.update(&xp);
                    let mut e = vec![0.0; nat];
                    match part {
                        0 => s.fixed_energy(&c, &mut e, Some(&mut g[k])),
                        _ => s.gradient(&c, &q, &mut g[k]),
                    }
                }
                for xx in 0..n3 {
                    h[xx * n3 + y] = (g[0][xx / 3][xx % 3] - g[1][xx / 3][xx % 3]) / (2.0 * step);
                }
            }
            h
        };
        for (part, name) in [(0, "surface tension"), (1, "polar + HB + shape")] {
            let h0 = fd(1e-5, part);
            for step in [1e-4, 1e-3, 5e-3] {
                let h1 = fd(step, part);
                let d = h0.iter().zip(&h1).fold(0.0f64, |m, (a, b)| m.max((a - b).abs()));
                eprintln!("{name}: |FD({step:.0e}) - FD(1e-5)| = {d:.2e}");
            }
        }
    }

    #[test]
    fn second_derivatives_match_finite_differences() {
        check(Model::Alpb, "water");
        check(Model::Gbsa, "water");
        check(Model::Alpb, "toluene");
    }
}
