// SPDX-License-Identifier: LGPL-3.0-or-later
//! One-electron integrals over contracted Gaussian shells: overlap, dipole and traceless
//! quadrupole moments, and their derivatives with respect to the ket center.
//!
//! Conventions for a shell pair with bra shell `cj` on center R_j and ket shell `ci` on
//! center R_i, and `vec = R_i - R_j`:
//! - blocks are stored `[jm + nsph(lj) * im]` (bra index fastest), spherical components
//!   ordered m = -l..l (p: y, z, x);
//! - `d_i`, `q_i` are moments of `(r - R_i)` (operator on the ket center), `d_j`, `q_j` of
//!   `(r - R_j)`;
//! - quadrupoles are packed (xx, xy, yy, xz, yz, zz) in the traceless form
//!   `Q = 3/2 r_a r_b - 1/2 r^2 δ_ab`;
//! - derivatives are with respect to R_i, so the derivative with respect to R_j of a
//!   translation-invariant quantity is its negative.

use crate::basis::{ncart, nsph, Cgto, MAXL};
use std::f64::consts::PI;
use std::sync::OnceLock;

/// Largest number of spherical components of a shell.
pub const MSAO: usize = 2 * MAXL + 1;
/// Largest number of Cartesian components of a shell.
const MCART: usize = (MAXL + 1) * (MAXL + 2) / 2;
/// Size of the 1D moment table per direction.
const MT: usize = MAXL + 4;
/// Size of one spherical shell-pair block.
pub const MBLK: usize = MSAO * MSAO;

/// Cartesian exponents (lx, ly, lz) in CCA order for each l.
pub(crate) fn cart_exponents(l: usize) -> &'static [[usize; 3]] {
    static TABLE: OnceLock<Vec<Vec<[usize; 3]>>> = OnceLock::new();
    &TABLE.get_or_init(|| {
        (0..=MAXL)
            .map(|l| {
                let mut v = Vec::new();
                for lx in (0..=l).rev() {
                    for ly in (0..=l - lx).rev() {
                        v.push([lx, ly, l - lx - ly]);
                    }
                }
                v
            })
            .collect()
    })[l]
}

/// Cartesian-to-spherical transformation matrices, `T[m][c]`, as dense row-major
/// `nsph(l) x ncart(l)` arrays.
pub(crate) fn sph_trafo(l: usize) -> &'static [f64] {
    static TABLE: OnceLock<Vec<Vec<f64>>> = OnceLock::new();
    &TABLE.get_or_init(|| {
        let mut out = Vec::new();
        // s
        out.push(vec![1.0]);
        // p: spherical order y, z, x from Cartesian x, y, z
        out.push(vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0]);
        // d: Cartesian xx, xy, xz, yy, yz, zz
        let s3 = 3f64.sqrt();
        let s3_4 = s3 * 0.5;
        let d = [
            [0.0, s3, 0.0, 0.0, 0.0, 0.0],     // m=-2
            [0.0, 0.0, 0.0, 0.0, s3, 0.0],     // m=-1
            [-0.5, 0.0, 0.0, -0.5, 0.0, 1.0],  // m=0
            [0.0, 0.0, s3, 0.0, 0.0, 0.0],     // m=1
            [s3_4, 0.0, 0.0, -s3_4, 0.0, 0.0], // m=2
        ];
        out.push(d.iter().flatten().copied().collect());
        // f: Cartesian xxx, xxy, xxz, xyy, xyz, xzz, yyy, yyz, yzz, zzz
        let s3_8 = (3.0f64 / 8.0).sqrt();
        let s5_8 = (5.0f64 / 8.0).sqrt();
        let s6 = 6f64.sqrt();
        let s10 = 10f64.sqrt();
        let s45_8 = 3.0 * s10 / 4.0;
        let s15 = 15f64.sqrt();
        let s15_4 = s15 / 2.0;
        let d3_2 = 1.5;
        let mut f = vec![0.0; 7 * 10];
        let mut set = |m: i32, c: usize, v: f64| f[(m + 3) as usize * 10 + c] = v;
        set(1, 0, -s3_8);
        set(3, 0, s5_8);
        set(-3, 1, s45_8);
        set(-1, 1, -s3_8);
        set(0, 2, -d3_2);
        set(2, 2, s15_4);
        set(1, 3, -s3_8);
        set(3, 3, -s45_8);
        set(-2, 4, s15);
        set(1, 5, s6);
        set(-3, 6, -s5_8);
        set(-1, 6, -s3_8);
        set(0, 7, -d3_2);
        set(2, 7, -s15_4);
        set(-1, 8, s6);
        set(0, 9, 1.0);
        out.push(f);
        out
    })[l]
}

/// 1D Gaussian moment table `t[k][a][b] = ∫ (x-A)^a (x-B)^b` scaled by the prefactor,
/// built by the Obara–Saika horizontal/vertical recursion around the product center.
#[inline]
fn build_1d(la: usize, lb: usize, rpa: [f64; 3], rpb: [f64; 3], oab: f64, t: &mut [[[f64; MT]; MT]; 3]) {
    let h = 0.5 * oab;
    for k in 0..3 {
        let tk = &mut t[k];
        tk[0][0] = 1.0;
        if la >= 1 {
            tk[1][0] = rpa[k];
            for i in 1..la {
                tk[i + 1][0] = rpa[k] * tk[i][0] + i as f64 * h * tk[i - 1][0];
            }
        }
        if lb >= 1 {
            tk[0][1] = rpb[k];
            for i in 1..=la {
                tk[i][1] = rpb[k] * tk[i][0] + i as f64 * h * tk[i - 1][0];
            }
            for j in 1..lb {
                tk[0][j + 1] = rpb[k] * tk[0][j] + j as f64 * h * tk[0][j - 1];
                for i in 1..=la {
                    tk[i][j + 1] = rpb[k] * tk[i][j] + h * (i as f64 * tk[i - 1][j] + j as f64 * tk[i][j - 1]);
                }
            }
        }
    }
}

/// Transform a Cartesian block `cart[jc + ncart(lj) * ic]` to spherical
/// `out[jm + nsph(lj) * im]`, for `n` interleaved components (`cart[(jc + ..)*n + x]`).
#[inline]
fn to_spherical<const N: usize>(lj: usize, li: usize, cart: &[[f64; N]], out: &mut [[f64; N]], tmp: &mut [[f64; N]]) {
    let (ncj, nci) = (ncart(lj), ncart(li));
    let (nsj, nsi) = (nsph(lj), nsph(li));
    if lj <= 1 && li <= 1 {
        // pure permutation
        let pj = |m: usize| if lj == 1 { [1, 2, 0][m] } else { 0 };
        let pi = |m: usize| if li == 1 { [1, 2, 0][m] } else { 0 };
        for im in 0..nsi {
            for jm in 0..nsj {
                out[jm + nsj * im] = cart[pj(jm) + ncj * pi(im)];
            }
        }
        return;
    }
    let tj = sph_trafo(lj);
    let ti = sph_trafo(li);
    // tmp[jm + nsj*ic] = sum_jc Tj[jm][jc] cart[jc][ic] (every entry is written before use)
    for ic in 0..nci {
        for jm in 0..nsj {
            let mut acc = [0.0; N];
            for jc in 0..ncj {
                let t = tj[jm * ncj + jc];
                if t != 0.0 {
                    let c = &cart[jc + ncj * ic];
                    for x in 0..N {
                        acc[x] += t * c[x];
                    }
                }
            }
            tmp[jm + nsj * ic] = acc;
        }
    }
    for im in 0..nsi {
        for jm in 0..nsj {
            let mut acc = [0.0; N];
            for ic in 0..nci {
                let t = ti[im * nci + ic];
                if t != 0.0 {
                    let c = &tmp[jm + nsj * ic];
                    for x in 0..N {
                        acc[x] += t * c[x];
                    }
                }
            }
            out[jm + nsj * im] = acc;
        }
    }
}

/// Reusable work buffers for the integral kernels (one per thread).
pub struct Scratch {
    cart10: Vec<[f64; 10]>,
    cart30: Vec<[f64; 30]>,
    cart60: Vec<[f64; 60]>,
    sph10: Vec<[f64; 10]>,
    sph30: Vec<[f64; 30]>,
    sph60: Vec<[f64; 60]>,
    tmp10: Vec<[f64; 10]>,
    tmp30: Vec<[f64; 30]>,
    tmp60: Vec<[f64; 60]>,
}

impl Default for Scratch {
    fn default() -> Self {
        Self {
            cart10: vec![[0.0; 10]; MCART * MCART],
            cart30: vec![[0.0; 30]; MCART * MCART],
            cart60: vec![[0.0; 60]; MCART * MCART],
            sph10: vec![[0.0; 10]; MBLK],
            sph30: vec![[0.0; 30]; MBLK],
            sph60: vec![[0.0; 60]; MBLK],
            tmp10: vec![[0.0; 10]; MSAO * MCART],
            tmp30: vec![[0.0; 30]; MSAO * MCART],
            tmp60: vec![[0.0; 60]; MSAO * MCART],
        }
    }
}

/// Whether every primitive pair of two shells with smallest exponents `a` and `b` at squared
/// distance `r2` is below the integral cutoff (`prefactor` skips each of them), so their
/// integral blocks are exactly zero: the Gaussian product exponent ab/(a+b) r^2 grows with
/// both exponents. With a relative margin of 1e-10 for the different rounding of the two
/// expressions.
#[inline]
pub fn pair_screened(a: f64, b: f64, r2: f64, intcut: f64) -> bool {
    a * b * r2 / (a + b) > intcut * (1.0 + 1e-10)
}

#[inline]
fn prefactor(ai: f64, aj: f64, r2: f64, intcut: f64) -> Option<(f64, f64)> {
    let eab = ai + aj;
    let oab = 1.0 / eab;
    let est = ai * aj * r2 * oab;
    if est > intcut {
        return None;
    }
    let sqrtpi3 = PI.sqrt().powi(3);
    Some((oab, (-est).exp() * sqrtpi3 * oab.sqrt().powi(3)))
}

/// Remove the trace: Q = 3/2 M - 1/2 tr(M).
#[inline]
fn traceless(q: &mut [f64; 6]) {
    let tr = 0.5 * (q[0] + q[2] + q[5]);
    q[0] = 1.5 * q[0] - tr;
    q[1] *= 1.5;
    q[2] = 1.5 * q[2] - tr;
    q[3] *= 1.5;
    q[4] *= 1.5;
    q[5] = 1.5 * q[5] - tr;
}

/// Overlap block of a shell pair.
pub fn overlap(cj: &Cgto, ci: &Cgto, vec: [f64; 3], intcut: f64, s: &mut [f64; MBLK]) {
    let r2 = vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2];
    let (li, lj) = (ci.l, cj.l);
    let (ei, ej) = (cart_exponents(li), cart_exponents(lj));
    let ncj = ncart(lj);
    let mut cart = [[0.0; 1]; MCART * MCART];
    let mut t = [[[0.0; MT]; MT]; 3];
    for ip in 0..ci.nprim {
        for jp in 0..cj.nprim {
            let (ai, aj) = (ci.alpha[ip], cj.alpha[jp]);
            let Some((oab, pre)) = prefactor(ai, aj, r2, intcut) else { continue };
            let rpi = [-vec[0] * aj * oab, -vec[1] * aj * oab, -vec[2] * aj * oab];
            let rpj = [vec[0] * ai * oab, vec[1] * ai * oab, vec[2] * ai * oab];
            build_1d(li, lj, rpi, rpj, oab, &mut t);
            let cc = ci.coeff[ip] * cj.coeff[jp] * pre;
            for (ic, a) in ei.iter().enumerate() {
                for (jc, b) in ej.iter().enumerate() {
                    cart[jc + ncj * ic][0] += cc * t[0][a[0]][b[0]] * t[1][a[1]][b[1]] * t[2][a[2]][b[2]];
                }
            }
        }
    }
    let mut out = [[0.0; 1]; MBLK];
    let mut tmp = [[0.0; 1]; MSAO * MCART];
    to_spherical(lj, li, &cart, &mut out, &mut tmp);
    for k in 0..nsph(lj) * nsph(li) {
        s[k] = out[k][0];
    }
}

/// Overlap, dipole and traceless quadrupole blocks of a shell pair, moments around the
/// ket center R_i.
pub fn multipole(
    cj: &Cgto,
    ci: &Cgto,
    vec: [f64; 3],
    intcut: f64,
    s: &mut [f64; MBLK],
    d: &mut [[f64; 3]; MBLK],
    q: &mut [[f64; 6]; MBLK],
    scratch: &mut Scratch,
) {
    multipole_moments(cj, ci, vec, intcut, s, d, q, scratch, true);
}

/// As [`multipole`], with the raw second moments (xx, xy, yy, xz, yz, zz) instead of the
/// traceless quadrupole.
pub fn multipole_raw(
    cj: &Cgto,
    ci: &Cgto,
    vec: [f64; 3],
    intcut: f64,
    s: &mut [f64; MBLK],
    d: &mut [[f64; 3]; MBLK],
    q: &mut [[f64; 6]; MBLK],
    scratch: &mut Scratch,
) {
    multipole_moments(cj, ci, vec, intcut, s, d, q, scratch, false);
}

fn multipole_moments(
    cj: &Cgto,
    ci: &Cgto,
    vec: [f64; 3],
    intcut: f64,
    s: &mut [f64; MBLK],
    d: &mut [[f64; 3]; MBLK],
    q: &mut [[f64; 6]; MBLK],
    scratch: &mut Scratch,
    remove_trace: bool,
) {
    let r2 = vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2];
    let (li, lj) = (ci.l, cj.l);
    let (ei, ej) = (cart_exponents(li), cart_exponents(lj));
    let ncj = ncart(lj);
    // components: s, dx, dy, dz, qxx, qxy, qyy, qxz, qyz, qzz
    let cart = &mut scratch.cart10[..ncj * ncart(li)];
    cart.iter_mut().for_each(|c| *c = [0.0; 10]);
    let mut t = [[[0.0; MT]; MT]; 3];
    for ip in 0..ci.nprim {
        for jp in 0..cj.nprim {
            let (ai, aj) = (ci.alpha[ip], cj.alpha[jp]);
            let Some((oab, pre)) = prefactor(ai, aj, r2, intcut) else { continue };
            let rpi = [-vec[0] * aj * oab, -vec[1] * aj * oab, -vec[2] * aj * oab];
            let rpj = [vec[0] * ai * oab, vec[1] * ai * oab, vec[2] * ai * oab];
            build_1d(li + 2, lj, rpi, rpj, oab, &mut t);
            let cc = ci.coeff[ip] * cj.coeff[jp] * pre;
            for (ic, a) in ei.iter().enumerate() {
                for (jc, b) in ej.iter().enumerate() {
                    let mut v = [[0.0; 3]; 3];
                    for k in 0..3 {
                        for m in 0..3 {
                            v[k][m] = t[k][a[k] + m][b[k]];
                        }
                    }
                    let c = &mut cart[jc + ncj * ic];
                    c[0] += cc * v[0][0] * v[1][0] * v[2][0];
                    c[1] += cc * v[0][1] * v[1][0] * v[2][0];
                    c[2] += cc * v[0][0] * v[1][1] * v[2][0];
                    c[3] += cc * v[0][0] * v[1][0] * v[2][1];
                    c[4] += cc * v[0][2] * v[1][0] * v[2][0];
                    c[5] += cc * v[0][1] * v[1][1] * v[2][0];
                    c[6] += cc * v[0][0] * v[1][2] * v[2][0];
                    c[7] += cc * v[0][1] * v[1][0] * v[2][1];
                    c[8] += cc * v[0][0] * v[1][1] * v[2][1];
                    c[9] += cc * v[0][0] * v[1][0] * v[2][2];
                }
            }
        }
    }
    let sph = &mut scratch.sph10;
    to_spherical(lj, li, cart, sph, &mut scratch.tmp10);
    for k in 0..nsph(lj) * nsph(li) {
        let c = &sph[k];
        s[k] = c[0];
        d[k] = [c[1], c[2], c[3]];
        let mut qq = [c[4], c[5], c[6], c[7], c[8], c[9]];
        if remove_trace {
            traceless(&mut qq);
        }
        q[k] = qq;
    }
}

/// Derivatives of the shell-pair integrals with respect to the ket center R_i.
#[derive(Clone)]
pub struct MultipoleGrad {
    /// Overlap.
    pub s: [f64; MBLK],
    /// d S / d R_i.
    pub ds: [[f64; 3]; MBLK],
    /// `d D_i[c] / d R_i[k]` as `dd_i[blk][k][c]` (dipole around R_i).
    pub dd_i: [[[f64; 3]; 3]; MBLK],
    /// `d D_j[c] / d R_i[k]` (dipole around R_j).
    pub dd_j: [[[f64; 3]; 3]; MBLK],
    /// `d Q_i[c] / d R_i[k]` (traceless quadrupole around R_i).
    pub dq_i: [[[f64; 6]; 3]; MBLK],
    /// `d Q_j[c] / d R_i[k]`.
    pub dq_j: [[[f64; 6]; 3]; MBLK],
}

impl Default for MultipoleGrad {
    fn default() -> Self {
        Self {
            s: [0.0; MBLK],
            ds: [[0.0; 3]; MBLK],
            dd_i: [[[0.0; 3]; 3]; MBLK],
            dd_j: [[[0.0; 3]; 3]; MBLK],
            dq_i: [[[0.0; 6]; 3]; MBLK],
            dq_j: [[[0.0; 6]; 3]; MBLK],
        }
    }
}

/// Overlap and multipole integrals with their ket-center derivatives.
pub fn multipole_grad(
    cj: &Cgto,
    ci: &Cgto,
    vec: [f64; 3],
    intcut: f64,
    out: &mut MultipoleGrad,
    scratch: &mut Scratch,
) {
    let r2 = vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2];
    let (li, lj) = (ci.l, cj.l);
    let (ei, ej) = (cart_exponents(li), cart_exponents(lj));
    let ncj = ncart(lj);
    // per Cartesian pair: [s, d_j(3), m_j(6)] values and their 3 derivatives
    const NV: usize = 10;
    let ncart_pair = ncj * ncart(li);
    let val = &mut scratch.cart10[..ncart_pair];
    val.iter_mut().for_each(|c| *c = [0.0; NV]);
    let der = &mut scratch.cart30[..ncart_pair];
    der.iter_mut().for_each(|c| *c = [0.0; 3 * NV]);
    let mut t = [[[0.0; MT]; MT]; 3];
    for ip in 0..ci.nprim {
        let ai2 = 2.0 * ci.alpha[ip];
        for jp in 0..cj.nprim {
            let (ai, aj) = (ci.alpha[ip], cj.alpha[jp]);
            let Some((oab, pre)) = prefactor(ai, aj, r2, intcut) else { continue };
            let rpi = [-vec[0] * aj * oab, -vec[1] * aj * oab, -vec[2] * aj * oab];
            let rpj = [vec[0] * ai * oab, vec[1] * ai * oab, vec[2] * ai * oab];
            build_1d(li + 1, lj + 2, rpi, rpj, oab, &mut t);
            let cc = ci.coeff[ip] * cj.coeff[jp] * pre;
            for (ic, a) in ei.iter().enumerate() {
                for (jc, b) in ej.iter().enumerate() {
                    let mut v = [[0.0; 3]; 3];
                    let mut g = [[0.0; 3]; 3];
                    for k in 0..3 {
                        for m in 0..3 {
                            v[k][m] = t[k][a[k]][b[k] + m];
                            g[k][m] = ai2 * t[k][a[k] + 1][b[k] + m];
                            if a[k] > 0 {
                                g[k][m] -= a[k] as f64 * t[k][a[k] - 1][b[k] + m];
                            }
                        }
                    }
                    // moment exponents (mx, my, mz) for s, dx, dy, dz, xx, xy, yy, xz, yz, zz
                    const MOM: [[usize; 3]; NV] = [
                        [0, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [2, 0, 0],
                        [1, 1, 0],
                        [0, 2, 0],
                        [1, 0, 1],
                        [0, 1, 1],
                        [0, 0, 2],
                    ];
                    let vc = &mut val[jc + ncj * ic];
                    let dc = &mut der[jc + ncj * ic];
                    for (n, m) in MOM.iter().enumerate() {
                        let (x, y, z) = (v[0][m[0]], v[1][m[1]], v[2][m[2]]);
                        vc[n] += cc * x * y * z;
                        dc[3 * n] += cc * g[0][m[0]] * y * z;
                        dc[3 * n + 1] += cc * x * g[1][m[1]] * z;
                        dc[3 * n + 2] += cc * x * y * g[2][m[2]];
                    }
                }
            }
        }
    }
    let sval = &mut scratch.sph10;
    let sder = &mut scratch.sph30;
    to_spherical(lj, li, val, sval, &mut scratch.tmp10);
    to_spherical(lj, li, der, sder, &mut scratch.tmp30);
    for blk in 0..nsph(lj) * nsph(li) {
        let v = &sval[blk];
        let dv = &sder[blk];
        let s = v[0];
        let dj = [v[1], v[2], v[3]];
        let ds = [dv[0], dv[1], dv[2]];
        out.s[blk] = s;
        out.ds[blk] = ds;
        // derivatives of the R_j-centered moments
        let mut ddj = [[0.0; 3]; 3];
        let mut dmj = [[0.0; 6]; 3];
        for k in 0..3 {
            for c in 0..3 {
                ddj[k][c] = dv[3 * (1 + c) + k];
            }
            for c in 0..6 {
                dmj[k][c] = dv[3 * (4 + c) + k];
            }
        }
        // shift to R_i: D_i = D_j - vec S ; M_i = M_j - vec_a D_j,b - vec_b D_j,a + vec_a vec_b S
        let mut ddi = [[0.0; 3]; 3];
        let mut dmi = [[0.0; 6]; 3];
        // packed (a, b) index pairs of the quadrupole components
        const AB: [(usize, usize); 6] = [(0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2)];
        for k in 0..3 {
            for c in 0..3 {
                ddi[k][c] = ddj[k][c] - vec[c] * ds[k];
            }
            ddi[k][k] -= s;
            for (n, &(a, b)) in AB.iter().enumerate() {
                let mut x = dmj[k][n] - vec[a] * ddj[k][b] - vec[b] * ddj[k][a] + vec[a] * vec[b] * ds[k];
                if a == k {
                    x += -dj[b] + vec[b] * s;
                }
                if b == k {
                    x += -dj[a] + vec[a] * s;
                }
                dmi[k][n] = x;
            }
        }
        for k in 0..3 {
            traceless(&mut dmi[k]);
            traceless(&mut dmj[k]);
        }
        out.dd_i[blk] = ddi;
        out.dd_j[blk] = ddj;
        out.dq_i[blk] = dmi;
        out.dq_j[blk] = dmj;
    }
}

/// Overlap block of the shell pair (`cj`, `ci`) and its derivative with respect to the ket
/// center R_i (the `s` and `ds` of [`multipole_grad`], bit for bit), plus the overlaps of the
/// same pair with the contraction coefficients of `dcj` on the bra (`sj`) and of `dci` on the
/// ket (`si`; [`overlap`]'s bits), from one pass over the primitive pairs (`dcj`, `dci` must
/// have the exponents of `cj`, `ci`; e.g. their coefficient derivatives).
pub fn overlap_grad_coeff(
    cj: &Cgto,
    ci: &Cgto,
    dcj: Option<&Cgto>,
    dci: Option<&Cgto>,
    vec: [f64; 3],
    intcut: f64,
    s: &mut [f64; MBLK],
    ds: &mut [[f64; 3]; MBLK],
    sj: &mut [f64; MBLK],
    si: &mut [f64; MBLK],
) {
    let r2 = vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2];
    let (li, lj) = (ci.l, cj.l);
    let (ei, ej) = (cart_exponents(li), cart_exponents(lj));
    let ncj = ncart(lj);
    // per Cartesian pair: s, its three ket derivatives, the bra- and ket-coefficient overlaps
    let mut val = [[0.0; 6]; MCART * MCART];
    let mut t = [[[0.0; MT]; MT]; 3];
    for ip in 0..ci.nprim {
        let ai2 = 2.0 * ci.alpha[ip];
        for jp in 0..cj.nprim {
            let (ai, aj) = (ci.alpha[ip], cj.alpha[jp]);
            let Some((oab, pre)) = prefactor(ai, aj, r2, intcut) else { continue };
            let rpi = [-vec[0] * aj * oab, -vec[1] * aj * oab, -vec[2] * aj * oab];
            let rpj = [vec[0] * ai * oab, vec[1] * ai * oab, vec[2] * ai * oab];
            build_1d(li + 1, lj, rpi, rpj, oab, &mut t);
            let cc = ci.coeff[ip] * cj.coeff[jp] * pre;
            let ccj = dcj.map_or(0.0, |d| ci.coeff[ip] * d.coeff[jp] * pre);
            let cci = dci.map_or(0.0, |d| d.coeff[ip] * cj.coeff[jp] * pre);
            for (ic, a) in ei.iter().enumerate() {
                for (jc, b) in ej.iter().enumerate() {
                    let mut v = [0.0; 3];
                    let mut g = [0.0; 3];
                    for k in 0..3 {
                        v[k] = t[k][a[k]][b[k]];
                        g[k] = ai2 * t[k][a[k] + 1][b[k]];
                        if a[k] > 0 {
                            g[k] -= a[k] as f64 * t[k][a[k] - 1][b[k]];
                        }
                    }
                    let (x, y, z) = (v[0], v[1], v[2]);
                    let c = &mut val[jc + ncj * ic];
                    c[0] += cc * x * y * z;
                    c[1] += cc * g[0] * y * z;
                    c[2] += cc * x * g[1] * z;
                    c[3] += cc * x * y * g[2];
                    c[4] += ccj * x * y * z;
                    c[5] += cci * x * y * z;
                }
            }
        }
    }
    let mut sph = [[0.0; 6]; MBLK];
    let mut tmp = [[0.0; 6]; MSAO * MCART];
    to_spherical(lj, li, &val, &mut sph, &mut tmp);
    for blk in 0..nsph(lj) * nsph(li) {
        let v = &sph[blk];
        s[blk] = v[0];
        ds[blk] = [v[1], v[2], v[3]];
        sj[blk] = v[4];
        si[blk] = v[5];
    }
}

/// Moments of a shell pair around both centres: overlap, dipoles and traceless quadrupoles
/// around the ket centre R_i (`d_i`, `q_i`) and the bra centre R_j (`d_j`, `q_j`).
#[derive(Clone)]
pub struct Moments {
    pub s: [f64; MBLK],
    pub d_i: [[f64; 3]; MBLK],
    pub q_i: [[f64; 6]; MBLK],
    pub d_j: [[f64; 3]; MBLK],
    pub q_j: [[f64; 6]; MBLK],
}

impl Default for Moments {
    fn default() -> Self {
        Self {
            s: [0.0; MBLK],
            d_i: [[0.0; 3]; MBLK],
            q_i: [[0.0; 6]; MBLK],
            d_j: [[0.0; 3]; MBLK],
            q_j: [[0.0; 6]; MBLK],
        }
    }
}

/// [`multipole_grad`] of the shell pair (`cj`, `ci`) plus the [`Moments`] of the same pair with
/// the contraction coefficients of `dcj` on the bra (`mj`) and of `dci` on the ket (`mi`), from
/// one pass over the primitive pairs (`dcj`, `dci` must have the exponents of `cj`, `ci`). The
/// `out` part equals [`multipole_grad`] bit for bit; the moments equal those of [`multipole`] up
/// to rounding (they are formed around R_j and shifted to R_i).
pub fn multipole_grad_coeff(
    cj: &Cgto,
    ci: &Cgto,
    dcj: &Cgto,
    dci: &Cgto,
    vec: [f64; 3],
    intcut: f64,
    out: &mut MultipoleGrad,
    mj: &mut Moments,
    mi: &mut Moments,
    scratch: &mut Scratch,
) {
    let r2 = vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2];
    let (li, lj) = (ci.l, cj.l);
    let (ei, ej) = (cart_exponents(li), cart_exponents(lj));
    let ncj = ncart(lj);
    const NV: usize = 10;
    let ncart_pair = ncj * ncart(li);
    // per Cartesian pair: [s, d_j(3), m_j(6)] with the base coefficients, with dcj (bra) and
    // with dci (ket), then the 30 derivatives of the base values
    let acc = &mut scratch.cart60[..ncart_pair];
    acc.iter_mut().for_each(|c| *c = [0.0; 6 * NV]);
    let mut t = [[[0.0; MT]; MT]; 3];
    for ip in 0..ci.nprim {
        let ai2 = 2.0 * ci.alpha[ip];
        for jp in 0..cj.nprim {
            let (ai, aj) = (ci.alpha[ip], cj.alpha[jp]);
            let Some((oab, pre)) = prefactor(ai, aj, r2, intcut) else { continue };
            let rpi = [-vec[0] * aj * oab, -vec[1] * aj * oab, -vec[2] * aj * oab];
            let rpj = [vec[0] * ai * oab, vec[1] * ai * oab, vec[2] * ai * oab];
            build_1d(li + 1, lj + 2, rpi, rpj, oab, &mut t);
            let cc = ci.coeff[ip] * cj.coeff[jp] * pre;
            let ccj = ci.coeff[ip] * dcj.coeff[jp] * pre;
            let cci = dci.coeff[ip] * cj.coeff[jp] * pre;
            for (ic, a) in ei.iter().enumerate() {
                for (jc, b) in ej.iter().enumerate() {
                    let mut v = [[0.0; 3]; 3];
                    let mut g = [[0.0; 3]; 3];
                    for k in 0..3 {
                        for m in 0..3 {
                            v[k][m] = t[k][a[k]][b[k] + m];
                            g[k][m] = ai2 * t[k][a[k] + 1][b[k] + m];
                            if a[k] > 0 {
                                g[k][m] -= a[k] as f64 * t[k][a[k] - 1][b[k] + m];
                            }
                        }
                    }
                    const MOM: [[usize; 3]; NV] = [
                        [0, 0, 0],
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                        [2, 0, 0],
                        [1, 1, 0],
                        [0, 2, 0],
                        [1, 0, 1],
                        [0, 1, 1],
                        [0, 0, 2],
                    ];
                    let c = &mut acc[jc + ncj * ic];
                    for (n, m) in MOM.iter().enumerate() {
                        let (x, y, z) = (v[0][m[0]], v[1][m[1]], v[2][m[2]]);
                        let xyz = x * y * z;
                        c[n] += cc * x * y * z;
                        c[NV + n] += ccj * xyz;
                        c[2 * NV + n] += cci * xyz;
                        c[3 * NV + 3 * n] += cc * g[0][m[0]] * y * z;
                        c[3 * NV + 3 * n + 1] += cc * x * g[1][m[1]] * z;
                        c[3 * NV + 3 * n + 2] += cc * x * y * g[2][m[2]];
                    }
                }
            }
        }
    }
    let sph = &mut scratch.sph60;
    to_spherical(lj, li, acc, sph, &mut scratch.tmp60);
    const AB: [(usize, usize); 6] = [(0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2)];
    for blk in 0..nsph(lj) * nsph(li) {
        let v = &sph[blk][..3 * NV];
        let dv = &sph[blk][3 * NV..];
        let s = v[0];
        let dj = [v[1], v[2], v[3]];
        let ds = [dv[0], dv[1], dv[2]];
        out.s[blk] = s;
        out.ds[blk] = ds;
        let mut ddj = [[0.0; 3]; 3];
        let mut dmj = [[0.0; 6]; 3];
        for k in 0..3 {
            for c in 0..3 {
                ddj[k][c] = dv[3 * (1 + c) + k];
            }
            for c in 0..6 {
                dmj[k][c] = dv[3 * (4 + c) + k];
            }
        }
        let mut ddi = [[0.0; 3]; 3];
        let mut dmi = [[0.0; 6]; 3];
        for k in 0..3 {
            for c in 0..3 {
                ddi[k][c] = ddj[k][c] - vec[c] * ds[k];
            }
            ddi[k][k] -= s;
            for (n, &(a, b)) in AB.iter().enumerate() {
                let mut x = dmj[k][n] - vec[a] * ddj[k][b] - vec[b] * ddj[k][a] + vec[a] * vec[b] * ds[k];
                if a == k {
                    x += -dj[b] + vec[b] * s;
                }
                if b == k {
                    x += -dj[a] + vec[a] * s;
                }
                dmi[k][n] = x;
            }
        }
        for k in 0..3 {
            traceless(&mut dmi[k]);
            traceless(&mut dmj[k]);
        }
        out.dd_i[blk] = ddi;
        out.dd_j[blk] = ddj;
        out.dq_i[blk] = dmi;
        out.dq_j[blk] = dmj;
        // the coefficient sets: moments around R_j, shifted to R_i
        for (o, m) in [(NV, &mut *mj), (2 * NV, &mut *mi)] {
            let s = v[o];
            let d = [v[o + 1], v[o + 2], v[o + 3]];
            let mut qj = [v[o + 4], v[o + 5], v[o + 6], v[o + 7], v[o + 8], v[o + 9]];
            let mut qi = [0.0; 6];
            for (n, &(a, b)) in AB.iter().enumerate() {
                qi[n] = qj[n] - vec[a] * d[b] - vec[b] * d[a] + vec[a] * vec[b] * s;
            }
            traceless(&mut qj);
            traceless(&mut qi);
            m.s[blk] = s;
            m.d_j[blk] = d;
            m.q_j[blk] = qj;
            m.d_i[blk] = [d[0] - vec[0] * s, d[1] - vec[1] * s, d[2] - vec[2] * s];
            m.q_i[blk] = qi;
        }
    }
}

/// Packed index pairs (k, l) of the six unique second derivatives.
pub const KL: [(usize, usize); 6] = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)];

/// Packed index of the second derivative (k, l).
#[inline]
pub const fn kl_index(k: usize, l: usize) -> usize {
    let (a, b) = if k <= l { (k, l) } else { (l, k) };
    match (a, b) {
        (0, 0) => 0,
        (0, 1) => 1,
        (0, 2) => 2,
        (1, 1) => 3,
        (1, 2) => 4,
        _ => 5,
    }
}

/// Second derivatives of the shell-pair integrals with respect to the ket center R_i
/// (packed `[kl]` over [`KL`]), with the values and first derivatives in `g`.
#[derive(Clone)]
pub struct MultipoleHess {
    pub g: MultipoleGrad,
    pub d2s: [[f64; 6]; MBLK],
    /// `d2d_i[blk][kl][c]`: dipole component c around R_i.
    pub d2d_i: [[[f64; 3]; 6]; MBLK],
    pub d2d_j: [[[f64; 3]; 6]; MBLK],
    pub d2q_i: [[[f64; 6]; 6]; MBLK],
    pub d2q_j: [[[f64; 6]; 6]; MBLK],
}

impl Default for MultipoleHess {
    fn default() -> Self {
        Self {
            g: MultipoleGrad::default(),
            d2s: [[0.0; 6]; MBLK],
            d2d_i: [[[0.0; 3]; 6]; MBLK],
            d2d_j: [[[0.0; 3]; 6]; MBLK],
            d2q_i: [[[0.0; 6]; 6]; MBLK],
            d2q_j: [[[0.0; 6]; 6]; MBLK],
        }
    }
}

/// Overlap and multipole integrals with their first and second ket-center derivatives.
pub fn multipole_hess(
    cj: &Cgto,
    ci: &Cgto,
    vec: [f64; 3],
    intcut: f64,
    out: &mut MultipoleHess,
    scratch: &mut Scratch,
) {
    let r2 = vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2];
    let (li, lj) = (ci.l, cj.l);
    let (ei, ej) = (cart_exponents(li), cart_exponents(lj));
    let ncj = ncart(lj);
    const NV: usize = 10;
    // moment exponents (mx, my, mz) for s, dx, dy, dz, xx, xy, yy, xz, yz, zz
    const MOM: [[usize; 3]; NV] =
        [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [2, 0, 0], [1, 1, 0], [0, 2, 0], [1, 0, 1], [0, 1, 1], [0, 0, 2]];
    let ncart_pair = ncj * ncart(li);
    let val = &mut scratch.cart10[..ncart_pair];
    val.iter_mut().for_each(|c| *c = [0.0; NV]);
    let der = &mut scratch.cart30[..ncart_pair];
    der.iter_mut().for_each(|c| *c = [0.0; 3 * NV]);
    let der2 = &mut scratch.cart60[..ncart_pair];
    der2.iter_mut().for_each(|c| *c = [0.0; 6 * NV]);
    let mut t = [[[0.0; MT]; MT]; 3];
    for ip in 0..ci.nprim {
        let ai = ci.alpha[ip];
        let (ai2, ai4) = (2.0 * ai, 4.0 * ai * ai);
        for jp in 0..cj.nprim {
            let aj = cj.alpha[jp];
            let Some((oab, pre)) = prefactor(ai, aj, r2, intcut) else { continue };
            let rpi = [-vec[0] * aj * oab, -vec[1] * aj * oab, -vec[2] * aj * oab];
            let rpj = [vec[0] * ai * oab, vec[1] * ai * oab, vec[2] * ai * oab];
            build_1d(li + 2, lj + 2, rpi, rpj, oab, &mut t);
            let cc = ci.coeff[ip] * cj.coeff[jp] * pre;
            for (ic, a) in ei.iter().enumerate() {
                for (jc, b) in ej.iter().enumerate() {
                    let mut v = [[0.0; 3]; 3];
                    let mut g = [[0.0; 3]; 3];
                    let mut h = [[0.0; 3]; 3];
                    for k in 0..3 {
                        let ak = a[k];
                        for m in 0..3 {
                            let bm = b[k] + m;
                            v[k][m] = t[k][ak][bm];
                            g[k][m] = ai2 * t[k][ak + 1][bm];
                            h[k][m] = ai4 * t[k][ak + 2][bm] - ai2 * (2 * ak + 1) as f64 * t[k][ak][bm];
                            if ak > 0 {
                                g[k][m] -= ak as f64 * t[k][ak - 1][bm];
                            }
                            if ak > 1 {
                                h[k][m] += (ak * (ak - 1)) as f64 * t[k][ak - 2][bm];
                            }
                        }
                    }
                    let vc = &mut val[jc + ncj * ic];
                    let dc = &mut der[jc + ncj * ic];
                    let d2c = &mut der2[jc + ncj * ic];
                    for (n, m) in MOM.iter().enumerate() {
                        let (x, y, z) = (v[0][m[0]], v[1][m[1]], v[2][m[2]]);
                        let (gx, gy, gz) = (g[0][m[0]], g[1][m[1]], g[2][m[2]]);
                        vc[n] += cc * x * y * z;
                        dc[3 * n] += cc * gx * y * z;
                        dc[3 * n + 1] += cc * x * gy * z;
                        dc[3 * n + 2] += cc * x * y * gz;
                        d2c[6 * n] += cc * h[0][m[0]] * y * z;
                        d2c[6 * n + 1] += cc * gx * gy * z;
                        d2c[6 * n + 2] += cc * gx * y * gz;
                        d2c[6 * n + 3] += cc * x * h[1][m[1]] * z;
                        d2c[6 * n + 4] += cc * x * gy * gz;
                        d2c[6 * n + 5] += cc * x * y * h[2][m[2]];
                    }
                }
            }
        }
    }
    let sval = &mut scratch.sph10;
    let sder = &mut scratch.sph30;
    let sder2 = &mut scratch.sph60;
    to_spherical(lj, li, val, sval, &mut scratch.tmp10);
    to_spherical(lj, li, der, sder, &mut scratch.tmp30);
    to_spherical(lj, li, der2, sder2, &mut scratch.tmp60);
    const AB: [(usize, usize); 6] = [(0, 0), (0, 1), (1, 1), (0, 2), (1, 2), (2, 2)];
    let delta = |a: usize, b: usize| if a == b { 1.0 } else { 0.0 };
    for blk in 0..nsph(lj) * nsph(li) {
        let v = &sval[blk];
        let dv = &sder[blk];
        let d2v = &sder2[blk];
        let s = v[0];
        let dj = [v[1], v[2], v[3]];
        let mj = [v[4], v[5], v[6], v[7], v[8], v[9]];
        let ds = [dv[0], dv[1], dv[2]];
        // first derivatives (as in multipole_grad)
        let mut ddj = [[0.0; 3]; 3];
        let mut dmj = [[0.0; 6]; 3];
        for k in 0..3 {
            for c in 0..3 {
                ddj[k][c] = dv[3 * (1 + c) + k];
            }
            for c in 0..6 {
                dmj[k][c] = dv[3 * (4 + c) + k];
            }
        }
        let mut ddi = [[0.0; 3]; 3];
        let mut dmi = [[0.0; 6]; 3];
        for k in 0..3 {
            for c in 0..3 {
                ddi[k][c] = ddj[k][c] - vec[c] * ds[k];
            }
            ddi[k][k] -= s;
            for (n, &(a, b)) in AB.iter().enumerate() {
                let mut x = dmj[k][n] - vec[a] * ddj[k][b] - vec[b] * ddj[k][a] + vec[a] * vec[b] * ds[k];
                if a == k {
                    x += -dj[b] + vec[b] * s;
                }
                if b == k {
                    x += -dj[a] + vec[a] * s;
                }
                dmi[k][n] = x;
            }
        }
        // second derivatives around R_j
        let mut d2s = [0.0; 6];
        let mut d2dj = [[0.0; 3]; 6];
        let mut d2mj = [[0.0; 6]; 6];
        for q in 0..6 {
            d2s[q] = d2v[q];
            for c in 0..3 {
                d2dj[q][c] = d2v[6 * (1 + c) + q];
            }
            for c in 0..6 {
                d2mj[q][c] = d2v[6 * (4 + c) + q];
            }
        }
        // shift to R_i
        let mut d2di = [[0.0; 3]; 6];
        let mut d2mi = [[0.0; 6]; 6];
        for (q, &(k, l)) in KL.iter().enumerate() {
            for c in 0..3 {
                d2di[q][c] = d2dj[q][c] - delta(c, k) * ds[l] - delta(c, l) * ds[k] - vec[c] * d2s[q];
            }
            for (n, &(a, b)) in AB.iter().enumerate() {
                let (dja, djb) = (dj[a], dj[b]);
                let _ = (dja, djb);
                d2mi[q][n] = d2mj[q][n]
                    - delta(a, k) * ddj[l][b]
                    - delta(a, l) * ddj[k][b]
                    - vec[a] * d2dj[q][b]
                    - delta(b, k) * ddj[l][a]
                    - delta(b, l) * ddj[k][a]
                    - vec[b] * d2dj[q][a]
                    + delta(a, k) * delta(b, l) * s
                    + delta(a, k) * vec[b] * ds[l]
                    + delta(a, l) * delta(b, k) * s
                    + vec[a] * delta(b, k) * ds[l]
                    + delta(a, l) * vec[b] * ds[k]
                    + vec[a] * delta(b, l) * ds[k]
                    + vec[a] * vec[b] * d2s[q];
            }
        }
        for k in 0..3 {
            traceless(&mut dmi[k]);
            traceless(&mut dmj[k]);
        }
        for q in 0..6 {
            traceless(&mut d2mi[q]);
            traceless(&mut d2mj[q]);
        }
        let _ = mj;
        out.g.s[blk] = s;
        out.g.ds[blk] = ds;
        out.g.dd_i[blk] = ddi;
        out.g.dd_j[blk] = ddj;
        out.g.dq_i[blk] = dmi;
        out.g.dq_j[blk] = dmj;
        out.d2s[blk] = d2s;
        out.d2d_i[blk] = d2di;
        out.d2d_j[blk] = d2dj;
        out.d2q_i[blk] = d2mi;
        out.d2q_j[blk] = d2mj;
    }
}

/// Shift moments around the ket center R_i to moments around the bra center R_j:
/// `D_j = D_i + vec S`, traceless `Q_j` from `Q_i` and the traceless shift contribution.
#[inline]
pub fn shift_to_bra(vec: [f64; 3], s: f64, di: [f64; 3], qi: [f64; 6]) -> ([f64; 3], [f64; 6]) {
    let dj = [di[0] + vec[0] * s, di[1] + vec[1] * s, di[2] + vec[2] * s];
    let mut qj = [
        2.0 * vec[0] * di[0] + vec[0] * vec[0] * s,
        vec[0] * di[1] + vec[1] * di[0] + vec[0] * vec[1] * s,
        2.0 * vec[1] * di[1] + vec[1] * vec[1] * s,
        vec[0] * di[2] + vec[2] * di[0] + vec[0] * vec[2] * s,
        vec[1] * di[2] + vec[2] * di[1] + vec[1] * vec[2] * s,
        2.0 * vec[2] * di[2] + vec[2] * vec[2] * s,
    ];
    traceless(&mut qj);
    for k in 0..6 {
        qj[k] += qi[k];
    }
    (dj, qj)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::{slater_to_gauss, StoExpansion};

    fn shells() -> Vec<Cgto> {
        vec![
            slater_to_gauss(4, 2, 0, 1.3, true, StoExpansion::Gfn2).unwrap(),
            slater_to_gauss(4, 2, 1, 1.1, true, StoExpansion::Gfn2).unwrap(),
            slater_to_gauss(3, 3, 2, 0.9, true, StoExpansion::Gfn2).unwrap(),
        ]
    }

    #[test]
    fn self_overlap_is_identity() {
        for c in shells() {
            let mut s = [0.0; MBLK];
            overlap(&c, &c, [0.0; 3], 25.0, &mut s);
            let n = nsph(c.l);
            for i in 0..n {
                for j in 0..n {
                    let expect = if i == j { 1.0 } else { 0.0 };
                    assert!((s[j + n * i] - expect).abs() < 1e-9, "l={} {} {} {}", c.l, i, j, s[j + n * i]);
                }
            }
        }
    }

    #[test]
    fn coefficient_kernels_equal_the_separate_ones() {
        let mut sh = shells();
        sh.push(slater_to_gauss(4, 4, 3, 1.2, true, StoExpansion::Gfn2).unwrap());
        // coefficient sets with the same exponents
        let dsh: Vec<Cgto> = sh
            .iter()
            .map(|c| {
                let mut d = *c;
                for (k, x) in d.coeff.iter_mut().enumerate() {
                    *x *= 0.3 + 0.1 * k as f64;
                }
                d
            })
            .collect();
        let mut scr = Scratch::default();
        let mut mg = MultipoleGrad::default();
        let mut mg2 = MultipoleGrad::default();
        let (mut mj, mut mi) = (Moments::default(), Moments::default());
        for (a, da) in sh.iter().zip(&dsh) {
            for (b, db) in sh.iter().zip(&dsh) {
                for vec in [[0.3, -0.7, 1.1], [2.5, 0.1, -3.0]] {
                    let nb = nsph(a.l) * nsph(b.l);
                    multipole_grad(a, b, vec, 25.0, &mut mg2, &mut scr);
                    // overlap version: bit for bit
                    let (mut s, mut ds, mut sj, mut si) = ([0.0; MBLK], [[0.0; 3]; MBLK], [0.0; MBLK], [0.0; MBLK]);
                    overlap_grad_coeff(a, b, Some(da), Some(db), vec, 25.0, &mut s, &mut ds, &mut sj, &mut si);
                    let (mut sj0, mut si0) = ([0.0; MBLK], [0.0; MBLK]);
                    overlap(da, b, vec, 25.0, &mut sj0);
                    overlap(a, db, vec, 25.0, &mut si0);
                    assert_eq!(
                        (&s[..nb], &ds[..nb], &sj[..nb], &si[..nb]),
                        (&mg2.s[..nb], &mg2.ds[..nb], &sj0[..nb], &si0[..nb])
                    );
                    // multipole version: the base part bit for bit, the moments to rounding
                    multipole_grad_coeff(a, b, da, db, vec, 25.0, &mut mg, &mut mj, &mut mi, &mut scr);
                    assert_eq!(
                        (&mg.s[..nb], &mg.ds[..nb], &mg.dd_i[..nb], &mg.dq_j[..nb]),
                        (&mg2.s[..nb], &mg2.ds[..nb], &mg2.dd_i[..nb], &mg2.dq_j[..nb])
                    );
                    assert_eq!((&mg.dd_j[..nb], &mg.dq_i[..nb]), (&mg2.dd_j[..nb], &mg2.dq_i[..nb]));
                    let mvec = [-vec[0], -vec[1], -vec[2]];
                    let (mut s1, mut d1, mut q1) = ([0.0; MBLK], [[0.0; 3]; MBLK], [[0.0; 6]; MBLK]);
                    let close = |x: f64, y: f64| (x - y).abs() < 1e-13 * (1.0 + y.abs());
                    // ket-coefficient set around the ket centre, and around the bra centre (transposed call)
                    multipole(a, db, vec, 25.0, &mut s1, &mut d1, &mut q1, &mut scr);
                    for k in 0..nb {
                        assert!(
                            close(mi.s[k], s1[k])
                                && (0..3).all(|x| close(mi.d_i[k][x], d1[k][x]))
                                && (0..6).all(|c| close(mi.q_i[k][c], q1[k][c]))
                        );
                    }
                    multipole(db, a, mvec, 25.0, &mut s1, &mut d1, &mut q1, &mut scr);
                    let (na, nbb) = (nsph(a.l), nsph(b.l));
                    for ja in 0..na {
                        for ib in 0..nbb {
                            let (k, kr) = (ja + na * ib, ib + nbb * ja);
                            assert!(
                                (0..3).all(|x| close(mi.d_j[k][x], d1[kr][x]))
                                    && (0..6).all(|c| close(mi.q_j[k][c], q1[kr][c]))
                            );
                        }
                    }
                    // bra-coefficient set
                    multipole(da, b, vec, 25.0, &mut s1, &mut d1, &mut q1, &mut scr);
                    for k in 0..nb {
                        assert!(
                            close(mj.s[k], s1[k])
                                && (0..3).all(|x| close(mj.d_i[k][x], d1[k][x]))
                                && (0..6).all(|c| close(mj.q_i[k][c], q1[k][c]))
                        );
                    }
                }
            }
        }
    }

    /// The screening is exact: around the screening distance of every shell pair, a pair that
    /// [`pair_screened`] skips has all-zero overlap, multipole and gradient blocks.
    #[test]
    fn screened_pairs_have_zero_blocks() {
        let mut sh = shells();
        sh.push(slater_to_gauss(4, 4, 3, 1.2, true, StoExpansion::Gfn2).unwrap());
        sh.push(slater_to_gauss(2, 1, 0, 2.6, true, StoExpansion::Gfn2).unwrap());
        let mut scr = Scratch::default();
        let mut mg = MultipoleGrad::default();
        let (mut s, mut d, mut q) = ([0.0; MBLK], [[0.0; 3]; MBLK], [[0.0; 6]; MBLK]);
        let dir = {
            let v: [f64; 3] = [0.3, -0.5, 0.81];
            let n = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            v.map(|x| x / n)
        };
        for intcut in [5.0, 25.0] {
            let (mut screened, mut kept) = (0, 0);
            for a in &sh {
                for b in &sh {
                    let (ea, eb) = (a.min_alpha(), b.min_alpha());
                    // the distance where ab/(a+b) r^2 reaches the cutoff
                    let rt = (intcut * (ea + eb) / (ea * eb)).sqrt();
                    for f in [0.99, 1.0, 1.0 + 1e-12, 1.0 + 1e-10, 1.0 + 1e-9, 1.0 + 1e-6, 1.01, 1.5] {
                        let r = rt * f;
                        let vec = dir.map(|x| x * r);
                        let r2 = vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2];
                        if !pair_screened(ea, eb, r2, intcut) {
                            kept += 1;
                            continue;
                        }
                        screened += 1;
                        let nb = nsph(a.l) * nsph(b.l);
                        multipole(a, b, vec, intcut, &mut s, &mut d, &mut q, &mut scr);
                        multipole_grad(a, b, vec, intcut, &mut mg, &mut scr);
                        let zero = s[..nb].iter().all(|x| *x == 0.0)
                            && d[..nb].iter().flatten().all(|x| *x == 0.0)
                            && q[..nb].iter().flatten().all(|x| *x == 0.0)
                            && mg.s[..nb].iter().chain(mg.ds[..nb].iter().flatten()).all(|x| *x == 0.0)
                            && mg.dd_i[..nb].iter().chain(&mg.dd_j[..nb]).flatten().flatten().all(|x| *x == 0.0)
                            && mg.dq_i[..nb].iter().chain(&mg.dq_j[..nb]).flatten().flatten().all(|x| *x == 0.0);
                        assert!(zero, "l = {}, {} at {f} x the screening distance", a.l, b.l);
                    }
                }
            }
            assert!(screened > 0 && kept > 0, "{screened} screened, {kept} kept");
        }
    }

    #[test]
    fn hessian_matches_finite_difference_of_gradient() {
        let sh = shells();
        let vec0 = [0.3, -0.7, 1.1];
        let h = 1e-5;
        let mut worst: f64 = 0.0;
        for ci in &sh {
            for cj in &sh {
                let mut hs = Box::<MultipoleHess>::default();
                let mut scr = Scratch::default();
                multipole_hess(cj, ci, vec0, 25.0, &mut hs, &mut scr);
                let n = nsph(ci.l) * nsph(cj.l);
                let mut g0 = MultipoleGrad::default();
                multipole_grad(cj, ci, vec0, 25.0, &mut g0, &mut scr);
                for b in 0..n {
                    assert!((g0.s[b] - hs.g.s[b]).abs() < 1e-14);
                    for k in 0..3 {
                        assert!((g0.ds[b][k] - hs.g.ds[b][k]).abs() < 1e-14);
                        for c in 0..3 {
                            assert!((g0.dd_i[b][k][c] - hs.g.dd_i[b][k][c]).abs() < 1e-13);
                            assert!((g0.dd_j[b][k][c] - hs.g.dd_j[b][k][c]).abs() < 1e-13);
                        }
                        for c in 0..6 {
                            assert!((g0.dq_i[b][k][c] - hs.g.dq_i[b][k][c]).abs() < 1e-13);
                            assert!((g0.dq_j[b][k][c] - hs.g.dq_j[b][k][c]).abs() < 1e-13);
                        }
                    }
                }
                for l in 0..3 {
                    let mut vp = vec0;
                    vp[l] += h;
                    let mut vm = vec0;
                    vm[l] -= h;
                    let (mut gp, mut gm) = (MultipoleGrad::default(), MultipoleGrad::default());
                    multipole_grad(cj, ci, vp, 25.0, &mut gp, &mut scr);
                    multipole_grad(cj, ci, vm, 25.0, &mut gm, &mut scr);
                    for b in 0..n {
                        for k in 0..3 {
                            let q = kl_index(k, l);
                            let fd = (gp.ds[b][k] - gm.ds[b][k]) / (2.0 * h);
                            worst = worst.max((fd - hs.d2s[b][q]).abs());
                            for c in 0..3 {
                                let fd = (gp.dd_i[b][k][c] - gm.dd_i[b][k][c]) / (2.0 * h);
                                worst = worst.max((fd - hs.d2d_i[b][q][c]).abs());
                                let fd = (gp.dd_j[b][k][c] - gm.dd_j[b][k][c]) / (2.0 * h);
                                worst = worst.max((fd - hs.d2d_j[b][q][c]).abs());
                            }
                            for c in 0..6 {
                                let fd = (gp.dq_i[b][k][c] - gm.dq_i[b][k][c]) / (2.0 * h);
                                worst = worst.max((fd - hs.d2q_i[b][q][c]).abs());
                                let fd = (gp.dq_j[b][k][c] - gm.dq_j[b][k][c]) / (2.0 * h);
                                worst = worst.max((fd - hs.d2q_j[b][q][c]).abs());
                            }
                        }
                    }
                }
            }
        }
        assert!(worst < 1e-8, "integral second derivatives: max error {worst:.3e}");
    }

    #[test]
    fn gradient_matches_finite_difference() {
        let sh = shells();
        let vec0 = [0.3, -0.7, 1.1];
        let h = 1e-5;
        for ci in &sh {
            for cj in &sh {
                let mut g = MultipoleGrad::default();
                let mut scr = Scratch::default();
                multipole_grad(cj, ci, vec0, 25.0, &mut g, &mut scr);
                let n = nsph(ci.l) * nsph(cj.l);
                for k in 0..3 {
                    let mut vp = vec0;
                    vp[k] += h;
                    let mut vm = vec0;
                    vm[k] -= h;
                    let (mut sp, mut dp, mut qp) = ([0.0; MBLK], [[0.0; 3]; MBLK], [[0.0; 6]; MBLK]);
                    let (mut sm, mut dm, mut qm) = ([0.0; MBLK], [[0.0; 3]; MBLK], [[0.0; 6]; MBLK]);
                    multipole(cj, ci, vp, 25.0, &mut sp, &mut dp, &mut qp, &mut scr);
                    multipole(cj, ci, vm, 25.0, &mut sm, &mut dm, &mut qm, &mut scr);
                    for b in 0..n {
                        let fd = (sp[b] - sm[b]) / (2.0 * h);
                        assert!((fd - g.ds[b][k]).abs() < 1e-7, "ds {fd} {}", g.ds[b][k]);
                        for c in 0..3 {
                            let fd = (dp[b][c] - dm[b][c]) / (2.0 * h);
                            assert!((fd - g.dd_i[b][k][c]).abs() < 1e-7, "dd_i {fd} {}", g.dd_i[b][k][c]);
                            // moments around R_j: D_j = D_i + vec S
                            let fdj = ((dp[b][c] + vp[c] * sp[b]) - (dm[b][c] + vm[c] * sm[b])) / (2.0 * h);
                            assert!((fdj - g.dd_j[b][k][c]).abs() < 1e-7, "dd_j {fdj} {}", g.dd_j[b][k][c]);
                        }
                        for c in 0..6 {
                            let fd = (qp[b][c] - qm[b][c]) / (2.0 * h);
                            assert!((fd - g.dq_i[b][k][c]).abs() < 1e-7, "dq_i {fd} {}", g.dq_i[b][k][c]);
                            let (_, qjp) = shift_to_bra(vp, sp[b], dp[b], qp[b]);
                            let (_, qjm) = shift_to_bra(vm, sm[b], dm[b], qm[b]);
                            let fdj = (qjp[c] - qjm[c]) / (2.0 * h);
                            assert!((fdj - g.dq_j[b][k][c]).abs() < 1e-7, "dq_j {fdj} {}", g.dq_j[b][k][c]);
                        }
                    }
                }
            }
        }
    }
}
