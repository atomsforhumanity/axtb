//! Local energy terms of GFN-FF with xtb's analytic gradients (`gfnff_eg.f90`), generic
//! over the scalar type (`f64` for energies and gradients, `Dual` for Hessians). Every
//! kernel follows the Fortran operation by operation; molecules only (translations are 0).

use crate::geom::{domegadr, dphidr, impsc, omega, valijklff, vlen};
use crate::param::PI;
use crate::real::{cross, dot, scale, sub, Real, V3};

/// Distance damping of bends and torsions (`gfnffdampa/t[_nci]`): (damp, d damp / d r² / r²-ish
/// factor `ddamp` as xtb defines it).
#[inline]
pub fn damp<T: Real>(cutfac: f64, rcovi: f64, rcovj: f64, r2: T) -> (T, T) {
    let rcut = cutfac * (rcovi + rcovj).powi(2);
    let rr = (r2 / rcut).powi(2);
    let d = (rr + 1.0).rdiv(1.0);
    let dd = rr * (-2.0e0 * 2.0) / (r2 * (rr + 1.0).powi(2));
    (d, dd)
}

/// Non-bonded repulsion of a pair (i outer). Returns (energy, gradient on i); j gets minus.
#[inline]
pub fn rep_nonbonded<T: Real>(xi: V3<T>, xj: V3<T>, alphanb: f64, zz: f64) -> (T, V3<T>) {
    let d = sub(xi, xj);
    let r2 = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().powi(2);
    let rab = r2.sqrt();
    let t16 = r2.powf(0.75);
    let t19 = t16 * t16;
    let t8 = t16 * alphanb;
    let t26 = (-t8).exp() * zz;
    let e = t26 / rab;
    let t27 = t26 * (t8 * 1.5e0 + 1.0e0) / t19;
    let r3 = scale(d, t27);
    (e, [-r3[0], -r3[1], -r3[2]])
}

/// Bonded repulsion of a bond (i = blist(2), j = blist(1)). Returns (energy, gradient on i).
#[inline]
pub fn rep_bonded<T: Real>(xi: V3<T>, xj: V3<T>, alpha: f64, repab: f64) -> (T, V3<T>) {
    let d = sub(xi, xj);
    let rab = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    let r2 = rab.powi(2);
    let t16 = r2.powf(0.75e0);
    let t19 = t16 * t16;
    let t26 = (-(t16 * alpha)).exp() * repab;
    let e = t26 / rab;
    let t27 = t26 * (t16 * (1.5e0 * alpha) + 1.0e0) / t19;
    (e, [-(d[0] * t27), -(d[1] * t27), -(d[2] * t27)])
}

/// Parameters of a bond-stretch term.
#[derive(Clone, Copy, Debug)]
pub struct BondPar {
    pub r0i: f64,
    pub cnfi: f64,
    pub r0j: f64,
    pub cnfj: f64,
    pub shift: f64,
    /// ff * scaleF of the bond-length estimate
    pub ff: f64,
    pub scale: f64,
    pub steep: f64,
    pub pref: f64,
}

/// Output of the bond kernel.
pub struct BondOut<T> {
    pub e: T,
    pub gi: V3<T>,
    pub gj: V3<T>,
    /// dE/dCN of atoms i and j (through the CN-dependent equilibrium distance)
    pub dcni: T,
    pub dcnj: T,
    /// dE/d hb_cn(H) of an HB-coupled bond (zz)
    pub dhb: T,
}

/// Bond stretch (`egbond`, and `egbond_hb` with `hb = Some((hb_cn(H), 1 - vbond_scale))`).
/// i = blist(2) (`iat`), j = blist(1) (`jat`).
#[inline]
pub fn bond<T: Real>(xi: V3<T>, xj: V3<T>, cni: T, cnj: T, p: &BondPar, hb: Option<(T, f64)>) -> BondOut<T> {
    // gfnffdrab: rab0 = (ra + rb + shift) * ff * scaleF
    let ra = cni * p.cnfi + p.r0i;
    let rb = cnj * p.cnfj + p.r0j;
    let rij = (ra + rb + p.shift) * p.ff * p.scale;
    let dcn1 = p.cnfi * p.ff * p.scale;
    let dcn2 = p.cnfj * p.ff * p.scale;
    let d = sub(xi, xj);
    let rab = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
    let (t8, t1) = match hb {
        Some((hbcn, t1)) => ((-(hbcn * t1) + 1.0) * p.steep, t1),
        None => (T::cst(p.steep), 0.0),
    };
    let dr = rab - rij;
    let dum = (-(t8 * dr.powi(2))).exp() * p.pref;
    let yy = t8 * 2.0e0 * dr * dum;
    let gi = [-(yy * d[0]) / rab, -(yy * d[1]) / rab, -(yy * d[2]) / rab];
    let gj = [yy * (d[0] / rab), yy * (d[1] / rab), yy * (d[2] / rab)];
    let dhb = if hb.is_some() { dum * p.steep * dr.powi(2) * t1 } else { T::cst(0.0) };
    BondOut { e: dum, gi, gj, dcni: yy * dcn1, dcnj: yy * dcn2, dhb }
}

/// Bending term (`egbend`): centre c, ligands j and k. Returns (energy, [g_c, g_j, g_k]).
#[inline]
pub fn bend<T: Real>(
    xc: V3<T>,
    xj: V3<T>,
    xk: V3<T>,
    c0: f64,
    kijk: f64,
    rcov: [f64; 3],
    atcut: f64,
) -> (T, [V3<T>; 3]) {
    let va = xj;
    let vb = xc;
    let vc = xk;
    let vab = sub(va, vb);
    let vcb = sub(vc, vb);
    let rab2 = vab[0] * vab[0] + vab[1] * vab[1] + vab[2] * vab[2];
    let rcb2 = vcb[0] * vcb[0] + vcb[1] * vcb[1] + vcb[2] * vcb[2];
    let vp = cross(vcb, vab);
    let rp = vlen(vp) + 1.0e-14;
    let mut cosa = impsc(vab, vcb);
    if cosa.val() > 1.0 {
        cosa = T::cst(1.0);
    }
    if cosa.val() < -1.0 {
        cosa = T::cst(-1.0);
    }
    let theta = cosa.acos();
    // rcov = [centre, j, k]
    let (dampij, damp2ij) = damp(atcut, rcov[1], rcov[0], rab2);
    let (dampjk, damp2jk) = damp(atcut, rcov[2], rcov[0], rcb2);
    let dmp = dampij * dampjk;
    let (ea, deddt);
    if PI - c0 < 1.0e-6 {
        let dt = theta - c0;
        ea = dt.powi(2) * kijk;
        deddt = dt * (2.0e0 * kijk);
    } else {
        ea = (cosa - c0.cos()).powi(2) * kijk;
        deddt = theta.sin() * (2.0e0 * kijk) * (cosa.rsub(c0.cos()));
    }
    let e = ea * dmp;
    let rmul1 = -(deddt / (rab2 * rp));
    let deda = scale(cross(vab, vp), rmul1);
    let rmul2 = deddt / (rcb2 * rp);
    let dedc = scale(cross(vcb, vp), rmul2);
    let dedb = [deda[0] + dedc[0], deda[1] + dedc[1], deda[2] + dedc[2]];
    let mut g = [[T::cst(0.0); 3]; 3];
    for c in 0..3 {
        let term1 = ea * damp2ij * dampjk * vab[c];
        let term2 = ea * damp2jk * dampij * vcb[c];
        g[0][c] = -dedb[c] * dmp - term1 - term2;
        g[1][c] = deda[c] * dmp + term1;
        g[2][c] = dedc[c] * dmp + term2;
    }
    (e, g)
}

/// Torsion (`nrot > 0`, atoms (l, i, j, k) of the setup = (a, b, c, d) here) or out-of-plane
/// term (`nrot <= 0`, atoms (centre, n1, n2, n3)) of `egtors`. `rcov` of the four atoms.
#[inline]
pub fn torsion<T: Real>(x: [V3<T>; 4], nrot: i32, phi0: f64, fc: f64, rcov: [f64; 4], atcut: f64) -> (T, [V3<T>; 4]) {
    let [va, vb, vc, vd] = x;
    let mut g = [[T::cst(0.0); 3]; 4];
    if nrot > 0 {
        let rn = nrot as f64;
        let vab = sub(va, vb);
        let vcb = sub(vb, vc);
        let vdc = sub(vc, vd);
        let rij = vab[0] * vab[0] + vab[1] * vab[1] + vab[2] * vab[2];
        let rjk = vcb[0] * vcb[0] + vcb[1] * vcb[1] + vcb[2] * vcb[2];
        let rkl = vdc[0] * vdc[0] + vdc[1] * vdc[1] + vdc[2] * vdc[2];
        let (dampij, damp2ij) = damp(atcut, rcov[0], rcov[1], rij);
        let (dampjk, damp2jk) = damp(atcut, rcov[2], rcov[1], rjk);
        let (dampkl, damp2kl) = damp(atcut, rcov[2], rcov[3], rkl);
        let dmp = dampjk * dampij * dampkl;
        let phi = valijklff(va, vb, vc, vd);
        let [dda, ddb, ddc, ddd] = dphidr(va, vb, vc, vd, phi, true);
        let dphi1 = phi - phi0;
        let c1 = dphi1 * rn + PI;
        let x1cos = c1.cos();
        let x1sin = c1.sin();
        let et = (x1cos + 1.0) * fc;
        let dij = -(x1sin * rn) * fc * dmp;
        for c in 0..3 {
            let term1 = et * damp2ij * dampjk * dampkl * vab[c];
            let term2 = et * damp2jk * dampij * dampkl * vcb[c];
            let term3 = et * damp2kl * dampij * dampjk * vdc[c];
            g[0][c] = dij * dda[c] + term1;
            g[1][c] = dij * ddb[c] - term1 + term2;
            g[2][c] = dij * ddc[c] + term3 - term2;
            g[3][c] = dij * ddd[c] - term3;
        }
        (et * dmp, g)
    } else {
        let vab = sub(vb, va);
        let vcb = sub(vb, vc);
        let vdc = sub(vb, vd);
        let rij = vab[0] * vab[0] + vab[1] * vab[1] + vab[2] * vab[2];
        let rjk = vcb[0] * vcb[0] + vcb[1] * vcb[1] + vcb[2] * vcb[2];
        let rjl = vdc[0] * vdc[0] + vdc[1] * vdc[1] + vdc[2] * vdc[2];
        let (dampij, damp2ij) = damp(atcut, rcov[0], rcov[1], rij);
        let (dampjk, damp2jk) = damp(atcut, rcov[2], rcov[1], rjk);
        let (dampjl, damp2jl) = damp(atcut, rcov[1], rcov[3], rjl);
        let dmp = dampjk * dampij * dampjl;
        let phi = omega(va, vb, vc, vd);
        let [dda, ddb, ddc, ddd] = domegadr(va, vb, vc, vd, phi);
        let (et, dij);
        if nrot == 0 {
            let dphi1 = phi - phi0;
            let c1 = dphi1 + PI;
            let x1cos = c1.cos();
            let x1sin = c1.sin();
            et = (x1cos + 1.0) * fc;
            dij = -x1sin * fc * dmp;
        } else {
            et = (phi.cos() - phi0.cos()).powi(2) * fc;
            dij = phi.sin() * (2.0 * fc) * (phi.cos().rsub(phi0.cos())) * dmp;
        }
        for c in 0..3 {
            let term1 = et * damp2ij * dampjk * dampjl * vab[c];
            let term2 = et * damp2jk * dampij * dampjl * vcb[c];
            let term3 = et * damp2jl * dampij * dampjk * vdc[c];
            g[0][c] = dij * dda[c] - term1;
            g[1][c] = dij * ddb[c] + term1 + term2 + term3;
            g[2][c] = dij * ddc[c] - term2;
            g[3][c] = dij * ddd[c] - term3;
        }
        (et * dmp, g)
    }
}

/// Triple-bond torsion (`sTors_eg`) of C1-C2-C3-C4.
pub fn stors<T: Real>(x: [V3<T>; 4]) -> (T, [V3<T>; 4]) {
    let phi = valijklff(x[0], x[1], x[2], x[3]);
    let dp = dphidr(x[0], x[1], x[2], x[3], phi, false);
    let erefhalf = 3.75e0 * 1.0e-4;
    let e = -((phi * 2.0).cos() * erefhalf) + erefhalf;
    let f = (phi * 2.0).sin() * (erefhalf * 2.0);
    let mut g = [[T::cst(0.0); 3]; 4];
    for a in 0..4 {
        for c in 0..3 {
            g[a][c] = f * dp[a][c];
        }
    }
    (e, g)
}

// ---------------------------------------------------------------------------------------
// Smooth forms of the angle terms for the Hessian. xtb writes the gradients of bends and
// torsions through the angle itself (acos, 1/sin), which is exact but 0/0 at linear bends and
// planar torsions: there the forward-mode derivative of the literal gradient is wrong (and
// it loses digits close to them), although the gradient is smooth. These kernels compute
// the same gradient through the cosine, whose derivatives exist everywhere; they are used
// for the dual-number columns of the Hessian, the literal kernels for energies and gradients.

/// cos of the angle between `a` and `b` and its gradients with respect to `a` and `b`.
#[inline]
fn cos_and_grad<T: Real>(a: V3<T>, b: V3<T>) -> (T, V3<T>, V3<T>) {
    let ra2 = dot(a, a);
    let rb2 = dot(b, b);
    let inv = (ra2 * rb2).sqrt().rdiv(1.0);
    let mut c = dot(a, b) * inv;
    // rounding can put |c| above 1 for (anti)parallel vectors, where dc/dx = 0
    if c.val() > 1.0 {
        c = c + (1.0 - c.val());
    } else if c.val() < -1.0 {
        c = c + (-1.0 - c.val());
    }
    let ga = [b[0] * inv - c * a[0] / ra2, b[1] * inv - c * a[1] / ra2, b[2] * inv - c * a[2] / ra2];
    let gb = [a[0] * inv - c * b[0] / rb2, a[1] * inv - c * b[1] / rb2, a[2] * inv - c * b[2] / rb2];
    (c, ga, gb)
}

/// [`bend`] through cos(theta): the same energy and gradient (for a linear reference angle
/// (theta - pi)^2 = u^2 and u / sin u, u = acos(-cos theta), as series in 1 + cos theta
/// near linearity).
pub fn bend_smooth<T: Real>(
    xc: V3<T>,
    xj: V3<T>,
    xk: V3<T>,
    c0: f64,
    kijk: f64,
    rcov: [f64; 3],
    atcut: f64,
) -> (T, [V3<T>; 3]) {
    let vab = sub(xj, xc);
    let vcb = sub(xk, xc);
    let rab2 = dot(vab, vab);
    let rcb2 = dot(vcb, vcb);
    let (cosa, dcj, dck) = cos_and_grad(vab, vcb);
    let (dampij, damp2ij) = damp(atcut, rcov[1], rcov[0], rab2);
    let (dampjk, damp2jk) = damp(atcut, rcov[2], rcov[0], rcb2);
    let dmp = dampij * dampjk;
    let (ea, dedc);
    if PI - c0 < 1.0e-6 && (c0 - PI).abs() > 1.0e-12 {
        // (theta - c0)^2 with c0 beyond pi: |theta - c0| has a kink at linearity anyway
        let theta = cosa.acos();
        ea = (theta - c0).powi(2) * kijk;
        dedc = (theta - c0) * (-2.0 * kijk) / (cosa * cosa).rsub(1.0).sqrt();
    } else if PI - c0 < 1.0e-6 {
        let s = cosa + 1.0;
        let (u2, f) = if s.val() < 1.0e-3 {
            // acos(1 - s) = sqrt(2 s) P(s)
            let p = (((s * (35.0 / 18432.0) + 5.0 / 896.0) * s + 3.0 / 160.0) * s + 1.0 / 12.0) * s + 1.0;
            (s * 2.0 * p * p, p / (s * -0.5 + 1.0).sqrt())
        } else {
            let u = (cosa * -1.0).acos();
            (u * u, u / (cosa * cosa).rsub(1.0).sqrt())
        };
        ea = u2 * kijk;
        dedc = f * (2.0 * kijk);
    } else {
        ea = (cosa - c0.cos()).powi(2) * kijk;
        dedc = (cosa - c0.cos()) * (2.0 * kijk);
    }
    let e = ea * dmp;
    let mut g = [[T::cst(0.0); 3]; 3];
    for c in 0..3 {
        let dej = dedc * dcj[c];
        let dek = dedc * dck[c];
        let term1 = ea * damp2ij * dampjk * vab[c];
        let term2 = ea * damp2jk * dampij * vcb[c];
        g[0][c] = -((dej + dek) * dmp) - term1 - term2;
        g[1][c] = dej * dmp + term1;
        g[2][c] = dek * dmp + term2;
    }
    (e, g)
}

/// Gradient of cos(phi), phi the dihedral angle a-b-c-d (as `valijklff`), w.r.t. a, b, c, d.
#[inline]
fn cos_dihedral_and_grad<T: Real>(va: V3<T>, vb: V3<T>, vc: V3<T>, vd: V3<T>) -> (T, [V3<T>; 4]) {
    let ra = sub(vb, va);
    let rb = sub(vc, vb);
    let rc = sub(vd, vc);
    let na = cross(ra, rb);
    let nb = cross(rb, rc);
    let (cphi, dna, dnb) = cos_and_grad(na, nb);
    let dra = cross(rb, dna);
    let x = cross(dna, ra);
    let y = cross(rc, dnb);
    let drb = [x[0] + y[0], x[1] + y[1], x[2] + y[2]];
    let drc = cross(dnb, rb);
    let neg = |v: V3<T>| [v[0] * -1.0, v[1] * -1.0, v[2] * -1.0];
    (cphi, [neg(dra), sub(dra, drb), sub(drb, drc), drc])
}

/// [`torsion`] through cos(phi) where the term is an even function of the dihedral
/// (sin(n phi0) = 0, every planar or staggered reference): fc (1 - cos(n phi0) T_n(cos phi))
/// with the Chebyshev polynomial T_n. Other terms (and out-of-plane terms, smooth at planarity)
/// use the literal kernel.
pub fn torsion_smooth<T: Real>(
    x: [V3<T>; 4],
    nrot: i32,
    phi0: f64,
    fc: f64,
    rcov: [f64; 4],
    atcut: f64,
) -> (T, [V3<T>; 4]) {
    let rn = nrot as f64;
    if nrot <= 0 || (rn * phi0).sin().abs() > 1.0e-10 {
        return torsion(x, nrot, phi0, fc, rcov, atcut);
    }
    let [va, vb, vc, vd] = x;
    let vab = sub(va, vb);
    let vcb = sub(vb, vc);
    let vdc = sub(vc, vd);
    let rij = dot(vab, vab);
    let rjk = dot(vcb, vcb);
    let rkl = dot(vdc, vdc);
    let (dampij, damp2ij) = damp(atcut, rcov[0], rcov[1], rij);
    let (dampjk, damp2jk) = damp(atcut, rcov[2], rcov[1], rjk);
    let (dampkl, damp2kl) = damp(atcut, rcov[2], rcov[3], rkl);
    let dmp = dampjk * dampij * dampkl;
    let (cphi, dc) = cos_dihedral_and_grad(va, vb, vc, vd);
    // T_n(c) and U_{n-1}(c) (dT_n/dc = n U_{n-1})
    let (mut t_prev, mut t) = (T::cst(1.0), cphi);
    let (mut u_prev, mut u) = (T::cst(0.0), T::cst(1.0));
    for _ in 1..nrot {
        let tn = cphi * 2.0 * t - t_prev;
        t_prev = t;
        t = tn;
        let un = cphi * 2.0 * u - u_prev;
        u_prev = u;
        u = un;
    }
    let cn0 = (rn * phi0).cos();
    let et = (t * cn0).rsub(1.0) * fc;
    let dedc = u * (-(fc * cn0 * rn)) * dmp;
    let mut g = [[T::cst(0.0); 3]; 4];
    for c in 0..3 {
        let term1 = et * damp2ij * dampjk * dampkl * vab[c];
        let term2 = et * damp2jk * dampij * dampkl * vcb[c];
        let term3 = et * damp2kl * dampij * dampjk * vdc[c];
        g[0][c] = dedc * dc[0][c] + term1;
        g[1][c] = dedc * dc[1][c] - term1 + term2;
        g[2][c] = dedc * dc[2][c] + term3 - term2;
        g[3][c] = dedc * dc[3][c] - term3;
    }
    (et * dmp, g)
}

/// [`stors`] through cos(phi): erefhalf (1 - cos 2 phi) = 2 erefhalf (1 - cos^2 phi).
pub fn stors_smooth<T: Real>(x: [V3<T>; 4]) -> (T, [V3<T>; 4]) {
    let erefhalf = 3.75e0 * 1.0e-4;
    let (cphi, dc) = cos_dihedral_and_grad(x[0], x[1], x[2], x[3]);
    let e = (cphi * cphi).rsub(1.0) * (2.0 * erefhalf);
    let f = cphi * (-4.0 * erefhalf);
    let mut g = [[T::cst(0.0); 3]; 4];
    for a in 0..4 {
        for c in 0..3 {
            g[a][c] = f * dc[a][c];
        }
    }
    (e, g)
}

/// Bonded three-body (ATM) term (`batmgfnff_eg`); `c9` = charge factor times the three
/// zb3atm. Returns (energy, [g_i, g_j, g_k]).
pub fn batm<T: Real>(xi: V3<T>, xj: V3<T>, xk: V3<T>, c9: f64) -> (T, [V3<T>; 3]) {
    let dij = sub(xi, xj);
    let dik = sub(xi, xk);
    let djk = sub(xj, xk);
    let r2ij = dij[0] * dij[0] + dij[1] * dij[1] + dij[2] * dij[2];
    let r2ik = dik[0] * dik[0] + dik[1] * dik[1] + dik[2] * dik[2];
    let r2jk = djk[0] * djk[0] + djk[1] * djk[1] + djk[2] * djk[2];
    let sr2ij = r2ij.sqrt();
    let sr2ik = r2ik.sqrt();
    let sr2jk = r2jk.sqrt();
    let invsr2ij = sr2ij.rdiv(1.0);
    let invsr2ik = sr2ik.rdiv(1.0);
    let invsr2jk = sr2jk.rdiv(1.0);
    let mijk = -r2ij + r2jk + r2ik;
    let imjk = r2ij - r2jk + r2ik;
    let ijmk = r2ij + r2jk - r2ik;
    let rijk3 = r2ij * r2jk * r2ik;
    let rav3 = rijk3 * sr2ij * sr2jk * sr2ik;
    let ang = ijmk * 0.375 * imjk * mijk / rijk3;
    let angr9 = (ang + 1.0) / rav3;
    let e = angr9 * c9;
    let dang_of = |a: T, b: T, c: T, sra: T| -> T {
        // -0.375 (a^3 + a^2 (b + c) + a (3 b^2 + 2 b c + 3 c^2) - 5 (b - c)^2 (b + c)) / (sra rijk3 rav3)
        -((a.powi(3) + a.powi(2) * (b + c) + a * (b.powi(2) * 3.0 + b * 2.0 * c + c.powi(2) * 3.0)
            - (b - c).powi(2) * 5.0 * (b + c))
            * 0.375)
            / (sra * rijk3 * rav3)
    };
    let drij = -dang_of(r2ij, r2jk, r2ik, sr2ij) * c9;
    let drjk = -dang_of(r2jk, r2ik, r2ij, sr2jk) * c9;
    let drik = -dang_of(r2ik, r2jk, r2ij, sr2ik) * c9;
    let rij = sub(xj, xi);
    let rik = sub(xk, xi);
    let rjk = sub(xk, xj);
    let mut g = [[T::cst(0.0); 3]; 3];
    for c in 0..3 {
        g[0][c] = drij * rij[c] * invsr2ij;
        g[0][c] += drik * rik[c] * invsr2ik;
        g[1][c] = drjk * rjk[c] * invsr2jk;
        g[1][c] -= drij * rij[c] * invsr2ij;
        g[2][c] = -drik * rik[c] * invsr2ik;
        g[2][c] -= drjk * rjk[c] * invsr2jk;
    }
    (e, g)
}

/// Charge factor of the bonded ATM term.
pub fn batm_c9(qi: f64, qj: f64, qk: f64, zi: f64, zj: f64, zk: f64) -> f64 {
    let fqq = 3.0;
    let f = |q: f64| (1.0 - fqq * q).clamp(-4.0, 4.0);
    let ff = f(qi) * f(qj) * f(qk);
    ff * zi * zj * zk
}

/// Dispersion pair (`d3_gradient`): energy contribution (both atoms), gradient on i and
/// dE/dCN_i, dE/dCN_j. `c6`, `dc6i`, `dc6j` from the reference weights.
#[inline]
pub fn disp_pair<T: Real>(
    xi: V3<T>,
    xj: V3<T>,
    c6: T,
    dc6i: T,
    dc6j: T,
    r4r2ij: f64,
    r0: f64,
    zeta: f64,
    dispscale: f64,
) -> (T, V3<T>, T, T) {
    let rij = sub(xi, xj);
    let r2 = rij[0] * rij[0] + rij[1] * rij[1] + rij[2] * rij[2];
    let t6 = (r2.powi(3) + r0.powi(3)).rdiv(1.0);
    let t8 = (r2.powi(4) + r0.powi(4)).rdiv(1.0);
    let d6 = -(r2.powi(2) * 6.0 * t6.powi(2));
    let d8 = -(r2.powi(3) * 8.0 * t8.powi(2));
    let disp = (t6 + t8 * (2.0 * r4r2ij)) * zeta * dispscale;
    let ddisp = (d6 + d8 * (2.0 * r4r2ij)) * zeta * dispscale;
    let de = -c6 * disp * 0.5;
    let dg = scale(rij, -c6 * ddisp);
    (de + de, dg, -(dc6i * disp), -(dc6j * disp))
}

/// HB/XB charge scaling `exp(s q)/(exp(s q) + f)`.
#[inline]
pub fn qscale(st: f64, sf: f64, q: f64) -> f64 {
    let ex1 = (st * q).exp();
    let ex2 = ex1 + sf;
    ex1 / ex2
}

/// HB parameters shared by the kernels.
#[derive(Clone, Copy, Debug)]
pub struct HbPar {
    pub hbacut: f64,
    pub hbscut: f64,
    pub hbalp: f64,
    pub hblongcut: f64,
    pub hbabmix: f64,
    pub hbnbcut: f64,
    pub xhaci_globabh: f64,
    pub xhaci_coh: f64,
    pub tors_hb: f64,
    pub bend_hb: f64,
}

/// Case 1 HB (`abhgfnff_eg1`): A...H...B with H not bonded to A. `ca`, `cb` = (basicity,
/// acidity) of A and B, `qa`, `qb`, `qh` the charge scalings, `radab` = rad(A) + rad(B).
/// Returns (energy, [g_A, g_B, g_H]); `None` if out of line (energy 0).
pub fn hb1<T: Real>(
    xa: V3<T>,
    xb: V3<T>,
    xh: V3<T>,
    ca: [f64; 2],
    cb: [f64; 2],
    qa: f64,
    qb: f64,
    qh: f64,
    radab: f64,
    p: &HbPar,
) -> Option<(T, [V3<T>; 3])> {
    let rab = crate::real::norm2(sub(xa, xb));
    let rab2 = rab.powi(2);
    let rah = crate::real::norm2(sub(xa, xh));
    let rah2 = rah.powi(2);
    let rbh = crate::real::norm2(sub(xb, xh));
    let rbh2 = rbh.powi(2);
    let rahprbh = rah + rbh + 1.0e-12;
    let expo = (rahprbh / rab - 1.0e0) * (p.hbacut / radab);
    if expo.val() > 15.0 {
        return None;
    }
    let ratio2 = expo.exp();
    let outl = (ratio2 + 1.0e0).rdiv(2.0e0);
    let ratio1 = (rab2 / p.hblongcut).powf(p.hbalp);
    let dampl = (ratio1 + 1.0e0).rdiv(1.0e0);
    let shortcut = p.hbscut * radab;
    let ratio3 = (rab2.rdiv(shortcut)).powf(p.hbalp);
    let damps = (ratio3 + 1.0e0).rdiv(1.0e0);
    let dmp = damps * dampl;
    let rdamp = dmp / rab2 / rab;
    let rah4 = rah2 * rah2;
    let rbh4 = rbh2 * rbh2;
    let denom = (rah4 + rbh4).rdiv(1.0e0);
    let caa = qa * ca[0];
    let cbb = qb * cb[0];
    let qhoutl = outl * qh;
    let bas = (rah4 * caa + rbh4 * cbb) * denom;
    let aci = (rah4 * cb[1] + rbh4 * ca[1]) * denom;
    let rterm = -(aci * rdamp * qhoutl);
    let energy = bas * rterm;
    let drah = sub(xa, xh);
    let drbh = sub(xb, xh);
    let drab = sub(xa, xb);
    let aterm = -(aci * bas * rdamp * qh);
    let sterm = -(rdamp * bas * qhoutl);
    let dterm = -(aci * bas * qhoutl);
    let tmp = denom * denom * 4.0e0;
    let dd24a = rah2 * rbh4 * tmp;
    let dd24b = rbh2 * rah4 * tmp;
    let mut ga = [T::cst(0.0); 3];
    let mut gb = [T::cst(0.0); 3];
    let mut gh = [T::cst(0.0); 3];
    let gi = dd24a * (caa - cbb) * rterm;
    for c in 0..3 {
        ga[c] = gi * drah[c];
    }
    let gi = dd24b * (cbb - caa) * rterm;
    for c in 0..3 {
        gb[c] = gi * drbh[c];
        gh[c] = -ga[c] - gb[c];
    }
    let gi = dd24a * (cb[1] - ca[1]);
    let dga: V3<T> = [gi * drah[0] * sterm, gi * drah[1] * sterm, gi * drah[2] * sterm];
    let gi = dd24b * (ca[1] - cb[1]);
    let dgb: V3<T> = [gi * drbh[0] * sterm, gi * drbh[1] * sterm, gi * drbh[2] * sterm];
    for c in 0..3 {
        ga[c] += dga[c];
        gb[c] += dgb[c];
        gh[c] += -dga[c] - dgb[c];
    }
    let gi = rdamp
        * (-(ratio1 * (2.0e0 * p.hbalp) / (ratio1 + 1.0)) + ratio3 * (2.0e0 * p.hbalp) / (ratio3 + 1.0) - 3.0e0)
        / rab2;
    for c in 0..3 {
        let dg = gi * drab[c] * dterm;
        ga[c] += dg;
        gb[c] -= dg;
    }
    // expo / (rahprbh - rab) = (hbacut / radab) / rab: 0/0 at a linear A-H...B, so the dual
    // path uses the right-hand side
    let gi = if T::DUAL {
        aterm * 2.0e0 * ratio2 * rahprbh / (ratio2 + 1.0).powi(2) * rab.rdiv(p.hbacut / radab) / rab2
    } else {
        aterm * 2.0e0 * ratio2 * expo * rahprbh / (ratio2 + 1.0).powi(2) / (rahprbh - rab) / rab2
    };
    for c in 0..3 {
        let dg = gi * drab[c];
        ga[c] += dg;
        gb[c] -= dg;
    }
    let tmp = if T::DUAL {
        -(aterm * 2.0e0) * ratio2 / (ratio2 + 1.0).powi(2) * rab.rdiv(p.hbacut / radab)
    } else {
        -(aterm * 2.0e0) * ratio2 * expo / (ratio2 + 1.0).powi(2) / (rahprbh - rab)
    };
    for c in 0..3 {
        let dga = drah[c] * tmp / rah;
        let dgb = drbh[c] * tmp / rbh;
        ga[c] += dga;
        gb[c] += dgb;
        gh[c] += -dga - dgb;
    }
    Some((energy, [ga, gb, gh]))
}

/// Out-of-line factors of the B neighbors (`outl_nb`): 2/(1+exp(-expo)) - 1.
#[inline]
fn outl_nb<T: Real>(ranbprbnb: T, rab: T, cut: f64, radab: f64) -> (T, T, T) {
    let expo_nb = (ranbprbnb / rab - 1.0e0) * (cut / radab);
    let ratio2_nb = (-expo_nb).exp().powf(1.0);
    let o = (ratio2_nb + 1.0e0).rdiv(2.0e0) - 1.0e0;
    (o, ratio2_nb, expo_nb)
}

/// Product of all but the `skip`-th element (Fortran `product(outl_nb, mask_nb)`).
fn product_but<T: Real>(v: &[T], skip: usize) -> T {
    let mut p = T::cst(1.0);
    for (i, &x) in v.iter().enumerate() {
        if i != skip {
            p *= x;
        }
    }
    p
}

fn product<T: Real>(v: &[T]) -> T {
    let mut p = T::cst(1.0);
    for &x in v {
        p *= x;
    }
    p
}

/// Case 2 HB (`abhgfnff_eg2new`): A-H...B with the orientation of B's neighbors `xnb`.
/// Returns (energy, gradient on A, B, H, then the neighbors).
pub fn hb2<T: Real>(
    xa: V3<T>,
    xb: V3<T>,
    xh: V3<T>,
    xnb: &[V3<T>],
    b_is_n_single: bool,
    ca: [f64; 2],
    cb: [f64; 2],
    qa: f64,
    qb: f64,
    qh: f64,
    radab: f64,
    p: &HbPar,
) -> Option<(T, Vec<V3<T>>)> {
    let nbb = xnb.len();
    let p_bh = 1.0e0 + p.hbabmix;
    let p_ab = -p.hbabmix;
    let mut dranb = Vec::with_capacity(nbb);
    let mut drbnb = Vec::with_capacity(nbb);
    let mut ranb = Vec::with_capacity(nbb);
    let mut rbnb = Vec::with_capacity(nbb);
    for x in xnb {
        let da = sub(xa, *x);
        let db = sub(xb, *x);
        ranb.push((da[0].powi(2) + da[1].powi(2) + da[2].powi(2)).sqrt());
        rbnb.push((db[0].powi(2) + db[1].powi(2) + db[2].powi(2)).sqrt());
        dranb.push(da);
        drbnb.push(db);
    }
    let rab = crate::real::norm2(sub(xa, xb));
    let rab2 = rab.powi(2);
    let rah = crate::real::norm2(sub(xa, xh));
    let rbh = crate::real::norm2(sub(xb, xh));
    let rbh2 = rbh.powi(2);
    let rahprbh = rah + rbh + 1.0e-12;
    let expo = (rahprbh / rab - 1.0e0) * (p.hbacut / radab);
    if expo.val() > 15.0 {
        return None;
    }
    let ratio2 = expo.exp();
    let outl = (ratio2 + 1.0e0).rdiv(2.0e0);
    let hbnbcut_save = if b_is_n_single { 2.0 } else { p.hbnbcut };
    let mut ranbprbnb = Vec::with_capacity(nbb);
    let mut onb = Vec::with_capacity(nbb);
    let mut r2nb = Vec::with_capacity(nbb);
    let mut enb = Vec::with_capacity(nbb);
    for i in 0..nbb {
        let s = ranb[i] + rbnb[i] + 1.0e-12;
        let (o, r, e) = outl_nb(s, rab, hbnbcut_save, radab);
        ranbprbnb.push(s);
        onb.push(o);
        r2nb.push(r);
        enb.push(e);
    }
    let outl_nb_tot = product(&onb);
    let ratio1 = (rab2 / p.hblongcut).powf(p.hbalp);
    let dampl = (ratio1 + 1.0e0).rdiv(1.0e0);
    let shortcut = p.hbscut * radab;
    let ratio3 = (rab2.rdiv(shortcut)).powf(p.hbalp);
    let damps = (ratio3 + 1.0e0).rdiv(1.0e0);
    let dmp = damps * dampl;
    let ddamp = -(ratio1 * (2.0e0 * p.hbalp) / (ratio1 + 1.0e0)) + ratio3 * (2.0e0 * p.hbalp) / (ratio3 + 1.0e0);
    let rbhdamp = dmp * (rbh2.rdiv(p_bh) / rbh);
    let rabdamp = dmp * (rab2.rdiv(p_ab) / rab);
    let rdamp = rbhdamp + rabdamp;
    let qhoutl = outl * qh * outl_nb_tot;
    let cnst = ca[1] * qa * cb[0] * qb * p.xhaci_globabh;
    let energy = -(rdamp * qhoutl * cnst);
    let drah = sub(xa, xh);
    let drbh = sub(xb, xh);
    let drab = sub(xa, xb);
    let aterm = -(rdamp * qh * outl_nb_tot * cnst);
    let nbterm = -(rdamp * qh * outl * cnst);
    let dterm = -(qhoutl * cnst);
    let mut ga = [T::cst(0.0); 3];
    let mut gb = [T::cst(0.0); 3];
    let mut gh = [T::cst(0.0); 3];
    let gi = ((rabdamp + rbhdamp) * ddamp - rabdamp * 3.0e0) / rab2 * dterm;
    for c in 0..3 {
        let dg = gi * drab[c];
        ga[c] = dg;
        gb[c] = -dg;
    }
    let gi = -(rbhdamp * 3.0e0) / rbh2 * dterm;
    for c in 0..3 {
        let dg = gi * drbh[c];
        gb[c] += dg;
        gh[c] = -dg;
    }
    let tmp1 = if T::DUAL {
        -(aterm * 2.0e0) * ratio2 / (ratio2 + 1.0).powi(2) * rab.rdiv(p.hbacut / radab)
    } else {
        -(aterm * 2.0e0) * ratio2 * expo / (ratio2 + 1.0).powi(2) / (rahprbh - rab)
    };
    let gi = -tmp1 * rahprbh / rab2;
    for c in 0..3 {
        let dg = gi * drab[c];
        ga[c] += dg;
        gb[c] -= dg;
    }
    let gi_a = tmp1 / rah;
    let gi_b = tmp1 / rbh;
    for c in 0..3 {
        let dga = gi_a * drah[c];
        let dgb = gi_b * drbh[c];
        ga[c] += dga;
        gb[c] += dgb;
        gh[c] += -dga - dgb;
    }
    let mut tmp2 = Vec::with_capacity(nbb);
    for i in 0..nbb {
        let t = if T::DUAL {
            nbterm * 2.0e0 * product_but(&onb, i) * r2nb[i] / (r2nb[i] + 1.0).powi(2) * rab.rdiv(p.hbnbcut / radab)
        } else {
            nbterm * 2.0e0 * product_but(&onb, i) * r2nb[i] * enb[i] / (r2nb[i] + 1.0).powi(2) / (ranbprbnb[i] - rab)
        };
        let gi = -t * ranbprbnb[i] / rab2;
        for c in 0..3 {
            let dg = gi * drab[c];
            ga[c] += dg;
            gb[c] -= dg;
        }
        tmp2.push(t);
    }
    let mut gnb = Vec::with_capacity(nbb);
    for i in 0..nbb {
        let gia = tmp2[i] / ranb[i];
        let gib = tmp2[i] / rbnb[i];
        let mut g = [T::cst(0.0); 3];
        for c in 0..3 {
            let dga = gia * dranb[i][c];
            ga[c] += dga;
            let dgb = gib * drbnb[i][c];
            gb[c] += dgb;
            g[c] = -dga - dgb;
        }
        gnb.push(g);
    }
    let mut out = vec![ga, gb, gh];
    out.extend(gnb);
    Some((energy, out))
}

/// Case 2 HB at a pyridine-type N (`abhgfnff_eg2_rnr`): B has two neighbors `xnb` and a
/// lone pair. Returns (energy, gradient on A, B, H, nb1, nb2).
pub fn hb2_rnr<T: Real>(
    xa: V3<T>,
    xb: V3<T>,
    xh: V3<T>,
    xnb: [V3<T>; 2],
    repz_b: f64,
    ca: [f64; 2],
    cb: [f64; 2],
    qa: f64,
    qb: f64,
    qh: f64,
    radab: f64,
    p: &HbPar,
) -> Option<(T, Vec<V3<T>>)> {
    let p_bh = 1.0e0 + p.hbabmix;
    let p_ab = -p.hbabmix;
    let lp_dist = 0.50 - 0.018 * repz_b;
    let hblpcut = 56.0;
    let mut nbb = 2usize;
    let mut dranb = [[T::cst(0.0); 3]; 2];
    let mut drbnb = [[T::cst(0.0); 3]; 2];
    let mut ranb = [T::cst(0.0); 2];
    let mut rbnb = [T::cst(0.0); 2];
    let mut vector = [T::cst(0.0); 3];
    for i in 0..2 {
        dranb[i] = sub(xa, xnb[i]);
        drbnb[i] = sub(xb, xnb[i]);
        ranb[i] = (dranb[i][0].powi(2) + dranb[i][1].powi(2) + dranb[i][2].powi(2)).sqrt();
        rbnb[i] = (drbnb[i][0].powi(2) + drbnb[i][1].powi(2) + drbnb[i][2].powi(2)).sqrt();
        let drnb = sub(xnb[i], xb);
        for c in 0..3 {
            vector[c] += drnb[c];
        }
    }
    let vnorm = crate::real::norm2(vector);
    let lp: V3<T>;
    if vnorm.val() > 1.0e-10 {
        lp = [
            xb[0] - vector[0] / vnorm * lp_dist,
            xb[1] - vector[1] / vnorm * lp_dist,
            xb[2] - vector[2] / vnorm * lp_dist,
        ];
    } else {
        lp = xb;
        nbb = 0;
    }
    let rab = crate::real::norm2(sub(xa, xb));
    let rab2 = rab.powi(2);
    let rah = crate::real::norm2(sub(xa, xh));
    let rbh = crate::real::norm2(sub(xb, xh));
    let rbh2 = rbh.powi(2);
    let rahprbh = rah + rbh + 1.0e-12;
    let expo = (rahprbh / rab - 1.0e0) * (p.hbacut / radab);
    if expo.val() > 15.0 {
        return None;
    }
    let ratio2 = expo.exp();
    let outl = (ratio2 + 1.0e0).rdiv(2.0e0);
    let dblp = sub(xb, lp);
    let rblp = (dblp[0].powi(2) + dblp[1].powi(2) + dblp[2].powi(2)).sqrt();
    let dalp = sub(xa, lp);
    let ralp = (dalp[0].powi(2) + dalp[1].powi(2) + dalp[2].powi(2)).sqrt();
    let ralpprblp = ralp + rblp + 1.0e-12;
    let expo_lp = (ralpprblp / rab - 1.0e0) * (hblpcut / radab);
    let ratio2_lp = expo_lp.exp();
    let outl_lp = (ratio2_lp + 1.0e0).rdiv(2.0e0);
    let mut ranbprbnb = [T::cst(0.0); 2];
    let mut onb = Vec::with_capacity(2);
    let mut r2nb = [T::cst(0.0); 2];
    let mut enb = [T::cst(0.0); 2];
    for i in 0..nbb {
        ranbprbnb[i] = ranb[i] + rbnb[i] + 1.0e-12;
        let (o, r, e) = outl_nb(ranbprbnb[i], rab, p.hbnbcut, radab);
        onb.push(o);
        r2nb[i] = r;
        enb[i] = e;
    }
    // with nbb = 0 xtb still multiplies the two (uninitialized) array elements; for a
    // well-defined result they are taken as 1
    let outl_nb_tot = product(&onb);
    let ratio1 = (rab2 / p.hblongcut).powf(p.hbalp);
    let dampl = (ratio1 + 1.0e0).rdiv(1.0e0);
    let shortcut = p.hbscut * radab;
    let ratio3 = (rab2.rdiv(shortcut)).powf(p.hbalp);
    let damps = (ratio3 + 1.0e0).rdiv(1.0e0);
    let dmp = damps * dampl;
    let ddamp = -(ratio1 * (2.0e0 * p.hbalp) / (ratio1 + 1.0e0)) + ratio3 * (2.0e0 * p.hbalp) / (ratio3 + 1.0e0);
    let rbhdamp = dmp * (rbh2.rdiv(p_bh) / rbh);
    let rabdamp = dmp * (rab2.rdiv(p_ab) / rab);
    let rdamp = rbhdamp + rabdamp;
    let qhoutl = outl * qh * outl_nb_tot * outl_lp;
    let cnst = ca[1] * qa * cb[0] * qb * p.xhaci_globabh;
    let energy = -(rdamp * qhoutl * cnst);
    let drah = sub(xa, xh);
    let drbh = sub(xb, xh);
    let drab = sub(xa, xb);
    let dralp = sub(xa, lp);
    let aterm = -(rdamp * qh * outl_nb_tot * outl_lp * cnst);
    let nbterm = -(rdamp * qh * outl * outl_lp * cnst);
    let lpterm = -(rdamp * qh * outl * outl_nb_tot * cnst);
    let dterm = -(qhoutl * cnst);
    let mut ga = [T::cst(0.0); 3];
    let mut gb = [T::cst(0.0); 3];
    let mut gh = [T::cst(0.0); 3];
    let gi = ((rabdamp + rbhdamp) * ddamp - rabdamp * 3.0e0) / rab2 * dterm;
    for c in 0..3 {
        let dg = gi * drab[c];
        ga[c] = dg;
        gb[c] = -dg;
    }
    let gi = -(rbhdamp * 3.0e0) / rbh2 * dterm;
    for c in 0..3 {
        let dg = gi * drbh[c];
        gb[c] += dg;
        gh[c] = -dg;
    }
    let tmp1 = if T::DUAL {
        -(aterm * 2.0e0) * ratio2 / (ratio2 + 1.0).powi(2) * rab.rdiv(p.hbacut / radab)
    } else {
        -(aterm * 2.0e0) * ratio2 * expo / (ratio2 + 1.0).powi(2) / (rahprbh - rab)
    };
    let gi = -tmp1 * rahprbh / rab2;
    for c in 0..3 {
        let dg = gi * drab[c];
        ga[c] += dg;
        gb[c] -= dg;
    }
    let gi_a = tmp1 / rah;
    let gi_b = tmp1 / rbh;
    for c in 0..3 {
        let dga = gi_a * drah[c];
        let dgb = gi_b * drbh[c];
        ga[c] += dga;
        gb[c] += dgb;
        gh[c] += -dga - dgb;
    }
    // A...LP-B term (xtb keeps only the A and LP parts of the gradient)
    let tmp3 = if T::DUAL {
        -(lpterm * 2.0e0) * ratio2_lp / (ratio2_lp + 1.0).powi(2) * rab.rdiv(hblpcut / radab)
    } else {
        -(lpterm * 2.0e0) * ratio2_lp * expo_lp / (ratio2_lp + 1.0).powi(2) / (ralpprblp - rab)
    };
    let gi = -tmp3 * ralpprblp / rab2;
    for c in 0..3 {
        let dg = gi * drab[c];
        ga[c] += dg;
        gb[c] -= dg;
    }
    let gi = tmp3 / ralp;
    let mut glp = [T::cst(0.0); 3];
    for c in 0..3 {
        let dga = gi * dralp[c];
        ga[c] += dga;
        gb[c] -= dga;
        glp[c] = -dga;
    }
    // d lp / d neighbors (gii), applied to glp
    let mut gnb_lp = [T::cst(0.0); 3];
    let vsq = vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2];
    let v15 = vsq.powf(1.5e0);
    for i in 0..3 {
        // column i of gii: -lp_dist * nbb * (unit_vec/vnorm + vector*vector(i)/|v|^3), unit_vec = -e_i
        for r in 0..3 {
            let unit = if r == i { -1.0 } else { 0.0 };
            let gii = -((vnorm.rdiv(unit) + vector[r] * vector[i] / v15) * (lp_dist * nbb as f64));
            gnb_lp[r] += gii * glp[i];
        }
    }
    let mut tmp2 = [T::cst(0.0); 2];
    for i in 0..nbb {
        let t = if T::DUAL {
            nbterm * 2.0e0 * product_but(&onb, i) * r2nb[i] / (r2nb[i] + 1.0).powi(2) * rab.rdiv(p.hbnbcut / radab)
        } else {
            nbterm * 2.0e0 * product_but(&onb, i) * r2nb[i] * enb[i] / (r2nb[i] + 1.0).powi(2) / (ranbprbnb[i] - rab)
        };
        let gi = -t * ranbprbnb[i] / rab2;
        for c in 0..3 {
            let dg = gi * drab[c];
            ga[c] += dg;
            gb[c] -= dg;
        }
        tmp2[i] = t;
    }
    let mut gnb = [[T::cst(0.0); 3]; 2];
    for i in 0..nbb {
        let gia = tmp2[i] / ranb[i];
        let gib = tmp2[i] / rbnb[i];
        for c in 0..3 {
            let dga = gia * dranb[i][c];
            ga[c] += dga;
            let dgb = gib * drbnb[i][c];
            gb[c] += dgb;
            gnb[i][c] = -dga - dgb;
        }
    }
    let mut out = vec![ga, [gb[0] + gnb_lp[0], gb[1] + gnb_lp[1], gb[2] + gnb_lp[2]], gh];
    for i in 0..2 {
        if i < nbb {
            let f = nbb as f64;
            out.push([gnb[i][0] - gnb_lp[0] / f, gnb[i][1] - gnb_lp[1] / f, gnb[i][2] - gnb_lp[2] / f]);
        } else {
            out.push([T::cst(0.0); 3]);
        }
    }
    Some((energy, out))
}

/// Case 3 HB (`abhgfnff_eg3`): A-H...O=C with the bend H...O=C and the torsions
/// H...O=C-R. `xr` are the other neighbors R of C. Returns (energy, gradient on
/// A, B, H, C, then the R atoms).
pub fn hb3<T: Real>(
    xa: V3<T>,
    xb: V3<T>,
    xh: V3<T>,
    xc: V3<T>,
    xr: &[V3<T>],
    ca: [f64; 2],
    cb: [f64; 2],
    qa: f64,
    qb: f64,
    qh: f64,
    radab: f64,
    p: &HbPar,
) -> Option<(T, Vec<V3<T>>)> {
    let p_bh = 1.0e0 + p.hbabmix;
    let p_ab = -p.hbabmix;
    let ntors = xr.len();
    let dranb = sub(xa, xc);
    let drbnb = sub(xb, xc);
    let ranb = (dranb[0].powi(2) + dranb[1].powi(2) + dranb[2].powi(2)).sqrt();
    let rbnb = (drbnb[0].powi(2) + drbnb[1].powi(2) + drbnb[2].powi(2)).sqrt();
    let rab = crate::real::norm2(sub(xa, xb));
    let rab2 = rab.powi(2);
    let rah = crate::real::norm2(sub(xa, xh));
    let rbh = crate::real::norm2(sub(xb, xh));
    let rbh2 = rbh.powi(2);
    let rahprbh = rah + rbh + 1.0e-12;
    let expo = (rahprbh / rab - 1.0e0) * (p.hbacut / radab);
    if expo.val() > 15.0 {
        return None;
    }
    let ratio2 = expo.exp();
    let outl = (ratio2 + 1.0e0).rdiv(2.0e0);
    let ranbprbnb = ranb + rbnb + 1.0e-12;
    let expo_nb = (ranbprbnb / rab - 1.0e0) * (p.hbnbcut / radab);
    let ratio2_nb = (-expo_nb).exp();
    let outl_nb_tot = (ratio2_nb + 1.0e0).rdiv(2.0e0) - 1.0e0;
    let ratio1 = (rab2 / p.hblongcut).powf(p.hbalp);
    let dampl = (ratio1 + 1.0e0).rdiv(1.0e0);
    let shortcut = p.hbscut * radab;
    let ratio3 = (rab2.rdiv(shortcut)).powi(6);
    let damps = (ratio3 + 1.0e0).rdiv(1.0e0);
    let dmp = damps * dampl;
    let ddamp = -(ratio1 * (2.0e0 * p.hbalp) / (ratio1 + 1.0e0)) + ratio3 * (2.0e0 * p.hbalp) / (ratio3 + 1.0e0);
    let rbhdamp = dmp * (rbh2.rdiv(p_bh) / rbh);
    let rabdamp = dmp * (rab2.rdiv(p_ab) / rab);
    let rdamp = rbhdamp + rabdamp;
    // torsions R-B...C... : tors_nci_mul(R, B, C, H) with phi0 = pi/2, tshift = tors_hb, rn = 2
    let phi0 = PI / 2.0;
    let tshift = p.tors_hb;
    let mut etmp = Vec::with_capacity(ntors);
    let mut g4: Vec<[V3<T>; 4]> = Vec::with_capacity(ntors);
    for r in xr {
        let fc = (1.0e0 - tshift) / 2.0e0;
        if T::DUAL {
            // cos(2 (phi - pi/2) + pi) = cos(2 phi) = 2 cos^2 phi - 1: through cos(phi), smooth at
            // planar R-B...C...H (see `torsion_smooth`)
            let (cphi, dc) = cos_dihedral_and_grad(*r, xb, xc, xh);
            let f = cphi * (4.0 * fc);
            etmp.push(cphi * cphi * (2.0 * fc) + tshift);
            g4.push([scale(dc[0], f), scale(dc[1], f), scale(dc[2], f), scale(dc[3], f)]);
            continue;
        }
        let phi = valijklff(*r, xb, xc, xh);
        let d = dphidr(*r, xb, xc, xh, phi, true);
        let rn = 2.0;
        let dphi1 = phi - phi0;
        let c1 = dphi1 * rn + PI;
        let et = (c1.cos() + 1.0) * fc + tshift;
        let dij = -(c1.sin() * rn) * fc;
        etmp.push(et);
        g4.push([scale(d[0], dij), scale(d[1], dij), scale(d[2], dij), scale(d[3], dij)]);
    }
    let etors = product(&etmp);
    // gtors per atom: R_i, B, C, H
    let mut gt_r = vec![[T::cst(0.0); 3]; ntors];
    let mut gt_b = [T::cst(0.0); 3];
    let mut gt_c = [T::cst(0.0); 3];
    let mut gt_h = [T::cst(0.0); 3];
    for i in 0..ntors {
        let pm = product_but(&etmp, i);
        for c in 0..3 {
            gt_r[i][c] += g4[i][0][c] * pm;
            gt_b[c] += g4[i][1][c] * pm;
            gt_c[c] += g4[i][2][c] * pm;
            gt_h[c] += g4[i][3][c] * pm;
        }
    }
    // bend C=B...H (egbend_nci_mul with j = B, i = C, k = H)
    let r0 = 120.0;
    let bphi0 = r0 * PI / 180.;
    let fc = 1.0e0 - p.bend_hb;
    let (eangl, gb3) = bend_nci_mul(xb, xc, xh, bphi0, fc);
    let qhoutl = outl * qh * outl_nb_tot;
    let cnst = ca[1] * qa * cb[0] * qb * p.xhaci_coh;
    let energy = -(rdamp * qhoutl * eangl * etors * cnst);
    let drah = sub(xa, xh);
    let drbh = sub(xb, xh);
    let drab = sub(xa, xb);
    let aterm = -(rdamp * qh * outl_nb_tot * eangl * etors * cnst);
    let nbterm = -(rdamp * qh * outl * eangl * etors * cnst);
    let dterm = -(qhoutl * eangl * etors * cnst);
    let tterm = -(rdamp * qhoutl * eangl * cnst);
    let bterm = -(rdamp * qhoutl * etors * cnst);
    let mut ga = [T::cst(0.0); 3];
    let mut gb = [T::cst(0.0); 3];
    let mut gh = [T::cst(0.0); 3];
    let gi = ((rabdamp + rbhdamp) * ddamp - rabdamp * 3.0e0) / rab2 * dterm;
    for c in 0..3 {
        let dg = gi * drab[c];
        ga[c] = dg;
        gb[c] = -dg;
    }
    let gi = -(rbhdamp * 3.0e0) / rbh2 * dterm;
    for c in 0..3 {
        let dg = gi * drbh[c];
        gb[c] += dg;
        gh[c] = -dg;
    }
    let tmp1 = if T::DUAL {
        -(aterm * 2.0e0) * ratio2 / (ratio2 + 1.0).powi(2) * rab.rdiv(p.hbacut / radab)
    } else {
        -(aterm * 2.0e0) * ratio2 * expo / (ratio2 + 1.0).powi(2) / (rahprbh - rab)
    };
    let gi = -tmp1 * rahprbh / rab2;
    for c in 0..3 {
        let dg = gi * drab[c];
        ga[c] += dg;
        gb[c] -= dg;
    }
    let gi_a = tmp1 / rah;
    let gi_b = tmp1 / rbh;
    for c in 0..3 {
        let dga = gi_a * drah[c];
        let dgb = gi_b * drbh[c];
        ga[c] += dga;
        gb[c] += dgb;
        gh[c] += -dga - dgb;
    }
    let tmp2 = if T::DUAL {
        nbterm * 2.0e0 * ratio2_nb / (ratio2_nb + 1.0).powi(2) * rab.rdiv(p.hbnbcut / radab)
    } else {
        nbterm * 2.0e0 * ratio2_nb * expo_nb / (ratio2_nb + 1.0).powi(2) / (ranbprbnb - rab)
    };
    let gi_nb = -tmp2 * ranbprbnb / rab2;
    for c in 0..3 {
        let dg = gi_nb * drab[c];
        ga[c] += dg;
        gb[c] -= dg;
    }
    let gia = tmp2 / ranb;
    let gib = tmp2 / rbnb;
    let mut gnb = [T::cst(0.0); 3];
    for c in 0..3 {
        let dga = gia * dranb[c];
        ga[c] += dga;
        let dgb = gib * drbnb[c];
        gb[c] += dgb;
        gnb[c] = -dga - dgb;
    }
    // torsion and bend parts go to R, B, C, H
    let mut g_r = vec![[T::cst(0.0); 3]; ntors];
    for i in 0..ntors {
        for c in 0..3 {
            g_r[i][c] = gt_r[i][c] * tterm;
        }
    }
    let mut out_b = [T::cst(0.0); 3];
    let mut out_c = [T::cst(0.0); 3];
    let mut out_h = [T::cst(0.0); 3];
    for c in 0..3 {
        out_b[c] = gt_b[c] * tterm + gb3[0][c] * bterm + gb[c];
        out_c[c] = gt_c[c] * tterm + gb3[1][c] * bterm + gnb[c];
        out_h[c] = gt_h[c] * tterm + gb3[2][c] * bterm + gh[c];
    }
    let mut out = vec![ga, out_b, out_h, out_c];
    out.extend(g_r);
    Some((energy, out))
}

/// `egbend_nci_mul`: 1 - k (cos theta - cos theta0)^2 at centre B between C and H
/// (`j` = B centre, `i` = C, `k` = H in xtb's call). Returns (energy, [g_B, g_C, g_H]).
fn bend_nci_mul<T: Real>(xb: V3<T>, xc: V3<T>, xh: V3<T>, c0: f64, fc: f64) -> (T, [V3<T>; 3]) {
    let kijk = fc / (0.0f64.cos() - c0.cos()).powi(2);
    let va = xc;
    let vb = xb;
    let vc = xh;
    let vab = sub(va, vb);
    let vcb = sub(vc, vb);
    let rab2 = vab[0] * vab[0] + vab[1] * vab[1] + vab[2] * vab[2];
    let rcb2 = vcb[0] * vcb[0] + vcb[1] * vcb[1] + vcb[2] * vcb[2];
    if T::DUAL && PI - c0 >= 1.0e-6 {
        // through cos(theta), smooth at a linear C=B...H (see `bend_smooth`)
        let (c, dca, dcc) = cos_and_grad(vab, vcb);
        let f = (c - c0.cos()) * (2.0 * kijk);
        let e = ((c - c0.cos()).powi(2) * kijk).rsub(1.0);
        let (deda, dedc) = (scale(dca, f), scale(dcc, f));
        let neg = |v: V3<T>| [v[0] * -1.0, v[1] * -1.0, v[2] * -1.0];
        return (e, [[deda[0] + dedc[0], deda[1] + dedc[1], deda[2] + dedc[2]], neg(deda), neg(dedc)]);
    }
    let vp = cross(vcb, vab);
    let rp = vlen(vp) + 1.0e-14;
    let mut cosa = impsc(vab, vcb);
    if cosa.val() > 1.0 {
        cosa = T::cst(1.0);
    }
    if cosa.val() < -1.0 {
        cosa = T::cst(-1.0);
    }
    let theta = cosa.acos();
    let (ea, deddt);
    if PI - c0 < 1.0e-6 {
        let dt = theta - c0;
        ea = dt.powi(2) * kijk;
        deddt = dt * (2.0e0 * kijk);
    } else {
        ea = (cosa - c0.cos()).powi(2) * kijk;
        deddt = theta.sin() * (2.0e0 * kijk) * (cosa.rsub(c0.cos()));
    }
    let e = ea.rsub(1.0e0);
    let rmul1 = -(deddt / (rab2 * rp));
    let deda = scale(cross(vab, vp), rmul1);
    let rmul2 = deddt / (rcb2 * rp);
    let dedc = scale(cross(vcb, vp), rmul2);
    let dedb = [deda[0] + dedc[0], deda[1] + dedc[1], deda[2] + dedc[2]];
    (e, [dedb, [-deda[0], -deda[1], -deda[2]], [-dedc[0], -dedc[1], -dedc[2]]])
}

/// Halogen bond (`rbxgfnff_eg`): A-X...B. Returns (energy, [g_A, g_B, g_X]).
pub fn xb<T: Real>(
    xa: V3<T>,
    xbb: V3<T>,
    xx: V3<T>,
    cx: f64,
    qb: f64,
    qx: f64,
    radab: f64,
    par: &crate::param::Param,
) -> Option<(T, [V3<T>; 3])> {
    let cb = 1.0;
    let drax = sub(xa, xx);
    let drbx = sub(xbb, xx);
    let drab = sub(xa, xbb);
    let rab2 = drab[0].powi(2) + drab[1].powi(2) + drab[2].powi(2);
    let rab = rab2.sqrt();
    let rax2 = drax[0].powi(2) + drax[1].powi(2) + drax[2].powi(2);
    let rax = rax2.sqrt() + 1.0e-12;
    let rbx2 = drbx[0].powi(2) + drbx[1].powi(2) + drbx[2].powi(2);
    let rbx = rbx2.sqrt() + 1.0e-12;
    let expo = ((rax + rbx) / rab - 1.0e0) * par.xbacut;
    if expo.val() > 15.0 {
        return None;
    }
    let ratio2 = expo.exp();
    let outl = (ratio2 + 1.0e0).rdiv(2.0e0);
    let ratio1 = (rbx2 / par.hblongcut_xb).powf(par.hbalp);
    let dampl = (ratio1 + 1.0e0).rdiv(1.0e0);
    let shortcut = par.xbscut * radab;
    let ratio3 = (rbx2.rdiv(shortcut)).powf(par.hbalp);
    let damps = (ratio3 + 1.0e0).rdiv(1.0e0);
    let dmp = damps * dampl;
    let rdamp = dmp / rbx2 / rbx;
    let cnst = cb * qb * cx * qx;
    let aterm = -(rdamp * cnst);
    let dterm = -(outl * cnst);
    let energy = -(rdamp * outl * cnst);
    let mut ga = [T::cst(0.0); 3];
    let mut gb = [T::cst(0.0); 3];
    let mut gx = [T::cst(0.0); 3];
    let gi = rdamp
        * (-(ratio1 * (2.0e0 * par.hbalp) / (ratio1 + 1.0e0)) + ratio3 * (2.0e0 * par.hbalp) / (ratio3 + 1.0e0)
            - 3.0e0)
        / rbx2
        * dterm;
    for c in 0..3 {
        let dg = gi * drbx[c];
        gb[c] = dg;
        gx[c] = -dg;
    }
    // expo / (rax + rbx - rab) = xbacut / rab (0/0 at a linear A-X...B; dual path)
    let eod = if T::DUAL { rab.rdiv(par.xbacut) } else { T::cst(0.0) };
    let gi = if T::DUAL {
        ratio2 * 2.0e0 * (rax + rbx) / (ratio2 + 1.0e0).powi(2) * eod / rab2 * aterm
    } else {
        ratio2 * 2.0e0 * expo * (rax + rbx) / (ratio2 + 1.0e0).powi(2) / (rax + rbx - rab) / rab2 * aterm
    };
    for c in 0..3 {
        let dg = gi * drab[c];
        ga[c] = dg;
        gb[c] -= dg;
    }
    let (gi, gj) = if T::DUAL {
        (
            -(ratio2 * 2.0e0) / (ratio2 + 1.0e0).powi(2) * eod / rax * aterm,
            -(ratio2 * 2.0e0) / (ratio2 + 1.0e0).powi(2) * eod / rbx * aterm,
        )
    } else {
        (
            -(ratio2 * 2.0e0) * expo / (ratio2 + 1.0e0).powi(2) / (rax + rbx - rab) / rax * aterm,
            -(ratio2 * 2.0e0) * expo / (ratio2 + 1.0e0).powi(2) / (rax + rbx - rab) / rbx * aterm,
        )
    };
    for c in 0..3 {
        let dga = gi * drax[c];
        ga[c] += dga;
        let dgb = gj * drbx[c];
        gb[c] += dgb;
        gx[c] += -dga - dgb;
    }
    Some((energy, [ga, gb, gx]))
}

/// Gradient on atom i of the EEQ pair energy q_i q_j erf(g r)/r at fixed charges, as the
/// `gfnff_eg` ES gradient loop computes it (atom j gets minus).
#[inline]
pub fn eeq_pair_grad<T: Real>(xi: V3<T>, xj: V3<T>, alpi: f64, alpj: f64, qi: f64, qj: f64) -> V3<T> {
    let d = sub(xi, xj);
    let r2 = d[0].powi(2) + d[1].powi(2) + d[2].powi(2);
    let rab = r2.sqrt();
    let gammij = 1. / (alpi + alpj).sqrt();
    let erff = (rab * gammij).erf();
    let sqrtpi = crate::param::sqrtpi();
    let dd = ((-(r2 * gammij.powi(2))).exp() * (2.0e0 * gammij) / (r2 * sqrtpi) - erff / (rab * r2)) * qi * qj;
    [d[0] * dd, d[1] * dd, d[2] * dd]
}

/// One H...B pair of the HB coordination number (`dncoord_erf`): (count, d count / d x_B)
/// with xtb's truncated sqrt(pi) in the derivative; `None` beyond the 900 bohr² cutoff.
#[inline]
pub fn hbcn_pair<T: Real>(xh: V3<T>, xb: V3<T>, rcovij: f64) -> Option<(T, V3<T>)> {
    use crate::energy::{HB_HLFOSQRTPI, HB_KN};
    let rij = sub(xb, xh);
    let r2 = rij[0].powi(2) + rij[1].powi(2) + rij[2].powi(2);
    if r2.val() > 900.0e0 {
        return None;
    }
    let r = r2.sqrt();
    let tmp = ((r - rcovij) * (-HB_KN) / rcovij).erf() * 0.5 + 0.5;
    let dtmp = -((-((r - rcovij).powi(2) * HB_KN.powi(2) / rcovij.powi(2))).exp() * (HB_HLFOSQRTPI * HB_KN) / rcovij);
    Some((tmp, [dtmp * rij[0] / r, dtmp * rij[1] / r, dtmp * rij[2] / r]))
}

#[cfg(test)]
#[path = "terms_tests.rs"]
mod tests;
