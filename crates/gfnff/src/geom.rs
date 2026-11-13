// SPDX-License-Identifier: LGPL-3.0-or-later
//! Internal coordinates and their derivatives as GFN-FF evaluates them (xtb
//! `src/constr.f90`, `src/gfnff/math.f`), for molecules (all translation vectors zero).

use crate::real::{cross, norm2, sq, sub, Real, V3};

const EPS: f64 = 1.0e-14;

/// `vecnorm(r,3,1)`: normalizes `r` in place if its length exceeds 1e-14; returns the length.
#[inline]
pub fn vecnorm_inplace<T: Real>(r: &mut V3<T>) -> T {
    let sp = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    let rn = sp.sqrt();
    if rn.val().abs() > EPS {
        let or = rn.rdiv(1.0);
        r[0] = or * r[0];
        r[1] = or * r[1];
        r[2] = or * r[2];
    }
    rn
}

/// `vlen`: length, 0 for the zero vector.
#[inline]
pub fn vlen<T: Real>(a: V3<T>) -> T {
    let tot = a[0] * a[0] + a[1] * a[1] + a[2] * a[2];
    if tot.val() > 0.0 {
        tot.sqrt()
    } else {
        T::cst(0.0)
    }
}

/// `impsc`: cosine between two vectors.
#[inline]
pub fn impsc<T: Real>(a: V3<T>, b: V3<T>) -> T {
    let rimp = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let al = norm2(a);
    let bl = norm2(b);
    if al.val() > 0.0 && bl.val() > 0.0 {
        rimp / (al * bl)
    } else {
        T::cst(0.0)
    }
}

/// `bangl`/`banglPBC` (mode 1, no translation): angle i-j-k with j in the middle.
pub fn bangl(xi: [f64; 3], xj: [f64; 3], xk: [f64; 3]) -> f64 {
    let d2ij = sq(sub(xi, xj));
    let d2jk = sq(sub(xj, xk));
    let d2ik = sq(sub(xi, xk));
    let xy = (d2ij * d2jk).sqrt();
    let temp = 0.5 * (d2ij + d2jk - d2ik) / xy;
    temp.clamp(-1.0, 1.0).acos()
}

/// `valijklff` (and `valijklffPBC` without translations): dihedral angle i-j-k-l in [0, pi].
pub fn valijklff<T: Real>(xi: V3<T>, xj: V3<T>, xk: V3<T>, xl: V3<T>) -> T {
    let ra = sub(xj, xi);
    let rb = sub(xk, xj);
    let rc = sub(xl, xk);
    let mut na = cross(ra, rb);
    let mut nb = cross(rb, rc);
    vecnorm_inplace(&mut na);
    vecnorm_inplace(&mut nb);
    let mut snanb = T::cst(0.0);
    for ic in 0..3 {
        snanb += na[ic] * nb[ic];
    }
    if (snanb.val().abs() - 1.0).abs() < EPS {
        snanb = T::cst(1.0f64.copysign(snanb.val()));
    }
    snanb.acos()
}

/// `dphidr`/`dphidrPBC`: derivatives of the dihedral angle with respect to the four atoms.
/// `pbc_variant` selects the zero-guard of `dphidrPBC` (used by `egtors`) over that of
/// `dphidr` (used by the triple-bond torsion); they differ only for degenerate geometries.
pub fn dphidr<T: Real>(xi: V3<T>, xj: V3<T>, xk: V3<T>, xl: V3<T>, phi: T, pbc_variant: bool) -> [V3<T>; 4] {
    let cosphi = phi.cos();
    let sinphi = phi.sin();
    let ra = sub(xj, xi);
    let rb = sub(xk, xj);
    let rc = sub(xl, xk);
    let rapb = [ra[0] + rb[0], ra[1] + rb[1], ra[2] + rb[2]];
    let rbpc = [rb[0] + rc[0], rb[1] + rc[1], rb[2] + rc[2]];
    let na = cross(ra, rb);
    let nb = cross(rb, rc);
    let nan = norm_plain(na);
    let nbn = norm_plain(nb);
    let nenner = nan * nbn * sinphi;
    let zero = [T::cst(0.0); 3];
    let onenner;
    if nenner.val().abs() < EPS {
        if pbc_variant {
            if (nan * nbn).val().abs() > EPS {
                onenner = (nan * nbn).rdiv(1.0);
            } else {
                onenner = T::cst(0.0);
            }
        } else {
            onenner = (nan * nbn).rdiv(1.0);
        }
    } else {
        onenner = nenner.rdiv(1.0);
    }
    if pbc_variant && onenner.val().abs() <= EPS {
        return [zero, zero, zero, zero];
    }
    let rab = cross(na, rb);
    let rba = cross(nb, ra);
    let rac = cross(na, rc);
    let rbb = cross(nb, rb);
    let rbc = cross(nb, rc);
    let raa = cross(na, ra);
    let rapba = cross(rapb, na);
    let rapbb = cross(rapb, nb);
    let rbpca = cross(rbpc, na);
    let rbpcb = cross(rbpc, nb);
    let mut di = zero;
    let mut dj = zero;
    let mut dk = zero;
    let mut dl = zero;
    for ic in 0..3 {
        di[ic] = onenner * (cosphi * nbn / nan * rab[ic] - rbb[ic]);
        dj[ic] = onenner * (cosphi * (nbn / nan * rapba[ic] + nan / nbn * rbc[ic]) - (rac[ic] + rapbb[ic]));
        dk[ic] = onenner * (cosphi * (nbn / nan * raa[ic] + nan / nbn * rbpcb[ic]) - (rba[ic] + rbpca[ic]));
        dl[ic] = onenner * (cosphi * nan / nbn * rbb[ic] - rab[ic]);
    }
    [di, dj, dk, dl]
}

/// `vecnorm(r,3,0)`: length only.
#[inline]
fn norm_plain<T: Real>(r: V3<T>) -> T {
    let sp = r[0] * r[0] + r[1] * r[1] + r[2] * r[2];
    sp.sqrt()
}

/// `omegaPBC` without translations: inversion angle of centre i over j, k, l (in that order
/// of distance): asin of the normalized (re x rd) . rv.
pub fn omega<T: Real>(xi: V3<T>, xj: V3<T>, xk: V3<T>, xl: V3<T>) -> T {
    let re = sub(xi, xj);
    let rd = sub(xk, xj);
    let mut rv = sub(xl, xi);
    let mut rn = cross(re, rd);
    vecnorm_inplace(&mut rn);
    vecnorm_inplace(&mut rv);
    let rnv = rn[0] * rv[0] + rn[1] * rv[1] + rn[2] * rv[2];
    rnv.asin()
}

/// `domegadrPBC` without translations.
pub fn domegadr<T: Real>(xi: V3<T>, xj: V3<T>, xk: V3<T>, xl: V3<T>, omega: T) -> [V3<T>; 4] {
    let sinomega = omega.sin();
    let re = sub(xi, xj);
    let rd = sub(xk, xj);
    let rv = sub(xl, xi);
    let rdme = sub(rd, re);
    let rn = cross(re, rd);
    let rvn = norm_plain(rv);
    let rnn = norm_plain(rn);
    let rve = cross(rv, re);
    let rne = cross(rn, re);
    let rdv = cross(rd, rv);
    let rdn = cross(rd, rn);
    let rvdme = cross(rv, rdme);
    let rndme = cross(rn, rdme);
    let nenner = rnn * rvn * omega.cos();
    let zero = [T::cst(0.0); 3];
    if nenner.val().abs() > EPS {
        let onenner = nenner.rdiv(1.0);
        let mut di = zero;
        let mut dj = zero;
        let mut dk = zero;
        let mut dl = zero;
        for ic in 0..3 {
            di[ic] = onenner * (rdv[ic] - rn[ic] - sinomega * (rvn / rnn * rdn[ic] - rnn / rvn * rv[ic]));
            dj[ic] = onenner * (rvdme[ic] - sinomega * rvn / rnn * rndme[ic]);
            dk[ic] = onenner * (rve[ic] - sinomega * rvn / rnn * rne[ic]);
            dl[ic] = onenner * (rn[ic] - sinomega * rnn / rvn * rv[ic]);
        }
        [di, dj, dk, dl]
    } else {
        [zero, zero, zero, zero]
    }
}
