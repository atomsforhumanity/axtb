// SPDX-License-Identifier: LGPL-3.0-or-later
//! GBOBC-II Born radii integrator (port of xtb `src/solv/born.f90`).
//!
//! The pair loop runs over the pairs (i, j < i) in xtb's order so that the
//! accumulated psi values are bitwise identical.  The radii derivatives are not
//! stored; [`add_psi_gradient`] re-runs the pair loop and contracts the pair
//! derivatives directly with dE/dpsi.

use crate::params::{OBC_ALPHA, OBC_BETA, OBC_GAMMA};

/// Contribution of one pair to psi of one atom: value and derivative over r.
#[derive(Clone, Copy)]
struct Contrib {
    g: f64,
    dg: f64,
}

/// Non-overlapping sphere integral (xtb: ap/am/ab/rhab/lnab block).
/// Uses the current value of `r1` like the Fortran code.
#[inline(always)]
fn nonoverlap(r: f64, rho: f64, r1: f64) -> Contrib {
    let ap = r + rho;
    let am = r - rho;
    let ab = ap * am;
    let rhab = rho / ab;
    let lnab = 0.5 * (am / ap).ln() * r1;
    let g = rhab + lnab;
    let dg = -2.0 * rhab / ab + (rhab - lnab) * r1 * r1;
    Contrib { g, dg }
}

/// Overlapping sphere integral (xtb: rh1/rhr1/aprh1 block); sets `r1`.
#[inline(always)]
fn overlap(r: f64, rho: f64, rvdw: f64, r1: &mut f64) -> Contrib {
    *r1 = 1.0 / r;
    let r1 = *r1;
    let r12 = 0.5 * r1;
    let r24 = r12 * r12;
    let ap = r + rho;
    let am = r - rho;
    let rh1 = 1.0 / rvdw;
    let rhr1 = 1.0 / ap;
    let aprh1 = ap * rh1;
    let lnab = aprh1.ln();
    let g = rh1 - rhr1 + r12 * (0.5 * am * (rhr1 - rh1 * aprh1) - lnab);
    let dg = rhr1 * rhr1 * (1.0 - 0.25 * am * r1 * (1.0 + aprh1 * aprh1))
        + rho * r24 * (rhr1 - rh1 * aprh1)
        + r12 * (r1 * lnab - rhr1);
    Contrib { g, dg: dg * r1 }
}

/// Contributions of the pair (ii, jj) to psi(ii) and psi(jj), following the
/// four overlap cases of xtb's `compute_psi`.
///
/// `r1` carries the value of the Fortran local variable of the same name across
/// pairs: in the "ij overlap, ji no overlap" case where the descreened sphere
/// of j lies inside the vdW sphere of i (r + rho_j <= R_i, i the higher atom
/// index), xtb skips the ij term and computes the ji term with a stale `r1`
/// left over from an earlier pair.  This is reproduced on purpose: it changes
/// the energy (by 2.8e-3 Eh for the compressed CsOH test case) and makes xtb's
/// gradient inconsistent with its energy there.  It needs a heavy atom with a
/// small descreening parameter close to a light one (e.g. K/Rb/Cs/Ba/Hg-H,
/// Cs/Ba-O in water, Mo-O in ethanol, Zr-H in CH2Cl2) at distances inside a
/// window of up to ~0.5 Angstrom; ordinary organic molecules never hit it.
/// tblite 0.8 has the same code (with its own pair order).
#[inline(always)]
fn pair(r: f64, rhoi: f64, rhoj: f64, rvdwi: f64, rvdwj: f64, r1: &mut f64) -> (Option<Contrib>, Option<Contrib>) {
    let ovij = !(r >= rvdwi + rhoj);
    let ovji = !(r >= rhoi + rvdwj);
    match (ovij, ovji) {
        (false, false) => {
            *r1 = 1.0 / r;
            if (rhoi - rhoj).abs() < 1.0e-8 {
                let c = nonoverlap(r, rhoj, *r1);
                (Some(c), Some(c))
            } else {
                let ci = nonoverlap(r, rhoj, *r1);
                let cj = nonoverlap(r, rhoi, *r1);
                (Some(ci), Some(cj))
            }
        }
        (false, true) => {
            *r1 = 1.0 / r;
            let ci = nonoverlap(r, rhoj, *r1);
            let cj = if r + rhoi > rvdwj { Some(overlap(r, rhoi, rvdwj, r1)) } else { None };
            (Some(ci), cj)
        }
        (true, false) => {
            let ci = if r + rhoj > rvdwi { Some(overlap(r, rhoj, rvdwi, r1)) } else { None };
            // xtb uses whatever r1 holds here (see above)
            let cj = nonoverlap(r, rhoi, *r1);
            (ci, Some(cj))
        }
        (true, true) => {
            let ci = if r + rhoj > rvdwi { Some(overlap(r, rhoj, rvdwi, r1)) } else { None };
            let cj = if r + rhoi > rvdwj { Some(overlap(r, rhoi, rvdwj, r1)) } else { None };
            (ci, cj)
        }
    }
}

#[inline(always)]
fn pair_vec(xyz: &[[f64; 3]], i: usize, j: usize) -> ([f64; 3], f64, f64) {
    let x = xyz[i][0] - xyz[j][0];
    let y = xyz[i][1] - xyz[j][1];
    let z = xyz[i][2] - xyz[j][2];
    let dr2 = x * x + y * y + z * z;
    ([x, y, z], dr2, dr2.sqrt())
}

/// Born radii (xtb `compute_bornr`) and the chain factors dbrad/dpsi.
#[inline(never)]
pub(crate) fn born_radii(
    xyz: &[[f64; 3]],
    vdwr: &[f64],
    rho: &[f64],
    svdw: &[f64],
    c1: f64,
    lrcut: f64,
    brad: &mut [f64],
    dbrad: &mut [f64],
    fresh_r1: bool,
) {
    let nat = xyz.len();
    let lrcut2 = lrcut * lrcut;
    let psi = brad;
    psi.iter_mut().for_each(|p| *p = 0.0);
    let mut r1 = 0.0f64;
    for ii in 0..nat {
        for jj in 0..ii {
            let (_, dr2, r) = pair_vec(xyz, ii, jj);
            if !(dr2 < lrcut2) {
                continue;
            }
            if fresh_r1 {
                r1 = 1.0 / r;
            }
            let (ci, cj) = pair(r, rho[ii], rho[jj], vdwr[ii], vdwr[jj], &mut r1);
            if let Some(c) = ci {
                psi[ii] += c.g;
            }
            if let Some(c) = cj {
                psi[jj] += c.g;
            }
        }
    }
    for iat in 0..nat {
        let mut br = psi[iat];
        let svdwi = svdw[iat];
        let vdwri = vdwr[iat];
        let s1 = 1.0 / svdwi;
        let v1 = 1.0 / vdwri;
        let s2 = 0.5 * svdwi;
        br *= s2;
        let mut arg2 = br * (OBC_GAMMA * br - OBC_BETA);
        let arg = br * (OBC_ALPHA + arg2);
        arg2 = 2.0 * arg2 + OBC_ALPHA + OBC_GAMMA * br * br;
        let th = arg.tanh();
        let ch = arg.cosh();
        br = 1.0 / (s1 - v1 * th);
        br *= c1;
        let mut dpsi = ch * (s1 - v1 * th);
        dpsi = s2 * v1 * arg2 / (dpsi * dpsi);
        dpsi *= c1;
        psi[iat] = br;
        dbrad[iat] = dpsi;
    }
}

/// Add sum_i f_i * d psi_i / d R to the gradient (f_i = dE/dbrad_i * dbrad_i/dpsi_i).
#[inline(never)]
pub(crate) fn add_psi_gradient(
    xyz: &[[f64; 3]],
    vdwr: &[f64],
    rho: &[f64],
    lrcut: f64,
    f: &[f64],
    grad: &mut [[f64; 3]],
    fresh_r1: bool,
) {
    let nat = xyz.len();
    let lrcut2 = lrcut * lrcut;
    let mut r1 = 0.0f64;
    for ii in 0..nat {
        let mut gi = [0.0f64; 3];
        for jj in 0..ii {
            let (dr, dr2, r) = pair_vec(xyz, ii, jj);
            if !(dr2 < lrcut2) {
                continue;
            }
            if fresh_r1 {
                r1 = 1.0 / r;
            }
            let (ci, cj) = pair(r, rho[ii], rho[jj], vdwr[ii], vdwr[jj], &mut r1);
            // d psi_ii / d R_ii = +dgi*dr, d psi_ii / d R_jj = -dgi*dr
            // d psi_jj / d R_jj = -dgj*dr, d psi_jj / d R_ii = +dgj*dr
            let mut s = 0.0;
            if let Some(c) = ci {
                s += f[ii] * c.dg;
            }
            if let Some(c) = cj {
                s += f[jj] * c.dg;
            }
            if s != 0.0 {
                for k in 0..3 {
                    let d = s * dr[k];
                    gi[k] += d;
                    grad[jj][k] -= d;
                }
            }
        }
        for k in 0..3 {
            grad[ii][k] += gi[k];
        }
    }
}
