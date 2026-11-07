// SPDX-License-Identifier: LGPL-3.0-or-later
//! Generalized Born interaction kernels and the ALPB shape descriptor
//! (port of xtb `src/solv/kernel.f90` and `getADet`/`addADetDeriv` of `gbsa.f90`).

use crate::params::ZETA_P16;

const ZETA_P16_O16: f64 = ZETA_P16 / 16.0;
const A4: f64 = 0.25;

/// Interaction kernel of the generalized Born model.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    /// Classical Still kernel (GBSA)
    Still,
    /// P16 kernel by Lange (ALPB)
    P16,
}

#[inline(always)]
fn pair(xyz: &[[f64; 3]], i: usize, j: usize) -> ([f64; 3], f64) {
    let x = xyz[i][0] - xyz[j][0];
    let y = xyz[i][1] - xyz[j][1];
    let z = xyz[i][2] - xyz[j][2];
    ([x, y, z], (x * x + y * y + z * z).sqrt())
}

/// Fill the (full, row-major, symmetric) Born matrix: kernel part only
/// (xtb `addBornMatP16` / `addBornMatStill` on a zeroed matrix).
#[inline(never)]
pub(crate) fn born_matrix(kernel: Kernel, xyz: &[[f64; 3]], keps: f64, brad: &[f64], mat: &mut [f64]) {
    let nat = xyz.len();
    match kernel {
        Kernel::P16 => {
            for i in 0..nat {
                for j in 0..i {
                    let (_, r1) = pair(xyz, i, j);
                    let ab = (brad[i] * brad[j]).sqrt();
                    let mut arg = ab / (ab + ZETA_P16_O16 * r1);
                    arg *= arg;
                    arg *= arg;
                    arg *= arg;
                    arg *= arg;
                    let fgb = r1 + ab * arg;
                    let dfgb = 1.0 / fgb;
                    let v = keps * dfgb;
                    mat[i * nat + j] = v;
                    mat[j * nat + i] = v;
                }
            }
        }
        Kernel::Still => {
            for i in 0..nat {
                for j in 0..i {
                    let (_, r) = pair(xyz, i, j);
                    let r2 = r * r;
                    let aa = brad[i] * brad[j];
                    let dd = A4 * r2 / aa;
                    let expd = (-dd).exp();
                    let fgb2 = r2 + aa * expd;
                    let dfgb = 1.0 / fgb2.sqrt();
                    let v = keps * dfgb;
                    mat[i * nat + j] = v;
                    mat[j * nat + i] = v;
                }
            }
        }
    }
    for i in 0..nat {
        let bp = 1.0 / brad[i];
        mat[i * nat + i] = keps * bp;
    }
}

/// Explicit derivative of the polar energy 1/2 q^T B(R, a(R)) q at fixed
/// charges: direct pair terms are added to `grad`, the derivatives with
/// respect to the Born radii are returned in `dedbr` (overwritten).
#[inline(never)]
pub(crate) fn add_gradient(
    kernel: Kernel,
    xyz: &[[f64; 3]],
    q: &[f64],
    keps: f64,
    brad: &[f64],
    grad: &mut [[f64; 3]],
    dedbr: &mut [f64],
) {
    let nat = xyz.len();
    dedbr.iter_mut().for_each(|d| *d = 0.0);
    match kernel {
        Kernel::P16 => {
            for iat in 0..nat {
                let mut gi = [0.0f64; 3];
                let mut di = 0.0f64;
                for jat in 0..iat {
                    let (vec, r1) = pair(xyz, iat, jat);
                    let qq = q[iat] * q[jat];
                    let ab = (brad[iat] * brad[jat]).sqrt();
                    let arg1 = ab / (ab + ZETA_P16_O16 * r1);
                    let mut arg16 = arg1 * arg1;
                    arg16 *= arg16;
                    arg16 *= arg16;
                    arg16 *= arg16;
                    let fgb = r1 + ab * arg16;
                    let dfgb = 1.0 / fgb;
                    let dfgb2 = dfgb * dfgb;
                    let ap = (1.0 - ZETA_P16 * arg1 * arg16) * dfgb2;
                    for k in 0..3 {
                        let dg = ap * vec[k] * keps / r1 * qq;
                        gi[k] -= dg;
                        grad[jat][k] += dg;
                    }
                    let bp = -0.5 * (r1 * ZETA_P16 / ab * arg1 + 1.0) / ab * arg16 * dfgb2;
                    di += brad[jat] * bp * keps * qq;
                    dedbr[jat] += brad[iat] * bp * keps * qq;
                }
                for k in 0..3 {
                    grad[iat][k] += gi[k];
                }
                dedbr[iat] += di;
            }
        }
        Kernel::Still => {
            for iat in 0..nat {
                let mut gi = [0.0f64; 3];
                let mut di = 0.0f64;
                for jat in 0..iat {
                    let (vec, r) = pair(xyz, iat, jat);
                    let r2 = r * r;
                    let qq = q[iat] * q[jat];
                    let aa = brad[iat] * brad[jat];
                    let dd = A4 * r2 / aa;
                    let expd = (-dd).exp();
                    let fgb2 = r2 + aa * expd;
                    let dfgb2 = 1.0 / fgb2;
                    let dfgb = dfgb2.sqrt();
                    let dfgb3 = dfgb2 * dfgb * keps;
                    let ap = (1.0 - A4 * expd) * dfgb3;
                    for k in 0..3 {
                        let dr = ap * vec[k];
                        gi[k] -= dr * qq;
                        grad[jat][k] += dr * qq;
                    }
                    let bp = -0.5 * expd * (1.0 + dd) * dfgb3;
                    di += brad[jat] * bp * qq;
                    dedbr[jat] += brad[iat] * bp * qq;
                }
                for k in 0..3 {
                    grad[iat][k] += gi[k];
                }
                dedbr[iat] += di;
            }
        }
    }
    // self-energy part (identical for both kernels)
    for iat in 0..nat {
        let bp = 1.0 / brad[iat];
        let qq = q[iat] * bp;
        let dedbri = -0.5 * keps * qq * bp;
        dedbr[iat] += dedbri * q[iat];
    }
}

const TOF: f64 = 2.0 / 5.0;

fn inertia(xyz: &[[f64; 3]], rad: &[f64]) -> ([[f64; 3]; 3], [f64; 3], f64) {
    let mut tot_rad3 = 0.0f64;
    let mut center = [0.0f64; 3];
    for (x, &r) in xyz.iter().zip(rad) {
        let rad2 = r * r;
        let rad3 = rad2 * r;
        tot_rad3 += rad3;
        for k in 0..3 {
            center[k] += x[k] * rad3;
        }
    }
    for c in center.iter_mut() {
        *c /= tot_rad3;
    }
    // inertia[row][col] == Fortran inertia(row, col)
    let mut inertia = [[0.0f64; 3]; 3];
    for (x, &r) in xyz.iter().zip(rad) {
        let rad2 = r * r;
        let rad3 = rad2 * r;
        let vec = [x[0] - center[0], x[1] - center[1], x[2] - center[2]];
        let r2 = vec[0] * vec[0] + vec[1] * vec[1] + vec[2] * vec[2];
        let diag = rad3 * (r2 + TOF * rad2);
        for (k, row) in inertia.iter_mut().enumerate() {
            row[k] += diag;
        }
        for col in 0..3 {
            for row in 0..3 {
                inertia[row][col] -= rad3 * vec[col] * vec[row];
            }
        }
    }
    (inertia, center, tot_rad3)
}

fn det3(a: &[[f64; 3]; 3]) -> f64 {
    a[0][0] * a[1][1] * a[2][2] - a[0][0] * a[1][2] * a[2][1] - a[0][1] * a[1][0] * a[2][2]
        + a[0][1] * a[1][2] * a[2][0]
        + a[0][2] * a[1][0] * a[2][1]
        - a[0][2] * a[1][1] * a[2][0]
}

/// ALPB shape descriptor (xtb `getADet`).
pub(crate) fn adet(xyz: &[[f64; 3]], rad: &[f64]) -> f64 {
    let (inertia, _, tot_rad3) = inertia(xyz, rad);
    (det3(&inertia).powf(1.0 / 3.0) / (TOF * tot_rad3)).sqrt()
}

/// Gradient of 1/2 keps alpbet / aDet * (sum q)^2 (xtb `addADetDeriv`,
/// called with `keps_alpbet = keps * alpbet`).
pub(crate) fn add_adet_deriv(xyz: &[[f64; 3]], rad: &[f64], keps_alpbet: f64, q: &[f64], grad: &mut [[f64; 3]]) {
    let qtotal: f64 = q.iter().fold(0.0, |s, &x| s + x);
    let (i, center, tot_rad3) = inertia(xyz, rad);
    let adet = (det3(&i).powf(1.0 / 3.0) / (TOF * tot_rad3)).sqrt();
    // Fortran: x**3 and x**5 are expanded as x2*x and (x2*x2)*x
    let t2 = tot_rad3 * tot_rad3;
    let a2 = adet * adet;
    let fa = 250.0 / (48.0 * (t2 * tot_rad3) * ((a2 * a2) * adet));
    let fb = -0.5 * keps_alpbet * (qtotal * qtotal) / a2;
    let f = |x: f64| x * fa * fb;
    let xy = i[0][1] * i[2][2] - i[0][2] * i[1][2];
    let xz = i[0][2] * i[1][1] - i[0][1] * i[2][1];
    let yz = i[0][0] * i[1][2] - i[0][1] * i[0][2];
    let xx = i[0][0] * (i[1][1] + i[2][2]) - i[0][1] * i[0][1] - i[0][2] * i[0][2];
    let yy = i[1][1] * (i[0][0] + i[2][2]) - i[0][1] * i[0][1] - i[1][2] * i[1][2];
    let zz = i[2][2] * (i[0][0] + i[1][1]) - i[0][2] * i[0][2] - i[1][2] * i[1][2];
    let d = [[f(xx), f(xy), f(xz)], [f(xy), f(yy), f(yz)], [f(xz), f(yz), f(zz)]];
    for (g, (x, &r)) in grad.iter_mut().zip(xyz.iter().zip(rad)) {
        let rad3 = r * r * r;
        let vec = [x[0] - center[0], x[1] - center[1], x[2] - center[2]];
        for k in 0..3 {
            g[k] += rad3 * (d[k][0] * vec[0] + d[k][1] * vec[1] + d[k][2] * vec[2]);
        }
    }
}
