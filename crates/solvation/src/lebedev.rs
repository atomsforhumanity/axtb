// SPDX-License-Identifier: LGPL-3.0-or-later
//! Lebedev-Laikov angular grids, generated exactly as in xtb's `lebedev.f90`
//! (same generator sequence, same point order, same arithmetic).

use crate::lebedev_data::{GRIDS, GRID_SIZE};

/// One Oh-symmetric generator of a Lebedev-Laikov grid (xtb `genOh1`..`genOh6`).
#[derive(Clone, Copy, Debug)]
pub enum Gen {
    /// 6 points from (0,0,1), weight v
    Oh1(f64),
    /// 12 points from (0,a,a), a=1/sqrt(2), weight v
    Oh2(f64),
    /// 8 points from (a,a,a), a=1/sqrt(3), weight v
    Oh3(f64),
    /// 24 points from (a,a,b), b=sqrt(1-2a^2): (a, v)
    Oh4(f64, f64),
    /// 24 points from (a,b,0), b=sqrt(1-a^2): (a, v)
    Oh5(f64, f64),
    /// 48 points from (a,b,c), c=sqrt(1-a^2-b^2): (a, b, v)
    Oh6(f64, f64, f64),
}

/// Grid sizes supported by xtb.
pub fn grid_sizes() -> &'static [usize] {
    &GRID_SIZE
}

/// xtb's `bisectSearch` on the grid sizes: 1-based index j with
/// `GRID_SIZE[j] <= n < GRID_SIZE[j+1]`, including its quirk that the largest
/// grid size maps to the second largest grid.
pub(crate) fn bisect_grid(n: usize) -> usize {
    let xx = &GRID_SIZE;
    let len = xx.len();
    if n < xx[0] {
        0
    } else if n == xx[0] {
        1
    } else if n == xx[len - 1] {
        len - 1
    } else if n > xx[len - 1] {
        len
    } else {
        let (mut jlower, mut jcurr) = (0usize, len + 1);
        while jcurr - jlower > 1 {
            let jupper = (jcurr + jlower) / 2;
            if n >= xx[jupper - 1] {
                jlower = jupper;
            } else {
                jcurr = jupper;
            }
        }
        jlower
    }
}

/// Generate grid number `iang` (1-based, as in xtb `getAngGrid`).
/// Returns `None` for an invalid index.
pub(crate) fn angular_grid(iang: usize) -> Option<(Vec<[f64; 3]>, Vec<f64>)> {
    if iang == 0 || iang > GRIDS.len() {
        return None;
    }
    let n = GRID_SIZE[iang - 1];
    let mut x: Vec<[f64; 3]> = Vec::with_capacity(n);
    let mut w: Vec<f64> = Vec::with_capacity(n);
    for g in GRIDS[iang - 1] {
        gen_points(*g, &mut x, &mut w);
    }
    debug_assert_eq!(x.len(), n);
    Some((x, w))
}

fn gen_points(g: Gen, x: &mut Vec<[f64; 3]>, w: &mut Vec<f64>) {
    const Z: f64 = 0.0;
    match g {
        Gen::Oh1(v) => {
            let a = 1.0;
            x.extend_from_slice(&[[a, Z, Z], [-a, Z, Z], [Z, a, Z], [Z, -a, Z], [Z, Z, a], [Z, Z, -a]]);
            w.extend(std::iter::repeat_n(v, 6));
        }
        Gen::Oh2(v) => {
            let a = 0.5f64.sqrt();
            x.extend_from_slice(&[
                [Z, a, a],
                [Z, -a, a],
                [Z, a, -a],
                [Z, -a, -a],
                [a, Z, a],
                [-a, Z, a],
                [a, Z, -a],
                [-a, Z, -a],
                [a, a, Z],
                [-a, a, Z],
                [a, -a, Z],
                [-a, -a, Z],
            ]);
            w.extend(std::iter::repeat_n(v, 12));
        }
        Gen::Oh3(v) => {
            let a = (1.0f64 / 3.0).sqrt();
            x.extend_from_slice(&[
                [a, a, a],
                [-a, a, a],
                [a, -a, a],
                [-a, -a, a],
                [a, a, -a],
                [-a, a, -a],
                [a, -a, -a],
                [-a, -a, -a],
            ]);
            w.extend(std::iter::repeat_n(v, 8));
        }
        Gen::Oh4(a, v) => {
            let b = (1.0 - 2.0 * a * a).sqrt();
            x.extend_from_slice(&[
                [a, a, b],
                [-a, a, b],
                [a, -a, b],
                [-a, -a, b],
                [a, a, -b],
                [-a, a, -b],
                [a, -a, -b],
                [-a, -a, -b],
                [a, b, a],
                [-a, b, a],
                [a, -b, a],
                [-a, -b, a],
                [a, b, -a],
                [-a, b, -a],
                [a, -b, -a],
                [-a, -b, -a],
                [b, a, a],
                [-b, a, a],
                [b, -a, a],
                [-b, -a, a],
                [b, a, -a],
                [-b, a, -a],
                [b, -a, -a],
                [-b, -a, -a],
            ]);
            w.extend(std::iter::repeat_n(v, 24));
        }
        Gen::Oh5(a, v) => {
            let b = (1.0 - a * a).sqrt();
            x.extend_from_slice(&[
                [a, b, Z],
                [-a, b, Z],
                [a, -b, Z],
                [-a, -b, Z],
                [b, a, Z],
                [-b, a, Z],
                [b, -a, Z],
                [-b, -a, Z],
                [a, Z, b],
                [-a, Z, b],
                [a, Z, -b],
                [-a, Z, -b],
                [b, Z, a],
                [-b, Z, a],
                [b, Z, -a],
                [-b, Z, -a],
                [Z, a, b],
                [Z, -a, b],
                [Z, a, -b],
                [Z, -a, -b],
                [Z, b, a],
                [Z, -b, a],
                [Z, b, -a],
                [Z, -b, -a],
            ]);
            w.extend(std::iter::repeat_n(v, 24));
        }
        Gen::Oh6(a, b, v) => {
            let c = (1.0 - a * a - b * b).sqrt();
            x.extend_from_slice(&[
                [a, b, c],
                [-a, b, c],
                [a, -b, c],
                [-a, -b, c],
                [a, b, -c],
                [-a, b, -c],
                [a, -b, -c],
                [-a, -b, -c],
                [a, c, b],
                [-a, c, b],
                [a, -c, b],
                [-a, -c, b],
                [a, c, -b],
                [-a, c, -b],
                [a, -c, -b],
                [-a, -c, -b],
                [b, a, c],
                [-b, a, c],
                [b, -a, c],
                [-b, -a, c],
                [b, a, -c],
                [-b, a, -c],
                [b, -a, -c],
                [-b, -a, -c],
                [b, c, a],
                [-b, c, a],
                [b, -c, a],
                [-b, -c, a],
                [b, c, -a],
                [-b, c, -a],
                [b, -c, -a],
                [-b, -c, -a],
                [c, a, b],
                [-c, a, b],
                [c, -a, b],
                [-c, -a, b],
                [c, a, -b],
                [-c, a, -b],
                [c, -a, -b],
                [-c, -a, -b],
                [c, b, a],
                [-c, b, a],
                [c, -b, a],
                [-c, -b, a],
                [c, b, -a],
                [-c, b, -a],
                [c, -b, -a],
                [-c, -b, -a],
            ]);
            w.extend(std::iter::repeat_n(v, 48));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_grids_have_expected_size_and_unit_weight() {
        for (i, &n) in GRID_SIZE.iter().enumerate() {
            let (x, w) = angular_grid(i + 1).unwrap();
            assert_eq!(x.len(), n);
            let s: f64 = w.iter().sum();
            assert!((s - 1.0).abs() < 1e-13, "grid {n}: sum of weights {s}");
            for p in &x {
                let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                assert!((r - 1.0).abs() < 1e-14);
            }
            // integrates x^2 exactly: 1/3
            let m: f64 = x.iter().zip(&w).map(|(p, w)| w * p[0] * p[0]).sum();
            assert!((m - 1.0 / 3.0).abs() < 1e-13);
        }
    }

    #[test]
    fn bisection_matches_xtb() {
        assert_eq!(bisect_grid(230), 12);
        assert_eq!(bisect_grid(231), 12);
        assert_eq!(bisect_grid(6), 1);
        assert_eq!(bisect_grid(5), 0);
        assert_eq!(bisect_grid(5294), 31);
        assert_eq!(bisect_grid(5810), 31); // xtb quirk
        assert_eq!(bisect_grid(6000), 32);
        assert_eq!(bisect_grid(974), 19);
    }
}
