// SPDX-License-Identifier: LGPL-3.0-or-later
//! The error function of the reference programs, 3-vector helpers, the Cartesian Hessian
//! block of a pair term, the normalization of Gaussian primitives and the atomic sums of shell
//! quantities.
//!
//! Sources: the C library's `erf`; the 3-vector and pair-Hessian helpers of `axtb` and `axtb-d4`;
//! the primitive normalization of axtb's STO-NG basis (tblite's `slater_to_gauss`) and Molden
//! writer, which associate the factor 2α/π differently, both kept ([`primitive_norm_tblite`],
//! [`primitive_norm`]); the shell-to-atom sum of axtb's SCF.

extern "C" {
    // C99 libm error function (the routine gfortran's ERF intrinsic calls).
    #[link_name = "erf"]
    fn libm_erf(x: f64) -> f64;
}

/// The error function of the C library (libm), the routine gfortran's `ERF` intrinsic calls, so
/// that terms with an error function keep the reference programs' bits. Every crate of the
/// workspace uses this one.
#[inline]
pub fn erf(x: f64) -> f64 {
    // SAFETY: a pure libm function of a double, without side effects
    unsafe { libm_erf(x) }
}

/// Difference vector `a - b`.
#[inline]
pub fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Squared norm, summed in index order.
#[inline]
pub fn norm2_sq(v: [f64; 3]) -> f64 {
    v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
}

/// Dot product, summed in index order.
#[inline]
pub fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Add the Cartesian Hessian of a pair term `E(r)` with `dE/dr = e1` and `d²E/dr² = e2` to `h`
/// (row-major `n3 x n3`): with `u = v/r`, the 3x3 block `e2 u uᵀ + (e1/r) (1 - u uᵀ)` is added to
/// the (i, i) and (j, j) blocks and subtracted from the (i, j) and (j, i) blocks.
/// `v = R_i - R_j`, `r = |v|`.
#[inline]
pub fn add_radial_hessian(h: &mut [f64], n3: usize, i: usize, j: usize, v: [f64; 3], r: f64, e1: f64, e2: f64) {
    let u = [v[0] / r, v[1] / r, v[2] / r];
    for a in 0..3 {
        for b in 0..3 {
            let d = if a == b { 1.0 } else { 0.0 };
            let blk = e2 * u[a] * u[b] + e1 / r * (d - u[a] * u[b]);
            h[(3 * i + a) * n3 + 3 * i + b] += blk;
            h[(3 * j + a) * n3 + 3 * j + b] += blk;
            h[(3 * i + a) * n3 + 3 * j + b] -= blk;
            h[(3 * j + a) * n3 + 3 * i + b] -= blk;
        }
    }
}

/// Double factorials (2l − 1)!! for l = 0, 1, 2, ... (with (−1)!! = 1).
pub const DFACTORIAL: [f64; 8] = [1.0, 1.0, 3.0, 15.0, 105.0, 945.0, 10395.0, 135135.0];

/// Normalization of the Gaussian primitive x^l exp(−α r²) (one Cartesian component of angular
/// momentum l): (2α/π)^¾ (4α)^(l/2) / √((2l − 1)!!), evaluated as
/// `(2.0 * alpha / PI).powf(0.75) * (4.0 * alpha).sqrt().powi(l) / DFACTORIAL[l].sqrt()`.
/// [`primitive_norm_tblite`] associates the first factor differently.
#[inline]
pub fn primitive_norm(l: usize, alpha: f64) -> f64 {
    (2.0 * alpha / std::f64::consts::PI).powf(0.75) * (4.0 * alpha).sqrt().powi(l as i32) / DFACTORIAL[l].sqrt()
}

/// [`primitive_norm`] with the association of tblite's `slater_to_gauss` (and xtb's):
/// `((2/π) α)^¾`, i.e. `(2.0 / PI * alpha).powf(0.75) * ...`, which can differ from
/// `(2α/π)^¾` in the last bit.
#[inline]
pub fn primitive_norm_tblite(l: usize, alpha: f64) -> f64 {
    let top = 2.0 / std::f64::consts::PI;
    (top * alpha).powf(0.75) * (4.0 * alpha).sqrt().powi(l as i32) / DFACTORIAL[l].sqrt()
}

/// Atomic sums of shell quantities, e.g. atomic charges from shell charges: `qat[A]` is set
/// to the sum of `qsh[l]` over the shells l of atom A (`shell_atoms` gives the atom of each
/// shell), added in shell order.
pub fn atom_charges_into(qat: &mut [f64], shell_atoms: impl IntoIterator<Item = usize>, qsh: &[f64]) {
    qat.iter_mut().for_each(|x| *x = 0.0);
    for (a, &q) in shell_atoms.into_iter().zip(qsh) {
        qat[a] += q;
    }
}

/// [`atom_charges_into`] into a new vector of `nat` atoms.
pub fn atom_charges(nat: usize, shell_atoms: impl IntoIterator<Item = usize>, qsh: &[f64]) -> Vec<f64> {
    let mut qat = vec![0.0; nat];
    atom_charges_into(&mut qat, shell_atoms, qsh);
    qat
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn primitive_norms_keep_their_association() {
        for l in 0..5 {
            for a in [0.1, 0.37, 1.7, 23.0, 1234.5] {
                let molden = (2.0 * a / PI).powf(0.75) * (4.0 * a).sqrt().powi(l as i32) / DFACTORIAL[l].sqrt();
                let top = 2.0 / PI;
                let tblite = (top * a).powf(0.75) * (4.0 * a).sqrt().powi(l as i32) / DFACTORIAL[l].sqrt();
                assert_eq!(primitive_norm(l, a).to_bits(), molden.to_bits());
                assert_eq!(primitive_norm_tblite(l, a).to_bits(), tblite.to_bits());
            }
        }
    }

    #[test]
    fn atom_charges_sum_shells_in_order() {
        let q = atom_charges(3, [0, 0, 2, 2, 2], &[0.5, -0.25, 1.0, 2.0, -3.0]);
        assert_eq!(q, [0.25, 0.0, 0.0]);
    }
}
