//! The layout of the self-consistent variables in one flat vector, as mixers and response
//! equations use them:
//!
//! ```text
//! [ q_sh (nsh) | μ_A (3 per atom) | Θ_A (6 per atom) | m_sh (nsh) ]
//! ```
//!
//! shell charges, atomic dipoles (x, y, z), traceless atomic quadrupoles (packed xx, xy, yy, xz,
//! yz, zz) and shell magnetizations (alpha minus beta population). The multipole blocks are
//! present for methods with anisotropic electrostatics, the magnetization block for
//! spin-polarized ones. tblite and xtb 6.8.0 scale the off-diagonal quadrupole components by
//! √2 in their mixer, so that the residual norm counts each off-diagonal element of the
//! symmetric tensor twice; the layout holds that factor (1 for none: multiplying and dividing
//! by 1 are exact).

/// Whether each packed quadrupole component (xx, xy, yy, xz, yz, zz) is an off-diagonal one.
pub const QUAD_OFFDIAG: [bool; 6] = [false, true, false, true, true, false];

/// Index layout of the self-consistent variables of one molecule.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ScfVariables {
    /// Shells and atoms.
    pub nsh: usize,
    pub nat: usize,
    /// Whether the atomic dipoles and quadrupoles are variables.
    pub multipoles: bool,
    /// Whether the shell magnetizations are variables.
    pub magnetization: bool,
    /// Factor of the off-diagonal quadrupole components in the packed vector (1: unscaled).
    pub quad_offdiag_scale: f64,
}

impl ScfVariables {
    /// The layout of `nsh` shell charges, the multipoles of `nat` atoms when `multipoles`, and
    /// `nsh` magnetizations when `magnetization`; quadrupoles unscaled.
    pub fn new(nsh: usize, nat: usize, multipoles: bool, magnetization: bool) -> Self {
        Self { nsh, nat, multipoles, magnetization, quad_offdiag_scale: 1.0 }
    }

    /// The same layout with the off-diagonal quadrupole components multiplied by `scale` in
    /// the packed vector.
    pub fn with_quad_offdiag_scale(self, scale: f64) -> Self {
        Self { quad_offdiag_scale: scale, ..self }
    }

    /// Length of the packed vector.
    pub fn dim(&self) -> usize {
        self.mag_offset() + if self.magnetization { self.nsh } else { 0 }
    }

    /// Index of dipole component `c` of atom `a`.
    #[inline]
    pub fn dip(&self, a: usize, c: usize) -> usize {
        self.nsh + 3 * a + c
    }

    /// Index of packed quadrupole component `c` of atom `a`.
    #[inline]
    pub fn quad(&self, a: usize, c: usize) -> usize {
        self.nsh + 3 * self.nat + 6 * a + c
    }

    /// Index of the magnetization of shell `ish`.
    #[inline]
    pub fn mag(&self, ish: usize) -> usize {
        self.mag_offset() + ish
    }

    fn mag_offset(&self) -> usize {
        self.nsh + if self.multipoles { 9 * self.nat } else { 0 }
    }

    /// Factor of packed quadrupole component `c`.
    #[inline]
    fn quad_scale(&self, c: usize) -> f64 {
        if QUAD_OFFDIAG[c] {
            self.quad_offdiag_scale
        } else {
            1.0
        }
    }

    /// Pack shell values, atomic dipole and quadrupole values (read when the layout has
    /// multipoles) and shell magnetization values (when it has magnetizations; `None`: zeros)
    /// into `out` (length [`ScfVariables::dim`]).
    pub fn pack(&self, qsh: &[f64], dip: &[[f64; 3]], quad: &[[f64; 6]], mag: Option<&[f64]>, out: &mut [f64]) {
        out[..self.nsh].copy_from_slice(qsh);
        if self.multipoles {
            for a in 0..self.nat {
                for c in 0..3 {
                    out[self.dip(a, c)] = dip[a][c];
                }
                for c in 0..6 {
                    out[self.quad(a, c)] = quad[a][c] * self.quad_scale(c);
                }
            }
        }
        if self.magnetization {
            let o = self.mag_offset();
            match mag {
                Some(m) => out[o..o + self.nsh].copy_from_slice(m),
                None => out[o..o + self.nsh].fill(0.0),
            }
        }
    }

    /// Unpack a vector into shell values, atomic dipoles and quadrupoles (zeros without
    /// multipoles) and, when given and the layout has them, magnetizations.
    pub fn unpack(
        &self,
        v: &[f64],
        qsh: &mut [f64],
        dip: &mut [[f64; 3]],
        quad: &mut [[f64; 6]],
        mag: Option<&mut [f64]>,
    ) {
        qsh.copy_from_slice(&v[..self.nsh]);
        for a in 0..self.nat {
            for c in 0..3 {
                dip[a][c] = if self.multipoles { v[self.dip(a, c)] } else { 0.0 };
            }
            for c in 0..6 {
                quad[a][c] = if self.multipoles { v[self.quad(a, c)] / self.quad_scale(c) } else { 0.0 };
            }
        }
        if let (true, Some(m)) = (self.magnetization, mag) {
            let o = self.mag_offset();
            m.copy_from_slice(&v[o..o + self.nsh]);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pack_unpack_round_trip() {
        let lay = ScfVariables::new(3, 2, true, true).with_quad_offdiag_scale(2f64.sqrt());
        assert_eq!(lay.dim(), 3 + 18 + 3);
        let q = [0.1, -0.2, 0.3];
        let d = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let t = [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]];
        let m = [0.5, 0.0, -0.5];
        let mut v = vec![0.0; lay.dim()];
        lay.pack(&q, &d, &t, Some(&m), &mut v);
        assert_eq!(v[lay.dip(1, 2)], 6.0);
        assert_eq!(v[lay.quad(0, 1)], 2.0 * 2f64.sqrt());
        assert_eq!(v[lay.quad(0, 2)], 3.0);
        assert_eq!(v[lay.mag(2)], -0.5);
        let (mut q2, mut d2, mut t2, mut m2) = ([0.0; 3], [[0.0; 3]; 2], [[0.0; 6]; 2], [0.0; 3]);
        lay.unpack(&v, &mut q2, &mut d2, &mut t2, Some(&mut m2));
        assert_eq!((q2, d2, m2), (q, d, m));
        for a in 0..2 {
            for c in 0..6 {
                assert!((t2[a][c] - t[a][c]).abs() <= 1e-15 * t[a][c].abs());
            }
        }
        let plain = ScfVariables::new(3, 2, false, false);
        assert_eq!(plain.dim(), 3);
        let mut v = vec![0.0; 3];
        plain.pack(&q, &d, &t, None, &mut v);
        plain.unpack(&v, &mut q2, &mut d2, &mut t2, None);
        assert_eq!((q2, d2, t2), (q, [[0.0; 3]; 2], [[0.0; 6]; 2]));
    }
}
