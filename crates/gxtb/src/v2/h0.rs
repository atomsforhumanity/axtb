//! One-electron Hamiltonian of g-xTB 2.0.1 (spec §4.5, §4.5.1, §4.6):
//!
//! - self-energies se_l = `ps_h0_selfenergy` - `ps_h0_selfenergy_cn` CN_A (raw g-xTB CN);
//! - off-diagonal (A != B): H_mn = 1/2 (se_m + se_n) K(l_m, l_n) pi_m pi_n S~_mn with
//!   K = 1/2 (k_l + k_l') (`pg_h0_kshell`, x 1.5 for d-f pairs), the distance polynomial
//!   pi_A,l = 1 + 2 kp_A F_l x + 0.0804696812 kp_A F_l R x, x = R/(rad_A + rad_B), and S~ the
//!   H0-basis overlap scaled in the diatomic frame by the harmonic means of
//!   `ps_h0_diat_scale` (sigma, pi, delta; blocks with f functions unscaled);
//! - onsite: H_mm = se_m;
//! - anisotropy: Delta_mn = -1/2 [a_B . D^B_mn + a_A . D^A_mn] with the dipole integrals D of the
//!   SCF basis and a_A = sum_B s_AB 1/2 [1 + erf(-1.5 (R_AB - R̄_AB))] (R_B - R_A),
//!   s_AB = 1/2 (`pa_h0_dip_scale`_A + `pa_h0_dip_scale`_B); its energy is sum_A a_A . d_A with
//!   the Mulliken atomic dipoles d_A of the density;
//! - ACP: H += S_pv^T diag(w) S_pv, normalized single-Gaussian projectors (`pa_nacp` per
//!   element, l = `pa_l_acp`, exponent `ps_acp_exp`, weight `ps_acp_level`).

use super::basis::{ao_offsets, atom_min_exponents, min_exponent, screened, INTCUT};
use super::Model;
use crate::acp::projector;
use crate::data_v2::{
    ACP_EXP, ACP_L, ACP_LEVEL, ACP_N, H0_DIAT_SCALE, H0_DIP_SCALE, H0_SELFENERGY, H0_SELFENERGY_CN, H0_SHPOLY2,
    PG_H0_KSHELL, PG_H0_SHPOLY2,
};
use crate::eeqbc::erf_f64;
use crate::h0::scale_diatomic_block;
use crate::parallel::map_items;
use axtb::basis::Cgto;
use axtb::integrals::{overlap, MBLK};
use axtb::molecule::{norm2_sq, sub3};
use axtb::Molecule;
use faer::linalg::matmul::triangular::BlockStructure;
use faer::{Accum, MatMut, MatRef, Par};

/// Coefficient of the R^2 term of the distance polynomial.
pub const SHPOLY4: f64 = 0.0804696812;
/// Steepness of the anisotropy damping.
pub const ANISO_STEEP: f64 = 1.5;
const SQRTPI: f64 = 1.772_453_850_905_516;

/// Shell factor K(l, l').
#[inline]
pub fn kshell(la: usize, lb: usize) -> f64 {
    let k = 0.5 * (PG_H0_KSHELL[la] + PG_H0_KSHELL[lb]);
    if (la == 2 && lb == 3) || (la == 3 && lb == 2) {
        k * 1.5
    } else {
        k
    }
}

/// Self-energy of shell `ish` of element `z` at g-xTB CN `cn` and its CN derivative.
#[inline]
pub fn selfenergy(z: usize, ish: usize, cn: f64) -> (f64, f64) {
    (H0_SELFENERGY[z - 1][ish] - H0_SELFENERGY_CN[z - 1][ish] * cn, -H0_SELFENERGY_CN[z - 1][ish])
}

/// Diatomic-frame scaling factors (sigma, pi, delta) of a pair: harmonic means, 0 if either
/// element has 0.
pub fn diat_scale(za: usize, zb: usize) -> [f64; 3] {
    let (a, b) = (H0_DIAT_SCALE[za - 1], H0_DIAT_SCALE[zb - 1]);
    [0, 1, 2].map(|k| 2.0 / (1.0 / a[k] + 1.0 / b[k]))
}

/// Distance polynomial factors (pi_A,l, pi_B,l') of a pair at distance `r` with
/// rad_A + rad_B = `radsum`, and their R derivatives.
#[inline]
pub fn shell_poly(z: usize, l: usize, r: f64, radsum: f64) -> (f64, f64) {
    let kp = H0_SHPOLY2[z - 1];
    let x = r / radsum;
    let s2 = 2.0 * kp * PG_H0_SHPOLY2[l];
    let s4 = SHPOLY4 * kp * PG_H0_SHPOLY2[l];
    (1.0 + s2 * x + s4 * r * x, s2 / radsum + 2.0 * s4 * r / radsum)
}

/// H0 (without anisotropy) from the H0-basis shells, the g-xTB CN and the integral pair
/// cutoff (bohr); full row-major nao x nao, axtb AO order. The lower blocks of each atom's rows
/// are one task (in parallel when `parallel`); the upper triangle is their mirror.
pub fn build_h0(
    model: &Model,
    mol: &Molecule,
    h0_shells: &[Vec<Cgto>],
    cn: &[f64],
    cutoff: f64,
    parallel: bool,
) -> Vec<f64> {
    let nat = mol.nat();
    let first = ao_offsets(h0_shells);
    let nao: usize = h0_shells.iter().flatten().map(|s| 2 * s.l + 1).sum();
    let mut h = vec![0.0; nao * nao];
    let mut se = vec![0.0; nao];
    for ia in 0..nat {
        for (is, s) in h0_shells[ia].iter().enumerate() {
            let e = selfenergy(mol.numbers[ia], is, cn[ia]).0;
            for a in 0..2 * s.l + 1 {
                let mu = first[ia][is] + a;
                se[mu] = e;
                h[mu * nao + mu] = e;
            }
        }
    }
    let amin = atom_min_exponents(h0_shells);
    let mut bounds: Vec<usize> = (0..nat).map(|a| first[a].first().copied().unwrap_or(0)).collect();
    bounds.push(nao);
    let cut2 = cutoff * cutoff;
    crate::parallel::for_row_blocks(&mut [&mut h], nao, &bounds, parallel, |ia, rows| {
        let rows = &mut rows[0];
        let row0 = bounds[ia];
        let za = mol.numbers[ia];
        let mut blk = [0.0; MBLK];
        for ib in 0..ia {
            let zb = mol.numbers[ib];
            let rab = sub3(mol.positions[ib], mol.positions[ia]);
            let r2 = norm2_sq(rab);
            if r2 > cut2 || screened(amin[ia], amin[ib], r2) {
                continue;
            }
            let r = r2.sqrt();
            let nv = [rab[0] / r, rab[1] / r, rab[2] / r];
            let k = diat_scale(za, zb);
            let radsum = model.h0_rad(za) + model.h0_rad(zb);
            let vec = sub3(mol.positions[ia], mol.positions[ib]);
            for (isa, sa) in h0_shells[ia].iter().enumerate() {
                let pia = shell_poly(za, sa.l, r, radsum).0;
                for (isb, sb) in h0_shells[ib].iter().enumerate() {
                    if screened(min_exponent(sa), min_exponent(sb), r2) {
                        continue;
                    }
                    let pib = shell_poly(zb, sb.l, r, radsum).0;
                    overlap(sb, sa, vec, INTCUT, &mut blk);
                    let (na, nb) = (2 * sa.l + 1, 2 * sb.l + 1);
                    let mut s: Vec<f64> = (0..na * nb).map(|x| blk[(x % nb) + nb * (x / nb)]).collect();
                    scale_diatomic_block(sa.l, sb.l, nv, k, &mut s);
                    let ks = kshell(sa.l, sb.l);
                    for a in 0..na {
                        let i = first[ia][isa] + a;
                        for b in 0..nb {
                            let j = first[ib][isb] + b;
                            rows[(i - row0) * nao + j] = 0.5 * (se[i] + se[j]) * ks * pia * pib * s[a * nb + b];
                        }
                    }
                }
            }
        }
    });
    h
}

/// Anisotropy damping of a pair and its R derivative: s_AB 1/2 [1 + erf(-1.5 (R - R̄))].
#[inline]
pub fn aniso_pair(model: &Model, za: usize, zb: usize, r: f64) -> (f64, f64) {
    let s = 0.5 * (H0_DIP_SCALE[za - 1] + H0_DIP_SCALE[zb - 1]);
    let arg = -ANISO_STEEP * (r - model.rvdw(za, zb));
    (s * 0.5 * (1.0 + erf_f64(arg)), s * (-ANISO_STEEP) * (-arg * arg).exp() / SQRTPI)
}

/// Anisotropy vectors a_A.
pub fn anisotropy(model: &Model, mol: &Molecule) -> Vec<[f64; 3]> {
    anisotropy_par(model, mol, false)
}

/// [`anisotropy`], atoms in parallel when `parallel`.
pub fn anisotropy_par(model: &Model, mol: &Molecule, parallel: bool) -> Vec<[f64; 3]> {
    let n = mol.nat();
    map_items(n, parallel, |i| {
        let mut a = [0.0; 3];
        for j in 0..n {
            if i == j {
                continue;
            }
            let v = sub3(mol.positions[j], mol.positions[i]);
            let r = norm2_sq(v).sqrt();
            let (f, _) = aniso_pair(model, mol.numbers[i], mol.numbers[j], r);
            for x in 0..3 {
                a[x] += f * v[x];
            }
        }
        a
    })
}

/// Adds sum_A va_A . da_A/dR to `g` for a weight `va` per atom (dE/da_A).
pub fn anisotropy_gradient(model: &Model, mol: &Molecule, va: &[[f64; 3]], g: &mut [[f64; 3]]) {
    anisotropy_gradient_par(model, mol, va, g, false);
}

/// [`anisotropy_gradient`], atoms in fixed chunks (in parallel when `parallel`).
pub fn anisotropy_gradient_par(model: &Model, mol: &Molecule, va: &[[f64; 3]], g: &mut [[f64; 3]], parallel: bool) {
    let n = mol.nat();
    let acc = crate::parallel::reduce_rows(
        n,
        parallel,
        || vec![[0.0; 3]; n],
        |i, g| {
            for j in 0..n {
                if i == j {
                    continue;
                }
                // a_i += f(r) (R_j - R_i)
                let v = sub3(mol.positions[j], mol.positions[i]);
                let r = norm2_sq(v).sqrt();
                let (f, df) = aniso_pair(model, mol.numbers[i], mol.numbers[j], r);
                let w = va[i];
                let wv = w[0] * v[0] + w[1] * v[1] + w[2] * v[2];
                // d/dR_j of f(r) v = f I + v (df/r) v^T
                for x in 0..3 {
                    let d = f * w[x] + wv * df * v[x] / r;
                    g[j][x] += d;
                    g[i][x] -= d;
                }
            }
        },
        |a, b| crate::parallel::add_into(a.as_flattened_mut(), b.as_flattened()),
    );
    crate::parallel::add_into(g.as_flattened_mut(), acc.as_flattened());
}

/// Adds the anisotropy `Delta_mn = -1/2 [a_A(n) . dp[n][m] + a_A(m) . dp[m][n]]` to the full
/// matrix `h` (the setup uses [`mirror_add_anisotropy`]).
pub fn add_anisotropy(nao: usize, ao_atom: &[usize], dpint: &[Vec<f64>; 3], a: &[[f64; 3]], h: &mut [f64]) {
    for i in 0..nao {
        let ai = ao_atom[i];
        for j in 0..nao {
            let aj = ao_atom[j];
            let (ij, ji) = (i * nao + j, j * nao + i);
            let mut t = 0.0;
            for x in 0..3 {
                t += a[ai][x] * dpint[x][ij] + a[aj][x] * dpint[x][ji];
            }
            h[ij] -= 0.5 * t;
        }
    }
}

/// Completes `h` from its lower triangle (the upper triangle is its mirror) and adds the
/// anisotropy: `Delta_mn = -1/2 [a_A(n) . dp[n][m] + a_A(m) . dp[m][n]]` (tiled, in parallel when
/// `parallel`).
pub fn mirror_add_anisotropy(
    nao: usize,
    ao_atom: &[usize],
    dpint: &[Vec<f64>; 3],
    a: &[[f64; 3]],
    h: &mut [f64],
    parallel: bool,
) {
    let low = h.to_vec();
    crate::parallel::fill_tiled(h, nao, parallel, |i, j| {
        let (ij, ji) = (i * nao + j, j * nao + i);
        let (ai, aj) = (ao_atom[i], ao_atom[j]);
        let mut t = 0.0;
        for x in 0..3 {
            t += a[ai][x] * dpint[x][ij] + a[aj][x] * dpint[x][ji];
        }
        (if j > i { low[ji] } else { low[ij] }) - 0.5 * t
    });
}

/// The ACP projectors of element `z`: (l, exponent, weight).
pub fn acp_projectors(z: usize) -> Vec<(usize, f64, f64)> {
    (0..ACP_N[z - 1]).map(|k| (ACP_L[z - 1][k], ACP_EXP[z - 1][k], ACP_LEVEL[z - 1][k])).collect()
}

/// The ACP projectors of a molecule with nonzero weight and the rows of their components in
/// the projector-overlap matrix.
pub struct AcpSet {
    /// (atom, normalized projector shell, weight, first row).
    pub list: Vec<(usize, Cgto, f64, usize)>,
    /// Number of rows (projector components).
    pub nrow: usize,
}

impl AcpSet {
    pub fn new(mol: &Molecule) -> Self {
        let mut list = Vec::new();
        let mut nrow = 0;
        for ip in 0..mol.nat() {
            for (l, alpha, w) in acp_projectors(mol.numbers[ip]) {
                if w != 0.0 {
                    list.push((ip, projector(l, alpha), w, nrow));
                    nrow += 2 * l + 1;
                }
            }
        }
        Self { list, nrow }
    }

    /// Weight of every row.
    pub fn row_weights(&self) -> Vec<f64> {
        self.list.iter().flat_map(|(_, p, w, _)| std::iter::repeat_n(*w, 2 * p.l + 1)).collect()
    }
}

/// Overlaps <p_r|mu> of every projector component with every SCF AO (row-major nrow x n), pairs
/// within `cutoff` (bohr); exactly zero blocks are skipped.
pub fn acp_overlaps(mol: &Molecule, shells: &[Vec<Cgto>], set: &AcpSet, cutoff: f64, parallel: bool) -> Vec<f64> {
    let first = ao_offsets(shells);
    let n: usize = shells.iter().flatten().map(|s| 2 * s.l + 1).sum();
    let amin = atom_min_exponents(shells);
    let cut2 = cutoff * cutoff;
    let blocks = map_items(set.list.len(), parallel, |k| {
        let (ip, ref proj, _, _) = set.list[k];
        let np = 2 * proj.l + 1;
        let mut out = vec![0.0; np * n];
        let mut blk = [0.0; MBLK];
        for (ia, sh) in shells.iter().enumerate() {
            let vec = sub3(mol.positions[ia], mol.positions[ip]);
            let r2 = norm2_sq(vec);
            if r2 > cut2 || screened(proj.alpha[0], amin[ia], r2) {
                continue;
            }
            for (is, s) in sh.iter().enumerate() {
                overlap(proj, s, vec, INTCUT, &mut blk);
                for a in 0..2 * s.l + 1 {
                    for m in 0..np {
                        out[m * n + first[ia][is] + a] = blk[m + np * a];
                    }
                }
            }
        }
        out
    });
    blocks.concat()
}

/// Adds H^ACP = S_p^T diag(w) S_p to the lower triangle of `h` (nao x nao) as one matrix
/// product.
pub fn add_acp(mol: &Molecule, shells: &[Vec<Cgto>], cutoff: f64, h: &mut [f64], par: Par, parallel: bool) {
    let n: usize = shells.iter().flatten().map(|s| 2 * s.l + 1).sum();
    let set = AcpSet::new(mol);
    if set.nrow == 0 {
        return;
    }
    let sp = acp_overlaps(mol, shells, &set, cutoff, parallel);
    let mut wsp = sp.clone();
    for (r, w) in set.row_weights().into_iter().enumerate() {
        wsp[r * n..(r + 1) * n].iter_mut().for_each(|x| *x *= w);
    }
    faer::linalg::matmul::triangular::matmul(
        MatMut::from_row_major_slice_mut(h, n, n),
        BlockStructure::TriangularLower,
        Accum::Add,
        MatRef::from_row_major_slice(&sp, set.nrow, n).transpose(),
        BlockStructure::Rectangular,
        MatRef::from_row_major_slice(&wsp, set.nrow, n),
        BlockStructure::Rectangular,
        1.0,
        par,
    );
}
