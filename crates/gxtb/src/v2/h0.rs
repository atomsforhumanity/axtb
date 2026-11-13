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
//!
//! v1.1 (the 1.1.0 program's `scf_`, `sintm_diat_`,
//! `setupcbas_`) has the same structure with its parameters: levels -(lev_l - lev_cn_l CN),
//! K(l, l') = 1/2 (K_l + K_l') with K = (glob(1), glob(2), 1.4, 2.85) and no d-f factor, the
//! linear polynomial 1 + poly_A F_l R / ((rcov_A + rcov_B)/2) (F = (1, glob(3), 2, 2)), the
//! diatomic scaling by the harmonic means of h0k (sigma, pi) and 5 (delta), primitive
//! screening at 20, no anisotropy, and one ACP projector per l = s..f per element with the
//! CN-dependent weight acp_coeff_l (1 + acpcn sqrt(CN)), pairs within 40 bohr.

use super::Model;
use crate::data_v2::{
    ACP_EXP, ACP_L, ACP_LEVEL, ACP_N, H0_DIAT_SCALE, H0_DIP_SCALE, H0_SELFENERGY, H0_SELFENERGY_CN, H0_SHPOLY2,
    PG_H0_KSHELL, PG_H0_SHPOLY2,
};
use crate::eeqbc::erf_f64;
use crate::options::Version;
use axtb::ao::ShellPairs;
use axtb::basis::{Basis as AoBasis, Cgto, MAXPRIM};
use axtb::integrals::{overlap, MBLK};
use axtb::molecule::{norm2_sq, sub3};
use axtb::Molecule;
use axtb_base::math::primitive_norm;
use axtb_base::threads;
use faer::linalg::matmul::triangular::BlockStructure;
use faer::{Accum, MatMut, MatRef, Par};

/// Coefficient of the R^2 term of the distance polynomial.
pub const SHPOLY4: f64 = 0.0804696812;
/// Steepness of the anisotropy damping.
pub const ANISO_STEEP: f64 = 1.5;
const SQRTPI: f64 = 1.772_453_850_905_516;

/// The v1.1 parameters of element `z`.
#[inline]
fn v1_1(z: usize) -> &'static crate::params::ElementParams {
    crate::params::gxtb_v1_1().element(z).expect("g-xTB v1.1 parameters of a checked element")
}

/// Shell factor K(l, l') (v1.1: the mean of (glob(1), glob(2), 1.4, 2.85), no d-f factor).
#[inline]
pub fn kshell(model: &Model, la: usize, lb: usize) -> f64 {
    if model.version == Version::V1_1 {
        let g = &crate::params::gxtb_v1_1().glob;
        let kw = [g[0], g[1], 1.4, 2.85];
        return 0.5 * (kw[la] + kw[lb]);
    }
    let k = 0.5 * (PG_H0_KSHELL[la] + PG_H0_KSHELL[lb]);
    if (la == 2 && lb == 3) || (la == 3 && lb == 2) {
        k * 1.5
    } else {
        k
    }
}

/// Self-energy of shell `ish` of element `z` at g-xTB CN `cn` and its CN derivative (v1.1:
/// -(lev - lev_cn CN), the same bits as (-lev) - (-lev_cn) CN).
#[inline]
pub fn selfenergy(model: &Model, z: usize, ish: usize, cn: f64) -> (f64, f64) {
    if model.version == Version::V1_1 {
        let e = v1_1(z);
        return (-e.lev[ish] - -e.lev_cn[ish] * cn, e.lev_cn[ish]);
    }
    (H0_SELFENERGY[z - 1][ish] - H0_SELFENERGY_CN[z - 1][ish] * cn, -H0_SELFENERGY_CN[z - 1][ish])
}

/// Diatomic-frame scaling factors (sigma, pi, delta) of a pair: harmonic means, 0 if either
/// element has 0 (v1.1: of h0k and of 5 for delta).
pub fn diat_scale(model: &Model, za: usize, zb: usize) -> [f64; 3] {
    let (a, b) = match model.version {
        Version::V2_0_1 => (H0_DIAT_SCALE[za - 1], H0_DIAT_SCALE[zb - 1]),
        Version::V1_1 => {
            let (ea, eb) = (v1_1(za), v1_1(zb));
            ([ea.h0k[0], ea.h0k[1], K_DELTA], [eb.h0k[0], eb.h0k[1], K_DELTA])
        }
    };
    [0, 1, 2].map(|k| 2.0 / (1.0 / a[k] + 1.0 / b[k]))
}

/// Distance polynomial factors (pi_A,l, pi_B,l') of a pair at distance `r` with
/// rad_A + rad_B = `radsum`, and their R derivatives (v1.1: the linear polynomial
/// 1 + poly_A F_l R / Rc with Rc = radsum/2, the program's expression).
#[inline]
pub fn shell_poly(model: &Model, z: usize, l: usize, r: f64, radsum: f64) -> (f64, f64) {
    if model.version == Version::V1_1 {
        let rc = 0.5 * radsum;
        let c = v1_1(z).poly * [1.0, crate::params::gxtb_v1_1().glob[2], 2.0, 2.0][l];
        return (1.0 + c * r / rc, c / rc);
    }
    let kp = H0_SHPOLY2[z - 1];
    let x = r / radsum;
    let s2 = 2.0 * kp * PG_H0_SHPOLY2[l];
    let s4 = SHPOLY4 * kp * PG_H0_SHPOLY2[l];
    (1.0 + s2 * x + s4 * r * x, s2 / radsum + 2.0 * s4 * r / radsum)
}

/// H0 (without anisotropy) from the H0 basis, the g-xTB CN and the integral pair cutoff
/// (bohr); full row-major nao x nao, axtb AO order. The lower blocks of each atom's rows are one
/// task (in parallel when `parallel`); the upper triangle is their mirror.
pub fn build_h0(model: &Model, mol: &Molecule, h0: &AoBasis, cn: &[f64], cutoff: f64, parallel: bool) -> Vec<f64> {
    let nao = h0.nao;
    let mut h = vec![0.0; nao * nao];
    let mut se = vec![0.0; nao];
    for (ia, is, s) in h0.shell_iter() {
        let e = selfenergy(model, mol.numbers[ia], is, cn[ia]).0;
        let first = h0.iao_sh[h0.ish_at[ia] + is];
        for mu in first..first + 2 * s.l + 1 {
            se[mu] = e;
            h[mu * nao + mu] = e;
        }
    }
    let mut bounds = h0.iao_at.clone();
    bounds.push(nao);
    let intcut = model.intcut_h0();
    let pairs = ShellPairs::screened(mol, h0, cutoff, intcut);
    crate::parallel::for_row_blocks(&mut [&mut h], nao, &bounds, parallel, |ia, rows| {
        let rows = &mut rows[0];
        let row0 = bounds[ia];
        let za = mol.numbers[ia];
        let mut blk = [0.0; MBLK];
        for ib in 0..ia {
            let Some(p) = pairs.pair(ia, ib) else { continue };
            let zb = mol.numbers[ib];
            let rab = sub3(mol.positions[ib], mol.positions[ia]);
            let r = p.r2.sqrt();
            let nv = [rab[0] / r, rab[1] / r, rab[2] / r];
            let k = diat_scale(model, za, zb);
            let radsum = model.h0_rad(za) + model.h0_rad(zb);
            pairs.for_each(&p, |sp| {
                let (sa, sb) = (sp.ci, sp.cj);
                let pia = shell_poly(model, za, sa.l, r, radsum).0;
                let pib = shell_poly(model, zb, sb.l, r, radsum).0;
                overlap(sb, sa, p.vec, intcut, &mut blk);
                let (na, nb) = (2 * sa.l + 1, 2 * sb.l + 1);
                let mut s: Vec<f64> = (0..na * nb).map(|x| blk[(x % nb) + nb * (x / nb)]).collect();
                scale_diatomic_block(sa.l, sb.l, nv, k, &mut s);
                let ks = kshell(model, sa.l, sb.l);
                for a in 0..na {
                    let i = sp.ii + a;
                    for b in 0..nb {
                        let j = sp.jj + b;
                        rows[(i - row0) * nao + j] = 0.5 * (se[i] + se[j]) * ks * pia * pib * s[a * nb + b];
                    }
                }
            });
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
    threads::map(n, parallel, |i| {
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

/// The ACP projectors of element `z` (2.0.1): (l, exponent, weight).
pub fn acp_projectors(z: usize) -> Vec<(usize, f64, f64)> {
    (0..ACP_N[z - 1]).map(|k| (ACP_L[z - 1][k], ACP_EXP[z - 1][k], ACP_LEVEL[z - 1][k])).collect()
}

/// v1.1's ACP pair cutoff (bohr; the program skips R^2 > 1600).
pub const ACP_CUTOFF_V1_1: f64 = 40.0;

/// The ACP projectors of a molecule with nonzero weight and the rows of their components in
/// the projector-overlap matrix.
pub struct AcpSet {
    /// (atom, normalized projector shell, weight, first row).
    pub list: Vec<(usize, Cgto, f64, usize)>,
    /// The CN derivative of every weight (v1.1; 2.0.1's weights do not depend on the CN: zero).
    pub dw: Vec<f64>,
    /// Number of rows (projector components).
    pub nrow: usize,
    /// The pair cutoff (bohr) and primitive screening of the projector overlaps.
    pub cutoff: f64,
    pub intcut: f64,
    /// The version of the weights.
    pub version: Version,
}

impl AcpSet {
    /// The projectors of `mol` in `model` at the g-xTB CN `cn`, with the integral pair cutoff
    /// `cutoff` of the setup (v1.1: its own 40 bohr).
    pub fn new(model: &Model, mol: &Molecule, cn: &[f64], cutoff: f64) -> Self {
        let mut list = Vec::new();
        let mut dw = Vec::new();
        let mut nrow = 0;
        match model.version {
            Version::V2_0_1 => {
                for ip in 0..mol.nat() {
                    for (l, alpha, w) in acp_projectors(mol.numbers[ip]) {
                        if w != 0.0 {
                            list.push((ip, projector(l, alpha), w, nrow));
                            dw.push(0.0);
                            nrow += 2 * l + 1;
                        }
                    }
                }
                Self { list, dw, nrow, cutoff, intcut: model.intcut_h0(), version: model.version }
            }
            Version::V1_1 => {
                // `setupcbas_`: w = acp_coeff_l (1 + acpcn sqrt(CN)), one projector per l
                for ip in 0..mol.nat() {
                    let e = v1_1(mol.numbers[ip]);
                    let sqcn = cn[ip].sqrt();
                    let f = 1.0 + e.acpcn * sqcn;
                    for l in 0..4 {
                        let w = e.acp_coeff[l] * f;
                        if w != 0.0 {
                            list.push((ip, projector(l, e.acp_exp[l]), w, nrow));
                            dw.push(if sqcn > 0.0 { e.acp_coeff[l] * e.acpcn / (2.0 * sqcn) } else { 0.0 });
                            nrow += 2 * l + 1;
                        }
                    }
                }
                Self { list, dw, nrow, cutoff: ACP_CUTOFF_V1_1, intcut: model.intcut_h0(), version: model.version }
            }
        }
    }

    /// Weight of every row.
    pub fn row_weights(&self) -> Vec<f64> {
        self.list.iter().flat_map(|(_, p, w, _)| std::iter::repeat_n(*w, 2 * p.l + 1)).collect()
    }
}

/// Overlaps <p_r|mu> of every projector component with every AO of the SCF basis `bas`
/// (row-major nrow x n), pairs within the set's cutoff (bohr); screened blocks, which are
/// exactly zero, are skipped ([`ShellPairs::screened`]).
pub fn acp_overlaps(mol: &Molecule, bas: &AoBasis, set: &AcpSet, parallel: bool) -> Vec<f64> {
    let n = bas.nao;
    let intcut = set.intcut;
    let pairs = ShellPairs::screened(mol, bas, set.cutoff, intcut);
    let blocks = threads::map(set.list.len(), parallel, |k| {
        let (ip, ref proj, _, _) = set.list[k];
        let np = 2 * proj.l + 1;
        let mut out = vec![0.0; np * n];
        let mut blk = [0.0; MBLK];
        for ia in 0..mol.nat() {
            let Some(p) = pairs.pair_with(ia, ip, proj.min_alpha()) else { continue };
            pairs.for_each_against(&p, proj, |sp| {
                overlap(proj, sp.ci, p.vec, intcut, &mut blk);
                for a in 0..2 * sp.ci.l + 1 {
                    for m in 0..np {
                        out[m * n + sp.ii + a] = blk[m + np * a];
                    }
                }
            });
        }
        out
    });
    blocks.concat()
}

/// Basis size up to which v1.1's ACP product runs as plain loops: the first product of faer's
/// matrix multiplication detects the CPU's caches (~0.45 ms, once per process), which the small
/// products of such a basis otherwise never pay (they take faer's small-matrix kernels).
const ACP_LOOPS_MAX_AO: usize = 16;

/// Adds H^ACP = S_p^T diag(w) S_p of the projectors `set` to the lower triangle of `h`
/// (nao x nao) as one matrix product (v1.1 with at most `ACP_LOOPS_MAX_AO` = 16 basis functions:
/// as loops over the rows).
pub fn add_acp(mol: &Molecule, bas: &AoBasis, set: &AcpSet, h: &mut [f64], par: Par, parallel: bool) {
    let n = bas.nao;
    if set.nrow == 0 {
        return;
    }
    let sp = acp_overlaps(mol, bas, set, parallel);
    let mut wsp = sp.clone();
    for (r, w) in set.row_weights().into_iter().enumerate() {
        wsp[r * n..(r + 1) * n].iter_mut().for_each(|x| *x *= w);
    }
    if set.version == Version::V1_1 && n <= ACP_LOOPS_MAX_AO {
        for r in 0..set.nrow {
            let (s, ws) = (&sp[r * n..(r + 1) * n], &wsp[r * n..(r + 1) * n]);
            for i in 0..n {
                for j in 0..=i {
                    h[i * n + j] += s[i] * ws[j];
                }
            }
        }
        return;
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

// ---- shared with v1.1's H0 (from the old tree's h0.rs and acp.rs) ----

/// Primitive-pair screening of the program's overlap code (`xtb_intgrad::get_overlap`, as
/// called from `sintm_diat_`): pairs with a_i a_j R^2/(a_i + a_j) > 20 are skipped.
pub const INTCUT_V1_1: f64 = 20.0;

/// Delta-bond scaling of the diatomic-frame overlap.
pub const K_DELTA: f64 = 5.0;

/// Normalized projector shell.
pub fn projector(l: usize, alpha: f64) -> Cgto {
    let mut c = Cgto { l, nprim: 1, alpha: [0.0; MAXPRIM], coeff: [0.0; MAXPRIM] };
    c.alpha[0] = alpha;
    c.coeff[0] = primitive_norm(l, alpha);
    c
}

/// Projectors onto the bond-frame |m| = 0, 1, 2 subspaces of a shell with angular momentum
/// `l` <= 2, in axtb's real spherical-harmonic order, for bond direction `n` (unit vector).
/// Returned as `proj[mu][a * nsph + b]`.
fn bond_projectors(l: usize, n: [f64; 3]) -> [Vec<f64>; 3] {
    let ns = 2 * l + 1;
    let mut proj = [vec![0.0; ns * ns], vec![0.0; ns * ns], vec![0.0; ns * ns]];
    if l == 0 {
        proj[0][0] = 1.0;
        return proj;
    }
    // orthonormal frame (e1, e2, n)
    let t = if n[0].abs() < 0.9 { [1.0, 0.0, 0.0] } else { [0.0, 1.0, 0.0] };
    let d = t[0] * n[0] + t[1] * n[1] + t[2] * n[2];
    let mut e1 = [t[0] - d * n[0], t[1] - d * n[1], t[2] - d * n[2]];
    let ne = norm2_sq(e1).sqrt();
    e1 = [e1[0] / ne, e1[1] / ne, e1[2] / ne];
    let e2 = [n[1] * e1[2] - n[2] * e1[1], n[2] * e1[0] - n[0] * e1[2], n[0] * e1[1] - n[1] * e1[0]];
    let frame = [e1, e2, n]; // bond-frame x, y, z axes in lab coordinates
                             // u[m_bond][m_lab]: lab components of the bond-frame functions
    let mut u = vec![0.0; ns * ns];
    let mu_of: Vec<usize> = match l {
        1 => {
            // axtb order y, z, x -> axis index 1, 2, 0
            let axis = [1usize, 2, 0];
            for (mb, &ab) in axis.iter().enumerate() {
                for (ml, &al) in axis.iter().enumerate() {
                    u[mb * ns + ml] = frame[ab][al];
                }
            }
            vec![1, 0, 1]
        }
        2 => {
            let s3 = 3f64.sqrt() / 2.0;
            // quadratic forms of the axtb d functions (m = -2..2), Frobenius norm^2 = 3/2
            let mut q = [[[0.0; 3]; 3]; 5];
            q[0][0][1] = s3;
            q[0][1][0] = s3;
            q[1][1][2] = s3;
            q[1][2][1] = s3;
            q[2][0][0] = -0.5;
            q[2][1][1] = -0.5;
            q[2][2][2] = 1.0;
            q[3][0][2] = s3;
            q[3][2][0] = s3;
            q[4][0][0] = s3;
            q[4][1][1] = -s3;
            for mb in 0..5 {
                // bond-frame form in lab coordinates: sum_ab q[a][b] frame[a] (x) frame[b]
                let mut lab = [[0.0; 3]; 3];
                for a in 0..3 {
                    for b in 0..3 {
                        if q[mb][a][b] != 0.0 {
                            for x in 0..3 {
                                for y in 0..3 {
                                    lab[x][y] += q[mb][a][b] * frame[a][x] * frame[b][y];
                                }
                            }
                        }
                    }
                }
                for ml in 0..5 {
                    let mut dot = 0.0;
                    for x in 0..3 {
                        for y in 0..3 {
                            dot += q[ml][x][y] * lab[x][y];
                        }
                    }
                    u[mb * ns + ml] = dot / 1.5;
                }
            }
            vec![2, 1, 0, 1, 2]
        }
        _ => unreachable!(),
    };
    for (mb, &mu) in mu_of.iter().enumerate() {
        for a in 0..ns {
            for b in 0..ns {
                proj[mu][a * ns + b] += u[mb * ns + a] * u[mb * ns + b];
            }
        }
    }
    proj
}

/// Quadratic forms of axtb's real d functions (m = -2..2: xy, yz, z2, xz, x2-y2), Frobenius
/// norm^2 3/2.
fn d_forms() -> [[[f64; 3]; 3]; 5] {
    let s3 = 3f64.sqrt() / 2.0;
    let mut q = [[[0.0; 3]; 3]; 5];
    q[0][0][1] = s3;
    q[0][1][0] = s3;
    q[1][1][2] = s3;
    q[1][2][1] = s3;
    q[2][0][0] = -0.5;
    q[2][1][1] = -0.5;
    q[2][2][2] = 1.0;
    q[3][0][2] = s3;
    q[3][2][0] = s3;
    q[4][0][0] = s3;
    q[4][1][1] = -s3;
    q
}

/// Bond-frame projectors P_mu (|m| = mu = 0, 1, 2) of a shell with l <= 2 for the unit bond
/// vector `n` in closed form, and their derivatives dP_mu/dn_k (`n` treated as a free vector;
/// contract with the tangential dn). Row-major `(2l+1) x (2l+1)`, axtb order.
/// p: P_0 = n n^T, P_1 = 1 - P_0. d (forms q_m): P_0 = a a^T with a_m = n^T q_m n,
/// (P_1)_{mm'} = (2 (q_m n).(q_m' n) - 2 a_m a_m')/(3/2), P_2 = 1 - P_0 - P_1.
#[allow(clippy::type_complexity)]
pub fn bond_projectors_deriv(l: usize, n: [f64; 3]) -> ([Vec<f64>; 3], [[Vec<f64>; 3]; 3]) {
    let ns = 2 * l + 1;
    let z = || vec![0.0; ns * ns];
    let mut p = [z(), z(), z()];
    let mut dp = [[z(), z(), z()], [z(), z(), z()], [z(), z(), z()]];
    match l {
        0 => p[0][0] = 1.0,
        1 => {
            let ax = [1usize, 2, 0];
            for a in 0..3 {
                for b in 0..3 {
                    let v = n[ax[a]] * n[ax[b]];
                    p[0][a * 3 + b] = v;
                    p[1][a * 3 + b] = if a == b { 1.0 } else { 0.0 } - v;
                    for k in 0..3 {
                        let d = if ax[a] == k { n[ax[b]] } else { 0.0 } + if ax[b] == k { n[ax[a]] } else { 0.0 };
                        dp[0][k][a * 3 + b] = d;
                        dp[1][k][a * 3 + b] = -d;
                    }
                }
            }
        }
        2 => {
            let q = d_forms();
            let mv = |m: &[[f64; 3]; 3], v: [f64; 3]| -> [f64; 3] {
                [0, 1, 2].map(|x| m[x][0] * v[0] + m[x][1] * v[1] + m[x][2] * v[2])
            };
            let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
            let b: Vec<[f64; 3]> = (0..5).map(|m| mv(&q[m], n)).collect();
            let a: Vec<f64> = (0..5).map(|m| dot(n, b[m])).collect();
            // q_m b_m' (for the derivative of b_m . b_m')
            let qb: Vec<Vec<[f64; 3]>> = (0..5).map(|m| (0..5).map(|mp| mv(&q[m], b[mp])).collect()).collect();
            for m in 0..5 {
                for mp in 0..5 {
                    let k0 = a[m] * a[mp];
                    let k1 = (2.0 * dot(b[m], b[mp]) - 2.0 * k0) / 1.5;
                    let idx = m * 5 + mp;
                    p[0][idx] = k0;
                    p[1][idx] = k1;
                    p[2][idx] = if m == mp { 1.0 } else { 0.0 } - k0 - k1;
                    for k in 0..3 {
                        let (da, dap) = (2.0 * b[m][k], 2.0 * b[mp][k]);
                        let d0 = da * a[mp] + a[m] * dap;
                        let d1 = (2.0 * (qb[m][mp][k] + qb[mp][m][k]) - 2.0 * d0) / 1.5;
                        dp[0][k][idx] = d0;
                        dp[1][k][idx] = d1;
                        dp[2][k][idx] = -d0 - d1;
                    }
                }
            }
        }
        _ => panic!("bond projectors for l = {l}"),
    }
    (p, dp)
}

/// Scale an overlap block `s` (rows: shell of angular momentum `la` on A, columns: `lb` on
/// B, row-major) in the diatomic frame of the bond A -> B.
pub fn scale_diatomic_block(la: usize, lb: usize, n: [f64; 3], k: [f64; 3], s: &mut [f64]) {
    if la > 2 || lb > 2 {
        return;
    }
    let (na, nb) = (2 * la + 1, 2 * lb + 1);
    let pa = bond_projectors(la, n);
    let pb = bond_projectors(lb, n);
    let mut out = vec![0.0; na * nb];
    for mu in 0..=la.min(lb) {
        // out += k_mu * Pa S Pb
        let mut tmp = vec![0.0; na * nb];
        for i in 0..na {
            for j in 0..nb {
                tmp[i * nb + j] = (0..na).map(|a| pa[mu][i * na + a] * s[a * nb + j]).sum();
            }
        }
        for i in 0..na {
            for j in 0..nb {
                out[i * nb + j] += k[mu] * (0..nb).map(|b| tmp[i * nb + b] * pb[mu][b * nb + j]).sum::<f64>();
            }
        }
    }
    s.copy_from_slice(&out);
}
