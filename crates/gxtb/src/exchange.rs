// SPDX-License-Identifier: LGPL-3.0-or-later
//! Exchange of g-xTB 2.0.1 (spec §4.13, §4.13.1, §4.14): Mulliken-approximated range-separated
//! exchange (MFX), the bond-order correction and the Mulliken-form onsite exchange with
//! charge-dependent integrals. For a closed shell with the total density P (open shell:
//! E = sum_s 2 E(P^s), with the total shell charges in Lambda), with A o B the elementwise
//! product and (PS o SP)_mn = (PS)_mn (PS)_nm:
//!
//! E = sum [A1 o PS o SP] + sum [A2 o P o SPS] + sum [A3 o PS o PS]
//!     - 1/16 [diag(P) Lambda diag(SPS) + diag(SP) Lambda diag(SP)],
//! A1 = -1/8 G + 1/2 K + 1/4 R, A2 = -1/8 G + 1/4 R - 1/16 Lambda, A3 = -1/16 Lambda,
//!
//! - G: MFX kernel per shell pair (expanded to AOs),
//!   (0.15 + 0.85 erf(0.2 R)) / (R + exp(-(kx1 + kx2 R0_AB) R)/(eta g)), eta the general average
//!   of U^x = `ps_fock_shell_hubbard` `pa_hubbard_parameter` with xi = max(xi_l, xi_l'),
//!   g = sqrt(G_l G_l') (1.3826597204 for the same shell), R = 0 within an atom,
//!   R0_AB = R^D3_AB (bohr) 1/2 (s_A + s_B) aatoau;
//! - K: bond-order correction per atom pair (zero within an atom),
//!   K_AB = 1/2 (c_A + c_B) 1/2 [1 + erf(-aatoau (R - sqrt(r_A r_B))/R0_AB)];
//! - Lambda: onsite integrals of AO pairs on one atom,
//!   0.15 Lambda0_ll'(Z) [1 - 1/2 (kq_l q_l + kq_l' q_l')] with the Mulliken shell charges q;
//! - R: r_s = Lambda_ll(s)/(2 (2l+1)) on the whole block of a shell with l >= 1.
//!
//! The Fock matrix is dE/dP at fixed kernels plus the Mulliken shell potential dE/dq of the
//! charge dependence (the program's, exact).
//!
//! v1.1 (the 1.1.0 program's `setfock_mat_`, `setfock_indo_`): the same
//! MFX with its kernel ((0.15 + 0.85 erf(omega R))/(R + exp(-(glob(7) + glob(8) R0) R)/(eta g)),
//! omega = glob(19), R0 the D3 pair radius in Ångström used as bohr, eta the general average of
//! fam_l gam_Z with xi = 1 or 2 (polarization shells), g = 1.39 for the same shell, R = 1e-12 bohr
//! within an atom), no bond-order correction, and the dual-density onsite exchange
//! E = -1/2 t(P) per E(P) above, t = sum_{k != l} L_kl Pt_kl^2 + sum_{shells, l >= 1} f_kl L_kl
//! Pt_kk Pt_ll with Pt = (PS + SP)/2, L = 0.15 gmunu(l_k, l_l): in the A-matrix form
//! A1 += -L/4 and A3 += -L/4 on the same-atom pairs k != l, and -1/16 diag(SP)^T (8 f L)
//! diag(SP) on the shell blocks ([`Dual`]); nothing depends on the charges.

use crate::coulomb::spin_index;
use crate::coulomb::Shell;
use crate::data::v2_0_1::{
    FOCK_AVG_EXP, FOCK_CRAD, FOCK_CSCALE, FOCK_SHELL_HUBBARD, HUBBARD, ONECX, PG_FOCK_KQ, PG_FOCK_OFFDIAG,
};
use crate::dense::matmul;
use crate::eeqbc::erf_f64;
use crate::model::{Model, SQRTPI};
use crate::options::Version;
use crate::parallel::TILE;
use axtb::molecule::{norm2_sq, sub3};
use axtb::terms::{Chain, Geometry};
use axtb::Molecule;
use faer::{Accum, MatMut, MatRef, Par};
use rayon::prelude::*;

pub const ALPHA: f64 = 0.15;
pub const OMEGA: f64 = 0.2;
pub const LR: f64 = 0.85;
pub const SAME_SHELL: f64 = 1.3826597204;
pub const KX1: f64 = -0.2960502355;
/// k^x2 per Ångström of the pair radius (the program stores 0.0349817377 * autoaa).
pub const KX2_AA: f64 = 0.0349817377;

/// General average 2^(xi-1) (x y)^(xi/2) / (x + y)^(xi-1), evaluated as the program does.
#[inline]
pub fn general_average(x: f64, y: f64, xi: f64) -> f64 {
    (x * y).powf(xi * 0.5) * (2.0 / (x + y)).powf(xi - 1.0)
}

/// R0_AB of the exchange kernels (bohr).
#[inline]
pub fn r0(model: &Model, za: usize, zb: usize) -> f64 {
    model.rvdw(za, zb) * model.aatoau
}

/// MFX kernel of 2.0.1 at distance `r` (0 within an atom) and its R derivative.
#[inline]
pub fn mfx_kernel(r: f64, kap: f64, etag: f64) -> (f64, f64) {
    mfx_kernel_w(r, kap, etag, OMEGA)
}

/// The MFX kernel (0.15 + 0.85 erf(omega R))/(R + exp(-kappa R)/(eta g)) with the range
/// separation `omega` (2.0.1: 0.2; v1.1: glob(19)) and its R derivative.
#[inline]
pub fn mfx_kernel_w(r: f64, kap: f64, etag: f64, omega: f64) -> (f64, f64) {
    let num = ALPHA + LR * erf_f64(omega * r);
    let damp = (-kap * r).exp();
    let den = r + damp / etag;
    let dnum = LR * 2.0 * omega / SQRTPI * (-(omega * r) * (omega * r)).exp();
    let dden = 1.0 - kap * damp / etag;
    (num / den, (dnum * den - num * dden) / (den * den))
}

/// Bond-order kernel K_AB at distance `r` and its R derivative.
#[inline]
pub fn bo_kernel(model: &Model, za: usize, zb: usize, r: f64) -> (f64, f64) {
    let c = 0.5 * (FOCK_CSCALE[za - 1] + FOCK_CSCALE[zb - 1]);
    let rr = (FOCK_CRAD[za - 1] * FOCK_CRAD[zb - 1]).sqrt();
    let r0 = r0(model, za, zb);
    let arg = -model.aatoau * (r - rr) / r0;
    (c * 0.5 * (1.0 + erf_f64(arg)), c * (-model.aatoau / r0) * (-arg * arg).exp() / SQRTPI)
}

/// Exchange kernels of one geometry.
#[derive(Clone, Debug)]
pub struct Exchange {
    pub nao: usize,
    pub nat: usize,
    pub ao_shell: Vec<usize>,
    pub ao_atom: Vec<usize>,
    pub shells: Vec<Shell>,
    /// MFX shell kernel (nsh x nsh) and its AO expansion.
    pub gx_sh: Vec<f64>,
    pub gx: Vec<f64>,
    /// Bond-order kernel (nat x nat) and its AO expansion.
    pub kbo: Vec<f64>,
    pub kbo_ao: Vec<f64>,
    /// The charge-independent dense part of A1: -G/8 + K/2 (AO pairs).
    pub k1: Vec<f64>,
    /// 0.15 Lambda0 of same-atom AO pairs (nao x nao, zero between atoms).
    pub lam0: Vec<f64>,
    /// kq of every AO.
    pub kq: Vec<f64>,
    /// AO range [first, end) of every atom (an atom's AOs are contiguous).
    pub atom_ao: Vec<(usize, usize)>,
    /// The range separation of the MFX kernel ([`mfx_kernel_w`]) and whether the bond-order
    /// correction is part of the model (2.0.1).
    pub omega: f64,
    pub bond_order: bool,
    /// v1.1's onsite exchange (its `lam0` and `kq` are zero: no Mulliken-form onsite part).
    pub dual: Option<Dual>,
}

/// v1.1's dual-density onsite exchange in the A-matrix form of [`Exchange`] (per E(P)): A1 and
/// A3 get -`lam_off`/4 on the same-atom AO pairs k != l, and the energy
/// -1/16 diag(SP)^T `m2` diag(SP) with `m2` = 8 f L on the shell blocks (l >= 1), f_kl =
/// ((2l+1) delta_kl - 1)/(2l+1). Row-major nao x nao, zero between atoms.
#[derive(Clone, Debug)]
pub struct Dual {
    pub lam_off: Vec<f64>,
    pub m2: Vec<f64>,
}

impl Dual {
    /// The onsite integrals of the AOs `ao` ((atom, shell, l)) of the elements `numbers`.
    pub fn new(numbers: &[usize], ao: &[Shell]) -> Self {
        let n = ao.len();
        let mut lam_off = vec![0.0; n * n];
        let mut m2 = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let ((ai, si, li), (aj, sj, lj)) = (ao[i], ao[j]);
                if ai != aj {
                    continue;
                }
                let Some(k) = gmunu_index(li, lj) else { continue };
                let lam = OFX_SCALE * crate::data::v1_1::GMUNU[(k - 1) + 9 * (numbers[ai] - 1)];
                if i != j {
                    lam_off[i * n + j] = lam;
                }
                if si == sj && li >= 1 {
                    let f = (if i == j { (2 * li + 1) as f64 } else { 0.0 } - 1.0) / (2 * li + 1) as f64;
                    m2[i * n + j] = 8.0 * f * lam;
                }
            }
        }
        Self { lam_off, m2 }
    }
}

/// Per-shell exchange parameters: (U^x, xi, G_l).
fn shell_params(mol: &Molecule, sh: &Shell) -> (f64, f64, f64) {
    let (iat, ish, l) = *sh;
    let z = mol.numbers[iat] - 1;
    (FOCK_SHELL_HUBBARD[z][ish] * HUBBARD[z], FOCK_AVG_EXP[z][ish], PG_FOCK_OFFDIAG[l])
}

/// Shell-pair data of the MFX kernel: (kappa, eta g) for shells i, j.
pub fn mfx_pair(model: &Model, mol: &Molecule, shells: &[Shell], i: usize, j: usize) -> (f64, f64) {
    if model.version == Version::V1_1 {
        return mfx_pair_v1_1(mol, shells, i, j);
    }
    let (ui, xi_i, gi) = shell_params(mol, &shells[i]);
    let (uj, xi_j, gj) = shell_params(mol, &shells[j]);
    let eta = general_average(ui, uj, xi_i.max(xi_j));
    let same = shells[i].0 == shells[j].0;
    let g = if i == j { SAME_SHELL } else { (gi * gj).sqrt() };
    let kap = if same {
        0.0
    } else {
        KX1 + KX2_AA / model.aatoau * r0(model, mol.numbers[shells[i].0], mol.numbers[shells[j].0])
    };
    (kap, eta * g)
}

/// v1.1's shell-pair data of the MFX kernel (`setgab_lrao_`): kappa = glob(7) + glob(8) R0_AB
/// (R0 the D3 pair radius in Ångström, also within an atom) and eta g with the program's general
/// average 2^(xi-1) (U U')^(xi/2) / (U + U')^(xi-1).
fn mfx_pair_v1_1(mol: &Molecule, shells: &[Shell], i: usize, j: usize) -> (f64, f64) {
    use crate::data::v1_1::{BAS_POL, GAM};
    let p = crate::params::gxtb_v1_1();
    let g = &p.glob;
    let ((ia, ish, li), (ja, jsh, lj)) = (shells[i], shells[j]);
    let (zi, zj) = (mol.numbers[ia], mol.numbers[ja]);
    let ux = |z: usize, s: usize| p.element(z).expect("g-xTB v1.1 parameters of a checked element").fam[s] * GAM[z - 1];
    let (ui, uj) = (ux(zi, ish), ux(zj, jsh));
    let xi = (BAS_POL[ish * 103 + zi - 1].max(BAS_POL[jsh * 103 + zj - 1]) + 1) as f64;
    let eta = (xi - 1.0).exp2() * (ui * uj).powf(0.5 * xi) / (ui + uj).powf(xi - 1.0);
    let gl = [g[4], g[5], g[5], g[5]];
    let gg = if i == j { G_ONSITE } else { (gl[li] * gl[lj]).sqrt() };
    let r0 = crate::eeqbc::vdw_rad_pair(zi, zj);
    (g[6] + g[7] * r0, eta * gg)
}

impl Exchange {
    /// The exchange kernels of a geometry, kernel rows in parallel when `parallel` (same result).
    pub fn new(model: &Model, mol: &Molecule, shells: &[Shell], ao_shell: &[usize], parallel: bool) -> Self {
        let nsh = shells.len();
        let nao = ao_shell.len();
        let nat = mol.nat();
        let ao_atom: Vec<usize> = ao_shell.iter().map(|&s| shells[s].0).collect();
        let v1_1 = model.version == Version::V1_1;
        // v1.1: the program's distance of an atom from itself, its range separation
        let (r_same, omega) = if v1_1 { (R_SAME_ATOM, crate::params::gxtb_v1_1().glob[18]) } else { (0.0, OMEGA) };
        let [gx_sh] = crate::parallel::symmetric(nsh, parallel, |i, j| {
            let (ia, ja) = (shells[i].0, shells[j].0);
            let r = if ia == ja { r_same } else { norm2_sq(sub3(mol.positions[ia], mol.positions[ja])).sqrt() };
            let (kap, etag) = mfx_pair(model, mol, shells, i, j);
            [mfx_kernel_w(r, kap, etag, omega).0]
        });
        let [kbo] = crate::parallel::symmetric(nat, parallel, |i, j| {
            if i == j || v1_1 {
                return [0.0];
            }
            let r = norm2_sq(sub3(mol.positions[i], mol.positions[j])).sqrt();
            [bo_kernel(model, mol.numbers[i], mol.numbers[j], r).0]
        });
        let mut gx = vec![0.0; nao * nao];
        let mut kbo_ao = vec![0.0; nao * nao];
        let mut lam0 = vec![0.0; nao * nao];
        {
            let bounds: Vec<usize> = (0..=nao).collect();
            crate::parallel::for_row_blocks(
                &mut [&mut gx, &mut kbo_ao, &mut lam0],
                nao,
                &bounds,
                parallel,
                |m, rows| {
                    for n in 0..nao {
                        let (sm, sn) = (ao_shell[m], ao_shell[n]);
                        rows[0][n] = gx_sh[sm * nsh + sn];
                        let (am, an) = (ao_atom[m], ao_atom[n]);
                        rows[1][n] = kbo[am * nat + an];
                        if am == an && !v1_1 {
                            let z = mol.numbers[am] - 1;
                            rows[2][n] = ALPHA * ONECX[z][spin_index(shells[sm].2, shells[sn].2)];
                        }
                    }
                },
            );
        }
        let kq = ao_shell.iter().map(|&s| if v1_1 { 0.0 } else { PG_FOCK_KQ[shells[s].2] }).collect();
        let mut atom_ao = vec![(0, 0); nat];
        for (m, &a) in ao_atom.iter().enumerate() {
            if m == 0 || ao_atom[m - 1] != a {
                assert!(atom_ao[a].1 == 0, "the AOs of atom {a} are not contiguous");
                atom_ao[a].0 = m;
            }
            atom_ao[a].1 = m + 1;
        }
        let k1 = gx.iter().zip(&kbo_ao).map(|(g, k)| -0.125 * g + 0.5 * k).collect();
        let dual = v1_1.then(|| {
            let ao: Vec<Shell> = ao_shell.iter().map(|&s| shells[s]).collect();
            Dual::new(&mol.numbers, &ao)
        });
        Self {
            nao,
            nat,
            ao_shell: ao_shell.to_vec(),
            ao_atom,
            shells: shells.to_vec(),
            gx_sh,
            gx,
            kbo,
            kbo_ao,
            k1,
            lam0,
            kq,
            atom_ao,
            omega,
            bond_order: !v1_1,
            dual,
        }
    }

    /// The MFX kernel of this model at distance `r` for the pair data `kap`, `etag` of
    /// [`mfx_pair`], and its R derivative.
    #[inline]
    pub fn kernel(&self, r: f64, kap: f64, etag: f64) -> (f64, f64) {
        mfx_kernel_w(r, kap, etag, self.omega)
    }

    /// Onsite integrals Lambda(q) (nao x nao) and the shell-block coefficients r per AO
    /// (0 for s shells) at shell charges `q`.
    pub fn onsite(&self, q: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = self.nao;
        let mut lam = vec![0.0; n * n];
        let mut r = vec![0.0; n];
        for m in 0..n {
            let fm = self.kq[m] * q[self.ao_shell[m]];
            for k in 0..n {
                let l0 = self.lam0[m * n + k];
                if l0 != 0.0 {
                    lam[m * n + k] = l0 * (1.0 - 0.5 * (fm + self.kq[k] * q[self.ao_shell[k]]));
                }
            }
            let l = self.shells[self.ao_shell[m]].2;
            if l >= 1 {
                r[m] = self.lam0[m * n + m] * (1.0 - fm) / (2.0 * (2 * l + 1) as f64);
            }
        }
        (lam, r)
    }

    /// Kernel matrices A1, A2, A3 of the onsite integrals `lam` and coefficients `r`.
    fn kernels(&self, lam: &[f64], r: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
        let n = self.nao;
        let mut a1 = vec![0.0; n * n];
        let mut a2 = vec![0.0; n * n];
        let mut a3 = vec![0.0; n * n];
        for m in 0..n {
            for k in 0..n {
                let mk = m * n + k;
                let rr = if self.ao_shell[m] == self.ao_shell[k] { 0.25 * r[m] } else { 0.0 };
                a1[mk] = -0.125 * self.gx[mk] + 0.5 * self.kbo_ao[mk] + rr;
                a2[mk] = -0.125 * self.gx[mk] + rr - lam[mk] / 16.0;
                a3[mk] = -lam[mk] / 16.0;
            }
        }
        if let Some(d) = &self.dual {
            for (mk, &lo) in d.lam_off.iter().enumerate() {
                if lo != 0.0 {
                    a1[mk] += -0.25 * lo;
                    a3[mk] += -0.25 * lo;
                }
            }
        }
        (a1, a2, a3)
    }

    /// Add the shell potential dE/dq of the charge dependence of Lambda and r on the atom block
    /// of AOs [i0, i1) to `v`, from dE/dLambda (`dlam(m, k)`) and dE/dr (`dr(m)`) of its AOs:
    /// dLambda_mk/dq_s(m) = -1/2 Lambda0_mk kq_m (and likewise for k), dr_m/dq_s(m) =
    /// -Lambda0_mm kq_m / (2 (2l + 1)). Lambda0 lives on the atom blocks only.
    fn add_block_shell_potential(
        &self,
        (i0, i1): (usize, usize),
        dlam: impl Fn(usize, usize) -> f64,
        dr: impl Fn(usize) -> f64,
        v: &mut [f64],
    ) {
        let n = self.nao;
        for m in i0..i1 {
            let sm = self.ao_shell[m];
            for k in i0..i1 {
                let l0 = self.lam0[m * n + k];
                if l0 != 0.0 {
                    v[sm] -= 0.5 * l0 * self.kq[m] * (dlam(m, k) + dlam(k, m));
                }
            }
            let l = self.shells[sm].2;
            if l >= 1 {
                v[sm] -= dr(m) * self.lam0[m * n + m] * self.kq[m] / (2.0 * (2 * l + 1) as f64);
            }
        }
    }

    /// The products P S and S P S of every spin density `p` (closed shell: the total density).
    pub fn products(&self, s: &[f64], p: &[Vec<f64>], par: Par) -> Vec<Products> {
        let n = self.nao;
        p.iter()
            .map(|pm| {
                let ps = matmul(n, pm, s, par);
                let sps = matmul(n, s, &ps, par);
                Products { ps, sps }
            })
            .collect()
    }

    /// kq q_l per AO and the shell-block coefficients r per AO (0 for s shells) at shell
    /// charges `q`: Lambda_mk = Lambda0_mk [1 - (f_m + f_k)/2] within an atom.
    fn onsite_factors(&self, q: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = self.nao;
        let f: Vec<f64> = (0..n).map(|m| self.kq[m] * q[self.ao_shell[m]]).collect();
        let r = (0..n)
            .map(|m| {
                let l = self.shells[self.ao_shell[m]].2;
                if l >= 1 {
                    self.lam0[m * n + m] * (1.0 - f[m]) / (2.0 * (2 * l + 1) as f64)
                } else {
                    0.0
                }
            })
            .collect();
        (f, r)
    }

    /// The parts of one spin channel (closed shell: the total density) with products `pr`:
    /// energy E(P) at fixed kernels, the matrices B = A1 o (PS)^T + A3 o PS and
    /// X = A2 o P - D(Lambda diag P)/16 (with `mats`), y2 = Lambda diag(PS), y3 = Lambda diag(SPS)
    /// and the shell potential dE/dq of the charge dependence of Lambda and r. The dense kernels
    /// (A1 = K1 + ..., A2 = -G/8 + ...) are read from `k1` and `gx`; the charge-dependent parts
    /// live on the atom blocks.
    fn channel(&self, p: &[f64], pr: &Products, f: &[f64], r: &[f64], mats: bool, parallel: bool) -> ChannelPass {
        let n = self.nao;
        let (ps, sps) = (&pr.ps, &pr.sps);
        let (k1, gx) = (&self.k1, &self.gx);
        let ntile = n.div_ceil(TILE);
        // dense part, row tiles (column tiles inside for the transposed reads)
        let mut b = if mats { vec![0.0; n * n] } else { Vec::new() };
        let mut x = if mats { vec![0.0; n * n] } else { Vec::new() };
        let dense = |t: usize, b: &mut [f64], x: &mut [f64]| -> f64 {
            let i0 = t * TILE;
            let i1 = (i0 + TILE).min(n);
            let mut e = 0.0;
            for j0 in (0..n).step_by(TILE) {
                let j1 = (j0 + TILE).min(n);
                for m in i0..i1 {
                    let mut em = 0.0;
                    for k in j0..j1 {
                        let mk = m * n + k;
                        let pkm = ps[k * n + m];
                        let a2 = -0.125 * gx[mk];
                        em += k1[mk] * (ps[mk] * pkm) + a2 * (p[mk] * sps[mk]);
                        if mats {
                            let o = (m - i0) * n + k;
                            b[o] = k1[mk] * pkm;
                            x[o] = a2 * p[mk];
                        }
                    }
                    e += em;
                }
            }
            e
        };
        let parts: Vec<f64> = if mats {
            let rows = TILE * n;
            if parallel {
                b.par_chunks_mut(rows)
                    .zip(x.par_chunks_mut(rows))
                    .enumerate()
                    .map(|(t, (bt, xt))| dense(t, bt, xt))
                    .collect()
            } else {
                b.chunks_mut(rows).zip(x.chunks_mut(rows)).enumerate().map(|(t, (bt, xt))| dense(t, bt, xt)).collect()
            }
        } else {
            axtb_base::threads::map(ntile, parallel, |t| dense(t, &mut [], &mut []))
        };
        let mut e: f64 = parts.iter().sum();
        // atom blocks: the onsite (Lambda) and same-shell (r) parts of the kernels and their
        // charge derivatives
        let nsh = self.shells.len();
        let mut v = vec![0.0; nsh];
        let mut y2 = vec![0.0; n];
        let mut y3 = vec![0.0; n];
        let mut dlam = Vec::new();
        let mut dr = Vec::new();
        if let Some(d) = &self.dual {
            // v1.1: A1 and A3 on the same-atom pairs, the diagonal coupling of diag(SP)
            let mut e_on = 0.0;
            for &(i0, i1) in &self.atom_ao {
                for m in i0..i1 {
                    let mut y2m = 0.0;
                    for k in i0..i1 {
                        let mk = m * n + k;
                        let lo = d.lam_off[mk];
                        if lo != 0.0 {
                            let a = -0.25 * lo;
                            let pkm = ps[k * n + m];
                            e_on += a * (ps[mk] * pkm) + a * (ps[mk] * ps[mk]);
                            if mats {
                                b[mk] += a * pkm + a * ps[mk];
                            }
                        }
                        let l2 = d.m2[mk];
                        if l2 != 0.0 {
                            y2m += l2 * ps[k * n + k];
                        }
                    }
                    e_on -= ps[m * n + m] * y2m / 16.0;
                    y2[m] = y2m;
                }
            }
            return ChannelPass { e, e_onsite: e_on, b, x, y2, y3, v };
        }
        for &(i0, i1) in &self.atom_ao {
            let na = i1 - i0;
            dlam.clear();
            dlam.resize(na * na, 0.0);
            dr.clear();
            dr.resize(na, 0.0);
            for m in i0..i1 {
                let sm = self.ao_shell[m];
                let (dpm, dspm) = (p[m * n + m], ps[m * n + m]);
                let (mut y1m, mut y2m, mut y3m) = (0.0, 0.0, 0.0);
                for k in i0..i1 {
                    let mk = m * n + k;
                    let l0 = self.lam0[mk];
                    let lam = if l0 != 0.0 { l0 * (1.0 - 0.5 * (f[m] + f[k])) } else { 0.0 };
                    let same = self.ao_shell[k] == sm;
                    let rr = if same { 0.25 * r[m] } else { 0.0 };
                    let pkm = ps[k * n + m];
                    let t1 = ps[mk] * pkm;
                    let t2 = p[mk] * sps[mk];
                    let t3 = ps[mk] * ps[mk];
                    let a3 = -lam / 16.0;
                    let a2 = rr + a3;
                    e += rr * t1 + a2 * t2 + a3 * t3;
                    if mats {
                        b[mk] += rr * pkm + a3 * ps[mk];
                        x[mk] += a2 * p[mk];
                    }
                    if l0 != 0.0 {
                        let kk = k * n + k;
                        y1m += lam * p[kk];
                        y2m += lam * ps[kk];
                        y3m += lam * sps[kk];
                        dlam[(m - i0) * na + k - i0] = -(t2 + t3 + dpm * sps[kk] + dspm * ps[kk]) / 16.0;
                    }
                    if same {
                        dr[m - i0] += 0.25 * (t1 + t2);
                    }
                }
                e -= (dpm * y3m + dspm * y2m) / 16.0;
                if mats {
                    x[m * n + m] -= y1m / 16.0;
                }
                y2[m] = y2m;
                y3[m] = y3m;
            }
            self.add_block_shell_potential((i0, i1), |m, k| dlam[(m - i0) * na + k - i0], |m| dr[m - i0], &mut v);
        }
        ChannelPass { e, e_onsite: 0.0, b, x, y2, y3, v }
    }

    /// The exchange Fock matrix of a channel from its [`ChannelPass`] (with matrices):
    /// F = Z S + (Z S)^T + A2 o SPS - 1/16 [D(y3) + D(y2) S + S D(y2)], Z = B + S X/2 (X
    /// symmetric: S X S = (S X/2) S + its transpose), scaled by `w`.
    fn channel_fock(
        &self,
        s: &[f64],
        f: &[f64],
        r: &[f64],
        pr: &Products,
        cp: ChannelPass,
        w: f64,
        par: Par,
        parallel: bool,
    ) -> Vec<f64> {
        let n = self.nao;
        let ChannelPass { b: mut z, x, y2, y3, .. } = cp;
        let sx = matmul(n, s, &x, par);
        drop(x);
        for (a, b) in z.iter_mut().zip(&sx) {
            *a += 0.5 * b;
        }
        drop(sx);
        let zs = matmul(n, &z, s, par);
        drop(z);
        let sps = &pr.sps;
        let gx = &self.gx;
        let mut out = vec![0.0; n * n];
        axtb_base::threads::for_slices(&mut out, TILE * n, parallel, |t, rows| {
            let i0 = t * TILE;
            let i1 = (i0 + TILE).min(n);
            for j0 in (0..n).step_by(TILE) {
                let j1 = (j0 + TILE).min(n);
                for m in i0..i1 {
                    for k in j0..j1 {
                        let mk = m * n + k;
                        rows[(m - i0) * n + k] =
                            zs[mk] + zs[k * n + m] - 0.125 * gx[mk] * sps[mk] - (y2[m] + y2[k]) * s[mk] / 16.0;
                    }
                }
            }
        });
        // atom blocks: the onsite and same-shell parts of A2 (none in v1.1)
        for &(i0, i1) in self.atom_ao.iter().filter(|_| self.dual.is_none()) {
            for m in i0..i1 {
                let sm = self.ao_shell[m];
                for k in i0..i1 {
                    let mk = m * n + k;
                    let l0 = self.lam0[mk];
                    let lam = if l0 != 0.0 { l0 * (1.0 - 0.5 * (f[m] + f[k])) } else { 0.0 };
                    let rr = if self.ao_shell[k] == sm { 0.25 * r[m] } else { 0.0 };
                    out[mk] += (rr - lam / 16.0) * sps[mk];
                }
                out[m * n + m] -= y3[m] / 16.0;
            }
        }
        if w != 1.0 {
            out.iter_mut().for_each(|v| *v *= w);
        }
        out
    }

    /// Energy, Fock matrices dE/dP^s at fixed charges (closed shell: one, with the total
    /// density) and the shell potential dE/dq of spin densities `p` with shell charges `q` and
    /// their [`Exchange::products`] `pr`: matrix products on `par`, the other loops on the
    /// current rayon pool when `parallel` (fixed tiles summed in order).
    #[allow(clippy::type_complexity)]
    pub fn evaluate_with(
        &self,
        s: &[f64],
        p: &[Vec<f64>],
        pr: &[Products],
        q: &[f64],
        fock: bool,
        par: Par,
        parallel: bool,
    ) -> (f64, Vec<Vec<f64>>, Vec<f64>) {
        let (e, _, fm, v) = self.evaluate_parts(s, p, pr, q, fock, par, parallel);
        (e, fm, v)
    }

    /// [`Exchange::evaluate_with`] with v1.1's onsite exchange apart: (E_MFX (2.0.1: the whole
    /// exchange), E_onsite (v1.1; 0 for 2.0.1), Fock matrices, shell potential).
    #[allow(clippy::type_complexity, clippy::too_many_arguments)]
    pub fn evaluate_parts(
        &self,
        s: &[f64],
        p: &[Vec<f64>],
        pr: &[Products],
        q: &[f64],
        fock: bool,
        par: Par,
        parallel: bool,
    ) -> (f64, f64, Vec<Vec<f64>>, Vec<f64>) {
        let (f, r) = self.onsite_factors(q);
        let w = if p.len() == 1 { 1.0 } else { 2.0 };
        let mut e = 0.0;
        let mut e_on = 0.0;
        let mut fm = Vec::new();
        let mut v = vec![0.0; self.shells.len()];
        for (pm, prm) in p.iter().zip(pr) {
            let cp = self.channel(pm, prm, &f, &r, fock, parallel);
            e += w * cp.e;
            e_on += w * cp.e_onsite;
            for (a, b) in v.iter_mut().zip(&cp.v) {
                *a += w * b;
            }
            if fock {
                fm.push(self.channel_fock(s, &f, &r, prm, cp, w, par, parallel));
            }
        }
        (e, e_on, fm, v)
    }

    /// dE/dG (shell pairs, nsh x nsh) and dE/dK (atom pairs, nat x nat) of the exchange at fixed
    /// spin densities `p` with products `pr`: sums of (PS o SP) and (P o SPS) over the AO blocks
    /// of each shell pair (shell rows in parallel when `parallel`).
    pub fn kernel_weights_with(&self, p: &[Vec<f64>], pr: &[Products], parallel: bool) -> (Vec<f64>, Vec<f64>) {
        let n = self.nao;
        let nsh = self.shells.len();
        let nat = self.nat;
        let w = if p.len() == 1 { 1.0 } else { 2.0 };
        let mut sr = vec![(0, 0); nsh];
        for (m, &sh) in self.ao_shell.iter().enumerate() {
            if sr[sh].1 == 0 {
                sr[sh].0 = m;
            }
            sr[sh].1 = m + 1;
        }
        let mut tsh = vec![0.0; nsh * nsh];
        let mut tat = vec![0.0; nat * nat];
        for (pm, prm) in p.iter().zip(pr) {
            let (ps, sps) = (&prm.ps, &prm.sps);
            let rows: Vec<Vec<(f64, f64)>> = axtb_base::threads::map(nsh, parallel, |i| {
                let (i0, i1) = sr[i];
                sr.iter()
                    .map(|&(j0, j1)| {
                        let (mut s1, mut s2) = (0.0, 0.0);
                        for m in i0..i1 {
                            for k in j0..j1 {
                                let mk = m * n + k;
                                s1 += ps[mk] * ps[k * n + m];
                                s2 += pm[mk] * sps[mk];
                            }
                        }
                        (s1, s2)
                    })
                    .collect()
            });
            for (i, row) in rows.iter().enumerate() {
                let ai = self.shells[i].0;
                for (j, &(s1, s2)) in row.iter().enumerate() {
                    tsh[i * nsh + j] -= w * 0.125 * (s1 + s2);
                    tat[ai * nat + self.shells[j].0] += w * 0.5 * s1;
                }
            }
        }
        (tsh, tat)
    }

    /// dE/dS (symmetric) of the exchange at fixed kernels and fixed Lambda, r for spin
    /// densities `p` (closed shell: total density) with shell charges `q`.
    pub fn overlap_weight(&self, s: &[f64], p: &[Vec<f64>], q: &[f64], par: Par) -> Vec<f64> {
        let pr = self.products(s, p, par);
        self.overlap_weight_with(p, &pr, q, par, !matches!(par, Par::Seq))
    }

    /// [`Exchange::overlap_weight`] with the [`Exchange::products`] of `p`:
    /// sum_s w [P B + (P B)^T + X SP + (X SP)^T - 1/16 (D(y2) P + P D(y2))] with B, X, y2 of
    /// `Exchange::channel` (B = A1 o SP + A3 o PS, X = A2 o P - D(y1)/16), symmetrized.
    pub fn overlap_weight_with(
        &self,
        p: &[Vec<f64>],
        pr: &[Products],
        q: &[f64],
        par: Par,
        parallel: bool,
    ) -> Vec<f64> {
        let n = self.nao;
        let (f, r) = self.onsite_factors(q);
        let w = if p.len() == 1 { 1.0 } else { 2.0 };
        let mut out = vec![0.0; n * n];
        for (pm, prm) in p.iter().zip(pr) {
            let cp = self.channel(pm, prm, &f, &r, true, parallel);
            let pb = matmul(n, pm, &cp.b, par);
            // X (S P) with S P = (P S)^T
            let mut csp = vec![0.0; n * n];
            faer::linalg::matmul::matmul(
                MatMut::from_row_major_slice_mut(&mut csp, n, n),
                Accum::Replace,
                MatRef::from_row_major_slice(&cp.x, n, n),
                MatRef::from_row_major_slice(&prm.ps, n, n).transpose(),
                1.0,
                par,
            );
            let y2 = &cp.y2;
            axtb_base::threads::for_slices(&mut out, TILE * n, parallel, |t, rows| {
                let i0 = t * TILE;
                let i1 = (i0 + TILE).min(n);
                for j0 in (0..n).step_by(TILE) {
                    let j1 = (j0 + TILE).min(n);
                    for m in i0..i1 {
                        for k in j0..j1 {
                            let (mk, km) = (m * n + k, k * n + m);
                            let v = pb[mk] + pb[km] + csp[mk] + csp[km] - (y2[m] + y2[k]) * (pm[mk] + pm[km]) / 32.0;
                            rows[(m - i0) * n + k] += w * v;
                        }
                    }
                }
            });
        }
        out
    }
}

impl Exchange {
    /// Add the gradient of the exchange kernels (MFX shell kernel, bond-order kernel) at fixed
    /// weights dE/dG (`tsh`, shell pairs) and dE/dK (`tat`, atom pairs; from
    /// [`Exchange::kernel_weights_with`]) to `ch.g`: shell rows, then atom rows, each in fixed
    /// chunks (in parallel with `geo.parallel`) summed in order.
    pub fn add_kernel_gradient(&self, model: &Model, geo: &Geometry<'_>, tsh: &[f64], tat: &[f64], ch: &mut Chain) {
        use crate::gradient::{add_pair, chain_rows};
        let mol = geo.mol;
        let nsh = self.shells.len();
        let nat = mol.nat();
        let shells = &self.shells;
        let acc = chain_rows(nsh, ch, geo.parallel, |i, ch| {
            for j in 0..i {
                let (ia, ja) = (shells[i].0, shells[j].0);
                if ia == ja {
                    continue;
                }
                let v = sub3(mol.positions[ia], mol.positions[ja]);
                let r = norm2_sq(v).sqrt();
                let (kap, etag) = mfx_pair(model, mol, shells, i, j);
                let dg = self.kernel(r, kap, etag).1;
                let w = tsh[i * nsh + j] + tsh[j * nsh + i];
                add_pair(&mut ch.g, ia, ja, w * dg / r, v);
            }
        });
        ch.add(&acc);
        if !self.bond_order {
            return;
        }
        let acc = chain_rows(nat, ch, geo.parallel, |i, ch| {
            for j in 0..i {
                let v = sub3(mol.positions[i], mol.positions[j]);
                let r = norm2_sq(v).sqrt();
                let dk = bo_kernel(model, mol.numbers[i], mol.numbers[j], r).1;
                let w = tat[i * nat + j] + tat[j * nat + i];
                add_pair(&mut ch.g, i, j, w * dk / r, v);
            }
        });
        ch.add(&acc);
    }
}

/// P S and S P S of one spin density (row-major n x n).
#[derive(Clone, Debug)]
pub struct Products {
    pub ps: Vec<f64>,
    pub sps: Vec<f64>,
}

/// The per-channel parts of the exchange of [`Exchange::channel`].
struct ChannelPass {
    e: f64,
    /// v1.1: the onsite exchange (not in `e`).
    e_onsite: f64,
    b: Vec<f64>,
    x: Vec<f64>,
    y2: Vec<f64>,
    y3: Vec<f64>,
    v: Vec<f64>,
}

/// Accumulators of a change of the closed-shell exchange Fock matrix dE_closed/dP:
/// B = sum A1 o (PS)^T + A3 o PS, X = sum A2 o P - D(y1)/16, the elementwise part
/// sum A2 o SPS and the diagonals y2 = Lambda diag(PS), y3 = Lambda diag(SPS).
struct FockAcc {
    b: Vec<f64>,
    x: Vec<f64>,
    f: Vec<f64>,
    y2: Vec<f64>,
    y3: Vec<f64>,
}

/// Exchange data of a converged density that the linear response reuses: Lambda(q), r(q), the
/// kernels and the products P S, S P S of every spin density.
#[derive(Clone, Debug)]
pub struct ResponseBase {
    pub lam: Vec<f64>,
    /// v1.1: the diagonal coupling `m2` of [`Dual`] (its y2 = m2 diag(PS); 2.0.1 uses `lam`).
    pub lam2: Option<Vec<f64>>,
    pub a1: Vec<f64>,
    pub a2: Vec<f64>,
    pub a3: Vec<f64>,
    pub ps: Vec<Vec<f64>>,
    pub sps: Vec<Vec<f64>>,
    /// Per channel: B = A1 o SP + A3 o PS and X S with X = A2 o P - D(Lambda diag P)/16 (the
    /// Fock matrix is B S + (B S)^T + A2 o SPS + S X S - ...).
    pub b: Vec<Vec<f64>>,
    pub xs: Vec<Vec<f64>>,
}

/// Overlap derivatives of the exchange along one overlap direction dS at fixed densities,
/// kernels and charges: Fock matrices, shell potential dE/dq, dE/dG and dE/dK weights (as
/// [`crate::gradient::exchange_kernel_weights`]) and dE/dS.
pub struct OverlapTangent {
    pub f: Vec<Vec<f64>>,
    pub vx: Vec<f64>,
    pub tsh: Vec<f64>,
    pub tat: Vec<f64>,
    pub psi: Vec<f64>,
}

impl Exchange {
    /// The data of spin densities `p` with shell charges `q` that [`Exchange::response`] reuses.
    pub fn response_base(&self, s: &[f64], p: &[Vec<f64>], q: &[f64], par: Par) -> ResponseBase {
        let n = self.nao;
        let (lam, r) = self.onsite(q);
        let (a1, a2, a3) = self.kernels(&lam, &r);
        let ps: Vec<Vec<f64>> = p.iter().map(|pm| matmul(n, pm, s, par)).collect();
        let sps: Vec<Vec<f64>> = ps.iter().map(|x| matmul(n, s, x, par)).collect();
        let mut bs = Vec::with_capacity(p.len());
        let mut xs = Vec::with_capacity(p.len());
        for (pm, psm) in p.iter().zip(&ps) {
            let mut b = vec![0.0; n * n];
            let mut x = vec![0.0; n * n];
            let mut y1 = vec![0.0; n];
            for m in 0..n {
                for k in 0..n {
                    let mk = m * n + k;
                    if self.lam0[mk] != 0.0 {
                        y1[m] += lam[mk] * pm[k * n + k];
                    }
                    b[mk] = a1[mk] * psm[k * n + m] + a3[mk] * psm[mk];
                    x[mk] = a2[mk] * pm[mk];
                }
            }
            for m in 0..n {
                x[m * n + m] -= y1[m] / 16.0;
            }
            xs.push(matmul(n, &x, s, par));
            bs.push(b);
        }
        let lam2 = self.dual.as_ref().map(|d| d.m2.clone());
        ResponseBase { lam, lam2, a1, a2, a3, ps, sps, b: bs, xs }
    }

    /// Onsite shift of shell charge changes `dq`: kq dq per AO and the change of the
    /// shell-block coefficient r per AO (the changes of Lambda and r are affine in q:
    /// dLambda_mk = -1/2 Lambda0_mk (f_m + f_k), f = kq dq).
    fn onsite_shift(&self, dq: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let n = self.nao;
        let fm: Vec<f64> = (0..n).map(|m| self.kq[m] * dq[self.ao_shell[m]]).collect();
        let dr = (0..n)
            .map(|m| {
                let l = self.shells[self.ao_shell[m]].2;
                if l >= 1 {
                    -self.lam0[m * n + m] * fm[m] / (2.0 * (2 * l + 1) as f64)
                } else {
                    0.0
                }
            })
            .collect();
        (fm, dr)
    }

    fn acc_new(&self) -> FockAcc {
        let n = self.nao;
        FockAcc { b: vec![0.0; n * n], x: vec![0.0; n * n], f: vec![0.0; n * n], y2: vec![0.0; n], y3: vec![0.0; n] }
    }

    /// Add a term with dense kernels `a1`, `a2` (`a3`; zero if absent) and onsite integrals
    /// `lam` (block diagonal; none if absent) for density `p` with products `ps`, `sps`.
    fn acc_dense(
        &self,
        acc: &mut FockAcc,
        p: &[f64],
        ps: &[f64],
        sps: &[f64],
        a1: &[f64],
        a2: &[f64],
        a3: Option<&[f64]>,
        lam: Option<&[f64]>,
        lam2: Option<&[f64]>,
    ) {
        let n = self.nao;
        for i0 in (0..n).step_by(TILE) {
            for j0 in (0..n).step_by(TILE) {
                for m in i0..(i0 + TILE).min(n) {
                    for k in j0..(j0 + TILE).min(n) {
                        acc.b[m * n + k] += a1[m * n + k] * ps[k * n + m];
                    }
                }
            }
        }
        if let Some(a3) = a3 {
            for ((b, a), v) in acc.b.iter_mut().zip(a3).zip(ps) {
                *b += a * v;
            }
        }
        for (((x, f), a), (pv, sv)) in acc.x.iter_mut().zip(acc.f.iter_mut()).zip(a2).zip(p.iter().zip(sps)) {
            *x += a * pv;
            *f += a * sv;
        }
        if let Some(lam) = lam {
            for &(i0, i1) in &self.atom_ao {
                for m in i0..i1 {
                    let (mut y1, mut y2, mut y3) = (0.0, 0.0, 0.0);
                    for k in i0..i1 {
                        let l = lam[m * n + k];
                        let kk = k * n + k;
                        y1 += l * p[kk];
                        y2 += l * ps[kk];
                        y3 += l * sps[kk];
                    }
                    acc.x[m * n + m] -= y1 / 16.0;
                    acc.y2[m] += y2;
                    acc.y3[m] += y3;
                }
            }
        }
        if let Some(l2) = lam2 {
            for &(i0, i1) in &self.atom_ao {
                for m in i0..i1 {
                    let mut y2 = 0.0;
                    for k in i0..i1 {
                        y2 += l2[m * n + k] * ps[k * n + k];
                    }
                    acc.y2[m] += y2;
                }
            }
        }
    }

    /// Add the term of the onsite-kernel changes of shell charge changes (`fm`, `dr` from
    /// [`Exchange::onsite_shift`]) for density `p` with products `ps`, `sps`: block diagonal.
    fn acc_onsite(&self, acc: &mut FockAcc, p: &[f64], ps: &[f64], sps: &[f64], fm: &[f64], dr: &[f64]) {
        let n = self.nao;
        for &(i0, i1) in &self.atom_ao {
            for m in i0..i1 {
                let sm = self.ao_shell[m];
                let (mut y1, mut y2, mut y3) = (0.0, 0.0, 0.0);
                for k in i0..i1 {
                    let mk = m * n + k;
                    let kk = k * n + k;
                    let dl = -0.5 * self.lam0[mk] * (fm[m] + fm[k]);
                    let rr = if self.ao_shell[k] == sm { 0.25 * dr[m] } else { 0.0 };
                    let a2 = rr - dl / 16.0;
                    acc.b[mk] += rr * ps[k * n + m] - dl / 16.0 * ps[mk];
                    acc.x[mk] += a2 * p[mk];
                    acc.f[mk] += a2 * sps[mk];
                    y1 += dl * p[kk];
                    y2 += dl * ps[kk];
                    y3 += dl * sps[kk];
                }
                acc.x[m * n + m] -= y1 / 16.0;
                acc.y2[m] += y2;
                acc.y3[m] += y3;
            }
        }
    }

    /// The Fock matrix of the accumulators: B S + (B S)^T + S X S + sum A2 o SPS -
    /// 1/16 [D(y3) + D(y2) S + S D(y2)], with X symmetric: (B + S X/2) S plus its transpose
    /// (two matrix products).
    fn acc_finish(&self, s: &[f64], acc: FockAcc, w: f64, par: Par) -> Vec<f64> {
        let n = self.nao;
        let FockAcc { b: mut z, x, mut f, y2, y3 } = acc;
        let sx = matmul(n, s, &x, par);
        drop(x);
        for (a, b) in z.iter_mut().zip(&sx) {
            *a += 0.5 * b;
        }
        drop(sx);
        let zs = matmul(n, &z, s, par);
        drop(z);
        for i0 in (0..n).step_by(TILE) {
            for j0 in (0..n).step_by(TILE) {
                for m in i0..(i0 + TILE).min(n) {
                    for k in j0..(j0 + TILE).min(n) {
                        let mk = m * n + k;
                        f[mk] = w * (f[mk] + zs[mk] + zs[k * n + m] - (y2[m] + y2[k]) * s[mk] / 16.0);
                    }
                }
            }
        }
        for m in 0..n {
            f[m * n + m] -= w * y3[m] / 16.0;
        }
        f
    }

    /// Change of the exchange Fock matrices at fixed spin densities `p` for shell charge
    /// changes `dq` plus extra kernel changes (`e1`, `e2` added to A1 and A2, e.g. the MFX and
    /// bond-order kernels' geometry derivatives): one Fock sum per channel.
    pub fn charge_response(
        &self,
        s: &[f64],
        p: &[Vec<f64>],
        base: &ResponseBase,
        dq: &[f64],
        extra: Option<(&[f64], &[f64])>,
        par: Par,
    ) -> Vec<Vec<f64>> {
        let (fm, dr) = self.onsite_shift(dq);
        let w = if p.len() == 1 { 1.0 } else { 2.0 };
        p.iter()
            .enumerate()
            .map(|(k, pm)| {
                let mut acc = self.acc_new();
                if let Some((e1, e2)) = extra {
                    self.acc_dense(&mut acc, pm, &base.ps[k], &base.sps[k], e1, e2, None, None, None);
                }
                if self.dual.is_none() {
                    self.acc_onsite(&mut acc, pm, &base.ps[k], &base.sps[k], &fm, &dr);
                }
                self.acc_finish(s, acc, w, par)
            })
            .collect()
    }

    /// Derivatives of the exchange along the overlap direction `ds` at spin densities `p` with
    /// the data `base` of [`Exchange::response_base`]; E_x is a homogeneous quadratic in S at
    /// fixed P, kernels and charges, so these are the exact tangents (seven matrix products per
    /// spin channel).
    pub fn overlap_tangent(
        &self,
        s: &[f64],
        p: &[Vec<f64>],
        base: &ResponseBase,
        ds: &[f64],
        par: Par,
    ) -> OverlapTangent {
        let n = self.nao;
        let nsh = self.shells.len();
        let nat = self.nat;
        let w = if p.len() == 1 { 1.0 } else { 2.0 };
        let (lam, a1, a2, a3) = (&base.lam, &base.a1, &base.a2, &base.a3);
        let mut out = OverlapTangent {
            f: Vec::new(),
            vx: vec![0.0; nsh],
            tsh: vec![0.0; nsh * nsh],
            tat: vec![0.0; nat * nat],
            psi: vec![0.0; n * n],
        };
        let mut ddlam = vec![0.0; n * n];
        let mut ddr = vec![0.0; n];
        for (k, pm) in p.iter().enumerate() {
            let (ps, b, xs) = (&base.ps[k], &base.b[k], &base.xs[k]);
            let a = matmul(n, pm, ds, par); // d(PS) = P dS
            let c = matmul(n, ds, ps, par); // dS P S; d(SPS) = C + C^T
            let dsps = |m: usize, l: usize| c[m * n + l] + c[l * n + m];
            let dp: Vec<f64> = (0..n).map(|i| pm[i * n + i]).collect();
            let dsp: Vec<f64> = (0..n).map(|i| ps[i * n + i]).collect();
            let ddsp: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
            let ddsps: Vec<f64> = (0..n).map(|i| 2.0 * c[i * n + i]).collect();
            let mut y2 = vec![0.0; n];
            let mut dy2 = vec![0.0; n];
            let mut dy3 = vec![0.0; n];
            let mut y1 = vec![0.0; n];
            for m in 0..n {
                for l in 0..n {
                    let ml = m * n + l;
                    if self.lam0[ml] != 0.0 {
                        y1[m] += lam[ml] * dp[l];
                        y2[m] += lam[ml] * dsp[l];
                        dy2[m] += lam[ml] * ddsp[l];
                        dy3[m] += lam[ml] * ddsps[l];
                    }
                }
            }
            if let Some(l2) = &base.lam2 {
                for m in 0..n {
                    for l in 0..n {
                        let ml = m * n + l;
                        if l2[ml] != 0.0 {
                            y2[m] += l2[ml] * dsp[l];
                            dy2[m] += l2[ml] * ddsp[l];
                        }
                    }
                }
            }
            // Fock: d(B S + (B S)^T) with dB = A1 o (P dS)^T + A3 o P dS
            let mut db = vec![0.0; n * n];
            for m in 0..n {
                for l in 0..n {
                    let ml = m * n + l;
                    db[ml] = a1[ml] * a[l * n + m] + a3[ml] * a[ml];
                }
            }
            let mut dm = matmul(n, &db, s, par);
            let bds = matmul(n, b, ds, par);
            for (x, y) in dm.iter_mut().zip(&bds) {
                *x += y;
            }
            let dsxs = matmul(n, ds, xs, par); // dS X S; d(S X S) = this + transpose
            let mut f = vec![0.0; n * n];
            for m in 0..n {
                for l in 0..n {
                    let ml = m * n + l;
                    let lm = l * n + m;
                    f[ml] = w
                        * (dm[ml] + dm[lm] + a2[ml] * dsps(m, l) + dsxs[ml] + dsxs[lm]
                            - ((dy2[m] + dy2[l]) * s[ml] + (y2[m] + y2[l]) * ds[ml]) / 16.0);
                }
                f[m * n + m] -= w * dy3[m] / 16.0;
            }
            out.f.push(f);
            // dE/dLambda, dE/dr (shell potential) and the kernel weights
            for m in 0..n {
                let mm = m * n + m;
                for l in 0..n {
                    let ml = m * n + l;
                    let lm = l * n + m;
                    let dt1 = a[ml] * ps[lm] + ps[ml] * a[lm];
                    let dt2 = pm[ml] * dsps(m, l);
                    if self.lam0[ml] != 0.0 {
                        let dt3 = 2.0 * ps[ml] * a[ml];
                        let ll = l * n + l;
                        let dd = pm[mm] * ddsps[l] + a[mm] * ps[ll] + ps[mm] * a[ll];
                        ddlam[ml] -= w * (dt2 + dt3 + dd) / 16.0;
                    }
                    if self.ao_shell[m] == self.ao_shell[l] {
                        ddr[m] += w * 0.25 * (dt1 + dt2);
                    }
                    out.tsh[self.ao_shell[m] * nsh + self.ao_shell[l]] -= w * 0.125 * (dt1 + dt2);
                    out.tat[self.ao_atom[m] * nat + self.ao_atom[l]] += w * 0.5 * dt1;
                }
            }
            // dE/dS is linear in S: its derivative is dE/dS evaluated with S -> dS
            let mut bb = vec![0.0; n * n];
            let mut cc = vec![0.0; n * n];
            let mut z2 = vec![0.0; n];
            for m in 0..n {
                for l in 0..n {
                    let ml = m * n + l;
                    if lam[ml] != 0.0 {
                        z2[m] += lam[ml] * ddsp[l];
                    }
                    if let Some(l2) = &base.lam2 {
                        z2[m] += l2[ml] * ddsp[l];
                    }
                    bb[ml] = a1[ml] * a[l * n + m] + a3[ml] * a[ml];
                    cc[ml] = a2[ml] * pm[ml];
                }
                cc[m * n + m] -= y1[m] / 16.0;
            }
            let pb = matmul(n, pm, &bb, par);
            let spm: Vec<f64> = (0..n * n).map(|x| a[(x % n) * n + x / n]).collect();
            let csp = matmul(n, &cc, &spm, par);
            for m in 0..n {
                for l in 0..n {
                    let ml = m * n + l;
                    let lm = l * n + m;
                    out.psi[ml] += w * (pb[ml] + pb[lm] + csp[ml] + csp[lm] - (z2[m] + z2[l]) * pm[ml] / 16.0);
                }
            }
        }
        for &block in &self.atom_ao {
            self.add_block_shell_potential(block, |m, k| ddlam[m * n + k], |m| ddr[m], &mut out.vx);
        }
        out
    }

    /// Linear response of the exchange at spin densities `p` (shell charges `q`, data `base`)
    /// to density changes `dp` with shell charge changes `dq`: the changes of the Fock matrices
    /// (exact: E_x is quadratic in P and affine in q), of the shell potential dE_x/dq, and the
    /// products dP S and S dP S of every channel (reused by the analytic Hessian's contraction,
    /// [`Exchange::overlap_weight_tangent`] and [`Exchange::kernel_weight_tangent`]).
    #[allow(clippy::type_complexity)]
    pub fn response(
        &self,
        s: &[f64],
        p: &[Vec<f64>],
        base: &ResponseBase,
        dp: &[Vec<f64>],
        dq: &[f64],
        par: Par,
    ) -> (Vec<Vec<f64>>, Vec<f64>, Vec<[Vec<f64>; 2]>) {
        let n = self.nao;
        let (fm, dr) = self.onsite_shift(dq);
        let w = if p.len() == 1 { 1.0 } else { 2.0 };
        let mut dv = vec![0.0; self.shells.len()];
        let mut out = Vec::with_capacity(p.len());
        let mut prods = Vec::with_capacity(p.len());
        let mut ddl = Vec::new();
        let mut ddr = Vec::new();
        for (k, (pm, dpm)) in p.iter().zip(dp).enumerate() {
            let (ps, sps) = (&base.ps[k], &base.sps[k]);
            let dps = matmul(n, dpm, s, par);
            let dsps = matmul(n, s, &dps, par);
            let mut acc = self.acc_new();
            self.acc_dense(
                &mut acc,
                dpm,
                &dps,
                &dsps,
                &base.a1,
                &base.a2,
                Some(&base.a3),
                Some(&base.lam),
                base.lam2.as_deref(),
            );
            if self.dual.is_none() {
                self.acc_onsite(&mut acc, pm, ps, sps, &fm, &dr);
            }
            out.push(self.acc_finish(s, acc, w, par));
            // derivative of dE/dLambda and dE/dr (quadratic in P) along dp, on the atom blocks,
            // and its shell potential
            for &(i0, i1) in &self.atom_ao {
                let na = i1 - i0;
                ddl.clear();
                ddl.resize(na * na, 0.0);
                ddr.clear();
                ddr.resize(na, 0.0);
                for m in i0..i1 {
                    let mm = m * n + m;
                    for kk in i0..i1 {
                        let mk = m * n + kk;
                        let km = kk * n + m;
                        let kkd = kk * n + kk;
                        let dt2 = dpm[mk] * sps[mk] + pm[mk] * dsps[mk];
                        if self.lam0[mk] != 0.0 {
                            let dt3 = 2.0 * ps[mk] * dps[mk];
                            let dd = dpm[mm] * sps[kkd] + pm[mm] * dsps[kkd] + dps[mm] * ps[kkd] + ps[mm] * dps[kkd];
                            ddl[(m - i0) * na + kk - i0] = -w * (dt2 + dt3 + dd) / 16.0;
                        }
                        if self.ao_shell[m] == self.ao_shell[kk] {
                            let dt1 = dps[mk] * ps[km] + ps[mk] * dps[km];
                            ddr[m - i0] += w * 0.25 * (dt1 + dt2);
                        }
                    }
                }
                self.add_block_shell_potential((i0, i1), |m, k| ddl[(m - i0) * na + k - i0], |m| ddr[m - i0], &mut dv);
            }
            prods.push([dps, dsps]);
        }
        (out, dv, prods)
    }

    /// Derivative of [`Exchange::overlap_weight`] (dE/dS at fixed kernels and charges) along
    /// density changes `dp` with the products `prods` = [dP S, S dP S] of every channel (from
    /// [`Exchange::response`]); dE/dS is quadratic in P, so this is exact.
    pub fn overlap_weight_tangent(
        &self,
        p: &[Vec<f64>],
        base: &ResponseBase,
        dp: &[Vec<f64>],
        prods: &[[Vec<f64>; 2]],
        par: Par,
    ) -> Vec<f64> {
        let n = self.nao;
        let (lam, a1, a2, a3) = (&base.lam, &base.a1, &base.a2, &base.a3);
        let closed = p.len() == 1;
        let w = if closed { 1.0 } else { 2.0 };
        let mut x = vec![0.0; n * n];
        for (k, (pm, dpm)) in p.iter().zip(dp).enumerate() {
            let (ps, b, xs) = (&base.ps[k], &base.b[k], &base.xs[k]);
            let dps = &prods[k][0];
            // y1' = Lambda diag(dP), y2 = Lambda diag(PS), y2' = Lambda diag(dP S)
            let mut dy1 = vec![0.0; n];
            let mut y2 = vec![0.0; n];
            let mut dy2 = vec![0.0; n];
            for m in 0..n {
                for l in 0..n {
                    let ml = m * n + l;
                    if self.lam0[ml] != 0.0 {
                        let ll = l * n + l;
                        dy1[m] += lam[ml] * dpm[ll];
                        y2[m] += lam[ml] * ps[ll];
                        dy2[m] += lam[ml] * dps[ll];
                    }
                }
            }
            if let Some(l2) = &base.lam2 {
                for m in 0..n {
                    for l in 0..n {
                        let ml = m * n + l;
                        if l2[ml] != 0.0 {
                            let ll = l * n + l;
                            y2[m] += l2[ml] * ps[ll];
                            dy2[m] += l2[ml] * dps[ll];
                        }
                    }
                }
            }
            // d[P B] = dP B + P dB, dB = A1 o (dP S)^T + A3 o dP S;
            // d[C S P] = dC (S P) + (C S) dP, dC = A2 o dP - D(y1')/16 (base.xs = C S)
            let mut db = vec![0.0; n * n];
            let mut dc = vec![0.0; n * n];
            let mut sp = vec![0.0; n * n];
            for m in 0..n {
                for l in 0..n {
                    let ml = m * n + l;
                    let lm = l * n + m;
                    db[ml] = a1[ml] * dps[lm] + a3[ml] * dps[ml];
                    dc[ml] = a2[ml] * dpm[ml];
                    sp[ml] = ps[lm];
                }
                dc[m * n + m] -= dy1[m] / 16.0;
            }
            let mut t = matmul(n, dpm, b, par);
            for (u, v) in t.iter_mut().zip(matmul(n, pm, &db, par)) {
                *u += v;
            }
            for (u, v) in t.iter_mut().zip(matmul(n, &dc, &sp, par)) {
                *u += v;
            }
            for (u, v) in t.iter_mut().zip(matmul(n, xs, dpm, par)) {
                *u += v;
            }
            for m in 0..n {
                for l in 0..n {
                    let ml = m * n + l;
                    let lm = l * n + m;
                    x[ml] += w * (t[ml] + t[lm] - ((dy2[m] + dy2[l]) * pm[ml] + (y2[m] + y2[l]) * dpm[ml]) / 16.0);
                }
            }
        }
        let mut xs = vec![0.0; n * n];
        for m in 0..n {
            for k in 0..n {
                xs[m * n + k] = 0.5 * (x[m * n + k] + x[k * n + m]);
            }
        }
        xs
    }

    /// Derivative of the kernel weights dE/dG (shell pairs) and dE/dK (atom pairs) of
    /// [`crate::gradient::exchange_kernel_weights`] along density changes `dp` with the
    /// products `prods` = [dP S, S dP S] of every channel.
    pub fn kernel_weight_tangent(
        &self,
        p: &[Vec<f64>],
        base: &ResponseBase,
        dp: &[Vec<f64>],
        prods: &[[Vec<f64>; 2]],
    ) -> (Vec<f64>, Vec<f64>) {
        let n = self.nao;
        let nsh = self.shells.len();
        let nat = self.nat;
        let w = if p.len() == 1 { 1.0 } else { 2.0 };
        let mut tsh = vec![0.0; nsh * nsh];
        let mut tat = vec![0.0; nat * nat];
        for (k, (pm, dpm)) in p.iter().zip(dp).enumerate() {
            let (ps, sps) = (&base.ps[k], &base.sps[k]);
            let [dps, dsps] = &prods[k];
            for m in 0..n {
                let (sm, am) = (self.ao_shell[m] * nsh, self.ao_atom[m] * nat);
                for l in 0..n {
                    let ml = m * n + l;
                    let lm = l * n + m;
                    let dt1 = dps[ml] * ps[lm] + ps[ml] * dps[lm];
                    let dt2 = dpm[ml] * sps[ml] + pm[ml] * dsps[ml];
                    tsh[sm + self.ao_shell[l]] -= w * 0.125 * (dt1 + dt2);
                    tat[am + self.ao_atom[l]] += w * 0.5 * dt1;
                }
            }
        }
        (tsh, tat)
    }
}

// ---- v1.1's exchange constants (from the old tree's exchange.rs and kernels.rs) ----

/// Onsite exchange integral index (1-based into the 9 per-element entries of `com::gmunu`)
/// of an angular-momentum pair (`setfock_indo_$IPOINT`): sp 1, pp 2, sd 3, pd 4, dd 5,
/// sf 6, pf 7, df 8, ff 9; s-s has none.
pub fn gmunu_index(la: usize, lb: usize) -> Option<usize> {
    match (la.min(lb), la.max(lb)) {
        (0, 1) => Some(1),
        (1, 1) => Some(2),
        (0, 2) => Some(3),
        (1, 2) => Some(4),
        (2, 2) => Some(5),
        (0, 3) => Some(6),
        (1, 3) => Some(7),
        (2, 3) => Some(8),
        (3, 3) => Some(9),
        _ => None,
    }
}

/// Exchange-integral scaling applied by the program at start-up (alpha of the reference
/// functional's short-range exact exchange).
pub const OFX_SCALE: f64 = 0.15;

/// Onsite same-shell factor of the MFX kernel.
pub const G_ONSITE: f64 = 1.39;

/// Distance the program stores for an atom with itself (diagonal of its packed distance
/// array); it enters the onsite second-order kernel.
pub const R_SAME_ATOM: f64 = 1e-12;
