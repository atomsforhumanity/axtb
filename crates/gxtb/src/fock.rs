//! Fock matrices of g-xTB v1.1: the derivatives of the energy terms with respect to the
//! spin densities, as the 1.1.0 program builds them in its SCF (`set1espot_`, `setespot_`,
//! `set3espot_`, `setvsdq_` + `addanisotropich1_`, `setospot_`, `setfock_mat_`,
//! `setfock_indo_`, `tblite_disp_d4_mp_get_potential_`), verified against the program's
//! Fock matrices (reference tests).
//!
//! Conventions: full row-major `n x n` matrices in axtb AO order; densities per spin (closed
//! shell: one matrix, the total density); `F^s = dE/dP^s` (closed shell: `dE/dP`).
//!
//! - Mulliken charge potentials: with V_l = dE/dq_l for the shell charges
//!   q_l = n^ref_l - sum_{mu in l} (P S)_{mu mu}, F_{mu nu} += -1/2 (V_l(mu) + V_l(nu)) S_{mu nu}
//!   (both spins). Atomic potentials (revD4, AES charge part) enter every shell of the atom.
//! - Multipole potentials (AES): F_{mu nu} += -1/2 [v_d(A).D_{mu nu} + v_d(B).D_{nu mu} +
//!   v_Q(A).Q_{mu nu} + v_Q(B).Q_{nu mu}] with the dipole/quadrupole integrals around the
//!   atom of the first index. Only the charge-dipole, charge-quadrupole and dipole-dipole parts
//!   of E_AES are differentiated (the program's `setvsdq_`); the dipole-quadrupole and
//!   quadrupole-quadrupole energies have no potential.
//! - Spin polarization: w_l = dE/dm_l, F^alpha += 1/2 (w_l(mu) + w_l(nu)) S, F^beta -= the same.
//! - Mulliken exchange: F^s = -1/4 [(g o SP) S + S (g o PS) + g o SPS + S (g o P) S]
//!   (P = P^s; closed shell -1/8 with the total density), `o` the elementwise product with the
//!   AO kernel g.
//! - Onsite exchange: F^s = 1/2 (Y S + S Y) + D with Y = dE/dPt (dual density Pt) for the
//!   exchange-type pairs and, for the population part y_k = dE/dPt_kk, D_kk = y_k and
//!   D_kl = (y_k + y_l) S_kl (k != l): the program applies the latter without the factor 1/2,
//!   i.e. at twice its derivative off the diagonal.
//!
//! Apart from the two deviations noted (AES, onsite exchange) the Fock matrices are the exact
//! derivatives of the energy (checked by finite differences).
//!
//! The potential part F = -1/2 (v_mu + v_nu) S - 1/2 (G + G^T), G = D.v_d + Q.v_q by rows, is
//! written once (`potential_part`): added to a matrix by [`add_potentials`] (the response and
//! Hessian) and to H + X by [`from_potentials_fused`] (the SCF). GFN2-xTB's form (`axtb::scf`,
//! B = S v + D.v_d + Q.v_q by columns, then 1/2 (B + B^T)) is another order.

use axtb::basis::Basis;

/// Shell list entry: (atom, shell index on the atom, l).
pub type Shell = (usize, usize, usize);

/// The shell list of a basis (both versions): every shell in basis order.
pub fn shell_list(bas: &Basis) -> Vec<Shell> {
    bas.shell_iter().map(|(iat, ish, c)| (iat, ish, c.l)).collect()
}

/// The total density of spin densities `p` (closed shell: the one matrix, copied), for both
/// versions.
pub fn total_density(p: &[Vec<f64>]) -> Vec<f64> {
    if p.len() == 1 {
        p[0].clone()
    } else {
        p[0].iter().zip(&p[1]).map(|(a, b)| a + b).collect()
    }
}

/// Add -1/2 (v(mu) + v(nu)) S_{mu nu} for an AO potential `v` (times `scale`; -1 and +1 give
/// the alpha and beta spin potentials).
pub fn add_charge_potential(n: usize, s: &[f64], v: &[f64], f: &mut [f64], scale: f64) {
    for i in 0..n {
        for j in 0..n {
            f[i * n + j] -= scale * 0.5 * (v[i] + v[j]) * s[i * n + j];
        }
    }
}

/// The multipole potentials' row part G_mk = vd(A(m)).D_mk + vq(A(m)).Q_mk (the nine products
/// summed in this order from the first, one component at a time over a row), rows in parallel
/// with `parallel`.
fn multipole_rows(
    n: usize,
    ao_atom: &[usize],
    dpint: &[Vec<f64>; 3],
    qpint: &[Vec<f64>; 6],
    vd: &[[f64; 3]],
    vq: &[[f64; 6]],
    parallel: bool,
) -> Vec<f64> {
    let mut g = vec![0.0; n * n];
    axtb_base::threads::for_slices(&mut g, n, parallel, |m, row| {
        let (vdm, vqm) = (vd[ao_atom[m]], vq[ao_atom[m]]);
        let r = m * n..(m + 1) * n;
        for (x, d) in row.iter_mut().zip(&dpint[0][r.clone()]) {
            *x = vdm[0] * d;
        }
        let rest = [(vdm[1], &dpint[1]), (vdm[2], &dpint[2])].into_iter().chain((0..6).map(|c| (vqm[c], &qpint[c])));
        for (c, a) in rest {
            for (x, d) in row.iter_mut().zip(&a[r.clone()]) {
                *x += c * d;
            }
        }
    });
    g
}

/// Element (m, k) of `b` with the potential part: b - 1/2 (v_m + v_k) S_mk - 1/2 (G_mk + G_km).
#[inline]
fn potential_part(b: f64, n: usize, m: usize, k: usize, s: &[f64], vao: &[f64], g: &[f64]) -> f64 {
    let mk = m * n + k;
    b - 0.5 * (vao[m] + vao[k]) * s[mk] - 0.5 * (g[mk] + g[k * n + m])
}

/// Add the potential part of a 2.0.1 Fock matrix (change) to `f` (`potential_part`: the charge
/// potentials per AO `vao`, the multipole potentials `vd`, `vq` of the atoms), in tiles, in
/// parallel with `parallel` (the response and Hessian).
pub fn add_potentials(
    n: usize,
    s: &[f64],
    ao_atom: &[usize],
    dpint: &[Vec<f64>; 3],
    qpint: &[Vec<f64>; 6],
    vao: &[f64],
    vd: &[[f64; 3]],
    vq: &[[f64; 6]],
    f: &mut [f64],
    parallel: bool,
) {
    let g = multipole_rows(n, ao_atom, dpint, qpint, vd, vq, parallel);
    crate::parallel::update_tiled(f, n, parallel, |m, k, b| potential_part(b, n, m, k, s, vao, &g));
}

/// The 2.0.1 SCF's Fock matrices, one fused expression per element: for every channel s with
/// the density-dependent part `x[s]` (the exchange),
///
/// ```text
/// F_s = H + X_s - 1/2 (v_m + v_k) S - 1/2 (G + G^T) +- 1/2 (w_m + w_k) S
/// ```
///
/// with the charge potentials per AO `vao`, G_mk = vd(A(m)).D_mk + vq(A(m)).Q_mk (by rows, all
/// nine products in one sum: `potential_part`) and, with `wao`, the spin potentials per AO
/// (+ on alpha, the first channel, - on beta). G by rows and F in tiles, in parallel with
/// `parallel` (each element by one task: the result does not depend on the thread count).
pub fn from_potentials_fused(
    n: usize,
    h: &[f64],
    s: &[f64],
    ao_atom: &[usize],
    dpint: &[Vec<f64>; 3],
    qpint: &[Vec<f64>; 6],
    vao: &[f64],
    vd: &[[f64; 3]],
    vq: &[[f64; 6]],
    wao: Option<&[f64]>,
    x: &[Vec<f64>],
    parallel: bool,
) -> Vec<Vec<f64>> {
    let g = multipole_rows(n, ao_atom, dpint, qpint, vd, vq, parallel);
    let mut fm = Vec::with_capacity(x.len());
    for (k, xs) in x.iter().enumerate() {
        let mut f = vec![0.0; n * n];
        let g = &g;
        let sw = if k == 0 { 0.5 } else { -0.5 };
        crate::parallel::fill_tiled(&mut f, n, parallel, |m, k| {
            let mk = m * n + k;
            let mut v = potential_part(h[mk] + xs[mk], n, m, k, s, vao, g);
            if let Some(w) = wao {
                v += sw * (w[m] + w[k]) * s[mk];
            }
            v
        });
        fm.push(f);
    }
    fm
}
