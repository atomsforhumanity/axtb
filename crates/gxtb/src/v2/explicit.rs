//! Explicit second derivatives of the integral terms of g-xTB 2.0.1 at fixed AO density and
//! fixed gradient weights, in local variables, and the chain second derivatives.
//!
//! The integral terms of the gradient are pair (or projector-atom) sums of integral derivatives
//! contracted with weights that do not depend on the geometry at this level (their geometry
//! dependence enters through the SCC and exchange couplings of [`super::analytic`]):
//!
//! - SCF-basis overlap, dipole and quadrupole integrals of an atom pair with the overlap weight
//!   X (Mulliken, spin and exchange potentials, -W) and the dipole (incl. anisotropy) and
//!   quadrupole potentials: local variables R_A - R_B, q_eff of A and B; onsite moments: q_eff;
//! - H0 blocks of an atom pair (diatomic frame, distance polynomial, self-energies) with P:
//!   R_A - R_B, q_eff0 of A and B, CN of A and B;
//! - ACP: sum_k w_k s_k^T P s_k with the projector overlaps s_k: positions and q_eff.
//!
//! Second derivatives of each pair term: 4-point central differences of its analytic first
//! derivatives over its local variables (deterministic, no SCF, only the pair's integrals);
//! ACP: the products of analytic first derivatives plus pair differences for the second
//! derivatives of the projector overlaps. The local Hessian (variables: positions, q_eff,
//! q_eff0, CN) is chained to Cartesians with the Jacobians, and the second derivatives of the
//! chains themselves (EEQ-BC charges, basis and g-xTB CNs, the q_eff formula) are added with
//! the gradient's dE/dq_eff, dE/dq_eff0 and the H0 part of dE/dCN.

use super::basis::{ao_offsets, effective_charge_deriv, h0_shells, qeff_h0_weights, qeff_weights, scf_shells, INTCUT};
use super::cn::{basis_cn_gradient, gxtb_cn_gradient};
use super::gradient::{Chain, Potentials};
use super::h0::acp_projectors;
use super::scf::{Setup, State};
use super::tangent::{h0_pair, Jacobians};
use crate::acp::projector;
use axtb::basis::Cgto;
use axtb::integrals::{multipole, multipole_grad, overlap, MultipoleGrad, Scratch, MBLK};
use axtb::molecule::{norm2_sq, sub3};
use axtb::Result;
use axtb_base::fd;

/// Map `f` over 0..n in chunks of `chunk` (each chunk in parallel when `parallel`) and pass the
/// results to `sink` in order: the memory of pending results is bounded by the chunk, and
/// whatever `sink` sums runs in the same order for every chunk size and thread count.
fn for_each_ordered<T: Send>(
    n: usize,
    chunk: usize,
    parallel: bool,
    f: impl Fn(usize) -> T + Sync + Send,
    mut sink: impl FnMut(T),
) {
    let chunk = chunk.max(1);
    let mut k0 = 0;
    while k0 < n {
        let k = chunk.min(n - k0);
        for r in crate::parallel::map_items(k, parallel, |j| f(k0 + j)) {
            sink(r);
        }
        k0 += k;
    }
}

/// Add a block to the row-major nl x nl local Hessian.
fn add_block(hl: &mut [f64], nl: usize, b: &Block) {
    let m = b.idx.len();
    for i in 0..m {
        for j in 0..m {
            hl[b.idx[i] * nl + b.idx[j]] += b.h[i * m + j];
        }
    }
}

/// Step of the 4-point differences of the analytic Hessian (bohr, and units of charge and CN):
/// the local differences here and the kernel terms of [`super::analytic`].
pub(crate) const STEP: f64 = 1e-3;

/// Smallest primitive exponent of a set of shells (the q-vSZP exponents do not depend on q).
fn min_exponent(shells: &[Cgto]) -> f64 {
    shells.iter().flat_map(|s| s.alpha[..s.nprim].iter().copied()).fold(f64::INFINITY, f64::min)
}

/// Whether every primitive pair of two atoms with smallest exponents `a` and `b` at distance
/// `r` is screened (a_i a_j R^2/(a_i + a_j) > INTCUT) at every geometry of the local
/// differences (atoms moved by up to 2 STEP): their integrals and blocks are exactly zero.
fn screened(a: f64, b: f64, r: f64) -> bool {
    let rmin = r - 3.0 * STEP;
    rmin > 0.0 && a * b * rmin * rmin / (a + b) > INTCUT * (1.0 + 1e-6)
}

/// 4-point central differences of a gradient function `g` of K variables (row i, column v:
/// dg_i/dx_v), symmetrized.
fn fd_hessian<const K: usize>(mut g: impl FnMut(&[f64; K]) -> [f64; K]) -> [[f64; K]; K] {
    let mut h = [[0.0; K]; K];
    for v in 0..K {
        let col = fd::derivative(fd::Stencil::Central4, STEP, |d| {
            let mut x = [0.0; K];
            x[v] = d;
            g(&x)
        });
        for i in 0..K {
            h[i][v] = col[i];
        }
    }
    fd::symmetrize(h.as_flattened_mut(), K);
    h
}

/// Index of a difference offset (0, h, -h, 2h, -2h).
#[inline]
fn offset_index(d: f64) -> usize {
    if d == 0.0 {
        0
    } else if (d - STEP).abs() < 1e-3 * STEP {
        1
    } else if (d + STEP).abs() < 1e-3 * STEP {
        2
    } else if d > 0.0 {
        3
    } else {
        4
    }
}

/// Shells of every atom at the effective-charge offsets (0, h, -h, 2h, -2h): (shells, their
/// q derivatives).
type ShellCache = Vec<[(Vec<Cgto>, Vec<Cgto>); 5]>;

fn shell_cache(numbers: &[usize], q: &[f64], f: impl Fn(usize, f64, bool) -> Result<Vec<Cgto>>) -> Result<ShellCache> {
    let offs = [0.0, STEP, -STEP, 2.0 * STEP, -2.0 * STEP];
    numbers
        .iter()
        .zip(q)
        .map(|(&z, &q0)| {
            let v: Vec<(Vec<Cgto>, Vec<Cgto>)> =
                offs.iter().map(|d| Ok((f(z, q0 + d, false)?, f(z, q0 + d, true)?))).collect::<Result<_>>()?;
            Ok(v.try_into().unwrap())
        })
        .collect()
}

/// A block of the local Hessian: variable indices and the dense block.
struct Block {
    idx: Vec<usize>,
    h: Vec<f64>,
}

/// Translation-invariant pair block: the variables are [R_A (3), extra...] with R_B = -R_A in
/// effect; expand to [R_A (3), R_B (3), extra...].
fn pair_block<const K: usize>(h: &[[f64; K]; K], ra: usize, rb: usize, extra: &[usize]) -> Block {
    let mut idx: Vec<usize> = (0..3).map(|c| 3 * ra + c).chain((0..3).map(|c| 3 * rb + c)).collect();
    idx.extend_from_slice(extra);
    let m = idx.len();
    // map of each expanded variable to (source variable, sign)
    let src = |k: usize| -> (usize, f64) {
        if k < 3 {
            (k, 1.0)
        } else if k < 6 {
            (k - 3, -1.0)
        } else {
            (k - 3, 1.0)
        }
    };
    let mut out = vec![0.0; m * m];
    for i in 0..m {
        let (si, fi) = src(i);
        for j in 0..m {
            let (sj, fj) = src(j);
            out[i * m + j] = fi * fj * h[si][sj];
        }
    }
    Block { idx, h: out }
}

impl Setup {
    /// The gradient's overlap weight: exchange dE/dS, Mulliken and spin potentials, plus
    /// `x_extra` (-W for the gradient).
    pub fn overlap_weight_total(&self, st: &State, pot: &Potentials, x_extra: &[f64]) -> Vec<f64> {
        let n = self.nao;
        let ptot = self.total_density(&st.p);
        let mut x = self.exch.overlap_weight(&self.s, &st.p, &st.q, self.par);
        for i in 0..n {
            for j in 0..n {
                x[i * n + j] -= 0.5 * (pot.vsh[self.ao_shell[i]] + pot.vsh[self.ao_shell[j]]) * ptot[i * n + j];
            }
        }
        if let Some(w) = &pot.w {
            for i in 0..n {
                for j in 0..n {
                    let pd = st.p[0][i * n + j] - st.p[1][i * n + j];
                    x[i * n + j] += 0.5 * (w[self.ao_shell[i]] + w[self.ao_shell[j]]) * pd;
                }
            }
        }
        for (a, b) in x.iter_mut().zip(x_extra) {
            *a += b;
        }
        x
    }

    /// Local Hessian (row-major 6N x 6N; variables: positions, q_eff, q_eff0, CN) of the
    /// integral terms at fixed weights: overlap weight `x`, dipole potential `vd` (incl. the
    /// anisotropy vectors), quadrupole potential `vq` and total density `p`.
    pub fn integral_local_hessian(
        &self,
        x: &[f64],
        p: &[f64],
        vd: &[[f64; 3]],
        vq: &[[f64; 6]],
        parallel: bool,
    ) -> Result<Vec<f64>> {
        let n = self.nao;
        let mol = &self.mol;
        let nat = mol.nat();
        let nl = 6 * nat;
        let pos = &mol.positions;
        let first = ao_offsets(&self.basis.shells);
        let cut2 = self.cutoff * self.cutoff;
        let vdot = |vd: &[f64; 3], vq: &[f64; 6], d: &[f64; 3], q: &[f64; 6]| -> f64 {
            vd[0] * d[0] + vd[1] * d[1] + vd[2] * d[2] + (0..6).map(|c| vq[c] * q[c]).sum::<f64>()
        };
        let pairs: Vec<(usize, usize)> = (0..nat)
            .flat_map(|a| (0..a).map(move |b| (a, b)))
            .filter(|&(a, b)| norm2_sq(sub3(pos[a], pos[b])) <= cut2)
            .collect();
        let scf_cache = shell_cache(&mol.numbers, &self.basis.qeff, scf_shells)?;
        let h0_cache = shell_cache(&mol.numbers, &self.basis.qeff_h0, h0_shells)?;
        // pairs whose integrals are screened at every displaced geometry contribute nothing
        let amin = |c: &ShellCache| -> Vec<f64> {
            c.iter()
                .map(|v| v.iter().map(|(a, b)| min_exponent(a).min(min_exponent(b))).fold(f64::INFINITY, f64::min))
                .collect()
        };
        let (scf_amin, h0_amin) = (amin(&scf_cache), amin(&h0_cache));
        let dist = |a: usize, b: usize| norm2_sq(sub3(pos[a], pos[b])).sqrt();
        let scf_pairs: Vec<(usize, usize)> =
            pairs.iter().copied().filter(|&(a, b)| !screened(scf_amin[a], scf_amin[b], dist(a, b))).collect();
        let h0_pairs: Vec<(usize, usize)> =
            pairs.iter().copied().filter(|&(a, b)| !screened(h0_amin[a], h0_amin[b], dist(a, b))).collect();
        // SCF-basis integral pairs: variables R_A (3), q_eff A, q_eff B
        let scf_pair = |&(ia, ib): &(usize, usize)| -> Result<Block> {
            let mut mg = Box::<MultipoleGrad>::default();
            let mut scr = Box::<Scratch>::default();
            let z = || Box::new(([0.0; MBLK], [[0.0; 3]; MBLK], [[0.0; 6]; MBLK]));
            let (mut t1, mut t2, mut t3, mut t4) = (z(), z(), z(), z());
            let g = |v: &[f64; 5]| -> [f64; 5] {
                let (sa_, dsa) = &scf_cache[ia][offset_index(v[3])];
                let (sb_, dsb) = &scf_cache[ib][offset_index(v[4])];
                let ra = [pos[ia][0] + v[0], pos[ia][1] + v[1], pos[ia][2] + v[2]];
                let vec = sub3(ra, pos[ib]);
                let mvec = [-vec[0], -vec[1], -vec[2]];
                let (s1, d1, q1) = &mut *t1;
                let (s2, d2, q2) = &mut *t2;
                let (s3, d3, q3) = &mut *t3;
                let (s4, d4, q4) = &mut *t4;
                let mut out = [0.0; 5];
                let (vda, vqa, vdb, vqb) = (&vd[ia], &vq[ia], &vd[ib], &vq[ib]);
                for (isa, sa) in sa_.iter().enumerate() {
                    for (isb, sb) in sb_.iter().enumerate() {
                        multipole_grad(sb, sa, vec, INTCUT, &mut mg, &mut scr);
                        multipole(sb, &dsa[isa], vec, INTCUT, s1, d1, q1, &mut scr);
                        multipole(&dsb[isb], sa, vec, INTCUT, s2, d2, q2, &mut scr);
                        multipole(sa, &dsb[isb], mvec, INTCUT, s3, d3, q3, &mut scr);
                        multipole(&dsa[isa], sb, mvec, INTCUT, s4, d4, q4, &mut scr);
                        let (ma, mb) = (2 * sa.l + 1, 2 * sb.l + 1);
                        for a in 0..ma {
                            let mu = first[ia][isa] + a;
                            for b in 0..mb {
                                let nu = first[ib][isb] + b;
                                let k = b + mb * a;
                                let kr = a + ma * b;
                                let w = 2.0 * x[mu * n + nu];
                                let pmn = p[mu * n + nu];
                                for c in 0..3 {
                                    out[c] += w * mg.ds[k][c]
                                        - pmn
                                            * (vdot(vda, vqa, &mg.dd_i[k][c], &mg.dq_i[k][c])
                                                + vdot(vdb, vqb, &mg.dd_j[k][c], &mg.dq_j[k][c]));
                                }
                                out[3] += w * s1[k]
                                    - pmn * (vdot(vda, vqa, &d1[k], &q1[k]) + vdot(vdb, vqb, &d4[kr], &q4[kr]));
                                out[4] += w * s2[k]
                                    - pmn * (vdot(vda, vqa, &d2[k], &q2[k]) + vdot(vdb, vqb, &d3[kr], &q3[kr]));
                            }
                        }
                    }
                }
                out
            };
            let h = fd_hessian::<5>(g);
            Ok(pair_block(&h, ia, ib, &[3 * nat + ia, 3 * nat + ib]))
        };
        // onsite moments: variable q_eff
        let scf_onsite = |ia: usize| -> Result<Block> {
            let mut scr = Box::<Scratch>::default();
            let z = || Box::new(([0.0; MBLK], [[0.0; 3]; MBLK], [[0.0; 6]; MBLK]));
            let (mut t1, mut t2) = (z(), z());
            let g = |v: &[f64; 1]| -> [f64; 1] {
                let (sh, dsh) = &scf_cache[ia][offset_index(v[0])];
                let (s1, d1, q1) = &mut *t1;
                let (s2, d2, q2) = &mut *t2;
                let mut out = 0.0;
                for (isa, sa) in sh.iter().enumerate() {
                    for (isb, sb) in sh.iter().enumerate() {
                        multipole(sb, &dsh[isa], [0.0; 3], INTCUT, s1, d1, q1, &mut scr);
                        multipole(&dsh[isb], sa, [0.0; 3], INTCUT, s2, d2, q2, &mut scr);
                        let (ma, mb) = (2 * sa.l + 1, 2 * sb.l + 1);
                        for a in 0..ma {
                            let mu = first[ia][isa] + a;
                            for b in 0..mb {
                                let nu = first[ia][isb] + b;
                                let k = b + mb * a;
                                out -= p[mu * n + nu]
                                    * (vdot(&vd[ia], &vq[ia], &d1[k], &q1[k]) + vdot(&vd[ia], &vq[ia], &d2[k], &q2[k]));
                            }
                        }
                    }
                }
                [out]
            };
            let h = fd_hessian::<1>(g);
            Ok(Block { idx: vec![3 * nat + ia], h: vec![h[0][0]] })
        };
        // H0 pairs: variables R_A (3), q_eff0 A, B, CN A, B
        let h0_block = |&(ia, ib): &(usize, usize)| -> Result<Block> {
            let (za, zb) = (mol.numbers[ia], mol.numbers[ib]);
            let (oa, ob) = (first[ia][0], first[ib][0]);
            let mut mg = Box::<MultipoleGrad>::default();
            let mut scr = Box::<Scratch>::default();
            let g = |v: &[f64; 7]| -> [f64; 7] {
                let (sha, dsha) = &h0_cache[ia][offset_index(v[3])];
                let (shb, dshb) = &h0_cache[ib][offset_index(v[4])];
                let ra = [pos[ia][0] + v[0], pos[ia][1] + v[1], pos[ia][2] + v[2]];
                let pr = h0_pair(
                    &self.model,
                    za,
                    zb,
                    ra,
                    pos[ib],
                    sha,
                    shb,
                    dsha,
                    dshb,
                    self.cn[ia] + v[5],
                    self.cn[ib] + v[6],
                    &mut mg,
                    &mut scr,
                );
                let mut out = [0.0; 7];
                for a in 0..pr.na {
                    for b in 0..pr.nb {
                        let w = 2.0 * p[(oa + a) * n + ob + b];
                        let o = a * pr.nb + b;
                        // pr.d[0..3]: derivatives with respect to R_B = -R_A
                        for c in 0..3 {
                            out[c] -= w * pr.d[c][o];
                        }
                        for k in 3..7 {
                            out[k] += w * pr.d[k][o];
                        }
                    }
                }
                out
            };
            let h = fd_hessian::<7>(g);
            Ok(pair_block(&h, ia, ib, &[4 * nat + ia, 4 * nat + ib, 5 * nat + ia, 5 * nat + ib]))
        };
        // summed in a fixed order (SCF pairs, onsite, H0 pairs), in chunks
        const CHUNK: usize = 4096;
        let mut hl = vec![0.0; nl * nl];
        let mut err = Ok(());
        let mut sink = |b: Result<Block>| match b {
            Ok(b) => add_block(&mut hl, nl, &b),
            Err(e) => err = Err(e),
        };
        for_each_ordered(scf_pairs.len(), CHUNK, parallel, |k| scf_pair(&scf_pairs[k]), &mut sink);
        for_each_ordered(nat, CHUNK, parallel, scf_onsite, &mut sink);
        for_each_ordered(h0_pairs.len(), CHUNK, parallel, |k| h0_block(&h0_pairs[k]), &mut sink);
        err?;
        self.add_acp_local_hessian(p, &scf_cache, &mut hl, parallel)?;
        Ok(hl)
    }

    /// ACP part of the local Hessian: sum_k w_k s_k^T P s_k, with the products of the
    /// projector-overlap derivatives and pair differences for their second derivatives.
    fn add_acp_local_hessian(&self, p: &[f64], scf_cache: &ShellCache, hl: &mut [f64], parallel: bool) -> Result<()> {
        let n = self.nao;
        let mol = &self.mol;
        let nat = mol.nat();
        let nl = 6 * nat;
        let pos = &mol.positions;
        let shells = &self.basis.shells;
        let first = ao_offsets(shells);
        let na: Vec<usize> = (0..nat).map(|a| shells[a].iter().map(|s| 2 * s.l + 1).sum()).collect();
        let dsh: Vec<Vec<Cgto>> =
            (0..nat).map(|i| scf_shells(mol.numbers[i], self.basis.qeff[i], true)).collect::<Result<_>>()?;
        let cut2 = self.cutoff * self.cutoff;
        let scf_amin: Vec<f64> = scf_cache
            .iter()
            .map(|v| v.iter().map(|(a, b)| min_exponent(a).min(min_exponent(b))).fold(f64::INFINITY, f64::min))
            .collect();
        let projs: Vec<(usize, usize, f64, f64)> = (0..nat)
            .flat_map(|ip| {
                acp_projectors(mol.numbers[ip]).into_iter().filter(|t| t.2 != 0.0).map(move |(l, a, w)| (ip, l, a, w))
            })
            .collect();
        let per_proj = |k: usize| -> Vec<Block> {
            let (ip, l, alpha, w) = projs[k];
            let proj = projector(l, alpha);
            let np = 2 * l + 1;
            let mut s = vec![0.0; n * np];
            // u[atom][var]: derivative vectors on the atom's AOs (n_A x np), var 0..3 position, 3 q_eff
            let mut u: Vec<Option<[Vec<f64>; 4]>> = vec![None; nat];
            let mut mg = MultipoleGrad::default();
            let mut scr = Scratch::default();
            let (mut blk, mut bq) = ([0.0; MBLK], [0.0; MBLK]);
            for ia in 0..nat {
                let vec = sub3(pos[ia], pos[ip]);
                if norm2_sq(vec) > cut2 {
                    continue;
                }
                let mut ua =
                    [vec![0.0; na[ia] * np], vec![0.0; na[ia] * np], vec![0.0; na[ia] * np], vec![0.0; na[ia] * np]];
                for (is, sh) in shells[ia].iter().enumerate() {
                    overlap(&proj, sh, vec, INTCUT, &mut blk);
                    overlap(&proj, &dsh[ia][is], vec, INTCUT, &mut bq);
                    if ia != ip {
                        multipole_grad(&proj, sh, vec, INTCUT, &mut mg, &mut scr);
                    }
                    for a in 0..2 * sh.l + 1 {
                        let mu = first[ia][is] + a;
                        let ml = mu - first[ia][0];
                        for m in 0..np {
                            let kk = m + np * a;
                            s[mu * np + m] = blk[kk];
                            ua[3][ml * np + m] = bq[kk];
                            if ia != ip {
                                for c in 0..3 {
                                    ua[c][ml * np + m] = mg.ds[kk][c];
                                }
                            }
                        }
                    }
                }
                u[ia] = Some(ua);
            }
            // y = P s
            let mut y = vec![0.0; n * np];
            for mu in 0..n {
                for nu in 0..n {
                    let pv = p[mu * n + nu];
                    if pv != 0.0 {
                        for m in 0..np {
                            y[mu * np + m] += pv * s[nu * np + m];
                        }
                    }
                }
            }
            // local variables: (atom, var); the projector atom's position is -sum of the others
            // (variables whose overlap derivatives all vanish, e.g. screened far atoms, have zero
            // rows and are left out)
            let vars: Vec<(usize, usize)> = (0..nat)
                .filter(|&a| u[a].is_some())
                .flat_map(|a| (0..4).map(move |v| (a, v)))
                .filter(|&(a, v)| !(a == ip && v < 3) && u[a].as_ref().unwrap()[v].iter().any(|&x| x != 0.0))
                .collect();
            let gidx = |a: usize, v: usize| if v < 3 { 3 * a + v } else { 3 * nat + a };
            // z_eta = P u_eta (n x np)
            let zs: Vec<Vec<f64>> = vars
                .iter()
                .map(|&(b, v)| {
                    let ub = &u[b].as_ref().unwrap()[v];
                    let mut z = vec![0.0; n * np];
                    for mu in 0..n {
                        for nl_ in 0..na[b] {
                            let pv = p[mu * n + first[b][0] + nl_];
                            if pv != 0.0 {
                                for m in 0..np {
                                    z[mu * np + m] += pv * ub[nl_ * np + m];
                                }
                            }
                        }
                    }
                    z
                })
                .collect();
            let nv = vars.len();
            let mut h1 = vec![0.0; nv * nv];
            for (i, &(a, va)) in vars.iter().enumerate() {
                let ua = &u[a].as_ref().unwrap()[va];
                for j in 0..nv {
                    let z = &zs[j];
                    let mut acc = 0.0;
                    for ml in 0..na[a] {
                        let mu = first[a][0] + ml;
                        for m in 0..np {
                            acc += ua[ml * np + m] * z[mu * np + m];
                        }
                    }
                    h1[i * nv + j] = 2.0 * w * acc;
                }
            }
            // expand with the projector atom's position (translation invariance)
            let mut idx: Vec<usize> = vars.iter().map(|&(a, v)| gidx(a, v)).collect();
            idx.extend((0..3).map(|c| 3 * ip + c));
            let m = idx.len();
            let mut full = vec![0.0; m * m];
            // the projector atom's displacement c is minus the sum of the AO atoms' (a != ip)
            let sel: [Vec<usize>; 3] =
                [0, 1, 2].map(|c| (0..nv).filter(|&i| vars[i].1 == c && vars[i].0 != ip).collect());
            for i in 0..nv {
                for j in 0..nv {
                    full[i * m + j] = h1[i * nv + j];
                }
            }
            // rows/columns of the projector atom: -sum over the selected base variables
            let mut rowsum = vec![[0.0; 3]; nv]; // sum_{i in sel[c]} h1[i][j]
            for c in 0..3 {
                for &i in &sel[c] {
                    for j in 0..nv {
                        rowsum[j][c] += h1[i * nv + j];
                    }
                }
            }
            for c in 0..3 {
                for j in 0..nv {
                    full[(nv + c) * m + j] = -rowsum[j][c];
                    full[j * m + nv + c] = -rowsum[j][c];
                }
                for d in 0..3 {
                    let v: f64 = sel[d].iter().map(|&j| rowsum[j][c]).sum();
                    full[(nv + c) * m + nv + d] = v;
                }
            }
            let mut out = vec![Block { idx, h: full }];
            // second derivatives of the overlaps: pair (projector atom, AO atom) differences
            for ia in 0..nat {
                if u[ia].is_none() || screened(alpha, scf_amin[ia], norm2_sq(sub3(pos[ia], pos[ip])).sqrt()) {
                    continue;
                }
                let g = |v: &[f64; 4]| -> [f64; 4] {
                    let (sh, dq) = &scf_cache[ia][offset_index(v[3])];
                    let ra = [pos[ia][0] + v[0], pos[ia][1] + v[1], pos[ia][2] + v[2]];
                    let vec = sub3(ra, pos[ip]);
                    let mut out = [0.0; 4];
                    for (is, s_) in sh.iter().enumerate() {
                        if ia != ip {
                            multipole_grad(&proj, s_, vec, INTCUT, &mut mg, &mut scr);
                        }
                        overlap(&proj, &dq[is], vec, INTCUT, &mut bq);
                        for a in 0..2 * s_.l + 1 {
                            let mu = first[ia][is] + a;
                            for m in 0..np {
                                let kk = m + np * a;
                                let yv = 2.0 * w * y[mu * np + m];
                                if ia != ip {
                                    for c in 0..3 {
                                        out[c] += yv * mg.ds[kk][c];
                                    }
                                }
                                out[3] += yv * bq[kk];
                            }
                        }
                    }
                    out
                };
                let h = fd_hessian::<4>(g);
                if ia == ip {
                    out.push(Block { idx: vec![3 * nat + ia], h: vec![h[3][3]] });
                } else {
                    out.push(pair_block(&h, ia, ip, &[3 * nat + ia]));
                }
            }
            out
        };
        // one dense block over the projector's variables each: a chunk of projectors at a time
        let chunk = if parallel { rayon::current_num_threads() } else { 1 };
        for_each_ordered(projs.len(), chunk, parallel, per_proj, |bl| {
            for b in &bl {
                add_block(hl, nl, b);
            }
        });
        Ok(())
    }

    /// Chain second derivatives (row-major 3N x 3N): sum_A [dE/dq_eff,A d^2 q_eff,A/dR dR +
    /// dE/dq_eff0,A d^2 q_eff0,A/dR dR + dE/dCN_A d^2 CN_A/dR dR] for the integral terms' weights
    /// `dqeff`, `dqeff0` and `dcn` (the H0 part). The second derivatives of the EEQ-BC charges
    /// and of the CNs contracted with their weights come from 4-point differences of their
    /// analytic adjoint gradients.
    pub fn chain_second_derivatives(
        &self,
        jac: &Jacobians,
        dqeff: &[f64],
        dqeff0: &[f64],
        dcn: &[f64],
        parallel: bool,
    ) -> Result<Vec<f64>> {
        let mol = &self.mol;
        let nat = mol.nat();
        let n3 = 3 * nat;
        let mut u = vec![0.0; nat];
        let mut v = vec![0.0; nat];
        let mut alpha = vec![0.0; nat];
        let mut beta = vec![0.0; nat];
        let mut gamma = vec![0.0; nat];
        for a in 0..nat {
            let z = mol.numbers[a];
            let (q, c) = (self.eeq.charges[a], self.basis.cn[a]);
            let s2 = -0.25 * (c + 1e-12).powf(-1.5);
            for (w, k) in [(dqeff[a], qeff_weights(z)), (dqeff0[a], qeff_h0_weights(z))] {
                let (fq, fc) = effective_charge_deriv(k, q, c);
                u[a] += w * fq;
                v[a] += w * fc;
                alpha[a] += w * (-2.0 * k[0] * k[1]);
                beta[a] += w * k[3];
                gamma[a] += w * k[2] * s2;
            }
        }
        let column = |x: usize| -> Result<Vec<f64>> {
            fd::try_derivative(fd::Stencil::Central4, STEP, |d| -> Result<Vec<f64>> {
                let mut m = mol.clone();
                m.positions[x / 3][x % 3] += d;
                let mut g = vec![[0.0; 3]; nat];
                super::eeqbc::eeqbc_charge_gradient(&self.model, &m, &u, &mut g)?;
                basis_cn_gradient(&self.model, &m, &v, &mut g);
                gxtb_cn_gradient(&self.model, &m, dcn, &mut g);
                Ok(g.concat())
            })
        };
        let cols = fd::columns(n3, parallel, column).into_iter().collect::<Result<Vec<_>>>()?;
        let mut out = fd::from_columns(&cols, n3);
        // second derivatives of the q_eff formula itself
        for a in 0..nat {
            let (je, jb) = (&jac.eeq[a * n3..(a + 1) * n3], &jac.cnb[a * n3..(a + 1) * n3]);
            for x in 0..n3 {
                for y in 0..n3 {
                    out[x * n3 + y] +=
                        alpha[a] * je[x] * je[y] + beta[a] * (je[x] * jb[y] + jb[x] * je[y]) + gamma[a] * jb[x] * jb[y];
                }
            }
        }
        Ok(out)
    }

    /// Hessian (row-major 3N x 3N) of the revD4 three-body energy E3(R, CN(R)) (C6 at q = 0):
    /// local Hessian over positions and revD4 CNs by 4-point differences of the analytic
    /// gradient of the triples that contain the perturbed atom (the others do not change),
    /// chained with the CN Jacobian, plus dE3/dCN times the CN second derivatives.
    pub fn three_body_hessian(&self, jac_cnd4: &[f64], parallel: bool) -> Result<Vec<f64>> {
        let mol = &self.mol;
        let model = &self.model;
        let nat = mol.nat();
        let n3 = 3 * nat;
        let nl = 4 * nat;
        let cn0 = self.disp.cn.clone();
        // local gradient: positions (direct) and dE/dCN, of the triples with atom a
        let local = |a: usize, v: usize, d: f64| -> Result<Vec<f64>> {
            let mut g = vec![[0.0; 3]; nat];
            let dedcn = if v < 3 {
                let mut m = mol.clone();
                m.positions[a][v] += d;
                self.disp.three_body_of(model, &m, Some(&mut g), Some(a)).1
            } else {
                let mut cn = cn0.clone();
                cn[a] += d;
                let disp = super::disp::RevD4::with_cn(model, mol, cn)?;
                disp.three_body_of(model, mol, Some(&mut g), Some(a)).1
            };
            let mut out = g.concat();
            out.extend(dedcn);
            Ok(out)
        };
        let column = |c: usize| -> Result<Vec<f64>> {
            let (a, v) = if c < n3 { (c / 3, c % 3) } else { (c - n3, 3) };
            fd::try_derivative(fd::Stencil::Central4, STEP, |d| local(a, v, d))
        };
        let cols = fd::columns(nl, parallel, column).into_iter().collect::<Result<Vec<_>>>()?;
        let mut hl = fd::from_columns(&cols, nl);
        fd::symmetrize(&mut hl, nl);
        // chain: J = [I; dCN/dR]
        let mut j = vec![0.0; nl * n3];
        for x in 0..n3 {
            j[x * n3 + x] = 1.0;
        }
        j[n3 * n3..].copy_from_slice(jac_cnd4);
        let jm = faer::MatRef::from_row_major_slice(&j, nl, n3);
        let hm = faer::MatRef::from_row_major_slice(&hl, nl, nl);
        let mut tmp = faer::Mat::<f64>::zeros(nl, n3);
        faer::linalg::matmul::matmul(tmp.as_mut(), faer::Accum::Replace, hm, jm, 1.0, faer::Par::Seq);
        let mut hc = faer::Mat::<f64>::zeros(n3, n3);
        faer::linalg::matmul::matmul(
            hc.as_mut(),
            faer::Accum::Replace,
            jm.transpose(),
            tmp.as_ref(),
            1.0,
            faer::Par::Seq,
        );
        // dE3/dCN times the CN second derivatives
        let mut dummy = vec![[0.0; 3]; nat];
        let (_, dedcn) = self.disp.three_body(model, mol, Some(&mut dummy));
        let cncol = |x: usize| -> Vec<f64> {
            fd::derivative(fd::Stencil::Central4, STEP, |d| {
                let mut m = mol.clone();
                m.positions[x / 3][x % 3] += d;
                let mut g = vec![[0.0; 3]; nat];
                super::disp::add_cn_gradient(model, &m, &dedcn, &mut g);
                g.concat()
            })
        };
        let cc = fd::columns(n3, parallel, cncol);
        let mut out = vec![0.0; n3 * n3];
        for x in 0..n3 {
            for y in 0..n3 {
                out[x * n3 + y] = hc[(x, y)] + cc[y][x];
            }
        }
        Ok(out)
    }

    /// The H0 part of dE/dCN (self-energies) at total density `p`.
    pub fn h0_cn_weights(&self, p: &[f64]) -> Result<Vec<f64>> {
        let mut ch = Chain::new(self.mol.nat());
        self.add_h0(p, &mut ch)?;
        Ok(ch.dcn)
    }
}
