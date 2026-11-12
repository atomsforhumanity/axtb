//! Analytic nuclear gradient of g-xTB 2.0.1.
//!
//! The 2.0.1 Fock matrix is the exact derivative of the energy (in both conventions), so the
//! converged SCF is stationary and
//!
//!   dE/dR = dE/dR|_P - Tr(W dS/dR),   W = sum_s C n e C^T
//!
//! (orbitals, occupations and energies of the last diagonalization). dE/dR|_P differentiates
//! everything that depends on the geometry at fixed AO densities: the SCF-basis overlap (Mulliken
//! charges and magnetizations, exchange) and multipole integrals (AES moments, the anisotropy
//! term) with moving centres and q_eff-dependent contractions, the H0 basis and its diatomic-frame
//! scaling, the self-energies, the distance polynomial, the anisotropy vectors, the ACP projector
//! overlaps, all kernels (second and third order, AES, MFX, bond order), the CN-dependent
//! parameters (first order, Hubbard parameters, AES dipole scaling, repulsion), the repulsion and
//! revD4 (two- and three-body, through the revD4 CN); then the chain rules q_eff -> EEQ-BC charges
//! (adjoint of the charge equations) and basis CN, and all CNs -> R.

use super::basis::{h0_shells, scf_shells, INTCUT};
use super::exchange::Products;
use super::h0::{acp_overlaps, anisotropy_gradient_par, diat_scale, kshell, selfenergy, shell_poly, AcpSet};
use super::scf::{ScfResult, Setup, State};
use crate::h0::bond_projectors_deriv;
use crate::timing::Laps;
use axtb::ao::{AtomPair, ShellPairs};
use axtb::basis::Cgto;
use axtb::integrals::{multipole, multipole_grad_coeff, overlap_grad_coeff, Moments, MultipoleGrad, Scratch, MBLK};
use axtb::molecule::sub3;
use axtb::terms::{Chain, MomentTerm, TermEnergy};
use axtb::Result;
use axtb_base::threads::map_chunks;

/// Bond-frame projectors P_mu and their derivatives dP_mu/dn_k of a shell with l <= 2
/// ([`bond_projectors_deriv`]) in fixed-size row-major blocks.
struct BondProj {
    p: [[f64; MBLK]; 3],
    dp: [[[f64; MBLK]; 3]; 3],
}

impl BondProj {
    fn new(l: usize, nv: [f64; 3]) -> Self {
        let (pv, dpv) = bond_projectors_deriv(l, nv);
        let mut out = Self { p: [[0.0; MBLK]; 3], dp: [[[0.0; MBLK]; 3]; 3] };
        for mu in 0..3 {
            out.p[mu][..pv[mu].len()].copy_from_slice(&pv[mu]);
            for k in 0..3 {
                out.dp[mu][k][..dpv[mu][k].len()].copy_from_slice(&dpv[mu][k]);
            }
        }
        out
    }
}

/// Row-major (r x k) (k x c) product of small blocks (zero entries of `x` skipped).
#[inline]
fn mm(x: &[f64], y: &[f64], r: usize, k: usize, c: usize) -> [f64; MBLK] {
    let mut o = [0.0; MBLK];
    for i in 0..r {
        for kk in 0..k {
            let v = x[i * k + kk];
            if v != 0.0 {
                for j in 0..c {
                    o[i * c + j] += v * y[kk * c + j];
                }
            }
        }
    }
    o
}

#[inline]
pub(crate) fn add_pair(g: &mut [[f64; 3]], i: usize, j: usize, f: f64, v: [f64; 3]) {
    for k in 0..3 {
        g[i][k] += f * v[k];
        g[j][k] -= f * v[k];
    }
}

/// Potentials of a state: shell potential of all charge-dependent terms (atomic ones added to
/// every shell of the atom), AES dipole (dE/dd) and quadrupole potentials, the AES potential of
/// the scaled dipoles (dE/d(c d)), spin potential.
pub struct Potentials {
    pub vsh: Vec<f64>,
    pub vd: Vec<[f64; 3]>,
    pub vq: Vec<[f64; 6]>,
    pub vd_scaled: Vec<[f64; 3]>,
    pub w: Option<Vec<f64>>,
}

/// `f(i, chain)` for every row i of `0..n` with a [`Chain`] of the shape of `like` per fixed chunk
/// of rows (in parallel when `parallel`), the chunks' chains summed in order.
pub(crate) fn chain_rows(n: usize, like: &Chain, parallel: bool, f: impl Fn(usize, &mut Chain) + Sync + Send) -> Chain {
    crate::parallel::reduce_rows(n, parallel, || like.zeroed_like(), f, |a, b| a.add(&b))
}

/// dE/dG (shell pairs) and dE/dK (atom pairs) of the exchange at fixed spin densities `p`
/// (closed shell: total density) and overlap `s`.
pub fn exchange_kernel_weights(
    x: &super::exchange::Exchange,
    s: &[f64],
    p: &[Vec<f64>],
    par: faer::Par,
) -> (Vec<f64>, Vec<f64>) {
    let pr = x.products(s, p, par);
    x.kernel_weights_with(p, &pr, !matches!(par, faer::Par::Seq))
}

impl Setup {
    pub fn potentials(&self, st: &State) -> Potentials {
        self.potentials_with(st, &self.exch.products(&self.s, &st.p, self.par))
    }

    /// [`Setup::potentials`] with the exchange [`Exchange::products`](super::exchange::Exchange::products) of `st`.
    pub fn potentials_with(&self, st: &State, pr: &[Products]) -> Potentials {
        let (tg, s) = (self.term_geometry(), st.moments());
        let terms = &self.terms;
        use super::terms::buffer;
        let (nsh, nat) = (self.shells.len(), self.mol.nat());
        let (mut piso, mut paes) = (buffer(&terms.iso, nsh, nat), buffer(&terms.aes, nsh, nat));
        let (mut prep, mut pdisp) = (buffer(&terms.rep, nsh, nat), buffer(&terms.disp, nsh, nat));
        let mut pspin = buffer(&terms.spin, nsh, nat);
        terms.iso.potential(&tg, &self.coulomb, &s, &mut piso);
        let vd_scaled = terms.aes.potential_scaled(&tg, &self.coulomb, &s, &mut paes);
        terms.rep.potential(&tg, &self.rep, &s, &mut prep);
        terms.disp.potential(&tg, &self.disp, &s, &mut pdisp);
        let (_, _, vx) = self.exch.evaluate_with(&self.s, &st.p, pr, &st.q, false, self.par, self.parallel);
        let mut vsh = piso.shell;
        for (ish, &(iat, _, _)) in self.shells.iter().enumerate() {
            vsh[ish] += vx[ish] + paes.atom[iat] + prep.atom[iat] + pdisp.atom[iat];
        }
        let w = st.mag.is_some().then(|| {
            terms.spin.potential(&tg, &(), &s, &mut pspin);
            pspin.magnetization
        });
        Potentials { vsh, vd: paes.dipole, vq: paes.quadrupole, vd_scaled, w }
    }

    /// The q_eff derivatives of the SCF-basis shells of every atom.
    pub(crate) fn scf_shell_derivatives(&self) -> Result<Vec<Vec<Cgto>>> {
        let mol = &self.mol;
        (0..mol.nat()).map(|i| scf_shells(mol.numbers[i], self.basis.qeff[i], true)).collect()
    }

    /// Sum_mn X_mn dS_mn/dR (symmetric X) plus Sum_A vd_A . dd_A/dR + vq_A . dQ_A/dR of the atomic
    /// multipoles of the total density `p` (SCF basis): centres to `ch.g`, effective charges to
    /// `ch.dqeff`. `dsh`: [`Setup::scf_shell_derivatives`]. Atom pairs in fixed chunks (in
    /// parallel with `self.parallel`), summed in order; blocks that are exactly zero are skipped.
    pub(crate) fn add_overlap_multipole(
        &self,
        x: &[f64],
        p: &[f64],
        vd: &[[f64; 3]],
        vq: &[[f64; 6]],
        dsh: &[Vec<Cgto>],
        ch: &mut Chain,
    ) -> Result<()> {
        let n = self.nao;
        let mol = &self.mol;
        let nat = mol.nat();
        let vdot = |vd: &[f64; 3], vq: &[f64; 6], d: &[f64; 3], q: &[f64; 6]| -> f64 {
            vd[0] * d[0] + vd[1] * d[1] + vd[2] * d[2] + (0..6).map(|c| vq[c] * q[c]).sum::<f64>()
        };
        let pairs = ShellPairs::screened(mol, &self.basis.scf, self.cutoff, INTCUT);
        let qeff = self.slots.qeff;
        let parts = map_chunks(nat, self.parallel, |atoms| {
            let mut c = ch.zeroed_like();
            let (cg, [cq]) = c.g_and_slots([qeff]);
            let mut mg = MultipoleGrad::default();
            let mut scr = Scratch::default();
            // moments with the q_eff derivatives of B's (mj) and A's (mi) contraction coefficients
            let (mut mj, mut mi) = (Moments::default(), Moments::default());
            let z = || ([0.0; MBLK], [[0.0; 3]; MBLK], [[0.0; 6]; MBLK]);
            let (mut sa1, mut da1, mut qa1) = z();
            let (mut sa2, mut da2, mut qa2) = z();
            for ia in atoms {
                for ib in 0..ia {
                    let Some(pr) = pairs.pair(ia, ib) else { continue };
                    pairs.for_each(&pr, |sp| {
                        let (sa, sb) = (sp.ci, sp.cj);
                        multipole_grad_coeff(
                            sb,
                            sa,
                            &dsh[ib][sp.jsh],
                            &dsh[ia][sp.ish],
                            pr.vec,
                            INTCUT,
                            &mut mg,
                            &mut mj,
                            &mut mi,
                            &mut scr,
                        );
                        let (na, nb) = (2 * sa.l + 1, 2 * sb.l + 1);
                        for a in 0..na {
                            let mu = sp.ii + a;
                            for b in 0..nb {
                                let nu = sp.jj + b;
                                let k = b + nb * a;
                                let w = 2.0 * x[mu * n + nu];
                                let mut dg = [w * mg.ds[k][0], w * mg.ds[k][1], w * mg.ds[k][2]];
                                let mut dqa = w * mi.s[k];
                                let mut dqb = w * mj.s[k];
                                let pmn = p[mu * n + nu];
                                if pmn != 0.0 {
                                    let (vda, vqa, vdb, vqb) = (&vd[ia], &vq[ia], &vd[ib], &vq[ib]);
                                    for k3 in 0..3 {
                                        dg[k3] -= pmn
                                            * (vdot(vda, vqa, &mg.dd_i[k][k3], &mg.dq_i[k][k3])
                                                + vdot(vdb, vqb, &mg.dd_j[k][k3], &mg.dq_j[k][k3]));
                                    }
                                    dqa -= pmn
                                        * (vdot(vda, vqa, &mi.d_i[k], &mi.q_i[k])
                                            + vdot(vdb, vqb, &mi.d_j[k], &mi.q_j[k]));
                                    dqb -= pmn
                                        * (vdot(vda, vqa, &mj.d_i[k], &mj.q_i[k])
                                            + vdot(vdb, vqb, &mj.d_j[k], &mj.q_j[k]));
                                }
                                for k3 in 0..3 {
                                    cg[ia][k3] += dg[k3];
                                    cg[ib][k3] -= dg[k3];
                                }
                                cq[ia] += dqa;
                                cq[ib] += dqb;
                            }
                        }
                    });
                }
                // onsite blocks: the moments depend on the effective charge only (the onsite
                // overlap is the unit matrix)
                pairs.for_each(&AtomPair::onsite(ia), |sp| {
                    let (sa, sb) = (sp.ci, sp.cj);
                    multipole(sb, &dsh[ia][sp.ish], [0.0; 3], INTCUT, &mut sa1, &mut da1, &mut qa1, &mut scr);
                    multipole(&dsh[ia][sp.jsh], sa, [0.0; 3], INTCUT, &mut sa2, &mut da2, &mut qa2, &mut scr);
                    let (na, nb) = (2 * sa.l + 1, 2 * sb.l + 1);
                    for a in 0..na {
                        let mu = sp.ii + a;
                        for b in 0..nb {
                            let nu = sp.jj + b;
                            let k = b + nb * a;
                            let pmn = p[mu * n + nu];
                            let (vda, vqa) = (&vd[ia], &vq[ia]);
                            cq[ia] -= pmn * (vdot(vda, vqa, &da1[k], &qa1[k]) + vdot(vda, vqa, &da2[k], &qa2[k]));
                        }
                    }
                });
            }
            c
        });
        for c in &parts {
            ch.add(c);
        }
        Ok(())
    }

    /// Tr(P dH0/dR) at fixed total density `p`: self-energies (g-xTB CN), distance polynomial,
    /// the diatomic-frame scaled H0-basis overlap (centres, bond direction, effective charges).
    /// Atom pairs in fixed chunks (in parallel with `self.parallel`), summed in order.
    pub(crate) fn add_h0(&self, p: &[f64], ch: &mut Chain) -> Result<()> {
        let n = self.nao;
        let mol = &self.mol;
        let model = &self.model;
        let nat = mol.nat();
        // the setup's H0 basis (the same shells) and their q_eff derivatives
        let hb = &self.basis.h0;
        let mut dsh = Vec::with_capacity(nat);
        for i in 0..nat {
            dsh.push(h0_shells(mol.numbers[i], self.basis.qeff_h0[i], true)?);
        }
        let mut level = vec![0.0; n];
        let mut dlevel = vec![0.0; n];
        let (cn, qeff0) = (self.slots.cn, self.slots.qeff0);
        let dcn = ch.slot(cn);
        for (ia, is, s) in hb.shell_iter() {
            let (e, de) = selfenergy(mol.numbers[ia], is, self.cn[ia]);
            let first = hb.iao_sh[hb.ish_at[ia] + is];
            for mu in first..first + 2 * s.l + 1 {
                level[mu] = e;
                dlevel[mu] = de;
                dcn[ia] += p[mu * n + mu] * de;
            }
        }
        let pos = &mol.positions;
        let pairs = ShellPairs::screened(mol, hb, self.cutoff, INTCUT);
        let parts = map_chunks(nat, self.parallel, |atoms| {
            let mut c = ch.zeroed_like();
            let (cg, [cq0, ccn]) = c.g_and_slots([qeff0, cn]);
            let (mut sab, mut dsab) = ([0.0; MBLK], [[0.0; 3]; MBLK]);
            let (mut ba, mut bb) = ([0.0; MBLK], [0.0; MBLK]);
            for ia in atoms {
                let za = mol.numbers[ia];
                for ib in 0..ia {
                    let Some(pr) = pairs.pair(ia, ib) else { continue };
                    let zb = mol.numbers[ib];
                    let rab = sub3(pos[ib], pos[ia]);
                    let r = pr.r2.sqrt();
                    let nv = [rab[0] / r, rab[1] / r, rab[2] / r];
                    let kmu = diat_scale(za, zb);
                    let radsum = model.h0_rad(za) + model.h0_rad(zb);
                    // bond-frame projectors and their derivatives per l <= 2, once per bond
                    let mut proj: [Option<BondProj>; 3] = [None, None, None];
                    let mut tdir = [0.0; 3];
                    let mut dedr = 0.0;
                    pairs.for_each(&pr, |shp| {
                        let (sa, sb) = (shp.ci, shp.cj);
                        let (pia, dpia) = shell_poly(za, sa.l, r, radsum);
                        let (pib, dpib) = shell_poly(zb, sb.l, r, radsum);
                        overlap_grad_coeff(
                            sb,
                            sa,
                            Some(&dsh[ib][shp.jsh]),
                            Some(&dsh[ia][shp.ish]),
                            pr.vec,
                            INTCUT,
                            &mut sab,
                            &mut dsab,
                            &mut bb,
                            &mut ba,
                        );
                        let (la, lb) = (sa.l, sb.l);
                        let (na, nb) = (2 * la + 1, 2 * lb + 1);
                        let nab = na * nb;
                        // the block in (a, b) row-major order: S[b + nb a] is the same index
                        let raw = &sab[..nab];
                        let ks = kshell(la, lb);
                        let mut wst = [0.0; MBLK];
                        let mut wcn = [0.0; MBLK];
                        for a in 0..na {
                            let i = shp.ii + a;
                            for b in 0..nb {
                                let j = shp.jj + b;
                                let pij = 2.0 * p[i * n + j];
                                wst[a * nb + b] = pij * 0.5 * (level[i] + level[j]) * ks * pia * pib;
                                wcn[a * nb + b] = pij * 0.5 * ks * pia * pib;
                            }
                        }
                        let mut st = [0.0; MBLK];
                        let mut mw = [0.0; MBLK];
                        if la <= 2 && lb <= 2 {
                            for l in [la, lb] {
                                if proj[l].is_none() {
                                    proj[l] = Some(BondProj::new(l, nv));
                                }
                            }
                            let (pa, pb) = (proj[la].as_ref().unwrap(), proj[lb].as_ref().unwrap());
                            for mu in 0..=la.min(lb) {
                                let a1 = mm(&mm(&pa.p[mu], raw, na, na, nb), &pb.p[mu], na, nb, nb);
                                let a2 = mm(&mm(&pa.p[mu], &wst, na, na, nb), &pb.p[mu], na, nb, nb);
                                for x in 0..nab {
                                    st[x] += kmu[mu] * a1[x];
                                    mw[x] += kmu[mu] * a2[x];
                                }
                                let sp = mm(raw, &pb.p[mu], na, nb, nb);
                                let ps = mm(&pa.p[mu], raw, na, na, nb);
                                for k3 in 0..3 {
                                    let t1 = mm(&pa.dp[mu][k3], &sp, na, na, nb);
                                    let t2 = mm(&ps, &pb.dp[mu][k3], na, nb, nb);
                                    let mut t = 0.0;
                                    for x in 0..nab {
                                        t += wst[x] * (t1[x] + t2[x]);
                                    }
                                    tdir[k3] += kmu[mu] * t;
                                }
                            }
                        } else {
                            st[..nab].copy_from_slice(raw);
                            mw[..nab].copy_from_slice(&wst[..nab]);
                        }
                        for a in 0..na {
                            let i = shp.ii + a;
                            for b in 0..nb {
                                let j = shp.jj + b;
                                let x = a * nb + b;
                                let k = b + nb * a;
                                for k3 in 0..3 {
                                    cg[ia][k3] += mw[x] * dsab[k][k3];
                                    cg[ib][k3] -= mw[x] * dsab[k][k3];
                                }
                                cq0[ia] += mw[x] * ba[k];
                                cq0[ib] += mw[x] * bb[k];
                                ccn[ia] += wcn[x] * st[x] * dlevel[i];
                                ccn[ib] += wcn[x] * st[x] * dlevel[j];
                                let pij = 2.0 * p[i * n + j];
                                dedr += pij * 0.5 * (level[i] + level[j]) * ks * (dpia * pib + pia * dpib) * st[x];
                            }
                        }
                    });
                    // n = (R_b - R_a)/r: dn/dR_b = (1 - n n^T)/r
                    let tn: f64 = (0..3).map(|k3| tdir[k3] * nv[k3]).sum();
                    let tperp = [0, 1, 2].map(|k3| (tdir[k3] - tn * nv[k3]) / r);
                    add_pair(cg, ib, ia, 1.0, tperp);
                    add_pair(cg, ib, ia, dedr, nv);
                }
            }
            c
        });
        for c in &parts {
            ch.add(c);
        }
        Ok(())
    }

    /// Tr(P dH_ACP/dR) at fixed total density `p` (projector overlaps: centres and the SCF-basis
    /// effective charges): 2 sum_r w_r (P s_r) . ds_r, all P s_r from one matrix product;
    /// projectors in fixed chunks (in parallel with `self.parallel`), summed in order.
    /// `dsh`: [`Setup::scf_shell_derivatives`].
    pub(crate) fn add_acp(&self, p: &[f64], dsh: &[Vec<Cgto>], ch: &mut Chain) -> Result<()> {
        let n = self.nao;
        let mol = &self.mol;
        let nat = mol.nat();
        let bas = &self.basis.scf;
        let set = AcpSet::new(mol);
        let nrow = set.nrow;
        if nrow == 0 {
            return Ok(());
        }
        let sp = acp_overlaps(mol, bas, &set, self.cutoff, self.parallel);
        // pc = P S_p^T (row-major n x nrow)
        let mut pc = vec![0.0; n * nrow];
        faer::linalg::matmul::matmul(
            faer::MatMut::from_row_major_slice_mut(&mut pc, n, nrow),
            faer::Accum::Replace,
            faer::MatRef::from_row_major_slice(p, n, n),
            faer::MatRef::from_row_major_slice(&sp, nrow, n).transpose(),
            1.0,
            self.par,
        );
        drop(sp);
        let pairs = ShellPairs::screened(mol, bas, self.cutoff, INTCUT);
        let qeff = self.slots.qeff;
        let parts = map_chunks(set.list.len(), self.parallel, |projs| {
            let mut c = ch.zeroed_like();
            let (cg, [cq]) = c.g_and_slots([qeff]);
            let (mut s, mut ds, mut sj, mut bq) = ([0.0; MBLK], [[0.0; 3]; MBLK], [0.0; MBLK], [0.0; MBLK]);
            for k in projs {
                let (ip, ref proj, w, row) = set.list[k];
                let np = 2 * proj.l + 1;
                for ia in 0..nat {
                    let Some(pr) = pairs.pair_with(ia, ip, proj.min_alpha()) else { continue };
                    pairs.for_each_against(&pr, proj, |shp| {
                        let sh = shp.ci;
                        overlap_grad_coeff(
                            proj,
                            sh,
                            None,
                            Some(&dsh[ia][shp.ish]),
                            pr.vec,
                            INTCUT,
                            &mut s,
                            &mut ds,
                            &mut sj,
                            &mut bq,
                        );
                        for a in 0..2 * sh.l + 1 {
                            let mu = shp.ii + a;
                            for m in 0..np {
                                let fac = 2.0 * w * pc[mu * nrow + row + m];
                                let kk = m + np * a;
                                if ia != ip {
                                    for x in 0..3 {
                                        cg[ia][x] += fac * ds[kk][x];
                                        cg[ip][x] -= fac * ds[kk][x];
                                    }
                                }
                                cq[ia] += fac * bq[kk];
                            }
                        }
                    });
                }
            }
            c
        });
        for c in &parts {
            ch.add(c);
        }
        Ok(())
    }

    /// Chain rules of the intermediates: effective charges -> EEQ-BC charges and basis CN;
    /// CNs and EEQ-BC charges (the setup's factorization) -> geometry.
    pub fn finish_chain(&self, ch: Chain) -> Result<Vec<[f64; 3]>> {
        ch.finish(&self.term_geometry())
    }

    /// Energy-weighted density W = sum_s C n e C^T of a converged SCF (total).
    pub fn energy_weighted_density(&self, r: &ScfResult) -> Vec<f64> {
        let n = self.nao;
        let f: Vec<Vec<f64>> = (0..r.coefficients.len())
            .map(|s| (0..n).map(|k| r.occupations[s][k] * r.orbital_energies[s][k]).collect())
            .collect();
        let ch: Vec<(&[f64], &[f64])> =
            r.coefficients.iter().zip(&f).map(|(c, f)| (c.as_slice(), f.as_slice())).collect();
        super::scf::weighted_density(n, &ch, self.par)
    }

    /// Explicit gradient dE/dR at fixed AO spin densities of the state `st` (Mulliken charges,
    /// moments and magnetizations of these densities), plus sum X dS/dR for an extra overlap
    /// weight `x_extra` (the gradient adds -W).
    pub fn explicit_gradient(&self, st: &State, x_extra: Option<&[f64]>) -> Result<Vec<[f64; 3]>> {
        self.explicit_gradient_with(st, st, x_extra)
    }

    /// [`Setup::explicit_gradient`] with the Mulliken, spin and multipole potentials of the
    /// overlap and multipole-integral derivatives taken from `pot_state`.
    pub fn explicit_gradient_with(
        &self,
        st: &State,
        pot_state: &State,
        x_extra: Option<&[f64]>,
    ) -> Result<Vec<[f64; 3]>> {
        self.explicit_gradient_timed(st, pot_state, x_extra, &mut Laps::new())
    }

    fn explicit_gradient_timed(
        &self,
        st: &State,
        pot_state: &State,
        x_extra: Option<&[f64]>,
        t: &mut Laps,
    ) -> Result<Vec<[f64; 3]>> {
        let ch = self.explicit_chain_timed(st, pot_state, x_extra, t)?;
        let g = self.finish_chain(ch);
        t.lap("gradient: chain rules");
        g
    }

    /// The derivative accumulators of [`Setup::explicit_gradient_with`] before the chain rules
    /// (direct position derivatives at fixed coordination numbers and effective charges, and
    /// dE/dCN, dE/dq_eff of every kind).
    pub fn explicit_chain(&self, st: &State, pot_state: &State, x_extra: Option<&[f64]>) -> Result<Chain> {
        self.explicit_chain_timed(st, pot_state, x_extra, &mut Laps::new())
    }

    fn explicit_chain_timed(
        &self,
        st: &State,
        pot_state: &State,
        x_extra: Option<&[f64]>,
        t: &mut Laps,
    ) -> Result<Chain> {
        let n = self.nao;
        let mol = &self.mol;
        let nat = mol.nat();
        t.skip();
        let ptot = self.total_density(&st.p);
        // the exchange products of the density, shared by its potential, overlap weight and
        // kernel weights
        let pr = self.exch.products(&self.s, &st.p, self.par);
        let pot_now = self.potentials_with(st, &pr);
        let pot_other = (!std::ptr::eq(st, pot_state)).then(|| self.potentials(pot_state));
        let pot = pot_other.as_ref().unwrap_or(&pot_now);
        let mut ch = Chain::new(&self.layout, nat);
        t.lap("gradient: potentials");
        // overlap weight: Mulliken charges, magnetizations, exchange
        let mut x = self.exch.overlap_weight_with(&st.p, &pr, &st.q, self.par, self.parallel);
        for i in 0..n {
            for j in 0..n {
                x[i * n + j] -=
                    0.5 * (pot.vsh[self.basis.scf.ao2sh[i]] + pot.vsh[self.basis.scf.ao2sh[j]]) * ptot[i * n + j];
            }
        }
        if let Some(w) = &pot.w {
            let w: &Vec<f64> = w;
            for i in 0..n {
                for j in 0..n {
                    let pd = st.p[0][i * n + j] - st.p[1][i * n + j];
                    x[i * n + j] += 0.5 * (w[self.basis.scf.ao2sh[i]] + w[self.basis.scf.ao2sh[j]]) * pd;
                }
            }
        }
        if let Some(xe) = x_extra {
            for (a, b) in x.iter_mut().zip(xe) {
                *a += b;
            }
        }
        // dipole potential: AES plus the anisotropy (E_aniso = sum_A a_A . d_A)
        let vd: Vec<[f64; 3]> =
            pot.vd.iter().zip(&self.aniso).map(|(v, a)| [v[0] + a[0], v[1] + a[1], v[2] + a[2]]).collect();
        t.lap("gradient: overlap weight");
        let dsh = self.scf_shell_derivatives()?;
        self.add_overlap_multipole(&x, &ptot, &vd, &pot.vq, &dsh, &mut ch)?;
        t.lap("gradient: overlap+multipole integrals");
        anisotropy_gradient_par(&self.model, mol, &st.dip, &mut ch.g, self.parallel);
        t.lap("gradient: anisotropy");
        self.add_h0(&ptot, &mut ch)?;
        t.lap("gradient: h0");
        self.add_acp(&ptot, &dsh, &mut ch)?;
        t.lap("gradient: acp");
        let (tg, s) = (self.term_geometry(), st.moments());
        self.terms.iso.gradient(&tg, &self.coulomb, &s, &mut ch);
        self.terms.aes.gradient_with(&tg, &self.coulomb, &s, &pot_now.vd_scaled, &mut ch);
        t.lap("gradient: electrostatics");
        let (tsh, tat) = self.exch.kernel_weights_with(&st.p, &pr, self.parallel);
        self.exch.add_kernel_gradient(&self.model, &tg, &tsh, &tat, &mut ch);
        t.lap("gradient: exchange kernels");
        self.terms.rep.gradient(&tg, &self.rep, &s, &mut ch);
        self.terms.disp.gradient(&tg, &self.disp, &s, &mut ch);
        t.lap("gradient: repulsion+revd4");
        let mut e3 = [0.0];
        self.terms.disp.fixed_energy(&tg, &self.disp, &mut TermEnergy { pieces: &mut e3, atoms: None }, Some(&mut ch));
        t.lap("gradient: revd4 three-body");
        Ok(ch)
    }

    /// Analytic gradient (Eh/bohr) of a converged SCF.
    ///
    /// Program mode evaluates the potential terms (Mulliken charge, spin and multipole
    /// potentials of the overlap and multipole-integral derivatives) with the charges and
    /// moments the last Fock matrix was built from, as the 2.0.1 program does (its gradient
    /// keeps the last iteration's potential); this reproduces the program's gradients to
    /// ~1e-13 Eh/bohr at any SCF threshold. The default conventions take every term from the
    /// final density. Both agree at self-consistency; with the default thresholds they differ
    /// by up to ~1e-5 Eh/bohr (the SCF residual), as each does from the converged gradient.
    pub fn gradient(&self, r: &ScfResult) -> Result<Vec<[f64; 3]>> {
        Ok(self.gradient_timed(r)?.0)
    }

    /// [`Setup::gradient`] and the wall time of its phases.
    pub fn gradient_timed(&self, r: &ScfResult) -> Result<(Vec<[f64; 3]>, crate::timing::Timings)> {
        let mut t = Laps::new();
        let mut w = self.energy_weighted_density(r);
        w.iter_mut().for_each(|x| *x = -*x);
        t.lap("gradient: energy-weighted density");
        let pot_state = if self.model.program() { &r.input_state } else { &r.state };
        let g = self.explicit_gradient_timed(&r.state, pot_state, Some(&w), &mut t)?;
        Ok((g, t.laps))
    }
}
