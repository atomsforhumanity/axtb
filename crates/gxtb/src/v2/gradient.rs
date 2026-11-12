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

use super::basis::{effective_charge_deriv, h0_shells, qeff_h0_weights, qeff_weights, scf_shells, INTCUT};
use super::cn::{basis_cn_gradient, gxtb_cn_gradient};
use super::coulomb::{aes_kernel, first_order_g, gamma_kernel, spin_potential, tau_kernel};
use super::exchange::{bo_kernel, mfx_kernel, mfx_pair, Products};
use super::h0::{acp_overlaps, anisotropy_gradient_par, diat_scale, kshell, selfenergy, shell_poly, AcpSet};
use super::scf::{ScfResult, Setup, State};
use super::sqrt_cn;
use crate::data_v2::{AES_DIP_SCALE, HUBBARD, HUBBARD_CN, IPEA, IPEA_CN, SHELL_HUBBARD};
use crate::h0::bond_projectors_deriv;
use crate::timing::Laps;
use axtb::ao::{AtomPair, ShellPairs};
use axtb::basis::Cgto;
use axtb::integrals::{multipole, multipole_grad_coeff, overlap_grad_coeff, Moments, MultipoleGrad, Scratch, MBLK};
use axtb::molecule::{norm2_sq, sub3};
use axtb::Result;
use axtb_base::threads::map_chunks;

/// Derivative accumulators before the chain rules.
#[derive(Clone, Debug)]
pub struct Chain {
    /// Direct geometry derivative (Eh/bohr).
    pub g: Vec<[f64; 3]>,
    /// dE/dCN of the g-xTB CN, the basis CN and the revD4 CN.
    pub dcn: Vec<f64>,
    pub dcnb: Vec<f64>,
    pub dcnd4: Vec<f64>,
    /// dE/dq_eff of the SCF and of the H0 basis.
    pub dqeff: Vec<f64>,
    pub dqeff0: Vec<f64>,
}

impl Chain {
    pub fn new(nat: usize) -> Self {
        let z = || vec![0.0; nat];
        Self { g: vec![[0.0; 3]; nat], dcn: z(), dcnb: z(), dcnd4: z(), dqeff: z(), dqeff0: z() }
    }

    /// Add the accumulators of `o`.
    pub fn add(&mut self, o: &Chain) {
        for (a, b) in self.g.iter_mut().zip(&o.g) {
            for x in 0..3 {
                a[x] += b[x];
            }
        }
        for (a, b) in [
            (&mut self.dcn, &o.dcn),
            (&mut self.dcnb, &o.dcnb),
            (&mut self.dcnd4, &o.dcnd4),
            (&mut self.dqeff, &o.dqeff),
            (&mut self.dqeff0, &o.dqeff0),
        ] {
            for (x, y) in a.iter_mut().zip(b) {
                *x += y;
            }
        }
    }
}

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
fn add_pair(g: &mut [[f64; 3]], i: usize, j: usize, f: f64, v: [f64; 3]) {
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

/// The integral-free, geometry-dependent kernels of one geometry: everything the
/// charge-, multipole- and exchange-kernel terms of the energy need besides the SCC variables.
pub struct KernelRef<'a> {
    pub model: &'a super::Model,
    pub mol: &'a axtb::Molecule,
    pub cn: &'a [f64],
    pub coulomb: &'a super::coulomb::Coulomb,
    pub rep: &'a super::repulsion::Repulsion,
    pub disp: &'a super::disp::RevD4,
    pub exch: &'a super::exchange::Exchange,
    /// Pair loops in fixed chunks on the current rayon pool.
    pub parallel: bool,
}

/// `f(i, chain)` for every row i of `0..n` with a [`Chain`] per fixed chunk of rows (in
/// parallel when `parallel`), the chunks' chains summed in order.
fn chain_rows(n: usize, nat: usize, parallel: bool, f: impl Fn(usize, &mut Chain) + Sync + Send) -> Chain {
    crate::parallel::reduce_rows(n, parallel, || Chain::new(nat), f, |a, b| a.add(&b))
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

impl KernelRef<'_> {
    /// First- to fourth-order terms at fixed shell charges: kernels (distances, Hubbard
    /// parameters through the CN) and the CN-dependent first-order potentials.
    pub fn add_isotropic(&self, q: &[f64], ch: &mut Chain) {
        let mol = self.mol;
        let c = self.coulomb;
        let sh = &c.shells;
        let ns = sh.len();
        let qa = c.atom_charges(q);
        let mut du = vec![0.0; ns];
        for (i, &(iat, ish, _)) in sh.iter().enumerate() {
            let z = mol.numbers[iat] - 1;
            let df = sqrt_cn(self.cn[iat]).1;
            du[i] = SHELL_HUBBARD[z][ish] * HUBBARD[z] * HUBBARD_CN[z] * df;
            let g = first_order_g(qa[iat]).0;
            ch.dcn[iat] += IPEA[z][ish] * IPEA_CN[z] * df * g * q[i];
        }
        let u = &c.u;
        let acc = chain_rows(ns, mol.nat(), self.parallel, |i, ch| {
            for j in 0..=i {
                let (ia, ja) = (sh[i].0, sh[j].0);
                let same = ia == ja;
                // dE/dgamma_ij and dE/dtau_ij of the symmetric pair
                let (wg, wt) = if i == j {
                    (0.5 * q[i] * q[i], q[i] * q[i] * q[i] * c.gam3[i] / 3.0)
                } else {
                    let mut wg = q[i] * q[j];
                    if !same {
                        wg += q[i] * c.rref[j] + q[j] * c.rref[i];
                    }
                    (wg, (q[i] * q[i] * c.gam3[i] * q[j] + q[j] * q[j] * c.gam3[j] * q[i]) / 3.0)
                };
                let r = if same { 0.0 } else { norm2_sq(sub3(mol.positions[ia], mol.positions[ja])).sqrt() };
                let eta = 2.0 / (1.0 / u[i] + 1.0 / u[j]);
                let (_, dgdr, dgde) = gamma_kernel(r, eta);
                let (_, dtdr, dtde) = tau_kernel(r, eta, same);
                let dede = wg * dgde + wt * dtde;
                let s2 = (u[i] + u[j]) * (u[i] + u[j]);
                ch.dcn[ia] += dede * 2.0 * u[j] * u[j] / s2 * du[i];
                ch.dcn[ja] += dede * 2.0 * u[i] * u[i] / s2 * du[j];
                if !same {
                    let dedr = wg * dgdr + wt * dtdr;
                    let v = sub3(mol.positions[ia], mol.positions[ja]);
                    add_pair(&mut ch.g, ia, ja, dedr / r, v);
                }
            }
        });
        ch.add(&acc);
    }

    /// Anisotropic electrostatics at fixed charges and moments: kernels (explicit distance
    /// vectors) and the CN-dependent dipole scaling (`vd_scaled` = dE/d(c d)).
    pub fn add_aes(&self, qa: &[f64], dip: &[[f64; 3]], qp: &[[f64; 6]], vd_scaled: &[[f64; 3]], ch: &mut Chain) {
        let mol = self.mol;
        let n = mol.nat();
        let d = self.coulomb.scaled_dipoles(dip);
        for i in 0..n {
            let z = mol.numbers[i] - 1;
            let dc = AES_DIP_SCALE[z] * sqrt_cn(self.cn[i]).1;
            ch.dcn[i] += dc * (0..3).map(|x| dip[i][x] * vd_scaled[i][x]).sum::<f64>();
        }
        let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let qm = |q: &[f64; 6]| [[q[0], q[1], q[3]], [q[1], q[2], q[4]], [q[3], q[4], q[5]]];
        let mv = |m: &[[f64; 3]; 3], v: [f64; 3]| [dot(m[0], v), dot(m[1], v), dot(m[2], v)];
        let q = qa;
        let acc = chain_rows(n, n, self.parallel, |i, ch| {
            for j in 0..i {
                let r = sub3(mol.positions[j], mol.positions[i]);
                let r2 = dot(r, r);
                let rr = r2.sqrt();
                let r0 = self.model.rvdw(mol.numbers[i], mol.numbers[j]);
                let (g3, dg3) = aes_kernel(0, rr, r0);
                let (g5, dg5) = aes_kernel(1, rr, r0);
                let (g7, dg7) = aes_kernel(2, rr, r0);
                let (g9, dg9) = aes_kernel(3, rr, r0);
                let (qi, qj) = (qm(&qp[i]), qm(&qp[j]));
                let (di, dj) = (d[i], d[j]);
                let (qir, qjr) = (mv(&qi, r), mv(&qj, r));
                let (rqir, rqjr) = (dot(r, qir), dot(r, qjr));
                let (dir, djr) = (dot(di, r), dot(dj, r));
                let e_qd = q[j] * dir - q[i] * djr;
                let grad_qd = [0, 1, 2].map(|c| q[j] * di[c] - q[i] * dj[c]);
                let e_qq = q[j] * rqir + q[i] * rqjr;
                let grad_qq = [0, 1, 2].map(|c| 2.0 * (q[j] * qir[c] + q[i] * qjr[c]));
                let e_dd = dot(di, dj) * r2 - 3.0 * dir * djr;
                let grad_dd = [0, 1, 2].map(|c| 2.0 * dot(di, dj) * r[c] - 3.0 * (djr * di[c] + dir * dj[c]));
                let (qidj, qjdi) = (mv(&qi, dj), mv(&qj, di));
                let (rqidj, rqjdi) = (dot(r, qidj), dot(r, qjdi));
                let e_dq = 5.0 * rqir * djr - 2.0 * rqidj * r2 - (5.0 * dir * rqjr - 2.0 * rqjdi * r2);
                let grad_dq = [0, 1, 2].map(|c| {
                    5.0 * (2.0 * qir[c] * djr + rqir * dj[c])
                        - 2.0 * (qidj[c] * r2 + 2.0 * rqidj * r[c])
                        - 5.0 * (di[c] * rqjr + 2.0 * dir * qjr[c])
                        + 2.0 * (qjdi[c] * r2 + 2.0 * rqjdi * r[c])
                });
                let mut tr = 0.0;
                for a in 0..3 {
                    for b in 0..3 {
                        tr += qi[a][b] * qj[a][b];
                    }
                }
                let qiqjr = mv(&qi, qjr);
                let qjqir = mv(&qj, qir);
                let qrqr = dot(qir, qjr);
                let e_q2 = 35.0 * rqir * rqjr - 20.0 * qrqr * r2 + 3.0 * tr * r2 * r2;
                let grad_q2 = [0, 1, 2].map(|c| {
                    35.0 * (2.0 * qir[c] * rqjr + 2.0 * qjr[c] * rqir)
                        - 20.0 * ((qiqjr[c] + qjqir[c]) * r2 + 2.0 * qrqr * r[c])
                        + 12.0 * tr * r2 * r[c]
                });
                let dedr = e_qd * dg3 + (e_qq + e_dd) * dg5 - e_dq * dg7 + e_q2 * dg9 / 3.0;
                let mut gv = [0.0; 3];
                for c in 0..3 {
                    gv[c] = g3 * grad_qd[c] + g5 * (grad_qq[c] + grad_dd[c]) - g7 * grad_dq[c]
                        + g9 * grad_q2[c] / 3.0
                        + dedr * r[c] / rr;
                }
                add_pair(&mut ch.g, j, i, 1.0, gv);
            }
        });
        ch.add(&acc);
    }

    /// Exchange kernels (MFX shell kernel, bond-order kernel) at fixed weights dE/dG (`tsh`,
    /// shell pairs) and dE/dK (`tat`, atom pairs) from [`exchange_kernel_weights`].
    pub fn add_exchange_kernels(&self, tsh: &[f64], tat: &[f64], ch: &mut Chain) {
        let mol = self.mol;
        let x = self.exch;
        let nsh = x.shells.len();
        let nat = mol.nat();
        let shells = &x.shells;
        let acc = chain_rows(nsh, nat, self.parallel, |i, ch| {
            for j in 0..i {
                let (ia, ja) = (shells[i].0, shells[j].0);
                if ia == ja {
                    continue;
                }
                let v = sub3(mol.positions[ia], mol.positions[ja]);
                let r = norm2_sq(v).sqrt();
                let (kap, etag) = mfx_pair(self.model, mol, shells, i, j);
                let dg = mfx_kernel(r, kap, etag).1;
                let w = tsh[i * nsh + j] + tsh[j * nsh + i];
                add_pair(&mut ch.g, ia, ja, w * dg / r, v);
            }
        });
        ch.add(&acc);
        let acc = chain_rows(nat, nat, self.parallel, |i, ch| {
            for j in 0..i {
                let v = sub3(mol.positions[i], mol.positions[j]);
                let r = norm2_sq(v).sqrt();
                let dk = bo_kernel(self.model, mol.numbers[i], mol.numbers[j], r).1;
                let w = tat[i * nat + j] + tat[j * nat + i];
                add_pair(&mut ch.g, i, j, w * dk / r, v);
            }
        });
        ch.add(&acc);
    }
}

impl Setup {
    /// The kernels of this setup.
    pub fn kernel_ref(&self) -> KernelRef<'_> {
        KernelRef {
            model: &self.model,
            mol: &self.mol,
            cn: &self.cn,
            coulomb: &self.coulomb,
            rep: &self.rep,
            disp: &self.disp,
            exch: &self.exch,
            parallel: self.parallel,
        }
    }

    pub fn potentials(&self, st: &State) -> Potentials {
        self.potentials_with(st, &self.exch.products(&self.s, &st.p, self.par))
    }

    /// [`Setup::potentials`] with the exchange [`Exchange::products`](super::exchange::Exchange::products) of `st`.
    pub fn potentials_with(&self, st: &State, pr: &[Products]) -> Potentials {
        let mol = &self.mol;
        let par = self.parallel;
        let mut vsh = self.coulomb.isotropic_potential(&st.q, par);
        let (vs, vd_scaled, vq) = self.coulomb.aes_potentials_raw(mol, &st.qa, &st.dip, &st.qp, par);
        let vd = self.coulomb.scaled_dipoles(&vd_scaled);
        let vrep = self.rep.potential_par(&st.qa, par);
        let (_, vdisp) = self.disp.two_body_par(&st.qa, par);
        let (_, _, vx) = self.exch.evaluate_with(&self.s, &st.p, pr, &st.q, false, self.par, self.parallel);
        for (ish, &(iat, _, _)) in self.shells.iter().enumerate() {
            vsh[ish] += vx[ish] + vs[iat] + vrep[iat] + vdisp[iat];
        }
        let w = st.mag.as_ref().map(|m| spin_potential(mol, &self.shells, m));
        Potentials { vsh, vd, vq, vd_scaled, w }
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
        let parts = map_chunks(nat, self.parallel, |atoms| {
            let mut c = Chain::new(nat);
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
                                    c.g[ia][k3] += dg[k3];
                                    c.g[ib][k3] -= dg[k3];
                                }
                                c.dqeff[ia] += dqa;
                                c.dqeff[ib] += dqb;
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
                            c.dqeff[ia] -= pmn * (vdot(vda, vqa, &da1[k], &qa1[k]) + vdot(vda, vqa, &da2[k], &qa2[k]));
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
        for (ia, is, s) in hb.shell_iter() {
            let (e, de) = selfenergy(mol.numbers[ia], is, self.cn[ia]);
            let first = hb.iao_sh[hb.ish_at[ia] + is];
            for mu in first..first + 2 * s.l + 1 {
                level[mu] = e;
                dlevel[mu] = de;
                ch.dcn[ia] += p[mu * n + mu] * de;
            }
        }
        let pos = &mol.positions;
        let pairs = ShellPairs::screened(mol, hb, self.cutoff, INTCUT);
        let parts = map_chunks(nat, self.parallel, |atoms| {
            let mut c = Chain::new(nat);
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
                                    c.g[ia][k3] += mw[x] * dsab[k][k3];
                                    c.g[ib][k3] -= mw[x] * dsab[k][k3];
                                }
                                c.dqeff0[ia] += mw[x] * ba[k];
                                c.dqeff0[ib] += mw[x] * bb[k];
                                c.dcn[ia] += wcn[x] * st[x] * dlevel[i];
                                c.dcn[ib] += wcn[x] * st[x] * dlevel[j];
                                let pij = 2.0 * p[i * n + j];
                                dedr += pij * 0.5 * (level[i] + level[j]) * ks * (dpia * pib + pia * dpib) * st[x];
                            }
                        }
                    });
                    // n = (R_b - R_a)/r: dn/dR_b = (1 - n n^T)/r
                    let tn: f64 = (0..3).map(|k3| tdir[k3] * nv[k3]).sum();
                    let tperp = [0, 1, 2].map(|k3| (tdir[k3] - tn * nv[k3]) / r);
                    add_pair(&mut c.g, ib, ia, 1.0, tperp);
                    add_pair(&mut c.g, ib, ia, dedr, nv);
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
        let parts = map_chunks(set.list.len(), self.parallel, |projs| {
            let mut c = Chain::new(nat);
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
                                        c.g[ia][x] += fac * ds[kk][x];
                                        c.g[ip][x] -= fac * ds[kk][x];
                                    }
                                }
                                c.dqeff[ia] += fac * bq[kk];
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

    /// Chain rules: effective charges -> EEQ-BC charges and basis CN; CNs and EEQ-BC charges ->
    /// geometry.
    pub fn finish_chain(&self, mut ch: Chain) -> Result<Vec<[f64; 3]>> {
        let mol = &self.mol;
        let n = mol.nat();
        let mut dqeeq = vec![0.0; n];
        for i in 0..n {
            let z = mol.numbers[i];
            let (q, cnb) = (self.eeq.charges[i], self.basis.cn[i]);
            let (a, b) = effective_charge_deriv(qeff_weights(z), q, cnb);
            let (a0, b0) = effective_charge_deriv(qeff_h0_weights(z), q, cnb);
            dqeeq[i] += ch.dqeff[i] * a + ch.dqeff0[i] * a0;
            ch.dcnb[i] += ch.dqeff[i] * b + ch.dqeff0[i] * b0;
        }
        let mut g = ch.g;
        gxtb_cn_gradient(&self.model, mol, &ch.dcn, &mut g);
        basis_cn_gradient(&self.model, mol, &ch.dcnb, &mut g);
        super::disp::add_cn_gradient(&self.model, mol, &ch.dcnd4, &mut g);
        // the setup's factorization of the EEQ-BC equations (this geometry, the same kernel)
        super::eeqbc::eeqbc(&self.model, mol)?.add_charge_gradient(&self.eeq_solution, &dqeeq, &mut g);
        Ok(g)
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
        let mut ch = Chain::new(nat);
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
        let k = self.kernel_ref();
        k.add_isotropic(&st.q, &mut ch);
        k.add_aes(&st.qa, &st.dip, &st.qp, &pot_now.vd_scaled, &mut ch);
        t.lap("gradient: electrostatics");
        let (tsh, tat) = self.exch.kernel_weights_with(&st.p, &pr, self.parallel);
        k.add_exchange_kernels(&tsh, &tat, &mut ch);
        t.lap("gradient: exchange kernels");
        self.rep.gradient_par(&self.model, mol, &self.cn, &st.qa, &mut ch.g, &mut ch.dcn, self.parallel);
        let d2 = self.disp.two_body_gradient_par(&self.model, mol, &st.qa, &mut ch.g, self.parallel);
        t.lap("gradient: repulsion+revd4");
        let (_, d3) = self.disp.three_body_par(&self.model, mol, Some(&mut ch.g), self.parallel);
        for i in 0..nat {
            ch.dcnd4[i] += d2[i] + d3[i];
        }
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
