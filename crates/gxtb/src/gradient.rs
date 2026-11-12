//! Analytic nuclear gradients of g-xTB v1.1.
//!
//! The energy depends on the geometry R directly and through the AO density P. With the SCF
//! orbitals C, energies e and occupations n (the eigenvectors of the Fock matrix F_used the
//! program diagonalizes):
//!
//!   dE/dR = dE/dR|_P - Tr(W dS/dR) + Tr((F_exact - F_used) dP/dR),   W = C n e C^T,
//!
//! where dE/dR|_P is the explicit derivative at fixed AO density ([`Setup::explicit_gradient`]),
//! dS/dR the total derivative of the overlap (moving centres and the charge- and CN-dependent
//! q-vSZP contraction) and the last term the response of the non-variational Fock matrix
//! (the program leaves out the dipole-quadrupole and quadrupole-quadrupole AES potentials and
//! doubles the off-diagonal population part of the onsite exchange; see `fock.rs`), obtained
//! from one Z-vector equation ([`Setup::gradient`] in `response.rs`). This module holds the
//! explicit part: [`Setup::explicit_gradient`], split into the density-dependent weights
//! ([`Setup::explicit_parts`]) and the integral derivatives and chain rules
//! ([`Setup::finish_explicit`]), so that the response can contract all integral derivatives
//! in one pass.
//!
//! Every quantity that depends on R is differentiated: the integrals (centres, and the
//! effective charges of the SCF and H0 bases, which depend on the EEQ-BC charges and the basis
//! CN), the diatomic-frame scaling of H0 (bond direction), the g-xTB CN (levels, Hubbard
//! parameters, ACP weights, repulsion, first-order potentials), the EEQ-BC charges (adjoint of
//! the charge equations), the kernels (second/third order, Mulliken exchange, AES) and revD4
//! (pair weights through the revD4 CN, damping, three-body).

use crate::acp::{projector, INTCUT as ACP_INTCUT, R2_CUTOFF as ACP_R2};
use crate::aes::{INTCUT as AES_INTCUT, K9, KDMP3, KDMP5, R_MAX as AES_RMAX};
use crate::data::{GAM, GAM3, RCOV_BASIS_CN, RCOV_COM};
use crate::disp::RevD4;
use crate::eeqbc::{eeqbc_charge_gradient, erf_f64, vdw_rad_pair};
use crate::es::{atom_charges, reference_drho, K1_DIS, K1_S};
use crate::fock::{aes_potentials, isotropic_shell_potential, matmul, spin_shell_potential};
use crate::h0::{
    bond_projectors_deriv, h0_atom_shells, h0_atom_shells_dq, h0_effective_charge, k_poly, k_wolfsberg,
    INTCUT as H0_INTCUT, K_DELTA,
};
use crate::kernels::{hubbard_u, ALPHA_FX, R_SAME_ATOM};
use crate::params::{EeqBcParams, QvszpParams};
use crate::qvszp::{atom_shells_dq, KCN_BASIS};
use crate::scf::Setup;
use axtb::basis::{ao_offsets, Cgto};
use axtb::integrals::{multipole, multipole_grad, overlap, MultipoleGrad, Scratch, MBLK};
use axtb::molecule::{norm2_sq, sub3};
use axtb::Result;

const SQRTPI: f64 = 1.772_453_850_905_516;

/// Derivative accumulators of the explicit gradient before the chain rules.
#[derive(Clone, Debug)]
pub struct Chain {
    /// Direct geometry derivative (Eh/bohr).
    pub g: Vec<[f64; 3]>,
    /// dE/dCN of the g-xTB CN, the basis CN and the revD4 CN.
    pub dcn: Vec<f64>,
    pub dcnb: Vec<f64>,
    pub dcnd4: Vec<f64>,
    /// dE/dq of the EEQ-BC charges.
    pub dqeeq: Vec<f64>,
    /// dE/dq_eff of the SCF basis and dE/dq'_eff of the H0 basis.
    pub dqeff: Vec<f64>,
    pub dqeff0: Vec<f64>,
}

impl Chain {
    pub fn new(nat: usize) -> Self {
        Self {
            g: vec![[0.0; 3]; nat],
            dcn: vec![0.0; nat],
            dcnb: vec![0.0; nat],
            dcnd4: vec![0.0; nat],
            dqeeq: vec![0.0; nat],
            dqeff: vec![0.0; nat],
            dqeff0: vec![0.0; nat],
        }
    }

    /// self += f * other.
    pub fn add_scaled(&mut self, f: f64, o: &Chain) {
        for (a, b) in self.g.iter_mut().zip(&o.g) {
            for k in 0..3 {
                a[k] += f * b[k];
            }
        }
        let pairs = [
            (&mut self.dcn, &o.dcn),
            (&mut self.dcnb, &o.dcnb),
            (&mut self.dcnd4, &o.dcnd4),
            (&mut self.dqeeq, &o.dqeeq),
            (&mut self.dqeff, &o.dqeff),
            (&mut self.dqeff0, &o.dqeff0),
        ];
        for (a, b) in pairs {
            for (x, y) in a.iter_mut().zip(b) {
                *x += f * y;
            }
        }
    }
}

/// Density-dependent parts of the explicit gradient ([`Setup::explicit_parts`]).
pub struct ExplicitParts {
    /// Total density.
    pub ptot: Vec<f64>,
    /// Overlap weight dE/dS (symmetric).
    pub x: Vec<f64>,
    /// Atomic dipole and quadrupole potentials.
    pub vd: Vec<[f64; 3]>,
    pub vq: Vec<[f64; 6]>,
    /// Second- to fourth-order ES, AES kernels, Mulliken-exchange kernel and revD4
    /// two-body terms.
    pub chain: Chain,
}

/// A density and the atomic multipole potentials contracted with the derivatives of its
/// atomic multipoles ([`Setup::add_overlap_multipole`]).
pub struct MultipoleTerm<'a> {
    pub p: &'a [f64],
    pub vd: &'a [[f64; 3]],
    pub vq: &'a [[f64; 6]],
}

/// Add `f * v` to `g[i]` and subtract it from `g[j]`.
#[inline]
fn add_pair(g: &mut [[f64; 3]], i: usize, j: usize, f: f64, v: [f64; 3]) {
    for k in 0..3 {
        g[i][k] += f * v[k];
        g[j][k] -= f * v[k];
    }
}

impl Setup {
    /// Shells of the SCF basis differentiated with respect to the effective charge.
    fn scf_basis_dq(&self) -> Result<Vec<Vec<Cgto>>> {
        let qparam = QvszpParams::v1_1();
        (0..self.mol.nat()).map(|i| atom_shells_dq(&qparam, self.mol.numbers[i], self.basis.qeff[i].qeff)).collect()
    }

    /// Sum_mu,nu X_mu,nu dS_mu,nu/dR for a symmetric weight matrix `x` plus, for every
    /// term (P, v_d, v_Q), Sum_A v_d,A . dd_A(P)/dR + v_Q,A . dQ_A(P)/dR of the atomic
    /// multipoles of P at fixed P (SCF basis): centres to `ch.g`, effective charges to
    /// `ch.dqeff`. One pass over the atom pairs: the overlap and multipole derivatives of a
    /// shell pair come from one integral call, moments around both atoms.
    pub fn add_overlap_multipole(&self, x: &[f64], terms: &[MultipoleTerm], ch: &mut Chain) -> Result<()> {
        let n = self.nao;
        let shells = &self.basis.scf.cgto;
        let dsh = self.scf_basis_dq()?;
        let first = ao_offsets(shells);
        let mut mg = MultipoleGrad::default();
        let mut scr = Scratch::default();
        let z = || ([0.0; MBLK], [[0.0; 3]; MBLK], [[0.0; 6]; MBLK]);
        // q_eff derivatives: around R_A (wrt q_A, q_B), around R_B (wrt q_B, q_A)
        let (mut sa1, mut da1, mut qa1) = z();
        let (mut sa2, mut da2, mut qa2) = z();
        let (mut sb1, mut db1, mut qb1) = z();
        let (mut sb2, mut db2, mut qb2) = z();
        let pos = &self.mol.positions;
        let nat = self.mol.nat();
        let vdot = |vd: &[f64; 3], vq: &[f64; 6], d: &[f64; 3], q: &[f64; 6]| -> f64 {
            vd[0] * d[0] + vd[1] * d[1] + vd[2] * d[2] + (0..6).map(|c| vq[c] * q[c]).sum::<f64>()
        };
        for ia in 0..nat {
            for ib in 0..ia {
                let vec = sub3(pos[ia], pos[ib]);
                let mvec = [-vec[0], -vec[1], -vec[2]];
                for (isa, sa) in shells[ia].iter().enumerate() {
                    for (isb, sb) in shells[ib].iter().enumerate() {
                        multipole_grad(sb, sa, vec, crate::ints::INTCUT, &mut mg, &mut scr);
                        // around R_A: d/dq_A (ket), d/dq_B (bra); blocks k = b + nb a
                        multipole(sb, &dsh[ia][isa], vec, AES_INTCUT, &mut sa1, &mut da1, &mut qa1, &mut scr);
                        multipole(&dsh[ib][isb], sa, vec, AES_INTCUT, &mut sa2, &mut da2, &mut qa2, &mut scr);
                        if !terms.is_empty() {
                            // around R_B: d/dq_B (ket), d/dq_A (bra); blocks k = a + na b
                            multipole(sa, &dsh[ib][isb], mvec, AES_INTCUT, &mut sb1, &mut db1, &mut qb1, &mut scr);
                            multipole(&dsh[ia][isa], sb, mvec, AES_INTCUT, &mut sb2, &mut db2, &mut qb2, &mut scr);
                        }
                        let (na, nb) = (2 * sa.l + 1, 2 * sb.l + 1);
                        for a in 0..na {
                            let mu = first[ia][isa] + a;
                            for b in 0..nb {
                                let nu = first[ib][isb] + b;
                                let k = b + nb * a;
                                let kr = a + na * b;
                                let w = 2.0 * x[mu * n + nu];
                                let mut dg = [w * mg.ds[k][0], w * mg.ds[k][1], w * mg.ds[k][2]];
                                // overlap q_eff derivative (same as the moments' s)
                                let mut dqa = w * sa1[k];
                                let mut dqb = w * sa2[k];
                                for t in terms {
                                    let pmn = t.p[mu * n + nu];
                                    if pmn == 0.0 {
                                        continue;
                                    }
                                    let (vda, vqa, vdb, vqb) = (&t.vd[ia], &t.vq[ia], &t.vd[ib], &t.vq[ib]);
                                    for c in 0..3 {
                                        dg[c] -= pmn
                                            * (vdot(vda, vqa, &mg.dd_i[k][c], &mg.dq_i[k][c])
                                                + vdot(vdb, vqb, &mg.dd_j[k][c], &mg.dq_j[k][c]));
                                    }
                                    dqa -=
                                        pmn * (vdot(vda, vqa, &da1[k], &qa1[k]) + vdot(vdb, vqb, &db2[kr], &qb2[kr]));
                                    dqb -=
                                        pmn * (vdot(vda, vqa, &da2[k], &qa2[k]) + vdot(vdb, vqb, &db1[kr], &qb1[kr]));
                                }
                                for c in 0..3 {
                                    ch.g[ia][c] += dg[c];
                                    ch.g[ib][c] -= dg[c];
                                }
                                ch.dqeff[ia] += dqa;
                                ch.dqeff[ib] += dqb;
                            }
                        }
                    }
                }
            }
            // onsite blocks: moments depend on the effective charge only (the onsite overlap
            // is the unit matrix: no two shells of an atom share l)
            if terms.is_empty() {
                continue;
            }
            for (isa, sa) in shells[ia].iter().enumerate() {
                for (isb, sb) in shells[ia].iter().enumerate() {
                    multipole(sb, &dsh[ia][isa], [0.0; 3], AES_INTCUT, &mut sa1, &mut da1, &mut qa1, &mut scr);
                    multipole(&dsh[ia][isb], sa, [0.0; 3], AES_INTCUT, &mut sa2, &mut da2, &mut qa2, &mut scr);
                    let (na, nb) = (2 * sa.l + 1, 2 * sb.l + 1);
                    for a in 0..na {
                        let mu = first[ia][isa] + a;
                        for b in 0..nb {
                            let nu = first[ia][isb] + b;
                            let k = b + nb * a;
                            for t in terms {
                                let pmn = t.p[mu * n + nu];
                                let (vd, vq) = (&t.vd[ia], &t.vq[ia]);
                                ch.dqeff[ia] -= pmn * (vdot(vd, vq, &da1[k], &qa1[k]) + vdot(vd, vq, &da2[k], &qa2[k]));
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Tr(P dH0/dR) at fixed total density `p`: levels (g-xTB CN), distance polynomial, the
    /// diatomic-frame scaled overlap (centres, bond direction, H0-basis effective charges).
    pub fn add_h0(&self, p: &[f64], ch: &mut Chain) -> Result<()> {
        let n = self.nao;
        let mol = &self.mol;
        let nat = mol.nat();
        let qparam = QvszpParams::v1_1();
        let param = &self.param;
        let mut sh = Vec::with_capacity(nat);
        let mut dsh = Vec::with_capacity(nat);
        for i in 0..nat {
            let q = h0_effective_charge(param, mol.numbers[i], &self.basis.qeff[i])?;
            sh.push(h0_atom_shells(param, &qparam, mol.numbers[i], q)?);
            dsh.push(h0_atom_shells_dq(param, &qparam, mol.numbers[i], q)?);
        }
        let first = ao_offsets(&sh);
        let kw = k_wolfsberg(param);
        let kp = k_poly(param);
        let mut level = vec![0.0; n];
        let mut dlevel = vec![0.0; n];
        for ia in 0..nat {
            let e = param.element(mol.numbers[ia])?;
            for (is, s) in sh[ia].iter().enumerate() {
                for a in 0..2 * s.l + 1 {
                    let mu = first[ia][is] + a;
                    level[mu] = -(e.lev[is] - e.lev_cn[is] * self.cn[ia]);
                    dlevel[mu] = e.lev_cn[is];
                    ch.dcn[ia] += p[mu * n + mu] * dlevel[mu];
                }
            }
        }
        let mut mg = MultipoleGrad::default();
        let mut scr = Scratch::default();
        let (mut ba, mut bb) = ([0.0; MBLK], [0.0; MBLK]);
        let pos = &mol.positions;
        for ia in 0..nat {
            let ea = param.element(mol.numbers[ia])?;
            for ib in 0..ia {
                let eb = param.element(mol.numbers[ib])?;
                let rab = sub3(pos[ib], pos[ia]);
                let r = norm2_sq(rab).sqrt();
                let nv = [rab[0] / r, rab[1] / r, rab[2] / r];
                let ksig = 2.0 / (1.0 / ea.h0k[0] + 1.0 / eb.h0k[0]);
                let kpi = 2.0 / (1.0 / ea.h0k[1] + 1.0 / eb.h0k[1]);
                let kmu = [ksig, kpi, K_DELTA];
                let rc = 0.5 * (RCOV_COM[mol.numbers[ia] - 1] + RCOV_COM[mol.numbers[ib] - 1]);
                let vec = sub3(pos[ia], pos[ib]);
                let mut tdir = [0.0; 3]; // dE/dn (free n)
                let mut dedr = 0.0; // distance polynomial
                for (isa, sa) in sh[ia].iter().enumerate() {
                    for (isb, sb) in sh[ib].iter().enumerate() {
                        multipole_grad(sb, sa, vec, H0_INTCUT, &mut mg, &mut scr);
                        overlap(sb, &dsh[ia][isa], vec, H0_INTCUT, &mut ba);
                        overlap(&dsh[ib][isb], sa, vec, H0_INTCUT, &mut bb);
                        let (la, lb) = (sa.l, sb.l);
                        let (na, nb) = (2 * la + 1, 2 * lb + 1);
                        let raw: Vec<f64> = (0..na * nb).map(|x| mg.s[(x % nb) + nb * (x / nb)]).collect();
                        let pia = 1.0 + ea.poly * kp[la] * r / rc;
                        let pib = 1.0 + eb.poly * kp[lb] * r / rc;
                        let (dpia, dpib) = (ea.poly * kp[la] / rc, eb.poly * kp[lb] / rc);
                        let kk = kw[la] + kw[lb];
                        // weights of S~ (2 for the two triangles)
                        let mut wst = vec![0.0; na * nb];
                        let mut wcn = vec![0.0; na * nb];
                        for a in 0..na {
                            let i = first[ia][isa] + a;
                            for b in 0..nb {
                                let j = first[ib][isb] + b;
                                let pij = 2.0 * p[i * n + j];
                                wst[a * nb + b] = pij * 0.25 * (level[i] + level[j]) * kk * pia * pib;
                                wcn[a * nb + b] = pij * 0.25 * kk * pia * pib;
                            }
                        }
                        // scaled block and the weight transformed back: M = sum_mu k_mu P^a W P^b
                        let scaled = la <= 2 && lb <= 2;
                        let (pa, dpa) = if scaled { bond_projectors_deriv(la, nv) } else { Default::default() };
                        let (pb, dpb) = if scaled { bond_projectors_deriv(lb, nv) } else { Default::default() };
                        let mm = |x: &[f64], y: &[f64], r: usize, k: usize, c: usize| -> Vec<f64> {
                            let mut o = vec![0.0; r * c];
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
                        };
                        let (st, mw) = if scaled {
                            let mut st = vec![0.0; na * nb];
                            let mut mw = vec![0.0; na * nb];
                            for mu in 0..=la.min(lb) {
                                let a1 = mm(&mm(&pa[mu], &raw, na, na, nb), &pb[mu], na, nb, nb);
                                let a2 = mm(&mm(&pa[mu], &wst, na, na, nb), &pb[mu], na, nb, nb);
                                for x in 0..na * nb {
                                    st[x] += kmu[mu] * a1[x];
                                    mw[x] += kmu[mu] * a2[x];
                                }
                                // bond direction: tr(W^T dP^a S P^b) + tr(W^T P^a S dP^b)
                                let sp = mm(&raw, &pb[mu], na, nb, nb);
                                let ps = mm(&pa[mu], &raw, na, na, nb);
                                for c in 0..3 {
                                    let t1 = mm(&dpa[mu][c], &sp, na, na, nb);
                                    let t2 = mm(&ps, &dpb[mu][c], na, nb, nb);
                                    let mut t = 0.0;
                                    for x in 0..na * nb {
                                        t += wst[x] * (t1[x] + t2[x]);
                                    }
                                    tdir[c] += kmu[mu] * t;
                                }
                            }
                            (st, mw)
                        } else {
                            (raw.clone(), wst.clone())
                        };
                        for a in 0..na {
                            let i = first[ia][isa] + a;
                            for b in 0..nb {
                                let j = first[ib][isb] + b;
                                let x = a * nb + b;
                                let k = b + nb * a;
                                for c in 0..3 {
                                    ch.g[ia][c] += mw[x] * mg.ds[k][c];
                                    ch.g[ib][c] -= mw[x] * mg.ds[k][c];
                                }
                                ch.dqeff0[ia] += mw[x] * ba[k];
                                ch.dqeff0[ib] += mw[x] * bb[k];
                                ch.dcn[ia] += wcn[x] * st[x] * dlevel[i];
                                ch.dcn[ib] += wcn[x] * st[x] * dlevel[j];
                                let pij = 2.0 * p[i * n + j];
                                dedr += pij * 0.25 * (level[i] + level[j]) * kk * (dpia * pib + pia * dpib) * st[x];
                            }
                        }
                    }
                }
                // n = (R_b - R_a)/r: dn/dR_b = (1 - n n^T)/r
                let tn: f64 = (0..3).map(|c| tdir[c] * nv[c]).sum();
                let tperp = [0, 1, 2].map(|c| (tdir[c] - tn * nv[c]) / r);
                add_pair(&mut ch.g, ib, ia, 1.0, tperp);
                add_pair(&mut ch.g, ib, ia, dedr, nv);
            }
        }
        Ok(())
    }

    /// Tr(P dH_ACP/dR) at fixed total density `p`.
    pub fn add_acp(&self, p: &[f64], ch: &mut Chain) -> Result<()> {
        let n = self.nao;
        let mol = &self.mol;
        let nat = mol.nat();
        let shells = &self.basis.scf.cgto;
        let dsh = self.scf_basis_dq()?;
        let first = ao_offsets(shells);
        let mut mg = MultipoleGrad::default();
        let mut scr = Scratch::default();
        let mut bq = [0.0; MBLK];
        let mut blk = [0.0; MBLK];
        let pos = &mol.positions;
        let pm = faer::MatRef::from_row_major_slice(p, n, n);
        for ip in 0..nat {
            let e = self.param.element(mol.numbers[ip])?;
            let sqcn = self.cn[ip].sqrt();
            let f = 1.0 + e.acpcn * sqcn;
            for l in 0..4 {
                if e.acp_coeff[l] * f == 0.0 {
                    continue;
                }
                let w = e.acp_coeff[l] * f;
                let proj = projector(l, e.acp_exp[l]);
                let np = 2 * l + 1;
                // overlaps of all AOs with the projector components (n x np) and P times them
                let mut col = faer::Mat::<f64>::zeros(n, np);
                for ia in 0..nat {
                    let vec = sub3(pos[ia], pos[ip]);
                    if norm2_sq(vec) > ACP_R2 {
                        continue;
                    }
                    for (is, s) in shells[ia].iter().enumerate() {
                        overlap(&proj, s, vec, ACP_INTCUT, &mut blk);
                        for a in 0..2 * s.l + 1 {
                            for m in 0..np {
                                col[(first[ia][is] + a, m)] = blk[m + np * a];
                            }
                        }
                    }
                }
                let mut pc = faer::Mat::<f64>::zeros(n, np);
                faer::linalg::matmul::matmul(pc.as_mut(), faer::Accum::Replace, pm, col.as_ref(), 1.0, self.par);
                if sqcn > 0.0 {
                    let er: f64 = (0..np).map(|m| (0..n).map(|mu| col[(mu, m)] * pc[(mu, m)]).sum::<f64>()).sum();
                    ch.dcn[ip] += e.acp_coeff[l] * e.acpcn / (2.0 * sqcn) * er;
                }
                for ia in 0..nat {
                    let vec = sub3(pos[ia], pos[ip]);
                    if norm2_sq(vec) > ACP_R2 {
                        continue;
                    }
                    for (is, s) in shells[ia].iter().enumerate() {
                        if ia != ip {
                            multipole_grad(&proj, s, vec, ACP_INTCUT, &mut mg, &mut scr);
                        }
                        overlap(&proj, &dsh[ia][is], vec, ACP_INTCUT, &mut bq);
                        for a in 0..2 * s.l + 1 {
                            let mu = first[ia][is] + a;
                            for m in 0..np {
                                let fac = 2.0 * w * pc[(mu, m)];
                                let k = m + np * a;
                                if ia != ip {
                                    for c in 0..3 {
                                        ch.g[ia][c] += fac * mg.ds[k][c];
                                        ch.g[ip][c] -= fac * mg.ds[k][c];
                                    }
                                }
                                ch.dqeff[ia] += fac * bq[k];
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    /// Repulsion: distances, EEQ-BC charges (effective nuclear charges) and g-xTB CN.
    pub fn add_repulsion(&self, ch: &mut Chain) -> Result<()> {
        let mol = &self.mol;
        let n = mol.nat();
        let g = &self.param.glob;
        let q = &self.eeq.charges;
        let mut zeff = vec![0.0; n];
        let mut dz = vec![0.0; n];
        let mut alpha = vec![0.0; n];
        let mut dalpha = vec![0.0; n];
        let mut k1 = vec![0.0; n];
        for i in 0..n {
            let z = mol.numbers[i];
            let r = &self.param.element(z)?.rep;
            zeff[i] = (1.0 - r[4] * q[i] - r[5] * q[i] * q[i]) * r[2];
            dz[i] = (-r[4] - 2.0 * r[5] * q[i]) * r[2];
            let sq = self.cn[i].sqrt();
            alpha[i] = (r[3] * sq + 1.0) * r[0];
            dalpha[i] = if sq > 0.0 { r[0] * r[3] / (2.0 * sq) } else { 0.0 };
            k1[i] = if z <= 2 { g[8] } else { g[3] };
        }
        for i in 0..n {
            let ri = &self.param.element(mol.numbers[i])?.rep;
            for j in i + 1..n {
                let rj = &self.param.element(mol.numbers[j])?.rep;
                let v = sub3(mol.positions[i], mol.positions[j]);
                let r = norm2_sq(v).sqrt();
                let a = 2.0 / (1.0 / alpha[j] + 1.0 / alpha[i]);
                let x = (ri[1] * rj[1]).sqrt() + r;
                let x15 = x.sqrt() * x;
                let damp = (-(x15 * a)).exp();
                let (r2, r3) = (r * r, r * r * r);
                let poly = (k1[j] + k1[i]) * 0.5 / r + 1.0 + g[12] / r2 + g[13] / r3 + g[14] / (r * r3);
                let dpoly =
                    -(k1[j] + k1[i]) * 0.5 / r2 - 2.0 * g[12] / r3 - 3.0 * g[13] / (r * r3) - 4.0 * g[14] / (r2 * r3);
                let zz = zeff[i] * zeff[j];
                let dedr = zz * damp * (-1.5 * a * x.sqrt() * poly + dpoly);
                add_pair(&mut ch.g, i, j, dedr / r, v);
                ch.dqeeq[i] += dz[i] * zeff[j] * damp * poly;
                ch.dqeeq[j] += zeff[i] * dz[j] * damp * poly;
                let dade = -zz * damp * poly * x15; // dE/da
                let s = alpha[i] + alpha[j];
                ch.dcn[i] += dade * 2.0 * alpha[j] * alpha[j] / (s * s) * dalpha[i];
                ch.dcn[j] += dade * 2.0 * alpha[i] * alpha[i] / (s * s) * dalpha[j];
            }
        }
        Ok(())
    }

    /// First- to fourth-order terms at fixed shell charges `q`: kernels (distance, Hubbard
    /// parameters through the g-xTB CN) and the CN-dependent chemical potentials.
    pub fn add_isotropic_es(&self, q: &[f64], ch: &mut Chain) -> Result<()> {
        let mol = &self.mol;
        let sh = &self.shells;
        let ns = sh.len();
        let qa = atom_charges(mol.nat(), sh, q);
        let drho = reference_drho(mol, sh);
        let u = hubbard_u(&self.param, mol, sh, &self.cn)?;
        let mut du = vec![0.0; ns]; // dU_i/dCN_A(i)
        for (i, &(iat, ish, _)) in sh.iter().enumerate() {
            let z = mol.numbers[iat];
            let e = self.param.element(z)?;
            let sq = self.cn[iat].sqrt();
            if sq > 0.0 {
                du[i] = e.gam2[ish] * GAM[z - 1] * e.g2cn / (2.0 * sq);
                // first order: mu_l = gam1 (1 + e1cn sqrt(CN))
                let x = qa[iat];
                let f = 1.0 + K1_DIS * (erf_f64(x - K1_S) + erf_f64(x + K1_S));
                ch.dcn[iat] += e.gam1[ish] * e.e1cn / (2.0 * sq) * q[i] * f;
            }
        }
        let k = self.param.glob[11];
        let gfac: Vec<f64> = sh.iter().map(|&(iat, _, _)| qa[iat] * GAM3[mol.numbers[iat] - 1]).collect();
        for i in 0..ns {
            for j in 0..=i {
                let (ia, ja) = (sh[i].0, sh[j].0);
                let same = ia == ja;
                let mult = if i == j { 1.0 } else { 2.0 };
                // dE/dgamma (symmetric pair) and dE/dtau
                let mut wg = mult * 0.5 * q[i] * q[j];
                if !same {
                    wg -= drho[j] * q[i] + drho[i] * q[j];
                }
                let wt = if i == j { q[i] * q[j] * gfac[i] / 3.0 } else { q[i] * q[j] * (gfac[i] + gfac[j]) / 3.0 };
                let r = if same { R_SAME_ATOM } else { norm2_sq(sub3(mol.positions[ia], mol.positions[ja])).sqrt() };
                let eta = 2.0 / (1.0 / u[i] + 1.0 / u[j]);
                let ex = (-k * r).exp();
                let den = r + ex / eta;
                let gam = 1.0 / den;
                let dgdr = -(1.0 - k * ex / eta) / (den * den);
                let dgde = ex / (eta * eta * den * den);
                let (dtdr, dtde) = if same {
                    (0.0, -gam * dgde)
                } else {
                    let s = r / eta;
                    let e = (-1.15 * s).exp();
                    let dphi = 0.5 * e * ((3.2 - 1.84 * s) - 1.84 * s - 1.15 * s * (3.2 - 1.84 * s));
                    (dphi / eta, -dphi * r / (eta * eta))
                };
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
        }
        Ok(())
    }

    /// Anisotropic electrostatics at fixed charges and multipoles: kernels and the explicit
    /// distance vectors (`full = false`: without the dipole-quadrupole and
    /// quadrupole-quadrupole parts).
    pub fn add_aes_explicit(&self, q: &[f64], dip: &[[f64; 3]], qp: &[[f64; 6]], full: bool, ch: &mut Chain) {
        let mol = &self.mol;
        let n = mol.nat();
        let g = &self.param.glob;
        let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
        let qm = |q: &[f64; 6]| [[q[0], q[1], q[3]], [q[1], q[2], q[4]], [q[3], q[4], q[5]]];
        let mv = |m: &[[f64; 3]; 3], v: [f64; 3]| [dot(m[0], v), dot(m[1], v), dot(m[2], v)];
        for i in 0..n {
            for j in 0..i {
                let r = sub3(mol.positions[j], mol.positions[i]);
                let r2 = dot(r, r);
                let rr = r2.sqrt();
                if rr > AES_RMAX {
                    continue;
                }
                let r0 = vdw_rad_pair(mol.numbers[i], mol.numbers[j]);
                // kernels g_n = c_n (1 - erf(-a (R - R0)))/R^n and dg_n/dR
                let kern = |c: f64, a: f64, p: i32| {
                    let f = 1.0 - erf_f64(-a * (rr - r0));
                    let df = 2.0 * a / SQRTPI * (-(a * (rr - r0)).powi(2)).exp();
                    (c * f / rr.powi(p), c * (df / rr.powi(p) - p as f64 * f / rr.powi(p + 1)))
                };
                let (g3, dg3) = kern(0.5 * g[15], KDMP3, 3);
                let (g5, dg5) = kern(0.5 * g[16], KDMP5, 5);
                let (g7, dg7) = kern(0.5 * g[17], KDMP5, 7);
                let (g9, dg9) = kern(0.5 * K9, KDMP5, 9);
                let (qi, qj) = (qm(&qp[i]), qm(&qp[j]));
                let (di, dj) = (dip[i], dip[j]);
                let (qir, qjr) = (mv(&qi, r), mv(&qj, r));
                let (rqir, rqjr) = (dot(r, qir), dot(r, qjr));
                let (dir, djr) = (dot(di, r), dot(dj, r));
                let e_qd = q[j] * dir - q[i] * djr;
                let grad_qd = [0, 1, 2].map(|c| q[j] * di[c] - q[i] * dj[c]);
                let e_qq = q[j] * rqir + q[i] * rqjr;
                let grad_qq = [0, 1, 2].map(|c| 2.0 * (q[j] * qir[c] + q[i] * qjr[c]));
                let e_dd = dot(di, dj) * r2 - 3.0 * dir * djr;
                let grad_dd = [0, 1, 2].map(|c| 2.0 * dot(di, dj) * r[c] - 3.0 * (djr * di[c] + dir * dj[c]));
                let mut gv = [0.0; 3];
                let mut dedr = e_qd * dg3 + (e_qq + e_dd) * dg5;
                for c in 0..3 {
                    gv[c] += g3 * grad_qd[c] + g5 * (grad_qq[c] + grad_dd[c]);
                }
                if full {
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
                    dedr += -e_dq * dg7 + e_q2 * dg9 / 3.0;
                    for c in 0..3 {
                        gv[c] += -g7 * grad_dq[c] + g9 * grad_q2[c] / 3.0;
                    }
                }
                for c in 0..3 {
                    gv[c] += dedr * r[c] / rr;
                }
                // r = R_j - R_i
                add_pair(&mut ch.g, j, i, 1.0, gv);
            }
        }
    }

    /// Mulliken exchange: derivative of the kernel (distance) at fixed spin densities.
    pub fn add_mfx_kernel(&self, p: &[Vec<f64>], ch: &mut Chain) -> Result<()> {
        let n = self.nao;
        let mol = &self.mol;
        let closed = p.len() == 1;
        // T_mu,nu = dE/dgamma_mu,nu
        let mut t = vec![0.0; n * n];
        for ps in p {
            let psm = matmul(n, ps, &self.s, self.par);
            let sps = matmul(n, &self.s, &psm, self.par);
            let scale = if closed { -0.125 } else { -0.25 };
            for i in 0..n {
                for j in 0..n {
                    let ij = i * n + j;
                    t[ij] += scale * (psm[ij] * psm[j * n + i] + ps[ij] * sps[ij]);
                }
            }
        }
        let sh = &self.shells;
        let ns = sh.len();
        let mut tsh = vec![0.0; ns * ns];
        for i in 0..n {
            for j in 0..n {
                tsh[self.basis.scf.ao2sh[i] * ns + self.basis.scf.ao2sh[j]] += t[i * n + j];
            }
        }
        let g = &self.param.glob;
        let omega = g[18];
        let gl = [g[4], g[5], g[5], g[5]];
        let mut ux = Vec::with_capacity(ns);
        let mut pol = Vec::with_capacity(ns);
        for &(iat, ish, _) in sh {
            let z = mol.numbers[iat];
            ux.push(self.param.element(z)?.fam[ish] * GAM[z - 1]);
            pol.push(crate::data::BAS_POL[ish * 103 + z - 1]);
        }
        for i in 0..ns {
            for j in 0..i {
                let (ia, ja) = (sh[i].0, sh[j].0);
                if ia == ja {
                    continue;
                }
                let xi = (pol[i].max(pol[j]) + 1) as f64;
                let eta = (xi - 1.0).exp2() * (ux[i] * ux[j]).powf(0.5 * xi) / (ux[i] + ux[j]).powf(xi - 1.0);
                let gg = (gl[sh[i].2] * gl[sh[j].2]).sqrt();
                let v = sub3(mol.positions[ia], mol.positions[ja]);
                let r = norm2_sq(v).sqrt();
                let r0 = vdw_rad_pair(mol.numbers[ia], mol.numbers[ja]);
                let kap = g[6] + g[7] * r0;
                let damp = (-kap * r).exp();
                let num = ALPHA_FX + (1.0 - ALPHA_FX) * erf_f64(omega * r);
                let dnum = (1.0 - ALPHA_FX) * 2.0 * omega / SQRTPI * (-(omega * r) * (omega * r)).exp();
                let den = r + damp / (eta * gg);
                let dden = 1.0 - kap * damp / (eta * gg);
                let dgam = (dnum * den - num * dden) / (den * den);
                let w = tsh[i * ns + j] + tsh[j * ns + i];
                add_pair(&mut ch.g, ia, ja, w * dgam / r, v);
            }
        }
        Ok(())
    }

    /// Chain rules: effective charges -> EEQ-BC charges and basis CN; CNs and EEQ-BC charges
    /// -> geometry. Returns the gradient.
    pub fn finish_chain(&self, mut ch: Chain) -> Result<Vec<[f64; 3]>> {
        let mol = &self.mol;
        let n = mol.nat();
        let qparam = QvszpParams::v1_1();
        for i in 0..n {
            let el = qparam.element(mol.numbers[i])?;
            let q = self.eeq.charges[i];
            let cnb = self.basis.cn[i];
            let sq = cnb.sqrt();
            let dqterm = 1.0 - 2.0 * el.k_q2 * q;
            let dcnterm = if sq > 0.0 { el.k_cn / (2.0 * sq) } else { 0.0 };
            let k = self.param.element(mol.numbers[i])?.kqvszp;
            // SCF basis: q_eff = q_term + cn_term + qcn_term
            ch.dqeeq[i] += ch.dqeff[i] * (dqterm + el.k_qcn * cnb);
            ch.dcnb[i] += ch.dqeff[i] * (dcnterm + el.k_qcn * q);
            // H0 basis: k0 q_term + k2 cn_term + k3 qcn_term
            ch.dqeeq[i] += ch.dqeff0[i] * (k[0] * dqterm + k[2] * el.k_qcn * cnb);
            ch.dcnb[i] += ch.dqeff0[i] * (k[1] * dcnterm + k[2] * el.k_qcn * q);
        }
        let mut g = ch.g;
        // CNs: g-xTB (erf, glob(11)), basis (3.75), revD4
        let kg = self.param.glob[10];
        for i in 0..n {
            for j in 0..i {
                let v = sub3(mol.positions[i], mol.positions[j]);
                let r = norm2_sq(v).sqrt();
                let ri = self.param.element(mol.numbers[i])?.rad;
                let rj = self.param.element(mol.numbers[j])?.rad;
                let rc = 0.5 * (ri + rj);
                let arg = kg * (r - rc) / rc;
                let dc = kg / rc * (-arg * arg).exp() / SQRTPI;
                let rcb = RCOV_BASIS_CN[mol.numbers[i] - 1] + RCOV_BASIS_CN[mol.numbers[j] - 1];
                let argb = -KCN_BASIS * (r - rcb) / rcb;
                let dcb = -KCN_BASIS / rcb * (-argb * argb).exp() / SQRTPI;
                let f = (ch.dcn[i] + ch.dcn[j]) * dc + (ch.dcnb[i] + ch.dcnb[j]) * dcb;
                add_pair(&mut g, i, j, f / r, v);
            }
        }
        RevD4::add_cn_gradient(mol, &ch.dcnd4, &mut g);
        eeqbc_charge_gradient(&EeqBcParams::v1_1(), mol, &ch.dqeeq, &mut g)?;
        Ok(g)
    }

    /// The density-dependent parts of the explicit gradient at fixed AO spin densities `p`:
    /// the weights of the integral derivatives and the chain of the remaining (cheap) terms.
    /// `full = false` differentiates the energy without the dipole-quadrupole and
    /// quadrupole-quadrupole AES parts.
    pub fn explicit_parts(&self, p: &[Vec<f64>], full: bool) -> Result<ExplicitParts> {
        let n = self.nao;
        let mol = &self.mol;
        let nat = mol.nat();
        let closed = p.len() == 1;
        let ptot = crate::fock::total_density(p);
        let q = crate::mulliken::shell_charges(&self.basis.scf, &self.s, &self.ref_occ, &ptot, false);
        let qa = atom_charges(nat, &self.shells, &q);
        let (dip, qp) = self.atomic_moments(&ptot);
        let mut ch = Chain::new(nat);
        // Mulliken potentials of the charge-dependent terms (exact derivatives)
        let mut vsh = isotropic_shell_potential(&self.param, mol, &self.shells, &self.cn, &q, &self.gamma2, &self.tau)?;
        let (vs, vd, vq) = aes_potentials(mol, &qa, &dip, &qp, &self.aes_kernels, full);
        let (_, vdisp) = self.disp.two_body_potential(&qa);
        for (ish, &(iat, _, _)) in self.shells.iter().enumerate() {
            vsh[ish] += vs[iat] + vdisp[iat];
        }
        // overlap weight: Mulliken charges, magnetizations, exchange
        let mut x = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                x[i * n + j] = -0.5 * (vsh[self.basis.scf.ao2sh[i]] + vsh[self.basis.scf.ao2sh[j]]) * ptot[i * n + j];
            }
        }
        if let Some(mag) = crate::mulliken::magnetizations(&self.basis.scf, &self.s, p, false) {
            let w = spin_shell_potential(&self.param, mol, &self.shells, &mag)?;
            for i in 0..n {
                for j in 0..n {
                    let pd = p[0][i * n + j] - p[1][i * n + j];
                    x[i * n + j] += 0.5 * (w[self.basis.scf.ao2sh[i]] + w[self.basis.scf.ao2sh[j]]) * pd;
                }
            }
        }
        for ps in p {
            let xs = self.exchange_overlap_weight(ps, closed);
            for k in 0..n * n {
                x[k] += xs[k];
            }
        }
        self.add_isotropic_es(&q, &mut ch)?;
        self.add_aes_explicit(&qa, &dip, &qp, full, &mut ch);
        self.add_mfx_kernel(p, &mut ch)?;
        let d2 = self.disp.two_body_gradient(&mol.positions, &qa, &mut ch.g);
        for i in 0..nat {
            ch.dcnd4[i] += d2[i];
        }
        Ok(ExplicitParts { ptot, x, vd, vq, chain: ch })
    }

    /// Integral derivatives (overlap and multipoles with the overlap weight `x` and the
    /// multipole `terms`, H0 and ACP with the density `p_one`), the density-independent terms
    /// (repulsion, three-body dispersion) and the chain rules, added to the chain `ch`.
    pub fn finish_explicit(
        &self,
        x: &[f64],
        terms: &[MultipoleTerm],
        p_one: &[f64],
        mut ch: Chain,
    ) -> Result<Vec<[f64; 3]>> {
        self.add_overlap_multipole(x, terms, &mut ch)?;
        self.add_h0(p_one, &mut ch)?;
        self.add_acp(p_one, &mut ch)?;
        self.add_repulsion(&mut ch)?;
        let d3 = self.disp.three_body_gradient(&mut ch.g);
        for i in 0..self.mol.nat() {
            ch.dcnd4[i] += d3[i];
        }
        self.finish_chain(ch)
    }

    /// Explicit gradient dE/dR at fixed AO spin densities `p` (closed shell: the total
    /// density), plus sum X dS/dR for an optional extra overlap weight `x_extra`.
    /// `full = false` differentiates the energy without the dipole-quadrupole and
    /// quadrupole-quadrupole AES parts.
    pub fn explicit_gradient(&self, p: &[Vec<f64>], full: bool, x_extra: Option<&[f64]>) -> Result<Vec<[f64; 3]>> {
        let mut e = self.explicit_parts(p, full)?;
        if let Some(xe) = x_extra {
            for (a, b) in e.x.iter_mut().zip(xe) {
                *a += b;
            }
        }
        let term = MultipoleTerm { p: &e.ptot, vd: &e.vd, vq: &e.vq };
        self.finish_explicit(&e.x, &[term], &e.ptot, e.chain)
    }

    /// d(E_MFX + E_OFX)/dS (symmetrized) of one spin density (closed shell: total density).
    pub fn exchange_overlap_weight(&self, ps: &[f64], closed: bool) -> Vec<f64> {
        let n = self.nao;
        let s = &self.s;
        let par = self.par;
        let g = &self.gamma_x;
        let sp = matmul(n, s, ps, par);
        let psm = matmul(n, ps, s, par);
        let mut gsp = vec![0.0; n * n];
        let mut gps = vec![0.0; n * n];
        let mut gp = vec![0.0; n * n];
        for k in 0..n * n {
            gsp[k] = g[k] * sp[k];
            gps[k] = g[k] * psm[k];
            gp[k] = g[k] * ps[k];
        }
        // MFX: G = -1/4 [P (g o SP) + (g o PS) P + (g o P) SP + PS (g o P)]
        let a1 = matmul(n, ps, &gsp, par);
        let a2 = matmul(n, &gps, ps, par);
        let a3 = matmul(n, &gp, &sp, par);
        let a4 = matmul(n, &psm, &gp, par);
        let scale = if closed { -0.125 } else { -0.25 };
        let mut x = vec![0.0; n * n];
        for k in 0..n * n {
            x[k] = scale * (a1[k] + a2[k] + a3[k] + a4[k]);
        }
        // OFX: G = 1/2 (P Y + Y P) with Y = dE/dPt (exact derivative)
        let y = self.ofx_dual_derivative(ps, closed);
        let py = matmul(n, ps, &y, par);
        let yp = matmul(n, &y, ps, par);
        // closed shell: E = 2 e(P/2), Y at P/2, so dE/dS = 1/2 (P Y + Y P) with the total P too
        for k in 0..n * n {
            x[k] += 0.5 * (py[k] + yp[k]);
        }
        // symmetrize
        let mut xs = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                xs[i * n + j] = 0.5 * (x[i * n + j] + x[j * n + i]);
            }
        }
        xs
    }

    /// Y = dE_OFX/dPt (onsite blocks) of one spin density (closed shell: total density,
    /// Pt from P/2); the exact derivative, population part included.
    pub fn ofx_dual_derivative(&self, ps: &[f64], closed: bool) -> Vec<f64> {
        use crate::data::GMUNU;
        use crate::exchange::{gmunu_index, OFX_SCALE};
        let n = self.nao;
        let ao = &self.ao;
        let scale = if closed { 0.5 } else { 1.0 };
        let psm = matmul(n, ps, &self.s, self.par);
        let mut pt = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                if ao[i].0 == ao[j].0 {
                    pt[i * n + j] = 0.5 * scale * (psm[i * n + j] + psm[j * n + i]);
                }
            }
        }
        let mut y = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                let (ai, si, li) = ao[i];
                let (aj, sj, lj) = ao[j];
                if ai != aj {
                    continue;
                }
                let Some(k) = gmunu_index(li, lj) else { continue };
                let lam = OFX_SCALE * GMUNU[(k - 1) + 9 * (self.mol.numbers[ai] - 1)];
                if i != j {
                    y[i * n + j] -= 2.0 * lam * pt[i * n + j];
                }
                if si == sj && li >= 1 {
                    let f = (if i == j { (2 * li + 1) as f64 } else { 0.0 } - 1.0) / (2 * li + 1) as f64;
                    y[i * n + i] -= 2.0 * f * lam * pt[j * n + j];
                }
            }
        }
        y
    }
}
