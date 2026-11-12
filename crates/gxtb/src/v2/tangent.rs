//! First geometry derivatives of the setup of g-xTB 2.0.1 at fixed AO density: the analytic
//! derivative matrices of the SCF-basis overlap, dipole and quadrupole integrals and of the
//! one-electron Hamiltonian H0 + ACP (the anisotropy term is treated as a dipole potential,
//! see [`super::analytic`]) for every Cartesian displacement.
//!
//! Every geometry dependence factors into local perturbations whose derivative matrices live
//! in the AO rows (and columns) of one atom: the displacement of an atom (centres, bond
//! direction and distance polynomial of H0, ACP projector overlaps), the effective charges
//! q_eff (SCF basis) and q_eff0 (H0 basis) of an atom, and its g-xTB coordination number
//! (H0 self-energies). The Cartesian derivative is the sum of these local blocks with the
//! Jacobians d q_eff/dR (EEQ-BC charges and basis CN), d q_eff0/dR and d CN/dR
//! ([`Jacobians`]); the ACP projectors of a displaced atom add a low-rank term.

use super::basis::{effective_charge_deriv, h0_shells, qeff_h0_weights, qeff_weights, scf_shells, INTCUT};
use super::cn::{basis_cn_gradient, gxtb_cn_gradient};
use super::h0::{acp_projectors, diat_scale, kshell, selfenergy, shell_poly};
use super::scf::Setup;
use crate::acp::projector;
use crate::h0::bond_projectors_deriv;
use axtb::basis::{ao_offsets, Cgto};
use axtb::integrals::{multipole, multipole_grad, overlap, MultipoleGrad, Scratch, MBLK};
use axtb::molecule::{norm2_sq, sub3};
use axtb::Result;

/// Jacobians (row-major N x 3N, row = atom, column = Cartesian displacement) of the
/// coordination numbers, the EEQ-BC charges and the effective charges.
#[derive(Clone, Debug)]
pub struct Jacobians {
    pub cn: Vec<f64>,
    pub cnb: Vec<f64>,
    pub cnd4: Vec<f64>,
    pub eeq: Vec<f64>,
    pub qeff: Vec<f64>,
    pub qeff0: Vec<f64>,
}

/// Derivative of the SCF-basis integrals local to one atom: rows (n_A x n, row-major) of the
/// overlap (symmetric: the full derivative is rows + transpose), rows and columns
/// (n x n_A, row-major; entries of other atoms' rows only) of each dipole and quadrupole
/// component (moments around the atom of the first index).
#[derive(Clone, Debug)]
pub struct IntegralBlock {
    pub s: Vec<f64>,
    pub d: [Vec<f64>; 3],
    pub dc: [Vec<f64>; 3],
    pub q: [Vec<f64>; 6],
    pub qc: [Vec<f64>; 6],
}

impl IntegralBlock {
    fn zeros(na: usize, n: usize) -> Self {
        let z = || vec![0.0; na * n];
        Self {
            s: z(),
            d: [z(), z(), z()],
            dc: [z(), z(), z()],
            q: [z(), z(), z(), z(), z(), z()],
            qc: [z(), z(), z(), z(), z(), z()],
        }
    }
}

/// One ACP projector shell: atom, weight and the overlaps of its 2l+1 components with all AOs
/// (`col[mu * np + m]`), with their derivatives with respect to the position (`dcol[c]`) and
/// the effective charge (`qcol`) of the AO's atom.
#[derive(Clone, Debug)]
pub struct AcpShell {
    pub atom: usize,
    pub w: f64,
    pub np: usize,
    pub col: Vec<f64>,
    pub dcol: [Vec<f64>; 3],
    pub qcol: Vec<f64>,
}

/// Local derivative blocks of one molecule.
#[derive(Clone, Debug)]
pub struct Tangents {
    pub n: usize,
    pub nat: usize,
    /// First AO and number of AOs of every atom.
    pub off: Vec<usize>,
    pub na: Vec<usize>,
    /// SCF-basis integrals: displacement (3 per atom) and effective charge of every atom.
    pub disp: Vec<[IntegralBlock; 3]>,
    pub qeff: Vec<IntegralBlock>,
    /// H0 + ACP rows (n_A x n; the derivative is rows + transpose): displacement, q_eff0, CN,
    /// and the ACP part of q_eff.
    pub h_disp: Vec<[Vec<f64>; 3]>,
    pub h_qeff0: Vec<Vec<f64>>,
    pub h_cn: Vec<Vec<f64>>,
    pub h_qeff: Vec<Vec<f64>>,
    pub acp: Vec<AcpShell>,
    pub jac: Jacobians,
    /// Set by [`Tangents::strip`].
    pub stripped: bool,
}

/// Dense derivative matrices (row-major n x n) of one Cartesian displacement.
#[derive(Clone, Debug)]
pub struct Tangent {
    pub s: Vec<f64>,
    pub d: [Vec<f64>; 3],
    pub q: [Vec<f64>; 6],
    pub h: Vec<f64>,
}

/// H0 block of an atom pair A, B (rows A, columns B) and its derivatives with respect to the
/// position of B (3), q_eff0 of A and B, and the CNs of A and B.
pub struct H0Pair {
    pub na: usize,
    pub nb: usize,
    pub h: Vec<f64>,
    pub d: [Vec<f64>; 7],
}

pub fn h0_pair(
    model: &super::Model,
    za: usize,
    zb: usize,
    ra: [f64; 3],
    rb: [f64; 3],
    sha: &[Cgto],
    shb: &[Cgto],
    dsha: &[Cgto],
    dshb: &[Cgto],
    cna: f64,
    cnb: f64,
    mg: &mut MultipoleGrad,
    scr: &mut Scratch,
) -> H0Pair {
    let na: usize = sha.iter().map(|s| 2 * s.l + 1).sum();
    let nb: usize = shb.iter().map(|s| 2 * s.l + 1).sum();
    let z = || vec![0.0; na * nb];
    let mut out = H0Pair { na, nb, h: z(), d: [z(), z(), z(), z(), z(), z(), z()] };
    let rab = sub3(rb, ra);
    let r = norm2_sq(rab).sqrt();
    let nv = [rab[0] / r, rab[1] / r, rab[2] / r];
    let kmu = diat_scale(za, zb);
    let radsum = model.h0_rad(za) + model.h0_rad(zb);
    let vec = sub3(ra, rb);
    let (mut ba, mut bb) = ([0.0; MBLK], [0.0; MBLK]);
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
    let mut oa = 0;
    for (isa, sa) in sha.iter().enumerate() {
        let (pia, dpia) = shell_poly(za, sa.l, r, radsum);
        let (sea, dsea) = selfenergy(za, isa, cna);
        let mut ob = 0;
        for (isb, sb) in shb.iter().enumerate() {
            let (pib, dpib) = shell_poly(zb, sb.l, r, radsum);
            let (seb, dseb) = selfenergy(zb, isb, cnb);
            multipole_grad(sb, sa, vec, INTCUT, mg, scr);
            overlap(sb, &dsha[isa], vec, INTCUT, &mut ba);
            overlap(&dshb[isb], sa, vec, INTCUT, &mut bb);
            let (la, lb) = (sa.l, sb.l);
            let (ma, mb) = (2 * la + 1, 2 * lb + 1);
            let pick = |src: &dyn Fn(usize) -> f64| -> Vec<f64> {
                (0..ma * mb).map(|x| src(x % mb + mb * (x / mb))).collect()
            };
            let raw = pick(&|k| mg.s[k]);
            let draw: [Vec<f64>; 3] = [0, 1, 2].map(|c| pick(&|k| -mg.ds[k][c]));
            let rqa = pick(&|k| ba[k]);
            let rqb = pick(&|k| bb[k]);
            let (st, dst, sqa, sqb) = if la <= 2 && lb <= 2 {
                let (pa, dpa) = bond_projectors_deriv(la, nv);
                let (pb, dpb) = bond_projectors_deriv(lb, nv);
                let mut st = vec![0.0; ma * mb];
                let mut dst = [vec![0.0; ma * mb], vec![0.0; ma * mb], vec![0.0; ma * mb]];
                let mut sqa = vec![0.0; ma * mb];
                let mut sqb = vec![0.0; ma * mb];
                for mu in 0..=la.min(lb) {
                    let k = kmu[mu];
                    let tr = |x: &[f64]| mm(&mm(&pa[mu], x, ma, ma, mb), &pb[mu], ma, mb, mb);
                    let a = tr(&raw);
                    let aqa = tr(&rqa);
                    let aqb = tr(&rqb);
                    // d/dn_d of Pa S Pb
                    let sp = mm(&raw, &pb[mu], ma, mb, mb);
                    let ps = mm(&pa[mu], &raw, ma, ma, mb);
                    let dn: Vec<Vec<f64>> = (0..3)
                        .map(|d| {
                            let t1 = mm(&dpa[mu][d], &sp, ma, ma, mb);
                            let t2 = mm(&ps, &dpb[mu][d], ma, mb, mb);
                            t1.iter().zip(&t2).map(|(x, y)| x + y).collect()
                        })
                        .collect();
                    for c in 0..3 {
                        let ac = tr(&draw[c]);
                        for x in 0..ma * mb {
                            let mut v = ac[x];
                            for d in 0..3 {
                                let dnd = (if c == d { 1.0 } else { 0.0 } - nv[c] * nv[d]) / r;
                                v += dn[d][x] * dnd;
                            }
                            dst[c][x] += k * v;
                        }
                    }
                    for x in 0..ma * mb {
                        st[x] += k * a[x];
                        sqa[x] += k * aqa[x];
                        sqb[x] += k * aqb[x];
                    }
                }
                (st, dst, sqa, sqb)
            } else {
                (raw, draw, rqa, rqb)
            };
            let ks = kshell(la, lb);
            let pre = 0.5 * (sea + seb) * ks;
            let pp = pia * pib;
            let dpp = dpia * pib + pia * dpib;
            for a in 0..ma {
                for b in 0..mb {
                    let x = a * mb + b;
                    let o = (oa + a) * nb + ob + b;
                    out.h[o] = pre * pp * st[x];
                    for c in 0..3 {
                        out.d[c][o] = pre * (dpp * nv[c] * st[x] + pp * dst[c][x]);
                    }
                    out.d[3][o] = pre * pp * sqa[x];
                    out.d[4][o] = pre * pp * sqb[x];
                    out.d[5][o] = 0.5 * dsea * ks * pp * st[x];
                    out.d[6][o] = 0.5 * dseb * ks * pp * st[x];
                }
            }
            ob += mb;
        }
        oa += 2 * sa.l + 1;
    }
    out
}

impl Setup {
    /// Jacobians of the coordination numbers, EEQ-BC charges and effective charges.
    pub fn jacobians(&self) -> Result<Jacobians> {
        let mol = &self.mol;
        let nat = mol.nat();
        let n3 = 3 * nat;
        type Chain<'f> = &'f dyn Fn(&[f64], &mut [[f64; 3]]) -> Result<()>;
        let rows = |f: Chain| -> Result<Vec<f64>> {
            let mut out = vec![0.0; nat * n3];
            for a in 0..nat {
                let mut w = vec![0.0; nat];
                w[a] = 1.0;
                let mut g = vec![[0.0; 3]; nat];
                f(&w, &mut g)?;
                for (k, v) in g.iter().flatten().enumerate() {
                    out[a * n3 + k] = *v;
                }
            }
            Ok(out)
        };
        let cn = rows(&|w, g| {
            gxtb_cn_gradient(&self.model, mol, w, g);
            Ok(())
        })?;
        let cnb = rows(&|w, g| {
            basis_cn_gradient(&self.model, mol, w, g);
            Ok(())
        })?;
        let cnd4 = rows(&|w, g| {
            super::disp::add_cn_gradient(&self.model, mol, w, g);
            Ok(())
        })?;
        // all rows from the setup's factorization of the EEQ-BC equations
        let eeqbc = super::eeqbc::eeqbc(&self.model, mol)?;
        let eeq = rows(&|w, g| {
            eeqbc.add_charge_gradient(&self.eeq_solution, w, g);
            Ok(())
        })?;
        let mut qeff = vec![0.0; nat * n3];
        let mut qeff0 = vec![0.0; nat * n3];
        for a in 0..nat {
            let z = mol.numbers[a];
            let (q, c) = (self.eeq.charges[a], self.basis.cn[a]);
            let (fq, fc) = effective_charge_deriv(qeff_weights(z), q, c);
            let (gq, gc) = effective_charge_deriv(qeff_h0_weights(z), q, c);
            for x in 0..n3 {
                let (e, b) = (eeq[a * n3 + x], cnb[a * n3 + x]);
                qeff[a * n3 + x] = fq * e + fc * b;
                qeff0[a * n3 + x] = gq * e + gc * b;
            }
        }
        Ok(Jacobians { cn, cnb, cnd4, eeq, qeff, qeff0 })
    }

    /// Local derivative blocks and Jacobians.
    pub fn tangents(&self) -> Result<Tangents> {
        let n = self.nao;
        let mol = &self.mol;
        let nat = mol.nat();
        let pos = &mol.positions;
        let shells = &self.basis.shells;
        let first = ao_offsets(shells);
        let off: Vec<usize> = (0..nat).map(|a| first[a][0]).collect();
        let na: Vec<usize> = (0..nat).map(|a| shells[a].iter().map(|s| 2 * s.l + 1).sum()).collect();
        let dsh: Vec<Vec<Cgto>> =
            (0..nat).map(|i| scf_shells(mol.numbers[i], self.basis.qeff[i], true)).collect::<Result<_>>()?;
        let mut disp: Vec<[IntegralBlock; 3]> =
            (0..nat).map(|a| [0, 1, 2].map(|_| IntegralBlock::zeros(na[a], n))).collect();
        let mut qb: Vec<IntegralBlock> = (0..nat).map(|a| IntegralBlock::zeros(na[a], n)).collect();
        let mut mg = MultipoleGrad::default();
        let mut scr = Scratch::default();
        let zb = || ([0.0; MBLK], [[0.0; 3]; MBLK], [[0.0; 6]; MBLK]);
        let (mut sa1, mut da1, mut qa1) = zb();
        let (mut sa2, mut da2, mut qa2) = zb();
        let (mut sb1, mut db1, mut qb1) = zb();
        let (mut sb2, mut db2, mut qb2) = zb();
        let cut2 = self.cutoff * self.cutoff;
        for ia in 0..nat {
            for ib in 0..ia {
                let vec = sub3(pos[ia], pos[ib]);
                if norm2_sq(vec) > cut2 {
                    continue;
                }
                let mvec = [-vec[0], -vec[1], -vec[2]];
                for (isa, sa) in shells[ia].iter().enumerate() {
                    for (isb, sbb) in shells[ib].iter().enumerate() {
                        multipole_grad(sbb, sa, vec, INTCUT, &mut mg, &mut scr);
                        multipole(sbb, &dsh[ia][isa], vec, INTCUT, &mut sa1, &mut da1, &mut qa1, &mut scr);
                        multipole(&dsh[ib][isb], sa, vec, INTCUT, &mut sa2, &mut da2, &mut qa2, &mut scr);
                        multipole(sa, &dsh[ib][isb], mvec, INTCUT, &mut sb1, &mut db1, &mut qb1, &mut scr);
                        multipole(&dsh[ia][isa], sbb, mvec, INTCUT, &mut sb2, &mut db2, &mut qb2, &mut scr);
                        let (ma, mb) = (2 * sa.l + 1, 2 * sbb.l + 1);
                        for a in 0..ma {
                            let mu = first[ia][isa] + a;
                            let ml = mu - off[ia];
                            for b in 0..mb {
                                let nu = first[ib][isb] + b;
                                let nl = nu - off[ib];
                                let k = b + mb * a;
                                let kr = a + ma * b;
                                for c in 0..3 {
                                    let blk = &mut disp[ia][c];
                                    blk.s[ml * n + nu] = mg.ds[k][c];
                                    for x in 0..3 {
                                        blk.d[x][ml * n + nu] = mg.dd_i[k][c][x];
                                        blk.dc[x][nu * na[ia] + ml] = mg.dd_j[k][c][x];
                                    }
                                    for x in 0..6 {
                                        blk.q[x][ml * n + nu] = mg.dq_i[k][c][x];
                                        blk.qc[x][nu * na[ia] + ml] = mg.dq_j[k][c][x];
                                    }
                                    let blk = &mut disp[ib][c];
                                    blk.s[nl * n + mu] = -mg.ds[k][c];
                                    for x in 0..3 {
                                        blk.d[x][nl * n + mu] = -mg.dd_j[k][c][x];
                                        blk.dc[x][mu * na[ib] + nl] = -mg.dd_i[k][c][x];
                                    }
                                    for x in 0..6 {
                                        blk.q[x][nl * n + mu] = -mg.dq_j[k][c][x];
                                        blk.qc[x][mu * na[ib] + nl] = -mg.dq_i[k][c][x];
                                    }
                                }
                                let blk = &mut qb[ia];
                                blk.s[ml * n + nu] = sa1[k];
                                for x in 0..3 {
                                    blk.d[x][ml * n + nu] = da1[k][x];
                                    blk.dc[x][nu * na[ia] + ml] = db2[kr][x];
                                }
                                for x in 0..6 {
                                    blk.q[x][ml * n + nu] = qa1[k][x];
                                    blk.qc[x][nu * na[ia] + ml] = qb2[kr][x];
                                }
                                let blk = &mut qb[ib];
                                blk.s[nl * n + mu] = sa2[k];
                                for x in 0..3 {
                                    blk.d[x][nl * n + mu] = db1[kr][x];
                                    blk.dc[x][mu * na[ib] + nl] = da2[k][x];
                                }
                                for x in 0..6 {
                                    blk.q[x][nl * n + mu] = qb1[kr][x];
                                    blk.qc[x][mu * na[ib] + nl] = qa2[k][x];
                                }
                            }
                        }
                    }
                }
            }
            // onsite moments depend on the effective charge only
            for (isa, sa) in shells[ia].iter().enumerate() {
                for (isb, sbb) in shells[ia].iter().enumerate() {
                    multipole(sbb, &dsh[ia][isa], [0.0; 3], INTCUT, &mut sa1, &mut da1, &mut qa1, &mut scr);
                    multipole(&dsh[ia][isb], sa, [0.0; 3], INTCUT, &mut sa2, &mut da2, &mut qa2, &mut scr);
                    let (ma, mb) = (2 * sa.l + 1, 2 * sbb.l + 1);
                    for a in 0..ma {
                        let ml = first[ia][isa] + a - off[ia];
                        for b in 0..mb {
                            let nu = first[ia][isb] + b;
                            let k = b + mb * a;
                            let blk = &mut qb[ia];
                            for x in 0..3 {
                                blk.d[x][ml * n + nu] = da1[k][x] + da2[k][x];
                            }
                            for x in 0..6 {
                                blk.q[x][ml * n + nu] = qa1[k][x] + qa2[k][x];
                            }
                        }
                    }
                }
            }
        }
        // H0
        let mut h_disp: Vec<[Vec<f64>; 3]> = (0..nat).map(|a| [0, 1, 2].map(|_| vec![0.0; na[a] * n])).collect();
        let mut h_qeff0: Vec<Vec<f64>> = (0..nat).map(|a| vec![0.0; na[a] * n]).collect();
        let mut h_cn: Vec<Vec<f64>> = (0..nat).map(|a| vec![0.0; na[a] * n]).collect();
        let mut h_qeff: Vec<Vec<f64>> = (0..nat).map(|a| vec![0.0; na[a] * n]).collect();
        let hsh: Vec<Vec<Cgto>> =
            (0..nat).map(|i| h0_shells(mol.numbers[i], self.basis.qeff_h0[i], false)).collect::<Result<_>>()?;
        let hdsh: Vec<Vec<Cgto>> =
            (0..nat).map(|i| h0_shells(mol.numbers[i], self.basis.qeff_h0[i], true)).collect::<Result<_>>()?;
        for ia in 0..nat {
            // onsite self-energies
            for (is, s) in hsh[ia].iter().enumerate() {
                let (_, de) = selfenergy(mol.numbers[ia], is, self.cn[ia]);
                for a in 0..2 * s.l + 1 {
                    let mu = first[ia][is] + a;
                    h_cn[ia][(mu - off[ia]) * n + mu] = 0.5 * de;
                }
            }
            for ib in 0..ia {
                let v = sub3(pos[ia], pos[ib]);
                if norm2_sq(v) > cut2 {
                    continue;
                }
                let pr = h0_pair(
                    &self.model,
                    mol.numbers[ia],
                    mol.numbers[ib],
                    pos[ia],
                    pos[ib],
                    &hsh[ia],
                    &hsh[ib],
                    &hdsh[ia],
                    &hdsh[ib],
                    self.cn[ia],
                    self.cn[ib],
                    &mut mg,
                    &mut scr,
                );
                for a in 0..pr.na {
                    let mu = off[ia] + a;
                    for b in 0..pr.nb {
                        let nu = off[ib] + b;
                        let o = a * pr.nb + b;
                        for c in 0..3 {
                            h_disp[ib][c][b * n + mu] += pr.d[c][o];
                            h_disp[ia][c][a * n + nu] -= pr.d[c][o];
                        }
                        h_qeff0[ia][a * n + nu] += pr.d[3][o];
                        h_qeff0[ib][b * n + mu] += pr.d[4][o];
                        h_cn[ia][a * n + nu] += pr.d[5][o];
                        h_cn[ib][b * n + mu] += pr.d[6][o];
                    }
                }
            }
        }
        // ACP: H = sum_k w_k s_k s_k^T
        let mut acp = Vec::new();
        let mut bq = [0.0; MBLK];
        let mut blk = [0.0; MBLK];
        for ip in 0..nat {
            for (l, alpha, w) in acp_projectors(mol.numbers[ip]) {
                if w == 0.0 {
                    continue;
                }
                let proj = projector(l, alpha);
                let np = 2 * l + 1;
                let mut sh = AcpShell {
                    atom: ip,
                    w,
                    np,
                    col: vec![0.0; n * np],
                    dcol: [vec![0.0; n * np], vec![0.0; n * np], vec![0.0; n * np]],
                    qcol: vec![0.0; n * np],
                };
                for ia in 0..nat {
                    let vec = sub3(pos[ia], pos[ip]);
                    if norm2_sq(vec) > cut2 {
                        continue;
                    }
                    for (is, s) in shells[ia].iter().enumerate() {
                        overlap(&proj, s, vec, INTCUT, &mut blk);
                        overlap(&proj, &dsh[ia][is], vec, INTCUT, &mut bq);
                        if ia != ip {
                            multipole_grad(&proj, s, vec, INTCUT, &mut mg, &mut scr);
                        }
                        for a in 0..2 * s.l + 1 {
                            let mu = first[ia][is] + a;
                            for m in 0..np {
                                let k = m + np * a;
                                sh.col[mu * np + m] = blk[k];
                                sh.qcol[mu * np + m] = bq[k];
                                if ia != ip {
                                    for c in 0..3 {
                                        sh.dcol[c][mu * np + m] = mg.ds[k][c];
                                    }
                                }
                            }
                        }
                    }
                }
                // rows of the AO atoms: w (u s^T) with u on the AOs of one atom
                for ia in 0..nat {
                    for ml in 0..na[ia] {
                        let mu = off[ia] + ml;
                        for m in 0..np {
                            let q = sh.qcol[mu * np + m];
                            for nu in 0..n {
                                let cnu = w * sh.col[nu * np + m];
                                h_qeff[ia][ml * n + nu] += q * cnu;
                                if ia != ip {
                                    for c in 0..3 {
                                        h_disp[ia][c][ml * n + nu] += sh.dcol[c][mu * np + m] * cnu;
                                    }
                                }
                            }
                        }
                    }
                }
                acp.push(sh);
            }
        }
        let jac = self.jacobians()?;
        Ok(Tangents { n, nat, off, na, disp, qeff: qb, h_disp, h_qeff0, h_cn, h_qeff, acp, jac, stripped: false })
    }
}

impl Tangents {
    /// Drop the dipole and quadrupole blocks and the H0 rows of the displacements and of
    /// q_eff (the analytic Hessian folds them into its Fock rows): keeps the overlap rows,
    /// the H0 rows of q_eff0 and the CN, the ACP projectors and the Jacobians.
    /// [`Tangents::cartesian`] is not available afterwards.
    pub fn strip(&mut self) {
        let clear = |b: &mut IntegralBlock| {
            for v in b.d.iter_mut().chain(b.dc.iter_mut()).chain(b.q.iter_mut()).chain(b.qc.iter_mut()) {
                *v = Vec::new();
            }
        };
        for blocks in &mut self.disp {
            blocks.iter_mut().for_each(clear);
        }
        self.qeff.iter_mut().for_each(clear);
        for h in &mut self.h_disp {
            h.iter_mut().for_each(|v| *v = Vec::new());
        }
        self.h_qeff.iter_mut().for_each(|v| *v = Vec::new());
        self.stripped = true;
    }

    fn add_rows(&self, m: &mut [f64], a: usize, rows: &[f64], w: f64) {
        let n = self.n;
        if w == 0.0 {
            return;
        }
        for ml in 0..self.na[a] {
            let mu = self.off[a] + ml;
            for nu in 0..n {
                let v = w * rows[ml * n + nu];
                m[mu * n + nu] += v;
                m[nu * n + mu] += v;
            }
        }
    }

    fn add_cols(&self, m: &mut [f64], a: usize, cols: &[f64], w: f64) {
        let n = self.n;
        let na = self.na[a];
        for nu in 0..n {
            for ml in 0..na {
                m[nu * n + self.off[a] + ml] += w * cols[nu * na + ml];
            }
        }
    }

    fn add_block(&self, t: &mut Tangent, a: usize, b: &IntegralBlock, w: f64) {
        if w == 0.0 {
            return;
        }
        let n = self.n;
        self.add_rows(&mut t.s, a, &b.s, w);
        for x in 0..3 {
            for ml in 0..self.na[a] {
                let mu = self.off[a] + ml;
                for nu in 0..n {
                    t.d[x][mu * n + nu] += w * b.d[x][ml * n + nu];
                }
            }
            self.add_cols(&mut t.d[x], a, &b.dc[x], w);
        }
        for x in 0..6 {
            for ml in 0..self.na[a] {
                let mu = self.off[a] + ml;
                for nu in 0..n {
                    t.q[x][mu * n + nu] += w * b.q[x][ml * n + nu];
                }
            }
            self.add_cols(&mut t.q[x], a, &b.qc[x], w);
        }
    }

    /// Dense derivative matrices of Cartesian displacement `x` (atom x/3, direction x%3).
    pub fn cartesian(&self, x: usize) -> Tangent {
        assert!(!self.stripped, "Tangents::cartesian after Tangents::strip");
        let n = self.n;
        let nat = self.nat;
        let n3 = 3 * nat;
        let z = || vec![0.0; n * n];
        let mut t = Tangent { s: z(), d: [z(), z(), z()], q: [z(), z(), z(), z(), z(), z()], h: z() };
        let (ax, c) = (x / 3, x % 3);
        self.add_block(&mut t, ax, &self.disp[ax][c], 1.0);
        let mut h = z();
        self.add_rows(&mut h, ax, &self.h_disp[ax][c], 1.0);
        for a in 0..nat {
            let (jq, jq0, jc) = (self.jac.qeff[a * n3 + x], self.jac.qeff0[a * n3 + x], self.jac.cn[a * n3 + x]);
            self.add_block(&mut t, a, &self.qeff[a], jq);
            self.add_rows(&mut h, a, &self.h_qeff[a], jq);
            self.add_rows(&mut h, a, &self.h_qeff0[a], jq0);
            self.add_rows(&mut h, a, &self.h_cn[a], jc);
        }
        // ACP projectors on the displaced atom: u = -d s/dR_A of every other atom's AOs
        for sh in self.acp.iter().filter(|s| s.atom == ax) {
            let np = sh.np;
            for m in 0..np {
                let u: Vec<f64> = (0..n).map(|mu| -sh.dcol[c][mu * np + m]).collect();
                for mu in 0..n {
                    if u[mu] == 0.0 {
                        continue;
                    }
                    for nu in 0..n {
                        let v = sh.w * u[mu] * sh.col[nu * np + m];
                        h[mu * n + nu] += v;
                        h[nu * n + mu] += v;
                    }
                }
            }
        }
        t.h = h;
        t
    }
}
