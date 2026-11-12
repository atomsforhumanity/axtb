//! Extended Hückel-type core Hamiltonian of the GFN-xTB methods and its nuclear gradient.

use crate::ao::{AoMatrices, AtomPair, ShellPair, ShellPairs};
use crate::basis::{nsph, Basis};
use crate::integrals::{
    multipole, multipole_grad, multipole_hess, multipole_raw, MultipoleGrad, MultipoleHess, Scratch, KL, MBLK,
};
use crate::molecule::Molecule;
use axtb_base::threads;
use faer::Mat;

/// Species-resolved parameters of the core Hamiltonian for one molecule.
#[derive(Clone, Debug)]
pub struct H0Params {
    /// Atomic levels per species and shell (Eh).
    pub selfenergy: Vec<Vec<f64>>,
    /// CN shift of the atomic levels per species and shell (Eh per CN).
    pub kcn: Vec<Vec<f64>>,
    /// Shell polynomial coefficients per species and shell.
    pub shpoly: Vec<Vec<f64>>,
    /// Reference occupation per species and shell.
    pub refocc: Vec<Vec<f64>>,
    /// Atomic radius per species for the shell polynomial (bohr).
    pub rad: Vec<f64>,
    /// Scaling of the off-diagonal elements, `hscale[isp][jsp][ish][jsh]`.
    pub hscale: Vec<Vec<Vec<Vec<f64>>>>,
}

/// AO matrices of one geometry. Moments are taken around the atom of the column AO:
/// `dip[k][(mu, nu)] = <mu| (r - R_atom(nu))_k |nu>`.
pub struct Integrals {
    pub overlap: Mat<f64>,
    pub h0: Mat<f64>,
    pub dip: [Mat<f64>; 3],
    pub quad: [Mat<f64>; 6],
}

impl H0Params {
    /// Shell-resolved atomic levels and their CN derivatives.
    pub fn selfenergy(&self, mol: &Molecule, bas: &Basis, cn: &[f64]) -> (Vec<f64>, Vec<f64>) {
        let mut se = vec![0.0; bas.nsh];
        let mut dsedcn = vec![0.0; bas.nsh];
        for iat in 0..mol.nat() {
            let isp = mol.id[iat];
            let is = bas.ish_at[iat];
            for ish in 0..bas.nsh_id[isp] {
                se[is + ish] = self.selfenergy[isp][ish] - self.kcn[isp][ish] * cn[iat];
                dsedcn[is + ish] = -self.kcn[isp][ish];
            }
        }
        (se, dsedcn)
    }

    /// Reference shell occupations and the number of electrons of the neutral atoms.
    pub fn reference_occupation(&self, mol: &Molecule, bas: &Basis) -> (Vec<f64>, f64) {
        let mut n0sh = vec![0.0; bas.nsh];
        let mut nel = 0.0;
        for iat in 0..mol.nat() {
            let isp = mol.id[iat];
            for ish in 0..bas.nsh_id[isp] {
                n0sh[bas.ish_at[iat] + ish] = self.refocc[isp][ish];
                nel += self.refocc[isp][ish];
            }
        }
        (n0sh, nel)
    }

    /// Build overlap, core Hamiltonian, dipole and quadrupole matrices
    /// ([`AoMatrices::build_with_scaled`]: each atom pair once, mirrored blocks shifted to the
    /// bra; H0 is the overlap scaled per shell pair).
    ///
    /// With `parallel`, atoms are distributed over the current rayon pool. Every matrix
    /// element is written by exactly one atom task (the larger atom index of its pair), so
    /// tasks never write the same memory and the result does not depend on the thread count.
    pub fn build(&self, mol: &Molecule, bas: &Basis, se: &[f64], cutoff: f64, parallel: bool) -> Integrals {
        let (ao, h0) = AoMatrices::build_with_scaled(mol, bas, cutoff, parallel, |p: &AtomPair| {
            let (isp, jsp) = (mol.id[p.i], mol.id[p.j]);
            // the shell polynomial of a pair; on-site: the average of the levels only
            let onsite = p.i == p.j;
            let rr = if onsite { 0.0 } else { (p.r2.sqrt() / (self.rad[jsp] + self.rad[isp])).sqrt() };
            move |s: &ShellPair| {
                if onsite {
                    0.5 * (se[s.si] + se[s.sj])
                } else {
                    let shpoly = (1.0 + self.shpoly[isp][s.ish] * rr) * (1.0 + self.shpoly[jsp][s.jsh] * rr);
                    0.5 * (se[s.si] + se[s.sj]) * self.hscale[jsp][isp][s.jsh][s.ish] * shpoly
                }
            }
        });
        let AoMatrices { overlap, dip, quad } = ao;
        Integrals { overlap, h0, dip, quad }
    }

    /// Nuclear gradient of the one-electron part of the energy.
    ///
    /// `p` is the total density matrix, `w` the energy-weighted density matrix, `vao` the
    /// AO-resolved charge potential and `vdp`, `vqp` the atomic dipole and quadrupole
    /// potentials of the final Fock matrix. Adds to `grad` and to `dedcn`.
    pub fn add_gradient(
        &self,
        mol: &Molecule,
        bas: &Basis,
        se: &[f64],
        dsedcn: &[f64],
        cutoff: f64,
        p: &Mat<f64>,
        w: &Mat<f64>,
        vao: &[f64],
        vdp: &[[f64; 3]],
        vqp: &[[f64; 6]],
        dedcn: &mut [f64],
        grad: &mut [[f64; 3]],
        parallel: bool,
    ) {
        let nat = mol.nat();
        // fixed chunks (the same for serial and parallel runs) reduced in order, so the
        // result does not depend on the thread count
        let pairs = ShellPairs::screened(mol, bas, cutoff, bas.intcut);
        let parts = threads::map_chunks(nat, parallel, |atoms| {
            let mut gr = vec![[0.0; 3]; nat];
            let mut dc = vec![0.0; nat];
            self.gradient_atoms(mol, &pairs, se, dsedcn, p, w, vao, vdp, vqp, atoms, &mut dc, &mut gr);
            (gr, dc)
        });
        for (gr, dc) in parts {
            for i in 0..nat {
                dedcn[i] += dc[i];
                for k in 0..3 {
                    grad[i][k] += gr[i][k];
                }
            }
        }
    }

    fn gradient_atoms(
        &self,
        mol: &Molecule,
        pairs: &ShellPairs,
        se: &[f64],
        dsedcn: &[f64],
        p: &Mat<f64>,
        w: &Mat<f64>,
        vao: &[f64],
        vdp: &[[f64; 3]],
        vqp: &[[f64; 6]],
        atoms: std::ops::Range<usize>,
        dedcn: &mut [f64],
        grad: &mut [[f64; 3]],
    ) {
        let bas = pairs.basis();
        let mut g = MultipoleGrad::default();
        let mut scr = Scratch::default();
        for iat in atoms {
            let isp = mol.id[iat];
            let is = bas.ish_at[iat];
            for jat in 0..iat {
                // screened shell pairs contribute exactly zero
                let Some(pair) = pairs.pair(iat, jat) else { continue };
                let (vec, r2) = (pair.vec, pair.r2);
                if r2 < f64::EPSILON {
                    continue;
                }
                let jsp = mol.id[jat];
                let rr = (r2.sqrt() / (self.rad[jsp] + self.rad[isp])).sqrt();
                let mut dg = [0.0; 3];
                let (mut dcni, mut dcnj) = (0.0, 0.0);
                pairs.for_each(&pair, |sp| {
                    let (ish, jsh, ci, cj) = (sp.ish, sp.jsh, sp.ci, sp.cj);
                    multipole_grad(cj, ci, vec, bas.intcut, &mut g, &mut scr);
                    let shpolyi = 1.0 + self.shpoly[isp][ish] * rr;
                    let shpolyj = 1.0 + self.shpoly[jsp][jsh] * rr;
                    let shpoly = shpolyi * shpolyj;
                    let dshpoly = (shpolyi * self.shpoly[jsp][jsh] + shpolyj * self.shpoly[isp][ish]) * 0.5 * rr / r2;
                    let dsv = [dshpoly / shpoly * vec[0], dshpoly / shpoly * vec[1], dshpoly / shpoly * vec[2]];
                    let hs = self.hscale[jsp][isp][jsh][ish] * shpoly;
                    let hij = 0.5 * (se[sp.si] + se[sp.sj]) * hs;
                    let dhdcni = dsedcn[sp.si] * hs;
                    let dhdcnj = dsedcn[sp.sj] * hs;
                    let nj = nsph(cj.l);
                    for iao in 0..nsph(ci.l) {
                        for jao in 0..nj {
                            let ij = jao + nj * iao;
                            let (mu, nu) = (sp.jj + jao, sp.ii + iao);
                            let pij = p[(mu, nu)];
                            let hpij = pij * hij;
                            let sval = -pij * (vao[mu] + vao[nu]) + 2.0 * hpij - 2.0 * w[(mu, nu)];
                            let ds = g.ds[ij];
                            let (ddi, ddj, dqi, dqj) = (&g.dd_i[ij], &g.dd_j[ij], &g.dq_i[ij], &g.dq_j[ij]);
                            for k in 0..3 {
                                let mut mp = 0.0;
                                for c in 0..3 {
                                    mp += ddi[k][c] * vdp[iat][c] + ddj[k][c] * vdp[jat][c];
                                }
                                for c in 0..6 {
                                    mp += dqi[k][c] * vqp[iat][c] + dqj[k][c] * vqp[jat][c];
                                }
                                dg[k] += sval * ds[k] + 2.0 * hpij * g.s[ij] * dsv[k] - pij * mp;
                            }
                            dcni += dhdcni * pij * g.s[ij];
                            dcnj += dhdcnj * pij * g.s[ij];
                        }
                    }
                });
                dedcn[iat] += dcni;
                dedcn[jat] += dcnj;
                for k in 0..3 {
                    grad[iat][k] += dg[k];
                    grad[jat][k] -= dg[k];
                }
            }
            // on-site: only the CN dependence of the atomic levels
            for ish in 0..bas.nsh_id[isp] {
                let ii = bas.iao_sh[is + ish];
                let mut dcni = 0.0;
                for iao in 0..bas.nao_sh[is + ish] {
                    dcni += dsedcn[is + ish] * p[(ii + iao, ii + iao)];
                }
                dedcn[iat] += dcni;
            }
        }
    }

    /// The AO pairs whose multipole terms the xtb program leaves out of its Fock matrix
    /// ([`MultipoleNeglect`]) at SCF accuracy `acc`.
    pub(crate) fn multipole_neglect(&self, mol: &Molecule, bas: &Basis, cutoff: f64, acc: f64) -> MultipoleNeglect {
        // scf_module.F90: neglect = 1e-8 acc; setdqlist compares squared elements with 1e-2 neglect
        let thr = 1.0e-10 * acc;
        let pairs = ShellPairs::unscreened(mol, bas, cutoff);
        let mut out = MultipoleNeglect { dip: Vec::new(), quad: Vec::new() };
        let mut s = [0.0; MBLK];
        let mut d = [[0.0; 3]; MBLK];
        let mut q = [[0.0; 6]; MBLK];
        let mut scr = Scratch::default();
        for iat in 0..mol.nat() {
            let rb = mol.positions[iat];
            for jat in 0..=iat {
                let Some(pair) = pairs.pair(iat, jat) else { continue };
                pairs.for_each(&pair, |sp| {
                    multipole_raw(sp.cj, sp.ci, pair.vec, bas.intcut, &mut s, &mut d, &mut q, &mut scr);
                    let nj = nsph(sp.cj.l);
                    for iao in 0..nsph(sp.ci.l) {
                        for jao in 0..nj {
                            let (mu, nu) = (sp.jj + jao, sp.ii + iao);
                            if mu > nu {
                                continue; // on-site: each unordered pair once
                            }
                            let ij = jao + nj * iao;
                            let (sv, dc, mc) = (s[ij], d[ij], q[ij]);
                            // moments around the coordinate origin (xtb's `point = 0`)
                            let dg = [dc[0] + rb[0] * sv, dc[1] + rb[1] * sv, dc[2] + rb[2] * sv];
                            let m2 = |a: usize, b: usize, k: usize| {
                                mc[k] + rb[a] * dc[b] + dc[a] * rb[b] + rb[a] * rb[b] * sv
                            };
                            let mg = [m2(0, 0, 0), m2(0, 1, 1), m2(1, 1, 2), m2(0, 2, 3), m2(1, 2, 4), m2(2, 2, 5)];
                            if dg.iter().map(|x| x * x).sum::<f64>() <= thr {
                                out.dip.push((mu, nu, jat, iat, dg));
                            }
                            let tq = mg[0] * mg[0]
                                + mg[2] * mg[2]
                                + mg[5] * mg[5]
                                + 2.0 * (mg[1] * mg[1] + mg[3] * mg[3] + mg[4] * mg[4]);
                            if tq <= thr {
                                out.quad.push((mu, nu, jat, iat, traceless6(mg)));
                            }
                        }
                    }
                });
            }
        }
        out
    }
}

/// `3/2 M - 1/2 tr(M)` of a symmetric matrix stored as xx, xy, yy, xz, yz, zz.
fn traceless6(m: [f64; 6]) -> [f64; 6] {
    let tr = 0.5 * (m[0] + m[2] + m[5]);
    [1.5 * m[0] - tr, 1.5 * m[1], 1.5 * m[2] - tr, 1.5 * m[3], 1.5 * m[4], 1.5 * m[5] - tr]
}

/// The xtb 6.7.1 program's neglect of small multipole integrals in its Fock matrix (xtb 6.7.1
/// conventions, [`ConventionParams::scc_multipole_neglect`](crate::ConventionParams::scc_multipole_neglect)). xtb's moment integrals are
/// taken around the coordinate origin, and `setdqlist` (`aespot.F90`) keeps the dipole term of
/// an AO pair only when Σ_k d_k² > 1e-10·acc and the quadrupole term only when
/// Σ_diag M² + 2 Σ_offdiag M² > 1e-10·acc (raw second moments); `buildIsoAnisotropicH1`
/// (`scc_core.f90`) leaves the rest out, while the overlap part of every potential stays.
/// Mulliken moments, energies and gradients use all integrals. At xtb's default accuracy 1
/// this moves the SCF solution by ~1e-8 e (caffeine), and the gradient by as much.
pub(crate) struct MultipoleNeglect {
    /// Pairs without the dipole term: AOs μ ≤ ν, their atoms, dipole around the origin.
    dip: Vec<(usize, usize, usize, usize, [f64; 3])>,
    /// Pairs without the quadrupole term: AOs μ ≤ ν, their atoms, `3/2 M - 1/2 tr M` of the
    /// second moments around the origin.
    quad: Vec<(usize, usize, usize, usize, [f64; 6])>,
}

impl MultipoleNeglect {
    /// Remove the left-out terms from a Fock matrix built with all integrals and the atomic
    /// dipole and quadrupole potentials `vdp`, `vqp`.
    pub(crate) fn apply(&self, mol: &Molecule, vdp: &[[f64; 3]], vqp: &[[f64; 6]], fock: &mut Mat<f64>) {
        let dot6 = |a: &[f64; 6], b: &[f64; 6]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
        let mut sub = |mu: usize, nu: usize, part: f64| {
            fock[(mu, nu)] -= part;
            if mu != nu {
                fock[(nu, mu)] -= part;
            }
        };
        for &(mu, nu, a, b, dg) in &self.dip {
            // the part of F(μ,ν) proportional to the origin dipole integral: the dipole
            // potentials and the dipole shift of the quadrupole moments around each atom
            let mut part = 0.0;
            for k in 0..3 {
                part -= 0.5 * dg[k] * (vdp[a][k] + vdp[b][k]);
            }
            for x in [a, b] {
                let r = mol.positions[x];
                let m = [
                    2.0 * r[0] * dg[0],
                    r[0] * dg[1] + r[1] * dg[0],
                    2.0 * r[1] * dg[1],
                    r[0] * dg[2] + r[2] * dg[0],
                    r[1] * dg[2] + r[2] * dg[1],
                    2.0 * r[2] * dg[2],
                ];
                part += 0.5 * dot6(&traceless6(m), &vqp[x]);
            }
            sub(mu, nu, part);
        }
        for &(mu, nu, a, b, tq) in &self.quad {
            let v = std::array::from_fn(|k| vqp[a][k] + vqp[b][k]);
            sub(mu, nu, -0.5 * dot6(&tq, &v));
        }
    }
}

/// First-order AO perturbations local to one atom A (see [`H0Params::perturbation_blocks`]).
///
/// A symmetric AO matrix `M` that is nonzero only in the rows and columns of atom A is
/// stored as its column block `L = M[:, A]` with the on-site block `M[A, A]` halved, so
/// that `M = L Eᵀ + E Lᵀ` with `E` the AO selector of atom A.
pub(crate) struct AtomBlocks {
    /// dF/dR_{A,k} at fixed potentials and coordination numbers (`nao x nao_A` each).
    pub fock: [Mat<f64>; 3],
    /// dS/dR_{A,k}.
    pub overlap: [Mat<f64>; 3],
    /// dH0/dCN_A.
    pub cn: Mat<f64>,
    /// Explicit derivative of the Mulliken moments at fixed density, d s / dR_{A,k}
    /// (SCC layout: shell charges, then 3 dipole and 6 quadrupole components per atom).
    pub b: [Vec<f64>; 3],
}

impl H0Params {
    /// First-order perturbations of every atom: Fock and overlap derivatives for each
    /// Cartesian displacement, the core-Hamiltonian derivative for a change of the atom's
    /// coordination number, and the explicit moment derivatives at fixed density `p`.
    /// `vao`, `vdp`, `vqp` are the potentials of the Fock matrix. Each atom is one task
    /// (parallel over atoms with `parallel`), so the result does not depend on threads.
    pub(crate) fn perturbation_blocks(
        &self,
        mol: &Molecule,
        bas: &Basis,
        se: &[f64],
        dsedcn: &[f64],
        cutoff: f64,
        p: &Mat<f64>,
        vao: &[f64],
        vdp: &[[f64; 3]],
        vqp: &[[f64; 6]],
        multipoles: bool,
        parallel: bool,
    ) -> Vec<AtomBlocks> {
        let task = |iat: usize| self.perturbation_atom(mol, bas, se, dsedcn, cutoff, p, vao, vdp, vqp, multipoles, iat);
        threads::map(mol.nat(), parallel, task)
    }

    fn perturbation_atom(
        &self,
        mol: &Molecule,
        bas: &Basis,
        se: &[f64],
        dsedcn: &[f64],
        cutoff: f64,
        p: &Mat<f64>,
        vao: &[f64],
        vdp: &[[f64; 3]],
        vqp: &[[f64; 6]],
        multipoles: bool,
        iat: usize,
    ) -> AtomBlocks {
        let nat = mol.nat();
        let nao = bas.nao;
        let nsh = bas.nsh;
        let ndim = nsh + if multipoles { 9 * nat } else { 0 };
        let dip = |a: usize, c: usize| nsh + 3 * a + c;
        let quad = |a: usize, c: usize| nsh + 3 * nat + 6 * a + c;
        let isp = mol.id[iat];
        let is = bas.ish_at[iat];
        let off = bas.iao_sh[is];
        let na: usize = (0..bas.nsh_id[isp]).map(|k| bas.nao_sh[is + k]).sum();
        let mut fock: [Mat<f64>; 3] = std::array::from_fn(|_| Mat::zeros(nao, na));
        let mut overlap: [Mat<f64>; 3] = std::array::from_fn(|_| Mat::zeros(nao, na));
        let mut cn = Mat::<f64>::zeros(nao, na);
        let mut b: [Vec<f64>; 3] = std::array::from_fn(|_| vec![0.0; ndim]);
        let pairs = ShellPairs::unscreened(mol, bas, cutoff);
        let mut g = MultipoleGrad::default();
        let mut scr = Scratch::default();
        for jat in 0..nat {
            if jat == iat {
                continue;
            }
            let Some(pair) = pairs.pair(iat, jat) else { continue };
            let (vec, r2) = (pair.vec, pair.r2);
            if r2 < f64::EPSILON {
                continue;
            }
            let jsp = mol.id[jat];
            let rr = (r2.sqrt() / (self.rad[jsp] + self.rad[isp])).sqrt();
            pairs.for_each(&pair, |sp| {
                let (ish, jsh, ci, cj) = (sp.ish, sp.jsh, sp.ci, sp.cj);
                multipole_grad(cj, ci, vec, bas.intcut, &mut g, &mut scr);
                let shpolyi = 1.0 + self.shpoly[isp][ish] * rr;
                let shpolyj = 1.0 + self.shpoly[jsp][jsh] * rr;
                let shpoly = shpolyi * shpolyj;
                let dshpoly = (shpolyi * self.shpoly[jsp][jsh] + shpolyj * self.shpoly[isp][ish]) * 0.5 * rr / r2;
                let dsv = [dshpoly / shpoly * vec[0], dshpoly / shpoly * vec[1], dshpoly / shpoly * vec[2]];
                let hs = self.hscale[jsp][isp][jsh][ish] * shpoly;
                let hij = 0.5 * (se[sp.si] + se[sp.sj]) * hs;
                let nj = nsph(cj.l);
                for iao in 0..nsph(ci.l) {
                    for jao in 0..nj {
                        let ij = jao + nj * iao;
                        let (mu, nu) = (sp.jj + jao, sp.ii + iao);
                        let col = nu - off;
                        let pij = p[(mu, nu)];
                        let ds = g.ds[ij];
                        let sv = g.s[ij];
                        let (ddi, ddj, dqi, dqj) = (&g.dd_i[ij], &g.dd_j[ij], &g.dq_i[ij], &g.dq_j[ij]);
                        for k in 0..3 {
                            let mut mp = 0.0;
                            for c in 0..3 {
                                mp += ddi[k][c] * vdp[iat][c] + ddj[k][c] * vdp[jat][c];
                            }
                            for c in 0..6 {
                                mp += dqi[k][c] * vqp[iat][c] + dqj[k][c] * vqp[jat][c];
                            }
                            fock[k][(mu, col)] =
                                hij * (sv * dsv[k] + ds[k]) - 0.5 * ds[k] * (vao[mu] + vao[nu]) - 0.5 * mp;
                            overlap[k][(mu, col)] = ds[k];
                            b[k][bas.ao2sh[nu]] -= pij * ds[k];
                            b[k][bas.ao2sh[mu]] -= pij * ds[k];
                            if multipoles {
                                for c in 0..3 {
                                    b[k][dip(iat, c)] -= pij * ddi[k][c];
                                    b[k][dip(jat, c)] -= pij * ddj[k][c];
                                }
                                for c in 0..6 {
                                    b[k][quad(iat, c)] -= pij * dqi[k][c];
                                    b[k][quad(jat, c)] -= pij * dqj[k][c];
                                }
                            }
                        }
                        cn[(mu, col)] = 0.5 * dsedcn[sp.si] * hs * sv;
                    }
                }
            });
        }
        // on-site block of dH0/dCN (halved)
        let mut s = [0.0; MBLK];
        let mut d = [[0.0; 3]; MBLK];
        let mut q = [[0.0; 6]; MBLK];
        for ish in 0..bas.nsh_id[isp] {
            let ci = &bas.cgto[isp][ish];
            let ii = bas.iao_sh[is + ish];
            for jsh in 0..bas.nsh_id[isp] {
                let cj = &bas.cgto[isp][jsh];
                let jj = bas.iao_sh[is + jsh];
                multipole(cj, ci, [0.0; 3], bas.intcut, &mut s, &mut d, &mut q, &mut scr);
                let nj = nsph(cj.l);
                for iao in 0..nsph(ci.l) {
                    for jao in 0..nj {
                        let ij = jao + nj * iao;
                        let (mu, nu) = (jj + jao, ii + iao);
                        cn[(mu, nu - off)] = 0.25 * (dsedcn[is + ish] + dsedcn[is + jsh]) * s[ij];
                    }
                }
            }
        }
        AtomBlocks { fock, overlap, cn, b }
    }
}

impl H0Params {
    /// Second derivatives of the one-electron energy functional
    /// `Φ(R) = Tr(P H0) − Tr(W S) − Σ_t V_t Tr(P O_t)` at fixed density `p`,
    /// energy-weighted density `w` and potentials (`vao`, `vdp`, `vqp`), including the
    /// coordination-number dependence of the self-energies (`jac` = dCN/dR of `ncoord`,
    /// row-major `nat x 3 nat`). Added to `hess` (row-major `3 nat x 3 nat`). This is the
    /// derivative of [`add_gradient`](Self::add_gradient) plus its CN chain rule.
    pub(crate) fn add_hessian(
        &self,
        mol: &Molecule,
        bas: &Basis,
        se: &[f64],
        dsedcn: &[f64],
        cutoff: f64,
        p: &Mat<f64>,
        w: &Mat<f64>,
        vao: &[f64],
        vdp: &[[f64; 3]],
        vqp: &[[f64; 6]],
        ncoord: &crate::ncoord::Ncoord,
        jac: &[f64],
        hess: &mut [f64],
        parallel: bool,
    ) {
        let nat = mol.nat();
        let n3 = 3 * nat;
        let task = |iat: usize| self.hessian_atom(mol, bas, se, dsedcn, cutoff, p, w, vao, vdp, vqp, iat);
        let rows = threads::map(nat, parallel, task);
        let mut dedcn = vec![0.0; nat];
        // X = d²Φ/dR dCN (n3 x nat)
        let mut x = vec![0.0; n3 * nat];
        for (iat, (hrow, xrow, dcn)) in rows.into_iter().enumerate() {
            dedcn[iat] = dcn;
            for k in 0..3 {
                let r = 3 * iat + k;
                for y in 0..n3 {
                    hess[r * n3 + y] += hrow[k * n3 + y];
                }
                x[r * nat..(r + 1) * nat].copy_from_slice(&xrow[k * nat..(k + 1) * nat]);
            }
        }
        // X J + (X J)ᵀ
        let mut xj = vec![0.0; n3 * n3];
        for r in 0..n3 {
            for c in 0..nat {
                let v = x[r * nat + c];
                if v != 0.0 {
                    let jrow = &jac[c * n3..(c + 1) * n3];
                    let out = &mut xj[r * n3..(r + 1) * n3];
                    for y in 0..n3 {
                        out[y] += v * jrow[y];
                    }
                }
            }
        }
        for r in 0..n3 {
            for y in 0..n3 {
                hess[r * n3 + y] += xj[r * n3 + y] + xj[y * n3 + r];
            }
        }
        ncoord.add_hessian(mol, &dedcn, hess);
    }

    /// Rows of atom `iat`: the pair second derivatives (3 x 3N), the CN cross terms
    /// d²Φ/dR_iat dCN (3 x nat) and dΦ/dCN_iat.
    fn hessian_atom(
        &self,
        mol: &Molecule,
        bas: &Basis,
        se: &[f64],
        dsedcn: &[f64],
        cutoff: f64,
        p: &Mat<f64>,
        w: &Mat<f64>,
        vao: &[f64],
        vdp: &[[f64; 3]],
        vqp: &[[f64; 6]],
        iat: usize,
    ) -> (Vec<f64>, Vec<f64>, f64) {
        let nat = mol.nat();
        let n3 = 3 * nat;
        let mut hrow = vec![0.0; 3 * n3];
        let mut xrow = vec![0.0; 3 * nat];
        let mut dcn = 0.0;
        let isp = mol.id[iat];
        let is = bas.ish_at[iat];
        let pairs = ShellPairs::unscreened(mol, bas, cutoff);
        let mut hs = Box::<MultipoleHess>::default();
        let mut scr = Scratch::default();
        for jat in 0..nat {
            if jat == iat {
                continue;
            }
            let Some(pair) = pairs.pair(iat, jat) else { continue };
            let (vec, r2) = (pair.vec, pair.r2);
            if r2 < f64::EPSILON {
                continue;
            }
            let jsp = mol.id[jat];
            let r = r2.sqrt();
            let rr = (r / (self.rad[jsp] + self.rad[isp])).sqrt();
            let drr = 0.5 * rr / r;
            let d2rr = -0.25 * rr / r2;
            let mut hvv = [[0.0; 3]; 3];
            pairs.for_each(&pair, |sp| {
                let (ish, jsh, ci, cj) = (sp.ish, sp.jsh, sp.ci, sp.cj);
                let ki = self.shpoly[isp][ish];
                let kj = self.shpoly[jsp][jsh];
                multipole_hess(cj, ci, vec, bas.intcut, &mut hs, &mut scr);
                let shp = (1.0 + ki * rr) * (1.0 + kj * rr);
                let sp1 = (ki + kj + 2.0 * ki * kj * rr) * drr;
                let sp2 = 2.0 * ki * kj * drr * drr + (ki + kj + 2.0 * ki * kj * rr) * d2rr;
                let shp_k = [sp1 * vec[0] / r, sp1 * vec[1] / r, sp1 * vec[2] / r];
                let mut shp_kl = [0.0; 6];
                for (q, &(k, l)) in KL.iter().enumerate() {
                    let d = if k == l { 1.0 / r } else { 0.0 };
                    shp_kl[q] = sp2 * vec[k] * vec[l] / r2 + sp1 * (d - vec[k] * vec[l] / (r2 * r));
                }
                let hsc = self.hscale[jsp][isp][jsh][ish];
                let hs0 = 0.5 * (se[sp.si] + se[sp.sj]) * hsc;
                let (dsei, dsej) = (dsedcn[sp.si], dsedcn[sp.sj]);
                let nj = nsph(cj.l);
                let g = &hs.g;
                for iao in 0..nsph(ci.l) {
                    for jao in 0..nj {
                        let ij = jao + nj * iao;
                        let (mu, nu) = (sp.jj + jao, sp.ii + iao);
                        let pij = p[(mu, nu)];
                        let wij = w[(mu, nu)];
                        let vsum = vao[mu] + vao[nu];
                        let sv = g.s[ij];
                        let ds = g.ds[ij];
                        for (q, &(k, l)) in KL.iter().enumerate() {
                            let mut mp = 0.0;
                            for c in 0..3 {
                                mp += hs.d2d_i[ij][q][c] * vdp[iat][c] + hs.d2d_j[ij][q][c] * vdp[jat][c];
                            }
                            for c in 0..6 {
                                mp += hs.d2q_i[ij][q][c] * vqp[iat][c] + hs.d2q_j[ij][q][c] * vqp[jat][c];
                            }
                            let d2s = hs.d2s[ij][q];
                            let val =
                                2.0 * pij * hs0 * (shp_kl[q] * sv + shp_k[k] * ds[l] + shp_k[l] * ds[k] + shp * d2s)
                                    - 2.0 * wij * d2s
                                    - pij * (d2s * vsum + mp);
                            hvv[k][l] += val;
                            if k != l {
                                hvv[l][k] += val;
                            }
                        }
                        for k in 0..3 {
                            let t = pij * hsc * (shp_k[k] * sv + shp * ds[k]);
                            xrow[k * nat + iat] += dsei * t;
                            xrow[k * nat + jat] += dsej * t;
                        }
                        dcn += dsei * pij * hsc * shp * sv;
                    }
                }
            });
            for k in 0..3 {
                for l in 0..3 {
                    hrow[k * n3 + 3 * iat + l] += hvv[k][l];
                    hrow[k * n3 + 3 * jat + l] -= hvv[k][l];
                }
            }
        }
        // on-site: only the CN dependence of the atomic levels
        for ish in 0..bas.nsh_id[isp] {
            let ii = bas.iao_sh[is + ish];
            for iao in 0..bas.nao_sh[is + ish] {
                dcn += dsedcn[is + ish] * p[(ii + iao, ii + iao)];
            }
        }
        (hrow, xrow, dcn)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gfn2;
    use crate::molecule::Molecule;

    /// E(R) = Tr(P H0) - Tr(W S) + Σ v_sh q_sh + Σ vdp·μ + Σ vqp·Θ at fixed P, W and potentials.
    fn model_energy(
        mol: &Molecule,
        p: &Mat<f64>,
        w: &Mat<f64>,
        vao: &[f64],
        vdp: &[[f64; 3]],
        vqp: &[[f64; 6]],
    ) -> f64 {
        let model = gfn2::build(mol, 1.0, &crate::Conventions::Tblite.params()).unwrap();
        let bas = &model.basis;
        let cn = model.ncoord.compute(mol);
        let (se, _) = model.h0.selfenergy(mol, bas, &cn);
        let ints = model.h0.build(mol, bas, &se, 40.0, false);
        let mut e = 0.0;
        for nu in 0..bas.nao {
            let at = bas.ao2at[nu];
            for mu in 0..bas.nao {
                let pmn = p[(mu, nu)];
                e += pmn * ints.h0[(mu, nu)] - w[(mu, nu)] * ints.overlap[(mu, nu)];
                e -= vao[nu] * pmn * ints.overlap[(mu, nu)];
                for k in 0..3 {
                    e -= vdp[at][k] * pmn * ints.dip[k][(mu, nu)];
                }
                for k in 0..6 {
                    e -= vqp[at][k] * pmn * ints.quad[k][(mu, nu)];
                }
            }
        }
        e
    }

    /// Gradient of the one-electron functional at fixed P, W and potentials (with CN chain).
    fn model_gradient(
        mol: &Molecule,
        p: &Mat<f64>,
        w: &Mat<f64>,
        vao: &[f64],
        vdp: &[[f64; 3]],
        vqp: &[[f64; 6]],
    ) -> Vec<[f64; 3]> {
        let model = gfn2::build(mol, 1.0, &crate::Conventions::Axtb.params()).unwrap();
        let bas = &model.basis;
        let cn = model.ncoord.compute(mol);
        let (se, dsedcn) = model.h0.selfenergy(mol, bas, &cn);
        let mut grad = vec![[0.0; 3]; mol.nat()];
        let mut dedcn = vec![0.0; mol.nat()];
        model.h0.add_gradient(mol, bas, &se, &dsedcn, f64::INFINITY, p, w, vao, vdp, vqp, &mut dedcn, &mut grad, false);
        model.ncoord.add_gradient(mol, &dedcn, &mut grad);
        grad
    }

    #[test]
    fn h0_hessian_matches_finite_difference() {
        let numbers = [8, 1, 1, 6, 26];
        let pos = [[0.1, 0.2, 0.22], [0.3, 1.43, -0.9], [-0.2, -1.4, -0.85], [2.9, 0.4, 1.7], [-2.6, 1.1, -1.9]];
        let mol = Molecule::new(&numbers, &pos, 0.0, None).unwrap();
        let model = gfn2::build(&mol, 1.0, &crate::Conventions::Axtb.params()).unwrap();
        let bas = &model.basis;
        let n = bas.nao;
        let mut seed = 4242u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        let mut p = Mat::<f64>::zeros(n, n);
        let mut w = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..=i {
                let (a, b) = (rnd(), rnd());
                p[(i, j)] = a;
                p[(j, i)] = a;
                w[(i, j)] = b;
                w[(j, i)] = b;
            }
        }
        let vao: Vec<f64> = (0..n).map(|_| rnd()).collect();
        let vdp: Vec<[f64; 3]> = (0..mol.nat()).map(|_| [rnd(), rnd(), rnd()]).collect();
        let vqp: Vec<[f64; 6]> = (0..mol.nat()).map(|_| std::array::from_fn(|_| rnd())).collect();
        let cn = model.ncoord.compute(&mol);
        let (se, dsedcn) = model.h0.selfenergy(&mol, bas, &cn);
        let n3 = 3 * mol.nat();
        let jac = model.ncoord.jacobian(&mol);
        let mut hess = vec![0.0; n3 * n3];
        model.h0.add_hessian(
            &mol,
            bas,
            &se,
            &dsedcn,
            f64::INFINITY,
            &p,
            &w,
            &vao,
            &vdp,
            &vqp,
            &model.ncoord,
            &jac,
            &mut hess,
            true,
        );
        let h = 1e-5;
        let mut worst: f64 = 0.0;
        let mut scale: f64 = 0.0;
        for y in 0..n3 {
            let mut m = mol.clone();
            m.positions[y / 3][y % 3] += h;
            let gp = model_gradient(&m, &p, &w, &vao, &vdp, &vqp);
            m.positions[y / 3][y % 3] -= 2.0 * h;
            let gm = model_gradient(&m, &p, &w, &vao, &vdp, &vqp);
            for x in 0..n3 {
                let fd = (gp[x / 3][x % 3] - gm[x / 3][x % 3]) / (2.0 * h);
                worst = worst.max((fd - hess[x * n3 + y]).abs());
                scale = scale.max(fd.abs());
            }
        }
        eprintln!("H0 Hessian: max |H| {scale:.3e}, max error {worst:.3e}");
        assert!(worst < 1e-7 * scale.max(1.0), "max error {worst:.3e}");
    }

    #[test]
    fn h0_gradient_matches_finite_difference() {
        let numbers = [8, 1, 1, 6];
        let pos = [[0.1, 0.2, 0.22], [0.3, 1.43, -0.9], [-0.2, -1.4, -0.85], [2.9, 0.4, 1.7]];
        let mol = Molecule::new(&numbers, &pos, 0.0, None).unwrap();
        let model = gfn2::build(&mol, 1.0, &crate::Conventions::Tblite.params()).unwrap();
        let bas = &model.basis;
        let n = bas.nao;
        // deterministic pseudo-random symmetric P and W, random potentials
        let mut seed = 12345u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 11) as f64 / (1u64 << 53) as f64) - 0.5
        };
        let mut p = Mat::<f64>::zeros(n, n);
        let mut w = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            for j in 0..=i {
                let (a, b) = (rnd(), rnd());
                p[(i, j)] = a;
                p[(j, i)] = a;
                w[(i, j)] = b;
                w[(j, i)] = b;
            }
        }
        let vao: Vec<f64> = (0..n).map(|_| rnd()).collect();
        let vdp: Vec<[f64; 3]> = (0..mol.nat()).map(|_| [rnd(), rnd(), rnd()]).collect();
        let vqp: Vec<[f64; 6]> = (0..mol.nat()).map(|_| std::array::from_fn(|_| rnd())).collect();

        let cn = model.ncoord.compute(&mol);
        let (se, dsedcn) = model.h0.selfenergy(&mol, bas, &cn);
        let mut grad = vec![[0.0; 3]; mol.nat()];
        let mut dedcn = vec![0.0; mol.nat()];
        model.h0.add_gradient(&mol, bas, &se, &dsedcn, 40.0, &p, &w, &vao, &vdp, &vqp, &mut dedcn, &mut grad, true);
        model.ncoord.add_gradient(&mol, &dedcn, &mut grad);

        let h = 1e-5;
        for i in 0..mol.nat() {
            for k in 0..3 {
                let mut m = mol.clone();
                m.positions[i][k] += h;
                let ep = model_energy(&m, &p, &w, &vao, &vdp, &vqp);
                m.positions[i][k] -= 2.0 * h;
                let em = model_energy(&m, &p, &w, &vao, &vdp, &vqp);
                let fd = (ep - em) / (2.0 * h);
                assert!((fd - grad[i][k]).abs() < 1e-6, "atom {i} dir {k}: fd {fd} analytic {}", grad[i][k]);
            }
        }
    }
}
