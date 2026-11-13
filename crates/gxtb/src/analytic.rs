//! Analytic nuclear Hessian of g-xTB 2.0.1.
//!
//! With the gradient written as a functional g_x(R; P, W) = dE/dR_x|_P - Tr(W dS/dR_x) of the
//! AO spin densities and the energy-weighted density, the Hessian is
//!
//! ```text
//! H_xy = E^xy|_{P,W} + sum_s Tr(F^(x)_s dP^y_s) - Tr(S^x dW^y)
//! ```
//!
//! with F^(x) = dF/dR_x at fixed P (the perturbation of the coupled-perturbed equations,
//! [`crate::response`]) and dP^y, dW^y their solution. The explicit part at fixed P and W
//! splits like the energy: the SCC variables s (Mulliken shell charges, atomic dipoles and
//! quadrupoles, shell magnetizations) depend on the geometry through the integrals at fixed P
//! (b^x = ds/dR_x|_P), the potentials V = dE/ds explicitly (c^x = dV/dR_x|_s, including the
//! exchange shell potential's overlap dependence and the anisotropy vectors as a dipole
//! potential), and
//!
//! ```text
//! E^xy = [integral second derivatives contracted with the gradient's weights]
//!      + b^x . K b^y + c^x . b^y + b^x . c^y          (K = d^2E/ds ds)
//!      + d^2E_x/dS dS [S^x, S^y] + exchange overlap-kernel cross terms
//!      + [kernel terms at fixed s] + [chain second derivatives]
//! ```
//!
//! This module holds the per-displacement quantities (F^(x), S^x, b^x, K b^x, c^x, the
//! kernel second derivatives and the exchange couplings).
//!
//! The exchange depends on the overlap through a homogeneous polynomial of degree 2 in S at
//! fixed P, so its overlap derivatives (Fock matrix, shell potential, dE/dS, dE/dG) follow
//! exactly from polarization identities: f(S + dS) - f(S - dS) = 2 df(dS) for quadratic f,
//! and dE/dS is linear in S.

use crate::cn::gxtb_cn;
use crate::coulomb::{AesCache, IsotropicCache};
use crate::dense::dot;
use crate::exchange::{bo_kernel, mfx_kernel, mfx_pair, Exchange, ResponseBase};
use crate::explicit::STEP;
use crate::fock::{add_charge_potential, add_potentials_rows};
use crate::gradient::{exchange_kernel_weights, Potentials};
use crate::h0::{anisotropy, anisotropy_gradient};
use crate::intermediates::Slots;
use crate::model::Model;
use crate::repulsion::Repulsion;
use crate::response::KernelData;
use crate::revd4::RevD4;
use crate::scf::{Setup, State};
use crate::tangent::{IntegralBlock, Tangents};
use axtb::molecule::{norm2_sq, sub3};
use axtb::scf::ScfVariables;
use axtb::terms::{Chain, ChainLayout, Geometry, MomentTerm, Values};
use axtb::{Molecule, Result};
use axtb_base::fd;
use faer::{Accum, Mat, MatMut, MatRef, Par};

/// The integral-free kernels of one geometry: coordination number, electrostatic, repulsion
/// and dispersion kernels and the anisotropy vectors (the exchange kernels' geometry derivatives
/// are evaluated on the fly from the positions; the [`Exchange`] only provides the shells).
pub struct Kernels<'a> {
    pub mol: Molecule,
    pub model: Model,
    pub cn: Vec<f64>,
    pub isotropic: IsotropicCache,
    pub aes: AesCache,
    pub rep: Repulsion,
    pub disp: RevD4,
    pub exch: &'a Exchange,
    pub aniso: Vec<[f64; 3]>,
    /// The setup's chain layout (the shape of the gradient's accumulators) and its slots.
    pub layout: &'a ChainLayout,
    pub(crate) slots: Slots,
    /// The setup's terms (they do not depend on the geometry; their caches are the fields above)
    /// and the values of the intermediates they read at this geometry (the g-xTB and revD4 CNs).
    pub(crate) terms: &'a crate::terms::Terms,
    pub values: Values,
}

impl<'a> Kernels<'a> {
    pub(crate) fn new(
        model: &Model,
        mol: &Molecule,
        exch: &'a Exchange,
        layout: &'a ChainLayout,
        slots: Slots,
        terms: &'a crate::terms::Terms,
    ) -> Result<Self> {
        let cn = gxtb_cn(model, mol);
        let disp = RevD4::new(model, mol)?;
        let mut values = Values::new(layout);
        values.set(slots.cn, cn.clone());
        values.set(slots.cnd4, disp.cn.clone());
        Ok(Self {
            mol: mol.clone(),
            model: *model,
            isotropic: IsotropicCache::new(model, mol, &exch.shells, &cn),
            aes: AesCache::new(model, mol, &cn),
            rep: Repulsion::new(model, mol, &cn),
            disp,
            exch,
            aniso: anisotropy(model, mol),
            cn,
            layout,
            slots,
            terms,
            values,
        })
    }

    /// The geometry of the terms at this geometry (one thread).
    fn term_geometry(&self) -> Geometry<'_> {
        Geometry { mol: &self.mol, layout: self.layout, values: &self.values, parallel: false }
    }

    /// Potentials at fixed SCC variables (packed): shell potential without the exchange,
    /// dipole potential with the anisotropy vectors, quadrupole potential; no spin part (the
    /// spin constants do not depend on the geometry).
    pub fn potentials(&self, st: &State, lay: &ScfVariables) -> Vec<f64> {
        let (tg, s, terms) = (self.term_geometry(), st.moments(), &self.terms);
        use crate::terms::buffer;
        let (nsh, nat) = (self.isotropic.shells.len(), self.mol.nat());
        let (mut piso, mut paes) = (buffer(&terms.iso, nsh, nat), buffer(&terms.aes, nsh, nat));
        let (mut prep, mut pdisp) = (buffer(&terms.rep, nsh, nat), buffer(&terms.disp, nsh, nat));
        terms.iso.potential(&tg, &self.isotropic, &s, &mut piso);
        terms.aes.potential(&tg, &self.aes, &s, &mut paes);
        terms.rep.potential(&tg, &self.rep, &s, &mut prep);
        terms.disp.potential(&tg, &self.disp, &s, &mut pdisp);
        let (mut v, mut vd, vq) = (piso.shell, paes.dipole, paes.quadrupole);
        for (ish, &(iat, _, _)) in self.isotropic.shells.iter().enumerate() {
            v[ish] += paes.atom[iat] + prep.atom[iat] + pdisp.atom[iat];
        }
        for (d, a) in vd.iter_mut().zip(&self.aniso) {
            for c in 0..3 {
                d[c] += a[c];
            }
        }
        let mut out = vec![0.0; lay.dim()];
        lay.pack(&v, &vd, &vq, None, &mut out);
        out
    }

    /// Gradient (flat 3N) of the kernel terms at fixed SCC variables, with the CN chains: first
    /// to fourth order, AES, repulsion, revD4 two-body, anisotropy vectors (the exchange kernels
    /// at fixed weights have a closed-form Hessian, [`Analytic::exchange_kernel_hessian`], and
    /// the revD4 three-body term its own, [`Setup::three_body_hessian`]).
    pub fn gradient(&self, st: &State) -> Vec<f64> {
        let mol = &self.mol;
        let nat = mol.nat();
        let (tg, s, terms) = (self.term_geometry(), st.moments(), &self.terms);
        let mut ch = Chain::new(self.layout, nat);
        terms.iso.gradient(&tg, &self.isotropic, &s, &mut ch);
        terms.aes.gradient(&tg, &self.aes, &s, &mut ch);
        terms.rep.gradient(&tg, &self.rep, &s, &mut ch);
        terms.disp.gradient(&tg, &self.disp, &s, &mut ch);
        anisotropy_gradient(&self.model, mol, &st.dip, &mut ch.g);
        let dcn = ch.weights(self.slots.cn).to_vec();
        crate::cn::gxtb_cn_gradient(&self.model, mol, &dcn, &mut ch.g);
        let d2 = ch.weights(self.slots.cnd4).to_vec();
        crate::revd4::add_cn_gradient(&self.model, mol, &d2, &mut ch.g);
        ch.g.concat()
    }
}

/// Per-displacement quantities of the analytic Hessian.
#[derive(Clone, Debug)]
pub struct Displacement {
    /// F^(x) per spin channel and S^x (row-major n x n).
    pub f: Vec<Vec<f64>>,
    pub s: Vec<f64>,
    /// b^x, K b^x and c^x (packed SCC layout).
    pub b: Vec<f64>,
    pub kb: Vec<f64>,
    pub c: Vec<f64>,
    /// Kernel terms at fixed s: column x of their Hessian (3N).
    pub hk: Vec<f64>,
    /// d^2E_x/dS dS [S^x, S^y] over y (3N) and the overlap-kernel cross terms
    /// d^2E_x/dS dG [S^x, G^y] (+ bond-order kernel) over y.
    pub hss: Vec<f64>,
    pub hsg: Vec<f64>,
    /// Dipole derivative at fixed density: q_A e_k for the displaced atom plus the change of
    /// the Mulliken charges and atomic dipoles through the integrals (e).
    pub dmu: [f64; 3],
}

/// Column y of the response part sum_s Tr(F^(x)_s dP^y_s) - Tr(S^x dW^y) of the Hessian,
/// contracted over every x from the local derivative rows ([`Analytic::response_column`]):
/// `h` holds all terms except those of the SCC-variable potentials of F^(x), which are
/// (c^x + K b^x) . `s` + b^x_q . `vx` (the compact part, assembled for all pairs at the end).
#[derive(Clone, Debug)]
pub struct ResponseColumn {
    pub h: Vec<f64>,
    /// Mulliken analysis of dP^y (packed SCC layout).
    pub s: Vec<f64>,
    /// Change of the exchange shell potential dE_x/dq along dP^y.
    pub vx: Vec<f64>,
}

/// The local rows of the analytic Hessian beyond the overlap and H0 rows of [`Tangents`].
struct LocalRows {
    /// Rows (n_A x n; the matrix is rows + transpose) of F^(x) at the reference potentials,
    /// i.e. H0 + ACP rows + the Mulliken shell potential times the overlap rows + the multipole
    /// potentials times the dipole and quadrupole rows and columns: displacement (3 per atom)
    /// and q_eff of every atom (q_eff0 and the CN have H0 rows only).
    f_disp: Vec<[Vec<f64>; 3]>,
    f_qeff: Vec<Vec<f64>>,
    /// Overlaps of every ACP projector component with all AOs as columns (n x ncomp) and the
    /// first column of each projector shell.
    acp_cols: Mat<f64>,
    acp_first: Vec<usize>,
    /// (dG/dR)/R of the MFX shell kernel (shell pairs i > j on different atoms) and (dK/dR)/R
    /// of the bond-order kernel (atom pairs i > j).
    dgr: Vec<f64>,
    dkr: Vec<f64>,
}

/// The reference data of the analytic Hessian of one converged state.
pub struct Analytic<'a> {
    pub setup: &'a Setup,
    pub st: &'a State,
    /// Local derivative rows: overlap rows, H0 rows of q_eff0 and CN, ACP projectors and
    /// Jacobians (the dipole, quadrupole and remaining H0 rows are folded into the Fock rows
    /// and into `bmat`, then dropped: [`Tangents::cartesian`] is not available on it).
    pub tan: Tangents,
    pub pot: Potentials,
    /// Dipole potential including the anisotropy vectors.
    pub vd_tot: Vec<[f64; 3]>,
    pub base: ResponseBase,
    pub lay: ScfVariables,
    /// Exchange weights dE/dG (shell pairs) and dE/dK (atom pairs).
    pub tsh: Vec<f64>,
    pub tat: Vec<f64>,
    /// b^x of every displacement (row-major 3N x dim, packed SCC layout).
    pub bmat: Vec<f64>,
    rows: LocalRows,
}

/// Rows (n_A x n) of the Fock tangent of one local block at the reference potentials: H0 rows
/// `h`, the Mulliken shell potential `vao` (per AO) times the overlap rows (F -= 1/2 (v_m + v_n)
/// dS) and the multipole potentials times the dipole and quadrupole rows and columns
/// (F -= 1/2 (R + R^T), R_ij = vd(A(i)).dD_ij + vq(A(i)).dQ_ij).
fn fock_rows(
    n: usize,
    ao_atom: &[usize],
    off: usize,
    na: usize,
    blk: &IntegralBlock,
    h: &[f64],
    vao: &[f64],
    vd: &[[f64; 3]],
    vq: &[[f64; 6]],
) -> Vec<f64> {
    let a = ao_atom[off];
    let mut out = h.to_vec();
    for ml in 0..na {
        let mu = off + ml;
        for nu in 0..n {
            let b = nu * na + ml;
            let o = ml * n + nu;
            let an = ao_atom[nu];
            let mut r = 0.0;
            for k in 0..3 {
                r += vd[a][k] * blk.d[k][o] + vd[an][k] * blk.dc[k][b];
            }
            for k in 0..6 {
                r += vq[a][k] * blk.q[k][o] + vq[an][k] * blk.qc[k][b];
            }
            out[o] -= 0.5 * (vao[mu] + vao[nu]) * blk.s[o] + 0.5 * r;
        }
    }
    out
}

/// SCC-variable change (packed layout) of one local block at the fixed densities: shell
/// charges -pop (overlap rows + transpose), atomic dipoles and quadrupoles (rows around the
/// block's atom, columns around the atom of the other AO), magnetizations.
fn block_moments(
    s: &Setup,
    lay: &ScfVariables,
    off: usize,
    na: usize,
    blk: &IntegralBlock,
    ptot: &[f64],
    spin: Option<(&[f64], &[f64])>,
) -> Vec<f64> {
    let n = s.nao;
    let a = s.basis.scf.ao2at[off];
    let mut v = vec![0.0; lay.dim()];
    for ml in 0..na {
        let mu = off + ml;
        let smu = s.basis.scf.ao2sh[mu];
        for nu in 0..n {
            let o = ml * n + nu;
            let b = nu * na + ml;
            let p = ptot[mu * n + nu];
            let pv = p * blk.s[o];
            v[smu] -= pv;
            v[s.basis.scf.ao2sh[nu]] -= pv;
            if let Some((pa, pb)) = spin {
                let m = (pa[mu * n + nu] - pb[mu * n + nu]) * blk.s[o];
                v[lay.mag(smu)] += m;
                v[lay.mag(s.basis.scf.ao2sh[nu])] += m;
            }
            let an = s.basis.scf.ao2at[nu];
            for k in 0..3 {
                v[lay.dip(a, k)] -= p * blk.d[k][o];
                v[lay.dip(an, k)] -= p * blk.dc[k][b];
            }
            for k in 0..6 {
                v[lay.quad(a, k)] -= p * blk.q[k][o];
                v[lay.quad(an, k)] -= p * blk.qc[k][b];
            }
        }
    }
    v
}

/// X + X^T of a row-major n x n matrix.
fn sym(n: usize, x: &[f64]) -> Vec<f64> {
    let mut out = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = x[i * n + j] + x[j * n + i];
        }
    }
    out
}

impl<'a> Analytic<'a> {
    /// `setup` must use sequential linear algebra (`par = Seq`).
    pub fn new(setup: &'a Setup, st: &'a State) -> Result<Self> {
        let mut tan = setup.tangents()?;
        let pot = setup.potentials(st);
        let vd_tot: Vec<[f64; 3]> =
            pot.vd.iter().zip(&setup.aniso).map(|(v, a)| [v[0] + a[0], v[1] + a[1], v[2] + a[2]]).collect();
        let base = setup.response_base(st, Par::Seq);
        let lay = ScfVariables::new(setup.shells.len(), setup.mol.nat(), true, st.p.len() == 2);
        let (tsh, tat) = exchange_kernel_weights(&setup.exch, &setup.s, &st.p, Par::Seq);
        let n = setup.nao;
        let nat = setup.mol.nat();
        let n3 = 3 * nat;
        let dim = lay.dim();
        // Fock rows at the reference potentials
        let vao: Vec<f64> = setup.basis.scf.ao2sh.iter().map(|&i| pot.vsh[i]).collect();
        let fr = |a: usize, blk: &IntegralBlock, h: &[f64]| {
            fock_rows(n, &setup.basis.scf.ao2at, tan.off[a], tan.na[a], blk, h, &vao, &vd_tot, &pot.vq)
        };
        let f_disp: Vec<[Vec<f64>; 3]> =
            (0..nat).map(|a| [0, 1, 2].map(|c| fr(a, &tan.disp[a][c], &tan.h_disp[a][c]))).collect();
        let f_qeff: Vec<Vec<f64>> = (0..nat).map(|a| fr(a, &tan.qeff[a], &tan.h_qeff[a])).collect();
        // b^x = (displacement block) + sum_A dq_eff,A/dR_x (q_eff block of A)
        let ptot = setup.total_density(&st.p);
        let spin = if st.p.len() == 2 { Some((st.p[0].as_slice(), st.p[1].as_slice())) } else { None };
        let mut bmat = vec![0.0; n3 * dim];
        for a in 0..nat {
            for c in 0..3 {
                let v = block_moments(setup, &lay, tan.off[a], tan.na[a], &tan.disp[a][c], &ptot, spin);
                bmat[(3 * a + c) * dim..(3 * a + c + 1) * dim].copy_from_slice(&v);
            }
        }
        let mut mq = vec![0.0; nat * dim];
        for a in 0..nat {
            let v = block_moments(setup, &lay, tan.off[a], tan.na[a], &tan.qeff[a], &ptot, spin);
            mq[a * dim..(a + 1) * dim].copy_from_slice(&v);
        }
        faer::linalg::matmul::matmul(
            MatMut::from_row_major_slice_mut(&mut bmat, n3, dim),
            Accum::Add,
            MatRef::from_row_major_slice(tan.jac.qeff(), nat, n3).transpose(),
            MatRef::from_row_major_slice(&mq, nat, dim),
            1.0,
            Par::Seq,
        );
        drop(mq);
        tan.strip();
        // ACP projector columns
        let ncomp: usize = tan.acp.iter().map(|s| s.np).sum();
        let mut acp_cols = Mat::<f64>::zeros(n, ncomp);
        let mut acp_first = Vec::with_capacity(tan.acp.len());
        let mut k = 0;
        for sh in &tan.acp {
            acp_first.push(k);
            for m in 0..sh.np {
                for mu in 0..n {
                    acp_cols[(mu, k + m)] = sh.col[mu * sh.np + m];
                }
            }
            k += sh.np;
        }
        // exchange kernel derivatives
        let mol = &setup.mol;
        let ex = &setup.exch;
        let nsh = ex.shells.len();
        let mut dgr = vec![0.0; nsh * nsh];
        for i in 0..nsh {
            for j in 0..i {
                let (ia, ja) = (ex.shells[i].0, ex.shells[j].0);
                if ia == ja {
                    continue;
                }
                let r = norm2_sq(sub3(mol.positions[ia], mol.positions[ja])).sqrt();
                let (kap, etag) = mfx_pair(&setup.model, mol, &ex.shells, i, j);
                dgr[i * nsh + j] = mfx_kernel(r, kap, etag).1 / r;
            }
        }
        let mut dkr = vec![0.0; nat * nat];
        for i in 0..nat {
            for j in 0..i {
                let r = norm2_sq(sub3(mol.positions[i], mol.positions[j])).sqrt();
                dkr[i * nat + j] = bo_kernel(&setup.model, mol.numbers[i], mol.numbers[j], r).1 / r;
            }
        }
        let rows = LocalRows { f_disp, f_qeff, acp_cols, acp_first, dgr, dkr };
        Ok(Self { setup, st, tan, pot, vd_tot, base, lay, tsh, tat, bmat, rows })
    }

    /// Kernel potentials and gradient at a displaced geometry (fixed SCC variables).
    fn kernel_at(&self, x: usize, d: f64) -> Result<(Vec<f64>, Vec<f64>)> {
        let mut m = self.setup.mol.clone();
        m.positions[x / 3][x % 3] += d;
        let s = self.setup;
        let k = Kernels::new(&s.model, &m, &s.exch, &s.layout, s.slots, &s.terms)?;
        Ok((k.potentials(self.st, &self.lay), k.gradient(self.st)))
    }

    /// AO-expanded derivatives of the MFX and bond-order kernels for displacement x.
    fn exchange_kernel_tangent(&self, x: usize) -> (Vec<f64>, Vec<f64>) {
        let s = self.setup;
        let mol = &s.mol;
        let ex = &s.exch;
        let n = s.nao;
        let nsh = ex.shells.len();
        let nat = mol.nat();
        let (ax, c) = (x / 3, x % 3);
        let mut gsh = vec![0.0; nsh * nsh];
        for i in 0..nsh {
            for j in 0..nsh {
                let (ia, ja) = (ex.shells[i].0, ex.shells[j].0);
                if ia == ja || (ia != ax && ja != ax) {
                    continue;
                }
                let v = sub3(mol.positions[ia], mol.positions[ja]);
                let r = norm2_sq(v).sqrt();
                let (kap, etag) = mfx_pair(&s.model, mol, &ex.shells, i, j);
                let dg = mfx_kernel(r, kap, etag).1 * v[c] / r;
                gsh[i * nsh + j] = if ia == ax { dg } else { -dg };
            }
        }
        let mut kat = vec![0.0; nat * nat];
        for b in 0..nat {
            if b == ax {
                continue;
            }
            let v = sub3(mol.positions[ax], mol.positions[b]);
            let r = norm2_sq(v).sqrt();
            let dk = bo_kernel(&s.model, mol.numbers[ax], mol.numbers[b], r).1 * v[c] / r;
            kat[ax * nat + b] = dk;
            kat[b * nat + ax] = dk;
        }
        let mut g = vec![0.0; n * n];
        let mut k = vec![0.0; n * n];
        for m in 0..n {
            for l in 0..n {
                g[m * n + l] = gsh[ex.ao_shell[m] * nsh + ex.ao_shell[l]];
                k[m * n + l] = kat[ex.ao_atom[m] * nat + ex.ao_atom[l]];
            }
        }
        (g, k)
    }

    /// Sum_mn X_mn M_mn of the overlap weight X with the overlap derivative of every
    /// displacement (3N), from the local blocks.
    pub fn contract_overlap(&self, x: &[f64]) -> Vec<f64> {
        let tan = &self.tan;
        let n = tan.n;
        let nat = tan.nat;
        let n3 = 3 * nat;
        let rows = |a: usize, r: &[f64]| -> f64 {
            let mut acc = 0.0;
            for ml in 0..tan.na[a] {
                let mu = tan.off[a] + ml;
                for nu in 0..n {
                    acc += r[ml * n + nu] * (x[mu * n + nu] + x[nu * n + mu]);
                }
            }
            acc
        };
        let mut out = vec![0.0; n3];
        for a in 0..nat {
            for c in 0..3 {
                out[3 * a + c] += rows(a, &tan.disp[a][c].s);
            }
            let w = rows(a, &tan.qeff[a].s);
            for y in 0..n3 {
                out[y] += w * tan.jac.qeff()[a * n3 + y];
            }
        }
        out
    }

    /// Add `w` times the local rows `rows` of atom `a` to the rows of `m` (no transpose).
    fn add_rows(&self, m: &mut [f64], a: usize, rows: &[f64], w: f64) {
        if w == 0.0 {
            return;
        }
        let n = self.tan.n;
        let o = self.tan.off[a] * n;
        for (x, r) in m[o..o + rows.len()].iter_mut().zip(rows) {
            *x += w * r;
        }
    }

    /// The quantities of displacement x.
    pub fn displacement(&self, x: usize) -> Result<Displacement> {
        let s = self.setup;
        let n = s.nao;
        let nat = s.mol.nat();
        let n3 = 3 * nat;
        let ns = self.st.p.len();
        let par = Par::Seq;
        let tan = &self.tan;
        let jac = &tan.jac;
        let (ax, cx) = (x / 3, x % 3);
        // S^x and the Fock tangent at the reference potentials, as rows (+ transposes)
        let mut sr = vec![0.0; n * n];
        let mut fr0 = vec![0.0; n * n];
        self.add_rows(&mut sr, ax, &tan.disp[ax][cx].s, 1.0);
        self.add_rows(&mut fr0, ax, &self.rows.f_disp[ax][cx], 1.0);
        for a in 0..nat {
            let (jq, jq0, jc) = (jac.qeff()[a * n3 + x], jac.qeff0()[a * n3 + x], jac.cn()[a * n3 + x]);
            self.add_rows(&mut sr, a, &tan.qeff[a].s, jq);
            self.add_rows(&mut fr0, a, &self.rows.f_qeff[a], jq);
            self.add_rows(&mut fr0, a, &tan.h_qeff0[a], jq0);
            self.add_rows(&mut fr0, a, &tan.h_cn[a], jc);
        }
        // ACP projectors on the displaced atom: w u s^T with u = -d s/dR_A of every other atom's AOs
        for sh in tan.acp.iter().filter(|s| s.atom == ax) {
            let np = sh.np;
            for m in 0..np {
                for mu in 0..n {
                    let u = -sh.dcol[cx][mu * np + m];
                    if u == 0.0 {
                        continue;
                    }
                    for nu in 0..n {
                        fr0[mu * n + nu] += sh.w * u * sh.col[nu * np + m];
                    }
                }
            }
        }
        let sx = sym(n, &sr);
        drop(sr);
        let mut f0 = sym(n, &fr0);
        drop(fr0);
        // kernel terms by 4-point differences at fixed SCC variables
        let (mut c, hk) = fd::try_derivative(fd::Stencil::Central4, STEP, |d| self.kernel_at(x, d))?;
        // exchange: exact overlap tangents (Fock, shell potential, dE/dG, dE/dK, dE/dS)
        let ot = s.exch.overlap_tangent(&s.s, &self.st.p, &self.base, &sx, par);
        for (k, v) in ot.vx.iter().enumerate() {
            c[k] += v;
        }
        let mut hsg = vec![0.0; n3];
        self.add_exchange_kernels(&ot.tsh, &ot.tat, &mut hsg);
        let hss = self.contract_overlap(&ot.psi);
        // SCC variables at fixed density and the kernel response
        let lay = &self.lay;
        let b = self.bmat[x * lay.dim()..(x + 1) * lay.dim()].to_vec();
        let (mut q, mut dip, mut qp) = (vec![0.0; lay.nsh], vec![[0.0; 3]; lay.nat], vec![[0.0; 6]; lay.nat]);
        let mut mag = lay.magnetization.then(|| vec![0.0; lay.nsh]);
        lay.unpack(&b, &mut q, &mut dip, &mut qp, mag.as_deref_mut());
        let qa = s.isotropic.atom_charges(&q);
        let ds = State { p: Vec::new(), q, qa, dip, qp, mag };
        let mut dmu = [0.0; 3];
        dmu[x % 3] += self.st.qa[x / 3];
        for (a, p) in s.mol.positions.iter().enumerate() {
            for k in 0..3 {
                dmu[k] += p[k] * ds.qa[a] + ds.dip[a][k];
            }
        }
        let (kv, kd, kq, kw) = s.scc_kernel(self.st, &ds, None);
        let mut kb = vec![0.0; lay.dim()];
        lay.pack(&kv, &kd, &kq, kw.as_deref(), &mut kb);
        // F^(x): integrals x old potentials + potential changes x integrals + exchange
        let (mut cv, mut cd, mut cq) = (vec![0.0; lay.nsh], vec![[0.0; 3]; lay.nat], vec![[0.0; 6]; lay.nat]);
        lay.unpack(&c, &mut cv, &mut cd, &mut cq, None);
        let cao: Vec<f64> = s.basis.scf.ao2sh.iter().map(|&i| cv[i]).collect();
        add_potentials_rows(n, &s.s, &s.basis.scf.ao2at, &s.dpint, &s.qpint, &cao, &cd, &cq, &mut f0);
        let (dg, dk) = self.exchange_kernel_tangent(x);
        let a1: Vec<f64> = dg.iter().zip(&dk).map(|(g, k)| -0.125 * g + 0.5 * k).collect();
        let a2: Vec<f64> = dg.iter().map(|g| -0.125 * g).collect();
        drop((dg, dk));
        let fr = s.fock_response_fixed_density(self.st, &self.base, &ds, Some((&a1, &a2)), par);
        drop((a1, a2));
        let wspin: Option<Vec<f64>> = self.pot.w.as_ref().map(|w| s.basis.scf.ao2sh.iter().map(|&i| w[i]).collect());
        let mut f = Vec::with_capacity(ns);
        for k in 0..ns {
            let mut fk = f0.clone();
            for e in 0..n * n {
                fk[e] += fr[k][e] + ot.f[k][e];
            }
            if let Some(w) = &wspin {
                add_charge_potential(n, &sx, w, &mut fk, if k == 0 { -1.0 } else { 1.0 });
            }
            f.push(fk);
        }
        Ok(Displacement { f, s: sx, b, kb, c, hk, hss, hsg, dmu })
    }

    /// Add sum_ij t_ij dG_ij/dR (MFX shell kernel, weights `tsh`) and sum_AB t_AB dK_AB/dR
    /// (bond-order kernel, weights `tat`) to the flat gradient `g`.
    fn add_exchange_kernels(&self, tsh: &[f64], tat: &[f64], g: &mut [f64]) {
        let s = self.setup;
        let mol = &s.mol;
        let nat = mol.nat();
        let nsh = s.exch.shells.len();
        let mut pair = |i: usize, j: usize, f: f64| {
            let v = sub3(mol.positions[i], mol.positions[j]);
            for k in 0..3 {
                g[3 * i + k] += f * v[k];
                g[3 * j + k] -= f * v[k];
            }
        };
        for i in 0..nsh {
            for j in 0..i {
                let (ia, ja) = (s.exch.shells[i].0, s.exch.shells[j].0);
                if ia != ja {
                    pair(ia, ja, (tsh[i * nsh + j] + tsh[j * nsh + i]) * self.rows.dgr[i * nsh + j]);
                }
            }
        }
        for i in 0..nat {
            for j in 0..i {
                pair(i, j, (tat[i * nat + j] + tat[j * nat + i]) * self.rows.dkr[i * nat + j]);
            }
        }
    }

    /// Hessian (row-major 3N x 3N) of the exchange kernels at the fixed reference weights
    /// dE/dG, dE/dK: sum over pairs of t f(R) with f the MFX shell kernel or the bond-order
    /// kernel, f''(R) u u^T + f'(R)/R (1 - u u^T) on the diagonal atom blocks and minus that
    /// off the diagonal (closed form).
    pub fn exchange_kernel_hessian(&self) -> Vec<f64> {
        let s = self.setup;
        let mol = &s.mol;
        let ex = &s.exch;
        let nat = mol.nat();
        let n3 = 3 * nat;
        let nsh = ex.shells.len();
        let mut h = vec![0.0; n3 * n3];
        let mut pair = |a: usize, b: usize, w: f64, d1: f64, d2: f64| {
            let v = sub3(mol.positions[a], mol.positions[b]);
            let r = norm2_sq(v).sqrt();
            let u = [v[0] / r, v[1] / r, v[2] / r];
            for i in 0..3 {
                for j in 0..3 {
                    let m = w * (d2 * u[i] * u[j] + d1 / r * (if i == j { 1.0 } else { 0.0 } - u[i] * u[j]));
                    h[(3 * a + i) * n3 + 3 * a + j] += m;
                    h[(3 * b + i) * n3 + 3 * b + j] += m;
                    h[(3 * a + i) * n3 + 3 * b + j] -= m;
                    h[(3 * b + i) * n3 + 3 * a + j] -= m;
                }
            }
        };
        const SQRTPI: f64 = 1.772_453_850_905_516;
        for i in 0..nsh {
            for j in 0..i {
                let (ia, ja) = (ex.shells[i].0, ex.shells[j].0);
                if ia == ja {
                    continue;
                }
                let w = self.tsh[i * nsh + j] + self.tsh[j * nsh + i];
                let r = norm2_sq(sub3(mol.positions[ia], mol.positions[ja])).sqrt();
                let (kap, etag) = mfx_pair(&s.model, mol, &ex.shells, i, j);
                // G = num/den, num = 0.15 + 0.85 erf(0.2 R), den = R + exp(-kappa R)/(eta g)
                let (om, lr) = (crate::exchange::OMEGA, crate::exchange::LR);
                let num = crate::exchange::ALPHA + lr * crate::eeqbc::erf_f64(om * r);
                let dnum = lr * 2.0 * om / SQRTPI * (-(om * r) * (om * r)).exp();
                let d2num = -2.0 * om * om * r * dnum;
                let damp = (-kap * r).exp() / etag;
                let (den, dden, d2den) = (r + damp, 1.0 - kap * damp, kap * kap * damp);
                let d1 = (dnum * den - num * dden) / (den * den);
                let d2 = (d2num * den - num * d2den) / (den * den) - 2.0 * d1 * dden / den;
                pair(ia, ja, w, d1, d2);
            }
        }
        for a in 0..nat {
            for b in 0..a {
                let w = self.tat[a * nat + b] + self.tat[b * nat + a];
                let r = norm2_sq(sub3(mol.positions[a], mol.positions[b])).sqrt();
                let (za, zb) = (mol.numbers[a], mol.numbers[b]);
                let d1 = bo_kernel(&s.model, za, zb, r).1;
                // K = c/2 (1 + erf(x)), x = -k (R - R_ab), k = aatoau/R0: K'' = 2 x k K'
                let rr = (crate::data_v2::FOCK_CRAD[za - 1] * crate::data_v2::FOCK_CRAD[zb - 1]).sqrt();
                let k = s.model.aatoau / crate::exchange::r0(&s.model, za, zb);
                let d2 = 2.0 * (-k * (r - rr)) * k * d1;
                pair(a, b, w, d1, d2);
            }
        }
        h
    }

    /// Column y of sum_s Tr(F^(x)_s dP^y_s) - Tr(S^x dW^y) for every x, without forming F^(x):
    /// the derivative of the gradient functional g_x(P, W) along (dP^y, dW^y), contracted from
    /// the local rows. `kd` holds the solution dP^y with its kernel intermediates
    /// ([`crate::response::Cpks::solve_parts`]) and `dw` the total dW^y (row-major).
    ///
    /// Terms of F^(x) and their contraction with dP^y:
    /// - Fock tangent at the reference potentials (local rows, plus the ACP projectors of the
    ///   displaced atom): rows . (dP + dP^T);
    /// - overlap: the exchange's dE/dS tangent along dP (the transpose of its overlap tangent
    ///   Fock), the spin potential times the magnetization density and -dW^y, contracted with
    ///   the overlap rows;
    /// - exchange kernels: the derivative of dE_x/dG, dE_x/dK along dP times dG/dR, dK/dR;
    /// - potential changes (c^x + K b^x) and the exchange's charge coupling: through the
    ///   Mulliken analysis of dP^y and the exchange shell potential change (returned).
    pub fn response_column(&self, kd: &KernelData, dw: &[f64]) -> ResponseColumn {
        let s = self.setup;
        let n = s.nao;
        let nat = s.mol.nat();
        let n3 = 3 * nat;
        let tan = &self.tan;
        let jac = &tan.jac;
        let dp = &kd.ds.p;
        let xs = sym(n, &s.total_density(dp));
        let mut y = s.exch.overlap_weight_tangent(&self.st.p, &self.base, dp, &kd.prods, Par::Seq);
        for (a, b) in y.iter_mut().zip(dw) {
            *a -= b;
        }
        if let Some(w) = &self.pot.w {
            for i in 0..n {
                for j in 0..n {
                    let pd = dp[0][i * n + j] - dp[1][i * n + j];
                    y[i * n + j] += 0.5 * (w[s.basis.scf.ao2sh[i]] + w[s.basis.scf.ao2sh[j]]) * pd;
                }
            }
        }
        let ys = sym(n, &y);
        drop(y);
        let mut h = vec![0.0; n3];
        let mut wq = vec![0.0; nat];
        let mut wq0 = vec![0.0; nat];
        let mut wcn = vec![0.0; nat];
        for a in 0..nat {
            let (o, e) = (tan.off[a] * n, (tan.off[a] + tan.na[a]) * n);
            let (xa, ya) = (&xs[o..e], &ys[o..e]);
            for c in 0..3 {
                h[3 * a + c] += dot(&self.rows.f_disp[a][c], xa) + dot(&tan.disp[a][c].s, ya);
            }
            wq[a] = dot(&self.rows.f_qeff[a], xa) + dot(&tan.qeff[a].s, ya);
            wq0[a] = dot(&tan.h_qeff0[a], xa);
            wcn[a] = dot(&tan.h_cn[a], xa);
        }
        for a in 0..nat {
            let (jq, jq0, jc) = (
                &jac.qeff()[a * n3..(a + 1) * n3],
                &jac.qeff0()[a * n3..(a + 1) * n3],
                &jac.cn()[a * n3..(a + 1) * n3],
            );
            for x in 0..n3 {
                h[x] += jq[x] * wq[a] + jq0[x] * wq0[a] + jc[x] * wcn[a];
            }
        }
        // ACP projectors of the displaced atom: -w (dS/dR_A)^T (dP + dP^T) s
        let ncomp = self.rows.acp_cols.ncols();
        if ncomp > 0 {
            let mut xc = Mat::<f64>::zeros(n, ncomp);
            faer::linalg::matmul::matmul(
                xc.as_mut(),
                Accum::Replace,
                MatRef::from_row_major_slice(&xs, n, n),
                self.rows.acp_cols.as_ref(),
                1.0,
                Par::Seq,
            );
            for (sh, &first) in tan.acp.iter().zip(&self.rows.acp_first) {
                let np = sh.np;
                for m in 0..np {
                    let col = xc.col_as_slice(first + m);
                    for c in 0..3 {
                        let mut acc = 0.0;
                        for mu in 0..n {
                            acc += sh.dcol[c][mu * np + m] * col[mu];
                        }
                        h[3 * sh.atom + c] -= sh.w * acc;
                    }
                }
            }
        }
        // exchange kernels
        let (tsh, tat) = s.exch.kernel_weight_tangent(&self.st.p, &self.base, dp, &kd.prods);
        self.add_exchange_kernels(&tsh, &tat, &mut h);
        let mut sp = vec![0.0; self.lay.dim()];
        self.lay.pack(&kd.ds.q, &kd.ds.dip, &kd.ds.qp, kd.ds.mag.as_deref(), &mut sp);
        ResponseColumn { h, s: sp, vx: kd.vx.clone() }
    }
}
