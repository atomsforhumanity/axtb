//! Coupled-perturbed SCC response of GFN2-xTB to nuclear displacements.
//!
//! The SCC variables `s` are the Mulliken shell charges, atomic dipoles and packed
//! traceless atomic quadrupoles (`M = nsh + 9 nat`). Each is `s_t = s0_t - Tr(P O_t)` with
//! a symmetric AO operator `O_t` (overlap, dipole or quadrupole integrals restricted to one
//! shell or atom), and the Fock matrix depends on them only through the potentials
//! `V_t = ∂E/∂s_t`: `F = H0 - Σ_t V_t O_t`.
//!
//! The density and the energy-weighted density respond to a perturbation with first-order
//! Fock matrix `F^x` and overlap `S^x` through the occupations ([`OccupationResponse`]: the
//! divided differences of the Fermi occupations summed over the two spins, whose open shells
//! have different alpha and beta occupations of the same orbitals, the degenerate levels
//! within 1e-5 kT, and the Fermi-level shift of each spin).
//!
//! With the non-interacting response `χ0 = ∂s/∂V`, the kernel `K = ∂²E/∂s∂s`, the explicit
//! potential derivative `c^x = ∂V/∂x|_s` and the response `u^x` of `s` to the skeleton
//! perturbation (fixed potentials) including the explicit moment derivative
//! `b^x = −Tr(P ∂O/∂x)`, the SCC response solves
//!
//! ```text
//! (1 − χ0 K) δs^x = u^x + χ0 c^x
//! ```
//!
//! for all 3N displacements at once. The coordination-number dependence of the core
//! Hamiltonian is handled through N extra perturbations (one per atomic CN), combined
//! with the CN Jacobian, so every perturbation is local to one atom's AO rows/columns.

use super::{within_kt, MoResponse, MoSpace, OccupationResponse};
use crate::basis::Basis;
use crate::hamiltonian::AtomBlocks;
use crate::linalg::Lu;
use crate::molecule::Molecule;
use crate::scf::{mulliken, ScfVariables};
use crate::xtb::{ChargeDependentTerm, ScfState, XtbModel};
use crate::{Error, Result};
use axtb_base::occupations::fermi;
use axtb_base::threads;
use faer::{Accum, Mat, MatRef, Par};

/// Orbitals with occupation above this count as occupied; below `1 - OCC_THR` per spin as
/// not full.
const OCC_THR: f64 = 1e-12;

/// The occupation response of a converged GFN2-xTB SCF: the alpha and beta occupations of
/// the common orbitals, their derivatives from the Fermi levels (none without smearing) and
/// the degenerate levels within 1e-5 kT.
fn occupations(st: &ScfState) -> OccupationResponse {
    let nao = st.emo.len();
    let kt = st.kt;
    let mut df = vec![vec![0.0; nao], vec![0.0; nao]];
    for s in 0..2 {
        if let Some(ef) = st.efermi[s] {
            for p in 0..nao {
                df[s][p] = fermi((st.emo[p] - ef) / kt, kt).1;
            }
        }
    }
    let f = st.occ.to_vec();
    let n_o = super::occupied_count(&f, OCC_THR);
    OccupationResponse::new(&st.emo, f, df, n_o, within_kt(1e-5, kt))
}

/// Linear response machinery for one converged SCF.
pub(crate) struct Response<'a> {
    pub model: &'a XtbModel,
    pub extra: &'a [&'a dyn ChargeDependentTerm],
    pub mol: &'a Molecule,
    pub st: &'a ScfState,
    pub lay: ScfVariables,
    pub occ: OccupationResponse,
    /// Orbitals `[0, n_f)` are full in both spins.
    pub n_f: usize,
    /// First-order perturbation blocks per atom.
    pub blocks: Vec<AtomBlocks>,
    /// First AO of each atom.
    pub ao_off: Vec<usize>,
    /// CN Jacobian (row-major nat x 3 nat).
    pub jac: Vec<f64>,
    /// `S C` and `X_kᵀ C` for the dipole (k < 3) and quadrupole integrals.
    sc: Mat<f64>,
    gx: Vec<Mat<f64>>,
    pub threads: usize,
}

impl<'a> Response<'a> {
    pub fn new(
        model: &'a XtbModel,
        extra: &'a [&'a dyn ChargeDependentTerm],
        mol: &'a Molecule,
        st: &'a ScfState,
        threads: usize,
    ) -> Self {
        let bas = &model.basis;
        let nat = mol.nat();
        let nao = bas.nao;
        let lay = ScfVariables::new(bas.nsh, nat, model.aes2.is_some(), false);
        let occ = occupations(st);
        let n_f = occ.n_full(1.0, OCC_THR);
        let parallel = threads > 1;
        let vao = st.pot.vao(bas);
        let blocks = model.h0.perturbation_blocks(
            mol,
            bas,
            &st.geo.se,
            &st.geo.dsedcn,
            st.geo.cutoff,
            &st.density,
            &vao,
            &st.pot.vdp,
            &st.pot.vqp,
            &lay,
            parallel,
        );
        let ao_off: Vec<usize> = (0..nat).map(|a| bas.iao_sh[bas.ish_at[a]]).collect();
        let jac = model.ncoord.jacobian(mol);
        let par = threads::par(threads);
        let c = st.coeff.as_ref();
        let mut sc = Mat::<f64>::zeros(nao, nao);
        faer::linalg::matmul::matmul(sc.as_mut(), Accum::Replace, st.ints.overlap.as_ref(), c, 1.0, par);
        let mut gx = Vec::new();
        if lay.multipoles {
            for m in st.ints.dip.iter().chain(st.ints.quad.iter()) {
                let mut g = Mat::<f64>::zeros(nao, nao);
                faer::linalg::matmul::matmul(g.as_mut(), Accum::Replace, m.transpose(), c, 1.0, par);
                gx.push(g);
            }
        }
        Self { model, extra, mol, st, lay, occ, n_f, blocks, ao_off, jac, sc, gx, threads }
    }

    fn bas(&self) -> &Basis {
        &self.model.basis
    }

    pub fn par(&self) -> Par {
        threads::par(self.threads)
    }

    /// The orbitals and the rows `[0, n_o)` of the response matrices.
    pub fn mo(&self) -> MoSpace<'_> {
        MoSpace::new(self.st.coeff.as_ref(), self.occ.n_o())
    }

    fn map<T: Send>(&self, n: usize, f: impl Fn(usize) -> T + Sync + Send) -> Vec<T> {
        threads::map(n, self.threads > 1, f)
    }

    /// `Õ_t,pq` for all SCC variables t (added to `out`).
    fn pair_ops(&self, p: usize, q: usize, out: &mut [f64]) {
        let bas = self.bas();
        let c = &self.st.coeff;
        let (cp, cq) = (c.col_as_slice(p), c.col_as_slice(q));
        let (sp, sq) = (self.sc.col_as_slice(p), self.sc.col_as_slice(q));
        for mu in 0..bas.nao {
            out[bas.ao2sh[mu]] += 0.5 * (cp[mu] * sq[mu] + sp[mu] * cq[mu]);
        }
        if self.lay.multipoles {
            for (k, g) in self.gx.iter().enumerate() {
                let (gp, gq) = (g.col_as_slice(p), g.col_as_slice(q));
                for nu in 0..bas.nao {
                    let a = bas.ao2at[nu];
                    let t = if k < 3 { self.lay.dip(a, k) } else { self.lay.quad(a, k - 3) };
                    out[t] += 0.5 * (gp[nu] * cq[nu] + cp[nu] * gq[nu]);
                }
            }
        }
    }

    /// Non-interacting response `χ0 = ∂s/∂V` (M x M, symmetric negative semidefinite).
    pub fn chi0(&self) -> Mat<f64> {
        let m = self.lay.dim();
        let nao = self.bas().nao;
        let (n_o, n_f) = (self.occ.n_o(), self.n_f);
        // pairs p <= q with p occupied and q not full; weight 2 D1 (p < q) or D1 (p = q)
        let mut pairs: Vec<(u32, u32, f64)> = Vec::new();
        let d1s = self.occ.d1();
        for p in 0..n_o {
            for q in p.max(n_f)..nao {
                let d1 = d1s[(p, q)];
                let w = if p == q { d1 } else { 2.0 * d1 };
                if w < 0.0 {
                    pairs.push((p as u32, q as u32, (-w).sqrt()));
                }
            }
        }
        let mut chi = Mat::<f64>::zeros(m, m);
        const NB: usize = 1024;
        let par = self.par();
        let mut buf = vec![0.0; m * NB];
        for chunk in pairs.chunks(NB) {
            let n = chunk.len();
            let cols = &mut buf[..m * n];
            threads::for_slices(cols, m, self.threads > 1, |k, col| {
                let (p, q, w) = chunk[k];
                col.iter_mut().for_each(|x| *x = 0.0);
                self.pair_ops(p as usize, q as usize, col);
                col.iter_mut().for_each(|x| *x *= w);
            });
            let a = MatRef::from_column_major_slice(cols, m, n);
            // symmetric rank-n update: lower triangle only
            use faer::linalg::matmul::triangular::{matmul as tmatmul, BlockStructure};
            tmatmul(
                chi.as_mut(),
                BlockStructure::TriangularLower,
                Accum::Add,
                a,
                BlockStructure::Rectangular,
                a.transpose(),
                BlockStructure::Rectangular,
                -1.0,
                par,
            );
        }
        for j in 0..m {
            for i in 0..j {
                chi[(i, j)] = chi[(j, i)];
            }
        }
        // Fermi-level shifts: χ0 -= a aᵀ / Z per spin
        for s in 0..2 {
            let z = self.occ.z()[s];
            if z == 0.0 {
                continue;
            }
            let mut a = vec![0.0; m];
            let mut tmp = vec![0.0; m];
            let df = &self.occ.df()[s];
            for p in 0..n_o {
                let w = df[p];
                if w == 0.0 {
                    continue;
                }
                tmp.iter_mut().for_each(|x| *x = 0.0);
                self.pair_ops(p, p, &mut tmp);
                for t in 0..m {
                    a[t] += w * tmp[t];
                }
            }
            for j in 0..m {
                for i in 0..m {
                    chi[(i, j)] -= a[i] * a[j] / z;
                }
            }
        }
        chi
    }

    /// Kernel `K = ∂²E/∂s∂s` (M x M).
    pub fn kernel(&self) -> Mat<f64> {
        let st = self.st;
        let bas = self.bas();
        let lay = self.lay;
        let (nsh, nat) = (lay.nsh, lay.nat);
        let m = lay.dim();
        let mut k = Mat::<f64>::zeros(m, m);
        for a in 0..nsh {
            for b in 0..nsh {
                k[(a, b)] = st.geo.gamma[a * nsh + b];
            }
        }
        if let Some(es3) = &self.model.es3 {
            let mut d = vec![0.0; nsh];
            es3.kernel_diagonal(self.mol, &bas.ish_at, &st.qsh, &mut d);
            for a in 0..nsh {
                k[(a, a)] += d[a];
            }
        }
        // atomic-charge kernels of the extra terms
        let mut kq = vec![0.0; nat * nat];
        for (t, c) in self.extra.iter().zip(&st.geo.caches) {
            t.kernel(self.mol, c.as_ref(), &st.qat, &mut kq);
        }
        // multipole kernel in the atomic layout [q | μ | Θ]
        let n10 = 10 * nat;
        let mut k10 = vec![0.0; if lay.multipoles { n10 * n10 } else { 0 }];
        if let (Some(aes), Some(c)) = (&self.model.aes2, &st.geo.mp_cache) {
            aes.kernel(self.mol, c, &mut k10);
            for i in 0..nat {
                for j in 0..nat {
                    k10[i * n10 + j] += kq[i * nat + j];
                }
            }
        }
        // map the atomic layout to the SCC layout
        let row = |r: usize| -> Vec<usize> {
            if r < nat {
                (bas.ish_at[r]..bas.ish_at[r] + bas.nsh_at[r]).collect()
            } else if r < 4 * nat {
                vec![lay.dip((r - nat) / 3, (r - nat) % 3)]
            } else {
                vec![lay.quad((r - 4 * nat) / 6, (r - 4 * nat) % 6)]
            }
        };
        if lay.multipoles {
            let rows: Vec<Vec<usize>> = (0..n10).map(row).collect();
            for i in 0..n10 {
                for j in 0..n10 {
                    let v = k10[i * n10 + j];
                    if v != 0.0 {
                        for &a in &rows[i] {
                            for &b in &rows[j] {
                                k[(a, b)] += v;
                            }
                        }
                    }
                }
            }
        } else {
            for a in 0..nsh {
                for b in 0..nsh {
                    k[(a, b)] += kq[bas.sh2at[a] * nat + bas.sh2at[b]];
                }
            }
        }
        k
    }

    /// Explicit derivatives of the potentials `c = ∂V/∂R|_s` (M x 3N); with `extra_hessian`
    /// also the nuclear Hessian of the charge-dependent extra terms at fixed charges.
    pub fn potential_derivative(&self, mut extra_hessian: Option<&mut Vec<f64>>) -> Mat<f64> {
        let st = self.st;
        let bas = self.bas();
        let lay = self.lay;
        let (nsh, nat) = (lay.nsh, lay.nat);
        let n3 = 3 * nat;
        let m = lay.dim();
        let mut c = Mat::<f64>::zeros(m, n3);
        let mut dsh = vec![0.0; nsh * n3];
        self.model.es2.potential_derivative(self.mol, &bas.ish_at, &bas.nsh_at, &st.qsh, &mut dsh);
        for a in 0..nsh {
            for x in 0..n3 {
                c[(a, x)] = dsh[a * n3 + x];
            }
        }
        // atomic potentials: extra terms, multipole terms
        let n10 = 10 * nat;
        let mut dv = vec![0.0; n10 * n3];
        if let Some(h) = extra_hessian.as_deref_mut() {
            h.clear();
            h.resize(n3 * n3, 0.0);
        }
        for (t, cc) in self.extra.iter().zip(&st.geo.caches) {
            t.second_derivatives(
                self.mol,
                cc.as_ref(),
                &st.qat,
                &mut dv[..nat * n3],
                extra_hessian.as_deref_mut().map(|h| h.as_mut_slice()),
                self.threads > 1,
            );
        }
        if let (Some(aes), Some(cache)) = (&self.model.aes2, &st.geo.mp_cache) {
            let mut dvcn = vec![0.0; n10 * nat];
            aes.potential_derivative(self.mol, cache, &st.qat, &st.dpat, &st.qpat, &mut dv, &mut dvcn);
            // chain through the coordination numbers
            for r in 0..n10 {
                for a in 0..nat {
                    let w = dvcn[r * nat + a];
                    if w != 0.0 {
                        let jrow = &self.jac[a * n3..(a + 1) * n3];
                        let out = &mut dv[r * n3..(r + 1) * n3];
                        for x in 0..n3 {
                            out[x] += w * jrow[x];
                        }
                    }
                }
            }
        }
        for a in 0..nsh {
            let at = bas.sh2at[a];
            for x in 0..n3 {
                c[(a, x)] += dv[at * n3 + x];
            }
        }
        if lay.multipoles {
            for i in 0..nat {
                for k in 0..3 {
                    for x in 0..n3 {
                        c[(lay.dip(i, k), x)] = dv[(nat + 3 * i + k) * n3 + x];
                    }
                }
                for k in 0..6 {
                    for x in 0..n3 {
                        c[(lay.quad(i, k), x)] = dv[(4 * nat + 6 * i + k) * n3 + x];
                    }
                }
            }
        }
        c
    }

    /// Number of local perturbations: 3N displacements, then N coordination numbers.
    pub fn nxi(&self) -> usize {
        4 * self.lay.nat
    }

    /// Atom and AO column block of the Fock (and overlap) perturbation `xi`.
    fn local(&self, xi: usize) -> (usize, &Mat<f64>, Option<&Mat<f64>>) {
        let n3 = 3 * self.lay.nat;
        if xi < n3 {
            let (a, k) = (xi / 3, xi % 3);
            (a, &self.blocks[a].fock[k], Some(&self.blocks[a].overlap[k]))
        } else {
            let a = xi - n3;
            (a, &self.blocks[a].cn, None)
        }
    }

    /// MO response (rows `[0, n_o)`) to a perturbation with first-order Fock matrix `f`
    /// and overlap `s` in the MO basis (first `n_o` rows).
    pub fn mo_response(&self, f: MatRef<'_, f64>, s: Option<MatRef<'_, f64>>) -> MoResponse {
        self.occ.density_response(f, s)
    }

    /// AO matrix of an MO response (first `n_o` rows, oo block halved).
    pub fn to_ao(&self, a: &Mat<f64>, par: Par) -> Mat<f64> {
        self.mo().to_ao(a.as_ref(), par)
    }

    /// `-Tr(X O_t)` for all SCC variables t.
    pub fn moments_of(&self, x: &Mat<f64>, out: &mut [f64]) {
        let lay = self.lay;
        let bas = self.bas();
        let mut qsh = vec![0.0; lay.nsh];
        let mut dpat = vec![[0.0; 3]; lay.nat];
        let mut qpat = vec![[0.0; 6]; lay.nat];
        mulliken::moments(
            bas,
            &self.st.ints,
            x,
            &vec![0.0; lay.nsh],
            &mut qsh,
            &mut dpat,
            &mut qpat,
            lay.multipoles,
            false,
        );
        lay.pack(&qsh, &dpat, &qpat, None, out);
    }

    /// Skeleton response of local perturbation `xi` (fixed potentials): the response of
    /// the SCC variables `u` (including the explicit moment derivative) and, with
    /// `contract`, the column `Tr(δP A^x) − Tr(δW S^x)` over all local perturbations x.
    pub fn skeleton(&self, xi: usize, contract: bool) -> (Vec<f64>, Vec<f64>) {
        let par = Par::Seq;
        let n3 = 3 * self.lay.nat;
        let mo = self.mo();
        let (a, lf, ls) = self.local(xi);
        let off = self.ao_off[a];
        let f = mo.local_to_mo(lf.as_ref(), off, par);
        let s = ls.map(|ls| mo.local_to_mo(ls.as_ref(), off, par));
        let r = self.mo_response(f.as_ref(), s.as_ref().map(|s| s.as_ref()));
        let dp = self.to_ao(&r.dp, par);
        let mut u = vec![0.0; self.lay.dim()];
        self.moments_of(&dp, &mut u);
        if xi < n3 {
            let b = &self.blocks[a].b[xi % 3];
            for t in 0..u.len() {
                u[t] += b[t];
            }
        }
        let mut col = Vec::new();
        if contract {
            let dw = self.to_ao(&r.dw, par);
            col = vec![0.0; self.nxi()];
            for (x, v) in col.iter_mut().enumerate() {
                let (b, lf, ls) = self.local(x);
                let off = self.ao_off[b];
                let mut acc = 0.0;
                for j in 0..lf.ncols() {
                    acc += crate::scf::dot(dp.col_as_slice(off + j), lf.col_as_slice(j));
                    if let Some(ls) = ls {
                        acc -= crate::scf::dot(dw.col_as_slice(off + j), ls.col_as_slice(j));
                    }
                }
                *v = 2.0 * acc;
            }
        }
        (u, col)
    }

    /// Combine local quantities (columns over the 4N local perturbations) into the 3N
    /// displacements: `v_x = v_dir,x + Σ_C v_CN,C J_Cx`.
    pub fn combine_columns(&self, local: &Mat<f64>) -> Mat<f64> {
        let nat = self.lay.nat;
        let n3 = 3 * nat;
        let m = local.nrows();
        let mut out = Mat::<f64>::zeros(m, n3);
        for x in 0..n3 {
            for t in 0..m {
                out[(t, x)] = local[(t, x)];
            }
        }
        for cc in 0..nat {
            let jrow = &self.jac[cc * n3..(cc + 1) * n3];
            let col = local.col_as_slice(n3 + cc);
            for x in 0..n3 {
                let w = jrow[x];
                if w != 0.0 {
                    let o = out.col_as_slice_mut(x);
                    for t in 0..m {
                        o[t] += w * col[t];
                    }
                }
            }
        }
        out
    }

    /// Dipole derivatives ∂μ/∂R_x (one row per displacement x, e) from the SCC response
    /// `ds = δs/δR` (M x 3N): μ = Σ_A q_A R_A + μ_A with the Mulliken charges q_A (sums of the
    /// shell charges) and the atomic dipoles μ_A, both SCC variables. `None` without atomic
    /// dipoles among the SCC variables.
    pub fn dipole_derivatives(&self, ds: &Mat<f64>) -> Option<Vec<[f64; 3]>> {
        let lay = self.lay;
        if !lay.multipoles {
            return None;
        }
        let bas = self.bas();
        let nat = lay.nat;
        let n3 = 3 * nat;
        let pos = &self.mol.positions;
        Some(
            (0..n3)
                .map(|x| {
                    let mut d = [0.0; 3];
                    d[x % 3] += self.st.qat[x / 3];
                    for a in 0..nat {
                        let dq: f64 = (bas.ish_at[a]..bas.ish_at[a] + bas.nsh_at[a]).map(|sh| ds[(sh, x)]).sum();
                        for k in 0..3 {
                            d[k] += pos[a][k] * dq + ds[(lay.dip(a, k), x)];
                        }
                    }
                    d
                })
                .collect(),
        )
    }

    /// Solve `(1 − χ0 K) X = rhs`.
    pub fn solve(&self, chi0: &Mat<f64>, k: &Mat<f64>, rhs: &Mat<f64>) -> Result<Mat<f64>> {
        let m = self.lay.dim();
        let par = self.par();
        let mut a = Mat::<f64>::zeros(m, m);
        faer::linalg::matmul::matmul(a.as_mut(), Accum::Replace, chi0.as_ref(), k.as_ref(), -1.0, par);
        for i in 0..m {
            a[(i, i)] += 1.0;
        }
        let mut x = rhs.clone();
        let lu = Lu::try_new(a, par).ok_or_else(|| Error::Numerical("singular response matrix".into()))?;
        lu.solve(x.as_mut());
        Ok(x)
    }
}

/// SCC response `δs/δR` (M x 3N) and the intermediates of its equation.
pub(crate) struct SccResponse {
    pub kernel: Mat<f64>,
    /// Explicit potential derivatives (M x 3N).
    pub c: Mat<f64>,
    /// Skeleton responses of the displacements (M x 3N).
    pub u: Mat<f64>,
    /// Skeleton Hessian over the local perturbations (4N x 4N), when requested.
    pub hskel: Option<Mat<f64>>,
    /// δs/δR.
    pub ds: Mat<f64>,
    /// Nuclear Hessian of the charge-dependent extra terms at fixed charges (row-major
    /// 3N x 3N), when the skeleton Hessian was requested.
    pub extra_hessian: Vec<f64>,
    pub timings: Vec<(&'static str, f64)>,
}

impl Response<'_> {
    /// Solve the coupled-perturbed SCC equations for all displacements.
    pub fn scc_response(&self, skeleton_hessian: bool) -> Result<SccResponse> {
        let mut timings = Vec::new();
        let mut t0 = std::time::Instant::now();
        let mut lap = |name: &'static str, t0: &mut std::time::Instant| {
            timings.push((name, t0.elapsed().as_secs_f64()));
            *t0 = std::time::Instant::now();
        };
        let chi0 = self.chi0();
        lap("chi0", &mut t0);
        let kernel = self.kernel();
        let mut extra_hessian = Vec::new();
        let c = self.potential_derivative(if skeleton_hessian { Some(&mut extra_hessian) } else { None });
        lap("kernel+potential derivatives", &mut t0);
        let nxi = self.nxi();
        let m = self.lay.dim();
        let cols = self.map(nxi, |xi| self.skeleton(xi, skeleton_hessian));
        let mut u_local = Mat::<f64>::zeros(m, nxi);
        let mut hskel = if skeleton_hessian { Some(Mat::<f64>::zeros(nxi, nxi)) } else { None };
        for (xi, (u, h)) in cols.into_iter().enumerate() {
            u_local.col_as_slice_mut(xi).copy_from_slice(&u);
            if let Some(hs) = hskel.as_mut() {
                hs.col_as_slice_mut(xi).copy_from_slice(&h);
            }
        }
        lap("skeleton responses", &mut t0);
        let u = self.combine_columns(&u_local);
        let mut rhs = u.clone();
        faer::linalg::matmul::matmul(rhs.as_mut(), Accum::Add, chi0.as_ref(), c.as_ref(), 1.0, self.par());
        let ds = self.solve(&chi0, &kernel, &rhs)?;
        lap("solve", &mut t0);
        Ok(SccResponse { kernel, c, u, hskel, ds, extra_hessian, timings })
    }
}
