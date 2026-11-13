//! Coupled-perturbed SCF response of g-xTB 2.0.1 for the full density matrix.
//!
//! The 2.0.1 Fock matrix is the exact derivative of the energy, F^s = dE/dP^s (closed shell:
//! one channel, the total density), so the response of the converged SCF to a perturbation
//! with first-order Fock matrix F^(y) (at fixed AO density) and overlap S^y solves
//!
//! ```text
//! dF = F^(y) + K[dP],    dP = dP^S + chi0[dF]
//! ```
//!
//! with the kernel K(dP) = d^2E/dP dP . dP ([`Setup::fock_response`], exact: the linearization of
//! every Fock contribution), the orthonormality part dP^S = -C (S~ o D2) C^T and the
//! non-interacting response chi0(X) = C (C^T X C o D1) C^T (S~ = C^T S^y C; D1, D2, D3 the divided
//! differences of f, f e and f e^2 between orbitals, per spin channel). At 0 K the occupations
//! are fixed; at a finite electronic temperature D1, D2, D3 carry the derivatives n' = dn/de on
//! the diagonal and between degenerate orbitals, and the Fermi level of every channel shifts to
//! keep its electron count: dn_p = n'_p (de_p - dmu), dmu = sum_p n'_p de_p / sum_p n'_p, so
//! chi0 loses n' n'^T / sum n' on the diagonal (dP^S and dW gain the matching terms). In the
//! symmetric system below the diagonal pairs then carry d^1/2 = sqrt(-n'/2) and the projector
//! 1 - v v^T, v = sqrt(n' / sum n'). The kernel cannot be reduced to the charges and multipoles as for GFN2:
//! the exchange (Mulliken-approximated range-separated exchange, the bond-order correction
//! quadratic in P, the charge-dependent onsite exchange) depends on the whole density matrix.
//!
//! In the space V of orbital pairs p < q with D1_pq < 0 (occupied-virtual, and pairs with
//! different fractional occupations), x_v = c_p^T dF c_q satisfies (1 + K~ d) x = b with
//! d = -D1, K~_vw = c_p^T K[c_r c_s^T + c_s c_r^T] c_q (symmetric) and
//! b = C^T (F^(y) + K[dP^S]) C. With z = d^1/2 x this is the symmetric positive definite system
//! (1 + d^1/2 K~ d^1/2) z = d^1/2 b, solved by conjugate gradients ([`crate::krylov::cg`]);
//! then dP = dP^S - C (d^1/2 z) C^T (symmetrized), dF = F^(y) + K(dP) and
//! dW = C (C^T dF C o D2 - S~ o D3) C^T.
//!
//! The divided differences and the AO/MO transforms are the shared ones of `axtb::response`
//! (`OccupationResponse` with one occupation vector per channel, levels degenerate when their
//! occupations are equal or within 1e-10 Eh; `MoSpace`, (C_o^T X) C). The 2.0.1 forms stay here:
//! the Fermi-level shift normalized over the occupied rows, the overlap part dP^S and the
//! diagonal correction of dW (n'_p e_p) dmu.

use crate::dense::{dot, from_mat_sym, to_mat, to_mat_sym};
use crate::exchange::ResponseBase;
use crate::fock::{add_charge_potential, add_potentials};
use crate::krylov::{self, CgError};
use crate::scf::{ScfResult, Setup, State};
use axtb::response::{occupied_count, MoSpace, OccupationResponse};
use axtb::{Error, Result};
use axtb_base::fd;
use faer::{Mat, Par};

/// Orbitals, occupations and divided differences of one spin channel.
#[derive(Clone, Debug)]
pub struct Channel {
    pub c: Mat<f64>,
    /// Levels, occupations, their derivatives n' = dn/de and the divided differences D1, D2,
    /// D3 for the rows [0, n_o), the orbitals with occupation above 1e-14 (levels degenerate when
    /// their occupations are equal or they lie within 1e-10 Eh).
    pub occ: OccupationResponse,
    /// sqrt(-D1) on the response pairs p < q with D1 < 0, sqrt(-D1/2) on the diagonal (finite
    /// temperature: orbital p's occupation responds to its level), zero elsewhere (n_o x n).
    pub dh: Mat<f64>,
    /// Finite temperature: Z = sum_{p < n_o} n'_p and the unit vector v_p = sqrt(n'_p / Z) of
    /// the Fermi-level shift, which projects the diagonal response pairs (`None` at 0 K).
    pub fermi: Option<(f64, Vec<f64>)>,
    /// First column of a response pair: dh vanishes in the columns before it (the occupied
    /// ones for integer occupations at 0 K; 0 at a finite temperature, where the diagonal
    /// pairs respond).
    pub v0: usize,
}

impl Channel {
    /// Orbitals `coef`, levels `eps` and occupations `occ` (closed shell: 0..2) of one channel;
    /// at `kt` > 0 the occupations are Fermi functions of the levels (n' = -n (1 - n/n_max)/kT).
    pub fn new(n: usize, coef: &[f64], eps: &[f64], occ: &[f64], kt: f64, nmax: f64) -> Self {
        let c = to_mat(n, coef);
        let n_o = occupied_count(&[occ], 1e-14);
        let dn: Vec<f64> =
            if kt > 0.0 { occ.iter().map(|&o| -o * (1.0 - o / nmax) / kt).collect() } else { vec![0.0; n] };
        // degenerate pairs and the diagonal: the derivative (0 at 0 K)
        let occ =
            OccupationResponse::new(eps, vec![occ.to_vec()], vec![dn], n_o, |de, fp, fq| fp == fq || de.abs() < 1e-10);
        let mut dh = Mat::<f64>::zeros(n_o, n);
        let d1 = occ.d1();
        for q in 0..n {
            for p in 0..n_o {
                let a = d1[(p, q)];
                if q > p && a < 0.0 {
                    dh[(p, q)] = (-a).sqrt();
                } else if q == p && a < 0.0 {
                    dh[(p, q)] = (-0.5 * a).sqrt();
                }
            }
        }
        let dn = &occ.df()[0][..n_o];
        let z: f64 = dn.iter().sum();
        let fermi = (z < 0.0).then(|| (z, dn.iter().map(|&d| (d / z).sqrt()).collect()));
        let v0 = (0..n).find(|&q| (0..n_o).any(|p| dh[(p, q)] != 0.0)).unwrap_or(n);
        Self { c, occ, dh, fermi, v0 }
    }

    /// The orbitals and the rows [0, n_o) of the response.
    pub fn mo(&self) -> MoSpace<'_> {
        MoSpace::new(self.c.as_ref(), self.occ.n_o())
    }

    /// Project the diagonal response pairs of `z` (rows [0, n_o)) onto the complement of the
    /// Fermi-level shift: z_pp -= v_p sum_r v_r z_rr.
    fn project(&self, z: &mut Mat<f64>) {
        if let Some((_, v)) = &self.fermi {
            let n_o = self.occ.n_o();
            let s: f64 = (0..n_o).map(|p| v[p] * z[(p, p)]).sum();
            for p in 0..n_o {
                z[(p, p)] -= v[p] * s;
            }
        }
    }

    /// The Fermi-level shift of the level changes `x(p)` of the rows [0, n_o): the core's
    /// [`OccupationResponse::fermi_shift`], sum_{p < n_o} n'_p x(p) / sum_p n'_p (0 at 0 K).
    fn fermi_shift(&self, x: impl Fn(usize) -> f64) -> f64 {
        self.occ.fermi_shift(0, x)
    }
}

/// Response of the converged SCF to one perturbation (AO matrices per spin channel; closed
/// shell: the total density).
#[derive(Clone, Debug)]
pub struct Perturbed {
    pub dp: Vec<Mat<f64>>,
    pub dw: Vec<Mat<f64>>,
    /// Total first-order Fock matrices F^(y) + K(dP).
    pub df: Vec<Mat<f64>>,
    pub iterations: usize,
    /// With [`Cpks::keep_solution`]: the solution z and the final residual of the symmetric
    /// system on the response columns [v0, n) of every channel, in single precision (the
    /// analytic Hessian's symmetric form, [`AnalyticOptions::symmetric`](crate::hessian::AnalyticOptions)).
    pub zr: Option<(Vec<f32>, Vec<f32>)>,
}

impl Setup {
    /// Mulliken analysis of a density change `dp` (per channel): shell charge changes
    /// -pop(dP), atomic charge, dipole and quadrupole changes, magnetization changes.
    pub fn delta_state(&self, dp: &[Vec<f64>]) -> State {
        let n = self.nao;
        let nat = self.mol.nat();
        let ptot = self.total_density(dp);
        // rows of the (symmetric) overlap, dipole and quadrupole integrals
        let pops = |p: &[f64]| -> Vec<f64> {
            let mut pop = vec![0.0; self.shells.len()];
            for mu in 0..n {
                pop[self.basis.scf.ao2sh[mu]] += dot(&p[mu * n..(mu + 1) * n], &self.s[mu * n..(mu + 1) * n]);
            }
            pop
        };
        let pop = pops(&ptot);
        let q: Vec<f64> = pop.iter().map(|x| -x).collect();
        let qa = self.isotropic.atom_charges(&q);
        let mut dip = vec![[0.0; 3]; nat];
        let mut qp = vec![[0.0; 6]; nat];
        for mu in 0..n {
            let a = self.basis.scf.ao2at[mu];
            let row = &ptot[mu * n..(mu + 1) * n];
            for x in 0..3 {
                dip[a][x] -= dot(row, &self.dpint[x][mu * n..(mu + 1) * n]);
            }
            for c in 0..6 {
                qp[a][c] -= dot(row, &self.qpint[c][mu * n..(mu + 1) * n]);
            }
        }
        let mag = if dp.len() == 2 {
            let (pa, pb) = (pops(&dp[0]), pops(&dp[1]));
            Some(pa.iter().zip(&pb).map(|(a, b)| a - b).collect())
        } else {
            None
        };
        State { p: dp.to_vec(), q, qa, dip, qp, mag }
    }

    /// The SCC kernel K = d^2E/ds ds (without the exchange, whose charge dependence is linear)
    /// applied to SCC-variable changes `ds` at the state `st`: changes of the shell potential
    /// (atomic parts added to every shell of the atom, and the exchange's `vx` when given), the
    /// dipole and quadrupole potentials and the spin potential.
    #[allow(clippy::type_complexity)]
    pub fn scc_kernel(
        &self,
        st: &State,
        ds: &State,
        vx: Option<&[f64]>,
    ) -> (Vec<f64>, Vec<[f64; 3]>, Vec<[f64; 6]>, Option<Vec<f64>>) {
        use axtb::terms::MomentTerm;
        let (tg, s, d, terms) = (self.term_geometry(), st.moments(), ds.moments(), &self.terms);
        use crate::terms::buffer;
        let (nsh, nat) = (self.shells.len(), self.mol.nat());
        let (mut piso, mut paes) = (buffer(&terms.iso, nsh, nat), buffer(&terms.aes, nsh, nat));
        let (mut prep, mut pdisp) = (buffer(&terms.rep, nsh, nat), buffer(&terms.disp, nsh, nat));
        let mut pspin = buffer(&terms.spin, nsh, nat);
        terms.iso.kernel_apply(&tg, &self.isotropic, &s, &d, &mut piso);
        terms.aes.kernel_apply(&tg, &self.aes, &s, &d, &mut paes);
        terms.rep.kernel_apply(&tg, &self.rep, &s, &d, &mut prep);
        terms.disp.kernel_apply(&tg, &self.disp, &s, &d, &mut pdisp);
        let mut dv = piso.shell;
        for (ish, &(iat, _, _)) in self.shells.iter().enumerate() {
            let x = vx.map_or(0.0, |v| v[ish]);
            dv[ish] += x + paes.atom[iat] + prep.atom[iat] + pdisp.atom[iat];
        }
        let w = ds.mag.is_some().then(|| {
            terms.spin.kernel_apply(&tg, &(), &s, &d, &mut pspin);
            pspin.magnetization
        });
        let (vd, vq) = (paes.dipole, paes.quadrupole);
        (dv, vd, vq, w)
    }

    /// The exchange data of a converged state reused by [`Setup::fock_response`].
    pub fn response_base(&self, st: &State, par: Par) -> ResponseBase {
        self.exch.response_base(&self.s, &st.p, &st.q, par)
    }

    /// The kernel: change of the Fock matrices for a density change `ds` (from
    /// [`Setup::delta_state`]) at the state `st`, i.e. d^2E/dP dP . dP at fixed geometry.
    pub fn fock_response(&self, st: &State, base: &ResponseBase, ds: &State, par: Par) -> Vec<Vec<f64>> {
        self.fock_response_parts(st, base, ds, par).0
    }

    /// [`Setup::fock_response`] that also returns the exchange shell potential change and the
    /// exchange's products [dP S, S dP S] of every channel ([`Exchange::response`]).
    ///
    /// [`Exchange::response`]: crate::exchange::Exchange::response
    #[allow(clippy::type_complexity)]
    pub fn fock_response_parts(
        &self,
        st: &State,
        base: &ResponseBase,
        ds: &State,
        par: Par,
    ) -> (Vec<Vec<f64>>, Vec<f64>, Vec<[Vec<f64>; 2]>) {
        let (fx, vx, prods) = self.exch.response(&self.s, &st.p, base, &ds.p, &ds.q, par);
        let kernel = self.scc_kernel(st, ds, Some(&vx));
        (self.kernel_fock(kernel, fx), vx, prods)
    }

    /// [`Setup::fock_response`] for SCC-variable changes `ds` at fixed density (the density
    /// change of `ds` is ignored), with extra exchange kernel changes added to A1 and A2.
    pub fn fock_response_fixed_density(
        &self,
        st: &State,
        base: &ResponseBase,
        ds: &State,
        extra: Option<(&[f64], &[f64])>,
        par: Par,
    ) -> Vec<Vec<f64>> {
        let kernel = self.scc_kernel(st, ds, None);
        let fx = self.exch.charge_response(&self.s, &st.p, base, &ds.q, extra, par);
        self.kernel_fock(kernel, fx)
    }

    /// The Fock matrices of the potentials of [`Setup::scc_kernel`] added to the exchange's
    /// Fock matrices `fx` (one per channel).
    #[allow(clippy::type_complexity)]
    fn kernel_fock(
        &self,
        (dv, vd, vq, w): (Vec<f64>, Vec<[f64; 3]>, Vec<[f64; 6]>, Option<Vec<f64>>),
        fx: Vec<Vec<f64>>,
    ) -> Vec<Vec<f64>> {
        let n = self.nao;
        let mut f0 = vec![0.0; n * n];
        let vao: Vec<f64> = self.basis.scf.ao2sh.iter().map(|&ish| dv[ish]).collect();
        add_potentials(n, &self.s, &self.basis.scf.ao2at, &self.dpint, &self.qpint, &vao, &vd, &vq, &mut f0, false);
        let wao: Option<Vec<f64>> = w.map(|w| self.basis.scf.ao2sh.iter().map(|&ish| w[ish]).collect());
        fx.into_iter()
            .enumerate()
            .map(|(k, mut f)| {
                for x in 0..n * n {
                    f[x] += f0[x];
                }
                if let Some(w) = &wao {
                    add_charge_potential(n, &self.s, w, &mut f, if k == 0 { -1.0 } else { 1.0 });
                }
                f
            })
            .collect()
    }

    /// Fock matrices of the AO spin densities `p` at this geometry.
    pub fn fock_of_density(&self, p: &[Vec<f64>]) -> Vec<Vec<f64>> {
        self.evaluate(&self.state(p), true).1
    }
}

/// Settings of the coupled-perturbed solver.
#[derive(Clone, Copy, Debug)]
pub struct CpksOptions {
    /// Convergence: |residual| <= tol |right-hand side| (conjugate gradients).
    pub tol: f64,
    pub max_iter: usize,
}

impl Default for CpksOptions {
    fn default() -> Self {
        Self { tol: 1e-10, max_iter: 200 }
    }
}

/// Linear response of one converged SCF.
pub struct Cpks<'a> {
    pub setup: &'a Setup,
    pub state: &'a State,
    pub base: ResponseBase,
    pub ch: Vec<Channel>,
    pub opts: CpksOptions,
    /// Parallelism of the linear algebra inside one perturbation.
    pub par: Par,
    /// Return the solution and the final residual of every solve ([`Perturbed::zr`]).
    pub keep_solution: bool,
}

impl<'a> Cpks<'a> {
    pub fn new(setup: &'a Setup, r: &'a ScfResult, opts: CpksOptions, par: Par) -> Self {
        let n = setup.nao;
        let nmax = if r.coefficients.len() == 1 { 2.0 } else { 1.0 };
        let ch = (0..r.coefficients.len())
            .map(|s| Channel::new(n, &r.coefficients[s], &r.orbital_energies[s], &r.occupations[s], r.kt, nmax))
            .collect();
        let base = setup.response_base(&r.state, par);
        Self { setup, state: &r.state, base, ch, opts, par, keep_solution: false }
    }

    /// K(dP) for AO density changes (faer matrices per channel).
    pub fn kernel(&self, dp: &[Mat<f64>]) -> Vec<Mat<f64>> {
        let n = self.setup.nao;
        let v: Vec<Vec<f64>> = dp.iter().map(from_mat_sym).collect();
        let ds = self.setup.delta_state(&v);
        self.setup.fock_response(self.state, &self.base, &ds, self.par).iter().map(|f| to_mat_sym(n, f)).collect()
    }

    /// [`Cpks::kernel`] with its intermediates ([`KernelData`]).
    pub fn kernel_parts(&self, dp: &[Mat<f64>]) -> (Vec<Mat<f64>>, KernelData) {
        let n = self.setup.nao;
        let v: Vec<Vec<f64>> = dp.iter().map(from_mat_sym).collect();
        let ds = self.setup.delta_state(&v);
        let (f, vx, prods) = self.setup.fock_response_parts(self.state, &self.base, &ds, self.par);
        (f.iter().map(|f| to_mat_sym(n, f)).collect(), KernelData { ds, vx, prods })
    }

    /// A z = z + d^1/2 o (C^T K[C (d^1/2 o z) C^T] C) on the response pairs.
    fn apply(&self, z: &[Mat<f64>]) -> Vec<Mat<f64>> {
        let dp: Vec<Mat<f64>> = self
            .ch
            .iter()
            .zip(z)
            .map(|(ch, z)| {
                let mut u = z.clone();
                ch.project(&mut u);
                for j in 0..u.ncols() {
                    for (x, d) in u.col_as_slice_mut(j).iter_mut().zip(ch.dh.col_as_slice(j)) {
                        *x *= d;
                    }
                }
                ch.mo().to_ao_from(u.as_ref(), ch.v0, self.par)
            })
            .collect();
        let k = self.kernel(&dp);
        self.ch
            .iter()
            .zip(z)
            .zip(&k)
            .map(|((ch, z), k)| {
                let mut out = ch.mo().to_mo_from(k.as_ref(), ch.v0, self.par);
                for j in 0..out.ncols() {
                    for (o, d) in out.col_as_slice_mut(j).iter_mut().zip(ch.dh.col_as_slice(j)) {
                        *o *= d;
                    }
                }
                ch.project(&mut out);
                for j in 0..out.ncols() {
                    for (o, zv) in out.col_as_slice_mut(j).iter_mut().zip(z.col_as_slice(j)) {
                        *o += zv;
                    }
                }
                out
            })
            .collect()
    }

    /// Response to a perturbation with first-order Fock matrices `fy` (per channel, at fixed
    /// AO density) and overlap derivative `sy` (AO matrices).
    pub fn solve(&self, fy: &[Mat<f64>], sy: &Mat<f64>) -> Result<Perturbed> {
        Ok(self.solve_parts(fy, sy)?.0)
    }

    /// [`Cpks::solve`] that also returns the intermediates of the kernel applied to the
    /// solution dP ([`KernelData`]: its Mulliken analysis and exchange products).
    pub fn solve_parts(&self, fy: &[Mat<f64>], sy: &Mat<f64>) -> Result<(Perturbed, KernelData)> {
        let par = self.par;
        let st: Vec<Mat<f64>> = self.ch.iter().map(|ch| ch.mo().to_mo(sy.as_ref(), par)).collect();
        // orthonormality part dP^S = -C (S~ o D2) C^T
        let dps: Vec<Mat<f64>> = self
            .ch
            .iter()
            .zip(&st)
            .map(|(ch, s)| {
                let (oc, mo) = (&ch.occ, ch.mo());
                let (n_o, eps, d2) = (oc.n_o(), oc.eps(), oc.d2());
                let mut a = Mat::<f64>::zeros(n_o, s.ncols());
                for j in 0..s.ncols() {
                    for i in 0..n_o {
                        a[(i, j)] = -s[(i, j)] * d2[(i, j)];
                    }
                }
                // finite temperature: the Fermi level follows the overlap part of the levels
                if ch.fermi.is_some() {
                    let mu = ch.fermi_shift(|p| eps[p] * s[(p, p)]);
                    let dn = &oc.df()[0];
                    for p in 0..n_o {
                        a[(p, p)] += dn[p] * mu;
                    }
                }
                mo.halve_occupied(&mut a);
                mo.to_ao(a.as_ref(), par)
            })
            .collect();
        let k0 = self.kernel(&dps);
        // right-hand side d^1/2 o b
        let rhs: Vec<Mat<f64>> = self
            .ch
            .iter()
            .zip(fy.iter().zip(&k0))
            .map(|(ch, (f, k))| {
                let mut b = ch.mo().to_mo_from((f + k).as_ref(), ch.v0, par);
                for j in 0..b.ncols() {
                    for i in 0..b.nrows() {
                        b[(i, j)] *= ch.dh[(i, j)];
                    }
                }
                ch.project(&mut b);
                b
            })
            .collect();
        // conjugate gradients
        let name = self.setup.model.version.label();
        let krylov::CgSolution { x: z, residual: res, iterations } =
            krylov::cg(|p| self.apply(p), &rhs, self.opts.tol, self.opts.max_iter).map_err(|e| match e {
                CgError::NotConverged { iterations, residual } => Error::Numerical(format!(
                    "{name} response: no convergence in {iterations} iterations (residual {residual:.1e})"
                )),
                CgError::Indefinite => Error::Numerical(format!(
                    "{name} response: electronic Hessian not positive definite (unstable SCF solution)"
                )),
            })?;
        let zr = self.keep_solution.then(|| {
            let flat = |ms: &[Mat<f64>]| -> Vec<f32> {
                let mut out = Vec::new();
                for (ch, m) in self.ch.iter().zip(ms) {
                    for j in ch.v0..m.ncols() {
                        out.extend(m.col_as_slice(j).iter().map(|&v| v as f32));
                    }
                }
                out
            };
            // a zero right-hand side: z = 0 and no residual
            let r = match &res {
                Some(res) => flat(res),
                None => flat(&z).iter().map(|_| 0.0).collect(),
            };
            (flat(&z), r)
        });
        // dP = dP^S - C (d^1/2 z) C^T
        let dp: Vec<Mat<f64>> = self
            .ch
            .iter()
            .zip(&z)
            .zip(&dps)
            .map(|((ch, z), d0)| {
                let mut u = z.clone();
                ch.project(&mut u);
                for j in 0..u.ncols() {
                    for i in 0..u.nrows() {
                        u[(i, j)] *= -ch.dh[(i, j)];
                    }
                }
                d0 + ch.mo().to_ao_from(u.as_ref(), ch.v0, par)
            })
            .collect();
        let (k, kd) = self.kernel_parts(&dp);
        let df: Vec<Mat<f64>> = fy.iter().zip(&k).map(|(f, k)| f + k).collect();
        // dW: the core's occupation response of F~ and S~ (with the Fermi-level shift at a finite
        // temperature: d(n_p e_p) gains -e_p n'_p dmu)
        let dw: Vec<Mat<f64>> = self
            .ch
            .iter()
            .zip(&df)
            .zip(&st)
            .map(|((ch, f), s)| {
                let mo = ch.mo();
                let ft = mo.to_mo(f.as_ref(), par);
                let r = ch.occ.density_response(ft.as_ref(), Some(s.as_ref()));
                mo.to_ao(r.dw.as_ref(), par)
            })
            .collect();
        Ok((Perturbed { dp, dw, df, iterations, zr }, kd))
    }
}

/// Intermediates of one kernel application K(dP) that the analytic Hessian's contraction
/// reuses: the Mulliken analysis of dP (shell charges, atomic moments, magnetizations), the
/// change of the exchange shell potential d(dE_x/dq) and the exchange's products
/// [dP S, S dP S] of every channel.
pub struct KernelData {
    pub ds: State,
    pub vx: Vec<f64>,
    pub prods: Vec<[Vec<f64>; 2]>,
}

/// First-order Fock matrices (per channel, at the fixed AO spin densities `p`) and overlap
/// derivative for displacement `y` (atom y/3, direction y%3) by central differences of the
/// geometry-dependent setup (no SCF): 4-point stencil with step `h` (bohr).
pub fn rhs_finite_difference(setup: &Setup, p: &[Vec<f64>], y: usize, h: f64) -> Result<(Vec<Mat<f64>>, Mat<f64>)> {
    let n = setup.nao;
    let (fy, sy) = fd::try_derivative(fd::Stencil::Central4, h, |d| {
        let mut m = setup.mol.clone();
        m.positions[y / 3][y % 3] += d;
        let s = Setup::new(&m, setup.model)?;
        Ok::<_, Error>((s.fock_of_density(p), s.s.clone()))
    })?;
    Ok((fy.iter().map(|f| to_mat(n, f)).collect(), to_mat(n, &sy)))
}
