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
//! (1 + d^1/2 K~ d^1/2) z = d^1/2 b, solved by conjugate gradients; then
//! dP = dP^S - C (d^1/2 z) C^T (symmetrized), dF = F^(y) + K(dP) and
//! dW = C (C^T dF C o D2 - S~ o D3) C^T.

use super::exchange::ResponseBase;
use super::scf::{ScfResult, Setup, State};
use crate::fock::{add_charge_potential, add_potentials_rows};
use crate::parallel::TILE;
use axtb::{Error, Result};
use axtb_base::fd;
use faer::{Accum, Mat, MatRef, Par};

/// Row-major (or symmetric) n x n slice as a faer matrix.
pub fn to_mat(n: usize, v: &[f64]) -> Mat<f64> {
    MatRef::from_row_major_slice(v, n, n).to_owned()
}

/// faer matrix as a row-major vector.
pub fn from_mat(m: &Mat<f64>) -> Vec<f64> {
    let (r, c) = (m.nrows(), m.ncols());
    let mut out = vec![0.0; r * c];
    for j in 0..c {
        let col = m.col_as_slice(j);
        for i in 0..r {
            out[i * c + j] = col[i];
        }
    }
    out
}

/// Dot product with eight partial sums (vectorizes; the order of a sequential sum does not).
#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    let mut acc = [0.0; 8];
    let ((ca, ra), (cb, rb)) = (a.as_chunks::<8>(), b.as_chunks::<8>());
    for (x, y) in ca.iter().zip(cb) {
        for k in 0..8 {
            acc[k] += x[k] * y[k];
        }
    }
    let mut s = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

/// A symmetric faer matrix as a row-major vector (its columns are its rows: plain copies).
pub fn from_mat_sym(m: &Mat<f64>) -> Vec<f64> {
    let n = m.nrows();
    let mut out = vec![0.0; n * n];
    for j in 0..n {
        out[j * n..(j + 1) * n].copy_from_slice(m.col_as_slice(j));
    }
    out
}

/// A (nearly) symmetric row-major n x n slice as a faer matrix by plain copies (the transpose
/// of [`to_mat`]: equal for a symmetric matrix, round-off apart for a nearly symmetric one).
pub fn to_mat_sym(n: usize, v: &[f64]) -> Mat<f64> {
    let mut m = Mat::<f64>::zeros(n, n);
    for j in 0..n {
        m.col_as_slice_mut(j).copy_from_slice(&v[j * n..(j + 1) * n]);
    }
    m
}

/// X + X^T of a square faer matrix (tiled).
fn sym_mat(x: &Mat<f64>) -> Mat<f64> {
    let n = x.nrows();
    let mut out = Mat::<f64>::zeros(n, n);
    for j0 in (0..n).step_by(TILE) {
        for i0 in (0..n).step_by(TILE) {
            for j in j0..(j0 + TILE).min(n) {
                let (xj, oj) = (x.col_as_slice(j), out.col_as_slice_mut(j));
                for i in i0..(i0 + TILE).min(n) {
                    oj[i] = xj[i] + x.col_as_slice(i)[j];
                }
            }
        }
    }
    out
}

#[inline]
fn mm(out: &mut Mat<f64>, a: MatRef<'_, f64>, b: MatRef<'_, f64>, accum: Accum, par: Par) {
    faer::linalg::matmul::matmul(out.as_mut(), accum, a, b, 1.0, par);
}

/// Orbitals, occupations and divided differences of one spin channel.
#[derive(Clone, Debug)]
pub struct Channel {
    pub c: Mat<f64>,
    pub eps: Vec<f64>,
    pub occ: Vec<f64>,
    /// Orbitals [0, n_o) carry electrons.
    pub n_o: usize,
    /// Divided differences of f, f e and f e^2 for rows [0, n_o) (n_o x n).
    pub d1: Mat<f64>,
    pub d2: Mat<f64>,
    pub d3: Mat<f64>,
    /// sqrt(-D1) on the response pairs p < q with D1 < 0, sqrt(-D1/2) on the diagonal (finite
    /// temperature: orbital p's occupation responds to its level), zero elsewhere (n_o x n).
    pub dh: Mat<f64>,
    /// Finite temperature: dn/de of every orbital and the unit vector v_p = sqrt(n'_p / Z)
    /// (Z = sum_p n'_p) of the Fermi-level shift, which projects the diagonal response pairs
    /// (`None` at 0 K).
    pub fermi: Option<(Vec<f64>, Vec<f64>)>,
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
        let n_o = (0..n).rev().find(|&p| occ[p] > 1e-14).map_or(0, |p| p + 1);
        let dn: Vec<f64> =
            if kt > 0.0 { occ.iter().map(|&o| -o * (1.0 - o / nmax) / kt).collect() } else { vec![0.0; n] };
        let mut d1 = Mat::<f64>::zeros(n_o, n);
        let mut d2 = Mat::<f64>::zeros(n_o, n);
        let mut d3 = Mat::<f64>::zeros(n_o, n);
        let mut dh = Mat::<f64>::zeros(n_o, n);
        for q in 0..n {
            for p in 0..n_o {
                let (ep, eq) = (eps[p], eps[q]);
                let (fp, fq) = (occ[p], occ[q]);
                let de = ep - eq;
                // degenerate pairs and the diagonal: the derivative (0 at 0 K)
                let a = if fp == fq || de.abs() < 1e-10 { 0.5 * (dn[p] + dn[q]) } else { (fp - fq) / de };
                let fsum = fp + fq;
                d1[(p, q)] = a;
                d2[(p, q)] = 0.5 * fsum + 0.5 * (ep + eq) * a;
                d3[(p, q)] = 0.5 * (ep * ep + eq * eq) * a + 0.5 * fsum * (ep + eq);
                if q > p && a < 0.0 {
                    dh[(p, q)] = (-a).sqrt();
                } else if q == p && a < 0.0 {
                    dh[(p, q)] = (-0.5 * a).sqrt();
                }
            }
        }
        let z: f64 = dn[..n_o].iter().sum();
        let fermi = (z < 0.0).then(|| {
            let v = dn[..n_o].iter().map(|&d| (d / z).sqrt()).collect();
            (dn, v)
        });
        let v0 = (0..n).find(|&q| (0..n_o).any(|p| dh[(p, q)] != 0.0)).unwrap_or(n);
        Self { c, eps: eps.to_vec(), occ: occ.to_vec(), n_o, d1, d2, d3, dh, fermi, v0 }
    }

    /// Project the diagonal response pairs of `z` (rows [0, n_o)) onto the complement of the
    /// Fermi-level shift: z_pp -= v_p sum_r v_r z_rr.
    fn project(&self, z: &mut Mat<f64>) {
        if let Some((_, v)) = &self.fermi {
            let s: f64 = (0..self.n_o).map(|p| v[p] * z[(p, p)]).sum();
            for p in 0..self.n_o {
                z[(p, p)] -= v[p] * s;
            }
        }
    }

    /// The Fermi-level shift of a level-weighted diagonal: sum_p n'_p x_pp / sum_p n'_p (0 at
    /// 0 K) for an MO matrix `x` (rows [0, n_o)).
    fn fermi_shift(&self, x: impl Fn(usize) -> f64) -> f64 {
        match &self.fermi {
            Some((dn, _)) => {
                let z: f64 = dn[..self.n_o].iter().sum();
                (0..self.n_o).map(|p| dn[p] * x(p)).sum::<f64>() / z
            }
            None => 0.0,
        }
    }

    /// [`Channel::to_mo`] restricted to the response columns [v0, n) (zero before).
    pub fn to_mo_v(&self, x: &Mat<f64>, par: Par) -> Mat<f64> {
        let n = self.c.nrows();
        let co = self.c.as_ref().subcols(0, self.n_o);
        let mut t = Mat::<f64>::zeros(self.n_o, n);
        mm(&mut t, co.transpose(), x.as_ref(), Accum::Replace, par);
        let mut out = Mat::<f64>::zeros(self.n_o, n);
        faer::linalg::matmul::matmul(
            out.as_mut().subcols_mut(self.v0, n - self.v0),
            Accum::Replace,
            t.as_ref(),
            self.c.as_ref().subcols(self.v0, n - self.v0),
            1.0,
            par,
        );
        out
    }

    /// [`Channel::to_ao`] of rows `a` that vanish in the columns before v0.
    pub fn to_ao_v(&self, a: &Mat<f64>, par: Par) -> Mat<f64> {
        let n = self.c.nrows();
        let nv = n - self.v0;
        let co = self.c.as_ref().subcols(0, self.n_o);
        let mut t = Mat::<f64>::zeros(self.n_o, n);
        faer::linalg::matmul::matmul(
            t.as_mut(),
            Accum::Replace,
            a.as_ref().subcols(self.v0, nv),
            self.c.as_ref().subcols(self.v0, nv).transpose(),
            1.0,
            par,
        );
        let mut x = Mat::<f64>::zeros(n, n);
        mm(&mut x, co, t.as_ref(), Accum::Replace, par);
        sym_mat(&x)
    }

    /// Rows [0, n_o) of C^T X C for a symmetric AO matrix X.
    pub fn to_mo(&self, x: &Mat<f64>, par: Par) -> Mat<f64> {
        let n = self.c.nrows();
        let co = self.c.as_ref().subcols(0, self.n_o);
        let mut t = Mat::<f64>::zeros(self.n_o, n);
        mm(&mut t, co.transpose(), x.as_ref(), Accum::Replace, par);
        let mut out = Mat::<f64>::zeros(self.n_o, n);
        mm(&mut out, t.as_ref(), self.c.as_ref(), Accum::Replace, par);
        out
    }

    /// C_o A C^T + its transpose for rows [0, n_o) `a` of an MO matrix: the AO matrix of the
    /// symmetric MO matrix with these rows when the occupied-occupied block of `a` is halved.
    pub fn to_ao(&self, a: &Mat<f64>, par: Par) -> Mat<f64> {
        let n = self.c.nrows();
        let co = self.c.as_ref().subcols(0, self.n_o);
        let mut t = Mat::<f64>::zeros(self.n_o, n);
        mm(&mut t, a.as_ref(), self.c.transpose(), Accum::Replace, par);
        let mut x = Mat::<f64>::zeros(n, n);
        mm(&mut x, co, t.as_ref(), Accum::Replace, par);
        sym_mat(&x)
    }

    /// Halve the occupied-occupied block of rows [0, n_o).
    fn halve_oo(&self, a: &mut Mat<f64>) {
        for q in 0..self.n_o {
            for p in 0..self.n_o {
                a[(p, q)] *= 0.5;
            }
        }
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
    /// analytic Hessian's symmetric form, [`AnalyticOptions::symmetric`](super::hessian::AnalyticOptions)).
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
                pop[self.ao_shell[mu]] += dot(&p[mu * n..(mu + 1) * n], &self.s[mu * n..(mu + 1) * n]);
            }
            pop
        };
        let pop = pops(&ptot);
        let q: Vec<f64> = pop.iter().map(|x| -x).collect();
        let qa = self.coulomb.atom_charges(&q);
        let mut dip = vec![[0.0; 3]; nat];
        let mut qp = vec![[0.0; 6]; nat];
        for mu in 0..n {
            let a = self.ao_atom[mu];
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
        let mol = &self.mol;
        let mut dv = self.coulomb.isotropic_hessian_vec(&st.q, &ds.q);
        let (vs, vd, vq) = self.coulomb.aes_potentials(mol, &ds.qa, &ds.dip, &ds.qp, false);
        let vrep = self.rep.potential_delta(&ds.qa);
        let vdisp = self.disp.two_body_hessian_vec(&st.qa, &ds.qa);
        for (ish, &(iat, _, _)) in self.shells.iter().enumerate() {
            let x = vx.map_or(0.0, |v| v[ish]);
            dv[ish] += x + vs[iat] + vrep[iat] + vdisp[iat];
        }
        let w = ds.mag.as_ref().map(|m| super::coulomb::spin_potential(mol, &self.shells, m));
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
    /// [`Exchange::response`]: super::exchange::Exchange::response
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
        let vao: Vec<f64> = self.ao_shell.iter().map(|&ish| dv[ish]).collect();
        add_potentials_rows(n, &self.s, &self.ao_atom, &self.dpint, &self.qpint, &vao, &vd, &vq, &mut f0);
        let wao: Option<Vec<f64>> = w.map(|w| self.ao_shell.iter().map(|&ish| w[ish]).collect());
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
                ch.to_ao_v(&u, self.par)
            })
            .collect();
        let k = self.kernel(&dp);
        self.ch
            .iter()
            .zip(z)
            .zip(&k)
            .map(|((ch, z), k)| {
                let mut out = ch.to_mo_v(k, self.par);
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

    fn dot(a: &[Mat<f64>], b: &[Mat<f64>]) -> f64 {
        let mut s = 0.0;
        for (x, y) in a.iter().zip(b) {
            for j in 0..x.ncols() {
                s += dot(x.col_as_slice(j), y.col_as_slice(j));
            }
        }
        s
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
        let st: Vec<Mat<f64>> = self.ch.iter().map(|ch| ch.to_mo(sy, par)).collect();
        // orthonormality part dP^S = -C (S~ o D2) C^T
        let dps: Vec<Mat<f64>> = self
            .ch
            .iter()
            .zip(&st)
            .map(|(ch, s)| {
                let mut a = Mat::<f64>::zeros(ch.n_o, s.ncols());
                for j in 0..s.ncols() {
                    for i in 0..ch.n_o {
                        a[(i, j)] = -s[(i, j)] * ch.d2[(i, j)];
                    }
                }
                // finite temperature: the Fermi level follows the overlap part of the levels
                if let Some((dn, _)) = &ch.fermi {
                    let mu = ch.fermi_shift(|p| ch.eps[p] * s[(p, p)]);
                    for p in 0..ch.n_o {
                        a[(p, p)] += dn[p] * mu;
                    }
                }
                ch.halve_oo(&mut a);
                ch.to_ao(&a, par)
            })
            .collect();
        let k0 = self.kernel(&dps);
        // right-hand side d^1/2 o b
        let rhs: Vec<Mat<f64>> = self
            .ch
            .iter()
            .zip(fy.iter().zip(&k0))
            .map(|(ch, (f, k))| {
                let mut b = ch.to_mo_v(&(f + k), par);
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
        let bnorm = Self::dot(&rhs, &rhs).sqrt();
        let mut z: Vec<Mat<f64>> = rhs.iter().map(|r| Mat::<f64>::zeros(r.nrows(), r.ncols())).collect();
        let mut iterations = 0;
        let mut res: Vec<Mat<f64>> = Vec::new();
        if bnorm > 0.0 {
            let mut r = rhs.clone();
            let mut p = r.clone();
            let mut rr = Self::dot(&r, &r);
            loop {
                if rr.sqrt() <= self.opts.tol * bnorm {
                    break;
                }
                if iterations >= self.opts.max_iter {
                    return Err(Error::Numerical(format!(
                        "g-xTB 2.0.1 response: no convergence in {} iterations (residual {:.1e})",
                        iterations,
                        rr.sqrt() / bnorm
                    )));
                }
                iterations += 1;
                let ap = self.apply(&p);
                let pap = Self::dot(&p, &ap);
                if pap <= 0.0 {
                    return Err(Error::Numerical(
                        "g-xTB 2.0.1 response: electronic Hessian not positive definite (unstable SCF solution)".into(),
                    ));
                }
                let alpha = rr / pap;
                for k in 0..z.len() {
                    for j in 0..z[k].ncols() {
                        for (u, v) in z[k].col_as_slice_mut(j).iter_mut().zip(p[k].col_as_slice(j)) {
                            *u += alpha * v;
                        }
                        for (u, v) in r[k].col_as_slice_mut(j).iter_mut().zip(ap[k].col_as_slice(j)) {
                            *u -= alpha * v;
                        }
                    }
                }
                let rr_new = Self::dot(&r, &r);
                let beta = rr_new / rr;
                rr = rr_new;
                for k in 0..p.len() {
                    for j in 0..p[k].ncols() {
                        for (u, v) in p[k].col_as_slice_mut(j).iter_mut().zip(r[k].col_as_slice(j)) {
                            *u = v + beta * *u;
                        }
                    }
                }
            }
            res = r;
        }
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
            let r = if res.is_empty() { flat(&z).iter().map(|_| 0.0).collect() } else { flat(&res) };
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
                d0 + ch.to_ao_v(&u, par)
            })
            .collect();
        let (k, kd) = self.kernel_parts(&dp);
        let df: Vec<Mat<f64>> = fy.iter().zip(&k).map(|(f, k)| f + k).collect();
        let dw: Vec<Mat<f64>> = self
            .ch
            .iter()
            .zip(&df)
            .zip(&st)
            .map(|((ch, f), s)| {
                let ft = ch.to_mo(f, par);
                let mut a = Mat::<f64>::zeros(ch.n_o, ft.ncols());
                for j in 0..ft.ncols() {
                    for i in 0..ch.n_o {
                        a[(i, j)] = ft[(i, j)] * ch.d2[(i, j)] - s[(i, j)] * ch.d3[(i, j)];
                    }
                }
                // finite temperature: d(n_p e_p) gains -n'_p e_p dmu
                if let Some((dn, _)) = &ch.fermi {
                    let mu = ch.fermi_shift(|p| ft[(p, p)] - ch.eps[p] * s[(p, p)]);
                    for p in 0..ch.n_o {
                        a[(p, p)] -= dn[p] * ch.eps[p] * mu;
                    }
                }
                ch.halve_oo(&mut a);
                ch.to_ao(&a, par)
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
        let s = Setup::new(&m, setup.model.conventions)?;
        Ok::<_, Error>((s.fock_of_density(p), s.s.clone()))
    })?;
    Ok((fy.iter().map(|f| to_mat(n, f)).collect(), to_mat(n, &sy)))
}
