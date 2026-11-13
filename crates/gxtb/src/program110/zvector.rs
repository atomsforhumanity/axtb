//! Analytic gradient of the converged v1.1 SCF energy in the program's conventions
//! ([`Setup::program_gradient`]): the response of the program's Fock matrix (Z-vector method)
//! on the engine.
//!
//! The SCF diagonalizes the program's Fock matrix F_used ([`super::ProgramFock`]), which
//! differs from the exact derivative F = dE/dP by dF = F - F_used = F_AES'' - F_x: F_AES'' is
//! the AES Fock matrix of the dipole-quadrupole and quadrupole-quadrupole energies (which the
//! program leaves out) and F_x (per spin) the extra half of the population part of the
//! onsite exchange the program applies off the diagonal, F_x,kl = 1/2 (y_k + y_l) S_kl
//! (k != l), linear in the density ([`Setup::ofx_extra`]). The gradient then needs the
//! response of the density.
//!
//! With spin densities P_s = n_s C_o C_o^T (n_s = 2 for a closed shell with the total
//! density, 1 per spin otherwise), orbital rotations U (virtual-occupied) and
//! S1 = C^T dS C, dP_s = -(1/n_s) P_s dS P_s + n_s (C_v U C_o^T + h.c.). The SCF condition
//! C_v^T F_used C_o = 0 at every geometry gives A U = -b with
//!
//! ```text
//! (A U)_ai = (e_a - e_i) U_ai + (C_v^T dF_used(dP_U) C_o)_ai
//! b_ai     = (C_v^T (dF_used/dR|_P + dF_used(dP_oo)) C_o)_ai - e_i S1_ai
//! ```
//!
//! where dF_used(D) is the linear response of the Fock matrices to density changes D. Then
//!
//! ```text
//! dE/dR = dE/dR|_P - Tr(W dS) - sum_s (1/n_s) Tr(dF_s P_s dS P_s) - <z, b>
//! A^T z = g,   g_s = 2 n_s C_v^T dF_s C_o,   W = sum_s n_s C_o e_o C_o^T
//! ```
//!
//! A^T needs the adjoint of dF_used: the exact part F_used - F_x = dE'/dP (E' = E without the
//! dipole-quadrupole and quadrupole-quadrupole AES energies) has a symmetric Hessian, and F_x
//! is linear in P with an explicit adjoint. `<z, b>` splits into
//! - Tr(Z dF'/dR|_P) = d/dh dE'/dR|_(P + h Z) at h = 0 (Z = sym(C_v z C_o^T) per spin), the
//!   directional derivative of the explicit gradient of E': the integral derivatives are
//!   linear in their weights (one pass with combined weights), the weights and the terms'
//!   gradients are differentiated by central differences (exact for energies up to third
//!   order in P; the charge dependence of revD4 leaves < 1e-12);
//! - Tr(Z dF_x/dR|_P) and the dS terms, contracted analytically with the overlap derivative.
//!
//! In the Z-vector equations (restarted GMRES, diagonal preconditioner: [`crate::krylov::gmres`])
//! dF_used(D) is the exchange of D (linear in D, with the program's extra onsite part) plus the
//! engine's analytic SCC kernels applied to the charges and moments of D, with the AES'
//! potentials of those changes for the program's AES ([`Setup::program_kernel`]).
//!
//! At a finite electronic temperature the divided differences of the occupations come from
//! `axtb::response::OccupationResponse` (every orbital a row, levels within 1e-10 Eh
//! degenerate) and C^T X C from `axtb::response::MoSpace` ([`Setup::program_fermi_gradient`]).

use super::ProgramFock;
use crate::dense::matmul as matmul_rows;
use crate::krylov::{gmres, GmresError};
use crate::v2::gradient::{MultipoleTerm, Potentials};
use crate::v2::scf::{ScfResult, Setup, State};
use crate::v2::terms::buffer;
use axtb::response::{MoSpace, OccupationResponse};
use axtb::terms::{Chain, MomentTerm};
use axtb::{Error, Result};
use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatRef, Par};

/// Largest density change of the finite-difference linearizations (elementwise).
const FD_STEP: f64 = 1e-4;
/// Relative residual of the Z-vector equations.
const Z_TOL: f64 = 1e-9;
const Z_MAX_ITER: usize = 300;
const GMRES_RESTART: usize = 40;

/// Occupied/virtual partition of one spin channel of a converged SCF.
struct Channel {
    /// Occupation number of the occupied orbitals (2 closed shell, 1 open shell).
    occ: f64,
    nocc: usize,
    /// Orbital energies and coefficients (row-major n x n, column k = orbital k).
    eps: Vec<f64>,
    c: Vec<f64>,
}

impl Channel {
    fn nvir(&self, n: usize) -> usize {
        n - self.nocc
    }

    fn cmat(&self, n: usize) -> MatRef<'_, f64> {
        MatRef::from_row_major_slice(&self.c, n, n)
    }

    /// C_v^T M C_o (nvir x nocc, row-major).
    fn vo(&self, n: usize, m: &[f64], par: Par) -> Vec<f64> {
        let (no, nv) = (self.nocc, self.nvir(n));
        let c = self.cmat(n);
        let mut mc = Mat::<f64>::zeros(n, no);
        matmul(mc.as_mut(), Accum::Replace, MatRef::from_row_major_slice(m, n, n), c.subcols(0, no), 1.0, par);
        let mut out = Mat::<f64>::zeros(nv, no);
        matmul(out.as_mut(), Accum::Replace, c.subcols(no, nv).transpose(), mc.as_ref(), 1.0, par);
        let mut v = vec![0.0; nv * no];
        for a in 0..nv {
            for i in 0..no {
                v[a * no + i] = out[(a, i)];
            }
        }
        v
    }

    /// sym(C_v z diag(f) C_o^T) for a virtual-occupied z (f = 1 when `f` is None).
    fn ao(&self, n: usize, z: &[f64], f: Option<&[f64]>, par: Par) -> Vec<f64> {
        let (no, nv) = (self.nocc, self.nvir(n));
        let c = self.cmat(n);
        let zm = MatRef::from_row_major_slice(z, nv, no);
        let mut cz = Mat::<f64>::zeros(n, no);
        matmul(cz.as_mut(), Accum::Replace, c.subcols(no, nv), zm, 1.0, par);
        if let Some(f) = f {
            for i in 0..no {
                for mu in 0..n {
                    cz[(mu, i)] *= f[i];
                }
            }
        }
        let mut d = Mat::<f64>::zeros(n, n);
        matmul(d.as_mut(), Accum::Replace, cz.as_ref(), c.subcols(0, no).transpose(), 1.0, par);
        let mut out = vec![0.0; n * n];
        for mu in 0..n {
            for nu in 0..n {
                out[mu * n + nu] = 0.5 * (d[(mu, nu)] + d[(nu, mu)]);
            }
        }
        out
    }
}

/// a += f b for chains of one layout (the direct gradient and every slot of `slots`).
fn add_scaled(a: &mut Chain, f: f64, b: &Chain, slots: &[axtb::terms::Slot]) {
    for (x, y) in a.g.iter_mut().zip(&b.g) {
        for k in 0..3 {
            x[k] += f * y[k];
        }
    }
    for &s in slots {
        for (x, y) in a.slot(s).iter_mut().zip(b.weights(s)) {
            *x += f * y;
        }
    }
}

impl Setup {
    /// The program's Fock correction (v1.1 in the program's conventions).
    fn program_fock(&self) -> Result<&ProgramFock> {
        self.program.as_ref().ok_or_else(|| Error::Input("g-xTB: the Z-vector gradient is the program's".into()))
    }

    /// The distinct slots of the chain layout.
    fn chain_slots(&self) -> Vec<axtb::terms::Slot> {
        let s = self.slots;
        let mut out: Vec<axtb::terms::Slot> = Vec::new();
        for x in [s.qeff, s.qeff0, s.cn, s.cnb, s.cnd4, s.eeq] {
            if !out.iter().any(|y| y.index() == x.index()) {
                out.push(x);
            }
        }
        debug_assert_eq!(out.len(), self.layout.slots());
        out
    }

    /// y~ of F_x of one spin density `d` (closed shell: the total density):
    /// y~_m = -(w/8) sum_k m2_mk (D S)_kk within the atom of m, w = 1 (closed) or 2, with the
    /// couplings m2 of the onsite exchange ([`crate::v2::exchange::Dual`]).
    fn ofx_y(&self, d: &[f64], closed: bool) -> Vec<f64> {
        let n = self.nao;
        let Some(dual) = &self.exch.dual else { return vec![0.0; n] };
        let w = if closed { 1.0 } else { 2.0 };
        let s = &self.s;
        let ds: Vec<f64> = (0..n).map(|k| (0..n).map(|j| d[k * n + j] * s[j * n + k]).sum::<f64>()).collect();
        let mut y = vec![0.0; n];
        for &(i0, i1) in &self.exch.atom_ao {
            for m in i0..i1 {
                for k in i0..i1 {
                    let l2 = dual.m2[m * n + k];
                    if l2 != 0.0 {
                        y[m] += l2 * ds[k];
                    }
                }
            }
        }
        y.iter_mut().for_each(|v| *v *= -w / 8.0);
        y
    }

    /// u = (dy~/d(DS)_jj)^T r with r_m = sum_(k != m) E_mk S_mk, for a symmetric E: the
    /// vector of the adjoint of F_x.
    fn ofx_u(&self, e: &[f64], closed: bool) -> Vec<f64> {
        let n = self.nao;
        let Some(dual) = &self.exch.dual else { return vec![0.0; n] };
        let w = if closed { 1.0 } else { 2.0 };
        let s = &self.s;
        let r: Vec<f64> =
            (0..n).map(|m| (0..n).filter(|&k| k != m).map(|k| e[m * n + k] * s[m * n + k]).sum()).collect();
        let mut u = vec![0.0; n];
        for &(i0, i1) in &self.exch.atom_ao {
            for m in i0..i1 {
                for j in i0..i1 {
                    let l2 = dual.m2[m * n + j];
                    if l2 != 0.0 {
                        u[j] += l2 * r[m];
                    }
                }
            }
        }
        u.iter_mut().for_each(|v| *v *= -w / 8.0);
        u
    }

    /// F_x of one spin density `d`: the part of the program's onsite-exchange Fock matrix
    /// beyond the exact derivative, 1/2 (y~_k + y~_l) S_kl off the diagonal (linear in `d`).
    pub(crate) fn ofx_extra(&self, d: &[f64], closed: bool) -> Vec<f64> {
        let n = self.nao;
        let y = self.ofx_y(d, closed);
        let mut f = vec![0.0; n * n];
        for k in 0..n {
            for l in 0..n {
                if k != l {
                    f[k * n + l] = 0.5 * (y[k] + y[l]) * self.s[k * n + l];
                }
            }
        }
        f
    }

    /// Adjoint of F_x: the symmetric G with Tr(E F_x(P)) = Tr(G P) for all symmetric P.
    fn ofx_extra_adjoint(&self, e: &[f64], closed: bool) -> Vec<f64> {
        let n = self.nao;
        let u = self.ofx_u(e, closed);
        let mut g = vec![0.0; n * n];
        for k in 0..n {
            for l in 0..n {
                g[k * n + l] = 0.5 * (u[k] + u[l]) * self.s[k * n + l];
            }
        }
        g
    }

    /// Overlap weight X of Tr(E dF_x(P)/dR) = Tr(X dS/dR) at fixed P (symmetric E).
    fn ofx_extra_overlap_weight(&self, p: &[f64], e: &[f64], closed: bool) -> Vec<f64> {
        let n = self.nao;
        let y = self.ofx_y(p, closed);
        let u = self.ofx_u(e, closed);
        let mut x = vec![0.0; n * n];
        for k in 0..n {
            for l in 0..n {
                let mut v = 0.5 * (u[k] + u[l]) * p[k * n + l];
                if k != l {
                    v += 0.5 * (y[k] + y[l]) * e[k * n + l];
                }
                x[k * n + l] = v;
            }
        }
        x
    }

    /// The program's potentials of the charges and moments of `st` (its Fock matrix but the
    /// exchange): the shell potential (isotropic, the AES' and revD4 potentials of the shell's
    /// atom), the AES' dipole and quadrupole potentials, the spin potential.
    fn program_potentials(&self, pf: &ProgramFock, st: &State) -> Potentials {
        let (tg, s, terms) = (self.term_geometry(), st.moments(), &self.terms);
        let (nsh, nat) = (self.shells.len(), self.mol.nat());
        let (mut piso, mut prep) = (buffer(&terms.iso, nsh, nat), buffer(&terms.rep, nsh, nat));
        let (mut pdisp, mut pspin) = (buffer(&terms.disp, nsh, nat), buffer(&terms.spin, nsh, nat));
        terms.iso.potential(&tg, &self.isotropic, &s, &mut piso);
        let (va, vd, vq) = pf.aes_potentials(&self.mol, s.qat, s.dip, s.quad, self.parallel);
        terms.rep.potential(&tg, &self.rep, &s, &mut prep);
        terms.disp.potential(&tg, &self.disp, &s, &mut pdisp);
        let mut vsh = piso.shell;
        for (ish, &(iat, _, _)) in self.shells.iter().enumerate() {
            vsh[ish] += va[iat] + prep.atom[iat] + pdisp.atom[iat];
        }
        let w = st.mag.is_some().then(|| {
            terms.spin.potential(&tg, &(), &s, &mut pspin);
            pspin.magnetization
        });
        Potentials { vsh, vd_scaled: vd.clone(), vd, vq, w }
    }

    /// The Fock matrices (one per spin density of `x`) of the potentials `pot` and the
    /// exchange parts `x`, without H.
    fn fock_of(&self, pot: &Potentials, x: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = self.nao;
        let vao: Vec<f64> = self.basis.scf.ao2sh.iter().map(|&ish| pot.vsh[ish]).collect();
        let wao: Option<Vec<f64>> = pot.w.as_ref().map(|w| self.basis.scf.ao2sh.iter().map(|&ish| w[ish]).collect());
        crate::fock::from_potentials_fused(
            n,
            &vec![0.0; n * n],
            &self.s,
            &self.basis.scf.ao2at,
            &self.dpint,
            &self.qpint,
            &vao,
            &pot.vd,
            &pot.vq,
            wao.as_deref(),
            x,
            self.parallel,
        )
    }

    /// F_AES'' of the state `st`: the AES Fock matrix of the dipole-quadrupole and
    /// quadrupole-quadrupole energies (the exact AES Fock matrix less the program's).
    fn aes_fock_difference(&self, pf: &ProgramFock, st: &State) -> Vec<f64> {
        let n = self.nao;
        let (tg, s) = (self.term_geometry(), st.moments());
        let (nsh, nat) = (self.shells.len(), self.mol.nat());
        let mut paes = buffer(&self.terms.aes, nsh, nat);
        self.terms.aes.potential(&tg, &self.aes, &s, &mut paes);
        let (va, vd, vq) = pf.aes_potentials(&self.mol, s.qat, s.dip, s.quad, self.parallel);
        let vsh: Vec<f64> = self.shells.iter().map(|&(iat, _, _)| paes.atom[iat] - va[iat]).collect();
        let dvd: Vec<[f64; 3]> =
            paes.dipole.iter().zip(&vd).map(|(a, b)| std::array::from_fn(|k| a[k] - b[k])).collect();
        let dvq: Vec<[f64; 6]> =
            paes.quadrupole.iter().zip(&vq).map(|(a, b)| std::array::from_fn(|k| a[k] - b[k])).collect();
        let pot = Potentials { vsh, vd_scaled: dvd.clone(), vd: dvd, vq: dvq, w: None };
        self.fock_of(&pot, &[vec![0.0; n * n]]).pop().expect("one Fock matrix")
    }

    /// The program's SCC kernel applied to SCC-variable changes `ds` at the state `st`: the
    /// isotropic, repulsion, revD4 and spin kernels of the engine's terms and the AES' potentials
    /// of the changes (E_AES' is bilinear in the moments: its potentials of the changes are its
    /// kernel), with the exchange's shell potential change `vx`.
    #[allow(clippy::type_complexity)]
    fn program_kernel(
        &self,
        pf: &ProgramFock,
        st: &State,
        ds: &State,
        vx: &[f64],
    ) -> (Vec<f64>, Vec<[f64; 3]>, Vec<[f64; 6]>, Option<Vec<f64>>) {
        let (tg, s, d, terms) = (self.term_geometry(), st.moments(), ds.moments(), &self.terms);
        let (nsh, nat) = (self.shells.len(), self.mol.nat());
        let (mut piso, mut prep) = (buffer(&terms.iso, nsh, nat), buffer(&terms.rep, nsh, nat));
        let (mut pdisp, mut pspin) = (buffer(&terms.disp, nsh, nat), buffer(&terms.spin, nsh, nat));
        terms.iso.kernel_apply(&tg, &self.isotropic, &s, &d, &mut piso);
        let (va, vd, vq) = pf.aes_potentials(&self.mol, d.qat, d.dip, d.quad, self.parallel);
        terms.rep.kernel_apply(&tg, &self.rep, &s, &d, &mut prep);
        terms.disp.kernel_apply(&tg, &self.disp, &s, &d, &mut pdisp);
        let mut dv = piso.shell;
        for (ish, &(iat, _, _)) in self.shells.iter().enumerate() {
            dv[ish] += vx[ish] + va[iat] + prep.atom[iat] + pdisp.atom[iat];
        }
        let w = ds.mag.is_some().then(|| {
            terms.spin.kernel_apply(&tg, &(), &s, &d, &mut pspin);
            pspin.magnetization
        });
        (dv, vd, vq, w)
    }

    /// Linear response of the program's Fock matrices to the spin-density changes `d` at the
    /// state `st`: the engine's analytic kernels applied to the charges and moments of `d`, with
    /// the program's AES' ([`Setup::program_kernel`]), and the exchange of `d` with the program's
    /// extra onsite part (both linear in the density).
    fn program_fock_response(&self, pf: &ProgramFock, st: &State, d: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let n = self.nao;
        if d.iter().flatten().all(|&v| v == 0.0) {
            return vec![vec![0.0; n * n]; d.len()];
        }
        // charges and moments of d (no reference occupations)
        let dt = self.state(d);
        let dq: Vec<f64> = dt.q.iter().zip(&self.ref_occ).map(|(q, r)| q - r).collect();
        let dqa = self.isotropic.atom_charges(&dq);
        let ds = State { p: Vec::new(), q: dq, qa: dqa, dip: dt.dip, qp: dt.qp, mag: dt.mag };
        // the exchange of d, with the program's extra onsite part
        let pr = self.exch.products(&self.s, d, self.par);
        let (_, _, mut fx, vx) = self.exch.evaluate_parts(&self.s, d, &pr, &ds.q, true, self.par, self.parallel);
        pf.add_ofx_extra(self, &pr, &mut fx);
        let (vsh, vd, vq, w) = self.program_kernel(pf, st, &ds, &vx);
        let pot = Potentials { vsh, vd_scaled: vd.clone(), vd, vq, w };
        self.fock_of(&pot, &fx)
    }

    /// Adjoint response G = dF_used^T[d]: Tr(d dF_used[E]) = Tr(G E) for symmetric E (per spin).
    fn program_fock_response_adjoint(&self, pf: &ProgramFock, st: &State, d: &[Vec<f64>]) -> Vec<Vec<f64>> {
        let closed = d.len() == 1;
        let mut g = self.program_fock_response(pf, st, d);
        for (gs, ds) in g.iter_mut().zip(d) {
            let fx = self.ofx_extra(ds, closed);
            let ft = self.ofx_extra_adjoint(ds, closed);
            for k in 0..gs.len() {
                gs[k] += ft[k] - fx[k];
            }
        }
        g
    }

    /// dF = F - F_used per spin density of `st`.
    fn fock_difference(&self, pf: &ProgramFock, st: &State) -> Vec<Vec<f64>> {
        let n = self.nao;
        let closed = st.p.len() == 1;
        let fa = self.aes_fock_difference(pf, st);
        st.p.iter()
            .map(|ps| {
                let fx = self.ofx_extra(ps, closed);
                (0..n * n).map(|k| fa[k] - fx[k]).collect()
            })
            .collect()
    }

    /// The explicit gradient's parts at the state `st`: the integral weights and the terms'
    /// derivatives (`full`: of E, otherwise of E').
    fn explicit_parts(
        &self,
        pf: &ProgramFock,
        st: &State,
        full: bool,
    ) -> (crate::v2::gradient::IntegralWeights, Chain) {
        let pr = self.exch.products(&self.s, &st.p, self.par);
        let pot = if full { self.potentials_with(st, &pr) } else { self.potentials_program(pf, st, &pr) };
        let w = self.integral_weights(st, &pr, &pot, None);
        let mut ch = Chain::new(&self.layout, self.mol.nat());
        self.add_terms(st, &pr, &pot.vd_scaled, full, &mut ch, &mut crate::timing::Laps::new());
        (w, ch)
    }

    /// The potentials of the explicit gradient of E' at `st` (exchange products `pr`): those of
    /// [`Setup::potentials_with`] with the program's AES.
    fn potentials_program(&self, pf: &ProgramFock, st: &State, pr: &[crate::v2::exchange::Products]) -> Potentials {
        let mut pot = self.program_potentials(pf, st);
        let (_, _, vx) = self.exch.evaluate_with(&self.s, &st.p, pr, &st.q, false, self.par, self.parallel);
        for (v, x) in pot.vsh.iter_mut().zip(&vx) {
            *v += x;
        }
        pot
    }

    /// dE/dR|_P - d/dh dE'/dR|_(P + h Z) + Tr(X dS/dR): the explicit gradient at P (the state
    /// `st`) minus its derivative along the response density `zao` (energy without the
    /// dipole-quadrupole and quadrupole-quadrupole AES parts), plus the overlap weight `x`.
    fn response_tail(&self, pf: &ProgramFock, st: &State, zao: &[Vec<f64>], x: &[f64]) -> Result<Vec<[f64; 3]>> {
        let n = self.nao;
        let nat = self.mol.nat();
        let p = &st.p;
        let closed = p.len() == 1;
        // the integral derivatives are linear in their weights, so one pass with combined
        // weights; the terms and the weights by central differences
        let zmax = zao.iter().flatten().fold(0.0_f64, |a, v| a.max(v.abs()));
        let (mut w0, mut ch0) = self.explicit_parts(pf, st, true);
        let ztot: Vec<f64> = if closed { zao[0].clone() } else { (0..n * n).map(|k| zao[0][k] + zao[1][k]).collect() };
        let mut vs = None;
        if zmax > 0.0 {
            let h = FD_STEP / zmax;
            let shifted = |sgn: f64| -> State {
                self.state_of(
                    p.iter()
                        .zip(zao)
                        .map(|(ps, ds)| ps.iter().zip(ds).map(|(a, b)| a + sgn * h * b).collect())
                        .collect(),
                )
            };
            let (wp, cp) = self.explicit_parts(pf, &shifted(1.0), false);
            let (wm, cm) = self.explicit_parts(pf, &shifted(-1.0), false);
            let f = 1.0 / (2.0 * h);
            for k in 0..n * n {
                w0.x[k] -= f * (wp.x[k] - wm.x[k]);
            }
            let slots = self.chain_slots();
            add_scaled(&mut ch0, -f, &cp, &slots);
            add_scaled(&mut ch0, f, &cm, &slots);
            // B(P, v0) - [B(P + hZ, v+) - B(P - hZ, v-)]/2h = B(P, v0 - dv) + B(Z, -v_s)
            let mut vd_s = vec![[0.0; 3]; nat];
            let mut vq_s = vec![[0.0; 6]; nat];
            for a in 0..nat {
                for c in 0..3 {
                    w0.vd[a][c] -= f * (wp.vd[a][c] - wm.vd[a][c]);
                    vd_s[a][c] = -0.5 * (wp.vd[a][c] + wm.vd[a][c]);
                }
                for c in 0..6 {
                    w0.vq[a][c] -= f * (wp.vq[a][c] - wm.vq[a][c]);
                    vq_s[a][c] = -0.5 * (wp.vq[a][c] + wm.vq[a][c]);
                }
            }
            vs = Some((vd_s, vq_s));
        }
        for k in 0..n * n {
            w0.x[k] += x[k];
        }
        let mut terms = vec![MultipoleTerm { p: &w0.ptot, vd: &w0.vd, vq: &w0.vq }];
        if let Some((vd_s, vq_s)) = &vs {
            terms.push(MultipoleTerm { p: &ztot, vd: vd_s, vq: vq_s });
        }
        let p_one: Vec<f64> = w0.ptot.iter().zip(&ztot).map(|(a, b)| a - b).collect();
        let mut ch = Chain::new(&self.layout, nat);
        self.add_integrals(&w0.x, &terms, &p_one, &st.dip, &mut ch, &mut crate::timing::Laps::new())?;
        ch.add(&ch0);
        self.finish_chain(ch)
    }

    /// Analytic nuclear gradient (Eh/bohr) of the energy of a converged SCF `r` of this
    /// geometry in the program's conventions, including the response of the program's
    /// non-variational Fock matrix.
    pub(crate) fn program_gradient(&self, r: &ScfResult) -> Result<Vec<[f64; 3]>> {
        let pf = self.program_fock()?;
        let n = self.nao;
        let st = &r.state;
        let p = &st.p;
        let ns = p.len();
        let closed = ns == 1;
        if !r.all_orbitals || r.coefficients.len() != ns || r.orbital_energies.len() != ns || r.occupations.len() != ns
        {
            return Err(Error::Input("g-xTB gradient: SCF result without orbitals".into()));
        }
        if r.kt > 0.0 {
            return self.program_fermi_gradient(pf, r);
        }
        let ch: Vec<Channel> = (0..ns)
            .map(|s| {
                let occ = if closed { 2.0 } else { 1.0 };
                let nocc = r.occupations[s].iter().filter(|&&o| o > 0.0).count();
                Channel { occ, nocc, eps: r.orbital_energies[s].clone(), c: r.coefficients[s].clone() }
            })
            .collect();
        // dF = F_exact - F_used per spin
        let df = self.fock_difference(pf, st);
        // right-hand side g_s = 2 n_s C_v^T dF_s C_o
        let offsets: Vec<usize> = ch
            .iter()
            .scan(0, |acc, c| {
                let o = *acc;
                *acc += c.nvir(n) * c.nocc;
                Some(o)
            })
            .collect();
        let dim: usize = ch.iter().map(|c| c.nvir(n) * c.nocc).sum();
        let mut rhs = vec![0.0; dim];
        let mut diag = vec![0.0; dim];
        for (s, c) in ch.iter().enumerate() {
            let g = c.vo(n, &df[s], self.par);
            let (no, nv) = (c.nocc, c.nvir(n));
            for a in 0..nv {
                for i in 0..no {
                    rhs[offsets[s] + a * no + i] = 2.0 * c.occ * g[a * no + i];
                    diag[offsets[s] + a * no + i] = c.eps[no + a] - c.eps[i];
                }
            }
        }
        // A^T y = (e_a - e_i) y + 2 n_s C_v^T dF_used^T[sym(C_v y C_o^T)] C_o
        let to_ao = |y: &[f64]| -> Vec<Vec<f64>> {
            ch.iter()
                .enumerate()
                .map(|(s, c)| c.ao(n, &y[offsets[s]..offsets[s] + c.nvir(n) * c.nocc], None, self.par))
                .collect()
        };
        let apply = |y: &[f64]| -> Result<Vec<f64>> {
            let d = to_ao(y);
            let g = self.program_fock_response_adjoint(pf, st, &d);
            let mut out: Vec<f64> = y.iter().zip(&diag).map(|(a, b)| a * b).collect();
            for (s, c) in ch.iter().enumerate() {
                let v = c.vo(n, &g[s], self.par);
                for (k, x) in v.iter().enumerate() {
                    out[offsets[s] + k] += 2.0 * c.occ * x;
                }
            }
            Ok(out)
        };
        let z = gmres(apply, &diag, &rhs, Z_TOL, Z_MAX_ITER, GMRES_RESTART).map_err(z_vector_error)?;
        let zao = to_ao(&z);
        // overlap weight: -W - sum_s (1/n_s) sym(P dF P) + (1/n_s) sym(P G P) + sym(C_v z e_o C_o^T) - X_x
        let g = self.program_fock_response_adjoint(pf, st, &zao);
        let mut x = vec![0.0; n * n];
        for (s, c) in ch.iter().enumerate() {
            add_energy_weighted(n, c, &mut x);
            let mut t = df[s].clone();
            for k in 0..n * n {
                t[k] = g[s][k] - t[k];
            }
            let pt = matmul_rows(n, &p[s], &matmul_rows(n, &t, &p[s], self.par), self.par);
            let ze = c.ao(n, &z[offsets[s]..offsets[s] + c.nvir(n) * c.nocc], Some(&c.eps[..c.nocc]), self.par);
            let xx = self.ofx_extra_overlap_weight(&p[s], &zao[s], closed);
            for i in 0..n {
                for j in 0..n {
                    let k = i * n + j;
                    x[k] += 0.5 * (pt[k] + pt[j * n + i]) / c.occ + ze[k] - xx[k];
                }
            }
        }
        self.response_tail(pf, st, &zao, &x)
    }

    /// Program-mode gradient at a finite electronic temperature (fractional occupations).
    ///
    /// The free energy E[P] - TS is stationary in the occupations (Fermi levels) and in the
    /// rotations among orbitals of the program's Fock matrix F_used, so
    /// dE/dR = dE/dR|_P - Tr(W dS/dR) + sum_s Tr(D_s dP_s/dR) with D = F - F_used as at 0 K. The
    /// response of the program's SCF in the orbital basis of each channel,
    /// dP = C [chi(C^T (F_used^(R) + K[dP]) C) + sigma(C^T S^(R) C)] C^T, has
    /// chi(X)_pq = D1_pq X_pq - delta_pq n'_p sum_r n'_r X_rr / sum_r n'_r and
    /// sigma(S)_pq = -D2_pq S_pq + delta_pq n'_p sum_r n'_r e_r S_rr / sum_r n'_r (divided
    /// differences D1, D2 of n and n e between orbitals, the derivatives n' = dn/de on the
    /// diagonal and between degenerate orbitals, and the Fermi-level shift that keeps every
    /// channel's electron count). With the adjoint y = D + K^T[Z], Z = C chi(C^T y C) C^T
    /// (GMRES on y), the response term is Tr(Z F_used^(R)|_P) + Tr(C sigma^T(C^T y C) C^T S^(R)).
    fn program_fermi_gradient(&self, pf: &ProgramFock, r: &ScfResult) -> Result<Vec<[f64; 3]>> {
        let n = self.nao;
        let st = &r.state;
        let p = &st.p;
        let ns = p.len();
        let closed = ns == 1;
        let kt = r.kt;
        let nmax = if closed { 2.0 } else { 1.0 };
        let par = self.par;
        // D = F_exact - F_used per spin
        let mut df = self.fock_difference(pf, st);
        // The program's entropy leaves out levels with f <= 1e-9 or 1 - f <= 1e-9, so the
        // energy is not stationary in their occupations: dE/dR gains sum_p (e_p - mu) dn_p/dR
        // over those levels, with dn_p = (C^-1 dP C^-T)_pp + n_p S~_pp, i.e. a source
        // S C diag(q) C^T S in D and C diag(q n) C^T in the overlap weight (q_p = e_p - mu).
        let mut x_cut = vec![0.0; n * n];
        for s in 0..ns {
            let mu = r.efermi[s].or(r.efermi[0]).unwrap_or(0.0);
            let (c, f, e) = (&r.coefficients[s], &r.occupations[s], &r.orbital_energies[s]);
            let q: Vec<f64> = (0..n)
                .map(|k| {
                    let fk = f[k] / nmax;
                    if fk > 0.0 && fk < 1.0 && (fk <= 1e-9 || 1.0 - fk <= 1e-9) {
                        e[k] - mu
                    } else {
                        0.0
                    }
                })
                .collect();
            if q.iter().all(|&v| v == 0.0) {
                continue;
            }
            let mut cq = vec![0.0; n * n];
            let mut cqn = vec![0.0; n * n];
            for i in 0..n {
                for k in 0..n {
                    cq[i * n + k] = c[i * n + k] * q[k];
                    cqn[i * n + k] = c[i * n + k] * q[k] * f[k];
                }
            }
            let ct: Vec<f64> = (0..n * n).map(|k| c[(k % n) * n + k / n]).collect();
            let cqc = matmul_rows(n, &cq, &ct, par);
            let src = matmul_rows(n, &matmul_rows(n, &self.s, &cqc, par), &self.s, par);
            let w = matmul_rows(n, &cqn, &ct, par);
            for k in 0..n * n {
                df[s][k] += 0.5 * (src[k] + src[(k % n) * n + k / n]);
                x_cut[k] += 0.5 * (w[k] + w[(k % n) * n + k / n]);
            }
        }
        // every orbital's occupation responds (all n rows); levels within 1e-10 Eh are degenerate
        struct Ch {
            c: Mat<f64>,
            occ: OccupationResponse,
        }
        let ch: Vec<Ch> = (0..ns)
            .map(|s| {
                let occ = &r.occupations[s];
                let dn: Vec<f64> = occ.iter().map(|&o| -o * (1.0 - o / nmax) / kt).collect();
                let resp =
                    OccupationResponse::new(&r.orbital_energies[s], vec![occ.clone()], vec![dn], n, |de, _, _| {
                        de.abs() <= 1e-10
                    });
                Ch { c: MatRef::from_row_major_slice(&r.coefficients[s], n, n).to_owned(), occ: resp }
            })
            .collect();
        // C^T X C and C Y C^T (row-major n x n)
        let to_mo = |c: &Mat<f64>, x: &[f64]| -> Vec<f64> {
            let o = MoSpace::new(c.as_ref(), n).to_mo(MatRef::from_row_major_slice(x, n, n), par);
            (0..n * n).map(|k| o[(k / n, k % n)]).collect()
        };
        let to_ao = |c: &Mat<f64>, y: &[f64]| -> Vec<f64> {
            let mut t = Mat::<f64>::zeros(n, n);
            matmul(t.as_mut(), Accum::Replace, c.as_ref(), MatRef::from_row_major_slice(y, n, n), 1.0, par);
            let mut o = Mat::<f64>::zeros(n, n);
            matmul(o.as_mut(), Accum::Replace, t.as_ref(), c.transpose(), 1.0, par);
            (0..n * n).map(|k| 0.5 * (o[(k / n, k % n)] + o[(k % n, k / n)])).collect()
        };
        let chi = |c: &Ch, y: &[f64]| -> Vec<f64> {
            let mut x = to_mo(&c.c, y);
            let (dn, dnsum) = (&c.occ.df()[0], c.occ.z()[0]);
            let a = if dnsum != 0.0 { (0..n).map(|q| dn[q] * x[q * n + q]).sum::<f64>() / dnsum } else { 0.0 };
            let d1 = c.occ.d1();
            for k in 0..n * n {
                x[k] *= d1[(k / n, k % n)];
            }
            for q in 0..n {
                x[q * n + q] -= dn[q] * a;
            }
            to_ao(&c.c, &x)
        };
        let sigma_t = |c: &Ch, y: &[f64]| -> Vec<f64> {
            let mut x = to_mo(&c.c, y);
            let (dn, dnsum) = (&c.occ.df()[0], c.occ.z()[0]);
            let a = if dnsum != 0.0 { (0..n).map(|q| dn[q] * x[q * n + q]).sum::<f64>() / dnsum } else { 0.0 };
            let (d2, eps) = (c.occ.d2(), c.occ.eps());
            for k in 0..n * n {
                x[k] *= -d2[(k / n, k % n)];
            }
            for q in 0..n {
                x[q * n + q] += dn[q] * eps[q] * a;
            }
            to_ao(&c.c, &x)
        };
        // (1 - K^T chi) y = D
        let rhs: Vec<f64> = df.concat();
        let apply = |y: &[f64]| -> Result<Vec<f64>> {
            let z: Vec<Vec<f64>> = (0..ns).map(|s| chi(&ch[s], &y[s * n * n..(s + 1) * n * n])).collect();
            let g = self.program_fock_response_adjoint(pf, st, &z);
            Ok(y.iter().zip(g.concat()).map(|(a, b)| a - b).collect())
        };
        let y = gmres(apply, &vec![1.0; rhs.len()], &rhs, Z_TOL, Z_MAX_ITER, GMRES_RESTART).map_err(z_vector_error)?;
        let z: Vec<Vec<f64>> = (0..ns).map(|s| chi(&ch[s], &y[s * n * n..(s + 1) * n * n])).collect();
        // overlap weight: -W + sum_s C sigma^T(C^T y_s C) C^T + X_x(Z) (the F_x part of
        // Tr(Z F_used^(R)|_P))
        let mut x = x_cut;
        for s in 0..ns {
            let (c, f, e) = (&r.coefficients[s], &r.occupations[s], &r.orbital_energies[s]);
            for k in 0..n {
                let w = f[k] * e[k];
                if w == 0.0 {
                    continue;
                }
                for i in 0..n {
                    let ci = w * c[i * n + k];
                    for j in 0..n {
                        x[i * n + j] -= ci * c[j * n + k];
                    }
                }
            }
            let st_ = sigma_t(&ch[s], &y[s * n * n..(s + 1) * n * n]);
            let xx = self.ofx_extra_overlap_weight(&p[s], &z[s], closed);
            for k in 0..n * n {
                x[k] += st_[k] + xx[k];
            }
        }
        // the tail subtracts the derivative along its density: pass -Z
        let zao: Vec<Vec<f64>> = z.iter().map(|zs| zs.iter().map(|v| -v).collect()).collect();
        self.response_tail(pf, st, &zao, &x)
    }
}

/// x -= W = n_s C_o e_o C_o^T of one spin channel.
fn add_energy_weighted(n: usize, c: &Channel, x: &mut [f64]) {
    for k in 0..c.nocc {
        let f = c.occ * c.eps[k];
        for i in 0..n {
            let ci = f * c.c[i * n + k];
            for j in 0..n {
                x[i * n + j] -= ci * c.c[j * n + k];
            }
        }
    }
}

/// The error of a failed Z-vector solve.
fn z_vector_error(e: GmresError) -> Error {
    match e {
        GmresError::Operator(e) => e,
        GmresError::NotConverged { residual } => {
            Error::Input(format!("g-xTB gradient: Z-vector equations not converged (residual {residual:.1e})"))
        }
    }
}
