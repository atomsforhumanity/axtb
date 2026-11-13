//! Nuclear Hessians of g-xTB 2.0.1 from the coupled-perturbed response ([`crate::response`]).
//!
//! With the analytic gradient written as a functional g(R; P, W) = dE/dR|_P - Tr(W dS/dR) of
//! the AO spin densities and the energy-weighted density ([`Setup::explicit_gradient`]):
//!
//! Both run on the setup's threads ([`Setup::set_threads`]): the displacements in parallel,
//! each sequentially, with every sum in a fixed order, so the result does not depend on the
//! thread count. Both need every orbital of the reference SCF ([`ScfResult::all_orbitals`]).
//!
//! - **response** ([`Setup::response_hessian`]): column y is the central difference of
//!   g(R +- h e_y; P +- h dP^y, W +- h dW^y), the wavefunction extrapolated to first order by the
//!   exact response instead of an SCF per displacement. The second-order errors of the
//!   extrapolation are the same on both sides and cancel, so the error is O(h^2) like any
//!   central difference, without SCF noise. The perturbation (F^(y), S^y) comes from the same
//!   two displaced setups (central difference at fixed density).
//! - **analytic** ([`Setup::analytic_hessian`], the default): H_xy = E^xy|_{P,W} +
//!   sum_s Tr(F^(x)_s dP^y_s) - Tr(S^x dW^y) with the analytic perturbations F^(x), S^x and the
//!   explicit second derivatives of [`crate::analytic`], [`crate::explicit`] and
//!   [`crate::tangent`]; no displaced setup and no displaced gradient.

use crate::dense::{from_mat, to_mat};
use crate::response::{Cpks, CpksOptions};
use crate::scf::{ScfResult, Setup};
use crate::timing::Timings;
use axtb::{Error, Result};
use axtb_base::{fd, threads};
use faer::{Mat, MatMut, MatRef, Par};

/// A Hessian of [`Setup::response_hessian`] or [`Setup::analytic_hessian`]: the row-major
/// 3N x 3N matrix (not symmetrized; [`fd::symmetrize`]), the dipole derivatives ∂μ/∂R_x (3N rows,
/// e) and the wall time per phase.
pub type HessianParts = (Vec<f64>, Vec<[f64; 3]>, Timings);

impl Setup {
    /// Response Hessian of a converged SCF `r` with step `step` (bohr), with the dipole
    /// derivatives as central differences of the dipoles of the extrapolated densities at the
    /// displaced geometries.
    pub fn response_hessian(&mut self, r: &ScfResult, step: f64, cpks: CpksOptions) -> Result<HessianParts> {
        if !(step > 0.0 && step.is_finite()) {
            return Err(Error::Input(format!("Hessian step must be positive and finite, got {step}")));
        }
        self.on_displacements(r, |s, parallel| s.response_columns(r, step, cpks, parallel))
    }

    /// Analytic Hessian of a converged SCF `r` ([`crate::analytic`]), with the dipole
    /// derivatives: the fixed-density part (integral derivatives of the Mulliken charges and
    /// atomic dipoles) plus the Mulliken analysis of the density response dP^x.
    ///
    /// Displacements are processed in blocks ([`AnalyticOptions::block`]); for each, the
    /// perturbation F^(x), S^x, the coupled-perturbed response dP^x, dW^x and its contraction
    /// with every displacement's local derivative rows
    /// ([`crate::analytic::Analytic::response_column`]). Only Hessian columns and vectors of the
    /// SCC-variable layout are kept, so the memory grows like N^2 (plus a few n x n matrices per
    /// thread). The phases "perturbations" and "response" are the wall time of the
    /// displacement loop split in proportion to the summed time of each part.
    pub fn analytic_hessian(&mut self, r: &ScfResult, opts: &AnalyticOptions) -> Result<HessianParts> {
        self.on_displacements(r, |s, parallel| s.analytic_columns(r, opts, parallel))
    }

    /// `f(setup, parallel)` with the setup's inner work sequential and `parallel` (its loop over
    /// displacements on the current rayon pool) for more than one thread; the threads are
    /// restored afterwards.
    fn on_displacements<T>(&mut self, r: &ScfResult, f: impl FnOnce(&Setup, bool) -> Result<T>) -> Result<T> {
        if !r.all_orbitals {
            return Err(Error::Input(
                "g-xTB 2.0.1 Hessians need every orbital: converge the reference SCF with all_orbitals".into(),
            ));
        }
        let threads = self.threads;
        self.set_threads(1)?;
        let out = f(self, threads > 1);
        self.set_threads(threads)?;
        out
    }

    fn response_columns(&self, r: &ScfResult, step: f64, cpks: CpksOptions, parallel: bool) -> Result<HessianParts> {
        let t0 = std::time::Instant::now();
        let n = self.nao;
        let nat = self.mol.nat();
        let n3 = 3 * nat;
        let seq = self;
        let cp = Cpks::new(seq, r, cpks, Par::Seq);
        let w = seq.energy_weighted_density(r);
        let mut timings = vec![("response setup", t0.elapsed().as_secs_f64())];
        let t0 = std::time::Instant::now();
        let column = |y: usize| -> Result<(Vec<f64>, [f64; 3])> {
            // the setups at R + h e_y and R - h e_y (in that order)
            let displaced = |d: f64| -> Result<Setup> {
                let mut m = seq.mol.clone();
                m.positions[y / 3][y % 3] += d;
                Setup::new(&m, seq.model)
            };
            let (sp, sm) = (displaced(step)?, displaced(-step)?);
            let (fp, fm) = (sp.fock_of_density(&r.density), sm.fock_of_density(&r.density));
            let fy: Vec<Mat<f64>> = fd::central2(&fp, &fm, step).iter().map(|v| to_mat(n, v)).collect();
            let sy = to_mat(n, &fd::central2(&sp.s, &sm.s, step));
            let resp = cp.solve(&fy, &sy)?;
            let dp: Vec<Vec<f64>> = resp.dp.iter().map(from_mat).collect();
            let mut dw = vec![0.0; n * n];
            for m in &resp.dw {
                for (a, b) in dw.iter_mut().zip(from_mat(m)) {
                    *a += b;
                }
            }
            // gradient and dipole of the setup `s` at offset `hs` with the extrapolated densities
            let at = |s: &Setup, hs: f64| -> Result<(Vec<[f64; 3]>, [f64; 3])> {
                let p: Vec<Vec<f64>> = r
                    .density
                    .iter()
                    .zip(&dp)
                    .map(|(p0, d)| p0.iter().zip(d).map(|(a, b)| a + hs * b).collect())
                    .collect();
                let wneg: Vec<f64> = w.iter().zip(&dw).map(|(a, b)| -(a + hs * b)).collect();
                let st = s.state(&p);
                let mut mu = [0.0; 3];
                for (a, pos) in s.mol.positions.iter().enumerate() {
                    for k in 0..3 {
                        mu[k] += st.qa[a] * pos[k] + st.dip[a][k];
                    }
                }
                Ok((s.explicit_gradient(&st, Some(&wneg))?, mu))
            };
            let (g, dmu) = fd::central2(&at(&sp, step)?, &at(&sm, -step)?, step);
            Ok((g.into_flattened(), dmu))
        };
        let cols = threads::map(n3, parallel, column).into_iter().collect::<Result<Vec<_>>>()?;
        let (cols, dipd): (Vec<Vec<f64>>, Vec<[f64; 3]>) = cols.into_iter().unzip();
        let h = fd::from_columns(&cols, n3);
        timings.push(("displaced gradients", t0.elapsed().as_secs_f64()));
        Ok((h, dipd, timings))
    }
}

/// Settings of the analytic Hessian.
#[derive(Clone, Copy, Debug)]
pub struct AnalyticOptions {
    /// The response solver; the default tolerance is 1e-6, which with `symmetric` gives the
    /// Hessian of a 1e-10 solve (tests/gxtb2_hessian_symmetric.py).
    pub cpks: CpksOptions,
    /// The response terms in their symmetric form: H - (M + M^T) with M_xy = z_x . r_y, z_x the
    /// solution of displacement x's symmetric system and r_y the final residual of y's. The
    /// Hessian's error is then quadratic in the solver's instead of linear, so a looser
    /// tolerance gives the same Hessian in fewer iterations (about 1.35x faster from ~100 atoms
    /// on). Costs one product Z R^T after all displacements and keeps z and r of every
    /// displacement in single precision: 2 * 3N * n_occ * n_virt * 4 bytes (taxol 55 MB,
    /// (H2O)100 0.58 GB).
    pub symmetric: bool,
    /// Displacements per block. Each displacement x is one task: its perturbation F^(x), S^x,
    /// the coupled-perturbed response dP^x, dW^x and their contraction into Hessian column x,
    /// after which its n x n matrices are freed; a block's tasks run in parallel and their
    /// results (vectors of length 3N and of the SCC layout) are summed in order. The work space
    /// is a few n x n matrices per thread; the result depends neither on the block size nor
    /// on the thread count.
    pub block: usize,
}

impl Default for AnalyticOptions {
    fn default() -> Self {
        Self { cpks: CpksOptions { tol: 1e-6, ..CpksOptions::default() }, symmetric: true, block: 64 }
    }
}

impl Setup {
    fn analytic_columns(&self, r: &ScfResult, opts: &AnalyticOptions, parallel: bool) -> Result<HessianParts> {
        let mut timings: Timings = Vec::new();
        let t0 = std::time::Instant::now();
        let n = self.nao;
        let nat = self.mol.nat();
        let n3 = 3 * nat;
        let seq = self;
        let an = crate::analytic::Analytic::new(seq, &r.state)?;
        let mut cp = Cpks::new(seq, r, opts.cpks, Par::Seq);
        cp.keep_solution = opts.symmetric;
        let cp = cp;
        timings.push(("tangents", t0.elapsed().as_secs_f64()));
        let t0 = std::time::Instant::now();
        let dim = an.lay.dim();
        let nsh = an.lay.nsh;
        // per displacement x: c^x, c^x + K b^x, the Mulliken analysis s^x of dP^x and the
        // exchange shell potential change v^x (b^x is an.bmat)
        let mut cm = vec![0.0; n3 * dim];
        let mut um = vec![0.0; n3 * dim];
        let mut sm = vec![0.0; n3 * dim];
        let mut vm = vec![0.0; n3 * nsh];
        let mut h = vec![0.0; n3 * n3];
        let mut dipd = vec![[0.0; 3]; n3];
        let (mut t_pert, mut t_resp) = (0.0, 0.0);
        #[allow(clippy::type_complexity)]
        let task = |x: usize| -> Result<(
            crate::analytic::Displacement,
            crate::analytic::ResponseColumn,
            [f64; 3],
            f64,
            f64,
            Option<(Vec<f32>, Vec<f32>)>,
        )> {
            let t = std::time::Instant::now();
            let mut d = an.displacement(x)?;
            let tp = t.elapsed().as_secs_f64();
            let t = std::time::Instant::now();
            let fy: Vec<Mat<f64>> = d.f.iter().map(|f| to_mat(n, f)).collect();
            let sy = to_mat(n, &d.s);
            (d.f, d.s) = (Vec::new(), Vec::new());
            let (mut rs, kd) = cp.solve_parts(&fy, &sy)?;
            drop((fy, sy));
            // dipole derivative: fixed-density part plus the Mulliken analysis of dP^x
            let mut dmu = d.dmu;
            for (a, p) in seq.mol.positions.iter().enumerate() {
                for c in 0..3 {
                    dmu[c] += p[c] * kd.ds.qa[a] + kd.ds.dip[a][c];
                }
            }
            let mut dw = vec![0.0; n * n];
            for m in &rs.dw {
                for (a, b) in dw.iter_mut().zip(from_mat(m)) {
                    *a += b;
                }
            }
            let zr = rs.zr.take();
            drop(rs);
            let col = an.response_column(&kd, &dw);
            Ok((d, col, dmu, tp, t.elapsed().as_secs_f64(), zr))
        };
        let block = opts.block.max(1);
        // symmetric form: z_x and the final residual r_x of every displacement (rows of two
        // n3 x len matrices, single precision)
        let (mut zm, mut rm, mut len): (Vec<f32>, Vec<f32>, usize) = (Vec::new(), Vec::new(), 0);
        let mut y0 = 0;
        while y0 < n3 {
            let k = block.min(n3 - y0);
            let res = threads::map(k, parallel, |j| task(y0 + j));
            for (j, r) in res.into_iter().enumerate() {
                let (d, col, dmu, tp, tr, zr) = r?;
                t_pert += tp;
                t_resp += tr;
                let x = y0 + j;
                if let Some((z, r)) = zr {
                    if zm.is_empty() {
                        len = z.len();
                        zm = vec![0.0; n3 * len];
                        rm = vec![0.0; n3 * len];
                    }
                    zm[x * len..(x + 1) * len].copy_from_slice(&z);
                    rm[x * len..(x + 1) * len].copy_from_slice(&r);
                }
                dipd[x] = dmu;
                // row x: d^2E_x/dS dS and the overlap-kernel cross terms; column x: the
                // overlap-kernel cross terms of the other index, the kernel terms at fixed s
                // and the response
                for y in 0..n3 {
                    h[x * n3 + y] += d.hss[y] + d.hsg[y];
                }
                for y in 0..n3 {
                    h[y * n3 + x] += d.hsg[y] + d.hk[y] + col.h[y];
                }
                cm[x * dim..(x + 1) * dim].copy_from_slice(&d.c);
                for (u, (c, kb)) in um[x * dim..(x + 1) * dim].iter_mut().zip(d.c.iter().zip(&d.kb)) {
                    *u = c + kb;
                }
                sm[x * dim..(x + 1) * dim].copy_from_slice(&col.s);
                vm[x * nsh..(x + 1) * nsh].copy_from_slice(&col.vx);
            }
            y0 += k;
        }
        if len > 0 {
            // H -= M + M^T, M = Z R^T: the response terms' error becomes quadratic in the solver's
            let mut m = Mat::<f32>::zeros(n3, n3);
            faer::linalg::matmul::matmul(
                m.as_mut(),
                faer::Accum::Replace,
                MatRef::from_row_major_slice(&zm, n3, len),
                MatRef::from_row_major_slice(&rm, n3, len).transpose(),
                1.0f32,
                if parallel { Par::rayon(0) } else { Par::Seq },
            );
            for x in 0..n3 {
                for y in 0..n3 {
                    h[x * n3 + y] -= m[(x, y)] as f64 + m[(y, x)] as f64;
                }
            }
        }
        drop((zm, rm));
        let wall = t0.elapsed().as_secs_f64();
        let f = if t_pert + t_resp > 0.0 { t_pert / (t_pert + t_resp) } else { 0.0 };
        timings.push(("perturbations", wall * f));
        timings.push(("response", wall * (1.0 - f)));
        let t0 = std::time::Instant::now();
        // SCC couplings b^x.K b^y + c^x.b^y + b^x.c^y = B U^T + C B^T (U = C + K B) and the
        // response's potential terms (c^x + K b^x).s^y + b^x_q.v^y = U S^T + B_q V^T
        {
            let mut hm = MatMut::from_row_major_slice_mut(&mut h, n3, n3);
            let (bm, cmr, umr, smr, vmr) = (
                MatRef::from_row_major_slice(&an.bmat, n3, dim),
                MatRef::from_row_major_slice(&cm, n3, dim),
                MatRef::from_row_major_slice(&um, n3, dim),
                MatRef::from_row_major_slice(&sm, n3, dim),
                MatRef::from_row_major_slice(&vm, n3, nsh),
            );
            let bq = bm.subcols(0, nsh);
            for (a, b) in [(bm, umr), (cmr, bm), (umr, smr), (bq, vmr)] {
                faer::linalg::matmul::matmul(hm.as_mut(), faer::Accum::Add, a, b.transpose(), 1.0, Par::Seq);
            }
        }
        drop((cm, um, sm, vm));
        // integral terms: local Hessian chained to Cartesians, plus the chain second derivatives
        let mut wneg = seq.energy_weighted_density(r);
        wneg.iter_mut().for_each(|x| *x = -*x);
        let ptot = seq.total_density(&r.state.p);
        let xw = seq.overlap_weight_total(&r.state, &an.pot, &wneg);
        let hl = seq.integral_local_hessian(&xw, &ptot, &an.vd_tot, &an.pot.vq, parallel)?;
        let nl = 6 * nat;
        let jac = &an.tan.jac;
        // J (nl x n3): identity for positions, then q_eff, q_eff0, CN Jacobians
        let mut j = vec![0.0; nl * n3];
        for x in 0..n3 {
            j[x * n3 + x] = 1.0;
        }
        for a in 0..nat {
            for x in 0..n3 {
                j[(n3 + a) * n3 + x] = jac.qeff()[a * n3 + x];
                j[(n3 + nat + a) * n3 + x] = jac.qeff0()[a * n3 + x];
                j[(n3 + 2 * nat + a) * n3 + x] = jac.cn()[a * n3 + x];
            }
        }
        let jm = faer::MatRef::from_row_major_slice(&j, nl, n3);
        let hm = faer::MatRef::from_row_major_slice(&hl, nl, nl);
        let mut tmp = Mat::<f64>::zeros(nl, n3);
        faer::linalg::matmul::matmul(tmp.as_mut(), faer::Accum::Replace, hm, jm, 1.0, Par::Seq);
        let mut hi = Mat::<f64>::zeros(n3, n3);
        faer::linalg::matmul::matmul(hi.as_mut(), faer::Accum::Replace, jm.transpose(), tmp.as_ref(), 1.0, Par::Seq);
        let ch = seq.explicit_chain(&r.state, &r.state, Some(&wneg))?;
        let dcn_h0 = seq.h0_cn_weights(&ptot)?;
        let (dqeff, dqeff0) = (ch.weights(seq.slots.qeff), ch.weights(seq.slots.qeff0));
        let j2 = seq.chain_second_derivatives(jac, dqeff, dqeff0, &dcn_h0, parallel)?;
        let h3 = seq.three_body_hessian(jac.cnd4(), parallel)?;
        let hx = an.exchange_kernel_hessian();
        for x in 0..n3 {
            for y in 0..n3 {
                h[x * n3 + y] += hi[(x, y)] + j2[x * n3 + y] + h3[x * n3 + y] + hx[x * n3 + y];
            }
        }
        timings.push(("explicit integrals", t0.elapsed().as_secs_f64()));
        Ok((h, dipd, timings))
    }
}
