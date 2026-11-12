//! Analytic gradient of the converged g-xTB v1.1 SCF energy ([`Setup::gradient`]).
//!
//! With [`Conventions::Axtb`] the SCF diagonalizes the exact derivative of the energy, the
//! solution is stationary and dE/dR = dE/dR|_P - Tr(W dS/dR) ([`Setup::explicit_gradient`]).
//!
//! With [`Conventions::Gxtb`] the SCF diagonalizes the program's Fock matrix F_used, which
//! differs from the exact derivative F = dE/dP by dF = F - F_used = F_AES' - F_x: F_AES' is
//! the AES Fock matrix of the dipole-quadrupole and quadrupole-quadrupole energies (which the
//! program leaves out) and F_x (per spin) the extra half of the population part of the
//! onsite exchange the program applies off the diagonal, F_x,kl = 1/2 (y_k + y_l) S_kl
//! (k != l), with y = M diag(Pt), Pt_kk = c (P S)_kk (c = 1/2 closed shell, 1 open shell)
//! and M the population couplings of [`Setup::ofx_population_couplings`]. The gradient then
//! needs the response of the density (Z-vector method).
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
//!   linear in their weights (one pass with combined weights), the weights and the cheap
//!   terms are differentiated by central differences (exact for energies up to third order
//!   in P; the fourth-order charge term and the charge dependence of revD4 leave < 1e-12);
//! - Tr(Z dF_x/dR|_P) and the dS terms, contracted analytically with the overlap derivative.
//!
//! In the Z-vector equations (restarted GMRES, diagonal preconditioner) dF_used(D) is the
//! exchange part (linear in D) plus the potentials differentiated by central differences of
//! the cheap potential functions of the charges and moments.

use crate::data::GMUNU;
use crate::exchange::{gmunu_index, OFX_SCALE};
use crate::gradient::MultipoleTerm;
use crate::scf::{Conventions, DensityTerms, Potentials, ScfResult, Setup};
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

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn norm(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

impl Setup {
    /// Population couplings of the onsite exchange, (k, l, M_kl): y_k = sum_l M_kl Pt_ll for
    /// the AOs k, l of one shell with l >= 1, M_kl = -2 L_kl ((2l+1) d_kl - 1)/(2l+1).
    pub fn ofx_population_couplings(&self) -> Vec<(usize, usize, f64)> {
        let n = self.nao;
        let ao = &self.ao;
        let mut out = Vec::new();
        for i in 0..n {
            for j in 0..n {
                let (ai, si, li) = ao[i];
                let (aj, sj, lj) = ao[j];
                if ai != aj || si != sj || li == 0 {
                    continue;
                }
                let Some(k) = gmunu_index(li, lj) else { continue };
                let lam = OFX_SCALE * GMUNU[(k - 1) + 9 * (self.mol.numbers[ai] - 1)];
                let f = (if i == j { (2 * li + 1) as f64 } else { 0.0 } - 1.0) / (2 * li + 1) as f64;
                out.push((i, j, -2.0 * f * lam));
            }
        }
        out
    }

    fn ofx_scale(closed: bool) -> f64 {
        if closed {
            0.5
        } else {
            1.0
        }
    }

    /// F_x of one spin density: the part of the program's onsite-exchange Fock matrix beyond
    /// the exact derivative (linear in the density).
    pub fn ofx_extra(&self, ps: &[f64], closed: bool, m: &[(usize, usize, f64)]) -> Vec<f64> {
        let n = self.nao;
        let c = Self::ofx_scale(closed);
        let ptd: Vec<f64> =
            (0..n).map(|k| c * (0..n).map(|j| ps[k * n + j] * self.s[j * n + k]).sum::<f64>()).collect();
        let mut y = vec![0.0; n];
        for &(k, l, v) in m {
            y[k] += v * ptd[l];
        }
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

    /// w = M^T r with r_k = sum_(l != k) D_kl S_kl, for a symmetric D.
    fn ofx_extra_w(&self, d: &[f64], m: &[(usize, usize, f64)]) -> Vec<f64> {
        let n = self.nao;
        let r: Vec<f64> =
            (0..n).map(|k| (0..n).filter(|&l| l != k).map(|l| d[k * n + l] * self.s[k * n + l]).sum()).collect();
        let mut w = vec![0.0; n];
        for &(k, l, v) in m {
            w[l] += v * r[k];
        }
        w
    }

    /// Adjoint of F_x: the symmetric G with Tr(D F_x(P)) = Tr(G P) for all symmetric P.
    fn ofx_extra_adjoint(&self, d: &[f64], closed: bool, m: &[(usize, usize, f64)]) -> Vec<f64> {
        let n = self.nao;
        let c = Self::ofx_scale(closed);
        let w = self.ofx_extra_w(d, m);
        let mut g = vec![0.0; n * n];
        for k in 0..n {
            for l in 0..n {
                g[k * n + l] = 0.5 * c * (w[k] + w[l]) * self.s[k * n + l];
            }
        }
        g
    }

    /// Overlap weight X of Tr(D dF_x(P)/dR) = Tr(X dS/dR) at fixed P (symmetric D).
    fn ofx_extra_overlap_weight(&self, ps: &[f64], d: &[f64], closed: bool, m: &[(usize, usize, f64)]) -> Vec<f64> {
        let n = self.nao;
        let c = Self::ofx_scale(closed);
        let ptd: Vec<f64> =
            (0..n).map(|k| c * (0..n).map(|j| ps[k * n + j] * self.s[j * n + k]).sum::<f64>()).collect();
        let mut y = vec![0.0; n];
        for &(k, l, v) in m {
            y[k] += v * ptd[l];
        }
        let w = self.ofx_extra_w(d, m);
        let mut x = vec![0.0; n * n];
        for k in 0..n {
            for l in 0..n {
                let mut v = 0.5 * c * (w[k] + w[l]) * ps[k * n + l];
                if k != l {
                    v += 0.5 * (y[k] + y[l]) * d[k * n + l];
                }
                x[k * n + l] = v;
            }
        }
        x
    }

    /// Linear response of the program's Fock matrices to the spin-density changes `d` at `p`
    /// (terms `t` of `p`): the Mulliken and onsite exchange are linear in the density, the
    /// potentials are differentiated by central differences of the potential functions
    /// (cheap: charges and moments are linear in the density; exact for the terms up to
    /// third order in the charges).
    fn fock_response(&self, t: &DensityTerms, d: &[Vec<f64>]) -> Result<Vec<Vec<f64>>> {
        let n = self.nao;
        let dmax = d.iter().flatten().fold(0.0_f64, |m, v| m.max(v.abs()));
        if dmax == 0.0 {
            return Ok(vec![vec![0.0; n * n]; d.len()]);
        }
        let h = FD_STEP / dmax;
        // charges and moments of d (no reference occupations)
        let dt = self.density_terms(d);
        let dq: Vec<f64> = dt.q.iter().zip(&self.ref_occ).map(|(q, r)| q - r).collect();
        let dqa = crate::es::atom_charges(self.mol.nat(), &self.shells, &dq);
        let shifted = |sgn: f64| -> DensityTerms {
            let f = sgn * h;
            DensityTerms {
                ptot: Vec::new(),
                q: t.q.iter().zip(&dq).map(|(a, b)| a + f * b).collect(),
                qa: t.qa.iter().zip(&dqa).map(|(a, b)| a + f * b).collect(),
                dip: t.dip.iter().zip(&dt.dip).map(|(a, b)| std::array::from_fn(|k| a[k] + f * b[k])).collect(),
                qp: t.qp.iter().zip(&dt.qp).map(|(a, b)| std::array::from_fn(|k| a[k] + f * b[k])).collect(),
                mag: t.mag.as_ref().map(|m| m.iter().zip(dt.mag.as_ref().unwrap()).map(|(a, b)| a + f * b).collect()),
            }
        };
        let pp = self.potentials(&shifted(1.0), Conventions::Gxtb)?;
        let pm = self.potentials(&shifted(-1.0), Conventions::Gxtb)?;
        let f = 1.0 / (2.0 * h);
        let dpot = Potentials {
            vsh: pp.vsh.iter().zip(&pm.vsh).map(|(a, b)| f * (a - b)).collect(),
            vd: pp.vd.iter().zip(&pm.vd).map(|(a, b)| std::array::from_fn(|k| f * (a[k] - b[k]))).collect(),
            vq: pp.vq.iter().zip(&pm.vq).map(|(a, b)| std::array::from_fn(|k| f * (a[k] - b[k]))).collect(),
            w: pp.w.as_ref().map(|w| w.iter().zip(pm.w.as_ref().unwrap()).map(|(a, b)| f * (a - b)).collect()),
        };
        Ok(self.assemble_fock(vec![0.0; n * n], &dpot, d, Conventions::Gxtb))
    }

    /// Adjoint response G = dF_used^T[d]: Tr(d dF_used[E]) = Tr(G E) for symmetric E (per spin).
    fn fock_response_adjoint(
        &self,
        t: &DensityTerms,
        d: &[Vec<f64>],
        m: &[(usize, usize, f64)],
    ) -> Result<Vec<Vec<f64>>> {
        let closed = d.len() == 1;
        let mut g = self.fock_response(t, d)?;
        for (gs, ds) in g.iter_mut().zip(d) {
            let fx = self.ofx_extra(ds, closed, m);
            let ft = self.ofx_extra_adjoint(ds, closed, m);
            for k in 0..gs.len() {
                gs[k] += ft[k] - fx[k];
            }
        }
        Ok(g)
    }

    /// Analytic nuclear gradient (Eh/bohr) of the energy of a converged SCF `r` of this
    /// geometry, including the response of the program's non-variational Fock matrix.
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
    fn fermi_gradient(&self, r: &ScfResult) -> Result<Vec<[f64; 3]>> {
        let n = self.nao;
        let p = &r.density;
        let ns = p.len();
        let closed = ns == 1;
        let kt = r.kt;
        let nmax = if closed { 2.0 } else { 1.0 };
        let par = self.par;
        let m = self.ofx_population_couplings();
        let terms = self.density_terms(p);
        let ptot = crate::fock::total_density(p);
        // D = F_exact - F_used per spin
        let fa = self.aes_fock(&ptot, true);
        let fs = self.aes_fock(&ptot, false);
        let mut df: Vec<Vec<f64>> = p
            .iter()
            .map(|ps| {
                let fx = self.ofx_extra(ps, closed, &m);
                (0..n * n).map(|k| fa[k] - fs[k] - fx[k]).collect()
            })
            .collect();
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
            let cqc = crate::fock::matmul(n, &cq, &ct, par);
            let src = crate::fock::matmul(n, &crate::fock::matmul(n, &self.s, &cqc, par), &self.s, par);
            let w = crate::fock::matmul(n, &cqn, &ct, par);
            for k in 0..n * n {
                df[s][k] += 0.5 * (src[k] + src[(k % n) * n + k / n]);
                x_cut[k] += 0.5 * (w[k] + w[(k % n) * n + k / n]);
            }
        }
        struct Ch {
            c: Mat<f64>,
            eps: Vec<f64>,
            dn: Vec<f64>,
            dnsum: f64,
            d1: Vec<f64>,
            d2: Vec<f64>,
        }
        let ch: Vec<Ch> = (0..ns)
            .map(|s| {
                let eps = r.orbital_energies[s].clone();
                let occ = &r.occupations[s];
                let dn: Vec<f64> = occ.iter().map(|&o| -o * (1.0 - o / nmax) / kt).collect();
                let mut d1 = vec![0.0; n * n];
                let mut d2 = vec![0.0; n * n];
                for a in 0..n {
                    for b in 0..n {
                        let de = eps[a] - eps[b];
                        let v = if de.abs() > 1e-10 { (occ[a] - occ[b]) / de } else { 0.5 * (dn[a] + dn[b]) };
                        d1[a * n + b] = v;
                        d2[a * n + b] = 0.5 * (occ[a] + occ[b]) + 0.5 * (eps[a] + eps[b]) * v;
                    }
                }
                let dnsum = dn.iter().sum();
                Ch { c: MatRef::from_row_major_slice(&r.coefficients[s], n, n).to_owned(), eps, dn, dnsum, d1, d2 }
            })
            .collect();
        // C^T X C and C Y C^T (row-major n x n)
        let to_mo = |c: &Mat<f64>, x: &[f64]| -> Vec<f64> {
            let mut t = Mat::<f64>::zeros(n, n);
            matmul(t.as_mut(), Accum::Replace, c.transpose(), MatRef::from_row_major_slice(x, n, n), 1.0, par);
            let mut o = Mat::<f64>::zeros(n, n);
            matmul(o.as_mut(), Accum::Replace, t.as_ref(), c.as_ref(), 1.0, par);
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
            let a = if c.dnsum != 0.0 { (0..n).map(|q| c.dn[q] * x[q * n + q]).sum::<f64>() / c.dnsum } else { 0.0 };
            for k in 0..n * n {
                x[k] *= c.d1[k];
            }
            for q in 0..n {
                x[q * n + q] -= c.dn[q] * a;
            }
            to_ao(&c.c, &x)
        };
        let sigma_t = |c: &Ch, y: &[f64]| -> Vec<f64> {
            let mut x = to_mo(&c.c, y);
            let a = if c.dnsum != 0.0 { (0..n).map(|q| c.dn[q] * x[q * n + q]).sum::<f64>() / c.dnsum } else { 0.0 };
            for k in 0..n * n {
                x[k] *= -c.d2[k];
            }
            for q in 0..n {
                x[q * n + q] += c.dn[q] * c.eps[q] * a;
            }
            to_ao(&c.c, &x)
        };
        // (1 - K^T chi) y = D
        let rhs: Vec<f64> = df.concat();
        let apply = |y: &[f64]| -> Result<Vec<f64>> {
            let z: Vec<Vec<f64>> = (0..ns).map(|s| chi(&ch[s], &y[s * n * n..(s + 1) * n * n])).collect();
            let g = self.fock_response_adjoint(&terms, &z, &m)?;
            Ok(y.iter().zip(g.concat()).map(|(a, b)| a - b).collect())
        };
        let y = gmres(apply, &vec![1.0; rhs.len()], &rhs)?;
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
            let st = sigma_t(&ch[s], &y[s * n * n..(s + 1) * n * n]);
            let xx = self.ofx_extra_overlap_weight(&p[s], &z[s], closed, &m);
            for k in 0..n * n {
                x[k] += st[k] + xx[k];
            }
        }
        // the tail subtracts the derivative along its density: pass -Z
        let zao: Vec<Vec<f64>> = z.iter().map(|zs| zs.iter().map(|v| -v).collect()).collect();
        self.response_tail(p, &zao, &x)
    }

    pub fn gradient(&self, r: &ScfResult) -> Result<Vec<[f64; 3]>> {
        let n = self.nao;
        let p = &r.density;
        let ns = p.len();
        let closed = ns == 1;
        if r.coefficients.len() != ns || r.orbital_energies.len() != ns || r.occupations.len() != ns {
            return Err(Error::Input("g-xTB gradient: SCF result without orbitals".into()));
        }
        let ch: Vec<Channel> = (0..ns)
            .map(|s| {
                let occ = if closed { 2.0 } else { 1.0 };
                let nocc = r.occupations[s].iter().filter(|&&o| o > 0.0).count();
                Channel { occ, nocc, eps: r.orbital_energies[s].clone(), c: r.coefficients[s].clone() }
            })
            .collect();
        if r.conventions == Conventions::Axtb {
            // stationary SCF (at a finite temperature stationary in the occupations too, with the
            // exact Fermi level): dE/dR = dE/dR|_P - Tr(W dS/dR), W = C diag(f e) C^T
            let mut x = vec![0.0; n * n];
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
            }
            return self.explicit_gradient(p, true, Some(&x));
        }
        if r.kt > 0.0 {
            return self.fermi_gradient(r);
        }
        let m = self.ofx_population_couplings();
        let terms = self.density_terms(p);
        let ptot = crate::fock::total_density(p);
        // dF = F_exact - F_used per spin
        let fa = self.aes_fock(&ptot, true);
        let fs = self.aes_fock(&ptot, false);
        let df: Vec<Vec<f64>> = p
            .iter()
            .map(|ps| {
                let fx = self.ofx_extra(ps, closed, &m);
                (0..n * n).map(|k| fa[k] - fs[k] - fx[k]).collect()
            })
            .collect();
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
            let g = self.fock_response_adjoint(&terms, &d, &m)?;
            let mut out: Vec<f64> = y.iter().zip(&diag).map(|(a, b)| a * b).collect();
            for (s, c) in ch.iter().enumerate() {
                let v = c.vo(n, &g[s], self.par);
                for (k, x) in v.iter().enumerate() {
                    out[offsets[s] + k] += 2.0 * c.occ * x;
                }
            }
            Ok(out)
        };
        let z = gmres(apply, &diag, &rhs)?;
        let zao = to_ao(&z);
        // overlap weight: -W - sum_s (1/n_s) sym(P dF P) + (1/n_s) sym(P G P) + sym(C_v z e_o C_o^T) - X_x
        let g = self.fock_response_adjoint(&terms, &zao, &m)?;
        let mut x = vec![0.0; n * n];
        for (s, c) in ch.iter().enumerate() {
            add_energy_weighted(n, c, &mut x);
            let mut t = df[s].clone();
            for k in 0..n * n {
                t[k] = g[s][k] - t[k];
            }
            let pt = crate::fock::matmul(n, &p[s], &crate::fock::matmul(n, &t, &p[s], self.par), self.par);
            let ze = c.ao(n, &z[offsets[s]..offsets[s] + c.nvir(n) * c.nocc], Some(&c.eps[..c.nocc]), self.par);
            let xx = self.ofx_extra_overlap_weight(&p[s], &zao[s], closed, &m);
            for i in 0..n {
                for j in 0..n {
                    let k = i * n + j;
                    x[k] += 0.5 * (pt[k] + pt[j * n + i]) / c.occ + ze[k] - xx[k];
                }
            }
        }
        self.response_tail(p, &zao, &x)
    }

    /// dE/dR|_P - d/dh dE'/dR|_(P + h Z) + Tr(X dS/dR): the explicit gradient at P minus its
    /// derivative along the response density `zao` (energy without the dipole-quadrupole and
    /// quadrupole-quadrupole AES parts), plus the overlap weight `x`.
    fn response_tail(&self, p: &[Vec<f64>], zao: &[Vec<f64>], x: &[f64]) -> Result<Vec<[f64; 3]>> {
        let n = self.nao;
        let closed = p.len() == 1;
        // the integral derivatives are linear in their weights, so one pass with combined
        // weights; the cheap terms and the weights by central differences
        let zmax = zao.iter().flatten().fold(0.0_f64, |a, v| a.max(v.abs()));
        let mut e0 = self.explicit_parts(p, true)?;
        let ztot: Vec<f64> = if closed { zao[0].clone() } else { (0..n * n).map(|k| zao[0][k] + zao[1][k]).collect() };
        let mut terms_v = Vec::new();
        if zmax > 0.0 {
            let h = FD_STEP / zmax;
            let shifted = |sgn: f64| -> Vec<Vec<f64>> {
                p.iter().zip(zao).map(|(ps, ds)| ps.iter().zip(ds).map(|(a, b)| a + sgn * h * b).collect()).collect()
            };
            let ep = self.explicit_parts(&shifted(1.0), false)?;
            let em = self.explicit_parts(&shifted(-1.0), false)?;
            let f = 1.0 / (2.0 * h);
            for k in 0..n * n {
                e0.x[k] -= f * (ep.x[k] - em.x[k]);
            }
            e0.chain.add_scaled(-f, &ep.chain);
            e0.chain.add_scaled(f, &em.chain);
            // B(P, v0) - [B(P + hZ, v+) - B(P - hZ, v-)]/2h = B(P, v0 - dv) + B(Z, -v_s)
            let nat = self.mol.nat();
            let mut vd_s = vec![[0.0; 3]; nat];
            let mut vq_s = vec![[0.0; 6]; nat];
            for a in 0..nat {
                for c in 0..3 {
                    e0.vd[a][c] -= f * (ep.vd[a][c] - em.vd[a][c]);
                    vd_s[a][c] = -0.5 * (ep.vd[a][c] + em.vd[a][c]);
                }
                for c in 0..6 {
                    e0.vq[a][c] -= f * (ep.vq[a][c] - em.vq[a][c]);
                    vq_s[a][c] = -0.5 * (ep.vq[a][c] + em.vq[a][c]);
                }
            }
            terms_v.push((vd_s, vq_s));
        }
        for k in 0..n * n {
            e0.x[k] += x[k];
        }
        let mut terms = vec![MultipoleTerm { p: &e0.ptot, vd: &e0.vd, vq: &e0.vq }];
        if let Some((vd_s, vq_s)) = terms_v.first() {
            terms.push(MultipoleTerm { p: &ztot, vd: vd_s, vq: vq_s });
        }
        let p_one: Vec<f64> = e0.ptot.iter().zip(&ztot).map(|(a, b)| a - b).collect();
        let grad = self.finish_explicit(&e0.x, &terms, &p_one, e0.chain.clone())?;
        Ok(grad)
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

/// Right-preconditioned restarted GMRES for `op(x) = b` with the diagonal preconditioner
/// `diag` (x = u / diag).
fn gmres(mut op: impl FnMut(&[f64]) -> Result<Vec<f64>>, diag: &[f64], b: &[f64]) -> Result<Vec<f64>> {
    let dim = b.len();
    let mut x = vec![0.0; dim];
    let bnorm = norm(b);
    if bnorm == 0.0 {
        return Ok(x);
    }
    let pre =
        |v: &[f64]| -> Vec<f64> { v.iter().zip(diag).map(|(a, d)| if d.abs() > 1e-12 { a / d } else { *a }).collect() };
    let mut iters = 0;
    loop {
        let ax = op(&x)?;
        let r: Vec<f64> = b.iter().zip(&ax).map(|(a, c)| a - c).collect();
        let beta = norm(&r);
        if beta <= Z_TOL * bnorm {
            return Ok(x);
        }
        if iters >= Z_MAX_ITER {
            return Err(Error::Input(format!(
                "g-xTB gradient: Z-vector equations not converged (residual {:.1e})",
                beta / bnorm
            )));
        }
        let m = GMRES_RESTART.min(dim);
        let mut v: Vec<Vec<f64>> = vec![r.iter().map(|a| a / beta).collect()];
        let mut h = vec![vec![0.0; m]; m + 1];
        let (mut cs, mut sn) = (vec![0.0; m], vec![0.0; m]);
        let mut e1 = vec![0.0; m + 1];
        e1[0] = beta;
        let mut k_used = 0;
        for k in 0..m {
            iters += 1;
            let mut w = op(&pre(&v[k]))?;
            for j in 0..=k {
                h[j][k] = dot(&w, &v[j]);
                for t in 0..dim {
                    w[t] -= h[j][k] * v[j][t];
                }
            }
            h[k + 1][k] = norm(&w);
            for j in 0..k {
                let t = cs[j] * h[j][k] + sn[j] * h[j + 1][k];
                h[j + 1][k] = -sn[j] * h[j][k] + cs[j] * h[j + 1][k];
                h[j][k] = t;
            }
            let rr = (h[k][k] * h[k][k] + h[k + 1][k] * h[k + 1][k]).sqrt();
            cs[k] = h[k][k] / rr;
            sn[k] = h[k + 1][k] / rr;
            h[k][k] = rr;
            let hk1 = h[k + 1][k];
            h[k + 1][k] = 0.0;
            e1[k + 1] = -sn[k] * e1[k];
            e1[k] *= cs[k];
            k_used = k + 1;
            if e1[k + 1].abs() <= 0.1 * Z_TOL * bnorm || hk1 == 0.0 || iters >= Z_MAX_ITER {
                break;
            }
            v.push(w.iter().map(|a| a / hk1).collect());
        }
        // solve the triangular system and update x
        let mut yv = vec![0.0; k_used];
        for i in (0..k_used).rev() {
            let mut t = e1[i];
            for j in i + 1..k_used {
                t -= h[i][j] * yv[j];
            }
            yv[i] = t / h[i][i];
        }
        let mut u = vec![0.0; dim];
        for (j, yj) in yv.iter().enumerate() {
            for t in 0..dim {
                u[t] += yj * v[j][t];
            }
        }
        let du = pre(&u);
        for t in 0..dim {
            x[t] += du[t];
        }
    }
}
