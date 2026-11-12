//! Nuclear Hessians from the coupled-perturbed SCC response.
//!
//! - [`HessianMethod::Response`]: column y is the central difference of the gradient
//!   functional `g(R; P, W, s)` at `R ± h e_y`, with the density, energy-weighted density
//!   and SCC variables extrapolated to first order by the exact response
//!   (`P ± h δP^y`, `W ± h δW^y`, `s ± h δs^y`) instead of an SCF per displacement. The
//!   second-order errors of the extrapolation are the same on both sides and cancel, so
//!   the error is O(h²) like any central difference, without SCF noise.
//! - [`HessianMethod::Analytic`]: explicit second derivatives plus the response terms
//!   (see `analytic.rs`).

use crate::basis::Basis;
use crate::hamiltonian::Integrals;
use crate::molecule::Molecule;
use crate::response::Response;
use crate::xtb::{ChargeDependentTerm, ScfState, XtbModel};
use crate::Result;
use axtb_base::fd;
use faer::{Accum, Mat, Par};

/// How [`crate::Calculator::hessian`] computes second derivatives.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum HessianMethod {
    /// Central differences of analytic gradients, one SCF per displacement.
    Numerical,
    /// Central differences of the gradient functional with the wavefunction extrapolated
    /// by the coupled-perturbed SCC response (no SCF per displacement).
    Response,
    /// Analytic second derivatives with the coupled-perturbed SCC response.
    #[default]
    Analytic,
}

impl std::str::FromStr for HessianMethod {
    type Err = crate::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "numerical" | "numeric" | "fd" => Ok(HessianMethod::Numerical),
            "response" => Ok(HessianMethod::Response),
            "analytic" | "analytical" => Ok(HessianMethod::Analytic),
            other => {
                Err(crate::Error::Input(format!("unknown Hessian method {other:?} (numerical, response, analytic)")))
            }
        }
    }
}

/// Settings of a Hessian calculation.
#[derive(Clone, Debug, Default)]
pub struct HessianOptions {
    pub method: HessianMethod,
    /// Finite-difference step in bohr (Numerical: default 0.005; Response: default 5e-4;
    /// unused by Analytic).
    pub step: Option<f64>,
    /// SCF accuracy: of every displacement for Numerical (default 0.01), of the reference
    /// wavefunction for Response and Analytic (default 1e-3). Never looser than the
    /// calculator's own accuracy.
    pub accuracy: Option<f64>,
}

/// `M += L Eᵀ + E Lᵀ` (scaled by `w`) for a column block `l` of the AOs starting at `off`.
pub(crate) fn add_local(m: &mut Mat<f64>, l: &Mat<f64>, off: usize, w: f64) {
    let nao = l.nrows();
    for j in 0..l.ncols() {
        let col = l.col_as_slice(j);
        for mu in 0..nao {
            let v = w * col[mu];
            m[(mu, off + j)] += v;
            m[(off + j, mu)] += v;
        }
    }
}

/// `-Σ_t V_t O_t` for potentials per shell (`vsh`, atom parts folded in), atomic dipole
/// and quadrupole potentials: the potential part of the Fock matrix.
pub(crate) fn potential_matrix(
    bas: &Basis,
    ints: &Integrals,
    vsh: &[f64],
    vdp: &[[f64; 3]],
    vqp: &[[f64; 6]],
) -> Mat<f64> {
    let nao = bas.nao;
    let vao: Vec<f64> = (0..nao).map(|nu| vsh[bas.ao2sh[nu]]).collect();
    let mut b = Mat::<f64>::zeros(nao, nao);
    crate::xtb::potential_columns(bas, ints, &vao, vdp, vqp, &mut b, false);
    let mut out = Mat::<f64>::zeros(nao, nao);
    for j in 0..nao {
        for i in 0..nao {
            out[(i, j)] = -0.5 * (b[(i, j)] + b[(j, i)]);
        }
    }
    out
}

/// Full first-order response of displacement `y`: AO density and energy-weighted density.
pub(crate) fn displacement_response(resp: &Response, dv: &Mat<f64>, y: usize, par: Par) -> (Mat<f64>, Mat<f64>) {
    let st = resp.st;
    let bas = &resp.model.basis;
    let lay = resp.lay;
    let nao = bas.nao;
    let nat = lay.nat;
    let n3 = 3 * nat;
    let n_o = resp.occ.n_o;
    let (a, k) = (y / 3, y % 3);
    let mut df = Mat::<f64>::zeros(nao, nao);
    let mut ds = Mat::<f64>::zeros(nao, nao);
    add_local(&mut df, &resp.blocks[a].fock[k], resp.ao_off[a], 1.0);
    add_local(&mut ds, &resp.blocks[a].overlap[k], resp.ao_off[a], 1.0);
    for c in 0..nat {
        let w = resp.jac[c * n3 + y];
        if w != 0.0 {
            add_local(&mut df, &resp.blocks[c].cn, resp.ao_off[c], w);
        }
    }
    let col = dv.col_as_slice(y);
    let mut vsh = col[..lay.nsh].to_vec();
    let mut vdp = vec![[0.0; 3]; nat];
    let mut vqp = vec![[0.0; 6]; nat];
    lay.unpack(col, &mut vsh, &mut vdp, &mut vqp);
    let vm = potential_matrix(bas, &st.ints, &vsh, &vdp, &vqp);
    for j in 0..nao {
        for i in 0..nao {
            df[(i, j)] += vm[(i, j)];
        }
    }
    let c = st.coeff.as_ref();
    let to_mo = |m: &Mat<f64>| {
        let mut x = Mat::<f64>::zeros(nao, nao);
        faer::linalg::matmul::matmul(x.as_mut(), Accum::Replace, m.as_ref(), c, 1.0, par);
        let mut out = Mat::<f64>::zeros(n_o, nao);
        faer::linalg::matmul::matmul(out.as_mut(), Accum::Replace, c.subcols(0, n_o).transpose(), x.as_ref(), 1.0, par);
        out
    };
    let ft = to_mo(&df);
    let stt = to_mo(&ds);
    let r = resp.mo_response(ft.as_ref(), Some(stt.as_ref()));
    (resp.to_ao(&r.dp, par), resp.to_ao(&r.dw, par))
}

/// A Hessian (row-major 3N x 3N, not symmetrized), its dipole derivatives (3N rows ∂μ/∂R_x)
/// and the timings of its phases.
pub(crate) type HessianParts = (Vec<f64>, Option<Vec<[f64; 3]>>, Vec<(&'static str, f64)>);

/// Response Hessian (row-major 3N x 3N, not symmetrized) and timings.
pub(crate) fn response_hessian(
    model: &XtbModel,
    extra: &[&dyn ChargeDependentTerm],
    mol: &Molecule,
    st: &ScfState,
    threads: usize,
    step: f64,
) -> Result<HessianParts> {
    let resp = Response::new(model, extra, mol, st, threads);
    let r = resp.scc_response(false)?;
    let dipd = resp.dipole_derivatives(&r.ds);
    let mut timings = r.timings.clone();
    let t0 = std::time::Instant::now();
    let lay = resp.lay;
    let nat = lay.nat;
    let n3 = 3 * nat;
    let m = lay.dim();
    // δV = K δs + c
    let mut dv = r.c.clone();
    faer::linalg::matmul::matmul(dv.as_mut(), Accum::Add, r.kernel.as_ref(), r.ds.as_ref(), 1.0, resp.par());
    let mut s0 = vec![0.0; m];
    lay.pack(&st.qsh, &st.dpat, &st.qpat, &mut s0);
    // column y: the gradient functional at R + d e_y with P + d δP^y, W + d δW^y, s + d δs^y
    let column = |y: usize| -> Vec<f64> {
        let (dp, dw) = displacement_response(&resp, &dv, y, Par::Seq);
        let (a, k) = (y / 3, y % 3);
        let g = fd::derivative(fd::Stencil::Central2, step, |hs| {
            let mut mm = mol.clone();
            mm.positions[a][k] += hs;
            let p = &st.density + &dp * faer::Scale(hs);
            let w = &st.wdensity + &dw * faer::Scale(hs);
            let s: Vec<f64> = (0..m).map(|t| s0[t] + hs * r.ds[(t, y)]).collect();
            let mut qsh = vec![0.0; lay.nsh];
            let mut dpat = vec![[0.0; 3]; nat];
            let mut qpat = vec![[0.0; 6]; nat];
            lay.unpack(&s, &mut qsh, &mut dpat, &mut qpat);
            model.gradient_functional(&mm, extra, &p, &w, &qsh, &dpat, &qpat, false)
        });
        g.into_flattened()
    };
    let h = fd::from_columns(&axtb_base::threads::map(n3, threads > 1, column), n3);
    timings.push(("displaced gradients", t0.elapsed().as_secs_f64()));
    Ok((h, dipd, timings))
}
