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

use crate::molecule::Molecule;
use crate::response::Response;
use crate::scf::fock;
use crate::terms::AnyMomentTerm;
use crate::xtb::{ScfState, XtbModel};
use crate::Result;
use axtb_base::fd;
use faer::{Accum, Mat, Par};

pub use axtb_base::method::{HessianMethod, HessianOptions};

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

/// Full first-order response of displacement `y`: AO density and energy-weighted density.
pub(crate) fn displacement_response(resp: &Response, dv: &Mat<f64>, y: usize, par: Par) -> (Mat<f64>, Mat<f64>) {
    let st = resp.st;
    let bas = &resp.model.basis;
    let lay = resp.lay;
    let nao = bas.nao;
    let nat = lay.nat;
    let n3 = 3 * nat;
    let (a, k) = (y / 3, y % 3);
    let mut df = Mat::<f64>::zeros(nao, nao);
    let mut ds = Mat::<f64>::zeros(nao, nao);
    add_local(&mut df, &resp.blocks[a].fock[k], resp.ao_off[a], 1.0);
    add_local(&mut ds, &resp.blocks[a].overlap[k], resp.ao_off[a], 1.0);
    for c in 0..nat {
        let w = resp.jac()[c * n3 + y];
        if w != 0.0 {
            add_local(&mut df, &resp.blocks[c].cn, resp.ao_off[c], w);
        }
    }
    let col = dv.col_as_slice(y);
    let mut vsh = vec![0.0; lay.nsh];
    let mut vdp = vec![[0.0; 3]; nat];
    let mut vqp = vec![[0.0; 6]; nat];
    lay.unpack(col, &mut vsh, &mut vdp, &mut vqp, None);
    // the potential part of the Fock matrix change, -Σ_t δV_t O_t
    let vao: Vec<f64> = (0..nao).map(|nu| vsh[bas.ao2sh[nu]]).collect();
    let (mut b, mut vm) = (Mat::<f64>::zeros(nao, nao), Mat::<f64>::zeros(nao, nao));
    fock::potential_part(bas, &st.ints, &vao, &vdp, &vqp, &mut b, &mut vm, false);
    drop(b);
    for j in 0..nao {
        for i in 0..nao {
            df[(i, j)] += vm[(i, j)];
        }
    }
    let mo = resp.mo();
    let ft = mo.to_mo_full_first(df.as_ref(), par);
    let stt = mo.to_mo_full_first(ds.as_ref(), par);
    let r = resp.mo_response(ft.as_ref(), Some(stt.as_ref()));
    (resp.to_ao(&r.dp, par), resp.to_ao(&r.dw, par))
}

/// A Hessian (row-major 3N x 3N, not symmetrized), its dipole derivatives (3N rows ∂μ/∂R_x)
/// and the timings of its phases.
pub(crate) type HessianParts = (Vec<f64>, Option<Vec<[f64; 3]>>, Vec<(&'static str, f64)>);

/// Response Hessian (row-major 3N x 3N, not symmetrized) and timings.
pub(crate) fn response_hessian(
    model: &XtbModel,
    extra: &[std::sync::Arc<dyn AnyMomentTerm>],
    mol: &Molecule,
    st: &ScfState,
    threads: usize,
    step: f64,
) -> Result<HessianParts> {
    let resp = Response::new(model, mol, st, threads);
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
    lay.pack(&st.qsh, &st.dpat, &st.qpat, None, &mut s0);
    // column y: the gradient functional at R + d e_y with P + d δP^y, W + d δW^y, s + d δs^y
    let column = |y: usize| -> Result<Vec<f64>> {
        let (dp, dw) = displacement_response(&resp, &dv, y, Par::Seq);
        let (a, k) = (y / 3, y % 3);
        let g = fd::try_derivative(fd::Stencil::Central2, step, |hs| {
            let mut mm = mol.clone();
            mm.positions[a][k] += hs;
            let p = &st.density + &dp * faer::Scale(hs);
            let w = &st.wdensity + &dw * faer::Scale(hs);
            let s: Vec<f64> = (0..m).map(|t| s0[t] + hs * r.ds[(t, y)]).collect();
            let mut qsh = vec![0.0; lay.nsh];
            let mut dpat = vec![[0.0; 3]; nat];
            let mut qpat = vec![[0.0; 6]; nat];
            lay.unpack(&s, &mut qsh, &mut dpat, &mut qpat, None);
            model.gradient_functional(&mm, extra, &p, &w, &qsh, &dpat, &qpat, false)
        })?;
        Ok(g.into_flattened())
    };
    let cols = axtb_base::threads::map(n3, threads > 1, column).into_iter().collect::<Result<Vec<_>>>()?;
    let h = fd::from_columns(&cols, n3);
    timings.push(("displaced gradients", t0.elapsed().as_secs_f64()));
    Ok((h, dipd, timings))
}
