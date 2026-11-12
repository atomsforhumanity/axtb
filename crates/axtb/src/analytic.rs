//! Analytic nuclear Hessian of the SCC tight-binding energy.
//!
//! With the gradient written as a functional `g_x(R; P, W, s)` of the density, the
//! energy-weighted density and the SCC variables, and `V = V(R, s)` the potentials,
//!
//! ```text
//! H_xy = E''_xy + H^skel_xy + u^x·K δs^y + u^x·c^y + c^x·δs^y
//! ```
//!
//! - `E''`: explicit second derivatives at fixed P, W, V (core Hamiltonian, overlap and
//!   multipole integrals with the CN chain rule) plus those of the electrostatic,
//!   dispersion, solvation and repulsion energies at fixed SCC variables;
//! - `H^skel_xy = Tr(δP^y A^x) − Tr(δW^y S^x)` with the skeleton (fixed-potential) response
//!   of the density to perturbation y, `A^x = ∂F/∂x|_V`, `S^x = ∂S/∂x`;
//! - `u^x` the skeleton response of the SCC variables (including `b^x = −Tr(P ∂O/∂x)`),
//!   `c^x = ∂V/∂x|_s`, `K = ∂²E/∂s∂s` and `δs` the coupled-perturbed SCC response.
//!
//! The expression is symmetric term by term (see `response/occupation.rs` and `response/scc.rs`
//! for the derivation of the response quantities).

use crate::molecule::Molecule;
use crate::response::Response;
use crate::terms::{Chain2, GeometryTerm, MomentTerm};
use crate::xtb::{ChargeDependentTerm, ScfState, XtbModel};
use crate::Result;
use faer::{Accum, Mat};

/// Analytic Hessian (row-major 3N x 3N), dipole derivatives (3N rows ∂μ/∂R_x) and timings.
pub(crate) fn analytic_hessian(
    model: &XtbModel,
    extra: &[&dyn ChargeDependentTerm],
    mol: &Molecule,
    st: &ScfState,
    threads: usize,
) -> Result<crate::hessian::HessianParts> {
    let resp = Response::new(model, extra, mol, st, threads);
    let r = resp.scc_response(true)?;
    let dipd = resp.dipole_derivatives(&r.ds);
    let mut timings = r.timings.clone();
    let t0 = std::time::Instant::now();
    let nat = mol.nat();
    let n3 = 3 * nat;
    let nxi = resp.nxi();
    let par = resp.par();
    // T = [I; J]: local perturbations -> displacements
    let mut t = Mat::<f64>::zeros(nxi, n3);
    for x in 0..n3 {
        t[(x, x)] = 1.0;
    }
    for c in 0..nat {
        for x in 0..n3 {
            t[(n3 + c, x)] = resp.jac()[c * n3 + x];
        }
    }
    let hs = r.hskel.as_ref().expect("skeleton Hessian");
    let mut tmp = Mat::<f64>::zeros(nxi, n3);
    faer::linalg::matmul::matmul(tmp.as_mut(), Accum::Replace, hs.as_ref(), t.as_ref(), 1.0, par);
    let mut h = Mat::<f64>::zeros(n3, n3);
    faer::linalg::matmul::matmul(h.as_mut(), Accum::Replace, t.transpose(), tmp.as_ref(), 1.0, par);
    // u·K δs + u·c (y) + c·δs
    let mut kds = r.c.clone();
    faer::linalg::matmul::matmul(kds.as_mut(), Accum::Add, r.kernel.as_ref(), r.ds.as_ref(), 1.0, par);
    faer::linalg::matmul::matmul(h.as_mut(), Accum::Add, r.u.transpose(), kds.as_ref(), 1.0, par);
    faer::linalg::matmul::matmul(h.as_mut(), Accum::Add, r.c.transpose(), r.ds.as_ref(), 1.0, par);
    timings.push(("response terms", t0.elapsed().as_secs_f64()));
    let e2 = explicit(&resp, &r.extra_hessian, threads, &mut timings);
    let mut out = vec![0.0; n3 * n3];
    for x in 0..n3 {
        for y in 0..n3 {
            out[x * n3 + y] = h[(x, y)] + e2[x * n3 + y];
        }
    }
    Ok((out, dipd, timings))
}

/// Explicit second derivatives `E''` at fixed P, W, potentials and SCC variables (row-major
/// 3N x 3N): one-electron integrals with the CN chain rule, isotropic and anisotropic
/// electrostatics, repulsion, and the charge-dependent extra terms at fixed charges.
pub(crate) fn explicit(
    resp: &Response,
    extra_hessian: &[f64],
    threads: usize,
    timings: &mut Vec<(&'static str, f64)>,
) -> Vec<f64> {
    let st = resp.st;
    let mol = resp.mol;
    let model = resp.model;
    let bas = &model.basis;
    let n3 = 3 * mol.nat();
    let parallel = threads > 1;
    let t0 = std::time::Instant::now();
    let mut h = extra_hessian.to_vec();
    h.resize(n3 * n3, 0.0);
    let vao = crate::xtb::vao(&st.pot, bas);
    let geo = &st.geo;
    model.h0.add_hessian(
        mol,
        bas,
        &geo.se,
        &geo.dsedcn,
        geo.cutoff,
        &st.density,
        &st.wdensity,
        &vao,
        &st.pot.dipole,
        &st.pot.quadrupole,
        &model.ncoord,
        resp.jac(),
        &mut h,
        parallel,
    );
    timings.push(("explicit: integrals", t0.elapsed().as_secs_f64()));
    let t0 = std::time::Instant::now();
    let tg = model.term_geometry(mol, geo, parallel);
    let s = st.moments();
    let h2 = &mut Chain2 { h: &mut h, jacobians: &resp.jacobians };
    let seconds = "GFN2-xTB's analytic second derivatives cannot fail";
    model.es2.second_derivatives(&tg, &geo.gamma, &s, None, Some(&mut *h2)).expect(seconds);
    if let (Some(aes), Some(cache)) = (&model.aes2, &geo.mp_cache) {
        aes.second_derivatives(&tg, cache, &s, None, Some(&mut *h2)).expect(seconds);
    }
    model.repulsion.add_hessian(&tg, &(), h2).expect(seconds);
    timings.push(("explicit: electrostatics+repulsion", t0.elapsed().as_secs_f64()));
    h
}
