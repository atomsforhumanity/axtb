//! The Fock matrix of the SCC potentials (tblite's `add_pot_to_h1` form):
//!
//! ```text
//! F = H0 - 1/2 (B + Bᵀ),   B(μ,ν) = S(μ,ν) v(ν) + D(μ,ν)·v_d(A(ν)) + Q(μ,ν)·v_q(A(ν))
//! ```
//!
//! with `v` the charge potential per AO (shell plus atom part), `v_d`, `v_q` the atomic dipole
//! and quadrupole potentials and the dipole and quadrupole integrals taken around the atom of
//! the column AO ν. This equals `H0 - 1/2 S(μ,ν) (v_μ + v_ν) - 1/2 [D(μ,ν)·v_d(A(ν)) +
//! D(ν,μ)·v_d(A(μ))] - (Q likewise)`. B is built column by column (contiguous memory, in
//! parallel when asked: each column by one task, so the result does not depend on threads),
//! then symmetrized in one pass.

use super::for_columns;
use crate::basis::Basis;
use crate::hamiltonian::Integrals;
use faer::Mat;

/// B(μ,ν) = S(μ,ν) v(ν) + D(μ,ν)·v_d(A(ν)) + Q(μ,ν)·v_q(A(ν)) for the potentials per AO
/// (`vao`) and the atomic dipole and quadrupole potentials, column by column.
fn potential_columns(
    bas: &Basis,
    ints: &Integrals,
    vao: &[f64],
    vdp: &[[f64; 3]],
    vqp: &[[f64; 6]],
    b: &mut Mat<f64>,
    parallel: bool,
) {
    let nao = bas.nao;
    for_columns(b, parallel, |nu, b| {
        let at = bas.ao2at[nu];
        let s = ints.overlap.col_as_slice(nu);
        let v = vao[nu];
        for mu in 0..nao {
            b[mu] = s[mu] * v;
        }
        for k in 0..3 {
            let c = vdp[at][k];
            if c != 0.0 {
                let d = ints.dip[k].col_as_slice(nu);
                for mu in 0..nao {
                    b[mu] += d[mu] * c;
                }
            }
        }
        for k in 0..6 {
            let c = vqp[at][k];
            if c != 0.0 {
                let q = ints.quad[k].col_as_slice(nu);
                for mu in 0..nao {
                    b[mu] += q[mu] * c;
                }
            }
        }
    });
}

/// `out = H0 - 1/2 (B + Bᵀ)` with `H0 = ints.h0`, or `-1/2 (B + Bᵀ)` without it; `work` holds
/// B afterwards.
fn assemble(
    bas: &Basis,
    ints: &Integrals,
    with_h0: bool,
    vao: &[f64],
    vdp: &[[f64; 3]],
    vqp: &[[f64; 6]],
    work: &mut Mat<f64>,
    out: &mut Mat<f64>,
    parallel: bool,
) {
    let nao = bas.nao;
    potential_columns(bas, ints, vao, vdp, vqp, work, parallel);
    let work = &*work;
    for_columns(out, parallel, |nu, f| {
        let b = work.col_as_slice(nu);
        if with_h0 {
            let h = ints.h0.col_as_slice(nu);
            for mu in 0..nao {
                f[mu] = h[mu] - 0.5 * (b[mu] + work[(nu, mu)]);
            }
        } else {
            for mu in 0..nao {
                f[mu] = -0.5 * (b[mu] + work[(nu, mu)]);
            }
        }
    });
}

/// The Fock matrix `F = H0 - 1/2 (B + Bᵀ)` of the potentials per AO (`vao`, shell plus atom
/// part) and the atomic dipole and quadrupole potentials, with B in `work`.
pub(crate) fn from_potentials(
    bas: &Basis,
    ints: &Integrals,
    vao: &[f64],
    vdp: &[[f64; 3]],
    vqp: &[[f64; 6]],
    work: &mut Mat<f64>,
    fock: &mut Mat<f64>,
    parallel: bool,
) {
    assemble(bas, ints, true, vao, vdp, vqp, work, fock, parallel);
}

/// The potential part `-1/2 (B + Bᵀ)` of the Fock matrix alone (`-Σ_t V_t O_t`, e.g. for a
/// change of the potentials), with B in `work`.
pub(crate) fn potential_part(
    bas: &Basis,
    ints: &Integrals,
    vao: &[f64],
    vdp: &[[f64; 3]],
    vqp: &[[f64; 6]],
    work: &mut Mat<f64>,
    out: &mut Mat<f64>,
    parallel: bool,
) {
    assemble(bas, ints, false, vao, vdp, vqp, work, out, parallel);
}
