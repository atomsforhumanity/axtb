//! The finite-difference defaults of the term traits: the kernel and its product with a vector
//! from central differences of the analytic potential, and the second derivatives with respect
//! to the nuclear positions from central differences of the analytic potential and gradient
//! (from the charge-dependent terms of GFN2-xTB, `xtb.rs`, before this module).

use super::chain::{Chain, Chain2, Values};
use super::sinks::{Geometry, Kernel, PotentialDerivative, Potentials, SccMoments, TermEnergy};
use super::{GeometryTerm, MomentTerm};
use crate::Result;
use axtb_base::fd::{self, Stencil};
use axtb_base::threads;

/// Step of the central differences of the potential in the moments.
const KERNEL_STEP: f64 = 1e-4;

/// Step (bohr) of the central differences in the nuclear positions.
pub(crate) const FD_STEP: f64 = 2e-5;

/// SCC moments that a finite difference perturbs.
struct Moments {
    qsh: Vec<f64>,
    qat: Vec<f64>,
    dip: Vec<[f64; 3]>,
    quad: Vec<[f64; 6]>,
    mag: Option<Vec<f64>>,
}

impl Moments {
    fn of(s: &SccMoments<'_>) -> Self {
        Self {
            qsh: s.qsh.to_vec(),
            qat: s.qat.to_vec(),
            dip: s.dip.to_vec(),
            quad: s.quad.to_vec(),
            mag: s.mag.map(|m| m.to_vec()),
        }
    }

    fn view(&self) -> SccMoments<'_> {
        SccMoments { qsh: &self.qsh, qat: &self.qat, dip: &self.dip, quad: &self.quad, mag: self.mag.as_deref() }
    }
}

/// The flattened potentials (shell, atom, dipole, quadrupole, magnetization) of `t` at `m`.
fn potentials<T: MomentTerm + ?Sized>(t: &T, geo: &Geometry<'_>, c: &T::Cache, m: &SccMoments<'_>) -> Vec<f64> {
    let mut v = Potentials::new(m.qsh.len(), m.qat.len(), m.mag.is_some());
    t.potential(geo, c, m, &mut v);
    v.flatten()
}

/// See [`MomentTerm::kernel`].
pub(super) fn kernel<T: MomentTerm + ?Sized>(
    t: &T,
    geo: &Geometry<'_>,
    c: &T::Cache,
    s: &SccMoments<'_>,
    k: &mut Kernel,
) {
    let needs = t.needs();
    let geo = Geometry { parallel: false, ..*geo };
    let (nsh, nat) = (s.qsh.len(), s.qat.len());
    let mut m = Moments::of(s);
    // the potentials for a change of one variable (`at` moves it by d from its value in `s`)
    let mut column = |at: &dyn Fn(&mut Moments, f64)| -> Vec<f64> {
        let col = fd::derivative(Stencil::Central2, KERNEL_STEP, |d| {
            at(&mut m, d);
            potentials(t, &geo, c, &m.view())
        });
        at(&mut m, 0.0);
        col
    };
    if needs.shell && !k.shell.is_empty() {
        for b in 0..nsh {
            let col = column(&|m, d| m.qsh[b] = s.qsh[b] + d);
            for a in 0..nsh {
                k.shell[a * nsh + b] += col[a];
            }
        }
    }
    if (needs.dipole || needs.quadrupole) && !k.multipole.is_empty() {
        let n10 = 10 * nat;
        let mut add = |b: usize, col: Vec<f64>| {
            for a in 0..n10 {
                k.multipole[a * n10 + b] += col[nsh + a];
            }
        };
        if needs.atom {
            for b in 0..nat {
                add(b, column(&|m, d| m.qat[b] = s.qat[b] + d));
            }
        }
        if needs.dipole {
            for i in 0..nat {
                for x in 0..3 {
                    add(nat + 3 * i + x, column(&|m, d| m.dip[i][x] = s.dip[i][x] + d));
                }
            }
        }
        if needs.quadrupole {
            for i in 0..nat {
                for x in 0..6 {
                    add(4 * nat + 6 * i + x, column(&|m, d| m.quad[i][x] = s.quad[i][x] + d));
                }
            }
        }
    } else if needs.atom && !k.atom.is_empty() {
        for b in 0..nat {
            let col = column(&|m, d| m.qat[b] = s.qat[b] + d);
            for a in 0..nat {
                k.atom[a * nat + b] += col[nsh + a];
            }
        }
    }
    if let (true, Some(mag)) = (needs.magnetization && !k.magnetization.is_empty(), s.mag) {
        let off = nsh + 10 * nat;
        for b in 0..nsh {
            let col = column(&|m, d| m.mag.as_mut().expect("magnetizations")[b] = mag[b] + d);
            for a in 0..nsh {
                k.magnetization[a * nsh + b] += col[off + a];
            }
        }
    }
}

/// See [`MomentTerm::kernel_apply`].
pub(super) fn kernel_apply<T: MomentTerm + ?Sized>(
    t: &T,
    geo: &Geometry<'_>,
    c: &T::Cache,
    s: &SccMoments<'_>,
    ds: &SccMoments<'_>,
    dv: &mut Potentials,
) {
    let geo = Geometry { parallel: false, ..*geo };
    let mut m = Moments::of(s);
    let col = fd::derivative(Stencil::Central2, KERNEL_STEP, |d| {
        let shift = |out: &mut [f64], a: &[f64], b: &[f64]| {
            for ((o, x), y) in out.iter_mut().zip(a).zip(b) {
                *o = x + d * y;
            }
        };
        shift(&mut m.qsh, s.qsh, ds.qsh);
        shift(&mut m.qat, s.qat, ds.qat);
        shift(m.dip.as_flattened_mut(), s.dip.as_flattened(), ds.dip.as_flattened());
        shift(m.quad.as_flattened_mut(), s.quad.as_flattened(), ds.quad.as_flattened());
        if let (Some(out), Some(a), Some(b)) = (m.mag.as_mut(), s.mag, ds.mag) {
            shift(out, a, b);
        }
        potentials(t, &geo, c, &m.view())
    });
    // the flattened layout of `Potentials::flatten`
    let (nsh, nat) = (s.qsh.len(), s.qat.len());
    let add = |out: &mut [f64], off: usize, len: usize| {
        for (o, x) in out.iter_mut().zip(&col[off..off + len]) {
            *o += x;
        }
    };
    add(&mut dv.shell, 0, nsh);
    add(&mut dv.atom, nsh, nat);
    add(dv.dipole.as_flattened_mut(), nsh + nat, 3 * nat);
    add(dv.quadrupole.as_flattened_mut(), nsh + 4 * nat, 6 * nat);
    if s.mag.is_some() {
        add(&mut dv.magnetization, nsh + 10 * nat, nsh);
    }
}

/// The values of the intermediates at a displaced geometry: recomputed only for a term that
/// reads some.
fn values_at(geo: &Geometry<'_>, mol: &crate::Molecule, uses: bool) -> Result<Values> {
    if uses {
        geo.layout.compute(mol)
    } else {
        Ok(Values::new(geo.layout))
    }
}

/// See [`MomentTerm::second_derivatives`].
pub(crate) fn second_derivatives<T: MomentTerm + ?Sized>(
    t: &T,
    geo: &Geometry<'_>,
    s: &SccMoments<'_>,
    dv: Option<&mut PotentialDerivative>,
    h: Option<&mut Chain2<'_>>,
) -> Result<()> {
    let mol = geo.mol;
    let nat = mol.nat();
    let n3 = 3 * nat;
    let nsh = s.qsh.len();
    let want_h = h.is_some();
    let uses = !t.slots().is_empty();
    // column x: the potentials and (with `h`) the gradient at R + d e_x
    let column = |x: usize| -> Result<(Vec<f64>, Vec<[f64; 3]>)> {
        let (a, k) = (x / 3, x % 3);
        let mut m = mol.clone();
        fd::try_derivative(Stencil::Central2, FD_STEP, |d| {
            m.positions[a][k] = mol.positions[a][k] + d;
            let values = values_at(geo, &m, uses)?;
            let g = Geometry { mol: &m, layout: geo.layout, values: &values, parallel: false };
            let c = t.update(&g)?;
            let v = potentials(t, &g, &c, s);
            let mut grad = Vec::new();
            if want_h {
                let mut chain = Chain::new(geo.layout, nat);
                let mut pieces = vec![0.0; t.fixed_names().len()];
                t.fixed_energy(&g, &c, &mut TermEnergy { pieces: &mut pieces, atoms: None }, Some(&mut chain));
                t.gradient(&g, &c, s, &mut chain);
                grad = chain.finish(&g)?;
            }
            Ok((v, grad))
        })
    };
    let needs = t.needs();
    let (mut dv, mut h) = (dv, h);
    for (x, col) in threads::map(n3, geo.parallel, column).into_iter().enumerate() {
        let (dvx, hx) = col?;
        if let Some(dv) = dv.as_deref_mut() {
            if needs.shell && !dv.shell.is_empty() {
                for i in 0..nsh {
                    dv.shell[i * n3 + x] += dvx[i];
                }
            }
            // the atomic rows: charges, then dipoles and quadrupoles where the model has them
            let rows = dv.atomic.len() / n3.max(1);
            let mut add_rows = |r0: usize, r1: usize| {
                for r in r0..r1.min(rows) {
                    dv.atomic[r * n3 + x] += dvx[nsh + r];
                }
            };
            if needs.atom {
                add_rows(0, nat);
            }
            if needs.dipole {
                add_rows(nat, 4 * nat);
            }
            if needs.quadrupole {
                add_rows(4 * nat, 10 * nat);
            }
        }
        if let Some(h) = h.as_deref_mut() {
            for y in 0..n3 {
                h.h[y * n3 + x] += hx[y / 3][y % 3];
            }
        }
    }
    Ok(())
}

/// See [`GeometryTerm::add_hessian`].
pub(super) fn geometry_hessian<T: GeometryTerm + ?Sized>(t: &T, geo: &Geometry<'_>, h: &mut Chain2<'_>) -> Result<()> {
    let mol = geo.mol;
    let nat = mol.nat();
    let n3 = 3 * nat;
    let uses = !t.slots().is_empty();
    let column = |x: usize| -> Result<Vec<[f64; 3]>> {
        let (a, k) = (x / 3, x % 3);
        let mut m = mol.clone();
        fd::try_derivative(Stencil::Central2, FD_STEP, |d| {
            m.positions[a][k] = mol.positions[a][k] + d;
            let values = values_at(geo, &m, uses)?;
            let g = Geometry { mol: &m, layout: geo.layout, values: &values, parallel: false };
            let c = t.update(&g)?;
            let mut chain = Chain::new(geo.layout, nat);
            let mut pieces = vec![0.0; t.names().len()];
            t.energy(&g, &c, &mut TermEnergy { pieces: &mut pieces, atoms: None }, Some(&mut chain));
            chain.finish(&g)
        })
    };
    for (x, col) in threads::map(n3, geo.parallel, column).into_iter().enumerate() {
        let col = col?;
        for y in 0..n3 {
            h.h[y * n3 + x] += col[y / 3][y % 3];
        }
    }
    Ok(())
}
