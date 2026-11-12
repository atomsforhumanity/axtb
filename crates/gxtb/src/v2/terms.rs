//! The energy terms of g-xTB 2.0.1 as core terms ([`axtb::terms`]). The setup ([`Setup`]) is
//! the model at one geometry: it holds the terms' caches (`disp`, ...) and calls the terms in
//! the order of the program's arithmetic (its energy sum, its shell-potential fold, its
//! gradient order), which the core leaves to the model. Their energies are totals (g-xTB sums
//! no per-atom energies).
//!
//! [`Setup`]: super::scf::Setup

use super::disp::RevD4;
use super::intermediates::Slots;
use super::Model;
use crate::data_v2::INCREMENT;
use axtb::terms::{Chain, Chain2, Geometry, GeometryTerm, MomentTerm, Needs, Potentials, SccMoments, Slot, TermEnergy};
use axtb::Result;

/// The terms of a 2.0.1 model.
#[derive(Clone, Debug)]
pub struct Terms {
    pub increments: Increments,
    pub disp: RevD4Term,
}

impl Terms {
    /// The terms of `model` with the intermediates `slots`.
    pub fn new(model: &Model, slots: &Slots) -> Self {
        Self { increments: Increments, disp: RevD4Term { model: *model, cnd4: [slots.cnd4] } }
    }
}

/// The atomic energy increments: a constant per element.
#[derive(Clone, Copy, Debug)]
pub struct Increments;

impl GeometryTerm for Increments {
    type Cache = ();

    fn names(&self) -> &[&'static str] {
        &["increments"]
    }

    fn update(&self, _geo: &Geometry<'_>) -> Result<()> {
        Ok(())
    }

    fn energy(&self, geo: &Geometry<'_>, _c: &(), e: &mut TermEnergy<'_>, _chain: Option<&mut Chain>) {
        e.pieces[0] = geo.mol.numbers.iter().map(|&z| INCREMENT[z - 1]).sum();
    }

    fn add_hessian(&self, _geo: &Geometry<'_>, _c: &(), _h: &mut Chain2<'_>) -> Result<()> {
        Ok(())
    }
}

/// revD4 ([`super::disp`]): the two-body energy of the SCF charges (its energy and potential in
/// one pass) and the three-body energy at zero charges (the moment-independent part, once per
/// geometry); both depend on the revD4 coordination number.
#[derive(Clone, Debug)]
pub struct RevD4Term {
    model: Model,
    /// The revD4 CN.
    cnd4: [Slot; 1],
}

impl MomentTerm for RevD4Term {
    type Cache = RevD4;

    fn names(&self) -> &[&'static str] {
        &["dispersion"]
    }

    fn fixed_names(&self) -> &[&'static str] {
        &["dispersion_three_body"]
    }

    fn needs(&self) -> Needs {
        Needs { atom: true, ..Needs::default() }
    }

    fn slots(&self) -> &[Slot] {
        &self.cnd4
    }

    fn update(&self, geo: &Geometry<'_>) -> Result<RevD4> {
        RevD4::with_cn_par(&self.model, geo.mol, geo.value(self.cnd4[0]).to_vec(), geo.parallel)
    }

    /// The three-body energy and, with `chain`, its gradient: positions and dE/dCN (revD4).
    fn fixed_energy(&self, geo: &Geometry<'_>, c: &RevD4, e: &mut TermEnergy<'_>, chain: Option<&mut Chain>) {
        e.pieces[0] = match chain {
            Some(ch) => {
                let (e3, d3) = c.three_body_par(&self.model, geo.mol, Some(&mut ch.g), geo.parallel);
                let dcn = ch.slot(self.cnd4[0]);
                for (w, d) in dcn.iter_mut().zip(d3) {
                    *w += d;
                }
                e3
            }
            None => c.three_body_par(&self.model, geo.mol, None, geo.parallel).0,
        };
    }

    fn energy(&self, geo: &Geometry<'_>, c: &RevD4, s: &SccMoments<'_>, e: &mut TermEnergy<'_>) {
        e.pieces[0] = c.two_body_par(s.qat, geo.parallel).0;
    }

    fn potential(&self, geo: &Geometry<'_>, c: &RevD4, s: &SccMoments<'_>, v: &mut Potentials) {
        let (_, vd) = c.two_body_par(s.qat, geo.parallel);
        add(&mut v.atom, &vd);
    }

    fn energy_and_potential(
        &self,
        geo: &Geometry<'_>,
        c: &RevD4,
        s: &SccMoments<'_>,
        e: &mut TermEnergy<'_>,
        v: &mut Potentials,
    ) {
        let (e2, vd) = c.two_body_par(s.qat, geo.parallel);
        e.pieces[0] = e2;
        add(&mut v.atom, &vd);
    }

    /// The two-body gradient at fixed charges: positions and dE/dCN (revD4).
    fn gradient(&self, geo: &Geometry<'_>, c: &RevD4, s: &SccMoments<'_>, chain: &mut Chain) {
        let d2 = c.two_body_gradient_par(&self.model, geo.mol, s.qat, &mut chain.g, geo.parallel);
        add(chain.slot(self.cnd4[0]), &d2);
    }

    fn kernel_apply(
        &self,
        _geo: &Geometry<'_>,
        c: &RevD4,
        s: &SccMoments<'_>,
        ds: &SccMoments<'_>,
        dv: &mut Potentials,
    ) {
        add(&mut dv.atom, &c.two_body_hessian_vec(s.qat, ds.qat));
    }
}

/// `out += x` element by element. 2.0.1's terms compute their potentials and dE/dCN into fresh
/// vectors, as before, and add them to buffers the setup zeroed: the same values, except that a
/// -0 becomes +0. The setup folds each potential buffer into a shell's sum of the exchange,
/// AES, repulsion, dispersion and isotropic potentials, which can differ only in the sign of a
/// zero, and only where every one of them is exactly zero; the dE/dCN of the two- and
/// three-body parts meet in a zeroed slot, where 0 + (a + b) and (0 + a) + b agree for all a, b,
/// signed zeros included, and the revD4 CN's chain rule skips zero weights.
fn add(out: &mut [f64], x: &[f64]) {
    for (o, v) in out.iter_mut().zip(x) {
        *o += v;
    }
}
