// SPDX-License-Identifier: LGPL-3.0-or-later
//! D4 dispersion (crate `axtb-d4`) as a self-consistent, charge-dependent SCF term.

use crate::molecule::Molecule;
use crate::terms::{
    Chain, Chain2, Geometry, Kernel, MomentTerm, Needs, PotentialDerivative, Potentials, SccMoments, TermEnergy,
};
use crate::ConventionParams;
use crate::{Error, Result};
use axtb_d4::{ChargeModel, D4Cache, D4Dispersion, D4Model, RationalDamping};

/// Self-consistent D4: two-body energy with the SCF Mulliken charges, three-body (ATM) at
/// zero charges, both with GFN2-xTB damping parameters.
pub(crate) struct D4Term(D4Dispersion);

impl D4Term {
    /// GFN2-xTB D4 with the coordination-number tables and real-space cutoffs of a
    /// convention ([`ConventionParams::d4_rcov`], [`d4_en`](ConventionParams::d4_en),
    /// [`d4_cutoff`](ConventionParams::d4_cutoff)).
    pub(crate) fn gfn2(mol: &Molecule, params: &ConventionParams) -> Result<Self> {
        let model = D4Model::new(&mol.numbers, ChargeModel::Gfn2)
            .map_err(|e| Error::Input(format!("D4: {e}")))?
            .with_cn_parameters(params.d4_rcov, params.d4_en);
        Ok(D4Term(
            D4Dispersion::new(model, RationalDamping::GFN2, params.d4_cutoff).with_pair_sums(params.d4_atm_pair_sums),
        ))
    }
}

/// GFN2-xTB's D4 as a moment term: the three-body energy at zero charges is its
/// moment-independent part (per atom, with its gradient), the two-body energy, potential,
/// gradient, charge kernel, potential derivatives and Hessian (of both parts) come from axtb-d4.
impl MomentTerm for D4Term {
    type Cache = D4Cache;

    fn names(&self) -> &[&'static str] {
        &["dispersion"]
    }

    fn fixed_names(&self) -> &[&'static str] {
        &["dispersion"]
    }

    fn needs(&self) -> Needs {
        Needs { atom: true, ..Needs::default() }
    }

    fn partitions_energy(&self) -> bool {
        true
    }

    fn update(&self, geo: &Geometry<'_>) -> Result<D4Cache> {
        Ok(self.0.update(&geo.mol.positions))
    }

    fn fixed_energy(&self, geo: &Geometry<'_>, c: &D4Cache, e: &mut TermEnergy<'_>, chain: Option<&mut Chain>) {
        let mut own = Vec::new();
        let atoms = e.atoms_or(&mut own, geo.mol.nat());
        self.0.three_body(c, atoms, chain.map(|ch| &mut ch.g[..]), geo.parallel);
        let total = atoms.iter().sum();
        e.pieces[0] = total;
    }

    fn energy(&self, geo: &Geometry<'_>, c: &D4Cache, s: &SccMoments<'_>, e: &mut TermEnergy<'_>) {
        let mut own = Vec::new();
        let atoms = e.atoms_or(&mut own, geo.mol.nat());
        self.0.energy(c, s.qat, atoms, geo.parallel);
        let total = atoms.iter().sum();
        e.pieces[0] = total;
    }

    fn potential(&self, geo: &Geometry<'_>, c: &D4Cache, s: &SccMoments<'_>, v: &mut Potentials) {
        self.0.potential(c, s.qat, &mut v.atom, geo.parallel);
    }

    fn gradient(&self, _geo: &Geometry<'_>, c: &D4Cache, s: &SccMoments<'_>, chain: &mut Chain) {
        self.0.gradient(c, s.qat, &mut chain.g);
    }

    fn kernel(&self, _geo: &Geometry<'_>, c: &D4Cache, s: &SccMoments<'_>, k: &mut Kernel) {
        self.0.charge_hessian(c, s.qat, &mut k.atom);
    }

    fn second_derivatives(
        &self,
        geo: &Geometry<'_>,
        c: &D4Cache,
        s: &SccMoments<'_>,
        dv: Option<&mut PotentialDerivative>,
        h: Option<&mut Chain2<'_>>,
    ) -> Result<()> {
        let nat = geo.mol.nat();
        if let Some(dv) = dv {
            self.0.potential_derivative(c, s.qat, &mut dv.atomic[..3 * nat * nat]);
        }
        if let Some(h) = h {
            self.0.hessian(c, s.qat, h.h, geo.parallel);
        }
        Ok(())
    }
}
