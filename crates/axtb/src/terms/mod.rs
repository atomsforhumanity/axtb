//! Energy terms of a tight-binding model, written once against a common interface.
//!
//! **Unstable**: these traits and types may change in any release before 1.0.
//!
//! A tight-binding model is a basis, a core Hamiltonian, an SCF strategy and a list of energy
//! terms. A term depends either on the geometry alone ([`GeometryTerm`]: the repulsion of
//! GFN2-xTB, dispersion without charges) or also on the SCC moments of the density
//! ([`MomentTerm`]: the Mulliken shell and atomic charges, atomic dipoles and quadrupoles, shell
//! magnetizations; electrostatics, self-consistent dispersion, implicit solvation). A moment
//! term may have a geometry-only part sharing its cache (dftd4's three-body term at zero
//! charges, the surface terms of a solvation model).
//!
//! ## Caches and binding
//!
//! A term is built for one molecule and holds what does not change with the geometry. What
//! does is its cache (`type Cache`), built by `update` once per geometry and passed back to
//! every method at that geometry. A model keeps the cache next to the term, so the term itself
//! is shared and immutable (`Send + Sync`): parallel displacements of a finite difference each
//! build their own cache.
//!
//! ## Sinks
//!
//! Terms write into sinks ([`TermEnergy`], [`Potentials`], [`Kernel`], [`PotentialDerivative`],
//! [`Chain`], [`Chain2`]) by adding, never by assigning. The model sizes and zeroes them and
//! decides which terms share one and in which order the terms are called: the order of every
//! sum is the model's, so that a model can reproduce a reference program's arithmetic bit for
//! bit. GFN2-xTB sums its energies per atom as xtb and tblite do ([`TermEnergy::atoms`], for
//! terms that [partition](MomentTerm::partitions_energy) their energy); g-xTB sums totals.
//! Terms added to a model (`Calculator::push_term` for GFN2-xTB) come after the model's own
//! terms in every sum.
//!
//! ## Intermediates and the chain rule
//!
//! Quantities that several terms differentiate through (coordination numbers, model charges)
//! are [`Intermediate`]s registered in the model's [`ChainLayout`]; their values at a geometry
//! are in [`Geometry::values`]. A term that depends on one is built with its [`Slot`] (found
//! with [`ChainLayout::find`], e.g. [`Gfn2Cn::NAME`]), reads its values and adds dE/dX to
//! [`Chain::slot`] in its gradient; [`Chain::finish`] applies every intermediate's chain rule
//! once at the end. Second derivatives get the intermediates' Jacobians in [`Chain2`].
//!
//! ## Defaults
//!
//! Only the energy, the potential and the explicit gradient are required. The kernel
//! d²E/ds ds, its product with a vector, and the second derivatives with respect to the nuclear
//! positions default to central differences of the analytic potential and gradient (the
//! response and analytic Hessians of GFN2-xTB use them for terms without analytic second
//! derivatives).

mod chain;
mod fd;
mod sinks;
#[cfg(test)]
mod tests;

pub use crate::ncoord::Gfn2Cn;
pub use chain::{Chain, Chain2, ChainLayout, Intermediate, Jacobians, Slot, Values};
pub use sinks::{Geometry, Kernel, Needs, PotentialDerivative, Potentials, SccMoments, TermEnergy};

use crate::Result;

/// An energy term that depends on the nuclear positions alone.
pub trait GeometryTerm: Send + Sync {
    /// What the term keeps of one geometry.
    type Cache: Send + Sync;

    /// Names of its energy pieces, in the order of [`TermEnergy::pieces`].
    fn names(&self) -> &[&'static str];

    /// Whether [`energy`](Self::energy) partitions the energy over the atoms
    /// ([`TermEnergy::atoms`]); false by default.
    fn partitions_energy(&self) -> bool {
        false
    }

    /// The intermediates it reads or differentiates through (empty by default; the finite
    /// differences of the defaults recompute their values only when this is not empty).
    fn slots(&self) -> &[Slot] {
        &[]
    }

    /// The cache of the geometry `geo`.
    fn update(&self, geo: &Geometry<'_>) -> Result<Self::Cache>;

    /// The energy and, with `chain`, its gradient (positions and intermediates).
    fn energy(&self, geo: &Geometry<'_>, c: &Self::Cache, e: &mut TermEnergy<'_>, chain: Option<&mut Chain>);

    /// Second derivatives with respect to the nuclear positions, added to `h.h`. The default
    /// takes central differences of the gradient of [`energy`](Self::energy), one cache per
    /// displaced geometry, parallel over displacements with `geo.parallel`.
    fn add_hessian(&self, geo: &Geometry<'_>, c: &Self::Cache, h: &mut Chain2<'_>) -> Result<()> {
        let _ = c;
        fd::geometry_hessian(self, geo, h)
    }
}

/// An energy term that depends on the SCC moments (and the positions).
pub trait MomentTerm: Send + Sync {
    /// What the term keeps of one geometry.
    type Cache: Send + Sync;

    /// Names of the pieces of its moment-dependent energy, in the order of
    /// [`TermEnergy::pieces`].
    fn names(&self) -> &[&'static str];

    /// Names of the pieces of its moment-independent energy ([`fixed_energy`](Self::fixed_energy));
    /// none by default.
    fn fixed_names(&self) -> &[&'static str] {
        &[]
    }

    /// The SCC moments it reads.
    fn needs(&self) -> Needs;

    /// Whether [`energy`](Self::energy) and [`fixed_energy`](Self::fixed_energy) partition the
    /// energy over the atoms ([`TermEnergy::atoms`]); false by default. A model that sums per
    /// atom adds the pieces of a term that does not after the atomic sum.
    fn partitions_energy(&self) -> bool {
        false
    }

    /// The intermediates it reads or differentiates through (empty by default; the finite
    /// differences of the defaults recompute their values only when this is not empty).
    fn slots(&self) -> &[Slot] {
        &[]
    }

    /// The cache of the geometry `geo`.
    fn update(&self, geo: &Geometry<'_>) -> Result<Self::Cache>;

    /// The moment-independent part of the energy and, with `chain`, its gradient; none by
    /// default.
    fn fixed_energy(&self, geo: &Geometry<'_>, c: &Self::Cache, e: &mut TermEnergy<'_>, chain: Option<&mut Chain>) {
        let _ = (geo, c, e, chain);
    }

    /// The moment-dependent energy at the moments `s`.
    fn energy(&self, geo: &Geometry<'_>, c: &Self::Cache, s: &SccMoments<'_>, e: &mut TermEnergy<'_>);

    /// The potentials dE/ds at the moments `s`, added to `v`.
    fn potential(&self, geo: &Geometry<'_>, c: &Self::Cache, s: &SccMoments<'_>, v: &mut Potentials);

    /// [`energy`](Self::energy) and [`potential`](Self::potential) together, for terms that
    /// compute both in one pass; the default calls the two.
    fn energy_and_potential(
        &self,
        geo: &Geometry<'_>,
        c: &Self::Cache,
        s: &SccMoments<'_>,
        e: &mut TermEnergy<'_>,
        v: &mut Potentials,
    ) {
        self.energy(geo, c, s, e);
        self.potential(geo, c, s, v);
    }

    /// The gradient of the moment-dependent energy at fixed moments `s`: positions to
    /// `chain.g`, intermediates to their slots.
    fn gradient(&self, geo: &Geometry<'_>, c: &Self::Cache, s: &SccMoments<'_>, chain: &mut Chain);

    /// The second derivatives d²E/ds ds at `s`, added to the blocks of `k` the model sized. The
    /// default takes central differences of [`potential`](Self::potential) for every variable
    /// the term [needs](Self::needs), each kind on its own (shell charges into `k.shell`, atomic
    /// charges, dipoles and quadrupoles into `k.multipole`, or atomic charges alone into
    /// `k.atom`, magnetizations into `k.magnetization`); a term that couples shell charges with
    /// atomic moments implements it.
    fn kernel(&self, geo: &Geometry<'_>, c: &Self::Cache, s: &SccMoments<'_>, k: &mut Kernel) {
        fd::kernel(self, geo, c, s, k)
    }

    /// The change of the potentials for a change `ds` of the moments at `s` (the kernel times
    /// `ds`), added to `dv`. The default takes a central difference of
    /// [`potential`](Self::potential) along `ds`.
    fn kernel_apply(
        &self,
        geo: &Geometry<'_>,
        c: &Self::Cache,
        s: &SccMoments<'_>,
        ds: &SccMoments<'_>,
        dv: &mut Potentials,
    ) {
        fd::kernel_apply(self, geo, c, s, ds, dv)
    }

    /// Second derivatives involving the nuclear positions at fixed moments `s`: with `dv`, the
    /// derivatives of the potentials (added to the rows of the moments it needs); with `h`, the
    /// Hessian of the whole term, its moment-independent part included (added to `h.h`). The
    /// default takes central differences of [`potential`](Self::potential) and of the gradient
    /// ([`fixed_energy`](Self::fixed_energy) and [`gradient`](Self::gradient), with the chain
    /// rules of the term's intermediates), one cache per displaced geometry, parallel over
    /// displacements with `geo.parallel`.
    fn second_derivatives(
        &self,
        geo: &Geometry<'_>,
        c: &Self::Cache,
        s: &SccMoments<'_>,
        dv: Option<&mut PotentialDerivative>,
        h: Option<&mut Chain2<'_>>,
    ) -> Result<()> {
        let _ = c;
        fd::second_derivatives(self, geo, s, dv, h)
    }
}
