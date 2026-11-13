// SPDX-License-Identifier: LGPL-3.0-or-later
//! Terms without their types, as a model holds the terms added to it: a term behind an `Arc`,
//! and a term bound to one geometry, which owns the term's typed cache. Crate-private: users
//! write [`MomentTerm`]s and hand them to a model (`Calculator::push_term`).

use super::{Chain, Chain2, Geometry, Kernel, MomentTerm, PotentialDerivative, Potentials, SccMoments, TermEnergy};
use crate::Result;
use std::sync::Arc;

/// A moment term without its type.
pub(crate) trait AnyMomentTerm: Send + Sync {
    /// The term bound to the geometry `geo` (its cache built).
    fn bind(self: Arc<Self>, geo: &Geometry<'_>) -> Result<Box<dyn BoundMoment>>;
}

impl<T: MomentTerm + 'static> AnyMomentTerm for T
where
    T::Cache: 'static,
{
    fn bind(self: Arc<Self>, geo: &Geometry<'_>) -> Result<Box<dyn BoundMoment>> {
        let cache = self.update(geo)?;
        Ok(Box::new(Bound { term: self, cache }))
    }
}

/// A moment term bound to one geometry: the methods of [`MomentTerm`] with its cache.
pub(crate) trait BoundMoment: Send + Sync {
    fn names(&self) -> &[&'static str];
    fn fixed_names(&self) -> &[&'static str];
    fn partitions_energy(&self) -> bool;
    fn fixed_energy(&self, geo: &Geometry<'_>, e: &mut TermEnergy<'_>, chain: Option<&mut Chain>);
    fn energy(&self, geo: &Geometry<'_>, s: &SccMoments<'_>, e: &mut TermEnergy<'_>);
    fn potential(&self, geo: &Geometry<'_>, s: &SccMoments<'_>, v: &mut Potentials);
    fn gradient(&self, geo: &Geometry<'_>, s: &SccMoments<'_>, chain: &mut Chain);
    fn kernel(&self, geo: &Geometry<'_>, s: &SccMoments<'_>, k: &mut Kernel);
    fn second_derivatives(
        &self,
        geo: &Geometry<'_>,
        s: &SccMoments<'_>,
        dv: Option<&mut PotentialDerivative>,
        h: Option<&mut Chain2<'_>>,
    ) -> Result<()>;
}

/// A term and its cache at one geometry.
struct Bound<T: MomentTerm> {
    term: Arc<T>,
    cache: T::Cache,
}

impl<T: MomentTerm> BoundMoment for Bound<T> {
    fn names(&self) -> &[&'static str] {
        self.term.names()
    }

    fn fixed_names(&self) -> &[&'static str] {
        self.term.fixed_names()
    }

    fn partitions_energy(&self) -> bool {
        self.term.partitions_energy()
    }

    fn fixed_energy(&self, geo: &Geometry<'_>, e: &mut TermEnergy<'_>, chain: Option<&mut Chain>) {
        self.term.fixed_energy(geo, &self.cache, e, chain)
    }

    fn energy(&self, geo: &Geometry<'_>, s: &SccMoments<'_>, e: &mut TermEnergy<'_>) {
        self.term.energy(geo, &self.cache, s, e)
    }

    fn potential(&self, geo: &Geometry<'_>, s: &SccMoments<'_>, v: &mut Potentials) {
        self.term.potential(geo, &self.cache, s, v)
    }

    fn gradient(&self, geo: &Geometry<'_>, s: &SccMoments<'_>, chain: &mut Chain) {
        self.term.gradient(geo, &self.cache, s, chain)
    }

    fn kernel(&self, geo: &Geometry<'_>, s: &SccMoments<'_>, k: &mut Kernel) {
        self.term.kernel(geo, &self.cache, s, k)
    }

    fn second_derivatives(
        &self,
        geo: &Geometry<'_>,
        s: &SccMoments<'_>,
        dv: Option<&mut PotentialDerivative>,
        h: Option<&mut Chain2<'_>>,
    ) -> Result<()> {
        self.term.second_derivatives(geo, &self.cache, s, dv, h)
    }
}
