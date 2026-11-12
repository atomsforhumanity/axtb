//! Implicit solvation (ALPB, GBSA; crate `axtb-solvation`) as a charge-dependent SCF term:
//! the GFN2-xTB models of the xtb 6.8.0 and 6.7.1 programs, and the corrected model.

use crate::molecule::Molecule;
use crate::terms::{
    Chain, Chain2, Geometry, Kernel, MomentTerm, Needs, PotentialDerivative, Potentials, SccMoments, TermEnergy,
};
use crate::{Error, Result};
use axtb_solvation::{Conventions as SolvConventions, SolvCache, SolvOptions, Solvation};
pub use axtb_solvation::{Model as SolvationModel, ReferenceState};

/// Which implicit solvation model to add.
#[derive(Clone, Debug, PartialEq)]
pub struct SolvationSpec {
    /// ALPB (`xtb --alpb`) or GBSA (`xtb --gbsa`).
    pub model: SolvationModel,
    /// Solvent name as xtb accepts it (case-insensitive, with xtb's aliases).
    pub solvent: String,
    /// Reference state of the solvation free energy (xtb default `gsolv`).
    pub state: ReferenceState,
}

impl SolvationSpec {
    /// Parse a reference state name (`gsolv`, `bar1M`, `reference`).
    pub fn parse_state(s: &str) -> Result<ReferenceState> {
        match s.to_ascii_lowercase().as_str() {
            "gsolv" => Ok(ReferenceState::Gsolv),
            "bar1m" => Ok(ReferenceState::Bar1M),
            "reference" => Ok(ReferenceState::Reference),
            other => {
                Err(Error::Input(format!("unknown solvation reference state {other:?} (gsolv, bar1M, reference)")))
            }
        }
    }
}

/// The solvation model bound to one molecule.
pub struct SolvationTerm(Solvation);

impl SolvationTerm {
    /// `variant` ([`ConventionParams::solvation`](crate::ConventionParams::solvation)): xtb's
    /// solvation (quirks included) for xtb conventions; the corrected solvation for axtb and
    /// tblite conventions (tblite's own ALPB/GBSA has neither xtb bug and agrees with the
    /// corrected model to ~1e-9 Eh).
    pub fn new(mol: &Molecule, spec: &SolvationSpec, variant: SolvConventions) -> Result<Self> {
        let opts = SolvOptions { conventions: variant, ..SolvOptions::default() };
        Solvation::with_options(&mol.numbers, &spec.solvent, spec.model, spec.state, &opts)
            .map(SolvationTerm)
            .map_err(|e| Error::Input(format!("solvation: {e}")))
    }
}

/// ALPB/GBSA as a moment term: the surface and shift energies are its moment-independent part,
/// the polar and hydrogen-bond energies its charge-dependent part (axtb-solvation).
impl MomentTerm for SolvationTerm {
    type Cache = SolvCache;

    fn names(&self) -> &[&'static str] {
        &["solvation"]
    }

    fn fixed_names(&self) -> &[&'static str] {
        &["solvation"]
    }

    fn needs(&self) -> Needs {
        Needs { atom: true, ..Needs::default() }
    }

    fn partitions_energy(&self) -> bool {
        true
    }

    fn update(&self, geo: &Geometry<'_>) -> Result<SolvCache> {
        Ok(self.0.update(&geo.mol.positions))
    }

    fn fixed_energy(&self, geo: &Geometry<'_>, c: &SolvCache, e: &mut TermEnergy<'_>, chain: Option<&mut Chain>) {
        let mut own = Vec::new();
        let atoms = e.atoms_or(&mut own, geo.mol.nat());
        self.0.fixed_energy(c, atoms, chain.map(|ch| &mut ch.g[..]));
        let total = atoms.iter().sum();
        e.pieces[0] = total;
    }

    fn energy(&self, geo: &Geometry<'_>, c: &SolvCache, s: &SccMoments<'_>, e: &mut TermEnergy<'_>) {
        let mut own = Vec::new();
        let atoms = e.atoms_or(&mut own, geo.mol.nat());
        self.0.energy(c, s.qat, atoms);
        let total = atoms.iter().sum();
        e.pieces[0] = total;
    }

    fn potential(&self, _geo: &Geometry<'_>, c: &SolvCache, s: &SccMoments<'_>, v: &mut Potentials) {
        self.0.potential(c, s.qat, &mut v.atom);
    }

    fn gradient(&self, _geo: &Geometry<'_>, c: &SolvCache, s: &SccMoments<'_>, chain: &mut Chain) {
        self.0.gradient(c, s.qat, &mut chain.g);
    }

    /// The charge-dependent energy is `q^T B q / 2`: its kernel is the Born matrix.
    fn kernel(&self, _geo: &Geometry<'_>, c: &SolvCache, _s: &SccMoments<'_>, k: &mut Kernel) {
        for (x, b) in k.atom.iter_mut().zip(c.born_matrix()) {
            *x += b;
        }
    }

    /// Analytic for the corrected Born integrator (axtb and tblite conventions); xtb's
    /// stale-`r1` integrator (xtb conventions) has no consistent derivatives and takes the
    /// central differences of the trait's default.
    fn second_derivatives(
        &self,
        geo: &Geometry<'_>,
        c: &SolvCache,
        s: &SccMoments<'_>,
        mut dv: Option<&mut PotentialDerivative>,
        mut h: Option<&mut Chain2<'_>>,
    ) -> Result<()> {
        let nat = geo.mol.nat();
        let mut scratch = Vec::new();
        let rows = match dv.as_deref_mut() {
            Some(d) => &mut d.atomic[..3 * nat * nat],
            None => {
                scratch.resize(3 * nat * nat, 0.0);
                &mut scratch[..]
            }
        };
        if !self.0.second_derivatives(c, s.qat, rows, h.as_deref_mut().map(|h| &mut *h.h)) {
            crate::terms::moment_fd_second_derivatives(self, geo, s, dv, h)?;
        }
        Ok(())
    }
}
