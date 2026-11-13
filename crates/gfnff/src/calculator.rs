// SPDX-License-Identifier: LGPL-3.0-or-later
//! Stateful GFN-FF calculator: the topology is set up once (from the first structure, or
//! from a reference structure) and reused for every later geometry, as xtb does within an
//! optimization. It implements [`Method`]: energies, gradients and EEQ charges
//! ([`Method::singlepoint`]), analytic and numerical Hessians ([`Method::hessian`]).
//!
//! Threading: `threads > 1` creates one rayon pool of exactly that size owned by the
//! calculator ([`Threads`]); all parallel work (pair loops, EEQ solve, Hessian) runs inside
//! it. With `threads = 1` no thread is spawned. Results repeat bit for bit on a given thread
//! count (see the crate documentation for different thread counts).

use crate::conventions::Conventions;
use crate::energy::{self, Model};
use crate::error::{Error, Result};
use crate::hessian::{self, FfHessian};
use crate::setup::{check_positions, setup, SetupOptions};
use crate::topology::Topology;
use axtb_base::method::{
    AnalysisConventions, AnalysisDefaults, Electrons, Extras, Fingerprint, GfnffReport, GfnffTopology, Hessian,
    HessianDefault, HessianDefaults, HessianMethod, HessianOptions, Method, MethodInfo, SinglePoint, Term, Want,
    Wavefunction,
};
use axtb_base::threads::Threads;

/// Options of a calculator.
#[derive(Clone, Debug)]
pub struct Options {
    /// Threads of the calculator's own pool (at least 1; 1 = no threads).
    pub threads: usize,
    /// Accuracy of the cutoffs (xtb's GFN-FF `acc`), in (0, 10] (default: xtb's, 0.1; 2.0
    /// above 10000 atoms). Not an SCF accuracy: GFN-FF has no SCF.
    pub cutoff_accuracy: Option<f64>,
    /// Charges of the molecular fragments (the bonded molecules of the setup structure) in
    /// the order of their first atoms, as the second line of xtb's `.CHRG` file: one per
    /// fragment, adding up to the total charge, else an input error (xtb ignores extra
    /// charges and zeroes inconsistent ones). Default: the total charge on the fragment of
    /// atom 1, all others neutral.
    pub fragment_charges: Option<Vec<f64>>,
    /// The conventions (default: axtb's, the force field of xtb 6.8.0).
    pub conventions: Conventions,
}

impl Default for Options {
    fn default() -> Self {
        Options { threads: 1, cutoff_accuracy: None, fragment_charges: None, conventions: Conventions::default() }
    }
}

/// The step (bohr) of GFN-FF's numerical Hessian, xtb's `--hess` step.
pub const NUMERICAL_STEP: f64 = 0.005;

/// GFN-FF with a fixed topology.
pub struct Calculator {
    topo: Topology,
    model: Model,
    threads: Threads,
    info: MethodInfo,
    fingerprint: Fingerprint,
}

impl Calculator {
    /// Set up the topology from `positions` (bohr) of the atoms `numbers` (H-Lr) with total
    /// `charge` (within ± the nuclear charge) and build the calculator. Positions must be
    /// finite and no two atoms closer than 0.1 bohr (also at every later geometry).
    pub fn new(numbers: &[usize], positions: &[[f64; 3]], charge: f64, options: Options) -> Result<Self> {
        let threads = Threads::new(options.threads)?;
        let conventions = options.conventions;
        let sopts = SetupOptions {
            accuracy: options.cutoff_accuracy,
            fragment_charges: options.fragment_charges,
            par: Some(threads.par()),
            conventions,
        };
        let topo = threads.install(|| setup(numbers, positions, charge, &sopts))?;
        check_topology(&topo)?;
        let model = Model::new(&topo);
        let info = info(conventions);
        let fingerprint = Fingerprint::new(info.name, numbers, charge, None);
        Ok(Calculator { topo, model, threads, info, fingerprint })
    }

    /// The topology (set up once, used for every geometry).
    pub fn topology(&self) -> &Topology {
        &self.topo
    }

    /// Threads of the calculator's pool.
    pub fn threads(&self) -> usize {
        self.threads.count()
    }

    /// The topology as the single point reports it.
    fn topology_report(&self) -> GfnffTopology {
        let t = &self.topo;
        GfnffTopology {
            bonds: t.bond_pairs().iter().map(|&(i, j)| [i, j]).collect(),
            pi_bond_orders: t.bonds.iter().map(|b| b.pibo).collect(),
            bond_types: t.bonds.iter().map(|b| b.btyp).collect(),
            hybridization: t.hyb.clone(),
            pi_atoms: t.pi.clone(),
            fragments: t.fragments(),
            fragment_charges: t.qfrag.clone(),
            topology_charges: t.qa.clone(),
            n_angles: t.angles.len(),
            n_torsions: t.torsions.len(),
        }
    }
}

/// What GFN-FF is and offers in `conventions`.
fn info(conventions: Conventions) -> MethodInfo {
    MethodInfo {
        name: crate::METHOD,
        conventions: conventions.label(),
        electrons: Electrons::None,
        hessian: HessianDefaults {
            default: HessianMethod::Analytic,
            offered: vec![
                HessianDefault { method: HessianMethod::Analytic, step: None, accuracy: None },
                HessianDefault { method: HessianMethod::Numerical, step: Some(NUMERICAL_STEP), accuracy: None },
            ],
        },
        analysis: AnalysisDefaults {
            conventions: match conventions {
                Conventions::Axtb => AnalysisConventions::Axtb,
                Conventions::Xtb680 => AnalysisConventions::Xtb,
                Conventions::Xtb671 => AnalysisConventions::Xtb671,
            },
            // xtb scales GFN-FF's frequencies by 1.03 in its thermochemistry
            xtb_frequency_scale: Some(1.03),
        },
        quadrupole: None,
    }
}

impl Method for Calculator {
    fn info(&self) -> &MethodInfo {
        &self.info
    }

    /// GFN-FF has no orbitals and no SCF: no Molden file or bond orders, no response Hessian, no
    /// Hessian accuracy.
    fn check(&self, want: &Want, hessian: Option<&HessianOptions>) -> Result<()> {
        if let Some(h) = hessian {
            if h.method == Some(HessianMethod::Response) {
                return Err(Error::Input("gfnff Hessians are analytic or numerical (no SCF response)".into()));
            }
            if h.accuracy.is_some() {
                return Err(Error::Input("gfnff Hessians take no hessian_accuracy (no SCF)".into()));
            }
        }
        if want.orbitals || want.bond_orders {
            return Err(Error::Input("gfnff has no orbitals (no Molden file, no bond orders; see topology)".into()));
        }
        Ok(())
    }

    /// Energy (and gradient) at `positions` (bohr) with the calculator's topology; the EEQ
    /// charges and their dipole; the topology and this geometry's hydrogen and halogen bonds in
    /// [`Extras::gfnff`].
    fn singlepoint(&mut self, positions: &[[f64; 3]], want: &Want) -> Result<SinglePoint> {
        self.check(want, None)?;
        check_positions(positions, self.topo.natoms())?;
        let ev = self
            .threads
            .install(|| energy::evaluate(&self.topo, &self.model, positions, want.gradient, &self.threads))?;
        let mut dipole = [0.0; 3];
        for (i, q) in ev.charges.iter().enumerate() {
            for c in 0..3 {
                dipole[c] += positions[i][c] * q;
            }
        }
        let terms: Vec<Term> = ev.energies.named().into_iter().map(|(n, v)| Term::piece(n, v)).collect();
        finite("energy", terms.iter().map(|t| &t.value).chain([&ev.energies.total]))?;
        finite("gradient", ev.gradient.iter().flatten().flatten())?;
        finite("EEQ charge vector", &ev.charges)?;
        let report = GfnffReport { n_hbonds: [ev.nhb1, ev.nhb2], n_xbonds: ev.nxb, topology: self.topology_report() };
        Ok(SinglePoint {
            energy: ev.energies.total,
            terms,
            gradient: ev.gradient,
            charges: ev.charges,
            dipole,
            electronic: None,
            extras: Extras { gfnff: Some(report), ..Extras::default() },
            timings: ev.timings,
        })
    }

    /// The analytic Hessian (default; a numerical error where it is not defined, a hydrogen
    /// bond exactly along a C=O axis, grimme-lab/xtb#1516) or central differences of the
    /// analytic gradient (`step` bohr, default [`NUMERICAL_STEP`]), symmetrized, with dipole
    /// derivatives of the EEQ dipole.
    fn hessian(&mut self, positions: &[[f64; 3]], opts: &HessianOptions) -> Result<Hessian> {
        self.check(&Want::ENERGY, Some(opts))?;
        check_positions(positions, self.topo.natoms())?;
        let method = opts.method.unwrap_or(HessianMethod::Analytic);
        let h = match method {
            HessianMethod::Numerical => {
                let step = opts.step.unwrap_or(NUMERICAL_STEP);
                if !(step.is_finite() && step > 0.0) {
                    return Err(Error::Input(format!("the finite-difference step must be positive, not {step}")));
                }
                self.threads
                    .install(|| hessian::numerical_hessian(&self.topo, &self.model, positions, step, &self.threads))?
            }
            _ => self.threads.install(|| hessian::hessian(&self.topo, &self.model, positions, &self.threads))?,
        };
        check_hessian(&h)?;
        Ok(Hessian {
            energy: h.energy,
            gradient: h.gradient,
            hessian: h.hessian,
            max_asymmetry: h.max_asymmetry,
            dipole_derivatives: Some(h.dipole_derivatives),
            method,
            timings: h.timings,
            parts: Vec::new(),
        })
    }

    /// GFN-FF has no SCF: every wavefunction is refused.
    fn set_wavefunction(&mut self, wf: &Wavefunction) -> Result<()> {
        wf.check(&self.fingerprint)?;
        Err(Error::Input("gfnff has no SCF; starting moments are GFN2-only".into()))
    }
}

/// A numerical error unless every value is finite: no NaN or infinity leaves the crate.
fn finite<'a>(what: &str, values: impl IntoIterator<Item = &'a f64>) -> Result<()> {
    if values.into_iter().all(|v| v.is_finite()) {
        Ok(())
    } else {
        Err(Error::Numerical(format!("the {what} is not finite at this geometry")))
    }
}

fn check_hessian(h: &FfHessian) -> Result<()> {
    finite("energy", [&h.energy])?;
    finite("gradient", h.gradient.iter().flatten())?;
    finite("Hessian", &h.hessian)?;
    finite("dipole derivative", h.dipole_derivatives.iter().flatten())
}

/// The real parameters of a topology are finite.
fn check_topology(t: &Topology) -> Result<()> {
    let ok = [&t.qa, &t.qfrag, &t.chieeq, &t.gameeq, &t.alpeeq, &t.rep_f, &t.zeta, &t.hbbas, &t.hbaci]
        .iter()
        .all(|v| v.iter().all(|x| x.is_finite()))
        && t.bonds.iter().all(|b| b.pibo.is_finite() && b.vbond.iter().all(|x| x.is_finite()))
        && t.angles.iter().all(|a| a.phi0.is_finite() && a.fc.is_finite())
        && t.torsions.iter().all(|q| q.phi0.is_finite() && q.fc.is_finite());
    if ok {
        Ok(())
    } else {
        Err(Error::Numerical("the topology setup gave non-finite force-field parameters".into()))
    }
}
