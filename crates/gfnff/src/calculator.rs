//! Stateful GFN-FF calculator: the topology is set up once (from the first structure, or
//! from a reference structure) and reused for every later geometry, as xtb does within an
//! optimization.
//!
//! Threading: `threads > 1` creates one rayon pool of exactly that size owned by the
//! calculator; all parallel work (pair loops, EEQ solve, Hessian) runs inside it. With
//! `threads = 1` no thread is spawned. Results repeat bit for bit on a given thread count
//! (see the crate documentation for different thread counts).

use crate::conventions::Conventions;
use crate::energy::{self, Energies, Model};
use crate::error::{Error, Result};
use crate::hessian::{self, Hessian};
use crate::setup::{check_positions, setup, SetupOptions};
use crate::topology::Topology;
use faer::Par;

/// Options of a calculator.
#[derive(Clone, Debug)]
pub struct Options {
    /// Threads of the calculator's own pool (at least 1; 1 = no threads).
    pub threads: usize,
    /// Accuracy of the cutoffs, in (0, 10] (default: xtb's, 0.1; 2.0 above 10000 atoms).
    pub accuracy: Option<f64>,
    /// Charges of the molecular fragments (the bonded molecules of the setup structure) in
    /// the order of their first atoms, as the second line of xtb's `.CHRG` file: one per
    /// fragment, adding up to the total charge, else an input error (xtb ignores extra
    /// charges and zeroes inconsistent ones). Default: the total charge on the fragment of
    /// atom 1, all others neutral.
    pub fragment_charges: Option<Vec<f64>>,
    /// xtb version of the force field (default: 6.8.0).
    pub conventions: Conventions,
}

impl Default for Options {
    fn default() -> Self {
        Options { threads: 1, accuracy: None, fragment_charges: None, conventions: Conventions::default() }
    }
}

/// Results of a single point.
#[derive(Clone, Debug)]
pub struct Results {
    /// Total energy (Eh).
    pub energy: f64,
    /// The energy terms (Eh).
    pub energies: Energies,
    /// Eh/bohr, if requested.
    pub gradient: Option<Vec<[f64; 3]>>,
    /// EEQ charges at this geometry (e).
    pub charges: Vec<f64>,
    /// Dipole moment of the EEQ charges about the origin (e bohr).
    pub dipole: [f64; 3],
    /// Hydrogen bonds of case 1 (H not bonded to A) at this geometry.
    pub nhb1: usize,
    /// Hydrogen bonds of cases 2 and 3 (A-H...B, carbonyl and pyridine-type B) at this geometry.
    pub nhb2: usize,
    /// Halogen, chalcogen and pnictogen bonds at this geometry.
    pub nxb: usize,
    /// Wall time of the steps (s).
    pub timings: Vec<(&'static str, f64)>,
}

/// GFN-FF with a fixed topology.
pub struct Calculator {
    topo: Topology,
    model: Model,
    pool: Option<rayon::ThreadPool>,
    threads: usize,
}

impl Calculator {
    /// Set up the topology from `positions` (bohr) of the atoms `numbers` (H-Lr) with total
    /// `charge` (within ± the nuclear charge) and build the calculator. Positions must be
    /// finite and no two atoms closer than 0.1 bohr (also at every later geometry).
    pub fn new(numbers: &[usize], positions: &[[f64; 3]], charge: f64, options: Options) -> Result<Self> {
        let threads = options.threads;
        let pool = make_pool(threads)?;
        let sopts = SetupOptions {
            accuracy: options.accuracy,
            fragment_charges: options.fragment_charges,
            par: Some(par_of(threads)),
            conventions: options.conventions,
        };
        let topo = match &pool {
            Some(p) => p.install(|| setup(numbers, positions, charge, &sopts))?,
            None => setup(numbers, positions, charge, &sopts)?,
        };
        check_topology(&topo)?;
        let model = Model::new(&topo);
        Ok(Calculator { topo, model, pool, threads })
    }

    /// The topology (set up once, used for every geometry).
    pub fn topology(&self) -> &Topology {
        &self.topo
    }

    /// Threads of the calculator's pool.
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Run `f` with the calculator's parallelism inside its pool.
    fn run<T: Send>(&self, f: impl FnOnce(Option<&rayon::ThreadPool>, Par) -> T + Send) -> T {
        let par = par_of(self.threads);
        match &self.pool {
            Some(p) => p.install(|| f(Some(p), par)),
            None => f(None, par),
        }
    }

    /// Energy (and gradient) at `positions` (bohr) with the calculator's topology.
    pub fn singlepoint(&self, positions: &[[f64; 3]], gradient: bool) -> Result<Results> {
        check_positions(positions, self.topo.natoms())?;
        let ev = self.run(|pool, par| energy::evaluate(&self.topo, &self.model, positions, gradient, pool, par))?;
        let mut dipole = [0.0; 3];
        for (i, q) in ev.charges.iter().enumerate() {
            for c in 0..3 {
                dipole[c] += positions[i][c] * q;
            }
        }
        let terms = ev.energies.named();
        finite("energy", terms.iter().map(|(_, e)| e).chain([&ev.energies.total]))?;
        finite("gradient", ev.gradient.iter().flatten().flatten())?;
        finite("EEQ charge vector", &ev.charges)?;
        Ok(Results {
            energy: ev.energies.total,
            energies: ev.energies,
            gradient: ev.gradient,
            charges: ev.charges,
            dipole,
            nhb1: ev.nhb1,
            nhb2: ev.nhb2,
            nxb: ev.nxb,
            timings: ev.timings,
        })
    }

    /// Analytic Hessian at `positions` (bohr) with the calculator's topology, with dipole
    /// derivatives. A numerical error where it is not defined (a hydrogen bond exactly along a
    /// C=O axis, grimme-lab/xtb#1516), where [`Calculator::numerical_hessian`] is.
    pub fn hessian(&self, positions: &[[f64; 3]]) -> Result<Hessian> {
        check_positions(positions, self.topo.natoms())?;
        let h = self.run(|pool, par| hessian::hessian(&self.topo, &self.model, positions, pool, par))?;
        check_hessian(&h)?;
        Ok(h)
    }

    /// Hessian by central differences (`step` bohr; xtb uses 0.005) of the analytic gradient
    /// with the calculator's topology, symmetrized, with dipole derivatives from the
    /// displaced structures' EEQ dipoles.
    pub fn numerical_hessian(&self, positions: &[[f64; 3]], step: f64) -> Result<Hessian> {
        check_positions(positions, self.topo.natoms())?;
        if !(step.is_finite() && step > 0.0) {
            return Err(Error::Input(format!("the finite-difference step must be positive, not {step}")));
        }
        let h =
            self.run(|pool, par| hessian::numerical_hessian(&self.topo, &self.model, positions, step, pool, par))?;
        check_hessian(&h)?;
        Ok(h)
    }
}

fn make_pool(threads: usize) -> Result<Option<rayon::ThreadPool>> {
    match threads {
        0 => Err(Error::Input("threads must be at least 1".into())),
        1 => Ok(None),
        _ => rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map(Some)
            .map_err(|e| Error::Input(format!("cannot create thread pool: {e}"))),
    }
}

fn par_of(threads: usize) -> Par {
    if threads > 1 {
        Par::rayon(threads)
    } else {
        Par::Seq
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

fn check_hessian(h: &Hessian) -> Result<()> {
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
