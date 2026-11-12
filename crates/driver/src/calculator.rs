//! A method bound to one molecule: dispatch to GFN2-xTB, g-xTB (and its solvation composite)
//! and GFN-FF, and the response of each.

use crate::error::{Error, Result};
use crate::names::{self, Method};
use crate::request::Request;
use crate::response::{self, Attempt, Gfnff, HessianPart, Matrix, Named, Response, Scf, Topology, Vibrations};
use crate::task::Task;
use axtb::method::{Method as _, Want};
use axtb::vibrations::Conventions as VibConventions;
use axtb::ToMolden;
use std::time::Instant;

/// The calculator of a method.
enum Engine {
    Gfn2(Box<axtb::Calculator>),
    #[cfg(feature = "gxtb")]
    Gxtb(Box<axtb_gxtb::Calculator>),
    /// g-xTB in implicit solvent (the solvation composite).
    #[cfg(feature = "gxtb")]
    Composite(Box<axtb_gxtb::CompositeCalculator>),
    Gfnff(Box<axtb_gfnff::Calculator>),
}

/// A converged wavefunction of one calculator, to start another calculator's SCF from it
/// ([`Calculator::guess`], [`Calculator::set_guess`]). Opaque.
#[derive(Clone)]
pub struct ScfState {
    inner: StateInner,
    method: Method,
    natoms: usize,
}

#[derive(Clone)]
enum StateInner {
    Gfn2(axtb::ScfGuess),
    #[cfg(feature = "gxtb")]
    Gxtb(axtb_gxtb::GuessState),
    /// g-xTB part, GFN2 gas phase, GFN2 solvated
    #[cfg(feature = "gxtb")]
    Composite(Option<axtb_gxtb::GuessState>, Option<axtb::ScfGuess>, Option<axtb::ScfGuess>),
}

impl ScfState {
    /// The method of the calculation the wavefunction belongs to.
    pub fn method(&self) -> Method {
        self.method
    }

    /// Atoms of the molecule.
    pub fn natoms(&self) -> usize {
        self.natoms
    }
}

/// A method bound to one molecule (atomic numbers, charge, unpaired electrons) with its
/// settings, built from a [`Request`]: [`Calculator::compute`] runs a [`Task`] at any
/// geometry and starts each SCF from the previous geometry's wavefunction (as the method's
/// conventions or the request's `guess` say).
pub struct Calculator {
    engine: Engine,
    method: Method,
    conventions: &'static str,
    numbers: Vec<usize>,
    threads: usize,
    /// GFN2-xTB with `guess` the method's own start: every SCF starts from it.
    fresh: bool,
    /// The analysis conventions of the calculation's conventions.
    analysis: VibConventions,
}

fn input(msg: impl Into<String>) -> Error {
    Error::Input(msg.into())
}

/// Refuse the settings in `bad` for `method`.
fn refuse(method: Method, bad: &[&str]) -> Result<()> {
    if bad.is_empty() {
        Ok(())
    } else {
        Err(input(format!("{method} does not support {}", bad.join(", "))))
    }
}

/// The implicit-solvent setting of a request, if any.
fn solvation(req: &Request) -> Result<Option<axtb::SolvationSpec>> {
    let state = req.solvation_state.as_deref().map(names::solvation_state).transpose()?;
    let (model, solvent) = match (&req.alpb, &req.gbsa) {
        (Some(_), Some(_)) => return Err(input("give either alpb or gbsa, not both")),
        (Some(s), None) => (axtb::SolvationModel::Alpb, s),
        (None, Some(s)) => (axtb::SolvationModel::Gbsa, s),
        (None, None) => {
            if state.is_some() {
                return Err(input("solvation_state is a setting of the implicit solvent (alpb or gbsa)"));
            }
            return Ok(None);
        }
    };
    Ok(Some(axtb::SolvationSpec { model, solvent: solvent.clone(), state: state.unwrap_or_default() }))
}

/// The hint for a g-xTB method given an implicit solvent alone: g-xTB has no solvent
/// parameters; the solvation composite is the way.
pub fn gxtb_solvent_hint(method: Method) -> String {
    format!(
        "{method} has no implicit-solvent parameters (the g-xTB program stops with --alpb/--gbsa). For g-xTB in \
         solvent use the solvation composite E = E(g-xTB) + [E(GFN2-xTB + ALPB/GBSA) - E(GFN2-xTB)]: add \
         solvation_correction \"gfn2\" (CLI: --solvation-correction gfn2) to the solvent (alpb or gbsa)"
    )
}

impl Calculator {
    /// Set up the method of `req` for its molecule at `req.positions` (bohr): its settings are
    /// checked (unknown names, settings of other methods and impossible values are
    /// [`Error::Input`]); the fields of what to compute are not read (see
    /// [`Task::from_request`]).
    pub fn new(req: &Request) -> Result<Self> {
        let method = Method::parse(req.method.as_deref().unwrap_or("gfn2"))?;
        let nat = req.numbers.len();
        if req.positions.len() != nat {
            return Err(input(format!("positions must have one row per atom ({nat}), not {}", req.positions.len())));
        }
        if req.uhf.is_some_and(|u| u < 0) {
            return Err(input(format!("uhf (unpaired electrons) must not be negative, not {}", req.uhf.unwrap())));
        }
        if req.threads == Some(0) {
            return Err(input("threads must be at least 1"));
        }
        if req.max_iter == Some(0) {
            return Err(input("max_iter must be at least 1"));
        }
        if method != Method::Gfnff && (req.fragment_charges.is_some() || req.topology_positions.is_some()) {
            return Err(input("fragment_charges and topology_positions are GFN-FF settings"));
        }
        if !method.is_gxtb() && (req.solvation_correction.is_some() || req.correction_conventions.is_some()) {
            return Err(input(
                "solvation_correction and correction_conventions are settings of g-xTB in solvent; GFN2-xTB has its \
                 own implicit solvent (alpb or gbsa alone)",
            ));
        }
        let threads = req.threads.unwrap_or(1);
        let uhf = req.uhf.map(|u| u as usize);
        let guess = req.guess.as_deref().map(names::guess).transpose()?;
        let (engine, conventions, analysis, fresh) = match method {
            Method::Gfn2 => {
                let conventions =
                    req.conventions.as_deref().map(names::gfn2_conventions).transpose()?.unwrap_or_default();
                let mut opts = axtb::Options::default();
                if let Some(a) = req.accuracy {
                    opts.scf.accuracy = a;
                }
                if let Some(t) = req.etemp {
                    opts.scf.etemp = t;
                }
                if let Some(m) = req.max_iter {
                    opts.scf.max_iter = m;
                }
                opts.scf.mixer = req.mixer.as_deref().map(names::mixer).transpose()?;
                match guess {
                    Some(names::Guess::Sad) => opts.scf.guess = axtb::InitialGuess::Sad,
                    Some(names::Guess::EeqBc) => {
                        return Err(input(
                            "GFN2-xTB starts from EEQ charges or neutral atoms: guess \"eeq\" or \"sad\"",
                        ))
                    }
                    _ => {}
                }
                opts.scf.threads = threads;
                opts.scf.trace = req.scf_trace;
                opts.dispersion = req.dispersion.unwrap_or(true);
                opts.conventions = conventions;
                opts.solvation = solvation(req)?;
                let calc = axtb::Calculator::new(
                    axtb::Hamiltonian::Gfn2,
                    &req.numbers,
                    &req.positions,
                    req.charge,
                    uhf,
                    opts,
                )?;
                let fresh = matches!(guess, Some(names::Guess::Fresh | names::Guess::Eeq | names::Guess::Sad));
                (Engine::Gfn2(Box::new(calc)), conventions.label(), conventions.into(), fresh)
            }
            Method::Gfnff => {
                let mut bad = Vec::new();
                if req.alpb.is_some() || req.gbsa.is_some() || req.solvation_state.is_some() {
                    bad.push("implicit solvation");
                }
                if req.dispersion == Some(false) {
                    bad.push("switching off dispersion");
                }
                if req.accuracy.is_some()
                    || req.etemp.is_some()
                    || req.max_iter.is_some()
                    || req.mixer.is_some()
                    || req.guess.is_some()
                    || req.scf_trace
                {
                    bad.push("SCF settings (accuracy, etemp, max_iter, mixer, guess, scf_trace)");
                }
                refuse(method, &bad)?;
                let (ff, analysis, label) = names::gfnff_conventions(req.conventions.as_deref().unwrap_or("axtb"))?;
                let setup = match &req.topology_positions {
                    Some(t) if t.len() != nat => {
                        return Err(input(format!("topology_positions must have one row per atom ({nat})")))
                    }
                    Some(t) => t.as_slice(),
                    None => &req.positions,
                };
                let opts = axtb_gfnff::Options {
                    threads,
                    fragment_charges: req.fragment_charges.clone(),
                    conventions: ff,
                    ..Default::default()
                };
                let calc = axtb_gfnff::Calculator::new(&req.numbers, setup, req.charge, opts)?;
                (Engine::Gfnff(Box::new(calc)), label, analysis, false)
            }
            Method::GxtbV1_1 | Method::GxtbV2_0_1 => {
                #[cfg(feature = "gxtb")]
                {
                    let (engine, conventions) = gxtb_engine(req, method, threads, uhf, guess)?;
                    // the g-xTB 2.0.1 program is a 6.7.1-based xtb build
                    let analysis = match conventions {
                        axtb_gxtb::Conventions::Gxtb => VibConventions::Xtb671,
                        axtb_gxtb::Conventions::Axtb => VibConventions::Axtb,
                    };
                    (engine, names::gxtb_conventions_label(conventions), analysis, false)
                }
                #[cfg(not(feature = "gxtb"))]
                {
                    let _ = (uhf, guess);
                    return Err(input(format!("{method}: this axtb was built without g-xTB (feature `gxtb`)")));
                }
            }
        };
        Ok(Self { engine, method, conventions, numbers: req.numbers.clone(), threads, fresh, analysis })
    }

    /// The method.
    pub fn method(&self) -> Method {
        self.method
    }

    /// The canonical name of the method's conventions.
    pub fn conventions(&self) -> &'static str {
        self.conventions
    }

    /// Atomic numbers of the molecule.
    pub fn numbers(&self) -> &[usize] {
        &self.numbers
    }

    /// Threads of the calculator.
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Run `task` at `positions` (bohr). With a Hessian, the response is that of the single
    /// point at `positions` (energy, gradient, charges, orbitals; g-xTB's SCF from its own
    /// start, as without the Hessian) plus the Hessian, its dipole derivatives and its
    /// vibrational analysis.
    pub fn compute(&mut self, positions: &[[f64; 3]], task: &Task) -> Result<Response> {
        let t0 = Instant::now();
        let nat = self.numbers.len();
        if positions.len() != nat {
            return Err(input(format!("positions must have one row per atom ({nat}), not {}", positions.len())));
        }
        if self.fresh {
            self.reset_guess(false);
        }
        let mut out = Response {
            program: response::PROGRAM,
            version: response::VERSION,
            method: Some(self.method.name()),
            conventions: self.conventions,
            ..Default::default()
        };
        let molden = task.molden.is_some();
        let orbitals = match &mut self.engine {
            Engine::Gfn2(calc) => {
                let hess = match &task.hessian {
                    Some(h) => {
                        // the orbitals and bond orders of the single point only, not of every
                        // displacement
                        calc.options.scf.keep_orbitals = false;
                        calc.options.scf.bond_orders = false;
                        let o = axtb::HessianOptions {
                            method: Some(h.method.unwrap_or_default()),
                            step: h.step,
                            accuracy: h.accuracy,
                        };
                        Some(calc.hessian(positions, &o)?)
                    }
                    None => None,
                };
                calc.options.scf.keep_orbitals = molden;
                calc.options.scf.bond_orders = task.bond_orders;
                let sp = calc.singlepoint(positions, task.gradient || hess.is_some());
                calc.options.scf.keep_orbitals = false;
                calc.options.scf.bond_orders = false;
                let sp = sp?;
                let t = &sp.terms;
                let terms = vec![
                    ("repulsion", t.repulsion),
                    ("dispersion", t.dispersion),
                    ("electronic", t.electronic),
                    ("isotropic_es", t.isotropic_es),
                    ("third_order", t.third_order),
                    ("anisotropic", t.anisotropic),
                    ("entropy", t.entropy),
                    ("solvation", t.solvation),
                ];
                let form = axtb::QuadrupoleForm::from(calc.options.conventions);
                out.hessian = hess.map(hessian_part);
                scf_into(&mut out, sp, terms, positions, form)
            }
            #[cfg(feature = "gxtb")]
            Engine::Gxtb(calc) => {
                let hess = match &task.hessian {
                    Some(h) => {
                        calc.options.keep_orbitals = false;
                        calc.options.bond_orders = false;
                        let method = h.method.unwrap_or_else(|| calc.default_hessian_method());
                        let o = axtb::HessianOptions { method: Some(method), step: h.step, accuracy: h.accuracy };
                        let r = calc.hessian(positions, &o)?;
                        // the single point below starts from g-xTB's own guess, as without the
                        // Hessian
                        calc.reset_guess();
                        Some(r)
                    }
                    None => None,
                };
                calc.options.keep_orbitals = molden;
                calc.options.bond_orders = task.bond_orders;
                let r = calc.singlepoint(positions, task.gradient || hess.is_some());
                calc.options.keep_orbitals = false;
                calc.options.bond_orders = false;
                let r = r?;
                let form = calc.quadrupole_form();
                out.hessian = hess.map(hessian_part);
                scf_into(&mut out, r.singlepoint, r.energies.named(), positions, form)
            }
            #[cfg(feature = "gxtb")]
            Engine::Composite(calc) => {
                let hess = match &task.hessian {
                    Some(h) => {
                        calc.set_keep_orbitals(false);
                        calc.gxtb_mut().options.bond_orders = false;
                        let r = calc.hessian(positions, h.method, h.step, h.accuracy)?;
                        // the g-xTB single point below starts from its own guess, as without
                        // the Hessian
                        calc.gxtb_mut().reset_guess();
                        Some(r)
                    }
                    None => None,
                };
                calc.set_keep_orbitals(molden);
                calc.gxtb_mut().options.bond_orders = task.bond_orders;
                let r = calc.singlepoint(positions, task.gradient || hess.is_some());
                calc.set_keep_orbitals(false);
                calc.gxtb_mut().options.bond_orders = false;
                let r = r?;
                let mut terms = r.gxtb.energies.named();
                terms.push(("solvation_correction", r.solvation_energy));
                let form = calc.gxtb().quadrupole_form();
                out.composite = Some(composite::response(calc, &r, hess.as_ref().map(|h| &h.parts)));
                out.hessian = hess.map(|h| hessian_part(h.total));
                scf_into(&mut out, r.singlepoint, terms, positions, form)
            }
            Engine::Gfnff(calc) => {
                let hess = match &task.hessian {
                    Some(h) => {
                        let o = axtb::HessianOptions { method: h.method, step: h.step, accuracy: h.accuracy };
                        Some(hessian_part(calc.hessian(positions, &o)?))
                    }
                    None => None,
                };
                let r = calc.singlepoint(positions, &Want::gradient(task.gradient || hess.is_some()))?;
                let g = r.extras.gfnff.expect("GFN-FF reports its topology");
                let t = g.topology;
                out.energy = Some(r.energy);
                out.gradient = r.gradient;
                out.charges = Some(r.charges);
                out.dipole = Some(r.dipole);
                out.terms = Some(Named::new(r.terms.iter().map(|t| (t.name, t.value)).collect()));
                out.timings = Some(Named::new(r.timings));
                out.gfnff = Some(Gfnff {
                    n_hbonds: g.n_hbonds[0] + g.n_hbonds[1],
                    n_xbonds: g.n_xbonds,
                    topology: Topology {
                        bonds: t.bonds,
                        pi_bond_orders: t.pi_bond_orders,
                        bond_types: t.bond_types,
                        hybridization: t.hybridization,
                        pi_atoms: t.pi_atoms,
                        fragments: t.fragments,
                        fragment_charges: t.fragment_charges,
                        topology_charges: t.topology_charges,
                        n_angles: t.n_angles,
                        n_torsions: t.n_torsions,
                    },
                });
                out.hessian = hess;
                None
            }
        };
        check_finite(&out)?;
        if let Some(m) = &task.molden {
            let orbitals = orbitals.ok_or_else(|| Error::Numerical("the calculation returned no orbitals".into()))?;
            let title = format!("axtb {} {}", response::VERSION, self.method);
            let text = orbitals.to_molden(m.functions, &title)?;
            if let Some(path) = &m.file {
                std::fs::write(path, &text).map_err(|e| input(format!("cannot write the Molden file {path}: {e}")))?;
            }
            out.molden_file = m.file.clone();
            out.molden = m.text.then_some(text);
        }
        if let (Some(h), Some(hp)) = (&task.hessian, &out.hessian) {
            if let Some(v) = &h.vibrations {
                let opts = v.options(self.analysis, self.method == Method::Gfnff);
                let a = axtb::vibrations::analyze(
                    &self.numbers,
                    positions,
                    &hp.hessian.data,
                    hp.dipole_derivatives.as_deref(),
                    &opts,
                )?;
                out.vibrations = Some(Vibrations::new(a, out.energy));
            }
        }
        out.seconds = Some(t0.elapsed().as_secs_f64());
        Ok(out)
    }

    /// The converged wavefunction of the last SCF (`None` before the first, and for GFN-FF),
    /// for another calculator of the same method and molecule ([`Calculator::set_guess`]).
    pub fn guess(&self) -> Option<ScfState> {
        let inner = match &self.engine {
            Engine::Gfn2(c) => c.guess().cloned().map(StateInner::Gfn2),
            #[cfg(feature = "gxtb")]
            Engine::Gxtb(c) => c.guess().cloned().map(StateInner::Gxtb),
            #[cfg(feature = "gxtb")]
            Engine::Composite(c) => {
                let (gas, solv) = c.gfn2_guesses();
                let g = c.gxtb().guess().cloned();
                (g.is_some() || gas.is_some() || solv.is_some())
                    .then(|| StateInner::Composite(g, gas.cloned(), solv.cloned()))
            }
            Engine::Gfnff(_) => None,
        }?;
        Some(ScfState { inner, method: self.method, natoms: self.numbers.len() })
    }

    /// Start the next SCF from `state` (from [`Calculator::guess`] of a calculator for the same
    /// method and molecule), whatever the `guess` setting says.
    pub fn set_guess(&mut self, state: &ScfState) -> Result<()> {
        if state.method != self.method || state.natoms != self.numbers.len() {
            return Err(input(format!(
                "a {} wavefunction of {} atoms cannot start a {} SCF of {} atoms",
                state.method,
                state.natoms,
                self.method,
                self.numbers.len()
            )));
        }
        match (&mut self.engine, &state.inner) {
            (Engine::Gfn2(c), StateInner::Gfn2(g)) => c.set_guess(g.clone())?,
            #[cfg(feature = "gxtb")]
            (Engine::Gxtb(c), StateInner::Gxtb(g)) => c.set_guess(g.clone())?,
            #[cfg(feature = "gxtb")]
            (Engine::Composite(c), StateInner::Composite(g, gas, solv)) => {
                if let Some(g) = g {
                    c.gxtb_mut().set_guess(g.clone())?;
                }
                let (cg, cs) = c.gfn2_mut();
                if let Some(x) = gas {
                    cg.set_guess(x.clone())?;
                }
                if let Some(x) = solv {
                    cs.set_guess(x.clone())?;
                }
            }
            _ => return Err(input("the wavefunction belongs to another kind of calculation")),
        }
        Ok(())
    }

    /// Start the next GFN2-xTB SCF from given moments: shell charges, atomic dipoles and
    /// quadrupoles (e.g. those of a previous response, or another program's).
    pub fn set_moments(
        &mut self,
        shell_charges: Vec<f64>,
        dipoles: Vec<[f64; 3]>,
        quadrupoles: Vec<[f64; 6]>,
    ) -> Result<()> {
        match &mut self.engine {
            Engine::Gfn2(c) => {
                Ok(c.set_guess(axtb::ScfGuess { qsh: shell_charges, dpat: dipoles, qpat: quadrupoles })?)
            }
            Engine::Gfnff(_) => Err(input("gfnff has no SCF; starting moments are GFN2-only")),
            #[cfg(feature = "gxtb")]
            _ => Err(input(
                "starting moments are GFN2-only; g-xTB starts from a whole wavefunction (the guess of another \
                 calculator)",
            )),
        }
    }

    /// Forget the previous wavefunction(s): the next SCF starts from the method's own guess.
    /// `gxtb_only`: of the composite only the g-xTB part's (for every other method nothing
    /// but g-xTB's).
    pub fn reset_guess(&mut self, gxtb_only: bool) {
        match (&mut self.engine, gxtb_only) {
            (Engine::Gfn2(c), false) => c.reset_guess(),
            #[cfg(feature = "gxtb")]
            (Engine::Gxtb(c), _) => c.reset_guess(),
            #[cfg(feature = "gxtb")]
            (Engine::Composite(c), true) => c.gxtb_mut().reset_guess(),
            #[cfg(feature = "gxtb")]
            (Engine::Composite(c), false) => c.reset_guess(),
            _ => {}
        }
    }
}

/// The g-xTB calculator of a request: plain, or the solvation composite with
/// `solvation_correction`.
#[cfg(feature = "gxtb")]
fn gxtb_engine(
    req: &Request,
    method: Method,
    threads: usize,
    uhf: Option<usize>,
    guess: Option<names::Guess>,
) -> Result<(Engine, axtb_gxtb::Conventions)> {
    let version = method.gxtb_version().expect("a g-xTB method");
    let conventions = req.conventions.as_deref().map(names::gxtb_conventions).transpose()?.unwrap_or_default();
    if req.scf_trace {
        return Err(input(format!("{method} does not support scf_trace (GFN2-xTB only)")));
    }
    if let Some(c) = &req.solvation_correction {
        if req.dispersion == Some(false) {
            return Err(input("the g-xTB solvation composite does not support switching off dispersion"));
        }
        let correction = names::correction(c)?;
        let solvation = solvation(req)?
            .ok_or_else(|| input("solvation_correction needs a solvent: alpb or gbsa (e.g. alpb \"water\")"))?;
        let mut o = axtb_gxtb::CompositeOptions {
            version,
            correction,
            solvation,
            accuracy: req.accuracy.unwrap_or(1.0),
            etemp: req.etemp,
            max_iter: req.max_iter,
            threads,
            conventions,
            reuse_guess: match guess {
                None => None,
                Some(names::Guess::Previous) => Some(true),
                Some(names::Guess::Fresh) => Some(false),
                Some(_) => return Err(input(
                    "the g-xTB solvation composite starts each part from its method's own start: guess \"previous\" \
                         or \"fresh\"",
                )),
            },
            ..Default::default()
        };
        o.correction_conventions = req.correction_conventions.as_deref().map(names::gfn2_conventions).transpose()?;
        o.mixer = req.mixer.as_deref().map(names::mixer).transpose()?;
        let calc = axtb_gxtb::CompositeCalculator::new(&req.numbers, &req.positions, req.charge, uhf, o)?;
        return Ok((Engine::Composite(Box::new(calc)), conventions));
    }
    if req.alpb.is_some() || req.gbsa.is_some() || req.solvation_state.is_some() || req.correction_conventions.is_some()
    {
        return Err(input(gxtb_solvent_hint(method)));
    }
    if req.dispersion == Some(false) {
        return Err(input(format!("{method} does not support switching off dispersion")));
    }
    if req.mixer.is_some() {
        return Err(input(format!("{method} does not support choosing the SCF mixer")));
    }
    let mut o = axtb_gxtb::ScfOptions { threads, conventions, ..Default::default() };
    if let Some(a) = req.accuracy {
        o.accuracy = a;
    }
    if let Some(t) = req.etemp {
        o.etemp = t;
    }
    if let Some(m) = req.max_iter {
        o.max_iter = m;
    }
    o.guess = match guess {
        None => None,
        Some(names::Guess::Previous) => Some(axtb_gxtb::Guess::Previous),
        Some(names::Guess::Fresh | names::Guess::EeqBc) => Some(axtb_gxtb::Guess::EeqBc),
        Some(names::Guess::Eeq | names::Guess::Sad) => {
            return Err(input(format!("{method} starts from the program's EEQ-BC charges: guess \"eeq-bc\"")))
        }
    };
    let calc = axtb_gxtb::Calculator::new(version, &req.numbers, &req.positions, req.charge, uhf, o)?;
    Ok((Engine::Gxtb(Box::new(calc)), conventions))
}

/// A Hessian of an SCF method in the response.
fn hessian_part(h: axtb::Hessian) -> HessianPart {
    HessianPart {
        hessian: Matrix::square(h.hessian),
        hessian_method: names::hessian_method_label(h.method),
        hessian_max_asymmetry: h.max_asymmetry,
        hessian_timings: Named::new(h.timings),
        dipole_derivatives: h.dipole_derivatives,
    }
}

/// Fill the single point of an SCF method into `out`; returns its orbitals (when kept).
fn scf_into(
    out: &mut Response,
    sp: axtb::SinglePoint,
    terms: Vec<(&'static str, f64)>,
    positions: &[[f64; 3]],
    form: axtb::QuadrupoleForm,
) -> Option<axtb::Orbitals> {
    let [a, b] = axtb::properties::alpha_beta(&sp.levels);
    let nat = positions.len();
    out.eigensolver = Some(sp.eigensolver);
    out.energy = Some(sp.energy);
    out.gradient = sp.gradient;
    out.dipole = Some(sp.dipole);
    out.terms = Some(Named::new(terms));
    out.timings = Some(Named::new(sp.timings));
    out.scf = Some(Scf {
        iterations: sp.iterations,
        guess: sp.guess,
        mixer_used: sp.mixer_used,
        scf_attempts: (!sp.scf_attempts.is_empty()).then(|| {
            sp.scf_attempts
                .iter()
                .map(|a| Attempt { mixer: a.mixer.label(), iterations: a.iterations, converged: a.converged })
                .collect()
        }),
        homo_lumo_gap: sp.homo_lumo_gap,
        quadrupole: axtb::properties::molecular_quadrupole(positions, &sp.qat, &sp.dpat, &sp.qpat, form),
        shell_charges: sp.qsh,
        atomic_dipoles: sp.dpat,
        atomic_quadrupoles: sp.qpat,
        orbital_energies_alpha: a.energies,
        orbital_energies_beta: b.energies,
        occupations_alpha: a.occupations,
        occupations_beta: b.occupations,
        bond_orders: sp.bond_orders.map(|bo| Matrix { rows: nat, cols: nat, data: bo }),
    });
    out.charges = Some(sp.qat);
    sp.orbitals
}

/// Refuse a non-finite energy, gradient or Hessian (JSON cannot hold them; they would be
/// `null`).
fn check_finite(out: &Response) -> Result<()> {
    let bad = |what: &str| Err(Error::Numerical(format!("the calculation gave a non-finite {what}")));
    if out.energy.is_some_and(|e| !e.is_finite()) {
        return bad("energy");
    }
    if out.gradient.as_ref().is_some_and(|g| g.iter().flatten().any(|x| !x.is_finite())) {
        return bad("gradient");
    }
    if out.hessian.as_ref().is_some_and(|h| h.hessian.data.iter().any(|x| !x.is_finite())) {
        return bad("Hessian");
    }
    Ok(())
}

#[cfg(feature = "gxtb")]
mod composite {
    use crate::response::{Attempt, Components, Composite, CompositeEtemp, CompositeInfo, Part, Parts};

    fn part(p: &axtb_gxtb::PartInfo) -> Part {
        Part {
            method: p.method.clone(),
            conventions: p.conventions,
            energy: p.energy,
            iterations: p.iterations,
            mixer_used: p.mixer_used,
            scf_attempts: p
                .scf_attempts
                .iter()
                .map(|a| Attempt { mixer: a.mixer.label(), iterations: a.iterations, converged: a.converged })
                .collect(),
            etemp: p.etemp,
            max_iter: p.max_iter,
            retry: p.retry.clone(),
            seconds: p.seconds,
            hessian_method: p.hessian_method.map(crate::names::hessian_method_label),
        }
    }

    fn parts(p: &[axtb_gxtb::PartInfo; 3]) -> Parts {
        Parts { gxtb: part(&p[0]), gfn2_solvated: part(&p[1]), gfn2_gas: part(&p[2]) }
    }

    /// The composite's definition, settings, components and parts (`hessian`: the parts of
    /// its Hessian).
    pub(super) fn response(
        c: &axtb_gxtb::CompositeCalculator,
        r: &axtb_gxtb::CompositeResults,
        hessian: Option<&[axtb_gxtb::PartInfo; 3]>,
    ) -> Composite {
        let o = c.options();
        let (model, state) = c.solvation_labels();
        let (conventions, correction_conventions) = c.conventions_labels();
        let (gxtb, gfn2) = c.etemps();
        Composite {
            composite: CompositeInfo {
                definition: c.definition(),
                method: o.version.method(),
                solvation_correction: o.correction.name(),
                solvation_model: model,
                solvent: o.solvation.solvent.clone(),
                solvation_state: state,
                conventions,
                correction_conventions,
                etemp: CompositeEtemp { gxtb, gfn2 },
                reuse_guess: c.reuses_guess(),
                gxtb_reuse_guess: c.gxtb_reuses_guess(),
                retries: o.retries,
            },
            components: Components {
                gxtb: r.gxtb.singlepoint.energy,
                gfn2_solvated: r.gfn2_solvated.energy,
                gfn2_gas: r.gfn2_gas.energy,
                solvation_correction: r.solvation_energy,
            },
            parts: parts(&r.parts),
            hessian_parts: hessian.map(parts),
        }
    }
}
