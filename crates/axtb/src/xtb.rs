//! Self-consistent charge extended tight-binding single point: SCF, energy and analytic
//! nuclear gradient for the GFN-type Hamiltonians.

use crate::basis::Basis;
use crate::conventions::{SccGradient, OCCUPATION_CUTOFF};
use crate::coulomb::{DampedMultipole, EffectiveCoulomb, MultipoleCache, OnsiteThirdOrder};
use crate::hamiltonian::{H0Params, Integrals};
use crate::linalg::{weighted_outer, weighted_outer_sym, GenEigSolver};
use crate::molden::{MoSet, Orbitals, Spin};
use crate::molecule::Molecule;
use crate::ncoord::Ncoord;
use crate::properties::{GapRule, OrbitalLevels};
use crate::repulsion::Repulsion;
use crate::scf::convergence::XtbRule;
use crate::scf::mixing::{MixerKind, MixerPlan, ScfAttempt};
use crate::scf::{dot, fock, mulliken, ConvergenceRule, Iteration, ScfVariables, Verdict};
use crate::{Error, Result};
use axtb_base::fd;
use axtb_base::occupations::{self, Newton};
use axtb_base::threads;
use faer::Mat;
use rayon::prelude::*;

/// An atom-resolved, charge-dependent energy term that enters the SCF (e.g. the
/// self-consistent D4 dispersion of GFN2-xTB).
///
/// `parallel`: the method may spread its work over the current rayon pool (the caller runs
/// it inside the calculator's pool, and passes false where the call itself is one of many
/// running in parallel). The results must not depend on it.
pub(crate) trait ChargeDependentTerm: Send + Sync {
    /// Name used in energy decompositions.
    fn name(&self) -> &'static str;
    /// Geometry-dependent setup; returns an opaque cache.
    fn update(&self, mol: &Molecule) -> Box<dyn std::any::Any + Send + Sync>;
    /// Charge-independent energy (per atom) and optional gradient, evaluated once.
    fn fixed_energy(
        &self,
        mol: &Molecule,
        cache: &(dyn std::any::Any + Send + Sync),
        energies: &mut [f64],
        grad: Option<&mut [[f64; 3]]>,
        parallel: bool,
    );
    /// Charge-dependent energy per atom.
    fn energy(
        &self,
        mol: &Molecule,
        cache: &(dyn std::any::Any + Send + Sync),
        qat: &[f64],
        energies: &mut [f64],
        parallel: bool,
    );
    /// dE/dq per atom, added to `vat`.
    fn potential(
        &self,
        mol: &Molecule,
        cache: &(dyn std::any::Any + Send + Sync),
        qat: &[f64],
        vat: &mut [f64],
        parallel: bool,
    );
    /// Explicit gradient at fixed charges.
    fn gradient(&self, mol: &Molecule, cache: &(dyn std::any::Any + Send + Sync), qat: &[f64], grad: &mut [[f64; 3]]);

    /// Second derivatives d²E/dq_A dq_B, added to `k` (row-major `nat x nat`). The default
    /// takes central differences of [`potential`](Self::potential).
    fn kernel(&self, mol: &Molecule, cache: &(dyn std::any::Any + Send + Sync), qat: &[f64], k: &mut [f64]) {
        let nat = mol.nat();
        let h = 1e-4;
        let mut q = qat.to_vec();
        for b in 0..nat {
            let kb = fd::derivative(fd::Stencil::Central2, h, |d| {
                let mut v = vec![0.0; nat];
                q[b] = qat[b] + d;
                self.potential(mol, cache, &q, &mut v, false);
                v
            });
            q[b] = qat[b];
            for a in 0..nat {
                k[a * nat + b] += kb[a];
            }
        }
    }

    /// Second derivatives involving the nuclear positions at fixed charges: the derivative
    /// of the potential dE/dq_A, added to `dv` (row-major `nat x 3 nat`), and, when `hess`
    /// is given, the Hessian of the whole term (charge-independent part included), added
    /// to it (row-major `3 nat x 3 nat`). With `parallel`, work may be spread over the
    /// current rayon pool.
    ///
    /// The default takes central differences of the analytic potential and gradient, one
    /// geometry update per displaced geometry, parallel over displacements.
    fn second_derivatives(
        &self,
        mol: &Molecule,
        _cache: &(dyn std::any::Any + Send + Sync),
        qat: &[f64],
        dv: &mut [f64],
        hess: Option<&mut [f64]>,
        parallel: bool,
    ) {
        fd_second_derivatives(self, mol, qat, dv, hess, parallel);
    }
}

/// Central differences of the analytic potential and gradient of a charge-dependent term
/// (see [`ChargeDependentTerm::second_derivatives`]): one geometry update per displaced
/// geometry, parallel over displacements with `parallel`.
pub(crate) fn fd_second_derivatives<T: ChargeDependentTerm + ?Sized>(
    term: &T,
    mol: &Molecule,
    qat: &[f64],
    dv: &mut [f64],
    hess: Option<&mut [f64]>,
    parallel: bool,
) {
    let nat = mol.nat();
    let n3 = 3 * nat;
    let h = FD_STEP;
    let want_h = hess.is_some();
    // column x: the potential and (with `hess`) the gradient at R + d e_x
    let column = |x: usize| -> (Vec<f64>, Vec<[f64; 3]>) {
        let (a, k) = (x / 3, x % 3);
        let mut m = mol.clone();
        fd::derivative(fd::Stencil::Central2, h, |d| {
            m.positions[a][k] = mol.positions[a][k] + d;
            let c = term.update(&m);
            let mut v = vec![0.0; nat];
            term.potential(&m, c.as_ref(), qat, &mut v, false);
            let mut g = Vec::new();
            if want_h {
                g = vec![[0.0; 3]; nat];
                let mut e = vec![0.0; nat];
                term.fixed_energy(&m, c.as_ref(), &mut e, Some(&mut g), false);
                term.gradient(&m, c.as_ref(), qat, &mut g);
            }
            (v, g)
        })
    };
    let mut hess = hess;
    for (x, (dvx, hx)) in threads::map(n3, parallel, column).into_iter().enumerate() {
        for i in 0..nat {
            dv[i * n3 + x] += dvx[i];
        }
        if let Some(hs) = hess.as_deref_mut() {
            for y in 0..n3 {
                hs[y * n3 + x] += hx[y / 3][y % 3];
            }
        }
    }
}

/// Step (bohr) of the central differences used where a term has no analytic second
/// derivatives.
pub(crate) const FD_STEP: f64 = 2e-5;

/// A parametrized xTB Hamiltonian for one molecule.
pub(crate) struct XtbModel {
    pub basis: Basis,
    pub h0: H0Params,
    pub ncoord: Ncoord,
    pub repulsion: Repulsion,
    pub es2: EffectiveCoulomb,
    pub es3: Option<OnsiteThirdOrder>,
    pub aes2: Option<DampedMultipole>,
    /// Atom-pair distance cutoff (bohr) of the integrals; infinite where only primitive
    /// pairs are screened.
    pub pair_cutoff: f64,
    /// Settings of the numerical conventions.
    pub params: crate::ConventionParams,
}

/// Starting charges of an SCF without a previous wavefunction.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum InitialGuess {
    /// Neutral atoms (zero charges).
    Sad,
    /// EEQ charges distributed over the shells by reference occupation (xtb's default).
    #[default]
    Eeq,
}

/// SCF settings.
#[derive(Clone, Debug)]
pub struct ScfOptions {
    /// Accuracy multiplier: energy threshold 1e-6·acc, residual threshold 2e-5·acc (> 0).
    pub accuracy: f64,
    /// Iterations of one SCF attempt (at least 1).
    pub max_iter: usize,
    /// Electronic temperature in kelvin (Fermi smearing; >= 0).
    pub etemp: f64,
    /// Damping of the mixer (Broyden: of the residual; DIIS: of the combined residual), in
    /// (0, 1].
    pub damping: f64,
    /// Mixer of the SCC variables; `None` uses the conventions' choice
    /// ([`ConventionParams::mixer`](crate::ConventionParams::mixer)). With
    /// [`MixerKind::Auto`] an SCF that Broyden mixing does not converge within `max_iter`
    /// starts again from the initial guess with DIIS and another `max_iter` iterations.
    pub mixer: Option<MixerKind>,
    /// Threads of the calculation (at least 1; 1 = sequential): integrals, Fock matrix,
    /// potentials, dispersion and dense linear algebra, and the displacements or
    /// perturbations of a Hessian.
    pub threads: usize,
    /// Starting point when no previous wavefunction is given.
    pub guess: InitialGuess,
    /// Keep the converged orbitals and their basis in [`SinglePoint::orbitals`] (e.g. for a
    /// Molden file).
    pub keep_orbitals: bool,
    /// Compute the Mayer–Wiberg bond orders ([`SinglePoint::bond_orders`]).
    pub bond_orders: bool,
    /// Print every SCF iteration (energy, energy change, residual) to stderr, for debugging
    /// SCFs that do not converge.
    pub trace: bool,
}

impl Default for ScfOptions {
    fn default() -> Self {
        Self {
            accuracy: 1.0,
            max_iter: 250,
            etemp: 300.0,
            damping: 0.4,
            mixer: None,
            threads: 1,
            guess: InitialGuess::Eeq,
            keep_orbitals: false,
            bond_orders: false,
            trace: false,
        }
    }
}

/// Energy contributions (Hartree).
#[derive(Clone, Debug, Default)]
pub struct EnergyTerms {
    pub repulsion: f64,
    pub dispersion: f64,
    pub solvation: f64,
    pub electronic: f64,
    pub isotropic_es: f64,
    pub third_order: f64,
    pub anisotropic: f64,
    pub entropy: f64,
}

/// Results of a single point.
#[derive(Clone, Debug)]
pub struct SinglePoint {
    pub energy: f64,
    pub terms: EnergyTerms,
    pub gradient: Option<Vec<[f64; 3]>>,
    /// Mulliken atomic and shell charges, atomic dipoles and quadrupoles.
    pub qat: Vec<f64>,
    pub qsh: Vec<f64>,
    pub dpat: Vec<[f64; 3]>,
    pub qpat: Vec<[f64; 6]>,
    /// Molecular dipole moment (e·bohr) relative to the origin.
    pub dipole: [f64; 3],
    pub orbital_energies: Vec<f64>,
    pub occupations: Vec<f64>,
    /// SCF iterations of the attempt that converged (see `scf_attempts` for all).
    pub iterations: usize,
    /// Mixer that produced the result: `"broyden"`, `"diis"`, or `"diis-fallback"` (mixer
    /// `auto`: Broyden did not converge, DIIS from the initial guess did); `None` for
    /// methods whose SCF has no mixer choice (g-xTB).
    pub mixer_used: Option<&'static str>,
    /// Every SCF attempt in order: mixer, iterations, converged.
    pub scf_attempts: Vec<ScfAttempt>,
    /// Wall time per phase in seconds.
    pub timings: Vec<(&'static str, f64)>,
    /// The converged orbitals with their basis, when asked for
    /// ([`ScfOptions::keep_orbitals`]); [`Orbitals::to_molden`] writes a Molden file.
    pub orbitals: Option<Orbitals>,
    /// Orbital energies and occupations per spin channel: one for a closed shell (occupations
    /// up to 2), alpha and beta for an open shell (GFN2-xTB: the same orbitals with the alpha
    /// and beta occupations; g-xTB: the orbitals of each channel).
    pub levels: Vec<OrbitalLevels>,
    /// HOMO–LUMO gap (Eh; `crate::properties::homo_lumo_gap` with the method's and
    /// conventions' rule); `None` without a LUMO.
    pub homo_lumo_gap: Option<f64>,
    /// Mayer–Wiberg bond orders (N x N row-major, zero diagonal), when asked for
    /// ([`ScfOptions::bond_orders`]; see `crate::properties`).
    pub bond_orders: Option<Vec<f64>>,
    /// Where the SCF started: `"previous"` (a given wavefunction, usually the previous
    /// geometry's), or the method's own start: `"eeq"` (GFN2-xTB: EEQ charges), `"sad"` (neutral
    /// atoms, also when the EEQ charges cannot be computed), `"eeq-bc"` (g-xTB: the program's
    /// EEQ-BC start).
    pub guess: &'static str,
    /// The eigensolver of the SCF ([`GenEigSolver::backend`]): `"faer"`, or the run-time LAPACK
    /// library's name and version (`AXTB_LAPACK`), or faer with the reason the library was not
    /// used.
    pub eigensolver: String,
}

/// Starting point of an SCF: shell charges and atomic multipoles, e.g. from a previous
/// geometry of the same molecule.
#[derive(Clone, Debug)]
pub struct ScfGuess {
    pub qsh: Vec<f64>,
    pub dpat: Vec<[f64; 3]>,
    pub qpat: Vec<[f64; 6]>,
}

impl SinglePoint {
    /// The converged moments as a guess for a nearby geometry.
    pub fn guess(&self) -> ScfGuess {
        ScfGuess { qsh: self.qsh.clone(), dpat: self.dpat.clone(), qpat: self.qpat.clone() }
    }
}

/// Geometry-dependent intermediates of one geometry (everything but the AO matrices).
pub(crate) struct Geometry {
    /// Coordination numbers (H0 self-energies and multipole damping radii).
    pub cn: Vec<f64>,
    /// Shell self-energies and their CN derivatives.
    pub se: Vec<f64>,
    pub dsedcn: Vec<f64>,
    /// Atom-pair screening distance of the integrals.
    pub cutoff: f64,
    /// Shell-resolved second-order kernel (row-major nsh x nsh).
    pub gamma: Vec<f64>,
    pub mp_cache: Option<MultipoleCache>,
    /// Caches of the charge-dependent extra terms.
    pub caches: Vec<Box<dyn std::any::Any + Send + Sync>>,
}

/// Potential shifts from the charge-dependent terms.
#[derive(Clone, Debug)]
pub(crate) struct Potential {
    pub vat: Vec<f64>,
    pub vsh: Vec<f64>,
    pub vdp: Vec<[f64; 3]>,
    pub vqp: Vec<[f64; 6]>,
}

impl Potential {
    /// Charge potential per AO (shell plus atom part).
    pub fn vao(&self, bas: &Basis) -> Vec<f64> {
        (0..bas.nao).map(|mu| self.vsh[bas.ao2sh[mu]] + self.vat[bas.ao2at[mu]]).collect()
    }
}

/// Converged SCF wavefunction with the intermediates it was computed from: the starting
/// point of the response (coupled-perturbed SCC) and Hessian code.
pub(crate) struct ScfState {
    pub geo: Geometry,
    pub ints: Integrals,
    /// MO coefficients (columns) and orbital energies (ascending).
    pub coeff: Mat<f64>,
    pub emo: Vec<f64>,
    /// Alpha and beta occupations of the common orbitals.
    pub occ: [Vec<f64>; 2],
    /// Fermi levels of the two spins (`None` without smearing).
    pub efermi: [Option<f64>; 2],
    /// Electronic temperature (Eh).
    pub kt: f64,
    /// Total and energy-weighted density matrices.
    pub density: Mat<f64>,
    pub wdensity: Mat<f64>,
    /// SCC variables (output moments of the last iteration).
    pub qsh: Vec<f64>,
    pub qat: Vec<f64>,
    pub dpat: Vec<[f64; 3]>,
    pub qpat: Vec<[f64; 6]>,
    /// Potential of the final Fock matrix.
    pub pot: Potential,
}

impl XtbModel {
    /// Coordination numbers, self-energies, Coulomb kernels and extra-term caches.
    pub(crate) fn geometry(&self, mol: &Molecule, extra: &[&dyn ChargeDependentTerm]) -> Geometry {
        let bas = &self.basis;
        let cn = self.ncoord.compute(mol);
        let (se, dsedcn) = self.h0.selfenergy(mol, bas, &cn);
        let cutoff = self.pair_cutoff;
        let gamma = self.es2.matrix(mol, &bas.ish_at, &bas.nsh_at, bas.nsh);
        let mp_cache = self.aes2.as_ref().map(|m| m.update(mol, &cn));
        let caches = extra.iter().map(|t| t.update(mol)).collect();
        Geometry { cn, se, dsedcn, cutoff, gamma, mp_cache, caches }
    }

    /// Potentials of all charge-dependent terms for the given moments.
    pub(crate) fn potentials(
        &self,
        mol: &Molecule,
        extra: &[&dyn ChargeDependentTerm],
        geo: &Geometry,
        qsh: &[f64],
        qat: &[f64],
        dpat: &[[f64; 3]],
        qpat: &[[f64; 6]],
        pot: &mut Potential,
        parallel: bool,
    ) {
        let bas = &self.basis;
        pot.vat.iter_mut().for_each(|x| *x = 0.0);
        pot.vdp.iter_mut().for_each(|x| *x = [0.0; 3]);
        pot.vqp.iter_mut().for_each(|x| *x = [0.0; 6]);
        gamma_rows(&geo.gamma, qsh, &mut pot.vsh, parallel);
        if let Some(es3) = &self.es3 {
            es3.potential(mol, &bas.ish_at, qsh, &mut pot.vsh);
        }
        if let (Some(aes), Some(c)) = (&self.aes2, &geo.mp_cache) {
            aes.potential(mol, c, qat, dpat, qpat, &mut pot.vat, &mut pot.vdp, &mut pot.vqp, parallel);
        }
        for (t, c) in extra.iter().zip(&geo.caches) {
            t.potential(mol, c.as_ref(), qat, &mut pot.vat, parallel);
        }
    }

    /// Gradient of the self-consistent part at fixed density `p`, energy-weighted density
    /// `w`, moments and potential: explicit electrostatics and extra terms, the core
    /// Hamiltonian/overlap/multipole integral derivatives and the CN chain rule.
    pub(crate) fn add_scc_gradient(
        &self,
        mol: &Molecule,
        extra: &[&dyn ChargeDependentTerm],
        geo: &Geometry,
        p: &Mat<f64>,
        w: &Mat<f64>,
        qsh: &[f64],
        qat: &[f64],
        dpat: &[[f64; 3]],
        qpat: &[[f64; 6]],
        pot: &Potential,
        g: &mut [[f64; 3]],
        parallel: bool,
    ) {
        let bas = &self.basis;
        let nat = mol.nat();
        self.es2.add_gradient(mol, &bas.ish_at, &bas.nsh_at, qsh, g);
        if let (Some(aes), Some(c)) = (&self.aes2, &geo.mp_cache) {
            aes.add_gradient(mol, c, qat, dpat, qpat, g);
        }
        for (t, c) in extra.iter().zip(&geo.caches) {
            t.gradient(mol, c.as_ref(), qat, g);
        }
        let vao = pot.vao(bas);
        let mut dedcn = vec![0.0; nat];
        self.h0.add_gradient(
            mol,
            bas,
            &geo.se,
            &geo.dsedcn,
            geo.cutoff,
            p,
            w,
            &vao,
            &pot.vdp,
            &pot.vqp,
            &mut dedcn,
            g,
            parallel,
        );
        self.ncoord.add_gradient(mol, &dedcn, g);
    }

    /// Full nuclear gradient at `mol` for a given density `p`, energy-weighted density `w`
    /// and moments (not necessarily self-consistent): the gradient functional of the SCF
    /// energy, with the potentials recomputed from the moments at this geometry.
    pub(crate) fn gradient_functional(
        &self,
        mol: &Molecule,
        extra: &[&dyn ChargeDependentTerm],
        p: &Mat<f64>,
        w: &Mat<f64>,
        qsh: &[f64],
        dpat: &[[f64; 3]],
        qpat: &[[f64; 6]],
        parallel: bool,
    ) -> Vec<[f64; 3]> {
        let nat = mol.nat();
        let bas = &self.basis;
        let mut g = vec![[0.0; 3]; nat];
        let mut e = vec![0.0; nat];
        self.repulsion.compute(mol, &mut e, Some(&mut g));
        let geo = self.geometry(mol, extra);
        for (t, c) in extra.iter().zip(&geo.caches) {
            t.fixed_energy(mol, c.as_ref(), &mut e, Some(&mut g), parallel);
        }
        let mut qat = vec![0.0; nat];
        qat_from_qsh(bas, qsh, &mut qat);
        let mut pot = Potential {
            vat: vec![0.0; nat],
            vsh: vec![0.0; bas.nsh],
            vdp: vec![[0.0; 3]; nat],
            vqp: vec![[0.0; 6]; nat],
        };
        self.potentials(mol, extra, &geo, qsh, &qat, dpat, qpat, &mut pot, parallel);
        self.add_scc_gradient(mol, extra, &geo, p, w, qsh, &qat, dpat, qpat, &pot, &mut g, parallel);
        g
    }

    /// Run the SCF and optionally the analytic gradient.
    pub fn singlepoint(
        &self,
        mol: &Molecule,
        extra: &[&dyn ChargeDependentTerm],
        opts: &ScfOptions,
        guess: Option<&ScfGuess>,
        want_gradient: bool,
    ) -> Result<SinglePoint> {
        self.singlepoint_state(mol, extra, opts, guess, want_gradient, false).map(|(sp, _)| sp)
    }

    /// Run the SCF and optionally the analytic gradient; with `keep_state` also return the
    /// converged wavefunction and intermediates.
    pub(crate) fn singlepoint_state(
        &self,
        mol: &Molecule,
        extra: &[&dyn ChargeDependentTerm],
        opts: &ScfOptions,
        guess: Option<&ScfGuess>,
        want_gradient: bool,
        keep_state: bool,
    ) -> Result<(SinglePoint, Option<ScfState>)> {
        let mut timer = axtb_base::timing::Laps::new();
        check_scf_options(opts)?;
        let nat = mol.nat();
        let bas = &self.basis;
        let (nsh, nao) = (bas.nsh, bas.nao);
        // the kernels of axtb itself, on the current rayon pool (the calculator's)
        let parallel = opts.threads > 1;
        // faer's parallel dense kernels are slower than sequential ones for small matrices
        // (n = 100: 0.83 ms sequential vs 1.15 ms on 8 threads; n = 266: 7.45 vs 5.54 ms)
        // (bench/linalg), so small systems keep the linear algebra sequential
        let par = threads::dense_par(opts.threads, self.basis.nao);
        let acc = opts.accuracy;

        // electrons
        let (n0sh, nref) = self.h0.reference_occupation(mol, bas);
        let (nalpha, nbeta) = electrons(mol, nref, nao)?;
        let kt = if opts.etemp <= self.params.aufbau_max_etemp { 0.0 } else { opts.etemp * self.params.kb };

        let mut grad = if want_gradient { Some(vec![[0.0; 3]; nat]) } else { None };
        let mut terms = EnergyTerms::default();

        // non-self-consistent contributions
        let mut energies = vec![0.0; nat];
        let mut erep = vec![0.0; nat];
        self.repulsion.compute(mol, &mut erep, grad.as_deref_mut());
        terms.repulsion = erep.iter().sum();
        // geometry-dependent setup
        let geo = self.geometry(mol, extra);
        // fixed (charge-independent) energy of every extra term
        let mut efixed = vec![0.0; extra.len()];
        for (k, (t, c)) in extra.iter().zip(&geo.caches).enumerate() {
            let mut e = vec![0.0; nat];
            t.fixed_energy(mol, c.as_ref(), &mut e, grad.as_deref_mut(), parallel);
            efixed[k] = e.iter().sum();
            for i in 0..nat {
                energies[i] += e[i];
            }
        }
        for i in 0..nat {
            energies[i] += erep[i];
        }

        timer.lap("classical");
        let ints: Integrals = self.h0.build(mol, bas, &geo.se, geo.cutoff, parallel);
        let gamma = &geo.gamma;
        let mp_cache = &geo.mp_cache;
        let caches = &geo.caches;
        timer.lap("integrals");
        // xtb 6.7.1's Fock matrix leaves out small multipole integrals
        // (`ConventionParams::scc_multipole_neglect`)
        let neglect = if self.params.scc_multipole_neglect && self.aes2.is_some() {
            Some(self.h0.multipole_neglect(mol, bas, geo.cutoff, acc))
        } else {
            None
        };
        let mut solver = GenEigSolver::new(ints.overlap.as_ref(), par)?;
        timer.lap("cholesky");

        // SCF variables: shell charges, atomic dipoles, atomic quadrupoles (tblite, xtb 6.8.0:
        // off-diagonals scaled by sqrt(2) inside the mixer so the residual norm counts them
        // twice; xtb 6.7.1: unscaled; `ConventionParams::scc_quadrupole_scale`)
        let has_mp = self.aes2.is_some();
        let vars = ScfVariables::new(nsh, nat, has_mp, false).with_quad_offdiag_scale(self.params.scc_quadrupole_scale);
        let ndim = vars.dim();
        let kind = opts.mixer.unwrap_or(self.params.mixer);
        let mut qsh = vec![0.0; nsh];
        let mut qat = vec![0.0; nat];
        let mut dpat = vec![[0.0; 3]; nat];
        let mut qpat = vec![[0.0; 6]; nat];
        // where the SCF starts (`SinglePoint::guess`): a given guess of this molecule's size,
        // else EEQ charges when the options and conventions ask for them and the EEQ system
        // can be solved, else neutral atoms
        let start_label = match guess {
            Some(g) if g.qsh.len() == nsh && g.dpat.len() == nat && g.qpat.len() == nat => {
                qsh.copy_from_slice(&g.qsh);
                dpat.copy_from_slice(&g.dpat);
                qpat.copy_from_slice(&g.qpat);
                qat_from_qsh(bas, &qsh, &mut qat);
                "previous"
            }
            _ if opts.guess == InitialGuess::Eeq && self.params.eeq_guess => {
                // atomic EEQ charges, split over the shells by reference occupation
                match crate::eeq::eeq_charges_with(mol, &self.params.eeq_guess_cn) {
                    Ok(q) => {
                        for iat in 0..nat {
                            let is = bas.ish_at[iat];
                            let nsa = bas.nsh_at[iat];
                            let tot: f64 = n0sh[is..is + nsa].iter().sum();
                            for ish in 0..nsa {
                                qsh[is + ish] = if tot > 0.0 { q[iat] * n0sh[is + ish] / tot } else { 0.0 };
                            }
                        }
                        qat_from_qsh(bas, &qsh, &mut qat);
                        "eeq"
                    }
                    Err(_) => "sad",
                }
            }
            _ => "sad",
        };
        let mut fock = Mat::<f64>::zeros(nao, nao);
        let mut work = Mat::<f64>::zeros(nao, nao);
        let mut coeff = Mat::<f64>::zeros(nao, nao);
        // eigenvectors of the occupied orbitals only, where nothing needs all of them
        let all_orbitals = keep_state || opts.keep_orbitals || (opts.bond_orders && nalpha != nbeta);
        let lowest = self.params.occupied_orbitals_only && !all_orbitals;
        // the orbitals with eigenvectors (the first kocc columns of `coeff`)
        let mut kocc = nao;
        let mut density = Mat::<f64>::zeros(nao, nao);
        let mut emo = vec![0.0; nao];
        let mut focc = vec![0.0; nao];
        let mut occ_a = vec![0.0; nao];
        let mut occ_b = vec![0.0; nao];
        let mut efermi = [None, None];
        let mut pot =
            Potential { vat: vec![0.0; nat], vsh: vec![0.0; nsh], vdp: vec![[0.0; 3]; nat], vqp: vec![[0.0; 6]; nat] };
        let mut vao = vec![0.0; nao];
        let mut eelec = vec![0.0; nat];
        let mut vsh_out = vec![0.0; nsh];
        // the output moments in the mixer's layout
        let mut packed_out = vec![0.0; ndim];
        let mut converged = false;
        let mut iscf = 0;
        // xtb 6.7.1: the moments of the last iteration's input and as mixed after it
        type Moments = (Vec<f64>, Vec<f64>, Vec<[f64; 3]>, Vec<[f64; 6]>);
        let unpack = |q: &[f64]| -> Moments {
            let mut m: Moments = (vec![0.0; nsh], vec![0.0; nat], vec![[0.0; 3]; nat], vec![[0.0; 6]; nat]);
            vars.unpack(q, &mut m.0, &mut m.2, &mut m.3, None);
            qat_from_qsh(bas, &m.0, &mut m.1);
            m
        };
        let mut xtb_moments: Option<(Moments, Moments)> = None;
        let (mut de_last, mut pnorm) = (0.0, 0.0);

        // one attempt per mixer of the plan (auto: Broyden, then DIIS from the initial guess)
        let mut plan = MixerPlan::new(kind);
        let mixers = plan.mixers();
        let start = (qsh.clone(), qat.clone(), dpat.clone(), qpat.clone());
        for (ia, &mk) in mixers.iter().enumerate() {
            let last_attempt = ia + 1 == mixers.len();
            if ia > 0 {
                solver.restart();
                qsh.copy_from_slice(&start.0);
                qat.copy_from_slice(&start.1);
                dpat.copy_from_slice(&start.2);
                qpat.copy_from_slice(&start.3);
                eelec.iter_mut().for_each(|x| *x = 0.0);
                iscf = 0;
                (de_last, pnorm) = (0.0, 0.0);
            }
            let mut mixer = mk.mixer(ndim, opts.max_iter, opts.damping);
            // the conventions' test: the residual norm (xtb 6.7.1: per atom) and xtb 6.7.1's
            // extra iteration (`ConventionParams::scc_residual`, `scc_extra_iteration`)
            let mut rule = XtbRule::new(&self.params, acc, nat, has_mp);
            let mut stop = false;
            while !stop && iscf < opts.max_iter {
                let elast: f64 = eelec.iter().sum();
                if iscf > 0 {
                    if !mixer.next() {
                        if last_attempt {
                            let name = if mk == MixerKind::Broyden { "Broyden" } else { "DIIS" };
                            return Err(Error::Numerical(format!("{name} mixing failed")));
                        }
                        break;
                    }
                    // (without multipoles the dipoles and quadrupoles stay zero, as the
                    // Mulliken analysis left them)
                    vars.unpack(mixer.input(), &mut qsh, &mut dpat, &mut qpat, None);
                    qat_from_qsh(bas, &qsh, &mut qat);
                }
                iscf += 1;

                // potential from the input moments
                self.potentials(mol, extra, &geo, &qsh, &qat, &dpat, &qpat, &mut pot, parallel);
                timer.lap("scf: potentials");
                for mu in 0..nao {
                    vao[mu] = pot.vsh[bas.ao2sh[mu]] + pot.vat[bas.ao2at[mu]];
                }
                fock::from_potentials(bas, &ints, &vao, &pot.vdp, &pot.vqp, &mut work, &mut fock, parallel);
                if let Some(ng) = &neglect {
                    ng.apply(mol, &pot.vdp, &pot.vqp, &mut fock);
                }
                if (0..nao).any(|j| fock.col_as_slice(j).iter().any(|x| !x.is_finite())) {
                    return Err(Error::Numerical(format!("non-finite Fock matrix in SCF iteration {iscf}")));
                }
                timer.lap("scf: fock");

                vars.pack(&qsh, &dpat, &qpat, None, mixer.input_mut());

                // new density
                if lowest {
                    // the orbitals with occupation above the cutoff (from the eigenvalues)
                    kocc = solver.solve_lowest(fock.as_ref(), &mut emo, &mut coeff, |e| {
                        let (mut a, mut b) = (vec![0.0; e.len()], vec![0.0; e.len()]);
                        occupations::fill_tblite(nalpha, kt, e, &mut a, Newton::XTB680);
                        occupations::fill_tblite(nbeta, kt, e, &mut b, Newton::XTB680);
                        (0..e.len()).rposition(|i| a[i] + b[i] > OCCUPATION_CUTOFF).map_or(0, |i| i + 1)
                    })?;
                } else {
                    solver.solve(fock.as_ref(), &mut emo, &mut coeff)?;
                }
                // tblite's Fermi filling (xtb 6.8.0's `fermismear`) in every convention
                efermi[0] = occupations::fill_tblite(nalpha, kt, &emo, &mut occ_a, Newton::XTB680);
                efermi[1] = occupations::fill_tblite(nbeta, kt, &emo, &mut occ_b, Newton::XTB680);
                // The Newton search for the Fermi level stops after 200 steps or when no orbital
                // is fractionally occupied at the trial level; a guard against occupations that
                // do not add up to the electrons (fractional electron counts, where this used
                // to happen, are input errors).
                // (an earlier attempt that gets here gives way to the next mixer, which starts
                // from the initial guess again and reports the failure if it hits it too)
                let mut fermi_failure = None;
                for (n, occ) in [(nalpha, &occ_a), (nbeta, &occ_b)] {
                    let placed: f64 = occ.iter().sum();
                    if (placed - n).abs() > 1e-6 && fermi_failure.is_none() {
                        fermi_failure = Some(Error::Numerical(format!(
                            "Fermi level not found: {placed} of {n} electrons of one spin placed at {} K",
                            opts.etemp
                        )));
                    }
                }
                if let Some(e) = fermi_failure {
                    if last_attempt {
                        return Err(e);
                    }
                    break;
                }
                let ts = occupations::entropy_tblite(&occ_a, kt) + occupations::entropy_tblite(&occ_b, kt);
                for i in 0..nao {
                    focc[i] = occ_a[i] + occ_b[i];
                }
                timer.lap("scf: diagonalization");
                // the axtb conventions: the lower triangle alone (the xtb and tblite conventions
                // keep the full product, bit for bit)
                let outer = if self.params.occupied_orbitals_only { weighted_outer_sym } else { weighted_outer };
                outer(coeff.as_ref(), &focc[..kocc], &mut density, par);
                timer.lap("scf: density");

                // output moments
                mulliken::moments(bas, &ints, &density, &n0sh, &mut qsh, &mut dpat, &mut qpat, has_mp, parallel);
                qat_from_qsh(bas, &qsh, &mut qat);
                timer.lap("scf: mulliken");
                vars.pack(&qsh, &dpat, &qpat, None, &mut packed_out);
                mixer.set_residual(&packed_out);

                // energy with the output moments
                eelec.iter_mut().for_each(|x| *x = 0.0);
                // γ q of the output charges: the second-order energy, distributed over the atoms
                // of the shells
                gamma_rows(gamma, &qsh, &mut vsh_out, parallel);
                terms.isotropic_es = 0.5 * (0..nsh).fold(0.0, |e, a| e + qsh[a] * vsh_out[a]);
                for a in 0..nsh {
                    eelec[bas.sh2at[a]] += 0.5 * qsh[a] * vsh_out[a];
                }
                let mut e3 = vec![0.0; nat];
                if let Some(es3) = &self.es3 {
                    es3.energy(mol, &bas.ish_at, &qsh, &mut e3);
                }
                terms.third_order = e3.iter().sum();
                let mut eaes = vec![0.0; nat];
                if let (Some(aes), Some(c)) = (&self.aes2, mp_cache) {
                    aes.energy(mol, c, &qat, &dpat, &qpat, &mut eaes, parallel);
                }
                terms.anisotropic = eaes.iter().sum();
                let mut edsc = vec![0.0; nat];
                terms.dispersion = 0.0;
                terms.solvation = 0.0;
                for (k, (t, c)) in extra.iter().zip(caches).enumerate() {
                    let mut e = vec![0.0; nat];
                    t.energy(mol, c.as_ref(), &qat, &mut e, parallel);
                    let total = efixed[k] + e.iter().sum::<f64>();
                    match t.name() {
                        "dispersion" => terms.dispersion += total,
                        _ => terms.solvation += total,
                    }
                    for i in 0..nat {
                        edsc[i] += e[i];
                    }
                }
                terms.entropy = ts;
                for i in 0..nat {
                    eelec[i] += e3[i] + eaes[i] + edsc[i] + ts / nat as f64;
                }
                let eh0 = |nu: usize| dot(ints.h0.col_as_slice(nu), density.col_as_slice(nu));
                let eh0 = threads::map_with_min_len(nao, parallel, 16, eh0);
                for (nu, e) in eh0.into_iter().enumerate() {
                    eelec[bas.ao2at[nu]] += e;
                }

                let eiter: f64 = eelec.iter().sum();
                if !eiter.is_finite() {
                    return Err(Error::Numerical(format!("non-finite energy in SCF iteration {iscf}")));
                }
                de_last = eiter - elast;
                pnorm = rule.residual(mixer.residual());
                let it = Iteration {
                    iteration: iscf,
                    energy: eiter,
                    energy_change: de_last,
                    residual: pnorm,
                    density_change: f64::NAN,
                };
                stop = rule.update(&it) == Verdict::Done;
                if opts.trace {
                    eprintln!("scf {iscf:4} E {eiter:.12} dE {de_last:.3e} res {pnorm:.3e}");
                }
                timer.lap("scf: energy+mixing");
            }
            converged = rule.converged();
            plan.record(ScfAttempt { mixer: mk, iterations: iscf, converged });
            if converged {
                // the moments of the gradient besides the output moments
                // (`ConventionParams::scc_gradient`): the last iteration's input, and xtb 6.7.1's
                // moments as mixed once more after the last iteration
                let input = unpack(mixer.input());
                match self.params.scc_gradient {
                    SccGradient::Output => {}
                    SccGradient::OutputAes => {
                        let mut out: Moments = (qsh.clone(), vec![0.0; nat], dpat.clone(), qpat.clone());
                        qat_from_qsh(bas, &out.0, &mut out.1);
                        xtb_moments = Some((input, out));
                    }
                    SccGradient::Mixed => {
                        if mixer.next() {
                            xtb_moments = Some((input, unpack(mixer.input())));
                        }
                    }
                }
                break;
            }
        }
        if !converged {
            return Err(Error::ScfNotConverged {
                iterations: plan.iterations(),
                energy_change: de_last,
                residual: pnorm,
            });
        }
        let mixer_used = plan.used();
        for i in 0..nat {
            energies[i] += eelec[i];
        }
        terms.electronic = eelec.iter().sum();
        let energy: f64 = energies.iter().sum();

        let wdensity = if want_gradient || keep_state {
            let wocc: Vec<f64> = (0..kocc).map(|i| focc[i] * emo[i]).collect();
            let mut wdensity = Mat::<f64>::zeros(nao, nao);
            let outer = if self.params.occupied_orbitals_only { weighted_outer_sym } else { weighted_outer };
            outer(coeff.as_ref(), &wocc, &mut wdensity, par);
            wdensity
        } else {
            Mat::zeros(0, 0)
        };
        if let Some(g) = grad.as_mut() {
            match &xtb_moments {
                // xtb (scf_module.F90): the isotropic shell shifts of the last iteration's input,
                // the multipole potentials and all explicit charge derivatives from the moments
                // after the SCF (6.8.0: the output moments, 6.7.1: as mixed once more)
                Some((input, mixed)) => {
                    let mut pg = Potential {
                        vat: pot.vat.clone(),
                        vsh: pot.vsh.clone(),
                        vdp: vec![[0.0; 3]; nat],
                        vqp: vec![[0.0; 6]; nat],
                    };
                    if let (Some(aes), Some(c)) = (&self.aes2, mp_cache) {
                        let (mut v_in, mut v_mixed) = (vec![0.0; nat], vec![0.0; nat]);
                        let (mut d0, mut q0) = (vec![[0.0; 3]; nat], vec![[0.0; 6]; nat]);
                        aes.potential(mol, c, &input.1, &input.2, &input.3, &mut v_in, &mut d0, &mut q0, parallel);
                        aes.potential(
                            mol,
                            c,
                            &mixed.1,
                            &mixed.2,
                            &mixed.3,
                            &mut v_mixed,
                            &mut pg.vdp,
                            &mut pg.vqp,
                            parallel,
                        );
                        for i in 0..nat {
                            pg.vat[i] += v_mixed[i] - v_in[i];
                        }
                    } else {
                        pg.vdp.clone_from(&pot.vdp);
                        pg.vqp.clone_from(&pot.vqp);
                    }
                    self.add_scc_gradient(
                        mol, extra, &geo, &density, &wdensity, &mixed.0, &mixed.1, &mixed.2, &mixed.3, &pg, g, parallel,
                    );
                }
                None => self.add_scc_gradient(
                    mol, extra, &geo, &density, &wdensity, &qsh, &qat, &dpat, &qpat, &pot, g, parallel,
                ),
            }
            timer.lap("gradient");
        }

        let mut dipole = [0.0; 3];
        for i in 0..nat {
            for k in 0..3 {
                dipole[k] += qat[i] * mol.positions[i][k] + dpat[i][k];
            }
        }

        let orbitals = if opts.keep_orbitals {
            // one set of spatial orbitals; an open shell as alpha and beta sets of the same
            // orbitals with their own occupations
            let coefficients: Vec<f64> = (0..nao).flat_map(|k| coeff.col_as_slice(k).iter().copied()).collect();
            let sets = if nalpha == nbeta {
                vec![MoSet { spin: Spin::Alpha, energies: emo.clone(), occupations: focc.clone(), coefficients }]
            } else {
                vec![
                    MoSet {
                        spin: Spin::Alpha,
                        energies: emo.clone(),
                        occupations: occ_a.clone(),
                        coefficients: coefficients.clone(),
                    },
                    MoSet { spin: Spin::Beta, energies: emo.clone(), occupations: occ_b.clone(), coefficients },
                ]
            };
            Some(Orbitals {
                numbers: mol.numbers.clone(),
                positions: mol.positions.clone(),
                shells: (0..nat).map(|i| bas.cgto[mol.id[i]].clone()).collect(),
                sets,
            })
        } else {
            None
        };

        // per-spin levels, HOMO-LUMO gap (one set of orbitals: above the alpha HOMO, as xtb
        // and tblite print it) and bond orders (of all pairs; `Calculator` applies xtb's
        // distance cutoff in its conventions)
        let levels = if nalpha == nbeta {
            vec![OrbitalLevels { energies: emo.clone(), occupations: focc.clone() }]
        } else {
            vec![
                OrbitalLevels { energies: emo.clone(), occupations: occ_a.clone() },
                OrbitalLevels { energies: emo.clone(), occupations: occ_b.clone() },
            ]
        };
        let homo_lumo_gap = crate::properties::homo_lumo_gap(&levels[..1], nalpha, nbeta, GapRule::Alpha);
        let bond_orders = opts.bond_orders.then(|| {
            let s = ints.overlap.as_ref();
            if nalpha == nbeta {
                crate::properties::mayer_bond_orders(nat, &bas.ao2at, s, &[density.as_ref()], par)
            } else {
                let [pa, pb] = crate::properties::spin_densities(coeff.as_ref(), &occ_a, &occ_b, par);
                crate::properties::mayer_bond_orders(nat, &bas.ao2at, s, &[pa.as_ref(), pb.as_ref()], par)
            }
        });
        let state = if keep_state {
            Some(ScfState {
                geo,
                ints,
                coeff,
                emo: emo.clone(),
                occ: [occ_a, occ_b],
                efermi,
                kt,
                density,
                wdensity,
                qsh: qsh.clone(),
                qat: qat.clone(),
                dpat: dpat.clone(),
                qpat: qpat.clone(),
                pot,
            })
        } else {
            None
        };

        Ok((
            SinglePoint {
                energy,
                terms,
                gradient: grad,
                qat,
                qsh,
                dpat,
                qpat,
                dipole,
                orbital_energies: emo,
                occupations: focc,
                iterations: iscf,
                mixer_used: Some(mixer_used),
                scf_attempts: plan.into_attempts(),
                timings: timer.finish(),
                orbitals,
                levels,
                homo_lumo_gap,
                bond_orders,
                guess: start_label,
                eigensolver: solver.backend(),
            },
            state,
        ))
    }
}

/// SCF settings the SCF cannot honour fail here instead of after `max_iter` iterations, or
/// worse: a negative temperature treated as 0 K the way xtb does silently, a damping that
/// "converges" to nonsense, a NaN that reaches the eigensolver.
pub(crate) fn check_scf_options(opts: &ScfOptions) -> Result<()> {
    let acc = opts.accuracy;
    if !(acc > 0.0 && acc.is_finite()) {
        return Err(Error::Input(format!("SCF accuracy must be a positive number, got {acc}")));
    }
    if !(opts.etemp >= 0.0 && opts.etemp.is_finite()) {
        return Err(Error::Input(format!("electronic temperature must be >= 0 K, got {}", opts.etemp)));
    }
    if !(opts.damping > 0.0 && opts.damping <= 1.0) {
        return Err(Error::Input(format!("SCF damping must be in (0, 1], got {}", opts.damping)));
    }
    if opts.max_iter == 0 {
        return Err(Error::Input("the SCF needs at least 1 iteration (max_iter = 0)".into()));
    }
    threads::check(opts.threads)
}

/// Number of alpha and beta electrons of `mol` given the electron count of the neutral
/// reference atoms.
pub(crate) fn electrons(mol: &Molecule, nref: f64, nao: usize) -> Result<(f64, f64)> {
    // A fractional electron count has no Fermi level inside a gap larger than kT: the Newton
    // search leaves an integer number of electrons, which tblite returns silently as the
    // neutral result (the xtb and g-xTB programs take integer charges only).
    if mol.charge.fract() != 0.0 {
        return Err(Error::Input(format!("the charge must be an integer, got {}", mol.charge)));
    }
    let nocc = nref - mol.charge;
    let nel_int = nocc.round() as i64;
    let nuhf = match mol.uhf {
        Some(u) => {
            if (u as i64) % 2 != nel_int.rem_euclid(2) {
                return Err(Error::Input(format!("{nel_int} electrons and {u} unpaired electrons are not compatible")));
            }
            u as f64
        }
        None => nel_int.rem_euclid(2) as f64,
    };
    let nalpha = (nocc + nuhf) / 2.0;
    // Occupations that cannot exist. xtb and tblite run these anyway and return charges
    // that do not add up to the molecular charge or a different spin state (or abort);
    // without this check the filling indexes past the last orbital.
    if nocc < 0.0 {
        return Err(Error::Input(format!("charge {} leaves {nocc} electrons", mol.charge)));
    }
    if nuhf > nocc {
        return Err(Error::Input(format!("{nuhf} unpaired electrons but only {nocc} electrons")));
    }
    if nalpha > nao as f64 + 1e-9 {
        return Err(Error::Input(format!(
            "{nocc} electrons with {nuhf} unpaired need {nalpha} alpha orbitals, the basis has {nao}"
        )));
    }
    Ok((nalpha, (nocc - nuhf) / 2.0))
}

pub(crate) fn qat_from_qsh(bas: &Basis, qsh: &[f64], qat: &mut [f64]) {
    axtb_base::math::atom_charges_into(qat, bas.sh2at.iter().copied(), qsh);
}

/// `out = γ q` for the row-major square `gamma`, rows in parallel when asked (each row summed
/// in order: the same bits).
fn gamma_rows(gamma: &[f64], q: &[f64], out: &mut [f64], parallel: bool) {
    let n = q.len();
    let row = |a: usize| -> f64 { gamma[a * n..(a + 1) * n].iter().zip(q).map(|(g, x)| g * x).sum() };
    if parallel {
        out[..n].par_iter_mut().enumerate().with_min_len(32).for_each(|(a, v)| *v = row(a));
    } else {
        out[..n].iter_mut().enumerate().for_each(|(a, v)| *v = row(a));
    }
}
