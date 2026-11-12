//! Self-consistent charge extended tight-binding single point: SCF, energy and analytic
//! nuclear gradient for the GFN-type Hamiltonians.

use crate::basis::Basis;
use crate::conventions::{SccGradient, OCCUPATION_CUTOFF};
use crate::coulomb::{DampedMultipole, EffectiveCoulomb, MultipoleCache, OnsiteThirdOrder};
use crate::hamiltonian::{H0Params, Integrals, MultipoleNeglect};
use crate::linalg::{weighted_outer, weighted_outer_sym, GenEigSolver};
use crate::molden::{MoSet, Orbitals, Spin};
use crate::molecule::Molecule;
use crate::ncoord::Ncoord;
use crate::properties::{GapRule, OrbitalLevels};
use crate::repulsion::Repulsion;
use crate::scf::convergence::XtbRule;
use crate::scf::mixing::{MixerKind, MixerPlan, ScfAttempt};
use crate::scf::{
    dot, fock, mulliken, run_scf, Accelerator, FockSource, Iteration, ScfModel, ScfStrategy, ScfVariables, Stop,
};
use crate::terms::{
    self, Chain, ChainLayout, GeometryTerm, MomentTerm, Potentials, SccMoments, Slot, TermEnergy, Values,
};
use crate::{Error, Result};
use axtb_base::fd;
use axtb_base::occupations::{self, Newton};
use axtb_base::threads;
use axtb_base::timing::Laps;
use faer::{Mat, Par};
use std::ops::ControlFlow;

/// An atom-resolved, charge-dependent energy term that enters the SCF (e.g. the
/// self-consistent D4 dispersion of GFN2-xTB).
///
/// `parallel`: the method may spread its work over the current rayon pool (the caller runs
/// it inside the calculator's pool, and passes false where the call itself is one of many
/// running in parallel). The results must not depend on it.
pub(crate) trait ChargeDependentTerm: Send + Sync {
    /// Name used in energy decompositions.
    fn name(&self) -> &'static str;
    /// GeometryData-dependent setup; returns an opaque cache.
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
    /// The intermediates of the gradient (the coordination number `ncoord`) and the slot of
    /// the coordination number.
    pub layout: ChainLayout,
    pub cn: Slot,
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

/// GeometryData-dependent intermediates of one geometry (everything but the AO matrices).
pub(crate) struct GeometryData {
    /// The values of the intermediates: the coordination numbers (H0 self-energies and
    /// multipole damping radii).
    pub values: Values,
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

/// Charge potential per AO of the potentials `pot` (shell plus atom part).
pub(crate) fn vao(pot: &Potentials, bas: &Basis) -> Vec<f64> {
    (0..bas.nao).map(|mu| pot.shell[bas.ao2sh[mu]] + pot.atom[bas.ao2at[mu]]).collect()
}

/// Converged SCF wavefunction with the intermediates it was computed from: the starting
/// point of the response (coupled-perturbed SCC) and Hessian code.
pub(crate) struct ScfState {
    pub geo: GeometryData,
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
    pub pot: Potentials,
}

impl ScfState {
    /// The SCC moments of the converged SCF.
    pub fn moments(&self) -> SccMoments<'_> {
        SccMoments { qsh: &self.qsh, qat: &self.qat, dip: &self.dpat, quad: &self.qpat, mag: None }
    }
}

impl XtbModel {
    /// Coordination numbers, self-energies, Coulomb kernels and extra-term caches.
    pub(crate) fn geometry(&self, mol: &Molecule, extra: &[&dyn ChargeDependentTerm]) -> GeometryData {
        let bas = &self.basis;
        let cn = self.ncoord.compute(mol);
        let (se, dsedcn) = self.h0.selfenergy(mol, bas, &cn);
        let cutoff = self.pair_cutoff;
        let gamma = self.es2.matrix(mol, &bas.ish_at, &bas.nsh_at, bas.nsh);
        let mp_cache = self.aes2.as_ref().map(|m| m.cache(mol, &cn));
        let caches = extra.iter().map(|t| t.update(mol)).collect();
        let mut values = Values::new(&self.layout);
        values.set(self.cn, cn);
        GeometryData { values, se, dsedcn, cutoff, gamma, mp_cache, caches }
    }

    /// The geometry of the terms: `mol` with the values of the intermediates of `geo`.
    pub(crate) fn term_geometry<'a>(
        &'a self,
        mol: &'a Molecule,
        geo: &'a GeometryData,
        parallel: bool,
    ) -> terms::Geometry<'a> {
        terms::Geometry { mol, layout: &self.layout, values: &geo.values, parallel }
    }

    /// The gradient of `chain` with the chain rules of the intermediates applied.
    pub(crate) fn finish_gradient(
        &self,
        mol: &Molecule,
        geo: &GeometryData,
        chain: Chain,
        parallel: bool,
    ) -> Vec<[f64; 3]> {
        chain.finish(&self.term_geometry(mol, geo, parallel)).expect("the GFN2-xTB chain rules cannot fail")
    }

    /// Potentials of all charge-dependent terms for the given moments.
    pub(crate) fn potentials(
        &self,
        mol: &Molecule,
        extra: &[&dyn ChargeDependentTerm],
        geo: &GeometryData,
        qsh: &[f64],
        qat: &[f64],
        dpat: &[[f64; 3]],
        qpat: &[[f64; 6]],
        pot: &mut Potentials,
        parallel: bool,
    ) {
        let tg = self.term_geometry(mol, geo, parallel);
        let s = SccMoments { qsh, qat, dip: dpat, quad: qpat, mag: None };
        pot.zero();
        self.es2.potential(&tg, &geo.gamma, &s, pot);
        if let Some(es3) = &self.es3 {
            es3.potential(&tg, &(), &s, pot);
        }
        if let (Some(aes), Some(c)) = (&self.aes2, &geo.mp_cache) {
            aes.potential(&tg, c, &s, pot);
        }
        for (t, c) in extra.iter().zip(&geo.caches) {
            t.potential(mol, c.as_ref(), qat, &mut pot.atom, parallel);
        }
    }

    /// Gradient of the self-consistent part at fixed density `p`, energy-weighted density
    /// `w`, moments and potential: explicit electrostatics and extra terms and the core
    /// Hamiltonian/overlap/multipole integral derivatives, added to `chain` (its CN slot: the
    /// self-energies' dE/dCN; [`XtbModel::finish_gradient`] applies the chain rule).
    pub(crate) fn add_scc_gradient(
        &self,
        mol: &Molecule,
        extra: &[&dyn ChargeDependentTerm],
        geo: &GeometryData,
        p: &Mat<f64>,
        w: &Mat<f64>,
        qsh: &[f64],
        qat: &[f64],
        dpat: &[[f64; 3]],
        qpat: &[[f64; 6]],
        pot: &Potentials,
        chain: &mut Chain,
        parallel: bool,
    ) {
        let bas = &self.basis;
        let tg = self.term_geometry(mol, geo, parallel);
        let s = SccMoments { qsh, qat, dip: dpat, quad: qpat, mag: None };
        self.es2.gradient(&tg, &geo.gamma, &s, chain);
        if let (Some(aes), Some(c)) = (&self.aes2, &geo.mp_cache) {
            aes.gradient(&tg, c, &s, chain);
        }
        for (t, c) in extra.iter().zip(&geo.caches) {
            t.gradient(mol, c.as_ref(), qat, &mut chain.g);
        }
        let vao = vao(pot, bas);
        let (g, dedcn) = chain.g_and_slot(self.cn);
        self.h0.add_gradient(
            mol,
            bas,
            &geo.se,
            &geo.dsedcn,
            geo.cutoff,
            p,
            w,
            &vao,
            &pot.dipole,
            &pot.quadrupole,
            dedcn,
            g,
            parallel,
        );
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
        let mut chain = Chain::new(&self.layout, nat);
        let mut e = vec![0.0; nat];
        let geo = self.geometry(mol, extra);
        let mut erep = [0.0];
        let te = &mut TermEnergy { pieces: &mut erep, atoms: Some(&mut e) };
        self.repulsion.energy(&self.term_geometry(mol, &geo, parallel), &(), te, Some(&mut chain));
        for (t, c) in extra.iter().zip(&geo.caches) {
            t.fixed_energy(mol, c.as_ref(), &mut e, Some(&mut chain.g), parallel);
        }
        let mut qat = vec![0.0; nat];
        qat_from_qsh(bas, qsh, &mut qat);
        let mut pot = Potentials::new(bas.nsh, nat, false);
        self.potentials(mol, extra, &geo, qsh, &qat, dpat, qpat, &mut pot, parallel);
        self.add_scc_gradient(mol, extra, &geo, p, w, qsh, &qat, dpat, qpat, &pot, &mut chain, parallel);
        self.finish_gradient(mol, &geo, chain, parallel)
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
        let mut timer = Laps::new();
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

        let mut chain = want_gradient.then(|| Chain::new(&self.layout, nat));

        // non-self-consistent contributions
        let mut energies = vec![0.0; nat];
        let mut erep = vec![0.0; nat];
        // geometry-dependent setup
        let geo = self.geometry(mol, extra);
        let mut e_repulsion = [0.0];
        let te = &mut TermEnergy { pieces: &mut e_repulsion, atoms: Some(&mut erep) };
        self.repulsion.energy(&self.term_geometry(mol, &geo, parallel), &(), te, chain.as_mut());
        // fixed (charge-independent) energy of every extra term
        let mut efixed = vec![0.0; extra.len()];
        for (k, (t, c)) in extra.iter().zip(&geo.caches).enumerate() {
            let mut e = vec![0.0; nat];
            t.fixed_energy(mol, c.as_ref(), &mut e, chain.as_mut().map(|c| &mut c.g[..]), parallel);
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
        timer.lap("integrals");
        // xtb 6.7.1's Fock matrix leaves out small multipole integrals
        // (`ConventionParams::scc_multipole_neglect`)
        let neglect = if self.params.scc_multipole_neglect && self.aes2.is_some() {
            Some(self.h0.multipole_neglect(mol, bas, geo.cutoff, acc))
        } else {
            None
        };
        let solver = GenEigSolver::new(ints.overlap.as_ref(), par)?;
        timer.lap("cholesky");

        // SCF variables: shell charges, atomic dipoles, atomic quadrupoles (tblite, xtb 6.8.0:
        // off-diagonals scaled by sqrt(2) inside the mixer so the residual norm counts them
        // twice; xtb 6.7.1: unscaled; `ConventionParams::scc_quadrupole_scale`)
        let has_mp = self.aes2.is_some();
        let vars = ScfVariables::new(nsh, nat, has_mp, false).with_quad_offdiag_scale(self.params.scc_quadrupole_scale);
        let ndim = vars.dim();
        let kind = opts.mixer.unwrap_or(self.params.mixer);
        let mut start =
            Moments { qsh: vec![0.0; nsh], qat: vec![0.0; nat], dpat: vec![[0.0; 3]; nat], qpat: vec![[0.0; 6]; nat] };
        // where the SCF starts (`SinglePoint::guess`): a given guess of this molecule's size,
        // else EEQ charges when the options and conventions ask for them and the EEQ system
        // can be solved, else neutral atoms
        let start_label = match guess {
            Some(g) if g.qsh.len() == nsh && g.dpat.len() == nat && g.qpat.len() == nat => {
                start.qsh.copy_from_slice(&g.qsh);
                start.dpat.copy_from_slice(&g.dpat);
                start.qpat.copy_from_slice(&g.qpat);
                qat_from_qsh(bas, &start.qsh, &mut start.qat);
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
                                start.qsh[is + ish] = if tot > 0.0 { q[iat] * n0sh[is + ish] / tot } else { 0.0 };
                            }
                        }
                        qat_from_qsh(bas, &start.qsh, &mut start.qat);
                        "eeq"
                    }
                    Err(_) => "sad",
                }
            }
            _ => "sad",
        };
        // eigenvectors of the occupied orbitals only, where nothing needs all of them
        let all_orbitals = keep_state || opts.keep_orbitals || (opts.bond_orders && nalpha != nbeta);
        let lowest = self.params.occupied_orbitals_only && !all_orbitals;
        let mut scf = Gfn2Scf {
            xtb: self,
            mol,
            extra,
            geo,
            ints,
            neglect,
            solver,
            vars,
            n0sh,
            efixed,
            nalpha,
            nbeta,
            kt,
            etemp: opts.etemp,
            lowest,
            parallel,
            par,
            trace: opts.trace,
            fock: Mat::zeros(nao, nao),
            work: Mat::zeros(nao, nao),
            coeff: Mat::zeros(nao, nao),
            kocc: nao,
            density: Mat::zeros(nao, nao),
            emo: vec![0.0; nao],
            focc: vec![0.0; nao],
            occ_a: vec![0.0; nao],
            occ_b: vec![0.0; nao],
            efermi: [None, None],
            ts: 0.0,
            pot: Potentials::new(nsh, nat, false),
            vao: vec![0.0; nao],
            eelec: vec![0.0; nat],
            terms: EnergyTerms::default(),
            failure: None,
        };

        // one SCF per mixer of the plan (auto: Broyden, then DIIS from the initial guess)
        let mut plan = MixerPlan::new(kind);
        let mixers = plan.mixers();
        let mut start = Some(start);
        // the converged moments, and xtb's moments of the gradient besides them
        let mut result: Option<(Moments, Option<(Moments, Moments)>)> = None;
        let (mut iscf, mut de_last, mut pnorm) = (0, 0.0, 0.0);
        for (ia, &mk) in mixers.iter().enumerate() {
            let last_attempt = ia + 1 == mixers.len();
            if ia > 0 {
                scf.restart();
            }
            let st = if last_attempt { start.take() } else { start.clone() }.expect("the start of every attempt");
            let mut mixer = mk.mixer(ndim, opts.max_iter, opts.damping);
            // the conventions' test: the residual norm (xtb 6.7.1: per atom) and xtb 6.7.1's
            // extra iteration (`ConventionParams::scc_residual`, `scc_extra_iteration`)
            let mut rule = XtbRule::new(&self.params, acc, nat, has_mp);
            let strategy = ScfStrategy {
                accelerator: Accelerator::Moments(&mut *mixer),
                rule: &mut rule,
                max_iter: opts.max_iter,
            };
            let out = run_scf(&mut scf, strategy, st, &mut timer)?;
            // a failed mixer or a lost Fermi level ends the last attempt with its error; an
            // earlier attempt gives way to the next mixer, which starts from the initial guess
            // again and reports the failure if it hits it too
            match out.stop {
                Stop::MixerFailed if last_attempt => {
                    let name = if mk == MixerKind::Broyden { "Broyden" } else { "DIIS" };
                    return Err(Error::Numerical(format!("{name} mixing failed")));
                }
                Stop::Model if last_attempt => {
                    return Err(scf.failure.take().expect("a stopped GFN2-xTB SCF says why"));
                }
                _ => {}
            }
            (iscf, de_last, pnorm) = (out.iterations, out.last.energy_change, out.last.residual);
            plan.record(ScfAttempt { mixer: mk, iterations: iscf, converged: out.converged });
            if out.converged {
                // the moments of the gradient besides the output moments
                // (`ConventionParams::scc_gradient`): the last iteration's input, and xtb 6.7.1's
                // moments as mixed once more after the last iteration
                let input = scf.moments(mixer.input());
                let xtb_moments = match self.params.scc_gradient {
                    SccGradient::Output => None,
                    SccGradient::OutputAes => Some((input, out.state.clone())),
                    SccGradient::Mixed => mixer.next().then(|| (input, scf.moments(mixer.input()))),
                };
                result = Some((out.state, xtb_moments));
                break;
            }
        }
        let Some((Moments { qsh, qat, dpat, qpat }, xtb_moments)) = result else {
            return Err(Error::ScfNotConverged {
                iterations: plan.iterations(),
                energy_change: de_last,
                residual: pnorm,
            });
        };
        let mixer_used = plan.used();
        let Gfn2Scf {
            geo,
            ints,
            solver,
            coeff,
            kocc,
            density,
            emo,
            focc,
            occ_a,
            occ_b,
            efermi,
            pot,
            eelec,
            mut terms,
            ..
        } = scf;
        terms.repulsion = e_repulsion[0];
        let mp_cache = &geo.mp_cache;
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
        let grad = chain.map(|mut ch| {
            match &xtb_moments {
                // xtb (scf_module.F90): the isotropic shell shifts of the last iteration's input,
                // the multipole potentials and all explicit charge derivatives from the moments
                // after the SCF (6.8.0: the output moments, 6.7.1: as mixed once more)
                Some((input, mixed)) => {
                    let mut pg = Potentials {
                        shell: pot.shell.clone(),
                        atom: pot.atom.clone(),
                        dipole: vec![[0.0; 3]; nat],
                        quadrupole: vec![[0.0; 6]; nat],
                        magnetization: Vec::new(),
                    };
                    if let (Some(aes), Some(c)) = (&self.aes2, mp_cache) {
                        let tg = self.term_geometry(mol, &geo, parallel);
                        let (mut p_in, mut p_mixed) = (Potentials::new(0, nat, false), Potentials::new(0, nat, false));
                        aes.potential(&tg, c, &input.view(), &mut p_in);
                        aes.potential(&tg, c, &mixed.view(), &mut p_mixed);
                        (pg.dipole, pg.quadrupole) = (p_mixed.dipole, p_mixed.quadrupole);
                        for i in 0..nat {
                            pg.atom[i] += p_mixed.atom[i] - p_in.atom[i];
                        }
                    } else {
                        pg.dipole.clone_from(&pot.dipole);
                        pg.quadrupole.clone_from(&pot.quadrupole);
                    }
                    self.add_scc_gradient(
                        mol,
                        extra,
                        &geo,
                        &density,
                        &wdensity,
                        &mixed.qsh,
                        &mixed.qat,
                        &mixed.dpat,
                        &mixed.qpat,
                        &pg,
                        &mut ch,
                        parallel,
                    );
                }
                None => self.add_scc_gradient(
                    mol, extra, &geo, &density, &wdensity, &qsh, &qat, &dpat, &qpat, &pot, &mut ch, parallel,
                ),
            }
            let g = self.finish_gradient(mol, &geo, ch, parallel);
            timer.lap("gradient");
            g
        });

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

/// The SCC moments of an SCF iteration (GFN2-xTB's SCF state): shell charges, atomic charges,
/// atomic dipoles and quadrupoles.
#[derive(Clone, Debug)]
pub(crate) struct Moments {
    pub qsh: Vec<f64>,
    pub qat: Vec<f64>,
    pub dpat: Vec<[f64; 3]>,
    pub qpat: Vec<[f64; 6]>,
}

impl Moments {
    /// The moments as the terms read them.
    fn view(&self) -> SccMoments<'_> {
        SccMoments { qsh: &self.qsh, qat: &self.qat, dip: &self.dpat, quad: &self.qpat, mag: None }
    }
}

/// One GFN2-xTB SCF of [`XtbModel::singlepoint_state`] as the core's SCF model: the Fock
/// matrix of the mixed input moments, all or only the occupied eigenpairs, tblite's Fermi
/// filling, the density, its Mulliken moments and their energy; the moments are mixed as the
/// vector of [`ScfVariables`].
struct Gfn2Scf<'a> {
    xtb: &'a XtbModel,
    mol: &'a Molecule,
    extra: &'a [&'a dyn ChargeDependentTerm],
    geo: GeometryData,
    ints: Integrals,
    neglect: Option<MultipoleNeglect>,
    solver: GenEigSolver,
    vars: ScfVariables,
    /// Reference shell occupations; charge-independent energies of the extra terms.
    n0sh: Vec<f64>,
    efixed: Vec<f64>,
    nalpha: f64,
    nbeta: f64,
    /// kT (Eh) of the Fermi filling, and the temperature (K) for the error message.
    kt: f64,
    etemp: f64,
    /// Eigenvectors of the occupied orbitals only.
    lowest: bool,
    parallel: bool,
    par: Par,
    trace: bool,
    fock: Mat<f64>,
    work: Mat<f64>,
    /// The orbitals of the last iteration: the first `kocc` columns of `coeff` have
    /// eigenvectors.
    coeff: Mat<f64>,
    kocc: usize,
    density: Mat<f64>,
    emo: Vec<f64>,
    focc: Vec<f64>,
    occ_a: Vec<f64>,
    occ_b: Vec<f64>,
    efermi: [Option<f64>; 2],
    /// Entropy term -TS of the last occupations.
    ts: f64,
    /// Potential of the last Fock matrix.
    pot: Potentials,
    vao: Vec<f64>,
    /// Electronic energy per atom and the SCF terms of the last iteration.
    eelec: Vec<f64>,
    terms: EnergyTerms,
    /// Why the last diagonalization broke off the SCF.
    failure: Option<Error>,
}

impl Gfn2Scf<'_> {
    /// Ready for another SCF with the same overlap (the next attempt of the plan).
    fn restart(&mut self) {
        self.solver.restart();
        self.eelec.iter_mut().for_each(|x| *x = 0.0);
    }

    /// The moments of the vector `v` (in the layout of `vars`).
    fn moments(&self, v: &[f64]) -> Moments {
        let (nsh, nat) = (self.xtb.basis.nsh, self.mol.nat());
        let mut m =
            Moments { qsh: vec![0.0; nsh], qat: vec![0.0; nat], dpat: vec![[0.0; 3]; nat], qpat: vec![[0.0; 6]; nat] };
        self.unpack(v, &mut m);
        m
    }
}

impl ScfModel for Gfn2Scf<'_> {
    type State = Moments;

    fn vector_len(&self) -> usize {
        self.vars.dim()
    }

    fn pack(&self, m: &Moments, v: &mut [f64]) {
        self.vars.pack(&m.qsh, &m.dpat, &m.qpat, None, v);
    }

    fn unpack(&self, v: &[f64], m: &mut Moments) {
        // (without multipoles the dipoles and quadrupoles stay zero, as the Mulliken analysis
        // left them)
        self.vars.unpack(v, &mut m.qsh, &mut m.dpat, &mut m.qpat, None);
        qat_from_qsh(&self.xtb.basis, &m.qsh, &mut m.qat);
    }

    fn build_fock(&mut self, iter: usize, m: &Moments, laps: &mut Laps) -> Result<()> {
        let bas = &self.xtb.basis;
        let geo = &self.geo;
        self.xtb.potentials(self.mol, self.extra, geo, &m.qsh, &m.qat, &m.dpat, &m.qpat, &mut self.pot, self.parallel);
        laps.lap("scf: potentials");
        let pot = &self.pot;
        for mu in 0..bas.nao {
            self.vao[mu] = pot.shell[bas.ao2sh[mu]] + pot.atom[bas.ao2at[mu]];
        }
        fock::from_potentials(
            bas,
            &self.ints,
            &self.vao,
            &pot.dipole,
            &pot.quadrupole,
            &mut self.work,
            &mut self.fock,
            self.parallel,
        );
        if let Some(ng) = &self.neglect {
            ng.apply(self.mol, &pot.dipole, &pot.quadrupole, &mut self.fock);
        }
        if (0..bas.nao).any(|j| self.fock.col_as_slice(j).iter().any(|x| !x.is_finite())) {
            return Err(Error::Numerical(format!("non-finite Fock matrix in SCF iteration {iter}")));
        }
        laps.lap("scf: fock");
        Ok(())
    }

    fn diagonalize(&mut self, _fock: FockSource<'_>) -> Result<ControlFlow<()>> {
        let (nalpha, nbeta, kt) = (self.nalpha, self.nbeta, self.kt);
        if self.lowest {
            // the orbitals with occupation above the cutoff (from the eigenvalues)
            self.kocc = self.solver.solve_lowest(self.fock.as_ref(), &mut self.emo, &mut self.coeff, |e| {
                let (mut a, mut b) = (vec![0.0; e.len()], vec![0.0; e.len()]);
                occupations::fill_tblite(nalpha, kt, e, &mut a, Newton::XTB680);
                occupations::fill_tblite(nbeta, kt, e, &mut b, Newton::XTB680);
                (0..e.len()).rposition(|i| a[i] + b[i] > OCCUPATION_CUTOFF).map_or(0, |i| i + 1)
            })?;
        } else {
            self.solver.solve(self.fock.as_ref(), &mut self.emo, &mut self.coeff)?;
        }
        // tblite's Fermi filling (xtb 6.8.0's `fermismear`) in every convention
        self.efermi[0] = occupations::fill_tblite(nalpha, kt, &self.emo, &mut self.occ_a, Newton::XTB680);
        self.efermi[1] = occupations::fill_tblite(nbeta, kt, &self.emo, &mut self.occ_b, Newton::XTB680);
        // The Newton search for the Fermi level stops after 200 steps or when no orbital
        // is fractionally occupied at the trial level; a guard against occupations that
        // do not add up to the electrons (fractional electron counts, where this used
        // to happen, are input errors).
        for (n, occ) in [(nalpha, &self.occ_a), (nbeta, &self.occ_b)] {
            let placed: f64 = occ.iter().sum();
            if (placed - n).abs() > 1e-6 {
                self.failure = Some(Error::Numerical(format!(
                    "Fermi level not found: {placed} of {n} electrons of one spin placed at {} K",
                    self.etemp
                )));
                return Ok(ControlFlow::Break(()));
            }
        }
        self.ts = occupations::entropy_tblite(&self.occ_a, kt) + occupations::entropy_tblite(&self.occ_b, kt);
        for i in 0..self.focc.len() {
            self.focc[i] = self.occ_a[i] + self.occ_b[i];
        }
        Ok(ControlFlow::Continue(()))
    }

    fn density(&mut self) {
        // the axtb conventions: the lower triangle alone (the xtb and tblite conventions keep
        // the full product, bit for bit)
        let outer = if self.xtb.params.occupied_orbitals_only { weighted_outer_sym } else { weighted_outer };
        outer(self.coeff.as_ref(), &self.focc[..self.kocc], &mut self.density, self.par);
    }

    fn update_state(&mut self, m: &mut Moments) {
        let bas = &self.xtb.basis;
        let has_mp = self.xtb.aes2.is_some();
        mulliken::moments(
            bas,
            &self.ints,
            &self.density,
            &self.n0sh,
            &mut m.qsh,
            &mut m.dpat,
            &mut m.qpat,
            has_mp,
            self.parallel,
        );
        qat_from_qsh(bas, &m.qsh, &mut m.qat);
    }

    /// The energy of the output moments.
    fn evaluate(
        &mut self,
        iter: usize,
        m: &Moments,
        _fock: bool,
        _laps: &mut Laps,
    ) -> Result<(f64, Option<Vec<Vec<f64>>>)> {
        let (xtb, mol, parallel) = (self.xtb, self.mol, self.parallel);
        let bas = &xtb.basis;
        let (nat, nao) = (mol.nat(), bas.nao);
        let qat = &m.qat;
        let terms = &mut self.terms;
        let eelec = &mut self.eelec;
        eelec.iter_mut().for_each(|x| *x = 0.0);
        let tg = xtb.term_geometry(mol, &self.geo, parallel);
        let s = SccMoments { qsh: &m.qsh, qat, dip: &m.dpat, quad: &m.qpat, mag: None };
        // the second-order energy of the output charges, distributed over the atoms of the
        // shells (the first per-atom energy: into the zeroed `eelec`)
        let mut piece = [0.0];
        xtb.es2.energy(&tg, &self.geo.gamma, &s, &mut TermEnergy { pieces: &mut piece, atoms: Some(eelec) });
        terms.isotropic_es = piece[0];
        let mut e3 = vec![0.0; nat];
        let mut piece = [0.0];
        if let Some(es3) = &xtb.es3 {
            es3.energy(&tg, &(), &s, &mut TermEnergy { pieces: &mut piece, atoms: Some(&mut e3) });
        }
        terms.third_order = piece[0];
        let mut eaes = vec![0.0; nat];
        let mut piece = [0.0];
        if let (Some(aes), Some(c)) = (&xtb.aes2, &self.geo.mp_cache) {
            aes.energy(&tg, c, &s, &mut TermEnergy { pieces: &mut piece, atoms: Some(&mut eaes) });
        }
        terms.anisotropic = piece[0];
        let eelec = &mut self.eelec;
        let mut edsc = vec![0.0; nat];
        terms.dispersion = 0.0;
        terms.solvation = 0.0;
        for (k, (t, c)) in self.extra.iter().zip(&self.geo.caches).enumerate() {
            let mut e = vec![0.0; nat];
            t.energy(mol, c.as_ref(), qat, &mut e, parallel);
            let total = self.efixed[k] + e.iter().sum::<f64>();
            match t.name() {
                "dispersion" => terms.dispersion += total,
                _ => terms.solvation += total,
            }
            for i in 0..nat {
                edsc[i] += e[i];
            }
        }
        let ts = self.ts;
        terms.entropy = ts;
        for i in 0..nat {
            eelec[i] += e3[i] + eaes[i] + edsc[i] + ts / nat as f64;
        }
        let (h0, density) = (&self.ints.h0, &self.density);
        let eh0 = |nu: usize| dot(h0.col_as_slice(nu), density.col_as_slice(nu));
        let eh0 = threads::map_with_min_len(nao, parallel, 16, eh0);
        for (nu, e) in eh0.into_iter().enumerate() {
            eelec[bas.ao2at[nu]] += e;
        }

        let energy: f64 = eelec.iter().sum();
        if !energy.is_finite() {
            return Err(Error::Numerical(format!("non-finite energy in SCF iteration {iter}")));
        }
        Ok((energy, None))
    }

    fn record(&mut self, it: &Iteration) {
        if self.trace {
            let (i, e, de, res) = (it.iteration, it.energy, it.energy_change, it.residual);
            eprintln!("scf {i:4} E {e:.12} dE {de:.3e} res {res:.3e}");
        }
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
