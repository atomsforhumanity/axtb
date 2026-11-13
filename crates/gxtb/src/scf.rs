// SPDX-License-Identifier: LGPL-3.0-or-later
//! g-xTB 2.0.1 single points: the geometry-dependent setup, the energy and Fock matrices of a
//! density, and tblite's SCF as the 2.0.1 program runs it (spec §4.17; a model of the core's
//! SCF driver, `axtb::scf::run_scf`):
//!
//! - start: EEQ-BC atomic charges partitioned to shells, q_l = n0_l/n0_A q_A; no density and no
//!   multipoles in iteration 1;
//! - iteration k: Fock matrix F_k of the current charges, multipoles and density (of P_{k-1});
//!   mixing; diagonalization of the mixed matrix (generalized, S); occupations: 0 K Aufbau (the
//!   last orbital of a channel takes the fractional remainder of its electron count) or, at a
//!   finite electronic temperature, Fermi smearing ([`crate::fermi`]); P_k; its Mulliken charges
//!   and moments; `E_k = E[P_k] - TS_k`;
//! - mixing on the full Fock matrix (both spin channels in one vector): none in iteration 1,
//!   damping 0.2 in iterations 2 and 3, then DIIS with the residuals r_j = F_j - F~_{j-1} of
//!   iterations 3..k (at most 7), coefficients from the bordered B matrix; DIIS starts one
//!   iteration early after an energy change |E_k - E_{k-1}| below 1e-12 Eh (tblite's mixer
//!   stages; e.g. H2, whose density is fixed from iteration 1;
//!   `Gxtb201FockDiis`);
//! - convergence after iteration k once |E_j - E_{j-1}| < 1e-6 acc Eh in some iteration j <= k
//!   and the RMS of r_j over the full vector < 2e-5 acc in some iteration 1 < j <= k (tblite's
//!   `check_convergence` sets each flag and never clears it); no extra iteration
//!   (`Gxtb201Rule`).

use crate::basis::Basis;
use crate::cn::gxtb_cn;
use crate::coulomb::{AesCache, IsotropicCache, Shell};
use crate::data::v2_0_1::{REF_OCC, ZEFF_SHELL};
use crate::eeqbc::eeqbc;
use crate::eeqbc::EeqBcCharges;
use crate::exchange::Exchange;
use crate::fock::shell_list;
use crate::h0::{add_acp, anisotropy_par, build_h0, mirror_add_anisotropy, AcpSet};
use crate::intermediates::Slots;
use crate::model::Model;
use crate::options::{ScfOptions, Version};
use crate::repulsion::Repulsion;
use crate::revd4::RevD4;
use crate::strategy::{Gxtb201FockDiis, Gxtb201Rule};
use crate::terms::{buffer, Terms};
use crate::timing::{Laps, Timings};
use axtb::linalg::GenEigSolver;
use axtb::scf::{run_scf, Accelerator, FockMixer, FockSource, Iteration, ScfModel, ScfStrategy};
use axtb::terms::{ChainLayout, Geometry, GeometryTerm, MomentTerm, SccMoments, TermEnergy, Values};
use axtb::{Error, Molecule, Result};
use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatMut, MatRef, Par};
use std::ops::ControlFlow;
use std::sync::Arc;

/// Everything that depends on the geometry only.
#[derive(Clone, Debug)]
pub struct Setup {
    pub mol: Molecule,
    pub model: Model,
    pub eeq: EeqBcCharges,
    /// g-xTB coordination numbers.
    pub cn: Vec<f64>,
    /// The SCF and H0 bases (the AO maps of the SCF basis: `basis.scf.ao2sh`, `ao2at`).
    pub basis: Basis,
    /// Integral pair cutoff (bohr; infinite in the default conventions).
    pub cutoff: f64,
    /// (atom, shell, l) per shell of the SCF basis ([`crate::fock::shell_list`]).
    pub shells: Vec<Shell>,
    pub nao: usize,
    /// Overlap, dipole and quadrupole integrals of the SCF basis (around the atom of the
    /// first AO index).
    pub s: Vec<f64>,
    pub dpint: [Vec<f64>; 3],
    pub qpint: [Vec<f64>; 6],
    /// One-electron Hamiltonian H0 + anisotropy + ACP.
    pub h: Vec<f64>,
    /// Anisotropy vectors a_A.
    pub aniso: Vec<[f64; 3]>,
    /// Reference shell occupations n0_l.
    pub ref_occ: Vec<f64>,
    /// The isotropic and anisotropic electrostatics.
    pub isotropic: IsotropicCache,
    pub aes: AesCache,
    pub rep: Repulsion,
    pub disp: RevD4,
    pub exch: Exchange,
    pub e_incr: f64,
    pub e_atm: f64,
    /// v1.1's semi-classical repulsion (0 in 2.0.1, whose repulsion is self-consistent).
    pub e_rep: f64,
    /// v1.1 in the program's conventions: what turns the exact Fock matrix into the program's.
    pub(crate) program: Option<crate::program110::ProgramFock>,
    /// Electrons (total, alpha, beta; fractional in program mode) and spin channels.
    pub nel: f64,
    pub nalpha: f64,
    pub nbeta: f64,
    pub nspin: usize,
    /// Threads of the SCF, gradient and Hessians of this setup ([`Setup::set_threads`]): the
    /// parallelism of the dense linear algebra, and whether the setup's own loops run on the
    /// current rayon pool (in fixed chunks summed in order: the results do not depend on the
    /// thread count).
    pub(crate) threads: usize,
    pub(crate) par: Par,
    pub(crate) parallel: bool,
    /// Wall time of the setup phases (taken by the first single point that reports them).
    pub timings: Timings,
    /// The intermediates of the gradient (the g-xTB, basis and revD4 CNs, the EEQ-BC charges
    /// and the effective charges), their slots and their values at this geometry (the charges,
    /// CNs and effective charges above).
    pub layout: ChainLayout,
    pub(crate) slots: Slots,
    pub values: Values,
    /// The terms (their caches are the fields above).
    pub(crate) terms: Terms,
}

/// Energy terms of one density (Eh), named after the program's containers.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Energies {
    /// The version whose terms these are (their names and sums differ).
    pub version: Version,
    /// Tr(H P), H = H0 + anisotropy + ACP.
    pub one_electron: f64,
    /// First order (v1.1: its "ES1", with the reference-charge cross term).
    pub first_order: f64,
    /// Second order (2.0.1: including the reference-charge cross term).
    pub second_order: f64,
    pub third_order: f64,
    pub fourth_order: f64,
    pub aes: f64,
    /// MFX + bond-order correction + onsite exchange (v1.1: MFX).
    pub exchange: f64,
    /// v1.1's onsite exchange (0 in 2.0.1, whose onsite exchange is in `exchange`).
    pub exchange_onsite: f64,
    pub spin: f64,
    /// revD4 two-body (self-consistent).
    pub dispersion: f64,
    /// Repulsion (2.0.1: self-consistent; v1.1: semi-classical, outside `electronic`).
    pub repulsion: f64,
    /// Sum of the self-consistent terms (the program's "electronic energy"; v1.1: without
    /// dispersion and repulsion, as the 1.1.0 program prints it).
    pub electronic: f64,
    pub increments: f64,
    /// revD4 three-body (the program's pre-SCF "dispersion energy").
    pub dispersion_three_body: f64,
    /// Electronic entropy term -TS at a finite electronic temperature (<= 0; included in
    /// `electronic` and `total`).
    pub entropy: f64,
    pub total: f64,
}

impl Energies {
    /// Add the entropy term -TS of the occupations (to `electronic` and `total`).
    pub fn add_entropy(&mut self, ts: f64) {
        self.entropy += ts;
        self.electronic += ts;
        self.total += ts;
    }
}

/// Density-dependent quantities: spin densities, Mulliken shell and atomic charges, atomic
/// dipoles and quadrupoles (total density), shell magnetizations (open shell).
#[derive(Clone, Debug)]
pub struct State {
    pub p: Vec<Vec<f64>>,
    pub q: Vec<f64>,
    pub qa: Vec<f64>,
    pub dip: Vec<[f64; 3]>,
    pub qp: Vec<[f64; 6]>,
    pub mag: Option<Vec<f64>>,
}

impl State {
    /// A start of a v1.1 SCF: the spin densities (closed shell: the total density), whose
    /// charges and moments the SCF computes at its geometry ([`Setup::scf_from`] reads only
    /// the densities of a v1.1 start).
    pub fn densities(p: Vec<Vec<f64>>) -> Self {
        Self { p, q: Vec::new(), qa: Vec::new(), dip: Vec::new(), qp: Vec::new(), mag: None }
    }

    /// The SCC moments as the terms read them.
    pub fn moments(&self) -> SccMoments<'_> {
        SccMoments { qsh: &self.q, qat: &self.qa, dip: &self.dip, quad: &self.qp, mag: self.mag.as_deref() }
    }
}

impl Setup {
    /// Geometry-dependent setup of `mol` (positions in bohr, integer charge) for `model` (the
    /// version and conventions), on one thread.
    pub fn new(mol: &Molecule, model: Model) -> Result<Self> {
        Self::with_threads(mol, model, 1)
    }

    /// [`Setup::new`] on `threads` threads of the current rayon pool (the caller runs it inside a
    /// pool of that size): the setup's loops in parallel, its dense linear algebra from
    /// `axtb_base::threads::DENSE_MIN_AO` basis functions on, and so those of its SCF, gradient
    /// and Hessians ([`Setup::set_threads`]). Zero threads is an input error.
    pub fn with_threads(mol: &Molecule, model: Model, threads: usize) -> Result<Self> {
        let parallel = threads > 1;
        crate::check_charge(mol)?;
        crate::model::check_model_elements(&model, &mol.numbers)?;
        let mut t = Laps::new();
        let (eeq, eeq_solution) = eeqbc(&model, mol)?.charges_and_solution()?;
        t.lap("setup: eeq-bc");
        let cn = gxtb_cn(&model, mol);
        let basis = Basis::new(&model, mol, &eeq.charges)?;
        let cutoff = basis.pair_cutoff(&model);
        t.lap("setup: cn+basis");
        let shells = shell_list(&basis.scf);
        let nao = basis.scf.nao;
        axtb_base::threads::check(threads)?;
        let dense = axtb_base::threads::dense_par(threads, nao);
        let (s, dpint, qpint) = crate::basis::integrals(mol, &basis.scf, cutoff, parallel);
        t.lap("setup: integrals");
        let mut h = build_h0(&model, mol, &basis.h0, &cn, cutoff, parallel);
        t.lap("setup: h0");
        add_acp(mol, &basis.scf, &AcpSet::new(&model, mol, &cn, cutoff), &mut h, dense, parallel);
        t.lap("setup: acp");
        let aniso = if model.anisotropy() {
            let aniso = anisotropy_par(&model, mol, parallel);
            mirror_add_anisotropy(nao, &basis.scf.ao2at, &dpint, &aniso, &mut h, parallel);
            aniso
        } else {
            crate::parallel::mirror_lower(&mut h, nao, parallel);
            vec![[0.0; 3]; mol.nat()]
        };
        t.lap("setup: anisotropy");
        let ref_occ: Vec<f64> = match model.version {
            Version::V2_0_1 => shells.iter().map(|&(iat, ish, _)| REF_OCC[mol.numbers[iat] - 1][ish]).collect(),
            Version::V1_1 => shells
                .iter()
                .map(|&(iat, ish, _)| crate::data::runtime::REF_SHELL_OCC[mol.numbers[iat] - 1][ish])
                .collect(),
        };
        let isotropic = IsotropicCache::new_par(&model, mol, &shells, &cn, parallel);
        let aes = AesCache::new_par(&model, mol, &cn, parallel);
        let rep = Repulsion::new_par(&model, mol, &cn, parallel);
        t.lap("setup: coulomb+repulsion");
        let disp = RevD4::new_par(&model, mol, parallel)?;
        // the intermediates; the EEQ-BC charges keep the factorized equations the charges were
        // solved from: the gradients at this geometry (the analytic gradient, the Jacobian rows
        // of the Hessian) reuse them instead of factorizing again
        let (layout, slots) = crate::intermediates::layout(&model, mol, Some(Arc::new(eeq_solution)));
        let mut values = Values::new(&layout);
        values.set(slots.eeq, eeq.charges.clone());
        values.set(slots.cnd4, disp.cn.clone());
        values.set(slots.cnb, basis.cn.clone());
        values.set(slots.cn, cn.clone());
        values.set(slots.qeff, basis.qeff.clone());
        values.set(slots.qeff0, basis.qeff_h0.clone());
        let terms = Terms::new(&model, &shells, &slots);
        let tg = Geometry { mol, layout: &layout, values: &values, parallel };
        t.lap("setup: revd4");
        let mut e_atm = [0.0];
        terms.disp.fixed_energy(&tg, &disp, &mut TermEnergy { pieces: &mut e_atm, atoms: None }, None);
        let e_atm = e_atm[0];
        t.lap("setup: revd4 three-body");
        let exch = Exchange::new(&model, mol, &shells, &basis.scf.ao2sh, parallel);
        t.lap("setup: exchange");
        let mut e_incr = [0.0];
        terms.increments.energy(&tg, &(), &mut TermEnergy { pieces: &mut e_incr, atoms: None }, None);
        let e_incr = e_incr[0];
        let mut e_rep = [0.0];
        if let Some(c) = &terms.classical {
            c.energy(&tg, &(), &mut TermEnergy { pieces: &mut e_rep, atoms: None }, None);
        }
        let e_rep = e_rep[0];
        let program =
            (model.version == Version::V1_1 && model.program()).then(|| crate::program110::ProgramFock::new(&aes));
        let (nel, nalpha, nbeta, nspin) = match model.version {
            Version::V2_0_1 => electrons_v2_0_1(&model, mol, &shells, &ref_occ, nao)?,
            Version::V1_1 => electrons_v1_1(mol, &shells, nao)?,
        };
        Ok(Self {
            mol: mol.clone(),
            model,
            eeq,
            cn,
            basis,
            cutoff,
            shells,
            nao,
            s,
            dpint,
            qpint,
            h,
            aniso,
            ref_occ,
            isotropic,
            aes,
            rep,
            disp,
            exch,
            e_incr,
            e_atm,
            e_rep,
            program,
            nel,
            nalpha,
            nbeta,
            nspin,
            threads,
            par: dense,
            parallel,
            timings: t.laps,
            layout,
            slots,
            values,
            terms,
        })
    }
}

/// 2.0.1's electrons (total, alpha, beta) and spin channels: program mode the stored reference
/// occupations, otherwise the Aufbau valence electrons (integer).
fn electrons_v2_0_1(
    model: &Model,
    mol: &Molecule,
    shells: &[Shell],
    ref_occ: &[f64],
    nao: usize,
) -> Result<(f64, f64, f64, usize)> {
    {
        let nel = if model.program() {
            ref_occ.iter().sum::<f64>() - mol.charge
        } else {
            let zval: f64 = shells.iter().map(|&(iat, ish, _)| ZEFF_SHELL[mol.numbers[iat] - 1][ish]).sum();
            let n = zval - mol.charge;
            if !(n.is_finite() && (n - n.round()).abs() <= 1e-8) {
                return Err(Error::Input(format!("g-xTB 2.0.1: {n} electrons (non-integer charge)")));
            }
            n.round()
        };
        if nel.is_nan() || nel < 0.0 {
            return Err(Error::Input(format!("g-xTB 2.0.1: {nel} electrons")));
        }
        let nround = nel.round() as usize;
        let uhf = mol.uhf.unwrap_or(nround % 2);
        if uhf > nround || !(nround - uhf).is_multiple_of(2) {
            return Err(Error::Input(format!("g-xTB 2.0.1: {nround} electrons with {uhf} unpaired")));
        }
        let (nalpha, nbeta) = (0.5 * (nel + uhf as f64), 0.5 * (nel - uhf as f64));
        if nalpha.ceil() as usize > nao {
            return Err(Error::Input("g-xTB 2.0.1: more occupied orbitals than basis functions".into()));
        }
        let nspin = if uhf > 0 { 2 } else { 1 };
        Ok((nel, nalpha, nbeta, nspin))
    }
}

/// v1.1's electrons (total, alpha, beta) and spin channels: the valence electrons of the
/// Aufbau shell occupations (`com::zeff_shell`) less the charge, in both conventions.
fn electrons_v1_1(mol: &Molecule, shells: &[Shell], nao: usize) -> Result<(f64, f64, f64, usize)> {
    let zval: f64 = shells
        .iter()
        .map(|&(iat, ish, _)| crate::data::v1_1::ZEFF_SHELL[ish * 103 + mol.numbers[iat] - 1] as f64)
        .sum();
    let nelf = zval - mol.charge;
    if !((nelf - nelf.round()).abs() <= 1e-8 && nelf >= 0.0) {
        return Err(Error::Input(format!("g-xTB: {nelf} electrons")));
    }
    let nel = nelf.round() as usize;
    let uhf = mol.uhf.unwrap_or(nel % 2);
    if uhf > nel || !(nel - uhf).is_multiple_of(2) {
        return Err(Error::Input(format!("g-xTB: {nel} electrons with {uhf} unpaired")));
    }
    let nbeta = (nel - uhf) / 2;
    let nalpha = nel - nbeta;
    if nalpha > nao {
        return Err(Error::Input("g-xTB: more occupied orbitals than basis functions".into()));
    }
    let nspin = if nalpha == nbeta { 1 } else { 2 };
    Ok((nel as f64, nalpha as f64, nbeta as f64, nspin))
}

impl Setup {
    /// The geometry of the terms and intermediates: the molecule with the values of the
    /// intermediates, the setup's loops in parallel when it runs on more than one thread.
    pub fn term_geometry(&self) -> Geometry<'_> {
        Geometry { mol: &self.mol, layout: &self.layout, values: &self.values, parallel: self.parallel }
    }

    /// Run the SCF, gradient and Hessians of this setup on `threads` threads of the current
    /// rayon pool (the caller runs them inside a pool of that size): the loops in parallel for
    /// more than one thread, the dense linear algebra from `axtb_base::threads::DENSE_MIN_AO`
    /// basis functions on ([`axtb_base::threads::dense_par`]). The Hessians run their
    /// displacements in parallel instead, each sequentially. Zero threads is an input error.
    pub fn set_threads(&mut self, threads: usize) -> Result<()> {
        axtb_base::threads::check(threads)?;
        self.par = axtb_base::threads::dense_par(threads, self.nao);
        self.threads = threads;
        self.parallel = threads > 1;
        Ok(())
    }

    /// The threads of [`Setup::set_threads`].
    pub fn threads(&self) -> usize {
        self.threads
    }

    /// Mulliken shell populations of a density ([`crate::mulliken::shell_populations`]).
    pub fn shell_populations(&self, p: &[f64]) -> Vec<f64> {
        crate::mulliken::shell_populations(&self.basis.scf, &self.s, p, self.parallel)
    }

    /// Atomic dipoles and traceless quadrupoles of the total density.
    pub fn atomic_moments(&self, p: &[f64]) -> (Vec<[f64; 3]>, Vec<[f64; 6]>) {
        let (_, dip, qp) = self.mulliken(p);
        (dip, qp)
    }

    /// Shell populations, atomic dipoles and quadrupoles of a density (2.0.1's order,
    /// [`crate::mulliken::moments_by_rows`]).
    #[allow(clippy::type_complexity)]
    fn mulliken(&self, p: &[f64]) -> (Vec<f64>, Vec<[f64; 3]>, Vec<[f64; 6]>) {
        crate::mulliken::moments_by_rows(&self.basis.scf, &self.s, &self.dpint, &self.qpint, p, self.parallel)
    }

    /// Total density of spin densities `p` ([`crate::fock::total_density`]).
    pub fn total_density(&self, p: &[Vec<f64>]) -> Vec<f64> {
        crate::fock::total_density(p)
    }

    /// The total density of spin densities `p`, borrowed for a closed shell.
    pub fn total_density_ref<'a>(&self, p: &'a [Vec<f64>]) -> std::borrow::Cow<'a, [f64]> {
        if p.len() == 1 {
            std::borrow::Cow::Borrowed(&p[0])
        } else {
            std::borrow::Cow::Owned(crate::fock::total_density(p))
        }
    }

    /// Mulliken analysis of spin densities `p` (closed shell: one matrix, the total density).
    pub fn state(&self, p: &[Vec<f64>]) -> State {
        self.state_of(p.to_vec())
    }

    /// [`Setup::state`], taking the densities.
    pub fn state_of(&self, p: Vec<Vec<f64>>) -> State {
        let ptot = self.total_density_ref(&p);
        let (pop, dip, qp) = self.mulliken(&ptot);
        drop(ptot);
        let q: Vec<f64> = self.ref_occ.iter().zip(&pop).map(|(r, x)| r - x).collect();
        let qa = self.isotropic.atom_charges(&q);
        let mag = crate::mulliken::magnetizations(&self.basis.scf, &self.s, &p, self.parallel);
        State { p, q, qa, dip, qp, mag }
    }

    /// v1.1: the 1.1.0 program's start density at this geometry (spin densities; closed shell:
    /// the total density), from which its SCF starts in both conventions.
    pub fn start_density(&self) -> Vec<Vec<f64>> {
        crate::program110::guess_density(self)
    }

    /// The program's start (`eeqbc_guess`): EEQ-BC charges partitioned to the shells by the
    /// reference occupations, no density, no multipoles.
    pub fn guess_state(&self) -> State {
        let nat = self.mol.nat();
        let mut n0 = vec![0.0; nat];
        for (k, &(iat, _, _)) in self.shells.iter().enumerate() {
            n0[iat] += self.ref_occ[k];
        }
        let q: Vec<f64> = self
            .shells
            .iter()
            .enumerate()
            .map(|(k, &(iat, _, _))| self.ref_occ[k] / n0[iat] * self.eeq.charges[iat])
            .collect();
        let qa = self.isotropic.atom_charges(&q);
        let n = self.nao;
        State {
            p: vec![vec![0.0; n * n]; self.nspin],
            q,
            qa,
            dip: vec![[0.0; 3]; nat],
            qp: vec![[0.0; 6]; nat],
            mag: if self.nspin == 2 { Some(vec![0.0; self.shells.len()]) } else { None },
        }
    }

    /// Energies and (with `fock`) the Fock matrices of a state.
    pub fn evaluate(&self, st: &State, fock: bool) -> (Energies, Vec<Vec<f64>>) {
        self.evaluate_timed(st, fock, &mut Laps::new())
    }

    /// [`Setup::evaluate`] with its phases timed in `t`.
    pub fn evaluate_timed(&self, st: &State, fock: bool, t: &mut Laps) -> (Energies, Vec<Vec<f64>>) {
        let n = self.nao;
        let mol = &self.mol;
        t.skip();
        let ptot = self.total_density_ref(&st.p);
        let e_h = crate::parallel::sum(n, self.parallel, |i| {
            let r = i * n..(i + 1) * n;
            self.h[r.clone()].iter().zip(&ptot[r]).map(|(a, b)| a * b).sum::<f64>()
        });
        drop(ptot);
        // the terms' energies (the dispersion's potential from the same pass)
        let (tg, s, terms) = (self.term_geometry(), st.moments(), &self.terms);
        let (mut eiso, mut e_aes, mut e_spin, mut e_disp, mut e_rep) = ([0.0; 4], [0.0], [0.0], [0.0], [0.0]);
        terms.iso.energy(&tg, &self.isotropic, &s, &mut totals(&mut eiso));
        terms.aes.energy(&tg, &self.aes, &s, &mut totals(&mut e_aes));
        terms.spin.energy(&tg, &(), &s, &mut totals(&mut e_spin));
        let (nsh, nat) = (self.shells.len(), mol.nat());
        let mut pdisp = buffer(&terms.disp, nsh, nat);
        terms.disp.energy_and_potential(&tg, &self.disp, &s, &mut totals(&mut e_disp), &mut pdisp);
        terms.rep.energy(&tg, &self.rep, &s, &mut totals(&mut e_rep));
        let ([e1, e2, e3, e4], [e_aes], [e_spin], [e_disp], [e_rep]) = (eiso, e_aes, e_spin, e_disp, e_rep);
        t.lap("scf: fock energies");
        let pr = self.exch.products(&self.s, &st.p, self.par);
        let (e_x, e_on, fx, vx) = self.exch.evaluate_parts(&self.s, &st.p, &pr, &st.q, fock, self.par, self.parallel);
        t.lap("scf: fock exchange");
        let energies = match self.model.version {
            Version::V2_0_1 => {
                let electronic = e_h + e1 + e2 + e3 + e4 + e_aes + e_x + e_spin + e_disp + e_rep;
                Energies {
                    version: Version::V2_0_1,
                    one_electron: e_h,
                    first_order: e1,
                    second_order: e2,
                    third_order: e3,
                    fourth_order: e4,
                    aes: e_aes,
                    exchange: e_x,
                    exchange_onsite: 0.0,
                    spin: e_spin,
                    dispersion: e_disp,
                    repulsion: e_rep,
                    electronic,
                    increments: self.e_incr,
                    dispersion_three_body: self.e_atm,
                    entropy: 0.0,
                    total: self.e_incr + self.e_atm + electronic,
                }
            }
            Version::V1_1 => {
                // the 1.1.0 program's sums: dispersion and repulsion after the electronic energy
                let electronic = e_h + e1 + e2 + e3 + e4 + e_aes + e_x + e_on + e_spin;
                Energies {
                    version: Version::V1_1,
                    one_electron: e_h,
                    first_order: e1,
                    second_order: e2,
                    third_order: e3,
                    fourth_order: e4,
                    aes: e_aes,
                    exchange: e_x,
                    exchange_onsite: e_on,
                    spin: e_spin,
                    dispersion: e_disp,
                    repulsion: self.e_rep,
                    electronic,
                    increments: self.e_incr,
                    dispersion_three_body: self.e_atm,
                    entropy: 0.0,
                    total: electronic + self.e_incr + e_disp + self.e_atm + self.e_rep,
                }
            }
        };
        if !fock {
            return (energies, Vec::new());
        }
        t.skip();
        // the shell potential: the isotropic one plus (in this order) exchange, AES, repulsion
        // and dispersion of the shell's atom (the program's sum)
        let (mut piso, mut paes) = (buffer(&terms.iso, nsh, nat), buffer(&terms.aes, nsh, nat));
        let (mut prep, mut pspin) = (buffer(&terms.rep, nsh, nat), buffer(&terms.spin, nsh, nat));
        terms.iso.potential(&tg, &self.isotropic, &s, &mut piso);
        match &self.program {
            None => terms.aes.potential(&tg, &self.aes, &s, &mut paes),
            Some(pf) => {
                // the program's AES potentials: those of E_AES'
                let (vs, vd, vq) = pf.aes_potentials(mol, s.qat, s.dip, s.quad, self.parallel);
                (paes.atom, paes.dipole, paes.quadrupole) = (vs, vd, vq);
            }
        }
        terms.rep.potential(&tg, &self.rep, &s, &mut prep);
        let mut vsh = piso.shell;
        for (ish, &(iat, _, _)) in self.shells.iter().enumerate() {
            vsh[ish] += vx[ish] + paes.atom[iat] + prep.atom[iat] + pdisp.atom[iat];
        }
        let (vd, vq) = (paes.dipole, paes.quadrupole);
        let vao: Vec<f64> = self.basis.scf.ao2sh.iter().map(|&ish| vsh[ish]).collect();
        // spin potential: +1/2 (w_m + w_n) S on alpha, - on beta
        let wao: Option<Vec<f64>> = st.mag.is_some().then(|| {
            terms.spin.potential(&tg, &(), &s, &mut pspin);
            self.basis.scf.ao2sh.iter().map(|&ish| pspin.magnetization[ish]).collect()
        });
        let mut fm = crate::fock::from_potentials_fused(
            n,
            &self.h,
            &self.s,
            &self.basis.scf.ao2at,
            &self.dpint,
            &self.qpint,
            &vao,
            &vd,
            &vq,
            wao.as_deref(),
            &fx,
            self.parallel,
        );
        if let Some(pf) = &self.program {
            pf.add_ofx_extra(self, &pr, &mut fm);
        }
        t.lap("scf: fock potentials");
        (energies, fm)
    }

    /// Occupation numbers of a channel with `nel` (real) electrons over `n` orbitals at 0 K:
    /// 1 for the first floor(nel), the remainder for the next.
    pub fn aufbau(nel: f64, n: usize) -> Vec<f64> {
        crate::fermi::aufbau(nel, n)
    }

    /// Occupations per spin channel at 0 K (closed shell: doubly occupied).
    pub fn occupations(&self) -> Vec<Vec<f64>> {
        let n = self.nao;
        if self.nspin == 1 {
            let a = Self::aufbau(self.nalpha, n);
            let b = Self::aufbau(self.nbeta, n);
            vec![a.iter().zip(&b).map(|(x, y)| x + y).collect()]
        } else {
            vec![Self::aufbau(self.nalpha, n), Self::aufbau(self.nbeta, n)]
        }
    }

    /// kT (Eh) of an electronic temperature (K) in these conventions (0 at 0 K; v1.1's program
    /// smears above 1e-6 K with its single-precision kB).
    pub fn kt(&self, etemp: f64) -> f64 {
        if self.model.version == Version::V1_1 && self.model.program() {
            return if etemp > crate::fermi::TEL_MIN { etemp * crate::occupations::KB_GXTB110 } else { 0.0 };
        }
        if etemp > 0.0 {
            etemp * crate::fermi::boltzmann(self.model.program())
        } else {
            0.0
        }
    }

    /// Occupations, Fermi level and entropy term of one channel with `nel` electrons over the
    /// levels `e` at `kt` in this model (v1.1: its rules, [`crate::fermi::fill_channel_v1_1`]).
    pub fn fill_channel(&self, nel: f64, kt: f64, e: &[f64]) -> (Vec<f64>, Option<f64>, f64) {
        match self.model.version {
            Version::V2_0_1 => crate::fermi::fill_channel(nel, kt, e, self.model.program()),
            Version::V1_1 => crate::fermi::fill_channel_v1_1(nel as usize, kt, e, self.model.program()),
        }
    }

    /// Occupations per spin channel for the orbital energies `eps` (one channel per spin
    /// channel of the calculation) at `kt`, the Fermi levels per electron spin (alpha, beta)
    /// and the entropy term -TS. Closed shell: both electron spins fill the same levels and the
    /// occupations are summed. At kT = 0 the Aufbau occupations.
    pub fn fill(&self, eps: &[Vec<f64>], kt: f64) -> (Vec<Vec<f64>>, [Option<f64>; 2], f64) {
        let n = self.nao;
        if self.model.version == Version::V1_1 && self.model.program() {
            // the program's eigenvalue cap at N_alpha + 10
            let na = self.nalpha as usize;
            let (ea, eb) = (
                crate::program110::cap(&eps[0][..n], na),
                crate::program110::cap(&eps[if self.nspin == 1 { 0 } else { 1 }][..n], na),
            );
            let (a, mua, tsa) = self.fill_channel(self.nalpha, kt, &ea);
            let (b, mub, tsb) = self.fill_channel(self.nbeta, kt, &eb);
            return if self.nspin == 1 {
                (vec![a.iter().zip(&b).map(|(x, y)| x + y).collect()], [mua, mub], tsa + tsb)
            } else {
                (vec![a, b], [mua, mub], tsa + tsb)
            };
        }
        if self.nspin == 1 {
            let (a, mua, tsa) = self.fill_channel(self.nalpha, kt, &eps[0][..n]);
            let (b, mub, tsb) = self.fill_channel(self.nbeta, kt, &eps[0][..n]);
            (vec![a.iter().zip(&b).map(|(x, y)| x + y).collect()], [mua, mub], tsa + tsb)
        } else {
            let (a, mua, tsa) = self.fill_channel(self.nalpha, kt, &eps[0][..n]);
            let (b, mub, tsb) = self.fill_channel(self.nbeta, kt, &eps[1][..n]);
            (vec![a, b], [mua, mub], tsa + tsb)
        }
    }
}

/// sum_s C_s diag(w_s) C_s^T over the orbitals with nonzero weight, as one matrix product
/// (row-major n x n coefficients, column k = orbital k): the density (weights: occupations)
/// or the energy-weighted density (occupations times orbital energies) of the channels.
pub fn weighted_density(n: usize, channels: &[(&[f64], &[f64])], par: Par) -> Vec<f64> {
    let cols: Vec<(usize, usize)> = channels
        .iter()
        .enumerate()
        .flat_map(|(s, (_, w))| (0..n).filter(|&k| w[k] != 0.0).map(move |k| (s, k)))
        .collect();
    let m = cols.len();
    let mut out = vec![0.0; n * n];
    if m == 0 {
        return out;
    }
    let mut cw = vec![0.0; n * m];
    let mut cc = vec![0.0; n * m];
    for i in 0..n {
        for (kk, &(s, k)) in cols.iter().enumerate() {
            let (c, w) = channels[s];
            let v = c[i * n + k];
            cc[i * m + kk] = v;
            cw[i * m + kk] = w[k] * v;
        }
    }
    matmul(
        MatMut::from_row_major_slice_mut(&mut out, n, n),
        Accum::Replace,
        MatRef::from_row_major_slice(&cw, n, m),
        MatRef::from_row_major_slice(&cc, n, m).transpose(),
        1.0,
        par,
    );
    out
}

/// The 2.0.1 program's SCF iteration limit.
pub const MAX_ITER: usize = 250;
/// The 2.0.1 program's damping factor (iterations 2 and 3) and number of stored DIIS vectors.
pub const DAMPING: f64 = 0.2;
pub const DIIS_SIZE: usize = 7;
/// Basis size up to which v1.1's SCF in the default conventions computes every eigenpair (the
/// full solver is cheaper than the occupied eigenvectors alone for such small bases).
const FULL_SOLVE_MAX_AO: usize = 16;

/// Converged SCF.
#[derive(Clone, Debug)]
pub struct ScfResult {
    pub energies: Energies,
    pub shell_charges: Vec<f64>,
    pub atom_charges: Vec<f64>,
    /// Orbital energies (eigenvalues of the last mixed Fock matrix) and occupations per spin
    /// channel (closed shell: one, doubly occupied).
    pub orbital_energies: Vec<Vec<f64>>,
    pub occupations: Vec<Vec<f64>>,
    /// kT (Eh) of the occupations (0: Aufbau) and the Fermi levels the occupations of the
    /// alpha and beta electrons were computed with (`None` at 0 K; closed shell: both fill
    /// the one channel).
    pub kt: f64,
    pub efermi: [Option<f64>; 2],
    /// MO coefficients per channel (row-major n x n, column k = orbital k): every orbital's with
    /// `all_orbitals`, otherwise only those of the orbitals with nonzero occupation (the other
    /// columns zero).
    pub coefficients: Vec<Vec<f64>>,
    /// Whether `coefficients` hold every orbital ([`crate::ScfOptions::all_orbitals`]; the
    /// Hessians need them).
    pub all_orbitals: bool,
    /// Converged spin densities (closed shell: total density).
    pub density: Vec<Vec<f64>>,
    pub state: State,
    /// The state the last Fock matrix was built from (P of the previous iteration, or the
    /// start); the program evaluates the potential terms of its gradient with it.
    pub input_state: State,
    /// SCF iterations (diagonalizations) and the energy of every iteration.
    pub iterations: usize,
    pub history: Vec<(f64, f64)>,
    /// Wall time of the SCF phases.
    pub timings: Timings,
    /// The eigensolver of the SCF (`axtb::linalg::GenEigSolver::backend`).
    pub eigensolver: String,
}

impl Setup {
    /// Self-consistent field as the 2.0.1 program runs it (see the module documentation) with
    /// the settings `opts`, whose conventions must be the setup's.
    pub fn scf(&self, opts: &ScfOptions) -> Result<ScfResult> {
        self.scf_from(opts, None)
    }

    /// The state an SCF of this setup can start from: `start` with its atomic charges
    /// recomputed from its shell charges (as the program does after reading `xtbrestart`), or
    /// an error if its dimensions do not fit the molecule.
    pub fn restart_state(&self, start: &State) -> Result<State> {
        let (n, nsh, nat) = (self.nao, self.shells.len(), self.mol.nat());
        let fits = start.p.len() == self.nspin
            && start.p.iter().all(|p| p.len() == n * n)
            && start.q.len() == nsh
            && start.dip.len() == nat
            && start.qp.len() == nat
            && start.mag.as_ref().is_none_or(|m| m.len() == nsh);
        if !fits {
            return Err(Error::Input("g-xTB 2.0.1: the start wavefunction does not fit this molecule".into()));
        }
        let mut st = start.clone();
        st.qa = self.isotropic.atom_charges(&st.q);
        if self.nspin == 2 && st.mag.is_none() {
            st.mag = Some(vec![0.0; nsh]);
        }
        if self.nspin == 1 {
            st.mag = None;
        }
        Ok(st)
    }

    /// [`Setup::scf`] from the state `start` (e.g. the converged state of a previous geometry,
    /// as the program's `xtbrestart` and its in-run restarts carry it: shell charges, atomic
    /// dipoles and quadrupoles, shell magnetizations and the density matrices, taken over as
    /// they are although the q-vSZP basis of this geometry differs) instead of the EEQ-BC
    /// start; `None`: the program's start.
    pub fn scf_from(&self, opts: &ScfOptions, start: Option<&State>) -> Result<ScfResult> {
        let version = self.model.version;
        opts.check(version)?;
        if opts.conventions != self.model.conventions {
            return Err(Error::Input(format!(
                "{}: SCF options in the {:?} conventions for a setup in the {:?} conventions",
                version.label(),
                opts.conventions,
                self.model.conventions
            )));
        }
        let n = self.nao;
        let ns = self.nspin;
        let max_iter = opts.iteration_limit(version);
        let kt = self.kt(opts.etemp);
        let mut t = Laps::new();
        let solver = GenEigSolver::new(MatRef::from_row_major_slice(&self.s, n, n), self.par)?;
        t.lap("scf: cholesky");
        let st = match (start, version) {
            (Some(s), Version::V2_0_1) => self.restart_state(s)?,
            (None, Version::V2_0_1) => self.guess_state(),
            // v1.1: the densities of the start (the program's `gxtbrestart` carries them), or the
            // program's start density, analysed at this geometry
            (Some(s), Version::V1_1) => {
                if s.p.len() != ns || s.p.iter().any(|x| x.len() != n * n) {
                    return Err(Error::Input("g-xTB: the start density does not fit this molecule".into()));
                }
                self.state(&s.p)
            }
            (None, Version::V1_1) => self.state_of(crate::program110::guess_density(self)),
        };
        // v1.1 in the program's conventions: every eigenpair, as the program (and its cap) needs;
        // in the default conventions also for the smallest bases, where the full solver is the
        // cheaper one
        let every = version == Version::V1_1 && (self.model.program() || n <= FULL_SOLVE_MAX_AO);
        let mut scf = Scf201 {
            setup: self,
            solver,
            kt,
            occ: self.occupations(),
            efermi: [None, None],
            ts: 0.0,
            c: Mat::zeros(n, n),
            eps: vec![vec![0.0; n]; ns],
            coef: vec![vec![0.0; n * n]; ns],
            full: vec![false; ns],
            pnew: Vec::new(),
            input: None,
            energies: Energies::default(),
            history: Vec::new(),
            every,
            last: (version == Version::V1_1 && opts.all_orbitals && !every).then(Vec::new),
            dp: (version == Version::V1_1).then_some(f64::NAN),
        };
        let mut v2_mixer: Option<Gxtb201FockDiis> = None;
        let (mut v1_mixer, mut v2_rule, mut v1_rule) = (None, None, None);
        let strategy = match version {
            Version::V2_0_1 => {
                let (damping, diis_size) = (opts.damping.unwrap_or(DAMPING), opts.diis_size.unwrap_or(DIIS_SIZE));
                let ethr = opts.energy_tol.unwrap_or(1e-6 * opts.accuracy);
                let pthr = opts.residual_tol.unwrap_or(2e-5 * opts.accuracy);
                let mixer = v2_mixer.insert(Gxtb201FockDiis::new(n, damping, diis_size, self.parallel));
                let rule = v2_rule.insert(Gxtb201Rule::new(ethr, pthr));
                ScfStrategy { accelerator: Accelerator::Fock(mixer), rule, max_iter }
            }
            Version::V1_1 => {
                // the 1.1.0 program's `diisfock_` and convergence test (both conventions)
                let (ethr, pthr, ndamp, ndiis) = if ns == 1 { (1e-8, 2e-5, 3, 7) } else { (4e-7, 5e-5, 4, 8) };
                let ethr = opts.energy_tol.unwrap_or(ethr * opts.accuracy);
                let pthr = opts.density_tol.unwrap_or(pthr * opts.accuracy);
                let ndamp = opts.transition_iteration.unwrap_or(ndamp);
                let ndiis = opts.diis_size.unwrap_or(ndiis);
                let damping = opts.damping.unwrap_or(crate::options::DAMPING_V1_1);
                let mixer = v1_mixer.insert(crate::program110::strategy::Gxtb110FockMixer::new(
                    n,
                    damping,
                    ndamp,
                    ndiis,
                    start.is_some(),
                ));
                let rule = v1_rule.insert(crate::program110::strategy::Gxtb110Rule::new(ethr, pthr));
                ScfStrategy { accelerator: Accelerator::Fock(mixer), rule, max_iter }
            }
        };
        let out = run_scf(&mut scf, strategy, st, &mut t)?;
        if !out.converged {
            let (de, res) = match (version, out.iterations) {
                (Version::V1_1, _) => (out.last.energy_change, out.last.density_change),
                (_, 0 | 1) => (f64::NAN, f64::NAN),
                _ => (out.last.energy_change, out.last.residual),
            };
            return Err(Error::ScfNotConverged { iterations: max_iter, energy_change: de, residual: res });
        }
        let Scf201 { mut solver, occ, efermi, mut c, mut eps, mut coef, full, input, energies, history, last, .. } =
            scf;
        if opts.all_orbitals && !every {
            // every orbital, the occupied ones as in the iteration (same bits)
            for s in 0..ns {
                let f = match &last {
                    Some(last) => &last[s],
                    None => v2_mixer.as_ref().expect("2.0.1 diagonalizes the mixed Fock matrices").mixed(s),
                };
                solver.solve_all_with(MatRef::from_row_major_slice(f, n, n), &mut eps[s], &mut c, full[s])?;
                MatMut::from_row_major_slice_mut(&mut coef[s], n, n).copy_from(&c);
            }
        }
        let st = out.state;
        Ok(ScfResult {
            energies,
            shell_charges: st.q.clone(),
            atom_charges: st.qa.clone(),
            orbital_energies: eps,
            occupations: occ,
            kt,
            efermi,
            coefficients: coef,
            all_orbitals: opts.all_orbitals || every,
            density: st.p.clone(),
            state: st,
            input_state: input.expect("a converged SCF ran an iteration"),
            iterations: out.iterations,
            history,
            timings: t.laps,
            eigensolver: solver.backend(),
        })
    }
}

/// One 2.0.1 SCF of [`Setup::scf_from`] as the core's SCF model (the program's iteration,
/// see the module documentation): the occupied eigenpairs of the mixed Fock matrices, the
/// Aufbau or Fermi occupations, the densities, their state and energy with the Fock matrices
/// of the next iteration.
struct Scf201<'a> {
    setup: &'a Setup,
    solver: GenEigSolver,
    kt: f64,
    /// Occupations per channel (at 0 K the Aufbau ones throughout), Fermi levels and the
    /// entropy term -TS of the last iteration.
    occ: Vec<Vec<f64>>,
    efermi: [Option<f64>; 2],
    ts: f64,
    c: Mat<f64>,
    /// Orbital energies and coefficients per channel (row-major; the columns of the orbitals
    /// without occupation zero) of the last iteration.
    eps: Vec<Vec<f64>>,
    coef: Vec<Vec<f64>>,
    /// Whether the last diagonalization of each channel used the full solver.
    full: Vec<bool>,
    /// The densities of the last iteration, until its state takes them.
    pnew: Vec<Vec<f64>>,
    /// The state the last Fock matrices were built from (the program's gradient evaluates
    /// its potential terms with it).
    input: Option<State>,
    /// Energies of the last iteration, and every iteration's energy and Fock residual.
    energies: Energies,
    history: Vec<(f64, f64)>,
    /// Every eigenpair in every iteration (v1.1 in the program's conventions).
    every: bool,
    /// The Fock matrices of the last diagonalization (v1.1 with every orbital: its last
    /// iteration diagonalizes the unmixed matrix, which the mixer does not hold).
    last: Option<Vec<Vec<f64>>>,
    /// The RMS change of the packed spin densities in the last iteration (v1.1's convergence
    /// test; 2.0.1 measures the Fock residual instead).
    dp: Option<f64>,
}

impl ScfModel for Scf201<'_> {
    type State = State;

    fn diagonalize(&mut self, fock: FockSource<'_>) -> Result<ControlFlow<Error>> {
        let setup = self.setup;
        let (n, ns, kt) = (setup.nao, setup.nspin, self.kt);
        if self.every {
            for s in 0..ns {
                let f = fock.matrix(s).expect("the v1.1 SCF diagonalizes row-major Fock matrices");
                self.solver.solve(MatRef::from_row_major_slice(f, n, n), &mut self.eps[s], &mut self.c)?;
                self.full[s] = true;
                MatMut::from_row_major_slice_mut(&mut self.coef[s], n, n).copy_from(&self.c);
            }
            return Ok(ControlFlow::Continue(()));
        }
        if let Some(last) = &mut self.last {
            *last = (0..ns).map(|s| fock.matrix(s).expect("row-major Fock matrices").to_vec()).collect();
        }
        let occ = &self.occ;
        // the eigenvectors of the orbitals with nonzero occupation (their occupations at a
        // finite temperature follow from the eigenvalues)
        for s in 0..ns {
            let f = fock.matrix(s).expect("the 2.0.1 SCF diagonalizes mixed Fock matrices");
            let nel = if s == 0 { setup.nalpha } else { setup.nbeta };
            let need = |e: &[f64]| -> usize {
                let o = if kt > 0.0 {
                    let (a, _, _) = setup.fill_channel(nel, kt, e);
                    if ns == 1 {
                        let (b, _, _) = setup.fill_channel(setup.nbeta, kt, e);
                        a.iter().zip(&b).map(|(x, y)| x + y).collect()
                    } else {
                        a
                    }
                } else {
                    occ[s].clone()
                };
                o.iter().rposition(|&x| x != 0.0).map_or(0, |i| i + 1)
            };
            let k =
                self.solver.solve_lowest(MatRef::from_row_major_slice(f, n, n), &mut self.eps[s], &mut self.c, need)?;
            self.full[s] = self.solver.used_full();
            let mut cm = MatMut::from_row_major_slice_mut(&mut self.coef[s], n, n);
            cm.as_mut().subcols_mut(0, k).copy_from(self.c.as_ref().subcols(0, k));
            cm.subcols_mut(k, n - k).fill(0.0);
        }
        Ok(ControlFlow::Continue(()))
    }

    fn density(&mut self) {
        let setup = self.setup;
        if self.kt > 0.0 {
            (self.occ, self.efermi, self.ts) = setup.fill(&self.eps, self.kt);
        }
        let (coef, occ) = (&self.coef, &self.occ);
        self.pnew = (0..setup.nspin).map(|s| weighted_density(setup.nao, &[(&coef[s], &occ[s])], setup.par)).collect();
    }

    fn update_state(&mut self, st: &mut State) {
        if self.dp.is_some() {
            // v1.1: RMS change of the packed spin densities
            let n = self.setup.nao;
            let mut dp2 = 0.0;
            for (pn, po) in self.pnew.iter().zip(&st.p) {
                for i in 0..n {
                    for j in 0..=i {
                        let d = pn[i * n + j] - po[i * n + j];
                        dp2 += d * d;
                    }
                }
            }
            self.dp = Some((dp2 / (n * (n + 1) / 2) as f64).sqrt());
        }
        let new = self.setup.state_of(std::mem::take(&mut self.pnew));
        self.input = Some(std::mem::replace(st, new));
    }

    fn density_change(&self) -> f64 {
        self.dp.unwrap_or(f64::NAN)
    }

    fn evaluate(
        &mut self,
        _iter: usize,
        st: &State,
        fock: bool,
        laps: &mut Laps,
    ) -> Result<(f64, Option<Vec<Vec<f64>>>)> {
        let (mut energies, f) = self.setup.evaluate_timed(st, fock, laps);
        energies.add_entropy(self.ts);
        let e = energies.total;
        self.energies = energies;
        Ok((e, fock.then_some(f)))
    }

    fn record(&mut self, it: &Iteration) {
        self.history.push((it.energy, it.residual));
    }
}

/// A term's energy as totals only (g-xTB sums no per-atom energies).
fn totals(pieces: &mut [f64]) -> TermEnergy<'_> {
    TermEnergy { pieces, atoms: None }
}
