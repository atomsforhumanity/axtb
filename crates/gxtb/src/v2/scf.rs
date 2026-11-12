//! g-xTB 2.0.1 single points: the geometry-dependent setup, the energy and Fock matrices of a
//! density, and tblite's SCF as the 2.0.1 program runs it (spec §4.17):
//!
//! - start: EEQ-BC atomic charges partitioned to shells, q_l = n0_l/n0_A q_A; no density and no
//!   multipoles in iteration 1;
//! - iteration k: Fock matrix F_k of the current charges, multipoles and density (of P_{k-1});
//!   mixing; diagonalization of the mixed matrix (generalized, S); occupations: 0 K Aufbau (the
//!   last orbital of a channel takes the fractional remainder of its electron count) or, at a
//!   finite electronic temperature, Fermi smearing ([`super::fermi`]); P_k; its Mulliken charges
//!   and moments; `E_k = E[P_k] - TS_k`;
//! - mixing on the full Fock matrix (both spin channels in one vector): none in iteration 1,
//!   damping 0.2 in iterations 2 and 3, then DIIS with the residuals r_j = F_j - F~_{j-1} of
//!   iterations 3..k (at most 7), coefficients from the bordered B matrix; DIIS starts one
//!   iteration early after an energy change |E_k - E_{k-1}| below 1e-12 Eh (tblite's mixer
//!   stages; e.g. H2, whose density is fixed from iteration 1);
//! - convergence after iteration k once |E_j - E_{j-1}| < 1e-6 acc Eh in some iteration j <= k
//!   and the RMS of r_j over the full vector < 2e-5 acc in some iteration 1 < j <= k (tblite's
//!   `check_convergence` sets each flag and never clears it); no extra iteration.

use super::basis::Basis;
use super::cn::gxtb_cn;
use super::coulomb::{spin_energy, spin_potential, Coulomb, Shell};
use super::disp::RevD4;
use super::eeqbc::eeqbc;
use super::exchange::Exchange;
use super::h0::{add_acp, anisotropy_par, build_h0, mirror_add_anisotropy};
use super::repulsion::Repulsion;
use super::Model;
use crate::calculator::Version;
use crate::data_v2::{INCREMENT, REF_OCC, ZEFF_SHELL};
use crate::eeqbc::{EeqBcCharges, EeqBcSolution};
use crate::fock::shell_list;
use crate::scf::{Conventions, ScfOptions};
use crate::timing::{Laps, Timings};
use axtb::{Error, Molecule, Result};
use faer::linalg::matmul::matmul;
use faer::{Accum, Mat, MatMut, MatRef, Par};
use std::sync::Arc;

/// Everything that depends on the geometry only.
#[derive(Clone, Debug)]
pub struct Setup {
    pub mol: Molecule,
    pub model: Model,
    pub eeq: EeqBcCharges,
    /// The factorized EEQ-BC equations the charges `eeq` were solved from: the 2.0.1 gradient
    /// uses the same kernel, so the gradients at this geometry (the analytic gradient, the
    /// Jacobian rows of the Hessian) reuse them instead of factorizing again.
    pub(crate) eeq_solution: Arc<EeqBcSolution>,
    /// g-xTB coordination numbers.
    pub cn: Vec<f64>,
    pub basis: Basis,
    /// Integral pair cutoff (bohr; infinite in the default conventions).
    pub cutoff: f64,
    pub shells: Vec<Shell>,
    pub ao_shell: Vec<usize>,
    pub ao_atom: Vec<usize>,
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
    pub coulomb: Coulomb,
    pub rep: Repulsion,
    pub disp: RevD4,
    pub exch: Exchange,
    pub e_incr: f64,
    pub e_atm: f64,
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
}

/// Energy terms of one density (Eh), named after the program's containers.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Energies {
    /// Tr(H P), H = H0 + anisotropy + ACP.
    pub one_electron: f64,
    pub first_order: f64,
    /// Second order including the reference-charge cross term.
    pub second_order: f64,
    pub third_order: f64,
    pub fourth_order: f64,
    pub aes: f64,
    /// MFX + bond-order correction + onsite exchange.
    pub exchange: f64,
    pub spin: f64,
    /// revD4 two-body (self-consistent).
    pub dispersion: f64,
    /// Repulsion (self-consistent).
    pub repulsion: f64,
    /// Sum of the self-consistent terms (the program's "electronic energy").
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

impl Setup {
    /// Geometry-dependent setup of `mol` (positions in bohr, integer charge), on one thread.
    pub fn new(mol: &Molecule, conventions: Conventions) -> Result<Self> {
        Self::with_threads(mol, conventions, 1)
    }

    /// [`Setup::new`] on `threads` threads of the current rayon pool (the caller runs it inside a
    /// pool of that size): the setup's loops in parallel, its dense linear algebra from
    /// `axtb_base::threads::DENSE_MIN_AO` basis functions on, and so those of its SCF, gradient
    /// and Hessians ([`Setup::set_threads`]). Zero threads is an input error.
    pub fn with_threads(mol: &Molecule, conventions: Conventions, threads: usize) -> Result<Self> {
        let model = Model::new(conventions);
        let parallel = threads > 1;
        crate::check_charge(mol)?;
        super::check_elements(&mol.numbers)?;
        let mut t = Laps::new();
        let (eeq, eeq_solution) = eeqbc(&model, mol)?.charges_and_solution()?;
        t.lap("setup: eeq-bc");
        let cn = gxtb_cn(&model, mol);
        let basis = Basis::new(&model, mol, &eeq.charges)?;
        let cutoff = basis.pair_cutoff(&model);
        t.lap("setup: cn+basis");
        let shells = shell_list(&basis.scf);
        let ao_shell = basis.scf.ao2sh.clone();
        let ao_atom = basis.scf.ao2at.clone();
        let nao = basis.scf.nao;
        axtb_base::threads::check(threads)?;
        let dense = axtb_base::threads::dense_par(threads, nao);
        let (s, dpint, qpint) = super::basis::integrals(mol, &basis.scf, cutoff, parallel);
        t.lap("setup: integrals");
        let mut h = build_h0(&model, mol, &basis.h0, &cn, cutoff, parallel);
        t.lap("setup: h0");
        add_acp(mol, &basis.scf, cutoff, &mut h, dense, parallel);
        t.lap("setup: acp");
        let aniso = anisotropy_par(&model, mol, parallel);
        mirror_add_anisotropy(nao, &ao_atom, &dpint, &aniso, &mut h, parallel);
        t.lap("setup: anisotropy");
        let ref_occ: Vec<f64> = shells.iter().map(|&(iat, ish, _)| REF_OCC[mol.numbers[iat] - 1][ish]).collect();
        let coulomb = Coulomb::new_par(&model, mol, &shells, &cn, parallel);
        let rep = Repulsion::new_par(&model, mol, &cn, parallel);
        t.lap("setup: coulomb+repulsion");
        let disp = RevD4::new_par(&model, mol, parallel)?;
        t.lap("setup: revd4");
        let e_atm = disp.three_body_par(&model, mol, None, parallel).0;
        t.lap("setup: revd4 three-body");
        let exch = Exchange::new(&model, mol, &shells, &ao_shell, parallel);
        t.lap("setup: exchange");
        let e_incr: f64 = mol.numbers.iter().map(|&z| INCREMENT[z - 1]).sum();
        // electrons: program mode the stored reference occupations, otherwise the Aufbau
        // valence electrons (integer)
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
        Ok(Self {
            mol: mol.clone(),
            model,
            eeq,
            eeq_solution: Arc::new(eeq_solution),
            cn,
            basis,
            cutoff,
            shells,
            ao_shell,
            ao_atom,
            nao,
            s,
            dpint,
            qpint,
            h,
            aniso,
            ref_occ,
            coulomb,
            rep,
            disp,
            exch,
            e_incr,
            e_atm,
            nel,
            nalpha,
            nbeta,
            nspin,
            threads,
            par: dense,
            parallel,
            timings: t.laps,
        })
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
        let qa = self.coulomb.atom_charges(&q);
        let mag = crate::mulliken::magnetizations(&self.basis.scf, &self.s, &p, self.parallel);
        State { p, q, qa, dip, qp, mag }
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
        let qa = self.coulomb.atom_charges(&q);
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
        let par = self.parallel;
        let [e1, e2, e3, e4] = self.coulomb.isotropic_energies(&st.q, par);
        let e_aes = self.coulomb.aes_energy(mol, &st.qa, &st.dip, &st.qp, par);
        let e_spin = st.mag.as_ref().map_or(0.0, |m| spin_energy(mol, &self.shells, m));
        let (e_disp, vdisp) = self.disp.two_body_par(&st.qa, par);
        let e_rep = self.rep.energy_par(&st.qa, par);
        t.lap("scf: fock energies");
        let pr = self.exch.products(&self.s, &st.p, self.par);
        let (e_x, fx, vx) = self.exch.evaluate_with(&self.s, &st.p, &pr, &st.q, fock, self.par, self.parallel);
        t.lap("scf: fock exchange");
        let electronic = e_h + e1 + e2 + e3 + e4 + e_aes + e_x + e_spin + e_disp + e_rep;
        let energies = Energies {
            one_electron: e_h,
            first_order: e1,
            second_order: e2,
            third_order: e3,
            fourth_order: e4,
            aes: e_aes,
            exchange: e_x,
            spin: e_spin,
            dispersion: e_disp,
            repulsion: e_rep,
            electronic,
            increments: self.e_incr,
            dispersion_three_body: self.e_atm,
            entropy: 0.0,
            total: self.e_incr + self.e_atm + electronic,
        };
        if !fock {
            return (energies, Vec::new());
        }
        t.skip();
        let mut vsh = self.coulomb.isotropic_potential(&st.q, par);
        let (vs, vd, vq) = self.coulomb.aes_potentials(mol, &st.qa, &st.dip, &st.qp, par);
        let vrep = self.rep.potential_par(&st.qa, par);
        for (ish, &(iat, _, _)) in self.shells.iter().enumerate() {
            vsh[ish] += vx[ish] + vs[iat] + vrep[iat] + vdisp[iat];
        }
        let vao: Vec<f64> = self.ao_shell.iter().map(|&ish| vsh[ish]).collect();
        // spin potential: +1/2 (w_m + w_n) S on alpha, - on beta
        let wao: Option<Vec<f64>> = st
            .mag
            .as_ref()
            .map(|m| spin_potential(mol, &self.shells, m))
            .map(|w| self.ao_shell.iter().map(|&ish| w[ish]).collect());
        let fm = crate::fock::from_potentials_fused(
            n,
            &self.h,
            &self.s,
            &self.ao_atom,
            &self.dpint,
            &self.qpint,
            &vao,
            &vd,
            &vq,
            wao.as_deref(),
            &fx,
            self.parallel,
        );
        t.lap("scf: fock potentials");
        (energies, fm)
    }

    /// Occupation numbers of a channel with `nel` (real) electrons over `n` orbitals at 0 K:
    /// 1 for the first floor(nel), the remainder for the next.
    pub fn aufbau(nel: f64, n: usize) -> Vec<f64> {
        super::fermi::aufbau(nel, n)
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

    /// kT (Eh) of an electronic temperature (K) in these conventions (0 at 0 K).
    pub fn kt(&self, etemp: f64) -> f64 {
        if etemp > 0.0 {
            etemp * super::fermi::boltzmann(self.model.program())
        } else {
            0.0
        }
    }

    /// Occupations per spin channel for the orbital energies `eps` (one channel per spin
    /// channel of the calculation) at `kt`, the Fermi levels per electron spin (alpha, beta)
    /// and the entropy term -TS. Closed shell: both electron spins fill the same levels and the
    /// occupations are summed. At kT = 0 the Aufbau occupations.
    pub fn fill(&self, eps: &[Vec<f64>], kt: f64) -> (Vec<Vec<f64>>, [Option<f64>; 2], f64) {
        let program = self.model.program();
        let n = self.nao;
        if self.nspin == 1 {
            let (a, mua, tsa) = super::fermi::fill_channel(self.nalpha, kt, &eps[0][..n], program);
            let (b, mub, tsb) = super::fermi::fill_channel(self.nbeta, kt, &eps[0][..n], program);
            (vec![a.iter().zip(&b).map(|(x, y)| x + y).collect()], [mua, mub], tsa + tsb)
        } else {
            let (a, mua, tsa) = super::fermi::fill_channel(self.nalpha, kt, &eps[0][..n], program);
            let (b, mub, tsb) = super::fermi::fill_channel(self.nbeta, kt, &eps[1][..n], program);
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

/// Energy change (Eh) after which the program's SCF switches from damping to DIIS before
/// iteration 4 (the threshold of tblite's DIIS mixer stage; independent of the accuracy).
const DIIS_SWITCH_ENERGY: f64 = 1e-12;

/// The 2.0.1 program's SCF iteration limit.
pub const MAX_ITER: usize = 250;
/// The 2.0.1 program's damping factor (iterations 2 and 3) and number of stored DIIS vectors.
pub const DAMPING: f64 = 0.2;
pub const DIIS_SIZE: usize = 7;

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
    /// Whether `coefficients` hold every orbital ([`crate::ScfOptions::keep_orbitals`]; the
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

/// Solve the bordered DIIS equations with an unblocked LU with partial pivoting (LAPACK
/// `getf2` order) and return the coefficients.
fn diis_coefficients(b: &[Vec<f64>]) -> Option<Vec<f64>> {
    let m = b.len();
    let n = m + 1;
    let mut a = vec![0.0; n * n];
    for i in 0..m {
        for j in 0..m {
            a[i * n + j] = b[i][j];
        }
        a[i * n + m] = 1.0;
        a[m * n + i] = 1.0;
    }
    let mut rhs = vec![0.0; n];
    rhs[m] = 1.0;
    let mut piv = vec![0usize; n];
    for k in 0..n {
        let mut p = k;
        let mut amax = a[k * n + k].abs();
        for i in k + 1..n {
            if a[i * n + k].abs() > amax {
                amax = a[i * n + k].abs();
                p = i;
            }
        }
        piv[k] = p;
        if a[p * n + k] == 0.0 {
            return None;
        }
        if p != k {
            for j in 0..n {
                a.swap(k * n + j, p * n + j);
            }
        }
        let inv = 1.0 / a[k * n + k];
        for i in k + 1..n {
            a[i * n + k] *= inv;
        }
        for i in k + 1..n {
            let l = a[i * n + k];
            if l != 0.0 {
                for j in k + 1..n {
                    a[i * n + j] -= l * a[k * n + j];
                }
            }
        }
    }
    for k in 0..n {
        if piv[k] != k {
            rhs.swap(k, piv[k]);
        }
    }
    for i in 0..n {
        let mut s = rhs[i];
        for j in 0..i {
            s -= a[i * n + j] * rhs[j];
        }
        rhs[i] = s;
    }
    for i in (0..n).rev() {
        let mut s = rhs[i];
        for j in i + 1..n {
            s -= a[i * n + j] * rhs[j];
        }
        rhs[i] = s / a[i * n + i];
    }
    if rhs.iter().any(|x| !x.is_finite()) {
        return None;
    }
    Some(rhs[..m].to_vec())
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
        st.qa = self.coulomb.atom_charges(&st.q);
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
        opts.check(Version::V2_0_1)?;
        if opts.conventions != self.model.conventions {
            return Err(Error::Input(format!(
                "g-xTB 2.0.1: SCF options in the {:?} conventions for a setup in the {:?} conventions",
                opts.conventions, self.model.conventions
            )));
        }
        let n = self.nao;
        let ns = self.nspin;
        let max_iter = opts.iteration_limit(Version::V2_0_1);
        let (damping, diis_size) = (opts.damping.unwrap_or(DAMPING), opts.diis_size.unwrap_or(DIIS_SIZE));
        let kt = self.kt(opts.etemp);
        let mut t = Laps::new();
        let mut solver = axtb::linalg::GenEigSolver::new(MatRef::from_row_major_slice(&self.s, n, n), self.par)?;
        t.lap("scf: cholesky");
        let mut occ = self.occupations();
        let mut efermi = [None, None];
        let ethr = opts.energy_tol.unwrap_or(1e-6 * opts.accuracy);
        let pthr = opts.residual_tol.unwrap_or(2e-5 * opts.accuracy);
        let len = ns * n * n;
        let mut st = match start {
            Some(s) => self.restart_state(s)?,
            None => self.guess_state(),
        };
        let (_, mut fock) = self.evaluate_timed(&st, true, &mut t);
        let mut fmix: Vec<f64> = Vec::new();
        let mut hist: Vec<(Vec<f64>, Vec<f64>)> = Vec::new();
        // the DIIS matrix B_ij = r_i . r_j of the stored residuals
        let mut bmat: Vec<Vec<f64>> = Vec::new();
        let par = self.parallel;
        let dot = |a: &[f64], b: &[f64]| crate::parallel::sum(a.len(), par, |k| a[k] * b[k]);
        let mut e_old = 0.0;
        let mut c = Mat::<f64>::zeros(n, n);
        let mut eps = vec![vec![0.0; n]; ns];
        let mut coef = vec![vec![0.0; n * n]; ns];
        // whether the last diagonalization of each spin used the full solver
        let mut full = vec![false; ns];
        let mut history = Vec::new();
        // the program's convergence flags are sticky: each criterion counts once met in any
        // iteration (tblite `check_convergence` only ever sets them)
        let (mut econverged, mut pconverged) = (false, false);
        // DIIS replaces the damping after iteration 3, or after an iteration whose energy
        // changed by less than 1e-12 Eh (tblite_scf_iterator `update` switches mixer stages at
        // the stage's start iteration or energy threshold: 4 and 1e-12, for every accuracy and
        // electronic temperature)
        let mut diis = false;
        for iter in 1..=max_iter {
            t.skip();
            let fvec: Vec<f64> = if ns == 1 { std::mem::take(&mut fock[0]) } else { fock.concat() };
            let mut err = 0.0;
            if iter == 1 {
                fmix = fvec;
            } else {
                let mut r = vec![0.0; len];
                crate::parallel::for_each_mut(&mut r, par, |k, x| *x = fvec[k] - fmix[k]);
                err = (dot(&r, &r) / len as f64).sqrt();
                if !diis {
                    let d = damping;
                    crate::parallel::for_each_mut(&mut fmix, par, |k, x| *x += d * r[k]);
                }
                let mut fvec = Some(fvec);
                if iter >= 3 {
                    let row: Vec<f64> =
                        hist.iter().map(|(_, rj)| dot(&r, rj)).chain(std::iter::once(dot(&r, &r))).collect();
                    for (bj, &v) in bmat.iter_mut().zip(&row) {
                        bj.push(v);
                    }
                    bmat.push(row);
                    hist.push((fvec.take().unwrap(), r));
                    if hist.len() > diis_size {
                        hist.remove(0);
                        bmat.remove(0);
                        bmat.iter_mut().for_each(|b| {
                            b.remove(0);
                        });
                    }
                }
                if diis {
                    match diis_coefficients(&bmat) {
                        Some(cf) => {
                            let mut v = vec![0.0; len];
                            crate::parallel::for_each_mut(&mut v, par, |k, x| {
                                for (ci, (fi, _)) in cf.iter().zip(&hist) {
                                    *x += ci * fi[k];
                                }
                            });
                            fmix = v;
                        }
                        None => fmix = fvec.unwrap_or_else(|| hist.last().unwrap().0.clone()),
                    }
                }
            }
            t.lap("scf: mixing");
            // the eigenvectors of the orbitals with nonzero occupation (their occupations at a
            // finite temperature follow from the eigenvalues)
            for s in 0..ns {
                let nel = if s == 0 { self.nalpha } else { self.nbeta };
                let need = |e: &[f64]| -> usize {
                    let o = if kt > 0.0 {
                        let (a, _, _) = super::fermi::fill_channel(nel, kt, e, self.model.program());
                        if ns == 1 {
                            let (b, _, _) = super::fermi::fill_channel(self.nbeta, kt, e, self.model.program());
                            a.iter().zip(&b).map(|(x, y)| x + y).collect()
                        } else {
                            a
                        }
                    } else {
                        occ[s].clone()
                    };
                    o.iter().rposition(|&x| x != 0.0).map_or(0, |i| i + 1)
                };
                let k = solver.solve_lowest(
                    MatRef::from_row_major_slice(&fmix[s * n * n..(s + 1) * n * n], n, n),
                    &mut eps[s],
                    &mut c,
                    need,
                )?;
                full[s] = solver.used_full();
                let mut cm = MatMut::from_row_major_slice_mut(&mut coef[s], n, n);
                cm.as_mut().subcols_mut(0, k).copy_from(c.as_ref().subcols(0, k));
                cm.subcols_mut(k, n - k).fill(0.0);
            }
            t.lap("scf: diagonalization");
            let mut ts = 0.0;
            if kt > 0.0 {
                (occ, efermi, ts) = self.fill(&eps, kt);
            }
            let pnew: Vec<Vec<f64>> = (0..ns).map(|s| weighted_density(n, &[(&coef[s], &occ[s])], self.par)).collect();
            t.lap("scf: density");
            let input = std::mem::replace(&mut st, self.state_of(pnew));
            t.lap("scf: mulliken");
            let (mut energies, fnew) = self.evaluate_timed(&st, true, &mut t);
            energies.add_entropy(ts);
            let e = energies.total;
            history.push((e, err));
            econverged |= (e - e_old).abs() < ethr;
            pconverged |= iter > 1 && err < pthr;
            diis |= iter >= 3 || (e - e_old).abs() < DIIS_SWITCH_ENERGY;
            if econverged && pconverged {
                if opts.keep_orbitals {
                    // every orbital, the occupied ones as in the iteration (same bits)
                    for s in 0..ns {
                        solver.solve_all_with(
                            MatRef::from_row_major_slice(&fmix[s * n * n..(s + 1) * n * n], n, n),
                            &mut eps[s],
                            &mut c,
                            full[s],
                        )?;
                        MatMut::from_row_major_slice_mut(&mut coef[s], n, n).copy_from(&c);
                    }
                }
                return Ok(ScfResult {
                    energies,
                    shell_charges: st.q.clone(),
                    atom_charges: st.qa.clone(),
                    orbital_energies: eps,
                    occupations: occ,
                    kt,
                    efermi,
                    coefficients: coef,
                    all_orbitals: opts.keep_orbitals,
                    density: st.p.clone(),
                    state: st,
                    input_state: input,
                    iterations: iter,
                    history,
                    timings: t.laps,
                    eigensolver: solver.backend(),
                });
            }
            e_old = e;
            fock = fnew;
        }
        let (de, res) = match history.len() {
            0 | 1 => (f64::NAN, f64::NAN),
            k => (history[k - 1].0 - history[k - 2].0, history[k - 1].1),
        };
        Err(Error::ScfNotConverged { iterations: max_iter, energy_change: de, residual: res })
    }
}
