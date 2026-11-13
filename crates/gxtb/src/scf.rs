//! g-xTB v1.1 single points: the geometry-dependent setup, the energy and Fock matrices as
//! functions of the spin densities, and the 1.1.0 program's SCF as a model of the core's SCF
//! driver (`axtb::scf::run_scf`) with the program's strategy (`crate::strategy`).

use crate::aes::{aes_energy, aes_kernels, INTCUT as AES_INTCUT};
use crate::classical::{gxtb_cn, increment_energy, repulsion_energy};
use crate::dense::matmul;
use crate::disp::RevD4;
use crate::eeqbc::{eeqbc_charges, EeqBcCharges};
use crate::es::{first_order_energy, higher_order_energies, spin_energy};
use crate::fock::{
    add_charge_potential, add_potentials, aes_potentials, isotropic_shell_potential, mfx_fock, ofx_fock, shell_list,
    spin_shell_potential, Shell,
};
use crate::h0::build_h0;
use crate::ints::overlap_matrix;
use crate::kernels::{gamma2, gamma_mfx, hubbard_u, tau3};
use crate::options::Version;
pub use crate::options::{Conventions, ScfOptions};
use crate::params::GxtbParams;
use crate::qvszp::QvszpBasis;
use crate::strategy::{Gxtb110FockMixer, Gxtb110Rule};
use crate::timing::{Laps, Timings};
use axtb::ao::ShellPairs;
use axtb::integrals::{multipole, Scratch, MBLK};
use axtb::linalg::GenEigSolver;
use axtb::scf::{run_scf, Accelerator, FockSource, ScfModel, ScfStrategy};
use axtb::{Error, Molecule, Result};
use faer::{Mat, MatMut, MatRef, Par};
use std::ops::ControlFlow;

/// Everything that depends on the geometry only.
#[derive(Clone, Debug)]
pub struct Setup {
    pub mol: Molecule,
    pub param: GxtbParams,
    /// EEQ-BC charges and CNs.
    pub eeq: EeqBcCharges,
    /// g-xTB coordination numbers.
    pub cn: Vec<f64>,
    /// q-vSZP basis (its SCF basis `basis.scf` has the AO maps: `ao2sh`, `ao2at`).
    pub basis: QvszpBasis,
    /// (atom, shell, l) per shell ([`crate::fock::shell_list`]) and per AO (the OFX kernels'
    /// layout).
    pub shells: Vec<Shell>,
    pub ao: Vec<Shell>,
    pub nao: usize,
    /// Overlap and one-electron Hamiltonian H0 + ACP (full, row-major).
    pub s: Vec<f64>,
    pub h: Vec<f64>,
    /// Second- and third-order shell kernels (nsh x nsh), Mulliken-exchange AO kernel (nao x nao).
    pub gamma2: Vec<f64>,
    pub tau: Vec<f64>,
    pub gamma_x: Vec<f64>,
    /// AES kernels g3, g5, g7, g9 (nat x nat).
    pub aes_kernels: [Vec<f64>; 4],
    /// Dipole and traceless quadrupole integrals around the atom of the first AO index.
    pub dpint: [Vec<f64>; 3],
    pub qpint: [Vec<f64>; 6],
    /// Reference shell occupations.
    pub ref_occ: Vec<f64>,
    pub disp: RevD4,
    /// Density-independent energies.
    pub e_rep: f64,
    pub e_incr: f64,
    pub e_atm: f64,
    /// Electrons (total, alpha, beta).
    pub nel: usize,
    pub nalpha: usize,
    pub nbeta: usize,
    /// Parallelism of the dense linear algebra ([`Setup::set_threads`]; sequential by default).
    pub(crate) par: Par,
}

/// Energy terms of one density (Eh), named as in the program's printout.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Energies {
    /// Tr((H0 + ACP) P).
    pub one_electron: f64,
    /// "ES1 (charge SIE)".
    pub es1: f64,
    /// "ES2+3" = E2 + E3 + E4.
    pub es23: f64,
    /// "ES multipole".
    pub aes: f64,
    /// "Ex (Mulliken)".
    pub mfx: f64,
    /// Onsite exchange (not printed).
    pub ofx: f64,
    /// "Espinpol".
    pub spin: f64,
    /// "electronic".
    pub electronic: f64,
    /// "atomic core increments".
    pub increments: f64,
    /// "dispersion" (revD4 two-body).
    pub disp2: f64,
    /// revD4 three-body (in the total, not printed).
    pub disp3: f64,
    /// "nuclear repulsion".
    pub repulsion: f64,
    /// Electronic entropy term -TS at a finite electronic temperature (<= 0; in `electronic`
    /// and `total`, as the program adds it to its electronic energy).
    pub entropy: f64,
    pub total: f64,
}

impl Energies {
    /// Add the entropy term -TS (to `electronic` and `total`).
    pub fn add_entropy(&mut self, ts: f64) {
        self.entropy += ts;
        self.electronic += ts;
        self.total += ts;
    }
}

/// Energies, Mulliken charges and (optionally) Fock matrices of a density.
#[derive(Clone, Debug)]
pub struct Evaluation {
    pub energies: Energies,
    pub shell_charges: Vec<f64>,
    pub atom_charges: Vec<f64>,
    /// Fock matrix per spin (closed shell: one).
    pub fock: Vec<Vec<f64>>,
}

impl Setup {
    /// Geometry-dependent setup for `mol` (positions in bohr, integer charge, unpaired
    /// electrons); its linear algebra is sequential ([`Setup::set_threads`]).
    pub fn new(mol: &Molecule) -> Result<Self> {
        crate::check_charge(mol)?;
        let param = crate::params::gxtb_v1_1().clone();
        let qparam = crate::v2::qvszp();
        let eeq = eeqbc_charges(crate::params::eeqbc_v1_1(), mol)?;
        let cn = gxtb_cn(&param, mol)?;
        let basis = QvszpBasis::new(qparam, mol, &eeq.charges)?;
        let shells = shell_list(&basis.scf);
        let ao: Vec<Shell> = basis.scf.ao2sh.iter().map(|&s| shells[s]).collect();
        let nao = basis.scf.nao;
        let s = overlap_matrix(mol, &basis.scf);
        let mut h = build_h0(&param, qparam, mol, &basis, &cn)?.h;
        crate::acp::add_acp(&param, mol, &basis.scf, &cn, &mut h)?;
        let u = hubbard_u(&param, mol, &shells, &cn)?;
        let g2 = gamma2(&param, mol, &shells, &u);
        let tau = tau3(mol, &shells, &u, &g2);
        let gx_sh = gamma_mfx(&param, mol, &shells)?;
        let nsh = shells.len();
        let mut gamma_x = vec![0.0; nao * nao];
        for i in 0..nao {
            for j in 0..nao {
                gamma_x[i * nao + j] = gx_sh[basis.scf.ao2sh[i] * nsh + basis.scf.ao2sh[j]];
            }
        }
        let (dpint, qpint) = multipole_integrals(mol, &basis);
        let aes_kernels = aes_kernels(&param, mol);
        let ref_occ =
            shells.iter().map(|&(iat, ish, _)| crate::data_runtime::REF_SHELL_OCC[mol.numbers[iat] - 1][ish]).collect();
        let disp = RevD4::new(&param, mol)?;
        let e_atm = disp.three_body_energy();
        let e_rep = repulsion_energy(&param, mol, &eeq.charges, &cn)?;
        let e_incr = increment_energy(&param, mol)?;
        // electrons: valence electrons of the reference occupations minus the charge
        let zval: f64 =
            shells.iter().map(|&(iat, ish, _)| crate::data::ZEFF_SHELL[ish * 103 + mol.numbers[iat] - 1] as f64).sum();
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
        Ok(Self {
            mol: mol.clone(),
            param,
            eeq,
            cn,
            basis,
            shells,
            ao,
            nao,
            s,
            h,
            gamma2: g2,
            tau,
            gamma_x,
            aes_kernels,
            dpint,
            qpint,
            ref_occ,
            disp,
            e_rep,
            e_incr,
            e_atm,
            nel,
            nalpha: nel - nbeta,
            nbeta,
            par: Par::Seq,
        })
    }

    /// Run the dense linear algebra (matrix products, eigensolver) on `threads` threads of the
    /// current rayon pool, from `axtb_base::threads::DENSE_MIN_AO` basis functions on (below,
    /// and for 1 thread, sequentially; [`axtb_base::threads::dense_par`]). The caller runs the
    /// setup's work inside a pool of that size. Zero threads is an input error.
    pub fn set_threads(&mut self, threads: usize) -> Result<()> {
        axtb_base::threads::check(threads)?;
        self.par = axtb_base::threads::dense_par(threads, self.nao);
        Ok(())
    }

    /// Mulliken shell populations of a (spin) density ([`crate::mulliken::shell_populations`]).
    pub fn shell_populations(&self, p: &[f64]) -> Vec<f64> {
        crate::mulliken::shell_populations(&self.basis.scf, &self.s, p, false)
    }

    /// Atomic dipoles and traceless quadrupoles of the total density (v1.1's order,
    /// [`crate::mulliken::moments_by_elements`]).
    pub fn atomic_moments(&self, p: &[f64]) -> (Vec<[f64; 3]>, Vec<[f64; 6]>) {
        crate::mulliken::moments_by_elements(&self.basis.scf, &self.dpint, &self.qpint, p)
    }

    /// Fock contribution of the anisotropic electrostatics for the total density `ptot`
    /// (`full`: exact derivative incl. the dipole-quadrupole and quadrupole-quadrupole parts).
    pub fn aes_fock(&self, ptot: &[f64], full: bool) -> Vec<f64> {
        let n = self.nao;
        let q = crate::mulliken::shell_charges(&self.basis.scf, &self.s, &self.ref_occ, ptot, false);
        let qa = crate::es::atom_charges(self.mol.nat(), &self.shells, &q);
        let (dip, qp) = self.atomic_moments(ptot);
        let (vs, vd, vq) = aes_potentials(&self.mol, &qa, &dip, &qp, &self.aes_kernels, full);
        let vao: Vec<f64> = self.basis.scf.ao2at.iter().map(|&a| vs[a]).collect();
        let mut f = vec![0.0; n * n];
        add_potentials(n, &self.s, &self.basis.scf.ao2at, &self.dpint, &self.qpint, &vao, &vd, &vq, &mut f);
        f
    }

    /// Energies, Mulliken charges and (with `fock`) the Fock matrices of the given conventions
    /// of the spin densities `p` (closed shell: one matrix, the total density; open shell:
    /// alpha and beta).
    pub fn evaluate(&self, p: &[Vec<f64>], fock: Option<Conventions>) -> Result<Evaluation> {
        self.evaluate_timed(p, fock, &mut Laps::new())
    }

    /// [`Setup::evaluate`] with its phases timed in `t`: the Mulliken analysis and the energies
    /// (`scf: fock energies`), the potentials and Fock matrices (`scf: fock potentials`).
    pub(crate) fn evaluate_timed(&self, p: &[Vec<f64>], fock: Option<Conventions>, t: &mut Laps) -> Result<Evaluation> {
        t.skip();
        let n = self.nao;
        let mol = &self.mol;
        let closed = p.len() == 1;
        let d = self.density_terms(p);
        let ptot = &d.ptot;
        let e_h: f64 = (0..n * n).map(|k| self.h[k] * ptot[k]).sum();
        let es1 = first_order_energy(&self.param, mol, &self.shells, &self.cn, &d.q, &self.gamma2)?;
        let (e2, e3, e4) = higher_order_energies(mol, &self.shells, &d.q, &self.gamma2, &self.tau);
        let e_aes = aes_energy(mol, &d.qa, &d.dip, &d.qp, &self.aes_kernels);
        let (e_disp2, _) = self.disp.two_body_potential(&d.qa);
        let e_spin = match &d.mag {
            Some(m) => spin_energy(&self.param, mol, &self.shells, m)?,
            None => 0.0,
        };
        let e_ofx = crate::exchange::ofx_energy(&mol.numbers, &self.ao, &self.s, p);
        let e_x: f64 = p.iter().map(|ps| mfx_energy_fast(n, &self.s, ps, &self.gamma_x, closed, self.par)).sum();
        let electronic = e_h + es1 + e2 + e3 + e4 + e_aes + e_x + e_ofx + e_spin;
        let energies = Energies {
            one_electron: e_h,
            es1,
            es23: e2 + e3 + e4,
            aes: e_aes,
            mfx: e_x,
            ofx: e_ofx,
            spin: e_spin,
            electronic,
            increments: self.e_incr,
            disp2: e_disp2,
            disp3: self.e_atm,
            repulsion: self.e_rep,
            entropy: 0.0,
            total: electronic + self.e_incr + e_disp2 + self.e_atm + self.e_rep,
        };
        t.lap("scf: fock energies");
        let fmats = match fock {
            Some(c) => {
                let f = self.fock_from(p, &d, c)?;
                t.lap("scf: fock potentials");
                f
            }
            None => Vec::new(),
        };
        Ok(Evaluation { energies, shell_charges: d.q, atom_charges: d.qa, fock: fmats })
    }

    /// Total density, Mulliken charges, atomic moments and magnetizations of `p`.
    pub(crate) fn density_terms(&self, p: &[Vec<f64>]) -> DensityTerms {
        let ptot = crate::fock::total_density(p);
        let q = crate::mulliken::shell_charges(&self.basis.scf, &self.s, &self.ref_occ, &ptot, false);
        let qa = crate::es::atom_charges(self.mol.nat(), &self.shells, &q);
        let (dip, qp) = self.atomic_moments(&ptot);
        let mag = crate::mulliken::magnetizations(&self.basis.scf, &self.s, p, false);
        DensityTerms { ptot, q, qa, dip, qp, mag }
    }

    fn fock_from(&self, p: &[Vec<f64>], d: &DensityTerms, c: Conventions) -> Result<Vec<Vec<f64>>> {
        let pot = self.potentials(d, c)?;
        Ok(self.assemble_fock(self.h.clone(), &pot, p, c))
    }

    /// Potentials of the density terms: shell potentials of the charges (E1-E4, AES charge
    /// part, revD4), AES dipole and quadrupole potentials (`Conventions::Gxtb`: the program's
    /// subset), spin potentials (open shell).
    pub(crate) fn potentials(&self, d: &DensityTerms, c: Conventions) -> Result<Potentials> {
        let mol = &self.mol;
        let (_, vdisp) = self.disp.two_body_potential(&d.qa);
        let mut vsh =
            isotropic_shell_potential(&self.param, mol, &self.shells, &self.cn, &d.q, &self.gamma2, &self.tau)?;
        let (vs, vd, vq) = aes_potentials(mol, &d.qa, &d.dip, &d.qp, &self.aes_kernels, c == Conventions::Axtb);
        for (ish, &(iat, _, _)) in self.shells.iter().enumerate() {
            vsh[ish] += vs[iat] + vdisp[iat];
        }
        let w = match &d.mag {
            Some(m) => Some(spin_shell_potential(&self.param, mol, &self.shells, m)?),
            None => None,
        };
        Ok(Potentials { vsh, vd, vq, w })
    }

    /// Fock matrices `f0` + potentials + Mulliken and onsite exchange of the spin densities
    /// `p` (linear in `p`).
    pub(crate) fn assemble_fock(
        &self,
        mut f0: Vec<f64>,
        pot: &Potentials,
        p: &[Vec<f64>],
        c: Conventions,
    ) -> Vec<Vec<f64>> {
        let n = self.nao;
        let closed = p.len() == 1;
        let vao: Vec<f64> = self.basis.scf.ao2sh.iter().map(|&ish| pot.vsh[ish]).collect();
        add_potentials(n, &self.s, &self.basis.scf.ao2at, &self.dpint, &self.qpint, &vao, &pot.vd, &pot.vq, &mut f0);
        let wao: Option<Vec<f64>> = pot.w.as_ref().map(|w| self.basis.scf.ao2sh.iter().map(|&ish| w[ish]).collect());
        let mut fmats = Vec::with_capacity(p.len());
        for (k, ps) in p.iter().enumerate() {
            let fx = mfx_fock(n, &self.s, ps, &self.gamma_x, closed, self.par);
            let fo = ofx_fock(&self.mol.numbers, &self.ao, &self.s, ps, closed, c == Conventions::Axtb, self.par);
            let mut fs = f0.clone();
            for x in 0..n * n {
                fs[x] += fx[x] + fo[x];
            }
            if let Some(w) = &wao {
                // +1/2 (w_mu + w_nu) S for alpha, - for beta
                add_charge_potential(n, &self.s, w, &mut fs, if k == 0 { -1.0 } else { 1.0 });
            }
            fmats.push(fs);
        }
        fmats
    }
}

/// Potentials of [`Setup::potentials`].
pub(crate) struct Potentials {
    pub vsh: Vec<f64>,
    pub vd: Vec<[f64; 3]>,
    pub vq: Vec<[f64; 6]>,
    pub w: Option<Vec<f64>>,
}

/// Density-dependent quantities shared by the energy and the Fock matrices.
#[derive(Clone)]
pub(crate) struct DensityTerms {
    pub ptot: Vec<f64>,
    /// Shell and atomic Mulliken charges, atomic dipoles and quadrupoles, shell
    /// magnetizations (open shell).
    pub q: Vec<f64>,
    pub qa: Vec<f64>,
    pub dip: Vec<[f64; 3]>,
    pub qp: Vec<[f64; 6]>,
    pub mag: Option<Vec<f64>>,
}

/// Mulliken-exchange energy of one spin density (closed shell: total density, -1/8).
fn mfx_energy_fast(n: usize, s: &[f64], p: &[f64], gamma: &[f64], closed: bool, par: Par) -> f64 {
    let ps = matmul(n, p, s, par);
    let sps = matmul(n, s, &ps, par);
    let mut t = 0.0;
    for i in 0..n {
        for j in 0..n {
            let ij = i * n + j;
            // (P S)_ij (S P)_ij = (PS)_ij (PS)_ji
            t += gamma[ij] * (ps[ij] * ps[j * n + i] + p[ij] * sps[ij]);
        }
    }
    if closed {
        -0.125 * t
    } else {
        -0.25 * t
    }
}

/// Dipole and traceless quadrupole integrals of all AO pairs around the atom of the first
/// index (`[x][mu*n + nu]`), as the moments of [`crate::aes::atomic_moments`].
#[allow(clippy::type_complexity)]
pub fn multipole_integrals(mol: &Molecule, basis: &QvszpBasis) -> ([Vec<f64>; 3], [Vec<f64>; 6]) {
    let nao = basis.scf.nao;
    let z = || vec![0.0; nao * nao];
    let mut dp = [z(), z(), z()];
    let mut qp = [z(), z(), z(), z(), z(), z()];
    let mut s = [0.0; MBLK];
    let mut d = [[0.0; 3]; MBLK];
    let mut q = [[0.0; 6]; MBLK];
    let mut scratch = Scratch::default();
    // every ordered atom pair (the program's choice), no screening
    let pairs = ShellPairs::unscreened(mol, &basis.scf, f64::INFINITY);
    for ia in 0..mol.nat() {
        for ib in 0..mol.nat() {
            let Some(p) = pairs.pair(ia, ib) else { continue };
            pairs.for_each(&p, |sp| {
                multipole(sp.cj, sp.ci, p.vec, AES_INTCUT, &mut s, &mut d, &mut q, &mut scratch);
                let (na, nb) = (2 * sp.ci.l + 1, 2 * sp.cj.l + 1);
                for a in 0..na {
                    let mu = sp.ii + a;
                    for b in 0..nb {
                        let nu = sp.jj + b;
                        let k = b + nb * a;
                        for x in 0..3 {
                            dp[x][mu * nao + nu] = d[k][x];
                        }
                        for c in 0..6 {
                            qp[c][mu * nao + nu] = q[k][c];
                        }
                    }
                }
            });
        }
    }
    (dp, qp)
}

/// The g-xTB 1.1.0 program's SCF iteration limit.
pub const MAX_ITER: usize = 1000;
/// The g-xTB 1.1.0 program's damping factor (single precision).
pub const DAMPING: f64 = 0.1_f32 as f64;

/// Converged SCF.
#[derive(Clone, Debug)]
pub struct ScfResult {
    pub energies: Energies,
    /// Mulliken shell charges (reference occupation minus population) and atomic charges.
    pub shell_charges: Vec<f64>,
    pub atom_charges: Vec<f64>,
    /// Orbital energies per spin (Eh, ascending) and occupations.
    pub orbital_energies: Vec<Vec<f64>>,
    pub occupations: Vec<Vec<f64>>,
    /// kT (Eh) of the occupations (0: Aufbau).
    pub kt: f64,
    /// Fermi levels the occupations of the alpha and beta electrons were computed with (`None`
    /// without smearing).
    pub efermi: [Option<f64>; 2],
    /// MO coefficients per spin (row-major n x n, column k = orbital k) of the last
    /// diagonalization, whose orbitals give `density`.
    pub coefficients: Vec<Vec<f64>>,
    /// Converged spin densities (closed shell: total density), axtb AO order.
    pub density: Vec<Vec<f64>>,
    /// SCF iterations (diagonalizations).
    pub iterations: usize,
    /// Fock matrix the SCF diagonalized.
    pub conventions: Conventions,
    /// The eigensolver of the SCF (`axtb::linalg::GenEigSolver::backend`).
    pub eigensolver: String,
    /// Wall time of the SCF phases (`scf: cholesky`, the core driver's laps, and the
    /// evaluation's `scf: fock energies` and `scf: fock potentials`).
    pub timings: Timings,
}

impl Setup {
    /// Number of spin channels: 1 for a closed shell without unpaired electrons, else 2.
    pub fn nspin(&self) -> usize {
        if self.nalpha == self.nbeta {
            1
        } else {
            2
        }
    }

    /// The program's start density (`guess_psh_`, `guess_p_`): diagonal, with the reference
    /// shell occupations of every atom scaled to its EEQ-BC electron count, n_l (Z_A - q_A)/Z_A,
    /// spread evenly over the 2l+1 AOs of the shell (open shell: half per spin).
    pub fn guess(&self) -> Vec<Vec<f64>> {
        let n = self.nao;
        let ns = self.nspin();
        let fac = if ns == 2 { 0.5 } else { 1.0 };
        let mut zval = vec![0.0; self.mol.nat()];
        for &(iat, ish, _) in &self.shells {
            zval[iat] += crate::data::ZEFF_SHELL[ish * 103 + self.mol.numbers[iat] - 1] as f64;
        }
        let mut p = vec![0.0; n * n];
        for (mu, &(iat, _, l)) in self.ao.iter().enumerate() {
            let ish = self.basis.scf.ao2sh[mu];
            let z = zval[iat];
            let psh = (z - self.eeq.charges[iat]) * self.ref_occ[ish] / z;
            p[mu * n + mu] = fac * psh / (2 * l + 1) as f64;
        }
        vec![p; ns]
    }

    /// Self-consistent field as the program runs it (see [`ScfOptions`]): start density,
    /// Fock build, mixing, diagonalization with Aufbau occupation, until converged plus one
    /// iteration.
    pub fn scf(&self, opts: &ScfOptions) -> Result<ScfResult> {
        self.scf_from(opts, None)
    }

    /// [`Setup::scf`] from the spin densities `start` (e.g. the converged density of a
    /// previous geometry, as the 1.1.0 program's `gxtbrestart` carries it, taken over as it is
    /// although the q-vSZP basis of this geometry differs) instead of the program's start
    /// density; `None`: the program's start.
    pub fn scf_from(&self, opts: &ScfOptions, start: Option<&[Vec<f64>]>) -> Result<ScfResult> {
        opts.check(Version::V1_1)?;
        if let Some(p) = start {
            if p.len() != self.nspin() || p.iter().any(|x| x.len() != self.nao * self.nao) {
                return Err(Error::Input("g-xTB: the start density does not fit this molecule".into()));
            }
        }
        let n = self.nao;
        let ns = self.nspin();
        let program = opts.conventions == Conventions::Gxtb;
        // the program smears above 1e-6 K with its single-precision kB; the default conventions
        // at any temperature above 0 K with CODATA 2022 kB
        let kt = if program {
            if opts.etemp > crate::fermi::TEL_MIN {
                opts.etemp * crate::occupations::KB_GXTB110
            } else {
                0.0
            }
        } else if opts.etemp > 0.0 {
            opts.etemp * axtb::units::KB_HARTREE
        } else {
            0.0
        };
        let mut t = Laps::new();
        let solver = GenEigSolver::new(MatRef::from_row_major_slice(&self.s, n, n), self.par)?;
        t.lap("scf: cholesky");
        let nocc = if ns == 1 { vec![self.nalpha] } else { vec![self.nalpha, self.nbeta] };
        if nocc.iter().any(|&k| k > n) {
            return Err(Error::Input("g-xTB: more occupied orbitals than basis functions".into()));
        }
        let (ethr, pthr, ndamp, ndiis) = if ns == 1 { (1e-8, 2e-5, 3, 7) } else { (4e-7, 5e-5, 4, 8) };
        let ethr = opts.energy_tol.unwrap_or(ethr * opts.accuracy);
        let pthr = opts.density_tol.unwrap_or(pthr * opts.accuracy);
        let ndamp = opts.transition_iteration.unwrap_or(ndamp);
        let ndiis = opts.diis_size.unwrap_or(ndiis);
        let occ_value = if ns == 1 { 2.0 } else { 1.0 };
        let aufbau: Vec<Vec<f64>> =
            nocc.iter().map(|&k| (0..n).map(|i| if i < k { occ_value } else { 0.0 }).collect()).collect();
        let mut mixer = Gxtb110FockMixer::new(n, opts.damping.unwrap_or(DAMPING), ndamp, ndiis, start.is_some());
        let mut rule = Gxtb110Rule::new(ethr, pthr);
        let p = match start {
            Some(p) => p.to_vec(),
            None => self.guess(),
        };
        let mut scf = Scf110 {
            setup: self,
            solver,
            conventions: opts.conventions,
            program,
            lowest: !program && !opts.all_orbitals,
            kt,
            occ: aufbau,
            efermi: [None, None],
            ts: 0.0,
            c: Mat::zeros(n, n),
            eps: vec![vec![0.0; n]; ns],
            coef: vec![vec![0.0; n * n]; ns],
            pnew: Vec::new(),
            dp: f64::NAN,
            ev: None,
        };
        let max_iter = opts.iteration_limit(Version::V1_1);
        let strategy = ScfStrategy { accelerator: Accelerator::Fock(&mut mixer), rule: &mut rule, max_iter };
        let out = run_scf(&mut scf, strategy, p, &mut t)?;
        if !out.converged {
            let (de, dp) = (out.last.energy_change, out.last.density_change);
            return Err(Error::ScfNotConverged { iterations: max_iter, energy_change: de, residual: dp });
        }
        // orbital energies of the last diagonalization (unmixed Fock matrix of the previous
        // density), as the program prints them
        let Scf110 { solver, occ, efermi, eps, coef, ev, .. } = scf;
        let ev = ev.expect("a converged SCF evaluated its density");
        let eigensolver = solver.backend();
        Ok(self.result(
            ev,
            out.state,
            eps,
            coef,
            occ,
            (kt, efermi),
            out.iterations,
            opts.conventions,
            eigensolver,
            t.finish(),
        ))
    }

    fn result(
        &self,
        ev: Evaluation,
        p: Vec<Vec<f64>>,
        orbital_energies: Vec<Vec<f64>>,
        coefficients: Vec<Vec<f64>>,
        occupations: Vec<Vec<f64>>,
        (kt, efermi): (f64, [Option<f64>; 2]),
        iterations: usize,
        conventions: Conventions,
        eigensolver: String,
        timings: Timings,
    ) -> ScfResult {
        ScfResult {
            energies: ev.energies,
            shell_charges: ev.shell_charges,
            atom_charges: ev.atom_charges,
            orbital_energies,
            occupations,
            kt,
            efermi,
            coefficients,
            density: p,
            iterations,
            conventions,
            eigensolver,
            timings,
        }
    }
}

/// One v1.1 SCF of [`Setup::scf_from`] as the core's SCF model (the 1.1.0 program's iteration,
/// see [`ScfOptions`]): the eigenpairs of the mixed (in the last iteration the unmixed) Fock
/// matrices (every one in the program's conventions or with `all_orbitals`, otherwise the
/// eigenvalues and the occupied eigenvectors), the Aufbau or Fermi occupations (the program's eigenvalue cap at N_alpha + 10 in
/// its conventions), the density as one matrix product
/// ([`crate::v2::scf::weighted_density`]), its RMS change, and the evaluation of the density
/// with the next Fock matrices (none in the last iteration).
struct Scf110<'a> {
    setup: &'a Setup,
    solver: GenEigSolver,
    conventions: Conventions,
    program: bool,
    /// Only the eigenvectors of the orbitals with nonzero occupation (the default conventions
    /// without `all_orbitals`); otherwise every eigenpair.
    lowest: bool,
    kt: f64,
    /// Occupations per channel (at 0 K the Aufbau ones throughout), Fermi levels and the
    /// entropy term -TS of the last iteration.
    occ: Vec<Vec<f64>>,
    efermi: [Option<f64>; 2],
    ts: f64,
    c: Mat<f64>,
    /// Orbital energies and coefficients (row-major; with `lowest` the columns of the orbitals
    /// without occupation zero) per channel of the last iteration.
    eps: Vec<Vec<f64>>,
    coef: Vec<Vec<f64>>,
    /// The densities of the last iteration, until the state takes them.
    pnew: Vec<Vec<f64>>,
    /// RMS change of the packed spin densities in the last iteration.
    dp: f64,
    /// The evaluation (energies, charges) of the last iteration.
    ev: Option<Evaluation>,
}

impl ScfModel for Scf110<'_> {
    /// The spin densities (closed shell: the total density).
    type State = Vec<Vec<f64>>;

    fn diagonalize(&mut self, fock: FockSource<'_>) -> Result<ControlFlow<Error>> {
        let setup = self.setup;
        let (n, ns, kt) = (setup.nao, self.eps.len(), self.kt);
        for s in 0..ns {
            let fs = fock.matrix(s).expect("the v1.1 SCF diagonalizes row-major Fock matrices");
            if !self.lowest {
                self.solver.solve(MatRef::from_row_major_slice(fs, n, n), &mut self.eps[s], &mut self.c)?;
                MatMut::from_row_major_slice_mut(&mut self.coef[s], n, n).copy_from(&self.c);
                continue;
            }
            // the eigenvectors of the orbitals with nonzero occupation (at a finite temperature
            // the occupations follow from the eigenvalues, as in `density`)
            let occ = &self.occ;
            let need = |e: &[f64]| -> usize {
                let o = if kt > 0.0 {
                    let nel = if s == 0 { setup.nalpha } else { setup.nbeta };
                    let (a, _, _) = crate::fermi::fill_channel(nel, kt, e, false);
                    if ns == 1 {
                        let (b, _, _) = crate::fermi::fill_channel(setup.nbeta, kt, e, false);
                        a.iter().zip(&b).map(|(x, y)| x + y).collect()
                    } else {
                        a
                    }
                } else {
                    occ[s].clone()
                };
                o.iter().rposition(|&x| x != 0.0).map_or(0, |i| i + 1)
            };
            let k = self.solver.solve_lowest(
                MatRef::from_row_major_slice(fs, n, n),
                &mut self.eps[s],
                &mut self.c,
                need,
            )?;
            let mut cm = MatMut::from_row_major_slice_mut(&mut self.coef[s], n, n);
            cm.as_mut().subcols_mut(0, k).copy_from(self.c.as_ref().subcols(0, k));
            cm.subcols_mut(k, n - k).fill(0.0);
        }
        Ok(ControlFlow::Continue(()))
    }

    fn density(&mut self) {
        let setup = self.setup;
        let (n, ns, kt, program) = (setup.nao, self.eps.len(), self.kt, self.program);
        if kt > 0.0 {
            // closed shell: alpha and beta electrons fill the one set of levels. The program
            // computes only the lowest min(N_alpha + 10, n) eigenpairs (DSYGVX) and sets the
            // other eigenvalues to 100 Eh, so those levels stay empty at any temperature.
            let (na, nb) = (setup.nalpha, setup.nbeta);
            let nev = if program { (na + 10).min(n) } else { n };
            let capped = |e: &[f64]| -> Vec<f64> {
                e.iter().enumerate().map(|(k, &x)| if k < nev { x } else { 100.0 }).collect()
            };
            let eps = &self.eps;
            let (ea, eb) = (capped(&eps[0]), capped(&eps[if ns == 1 { 0 } else { 1 }]));
            let (fa, mua, ta) = crate::fermi::fill_channel(na, kt, &ea, program);
            let (fb, mub, tb) = crate::fermi::fill_channel(nb, kt, &eb, program);
            self.efermi = [mua, mub];
            self.ts = ta + tb;
            self.occ = if ns == 1 { vec![fa.iter().zip(&fb).map(|(a, b)| a + b).collect()] } else { vec![fa, fb] };
        }
        let (occ, coef) = (&self.occ, &self.coef);
        self.pnew = (0..ns).map(|s| crate::v2::scf::weighted_density(n, &[(&coef[s], &occ[s])], setup.par)).collect();
    }

    fn update_state(&mut self, p: &mut Vec<Vec<f64>>) {
        let n = self.setup.nao;
        let pnew = std::mem::take(&mut self.pnew);
        // RMS change of the packed spin densities
        let mut dp2 = 0.0;
        for s in 0..pnew.len() {
            for i in 0..n {
                for j in 0..=i {
                    let d = pnew[s][i * n + j] - p[s][i * n + j];
                    dp2 += d * d;
                }
            }
        }
        self.dp = (dp2 / (n * (n + 1) / 2) as f64).sqrt();
        *p = pnew;
    }

    fn evaluate(
        &mut self,
        _iter: usize,
        p: &Vec<Vec<f64>>,
        fock: bool,
        laps: &mut Laps,
    ) -> Result<(f64, Option<Vec<Vec<f64>>>)> {
        let mut ev = self.setup.evaluate_timed(p, fock.then_some(self.conventions), laps)?;
        ev.energies.add_entropy(self.ts);
        let e = ev.energies.total;
        let f = fock.then(|| std::mem::take(&mut ev.fock));
        self.ev = Some(ev);
        Ok((e, f))
    }

    fn density_change(&self) -> f64 {
        self.dp
    }
}

/// g-xTB v1.1 single point for atomic numbers, positions (bohr), total charge and number of
/// unpaired electrons (`None`: 0 for an even, 1 for an odd electron count) with the default
/// settings (exact Fock matrix; `Setup::scf` with `Conventions::Gxtb` reproduces the
/// program).
pub fn singlepoint(
    numbers: &[usize],
    positions_bohr: &[[f64; 3]],
    charge: f64,
    uhf: Option<usize>,
) -> Result<ScfResult> {
    let mol = Molecule::new(numbers, positions_bohr, charge, uhf)?;
    Setup::new(&mol)?.scf(&ScfOptions::default())
}
