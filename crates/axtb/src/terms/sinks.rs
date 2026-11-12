//! What a term reads (the geometry it is bound to, the SCC moments) and what it writes into
//! (energies, potentials, kernels, potential derivatives). Every sink is added to, never
//! assigned: the model sizes and zeroes it and decides which terms share it.

use super::chain::{ChainLayout, Slot, Values};
use crate::molecule::Molecule;

/// The geometry a term is bound to: the molecule (positions, atomic numbers, species), the
/// model's intermediates with their values at these positions, and whether the term may spread
/// its loops over the current rayon pool (false where the call itself is one of many running in
/// parallel, e.g. a displaced geometry of a finite difference). The results must not depend on
/// `parallel`.
///
/// A term is built for one molecule, so it keeps what does not change with the geometry (the
/// parameters of its atoms, the shell layout); the geometry carries only what moves.
#[derive(Clone, Copy)]
pub struct Geometry<'a> {
    pub mol: &'a Molecule,
    pub layout: &'a ChainLayout,
    pub values: &'a Values,
    pub parallel: bool,
}

impl<'a> Geometry<'a> {
    /// The values of the intermediate output `s` (one per atom).
    #[inline]
    pub fn value(&self, s: Slot) -> &'a [f64] {
        self.values.get(s)
    }
}

/// Which SCC variables a moment term reads: they decide the layout of the SCF variables (atomic
/// multipoles, shell magnetizations) and the kernel blocks it writes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Needs {
    /// Mulliken shell charges.
    pub shell: bool,
    /// Atomic charges (sums of the shell charges of each atom).
    pub atom: bool,
    /// Atomic dipoles.
    pub dipole: bool,
    /// Atomic quadrupoles (traceless, packed xx, xy, yy, xz, yz, zz).
    pub quadrupole: bool,
    /// Shell magnetizations (open shells).
    pub magnetization: bool,
}

/// The SCC moments of one state, borrowed: shell charges, atomic charges, atomic dipoles and
/// packed traceless quadrupoles (xx, xy, yy, xz, yz, zz) and, for spin-polarized methods, shell
/// magnetizations. Also the changes δs of a response.
#[derive(Clone, Copy, Debug)]
pub struct SccMoments<'a> {
    pub qsh: &'a [f64],
    pub qat: &'a [f64],
    pub dip: &'a [[f64; 3]],
    pub quad: &'a [[f64; 6]],
    pub mag: Option<&'a [f64]>,
}

/// The derivatives dE/ds of an energy with respect to the SCC moments (the potentials of the
/// Fock matrix), added to by the terms. The model combines the atomic potentials with the shell
/// potentials in its own way (GFN2-xTB per basis function, g-xTB into the shells of each atom).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Potentials {
    /// dE/dq of every shell charge.
    pub shell: Vec<f64>,
    /// dE/dq of every atomic charge.
    pub atom: Vec<f64>,
    /// dE/dμ of every atomic dipole.
    pub dipole: Vec<[f64; 3]>,
    /// dE/dΘ of every atomic quadrupole (packed; the derivative with respect to each stored
    /// component, so an off-diagonal element counts twice in the energy).
    pub quadrupole: Vec<[f64; 6]>,
    /// dE/dm of every shell magnetization (empty without magnetizations).
    pub magnetization: Vec<f64>,
}

impl Potentials {
    /// Zero potentials of `nsh` shells and `nat` atoms, with shell magnetizations when asked.
    pub fn new(nsh: usize, nat: usize, magnetization: bool) -> Self {
        Self {
            shell: vec![0.0; nsh],
            atom: vec![0.0; nat],
            dipole: vec![[0.0; 3]; nat],
            quadrupole: vec![[0.0; 6]; nat],
            magnetization: vec![0.0; if magnetization { nsh } else { 0 }],
        }
    }

    /// Set every potential to zero.
    pub fn zero(&mut self) {
        self.shell.fill(0.0);
        self.atom.fill(0.0);
        self.dipole.fill([0.0; 3]);
        self.quadrupole.fill([0.0; 6]);
        self.magnetization.fill(0.0);
    }

    /// Shell, atomic, dipole, quadrupole and magnetization potentials in one vector, in that
    /// order (for finite differences).
    pub(crate) fn flatten(&self) -> Vec<f64> {
        let mut v = Vec::with_capacity(self.shell.len() + 10 * self.atom.len() + self.magnetization.len());
        v.extend_from_slice(&self.shell);
        v.extend_from_slice(&self.atom);
        v.extend(self.dipole.iter().flatten());
        v.extend(self.quadrupole.iter().flatten());
        v.extend_from_slice(&self.magnetization);
        v
    }
}

/// The energy of one term in one evaluation: one total per piece of the term's names (set by
/// the term), and, when the model sums its energy per atom and the term partitions its energy
/// ([`partitions_energy`](super::MomentTerm::partitions_energy)), the energy of all its pieces
/// partitioned over the atoms: the term's own row, zero on entry, added to (one value per
/// atom).
pub struct TermEnergy<'a> {
    pub pieces: &'a mut [f64],
    pub atoms: Option<&'a mut [f64]>,
}

/// Second derivatives d²E/ds ds of the SCC moments, added to by blocks (the model sizes the
/// blocks it uses, leaves the others empty, and maps them to the layout of its SCF variables).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Kernel {
    /// Shell charge by shell charge (row-major nsh x nsh).
    pub shell: Vec<f64>,
    /// Atomic charge by atomic charge (row-major nat x nat), for models without multipoles or
    /// for terms that read only the atomic charges.
    pub atom: Vec<f64>,
    /// The atomic moments [q (nat) | μ (3 nat) | Θ (6 nat, packed)] by themselves (row-major,
    /// 10 nat square).
    pub multipole: Vec<f64>,
    /// Shell magnetization by shell magnetization (row-major nsh x nsh).
    pub magnetization: Vec<f64>,
}

/// The derivatives c = ∂V/∂R of the potentials with respect to the nuclear positions at fixed
/// SCC moments, added to by blocks (row-major, one column per Cartesian coordinate 3·atom + k).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PotentialDerivative {
    /// The shell potentials (nsh x 3N; empty when the model has no shell rows).
    pub shell: Vec<f64>,
    /// The atomic potentials [q (nat) | μ (3 nat) | Θ (6 nat)] (10 nat x 3N), or only the
    /// atomic charges (nat x 3N) in a model without multipoles.
    pub atomic: Vec<f64>,
    /// Derivatives of the atomic rows with respect to the values of an intermediate (rows of
    /// `atomic` x nat each), for the model to chain with the intermediate's Jacobian.
    wrt: Vec<(Slot, Vec<f64>)>,
}

impl PotentialDerivative {
    /// Zero derivatives of `shell_rows` shell and `atomic_rows` atomic potentials over `n3`
    /// coordinates.
    pub fn new(shell_rows: usize, atomic_rows: usize, n3: usize) -> Self {
        Self { shell: vec![0.0; shell_rows * n3], atomic: vec![0.0; atomic_rows * n3], wrt: Vec::new() }
    }

    /// The derivative of the atomic rows with respect to the values of intermediate output `s`
    /// (row-major `rows x nat`, zero when first asked for), to add to.
    pub fn wrt(&mut self, s: Slot, rows: usize, nat: usize) -> &mut [f64] {
        let k = match self.wrt.iter().position(|(t, _)| *t == s) {
            Some(k) => k,
            None => {
                self.wrt.push((s, vec![0.0; rows * nat]));
                self.wrt.len() - 1
            }
        };
        &mut self.wrt[k].1
    }

    /// The derivatives with respect to intermediates written so far.
    pub fn wrt_slots(&self) -> &[(Slot, Vec<f64>)] {
        &self.wrt
    }
}
