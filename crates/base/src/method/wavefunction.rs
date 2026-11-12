// SPDX-License-Identifier: LGPL-3.0-or-later
//! A converged wavefunction to start another SCF from, with the fingerprint of what it belongs
//! to.

use crate::{Error, Result};

/// What a wavefunction belongs to: the method and the molecule (atomic numbers, charge,
/// unpaired electrons). Conventions are not part of it: a wavefunction of one conventions may
/// start an SCF in another.
#[derive(Clone, Debug, PartialEq)]
pub struct Fingerprint {
    method: &'static str,
    numbers: Vec<usize>,
    charge: f64,
    uhf: usize,
}

impl Fingerprint {
    /// The fingerprint of `method` (its [`MethodInfo::name`](super::MethodInfo::name)) for the
    /// molecule `numbers`, `charge`, `uhf` (`None`: 0 or 1 by the electron count).
    pub fn new(method: &'static str, numbers: &[usize], charge: f64, uhf: Option<usize>) -> Self {
        let uhf = uhf.unwrap_or_else(|| {
            let nel = numbers.iter().sum::<usize>() as f64 - charge;
            (nel.round() as i64).rem_euclid(2) as usize
        });
        Self { method, numbers: numbers.to_vec(), charge, uhf }
    }

    /// The fingerprint of `method` for the molecule `mol`.
    pub fn of(method: &'static str, mol: &crate::Molecule) -> Self {
        Self::new(method, &mol.numbers, mol.charge, mol.uhf)
    }

    /// The method's name.
    pub fn method(&self) -> &'static str {
        self.method
    }

    /// Atoms of the molecule.
    pub fn natoms(&self) -> usize {
        self.numbers.len()
    }
}

/// A converged SCF wavefunction of one method and molecule, to start another SCF of them
/// ([`Method::wavefunction`](super::Method::wavefunction),
/// [`Method::set_wavefunction`](super::Method::set_wavefunction)). Its content is the method's
/// own: nothing but the method that made it interprets it.
#[derive(Clone, Debug, PartialEq)]
pub struct Wavefunction {
    fingerprint: Fingerprint,
    data: WavefunctionData,
}

/// What a wavefunction holds.
#[derive(Clone, Debug, PartialEq)]
pub enum WavefunctionData {
    /// The self-consistent moments and densities of one SCF.
    Scf(ScfData),
    /// A composite's: one per part, in the composite's order (`None`: that part has none).
    Parts(Vec<Option<Wavefunction>>),
}

/// The self-consistent quantities of one SCF; empty where the method keeps none (GFN2-xTB:
/// the moments, no densities; g-xTB v1.1: the densities alone).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ScfData {
    pub shell_charges: Vec<f64>,
    pub atomic_dipoles: Vec<[f64; 3]>,
    pub atomic_quadrupoles: Vec<[f64; 6]>,
    pub shell_magnetizations: Option<Vec<f64>>,
    /// Density matrices per spin channel (row-major n x n; a closed shell: the total density).
    pub densities: Vec<Vec<f64>>,
}

impl Wavefunction {
    pub fn new(fingerprint: Fingerprint, data: WavefunctionData) -> Self {
        Self { fingerprint, data }
    }

    pub fn fingerprint(&self) -> &Fingerprint {
        &self.fingerprint
    }

    /// The method it belongs to.
    pub fn method(&self) -> &'static str {
        self.fingerprint.method
    }

    /// Atoms of its molecule.
    pub fn natoms(&self) -> usize {
        self.fingerprint.natoms()
    }

    pub fn data(&self) -> &WavefunctionData {
        &self.data
    }

    /// Refuse a wavefunction of another method or molecule than `target`'s.
    pub fn check(&self, target: &Fingerprint) -> Result<()> {
        let (own, t) = (&self.fingerprint, target);
        if own.method != t.method || own.natoms() != t.natoms() {
            return Err(Error::Input(format!(
                "a {} wavefunction of {} atoms cannot start a {} SCF of {} atoms",
                own.method,
                own.natoms(),
                t.method,
                t.natoms()
            )));
        }
        let differs = if own.numbers != t.numbers {
            "other atomic numbers".to_string()
        } else if own.charge != t.charge {
            format!("charge {} instead of {}", own.charge, t.charge)
        } else if own.uhf != t.uhf {
            format!("{} unpaired electrons instead of {}", own.uhf, t.uhf)
        } else {
            return Ok(());
        };
        Err(Error::Input(format!(
            "a {} wavefunction of another molecule ({differs}) cannot start this SCF",
            own.method
        )))
    }
}
