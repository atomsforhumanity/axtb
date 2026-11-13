// SPDX-License-Identifier: LGPL-3.0-or-later
//! The GFN-FF topology: everything the force field derives from the setup structure.
//!
//! It corresponds to xtb's `TGFFTopology` plus the parts of `TNeigh` used by the energy
//! (`nb`, `bpair`, `blist`, `vbond`, `nr_hb`), for molecules (no periodic images). Lists
//! that xtb keeps as dense N x N or N x N/2 arrays (`bpair`, `alphanb`, `zetac6`,
//! `hbatABl`, `xbatABl`) are stored as per-atom data and evaluated per pair; the HB and XB
//! pair lists come out in xtb's order when generated (see `energy::hb_lists`).

use std::collections::HashMap;

/// Number of covalent bonds between atoms (`bpair`): 1, 2, 3 for 1-2, 1-3, 1-4 pairs,
/// 5 for everything further apart, 0 for an atom with itself.
#[derive(Clone, Debug, Default)]
pub struct BondDistance {
    map: Vec<HashMap<u32, u8>>,
}

impl BondDistance {
    pub(crate) fn new(n: usize) -> Self {
        BondDistance { map: vec![HashMap::new(); n] }
    }
    /// `bpair(j, i)` (j seen from i). Panics if `i` is not an atom index.
    #[inline]
    pub fn get(&self, j: usize, i: usize) -> u8 {
        if i == j {
            return 0;
        }
        self.map[i].get(&(j as u32)).copied().unwrap_or(5)
    }
    #[inline]
    pub(crate) fn set(&mut self, j: usize, i: usize, v: u8) {
        self.map[i].insert(j as u32, v);
    }
    /// Atoms j with bpair(j, i) = v, ascending. Panics if `i` is not an atom index.
    pub fn with_value(&self, i: usize, v: u8) -> Vec<usize> {
        let mut out: Vec<usize> = self.map[i].iter().filter(|(_, &b)| b == v).map(|(&j, _)| j as usize).collect();
        out.sort_unstable();
        out
    }
    pub fn len(&self) -> usize {
        self.map.len()
    }
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }
}

/// A bond of the bond list (`blist`): `blist(1) = j`, `blist(2) = i`.
#[derive(Clone, Debug)]
pub struct Bond {
    pub j: usize,
    pub i: usize,
    /// shift of the equilibrium distance, steepness, prefactor (`vbond`)
    pub vbond: [f64; 3],
    /// number of HB partners of an N-H/O-H bond (`nr_hb`; >= 1 selects the HB-modified bond)
    pub nr_hb: usize,
    /// Hueckel pi bond order (0 if none)
    pub pibo: f64,
    /// bond type of the setup (1 single, 2 pi, 3 sp, 4 hypervalent, 5 metal, 6 eta, 7 M-M)
    pub btyp: i32,
}

/// A bending term (`alist`): centre `i`, ligands `j`, `k`, with `vangl = (phi0, k)`.
#[derive(Clone, Debug)]
pub struct Angle {
    pub i: usize,
    pub j: usize,
    pub k: usize,
    pub phi0: f64,
    pub fc: f64,
}

/// A torsion or out-of-plane term (`tlist(1:5)`, `vtors`). For proper torsions
/// `atoms = (l, i, j, k)` of xtb's naming (central bond i-j); for out-of-plane terms
/// `atoms = (centre, nb1, nb2, nb3)` sorted by distance. `nrot` > 0: torsion with
/// periodicity nrot; 0: out-of-plane with phi0 = 0; -1: double-minimum inversion.
#[derive(Clone, Debug)]
pub struct Torsion {
    pub atoms: [usize; 4],
    pub nrot: i32,
    pub phi0: f64,
    pub fc: f64,
}

/// One A-H bond of the bond/HB coupling (`bond_hb_AH`, `bond_hb_B`).
#[derive(Clone, Debug)]
pub struct BondHb {
    pub a: usize,
    pub h: usize,
    pub b: Vec<usize>,
}

/// The complete GFN-FF topology of a molecule.
#[derive(Clone, Debug)]
pub struct Topology {
    pub numbers: Vec<usize>,
    pub charge: f64,
    /// accuracy of the cutoffs (xtb: 0.1, 2.0 above 10000 atoms)
    pub accuracy: f64,
    /// xtb version of the force field (the setup's and every evaluation's)
    pub conventions: crate::conventions::Conventions,
    /// neighbor lists of the force field (`neigh%nb`)
    pub nb: Vec<Vec<usize>>,
    /// hybridization (0 unknown, 1 sp, 2 sp2, 3 sp3, 5 hypervalent)
    pub hyb: Vec<i32>,
    /// special tags (1 carbene / nitro N, -1 eta-coordinated)
    pub itag: Vec<i32>,
    /// metal type used by the setup
    pub imetal: Vec<i32>,
    /// atoms in a Hueckel pi system
    pub pi: Vec<bool>,
    /// pi-system candidates of the setup (`piadr2 > 0`), used by the HB/XB selection
    pub pi_cand: Vec<bool>,
    /// amide N (sign of the hybridization column of xtb's output)
    pub amide: Vec<bool>,
    /// number of bonds between atom pairs
    pub bpair: BondDistance,
    pub bonds: Vec<Bond>,
    pub angles: Vec<Angle>,
    pub torsions: Vec<Torsion>,
    /// bonded three-body (ATM) triples (`b3list`)
    pub batm: Vec<[usize; 3]>,
    /// triple-bond torsions (`sTorsl`): C1, C2, Ci, Cnbi, C3, C4
    pub stors: Vec<[usize; 6]>,
    /// molecular fragments (1-based ids) and their charges
    pub nfrag: usize,
    pub fraglist: Vec<usize>,
    pub qfrag: Vec<f64>,
    /// topology charges (`qa`)
    pub qa: Vec<f64>,
    /// EEQ parameters of every atom
    pub chieeq: Vec<f64>,
    pub gameeq: Vec<f64>,
    pub alpeeq: Vec<f64>,
    /// non-bonded repulsion exponent factor per atom (alphanb = sqrt(f_i f_j) x pair factor)
    pub rep_f: Vec<f64>,
    /// dispersion charge scaling per atom (zetac6 = z_i z_j)
    pub zeta: Vec<f64>,
    /// HB basicity and acidity per atom
    pub hbbas: Vec<f64>,
    pub hbaci: Vec<f64>,
    /// H atoms that can form hydrogen bonds (`hbatHl`)
    pub hb_h: Vec<usize>,
    /// atoms that can be A or B of a hydrogen bond (the per-atom part of `hbatABl`)
    pub hb_ab: Vec<bool>,
    /// A-X bonds of halogen/chalcogen/pnictogen bonds (A, X), in xtb's order
    pub xb_ax: Vec<(usize, usize)>,
    /// atoms that can be B of a halogen/chalcogen/pnictogen bond (the B part of `xbatABl`)
    pub xb_b: Vec<bool>,
    /// A-H bonds coupled to hydrogen bonds
    pub bond_hb: Vec<BondHb>,
    /// coordination number of the setup structure (logarithmic erf CN) and metallic character
    pub cn_setup: Vec<f64>,
    pub mchar: Vec<f64>,
}

impl Topology {
    /// Number of atoms.
    pub fn natoms(&self) -> usize {
        self.numbers.len()
    }

    /// Exponent of the non-bonded repulsion of atoms i and j (xtb's `alphanb`). Panics if `i`
    /// or `j` is not an atom index.
    #[inline]
    pub fn alphanb(&self, i: usize, j: usize) -> f64 {
        (self.rep_f[i] * self.rep_f[j]).sqrt() * self.rep_pair_factor(i, j)
    }

    /// Pair factor of the non-bonded repulsion exponent (`ff` of `alphanb`).
    #[inline]
    fn rep_pair_factor(&self, i: usize, j: usize) -> f64 {
        use crate::param::{HH13REP, HH14REP, HHFAC};
        let ati = self.numbers[i];
        let atj = self.numbers[j];
        let mut ff = 1.0;
        if ati == 1 && atj == 1 {
            ff = 1.0 * HHFAC;
            let bp = self.bpair.get(i, j);
            if bp == 3 {
                ff *= HH14REP;
            }
            if bp == 2 {
                ff *= HH13REP;
            }
        }
        if (ati == 1 && crate::param::metal(atj) > 0) || (atj == 1 && crate::param::metal(ati) > 0) {
            ff = 0.85;
        }
        if (ati == 1 && atj == 6) || (atj == 1 && ati == 6) {
            ff = 0.91;
        }
        if (ati == 1 && atj == 8) || (atj == 1 && ati == 8) {
            ff = 1.04;
        }
        ff
    }

    /// Bond list as atom index pairs (i < j).
    pub fn bond_pairs(&self) -> Vec<(usize, usize)> {
        self.bonds.iter().map(|b| (b.i.min(b.j), b.i.max(b.j))).collect()
    }

    /// Fragment (0-based, in the order of the fragments' first atoms) of every atom.
    pub fn fragments(&self) -> Vec<usize> {
        self.fraglist.iter().map(|f| f - 1).collect()
    }
}
