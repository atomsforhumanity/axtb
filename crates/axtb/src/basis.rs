//! Contracted Gaussian basis functions and the AO/shell/atom bookkeeping.

use crate::molecule::Molecule;
use crate::param::sto_data::*;
use crate::{Error, Result};
use axtb_base::math::primitive_norm_tblite;

/// Highest angular momentum supported by the integral code.
pub const MAXL: usize = 4;
/// Maximum number of primitives in a contracted Gaussian.
pub const MAXPRIM: usize = 16;

/// Number of spherical components for angular momentum `l`.
#[inline]
pub const fn nsph(l: usize) -> usize {
    2 * l + 1
}

/// Number of Cartesian components for angular momentum `l`.
#[inline]
pub const fn ncart(l: usize) -> usize {
    (l + 1) * (l + 2) / 2
}

/// A contracted Cartesian Gaussian shell; the coefficients include the normalization of
/// the x^l component (the spherical transformation completes the normalization).
#[derive(Clone, Copy, Debug)]
pub struct Cgto {
    /// Angular momentum.
    pub l: usize,
    /// Number of primitives.
    pub nprim: usize,
    /// Primitive exponents.
    pub alpha: [f64; MAXPRIM],
    /// Contraction coefficients.
    pub coeff: [f64; MAXPRIM],
}

impl Cgto {
    /// Smallest primitive exponent.
    pub fn min_alpha(&self) -> f64 {
        self.alpha[..self.nprim].iter().copied().fold(f64::INFINITY, f64::min)
    }
}

/// Which STO-NG tables to use (GFN1 has dedicated 4sp/5sp six-primitive expansions).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StoExpansion {
    Gfn2,
    Gfn1,
}

/// Expand a Slater function (principal quantum number `n`, angular momentum `l`, exponent
/// `zeta`) in `ng` Gaussians (STO-NG, Stewart's tables), optionally normalized.
pub fn slater_to_gauss(ng: usize, n: usize, l: usize, zeta: f64, norm: bool, expansion: StoExpansion) -> Result<Cgto> {
    if (n > 5 || n <= l) && !(n == 6 && ng == 6) {
        return Err(Error::Input(format!("STO-NG: unsupported n={n}, l={l}")));
    }
    if zeta <= 0.0 {
        return Err(Error::Input("STO-NG: non-positive Slater exponent".into()));
    }
    // ityp: 1s 2s 3s 4s 5s 2p 3p 4p 5p 3d 4d 5d 4f 5f 5g (0-based index below)
    let ityp = match l {
        0 => n,
        1 => 4 + n,
        2 => 7 + n,
        3 => 9 + n,
        4 => 10 + n,
        _ => return Err(Error::Input("STO-NG: l > 4".into())),
    } - 1;
    let mut cgto = Cgto { l, nprim: ng, alpha: [0.0; MAXPRIM], coeff: [0.0; MAXPRIM] };
    let z2 = zeta * zeta;
    macro_rules! fill {
        ($a:expr, $c:expr) => {{
            for i in 0..ng {
                cgto.alpha[i] = $a[i] * z2;
                cgto.coeff[i] = $c[i];
            }
        }};
    }
    match ng {
        1 => {
            cgto.alpha[0] = P_ALPHA1[ityp] * z2;
            cgto.coeff[0] = 1.0;
        }
        2 => fill!(P_ALPHA2[ityp], P_COEFF2[ityp]),
        3 => fill!(P_ALPHA3[ityp], P_COEFF3[ityp]),
        4 => fill!(P_ALPHA4[ityp], P_COEFF4[ityp]),
        5 => fill!(P_ALPHA5[ityp], P_COEFF5[ityp]),
        6 => {
            if n == 6 {
                match l {
                    0 => fill!(P_ALPHA_6S, P_COEFF_6S),
                    1 => fill!(P_ALPHA_6P, P_COEFF_6P),
                    _ => return Err(Error::Input("STO-6G: n=6 only for s and p".into())),
                }
            } else if expansion == StoExpansion::Gfn1 && (n == 4 || n == 5) && l <= 1 {
                let k = n - 4;
                if l == 0 {
                    fill!(P_ALPHA_6SP[k], P_COEFF_6SP_S[k]);
                } else {
                    fill!(P_ALPHA_6SP[k], P_COEFF_6SP_P[k]);
                }
            } else {
                fill!(P_ALPHA6[ityp], P_COEFF6[ityp]);
            }
        }
        _ => return Err(Error::Input(format!("STO-NG: {ng} primitives not supported"))),
    }
    if norm {
        for i in 0..ng {
            cgto.coeff[i] *= primitive_norm_tblite(l, cgto.alpha[i]);
        }
    }
    Ok(cgto)
}

/// Basis set for one molecule: its shell sets plus index maps.
///
/// The shells come in sets: one per species ([`Basis::new`]; the atoms of a species share
/// their shells) or one per atom ([`Basis::from_atom_shells`], e.g. charge-adapted shells).
/// [`Basis::id`] gives the set of every atom and [`Basis::shells`] the shells of an atom; the
/// index maps are the same for both kinds.
///
/// Indices are 0-based. AOs are ordered atom by atom, shell by shell, with spherical
/// components m = -l..l (p: y, z, x).
#[derive(Clone, Debug)]
pub struct Basis {
    /// Shells of each shell set (species, or atom for [`Basis::from_atom_shells`]).
    pub cgto: Vec<Vec<Cgto>>,
    /// Number of shells of each shell set.
    pub nsh_id: Vec<usize>,
    /// Shell set of each atom: its species index (`Molecule::id`) for [`Basis::new`], the atom
    /// itself for [`Basis::from_atom_shells`].
    pub id: Vec<usize>,
    /// First shell of each atom.
    pub ish_at: Vec<usize>,
    /// Number of shells of each atom.
    pub nsh_at: Vec<usize>,
    /// First AO of each atom.
    pub iao_at: Vec<usize>,
    /// Number of AOs of each atom.
    pub nao_at: Vec<usize>,
    /// Atom of each shell.
    pub sh2at: Vec<usize>,
    /// First AO of each shell.
    pub iao_sh: Vec<usize>,
    /// Number of AOs of each shell.
    pub nao_sh: Vec<usize>,
    /// Atom of each AO.
    pub ao2at: Vec<usize>,
    /// Shell of each AO.
    pub ao2sh: Vec<usize>,
    /// Total number of shells and AOs.
    pub nsh: usize,
    pub nao: usize,
    /// Highest angular momentum.
    pub maxl: usize,
    /// Smallest primitive exponent.
    pub min_alpha: f64,
    /// Smallest primitive exponent of each shell ([`Cgto::min_alpha`]) and of each atom (over
    /// its shells), for the shell-pair screening ([`crate::integrals::pair_screened`]).
    pub min_alpha_sh: Vec<f64>,
    pub min_alpha_at: Vec<f64>,
    /// Primitive-pair screening threshold on the Gaussian product exponent.
    pub intcut: f64,
}

/// The xtb program's primitive screening threshold: max(20, 25 - 10 log10(acc)).
pub fn integral_cutoff_xtb(acc: f64) -> f64 {
    (25.0 - 10.0 * acc.log10()).max(20.0)
}

/// tblite's integral screening threshold for a given accuracy.
pub fn integral_cutoff(acc: f64) -> f64 {
    let acc = acc.clamp(1.0e-4, 1.0e3);
    (25.0 - 10.0 * acc.log10()).clamp(5.0, 25.0)
}

impl Basis {
    /// Build the basis for `mol` from the shells of each species (`cgto[Molecule::id[atom]]`),
    /// with tblite's primitive screening at accuracy `acc` ([`integral_cutoff`]).
    pub fn new(mol: &Molecule, cgto: Vec<Vec<Cgto>>, acc: f64) -> Self {
        Self::from_sets(cgto, mol.id.clone(), integral_cutoff(acc))
    }

    /// Build the basis for `mol` from the shells of each atom (`shells[atom]`) with the
    /// primitive screening threshold `intcut`. The index maps are those [`Basis::new`] gives
    /// for the same shells.
    ///
    /// # Panics
    /// If `shells` does not have one entry per atom.
    pub fn from_atom_shells(mol: &Molecule, shells: Vec<Vec<Cgto>>, intcut: f64) -> Self {
        assert_eq!(shells.len(), mol.nat(), "Basis::from_atom_shells: one shell list per atom");
        let id = (0..shells.len()).collect();
        Self::from_sets(shells, id, intcut)
    }

    /// The basis of shell sets `cgto` with set `id[atom]` for every atom.
    fn from_sets(cgto: Vec<Vec<Cgto>>, id: Vec<usize>, intcut: f64) -> Self {
        let nsh_id: Vec<usize> = cgto.iter().map(|s| s.len()).collect();
        let nat = id.len();
        let mut ish_at = Vec::with_capacity(nat);
        let mut nsh_at = Vec::with_capacity(nat);
        let mut iao_at = Vec::with_capacity(nat);
        let mut nao_at = Vec::with_capacity(nat);
        let mut min_alpha_at = Vec::with_capacity(nat);
        let mut sh2at = Vec::new();
        let mut iao_sh = Vec::new();
        let mut nao_sh = Vec::new();
        let mut min_alpha_sh = Vec::new();
        let mut ao2at = Vec::new();
        let mut ao2sh = Vec::new();
        let mut ish = 0;
        let mut iao = 0;
        for (iat, &isp) in id.iter().enumerate() {
            ish_at.push(ish);
            nsh_at.push(nsh_id[isp]);
            iao_at.push(iao);
            let mut amin = f64::INFINITY;
            for sh in &cgto[isp] {
                sh2at.push(iat);
                iao_sh.push(iao);
                nao_sh.push(nsph(sh.l));
                let a = sh.min_alpha();
                min_alpha_sh.push(a);
                amin = amin.min(a);
                for _ in 0..nsph(sh.l) {
                    ao2at.push(iat);
                    ao2sh.push(ish);
                }
                iao += nsph(sh.l);
                ish += 1;
            }
            nao_at.push(iao - iao_at[iat]);
            min_alpha_at.push(amin);
        }
        let maxl = cgto.iter().flatten().map(|c| c.l).max().unwrap_or(0);
        let min_alpha = cgto.iter().flatten().map(|c| c.min_alpha()).fold(f64::INFINITY, f64::min);
        Self {
            cgto,
            nsh_id,
            id,
            ish_at,
            nsh_at,
            iao_at,
            nao_at,
            sh2at,
            iao_sh,
            nao_sh,
            ao2at,
            ao2sh,
            nsh: ish,
            nao: iao,
            maxl,
            min_alpha,
            min_alpha_sh,
            min_alpha_at,
            intcut,
        }
    }

    /// Shells of atom `iat`.
    #[inline]
    pub fn shells(&self, iat: usize) -> &[Cgto] {
        &self.cgto[self.id[iat]]
    }

    /// Every shell in basis order with its atom and its index within the atom:
    /// `(atom, ish, shell)`.
    pub fn shell_iter(&self) -> impl Iterator<Item = (usize, usize, &Cgto)> + '_ {
        self.sh2at.iter().enumerate().map(|(s, &iat)| {
            let ish = s - self.ish_at[iat];
            (iat, ish, &self.shells(iat)[ish])
        })
    }

    /// Real-space cutoff (bohr) beyond which atom pairs have negligible overlap, as in tblite.
    pub fn cutoff(&self, acc: f64) -> f64 {
        (2.0 * integral_cutoff(acc) / self.min_alpha).sqrt().min(40.0)
    }
}

/// First AO of every shell of every atom for shells given per atom (`shells[atom]`), with the
/// AOs ordered as in [`Basis::new`]: atom by atom, shell by shell, [`nsph`] each.
pub fn ao_offsets(shells: &[Vec<Cgto>]) -> Vec<Vec<usize>> {
    let mut iao = 0;
    shells
        .iter()
        .map(|sh| {
            sh.iter()
                .map(|s| {
                    let first = iao;
                    iao += nsph(s.l);
                    first
                })
                .collect()
        })
        .collect()
}

/// Number of AOs of shells given per atom (the AO count of [`ao_offsets`]).
pub fn ao_count(shells: &[Vec<Cgto>]) -> usize {
    shells.iter().flatten().map(|s| nsph(s.l)).sum()
}
