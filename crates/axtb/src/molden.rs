//! Molden files of converged molecular orbitals, for orbital viewers (Molden, Jmol,
//! Avogadro, IboView, Multiwfn, PySCF/IOData readers).
//!
//! [`Orbitals`] holds what a file needs: the atoms, each atom's contracted shells (for g-xTB
//! every atom has its own, charge-adapted contraction) and one set of orbitals (closed shell)
//! or an alpha and a beta set (open shell). [`Orbitals::to_molden`] writes them in the
//! conventions of the Molden format
//! (<https://www.theochem.ru.nl/molden/molden_format.html>) as standard readers interpret it:
//!
//! - `[Atoms] AU`: element, index, atomic number, position (bohr);
//! - `[GTO]`, atom by atom: the contraction coefficients are those of *normalized*
//!   primitives and the contracted functions are normalized (axtb's internal coefficients
//!   include the primitive normalization; it is divided out here);
//! - [`MoldenFunctions::Spherical`] (default): spherical functions, flagged `[7F]` and `[5D]`
//!   (whichever shells occur; this order reads as 5D/7F both in readers that take `[7F]` as
//!   "6D and 7F", as the format description does, and in readers that set one flag per
//!   shell type), in Molden's order: p x, y, z; d 0, +1, -1, +2, -2; f 0, +1, -1, +2, -2,
//!   +3, -3 (m = 0, +1, -1, ...). Each is a normalized real solid harmonic with the
//!   standard signs (d+1 ∝ xz, d-1 ∝ yz, d+2 ∝ x² - y², d-2 ∝ xy, f+3 ∝ x(x² - 3y²),
//!   f-3 ∝ y(3x² - y²), ...), which are axtb's own, so only the order changes;
//! - [`MoldenFunctions::Cartesian`]: Cartesian functions (`[6D]`, `[10F]`), each Cartesian
//!   component normalized on its own, in Molden's order (d: xx, yy, zz, xy, xz, yz; f: xxx,
//!   yyy, zzz, xyy, xxy, xxz, xzz, yzz, yyz, xyz), for readers without spherical functions;
//! - `[MO]`: every orbital with `Sym=`, `Ene=` (Eh), `Spin=` (Alpha/Beta), `Occup=` and its
//!   coefficients. A closed shell is one set of spatial orbitals (occupations 0 to 2); an
//!   open shell is an alpha set followed by a beta set (GFN2-xTB: the same spatial orbitals
//!   with the alpha and beta occupations; g-xTB: its two spin channels).
//!
//! The coefficients of the MOs refer to the normalized functions, so a reader that
//! normalizes the contracted functions itself and one that takes them as written get the
//! same orbitals. The xtb and tblite programs write the coefficients of unnormalized
//! primitives in Cartesian functions with their own component normalizations; their files
//! and these describe the same orbitals (`tests/molden_check.py`).

use crate::basis::{ncart, nsph, Cgto};
use crate::error::{Error, Result};
use crate::integrals::{cart_exponents, sph_trafo};
use std::f64::consts::PI;
use std::fmt::Write;

/// Spin of a set of orbitals.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Spin {
    Alpha,
    Beta,
}

impl Spin {
    /// The label of the Molden `Spin=` field.
    pub fn label(&self) -> &'static str {
        match self {
            Spin::Alpha => "Alpha",
            Spin::Beta => "Beta",
        }
    }
}

/// One set of molecular orbitals.
#[derive(Clone, Debug)]
pub struct MoSet {
    pub spin: Spin,
    /// Orbital energies (Eh).
    pub energies: Vec<f64>,
    /// Occupations: 0 to 2 in the single set of a closed shell, 0 to 1 in an alpha or beta set.
    pub occupations: Vec<f64>,
    /// Coefficients orbital by orbital: `coefficients[k * nao + mu]` is AO `mu` in orbital
    /// `k`, in axtb's AO order (atom by atom, shell by shell, m = -l..l; p: y, z, x).
    pub coefficients: Vec<f64>,
}

/// Converged molecular orbitals with the basis they are expanded in.
#[derive(Clone, Debug)]
pub struct Orbitals {
    /// Atomic numbers and positions (bohr).
    pub numbers: Vec<usize>,
    pub positions: Vec<[f64; 3]>,
    /// Contracted shells of every atom ([`Cgto`] convention: coefficients include the
    /// normalization of the x^l primitive).
    pub shells: Vec<Vec<Cgto>>,
    /// One set (closed shell) or an alpha and a beta set (open shell).
    pub sets: Vec<MoSet>,
}

/// Which functions the `[MO]` coefficients of a Molden file refer to.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MoldenFunctions {
    /// Spherical functions (`[5D]`, `[7F]`), as the calculation uses them (default).
    #[default]
    Spherical,
    /// Cartesian functions (`[6D]`, `[10F]`), each component normalized.
    Cartesian,
}

impl std::str::FromStr for MoldenFunctions {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        match s.to_ascii_lowercase().as_str() {
            "spherical" | "pure" | "5d" => Ok(Self::Spherical),
            "cartesian" | "6d" => Ok(Self::Cartesian),
            other => Err(Error::Input(format!("unknown Molden functions {other:?} (spherical or cartesian)"))),
        }
    }
}

/// Double factorial (2l - 1)!! for l = 0..=4.
const DFACT: [f64; 5] = [1.0, 1.0, 3.0, 15.0, 105.0];

/// Normalization of the primitive x^l exp(-alpha r^2).
fn primitive_norm(l: usize, alpha: f64) -> f64 {
    (2.0 * alpha / PI).powf(0.75) * (4.0 * alpha).sqrt().powi(l as i32) / DFACT[l].sqrt()
}

/// Coefficients d_k of the normalized primitives of `cgto` and the norm of its contracted
/// x^l component: x^l cgto = norm * sum_k d_k N_k x^l exp(-a_k r^2), where N_k normalizes the
/// primitive and the sum has norm 1.
fn normalized_contraction(cgto: &Cgto) -> (Vec<f64>, f64) {
    let n = cgto.nprim;
    let raw: Vec<f64> = (0..n).map(|k| cgto.coeff[k] / primitive_norm(cgto.l, cgto.alpha[k])).collect();
    let mut s = 0.0;
    for i in 0..n {
        for j in 0..n {
            let (a, b) = (cgto.alpha[i], cgto.alpha[j]);
            s += raw[i] * raw[j] * (2.0 * (a * b).sqrt() / (a + b)).powf(cgto.l as f64 + 1.5);
        }
    }
    let norm = s.sqrt();
    (raw.iter().map(|c| c / norm).collect(), norm)
}

/// Molden's Cartesian exponents of shell `l` in file order.
fn molden_cartesian(l: usize) -> &'static [[usize; 3]] {
    const S: [[usize; 3]; 1] = [[0, 0, 0]];
    const P: [[usize; 3]; 3] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    const D: [[usize; 3]; 6] = [[2, 0, 0], [0, 2, 0], [0, 0, 2], [1, 1, 0], [1, 0, 1], [0, 1, 1]];
    const F: [[usize; 3]; 10] =
        [[3, 0, 0], [0, 3, 0], [0, 0, 3], [1, 2, 0], [2, 1, 0], [2, 0, 1], [1, 0, 2], [0, 1, 2], [0, 2, 1], [1, 1, 1]];
    match l {
        0 => &S,
        1 => &P,
        2 => &D,
        _ => &F,
    }
}

/// axtb's index (m + l) of every spherical function of shell `l` in Molden's order: p x, y,
/// z; otherwise m = 0, +1, -1, +2, -2, ...
fn molden_spherical(l: usize) -> Vec<usize> {
    if l == 1 {
        return vec![2, 0, 1];
    }
    let mut v = vec![l];
    for m in 1..=l {
        v.push(l + m);
        v.push(l - m);
    }
    v
}

/// `x` in Fortran-like E notation with `digits` decimals and a two-digit exponent.
fn sci(x: f64, digits: usize) -> String {
    let s = format!("{x:.digits$E}");
    match s.split_once('E') {
        Some((mant, exp)) => {
            let e: i32 = exp.parse().unwrap_or(0);
            let sign = if e < 0 { '-' } else { '+' };
            format!("{mant}E{sign}{:02}", e.abs())
        }
        None => s,
    }
}

impl Orbitals {
    /// Number of (spherical) basis functions.
    pub fn nao(&self) -> usize {
        self.shells.iter().flatten().map(|s| nsph(s.l)).sum()
    }

    fn check(&self) -> Result<()> {
        let nat = self.numbers.len();
        if self.positions.len() != nat || self.shells.len() != nat {
            return Err(Error::Input("orbitals: numbers, positions and shells must have one entry per atom".into()));
        }
        if let Some(&z) = self.numbers.iter().find(|&&z| z == 0 || z > 118) {
            return Err(Error::Input(format!("orbitals: no element with atomic number {z}")));
        }
        if let Some(s) = self.shells.iter().flatten().find(|s| s.l > 3) {
            return Err(Error::Input(format!("Molden output supports s, p, d and f shells, not l = {}", s.l)));
        }
        let nao = self.nao();
        if self.sets.is_empty() || self.sets.len() > 2 {
            return Err(Error::Input("orbitals: one set (closed shell) or two (alpha, beta) expected".into()));
        }
        for set in &self.sets {
            let nmo = set.energies.len();
            if set.occupations.len() != nmo || set.coefficients.len() != nmo * nao {
                return Err(Error::Input(format!(
                    "orbitals: {nmo} energies, {} occupations and {} coefficients for {nao} basis functions",
                    set.occupations.len(),
                    set.coefficients.len()
                )));
            }
        }
        Ok(())
    }

    /// The orbitals as the text of a Molden file (see the [module docs](self)); `title` goes
    /// into the `[Title]` section.
    pub fn to_molden(&self, functions: MoldenFunctions, title: &str) -> Result<String> {
        self.check()?;
        let mut out = String::new();
        out.push_str("[Molden Format]\n[Title]\n");
        out.push_str(title.lines().next().unwrap_or("").trim());
        out.push('\n');
        out.push_str("[Atoms] AU\n");
        for (i, (&z, p)) in self.numbers.iter().zip(&self.positions).enumerate() {
            let _ = writeln!(
                out,
                "{:<2} {:>6} {:>4} {:>22} {:>22} {:>22}",
                crate::io::symbol(z).unwrap_or("X"),
                i + 1,
                z,
                sci(p[0], 14),
                sci(p[1], 14),
                sci(p[2], 14)
            );
        }
        // per shell: the norm of the contracted function (the MO coefficients are scaled
        // by it, the written contraction is normalized)
        let mut norms = Vec::new();
        out.push_str("[GTO]\n");
        for (i, shells) in self.shells.iter().enumerate() {
            let _ = writeln!(out, "{:>6} 0", i + 1);
            for sh in shells {
                let (d, norm) = normalized_contraction(sh);
                norms.push(norm);
                let _ = writeln!(out, "{} {:>4} 1.00", ['s', 'p', 'd', 'f'][sh.l], sh.nprim);
                for (a, c) in sh.alpha.iter().zip(&d) {
                    let _ = writeln!(out, "{:>24} {:>24}", sci(*a, 16), sci(*c, 16));
                }
            }
            out.push('\n');
        }
        let maxl = self.shells.iter().flatten().map(|s| s.l).max().unwrap_or(0);
        match functions {
            MoldenFunctions::Spherical => {
                if maxl >= 3 {
                    out.push_str("[7F]\n");
                }
                if maxl >= 2 {
                    out.push_str("[5D]\n");
                }
            }
            MoldenFunctions::Cartesian => {
                if maxl >= 2 {
                    out.push_str("[6D]\n");
                }
                if maxl >= 3 {
                    out.push_str("[10F]\n");
                }
            }
        }
        // (first axtb AO, l, norm) of every shell
        let mut layout = Vec::new();
        let mut iao = 0;
        for (sh, &norm) in self.shells.iter().flatten().zip(&norms) {
            layout.push((iao, sh.l, norm));
            iao += nsph(sh.l);
        }
        let nao = iao;
        out.push_str("[MO]\n");
        let mut coef = Vec::new();
        for set in &self.sets {
            for (k, (e, occ)) in set.energies.iter().zip(&set.occupations).enumerate() {
                let c = &set.coefficients[k * nao..(k + 1) * nao];
                coef.clear();
                for &(i0, l, norm) in &layout {
                    match functions {
                        MoldenFunctions::Spherical => {
                            coef.extend(molden_spherical(l).into_iter().map(|m| norm * c[i0 + m]))
                        }
                        MoldenFunctions::Cartesian => cartesian_coefficients(l, norm, &c[i0..i0 + nsph(l)], &mut coef),
                    }
                }
                let _ = writeln!(out, "Sym= {}a", k + 1);
                let _ = writeln!(out, "Ene= {}", sci(*e, 14));
                let _ = writeln!(out, "Spin= {}", set.spin.label());
                let _ = writeln!(out, "Occup= {:.10}", occ);
                for (j, x) in coef.iter().enumerate() {
                    let _ = writeln!(out, "{:>6} {:>24}", j + 1, sci(*x, 16));
                }
            }
        }
        Ok(out)
    }
}

/// Coefficients of the separately normalized Cartesian functions (Molden order) of a shell
/// with spherical coefficients `c` (axtb order) and contraction norm `norm`.
fn cartesian_coefficients(l: usize, norm: f64, c: &[f64], out: &mut Vec<f64>) {
    let (ns, nc) = (nsph(l), ncart(l));
    let cca = cart_exponents(l);
    let t = sph_trafo(l);
    for e in molden_cartesian(l) {
        let ic = cca.iter().position(|x| x == e).expect("Cartesian component");
        // x^a y^b z^c with the norm of x^l has norm sqrt((2a-1)!! (2b-1)!! (2c-1)!! / (2l-1)!!)
        let ratio = (DFACT[e[0]] * DFACT[e[1]] * DFACT[e[2]] / DFACT[l]).sqrt();
        let v: f64 = (0..ns).map(|m| c[m] * t[m * nc + ic]).sum();
        out.push(norm * ratio * v);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::integrals::{overlap, MBLK};

    /// A shell with a normalized contraction of `alpha` with weights `w` (normalized
    /// primitives), times `scale`.
    fn shell(l: usize, alpha: &[f64], w: &[f64], scale: f64) -> Cgto {
        let mut c = Cgto { l, nprim: alpha.len(), alpha: [0.0; 16], coeff: [0.0; 16] };
        let mut s = 0.0;
        for i in 0..alpha.len() {
            for j in 0..alpha.len() {
                let (a, b) = (alpha[i], alpha[j]);
                s += w[i] * w[j] * (2.0 * (a * b).sqrt() / (a + b)).powf(l as f64 + 1.5);
            }
        }
        for k in 0..alpha.len() {
            c.alpha[k] = alpha[k];
            c.coeff[k] = scale * w[k] * primitive_norm(l, alpha[k]) / s.sqrt();
        }
        c
    }

    /// Orbitals read back: spin, occupation, coefficients (axtb AO order).
    type Mo = (String, f64, Vec<f64>);

    /// Minimal reader of a spherical file written above: shells (as Cgto, normalized) and
    /// orbitals in axtb's AO order.
    fn read(text: &str) -> (Vec<[f64; 3]>, Vec<Vec<Cgto>>, Vec<Mo>) {
        let mut pos = Vec::new();
        let mut shells: Vec<Vec<Cgto>> = Vec::new();
        let mut mos: Vec<Mo> = Vec::new();
        let mut section = String::new();
        let mut lines = text.lines().peekable();
        while let Some(line) = lines.next() {
            let t = line.trim();
            if t.starts_with('[') {
                section = t.to_ascii_lowercase();
                continue;
            }
            let f: Vec<&str> = t.split_whitespace().collect();
            if f.is_empty() {
                continue;
            }
            if section.starts_with("[atoms]") {
                pos.push([f[3].parse().unwrap(), f[4].parse().unwrap(), f[5].parse().unwrap()]);
            } else if section == "[gto]" {
                if f.len() == 2 && f[1] == "0" {
                    shells.push(Vec::new());
                } else {
                    let l = "spdf".find(f[0]).unwrap();
                    let n: usize = f[1].parse().unwrap();
                    let (mut a, mut w) = (Vec::new(), Vec::new());
                    for _ in 0..n {
                        let p: Vec<f64> =
                            lines.next().unwrap().split_whitespace().map(|x| x.parse().unwrap()).collect();
                        a.push(p[0]);
                        w.push(p[1]);
                    }
                    shells.last_mut().unwrap().push(shell(l, &a, &w, 1.0));
                }
            } else if section == "[mo]" {
                if f[0] == "Sym=" {
                    mos.push((String::new(), 0.0, Vec::new()));
                } else if f[0] == "Spin=" {
                    mos.last_mut().unwrap().0 = f[1].into();
                } else if f[0] == "Occup=" {
                    mos.last_mut().unwrap().1 = f[1].parse().unwrap();
                } else if f[0] != "Ene=" {
                    mos.last_mut().unwrap().2.push(f[1].parse().unwrap());
                }
            }
        }
        // Molden order -> axtb order
        for mo in mos.iter_mut() {
            let mut c = vec![0.0; mo.2.len()];
            let mut i = 0;
            for sh in shells.iter().flatten() {
                for (j, m) in molden_spherical(sh.l).into_iter().enumerate() {
                    c[i + m] = mo.2[i + j];
                }
                i += nsph(sh.l);
            }
            mo.2 = c;
        }
        (pos, shells, mos)
    }

    fn overlap_matrix(pos: &[[f64; 3]], shells: &[Vec<Cgto>]) -> (usize, Vec<f64>) {
        let nao: usize = shells.iter().flatten().map(|s| nsph(s.l)).sum();
        let mut s = vec![0.0; nao * nao];
        let mut blk = [0.0; MBLK];
        let mut ia0 = 0;
        for (a, sa) in shells.iter().enumerate() {
            for cha in sa {
                let mut ib0 = 0;
                for (b, sb) in shells.iter().enumerate() {
                    for chb in sb {
                        let v = [pos[a][0] - pos[b][0], pos[a][1] - pos[b][1], pos[a][2] - pos[b][2]];
                        overlap(chb, cha, v, 1e3, &mut blk);
                        for i in 0..nsph(cha.l) {
                            for j in 0..nsph(chb.l) {
                                s[(ia0 + i) * nao + ib0 + j] = blk[j + nsph(chb.l) * i];
                            }
                        }
                        ib0 += nsph(chb.l);
                    }
                }
                ia0 += nsph(cha.l);
            }
        }
        (nao, s)
    }

    #[test]
    fn spherical_round_trip_keeps_orbitals() {
        // two atoms with s, p, d, f shells whose contractions are not normalized (scale
        // 1.7 etc.): orbitals orthonormal under the true overlap must come back orthonormal
        let pos = vec![[0.0, 0.0, 0.0], [0.3, -0.8, 1.9]];
        let a = [3.1, 0.9, 0.25];
        let w = [0.2, 0.5, 0.4];
        let shells = vec![
            vec![shell(0, &a, &w, 1.7), shell(1, &a, &[0.3, 0.6, 0.2], 0.6), shell(2, &a, &[0.1, 0.4, 0.6], 2.2)],
            vec![shell(0, &a[1..], &w[1..], 0.9), shell(3, &a, &[0.2, 0.3, 0.7], 1.3), shell(1, &a[..2], &w[..2], 1.0)],
        ];
        let (nao, s) = overlap_matrix(&pos, &shells);
        // orthonormal orbitals: Löwdin S^-1/2 by Gram-Schmidt on a dense random start
        let mut c = vec![0.0; nao * nao];
        for k in 0..nao {
            for mu in 0..nao {
                c[k * nao + mu] = ((k * 31 + mu * 17) % 13) as f64 - 6.0 + if k == mu { 9.0 } else { 0.0 };
            }
            for _ in 0..2 {
                for j in 0..k {
                    let ov: f64 = (0..nao)
                        .map(|m| (0..nao).map(|n| c[k * nao + m] * s[m * nao + n] * c[j * nao + n]).sum::<f64>())
                        .sum();
                    for m in 0..nao {
                        c[k * nao + m] -= ov * c[j * nao + m];
                    }
                }
            }
            let nn: f64 =
                (0..nao).map(|m| (0..nao).map(|n| c[k * nao + m] * s[m * nao + n] * c[k * nao + n]).sum::<f64>()).sum();
            for m in 0..nao {
                c[k * nao + m] /= nn.sqrt();
            }
        }
        let orb = Orbitals {
            numbers: vec![26, 92],
            positions: pos,
            shells,
            sets: vec![
                MoSet {
                    spin: Spin::Alpha,
                    energies: (0..nao).map(|k| k as f64 * 0.1 - 1.0).collect(),
                    occupations: vec![1.0; nao],
                    coefficients: c.clone(),
                },
                MoSet { spin: Spin::Beta, energies: vec![0.0; nao], occupations: vec![0.0; nao], coefficients: c },
            ],
        };
        let text = orb.to_molden(MoldenFunctions::Spherical, "test").unwrap();
        assert!(text.contains("[7F]\n[5D]\n"));
        let (p2, sh2, mos) = read(&text);
        assert_eq!(mos.len(), 2 * nao);
        assert_eq!(mos[nao].0, "Beta");
        let (n2, s2) = overlap_matrix(&p2, &sh2);
        assert_eq!(n2, nao);
        let mut worst: f64 = 0.0;
        for i in 0..nao {
            for j in 0..nao {
                let (ci, cj) = (&mos[i].2, &mos[j].2);
                let ov: f64 = (0..nao).map(|m| (0..nao).map(|n| ci[m] * s2[m * nao + n] * cj[n]).sum::<f64>()).sum();
                worst = worst.max((ov - if i == j { 1.0 } else { 0.0 }).abs());
            }
        }
        assert!(worst < 1e-12, "orthonormality lost: {worst:e}");
    }

    #[test]
    fn cartesian_components_reproduce_spherical_functions() {
        // the Cartesian coefficients of each unit spherical function, evaluated with
        // individually normalized Cartesian Gaussians, give a function of norm 1
        for l in 0..=3 {
            let cca = molden_cartesian(l);
            for m in 0..nsph(l) {
                let mut c = vec![0.0; nsph(l)];
                c[m] = 1.0;
                let mut cart = Vec::new();
                cartesian_coefficients(l, 1.0, &c, &mut cart);
                // <x^a y^b z^c|x^d y^e z^f> of normalized components, same radial part:
                // prod (2a+2d-1)!!/sqrt((4a-1)!!(4d-1)!!) per axis (only even sums survive)
                let df = |n: i64| -> f64 {
                    if n <= 0 {
                        1.0
                    } else {
                        (1..=n).rev().step_by(2).map(|k| k as f64).product()
                    }
                };
                let mut norm = 0.0;
                for (i, a) in cca.iter().enumerate() {
                    for (j, b) in cca.iter().enumerate() {
                        let mut v = 1.0;
                        for x in 0..3 {
                            let s = (a[x] + b[x]) as i64;
                            if s % 2 == 1 {
                                v = 0.0;
                                break;
                            }
                            v *= df(s - 1) / (df(2 * a[x] as i64 - 1) * df(2 * b[x] as i64 - 1)).sqrt();
                        }
                        norm += cart[i] * cart[j] * v;
                    }
                }
                assert!((norm - 1.0).abs() < 1e-13, "l={l} m={m}: norm {norm}");
            }
        }
    }

    #[test]
    fn formats() {
        assert_eq!(sci(-0.0123, 4), "-1.2300E-02");
        assert_eq!(sci(1234.5, 2), "1.23E+03");
        assert_eq!(molden_spherical(2), vec![2, 3, 1, 4, 0]);
        assert_eq!(molden_spherical(3), vec![3, 4, 2, 5, 1, 6, 0]);
        assert!("cartesian".parse::<MoldenFunctions>().unwrap() == MoldenFunctions::Cartesian);
        assert!("x".parse::<MoldenFunctions>().is_err());
    }
}
