//! Parsers for the three g-xTB v1.1 parameter files.
//!
//! The layouts were established from the read statements of the program (`rdpar_`,
//! `rdbas_`, `rdpar_eeq_` in the g-xTB 1.1.0 binary) and by perturbing single entries;
//! `docs/gxtb-v1.1-spec.md` documents every field and the term that uses it. Field names
//! below follow the program's internal array names (`parcom::gp3_*`, `eeqcom::eeq_*`) so
//! the mapping stays traceable.

use axtb::{Error, Result};

/// Highest atomic number of the g-xTB tables.
pub const MAX_ELEM: usize = 103;

/// The g-xTB v1.1 method parameter file (`gxtb_parameters`, read as `~/.gxtb`).
pub const GXTB_PARAMETERS: &str = include_str!("../param/gxtb_parameters");
/// The q-vSZP basis file (`basisq`, read as `~/.basisq`).
pub const BASISQ: &str = include_str!("../param/basisq");
/// The EEQ-BC charge model parameter file (`eeq`, read as `~/.eeq`).
pub const EEQ: &str = include_str!("../param/eeq");

/// Element block of `gxtb_parameters` (10 lines: Z, then 10 + 6 x 4 + 8 + 8 numbers).
///
/// Shell-resolved arrays are indexed by the element's shells in basis order (column
/// 0..3 of lines 2-7 of the block), not by angular momentum.
#[derive(Clone, Debug, PartialEq)]
pub struct ElementParams {
    /// Line 1, columns 0-4, and line 8 column 6: repulsion parameters (`gp3_rep(1:6)`).
    pub rep: [f64; 6],
    /// Line 1 col 5: covalent radius (bohr) of the g-xTB CN and the H0 distance polynomial (`gp3_rad`).
    pub rad: f64,
    /// Line 1 col 6: CN dependence of the shell Hubbard parameters (`gp3_g2cn`).
    pub g2cn: f64,
    /// Line 1 col 7: element factor of the H0 distance polynomial (`gp3_poly`).
    pub poly: f64,
    /// Line 1 col 8: CN dependence of the first-order chemical potential (`gp3_e1cn`).
    pub e1cn: f64,
    /// Line 1 col 9: spin-polarization scaling k^W_A (`gp3_atspin`).
    pub atspin: f64,
    /// Line 2: shell parameter `gp3_lev` (H0 shell level).
    pub lev: [f64; 4],
    /// Line 3: CN dependence of the shell level (`gp3_lev_cn`).
    pub lev_cn: [f64; 4],
    /// Line 4: shell exponent scaling of the H0 basis (`gp3_pln`).
    pub pln: [f64; 4],
    /// Line 5: shell Hubbard parameter of the Mulliken-approximated Fock exchange (`gp3_fam`).
    pub fam: [f64; 4],
    /// Line 6: shell Hubbard parameter of the second-order term (`gp3_gam2`).
    pub gam2: [f64; 4],
    /// Line 7: shell first-order chemical potential (`gp3_gam1`).
    pub gam1: [f64; 4],
    /// Line 8 cols 0-1: diatomic-frame overlap scaling of H0 (`gp3_h0k`).
    pub h0k: [f64; 2],
    /// Line 8 cols 2-4: effective-charge prefactors of the H0 basis (`gp3_kqvszp`).
    pub kqvszp: [f64; 3],
    /// Line 8 col 5: atomic energy increment (Eh) (`gp3_at`).
    pub increment: f64,
    /// Line 8 col 7: CN dependence of the ACP coefficients (`gp3_acpcn`).
    pub acpcn: f64,
    /// Line 9 cols 0-3: ACP projector coefficients for l = s, p, d, f (`gp3_acp(1, l)`).
    pub acp_coeff: [f64; 4],
    /// Line 9 cols 4-7: ACP projector exponents for l = s, p, d, f (`gp3_acp(2, l)`).
    pub acp_exp: [f64; 4],
}

/// The parsed `gxtb_parameters` file.
#[derive(Clone, Debug, PartialEq)]
pub struct GxtbParams {
    /// Twenty global parameters (`gp3_glob(1:20)`, file lines 1-2).
    pub glob: [f64; 20],
    /// Element blocks indexed by atomic number (index 0 unused); `None` if absent.
    pub elements: Vec<Option<ElementParams>>,
}

fn numbers(line: &str, what: &str) -> Result<Vec<f64>> {
    line.split_whitespace()
        .map(|t| t.replace(['d', 'D'], "e").parse::<f64>())
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|e| Error::Input(format!("{what}: cannot parse {line:?}: {e}")))
}

fn take<const N: usize>(v: &[f64], what: &str) -> Result<[f64; N]> {
    if v.len() < N {
        return Err(Error::Input(format!("{what}: expected {N} numbers, found {}", v.len())));
    }
    let mut out = [0.0; N];
    out.copy_from_slice(&v[..N]);
    Ok(out)
}

impl GxtbParams {
    /// The parameters shipped with g-xTB 1.1.0.
    pub fn v1_1() -> Self {
        Self::parse(GXTB_PARAMETERS).expect("bundled gxtb_parameters")
    }

    /// Parse the text of a `gxtb_parameters` file.
    pub fn parse(text: &str) -> Result<Self> {
        let lines: Vec<&str> = text.lines().filter(|l| !l.trim().is_empty()).collect();
        if lines.len() < 2 {
            return Err(Error::Input("gxtb_parameters: missing global lines".into()));
        }
        let g1 = numbers(lines[0], "gxtb_parameters global line 1")?;
        let g2 = numbers(lines[1], "gxtb_parameters global line 2")?;
        let mut glob = [0.0; 20];
        glob[..10].copy_from_slice(&take::<10>(&g1, "global line 1")?);
        glob[10..].copy_from_slice(&take::<10>(&g2, "global line 2")?);
        let mut elements = vec![None; MAX_ELEM + 1];
        let mut i = 2;
        while i < lines.len() {
            let z: usize = lines[i]
                .trim()
                .parse()
                .map_err(|_| Error::Input(format!("gxtb_parameters: expected atomic number, got {:?}", lines[i])))?;
            if z == 0 || z > MAX_ELEM {
                return Err(Error::Input(format!("gxtb_parameters: atomic number {z} out of range")));
            }
            if i + 9 >= lines.len() {
                return Err(Error::Input(format!("gxtb_parameters: truncated block for Z={z}")));
            }
            let what = format!("gxtb_parameters Z={z}");
            let l: Vec<Vec<f64>> = (1..=9).map(|k| numbers(lines[i + k], &what)).collect::<Result<_>>()?;
            let l1 = take::<10>(&l[0], &what)?;
            let l8 = take::<8>(&l[7], &what)?;
            let l9 = take::<8>(&l[8], &what)?;
            let shell = |k: usize| take::<4>(&l[k], &what);
            elements[z] = Some(ElementParams {
                rep: [l1[0], l1[1], l1[2], l1[3], l1[4], l8[6]],
                rad: l1[5],
                g2cn: l1[6],
                poly: l1[7],
                e1cn: l1[8],
                atspin: l1[9],
                lev: shell(1)?,
                lev_cn: shell(2)?,
                pln: shell(3)?,
                fam: shell(4)?,
                gam2: shell(5)?,
                gam1: shell(6)?,
                h0k: [l8[0], l8[1]],
                kqvszp: [l8[2], l8[3], l8[4]],
                increment: l8[5],
                acpcn: l8[7],
                acp_coeff: [l9[0], l9[1], l9[2], l9[3]],
                acp_exp: [l9[4], l9[5], l9[6], l9[7]],
            });
            i += 10;
        }
        Ok(Self { glob, elements })
    }

    /// Parameters of element `z`.
    pub fn element(&self, z: usize) -> Result<&ElementParams> {
        self.elements.get(z).and_then(|e| e.as_ref()).ok_or(Error::UnsupportedElement { method: "g-xTB v1.1", z })
    }
}

/// One contracted shell of the q-vSZP basis: exponents and the charge-independent and
/// charge-linear contraction coefficients, c_i(q_eff) = c0_i + c1_i q_eff (unnormalized
/// primitives exp(-alpha r^2)).
#[derive(Clone, Debug, PartialEq)]
pub struct QvszpShell {
    pub l: usize,
    pub alpha: Vec<f64>,
    pub c0: Vec<f64>,
    pub c1: Vec<f64>,
}

/// q-vSZP data of one element.
#[derive(Clone, Debug, PartialEq)]
pub struct QvszpElement {
    /// Coefficient of sqrt(CN) in the effective charge (header column 1).
    pub k_cn: f64,
    /// Coefficient of -q^2 in the effective charge (header column 2).
    pub k_q2: f64,
    /// Coefficient of q*CN in the effective charge (header column 3).
    pub k_qcn: f64,
    pub shells: Vec<QvszpShell>,
}

/// The parsed `basisq` file.
#[derive(Clone, Debug, PartialEq)]
pub struct QvszpParams {
    /// Indexed by atomic number (index 0 unused).
    pub elements: Vec<Option<QvszpElement>>,
}

impl QvszpParams {
    /// The basis shipped with g-xTB 1.1.0.
    pub fn v1_1() -> Self {
        Self::parse(BASISQ).expect("bundled basisq")
    }

    /// Parse the text of a `basisq` file: blocks separated by `*` lines, each with a header
    /// `Z k_cn k_q2 k_qcn` and shells `nprim l` followed by `nprim` lines `alpha c0 c1`.
    pub fn parse(text: &str) -> Result<Self> {
        let mut elements = vec![None; MAX_ELEM + 1];
        let mut lines = text.lines().map(str::trim).filter(|l| !l.is_empty()).peekable();
        while let Some(line) = lines.next() {
            if line.starts_with('*') {
                continue;
            }
            let head: Vec<&str> = line.split_whitespace().collect();
            if head.len() != 4 {
                return Err(Error::Input(format!("basisq: expected element header, got {line:?}")));
            }
            let z: usize = head[0].parse().map_err(|_| Error::Input(format!("basisq: bad header {line:?}")))?;
            if z == 0 || z > MAX_ELEM {
                return Err(Error::Input(format!("basisq: atomic number {z} out of range")));
            }
            let k = numbers(&head[1..].join(" "), "basisq header")?;
            let mut shells = Vec::new();
            while let Some(&next) = lines.peek() {
                if next.starts_with('*') {
                    break;
                }
                let sh: Vec<&str> = next.split_whitespace().collect();
                if sh.len() != 2 {
                    return Err(Error::Input(format!("basisq Z={z}: expected shell line, got {next:?}")));
                }
                lines.next();
                let nprim: usize = sh[0].parse().map_err(|_| Error::Input(format!("basisq: bad shell {next:?}")))?;
                let l = match sh[1].to_ascii_lowercase().as_str() {
                    "s" => 0,
                    "p" => 1,
                    "d" => 2,
                    "f" => 3,
                    "g" => 4,
                    other => return Err(Error::Input(format!("basisq Z={z}: unknown shell type {other:?}"))),
                };
                let mut shell = QvszpShell { l, alpha: Vec::new(), c0: Vec::new(), c1: Vec::new() };
                for _ in 0..nprim {
                    let p = lines.next().ok_or_else(|| Error::Input(format!("basisq Z={z}: truncated shell")))?;
                    let v = numbers(p, "basisq primitive")?;
                    let [a, c0, c1] = take::<3>(&v, "basisq primitive")?;
                    shell.alpha.push(a);
                    shell.c0.push(c0);
                    shell.c1.push(c1);
                }
                shells.push(shell);
            }
            elements[z] = Some(QvszpElement { k_cn: k[0], k_q2: k[1], k_qcn: k[2], shells });
        }
        Ok(Self { elements })
    }

    pub fn element(&self, z: usize) -> Result<&QvszpElement> {
        self.elements.get(z).and_then(|e| e.as_ref()).ok_or(Error::UnsupportedElement { method: "q-vSZP", z })
    }
}

/// EEQ-BC element parameters (one line of the `eeq` file; columns 8-9 are unused zeros).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct EeqBcElement {
    /// Col 0: electronegativity chi (`eeq_xi`).
    pub chi: f64,
    /// Col 1: chemical hardness eta (`eeq_gam2`).
    pub eta: f64,
    /// Col 2: Gaussian charge width (`eeq_alp`).
    pub rad: f64,
    /// Col 3: CN scaling of chi (`eeq_cn1`).
    pub kcnchi: f64,
    /// Col 4: twice the covalent radius of the CN, bohr (`eeq_rad`); the CN uses
    /// R_cov,A + R_cov,B = (col4_A + col4_B) / 2.
    pub rcov2: f64,
    /// Col 5: local-charge scaling of eta (`eeq_gamq`).
    pub kqeta: f64,
    /// Col 6: local-charge scaling of chi (`eeq_cn2`).
    pub kqchi: f64,
    /// Col 7: bond capacitance (`eeq_boca`).
    pub cap: f64,
}

/// The parsed `eeq` file: 103 lines, one per element, then a date line.
#[derive(Clone, Debug, PartialEq)]
pub struct EeqBcParams {
    /// Indexed by atomic number (index 0 unused).
    pub elements: Vec<EeqBcElement>,
}

impl EeqBcParams {
    pub fn v1_1() -> Self {
        Self::parse(EEQ).expect("bundled eeq")
    }

    pub fn parse(text: &str) -> Result<Self> {
        let zero =
            EeqBcElement { chi: 0.0, eta: 0.0, rad: 0.0, kcnchi: 0.0, rcov2: 0.0, kqeta: 0.0, kqchi: 0.0, cap: 0.0 };
        let mut elements = vec![zero; MAX_ELEM + 1];
        let mut z = 0;
        for line in text.lines() {
            if z == MAX_ELEM {
                break;
            }
            if line.trim().is_empty() {
                continue;
            }
            z += 1;
            let v = take::<8>(&numbers(line, "eeq")?, &format!("eeq line {z}"))?;
            elements[z] = EeqBcElement {
                chi: v[0],
                eta: v[1],
                rad: v[2],
                kcnchi: v[3],
                rcov2: v[4],
                kqeta: v[5],
                kqchi: v[6],
                cap: v[7],
            };
        }
        if z < MAX_ELEM {
            return Err(Error::Input(format!("eeq: only {z} element lines")));
        }
        Ok(Self { elements })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bundled_files_parse() {
        let p = GxtbParams::v1_1();
        assert_eq!(p.glob[0], 0.7188463521);
        assert_eq!(p.glob[19], 0.3042947280);
        let present: Vec<usize> = (1..=MAX_ELEM).filter(|&z| p.elements[z].is_some()).collect();
        assert_eq!(present.len(), 79);
        let o = p.element(8).unwrap();
        assert_eq!(o.increment, -69.0180177045);
        assert_eq!(o.rep[5], 0.0200866014);
        assert_eq!(o.acp_exp[3], 0.6443090229);
        assert!(p.element(64).is_err());

        let b = QvszpParams::v1_1();
        assert!((1..=MAX_ELEM).all(|z| b.elements[z].is_some()));
        let o = b.element(8).unwrap();
        assert_eq!((o.k_cn, o.k_q2, o.k_qcn), (0.313710722100, 0.011817698900, 0.179556613700));
        assert_eq!(o.shells.len(), 2);
        assert_eq!(o.shells[1].l, 1);
        assert_eq!(o.shells[1].alpha[0], 37.641471325500);

        let e = EeqBcParams::v1_1();
        assert_eq!(e.elements[1].chi, 1.7500687479);
        assert_eq!(e.elements[103].cap, 1.1930672532);
        assert_eq!(e.elements[8].rcov2, 2.6466398902);
    }
}
