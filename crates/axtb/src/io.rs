//! Structure input.

use crate::{Error, Result};

const SYMBOLS: [&str; 118] = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",
    "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce",
    "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir",
    "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
    "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc",
    "Lv", "Ts", "Og",
];

/// Atomic number (1 to 118) of an element symbol (case-insensitive) or a number string.
pub fn atomic_number(sym: &str) -> Result<usize> {
    let z = match sym.parse::<usize>() {
        Ok(z) => Some(z).filter(|z| (1..=SYMBOLS.len()).contains(z)),
        Err(_) => SYMBOLS.iter().position(|s| s.eq_ignore_ascii_case(sym)).map(|i| i + 1),
    };
    z.ok_or_else(|| Error::Input(format!("unknown element {sym:?}")))
}

/// Element symbol of atomic number `z` (`None` outside 1 to 118).
pub fn symbol(z: usize) -> Option<&'static str> {
    SYMBOLS.get(z.wrapping_sub(1)).copied()
}

/// Parse an XYZ file (Ångström) into atomic numbers and bohr positions, each coordinate
/// multiplied by `aatoau` (Ångström to bohr). The factor is part of the conventions:
/// [`crate::units::AATOAU`] (CODATA 2022) in axtb's own, mctc-lib's
/// [`crate::units::reference::MCTC_AATOAU`] where a program reads XYZ files with mctc-lib
/// (xtb 6.7.1 and 6.8.0, tblite, the g-xTB 2.0.1 program).
pub fn read_xyz(text: &str, aatoau: f64) -> Result<(Vec<usize>, Vec<[f64; 3]>)> {
    read_xyz_with(text, |x| x * aatoau)
}

/// [`read_xyz`] with each coordinate (Ångström) converted to bohr by `to_bohr`, for a program
/// that converts otherwise (the g-xTB 1.1.0 program divides by its bohr radius).
pub fn read_xyz_with(text: &str, to_bohr: impl Fn(f64) -> f64) -> Result<(Vec<usize>, Vec<[f64; 3]>)> {
    let mut lines = text.lines();
    let nat: usize = lines
        .next()
        .and_then(|l| l.trim().parse().ok())
        .ok_or_else(|| Error::Input("xyz: first line must hold the atom count".into()))?;
    lines.next();
    // no capacity from the count: a garbage count must not allocate
    let (mut numbers, mut positions) = (Vec::new(), Vec::new());
    for line in lines.take(nat) {
        let f: Vec<&str> = line.split_whitespace().collect();
        if f.len() < 4 {
            return Err(Error::Input(format!("xyz: bad line {line:?}")));
        }
        numbers.push(atomic_number(f[0])?);
        let mut r = [0.0; 3];
        for k in 0..3 {
            r[k] = to_bohr(f[k + 1].parse::<f64>().map_err(|_| Error::Input(format!("xyz: bad number in {line:?}")))?);
        }
        positions.push(r);
    }
    if numbers.len() != nat {
        return Err(Error::Input(format!("xyz: {} atoms, {nat} declared", numbers.len())));
    }
    Ok((numbers, positions))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::{reference::MCTC_AATOAU, AATOAU};

    #[test]
    fn xyz_errors_are_input_errors() {
        let (z, r) = read_xyz("2\n\nH 0 0 0\nh 0 0 0.74\n", AATOAU).unwrap();
        assert_eq!(z, [1, 1]);
        assert_eq!(r[1][2], 0.74 * AATOAU);
        assert_eq!(read_xyz("1\n\nH 0 0 0.74\n", MCTC_AATOAU).unwrap().1[0][2], 0.74 * MCTC_AATOAU);
        assert_eq!(read_xyz_with("1\n\nH 0 0 0.74\n", |x| x / 0.5).unwrap().1[0][2], 0.74 / 0.5);
        // a huge count must not allocate (it aborted the process)
        assert!(matches!(read_xyz("100000000000000\n\nH 0 0 0\n", AATOAU), Err(Error::Input(_))));
        for bad in ["", "x\n", "2\n\nH 0 0 0\n", "1\n\nH 0 0\n", "1\n\nH 0 a 0\n", "1\n\nXx 0 0 0\n", "1\n\n0 0 0 0\n"]
        {
            assert!(matches!(read_xyz(bad, AATOAU), Err(Error::Input(_))), "{bad:?}");
        }
        assert_eq!(atomic_number("118").unwrap(), 118);
        assert!(atomic_number("119").is_err());
        assert_eq!(symbol(8), Some("O"));
        assert_eq!((symbol(0), symbol(119)), (None, None));
    }
}
