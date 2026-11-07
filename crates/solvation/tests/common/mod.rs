// SPDX-License-Identifier: LGPL-3.0-or-later
//! Reader for the xtb reference files in tests/data (see scripts/xtb_reference.py).
#![allow(dead_code)]

use axtb_solvation::{Model, ReferenceState, SolvEnergies};
use std::path::{Path, PathBuf};

pub struct Case {
    pub name: String,
    pub model: Model,
    pub solvent: String,
    pub state: ReferenceState,
    pub numbers: Vec<usize>,
    pub positions: Vec<[f64; 3]>,
    pub charges: Vec<f64>,
    pub reference: SolvEnergies,
    /// requested number of Lebedev points (xtb gbsagrid)
    pub nang: usize,
}

pub fn parse(path: &Path) -> Case {
    let text = std::fs::read_to_string(path).unwrap();
    let mut c = Case {
        name: String::new(),
        model: Model::Alpb,
        solvent: String::new(),
        state: ReferenceState::Gsolv,
        numbers: vec![],
        positions: vec![],
        charges: vec![],
        reference: SolvEnergies::default(),
        nang: 230,
    };
    for line in text.lines() {
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }
        let t: Vec<&str> = line.split_whitespace().collect();
        let f = |s: &str| s.parse::<f64>().unwrap();
        match t[0] {
            "case" => c.name = t[1].to_string(),
            "model" => {
                c.model = match t[1] {
                    "alpb" => Model::Alpb,
                    "gbsa" => Model::Gbsa,
                    m => panic!("model {m}"),
                }
            }
            "solvent" => c.solvent = t[1].to_string(),
            "nang" => c.nang = t[1].parse().unwrap(),
            "state" => {
                c.state = match t[1] {
                    "gsolv" => ReferenceState::Gsolv,
                    "bar1M" => ReferenceState::Bar1M,
                    "reference" => ReferenceState::Reference,
                    s => panic!("state {s}"),
                }
            }
            "atom" => {
                c.numbers.push(t[1].parse().unwrap());
                c.positions.push([f(t[2]), f(t[3]), f(t[4])]);
                c.charges.push(f(t[5]));
            }
            "gsolv" => c.reference.gsolv = f(t[1]),
            "gelec" => c.reference.gelec = f(t[1]),
            "gsasa" => c.reference.gsasa = f(t[1]),
            "ghb" => c.reference.ghb = f(t[1]),
            "gshift" => c.reference.gshift = f(t[1]),
            _ => {}
        }
    }
    c
}

pub fn ref_files() -> Vec<PathBuf> {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let mut v: Vec<PathBuf> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().is_some_and(|e| e == "ref"))
        .collect();
    v.sort();
    v
}

/// Load the reference case of the given name.
pub fn load(name: &str) -> Case {
    parse(&Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data").join(format!("{name}.ref")))
}

/// Solvation model for a reference case, with its grid option.
pub fn model(c: &Case) -> axtb_solvation::Solvation {
    // the reference data are xtb 6.7.1's (scripts/xtb_reference.py)
    let opts = axtb_solvation::SolvOptions {
        n_ang: c.nang,
        conventions: axtb_solvation::Conventions::Xtb671,
        ..Default::default()
    };
    axtb_solvation::Solvation::with_options(&c.numbers, &c.solvent, c.model, c.state, &opts).unwrap()
}
