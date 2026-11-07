//! Analytic gradients of g-xTB v1.1 ([`Setup::gradient`]) against finite differences of our
//! own energies at tight SCF convergence, for the exact Fock matrix (default, stationary SCF)
//! and the program's (with the Z-vector response), and against the g-xTB 1.1.0 program's
//! numerical gradients (program conventions).
//!
//! The program's gradient (`-grad`) is a central difference with h = 0.0075 bohr for every
//! atom but the first, whose gradient it sets to minus the sum of the others (translational
//! invariance). It carries the h^2 truncation error (up to ~2e-5 Eh/bohr here) and, with
//! the default SCF thresholds (`gradient`), SCF noise of up to ~2e-5 Eh/bohr, because the
//! energy is not stationary in the density. `gradient_tight` is the same run with the SCF
//! thresholds tightened to 1e-12 Eh / 1e-10 (gdb, `scripts/make_reference.py`).

use axtb_gxtb::{read_xyz, Calculator, Conventions, ScfOptions, Setup};
use serde_json::Value;
use std::path::Path;

const ALL: [&str; 14] = [
    "water",
    "methane",
    "ethanol",
    "benzene",
    "glycine",
    "acetate",
    "hydroxide",
    "ammonium",
    "dioxygen_triplet",
    "mesyl_chloride",
    "iodomethane",
    "silane_phosphine",
    "nickel_tetracarbonyl",
    "tetrachlorocuprate",
];

fn dir(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/reference").join(name)
}

fn molecule(name: &str) -> axtb::Molecule {
    let input: Value = serde_json::from_str(&std::fs::read_to_string(dir(name).join("input.json")).unwrap()).unwrap();
    read_xyz(
        &std::fs::read_to_string(dir(name).join("mol.xyz")).unwrap(),
        input["charge"].as_f64().unwrap(),
        Some(input["uhf"].as_u64().unwrap() as usize),
    )
    .unwrap()
}

/// TURBOMOLE `gradient` file of the program.
fn program_gradient(name: &str, file: &str, nat: usize) -> Vec<[f64; 3]> {
    let text = std::fs::read_to_string(dir(name).join(file)).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    let first = lines.iter().position(|l| l.contains("cycle")).unwrap() + 1 + nat;
    (0..nat)
        .map(|i| {
            let v: Vec<f64> =
                lines[first + i].split_whitespace().map(|x| x.replace('D', "E").parse().unwrap()).collect();
            [v[0], v[1], v[2]]
        })
        .collect()
}

fn tight(conventions: Conventions) -> ScfOptions {
    ScfOptions { energy_tol: Some(1e-13), density_tol: Some(1e-11), conventions, ..Default::default() }
}

fn energy(mol: &axtb::Molecule, atom: usize, dir: usize, d: f64, c: Conventions) -> f64 {
    let mut m = mol.clone();
    m.positions[atom][dir] += d;
    Setup::new(&m).unwrap().scf(&tight(c)).unwrap().energies.total
}

fn analytic(mol: &axtb::Molecule, opts: &ScfOptions) -> Vec<[f64; 3]> {
    let st = Setup::new(mol).unwrap();
    st.gradient(&st.scf(opts).unwrap()).unwrap()
}

/// Closed and open shells, d shells, anions; five-point differences of tight energies.
fn check_finite_differences(c: Conventions) {
    let h = 5e-4;
    for name in ["water", "dioxygen_triplet", "mesyl_chloride", "nickel_tetracarbonyl", "tetrachlorocuprate"] {
        let mol = molecule(name);
        let g = analytic(&mol, &tight(c));
        for a in 0..mol.nat() {
            for k in 0..3 {
                let e = |d: f64| energy(&mol, a, k, d, c);
                let fd = (8.0 * (e(h) - e(-h)) - (e(2.0 * h) - e(-2.0 * h))) / (12.0 * h);
                assert!((fd - g[a][k]).abs() < 2e-9, "{c:?} {name} atom {a} {k}: analytic {} FD {fd}", g[a][k]);
            }
        }
    }
}

#[test]
fn gradient_matches_finite_differences_exact_fock() {
    check_finite_differences(Conventions::Axtb);
}

#[test]
fn gradient_matches_finite_differences_program_fock() {
    check_finite_differences(Conventions::Gxtb);
}

fn check_program(names: &[&str]) {
    let h = 0.0075;
    for &name in names {
        let mol = molecule(name);
        let nat = mol.nat();
        let prog = program_gradient(name, "gradient", nat);
        let prog_tight = program_gradient(name, "gradient_tight", nat);
        let c = Conventions::Gxtb;
        let g = analytic(&mol, &tight(c));
        // the program's first atom: minus the sum of the others
        for k in 0..3 {
            let s: f64 = prog_tight[1..].iter().map(|v| v[k]).sum();
            assert!((prog_tight[0][k] + s).abs() < 1e-11, "{name}: first atom {k}");
        }
        for a in 0..nat {
            for k in 0..3 {
                let fd = |h: f64| (energy(&mol, a, k, h, c) - energy(&mol, a, k, -h, c)) / (2.0 * h);
                let (f1, f2) = (fd(h), fd(h / 2.0));
                // the program's central differences are ours (its energies at the displaced
                // geometries are reproduced) ...
                if a > 0 {
                    assert!(
                        (prog_tight[a][k] - f1).abs() < 1e-9,
                        "{name} atom {a} {k}: program {} ours {f1}",
                        prog_tight[a][k]
                    );
                }
                // ... whose step-size error Richardson extrapolation removes
                let rich = (4.0 * f2 - f1) / 3.0;
                assert!((g[a][k] - rich).abs() < 1e-7, "{name} atom {a} {k}: analytic {} extrapolated {rich}", g[a][k]);
                // raw comparison with the program's default run (h^2 error and SCF noise)
                assert!(
                    (g[a][k] - prog[a][k]).abs() < 6e-5,
                    "{name} atom {a} {k}: analytic {} program {}",
                    g[a][k],
                    prog[a][k]
                );
            }
        }
    }
}

#[test]
fn gradient_matches_program_small() {
    check_program(&[
        "water",
        "methane",
        "hydroxide",
        "ammonium",
        "dioxygen_triplet",
        "iodomethane",
        "silane_phosphine",
    ]);
}

#[test]
fn gradient_matches_program_large() {
    check_program(&[
        "ethanol",
        "benzene",
        "glycine",
        "acetate",
        "mesyl_chloride",
        "nickel_tetracarbonyl",
        "tetrachlorocuprate",
    ]);
}

#[test]
fn gradient_is_translation_invariant_and_default_convergence_suffices() {
    // the default SCF thresholds (the program's) leave gradient errors of up to ~3e-6 Eh/bohr
    // (first order in the density error, with either Fock matrix); accuracy 0.01 gives ~1e-8
    for (c, tol) in [(Conventions::Axtb, 5e-6), (Conventions::Gxtb, 5e-6)] {
        for name in ALL {
            let mol = molecule(name);
            let opts = ScfOptions { conventions: c, ..Default::default() };
            let mut calc =
                Calculator::new(axtb_gxtb::Version::V1_1, &mol.numbers, &mol.positions, mol.charge, mol.uhf, opts)
                    .unwrap();
            let g = calc.singlepoint(&mol.positions, true).unwrap().singlepoint.gradient.unwrap();
            for k in 0..3 {
                let s: f64 = g.iter().map(|v| v[k]).sum();
                assert!(s.abs() < 1e-9, "{c:?} {name}: sum {k} = {s}");
            }
            let gt = analytic(&mol, &tight(c));
            for (a, b) in g.iter().zip(&gt) {
                for k in 0..3 {
                    assert!((a[k] - b[k]).abs() < tol, "{c:?} {name}: default {} tight {}", a[k], b[k]);
                }
            }
        }
    }
}
