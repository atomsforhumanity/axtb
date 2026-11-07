//! g-xTB 2.0.1 analytic gradients against finite differences of the energy and against the
//! 2.0.1 program's analytic gradients (tests/reference_v2/*/*/gradient_tight).

use axtb_gxtb::v2::scf::Setup;
use axtb_gxtb::{Conventions, ScfOptions};
use serde_json::Value;
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/reference_v2")
}

fn json(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn molecule(rel: &str, conv: Conventions) -> axtb::Molecule {
    let dir = root().join(rel);
    let input = json(&dir.join("input.json"));
    let charge = input["charge"].as_f64().unwrap();
    let uhf = input["uhf"].as_u64().unwrap() as usize;
    let xyz = std::fs::read_to_string(dir.join("mol.xyz")).unwrap();
    axtb_gxtb::v2::read_xyz(&xyz, charge, Some(uhf), conv).unwrap()
}

fn tight(c: Conventions) -> ScfOptions {
    ScfOptions {
        conventions: c,
        energy_tol: Some(1e-13),
        residual_tol: Some(1e-11),
        max_iter: 500,
        ..Default::default()
    }
}

fn energy(m: &axtb::Molecule, c: Conventions) -> f64 {
    Setup::new(m, c).unwrap().scf(&tight(c)).unwrap().energies.total
}

fn fd_check(rel: &str, c: Conventions) -> f64 {
    let mol = molecule(rel, c);
    let setup = Setup::new(&mol, c).unwrap();
    let r = setup.scf(&tight(c)).unwrap();
    let g = setup.gradient(&r).unwrap();
    let h = 5e-4;
    let mut worst: f64 = 0.0;
    for a in 0..mol.nat() {
        for x in 0..3 {
            let e = |d: f64| {
                let mut m = mol.clone();
                m.positions[a][x] += d;
                energy(&m, c)
            };
            let fd = (8.0 * (e(h) - e(-h)) - (e(2.0 * h) - e(-2.0 * h))) / (12.0 * h);
            worst = worst.max((fd - g[a][x]).abs());
        }
    }
    worst
}

#[test]
fn gradient_matches_finite_differences() {
    let cases = std::env::var("V2_GRAD")
        .unwrap_or("reference/water,reference/dioxygen_triplet,reference/mesyl_chloride,reference/tetrachlorocuprate,reference/nickel_tetracarbonyl".into());
    for rel in cases.split(',') {
        for c in [Conventions::Gxtb, Conventions::Axtb] {
            let d = fd_check(rel, c);
            println!("{rel:32} {c:?} max |analytic - FD| = {d:.2e}");
            assert!(d < 2e-9, "{rel} {c:?}: {d:e}");
        }
    }
}

fn program_gradient(rel: &str, file: &str) -> Vec<[f64; 3]> {
    let text = std::fs::read_to_string(root().join(rel).join(file)).unwrap();
    let lines: Vec<&str> = text.lines().collect();
    let mut nat = 0;
    for l in &lines[2..] {
        if l.split_whitespace().count() == 4 {
            nat += 1;
        } else {
            break;
        }
    }
    lines[2 + nat..2 + 2 * nat]
        .iter()
        .map(|l| {
            let v: Vec<f64> = l.split_whitespace().map(|x| x.replace(['D', 'd'], "E").parse().unwrap()).collect();
            [v[0], v[1], v[2]]
        })
        .collect()
}

/// Program-mode analytic gradients with the program's tight thresholds (`--acc 0.0001`) and
/// default thresholds against the program's analytic gradients, all reference molecules.
#[test]
fn gradient_matches_program() {
    let mut worst: f64 = 0.0;
    let mut dirs = Vec::new();
    for g in ["reference", "elements", "large"] {
        for e in std::fs::read_dir(root().join(g)).unwrap() {
            let p = e.unwrap().path();
            if p.join("gradient_tight").exists() {
                dirs.push(format!("{g}/{}", p.file_name().unwrap().to_string_lossy()));
            }
        }
    }
    dirs.sort();
    for rel in &dirs {
        let c = Conventions::Gxtb;
        let mol = molecule(rel, c);
        let setup = Setup::new(&mol, c).unwrap();
        let r = setup.scf(&ScfOptions { conventions: c, accuracy: 1e-4, ..Default::default() }).unwrap();
        let g = setup.gradient(&r).unwrap();
        let gp = program_gradient(rel, "gradient_tight");
        let gd = program_gradient(rel, "gradient");
        let d = g.iter().flatten().zip(gp.iter().flatten()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        // default thresholds against the program's default gradient
        let r1 = setup.scf(&ScfOptions { conventions: c, ..Default::default() }).unwrap();
        let g1 = setup.gradient(&r1).unwrap();
        let d1 = g1.iter().flatten().zip(gd.iter().flatten()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        let summ = json(&root().join(rel).join("summary.json"));
        let it = summ["tight"]["scf_iterations"].as_u64().unwrap_or(0);
        println!("{rel:32} tight {d:9.2e} (iter {} / {it})  default {d1:9.2e}", r.iterations);
        // CeCl3 (f1 doublet, seven near-degenerate f orbitals): the 45 iterations of the tight
        // SCF amplify round-off differences (same energy to 1e-12 Eh, same iterations): the
        // summation order of the density matrix alone moves this gradient by up to 6e-8 Eh/bohr
        // (2e-9 from the program summing over orbitals in ascending order, 6e-8 in descending
        // order, 4e-8 as one matrix product); everything else agrees to <= 6e-11 (RuO4, tight)
        let tol = if rel.ends_with("CeCl3") { 1e-7 } else { 1e-10 };
        assert!(d < tol, "{rel}: {d:e}");
        assert!(d1 < tol, "{rel}: default thresholds {d1:e}");
        if !rel.ends_with("CeCl3") {
            worst = worst.max(d).max(d1);
        }
        assert_eq!(r.iterations as u64, it, "{rel}: tight SCF iterations");
    }
    println!("worst {worst:e}");
}

/// Default conventions: the smooth switches of the cutoffs (CNs and repulsion 25 bohr, revD4
/// three-body 25 and two-body 50 bohr) keep energy and gradient consistent: water + water with
/// O-O in the 25-26 bohr switch window and a helium atom 50.5 bohr away.
#[test]
fn smooth_cutoffs_finite_differences() {
    let w = [[0.0, 0.0, 0.2253], [0.0, 1.4423, -0.9015], [0.0, -1.4423, -0.9015]];
    let mut pos: Vec<[f64; 3]> = w.to_vec();
    pos.extend(w.iter().map(|p| [p[0] + 25.4, p[1] + 0.3, p[2] - 0.2]));
    pos.push([-50.4, 0.5, 0.2]);
    let numbers = [8, 1, 1, 8, 1, 1, 2];
    let mol = axtb::Molecule::new(&numbers, &pos, 0.0, None).unwrap();
    let c = Conventions::Axtb;
    let setup = Setup::new(&mol, c).unwrap();
    let r = setup.scf(&tight(c)).unwrap();
    let g = setup.gradient(&r).unwrap();
    let h = 1e-3;
    let mut worst: f64 = 0.0;
    for a in [0, 3, 4, 6] {
        for x in 0..3 {
            let e = |d: f64| {
                let mut m = mol.clone();
                m.positions[a][x] += d;
                energy(&m, c)
            };
            let fd = (8.0 * (e(h) - e(-h)) - (e(2.0 * h) - e(-2.0 * h))) / (12.0 * h);
            worst = worst.max((fd - g[a][x]).abs());
        }
    }
    println!("smooth cutoffs: max |analytic - FD| = {worst:.2e}");
    assert!(worst < 1e-9, "{worst:e}");
}
