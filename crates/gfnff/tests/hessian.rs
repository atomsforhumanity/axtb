// SPDX-License-Identifier: LGPL-3.0-or-later
//! Analytic GFN-FF Hessian against central differences of the analytic gradient
//! (Richardson-extrapolated), dipole derivatives against finite differences of the EEQ
//! dipole, and thread-count independence; the force fields of xtb 6.8.0 and 6.7.1.

use axtb_gfnff::method::{HessianMethod, HessianOptions, Method, Want};
use axtb_gfnff::{Calculator, Conventions, Options};

const BOTH: [Conventions; 2] = [Conventions::Xtb680, Conventions::Xtb671];

/// Positions (bohr).
type Structure = Vec<[f64; 3]>;
use serde_json::Value;
use std::path::PathBuf;

fn load(name: &str) -> (Vec<usize>, Vec<[f64; 3]>, f64) {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/reference-xtb-6.7.1").join(format!("{name}.json"));
    let c: Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
    let numbers = c["numbers"].as_array().unwrap().iter().map(|z| z.as_u64().unwrap() as usize).collect();
    let mut seed: u64 = 7;
    let mut rnd = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    // displaced by up to 0.05 bohr: the built structures have exactly linear arrangements,
    // where bends and out-of-line HB terms are not differentiable
    let pos = c["positions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| {
            let a = p.as_array().unwrap();
            [
                a[0].as_f64().unwrap() + 0.05 * rnd(),
                a[1].as_f64().unwrap() + 0.05 * rnd(),
                a[2].as_f64().unwrap() + 0.05 * rnd(),
            ]
        })
        .collect();
    (numbers, pos, c["charge"].as_f64().unwrap())
}

#[test]
fn analytic_hessian_matches_finite_differences() {
    let names = [
        "struct-caffeine",
        "glycine-zwitterion-water",
        "dimer-pyridine-HF",
        "dimer-acetone-water",
        "xb-ICl-NH3",
        "struct-cu-nh3-4",
    ];
    for (name, conventions) in names.iter().flat_map(|n| BOTH.map(|c| (*n, c))) {
        let (numbers, pos, charge) = load(name);
        let mut calc = Calculator::new(&numbers, &pos, charge, Options { conventions, ..Default::default() }).unwrap();
        let h = calc.hessian(&pos, &HessianOptions::default()).unwrap();
        let s = 2e-4;
        let h1 = calc.hessian(&pos, &numerical(s)).unwrap().hessian;
        let h2 = calc.hessian(&pos, &numerical(s / 2.0)).unwrap().hessian;
        let n3 = 3 * numbers.len();
        let mut worst: f64 = 0.0;
        for i in 0..n3 {
            for j in 0..n3 {
                let fd = |m: &Vec<f64>, a: usize, b: usize| m[a * n3 + b];
                let r = (4.0 * fd(&h2, i, j) - fd(&h1, i, j)) / 3.0;
                let rt = (4.0 * fd(&h2, j, i) - fd(&h1, j, i)) / 3.0;
                worst = worst.max((h.hessian[i * n3 + j] - 0.5 * (r + rt)).abs());
            }
        }
        assert!(worst < 1e-8, "{name} ({conventions}): analytic vs finite-difference Hessian {worst:e}");
        assert!(h.max_asymmetry < 1e-10, "{name}: asymmetry {}", h.max_asymmetry);
        // dipole derivatives
        let mut dd: f64 = 0.0;
        let mut cd = |k: usize, s: f64| -> [f64; 3] {
            let mut xp = pos.clone();
            let mut xm = pos.clone();
            xp[k / 3][k % 3] += s;
            xm[k / 3][k % 3] -= s;
            let dp = calc.singlepoint(&xp, &Want::gradient(false)).unwrap().dipole;
            let dm = calc.singlepoint(&xm, &Want::gradient(false)).unwrap().dipole;
            [0, 1, 2].map(|e| (dp[e] - dm[e]) / (2.0 * s))
        };
        for k in 0..n3 {
            let (a, b) = (cd(k, 4e-4), cd(k, 2e-4));
            for e in 0..3 {
                dd = dd.max(((4.0 * b[e] - a[e]) / 3.0 - h.dipole_derivatives.as_ref().unwrap()[k][e]).abs());
            }
        }
        assert!(dd < 1e-7, "{name} ({conventions}): dipole derivatives {dd:e}");
    }
}

/// Exactly linear and planar structures (bends at 180°, torsions at 0°/180°, linear
/// A-H...B and A-X...B), where xtb's angle and out-of-line gradients are 0/0: the Hessian
/// still matches finite differences of the gradient (step 1e-3 bohr, Richardson; smaller
/// steps see the literal gradient's lost digits next to these points).
#[test]
fn hessian_at_linear_and_planar_geometries() {
    const AA: f64 = 1.0 / 0.529177210544;
    let mols: Vec<(&str, Vec<usize>, Structure)> = vec![
        ("CO2", vec![6, 8, 8], vec![[0.0, 0.0, 0.0], [0.0, 0.0, 1.16], [0.0, 0.0, -1.16]]),
        ("HCN", vec![1, 6, 7], vec![[0.0, 0.0, -1.065], [0.0, 0.0, 0.0], [0.0, 0.0, 1.155]]),
        ("acetylene", vec![1, 6, 6, 1], vec![[0.0, 0.0, -1.665], [0.0, 0.0, -0.6], [0.0, 0.0, 0.6], [0.0, 0.0, 1.665]]),
        (
            "ethylene",
            vec![6, 6, 1, 1, 1, 1],
            vec![
                [0.0, 0.0, 0.667],
                [0.0, 0.0, -0.667],
                [0.0, 0.923, 1.238],
                [0.0, -0.923, 1.238],
                [0.0, 0.923, -1.238],
                [0.0, -0.923, -1.238],
            ],
        ),
        (
            "benzene",
            vec![6, 6, 6, 6, 6, 6, 1, 1, 1, 1, 1, 1],
            (0..12)
                .map(|i| {
                    let a = std::f64::consts::PI / 3.0 * (i % 6) as f64;
                    let r = if i < 6 { 1.39 } else { 2.47 };
                    [r * a.cos(), r * a.sin(), 0.0]
                })
                .collect(),
        ),
    ];
    let mut cases: Vec<(String, Vec<usize>, Structure, f64)> = mols
        .into_iter()
        .map(|(n, z, x)| (n.to_string(), z, x.iter().map(|p| p.map(|v| v * AA)).collect(), 0.0))
        .collect();
    for name in ["dimer-water", "dimer-formicacid", "xb-ICl-NH3", "xb-CF3Cl-formaldehyde", "struct-cu-nh3-4"] {
        let p =
            PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/reference-xtb-6.7.1").join(format!("{name}.json"));
        let c: Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
        let z = c["numbers"].as_array().unwrap().iter().map(|z| z.as_u64().unwrap() as usize).collect();
        let x = serde_json::from_value(c["positions"].clone()).unwrap();
        cases.push((name.to_string(), z, x, c["charge"].as_f64().unwrap()));
    }
    for ((name, numbers, pos, charge), conventions) in cases.iter().flat_map(|c| BOTH.map(|v| (c, v))) {
        let mut calc = Calculator::new(numbers, pos, *charge, Options { conventions, ..Default::default() }).unwrap();
        let h = calc.hessian(pos, &HessianOptions::default()).unwrap();
        let s = 1e-3;
        let h1 = calc.hessian(pos, &numerical(s)).unwrap().hessian;
        let h2 = calc.hessian(pos, &numerical(s / 2.0)).unwrap().hessian;
        let worst = (0..h1.len()).map(|k| (h.hessian[k] - (4.0 * h2[k] - h1[k]) / 3.0).abs()).fold(0.0, f64::max);
        assert!(worst < 1e-8, "{name} ({conventions}): analytic vs finite-difference Hessian {worst:e}");
    }
}

#[test]
fn hessian_independent_of_threads() {
    let (numbers, pos, charge) = load("glycine-zwitterion-water");
    let h1 = Calculator::new(&numbers, &pos, charge, Options::default())
        .unwrap()
        .hessian(&pos, &HessianOptions::default())
        .unwrap();
    let h4 = Calculator::new(&numbers, &pos, charge, Options { threads: 4, ..Default::default() })
        .unwrap()
        .hessian(&pos, &HessianOptions::default())
        .unwrap();
    let d = h1.hessian.iter().zip(&h4.hessian).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    assert!(d < 1e-13, "threads change the Hessian by {d:e}");
    let e1 = Calculator::new(&numbers, &pos, charge, Options::default())
        .unwrap()
        .singlepoint(&pos, &Want::gradient(true))
        .unwrap();
    let e4 = Calculator::new(&numbers, &pos, charge, Options { threads: 4, ..Default::default() })
        .unwrap()
        .singlepoint(&pos, &Want::gradient(true))
        .unwrap();
    assert!((e1.energy - e4.energy).abs() < 1e-13);
}

/// (H2O)200: bit for bit the same on a given thread count (fresh calculators), within
/// round-off between thread counts (the parallel LU of the EEQ systems splits by thread count).
#[test]
fn results_repeat_on_a_thread_count() {
    let xyz = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/structures/large/water200.xyz"),
    )
    .unwrap();
    let (numbers, pos): (Vec<usize>, Vec<[f64; 3]>) = xyz
        .lines()
        .skip(2)
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let f: Vec<&str> = l.split_whitespace().collect();
            let p: Vec<f64> = f[1..4].iter().map(|v| v.parse::<f64>().unwrap() / 0.529177210544).collect();
            (if f[0] == "O" { 8 } else { 1 }, [p[0], p[1], p[2]])
        })
        .unzip();
    let run = |threads: usize| {
        let mut calc = Calculator::new(&numbers, &pos, 0.0, Options { threads, ..Default::default() }).unwrap();
        let r = calc.singlepoint(&pos, &Want::gradient(true)).unwrap();
        let mut bits: Vec<u64> = vec![r.energy.to_bits()];
        bits.extend(r.gradient.unwrap().iter().flatten().map(|g| g.to_bits()));
        bits.extend(calc.topology().qa.iter().map(|q| q.to_bits()));
        bits
    };
    let (a, b, one) = (run(3), run(3), run(1));
    assert_eq!(a, b, "two runs on 3 threads differ");
    let d = a.iter().zip(&one).map(|(x, y)| (f64::from_bits(*x) - f64::from_bits(*y)).abs()).fold(0.0, f64::max);
    assert!(d < 1e-11, "1 vs 3 threads: {d:e}");
}

/// Central differences of the analytic gradient with step `step` (bohr).
fn numerical(step: f64) -> HessianOptions {
    HessianOptions { method: Some(HessianMethod::Numerical), step: Some(step), accuracy: None }
}
