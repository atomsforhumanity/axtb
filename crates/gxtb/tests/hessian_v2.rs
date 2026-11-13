// SPDX-License-Identifier: LGPL-3.0-or-later
//! g-xTB 2.0.1 Hessians against a tight reference: Richardson extrapolation of central
//! differences of analytic gradients (h = 2e-3 and 1e-3 bohr) with SCFs converged to 1e-13.
//!
//! `V2_HESS=reference/water,...` selects molecules.

use axtb_base::fd::symmetrize;
use axtb_gxtb::hessian::AnalyticOptions;
use axtb_gxtb::response::CpksOptions;
use axtb_gxtb::scf::{ScfResult, Setup};
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
    axtb_gxtb::model::read_xyz(&xyz, charge, Some(uhf), conv).unwrap()
}

fn tight(c: Conventions) -> ScfOptions {
    ScfOptions {
        conventions: c,
        energy_tol: Some(1e-12),
        residual_tol: Some(1e-13),
        max_iter: 1000,
        all_orbitals: true,
        ..Default::default()
    }
}

fn scf(m: &axtb::Molecule, c: Conventions, opts: &ScfOptions) -> (Setup, ScfResult) {
    let s = Setup::new(m, axtb_gxtb::model::Model::v2_0_1(c)).unwrap();
    let r = s.scf(opts).unwrap();
    (s, r)
}

/// Analytic gradient of the final density (the energy derivative in both conventions).
fn gradient(s: &Setup, r: &ScfResult) -> Vec<f64> {
    let mut w = s.energy_weighted_density(r);
    w.iter_mut().for_each(|x| *x = -*x);
    s.explicit_gradient(&r.state, Some(&w)).unwrap().concat()
}

fn numerical(mol: &axtb::Molecule, c: Conventions, h: f64, opts: &ScfOptions) -> Vec<f64> {
    let n3 = 3 * mol.nat();
    let mut out = vec![0.0; n3 * n3];
    for y in 0..n3 {
        let g = |d: f64| {
            let mut m = mol.clone();
            m.positions[y / 3][y % 3] += d;
            let (s, r) = scf(&m, c, opts);
            gradient(&s, &r)
        };
        let (gp, gm) = (g(h), g(-h));
        for x in 0..n3 {
            out[x * n3 + y] = (gp[x] - gm[x]) / (2.0 * h);
        }
    }
    symmetrize(&mut out, n3);
    out
}

fn reference(mol: &axtb::Molecule, c: Conventions) -> Vec<f64> {
    let h1 = numerical(mol, c, 2e-3, &tight(c));
    let h2 = numerical(mol, c, 1e-3, &tight(c));
    h1.iter().zip(&h2).map(|(a, b)| (4.0 * b - a) / 3.0).collect()
}

fn maxdiff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).fold(0.0, |m: f64, (x, y)| m.max((x - y).abs()))
}

#[test]
fn hessians_match_reference() {
    let cases = std::env::var("V2_HESS").unwrap_or("reference/water,reference/dioxygen_triplet".into());
    for rel in cases.split(',') {
        for c in [Conventions::Axtb, Conventions::Gxtb] {
            let mol = molecule(rel, c);
            let n3 = 3 * mol.nat();
            let href = reference(&mol, c);
            let (mut setup, r) = scf(&mol, c, &tight(c));
            let t = std::time::Instant::now();
            let step = std::env::var("V2_STEP").map(|v| v.parse().unwrap()).unwrap_or(2e-4);
            let (mut hr, _, _) = setup.response_hessian(&r, step, CpksOptions::default()).unwrap();
            let tr = t.elapsed().as_secs_f64();
            let asym = symmetrize(&mut hr, n3);
            let hmax = href.iter().fold(0.0f64, |m, x| m.max(x.abs()));
            let dr = maxdiff(&hr, &href);
            // the default numerical Hessian (step 0.005, SCF accuracy 0.01)
            let nopts = ScfOptions { conventions: c, accuracy: 0.01, ..Default::default() };
            let hn = numerical(&mol, c, 5e-3, &nopts);
            let dn = maxdiff(&hn, &href);
            let t = std::time::Instant::now();
            let tol = std::env::var("V2_TOL").map(|v| v.parse().unwrap()).unwrap_or(1e-10);
            let ao = AnalyticOptions { cpks: CpksOptions { tol, max_iter: 500 }, ..Default::default() };
            let (mut ha, _, _) = setup.analytic_hessian(&r, &ao).unwrap();
            let ta = t.elapsed().as_secs_f64();
            let asa = symmetrize(&mut ha, n3);
            let da = maxdiff(&ha, &href);
            println!(
                "{rel:28} {c:?} max|H| {hmax:.3} analytic {da:.1e} (asym {asa:.1e}, {ta:.2} s) response {dr:.1e} (asym {asym:.1e}, {tr:.2} s) numerical {dn:.1e}"
            );
            assert!(dr < 1e-6, "{rel} {c:?}: {dr:e}");
            assert!(da < 1e-7 && asa < 1e-9, "{rel} {c:?}: analytic {da:e}, asymmetry {asa:e}");
        }
    }
}
