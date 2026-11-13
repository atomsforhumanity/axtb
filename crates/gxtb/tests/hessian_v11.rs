//! g-xTB v1.1 Hessians in the default conventions (step 17: v1.1 on the engine) against a tight
//! reference: Richardson extrapolation of central differences of analytic gradients (h = 2e-3
//! and 1e-3 bohr) with SCFs converged to 1e-13; the response Hessian's dipole derivatives against
//! central differences of the dipoles; 1 vs 4 threads bit for bit.
//!
//! `V11_HESS=water,...` selects molecules (tests/reference).

use axtb_base::fd::symmetrize;
use axtb_gxtb::v2::response::CpksOptions;
use axtb_gxtb::v2::scf::{ScfResult, Setup};
use axtb_gxtb::{Conventions, ScfOptions};
use serde_json::Value;
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/reference")
}

fn json(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn molecule(name: &str) -> axtb::Molecule {
    let dir = root().join(name);
    let input = json(&dir.join("input.json"));
    let charge = input["charge"].as_f64().unwrap();
    let uhf = input["uhf"].as_u64().unwrap() as usize;
    let xyz = std::fs::read_to_string(dir.join("mol.xyz")).unwrap();
    axtb_gxtb::read_xyz(&xyz, charge, Some(uhf)).unwrap()
}

fn model() -> axtb_gxtb::v2::Model {
    axtb_gxtb::v2::Model::v1_1(Conventions::Axtb)
}

fn tight(etemp: f64) -> ScfOptions {
    ScfOptions {
        energy_tol: Some(1e-13),
        density_tol: Some(1e-11),
        max_iter: 1000,
        all_orbitals: true,
        etemp,
        ..Default::default()
    }
}

fn scf(m: &axtb::Molecule, opts: &ScfOptions) -> (Setup, ScfResult) {
    let s = Setup::new(m, model()).unwrap();
    let r = s.scf(opts).unwrap();
    (s, r)
}

/// Analytic gradient and dipole of the final density.
fn gradient_dipole(s: &Setup, r: &ScfResult) -> (Vec<f64>, [f64; 3]) {
    let g = s.gradient(r).unwrap().concat();
    let mut mu = [0.0; 3];
    for (a, p) in s.mol.positions.iter().enumerate() {
        for k in 0..3 {
            mu[k] += r.state.qa[a] * p[k] + r.state.dip[a][k];
        }
    }
    (g, mu)
}

/// Central differences of analytic gradients and dipoles with step `h`.
fn numerical(mol: &axtb::Molecule, h: f64, opts: &ScfOptions) -> (Vec<f64>, Vec<[f64; 3]>) {
    let n3 = 3 * mol.nat();
    let mut out = vec![0.0; n3 * n3];
    let mut dmu = vec![[0.0; 3]; n3];
    for y in 0..n3 {
        let g = |d: f64| {
            let mut m = mol.clone();
            m.positions[y / 3][y % 3] += d;
            let (s, r) = scf(&m, opts);
            gradient_dipole(&s, &r)
        };
        let ((gp, mp), (gm, mm)) = (g(h), g(-h));
        for x in 0..n3 {
            out[x * n3 + y] = (gp[x] - gm[x]) / (2.0 * h);
        }
        dmu[y] = std::array::from_fn(|k| (mp[k] - mm[k]) / (2.0 * h));
    }
    symmetrize(&mut out, n3);
    (out, dmu)
}

fn reference(mol: &axtb::Molecule, etemp: f64) -> (Vec<f64>, Vec<[f64; 3]>) {
    let (h1, d1) = numerical(mol, 2e-3, &tight(etemp));
    let (h2, d2) = numerical(mol, 1e-3, &tight(etemp));
    let h = h1.iter().zip(&h2).map(|(a, b)| (4.0 * b - a) / 3.0).collect();
    let d = d1.iter().zip(&d2).map(|(a, b)| std::array::from_fn(|k| (4.0 * b[k] - a[k]) / 3.0)).collect();
    (h, d)
}

fn maxdiff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).fold(0.0, |m: f64, (x, y)| m.max((x - y).abs()))
}

#[test]
fn v11_response_hessians_match_reference() {
    let cases =
        std::env::var("V11_HESS").unwrap_or("water,ammonium,dioxygen_triplet,iodomethane,nickel_tetracarbonyl".into());
    for name in cases.split(',') {
        for etemp in [0.0, 3000.0] {
            let mol = molecule(name);
            let n3 = 3 * mol.nat();
            let (href, dref) = reference(&mol, etemp);
            let (mut setup, r) = scf(&mol, &tight(etemp));
            let (mut hr, dmu, _) = setup.response_hessian(&r, 2e-4, CpksOptions::default()).unwrap();
            let asym = symmetrize(&mut hr, n3);
            let hmax = href.iter().fold(0.0f64, |m, x| m.max(x.abs()));
            let dr = maxdiff(&hr, &href);
            let dd = maxdiff(dmu.as_flattened(), dref.as_flattened());
            println!("{name:24} {etemp:6.0} K max|H| {hmax:.3} response {dr:.1e} (asym {asym:.1e}) dipole derivatives {dd:.1e}");
            assert!(dr < 1e-6, "{name} {etemp} K: {dr:e}");
            assert!(dd < 1e-6, "{name} {etemp} K: dipole derivatives {dd:e}");
        }
    }
}

/// The calculator offers v1.1's response Hessian in the default conventions only (the program's
/// Fock matrix is not the derivative of its energy), and the analytic one in neither; on 1 and 4
/// threads it gives the same bits.
#[test]
fn v11_response_hessian_through_the_calculator() {
    use axtb::method::Method;
    use axtb::{HessianMethod, HessianOptions};
    use axtb_gxtb::{Calculator, Options, Version};
    let mol = molecule("water");
    let calc = |conventions: Conventions, threads: usize| {
        let o = Options { conventions, threads, ..Default::default() };
        Calculator::new(Version::V1_1, &mol.numbers, &mol.positions, mol.charge, mol.uhf, o).unwrap()
    };
    let response = HessianOptions { method: Some(HessianMethod::Response), step: None, accuracy: None };
    let analytic = HessianOptions { method: Some(HessianMethod::Analytic), step: None, accuracy: None };
    for c in [Conventions::Axtb, Conventions::Gxtb] {
        assert!(calc(c, 1).hessian(&mol.positions, &analytic).is_err(), "{c:?}: analytic");
    }
    assert!(calc(Conventions::Gxtb, 1).hessian(&mol.positions, &response).is_err());
    let h1 = calc(Conventions::Axtb, 1).hessian(&mol.positions, &response).unwrap();
    let h4 = calc(Conventions::Axtb, 4).hessian(&mol.positions, &response).unwrap();
    assert_eq!(h1.method, HessianMethod::Response);
    assert_eq!(h1.hessian, h4.hessian, "1 vs 4 threads");
    assert_eq!(h1.dipole_derivatives, h4.dipole_derivatives, "1 vs 4 threads");
    // the calculator's reference SCF (accuracy 1e-5) against the tight reference of this file
    let (href, _) = reference(&mol, 0.0);
    assert!(maxdiff(&h1.hessian, &href) < 1e-6, "{:e}", maxdiff(&h1.hessian, &href));
}
