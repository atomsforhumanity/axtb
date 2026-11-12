//! g-xTB 2.0.1 Hessians at a finite electronic temperature (Fermi smearing) against a tight
//! reference: Richardson extrapolation of central differences of analytic gradients (h = 2e-3
//! and 1e-3 bohr) with SCFs converged to 1e-12 Eh / 5e-13, in both conventions.
//!
//! `V2_ETEMP_HESS=reference/water:1000,...` selects molecule:temperature pairs.

use axtb_base::fd::symmetrize;
use axtb_gxtb::v2::hessian::AnalyticOptions;
use axtb_gxtb::v2::response::CpksOptions;
use axtb_gxtb::v2::scf::{ScfResult, Setup};
use axtb_gxtb::{Conventions, ScfOptions};
use serde_json::Value;
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/reference_v2")
}

fn molecule(rel: &str, conv: Conventions) -> axtb::Molecule {
    let dir = root().join(rel);
    let input: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("input.json")).unwrap()).unwrap();
    let uhf = input["uhf"].as_u64().unwrap() as usize;
    let xyz = std::fs::read_to_string(dir.join("mol.xyz")).unwrap();
    axtb_gxtb::v2::read_xyz(&xyz, input["charge"].as_f64().unwrap(), Some(uhf), conv).unwrap()
}

fn tight(c: Conventions, etemp: f64) -> ScfOptions {
    ScfOptions {
        conventions: c,
        etemp,
        energy_tol: Some(1e-12),
        residual_tol: Some(5e-13),
        max_iter: 1000,
        all_orbitals: true,
        ..Default::default()
    }
}

fn scf(m: &axtb::Molecule, opts: &ScfOptions) -> (Setup, ScfResult) {
    let s = Setup::new(m, opts.conventions).unwrap();
    let r = s.scf(opts).unwrap();
    (s, r)
}

/// Analytic gradient of the final density.
fn gradient(s: &Setup, r: &ScfResult) -> Vec<f64> {
    let mut w = s.energy_weighted_density(r);
    w.iter_mut().for_each(|x| *x = -*x);
    s.explicit_gradient(&r.state, Some(&w)).unwrap().concat()
}

fn numerical(mol: &axtb::Molecule, h: f64, opts: &ScfOptions) -> Vec<f64> {
    let n3 = 3 * mol.nat();
    let mut out = vec![0.0; n3 * n3];
    for y in 0..n3 {
        let g = |d: f64| {
            let mut m = mol.clone();
            m.positions[y / 3][y % 3] += d;
            let (s, r) = scf(&m, opts);
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

fn maxdiff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).fold(0.0, |m: f64, (x, y)| m.max((x - y).abs()))
}

/// Analytic and response Hessians at 1000 and 3000 K, and at 5000 K for stretched N2 and AgCl
/// (-TS = -4.4e-5 and -3.6e-4 Eh; without the occupation response the AgCl Hessian is off by
/// 9e-5 Eh/bohr^2), against the Richardson reference. The full check (water, caffeine, O2,
/// [CuCl4]2-, Ni(CO)4 at 1000 and 3000 K) runs with V2_ETEMP_HESS.
#[test]
fn etemp_hessians_match_reference() {
    let cases = std::env::var("V2_ETEMP_HESS").unwrap_or(
        "reference/water:1000,reference/dioxygen_triplet:3000,reference/tetrachlorocuprate:3000,etemp/extra/N2_stretched:5000,\
         elements/AgCl:5000"
            .into(),
    );
    for case in cases.split(',') {
        let (rel, t) = case.split_once(':').unwrap();
        let etemp: f64 = t.parse().unwrap();
        for c in [Conventions::Axtb, Conventions::Gxtb] {
            let mol = molecule(rel, c);
            let n3 = 3 * mol.nat();
            let opts = tight(c, etemp);
            let h1 = numerical(&mol, 2e-3, &opts);
            let h2 = numerical(&mol, 1e-3, &opts);
            let href: Vec<f64> = h1.iter().zip(&h2).map(|(a, b)| (4.0 * b - a) / 3.0).collect();
            let (mut setup, r) = scf(&mol, &opts);
            let ao = AnalyticOptions { cpks: CpksOptions { tol: 1e-10, max_iter: 500 }, ..Default::default() };
            let (mut ha, _, _) = setup.analytic_hessian(&r, &ao).unwrap();
            let asa = symmetrize(&mut ha, n3);
            let (mut hr, _, _) = setup.response_hessian(&r, 2e-4, CpksOptions::default()).unwrap();
            symmetrize(&mut hr, n3);
            let hmax = href.iter().fold(0.0f64, |m, x| m.max(x.abs()));
            let (da, dr) = (maxdiff(&ha, &href), maxdiff(&hr, &href));
            println!(
                "{rel:32} {etemp:5} K {c:?} -TS {:9.2e} max|H| {hmax:.3} analytic {da:.1e} (asym {asa:.1e}) response {dr:.1e}",
                r.energies.entropy
            );
            assert!(da < 1e-7 && asa < 1e-9, "{rel} {etemp} K {c:?}: analytic {da:e}, asymmetry {asa:e}");
            assert!(dr < 1e-6, "{rel} {etemp} K {c:?}: response {dr:e}");
        }
    }
}
