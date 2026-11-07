//! g-xTB 2.0.1 analytic Hessian pieces against finite differences: the perturbation F^(x), S^x
//! of every displacement against 4-point differences of the setup at fixed density.
//! `V2_ANA=reference/water,...` selects molecules. Heavy elements (UF6: 4e-7) see tblite's
//! primitive screening step the finite-difference reference by ~1e-11 per crossing; with the
//! screening switched off they agree to 5e-12 like the light ones.

use axtb_gxtb::v2::analytic::Analytic;
use axtb_gxtb::v2::response::{from_mat, rhs_finite_difference};
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
    let charge = input["charge"].as_f64().unwrap();
    let uhf = input["uhf"].as_u64().unwrap() as usize;
    let xyz = std::fs::read_to_string(dir.join("mol.xyz")).unwrap();
    axtb_gxtb::v2::read_xyz(&xyz, charge, Some(uhf), conv).unwrap()
}

fn tight(c: Conventions) -> ScfOptions {
    ScfOptions {
        conventions: c,
        energy_tol: Some(1e-12),
        residual_tol: Some(1e-13),
        max_iter: 1000,
        keep_orbitals: true,
        ..Default::default()
    }
}

fn scf(m: &axtb::Molecule, c: Conventions) -> (Setup, ScfResult) {
    let s = Setup::new(m, c).unwrap();
    let r = s.scf(&tight(c)).unwrap();
    (s, r)
}

fn maxdiff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).fold(0.0, |m: f64, (x, y)| m.max((x - y).abs()))
}

fn maxabs(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |a: f64, b| a.max(b.abs()))
}

#[test]
fn fock_derivative_matches_finite_differences() {
    let cases = std::env::var("V2_ANA")
        .unwrap_or("reference/water,reference/dioxygen_triplet,reference/nickel_tetracarbonyl".into());
    for rel in cases.split(',') {
        for c in [Conventions::Axtb, Conventions::Gxtb] {
            let mol = molecule(rel, c);
            let (setup, r) = scf(&mol, c);
            let an = Analytic::new(&setup, &r.state).unwrap();
            let n3 = 3 * mol.nat();
            let (mut ef, mut sf, mut es, mut ss) = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
            for x in 0..n3 {
                let d = an.displacement(x).unwrap();
                let (fy, sy) = rhs_finite_difference(&setup, &r.density, x, 1e-3).unwrap();
                for k in 0..fy.len() {
                    let fd = from_mat(&fy[k]);
                    ef = ef.max(maxdiff(&fd, &d.f[k]));
                    sf = sf.max(maxabs(&fd));
                }
                let sd = from_mat(&sy);
                es = es.max(maxdiff(&sd, &d.s));
                ss = ss.max(maxabs(&sd));
            }
            println!("{rel:30} {c:?}: F^(x) {:.1e} (max {sf:.2e}), S^x {:.1e}", ef / sf, es / ss);
            assert!(ef < 1e-8 * sf && es < 1e-9 * ss, "{rel} {c:?}");
        }
    }
}
