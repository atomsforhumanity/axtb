//! The analytic g-xTB 2.0.1 Hessian's symmetric response terms
//! ([`AnalyticOptions::symmetric`]): at the default solver tolerance (1e-6) the Hessian equals
//! that of a tight solve (1e-12) to well below the analytic Hessian's accuracy, while the same
//! tolerance without the symmetric form is visibly off. Closed and open shells, 0 K and a
//! finite electronic temperature, both conventions.

use axtb_gxtb::v2::hessian::{symmetrize, AnalyticOptions};
use axtb_gxtb::v2::response::CpksOptions;
use axtb_gxtb::v2::scf::Setup;
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

fn maxdiff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

#[test]
fn symmetric_response_terms_match_a_tight_solve() {
    for (rel, conv, etemp) in [
        ("reference/caffeine", Conventions::Axtb, 0.0),
        ("reference/caffeine", Conventions::Gxtb, 0.0),
        ("reference/cu-nh3-4", Conventions::Axtb, 0.0),
        ("reference/tetrachlorocuprate", Conventions::Axtb, 3000.0),
    ] {
        let mol = molecule(rel, conv);
        let mut setup = Setup::new(&mol, conv).unwrap();
        let opts = ScfOptions {
            conventions: conv,
            etemp,
            energy_tol: Some(1e-12),
            residual_tol: Some(1e-11),
            max_iter: 500,
            keep_orbitals: true,
            ..Default::default()
        };
        let r = setup.scf(&opts).unwrap();
        let n3 = 3 * mol.nat();
        // symmetrized, as every calculator returns it: the correction removes the symmetric part
        // of the response error, symmetrizing the antisymmetric part
        let hessian = |setup: &mut Setup, ao: &AnalyticOptions| {
            let mut h = setup.analytic_hessian(&r, ao).unwrap().0;
            symmetrize(&mut h, n3);
            h
        };
        let tol = |tol: f64, symmetric: bool| AnalyticOptions {
            cpks: CpksOptions { tol, max_iter: 500 },
            symmetric,
            ..Default::default()
        };
        let tight = hessian(&mut setup, &tol(1e-12, false));
        let default = hessian(&mut setup, &AnalyticOptions::default());
        let plain = hessian(&mut setup, &tol(1e-6, false));
        let (d_sym, d_plain) = (maxdiff(&default, &tight), maxdiff(&plain, &tight));
        println!("{rel:30} {conv:?} {etemp:6} K  default (1e-6, symmetric) {d_sym:.1e}  1e-6 plain {d_plain:.1e}");
        assert!(d_sym < 1e-9, "{rel} {conv:?} {etemp} K: symmetric form at the default tolerance {d_sym:e}");
        assert!(d_plain > 10.0 * d_sym, "{rel} {conv:?} {etemp} K: plain {d_plain:e}, symmetric {d_sym:e}");
    }
}
