//! Input errors and numerical errors of the public API: fragment charges, positions, threads,
//! settings, and the analytic Hessian where it is not defined.

use axtb_gfnff::method::{HessianMethod, HessianOptions, Method, TermKind, Want};
use axtb_gfnff::{Calculator, Error, Options};
use serde_json::Value;
use std::path::PathBuf;

const WATER: [[f64; 3]; 3] = [[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [-0.45, 1.75, 0.0]];

/// Two waters 9 bohr apart: two fragments.
fn waters() -> (Vec<usize>, Vec<[f64; 3]>) {
    let mut pos = WATER.to_vec();
    pos.extend(WATER.iter().map(|p| [p[0] + 9.0, p[1], p[2]]));
    (vec![8, 1, 1, 8, 1, 1], pos)
}

fn with_fragments(fc: &[f64]) -> Options {
    Options { fragment_charges: Some(fc.to_vec()), ..Default::default() }
}

fn input_error<T>(r: Result<T, Error>, what: &str) -> String {
    match r {
        Err(Error::Input(m)) => m,
        Err(e) => panic!("{what}: expected an input error, got {e}"),
        Ok(_) => panic!("{what}: expected an input error"),
    }
}

#[test]
fn fragment_charges_must_fit_the_fragments() {
    let numbers = [8, 1, 1];
    for fc in [&[1.0, -1.0][..], &[1.0], &[0.5, 0.5], &[], &[f64::NAN]] {
        let m = input_error(Calculator::new(&numbers, &WATER, 0.0, with_fragments(fc)), &format!("water {fc:?}"));
        assert!(m.contains("fragment"), "{m}");
    }
    // one fragment, its charge right
    let calc = Calculator::new(&numbers, &WATER, 0.0, with_fragments(&[0.0])).unwrap();
    assert_eq!(calc.topology().qfrag, vec![0.0]);
    // two fragments
    let (numbers, pos) = waters();
    let mut calc = Calculator::new(&numbers, &pos, 0.0, with_fragments(&[1.0, -1.0])).unwrap();
    assert_eq!(calc.topology().qfrag, vec![1.0, -1.0]);
    let q = calc.singlepoint(&pos, &Want::gradient(false)).unwrap().charges;
    assert!((q[..3].iter().sum::<f64>() - 1.0).abs() < 1e-10 && (q[3..].iter().sum::<f64>() + 1.0).abs() < 1e-10);
    for (fc, charge) in [(&[1.0][..], 1.0), (&[1.0, -1.0, 0.0], 0.0), (&[1.0, 1.0], 0.0)] {
        let m = input_error(Calculator::new(&numbers, &pos, charge, with_fragments(fc)), &format!("waters {fc:?}"));
        assert!(m.contains("fragment"), "{m}");
    }
    // the default: the total charge on the first fragment
    let calc = Calculator::new(&numbers, &pos, -1.0, Options::default()).unwrap();
    assert_eq!(calc.topology().qfrag, vec![-1.0, 0.0]);
}

#[test]
fn positions_are_checked() {
    let numbers = [8, 1, 1];
    let mut calc = Calculator::new(&numbers, &WATER, 0.0, Options::default()).unwrap();
    let mut bad = WATER;
    bad[1][2] = f64::NAN;
    let mut close = WATER;
    close[2] = [1.8, 0.0, 0.09];
    for (pos, what) in [(&bad, "not finite"), (&close, "closer than 0.1 bohr")] {
        let m = input_error(Calculator::new(&numbers, pos, 0.0, Options::default()), what);
        assert!(m.contains(what), "{m}");
        assert!(input_error(calc.singlepoint(pos, &Want::gradient(true)), what).contains(what));
        assert!(input_error(calc.hessian(pos, &HessianOptions::default()), what).contains(what));
        assert!(input_error(calc.hessian(pos, &numerical(0.005)), what).contains(what));
    }
    input_error(calc.singlepoint(&WATER[..2], &Want::gradient(true)), "two positions");
    let mut inf = WATER;
    inf[0][0] = f64::INFINITY;
    input_error(calc.singlepoint(&inf, &Want::gradient(false)), "infinite");
    // far apart is fine
    let far = [[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [-0.45, 1.75, 0.0], [1e9, 0.0, 0.0]];
    let mut calc = Calculator::new(&[8, 1, 1, 18], &far, 0.0, Options::default()).unwrap();
    assert!(calc.singlepoint(&far, &Want::gradient(true)).unwrap().energy.is_finite());
}

#[test]
fn settings_are_checked() {
    let numbers = [8, 1, 1];
    let m =
        input_error(Calculator::new(&numbers, &WATER, 0.0, Options { threads: 0, ..Default::default() }), "threads");
    assert!(m.contains("threads"), "{m}");
    for accuracy in [0.0, -1.0, f64::NAN, 100.0] {
        let opts = Options { cutoff_accuracy: Some(accuracy), ..Default::default() };
        input_error(Calculator::new(&numbers, &WATER, 0.0, opts), &format!("accuracy {accuracy}"));
    }
    for charge in [f64::NAN, f64::INFINITY, 11.0, -1e300] {
        input_error(Calculator::new(&numbers, &WATER, charge, Options::default()), &format!("charge {charge}"));
    }
    let (waters, pos) = waters();
    let opts = Options { fragment_charges: Some(vec![1e18, -1e18]), ..Default::default() };
    input_error(Calculator::new(&waters, &pos, 0.0, opts), "fragment charges 1e18");
    // the extremes are allowed
    Calculator::new(&numbers, &WATER, 10.0, Options::default()).unwrap();
    input_error(Calculator::new(&numbers, &[], 0.0, Options::default()), "no positions");
    input_error(Calculator::new(&[], &[], 0.0, Options::default()), "no atoms");
    assert!(matches!(
        Calculator::new(&[8, 1, 104], &WATER, 0.0, Options::default()),
        Err(Error::UnsupportedElement { z: 104, .. })
    ));
    let mut calc = Calculator::new(&numbers, &WATER, 0.0, Options::default()).unwrap();
    for step in [0.0, -0.005, f64::NAN, f64::INFINITY] {
        input_error(calc.hessian(&WATER, &numerical(step)), &format!("step {step}"));
    }
}

/// The acetone-water dimer of grimme-lab/xtb#1516, with the water H exactly on the C=O axis:
/// the analytic Hessian is undefined (a numerical error naming the hydrogen bond), the
/// numerical one is not.
#[test]
fn linear_carbonyl_hydrogen_bond() {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/reference-xtb-6.8.0/dimer-acetone-water.json");
    let c: Value = serde_json::from_str(&std::fs::read_to_string(p).unwrap()).unwrap();
    let numbers: Vec<usize> = serde_json::from_value(c["numbers"].clone()).unwrap();
    let pos: Vec<[f64; 3]> = serde_json::from_value(c["positions"].clone()).unwrap();
    let mut calc = Calculator::new(&numbers, &pos, 0.0, Options::default()).unwrap();
    match calc.hessian(&pos, &HessianOptions::default()) {
        Err(Error::Numerical(m)) => assert!(m.contains("xtb#1516") && m.contains("numerical Hessian"), "{m}"),
        Err(e) => panic!("expected a numerical error, got {e}"),
        Ok(_) => panic!("expected a numerical error"),
    }
    let h = calc.hessian(&pos, &numerical(0.005)).unwrap();
    assert!(h.hessian.iter().all(|v| v.is_finite()));
    // every term is a piece of the energy, and the pieces sum to it (to rounding)
    let r = calc.singlepoint(&pos, &Want::ENERGY).unwrap();
    assert!(r.terms.iter().all(|t| t.kind == TermKind::Piece) && r.terms.len() == 10);
    assert!((r.pieces() - r.energy).abs() < 1e-13, "{} vs {}", r.pieces(), r.energy);
    // no SCF: no wavefunction, no SCF settings, no orbitals
    assert!(calc.wavefunction().is_none() && calc.scf_settings().is_none());
    let orbitals = Want { orbitals: true, ..Want::ENERGY };
    assert!(input_error(calc.singlepoint(&pos, &orbitals), "orbitals").contains("no orbitals"));
}

/// Central differences of the analytic gradient with step `step` (bohr).
fn numerical(step: f64) -> HessianOptions {
    HessianOptions { method: Some(HessianMethod::Numerical), step: Some(step), accuracy: None }
}
