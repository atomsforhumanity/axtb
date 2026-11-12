//! The driver's rules: one request for every method, settings that do not apply refused, the
//! same response fields, the error kinds, and a stateful calculator that gives the one-shot
//! results.

use axtb_driver::{run, Calculator, Error, Method, Request, Task};

fn water() -> Request {
    Request::new(vec![8, 1, 1], vec![[0.0, 0.0, 0.22], [0.0, 1.43, -0.89], [0.0, -1.43, -0.89]])
}

fn with(method: &str) -> Request {
    Request { method: Some(method.into()), ..water() }
}

fn json(req: &str) -> Result<Request, serde_json::Error> {
    serde_json::from_str(req)
}

#[test]
fn unknown_fields_are_refused() {
    let base = r#""numbers": [1, 1], "positions": [[0, 0, 0], [0, 0, 1.4]]"#;
    assert!(json(&format!("{{{base}}}")).is_ok());
    let err = json(&format!("{{{base}, \"gradiant\": true}}")).unwrap_err().to_string();
    assert!(err.contains("unknown field `gradiant`"), "{err}");
    assert!(json(r#"{"numbers": [1]}"#).is_err(), "positions are required");
}

#[test]
fn settings_that_do_not_apply_are_errors() {
    let input = |req: Request| match run(&req) {
        Err(e @ Error::Input(_)) => e.to_string(),
        other => panic!("{:?}", other.map(|r| r.energy)),
    };
    for req in [
        Request { threads: Some(0), ..water() },
        Request { uhf: Some(-1), ..water() },
        Request { max_iter: Some(0), ..water() },
        Request { fragment_charges: Some(vec![0.0]), ..water() },
        Request { solvation_state: Some("gsolv".into()), ..water() },
        Request { solvation_correction: Some("gfn2".into()), alpb: Some("water".into()), ..water() },
        Request { accuracy: Some(1.0), ..with("gfnff") },
        Request { bond_orders: true, ..with("gfnff") },
        Request { pressure: Some(1e5), ..water() },
        Request { hessian: true, hessian_step: Some(0.01), ..water() },
        Request { hessian: true, hessian_method: Some("response".into()), ..with("gfnff") },
        Request { molden_cartesian: true, ..water() },
        Request { analyze_hessian: Some(vec![vec![0.0; 9]; 9]), conventions: Some("xtb".into()), ..water() },
    ] {
        input(req);
    }
    let msg = input(Request { alpb: Some("water".into()), ..with("g-xTB 2.0") });
    assert!(msg.contains("no implicit-solvent parameters") && msg.contains("solvation_correction"), "{msg}");
}

#[test]
fn named_starts_are_checked_per_method() {
    let start = |req: Request| run(&req).map(|r| r.scf.expect("an SCF method").guess);
    assert_eq!(start(Request { guess: Some("sad".into()), ..with("gfn2") }).unwrap(), "sad");
    assert_eq!(start(Request { guess: Some("eeq".into()), ..with("gfn2") }).unwrap(), "eeq");
    assert_eq!(start(Request { guess: Some("fresh".into()), ..with("gxtb-v2.0.1") }).unwrap(), "eeq-bc");
    for req in [
        Request { guess: Some("eeq-bc".into()), ..with("gfn2") },
        Request { guess: Some("sad".into()), ..with("gxtb-v2.0.1") },
        Request { guess: Some("eeq".into()), ..with("gxtb-v1.1") },
        Request {
            guess: Some("eeq-bc".into()),
            alpb: Some("water".into()),
            solvation_correction: Some("gfn2".into()),
            ..with("gxtb-v2.0.1")
        },
    ] {
        assert!(matches!(run(&req), Err(Error::Input(_))), "{:?}", req.guess);
    }
}

#[test]
fn both_gxtb_versions_report_the_entropy_term() {
    for method in ["gxtb-v1.1", "gxtb-v2.0.1"] {
        let r = run(&Request { etemp: Some(20000.0), ..with(method) }).unwrap();
        let terms = r.terms.expect("terms").0;
        let ts = terms.iter().find(|(n, _)| *n == "entropy").map(|t| t.1);
        assert!(ts.is_some_and(|ts| ts < 0.0), "{method}: {ts:?}");
    }
}

#[test]
fn unpaired_electrons_warn_for_gfnff() {
    assert!(axtb_driver::uhf_warning(Method::Gfnff, Some(1)).is_some());
    assert!(axtb_driver::uhf_warning(Method::Gfnff, Some(0)).is_none());
    assert!(axtb_driver::uhf_warning(Method::Gfn2, Some(1)).is_none());
}

#[test]
fn error_kinds() {
    let e = run(&Request { max_iter: Some(2), ..water() }).unwrap_err();
    assert!(matches!(e, Error::ScfNotConverged { .. }) && e.kind() == "scf_not_converged");
    let json = serde_json::to_value(&e).unwrap();
    assert!(json["iterations"].is_u64() && json["residual"].is_number(), "{json}");
    let e = run(&Request::new(vec![87], vec![[0.0; 3]])).unwrap_err();
    assert!(e.kind() == "unsupported_element" && e.is_input());
    assert_eq!(serde_json::to_value(&e).unwrap()["z"], 87);
    let e = run(&Request { method: Some("gfn1".into()), ..water() }).unwrap_err();
    assert_eq!(e.kind(), "input");
}

#[test]
fn every_response_has_the_header() {
    for (method, conventions, canonical) in [
        ("GFN2-xTB", "XTB 6.7.1", "xtb-6.7.1"),
        ("gfn-ff", "default", "axtb"),
        ("g-xTB 2.0.1", "gxtb", "gxtb"),
        ("gxtb_v1.1", "axtb", "axtb"),
    ] {
        let r = run(&Request { conventions: Some(conventions.into()), gradient: true, ..with(method) }).unwrap();
        let v = serde_json::to_value(&r).unwrap();
        assert_eq!(v["program"], "axtb");
        assert_eq!(v["version"], env!("CARGO_PKG_VERSION"));
        assert_eq!(v["method"], Method::parse(method).unwrap().name());
        assert_eq!(v["conventions"], canonical);
        assert_eq!(v.get("eigensolver").is_some(), method != "gfn-ff", "{method}");
        assert!(v["energy"].is_f64() && v["gradient"].is_array() && v["timings"].is_object(), "{method}");
    }
}

#[test]
fn calculator_gives_the_one_shot_results() {
    for method in ["gfn2", "gfnff", "gxtb-v2.0.1"] {
        let req = Request { gradient: true, hessian: true, ..with(method) };
        let one = run(&req).unwrap();
        let mut calc = Calculator::new(&with(method)).unwrap();
        let task = Task::from_request(&Request { gradient: true, hessian: true, ..Request::default() }, calc.method())
            .unwrap();
        let r = calc.compute(&req.positions, &task).unwrap();
        assert_eq!(one.energy, r.energy, "{method}");
        assert_eq!(one.gradient, r.gradient, "{method}");
        assert_eq!(one.hessian.as_ref().unwrap().hessian, r.hessian.as_ref().unwrap().hessian, "{method}");
        let v = r.vibrations.as_ref().unwrap();
        assert_eq!(v.frequencies.len(), 3, "{method}");
    }
}

#[test]
fn the_composite_reports_its_parts() {
    let req = Request {
        alpb: Some("water".into()),
        solvation_correction: Some("gfn2".into()),
        hessian: true,
        vibrations: Some(false),
        ..with("gxtb-v2.0.1")
    };
    let r = run(&req).unwrap();
    let c = r.composite.as_ref().unwrap();
    assert!(c.parts.gxtb.hessian_method.is_none());
    assert_eq!(c.hessian_parts.as_ref().unwrap().gfn2_gas.hessian_method, Some("analytic"));
    assert_eq!(r.terms.as_ref().unwrap().get("solvation_correction"), Some(c.components.solvation_correction));
    assert!(r.vibrations.is_none());
}

#[test]
fn analysis_of_a_given_hessian() {
    let h = run(&Request { hessian: true, vibrations: Some(false), ..water() }).unwrap().hessian.unwrap().hessian;
    let rows: Vec<Vec<f64>> = h.data.chunks(9).map(<[f64]>::to_vec).collect();
    let req = Request {
        analyze_hessian: Some(rows),
        analysis_conventions: Some("xtb".into()),
        energy: Some(-5.0),
        ..water()
    };
    let r = run(&req).unwrap();
    assert_eq!((r.method, r.conventions), (None, "xtb"));
    let t = r.vibrations.unwrap().thermo.unwrap();
    assert_eq!(t.last.free_energy, Some(-5.0 + t.last.g_rrho));
}
