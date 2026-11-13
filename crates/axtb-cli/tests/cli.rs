// SPDX-License-Identifier: LGPL-3.0-or-later
//! The command line itself: help and version, usage errors (exit code 2), JSON errors with
//! their kinds (exit code 1), the response's header, the one alias rule, settings that do not
//! apply, and XYZ files converted with the factor of the conventions.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

fn structure(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/structures").join(name)
}

fn axtb(args: &[&str], stdin: Option<&str>) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_axtb"))
        .args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let mut pipe = child.stdin.take().unwrap();
    if let Some(s) = stdin {
        // a usage error exits before reading stdin: a broken pipe here is not a failure
        let _ = pipe.write_all(s.as_bytes());
    }
    drop(pipe);
    child.wait_with_output().unwrap()
}

/// A JSON request's exit code and response (or `{"error": ...}`).
fn json(request: serde_json::Value) -> (i32, serde_json::Value) {
    let out = axtb(&["--json"], Some(&request.to_string()));
    (out.status.code().unwrap(), serde_json::from_slice(&out.stdout).unwrap())
}

fn water() -> serde_json::Value {
    serde_json::json!({"numbers": [8, 1, 1], "positions": [[0.0, 0.0, 0.22], [0.0, 1.43, -0.89], [0.0, -1.43, -0.89]]})
}

fn with(mut base: serde_json::Value, extra: serde_json::Value) -> serde_json::Value {
    for (k, v) in extra.as_object().unwrap() {
        base[k] = v.clone();
    }
    base
}

#[test]
fn help_version_and_usage_errors() {
    for flag in ["-h", "--help"] {
        let out = axtb(&[flag], None);
        let text = String::from_utf8(out.stdout).unwrap();
        assert!(out.status.success() && text.contains("--hess-method") && text.contains("--scf-trace"), "{text}");
    }
    let out = axtb(&["--version"], None);
    assert!(out.status.success());
    assert_eq!(String::from_utf8(out.stdout).unwrap().trim(), format!("axtb {}", env!("CARGO_PKG_VERSION")));
    let water = structure("water.xyz");
    let water = water.to_str().unwrap();
    for args in [
        vec![water, water],                                // a second structure
        vec![water, "--chrg", "abc"],                      // a bad number
        vec![water, "--chrg"],                             // a flag without its value
        vec![water, "--threads", "x"],                     // a bad integer
        vec![water, "--temp", "1,a"],                      // a bad list
        vec![water, "--alpb", "water", "--gbsa", "water"], // either solvent model
        vec!["no-such-file.xyz"],                          // an unreadable file
        vec![],                                            // no structure
        vec!["--json", "--grad"],                          // --json takes everything from the request
        vec!["--json", water],
    ] {
        let out = axtb(&args, Some("{}"));
        let err = String::from_utf8(out.stderr).unwrap();
        assert_eq!(out.status.code(), Some(2), "{args:?}: {err}");
        assert!(err.starts_with("error: ") && !err.contains("panicked"), "{args:?}: {err}");
    }
}

#[test]
fn response_header_and_names() {
    let (code, r) = json(with(water(), serde_json::json!({"method": "GFN2-xTB", "conventions": "XTB 6.7.1"})));
    assert_eq!(code, 0, "{r}");
    assert_eq!(r["program"], "axtb");
    assert_eq!(r["version"], env!("CARGO_PKG_VERSION"));
    assert_eq!((r["method"].as_str(), r["conventions"].as_str()), (Some("gfn2"), Some("xtb-6.7.1")));
    assert_eq!(r["eigensolver"], "faer");
    assert!(r.get("gradient").is_none() && r["timings"].is_object() && r["seconds"].is_f64());
    for key in ["shell_charges", "atomic_dipoles", "atomic_quadrupoles", "orbital_energies_alpha", "homo_lumo_gap"] {
        assert!(r.get(key).is_some(), "{key}");
    }
    let (code, r) = json(with(water(), serde_json::json!({"method": "gfn-ff", "conventions": "xtb_6.8.0"})));
    assert_eq!(code, 0, "{r}");
    assert_eq!((r["method"].as_str(), r["conventions"].as_str()), (Some("gfnff"), Some("xtb")));
    assert!(r.get("eigensolver").is_none() && r.get("iterations").is_none() && r["n_hbonds"].is_u64());
}

#[test]
fn json_errors_have_kinds() {
    let kind = |req: serde_json::Value| {
        let (code, r) = json(req);
        assert_eq!(code, 1, "{r}");
        (r["error"]["kind"].as_str().unwrap().to_string(), r["error"].clone())
    };
    // unknown fields, settings of another method or without their calculation, bad values
    for extra in [
        serde_json::json!({"gradiant": true}),
        serde_json::json!({"threads": 0}),
        serde_json::json!({"uhf": -1}),
        serde_json::json!({"fragment_charges": [0.0]}),
        serde_json::json!({"solvation_state": "gsolv"}),
        serde_json::json!({"temperature": 300.0}),
        serde_json::json!({"hessian": true, "hessian_method": "analytic", "hessian_step": 0.01}),
        serde_json::json!({"method": "gfnff", "bond_orders": true}),
        serde_json::json!({"method": "gfnff", "accuracy": 1.0}),
        serde_json::json!({"method": "gfn1"}),
        serde_json::json!({"molden_cartesian": true}),
    ] {
        let (k, e) = kind(with(water(), extra.clone()));
        assert_eq!(k, "input", "{extra}: {e}");
    }
    // an analysis of a given Hessian takes no calculation settings
    let h = vec![vec![0.0; 9]; 9];
    let (k, e) = kind(with(water(), serde_json::json!({"analyze_hessian": h, "conventions": "xtb"})));
    assert!(k == "input" && e["message"].as_str().unwrap().contains("analysis_conventions"), "{e}");
    let (k, e) = kind(with(water(), serde_json::json!({"max_iter": 2})));
    assert_eq!(k, "scf_not_converged");
    assert!(e["iterations"].is_u64() && e["energy_change"].is_number() && e["residual"].is_number(), "{e}");
    let (k, e) = kind(serde_json::json!({"numbers": [87], "positions": [[0.0, 0.0, 0.0]], "uhf": 1}));
    assert!(k == "unsupported_element" && e["z"] == 87, "{e}");
    let (k, _) = kind(serde_json::json!({"numbers": [8, 8], "positions": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.01]]}));
    assert_eq!(k, "input");
}

#[test]
fn file_errors_are_on_stderr() {
    let out = axtb(&[structure("water.xyz").to_str().unwrap(), "--method", "gfnff", "--wbo"], None);
    let err = String::from_utf8(out.stderr).unwrap();
    assert_eq!(out.status.code(), Some(1));
    assert!(out.stdout.is_empty() && err.starts_with("error: ") && err.contains("orbitals"), "{err}");
}

/// The conventions' Ångström-to-bohr factor: an XYZ file gives the same bits as a JSON request
/// with the positions converted by that factor.
#[test]
fn xyz_units_follow_the_conventions() {
    let path = structure("caffeine.xyz");
    let text = std::fs::read_to_string(&path).unwrap();
    for (conventions, factor) in [
        ("axtb", axtb::units::AATOAU),
        ("xtb", axtb::units::reference::MCTC_AATOAU),
        ("xtb-6.7.1", axtb::units::reference::MCTC_AATOAU),
        ("tblite", axtb::units::reference::MCTC_AATOAU),
    ] {
        let (numbers, positions) = axtb::io::read_xyz(&text, factor).unwrap();
        let out = axtb(&[path.to_str().unwrap(), "--conventions", conventions, "--grad"], None);
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        let file: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        let req = serde_json::json!({"numbers": numbers, "positions": positions, "conventions": conventions,
                                     "gradient": true});
        let (code, r) = json(req);
        assert_eq!(code, 0, "{r}");
        assert_eq!(file["energy"], r["energy"], "{conventions}");
        assert_eq!(file["gradient"], r["gradient"], "{conventions}");
    }
}
