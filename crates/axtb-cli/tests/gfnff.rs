//! The CLI's GFN-FF results equal the Rust API's (axtb_gfnff::Calculator) bit for bit, XYZ
//! files with `--conventions xtb` reproduce `xtb mol.xyz --gfnff` (xtb 6.8.0; `xtb-6.7.1`:
//! 6.7.1), the Hessian in xtb's conventions reproduces `xtb --gfnff --hess` thermochemistry, a
//! reference structure fixes the topology, and SCF settings are rejected.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

fn structure(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/structures").join(name)
}

/// Run a JSON request: success, stdout and stderr (a failure: the JSON error's message).
fn cli_json(request: &serde_json::Value) -> (bool, String, String) {
    let mut child = Command::new(env!("CARGO_BIN_EXE_axtb"))
        .arg("--json")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(request.to_string().as_bytes()).unwrap();
    let out = child.wait_with_output().unwrap();
    let stdout = String::from_utf8(out.stdout).unwrap();
    // a failed request: exit code 1 and the error as JSON on stdout
    let err = if out.status.success() {
        String::from_utf8(out.stderr).unwrap()
    } else {
        assert_eq!(out.status.code(), Some(1), "{stdout}");
        let v: serde_json::Value = serde_json::from_str(&stdout).unwrap();
        v["error"]["message"].as_str().unwrap().to_string()
    };
    (out.status.success(), stdout, err)
}

fn cli_args(args: &[&str]) -> serde_json::Value {
    let out = Command::new(env!("CARGO_BIN_EXE_axtb")).args(args).output().unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    serde_json::from_slice(&out.stdout).unwrap()
}

fn read(name: &str) -> (Vec<usize>, Vec<[f64; 3]>) {
    axtb::io::read_xyz(&std::fs::read_to_string(structure(name)).unwrap(), axtb::units::AATOAU).unwrap()
}

#[test]
fn cli_equals_rust_api() {
    let (numbers, pos) = read("caffeine.xyz");
    let calc = axtb_gfnff::Calculator::new(&numbers, &pos, 0.0, Default::default()).unwrap();
    let rust = calc.singlepoint(&pos, true).unwrap();
    let req = serde_json::json!({"numbers": numbers, "positions": pos, "method": "gfnff", "gradient": true});
    let (ok, out, err) = cli_json(&req);
    assert!(ok, "{err}");
    let r: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert_eq!(r["energy"].as_f64().unwrap(), rust.energy);
    let grad: Vec<[f64; 3]> = serde_json::from_value(r["gradient"].clone()).unwrap();
    assert_eq!(&grad, rust.gradient.as_ref().unwrap());
    let q: Vec<f64> = serde_json::from_value(r["charges"].clone()).unwrap();
    assert_eq!(q, rust.charges);
    assert_eq!(r["terms"]["bond"].as_f64().unwrap(), rust.energies.bond);
    let bonds: Vec<[usize; 2]> = serde_json::from_value(r["topology"]["bonds"].clone()).unwrap();
    let expect: Vec<[usize; 2]> = calc.topology().bond_pairs().iter().map(|&(i, j)| [i, j]).collect();
    assert_eq!(bonds, expect);
    assert_eq!(r["topology"]["fragments"].as_array().unwrap().len(), numbers.len());
    // the analytic Hessian and its dipole derivatives
    let mut hreq = req.clone();
    hreq["hessian"] = true.into();
    hreq["vibrations"] = false.into();
    let (ok, out, err) = cli_json(&hreq);
    assert!(ok, "{err}");
    let r: serde_json::Value = serde_json::from_str(&out).unwrap();
    let h = calc.hessian(&pos).unwrap();
    let n3 = 3 * numbers.len();
    let rows: Vec<Vec<f64>> = serde_json::from_value(r["hessian"].clone()).unwrap();
    assert!(rows.iter().enumerate().all(|(i, row)| row[..] == h.hessian[i * n3..(i + 1) * n3]));
    let dd: Vec<[f64; 3]> = serde_json::from_value(r["dipole_derivatives"].clone()).unwrap();
    assert_eq!(dd, h.dipole_derivatives);
    assert_eq!(r["hessian_method"], "analytic");
}

#[test]
fn xyz_with_xtb_conventions_reproduces_xtb() {
    // `xtb taxol.xyz --gfnff --grad`: total energy of xtb 6.8.0 (4b8c987) and 6.7.1 (26dd68d)
    let path = structure("taxol.xyz");
    for (conventions, xtb) in [("xtb", -20.502152155531), ("xtb-6.7.1", -20.502152181536)] {
        let r = cli_args(&[path.to_str().unwrap(), "--method", "gfnff", "--conventions", conventions]);
        assert!((r["energy"].as_f64().unwrap() - xtb).abs() < 2e-12, "{conventions}: {}", r["energy"]);
    }
    // the default conventions read Angstrom with CODATA 2022: 6.8.0's force field, other bohr
    let r = cli_args(&[path.to_str().unwrap(), "--method", "gfnff"]);
    assert!((r["energy"].as_f64().unwrap() - -20.502152155531).abs() < 1e-8);
}

#[test]
fn hessian_thermo_in_xtb_conventions() {
    // `xtb caffeine.xyz --gfnff --hess` of xtb 6.8.0 (mass-weighted projection, no rotation) and
    // 6.7.1 (Cartesian projection of the slightly rotated molecule): total free energy, ZPVE,
    // G(RRHO); the thermochemistry scales GFN-FF frequencies by 1.03. Caffeine is not a GFN-FF
    // minimum, so the two projections differ by 9e-4 Eh.
    let path = structure("caffeine.xyz");
    for (conventions, g, zpve, grrho) in [
        ("xtb", -4.533411183087, 0.178808030298, 0.138385932403),
        ("xtb-6.7.1", -4.532486509084, 0.178674505263, 0.139310586115),
    ] {
        let r = cli_args(&[
            path.to_str().unwrap(),
            "--method",
            "gfnff",
            "--conventions",
            conventions,
            "--hess",
            "--hess-method",
            "numerical",
        ]);
        let t = &r["thermo"];
        assert_eq!(t["frequency_scale"].as_f64().unwrap(), 1.03);
        assert!((t["free_energy"].as_f64().unwrap() - g).abs() < 1e-10, "{conventions}: {}", t["free_energy"]);
        assert!((t["zpve"].as_f64().unwrap() - zpve).abs() < 1e-10, "{conventions}");
        assert!((t["g_rrho"].as_f64().unwrap() - grrho).abs() < 1e-10, "{conventions}");
        assert_eq!(r["vibrational_conventions"], conventions);
        // analytic: xtb's finite differences (step 0.005) are within 2e-6 Eh of it (6.8.0: 1.5e-6,
        // 6.7.1: 6e-7)
        let r = cli_args(&[path.to_str().unwrap(), "--method", "gfnff", "--conventions", conventions, "--hess"]);
        assert!((r["thermo"]["free_energy"].as_f64().unwrap() - g).abs() < 2e-6, "{conventions}");
    }
    // default conventions: CODATA 2022 analysis without the scale
    let r = cli_args(&[path.to_str().unwrap(), "--method", "gfnff", "--hess"]);
    assert_eq!(r["thermo"]["frequency_scale"].as_f64().unwrap(), 1.0);
    assert_eq!(r["vibrational_conventions"], "axtb");
}

#[test]
fn reference_topology_and_fragment_charges() {
    let (numbers, pos) = read("caffeine.xyz");
    let moved: Vec<[f64; 3]> =
        pos.iter().enumerate().map(|(i, p)| [p[0] + 0.01 * (i % 3) as f64, p[1], p[2] - 0.02]).collect();
    let calc = axtb_gfnff::Calculator::new(&numbers, &pos, 0.0, Default::default()).unwrap();
    let rust = calc.singlepoint(&moved, true).unwrap();
    let req = serde_json::json!({"numbers": numbers, "positions": moved, "topology_positions": pos, "method": "gfnff",
                                 "gradient": true});
    let (ok, out, err) = cli_json(&req);
    assert!(ok, "{err}");
    let r: serde_json::Value = serde_json::from_str(&out).unwrap();
    assert_eq!(r["energy"].as_f64().unwrap(), rust.energy);
    // two waters, charges +1 / -1 put on the fragments explicitly
    let w = [[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [-0.45, 1.75, 0.0]];
    let mut pos2: Vec<[f64; 3]> = w.to_vec();
    pos2.extend(w.iter().map(|p| [p[0] + 9.0, p[1], p[2]]));
    let req = serde_json::json!({"numbers": [8, 1, 1, 8, 1, 1], "positions": pos2, "method": "gfnff",
                                 "fragment_charges": [1.0, -1.0]});
    let (ok, out, err) = cli_json(&req);
    assert!(ok, "{err}");
    let r: serde_json::Value = serde_json::from_str(&out).unwrap();
    let qf: Vec<f64> = serde_json::from_value(r["topology"]["fragment_charges"].clone()).unwrap();
    assert_eq!(qf, vec![1.0, -1.0]);
    let q: Vec<f64> = serde_json::from_value(r["charges"].clone()).unwrap();
    assert!((q[..3].iter().sum::<f64>() - 1.0).abs() < 1e-10 && (q[3..].iter().sum::<f64>() + 1.0).abs() < 1e-10);
}

#[test]
fn scf_settings_are_rejected() {
    let base = serde_json::json!({"numbers": [8, 1, 1], "positions": [[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [-0.45, 1.75, 0.0]],
                                  "method": "gfnff"});
    for (k, v) in [
        ("etemp", serde_json::json!(300.0)),
        ("alpb", serde_json::json!("water")),
        ("mixer", serde_json::json!("diis")),
        ("accuracy", serde_json::json!(0.1)),
        ("molden", serde_json::json!(true)),
        ("conventions", serde_json::json!("tblite")),
    ] {
        let mut req = base.clone();
        req[k] = v;
        let (ok, _, err) = cli_json(&req);
        assert!(!ok && err.contains("gfnff"), "{k}: {err}");
    }
    let mut req = base.clone();
    req["hessian"] = true.into();
    req["hessian_method"] = "response".into();
    let (ok, _, err) = cli_json(&req);
    assert!(!ok && err.contains("analytic or numerical"), "{err}");
}
