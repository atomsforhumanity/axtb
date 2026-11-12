//! The CLI's g-xTB v1.1 results (energies and gradients) equal the Rust API's
//! (axtb_gxtb::Calculator) bit for bit, XYZ input reproduces the g-xTB 1.1.0 program, and
//! GFN2-only settings are rejected.

#![cfg(feature = "gxtb")]

use std::io::Write;
use std::path::Path;
use std::process::{Command, Stdio};

fn reference_dir(name: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../gxtb/tests/reference").join(name)
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

#[test]
fn cli_energies_equal_rust_api() {
    for (name, charge, uhf) in [("water", 0.0, 0usize), ("tetrachlorocuprate", -2.0, 1)] {
        let dir = reference_dir(name);
        let mol =
            axtb_gxtb::read_xyz(&std::fs::read_to_string(dir.join("mol.xyz")).unwrap(), charge, Some(uhf)).unwrap();
        let mut calc = axtb_gxtb::Calculator::new(
            axtb_gxtb::Version::V1_1,
            &mol.numbers,
            &mol.positions,
            charge,
            Some(uhf),
            Default::default(),
        )
        .unwrap();
        let rust = calc.singlepoint(&mol.positions, true).unwrap();
        let req = serde_json::json!({"numbers": mol.numbers, "positions": mol.positions, "charge": charge, "uhf": uhf,
                                     "method": "gxtb-v1.1", "gradient": true});
        // program conventions through the CLI equal the Rust API's too
        let opts = axtb_gxtb::ScfOptions { conventions: axtb_gxtb::Conventions::Gxtb, ..Default::default() };
        let mut pcalc =
            axtb_gxtb::Calculator::new(axtb_gxtb::Version::V1_1, &mol.numbers, &mol.positions, charge, Some(uhf), opts)
                .unwrap();
        let prust = pcalc.singlepoint(&mol.positions, true).unwrap();
        let mut preq = req.clone();
        preq["conventions"] = "gxtb".into();
        let (ok, out, err) = cli_json(&preq);
        assert!(ok, "{err}");
        let r: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(r["energy"].as_f64().unwrap(), prust.singlepoint.energy, "{name}");
        let grad: Vec<[f64; 3]> = serde_json::from_value(r["gradient"].clone()).unwrap();
        assert_eq!(&grad, prust.singlepoint.gradient.as_ref().unwrap(), "{name}");
        let (ok, out, err) = cli_json(&req);
        assert!(ok, "{err}");
        let r: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(r["energy"].as_f64().unwrap(), rust.singlepoint.energy, "{name}");
        let grad: Vec<[f64; 3]> = serde_json::from_value(r["gradient"].clone()).unwrap();
        assert_eq!(&grad, rust.singlepoint.gradient.as_ref().unwrap(), "{name}");
        assert_eq!(r["iterations"].as_u64().unwrap() as usize, rust.singlepoint.iterations);
        for (k, v) in rust.energies.named() {
            assert_eq!(r["terms"][k].as_f64().unwrap(), v, "{name}: {k}");
        }
        // XYZ file: the program's own Angstrom-to-bohr factor and conventions, so the
        // program's energy
        let out = Command::new(env!("CARGO_BIN_EXE_axtb"))
            .args([
                dir.join("mol.xyz").to_str().unwrap(),
                "--method",
                "gxtb-v1.1",
                "--conventions",
                "gxtb",
                "--chrg",
                &format!("{charge}"),
                "--uhf",
                &format!("{uhf}"),
            ])
            .output()
            .unwrap();
        assert!(out.status.success());
        let r: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        let summary: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("summary.json")).unwrap()).unwrap();
        let eprog = summary["total_energy"].as_f64().unwrap();
        assert!((r["energy"].as_f64().unwrap() - eprog).abs() < 1e-11, "{name}: {} vs program {eprog}", r["energy"]);
    }
}

#[test]
fn cli_rejects_gfn2_settings() {
    let req = serde_json::json!({"numbers": [8, 1, 1], "positions": [[0.0, 0.0, 0.22], [0.0, 1.43, -0.89], [0.0, -1.43, -0.89]],
                                 "method": "gxtb-v1.1", "alpb": "water", "etemp": 300.0});
    let (ok, _, err) = cli_json(&req);
    // implicit solvent alone points to the composite (solvation_correction)
    assert!(!ok && err.contains("no implicit-solvent parameters") && err.contains("solvation-correction"), "{err}");
    let req = serde_json::json!({"numbers": [8, 1, 1], "positions": [[0.0, 0.0, 0.22], [0.0, 1.43, -0.89], [0.0, -1.43, -0.89]],
                                 "method": "gxtb-v1.1", "etemp": 300.0, "dispersion": false});
    let (ok, _, err) = cli_json(&req);
    // etemp (Fermi smearing) is a g-xTB setting too; switching off dispersion is not
    assert!(!ok && err.contains("does not support") && err.contains("dispersion") && !err.contains("etemp"), "{err}");
    let req = serde_json::json!({"numbers": [8, 1, 1], "positions": [[0.0, 0.0, 0.22], [0.0, 1.43, -0.89], [0.0, -1.43, -0.89]],
                                 "method": "gxtb-v1.1", "conventions": "xtb"});
    let (ok, _, err) = cli_json(&req);
    assert!(!ok && err.contains("unknown g-xTB conventions"), "{err}");
}

fn reference_v2_dir(rel: &str) -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../gxtb/tests/reference_v2").join(rel)
}

/// g-xTB 2.0.1: the CLI equals the Rust API bit for bit in both conventions, and XYZ input with
/// `--conventions gxtb` reproduces the 2.0.1 program.
#[test]
fn cli_v2_equals_rust_api_and_program() {
    use axtb_gxtb::{Calculator, Conventions, ScfOptions, Version};
    for (rel, charge, uhf) in
        [("reference/water", 0.0, 0usize), ("reference/tetrachlorocuprate", -2.0, 1), ("elements/GdCl3", 0.0, 7)]
    {
        let dir = reference_v2_dir(rel);
        let text = std::fs::read_to_string(dir.join("mol.xyz")).unwrap();
        for conv in ["gxtb", "axtb"] {
            let c: Conventions = conv.parse().unwrap();
            let mol = axtb_gxtb::v2::read_xyz(&text, charge, Some(uhf), c).unwrap();
            let opts = ScfOptions { conventions: c, ..Default::default() };
            let mut calc =
                Calculator::new(Version::V2_0_1, &mol.numbers, &mol.positions, charge, Some(uhf), opts).unwrap();
            let rust = calc.singlepoint(&mol.positions, true).unwrap();
            let req = serde_json::json!({"numbers": mol.numbers, "positions": mol.positions, "charge": charge, "uhf": uhf,
                                         "method": "gxtb-v2.0.1", "gradient": true, "conventions": conv});
            let (ok, out, err) = cli_json(&req);
            assert!(ok, "{err}");
            let r: serde_json::Value = serde_json::from_str(&out).unwrap();
            assert_eq!(r["energy"].as_f64().unwrap(), rust.singlepoint.energy, "{rel} {conv}");
            let grad: Vec<[f64; 3]> = serde_json::from_value(r["gradient"].clone()).unwrap();
            assert_eq!(&grad, rust.singlepoint.gradient.as_ref().unwrap(), "{rel} {conv}");
            assert_eq!(r["iterations"].as_u64().unwrap() as usize, rust.singlepoint.iterations);
            for (k, v) in rust.energies.named() {
                assert_eq!(r["terms"][k].as_f64().unwrap(), v, "{rel}: {k}");
            }
            // XYZ file through the CLI
            let out = Command::new(env!("CARGO_BIN_EXE_axtb"))
                .args([
                    dir.join("mol.xyz").to_str().unwrap(),
                    "--method",
                    "gxtb-v2.0.1",
                    "--conventions",
                    conv,
                    "--chrg",
                    &format!("{charge}"),
                    "--uhf",
                    &format!("{uhf}"),
                ])
                .output()
                .unwrap();
            assert!(out.status.success());
            let r: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
            assert_eq!(r["energy"].as_f64().unwrap(), rust.singlepoint.energy, "{rel} {conv} (xyz)");
            if conv == "gxtb" {
                let printout = std::fs::read_to_string(dir.join("xtb.out")).unwrap();
                let line = printout.lines().find(|l| l.contains(":: total energy")).unwrap();
                let eprog: f64 = line.split_whitespace().nth(3).unwrap().parse().unwrap();
                assert!(
                    (r["energy"].as_f64().unwrap() - eprog).abs() < 1e-11,
                    "{rel}: {} vs program {eprog}",
                    r["energy"]
                );
            }
        }
    }
    let req = serde_json::json!({"numbers": [8, 1, 1], "positions": [[0.0, 0.0, 0.22], [0.0, 1.43, -0.89], [0.0, -1.43, -0.89]],
                                 "method": "gxtb-v2.0.1", "gbsa": "water", "dispersion": false});
    let (ok, _, err) = cli_json(&req);
    assert!(
        !ok && err.contains("gxtb-v2.0.1 has no implicit-solvent parameters") && err.contains("solvation-correction"),
        "{err}"
    );
    let req = serde_json::json!({"numbers": [8, 1, 1], "positions": [[0.0, 0.0, 0.22], [0.0, 1.43, -0.89], [0.0, -1.43, -0.89]],
                                 "method": "gxtb-v2.0.1", "dispersion": false});
    let (ok, _, err) = cli_json(&req);
    assert!(!ok && err.contains("gxtb-v2.0.1 does not support") && err.contains("dispersion"), "{err}");
    // response and analytic Hessians (2.0.1 only; analytic is the default) through the CLI
    // equal the Rust API
    let water = [[0.0, 0.0, 0.22], [0.0, 1.43, -0.89], [0.0, -1.43, -0.89]];
    for (method, name) in [(Some("response"), "response"), (None, "analytic")] {
        let mut req =
            serde_json::json!({"numbers": [8, 1, 1], "positions": water, "method": "gxtb-v2.0.1", "hessian": true});
        if let Some(m) = method {
            req["hessian_method"] = serde_json::json!(m);
        }
        let (ok, out, err) = cli_json(&req);
        assert!(ok, "{err}");
        let r: serde_json::Value = serde_json::from_str(&out).unwrap();
        assert_eq!(r["hessian_method"], name);
        let mut calc = Calculator::new(Version::V2_0_1, &[8, 1, 1], &water, 0.0, None, ScfOptions::default()).unwrap();
        let hm = if name == "response" { axtb::HessianMethod::Response } else { axtb::HessianMethod::Analytic };
        let h = calc.hessian(&water, &axtb::HessianOptions { method: hm, step: None, accuracy: None }).unwrap();
        let cli_h: Vec<f64> = r["hessian"]
            .as_array()
            .unwrap()
            .iter()
            .flat_map(|row| row.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()))
            .collect();
        assert_eq!(cli_h, h.hessian);
    }
    let req = serde_json::json!({"numbers": [8, 1, 1], "positions": water, "method": "gxtb-v1.1", "hessian": true, "hessian_method": "response"});
    let (ok, _, err) = cli_json(&req);
    assert!(!ok && err.contains("gxtb-v1.1 Hessians: the Response method is not available"), "{err}");
}

#[test]
fn cli_molden_equals_rust_api() {
    // JSON `"molden": true` returns the file of the Rust API's orbitals, `molden_file` writes
    // it and `molden_cartesian` selects Cartesian functions ([CuCl4]2-: open shell, d shells)
    let cu = [[0.0, 0.0, 0.0], [4.27, 0.0, 0.0], [-4.27, 0.0, 0.0], [0.0, 4.27, 0.0], [0.0, -4.27, 0.0]];
    let numbers = [29, 17, 17, 17, 17];
    let dir = std::env::temp_dir().join(format!("axtb-cli-molden-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    for method in ["gfn2", "gxtb-v1.1", "gxtb-v2.0.1"] {
        for cartesian in [false, true] {
            let file = dir.join(format!("{method}-{cartesian}.molden"));
            let req = serde_json::json!({"numbers": numbers, "positions": cu, "charge": -2.0, "uhf": 1, "method": method,
                                         "molden": true, "molden_file": file, "molden_cartesian": cartesian});
            let (ok, out, err) = cli_json(&req);
            assert!(ok, "{err}");
            let r: serde_json::Value = serde_json::from_str(&out).unwrap();
            let orbitals = if method == "gfn2" {
                let mut opts = axtb::Options::default();
                opts.scf.keep_orbitals = true;
                let mol = axtb::Molecule::new(&numbers, &cu, -2.0, Some(1)).unwrap();
                axtb::singlepoint(axtb::Method::Gfn2, &mol, &opts, false).unwrap().orbitals.unwrap()
            } else {
                let version = axtb_gxtb::Version::from_method(method).unwrap();
                let opts = axtb_gxtb::ScfOptions { keep_orbitals: true, ..Default::default() };
                let mut calc = axtb_gxtb::Calculator::new(version, &numbers, &cu, -2.0, Some(1), opts).unwrap();
                calc.singlepoint(&cu, false).unwrap().singlepoint.orbitals.unwrap()
            };
            let functions = if cartesian { axtb::MoldenFunctions::Cartesian } else { axtb::MoldenFunctions::Spherical };
            let title = format!("axtb {} {method}", env!("CARGO_PKG_VERSION"));
            let text = orbitals.to_molden(functions, &title).unwrap();
            assert_eq!(r["molden"].as_str().unwrap(), text, "{method}");
            assert_eq!(r["molden_file"].as_str().unwrap(), file.to_str().unwrap());
            assert_eq!(std::fs::read_to_string(&file).unwrap(), text, "{method}");
            assert_eq!(text.contains("[5D]"), !cartesian);
            assert_eq!(text.contains("[6D]"), cartesian);
        }
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// Fermi smearing at 1000 K through the CLI (`--etemp 1000`, as the composite's g-xTB retry ladder runs
/// the program): XYZ input with the program's conventions reproduces the g-xTB 2.0.1 and 1.1.0
/// programs' energies (tests/reference_v2/etemp, tests/reference/etemp.json) and the 2.0.1
/// gradient; the result equals the Rust API's bit for bit.
#[test]
fn cli_etemp_reproduces_programs() {
    use axtb_gxtb::{Calculator, Conventions, ScfOptions, Version};
    let rows = |path: &str| -> Vec<serde_json::Value> {
        let doc: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(Path::new(env!("CARGO_MANIFEST_DIR")).join(path)).unwrap())
                .unwrap();
        doc["rows"].as_array().unwrap().clone()
    };
    let v2 = rows("../gxtb/tests/reference_v2/etemp/etemp.json");
    let v1 = rows("../gxtb/tests/reference/etemp.json");
    for (rel, name, charge, uhf) in
        [("reference/water", "water", 0.0, 0usize), ("reference/tetrachlorocuprate", "tetrachlorocuprate", -2.0, 1)]
    {
        // 2.0.1
        let dir = reference_v2_dir(rel);
        let r2 = v2.iter().find(|r| r["name"] == name && r["etemp"] == 1000.0).unwrap();
        let out = Command::new(env!("CARGO_BIN_EXE_axtb"))
            .args([dir.join("mol.xyz").to_str().unwrap(), "--method", "gxtb-v2.0.1", "--conventions", "gxtb"])
            .args(["--chrg", &format!("{charge}"), "--uhf", &format!("{uhf}"), "--etemp", "1000", "--grad"])
            .output()
            .unwrap();
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        let r: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        let e = r["energy"].as_f64().unwrap();
        assert!((e - r2["total_energy"].as_f64().unwrap()).abs() < 1e-11, "{name}: {e} vs {}", r2["total_energy"]);
        assert_eq!(r["iterations"], r2["scf_iterations"], "{name}");
        let g: Vec<[f64; 3]> = serde_json::from_value(r["gradient"].clone()).unwrap();
        let gp: Vec<[f64; 3]> = serde_json::from_value(r2["gradient"].clone()).unwrap();
        for (a, b) in g.iter().zip(&gp) {
            for k in 0..3 {
                assert!((a[k] - b[k]).abs() < 1e-9, "{name}: gradient");
            }
        }
        let text = std::fs::read_to_string(dir.join("mol.xyz")).unwrap();
        let mol = axtb_gxtb::v2::read_xyz(&text, charge, Some(uhf), Conventions::Gxtb).unwrap();
        let opts = ScfOptions { conventions: Conventions::Gxtb, etemp: 1000.0, ..Default::default() };
        let mut calc = Calculator::new(Version::V2_0_1, &mol.numbers, &mol.positions, charge, Some(uhf), opts).unwrap();
        assert_eq!(calc.singlepoint(&mol.positions, false).unwrap().singlepoint.energy, e, "{name}");
        // 1.1.0
        let r1 = v1.iter().find(|r| r["name"] == name && r["etemp"] == 1000.0).unwrap();
        let out = Command::new(env!("CARGO_BIN_EXE_axtb"))
            .args([
                reference_dir(name).join("mol.xyz").to_str().unwrap(),
                "--method",
                "gxtb-v1.1",
                "--conventions",
                "gxtb",
            ])
            .args(["--chrg", &format!("{charge}"), "--uhf", &format!("{uhf}"), "--etemp", "1000"])
            .output()
            .unwrap();
        assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
        let r: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        let e = r["energy"].as_f64().unwrap();
        assert!((e - r1["total_energy"].as_f64().unwrap()).abs() < 1e-11, "{name} v1.1: {e} vs {}", r1["total_energy"]);
        assert_eq!(r["iterations"], r1["scf_iterations"], "{name} v1.1");
    }
}

#[test]
fn cli_broken_symmetry() {
    let path = Path::new(env!("CARGO_TARGET_TMPDIR")).join("h2-3A.xyz");
    std::fs::write(&path, "2\n\nH 0 0 0\nH 0 0 3.0\n").unwrap();
    let run = |args: &[&str]| {
        let out = Command::new(env!("CARGO_BIN_EXE_axtb")).arg(&path).args(args).output().unwrap();
        (out.status.code(), String::from_utf8(out.stdout).unwrap(), String::from_utf8(out.stderr).unwrap())
    };
    let (code, stdout, _) = run(&["--method", "gxtb-v2.0.1", "--broken-symmetry", "--grad"]);
    assert_eq!(code, Some(0));
    let r: serde_json::Value = serde_json::from_str(&stdout).unwrap();
    let b = &r["broken_symmetry"];
    assert_eq!((b["broken"].as_bool(), b["outcome"].as_str()), (Some(true), Some("broken")), "{b}");
    assert_eq!(r["energy"], b["unrestricted_energy"]);
    assert!(b["restricted_energy"].as_f64().unwrap() - r["energy"].as_f64().unwrap() > 0.08);
    assert_eq!(r["occupations_alpha"].as_array().unwrap()[0], 1.0);
    let (code, _, stderr) = run(&["--broken-symmetry"]);
    assert!(code == Some(1) && stderr.contains("GFN2-xTB has no spin-dependent term"), "{stderr}");
    let (code, _, stderr) = run(&["--method", "gxtb-v2.0.1", "--broken-symmetry", "--uhf", "2"]);
    assert!(code == Some(1) && stderr.contains("singlets"), "{stderr}");
}
