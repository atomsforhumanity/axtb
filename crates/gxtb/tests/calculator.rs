//! The calculator interface used by the axtb front ends (CLI, Python, qccodec): same results
//! as the SCF driver, explicit threads; gradients and Hessians.

use axtb_gxtb::{read_xyz, Calculator, Conventions, ScfOptions, Setup, Version};
use serde_json::Value;
use std::path::Path;

fn reference(name: &str) -> (axtb::Molecule, f64) {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/reference").join(name);
    let input: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("input.json")).unwrap()).unwrap();
    let summary: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("summary.json")).unwrap()).unwrap();
    let mol = read_xyz(
        &std::fs::read_to_string(dir.join("mol.xyz")).unwrap(),
        input["charge"].as_f64().unwrap(),
        Some(input["uhf"].as_u64().unwrap() as usize),
    )
    .unwrap();
    (mol, summary["total_energy"].as_f64().unwrap())
}

fn program() -> ScfOptions {
    ScfOptions { conventions: Conventions::Gxtb, ..Default::default() }
}

#[test]
fn calculator_matches_scf_and_program() {
    for name in ["water", "dioxygen_triplet", "tetrachlorocuprate", "nickel_tetracarbonyl"] {
        let (mol, eref) = reference(name);
        let direct = Setup::new(&mol).unwrap().scf(&program()).unwrap();
        for threads in [1, 2] {
            let opts = ScfOptions { threads, ..program() };
            let mut calc =
                Calculator::new(Version::V1_1, &mol.numbers, &mol.positions, mol.charge, mol.uhf, opts).unwrap();
            let r = calc.singlepoint(&mol.positions, false).unwrap();
            let sp = &r.singlepoint;
            // the direct SCF runs sequentially: identical on 1 thread, to round-off on more
            // (the eigensolver's parallel tridiagonalization)
            if threads == 1 {
                assert_eq!(sp.energy, direct.energies.total, "{name}");
                assert_eq!(sp.qat, direct.atom_charges);
            } else {
                assert!((sp.energy - direct.energies.total).abs() < 1e-11, "{name}, {threads} threads");
                assert!(sp.qat.iter().zip(&direct.atom_charges).all(|(a, b)| (a - b).abs() < 1e-10));
            }
            assert!((sp.energy - eref).abs() < 1e-11, "{name}: {} vs program {eref}", sp.energy);
            assert_eq!(sp.iterations, direct.iterations);
            let t = &sp.terms;
            assert!((t.repulsion + t.dispersion + t.electronic - sp.energy).abs() < 1e-10);
            let named: f64 = r
                .energies
                .named()
                .iter()
                .filter(|(k, _)| {
                    ["electronic", "increments", "dispersion", "dispersion_three_body", "repulsion"].contains(k)
                })
                .map(|(_, v)| v)
                .sum();
            assert!((named - sp.energy).abs() < 1e-10);
            // the cached setup is reused for the same geometry, rebuilt for a new one
            assert_eq!(calc.singlepoint(&mol.positions, false).unwrap().singlepoint.energy, sp.energy);
            let mut moved = mol.positions.clone();
            moved[0][2] += 0.01;
            let e2 = calc.singlepoint(&moved, false).unwrap().singlepoint.energy;
            let mut fresh =
                Calculator::new(Version::V1_1, &mol.numbers, &moved, mol.charge, mol.uhf, program()).unwrap();
            assert_eq!(fresh.singlepoint(&moved, false).unwrap().singlepoint.energy, e2);
        }
        // the default diagonalizes the exact derivative: same energy expression, stationary
        // SCF, slightly lower energy than the program's fixed point
        let mut calc =
            Calculator::new(Version::V1_1, &mol.numbers, &mol.positions, mol.charge, mol.uhf, ScfOptions::default())
                .unwrap();
        let e = calc.singlepoint(&mol.positions, false).unwrap().singlepoint.energy;
        let exact = Setup::new(&mol)
            .unwrap()
            .scf(&ScfOptions { conventions: Conventions::Axtb, ..Default::default() })
            .unwrap();
        assert_eq!(e, exact.energies.total);
        assert!(e < eref && e > eref - 1e-4, "{name}: {e} vs program {eref}");
    }
}

#[test]
fn scf_dipole_matches_program_printout() {
    // the program prints the SCF dipole (au, 6 decimals) after the SCF
    let (mol, _) = reference("water");
    let mut calc =
        Calculator::new(Version::V1_1, &mol.numbers, &mol.positions, mol.charge, mol.uhf, program()).unwrap();
    let d = calc.singlepoint(&mol.positions, false).unwrap().singlepoint.dipole;
    for (x, p) in d.iter().zip([0.0, 0.0, -0.933216]) {
        assert!((x - p).abs() <= 5.0001e-7, "dipole {d:?}");
    }
}

#[test]
fn gradient_and_hessian() {
    let (mol, _) = reference("water");
    let st = Setup::new(&mol).unwrap();
    for (conventions, threads) in
        [(Conventions::Axtb, 1), (Conventions::Axtb, 2), (Conventions::Gxtb, 1), (Conventions::Gxtb, 2)]
    {
        let direct = st.gradient(&st.scf(&ScfOptions { conventions, ..Default::default() }).unwrap()).unwrap();
        let opts = ScfOptions { threads, conventions, ..Default::default() };
        let mut calc = Calculator::new(Version::V1_1, &mol.numbers, &mol.positions, mol.charge, mol.uhf, opts).unwrap();
        let sp = calc.singlepoint(&mol.positions, true).unwrap().singlepoint;
        let g = sp.gradient.as_ref().unwrap();
        if threads == 1 {
            assert_eq!(g, &direct, "{conventions:?}");
        } else {
            let d = g.as_flattened().iter().zip(direct.as_flattened()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            assert!(d < 1e-10, "{conventions:?}, {threads} threads: {d:e}");
        }
        assert!(sp.timings.iter().any(|(k, _)| *k == "gradient"));
        // semi-numerical Hessian: symmetric, translation-invariant rows, gradient at accuracy 0.01
        let h = calc.numerical_hessian(&mol.positions, 0.005, None).unwrap();
        let n3 = 3 * mol.nat();
        assert!(h.max_asymmetry < 2e-5, "{}", h.max_asymmetry);
        for i in 0..n3 {
            for k in 0..3 {
                let row: f64 = (0..mol.nat()).map(|a| h.hessian[i * n3 + 3 * a + k]).sum();
                assert!(row.abs() < 1e-5, "row {i} sum {row}");
            }
        }
        for (a, b) in h.gradient.iter().zip(&direct) {
            for k in 0..3 {
                assert!((a[k] - b[k]).abs() < 1e-6);
            }
        }
        // O-H stretch block: d2E/dz_H^2 from second differences of energies is 0.1887 Eh/bohr^2
        assert!((h.hessian[8 * n3 + 8] - 0.188710).abs() < 2e-5, "{}", h.hessian[8 * n3 + 8]);
    }
    let mut calc =
        Calculator::new(Version::V1_1, &mol.numbers, &mol.positions, mol.charge, mol.uhf, ScfOptions::default())
            .unwrap();
    assert!(calc.numerical_hessian(&mol.positions, 0.0, None).is_err());
    assert!(calc.numerical_hessian(&mol.positions, f64::NAN, None).is_err());
    assert!(calc.numerical_hessian(&mol.positions, 0.005, Some(f64::NAN)).is_err());
}

#[test]
fn input_errors() {
    use axtb::Error;
    let (mol, _) = reference("water");
    let (z, x) = (&mol.numbers, &mol.positions);
    let new = |v: Version, charge: f64, o: ScfOptions| Calculator::new(v, z, x, charge, None, o);
    let refused = |v: Version, o: ScfOptions| matches!(new(v, 0.0, o), Err(Error::Input(_)));
    let d = ScfOptions::default;
    for v in [Version::V2_0_1, Version::V1_1] {
        assert!(refused(v, ScfOptions { threads: 0, ..d() }));
        assert!(refused(v, ScfOptions { accuracy: f64::NAN, ..d() }));
        assert!(refused(v, ScfOptions { etemp: f64::INFINITY, ..d() }));
        assert!(refused(v, ScfOptions { etemp: -1.0, ..d() }));
        assert!(refused(v, ScfOptions { energy_tol: Some(f64::NAN), ..d() }));
        assert!(refused(v, ScfOptions { diis_size: Some(0), ..d() }));
        for damping in [0.0, -0.1, 1.5, f64::NAN] {
            assert!(refused(v, ScfOptions { damping: Some(damping), ..d() }), "{v:?} damping {damping}");
        }
        assert!(new(v, 0.0, ScfOptions { damping: Some(1.0), diis_size: Some(1), ..d() }).is_ok());
        // a fractional or non-finite charge
        for charge in [0.5, f64::NAN, f64::INFINITY] {
            assert!(matches!(new(v, charge, d()), Err(Error::Input(_))), "{v:?} charge {charge}");
        }
        // positions that are not finite, coincident atoms: an error, no panic
        let mut calc = new(v, 0.0, d()).unwrap();
        let mut bad = x.clone();
        bad[1][0] = f64::NAN;
        assert!(calc.singlepoint(&bad, true).is_err());
        let mut bad = x.clone();
        bad[1] = bad[0];
        assert!(calc.singlepoint(&bad, true).is_err());
        assert!(calc.singlepoint(x, true).is_ok());
        // the thread count is the pool's
        calc.options.threads = 2;
        assert!(calc.singlepoint(x, false).is_err());
    }
    // settings of the other version
    assert!(refused(Version::V2_0_1, ScfOptions { density_tol: Some(1e-6), ..d() }));
    assert!(refused(Version::V2_0_1, ScfOptions { transition_iteration: Some(3), ..d() }));
    assert!(refused(Version::V1_1, ScfOptions { residual_tol: Some(1e-6), ..d() }));
    assert!(new(Version::V2_0_1, 0.0, ScfOptions { residual_tol: Some(1e-6), ..d() }).is_ok());
    let v1 = ScfOptions { density_tol: Some(1e-6), transition_iteration: Some(2), ..d() };
    assert!(new(Version::V1_1, 0.0, v1).is_ok());
    // element without v1.1 parameters (Z = 60 is absent from the parameter file)
    assert!(Calculator::new(Version::V1_1, &[60, 1], &[[0.0; 3], [0.0, 0.0, 3.0]], 0.0, None, d()).is_err());
    // a 2.0.1 setup refuses SCF options of other conventions; its Hessians need every orbital
    let mut setup = axtb_gxtb::v2::scf::Setup::new(&mol, Conventions::Axtb).unwrap();
    assert!(setup.scf(&ScfOptions { conventions: Conventions::Gxtb, ..d() }).is_err());
    let r = setup.scf(&d()).unwrap();
    assert!(!r.all_orbitals);
    assert!(setup.analytic_hessian(&r, &Default::default()).is_err());
}

/// The mixer settings apply to both versions; an SCF that does not converge reports the last
/// energy change and residual.
#[test]
fn mixer_settings_and_failures() {
    let (mol, _) = reference("water");
    let (z, x) = (&mol.numbers, &mol.positions);
    for v in [Version::V2_0_1, Version::V1_1] {
        let energy = |o: ScfOptions| {
            let mut c = Calculator::new(v, z, x, 0.0, None, o).unwrap();
            c.singlepoint(x, false).map(|r| (r.singlepoint.energy, r.singlepoint.iterations))
        };
        let (e0, it0) = energy(ScfOptions::default()).unwrap();
        let (e1, it1) = energy(ScfOptions { damping: Some(0.5), diis_size: Some(3), ..Default::default() }).unwrap();
        // the same solution, along another path
        assert!((e0 - e1).abs() < 1e-7 && (e0 != e1 || it0 != it1), "{v:?}: {e0} ({it0}) vs {e1} ({it1})");
        match energy(ScfOptions { max_iter: 2, ..Default::default() }) {
            Err(axtb::Error::ScfNotConverged { iterations: 2, energy_change, residual }) => {
                assert!(energy_change.is_finite() && residual.is_finite(), "{v:?}")
            }
            other => panic!("{v:?}: {other:?}"),
        }
    }
}

#[test]
fn version_and_conventions_names() {
    for m in ["gxtb-v2.0.1", "GXTB_V2.0.1", "gxtb-2.0.1", "g-xTB 2.0.1", "g-xTB 2.0", "gxtbv201"] {
        assert_eq!(Version::from_method(m), Some(Version::V2_0_1), "{m}");
    }
    for m in ["gxtb-v1.1", "GXTB_V1.1", "gxtbv11", "g-xTB v1.1", "gxtb-1.1"] {
        assert_eq!(Version::from_method(m), Some(Version::V1_1), "{m}");
    }
    for m in ["gfn2", "gxtb", "gxtb-v2", "gxtb-2.1"] {
        assert_eq!(Version::from_method(m), None, "{m}");
    }
    assert_eq!(Version::V2_0_1.method(), "gxtb-v2.0.1");
    assert_eq!(Version::V1_1.method(), "gxtb-v1.1");
    assert_eq!(Version::default(), Version::V2_0_1);
    for (name, c) in [("axtb", Conventions::Axtb), ("Default", Conventions::Axtb), ("GXTB", Conventions::Gxtb)] {
        assert_eq!(name.parse::<Conventions>().unwrap(), c, "{name}");
    }
    assert!("gxtb_program".parse::<Conventions>().is_ok() && "xtb".parse::<Conventions>().is_err());
}

/// g-xTB 2.0.1 through the calculator: the same results as the setup/SCF/gradient API, and a
/// numerical Hessian that is symmetric, translation invariant and matches finite differences
/// of the gradient.
#[test]
fn v2_calculator_and_hessian() {
    use axtb_gxtb::v2::scf::Setup as V2Setup;
    let (mol, _) = reference("water");
    for conv in [Conventions::Axtb, Conventions::Gxtb] {
        let opts = ScfOptions { conventions: conv, ..Default::default() };
        let mut calc = Calculator::new(Version::V2_0_1, &mol.numbers, &mol.positions, 0.0, None, opts).unwrap();
        let r = calc.singlepoint(&mol.positions, true).unwrap();
        let setup = V2Setup::new(&mol, conv).unwrap();
        let direct = setup.scf(&ScfOptions { conventions: conv, ..Default::default() }).unwrap();
        assert_eq!(r.singlepoint.energy, direct.energies.total);
        assert_eq!(r.singlepoint.gradient.as_ref().unwrap(), &setup.gradient(&direct).unwrap());
        assert_eq!(r.energies.total(), r.singlepoint.energy);
        let named: f64 = r
            .energies
            .named()
            .iter()
            .filter(|(k, _)| ["electronic", "increments", "dispersion_three_body"].contains(k))
            .map(|(_, v)| v)
            .sum();
        assert!((named - r.singlepoint.energy).abs() < 1e-10);
        let h = calc.numerical_hessian(&mol.positions, 0.005, None).unwrap();
        assert!(h.max_asymmetry < 2e-5, "{}", h.max_asymmetry);
        for i in 0..9 {
            let row: f64 = (0..3).map(|a| h.hessian[i * 9 + 3 * a + 2]).sum();
            assert!(row.abs() < 1e-5, "translation invariance {row}");
        }
    }
}
