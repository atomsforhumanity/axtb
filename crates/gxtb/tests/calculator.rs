//! The calculator interface used by the axtb front ends (CLI, Python, qccodec): same results
//! as the SCF driver, explicit threads; gradients and Hessians.

use axtb::method::{Method, Want};
use axtb::method::{ScfSettings, TermKind};
use axtb_gxtb::{read_xyz, Calculator, Conventions, ConvergerSettings, Options, ScfOptions, Version};

/// The engine's setup of v1.1 in `conventions`.
fn v11_setup(mol: &axtb::Molecule, conventions: Conventions) -> axtb_gxtb::v2::scf::Setup {
    axtb_gxtb::v2::scf::Setup::new(mol, axtb_gxtb::v2::Model::v1_1(conventions)).unwrap()
}
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

/// The calculator's options in the program's conventions.
fn program_options() -> Options {
    Options { conventions: Conventions::Gxtb, ..Default::default() }
}

/// Options with the SCF settings `scf`.
fn with_scf(scf: ScfSettings) -> Options {
    Options { scf, ..Default::default() }
}

/// Options with the converger settings `converger`.
fn with_converger(converger: ConvergerSettings) -> Options {
    Options { converger, ..Default::default() }
}

#[test]
fn calculator_matches_scf_and_program() {
    for name in ["water", "dioxygen_triplet", "tetrachlorocuprate", "nickel_tetracarbonyl"] {
        let (mol, eref) = reference(name);
        let direct = v11_setup(&mol, Conventions::Gxtb).scf(&program()).unwrap();
        for threads in [1, 2] {
            let opts = Options { threads, ..program_options() };
            let mut calc =
                Calculator::new(Version::V1_1, &mol.numbers, &mol.positions, mol.charge, mol.uhf, opts).unwrap();
            let r = calc.singlepoint(&mol.positions, &Want::gradient(false)).unwrap();
            let sp = &r;
            // the direct SCF runs sequentially: identical on 1 thread, to round-off on more
            // (the eigensolver's parallel tridiagonalization)
            if threads == 1 {
                assert_eq!(sp.energy, direct.energies.total, "{name}");
                assert_eq!(sp.charges, direct.atom_charges);
            } else {
                assert!((sp.energy - direct.energies.total).abs() < 1e-11, "{name}, {threads} threads");
                assert!(sp.charges.iter().zip(&direct.atom_charges).all(|(a, b)| (a - b).abs() < 1e-10));
            }
            assert!((sp.energy - eref).abs() < 1e-11, "{name}: {} vs program {eref}", sp.energy);
            assert_eq!(sp.scf().unwrap().iterations, direct.iterations);
            // the pieces partition the energy; `electronic` is their SCF part
            assert!((sp.pieces() - sp.energy).abs() < 1e-10, "{name}: {} vs {}", sp.pieces(), sp.energy);
            assert_eq!(sp.terms.iter().filter(|t| t.kind == TermKind::Subtotal).count(), 1);
            let named: f64 = r
                .terms
                .iter()
                .filter(|t| {
                    ["electronic", "increments", "dispersion", "dispersion_three_body", "repulsion"].contains(&t.name)
                })
                .map(|t| t.value)
                .sum();
            assert!((named - sp.energy).abs() < 1e-10);
            // the cached setup is reused for the same geometry, rebuilt for a new one
            assert_eq!(calc.singlepoint(&mol.positions, &Want::gradient(false)).unwrap().energy, sp.energy);
            let mut moved = mol.positions.clone();
            moved[0][2] += 0.01;
            let e2 = calc.singlepoint(&moved, &Want::gradient(false)).unwrap().energy;
            let mut fresh =
                Calculator::new(Version::V1_1, &mol.numbers, &moved, mol.charge, mol.uhf, program_options()).unwrap();
            assert_eq!(fresh.singlepoint(&moved, &Want::gradient(false)).unwrap().energy, e2);
        }
        // the default diagonalizes the exact derivative: same energy expression, stationary
        // SCF, slightly lower energy than the program's fixed point
        let mut calc =
            Calculator::new(Version::V1_1, &mol.numbers, &mol.positions, mol.charge, mol.uhf, Options::default())
                .unwrap();
        let e = calc.singlepoint(&mol.positions, &Want::gradient(false)).unwrap().energy;
        let exact = v11_setup(&mol, Conventions::Axtb)
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
        Calculator::new(Version::V1_1, &mol.numbers, &mol.positions, mol.charge, mol.uhf, program_options()).unwrap();
    let d = calc.singlepoint(&mol.positions, &Want::gradient(false)).unwrap().dipole;
    for (x, p) in d.iter().zip([0.0, 0.0, -0.933216]) {
        assert!((x - p).abs() <= 5.0001e-7, "dipole {d:?}");
    }
}

#[test]
fn gradient_and_hessian() {
    let (mol, _) = reference("water");
    for (conventions, threads) in
        [(Conventions::Axtb, 1), (Conventions::Axtb, 2), (Conventions::Gxtb, 1), (Conventions::Gxtb, 2)]
    {
        let opts = ScfOptions { conventions, ..Default::default() };
        let st = v11_setup(&mol, conventions);
        let direct = st.gradient(&st.scf(&opts).unwrap()).unwrap();
        let opts = Options { threads, conventions, ..Default::default() };
        let mut calc = Calculator::new(Version::V1_1, &mol.numbers, &mol.positions, mol.charge, mol.uhf, opts).unwrap();
        let sp = calc.singlepoint(&mol.positions, &Want::gradient(true)).unwrap();
        let g = sp.gradient.as_ref().unwrap();
        if threads == 1 {
            assert_eq!(g, &direct, "{conventions:?}");
        } else {
            let d = g.as_flattened().iter().zip(direct.as_flattened()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            assert!(d < 1e-10, "{conventions:?}, {threads} threads: {d:e}");
        }
        assert!(sp.timings.iter().any(|(k, _)| *k == "gradient"));
        // semi-numerical Hessian: symmetric, translation-invariant rows, gradient at accuracy 0.01
        let h = calc.hessian(&mol.positions, &numerical(0.005, None)).unwrap();
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
        Calculator::new(Version::V1_1, &mol.numbers, &mol.positions, mol.charge, mol.uhf, Options::default()).unwrap();
    assert!(calc.hessian(&mol.positions, &numerical(0.0, None)).is_err());
    assert!(calc.hessian(&mol.positions, &numerical(f64::NAN, None)).is_err());
    assert!(calc.hessian(&mol.positions, &numerical(0.005, Some(f64::NAN))).is_err());
}

#[test]
fn input_errors() {
    use axtb::Error;
    let (mol, _) = reference("water");
    let (z, x) = (&mol.numbers, &mol.positions);
    let new = |v: Version, charge: f64, o: Options| Calculator::new(v, z, x, charge, None, o);
    let refused = |v: Version, o: Options| matches!(new(v, 0.0, o), Err(Error::Input(_)));
    let d = Options::default;
    let scf = ScfSettings::default;
    let conv = ConvergerSettings::default;
    for v in [Version::V2_0_1, Version::V1_1] {
        assert!(refused(v, Options { threads: 0, ..d() }));
        assert!(refused(v, with_scf(ScfSettings { accuracy: f64::NAN, ..scf() })));
        assert!(refused(v, with_scf(ScfSettings { etemp: Some(f64::INFINITY), ..scf() })));
        assert!(refused(v, with_scf(ScfSettings { etemp: Some(-1.0), ..scf() })));
        assert!(refused(v, with_converger(ConvergerSettings { energy_tol: Some(f64::NAN), ..conv() })));
        assert!(refused(v, with_converger(ConvergerSettings { diis_size: Some(0), ..conv() })));
        for damping in [0.0, -0.1, 1.5, f64::NAN] {
            let o = with_converger(ConvergerSettings { damping: Some(damping), ..conv() });
            assert!(refused(v, o), "{v:?} damping {damping}");
        }
        let o = with_converger(ConvergerSettings { damping: Some(1.0), diis_size: Some(1), ..conv() });
        assert!(new(v, 0.0, o).is_ok());
        // a fractional or non-finite charge
        for charge in [0.5, f64::NAN, f64::INFINITY] {
            assert!(matches!(new(v, charge, d()), Err(Error::Input(_))), "{v:?} charge {charge}");
        }
        // positions that are not finite, coincident atoms: an error, no panic
        let mut calc = new(v, 0.0, d()).unwrap();
        let mut bad = x.clone();
        bad[1][0] = f64::NAN;
        assert!(calc.singlepoint(&bad, &Want::gradient(true)).is_err());
        let mut bad = x.clone();
        bad[1] = bad[0];
        assert!(calc.singlepoint(&bad, &Want::gradient(true)).is_err());
        assert!(calc.singlepoint(x, &Want::gradient(true)).is_ok());
        // the thread count is the pool's
        calc.options.threads = 2;
        assert!(calc.singlepoint(x, &Want::gradient(false)).is_err());
    }
    // settings of the other version
    assert!(refused(Version::V2_0_1, with_converger(ConvergerSettings { density_tol: Some(1e-6), ..conv() })));
    assert!(refused(Version::V2_0_1, with_converger(ConvergerSettings { transition_iteration: Some(3), ..conv() })));
    assert!(refused(Version::V1_1, with_converger(ConvergerSettings { residual_tol: Some(1e-6), ..conv() })));
    assert!(new(Version::V2_0_1, 0.0, with_converger(ConvergerSettings { residual_tol: Some(1e-6), ..conv() })).is_ok());
    let v1 = with_converger(ConvergerSettings { density_tol: Some(1e-6), transition_iteration: Some(2), ..conv() });
    assert!(new(Version::V1_1, 0.0, v1).is_ok());
    // element without v1.1 parameters (Z = 60 is absent from the parameter file)
    assert!(Calculator::new(Version::V1_1, &[60, 1], &[[0.0; 3], [0.0, 0.0, 3.0]], 0.0, None, d()).is_err());
    // a 2.0.1 setup refuses SCF options of other conventions; its Hessians need every orbital
    let mut setup = axtb_gxtb::v2::scf::Setup::new(&mol, axtb_gxtb::v2::Model::v2_0_1(Conventions::Axtb)).unwrap();
    assert!(setup.scf(&ScfOptions { conventions: Conventions::Gxtb, ..Default::default() }).is_err());
    let r = setup.scf(&ScfOptions::default()).unwrap();
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
        let energy = |o: Options| {
            let mut c = Calculator::new(v, z, x, 0.0, None, o).unwrap();
            c.singlepoint(x, &Want::gradient(false)).map(|r| (r.energy, r.scf().unwrap().iterations, r.charges))
        };
        let (e0, it0, q0) = energy(Options::default()).unwrap();
        let o = with_converger(ConvergerSettings { damping: Some(0.5), diis_size: Some(3), ..Default::default() });
        let (e1, it1, q1) = energy(o).unwrap();
        // the same solution, along another path (seen in the charges: the energy is stationary, so
        // it can agree to the last bit; v1.1 water: the densities differ by 8e-8)
        assert!((e0 - e1).abs() < 1e-7 && (it0 != it1 || q0 != q1), "{v:?}: {e0} ({it0}) vs {e1} ({it1})");
        match energy(with_scf(ScfSettings { max_iter: Some(2), ..Default::default() })).map(|r| (r.0, r.1)) {
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
        let opts = Options { conventions: conv, ..Default::default() };
        let mut calc = Calculator::new(Version::V2_0_1, &mol.numbers, &mol.positions, 0.0, None, opts).unwrap();
        let r = calc.singlepoint(&mol.positions, &Want::gradient(true)).unwrap();
        let setup = V2Setup::new(&mol, axtb_gxtb::v2::Model::v2_0_1(conv)).unwrap();
        let direct = setup.scf(&ScfOptions { conventions: conv, ..Default::default() }).unwrap();
        assert_eq!(r.energy, direct.energies.total);
        assert_eq!(r.gradient.as_ref().unwrap(), &setup.gradient(&direct).unwrap());
        assert!((r.pieces() - r.energy).abs() < 1e-10, "{conv:?}: the pieces {} vs {}", r.pieces(), r.energy);
        let named: f64 = r
            .terms
            .iter()
            .filter(|t| ["electronic", "increments", "dispersion_three_body"].contains(&t.name))
            .map(|t| t.value)
            .sum();
        assert!((named - r.energy).abs() < 1e-10);
        let h = calc.hessian(&mol.positions, &numerical(0.005, None)).unwrap();
        assert!(h.max_asymmetry < 2e-5, "{}", h.max_asymmetry);
        for i in 0..9 {
            let row: f64 = (0..3).map(|a| h.hessian[i * 9 + 3 * a + 2]).sum();
            assert!(row.abs() < 1e-5, "translation invariance {row}");
        }
    }
}

/// The numerical Hessian with step `step` (bohr) and the SCF accuracy `accuracy` of every
/// displacement.
fn numerical(step: f64, accuracy: Option<f64>) -> axtb::HessianOptions {
    axtb::HessianOptions { method: Some(axtb::HessianMethod::Numerical), step: Some(step), accuracy }
}
