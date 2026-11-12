//! Broken-symmetry unrestricted singlets (`ScfOptions::broken_symmetry`, an axtb extension):
//! the acceptance rule, ⟨S²⟩, restarts, gradients and Hessians on the unrestricted surface, and
//! the refusals.

use axtb::{Error, HessianMethod, HessianOptions};
use axtb_gxtb::broken_symmetry::{s2_v1, Outcome};
use axtb_gxtb::{read_xyz, Calculator, CompositeCalculator, CompositeOptions, Conventions, ScfOptions, Setup, Version};
use serde_json::Value;
use std::path::Path;

/// Bohr per Ångström (CODATA 2022).
const AA: f64 = 1.0 / 0.529177210544;

fn h2(d: f64) -> (Vec<usize>, Vec<[f64; 3]>) {
    // off the axes, so that every Cartesian derivative is tested
    (vec![1, 1], vec![[0.0, 0.0, 0.0], [0.1, 0.2, (d * d - 0.05 / (AA * AA)).sqrt() * AA]])
}

fn bs(accuracy: f64) -> ScfOptions {
    ScfOptions { broken_symmetry: true, accuracy, ..Default::default() }
}

fn calc(version: Version, numbers: &[usize], x: &[[f64; 3]], uhf: Option<usize>, o: ScfOptions) -> Calculator {
    Calculator::new(version, numbers, x, 0.0, uhf, o).unwrap()
}

/// The 1.1.0 program prints ⟨S²⟩ − S_z(S_z + 1) of its open shells as "spin-contamination"
/// (6 decimals): `axtb_gxtb::broken_symmetry::s2` on its densities gives the same.
#[test]
fn s2_matches_the_v11_programs_spin_contamination() {
    for (name, printed, sz) in [("tetrachlorocuprate", 0.000796, 0.5), ("dioxygen_triplet", 0.000253, 1.0)] {
        let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/reference").join(name);
        let input: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("input.json")).unwrap()).unwrap();
        let mol = read_xyz(
            &std::fs::read_to_string(dir.join("mol.xyz")).unwrap(),
            input["charge"].as_f64().unwrap(),
            Some(input["uhf"].as_u64().unwrap() as usize),
        )
        .unwrap();
        let setup = Setup::new(&mol).unwrap();
        let r = setup.scf(&ScfOptions { conventions: Conventions::Gxtb, ..Default::default() }).unwrap();
        let contamination = s2_v1(&setup, &r) - sz * (sz + 1.0);
        assert!((contamination - printed).abs() <= 5e-7, "{name}: {contamination} vs the program's {printed}");
    }
}

/// H2: restricted near equilibrium (the result is the restricted SCF's, bit for bit), the
/// radical pair at 3 Å (⟨S²⟩ near 1, at the triplet's energy), in both versions.
#[test]
fn h2_reaches_the_radical_pair_limit() {
    for version in [Version::V2_0_1, Version::V1_1] {
        let (z, x) = h2(0.74);
        let plain = calc(version, &z, &x, None, ScfOptions::default()).singlepoint(&x, true).unwrap();
        let r = calc(version, &z, &x, None, bs(1.0)).singlepoint(&x, true).unwrap();
        let b = r.broken_symmetry.as_ref().unwrap();
        assert_eq!(b.outcome, Outcome::NotLower, "{version:?}");
        assert_eq!((b.s2, b.start, b.pairs), (0.0, Some("homo-lumo"), 1));
        assert_eq!(r.singlepoint.energy, plain.singlepoint.energy);
        assert_eq!(r.singlepoint.gradient, plain.singlepoint.gradient);
        assert_eq!(r.singlepoint.orbital_energies.len(), plain.singlepoint.orbital_energies.len());
        assert!(plain.broken_symmetry.is_none());
        assert!((b.unrestricted_energy.unwrap() - b.restricted_energy.unwrap()).abs() < 1e-8);

        let (z, x) = h2(3.0);
        let rest = calc(version, &z, &x, None, ScfOptions::default()).singlepoint(&x, false).unwrap();
        let triplet = calc(version, &z, &x, Some(2), ScfOptions::default()).singlepoint(&x, false).unwrap();
        let r = calc(version, &z, &x, None, bs(1.0)).singlepoint(&x, false).unwrap();
        let b = r.broken_symmetry.as_ref().unwrap();
        assert!(b.broken(), "{version:?}: {b:?}");
        assert_eq!(b.restricted_energy, Some(rest.singlepoint.energy));
        assert_eq!(b.unrestricted_energy, Some(r.singlepoint.energy));
        assert!(b.s2 > 0.97 && b.s2 < 1.0, "{version:?}: <S2> {}", b.s2);
        let (e, et, er) = (r.singlepoint.energy, triplet.singlepoint.energy, rest.singlepoint.energy);
        // 58 kcal/mol below the restricted singlet, within 2 kcal/mol of the triplet
        assert!(er - e > 0.08, "{version:?}: {e} vs restricted {er}");
        assert!((e - et).abs() < 0.0032, "{version:?}: {e} vs triplet {et}");
        // two spin channels with the same electron count
        assert_eq!(r.singlepoint.levels.len(), 2);
        let occ: Vec<f64> = r.singlepoint.levels.iter().map(|l| l.occupations.iter().sum()).collect();
        assert_eq!(occ, vec![1.0, 1.0]);
    }
}

/// Central differences (h = 1e-4 bohr) of broken-symmetry energies against the analytic
/// gradient, where the symmetry is broken (H2 at 1.6 and 2 Å), both versions.
#[test]
fn gradients_match_finite_differences() {
    for version in [Version::V2_0_1, Version::V1_1] {
        for d in [1.6, 2.0] {
            let (z, x) = h2(d);
            let mut c = calc(version, &z, &x, None, bs(1e-4));
            let r = c.singlepoint(&x, true).unwrap();
            assert!(r.broken_symmetry.as_ref().unwrap().broken());
            let g = r.singlepoint.gradient.unwrap();
            let h = 1e-4;
            for (a, k) in [(1, 0), (1, 1), (1, 2), (0, 2)] {
                let mut e = [0.0; 2];
                for (i, s) in [1.0, -1.0].iter().enumerate() {
                    let mut y = x.clone();
                    y[a][k] += s * h;
                    let ry = c.singlepoint(&y, false).unwrap();
                    assert_eq!(ry.broken_symmetry.as_ref().unwrap().start, Some("previous"));
                    e[i] = ry.singlepoint.energy;
                }
                let fd = (e[0] - e[1]) / (2.0 * h);
                assert!((fd - g[a][k]).abs() < 2e-8, "{version:?} {d} Å ({a}, {k}): {} vs {fd}", g[a][k]);
            }
        }
    }
}

/// Analytic and response Hessians (2.0.1) of a broken-symmetry reference against central
/// differences of its analytic gradients; the numerical Hessian's displacements stay on the
/// unrestricted surface.
#[test]
fn hessians_on_the_unrestricted_surface() {
    let (z, x) = h2(2.0);
    let mut c = calc(Version::V2_0_1, &z, &x, None, bs(1e-5));
    let r = c.singlepoint(&x, true).unwrap();
    assert!(r.broken_symmetry.as_ref().unwrap().broken());
    let h = 5e-4;
    let mut fd = vec![0.0; 36];
    for j in 0..6 {
        let mut g = [[0.0; 6]; 2];
        for (i, s) in [1.0, -1.0].iter().enumerate() {
            let mut y = x.clone();
            y[j / 3][j % 3] += s * h;
            let gy = c.singlepoint(&y, true).unwrap().singlepoint.gradient.unwrap();
            g[i] = std::array::from_fn(|k| gy[k / 3][k % 3]);
        }
        for k in 0..6 {
            fd[k * 6 + j] = (g[0][k] - g[1][k]) / (2.0 * h);
        }
    }
    for method in [HessianMethod::Analytic, HessianMethod::Response, HessianMethod::Numerical] {
        let mut c = calc(Version::V2_0_1, &z, &x, None, bs(1.0));
        let hess = c.hessian(&x, &HessianOptions { method, step: None, accuracy: None }).unwrap();
        assert!((hess.energy - r.singlepoint.energy).abs() < 1e-9, "{method:?}");
        let tol = if method == HessianMethod::Numerical { 2e-6 } else { 1e-8 };
        for k in 0..36 {
            let s = 0.5 * (fd[k] + fd[(k % 6) * 6 + k / 6]);
            assert!((hess.hessian[k] - s).abs() < tol, "{method:?} [{k}]: {} vs {s}", hess.hessian[k]);
        }
    }
}

/// A calculator whose previous state is unrestricted runs one unrestricted SCF from it; when
/// the magnetization collapses, the restricted solution is computed and returned, and the next
/// geometry starts over from the restricted SCF.
#[test]
fn restarts_follow_the_unrestricted_state() {
    for version in [Version::V2_0_1, Version::V1_1] {
        let (z, x) = h2(3.0);
        let mut c = calc(version, &z, &x, None, bs(1.0));
        let first = c.singlepoint(&x, false).unwrap();
        assert_eq!(first.broken_symmetry.unwrap().start, Some("homo-lumo"));
        let (_, y) = h2(2.9);
        let r = c.singlepoint(&y, false).unwrap();
        let b = r.broken_symmetry.unwrap();
        assert_eq!((b.outcome, b.start, b.restricted_energy), (Outcome::Broken, Some("previous"), None));
        assert_eq!(r.singlepoint.guess, "previous");
        // the same geometry once more: the restart lands on the same solution
        let fresh = calc(version, &z, &y, None, bs(1.0)).singlepoint(&y, false).unwrap();
        assert!((r.singlepoint.energy - fresh.singlepoint.energy).abs() < 1e-7, "{version:?}");
        // down to the bonded region: collapse, restricted result
        let mut last = Outcome::Broken;
        for d in [2.5, 2.0, 1.6, 1.4, 1.2, 1.0, 0.9] {
            let (_, y) = h2(d);
            let r = c.singlepoint(&y, false).unwrap();
            let b = r.broken_symmetry.unwrap();
            if last == Outcome::Broken && !b.broken() {
                assert_eq!(b.outcome, Outcome::Collapsed, "{version:?} {d} Å: {b:?}");
                assert!(b.restricted_energy.is_some());
            }
            if last != Outcome::Broken {
                assert_eq!(b.start, Some("homo-lumo"), "{version:?} {d} Å: {b:?}");
            }
            last = b.outcome;
        }
        assert_ne!(last, Outcome::Broken, "{version:?}: H2 at 0.9 Å stays restricted");
    }
}

/// The setting is refused where it would mean nothing or break a promise.
#[test]
fn refusals() {
    let (z, x) = h2(3.0);
    for version in [Version::V2_0_1, Version::V1_1] {
        // the program conventions promise the programs' results
        let o = ScfOptions { conventions: Conventions::Gxtb, ..bs(1.0) };
        assert!(matches!(Calculator::new(version, &z, &x, 0.0, None, o), Err(Error::Input(_))));
        // open shells already have two spin channels
        assert!(matches!(Calculator::new(version, &z, &x, 0.0, Some(2), bs(1.0)), Err(Error::Input(_))));
        assert!(matches!(Calculator::new(version, &z[..1], &x[..1], 0.0, None, bs(1.0)), Err(Error::Input(_))));
        // switched on later
        let mut c = calc(version, &z, &x, Some(2), ScfOptions::default());
        c.options.broken_symmetry = true;
        assert!(matches!(c.singlepoint(&x, false), Err(Error::Input(_))));
    }
    let mol = axtb::Molecule::new(&z, &x, 0.0, Some(2)).unwrap();
    let mut s = axtb_gxtb::v2::scf::Setup::new(&mol, Conventions::Axtb).unwrap();
    assert!(s.set_unrestricted(false).is_err());
    let mut s = Setup::new(&mol).unwrap();
    assert!(s.set_unrestricted(false).is_err());
}

/// A broken-symmetry state given to a restricted calculator starts it from its total density.
#[test]
fn a_restricted_calculator_takes_a_broken_symmetry_state() {
    for version in [Version::V2_0_1, Version::V1_1] {
        let (z, x) = h2(3.0);
        let mut a = calc(version, &z, &x, None, bs(1.0));
        a.singlepoint(&x, false).unwrap();
        let mut b = calc(version, &z, &x, None, ScfOptions::default());
        b.set_guess(a.guess().unwrap().clone()).unwrap();
        let r = b.singlepoint(&x, false).unwrap();
        assert_eq!(r.singlepoint.guess, "previous");
        assert_eq!(r.singlepoint.levels.len(), 1);
        let plain = calc(version, &z, &x, None, ScfOptions::default()).singlepoint(&x, false).unwrap();
        assert!((r.singlepoint.energy - plain.singlepoint.energy).abs() < 1e-7);
    }
}

/// Fermi smearing: the start is the integer-occupied mixed determinant; the unrestricted SCF
/// smears its own occupations.
#[test]
fn finite_electronic_temperature() {
    let (z, x) = h2(3.0);
    let o = ScfOptions { etemp: 3000.0, ..bs(1.0) };
    let r = calc(Version::V2_0_1, &z, &x, None, o).singlepoint(&x, false).unwrap();
    let b = r.broken_symmetry.unwrap();
    assert!(b.broken(), "{b:?}");
    assert!(r.energies.named().iter().any(|&(k, v)| k == "entropy" && v <= 0.0));
}

/// The solvation composite applies the setting to its g-xTB part.
#[test]
fn the_composite_breaks_its_gxtb_part() {
    let (z, x) = h2(3.0);
    let o = CompositeOptions { broken_symmetry: true, ..Default::default() };
    let r = CompositeCalculator::new(&z, &x, 0.0, None, o).unwrap().singlepoint(&x, true).unwrap();
    assert!(r.gxtb.broken_symmetry.as_ref().unwrap().broken());
    let mut plain = CompositeCalculator::new(&z, &x, 0.0, None, CompositeOptions::default()).unwrap();
    let plain = plain.singlepoint(&x, false).unwrap();
    assert!(plain.gxtb.broken_symmetry.is_none());
    assert!(r.singlepoint.energy < plain.singlepoint.energy - 0.08);
    assert_eq!(r.solvation_energy, plain.solvation_energy);
    let o = CompositeOptions { broken_symmetry: true, conventions: Conventions::Gxtb, ..Default::default() };
    assert!(CompositeCalculator::new(&z, &x, 0.0, None, o).is_err());
}
