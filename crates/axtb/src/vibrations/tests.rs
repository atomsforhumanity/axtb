use super::*;

/// Hessian (row-major 3N x 3N) of harmonic springs V = Σ ½ k (|r_i − r_j| − d0)².
fn springs(pos: &[[f64; 3]], bonds: &[(usize, usize, f64, f64)]) -> Vec<f64> {
    let n3 = 3 * pos.len();
    let mut h = vec![0.0; n3 * n3];
    for &(i, j, k, d0) in bonds {
        let r = [pos[j][0] - pos[i][0], pos[j][1] - pos[i][1], pos[j][2] - pos[i][2]];
        let d = (r[0] * r[0] + r[1] * r[1] + r[2] * r[2]).sqrt();
        let u = r.map(|x| x / d);
        for a in 0..3 {
            for b in 0..3 {
                let delta = if a == b { 1.0 } else { 0.0 };
                let v = k * u[a] * u[b] + k * (d - d0) / d * (delta - u[a] * u[b]);
                for (p, q, s) in [(i, i, 1.0), (j, j, 1.0), (i, j, -1.0), (j, i, -1.0)] {
                    h[(3 * p + a) * n3 + 3 * q + b] += s * v;
                }
            }
        }
    }
    h
}

/// Atomic numbers, positions (bohr) and the bonds (atoms, length, force constant).
type Model = (Vec<usize>, Vec<[f64; 3]>, Vec<(usize, usize, f64, f64)>);

fn water() -> Model {
    let pos = vec![[0.0, 0.0, 0.0], [1.43, 0.0, 1.1], [-1.43, 0.0, 1.1]];
    let d = |a: [f64; 3], b: [f64; 3]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
    let bonds = vec![(0, 1, 0.5, d(pos[0], pos[1])), (0, 2, 0.5, d(pos[0], pos[2])), (1, 2, 0.05, d(pos[1], pos[2]))];
    (vec![8, 1, 1], pos, bonds)
}

fn rotate(p: [f64; 3]) -> [f64; 3] {
    let (s1, c1) = 0.4f64.sin_cos();
    let (s2, c2) = 1.1f64.sin_cos();
    let q = [c1 * p[0] - s1 * p[1], s1 * p[0] + c1 * p[1], p[2]];
    [q[0] + 3.0, c2 * q[1] - s2 * q[2] - 1.0, s2 * q[1] + c2 * q[2] + 0.5]
}

#[test]
fn diatomic() {
    let (k, d0) = (0.4, 1.4);
    let pos = vec![[0.0, 0.0, 0.0], [0.0, 0.0, d0]];
    let h = springs(&pos, &[(0, 1, k, d0)]);
    let masses = vec![1.0, 2.0];
    for conv in [Conventions::Axtb, Conventions::Xtb, Conventions::Xtb671] {
        let opts = VibOptions { conventions: conv, masses: Some(masses.clone()), thermo: None };
        let dmu = vec![[0.0; 3], [0.0; 3], [0.0, 0.0, 0.3], [0.0; 3], [0.0; 3], [0.0, 0.0, -0.3]];
        let r = analyze(&[1, 1], &pos, &h, Some(&dmu), &opts).unwrap();
        assert!(r.linear);
        assert_eq!(r.frequencies.len(), 1);
        let (amu, wn, irf) = match conv {
            Conventions::Axtb => {
                (1.0 / units::ELECTRON_MASS_IN_U, units::HARTREE_TO_WAVENUMBER, units::AU_TO_KM_PER_MOL)
            }
            Conventions::Xtb | Conventions::Xtb671 => (xtbc::AMUTOAU, xtbc::AUTORCM, xtbc::AUTOKMMOL),
        };
        let mu = 2.0 / 3.0;
        let w = (k / (mu * amu)).sqrt() * wn;
        assert!((r.frequencies[0] - w).abs() < 1e-9 * w, "{conv:?} {} vs {w}", r.frequencies[0]);
        // reduced mass of the normalized Cartesian displacement (Gaussian's definition):
        // m1 m2 (m1 + m2) / (m1² + m2²)
        let mu_red = 1.2;
        assert!((r.reduced_masses[0] - mu_red).abs() < 1e-12, "{}", r.reduced_masses[0]);
        let ir = irf * 0.09 / (mu * amu);
        let got = r.ir_intensities.as_ref().unwrap()[0];
        assert!((got - ir).abs() < 1e-9 * ir, "{got} vs {ir}");
        // unit-length Cartesian mode along z
        let m = &r.normal_modes[0];
        let nn: f64 = m.iter().flatten().map(|x| x * x).sum();
        assert!((nn - 1.0).abs() < 1e-12 && m[0][0].abs() < 1e-12);
        // force constant μ_red ω² in mdyn/Å
        let kmd = k * mu_red / mu * units::HARTREE_TO_JOULE / (units::AUTOAA * 1e-10).powi(2) / 100.0;
        if conv == Conventions::Axtb {
            assert!((r.force_constants[0] - kmd).abs() < 1e-9 * kmd);
        }
    }
}

#[test]
fn water_projection_and_invariance() {
    let (z, pos, bonds) = water();
    let h = springs(&pos, &bonds);
    let masses = vec![15.999, 1.008, 1.008];
    let mut freq = Vec::new();
    for conv in [Conventions::Axtb, Conventions::Xtb, Conventions::Xtb671] {
        let opts = VibOptions { conventions: conv, masses: Some(masses.clone()), thermo: None };
        let r = analyze(&z, &pos, &h, None, &opts).unwrap();
        assert_eq!(r.frequencies.len(), 3);
        assert!(!r.linear && r.n_imaginary == 0);
        freq.push(r.frequencies.clone());
        // rotated and translated molecule, Hessian rotated with it
        let rp: Vec<[f64; 3]> = pos.iter().map(|p| rotate(*p)).collect();
        let hr = springs(&rp, &bonds);
        let r2 = analyze(&z, &rp, &hr, None, &opts).unwrap();
        for (a, b) in r.frequencies.iter().zip(&r2.frequencies) {
            assert!((a - b).abs() < 1e-8, "{conv:?}: {a} vs {b}");
        }
    }
    // same masses: the conventions differ by the constants only (at a minimum both xtb
    // projections remove the same rigid-body motions)
    let ratio = (units::HARTREE_TO_WAVENUMBER / xtbc::AUTORCM) * (xtbc::AMUTOAU * units::ELECTRON_MASS_IN_U).sqrt();
    for k in 1..3 {
        for (a, b) in freq[0].iter().zip(&freq[k]) {
            assert!((a - b * ratio).abs() < 1e-8 * a, "{a} vs {}", b * ratio);
        }
    }
}

#[test]
fn xtb680_mass_weighted_projection() {
    // away from the minimum xtb 6.8.0 projects in mass-weighted coordinates: the frequencies
    // equal axtb's (same masses, xtb's constants) and differ from 6.7.1's Cartesian projection
    let (z, pos, mut bonds) = water();
    for b in &mut bonds {
        b.3 *= 0.9;
    }
    let h = springs(&pos, &bonds);
    let masses = vec![15.999, 1.008, 1.008];
    let run = |conv| {
        analyze(&z, &pos, &h, None, &VibOptions { conventions: conv, masses: Some(masses.clone()), thermo: None })
            .unwrap()
    };
    let (ax, x680, x671) = (run(Conventions::Axtb), run(Conventions::Xtb), run(Conventions::Xtb671));
    let ratio = (units::HARTREE_TO_WAVENUMBER / xtbc::AUTORCM) * (xtbc::AMUTOAU * units::ELECTRON_MASS_IN_U).sqrt();
    for (a, b) in ax.frequencies.iter().zip(&x680.frequencies) {
        assert!((a - b * ratio).abs() < 1e-8 * a.abs(), "{a} vs {}", b * ratio);
    }
    let d671 = x680.frequencies.iter().zip(&x671.frequencies).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    assert!(d671 > 1.0, "6.7.1's Cartesian projection should differ away from the minimum: {d671}");
    // the Hessian projected as 6.8.0 writes it gives the same analysis, also in 6.7.1's
    // conventions (its Cartesian projection of an already projected Hessian is a no-op up to
    // the rigid-body components)
    let hp = project_xtb(&z, &pos, &h, Conventions::Xtb).unwrap();
    let again = analyze(&z, &pos, &hp, None, &VibOptions { conventions: Conventions::Xtb, masses: None, thermo: None })
        .unwrap();
    let direct =
        analyze(&z, &pos, &h, None, &VibOptions { conventions: Conventions::Xtb, masses: None, thermo: None }).unwrap();
    for (a, b) in again.frequencies.iter().zip(&direct.frequencies) {
        assert!((a - b).abs() < 1e-8 * a.abs(), "{a} vs {b}");
    }
    let hpp = project_xtb(&z, &pos, &hp, Conventions::Xtb).unwrap();
    let dp = hp.iter().zip(&hpp).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
    assert!(dp < 1e-14, "projection not idempotent: {dp:e}");
}

#[test]
fn xtb680_linear_molecules() {
    // CO2-like springs: exactly linear and bent by 0.1 degree; xtb 6.8.0 counts a molecule as
    // linear when a unit-mass moment about the centroid is below 1e-4 bohr²
    let z = [8, 6, 8];
    for (bend, linear) in [(0.0f64, true), (0.004, true), (0.05, false)] {
        let pos = vec![[-2.2, 0.0, 0.0], [0.0, bend, 0.0], [2.2, 0.0, 0.0]];
        let d =
            |a: [f64; 3], b: [f64; 3]| ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
        let bonds =
            vec![(0, 1, 0.8, d(pos[0], pos[1])), (1, 2, 0.8, d(pos[1], pos[2])), (0, 2, 0.05, d(pos[0], pos[2]))];
        let h = springs(&pos, &bonds);
        assert_eq!(super::is_linear_xtb680(&pos), linear, "bend {bend}");
        let r = analyze(
            &z,
            &pos,
            &h,
            None,
            &VibOptions { conventions: Conventions::Xtb, masses: None, thermo: Some(ThermoOptions::default()) },
        )
        .unwrap();
        assert_eq!(r.linear, linear, "bend {bend}");
        assert_eq!(r.frequencies.len(), if linear { 4 } else { 3 }, "bend {bend}: {:?}", r.frequencies);
        let th = r.thermo.unwrap();
        assert_eq!(th.linear, linear);
    }
}

#[test]
fn xtb680_and_671_thermo_differ_by_the_bohr() {
    // same frequencies: G(RRHO) differs only through the bohr radius of the moments of inertia
    let (z, pos, bonds) = water();
    let h = springs(&pos, &bonds);
    let opts = |conv| VibOptions { conventions: conv, masses: None, thermo: Some(ThermoOptions::default()) };
    let a = analyze(&z, &pos, &h, None, &opts(Conventions::Xtb)).unwrap().thermo.unwrap();
    let b = analyze(&z, &pos, &h, None, &opts(Conventions::Xtb671)).unwrap().thermo.unwrap();
    let d = (a.points[0].g_rrho - b.points[0].g_rrho).abs();
    assert!(d > 0.0 && d < 1e-9, "{d:e}");
    assert_eq!(a.zpve.to_bits(), b.zpve.to_bits());
    let r = (a.rotational_constants[0] / b.rotational_constants[0] - 1.0).abs();
    assert!(r > 1e-8 && r < 1e-6, "{r:e}");
}

#[test]
fn analysis_convention_names() {
    for (s, c) in [
        ("axtb", Conventions::Axtb),
        ("xtb", Conventions::Xtb),
        ("xtb-6.8.0", Conventions::Xtb),
        ("xtb-6.7.1", Conventions::Xtb671),
        ("gxtb", Conventions::Xtb671),
    ] {
        assert_eq!(s.parse::<Conventions>().unwrap(), c);
    }
    assert_eq!(Conventions::from(crate::Conventions::Xtb), Conventions::Xtb);
    assert_eq!(Conventions::from(crate::Conventions::Xtb671), Conventions::Xtb671);
    assert_eq!(Conventions::from(crate::Conventions::Tblite), Conventions::Axtb);
    assert_eq!(Conventions::Xtb671.label(), "xtb-6.7.1");
}

#[test]
fn stretched_water_has_three_modes() {
    // away from the minimum the rotations are not zero modes of H; both projections remove them
    let (z, pos, mut bonds) = water();
    for b in &mut bonds {
        b.3 *= 0.9;
    }
    let h = springs(&pos, &bonds);
    for conv in [Conventions::Axtb, Conventions::Xtb, Conventions::Xtb671] {
        let r = analyze(&z, &pos, &h, None, &VibOptions { conventions: conv, ..Default::default() }).unwrap();
        assert_eq!(r.frequencies.len(), 3, "{conv:?}");
    }
}

#[test]
fn argon_sackur_tetrode() {
    let t = 298.15;
    let m = 39.948;
    let opts = VibOptions {
        conventions: Conventions::Axtb,
        masses: Some(vec![m]),
        thermo: Some(ThermoOptions { temperatures: vec![t], ..Default::default() }),
    };
    let r = analyze(&[18], &[[0.0; 3]], &[0.0; 9], None, &opts).unwrap();
    assert!(r.frequencies.is_empty());
    let th = r.thermo.unwrap();
    assert!(th.atom && th.symmetry_number == 1);
    // SI Sackur-Tetrode with CODATA 2022
    let (kb, h, na, mu) = (1.380649e-23, 6.62607015e-34, 6.02214076e23, 1.66053906892e-27);
    let p = 101325.0;
    let mass = m * mu;
    let s = kb * ((2.0 * std::f64::consts::PI * mass * kb * t / (h * h)).powf(1.5) * kb * t / p).ln() + 2.5 * kb;
    // Eh/K per molecule from J/K per molecule
    let s_eh = s / units::HARTREE_TO_JOULE;
    let got = th.points[0].entropy;
    assert!((got - s_eh).abs() < 1e-9 * s_eh, "{got} vs {s_eh}; {} J/mol/K", s * na);
    assert!((th.points[0].enthalpy - 2.5 * units::KB_HARTREE * t).abs() < 1e-15);
}

/// An atom has no vibrations in any conventions (the axtb conventions diagonalize the empty
/// Hessian of the complement of the rigid-body motions; this runs in every profile, also with
/// overflow checks).
#[test]
fn an_atom_has_no_vibrations() {
    assert!(eigh(&Mat::zeros(0, 0)).unwrap().0.is_empty());
    for conventions in [Conventions::Axtb, Conventions::Xtb, Conventions::Xtb671] {
        let opts = VibOptions { conventions, masses: None, thermo: None };
        let r = analyze(&[18], &[[0.0; 3]], &[0.0; 9], None, &opts).unwrap();
        assert!(r.frequencies.is_empty(), "{conventions:?}");
    }
}

#[test]
fn water_thermo_conventions_agree() {
    let (z, pos, bonds) = water();
    let h = springs(&pos, &bonds);
    let mut g = Vec::new();
    for conv in [Conventions::Axtb, Conventions::Xtb, Conventions::Xtb671] {
        let opts = VibOptions { conventions: conv, masses: None, thermo: Some(ThermoOptions::default()) };
        let r = analyze(&z, &pos, &h, None, &opts).unwrap();
        let th = r.thermo.unwrap();
        assert_eq!(th.point_group, "C2v");
        assert_eq!(th.symmetry_number, 2);
        assert_eq!(th.n_modes, 3);
        g.push(th.points[0].g_rrho);
    }
    assert!((g[0] - g[1]).abs() < 2e-6, "{g:?}");
}

/// GFN2-xTB dipole derivatives of the analytic and response Hessians against central
/// differences of converged dipoles (step 1e-4 bohr, tight SCF).
#[test]
fn gfn2_dipole_derivatives() {
    use crate::{Calculator, HessianMethod, HessianOptions, Method, Options};
    let cases = [
        (include_str!("../../../../tests/structures/water.xyz"), 0.0, None),
        // hydroxide (charged: ∂μ/∂R includes the charge times the displacement)
        ("2\n\nO 0.0 0.0 0.0\nH 0.0 0.1 0.97\n", -1.0, None),
        // OH radical, open shell
        ("2\n\nO 0.0 0.0 0.0\nH 0.0 0.1 0.97\n", 0.0, Some(1)),
    ];
    for (xyz, charge, uhf) in cases {
        let (num, pos) = crate::io::read_xyz(xyz, crate::units::AATOAU).unwrap();
        let mut o = Options::default();
        o.scf.accuracy = 1e-6;
        let mut calc = Calculator::new(Method::Gfn2, &num, &pos, charge, uhf, o).unwrap();
        let n3 = 3 * num.len();
        let h = 1e-4;
        let mut fd = vec![[0.0; 3]; n3];
        for x in 0..n3 {
            let mut d = [[0.0; 3]; 2];
            for (i, s) in [1.0, -1.0].into_iter().enumerate() {
                let mut p = pos.clone();
                p[x / 3][x % 3] += s * h;
                calc.reset_guess();
                d[i] = calc.singlepoint(&p, false).unwrap().dipole;
            }
            fd[x] = [0, 1, 2].map(|k| (d[0][k] - d[1][k]) / (2.0 * h));
        }
        for method in [HessianMethod::Analytic, HessianMethod::Response, HessianMethod::Numerical] {
            let step = if method == HessianMethod::Numerical { Some(1e-3) } else { None };
            let hs = calc.hessian(&pos, &HessianOptions { method, step, accuracy: Some(1e-8) }).unwrap();
            let dd = hs.dipole_derivatives.unwrap();
            let err =
                dd.iter().zip(&fd).flat_map(|(a, b)| (0..3).map(move |k| (a[k] - b[k]).abs())).fold(0.0f64, f64::max);
            let tol = if method == HessianMethod::Numerical { 1e-5 } else { 1e-6 };
            eprintln!(
                "{method:?} charge {charge} uhf {uhf:?}: max error {err:.2e} (max {:.3})",
                fd.iter().flatten().fold(0.0f64, |a, b| a.max(b.abs()))
            );
            assert!(err < tol, "{method:?} charge {charge} uhf {uhf:?}: max error {err:.2e}");
        }
    }
}
