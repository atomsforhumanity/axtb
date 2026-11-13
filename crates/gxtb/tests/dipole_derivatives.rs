// SPDX-License-Identifier: LGPL-3.0-or-later
//! Dipole derivatives of the g-xTB Hessians (for IR intensities) against central differences
//! of converged dipoles: 2.0.1 analytic, response and numerical; v1.1 numerical.

use axtb::method::{Method, Want};
use axtb::method::{Restart, ScfSettings};
use axtb_gxtb::{Calculator, Options, Version};

fn fd_dipole(calc: &mut Calculator, pos: &[[f64; 3]], h: f64) -> Vec<[f64; 3]> {
    let n3 = 3 * pos.len();
    (0..n3)
        .map(|x| {
            let mut d = [[0.0; 3]; 2];
            for (i, s) in [1.0, -1.0].into_iter().enumerate() {
                let mut p = pos.to_vec();
                p[x / 3][x % 3] += s * h;
                d[i] = calc.singlepoint(&p, &Want::gradient(false)).unwrap().dipole;
            }
            [0, 1, 2].map(|k| (d[0][k] - d[1][k]) / (2.0 * h))
        })
        .collect()
}

fn max_err(a: &[[f64; 3]], b: &[[f64; 3]]) -> f64 {
    a.iter().zip(b).flat_map(|(x, y)| (0..3).map(move |k| (x[k] - y[k]).abs())).fold(0.0, f64::max)
}

#[test]
fn dipole_derivatives() {
    let cases: [(&str, f64, Option<usize>); 3] = [
        ("3\n\nO 0.0 0.0 0.12\nH 0.0 0.76 -0.47\nH 0.0 -0.77 -0.46\n", 0.0, None),
        ("2\n\nO 0.0 0.0 0.0\nH 0.0 0.1 0.97\n", -1.0, None),
        ("2\n\nO 0.0 0.0 0.0\nH 0.0 0.1 0.97\n", 0.0, Some(1)),
    ];
    for (xyz, charge, uhf) in cases {
        let (num, pos) = axtb::io::read_xyz(xyz, axtb::units::AATOAU).unwrap();
        for version in [Version::V2_0_1, Version::V1_1] {
            // v1.1's open-shell SCF needs tighter thresholds for 1e-4 bohr differences
            let acc = if version == Version::V1_1 { 1e-7 } else { 1e-4 };
            // every displaced SCF from the program's start, so that its convergence error is the
            // same on both sides of the central differences
            let scf = ScfSettings { accuracy: acc, restart: Some(Restart::Fresh), ..Default::default() };
            let opts = Options { scf, ..Default::default() };
            let mut calc = Calculator::new(version, &num, &pos, charge, uhf, opts).unwrap();
            let fd = fd_dipole(&mut calc, &pos, 1e-4);
            let methods: &[axtb::HessianMethod] = match version {
                Version::V2_0_1 => {
                    &[axtb::HessianMethod::Analytic, axtb::HessianMethod::Response, axtb::HessianMethod::Numerical]
                }
                Version::V1_1 => &[axtb::HessianMethod::Numerical],
            };
            for &method in methods {
                let step = if method == axtb::HessianMethod::Numerical { Some(1e-3) } else { None };
                let h = calc
                    .hessian(&pos, &axtb::HessianOptions { method: Some(method), step, accuracy: Some(acc) })
                    .unwrap();
                let e = max_err(h.dipole_derivatives.as_ref().unwrap(), &fd);
                eprintln!("{version:?} {method:?} charge {charge} uhf {uhf:?}: max error {e:.2e}");
                assert!(e < 2e-6, "{version:?} {method:?} charge {charge} uhf {uhf:?}: max error {e:.2e}");
            }
        }
    }
}
