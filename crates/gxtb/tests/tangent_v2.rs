// SPDX-License-Identifier: LGPL-3.0-or-later
//! g-xTB 2.0.1 analytic setup derivatives (overlap, dipole and quadrupole integrals, H0 + ACP)
//! against 4-point differences of the setup. `V2_TAN=reference/water,...` selects molecules.

use axtb_gxtb::h0::add_anisotropy;
use axtb_gxtb::scf::Setup;
use axtb_gxtb::Conventions;
use serde_json::Value;
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/reference_v2")
}

fn molecule(rel: &str, conv: Conventions) -> axtb::Molecule {
    let dir = root().join(rel);
    let input: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("input.json")).unwrap()).unwrap();
    let charge = input["charge"].as_f64().unwrap();
    let uhf = input["uhf"].as_u64().unwrap() as usize;
    let xyz = std::fs::read_to_string(dir.join("mol.xyz")).unwrap();
    axtb_gxtb::model::read_xyz(&xyz, charge, Some(uhf), conv).unwrap()
}

/// H0 + ACP (the setup's one-electron Hamiltonian without the anisotropy term).
fn h_noaniso(s: &Setup) -> Vec<f64> {
    let mut a = vec![0.0; s.nao * s.nao];
    add_anisotropy(s.nao, &s.basis.scf.ao2at, &s.dpint, &s.aniso, &mut a);
    s.h.iter().zip(&a).map(|(h, x)| h - x).collect()
}

fn mats(s: &Setup) -> Vec<Vec<f64>> {
    let mut v = vec![s.s.clone()];
    v.extend(s.dpint.iter().cloned());
    v.extend(s.qpint.iter().cloned());
    v.push(h_noaniso(s));
    v
}

#[test]
fn tangents_match_finite_differences() {
    let cases = std::env::var("V2_TAN").unwrap_or("reference/water,reference/nickel_tetracarbonyl".into());
    for rel in cases.split(',') {
        for c in [Conventions::Axtb, Conventions::Gxtb] {
            let mol = molecule(rel, c);
            let setup = Setup::new(&mol, axtb_gxtb::model::Model::v2_0_1(c)).unwrap();
            let tan = setup.tangents().unwrap();
            let n3 = 3 * mol.nat();
            let h = 1e-3;
            let names = ["S", "Dx", "Dy", "Dz", "Q0", "Q1", "Q2", "Q3", "Q4", "Q5", "H"];
            let mut worst = [0.0f64; 11];
            let mut scale = [0.0f64; 11];
            for x in 0..n3 {
                let at = |d: f64| {
                    let mut m = mol.clone();
                    m.positions[x / 3][x % 3] += d;
                    mats(&Setup::new(&m, axtb_gxtb::model::Model::v2_0_1(c)).unwrap())
                };
                let (p1, m1, p2, m2) = (at(h), at(-h), at(2.0 * h), at(-2.0 * h));
                let t = tan.cartesian(x);
                let mut an = vec![t.s.clone()];
                an.extend(t.d.iter().cloned());
                an.extend(t.q.iter().cloned());
                an.push(t.h.clone());
                for k in 0..11 {
                    for e in 0..an[k].len() {
                        let fd = (8.0 * (p1[k][e] - m1[k][e]) - (p2[k][e] - m2[k][e])) / (12.0 * h);
                        worst[k] = worst[k].max((fd - an[k][e]).abs());
                        scale[k] = scale[k].max(fd.abs());
                    }
                }
            }
            let rel_err: Vec<String> =
                (0..11).map(|k| format!("{} {:.1e}", names[k], worst[k] / scale[k].max(1e-300))).collect();
            println!("{rel:30} {c:?}: {}", rel_err.join(", "));
            for k in 0..11 {
                assert!(worst[k] <= 1e-8 * scale[k].max(1.0), "{rel} {c:?} {}: {:e}", names[k], worst[k]);
            }
        }
    }
}
