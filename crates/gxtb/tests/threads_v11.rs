// SPDX-License-Identifier: LGPL-3.0-or-later
//! g-xTB v1.1 on several threads (v1.1 runs on the engine, whose setup and gradient loops run in
//! parallel in fixed chunks summed in order): 4 threads give the 1-thread bits. Caffeine stays
//! below the size from which the dense linear algebra runs in parallel; taxol (299 basis functions
//! in v1.1's q-vSZP basis) is above it. Both conventions, energy, SCF iterations and analytic
//! gradient; caffeine also at 3000 K.
//!
//! Own test binary: the 4-thread pools would disturb the thread counts of `threads.rs`.

use axtb::method::{Method, ScfSettings, Want};
use axtb_base::threads::DENSE_MIN_AO;
use axtb_gxtb::{read_xyz, Calculator, Conventions, Options, Version};

type Run = (f64, usize, Vec<[f64; 3]>);

fn run(mol: &axtb::Molecule, threads: usize, conventions: Conventions, etemp: f64) -> Run {
    let o = Options {
        threads,
        conventions,
        scf: ScfSettings { etemp: Some(etemp), ..Default::default() },
        ..Default::default()
    };
    let mut c = Calculator::new(Version::V1_1, &mol.numbers, &mol.positions, mol.charge, mol.uhf, o).unwrap();
    let sp = c.singlepoint(&mol.positions, &Want::gradient(true)).unwrap();
    (sp.energy, sp.scf().unwrap().iterations, sp.gradient.unwrap())
}

fn nao(mol: &axtb::Molecule) -> usize {
    axtb_gxtb::scf::Setup::new(mol, axtb_gxtb::model::Model::v1_1(Conventions::Axtb)).unwrap().nao
}

#[test]
fn v11_threads_bitwise_below_the_dense_size() {
    let mol = read_xyz(include_str!("../../../tests/structures/caffeine.xyz"), 0.0, None).unwrap();
    assert!(nao(&mol) < DENSE_MIN_AO, "{} basis functions", nao(&mol));
    for conventions in [Conventions::Axtb, Conventions::Gxtb] {
        for etemp in [0.0, 3000.0] {
            assert_eq!(run(&mol, 1, conventions, etemp), run(&mol, 4, conventions, etemp), "{conventions:?} {etemp} K");
        }
    }
}

#[test]
fn v11_threads_bitwise_above_the_dense_size() {
    let mol = read_xyz(include_str!("../../../tests/structures/taxol.xyz"), 0.0, None).unwrap();
    assert!(nao(&mol) >= DENSE_MIN_AO, "{} basis functions", nao(&mol));
    for conventions in [Conventions::Axtb, Conventions::Gxtb] {
        assert_eq!(run(&mol, 1, conventions, 0.0), run(&mol, 4, conventions, 0.0), "{conventions:?}");
    }
}
