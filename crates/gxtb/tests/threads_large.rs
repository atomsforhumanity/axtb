//! g-xTB 2.0.1 on several threads at a size where the dense linear algebra runs in parallel
//! (at least `axtb_base::threads::DENSE_MIN_AO` basis functions, and above the ~313 rows from which the
//! eigensolver's tridiagonalization is threaded): energy and gradient on 4 threads equal the
//! 1-thread results to round-off with the same SCF iterations, and a repeat on 4 threads gives
//! the same bits. Both conventions, energy and analytic gradient.
//!
//! Own test binary: the 4-thread pools would disturb the thread counts of `threads.rs`.

use axtb::method::{Method, Want};
use axtb_base::threads::DENSE_MIN_AO;
use axtb_gxtb::{Calculator, Conventions, Options, Version};

#[test]
fn threads_agree_on_a_large_molecule() {
    let xyz = include_str!("../../../tests/structures/large/water100.xyz");
    for conventions in [Conventions::Axtb, Conventions::Gxtb] {
        let mol = axtb_gxtb::model::read_xyz(xyz, 0.0, None, conventions).unwrap();
        let nao = axtb_gxtb::scf::Setup::new(&mol, axtb_gxtb::model::Model::v2_0_1(conventions)).unwrap().nao;
        assert!(nao >= DENSE_MIN_AO.max(314), "{nao} basis functions");
        let run = |threads: usize| {
            let o = Options { threads, conventions, ..Default::default() };
            let mut c = Calculator::new(Version::V2_0_1, &mol.numbers, &mol.positions, 0.0, None, o).unwrap();
            let sp = c.singlepoint(&mol.positions, &Want::gradient(true)).unwrap();
            (sp.energy, sp.scf().unwrap().iterations, sp.gradient.unwrap())
        };
        let (e1, it1, g1) = run(1);
        let (e4, it4, g4) = run(4);
        let dg = g1.as_flattened().iter().zip(g4.as_flattened()).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
        assert!((e1 - e4).abs() <= 1e-11, "{conventions:?}: energy {e1} vs {e4}");
        assert_eq!(it1, it4, "{conventions:?}: iterations");
        assert!(dg <= 1e-10, "{conventions:?}: gradient {dg:e}");
        assert_eq!(run(4), (e4, it4, g4), "{conventions:?}: a repeat on 4 threads");
    }
}
