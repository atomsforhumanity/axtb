//! A calculator with `threads > 1` must run all of its work inside its own pool: every SCF,
//! including the retry from scratch after an SCF that started from a bad guess failed, and
//! every Hessian method, including the reference SCF of the numerical Hessian. Before the
//! fixes the retry and that reference SCF ran outside the pool, so rayon's global pool (one
//! thread per core) did their parallel work.
//!
//! Own test binary, one test: whether rayon's global pool was ever used is observable only
//! once per process (`build_global` fails after its first use).

use axtb::method::{Method, ScfData, Want, Wavefunction, WavefunctionData};
use axtb::{Calculator, Hamiltonian, HessianMethod, HessianOptions, Options};

#[test]
fn calculations_stay_in_the_calculator_pool() {
    let numbers = [8, 1, 1];
    let positions = [[0.0, 0.0, 0.22], [0.0, 1.43, -0.89], [0.0, -1.43, -0.89]];
    let mut opts = Options { threads: 2, ..Default::default() };
    // water converges in 8 iterations from scratch and needs 11 from the bad guess below
    opts.scf.max_iter = Some(9);
    let fresh = Calculator::new(Hamiltonian::Gfn2, &numbers, &positions, 0.0, None, opts.clone())
        .unwrap()
        .singlepoint(&positions, &Want::gradient(true))
        .unwrap();

    // a guess the SCF cannot recover from within 9 iterations: the first attempt fails and
    // the calculator retries from scratch
    let nsh = fresh.electronic.as_ref().unwrap().shell_charges.len();
    let bad = ScfData {
        shell_charges: (0..nsh).map(|i| if i % 2 == 0 { 4.0 } else { -4.0 }).collect(),
        atomic_dipoles: vec![[3.0, -3.0, 3.0]; 3],
        atomic_quadrupoles: vec![[5.0, 0.0, -5.0, 0.0, 0.0, 0.0]; 3],
        ..ScfData::default()
    };
    let mut calc = Calculator::new(Hamiltonian::Gfn2, &numbers, &positions, 0.0, None, opts.clone()).unwrap();
    let fingerprint = axtb::method::Fingerprint::new("gfn2", &numbers, 0.0, None);
    calc.set_wavefunction(&Wavefunction::new(fingerprint, WavefunctionData::Scf(bad))).unwrap();
    let retried = calc.singlepoint(&positions, &Want::gradient(true)).unwrap();
    // same start (scratch), same iterations, same result as the fresh calculation
    assert_eq!(retried.scf().unwrap().iterations, fresh.scf().unwrap().iterations);
    assert!((retried.energy - fresh.energy).abs() < 1e-12);

    // the three Hessian methods, each from a new calculator (no previous wavefunction)
    opts.scf.max_iter = Some(250);
    for method in [HessianMethod::Analytic, HessianMethod::Response, HessianMethod::Numerical] {
        let mut calc = Calculator::new(Hamiltonian::Gfn2, &numbers, &positions, 0.0, None, opts.clone()).unwrap();
        let h = calc.hessian(&positions, &HessianOptions { method: Some(method), ..Default::default() }).unwrap();
        assert!(h.hessian.iter().all(|x| x.is_finite()), "{method:?}");
    }

    assert!(
        rayon::ThreadPoolBuilder::new().num_threads(1).build_global().is_ok(),
        "rayon's global pool was used by a calculator that owns a pool"
    );
}
