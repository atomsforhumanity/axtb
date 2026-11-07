//! Results on several threads (README, design rules): axtb's own parallel kernels (integrals,
//! Fock matrix, potentials incl. the multipole gather and the γ rows, D4's pair sums, Mulliken
//! charges) sum in a fixed order, so below faer's parallel tridiagonalization (~313 rows) a
//! calculation gives the same bits on 1 and 3 threads; above it, the same bits on repeats with
//! the same thread count, and round-off between thread counts.

use axtb::{Calculator, Conventions, Method, Options, SinglePoint};

/// A zigzag n-alkane CnH(2n+2) (bohr), idealized geometry.
fn alkane(n: usize) -> (Vec<usize>, Vec<[f64; 3]>) {
    const AA: f64 = 1.0 / 0.529177210544;
    let (mut z, mut r) = (Vec::new(), Vec::new());
    for i in 0..n {
        let up = if i % 2 == 0 { -1.0 } else { 1.0 };
        let c = [1.268 * i as f64, 0.428 * up, 0.0];
        z.push(6);
        r.push(c);
        for side in [-1.0, 1.0] {
            z.push(1);
            r.push([c[0], c[1] + 0.63 * up, 0.89 * side]);
        }
        if i == 0 || i == n - 1 {
            z.push(1);
            r.push([c[0] + if i == 0 { -1.03 } else { 1.03 }, c[1] - 0.36 * up, 0.0]);
        }
    }
    (z, r.into_iter().map(|p| p.map(|x| x * AA)).collect())
}

fn run(mol: &(Vec<usize>, Vec<[f64; 3]>), conventions: Conventions, threads: usize) -> SinglePoint {
    let mut o = Options { conventions, ..Default::default() };
    o.scf.threads = threads;
    let mut calc = Calculator::new(Method::Gfn2, &mol.0, &mol.1, 0.0, None, o).unwrap();
    calc.singlepoint(&mol.1, true).unwrap()
}

fn bits(sp: &SinglePoint) -> Vec<u64> {
    let g = sp.gradient.as_ref().unwrap().iter().flatten();
    std::iter::once(&sp.energy).chain(g).chain(&sp.qat).map(|x| x.to_bits()).collect()
}

/// C50H102: 302 basis functions, where the dense linear algebra runs on the pool but faer's
/// tridiagonalization does not.
#[test]
fn same_bits_on_one_and_three_threads_below_parallel_tridiagonalization() {
    // with a run-time LAPACK the library's own threaded tridiagonalization runs from 200 rows on
    if std::env::var_os("AXTB_LAPACK").is_some() {
        eprintln!("skipped: AXTB_LAPACK is set; this test is about faer's kernels");
        return;
    }
    let mol = alkane(50);
    for conv in [Conventions::Axtb, Conventions::Xtb] {
        let one = run(&mol, conv, 1);
        assert_eq!(one.orbital_energies.len(), 302);
        assert_eq!(bits(&run(&mol, conv, 3)), bits(&one), "{conv:?}");
    }
}

/// C80H162: 482 basis functions, with faer's parallel tridiagonalization.
#[test]
fn same_bits_on_repeats_above_parallel_tridiagonalization() {
    let mol = alkane(80);
    let one = run(&mol, Conventions::Axtb, 1);
    assert_eq!(one.orbital_energies.len(), 482);
    let three = run(&mol, Conventions::Axtb, 3);
    assert_eq!(bits(&run(&mol, Conventions::Axtb, 3)), bits(&three));
    assert!((one.energy - three.energy).abs() < 1e-10, "{} {}", one.energy, three.energy);
}
