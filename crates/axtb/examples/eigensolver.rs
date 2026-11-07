//! Time per call of the SCF eigensolver for F C = S C ε: `GenEigSolver::solve_lowest` with
//! the lowest two thirds of the eigenvectors (a closed-shell SCF's occupied orbitals in a
//! minimal basis) and `solve` with all of them, on a random symmetric F and an overlap-like S
//! (unit diagonal, off-diagonal elements decaying with |i - j|). faer, or from 200 rows on the
//! library that `AXTB_LAPACK` names. Prints the median over the repeats (after one unrecorded
//! call that allocates the work space). For the block sizes of the reduction to standard form
//! (`SYGST_BLOCK` and its neighbours in `src/linalg.rs`): edit, rebuild, rerun.
//!
//!     cargo run --release -p axtb --example eigensolver -- N THREADS [REPEATS]

use axtb::linalg::GenEigSolver;
use faer::{Mat, Par};
use std::time::Instant;

fn main() {
    let args: Vec<usize> = std::env::args().skip(1).map(|a| a.parse().expect("N THREADS [REPEATS]")).collect();
    let (n, threads, repeats) = match args[..] {
        [n, t] => (n, t, 9),
        [n, t, r] => (n, t, r),
        _ => panic!("usage: eigensolver N THREADS [REPEATS]"),
    };
    let mut seed = 3u64;
    let mut rnd = || {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((seed >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
    };
    let (mut f, mut s) = (Mat::<f64>::zeros(n, n), Mat::<f64>::identity(n, n));
    for i in 0..n {
        for j in 0..=i {
            let v = rnd();
            f[(i, j)] = v;
            f[(j, i)] = v;
            if j < i {
                let w = 0.3 * rnd() / (1.0 + (i - j) as f64);
                s[(i, j)] = w;
                s[(j, i)] = w;
            }
        }
    }
    let k = 2 * n / 3;
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
    let par = if threads == 1 { Par::Seq } else { Par::rayon(threads) };
    let median = |mut t: Vec<f64>| {
        t.sort_by(f64::total_cmp);
        t[t.len() / 2]
    };
    pool.install(|| {
        let mut solver = GenEigSolver::new(s.as_ref(), par).unwrap();
        let (mut eps, mut c) = (vec![0.0; n], Mat::<f64>::zeros(n, n));
        let mut time = |lowest: bool| {
            let times: Vec<f64> = (0..=repeats)
                .map(|_| {
                    let t = Instant::now();
                    if lowest {
                        solver.solve_lowest(f.as_ref(), &mut eps, &mut c, |_| k).unwrap();
                    } else {
                        solver.solve(f.as_ref(), &mut eps, &mut c).unwrap();
                    }
                    t.elapsed().as_secs_f64()
                })
                .collect();
            median(times[1..].to_vec())
        };
        let (lowest, full) = (time(true), time(false));
        println!(
            "n {n} threads {threads} ({}): solve_lowest (k = {k}) {:.2} ms, solve {:.2} ms",
            solver.backend(),
            lowest * 1e3,
            full * 1e3
        );
    });
}
