// SPDX-License-Identifier: LGPL-3.0-or-later
//! Time and memory of the D4 nuclear Hessian (GFN2-xTB setup, Mulliken-like test charges) of a
//! structure: `cargo run --release -p axtb-d4 --example hessian_timing -- FILE.xyz THREADS
//! [REPEAT [CUTOFFS]]` with CUTOFFS `tblite` (default, smooth), `xtb` (hard) or `short` (a 12 bohr
//! three-body cutoff with a 2 bohr switch, for tests). Run it under `/usr/bin/time -v` for the
//! peak memory. Prints the best wall time of the repeats and a hash of the Hessian's bits, to
//! compare builds bit for bit.
use axtb_d4::{ChargeModel, D4Dispersion, D4Model, RationalDamping, RealspaceCutoff, AATOAU};
use std::hash::{Hash, Hasher};
use std::time::Instant;

const SYMBOLS: [&str; 18] =
    ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar"];

fn main() {
    let mut args = std::env::args().skip(1);
    let path = args.next().expect("usage: hessian_timing FILE.xyz THREADS [REPEAT [CUTOFFS]]");
    let threads: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1);
    let repeat: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(1);
    let cutoff = match args.next().as_deref() {
        None | Some("tblite") => RealspaceCutoff::TBLITE_GFN2,
        Some("xtb") => RealspaceCutoff::XTB_GFN2,
        Some("short") => RealspaceCutoff { cn: 30.0, disp2: 50.0, disp3: 12.0, width2: 0.05, width3: 2.0 },
        Some(c) => panic!("unknown cutoffs {c}"),
    };
    let text = std::fs::read_to_string(&path).expect("read the structure");
    let mut lines = text.lines();
    let nat: usize = lines.next().and_then(|l| l.trim().parse().ok()).expect("atom count");
    lines.next();
    let (mut numbers, mut pos) = (Vec::new(), Vec::new());
    for l in lines.take(nat) {
        let f: Vec<&str> = l.split_whitespace().collect();
        let z = SYMBOLS.iter().position(|s| s.eq_ignore_ascii_case(f[0])).expect("element H to Ar") + 1;
        numbers.push(z);
        pos.push([1, 2, 3].map(|k| f[k].parse::<f64>().unwrap() * AATOAU));
    }
    // charges of a polar molecule's size, the same every run
    let q: Vec<f64> = (0..nat).map(|i| 0.1 * ((i as f64) * 0.7).sin()).collect();
    let model = D4Model::new(&numbers, ChargeModel::Gfn2).unwrap();
    let disp = D4Dispersion::new(model, RationalDamping::GFN2, cutoff);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
    let mut best = f64::INFINITY;
    let mut hash = 0u64;
    for _ in 0..repeat {
        let (dt, h) = pool.install(|| {
            let t = Instant::now();
            let cache = disp.update(&pos);
            let mut h = vec![0.0; 9 * nat * nat];
            disp.hessian(&cache, &q, &mut h, threads > 1);
            (t.elapsed().as_secs_f64(), h)
        });
        best = best.min(dt);
        let mut s = std::collections::hash_map::DefaultHasher::new();
        h.iter().for_each(|x| x.to_bits().hash(&mut s));
        hash = s.finish();
    }
    println!("nat {nat}  threads {threads}  best of {repeat}: {best:.3} s  hessian bits {hash:016x}");
}
