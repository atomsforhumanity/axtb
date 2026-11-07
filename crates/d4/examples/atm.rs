// SPDX-License-Identifier: LGPL-3.0-or-later
//! Timing of the D4 three-body term on a structure (xyz, Ångström) with GFN2-xTB damping:
//! `cargo run --release --example atm -- file.xyz [threads] [cutoff width] [pair_sums 0|1]`
use axtb_d4::{ChargeModel, D4Dispersion, D4Model, RationalDamping, RealspaceCutoff};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let text = std::fs::read_to_string(&args[1]).unwrap();
    let threads: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(1);
    let cut: f64 = args.get(3).map(|s| s.parse().unwrap()).unwrap_or(40.0);
    let width: f64 = args.get(4).map(|s| s.parse().unwrap()).unwrap_or(1.0);
    let sums = args.get(5).map(|s| s == "1").unwrap_or(true);
    let sym = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"];
    let mut lines = text.lines();
    let nat: usize = lines.next().unwrap().trim().parse().unwrap();
    lines.next();
    let (mut numbers, mut pos) = (Vec::new(), Vec::new());
    for l in lines.take(nat) {
        let f: Vec<&str> = l.split_whitespace().collect();
        numbers.push(sym.iter().position(|s| s.eq_ignore_ascii_case(f[0])).unwrap() + 1);
        let x: Vec<f64> = f[1..4].iter().map(|s| s.parse::<f64>().unwrap() / 0.529177210544).collect();
        pos.push([x[0], x[1], x[2]]);
    }
    let cutoff = RealspaceCutoff { cn: 40.0, disp2: 72.0, disp3: cut, width2: width, width3: width };
    let model = D4Model::new(&numbers, ChargeModel::Gfn2).unwrap();
    let disp = D4Dispersion::new(model, RationalDamping::GFN2, cutoff).with_pair_sums(sums);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
    pool.install(|| {
        let cache = disp.update(&pos);
        let mut best = (f64::MAX, f64::MAX);
        let (mut e3, mut g) = (vec![0.0; nat], vec![[0.0; 3]; nat]);
        for _ in 0..5 {
            e3.iter_mut().for_each(|x| *x = 0.0);
            g.iter_mut().for_each(|x| *x = [0.0; 3]);
            let t = Instant::now();
            disp.three_body(&cache, &mut e3, Some(&mut g), threads > 1);
            best.0 = best.0.min(t.elapsed().as_secs_f64());
            let mut e = vec![0.0; nat];
            let t = Instant::now();
            disp.three_body(&cache, &mut e, None, threads > 1);
            best.1 = best.1.min(t.elapsed().as_secs_f64());
        }
        let gn: f64 = g.iter().flatten().map(|x| x * x).sum::<f64>().sqrt();
        println!("nat {nat} threads {threads} cutoff {cut} width {width} pair_sums {sums}: e+g {:.4} s, e {:.4} s  E3 {:.15e} |G| {:.15e}",
            best.0, best.1, e3.iter().sum::<f64>(), gn);
    });
}
