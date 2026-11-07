// SPDX-License-Identifier: LGPL-3.0-or-later
//! Timing of the solvation steps for a reference structure, optionally
//! replicated on a grid to make a larger system.
//!
//!     cargo run --release -p axtb-solvation --example solvation_timing -- [case] [nrep]

use axtb_solvation::{Model, ReferenceState, Solvation};
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let case = args.get(1).map(String::as_str).unwrap_or("taxol_alpb_water");
    let nrep: usize = args.get(2).map(|s| s.parse().unwrap()).unwrap_or(1);
    let path = format!("{}/tests/data/{case}.ref", env!("CARGO_MANIFEST_DIR"));
    let text = std::fs::read_to_string(path).unwrap();
    let (mut z, mut x, mut q) = (vec![], vec![], vec![]);
    for line in text.lines() {
        let t: Vec<&str> = line.split_whitespace().collect();
        if t.first() == Some(&"atom") {
            z.push(t[1].parse::<usize>().unwrap());
            x.push([t[2].parse::<f64>().unwrap(), t[3].parse().unwrap(), t[4].parse().unwrap()]);
            q.push(t[5].parse::<f64>().unwrap());
        }
    }
    // replicate along a cubic grid with 30 bohr spacing
    let n0 = z.len();
    let side = (nrep as f64).cbrt().ceil() as usize;
    let (mut zz, mut xx, mut qq) = (vec![], vec![], vec![]);
    'outer: for a in 0..side {
        for b in 0..side {
            for c in 0..side {
                if zz.len() >= n0 * nrep {
                    break 'outer;
                }
                for i in 0..n0 {
                    zz.push(z[i]);
                    xx.push([x[i][0] + 30.0 * a as f64, x[i][1] + 30.0 * b as f64, x[i][2] + 30.0 * c as f64]);
                    qq.push(q[i]);
                }
            }
        }
    }
    let nat = zz.len();
    let solv = Solvation::new(&zz, "water", Model::Alpb, ReferenceState::Gsolv).unwrap();
    let reps = (200_000 / (nat * nat / 50 + 1)).clamp(1, 200);
    let t = Instant::now();
    let mut cache = solv.update(&xx);
    for _ in 1..reps {
        solv.update_into(&xx, &mut cache);
    }
    let t_upd = t.elapsed().as_secs_f64() / reps as f64;
    let mut v = vec![0.0; nat];
    let mut e = vec![0.0; nat];
    let t = Instant::now();
    for _ in 0..reps {
        solv.potential(&cache, &qq, &mut v);
        solv.energy(&cache, &qq, &mut e);
    }
    let t_pot = t.elapsed().as_secs_f64() / reps as f64;
    let mut g = vec![[0.0; 3]; nat];
    let t = Instant::now();
    for _ in 0..reps {
        solv.fixed_energy(&cache, &mut e, Some(&mut g));
        solv.gradient(&cache, &qq, &mut g);
    }
    let t_grad = t.elapsed().as_secs_f64() / reps as f64;
    let parts = solv.energy_parts(&cache, &qq);
    println!(
        "{case} x{nrep}: nat {nat}  update {:.3} ms  potential+energy {:.3} ms  gradient {:.3} ms  (Gsolv {:.9})",
        t_upd * 1e3,
        t_pot * 1e3,
        t_grad * 1e3,
        parts.gsolv
    );
}
