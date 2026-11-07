// SPDX-License-Identifier: LGPL-3.0-or-later
//! Timing of the D4 pieces on a water cluster: `cargo run --release -p axtb-d4 --example d4_timing -- 1500`
use axtb_d4::D4Dispersion;
use std::time::Instant;

fn main() {
    let nmol: usize = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(1000);
    // water molecules on a cubic grid with 5.7 bohr spacing (liquid density)
    let side = (nmol as f64).cbrt().ceil() as usize;
    let (mut numbers, mut pos) = (Vec::new(), Vec::new());
    'outer: for a in 0..side {
        for b in 0..side {
            for c in 0..side {
                if numbers.len() / 3 == nmol {
                    break 'outer;
                }
                let o = [a as f64 * 5.7, b as f64 * 5.7, c as f64 * 5.7];
                let t = 0.37 * (a + 2 * b + 3 * c) as f64;
                numbers.extend([8, 1, 1]);
                pos.push(o);
                pos.push([o[0] + 1.81 * t.cos(), o[1] + 1.81 * t.sin(), o[2]]);
                pos.push([o[0] - 0.45, o[1] + 0.3, o[2] + 1.75]);
            }
        }
    }
    let nat = numbers.len();
    let q: Vec<f64> = (0..nat).map(|i| if i % 3 == 0 { -0.6 } else { 0.3 }).collect();
    let disp = D4Dispersion::gfn2(&numbers).unwrap();

    let t = Instant::now();
    let cache = disp.update(&pos);
    let t_update = t.elapsed();
    let mut e = vec![0.0; nat];
    let mut v = vec![0.0; nat];
    let t = Instant::now();
    disp.energy_and_potential(&cache, &q, &mut e, &mut v);
    let t_ep = t.elapsed();
    let mut g = vec![[0.0; 3]; nat];
    let t = Instant::now();
    disp.gradient(&cache, &q, &mut g);
    let t_grad = t.elapsed();
    let mut e3 = vec![0.0; nat];
    let t = Instant::now();
    disp.three_body(&cache, &mut e3, Some(&mut g), false);
    let t_3b = t.elapsed();
    println!(
        "nat {nat}  pairs {}  update {:?}  energy+potential {:?}  gradient(2b) {:?}  three-body e+g {:?}",
        cache.npairs(),
        t_update,
        t_ep,
        t_grad,
        t_3b
    );
    println!("E2 = {:.12e}  E3 = {:.12e}", e.iter().sum::<f64>(), e3.iter().sum::<f64>());
}
