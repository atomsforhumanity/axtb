//! Analytic Hessian vs central differences of the analytic gradient (Richardson-extrapolated
//! from steps h and h/2; env STEP, default 2e-3 bohr) for reference structures
//! (`tests/reference-xtb-6.7.1/*.json`) given as arguments.

use axtb_gfnff::method::{HessianMethod, HessianOptions, Method, Want};
use axtb_gfnff::{Calculator, Options};
use serde_json::Value;

fn main() {
    let h: f64 = std::env::var("STEP").ok().and_then(|v| v.parse().ok()).unwrap_or(2e-3);
    let threads: usize = std::env::var("THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or(8);
    for path in std::env::args().skip(1) {
        let c: Value = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        let numbers: Vec<usize> =
            c["numbers"].as_array().unwrap().iter().map(|z| z.as_u64().unwrap() as usize).collect();
        let pos: Vec<[f64; 3]> = c["positions"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| {
                let a = p.as_array().unwrap();
                [a[0].as_f64().unwrap(), a[1].as_f64().unwrap(), a[2].as_f64().unwrap()]
            })
            .collect();
        let mut calc =
            Calculator::new(&numbers, &pos, c["charge"].as_f64().unwrap(), Options { threads, ..Default::default() })
                .unwrap();
        // evaluate at a randomly displaced geometry (PERTURB bohr; default 0.05): exactly
        // linear or symmetric arrangements of the built structures are non-differentiable
        // points of bends and out-of-line HB terms
        let amp: f64 = std::env::var("PERTURB").ok().and_then(|v| v.parse().ok()).unwrap_or(0.05);
        let mut seed: u64 = 12345;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((seed >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
        };
        let pos: Vec<[f64; 3]> =
            pos.iter().map(|p| [p[0] + amp * rnd(), p[1] + amp * rnd(), p[2] + amp * rnd()]).collect();
        let t = std::time::Instant::now();
        let ha = calc.hessian(&pos, &HessianOptions::default()).unwrap();
        let ta = t.elapsed().as_secs_f64();
        let h1 = calc.hessian(&pos, &numerical(h)).unwrap().hessian;
        let h2 = calc.hessian(&pos, &numerical(h / 2.0)).unwrap().hessian;
        let n3 = 3 * numbers.len();
        let (mut dmax, mut hmax, mut arg) = (0.0f64, 0.0f64, 0);
        for k in 0..n3 * n3 {
            let (i, j) = (k / n3, k % n3);
            let r = (4.0 * h2[k] - h1[k]) / 3.0;
            let rt = (4.0 * h2[j * n3 + i] - h1[j * n3 + i]) / 3.0;
            let d = (ha.hessian[k] - 0.5 * (r + rt)).abs();
            if d > dmax {
                dmax = d;
                arg = k;
            }
            hmax = hmax.max(ha.hessian[k].abs());
        }
        // dipole derivatives vs central differences of the EEQ dipole
        let mut ddmax: f64 = 0.0;
        for k in 0..n3 {
            let mut xp = pos.clone();
            let mut xm = pos.clone();
            xp[k / 3][k % 3] += 1e-4;
            xm[k / 3][k % 3] -= 1e-4;
            let dp = calc.singlepoint(&xp, &Want::gradient(false)).unwrap().dipole;
            let dm = calc.singlepoint(&xm, &Want::gradient(false)).unwrap().dipole;
            for e in 0..3 {
                ddmax = ddmax.max(((dp[e] - dm[e]) / 2e-4 - ha.dipole_derivatives.as_ref().unwrap()[k][e]).abs());
            }
        }
        println!("   dipole derivatives: max|an - fd| {ddmax:.1e}");
        if std::env::var("DETAIL").is_ok() {
            let col = arg % n3;
            for row in 0..n3 {
                let r = (4.0 * h2[row * n3 + col] - h1[row * n3 + col]) / 3.0;
                let rt = (4.0 * h2[col * n3 + row] - h1[col * n3 + row]) / 3.0;
                let a = ha.hessian[row * n3 + col];
                if (a - 0.5 * (r + rt)).abs() > 1e-7 {
                    println!(
                        "   row {:3} (atom {:3} Z={:2}) an {:+.6e} fd {:+.6e} fd^T {:+.6e}",
                        row,
                        row / 3,
                        numbers[row / 3],
                        a,
                        r,
                        rt
                    );
                }
            }
        }
        println!(
            "{:42} n={:4} max|H_an - H_fd| {:.2e} (at {},{}) max|H| {:.2e} asym {:.1e} t_an {:.3}s",
            path.rsplit('/').next().unwrap(),
            numbers.len(),
            dmax,
            arg / n3,
            arg % n3,
            hmax,
            ha.max_asymmetry,
            ta
        );
    }
}

/// Central differences of the analytic gradient with step `step` (bohr).
fn numerical(step: f64) -> HessianOptions {
    HessianOptions { method: Some(HessianMethod::Numerical), step: Some(step), accuracy: None }
}
