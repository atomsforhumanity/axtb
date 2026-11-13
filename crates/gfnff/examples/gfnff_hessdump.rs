// SPDX-License-Identifier: LGPL-3.0-or-later
//! Analytic GFN-FF Hessian as JSON: stdin `{"numbers", "positions" (bohr), "charge",
//! "threads", "conventions"}`, stdout `{"energy", "hessian" (row-major 3N x 3N), "seconds"}`.

use axtb_gfnff::method::{HessianOptions, Method};
use axtb_gfnff::{Calculator, Conventions, Options};
use serde_json::{json, Value};
use std::io::Read;

/// The `conventions` of the request (`xtb`/`axtb`/`xtb-6.8.0`: xtb 6.8.0's force field, the
/// default; `xtb-6.7.1`: xtb 6.7.1's).
fn conventions(req: &Value) -> Conventions {
    match req.get("conventions").and_then(|v| v.as_str()) {
        Some(s) => Conventions::from_calculation(s).unwrap_or_else(|| panic!("unknown conventions {s:?}")),
        None => Conventions::default(),
    }
}

fn main() {
    let mut s = String::new();
    std::io::stdin().read_to_string(&mut s).unwrap();
    let req: Value = serde_json::from_str(&s).unwrap();
    let numbers: Vec<usize> = req["numbers"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let pos: Vec<[f64; 3]> = req["positions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| {
            let a = p.as_array().unwrap();
            [a[0].as_f64().unwrap(), a[1].as_f64().unwrap(), a[2].as_f64().unwrap()]
        })
        .collect();
    let threads = req["threads"].as_u64().unwrap_or(1) as usize;
    let t = std::time::Instant::now();
    let mut calc = Calculator::new(
        &numbers,
        &pos,
        req["charge"].as_f64().unwrap_or(0.0),
        Options { threads, conventions: conventions(&req), ..Default::default() },
    )
    .unwrap();
    let h = calc.hessian(&pos, &HessianOptions::default()).unwrap();
    println!(
        "{}",
        json!({"energy": h.energy, "hessian": h.hessian, "seconds": t.elapsed().as_secs_f64(), "timings": h.timings,
               "max_asymmetry": h.max_asymmetry})
    );
}
