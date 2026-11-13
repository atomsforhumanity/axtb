// SPDX-License-Identifier: LGPL-3.0-or-later
//! Timing of GFN-FF setup, energy + gradient and analytic Hessian for scaling benchmarks:
//! stdin `{"numbers", "positions" (bohr), "charge", "threads", "mode": "grad" | "hess",
//! "conventions": "xtb" | "xtb-6.7.1"}`,
//! stdout `{"natoms", "energy", "gnorm", "setup_s", "eval_s", "timings", ...}` plus the
//! gradient in "grad" mode (the Hessian is not printed; peak memory is measured from outside).

use axtb_gfnff::method::{HessianOptions, Method, Want};
use axtb_gfnff::{Calculator, Conventions, Options};
use serde_json::{json, Value};
use std::io::Read;
use std::time::Instant;

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
    drop(s);
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
    let mode = req["mode"].as_str().unwrap_or("grad").to_string();
    let charge = req["charge"].as_f64().unwrap_or(0.0);
    let conventions = conventions(&req);
    drop(req);
    let t = Instant::now();
    let mut calc =
        Calculator::new(&numbers, &pos, charge, Options { threads, conventions, ..Default::default() }).unwrap();
    let setup_s = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let out = if mode == "hess" {
        let h = calc.hessian(&pos, &HessianOptions::default()).unwrap();
        let n3 = 3 * numbers.len();
        let trace: f64 = (0..n3).map(|i| h.hessian[i * n3 + i]).sum();
        let gnorm = h.gradient.iter().flatten().map(|g| g * g).sum::<f64>().sqrt();
        json!({"energy": h.energy, "gnorm": gnorm, "trace": trace, "max_asymmetry": h.max_asymmetry,
               "timings": h.timings})
    } else {
        let r = calc.singlepoint(&pos, &Want::gradient(true)).unwrap();
        let gnorm = r.gradient.as_ref().unwrap().iter().flatten().map(|g| g * g).sum::<f64>().sqrt();
        json!({"energy": r.energy, "gnorm": gnorm, "timings": r.timings, "gradient": r.gradient})
    };
    let eval_s = t.elapsed().as_secs_f64();
    let mut out = out;
    out["natoms"] = json!(numbers.len());
    out["nbonds"] = json!(calc.topology().bonds.len());
    // sizes of xtb's HB index maps (atoms of the A-H...B bond list, hydrogens and others),
    // whose product NAB^2 NH sizes its `considered_ABH` array
    let t = calc.topology();
    let mut abh = vec![false; numbers.len()];
    for b in &t.bond_hb {
        abh[b.a] = true;
        abh[b.h] = true;
        for &x in &b.b {
            abh[x] = true;
        }
    }
    let nh = (0..numbers.len()).filter(|&i| abh[i] && numbers[i] == 1).count();
    let nab = (0..numbers.len()).filter(|&i| abh[i] && numbers[i] != 1).count();
    out["xtb_hb_map"] = json!({"nab": nab, "nh": nh});
    out["setup_s"] = json!(setup_s);
    out["eval_s"] = json!(eval_s);
    out["threads"] = json!(threads);
    println!("{out}");
}
