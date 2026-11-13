// SPDX-License-Identifier: LGPL-3.0-or-later
//! Dump the GFN-FF topology, charges, energies and gradient of a structure as JSON (for
//! the comparisons with xtb in `tests/gfnff_reproduction.py`).
//!
//! Input on stdin: `{"numbers": [...], "positions": [[x,y,z],...] (bohr), "charge": q,
//! "gradient": true, "topology": true, "threads": 1, "reference_positions": null,
//! "conventions": "xtb"}`; with `reference_positions` the topology is set up there and the
//! energy evaluated at `positions`; `conventions` `xtb-6.7.1` selects xtb 6.7.1's GFN-FF.

use axtb_gfnff::method::{Method, Want};
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
    let positions: Vec<[f64; 3]> = req["positions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| {
            let a = p.as_array().unwrap();
            [a[0].as_f64().unwrap(), a[1].as_f64().unwrap(), a[2].as_f64().unwrap()]
        })
        .collect();
    let charge = req["charge"].as_f64().unwrap_or(0.0);
    let grad = req["gradient"].as_bool().unwrap_or(true);
    let want_topo = req["topology"].as_bool().unwrap_or(true);
    let threads = req["threads"].as_u64().unwrap_or(1) as usize;
    let t0 = std::time::Instant::now();
    // the topology comes from `reference_positions` if given (fixed-topology evaluation)
    let refpos: Vec<[f64; 3]> = match req.get("reference_positions") {
        Some(v) if !v.is_null() => v
            .as_array()
            .unwrap()
            .iter()
            .map(|p| {
                let a = p.as_array().unwrap();
                [a[0].as_f64().unwrap(), a[1].as_f64().unwrap(), a[2].as_f64().unwrap()]
            })
            .collect(),
        _ => positions.clone(),
    };
    let mut calc = match Calculator::new(
        &numbers,
        &refpos,
        charge,
        Options { threads, conventions: conventions(&req), ..Default::default() },
    ) {
        Ok(c) => c,
        Err(e) => {
            println!("{}", json!({"error": e.to_string()}));
            return;
        }
    };
    let t_setup = t0.elapsed().as_secs_f64();
    let t1 = std::time::Instant::now();
    let res = match calc.singlepoint(&positions, &Want::gradient(grad)) {
        Ok(r) => r,
        Err(e) => {
            println!("{}", json!({"error": e.to_string()}));
            return;
        }
    };
    let t_sp = t1.elapsed().as_secs_f64();
    let hb = res.extras.gfnff.as_ref().expect("GFN-FF's report");
    let mut out = json!({
        "energy": res.energy,
        "terms": res.terms.iter().map(|t| (t.name, t.value)).collect::<std::collections::BTreeMap<_, _>>(),
        "gradient": res.gradient,
        "charges": res.charges,
        "nhb1": hb.n_hbonds[0], "nhb2": hb.n_hbonds[1], "nxb": hb.n_xbonds,
        "time_setup": t_setup, "time_singlepoint": t_sp,
        "timings": res.timings,
    });
    if want_topo {
        let t = calc.topology();
        let n = t.natoms();
        // explicit pair lists as xtb stores them (1-based)
        let mut hbab = Vec::new();
        for i in 0..n {
            if !t.hb_ab[i] {
                continue;
            }
            for j in 0..i {
                if !t.hb_ab[j] {
                    continue;
                }
                if t.hbbas[i] * t.hbaci[j] < 1.0e-6 && t.hbaci[i] * t.hbbas[j] < 1.0e-6 {
                    continue;
                }
                hbab.push([i + 1, j + 1]);
            }
        }
        let mut xbab = Vec::new();
        for &(a, x) in &t.xb_ax {
            for j in 0..n {
                if j == a || j == x {
                    continue;
                }
                if t.bpair.get(x, j) <= 3 {
                    continue;
                }
                if !t.xb_b[j] {
                    continue;
                }
                xbab.push([a + 1, j + 1, x + 1]);
            }
        }
        let mut alphanb = Vec::new();
        let mut zetac6 = Vec::new();
        let mut bpair = Vec::new();
        for i in 0..n {
            for j in 0..=i {
                alphanb.push(t.alphanb(i, j));
                zetac6.push(t.zeta[i] * t.zeta[j]);
            }
            let row: Vec<u8> = (0..n).map(|j| t.bpair.get(j, i)).collect();
            bpair.push(row);
        }
        out["topology"] = json!({
            "nb": t.nb.iter().map(|l| l.iter().map(|&k| k + 1).collect::<Vec<_>>()).collect::<Vec<_>>(),
            "hyb": t.hyb, "itag": t.itag, "imetal": t.imetal, "pi": t.pi, "amide": t.amide,
            "bpair": bpair,
            "blist": t.bonds.iter().map(|b| [b.j + 1, b.i + 1]).collect::<Vec<_>>(),
            "vbond": t.bonds.iter().map(|b| b.vbond).collect::<Vec<_>>(),
            "nr_hb": t.bonds.iter().map(|b| b.nr_hb).collect::<Vec<_>>(),
            "pibo": t.bonds.iter().map(|b| b.pibo).collect::<Vec<_>>(),
            "alist": t.angles.iter().map(|a| [a.i + 1, a.j + 1, a.k + 1]).collect::<Vec<_>>(),
            "vangl": t.angles.iter().map(|a| [a.phi0, a.fc]).collect::<Vec<_>>(),
            "tlist": t.torsions.iter().map(|q| [q.atoms[0] as i64 + 1, q.atoms[1] as i64 + 1, q.atoms[2] as i64 + 1, q.atoms[3] as i64 + 1, q.nrot as i64]).collect::<Vec<_>>(),
            "vtors": t.torsions.iter().map(|q| [q.phi0, q.fc]).collect::<Vec<_>>(),
            "b3list": t.batm.iter().map(|b| [b[0] + 1, b[1] + 1, b[2] + 1]).collect::<Vec<_>>(),
            "stors": t.stors.iter().map(|s| s.map(|v| v + 1)).collect::<Vec<_>>(),
            "nfrag": t.nfrag, "fraglist": t.fraglist, "qfrag": t.qfrag,
            "qa": t.qa, "chieeq": t.chieeq, "gameeq": t.gameeq, "alpeeq": t.alpeeq,
            "hbbas": t.hbbas, "hbaci": t.hbaci,
            "hbatHl": t.hb_h.iter().map(|&h| h + 1).collect::<Vec<_>>(),
            "hbatABl": hbab, "xbatABl": xbab,
            "bond_hb": t.bond_hb.iter().map(|e| json!({"a": e.a + 1, "h": e.h + 1, "b": e.b.iter().map(|&b| b + 1).collect::<Vec<_>>()})).collect::<Vec<_>>(),
            "alphanb": alphanb, "zetac6": zetac6,
            "cn": t.cn_setup, "mchar": t.mchar,
        });
    }
    println!("{}", out);
}
