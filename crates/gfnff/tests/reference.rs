//! GFN-FF against xtb 6.8.0 and 6.7.1: topology, EEQ charges, per-term energies and gradients
//! of the reference sets in `tests/reference-xtb-6.8.0/` (xtb 6.8.0, default conventions) and
//! `tests/reference-xtb-6.7.1/` (xtb 6.7.1, the g-xTB 2.0.1 build; conventions `Xtb671`), written by
//! `scripts/make_reference.py`.

use axtb_gfnff::{Calculator, Conventions, Options, Topology};
use serde_json::Value;
use std::path::PathBuf;

fn cases(dir: &str) -> Vec<(String, Value)> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join(dir);
    let mut out: Vec<(String, Value)> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.extension().map(|e| e == "json").unwrap_or(false))
        .map(|p| {
            let v: Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap()).unwrap();
            (p.file_stem().unwrap().to_string_lossy().into_owned(), v)
        })
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

fn positions(v: &Value) -> Vec<[f64; 3]> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|p| {
            let a = p.as_array().unwrap();
            [a[0].as_f64().unwrap(), a[1].as_f64().unwrap(), a[2].as_f64().unwrap()]
        })
        .collect()
}

fn ints2(v: &Value) -> Vec<Vec<i64>> {
    v.as_array().unwrap().iter().map(|r| r.as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect()).collect()
}

fn reals(v: &Value) -> Vec<f64> {
    let mut out = Vec::new();
    fn walk(v: &Value, out: &mut Vec<f64>) {
        match v {
            Value::Array(a) => a.iter().for_each(|x| walk(x, out)),
            Value::Number(n) => out.push(n.as_f64().unwrap()),
            _ => {}
        }
    }
    walk(v, &mut out);
    out
}

fn maxdiff(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

/// Topology lists of axtb in xtb's 1-based layout.
fn ours_topology(t: &Topology) -> serde_json::Map<String, Value> {
    let mut m = serde_json::Map::new();
    let v = |x: Value| x;
    m.insert(
        "nb".into(),
        v(serde_json::json!(t
            .nb
            .iter()
            .map(|l| {
                let mut l: Vec<usize> = l.iter().map(|k| k + 1).collect();
                l.sort();
                l
            })
            .collect::<Vec<_>>())),
    );
    m.insert("blist".into(), serde_json::json!(t.bonds.iter().map(|b| [b.j + 1, b.i + 1]).collect::<Vec<_>>()));
    m.insert(
        "alist".into(),
        serde_json::json!(t.angles.iter().map(|a| [a.i + 1, a.j + 1, a.k + 1]).collect::<Vec<_>>()),
    );
    m.insert(
        "tlist".into(),
        serde_json::json!(t
            .torsions
            .iter()
            .map(|q| [
                q.atoms[0] as i64 + 1,
                q.atoms[1] as i64 + 1,
                q.atoms[2] as i64 + 1,
                q.atoms[3] as i64 + 1,
                q.nrot as i64
            ])
            .collect::<Vec<_>>()),
    );
    m.insert(
        "b3list".into(),
        serde_json::json!(t.batm.iter().map(|b| [b[0] + 1, b[1] + 1, b[2] + 1]).collect::<Vec<_>>()),
    );
    m.insert("fraglist".into(), serde_json::json!(t.fraglist));
    m.insert("hbatHl".into(), serde_json::json!(t.hb_h.iter().map(|h| h + 1).collect::<Vec<_>>()));
    m.insert("nr_hb".into(), serde_json::json!(t.bonds.iter().map(|b| b.nr_hb).collect::<Vec<_>>()));
    m
}

#[test]
fn reproduces_xtb_680() {
    reproduces("reference-xtb-6.8.0", Conventions::Xtb680, "xtb 6.8.0");
}

#[test]
fn reproduces_xtb_671() {
    reproduces("reference-xtb-6.7.1", Conventions::Xtb671, "xtb 6.7.1");
}

fn reproduces(dir: &str, conventions: Conventions, program: &str) {
    let mut worst = (0.0f64, 0.0f64, 0.0f64, 0.0f64);
    let cs = cases(dir);
    assert!(cs.len() >= 20, "reference data missing");
    for (name, c) in &cs {
        assert!(c["program"].as_str().unwrap().starts_with(program), "{name}: {}", c["program"]);
        let numbers: Vec<usize> =
            c["numbers"].as_array().unwrap().iter().map(|z| z.as_u64().unwrap() as usize).collect();
        let pos = positions(&c["positions"]);
        let charge = c["charge"].as_f64().unwrap();
        let eval_pos = if c["positions_eval"].is_null() { pos.clone() } else { positions(&c["positions_eval"]) };
        let calc = Calculator::new(&numbers, &pos, charge, Options { conventions, ..Default::default() }).unwrap();
        let res = calc.singlepoint(&eval_pos, true).unwrap();
        // topology: integer lists identical
        let t = calc.topology();
        let ours = ours_topology(t);
        let rt = &c["topology"];
        for key in ["blist", "alist", "tlist", "b3list"] {
            let a = ints2(&rt[key]);
            let b = ints2(&ours[key]);
            assert_eq!(a, b, "{name}: {key}");
        }
        assert_eq!(rt["nb"], ours["nb"], "{name}: nb");
        assert_eq!(rt["fraglist"], ours["fraglist"], "{name}: fraglist");
        assert_eq!(rt["hbatHl"], ours["hbatHl"], "{name}: hbatHl");
        assert_eq!(rt["nr_hb"], ours["nr_hb"], "{name}: nr_hb");
        let bond_hb: Vec<Value> = t
            .bond_hb
            .iter()
            .map(
                |e| serde_json::json!({"a": e.a + 1, "h": e.h + 1, "b": e.b.iter().map(|b| b + 1).collect::<Vec<_>>()}),
            )
            .collect();
        assert_eq!(rt["bond_hb"], Value::Array(bond_hb), "{name}: bond_hb");
        // topology: real parameters (xtb's LAPACK vs ours: last bits)
        let dq = maxdiff(&reals(&rt["qa"]), &t.qa);
        assert!(dq < 1e-10, "{name}: topology charges differ by {dq:e}");
        for (key, ours) in [
            ("chieeq", &t.chieeq),
            ("gameeq", &t.gameeq),
            ("alpeeq", &t.alpeeq),
            ("hbbas", &t.hbbas),
            ("hbaci", &t.hbaci),
        ] {
            let d = maxdiff(&reals(&rt[key]), ours);
            assert!(d < 1e-10, "{name}: {key} differs by {d:e}");
        }
        let vb: Vec<f64> = t.bonds.iter().flat_map(|b| b.vbond).collect();
        let va: Vec<f64> = t.angles.iter().flat_map(|a| [a.phi0, a.fc]).collect();
        let vt: Vec<f64> = t.torsions.iter().flat_map(|q| [q.phi0, q.fc]).collect();
        for (key, ours) in [("vbond", &vb), ("vangl", &va), ("vtors", &vt)] {
            let d = maxdiff(&reals(&rt[key]), ours);
            assert!(d < 1e-10, "{name}: {key} differs by {d:e}");
        }
        // charges, energies, gradient
        let dqe = maxdiff(&reals(&c["charges"]), &res.charges);
        assert!(dqe < 1e-10, "{name}: EEQ charges differ by {dqe:e}");
        let terms = c["terms"].as_object().unwrap();
        let named: std::collections::HashMap<&str, f64> = res.energies.named().into_iter().collect();
        let mut de: f64 = 0.0;
        for (k, v) in terms {
            let ours = if k == "total" { res.energy } else { named[k.as_str()] };
            de = de.max((v.as_f64().unwrap() - ours).abs());
        }
        assert!(de < 1e-10, "{name}: energy terms differ by {de:e}");
        let g: Vec<f64> = res.gradient.as_ref().unwrap().iter().flatten().copied().collect();
        let dg = maxdiff(&reals(&c["gradient"]), &g);
        assert!(dg < 1e-9, "{name}: gradient differs by {dg:e}");
        worst = (worst.0.max(dq.max(dqe)), worst.1.max(de), worst.2.max(dg), 0.0);
    }
    eprintln!(
        "{program}: {} cases: max |dq| {:.1e}, max |dE| {:.1e} Eh, max |dg| {:.1e} Eh/bohr",
        cs.len(),
        worst.0,
        worst.1,
        worst.2
    );
}
