//! Every calculation runs on its own thread count: nothing may use faer's global parallelism,
//! which is every CPU unless a program changes it. With it disabled, faer panics on any use.
//! In a file of its own, since the setting is per process.

use axtb_driver::{run, task_threads, Calculator, Request, Task};

#[test]
fn no_calculation_uses_faers_global_parallelism() {
    faer::disable_global_parallelism();
    let water = Request::new(vec![8, 1, 1], vec![[0.0, 0.0, 0.22], [0.0, 1.43, -0.89], [0.0, -1.43, -0.89]]);
    let hcn = Request::new(vec![1, 6, 7], vec![[0.0, 0.0, -2.01], [0.0, 0.0, 0.0], [0.0, 0.0, 2.18]]);
    let mut requests = Vec::new();
    for method in ["gfn2", "gxtb-v2.0.1", "gxtb-v1.1", "gfnff"] {
        for analysis in ["axtb", "xtb", "xtb671"] {
            for mol in [&water, &hcn] {
                requests.push(Request {
                    method: Some(method.into()),
                    gradient: true,
                    hessian: true,
                    analysis_conventions: Some(analysis.into()),
                    threads: Some(2),
                    ..mol.clone()
                });
            }
        }
    }
    let solvated =
        Request { gradient: true, hessian: true, threads: Some(2), alpb: Some("water".into()), ..water.clone() };
    requests.push(Request { method: Some("gfn2".into()), ..solvated.clone() });
    requests.push(Request {
        method: Some("gxtb-v2.0.1".into()),
        solvation_correction: Some("gfn2".into()),
        ..solvated.clone()
    });
    requests.push(Request { method: Some("gfn2".into()), molden: true, bond_orders: true, ..water.clone() });
    for req in &requests {
        // on the requested threads (`run` would take one thread for these small molecules)
        let task = Task::from_request(req).unwrap();
        let out = Calculator::new(req).and_then(|mut c| c.compute(&req.positions, &task));
        assert!(out.is_ok(), "{:?} {:?}: {:?}", req.method, req.analysis_conventions, out.err());
    }
}

/// A one-off task of a tiny molecule runs on one thread, with the same results; larger
/// molecules and g-xTB's Hessians keep the requested threads; zero threads stays the input error.
#[test]
fn tiny_one_off_tasks_run_on_one_thread() {
    let water = Request::new(vec![8, 1, 1], vec![[0.0, 0.0, 0.22], [0.0, 1.43, -0.89], [0.0, -1.43, -0.89]]);
    let threads = |method: &str, hessian: bool, threads: usize| {
        let req = Request { method: Some(method.into()), hessian, threads: Some(threads), ..water.clone() };
        task_threads(&req, &Task::from_request(&req).unwrap())
    };
    for method in ["gfn2", "gxtb-v2.0.1", "gxtb-v1.1", "gfnff"] {
        assert_eq!(threads(method, false, 8), 1, "{method}");
    }
    assert_eq!(threads("gfn2", true, 8), 1);
    assert_eq!(threads("gfnff", true, 8), 1);
    assert_eq!(threads("gxtb-v2.0.1", true, 8), 8);
    assert_eq!(threads("gxtb-v1.1", true, 8), 8);
    assert_eq!(threads("gfn2", false, 0), 0);
    let big = Request { numbers: vec![1; 12], positions: vec![[0.0; 3]; 12], threads: Some(8), ..water.clone() };
    assert_eq!(task_threads(&big, &Task::from_request(&big).unwrap()), 8);
    let mut out = Vec::new();
    for t in [1, 8] {
        let req = Request { method: Some("gxtb-v2.0.1".into()), gradient: true, threads: Some(t), ..water.clone() };
        let mut r = run(&req).unwrap();
        r.seconds = None;
        r.timings = None;
        out.push(serde_json::to_string(&r).unwrap());
    }
    assert_eq!(out[0], out[1]);
}
