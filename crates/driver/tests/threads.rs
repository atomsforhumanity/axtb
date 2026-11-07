//! Every calculation runs on its own thread count: nothing may use faer's global parallelism,
//! which is every CPU unless a program changes it. With it disabled, faer panics on any use.
//! In a file of its own, since the setting is per process.

use axtb_driver::{run, Request};

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
        let out = run(req);
        assert!(out.is_ok(), "{:?} {:?}: {:?}", req.method, req.analysis_conventions, out.err());
    }
}
