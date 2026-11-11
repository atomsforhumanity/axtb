//! One analytic g-xTB 2.0.1 Hessian with a chosen response tolerance and with or without the
//! symmetric response terms ([`AnalyticOptions::symmetric`]), for
//! `tests/gxtb2_hessian_symmetric.py`, which compares the settings.
//!
//! ```text
//! cargo run --release -p axtb-gxtb --example hessian_symmetric -- \
//!     MOL.xyz CHARGE UHF ETEMP CONVENTIONS(axtb|gxtb) TOL SYMMETRIC(0|1) THREADS
//! ```
//!
//! Prints JSON: the symmetrized Hessian (Eh/bohr², row-major), its largest asymmetry and the
//! analytic Hessian's phases (s). The reference SCF is converged tightly (1e-12 Eh, residual
//! 1e-11) and is not timed.

use axtb_gxtb::v2::hessian::{symmetrize, AnalyticOptions};
use axtb_gxtb::v2::response::CpksOptions;
use axtb_gxtb::v2::scf::Setup;
use axtb_gxtb::{Conventions, ScfOptions};

fn main() {
    let a: Vec<String> = std::env::args().collect();
    if a.len() != 9 {
        eprintln!("usage: hessian_symmetric MOL.xyz CHARGE UHF ETEMP CONVENTIONS TOL SYMMETRIC THREADS");
        std::process::exit(2);
    }
    let xyz = std::fs::read_to_string(&a[1]).expect("read the xyz file");
    let charge: f64 = a[2].parse().expect("charge");
    let uhf: usize = a[3].parse().expect("unpaired electrons");
    let etemp: f64 = a[4].parse().expect("electronic temperature");
    let conv = match a[5].as_str() {
        "gxtb" => Conventions::Gxtb,
        _ => Conventions::Axtb,
    };
    let tol: f64 = a[6].parse().expect("tolerance");
    let symmetric = a[7] == "1";
    let threads: usize = a[8].parse().expect("threads");
    let mol = axtb_gxtb::v2::read_xyz(&xyz, charge, Some(uhf), conv).expect("molecule");
    let mut setup = Setup::with_threads(&mol, conv, threads).expect("setup");
    let opts = ScfOptions {
        conventions: conv,
        etemp,
        energy_tol: Some(1e-12),
        residual_tol: Some(1e-11),
        max_iter: 500,
        keep_orbitals: true,
        threads,
        ..Default::default()
    };
    let r = setup.scf(&opts).expect("scf");
    let ao = AnalyticOptions { cpks: CpksOptions { tol, max_iter: 500 }, symmetric, ..Default::default() };
    let (mut h, _, timings) = setup.analytic_hessian(&r, &ao).expect("hessian");
    let n3 = 3 * mol.nat();
    let asym = symmetrize(&mut h, n3);
    let phases: Vec<String> = timings.iter().map(|(k, v)| format!("\"{k}\": {v}")).collect();
    let hs: Vec<String> = h.iter().map(|v| format!("{v:e}")).collect();
    println!(
        "{{\"n3\": {n3}, \"max_asymmetry\": {asym:e}, \"phases\": {{{}}}, \"hessian\": [{}]}}",
        phases.join(", "),
        hs.join(", ")
    );
}
