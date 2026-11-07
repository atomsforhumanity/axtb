//! `threads = 1` must not create any thread (in particular not rayon's global pool, which
//! faer's high-level solvers would start at the machine's full size), and `threads = n > 1`
//! must stay inside the calculator's own pool. Results repeat bit for bit on a given thread
//! count; between thread counts they may differ by round-off (the eigensolver's parallel
//! tridiagonalization, above ~313 rows; this small molecule stays below it).
//!
//! Own test binary with a single test: thread counts and the use of rayon's global pool are
//! process-wide.

use axtb_gxtb::{Calculator, Conventions, ScfOptions, Version};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;

/// Largest absolute difference of two equally long sequences.
fn maxdiff<'a>(a: impl IntoIterator<Item = &'a f64>, b: impl IntoIterator<Item = &'a f64>) -> f64 {
    a.into_iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

fn os_threads() -> usize {
    std::fs::read_dir("/proc/self/task").map(|d| d.count()).unwrap_or(0)
}

#[test]
fn threads_one_spawns_nothing() {
    // CuCl4^2- doublet: EEQ-BC solve, unrestricted SCF with Pulay mixing, revD4, gradient
    // and Hessian (v1.1 and 2.0.1)
    let numbers = [29, 17, 17, 17, 17];
    let a = 2.26 / axtb_gxtb::BOHR_ANGSTROM;
    let pos = [[0.0, 0.0, 0.0], [a, 0.0, 0.0], [-a, 0.0, 0.0], [0.0, a, 0.0], [0.0, -a, 0.0]];

    // a monitor samples the number of OS threads while the calculation runs
    let stop = Arc::new(AtomicBool::new(false));
    let peak = Arc::new(AtomicUsize::new(0));
    let monitor = {
        let (stop, peak) = (stop.clone(), peak.clone());
        std::thread::spawn(move || {
            while !stop.load(Ordering::Relaxed) {
                peak.fetch_max(os_threads(), Ordering::Relaxed);
                std::thread::yield_now();
            }
        })
    };
    std::thread::sleep(std::time::Duration::from_millis(20));
    let baseline = os_threads();
    let mut calc =
        Calculator::new(axtb_gxtb::Version::V1_1, &numbers, &pos, -2.0, Some(1), ScfOptions::default()).unwrap();
    let e1 = calc.singlepoint(&pos, false).unwrap().singlepoint.energy;
    // gradient (Z-vector response) and a few Hessian columns' worth of displacements
    let g1 = calc.singlepoint(&pos, true).unwrap().singlepoint.gradient.unwrap();
    let h1 = calc.numerical_hessian(&pos, 0.005, None).unwrap();
    drop(calc);
    // program conventions: Z-vector response
    let program = ScfOptions { conventions: Conventions::Gxtb, ..Default::default() };
    let mut calc = Calculator::new(axtb_gxtb::Version::V1_1, &numbers, &pos, -2.0, Some(1), program.clone()).unwrap();
    let gp1 = calc.singlepoint(&pos, true).unwrap().singlepoint.gradient.unwrap();
    drop(calc);
    // ... and at 3000 K (Fermi smearing; Z-vector with fractional occupations)
    let hot = ScfOptions { etemp: 3000.0, ..program.clone() };
    let mut calc = Calculator::new(axtb_gxtb::Version::V1_1, &numbers, &pos, -2.0, Some(1), hot.clone()).unwrap();
    let gt1 = calc.singlepoint(&pos, true).unwrap().singlepoint.gradient.unwrap();
    drop(calc);
    // g-xTB 2.0.1 in both conventions: energy, gradient, numerical, response and analytic Hessians
    let v2 = |threads: usize, conventions: Conventions, etemp: f64| {
        let o = ScfOptions { threads, conventions, etemp, ..Default::default() };
        let mut calc = Calculator::new(Version::V2_0_1, &numbers, &pos, -2.0, Some(1), o).unwrap();
        let e = calc.singlepoint(&pos, false).unwrap().singlepoint.energy;
        let g = calc.singlepoint(&pos, true).unwrap().singlepoint.gradient.unwrap();
        let h = calc.numerical_hessian(&pos, 0.005, None).unwrap().hessian;
        let hopts = axtb::HessianOptions { method: axtb::HessianMethod::Response, step: None, accuracy: None };
        let hr = calc.hessian(&pos, &hopts).unwrap().hessian;
        let hopts = axtb::HessianOptions { method: axtb::HessianMethod::Analytic, step: None, accuracy: None };
        let ha = calc.hessian(&pos, &hopts).unwrap().hessian;
        (e, g, h, hr, ha)
    };
    // 0 K and 3000 K (Fermi smearing, occupation response in the Hessians)
    let v2_1 = [
        v2(1, Conventions::Axtb, 0.0),
        v2(1, Conventions::Gxtb, 0.0),
        v2(1, Conventions::Axtb, 3000.0),
        v2(1, Conventions::Gxtb, 3000.0),
    ];
    stop.store(true, Ordering::Relaxed);
    monitor.join().unwrap();
    assert!(
        peak.load(Ordering::Relaxed) <= baseline,
        "threads=1 spawned threads: peak {} > {baseline}",
        peak.load(Ordering::Relaxed)
    );
    // the kernel can list the monitor's task for a moment after join returns
    let gone = (0..200).any(|_| {
        let below = os_threads() < baseline;
        if !below {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        below
    });
    assert!(gone, "monitor gone, nothing else left: {} threads, baseline {baseline}", os_threads());

    // two threads, inside the calculator's pool: the same results to round-off
    let g = |c: &mut Calculator| c.singlepoint(&pos, true).unwrap().singlepoint.gradient.unwrap();
    let mut calc = Calculator::new(
        axtb_gxtb::Version::V1_1,
        &numbers,
        &pos,
        -2.0,
        Some(1),
        ScfOptions { threads: 2, ..Default::default() },
    )
    .unwrap();
    let e2 = calc.singlepoint(&pos, false).unwrap().singlepoint.energy;
    assert!((e1 - e2).abs() < 1e-11);
    assert!(maxdiff(g(&mut calc).as_flattened(), g1.as_flattened()) < 1e-10);
    assert!(maxdiff(&calc.numerical_hessian(&pos, 0.005, None).unwrap().hessian, &h1.hessian) < 1e-8);
    drop(calc);
    let mut calc =
        Calculator::new(axtb_gxtb::Version::V1_1, &numbers, &pos, -2.0, Some(1), ScfOptions { threads: 2, ..program })
            .unwrap();
    assert!(maxdiff(g(&mut calc).as_flattened(), gp1.as_flattened()) < 1e-10);
    drop(calc);
    let mut calc =
        Calculator::new(axtb_gxtb::Version::V1_1, &numbers, &pos, -2.0, Some(1), ScfOptions { threads: 2, ..hot })
            .unwrap();
    assert!(maxdiff(g(&mut calc).as_flattened(), gt1.as_flattened()) < 1e-10);
    drop(calc);
    // 2.0.1 on two threads: the same results to round-off, and bit for bit on a repeat
    let cases =
        [(Conventions::Axtb, 0.0), (Conventions::Gxtb, 0.0), (Conventions::Axtb, 3000.0), (Conventions::Gxtb, 3000.0)];
    for (k, &(conventions, etemp)) in cases.iter().enumerate() {
        let (e, gr, h, hr, ha) = v2(2, conventions, etemp);
        let r = &v2_1[k];
        assert!((e - r.0).abs() < 1e-11, "{conventions:?} {etemp} K: energy");
        assert!(maxdiff(gr.as_flattened(), r.1.as_flattened()) < 1e-10, "{conventions:?} {etemp} K: gradient");
        assert!(
            maxdiff(&h, &r.2) < 1e-8 && maxdiff(&hr, &r.3) < 1e-8 && maxdiff(&ha, &r.4) < 1e-8,
            "{conventions:?} {etemp} K: Hessians"
        );
    }
    assert_eq!(v2(2, Conventions::Axtb, 0.0), v2(2, Conventions::Axtb, 0.0));
    assert!(v2_1[2].0 != v2_1[0].0, "smearing changes the energy");
    assert!(rayon::ThreadPoolBuilder::new().num_threads(1).build_global().is_ok(), "rayon's global pool was used");
}
