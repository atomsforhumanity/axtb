// SPDX-License-Identifier: LGPL-3.0-or-later
//! Analytic derivatives against central finite differences of the energy:
//! nuclear gradient at fixed charges, and the potential dE/dq.

mod common;

use axtb_solvation::{Model, ReferenceState, Solvation};

/// Total solvation energy at fixed charges, from the atom-resolved API.
fn total(solv: &Solvation, pos: &[[f64; 3]], q: &[f64]) -> f64 {
    let cache = solv.update(pos);
    let mut e = vec![0.0; q.len()];
    solv.fixed_energy(&cache, &mut e, None);
    solv.energy(&cache, q, &mut e);
    e.iter().sum()
}

fn analytic_gradient(solv: &Solvation, pos: &[[f64; 3]], q: &[f64]) -> Vec<[f64; 3]> {
    let cache = solv.update(pos);
    let mut g = vec![[0.0; 3]; q.len()];
    let mut e = vec![0.0; q.len()];
    solv.fixed_energy(&cache, &mut e, Some(&mut g));
    solv.gradient(&cache, q, &mut g);
    g
}

/// Returns (max |analytic - numerical|, max |analytic|, |sum of gradient|).
fn check_gradient(solv: &Solvation, pos: &[[f64; 3]], q: &[f64]) -> (f64, f64, f64) {
    let g = analytic_gradient(solv, pos, q);
    let h = 1.0e-5;
    let mut maxdev = 0.0f64;
    let mut maxg = 0.0f64;
    let mut p = pos.to_vec();
    for i in 0..pos.len() {
        for k in 0..3 {
            p[i][k] = pos[i][k] + h;
            let ep = total(solv, &p, q);
            p[i][k] = pos[i][k] - h;
            let em = total(solv, &p, q);
            p[i][k] = pos[i][k];
            let num = (ep - em) / (2.0 * h);
            maxdev = maxdev.max((num - g[i][k]).abs());
            maxg = maxg.max(g[i][k].abs());
        }
    }
    let mut s = [0.0f64; 3];
    for gi in &g {
        for k in 0..3 {
            s[k] += gi[k];
        }
    }
    let sumg = (s[0] * s[0] + s[1] * s[1] + s[2] * s[2]).sqrt();
    (maxdev, maxg, sumg)
}

fn check_potential(solv: &Solvation, pos: &[[f64; 3]], q: &[f64]) -> f64 {
    let cache = solv.update(pos);
    let mut v = vec![0.0; q.len()];
    solv.potential(&cache, q, &mut v);
    let h = 1.0e-4;
    let mut qq = q.to_vec();
    let mut maxdev = 0.0f64;
    let e = |qq: &[f64]| {
        let mut e = vec![0.0; qq.len()];
        solv.energy(&cache, qq, &mut e);
        e.iter().sum::<f64>()
    };
    for i in 0..q.len() {
        qq[i] = q[i] + h;
        let ep = e(&qq);
        qq[i] = q[i] - h;
        let em = e(&qq);
        qq[i] = q[i];
        maxdev = maxdev.max(((ep - em) / (2.0 * h) - v[i]).abs());
    }
    maxdev
}

const CASES: &[&str] = &[
    "water_alpb_water",
    "caffeine_alpb_water",
    "caffeine_gbsa_water",
    "zncl4_alpb_water",
    "zncl4_gbsa_water",
    "cu-nh3-4_alpb_water",
    "cu-nh3-4_gbsa_water",
    "cf3br-benzene_alpb_water",
    "beni-acetone_alpb_water",
    "ferrocene_alpb_water",
    "mb33_alpb_water",
    "s229-cl_alpb_toluene_reference",
    "tyr-xab_alpb_dmso_gsolv",
    "caffeine_alpb_n-hexane",
    "taxol_alpb_water",
];

#[test]
fn gradient_matches_finite_differences() {
    let mut worst = 0.0f64;
    for name in CASES {
        let c = common::load(name);
        let solv = Solvation::new(&c.numbers, &c.solvent, c.model, c.state).unwrap();
        let (dev, maxg, sumg) = check_gradient(&solv, &c.positions, &c.charges);
        println!("{name:<34} nat {:>4}  max|g| {maxg:.3e}  max|g-fd| {dev:.2e}  |sum g| {sumg:.1e}", c.numbers.len());
        assert!(dev < 1.0e-8, "{name}: gradient deviates from finite differences by {dev:e}");
        assert!(sumg < 1.0e-12, "{name}: gradient not translation invariant ({sumg:e})");
        worst = worst.max(dev);
    }
    println!("worst gradient deviation {worst:.2e}");
}

#[test]
fn potential_matches_finite_differences() {
    for name in CASES {
        let c = common::load(name);
        let solv = Solvation::new(&c.numbers, &c.solvent, c.model, c.state).unwrap();
        let dev = check_potential(&solv, &c.positions, &c.charges);
        println!("{name:<34} max|dE/dq - fd| {dev:.2e}");
        assert!(dev < 1.0e-10, "{name}: potential deviates from finite differences by {dev:e}");
    }
}

/// The energy is invariant under rotation and translation (up to the SASA
/// grid, which is fixed in space: only translations are exact).
#[test]
fn translation_invariance() {
    let c = common::load("caffeine_alpb_water");
    let solv = Solvation::new(&c.numbers, &c.solvent, c.model, c.state).unwrap();
    let e0 = total(&solv, &c.positions, &c.charges);
    let shifted: Vec<[f64; 3]> = c.positions.iter().map(|p| [p[0] + 1.25, p[1] - 0.5, p[2] + 3.0]).collect();
    let e1 = total(&solv, &shifted, &c.charges);
    assert!((e0 - e1).abs() < 1e-13, "{e0} {e1}");
}

#[test]
fn update_into_reuses_buffers() {
    let c = common::load("caffeine_alpb_water");
    let solv = Solvation::new(&c.numbers, &c.solvent, c.model, c.state).unwrap();
    let big = common::load("taxol_alpb_water");
    let solv_big = Solvation::new(&big.numbers, &big.solvent, big.model, big.state).unwrap();
    // a cache used for another (larger) molecule first must give identical results
    let mut cache = solv_big.update(&big.positions);
    solv.update_into(&c.positions, &mut cache);
    let fresh = solv.update(&c.positions);
    let a = solv.energy_parts(&cache, &c.charges);
    let b = solv.energy_parts(&fresh, &c.charges);
    assert_eq!(a, b);
}

#[test]
fn errors_and_aliases() {
    assert!(Solvation::new(&[8, 1, 1], "aniline", Model::Gbsa, ReferenceState::Gsolv).is_err());
    assert!(Solvation::new(&[8, 1, 1], "Aniline", Model::Alpb, ReferenceState::Gsolv).is_ok());
    assert!(Solvation::new(&[95], "water", Model::Alpb, ReferenceState::Gsolv).is_err());
    let a = Solvation::new(&[8, 1, 1], "dcm", Model::Alpb, ReferenceState::Gsolv).unwrap();
    assert_eq!(a.parameter_set(), "gfn2_alpb_ch2cl2");
    let b = Solvation::new(&[8, 1, 1], "n-hexane", Model::Gbsa, ReferenceState::Gsolv).unwrap();
    assert_eq!(b.parameter_set(), "gfn2_nhexan");
    let names = axtb_solvation::solvent_names(Model::Alpb);
    assert!(names.contains(&"water") && names.contains(&"h2o") && names.contains(&"woctanol"));
}

/// Structures hitting the stale-r1 branch of xtb 6.7.1's Born-radius integral: the
/// energy matches xtb 6.7.1 (see xtb_reference.rs), and so does the gradient
/// formula, but its gradient is not the derivative of its energy there. xtb 6.8.0
/// fixed the stale `r1`, and its gradient is consistent.
#[test]
fn stale_r1_branch_gradient_is_inconsistent_like_xtb() {
    use axtb_solvation::{Conventions, SolvOptions};
    for name in ["hgh2-compressed_alpb_water", "csoh-compressed_alpb_water"] {
        let c = common::load(name);
        for (conventions, stale) in [(Conventions::Xtb671, true), (Conventions::Xtb, false)] {
            let opts = SolvOptions { conventions, ..SolvOptions::default() };
            let solv = Solvation::with_options(&c.numbers, &c.solvent, c.model, c.state, &opts).unwrap();
            let (dev, maxg, _) = check_gradient(&solv, &c.positions, &c.charges);
            println!("{name:<34} {conventions:?} max|g| {maxg:.3e}  max|g-fd| {dev:.2e}");
            if stale {
                // fails if the stale-r1 replication is removed from born.rs
                assert!(dev > 1.0e-8, "{name}: expected xtb 6.7.1's inconsistent gradient");
            } else {
                assert!(dev < 1.0e-8, "{name}: xtb 6.8.0 gradient off finite differences by {dev:e}");
            }
        }
    }
}

/// Corrected conventions: the solvation energy of compressed CsOH must not depend on the
/// atom order (xtb's stale `r1`, grimme-lab/xtb#1504, makes it differ by ~2.9e-3 Eh).
#[test]
fn corrected_is_atom_order_invariant() {
    use axtb_solvation::{Conventions, Model, ReferenceState, SolvOptions, Solvation};
    let aatoau = 1.0 / 0.529177210544;
    let geom = |order: &[usize]| -> (Vec<usize>, Vec<[f64; 3]>) {
        let atoms = [(1usize, [0.0, 0.0, -0.96]), (8, [0.0, 0.0, 0.0]), (55, [0.0, 0.0, 2.08])];
        let num = order.iter().map(|&i| atoms[i].0).collect();
        let pos = order.iter().map(|&i| atoms[i].1.map(|x| x * aatoau)).collect();
        (num, pos)
    };
    let energy = |order: &[usize], conventions: Conventions| -> f64 {
        let (num, pos) = geom(order);
        let opts = SolvOptions { conventions, ..SolvOptions::default() };
        let s = Solvation::with_options(&num, "water", Model::Alpb, ReferenceState::Gsolv, &opts).unwrap();
        let c = s.update(&pos);
        // fixed charges in the same physical assignment for both orders
        let q: Vec<f64> = num
            .iter()
            .map(|&z| match z {
                1 => 0.35,
                8 => -1.05,
                _ => 0.70,
            })
            .collect();
        let mut e = vec![0.0; num.len()];
        s.energy(&c, &q, &mut e);
        s.fixed_energy(&c, &mut e, None);
        e.iter().sum()
    };
    let (a, b) = (energy(&[0, 1, 2], Conventions::Corrected), energy(&[2, 1, 0], Conventions::Corrected));
    assert!((a - b).abs() < 1e-12, "corrected: {a} vs {b}");
    // xtb 6.8.0 fixed the stale r1 (born.f90 sets r1 for every pair)
    let (xa, xb) = (energy(&[0, 1, 2], Conventions::Xtb), energy(&[2, 1, 0], Conventions::Xtb));
    assert!((xa - xb).abs() < 1e-12, "xtb 6.8.0: {xa} vs {xb}");
    let (xa, xb) = (energy(&[0, 1, 2], Conventions::Xtb671), energy(&[2, 1, 0], Conventions::Xtb671));
    assert!((xa - xb).abs() > 1e-4, "xtb 6.7.1 mode should reproduce xtb's order dependence: {xa} vs {xb}");
}
