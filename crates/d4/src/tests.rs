// SPDX-License-Identifier: LGPL-3.0-or-later
//! Internal consistency tests: finite differences of every analytic derivative.

use super::*;

/// Water, benzene-like fragment and heavy atoms, positions in bohr.
fn mixed_molecule() -> (Vec<usize>, Vec<[f64; 3]>) {
    let numbers = vec![8, 1, 1, 6, 6, 1, 1, 1, 26, 17, 53, 82, 86, 57];
    let positions = vec![
        [0.0, 0.0, 0.0],
        [1.81, 0.0, 0.0],
        [-0.45, 1.75, 0.1],
        [5.0, 1.0, -0.5],
        [7.6, 1.2, 0.2],
        [4.3, -0.9, -0.2],
        [4.4, 2.3, 1.0],
        [8.4, 2.9, -0.6],
        [9.2, -1.6, 1.4],
        [12.9, -1.0, 2.8],
        [8.0, -5.6, 2.5],
        [3.1, -5.2, 4.9],
        [-2.0, -6.0, -3.0],
        [14.5, 4.0, -1.7],
    ];
    (numbers, positions)
}

/// 27 water molecules on a 3 x 3 x 3 grid (5.5 bohr apart: every pair inside the three-body
/// cutoffs, so an atom's neighbours span more than one block of 64), and far away a CO
/// molecule (a pair whose neighbour windows do not overlap) and a lone Ne atom.
fn large_molecule() -> (Vec<usize>, Vec<[f64; 3]>) {
    let (mut numbers, mut positions) = (Vec::new(), Vec::new());
    for i in 0..27 {
        let c = [(i % 3) as f64 * 5.5, (i / 3 % 3) as f64 * 5.5, (i / 9) as f64 * 5.5];
        numbers.extend([8, 1, 1]);
        positions.extend([c, [c[0] + 1.81, c[1], c[2]], [c[0] - 0.45, c[1] + 1.75, c[2] + 0.1]]);
    }
    numbers.extend([6, 8, 10]);
    positions.extend([[200.0, 0.0, 0.0], [202.13, 0.0, 0.0], [0.0, 300.0, 0.0]]);
    (numbers, positions)
}

fn charges(n: usize) -> Vec<f64> {
    // deterministic, some negative enough to hit q + zeff < 0 for nothing but
    // keeping derivatives smooth
    (0..n).map(|i| 0.37 * ((i as f64) * 1.7).sin() - 0.05).collect()
}

const H: f64 = 1e-3;

/// 5-point central difference
fn fd5(f: &mut dyn FnMut(f64) -> f64, h: f64) -> f64 {
    (f(-2.0 * h) - 8.0 * f(-h) + 8.0 * f(h) - f(2.0 * h)) / (12.0 * h)
}

fn max_abs_diff(a: &[f64], b: &[f64]) -> (f64, f64) {
    let mut d: f64 = 0.0;
    let mut m: f64 = 0.0;
    for (x, y) in a.iter().zip(b) {
        d = d.max((x - y).abs());
        m = m.max(y.abs());
    }
    (d, m)
}

#[test]
fn constants_match_fortran_folding() {
    assert_eq!(1.7724538509055159_f64, data::PI.sqrt());
    assert_eq!(data::PI, std::f64::consts::PI);
    // aatoau of mctc-lib
    assert_eq!(AATOAU, 1.8897261246204404); // h/(2 pi)/(me c alpha) folded in IEEE double
}

#[test]
fn unsupported_elements() {
    assert!(D4Model::new(&[1, 105], ChargeModel::Eeq).is_err());
    assert!(D4Model::new(&[0], ChargeModel::Eeq).is_err());
    assert!(D4Model::new(&[119], ChargeModel::Gfn2).is_err());
    assert!(D4Model::new(&[1, 103, 112, 118], ChargeModel::Gfn2).is_ok());
}

#[test]
fn cn_derivatives() {
    let (numbers, pos) = mixed_molecule();
    let n = numbers.len();
    let model = D4Model::new(&numbers, ChargeModel::Eeq).unwrap();
    let a: Vec<f64> = (0..n).map(|i| 1.0 + 0.3 * (i as f64).cos()).collect();
    let mut g = vec![[0.0; 3]; n];
    model.add_coordination_number_derivs(&pos, 30.0, &a, &mut g);
    let mut cn = vec![0.0; n];
    let mut gfd = vec![[0.0; 3]; n];
    for i in 0..n {
        for x in 0..3 {
            let mut f = |h: f64| {
                let mut p = pos.clone();
                p[i][x] += h;
                model.coordination_numbers(&p, 30.0, &mut cn);
                cn.iter().zip(&a).map(|(c, w)| c * w).sum::<f64>()
            };
            gfd[i][x] = fd5(&mut f, 1e-3);
        }
    }
    let (d, m) = max_abs_diff(g.as_flattened(), gfd.as_flattened());
    eprintln!("cn deriv: max |analytic - fd| = {d:.3e} (max |g| = {m:.3e})");
    assert!(d < 1e-9 * m.max(1.0));
}

#[test]
fn weight_derivatives() {
    let (numbers, pos) = mixed_molecule();
    let n = numbers.len();
    for qm in [ChargeModel::Eeq, ChargeModel::Gfn2] {
        let model = D4Model::new(&numbers, qm).unwrap();
        let mut cn = vec![0.0; n];
        model.coordination_numbers(&pos, 30.0, &mut cn);
        let q = charges(n);
        let w = model.weight_references(&cn, &q);
        let mut worst: f64 = 0.0;
        for i in 0..n {
            for r in 0..model.nref_atom(i) {
                let mut fcn = |h: f64| {
                    let mut c = cn.clone();
                    c[i] += h;
                    model.weight_references(&c, &q).gw[i][r]
                };
                let dcn = fd5(&mut fcn, 1e-4);
                let mut fq = |h: f64| {
                    let mut qq = q.clone();
                    qq[i] += h;
                    model.weight_references(&cn, &qq).gw[i][r]
                };
                let dq = fd5(&mut fq, 1e-4);
                worst = worst.max((dcn - w.dgwdcn[i][r]).abs()).max((dq - w.dgwdq[i][r]).abs());
            }
        }
        eprintln!("{qm:?}: weight derivative max error {worst:.3e}");
        assert!(worst < 1e-8);
    }
}

fn check_two_body(qm: ChargeModel, cutoff: RealspaceCutoff, numbers: &[usize], pos: &[[f64; 3]], hx: f64) {
    let n = numbers.len();
    let disp = D4Dispersion::new(D4Model::new(numbers, qm).unwrap(), RationalDamping::GFN2, cutoff);
    let q = charges(n);
    let cache = disp.update(pos);

    // potential vs FD in q
    let mut vat = vec![0.0; n];
    let mut e_atoms = vec![0.0; n];
    disp.energy_and_potential(&cache, &q, &mut e_atoms, &mut vat);
    let e_ref = disp.energy_total(&cache, &q);
    assert!((e_atoms.iter().sum::<f64>() - e_ref).abs() < 1e-15 * e_ref.abs().max(1e-300) * 10.0);
    let mut vpot = vec![0.0; n];
    disp.potential(&cache, &q, &mut vpot, false);
    let mut vfd = vec![0.0; n];
    for i in 0..n {
        let mut f = |h: f64| {
            let mut qq = q.clone();
            qq[i] += h;
            disp.energy_total(&cache, &qq)
        };
        vfd[i] = fd5(&mut f, 1e-4);
    }
    let (d, m) = max_abs_diff(&vat, &vfd);
    eprintln!("{qm:?} potential: max err {d:.3e} (max |V| {m:.3e})");
    assert!(d < 1e-9 * m.max(1e-6));
    assert_eq!(vat, vpot);

    // gradient at fixed q vs FD
    let mut g = vec![[0.0; 3]; n];
    disp.gradient(&cache, &q, &mut g);
    let mut gfd = vec![[0.0; 3]; n];
    for i in 0..n {
        for x in 0..3 {
            let mut f = |h: f64| {
                let mut p = pos.to_vec();
                p[i][x] += h;
                disp.energy_total(&disp.update(&p), &q)
            };
            gfd[i][x] = fd5(&mut f, hx);
        }
    }
    let (d, m) = max_abs_diff(g.as_flattened(), gfd.as_flattened());
    eprintln!("{qm:?} 2-body gradient: max err {d:.3e} (max |g| {m:.3e})");
    assert!(d < 1e-9 * m.max(1e-6));
}

fn check_three_body(qm: ChargeModel, cutoff: RealspaceCutoff, numbers: &[usize], pos: &[[f64; 3]], hx: f64) {
    let n = numbers.len();
    let disp = D4Dispersion::new(D4Model::new(numbers, qm).unwrap(), RationalDamping::GFN2, cutoff);
    let cache = disp.update(pos);
    let mut e = vec![0.0; n];
    let mut g = vec![[0.0; 3]; n];
    disp.three_body(&cache, &mut e, Some(&mut g), false);
    let mut e2 = vec![0.0; n];
    disp.three_body(&cache, &mut e2, None, false);
    let (d, m) = max_abs_diff(&e, &e2);
    assert!(d <= 1e-15 * m.max(1e-300) * 10.0, "energy paths differ {d:e}");
    let mut gfd = vec![[0.0; 3]; n];
    for i in 0..n {
        for x in 0..3 {
            let mut f = |h: f64| {
                let mut p = pos.to_vec();
                p[i][x] += h;
                let mut ee = vec![0.0; n];
                disp.three_body(&disp.update(&p), &mut ee, None, false);
                ee.iter().sum()
            };
            gfd[i][x] = fd5(&mut f, hx);
        }
    }
    let (d, m) = max_abs_diff(g.as_flattened(), gfd.as_flattened());
    eprintln!("{qm:?} 3-body gradient: max err {d:.3e} (max |g| {m:.3e})");
    assert!(d < 1e-9 * m.max(1e-8));
}

#[test]
fn two_body_finite_differences() {
    let (numbers, pos) = mixed_molecule();
    for qm in [ChargeModel::Eeq, ChargeModel::Gfn2] {
        check_two_body(qm, RealspaceCutoff::TBLITE_GFN2, &numbers, &pos, H);
        check_two_body(qm, RealspaceCutoff::DFTD4, &numbers, &pos, H);
    }
}

#[test]
fn three_body_finite_differences() {
    let (numbers, pos) = mixed_molecule();
    for qm in [ChargeModel::Eeq, ChargeModel::Gfn2] {
        check_three_body(qm, RealspaceCutoff::TBLITE_GFN2, &numbers, &pos, H);
    }
}

/// The three-body pair sums (xtb 6.8.0's loop) agree with the loop over the triples to
/// round-off, give the same energies with and without the gradient, and do not depend on the
/// thread count; also with more than one block of third atoms and with a pair whose
/// neighbours do not overlap ([`large_molecule`]).
#[test]
fn three_body_pair_sums() {
    let wide = RealspaceCutoff { cn: 30.0, disp2: 12.0, disp3: 9.0, width2: 6.0, width3: 5.0 };
    let axtb = RealspaceCutoff { cn: 40.0, disp2: 72.0, disp3: 40.0, width2: 1.0, width3: 1.0 };
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    let molecules = [mixed_molecule(), large_molecule()];
    for ((numbers, pos), cutoff) in
        molecules.iter().flat_map(|m| [wide, RealspaceCutoff::TBLITE_GFN2, axtb].map(|c| (m, c)))
    {
        let n = numbers.len();
        let disp = |sums: bool| {
            let model = D4Model::new(numbers, ChargeModel::Gfn2).unwrap();
            D4Dispersion::new(model, RationalDamping::GFN2, cutoff).with_pair_sums(sums)
        };
        let cache = disp(false).update(pos);
        let run = |d: &D4Dispersion, par: bool| {
            let (mut e, mut g, mut e0) = (vec![0.0; n], vec![[0.0; 3]; n], vec![0.0; n]);
            d.three_body(&cache, &mut e, Some(&mut g), par);
            d.three_body(&cache, &mut e0, None, par);
            (e, g, e0)
        };
        let triples = run(&disp(false), false);
        let sums = run(&disp(true), false);
        assert_eq!(sums.0, sums.2);
        let (d, m) = max_abs_diff(&sums.0, &triples.0);
        assert!(d < 1e-14 * m, "energies {d:e} (max {m:e})");
        let (d, m) = max_abs_diff(sums.1.as_flattened(), triples.1.as_flattened());
        assert!(d < 1e-13 * m, "gradients {d:e} (max {m:e})");
        assert_eq!(pool.install(|| run(&disp(true), true)), sums);
        assert_eq!(pool.install(|| run(&disp(false), true)), triples);
    }
}

/// The two-body energy and charge potential on several threads give the bits of the
/// sequential pair loop.
#[test]
fn two_body_parallel_bits() {
    let (numbers, pos) = mixed_molecule();
    let n = numbers.len();
    let q = charges(n);
    let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
    let run = |par: bool| {
        let disp = D4Dispersion::gfn2(&numbers).unwrap();
        let cache = disp.update(&pos);
        let (mut e, mut v) = (vec![0.1; n], vec![0.2; n]);
        disp.energy(&cache, &q, &mut e, par);
        disp.potential(&cache, &q, &mut v, par);
        (e, v)
    };
    assert_eq!(pool.install(|| run(true)), run(false));
}

/// Atoms placed so that many distances fall inside wide switching windows.
#[test]
fn smooth_cutoff_finite_differences() {
    let (numbers, pos) = mixed_molecule();
    let wide = RealspaceCutoff { cn: 30.0, disp2: 12.0, disp3: 9.0, width2: 6.0, width3: 5.0 };
    for qm in [ChargeModel::Eeq, ChargeModel::Gfn2] {
        check_two_body(qm, wide, &numbers, &pos, H);
        check_three_body(qm, wide, &numbers, &pos, H);
    }
    // a pair inside the default 0.05 bohr window of the tblite cutoffs
    let numbers = vec![18, 36, 54];
    let pos = vec![[0.0, 0.0, 0.0], [49.98, 0.0, 0.0], [12.0, 21.0, 0.0]];
    let disp = D4Dispersion::gfn2(&numbers).unwrap();
    let cache = disp.update(&pos);
    assert_eq!(cache.npairs(), 3);
    check_two_body(ChargeModel::Gfn2, RealspaceCutoff::TBLITE_GFN2, &numbers, &pos, 1e-5);
    let numbers = vec![18, 36, 54];
    let pos = vec![[0.0, 0.0, 0.0], [24.98, 0.0, 0.0], [12.0, 10.0, 0.0]];
    check_three_body(ChargeModel::Gfn2, RealspaceCutoff::TBLITE_GFN2, &numbers, &pos, 1e-5);
}

#[test]
fn convenience_matches_parts() {
    let (numbers, pos) = mixed_molecule();
    let n = numbers.len();
    let disp = D4Dispersion::gfn2(&numbers).unwrap();
    let q = charges(n);
    let mut e = vec![0.0; n];
    let mut g = vec![[0.0; 3]; n];
    let etot = disp.engrad_fixed_charges(&pos, &q, &mut e, Some(&mut g));
    let cache = disp.update(&pos);
    let mut e3 = vec![0.0; n];
    disp.three_body(&cache, &mut e3, None, false);
    let e2 = disp.energy_total(&cache, &q);
    let sum3: f64 = e3.iter().sum();
    assert!((etot - (e2 + sum3)).abs() < 1e-14 * etot.abs());
    // translational invariance of the gradient
    for x in 0..3 {
        let s: f64 = g.iter().map(|v| v[x]).sum();
        assert!(s.abs() < 1e-14, "net force {s:e}");
    }
}

/// Central-difference Jacobian of `f` (n outputs) over the 3N positions.
fn fd_positions(pos: &[[f64; 3]], n: usize, h: f64, f: &dyn Fn(&[[f64; 3]]) -> Vec<f64>) -> Vec<f64> {
    let n3 = 3 * pos.len();
    let mut out = vec![0.0; n * n3];
    let mut p = pos.to_vec();
    for x in 0..n3 {
        p[x / 3][x % 3] = pos[x / 3][x % 3] + h;
        let fp = f(&p);
        p[x / 3][x % 3] = pos[x / 3][x % 3] - h;
        let fm = f(&p);
        p[x / 3][x % 3] = pos[x / 3][x % 3];
        for i in 0..n {
            out[i * n3 + x] = (fp[i] - fm[i]) / (2.0 * h);
        }
    }
    out
}

fn second_derivative_checks(disp: &D4Dispersion) {
    let (num, pos) = mixed_molecule();
    let nat = num.len();
    let n3 = 3 * nat;
    let q = charges(nat);
    let cache = disp.update(&pos);
    let h = 1e-5;
    let m = disp.model();
    // CN Jacobian and second derivatives
    let w: Vec<f64> = (0..nat).map(|i| 0.3 + 0.1 * (i as f64).cos()).collect();
    let cut = disp.cutoff().cn;
    let jac = m.coordination_number_jacobian(&pos, cut);
    let fd = fd_positions(&pos, nat, h, &|p| {
        let mut cn = vec![0.0; nat];
        m.coordination_numbers(p, cut, &mut cn);
        cn
    });
    let (d, s) = max_abs_diff(&jac, &fd);
    assert!(d < 1e-8 * s.max(1.0), "CN jacobian {d:.3e}");
    let mut hcn = vec![0.0; n3 * n3];
    m.add_coordination_number_hessian(&pos, cut, &w, &mut hcn);
    let fd = fd_positions(&pos, n3, h, &|p| {
        let mut g = vec![[0.0; 3]; nat];
        m.add_coordination_number_derivs(p, cut, &w, &mut g);
        g.iter().flatten().copied().collect()
    });
    let (d, s) = max_abs_diff(&hcn, &fd);
    assert!(d < 1e-7 * s.max(1.0), "CN hessian {d:.3e} (max {s:.3e})");
    // charge kernel
    let mut k = vec![0.0; nat * nat];
    disp.charge_hessian(&cache, &q, &mut k);
    let mut fd = vec![0.0; nat * nat];
    for b in 0..nat {
        let mut qp = q.clone();
        qp[b] += h;
        let mut vp = vec![0.0; nat];
        disp.potential(&cache, &qp, &mut vp, false);
        qp[b] -= 2.0 * h;
        let mut vm = vec![0.0; nat];
        disp.potential(&cache, &qp, &mut vm, false);
        for a in 0..nat {
            fd[a * nat + b] = (vp[a] - vm[a]) / (2.0 * h);
        }
    }
    let (d, s) = max_abs_diff(&k, &fd);
    assert!(d < 1e-7 * s.max(1e-6), "charge kernel {d:.3e} (max {s:.3e})");
    // potential derivative
    let mut dv = vec![0.0; nat * n3];
    disp.potential_derivative(&cache, &q, &mut dv);
    let fd = fd_positions(&pos, nat, h, &|p| {
        let c = disp.update(p);
        let mut v = vec![0.0; nat];
        disp.potential(&c, &q, &mut v, false);
        v
    });
    let (d, s) = max_abs_diff(&dv, &fd);
    assert!(d < 1e-7 * s.max(1e-6), "potential derivative {d:.3e} (max {s:.3e})");
    // Hessian at fixed charges (two- and three-body)
    let mut hs = vec![0.0; n3 * n3];
    disp.hessian(&cache, &q, &mut hs, false);
    let fd = fd_positions(&pos, n3, h, &|p| {
        let c = disp.update(p);
        let mut g = vec![[0.0; 3]; nat];
        disp.gradient(&c, &q, &mut g);
        let mut e = vec![0.0; nat];
        disp.three_body(&c, &mut e, Some(&mut g), false);
        g.iter().flatten().copied().collect()
    });
    let (d, s) = max_abs_diff(&hs, &fd);
    eprintln!("D4 Hessian: max error {d:.3e} (max {s:.3e})");
    assert!(d < 1e-7 * s.max(1e-6), "hessian {d:.3e} (max {s:.3e})");
}

#[test]
fn second_derivatives_finite_differences() {
    let (num, _) = mixed_molecule();
    // smooth cutoffs inside the molecule's distance range
    let cutoff = RealspaceCutoff { cn: 30.0, disp2: 14.0, disp3: 12.0, width2: 2.0, width3: 2.0 };
    let model = D4Model::new(&num, ChargeModel::Gfn2).unwrap();
    second_derivative_checks(&D4Dispersion::new(model, RationalDamping::GFN2, cutoff));
    // hard cutoffs (xtb)
    let model = D4Model::new(&num, ChargeModel::Gfn2).unwrap();
    second_derivative_checks(&D4Dispersion::new(model, RationalDamping::GFN2, RealspaceCutoff::XTB_GFN2));
}

/// `with_reference_data` with the model's own reference systems reproduces the model
/// bitwise; a modified set changes the reference C6 as Casimir-Polder integration says.
#[test]
fn reference_data_hook() {
    let (numbers, pos) = mixed_molecule();
    let q = charges(numbers.len());
    for qm in [ChargeModel::Eeq, ChargeModel::Gfn2] {
        let model = D4Model::new(&numbers, qm).unwrap();
        let own = |z: usize| {
            let isp = model.species.iter().position(|&s| s == z).unwrap();
            let n = model.nref[isp];
            ElementReferences {
                cn: model.refcn[isp][..n].to_vec(),
                ngw: model.ngw[isp][..n].to_vec(),
                q: model.refq[isp][..n].to_vec(),
                alpha: model.aiw[isp][..n].to_vec(),
            }
        };
        let same = model.clone().with_reference_data(own);
        assert_eq!(same.c6ref, model.c6ref);
        assert_eq!(same.refcn, model.refcn);
        assert_eq!(same.ngw, model.ngw);
        assert_eq!(same.refq, model.refq);
        let d0 = D4Dispersion::new(model.clone(), RationalDamping::GFN2, RealspaceCutoff::DFTD4);
        let d1 = D4Dispersion::new(same, RationalDamping::GFN2, RealspaceCutoff::DFTD4);
        let (c0, c1) = (d0.update(&pos), d1.update(&pos));
        assert_eq!(d0.energy_total(&c0, &q), d1.energy_total(&c1, &q));
        // halve all polarizabilities of carbon: its reference C6 with any partner scale by 1/2
        let half = model.clone().with_reference_data(|z| {
            let mut r = own(z);
            if z == 6 {
                r.alpha.iter_mut().flatten().for_each(|a| *a *= 0.5);
            }
            r
        });
        let (ic, io) = (3, 0); // C and O atoms of mixed_molecule
        let (a, b) = (half.reference_c6(ic, io, 0, 1), model.reference_c6(ic, io, 0, 1));
        assert!((a - 0.5 * b).abs() <= 1e-15 * b.abs());
    }
}

/// The pair list of the cache gives the two-body energy, and the C6-matrix entry of the
/// three-body kernel reproduces the weight-based one.
#[test]
fn pair_list_and_c6_matrix_three_body() {
    let (numbers, pos) = mixed_molecule();
    let n = numbers.len();
    let q = charges(n);
    let disp = D4Dispersion::gfn2(&numbers).unwrap();
    let cache = disp.update(&pos);
    let w = disp.model().weight_references(&cache.cn, &q);
    let c6 = disp.model().atomic_c6(&w).c6;
    let e2: f64 = cache.pairs().map(|(i, j, f)| -c6[i * n + j] * f).sum();
    let e2_ref = disp.energy_total(&cache, &q);
    assert!((e2 - e2_ref).abs() < 1e-14 * e2_ref.abs(), "{e2} vs {e2_ref}");
    assert_eq!(cache.pairs().count(), cache.npairs());
    let w0 = disp.model().weight_references(&cache.cn, &vec![0.0; n]);
    let c6_0 = disp.model().atomic_c6(&w0).c6;
    let (mut e3, mut e3m) = (vec![0.0; n], vec![0.0; n]);
    disp.three_body(&cache, &mut e3, None, false);
    disp.three_body_c6(&cache, &c6_0, &mut e3m);
    assert_eq!(e3, e3m);
}

/// The C6-matrix entry with derivatives reproduces the weight-based three-body gradient once
/// the model's own CN derivative is applied to the returned dE/dCN.
#[test]
fn c6_matrix_three_body_gradient() {
    let (numbers, pos) = mixed_molecule();
    let n = numbers.len();
    let disp = D4Dispersion::gfn2(&numbers).unwrap();
    let cache = disp.update(&pos);
    let w0 = disp.model().weight_references(&cache.cn, &vec![0.0; n]);
    let c = disp.model().atomic_c6(&w0);
    let (mut e_ref, mut g_ref) = (vec![0.0; n], vec![[0.0; 3]; n]);
    disp.three_body(&cache, &mut e_ref, Some(&mut g_ref), false);
    let (mut e, mut g, mut dedcn) = (vec![0.0; n], vec![[0.0; 3]; n], vec![0.0; n]);
    disp.three_body_c6_gradient(&cache, &c.c6, &c.dc6dcn, &mut e, &mut g, &mut dedcn);
    disp.model().add_coordination_number_derivs(&pos, disp.cutoff().cn, &dedcn, &mut g);
    assert_eq!(e, e_ref);
    let (d, m) = max_abs_diff(&g.concat(), &g_ref.concat());
    assert!(d < 1e-15 * m.max(1.0), "{d:e}");
}
