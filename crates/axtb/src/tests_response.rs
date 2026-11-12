//! Validation of the coupled-perturbed SCC response and the Hessians against finite
//! differences (run with `cargo test --release -p axtb --lib response -- --nocapture`).

use crate::molecule::Molecule;
use crate::response::Response;
use crate::xtb::ChargeDependentTerm;
use crate::{Conventions, Method, Options};

pub fn mol_from(xyz: &str, charge: f64, uhf: Option<usize>) -> Molecule {
    let (num, pos) = crate::io::read_xyz(xyz, crate::units::AATOAU).unwrap();
    Molecule::new(&num, &pos, charge, uhf).unwrap()
}

pub fn tight(conv: Conventions, alpb: bool) -> Options {
    let mut o = Options { conventions: conv, ..Default::default() };
    o.scf.accuracy = 1e-9;
    if alpb {
        o.solvation = Some(crate::SolvationSpec {
            model: crate::SolvationModel::Alpb,
            solvent: "water".into(),
            state: crate::ReferenceState::Gsolv,
        });
    }
    o
}

/// δs/δR against central differences of converged SCF moments for coordinates `xs`;
/// returns the largest error relative to the largest component of each column.
pub fn check_scc_response(mol: &Molecule, o: &Options, xs: &[usize]) -> f64 {
    let (model, extra_box) = crate::build_method(Method::Gfn2, mol, o).unwrap();
    let extra: Vec<&dyn ChargeDependentTerm> = extra_box.iter().map(|b| b.as_ref()).collect();
    let (sp, st) = model.singlepoint_state(mol, &extra, &o.scf, None, false, true).unwrap();
    let st = st.unwrap();
    let resp = Response::new(&model, &extra, mol, &st, 1);
    eprintln!(
        "nao {} n_o {} n_f {} kT {:.3e} Z {:?}",
        model.basis.nao,
        resp.occ.n_o(),
        resp.n_f,
        resp.st.kt,
        resp.occ.z()
    );
    let r = resp.scc_response(false).unwrap();
    let lay = resp.lay;
    let m = lay.dim();
    let h = 1e-4;
    let guess = sp.guess();
    let mut worst: f64 = 0.0;
    for &x in xs {
        let (a, k) = (x / 3, x % 3);
        let mut s = [vec![0.0; m], vec![0.0; m]];
        for (i, sign) in [1.0, -1.0].into_iter().enumerate() {
            let mut mm = mol.clone();
            mm.positions[a][k] += sign * h;
            let sp = model.singlepoint(&mm, &extra, &o.scf, Some(&guess), false).unwrap();
            lay.pack(&sp.qsh, &sp.dpat, &sp.qpat, None, &mut s[i]);
        }
        let fd: Vec<f64> = (0..m).map(|t| (s[0][t] - s[1][t]) / (2.0 * h)).collect();
        let scale = fd.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
        let mut err = [0.0f64; 3];
        for t in 0..m {
            let e = (r.ds[(t, x)] - fd[t]).abs();
            let g = if t < lay.nsh {
                0
            } else if t < lay.nsh + 3 * lay.nat {
                1
            } else {
                2
            };
            err[g] = err[g].max(e);
        }
        let rel = err.iter().fold(0.0f64, |a, &b| a.max(b)) / scale;
        eprintln!(
            "x {x:3}: max |ds| {scale:.3e}, max err q {:.2e} dip {:.2e} quad {:.2e}, rel {rel:.2e}",
            err[0], err[1], err[2]
        );
        worst = worst.max(rel);
    }
    eprintln!("timings {:?}", r.timings);
    worst
}

/// Explicit potential derivatives `∂V/∂R|_s` against central differences of the potentials.
pub fn check_potential_derivative(mol: &Molecule, o: &Options) -> f64 {
    use crate::xtb::Potential;
    let (model, extra_box) = crate::build_method(Method::Gfn2, mol, o).unwrap();
    let extra: Vec<&dyn ChargeDependentTerm> = extra_box.iter().map(|b| b.as_ref()).collect();
    let (_, st) = model.singlepoint_state(mol, &extra, &o.scf, None, false, true).unwrap();
    let st = st.unwrap();
    let resp = Response::new(&model, &extra, mol, &st, 1);
    let c = resp.potential_derivative(None);
    let lay = resp.lay;
    let (nat, nsh) = (lay.nat, lay.nsh);
    let m = lay.dim();
    let h = 1e-5;
    let mut worst: f64 = 0.0;
    for x in 0..3 * nat {
        let (a, k) = (x / 3, x % 3);
        let mut v = [vec![0.0; m], vec![0.0; m]];
        for (i, sign) in [1.0, -1.0].into_iter().enumerate() {
            let mut mm = mol.clone();
            mm.positions[a][k] += sign * h;
            let geo = model.geometry(&mm, &extra);
            let mut pot = Potential {
                vat: vec![0.0; nat],
                vsh: vec![0.0; nsh],
                vdp: vec![[0.0; 3]; nat],
                vqp: vec![[0.0; 6]; nat],
            };
            model.potentials(&mm, &extra, &geo, &st.qsh, &st.qat, &st.dpat, &st.qpat, &mut pot, false);
            let vsh: Vec<f64> = (0..nsh).map(|s| pot.vsh[s] + pot.vat[model.basis.sh2at[s]]).collect();
            lay.pack(&vsh, &pot.vdp, &pot.vqp, None, &mut v[i]);
        }
        let mut err = [0.0f64; 3];
        let mut scale: f64 = 0.0;
        for t in 0..m {
            let fd = (v[0][t] - v[1][t]) / (2.0 * h);
            scale = scale.max(fd.abs());
            let g = if t < nsh {
                0
            } else if t < nsh + 3 * nat {
                1
            } else {
                2
            };
            err[g] = err[g].max((fd - c[(t, x)]).abs());
        }
        let e = err.iter().fold(0.0f64, |a, &b| a.max(b));
        eprintln!("c x {x:3}: scale {scale:.3e} err q {:.2e} dip {:.2e} quad {:.2e}", err[0], err[1], err[2]);
        worst = worst.max(e / scale);
    }
    worst
}

#[test]
fn response_potential_derivative_water() {
    let mol = mol_from(include_str!("../../../tests/structures/water.xyz"), 0.0, None);
    let mut o = tight(Conventions::Axtb, false);
    o.dispersion = false;
    let e = check_potential_derivative(&mol, &o);
    assert!(e < 1e-7, "relative error {e:.3e}");
}

/// Hessian of `mol` with the given method/step/accuracy.
pub fn hessian(
    mol: &Molecule,
    o: &Options,
    method: crate::HessianMethod,
    step: Option<f64>,
    acc: Option<f64>,
) -> crate::Hessian {
    let mut calc =
        crate::Calculator::new(Method::Gfn2, &mol.numbers, &mol.positions, mol.charge, mol.uhf, o.clone()).unwrap();
    let h = calc.hessian(&mol.positions, &crate::HessianOptions { method, step, accuracy: acc }).unwrap();
    eprintln!("{:?}: asym {:.2e} timings {:?}", method, h.max_asymmetry, h.timings);
    h
}

/// Largest |ΔH| between two Hessians and the largest |H|.
pub fn compare(a: &crate::Hessian, b: &crate::Hessian) -> (f64, f64) {
    let d = a.hessian.iter().zip(&b.hessian).fold(0.0f64, |m, (x, y)| m.max((x - y).abs()));
    let s = a.hessian.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    (d, s)
}

/// Richardson extrapolation `(4 H(h/2) - H(h)) / 3` of two central-difference Hessians.
pub fn richardson(h: &crate::Hessian, h2: &crate::Hessian) -> crate::Hessian {
    let mut out = h2.clone();
    for (o, (a, b)) in out.hessian.iter_mut().zip(h.hessian.iter().zip(&h2.hessian)) {
        *o = (4.0 * b - a) / 3.0;
    }
    out
}

/// Response Hessian vs tight numerical Hessians; both Richardson-extrapolated (O(h⁴)) so
/// that the comparison is not limited by the finite-difference error. Returns the largest
/// |ΔH| of the extrapolated Hessians.
pub fn check_response_hessian(mol: &Molecule, o: &Options) -> f64 {
    check_response_hessian2(mol, o).0
}

/// As [`check_response_hessian`], also returning |resp(5e-4) - num(0.001)| unextrapolated.
pub fn check_response_hessian2(mol: &Molecule, o: &Options) -> (f64, f64) {
    let n1 = hessian(mol, o, crate::HessianMethod::Numerical, Some(0.002), Some(1e-8));
    let n2 = hessian(mol, o, crate::HessianMethod::Numerical, Some(0.001), Some(1e-8));
    let r1 = hessian(mol, o, crate::HessianMethod::Response, Some(1e-3), Some(1e-8));
    let r2 = hessian(mol, o, crate::HessianMethod::Response, Some(5e-4), Some(1e-8));
    let (nx, rx) = (richardson(&n1, &n2), richardson(&r1, &r2));
    let (d, s) = compare(&rx, &nx);
    let (dn, _) = compare(&n1, &n2);
    let (dr, _) = compare(&r2, &n2);
    eprintln!("max|H| {s:.3e}: |num(0.002) - num(0.001)| {dn:.2e}, |resp(5e-4) - num(0.001)| {dr:.2e}, extrapolated |resp - num| {d:.2e}");
    (d, dr)
}

#[test]
fn response_hessian_water() {
    let mol = mol_from(include_str!("../../../tests/structures/water.xyz"), 0.0, None);
    let e = check_response_hessian(&mol, &tight(Conventions::Axtb, false));
    assert!(e < 1e-6, "max |dH| {e:.3e}");
}

#[test]
fn response_hessian_caffeine() {
    let e = check_response_hessian(&caffeine(), &tight(Conventions::Axtb, false));
    assert!(e < 1e-6, "max |dH| {e:.3e}");
}

/// Diagnostic: scan the ALPB gradient of water along the coordinate whose Hessian column
/// is least smooth (prints the gradient and its first differences).
#[test]
#[ignore]
fn diagnose_alpb_smoothness() {
    let mol = mol_from(include_str!("../../../tests/structures/water.xyz"), 0.0, None);
    let o = tight(Conventions::Axtb, true);
    let n1 = hessian(&mol, &o, crate::HessianMethod::Numerical, Some(0.002), Some(1e-8));
    let n2 = hessian(&mol, &o, crate::HessianMethod::Numerical, Some(0.001), Some(1e-8));
    let n3 = 9;
    let (mut best, mut bi) = (0.0, 0);
    for i in 0..n3 * n3 {
        let d = (n1.hessian[i] - n2.hessian[i]).abs();
        if d > best {
            best = d;
            bi = i;
        }
    }
    let (x, y) = (bi / n3, bi % n3);
    eprintln!("largest difference {best:.3e} at ({x}, {y})");
    for t in -8..=8 {
        let mut m = mol.clone();
        m.positions[y / 3][y % 3] += t as f64 * 2.5e-4;
        let g = crate::singlepoint(Method::Gfn2, &m, &o, true).unwrap().gradient.unwrap();
        eprintln!("{:+.5} {:.12}", t as f64 * 2.5e-4, g[x / 3][x % 3]);
    }
}

/// Largest error of an analytic Hessian against central differences of a gradient.
fn fd_check(name: &str, mol: &Molecule, hess: &[f64], grad: impl Fn(&Molecule) -> Vec<[f64; 3]>) -> f64 {
    let n3 = 3 * mol.nat();
    let h = 1e-5;
    let (mut worst, mut scale) = (0.0f64, 0.0f64);
    for y in 0..n3 {
        let mut m = mol.clone();
        m.positions[y / 3][y % 3] += h;
        let gp = grad(&m);
        m.positions[y / 3][y % 3] -= 2.0 * h;
        let gm = grad(&m);
        for x in 0..n3 {
            let fd = (gp[x / 3][x % 3] - gm[x / 3][x % 3]) / (2.0 * h);
            worst = worst.max((fd - hess[x * n3 + y]).abs());
            scale = scale.max(fd.abs());
        }
    }
    eprintln!("{name}: max |H| {scale:.3e}, max error {worst:.3e}");
    worst / scale.max(1e-3)
}

/// Explicit second derivatives of the electrostatic and repulsion terms at fixed moments.
pub fn check_term_hessians(mol: &Molecule, o: &Options) -> f64 {
    let (model, extra_box) = crate::build_method(Method::Gfn2, mol, o).unwrap();
    let extra: Vec<&dyn ChargeDependentTerm> = extra_box.iter().map(|b| b.as_ref()).collect();
    let (_, st) = model.singlepoint_state(mol, &extra, &o.scf, None, false, true).unwrap();
    let st = st.unwrap();
    let bas = &model.basis;
    let n3 = 3 * mol.nat();
    let mut worst: f64 = 0.0;
    // ES2
    let mut h = vec![0.0; n3 * n3];
    model.es2.add_hessian(mol, &bas.ish_at, &bas.nsh_at, &st.qsh, &mut h);
    worst = worst.max(fd_check("ES2", mol, &h, |m| {
        let mut g = vec![[0.0; 3]; m.nat()];
        model.es2.add_gradient(m, &bas.ish_at, &bas.nsh_at, &st.qsh, &mut g);
        g
    }));
    // AES
    if let Some(aes) = &model.aes2 {
        let mut h = vec![0.0; n3 * n3];
        let cn = model.ncoord.compute(mol);
        let jac = model.ncoord.jacobian(mol);
        let cache = aes.update(mol, &cn);
        aes.add_hessian(mol, &cache, &cn, &jac, &st.qat, &st.dpat, &st.qpat, &mut h);
        worst = worst.max(fd_check("AES", mol, &h, |m| {
            let values = model.layout.compute(m).unwrap();
            let geo = crate::terms::Geometry { mol: m, layout: &model.layout, values: &values, parallel: false };
            let cache = aes.update(m, values.get(model.cn));
            let mut chain = crate::terms::Chain::new(&model.layout, m.nat());
            aes.add_gradient(&geo, &cache, &st.qat, &st.dpat, &st.qpat, &mut chain);
            chain.finish(&geo).unwrap()
        }));
    }
    // repulsion
    let mut h = vec![0.0; n3 * n3];
    {
        use crate::terms::{Chain2, Geometry, GeometryTerm, Jacobians};
        let values = model.layout.compute(mol).unwrap();
        let geo = Geometry { mol, layout: &model.layout, values: &values, parallel: false };
        let jac = Jacobians::new(&model.layout);
        model.repulsion.add_hessian(&geo, &(), &mut Chain2 { h: &mut h, jacobians: &jac }).unwrap();
    }
    worst = worst.max(fd_check("repulsion", mol, &h, |m| {
        let mut g = vec![[0.0; 3]; m.nat()];
        let mut e = vec![0.0; m.nat()];
        model.repulsion.compute(m, &mut e, Some(&mut g));
        g
    }));
    worst
}

#[test]
fn analytic_term_hessians() {
    let e1 = check_term_hessians(&caffeine(), &tight(Conventions::Axtb, false));
    let mol = mol_from(include_str!("../../../tests/structures/cu-nh3-4.xyz"), 2.0, None);
    let e2 = check_term_hessians(&mol, &tight(Conventions::Tblite, false));
    assert!(e1 < 1e-8 && e2 < 1e-8, "relative errors {e1:.3e} {e2:.3e}");
}

/// Analytic Hessian vs Richardson-extrapolated Response Hessian.
pub fn check_analytic_hessian(mol: &Molecule, o: &Options) -> f64 {
    let r1 = hessian(mol, o, crate::HessianMethod::Response, Some(1e-3), Some(1e-8));
    let r2 = hessian(mol, o, crate::HessianMethod::Response, Some(5e-4), Some(1e-8));
    let rx = richardson(&r1, &r2);
    let ha = hessian(mol, o, crate::HessianMethod::Analytic, None, Some(1e-8));
    let (d, s) = compare(&ha, &rx);
    eprintln!(
        "max|H| {s:.3e}: |analytic - response(extrapolated)| {d:.2e}, analytic asymmetry {:.2e}",
        ha.max_asymmetry
    );
    d
}

#[test]
fn analytic_hessian_cases() {
    let o = tight(Conventions::Axtb, false);
    let water = || mol_from(include_str!("../../../tests/structures/water.xyz"), 0.0, None);
    let cases: Vec<(&str, Molecule, Options)> = vec![
        ("caffeine", caffeine(), o.clone()),
        ("doublet H2O+", mol_from(include_str!("../../../tests/structures/water.xyz"), 1.0, None), o.clone()),
        ("triplet O2", mol_from(O2, 0.0, Some(2)), o.clone()),
        ("OH-", mol_from(OH, -1.0, None), o.clone()),
        ("twisted C2H4", mol_from(TWISTED_ETHYLENE, 0.0, None), o.clone()),
        ("Cu(NH3)4 2+", mol_from(include_str!("../../../tests/structures/cu-nh3-4.xyz"), 2.0, None), o.clone()),
        ("ferrocene", mol_from(include_str!("../../../tests/structures/ferrocene.xyz"), 0.0, None), o.clone()),
        ("water xtb", water(), tight(Conventions::Xtb, false)),
        ("water xtb-6.7.1", water(), tight(Conventions::Xtb671, false)),
        ("water tblite", water(), tight(Conventions::Tblite, false)),
        ("caffeine tblite", caffeine(), tight(Conventions::Tblite, false)),
    ];
    let mut worst: f64 = 0.0;
    for (name, mol, o) in cases {
        eprintln!("== {name}");
        worst = worst.max(check_analytic_hessian(&mol, &o));
    }
    assert!(worst < 1e-8, "max |dH| {worst:.3e}");
}

#[test]
fn analytic_hessian_alpb() {
    // the ALPB surface term is C1 but not C2 (clamped cubic switching per grid point), so
    // its gradient has kinks densely spaced at the 1e-4 bohr scale: compare with a
    // small-step response Hessian instead of extrapolating
    for mol in [mol_from(include_str!("../../../tests/structures/water.xyz"), 0.0, None), caffeine()] {
        let o = tight(Conventions::Axtb, true);
        let r = hessian(&mol, &o, crate::HessianMethod::Response, Some(5e-5), Some(1e-8));
        let a = hessian(&mol, &o, crate::HessianMethod::Analytic, None, Some(1e-8));
        let (d, s) = compare(&a, &r);
        eprintln!("ALPB: max|H| {s:.3e}, |analytic - response(5e-5)| {d:.2e}, asymmetry {:.2e}", a.max_asymmetry);
        assert!(d < 1e-7, "max |dH| {d:.3e}");
    }
}

#[test]
fn analytic_hessian_solvation_variants() {
    // GBSA (Still kernel), ALPB in xtb 6.8.0 conventions and in xtb 6.7.1 conventions
    // (stale-r1 Born integral: finite differences for that term)
    let mol = caffeine();
    let mut gbsa = tight(Conventions::Axtb, false);
    gbsa.solvation = Some(crate::SolvationSpec {
        model: crate::SolvationModel::Gbsa,
        solvent: "water".into(),
        state: crate::ReferenceState::Gsolv,
    });
    for (name, o, tol) in [
        ("GBSA", gbsa, 1e-7),
        ("ALPB xtb conventions", tight(Conventions::Xtb, true), 1e-7),
        ("ALPB xtb-6.7.1 conventions", tight(Conventions::Xtb671, true), 1e-6),
    ] {
        let r = hessian(&mol, &o, crate::HessianMethod::Response, Some(5e-5), Some(1e-8));
        let a = hessian(&mol, &o, crate::HessianMethod::Analytic, None, Some(1e-8));
        let (d, s) = compare(&a, &r);
        eprintln!("{name}: max|H| {s:.3e}, |analytic - response(5e-5)| {d:.2e}, asymmetry {:.2e}", a.max_asymmetry);
        assert!(d < tol, "{name}: max |dH| {d:.3e}");
    }
}

#[test]
fn analytic_hessian_single_atom_and_threads() {
    let o = tight(Conventions::Axtb, false);
    let mol = mol_from("1\n\nNe 0.0 0.0 0.0\n", 0.0, None);
    let h = hessian(&mol, &o, crate::HessianMethod::Analytic, None, None);
    assert!(h.hessian.iter().all(|x| x.abs() < 1e-12), "single atom: {:?}", h.hessian);
    // the analytic Hessian does not depend on the thread count beyond round-off
    let mol = caffeine();
    let mut calc1 = crate::Calculator::new(Method::Gfn2, &mol.numbers, &mol.positions, 0.0, None, o.clone()).unwrap();
    let mut o8 = o.clone();
    o8.scf.threads = 4;
    let mut calc8 = crate::Calculator::new(Method::Gfn2, &mol.numbers, &mol.positions, 0.0, None, o8).unwrap();
    let ho = crate::HessianOptions::default();
    let h1 = calc1.hessian(&mol.positions, &ho).unwrap();
    let h8 = calc8.hessian(&mol.positions, &ho).unwrap();
    let (d, _) = compare(&h1, &h8);
    eprintln!("1 vs 4 threads: max |dH| {d:.2e}");
    assert!(d < 1e-12, "thread dependence {d:.3e}");
}

/// Diagnostic: how smooth is the exact ALPB Hessian of caffeine? Analytic Hessians at
/// default settings at the geometry and at slightly displaced geometries, and response
/// Hessians with small steps.
#[test]
#[ignore]
fn diagnose_alpb_smoothness_of_hessian() {
    let o = Options { solvation: tight(Conventions::Axtb, true).solvation, ..Default::default() };
    let mol = caffeine();
    let a0 = hessian(&mol, &o, crate::HessianMethod::Analytic, None, None);
    let at = hessian(&mol, &tight(Conventions::Axtb, true), crate::HessianMethod::Analytic, None, Some(1e-8));
    eprintln!("default vs tight accuracy: {:.2e}", compare(&a0, &at).0);
    for step in [5e-5, 2e-4, 1e-3, 5e-3] {
        let r = hessian(&mol, &o, crate::HessianMethod::Response, Some(step), None);
        eprintln!("response({step:.0e}) vs analytic: {:.2e}", compare(&a0, &r).0);
    }
    for (k, d) in [(0usize, 1e-4), (5, 1e-4), (17, 3e-4), (40, 1e-3)] {
        let mut m = mol.clone();
        m.positions[k / 3][k % 3] += d;
        let a = hessian(&m, &o, crate::HessianMethod::Analytic, None, None);
        eprintln!("analytic at R + {d:.0e} e_{k} vs at R: {:.2e}", compare(&a0, &a).0);
    }
    let mut og = o.clone();
    og.solvation = None;
    let g0 = hessian(&mol, &og, crate::HessianMethod::Analytic, None, None);
    let mut m = mol.clone();
    m.positions[0][0] += 1e-4;
    let g1 = hessian(&m, &og, crate::HessianMethod::Analytic, None, None);
    eprintln!("gas phase: analytic at R + 1e-4 e_0 vs at R: {:.2e}", compare(&g0, &g1).0);
}

/// Diagnostic: response Hessians of caffeine in ALPB water at several steps vs analytic.
#[test]
#[ignore]
fn diagnose_alpb_caffeine() {
    let o = tight(Conventions::Axtb, true);
    let mol = caffeine();
    let a = hessian(&mol, &o, crate::HessianMethod::Analytic, None, Some(1e-8));
    for step in [1e-3, 5e-4, 2.5e-4, 1e-4, 5e-5] {
        let r = hessian(&mol, &o, crate::HessianMethod::Response, Some(step), Some(1e-8));
        let (d, _) = compare(&a, &r);
        eprintln!("step {step:.1e}: |analytic - response| {d:.3e}, response asymmetry {:.2e}", r.max_asymmetry);
    }
}

#[test]
fn analytic_hessian_water() {
    let mol = mol_from(include_str!("../../../tests/structures/water.xyz"), 0.0, None);
    let e = check_analytic_hessian(&mol, &tight(Conventions::Axtb, false));
    assert!(e < 1e-8, "max |dH| {e:.3e}");
}

#[test]
fn response_hessian_special_cases() {
    let o = tight(Conventions::Axtb, false);
    let cases: Vec<(&str, Molecule, Options)> = vec![
        ("doublet H2O+", mol_from(include_str!("../../../tests/structures/water.xyz"), 1.0, None), o.clone()),
        ("triplet O2", mol_from(O2, 0.0, Some(2)), o.clone()),
        ("OH-", mol_from(OH, -1.0, None), o.clone()),
        ("twisted C2H4", mol_from(TWISTED_ETHYLENE, 0.0, None), o.clone()),
        ("Cu(NH3)4 2+", mol_from(include_str!("../../../tests/structures/cu-nh3-4.xyz"), 2.0, None), o.clone()),
        (
            "water xtb",
            mol_from(include_str!("../../../tests/structures/water.xyz"), 0.0, None),
            tight(Conventions::Xtb, false),
        ),
        (
            "water tblite",
            mol_from(include_str!("../../../tests/structures/water.xyz"), 0.0, None),
            tight(Conventions::Tblite, false),
        ),
    ];
    let mut worst: f64 = 0.0;
    for (name, mol, o) in cases {
        eprintln!("== {name}");
        worst = worst.max(check_response_hessian(&mol, &o));
    }
    assert!(worst < 1e-8, "max |dH| {worst:.3e}");
    // the ALPB gradient has kinks at the 1e-3 bohr scale (surface grid), so its numerical
    // Hessians cannot be extrapolated; compare the small-step ones directly
    let mol = mol_from(include_str!("../../../tests/structures/water.xyz"), 0.0, None);
    let (_, d) = check_response_hessian2(&mol, &tight(Conventions::Axtb, true));
    assert!(d < 2e-6, "ALPB max |dH| {d:.3e}");
}

#[test]
fn response_scc_water() {
    let mol = mol_from(include_str!("../../../tests/structures/water.xyz"), 0.0, None);
    let e = check_scc_response(&mol, &tight(Conventions::Axtb, false), &[0, 1, 2, 3, 4, 5, 6, 7, 8]);
    assert!(e < 1e-6, "relative error {e:.3e}");
}

/// Twisted ethylene (90°): degenerate frontier orbitals, fractional occupations at 300 K.
pub const TWISTED_ETHYLENE: &str = "6\n\nC 0.0 0.0 0.6650\nC 0.0 0.0 -0.6650\nH 0.0 0.9235 1.2360\nH 0.0 -0.9235 1.2360\nH 0.9235 0.0 -1.2360\nH -0.9235 0.0 -1.2360\n";
/// Triplet dioxygen.
pub const O2: &str = "2\n\nO 0.0 0.0 0.0\nO 0.05 0.02 1.21\n";
/// Hydroxide, slightly bent off-axis H position for a generic test.
pub const OH: &str = "2\n\nO 0.0 0.0 0.0\nH 0.1 0.05 0.97\n";

fn caffeine() -> Molecule {
    mol_from(include_str!("../../../tests/structures/caffeine.xyz"), 0.0, None)
}

#[test]
fn response_scc_caffeine() {
    let e = check_scc_response(&caffeine(), &tight(Conventions::Axtb, false), &[0, 1, 2, 21, 22, 23, 45, 46, 47]);
    assert!(e < 1e-6, "relative error {e:.3e}");
}

#[test]
fn response_scc_caffeine_alpb() {
    let e = check_scc_response(&caffeine(), &tight(Conventions::Axtb, true), &[0, 1, 2, 21, 22, 23, 45, 46, 47]);
    assert!(e < 1e-6, "relative error {e:.3e}");
}

#[test]
fn response_scc_caffeine_xtb_tblite_conventions() {
    let e1 = check_scc_response(&caffeine(), &tight(Conventions::Xtb, false), &[0, 1, 2, 45]);
    let e2 = check_scc_response(&caffeine(), &tight(Conventions::Tblite, false), &[0, 1, 2, 45]);
    assert!(e1 < 1e-6 && e2 < 1e-6, "relative errors {e1:.3e} {e2:.3e}");
}

#[test]
fn response_scc_doublet_cation() {
    let mol = mol_from(include_str!("../../../tests/structures/water.xyz"), 1.0, None);
    let e = check_scc_response(&mol, &tight(Conventions::Axtb, false), &[0, 1, 2, 3, 4, 5, 6, 7, 8]);
    assert!(e < 1e-6, "relative error {e:.3e}");
}

#[test]
fn response_scc_triplet_o2() {
    let mol = mol_from(O2, 0.0, Some(2));
    let e = check_scc_response(&mol, &tight(Conventions::Axtb, false), &[0, 1, 2, 3, 4, 5]);
    assert!(e < 1e-6, "relative error {e:.3e}");
}

#[test]
fn response_scc_anion() {
    let mol = mol_from(OH, -1.0, None);
    let e = check_scc_response(&mol, &tight(Conventions::Axtb, false), &[0, 1, 2, 3, 4, 5]);
    assert!(e < 1e-6, "relative error {e:.3e}");
}

#[test]
fn response_scc_small_gap() {
    let mol = mol_from(TWISTED_ETHYLENE, 0.0, None);
    let e = check_scc_response(&mol, &tight(Conventions::Axtb, false), &[0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 13, 14]);
    assert!(e < 1e-6, "relative error {e:.3e}");
}

#[test]
fn response_scc_transition_metal_open_shell() {
    let mol = mol_from(include_str!("../../../tests/structures/cu-nh3-4.xyz"), 2.0, None);
    let e = check_scc_response(&mol, &tight(Conventions::Axtb, false), &[0, 1, 2, 3, 4, 5, 9, 10, 11]);
    assert!(e < 1e-6, "relative error {e:.3e}");
}

#[test]
fn response_scc_ferrocene() {
    let mol = mol_from(include_str!("../../../tests/structures/ferrocene.xyz"), 0.0, None);
    let e = check_scc_response(&mol, &tight(Conventions::Axtb, false), &[0, 1, 2, 3, 4, 5, 33, 34, 35]);
    assert!(e < 1e-6, "relative error {e:.3e}");
}
