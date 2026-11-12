//! g-xTB in implicit solvent (`axtb_gxtb::composite`): the composite equals its three parts run
//! separately, its gradient and Hessian are the derivatives of its energy, threads change
//! nothing, the GFN2 pair follows its retry ladder, and the settings resolve as documented.
//! The comparison with the three xtb programs is `tests/solvation_composite.py`.

use axtb::method::{CompositeReport, Method, ScfSettings, SinglePoint, Want};
use axtb::{HessianMethod, HessianOptions, SolvationModel, SolvationSpec};
use axtb_gxtb::composite::GFN2_ETEMP;
use axtb_gxtb::{Calculator, CompositeCalculator, CompositeOptions, Conventions, Options, Version};

/// Acetate (anion), bohr.
fn acetate() -> (Vec<usize>, Vec<[f64; 3]>) {
    let a = 1.0 / axtb_gxtb::BOHR_ANGSTROM;
    let xyz = [
        (6, [0.95727, -0.02507, 0.04245]),
        (6, [2.47592, -0.04629, 0.05497]),
        (8, [3.03349, 0.92888, -0.52062]),
        (8, [2.98681, -1.04103, 0.64211]),
        (1, [0.57548, 0.85802, -0.47879]),
        (1, [0.57813, -0.01187, 1.06868]),
        (1, [0.57813, -0.91714, -0.46504]),
    ];
    (xyz.iter().map(|x| x.0).collect(), xyz.iter().map(|x| [x.1[0] * a, x.1[1] * a, x.1[2] * a]).collect())
}

/// The composite's report of a single point.
fn report(sp: &SinglePoint) -> &CompositeReport {
    sp.extras.composite.as_deref().expect("the composite's report")
}

/// The energy of the component `key` of a composite single point.
fn component(sp: &SinglePoint, key: &str) -> f64 {
    report(sp).components.iter().find(|(k, _)| *k == key).expect("the component").1
}

/// ΔG_solv of a composite single point.
fn dg_solv(sp: &SinglePoint) -> f64 {
    sp.term("solvation_correction").expect("the correction term")
}

fn water() -> (Vec<usize>, Vec<[f64; 3]>) {
    (vec![8, 1, 1], vec![[0.0, 0.0, 0.22], [0.0, 1.43, -0.89], [0.0, -1.43, -0.89]])
}

fn water_alpb() -> SolvationSpec {
    SolvationSpec { model: SolvationModel::Alpb, solvent: "water".into(), state: Default::default() }
}

/// The SCF settings with electronic temperature `etemp` and accuracy `accuracy`.
fn scf(etemp: Option<f64>, accuracy: f64) -> ScfSettings {
    ScfSettings { etemp, accuracy, ..Default::default() }
}

fn opts(conventions: Conventions) -> CompositeOptions {
    CompositeOptions { solvation: water_alpb(), conventions, ..Default::default() }
}

/// The three parts with their own calculators, at the composite's settings.
fn parts(nums: &[usize], pos: &[[f64; 3]], charge: f64, conventions: Conventions) -> (f64, Vec<[f64; 3]>, f64, f64) {
    let gopts = Options { conventions, ..Default::default() };
    let mut g = Calculator::new(Version::V2_0_1, nums, pos, charge, None, gopts).unwrap();
    let rg = g.singlepoint(pos, &Want::GRADIENT).unwrap();
    let fconv = match conventions {
        Conventions::Axtb => axtb::Conventions::Axtb,
        Conventions::Gxtb => axtb::Conventions::Xtb671,
    };
    let mut o = axtb::Options { conventions: fconv, ..Default::default() };
    o.scf.etemp = Some(GFN2_ETEMP);
    let mol = axtb::Molecule::new(nums, pos, charge, None).unwrap();
    let gas = axtb::singlepoint(axtb::Hamiltonian::Gfn2, &mol, &o, &Want::GRADIENT).unwrap();
    o.solvation = Some(water_alpb());
    let solv = axtb::singlepoint(axtb::Hamiltonian::Gfn2, &mol, &o, &Want::GRADIENT).unwrap();
    let grad: Vec<[f64; 3]> = (0..nums.len())
        .map(|i| {
            std::array::from_fn(|k| {
                rg.gradient.as_ref().unwrap()[i][k]
                    + (solv.gradient.as_ref().unwrap()[i][k] - gas.gradient.as_ref().unwrap()[i][k])
            })
        })
        .collect();
    (rg.energy + (solv.energy - gas.energy), grad, solv.energy - gas.energy, rg.energy)
}

#[test]
fn composite_is_the_sum_of_its_parts() {
    let (nums, pos) = acetate();
    for conv in [Conventions::Axtb, Conventions::Gxtb] {
        let (e, g, dg, eg) = parts(&nums, &pos, -1.0, conv);
        let mut c = CompositeCalculator::new(&nums, &pos, -1.0, None, opts(conv)).unwrap();
        let r = c.singlepoint(&pos, &Want::GRADIENT).unwrap();
        assert!((r.energy - e).abs() < 1e-12, "{conv:?}: {} vs {e}", r.energy);
        assert!((dg_solv(&r) - dg).abs() < 1e-12);
        assert!((component(&r, "gxtb") - eg).abs() < 1e-12);
        assert_eq!(component(&r, "solvation_correction"), dg_solv(&r));
        assert!(dg_solv(&r) < -0.05, "acetate in water: dG_solv {}", dg_solv(&r));
        // the pieces: g-xTB's and the correction
        assert!((r.pieces() - r.energy).abs() < 1e-10);
        let gr = r.gradient.as_ref().unwrap();
        for (a, b) in gr.iter().zip(&g) {
            for k in 0..3 {
                assert!((a[k] - b[k]).abs() < 1e-11);
            }
        }
        // what each part ran with: the programs' defaults, g-xTB 0 K and GFN2 300 K, no retry
        let parts = &report(&r).parts;
        assert_eq!(parts[0].etemp, 0.0);
        assert_eq!((parts[1].etemp, parts[2].etemp), (300.0, 300.0));
        assert!(parts.iter().all(|p| p.retry.is_none()));
        assert_eq!(parts[0].label, "gxtb-v2.0.1");
        assert_eq!(parts[1].label, "gfn2+alpb");
        let expect = if conv == Conventions::Axtb { ("axtb", "axtb", true) } else { ("gxtb", "xtb-6.7.1", false) };
        assert_eq!((c.conventions_labels().0, c.conventions_labels().1, c.reuses_guess()), expect);
        assert_eq!(c.definition(), "E = E[gxtb-v2.0.1] + (E[gfn2 + alpb(water, gsolv)] - E[gfn2])");
    }
}

#[test]
fn gradient_is_the_derivative_of_the_energy() {
    let (nums, pos) = acetate();
    let o = CompositeOptions { scf: scf(None, 1e-6), ..opts(Conventions::Axtb) };
    let mut c = CompositeCalculator::new(&nums, &pos, -1.0, None, o).unwrap();
    let g = c.singlepoint(&pos, &Want::GRADIENT).unwrap().gradient.unwrap();
    let h = 1e-4;
    let mut worst: f64 = 0.0;
    for (atom, dir) in [(0, 0), (1, 2), (2, 1), (3, 0), (6, 2)] {
        let mut p = pos.clone();
        p[atom][dir] += h;
        let ep = c.singlepoint(&p, &Want::ENERGY).unwrap().energy;
        p[atom][dir] -= 2.0 * h;
        let em = c.singlepoint(&p, &Want::ENERGY).unwrap().energy;
        worst = worst.max(((ep - em) / (2.0 * h) - g[atom][dir]).abs());
    }
    assert!(worst < 2e-7, "gradient vs finite differences: {worst:.2e}");
}

#[test]
fn hessian_is_the_derivative_of_the_gradient() {
    let (nums, pos) = water();
    let mut c = CompositeCalculator::new(&nums, &pos, 0.0, None, opts(Conventions::Axtb)).unwrap();
    let h = c.hessian(&pos, &HessianOptions::default()).unwrap();
    assert_eq!(h.method, HessianMethod::Analytic);
    assert!(h.parts.iter().all(|p| p.hessian_method == Some(HessianMethod::Analytic)));
    let o = CompositeOptions { scf: scf(None, 1e-6), ..opts(Conventions::Axtb) };
    let mut f = CompositeCalculator::new(&nums, &pos, 0.0, None, o).unwrap();
    let n3 = 9;
    let step = 1e-4;
    let mut worst: f64 = 0.0;
    for col in 0..n3 {
        let mut p = pos.clone();
        p[col / 3][col % 3] += step;
        let gp = f.singlepoint(&p, &Want::GRADIENT).unwrap().gradient.unwrap();
        p[col / 3][col % 3] -= 2.0 * step;
        let gm = f.singlepoint(&p, &Want::GRADIENT).unwrap().gradient.unwrap();
        for row in 0..n3 {
            let fd = (gp[row / 3][row % 3] - gm[row / 3][row % 3]) / (2.0 * step);
            worst = worst.max((fd - h.hessian[row * n3 + col]).abs());
        }
    }
    assert!(worst < 2e-5, "Hessian vs finite differences of the gradient: {worst:.2e}");
    // the total is the sum of the parts, each from its own calculator
    let ho = HessianOptions::default();
    let mut g = Calculator::new(Version::V2_0_1, &nums, &pos, 0.0, None, Options::default()).unwrap();
    let hg = g.hessian(&pos, &ho).unwrap().hessian;
    let mut o =
        axtb::Options { scf: ScfSettings { etemp: Some(GFN2_ETEMP), ..Default::default() }, ..Default::default() };
    let mut gas = axtb::Calculator::new(axtb::Hamiltonian::Gfn2, &nums, &pos, 0.0, None, o.clone()).unwrap();
    let hgas = gas.hessian(&pos, &ho).unwrap().hessian;
    o.solvation = Some(water_alpb());
    let mut solv = axtb::Calculator::new(axtb::Hamiltonian::Gfn2, &nums, &pos, 0.0, None, o).unwrap();
    let hsolv = solv.hessian(&pos, &ho).unwrap().hessian;
    for i in 0..n3 * n3 {
        let s = hg[i] + (hsolv[i] - hgas[i]);
        assert_eq!(s, h.hessian[i]);
    }
    // and its energy and gradient are those of the single point
    let r = c.singlepoint(&pos, &Want::GRADIENT).unwrap();
    assert!((r.energy - h.energy).abs() < 1e-8);
}

#[test]
fn threads_change_nothing() {
    let (nums, pos) = acetate();
    let mut one = CompositeCalculator::new(&nums, &pos, -1.0, None, opts(Conventions::Axtb)).unwrap();
    let mut two =
        CompositeCalculator::new(&nums, &pos, -1.0, None, CompositeOptions { threads: 2, ..opts(Conventions::Axtb) })
            .unwrap();
    let a = one.singlepoint(&pos, &Want::GRADIENT).unwrap();
    let b = two.singlepoint(&pos, &Want::GRADIENT).unwrap();
    assert!((a.energy - b.energy).abs() < 1e-12);
    let ha = one.hessian(&pos, &HessianOptions::default()).unwrap().hessian;
    let hb = two.hessian(&pos, &HessianOptions::default()).unwrap().hessian;
    assert!(ha.iter().zip(&hb).all(|(x, y)| (x - y).abs() < 1e-11));
}

/// The GFN2 pair of a diatomic at distance `r` (Å) in xtb conventions: the composite's
/// solvation energy, the rung both halves converged on, and the same halves run directly there.
fn ladder(z: [usize; 2], r: f64, etemp: f64) -> (Vec<usize>, Vec<[f64; 3]>, SinglePoint, f64) {
    let r = r / axtb_gxtb::BOHR_ANGSTROM;
    let (nums, pos) = (z.to_vec(), vec![[0.0, 0.0, -0.5 * r], [0.0, 0.0, 0.5 * r]]);
    let mut c = CompositeCalculator::new(&nums, &pos, 0.0, None, opts(Conventions::Gxtb)).unwrap();
    let res = c.singlepoint(&pos, &Want::GRADIENT).unwrap();
    let mut o = axtb::Options { conventions: axtb::Conventions::Xtb671, ..Default::default() };
    o.scf.etemp = Some(etemp);
    o.scf.max_iter = Some(1000);
    let mol = axtb::Molecule::new(&nums, &pos, 0.0, None).unwrap();
    let gas = axtb::singlepoint(axtb::Hamiltonian::Gfn2, &mol, &o, &Want::ENERGY).unwrap().energy;
    o.solvation = Some(water_alpb());
    let solv = axtb::singlepoint(axtb::Hamiltonian::Gfn2, &mol, &o, &Want::ENERGY).unwrap().energy;
    (nums, pos, res, solv - gas)
}

/// F2 at 3.0 Å: neither half converges at 300 K in 250 iterations (xtb: not converged; axtb:
/// the Fermi level is lost), both do at 1000 K, as with the xtb program: the first retry,
/// both halves together; g-xTB converges at 0 K.
#[test]
fn gfn2_pair_follows_the_retry_ladder() {
    let (nums, pos, res, direct) = ladder([9, 9], 3.0, 1000.0);
    let parts = &report(&res).parts;
    assert!(parts[0].retry.is_none());
    for p in &parts[1..] {
        assert_eq!(p.retry.as_deref(), Some("etemp 1000 K, 1000 iterations"));
        assert_eq!((p.etemp, p.max_iter), (1000.0, 1000));
    }
    assert!((dg_solv(&res) - direct).abs() < 1e-12);
    // without retries it fails like the first program call
    let mut strict = CompositeCalculator::new(
        &nums,
        &pos,
        0.0,
        None,
        CompositeOptions { retries: false, ..opts(Conventions::Gxtb) },
    )
    .unwrap();
    assert!(strict.singlepoint(&pos, &Want::ENERGY).is_err());
    // the next geometry starts at the base settings again
    let p2 = water();
    let mut w = CompositeCalculator::new(&p2.0, &p2.1, 0.0, None, opts(Conventions::Gxtb)).unwrap();
    let rw = w.singlepoint(&p2.1, &Want::ENERGY).unwrap();
    let parts = &report(&rw).parts;
    assert!(parts[1].retry.is_none() && parts[1].etemp == 300.0);
}

/// HF at 3.0 Å: the gas-phase half fails at 300 K, at 1000 K and with 1000 iterations and
/// converges at 5000 K, as with the xtb program (`tests/solvation_composite.py`): the last
/// rung. Slow (failing rungs of 1000 Broyden iterations with full history, ~1 min).
#[test]
#[ignore = "slow: about a minute of failing 1000-iteration rungs"]
fn gfn2_pair_reaches_the_last_rung() {
    let (_, _, res, direct) = ladder([1, 9], 3.0, 5000.0);
    for p in &report(&res).parts[1..] {
        assert_eq!(p.retry.as_deref(), Some("etemp 5000 K, 1000 iterations"));
    }
    assert!((dg_solv(&res) - direct).abs() < 1e-12);
}

#[test]
fn settings_resolve_as_documented() {
    let (nums, pos) = water();
    // an explicit temperature applies to all parts (g-xTB 2.0.1 with Fermi smearing)
    let c = CompositeCalculator::new(
        &nums,
        &pos,
        0.0,
        None,
        CompositeOptions { scf: scf(Some(0.0), 1.0), ..opts(Conventions::Axtb) },
    )
    .unwrap();
    assert_eq!(c.etemps(), (0.0, 0.0));
    let mut hot = CompositeCalculator::new(
        &nums,
        &pos,
        0.0,
        None,
        CompositeOptions { scf: scf(Some(3000.0), 1.0), ..opts(Conventions::Axtb) },
    )
    .unwrap();
    assert_eq!(hot.etemps(), (3000.0, 3000.0));
    let r = hot.singlepoint(&pos, &Want::GRADIENT).unwrap();
    assert!(report(&r).parts.iter().all(|p| p.etemp == 3000.0));
    let mut g = Calculator::new(
        Version::V2_0_1,
        &nums,
        &pos,
        0.0,
        None,
        Options { scf: ScfSettings { etemp: Some(3000.0), ..Default::default() }, ..Default::default() },
    )
    .unwrap();
    assert_eq!(component(&r, "gxtb"), g.singlepoint(&pos, &Want::ENERGY).unwrap().energy);
    let v1 = CompositeOptions { version: Version::V1_1, scf: scf(Some(300.0), 1.0), ..opts(Conventions::Axtb) };
    assert_eq!(CompositeCalculator::new(&nums, &pos, 0.0, None, v1).unwrap().etemps(), (300.0, 300.0));
    for t in [-1.0, f64::NAN, f64::INFINITY] {
        let bad = CompositeOptions { scf: scf(Some(t), 1.0), ..opts(Conventions::Axtb) };
        assert!(CompositeCalculator::new(&nums, &pos, 0.0, None, bad).is_err(), "etemp {t}");
    }
    // correction conventions can be chosen separately
    let c = CompositeCalculator::new(
        &nums,
        &pos,
        0.0,
        None,
        CompositeOptions { correction_conventions: Some(axtb::Conventions::Tblite), ..opts(Conventions::Gxtb) },
    )
    .unwrap();
    assert_eq!(c.conventions_labels(), ("gxtb", "tblite"));
    assert!(!c.reuses_guess());
    // unknown solvents fail at setup
    let bad = CompositeOptions {
        solvation: SolvationSpec { solvent: "lava".into(), ..water_alpb() },
        ..opts(Conventions::Axtb)
    };
    assert!(CompositeCalculator::new(&nums, &pos, 0.0, None, bad).is_err());
    assert!("GFN2-xTB".parse::<axtb_gxtb::Correction>().is_ok());
    assert!("gfn1".parse::<axtb_gxtb::Correction>().is_err());
    // v1.1 works too, with its numerical Hessian by default
    let mut v1 = CompositeCalculator::new(
        &nums,
        &pos,
        0.0,
        None,
        CompositeOptions { version: Version::V1_1, ..opts(Conventions::Axtb) },
    )
    .unwrap();
    assert_eq!(v1.info().hessian.default, HessianMethod::Numerical);
    assert!(v1.singlepoint(&pos, &Want::GRADIENT).is_ok());
    let _ = HessianOptions::default();
}
