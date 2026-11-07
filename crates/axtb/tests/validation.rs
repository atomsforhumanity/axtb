//! Input validation: broken structures, impossible charge/spin states and invalid SCF and
//! Hessian settings must be errors (never panics) in every conventions mode (xtb and tblite
//! compute garbage or panic-free nonsense for most of these; see tests/reference_matrix,
//! category `failures`).

use axtb::{Calculator, Conventions, Error, HessianMethod, HessianOptions, Method, Molecule, Options, ScfGuess};

const MODES: [Conventions; 4] = [Conventions::Axtb, Conventions::Xtb, Conventions::Xtb671, Conventions::Tblite];

fn h2() -> (Vec<usize>, Vec<[f64; 3]>) {
    (vec![1, 1], vec![[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
}

fn water() -> (Vec<usize>, Vec<[f64; 3]>) {
    (vec![8, 1, 1], vec![[0.0, 0.0, 0.22], [0.0, 1.43, -0.89], [0.0, -1.43, -0.89]])
}

fn run(
    mol: (Vec<usize>, Vec<[f64; 3]>),
    charge: f64,
    uhf: Option<usize>,
    conv: Conventions,
    f: impl Fn(&mut Options),
) -> axtb::Result<axtb::SinglePoint> {
    let m = Molecule::new(&mol.0, &mol.1, charge, uhf)?;
    let mut o = Options { conventions: conv, ..Default::default() };
    f(&mut o);
    axtb::singlepoint(Method::Gfn2, &m, &o, true)
}

fn is_input_error(r: axtb::Result<axtb::SinglePoint>) -> bool {
    matches!(r, Err(Error::Input(_)))
}

/// Before the check, H2 with charge +3 returned charges summing to +2, water +9 a positive
/// "energy", and H2 -5, H2 with 4 unpaired electrons and O2 with 10 panicked (index out of
/// bounds in the Fermi filling).
#[test]
fn impossible_electron_counts_are_input_errors() {
    for conv in MODES {
        assert!(is_input_error(run(h2(), 3.0, None, conv, |_| {})), "H2 +3 ({conv:?})");
        assert!(is_input_error(run(water(), 9.0, None, conv, |_| {})), "water +9 ({conv:?})");
        assert!(is_input_error(run(h2(), -5.0, None, conv, |_| {})), "H2 -5 ({conv:?})");
        assert!(is_input_error(run(water(), -30.0, None, conv, |_| {})), "water -30 ({conv:?})");
        assert!(is_input_error(run(h2(), 0.0, Some(4), conv, |_| {})), "H2 uhf 4 ({conv:?})");
        let o2 = (vec![8, 8], vec![[0.0, 0.0, 0.0], [0.0, 0.0, 2.29]]);
        assert!(is_input_error(run(o2, 0.0, Some(10), conv, |_| {})), "O2 uhf 10 ({conv:?})");
        // no electrons at all is a valid (if unusual) state that every program computes
        let sp = run(h2(), 2.0, None, conv, |_| {}).unwrap();
        assert!((sp.qat.iter().sum::<f64>() - 2.0).abs() < 1e-12);
    }
}

#[test]
fn invalid_scf_settings_are_input_errors() {
    for conv in MODES {
        for acc in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(is_input_error(run(water(), 0.0, None, conv, |o| o.scf.accuracy = acc)), "accuracy {acc}");
        }
        for t in [-5.0, f64::NAN] {
            assert!(is_input_error(run(water(), 0.0, None, conv, |o| o.scf.etemp = t)), "etemp {t}");
        }
        assert!(run(water(), 0.0, None, conv, |o| o.scf.etemp = 0.0).is_ok());
    }
}

/// A fractional electron count has no Fermi level inside a HOMO-LUMO gap larger than kT (H2 at
/// 300 K used to fail with "Fermi level not found", and tblite returns the neutral molecule):
/// a non-integer charge is an input error at every temperature, as in g-xTB.
#[test]
fn fractional_charge_is_an_input_error() {
    for conv in MODES {
        for etemp in [300.0, 0.0, 5000.0] {
            assert!(is_input_error(run(h2(), 0.5, None, conv, |o| o.scf.etemp = etemp)), "{conv:?} {etemp} K");
        }
        assert!(is_input_error(run(water(), -1e-9, None, conv, |_| {})), "{conv:?}");
    }
}

/// Structures no method can compute are input errors when the molecule is built, and when a
/// calculator gets new positions (which then stay as they were); before, coincident atoms
/// panicked in the eigensolver's sort and infinite positions or a NaN charge ran into NaN.
#[test]
fn broken_structures_are_input_errors() {
    let (z, r) = water();
    let input = |e: axtb::Result<Molecule>| matches!(e, Err(Error::Input(_)));
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut p = r.clone();
        p[1][2] = bad;
        assert!(input(Molecule::new(&z, &p, 0.0, None)), "position {bad}");
        assert!(input(Molecule::new(&z, &r, bad, None)), "charge {bad}");
    }
    // the closest accepted distance and just below it
    let pair = |d: f64| Molecule::new(&[8, 1], &[[0.0, 0.0, 0.0], [d * 0.6, d * 0.8, 0.0]], 0.0, None);
    assert!(pair(axtb::MIN_ATOM_DISTANCE).is_ok());
    assert!(input(pair(axtb::MIN_ATOM_DISTANCE * (1.0 - 1e-12))));
    let mut p = r.clone();
    p[2] = p[0];
    assert!(input(Molecule::new(&z, &p, 0.0, None)));
    for conv in MODES {
        assert!(is_input_error(run((z.clone(), p.clone()), 0.0, None, conv, |_| {})), "{conv:?}");
    }

    let mut calc = Calculator::new(Method::Gfn2, &z, &r, 0.0, None, Options::default()).unwrap();
    let mut far = r.clone();
    far[1][0] += 0.1;
    let mut inf = r.clone();
    inf[0][1] = f64::INFINITY;
    for bad in [p.clone(), inf, r[..2].to_vec()] {
        assert!(matches!(calc.singlepoint(&bad, true), Err(Error::Input(_))));
        assert!(matches!(calc.hessian(&bad, &HessianOptions::default()), Err(Error::Input(_))));
        assert_eq!(calc.molecule().positions, r);
    }
    assert!(calc.singlepoint(&far, true).is_ok());
}

#[test]
fn invalid_mixer_and_thread_settings_are_input_errors() {
    for conv in MODES {
        for damping in [0.0, -1.0, 1.5, f64::NAN, f64::INFINITY] {
            assert!(is_input_error(run(water(), 0.0, None, conv, |o| o.scf.damping = damping)), "damping {damping}");
        }
        assert!(run(water(), 0.0, None, conv, |o| o.scf.damping = 1.0).is_ok());
        assert!(is_input_error(run(water(), 0.0, None, conv, |o| o.scf.max_iter = 0)));
        assert!(is_input_error(run(water(), 0.0, None, conv, |o| o.scf.threads = 0)));
    }
    let (z, r) = water();
    let mut o = Options::default();
    o.scf.threads = 0;
    assert!(matches!(Calculator::new(Method::Gfn2, &z, &r, 0.0, None, o.clone()), Err(Error::Input(_))));
    assert!(matches!(Calculator::with_pool(Method::Gfn2, &z, &r, 0.0, None, o, None), Err(Error::Input(_))));
}

#[test]
fn invalid_hessian_settings_are_input_errors() {
    let (z, r) = water();
    let mut calc = Calculator::new(Method::Gfn2, &z, &r, 0.0, None, Options::default()).unwrap();
    for method in [HessianMethod::Numerical, HessianMethod::Response, HessianMethod::Analytic] {
        for bad in [0.0, -0.005, f64::NAN, f64::INFINITY] {
            let step = HessianOptions { method, step: Some(bad), accuracy: None };
            let acc = HessianOptions { method, step: None, accuracy: Some(bad) };
            assert!(matches!(calc.hessian(&r, &step), Err(Error::Input(_))), "{method:?} step {bad}");
            assert!(matches!(calc.hessian(&r, &acc), Err(Error::Input(_))), "{method:?} accuracy {bad}");
        }
    }
}

/// A guess must have this molecule's sizes and finite values; a finite guess so large that the
/// potentials overflow stops the SCF with a numerical error at its first Fock matrix, in every
/// conventions mode (before, the NaN eigenvalues panicked in the eigensolver's sort).
#[test]
fn bad_guesses_are_errors_not_panics() {
    let (z, r) = water();
    for conv in MODES {
        let o = Options { conventions: conv, ..Default::default() };
        let mut calc = Calculator::new(Method::Gfn2, &z, &r, 0.0, None, o).unwrap();
        let g = calc.singlepoint(&r, false).unwrap().guess();
        let mut nan = g.clone();
        nan.qpat[1][3] = f64::NAN;
        assert!(matches!(calc.set_guess(nan), Err(Error::Input(_))));
        let mut short = g.clone();
        short.qpat.pop();
        assert!(matches!(calc.set_guess(short), Err(Error::Input(_))));
        let huge = ScfGuess { qsh: vec![1e300; g.qsh.len()], ..g };
        calc.set_guess(huge).unwrap();
        let e = calc.singlepoint(&r, true);
        assert!(matches!(e, Err(Error::Numerical(_))), "{conv:?}: {e:?}");
    }
}

/// xtb (6.7.1 and 6.8.0) applies Fermi smearing only above 0.1 K; at 0.1 K singlet O2
/// (degenerate π*) keeps the aufbau occupation in the xtb program (-7.906638884 Eh at
/// 2.2866 bohr), 2.5e-3 Eh below the smeared result that tblite's rule gives at the same
/// temperature.
#[test]
fn xtb_conventions_smear_only_above_one_tenth_kelvin() {
    let o2 = (vec![8, 8], vec![[0.0, 0.0, 0.0], [0.0, 0.0, 1.21 / 0.529177210544]]);
    let e = |conv: Conventions, t: f64| {
        run(o2.clone(), 0.0, Some(0), conv, |o| {
            o.scf.etemp = t;
            o.scf.accuracy = 1e-6
        })
        .unwrap()
        .energy
    };
    for xtb in [Conventions::Xtb, Conventions::Xtb671] {
        let e0 = e(xtb, 0.0);
        assert!((e(xtb, 0.1) - e0).abs() < 1e-10, "{xtb:?}");
        // 0.11 K is smeared (fractional π* occupations) in every mode
        assert!(e(xtb, 0.11) - e0 > 2e-3, "{xtb:?}");
    }
    assert!(e(Conventions::Tblite, 0.2) - e(Conventions::Tblite, 0.0) > 2e-3);
}

/// The names of the conventions, as the CLI, Python and the qccodec codec parse them.
#[test]
fn conventions_names() {
    for (name, conv) in [
        ("axtb", Conventions::Axtb),
        ("xtb", Conventions::Xtb),
        ("xtb-6.8.0", Conventions::Xtb),
        ("xtb-6.7.1", Conventions::Xtb671),
        ("XTB-6.7.1", Conventions::Xtb671),
        ("tblite", Conventions::Tblite),
    ] {
        assert_eq!(name.parse::<Conventions>().unwrap(), conv);
    }
    for conv in [Conventions::Axtb, Conventions::Xtb, Conventions::Xtb671, Conventions::Tblite] {
        assert_eq!(conv.label().parse::<Conventions>().unwrap(), conv);
    }
    assert!("xtb-6.7".parse::<Conventions>().is_err());
}
