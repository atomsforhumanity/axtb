//! SCF mixers. The axtb conventions run Broyden and fall back to DIIS from the initial guess
//! when Broyden does not converge (`auto`): results Broyden converges are exactly Broyden's,
//! hard cases of `tests/scf_hard_set.py` converge through the fallback, and the result says
//! which mixer produced it. The xtb and tblite conventions keep Broyden alone.

use axtb::units::AATOAU;
use axtb::{io::read_xyz, Calculator, Conventions, Method, MixerKind, Molecule, Options, ScfAttempt};

const WATER_3A: &str = include_str!("../../../tests/scf_hard/scan-H2O-3.0.xyz");
const CAFFEINE: &str = include_str!("../../../tests/structures/caffeine.xyz");

fn options(conv: Conventions, mixer: Option<MixerKind>, acc: f64) -> Options {
    let mut o = Options { conventions: conv, ..Default::default() };
    o.scf.mixer = mixer;
    o.scf.accuracy = acc;
    o
}

fn singlepoint(xyz: &str, conv: Conventions, mixer: Option<MixerKind>, acc: f64) -> axtb::Result<axtb::SinglePoint> {
    let (num, pos) = read_xyz(xyz, AATOAU).unwrap();
    let mol = Molecule::new(&num, &pos, 0.0, Some(0)).unwrap();
    axtb::singlepoint(Method::Gfn2, &mol, &options(conv, mixer, acc), true)
}

fn same_bits(a: &axtb::SinglePoint, b: &axtb::SinglePoint) -> bool {
    let g =
        |s: &axtb::SinglePoint| s.gradient.as_ref().unwrap().iter().flatten().map(|x| x.to_bits()).collect::<Vec<_>>();
    a.energy.to_bits() == b.energy.to_bits() && g(a) == g(b) && a.iterations == b.iterations
}

#[test]
fn auto_is_broyden_where_broyden_converges() {
    for xyz in [CAFFEINE, include_str!("../../../tests/structures/ferrocene.xyz")] {
        let b = singlepoint(xyz, Conventions::Axtb, Some(MixerKind::Broyden), 1e-6).unwrap();
        let a = singlepoint(xyz, Conventions::Axtb, None, 1e-6).unwrap();
        assert!(same_bits(&a, &b));
        assert_eq!(a.mixer_used, Some("broyden"));
        assert_eq!(
            a.scf_attempts,
            vec![ScfAttempt { mixer: MixerKind::Broyden, iterations: b.iterations, converged: true }]
        );
        // DIIS alone lands on the same solution
        let d = singlepoint(xyz, Conventions::Axtb, Some(MixerKind::Diis), 1e-6).unwrap();
        assert!((b.energy - d.energy).abs() < 1e-10, "{} vs {}", b.energy, d.energy);
        assert_eq!(d.mixer_used, Some("diis"));
    }
}

/// Singlet water with both O-H bonds at 3 Å, 300 K: tblite's Broyden mixing oscillates
/// between two charge distributions for 250 iterations; DIIS from the initial guess
/// converges to the solution the xtb program also finds (after 239 iterations;
/// -4.60802310917 Eh in its conventions).
#[test]
fn auto_falls_back_to_diis_on_stretched_water() {
    assert!(matches!(
        singlepoint(WATER_3A, Conventions::Axtb, Some(MixerKind::Broyden), 1e-4),
        Err(axtb::Error::ScfNotConverged { .. })
    ));
    let d = singlepoint(WATER_3A, Conventions::Axtb, Some(MixerKind::Diis), 1e-4).unwrap();
    let a = singlepoint(WATER_3A, Conventions::Axtb, None, 1e-4).unwrap();
    assert!(same_bits(&a, &d));
    assert!((a.energy - -4.6080231096).abs() < 1e-8, "{}", a.energy);
    assert_eq!(a.mixer_used, Some("diis-fallback"));
    assert_eq!(
        a.scf_attempts,
        vec![
            ScfAttempt { mixer: MixerKind::Broyden, iterations: 250, converged: false },
            ScfAttempt { mixer: MixerKind::Diis, iterations: d.iterations, converged: true },
        ]
    );
    // the program conventions keep Broyden alone, failures included: like the programs,
    // xtb conventions converge this case with Broyden, tblite conventions do not
    for xtb in [Conventions::Xtb, Conventions::Xtb671] {
        let x = singlepoint(WATER_3A, xtb, None, 1e-4).unwrap();
        assert_eq!((x.mixer_used, x.scf_attempts.len()), (Some("broyden"), 1));
    }
    assert!(matches!(
        singlepoint(WATER_3A, Conventions::Tblite, None, 1e-4),
        Err(axtb::Error::ScfNotConverged { iterations: 250, .. })
    ));
    // both failing: the error counts the iterations of both attempts
    let (num, pos) = read_xyz(WATER_3A, AATOAU).unwrap();
    let mol = Molecule::new(&num, &pos, 0.0, Some(0)).unwrap();
    let mut o = options(Conventions::Axtb, None, 1e-4);
    o.scf.max_iter = 5;
    assert!(matches!(
        axtb::singlepoint(Method::Gfn2, &mol, &o, false),
        Err(axtb::Error::ScfNotConverged { iterations: 10, .. })
    ));
}

/// A rescued point along a path is visible in the calculator's results.
#[test]
fn calculator_reports_the_mixer_per_point() {
    let (num, pos) = read_xyz(CAFFEINE, AATOAU).unwrap();
    let mut calc =
        Calculator::new(Method::Gfn2, &num, &pos, 0.0, None, options(Conventions::Axtb, None, 1e-4)).unwrap();
    assert_eq!(calc.singlepoint(&pos, false).unwrap().mixer_used, Some("broyden"));
    let (num, pos) = read_xyz(WATER_3A, AATOAU).unwrap();
    let mut calc =
        Calculator::new(Method::Gfn2, &num, &pos, 0.0, Some(0), options(Conventions::Axtb, None, 1e-4)).unwrap();
    let r = calc.singlepoint(&pos, false).unwrap();
    assert_eq!(r.mixer_used, Some("diis-fallback"));
    assert_eq!(r.scf_attempts.len(), 2);
}

#[test]
fn conventions_choose_the_mixer() {
    assert_eq!(Conventions::Axtb.params().mixer, MixerKind::Auto);
    assert_eq!(Conventions::Xtb.params().mixer, MixerKind::Broyden);
    assert_eq!(Conventions::Xtb671.params().mixer, MixerKind::Broyden);
    assert_eq!(Conventions::Tblite.params().mixer, MixerKind::Broyden);
    for conv in [Conventions::Xtb, Conventions::Xtb671, Conventions::Tblite] {
        let a = singlepoint(CAFFEINE, conv, None, 1e-6).unwrap();
        let b = singlepoint(CAFFEINE, conv, Some(MixerKind::Broyden), 1e-6).unwrap();
        assert!(same_bits(&a, &b));
        assert_eq!(a.mixer_used, Some("broyden"));
    }
    assert_eq!("DIIS".parse::<MixerKind>().unwrap(), MixerKind::Diis);
    assert_eq!("pulay".parse::<MixerKind>().unwrap(), MixerKind::Diis);
    assert_eq!("broyden".parse::<MixerKind>().unwrap(), MixerKind::Broyden);
    assert_eq!("Auto".parse::<MixerKind>().unwrap(), MixerKind::Auto);
    assert!("anderson2".parse::<MixerKind>().is_err());
}
