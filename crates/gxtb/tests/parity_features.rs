//! The g-xTB program features of `docs/gxtb-feature-coverage.md`: SCF guess reuse
//! (the programs' restart files), Mayer-Wiberg bond orders, the HOMO-LUMO gap and the molecular
//! quadrupole, against numbers of the programs (g-xTB 2.0.1 = `xtb --gxtb`, 6.7.1 build; g-xTB
//! 1.1.0) on `tests/structures/water.xyz` and triplet O2; `tests/gxtb_parity_features.py`
//! compares more molecules (`bench/results/gxtb_parity_features.json`).

use axtb::method::{Method, Restart, ScfSettings, SinglePoint, Want};
use axtb_gxtb::{v2, Calculator, Conventions, Options, Version};

const WATER: &str = include_str!("../../../tests/structures/water.xyz");
const O2: &str = "2\nO2\nO 0.0 0.0 0.0\nO 0.0 0.0 1.21\n";
/// mctc-lib's (CODATA 2018) Hartree in eV, which tblite prints the gap with (the 1.1.0 program:
/// 27.212 in single precision)
const AUTOEV: f64 = 27.211386245988;

fn calc(version: Version, xyz: &str, uhf: Option<usize>, opts: Options) -> (Calculator, Vec<[f64; 3]>) {
    let mol = match version {
        Version::V2_0_1 => v2::read_xyz(xyz, 0.0, uhf, opts.conventions).unwrap(),
        Version::V1_1 => axtb_gxtb::read_xyz(xyz, 0.0, uhf).unwrap(),
    };
    let c = Calculator::new(version, &mol.numbers, &mol.positions, 0.0, uhf, opts).unwrap();
    (c, mol.positions)
}

fn program() -> Options {
    Options { conventions: Conventions::Gxtb, ..Default::default() }
}

/// Options with the restart policy `restart`.
fn restart(restart: Restart, o: Options) -> Options {
    Options { scf: ScfSettings { restart: Some(restart), ..o.scf.clone() }, ..o }
}

/// The single point with the bond orders.
const BOND_ORDERS: Want = Want { bond_orders: true, ..Want::ENERGY };

fn bond_orders(sp: &SinglePoint) -> &[f64] {
    sp.electronic.as_ref().unwrap().bond_orders.as_ref().unwrap()
}

fn gap(sp: &SinglePoint) -> f64 {
    sp.electronic.as_ref().unwrap().homo_lumo_gap.unwrap()
}

#[test]
fn bond_orders_gap_quadrupole_v2_water() {
    let (mut c, pos) = calc(Version::V2_0_1, WATER, None, program());
    let sp = c.singlepoint(&pos, &BOND_ORDERS).unwrap();
    let bo = bond_orders(&sp);
    // the program's `wbo` file (atoms 1-3 and 2-3, O is atom 3)
    assert!((bo[2] - 0.898496897498985).abs() < 1e-12, "{}", bo[2]);
    assert!((bo[5] - 0.898498124204562).abs() < 1e-12, "{}", bo[5]);
    assert_eq!(bo[2], bo[6]);
    assert_eq!(bo[0], 0.0);
    // "HOMO-LUMO gap 19.474052577788 eV"
    assert!((gap(&sp) * AUTOEV - 19.474052577788).abs() < 1e-9);
    // `--quadrupole`: total 0.4394 0.0000 -1.3541 1.2245 -0.0000 0.9147 (tblite's form)
    let e = sp.electronic.as_ref().unwrap();
    let q = e.quadrupole;
    for (a, b) in q.iter().zip([0.4394, 0.0, -1.3541, 1.2245, 0.0, 0.9147]) {
        assert!((a - b).abs() <= 5e-5 + 1e-12, "{q:?}");
    }
    let form = c.info().quadrupole.unwrap();
    assert_eq!(form, axtb::QuadrupoleForm::Tblite);
    let q2 = axtb::properties::molecular_quadrupole(&pos, &sp.charges, &e.atomic_dipoles, &e.atomic_quadrupoles, form);
    assert_eq!(q, q2);
    assert_eq!(sp.scf().unwrap().start, "eeq-bc");
    // one channel, doubly occupied
    assert_eq!(e.levels.len(), 1);
    assert_eq!(e.levels[0].occupations.iter().sum::<f64>().round(), 8.0);
}

#[test]
fn bond_orders_open_shell_v2_and_v1() {
    // 2.0.1: 2 (P^a S . P^a S + P^b S . P^b S), the program's `wbo`: 1.99999999995016
    let (mut c, pos) = calc(Version::V2_0_1, O2, Some(2), program());
    let sp = c.singlepoint(&pos, &BOND_ORDERS).unwrap();
    assert!((bond_orders(&sp)[1] - 1.99999999995016).abs() < 1e-11);
    // the alpha gap: 16.531940680325 eV
    assert!((gap(&sp) * AUTOEV - 16.531940680325).abs() < 1e-8);
    assert_eq!(sp.electronic.as_ref().unwrap().levels.len(), 2);
    // v1.1 (the 1.1.0 program prints 2.000 and "gap (eV) alpha->alpha : 17.17858483")
    let (mut c, pos) = calc(Version::V1_1, O2, Some(2), program());
    let sp = c.singlepoint(&pos, &BOND_ORDERS).unwrap();
    assert!((bond_orders(&sp)[1] - 2.000).abs() < 5e-4);
    assert!((gap(&sp) * (27.212_f32 as f64) - 17.17858483).abs() < 1e-6);
}

#[test]
fn bond_orders_v1_water() {
    // the 1.1.0 program prints 0.893 for both O-H pairs, gap 19.23283345 eV
    let (mut c, pos) = calc(Version::V1_1, WATER, None, program());
    let sp = c.singlepoint(&pos, &BOND_ORDERS).unwrap();
    let bo = bond_orders(&sp);
    assert!((bo[2] - 0.893).abs() < 5e-4 && (bo[5] - 0.893).abs() < 5e-4, "{bo:?}");
    assert!((gap(&sp) * (27.212_f32 as f64) - 19.23283345).abs() < 1e-6);
}

#[test]
fn restart_emulates_the_programs() {
    // a second run in the same directory reads the restart file: 2.0.1 two SCF cycles (water:
    // the program's first run 9), v1.1 three (the 1.1.0 program: 9, then 3)
    for (version, first, again) in [(Version::V2_0_1, 9, 2), (Version::V1_1, 9, 3)] {
        let (mut c, pos) = calc(version, WATER, None, restart(Restart::Previous, program()));
        let a = c.singlepoint(&pos, &Want::gradient(false)).unwrap();
        let b = c.singlepoint(&pos, &Want::gradient(false)).unwrap();
        let (sa, sb) = (a.scf().unwrap(), b.scf().unwrap());
        assert_eq!((sa.start, sb.start), ("eeq-bc", "previous"));
        assert_eq!((sa.iterations, sb.iterations), (first, again), "{version:?}");
        assert!((a.energy - b.energy).abs() < 1e-6);
        // program conventions without the opt-in: every SCF from the program's start
        let (mut c, pos) = calc(version, WATER, None, program());
        let a2 = c.singlepoint(&pos, &Want::gradient(false)).unwrap();
        let b2 = c.singlepoint(&pos, &Want::gradient(false)).unwrap();
        let (sa2, sb2) = (a2.scf().unwrap(), b2.scf().unwrap());
        assert_eq!((sb2.start, sb2.iterations, b2.energy), ("eeq-bc", sa2.iterations, a2.energy));
        assert_eq!(a2.energy, a.energy);
    }
}

#[test]
fn default_conventions_reuse_along_a_path() {
    // caffeine-sized enough to matter, small enough for a test: ethanol from the reference set
    let xyz = include_str!("reference/ethanol/mol.xyz");
    for version in [Version::V2_0_1, Version::V1_1] {
        let (mut reuse, pos) = calc(version, xyz, None, Options::default());
        let (mut fresh, _) = calc(version, xyz, None, restart(Restart::Fresh, Options::default()));
        assert!(reuse.reuses_guess() && !fresh.reuses_guess());
        let (mut it_reuse, mut it_fresh) = (0, 0);
        for step in 0..6 {
            let p: Vec<[f64; 3]> = pos
                .iter()
                .enumerate()
                .map(|(i, x)| {
                    let s = 0.01 * step as f64;
                    [x[0] + s * ((i % 3) as f64 - 1.0), x[1] + 0.5 * s, x[2] - s * ((i % 2) as f64)]
                })
                .collect();
            let a = reuse.singlepoint(&p, &Want::gradient(true)).unwrap();
            let b = fresh.singlepoint(&p, &Want::gradient(true)).unwrap();
            assert_eq!(a.scf().unwrap().start, if step == 0 { "eeq-bc" } else { "previous" });
            assert_eq!(b.scf().unwrap().start, "eeq-bc");
            // the same SCF solution to the convergence thresholds
            assert!((a.energy - b.energy).abs() < 2e-6, "{version:?} step {step}: {}", a.energy - b.energy);
            let dg = a
                .gradient
                .as_ref()
                .unwrap()
                .iter()
                .zip(b.gradient.as_ref().unwrap())
                .map(|(x, y)| (0..3).map(|k| (x[k] - y[k]).abs()).fold(0.0, f64::max))
                .fold(0.0, f64::max);
            assert!(dg < 1e-4, "{version:?} step {step}: {dg}");
            if step > 0 {
                it_reuse += a.scf().unwrap().iterations;
                it_fresh += b.scf().unwrap().iterations;
            }
        }
        assert!(it_reuse < it_fresh, "{version:?}: {it_reuse} vs {it_fresh}");
        // a state handed to another calculator for the same molecule
        let state = reuse.wavefunction().unwrap();
        fresh.set_wavefunction(&state).unwrap();
        let p = reuse.molecule().positions.clone();
        assert_eq!(fresh.singlepoint(&p, &Want::gradient(false)).unwrap().scf().unwrap().start, "previous");
        assert_eq!(fresh.singlepoint(&p, &Want::gradient(false)).unwrap().scf().unwrap().start, "eeq-bc");
    }
}

#[test]
fn reuse_is_thread_independent() {
    let xyz = include_str!("reference/ethanol/mol.xyz");
    let run = |threads: usize| {
        let (mut c, pos) = calc(Version::V2_0_1, xyz, None, Options { threads, ..Default::default() });
        (0..3)
            .map(|k| {
                let p: Vec<[f64; 3]> =
                    pos.iter().map(|x| [x[0] + 0.02 * k as f64, x[1], x[2] - 0.01 * k as f64 * x[0]]).collect();
                let sp = c.singlepoint(&p, &Want::gradient(true)).unwrap();
                (sp.energy, sp.scf().unwrap().iterations, sp.gradient.unwrap())
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(run(1), run(3));
}
