//! Energy/gradient consistency and invariances of the default (axtb) conventions, which
//! have no external reference program: analytic gradients against central finite
//! differences, and invariance under atom permutation.

use axtb::{io::read_xyz, Conventions, Molecule, Options, SolvationModel, SolvationSpec};

fn opts(conventions: Conventions, alpb: bool) -> Options {
    let mut o = Options { conventions, ..Default::default() };
    o.scf.accuracy = 1e-8;
    if alpb {
        o.solvation = Some(SolvationSpec {
            model: SolvationModel::Alpb,
            solvent: "water".into(),
            state: axtb::ReferenceState::Gsolv,
        });
    }
    o
}

fn energy(mol: &Molecule, o: &Options) -> f64 {
    axtb::singlepoint(axtb::Hamiltonian::Gfn2, mol, o, false).unwrap().energy
}

fn check_fd(xyz: &str, alpb: bool, atoms: &[usize], tol: f64) {
    let (num, pos) = read_xyz(xyz, axtb::units::AATOAU).unwrap();
    let mol = Molecule::new(&num, &pos, 0.0, None).unwrap();
    let o = opts(Conventions::Axtb, alpb);
    let g = axtb::singlepoint(axtb::Hamiltonian::Gfn2, &mol, &o, true).unwrap().gradient.unwrap();
    let h = 1e-4;
    for &i in atoms {
        for k in 0..3 {
            let mut m = mol.clone();
            m.positions[i][k] += h;
            let ep = energy(&m, &o);
            m.positions[i][k] -= 2.0 * h;
            let em = energy(&m, &o);
            let fd = (ep - em) / (2.0 * h);
            assert!((fd - g[i][k]).abs() < tol, "atom {i} dir {k}: analytic {} fd {fd} (alpb {alpb})", g[i][k]);
        }
    }
}

#[test]
fn gradient_matches_finite_differences_gas() {
    check_fd(include_str!("../../../tests/structures/caffeine.xyz"), false, &[0, 7, 15, 23], 2e-7);
}

#[test]
fn gradient_matches_finite_differences_alpb() {
    check_fd(include_str!("../../../tests/structures/caffeine.xyz"), true, &[0, 7, 15, 23], 2e-7);
}

#[test]
fn gradient_matches_finite_differences_transition_metal() {
    check_fd(include_str!("../../../tests/structures/ferrocene.xyz"), false, &[0, 1, 2], 2e-7);
}

/// The ALPB energy of compressed CsOH must not depend on the atom order in the axtb
/// conventions and xtb 6.8.0's (xtb 6.7.1's stale `r1`, grimme-lab/xtb#1504, makes it differ
/// by 2.9e-3 Eh).
#[test]
fn alpb_energy_is_atom_order_invariant() {
    let aatoau = axtb::units::AATOAU;
    let atoms = [(1usize, [0.0, 0.0, -0.96]), (8, [0.0, 0.0, 0.0]), (55, [0.0, 0.0, 2.08])];
    let build = |order: [usize; 3]| {
        let num: Vec<usize> = order.iter().map(|&i| atoms[i].0).collect();
        let pos: Vec<[f64; 3]> = order.iter().map(|&i| atoms[i].1.map(|x| x * aatoau)).collect();
        Molecule::new(&num, &pos, 0.0, None).unwrap()
    };
    let o = opts(Conventions::Axtb, true);
    let (a, b) = (energy(&build([0, 1, 2]), &o), energy(&build([2, 1, 0]), &o));
    assert!((a - b).abs() < 1e-9, "axtb conventions: {a} vs {b}");
    let o = opts(Conventions::Xtb, true);
    let (xa, xb) = (energy(&build([0, 1, 2]), &o), energy(&build([2, 1, 0]), &o));
    assert!((xa - xb).abs() < 1e-9, "xtb (6.8.0) conventions: {xa} vs {xb}");
    let o = opts(Conventions::Xtb671, true);
    let (xa, xb) = (energy(&build([0, 1, 2]), &o), energy(&build([2, 1, 0]), &o));
    assert!((xa - xb).abs() > 1e-4, "xtb-6.7.1 conventions reproduce xtb 6.7.1's order dependence: {xa} vs {xb}");
}

/// The default conventions differ from exact xtb reproduction only by the corrected
/// constants and cutoff switching: ~1e-8 Eh for an ordinary organic molecule.
#[test]
fn axtb_and_xtb_conventions_agree_closely() {
    let (num, pos) = read_xyz(include_str!("../../../tests/structures/caffeine.xyz"), axtb::units::AATOAU).unwrap();
    let mol = Molecule::new(&num, &pos, 0.0, None).unwrap();
    for xtb in [Conventions::Xtb, Conventions::Xtb671] {
        let (a, x) = (energy(&mol, &opts(Conventions::Axtb, false)), energy(&mol, &opts(xtb, false)));
        assert!((a - x).abs() < 1e-6, "axtb {a} vs {xtb} {x}: {:.2e}", a - x);
        eprintln!("caffeine: axtb - {xtb} conventions = {:.3e} Eh", a - x);
    }
}
