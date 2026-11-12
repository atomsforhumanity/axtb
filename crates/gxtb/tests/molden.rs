//! Orbitals for Molden files from the calculators (GFN2-xTB, g-xTB v1.1 and 2.0.1): kept only
//! when asked, orthonormal in the per-atom basis they come with (so the coefficients are laid
//! out orbital by orbital in axtb's AO order and each atom has its own charge-adapted shells),
//! one set for closed shells and alpha and beta sets for open shells, and written as Molden
//! files with the expected sections. The reading back and the comparison with the programs'
//! files are in `crates/axtb-py/tests/test_molden.py` and `tests/molden_check.py`.

use axtb::basis::{nsph, Cgto};
use axtb::integrals::{overlap, MBLK};
use axtb::method::{Method, Want};
use axtb::molden::{MoldenFunctions, Spin, ToMolden};
use axtb::{Molecule, OrbitalLevels, Orbitals};
use axtb_gxtb::{Calculator, Options, Version};
use std::path::Path;

fn molecule(rel: &str, charge: f64, uhf: usize) -> Molecule {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/reference_v2").join(rel).join("mol.xyz");
    axtb_gxtb::v2::read_xyz(&std::fs::read_to_string(path).unwrap(), charge, Some(uhf), Default::default()).unwrap()
}

/// Largest deviation of C^T S C from the unit matrix over the sets of `orb`, with S from the
/// orbitals' own shells.
fn orthonormality(orb: &Orbitals) -> f64 {
    let shells: Vec<(usize, &Cgto)> =
        orb.shells.iter().enumerate().flat_map(|(a, s)| s.iter().map(move |c| (a, c))).collect();
    let nao = orb.nao();
    let mut s = vec![0.0; nao * nao];
    let mut blk = [0.0; MBLK];
    let mut i0 = 0;
    for &(a, ca) in &shells {
        let mut j0 = 0;
        for &(b, cb) in &shells {
            let v = [0, 1, 2].map(|k| orb.positions[a][k] - orb.positions[b][k]);
            overlap(cb, ca, v, 1e3, &mut blk);
            for i in 0..nsph(ca.l) {
                for j in 0..nsph(cb.l) {
                    s[(i0 + i) * nao + j0 + j] = blk[j + nsph(cb.l) * i];
                }
            }
            j0 += nsph(cb.l);
        }
        i0 += nsph(ca.l);
    }
    let mut worst: f64 = 0.0;
    for set in &orb.sets {
        let c = &set.coefficients;
        let nmo = set.energies.len();
        let sc: Vec<f64> = (0..nmo)
            .flat_map(|k| (0..nao).map(move |m| (k, m)))
            .map(|(k, m)| (0..nao).map(|n| s[m * nao + n] * c[k * nao + n]).sum())
            .collect();
        for k in 0..nmo {
            for l in 0..nmo {
                let v: f64 = (0..nao).map(|m| c[k * nao + m] * sc[l * nao + m]).sum();
                worst = worst.max((v - if k == l { 1.0 } else { 0.0 }).abs());
            }
        }
    }
    worst
}

/// The orbitals `orb` of a single point with the orbital levels `levels` (per spin channel);
/// `one_set`: GFN2-xTB's one set of orbitals with alpha and beta occupations.
fn check(name: &str, orb: &Orbitals, levels: &[OrbitalLevels], uhf: usize, one_set: bool) {
    let nao = orb.nao();
    let err = orthonormality(orb);
    // (the SCF overlap screens primitive pairs, this one does not)
    assert!(err < 1e-8, "{name}: orbitals not orthonormal ({err:e})");
    assert_eq!(orb.sets.len(), if uhf > 0 { 2 } else { 1 }, "{name}");
    assert_eq!(orb.sets[0].spin, Spin::Alpha);
    assert_eq!(orb.sets.len(), levels.len(), "{name}");
    for (set, level) in orb.sets.iter().zip(levels) {
        assert_eq!(set.energies, level.energies, "{name}");
        assert_eq!(set.occupations, level.occupations, "{name}");
    }
    if one_set && orb.sets.len() == 2 {
        assert_eq!(orb.sets[0].coefficients, orb.sets[1].coefficients);
    }
    let nalpha: f64 = orb.sets[0].occupations.iter().sum();
    if uhf > 0 {
        let nbeta: f64 = orb.sets[1].occupations.iter().sum();
        assert!((nalpha - nbeta - uhf as f64).abs() < 1e-6, "{name}");
    }
    let maxl = orb.shells.iter().flatten().map(|s| s.l).max().unwrap();
    for functions in [MoldenFunctions::Spherical, MoldenFunctions::Cartesian] {
        let text = orb.to_molden(functions, name).unwrap();
        assert!(text.starts_with(&format!("[Molden Format]\n[Title]\n{name}\n[Atoms] AU\n")));
        assert_eq!(text.matches("Sym=").count(), nao * orb.sets.len());
        assert_eq!(text.matches("Spin= Beta").count(), if uhf > 0 { nao } else { 0 });
        let ncoef = match functions {
            MoldenFunctions::Spherical => nao,
            MoldenFunctions::Cartesian => orb.shells.iter().flatten().map(|s| (s.l + 1) * (s.l + 2) / 2).sum(),
        };
        let mo = &text[text.find("[MO]").unwrap()..];
        let first: Vec<&str> = mo.lines().skip(5).take_while(|l| !l.starts_with("Sym=")).collect();
        assert_eq!(first.len(), ncoef, "{name} {functions:?}");
        let flags = match functions {
            MoldenFunctions::Spherical => [(maxl >= 3, "[7F]"), (maxl >= 2, "[5D]")],
            MoldenFunctions::Cartesian => [(maxl >= 2, "[6D]"), (maxl >= 3, "[10F]")],
        };
        for (want, flag) in flags {
            assert_eq!(text.contains(flag), want, "{name} {flag}");
        }
    }
}

#[test]
fn gxtb_orbitals() {
    let cases = [
        ("reference/water", 0.0, 0),
        ("reference/dioxygen_triplet", 0.0, 2),
        ("reference/tetrachlorocuprate", -2.0, 1),
        ("elements/UF6", 0.0, 0),
        ("elements/CeCl3", 0.0, 1),
    ];
    for version in [Version::V1_1, Version::V2_0_1] {
        for (rel, charge, uhf) in cases {
            let mol = molecule(rel, charge, uhf);
            let name = format!("{rel} {}", version.method());
            let mut calc =
                Calculator::new(version, &mol.numbers, &mol.positions, charge, Some(uhf), Options::default()).unwrap();
            let r = calc.singlepoint(&mol.positions, &Want::gradient(false)).unwrap();
            assert!(r.electronic.unwrap().orbitals.is_none(), "orbitals only when asked");
            let want = Want { orbitals: true, ..Want::GRADIENT };
            let e = calc.singlepoint(&mol.positions, &want).unwrap().electronic.unwrap();
            let orb = e.orbitals.as_ref().expect("orbitals");
            check(&name, orb, &e.levels, uhf, false);
        }
    }
}

#[test]
fn gfn2_orbitals() {
    let cases = [("reference/water", 0.0, 0), ("reference/dioxygen_triplet", 0.0, 2), ("reference/cu-nh3-4", 2.0, 1)];
    for (rel, charge, uhf) in cases {
        let mol = molecule(rel, charge, uhf);
        let opts = axtb::Options::default();
        let mut calc = axtb::Calculator::new(
            axtb::Hamiltonian::Gfn2,
            &mol.numbers,
            &mol.positions,
            charge,
            Some(uhf),
            opts.clone(),
        )
        .unwrap();
        assert!(calc.singlepoint(&mol.positions, &Want::ENERGY).unwrap().electronic.unwrap().orbitals.is_none());
        let want = Want { orbitals: true, ..Want::ENERGY };
        let e = axtb::singlepoint(axtb::Hamiltonian::Gfn2, &mol, &opts, &want).unwrap().electronic.unwrap();
        check(&format!("{rel} gfn2"), e.orbitals.as_ref().unwrap(), &e.levels, uhf, true);
    }
}

#[test]
fn gxtb_shells_per_atom() {
    // q-vSZP adapts every atom's contraction to its charge and coordination: in ethanol the
    // hydroxyl H and the methyl H get different s shells, and the orbitals come with them
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/reference/ethanol/mol.xyz");
    let mol = axtb_gxtb::read_xyz(&std::fs::read_to_string(path).unwrap(), 0.0, None).unwrap();
    for version in [Version::V1_1, Version::V2_0_1] {
        let mut calc = Calculator::new(version, &mol.numbers, &mol.positions, 0.0, None, Options::default()).unwrap();
        let want = Want { orbitals: true, ..Want::ENERGY };
        let orb = calc.singlepoint(&mol.positions, &want).unwrap().electronic.unwrap().orbitals.unwrap();
        let h: Vec<usize> = (0..mol.nat()).filter(|&i| mol.numbers[i] == 1).collect();
        let distinct: std::collections::BTreeSet<String> =
            h.iter().map(|&i| format!("{:?}", orb.shells[i][0].coeff)).collect();
        assert!(distinct.len() > 1, "{}: all H atoms have the same shells", version.method());
        let text = orb.to_molden(MoldenFunctions::Spherical, "ethanol").unwrap();
        let gto = &text[text.find("[GTO]").unwrap()..text.find("[MO]").unwrap()];
        assert_eq!(gto.matches(" 0\n").count(), mol.nat(), "one [GTO] block per atom");
    }
}
