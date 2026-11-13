// SPDX-License-Identifier: LGPL-3.0-or-later
//! The xtb 6.8.0 conventions with one setting at a time put back to its xtb 6.7.1 value
//! (`tests/xtb680_changes.py` compares each with the xtb 6.8.0 program): how much each
//! documented 6.7.1 quirk would move the result.
//!
//! Input on stdin (bohr): a line `charge uhf accuracy etemp solvation`, where `uhf` is `-`
//! for the default and `solvation` is `-` or `alpb:water:gsolv`/`gbsa:...`; a line with the
//! number of atoms; one line `Z x y z` per atom. Output: one line per variant,
//! `name energy iterations g1x g1y g1z g2x ...` (or `name error message`).
//!
//!     cargo run --release -p axtb --example conventions_ablation < mol.txt

use axtb::conventions::{ConventionParams, SccGradient, SccResidual};
use axtb::method::Want;
use axtb::param::xtb_data::{
    XTB671_ATOMIC_RAD, XTB671_COVALENT_RAD_D3, XTB671_D4_EN, XTB671_D4_RCOV, XTB671_PAULING_EN,
};
use axtb::{Conventions, Hamiltonian, Molecule, Options, ReferenceState, SolvationModel, SolvationSpec};
use std::io::Read;

type Variant = (&'static str, fn(&mut ConventionParams));

const VARIANTS: [Variant; 12] = [
    ("xtb-6.8.0", |_| {}),
    ("radii: 6.7.1 tables (older bohr constant)", |p| {
        p.atomic_rad = |z| XTB671_ATOMIC_RAD[z - 1];
        p.cn_rcov = |z| XTB671_COVALENT_RAD_D3[z - 1];
        p.pauling_en = |z| XTB671_PAULING_EN[z - 1];
    }),
    ("D4 CN: 6.7.1 single-precision literals and older bohr constant", |p| {
        p.d4_rcov = |z| XTB671_D4_RCOV[z - 1];
        p.d4_en = |z| XTB671_D4_EN[z - 1];
        p.eeq_guess_cn.rcov = |z| XTB671_D4_RCOV[z - 1];
    }),
    ("cutoffs: 6.7.1 (CN, repulsion 40 bohr; D4 40/72 bohr hard)", |p| {
        p.cn_cutoff = 40.0;
        p.repulsion_cutoff = 40.0;
        p.d4_cutoff = axtb_d4::RealspaceCutoff::XTB_GFN2;
    }),
    ("integral screening: 6.7.1 (max(20, 25 - 10 log10 acc))", |p| {
        p.primitive_cutoff = axtb::basis::integral_cutoff_xtb
    }),
    ("solvation: 6.7.1 (stale r1, rho/M reference state, older bohr)", |p| {
        p.solvation = axtb_solvation::Conventions::Xtb671
    }),
    ("SCC 2: 6.7.1 residual test (per atom, 1e-4 acc)", |p| p.scc_residual = SccResidual::PerAtom),
    ("SCC 3: 6.7.1 extra final iteration", |p| p.scc_extra_iteration = true),
    ("SCC 4: 6.7.1 mixer metric (quadrupoles unscaled)", |p| p.scc_quadrupole_scale = 1.0),
    ("SCC 5: 6.7.1 neglected multipole integrals", |p| p.scc_multipole_neglect = true),
    ("SCC 6: 6.7.1 gradient from mixed moments", |p| p.scc_gradient = SccGradient::Mixed),
    ("SCC 6: tblite's gradient potentials", |p| p.scc_gradient = SccGradient::Output),
];

fn main() {
    let mut text = String::new();
    std::io::stdin().read_to_string(&mut text).unwrap();
    let mut lines = text.lines();
    let head: Vec<&str> = lines.next().unwrap().split_whitespace().collect();
    let charge: f64 = head[0].parse().unwrap();
    let uhf: Option<usize> = if head[1] == "-" { None } else { Some(head[1].parse().unwrap()) };
    let accuracy: f64 = head[2].parse().unwrap();
    let etemp: f64 = head[3].parse().unwrap();
    let solvation = if head[4] == "-" {
        None
    } else {
        let f: Vec<&str> = head[4].split(':').collect();
        Some(SolvationSpec {
            model: if f[0] == "alpb" { SolvationModel::Alpb } else { SolvationModel::Gbsa },
            solvent: f[1].to_string(),
            state: f.get(2).map_or(ReferenceState::Gsolv, |s| SolvationSpec::parse_state(s).unwrap()),
        })
    };
    let nat: usize = lines.next().unwrap().trim().parse().unwrap();
    let (mut numbers, mut positions) = (Vec::new(), Vec::new());
    for l in lines.take(nat) {
        let f: Vec<f64> = l.split_whitespace().map(|x| x.parse().unwrap()).collect();
        numbers.push(f[0] as usize);
        positions.push([f[1], f[2], f[3]]);
    }
    let mol = Molecule::new(&numbers, &positions, charge, uhf).unwrap();
    for (name, modify) in VARIANTS {
        let mut params = Conventions::Xtb.params();
        modify(&mut params);
        let mut opts = Options {
            conventions: Conventions::Xtb,
            params: Some(params),
            solvation: solvation.clone(),
            ..Default::default()
        };
        opts.scf.accuracy = accuracy;
        opts.scf.etemp = Some(etemp);
        match axtb::singlepoint(Hamiltonian::Gfn2, &mol, &opts, &Want::gradient(true)) {
            Ok(sp) => {
                let g: Vec<String> =
                    sp.gradient.as_ref().unwrap().iter().flatten().map(|x| format!("{x:.17e}")).collect();
                println!("{name}\t{:.17e}\t{}\t{}", sp.energy, sp.scf().unwrap().iterations, g.join(" "));
            }
            Err(e) => println!("{name}\terror\t{e}"),
        }
    }
}
