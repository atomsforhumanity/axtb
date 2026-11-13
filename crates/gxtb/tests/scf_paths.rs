//! SCF paths against the g-xTB programs where they are decided by round-off or by a switch of
//! the program's mixer (data in tests/reference_elements/ and tests/reference_v2/elements/,
//! see tests/gxtb_elements.py and docs/gxtb-v1.1-spec.md §4.17).

use axtb_gxtb::scf::{Conventions, Setup};
use faer::{Mat, MatRef, Par};
use serde_json::Value;
use std::path::{Path, PathBuf};

fn tests() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests")
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

/// Program AO index -> our AO index: p as x, y, z; d as (x2-y2, z2, xy, xz, yz); f as ours.
fn program_ao_map(ao: &[(usize, usize, usize)]) -> Vec<usize> {
    let mut map = Vec::with_capacity(ao.len());
    let mut k = 0;
    while k < ao.len() {
        let l = ao[k].2;
        match l {
            0 => map.push(k),
            1 => map.extend([k + 2, k, k + 1]),
            2 => map.extend([4, 2, 0, 3, 1].iter().map(|&m| k + m)),
            _ => map.extend(k..k + 2 * l + 1),
        }
        k += 2 * l + 1;
    }
    map
}

/// A packed lower triangle in the program's AO order -> full matrix in ours.
fn unpack(packed: &[f64], map: &[usize]) -> Vec<f64> {
    let n = map.len();
    let mut m = vec![0.0; n * n];
    let mut k = 0;
    for i in 0..n {
        for j in 0..=i {
            m[map[i] * n + map[j]] = packed[k];
            m[map[j] * n + map[i]] = packed[k];
            k += 1;
        }
    }
    m
}

/// UF6 with g-xTB v1.1 takes 40 SCF iterations in axtb and 50 in the 1.1.0 program, and both end
/// on the same state (total energies 1.1e-10 Eh apart). The cause: the damped Fock matrix of
/// iteration 2 has an exactly degenerate, partially filled HOMO set (three orbitals, two
/// doubly occupied). Aufbau occupation (the program's `occu_`, decoded: integer filling, its 0.99
/// only locates the HOMO) then fills whichever two-dimensional subspace the eigensolver returns,
/// which is round-off: the program's density of iteration 2 differs from ours only inside the
/// degenerate set, and the SCF paths part from there (iteration-2 energies -1039.6256 vs
/// -1039.65 Eh, both on the same final state). This test checks each step with the program's
/// densities of iterations 1 and 2 (read under gdb).
#[test]
fn uf6_v11_degenerate_iteration_two() {
    let d: Value = serde_json::from_str(
        &std::fs::read_to_string(tests().join("reference_elements/uf6_v1.1_iterations.json")).unwrap(),
    )
    .unwrap();
    let xyz = std::fs::read_to_string(tests().join("reference_v2/elements/UF6/mol.xyz")).unwrap();
    let mol = axtb_gxtb::read_xyz(&xyz, 0.0, Some(0)).unwrap();
    let setup = Setup::new(&mol).unwrap();
    let n = setup.nao;
    let map = program_ao_map(&setup.ao);
    let (p1, p2) = (unpack(&floats(&d["P1"]), &map), unpack(&floats(&d["P2"]), &map));
    let printed = floats(&d["printed_energies"]);
    let conv = Some(Conventions::Gxtb);
    // the program's printed iteration energies (8 decimals) are our energies of its densities
    let e1 = setup.evaluate(std::slice::from_ref(&p1), conv).unwrap();
    let e2 = setup.evaluate(std::slice::from_ref(&p2), conv).unwrap();
    assert!((e1.energies.total - printed[0]).abs() < 5.1e-9, "iteration 1: {}", e1.energies.total);
    assert!((e2.energies.total - printed[1]).abs() < 5.1e-9, "iteration 2: {}", e2.energies.total);
    // the damped Fock matrix of iteration 2 from the program's P1 (F~1 = a F(P0), F~2 = F~1 + a (F(P1) - F~1))
    let a = 0.1_f32 as f64;
    let f0 = setup.evaluate(&setup.guess(), conv).unwrap().fock[0].clone();
    let ft2: Vec<f64> = (0..n * n).map(|k| a * f0[k] + a * (e1.fock[0][k] - a * f0[k])).collect();
    let mut solver = axtb::linalg::GenEigSolver::new(MatRef::from_row_major_slice(&setup.s, n, n), Par::Seq).unwrap();
    let mut eps = vec![0.0; n];
    let mut c = Mat::<f64>::zeros(n, n);
    solver.solve(MatRef::from_row_major_slice(&ft2, n, n), &mut eps, &mut c).unwrap();
    let nocc = setup.nalpha;
    let set: Vec<usize> = (0..n).filter(|&k| (eps[k] - eps[nocc - 1]).abs() < 1e-12).collect();
    assert_eq!(set, vec![nocc - 2, nocc - 1, nocc], "HOMO set {:?}", &eps[nocc - 4..nocc + 4]);
    // the program's P2 in our orbitals: N = C^T S P2 S C
    let s = &setup.s;
    let sc: Vec<Vec<f64>> =
        (0..n).map(|k| (0..n).map(|i| (0..n).map(|j| s[i * n + j] * c[(j, k)]).sum()).collect()).collect();
    let occ = |x: usize, y: usize| -> f64 {
        let mut v = 0.0;
        for i in 0..n {
            for j in 0..n {
                v += sc[x][i] * p2[i * n + j] * sc[y][j];
            }
        }
        v
    };
    let (mut outside, mut inside_trace, mut inside_dev) = (0.0f64, 0.0, 0.0f64);
    for x in 0..n {
        for y in 0..n {
            let v = occ(x, y);
            if set.contains(&x) && set.contains(&y) {
                if x == y {
                    inside_trace += v;
                    inside_dev = inside_dev.max((v - if x < nocc { 2.0 } else { 0.0 }).abs());
                }
                continue;
            }
            let aufbau = if x == y && x < nocc { 2.0 } else { 0.0 };
            outside = outside.max((v - aufbau).abs());
        }
    }
    eprintln!("UF6 iteration 2: outside the degenerate set {outside:.1e}, inside trace {inside_trace:.12}, deviation from our occupations {inside_dev:.3}");
    assert!(outside < 1e-12, "program P2 differs outside the degenerate set: {outside:e}");
    // inside the set the program holds the same four electrons in another subspace (which one
    // is round-off in either eigensolver, so only reported)
    assert!((inside_trace - 4.0).abs() < 1e-12, "{inside_trace}");
}

/// g-xTB 2.0.1, H2: the density is fixed from iteration 1, so the energy does not change and
/// tblite switches from damping to DIIS after iteration 2 instead of 3 (its mixer stages start
/// DIIS at iteration 4 or after an energy change below 1e-12 Eh). The program then converges in
/// 4 iterations, with the residuals of its printout.
#[test]
fn h2_v2_early_diis() {
    use axtb_gxtb::{v2::scf::Setup, ScfOptions};
    let dir = tests().join("reference_v2/elements/H2");
    let summ: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("summary.json")).unwrap()).unwrap();
    let xyz = std::fs::read_to_string(dir.join("mol.xyz")).unwrap();
    let mol = axtb_gxtb::v2::read_xyz(&xyz, 0.0, Some(0), Conventions::Gxtb).unwrap();
    let r = Setup::new(&mol, axtb_gxtb::v2::Model::v2_0_1(Conventions::Gxtb))
        .unwrap()
        .scf(&ScfOptions { conventions: Conventions::Gxtb, ..Default::default() })
        .unwrap();
    let cyc = summ["scf_cycles"].as_array().unwrap();
    assert_eq!(r.iterations, cyc.len());
    for (k, (e, err)) in r.history.iter().enumerate() {
        let (pe, perr) = (cyc[k][1].as_f64().unwrap(), cyc[k][3].as_f64().unwrap());
        assert!(
            (e - pe).abs() < 1e-11 && (err - perr).abs() < 1e-8,
            "iteration {}: {e} {err:e} vs {pe} {perr:e}",
            k + 1
        );
    }
}

/// The 2.0.1 program's analytic gradient for [Gd(H2O)6]3+ with the axial waters on the z axis
/// is off by 8e-5 Eh/bohr in y on Gd and both axial O (it agrees with axtb to 1e-15 after a
/// rotation of the input; tests/gxtb_elements.py), while its energies agree with axtb's to
/// 1e-11 Eh: five-point differences of the program's own energies (--acc 0.0001) follow axtb's
/// gradient, not the program's.
#[test]
fn gd_aqua_v2_program_gradient_axis() {
    use axtb_gxtb::{v2::scf::Setup, ScfOptions};
    let d: Value = serde_json::from_str(
        &std::fs::read_to_string(tests().join("reference_elements/gd_aqua_v2_gradient.json")).unwrap(),
    )
    .unwrap();
    let mol = axtb_gxtb::v2::read_xyz(
        d["xyz"].as_str().unwrap(),
        d["charge"].as_f64().unwrap(),
        Some(d["uhf"].as_u64().unwrap() as usize),
        Conventions::Gxtb,
    )
    .unwrap();
    let setup = Setup::new(&mol, axtb_gxtb::v2::Model::v2_0_1(Conventions::Gxtb)).unwrap();
    let r = setup.scf(&ScfOptions { conventions: Conventions::Gxtb, accuracy: 1e-4, ..Default::default() }).unwrap();
    assert!((r.energies.total - d["energy"].as_f64().unwrap()).abs() < 1e-9, "energy {}", r.energies.total);
    let g = setup.gradient(&r).unwrap();
    let h = d["step_bohr"].as_f64().unwrap();
    for x in d["displaced"].as_array().unwrap() {
        let (a, k) = (x["atom"].as_u64().unwrap() as usize, x["axis"].as_u64().unwrap() as usize);
        let e = floats(&x["energies"]);
        let fd = (e[0] - 8.0 * e[1] + 8.0 * e[2] - e[3]) / (12.0 * h);
        let prog = d["gradient"][a][k].as_f64().unwrap();
        eprintln!("atom {a} axis {k}: program FD {fd:.9}, axtb {:.9}, program analytic {prog:.9}", g[a][k]);
        // the program's energies have 10 decimals: FD noise ~2e-7 Eh/bohr
        assert!((g[a][k] - fd).abs() < 1e-6, "axtb vs FD of the program's energies");
        assert!((prog - fd).abs() > 2e-5, "the program's gradient no longer differs");
    }
}
