//! g-xTB 2.0.1 coupled-perturbed response: the kernel against finite differences of the Fock
//! matrix, and the density, energy-weighted density and Mulliken moment responses to nuclear
//! displacements against central differences of tightly converged SCF densities.
//!
//! `V2_RESP=reference/water,...` selects molecules; `V2_RESP_ALL=1` checks every displacement
//! (default: up to 12 per molecule).

use axtb_gxtb::dense::{from_mat, to_mat};
use axtb_gxtb::v2::response::{rhs_finite_difference, Cpks, CpksOptions};
use axtb_gxtb::v2::scf::{ScfResult, Setup};
use axtb_gxtb::{Conventions, ScfOptions};
use faer::Par;
use serde_json::Value;
use std::path::{Path, PathBuf};

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/reference_v2")
}

fn json(path: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn molecule(rel: &str, conv: Conventions) -> axtb::Molecule {
    let dir = root().join(rel);
    let input = json(&dir.join("input.json"));
    let charge = input["charge"].as_f64().unwrap();
    let uhf = input["uhf"].as_u64().unwrap() as usize;
    let xyz = std::fs::read_to_string(dir.join("mol.xyz")).unwrap();
    axtb_gxtb::v2::read_xyz(&xyz, charge, Some(uhf), conv).unwrap()
}

fn tight(c: Conventions) -> ScfOptions {
    ScfOptions {
        conventions: c,
        energy_tol: Some(1e-12),
        residual_tol: Some(1e-13),
        max_iter: 1000,
        all_orbitals: true,
        ..Default::default()
    }
}

fn scf(m: &axtb::Molecule, c: Conventions) -> (Setup, ScfResult) {
    let s = Setup::new(m, c).unwrap();
    let r = s.scf(&tight(c)).unwrap();
    (s, r)
}

/// Deterministic pseudo-random numbers in [-1, 1).
struct Lcg(u64);
impl Lcg {
    fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    }
}

fn random_sym(n: usize, rng: &mut Lcg, scale: f64) -> Vec<f64> {
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let x = scale * rng.next();
            v[i * n + j] = x;
            v[j * n + i] = x;
        }
    }
    v
}

fn maxabs(v: &[f64]) -> f64 {
    v.iter().fold(0.0, |a: f64, b| a.max(b.abs()))
}

fn maxdiff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).fold(0.0, |m: f64, (x, y)| m.max((x - y).abs()))
}

/// Kernel against 4-point differences of the Fock matrix along a random density direction,
/// and its symmetry; returns (relative error, relative asymmetry).
fn kernel_check(setup: &Setup, r: &ScfResult) -> (f64, f64) {
    let n = setup.nao;
    let ns = r.density.len();
    let cp = Cpks::new(setup, r, CpksOptions::default(), Par::Seq);
    let mut rng = Lcg(42);
    let da: Vec<Vec<f64>> = (0..ns).map(|_| random_sym(n, &mut rng, 1e-2)).collect();
    let db: Vec<Vec<f64>> = (0..ns).map(|_| random_sym(n, &mut rng, 1e-2)).collect();
    let ka = cp.kernel(&da.iter().map(|d| to_mat(n, d)).collect::<Vec<_>>());
    let kb = cp.kernel(&db.iter().map(|d| to_mat(n, d)).collect::<Vec<_>>());
    let t = 1e-2;
    let f = |x: f64| -> Vec<Vec<f64>> {
        let p: Vec<Vec<f64>> =
            (0..ns).map(|s| r.density[s].iter().zip(&da[s]).map(|(p, d)| p + x * d).collect()).collect();
        setup.fock_of_density(&p)
    };
    let (f1, f2, f3, f4) = (f(t), f(-t), f(2.0 * t), f(-2.0 * t));
    let mut err: f64 = 0.0;
    let mut scale: f64 = 0.0;
    for s in 0..ns {
        let fd: Vec<f64> =
            (0..n * n).map(|k| (8.0 * (f1[s][k] - f2[s][k]) - (f3[s][k] - f4[s][k])) / (12.0 * t)).collect();
        let an = from_mat(&ka[s]);
        err = err.max(maxdiff(&fd, &an));
        scale = scale.max(maxabs(&fd));
    }
    let tr = |a: &[Vec<f64>], k: &[faer::Mat<f64>]| -> f64 {
        let mut x = 0.0;
        for s in 0..ns {
            let kk = from_mat(&k[s]);
            x += a[s].iter().zip(&kk).map(|(u, v)| u * v).sum::<f64>();
        }
        x
    };
    let (ab, ba) = (tr(&da, &kb), tr(&db, &ka));
    (err / scale, (ab - ba).abs() / ab.abs().max(ba.abs()))
}

struct Stats {
    dp: f64,
    dw: f64,
    dq: f64,
    ddip: f64,
    iters: usize,
}

/// Response of displacement y against 4-point differences of tight SCF densities.
fn displacement_check(setup: &Setup, r: &ScfResult, cp: &Cpks, y: usize, worst: &mut Stats, scale: &mut Stats) {
    let n = setup.nao;
    let c = setup.model.conventions;
    let (fy, sy) = rhs_finite_difference(setup, &r.density, y, 1e-3).unwrap();
    let resp = cp.solve(&fy, &sy).unwrap();
    worst.iters = worst.iters.max(resp.iterations);
    let h: f64 = std::env::var("V2_RESP_H").map(|v| v.parse().unwrap()).unwrap_or(1e-3);
    let at = |d: f64| {
        let mut m = setup.mol.clone();
        m.positions[y / 3][y % 3] += d;
        let (s, rr) = scf(&m, c);
        let w = s.energy_weighted_density(&rr);
        let st = &rr.state;
        (rr.density.clone(), w, st.q.clone(), st.dip.concat())
    };
    let (p1, p2, p3, p4) = (at(h), at(-h), at(2.0 * h), at(-2.0 * h));
    let d4 = |a: &[f64], b: &[f64], c: &[f64], d: &[f64]| -> Vec<f64> {
        (0..a.len()).map(|k| (8.0 * (a[k] - b[k]) - (c[k] - d[k])) / (12.0 * h)).collect()
    };
    let ns = r.density.len();
    for s in 0..ns {
        let fd = d4(&p1.0[s], &p2.0[s], &p3.0[s], &p4.0[s]);
        worst.dp = worst.dp.max(maxdiff(&fd, &from_mat(&resp.dp[s])));
        scale.dp = scale.dp.max(maxabs(&fd));
    }
    let wfd = d4(&p1.1, &p2.1, &p3.1, &p4.1);
    let mut wan = vec![0.0; n * n];
    for s in 0..ns {
        for (a, b) in wan.iter_mut().zip(from_mat(&resp.dw[s])) {
            *a += b;
        }
    }
    worst.dw = worst.dw.max(maxdiff(&wfd, &wan));
    scale.dw = scale.dw.max(maxabs(&wfd));
    // moments: Mulliken analysis of dP at the reference geometry plus the explicit change of
    // the integrals (at fixed P)
    let dpv: Vec<Vec<f64>> = resp.dp.iter().map(from_mat).collect();
    let ds = setup.delta_state(&dpv);
    let mut m1 = setup.mol.clone();
    let eps = 1e-4;
    m1.positions[y / 3][y % 3] += eps;
    let s1 = Setup::new(&m1, c).unwrap();
    let mut m2 = setup.mol.clone();
    m2.positions[y / 3][y % 3] -= eps;
    let s2 = Setup::new(&m2, c).unwrap();
    let (a, b) = (s1.state(&r.density), s2.state(&r.density));
    let dq: Vec<f64> = (0..ds.q.len()).map(|k| ds.q[k] + (a.q[k] - b.q[k]) / (2.0 * eps)).collect();
    let (ad, bd, dd) = (a.dip.concat(), b.dip.concat(), ds.dip.concat());
    let ddip: Vec<f64> = (0..dd.len()).map(|k| dd[k] + (ad[k] - bd[k]) / (2.0 * eps)).collect();
    let qfd = d4(&p1.2, &p2.2, &p3.2, &p4.2);
    let dfd = d4(&p1.3, &p2.3, &p3.3, &p4.3);
    worst.dq = worst.dq.max(maxdiff(&qfd, &dq));
    scale.dq = scale.dq.max(maxabs(&qfd));
    worst.ddip = worst.ddip.max(maxdiff(&dfd, &ddip));
    scale.ddip = scale.ddip.max(maxabs(&dfd));
}

#[test]
fn kernel_and_response_match_finite_differences() {
    let cases = std::env::var("V2_RESP").unwrap_or("reference/water,reference/dioxygen_triplet".into());
    let all = std::env::var("V2_RESP_ALL").is_ok();
    for rel in cases.split(',') {
        for c in [Conventions::Axtb, Conventions::Gxtb] {
            let mol = molecule(rel, c);
            let (setup, r) = scf(&mol, c);
            let (kerr, kasym) = kernel_check(&setup, &r);
            let cp = Cpks::new(&setup, &r, CpksOptions { tol: 1e-12, max_iter: 500 }, Par::Seq);
            let n3 = 3 * mol.nat();
            let ys: Vec<usize> =
                if all || n3 <= 12 { (0..n3).collect() } else { (0..12).map(|k| k * n3 / 12).collect() };
            let mut worst = Stats { dp: 0.0, dw: 0.0, dq: 0.0, ddip: 0.0, iters: 0 };
            let mut scale = Stats { dp: 0.0, dw: 0.0, dq: 0.0, ddip: 0.0, iters: 0 };
            for &y in &ys {
                displacement_check(&setup, &r, &cp, y, &mut worst, &mut scale);
            }
            let rel_dp = worst.dp / scale.dp;
            let rel_dw = worst.dw / scale.dw;
            let rel_dq = worst.dq / scale.dq;
            let rel_dd = worst.ddip / scale.ddip;
            println!(
                "{rel:28} {c:?} nao {:4} kernel {kerr:.1e} asym {kasym:.1e} | {} displ.: dP {rel_dp:.1e} dW {rel_dw:.1e} dq {rel_dq:.1e} ddip {rel_dd:.1e} (max |dP| {:.2e}) CG it <= {}",
                setup.nao,
                ys.len(),
                scale.dp,
                worst.iters
            );
            assert!(kerr < 1e-7 && kasym < 1e-10, "{rel} {c:?} kernel {kerr:e} {kasym:e}");
            assert!(rel_dp < 1e-6 && rel_dw < 1e-6 && rel_dq < 1e-6 && rel_dd < 1e-6, "{rel} {c:?}");
        }
    }
}
