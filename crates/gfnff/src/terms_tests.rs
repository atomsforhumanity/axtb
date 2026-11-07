//! The dual-number derivative of every kernel's gradient against central differences of
//! the f64 kernel (the analytic Hessian is built from these derivatives).
use super::*;
use crate::real::Dual;

fn check<F, G>(x0: &[[f64; 3]], f64k: F, dualk: G, name: &str)
where
    F: Fn(&[[f64; 3]]) -> Vec<[f64; 3]>,
    G: Fn(&[V3<Dual>]) -> Vec<V3<Dual>>,
{
    check_step(x0, f64k, dualk, name, 1e-5, 1e-6);
}

/// Dual derivative against central differences with step `h` (Richardson-extrapolated with
/// h/2 when `h` > 1e-5, for points where the literal gradient loses digits close by).
fn check_step<F, G>(x0: &[[f64; 3]], f64k: F, dualk: G, name: &str, h: f64, tol: f64)
where
    F: Fn(&[[f64; 3]]) -> Vec<[f64; 3]>,
    G: Fn(&[V3<Dual>]) -> Vec<V3<Dual>>,
{
    let n = x0.len();
    let mut worst: f64 = 0.0;
    let mut at = (0, 0, 0, 0);
    for k in 0..n {
        for c in 0..3 {
            let xd: Vec<V3<Dual>> = (0..n)
                .map(|i| {
                    let mut v = [Dual::new(x0[i][0], 0.0), Dual::new(x0[i][1], 0.0), Dual::new(x0[i][2], 0.0)];
                    if i == k {
                        v[c].d = 1.0;
                    }
                    v
                })
                .collect();
            let gd = dualk(&xd);
            let fd_at = |h: f64| {
                let mut xp = x0.to_vec();
                let mut xm = x0.to_vec();
                xp[k][c] += h;
                xm[k][c] -= h;
                let gp = f64k(&xp);
                let gm = f64k(&xm);
                (0..gd.len()).map(|a| [0, 1, 2].map(|cc| (gp[a][cc] - gm[a][cc]) / (2.0 * h))).collect::<Vec<_>>()
            };
            let f1 = fd_at(h);
            let f2 = if h > 1e-5 { fd_at(h / 2.0) } else { f1.clone() };
            for a in 0..gd.len() {
                for cc in 0..3 {
                    let fd = if h > 1e-5 { (4.0 * f2[a][cc] - f1[a][cc]) / 3.0 } else { f1[a][cc] };
                    let d = (fd - gd[a][cc].d).abs() / (1.0 + fd.abs());
                    if d > worst {
                        worst = d;
                        at = (a, cc, k, c);
                    }
                }
            }
        }
    }
    assert!(worst < tol, "{name}: dual derivative differs from finite differences by {worst:e} at {at:?}");
}

fn hbpar() -> HbPar {
    let p =
        crate::param::Param::new(&crate::param::Gen::default(), &crate::conventions::Conventions::default().params());
    HbPar {
        hbacut: p.hbacut,
        hbscut: p.hbscut,
        hbalp: p.hbalp,
        hblongcut: p.hblongcut,
        hbabmix: p.hbabmix,
        hbnbcut: p.hbnbcut,
        xhaci_globabh: p.xhaci_globabh,
        xhaci_coh: p.xhaci_coh,
        tors_hb: p.tors_hb,
        bend_hb: p.bend_hb,
    }
}

// A (F) - H ... B (N) with B's two neighbors and two more atoms, bohr
const X: [[f64; 3]; 7] = [
    [-5.3, 0.2, 0.1],
    [-3.55, 0.05, -0.05],
    [0.0, 0.0, 0.0],
    [1.2, 2.2, 0.1],
    [1.3, -2.1, -0.2],
    [3.1, 3.0, 1.0],
    [2.9, 3.4, -1.3],
];

#[test]
fn hb_kernels() {
    let p = hbpar();
    let (ca, cb) = ([0.52, 1.5], [1.68, 1.6]);
    let (qa, qb, qh, radab) = (0.8, 0.9, 0.7, 1.4);
    let x1 = [X[0], X[2], X[1]];
    check(
        &x1,
        |x| hb1(x[0], x[1], x[2], ca, cb, qa, qb, qh, radab, &p).unwrap().1.to_vec(),
        |x| hb1(x[0], x[1], x[2], ca, cb, qa, qb, qh, radab, &p).unwrap().1.to_vec(),
        "hb1",
    );
    let x2 = [X[0], X[2], X[1], X[3], X[4]];
    check(
        &x2,
        |x| hb2(x[0], x[1], x[2], &x[3..], false, ca, cb, qa, qb, qh, radab, &p).unwrap().1,
        |x| hb2(x[0], x[1], x[2], &x[3..], false, ca, cb, qa, qb, qh, radab, &p).unwrap().1,
        "hb2",
    );
    check(
        &x2,
        |x| hb2_rnr(x[0], x[1], x[2], [x[3], x[4]], 7.0, ca, cb, qa, qb, qh, radab, &p).unwrap().1,
        |x| hb2_rnr(x[0], x[1], x[2], [x[3], x[4]], 7.0, ca, cb, qa, qb, qh, radab, &p).unwrap().1,
        "hb2_rnr",
    );
    let x3 = [X[0], X[2], X[1], X[3], X[5], X[6]];
    check(
        &x3,
        |x| hb3(x[0], x[1], x[2], x[3], &x[4..], ca, cb, qa, qb, qh, radab, &p).unwrap().1,
        |x| hb3(x[0], x[1], x[2], x[3], &x[4..], ca, cb, qa, qb, qh, radab, &p).unwrap().1,
        "hb3",
    );
    let par =
        crate::param::Param::new(&crate::param::Gen::default(), &crate::conventions::Conventions::default().params());
    let xx = [X[0], X[3], X[2]];
    check(
        &xx,
        |x| xb(x[0], x[1], x[2], 1.2, 0.9, 0.8, 3.0, &par).unwrap().1.to_vec(),
        |x| xb(x[0], x[1], x[2], 1.2, 0.9, 0.8, 3.0, &par).unwrap().1.to_vec(),
        "xb",
    );
}

#[test]
fn bonded_kernels() {
    let x = [X[2], X[3], X[4], X[5]];
    check(
        &x[..3],
        |x| bend(x[0], x[1], x[2], 1.9, 0.3, [1.0, 0.8, 0.9], 0.595).1.to_vec(),
        |x| bend(x[0], x[1], x[2], 1.9, 0.3, [1.0, 0.8, 0.9], 0.595).1.to_vec(),
        "bend",
    );
    for nrot in [3, 0, -1] {
        check(
            &x,
            |x| torsion([x[0], x[1], x[2], x[3]], nrot, 0.4, 0.2, [1.0, 0.8, 0.9, 1.1], 0.505).1.to_vec(),
            |x| torsion([x[0], x[1], x[2], x[3]], nrot, 0.4, 0.2, [1.0, 0.8, 0.9, 1.1], 0.505).1.to_vec(),
            "torsion",
        );
    }
    check(&x[..3], |x| batm(x[0], x[1], x[2], -0.3).1.to_vec(), |x| batm(x[0], x[1], x[2], -0.3).1.to_vec(), "batm");
}

/// The smooth angle kernels used by the Hessian give xtb's energies and gradients (generic
/// geometries) and the right derivatives at linear bends and planar torsions, where the
/// literal kernels' dual derivatives are 0/0.
#[test]
fn smooth_angle_kernels() {
    let x = [X[2], X[3], X[4], X[5]];
    let rc = [1.0, 0.8, 0.9, 1.1];
    let same = |a: &[[f64; 3]], b: &[[f64; 3]], ea: f64, eb: f64, name: &str| {
        let d = a.iter().zip(b).flat_map(|(p, q)| (0..3).map(move |c| (p[c] - q[c]).abs())).fold(0.0, f64::max);
        assert!(d < 1e-14 && (ea - eb).abs() < 1e-14, "{name}: smooth vs literal {d:e} {:e}", (ea - eb).abs());
    };
    for c0 in [1.9, 2.0943951023931953, PI, 3.2114] {
        let (e1, g1) = bend(x[0], x[1], x[2], c0, 0.3, [1.0, 0.8, 0.9], 0.595);
        let (e2, g2) = bend_smooth(x[0], x[1], x[2], c0, 0.3, [1.0, 0.8, 0.9], 0.595);
        same(&g1, &g2, e1, e2, "bend");
        check(
            &x[..3],
            |x| bend(x[0], x[1], x[2], c0, 0.3, [1.0, 0.8, 0.9], 0.595).1.to_vec(),
            |x| bend_smooth(x[0], x[1], x[2], c0, 0.3, [1.0, 0.8, 0.9], 0.595).1.to_vec(),
            "bend_smooth",
        );
    }
    for (nrot, phi0) in [(1, 0.0), (2, PI), (3, PI), (2, PI / 2.0), (1, 1.0)] {
        let (e1, g1) = torsion(x, nrot, phi0, 0.2, rc, 0.505);
        let (e2, g2) = torsion_smooth(x, nrot, phi0, 0.2, rc, 0.505);
        same(&g1, &g2, e1, e2, "torsion");
        check(
            &x,
            |x| torsion([x[0], x[1], x[2], x[3]], nrot, phi0, 0.2, rc, 0.505).1.to_vec(),
            |x| torsion_smooth([x[0], x[1], x[2], x[3]], nrot, phi0, 0.2, rc, 0.505).1.to_vec(),
            "torsion_smooth",
        );
    }
    let (e1, g1) = stors(x);
    let (e2, g2) = stors_smooth(x);
    same(&g1, &g2, e1, e2, "stors");
    // linear bends (reference angle pi and 2 pi / 3) and a planar torsion
    let lin = [[0.0, 0.0, 0.0], [0.0, 0.0, 2.2], [0.0, 0.0, -2.1]];
    for c0 in [PI, 2.0943951023931953] {
        check_step(
            &lin,
            |x| bend(x[0], x[1], x[2], c0, 0.3, [1.0, 0.8, 0.9], 0.595).1.to_vec(),
            |x| bend_smooth(x[0], x[1], x[2], c0, 0.3, [1.0, 0.8, 0.9], 0.595).1.to_vec(),
            "linear bend",
            2e-3,
            1e-9,
        );
    }
    let planar = [[1.3, 2.2, 0.0], [0.0, 0.0, 0.0], [2.8, 0.0, 0.0], [4.1, -2.2, 0.0]];
    for (nrot, phi0) in [(1, 0.0), (2, PI), (3, PI)] {
        check_step(
            &planar,
            |x| torsion([x[0], x[1], x[2], x[3]], nrot, phi0, 0.2, rc, 0.505).1.to_vec(),
            |x| torsion_smooth([x[0], x[1], x[2], x[3]], nrot, phi0, 0.2, rc, 0.505).1.to_vec(),
            "planar torsion",
            2e-3,
            1e-9,
        );
    }
}

/// HB and XB kernels at a linear A-H...B / A-X...B, where xtb's out-of-line factor is 0/0.
#[test]
fn linear_hb_xb_kernels() {
    let p = hbpar();
    let (ca, cb) = ([0.52, 1.5], [1.68, 1.6]);
    let x = [[-5.3, 0.0, 0.0], [0.0, 0.0, 0.0], [-3.5, 0.0, 0.0]];
    check_step(
        &x,
        |x| hb1(x[0], x[1], x[2], ca, cb, 0.8, 0.9, 0.7, 1.4, &p).unwrap().1.to_vec(),
        |x| hb1(x[0], x[1], x[2], ca, cb, 0.8, 0.9, 0.7, 1.4, &p).unwrap().1.to_vec(),
        "linear hb1",
        2e-3,
        1e-8,
    );
    let par =
        crate::param::Param::new(&crate::param::Gen::default(), &crate::conventions::Conventions::default().params());
    let xx = [[-3.5, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 0.0, 0.0]];
    check_step(
        &xx,
        |x| xb(x[0], x[1], x[2], 1.2, 0.9, 0.8, 3.0, &par).unwrap().1.to_vec(),
        |x| xb(x[0], x[1], x[2], 1.2, 0.9, 0.8, 3.0, &par).unwrap().1.to_vec(),
        "linear xb",
        2e-3,
        1e-8,
    );
}
