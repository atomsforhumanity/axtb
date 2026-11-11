//! Symmetric tridiagonal eigenproblems: all eigenvalues by the root-free QL/QR iteration of
//! Pal, Walker and Kahan (LAPACK `dsterf`), and selected eigenvectors by inverse iteration with
//! modified Gram-Schmidt within clusters of close eigenvalues (LAPACK `dstein`, with the LU
//! factorization and the perturbed solve of `dlagtf`/`dlagts`). Used by the SCF eigensolver
//! ([`crate::linalg::GenEigSolver::solve_lowest`]) when only the occupied orbitals are needed:
//! that avoids the eigenvectors of the whole spectrum (divide and conquer) and their
//! back-transformation.

use crate::{Error, Result};

const EPS: f64 = f64::EPSILON * 0.5;
const SAFMIN: f64 = f64::MIN_POSITIVE;

/// sign(a, b): |a| with the sign of b (Fortran SIGN).
#[inline]
fn sign(a: f64, b: f64) -> f64 {
    if b >= 0.0 {
        a.abs()
    } else {
        -a.abs()
    }
}

/// Eigenvalues of the 2 x 2 symmetric matrix [[a, b], [b, c]] (LAPACK `dlae2`): the one of
/// larger absolute value first.
fn dlae2(a: f64, b: f64, c: f64) -> (f64, f64) {
    let sm = a + c;
    let df = a - c;
    let adf = df.abs();
    let tb = b + b;
    let ab = tb.abs();
    let (acmx, acmn) = if a.abs() > c.abs() { (a, c) } else { (c, a) };
    let rt = if adf > ab {
        adf * (1.0 + (ab / adf).powi(2)).sqrt()
    } else if adf < ab {
        ab * (1.0 + (adf / ab).powi(2)).sqrt()
    } else {
        ab * std::f64::consts::SQRT_2
    };
    if sm < 0.0 {
        let rt1 = 0.5 * (sm - rt);
        (rt1, (acmx / rt1) * acmn - (b / rt1) * b)
    } else if sm > 0.0 {
        let rt1 = 0.5 * (sm + rt);
        (rt1, (acmx / rt1) * acmn - (b / rt1) * b)
    } else {
        (0.5 * rt, -0.5 * rt)
    }
}

/// All eigenvalues (ascending, in `d`) of the symmetric tridiagonal matrix with diagonal `d`
/// (n) and off-diagonal `e` (n - 1; destroyed). A non-finite element is an error.
pub fn eigenvalues(d: &mut [f64], e: &mut [f64]) -> Result<()> {
    let n = d.len();
    if d.iter().chain(e.iter()).any(|x| !x.is_finite()) {
        return Err(Error::Numerical("tridiagonal matrix with non-finite elements".into()));
    }
    if n <= 1 {
        return Ok(());
    }
    let eps2 = EPS * EPS;
    let safmax = 1.0 / SAFMIN;
    let ssfmax = safmax.sqrt() / 3.0;
    let ssfmin = SAFMIN.sqrt() / eps2;
    let nmaxit = 30 * n;
    let mut jtot = 0;
    let mut l1 = 0;
    // 0-based: e[i] couples d[i] and d[i + 1]
    while l1 < n {
        if l1 > 0 {
            e[l1 - 1] = 0.0;
        }
        let mut m = l1;
        while m < n - 1 {
            if e[m].abs() <= d[m].abs().sqrt() * d[m + 1].abs().sqrt() * EPS {
                e[m] = 0.0;
                break;
            }
            m += 1;
        }
        let (mut l, lsv) = (l1, l1);
        let (mut lend, lendsv) = (m, m);
        l1 = m + 1;
        if lend == l {
            continue;
        }
        // scale the block if its entries are very large or very small
        let anorm = (l..=lend).map(|i| d[i].abs()).chain((l..lend).map(|i| e[i].abs())).fold(0.0, f64::max);
        if anorm == 0.0 {
            continue;
        }
        let scale = if anorm > ssfmax {
            ssfmax / anorm
        } else if anorm < ssfmin {
            ssfmin / anorm
        } else {
            1.0
        };
        if scale != 1.0 {
            for i in l..=lend {
                d[i] *= scale;
            }
            for i in l..lend {
                e[i] *= scale;
            }
        }
        for i in l..lend {
            e[i] *= e[i];
        }
        // QL if the small end is at the top, else QR
        if d[lend].abs() < d[l].abs() {
            lend = lsv;
            l = lendsv;
        }
        if lend >= l {
            // QL iteration: look for a small subdiagonal element
            loop {
                let mut m = l;
                while m < lend {
                    if e[m].abs() <= eps2 * (d[m] * d[m + 1]).abs() {
                        break;
                    }
                    m += 1;
                }
                if m < lend {
                    e[m] = 0.0;
                }
                let mut p = d[l];
                if m == l {
                    // eigenvalue found
                    d[l] = p;
                    l += 1;
                    if l <= lend {
                        continue;
                    }
                    break;
                }
                if m == l + 1 {
                    let (rt1, rt2) = dlae2(d[l], e[l].sqrt(), d[l + 1]);
                    d[l] = rt1;
                    d[l + 1] = rt2;
                    e[l] = 0.0;
                    l += 2;
                    if l <= lend {
                        continue;
                    }
                    break;
                }
                if jtot == nmaxit {
                    break;
                }
                jtot += 1;
                // shift
                let rte = e[l].sqrt();
                let mut sigma = (d[l + 1] - p) / (2.0 * rte);
                let r = sigma.hypot(1.0);
                sigma = p - (rte / (sigma + sign(r, sigma)));
                let (mut c, mut s) = (1.0, 0.0);
                let mut gamma = d[m] - sigma;
                p = gamma * gamma;
                // inner loop
                let mut i = m;
                while i > l {
                    i -= 1;
                    let bb = e[i];
                    let r = p + bb;
                    if i != m - 1 {
                        e[i + 1] = s * r;
                    }
                    let oldc = c;
                    c = p / r;
                    s = bb / r;
                    let oldgam = gamma;
                    let alpha = d[i];
                    gamma = c * (alpha - sigma) - s * oldgam;
                    d[i + 1] = oldgam + (alpha - gamma);
                    p = if c != 0.0 { (gamma * gamma) / c } else { oldc * bb };
                }
                e[l] = s * p;
                d[l] = sigma + gamma;
            }
        } else {
            // QR iteration: look for a small superdiagonal element
            loop {
                let mut m = l;
                while m > lend {
                    if e[m - 1].abs() <= eps2 * (d[m] * d[m - 1]).abs() {
                        break;
                    }
                    m -= 1;
                }
                if m > lend {
                    e[m - 1] = 0.0;
                }
                let mut p = d[l];
                if m == l {
                    d[l] = p;
                    if l == lend {
                        break;
                    }
                    l -= 1;
                    if l >= lend {
                        continue;
                    }
                    break;
                }
                if m + 1 == l {
                    let (rt1, rt2) = dlae2(d[l], e[l - 1].sqrt(), d[l - 1]);
                    d[l] = rt1;
                    d[l - 1] = rt2;
                    e[l - 1] = 0.0;
                    if l < lend + 2 {
                        break;
                    }
                    l -= 2;
                    continue;
                }
                if jtot == nmaxit {
                    break;
                }
                jtot += 1;
                let rte = e[l - 1].sqrt();
                let mut sigma = (d[l - 1] - p) / (2.0 * rte);
                let r = sigma.hypot(1.0);
                sigma = p - (rte / (sigma + sign(r, sigma)));
                let (mut c, mut s) = (1.0, 0.0);
                let mut gamma = d[m] - sigma;
                p = gamma * gamma;
                for i in m..l {
                    let bb = e[i];
                    let r = p + bb;
                    if i != m {
                        e[i - 1] = s * r;
                    }
                    let oldc = c;
                    c = p / r;
                    s = bb / r;
                    let oldgam = gamma;
                    let alpha = d[i + 1];
                    gamma = c * (alpha - sigma) - s * oldgam;
                    d[i] = oldgam + (alpha - gamma);
                    p = if c != 0.0 { (gamma * gamma) / c } else { oldc * bb };
                }
                e[l - 1] = s * p;
                d[l] = sigma + gamma;
            }
        }
        if scale != 1.0 {
            for x in &mut d[lsv..=lendsv] {
                *x /= scale;
            }
        }
        if jtot >= nmaxit && (lsv..lendsv).any(|i| e[i] != 0.0) {
            return Err(Error::Numerical("tridiagonal eigenvalues did not converge".into()));
        }
    }
    if d.iter().any(|x| !x.is_finite()) {
        return Err(Error::Numerical("tridiagonal eigenvalues overflowed".into()));
    }
    d.sort_by(f64::total_cmp);
    Ok(())
}

/// LU factorization with partial pivoting of T - lambda I (LAPACK `dlagtf` with tol = 0):
/// `a` the diagonal of U, `b` its first and `dd` its second superdiagonal, `c` the multipliers,
/// `piv` the interchanges.
fn lu(n: usize, a: &mut [f64], lambda: f64, b: &mut [f64], c: &mut [f64], dd: &mut [f64], piv: &mut [bool]) {
    a[0] -= lambda;
    if n == 1 {
        return;
    }
    let mut scale1 = a[0].abs() + b[0].abs();
    for k in 0..n - 1 {
        a[k + 1] -= lambda;
        let mut scale2 = c[k].abs() + a[k + 1].abs();
        if k < n - 2 {
            scale2 += b[k + 1].abs();
        }
        let piv1 = if a[k] == 0.0 { 0.0 } else { a[k].abs() / scale1 };
        if c[k] == 0.0 {
            piv[k] = false;
            scale1 = scale2;
            if k < n - 2 {
                dd[k] = 0.0;
            }
        } else {
            let piv2 = c[k].abs() / scale2;
            if piv2 <= piv1 {
                piv[k] = false;
                scale1 = scale2;
                c[k] /= a[k];
                a[k + 1] -= c[k] * b[k];
                if k < n - 2 {
                    dd[k] = 0.0;
                }
            } else {
                piv[k] = true;
                let mult = a[k] / c[k];
                a[k] = c[k];
                let temp = a[k + 1];
                a[k + 1] = b[k] - mult * temp;
                if k < n - 2 {
                    dd[k] = b[k + 1];
                    b[k + 1] = -mult * dd[k];
                }
                b[k] = temp;
                c[k] = mult;
            }
        }
    }
}

/// Solve (T - lambda I) x = y with the factorization of [`lu`], perturbing tiny pivots by
/// multiples of `tol` (LAPACK `dlagts`, job -1).
fn lu_solve(n: usize, a: &[f64], b: &[f64], c: &[f64], dd: &[f64], piv: &[bool], inv: &[f64], tol: f64, y: &mut [f64]) {
    let bignum = 1.0 / SAFMIN;
    for k in 1..n {
        if !piv[k - 1] {
            y[k] -= c[k - 1] * y[k - 1];
        } else {
            let temp = y[k - 1];
            y[k - 1] = y[k];
            y[k] = temp - c[k - 1] * y[k];
        }
    }
    for k in (0..n).rev() {
        let mut temp = if k + 2 < n {
            y[k] - b[k] * y[k + 1] - dd[k] * y[k + 2]
        } else if k + 1 < n {
            y[k] - b[k] * y[k + 1]
        } else {
            y[k]
        };
        // A deviation from dlagts, for speed: a product with the reciprocal pivot instead of
        // the checked division. `inv` is 0 for pivots up to 1e-150 (see `inverse_iteration`),
        // so |inv| < 1e150, and with |temp| < 1e150 the product stays below 1e300 (no
        // overflow); everything else takes dlagts' scaled division and pivot perturbation.
        if inv[k] != 0.0 && temp.abs() < 1e150 {
            y[k] = temp * inv[k];
            continue;
        }
        let mut ak = a[k];
        let mut pert = sign(tol, ak);
        loop {
            let absak = ak.abs();
            if absak < 1.0 {
                if absak < SAFMIN {
                    if absak == 0.0 || temp.abs() * SAFMIN > absak {
                        ak += pert;
                        pert *= 2.0;
                        continue;
                    }
                    temp *= bignum;
                    ak *= bignum;
                } else if temp.abs() > absak * bignum {
                    ak += pert;
                    pert *= 2.0;
                    continue;
                }
            }
            break;
        }
        y[k] = temp / ak;
    }
}

/// Eigenvectors of the symmetric tridiagonal matrix (`d`, `e`) for the eigenvalues `w`
/// (ascending, from [`eigenvalues`]), by inverse iteration: column j of the column-major
/// n x w.len() array `z` (leading dimension n). Vectors of eigenvalues closer than
/// 1e-3 ||T|| to their predecessor form a cluster and are orthogonalized against the previous
/// members of their cluster (modified Gram-Schmidt); clusters are independent and run in
/// parallel on the current rayon pool when `parallel` (each vector's start is fixed by its
/// index, so the result does not depend on the thread count).
pub fn eigenvectors(d: &[f64], e: &[f64], w: &[f64], z: &mut [f64], parallel: bool) -> Result<()> {
    use rayon::prelude::*;
    let n = d.len();
    let k = w.len();
    if n == 0 || k == 0 {
        return Ok(());
    }
    if n == 1 {
        z[0] = 1.0;
        return Ok(());
    }
    let Clusters { shifts, clusters, onenrm } = Clusters::new(d, e, w);
    let run = |&(c0, c1): &(usize, usize)| -> Result<Vec<f64>> {
        let mut out = vec![0.0; n * (c1 - c0)];
        let mut ws = Work::new(n);
        for j in c0..c1 {
            let (prev, cur) = out.split_at_mut((j - c0) * n);
            inverse_iteration(d, e, shifts[j], onenrm, j as u64, prev, &mut ws, &mut cur[..n])
                .map_err(|_| Error::Numerical(format!("inverse iteration did not converge for eigenvalue {j}")))?;
        }
        Ok(out)
    };
    let parts: Vec<Result<Vec<f64>>> =
        if parallel { clusters.par_iter().map(run).collect() } else { clusters.iter().map(run).collect() };
    for (&(c0, c1), part) in clusters.iter().zip(parts) {
        z[c0 * n..c1 * n].copy_from_slice(&part?);
    }
    Ok(())
}

/// Shifts of the inverse iteration (equal eigenvalues separated by a few ulps), the clusters
/// [start, end) of eigenvalues closer than 1e-3 ||T|| to their predecessor, and ||T||_1.
struct Clusters {
    shifts: Vec<f64>,
    clusters: Vec<(usize, usize)>,
    onenrm: f64,
}

impl Clusters {
    fn new(d: &[f64], e: &[f64], w: &[f64]) -> Self {
        let k = w.len();
        let onenrm = one_norm(d, e);
        let ortol = 1e-3 * onenrm;
        let mut shifts = vec![0.0; k];
        let mut clusters: Vec<(usize, usize)> = Vec::new();
        for j in 0..k {
            let mut xj = w[j];
            if j > 0 {
                let xjm = shifts[j - 1];
                let pertol = 10.0 * (EPS * xj).abs();
                if xj - xjm < pertol {
                    xj = xjm + pertol;
                }
                if (xj - xjm).abs() > ortol {
                    clusters.push((clusters.last().map_or(0, |c: &(usize, usize)| c.1), j));
                }
            }
            shifts[j] = xj;
        }
        clusters.push((clusters.last().map_or(0, |c| c.1), k));
        Self { shifts, clusters, onenrm }
    }
}

/// 1-norm of the symmetric tridiagonal matrix (`d`, `e`), n >= 2.
pub fn one_norm(d: &[f64], e: &[f64]) -> f64 {
    let n = d.len();
    let mut onenrm = (d[0].abs() + e[0].abs()).max(d[n - 1].abs() + e[n - 2].abs());
    for i in 1..n - 1 {
        onenrm = onenrm.max(d[i].abs() + e[i - 1].abs() + e[i].abs());
    }
    onenrm
}

/// Cost of [`eigenvectors`] for the lowest `k` of the eigenvalues `w` (all n of a symmetric
/// matrix, ascending) relative to computing every eigenvector by divide and conquer, from the
/// clusters of consecutive eigenvalues closer than 1e-3 `tnorm` (as in `eigenvectors` with
/// `tnorm` = [`one_norm`], or an estimate of it): above 1 when a cluster has more than n/6
/// eigenvalues (the Gram-Schmidt orthogonalization of a cluster runs on one thread) or the
/// squared cluster sizes sum to more than n^2/16 (the break-even points measured on 8 and 1
/// threads). Water clusters of 200 molecules in g-xTB's basis reach 1-2 (n = 1200, 800
/// occupied orbitals in clusters of up to ~300); alkanes and organic molecules of up to 300
/// atoms stay below 0.7.
pub fn cluster_cost(w: &[f64], k: usize, tnorm: f64) -> f64 {
    let n = w.len();
    let k = k.min(n);
    if n < 2 || k < 2 {
        return 0.0;
    }
    let ortol = 1e-3 * tnorm;
    let (mut max, mut sum, mut start) = (0usize, 0usize, 0usize);
    for j in 1..=k {
        if j == k || w[j] - w[j - 1] > ortol {
            // single eigenvalues need no orthogonalization
            let c = j - start;
            if c > 1 {
                max = max.max(c);
                sum += c * c;
            }
            start = j;
        }
    }
    let nf = n as f64;
    (6.0 * max as f64 / nf).max(16.0 * sum as f64 / (nf * nf))
}

/// Work space of one inverse iteration.
struct Work {
    a: Vec<f64>,
    b: Vec<f64>,
    c: Vec<f64>,
    dd: Vec<f64>,
    inv: Vec<f64>,
    piv: Vec<bool>,
    x: Vec<f64>,
}

impl Work {
    fn new(n: usize) -> Self {
        Self {
            a: vec![0.0; n],
            b: vec![0.0; n],
            c: vec![0.0; n],
            dd: vec![0.0; n],
            inv: vec![0.0; n],
            piv: vec![false; n],
            x: vec![0.0; n],
        }
    }
}

/// One eigenvector of (`d`, `e`) at shift `xj` by inverse iteration (LAPACK `dstein`), the start
/// vector from `seed`, orthogonalized against the vectors in `prev` (n each); into `out`.
fn inverse_iteration(
    d: &[f64],
    e: &[f64],
    xj: f64,
    onenrm: f64,
    seed: u64,
    prev: &[f64],
    ws: &mut Work,
    out: &mut [f64],
) -> std::result::Result<(), ()> {
    const MAXITS: usize = 5;
    const EXTRA: usize = 2;
    let n = d.len();
    let dtpcrt = (0.1 / n as f64).sqrt();
    let Work { a, b, c, dd, inv, piv, x } = ws;
    // start vector, uniform in (-1, 1), from a generator seeded with the vector's index
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0x2545_F491_4F6C_DD1D;
    for v in x.iter_mut() {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        *v = (s >> 11) as f64 / (1u64 << 52) as f64 - 1.0;
    }
    a.copy_from_slice(d);
    b[..n - 1].copy_from_slice(e);
    c[..n - 1].copy_from_slice(e);
    lu(n, a, xj, b, c, dd, piv);
    // tolerance of the perturbed solve (dlagts with tol <= 0)
    let mut tol = a[0].abs().max(a[1].abs()).max(b[0].abs());
    for i in 2..n {
        tol = tol.max(a[i].abs()).max(b[i - 1].abs()).max(dd[i - 2].abs());
    }
    tol *= EPS;
    if tol == 0.0 {
        tol = EPS;
    }
    // reciprocal pivots for `lu_solve`'s shortcut; 0 marks a pivot of at most 1e-150 (its
    // reciprocal could overflow a product), solved with dlagts' scaling and perturbation
    for (r, &p) in inv.iter_mut().zip(a.iter()) {
        *r = if p.abs() > 1e-150 { 1.0 / p } else { 0.0 };
    }
    let mut nrmchk = 0;
    for _ in 0..MAXITS {
        let jmax = argmax_abs(x);
        let scl = n as f64 * onenrm * EPS.max(a[n - 1].abs()) / x[jmax].abs();
        x.iter_mut().for_each(|v| *v *= scl);
        lu_solve(n, a, b, c, dd, piv, inv, tol, x);
        for zi in prev.chunks_exact(n) {
            let ztr = dot(zi, x);
            for (v, p) in x.iter_mut().zip(zi) {
                *v -= ztr * p;
            }
        }
        let nrm = x[argmax_abs(x)].abs();
        if nrm < dtpcrt {
            continue;
        }
        nrmchk += 1;
        if nrmchk < EXTRA + 1 {
            continue;
        }
        let norm = dot(x, x).sqrt();
        let jmax = argmax_abs(x);
        let scl = if x[jmax] < 0.0 { -1.0 / norm } else { 1.0 / norm };
        for (zv, v) in out.iter_mut().zip(x.iter()) {
            *zv = v * scl;
        }
        return Ok(());
    }
    Err(())
}

/// Dot product with eight partial sums in a fixed order (vectorizes; the result depends on
/// the length alone).
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    let m = a.len() / 8 * 8;
    let mut acc = [0.0f64; 8];
    for (x, y) in a[..m].as_chunks::<8>().0.iter().zip(b[..m].as_chunks::<8>().0) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut s = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
    for (x, y) in a[m..].iter().zip(&b[m..]) {
        s += x * y;
    }
    s
}

#[inline]
fn argmax_abs(x: &[f64]) -> usize {
    let mut best = 0;
    let mut m = -1.0;
    for (i, v) in x.iter().enumerate() {
        if v.abs() > m {
            m = v.abs();
            best = i;
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    fn rng(seed: &mut u64) -> f64 {
        *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*seed >> 11) as f64) / ((1u64 << 53) as f64) - 0.5
    }

    /// Eigenvalues and the lowest k eigenvectors against faer's dense solver.
    fn check(d: &[f64], e: &[f64], k: usize) {
        let n = d.len();
        let mut t = Mat::<f64>::zeros(n, n);
        for i in 0..n {
            t[(i, i)] = d[i];
            if i + 1 < n {
                t[(i + 1, i)] = e[i];
                t[(i, i + 1)] = e[i];
            }
        }
        let evd = t.as_ref().self_adjoint_eigen(faer::Side::Lower).unwrap();
        let reference: Vec<f64> = (0..n).map(|i| evd.S()[i]).collect();
        let (mut w, mut ee) = (d.to_vec(), e.to_vec());
        eigenvalues(&mut w, &mut ee).unwrap();
        let tnorm = reference.iter().fold(0.0f64, |a, x| a.max(x.abs())).max(1.0);
        for (a, b) in w.iter().zip(&reference) {
            assert!((a - b).abs() < 1e-13 * tnorm, "eigenvalue {a} vs {b}");
        }
        let mut z = vec![0.0; n * k];
        eigenvectors(d, e, &w[..k], &mut z, false).unwrap();
        let mut z2 = vec![0.0; n * k];
        eigenvectors(d, e, &w[..k], &mut z2, true).unwrap();
        assert_eq!(z, z2);
        for j in 0..k {
            let zj = &z[j * n..(j + 1) * n];
            // residual |T z - w z|
            for i in 0..n {
                let mut tz = d[i] * zj[i];
                if i > 0 {
                    tz += e[i - 1] * zj[i - 1];
                }
                if i + 1 < n {
                    tz += e[i] * zj[i + 1];
                }
                assert!((tz - w[j] * zj[i]).abs() < 1e-12 * tnorm, "residual of vector {j}");
            }
            for l in 0..=j {
                let zl = &z[l * n..(l + 1) * n];
                let dot: f64 = zj.iter().zip(zl).map(|(a, b)| a * b).sum();
                let expect = if l == j { 1.0 } else { 0.0 };
                assert!((dot - expect).abs() < 1e-12, "orthogonality {j} {l}: {dot}");
            }
        }
    }

    #[test]
    fn random_tridiagonal_matrices() {
        let mut seed = 7;
        for &n in &[1usize, 2, 3, 5, 17, 64, 200] {
            let d: Vec<f64> = (0..n).map(|_| rng(&mut seed) * 4.0).collect();
            let e: Vec<f64> = (0..n.saturating_sub(1)).map(|_| rng(&mut seed)).collect();
            check(&d, &e, n / 2 + 1.min(n));
        }
    }

    #[test]
    fn degenerate_and_split_matrices() {
        // repeated diagonal with zero couplings: exact multiplicities
        let n = 40;
        let d: Vec<f64> = (0..n).map(|i| (i % 4) as f64).collect();
        check(&d, &vec![0.0; n - 1], n);
        // a tight cluster: tiny couplings around equal diagonals
        let mut seed = 3;
        let d = vec![-1.0; n];
        let e: Vec<f64> = (0..n - 1).map(|_| 1e-9 * rng(&mut seed)).collect();
        check(&d, &e, n / 2);
        // the matrix of a uniform chain (clustered spectrum near the band edges)
        let d = vec![0.0; 300];
        let e = vec![1.0; 299];
        check(&d, &e, 150);
    }

    /// Non-finite elements (from a broken SCF) are an error, not a panic in the sort.
    #[test]
    fn non_finite_elements_are_errors() {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let (mut d, mut e) = (vec![1.0, bad, 3.0], vec![0.5, 0.5]);
            assert!(eigenvalues(&mut d, &mut e).is_err());
            let (mut d, mut e) = (vec![1.0, 2.0, 3.0], vec![bad, 0.5]);
            assert!(eigenvalues(&mut d, &mut e).is_err());
        }
        let (mut d, mut e) = (vec![f64::NAN], vec![]);
        assert!(eigenvalues(&mut d, &mut e).is_err());
    }
}
