//! Point groups and rotational symmetry numbers of molecules.
//!
//! The symmetry elements are found by testing candidate operations through the centroid:
//! inversion; proper axes C_n (n = 2..=[`MAX_ORDER`]) along principal axes, atom directions,
//! midpoints and normals of pairs, and normals of triangles of equivalent atoms (same element,
//! same distance from the centroid); mirror planes normal to differences of equivalent atoms,
//! to principal axes and to found axes, and containing found axes and atoms; improper axes
//! S_2n along found C_n axes. An operation is accepted when every atom is mapped onto an atom
//! of the same element within the tolerance (bohr) after its direction has been refined to
//! the best fit for that atom mapping. The point group follows from the elements found
//! (cubic and icosahedral groups from several axes of order >= 3; otherwise the main axis,
//! perpendicular C2 axes and mirror planes).
//!
//! The tolerance has the meaning of xtb's `desy` (default 0.1 bohr: xtb passes it as
//! `ToleranceFinal` to its symmetry analyzer, the largest atom deviation of an accepted
//! element). This is an independent implementation, not a port of xtb's analyzer
//! (`symmetry/symmetry_i.c`, GPL-2.0-or-later): both give the same point group for
//! structures that are symmetric well within the tolerance; for structures at the edge of
//! the tolerance they can differ.

/// Highest axis order tested.
pub const MAX_ORDER: u32 = 12;

/// A point group and its rotational symmetry number.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PointGroup {
    /// Schoenflies symbol: `C1`, `Cs`, `Ci`, `Cn`, `Cnv`, `Cnh`, `Dn`, `Dnh`, `Dnd`, `S2n`,
    /// `T`, `Td`, `Th`, `O`, `Oh`, `I`, `Ih`, `Cinfv`, `Dinfh`, or `Kh` (a single atom).
    pub name: String,
    /// Rotational symmetry number: the number of proper rotations of the group (including
    /// the identity); 1 for `Cinfv`, 2 for `Dinfh`, 1 for an atom.
    pub symmetry_number: u32,
}

impl PointGroup {
    fn new(name: String) -> Self {
        let symmetry_number = rotational_symmetry_number(&name);
        Self { name, symmetry_number }
    }
}

/// Rotational symmetry number of a Schoenflies symbol (see [`PointGroup::symmetry_number`]).
pub fn rotational_symmetry_number(name: &str) -> u32 {
    let order = |s: &str| -> u32 { s.trim_end_matches(|c: char| c.is_ascii_alphabetic()).parse().unwrap_or(1) };
    match name {
        "C1" | "Cs" | "Ci" | "Cinfv" | "Kh" => 1,
        "Dinfh" => 2,
        "T" | "Td" | "Th" => 12,
        "O" | "Oh" => 24,
        "I" | "Ih" => 60,
        n if n.starts_with('C') => order(&n[1..]),
        n if n.starts_with('D') => 2 * order(&n[1..]),
        n if n.starts_with('S') => order(&n[1..]) / 2,
        _ => 1,
    }
}

/// xtb's rotational symmetry number of a point group: the program keeps the first three
/// characters of the symbol (`getsymmetry` in `src/thermo.f90`: first letter in lower case
/// for C, D, T, O and I, a third character outside a..v replaced by a blank) and maps them by
/// substring tests (`getsymnum` in `src/getsymnum.f90`; linear molecules: 1 for `c`, 2 for
/// `d`). This equals [`rotational_symmetry_number`] except for C_n and D_n groups with n >= 10
/// (`c10` becomes `c1 `: 1) and the rotation group `I` (1 instead of 60).
pub fn xtb_symmetry_number(name: &str, linear: bool) -> u32 {
    let mut s: Vec<u8> = name.bytes().take(3).collect();
    while s.len() < 3 {
        s.push(b' ');
    }
    if matches!(s[0], b'C' | b'D' | b'T' | b'O' | b'I') {
        s[0] = s[0].to_ascii_lowercase();
    }
    let mut g = String::from_utf8_lossy(&s).into_owned();
    if g == "dih" {
        g = "d6h".into();
    }
    if g == "civ" {
        g = "c6v".into();
    }
    let c3 = g.as_bytes()[2];
    if !(b'a'..=b'v').contains(&c3) {
        g.replace_range(2..3, " ");
    }
    // `pgroup` is character(len=4): the fourth character is a blank
    g.push(' ');
    let has = |t: &str| g.contains(t);
    let mut n = 1;
    let rules_lower: &[(&str, u32)] = &[
        ("ci", 1),
        ("cs", 1),
        ("c2", 2),
        ("c3", 3),
        ("c4", 4),
        ("c5", 5),
        ("c6", 6),
        ("c7", 7),
        ("c8", 8),
        ("c9", 9),
        ("c10", 10),
        ("c11", 11),
        ("s4", 2),
        ("s6", 3),
        ("s8", 4),
        ("d2", 4),
        ("d3", 6),
        ("d4", 8),
        ("d5", 10),
        ("d6", 12),
        ("d7", 14),
        ("d8", 16),
        ("d9", 18),
        ("d10", 20),
        ("t", 12),
        ("th", 12),
        ("td", 12),
        ("o", 24),
        ("oh", 24),
        ("ih", 60),
    ];
    for (t, v) in rules_lower {
        if has(t) {
            n = *v;
        }
    }
    if has("c") && linear {
        n = 1;
    }
    if has("d") && linear {
        n = 2;
    }
    let rules_upper: &[(&str, u32)] = &[
        ("Ci", 1),
        ("Cs", 1),
        ("C2", 2),
        ("C3", 3),
        ("C4", 4),
        ("C5", 5),
        ("C6", 6),
        ("C7", 7),
        ("C8", 8),
        ("C9", 9),
        ("C10", 10),
        ("C11", 11),
        ("S4", 2),
        ("S6", 3),
        ("S8", 4),
        ("D2", 4),
        ("D3", 6),
        ("D4", 8),
        ("D5", 10),
        ("D6", 12),
        ("D7", 14),
        ("D8", 16),
        ("D9", 18),
        ("D10", 20),
        ("T", 12),
        ("Th", 12),
        ("Td", 12),
        ("O", 24),
        ("Oh", 24),
        ("Ih", 60),
    ];
    for (t, v) in rules_upper {
        if has(t) {
            n = *v;
        }
    }
    if has("C") && linear {
        n = 1;
    }
    if has("D") && linear {
        n = 2;
    }
    n
}

type V3 = [f64; 3];
type M3 = [[f64; 3]; 3];

fn dot(a: V3, b: V3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}
fn cross(a: V3, b: V3) -> V3 {
    [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
}
fn norm(a: V3) -> f64 {
    dot(a, a).sqrt()
}
fn scale(a: V3, s: f64) -> V3 {
    [a[0] * s, a[1] * s, a[2] * s]
}
fn add(a: V3, b: V3) -> V3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}
fn sub(a: V3, b: V3) -> V3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}
fn unit(a: V3) -> Option<V3> {
    let n = norm(a);
    (n > 1e-8).then(|| scale(a, 1.0 / n))
}
fn mat_vec(m: &M3, v: V3) -> V3 {
    [dot(m[0], v), dot(m[1], v), dot(m[2], v)]
}

/// Eigenvalues (ascending) and eigenvectors (columns) of a symmetric 3x3 matrix (Jacobi).
pub(crate) fn eigh3(a: &M3) -> (V3, M3) {
    let mut a = *a;
    let mut v = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    for _ in 0..100 {
        let off = a[0][1].abs() + a[0][2].abs() + a[1][2].abs();
        let scale_ = a[0][0].abs() + a[1][1].abs() + a[2][2].abs();
        if off <= 1e-300 || off <= 1e-17 * scale_ {
            break;
        }
        for (p, q) in [(0, 1), (0, 2), (1, 2)] {
            if a[p][q] == 0.0 {
                continue;
            }
            let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
            let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
            let t = if theta == 0.0 { 1.0 } else { t };
            let c = 1.0 / (t * t + 1.0).sqrt();
            let s = t * c;
            for k in 0..3 {
                let (akp, akq) = (a[k][p], a[k][q]);
                a[k][p] = c * akp - s * akq;
                a[k][q] = s * akp + c * akq;
            }
            for k in 0..3 {
                let (apk, aqk) = (a[p][k], a[q][k]);
                a[p][k] = c * apk - s * aqk;
                a[q][k] = s * apk + c * aqk;
            }
            for k in 0..3 {
                let (vkp, vkq) = (v[k][p], v[k][q]);
                v[k][p] = c * vkp - s * vkq;
                v[k][q] = s * vkp + c * vkq;
            }
        }
    }
    let mut idx = [0usize, 1, 2];
    idx.sort_by(|&i, &j| a[i][i].total_cmp(&a[j][j]));
    let w = [a[idx[0]][idx[0]], a[idx[1]][idx[1]], a[idx[2]][idx[2]]];
    let mut vs = [[0.0; 3]; 3];
    for (c, &i) in idx.iter().enumerate() {
        for r in 0..3 {
            vs[r][c] = v[r][i];
        }
    }
    (w, vs)
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kind {
    /// Rotation by 2π/n.
    Rotation(u32),
    /// Rotation by 2π/n followed by reflection through the perpendicular plane.
    Improper(u32),
    Mirror,
    Inversion,
}

/// The operation's matrix for a unit axis (plane normal for a mirror).
fn matrix(kind: Kind, a: V3) -> M3 {
    let outer = |c: f64| -> M3 {
        let mut m = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                m[i][j] = c * a[i] * a[j];
            }
        }
        m
    };
    match kind {
        Kind::Inversion => [[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]],
        Kind::Mirror => {
            let mut m = outer(-2.0);
            for i in 0..3 {
                m[i][i] += 1.0;
            }
            m
        }
        Kind::Rotation(n) | Kind::Improper(n) => {
            let th = 2.0 * std::f64::consts::PI / n as f64;
            let (s, c) = th.sin_cos();
            let c2 = if matches!(kind, Kind::Rotation(_)) { 1.0 - c } else { -(1.0 + c) };
            let mut m = outer(c2);
            for i in 0..3 {
                m[i][i] += c;
            }
            // + sin θ [a]x
            m[0][1] -= s * a[2];
            m[0][2] += s * a[1];
            m[1][0] += s * a[2];
            m[1][2] -= s * a[0];
            m[2][0] -= s * a[1];
            m[2][1] += s * a[0];
            m
        }
    }
}

struct Frame {
    z: Vec<usize>,
    r: Vec<V3>,
    tol: f64,
    primary: f64,
}

impl Frame {
    /// Map every atom with the operation onto its nearest atom of the same element; the
    /// mapping and the largest deviation, or `None` if an atom has no partner within the
    /// primary tolerance or the mapping is not a permutation.
    fn mapping(&self, op: &M3) -> Option<(Vec<usize>, f64)> {
        let n = self.r.len();
        let mut perm = vec![usize::MAX; n];
        let mut used = vec![false; n];
        let mut dmax: f64 = 0.0;
        for i in 0..n {
            let y = mat_vec(op, self.r[i]);
            let mut best = (f64::INFINITY, usize::MAX);
            for j in 0..n {
                if self.z[j] != self.z[i] {
                    continue;
                }
                let d = norm(sub(y, self.r[j]));
                if d < best.0 {
                    best = (d, j);
                }
            }
            if best.0 > self.primary || used[best.1] {
                return None;
            }
            used[best.1] = true;
            perm[i] = best.1;
            dmax = dmax.max(best.0);
        }
        Some((perm, dmax))
    }

    /// Best axis (normal) of `kind` for the atom mapping `perm`, starting from `a`.
    fn refine(&self, kind: Kind, a: V3, perm: &[usize]) -> V3 {
        // f(a) = Σ r_π(i)ᵀ O(a) r_i = const + b·a + c aᵀ M a
        let mut m = [[0.0; 3]; 3];
        let mut v = [0.0; 3];
        for (i, &j) in perm.iter().enumerate() {
            let (x, y) = (self.r[i], self.r[j]);
            for p in 0..3 {
                for q in 0..3 {
                    m[p][q] += 0.5 * (y[p] * x[q] + x[p] * y[q]);
                }
            }
            v = add(v, cross(x, y));
        }
        match kind {
            Kind::Inversion => a,
            // minimize aᵀ M a
            Kind::Mirror => {
                let (_, vecs) = eigh3(&m);
                let e = [vecs[0][0], vecs[1][0], vecs[2][0]];
                if dot(e, a) < 0.0 {
                    scale(e, -1.0)
                } else {
                    e
                }
            }
            Kind::Rotation(n) | Kind::Improper(n) => {
                let th = 2.0 * std::f64::consts::PI / n as f64;
                let (s, c) = th.sin_cos();
                let c2 = if matches!(kind, Kind::Rotation(_)) { 1.0 - c } else { -(1.0 + c) };
                let mnorm = m.iter().flatten().map(|x| x * x).sum::<f64>().sqrt();
                let mu = 2.0 * c2.abs() * mnorm + s.abs() * norm(v) + 1e-300;
                let mut a = a;
                for _ in 0..200 {
                    let ma = mat_vec(&m, a);
                    let g = [
                        s * v[0] + 2.0 * c2 * ma[0] + mu * a[0],
                        s * v[1] + 2.0 * c2 * ma[1] + mu * a[1],
                        s * v[2] + 2.0 * c2 * ma[2] + mu * a[2],
                    ];
                    let Some(next) = unit(g) else { break };
                    let change = norm(sub(next, a));
                    a = next;
                    if change < 1e-13 {
                        break;
                    }
                }
                a
            }
        }
    }

    /// Test operation `kind` along `a`; the refined axis if the element is present.
    fn test(&self, kind: Kind, a: V3) -> Option<V3> {
        let (perm, dev) = self.mapping(&matrix(kind, a))?;
        if dev < 1e-3 * self.tol {
            return Some(a);
        }
        let b = self.refine(kind, a, &perm);
        let (_, dev) = self.mapping(&matrix(kind, b))?;
        (dev < self.tol).then_some(b)
    }
}

/// Unit directions without (near-)duplicates, sign-normalized.
#[derive(Default)]
struct Directions {
    list: Vec<V3>,
    seen: std::collections::HashSet<[i64; 3]>,
}

impl Directions {
    fn push(&mut self, a: V3) {
        let Some(mut u) = unit(a) else { return };
        // canonical sign: first clearly nonzero component positive
        let k = (0..3).find(|&k| u[k].abs() > 1e-6).unwrap_or(0);
        if u[k] < 0.0 {
            u = scale(u, -1.0);
        }
        let key = [(u[0] * 1e4).round() as i64, (u[1] * 1e4).round() as i64, (u[2] * 1e4).round() as i64];
        if self.seen.insert(key) {
            self.list.push(u);
        }
    }
}

fn parallel(a: V3, b: V3) -> bool {
    dot(a, b).abs() > (1.0 - 1e-6)
}
fn perpendicular(a: V3, b: V3) -> bool {
    dot(a, b).abs() < 1e-3
}

/// Merge axes along the same direction, keeping the highest order.
fn merge(axes: &mut Vec<(V3, u32)>, a: V3, n: u32) {
    for (b, m) in axes.iter_mut() {
        if dot(a, *b).abs() > 0.999 {
            *m = (*m).max(n);
            return;
        }
    }
    axes.push((a, n));
}
fn merge_planes(planes: &mut Vec<V3>, a: V3) {
    if !planes.iter().any(|b| dot(a, *b).abs() > 0.999) {
        planes.push(a);
    }
}

/// Point group of a molecule (atomic numbers, positions in bohr) with tolerance `tol` (bohr).
pub fn point_group(numbers: &[usize], positions: &[[f64; 3]], tol: f64) -> PointGroup {
    let nat = numbers.len();
    if nat <= 1 {
        return PointGroup::new("Kh".into());
    }
    let mut c = [0.0; 3];
    for p in positions {
        c = add(c, *p);
    }
    c = scale(c, 1.0 / nat as f64);
    let r: Vec<V3> = positions.iter().map(|p| sub(*p, c)).collect();
    let f = Frame { z: numbers.to_vec(), r, tol, primary: (2.0 * tol).max(0.5) };
    // principal axes of the atomic-number-weighted "inertia" tensor
    let mut t = [[0.0; 3]; 3];
    for (z, x) in f.z.iter().zip(&f.r) {
        let w = *z as f64;
        let r2 = dot(*x, *x);
        for p in 0..3 {
            for q in 0..3 {
                t[p][q] += w * (if p == q { r2 } else { 0.0 } - x[p] * x[q]);
            }
        }
    }
    let (_, pax) = eigh3(&t);
    let principal: Vec<V3> = (0..3).map(|k| [pax[0][k], pax[1][k], pax[2][k]]).collect();
    let inversion = f.test(Kind::Inversion, [0.0, 0.0, 1.0]).is_some();
    // linear: every atom within the tolerance of the line along the least-moment axis
    let lin_axis = principal[0];
    let linear = f.r.iter().all(|x| norm(cross(*x, lin_axis)) < tol);
    if linear {
        return PointGroup::new(if inversion { "Dinfh" } else { "Cinfv" }.into());
    }
    // groups of equivalent atoms: same element, same distance from the centroid
    let radius: Vec<f64> = f.r.iter().map(|x| norm(*x)).collect();
    let equivalent = |i: usize, j: usize| f.z[i] == f.z[j] && (radius[i] - radius[j]).abs() < 2.0 * tol;
    let mut axis_dirs = Directions::default();
    let mut plane_dirs = Directions::default();
    for p in &principal {
        axis_dirs.push(*p);
        plane_dirs.push(*p);
    }
    for i in 0..nat {
        if radius[i] > tol {
            axis_dirs.push(f.r[i]);
        }
        for j in i + 1..nat {
            if !equivalent(i, j) {
                continue;
            }
            axis_dirs.push(add(f.r[i], f.r[j]));
            axis_dirs.push(cross(f.r[i], f.r[j]));
            plane_dirs.push(sub(f.r[i], f.r[j]));
        }
    }
    // normals of triangles of equivalent atoms (axes through face centers), for small sets
    let mut classes: Vec<Vec<usize>> = Vec::new();
    for i in 0..nat {
        match classes.iter_mut().find(|cl| equivalent(cl[0], i)) {
            Some(cl) => cl.push(i),
            None => classes.push(vec![i]),
        }
    }
    for cl in &classes {
        if cl.len() < 3 || cl.len() > 60 {
            continue;
        }
        for a in 0..cl.len() {
            for b in a + 1..cl.len() {
                for d in b + 1..cl.len() {
                    let (x, y, z) = (f.r[cl[a]], f.r[cl[b]], f.r[cl[d]]);
                    axis_dirs.push(cross(sub(y, x), sub(z, x)));
                }
            }
        }
    }
    // proper axes, highest order per direction
    let mut axes: Vec<(V3, u32)> = Vec::new();
    for a in &axis_dirs.list {
        for n in (2..=MAX_ORDER).rev() {
            if let Some(b) = f.test(Kind::Rotation(n), *a) {
                merge(&mut axes, b, n);
                break;
            }
        }
    }
    // mirror planes
    for (a, _) in &axes {
        plane_dirs.push(*a);
        for x in &f.r {
            plane_dirs.push(cross(*a, *x));
        }
    }
    let mut planes: Vec<V3> = Vec::new();
    for a in &plane_dirs.list {
        if let Some(b) = f.test(Kind::Mirror, *a) {
            merge_planes(&mut planes, b);
        }
    }
    // improper axes S_2n along the proper axes
    let improper: Vec<(V3, u32)> =
        axes.iter().filter_map(|(a, n)| f.test(Kind::Improper(2 * n), *a).map(|b| (b, 2 * n))).collect();
    PointGroup::new(classify(&axes, &planes, &improper, inversion))
}

fn classify(axes: &[(V3, u32)], planes: &[V3], improper: &[(V3, u32)], inversion: bool) -> String {
    let high: Vec<&(V3, u32)> = axes.iter().filter(|(_, n)| *n >= 3).collect();
    if high.len() >= 2 {
        let has = |k: u32| high.iter().any(|(_, n)| n % k == 0);
        return if has(5) {
            if inversion {
                "Ih"
            } else {
                "I"
            }
        } else if has(4) {
            if inversion {
                "Oh"
            } else {
                "O"
            }
        } else if inversion {
            "Th"
        } else if !planes.is_empty() {
            "Td"
        } else {
            "T"
        }
        .into();
    }
    if axes.is_empty() {
        return if !planes.is_empty() {
            "Cs"
        } else if inversion {
            "Ci"
        } else {
            "C1"
        }
        .into();
    }
    let nmax = axes.iter().map(|(_, n)| *n).max().unwrap_or(1);
    // main axis: of the highest order; among several C2 axes the one along an S4 axis
    let candidates: Vec<V3> = axes.iter().filter(|(_, n)| *n == nmax).map(|(a, _)| *a).collect();
    let main = candidates
        .iter()
        .copied()
        .find(|a| improper.iter().any(|(b, m)| *m == 2 * nmax && parallel(*a, *b)))
        .unwrap_or(candidates[0]);
    let n_perp = axes.iter().filter(|(a, k)| k % 2 == 0 && perpendicular(*a, main)).count() as u32;
    let sigma_h = planes.iter().any(|p| parallel(*p, main));
    let n_sigma_v = planes.iter().filter(|p| perpendicular(**p, main)).count() as u32;
    let s2n = improper.iter().any(|(b, m)| *m == 2 * nmax && parallel(*b, main));
    if n_perp >= nmax {
        if sigma_h {
            format!("D{nmax}h")
        } else if n_sigma_v >= nmax {
            format!("D{nmax}d")
        } else {
            format!("D{nmax}")
        }
    } else if sigma_h {
        format!("C{nmax}h")
    } else if n_sigma_v >= 1 {
        format!("C{nmax}v")
    } else if s2n {
        format!("S{}", 2 * nmax)
    } else {
        format!("C{nmax}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rot_z(p: V3, th: f64) -> V3 {
        let (s, c) = th.sin_cos();
        [c * p[0] - s * p[1], s * p[0] + c * p[1], p[2]]
    }

    fn ring(z: usize, n: usize, r: f64, h: f64, phase: f64) -> Vec<(usize, V3)> {
        (0..n).map(|k| (z, rot_z([r, 0.0, h], phase + 2.0 * std::f64::consts::PI * k as f64 / n as f64))).collect()
    }

    fn pg(atoms: &[(usize, V3)]) -> String {
        let z: Vec<usize> = atoms.iter().map(|a| a.0).collect();
        // arbitrary orientation and origin
        let pos: Vec<V3> = atoms
            .iter()
            .map(|a| {
                let p = rot_z(a.1, 0.3);
                let (s, c) = 0.7f64.sin_cos();
                add([p[0], c * p[1] - s * p[2], s * p[1] + c * p[2]], [1.0, -2.0, 0.5])
            })
            .collect();
        point_group(&z, &pos, 0.1).name
    }

    #[test]
    fn groups() {
        let ch4 = {
            let s = 1.2;
            vec![(6, [0.0; 3]), (1, [s, s, s]), (1, [s, -s, -s]), (1, [-s, s, -s]), (1, [-s, -s, s])]
        };
        assert_eq!(pg(&ch4), "Td");
        let mut benzene = ring(6, 6, 2.64, 0.0, 0.0);
        benzene.extend(ring(1, 6, 4.68, 0.0, 0.0));
        assert_eq!(pg(&benzene), "D6h");
        let mut nh3 = ring(1, 3, 1.8, -0.7, 0.0);
        nh3.push((7, [0.0, 0.0, 0.0]));
        assert_eq!(pg(&nh3), "C3v");
        let water = vec![(8, [0.0, 0.0, 0.0]), (1, [1.43, 0.0, 1.1]), (1, [-1.43, 0.0, 1.1])];
        assert_eq!(pg(&water), "C2v");
        let co2 = vec![(6, [0.0; 3]), (8, [0.0, 0.0, 2.2]), (8, [0.0, 0.0, -2.2])];
        assert_eq!(pg(&co2), "Dinfh");
        let hcn = vec![(6, [0.0; 3]), (1, [0.0, 0.0, -2.0]), (7, [0.0, 0.0, 2.2])];
        assert_eq!(pg(&hcn), "Cinfv");
        let sf6 = {
            let s = 3.0;
            vec![
                (16, [0.0; 3]),
                (9, [s, 0.0, 0.0]),
                (9, [-s, 0.0, 0.0]),
                (9, [0.0, s, 0.0]),
                (9, [0.0, -s, 0.0]),
                (9, [0.0, 0.0, s]),
                (9, [0.0, 0.0, -s]),
            ]
        };
        assert_eq!(pg(&sf6), "Oh");
        // ethane staggered (D3d) and eclipsed (D3h)
        let mut stag = vec![(6, [0.0, 0.0, 1.45]), (6, [0.0, 0.0, -1.45])];
        stag.extend(ring(1, 3, 1.95, 2.1, 0.0));
        stag.extend(ring(1, 3, 1.95, -2.1, std::f64::consts::PI / 3.0));
        assert_eq!(pg(&stag), "D3d");
        let mut ecl = vec![(6, [0.0, 0.0, 1.45]), (6, [0.0, 0.0, -1.45])];
        ecl.extend(ring(1, 3, 1.95, 2.1, 0.0));
        ecl.extend(ring(1, 3, 1.95, -2.1, 0.0));
        assert_eq!(pg(&ecl), "D3h");
        // allene-like D2d: two perpendicular CH2 groups
        let allene = vec![
            (6, [0.0; 3]),
            (6, [0.0, 0.0, 2.5]),
            (6, [0.0, 0.0, -2.5]),
            (1, [1.75, 0.0, 3.5]),
            (1, [-1.75, 0.0, 3.5]),
            (1, [0.0, 1.75, -3.5]),
            (1, [0.0, -1.75, -3.5]),
        ];
        assert_eq!(pg(&allene), "D2d");
        // trans-HN=NH (C2h), H2O2-like C2, CHFClBr-like C1, Cs
        let c2h = vec![(7, [0.0, 0.6, 0.0]), (7, [0.0, -0.6, 0.0]), (1, [1.8, 1.2, 0.0]), (1, [-1.8, -1.2, 0.0])];
        assert_eq!(pg(&c2h), "C2h");
        let c2 = vec![(8, [0.0, 1.4, 0.0]), (8, [0.0, -1.4, 0.0]), (1, [1.8, 1.7, 0.5]), (1, [-1.8, -1.7, 0.5])];
        assert_eq!(pg(&c2), "C2");
        let c1 = vec![
            (6, [0.0; 3]),
            (1, [1.2, 1.1, 0.9]),
            (9, [-1.3, 1.2, -1.0]),
            (17, [1.5, -1.6, -1.2]),
            (35, [-1.8, -1.5, 1.9]),
        ];
        assert_eq!(pg(&c1), "C1");
        let cs = vec![(6, [0.0; 3]), (1, [1.2, 1.1, 0.0]), (9, [-1.3, 1.2, 0.0]), (17, [1.5, -1.6, 0.0])];
        assert_eq!(pg(&cs), "Cs");
        // D3 propeller: three tilted pairs; S4 (tetraphenylmethane-like is too big; use a twisted D2d to S4)
        let s4 = {
            let (a, b) = (1.5, 0.6);
            vec![
                (6, [0.0; 3]),
                (1, [a, b, 1.0]),
                (1, [-a, -b, 1.0]),
                (1, [b, -a, -1.0]),
                (1, [-b, a, -1.0]),
                (9, [a + 1.0, b + 1.4, 1.6]),
                (9, [-a - 1.0, -b - 1.4, 1.6]),
                (9, [b + 1.4, -a - 1.0, -1.6]),
                (9, [-b - 1.4, a + 1.0, -1.6]),
            ]
        };
        assert_eq!(pg(&s4), "S4");
    }

    #[test]
    fn symmetry_numbers() {
        for (name, n) in [
            ("C1", 1),
            ("Cs", 1),
            ("C2v", 2),
            ("C3v", 3),
            ("D6h", 12),
            ("D3d", 6),
            ("Td", 12),
            ("Oh", 24),
            ("Ih", 60),
            ("S4", 2),
            ("S6", 3),
            ("Cinfv", 1),
            ("Dinfh", 2),
        ] {
            assert_eq!(rotational_symmetry_number(name), n, "{name}");
            let linear = name.contains("inf");
            assert_eq!(xtb_symmetry_number(name, linear), n, "{name}");
        }
        // xtb's quirks: three-character truncation and the rotation group I
        assert_eq!(xtb_symmetry_number("C10v", false), 1);
        assert_eq!(xtb_symmetry_number("D10h", false), 1);
        assert_eq!(xtb_symmetry_number("I", false), 1);
        assert_eq!(rotational_symmetry_number("C10v"), 10);
        assert_eq!(rotational_symmetry_number("D10h"), 20);
    }
}
