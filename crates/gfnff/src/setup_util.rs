//! Helpers of the GFN-FF topology setup, ported from xtb `src/gfnff/gfnff_ini2.F90`,
//! `mrec.f90` and `src/mctc/sort.f90`.

use crate::conventions::TieNorms;

/// Neighbor lists: `nb[i]` is the ordered list of neighbors of atom i (0-based).
pub type Nb = Vec<Vec<usize>>;

/// `indexHeapSort` of mctc (Numerical Recipes heap sort returning an index, tolerance
/// `epsilon`). Returns the permutation (0-based) that sorts `array` ascending.
pub fn index_heap_sort(array: &[f64]) -> Vec<usize> {
    let tol = f64::EPSILON;
    let n = array.len();
    // 1-based work array like the Fortran
    let mut indx: Vec<usize> = (0..=n).collect();
    if n <= 1 {
        return (0..n).collect();
    }
    let a = |k: usize| array[k - 1];
    let mut il = n / 2 + 1;
    let mut ir = n;
    loop {
        let indx_tmp;
        let array_tmp;
        if il > 1 {
            il -= 1;
            indx_tmp = indx[il];
            array_tmp = a(indx_tmp);
        } else {
            indx_tmp = indx[ir];
            array_tmp = a(indx_tmp);
            indx[ir] = indx[1];
            ir -= 1;
            if ir < 1 {
                indx[1] = indx_tmp;
                break;
            }
        }
        let mut ii = il;
        let mut ij = 2 * il;
        while ij <= ir {
            if ij < ir && a(indx[ij]) < a(indx[ij + 1]) - tol {
                ij += 1;
            }
            if array_tmp < a(indx[ij]) - tol {
                indx[ii] = indx[ij];
                ii = ij;
                ij *= 2;
            } else {
                ij = ir + 1;
            }
        }
        indx[ii] = indx_tmp;
    }
    indx[1..].iter().map(|&k| k - 1).collect()
}

/// `jth_nb`: the jth (1-based) nearest neighbor of atom i (by distance at `xyz`, rounded as
/// the reference binary rounds it, [`TieNorms`]), as xtb's heap sort orders them; None
/// without neighbors.
pub fn jth_nb(nb: &[usize], xyz: &[[f64; 3]], i: usize, jth: usize, norms: TieNorms) -> Option<usize> {
    if nb.is_empty() || jth == 0 || jth > nb.len() {
        return None;
    }
    let n = nb.len();
    let dist: Vec<f64> = nb
        .iter()
        .enumerate()
        .map(|(k, &j)| norms.jth_nb([xyz[i][0] - xyz[j][0], xyz[i][1] - xyz[j][1], xyz[i][2] - xyz[j][2]], k, n))
        .collect();
    let idx = index_heap_sort(&dist);
    Some(nb[idx[jth - 1]])
}

/// `mrecgffPBC`: fragment number (1-based) of every atom by directed reachability over
/// `nb`, seeds in index order (the set reached does not depend on the traversal order).
pub fn mrec(nb: &Nb) -> (usize, Vec<usize>) {
    let n = nb.len();
    let mut molvec = vec![0usize; n];
    let mut taken = vec![false; n];
    let mut count = 0;
    let mut stack = Vec::new();
    for i in 0..n {
        if taken[i] {
            continue;
        }
        count += 1;
        molvec[i] = count;
        taken[i] = true;
        stack.push(i);
        while let Some(a) = stack.pop() {
            for &j in &nb[a] {
                if j == a {
                    continue;
                }
                if !taken[j] {
                    taken[j] = true;
                    molvec[j] = count;
                    stack.push(j);
                }
            }
        }
    }
    (count, molvec)
}

/// `ssort`: selection sort of `edum` ascending, permuting `ind` alongside (ties pick the
/// last minimum, as the Fortran does).
pub fn ssort(edum: &mut [f64], ind: &mut [usize]) {
    let n = edum.len();
    for ii in 1..n {
        let i = ii - 1;
        let mut k = i;
        let mut pp = edum[i];
        for j in ii..n {
            if edum[j] > pp {
                continue;
            }
            k = j;
            pp = edum[j];
        }
        if k == i {
            continue;
        }
        edum[k] = edum[i];
        edum[i] = pp;
        ind.swap(i, k);
    }
}

/// Ring data of one atom (`getring36` output): up to 19 rings of size 3-6.
#[derive(Clone, Debug, Default)]
pub struct AtomRings {
    /// ring sizes
    pub size: Vec<usize>,
    /// atoms of each ring (1-based as in xtb for the set comparisons; 0 = unused)
    pub atoms: Vec<[usize; 10]>,
}

/// `getring36`: rings of size 3 to 6 that contain atom `a0` (0-based), using the no-metal
/// neighbor list `nbin` with the cluster tags `tag` (tagged atoms are skipped).
pub fn getring36(nbin: &Nb, tag: &[bool], a0: usize) -> AtomRings {
    const MAXR: usize = 500;
    let n = nbin.len();
    let mut out = AtomRings::default();
    if n <= 2 || tag[a0] {
        return out;
    }
    // counts with terminal atoms (exactly one neighbor) pruned
    let cnt: Vec<usize> = nbin.iter().map(|l| if l.len() == 1 { 0 } else { l.len() }).collect();
    let nn = nbin[a0].len();
    let mut cdum: Vec<[usize; 10]> = Vec::new();
    let mut idum: Vec<usize> = Vec::new();
    // 1-based atom ids for chkrng/the comparisons
    let chkrng = |c: &[usize]| -> bool {
        // all distinct
        for x in 0..c.len() {
            for y in 0..x {
                if c[x] == c[y] {
                    return false;
                }
            }
        }
        true
    };
    let mut full = false;
    for m in 0..nn {
        // `if(nb(m,a0_in).eq.1) cycle`: the m-th neighbor is atom number 1
        if nbin[a0][m] == 0 {
            continue;
        }
        // neighbor order of a0 with neighbor m first (ssort of w = 1..nn with w(m)=0)
        let mut w: Vec<f64> = (1..=nn).map(|v| v as f64).collect();
        let mut list: Vec<usize> = (0..nn).collect();
        w[m] = 0.0;
        ssort(&mut w, &mut list);
        let nb_a0: Vec<usize> = list.iter().map(|&l| nbin[a0][l]).collect();
        let nbr = |a: usize, k: usize| -> usize {
            if a == a0 {
                nb_a0[k]
            } else {
                nbin[a][k]
            }
        };
        let n0 = cnt[a0];
        let mut c = [0usize; 10];
        'search: for i1 in 0..n0 {
            let a1 = nbr(a0, i1);
            if a1 == a0 {
                continue;
            }
            let n1 = cnt[a1];
            for i2 in 0..n1 {
                let a2 = nbr(a1, i2);
                if a2 == a1 {
                    continue;
                }
                let n2 = cnt[a2];
                for i3 in 0..n2 {
                    let a3 = nbr(a2, i3);
                    let n3 = cnt[a3];
                    if a3 == a2 {
                        continue;
                    }
                    c[0] = a1 + 1;
                    c[1] = a2 + 1;
                    c[2] = a3 + 1;
                    if a3 == a0 && chkrng(&c[..3]) {
                        if cdum.len() == MAXR {
                            full = true;
                            break 'search;
                        }
                        let mut r = [0usize; 10];
                        r[..3].copy_from_slice(&c[..3]);
                        cdum.push(r);
                        idum.push(3);
                    }
                    for i4 in 0..n3 {
                        let a4 = nbr(a3, i4);
                        let n4 = cnt[a4];
                        if a4 == a3 {
                            continue;
                        }
                        c[3] = a4 + 1;
                        if a4 == a0 && chkrng(&c[..4]) {
                            if cdum.len() == MAXR {
                                full = true;
                                break 'search;
                            }
                            let mut r = [0usize; 10];
                            r[..4].copy_from_slice(&c[..4]);
                            cdum.push(r);
                            idum.push(4);
                        }
                        for i5 in 0..n4 {
                            let a5 = nbr(a4, i5);
                            let n5 = cnt[a5];
                            if a5 == a4 {
                                continue;
                            }
                            c[4] = a5 + 1;
                            if a5 == a0 && chkrng(&c[..5]) {
                                if cdum.len() == MAXR {
                                    full = true;
                                    break 'search;
                                }
                                let mut r = [0usize; 10];
                                r[..5].copy_from_slice(&c[..5]);
                                cdum.push(r);
                                idum.push(5);
                            }
                            for i6 in 0..n5 {
                                let a6 = nbr(a5, i6);
                                if a6 == a5 {
                                    continue;
                                }
                                c[5] = a6 + 1;
                                if a6 == a0 && chkrng(&c[..6]) {
                                    if cdum.len() == MAXR {
                                        full = true;
                                        break 'search;
                                    }
                                    let mut r = [0usize; 10];
                                    r[..6].copy_from_slice(&c[..6]);
                                    cdum.push(r);
                                    idum.push(6);
                                }
                            }
                        }
                    }
                }
            }
        }
        // every later m repeats the same search in another order: its rings are duplicates
        // of these unless the list overflowed, which ends all searches anyway
        if !full || cdum.len() >= MAXR {
            break;
        }
    }
    // remove duplicates (same size and same atom set), keep first occurrences
    let kk = cdum.len();
    let mut same = vec![false; kk];
    for i in 0..kk {
        for j in i + 1..kk {
            if idum[i] != idum[j] || same[j] {
                continue;
            }
            let mut a: Vec<usize> = cdum[i].to_vec();
            let mut b: Vec<usize> = cdum[j].to_vec();
            a.sort_unstable();
            b.sort_unstable();
            a.dedup();
            b.dedup();
            same[j] = a == b;
        }
    }
    for i in 0..kk {
        if !same[i] {
            out.size.push(idum[i]);
            out.atoms.push(cdum[i]);
            if out.size.len() > 19 {
                out.size.truncate(19);
                out.atoms.truncate(19);
                break;
            }
        }
    }
    out
}

/// `ringsatom`: smallest ring of atom i (99 if none).
pub fn ringsatom(r: &[AtomRings], i: usize) -> usize {
    let mut rings = 99;
    for &s in &r[i].size {
        if s < rings {
            rings = s;
        }
    }
    rings
}

/// `ringsbond`: smallest ring containing bond i-j (0 if none).
pub fn ringsbond(r: &[AtomRings], i: usize, j: usize) -> usize {
    let mut r1 = 99;
    let mut r2 = 99;
    for (k, &s) in r[i].size.iter().enumerate() {
        for l in 0..s {
            if r[i].atoms[k][l] == j + 1 && s < r1 {
                r1 = s;
            }
        }
    }
    for (k, &s) in r[j].size.iter().enumerate() {
        for l in 0..s {
            if r[j].atoms[k][l] == i + 1 && s < r2 {
                r2 = s;
            }
        }
    }
    let rings = r1.min(r2);
    if rings == 99 {
        0
    } else {
        rings
    }
}

/// `ringsbend`: smallest ring containing angle i-j-k (xtb's quirk for the third atom kept).
pub fn ringsbend(r: &[AtomRings], i: usize, j: usize, k: usize) -> usize {
    if r[i].size.is_empty() || r[j].size.is_empty() || r[k].size.is_empty() {
        return 0;
    }
    let (i1, j1, k1) = (i + 1, j + 1, k + 1);
    let mut r1 = 99;
    let mut r2 = 99;
    let mut r3 = 99;
    for (m, &s) in r[i].size.iter().enumerate() {
        let t = (0..s).filter(|&l| r[i].atoms[m][l] == j1 || r[i].atoms[m][l] == k1).count();
        if t == 2 && s < r1 {
            r1 = s;
        }
    }
    for (m, &s) in r[j].size.iter().enumerate() {
        let t = (0..s).filter(|&l| r[j].atoms[m][l] == i1 || r[j].atoms[m][l] == k1).count();
        if t == 2 && s < r2 {
            r2 = s;
        }
    }
    for (m, &s) in r[k].size.iter().enumerate() {
        let t = (0..s).filter(|&l| r[k].atoms[m][l] == i1 || r[k].atoms[m][l] == j1).count();
        // xtb: `if(itest.eq.2.and.s(m,j).lt.rings3) rings3=s(m,k)` (s of atom j, ring m)
        let sj = r[j].size.get(m).copied().unwrap_or(0);
        if t == 2 && sj < r3 {
            r3 = s;
        }
    }
    let rings = r1.min(r2).min(r3);
    if rings == 99 {
        0
    } else {
        rings
    }
}

fn rings_with_three(r: &[AtomRings], a: usize, others: [usize; 3]) -> impl Iterator<Item = usize> + '_ {
    let o = others.map(|x| x + 1);
    r[a].size.iter().enumerate().filter_map(move |(m, &s)| {
        let t = (0..s).filter(|&l| o.contains(&r[a].atoms[m][l])).count();
        if t == 3 {
            Some(s)
        } else {
            None
        }
    })
}

/// `ringstors`: smallest ring containing all of i, j, k, l (0 if none).
pub fn ringstors(r: &[AtomRings], i: usize, j: usize, k: usize, l: usize) -> usize {
    if r[i].size.is_empty() || r[j].size.is_empty() || r[k].size.is_empty() || r[l].size.is_empty() {
        return 0;
    }
    let mut best = 99;
    for (a, others) in [(i, [j, k, l]), (j, [i, k, l]), (k, [i, j, l]), (l, [i, k, j])] {
        for s in rings_with_three(r, a, others) {
            if s < best {
                best = s;
            }
        }
    }
    if best == 99 {
        0
    } else {
        best
    }
}

/// `ringstorl`: largest ring containing all of i, j, k, l (0 if none).
pub fn ringstorl(r: &[AtomRings], i: usize, j: usize, k: usize, l: usize) -> usize {
    if r[i].size.is_empty() || r[j].size.is_empty() || r[k].size.is_empty() || r[l].size.is_empty() {
        return 0;
    }
    let mut best: i64 = -99;
    for (a, others) in [(i, [j, k, l]), (j, [i, k, l]), (k, [i, j, l]), (l, [i, k, j])] {
        for s in rings_with_three(r, a, others) {
            if s as i64 > best {
                best = s as i64;
            }
        }
    }
    if best == -99 {
        0
    } else {
        best as usize
    }
}

/// `ctype`: 1 for a carbonyl-type C (exactly one pi O neighbor and a pi atom itself).
pub fn ctype(numbers: &[usize], nb: &Nb, pi: &dyn Fn(usize) -> bool, a: usize) -> i32 {
    let no = nb[a].iter().filter(|&&j| numbers[j] == 8 && pi(j)).count();
    if no == 1 && pi(a) {
        1
    } else {
        0
    }
}

/// `alphaCO`: a-b is a C(sp3)-C(=O) bond.
pub fn alpha_co(numbers: &[usize], hyb: &[i32], nb: &Nb, pi: &dyn Fn(usize) -> bool, a: usize, b: usize) -> bool {
    let check = |x: usize, y: usize| -> bool {
        if pi(x) && hyb[y] == 3 && numbers[x] == 6 && numbers[y] == 6 {
            let no = nb[x].iter().filter(|&&j| numbers[j] == 8 && pi(j) && nb[j].len() == 1).count();
            no == 1
        } else {
            false
        }
    };
    check(a, b) || check(b, a)
}

/// `amide`: a is an sp3 pi N with exactly one pi C neighbor that carries one terminal pi O.
pub fn amide(numbers: &[usize], hyb: &[i32], nb: &Nb, pi: &dyn Fn(usize) -> bool, a: usize) -> bool {
    if !pi(a) || hyb[a] != 3 || numbers[a] != 7 {
        return false;
    }
    let mut nc = 0;
    let mut ic = 0;
    for &j in &nb[a] {
        if numbers[j] == 6 && pi(j) {
            nc += 1;
            ic = j;
        }
    }
    let mut no = 0;
    if nc == 1 {
        for &j in &nb[ic] {
            if numbers[j] == 8 && pi(j) && nb[j].len() == 1 {
                no += 1;
            }
        }
    }
    no == 1
}

/// `amideH`: a (one neighbor) sits on an amide N that carries exactly one sp3 C.
pub fn amide_h(numbers: &[usize], hyb: &[i32], nb: &Nb, pi: &dyn Fn(usize) -> bool, a: usize) -> bool {
    if nb[a].len() != 1 {
        return false;
    }
    let nn = nb[a][0];
    if !amide(numbers, hyb, nb, pi, nn) {
        return false;
    }
    let nc = nb[nn].iter().filter(|&&j| numbers[j] == 6 && hyb[j] == 3).count();
    nc == 1
}

/// `qheavy`: condense H charges onto their neighbors.
pub fn qheavy(numbers: &[usize], nb: &Nb, q: &mut [f64]) {
    let mut qtmp = q.to_vec();
    for i in 0..numbers.len() {
        if numbers[i] != 1 {
            continue;
        }
        qtmp[i] = 0.0;
        let cnt = nb[i].len() as f64;
        for &k in &nb[i] {
            qtmp[k] += q[i] / cnt;
        }
    }
    q.copy_from_slice(&qtmp);
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn heap_sort_orders() {
        let a = [3.0, 1.0, 2.0, 5.0, 4.0];
        assert_eq!(index_heap_sort(&a), vec![1, 2, 0, 4, 3]);
    }
    #[test]
    fn ssort_orders() {
        let mut e = [3.0, 1.0, 2.0];
        let mut i = [10, 11, 12];
        ssort(&mut e, &mut i);
        assert_eq!(i, [11, 12, 10]);
    }
}
