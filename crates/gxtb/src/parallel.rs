//! g-xTB's deterministic parallel loops on top of `axtb_base::threads`: work is cut into
//! fixed chunks that do not depend on the thread count, each chunk is evaluated on its own (on
//! the current rayon pool when `parallel`, in order otherwise) and the chunk results are
//! combined in chunk order, so that reductions over them give the same bits for every thread
//! count. Without `parallel` no rayon call is made (a calculation with one thread spawns no
//! threads).
//!
//! The basic loops (`map`, `map_chunks` over the fixed `chunks`, `for_slices`) and the rule for
//! parallel dense linear algebra (`dense_par`, `DENSE_MIN_AO`) are axtb-base's; this module
//! adds the tiled fills, sums, row reductions and symmetric-matrix builders of g-xTB's terms.

use axtb_base::threads::{for_slices, map_chunks, NCHUNK};
use rayon::prelude::*;

/// Tile edge of the passes that read a matrix transposed.
pub const TILE: usize = 32;

/// `out[i n + j] = f(i, j)` for an n x n row-major `out`, in TILE x TILE tiles (so that reads
/// of a transposed element stay in cache), tile rows in parallel when `parallel`.
pub fn fill_tiled(out: &mut [f64], n: usize, parallel: bool, f: impl Fn(usize, usize) -> f64 + Sync + Send) {
    for_slices(out, TILE * n, parallel, |t, rows| {
        let i0 = t * TILE;
        let i1 = (i0 + TILE).min(n);
        for j0 in (0..n).step_by(TILE) {
            let j1 = (j0 + TILE).min(n);
            for i in i0..i1 {
                let row = &mut rows[(i - i0) * n..(i - i0 + 1) * n];
                for j in j0..j1 {
                    row[j] = f(i, j);
                }
            }
        }
    });
}

/// `out[i n + j] = f(i, j, out[i n + j])` for an n x n row-major `out`, in the tiles of
/// [`fill_tiled`].
pub fn update_tiled(out: &mut [f64], n: usize, parallel: bool, f: impl Fn(usize, usize, f64) -> f64 + Sync + Send) {
    for_slices(out, TILE * n, parallel, |t, rows| {
        let i0 = t * TILE;
        let i1 = (i0 + TILE).min(n);
        for j0 in (0..n).step_by(TILE) {
            let j1 = (j0 + TILE).min(n);
            for i in i0..i1 {
                let row = &mut rows[(i - i0) * n..(i - i0 + 1) * n];
                for j in j0..j1 {
                    row[j] = f(i, j, row[j]);
                }
            }
        }
    });
}

/// Sum of `f(k)` over `0..n` in fixed chunks, chunk sums added in order.
pub fn sum(n: usize, parallel: bool, f: impl Fn(usize) -> f64 + Sync + Send) -> f64 {
    map_chunks(n, parallel, |r| r.map(&f).sum::<f64>()).into_iter().sum()
}

/// `f(k, &mut out[k])` for every element, in chunks in parallel when `parallel`.
pub fn for_each_mut(out: &mut [f64], parallel: bool, f: impl Fn(usize, &mut f64) + Sync + Send) {
    let len = out.len().div_ceil(NCHUNK).max(1);
    for_slices(out, len, parallel, |c, part| {
        for (k, x) in part.iter_mut().enumerate() {
            f(c * len + k, x);
        }
    });
}

/// `f(i, &mut acc)` for every row `i` of `0..n` with an accumulator per fixed chunk of rows
/// (`init()`; in parallel when `parallel`), the chunk accumulators combined in order with `add`.
/// For pair loops that add to both atoms of a pair.
pub fn reduce_rows<A: Send>(
    n: usize,
    parallel: bool,
    init: impl Fn() -> A + Sync + Send,
    f: impl Fn(usize, &mut A) + Sync + Send,
    add: impl Fn(&mut A, A),
) -> A {
    let mut parts = map_chunks(n, parallel, |rows| {
        let mut acc = init();
        for i in rows {
            f(i, &mut acc);
        }
        acc
    })
    .into_iter();
    let mut out = parts.next().unwrap_or_else(&init);
    for p in parts {
        add(&mut out, p);
    }
    out
}

/// Add `b` to `a` elementwise.
pub fn add_into(a: &mut [f64], b: &[f64]) {
    for (x, y) in a.iter_mut().zip(b) {
        *x += y;
    }
}

/// Fill the upper triangle of a symmetric row-major n x n matrix from its lower triangle
/// (tiled, rows in parallel when `parallel`).
pub fn mirror_lower(m: &mut [f64], n: usize, parallel: bool) {
    let src = m.to_vec();
    fill_tiled(m, n, parallel, |i, j| if j > i { src[j * n + i] } else { src[i * n + j] });
}

/// Split a row-major matrix with rows of length `n` into the row blocks `bounds[k]..bounds[k + 1]`.
pub fn row_blocks<'a>(m: &'a mut [f64], n: usize, bounds: &[usize]) -> Vec<&'a mut [f64]> {
    let mut out = Vec::with_capacity(bounds.len().saturating_sub(1));
    let mut rest = m;
    for w in bounds.windows(2) {
        let (a, b) = std::mem::take(&mut rest).split_at_mut((w[1] - w[0]) * n);
        out.push(a);
        rest = b;
    }
    out
}

/// `f(k, blocks)` for every row block k of every matrix in `mats` (the k-th blocks of all
/// matrices together; [`row_blocks`]), in parallel when `parallel`.
pub fn for_row_blocks(
    mats: &mut [&mut [f64]],
    n: usize,
    bounds: &[usize],
    parallel: bool,
    f: impl Fn(usize, &mut [&mut [f64]]) + Sync + Send,
) {
    let mut its: Vec<_> = mats.iter_mut().map(|m| row_blocks(m, n, bounds).into_iter()).collect();
    let blocks: Vec<Vec<&mut [f64]>> =
        (0..bounds.len().saturating_sub(1)).map(|_| its.iter_mut().map(|i| i.next().unwrap()).collect()).collect();
    if parallel {
        blocks.into_par_iter().enumerate().for_each(|(k, mut b)| f(k, &mut b));
    } else {
        blocks.into_iter().enumerate().for_each(|(k, mut b)| f(k, &mut b));
    }
}

/// K row-major n x n matrices with their lower triangles (j <= i) set to `f(i, j)` (rows in
/// parallel when `parallel`), the strict upper triangles zero.
pub fn lower<const K: usize>(
    n: usize,
    parallel: bool,
    f: impl Fn(usize, usize) -> [f64; K] + Sync + Send,
) -> [Vec<f64>; K] {
    let mut out: [Vec<f64>; K] = std::array::from_fn(|_| vec![0.0; n * n]);
    {
        let mut views: Vec<&mut [f64]> = out.iter_mut().map(|m| m.as_mut_slice()).collect();
        let bounds: Vec<usize> = (0..=n).collect();
        for_row_blocks(&mut views, n, &bounds, parallel, |i, rows| {
            for j in 0..=i {
                let v = f(i, j);
                for k in 0..K {
                    rows[k][j] = v[k];
                }
            }
        });
    }
    out
}

/// K symmetric n x n matrices from their lower triangles `f(i, j)` (j <= i; [`lower`]).
pub fn symmetric<const K: usize>(
    n: usize,
    parallel: bool,
    f: impl Fn(usize, usize) -> [f64; K] + Sync + Send,
) -> [Vec<f64>; K] {
    let mut out = lower(n, parallel, f);
    for m in out.iter_mut() {
        mirror_lower(m, n, parallel);
    }
    out
}
