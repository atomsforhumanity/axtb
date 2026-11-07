//! Deterministic parallel loops: work is cut into fixed chunks that do not depend on the
//! thread count, each chunk is evaluated on its own (on the current rayon pool when
//! `parallel`, in order otherwise) and the chunk results are returned in chunk order, so that
//! reductions over them give the same bits for every thread count. Without `parallel` no rayon
//! call is made (a calculation with one thread spawns no threads).

use axtb::{Error, Result};
use faer::Par;
use rayon::prelude::*;
use std::ops::Range;

/// Basis size from which the dense linear algebra (matrix products, eigensolver) of a
/// calculation on more than one thread runs in parallel; below it the threads' start-up costs
/// more than they save (as for GFN2-xTB).
pub const DENSE_MIN_AO: usize = 200;

/// The parallelism of the dense linear algebra of `nao` basis functions on `threads` threads
/// of the current rayon pool ([`DENSE_MIN_AO`]); 0 threads is an error.
pub fn dense_par(threads: usize, nao: usize) -> Result<Par> {
    match threads {
        0 => Err(Error::Input("threads must be at least 1".into())),
        t if t > 1 && nao >= DENSE_MIN_AO => Ok(Par::rayon(t)),
        _ => Ok(Par::Seq),
    }
}

/// Number of chunks of a loop (fixed: results do not depend on the thread count).
pub const NCHUNK: usize = 64;

/// The fixed chunks of `0..n`: at most [`NCHUNK`], of equal size.
pub fn chunks(n: usize) -> Vec<Range<usize>> {
    let nc = n.clamp(1, NCHUNK);
    (0..nc).map(|c| c * n / nc..(c + 1) * n / nc).collect()
}

/// `f` on every chunk of `0..n` ([`chunks`]), results in chunk order.
pub fn map_chunks<T: Send>(n: usize, parallel: bool, f: impl Fn(Range<usize>) -> T + Sync + Send) -> Vec<T> {
    let c = chunks(n);
    if parallel {
        c.into_par_iter().map(f).collect()
    } else {
        c.into_iter().map(f).collect()
    }
}

/// `f` on every index of `0..n`, results in index order (for independent outputs, e.g. matrix
/// rows; no reduction).
pub fn map_items<T: Send>(n: usize, parallel: bool, f: impl Fn(usize) -> T + Sync + Send) -> Vec<T> {
    if parallel {
        (0..n).into_par_iter().map(f).collect()
    } else {
        (0..n).map(f).collect()
    }
}

/// Run `f(row, row data)` on the rows (of length `len`) of `out`, in parallel when `parallel`.
pub fn for_rows(out: &mut [f64], len: usize, parallel: bool, f: impl Fn(usize, &mut [f64]) + Sync + Send) {
    if len == 0 {
        return;
    }
    if parallel {
        out.par_chunks_mut(len).enumerate().for_each(|(i, r)| f(i, r));
    } else {
        out.chunks_mut(len).enumerate().for_each(|(i, r)| f(i, r));
    }
}

/// Tile edge of the passes that read a matrix transposed.
pub const TILE: usize = 32;

/// `out[i n + j] = f(i, j)` for an n x n row-major `out`, in TILE x TILE tiles (so that reads
/// of a transposed element stay in cache), tile rows in parallel when `parallel`.
pub fn fill_tiled(out: &mut [f64], n: usize, parallel: bool, f: impl Fn(usize, usize) -> f64 + Sync + Send) {
    for_rows(out, TILE * n, parallel, |t, rows| {
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

/// Sum of `f(k)` over `0..n` in fixed chunks, chunk sums added in order.
pub fn sum(n: usize, parallel: bool, f: impl Fn(usize) -> f64 + Sync + Send) -> f64 {
    map_chunks(n, parallel, |r| r.map(&f).sum::<f64>()).into_iter().sum()
}

/// `f(k, &mut out[k])` for every element, in chunks in parallel when `parallel`.
pub fn for_each_mut(out: &mut [f64], parallel: bool, f: impl Fn(usize, &mut f64) + Sync + Send) {
    let len = out.len().div_ceil(NCHUNK).max(1);
    for_rows(out, len, parallel, |c, part| {
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
