// SPDX-License-Identifier: LGPL-3.0-or-later
//! Threads, written once for every calculator: the calculator's thread pool, the rule for
//! when dense linear algebra runs in parallel, and the deterministic parallel loops.
//!
//! - [`Threads`] is the pool of a calculator: none for one thread (nothing is ever spawned
//!   and faer is only ever given `Par::Seq`), otherwise one rayon pool of exactly the
//!   requested size, shared by reference counting (the parts of a composite method run in one
//!   pool). [`Threads::install`] runs a calculation inside it: outside, rayon's parallel loops
//!   would run on its global pool, i.e. on every core instead of the threads asked for.
//! - `par` and `dense_par` give the faer parallelism of a thread count (feature `linalg`);
//!   dense matrix products and the eigensolver run in parallel only from [`DENSE_MIN_AO`]
//!   basis functions on, below which the threads' start-up costs more than it saves.
//! - The loops: [`map`] (one result per index), [`map_chunks`] (one result per fixed chunk of
//!   [`chunks`]) and [`for_slices`] / [`map_slices`] (one task per fixed-length slice of a
//!   buffer, e.g. per matrix row or column) run sequentially in order, or with `parallel` on
//!   the current rayon pool, and return their results in index order. The chunks depend only
//!   on the length of the loop, never on the thread count, so a reduction over the results in
//!   their order gives the same bits for every thread count, as long as each task runs
//!   sequentially. Without `parallel` no rayon call is made.
//!
//! Sources: the thread pools and pool wrappers of the calculators of `axtb` (GFN2-xTB)
//! and `axtb-gfnff`, the fixed chunks of the GFN2-xTB Hamiltonian gradient (at most 64 chunks
//! with bounds `c n / nchunk`), GFN-FF's chunked pair loops and the ordered parallel maps of
//! `axtb` and `axtb_base::fd`. `axtb-gxtb` (g-xTB) calls these routines too, and adds its own
//! tiled helpers on top of them.

use crate::{Error, Result};
use rayon::prelude::*;
use std::ops::Range;
use std::sync::Arc;

/// The thread pool of a calculator: none for one thread, otherwise a rayon pool of exactly
/// the requested size. Cloning shares the pool.
#[derive(Clone, Debug, Default)]
pub struct Threads {
    pool: Option<Arc<rayon::ThreadPool>>,
}

impl Threads {
    /// A pool of `threads` threads; one thread is no pool (nothing is spawned). Zero threads,
    /// or a pool the system cannot create, is [`Error::Input`].
    pub fn new(threads: usize) -> Result<Self> {
        check(threads)?;
        if threads == 1 {
            return Ok(Self::sequential());
        }
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .map_err(|e| Error::Input(format!("cannot create thread pool: {e}")))?;
        Ok(Self { pool: Some(Arc::new(pool)) })
    }

    /// One thread: no pool, every loop in order on the calling thread.
    pub const fn sequential() -> Self {
        Self { pool: None }
    }

    /// The number of threads (1 without a pool).
    pub fn count(&self) -> usize {
        self.pool.as_ref().map_or(1, |p| p.current_num_threads())
    }

    /// Whether there is a pool of more than one thread, i.e. whether the loops run in
    /// parallel.
    pub fn parallel(&self) -> bool {
        self.pool.is_some()
    }

    /// The rayon pool (`None` for one thread).
    pub fn pool(&self) -> Option<&rayon::ThreadPool> {
        self.pool.as_deref()
    }

    /// `f` inside the pool (directly on the calling thread without one), so that the parallel
    /// loops it runs use the pool's threads. From inside the pool, `f` runs directly.
    pub fn install<T: Send>(&self, f: impl FnOnce() -> T + Send) -> T {
        match &self.pool {
            Some(pool) => pool.install(f),
            None => f(),
        }
    }

    /// [`map`] inside the pool, in parallel when there is one.
    pub fn map<T: Send>(&self, n: usize, f: impl Fn(usize) -> T + Sync + Send) -> Vec<T> {
        let parallel = self.parallel();
        self.install(|| map(n, parallel, f))
    }

    /// The faer parallelism of the pool ([`par`]).
    #[cfg(feature = "linalg")]
    pub fn par(&self) -> faer::Par {
        par(self.count())
    }

    /// The faer parallelism of dense linear algebra of `nao` basis functions on the pool
    /// ([`dense_par`]).
    #[cfg(feature = "linalg")]
    pub fn dense_par(&self, nao: usize) -> faer::Par {
        dense_par(self.count(), nao)
    }
}

/// [`Error::Input`] for zero threads.
pub fn check(threads: usize) -> Result<()> {
    if threads == 0 {
        return Err(Error::Input("threads must be at least 1".into()));
    }
    Ok(())
}

/// Basis size from which the dense linear algebra (matrix products, eigensolver) of a
/// calculation on more than one thread runs in parallel (`dense_par`); below it the threads'
/// start-up costs more than they save.
pub const DENSE_MIN_AO: usize = 200;

/// faer's parallelism for `threads` threads of the current rayon pool: sequential for one.
#[cfg(feature = "linalg")]
pub fn par(threads: usize) -> faer::Par {
    if threads > 1 {
        faer::Par::rayon(threads)
    } else {
        faer::Par::Seq
    }
}

/// faer's parallelism for the dense linear algebra of `nao` basis functions on `threads`
/// threads of the current rayon pool: parallel for more than one thread from [`DENSE_MIN_AO`]
/// basis functions on, sequential otherwise.
#[cfg(feature = "linalg")]
pub fn dense_par(threads: usize, nao: usize) -> faer::Par {
    if threads > 1 && nao >= DENSE_MIN_AO {
        faer::Par::rayon(threads)
    } else {
        faer::Par::Seq
    }
}

/// `f` on every index of `0..n`, results in index order: on the current rayon pool when
/// `parallel`, in order on the calling thread otherwise. Each call runs on one thread, so if
/// `f` runs sequentially the results are the same for every thread count.
pub fn map<T: Send>(n: usize, parallel: bool, f: impl Fn(usize) -> T + Sync + Send) -> Vec<T> {
    if parallel {
        (0..n).into_par_iter().map(f).collect()
    } else {
        (0..n).map(f).collect()
    }
}

/// [`map`] with tasks of at least `min_len` indices on the current pool when `parallel`
/// (rayon's `with_min_len`: fewer, larger tasks for cheap `f`; it changes only the scheduling,
/// not the results).
pub fn map_with_min_len<T: Send>(
    n: usize,
    parallel: bool,
    min_len: usize,
    f: impl Fn(usize) -> T + Sync + Send,
) -> Vec<T> {
    if parallel {
        (0..n).into_par_iter().with_min_len(min_len).map(f).collect()
    } else {
        (0..n).map(f).collect()
    }
}

/// Maximum number of chunks of [`chunks`].
pub const NCHUNK: usize = 64;

/// The fixed chunks of `0..n`: `min(n, NCHUNK)` chunks (one for n = 0), chunk c being
/// `c n / nchunk .. (c + 1) n / nchunk`. They depend only on `n`, never on the thread count.
pub fn chunks(n: usize) -> Vec<Range<usize>> {
    let nchunk = n.clamp(1, NCHUNK);
    (0..nchunk).map(|c| c * n / nchunk..(c + 1) * n / nchunk).collect()
}

/// `f` on every chunk of `0..n` ([`chunks`]), results in chunk order (on the current rayon
/// pool when `parallel`). Summing the results in order gives the same bits for every thread
/// count.
pub fn map_chunks<T: Send>(n: usize, parallel: bool, f: impl Fn(Range<usize>) -> T + Sync + Send) -> Vec<T> {
    let c = chunks(n);
    if parallel {
        c.into_par_iter().map(f).collect()
    } else {
        c.into_iter().map(f).collect()
    }
}

/// `f(k, slice k)` for the consecutive slices of length `len` of `out` (the last one shorter
/// if `len` does not divide its length; nothing for `len = 0`), on the current rayon pool when
/// `parallel`. For matrices filled row by row (or column by column) in place.
pub fn for_slices<T: Send>(out: &mut [T], len: usize, parallel: bool, f: impl Fn(usize, &mut [T]) + Sync + Send) {
    if len == 0 {
        return;
    }
    if parallel {
        out.par_chunks_mut(len).enumerate().for_each(|(k, s)| f(k, s));
    } else {
        out.chunks_mut(len).enumerate().for_each(|(k, s)| f(k, s));
    }
}

/// [`for_slices`] with a result per slice, in slice order.
pub fn map_slices<T: Send, R: Send>(
    out: &mut [T],
    len: usize,
    parallel: bool,
    f: impl Fn(usize, &mut [T]) -> R + Sync + Send,
) -> Vec<R> {
    if len == 0 {
        return Vec::new();
    }
    if parallel {
        out.par_chunks_mut(len).enumerate().map(|(k, s)| f(k, s)).collect()
    } else {
        out.chunks_mut(len).enumerate().map(|(k, s)| f(k, s)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunks_cover_the_range_in_order() {
        assert_eq!(chunks(0), vec![Range { start: 0, end: 0 }]);
        assert_eq!(chunks(3), [0..1, 1..2, 2..3]);
        for n in [1, 63, 64, 65, 1000] {
            let c = chunks(n);
            assert_eq!(c.len(), n.min(NCHUNK));
            assert_eq!(c[0].start, 0);
            assert_eq!(c.last().unwrap().end, n);
            assert!(c.windows(2).all(|w| w[0].end == w[1].start));
        }
    }

    #[test]
    fn one_thread_is_no_pool() {
        let t = Threads::new(1).unwrap();
        assert!(!t.parallel() && t.pool().is_none() && t.count() == 1);
        assert!(matches!(Threads::new(0), Err(Error::Input(_))));
        assert_eq!(t.install(|| 7), 7);
    }

    #[test]
    fn loops_give_the_same_results_on_every_thread_count() {
        let f = |k: usize| (k as f64).sqrt() * 1e-3 + 1.0 / (1.0 + k as f64);
        let sum = |parallel| map_chunks(1001, parallel, |r| r.map(f).sum::<f64>()).into_iter().sum::<f64>();
        let reference = sum(false);
        for n in [2, 3, 5] {
            let t = Threads::new(n).unwrap();
            assert_eq!(t.count(), n);
            assert_eq!(t.install(|| sum(true)).to_bits(), reference.to_bits(), "{n} threads");
            assert_eq!(t.map(100, f), map(100, false, f));
            assert_eq!(t.install(|| map_with_min_len(100, true, 16, f)), map(100, false, f));
            let fill = |parallel| {
                let mut out = vec![0.0; 99];
                let firsts = map_slices(&mut out, 10, parallel, |k, s| {
                    s.iter_mut().enumerate().for_each(|(i, x)| *x = f(10 * k + i));
                    s[0]
                });
                let mut again = vec![0.0; 99];
                for_slices(&mut again, 10, parallel, |k, s| {
                    s.iter_mut().enumerate().for_each(|(i, x)| *x = f(10 * k + i))
                });
                assert_eq!(out, again);
                (out, firsts)
            };
            assert_eq!(t.install(|| fill(true)), fill(false));
        }
    }
}
