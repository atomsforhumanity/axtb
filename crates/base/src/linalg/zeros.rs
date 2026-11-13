// SPDX-License-Identifier: LGPL-3.0-or-later
//! Zero matrices filled by the threads of the pool.

use faer::{Mat, Par};
use rayon::prelude::*;

/// Matrices below this many elements are zeroed on the calling thread (`Mat::zeros`).
const PAR_MIN: usize = 1 << 16;

/// Columns per task of the parallel fill.
const COLS: usize = 16;

/// An `nrows x ncols` zero matrix; with `par` parallel and at least 65,536 elements its
/// columns are written by the threads of the current rayon pool. The values are those of
/// `Mat::zeros`; only who writes them first changes: a large matrix written by one thread pays
/// all of its first-touch page faults on that thread (the GFN2-xTB AO matrices of 1200 basis
/// functions: 11 x 11.5 MB, 34 ms).
pub fn zeros(nrows: usize, ncols: usize, par: Par) -> Mat<f64> {
    if matches!(par, Par::Seq) || nrows * ncols < PAR_MIN {
        return Mat::zeros(nrows, ncols);
    }
    let mut m = Mat::<f64>::with_capacity(nrows, ncols);
    let stride = m.col_stride() as usize;
    // the columns as raw pointers: the memory is not initialized until every column is written
    let base = m.as_ptr_mut() as usize;
    (0..ncols.div_ceil(COLS)).into_par_iter().for_each(|c| {
        for j in c * COLS..((c + 1) * COLS).min(ncols) {
            let col = (base as *mut f64).wrapping_add(j * stride);
            for i in 0..nrows {
                // SAFETY: column j < ncols of a buffer of column stride >= nrows and at
                // least `ncols` columns; each column is written by exactly one task
                unsafe { col.add(i).write(0.0) };
            }
        }
    });
    // SAFETY: every element (i, j) with i < nrows, j < ncols is initialized above and the
    // capacity holds the dimensions
    unsafe { m.set_dims(nrows, ncols) };
    m
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parallel_zeros_are_zeros() {
        let pool = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        for (r, c) in [(300, 300), (517, 263), (3, 70000)] {
            let m = pool.install(|| zeros(r, c, Par::rayon(4)));
            assert_eq!((m.nrows(), m.ncols()), (r, c));
            assert!((0..c).all(|j| (0..r).all(|i| m[(i, j)].to_bits() == 0)));
        }
    }
}
