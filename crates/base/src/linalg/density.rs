//! Density matrices from orbital coefficients: P = C diag(w) Cᵀ over the columns of C with a
//! non-negligible weight (occupations, or occupations times orbital energies for the
//! energy-weighted density), as a full matrix product or as its lower triangle mirrored
//! (the syrk of grimme-lab/xtb#1466).

use faer::{Mat, MatRef, Par};

thread_local! {
    /// The two n x k matrices of [`weighted_columns`], kept on the thread between calls (their
    /// memory lives as long as the thread). Made afresh in every SCF iteration, their pages
    /// were faulted in every time: glibc gives each block of this size back to the system when
    /// it is freed, unless a larger block was freed before (the n x n matrix that the run-time
    /// LAPACK path allocated per call until 83b5bd8 was one, by accident). GFN2 (H2O)200, MKL:
    /// `scf: density` 0.116 -> 0.084 s per run on 8 threads, 0.038 -> 0.030 s for (H2O)100 on 1.
    static WEIGHTED_COLUMNS: std::cell::RefCell<Option<(Mat<f64>, Mat<f64>)>> = const { std::cell::RefCell::new(None) };
}

/// The columns of `c` with a weight above 1e-300 times their weights (`cw`) and as they are
/// (`cc`), n x m each, passed to `f`: in the thread's kept matrices, or in new ones while those
/// are in use (a task of the pool run by this thread while it waits inside `f`). The kept
/// matrices are made anew when n changes, so their layout is that of new ones.
fn weighted_columns<T>(c: MatRef<'_, f64>, w: &[f64], f: impl FnOnce(MatRef<'_, f64>, MatRef<'_, f64>) -> T) -> T {
    let n = c.nrows();
    let cols: Vec<usize> = (0..w.len()).filter(|&i| w[i].abs() > 1e-300).collect();
    let m = cols.len();
    let fill = |cw: &mut Mat<f64>, cc: &mut Mat<f64>| {
        for (k, &i) in cols.iter().enumerate() {
            for r in 0..n {
                cc[(r, k)] = c[(r, i)];
                cw[(r, k)] = c[(r, i)] * w[i];
            }
        }
    };
    WEIGHTED_COLUMNS.with(|kept| match kept.try_borrow_mut() {
        Ok(mut kept) => {
            let (cw, cc) = kept.get_or_insert_with(|| (Mat::new(), Mat::new()));
            if cw.nrows() == n {
                cw.resize_with(n, m, |_, _| 0.0);
                cc.resize_with(n, m, |_, _| 0.0);
            } else {
                (*cw, *cc) = (Mat::zeros(n, m), Mat::zeros(n, m));
            }
            fill(cw, cc);
            f(cw.as_ref(), cc.as_ref())
        }
        Err(_) => {
            let (mut cw, mut cc) = (Mat::zeros(n, m), Mat::zeros(n, m));
            fill(&mut cw, &mut cc);
            f(cw.as_ref(), cc.as_ref())
        }
    })
}

/// `P = C diag(w) Cᵀ` for the columns of `c` with non-negligible weight.
pub fn weighted_outer(c: MatRef<'_, f64>, w: &[f64], out: &mut Mat<f64>, par: Par) {
    weighted_columns(c, w, |cw, cc| {
        faer::linalg::matmul::matmul(out.as_mut(), faer::Accum::Replace, cw, cc.transpose(), 1.0, par)
    });
}

/// [`weighted_outer`] computing the lower triangle alone (half the flops, the syrk of
/// grimme-lab/xtb#1466) and mirroring it: symmetric to the last bit, its elements rounded as a
/// triangular product rounds them (not as `weighted_outer`'s).
pub fn weighted_outer_sym(c: MatRef<'_, f64>, w: &[f64], out: &mut Mat<f64>, par: Par) {
    use faer::linalg::matmul::triangular::{matmul, BlockStructure};
    let n = c.nrows();
    weighted_columns(c, w, |cw, cc| {
        let rect = BlockStructure::Rectangular;
        matmul(
            out.as_mut(),
            BlockStructure::TriangularLower,
            faer::Accum::Replace,
            cw,
            rect,
            cc.transpose(),
            rect,
            1.0,
            par,
        )
    });
    for j in 1..n {
        for i in 0..j {
            out[(i, j)] = out[(j, i)];
        }
    }
}
