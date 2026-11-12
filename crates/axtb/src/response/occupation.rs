//! Response of the orbital occupations of a converged SCF to a perturbation: the divided
//! differences of the occupations between orbitals and the Fermi-level shifts that keep the
//! number of electrons of every spin.
//!
//! For orbitals with energies `ε_p`, occupations `f_p` and level derivatives
//! `f'_p = ∂f_p/∂ε_p` (Fermi smearing; zero at 0 K), a perturbation with first-order Fock
//! matrix `F^x` and overlap `S^x` (MO basis: `F̃ = Cᵀ F^x C`, `S̃ = Cᵀ S^x C`) changes the
//! density `P = C diag(f) Cᵀ` and the energy-weighted density `W = C diag(fε) Cᵀ` by
//!
//! ```text
//! δP̃_pq = F̃_pq D1_pq − S̃_pq D2_pq − δ_pq Σ_σ f'_σp δμ_σ
//! δW̃_pq = F̃_pq D2_pq − S̃_pq D3_pq − δ_pq ε_p Σ_σ f'_σp δμ_σ
//! ```
//!
//! with the divided differences of `f`, `fε` and `fε²` between the levels, written through D1,
//!
//! ```text
//! D1_pq = (f_p − f_q) / (ε_p − ε_q)
//! D2_pq = ½ (f_p + f_q) + ½ (ε_p + ε_q) D1_pq
//! D3_pq = ½ (ε_p² + ε_q²) D1_pq + ½ (f_p + f_q)(ε_p + ε_q)
//! ```
//!
//! and their limits `D1_pq = ½ (f'_p + f'_q)` for degenerate levels (and on the diagonal). The
//! Fermi level of each spin σ shifts by `δμ_σ = Σ_p f'_σp (F̃_pp − ε_p S̃_pp) / Σ_p f'_σp`, which
//! keeps its number of electrons. `W` is not `½ P F P` for fractional occupations: its response
//! follows from `W = C diag(fε) Cᵀ` as above.
//!
//! What a method chooses is passed in:
//! - the occupation vectors summed into one response: one (the total occupations of a closed
//!   shell, or one spin channel with its own orbitals) or two (the alpha and beta occupations of
//!   common orbitals, as in GFN2-xTB);
//! - the level derivatives `f'` (from the Fermi levels, or from the occupations);
//! - when two levels count as degenerate (GFN2-xTB: [`within_kt`]);
//! - the rows of the response: the orbitals that carry electrons ([`occupied_count`]; a
//!   symmetric response is then stored by these rows, see [`super::MoSpace`]) or all orbitals.

use faer::{Mat, MatRef};

/// Number of orbitals that carry electrons: one past the last orbital whose occupation, summed
/// over the vectors `f`, exceeds `threshold`.
pub fn occupied_count<T: AsRef<[f64]>>(f: &[T], threshold: f64) -> usize {
    let n = f.first().map_or(0, |fs| fs.as_ref().len());
    (0..n).rev().find(|&p| f.iter().map(|fs| fs.as_ref()[p]).sum::<f64>() > threshold).map_or(0, |p| p + 1)
}

/// Degenerate levels relative to the electronic temperature: `|ε_p − ε_q| ≤ factor · kT` (kT
/// floored at 1e-300 Eh) or equal levels (GFN2-xTB's response: factor 1e-5). The rule for
/// [`OccupationResponse::new`].
pub fn within_kt(factor: f64, kt: f64) -> impl Fn(f64, f64, f64) -> bool + Copy {
    let tol = factor * kt.max(1e-300);
    move |de, _, _| de.abs() <= tol || de == 0.0
}

/// Response of one perturbation in the MO basis: rows `[0, n_o)` of δP̃ and δW̃ with their
/// occupied-occupied blocks halved (the form [`super::MoSpace::to_ao`] takes).
pub struct MoResponse {
    pub dp: Mat<f64>,
    pub dw: Mat<f64>,
}

/// Occupations of a converged SCF, their level derivatives and their divided differences
/// between orbitals: the occupation part of a coupled-perturbed SCF. Read-only: D1, D2 and D3
/// follow from the energies, occupations and level derivatives they were built from.
#[derive(Clone, Debug)]
pub struct OccupationResponse {
    eps: Vec<f64>,
    f: Vec<Vec<f64>>,
    df: Vec<Vec<f64>>,
    z: Vec<f64>,
    n_o: usize,
    d1: Mat<f64>,
    d2: Mat<f64>,
    d3: Mat<f64>,
}

impl OccupationResponse {
    /// Orbital energies (ascending).
    #[inline]
    pub fn eps(&self) -> &[f64] {
        &self.eps
    }

    /// Occupations, one vector per spin summed into the response (one or two).
    #[inline]
    pub fn f(&self) -> &[Vec<f64>] {
        &self.f
    }

    /// Level derivatives `f'_p = ∂f_p/∂ε_p` of every vector (zero without smearing).
    #[inline]
    pub fn df(&self) -> &[Vec<f64>] {
        &self.df
    }

    /// `Σ_p f'_p` of every vector, over all orbitals (zero without smearing).
    #[inline]
    pub fn z(&self) -> &[f64] {
        &self.z
    }

    /// The rows of the response are the orbitals `[0, n_o)`.
    #[inline]
    pub fn n_o(&self) -> usize {
        self.n_o
    }

    /// D1 for the rows `[0, n_o)` (n_o x n).
    #[inline]
    pub fn d1(&self) -> &Mat<f64> {
        &self.d1
    }

    /// D2 for the rows `[0, n_o)` (n_o x n).
    #[inline]
    pub fn d2(&self) -> &Mat<f64> {
        &self.d2
    }

    /// D3 for the rows `[0, n_o)` (n_o x n).
    #[inline]
    pub fn d3(&self) -> &Mat<f64> {
        &self.d3
    }

    /// The response of orbitals with energies `eps`, occupations `f` and level derivatives `df`
    /// (one vector per spin summed into the response, each as long as `eps`) for the rows
    /// `[0, n_o)`. `degenerate(ε_p − ε_q, f_p, f_q)` tells, for every vector, whether D1 takes
    /// the derivative limit `½ (f'_p + f'_q)` instead of the difference quotient.
    pub fn new(
        eps: &[f64],
        f: Vec<Vec<f64>>,
        df: Vec<Vec<f64>>,
        n_o: usize,
        degenerate: impl Fn(f64, f64, f64) -> bool,
    ) -> Self {
        let n = eps.len();
        assert!(n_o <= n && !f.is_empty() && f.len() <= 2 && df.len() == f.len());
        let z = df.iter().map(|d| d.iter().sum()).collect();
        let mut d1 = Mat::<f64>::zeros(n_o, n);
        let mut d2 = Mat::<f64>::zeros(n_o, n);
        let mut d3 = Mat::<f64>::zeros(n_o, n);
        for q in 0..n {
            for p in 0..n_o {
                let (a, b, e) = divided(eps, &f, &df, p, q, &degenerate);
                d1[(p, q)] = a;
                d2[(p, q)] = b;
                d3[(p, q)] = e;
            }
        }
        Self { eps: eps.to_vec(), f, df, z, n_o, d1, d2, d3 }
    }

    /// Number of orbitals full in every vector, at most `n_o`: the first orbital with an
    /// occupation below `fmax - threshold` in some vector.
    pub fn n_full(&self, fmax: f64, threshold: f64) -> usize {
        let n = self.eps.len();
        let n_f = (0..n).find(|&p| self.f.iter().any(|fs| fs[p] < fmax - threshold)).unwrap_or(n);
        n_f.min(self.n_o)
    }

    /// Fermi-level shift of vector `s` for the level changes `x(p)` of the rows:
    /// `Σ_{p<n_o} f'_p x(p) / Σ_p f'_p` (zero without smearing).
    #[inline]
    pub fn fermi_shift(&self, s: usize, x: impl Fn(usize) -> f64) -> f64 {
        let z = self.z[s];
        if z == 0.0 {
            return 0.0;
        }
        let df = &self.df[s];
        let mut acc = 0.0;
        for p in 0..self.n_o {
            let w = df[p];
            if w != 0.0 {
                acc += w * x(p);
            }
        }
        acc / z
    }

    /// δP̃ and δW̃ (rows `[0, n_o)`, occupied-occupied blocks halved) for a perturbation with
    /// first-order Fock matrix `f` and overlap `s` (`None`: the overlap does not change) in the
    /// MO basis, rows `[0, n_o)`, with the Fermi-level shifts of every vector.
    pub fn density_response(&self, f: MatRef<'_, f64>, s: Option<MatRef<'_, f64>>) -> MoResponse {
        let n = self.eps.len();
        let n_o = self.n_o;
        let ns = self.f.len();
        let emo = &self.eps;
        let mut dmu = [0.0; 2];
        for (sp, d) in dmu.iter_mut().enumerate().take(ns) {
            *d = self.fermi_shift(sp, |p| f[(p, p)] - emo[p] * s.map_or(0.0, |s| s[(p, p)]));
        }
        let mut dp = Mat::<f64>::zeros(n_o, n);
        let mut dw = Mat::<f64>::zeros(n_o, n);
        for q in 0..n {
            for p in 0..n_o {
                let a = f[(p, q)];
                let sv = s.map_or(0.0, |s| s[(p, q)]);
                let (d1, d2, d3) = (self.d1[(p, q)], self.d2[(p, q)], self.d3[(p, q)]);
                let mut x = a * d1 - sv * d2;
                let mut y = a * d2 - sv * d3;
                if p == q {
                    let mut g = self.df[0][p] * dmu[0];
                    for sp in 1..ns {
                        g += self.df[sp][p] * dmu[sp];
                    }
                    x -= g;
                    y -= emo[p] * g;
                }
                if q < n_o {
                    x *= 0.5;
                    y *= 0.5;
                }
                dp[(p, q)] = x;
                dw[(p, q)] = y;
            }
        }
        MoResponse { dp, dw }
    }
}

/// Divided differences `(D1, D2, D3)` between orbitals p and q, summed over the occupation
/// vectors, with the derivative limit of D1 for the pairs `degenerate` accepts.
#[inline]
fn divided(
    eps: &[f64],
    f: &[Vec<f64>],
    df: &[Vec<f64>],
    p: usize,
    q: usize,
    degenerate: &impl Fn(f64, f64, f64) -> bool,
) -> (f64, f64, f64) {
    let (ep, eq) = (eps[p], eps[q]);
    let de = ep - eq;
    let mut d1 = 0.0;
    let mut fsum = 0.0;
    for (fs, dfs) in f.iter().zip(df) {
        let (fp, fq) = (fs[p], fs[q]);
        fsum += fp + fq;
        d1 += if degenerate(de, fp, fq) { 0.5 * (dfs[p] + dfs[q]) } else { (fp - fq) / de };
    }
    let d2 = 0.5 * fsum + 0.5 * (ep + eq) * d1;
    let d3 = 0.5 * (ep * ep + eq * eq) * d1 + 0.5 * fsum * (ep + eq);
    (d1, d2, d3)
}
