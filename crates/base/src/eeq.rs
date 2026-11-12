// SPDX-License-Identifier: LGPL-3.0-or-later
//! Electronegativity equilibration (EEQ): the bordered linear system of the charge models,
//! written once (feature `linalg`).
//!
//! An EEQ model takes the atomic charges q that minimize the electrostatic energy of atoms with
//! Gaussian charge distributions,
//!
//! E(q) = Σ_A (−x_A q_A + ½ J_AA q_A²) + Σ_{A<B} J_AB q_A q_B,
//!
//! under linear charge constraints C q = Q (the total charge, or the charge of each fragment).
//! With one Lagrange multiplier per constraint, the stationary point is the solution of the
//! bordered system
//!
//! ```text
//! [ J  Cᵀ ] [ q ]   [ x ]
//! [ C  0  ] [ λ ] = [ Q ]
//! ```
//!
//! (A. Ghasemi, S. Goedecker et al., Phys. Rev. B 92, 045131 (2015)). The models that use it
//! differ in the diagonal J_AA (a hardness plus the self-interaction √(2/π)/σ_A of the
//! Gaussian), the pair kernel J_AB (the Coulomb interaction erf(R/√(σ_A² + σ_B²))/R of two
//! Gaussians, from geometric or topological distances), the right-hand side x (a
//! coordination-number dependent electronegativity) and the constraint rows:
//!
//! - the EEQ 2019 model of D4 (E. Caldeweyher et al., J. Chem. Phys. 150, 154122 (2019)), the
//!   GFN2-xTB start guess in `axtb`: total charge;
//! - GFN-FF's topology EEQ (xtb's `goedeckera`) and geometry EEQ (`goed_gfnff`) in
//!   `axtb-gfnff` (P. Pracht, S. Spicher, S. Grimme): one row per fragment;
//! - the bond-capacitor EEQ-BC of g-xTB in `axtb-gxtb` (its matrix is the capacitance-scaled
//!   kernel): total charge.
//!
//! Each model keeps its own expressions as closures ([`ChargeEquations::new`]), so that its
//! evaluation order (`erf(γ R)/R` or `erf(√(R² γ²))/R`, literal or computed √(2/π)) keeps its
//! bits; this module holds the layout, the constraint rows and the solves. The matrix is
//! factorized once by LU with partial pivoting ([`Lu`]) and kept: for the charges of several
//! right-hand sides (fragment charges), the adjoint solve of a gradient and the charge response
//! of a Hessian.

use crate::linalg::Lu;
use faer::{Mat, MatMut, MatRef, Par};

/// The constraint rows (and columns) n.. of a bordered EEQ matrix of n atoms.
#[derive(Clone, Copy, Debug)]
pub enum Constraints<'a> {
    /// One row of ones: the charges sum to the total charge.
    Total,
    /// One row per fragment: `fragment[A]` is the fragment of atom A, numbered from 1 (xtb's
    /// `fraglist`); row n + f − 1 sums the charges of fragment f = 1..=`count`. An atom of no
    /// fragment in that range has no constraint entry.
    Fragments { fragment: &'a [usize], count: usize },
}

impl Constraints<'_> {
    /// The number of constraint rows.
    pub fn len(&self) -> usize {
        match self {
            Constraints::Total => 1,
            Constraints::Fragments { count, .. } => *count,
        }
    }

    /// True without constraint rows (zero fragments).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// An assembled bordered EEQ matrix (see the module documentation), ready to be factorized.
pub struct ChargeEquations {
    n: usize,
    a: Mat<f64>,
}

impl ChargeEquations {
    /// The matrix of `n` atoms with the constraint rows of `constraints`: J_AA = `diag(A)`,
    /// J_AB = J_BA = `pair(A, B)` for B < A, ones in the constraint rows and columns, zeros in
    /// the constraint block. `pair` is called row by row, A = 0..n and B = 0..A in increasing
    /// order, so that a kernel can walk a sorted neighbor list of A along a row.
    pub fn new(
        n: usize,
        constraints: Constraints<'_>,
        mut diag: impl FnMut(usize) -> f64,
        mut pair: impl FnMut(usize, usize) -> f64,
    ) -> Self {
        let m = n + constraints.len();
        let mut a = Mat::<f64>::zeros(m, m);
        for i in 0..n {
            a[(i, i)] = diag(i);
            for j in 0..i {
                let v = pair(i, j);
                a[(i, j)] = v;
                a[(j, i)] = v;
            }
        }
        match constraints {
            Constraints::Total => {
                for i in 0..n {
                    a[(i, n)] = 1.0;
                    a[(n, i)] = 1.0;
                }
            }
            Constraints::Fragments { fragment, count } => {
                for (j, &f) in fragment[..n].iter().enumerate() {
                    if (1..=count).contains(&f) {
                        a[(n + f - 1, j)] = 1.0;
                        a[(j, n + f - 1)] = 1.0;
                    }
                }
            }
        }
        ChargeEquations { n, a }
    }

    /// The number of atoms n.
    pub fn atoms(&self) -> usize {
        self.n
    }

    /// The assembled (n + constraints)² matrix.
    pub fn matrix(&self) -> MatRef<'_, f64> {
        self.a.as_ref()
    }

    /// Factorize on the parallelism `par` ([`Lu::new`]), which the solves use too. A singular
    /// matrix is factorized as well; its solves give charges that are not finite.
    pub fn factorize(self, par: Par) -> ChargeSolver {
        let m = self.a.nrows();
        ChargeSolver { n: self.n, m, lu: Lu::new(self.a, par) }
    }

    /// [`ChargeEquations::factorize`], or `None` when the matrix is singular to working
    /// precision (a pivot is zero or not finite; [`Lu::try_new`]).
    pub fn try_factorize(self, par: Par) -> Option<ChargeSolver> {
        let m = self.a.nrows();
        Lu::try_new(self.a, par).map(|lu| ChargeSolver { n: self.n, m, lu })
    }
}

/// The factorized bordered EEQ matrix A, kept for solves.
pub struct ChargeSolver {
    n: usize,
    m: usize,
    lu: Lu,
}

impl ChargeSolver {
    /// The number of atoms n.
    pub fn atoms(&self) -> usize {
        self.n
    }

    /// The solution y = (q, λ) of A y = `rhs` (the n atom entries x, then the constraint
    /// values Q).
    pub fn solve(&self, rhs: &[f64]) -> Vec<f64> {
        self.lu.solve_vec(rhs)
    }

    /// The charges q of the right-hand side `rhs` (the first n entries of [`ChargeSolver::solve`]).
    pub fn charges(&self, rhs: &[f64]) -> Vec<f64> {
        let mut y = self.solve(rhs);
        y.truncate(self.n);
        y
    }

    /// The adjoint solution w of A w = (`dedq`, 0): for a function E(q(R)) of the charges,
    /// dE/dR = wᵀ (∂x/∂R − ∂A/∂R y) (A is symmetric), with ∂x/∂R the derivative of the
    /// right-hand side.
    pub fn adjoint(&self, dedq: &[f64]) -> Vec<f64> {
        let mut rhs = vec![0.0; self.m];
        rhs[..dedq.len()].copy_from_slice(dedq);
        self.solve(&rhs)
    }

    /// Solve A X = B in place for the columns of `b` (`b` becomes X): the charge response of a
    /// Hessian, X = A⁻¹ ∂(Ay − rhs)/∂R.
    pub fn solve_in_place(&self, b: MatMut<'_, f64>) {
        self.lu.solve(b);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Three atoms in two fragments: the charges of each fragment sum to its charge, and the
    /// solve agrees with the energy minimum under the constraints.
    #[test]
    fn fragment_constraints() {
        let x = [0.3, -0.2, 0.1];
        let j = [[1.0, 0.2, 0.1], [0.2, 0.9, 0.3], [0.1, 0.3, 1.1]];
        let frag = [1, 1, 2];
        let eq =
            ChargeEquations::new(3, Constraints::Fragments { fragment: &frag, count: 2 }, |i| j[i][i], |a, b| j[a][b]);
        assert_eq!(eq.matrix().nrows(), 5);
        assert_eq!((eq.matrix()[(3, 1)], eq.matrix()[(4, 1)], eq.matrix()[(2, 4)]), (1.0, 0.0, 1.0));
        let s = eq.try_factorize(Par::Seq).unwrap();
        let y = s.solve(&[x[0], x[1], x[2], 0.5, -1.0]);
        assert!((y[0] + y[1] - 0.5).abs() < 1e-14 && (y[2] + 1.0).abs() < 1e-14);
        // stationarity: J q − x + Cᵀ λ = 0
        for a in 0..3 {
            let jq: f64 = (0..3).map(|b| j[a][b] * y[b]).sum();
            let l = if frag[a] == 1 { y[3] } else { y[4] };
            assert!((jq - x[a] + l).abs() < 1e-14);
        }
        assert_eq!(s.charges(&[x[0], x[1], x[2], 0.5, -1.0]), y[..3].to_vec());
    }

    /// The adjoint solve gives dE/dx of E = gᵀ q for a linear function of the charges (A is
    /// symmetric), compared with differences of the solution.
    #[test]
    fn adjoint_is_the_derivative() {
        let j = [[1.2, 0.4], [0.4, 0.8]];
        let s = ChargeEquations::new(2, Constraints::Total, |i| j[i][i], |a, b| j[a][b]).factorize(Par::Seq);
        let g = [0.7, -0.3];
        let w = s.adjoint(&g);
        let e = |x0: f64| {
            let q = s.charges(&[x0, 0.25, 0.0]);
            g[0] * q[0] + g[1] * q[1]
        };
        let h = 1e-4;
        assert!(((e(h) - e(-h)) / (2.0 * h) - w[0]).abs() < 1e-10);
        assert!(ChargeEquations::new(1, Constraints::Total, |_| 0.0, |_, _| 0.0).try_factorize(Par::Seq).is_some());
        let singular = ChargeEquations::new(2, Constraints::Total, |_| 1.0, |_, _| 1.0);
        assert!(singular.try_factorize(Par::Seq).is_none());
    }
}
