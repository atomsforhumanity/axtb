// SPDX-License-Identifier: LGPL-3.0-or-later
//! The SCF settings every SCF method shares.

/// Where the SCFs of a method start.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Restart {
    /// From the previous SCF's converged wavefunction (the reference programs' restart files).
    Previous,
    /// From the method's own start, at every SCF.
    Fresh,
}

/// The SCF settings every SCF method has, with one meaning in each; `None`: the method's
/// default. A method's own SCF settings live beside them in its options.
#[derive(Clone, Debug, PartialEq)]
pub struct ScfSettings {
    /// Multiplier of the method's convergence thresholds (> 0; default 1).
    pub accuracy: f64,
    /// Iterations of one SCF (at least 1); `None`: the method's (GFN2-xTB and g-xTB 2.0.1 250,
    /// g-xTB v1.1 1000).
    pub max_iter: Option<usize>,
    /// Electronic temperature (K, >= 0) of the Fermi smearing; `None`: the method's (GFN2-xTB
    /// 300, g-xTB 0).
    pub etemp: Option<f64>,
    /// Where SCFs start; `None`: the conventions' choice. A wavefunction given to the method
    /// starts the next SCF whatever this says.
    pub restart: Option<Restart>,
}

impl Default for ScfSettings {
    fn default() -> Self {
        Self { accuracy: 1.0, max_iter: None, etemp: None, restart: None }
    }
}

impl ScfSettings {
    /// Whether an SCF starts from the previous wavefunction, `default` (the conventions'
    /// choice) when [`restart`](Self::restart) is `None`.
    pub fn reuses(&self, default: Restart) -> bool {
        self.restart.unwrap_or(default) == Restart::Previous
    }
}
