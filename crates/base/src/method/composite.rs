// SPDX-License-Identifier: LGPL-3.0-or-later
//! Methods made of methods: the reports of a composite.

use super::{HessianMethod, ScfReport};

/// A named value of a composite's description.
#[derive(Clone, Debug, PartialEq)]
pub enum Detail {
    Text(String),
    Name(&'static str),
    Number(f64),
    Flag(bool),
    Group(Vec<(&'static str, Detail)>),
}

/// What a composite reports with a single point: its description (what its builder says about
/// it), the energies of its parts and corrections, and how each part converged.
#[derive(Clone, Debug, PartialEq)]
pub struct CompositeReport {
    pub description: Vec<(&'static str, Detail)>,
    /// The parts' energies and the corrections (Eh), in the report order.
    pub components: Vec<(&'static str, f64)>,
    pub parts: Vec<PartReport>,
}

/// How one part of a composite converged.
#[derive(Clone, Debug, PartialEq)]
pub struct PartReport {
    /// The part's key in the reports (e.g. `gfn2_solvated`).
    pub key: &'static str,
    /// What the part is (e.g. `gfn2+alpb`).
    pub label: String,
    /// The part's conventions.
    pub conventions: &'static str,
    /// The part's energy (Eh).
    pub energy: f64,
    /// Its SCF (a single point's part); `None` for a Hessian's parts.
    pub scf: Option<ScfReport>,
    /// Electronic temperature (K) and iteration limit of the try that converged.
    pub etemp: f64,
    pub max_iter: usize,
    /// The retry that converged (`None`: the first try).
    pub retry: Option<String>,
    /// Wall time of the part, all tries (s).
    pub seconds: f64,
    /// A Hessian's part: how its Hessian was computed.
    pub hessian_method: Option<HessianMethod>,
}
