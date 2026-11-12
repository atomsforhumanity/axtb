// SPDX-License-Identifier: LGPL-3.0-or-later
//! The one retry policy: a ladder of tries climbed until one succeeds.

use crate::{Error, Result};

/// One try of a calculation: what changes against the method's settings.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Rung {
    /// Forget the wavefunction before this try: start from the method's own start.
    pub fresh: bool,
    /// Electronic temperature (K) of this try; `None`: the method's setting.
    pub etemp: Option<f64>,
    /// Iteration limit of this try; `None`: the method's setting.
    pub max_iter: Option<usize>,
    /// How reports name this try (`None`: the first, unchanged one).
    pub label: Option<String>,
}

/// Which errors climb to the next rung; every other error ends the climb.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RetryOn {
    /// An SCF that did not converge ([`Error::ScfNotConverged`]).
    NotConverged,
    /// An SCF that failed: not converged or broken down ([`Error::Numerical`], e.g. a failed
    /// mixer or a lost Fermi level).
    Failed,
}

impl RetryOn {
    /// Whether `e` climbs.
    pub fn retries(self, e: &Error) -> bool {
        match self {
            RetryOn::NotConverged => matches!(e, Error::ScfNotConverged { .. }),
            RetryOn::Failed => matches!(e, Error::ScfNotConverged { .. } | Error::Numerical(_)),
        }
    }
}

/// Tries in order ([`RetryLadder::climb`]).
#[derive(Clone, Debug, PartialEq)]
pub struct RetryLadder {
    pub rungs: Vec<Rung>,
    pub on: RetryOn,
}

impl RetryLadder {
    /// One try, as the method stands.
    pub fn once() -> Self {
        Self { rungs: vec![Rung::default()], on: RetryOn::NotConverged }
    }

    /// The SCF as the method stands, and when `warm` (it started from a previous wavefunction)
    /// and did not converge, once more from the method's own start.
    pub fn warm_then_fresh(warm: bool) -> Self {
        let mut rungs = vec![Rung::default()];
        if warm {
            rungs.push(Rung { fresh: true, label: Some("fresh start".into()), ..Rung::default() });
        }
        Self { rungs, on: RetryOn::NotConverged }
    }

    /// Run `attempt` with each rung in order until one succeeds; an error the ladder does not
    /// retry ends the climb with it, and so does the last rung's error. Returns the result and
    /// the index of the rung that gave it.
    pub fn climb<T>(&self, mut attempt: impl FnMut(&Rung) -> Result<T>) -> Result<(T, usize)> {
        let mut last = None;
        for (i, rung) in self.rungs.iter().enumerate() {
            match attempt(rung) {
                Ok(v) => return Ok((v, i)),
                Err(e) if self.on.retries(&e) => last = Some(e),
                Err(e) => return Err(e),
            }
        }
        Err(last.unwrap_or_else(|| Error::Input("a retry ladder without rungs".into())))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn nc(i: usize) -> Error {
        Error::ScfNotConverged { iterations: i, energy_change: 0.0, residual: 0.0 }
    }

    #[test]
    fn climbs_until_one_succeeds_and_stops_on_other_errors() {
        let ladder = RetryLadder::warm_then_fresh(true);
        let mut tried = Vec::new();
        let r = ladder.climb(|rung| {
            tried.push(rung.fresh);
            if rung.fresh {
                Ok(7)
            } else {
                Err(nc(1))
            }
        });
        assert_eq!(r, Ok((7, 1)));
        assert_eq!(tried, [false, true]);
        // the last rung's error; a numerical failure ends a NotConverged ladder at once
        assert_eq!(ladder.climb(|r| Err::<(), _>(nc(r.fresh as usize))), Err(nc(1)));
        let mut n = 0;
        let e = ladder.climb(|_| {
            n += 1;
            Err::<(), _>(Error::Numerical("x".into()))
        });
        assert_eq!((e, n), (Err(Error::Numerical("x".into())), 1));
        let failed = RetryLadder { on: RetryOn::Failed, ..ladder };
        assert_eq!(failed.climb(|r| if r.fresh { Ok(()) } else { Err(Error::Numerical("x".into())) }), Ok(((), 1)));
        assert_eq!(RetryLadder::warm_then_fresh(false).rungs.len(), 1);
    }
}
