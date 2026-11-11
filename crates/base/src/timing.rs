//! Wall time per named phase (the `timings` of a calculation).

use std::time::Instant;

/// Wall time per phase (seconds), in the order the phases first ran.
pub type Timings = Vec<(&'static str, f64)>;

/// Accumulates the wall time since the previous lap under a phase name.
#[derive(Debug)]
pub struct Laps {
    last: Instant,
    /// The phases so far.
    pub laps: Timings,
}

impl Default for Laps {
    fn default() -> Self {
        Self::new()
    }
}

impl Laps {
    /// Start the clock.
    pub fn new() -> Self {
        Self { last: Instant::now(), laps: Vec::new() }
    }

    /// Add the time since the previous lap (or the start) to phase `name`.
    pub fn lap(&mut self, name: &'static str) {
        let now = Instant::now();
        let dt = (now - self.last).as_secs_f64();
        self.last = now;
        match self.laps.iter_mut().find(|(n, _)| *n == name) {
            Some(e) => e.1 += dt,
            None => self.laps.push((name, dt)),
        }
    }

    /// Restart the clock without recording.
    pub fn skip(&mut self) {
        self.last = Instant::now();
    }

    /// Total of all phases.
    pub fn total(&self) -> f64 {
        self.laps.iter().map(|x| x.1).sum()
    }

    /// The phases.
    pub fn finish(self) -> Timings {
        self.laps
    }
}
