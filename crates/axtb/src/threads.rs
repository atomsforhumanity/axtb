//! Guidance on thread counts, from the benchmarks in `bench/` (README, "Threads").

/// Above this many threads, one energy/gradient calculation got slower, not faster, on
/// every CPU and linear-algebra library we measured: the dense eigensolver of the SCF
/// stops scaling (64-core AMD Zen 5 node and 8-core Intel laptop, molecules up to ~1000
/// atoms). axtb does not cap the thread count; it only warns.
pub const SCF_THREADS_ADVISED_MAX: usize = 8;

/// A warning for running one SCF (energy/gradient) calculation on `threads` threads, if
/// that is more than [`SCF_THREADS_ADVISED_MAX`]. Hessians, which run their displacements
/// in parallel, do use more threads well and need no warning.
pub fn scf_threads_warning(threads: usize) -> Option<String> {
    (threads > SCF_THREADS_ADVISED_MAX).then(|| {
        format!(
            "{threads} threads requested for one energy/gradient calculation. In axtb's benchmarks a single \
             calculation gets slower beyond about {SCF_THREADS_ADVISED_MAX} threads, because the SCF eigensolver \
             stops scaling. For many molecules, run them in parallel with 1 thread each; Hessians, which run \
             their displacements in parallel, do use more threads well. See the README section \"Threads\"."
        )
    })
}

#[cfg(test)]
mod tests {
    #[test]
    fn warns_only_above_the_advised_maximum() {
        assert!(super::scf_threads_warning(1).is_none());
        assert!(super::scf_threads_warning(super::SCF_THREADS_ADVISED_MAX).is_none());
        assert!(super::scf_threads_warning(super::SCF_THREADS_ADVISED_MAX + 1).is_some());
    }
}
