# Vendored, patched crates

| crate | version | upstream | license | patch |
|---|---|---|---|---|
| gemm-common | 0.18.2 | https://github.com/sarah-quinones/gemm | MIT | `src/cache.rs`: read only cpu0's cache files |
| private-gemm-x86 | 0.1.22 | https://github.com/sarah-quinones/gemm-x64-v2 | MIT | `src/cache.rs`: read only cpu0's cache files |

Both crates (dependencies of faer 0.22) detect cache sizes by opening every file under
`/sys/devices/system/cpu/cpu*/cache/index*/` the first time a matrix product runs: about 5,800
opens per process on a 128-thread machine. When many processes start together the kernel
serializes these, and each process spends ~0.4 s in start-up (measured on a 64-core / 128-thread
AMD node: 1024 axtb CLI runs on 128 workers took 5.87 s; with this patch 0.52 s; see
bench/cluster/cli_throughput.py and bench/results/cluster-9985wx/cli_throughput.json).

The upstream loop overwrites its result with each CPU it visits, so the answer is the last CPU
in directory order; on machines whose cores share one cache layout, reading cpu0 gives the
same answer. Remove a crate here (and its `[patch.crates-io]` entry in the root Cargo.toml)
once upstream releases a fix.

Reported upstream as [sarah-quinones/gemm#53](https://github.com/sarah-quinones/gemm/issues/53)
(gemm-common) and [sarah-quinones/gemm-x64-v2#8](https://github.com/sarah-quinones/gemm-x64-v2/issues/8)
(private-gemm-x86); both are open, and upstream's `cache.rs` is unpatched.

The patch applies only to builds of this workspace: the CLI, the Python wheels and anything else
built inside this repository, and builds from the Python sdist, which carries this directory
(`crates/axtb-py/third_party` is a symbolic link to it, for maturin; see `pyproject.toml`). Cargo ignores the `[patch]` sections of dependencies, so a project
that depends on the axtb crates from crates.io or from this repository's URL builds the
unpatched gemm crates and pays the start-up cost on many-CPU machines, unless it adds the same
`[patch.crates-io]` entries to its own workspace.
