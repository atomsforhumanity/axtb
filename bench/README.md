# axtb benchmarks

| Path | What |
|---|---|
| `run_benchmarks.py` | axtb vs xtb (`--xtb`, default xtb 6.8.0 from `$AXTB_XTB680`, which the recorded results use; axtb in the conventions of its version) vs tblite 0.8.0 (GFN2-xTB): accuracy, solvation, timings, Hessians, throughput |
| `phases.py` | where axtb's time goes: per-phase times (`r["timings"]`) and SCF iterations for the timing molecules |
| `linalg/` | standalone crate comparing dense eigensolver backends (faer 0.22/0.24, OpenBLAS, MKL, AMD AOCL) on real GFN2 matrices |
| `cluster/` | image, task scripts, the job settings of the recorded runs and helpers to run all of it on whole cluster nodes |
| `results/results.json`, `results/phases.json`, `results/cli_throughput.json`, `linalg/results/` | laptop: Intel Core i9-11900H, 8 cores / 16 threads |
| `results/cluster-9985wx/` | cluster node: AMD Ryzen Threadripper PRO 9985WX (Zen 5), 64 cores / 128 threads (`machine.json`) |
| `final_timings.py`, `results/final-laptop/`, `results/final-9985wx/` | the final timings of axtb 0.1.0: GFN2-xTB vs xtb 6.8.0, g-xTB 2.0.1 and v1.1 vs their programs, energy + gradient on 1, 2, 4 and 8 threads with faer, MKL and (node) AOCL, Hessians on 1 and 8 threads, on the laptop and a cluster node, with attributions, spot checks and thread-determinism checks (see [Final timings](#final-timings-axtb-vs-xtb-680-and-the-g-xtb-programs)) |
| `attribution.py`, `attribution_table.py`, `spot_check.sh`, `spot_check_inputs.py` | the programs' own time breakdown next to axtb's phases; hand-run spot checks of `final_timings.py` |

Timings mean nothing on a busy machine. Record `uptime` before and after every timing run and
start only when the 1-minute load is below ~1.5 (laptop) or the node is reserved (cluster).

## On a laptop or workstation

Prerequisites: Rust 1.99 or newer, Python 3.12 with
`numpy tblite==0.8.0 dftd4==4.3.1 maturin`, and the reference programs in the variables of
`tests/programs.py` (`AXTB_XTB680`, `AXTB_GXTB2`, `AXTB_XTB671`, `AXTB_GXTB11_DIR`; the README's
"Checks" section says where to get them). Scripts that compare with one xtb take `--xtb` and
compare axtb in the conventions of its version (`xtb` for 6.8.0, `xtb-6.7.1` for 6.7.1).

```bash
cargo build --release -p axtb-cli                       # target/release/axtb (the CLI timings)
maturin build --release --compatibility linux -m crates/axtb-py/Cargo.toml -o dist && pip install dist/axtb-*.whl   # import axtb
python bench/run_benchmarks.py --quick                  # smoke test (~1 min)
python bench/run_benchmarks.py --out bench/results/results.json   # full suite, ~25 min
python bench/phases.py --threads 1,8 --out bench/results/phases.json
python bench/cluster/cli_throughput.py --root . --specs 64x8,128x16 --bin 3499913=target/release/axtb --out bench/results/cli_throughput.json
```

`run_benchmarks.py` options: `--threads 1,2,4,8` (timings), `--hess-threads 1,8` (Hessians),
`--throughput 64x8` (JOBSxWORKERS, several comma-separated allowed).

Linear algebra (`bench/linalg`, not part of the workspace):

```bash
cd bench/linalg
cargo build --release                                              # faer 0.22/0.24 + dlopen'ed LAPACKs
cargo +stable build --release --features faer-avx512 --target-dir target-v4   # faer 0.24 with AVX-512
# matrices: GFN2 H0 and S of 11 taxol copies (n = 3289); laptop and cluster use the same two files
(cd dump && cargo run --release -- ../../../tests/structures/taxol.xyz 11 /path/to/mats)
# libraries, all without root:
pip install --target /path/to/mkl mkl==2026.2.0                   # Intel MKL: /path/to/mkl/lib/libmkl_rt.so.3
pip install --target /path/to/ob scipy-openblas32==0.3.34.481.3   # OpenBLAS (OB_PREFIX=scipy_), as recorded
./build_aocl.sh /path/to/aocl-5.3                                 # AMD AOCL 5.3 from GitHub sources, ~10 min
# rounds: every backend x thread count, 3 interleaved rounds (laptop default: 1/4/8 threads)
AOCL=/path/to/aocl-5.3 BACKENDS="faer22 faer24 faer24v4 openblas mkl aocl" \
  ./run.sh /path/to/mats /path/to/mkl/lib target-v4/release/linalg-bench 3
# throughput: N concurrent single-thread processes, one per physical core
AOCL=/path/to/aocl-5.3 BACKENDS="faer22 openblas mkl aocl" ./throughput.sh /path/to/mats /path/to/mkl/lib 8
python3 summarize.py results/raw.tsv > results/summary_tables.txt
python3 summarize.py results/throughput.tsv evd,gen
```

`run.sh` and `throughput.sh` take `THREADS`, `BACKENDS`, `OB` (+ `OB_PREFIX=scipy_` for the
scipy-openblas32 wheel), `AOCL`, `BV4`, `B`, `PIN`, `OUT` and `ENV` from the environment; the
defaults are the laptop run, except `OB`, which defaults to Debian/Ubuntu's OpenMP build
(`libopenblas0-openmp`). The harness itself takes any OpenBLAS, but axtb's `AXTB_LAPACK` accepts
only OpenMP and sequential builds and refuses pthreads builds, whose thread count is
process-wide; the recorded rows used pthreads builds (the system's `libopenblas0-pthread` on the
laptop, the scipy-openblas32 wheel on the node). `openblas_get_parallel()` tells them apart (0
sequential, 1 pthreads, 2 OpenMP), e.g. `python3 -c "import ctypes;
print(ctypes.CDLL('/path/to/libopenblas.so.0').openblas_get_parallel())"` (the scipy-openblas32
wheel names it `scipy_openblas_get_parallel`). Each process sets its library's thread count
through the library's API (no `OMP_NUM_THREADS`). With `PIN=1` (default) processes are pinned
with `taskset`: 1 thread on CPU 2, t threads on CPUs 0..t-1 (one per physical core when SMT
siblings are numbered i and i+ncores; check
`/sys/devices/system/cpu/cpu0/topology/thread_siblings_list`). Do not pin faer above 8 threads
on the 64-core nodes (`PIN=0`): its rayon workers pinned to exactly t CPUs run up to 110x slower
there (medians at 32 and 64 threads, `results/cluster-9985wx/linalg/`). AOCL is loaded as
`lapack:<aocl>/lib/libflame.so --blas <aocl>/lib/libblis-mt.so`; `build_aocl.sh` builds it from
AMD's GitHub sources (BSD-3) rather than AMD's prebuilt tarball, whose evaluation EULA has a
confidentiality clause and limits use to AMD hardware.

The recorded laptop rows (`linalg/results/`, axtb 0.1.0 release run, 2026-11-07 14:48-15:49
UTC) ran all six backends in one session, started at a 1-minute load below 1.5 with nothing
else running: OpenBLAS 0.3.32 (the system's pthreads build), MKL 2026.2 (the PyPI wheel),
AOCL 5.3 built with `build_aocl.sh`, faer 0.22/0.24 with Rust 1.86 and the AVX-512 build
(`faer24v4`) with Rust 1.99.1 (`env.txt` records only the first).

## On whole cluster nodes

The recorded whole-node numbers (`results/cluster-9985wx/`) ran on whole nodes (AMD
Threadripper PRO 9985WX, 64 cores / 128 threads) of a Kubernetes cluster, one task per node in a
pod of 126 CPUs, so that no other pod shared the node. They are from the release run of 0.1.0,
axtb 3499913, 2026-11-07 16:20-17:50 UTC on two nodes at once (the linalg tasks on one, the
suite and then phases and the start-up probe on the other; 1-minute load below 0.2 at the
start of each). The tasks:

| task | what | time |
|---|---|---|
| `linalg` | `run.sh`: 6 backends x 1/8/32/64 threads x 3 rounds, pinned | ~66 min |
| `linalg-unpinned` | the same at 32/64 threads without pinning (`PIN=0`) | ~21 min |
| `linalg-tp` | `throughput.sh` with 64 and 128 concurrent processes | ~3 min |
| `suite` | `run_benchmarks.py --threads 1,8,32,64 --hess-threads 1,8,32,64 --throughput 128x64,512x64,1024x128` | ~14 min |

The image comes from `bench/cluster/Dockerfile` on a base image with Python 3, pip and the
g-xTB 2.0.1 build of xtb at `/opt/gxtb/bin/xtb`; it adds tblite, dftd4, MKL and OpenBLAS from
PyPI at the laptop's versions, AOCL 5.3 built from source, Rust 1.86 (and stable for the AVX-512
build), axtb and the harness:

```bash
bench/cluster/make_context.sh /path/ctx [rev]   # repo at rev (HEAD) + working-tree bench/
docker build --build-arg BASE_IMAGE=<image with the g-xTB 2.0.1 build> -t <registry>/axtb-bench:<tag> /path/ctx
```

Each task runs `bench/cluster/bench_task.sh` with the task's name, in a run directory
(`$RUN_DIR`) that holds `tasks.jsonl`, `bench_task.sh` and the matrices
(`mats/{h0.bin,s.bin,dim.txt}`, from `bench/linalg/dump`); `bench/cluster/run.yaml` records the
job settings of the recorded runs (one pod per run, CPUs, memory, timeout). The raw outputs hold
node names and paths; the report-facing copy has none (`machine.json`: CPU, cores and versions
only):

```bash
python3 bench/cluster/clean_results.py --raw <task outputs> --out bench/results/cluster-9985wx
```

After the suite, in the same pod: `bench/phases.py --threads 1,8 --big-threads 64` (->
`phases.json`), the CLI's start-up (files opened and threads started under `strace`, wall and
CPU time, best of 5) and `bench/cluster/cli_throughput.py --specs 128x64,512x64,1024x128` for
the release CLI (-> `cli_throughput.json`). The suite compares with xtb 6.8.0, so it runs in the
timing image (`axtb-timing:2026-11-07c`, built at 3499913, with `run_benchmarks.py` of f51dfe7,
the in-process timer fix; `bench_task.sh` sets `AXTB_XTB680`); the linalg tasks ran in
`axtb-bench:2026-11-07b` (3499913).

Cluster rules that matter here: thread counts are always explicit (never `nproc` or
`os.cpu_count()`, which report 128 in every pod); the job runner set `OMP_NUM_THREADS=1` in the
pods, and the scripts set every program's threads themselves (`run_benchmarks.py` passes
`OMP_NUM_THREADS` to xtb and tblite per run; the linalg harness unsets it and uses each
library's API).

### axtb CLI start-up on many-CPU machines

On the 128-thread node every axtb CLI process of the development commit 533c1a0 paid two
per-process costs that a 16-thread laptop barely sees. 0.1.0 has both fixes below (the
sequential LU and, in `third_party/`, the two crates patched to read only cpu0's cache files):
in the release run on a whole node a 1-thread CLI process opens 53 files and starts no
threads, an energy + gradient takes 2.2 ms for water, 7.3 ms for caffeine and 100 ms for taxol
(best of 5), and 128 / 512 / 1024 caffeine gradients as separate processes on 64 / 64 / 128
workers take 0.10 / 0.35 / 0.71 s against 0.16 / 0.55 / 1.03 s for xtb 6.8.0 with
`OMP_NUM_THREADS=1` (medians of 3, `cli_throughput.json`). The two costs of 533c1a0:

1. faer's x86 matmul kernels (`private-gemm-x86` 0.1.22, and `gemm-common` 0.18.2) read the
   cache sizes by opening every file under `/sys/devices/system/cpu/cpu*/cache/index*/`:
   5,771 opens per process (128 CPUs x 4 caches x ~11 files), ~21 ms of system time, and with
   64 processes starting at once the kernel serializes them (~0.4 s per process).
2. `eeq.rs` solved the EEQ system with faer's high-level `partial_piv_lu()`, which uses faer's
   global parallelism, i.e. rayon's global pool with one thread per visible CPU, even at
   `threads=1` (126 threads spawned per process in a 126-CPU pod). Fixed: sequential LU,
   identical energies.

`cluster/cli_startup_probe.sh` reproduces both (strace counts, CPU time, concurrency) and
reruns the throughput rows with `cluster/cli_throughput.py` for the original binary, the
`eeq.rs` fix, and the fix plus `cluster/patch_cpu0.py` (both crates read only cpu0's cache
files, via `[patch.crates-io]`).

## Final timings: axtb vs xtb 6.8.0 and the g-xTB programs

`final_timings.py` times axtb against the programs it replaces, one calculation at a time, on
the laptop (Intel i9-11900H, 8 cores / 16 threads) and on a shared cluster node (AMD
Threadripper PRO 9985WX, 64 cores / 128 hardware threads):

| method | reference program | energy + gradient | Hessian |
|---|---|---|---|
| GFN2-xTB | xtb 6.8.0 (official release binary) | `xtb coord --gfn 2 --grad` | `xtb coord --gfn 2 --hess` (6.8.0 default: accuracy 0.2) and `--hess --acc 0.001` vs axtb's analytic Hessian |
| g-xTB 2.0.1 | the g-xTB 2.0.1 build of xtb | `xtb coord --gxtb --grad` | `xtb coord --gxtb --hess` (default) and `--hess --acc 0.001` (a common tight setting) vs axtb's analytic Hessian |
| g-xTB v1.1 | the g-xtb 1.1.0 program | `gxtb -c mol.xyz -grad` (a numerical gradient; its energy alone is timed too) | `gxtb -c mol.xyz -hess` (numerical) vs axtb's semi-numerical Hessian |

axtb runs as the CLI (`axtb --json`, default conventions; GFN2 also in xtb conventions), with
faer (its default) and again with a run-time LAPACK library (`AXTB_LAPACK`, `libs` of the task):
Intel MKL on both machines, AMD AOCL on the node. Every axtb run gets the thread count in its
request. (The harness can also time tblite 0.8.0 and axtb's Python module; the recorded runs
leave them out, `programs` of the task; `run_benchmarks.py` compares them.)

How:
- Every calculation is a fresh process in a scratch directory of its own; `seconds` is its
  wall time from start to exit (what a caller pays).
- The programs and axtb alternate: for each molecule, `rounds` rounds (5 for energy +
  gradient, 3 for Hessians and for the slow v1.1 program) that run every program once, the
  order rotated from round to round; the summaries give min and median. Before each method's
  timings, every program runs once unrecorded (binaries into the page cache).
- Threads: energy + gradient on 1, 2, 4 and 8, Hessians on 1 and 8. The programs get
  `OMP_NUM_THREADS` = `MKL_NUM_THREADS` = threads; axtb gets the same count in its request. On
  the node the 1-thread runs are in a 2-CPU pod and the others in a 16-CPU pod, one calculation
  at a time, so our own threads never have to share a core (each core has two SMT threads); on
  the laptop each task starts at a 1-minute load below 1.5.
- Load: the node is shared. Every run records the machine's 1-minute load before and after,
  the average number of busy hardware threads while it ran (`/proc/stat`) and that minus the
  run's own CPU use (`other_busy_threads`: other work) and its own CPU time. (`max_rss_mb` is
  recorded too but is not the program's peak memory: Linux carries the forking driver's
  high-water RSS into the child at exec, so it mostly shows the driver's.)
- Accuracy: every Hessian is compared with a tight reference, axtb's analytic Hessian with SCF
  accuracy 1e-6 in the program's conventions (xtb 6.8.0: `xtb`; the g-xTB 2.0.1 build: `gxtb`,
  at the slightly rotated geometry its `--hess` uses), projected as the program projects its
  `hessian` file; axtb's own Hessian against the same in its default conventions; up to 62
  atoms also the tight analytic Hessian against a Richardson-extrapolated finite-difference
  one. Results: max |dH| (Eh/bohr²) and the largest harmonic-frequency difference (cm⁻¹).
- The molecules: water, ferrocene, caffeine, taxol (`tests/structures`), all-trans C20H42 and
  C100H202 (generated, like `large/c50h102.xyz`), C50H102, (H2O)100 and (H2O)200
  (`tests/structures/large`). Reference-program Hessians that would take hours are skipped
  (axtb's are still timed).

Run (image `cluster/Dockerfile.timing` on top of the bench image, context from
`cluster/make_timing_context.sh <dir> <rev> [<base rev>]`, which adds xtb 6.8.0's release tarball
and the g-xtb 1.1.0 program):

```bash
XTB680_TARBALL=<xtb-6.8.0-linux-x86_64-ifx-mkl.tar.xz> REGISTRY=<registry> \
  bench/cluster/make_timing_context.sh /path/ctx <rev> <base-rev>
docker build -t <registry>/axtb-timing:<tag> /path/ctx
# per thread count, one pod runs its tasks one after the other:
# cluster/timing/timing_task.sh in a working directory holding the task as task.json
python3 bench/cluster/collect_timings.py --raw <task outputs> --out bench/results/final-9985wx
```

`final_timings.py` reads the programs from the variables of `tests/programs.py`, the CLI from
`AXTB_BIN` (and the second CLI of the overhead check from `AXTB_BASE_BIN`), the repository from
`AXTB_ROOT`, the commit to record from `AXTB_COMMIT` or the file named by `AXTB_REV_FILE`, and the
Python with axtb and tblite from `PYTHON`; `timing_task.sh` sets them for the image. On the
laptop the same tasks run directly: `python bench/final_timings.py --task task.json --out
result.json`, then `collect_timings.py` on the outputs.

The tasks of the recorded run: energy + gradient, per thread count t, `{"id": "gfn2-eg-t<t>",
"parts": ["gfn2"], "threads": t, "eg": [the nine molecules], "rounds_eg": 5, "libs": {"MKL":
..., "AOCL": ...}, "programs": ["xtb 6.8.0", "axtb CLI", "axtb CLI, xtb conventions", "axtb CLI
[MKL]", ...]}` and the same for `gxtb2` (`cluster/timing/tasks-eg.jsonl`); Hessians
`gfn2-hess-t<t>`, `gxtb2-hess-t<t>` and `gxtb11-t<t>` with the molecule lists and `max_atoms`
of `cluster/timing/tasks-t1.jsonl` and `tasks-t8.jsonl`. `cluster/timing/run-t1.yaml` and
`run-t8.yaml` hold the job settings (2-CPU and 16-CPU pods, priority high). The `overhead` part
(two CLIs of different commits) is not part of the release run.

**Recorded run**: axtb 0.1.0, commit 3499913 (`results/final-laptop/`, `results/final-9985wx/`:
one file per task with every run, `summary.json` with all summaries, speed-ups and Hessian
accuracies, `machine.json`; every task's `meta.harness_check` is all true).

- **Laptop** (`results/final-laptop/`): i9-11900H, MKL 2025.3 (conda), energy + gradient
  2026-11-07 12:35-12:57 UTC (a second session 16:34-16:57 UTC, after all other laptop runs,
  `session2/`), Hessians 12:58-14:18 UTC; each task started at a 1-minute load below 1.5 with
  fewer than 0.6 hardware threads busy over 20 s (the load rose to 5-8 during the 8-thread tasks
  from these runs themselves). Scratch and output directories were on a tmpfs outside the home
  directory, the only one the desktop's file indexer scans; other work kept a median of 0.1 of
  the 16 hardware threads busy (`other_busy_threads`; `load.log`).
- **Node** (`results/final-9985wx/`): image `axtb-timing:2026-11-07c` (MKL 2026.2 from PyPI, AOCL
  5.3 built from source), energy + gradient 2026-11-07 12:18-12:25 UTC, one pod per thread
  count at the same time, Hessians and g-xTB v1.1 12:18-14:00 UTC. The nodes were shared and
  busy: other users' work kept a median of 8-25 of the 128 hardware threads busy during the 1-
  to 4-thread energy + gradient tasks and 69-72 during the 8-thread ones, whose pod ran on a
  busier node (`other_busy_threads`, up to 104; 1-minute load 19-78). Nothing was pinned.
  Absolute times move with that load (the programs' start-up alone takes 0.03-0.16 s there,
  against 0.01-0.02 s on the laptop), so read the ratios within a row: those runs alternated
  under the same conditions.

### Energy + gradient: the requirement

At 1 to 8 threads, axtb at its default conventions must be faster than the reference program at
its defaults: GFN2-xTB against xtb 6.8.0, g-xTB 2.0.1 against its program, energy + gradient,
whole processes, on both machines. On the Intel laptop axtb runs with MKL (`AXTB_LAPACK`;
faer alongside), on the AMD node with faer, its default (MKL and AOCL alongside). Median
speed-ups (program median / axtb median) on 1 / 2 / 4 / 8 threads; cells below 1.05x in bold:

| molecule | laptop GFN2-xTB [MKL] | laptop g-xTB 2.0.1 [MKL] | node GFN2-xTB (faer) | node g-xTB 2.0.1 (faer) |
|---|---|---|---|---|
| water | 8.38 / 7.34 / 6.94 / 7.13 | 2.97 / 2.26 / 2.05 / 1.96 | 17.15 / 12.34 / 10.83 / 39.17 | 19.57 / 10.06 / 8.22 / 20.48 |
| ferrocene | 2.86 / 2.70 / 2.74 / 2.90 | 3.52 / 3.00 / 2.56 / 2.19 | 6.25 / 6.18 / 5.16 / 14.71 | 8.59 / 5.15 / 4.40 / 11.98 |
| caffeine | 2.60 / 2.37 / 2.28 / 2.32 | 3.36 / 2.88 / 2.32 / 2.01 | 6.95 / 5.25 / 4.78 / 17.64 | 9.06 / 4.72 / 4.45 / 9.68 |
| C20H42 | 1.66 / 1.57 / 1.49 / 1.53 | 3.17 / 2.80 / 2.21 / 1.92 | 3.99 / 2.88 / 2.96 / 8.82 | 5.67 / 3.70 / 3.15 / 6.47 |
| taxol | 1.23 / 1.18 / **1.03** / 1.07 | 2.82 / 2.84 / 2.69 / 2.07 | 1.95 / 1.68 / 1.44 / 1.97 | 3.66 / 3.29 / 2.75 / 3.53 |
| C50H102 | 1.35 / 1.31 / 1.14 / 1.24 | 2.72 / 2.63 / 2.27 / 2.00 | 2.28 / 1.77 / 1.70 / 2.06 | 3.69 / 3.27 / 2.58 / 3.38 |
| (H2O)100 | 1.16 / 1.16 / 1.09 / 1.06 | 2.35 / 2.38 / 2.49 / 2.19 | 1.68 / 1.57 / 1.24 / 1.67 | 3.15 / 2.90 / 2.43 / 2.41 |
| C100H202 | 1.30 / 1.30 / 1.29 / 1.20 | 2.23 / 2.37 / 2.32 / 2.07 | 2.04 / 1.76 / 1.42 / 1.53 | 3.06 / 2.54 / 2.07 / 2.20 |
| (H2O)200 | 1.11 / 1.08 / 1.08 / 1.15 | 1.84 / 1.98 / 1.96 / 1.94 | 1.58 / 1.52 / 1.20 / 1.07 | 2.63 / 2.48 / 2.10 / 1.82 |

All 144 cells are above 1.0x. The lowest is taxol on 4 laptop threads with MKL, 1.03x
(per-round ratios 0.99-1.04; faer alongside 1.01x); the others below 1.1x are, on the laptop,
(H2O)100 on 8 and 4 threads (1.06x and 1.09x), taxol on 8 threads (1.07x) and (H2O)200 on 2 and
4 threads (1.08x each), and on the node (H2O)200 on 8 threads (1.07x, rounds 1.05-1.27; that pod
ran on the busier node). Ranges: laptop GFN2-xTB 1.03-8.4x, g-xTB 2.0.1 1.84-3.5x; node GFN2-xTB
1.07-39x, g-xTB 2.0.1 1.82-20x. The second laptop session (`final-laptop/session2/`, the same
tasks after all other laptop runs) gave the same picture: every cell above 1.0x, lowest again
taxol on 4 threads, 1.05x (rounds 0.97-1.06), every other cell 1.07x or more. On the node every
per-round ratio of these cells is above 1.0x (lowest 1.05x, (H2O)200 on 8 threads). On the
laptop three single rounds are below 1.0x, all taxol: on 4 threads 0.994x (session 1) and 0.966x
(session 2), on 8 threads 0.965x (session 1), one round of five each, with medians 1.03x, 1.05x
and 1.07x.

### Energy + gradient: all settings

Program / axtb, median seconds of whole processes (speed-up):

Laptop, GFN2-xTB, axtb with MKL (the requirement's setting there):

| molecule | 1 thr | 2 thr | 4 thr | 8 thr |
|---|---|---|---|---|
| water | 0.0138 / 0.0016 (8.38x) | 0.0139 / 0.0019 (7.34x) | 0.0135 / 0.0019 (6.94x) | 0.0133 / 0.0019 (7.13x) |
| ferrocene | 0.019 / 0.0066 (2.86x) | 0.0179 / 0.0066 (2.70x) | 0.0171 / 0.0062 (2.74x) | 0.0176 / 0.0061 (2.90x) |
| caffeine | 0.0208 / 0.008 (2.60x) | 0.0193 / 0.0081 (2.37x) | 0.018 / 0.0079 (2.28x) | 0.0188 / 0.0081 (2.32x) |
| C20H42 | 0.0319 / 0.0192 (1.66x) | 0.0284 / 0.0181 (1.57x) | 0.0264 / 0.0177 (1.49x) | 0.0266 / 0.0174 (1.53x) |
| taxol | 0.146 / 0.118 (1.23x) | 0.105 / 0.0892 (1.18x) | 0.0842 / 0.0818 (1.03x) | 0.0888 / 0.0828 (1.07x) |
| C50H102 | 0.112 / 0.0831 (1.35x) | 0.086 / 0.0659 (1.31x) | 0.0707 / 0.0617 (1.14x) | 0.081 / 0.0654 (1.24x) |
| (H2O)100 | 0.555 / 0.478 (1.16x) | 0.359 / 0.309 (1.16x) | 0.256 / 0.235 (1.09x) | 0.222 / 0.209 (1.06x) |
| C100H202 | 0.515 / 0.395 (1.30x) | 0.339 / 0.26 (1.30x) | 0.26 / 0.202 (1.29x) | 0.238 / 0.199 (1.20x) |
| (H2O)200 | 3.57 / 3.22 (1.11x) | 2.12 / 1.96 (1.08x) | 1.42 / 1.31 (1.08x) | 1.16 / 1.02 (1.15x) |

Laptop, GFN2-xTB, the other axtb settings (speed-ups on 1 / 2 / 4 / 8 threads):

| molecule | faer | xtb conv. | xtb conv. [MKL] |
|---|---|---|---|
| water | 8.03 / 7.00 / 6.66 / 6.67 | 10.40 / 8.99 / 8.89 / 9.09 | 11.30 / 9.52 / 9.04 / 8.82 |
| ferrocene | 2.88 / 2.60 / 2.71 / 2.83 | 2.90 / 2.69 / 2.75 / 2.91 | 2.96 / 2.75 / 2.78 / 2.91 |
| caffeine | 2.55 / 2.34 / 2.25 / 2.36 | 2.51 / 2.30 / 2.23 / 2.30 | 2.54 / 2.28 / 2.26 / 2.34 |
| C20H42 | 1.65 / 1.56 / 1.47 / 1.50 | 1.50 / 1.41 / 1.36 / 1.38 | 1.51 / 1.41 / 1.37 / 1.36 |
| taxol | 1.15 / 1.08 / 1.01 / 1.01 | 0.91 / 0.84 / 0.81 / 0.83 | 1.02 / 1.01 / 0.92 / 1.03 |
| C50H102 | 1.27 / 1.20 / 1.15 / 1.26 | 1.02 / 0.99 / 0.97 / 1.07 | 1.15 / 1.16 / 1.03 / 1.21 |
| (H2O)100 | 1.04 / 1.05 / 0.86 / 0.78 | 0.82 / 0.80 / 0.70 / 0.65 | 0.89 / 0.96 / 0.95 / 0.86 |
| C100H202 | 1.17 / 1.15 / 0.95 / 0.90 | 0.87 / 0.89 / 0.76 / 0.75 | 1.01 / 1.04 / 1.06 / 1.01 |
| (H2O)200 | 1.08 / 0.99 / 0.85 / 0.75 | 0.79 / 0.79 / 0.68 / 0.67 | 0.86 / 0.87 / 0.92 / 0.93 |

Laptop, g-xTB 2.0.1, axtb with MKL:

| molecule | 1 thr | 2 thr | 4 thr | 8 thr |
|---|---|---|---|---|
| water | 0.0068 / 0.0023 (2.97x) | 0.0064 / 0.0028 (2.26x) | 0.0062 / 0.003 (2.05x) | 0.0061 / 0.0031 (1.96x) |
| ferrocene | 0.0464 / 0.0132 (3.52x) | 0.0347 / 0.0116 (3.00x) | 0.0246 / 0.0096 (2.56x) | 0.0204 / 0.0093 (2.19x) |
| caffeine | 0.0538 / 0.016 (3.36x) | 0.0375 / 0.013 (2.88x) | 0.0271 / 0.0117 (2.32x) | 0.0223 / 0.0111 (2.01x) |
| C20H42 | 0.145 / 0.0458 (3.17x) | 0.0963 / 0.0343 (2.80x) | 0.0614 / 0.0278 (2.21x) | 0.0495 / 0.0258 (1.92x) |
| taxol | 0.845 / 0.3 (2.82x) | 0.541 / 0.19 (2.84x) | 0.357 / 0.133 (2.69x) | 0.251 / 0.121 (2.07x) |
| C50H102 | 0.623 / 0.229 (2.72x) | 0.388 / 0.147 (2.63x) | 0.246 / 0.108 (2.27x) | 0.194 / 0.0968 (2.00x) |
| (H2O)100 | 3.5 / 1.49 (2.35x) | 2.12 / 0.891 (2.38x) | 1.34 / 0.539 (2.49x) | 0.961 / 0.439 (2.19x) |
| C100H202 | 2.39 / 1.07 (2.23x) | 1.49 / 0.63 (2.37x) | 0.996 / 0.429 (2.32x) | 0.737 / 0.356 (2.07x) |
| (H2O)200 | 16.1 / 8.76 (1.84x) | 9.91 / 5.01 (1.98x) | 6.1 / 3.11 (1.96x) | 4.5 / 2.32 (1.94x) |

Laptop, g-xTB 2.0.1, axtb with faer (speed-ups):

| molecule | faer |
|---|---|
| water | 2.78 / 2.22 / 2.10 / 1.97 |
| ferrocene | 3.48 / 3.01 / 2.56 / 2.22 |
| caffeine | 3.31 / 2.83 / 2.39 / 2.03 |
| C20H42 | 3.16 / 2.80 / 2.18 / 1.89 |
| taxol | 2.79 / 2.76 / 2.72 / 2.24 |
| C50H102 | 2.65 / 2.56 / 2.24 / 1.92 |
| (H2O)100 | 2.33 / 2.32 / 2.15 / 2.05 |
| C100H202 | 2.12 / 2.09 / 1.88 / 1.68 |
| (H2O)200 | 1.73 / 1.76 / 1.65 / 1.54 |

Node, GFN2-xTB, axtb with faer (its default; the requirement's setting there):

| molecule | 1 thr | 2 thr | 4 thr | 8 thr |
|---|---|---|---|---|
| water | 0.04 / 0.0023 (17.15x) | 0.0339 / 0.0027 (12.34x) | 0.0311 / 0.0029 (10.83x) | 0.155 / 0.004 (39.17x) |
| ferrocene | 0.0367 / 0.0059 (6.25x) | 0.0393 / 0.0064 (6.18x) | 0.0309 / 0.006 (5.16x) | 0.155 / 0.0106 (14.71x) |
| caffeine | 0.0479 / 0.0069 (6.95x) | 0.0412 / 0.0079 (5.25x) | 0.0375 / 0.0078 (4.78x) | 0.166 / 0.0094 (17.64x) |
| C20H42 | 0.0567 / 0.0142 (3.99x) | 0.0424 / 0.0147 (2.88x) | 0.0427 / 0.0144 (2.96x) | 0.156 / 0.0177 (8.82x) |
| taxol | 0.175 / 0.0896 (1.95x) | 0.137 / 0.0815 (1.68x) | 0.111 / 0.0768 (1.44x) | 0.163 / 0.0829 (1.97x) |
| C50H102 | 0.138 / 0.0606 (2.28x) | 0.104 / 0.0589 (1.77x) | 0.0931 / 0.0548 (1.70x) | 0.121 / 0.0589 (2.06x) |
| (H2O)100 | 0.6 / 0.358 (1.68x) | 0.422 / 0.268 (1.57x) | 0.322 / 0.259 (1.24x) | 0.442 / 0.264 (1.67x) |
| C100H202 | 0.709 / 0.347 (2.04x) | 0.498 / 0.283 (1.76x) | 0.376 / 0.265 (1.42x) | 0.475 / 0.311 (1.53x) |
| (H2O)200 | 3.35 / 2.12 (1.58x) | 2.21 / 1.45 (1.52x) | 1.51 / 1.26 (1.20x) | 1.54 / 1.44 (1.07x) |

Node, GFN2-xTB, the other axtb settings (speed-ups on 1 / 2 / 4 / 8 threads):

| molecule | [AOCL] | [MKL] | xtb conv. | xtb conv. [AOCL] | xtb conv. [MKL] |
|---|---|---|---|---|---|
| water | 20.32 / 11.24 / 11.68 / 36.16 | 20.70 / 12.21 / 12.09 / 41.93 | 38.47 / 22.99 / 19.43 / 71.22 | 40.18 / 19.57 / 23.67 / 69.92 | 37.53 / 23.21 / 21.67 / 70.67 |
| ferrocene | 6.87 / 6.00 / 4.99 / 15.38 | 6.27 / 6.11 / 5.15 / 16.16 | 6.59 / 6.43 / 4.76 / 14.89 | 7.22 / 6.24 / 5.13 / 15.58 | 6.69 / 6.72 / 4.98 / 14.73 |
| caffeine | 7.15 / 5.73 / 5.16 / 18.18 | 7.32 / 5.40 / 5.04 / 17.97 | 6.76 / 5.43 / 4.98 / 18.78 | 7.29 / 5.60 / 4.99 / 13.76 | 6.59 / 5.33 / 5.04 / 14.19 |
| C20H42 | 4.07 / 2.86 / 3.00 / 8.73 | 4.02 / 2.92 / 2.94 / 8.35 | 3.55 / 2.61 / 2.76 / 8.33 | 3.62 / 2.57 / 2.65 / 7.76 | 3.57 / 2.66 / 2.68 / 7.32 |
| taxol | 2.03 / 1.48 / 1.23 / 1.56 | 1.30 / 1.09 / 0.94 / 1.15 | 1.70 / 1.41 / 1.09 / 1.62 | 1.88 / 1.35 / 1.03 / 1.35 | 1.28 / 1.09 / 0.91 / 1.10 |
| C50H102 | 2.40 / 1.65 / 1.42 / 1.75 | 1.39 / 1.11 / 1.11 / 1.17 | 2.05 / 1.58 / 1.50 / 1.70 | 2.23 / 1.48 / 1.31 / 1.57 | 1.36 / 1.07 / 1.01 / 1.16 |
| (H2O)100 | 1.97 / 1.66 / 1.34 / 1.60 | 1.19 / 1.18 / 1.09 / 1.40 | 1.42 / 1.35 / 1.19 / 1.24 | 1.58 / 1.34 / 1.19 / 1.23 | 1.13 / 1.08 / 1.04 / 1.28 |
| C100H202 | 2.29 / 1.82 / 1.59 / 1.63 | 1.33 / 1.35 / 1.20 / 1.25 | 1.84 / 1.52 / 1.07 / 1.27 | 1.96 / 1.64 / 1.43 / 1.35 | 1.15 / 1.16 / 1.15 / 1.16 |
| (H2O)200 | 1.84 / 1.53 / 1.23 / 1.10 | 1.15 / 1.20 / 1.11 / 0.99 | 1.38 / 1.33 / 1.13 / 0.94 | 1.50 / 1.37 / 1.05 / 0.96 | 1.08 / 1.08 / 1.06 / 0.94 |

Node, g-xTB 2.0.1, axtb with faer:

| molecule | 1 thr | 2 thr | 4 thr | 8 thr |
|---|---|---|---|---|
| water | 0.0606 / 0.0031 (19.57x) | 0.0336 / 0.0033 (10.06x) | 0.0334 / 0.0041 (8.22x) | 0.145 / 0.0071 (20.48x) |
| ferrocene | 0.0974 / 0.0113 (8.59x) | 0.0564 / 0.011 (5.15x) | 0.0468 / 0.0107 (4.40x) | 0.171 / 0.0143 (11.98x) |
| caffeine | 0.118 / 0.013 (9.06x) | 0.0556 / 0.0118 (4.72x) | 0.0501 / 0.0112 (4.45x) | 0.157 / 0.0162 (9.68x) |
| C20H42 | 0.193 / 0.034 (5.67x) | 0.102 / 0.0274 (3.70x) | 0.077 / 0.0244 (3.15x) | 0.193 / 0.0298 (6.47x) |
| taxol | 0.748 / 0.204 (3.66x) | 0.47 / 0.143 (3.29x) | 0.32 / 0.116 (2.75x) | 0.406 / 0.115 (3.53x) |
| C50H102 | 0.569 / 0.154 (3.69x) | 0.349 / 0.107 (3.27x) | 0.251 / 0.0973 (2.58x) | 0.414 / 0.123 (3.38x) |
| (H2O)100 | 2.9 / 0.923 (3.15x) | 1.73 / 0.599 (2.90x) | 1.09 / 0.45 (2.43x) | 1.03 / 0.429 (2.41x) |
| C100H202 | 2.09 / 0.684 (3.06x) | 1.29 / 0.507 (2.54x) | 0.81 / 0.391 (2.07x) | 0.91 / 0.414 (2.20x) |
| (H2O)200 | 13.3 / 5.07 (2.63x) | 8.14 / 3.28 (2.48x) | 4.85 / 2.31 (2.10x) | 4.09 / 2.24 (1.82x) |

Node, g-xTB 2.0.1, the other settings (speed-ups):

| molecule | [AOCL] | [MKL] |
|---|---|---|
| water | 24.72 / 10.16 / 8.76 / 21.60 | 18.73 / 9.46 / 8.30 / 28.01 |
| ferrocene | 9.55 / 5.42 / 4.37 / 11.11 | 9.01 / 5.31 / 4.58 / 11.68 |
| caffeine | 9.87 / 4.76 / 4.45 / 10.50 | 9.92 / 4.88 / 4.51 / 10.46 |
| C20H42 | 5.93 / 3.75 / 3.20 / 5.51 | 5.84 / 3.78 / 3.22 / 6.32 |
| taxol | 3.81 / 3.07 / 2.49 / 2.94 | 2.84 / 2.54 / 2.18 / 2.25 |
| C50H102 | 3.71 / 2.85 / 2.38 / 2.80 | 2.56 / 2.25 / 1.87 / 1.79 |
| (H2O)100 | 3.21 / 2.85 / 2.45 / 2.42 | 2.64 / 2.58 / 2.24 / 2.09 |
| C100H202 | 3.31 / 2.62 / 2.25 / 2.22 | 2.54 / 2.26 / 1.96 / 1.83 |
| (H2O)200 | 2.85 / 2.55 / 2.24 / 1.87 | 2.26 / 2.29 / 2.17 / 1.73 |

### Hessians

Node (median s; program / axtb in brackets; accuracy = largest frequency error vs the tight
reference, cm⁻¹, program `--hess` / `--acc 0.001` / axtb):

| | 1 thread | 8 threads | accuracy |
|---|---|---|---|
| GFN2, taxol (113 atoms): xtb 6.8.0 `--hess` / `--acc 0.001` / axtb analytic | 63.2 / 120 / 2.08 (30x / 58x) | 7.92 / 13.5 / 0.61 (13x / 22x) | 4.9 / 0.09 / 1e-5 |
| GFN2, C50H102 (152) | 58.7 / 96.3 / 2.65 (22x / 36x) | 11.8 / 18.3 / 0.95 (12x / 19x) | 2.4 / 0.13 / 2e-6 |
| GFN2, (H2O)100 (300) | axtb 30.2 | 82.8 / 118 / 6.42 (13x / 18x) | 14 / 0.35 / 1e-4 |
| GFN2, C100H202 (302) | axtb 25.2 | 62.3 / 113 / 5.35 (12x / 21x) | 4.8 / 0.13 / 2e-5 |
| g-xTB 2.0.1, taxol: program `--hess` / `--acc 0.001` / axtb analytic | 410 / 483 / 68.2 (6.0x / 7.1x) | 62.6 / 85.5 / 11.0 (5.7x / 7.8x) | 149 / 1.1 / 1e-5 |
| g-xTB 2.0.1, C50H102 | axtb 55.4 | 57.3 / 71.1 / 11.9 (4.8x / 6.0x) | 193 / 6.4 / 1e-6 |
| g-xTB 2.0.1, (H2O)100 | — | axtb 124 | — / — / 7e-6 |
| g-xTB v1.1, caffeine (24): program `-hess` / axtb numerical | axtb 3.79 | 433 / 0.66 (657x) | 3.9 / 0.15 |

Laptop (the same harness, caffeine and taxol):

| | 1 thread | 8 threads |
|---|---|---|
| GFN2, caffeine (24): xtb 6.8.0 `--hess` / `--acc 0.001` / axtb analytic | 0.519 / 0.736 / 0.034 (15x / 22x) | 0.130 / 0.163 / 0.022 (5.9x / 7.4x) |
| GFN2, taxol (113) | 44.3 / 76.0 / 2.89 (15x / 26x) | 8.90 / 14.9 / 0.87 (10x / 17x) |
| g-xTB 2.0.1, caffeine: program `--hess` / `--acc 0.001` / axtb analytic | 6.32 / 6.90 / 0.78 (8.1x / 8.8x) | 1.02 / 1.10 / 0.18 (5.6x / 6.0x) |
| g-xTB 2.0.1, taxol | 452 / 542 / 98.6 (4.6x / 5.5x) | 81.2 / 96.7 / 19.8 (4.1x / 4.9x) |

The accuracies are the node's (the laptop's are the same to the digits shown). Smaller molecules
on the node (water to C20H42): GFN2 21-47x (1 thread) and 10-31x (8 threads), g-xTB 2.0.1
11-18x and 7-16x. Up to 62 atoms, axtb's analytic Hessians agree with Richardson-extrapolated
finite differences to ≤ 2.7e-10 Eh/bohr² (GFN2) and ≤ 3.0e-9 (g-xTB 2.0.1; ferrocene 6.3e-8,
tblite's primitive screening). The programs' default `--hess` is off by 104-193 cm⁻¹ for
g-xTB 2.0.1 (water 0.2) and up to 14 cm⁻¹ for GFN2-xTB.

Every 1-thread axtb Hessian, its vibrational analysis included, used one core (`cores_used`
0.96-1.00 on the node, 0.99-1.00 on the laptop).

g-xTB v1.1: the 1.1.0 program's gradient is numerical (taxol 245 s on 1 thread and 236 s on 8
against axtb's 0.49 and 0.72 s); its energy alone takes 1.6-2.2x axtb's at 113-152 atoms (1 and
8 threads).

### Spot checks of the harness

`spot_check.sh` on the inputs of `spot_check_inputs.py`: each binary run by hand with bash's
`time`, alternating, three rounds, axtb with faer (`spot_checks.json` in both result
directories), wall s, program / axtb, hand runs against the harness medians above (faer column):

| cell | laptop, by hand | laptop, harness | node, by hand | node, harness |
|---|---|---|---|---|
| GFN2-xTB (H2O)100, 1 thread | 0.540-0.561 / 0.508-0.527 | 0.555 / 0.534 | 0.599-0.686 / 0.348-0.379 | 0.600 / 0.358 |
| GFN2-xTB (H2O)200, 8 threads | 1.023-1.044 / 1.470-1.487 | 1.163 / 1.544 | 1.179-1.249 / 1.109-1.441 | 1.540 / 1.444 |
| g-xTB 2.0.1 taxol, 4 threads | 0.351-0.362 / 0.127-0.141 | 0.357 / 0.131 | 0.377-0.571 / 0.120-0.153 | 0.320 / 0.116 |

The binaries were the ones the harness ran (sha256 prefixes in `spot_checks.json` and in the
tasks' `harness_check`). The node's hand runs ran in a 16-CPU pod (the harness's 1-thread runs in
a 2-CPU pod) on the busy node. On the laptop the program's (H2O)200 on 8 threads took 1.02-1.04 s
by hand against the harness's medians of 1.16 s (session 1) and 1.14 s (session 2), and 1.10 s
as the best of 3 in `attribution.py`; axtb with faer 1.47-1.49 s by hand against 1.54 s and
1.53 s, best of 3 1.53 s. Both are slower in the harness on 8 threads, xtb 6.8.0 more than axtb;
the attribution run, best of 3 with the settings interleaved, has axtb with MKL ahead there too
(1.01 s against 1.10 s).

### Where the time goes

`attribution.py` (best of 3, settings interleaved; `attribution.jsonl` in both result
directories, every case on 1, 2, 4 and 8 threads, table by `attribution_table.py`): the
program's own breakdown (xtb `--verbose`: setup = classical + integrals, SCC iterations,
gradient) next to axtb's phases (setup = classical + integrals, SCF incl. Cholesky
[diagonalization], gradient), wall s. The cases where axtb with faer is behind on the laptop,
the closest cell with MKL, and the node's closest cell:

| case | xtb 6.8.0: wall (setup / SCC / gradient) | axtb faer: wall (setup / SCF [diag.] / gradient) | axtb MKL | verdict |
|---|---|---|---|---|
| laptop (H2O)200, 1 thread | 3.63 (0.09 / 2.90 / 0.38) | 3.40 (0.28 / 3.00 [2.50] / 0.10) | 3.14 (0.28 / 2.74 [2.23] / 0.10) | ahead with both |
| laptop (H2O)200, 2 threads | 2.08 (0.08 / 1.55 / 0.24) | 2.10 (0.19 / 1.83 [1.50] / 0.07) | 1.92 (0.19 / 1.64 [1.31] / 0.07) | level with faer, ahead with MKL |
| laptop (H2O)200, 4 threads | 1.36 (0.07 / 0.92 / 0.17) | 1.61 (0.14 / 1.39 [1.14] / 0.06) | 1.27 (0.14 / 1.04 [0.79] / 0.06) | math library: faer's diagonalization alone takes longer than xtb's whole SCC |
| laptop (H2O)200, 8 threads | 1.10 (0.07 / 0.67 / 0.16) | 1.53 (0.13 / 1.32 [1.11] / 0.06) | 1.01 (0.12 / 0.80 [0.56] / 0.07) | math library (as above) |
| laptop (H2O)100, 4 / 8 threads | 0.249 / 0.221 (SCC 0.139 / 0.112) | 0.299 / 0.286 (diag. 0.195 / 0.190) | 0.225 / 0.200 (diag. 0.109 / 0.097) | math library |
| laptop C100H202, 4 / 8 threads | 0.247 / 0.230 (SCC 0.142 / 0.119) | 0.274 / 0.264 (diag. 0.191 / 0.184) | 0.201 / 0.206 (diag. 0.105 / 0.114) | math library |
| laptop taxol, 4 threads | 0.083 (0.005 / 0.048 / 0.007) | 0.082 (0.006 / 0.068 [0.056] / 0.004) | 0.082 (0.006 / 0.066 [0.043] / 0.005) | level: the library's faster diagonalization is offset by the rest of its SCF iterations at 299 basis functions |
| node (H2O)200, 8 threads | 1.37 (0.05 / 0.91 / 0.12) | 1.33 (0.10 / 1.16 [1.02] / 0.05) | 1.31 (0.10 / 1.14 [1.00] / 0.05) | ahead (AOCL 1.22, diag. 0.93) |

xtb 6.8.0 counts its three-body dispersion in its gradient, axtb in its setup (`classical`);
the sums setup + gradient are 0.47 s (xtb) and 0.38 s (axtb) for (H2O)200 on one laptop thread.
The xtb conventions (not a default) are behind on the laptop from taxol on: they keep the full
eigensolver (every eigenvector by divide and conquer, full density products) and dftd4's
per-triple three-body loop, so that they reproduce xtb 6.8.0 and tblite bit for bit.

## GFN2-xTB and g-xTB 2.0.1 performance work

Goal: axtb faster than the reference program at its defaults on 1 to 8 threads, on the laptop
and on the node: GFN2-xTB against the official xtb 6.8.0 release binary (ifx + MKL), g-xTB
2.0.1 against the g-xTB 2.0.1 build of xtb. Every phase is timed in every result (`timings`:
GFN2-xTB `classical`, `integrals`, `cholesky`, `scf: potentials`, `scf: fock`, `scf:
diagonalization`, `scf: density`, `scf: mulliken`, `scf: energy+mixing`, `gradient`; g-xTB
2.0.1 `setup: ...`, `scf: ...`, `gradient: ...`).

### What axtb took over from xtb 6.8.0

xtb 6.8.0 sped up GFN2-xTB with grimme-lab/xtb #1465, #1466, #1464, #1248, #1253 and #1272.
Whether each applies to GFN2-xTB and g-xTB 2.0.1 in axtb (the xtb and tblite conventions stay
bit for bit where noted; the gains were measured during development, before 0.1.0, and are
not repeated here):

| technique (xtb 6.8.0) | GFN2-xTB in axtb | g-xTB 2.0.1 in axtb |
|---|---|---|
| #1465: 1D Obara-Saika tables per primitive pair, all multipole components and their derivatives from one table | already | already (same kernels); extended to the q_eff coefficient-derivative sets of the gradient (one primitive pass for both contractions) |
| #1465: exact shell-pair screening (every primitive pair beyond the cutoff) | yes, bit-identical | yes (setup integrals, H0, ACP, gradient); few pairs are screened (diffuse q-vSZP functions) |
| #1466: Cholesky factor of S once per geometry | already | already |
| #1466: L⁻¹ once, reduction to standard form and back-transformation by triangular matrix products | no L⁻¹: in the default conventions, from 200 basis functions, LAPACK's two-sided reduction `dsygst` (n³ flops, half of two triangular solves): a port to faer's kernels whose last per-block solve is deferred and run in parallel (same bits on any thread count), or with `AXTB_LAPACK` the library's `dsygst` and, for C = L⁻ᵀ U, its `dtrsm`. Below 200 basis functions and in the program conventions two triangular solves, bit for bit | yes (same solver) |
| #1466: `sytrd` + `stedc`, eigenvectors back-transformed for the occupied orbitals only (occupations < 1e-16 left out) | yes, default conventions; further: the occupied eigenvectors of the tridiagonal matrix alone by inverse iteration (`crates/axtb/src/tridiag.rs`), divide and conquer for clustered spectra. The xtb and tblite conventions keep the full solver bit for bit | yes (both conventions) |
| #1466: density matrix by `syrk` | yes, default conventions: lower-triangle products | yes, also the energy-weighted density (matrix products) |
| #1466: threaded `sytrd` (MKL) | yes, faer's parallel tridiagonalization and the library's `dsytrd` (results differ between thread counts by round-off) | yes |
| #1464: three-body dispersion gradient by pair sums: pair quantities precomputed, a SIMD loop over the third atom per pair, per-thread O(N) buffers | yes, default conventions: the same loop (dense windows of pair data, a vectorized loop over the third atoms with fixed lane sums); the xtb and tblite conventions keep dftd4's triple loop bit for bit | already pair tables (one division per triple, pair derivatives summed over the third atom) |
| #1248: Fock matrix multipole potentials in one parallel pass; parallel Mulliken moments | already | yes (one tiled pass; row-parallel moments) |
| #1248: parallel isotropic and multipole electrostatics | yes, bit-identical: each atom gathers its pair terms in the order of the sequential loop (with the D4 two-body energy and potential) | yes |
| #1253: threaded gradient, faster Wiberg bond orders | already (gradient in fixed chunks; bond orders on request) | yes (gradient pair loops in fixed chunks) |
| #1272: non-periodic three-body gradient | not applicable (no periodic code) | not applicable |

Beyond these: x86-64 Linux builds target x86-64-v3 (AVX2, FMA), so axtb's own loops vectorize
4-wide with bit-identical results (Rust never fuses a multiply and an add on its own; README,
"Install", has the baseline build's times). g-xTB 2.0.1 only: the ACP as matrix products, the
exchange restructured (dense kernels once per geometry, four matrix products per Fock matrix),
DIIS with cached residual products, and every loop of its setup, SCF and gradient on the
calculator's threads in fixed chunks summed in order. With a run-time LAPACK library, matrices
below 200 rows stay on faer (the library's start-up cost more than it saved).

### Threads in the eigensolver

The SCF eigensolver's tridiagonalization (faer's, above ~313 rows; `dsytrd` with a run-time
LAPACK library) sums one partial matrix-vector product per thread, so results differ between
thread counts by round-off, as xtb's do. axtb's own reductions keep fixed chunks summed in
order and MKL its strict CNR mode, so results repeat bit for bit on a given thread count.
Measured on axtb 3499913 (`results/final-9985wx/thread_determinism.json`,
`results/final-laptop/thread_determinism.json`): on the node with faer, MKL and AOCL (taxol and
(H2O)200, GFN2-xTB and g-xTB 2.0.1, 1 vs 2, 4 and 8 threads) energies ≤ 1.8e-12 Eh apart,
gradients ≤ 8.0e-15 Eh/bohr, the same SCF iterations, and three runs per thread count
identical; GFN-FF (C50H102, (H2O)200, (H2O)333, crambin) energies ≤ 6.5e-13 Eh, gradients ≤
7.9e-14 Eh/bohr, analytic Hessians ≤ 2.3e-13 Eh/bohr²; on the laptop with faer and MKL the same
cases gave energies ≤ 1.8e-12 Eh, gradients ≤ 6.1e-15 Eh/bohr (GFN-FF ≤ 4.4e-13 Eh, ≤ 7.5e-14
Eh/bohr, Hessians ≤ 2.3e-13 Eh/bohr²), again the same iterations and identical repeats. The
gates run on one thread.

### What remains

- **faer on the Intel laptop** (the default, but not the requirement's setting there): behind
  xtb 6.8.0 for the water clusters on 4 and 8 threads ((H2O)200 also on 2) and C100H202 on 4 and
  8 threads (0.75-0.99x) and level for taxol on 4 and 8 threads (1.01x). The time is in faer's
  tridiagonalization and divide and conquer, which MKL runs faster on this CPU (AVX-512): with
  `AXTB_LAPACK` naming MKL, axtb is faster than xtb 6.8.0 in every GFN2-xTB cell on the laptop
  (1.03-8.4x), with taxol on 4 threads the closest. faer 0.24 with its AVX-512 kernels (Rust ≥ 1.89) was 6-10 % faster than
  faer 0.22 at n = 1000-3000 on one thread in `bench/linalg` and has not been tried in axtb.
- **The node**: faer, the default, is ahead everywhere (GFN2-xTB 1.07-39x, g-xTB 2.0.1
  1.82-20x). AOCL is faster still for the largest systems on 1 thread ((H2O)200: GFN2-xTB 1.84x
  against faer's 1.58x, g-xTB 2.0.1 2.85x against 2.63x) and about level with faer elsewhere,
  within the busy node's noise; MKL is slower than faer for every system large enough to use it
  (≥ 200 basis functions).
- **Three-body cutoff**: axtb's default conventions sum the three-body dispersion within
  40 bohr, xtb 6.8.0 within tblite's 25 bohr, which is part of axtb's three-body time for the
  water clusters. The cutoff is a choice of the method, not of the code, and is unchanged.
- **Sequential parts on 8 threads**: Broyden mixing, the three-body pair setup and the EEQ LU
  solve.
- **g-xTB 2.0.1** is 1.84-3.5x (laptop, MKL; faer 1.54-3.5x) and 1.82-20x (node) faster than its
  program; its revD4 three-body term was not changed.

## Release procedure

Every number in the README, `docs/validation.md`, this file and the report data
(`bench/results/report/`) is to come from runs of one commit, the release commit, not from
development runs. For axtb 0.1.0 that commit is 3499913 (2026-11-07; the results above; the
suite ran with `bench/run_benchmarks.py` of f51dfe7, which only moves its in-process timer past
an `xtb --version` call), with the GPU eigensolver's numbers of the report (`gpu_*.json`, an
experimental CUDA eigensolver measured during development, before 0.1.0, and not in this
repository) the one exception, labelled as such:

1. Record the commit (`git rev-parse HEAD`). `scripts/check.sh fast` passes on it.
2. Build the CLI and the wheel from it (README, "Install"), on the laptop and in fresh cluster
   images (`bench/cluster/make_context.sh`, `bench/cluster/make_timing_context.sh`).
3. Set the reference programs (`tests/programs.py`) and run `scripts/check.sh programs`: every
   gate passes, or is skipped for a reason recorded with the release.
4. Rerun the result files that the docs quote, each script writing its default output in
`bench/results/` (`docs/validation.md`, "Running the scripts by hand"): the reference matrix
with `--fresh` against xtb 6.8.0, the g-xTB build and the 6.7.1 release;
`program_differences.py`, `xtb680_changes.py`, `xtb_hess_acc.py`, `vibrational_check.py` and
`gfnff_reproduction.py` per xtb version; `gxtb_reproduction.py`, `gxtb2_reproduction.py`,
`gxtb_etemp_reproduction.py`, `gxtb_elements.py`, `gxtb_energy_differences.py`,
`gxtb_parity_features.py`, `solvation_composite.py`, `molden_check.py` and `scf_hard_set.py`.
5. The suite on the laptop (idle, load < 1.5) and on a reserved node: `bench/run_benchmarks.py`
   with the thread counts and throughput batches in `bench/cluster/tasks.jsonl`.
6. Per-phase timings (`bench/phases.py`) and CLI start-up (`bench/cluster/cli_throughput.py`) on
   both machines.
7. Eigensolver libraries (`bench/linalg`): rerun only if faer or a backend changed.
8. Hessians (`bench/hessian_benchmark.py`, `bench/ts_example.py`) on the node, and the
   head-to-head timings against xtb 6.8.0 and the g-xTB programs: `final_timings.py`
   ([Final timings](#final-timings-axtb-vs-xtb-680-and-the-g-xtb-programs)).
9. `python bench/make_report_data.py`; then bring the numbers in the README,
   `docs/validation.md` and this file in line with the new result files.
10. `scripts/check.sh fast` again before publishing: its privacy grep covers the new result
    files (set `AXTB_PRIVATE_PATTERN` to the site's host names).
11. Push the tag `vX.Y.Z`: `.github/workflows/release.yml` builds the wheel, the sdist and the
    CLI and publishes them (README, "Releases").
