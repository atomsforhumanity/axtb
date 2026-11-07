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

Prerequisites: Rust >= 1.86 (>= 1.89 for faer's AVX-512 build), Python 3.12 with
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
python bench/cluster/cli_throughput.py --root . --specs 64x8,128x16 --bin e6b4a18=target/release/axtb --out bench/results/cli_throughput.json
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
on the 64-core nodes (`PIN=0`): its rayon workers pinned to exactly t CPUs run 10-1000x slower
there. AOCL is loaded as `lapack:<aocl>/lib/libflame.so --blas <aocl>/lib/libblis-mt.so`;
`build_aocl.sh` builds it from AMD's GitHub sources (BSD-3) rather than AMD's prebuilt tarball,
whose evaluation EULA has a confidentiality clause and limits use to AMD hardware.

The recorded laptop rows (`linalg/results/`, axtb 0.1.0 release run, 2026-11-07 09:16-10:17
UTC) ran all six backends in one session, started at a 1-minute load below 1.5 with nothing
else running: OpenBLAS 0.3.32 (the system's pthreads build), MKL 2026.2 (the PyPI wheel),
AOCL 5.3 built with `build_aocl.sh`, faer 0.22/0.24 with Rust 1.86 and the AVX-512 build
(`faer24v4`) with Rust 1.99.1 (`env.txt` records only the first).

## On whole cluster nodes

The recorded whole-node numbers (`results/cluster-9985wx/`) ran on whole nodes (AMD
Threadripper PRO 9985WX, 64 cores / 128 threads) of a Kubernetes cluster, one task per node in a
pod of 126 CPUs, so that no other pod shared the node. They are from axtb 533c1a0 (2026-11-05):
during the release rerun of 0.1.0 no node was free for a whole-node pod, so they were not
repeated (the energy + gradient and Hessian timings of 0.1.0 on the same node, 1 to 8
threads, are in [Final timings](#final-timings-axtb-vs-xtb-680-and-the-g-xtb-programs)). The
tasks:

| task | what | time |
|---|---|---|
| `linalg` | `run.sh`: 6 backends x 1/8/32/64 threads x 3 rounds, pinned | ~65 min |
| `linalg-unpinned` | the same at 32/64 threads without pinning (`PIN=0`) | ~15 min |
| `linalg-tp` | `throughput.sh` with 64 and 128 concurrent processes | ~3 min |
| `suite` | `run_benchmarks.py --threads 1,8,32,64 --hess-threads 1,8,32,64 --throughput 128x64,512x64,1024x128` | ~17 min |

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
job settings of the recorded run (one pod per task, CPUs, memory, timeout). The raw outputs hold
node names and paths; the report-facing copy has none (`machine.json`: CPU, cores and versions
only):

```bash
python3 bench/cluster/clean_results.py --raw <task outputs> --out bench/results/cluster-9985wx
```

Then, in a container of the same image on an idle node (126 CPUs):
`bench/phases.py --threads 1,8 --big-threads 64` (-> `phases.json`) and
`bench/cluster/cli_startup_probe.sh` (-> `cli_throughput.json`; see below). The published
whole-node numbers used axtb commit 533c1a0 (image tag 2026-11-04c); for a rerun, the suite
needs an image with xtb 6.8.0 (the timing image) and `bench_task.sh` sets `AXTB_XTB680`.

Cluster rules that matter here: thread counts are always explicit (never `nproc` or
`os.cpu_count()`, which report 128 in every pod); the job runner set `OMP_NUM_THREADS=1` in the
pods, and the scripts set every program's threads themselves (`run_benchmarks.py` passes
`OMP_NUM_THREADS` to xtb and tblite per run; the linalg harness unsets it and uses each
library's API).

### axtb CLI start-up on many-CPU machines

On the 128-thread node every axtb CLI process of 533c1a0 paid two per-process costs that a
16-thread laptop barely sees (measured then, on a whole node; not repeated for 0.1.0, which has
both fixes below: the sequential LU and, in `third_party/`, the two crates patched to read only
cpu0's cache files):

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

**Recorded run**: axtb 0.1.0, commit e6b4a18 (`results/final-laptop/`, `results/final-9985wx/`:
one file per task with every run, `summary.json` with all summaries, speed-ups and Hessian
accuracies, `machine.json`; every task's `meta.harness_check` is all true).

- **Laptop** (`results/final-laptop/`): i9-11900H, MKL 2025.3 (conda), energy + gradient
  2026-11-07 07:04-07:27 UTC (a second session 10:22-10:45 UTC, `session2/`), Hessians
  07:28-08:47 UTC; each task started at a 1-minute load below 1.5 (it rose to 6-8 during the
  8-thread tasks from these runs themselves). The desktop's file indexer, which indexes the
  home directory and so the files these runs write, kept about one of the 16 hardware threads
  busy during most of both sessions (`other_busy_threads` median 1.1; `load.log`).
- **Node** (`results/final-9985wx/`): image `axtb-timing:2026-11-07b` (MKL 2026.2 from PyPI, AOCL
  5.3 built from source), energy + gradient 2026-11-07 07:07-07:16 UTC, one pod per thread
  count at the same time, Hessians and g-xTB v1.1 07:00-08:34 UTC. The node was shared and
  busy: other users' work kept a median of 36-66 of its 128 hardware threads busy during the
  energy + gradient tasks (`other_busy_threads`, up to 98; 1-minute load 39-95). Nothing was
  pinned. Absolute times move with that load (the programs' start-up alone takes 0.07-0.15 s
  there, against 0.01-0.02 s on the laptop), so read the ratios within a row: those runs
  alternated under the same conditions.

### Energy + gradient: the requirement

At 1 to 8 threads, axtb at its default conventions must be faster than the reference program at
its defaults: GFN2-xTB against xtb 6.8.0, g-xTB 2.0.1 against its program, energy + gradient,
whole processes, on both machines. On the Intel laptop axtb runs with MKL (`AXTB_LAPACK`;
faer alongside), on the AMD node with faer, its default (MKL and AOCL alongside). Median
speed-ups (program median / axtb median) on 1 / 2 / 4 / 8 threads; cells below 1.05x in bold:

| molecule | laptop GFN2-xTB [MKL] | laptop g-xTB 2.0.1 [MKL] | node GFN2-xTB (faer) | node g-xTB 2.0.1 (faer) |
|---|---|---|---|---|
| water | 8.59 / 7.34 / 7.25 / 7.40 | 3.11 / 2.67 / 2.25 / 2.26 | 46.40 / 39.93 / 28.60 / 33.17 | 31.82 / 20.83 / 21.75 / 28.38 |
| ferrocene | 3.07 / 2.95 / 2.94 / 3.38 | 3.56 / 3.13 / 2.54 / 2.83 | 20.23 / 20.71 / 17.38 / 16.77 | 11.14 / 8.85 / 11.29 / 10.59 |
| caffeine | 2.60 / 2.36 / 2.37 / 2.60 | 3.38 / 2.93 / 2.38 / 2.40 | 15.69 / 16.08 / 16.01 / 13.59 | 11.50 / 8.25 / 10.96 / 10.23 |
| C20H42 | 1.71 / 1.54 / 1.51 / 1.68 | 3.07 / 2.74 / 2.23 / 2.40 | 9.28 / 8.05 / 7.92 / 8.31 | 6.36 / 5.58 / 6.07 / 6.37 |
| taxol | 1.28 / 1.17 / **1.04** / 1.06 | 2.78 / 2.82 / 2.53 / 2.24 | 3.17 / 3.21 / 2.56 / 2.67 | 4.07 / 3.71 / 3.43 / 3.31 |
| C50H102 | 1.33 / 1.28 / 1.12 / 1.32 | 2.63 / 2.55 / 2.20 / 2.20 | 3.37 / 3.26 / 3.73 / 2.62 | 4.13 / 3.21 / 3.12 / 2.91 |
| (H2O)100 | 1.22 / 1.17 / 1.11 / 1.22 | 2.30 / 2.54 / 2.44 / 2.53 | 1.78 / 1.63 / 1.69 / 1.75 | 2.85 / 3.15 / 2.53 / 2.41 |
| C100H202 | 1.29 / 1.32 / 1.24 / 1.35 | 2.13 / 2.31 / 2.19 / 2.22 | 1.90 / 1.69 / 1.63 / 1.66 | 2.89 / 2.65 / 2.50 / 2.35 |
| (H2O)200 | 1.10 / 1.12 / 1.09 / 1.07 | 1.86 / 1.91 / 2.03 / 1.98 | 1.61 / 1.50 / 1.31 / 1.22 | 2.71 / 2.37 / 2.25 / 1.95 |

All 144 cells are above 1.0x. The lowest is taxol on 4 laptop threads with MKL, 1.04x
(per-round ratios 1.03-1.07; faer alongside 1.00x); the others below 1.1x are laptop GFN2-xTB
taxol on 8 threads (1.06x, rounds 1.00-1.11) and (H2O)200 on 4 and 8 threads (1.09x and 1.07x;
rounds 0.95-1.24 on 8 threads). Ranges: laptop GFN2-xTB 1.04-8.6x, g-xTB 2.0.1 1.86-3.6x; node
GFN2-xTB 1.22-46x, g-xTB 2.0.1 1.95-32x. The second laptop session (`final-laptop/session2/`,
the same tasks three hours later) gave the same picture: every cell above 1.0x, lowest again
taxol on 4 threads, 1.03x (rounds 1.01-1.04). On the node every per-round ratio of these cells
is above 1.0x (lowest 1.08x, (H2O)200 on 8 threads). On the laptop three single rounds of the
first session are at or below 1.0x: (H2O)200 on 8 threads 0.95x, C50H102 on 4 threads 0.98x and
taxol on 8 threads 1.00x (0.998), one round of five each, with medians 1.07x, 1.12x and 1.06x;
in the second session every per-round ratio is above 1.0x (lowest 1.01x, taxol on 4 threads).

### Energy + gradient: all settings

Program / axtb, median seconds of whole processes (speed-up):

Laptop, GFN2-xTB, axtb with MKL (the requirement's setting there):

| molecule | 1 thr | 2 thr | 4 thr | 8 thr |
|---|---|---|---|---|
| water | 0.0145 / 0.0017 (8.59x) | 0.0154 / 0.0021 (7.34x) | 0.0153 / 0.0021 (7.25x) | 0.0162 / 0.0022 (7.40x) |
| ferrocene | 0.0202 / 0.0066 (3.07x) | 0.0197 / 0.0067 (2.95x) | 0.0186 / 0.0064 (2.94x) | 0.0215 / 0.0064 (3.38x) |
| caffeine | 0.0211 / 0.0081 (2.60x) | 0.0205 / 0.0087 (2.36x) | 0.0197 / 0.0083 (2.37x) | 0.0216 / 0.0083 (2.60x) |
| C20H42 | 0.0334 / 0.0195 (1.71x) | 0.0298 / 0.0193 (1.54x) | 0.0274 / 0.0182 (1.51x) | 0.0311 / 0.0185 (1.68x) |
| taxol | 0.155 / 0.121 (1.28x) | 0.113 / 0.0967 (1.17x) | 0.0893 / 0.086 (1.04x) | 0.0985 / 0.0928 (1.06x) |
| C50H102 | 0.118 / 0.0882 (1.33x) | 0.0918 / 0.0717 (1.28x) | 0.0764 / 0.068 (1.12x) | 0.0882 / 0.067 (1.32x) |
| (H2O)100 | 0.579 / 0.475 (1.22x) | 0.378 / 0.322 (1.17x) | 0.272 / 0.246 (1.11x) | 0.289 / 0.237 (1.22x) |
| C100H202 | 0.54 / 0.418 (1.29x) | 0.373 / 0.282 (1.32x) | 0.271 / 0.218 (1.24x) | 0.293 / 0.217 (1.35x) |
| (H2O)200 | 3.7 / 3.35 (1.10x) | 2.32 / 2.07 (1.12x) | 1.52 / 1.39 (1.09x) | 1.37 / 1.28 (1.07x) |

Laptop, GFN2-xTB, the other axtb settings (speed-ups on 1 / 2 / 4 / 8 threads):

| molecule | faer | xtb conv. | xtb conv. [MKL] |
|---|---|---|---|
| water | 8.06 / 6.23 / 5.93 / 6.89 | 11.10 / 8.86 / 9.03 / 9.27 | 11.97 / 9.82 / 9.77 / 9.84 |
| ferrocene | 2.85 / 2.79 / 2.90 / 3.35 | 3.03 / 2.93 / 3.00 / 3.34 | 3.00 / 3.01 / 3.03 / 3.26 |
| caffeine | 2.48 / 2.35 / 2.33 / 2.59 | 2.47 / 2.27 / 2.31 / 2.54 | 2.53 / 2.29 / 2.32 / 2.53 |
| C20H42 | 1.71 / 1.56 / 1.49 / 1.70 | 1.54 / 1.42 / 1.35 / 1.56 | 1.52 / 1.40 / 1.38 / 1.53 |
| taxol | 1.16 / 1.06 / 1.00 / 1.09 | 0.94 / 0.87 / 0.81 / 0.89 | 1.03 / 1.00 / 0.94 / 0.97 |
| C50H102 | 1.28 / 1.24 / 1.21 / 1.31 | 1.06 / 1.05 / 0.99 / 1.10 | 1.19 / 1.16 / 1.04 / 1.05 |
| (H2O)100 | 1.09 / 1.06 / 0.83 / 0.96 | 0.86 / 0.82 / 0.70 / 0.81 | 0.92 / 0.94 / 0.94 / 0.95 |
| C100H202 | 1.20 / 1.18 / 0.92 / 1.05 | 0.91 / 0.90 / 0.75 / 0.86 | 1.02 / 1.09 / 1.03 / 1.14 |
| (H2O)200 | 1.05 / 1.05 / 0.86 / 0.81 | 0.82 / 0.82 / 0.71 / 0.72 | 0.88 / 0.91 / 0.91 / 0.89 |

Laptop, g-xTB 2.0.1, axtb with MKL:

| molecule | 1 thr | 2 thr | 4 thr | 8 thr |
|---|---|---|---|---|
| water | 0.008 / 0.0026 (3.11x) | 0.008 / 0.003 (2.67x) | 0.0074 / 0.0033 (2.25x) | 0.0078 / 0.0034 (2.26x) |
| ferrocene | 0.0499 / 0.014 (3.56x) | 0.0356 / 0.0114 (3.13x) | 0.0255 / 0.01 (2.54x) | 0.0272 / 0.0096 (2.83x) |
| caffeine | 0.056 / 0.0166 (3.38x) | 0.0398 / 0.0136 (2.93x) | 0.0291 / 0.0122 (2.38x) | 0.0288 / 0.012 (2.40x) |
| C20H42 | 0.15 / 0.0488 (3.07x) | 0.0977 / 0.0357 (2.74x) | 0.0639 / 0.0287 (2.23x) | 0.0657 / 0.0274 (2.40x) |
| taxol | 0.881 / 0.317 (2.78x) | 0.555 / 0.197 (2.82x) | 0.361 / 0.142 (2.53x) | 0.297 / 0.133 (2.24x) |
| C50H102 | 0.65 / 0.247 (2.63x) | 0.403 / 0.158 (2.55x) | 0.256 / 0.116 (2.20x) | 0.253 / 0.115 (2.20x) |
| (H2O)100 | 3.59 / 1.56 (2.30x) | 2.25 / 0.885 (2.54x) | 1.39 / 0.569 (2.44x) | 1.22 / 0.482 (2.53x) |
| C100H202 | 2.46 / 1.16 (2.13x) | 1.53 / 0.663 (2.31x) | 0.996 / 0.455 (2.19x) | 0.918 / 0.414 (2.22x) |
| (H2O)200 | 16.3 / 8.75 (1.86x) | 9.64 / 5.05 (1.91x) | 6.48 / 3.2 (2.03x) | 5.16 / 2.6 (1.98x) |

Laptop, g-xTB 2.0.1, axtb with faer (speed-ups):

| molecule | faer |
|---|---|
| water | 2.91 / 2.49 / 2.21 / 2.16 |
| ferrocene | 3.55 / 3.11 / 2.55 / 2.71 |
| caffeine | 3.21 / 2.81 / 2.41 / 2.43 |
| C20H42 | 3.09 / 2.68 / 2.23 / 2.40 |
| taxol | 2.76 / 2.79 / 2.65 / 2.44 |
| C50H102 | 2.62 / 2.53 / 2.18 / 2.41 |
| (H2O)100 | 2.25 / 2.39 / 2.18 / 2.38 |
| C100H202 | 2.07 / 2.10 / 1.82 / 1.96 |
| (H2O)200 | 1.73 / 1.69 / 1.70 / 1.63 |

Node, GFN2-xTB, axtb with faer (its default; the requirement's setting there):

| molecule | 1 thr | 2 thr | 4 thr | 8 thr |
|---|---|---|---|---|
| water | 0.143 / 0.0031 (46.40x) | 0.139 / 0.0035 (39.93x) | 0.131 / 0.0046 (28.60x) | 0.124 / 0.0037 (33.17x) |
| ferrocene | 0.144 / 0.0071 (20.23x) | 0.147 / 0.0071 (20.71x) | 0.133 / 0.0077 (17.38x) | 0.133 / 0.0079 (16.77x) |
| caffeine | 0.139 / 0.0088 (15.69x) | 0.152 / 0.0095 (16.08x) | 0.131 / 0.0082 (16.01x) | 0.126 / 0.0092 (13.59x) |
| C20H42 | 0.154 / 0.0166 (9.28x) | 0.153 / 0.019 (8.05x) | 0.14 / 0.0176 (7.92x) | 0.147 / 0.0178 (8.31x) |
| taxol | 0.341 / 0.108 (3.17x) | 0.302 / 0.0943 (3.21x) | 0.229 / 0.0892 (2.56x) | 0.235 / 0.0881 (2.67x) |
| C50H102 | 0.241 / 0.0715 (3.37x) | 0.214 / 0.0655 (3.26x) | 0.234 / 0.0628 (3.73x) | 0.213 / 0.0812 (2.62x) |
| (H2O)100 | 0.615 / 0.346 (1.78x) | 0.442 / 0.272 (1.63x) | 0.494 / 0.293 (1.69x) | 0.458 / 0.262 (1.75x) |
| C100H202 | 0.565 / 0.297 (1.90x) | 0.394 / 0.233 (1.69x) | 0.501 / 0.307 (1.63x) | 0.442 / 0.266 (1.66x) |
| (H2O)200 | 3.45 / 2.13 (1.61x) | 2.17 / 1.45 (1.50x) | 2.04 / 1.56 (1.31x) | 1.66 / 1.36 (1.22x) |

Node, GFN2-xTB, the other axtb settings (speed-ups on 1 / 2 / 4 / 8 threads):

| molecule | [AOCL] | [MKL] | xtb conv. | xtb conv. [AOCL] | xtb conv. [MKL] |
|---|---|---|---|---|---|
| water | 64.59 / 44.95 / 42.19 / 37.18 | 60.34 / 50.60 / 32.32 / 34.28 | 94.78 / 71.03 / 67.84 / 53.60 | 140.90 / 72.58 / 78.26 / 62.20 | 131.81 / 74.74 / 64.87 / 54.07 |
| ferrocene | 22.38 / 15.23 / 18.64 / 18.08 | 21.56 / 20.61 / 17.59 / 18.11 | 22.16 / 20.32 / 16.98 / 17.26 | 21.89 / 18.99 / 17.09 / 17.85 | 22.06 / 20.36 / 18.99 / 17.28 |
| caffeine | 18.05 / 19.14 / 17.03 / 14.72 | 16.10 / 17.52 / 15.11 / 14.21 | 16.02 / 15.57 / 15.32 / 14.04 | 16.92 / 14.29 / 16.83 / 13.55 | 17.33 / 17.45 / 16.08 / 14.00 |
| C20H42 | 9.41 / 8.82 / 8.25 / 8.32 | 9.38 / 8.39 / 8.36 / 7.75 | 8.24 / 7.47 / 7.36 / 7.05 | 8.21 / 7.93 / 7.70 / 5.86 | 8.15 / 7.78 / 7.27 / 7.71 |
| taxol | 3.38 / 2.80 / 2.19 / 1.92 | 1.27 / 1.38 / 1.22 / 1.30 | 2.75 / 2.76 / 1.93 / 2.06 | 3.05 / 2.87 / 1.89 / 1.78 | 1.36 / 1.40 / 1.32 / 1.32 |
| C50H102 | 3.51 / 3.25 / 3.02 / 2.59 | 1.33 / 1.48 / 1.52 / 1.40 | 3.03 / 2.88 / 2.93 / 2.81 | 3.21 / 2.66 / 2.57 / 2.23 | 1.28 / 1.32 / 1.46 / 1.33 |
| (H2O)100 | 1.98 / 1.62 / 1.71 / 1.48 | 1.24 / 1.25 / 1.10 / 1.20 | 1.55 / 1.39 / 1.41 / 1.33 | 1.60 / 1.33 / 1.50 / 1.29 | 1.14 / 1.12 / 1.15 / 1.17 |
| C100H202 | 2.17 / 1.64 / 1.88 / 1.74 | 1.28 / 1.23 / 1.31 / 1.27 | 1.62 / 1.41 / 1.47 / 1.26 | 1.76 / 1.37 / 1.40 / 1.44 | 1.23 / 1.13 / 1.32 / 1.19 |
| (H2O)200 | 1.93 / 1.52 / 1.34 / 1.26 | 1.18 / 1.13 / 1.17 / 1.05 | 1.45 / 1.36 / 1.11 / 1.08 | 1.49 / 1.28 / 1.19 / 1.08 | 1.11 / 1.08 / 1.10 / 1.04 |

Node, g-xTB 2.0.1, axtb with faer:

| molecule | 1 thr | 2 thr | 4 thr | 8 thr |
|---|---|---|---|---|
| water | 0.104 / 0.0033 (31.82x) | 0.0733 / 0.0035 (20.83x) | 0.104 / 0.0048 (21.75x) | 0.119 / 0.0042 (28.38x) |
| ferrocene | 0.133 / 0.0119 (11.14x) | 0.0957 / 0.0108 (8.85x) | 0.126 / 0.0112 (11.29x) | 0.137 / 0.013 (10.59x) |
| caffeine | 0.148 / 0.0129 (11.50x) | 0.1 / 0.0121 (8.25x) | 0.139 / 0.0127 (10.96x) | 0.136 / 0.0132 (10.23x) |
| C20H42 | 0.221 / 0.0347 (6.36x) | 0.158 / 0.0283 (5.58x) | 0.164 / 0.027 (6.07x) | 0.175 / 0.0274 (6.37x) |
| taxol | 0.848 / 0.208 (4.07x) | 0.557 / 0.15 (3.71x) | 0.447 / 0.13 (3.43x) | 0.38 / 0.115 (3.31x) |
| C50H102 | 0.911 / 0.221 (4.13x) | 0.401 / 0.125 (3.21x) | 0.327 / 0.105 (3.12x) | 0.305 / 0.105 (2.91x) |
| (H2O)100 | 3.58 / 1.25 (2.85x) | 1.93 / 0.614 (3.15x) | 1.37 / 0.543 (2.53x) | 0.978 / 0.406 (2.41x) |
| C100H202 | 3.12 / 1.08 (2.89x) | 1.36 / 0.511 (2.65x) | 0.995 / 0.398 (2.50x) | 0.885 / 0.376 (2.35x) |
| (H2O)200 | 19.9 / 7.35 (2.71x) | 11.8 / 4.98 (2.37x) | 5.57 / 2.48 (2.25x) | 3.9 / 2.01 (1.95x) |

Node, g-xTB 2.0.1, the other settings (speed-ups):

| molecule | [AOCL] | [MKL] |
|---|---|---|
| water | 36.03 / 21.63 / 24.79 / 32.46 | 36.39 / 22.67 / 20.47 / 31.20 |
| ferrocene | 11.94 / 9.08 / 11.66 / 11.13 | 11.23 / 8.98 / 11.18 / 11.42 |
| caffeine | 12.13 / 8.32 / 10.35 / 9.45 | 11.51 / 7.98 / 10.58 / 10.89 |
| C20H42 | 6.39 / 5.78 / 6.10 / 6.34 | 6.47 / 5.58 / 6.22 / 6.87 |
| taxol | 4.21 / 3.37 / 2.97 / 2.92 | 2.94 / 2.67 / 2.27 / 1.94 |
| C50H102 | 3.66 / 3.08 / 2.88 / 2.70 | 2.63 / 2.31 / 1.79 / 1.86 |
| (H2O)100 | 3.15 / 3.07 / 2.85 / 2.39 | 2.09 / 2.46 / 2.42 / 2.13 |
| C100H202 | 3.47 / 2.79 / 2.50 / 2.33 | 2.58 / 2.27 / 2.11 / 1.93 |
| (H2O)200 | 3.01 / 2.61 / 2.28 / 2.11 | 2.38 / 2.04 / 2.00 / 1.93 |

### Hessians

Node (median s; program / axtb in brackets; accuracy = largest frequency error vs the tight
reference, cm⁻¹, program `--hess` / `--acc 0.001` / axtb):

| | 1 thread | 8 threads | accuracy |
|---|---|---|---|
| GFN2, taxol (113 atoms): xtb 6.8.0 `--hess` / `--acc 0.001` / axtb analytic | 53.4 / 105 / 1.78 (30x / 59x) | 9.38 / 16.9 / 0.66 (14x / 26x) | 4.9 / 0.09 / 1e-5 |
| GFN2, C50H102 (152) | 57.2 / 98.0 / 2.54 (23x / 39x) | 11.9 / 20.2 / 0.90 (13x / 22x) | 2.4 / 0.13 / 2e-6 |
| GFN2, (H2O)100 (300) | axtb 33.3 | 90.2 / 118 / 7.75 (12x / 15x) | 14 / 0.35 / 1e-4 |
| GFN2, C100H202 (302) | axtb 29.3 | 68.6 / 115 / 5.84 (12x / 20x) | 4.8 / 0.13 / 2e-5 |
| g-xTB 2.0.1, taxol: program `--hess` / `--acc 0.001` / axtb analytic | 351 / 445 / 53.9 (6.5x / 8.2x) | 52.2 / 60.2 / 8.80 (5.9x / 6.8x) | 149 / 1.1 / 1e-5 |
| g-xTB 2.0.1, C50H102 | axtb 58.7 | 39.5 / 53.3 / 8.26 (4.8x / 6.4x) | 193 / 6.4 / 1e-6 |
| g-xTB 2.0.1, (H2O)100 | — | axtb 110 | — / — / 7e-6 |
| g-xTB v1.1, caffeine (24): program `-hess` / axtb numerical | axtb 3.85 | 413 / 0.61 (680x) | 3.9 / 0.15 |

Laptop (the same harness, caffeine and taxol):

| | 1 thread | 8 threads |
|---|---|---|
| GFN2, caffeine (24): xtb 6.8.0 `--hess` / `--acc 0.001` / axtb analytic | 0.536 / 0.759 / 0.038 (14x / 20x) | 0.169 / 0.221 / 0.026 (6.5x / 8.5x) |
| GFN2, taxol (113) | 44.5 / 76.3 / 2.96 (15x / 26x) | 9.25 / 15.5 / 0.92 (10x / 17x) |
| g-xTB 2.0.1, caffeine: program `--hess` / `--acc 0.001` / axtb analytic | 6.46 / 7.05 / 0.80 (8.1x / 8.9x) | 1.49 / 1.57 / 0.21 (7.3x / 7.6x) |
| g-xTB 2.0.1, taxol | 455 / 533 / 95.6 (4.8x / 5.6x) | 80.0 / 95.5 / 19.8 (4.0x / 4.8x) |

The accuracies are the node's (the laptop's are the same to the digits shown). Smaller molecules
on the node (water to C20H42): GFN2 21-41x (1 thread) and 10-24x (8 threads), g-xTB 2.0.1
11-20x and 6-12x. Up to 62 atoms, axtb's analytic Hessians agree with Richardson-extrapolated
finite differences to ≤ 2.7e-10 Eh/bohr² (GFN2) and ≤ 3.0e-9 (g-xTB 2.0.1; ferrocene 6.3e-8,
tblite's primitive screening). The programs' default `--hess` is off by 104-193 cm⁻¹ for
g-xTB 2.0.1 (water 0.2) and up to 14 cm⁻¹ for GFN2-xTB.

One thing to know when timing small molecules: a Hessian request with the vibrational analysis
(the default) runs faer's matrix products of the analysis on faer's global thread pool, one
thread per CPU, whatever `threads` says (`crates/axtb/src/vibrations/mod.rs`); on the laptop a
1-thread caffeine Hessian used 1.7-1.9 cores (with `vibrations: false` 1.0). The Hessian itself
stays on its threads.

g-xTB v1.1: the 1.1.0 program's gradient is numerical (taxol 231 s on 1 thread and 205 s on 8
against axtb's 0.49 and 0.51 s); its energy alone takes 2.0-2.3x axtb's at 113-152 atoms (1 and
8 threads).

### Spot checks of the harness

`spot_check.sh` on the inputs of `spot_check_inputs.py`: each binary run by hand with bash's
`time`, alternating, three rounds, axtb with faer (`spot_checks.json` in both result
directories), wall s, program / axtb, hand runs against the harness medians above (faer column):

| cell | laptop, by hand | laptop, harness | node, by hand | node, harness |
|---|---|---|---|---|
| GFN2-xTB (H2O)100, 1 thread | 0.554-0.560 / 0.515-0.535 | 0.579 / 0.533 | 0.663-0.733 / 0.379-0.395 | 0.615 / 0.346 |
| GFN2-xTB (H2O)200, 8 threads | 1.023-1.046 / 1.434-1.482 | 1.370 / 1.689 | 1.343-1.532 / 1.039-1.408 | 1.659 / 1.363 |
| g-xTB 2.0.1 taxol, 4 threads | 0.339-0.347 / 0.132-0.135 | 0.361 / 0.136 | 0.394-0.538 / 0.101-0.135 | 0.447 / 0.130 |

The binaries were the ones the harness ran (sha256 prefixes in `spot_checks.json` and in the
tasks' `harness_check`). The node's hand runs ran in a 16-CPU pod (the harness's 1-thread runs in
a 2-CPU pod) on the busy node. On the laptop the program's (H2O)200 on 8 threads took 1.02-1.05 s
by hand against the harness's medians of 1.37 s (session 1) and 1.59 s (session 2), and 1.12 s
as the best of 3 in `attribution.py`; axtb with faer 1.43-1.48 s by hand against 1.69 s and
1.65 s, best of 3 1.50 s. Both programs lose time in the harness on 8 threads while the indexer
keeps a hardware thread busy, xtb 6.8.0's OpenMP more than axtb; the attribution run, best of 3
with the settings interleaved, has axtb with MKL ahead there too (1.01 s against 1.12 s).

### Where the time goes

`attribution.py` (best of 3, settings interleaved; `attribution.jsonl` in both result
directories, every case on 1, 2, 4 and 8 threads, table by `attribution_table.py`): the
program's own breakdown (xtb `--verbose`: setup = classical + integrals, SCC iterations,
gradient) next to axtb's phases (setup = classical + integrals, SCF incl. Cholesky
[diagonalization], gradient), wall s. The cases where axtb with faer is behind on the laptop,
the closest cell with MKL, and the node's closest cell:

| case | xtb 6.8.0: wall (setup / SCC / gradient) | axtb faer: wall (setup / SCF [diag.] / gradient) | axtb MKL | verdict |
|---|---|---|---|---|
| laptop (H2O)200, 1 thread | 3.61 (0.09 / 2.88 / 0.38) | 3.37 (0.28 / 2.98 [2.48] / 0.10) | 3.12 (0.28 / 2.73 [2.23] / 0.10) | ahead with both |
| laptop (H2O)200, 4 threads | 1.39 (0.08 / 0.94 / 0.18) | 1.64 (0.15 / 1.41 [1.16] / 0.06) | 1.25 (0.14 / 1.03 [0.78] / 0.06) | math library: faer's diagonalization alone takes longer than xtb's whole SCC |
| laptop (H2O)200, 8 threads | 1.12 (0.07 / 0.69 / 0.16) | 1.50 (0.13 / 1.29 [1.08] / 0.06) | 1.01 (0.13 / 0.79 [0.54] / 0.06) | math library (as above) |
| laptop (H2O)100, 4 / 8 threads | 0.257 / 0.224 (SCC 0.144 / 0.113) | 0.300 / 0.284 (diag. 0.198 / 0.187) | 0.240 / 0.218 (diag. 0.127 / 0.109) | math library |
| laptop C100H202, 4 / 8 threads | 0.243 / 0.224 (SCC 0.139 / 0.116) | 0.270 / 0.263 (diag. 0.186 / 0.182) | 0.198 / 0.180 (diag. 0.103 / 0.093) | math library |
| laptop taxol, 4 threads | 0.084 (0.005 / 0.049 / 0.007) | 0.081 (0.006 / 0.067 [0.055] / 0.004) | 0.082 (0.006 / 0.066 [0.043] / 0.005) | level: the library's faster diagonalization is offset by the rest of its SCF iterations at 299 basis functions |
| node (H2O)200, 8 threads | 2.07 (0.08 / 1.33 / 0.20) | 1.50 (0.15 / 1.25 [1.09] / 0.07) | 1.86 (0.14 / 1.61 [1.43] / 0.07) | ahead (AOCL 1.67, diag. 1.27) |

xtb 6.8.0 counts its three-body dispersion in its gradient, axtb in its setup (`classical`);
the sums setup + gradient are 0.47 s (xtb) and 0.37 s (axtb) for (H2O)200 on one laptop thread.
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
Measured on axtb e6b4a18 (`results/final-9985wx/thread_determinism.json`,
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
  xtb 6.8.0 for the water clusters on 4 and 8 threads and C100H202 on 4 threads (0.81-0.96x)
  and level for taxol on 4 threads (1.00x). The time is in faer's tridiagonalization and divide
  and conquer, which MKL runs faster on this CPU (AVX-512): with `AXTB_LAPACK` naming MKL, axtb
  is faster than xtb 6.8.0 in every GFN2-xTB cell on the laptop (1.04-8.6x), with taxol on 4
  threads the closest. faer 0.24 with its AVX-512 kernels (Rust ≥ 1.89) was 6-10 % faster than
  faer 0.22 at n = 1000-3000 on one thread in `bench/linalg` and has not been tried in axtb.
- **The node**: faer, the default, is ahead everywhere (GFN2-xTB 1.22-46x, g-xTB 2.0.1
  1.95-32x). AOCL is faster still for the largest systems on 1 thread ((H2O)200: GFN2-xTB 1.93x
  against faer's 1.61x, g-xTB 2.0.1 3.01x against 2.71x) and about level with faer elsewhere,
  within the busy node's noise; MKL is slower than faer for every system large enough to use it
  (≥ 200 basis functions).
- **Three-body cutoff**: axtb's default conventions sum the three-body dispersion within
  40 bohr, xtb 6.8.0 within tblite's 25 bohr, which is part of axtb's three-body time for the
  water clusters. The cutoff is a choice of the method, not of the code, and is unchanged.
- **Sequential parts on 8 threads**: Broyden mixing, the three-body pair setup and the EEQ LU
  solve.
- **g-xTB 2.0.1** is 1.86-3.6x (laptop, MKL; faer 1.63-3.6x) and 1.95-32x (node) faster than its
  program; its revD4 three-body term was not changed.
- **The vibrational analysis** of a Hessian request runs faer's matrix products on faer's
  global thread pool (all CPUs) whatever the request's `threads` (see Hessians above).

## Release procedure

Every number in the README, `docs/validation.md`, this file and the report data
(`bench/results/report/`) is to come from runs of one commit, the release commit, not from
development runs. For axtb 0.1.0 that commit is e6b4a18 (2026-11-07; the results above), with
the GPU eigensolver's numbers of the report (`gpu_*.json`, branch `gpu-backend` of the
development repository, before 0.1.0) the one exception, labelled as such:

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
