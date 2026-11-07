# axtb benchmarks

| Path | What |
|---|---|
| `run_benchmarks.py` | axtb vs xtb (`--xtb`, default xtb 6.8.0 from `$AXTB_XTB680`; the recorded results used xtb 6.7.1, the g-xTB 2.0.1 build; axtb in the conventions of its version) vs tblite 0.8.0 (GFN2-xTB): accuracy, solvation, timings, Hessians, throughput |
| `phases.py` | where axtb's time goes: per-phase times (`r["timings"]`) and SCF iterations for the timing molecules |
| `linalg/` | standalone crate comparing dense eigensolver backends (faer 0.22/0.24, OpenBLAS, MKL, AMD AOCL) on real GFN2 matrices |
| `cluster/` | image, task scripts, the job settings of the recorded runs and helpers to run all of it on whole cluster nodes |
| `results/results.json`, `linalg/results/` | laptop: Intel Core i9-11900H, 8 cores / 16 threads |
| `results/cluster-9985wx/` | cluster node: AMD Ryzen Threadripper PRO 9985WX (Zen 5), 64 cores / 128 threads (`machine.json`) |
| `final_timings.py`, `results/final-9985wx/` | the final timings: GFN2-xTB vs xtb 6.8.0 (and tblite 0.8.0), g-xTB 2.0.1 and v1.1 vs their programs, energy + gradient and Hessians, 1 and 8 threads, on a cluster node (see [Final timings](#final-timings-axtb-vs-xtb-680-and-the-g-xtb-programs)) |
| `results/final-9985wx-gxtb-perf/`, `results/laptop-perf/` | energy + gradient timings at 1, 2, 4 and 8 threads after the GFN2-xTB and g-xTB 2.0.1 performance work, axtb with faer, MKL and AOCL, on the node and the laptop, with attributions and spot checks (see [the performance work](#gfn2-xtb-and-g-xtb-201-performance-work-resultsfinal-9985wx-gxtb-perf-resultslaptop-perf)) |
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
python bench/run_benchmarks.py --out bench/results/results.json   # full suite, ~30 min
python bench/phases.py --threads 1,8 --out bench/results/phases.json
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

The laptop's AOCL rows (`linalg/results/aocl_*`) were measured in a later session than
`raw.tsv`, with faer 0.22 re-run in the same session as an anchor. The laptop is shared with
other work, so each segment (one backend at one thread count) was started only at 1-minute
load < 1.5 and kept only if other processes used < 1.5 CPUs on average while it ran
(`aocl_segments.tsv`: every attempt, kept or not; `aocl_load.log`: `uptime` before/after).

## On whole cluster nodes

The recorded cluster numbers ran on whole nodes (AMD Threadripper PRO 9985WX, 64 cores / 128
threads) of a Kubernetes cluster, one task per node in a pod of 126 CPUs, so that no other pod
shared the node:

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
cluster numbers used axtb commit 533c1a0 (image tag 2026-11-04c).

Cluster rules that matter here: thread counts are always explicit (never `nproc` or
`os.cpu_count()`, which report 128 in every pod); the job runner set `OMP_NUM_THREADS=1` in the
pods, and the scripts set every program's threads themselves (`run_benchmarks.py` passes
`OMP_NUM_THREADS` to xtb and tblite per run; the linalg harness unsets it and uses each
library's API).

### axtb CLI start-up on many-CPU machines

On the 128-thread node every axtb CLI process (533c1a0) paid two per-process costs that a
16-thread laptop barely sees:

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

`final_timings.py` times axtb against the programs it replaces, one calculation at a time,
on a shared cluster node (AMD Threadripper PRO 9985WX, 64 cores / 128 hardware threads):

| method | reference program | energy + gradient | Hessian |
|---|---|---|---|
| GFN2-xTB | xtb 6.8.0 (official release binary), tblite 0.8.0 (PyPI) | `xtb coord --gfn 2 --grad`; tblite's Python `Calculator("GFN2-xTB")` | `xtb coord --gfn 2 --hess` (6.8.0 default: accuracy 0.2) and `--hess --acc 0.001` vs axtb's analytic Hessian |
| g-xTB 2.0.1 | the g-xTB 2.0.1 build of xtb | `xtb coord --gxtb --grad` | `xtb coord --gxtb --hess` (default) and `--hess --acc 0.001` (a common tight setting) vs axtb's analytic Hessian |
| g-xTB v1.1 | the g-xtb 1.1.0 program | `gxtb -c mol.xyz -grad` (a numerical gradient; its energy alone is timed too) | `gxtb -c mol.xyz -hess` (numerical) vs axtb's semi-numerical Hessian |

axtb runs as the CLI (`axtb --json`, default conventions; GFN2 also in xtb conventions) and
as its Python module in a fresh process; every axtb run gets the thread count in its request.

How:
- Every calculation is a fresh process in a scratch directory of its own; `seconds` is its
  wall time from start to exit (what a caller pays). tblite and axtb's Python module also
  report the calculation alone (`inner_seconds`, without Python start-up and imports).
- The programs and axtb alternate: for each molecule, `rounds` rounds (5 for energy +
  gradient, 3 for Hessians, 3 for the slow v1.1 program) that run every program once, the
  order rotated from round to round; the summaries give min and median. Before each method's
  timings, every program runs once unrecorded (binaries into the page cache).
- Threads: 1 (the setting for many calculations side by side) and 8. The programs get `OMP_NUM_THREADS` =
  `MKL_NUM_THREADS` = threads; axtb gets the same count in its request. The 1-thread runs are
  in a 2-CPU pod and the 8-thread runs in a 16-CPU pod, one calculation at a time, so our own
  threads never have to share a core (each core has two SMT threads).
- Load: the node is shared. Every run records the node's 1-minute load before and after, the
  average number of busy hardware threads on the node while it ran (`/proc/stat`) and that
  minus the run's own CPU use (`other_busy_threads`: other users' work) and its own CPU time.
  (`max_rss_mb` is recorded too but is not the program's peak memory: Linux carries the forking
  driver's high-water RSS into the child at exec, so it mostly shows the driver's.)
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
# per thread count, one pod runs the tasks of cluster/timing/tasks-t<N>.jsonl one after the other:
# cluster/timing/timing_task.sh in a working directory holding the task as task.json
python3 bench/cluster/collect_timings.py --raw <task outputs> --out bench/results/final-9985wx
```

`final_timings.py` reads the programs from the variables of `tests/programs.py`, the CLI from
`AXTB_BIN` (and the second CLI of the overhead check from `AXTB_BASE_BIN`), the repository from
`AXTB_ROOT`, the commit to record from `AXTB_COMMIT` or the file named by `AXTB_REV_FILE`, and the
Python with axtb and tblite from `PYTHON`; `timing_task.sh` sets them for the image.

The recorded run's job settings are `cluster/timing/run-t1.yaml` with `tasks-t1.jsonl` and
`run-t8.yaml` with `tasks-t8.jsonl` (two runs, one pod each, the tasks one after the other); `overhead-t1` times
the CLI of the commit before the g-xTB parity features (bond orders, quadrupole, guess reuse)
against the measured one on plain energy + gradient, and `tasks-null.jsonl` is the same
comparison with two CLIs built from one commit (the noise floor of that comparison).

**Recorded run** (`results/final-9985wx/`: one file per task with every run, `summary.json`
with all summaries, speed-ups and Hessian accuracies, `machine.json`): axtb 0812006, image
`axtb-timing:2026-11-05d`, 2026-11-06 04:54-06:38 UTC, the 1-thread and the 8-thread run on two
different nodes at the same time. The nodes were shared: another user's job (32 CPUs per node)
kept a median of ~32 of each node's 128 hardware threads busy (`other_busy_threads`; up to 64
in the last half hour of the 1-thread run, when a second job started), 1-minute load 3-66. With
at most ~70 busy threads on 64 cores our threads had idle cores most of the time; nothing was
pinned, so SMT sharing with other work cannot be ruled out in the busiest minutes. Absolute
times move by up to ~20 % with that load (taxol `xtb --hess`, 1 thread: 34.6 s in the pilot on
a quieter node, 41.4 s here), so read the ratios within a row: those runs alternated under the
same conditions. Process start-up dominates below ~50 atoms, and there the xtb programs' times
scatter by up to 3x between rounds (20-90 ms); the minimum is the better number there.

Hessians (median s; program / axtb in brackets; accuracy = largest frequency error vs the
tight reference, cm⁻¹):

| | 1 thread | 8 threads | accuracy |
|---|---|---|---|
| GFN2, taxol (113 atoms): xtb 6.8.0 `--hess` / `--acc 0.001` / axtb analytic | 41.4 / 71.5 / 2.33 (18x / 31x) | 5.55 / 9.65 / 0.68 (8x / 14x) | 4.9 / 0.09 / 1e-5 |
| GFN2, C50H102 (152) | 45.5 / 74.3 / 2.93 (16x / 25x) | 6.64 / 10.6 / 0.89 (7x / 12x) | 2.4 / 0.13 / 2e-6 |
| GFN2, (H2O)100 (300) | axtb 35.4 | 57.7 / 94.6 / 7.48 (8x / 13x) | 14 / 0.35 / 1e-4 |
| g-xTB 2.0.1, taxol: program `--hess` / `--acc 0.001` / axtb analytic | 336 / 415 / 82.5 (4.1x / 5.0x) | 46.7 / 55.5 / 13.8 (3.4x / 4.0x) | 149 / 1.1 / 1e-5 |
| g-xTB 2.0.1, C50H102 | axtb 89.0 | 39.4 / 51.2 / 13.8 (2.8x / 3.7x) | 193 / 6.4 / 1e-6 |
| g-xTB v1.1, caffeine (24): program `-hess` / axtb numerical | axtb 7.40 | 379 / 1.00 (380x) | 3.9 / 0.15 |

Smaller molecules (water to C20H42): GFN2 12-21x (1 thread) and 5-17x (8 threads), g-xTB
2.0.1 4-5.5x and 2.5-4x (water 6-14x). Up to 62 atoms, axtb's analytic Hessians agree with
Richardson-extrapolated finite differences to ≤ 3e-10 Eh/bohr² (GFN2) and ≤ 3e-9 (g-xTB
2.0.1; ferrocene 6e-8, tblite's primitive screening). The programs' default `--hess` is off by
100-193 cm⁻¹ for g-xTB 2.0.1 (water 0.2) and up to 14 cm⁻¹ for GFN2-xTB.

Energy + gradient (axtb CLI vs the program, both whole processes): GFN2 1 thread 4-22x faster
up to 62 atoms (start-up), 1.6-1.9x at 113-152 atoms, 1.1-1.2x at 300-600 atoms; 8 threads
1.4-1.6x at 113-152 atoms and 0.7-1.0x at 300-600 (xtb 6.8.0 threads better there); tblite
0.8.0 is 3.7-11x slower than axtb from 113 atoms on. g-xTB 2.0.1, 1 thread: 3x faster at 21-24
atoms (water 18x), 1.8x at 62, 1.1x at 113-152, 0.7-0.9x at 300-600; 8 threads: the program is
faster from 62 atoms on (axtb 0.2-0.8x), because axtb's g-xTB SCF and gradient use ~1.5 of the
8 cores (only its dense linear algebra is threaded, and only from 200 basis functions). g-xTB
v1.1: the 1.1.0 program's gradient is numerical (taxol 244 s, 1 thread, vs axtb's 0.93 s; its
8 threads barely help); its energy alone takes 1.2-1.8x axtb's at 113-152 atoms.

The g-xTB parity features (bond orders on request, quadrupole, gap; commit 0812006 vs e4e441f)
add no measurable time to plain energy + gradient: new/base medians 1.006-1.023 (caffeine,
taxol, (H2O)100; GFN2 and g-xTB 2.0.1), within the null test's 0.994-1.029.

## GFN2-xTB and g-xTB 2.0.1 performance work (`results/final-9985wx-gxtb-perf/`, `results/laptop-perf/`)

Goal: axtb faster than the reference program at its defaults on 1 to 8 threads, on the laptop
(Intel i9-11900H, 8 cores / 16 threads) and on a cluster node (AMD Threadripper PRO 9985WX, 64
cores / 128 threads): GFN2-xTB against the official xtb 6.8.0 release binary (ifx + MKL),
g-xTB 2.0.1 against the g-xTB 2.0.1 build of xtb. The final timings above (axtb 0812006) left
g-xTB 2.0.1 behind its program from 62 atoms on at 8 threads (its SCF and gradient used ~1.5 of
the 8 cores) and GFN2-xTB behind xtb 6.8.0 at 300-600 atoms on 8 threads. Every phase is timed
in every result (`timings`: GFN2-xTB `classical`, `integrals`, `cholesky`, `scf: potentials`,
`scf: fock`, `scf: diagonalization`, `scf: density`, `scf: mulliken`, `scf: energy+mixing`,
`gradient`; g-xTB 2.0.1 `setup: ...`, `scf: ...`, `gradient: ...`).

### What changed, by xtb 6.8.0's techniques

xtb 6.8.0 sped up GFN2-xTB with grimme-lab/xtb #1465, #1466, #1464, #1248, #1253 and #1272.
Whether each applies to GFN2-xTB and g-xTB 2.0.1 in axtb, and what it gained (laptop; the
xtb and tblite conventions stay bit for bit where noted):

| technique (xtb 6.8.0) | GFN2-xTB in axtb | g-xTB 2.0.1 in axtb | gain (laptop), or why not |
|---|---|---|---|
| #1465: 1D Obara-Saika tables per primitive pair, all multipole components and their derivatives from one table | already | already (same kernels); extended to the q_eff coefficient-derivative sets of the gradient (one primitive pass for both contractions, c45c395) | g-xTB taxol gradient 0.32 -> 0.23 s |
| #1465: exact shell-pair screening (every primitive pair beyond the cutoff) | yes, bit-identical (f29c8a8) | yes (setup integrals, H0, ACP, gradient) | GFN2 (H2O)200 integrals 0.100 -> 0.058 s, H0 gradient 0.31 -> 0.22 s (1 thread); g-xTB: few pairs are screened (diffuse q-vSZP functions) |
| #1466: Cholesky factor of S once per geometry | already | already | - |
| #1466: L⁻¹ once, reduction to standard form and back-transformation by triangular matrix products | no L⁻¹: used in 799a34e, replaced by two triangular solves in 35e32d2, then (83b5bd8, default conventions, from 200 basis functions) by LAPACK's two-sided reduction `dsygst` (n³ flops, half of two triangular solves; no L⁻¹): a port to faer's kernels whose last per-block solve is deferred and run in parallel (same bits on any thread count), or with AXTB_LAPACK the library's `dsygst` and, for C = L⁻ᵀ U, its `dtrsm`. Below 200 basis functions and in the program conventions the two triangular solves stay, bit for bit | yes (same solver) | the reduction alone, n = 1200: 64 -> 36 ms (1 thread), 18-20 -> 9-12 ms (8 threads); GFN2 (H2O)200 `scf: diagonalization` against the two solves: MKL 2.73 -> 2.38 s (1 thread), 0.80 -> 0.59 s (8), faer 2.89 -> 2.63 s, 1.25 -> 1.19 s. The two solves themselves ran near peak (57 GFlop/s per core): the gain is the flops, plus the copies and per-call allocations dropped |
| #1466: `sytrd` + `stedc`, eigenvectors back-transformed for the occupied orbitals only (occupations < 1e-16 left out) | yes, default conventions (dec131f); further: the occupied eigenvectors of the tridiagonal matrix alone by inverse iteration (`crates/axtb/src/tridiag.rs`, e381039), divide and conquer for clustered spectra (8c491f2) | yes (both conventions) | per generalized eigensolve -25 % (1 thread) / -30 % (8), n = 300-1200; GFN2 taxol -13 % and (H2O)100 -15 % whole run (8 threads). The xtb and tblite conventions keep the full solver bit for bit |
| #1466: density matrix by `syrk` | yes, default conventions: lower-triangle products (4022fee) | yes, also the energy-weighted density (matrix products) | GFN2 (H2O)200 density 0.46 -> 0.27 s (1 thread); g-xTB taxol 95 -> 6 ms per SCF (was a scalar triple loop) |
| #1466: threaded `sytrd` (MKL) | yes, faer's parallel tridiagonalization and the library's `dsytrd` (cfb4056; results differ between thread counts by round-off) | yes | see "Threads in the eigensolver" |
| #1464: three-body dispersion gradient by pair sums: pair quantities precomputed, a SIMD loop over the third atom per pair, per-thread O(N) buffers | yes, default conventions: the same loop (dense windows of pair data, a vectorized loop over the third atoms with fixed lane sums; 3586cd6, 860e733) | already pair tables (one division per triple, pair derivatives summed over the third atom; 2a9bf42); not changed further: 3 % of the run | GFN2 (H2O)200 three-body 0.80 -> 0.19 s (1 thread), 0.16 -> 0.05 s (8); the xtb and tblite conventions keep dftd4's triple loop bit for bit |
| #1248: Fock matrix multipole potentials in one parallel pass; parallel Mulliken moments | already | yes (one tiled pass; row-parallel moments) | g-xTB taxol Fock potentials 35 -> 6 ms per SCF (8 threads) |
| #1248: parallel isotropic and multipole electrostatics | yes, bit-identical: each atom gathers its pair terms in the order of the sequential loop (6153218; with the D4 two-body energy and potential) | yes | GFN2 taxol -6 % (faer) / -2 % (MKL), (H2O)200 -2 % / -3 % (8 threads) |
| #1253: threaded gradient, faster Wiberg bond orders | already (gradient in fixed chunks; bond orders on request) | yes (gradient pair loops in fixed chunks) | g-xTB (H2O)100 gradient 2.5 -> 0.18 s (8 threads, with all gradient changes) |
| #1272: non-periodic three-body gradient | not applicable (no periodic code) | not applicable | - |

Beyond these: x86-64 Linux builds target x86-64-v3 (AVX2, FMA; b29bb8f), so axtb's own loops
vectorize 4-wide with bit-identical results (Rust never fuses a multiply and an add on its own):
GFN2-xTB taxol 0.189 -> 0.160 s, g-xTB 2.0.1 taxol 0.460 -> 0.344 s (1 thread). g-xTB 2.0.1 only:
the ACP as matrix products, the exchange restructured (dense kernels once per geometry, four
matrix products per Fock matrix), DIIS with cached residual products, and every loop of its
setup, SCF and gradient on the calculator's threads in fixed chunks summed in order. With a
run-time LAPACK library, matrices below 200 rows stay on faer (the library's start-up cost more
than it saved, 0329853).

### Threads in the eigensolver

The SCF eigensolver's tridiagonalization (faer's, above ~313 rows; `dsytrd` with a run-time LAPACK
library) sums one partial matrix-vector product per thread. Between 54b693b and cfb4056 axtb kept
it sequential so that results were bit-identical on any thread count; since cfb4056 it is
threaded again, in faer and in the library, for speed. axtb's own reductions keep fixed
chunks summed in order and MKL its strict CNR mode, so results repeat bit for bit on a given
thread count and differ between thread counts by round-off, as xtb's do: taxol and (H2O)200,
GFN2-xTB and g-xTB 2.0.1, 1 vs 2, 4 and 8 threads, faer, MKL, OpenBLAS and AOCL: energies
≤ 5.7e-13 Eh, gradients ≤ 7.6e-15 Eh/bohr, the same iterations; three runs per thread count
identical. The gates run on one thread (the reference matrix's 8-thread cases are below the
threaded tridiagonalization's size) and did not change.

Gain on the laptop (whole CLI runs, sequential / threaded tridiagonalization, medians of 5
interleaved rounds; 4 / 8 threads):

| | faer | MKL |
|---|---|---|
| GFN2-xTB (H2O)200 | 1.24x / 1.19x | 1.47x / 1.62x |
| GFN2-xTB (H2O)100, C100H202 | 0.99-1.06x | 1.15-1.24x |
| GFN2-xTB taxol | 1.01x / 1.01x | 1.06x / 1.02x |
| g-xTB 2.0.1 (H2O)200 | 1.08x / 1.13x | 1.20x / 1.28x |
| g-xTB 2.0.1 (H2O)100, C100H202 | 0.96-1.06x | 1.05-1.18x |
| g-xTB 2.0.1 taxol | 0.97x / 1.04x | 1.05x / 1.00x |

### Final timings

`final_timings.py` as above, energy + gradient only, 5 rounds per molecule with the order
rotated, medians; every run records its command line, binary (path and sha256 prefix), process
id, library variables and axtb request, and each task checks that every column ran one binary
with one setting, that programs and axtb were different binaries and every run its own process
(`meta.harness_check`: all true in every task here). axtb (CLI, `--json`) runs with faer (its
default) and again with a run-time LAPACK library (`AXTB_LAPACK`, `libs` of the task), in the
default conventions and, for GFN2-xTB, in xtb conventions (`conventions="xtb"`, which reproduce
xtb 6.8.0 bit for bit and keep its full eigensolver and dftd4's three-body loop).

- **Laptop** (`results/laptop-perf/`): i9-11900H, faer and MKL 2025.3 (conda), axtb 6332d81
  (`summary.json` records 106cac8 for 7 of the 8 tasks; that commit changed only tests, docs and
  results, so the engine was 6332d81's), 2026-11-07 00:23-00:45 UTC, nothing else running: each task started at a 1-minute load below
  1.5 (it rose to 5.6 during the 8-thread tasks from these runs themselves).
- **Node** (`results/final-9985wx-gxtb-perf/`): one pod per thread count with twice the CPUs
  (image `axtb-timing:2026-11-06i`, axtb 6332d81, MKL 2026.2 from PyPI, AOCL 5.3 built from
  source), tasks started 2026-11-07 00:14-00:21 UTC; the node was shared: other users' work kept a median of
  18-75 of its 128 hardware threads busy (`other_busy_threads`).

Program / axtb, median seconds of whole processes (speed-up):

GFN2-xTB, axtb default (faer), laptop:

| molecule | 1 thr | 2 thr | 4 thr | 8 thr |
|---|---|---|---|---|
| water | 0.0158 / 0.002 (7.80x) | 0.0142 / 0.0021 (6.77x) | 0.0137 / 0.0022 (6.33x) | 0.0151 / 0.0023 (6.59x) |
| ferrocene | 0.0207 / 0.0066 (3.13x) | 0.0188 / 0.0072 (2.60x) | 0.018 / 0.0067 (2.69x) | 0.0197 / 0.0066 (2.96x) |
| caffeine | 0.0225 / 0.008 (2.82x) | 0.0221 / 0.009 (2.45x) | 0.0186 / 0.0083 (2.24x) | 0.023 / 0.0083 (2.76x) |
| C20H42 | 0.0335 / 0.0189 (1.78x) | 0.0297 / 0.019 (1.56x) | 0.0268 / 0.0184 (1.46x) | 0.0336 / 0.0187 (1.80x) |
| taxol | 0.155 / 0.138 (1.12x) | 0.109 / 0.101 (1.08x) | 0.0867 / 0.0851 (1.02x) | 0.0981 / 0.088 (1.11x) |
| C50H102 | 0.116 / 0.0926 (1.25x) | 0.0898 / 0.0753 (1.19x) | 0.0752 / 0.063 (1.19x) | 0.0781 / 0.066 (1.18x) |
| (H2O)100 | 0.581 / 0.575 (1.01x) | 0.371 / 0.36 (1.03x) | 0.261 / 0.313 (0.83x) | 0.251 / 0.293 (0.86x) |
| C100H202 | 0.529 / 0.472 (1.12x) | 0.363 / 0.32 (1.13x) | 0.268 / 0.286 (0.94x) | 0.254 / 0.275 (0.92x) |
| (H2O)200 | 3.66 / 3.69 (0.99x) | 2.24 / 2.25 (1.00x) | 1.42 / 1.8 (0.79x) | 1.22 / 1.64 (0.75x) |

GFN2-xTB, laptop, the other axtb settings (speed-ups on 1 / 2 / 4 / 8 threads):

| molecule | [MKL] | xtb conv. | xtb conv. [MKL] |
|---|---|---|---|
| water | 8.60 / 7.22 / 6.77 / 7.38 | 12.19 / 8.57 / 8.50 / 8.97 | 13.12 / 9.37 / 9.00 / 9.45 |
| ferrocene | 3.21 / 2.78 / 2.76 / 3.00 | 3.23 / 2.78 / 2.71 / 3.02 | 3.28 / 2.81 / 2.79 / 3.09 |
| caffeine | 2.93 / 2.50 / 2.25 / 2.62 | 2.78 / 2.43 / 2.16 / 2.59 | 2.82 / 2.50 / 2.22 / 2.59 |
| C20H42 | 1.75 / 1.56 / 1.46 / 1.75 | 1.59 / 1.42 / 1.31 / 1.61 | 1.61 / 1.43 / 1.33 / 1.60 |
| taxol | 1.19 / 1.14 / 1.05 / 1.06 | 0.89 / 0.86 / 0.81 / 0.88 | 1.03 / 1.01 / 0.94 / 0.91 |
| C50H102 | 1.27 / 1.22 / 1.16 / 1.07 | 1.01 / 1.00 / 0.97 / 0.95 | 1.14 / 1.15 / 1.08 / 0.94 |
| (H2O)100 | 1.15 / 1.14 / 1.04 / 1.07 | 0.84 / 0.82 / 0.68 / 0.69 | 0.92 / 0.92 / 0.92 / 0.88 |
| C100H202 | 1.20 / 1.29 / 1.23 / 1.20 | 0.90 / 0.92 / 0.76 / 0.73 | 1.01 / 1.08 / 1.07 / 0.97 |
| (H2O)200 | 1.02 / 1.08 / 1.06 / 1.02 | 0.82 / 0.84 / 0.67 / 0.64 | 0.87 / 0.92 / 0.90 / 0.84 |

g-xTB 2.0.1, axtb default (faer), laptop (with MKL: 1.77-3.60x):

| molecule | 1 thr | 2 thr | 4 thr | 8 thr |
|---|---|---|---|---|
| water | 0.0069 / 0.0022 (3.08x) | 0.0067 / 0.0027 (2.45x) | 0.0065 / 0.0028 (2.30x) | 0.0072 / 0.0037 (1.94x) |
| ferrocene | 0.049 / 0.0138 (3.55x) | 0.0355 / 0.0112 (3.19x) | 0.0252 / 0.0097 (2.59x) | 0.0214 / 0.0096 (2.23x) |
| caffeine | 0.0571 / 0.0168 (3.41x) | 0.0391 / 0.0136 (2.88x) | 0.0284 / 0.0119 (2.39x) | 0.0255 / 0.0112 (2.27x) |
| C20H42 | 0.153 / 0.0477 (3.21x) | 0.0982 / 0.0349 (2.81x) | 0.0631 / 0.0287 (2.20x) | 0.0579 / 0.0263 (2.20x) |
| taxol | 0.879 / 0.322 (2.73x) | 0.567 / 0.192 (2.95x) | 0.368 / 0.133 (2.76x) | 0.264 / 0.118 (2.23x) |
| C50H102 | 0.637 / 0.242 (2.63x) | 0.407 / 0.162 (2.52x) | 0.255 / 0.112 (2.28x) | 0.225 / 0.105 (2.14x) |
| (H2O)100 | 3.54 / 1.54 (2.31x) | 2.22 / 0.925 (2.41x) | 1.4 / 0.615 (2.28x) | 0.987 / 0.507 (1.95x) |
| C100H202 | 2.42 / 1.17 (2.08x) | 1.48 / 0.744 (1.99x) | 0.961 / 0.541 (1.77x) | 0.82 / 0.479 (1.71x) |
| (H2O)200 | 16.5 / 9.78 (1.69x) | 9.92 / 5.9 (1.68x) | 6.44 / 3.91 (1.64x) | 4.63 / 3.27 (1.42x) |

GFN2-xTB, axtb default (faer), node:

| molecule | 1 thr | 2 thr | 4 thr | 8 thr |
|---|---|---|---|---|
| water | 0.103 / 0.0027 (38.12x) | 0.161 / 0.0049 (32.69x) | 0.0837 / 0.0031 (26.94x) | 0.0947 / 0.0033 (28.48x) |
| ferrocene | 0.107 / 0.0063 (17.02x) | 0.172 / 0.0107 (16.15x) | 0.0868 / 0.0069 (12.56x) | 0.0838 / 0.007 (11.92x) |
| caffeine | 0.113 / 0.0073 (15.57x) | 0.169 / 0.0121 (13.96x) | 0.0839 / 0.008 (10.54x) | 0.084 / 0.008 (10.45x) |
| C20H42 | 0.133 / 0.0158 (8.44x) | 0.187 / 0.0245 (7.64x) | 0.0927 / 0.0153 (6.05x) | 0.101 / 0.0159 (6.38x) |
| taxol | 0.256 / 0.109 (2.35x) | 0.232 / 0.0907 (2.56x) | 0.166 / 0.0777 (2.14x) | 0.165 / 0.078 (2.11x) |
| C50H102 | 0.359 / 0.112 (3.19x) | 0.288 / 0.0916 (3.14x) | 0.116 / 0.0616 (1.88x) | 0.142 / 0.0541 (2.63x) |
| (H2O)100 | 0.789 / 0.394 (2.00x) | 0.619 / 0.352 (1.76x) | 0.323 / 0.257 (1.26x) | 0.464 / 0.278 (1.67x) |
| C100H202 | 0.599 / 0.346 (1.73x) | 0.462 / 0.257 (1.79x) | 0.301 / 0.225 (1.34x) | 0.289 / 0.22 (1.31x) |
| (H2O)200 | 3.74 / 2.4 (1.56x) | 2.6 / 1.78 (1.46x) | 1.64 / 1.25 (1.31x) | 1.14 / 1.09 (1.05x) |

GFN2-xTB, node, the other axtb settings (speed-ups on 1 / 2 / 4 / 8 threads):

| molecule | [AOCL] | [MKL] | xtb conv. | xtb conv. [AOCL] | xtb conv. [MKL] |
|---|---|---|---|---|---|
| water | 49.77 / 34.58 / 30.33 / 34.38 | 43.64 / 34.70 / 29.78 / 36.80 | 90.74 / 64.19 / 47.74 / 51.76 | 96.03 / 76.09 / 49.51 / 64.41 | 85.83 / 71.57 / 47.90 / 64.60 |
| ferrocene | 18.40 / 15.93 / 13.75 / 12.66 | 19.49 / 17.35 / 13.95 / 12.09 | 19.32 / 15.88 / 13.02 / 12.40 | 19.74 / 16.21 / 13.69 / 12.68 | 18.42 / 17.18 / 13.13 / 12.23 |
| caffeine | 17.08 / 19.25 / 10.68 / 10.80 | 17.29 / 13.79 / 10.91 / 11.21 | 15.94 / 13.68 / 10.64 / 9.97 | 16.42 / 19.23 / 10.71 / 11.07 | 16.18 / 16.77 / 10.78 / 11.12 |
| C20H42 | 8.47 / 10.15 / 6.45 / 6.55 | 8.63 / 7.91 / 6.41 / 6.72 | 7.56 / 8.38 / 5.90 / 5.89 | 7.61 / 6.89 / 5.98 / 5.85 | 7.58 / 7.77 / 5.79 / 5.96 |
| taxol | 2.72 / 2.81 / 1.81 / 1.74 | 1.37 / 1.33 / 1.29 / 1.23 | 2.14 / 1.92 / 1.77 / 1.57 | 2.44 / 2.40 / 1.60 / 1.40 | 1.27 / 1.32 / 1.14 / 1.13 |
| C50H102 | 3.26 / 3.07 / 1.52 / 2.10 | 1.65 / 1.48 / 1.16 / 1.32 | 3.53 / 2.66 / 1.53 / 1.99 | 3.30 / 2.88 / 1.41 / 1.71 | 1.47 / 1.49 / 1.02 / 1.24 |
| (H2O)100 | 2.01 / 2.20 / 1.27 / 1.48 | 1.40 / 1.17 / 1.12 / 1.30 | 1.81 / 1.52 / 1.01 / 1.47 | 1.59 / 1.82 / 1.09 / 1.39 | 1.09 / 1.08 / 0.97 / 1.12 |
| C100H202 | 2.15 / 1.82 / 1.31 / 1.28 | 1.41 / 1.32 / 1.19 / 1.19 | 1.59 / 1.61 / 1.09 / 1.03 | 1.86 / 1.53 / 1.12 / 1.09 | 1.25 / 1.09 / 1.09 / 1.06 |
| (H2O)200 | 1.81 / 1.57 / 1.29 / 1.04 | 1.31 / 1.12 / 1.18 / 1.09 | 1.40 / 1.31 / 1.14 / 0.90 | 1.53 / 1.35 / 1.17 / 0.94 | 1.13 / 0.99 / 1.11 / 0.99 |

g-xTB 2.0.1, axtb default (faer), node:

| molecule | 1 thr | 2 thr | 4 thr | 8 thr |
|---|---|---|---|---|
| water | 0.0773 / 0.0027 (28.37x) | 0.11 / 0.0041 (26.67x) | 0.0359 / 0.0043 (8.27x) | 0.0879 / 0.0047 (18.67x) |
| ferrocene | 0.109 / 0.0109 (9.98x) | 0.135 / 0.0117 (11.54x) | 0.0624 / 0.011 (5.69x) | 0.109 / 0.0097 (11.32x) |
| caffeine | 0.118 / 0.0124 (9.46x) | 0.124 / 0.0133 (9.28x) | 0.0635 / 0.0125 (5.09x) | 0.104 / 0.0101 (10.30x) |
| C20H42 | 0.235 / 0.0364 (6.45x) | 0.157 / 0.028 (5.61x) | 0.1 / 0.0244 (4.12x) | 0.127 / 0.0232 (5.48x) |
| taxol | 0.831 / 0.224 (3.71x) | 0.547 / 0.146 (3.75x) | 0.338 / 0.107 (3.16x) | 0.314 / 0.102 (3.08x) |
| C50H102 | 0.648 / 0.173 (3.76x) | 0.416 / 0.109 (3.82x) | 0.274 / 0.0981 (2.80x) | 0.281 / 0.0808 (3.48x) |
| (H2O)100 | 3.26 / 1.16 (2.81x) | 1.93 / 0.608 (3.17x) | 1.23 / 0.446 (2.76x) | 0.921 / 0.376 (2.45x) |
| C100H202 | 2.55 / 0.758 (3.37x) | 1.31 / 0.493 (2.66x) | 0.878 / 0.428 (2.05x) | 0.769 / 0.343 (2.24x) |
| (H2O)200 | 13.8 / 5.72 (2.41x) | 8.43 / 3.37 (2.50x) | 4.94 / 2.56 (1.93x) | 3.86 / 1.9 (2.03x) |

g-xTB 2.0.1, node, the other settings (speed-ups on 1 / 2 / 4 / 8 threads):

| molecule | [AOCL] | [MKL] |
|---|---|---|
| water | 27.43 / 28.77 / 8.54 / 23.71 | 32.52 / 26.61 / 8.55 / 20.27 |
| ferrocene | 10.69 / 11.57 / 6.06 / 12.57 | 10.38 / 11.72 / 5.78 / 12.68 |
| caffeine | 9.41 / 11.39 / 5.48 / 10.56 | 9.58 / 9.51 / 5.34 / 8.86 |
| C20H42 | 6.37 / 6.31 / 4.01 / 5.40 | 6.45 / 5.58 / 4.05 / 5.59 |
| taxol | 3.66 / 3.75 / 2.60 / 2.59 | 2.79 / 2.75 / 2.17 / 1.89 |
| C50H102 | 3.99 / 3.34 / 2.60 / 2.92 | 2.73 / 2.47 / 2.01 / 1.96 |
| (H2O)100 | 3.22 / 3.04 / 2.72 / 2.31 | 2.67 / 2.82 / 2.50 / 2.19 |
| C100H202 | 3.79 / 2.80 / 2.28 / 2.07 | 3.11 / 2.49 / 2.15 / 2.04 |
| (H2O)200 | 2.85 / 2.14 / 2.20 / 1.99 | 2.29 / 2.17 / 2.12 / 1.96 |

Before and after (node, the same runs, axtb CLI 7bb7c16 against 6332d81, median s, 1 / 8
threads): GFN2-xTB taxol 0.148 -> 0.098 / 0.120 -> 0.070, (H2O)100 0.549 -> 0.373 / 0.350 ->
0.237, (H2O)200 3.50 -> 2.35 / 1.75 -> 1.22; g-xTB 2.0.1 taxol 0.701 -> 0.210 / 0.794 -> 0.118,
(H2O)100 3.28 -> 0.949 / 3.99 -> 0.407, (H2O)200 19.8 -> 5.59 / 25.6 -> 2.17
(`before-after-eg-t*.json`).

**Spot checks of the harness** (`spot_check.sh` on the inputs of `spot_check_inputs.py`: each
binary run by hand with bash's `time`, alternating, three rounds; `spot_checks.json`), wall s,
program / axtb (faer), hand runs against the harness medians above:

| cell | laptop, by hand | laptop, harness | node, by hand | node, harness |
|---|---|---|---|---|
| GFN2-xTB (H2O)100, 1 thread | 0.570-0.589 / 0.545-0.583 | 0.581 / 0.575 | 0.730-0.936 / 0.400-0.570 | 0.789 / 0.394 |
| GFN2-xTB (H2O)200, 8 threads | 1.082-1.136 / 1.551-1.570 | 1.224 / 1.637 | 1.596-1.786 / 1.319-1.406 | 1.143 / 1.085 |
| g-xTB 2.0.1 taxol, 4 threads | 0.361-0.377 / 0.132-0.141 | 0.368 / 0.133 | 0.467-0.630 / 0.112-0.173 | 0.338 / 0.107 |

The binaries were the ones the harness ran (sha256 prefixes in `spot_checks.json` and in the
tasks' `harness_check`). The node's hand runs shared the node with the attribution run below and
are slower for both programs.

### Where the time goes where axtb is behind

`attribution.py` (best of 3, settings interleaved; `attribution.jsonl` in both result
directories, table by `attribution_table.py`): the program's own breakdown (xtb `--verbose`:
setup = classical + integrals, SCC iterations, gradient) next to axtb's phases (setup =
classical + integrals, SCF incl. Cholesky [diagonalization], gradient), wall s:

| case | xtb 6.8.0: wall (setup / SCC / gradient) | axtb faer: wall (setup / SCF [diag.] / gradient) | axtb MKL | verdict |
|---|---|---|---|---|
| laptop (H2O)200, 1 thread | 3.72 (0.10 / 2.97 / 0.39) | 3.70 (0.30 / 3.29 [2.73] / 0.10) | 3.48 (0.28 / 3.08 [2.56] / 0.10) | math library: faer's eigensolver (the same diagonalizations with MKL: 0.17 s less) |
| laptop (H2O)200, 4 threads | 1.44 (0.07 / 0.99 / 0.18) | 1.71 (0.15 / 1.48 [1.20] / 0.06) | 1.32 (0.15 / 1.09 [0.84] / 0.06) | math library: faer's diagonalization alone takes longer than xtb's whole SCC |
| laptop (H2O)200, 8 threads | 1.11 (0.07 / 0.67 / 0.17) | 1.59 (0.13 / 1.39 [1.16] / 0.05) | 1.19 (0.14 / 0.97 [0.73] / 0.06) | math library (as above); with MKL ahead by 2 % in the medians, the diagonalization through `libmkl_rt` (0.73 s) is still slower than xtb's whole SCC (0.67 s) |
| laptop (H2O)100, 4 / 8 threads | 0.256 / 0.240 (SCC 0.146 / 0.120) | 0.321 / 0.304 (diag. 0.210 / 0.196) | 0.236 / 0.226 (diag. 0.125 / 0.114) | math library |
| laptop C100H202, 4 / 8 threads | 0.259 / 0.241 (SCC 0.151 / 0.125) | 0.287 / 0.267 (diag. 0.198 / 0.182) | 0.216 / 0.201 (diag. 0.121 / 0.108) | math library |
| node (H2O)200, 8 threads (closest) | 1.18 (0.05 / 0.78 / 0.11) | 1.11 (0.08 / 0.96 [0.85] / 0.04) | 1.03 (0.08 / 0.89 [0.76] / 0.04) | ahead; the margin is the diagonalization (AOCL 1.12, diag. 0.86) |

xtb 6.8.0 counts its three-body dispersion in its gradient, axtb in its setup (`classical`);
the sums setup + gradient are 0.49 s (xtb) and 0.40 s (axtb) for (H2O)200 on one laptop thread.
The xtb conventions (not a default) are behind on the laptop from taxol on and on the node for
(H2O)200 on 8 threads: they keep the full eigensolver (every eigenvector by divide and conquer,
full density products) and dftd4's per-triple three-body loop, so that they reproduce xtb 6.8.0
and tblite bit for bit.

### What remains

- **faer on the Intel laptop** (the default): behind xtb 6.8.0 for the water clusters and
  C100H202 on 4 and 8 threads (0.75-0.94x) and level for (H2O)200 on 1 and 2 threads
  (0.99-1.00x). The time is in faer's tridiagonalization and divide and conquer, which MKL runs
  faster on this CPU (AVX-512): with `AXTB_LAPACK` naming MKL, axtb is faster than xtb 6.8.0 in
  every GFN2-xTB cell on the laptop (1.02-8.6x). faer 0.24 with its AVX-512 kernels (Rust
  ≥ 1.89) was 8-10 % faster than faer 0.22 at n = 1000-3000 on one thread in `bench/linalg`
  and has not been tried in axtb.
- **The node**: faer, the default, is ahead everywhere (GFN2-xTB 1.05-38x, g-xTB 2.0.1
  1.93-28x). AOCL is faster than faer on 1 and 2 threads for the largest systems ((H2O)200
  2.07 vs 2.40 s on 1 thread), MKL is slower than faer except (H2O)200 on 8 threads (its
  first call needed 0.1 s on this 128-CPU machine until 0329853 turned Intel OpenMP's
  affinity off).
- **Three-body cutoff**: axtb's default conventions sum the three-body dispersion within
  40 bohr, xtb 6.8.0 within tblite's 25 bohr: 34.9 M against 10.6 M triples for (H2O)200
  (E3 3.5e-4 Eh apart). With xtb 6.8.0's loop, axtb takes 0.19 s on one laptop thread at
  40 bohr and 0.12 s at 25 bohr. The cutoff is a choice of the method, not of the code, and
  is unchanged.
- **Sequential parts on 8 threads**: Broyden mixing (~3 ms per iteration for (H2O)200), the
  three-body pair setup (~20 ms), the EEQ LU solve; each ≤ 3 % of the run.
- **g-xTB 2.0.1** is 1.42-3.6x (laptop) and 1.93-28x (node) faster than its program; its revD4
  three-body term (3 % of a run) was not changed.

## Release procedure

Every number in the README, `docs/validation.md`, this file and the report data
(`bench/results/report/`) is to come from runs of one commit, the release commit, not from the
development runs above:

1. Record the commit (`git rev-parse HEAD`). `scripts/check.sh fast` passes on it.
2. Build the CLI and the wheel from it (README, "Install"), on the laptop and in fresh cluster
   images (`bench/cluster/make_context.sh`, `bench/cluster/make_timing_context.sh`).
3. Set the reference programs (`tests/programs.py`) and run `scripts/check.sh programs`: every
   gate passes, or is skipped for a reason recorded with the release.
4. Rerun the result files that the docs quote, each script writing its default output in
   `bench/results/` (`docs/validation.md`, "Running the scripts by hand"): the reference matrix
   with `--fresh` against xtb 6.8.0, the g-xTB build and the 6.7.1 release; `program_differences.py`,
   `xtb680_changes.py`, `xtb_hess_acc.py`, `vibrational_check.py` and `gfnff_reproduction.py` per
   xtb version; `gxtb_reproduction.py`, `gxtb2_reproduction.py`, `gxtb_etemp_reproduction.py`,
   `gxtb_elements.py`, `gxtb_energy_differences.py`, `gxtb_parity_features.py`,
   `solvation_composite.py`, `molden_check.py` and `scf_hard_set.py`.
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
