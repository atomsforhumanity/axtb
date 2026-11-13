# axtb — extended tight binding, rebuilt

A from-scratch Rust implementation of the GFN-xTB family of semiempirical methods
(GFN2-xTB; g-xTB v1.1 and 2.0.1) and of the GFN-FF force field. In each reference program's
conventions it reproduces that program's numbers to the last digits (xtb 6.8.0 and 6.7.1,
tblite 0.8.0, the g-xTB 1.1.0 and 2.0.1 programs); its default conventions correct the bugs
found on the way. It runs faster than the programs and is safe to run many at once: every
calculation runs on exactly the threads it is given. axtb is a Rust library, a command-line
program (JSON in and out), a Python package and a codec for qccompute.

License: LGPL-3.0-or-later, every crate, the CLI and the Python package ([License](#license)).

## Status

| method or feature | validated against |
|---|---|
| GFN2-xTB energies and analytic gradients (self-consistent D4, multipole electrostatics) | xtb 6.8.0 and 6.7.1, tblite 0.8.0 |
| GFN2-xTB Hessians: analytic (coupled-perturbed SCC response; the default), response and numerical | finite differences, `xtb --hess` |
| ALPB/GBSA implicit solvation for GFN2-xTB (25 ALPB / 14 GBSA solvents, 3 reference states) | xtb 6.8.0 and 6.7.1, tblite 0.8.0 |
| g-xTB v1.1 (public parameters): energies, analytic gradients, semi-numerical Hessians, finite electronic temperature | the g-xTB 1.1.0 program |
| g-xTB 2.0.1: energies, analytic gradients, analytic (default), response and numerical Hessians, finite electronic temperature | the g-xTB 2.0.1 program |
| g-xTB in implicit solvent: g-xTB plus GFN2-xTB's ALPB/GBSA solvation energy, as one calculation | the three xtb program runs it replaces |
| GFN-FF: energies, analytic gradients, analytic (default) and numerical Hessians with dipole derivatives | xtb 6.8.0 and 6.7.1 (`--gfnff`) |
| vibrational analysis of any Hessian: frequencies, normal modes, IR intensities, quasi-RRHO thermochemistry | `xtb --hess`, `xtb thermo` |
| Molden files of the converged orbitals | the programs' own Molden files |
| restarts from the previous geometry's wavefunction; Mayer–Wiberg bond orders, orbital levels, HOMO–LUMO gap, molecular quadrupole | the programs' restarts, `wbo` files and printouts |

Every feature is available from Rust, Python and the qccodec codec, and all but restarts from
the CLI (one geometry per call).
[docs/validation.md](docs/validation.md) has the agreement with each program, the test sets
and the timings; `scripts/check.sh` reruns the checks (see [Checks](#checks)). Not
implemented: periodic systems, GFN1-xTB, implicit solvation with GFN-FF, PDB residue charges.
A GPU eigensolver is not included (an experimental CUDA eigensolver for GFN2-xTB was measured
during development; the report's GPU rows come from it).

## Install

axtb is built with Rust 1.89 or newer ([rustup](https://rustup.rs)) and needs nothing outside
Cargo: no Fortran compiler, no BLAS or LAPACK. It is developed and tested on x86-64 Linux;
other platforms are untested.

**Released builds.** The Python package from PyPI, with wheels for x86-64 Linux
(manylinux_2_28: glibc 2.28 or newer; Python 3.10 or newer, one abi3 wheel for all versions);
elsewhere pip builds it from the source distribution, which needs Rust:

```bash
pip install axtb
pip install "axtb[qc]"                     # with qccodec and qcdata, for the codec
```

The CLI from crates.io, or as the stripped `axtb` binary attached to each GitHub release:

```bash
cargo install axtb-cli                     # into ~/.cargo/bin; see "CPU" for the target
```

**From a checkout.** The CLI:

```bash
cargo build --release -p axtb-cli          # target/release/axtb
cargo build --profile dist -p axtb-cli     # target/dist/axtb: the same without debug info, stripped
cargo install --path crates/axtb-cli       # or into ~/.cargo/bin
```

The release profile keeps line tables (`debug = 1`) for profiling, which makes the binary
several times larger; the `dist` profile (root `Cargo.toml`) is the release profile without
them and stripped, as the release binaries are built. The two give the same numbers.

The Python package (Python 3.10 or newer, numpy) from a wheel built with
[maturin](https://www.maturin.rs) (stripped, `crates/axtb-py/pyproject.toml`):

```bash
pip install maturin
maturin build --release --compatibility linux -m crates/axtb-py/Cargo.toml -o dist
pip install dist/axtb-*.whl
pip install qccompute                      # for the qccodec codec (optional)
```

`pip install ./crates/axtb-py` builds and installs it in one step. The codec registers itself
with qccodec when both are installed. `--compatibility linux` builds a wheel for this machine;
without it maturin builds the release's manylinux_2_28 wheel, which on a system newer than
glibc 2.28 needs the manylinux_2_28 container or `--zig` (`pip install ziglang`).

**CPU.** Builds from this repository on x86-64 Linux, and the released wheels and binaries,
target x86-64-v3 (`.cargo/config.toml`: AVX2, FMA, BMI2 and the rest of the level): Intel CPUs
since Haswell (2013) and AMD CPUs since Excavator (2015), but not many low-end Intel Atom,
Celeron and Pentium N-series CPUs. Results are bit-identical to a baseline x86-64 build, which
is slower: one thread on the 64-core node, energy + gradient of caffeine, taxol and (H2O)200,
GFN2-xTB 27-52 %, g-xTB 2.0.1 34-68 %, GFN-FF 1-10 % (axtb 0.1.0, the CLI built with
`RUSTFLAGS="-C target-cpu=x86-64"`; `bench/results/final-9985wx/baseline_build.json`). On a CPU
without these instructions the CLI stops with an error before it runs any of them, and `import
axtb` raises `ImportError` before it loads the extension module. The Python check reads the CPU
flags in `/proc/cpuinfo`; where that file is missing or lists no flags (other operating
systems, some emulators; user-mode QEMU shows the host's) it cannot check. For any x86-64 CPU,
build with

```bash
RUSTFLAGS="-C target-cpu=x86-64" cargo build --release -p axtb-cli
RUSTFLAGS="-C target-cpu=x86-64" maturin build --release --compatibility linux -m crates/axtb-py/Cargo.toml -o dist
```

(an explicit `RUSTFLAGS` replaces the setting in `.cargo/config.toml`). `cargo install
axtb-cli` from crates.io does not see `.cargo/config.toml` and builds for the baseline x86-64
unless you ask for more: `RUSTFLAGS="-C target-cpu=x86-64-v3" cargo install axtb-cli`.

**As a Rust library.** Depend on the crates from crates.io: `axtb` (GFN2-xTB), `axtb-gxtb`
(g-xTB), `axtb-gfnff` (GFN-FF), or `axtb-driver` (the request/response layer of the front
ends); `axtb-d4` and `axtb-solvation` come with `axtb`. The workspace patches two dependencies
of faer that read every CPU's cache files when the first matrix product runs
(`third_party/README.md`). Cargo applies that patch only to builds of this workspace; add the
same `[patch.crates-io]` entries to your workspace if you start many processes on many-CPU
machines.

## Use

The CLI, the Python module and the qccodec codec share one request/response layer
(`crates/driver`, `axtb_driver`): the same settings under the same names, the same checks and
the same response. `None` (or an absent setting) means the method's default. Names of methods,
conventions and settings ignore case and anything but letters and digits (`"GFN2-xTB"`,
`"g-xTB 2.0"`, `"XTB 6.7.1"`); responses carry the canonical names (`axtb.canonical_method`).
A setting that does not apply to the method (a GFN-FF option for GFN2-xTB, an SCF option for
GFN-FF, a solvent's reference state without a solvent, a Hessian's step without a Hessian or
with an analytic one, ...) is an error, never ignored. `threads` must be at least 1.

Python (see [Install](#install); type stubs included):

```python
import numpy as np, axtb
r = axtb.singlepoint(numbers, positions_bohr, charge=0, method="gfn2", gradient=True)  # any array-like (N, 3)
calc = axtb.Calculator("gfn2", numbers, positions_bohr, threads=8)   # reuses setup and wavefunction
r = calc.singlepoint(new_positions, gradient=True)                  # the gradient only when asked for
h = calc.hessian(positions_bohr)                 # the single point there, the Hessian and its analysis
h["hessian"], h["hessian_method"], h["frequencies"]                 # (3N, 3N) Eh/bohr², "analytic", cm⁻¹
h = calc.hessian(positions_bohr, method="numerical", step=0.005)    # or "response"
# r also holds the converged moments (shell_charges, atomic_dipoles, atomic_quadrupoles);
# calc.set_guess(...) starts the next SCF from them (or from another program's moments)
r["guess"], r["homo_lumo_gap"], r["quadrupole"], r["orbital_energies_alpha"]  # start, Eh, e·bohr²
r = calc.singlepoint(new_positions, bond_orders=True); r["bond_orders"]       # (N, N) Mayer-Wiberg
g = axtb.Calculator("gxtb-v2.0.1", numbers, positions_bohr)  # g-xTB too starts from the previous
g.singlepoint(p1); g.singlepoint(p2)["guess"]                 # geometry's state: "previous"
state = g.guess_state(); other.set_guess_state(state)         # move it to another calculator
text = axtb.singlepoint(numbers, positions_bohr, method="gxtb-v2.0.1", molden=True)["molden"]
open("mol.molden", "w").write(text)                                 # orbitals for Jmol, Avogadro, IboView
h = calc.hessian(positions_bohr, temperature=[298.15, 350.0])       # with the vibrational analysis
h["frequencies"], h["n_imaginary"], h["ir_intensities"]             # cm⁻¹ (imaginary < 0), km/mol
h["thermo"]["g_rrho"], h["thermo"]["free_energy"]                   # qRRHO at the last temperature
v = axtb.vibrational_analysis(numbers, positions_bohr, hessian, conventions="xtb")  # any Hessian
try:
    axtb.singlepoint(numbers, positions_bohr, max_iter=50)
except axtb.ScfNotConvergedError as e:                              # a RuntimeError
    e.iterations, e.energy_change, e.residual
```

Results are dicts with the keys of the CLI's JSON response (below), arrays as numpy arrays.
`Calculator.singlepoint` and `axtb.singlepoint` compute the gradient only with
`gradient=True`. `Calculator.hessian` returns what the CLI's `--hess` does: the single point at
the geometry (g-xTB's SCF from its own start, as without the Hessian), the Hessian, its dipole
derivatives and, unless `vibrations=False`, the vibrational analysis. Errors:
`axtb.InputError` (a `ValueError`) for invalid input, elements without parameters and
structures GFN-FF cannot set up; `axtb.ScfNotConvergedError` (a `RuntimeError`) with
`iterations`, `energy_change` and `residual`; `RuntimeError` for numerical failures (and, as
a bug report, for an internal panic); each carries `kind` as the CLI's JSON error does. The
calculator's construction and every calculation release the GIL. Python names that differ from
the request's or Rust's: `Calculator.guess_state()`, `set_guess_state(state)` and
`reset_guess()` are `axtb_driver::Calculator::wavefunction()`, `set_wavefunction()` and
`reset_wavefunction()`; `Calculator.set_guess(shell_charges,
atomic_dipoles, atomic_quadrupoles)` (GFN2-xTB moments) is `set_moments`;
`vibrational_analysis(conventions=...)` is the request's `analysis_conventions`; `molden=True`,
`"spherical"` or `"cartesian"` is the request's `molden` and `molden_cartesian`; the arguments
`step`, `accuracy` and `method` of `Calculator.hessian` are `hessian_step`,
`hessian_accuracy` and `hessian_method`.

qccompute (the codec is registered automatically when axtb is installed):

```python
from qccompute import compute
from qcdata import CalcType, Model, ProgramInput, ProgramSpec, Structure
s = Structure.open("mol.xyz")
out = compute(ProgramInput(program="axtb", calctype=CalcType.gradient, structure=s,
                           model=Model(method="gfn2"), keywords={"threads": 4}))
# optimizers (geomeTRIC, Sella) start every step's SCF from the previous step's wavefunction:
opt = compute(ProgramInput(program="geometric", calctype=CalcType.optimization, structure=s,
                           subprograms=[ProgramSpec(program="axtb", calctype=CalcType.gradient,
                                                    model=Model(method="gxtb-v2.0.1"))]),
              propagate_wfn=True)
```

The codec's keywords are the `axtb.Calculator` arguments and the Hessian and analysis settings
under the request's names (`hessian_method`, `hessian_step`, `hessian_accuracy`,
`analysis_conventions`, `temperature`, ..., `masses`, `thermo`, `dispersion`), checked per
method and type before the calculation (`EncoderError`); the module docs of `axtb.qccodec`
list them and where the results go (`extras`).

CLI: `axtb mol.xyz [options]` runs a structure file (XYZ, Ångström), `axtb --json` a JSON
request on stdin (positions in bohr; nothing else on the command line); either prints the
response as JSON on stdout. `axtb --help` lists every option, `axtb --version` the version.
`--molden [FILE]` writes the orbitals as a Molden file (see [Molden files](#molden-files)); `--wbo`
adds the bond orders, and every SCF result has the orbital levels, gap and quadrupole (see
[Bond orders, orbital levels, gap and quadrupole](#bond-orders-orbital-levels-gap-and-quadrupole)).
XYZ files are converted with the Ångström-to-bohr factor of the reproduced program: mctc-lib's
(CODATA 2018) for GFN2-xTB in the `xtb`, `xtb-6.7.1` and `tblite` conventions, GFN-FF in `xtb`
and `xtb-6.7.1` and g-xTB 2.0.1 in `gxtb`, the 1.1.0 program's for g-xTB v1.1, CODATA 2022 in
axtb's conventions. Errors: in `--json` mode `{"error": {"kind": ..., "message": ..., ...}}` on
stdout (`kind`: `input`, `unsupported_element` with `z`, `topology`, `scf_not_converged` with
`iterations`, `energy_change` and `residual`, `numerical`), otherwise `error: <message>` on
stderr; exit code 1. Invalid arguments and unreadable files are usage errors: `error: ...` on
stderr, exit code 2. Warnings go to stderr: more than 8 threads for one SCF single point (see
[Threads](#threads)), and a run-time LAPACK (`AXTB_LAPACK`) that was not used.

| request field (JSON, Python keyword) | CLI | meaning |
|---|---|---|
| `numbers`, `positions` | the structure file | atomic numbers; positions in bohr |
| `charge`, `uhf` | `--chrg Q`, `--uhf N` | total charge (default 0); unpaired electrons (default 0 or 1 by the electron count; GFN-FF ignores it) |
| `method` | `--method M` | `gfn2` (default), `gfnff`, `gxtb-v1.1`, `gxtb-v2.0.1` |
| `conventions` | `--conventions C` | GFN2-xTB `axtb` (default), `xtb` (6.8.0), `xtb-6.7.1`, `tblite`; g-xTB `axtb`, `gxtb`; GFN-FF `axtb`, `xtb`, `xtb-6.7.1` |
| `threads` | `--threads N` | default 1 |
| `accuracy`, `etemp`, `max_iter` | `--acc X`, `--etemp K`, `--max-iter N` | SCF methods: accuracy multiplier (1), electronic temperature (GFN2-xTB 300 K, g-xTB 0 K), iteration limit (GFN2-xTB and g-xTB 2.0.1 250, g-xTB v1.1 1000) |
| `mixer`, `dispersion`, `scf_trace` | `--mixer M`, `--nodisp`, `--scf-trace` | GFN2-xTB: SCF mixer (`auto`, `broyden`, `diis`), dispersion off (validation), every SCF iteration on stderr |
| `alpb`, `gbsa`, `solvation_state` | `--alpb S`, `--gbsa S`, `--solvation-state R` | implicit solvent and its reference state (`gsolv`, `bar1M`, `reference`) |
| `solvation_correction`, `correction_conventions` | `--solvation-correction gfn2`, `--correction-conventions C` | g-xTB in solvent, see [g-xTB in implicit solvent](#g-xtb-in-implicit-solvent) |
| `guess` | — | where a calculator's SCFs start, see [Restarts](#restarts-the-previous-geometrys-wavefunction) |
| `fragment_charges`, `topology_positions` | `--frag-charges Q1,Q2`, `--topology-ref ref.xyz` | GFN-FF, see [GFN-FF](#gfn-ff) |
| `gradient` | `--grad` | the gradient |
| `hessian`, `hessian_method`, `hessian_step`, `hessian_accuracy` | `--hess`, `--hess-method M`, `--hess-step H`, `--hess-acc X` | the Hessian, see [Hessians](#hessians) |
| `molden`, `molden_file`, `molden_cartesian` | `--molden[=FILE]`, `--molden-cartesian` | the Molden file's text in the response; a file to write; Cartesian functions |
| `bond_orders` | `--wbo` | Mayer-Wiberg bond orders |
| `vibrations`, `thermo`, `analysis_conventions`, `masses`, `temperature`, `pressure`, `rotor_cutoff`, `imaginary_cutoff`, `frequency_scale`, `symmetry_number`, `symmetry_threshold` | `--novib`, `--nothermo`, `--analysis-conventions C`, —, `--temp T[,T]`, `--pressure P`, `--sthr X`, `--ithr X`, `--scale X`, `--symnum N`, `--desy X` | the vibrational analysis of a Hessian, see [Vibrational analysis](#vibrational-analysis-and-thermochemistry) |
| `analyze_hessian`, `dipole_derivatives`, `energy` | — | the analysis of a given Hessian, like `xtb thermo` (no calculation settings then) |

The response, for every front end (a field is present when it applies):

| fields | what |
|---|---|
| `program`, `version`, `method`, `conventions`, `eigensolver` | `"axtb"`, axtb's version, the canonical method (`null` for an analysis) and conventions, the SCF's eigensolver (`faer` or the run-time LAPACK library) |
| `energy`, `gradient`, `charges`, `dipole`, `terms`, `timings`, `seconds` | Eh, Eh/bohr (with `gradient`), atomic charges (Mulliken; GFN-FF: EEQ), e·bohr, energy terms (Eh), wall time per phase and of the request (s) |
| `iterations`, `guess`, `mixer_used`, `scf_attempts`, `homo_lumo_gap`, `quadrupole`, `shell_charges`, `atomic_dipoles`, `atomic_quadrupoles`, `orbital_energies_alpha`/`_beta`, `occupations_alpha`/`_beta`, `bond_orders` | SCF methods (`mixer_used` and `scf_attempts`: GFN2-xTB) |
| `n_hbonds`, `n_xbonds`, `topology` | GFN-FF |
| `molden`, `molden_file` | Molden file |
| `hessian`, `hessian_method`, `hessian_max_asymmetry`, `hessian_timings`, `dipole_derivatives` | Hessian |
| `vibrational_conventions`, `frequencies`, `normal_modes`, `reduced_masses`, `force_constants`, `ir_intensities`, `n_imaginary`, `linear`, `masses`, `thermo` | vibrational analysis |
| `composite`, `components`, `parts`, `hessian_parts` | g-xTB in solvent |

## Design rules

- **No implicit threading.** Every calculation runs on exactly the number of threads you
  pass (default 1, at least 1; never clamped) inside its own pool; outer-level parallelism (Hessian displacements)
  runs inner work sequentially, so thread counts never multiply. No `OMP_NUM_THREADS`
  needed, ever. All of axtb's own reductions use fixed chunks summed in a fixed order, so
  results repeat bit for bit on a given thread count. Between thread counts they may differ
  by round-off, as xtb's do: the SCF eigensolver's tridiagonalization (faer's above ~313 rows,
  or the `dsytrd` of a run-time LAPACK library) sums one partial product per thread. Measured
  for GFN2-xTB and g-xTB 2.0.1 on taxol and (H2O)200, 1 vs 2, 4 and 8 threads, with faer and
  MKL on the laptop and faer, MKL and AOCL on the 64-core node (axtb 0.1.0): energies ≤ 1.8e-12
  Eh apart, gradients ≤ 8.0e-15 Eh/bohr, the same SCF iterations; three runs on each thread
  count gave identical bits. GFN-FF differs between thread counts through the LU factorizations
  of its EEQ systems (faer's parallel kernels above ~100 atoms): C50H102, (H2O)200, (H2O)333 and
  crambin on 1, 2, 4 and 8 threads gave energies ≤ 6.5e-13 Eh, gradients ≤ 7.9e-14 Eh/bohr and
  analytic Hessians ≤ 2.3e-13 Eh/bohr² apart (`bench/results/final-*/thread_determinism.json`).
- **Correct by default, exact on request.** The default `conventions="axtb"` uses CODATA
  2022 constants, double-precision tables, one coordination number for energy and
  gradient, smooth long-range cutoffs, and fixes the solvation bugs reported upstream
  (grimme-lab/xtb#1504, #1505). `conventions="xtb"` reproduces the xtb 6.8.0 program,
  `conventions="xtb-6.7.1"` the xtb 6.7.1 program (as the g-xTB 2.0.1 build of xtb has it;
  see [xtb 6.8.0 and 6.7.1](#xtb-680-and-671)) and `conventions="tblite"` tblite 0.8.0, to
  the last digits, quirks included. With implicit solvent, `xtb-6.7.1` mode keeps xtb 6.7.1's
  two solvation bugs, which xtb 6.8.0 fixed, and `tblite` mode uses the corrected solvation
  (tblite has neither bug; agreement ≤ 1.3e-10 Eh).
  For ordinary molecules the modes differ by ~1e-8 Eh. Constants that define the
  published parametrizations are kept in every mode. Parameter tables are generated from
  the reference sources by `crates/axtb/scripts/gen_data.py`, literal for literal. Every
  setting that differs between the modes is one table, `ConventionParams`
  (`crates/axtb/src/conventions.rs`), resolved once at setup. The SCF mixer is one of these
  settings: `xtb`, `xtb-6.7.1` and `tblite` use those programs' Broyden mixing (same
  iterations, same SCF solution where there are several), the default falls back to DIIS
  when Broyden does not converge (see [SCF](#scf)). `xtb` and `xtb-6.7.1` also run their
  program's own SCC loop (start, convergence test, final iteration, neglected integrals,
  gradient moments; see [xtb's SCC loop](#xtbs-scc-loop-conventionsxtb-and-xtb-671)), so they
  reproduce the program at its default accuracy too.
- **Atomic units at the API.** Positions in bohr, energies in Hartree. Physical constants
  are NIST CODATA 2022 via qcref (`crates/axtb/src/units.rs`); constants that are part of
  a parametrization are kept and labelled as such.
- **Fail loudly.** Unknown keywords, settings that do not apply to the method, impossible
  spin states and SCF failures raise errors; nothing is silently ignored (see [Use](#use)).

## Threads

axtb runs every calculation on the threads you ask for (default 1) and never caps them, but
more is not always faster. One exception: a one-off calculation (the CLI, Python's
`axtb.singlepoint` and `axtb.hessian`) of a molecule so small that threads only cost runs on
one thread, with the same results (GFN2-xTB single points up to 10 atoms and Hessians up to 6,
g-xTB 2.0.1 single points up to 9 atoms, v1.1 up to 6, GFN-FF single points up to 100 atoms and
Hessians up to 10; g-xTB's Hessians always use the threads; `axtb_driver::task_threads`).
An `axtb.Calculator` keeps the threads you give it. Measured on a whole 64-core AMD Threadripper PRO 9985WX
node with axtb 0.1.0 (`bench/results/cluster-9985wx`, the release run of 2026-11-07: energy +
gradient in one Python process, xtb conventions, seconds):

| molecule (atoms) | 1 thread | 8 threads | 32 threads | 64 threads |
|---|---|---|---|---|
| taxol (113) | 0.108 | 0.105 | 0.130 | 0.184 |
| (H2O)200 (600) | 2.59 | 0.98 | 1.41 | 1.87 |
| (H2O)333 (999) | 8.29 | 3.03 | 3.72 | 4.71 |

- **One calculation:** use at most about 8 threads. Beyond that the SCF eigensolver gets
  slower with every library we measured (faer, OpenBLAS, MKL, AOCL), and axtb warns about a
  single point of an SCF method (GFN2-xTB, g-xTB) on more threads (CLI: stderr; Python:
  `RuntimeWarning`; `axtb_driver::threads_warning` and `axtb::threads` in Rust). GFN-FF has no
  SCF and gets no warning.
- **Many calculations** (conformers, path images, batches): run them in parallel with 1
  thread each. This always gives the most work per node; small molecules gain little from
  threads at all (taxol in the release run: 1.1× on 8 threads, node; 1.5× on the laptop with
  faer).
- **Hessians** run their perturbations or displacements in parallel and do use many threads
  well, with no warning (taxol, GFN2-xTB: analytic 2.89 s on 1 thread, 0.87 s on 8 on the
  laptop; numerical 39.7 s on 1 thread and 5.2 s on 8 on the node, `bench/results/hessian.json`;
  on a whole node 1.5 s on 32 threads and 1.1 s on 64, `bench/results/cluster-9985wx`).
- **A LAPACK library for the SCF eigensolver** (optional): `AXTB_LAPACK=<path or soname>`
  names Intel MKL (`libmkl_rt`), OpenBLAS, or AMD AOCL's `libflame.so` together with
  `AXTB_LAPACK_BLAS=<libblis-mt.so>`. The library is opened at run time (nothing is linked) and
  solves the SCF's symmetric eigenproblems from 200 basis functions on, threaded with the
  calculator's thread count, which axtb sets for each call through the library's API. faer
  stays the default, the solver of smaller matrices (where the library's start-up would cost
  more than it saves) and the fallback. The rules:
  - OpenBLAS must be an OpenMP build (Debian/Ubuntu `libopenblas0-openmp`) or a sequential
    one. A pthreads build (`libopenblas0-pthread`) has one process-wide thread count and is
    refused; `openblas_get_parallel` tells them apart.
  - Only LP64 libraries (32-bit integers) work: an OpenBLAS built with 64-bit integers is
    refused, MKL is set to LP64, and an AOCL library must be an LP64 build.
  - MKL runs in its strict conditional numerical reproducibility mode (`MKL_CBWR_Set`), so its
    results repeat bit for bit on a given thread count. An MKL that was set up differently
    earlier in the process (for example by an MKL build of NumPy) is refused.
  - MKL's Intel OpenMP runs without thread affinity (its topology detection is slow on
    many-CPU machines and runs in every process). That setting (`kmp_set_defaults`) applies
    to every user of Intel OpenMP in the process.
  - `AXTB_LAPACK` and `AXTB_LAPACK_BLAS` are read once per process, at the first SCF large
    enough to use them; later changes have no effect.
  - When the library cannot be loaded or is refused, faer solves instead. The CLI and the Python
    module warn ("AXTB_LAPACK not used"), and the `eigensolver` field of every response says
    which solver ran (`faer` or the library); in Rust, `ScfReport::eigensolver` and
    `axtb::linalg::runtime_lapack()`.

  Between thread counts, and against faer, the results differ by round-off (≤ 2e-12 Eh in the
  release run). On the Intel laptop MKL makes GFN2-xTB faster than xtb 6.8.0 in every measured
  case (1.03-8.4x on 1-8 threads; with faer 0.75-0.99x for the water clusters on 2-8 threads and
  C100H202 on 4-8); on the AMD node faer is ahead of xtb 6.8.0 everywhere (1.07x and up), AOCL
  about as fast and faster for the largest systems on 1 thread, MKL slower; `bench/README.md`,
  "Final timings".

## Environment variables

At run time axtb reads two variables, and nothing else: thread counts are always explicit, so
`OMP_NUM_THREADS` and the like play no part.

| variable | meaning |
|---|---|
| `AXTB_LAPACK` | a LAPACK library for the SCF eigensolver (path or soname; see [Threads](#threads)) |
| `AXTB_LAPACK_BLAS` | the BLIS library that AOCL's libFLAME needs, loaded first |

The SCF trace that an environment variable used to switch on is the `--scf-trace` option
(`scf_trace` in requests and Python).

The validation scripts and benchmarks read:

| variable | meaning |
|---|---|
| `AXTB_XTB680`, `AXTB_GXTB2`, `AXTB_XTB671`, `AXTB_GXTB11_DIR` | the reference programs (`tests/programs.py`; see [Checks](#checks)); nothing else is searched, and a missing program skips the cases that need it |
| `AXTB_BIN` | the axtb CLI (default `target/release/axtb`) |
| `AXTB_REFERENCE` | the directory of upstream sources (default `reference/`, see `reference/README.md`); the Hessian comparisons read xtb's atomic masses there |
| `AXTB_SCRATCH` | where the programs run (default: the system's temporary directory) |
| `AXTB_XTB_CONVENTIONS` | the conventions the reference matrix compares an xtb binary with (default: from its version) |
| `AXTB_CHECK_DIR`, `AXTB_CHECK_PYTHON`, `AXTB_CHECK_JOBS`, `AXTB_PRIVATE_PATTERN` | `scripts/check.sh`: scratch directory, the venv's Python, parallel program runs, an extra pattern for the privacy grep |

`bench/README.md` lists the benchmark harness's own variables.

## xtb 6.8.0 and 6.7.1

`conventions="xtb"` reproduces xtb 6.8.0 (`xtb version 6.8.0 (4b8c987)`, released
2026-11-05). `conventions="xtb-6.7.1"` reproduces xtb 6.7.1 as the g-xTB 2.0.1 build of xtb
has it (`6.7.1 (26dd68d)`): that build is the 6.7.1 release plus 127 later commits of the main
branch, and the g-xTB composite's program mode (`conventions="gxtb"`) maps its GFN2 halves
to `xtb-6.7.1`, the xtb of that build. The two rows are tables of `ConventionParams`
(`crates/axtb/src/conventions.rs`); the names are the same in Rust (`Conventions::Xtb`, `Conventions::Xtb671`), the CLI, Python and the qccodec codec
(`"xtb-6.8.0"` is accepted for `"xtb"`).

For GFN2-xTB, xtb 6.8.0 adopted tblite's numerics ("the native xtb and tblite implementations
are now fully consistent"): at `--acc 1e-4` axtb's `tblite` conventions already reproduce
6.8.0's energies to 4.4e-13 Eh. What 6.8.0 kept of xtb is its EEQ start, aufbau occupations
up to 0.1 K, its solvation constants and its gradient potentials.
[docs/validation.md](docs/validation.md#what-xtb-680-changed-for-gfn2-xtb) lists each behaviour
of 6.7.1 that axtb reproduces, what 6.8.0 does instead and what the 6.7.1 behaviour costs
against the 6.8.0 program.

New in 6.8.0 and reproduced: the gradient potentials (SCC 6), the bit-exact 6.8.0 tables
(`crates/axtb/src/param/xtb_data.rs`, `XTB_*`, from the v6.8.0 source), the solvation variant
(`axtb_solvation::Conventions::Xtb`: mctc-lib's bohr for the radii, xtb's constants, both bugs
fixed). Also in 6.8.0, without a measurable effect on GFN2-xTB results here: Broyden solves
its linear system by LU instead of inverting it, the Fermi-level search stops at
min(√ε, 1e5·ε) electrons as tblite's, D4 Gaussian weights are no longer cut at 1e-7, the H0
gradient no longer divides by the overlap (xtb PR #1214), and single atoms use Broyden mixing
too. The vibrational analysis of `xtb --hess` and `xtb thermo` changed as well (projection in
mass-weighted coordinates, another linearity test, mctc-lib's bohr in the moments of inertia);
the `xtb` analysis conventions follow 6.8.0 and `xtb-6.7.1` keeps 6.7.1's (see [Vibrational
analysis](#vibrational-analysis-and-thermochemistry)). GFN-FF follows 6.8.0 too, with 6.7.1's
force field kept as `xtb-6.7.1` (see [GFN-FF](#gfn-ff)).

**The 6.7.1 release itself** (`edcfbbe`, e.g. the copy that ships with ORCA 6.0) is not the
g-xTB build: its GFN2 gradient leaves out the H0/S term of AO pairs whose overlap vanishes by
symmetry, i.e. of atoms that share a Cartesian coordinate (fixed by xtb PR #1208, March 2025,
before the g-xTB build). docs/validation.md compares the `xtb-6.7.1` conventions with it.

## SCF

The SCC variables (shell charges, atomic dipoles and quadrupoles) are mixed between
iterations by the mixer chosen with `mixer` (Python, CLI `--mixer`, JSON `"mixer"`, qccodec
keyword, `axtb::Options::mixing` in Rust; default: the conventions' choice):

- `"broyden"`: modified Broyden mixing with tblite's weights, damping 0.4 and full history.
  The `xtb`, `xtb-6.7.1` and `tblite` conventions use it alone, so they reproduce the programs' iteration
  counts, the solution where there are several, and their failures.
- `"diis"`: Pulay (DIIS, Anderson) mixing over the last 30 iterations, coefficients from a
  QR factorization of the residual differences that leaves out nearly dependent ones,
  damping 0.4, and a restart from the best iterate with half the damping when the residual
  stops improving for 12 iterations or grows a thousandfold.
- `"auto"` (default of the `axtb` conventions): Broyden; if it does not converge within
  `max_iter` (or breaks down), DIIS starts again from the initial guess with its own
  `max_iter` iterations. A failure is reported only if both fail, and the error then counts
  the iterations of both attempts. Every result Broyden converges is exactly Broyden's.

Results say which mixer produced them: `mixer_used` (`"broyden"`, `"diis"`, or
`"diis-fallback"` when Broyden failed and DIIS converged) and `scf_attempts` (mixer,
iterations, converged of each attempt; `iterations` is that of the converging attempt), in
`ScfReport` (Rust, `SinglePoint::scf()`), the result dict (Python), the CLI's JSON and the
qccodec `extras`, so a rescued point on a path is visible.

docs/validation.md has how the mixers do on a set of hard SCF cases.

### xtb's SCC loop (`conventions="xtb"` and `"xtb-6.7.1"`)

The `axtb` and `tblite` conventions run tblite's SCF loop. The `xtb-6.7.1` conventions run the
xtb 6.7.1 program's own (`scc_core.f90`, `scf_module.F90`, `aespot.F90`), which differs from
tblite's in six places. xtb 6.8.0 took over tblite's loop in four of them; `xtb` runs 6.8.0's,
which keeps xtb's start (1) and its own gradient potentials (6). Each place is a setting of
`ConventionParams` (`eeq_guess_cn` and the `scc_*` fields). They matter only while the SCF is
unconverged, so at xtb's default accuracy 1 (thresholds 1e-6 Eh and 2e-5 e, 6.7.1: 1e-4 e),
where tblite's loop does not reproduce the program's GFN2 gradients to the last digits.

1. **Start** (both xtb versions): `goedecker_chrgeq` EEQ charges on `ncoord_erf`'s
   coordination number (xtb's D4 radii, 40 bohr, no cap; 6.7.1's radii single-precision with
   the older bohr constant), not multicharge's CN capped at 8, and not tblite's zero charges.
   The first iteration's energy then agrees to all printed digits.
2. **Residual test** (`scc_residual`): 6.7.1 tests sqrt(Σ dq² / N_atoms) < 1e-4·acc over the
   mixed vector (shell charges, dipoles, quadrupoles); tblite and 6.8.0 the RMS over the vector
   < 2e-5·acc; plus |ΔE| < 1e-6·acc in all.
3. **One more iteration** after the first converged one (6.7.1 `lastdiag`,
   `scc_extra_iteration`). Once converged, the SCF counts as converged even if that last
   iteration misses the thresholds. 6.8.0 stops at the converged iteration without mixing.
4. **Mixer metric** (`scc_quadrupole_scale`): 6.7.1's mixed vector holds the quadrupole
   off-diagonals as they are. tblite and 6.8.0 scale them by √2, which changes Broyden's dot
   products and so its iterates.
5. **Neglected integrals** (6.7.1, `scc_multipole_neglect`): xtb's moment integrals are taken
   around the coordinate origin, and 6.7.1's Fock matrix leaves out an AO pair's dipole
   (quadrupole) term when Σ_k d_k² (Σ_diag M² + 2 Σ_off M²) ≤ 1e-10·acc (`setdqlist`,
   `buildIsoAnisotropicH1`). The overlap part of every potential stays, and the Mulliken
   moments, energy and gradient use all integrals (`hamiltonian::MultipoleNeglect`). The
   effect depends on where the molecule sits relative to the origin and is far below the SCF
   thresholds. 6.8.0 sets the threshold to 0.
6. **Gradient moments and potentials** (`scc_gradient`): 6.7.1 mixes once more after the last
   iteration and evaluates the ES2, AES, D4 and solvation gradients and the multipole
   potentials of the integral derivatives with those mixed moments. 6.8.0 uses the output
   moments for all of these and recomputes the multipole (AES) potentials from them
   (`setdvsdq`). In both, the isotropic shell shifts of the overlap term are the last
   iteration's input potentials (`ves`). tblite takes every potential from the last
   iteration's input. The energy is always the last iteration's, from its output moments.

The agreement this gives at xtb's default accuracy is in docs/validation.md.

### Restarts: the previous geometry's wavefunction

A `Calculator` starts each SCF from the previous geometry's converged state: GFN2-xTB always (its
shell charges and atomic multipoles), g-xTB in the default conventions. For g-xTB that state is
what the programs' restart files hold: v1.1 the density matrices (`gxtbrestart`), 2.0.1 the shell
charges and magnetizations, atomic dipoles and quadrupoles and the density matrices
(`xtbrestart`), taken over as they are although the charge- and CN-dependent q-vSZP basis of the
new geometry differs (the programs do the same). An SCF from a previous state that does not
converge runs once more from the program's EEQ-BC start. Results say which start produced them
(`guess`: `previous`, `eeq-bc`, `eeq`, `sad`; Rust `ScfReport::start`).

- `conventions="gxtb"` keeps the programs' fresh EEQ-BC start for every SCF, so each result is
  that of a fresh program run. `guess="previous"` (Rust `ScfSettings::restart`) reproduces the
  programs' restarts: a second run in the same directory or the next step of an optimization
  inside one run. Against the programs run that way (caffeine, water, triplet O2,
  [Cu(NH3)4]²⁺; the same geometry, then 0.02, 0.1 and 0.3 bohr rms displacements, each from the
  previous run's restart file): energies ≤ 4.3e-11 Eh, identical SCF iterations (2.0.1: 2 at the
  same geometry, as the program; 1.1.0: 3). The 1.1.0 program restarts without the transition
  step of its Fock mixing (v1.1 spec §4.17).
- `guess="fresh"` starts every SCF from the method's own start in any conventions: g-xTB's
  EEQ-BC charges (also `guess="eeq-bc"`), GFN2-xTB's EEQ charges (also `guess="eeq"`).
  `guess="sad"` starts every GFN2-xTB SCF from neutral atoms. A named start of another method
  is refused, and the solvation composite takes only `previous` or `fresh`.
- In the default conventions a restarted g-xTB 2.0.1 SCF converges to 0.1 × the thresholds
  (`axtb_gxtb::calculator::RESTART_ACCURACY`). A fresh 2.0.1 SCF ends with a DIIS step that
  overshoots tblite's thresholds; a restart meets them after a few iterations and would stop
  there, with larger gradient errors than a fresh SCF's that depend on the history and that
  lengthen a geomeTRIC minimization of caffeine. With the tighter thresholds it takes 25 steps
  (23 from fresh starts),
  restarted energies are 3–50 times closer to the converged ones than fresh ones and gradients
  about as close (table). v1.1 and GFN2-xTB restarts keep the thresholds: their errors are 1–4
  times a fresh SCF's but at most 8e-6 Eh/bohr, and the optimizers take the same steps.
- Python's `Calculator.guess_state()` returns the last converged state (an `axtb.ScfState`) and
  `set_guess_state(state)` starts another calculator for the same method and molecule from it.
  In Rust these are `wavefunction()` and `set_wavefunction(wf)` of the `Method` trait, which
  every calculator implements (an `axtb::method::Wavefunction`; it starts only a calculator of
  its method and molecule: atomic numbers, charge and unpaired electrons), and of
  `axtb_driver::Calculator`. A given wavefunction starts the next SCF whatever `guess` says,
  and after a failed calculation a calculator has none. Python's `Calculator.set_guess(shell_charges,
  atomic_dipoles, atomic_quadrupoles)` starts GFN2-xTB from given moments
  (`axtb_driver::Calculator::set_moments`).
  The codec uses this for qccompute's `propagate_wfn=True`: optimizers (geomeTRIC, Sella) start
  every step from the previous step's wavefunction, which stays in the process's memory; outputs
  carry its id (`extras['wavefunction_id']`). Without `propagate_wfn` codec calls stay
  independent.
- The CLI runs one geometry per call and always starts fresh (a g-xTB single point after a
  Hessian too, also in Python's `Calculator.hessian`), so CLI results do not change.

docs/validation.md has what restarts save along paths and in optimizations, and where g-xTB
has several SCF solutions, so that a restart can end on another one than a fresh start.

## Hessians

`Calculator::hessian(positions, &HessianOptions { method, step, accuracy })`:

- `Analytic` (default): analytic second derivatives with the coupled-perturbed SCC response.
  The SCC variables (shell charges, atomic dipoles and quadrupoles) respond to all 3N
  displacements at once through `(1 - χ0 K) δs = u + χ0 c` in SCC-variable space: χ0 from
  the orbitals with Fermi-smearing divided differences and per-spin Fermi-level shifts
  (open shells: different alpha/beta occupations of the same orbitals; degenerate levels
  via the derivative limit), K the second derivatives of the SCC energy (ES2, ES3, AES, D4,
  ALPB/GBSA Born matrix), u the response to the skeleton Fock/overlap perturbations, c the
  explicit geometry derivative of the potentials. The energy-weighted density responds
  through the divided differences of fε and fε², not as ½PFP. The Hessian is
  `E'' + H_skel + u·K δs + u·c + c·δs`; every explicit second derivative is analytic
  (`multipole_hess` integral kernel, CN chain rules, ES2, AES with CN-dependent damping, D4
  two- and three-body, repulsion, ALPB/GBSA incl. Born radii, shape term and SASA). It is
  symmetric to round-off. In xtb conventions with solvation, xtb's stale-`r1` Born
  integral has no consistent derivatives, and that one term uses central differences.
- `Response`: column y is the central difference (step 5e-4 bohr) of the gradient
  functional at R ± h e_y with P, W and the SCC variables extrapolated by the response, no
  SCF per displacement. Same accuracy as an FD Hessian without SCF noise.
- `Numerical`: central differences of analytic gradients (step 0.005 bohr), one SCF per
  displacement converged to 0.01.

`accuracy` is the SCF accuracy of the reference (analytic/response, default 1e-3) or of the
displacements (numerical, default 0.01). All methods run on the calculator's thread pool,
parallel over perturbations/displacements with sequential inner work.

Two things to know when comparing with FD Hessians: the ALPB/GBSA surface term is C¹ but not
C² (clamped cubic switching per surface grid point), so FD Hessians in implicit solvent
average over many second-derivative jumps and differ from the exact one by up to ~1.5e-4
Eh/bohr² for steps ≥ 2e-4 bohr (caffeine: up to 14 cm⁻¹ in soft modes, while the FD
Hessians of axtb and xtb agree with each other); they cannot be Richardson-extrapolated.
`Analytic` gives the exact derivative at the point; `Response` with `step=0.005` gives the
smoothed FD Hessian that xtb reports, without SCF noise.

## g-xTB

g-xTB v1.1: `method="gxtb-v1.1"` in Python and qccodec, `--method gxtb-v1.1` in the CLI,
`axtb_gxtb::Calculator` in Rust (implements the `Method` trait, as `axtb::Calculator`
does): energies, analytic gradients, semi-numerical Hessians. It uses the
g-xTB 1.1.0 program's SCF (thresholds, start density, mixing; the CLI reads XYZ files with
the program's Ångström factor), accepts `accuracy` (multiplier of the program's SCF
thresholds), `etemp` (Fermi smearing, see below), `max_iter`, `threads` and `conventions`, and
rejects the GFN2-only settings.
As for GFN2, the default is corrected and a mode reproduces the program exactly:
`conventions="axtb"` (default) diagonalizes the exact derivative of the energy, so the SCF
is stationary; `conventions="gxtb"` uses the program's Fock matrix (it leaves out two
potential terms) and reproduces its energies, with the gradient's response term computed
from a Z-vector equation. The two differ by up to ~4e-5 Eh in total energies (2.3e-4 for
ferrocene) and ≤ 1.1e-5 Eh in the energy differences tested (conformers, the SN2 identity
reaction and HCN → HNC with their barriers, bindings, dissociations;
`tests/gxtb_energy_differences.py`, spec §4.18). In program conventions, as in the program, results depend slightly on the
orientation of the molecule (up to ~1e-7 Eh; the program's onsite-exchange Fock matrix is not
rotation covariant); the default conventions are invariant.

g-xTB 2.0.1 (the grimme-lab/g-xtb 2.0.1 release, `xtb --gxtb`): `method="gxtb-v2.0.1"`
(also `gxtb-2.0.1`, `g-xTB 2.0`) everywhere `gxtb-v1.1` is accepted, with the same options.
2.0.1 refits every parameter, covers H–Lr, and adds an H0 anisotropy term, a self-consistent
repulsion, an exchange bond-order correction, charge-dependent onsite exchange and screened
revD4 damping (`docs/gxtb-v2.0.1-spec.md`). Its Fock matrix is the exact derivative of the
energy, so both conventions share the Fock matrix, the SCF (tblite's: damping, then DIIS on
the full Fock matrix) and analytic gradients without a response term.
`conventions="gxtb"` reproduces the program: mctc-lib's Ångström factor, its fractional
electron count (the sum of the stored reference occupations), its hard cutoffs (CNs and
repulsion 25 bohr, revD4 50/25 bohr, integral pairs) and its gradient formula.
`conventions="axtb"` (default) uses CODATA 2022 units, an integer electron count and smooth
cutoffs (1 bohr switches beyond the program's radii). The two differ by ≤ 3e-10 Eh for the
small reference molecules (same iterations) and by ≤ 9e-10 Eh in the energy differences
tested on them (2.3e-8 Eh for a 2 % expansion of C50H102), by ≤ 4e-11 Eh for the conformer,
reaction, barrier and binding energies of `tests/gxtb_energy_differences.py`; for (H2O)100
by 1.9e-6 Eh, almost all from the revD4 three-body tail beyond 25 bohr that the program cuts (`bench/results/gxtb2_reproduction.json`, from
`python tests/gxtb2_reproduction.py`). The program's analytic gradient is wrong (by up to
1.4e-4 Eh/bohr) for some complexes with bonds exactly along a coordinate axis
([Gd(H2O)6]³⁺, [Lu(H2O)6]³⁺; finite differences of its own energies follow axtb, spec §4.18);
axtb's gradient is the derivative of the energy in either orientation.

Electronic temperature (both versions): `etemp` (kelvin; CLI `--etemp`, Python `etemp=`,
qccodec keyword `etemp`) smears the occupations with a Fermi distribution and adds the entropy
term −TS (term `entropy`) to the energy; the default is 0 K, as in the g-xTB programs (GFN2
keeps 300 K). Energies, analytic gradients and, for 2.0.1, analytic and response Hessians
include the response of the occupations (Fermi-level shifts). `conventions="gxtb"` follows each
program, quirks included: 2.0.1 (`--etemp`) rounds its fractional electron count per spin
channel at any finite temperature and stops its Fermi-level search at 1.5e-8 electrons; 1.1.0
(`-tel`) uses a single-precision kB, smears the damped Fock matrices of the first SCF
iterations with their scale, and computes only the lowest N_α + 10 orbitals (the others stay
empty). `conventions="axtb"` uses the exact Fermi level and CODATA 2022 kB (9.4e-7 of −TS
apart from the programs' kB: ≤ 3e-11 Eh unless −TS > 1e-4 Eh, at most 4.5e-8 Eh, EuCl3 at
5000 K). The agreement with the programs at 300-5000 K is in docs/validation.md.

## g-xTB in implicit solvent

g-xTB has no implicit-solvent parametrization (the 2.0.1 program stops with `--alpb`/`--gbsa`).
axtb puts it in solvent by adding GFN2-xTB's solvation free energy, a composite:

    E_solv(x) = E_gxtb(x) + [E_GFN2+ALPB(x) − E_GFN2(x)]

and the same for gradients and Hessians, so optimizations, transition-state searches and IRCs
run on the solvated surface. The solvation correction ΔG_solv(x) = E_GFN2+ALPB(x) − E_GFN2(x) is
the difference of two GFN2-xTB calculations on the same geometry with the same settings, one of
them with ALPB or GBSA. The composite comes from a reaction-path workflow that ran the
three parts as three xtb program calls per gradient; axtb runs them as one calculation
(`axtb_gxtb::CompositeCalculator`), which keeps each part's setup between geometries:

```python
r = axtb.singlepoint(numbers, positions_bohr, method="gxtb-v2.0.1", gradient=True,
                     alpb="water", solvation_correction="gfn2")         # or gbsa=, solvation_state=
calc = axtb.Calculator("gxtb-v2.0.1", numbers, positions_bohr, alpb="water", solvation_correction="gfn2")
h = calc.hessian(positions_bohr)            # H_gxtb + H_GFN2+ALPB − H_GFN2, analytic parts
r["components"]    # gxtb, gfn2_solvated, gfn2_gas, solvation_correction (Eh)
r["parts"]         # per part: iterations, mixer_used, scf_attempts, etemp, max_iter, retry
h["hessian_parts"] # the same for the parts of the Hessian, with each part's hessian_method
r["composite"]     # definition "E = E[gxtb-v2.0.1] + (E[gfn2 + alpb(water, gsolv)] - E[gfn2])" and settings
```

CLI: `axtb mol.xyz --method gxtb-v2.0.1 --alpb water --solvation-correction gfn2 --grad`
(JSON `"solvation_correction": "gfn2"`); qccodec: `Model(method="gxtb-v2.0.1")` with keywords
`{"alpb": "water", "solvation_correction": "gfn2"}`, results in `extras["composite"]`,
`extras["components"]`, `extras["parts"]` (and `extras["hessian_parts"]`).

**Why a keyword and not a method name.** The method stays the g-xTB version that gives the
gas-phase energy. The solvent is named exactly as for GFN2-xTB (`alpb`/`gbsa`,
`solvation_state`), and one more keyword names the method that supplies the correction. A
g-xTB method with `alpb`/`gbsa` alone is an error that points to `solvation_correction`, so
solvent never silently means the composite. A GFN2-xTB method with `solvation_correction` is an
error too, since it has its own ALPB/GBSA. A compound method name would repeat the solvent
model given in `alpb=`/`gbsa=`, and a slash in it would read as method/basis to qcdata tools.

**The parts.**
- **The GFN2 halves** run at xtb's GFN2-xTB defaults (accuracy 1, 300 K, 250 iterations) in the
  `gsolv` reference state (1 M gas to 1 M solution), always with the same settings; only the
  solvent model differs.
- **Electronic temperatures.** By default g-xTB runs at its programs' 0 K and the GFN2 halves
  at xtb's 300 K. A given `etemp` applies to all three parts.
- **Retry ladders.** A GFN2 pair whose SCF fails (not converged, or broken down) runs again,
  both halves with the same settings: at 1000 K with 1000 iterations, then at 300 K with 1000
  iterations, then at 5000 K with 1000 iterations. A g-xTB SCF that fails runs again at 1000 K,
  then at 3000 K (1000 iterations each), unless `etemp` was given. The rung each part converged
  on is in `parts[...]["retry"]`, so a result from a smeared rung is never mistaken for one at
  the default temperature.
- **The zero check.** The calculation stops if the first geometry's ΔG_solv is zero
  (|ΔG| < 1e-9 Eh), instead of reporting gas-phase energies as solvated.

**Conventions.** `conventions` are the g-xTB part's (as for g-xTB alone). The GFN2 halves follow
them, `axtb` → `axtb` (corrected solvation) and `gxtb` → `xtb-6.7.1` (xtb 6.7.1's solvation and
SCC loop: the g-xTB 2.0.1 build of xtb is 6.7.1 for GFN2-xTB), unless `correction_conventions`
says otherwise. `conventions="gxtb"` therefore reproduces the three programs of that build:
g-xTB as `xtb --gxtb`, the halves as `xtb --alpb water` and `xtb`, each SCF from the program's
start; `correction_conventions="xtb"` gives halves in xtb 6.8.0's conventions. In the default
conventions all three parts start from the previous geometry's wavefunction (see
[Restarts](#restarts-the-previous-geometrys-wavefunction)); `guess` sets it for all three.

`accuracy` and `max_iter` apply to all parts and `mixer` to the GFN2 halves. The charges,
moments, orbitals (`molden=`) and dipole derivatives (IR intensities of the composite Hessian)
are g-xTB's. Hessians take each part's default method (analytic for 2.0.1 and GFN2) or the
`method` given for all. The parts run one after the other on the calculator's one thread
pool, so threads change nothing beyond the last bit.

docs/validation.md has the comparison with the three programs and the timings.

## GFN-FF

`method="gfnff"` (CLI `--method gfnff`, qccodec `Model(method="gfnff")`) is GFN-FF as xtb
6.8.0 computes it (`xtb --gfnff`; `conventions="xtb-6.7.1"`: as xtb 6.7.1, the g-xTB 2.0.1
build `26dd68d`, computes it), ported from xtb's `src/gfnff` term by term
(`docs/gfnff-spec.md`). The topology (neighbors, hybridization, rings, pi bond orders,
topology charges, fragments, bonded terms, HB/XB candidates) is set up from the first
structure and kept for every later geometry, as xtb does within an optimization; a reference
structure can set it instead:

```python
calc = axtb.Calculator("gfnff", numbers, start_bohr, threads=8, fragment_charges=None)
r = calc.singlepoint(positions_bohr, gradient=True)   # energy, gradient, EEQ charges, terms, topology
r["topology"]["bonds"], r["topology"]["fragments"], r["topology"]["topology_charges"]
r = axtb.singlepoint(numbers, positions_bohr, method="gfnff", gradient=True,
                     topology_positions=start_bohr)     # same, stateless
h = calc.hessian(positions_bohr)                # analytic, with dipole derivatives and IR
h = calc.hessian(positions_bohr, method="numerical")    # xtb's finite differences (step 0.005)
```

CLI: `axtb mol.xyz --method gfnff [--grad] [--hess] [--topology-ref ref.xyz] [--frag-charges 1,-1]`;
JSON requests take `topology_positions` and `fragment_charges`. `fragment_charges` are the
charges of the molecular fragments in the order of their first atoms (xtb's `.CHRG` second
line), one per fragment and summing to the total charge, else an error; default: the total
charge on the first fragment. `--uhf` has no effect; SCF, solvation, Molden and bond-order
options are rejected. `conventions` `"axtb"` (default) and `"xtb"` (or `"xtb-6.8.0"`) compute
xtb 6.8.0's force field, `"xtb-6.7.1"` 6.7.1's (Rust: `axtb_gfnff::Conventions`, with one
crate-private table of the settings that differ). `"xtb"`
and `"xtb-6.7.1"` read XYZ files with xtb's (mctc-lib's) Ångström factor and analyze Hessians in
that version's conventions with xtb's GFN-FF frequency scale 1.03: `--conventions xtb --hess
--hess-method numerical` reproduces `xtb --gfnff --hess` of 6.8.0 (`xtb-6.7.1`: of 6.7.1)
thermochemistry to ≤ 5e-13 Eh (6.7.1: 2e-11), the analytic Hessian differs from xtb's by xtb's
finite-difference error (≤ 0.3 cm⁻¹).

Not implemented: periodic systems, implicit solvation with GFN-FF, PDB residue charges.
docs/validation.md has the agreement with xtb, what xtb 6.8.0 changed in GFN-FF, and the
time and memory against xtb up to 9900 atoms.

## Vibrational analysis and thermochemistry

Every Hessian comes with its vibrational analysis (`axtb::vibrations::analyze`; Python
`Calculator.hessian`, or `axtb.vibrational_analysis` for a Hessian from any source; CLI `--hess`;
the qccodec codec), whatever the method: harmonic frequencies (cm⁻¹, imaginary ones negative) and
the number of imaginary modes (`n_imaginary`: every negative frequency, however small; the
thermochemistry's `n_dropped` counts those beyond `imaginary_cutoff`), normal modes
(Cartesian displacements of unit length, qcdata's
`normal_modes_cartesian`), reduced masses, force constants, IR intensities (km/mol) and xtb's
quasi-RRHO thermochemistry: ZPVE, H(T), S, T·S, G(RRHO) = H(T) − T·S, the total free energy
E + G(RRHO), heat capacities and the translational, rotational and vibrational parts, at any
temperatures and pressure. Low modes are interpolated to free rotors below `rotor_cutoff` (xtb's
`sthr`, 50 cm⁻¹; Grimme's modified RRHO), imaginary modes smaller than `imaginary_cutoff` (xtb's
`ithr`, 20 cm⁻¹) are inverted and larger ones left out (`n_dropped`), `frequency_scale` scales
the frequencies for the thermochemistry (xtb uses 1.03 for GFN-FF), and the rotational symmetry
number comes from the point group (detected with xtb's tolerance `desy`, 0.1 bohr, up to 200
atoms) unless `symmetry_number` is given.

The dipole derivatives for the IR intensities come with the Hessian (`dipole_derivatives`,
3N x 3): from the SCC response for GFN2-xTB (analytic and response Hessians), from the
coupled-perturbed density response for g-xTB 2.0.1 (analytic: the Mulliken charges' and atomic
dipoles' integral derivatives plus the Mulliken analysis of dP; response: the extrapolated
densities), and from the displaced dipoles of numerical Hessians (GFN2-xTB, g-xTB v1.1 and
2.0.1).

Three conventions, chosen with the calculation's (`xtb` for GFN2-xTB and GFN-FF
`conventions="xtb"`, `xtb-6.7.1` for `conventions="xtb-6.7.1"` and g-xTB's `conventions="gxtb"`,
`axtb` otherwise) or with `analysis_conventions` (CLI `--analysis-conventions`; the
`conventions` argument of `axtb.vibrational_analysis`):

- `xtb` reproduces `xtb --hess` and `xtb thermo` of xtb 6.8.0: xtb's atomic masses and
  constants; the molecule as given (6.8.0 no longer rotates it before the numerical Hessian,
  PR #1426); translations and the three rotations about the center of mass (two when a
  unit-mass moment of inertia about the centroid is below 1e-4 bohr², `is_linear`) projected in
  mass-weighted coordinates (PR #1435) and the projected Hessian diagonalized whole, its first
  modes below 0.01 cm⁻¹, as many as were projected, being the rigid-body modes; the
  thermochemistry through its cal/mol unit chain with mctc-lib's CODATA 2018 bohr in the moments
  of inertia (`axis2`, the free-rotor average moment; PR #1478), `is_linear` for the rotational
  partition function, and the symmetry number of its three-character point-group label (e.g. 1
  for C10v);
- `xtb-6.7.1` reproduces xtb 6.7.1 (the g-xTB 2.0.1 build, so also `xtb --gxtb --hess`): the same
  masses, constants and `thermodyn`, but translations and rotations about the geometric center
  projected from the Cartesian Hessian before mass weighting, no projection for linear
  molecules (smallest moment below 3e-4 u Å², the five lowest modes dropped), the bohr
  0.52917726 Å in the moments of inertia, and a rotational constant below 1e-10 cm⁻¹ as the
  linearity test. The CODATA 2018 R, h and k_B and the exact translational entropy of xtb's
  `thermodyn` are already in this build; 6.8.0 changed the projection, the linearity test and the
  bohr, which move G(RRHO) by up to 2.6e-10 Eh on the same Hessian at a minimum (caffeine) and by
  up to 2e-3 Eh away from one, where the two projections differ (the reference-matrix Hessian
  cases have gradients up to 4e-2 Eh/bohr: up to 180 cm⁻¹ in the frequencies);
- `axtb` (default): CODATA 2022 constants and qcref's standard atomic masses (the masses
  `qccodec.testing.check_normal_modes` uses), translations and rotations about the center of
  mass projected in mass-weighted coordinates and the Hessian diagonalized in their complement
  (always exactly 3N−6 or 3N−5 modes), every mode in the thermochemistry, the point group's
  exact symmetry number.

The point group is found by an independent implementation (not a port of xtb's GPL-licensed
`symmetry_i.c`): candidate axes and planes from the principal axes and from pairs and triangles
of equivalent atoms, each refined to the best fit of its atom mapping and accepted when every
atom lands within the tolerance. It agrees with xtb wherever a structure is symmetric well
within the tolerance; at its edge the two can differ.

docs/validation.md compares the analysis with `xtb --hess` and `xtb thermo`.

## Molden files

The converged orbitals of any method, as a Molden file for orbital viewers (Molden, Jmol,
Avogadro, IboView, Multiwfn, PySCF/IOData readers):

- Python: `axtb.singlepoint(..., molden=True)` or `Calculator.singlepoint(..., molden=True)`
  returns the file's text under `"molden"` (`molden="cartesian"` for Cartesian functions);
- CLI: `axtb mol.xyz [--method M] --molden [FILE]` writes FILE (default `molden.input`, the
  name `xtb --molden` uses; `--molden=FILE` when FILE could be taken for the structure),
  `--molden-cartesian` with Cartesian functions; JSON requests take
  `"molden": true` (the text in the response), `"molden_file"` and `"molden_cartesian"`;
- qccodec: keyword `molden` (`True`, `"spherical"` or `"cartesian"`) puts the file into
  `output.results.files["molden.input"]`, for every calculation type;
- Rust: `Want::orbitals` keeps them in `Electronic::orbitals` of the single point, and
  `axtb::ToMolden::to_molden` writes the file.

The file follows the Molden conventions standard readers use: `[Atoms] AU` with atomic
numbers; `[GTO]` atom by atom, each atom with its own contracted shells (g-xTB's q-vSZP
contractions depend on every atom's charge and coordination), the coefficients those of
normalized primitives with normalized contractions; spherical functions flagged `[7F]`/`[5D]`
(so that readers taking `[7F]` as "6D and 7F", as the format description does, and readers
that set one flag per shell type both read 5D and 7F) in Molden's order (p x, y, z; then
m = 0, +1, -1, +2, -2, ...; axtb's real solid harmonics have the standard signs, so only the
order changes), or Cartesian functions (`[6D]`/`[10F]`, Molden's order, each component
normalized); `[MO]` with `Sym=`, `Ene=` (Eh), `Spin=`, `Occup=`. Closed shells are one set of
orbitals with occupations up to 2. Open shells are an alpha and a beta set: g-xTB's two spin
channels, and for GFN2-xTB its one set of spatial orbitals twice, with the alpha and beta
occupations (`xtb --molden` writes it once with the summed occupations).

The programs' own files use other conventions. All three write Cartesian functions (without
flags, Molden's default) and the coefficients of unnormalized primitives. `xtb` (GFN2-xTB)
normalizes every Cartesian component on its own; tblite (`xtb --gxtb`) gives all components
of a shell the norm of its x^l component and writes the valence charge in the atomic-number
column (O as 6); g-xTB 1.1.0 writes d functions as xtb does and f functions as tblite does,
in xtb's internal order (xxx, yyy, zzz, xxy, xxz, xyy, yyz, xzz, yzz, xyz) instead of
Molden's.

## Bond orders, orbital levels, gap and quadrupole

What the programs print after every single point, for every SCF method (GFN2-xTB, g-xTB v1.1
and 2.0.1, the g-xTB part of the solvation composite):

| output | Python result | CLI JSON | qccodec | Rust |
|---|---|---|---|---|
| Mayer–Wiberg bond orders (N x N), on request | `singlepoint(..., bond_orders=True)` → `bond_orders` | `--wbo` (`"bond_orders": true`) → `bond_orders` | keyword `bond_orders` → `extras['bond_orders']` | `Want::bond_orders` → `Electronic::bond_orders` |
| orbital energies and occupations per spin | `orbital_energies_alpha/_beta`, `occupations_alpha/_beta` | the same | `wavefunction.scf_eigenvalues_a/b`, `scf_occupations_a/b`, `calcinfo_nbasis/nmo/nalpha/nbeta/natoms` | `Electronic::levels` |
| HOMO–LUMO gap (Eh) | `homo_lumo_gap` | the same | `extras['homo_lumo_gap']` | `Electronic::homo_lumo_gap` |
| molecular quadrupole (xx, xy, yy, xz, yz, zz; e·bohr², about the origin) | `quadrupole` | the same | `extras['quadrupole']` | `Electronic::quadrupole` |

qcdata has fields for the orbital levels but none for bond orders or quadrupoles. Bond orders cost
one matrix product per spin and are computed only when asked for; the levels, gap and
quadrupole come from what the SCF has anyway.
A closed shell's levels are its orbitals for both spins with half the occupations; GFN2-xTB's
open shells are one set of orbitals with the alpha and beta occupations, g-xTB's the orbitals
of its two spin channels.

What each program prints and what the conventions reproduce (`axtb::properties`):

- **Bond orders**: B_AB = Σ_{μ∈A,ν∈B} (PS)_μν (PS)_νμ with the total density of a closed shell,
  2 Σ_σ Σ (P^σ S)_μν (P^σ S)_νμ for open shells, in every program (xtb `get_wiberg`/
  `get_unrestricted_wiberg`, tblite `get_mayer_bond_orders`, the 1.1.0 program's
  "(PS)*(SP)"). xtb (6.7.1 and 6.8.0) sets pairs 10 bohr or more apart to zero, so the `xtb` and
  `xtb-6.7.1` conventions do too. The programs' `wbo` files list the pairs above 0.1; the 1.1.0
  program prints pairs above 0.05 with 3 decimals, and only for small molecules (not for
  caffeine; `-wrwbo` changes nothing). The 2.0.1 printout labels atoms with the wrong symbols
  (`mol%sym(mol%id)`: water's O appears as "H"); its `wbo` file is right.
- **Gap**: xtb and tblite print the alpha channel's gap (the 2.0.1 program and the 1.1.0 program
  also for open shells, "alpha->alpha"), which the program conventions report; the axtb
  conventions report the lowest unoccupied minus the highest occupied level of both spins. The
  HOMO of a channel is its Aufbau level, independent of smearing.
- **Quadrupole**: xtb's GFN2-xTB printout ("molecular quadrupole (traceless)", `molmom`) is the
  traceless tensor Σ_A [θ_A + 3/2 (q r r + r d + d r) − 1/2 (q r² + 2 r·d) 1] in the convention of
  the atomic quadrupoles θ_A. tblite's (`--quadrupole`, which the 2.0.1 program and
  `xtb --tblite` print) takes the off-diagonal charge and dipole parts twice, 3 instead of 3/2,
  so its "total" is not a tensor: the printed eigenvalues for water are −1.354/−0.570/1.924 and,
  for the same water rotated by 45°, −2.070/−0.196/2.265 (xtb's: −1.371/0.009/1.361 both). The
  `tblite` conventions and the g-xTB 2.0.1 `gxtb` conventions reproduce tblite's form; every other
  case is the tensor.

docs/validation.md has the agreement with each program's printout and files.

## Checks

`scripts/check.sh fast` needs no external program: `cargo fmt --check`, clippy and rustdoc with
warnings as errors, the workspace tests (all reference data is committed), the LGPL-only build,
a wheel installed in a fresh venv with the Python tests, a grep of the tracked files for
home paths, private addresses and host names, a check that the crates' copies of the license
files match the top level, and a check that every Rust and Python source file of the crates and
of `scripts/` starts with `SPDX-License-Identifier: LGPL-3.0-or-later`. `scripts/check.sh
programs` runs the validation gates against the reference programs; a gate whose programs are
not set is reported as skipped, and every gate writes its results under
`target/check/programs/`, never into the repository (`scripts/check.sh list` prints the gates
and their limits). `tests/programs.py` finds the
programs from one variable each and nothing else:

| program | variable | source |
|---|---|---|
| xtb 6.8.0 | `AXTB_XTB680` | the grimme-lab/xtb v6.8.0 release binary (`xtb-6.8.0-linux-x86_64-ifx-mkl.tar.xz`) |
| the g-xTB 2.0.1 build of xtb (xtb 6.7.1, 26dd68d) | `AXTB_GXTB2` | `bin/xtb` of the grimme-lab/g-xtb 2.0.1 release |
| the xtb 6.7.1 release (edcfbbe) | `AXTB_XTB671` | the grimme-lab/xtb v6.7.1 release binary (also shipped with ORCA 6.0) |
| the g-xTB 1.1.0 program | `AXTB_GXTB11_DIR` | a directory with `gxtb`, `gxtb_parameters`, `basisq` and `eeq` of the grimme-lab/g-xtb v1.1.0 release |
| tblite 0.8.0, dftd4 4.3.1 | — | PyPI (`scripts/check.sh` installs them in its venv) |
| xtb's source (atomic masses) | `AXTB_REFERENCE` | `reference/xtb-gxtb`, see `reference/README.md` |

What each check needs and when it fails. Every gate of the programs tier exits with status 1
when a measured quantity exceeds its limit or a calculation fails, except the known differences
that the docs explain (rounding-sensitive SCFs, the programs' own errors). The limits are the
numbers that docs/validation.md and the specs quote, compared at the two significant digits they
give (values the docs do not quote: about 10x the measured round-off, less for finite-difference
errors that do not move with round-off); each script keeps them in `LIMITS` and
`KNOWN_DIFFERENCES` with their sources (`tests/gate.py`), prints every quantity next to its
limit, and `scripts/check.sh list` prints them all.

| check | needs | what | fails when |
|---|---|---|---|
| fast: `cargo test --workspace --release` | nothing | every crate against committed program outputs, finite differences, invariances | a test fails |
| fast: `pytest crates/axtb-py/tests tests/reference_matrix` | the tblite wheel (qccompute for the codec tests) | Python API, CLI and codec; the matrix's fast subset without the xtb cases | a test fails |
| `validate-tblite` | tblite | GFN2-xTB `tblite` conventions vs tblite 0.8.0 on the test structures | an energy differs by more than 1e-7 Eh (tblite's SCF failing on `gh106` known) |
| `validate-xtb`, `validate-xtb-alpb` | xtb 6.8.0 | `xtb` conventions vs xtb on the test structures, gas phase and ALPB water | energies > 5.0e-13 Eh, gradients > 3.6e-14 Eh/bohr, charges > 1e-8 e, other SCC iterations (`gh106` known) |
| `validate-hessian` | the g-xTB build, xtb's source | numerical Hessians vs `xtb --hess` | Hessians > 6e-4 Eh/bohr² (xtb's noise), frequencies > 1 cm⁻¹ |
| `matrix-xtb-6.8.0`, `matrix-xtb-6.7.1` | xtb 6.8.0 and the g-xTB build (one compared, the other a start of the diagnoses of different SCF solutions); tblite | the reference matrix (991 cases) against xtb 6.8.0 or 6.7.1 | a check of the matrix fails |
| `pytest-programs` | xtb 6.8.0, the g-xTB build | the pytest cases that need xtb | a test fails |
| `xtb680-changes` | xtb 6.8.0 (the g-xTB build for its 6.7.1 rows) | each 6.7.1 behaviour put back into the 6.8.0 conventions | the 6.8.0 conventions > 4.8e-13 Eh, 3.9e-14 Eh/bohr or other iterations; the 6.7.1 conventions vs the 6.7.1 program > 1e-8 Eh |
| `xtb-hess-acc` | xtb 6.8.0 | what `--acc` and `$hess sccacc` do to `xtb --hess` | an xtb Hessian is further from axtb's analytic one than docs/validation.md quotes for its setting (3.7e-5 to 1.3e-3 Eh/bohr²) |
| `program-differences` | xtb 6.8.0, tblite | xtb vs tblite: solvation states, atom order, 0.1 K | axtb's default conventions vs tblite > 4e-10 Eh, its atom-order dependence > 1e-12 Eh, a program fails |
| `scf-hard` | the g-xTB build, tblite | SCF convergence on hard cases | Broyden fails on more than 20 cases, DIIS on more than 10, `auto` on another than the ten known 0 K cases or differs from Broyden where Broyden converges |
| `vibrational-xtb-6.8.0`, `vibrational-xtb-6.7.1` | xtb 6.8.0 or the g-xTB build | frequencies, IR intensities, thermochemistry vs `xtb --hess` | on the program's Hessian: frequencies > 5.0e-5 cm⁻¹, thermochemistry > 1.4e-10 Eh (6.7.1: 7.7e-11; g-xTB 2.7e-11), other symmetry numbers; axtb's Hessians beyond docs/validation.md |
| `gfnff-xtb-6.8.0`, `gfnff-xtb-6.7.1` | xtb 6.8.0 or the g-xTB build | GFN-FF topology, energies, gradients, Hessians (not the scaling timings) | a structure not reproduced (D5h ferrocenium known), energies > 7.0e-13 Eh (6.7.1: 8.4e-13), gradients > 4.3e-12 Eh/bohr (9.3e-12), the Hessian and topology rows of docs/validation.md |
| `gxtb-reproduction`, `gxtb2-reproduction`, `gxtb-etemp`, `gxtb-energy-differences` | nothing (committed program outputs) | g-xTB v1.1 and 2.0.1 in both conventions | v1.1 energies > 1e-11 Eh; 2.0.1 energies > 2.1e-12 Eh ((H2O)100 9.1e-12), terms, gradients and default conventions beyond the spec's tables, other iterations; at 300-5000 K as the spec (rounding-sensitive runs known); energy differences moved by the default conventions > 1.1e-5 Eh (v1.1), 4e-11 Eh (2.0.1) |
| `gxtb-elements` | the g-xTB build and/or the g-xTB 1.1.0 program | every element with parameters | a molecule is not `ok` and not one of the known ones (spec §5.1 and §12), energies of the `ok` ones > 2.8e-12 Eh (v1.1) / 4.9e-11 Eh (2.0.1), gradients > 2.3e-10 Eh/bohr (2.0.1) |
| `gxtb-parity` | any of the four programs (each its own cases) | bond orders, gap, quadrupole, restarts | bond orders and gaps beyond docs/validation.md's table, quadrupoles beyond the printed digits, restarted energies > 4.3e-11 Eh or other iterations |
| `solvation-composite` | the g-xTB build | g-xTB in implicit solvent vs the three programs | energies > 9.7e-11 Eh, gradients > 5.3e-10 Eh/bohr, the Hessian > 5.6e-8 Eh/bohr² from finite differences |
| `molden` | the g-xTB build and/or the g-xTB 1.1.0 program | Molden files vs the programs' | orbitals, occupied space and orbital energies beyond docs/validation.md's table, axtb's files orthonormal to worse than 6e-11 |
| `ts-example` | xtb's source (the g-xTB build for its `xtb --hess` row) | an SN2 saddle point with analytic Hessians | no saddle point, not one imaginary mode, frequencies of the other Hessians off the analytic one's (`xtb --hess` by more than 1 cm⁻¹) |

The benchmarks (`bench/`, timings) are not checks; `bench/README.md` describes them.

## Releases

Two GitHub Actions workflows (Linux x86-64):

- `.github/workflows/ci.yml`, on every push and pull request, on the newest stable Rust:
  `scripts/check.sh fast` and `cargo package` of the published crates, plus a build check on
  Rust 1.89, the minimum supported version (`rust-version`).
- `.github/workflows/release.yml`, on a tag `vX.Y.Z` that matches the workspace version, or by
  hand (then it builds only, unless its `publish` input is set): the manylinux_2_28 wheel (abi3,
  x86-64-v3, stripped) and the sdist with maturin, checked by installing each and running a
  calculation; the stripped CLI (`cargo build --profile dist`, built in the manylinux_2_28
  container so that it needs glibc 2.28 or newer) as `axtb-X.Y.Z-x86_64-linux.tar.gz` with the
  license files and the README. On a tag it creates the GitHub release with these files,
  publishes the wheel and the sdist to PyPI and runs `cargo publish` for `axtb-d4`,
  `axtb-solvation`, `axtb`, `axtb-gfnff`, `axtb-gxtb`, `axtb-driver` and `axtb-cli`, in that
  order (a rerun skips the crates already published).

Before the first release the maintainers set up:

1. **PyPI trusted publishing.** On pypi.org, a pending trusted publisher for the project `axtb`
   (the account's "Publishing" page): owner `atomsforhumanity`, repository `axtb`, workflow
   `release.yml`, environment `pypi`. In the repository's settings, an environment named `pypi`
   (required reviewers on it make every upload wait for an approval). No PyPI token is stored.
2. **crates.io token.** A crates.io API token with the scopes `publish-new` and
   `publish-update`, stored as the repository secret `CARGO_REGISTRY_TOKEN`. The account behind
   the token becomes the owner of each new crate; `cargo owner --add` adds others.
3. **Privacy pattern.** The repository secret `AXTB_PRIVATE_PATTERN`: an extended regular
   expression of the site's host names, which CI adds to the privacy grep of `scripts/check.sh
   fast`. Without it the grep runs its built-in patterns only.

A release: bump the version in the root `Cargo.toml` (`[workspace.package]` and the versions in
`[workspace.dependencies]`) and `Cargo.lock`, add the CHANGELOG entry, follow the release
procedure in `bench/README.md`, then push the tag `vX.Y.Z`. Builds made elsewhere embed their
build paths (panic messages, maturin's SBOM), so the released files come from the workflow.

## Layout

```
crates/base        `axtb-base`: the shared foundation (molecules, errors, units, threads, linear algebra, EEQ)
crates/axtb        core library: basis, integrals, Hamiltonian, electrostatics, SCF, gradients, Hessians
crates/d4          DFT-D4 dispersion (bit-identical to dftd4 4.3.1)
crates/solvation   ALPB/GBSA implicit solvation (xtb 6.8.0 and 6.7.1 compatible)
crates/gxtb        g-xTB v1.1 and 2.0.1, with their parameters (NOTICE)
crates/gfnff       GFN-FF force field (port of xtb's src/gfnff; spec in docs/gfnff-spec.md)
crates/driver      `axtb-driver`: the request/response layer of the front ends (checks, dispatch, response)
crates/axtb-cli    `axtb` command-line driver (JSON in/out)
crates/axtb-py     Python bindings (`import axtb`) and the qccodec codec
tests/             validation scripts against the reference programs, and structures
scripts/check.sh   the checks (fast and programs tiers)
docs/              specifications of the ports, validation, g-xTB feature coverage
bench/             accuracy/performance comparison suite and report
reference/         upstream sources and programs (git-ignored; see reference/README.md)
```

## License

Every crate of axtb is LGPL-3.0-or-later (`COPYING.LESSER` with `COPYING`), and so are the
`axtb` program and the Python package and its wheels. The crates reproduce xtb, tblite and
dftd4 (LGPL-3.0-or-later) and data of mctc-lib and multicharge (Apache-2.0). `axtb-gxtb` embeds
the g-xTB parameters: g-xTB itself is GPL-3.0 (grimme-lab/g-xtb), and its authors gave explicit
permission to include them and to distribute axtb's g-xTB implementation under
LGPL-3.0-or-later. [LICENSE](LICENSE) lists the crates, [NOTICE](NOTICE) quotes the permission
and credits the upstream work that axtb reproduces, and every published crate and wheel carries
the license texts it needs.
