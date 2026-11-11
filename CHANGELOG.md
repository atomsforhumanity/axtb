# Changelog

All notable changes to axtb. Versions follow [Semantic Versioning](https://semver.org); the
Rust crates, the CLI and the Python package share one version.

## Unreleased

- g-xTB 2.0.1 analytic Hessians: the response terms in a symmetric form
  (`AnalyticOptions::symmetric`, on by default) whose error is quadratic in the response
  solver's tolerance, and a default tolerance of 1e-6 (was 1e-10). Over 15 cases (closed and
  open shells, 0-5000 K, both conventions) the Hessians equal the 0.1.x ones to ≤ 2.2e-11
  Eh/bohr² (vibrational frequencies ≤ 2.5e-7 cm⁻¹), and they take 1.33-1.37x less time for taxol
  (1 and 8 threads) and (H2O)100 (8 threads) on the laptop (`tests/gxtb2_hessian_symmetric.py`).
  Memory: + 2 · 3N · n_occ · n_virt · 4 bytes (taxol +53 MB, (H2O)100 +0.67 GB). The reported
  `hessian_max_asymmetry` of these Hessians is now ~1e-7, removed by the symmetrization.

## 0.1.1 (2026-11-10)

- The minimum supported Rust is 1.89 for every crate (`rust-version`; 0.1.0 declared 1.99, which
  made Cargo refuse the crates in projects that support older compilers). CI checks the build on
  1.89. Nothing computed changes: the results and timings in this repository are those of 0.1.0.

## 0.1.0 (2026-11-07)

The first release: the Rust crates `axtb`, `axtb-d4`, `axtb-solvation`, `axtb-gfnff`,
`axtb-gxtb`, `axtb-driver` and `axtb-cli` (the `axtb` program, JSON in and out) on crates.io,
the Python package `axtb` on PyPI, and its codec for qccodec/qccompute. The README describes
the methods, conventions and settings; [docs/validation.md](docs/validation.md) has the
agreement with each program and the test sets, and [bench/results](bench/results) the result
files they come from.

### Methods, and the program each convention reproduces

Every method has axtb's default conventions (`axtb`) and the conventions of the program it
reproduces, which give that program's numbers to the last digits, quirks included.

- **GFN2-xTB** (`axtb`): energies, analytic gradients, Hessians (analytic, from the
  coupled-perturbed SCC response, by default; response and numerical), self-consistent D4
  dispersion (`axtb-d4`, bit-identical to dftd4 4.3.1), multipole electrostatics.
  Conventions `xtb`: the xtb 6.8.0 program (4b8c987); `xtb-6.7.1`: xtb 6.7.1 as the g-xTB 2.0.1
  build of xtb has it (26dd68d; the 6.7.1 release edcfbbe differs in some GFN2-xTB gradients,
  fixed upstream before that build); `tblite`: tblite 0.8.0. In `xtb` and `xtb-6.7.1` the SCF
  is the program's own SCC loop and Broyden mixer.
- **ALPB/GBSA implicit solvation** for GFN2-xTB (`axtb-solvation`): 25 ALPB and 14 GBSA
  solvents, the three reference states; xtb 6.8.0 and 6.7.1, and tblite 0.8.0 in `tblite`
  conventions.
- **g-xTB v1.1** (`axtb-gxtb`, method `gxtb-v1.1`), the public preprint version: energies,
  analytic gradients (which the program does not have), semi-numerical Hessians, finite
  electronic temperature. Conventions `gxtb`: the g-xTB 1.1.0 program (grimme-lab/g-xtb
  release v1.1.0).
- **g-xTB 2.0.1** (method `gxtb-v2.0.1`): energies, analytic gradients, analytic (default),
  response and numerical Hessians, finite electronic temperature, all 103 elements H–Lr.
  Conventions `gxtb`: the g-xTB 2.0.1 program (`xtb --gxtb` of the grimme-lab/g-xtb 2.0.1
  release). Its source is not public; [docs/gxtb-v2.0.1-spec.md](docs/gxtb-v2.0.1-spec.md)
  says how the method was reproduced from the released program.
- **g-xTB in implicit solvent**: the composite E_gxtb + [E_GFN2+ALPB/GBSA − E_GFN2] for
  energies, gradients and Hessians in one process (`CompositeCalculator`), reproducing the
  three xtb program runs it replaces.
- **GFN-FF** (`axtb-gfnff`, method `gfnff`): topology for every element H–Lr, energies,
  analytic gradients, analytic (default) and numerical Hessians with dipole derivatives.
  Conventions `xtb`: xtb 6.8.0 (`--gfnff`); `xtb-6.7.1`: xtb 6.7.1.
- **Vibrational analysis** of any Hessian: frequencies, normal modes, IR intensities and the
  quasi-RRHO thermochemistry. Conventions `xtb`: `xtb --hess` and `xtb thermo` of 6.8.0;
  `xtb-6.7.1`: those of 6.7.1.
- Also: Molden files of the converged orbitals; restarts from the previous geometry's
  wavefunction; Mayer–Wiberg bond orders, orbital energies, HOMO–LUMO gap, molecular
  quadrupole.

### Default conventions versus the program modes

The default `axtb` conventions use CODATA 2022 constants and correct what the programs get
wrong; the program modes keep it.

- GFN2-xTB: double-precision tables, one coordination number for energy and gradient, smooth
  long-range cutoffs, the solvation fixes of grimme-lab/xtb#1504 and #1505, and DIIS when
  Broyden mixing does not converge. For ordinary molecules the modes differ by about 1e-8 Eh.
- g-xTB v1.1: the exact derivative of the energy as the Fock matrix, so the SCF is stationary
  (the program's Fock matrix leaves out two potential terms).
- g-xTB 2.0.1: an integer electron count and smooth cutoffs; the same energy expression and
  SCF as the program.
- Vibrational analysis: standard atomic masses, rotations and translations projected in
  mass-weighted coordinates, always 3N−6 (3N−5) modes, the point group's exact symmetry
  number.

### Licenses

- `axtb`, `axtb-d4`, `axtb-solvation` and `axtb-gfnff`: LGPL-3.0-or-later.
- `axtb-gxtb`: GPL-3.0-only, because it embeds the g-xTB parameters (grimme-lab/g-xtb,
  GPL-3.0).
- `axtb-driver`, `axtb-cli` and the Python package build g-xTB in by default and are
  GPL-3.0-only as built. With `--no-default-features` they leave g-xTB out, and the build is
  LGPL-3.0-or-later: `cargo build -p axtb-cli --no-default-features`,
  `maturin build --no-default-features -m crates/axtb-py/Cargo.toml`.
- [LICENSE](LICENSE) explains the split; [NOTICE](NOTICE) credits the upstream work.

### Requirements

- Rust 1.89 or newer to build.
- x86-64 Linux is the developed and tested platform. The builds from this repository, the
  release binary and the wheels target x86-64-v3 (AVX2, FMA; Intel Haswell and AMD Excavator
  or newer). The CLI stops with an error and `import axtb` raises `ImportError` on a CPU
  without these instructions; `RUSTFLAGS="-C target-cpu=x86-64"` builds for any x86-64 CPU,
  with bit-identical results.
- Wheels: manylinux_2_28 (glibc 2.28 or newer), Python 3.10 or newer (abi3). The release
  binary needs glibc 2.28 or newer.

### Known limits

- No periodic systems; molecules only.
- No GFN1-xTB.
- No implicit solvation for GFN-FF.
- g-xTB has no implicit solvent of its own: only through the composite with GFN2-xTB's
  ALPB/GBSA.
- No GPU: the eigensolver runs on the CPU (faer, or a LAPACK library loaded at run time).
- No PDB residue charges.

### Program issues found and reported upstream

- [grimme-lab/xtb#1504](https://github.com/grimme-lab/xtb/issues/1504): a stale variable in
  the Born-radius integral makes ALPB/GBSA energies depend on the atom order (fixed in xtb
  6.8.0); the same in tblite,
  [tblite/tblite#430](https://github.com/tblite/tblite/issues/430).
- [grimme-lab/xtb#1505](https://github.com/grimme-lab/xtb/issues/1505): the ALPB/GBSA
  `reference` state used the wrong solvent density (fixed in xtb 6.8.0).
- [grimme-lab/xtb#1506](https://github.com/grimme-lab/xtb/issues/1506): single-precision
  literals in the D4 coordination number, and a small-gap Broyden damping that never ran
  (fixed and removed in xtb 6.8.0).
- [grimme-lab/xtb#1515](https://github.com/grimme-lab/xtb/issues/1515): GFN-FF allocates a
  hydrogen-bond array that grows as N³ in every energy evaluation (still in xtb 6.8.0).
- [grimme-lab/xtb#1516](https://github.com/grimme-lab/xtb/issues/1516): GFN-FF's carbonyl
  hydrogen-bond energy is not smooth where C=O···H is linear, and xtb 6.7.1's `--hess`
  crashes there (still in xtb 6.8.0).
- [sarah-quinones/gemm#53](https://github.com/sarah-quinones/gemm/issues/53) and
  [sarah-quinones/gemm-x64-v2#8](https://github.com/sarah-quinones/gemm-x64-v2/issues/8): the
  faer dependencies gemm-common and private-gemm-x86 open every CPU's cache files at
  start-up. This repository builds patched copies (`third_party/`); crates.io builds of the
  axtb crates do not get the patch.

docs/validation.md lists the other program behaviours found on the way, which are documented
but not reported.
