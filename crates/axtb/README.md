# axtb

GFN2-xTB in Rust: energies, analytic gradients and Hessians (analytic, response and numerical),
self-consistent D4 dispersion, multipole electrostatics, ALPB/GBSA implicit solvation, Molden
files, bond orders and vibrational analysis with thermochemistry. Molecules only (no periodic
systems).

In a reference program's conventions it reproduces that program's numbers to the last digits:
the xtb 6.8.0 and 6.7.1 programs and tblite 0.8.0. The default `axtb` conventions use CODATA
2022 constants and correct the program bugs found on the way. Every calculation runs on the
threads it is given (default 1), with no implicit threading.

This crate also holds what the other methods of the
[axtb workspace](https://github.com/atomsforhumanity/axtb) share (molecule, basis, integrals,
eigensolver, Molden files, vibrational analysis). g-xTB is
[`axtb-gxtb`](https://crates.io/crates/axtb-gxtb), GFN-FF
[`axtb-gfnff`](https://crates.io/crates/axtb-gfnff), and
[`axtb-driver`](https://crates.io/crates/axtb-driver) is the request/response layer of the
command-line program and the Python package.

The repository's README describes the methods, conventions and settings;
[docs/validation.md](https://github.com/atomsforhumanity/axtb/blob/main/docs/validation.md) has
the agreement with each program.

Builds use pure-Rust linear algebra (faer): no Fortran compiler, BLAS or LAPACK. The
repository's own builds target x86-64-v3 (AVX2, FMA); a crate built from crates.io uses your
project's target settings.

License: LGPL-3.0-or-later (`COPYING.LESSER`, with `COPYING`). `NOTICE` credits the upstream
work (xtb, tblite, dftd4, mctc-lib, multicharge) whose parameters and algorithms it reproduces.
