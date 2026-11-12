# axtb-base

The foundation every [axtb](https://github.com/atomsforhumanity/axtb) crate builds on: the
molecule and its checks, the one error type, the `Method` trait every calculator implements
(single points, Hessians, wavefunction restarts, one result type for every method, the retry
ladders and the generic composite of methods), the orbital data of the Molden files, physical
constants and the reference programs'
unit factors, the element tables several crates share, the error function the reference
programs use, small 3-vector helpers, coordination numbers (the counting functions of GFN2-xTB,
dftd4 and GFN-FF with their derivatives), orbital occupations (Aufbau, tblite's and xtb's Fermi
smearing, tblite's entropy term), phase timers, the calculators' thread pools with the
deterministic parallel loops (fixed chunks, results in order) and the finite differences of
every method (stencils, symmetrization, the numerical Hessian). Its only dependency is rayon,
so the dispersion, solvation and force-field crates can share it with the tight-binding
framework instead of keeping copies.

The feature `linalg` (off by default) adds the dense linear algebra of the methods on
[faer](https://crates.io/crates/faer): the SCF eigensolver with its optional LAPACK library
loaded at run time (`AXTB_LAPACK`), density matrices and linear solves, and the bordered linear
system of the electronegativity equilibration (EEQ) charge models with its constraint rows and
solves.

License: LGPL-3.0-or-later (`COPYING.LESSER`, with `COPYING`). `NOTICE` credits the upstream
work.
