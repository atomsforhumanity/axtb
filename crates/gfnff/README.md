# axtb-gfnff

The GFN-FF force field (S. Spicher, S. Grimme, Angew. Chem. Int. Ed. 59, 15665 (2020),
doi:10.1002/anie.202004239) in Rust: a port of xtb's implementation (`src/gfnff`) with the
topology setup for every element H–Lr, energies, analytic gradients, EEQ charges, and analytic
and numerical Hessians with dipole derivatives. It reproduces the xtb 6.8.0 program (default)
and the xtb 6.7.1 program (`Conventions`). Molecules in the gas phase only (no periodic
systems, no implicit solvation).

Part of [axtb](https://github.com/atomsforhumanity/axtb);
[docs/gfnff-spec.md](https://github.com/atomsforhumanity/axtb/blob/main/docs/gfnff-spec.md)
maps every term to xtb's code.

License: LGPL-3.0-or-later (`COPYING.LESSER`, with `COPYING`). `NOTICE` credits the upstream
work (xtb).
