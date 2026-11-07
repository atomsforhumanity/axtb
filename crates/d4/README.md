# axtb-d4

The DFT-D4 dispersion model as GFN2-xTB uses it, in Rust: energies, gradients and the
charge derivatives for a self-consistent D4 term. It matches dftd4 4.3.1 bit for bit and
tblite 0.8.0's `tblite_disp_d4`. Positions in bohr, energies in Hartree; molecules only.

Part of [axtb](https://github.com/atomsforhumanity/axtb), which uses it in GFN2-xTB
([`axtb`](https://crates.io/crates/axtb)) and in g-xTB's revD4
([`axtb-gxtb`](https://crates.io/crates/axtb-gxtb)). The reference tables are generated from
the dftd4 and mctc-lib sources (`src/data.rs` records the commits).

License: LGPL-3.0-or-later (`COPYING.LESSER`, with `COPYING`). `NOTICE` credits the upstream
work (dftd4, tblite, mctc-lib).
