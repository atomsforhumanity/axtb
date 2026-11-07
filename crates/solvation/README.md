# axtb-solvation

ALPB and GBSA implicit solvation for GFN2-xTB, in Rust: a port of xtb's own solvation code
(`src/solv`), which `xtb --alpb` and `xtb --gbsa` use for GFN2-xTB. 25 ALPB and 14 GBSA
solvents, the three reference states, energies and gradients, reproducing the xtb 6.8.0 and
6.7.1 programs. Molecules only.

Part of [axtb](https://github.com/atomsforhumanity/axtb); the GFN2-xTB calculator in
[`axtb`](https://crates.io/crates/axtb) uses it for `solvation`.

License: LGPL-3.0-or-later (`COPYING.LESSER`, with `COPYING`). `NOTICE` credits the upstream
work (xtb).
