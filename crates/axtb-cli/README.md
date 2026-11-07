# axtb-cli

`axtb`, the command-line program of [axtb](https://github.com/atomsforhumanity/axtb): GFN2-xTB,
g-xTB (v1.1 and 2.0.1) and GFN-FF energies, gradients and Hessians with vibrational analysis,
one calculation per call, the result as JSON on stdout.

```bash
cargo install axtb-cli
axtb caffeine.xyz --method gfn2 --grad
axtb caffeine.xyz --method gxtb-v2.0.1 --hess
axtb --help
```

Each method has the conventions of the program it reproduces (xtb 6.8.0 and 6.7.1, tblite
0.8.0, the g-xTB 1.1.0 and 2.0.1 programs) and axtb's default conventions; the repository's
README describes them.

`cargo install` builds for your CPU's baseline target. The repository's own builds target
x86-64-v3 (AVX2, FMA) and carry a patch of two dependencies that speeds up start-up on machines
with many CPUs; cargo does not apply either to crates.io installs (the repository's README,
"Install", and `third_party/README.md`).

License: GPL-3.0-only (`COPYING`) as built by default, which includes g-xTB (feature `gxtb`,
the GPL crate axtb-gxtb). `cargo install axtb-cli --no-default-features` builds a program
without g-xTB that is LGPL-3.0-or-later (`COPYING.LESSER`). `NOTICE` credits the upstream work.
