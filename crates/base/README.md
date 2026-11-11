# axtb-base

The foundation every [axtb](https://github.com/atomsforhumanity/axtb) crate builds on: the
molecule and its checks, the one error type, physical constants and the reference programs'
unit factors, the error function the reference programs use, small 3-vector helpers and phase
timers. It has no dependencies, so the dispersion, solvation and force-field crates can share it
with the tight-binding framework instead of keeping copies.

License: LGPL-3.0-or-later (`COPYING.LESSER`, with `COPYING`). `NOTICE` credits the upstream
work.
