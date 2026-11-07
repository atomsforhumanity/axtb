# axtb-driver

The request/response layer of the [axtb](https://github.com/atomsforhumanity/axtb) front ends:
the command-line program ([`axtb-cli`](https://crates.io/crates/axtb-cli)) and the Python
package (`axtb` on PyPI). A `Request` names the method (GFN2-xTB, g-xTB v1.1 or 2.0.1,
GFN-FF), the structure and the settings; `run` checks it, dispatches it, and returns a
`Response` with the same fields for every front end, including Hessians and their vibrational
analysis. A stateful `Calculator` runs one geometry after another.

Features: `gxtb` (default) adds g-xTB from [`axtb-gxtb`](https://crates.io/crates/axtb-gxtb).

License: GPL-3.0-only (`COPYING`) as built by default, because the feature `gxtb` includes the
GPL crate axtb-gxtb. Built with `--no-default-features` it contains no g-xTB and is
LGPL-3.0-or-later (`COPYING.LESSER`). `NOTICE` credits the upstream work.
