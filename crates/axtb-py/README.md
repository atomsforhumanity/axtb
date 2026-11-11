# axtb

Extended tight binding (GFN2-xTB; g-xTB v1.1 and 2.0.1) and the GFN-FF force field, in-process
from Python: energies, analytic gradients, Hessians with vibrational analysis and
thermochemistry, implicit solvation (ALPB/GBSA for GFN2-xTB, and for g-xTB through a composite),
Molden files, bond orders. Molecules only. The engine is
[axtb](https://github.com/atomsforhumanity/axtb), written in Rust.

In a reference program's conventions axtb reproduces that program's numbers to the last
digits (xtb 6.8.0 and 6.7.1, tblite 0.8.0, the g-xTB 1.1.0 and 2.0.1 programs); its default
conventions use CODATA 2022 constants and correct the program bugs found on the way.
Calculations run on the threads you pass (default 1) and never spawn more, so many can run
side by side.

```python
import axtb
pos = [[0, 0, 0.22], [0, 1.43, -0.89], [0, -1.43, -0.89]]          # bohr, any array-like
r = axtb.singlepoint([8, 1, 1], pos, method="gfn2", gradient=True)
r["energy"], r["gradient"]
calc = axtb.Calculator("gxtb-v2.0.1", [8, 1, 1], pos, threads=4)   # reuses setup and wavefunction
h = calc.hessian(pos)                                              # analytic Hessian and its analysis
h["frequencies"], h["ir_intensities"], h["thermo"]["free_energy"]
```

With `pip install "axtb[qc]"` the package also registers a codec for qccodec, so qccompute
runs axtb like any other program.

**Requirements.** The wheels are for x86-64 Linux with glibc 2.28 or newer and Python 3.10 or
newer, and need an x86-64-v3 CPU (AVX2, FMA: Intel since Haswell, AMD since Excavator).
`import axtb` checks the CPU and raises `ImportError` on one without these instructions. On
other platforms pip builds the package from the source distribution, which needs Rust 1.89 or
newer.

**Documentation.** The [README](https://github.com/atomsforhumanity/axtb#readme) describes the
methods, conventions and settings; [docs/validation.md](https://github.com/atomsforhumanity/axtb/blob/main/docs/validation.md)
has the agreement with each program; the
[changelog](https://github.com/atomsforhumanity/axtb/blob/main/CHANGELOG.md) lists what each
release contains.

**License.** GPL-3.0-only (`COPYING`), because the package includes g-xTB, whose parameters are
GPL-3.0 (grimme-lab/g-xtb). The rest of axtb is LGPL-3.0-or-later (`COPYING.LESSER`), and a
build without g-xTB (`maturin build --no-default-features`) is LGPL-3.0-or-later. `LICENSE`
explains the split; `NOTICE` credits the upstream work (xtb, tblite, dftd4, mctc-lib,
multicharge, g-xTB).
