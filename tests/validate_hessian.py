#!/usr/bin/env python3
"""Compare axtb's semi-numerical GFN2 Hessian with `xtb --hess` (same 0.005 bohr step) of xtb 6.7.1,
the g-xTB 2.0.1 build (`$AXTB_GXTB2`, see tests/programs.py), whose `hessian` file is projected in
Cartesian coordinates. Needs xtb's atomic masses from its source (xtb-gxtb in `$AXTB_REFERENCE`,
default reference/, see reference/README.md).

xtb's `hessian` file has translations/rotations projected out (non-mass-weighted), so
the same Cartesian projector is applied to axtb's Hessian before comparing. Harmonic
frequencies are compared with xtb's own atomic masses so only the Hessians differ.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_tblite import ROOT, read_xyz  # noqa: E402
from validate_xtb import write_coord  # noqa: E402
import programs  # noqa: E402

REFERENCE = Path(os.environ.get("AXTB_REFERENCE", ROOT / "reference"))

AMU_TO_ME = 1.0 / 5.485799090441e-4      # CODATA 2022 electron mass in u (qcref)
HARTREE_TO_CM = 2.1947463136314e7 / 100   # CODATA 2022 hartree-inverse meter (qcref)


def xtb_masses() -> np.ndarray:
    src = (REFERENCE / "xtb-gxtb/src/param/atomicmass.f90").read_text()
    body = src[src.index("atomicMass(1:118)"):]
    body = body[body.index("[") + 1:body.index("]")]
    return np.array([float(x.replace("_wp", "")) for x in re.findall(r"[\d.]+_wp", body)])


def tr_projector(pos: np.ndarray) -> np.ndarray:
    n = len(pos)
    c = pos - pos.mean(axis=0)
    vecs = []
    for k in range(3):
        t = np.zeros((n, 3)); t[:, k] = 1.0; vecs.append(t.ravel())
    for k in range(3):
        r = np.zeros((n, 3)); e = np.zeros(3); e[k] = 1.0
        r[:] = np.cross(e, c); vecs.append(r.ravel())
    q, _ = np.linalg.qr(np.array(vecs).T)
    q = q[:, np.abs(np.diag(_)) > 1e-8]
    return np.eye(3 * n) - q @ q.T


def frequencies(h: np.ndarray, nums, masses) -> np.ndarray:
    m = np.repeat(masses[nums - 1] * AMU_TO_ME, 3)
    hm = h / np.sqrt(np.outer(m, m))
    w = np.linalg.eigvalsh(hm)
    return np.sign(w) * np.sqrt(np.abs(w)) * HARTREE_TO_CM


def main() -> int:
    import axtb
    xtb = programs.require(programs.GXTB2)[0]
    if not (REFERENCE / "xtb-gxtb/src/param/atomicmass.f90").exists():
        programs.skip("needs the xtb source in $AXTB_REFERENCE/xtb-gxtb (reference/README.md) for its atomic masses")
    masses = xtb_masses()
    files = [Path(f) for f in sys.argv[1:]] or [ROOT / "tests/structures/water.xyz", ROOT / "tests/structures/caffeine.xyz"]
    for f in files:
        nums, pos, charge = read_xyz(f)
        with tempfile.TemporaryDirectory(dir=os.environ.get("AXTB_SCRATCH")) as d:
            write_coord(Path(d) / "coord", nums, pos)
            t = time.perf_counter()
            out = subprocess.run([xtb, "coord", "--gfn", "2", "--hess", "--acc", "0.01", "--chrg", str(charge)],
                                 cwd=d, capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="1"))
            txtb = time.perf_counter() - t
            vals = [float(x) for x in (Path(d) / "hessian").read_text().split()[1:]]
        n3 = 3 * len(nums)
        hx = np.array(vals).reshape(n3, n3)
        calc = axtb.Calculator("gfn2", nums, pos, charge=float(charge))
        t = time.perf_counter()
        h = calc.hessian(pos, step=0.005, accuracy=0.01, method="numerical")["hessian"]
        tax = time.perf_counter() - t
        p = tr_projector(pos)
        hp = p @ h @ p
        fx, fa = frequencies(hx, nums, masses), frequencies(hp, nums, masses)
        print(f"{f.stem}: max|dH| {np.abs(hp - hx).max():.2e} Eh/bohr², max|dfreq| (>50 cm-1) "
              f"{np.abs(fa - fx)[np.abs(fx) > 50].max():.3f} cm-1, xtb {txtb:.2f} s, axtb {tax:.2f} s (1 thread)")
        print("   lowest freqs xtb :", np.round(np.sort(fx)[6:10], 2))
        print("   lowest freqs axtb:", np.round(np.sort(fa)[6:10], 2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
