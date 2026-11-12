#!/usr/bin/env python3
"""Transition-state example: SN2 Cl- + CH3Cl -> ClCH3 + Cl- at GFN2-xTB.

From an asymmetric guess, a saddle point is located by eigenvector following
(Newton steps on |H| in the eigenbasis of the projected Hessian, uphill along the
lowest mode, trust radius 0.1 bohr) with the analytic Hessian recomputed every step.
At the converged structure the axtb Hessian methods and `xtb --hess --acc 0.001` are
compared: number of imaginary modes and the imaginary frequency.

Usage: python bench/ts_example.py   (AXTB_BIN, AXTB_REFERENCE and AXTB_GXTB2 as for hessian_benchmark.py)

It fails (exit status 1) when the search does not converge, a Hessian has more or fewer than one
imaginary mode, or the frequencies differ from the analytic Hessian's by more than `LIMITS`
(tests/gate.py).
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hessian_benchmark import AXTB, XTB, frequencies, programs, run_xtb, tr_projector, xtb_masses  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import gate  # noqa: E402

AATOAU = 1.0 / 0.529177210903
# max |dfreq| (cm-1, modes above 10 cm-1 in magnitude) against the analytic Hessian. Chosen from
# the measured values: the analytic Hessian in xtb conventions 1.5e-7 (limit 2e-6), the response
# Hessian 6.2e-4 and the numerical one 0.062 (their finite-difference errors, which do not move
# with round-off: limits 2e-3 and 0.2); `xtb --hess` within 1 cm-1, its finite-difference noise
# (docs/validation.md, Summary)
LIMITS = {"search not converged": 0, "Hessians without exactly one imaginary mode": 0,
          "axtb response": 2e-3, "axtb numerical": 0.2, "axtb analytic (xtb conventions)": 2e-6,
          "xtb --hess --acc 0.001": 1.0}
SYMBOLS = {6: "C", 1: "H", 17: "Cl"}
# asymmetric guess (Angstrom): C-Cl 2.25 / 2.45, slightly pyramidal methyl
GUESS = [
    (6, [0.0, 0.0, 0.0]),
    (17, [0.0, 0.0, 2.25]),
    (17, [0.05, 0.0, -2.45]),
    (1, [1.07, 0.0, 0.08]),
    (1, [-0.535, 0.927, 0.08]),
    (1, [-0.535, -0.927, 0.08]),
]
CHARGE = -1


def axtb(nums, pos, method="analytic", conventions="axtb", **kw):
    req = {"numbers": nums, "positions": pos.tolist(), "charge": CHARGE, "uhf": 0, "gradient": True,
           "hessian": True, "hessian_method": method, "conventions": conventions, "accuracy": 1e-3}
    req.update(kw)
    out = subprocess.run([AXTB, "--json"], input=json.dumps(req), capture_output=True, text=True, check=True)
    d = json.loads(out.stdout)
    return d["energy"], np.array(d["gradient"]).ravel(), np.array(d["hessian"]), dict(d["hessian_timings"])["total"]


def main() -> int:
    nums = [z for z, _ in GUESS]
    pos = np.array([p for _, p in GUESS]) * AATOAU
    masses = xtb_masses()
    print("eigenvector following with analytic Hessians")
    converged = False
    for it in range(50):
        e, g, h, t = axtb(nums, pos)
        p = tr_projector(pos)
        w, v = np.linalg.eigh(p @ h @ p)
        keep = np.abs(w) > 1e-6
        w, v = w[keep], v[:, keep]
        f = v.T @ (p @ g)
        gmax = np.abs(p @ g).max()
        nneg = int((w < 0).sum())
        print(f"  {it:2d}  E {e:.10f}  max|g| {gmax:.2e}  negative eigenvalues {nneg}  lowest {w[0]:+.4f}  ({t * 1e3:.1f} ms)")
        if gmax < 1e-6:
            converged = True
            break
        step = np.zeros_like(f)
        step[0] = f[0] / abs(w[0])          # uphill along the lowest mode
        step[1:] = -f[1:] / np.abs(w[1:])   # downhill along all others
        dx = v @ step
        n = np.linalg.norm(dx)
        if n > 0.1:
            dx *= 0.1 / n
        pos = pos + dx.reshape(-1, 3)
    d = [np.linalg.norm(pos[0] - pos[k]) / AATOAU for k in (1, 2)]
    print(f"saddle point: C-Cl {d[0]:.4f} / {d[1]:.4f} A")
    print("\nfrequencies at the saddle point (cm-1; imaginary as negative)")
    rows = []
    for method in ["analytic", "response", "numerical"]:
        _, _, h, t = axtb(nums, pos, method=method)
        rows.append((f"axtb {method}", frequencies(h, nums, pos, masses), t))
    _, _, h, t = axtb(nums, pos, method="analytic", conventions="xtb")
    rows.append(("axtb analytic (xtb conventions)", frequencies(h, nums, pos, masses), t))
    if XTB or programs.note_skip("the xtb --hess row", programs.GXTB2):
        hx, tx = run_xtb(np.array(nums), pos, CHARGE, 0, None, 1, acc=0.001)
        rows.append(("xtb --hess --acc 0.001", frequencies(hx, nums, pos, masses), tx))
    summary = {"search not converged": 0 if converged else 1, "Hessians without exactly one imaginary mode": []}
    ref = rows[0][1]
    for name, fr, t in rows:
        nimag = int((fr < -10.0).sum())
        real = fr[np.abs(fr) > 10.0]
        print(f"  {name:32s} imaginary modes {nimag}: {fr[0]:9.2f}   lowest real {real[real > 0][:3].round(2)}   ({t:.3f} s)")
        if nimag != 1:
            summary["Hessians without exactly one imaginary mode"].append(name)
        if name != rows[0][0]:
            modes = np.abs(ref) > 10.0
            summary[name] = float(np.abs(fr - ref)[modes].max())
    return gate.check(summary, LIMITS)


if __name__ == "__main__":
    sys.exit(main())
