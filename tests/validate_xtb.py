#!/usr/bin/env python3
"""Compare axtb GFN2-xTB with the xtb program (`--xtb`, default xtb 6.8.0 from `$AXTB_XTB680`, see
tests/programs.py): xtb 6.8.0 with axtb's `xtb` conventions, xtb 6.7.1 (also the g-xTB 2.0.1
build, `--xtb "$AXTB_GXTB2"`) with `xtb-6.7.1`.

Usage: python tests/validate_xtb.py [--acc 1e-4] [--xtb PATH] [--alpb water] [structure.xyz ...]

xtb gets the geometry as a Turbomole `coord` file in bohr (so its Å→bohr constant does
not matter), runs `--gfn 2 --grad` in its own scratch directory with OMP_NUM_THREADS=1,
and its energy and gradient are read from the printout and the `gradient` file.

The script fails (exit status 1) when a quantity exceeds its limit in `LIMITS` (tests/gate.py)
or a structure fails, except the known differences.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_tblite import AXTB, ROOT, SYMBOLS, read_xyz  # noqa: E402
import gate  # noqa: E402
import programs  # noqa: E402

# per axtb conventions (that of the xtb version). xtb 6.8.0: docs/validation.md, "xtb's SCC loop"
# (the 13 structures at --acc 1 and 1e-4, gas phase and ALPB water): energies <= 5.0e-13 Eh (xtb
# prints 12 decimals), gradients <= 3.6e-14 Eh/bohr, the same SCC iterations; charges chosen: the
# `charges` file's 8 decimals (5e-9) twice. xtb 6.7.1 (the g-xTB 2.0.1 build): energies <= 1e-8 Eh
# (docs/validation.md, Summary); chosen from the measured values at --acc 1e-4: gradients 6.8e-10
# Eh/bohr -> 1e-8, charges 1.0e-8 e -> 1e-7, the same iterations.
LIMITS = {
    "xtb": {"max_abs_dE": 5.0e-13, "max_dG": 3.6e-14, "max_dq": 1e-8, "iteration_mismatches": 0, "failures": 0},
    "xtb-6.7.1": {"max_abs_dE": 1e-8, "max_dG": 1e-8, "max_dq": 1e-7, "iteration_mismatches": 0, "failures": 0},
}
KNOWN_DIFFERENCES = {
    "gh106": "a pathological SCF with several solutions (docs/validation.md, xtb's SCC loop and SCF convergence): "
             "xtb 6.8.0 takes 193 iterations at --acc 1e-4 (axtb 194) and 161 at --acc 1 (axtb 166, 6.6e-10 Eh), in "
             "ALPB water xtb stops with an error; against 6.7.1 the programs reach different solutions "
             "(tests/reference_matrix/summary-xtb-6.7.1.md)",
}


def write_coord(path: Path, nums, pos):
    lines = ["$coord"] + [f"{x:22.14f} {y:22.14f} {z:22.14f} {SYMBOLS[n - 1].lower()}" for n, (x, y, z) in zip(nums, pos)]
    path.write_text("\n".join(lines + ["$end", ""]))


def xtb_version(xtb: str) -> str:
    out = subprocess.run([xtb, "--version"], capture_output=True, text=True).stdout
    m = re.search(r"xtb version (\S+) \((\w+)\)", out)
    return f"{m.group(1)} ({m.group(2)[:7]})" if m else "?"


def run_xtb(nums, pos, charge, uhf, acc, extra=(), xtb="xtb"):
    with tempfile.TemporaryDirectory(dir=os.environ.get("AXTB_SCRATCH")) as d:
        d = Path(d)
        write_coord(d / "coord", nums, pos)
        cmd = [xtb, "coord", "--gfn", "2", "--grad", "--acc", str(acc), "--chrg", str(int(charge))]
        if uhf is not None:
            cmd += ["--uhf", str(uhf)]
        cmd += list(extra)
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        t = time.perf_counter()
        out = subprocess.run(cmd, cwd=d, capture_output=True, text=True, env=env)
        dt = time.perf_counter() - t
        if out.returncode != 0 or "normal termination" not in out.stderr + out.stdout:
            raise RuntimeError(out.stdout[-500:] + out.stderr[-500:])
        energy = float(re.findall(r"TOTAL ENERGY\s+(-?\d+\.\d+)", out.stdout)[-1])
        glines = (d / "gradient").read_text().replace("D", "E").splitlines()
        nat = len(nums)
        grad = np.array([[float(x) for x in l.split()] for l in glines[-1 - nat:-1]])
        charges = np.array([float(x) for x in (d / "charges").read_text().split()])
        it = re.findall(r"convergence criteria satisfied after\s+(\d+)\s+iterations", out.stdout)
        return {"energy": energy, "gradient": grad, "charges": charges, "seconds": dt, "stdout": out.stdout,
                "iterations": int(it[-1]) if it else None}


def run_axtb(nums, pos, charge, uhf, acc, solv=None, conventions="xtb"):
    req = {"numbers": nums.tolist(), "positions": pos.tolist(), "charge": charge, "uhf": uhf,
           "accuracy": acc, "gradient": True, "conventions": conventions}
    if solv:
        req[solv[0]] = solv[1]
    out = subprocess.run([str(AXTB), "--json"], input=json.dumps(req), capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError((out.stderr + out.stdout).strip())
    r = json.loads(out.stdout)
    r["gradient"] = np.array(r["gradient"])
    r["charges"] = np.array(r["charges"])
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", type=Path)
    ap.add_argument("--acc", type=float, default=1e-4)
    ap.add_argument("--alpb", help="solvent for ALPB")
    ap.add_argument("--gbsa", help="solvent for GBSA")
    ap.add_argument("--xtb", default=programs.get(programs.XTB680), help="the xtb binary (default: xtb 6.8.0, $AXTB_XTB680)")
    args = ap.parse_args()
    if not args.xtb:
        programs.skip("pass --xtb or " + programs.reason(programs.XTB680))
    ver = xtb_version(args.xtb)
    conv = "xtb-6.7.1" if ver.startswith("6.7.") else "xtb"
    print(f"xtb {ver} vs axtb conventions {conv!r}, accuracy {args.acc}")
    solv = ("alpb", args.alpb) if args.alpb else (("gbsa", args.gbsa) if args.gbsa else None)
    extra = (f"--{solv[0]}", solv[1]) if solv else ()
    files = args.files or sorted((ROOT / "tests/structures").glob("*.xyz"))
    print(f"{'structure':18s} {'nat':>4s} {'E(xtb)':>18s} {'dE':>10s} {'max|dG|':>10s} {'max|dq|':>10s}  iterations")
    rows, failures = [], []
    for f in files:
        nums, pos, charge = read_xyz(f)
        nel = int(nums.sum() - charge)
        uhf = nel % 2 if nel % 2 else None
        try:
            ref = run_xtb(nums, pos, charge, uhf, args.acc, extra, xtb=args.xtb)
            new = run_axtb(nums, pos, charge, uhf, args.acc, solv, conventions=conv)
        except Exception as e:  # noqa: BLE001
            print(f"{f.stem:18s} failed: {str(e)[:300]}")
            failures.append(f.stem)
            continue
        de = new["energy"] - ref["energy"]
        dg = np.abs(new["gradient"] - ref["gradient"]).max()
        dq = np.abs(new["charges"] - ref["charges"]).max()
        rows.append((f.stem, abs(de), dg, dq, new.get("iterations") != ref.get("iterations")))
        print(f"{f.stem:18s} {len(nums):4d} {ref['energy']:18.10f} {de:10.2e} {dg:10.2e} {dq:10.2e}  "
              f"{new.get('iterations')}/{ref.get('iterations')}")
    gated = [r for r in rows if r[0] not in KNOWN_DIFFERENCES]
    summary = {"max_abs_dE": max((r[1] for r in gated), default=None),
               "max_dG": max((float(r[2]) for r in gated), default=None),
               "max_dq": max((float(r[3]) for r in gated), default=None),
               "iteration_mismatches": [r[0] for r in gated if r[4]],
               "failures": [s for s in failures if s not in KNOWN_DIFFERENCES]}
    run = {r[0] for r in rows} | set(failures)
    return gate.check(summary, LIMITS[conv], {k: v for k, v in KNOWN_DIFFERENCES.items() if k in run})


if __name__ == "__main__":
    sys.exit(main())
