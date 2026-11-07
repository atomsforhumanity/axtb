#!/usr/bin/env python3
"""Compare axtb GFN2-xTB with the xtb program (`--xtb`, default xtb 6.8.0 from `$AXTB_XTB680`, see
tests/programs.py): xtb 6.8.0 with axtb's `xtb` conventions, xtb 6.7.1 (also the g-xTB 2.0.1
build, `--xtb "$AXTB_GXTB2"`) with `xtb-6.7.1`.

Usage: python tests/validate_xtb.py [--acc 1e-4] [--xtb PATH] [--alpb water] [structure.xyz ...]

xtb gets the geometry as a Turbomole `coord` file in bohr (so its Å→bohr constant does
not matter), runs `--gfn 2 --grad` in its own scratch directory with OMP_NUM_THREADS=1,
and its energy and gradient are read from the `energy`/`gradient` files.
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
import programs  # noqa: E402


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
    for f in files:
        nums, pos, charge = read_xyz(f)
        nel = int(nums.sum() - charge)
        uhf = nel % 2 if nel % 2 else None
        try:
            ref = run_xtb(nums, pos, charge, uhf, args.acc, extra, xtb=args.xtb)
            new = run_axtb(nums, pos, charge, uhf, args.acc, solv, conventions=conv)
        except Exception as e:  # noqa: BLE001
            print(f"{f.stem:18s} failed: {str(e)[:300]}")
            continue
        de = new["energy"] - ref["energy"]
        dg = np.abs(new["gradient"] - ref["gradient"]).max()
        dq = np.abs(new["charges"] - ref["charges"]).max()
        print(f"{f.stem:18s} {len(nums):4d} {ref['energy']:18.10f} {de:10.2e} {dg:10.2e} {dq:10.2e}  "
              f"{new.get('iterations')}/{ref.get('iterations')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
