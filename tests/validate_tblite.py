#!/usr/bin/env python3
"""Compare axtb GFN2-xTB energies, gradients and charges with tblite.

Usage: python tests/validate_tblite.py [--nodisp] [--acc 1e-4] [structure.xyz ...]

tblite (PyPI wheel) is only a reference here; axtb never links or calls it. Both programs
receive the same bohr coordinates. With --nodisp, tblite's GFN2 parameter set is exported
with the dispersion section removed, which isolates everything except D4.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AXTB = ROOT / "target/release/axtb"
AATOAU = 1.0 / 0.529177210903  # any value works: both codes get the same bohr input

SYMBOLS = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
           "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
           "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn").split()


def read_xyz(path: Path):
    lines = path.read_text().splitlines()
    nat = int(lines[0].split()[0])
    m = re.search(r"charge\s*=\s*(-?\d+)", lines[1])
    charge = int(m.group(1)) if m else 0
    nums, pos = [], []
    for line in lines[2:2 + nat]:
        f = line.split()
        nums.append(SYMBOLS.index(f[0].capitalize()) + 1)
        pos.append([float(x) * AATOAU for x in f[1:4]])
    return np.array(nums), np.array(pos), charge


def gfn2_param(nodisp: bool) -> dict:
    from tblite import library as L
    p = L.new_param()
    L.export_gfn2_param(p)
    t = L.new_table()
    L.dump_param(p, t)
    d = L.table_to_dict(t)
    if nodisp:
        d.pop("dispersion")
    return d


def run_tblite(param, nums, pos, charge, acc):
    from tblite.interface import Calculator
    calc = Calculator(param, nums, pos, charge=float(charge))
    calc.set("accuracy", acc)
    calc.set("verbosity", 0)
    res = calc.singlepoint()
    return {"energy": res.get("energy"), "gradient": res.get("gradient"), "charges": res.get("charges")}


def run_axtb(nums, pos, charge, acc, nodisp):
    req = {"numbers": nums.tolist(), "positions": pos.tolist(), "charge": charge, "accuracy": acc,
           "gradient": True, "dispersion": not nodisp, "conventions": "tblite"}
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
    ap.add_argument("--nodisp", action="store_true")
    ap.add_argument("--acc", type=float, default=1e-4)
    args = ap.parse_args()
    files = args.files or sorted((ROOT / "tests/structures").glob("*.xyz"))
    param = gfn2_param(args.nodisp)
    worst = 0.0
    print(f"{'structure':18s} {'nat':>4s} {'E(tblite)':>18s} {'dE':>10s} {'max|dG|':>10s} {'max|dq|':>10s} {'it':>4s} {'t/s':>7s}")
    for f in files:
        nums, pos, charge = read_xyz(f)
        try:
            ref = run_tblite(param, nums, pos, charge, args.acc)
        except Exception as e:  # noqa: BLE001
            print(f"{f.stem:18s} tblite failed: {e}")
            continue
        try:
            new = run_axtb(nums, pos, charge, args.acc, args.nodisp)
        except Exception as e:  # noqa: BLE001
            print(f"{f.stem:18s} axtb failed: {e}")
            worst = float("inf")
            continue
        de = new["energy"] - ref["energy"]
        dg = np.abs(new["gradient"] - ref["gradient"]).max()
        dq = np.abs(new["charges"] - ref["charges"]).max()
        worst = max(worst, abs(de))
        print(f"{f.stem:18s} {len(nums):4d} {ref['energy']:18.10f} {de:10.2e} {dg:10.2e} {dq:10.2e} "
              f"{new['iterations']:4d} {new['seconds']:7.3f}")
    print(f"worst |dE| = {worst:.2e} Eh")
    return 0 if worst < 1e-7 else 1


if __name__ == "__main__":
    sys.exit(main())
