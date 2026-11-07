#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Compare the ALPB/GBSA solvation free energy of xtb 6.7.1 and tblite.

Neither program exposes its solvation term in isolation at fixed charges, so
the comparison uses dG = E(solvated SCF) - E(gas-phase SCF) of each program on
identical bohr geometries (taken from tests/data/*.ref), with tight SCF
convergence.  The gas-phase total energies are compared as well, to separate
GFN2 implementation differences from solvation-model differences.

Run with the workspace venv (needs the tblite python package)::

    .venv/bin/python crates/solvation/scripts/compare_tblite.py --scratch DIR [--xtb XTB]

The program is ``--xtb``, or else ``$AXTB_GXTB2`` (tests/programs.py: the g-xTB 2.0.1 build of
xtb 6.7.1). ``xtb`` is never taken from PATH.
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
from tblite.interface import Calculator

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "tests" / "data"
sys.path.insert(0, str(HERE.parents[2] / "tests"))
import programs  # noqa: E402

XTB = "xtb"  # set in main() from --xtb or $AXTB_GXTB2

SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
    "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn"
).split()
UHF = {"cu-nh3-4": 1}


def read_ref(name: str):
    nums, pos, strs, charge = [], [], [], 0
    for line in (DATA / f"{name}.ref").read_text().splitlines():
        t = line.split()
        if t and t[0] == "atom":
            nums.append(int(t[1]))
            strs.append(t[2:5])
            pos.append([float(x) for x in t[2:5]])
        elif t and t[0] == "charge":
            charge = int(t[1])
    return nums, np.array(pos), strs, charge


def xtb_energy(wd: Path, nums, strs, charge, uhf, solv: list[str]) -> float:
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "coord").write_text(
        "$coord\n" + "".join(f"{x} {y} {z} {SYMBOLS[n - 1].lower()}\n" for n, (x, y, z) in zip(nums, strs)) + "$end\n"
    )
    cmd = [XTB, "coord", "--acc", "0.001", "--chrg", str(charge), "--uhf", str(uhf)] + solv
    env = dict(os.environ, OMP_NUM_THREADS="1")
    out = subprocess.run(cmd, cwd=wd, env=env, capture_output=True, text=True).stdout
    (wd / "xtb.out").write_text(out)
    m = re.search(r":: total energy\s+(-?\d+\.\d+) Eh", out)
    if not m:
        raise RuntimeError(f"xtb failed in {wd}")
    return float(m.group(1))


def tblite_energy(nums, pos, charge, uhf, solv: tuple | None) -> float:
    calc = Calculator("GFN2-xTB", np.array(nums), pos, charge=float(charge), uhf=uhf)
    calc.set("verbosity", 0)
    calc.set("accuracy", 0.001)
    if solv:
        calc.add(*solv)
    return float(calc.singlepoint().get("energy"))


CASES = [
    # (reference file for the geometry, model, solvent, state)
    ("water_alpb_water", "alpb", "water", "gsolv"),
    ("caffeine_alpb_water", "alpb", "water", "gsolv"),
    ("caffeine_alpb_water", "alpb", "water", "bar1M"),
    ("caffeine_alpb_water", "alpb", "water", "reference"),
    ("caffeine_alpb_water", "gbsa", "water", "gsolv"),
    ("caffeine_alpb_water", "alpb", "toluene", "gsolv"),
    ("caffeine_alpb_water", "alpb", "dmso", "gsolv"),
    ("zncl4_alpb_water", "alpb", "water", "gsolv"),
    ("cf3br-benzene_alpb_water", "alpb", "water", "gsolv"),
    ("taxol_alpb_water", "alpb", "water", "gsolv"),
    # geometries hitting the stale-r1 branch of the Born integral (both codes)
    ("hgh2-compressed_alpb_water", "alpb", "water", "gsolv"),
    ("csoh-compressed_alpb_water", "alpb", "water", "gsolv"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scratch", type=Path, required=True)
    ap.add_argument("--xtb", default=None, help="the xtb program (default: $AXTB_GXTB2, see tests/programs.py)")
    args = ap.parse_args()
    global XTB
    XTB = args.xtb or programs.require(programs.GXTB2, what="the xtb side of the comparison")[0]
    gas: dict[str, tuple[float, float]] = {}
    print(f"{'geometry':<26} {'model':<5} {'solvent':<8} {'state':<9} {'dG xtb':>14} {'dG tblite':>14} "
          f"{'tblite-xtb':>11} {'Egas t-x':>10}")
    for ref, model, solvent, state in CASES:
        nums, pos, strs, charge = read_ref(ref)
        mol = ref.split("_")[0]
        uhf = UHF.get(mol, 0)
        try:
            if mol not in gas:
                ex = xtb_energy(args.scratch / f"tbl_{mol}_gas", nums, strs, charge, uhf, [])
                et = tblite_energy(nums, pos, charge, uhf, None)
                gas[mol] = (ex, et)
            xflags = [f"--{model}", solvent] + ([state] if state != "gsolv" else [])
            ex = xtb_energy(args.scratch / f"tbl_{mol}_{model}_{solvent}_{state}", nums, strs, charge, uhf, xflags)
            tstate = {"gsolv": "gsolv", "bar1M": "bar1mol", "reference": "reference"}[state]
            et = tblite_energy(nums, pos, charge, uhf, (f"{model}-solvation", solvent, tstate))
        except Exception as exc:  # SCF failure in one of the programs
            print(f"{mol:<26} {model:<5} {solvent:<8} {state:<9} skipped: {exc}")
            continue
        dx, dt = ex - gas[mol][0], et - gas[mol][1]
        print(f"{mol:<26} {model:<5} {solvent:<8} {state:<9} {dx:>14.9f} {dt:>14.9f} {dt - dx:>11.2e} "
              f"{gas[mol][1] - gas[mol][0]:>10.1e}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
