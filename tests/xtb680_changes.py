#!/usr/bin/env python3
"""What xtb 6.8.0 changed for GFN2-xTB, quirk by quirk: axtb's xtb (6.8.0) conventions with one
setting at a time put back to its xtb 6.7.1 value (`examples/conventions_ablation`), each
against the xtb 6.8.0 program, at tight accuracy (`--acc 1e-4`: the static quirks) and at xtb's
default accuracy 1 (the SCC loop's quirks, which only matter while the SCF is unconverged).
Also the 6.7.1 program (the g-xTB 2.0.1 build) and axtb's xtb-6.7.1 conventions on the same
molecules.

    cargo build --release -p axtb --example conventions_ablation
    python tests/xtb680_changes.py [--xtb PATH] [--xtb671 PATH] [--out bench/results/xtb680_changes.json]

`--xtb` defaults to xtb 6.8.0 (`$AXTB_XTB680`) and `--xtb671` to the g-xTB 2.0.1 build of xtb 6.7.1
(`$AXTB_GXTB2`; optional), see tests/programs.py.

Each xtb run: a Turbomole `coord` file in bohr, `--gfn 2 --grad --acc A`, OMP_NUM_THREADS=1,
energy from `TOTAL ENERGY`, gradient from the `gradient` file.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import programs  # noqa: E402

ABLATION = ROOT / "target/release/examples/conventions_ablation"
SYMBOLS = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
           "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
           "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn").split()
AATOAU = 1.0 / 0.52917721090380  # any value: every program gets the same bohr positions


def read_xyz(path: Path):
    lines = path.read_text().splitlines()
    nat = int(lines[0].split()[0])
    m = re.search(r"charge\s*=\s*(-?\d+)", lines[1])
    nums, pos = [], []
    for line in lines[2:2 + nat]:
        f = line.split()
        nums.append(SYMBOLS.index(f[0].capitalize()) + 1)
        pos.append([float(x) * AATOAU for x in f[1:4]])
    return nums, np.array(pos), int(m.group(1)) if m else 0


def csoh(order):
    """Compressed CsOH (hits xtb 6.7.1's stale Born-radius variable, grimme-lab/xtb#1504)."""
    atoms = [(1, [0.0, 0.0, -0.96]), (8, [0.0, 0.0, 0.0]), (55, [0.0, 0.0, 2.08])]
    return [atoms[i][0] for i in order], np.array([[x * AATOAU for x in atoms[i][1]] for i in order]), 0


S = ROOT / "tests/structures"
CASES = [
    ("water", read_xyz(S / "water.xyz"), None),
    ("caffeine", read_xyz(S / "caffeine.xyz"), None),
    ("ferrocene", read_xyz(S / "ferrocene.xyz"), None),
    ("[Cu(NH3)4]2+ (doublet)", read_xyz(S / "cu-nh3-4.xyz"), None),
    ("taxol (113 atoms, 30 bohr)", read_xyz(S / "taxol.xyz"), None),
    ("caffeine, ALPB water", read_xyz(S / "caffeine.xyz"), "alpb:water:gsolv"),
    ("caffeine, ALPB water, reference state", read_xyz(S / "caffeine.xyz"), "alpb:water:reference"),
    ("caffeine, GBSA water, reference state", read_xyz(S / "caffeine.xyz"), "gbsa:water:reference"),
    ("compressed CsOH (H, O, Cs), ALPB water", csoh([0, 1, 2]), "alpb:water:gsolv"),
    ("compressed CsOH (Cs, O, H), ALPB water", csoh([2, 1, 0]), "alpb:water:gsolv"),
]


def run_xtb(xtb, nums, pos, charge, uhf, acc, solv):
    with tempfile.TemporaryDirectory(dir=os.environ.get("AXTB_SCRATCH")) as d:
        d = Path(d)
        lines = ["$coord"] + [f"{x:24.16f} {y:24.16f} {z:24.16f} {SYMBOLS[n - 1].lower()}" for n, (x, y, z) in zip(nums, pos)]
        (d / "coord").write_text("\n".join(lines + ["$end", ""]))
        cmd = [xtb, "coord", "--gfn", "2", "--grad", "--acc", repr(acc), "--chrg", str(charge)]
        if uhf is not None:
            cmd += ["--uhf", str(uhf)]
        if solv:
            m, solvent, state = solv.split(":")
            cmd += [f"--{m}", solvent, state]
        out = subprocess.run(cmd, cwd=d, capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="1"))
        e = re.findall(r"TOTAL ENERGY\s+(-?\d+\.\d+)", out.stdout)
        it = re.findall(r"convergence criteria satisfied after\s+(\d+)\s+iterations", out.stdout)
        if not e or "normal termination" not in out.stdout + out.stderr:
            return None
        gl = (d / "gradient").read_text().replace("D", "E").splitlines()
        g = np.array([[float(x) for x in l.split()] for l in gl[-1 - len(nums):-1]])
        return {"energy": float(e[-1]), "gradient": g, "iterations": int(it[-1]) if it else None}


def ablation(nums, pos, charge, uhf, acc, solv):
    text = f"{charge} {'-' if uhf is None else uhf} {acc!r} 300.0 {solv or '-'}\n{len(nums)}\n"
    text += "".join(f"{z} {float(x)!r} {float(y)!r} {float(w)!r}\n" for z, (x, y, w) in zip(nums, pos))
    out = subprocess.run([str(ABLATION)], input=text, capture_output=True, text=True, check=True).stdout
    res = {}
    for line in out.splitlines():
        name, e, rest = line.split("\t", 2)
        if e == "error":
            res[name] = {"error": rest}
            continue
        it, g = rest.split("\t", 1) if "\t" in rest else (rest, "")
        res[name] = {"energy": float(e), "iterations": int(it),
                     "gradient": np.array([float(x) for x in g.split()]).reshape(-1, 3)}
    return res


def cli(nums, pos, charge, uhf, acc, solv, conventions):
    req = {"numbers": list(map(int, nums)), "positions": pos.tolist(), "charge": charge, "uhf": uhf, "accuracy": acc,
           "gradient": True, "conventions": conventions}
    if solv:
        m, solvent, state = solv.split(":")
        req[m] = solvent
        req["solvation_state"] = state
    out = subprocess.run([str(ROOT / "target/release/axtb"), "--json"], input=json.dumps(req), capture_output=True,
                         text=True, check=True)
    r = json.loads(out.stdout)
    return {"energy": r["energy"], "gradient": np.array(r["gradient"]), "iterations": r["iterations"]}


def diff(a, b):
    if a is None or b is None or "error" in a:
        return None
    return {"dE": a["energy"] - b["energy"], "dG": float(np.abs(a["gradient"] - b["gradient"]).max()),
            "iterations": [a["iterations"], b["iterations"]]}


def version(xtb):
    v = subprocess.run([xtb, "--version"], capture_output=True, text=True).stdout
    m = re.search(r"xtb version (\S+) \((\w+)\)", v)
    return f"{m.group(1)} ({m.group(2)[:7]})" if m else "?"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xtb", default=programs.get(programs.XTB680), help="the xtb 6.8.0 binary (default: $AXTB_XTB680)")
    ap.add_argument("--xtb671", default=programs.get(programs.GXTB2),
                    help="an xtb 6.7.1 binary (optional; default: the g-xTB 2.0.1 build, $AXTB_GXTB2)")
    ap.add_argument("--out", type=Path, default=ROOT / "bench/results/xtb680_changes.json")
    a = ap.parse_args()
    if not a.xtb:
        programs.skip("pass --xtb or " + programs.reason(programs.XTB680))
    if not a.xtb671:
        programs.note_skip("the xtb 6.7.1 rows", programs.GXTB2)
    rows = []
    for acc in (1e-4, 1.0):
        for name, (nums, pos, charge), solv in CASES:
            nel = sum(nums) - charge
            uhf = 1 if nel % 2 else None
            ref = run_xtb(a.xtb, nums, pos, charge, uhf, acc, solv)
            if ref is None:
                print(f"{name} acc {acc}: xtb 6.8.0 failed")
                continue
            ab = ablation(nums, pos, charge, uhf, acc, solv)
            row = {"case": name, "accuracy": acc, "solvation": solv, "natoms": len(nums), "E_xtb680": ref["energy"],
                   "iterations_xtb680": ref["iterations"],
                   "variants": {k: diff(v, ref) for k, v in ab.items()}}
            if a.xtb671:
                old = run_xtb(a.xtb671, nums, pos, charge, uhf, acc, solv)
                row["xtb671_minus_xtb680"] = diff(old, ref)
                row["axtb_xtb671_vs_xtb671"] = diff(cli(nums, pos, charge, uhf, acc, solv, "xtb-6.7.1"), old)
            rows.append(row)
            base = row["variants"]["xtb-6.8.0"]
            print(f"{name:42s} acc {acc:g}: xtb-6.8.0 dE {base['dE']:.1e} dG {base['dG']:.1e} it {base['iterations']}",
                  flush=True)
    # largest effect of each variant over the cases
    summary = {}
    for acc in (1e-4, 1.0):
        sub = [r for r in rows if r["accuracy"] == acc]
        names = list(sub[0]["variants"]) if sub else []
        summary[f"acc {acc:g}"] = {
            n: {"max_abs_dE": max((abs(r["variants"][n]["dE"]) for r in sub if r["variants"][n]), default=None),
                "max_dG": max((r["variants"][n]["dG"] for r in sub if r["variants"][n]), default=None),
                "iterations_differ": sum(1 for r in sub if r["variants"][n] and
                                         r["variants"][n]["iterations"][0] != r["variants"][n]["iterations"][1])}
            for n in names}
    load = [round(x, 2) for x in os.getloadavg()]
    out = {"xtb": version(a.xtb), "xtb671": version(a.xtb671) if a.xtb671 else None, "method": "GFN2-xTB",
           "note": ("axtb's xtb conventions (xtb 6.8.0) with one setting at a time put back to xtb 6.7.1's; "
                    "each row: variant minus the xtb 6.8.0 program (dE in Eh, max |dG| in Eh/bohr, iterations "
                    "axtb/xtb); OMP_NUM_THREADS=1"),
           "load_average": load, "summary": summary, "cases": rows}
    a.out.write_text(json.dumps(out, indent=1, default=lambda x: None if isinstance(x, float) and math.isnan(x) else x) + "\n")
    print(json.dumps(summary, indent=1))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
