#!/usr/bin/env python3
"""Measure where the xtb program and tblite 0.8.0 differ from each other for GFN2-xTB, and the
bugs found in them, with axtb's default conventions alongside.

    python tests/program_differences.py [--xtb PATH] [--out FILE]

Each row is one measurement: a case, the quantity, and its value from each program. The xtb
binary is `--xtb`, default xtb 6.8.0 (`$AXTB_XTB680`, see tests/programs.py); the output records its version and goes to
`bench/results/program_diffs.json` for xtb 6.7.1 (the record of what 6.7.1 did) and to
`bench/results/program_diffs_<version>.json` otherwise. Needs the tblite Python package; uses
the runners of the reference matrix.

The differences between the programs are measurements; the script fails (exit status 1) when
axtb's quantities exceed their limits in `LIMITS` (tests/gate.py) or a program fails.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "reference_matrix"))
if __name__ == "__main__" and "--xtb" in sys.argv:
    os.environ["AXTB_MATRIX_XTB"] = sys.argv[sys.argv.index("--xtb") + 1]
import runners as P  # noqa: E402
import structures as S  # noqa: E402

sys.path.insert(0, str(HERE))
import gate  # noqa: E402

AATOAU = 1.0 / 0.529177210544  # CODATA 2022 bohr in Å (qcref)
TIGHT = 1e-6
# axtb's default conventions against tblite (solvation states, O2 at 0.1 and 1 K): chosen, the
# measured 3.9e-11 Eh about 10x; axtb's atom-order dependence (compressed CsOH): round-off
# (measured 2.7e-15 Eh; docs/validation.md: 2.9e-3 Eh in xtb 6.7.1 and tblite -> 0), chosen 1e-12
LIMITS = {"axtb_minus_tblite_max_abs": 4e-10, "axtb_atom_order_abs": 1e-12, "failures": 0}


def spec(mol, charge=0, uhf=None, **opts):
    nums, pos = mol
    s = {"numbers": [int(z) for z in nums], "positions": np.asarray(pos, dtype=float).tolist(),
         "charge": charge, "uhf": uhf, "accuracy": TIGHT}
    s.update(opts)
    return s


def energies(s, programs=("xtb", "tblite", "axtb")):
    out = {}
    for p in programs:
        r = {"xtb": lambda: P.run_xtb(s), "tblite": lambda: P.run_tblite(s),
             "axtb": lambda: P.run_axtb(s, "axtb", gradient=False)}[p]()
        out[p] = r["energy"] if r.get("ok") else None
    return out


def diff(a, b):
    return None if a is None or b is None else a - b


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--xtb", help="the xtb binary (default: xtb 6.8.0, $AXTB_XTB680)")
    args = ap.parse_args()
    if P.XTB is None:
        P.programs.skip("pass --xtb or " + P.programs.reason(P.programs.XTB680))
    version = P.xtb_version()
    if args.out is None:
        name = "program_diffs.json" if version.startswith("6.7.") else f"program_diffs_{version.split()[0]}.json"
        args.out = HERE.parent / "bench/results" / name
    print(f"xtb {version}", flush=True)
    rows = []

    # 1. implicit solvation: the three reference states, xtb vs tblite (and axtb default vs tblite)
    species = [("water", 0, None), ("acetic_acid", 0, None), ("acetate", -1, None), ("methylammonium", 1, None)]
    for model in ("alpb", "gbsa"):
        for name, q, uhf in species:
            for state in ("gsolv", "bar1M", "reference"):
                e = energies(spec(S.molecule(name), q, uhf, **{model: "water", "solvation_state": state}))
                rows.append({"topic": "solvation reference state", "case": f"{name} in water, {model.upper()}, {state}",
                             "state": state, "model": model.upper(), "molecule": name,
                             "xtb_minus_tblite": diff(e["xtb"], e["tblite"]), "axtb_minus_tblite": diff(e["axtb"], e["tblite"]),
                             "E_xtb": e["xtb"], "E_tblite": e["tblite"], "E_axtb": e["axtb"]})
                print(rows[-1]["case"], rows[-1]["xtb_minus_tblite"], flush=True)

    # 2. atom-order dependence of ALPB (stale Born-radius variable, xtb#1504 / tblite#430)
    atoms = [(1, [0.0, 0.0, -0.96]), (8, [0.0, 0.0, 0.0]), (55, [0.0, 0.0, 2.08])]

    def csoh(order):
        return ([atoms[i][0] for i in order], [[x * AATOAU for x in atoms[i][1]] for i in order])
    for state in ("gsolv",):
        a = energies(spec(csoh([0, 1, 2]), alpb="water", solvation_state=state))
        b = energies(spec(csoh([2, 1, 0]), alpb="water", solvation_state=state))
        rows.append({"topic": "atom-order dependence", "case": "compressed CsOH in water (ALPB): E(H,O,Cs) − E(Cs,O,H)",
                     "xtb": diff(a["xtb"], b["xtb"]), "tblite": diff(a["tblite"], b["tblite"]), "axtb": diff(a["axtb"], b["axtb"])})
        print(rows[-1], flush=True)

    # 3. electronic temperature at or below 0.1 K: xtb switches to integer occupations
    for t in (0.1, 1.0):
        e = energies(spec(S.molecule("O2"), 0, 0, etemp=t))
        rows.append({"topic": "occupations at low electronic temperature", "case": f"O2 singlet (degenerate π*), {t} K",
                     "xtb_minus_tblite": diff(e["xtb"], e["tblite"]), "axtb_minus_tblite": diff(e["axtb"], e["tblite"]),
                     "E_xtb": e["xtb"], "E_tblite": e["tblite"], "E_axtb": e["axtb"]})
        print(rows[-1], flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    import importlib.metadata as md
    out = {"xtb": version, "tblite": md.version("tblite"), "accuracy": TIGHT,
           "note": "E in Eh; xtb, tblite and axtb (default conventions) on the same bohr geometries", "rows": rows}
    args.out.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", args.out)
    axtb = [abs(r["axtb_minus_tblite"]) for r in rows if r.get("axtb_minus_tblite") is not None]
    order = [abs(r["axtb"]) for r in rows if r["topic"] == "atom-order dependence" and r["axtb"] is not None]
    # a program that fails leaves its energy (and every difference with it) None
    failures = [f"{r['case']}: {k}" for r in rows for k in ("E_xtb", "E_tblite", "E_axtb", "xtb", "tblite", "axtb")
                if k in r and r[k] is None]
    return gate.check({"axtb_minus_tblite_max_abs": max(axtb, default=None),
                       "axtb_atom_order_abs": max(order, default=None), "failures": failures}, LIMITS)


if __name__ == "__main__":
    sys.exit(main())
