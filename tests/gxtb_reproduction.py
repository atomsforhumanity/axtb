#!/usr/bin/env python3
"""g-xTB v1.1 in axtb vs the g-xtb 1.1.0 program: total energies and SCF iteration counts for
the reference molecules stored in crates/gxtb/tests (program outputs recorded there), with
the program's Fock matrix (conventions "gxtb": E_axtb, dE, iter_axtb) and with the exact one
(the default conventions "axtb": E_corrected, dE_corrected, iter_corrected).

    python tests/gxtb_reproduction.py [--axtb target/release/axtb] [--out bench/results/gxtb_reproduction.json]

The script fails (exit status 1) when a quantity exceeds its limit in `LIMITS` (tests/gate.py)
or a molecule fails.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import gate

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "crates/gxtb/tests"
# program conventions: total energies of the 41 molecules <= 1e-11 Eh, the same SCF iterations
# (docs/validation.md, Summary); the default conventions differ by up to ~4e-5 Eh, 2.3e-4 for
# ferrocene (README, g-xTB)
LIMITS = {"max_abs_dE": 1e-11, "iteration_mismatches": 0, "failures": 0, "max_abs_dE_default": 2.3e-4}


def cases():
    for d in sorted(p for p in (REF / "reference").iterdir() if p.is_dir()):
        s = json.loads((d / "summary.json").read_text())
        i = json.loads((d / "input.json").read_text())
        yield {"name": d.name, "xyz": (d / "mol.xyz").read_text(), "charge": i["charge"], "uhf": i["uhf"],
               "E_program": s["total_energy"], "iter_program": s["scf_iterations"], "set": "reference"}
    for m in json.loads((REF / "reference_extra.json").read_text())["molecules"]:
        yield {"name": m["name"], "xyz": m["xyz"], "charge": m["charge"], "uhf": m["uhf"],
               "E_program": m["total_energy"], "iter_program": m["scf_iterations"], "set": "extra"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axtb", default=os.environ.get("AXTB_BIN", str(ROOT / "target/release/axtb")))
    ap.add_argument("--out", type=Path, default=ROOT / "bench/results/gxtb_reproduction.json")
    a = ap.parse_args()
    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for c in cases():
            f = Path(tmp) / f"{c['name']}.xyz"
            f.write_text(c["xyz"])
            res = {}
            for conv in ("gxtb", "axtb"):
                out = subprocess.run([a.axtb, str(f), "--method", "gxtb-v1.1", "--conventions", conv,
                                      "--chrg", str(c["charge"]), "--uhf", str(c["uhf"])],
                                     capture_output=True, text=True)
                res[conv] = (json.loads(out.stdout) if out.returncode == 0 else {},
                             None if out.returncode == 0 else out.stderr.strip()[-200:])
            (r, err), (rc, errc) = res["gxtb"], res["axtb"]
            e, ec = r.get("energy"), rc.get("energy")
            rows.append({"molecule": c["name"].replace("_", " "), "set": c["set"], "natoms": int(c["xyz"].split()[0]),
                         "charge": c["charge"], "uhf": c["uhf"], "E_program": c["E_program"], "E_axtb": e,
                         "dE": None if e is None else e - c["E_program"],
                         "iter_program": c["iter_program"], "iter_axtb": r.get("iterations"),
                         "E_corrected": ec, "dE_corrected": None if ec is None else ec - c["E_program"],
                         "iter_corrected": rc.get("iterations"), "error": err or errc})
            print(rows[-1]["molecule"], rows[-1]["dE"], rows[-1]["iter_program"], rows[-1]["iter_axtb"],
                  rows[-1]["dE_corrected"], rows[-1]["iter_corrected"], flush=True)
    a.out.write_text(json.dumps({"program": "g-xTB 1.1.0 (grimme-lab/g-xtb v1.1.0)", "rows": rows}, indent=1))
    print("wrote", a.out)
    ok = [r for r in rows if not r["error"]]
    return gate.check({"max_abs_dE": max((abs(r["dE"]) for r in ok), default=None),
                       "iteration_mismatches": [r["molecule"] for r in ok if r["iter_axtb"] != r["iter_program"]],
                       "failures": [r["molecule"] for r in rows if r["error"]],
                       "max_abs_dE_default": max((abs(r["dE_corrected"]) for r in ok), default=None)}, LIMITS)


if __name__ == "__main__":
    raise SystemExit(main())
