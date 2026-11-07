#!/usr/bin/env python3
"""g-xTB at finite electronic temperatures (Fermi smearing) in axtb vs the g-xTB programs.

For every row of crates/gxtb/tests/reference_v2/etemp/etemp.json (the 2.0.1 program, `xtb --gxtb
--etemp T --grad [--acc 0.0001]`: 57 reference molecules, Fe4 and stretched N2 at 300, 1000, 3000
and 5000 K) it runs the axtb CLI on the XYZ file

- with the program's conventions (`--conventions gxtb`) at the program's default and tight
  (`--acc 0.0001`) thresholds: total energy, SCF iterations and analytic gradient against the
  program's;
- with the default conventions (CODATA 2022 units and kB, integer electron count, exact Fermi
  level, smooth cutoffs): energy against the program, and the part of it that comes from kB
  (the default conventions at T kB_xtb / kB_CODATA, i.e. with the program's kB);

and for every row of crates/gxtb/tests/reference/etemp.json (the 1.1.0 program, `gxtb -tel T`)
the v1.1 program-mode energy and SCF iterations. Rounding-sensitive runs (an ulp of kT moves
our result as far as its difference to the program, spec §4.17.1) are flagged.

    python tests/gxtb_etemp_reproduction.py [--axtb target/release/axtb] [--out bench/results/gxtb_etemp.json] [--jobs 8]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF2 = ROOT / "crates/gxtb/tests/reference_v2"
REF1 = ROOT / "crates/gxtb/tests/reference"
KB_RATIO = 3.166808578545117e-06 / 3.1668115634564e-6  # program kB / CODATA 2022 kB
SENSITIVE = {(False, "Fe4"), (False, "TmH3", 3000.0), (False, "TmH3", 5000.0), (False, "ErF3", 5000.0),
             (True, "Fe4", 1000.0), (True, "CeCl3", 5000.0), (True, "ErF3", 3000.0), (True, "ErF3", 5000.0),
             (True, "N2_stretched", 5000.0), (True, "EuCl3", 5000.0)}


def sensitive(tight: bool, name: str, t: float) -> bool:
    return (tight, name) in SENSITIVE or (tight, name, t) in SENSITIVE


def run(axtb: str, xyz: Path, method: str, conv: str, q: int, u: int, t: float, *extra: str) -> dict:
    out = subprocess.run([axtb, str(xyz), "--method", method, "--conventions", conv, "--chrg", str(q), "--uhf", str(u),
                          "--etemp", repr(t), *extra], capture_output=True, text=True,
                         env=dict(os.environ, OMP_NUM_THREADS="1"))
    if out.returncode != 0:
        return {"error": (out.stderr + out.stdout).strip()[-200:]}
    return json.loads(out.stdout)


def maxdiff(a, b) -> float:
    return max(abs(x - y) for ra, rb in zip(a, b) for x, y in zip(ra, rb))


def v2_row(axtb: str, r: dict) -> dict:
    d = (REF2 / "etemp/extra" / r["name"]) if r["group"] == "extra" else (REF2 / r["group"] / r["name"])
    xyz, q, u, t = d / "mol.xyz", r["charge"], r["uhf"], r["etemp"]
    row = {"molecule": r["name"], "group": r["group"], "etemp": t, "E_program": r["total_energy"],
           "iter_program": r["scf_iterations"], "program_converged": r["converged"]}
    for tight in (False, True):
        ref = r["tight"] if tight else r
        key = "tight" if tight else "default"
        p = run(axtb, xyz, "gxtb-v2.0.1", "gxtb", q, u, t, "--grad", "--max-iter", "250", *(("--acc", "0.0001") if tight else ()))
        res = {"rounding_sensitive": sensitive(tight, r["name"], t)}
        if not ref["converged"]:
            res["program_fails"] = True
            res["axtb_fails"] = "error" in p
        elif "error" in p:
            res["error"] = p["error"]
        else:
            res.update(dE=p["energy"] - ref["total_energy"], iter_axtb=p["iterations"], iter_program=ref["scf_iterations"],
                       dgrad=maxdiff(p["gradient"], ref["gradient"]), entropy=p["terms"].get("entropy", 0.0))
        row[key] = res
    c = run(axtb, xyz, "gxtb-v2.0.1", "axtb", q, u, t)
    ck = run(axtb, xyz, "gxtb-v2.0.1", "axtb", q, u, t * KB_RATIO)
    if "error" not in c and "error" not in ck and r["converged"]:
        row["default_conventions"] = {"dE": c["energy"] - r["total_energy"], "kB_part": c["energy"] - ck["energy"],
                                      "iterations": c["iterations"], "entropy": c["terms"].get("entropy", 0.0)}
    return row


def v1_row(axtb: str, r: dict) -> dict:
    d = (REF2 / "etemp/extra" / r["name"]) if r["group"] == "extra" else (REF1 / r["name"])
    row = {"molecule": r["name"], "etemp": r["etemp"], "E_program": r["total_energy"], "iter_program": r["scf_iterations"]}
    p = run(axtb, d / "mol.xyz", "gxtb-v1.1", "gxtb", r["charge"], r["uhf"], r["etemp"], "--max-iter", "1000")
    if not r["converged"]:
        row.update(program_fails=True, axtb_fails="error" in p)
    elif "error" in p:
        row["error"] = p["error"]
    else:
        row.update(dE=p["energy"] - r["total_energy"], iter_axtb=p["iterations"])
    row["rounding_sensitive"] = r["name"] == "Fe4"
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axtb", default=os.environ.get("AXTB_BIN", str(ROOT / "target/release/axtb")))
    ap.add_argument("--out", type=Path, default=ROOT / "bench/results/gxtb_etemp.json")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    rows2 = json.loads((REF2 / "etemp/etemp.json").read_text())["rows"]
    rows1 = json.loads((REF1 / "etemp.json").read_text())["rows"]
    with ThreadPoolExecutor(a.jobs) as ex:
        v2 = list(ex.map(lambda r: v2_row(a.axtb, r), rows2))
        v1 = list(ex.map(lambda r: v1_row(a.axtb, r), rows1))

    def ok(res):
        return "dE" in res and not res["rounding_sensitive"]

    summary = {}
    for key in ("default", "tight"):
        good = [r[key] for r in v2 if ok(r[key])]
        summary[f"v2_{key}"] = {
            "runs": len(v2), "compared": len(good),
            "max_abs_dE": max(abs(x["dE"]) for x in good),
            "max_abs_dE_without_water100": max(abs(x["dE"]) for r in v2 if ok(r[key]) and r["molecule"] != "water100" for x in [r[key]]),
            "max_dgrad": max(x["dgrad"] for x in good),
            "iteration_mismatches": [f'{r["molecule"]} {r["etemp"]:g} K' for r in v2 if ok(r[key]) and r[key]["iter_axtb"] != r[key]["iter_program"]],
            "failures_as_program": [f'{r["molecule"]} {r["etemp"]:g} K' for r in v2 if r[key].get("program_fails") and r[key].get("axtb_fails")],
            "rounding_sensitive": {f'{r["molecule"]} {r["etemp"]:g} K': {k: r[key].get(k) for k in ("dE", "iter_axtb", "iter_program", "dgrad")}
                                   for r in v2 if r[key]["rounding_sensitive"]},
            "errors": [f'{r["molecule"]} {r["etemp"]:g} K: {r[key]["error"]}' for r in v2 if "error" in r[key]],
        }
    dc = [r["default_conventions"] for r in v2 if "default_conventions" in r and not r["default"]["rounding_sensitive"]]
    summary["v2_default_conventions"] = {"max_abs_dE_vs_program": max(abs(x["dE"]) for x in dc),
                                         "max_abs_kB_part": max(abs(x["kB_part"]) for x in dc),
                                         "kB_part_largest": sorted(((abs(r["default_conventions"]["kB_part"]), r["molecule"], r["etemp"])
                                                                    for r in v2 if "default_conventions" in r
                                                                    and not r["default"]["rounding_sensitive"]), reverse=True)[:5],
                                         "dE_largest": sorted(((abs(r["default_conventions"]["dE"]), r["molecule"], r["etemp"])
                                                               for r in v2 if "default_conventions" in r
                                                               and not r["default"]["rounding_sensitive"]), reverse=True)[:5]}
    g1 = [r for r in v1 if "dE" in r and not r["rounding_sensitive"]]
    summary["v1.1"] = {"runs": len(v1), "compared": len(g1), "max_abs_dE": max(abs(r["dE"]) for r in g1),
                       "iteration_mismatches": [f'{r["molecule"]} {r["etemp"]:g} K' for r in g1 if r["iter_axtb"] != r["iter_program"]]}
    a.out.write_text(json.dumps({"programs": {"2.0.1": "g-xTB 2.0.1 (grimme-lab/g-xtb 2.0.1: xtb 6.7.1 --gxtb --etemp T)",
                                              "1.1": "g-xTB 1.1.0 (grimme-lab/g-xtb v1.1.0: gxtb -tel T)"},
                                 "summary": summary, "v2": v2, "v1.1": v1}, indent=1))
    print(json.dumps(summary, indent=1))
    print("wrote", a.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
