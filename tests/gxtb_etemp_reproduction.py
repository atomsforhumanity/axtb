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

The script fails (exit status 1) when a quantity exceeds its limit in `LIMITS` (tests/gate.py)
or a run fails where the program converges, except the rounding-sensitive runs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gate

ROOT = Path(__file__).resolve().parents[1]
REF2 = ROOT / "crates/gxtb/tests/reference_v2"
REF1 = ROOT / "crates/gxtb/tests/reference"
KB_RATIO = 3.166808578545117e-06 / 3.1668115634564e-6  # program kB / CODATA 2022 kB

# docs/gxtb-v2.0.1-spec.md, "Finite electronic temperature" and "kB of the default conventions"
# (also docs/validation.md, g-xTB at finite electronic temperature), the rounding-sensitive runs
# left out. 2.0.1, default thresholds: energies <= 1.8e-12 Eh, (H2O)100 1.1e-11, the same
# iterations, gradients <= 4.7e-11 Eh/bohr; tight: energies <= 2.0e-12 Eh, (H2O)100 9.1e-12,
# gradients <= 2.9e-11, failures as the program's; v1.1: energies <= 2e-11 Eh, the same
# iterations. Default conventions: the kB part <= 4.5e-8 Eh (EuCl3 at 5000 K), the whole
# difference to the program <= 1.9e-6 Eh ((H2O)100, the revD4 three-body tail).
LIMITS = {
    "v2 default max_abs_dE": 1.8e-12, "v2 default max_abs_dE water100": 1.1e-11, "v2 default max_dgrad": 4.7e-11,
    "v2 default iteration_mismatches": 0, "v2 default failures": 0,
    "v2 tight max_abs_dE": 2.0e-12, "v2 tight max_abs_dE water100": 9.1e-12, "v2 tight max_dgrad": 2.9e-11,
    "v2 tight iteration_mismatches": 0, "v2 tight failures": 0,
    "v1.1 max_abs_dE": 2e-11, "v1.1 iteration_mismatches": 0, "v1.1 failures": 0,
    "v2 default conventions max_abs_kB_part": 4.5e-8, "v2 default conventions max_abs_dE_vs_program": 1.9e-6,
}
# The rounding-sensitive runs ("<version> <molecule>[ <T> K]"; spec §4.17.1 and its table):
# perturbing kT by one ulp in axtb's SCF moves each by as much as its difference to the program
KNOWN_DIFFERENCES = {
    "v2 default Fe4": "Fe4 at 300 K fails in both; at 1000/3000/5000 K 1.5e-6, 2.5e-6, 4.8e-7 Eh (33 vs 32 iterations at 3000 K)",
    "v2 default TmH3 3000 K": "1.4e-6 Eh, 75 vs 72 iterations",
    "v2 default TmH3 5000 K": "157 iterations or no convergence within 250, depending on the round-off (the program: 158)",
    "v2 default ErF3 5000 K": "5.7e-9 Eh",
    "v2 tight Fe4 1000 K": "75-250 iterations depending on the round-off, where the program converges",
    "v2 tight CeCl3 5000 K": "iteration counts (75 vs 51), energies equal to 1e-12 Eh",
    "v2 tight ErF3 3000 K": "iteration counts (29 vs 31), energies equal to 1e-12 Eh",
    "v2 tight ErF3 5000 K": "iteration counts (78 vs 79), energies equal to 1e-12 Eh",
    "v2 tight N2_stretched 5000 K": "iteration counts move with the round-off, energies equal to 1e-12 Eh",
    "v2 tight EuCl3 5000 K": "102-250 iterations depending on the round-off; the program stops at 250",
    "v1.1 Fe4": "as for 2.0.1: slowly converging SCFs whose last digits, or convergence, follow the round-off",
}


def sensitive(version: str, name: str, t: float) -> bool:
    """Whether a run ("v2 default", "v2 tight", "v1.1") is one of the known rounding-sensitive ones."""
    return f"{version} {name}" in KNOWN_DIFFERENCES or f"{version} {name} {t:g} K" in KNOWN_DIFFERENCES


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
        res = {"rounding_sensitive": sensitive(f"v2 {key}", r["name"], t)}
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
    row["rounding_sensitive"] = sensitive("v1.1", r["name"], r["etemp"])
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
    return check(v2, v1)


def check(v2: list[dict], v1: list[dict]) -> int:
    """The gate (tests/gate.py): every run except the rounding-sensitive ones."""
    def name(r):
        return f'{r["molecule"]} {r["etemp"]:g} K'

    def failed(res):  # axtb fails where the program converges, or converges where it fails
        return "error" in res or (res.get("program_fails") and not res["axtb_fails"])

    summary, known = {}, {}
    for key in ("default", "tight"):
        good = [r for r in v2 if "dE" in r[key] and not r[key]["rounding_sensitive"]]
        summary |= {
            f"v2 {key} max_abs_dE": max((abs(r[key]["dE"]) for r in good if r["molecule"] != "water100"), default=None),
            f"v2 {key} max_abs_dE water100": max((abs(r[key]["dE"]) for r in good if r["molecule"] == "water100"),
                                                 default=None),
            f"v2 {key} max_dgrad": max((r[key]["dgrad"] for r in good), default=None),
            f"v2 {key} iteration_mismatches": [name(r) for r in good if r[key]["iter_axtb"] != r[key]["iter_program"]],
            f"v2 {key} failures": [name(r) for r in v2 if failed(r[key]) and not r[key]["rounding_sensitive"]],
        }
        for r in v2:
            for k in (f"v2 {key} {r['molecule']}", f"v2 {key} {name(r)}"):
                if r[key]["rounding_sensitive"] and k in KNOWN_DIFFERENCES:
                    known[k] = KNOWN_DIFFERENCES[k]
    g1 = [r for r in v1 if "dE" in r and not r["rounding_sensitive"]]
    summary |= {"v1.1 max_abs_dE": max((abs(r["dE"]) for r in g1), default=None),
                "v1.1 iteration_mismatches": [name(r) for r in g1 if r["iter_axtb"] != r["iter_program"]],
                "v1.1 failures": [name(r) for r in v1 if failed(r) and not r["rounding_sensitive"]]}
    known |= {f"v1.1 {r['molecule']}": KNOWN_DIFFERENCES[f"v1.1 {r['molecule']}"] for r in v1 if r["rounding_sensitive"]
              and f"v1.1 {r['molecule']}" in KNOWN_DIFFERENCES}
    dc = [r["default_conventions"] for r in v2 if "default_conventions" in r and not r["default"]["rounding_sensitive"]]
    summary |= {"v2 default conventions max_abs_kB_part": max((abs(x["kB_part"]) for x in dc), default=None),
                "v2 default conventions max_abs_dE_vs_program": max((abs(x["dE"]) for x in dc), default=None)}
    return gate.check(summary, LIMITS, known)


if __name__ == "__main__":
    raise SystemExit(main())
