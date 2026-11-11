#!/usr/bin/env python3
"""g-xTB 2.0.1 in axtb vs the g-xTB 2.0.1 program (grimme-lab/g-xtb 2.0.1, `xtb --gxtb`).

For every molecule of crates/gxtb/tests/reference_v2 (program outputs and per-term energies
recorded there) it runs the axtb CLI on the XYZ file

- with the program's conventions (`--conventions gxtb`): total energy, SCF iterations, the
  largest per-term deviation, and the analytic gradient against the program's at the default
  (`gradient`) and tight (`--acc 0.0001`, `gradient_tight`) thresholds;
- with the default conventions (CODATA 2022 units, integer electron count, smooth cutoffs):
  energy and iterations against the program;

and energy differences in both conventions (binding energies of the complexes in
tests/structures, split into fragments by connectivity; 2 % uniform expansions of reference
molecules), so that the default-vs-program differences of energy differences are visible.

    python tests/gxtb2_reproduction.py [--axtb target/release/axtb] [--out bench/results/gxtb2_reproduction.json]

The script fails (exit status 1) when a quantity exceeds its limit in `LIMITS` (tests/gate.py)
or a molecule fails.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path

import gate

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "crates/gxtb/tests/reference_v2"
METHOD = "gxtb-v2.0.1"
# docs/gxtb-v2.0.1-spec.md, "Reproduction" and "Default vs program conventions" (also
# docs/validation.md, Summary). Program conventions: total energies <= 2.1e-12 Eh for 83
# molecules (C50H102 2.05e-12 of 1966 Eh), 9.1e-12 for (H2O)100 (1.2e-15 relative); the same SCF iterations at the default and
# tight thresholds (TmH3's tight SCF fails in both); per-term energies <= 9.1e-12 Eh, TbCl3
# 2.6e-11 (a near-degenerate HOMO set carries the program's fractional electron count, so the
# terms of the last, unconverged density follow the round-off); gradients <= 2e-11 Eh/bohr
# (default) and <= 6e-11 (tight), CeCl3 tight 4.0e-9 (seven near-degenerate f orbitals, where
# the summation order of the density alone moves it by up to 6e-8). Default conventions: <= 8.7e-10
# Eh for the 82 small molecules with the same iterations, C50H102 5.3e-8, (H2O)100 1.9e-6 (the
# revD4 three-body tail); in energy differences, bindings <= 1.7e-11 Eh, 2 % expansions <= 8.7e-10,
# C50H102's 2.3e-8.
LIMITS = {
    "failures": 0, "max_abs_dE": 2.1e-12, "max_abs_dE water100": 9.1e-12, "iteration_mismatches": 0,
    "tight_iteration_mismatches": 0, "tight_scf_failure_mismatches": 0, "max_dterm": 9.1e-12, "max_dterm TbCl3": 2.6e-11,
    "max_dgrad_default": 2e-11, "max_dgrad_tight": 6e-11, "max_dgrad_tight CeCl3": 4.0e-9,
    "max_abs_dE_default": 8.7e-10, "max_abs_dE_default c50h102": 5.3e-8, "max_abs_dE_default water100": 1.9e-6,
    "default_iteration_mismatches": 0, "binding default-program": 1.7e-11, "expansion default-program": 8.7e-10,
    "expansion default-program c50h102": 2.3e-8,
}
LARGE = {"water100", "c50h102"}  # the two molecules longer than 50 bohr
# program term name -> axtb CLI term name
TERMS = {"first_order": "first_order", "second_order": "second_order", "third_order": "third_order",
         "fourth_order": "fourth_order", "aes": "es_multipole", "exchange": "exchange", "spin": "spin_polarization",
         "dispersion_sc": "dispersion", "repulsion": "repulsion", "dispersion_nonsc": "dispersion_three_body",
         "increment": "increments"}
# covalent radii (Angstrom, Pyykko) for the fragment split
RCOV = {"H": 0.32, "C": 0.75, "N": 0.71, "O": 0.63, "F": 0.64, "S": 1.03, "Cl": 0.99, "Br": 1.14, "I": 1.33, "P": 1.11,
        "Si": 1.16, "B": 0.85}
ZVAL = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "S": 16, "Cl": 17, "Br": 35, "I": 53, "P": 15, "Si": 14, "B": 5}


def total_energy(printout: str) -> float:
    line = next(l for l in printout.splitlines() if ":: total energy" in l)
    return float(line.split()[3])


def read_gradient(path: Path) -> list[list[float]]:
    lines = path.read_text().splitlines()
    rows = [l.split() for l in lines[2:] if not l.startswith("$")]
    nat = sum(1 for r in rows if len(r) == 4)
    return [[float(x.replace("D", "E")) for x in r] for r in rows[nat:2 * nat]]


def maxdiff(a, b) -> float:
    return max(abs(x - y) for ra, rb in zip(a, b) for x, y in zip(ra, rb))


def run(axtb: str, xyz: Path, conv: str, charge: int, uhf: int, *extra: str) -> dict:
    out = subprocess.run([axtb, str(xyz), "--method", METHOD, "--conventions", conv, "--chrg", str(charge),
                          "--uhf", str(uhf), *extra], capture_output=True, text=True)
    if out.returncode != 0:
        return {"error": (out.stderr + out.stdout).strip()[-300:]}
    return json.loads(out.stdout)


def molecules():
    for group in ("reference", "elements", "large"):
        for d in sorted((REF / group).iterdir()):
            if (d / "summary.json").exists():
                yield group, d


def fragments(text: str):
    """Connected components (covalent radii) of an XYZ structure: lists of atom lines."""
    lines = text.splitlines()
    nat = int(lines[0])
    atoms = [l.split() for l in lines[2:2 + nat]]
    xyz = [[float(v) for v in a[1:4]] for a in atoms]
    parent = list(range(nat))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(nat):
        for j in range(i):
            r = math.dist(xyz[i], xyz[j])
            if r < 1.25 * (RCOV.get(atoms[i][0], 1.5) + RCOV.get(atoms[j][0], 1.5)):
                parent[find(i)] = find(j)
    comps: dict[int, list[int]] = {}
    for i in range(nat):
        comps.setdefault(find(i), []).append(i)
    return [[lines[2 + i] for i in c] for c in comps.values()], atoms


def write_xyz(path: Path, atom_lines: list[str]):
    path.write_text(f"{len(atom_lines)}\n\n" + "\n".join(atom_lines) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--axtb", default=os.environ.get("AXTB_BIN", str(ROOT / "target/release/axtb")))
    ap.add_argument("--out", type=Path, default=ROOT / "bench/results/gxtb2_reproduction.json")
    a = ap.parse_args()
    rows, diffs = [], []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for group, d in molecules():
            inp = json.loads((d / "input.json").read_text())
            summ = json.loads((d / "summary.json").read_text())
            q, u = inp["charge"], inp["uhf"]
            xyz = d / "mol.xyz"
            e_prog = total_energy((d / "xtb.out").read_text())
            p = run(a.axtb, xyz, "gxtb", q, u, "--grad")
            pt = run(a.axtb, xyz, "gxtb", q, u, "--grad", "--acc", "0.0001", "--max-iter", "250")
            c = run(a.axtb, xyz, "axtb", q, u, "--grad")
            # the program's tight SCF can fail (TmH3: no convergence in its 250 iterations); then
            # ours must fail as well
            tight_failed = summ["tight"]["total_energy"] is None
            if tight_failed:
                pt = {"tight_failed_as_program": "error" in pt}
            err = p.get("error") or pt.get("error") or c.get("error")
            row = {"molecule": d.name.replace("_", " "), "group": group, "natoms": int(xyz.read_text().split()[0]),
                   "charge": q, "uhf": u, "E_program": e_prog, "iter_program": summ["scf_iterations"],
                   "iter_program_tight": summ["tight"]["scf_iterations"], "error": err}
            if not err:
                dterm = max(abs(p["terms"][TERMS[k]] - v) for k, v in summ["terms"].items() if k != "increment")
                row.update({
                    "E_axtb": p["energy"], "dE": p["energy"] - e_prog, "iter_axtb": p["iterations"],
                    "max_dterm": dterm, "max_dgrad_default": maxdiff(p["gradient"], read_gradient(d / "gradient")),
                    "E_default": c["energy"], "dE_default": c["energy"] - e_prog, "iter_default": c["iterations"],
                    "seconds_axtb": p["seconds"],
                })
                if tight_failed:
                    row["tight"] = ("both tight SCFs fail (250 iterations)" if pt["tight_failed_as_program"]
                                    else "program tight SCF fails, axtb's converges")
                else:
                    row.update({"iter_axtb_tight": pt["iterations"],
                                "max_dgrad_tight": maxdiff(pt["gradient"], read_gradient(d / "gradient_tight")),
                                "max_dgrad_default_vs_program_tight": maxdiff(c["gradient"], read_gradient(d / "gradient_tight"))})
            rows.append(row)
            print(f"{d.name:22} dE {row.get('dE', float('nan')):9.1e} terms {row.get('max_dterm', float('nan')):8.1e} "
                  f"it {row.get('iter_axtb')}/{row['iter_program']} grad tight {row.get('max_dgrad_tight', float('nan')):8.1e} "
                  f"default conv dE {row.get('dE_default', float('nan')):9.1e}", flush=True)

        def energy(path: Path, conv: str, charge: int = 0, uhf: int = 0) -> float:
            r = run(a.axtb, path, conv, charge, uhf)
            if "error" in r:
                raise RuntimeError(f"{path.name}: {r['error']}")
            return r["energy"]

        # binding energies of the complexes in tests/structures (neutral closed-shell fragments)
        for f in sorted((ROOT / "tests/structures").glob("*.xyz")):
            frags, atoms = fragments(f.read_text())
            if len(frags) != 2 or any(a[0] not in ZVAL for a in atoms):
                continue
            if any(sum(ZVAL[l.split()[0]] for l in fr) % 2 for fr in frags):
                continue
            paths = [tmp / f"{f.stem}-{k}.xyz" for k in range(2)]
            for path, fr in zip(paths, frags):
                write_xyz(path, fr)
            res = {}
            for conv in ("gxtb", "axtb"):
                res[conv] = energy(f, conv) - sum(energy(path, conv) for path in paths)
            diffs.append({"quantity": f"{f.stem} binding (fragments {len(frags[0])} + {len(frags[1])} atoms)",
                          "program_conventions_kcal": res["gxtb"] * 627.5094740631, "default_minus_program_Eh": res["axtb"] - res["gxtb"]})
            print(diffs[-1], flush=True)
        # deformation energies: 2 % uniform expansion
        for rel in ("reference/water", "reference/caffeine", "reference/ferrocene", "reference/nickel_tetracarbonyl",
                    "reference/tetrachlorocuprate", "elements/UF6", "elements/GdCl3", "large/c50h102"):
            d = REF / rel
            inp = json.loads((d / "input.json").read_text())
            lines = (d / "mol.xyz").read_text().splitlines()
            nat = int(lines[0])
            scaled = [" ".join([l.split()[0]] + [f"{1.02 * float(x):.8f}" for x in l.split()[1:4]]) for l in lines[2:2 + nat]]
            path = tmp / f"{d.name}-x1.02.xyz"
            write_xyz(path, scaled)
            res = {}
            for conv in ("gxtb", "axtb"):
                res[conv] = energy(path, conv, inp["charge"], inp["uhf"]) - energy(d / "mol.xyz", conv, inp["charge"], inp["uhf"])
            diffs.append({"quantity": f"{d.name} 2 % expansion", "program_conventions_kcal": res["gxtb"] * 627.5094740631,
                          "default_minus_program_Eh": res["axtb"] - res["gxtb"]})
            print(diffs[-1], flush=True)
    ok = [r for r in rows if not r["error"]]
    summary = {
        "molecules": len(rows),
        "max_abs_dE": max(abs(r["dE"]) for r in ok),
        "iteration_mismatches": [r["molecule"] for r in ok if r["iter_axtb"] != r["iter_program"]],
        "tight_iteration_mismatches": [r["molecule"] for r in ok if "iter_axtb_tight" in r and r["iter_axtb_tight"] != r["iter_program_tight"]],
        "tight_scf_failures": {r["molecule"]: r["tight"] for r in ok if "tight" in r},
        "max_dterm": max(r["max_dterm"] for r in ok),
        "max_dgrad_default": max(r["max_dgrad_default"] for r in ok),
        "max_dgrad_tight": max(r["max_dgrad_tight"] for r in ok if "max_dgrad_tight" in r),
        "max_dgrad_tight_by_molecule": sorted(((r["max_dgrad_tight"], r["molecule"]) for r in ok if "max_dgrad_tight" in r))[-3:],
        "max_abs_dE_default": max(abs(r["dE_default"]) for r in ok),
        "default_iteration_mismatches": [r["molecule"] for r in ok if r["iter_default"] != r["iter_program"]],
    }
    a.out.write_text(json.dumps({"program": "g-xTB 2.0.1 (grimme-lab/g-xtb 2.0.1: xtb 6.7.1 --gxtb)", "method": METHOD,
                                 "summary": summary, "rows": rows, "energy_differences": diffs}, indent=1))
    print(json.dumps(summary, indent=1))
    print("wrote", a.out)
    return check(rows, diffs)


def check(rows: list[dict], diffs: list[dict]) -> int:
    """The gate (tests/gate.py): the molecules of `LIMITS` with their own limits, the rest together."""
    ok = [r for r in rows if not r["error"]]

    def worst(key, keep=lambda r: True):
        return max((abs(r[key]) for r in ok if key in r and keep(r)), default=None)

    def one(key, molecule):
        return worst(key, lambda r: r["molecule"] == molecule)

    expansion = [d for d in diffs if "expansion" in d["quantity"]]
    summary = {
        "failures": [r["molecule"] for r in rows if r["error"]],
        "max_abs_dE": worst("dE", lambda r: r["molecule"] != "water100"), "max_abs_dE water100": one("dE", "water100"),
        "iteration_mismatches": [r["molecule"] for r in ok if r["iter_axtb"] != r["iter_program"]],
        "tight_iteration_mismatches": [r["molecule"] for r in ok if "iter_axtb_tight" in r
                                       and r["iter_axtb_tight"] != r["iter_program_tight"]],
        "tight_scf_failure_mismatches": [r["molecule"] for r in ok if r.get("tight", "").startswith("program tight SCF fails")],
        "max_dterm": worst("max_dterm", lambda r: r["molecule"] != "TbCl3"), "max_dterm TbCl3": one("max_dterm", "TbCl3"),
        "max_dgrad_default": worst("max_dgrad_default"),
        "max_dgrad_tight": worst("max_dgrad_tight", lambda r: r["molecule"] != "CeCl3"),
        "max_dgrad_tight CeCl3": one("max_dgrad_tight", "CeCl3"),
        "max_abs_dE_default": worst("dE_default", lambda r: r["molecule"] not in LARGE),
        "max_abs_dE_default c50h102": one("dE_default", "c50h102"),
        "max_abs_dE_default water100": one("dE_default", "water100"),
        "default_iteration_mismatches": [r["molecule"] for r in ok if r["iter_default"] != r["iter_program"]],
        "binding default-program": max((abs(d["default_minus_program_Eh"]) for d in diffs if "binding" in d["quantity"]),
                                       default=None),
        "expansion default-program": max((abs(d["default_minus_program_Eh"]) for d in expansion
                                          if not d["quantity"].startswith("c50h102")), default=None),
        "expansion default-program c50h102": max((abs(d["default_minus_program_Eh"]) for d in expansion
                                                  if d["quantity"].startswith("c50h102")), default=None),
    }
    return gate.check(summary, LIMITS)


if __name__ == "__main__":
    raise SystemExit(main())
