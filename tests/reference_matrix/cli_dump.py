#!/usr/bin/env python3
"""Dump the release CLI's results for every reference-matrix case, to check that a change
leaves results byte-identical.

    python tests/reference_matrix/cli_dump.py AXTB_BIN OUT.jsonl [--conventions axtb,xtb,tblite]
                                              [--hessians analytic,response,numerical] [--jobs N]
    python tests/reference_matrix/cli_dump.py --compare A.jsonl B.jsonl
    python tests/reference_matrix/cli_dump.py --compare-numbers A.jsonl B.jsonl [ENERGY GRADIENT]

For every case of `cases.py` and every conventions: an energy+gradient run (`--json`
request with the case's options), and for the cases with Hessian checks one run per Hessian
method. Each line holds the case, the run and the CLI's JSON response without wall times
(`seconds`, `timings`, `hessian_timings`), or its exit code and error (`error`: the message,
`error_kind`: the kind of the JSON error). `--compare` reports, per conventions, how many runs
are byte-identical and lists the others.

`--compare-numbers A.jsonl B.jsonl [ENERGY GRADIENT]` compares numbers only, for dumps of CLIs
whose responses differ in form: per conventions and run kind the largest difference of every
numeric field the two results share (energy, gradient, charges, dipole, terms, Hessian,
frequencies, thermochemistry, ...), changed iteration counts and changed failures; runs beyond
the limits (default energy 1e-11 Eh, gradient 1e-10 Eh/bohr) are listed. The wall times and
the response header (`program`, `version`, `method`, `conventions`, `eigensolver`) are not
compared.

`--gxtb` adds g-xTB: v1.1 on its reference molecules (`crates/gxtb/tests/reference`) and
2.0.1 on its reference and element sets (`crates/gxtb/tests/reference_v2`), each in the
`axtb` and `gxtb` conventions (energy and gradient; Hessians of a few small molecules: v1.1
numerical, 2.0.1 analytic, response and numerical). Their conventions are reported as
`gxtb-v1.1/axtb`, `gxtb-v2.0.1/gxtb`, ...

`--compare` compares the keys of the first file's results: keys that only the second file's
results have (outputs a newer CLI adds) are listed but do not make a run different.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cases as C  # noqa: E402

KEYS = ("charge", "uhf", "accuracy", "etemp", "max_iter", "threads", "dispersion", "alpb", "gbsa", "solvation_state")
HESSIAN_CHECKS = {"hessian", "threads_hessian", "analytic_ref", "analytic_numref", "analytic_xtb", "analytic_threads"}
TIMES = ("seconds", "timings", "hessian_timings")


def runs(conventions: list[str], hessians: list[str]):
    for case in C.all_cases():
        base = {k: case[k] for k in KEYS if k in case and case[k] is not None}
        base.update(numbers=case["numbers"], positions=case["positions"], method="gfn2", gradient=True)
        for conv in conventions:
            yield case["id"], conv, "gradient", dict(base, conventions=conv)
            if HESSIAN_CHECKS & set(case["checks"]):
                for hm in hessians:
                    yield case["id"], conv, f"hessian-{hm}", dict(base, conventions=conv, hessian=True, hessian_method=hm)


GXTB_REF = HERE.parents[1] / "crates/gxtb/tests"
AATOAU = 1.0 / 0.529177210903
# molecules whose g-xTB Hessians are dumped as well
GXTB_HESSIANS = {"water", "dioxygen_triplet", "ammonium", "LiF"}
SYMBOLS = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K",
           "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr",
           "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La",
           "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os",
           "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am",
           "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"]


def _xyz(path: Path) -> tuple[list[int], list[list[float]]]:
    lines = path.read_text().splitlines()
    nat = int(lines[0].split()[0])
    numbers, positions = [], []
    for line in lines[2:2 + nat]:
        f = line.split()
        numbers.append(SYMBOLS.index(f[0].capitalize()) + 1)
        positions.append([float(x) * AATOAU for x in f[1:4]])
    return numbers, positions


def gxtb_runs():
    """g-xTB runs (see the module docs): case id, conventions label, run, request."""
    sets = [("gxtb-v1.1", sorted(p for p in (GXTB_REF / "reference").iterdir() if (p / "mol.xyz").exists()),
             ["numerical"])]
    v2 = [d for g in ("reference", "elements") for d in sorted((GXTB_REF / "reference_v2" / g).iterdir())
          if (d / "mol.xyz").exists()]
    sets.append(("gxtb-v2.0.1", v2, ["analytic", "response", "numerical"]))
    for method, dirs, hessians in sets:
        for d in dirs:
            inp = json.loads((d / "input.json").read_text())
            numbers, positions = _xyz(d / "mol.xyz")
            base = {"numbers": numbers, "positions": positions, "charge": inp["charge"], "uhf": inp["uhf"],
                    "method": method, "gradient": True}
            for conv in ("axtb", "gxtb"):
                label = f"{method}/{conv}"
                yield d.name, label, "gradient", dict(base, conventions=conv)
                if d.name in GXTB_HESSIANS:
                    for hm in hessians:
                        yield d.name, label, f"hessian-{hm}", dict(base, conventions=conv, hessian=True,
                                                                  hessian_method=hm)


def run_one(args):
    exe, cid, conv, kind, req = args
    out = subprocess.run([exe, "--json"], input=json.dumps(req), capture_output=True, text=True)
    rec = {"case": cid, "conventions": conv, "run": kind, "rc": out.returncode}
    if out.returncode == 0:
        res = json.loads(out.stdout)
        for k in TIMES:
            res.pop(k, None)
        rec["result"] = res
    else:
        try:
            err = json.loads(out.stdout)["error"]
            rec["error"], rec["error_kind"] = err["message"], err["kind"]
        except (ValueError, KeyError, TypeError):
            rec["error"] = (out.stdout + out.stderr).strip()
    return json.dumps(rec, sort_keys=True)


def compare(a: Path, b: Path) -> int:
    def load(p):
        out = {}
        for line in p.read_text().splitlines():
            r = json.loads(line)
            out[(r["case"], r["conventions"], r["run"])] = r
        return out

    da, db = load(a), load(b)
    stats = defaultdict(lambda: [0, 0])
    diff = []
    added = defaultdict(int)
    for key in sorted(set(da) | set(db)):
        ra, rb = da.get(key), db.get(key)
        if ra is not None and rb is not None and "result" in ra and "result" in rb:
            # outputs the second CLI adds do not count
            new = set(rb["result"]) - set(ra["result"])
            for k in new:
                added[k] += 1
            rb = dict(rb, result={k: v for k, v in rb["result"].items() if k not in new})
        same = ra is not None and rb is not None and json.dumps(ra, sort_keys=True) == json.dumps(rb, sort_keys=True)
        stats[key[1]][0 if same else 1] += 1
        if not same:
            diff.append(key)
    for conv, (same, other) in sorted(stats.items()):
        print(f"{conv:18s} identical {same:5d}  different {other:5d}")
    if added:
        print("outputs only in", b.name, "(not compared):", ", ".join(f"{k} ({n})" for k, n in sorted(added.items())))
    for key in diff[:50]:
        print("  differs:", *key)
    return 0


# not compared by --compare-numbers: wall times and the response header
NOT_NUMBERS = set(TIMES) | {"program", "version", "method", "conventions", "eigensolver"}


def _maxdiff(a, b) -> float:
    """Largest |a - b| over the numbers of two JSON values of the same shape (inf if the shapes
    or the non-numeric values differ)."""
    if isinstance(a, bool) or isinstance(b, bool) or isinstance(a, str) or isinstance(b, str) or a is None or b is None:
        return 0.0 if a == b else float("inf")
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(a - b)
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return float("inf")
        return max((_maxdiff(x, y) for x, y in zip(a, b)), default=0.0)
    if isinstance(a, dict) and isinstance(b, dict):
        keys = (set(a) & set(b)) - NOT_NUMBERS
        return max((_maxdiff(a[k], b[k]) for k in keys), default=0.0)
    return float("inf")


def compare_numbers(a: Path, b: Path, elim: float, glim: float) -> int:
    def load(p):
        return {(r["case"], r["conventions"], r["run"]): r for r in map(json.loads, p.read_text().splitlines())}

    da, db = load(a), load(b)
    stats: dict = defaultdict(lambda: defaultdict(float))
    notes = []
    for key in sorted(set(da) | set(db)):
        ra, rb = da.get(key), db.get(key)
        group = key[1] + ("" if key[2] == "gradient" else "/" + key[2])
        s = stats[group]
        s["runs"] += 1
        if ra is None or rb is None:
            notes.append((key, "missing in " + (b.name if rb is None else a.name)))
            continue
        if ("result" in ra) != ("result" in rb):
            s["failure changes"] += 1
            notes.append((key, f"failure changed: {ra.get('error', 'ok')[:90]!r} -> {rb.get('error', 'ok')[:90]!r}"))
            continue
        if "result" not in ra:
            s["both failed"] += 1
            continue
        xa, xb = ra["result"], rb["result"]
        for k in (set(xa) & set(xb)) - NOT_NUMBERS:
            d = _maxdiff(xa[k], xb[k])
            s["max " + k] = max(s["max " + k], d)
        de, dg = abs(xa["energy"] - xb["energy"]), _maxdiff(xa.get("gradient"), xb.get("gradient"))
        if xa.get("iterations") != xb.get("iterations"):
            s["iteration changes"] += 1
            notes.append((key, f"iterations {xa.get('iterations')} -> {xb.get('iterations')} (dE {de:.1e})"))
        if de > elim or dg > glim:
            notes.append((key, f"dE {de:.2e} dG {dg:.2e}"))
    for group, s in sorted(stats.items()):
        main = ["energy", "gradient", "charges", "dipole", "terms", "hessian", "frequencies"]
        fields = "  ".join(f"{k} {s['max ' + k]:.1e}" for k in main if "max " + k in s)
        rest = sorted(k[4:] for k in s if k.startswith("max ") and k[4:] not in main and s[k] > 0)
        print(f"{group:30s} runs {int(s['runs']):4d}  {fields}  iteration changes {int(s['iteration changes'])}  "
              f"failure changes {int(s['failure changes'])}  both failed {int(s['both failed'])}")
        if rest:
            print("    other fields that differ:", ", ".join(f"{k} {s['max ' + k]:.1e}" for k in rest))
    for key, msg in notes:
        print("  ", *key, msg)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--compare-numbers", action="store_true")
    ap.add_argument("--conventions", default="axtb,xtb,tblite")
    ap.add_argument("--hessians", default="analytic,response,numerical")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--gxtb", action="store_true", help="add the g-xTB v1.1 and 2.0.1 runs")
    ap.add_argument("--only-gxtb", action="store_true", help="only the g-xTB runs")
    args = ap.parse_args()
    if args.compare_numbers:
        limits = [float(x) for x in args.paths[2:4]] + [1e-11, 1e-10][len(args.paths[2:4]):]
        return compare_numbers(Path(args.paths[0]), Path(args.paths[1]), *limits)
    if len(args.paths) != 2:
        ap.error("two paths: AXTB_BIN OUT.jsonl, or A.jsonl B.jsonl with --compare")
    if args.compare:
        return compare(Path(args.paths[0]), Path(args.paths[1]))
    exe, out = args.paths
    work = [] if args.only_gxtb else [
        (exe, *r) for r in runs(args.conventions.split(","), [h for h in args.hessians.split(",") if h])]
    if args.gxtb or args.only_gxtb:
        work += [(exe, *r) for r in gxtb_runs()]
    print(f"{len(work)} runs", file=sys.stderr)
    with Pool(args.jobs) as pool, open(out, "w") as fh:
        for i, line in enumerate(pool.imap(run_one, work, chunksize=4)):
            fh.write(line + "\n")
            if i % 500 == 0:
                print(f"  {i}/{len(work)}", file=sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
