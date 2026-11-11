#!/usr/bin/env python3
"""Golden outputs for refactoring: run a fixed corpus of CLI requests (every method, convention
and feature, a few thread counts and the run-time LAPACK path) and compare outputs bit for bit.

    scripts/golden.py record DIR [--axtb BIN]     run the corpus, write DIR/<case>.json
    scripts/golden.py compare DIR [--axtb BIN]    run it again and compare with DIR
    scripts/golden.py list                        the cases

Outputs are compared exactly (every float, every field) except the timing fields. Golden files
are machine-specific (the matrix-product kernels depend on the CPU), so keep them local.
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "tests/structures"
R2 = ROOT / "crates/gxtb/tests/reference_v2"
BOHR = 0.529177210544
SYMBOLS = ["X", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
           "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn"]
TIMING = {"timings", "hessian_timings", "seconds"}
# a run-time LAPACK for the eigensolver path through the library (optional)
MKL = os.environ.get("AXTB_GOLDEN_LAPACK", "/usr/lib/x86_64-linux-gnu/libmkl_rt.so")


def mol(path: Path, charge=0.0, uhf=None):
    lines = path.read_text().split("\n")
    n = int(lines[0])
    atoms = [l.split() for l in lines[2:2 + n]]
    req = {"numbers": [SYMBOLS.index(a[0].capitalize()) for a in atoms],
           "positions": [[float(x) / BOHR for x in a[1:4]] for a in atoms], "charge": charge}
    if uhf is not None:
        req["uhf"] = uhf
    return req


WATER, CAFF, TAXOL = S / "water.xyz", S / "caffeine.xyz", S / "taxol.xyz"
W100, W333, C50 = S / "large/water100.xyz", S / "large/water333.xyz", S / "large/c50h102.xyz"
O2, CU, FC = R2 / "reference/dioxygen_triplet/mol.xyz", S / "cu-nh3-4.xyz", S / "ferrocene.xyz"


def corpus():
    """(name, request, extra environment)"""
    c = []
    add = lambda name, base, env=None, **kw: c.append((name, {**base, **kw}, env or {}))
    w, caf, tax = mol(WATER), mol(CAFF), mol(TAXOL)
    o2, cu = mol(O2, 0.0, 2), mol(CU, 2.0, 1)
    # GFN2-xTB
    add("gfn2-water-energy", w, method="gfn2")
    for conv in ("axtb", "xtb", "xtb671", "tblite"):
        add(f"gfn2-caffeine-grad-{conv}", caf, method="gfn2", conventions=conv, gradient=True)
    add("gfn2-taxol-grad-t1", tax, method="gfn2", gradient=True, threads=1)
    add("gfn2-taxol-grad-t4", tax, method="gfn2", gradient=True, threads=4)
    add("gfn2-water100-grad-t1", mol(W100), method="gfn2", gradient=True, threads=1)
    add("gfn2-water100-grad-t4", mol(W100), method="gfn2", gradient=True, threads=4)
    add("gfn2-water100-grad-t4-lapack", mol(W100), {"AXTB_LAPACK": MKL}, method="gfn2", gradient=True, threads=4)
    add("gfn2-taxol-grad-t1-lapack", tax, {"AXTB_LAPACK": MKL}, method="gfn2", gradient=True, threads=1)
    add("gfn2-o2-triplet-grad", o2, method="gfn2", gradient=True)
    add("gfn2-cu-doublet-grad", cu, method="gfn2", gradient=True)
    add("gfn2-caffeine-alpb", caf, method="gfn2", gradient=True, alpb="water")
    add("gfn2-caffeine-alpb-bar1m", caf, method="gfn2", gradient=True, alpb="water", solvation_state="bar1m")
    add("gfn2-caffeine-gbsa", caf, method="gfn2", gradient=True, gbsa="water")
    add("gfn2-caffeine-etemp", caf, method="gfn2", gradient=True, etemp=3000.0)
    add("gfn2-caffeine-nodisp", caf, method="gfn2", gradient=True, dispersion=False)
    for mixer in ("broyden", "diis"):
        add(f"gfn2-caffeine-mixer-{mixer}", caf, method="gfn2", gradient=True, mixer=mixer)
    add("gfn2-caffeine-hessian", caf, method="gfn2", hessian=True)
    add("gfn2-caffeine-hessian-t4", caf, method="gfn2", hessian=True, threads=4)
    add("gfn2-caffeine-hessian-xtb", caf, method="gfn2", hessian=True, conventions="xtb")
    add("gfn2-water-hessian-numerical", w, method="gfn2", hessian=True, hessian_method="numerical")
    add("gfn2-caffeine-hessian-alpb", caf, method="gfn2", hessian=True, alpb="water")
    add("gfn2-water-molden", w, method="gfn2", molden=True)
    add("gfn2-caffeine-bond-orders", caf, method="gfn2", bond_orders=True)
    # g-xTB 2.0.1
    add("gxtb2-water-energy", w, method="gxtb-v2.0.1")
    for conv in ("axtb", "gxtb"):
        add(f"gxtb2-caffeine-grad-{conv}", caf, method="gxtb-v2.0.1", conventions=conv, gradient=True)
    add("gxtb2-cu-doublet-grad", cu, method="gxtb-v2.0.1", gradient=True)
    add("gxtb2-o2-triplet-grad", o2, method="gxtb-v2.0.1", gradient=True)
    add("gxtb2-caffeine-etemp", caf, method="gxtb-v2.0.1", gradient=True, etemp=3000.0)
    add("gxtb2-taxol-grad-t1", tax, method="gxtb-v2.0.1", gradient=True, threads=1)
    add("gxtb2-taxol-grad-t4", tax, method="gxtb-v2.0.1", gradient=True, threads=4)
    add("gxtb2-c50h102-grad-t4-lapack", mol(C50), {"AXTB_LAPACK": MKL}, method="gxtb-v2.0.1", gradient=True, threads=4)
    add("gxtb2-caffeine-hessian", caf, method="gxtb-v2.0.1", hessian=True)
    add("gxtb2-caffeine-hessian-t4", caf, method="gxtb-v2.0.1", hessian=True, threads=4)
    add("gxtb2-cu-hessian", cu, method="gxtb-v2.0.1", hessian=True)
    add("gxtb2-water-hessian-response", w, method="gxtb-v2.0.1", hessian=True, hessian_method="response")
    add("gxtb2-water-hessian-numerical", w, method="gxtb-v2.0.1", hessian=True, hessian_method="numerical")
    add("gxtb2-caffeine-composite", caf, method="gxtb-v2.0.1", gradient=True, alpb="water", solvation_correction="gfn2")
    add("gxtb2-water-composite-hessian", w, method="gxtb-v2.0.1", hessian=True, alpb="water", solvation_correction="gfn2")
    add("gxtb2-water-molden", w, method="gxtb-v2.0.1", molden=True)
    add("gxtb2-caffeine-bond-orders", caf, method="gxtb-v2.0.1", bond_orders=True)
    # g-xTB v1.1
    for conv in ("axtb", "gxtb"):
        add(f"gxtb11-caffeine-grad-{conv}", caf, method="gxtb-v1.1", conventions=conv, gradient=True)
    add("gxtb11-cu-doublet-grad", cu, method="gxtb-v1.1", gradient=True)
    add("gxtb11-water-hessian", w, method="gxtb-v1.1", hessian=True)
    # GFN-FF
    for conv in ("axtb", "xtb", "xtb671"):
        add(f"gfnff-caffeine-grad-{conv}", caf, method="gfnff", conventions=conv, gradient=True)
    add("gfnff-taxol-grad", tax, method="gfnff", gradient=True)
    add("gfnff-water333-grad-t4", mol(W333), method="gfnff", gradient=True, threads=4)
    add("gfnff-caffeine-hessian", caf, method="gfnff", hessian=True)
    add("gfnff-ferrocene-grad", mol(FC), method="gfnff", gradient=True)
    # vibrational analysis of a given Hessian (identity-like test matrix of water)
    add("analysis-water", {k: w[k] for k in ("numbers", "positions")},
        analyze_hessian=[[0.5 if i == j else 0.01 for j in range(9)] for i in range(9)])
    # errors and refusals
    add("error-negative-uhf", w, uhf=-1)
    add("error-unknown-method", w, method="gfn3")
    add("error-gxtb-alpb", w, method="gxtb-v2.0.1", alpb="water")
    add("error-scf-not-converged", caf, method="gfn2", max_iter=2)
    add("error-zero-threads", w, threads=0)
    add("error-hessian-step-analytic", w, method="gfn2", hessian=True, hessian_step=0.01)
    add("error-gfnff-molden", w, method="gfnff", molden=True)
    return c


def run(axtb: str, req: dict, env_extra: dict):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("AXTB_LAPACK", "OMP_", "MKL_", "RAYON"))}
    env.update(env_extra)
    p = subprocess.run([axtb, "--json"], input=json.dumps(req), capture_output=True, text=True, env=env)
    try:
        out = json.loads(p.stdout)
    except json.JSONDecodeError:
        out = {"unparsed_stdout": p.stdout[-2000:]}
    return {"exit_code": p.returncode, "output": strip_timings(out)}


def strip_timings(x):
    """The output without its timing fields, at any depth."""
    if isinstance(x, dict):
        return {k: strip_timings(v) for k, v in x.items() if k not in TIMING}
    if isinstance(x, list):
        return [strip_timings(v) for v in x]
    return x


def diff(a, b, path=""):
    """List of (path, description) where a and b differ exactly."""
    if type(a) is not type(b) and not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        return [(path, f"type {type(a).__name__} vs {type(b).__name__}")]
    if isinstance(a, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append((f"{path}.{k}", "missing in " + ("new" if k not in b else "golden")))
            else:
                out += diff(a[k], b[k], f"{path}.{k}")
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return [(path, f"length {len(a)} vs {len(b)}")]
        flat_a, flat_b = flatten(a), flatten(b)
        if flat_a is not None and flat_b is not None and len(flat_a) == len(flat_b):
            bad = [(x, y) for x, y in zip(flat_a, flat_b) if not (x == y or (isinstance(x, float) and math.isnan(x) and math.isnan(y)))]
            if bad:
                m = max(abs(x - y) for x, y in bad if isinstance(x, (int, float)) and isinstance(y, (int, float)))
                return [(path, f"{len(bad)} of {len(flat_a)} numbers differ, max |d| {m:.3e}")]
            return []
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff(x, y, f"{path}[{i}]")
        return out
    if a != b:
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return [(path, f"{a!r} vs {b!r} (|d| {abs(a - b):.3e})")]
        return [(path, f"{str(a)[:80]!r} vs {str(b)[:80]!r}")]
    return []


def flatten(x):
    """All numbers of a nested list of numbers, or None."""
    out = []
    stack = [x]
    while stack:
        v = stack.pop()
        if isinstance(v, list):
            stack.extend(reversed(v))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out.append(v)
        else:
            return None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["record", "compare", "list"])
    ap.add_argument("dir", nargs="?")
    ap.add_argument("--axtb", default=str(ROOT / "target/release/axtb"))
    ap.add_argument("--only", default="", help="comma-separated case-name substrings")
    a = ap.parse_args()
    cases = corpus()
    if a.only:
        keys = a.only.split(",")
        cases = [c for c in cases if any(k in c[0] for k in keys)]
    if a.mode == "list":
        for name, req, env in cases:
            print(name, json.dumps({k: v for k, v in req.items() if k not in ("numbers", "positions")}), env or "")
        return
    d = Path(a.dir)
    sha = hashlib.sha256(Path(a.axtb).read_bytes()).hexdigest()[:16]
    if a.mode == "record":
        d.mkdir(parents=True, exist_ok=True)
        for name, req, env in cases:
            (d / f"{name}.json").write_text(json.dumps(run(a.axtb, req, env)))
            print("recorded", name, flush=True)
        (d / "_meta.json").write_text(json.dumps({"axtb": a.axtb, "sha256": sha, "cases": len(cases)}))
        return
    failed = 0
    for name, req, env in cases:
        gold = json.loads((d / f"{name}.json").read_text())
        new = run(a.axtb, req, env)
        ds = diff(gold, new)
        if ds:
            failed += 1
            print(f"DIFF {name}")
            for p, msg in ds[:8]:
                print(f"     {p}: {msg}")
        else:
            print(f"same {name}", flush=True)
    print(f"{len(cases) - failed} of {len(cases)} identical (binary {sha})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
