#!/usr/bin/env python3
"""g-xTB in implicit solvent: axtb's in-process composite against three xtb program runs.

g-xTB has no implicit solvent of its own; the composite adds GFN2-xTB's solvation free energy:

    E_solv(x) = E_gxtb(x) + [E_GFN2+ALPB(x) - E_GFN2(x)]     (likewise gradients, Hessians)

Run with programs, that is three xtb calls per gradient (`xtb --gxtb`, `xtb --alpb water`,
`xtb`; here the g-xTB 2.0.1 build of xtb 6.7.1, `$AXTB_GXTB2`, see tests/programs.py). axtb runs
the same composite in one process
(`axtb.Calculator(method, ..., alpb="water", solvation_correction="gfn2")`). For each system:

1. **programs**: the three xtb runs (`structure.xyz --silent --chrg C --grad [--uhf N]` plus
   `--gxtb` or `--alpb water`, one fresh directory each,
   OMP_NUM_THREADS=1), with the composite's retry ladder for the GFN2 pair (both halves at
   `--iterations 1000 --etemp 1000`, then `--iterations 1000`, then `--iterations 1000
   --etemp 5000`). Energies from `TOTAL ENERGY` (12 decimals), gradients from `gradient`.
2. **axtb, program mode** (`conventions="gxtb"`: g-xTB in the 2.0.1 program's conventions,
   the GFN2 halves in xtb 6.7.1's): |dE| and max |dg| against the sum of the three program
   results, with positions read from the same XYZ file with mctc-lib's Angstrom-to-bohr factor.
3. **mepd** (optional): the Python implementation in the private mepd workflow, which
   motivated the composite (`SolvatedEngine(GXTBCalculator(...), "water")`, i.e.
   `SolvationCorrection`), run with mepd's Python (`$MEPD_PYTHON`); skipped without it.
4. **axtb, default conventions** (`"axtb"`): the difference to program mode (information).
5. **Hessian**: the composite's analytic Hessian against central differences (steps 1e-4 and
   2e-5 bohr) of the composite's analytic gradient with all three SCFs at accuracy 1e-7, in
   both conventions; max |dH| (at the better step: the ALPB surface term's C2 kinks spoil the
   larger step for caffeine) and the largest frequency difference (cm-1).
6. **Speed**: one composite gradient in-process (new calculator, and the next geometry on a
   reused calculator) against the three program calls and mepd's code path (wall times,
   one thread, best of `--repeat`).

Writes bench/results/solvation_composite.json. The script fails (exit status 1) when a quantity
exceeds its limit in `LIMITS` (tests/gate.py).

Usage: python tests/solvation_composite.py [--cases water,caffeine] [--repeat 3] [--no-hessian]
Needs the axtb Python package built from this tree and the g-xTB 2.0.1 build of xtb (`$AXTB_GXTB2`).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

import numpy as np

import axtb

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import gate  # noqa: E402
import programs as reference_programs  # noqa: E402  (this script has its own programs())

OUT = ROOT / "bench/results/solvation_composite.json"
MCTC_AATOAU = 1.8897261246204404  # the xtb 6.7.1 / g-xTB 2.0.1 programs' Angstrom-to-bohr factor (mctc-lib)
XTB = reference_programs.get(reference_programs.GXTB2)
MEPD_PYTHON = Path(os.environ["MEPD_PYTHON"]) if os.environ.get("MEPD_PYTHON") else None
METHOD = "gxtb-v2.0.1"
SOLVENT = "water"
# docs/validation.md (g-xTB in implicit solvent): program mode against the three programs, energies
# <= 9.7e-11 Eh, gradients <= 5.3e-10 Eh/bohr; mepd's own code path <= 9.1e-12 Eh; the analytic
# composite Hessian against central differences of the composite gradient <= 5.6e-8 Eh/bohr^2
# (at the better step), frequencies <= 2.5e-4 cm-1
LIMITS = {"max_abs_dE": 9.7e-11, "max_abs_dgrad": 5.3e-10, "max_abs_dE_mepd": 9.1e-12, "max_abs_dH": 5.6e-8,
          "max_dfreq_cm": 2.5e-4}
GATE_ENERGY = LIMITS["max_abs_dE"]
GATE_GRADIENT = LIMITS["max_abs_dgrad"]
GATE_HESSIAN = LIMITS["max_abs_dH"]  # Eh/bohr^2, analytic vs central differences of the composite gradient
# the composite's retry ladder for the GFN2 pair (after the first try)
PAIR_LADDER = [[], ["--iterations", "1000", "--etemp", "1000"], ["--iterations", "1000"],
               ["--iterations", "1000", "--etemp", "5000"]]
SYMBOLS = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl"]
# standard atomic weights (IUPAC, abridged), for harmonic frequencies only
MASSES = {1: 1.008, 6: 12.011, 7: 14.007, 8: 15.999, 9: 18.998, 11: 22.990, 17: 35.45}
AMU_TO_ME = 1.0 / 5.485799090441e-4
HARTREE_TO_CM = 2.1947463136314e7 / 100


def sodium_acetate() -> str:
    """Contact ion pair Na+ CH3COO- (bidentate, Na-O 2.30 A), built on the acetate geometry."""
    lines = (ROOT / "crates/gxtb/tests/reference/acetate/mol.xyz").read_text().splitlines()[2:9]
    atoms = [(l.split()[0], np.array([float(x) for x in l.split()[1:4]])) for l in lines]
    c2, o1, o2 = atoms[1][1], atoms[2][1], atoms[3][1]
    mid = 0.5 * (o1 + o2)
    u = (mid - c2) / np.linalg.norm(mid - c2)
    d = np.sqrt(2.30**2 - (0.5 * np.linalg.norm(o1 - o2)) ** 2)
    atoms.append(("Na", mid + d * u))
    return f"{len(atoms)}\nNa+ acetate- contact ion pair\n" + "".join(
        f"{s:2s} {x[0]:15.8f} {x[1]:15.8f} {x[2]:15.8f}\n" for s, x in atoms)


def sn2_image() -> str:
    """The asymmetric SN2 guess of bench/ts_example.py (Cl- + CH3Cl, C-Cl 2.25/2.45 A), an image
    of the reaction path."""
    guess = [("C", [0.0, 0.0, 0.0]), ("Cl", [0.0, 0.0, 2.25]), ("Cl", [0.05, 0.0, -2.45]), ("H", [1.07, 0.0, 0.08]),
             ("H", [-0.535, 0.927, 0.08]), ("H", [-0.535, -0.927, 0.08])]
    return f"{len(guess)}\nSN2 Cl- + CH3Cl path image\n" + "".join(
        f"{s:2s} {x[0]:15.8f} {x[1]:15.8f} {x[2]:15.8f}\n" for s, x in guess)


def from_file(rel: str) -> str:
    return (ROOT / rel).read_text()


# id: (label, xyz text, charge, unpaired electrons, Hessian check)
CASES = {
    "water": ("water", lambda: from_file("tests/structures/water.xyz"), 0, 0, True),
    "caffeine": ("caffeine", lambda: from_file("tests/structures/caffeine.xyz"), 0, 0, True),
    "acetate": ("acetate (anion)", lambda: from_file("crates/gxtb/tests/reference/acetate/mol.xyz"), -1, 0, True),
    "water-cation": ("water radical cation (doublet)", lambda: from_file("tests/structures/water.xyz"), 1, 1, True),
    "na-acetate": ("Na+ acetate- contact ion pair", sodium_acetate, 0, 0, True),
    "sn2-image": ("SN2 Cl- + CH3Cl path image (bench/ts_example.py guess)", sn2_image, -1, 0, True),
    "f2-stretched": ("F2 at 3.0 A (GFN2 pair needs the 1000 K retry)",
                     lambda: from_file("tests/scf_hard/scan-F2-3.0.xyz"), 0, 0, False),
    "hf-stretched": ("HF at 3.0 A (GFN2 pair needs the 5000 K retry)",
                     lambda: from_file("tests/scf_hard/scan-HF-3.0.xyz"), 0, 0, False),
    "taxol": ("taxol (speed)", lambda: from_file("tests/structures/taxol.xyz"), 0, 0, False),
}


def parse_xyz(text: str):
    lines = text.splitlines()
    nat = int(lines[0])
    nums, pos = [], []
    for line in lines[2:2 + nat]:
        f = line.split()
        nums.append(SYMBOLS.index(f[0].capitalize()) + 1 if not f[0].isdigit() else int(f[0]))
        pos.append([float(x) * MCTC_AATOAU for x in f[1:4]])
    return nums, np.array(pos)


def run_xtb(text: str, charge: int, uhf: int, flags: list[str], scratch: Path) -> dict:
    """One xtb program call, as the three-program composite starts it."""
    with tempfile.TemporaryDirectory(dir=scratch) as d:
        d = Path(d)
        (d / "structure.xyz").write_text(text)
        cmd = [XTB, "structure.xyz", "--silent", "--chrg", str(charge), "--grad"]
        if uhf:
            cmd += ["--uhf", str(uhf)]
        cmd += flags
        t0 = time.perf_counter()
        out = subprocess.run(cmd, cwd=d, capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="1"))
        seconds = time.perf_counter() - t0
        if out.returncode != 0:
            return {"ok": False, "seconds": seconds, "flags": flags}
        energy = float(re.findall(r"TOTAL ENERGY\s+(-?\d+\.\d+)", out.stdout)[-1])
        nat = int(text.split()[0])
        rows = [l.split() for l in (d / "gradient").read_text().replace("D", "E").splitlines()]
        rows = [r for r in rows if len(r) == 3]
        grad = np.array([[float(x) for x in r] for r in rows[-nat:]])
        return {"ok": True, "energy": energy, "gradient": grad, "seconds": seconds, "flags": flags}


def programs(text: str, charge: int, uhf: int, scratch: Path) -> dict:
    """The three programs, with the retry ladder for the GFN2 pair."""
    g = run_xtb(text, charge, uhf, ["--gxtb"], scratch)
    if not g["ok"]:
        raise RuntimeError("xtb --gxtb failed")
    seconds = g["seconds"]
    calls = 1
    for rung, extra in enumerate(PAIR_LADDER):
        gas = run_xtb(text, charge, uhf, extra, scratch)
        seconds += gas["seconds"]
        calls += 1
        if not gas["ok"]:
            continue
        solv = run_xtb(text, charge, uhf, ["--alpb", SOLVENT, *extra], scratch)
        seconds += solv["seconds"]
        calls += 1
        if solv["ok"]:
            break
    else:
        raise RuntimeError("GFN2 pair did not converge on any rung")
    dg = solv["energy"] - gas["energy"]
    return {"gxtb": g["energy"], "gfn2_solvated": solv["energy"], "gfn2_gas": gas["energy"], "solvation_correction": dg,
            "energy": g["energy"] + dg, "gradient": g["gradient"] + solv["gradient"] - gas["gradient"],
            "gradients": {"gxtb": g["gradient"], "gfn2_solvated": solv["gradient"], "gfn2_gas": gas["gradient"]},
            "retry": " ".join(PAIR_LADDER[rung]) or None, "calls": calls, "seconds": seconds}


MEPD_CODE = r"""
import json, sys, time, shutil
import numpy as np
from qcdata import Structure
from qcref.constants import BOHR_TO_ANGSTROM
from mepd.nodes.node import StructureNode
from mepd.engines.gxtb import GXTBCalculator
from mepd.solvation import SolvatedEngine

req = json.load(sys.stdin)
xtb = shutil.which(req["xtb"]) or req["xtb"]
out = []
for case in req["cases"]:
    lines = case["xyz"].splitlines()
    nat = int(lines[0])
    syms = [l.split()[0] for l in lines[2:2 + nat]]
    geom = np.array([[float(x) for x in l.split()[1:4]] for l in lines[2:2 + nat]]) / BOHR_TO_ANGSTROM
    s = Structure(symbols=syms, geometry=geom, charge=case["charge"], multiplicity=case["uhf"] + 1)
    best = None
    for _ in range(case["repeat"]):
        base = GXTBCalculator(executable=xtb, n_threads=1, n_parallel=1)
        engine = SolvatedEngine(base=base, solvent=req["solvent"], model="alpb", executable=xtb)
        node = StructureNode(structure=s, has_molecular_graph=False)
        t0 = time.perf_counter()
        g = engine.compute_gradients([node])[0]
        e = float(engine.compute_energies([node])[0])
        dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    out.append({"id": case["id"], "energy": e, "gradient": np.asarray(g).tolist(), "seconds": best,
                "fallbacks": [list(f) for f in engine.correction.fallbacks]})
print(json.dumps(out))
"""


def mepd_path(cases: list[dict], repeat: int) -> dict | None:
    if MEPD_PYTHON is None or not MEPD_PYTHON.exists():
        return None
    req = {"xtb": XTB, "solvent": SOLVENT, "cases": [dict(c, repeat=repeat) for c in cases]}
    out = subprocess.run([str(MEPD_PYTHON), "-c", MEPD_CODE], input=json.dumps(req), capture_output=True, text=True,
                         env=dict(os.environ, OMP_NUM_THREADS="1"))
    if out.returncode != 0:
        raise RuntimeError((out.stderr + out.stdout)[-2000:])
    return {r["id"]: r for r in json.loads(out.stdout.strip().splitlines()[-1])}


def timed(nums, pos, charge, uhf, conventions, repeat):
    """The result and the best wall time of one composite gradient with a new calculator, and of
    the next geometry on a reused one."""
    cold, warm = [], []
    for _ in range(repeat):
        t0 = time.perf_counter()
        calc = composite(nums, pos, charge, uhf, conventions)
        r = calc.singlepoint(pos, gradient=True)
        cold.append(time.perf_counter() - t0)
        shifted = pos + 1e-3 * np.cos(np.arange(pos.size)).reshape(pos.shape)
        t0 = time.perf_counter()
        calc.singlepoint(shifted, gradient=True)
        warm.append(time.perf_counter() - t0)
    return r, min(cold), min(warm)


def composite(nums, pos, charge, uhf, conventions, **kw):
    return axtb.Calculator(METHOD, nums, pos, charge=charge, uhf=uhf or None, alpb=SOLVENT, solvation_correction="gfn2",
                           conventions=conventions, **kw)


def frequencies(h, nums, pos):
    n = len(nums)
    c = pos - pos.mean(axis=0)
    vecs = []
    for k in range(3):
        t = np.zeros((n, 3))
        t[:, k] = 1.0
        vecs.append(t.ravel())
    for k in range(3):
        e = np.zeros(3)
        e[k] = 1.0
        vecs.append(np.cross(e, c).ravel())
    q, r = np.linalg.qr(np.array(vecs).T)
    q = q[:, np.abs(np.diag(r)) > 1e-8]
    p = np.eye(3 * n) - q @ q.T
    m = np.repeat([MASSES[z] * AMU_TO_ME for z in nums], 3)
    hm = (p @ h @ p) / np.sqrt(np.outer(m, m))
    w = np.linalg.eigvalsh(0.5 * (hm + hm.T))
    return np.sign(w) * np.sqrt(np.abs(w)) * HARTREE_TO_CM


def hessian_check(nums, pos, charge, uhf, conventions, steps=(1e-4, 2e-5)) -> dict:
    """The composite's analytic Hessian (default settings) against central differences of its
    gradient (all SCFs at accuracy 1e-7) with two steps. The ALPB surface term is C1 but not C2:
    a step of 1e-4 bohr can cross its switching kinks (caffeine 3e-6 Eh/bohr^2, all from the
    solvated half), 2e-5 rarely does, and SCF noise grows as the step shrinks. The check passes
    when one of the steps agrees within GATE_HESSIAN."""
    calc = composite(nums, pos, charge, uhf, conventions)
    t0 = time.perf_counter()
    h = calc.hessian(pos)
    seconds = time.perf_counter() - t0
    ha = np.asarray(h["hessian"])
    fa = frequencies(ha, nums, pos)
    real = np.abs(fa) > 50.0
    fd_calc = composite(nums, pos, charge, uhf, conventions, accuracy=1e-7)
    n3 = 3 * len(nums)
    per_step = {}
    for step in steps:
        fd = np.zeros((n3, n3))
        for k in range(n3):
            dp = np.zeros(n3)
            dp[k] = step
            gp = fd_calc.singlepoint(pos + dp.reshape(-1, 3), gradient=True)["gradient"].ravel()
            gm = fd_calc.singlepoint(pos - dp.reshape(-1, 3), gradient=True)["gradient"].ravel()
            fd[k] = (gp - gm) / (2 * step)
        fd = 0.5 * (fd + fd.T)
        ff = frequencies(fd, nums, pos)
        per_step[step] = (float(np.abs(ha - fd).max()), float(np.abs(fa - ff)[real].max()))
    best = min(per_step, key=lambda k: per_step[k][0])
    return {"conventions": conventions, "methods": {k: v.get("hessian_method") for k, v in h["parts"].items()},
            "max_abs_dH": per_step[best][0], "max_dfreq_cm": per_step[best][1], "step": best,
            "by_step": {f"{k:g}": {"max_abs_dH": v[0], "max_dfreq_cm": v[1]} for k, v in per_step.items()},
            "max_abs_H": float(np.abs(ha).max()), "seconds": seconds, "pass": gate.within(per_step[best][0], GATE_HESSIAN)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--repeat", type=int, default=3)
    ap.add_argument("--no-hessian", action="store_true")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    reference_programs.require(reference_programs.GXTB2)
    ids = args.cases.split(",")
    scratch = Path(tempfile.mkdtemp(prefix="solvcomp-", dir=os.environ.get("AXTB_SCRATCH")))
    rows, mepd_in = [], []
    for cid in ids:
        label, xyz, charge, uhf, _ = CASES[cid]
        mepd_in.append({"id": cid, "xyz": xyz(), "charge": charge, "uhf": uhf})
    print("mepd's code path ..." if MEPD_PYTHON else "mepd's code path: skipped ($MEPD_PYTHON not set)", flush=True)
    mepd = mepd_path(mepd_in, args.repeat)
    print(f"{'case':14s} {'nat':>4s} {'E composite (programs)':>22s} {'dE axtb':>9s} {'dg axtb':>9s} {'dE mepd':>9s} "
          f"{'retry':>12s} {'t prog':>7s} {'t mepd':>7s} {'t gxtb':>7s} {'warm':>7s} {'t axtb':>7s} {'warm':>7s}")
    for cid in ids:
        label, xyz, charge, uhf, do_hess = CASES[cid]
        text = xyz()
        nums, pos = parse_xyz(text)
        prog = min((programs(text, charge, uhf, scratch) for _ in range(args.repeat)), key=lambda r: r["seconds"])
        # axtb in program mode: a new calculator per call (cold) and the next geometry on a reused one (warm)
        r, cold, warm = timed(nums, pos, charge, uhf, "gxtb", args.repeat)
        dE = r["energy"] - prog["energy"]
        dg = float(np.abs(np.asarray(r["gradient"]) - prog["gradient"]).max())
        comp = {k: r["components"][k] - prog[k] for k in ("gxtb", "gfn2_solvated", "gfn2_gas", "solvation_correction")}
        passed = gate.within(abs(dE), GATE_ENERGY) and gate.within(dg, GATE_GRADIENT)
        # default conventions
        d, dcold, dwarm = timed(nums, pos, charge, uhf, "axtb", args.repeat)
        row = {
            "id": cid, "label": label, "natoms": len(nums), "charge": charge, "uhf": uhf,
            "programs": {k: prog[k] for k in ("energy", "gxtb", "gfn2_solvated", "gfn2_gas", "solvation_correction",
                                              "retry", "calls", "seconds")},
            "axtb_program_mode": {
                "energy": r["energy"], "dE": dE, "max_abs_dgrad": dg, "components_dE": comp,
                "composite": dict(r["composite"]), "parts": {k: dict(v) for k, v in r["parts"].items()},
                "seconds_new_calculator": cold, "seconds_next_geometry": warm,
            },
            "axtb_default": {
                "energy": d["energy"], "dE_vs_program_mode": d["energy"] - r["energy"],
                "max_abs_dgrad_vs_program_mode": float(np.abs(np.asarray(d["gradient"]) - np.asarray(r["gradient"])).max()),
                "solvation_correction": d["components"]["solvation_correction"],
                "parts": {k: dict(v) for k, v in d["parts"].items()},
                "seconds_new_calculator": dcold, "seconds_next_geometry": dwarm,
            },
            "gate": {"energy_tol": GATE_ENERGY, "gradient_tol": GATE_GRADIENT, "pass": passed},
        }
        m = (mepd or {}).get(cid)
        if m is not None:
            row["mepd"] = {"energy": m["energy"], "dE_vs_programs": m["energy"] - prog["energy"],
                           "dE_vs_axtb": m["energy"] - r["energy"],
                           "max_abs_dgrad_vs_axtb": float(np.abs(np.asarray(m["gradient"]) - np.asarray(r["gradient"])).max()),
                           "fallbacks": m["fallbacks"], "seconds": m["seconds"]}
        if do_hess and not args.no_hessian:
            row["hessian"] = [hessian_check(nums, pos, charge, uhf, c) for c in ("axtb", "gxtb")]
        rows.append(row)
        retry = r["parts"]["gfn2_gas"]["retry"] or "-"
        print(f"{cid:14s} {len(nums):4d} {prog['energy']:22.12f} {dE:9.1e} {dg:9.1e} "
              f"{(m['energy'] - prog['energy']) if m else float('nan'):9.1e} {retry[:12]:>12s} {prog['seconds']:7.3f} "
              f"{m['seconds'] if m else float('nan'):7.3f} {cold:7.4f} {warm:7.4f} {dcold:7.4f} {dwarm:7.4f}", flush=True)
        for hc in row.get("hessian", []):
            print(f"{'':14s} Hessian ({hc['conventions']}): max|dH| {hc['max_abs_dH']:.1e} Eh/bohr^2, "
                  f"max dfreq {hc['max_dfreq_cm']:.4f} cm-1 vs FD of the composite gradient, step {hc['step']:g} "
                  f"({' '.join(f'{k}: {v["max_abs_dH"]:.1e}' for k, v in hc['by_step'].items())}; {hc['seconds']:.3f} s)")
    shutil.rmtree(scratch, ignore_errors=True)
    cpu = next((l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines()
                if l.startswith("model name")), platform.processor())
    xtb_version = subprocess.run([XTB, "--version"], capture_output=True, text=True).stdout
    xtb_version = re.search(r"xtb version (\S+ \(\S+\))", xtb_version)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    result = {
        "description": "g-xTB 2.0.1 in ALPB water: axtb's in-process composite E_gxtb + [E_GFN2+ALPB - E_GFN2] vs mepd's "
                       "three xtb programs (tests/solvation_composite.py)",
        "date": date.today().isoformat(),
        "machine": {"cpu": cpu, "threads": 1},
        "versions": {"axtb": axtb.__version__, "axtb_commit": commit, "xtb": xtb_version.group(1) if xtb_version else None,
                     "mepd": bool(mepd)},  # whether mepd's code path ran (results hold no paths)
        "settings": {"method": METHOD, "solvent": SOLVENT, "model": "alpb", "state": "gsolv",
                     "program_mode": "conventions='gxtb' (g-xTB: 2.0.1 program; GFN2 halves: xtb 6.7.1, "
                                     "conventions 'xtb-6.7.1')",
                     "timing": f"wall seconds, best of {args.repeat}, one thread; programs: sum of the calls incl. "
                               "retries; axtb: one composite gradient with a new calculator, and the next geometry on a "
                               "reused one, in program mode (t gxtb, conventions='gxtb') and default conventions (t axtb)",
                     "load_average": os.getloadavg()[0]},
        "cases": rows,
        "summary": {"max_abs_dE": max(abs(r["axtb_program_mode"]["dE"]) for r in rows),
                    "max_abs_dH": max((h["max_abs_dH"] for r in rows for h in r.get("hessian", [])), default=None),
                    "max_abs_dgrad": max(r["axtb_program_mode"]["max_abs_dgrad"] for r in rows),
                    "max_abs_dE_mepd": max((abs(r["mepd"]["dE_vs_programs"]) for r in rows if "mepd" in r), default=None)},
    }
    gated = {k: v for k, v in result["summary"].items() if v is not None}  # mepd and the Hessians when they ran
    if gated.get("max_abs_dH") is not None:
        gated["max_dfreq_cm"] = max(h["max_dfreq_cm"] for r in rows for h in r.get("hessian", []))
    rc = gate.check(gated, LIMITS)
    result["summary"]["gate_pass"] = rc == 0
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(f"wrote {args.out}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
