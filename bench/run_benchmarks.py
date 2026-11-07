#!/usr/bin/env python3
"""Accuracy and performance comparison of axtb with xtb and tblite 0.8.0 (GFN2-xTB).

Usage: python bench/run_benchmarks.py [--quick] [--xtb PATH] [--out bench/results/results.json]

The xtb program is `--xtb`, default xtb 6.8.0 (`$AXTB_XTB680`, see tests/programs.py); axtb is
compared in the conventions of its version. The recorded results.json used xtb 6.7.1, the g-xTB
2.0.1 build (`--xtb "$AXTB_GXTB2"`).

Every program gets identical bohr coordinates (xtb via a Turbomole `coord` file). Timings:
- xtb: end-to-end wall time of the `xtb` process (what a caller pays: start-up, setup,
  SCF, gradient, file output), OMP_NUM_THREADS = MKL_NUM_THREADS = threads;
- tblite: Calculator construction + single point inside a fresh Python process with the
  thread environment set before start (tblite reads OMP_NUM_THREADS);
- axtb: `axtb.singlepoint` in-process (threads=…), and the `axtb` CLI end-to-end.
Each timing is the minimum over `repeats` runs. Run on an otherwise idle machine.
"""

from __future__ import annotations

import argparse
import functools
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from validate_tblite import AXTB, gfn2_param, read_xyz  # noqa: E402
from validate_xtb import write_coord  # noqa: E402
import programs  # noqa: E402

XTB = programs.get(programs.XTB680)  # the xtb binary; --xtb

PY = sys.executable
SCRATCH = os.environ.get("AXTB_SCRATCH")

MOLECULES = [
    ("water", "tests/structures/water.xyz"),
    ("caffeine", "tests/structures/caffeine.xyz"),
    ("anthracene dimer", "tests/structures/anthracene2.xyz"),
    ("taxol", "tests/structures/taxol.xyz"),
    ("C50H102", "tests/structures/large/c50h102.xyz"),
    ("(H2O)100", "tests/structures/large/water100.xyz"),
    ("(H2O)200", "tests/structures/large/water200.xyz"),
    ("(H2O)333", "tests/structures/large/water333.xyz"),
]
ACCURACY_EXTRA = [
    ("ferrocene", "tests/structures/ferrocene.xyz"),
    ("[Cu(NH3)4]2+", "tests/structures/cu-nh3-4.xyz"),
    ("[ZnCl4]2-", "tests/structures/zncl4.xyz"),
    ("CF3Br-benzene", "tests/structures/cf3br-benzene.xyz"),
    ("iodobenzene-acetone", "tests/structures/beni-acetone.xyz"),
    ("MB16-43 #33 (Al,B,S,F)", "tests/structures/mb33.xyz"),
]


def thread_env(threads: int | None) -> dict:
    env = dict(os.environ)
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if threads is None:
            env.pop(k, None)
        else:
            env[k] = str(threads)
    return env


def uhf_for(nums, charge):
    nel = int(nums.sum() - charge)
    return 1 if nel % 2 else None


# ---------------------------------------------------------------------------- xtb
def xtb_run(nums, pos, charge, threads, extra=(), files=("gradient",)):
    with tempfile.TemporaryDirectory(dir=SCRATCH) as d:
        d = Path(d)
        write_coord(d / "coord", nums, pos)
        cmd = [XTB, "coord", "--gfn", "2", "--chrg", str(int(charge)), *extra]
        uhf = uhf_for(nums, charge)
        if uhf:
            cmd += ["--uhf", str(uhf)]
        t = time.perf_counter()
        out = subprocess.run(cmd, cwd=d, capture_output=True, text=True, env=thread_env(threads))
        wall = time.perf_counter() - t
        if "normal termination" not in out.stdout + out.stderr:
            raise RuntimeError("xtb failed: " + out.stderr[-300:])
        res = {"wall": wall, "energy": float(re.findall(r"TOTAL ENERGY\s+(-?\d+\.\d+)", out.stdout)[-1])}
        if "gradient" in files and (d / "gradient").exists():
            g = (d / "gradient").read_text().replace("D", "E").splitlines()
            n = len(nums)
            res["gradient"] = np.array([[float(x) for x in l.split()] for l in g[-1 - n:-1]])
        if "hessian" in files:
            res["hessian"] = np.array([float(x) for x in (d / "hessian").read_text().split()[1:]])
        return res


# ---------------------------------------------------------------------------- tblite
TBLITE_SNIPPET = r"""
import json, sys, time, numpy as np
from tblite.interface import Calculator
d = json.load(sys.stdin)
t = time.perf_counter()
c = Calculator(d["param"], np.array(d["numbers"]), np.array(d["positions"]), charge=d["charge"], uhf=d["uhf"])
c.set("verbosity", 0); c.set("accuracy", d["acc"])
r = c.singlepoint()
dt = time.perf_counter() - t
print(json.dumps({"wall": dt, "energy": float(r.get("energy")), "gradient": np.asarray(r.get("gradient")).tolist()}))
"""


def tblite_run(nums, pos, charge, threads, acc, param):
    req = {"param": param, "numbers": nums.tolist(), "positions": pos.tolist(), "charge": float(charge),
           "uhf": uhf_for(nums, charge), "acc": acc}
    out = subprocess.run([PY, "-c", TBLITE_SNIPPET], input=json.dumps(req), capture_output=True, text=True,
                         env=thread_env(threads))
    if out.returncode != 0:
        raise RuntimeError("tblite failed: " + out.stderr[-300:])
    r = json.loads(out.stdout)
    r["gradient"] = np.array(r["gradient"])
    return r


def xtb_path_conventions() -> str:
    """axtb's conventions for `XTB`: `xtb-6.7.1` for xtb 6.7.x (also the g-xTB 2.0.1 build),
    `xtb` for 6.8.0."""
    return _conventions_of(XTB)


@functools.cache
def _conventions_of(xtb) -> str:
    # runs `xtb --version` once per binary: never inside a timed region
    try:
        v = subprocess.run([xtb, "--version"], capture_output=True, text=True).stdout
    except (OSError, TypeError):
        return "xtb"
    m = re.search(r"xtb version (\S+)", v)
    return "xtb-6.7.1" if m and m.group(1).startswith("6.7.") else "xtb"


# ---------------------------------------------------------------------------- axtb
def axtb_run(nums, pos, charge, threads, acc, conventions=None, **kw):
    import axtb
    conventions = conventions or xtb_path_conventions()  # resolved before the timer (it may run `xtb --version`)
    t = time.perf_counter()
    r = axtb.singlepoint(nums, pos, charge=float(charge), uhf=uhf_for(nums, charge), gradient=True,
                         threads=threads, accuracy=acc, conventions=conventions, **kw)
    r = dict(r)
    r["wall"] = time.perf_counter() - t
    return r


def axtb_cli(nums, pos, charge, threads, acc):
    req = {"numbers": nums.tolist(), "positions": pos.tolist(), "charge": float(charge), "uhf": uhf_for(nums, charge),
           "accuracy": acc, "gradient": True, "threads": threads}
    t = time.perf_counter()
    out = subprocess.run([str(AXTB), "--json"], input=json.dumps(req), capture_output=True, text=True,
                         env=thread_env(None))
    wall = time.perf_counter() - t
    if out.returncode != 0:
        raise RuntimeError(out.stderr + out.stdout)
    r = json.loads(out.stdout)
    return {"wall": wall, "energy": r["energy"]}


def best(f, repeats):
    runs = [f() for _ in range(repeats)]
    return min(runs, key=lambda r: r["wall"])


def git_describe() -> str:
    try:
        return subprocess.run([XTB, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    except Exception:  # noqa: BLE001
        return "unknown"


def main() -> int:
    global XTB
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="small molecules, 1 and 8 threads only")
    ap.add_argument("--xtb", default=XTB, help="the xtb binary (default: xtb 6.8.0, $AXTB_XTB680)")
    ap.add_argument("--out", type=Path, default=ROOT / "bench/results/results.json")
    ap.add_argument("--threads", default="1,2,4,8")
    ap.add_argument("--hess-threads", default="1,8", help="thread counts for the Hessian timings")
    ap.add_argument("--throughput", default="64x8",
                    help="throughput runs as JOBSxWORKERS[,JOBSxWORKERS...], e.g. 512x64,1024x128 on a 64-core node")
    ap.add_argument("--retime-axtb", action="store_true",
                    help="re-measure only axtb's timing rows in an existing results file (others unchanged)")
    args = ap.parse_args()
    XTB = args.xtb
    if not XTB:
        programs.skip("pass --xtb or " + programs.reason(programs.XTB680))
    if args.retime_axtb:
        res = json.loads(args.out.read_text())
        paths = dict(MOLECULES)
        for row in res["timing"]:
            nums, pos, charge = read_xyz(ROOT / paths[row["name"]])
            reps = 3 if len(nums) < 200 else 1
            t = row["threads"]
            row["axtb"] = best(lambda: axtb_run(nums, pos, charge, t, 1.0), reps)["wall"]
            row["axtb_cli"] = best(lambda: axtb_cli(nums, pos, charge, t, 1.0), reps)["wall"]
            print(f"[retime] {row['name']} threads={t}: axtb {row['axtb']:.3f} cli {row['axtb_cli']:.3f}", flush=True)
        import axtb
        res["meta"]["axtb_retimed"] = time.strftime("%Y-%m-%d %H:%M")
        args.out.write_text(json.dumps(res, indent=1))
        return 0
    threads_list = [1, 8] if args.quick else [int(x) for x in args.threads.split(",")]
    mols = MOLECULES[:3] if args.quick else MOLECULES
    acc = 1.0  # the programs' default accuracy
    import axtb
    import tblite
    param = gfn2_param(False)
    results = {
        "meta": {
            "date": time.strftime("%Y-%m-%d %H:%M"),
            "machine": platform.processor() or platform.machine(),
            "cpu": subprocess.run(["sh", "-c", "lscpu | grep 'Model name' | cut -d: -f2"], capture_output=True, text=True).stdout.strip(),
            "xtb": git_describe(),
            "tblite": getattr(tblite, "__version__", "0.8.0"),
            "axtb": axtb.__version__,
            "accuracy": acc,
        },
        "accuracy": [],
        "solvation": [],
        "timing": [],
        "hessian": [],
        "throughput": [],
    }

    # ---- accuracy: values from every program (single thread)
    for name, rel in mols + ([] if args.quick else ACCURACY_EXTRA):
        nums, pos, charge = read_xyz(ROOT / rel)
        x = xtb_run(nums, pos, charge, 1, extra=("--grad", "--acc", "1e-4"))
        tb = tblite_run(nums, pos, charge, 1, 1e-4, param)
        ax = axtb_run(nums, pos, charge, 1, 1e-4, "xtb")
        at = axtb_run(nums, pos, charge, 1, 1e-4, "tblite")
        row = {
            "name": name, "nat": int(len(nums)), "charge": int(charge),
            "E_xtb": x["energy"], "E_tblite": tb["energy"], "E_axtb_xtbref": ax["energy"], "E_axtb_tbliteref": at["energy"],
            "gnorm_xtb": float(np.linalg.norm(x["gradient"])), "gnorm_axtb": float(np.linalg.norm(ax["gradient"])),
            "dG_vs_xtb": float(np.abs(ax["gradient"] - x["gradient"]).max()),
            "dG_vs_tblite": float(np.abs(at["gradient"] - tb["gradient"]).max()),
        }
        results["accuracy"].append(row)
        print(f"[accuracy] {name}: dE(xtb) {row['E_axtb_xtbref'] - row['E_xtb']:.1e}  dE(tblite) {row['E_axtb_tbliteref'] - row['E_tblite']:.1e}", flush=True)

    # ---- implicit solvation (GFN2/ALPB and GBSA) vs xtb, energies and timings, 1 thread
    for name, rel in ([MOLECULES[1]] if args.quick else [MOLECULES[1], MOLECULES[3], ACCURACY_EXTRA[2]]):
        nums, pos, charge = read_xyz(ROOT / rel)
        for model, solvent in ([("alpb", "water")] if args.quick else [("alpb", "water"), ("alpb", "thf"), ("gbsa", "water")]):
            x = xtb_run(nums, pos, charge, 1, extra=("--grad", "--acc", "1e-4", f"--{model}", solvent))
            a = axtb_run(nums, pos, charge, 1, 1e-4, None, **{model: solvent})
            results["solvation"].append({
                "name": name, "nat": int(len(nums)), "model": model, "solvent": solvent,
                "E_xtb": x["energy"], "E_axtb": a["energy"], "dG": float(np.abs(a["gradient"] - x["gradient"]).max()),
                "t_xtb": x["wall"], "t_axtb": a["wall"],
            })
            print(f"[solvation] {name} {model} {solvent}: dE {a['energy'] - x['energy']:.1e}", flush=True)

    # ---- timing: energy + gradient at the programs' default accuracy
    for name, rel in mols:
        nums, pos, charge = read_xyz(ROOT / rel)
        reps = 3 if len(nums) < 200 else 1
        for t in threads_list:
            row = {"name": name, "nat": int(len(nums)), "threads": t}
            row["xtb"] = best(lambda: xtb_run(nums, pos, charge, t, extra=("--grad",)), reps)["wall"]
            row["tblite"] = best(lambda: tblite_run(nums, pos, charge, t, 1.0, param), reps)["wall"]
            row["axtb"] = best(lambda: axtb_run(nums, pos, charge, t, 1.0), reps)["wall"]
            row["axtb_cli"] = best(lambda: axtb_cli(nums, pos, charge, t, 1.0), reps)["wall"]
            results["timing"].append(row)
            print(f"[timing] {name} threads={t}: xtb {row['xtb']:.3f} tblite {row['tblite']:.3f} axtb {row['axtb']:.3f} cli {row['axtb_cli']:.3f}", flush=True)

    # ---- Hessian (semi-numerical, 0.005 bohr) at acc 0.01
    for name, rel in ([MOLECULES[1]] if args.quick else [MOLECULES[1], MOLECULES[3]]):
        nums, pos, charge = read_xyz(ROOT / rel)
        for t in ([int(x) for x in args.hess_threads.split(",")] if not args.quick else [1]):
            x = xtb_run(nums, pos, charge, t, extra=("--hess", "--acc", "0.01"), files=("hessian",))
            calc = axtb.Calculator("gfn2", nums, pos, charge=float(charge), threads=t)
            t0 = time.perf_counter()
            h = calc.hessian(pos, step=0.005, accuracy=0.01, method="numerical")
            wall = time.perf_counter() - t0
            results["hessian"].append({"name": name, "nat": int(len(nums)), "threads": t, "xtb": x["wall"], "axtb": wall})
            print(f"[hessian] {name} threads={t}: xtb {x['wall']:.2f} axtb {wall:.2f}", flush=True)

    # ---- throughput: independent caffeine gradients on concurrent workers (default 64 jobs, 8 workers)
    nums, pos, charge = read_xyz(ROOT / MOLECULES[1][1])
    rng = np.random.default_rng(0)
    for spec in args.throughput.split(","):
        njobs, workers = (int(x) for x in spec.split("x"))
        if args.quick:
            njobs = min(njobs, 2 * workers)
        geoms = [pos + 0.02 * rng.standard_normal(pos.shape) for _ in range(njobs)]
        for label, threads in [("xtb, OMP_NUM_THREADS unset", None), ("xtb, OMP_NUM_THREADS=1", 1)]:
            t0 = time.perf_counter()
            with ThreadPoolExecutor(workers) as ex:
                list(ex.map(lambda g: xtb_run(nums, g, charge, threads, extra=("--grad",)), geoms))
            wall = time.perf_counter() - t0
            results["throughput"].append({"label": label, "jobs": len(geoms), "workers": workers, "wall": wall})
            print(f"[throughput] {label} ({len(geoms)} jobs, {workers} workers): {wall:.2f} s", flush=True)
        for label, fn in [("axtb CLI processes (no env vars)", lambda g: axtb_cli(nums, g, charge, 1, 1.0)),
                          (f"axtb in-process, {workers} Python threads", lambda g: axtb_run(nums, g, charge, 1, 1.0))]:
            t0 = time.perf_counter()
            with ThreadPoolExecutor(workers) as ex:
                list(ex.map(fn, geoms))
            wall = time.perf_counter() - t0
            results["throughput"].append({"label": label, "jobs": len(geoms), "workers": workers, "wall": wall})
            print(f"[throughput] {label} ({len(geoms)} jobs, {workers} workers): {wall:.2f} s", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=1))
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
