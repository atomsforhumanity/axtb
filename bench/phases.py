#!/usr/bin/env python3
"""Where axtb's time goes: per-phase wall times of axtb.singlepoint(..., gradient=True) for the
run_benchmarks.py timing molecules, as reported by axtb itself (r["timings"], r["iterations"]).
Each (molecule, threads) is the run with the smallest total of 3 (same settings as the suite's
timing rows: default accuracy, xtb conventions).

Usage: python bench/phases.py --threads 1,8 [--big-threads 64] --out bench/results/phases.json
--big-threads adds thread counts for the three largest molecules only.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_benchmarks as rb  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--threads", default="1,8")
ap.add_argument("--big-threads", default="")
ap.add_argument("--repeats", type=int, default=3)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()

import axtb  # noqa: E402

mols = [(name, rb.read_xyz(rb.ROOT / rel)) for name, rel in rb.MOLECULES]
largest = {name for name, (nums, _, _) in sorted(mols, key=lambda m: len(m[1][0]))[-3:]}
rows = []
for name, (nums, pos, charge) in mols:
    tlist = [int(t) for t in args.threads.split(",")]
    if name in largest and args.big_threads:
        tlist += [int(t) for t in args.big_threads.split(",")]
    for t in tlist:
        best = None
        for _ in range(args.repeats):
            t0 = time.perf_counter()
            r = axtb.singlepoint(nums, pos, charge=float(charge), uhf=rb.uhf_for(nums, charge), gradient=True,
                                 threads=t, accuracy=1.0, conventions="xtb")
            wall = time.perf_counter() - t0
            if best is None or wall < best[0]:
                best = (wall, r)
        wall, r = best
        row = {"molecule": name, "atoms": int(len(nums)), "threads": t, "iterations": int(r["iterations"])}
        row.update({k: float(v) for k, v in r["timings"].items()})
        row["total"] = wall
        rows.append(row)
        print(json.dumps(row), flush=True)
print("cpu:", rb.subprocess.run(["sh", "-c", "lscpu | grep 'Model name' | cut -d: -f2"], capture_output=True,
                                text=True).stdout.strip(), "| axtb", axtb.__version__, "|", time.strftime("%Y-%m-%d %H:%M"))
args.out.write_text(json.dumps(rows, indent=1))
