#!/usr/bin/env python3
"""Re-measure run_benchmarks.py's throughput rows (independent caffeine gradients on concurrent
workers) for several axtb CLI binaries, next to xtb with OMP_NUM_THREADS=1, using exactly
run_benchmarks.py's own launch code (xtb_run, axtb_cli).

Usage: cli_throughput.py --root <axtb repo with bench/> --specs 128x64,512x64 \
         --bin label=/path/to/axtb [--bin ...] --out rows.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--root", type=Path, required=True)
ap.add_argument("--specs", default="128x64,512x64,1024x128")
ap.add_argument("--bin", action="append", default=[], help="label=path of an axtb CLI binary")
ap.add_argument("--repeats", type=int, default=3, help="repeat each measurement, keep all walls")
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()

sys.path.insert(0, str(args.root / "bench"))
import run_benchmarks as rb  # noqa: E402

nums, pos, charge = rb.read_xyz(args.root / rb.MOLECULES[1][1])  # caffeine, as in run_benchmarks
rng = np.random.default_rng(0)
rows = []
for spec in args.specs.split(","):
    njobs, workers = (int(x) for x in spec.split("x"))
    geoms = [pos + 0.02 * rng.standard_normal(pos.shape) for _ in range(njobs)]
    cases = [("xtb, OMP_NUM_THREADS=1", lambda g: rb.xtb_run(nums, g, charge, 1, extra=("--grad",)), None)]
    for b in args.bin:
        label, path = b.split("=", 1)
        cases.append((f"axtb CLI processes (no env vars), {label}", lambda g: rb.axtb_cli(nums, g, charge, 1, 1.0), path))
    for rep in range(args.repeats):
        for label, fn, path in cases:
            if path:
                rb.AXTB = Path(path)  # axtb_cli() runs rb.AXTB
            t0 = time.perf_counter()
            with ThreadPoolExecutor(workers) as ex:
                list(ex.map(fn, geoms))
            wall = time.perf_counter() - t0
            rows.append({"label": label, "jobs": njobs, "workers": workers, "repeat": rep, "wall": wall})
            print(f"[throughput] {label} ({njobs} jobs, {workers} workers, rep {rep}): {wall:.2f} s", flush=True)
args.out.write_text(json.dumps(rows, indent=1))
