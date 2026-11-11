#!/usr/bin/env python3
"""Performance gate for refactoring: time two CLI binaries interleaved on the hot paths and flag
slowdowns.

    scripts/perf_gate.py BASELINE_BIN NEW_BIN [--rounds 5] [--only taxol,...] [--limit 1.02]

Each cell runs both binaries alternately (order swapped every round) at a quiet machine
(1-minute load below 1.5 before each pair); whole-process wall seconds, medians over rounds.
A cell fails when median(new) / median(baseline) exceeds --limit.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from golden import mol, S, TAXOL, CAFF  # noqa: E402

W200, C50, W333, CRAMBIN = S / "large/water200.xyz", S / "large/c50h102.xyz", S / "large/water333.xyz", S / "large/crambin_h.xyz"


def cells():
    tax, caf = mol(TAXOL), mol(CAFF)
    c = []
    add = lambda name, base, **kw: c.append((name, {**base, **kw}))
    for t in (1, 8):
        add(f"gfn2 taxol E+G {t}t", tax, method="gfn2", gradient=True, threads=t)
        add(f"gfn2 (H2O)200 E+G {t}t", mol(W200), method="gfn2", gradient=True, threads=t)
        add(f"gxtb2 taxol E+G {t}t", tax, method="gxtb-v2.0.1", gradient=True, threads=t)
    add("gfn2 C50H102 E+G 1t", mol(C50), method="gfn2", gradient=True, threads=1)
    add("gxtb2 (H2O)200 E+G 8t", mol(W200), method="gxtb-v2.0.1", gradient=True, threads=8)
    add("gxtb11 caffeine E+G 1t", caf, method="gxtb-v1.1", gradient=True, threads=1)
    add("gfnff (H2O)333 E+G 1t", mol(W333), method="gfnff", gradient=True, threads=1)
    add("gfnff crambin E+G 1t", mol(CRAMBIN), method="gfnff", gradient=True, threads=1)
    add("gfn2 taxol Hessian 1t", tax, method="gfn2", hessian=True, threads=1)
    add("gxtb2 caffeine Hessian 1t", caf, method="gxtb-v2.0.1", hessian=True, threads=1)
    add("gxtb2 taxol Hessian 8t", tax, method="gxtb-v2.0.1", hessian=True, threads=8)
    add("gfnff caffeine Hessian 1t", caf, method="gfnff", hessian=True, threads=1)
    return c


def load1():
    return float(Path("/proc/loadavg").read_text().split()[0])


def timed(binary, req):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("AXTB_LAPACK", "OMP_", "MKL_", "RAYON"))}
    t = time.perf_counter()
    p = subprocess.run([binary, "--json"], input=json.dumps(req), capture_output=True, text=True, env=env)
    w = time.perf_counter() - t
    if p.returncode != 0:
        sys.exit(f"{binary} failed: {p.stdout[-400:]} {p.stderr[-400:]}")
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline")
    ap.add_argument("new")
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--only", default="")
    ap.add_argument("--limit", type=float, default=1.02)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    cs = cells()
    if a.only:
        cs = [c for c in cs if any(k in c[0] for k in a.only.split(","))]
    rows, failed = [], 0
    for name, req in cs:
        tb, tn = [], []
        for r in range(a.rounds):
            while load1() > 1.5:
                time.sleep(3)
            pair = [(a.baseline, tb), (a.new, tn)]
            for binary, out in (pair if r % 2 == 0 else pair[::-1]):
                out.append(timed(binary, req))
        mb, mn = statistics.median(tb), statistics.median(tn)
        ratio = mn / mb
        flag = "SLOWER" if ratio > a.limit else ""
        failed += bool(flag)
        rows.append({"cell": name, "baseline_s": mb, "new_s": mn, "ratio": ratio, "baseline_runs": tb, "new_runs": tn})
        print(f"{name:28} baseline {mb:8.3f} s  new {mn:8.3f} s  new/baseline {ratio:5.3f} {flag}", flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))
    print(f"{len(rows) - failed} of {len(rows)} cells within {a.limit:.2f}x")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
