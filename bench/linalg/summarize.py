#!/usr/bin/env python3
"""Aggregate results/raw.tsv: median over rounds of the per-round medians (ms).
Usage: python3 summarize.py results/raw.tsv[,more.tsv,...] [ops] > results/summary_tables.txt
Throughput files (first column "mode", from throughput.sh) give one table per mode and op:
the median over the concurrent processes of each process's median."""
import csv, sys, statistics as st
from collections import defaultdict

paths = sys.argv[1].split(",")
d = defaultdict(list)
cpu = defaultdict(list)
tp = defaultdict(list)
for path in paths:
    for r in csv.DictReader(open(path), delimiter="\t"):
        if "mode" in r:
            tp[(r["mode"], r["backend"], int(r["n"]), r["op"])].append(float(r["median_ms"]))
            continue
        k = (r["backend"], int(r["threads"]), int(r["n"]), r["op"])
        d[k].append(float(r["median_ms"]))
        cpu[k].append(float(r["cpu_per_wall"]))
order = ["faer22", "faer24", "faer24v4", "openblas", "mkl", "aocl"]
seen = {k[0] for k in d} | {k[1] for k in tp}
backends = [b for b in order if b in seen] + sorted(seen - set(order))
ns = sorted({k[2] for k in d})
ops = sys.argv[2].split(",") if len(sys.argv) > 2 else ["evd", "gen", "gen2t", "evr40", "mr40", "genr40", "genmm", "gemm"]


def fmt(m):
    return f"{m:.3g}" if m < 10 else f"{m:.0f}"


spread_warn = []
for op in ops:
    for t in sorted({k[1] for k in d}):
        rows = [(b, [d.get((b, t, n, op)) for n in ns]) for b in backends if any((b, t, n, op) in d for n in ns)]
        if not rows:
            continue
        print(f"\n### {op}, {t} thread(s) [ms; median of rounds]  (cpu/wall in brackets at n=max)")
        print("| backend | " + " | ".join(f"n={n}" for n in ns) + " |")
        print("|---|" + "---|" * len(ns))
        for b, vals in rows:
            cells = []
            for n, v in zip(ns, vals):
                if v is None:
                    cells.append("–")
                    continue
                m = st.median(v)
                if len(v) > 1 and max(v) / min(v) > 1.15:
                    spread_warn.append(f"{b} t={t} n={n} {op}: {[round(x,1) for x in v]}")
                cells.append(fmt(m))
            c = cpu.get((b, t, ns[-1], op))
            print(f"| {b} | " + " | ".join(cells) + f" |" + (f" [{st.median(c):.1f}]" if c and t > 1 else ""))
for mode in sorted({k[0] for k in tp}):
    tns = sorted({k[2] for k in tp if k[0] == mode})
    for op in ops:
        rows = [b for b in backends if any((mode, b, n, op) in tp for n in tns)]
        if not rows:
            continue
        print(f"\n### throughput {mode}: {op}, 1 thread per process [ms; median over processes]  (vs faer22 in brackets)")
        print("| backend | " + " | ".join(f"n={n}" for n in tns) + " |")
        print("|---|" + "---|" * len(tns))
        for b in rows:
            cells = []
            for n in tns:
                v = tp.get((mode, b, n, op))
                if not v:
                    cells.append("–")
                    continue
                m = st.median(v)
                ref = tp.get((mode, "faer22", n, op))
                cells.append(fmt(m) + (f" ({m / st.median(ref):.2f}x)" if ref and b != "faer22" else ""))
            print(f"| {b} | " + " | ".join(cells) + " |")
if spread_warn:
    print("\nround-to-round spread > 15%:")
    for s in spread_warn:
        print("  " + s)
