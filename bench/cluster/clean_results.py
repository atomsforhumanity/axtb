#!/usr/bin/env python3
"""Turn the raw outputs of the cluster run (its output/<task>/ directories) into the report-facing
files in bench/results/cluster-<cpu>/: same data, but no host names, paths or IPs. The machine
is described by CPU model, core/thread counts and software versions only (machine.json).

Usage: clean_results.py --raw <dir with linalg/ [linalg-unpinned/] linalg-tp/ suite/ [probe/]> --out bench/results/cluster-9985wx
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--raw", type=Path, required=True)
ap.add_argument("--out", type=Path, required=True)
args = ap.parse_args()
raw, out = args.raw, args.out
(out / "linalg").mkdir(parents=True, exist_ok=True)

# home and cluster paths, private IPv4 addresses and the shared range (100.64/10) of overlay
# networks; site-specific host names come from $AXTB_PRIVATE_PATTERN (a regular expression,
# also read by scripts/check.sh). collect_timings.py reads this line.
FORBIDDEN = re.compile(r"/home/|/Users/|/srv/|/jobs/|/scratch/|/tmp/|192\.168\.\d|10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(1[6-9]|2\d|3[01])\.\d{1,3}\.\d|100\.(6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d")
if os.environ.get("AXTB_PRIVATE_PATTERN"):
    FORBIDDEN = re.compile(f"{FORBIDDEN.pattern}|{os.environ['AXTB_PRIVATE_PATTERN']}")


def lscpu(text: str, key: str) -> str:
    m = re.search(rf"^{re.escape(key)}:\s*(.+)$", text, re.M)
    return m.group(1).strip() if m else ""


node = (raw / "suite/node.txt").read_text()
lines = node.splitlines()
machine = {
    "cpu": lscpu(node, "Model name"),
    "sockets": int(lscpu(node, "Socket(s)")),
    "cores_per_socket": int(lscpu(node, "Core(s) per socket")),
    "threads_per_core": int(lscpu(node, "Thread(s) per core")),
    "l3": lscpu(node, "L3 cache"),
    "smt_numbering": "CPU i and i+64 are SMT siblings; pinned runs use CPUs 0..t-1 (one per physical core)",
    "pod": "one benchmark per node, 126 of 128 CPUs reserved (cgroup quota), node otherwise idle",
    "axtb": lines[2].split(" + ")[0].split()[1][:7] + " (git commit) + bench/ harness of this directory",
    "rust": [l for l in lines if l.startswith("rustc")],
    "aocl": [l for l in lines if l.split(" ")[0] in ("aocl-utils", "blis", "libflame")],
}
env = (raw / "linalg/env.txt").read_text() if (raw / "linalg/env.txt").exists() else ""
libs = sorted({l[2:].split(" | ")[0] + " | " + l.split(" | ")[1] for l in env.splitlines()
               if l.startswith("# ") and " | " in l})
machine["linalg_libraries"] = libs

res = json.loads((raw / "suite/results.json").read_text())
xtb = re.search(r"xtb version (\S+) \((\w+)\)", res["meta"]["xtb"])
res["meta"]["xtb"] = f"xtb {xtb.group(1)} ({xtb.group(2)}), g-xTB 2.0.1 build" if xtb else res["meta"]["xtb"]
res["meta"].update({"cores": machine["sockets"] * machine["cores_per_socket"],
                    "hw_threads": machine["sockets"] * machine["cores_per_socket"] * machine["threads_per_core"]})
machine["xtb"] = res["meta"]["xtb"]
machine["tblite"] = res["meta"]["tblite"]
(out / "results.json").write_text(json.dumps(res, indent=1))
(out / "run.log").write_text("".join(l for l in (raw / "suite/run.log").read_text().splitlines(True)
                                     if not l.startswith("wrote ")))
(out / "machine.json").write_text(json.dumps(machine, indent=1))

for src, dst in [("linalg/raw.tsv", "linalg/raw.tsv"), ("linalg/env.txt", "linalg/env.txt"),
                 ("linalg/summary_tables.txt", "linalg/summary_tables.txt"), ("linalg/progress.log", "linalg/progress.log"),
                 ("linalg-unpinned/raw_unpinned.tsv", "linalg/raw_unpinned.tsv"),
                 ("linalg-unpinned/env_unpinned.txt", "linalg/env_unpinned.txt"),
                 ("linalg-unpinned/summary_tables_unpinned.txt", "linalg/summary_tables_unpinned.txt"),
                 ("linalg-unpinned/progress.log", "linalg/progress_unpinned.log"),
                 ("linalg-tp/throughput64.tsv", "linalg/throughput64.tsv"),
                 ("linalg-tp/throughput128.tsv", "linalg/throughput128.tsv"),
                 ("linalg-tp/throughput_tables.txt", "linalg/throughput_tables.txt"),
                 ("probe/phases.json", "phases.json"), ("probe/cli_throughput.json", "cli_throughput.json")]:
    if (raw / src).exists():
        shutil.copy(raw / src, out / dst)

bad = [str(p) for p in out.rglob("*") if p.is_file() and FORBIDDEN.search(p.read_text(errors="ignore"))]
for f in ("node-suite.txt", "linalg/node-throughput.txt"):
    (out / f).unlink(missing_ok=True)
bad = [p for p in bad if Path(p).exists()]
if bad:
    raise SystemExit(f"host names/paths left in: {bad}")
print("clean:", sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file()))
