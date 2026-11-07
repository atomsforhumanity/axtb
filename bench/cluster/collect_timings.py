#!/usr/bin/env python3
"""Collect the final-timing tasks (bench/final_timings.py, one result.json per task in the raw
cluster output) into bench/results/<dir>/: one file per task, plus summary.json (every task's
summary, speed-up and Hessian-accuracy rows in one place) and machine.json. The raw node.txt
and logs (host names) stay in the raw directory; this checks that no host name, path or IP is
left in what it writes.

Usage: collect_timings.py --raw <dir with <run>/output/<task>/result.json ...> --out bench/results/final-9985wx
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

# the host-name/path/IP pattern of clean_results.py, plus the image's install prefix
_CLEAN = (Path(__file__).resolve().parent / "clean_results.py").read_text()
FORBIDDEN = re.compile(re.search(r'FORBIDDEN = re.compile\(r"(.+?)"\)\n', _CLEAN).group(1) + "|/opt/"
                       + (f"|{os.environ['AXTB_PRIVATE_PATTERN']}" if os.environ.get("AXTB_PRIVATE_PATTERN") else ""))


def scrub(text: str) -> str:
    return re.sub(r"(/[\w.+-]+)+/([\w.+-]+)", r"\2", text)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    summary = {"summary": [], "speedups": [], "hessian_accuracy": {}, "tasks": {}}
    machine = None
    for raw in a.raw:
        for res in sorted(raw.rglob("output/*/result.json")):
            d = json.loads(res.read_text())
            for r in d["runs"]:
                if "error" in r:
                    r["error"] = scrub(r["error"])
                # provenance without paths: binary names, hashes, library names
                if "binary" in r:
                    r["binary"] = Path(r["binary"]).name
                    r["cmd"] = [scrub(x) for x in r["cmd"]]
                    r["env"] = {k: scrub(v) for k, v in r["env"].items()}
            for c in d["meta"].get("harness_check", {}).get("columns", {}).values():
                c["binary"] = [[Path(b).name, h] for b, h in c["binary"]]
                c["env"] = [scrub(e) for e in c["env"]]
            for part, accs in d.get("hessian_accuracy", {}).items():
                for mol, acc in accs.items():
                    if "error" in acc:
                        acc["error"] = scrub(acc["error"])
                    summary["hessian_accuracy"].setdefault(part, {}).setdefault(f"{mol} | {d['meta']['threads']}t", acc)
            task = d["meta"]["task"]
            (a.out / f"{task}.json").write_text(json.dumps(d, indent=1))
            m = d["meta"]
            summary["tasks"][task] = {k: m.get(k) for k in ("threads", "axtb_commit", "date", "load1_start", "load1_end",
                                                             "pod_cpu_quota", "rounds_eg", "rounds_hess", "overhead_binaries")}
            summary["summary"] += [{"task": task, **r} for r in d["summary"]]
            summary["speedups"] += [{"task": task, **r} for r in d["speedups"]]
            mm = {k: m[k] for k in ("machine", "cpu", "cores", "threads_per_core", "hw_threads_visible", "versions")}
            if machine and {k: v for k, v in mm.items() if k != "versions"} != {k: v for k, v in machine.items() if k not in ("versions", "axtb_commits")}:
                raise SystemExit(f"{task}: different machine: {mm} vs {machine}")
            machine = machine or mm
            machine.setdefault("axtb_commits", {})[task] = m["axtb_commit"]
    machine["how"] = "bench/final_timings.py (docstring); bench/README.md, 'Final timings'"
    machine["notes"] = summary["notes"] = [
        "seconds: wall time of the whole process (start-up included); inner_seconds: the calculation inside it "
        "(tblite, axtb Python: without interpreter start-up and imports; axtb CLI: its own 'seconds'; axtb Hessians: "
        "the Hessian alone)",
        "max_rss_mb is not the program's peak memory: Linux carries the (v)forked driver's high-water RSS into the "
        "child at exec, so it is max(driver's peak so far, program's peak) and mostly the driver's",
        "other_busy_threads: average busy hardware threads of the node during the run (/proc/stat) minus the run's "
        "own CPU use; the node has 128 hardware threads on 64 cores"]
    (a.out / "machine.json").write_text(json.dumps(machine, indent=1))
    (a.out / "summary.json").write_text(json.dumps(summary, indent=1))
    bad = [str(p) for p in a.out.rglob("*.json") if FORBIDDEN.search(p.read_text())]
    if bad:
        raise SystemExit(f"host names/paths left in: {bad}")
    print("wrote", sorted(p.name for p in a.out.glob("*.json")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
