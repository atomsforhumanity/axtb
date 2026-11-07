#!/usr/bin/env python3
"""Build the report's datasets (bench/results/report/*.json, one JSON array of row objects
each) from the raw results:

  bench/results/final-laptop/, final-9985wx/       final timings (final_timings.py) on the laptop and
                                                    the 64-core node: energy + gradient, Hessians, the
                                                    performance requirement (FINAL, REQ_PARTS, TIGHT)
  bench/results/results.json                        laptop suite (run_benchmarks.py)
  bench/results/cluster-9985wx/results.json         64-core node suite (earlier whole-node runs)
  bench/linalg/results/{raw,aocl_raw,throughput}.tsv            laptop linear algebra
  bench/results/cluster-9985wx/linalg/{raw,throughput64,throughput128}.tsv   node
  tests/reference_matrix/results.json               991-case reference matrix

The datasets hold no host names or paths, only CPU models and versions, so the same files
can be published as they are. bench/check_report.py checks the page (index.html) against them.
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "bench/results/report"
NODE = ROOT / "bench/results/cluster-9985wx"

MACHINES = {
    "laptop": {"label": "Laptop", "cpu": "Intel Core i9-11900H (Tiger Lake), 8 cores / 16 threads",
               "suite": ROOT / "bench/results/results.json"},
    "node": {"label": "64-core node", "cpu": "AMD Ryzen Threadripper PRO 9985WX (Zen 5), 64 cores / 128 threads; jobs limited to 126 CPUs",
             "suite": NODE / "results.json"},
}
LIBS = {"faer22": "faer 0.22", "faer24": "faer 0.24", "faer24v4": "faer 0.24 (AVX-512)", "openblas": "OpenBLAS",
        "mkl": "Intel MKL", "aocl": "AMD AOCL"}


def write(name: str, rows: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(rows, indent=1))
    print(f"{name}: {len(rows)} rows")


def tsv(path: Path):
    if not path.exists():
        return []
    with open(path) as fh:
        return list(csv.DictReader((ln for ln in fh if not ln.startswith("#")), delimiter="\t"))


def suites() -> dict:
    return {m: json.loads(v["suite"].read_text()) for m, v in MACHINES.items() if v["suite"].exists()}


def xtb_label(text: str) -> str:
    """'xtb version 6.8.0 (4b8c987...)' -> 'xtb 6.8.0'; the g-xTB build of 6.7.1 named as such."""
    import re
    v = re.search(r"(\d+\.\d+\.\d+)", text or "")
    if not v:
        return text or None
    return f"xtb {v[1]}" + (" (the g-xTB 2.0.1 build)" if "g-xTB" in text else "")


def suite_build(m: str, meta: dict) -> str:
    """The axtb a suite ran: the commit its machine.json records, else the version in its meta."""
    mp = Path(MACHINES[m]["suite"]).parent / "machine.json"
    if mp.exists():
        a = json.loads(mp.read_text()).get("axtb", "")
        tok = a.split()[0] if a else ""
        if tok and all(ch in "0123456789abcdef" for ch in tok):
            return f"commit {tok}"
    return f"version {meta.get('axtb')}"


def final_versions() -> dict:
    """The axtb commits of the final timings and the run-time libraries' versions (thread_determinism.json)."""
    import re
    out = {"commits": set(), "libs": {}}
    for m, cfg in FINAL.items():
        sp = cfg["dir"] / "summary.json"
        if sp.exists():
            out["commits"] |= {t["axtb_commit"][:7] for t in json.loads(sp.read_text()).get("tasks", {}).values() if t.get("axtb_commit")}
        tp = cfg["dir"] / "thread_determinism.json"
        if tp.exists():
            for r in json.loads(tp.read_text()).get("rows", []):
                e = r.get("eigensolver") or ""
                v = re.search(r"Math Kernel Library Version ([\d.]+)", e) or re.search(r"AOCL-LAPACK ([\d.]+)", e)
                if v and r.get("solver"):
                    out["libs"].setdefault(r["solver"], {})[m] = v[1]
    return out


def machine_rows(res: dict) -> None:
    rows = []
    for m, v in MACHINES.items():
        if m not in res:
            continue
        meta = res[m]["meta"]
        rows.append({"machine": m, "label": v["label"], "cpu": v["cpu"], "date": meta["date"][:10],
                     "axtb": suite_build(m, meta), "xtb": xtb_label(meta.get("xtb", ""))})
    write("machines", rows)
    fv = final_versions()
    mkl = fv["libs"].get("MKL", {})
    mkl_text = ", ".join(f"{ver} ({FINAL[m]['label'].lower()} timings)" for m, ver in mkl.items())
    write("versions", [
        {"item": "xtb 6.8.0", "value": "xtb version 6.8.0 (4b8c987), official Linux release: the GFN2-xTB reference for xtb conventions",
         "reference": "Bannwarth et al., WIREs Comput. Mol. Sci. 11, e1493 (2021), doi:10.1002/wcms.1493", "license": "LGPL-3.0"},
        {"item": "xtb 6.7.1 / g-xTB 2.0.1 program", "value": "xtb version 6.7.1 (26dd68d), the g-xTB 2.0.1 build: g-xTB with --gxtb, GFN2-xTB 6.7.1 and GFN-FF without it",
         "reference": "github.com/grimme-lab/g-xtb", "license": "GPL-3.0"},
        {"item": "g-xtb 1.1.0", "value": "the g-xTB v1.1 program", "reference": "github.com/grimme-lab/g-xtb", "license": "GPL-3.0"},
        {"item": "tblite", "value": f"tblite {res['laptop']['meta']['tblite']} (PyPI wheel), dftd4 4.3.1",
         "reference": "github.com/tblite/tblite", "license": "LGPL-3.0"},
        {"item": "axtb", "value": f"axtb {res['laptop']['meta']['axtb']}" + (f" (commit {', '.join(sorted(fv['commits']))} in the final timings)" if fv["commits"] else "") + ", Rust, faer 0.22",
         "reference": "github.com/atomsforhumanity/axtb", "license": "LGPL-3.0-or-later; GPL-3.0-only with g-xTB"},
        {"item": "GFN2-xTB", "value": "self-consistent D4 dispersion; ALPB/GBSA implicit solvation",
         "reference": "Bannwarth, Ehlert, Grimme, JCTC 15, 1652 (2019), doi:10.1021/acs.jctc.8b01176; Ehlert et al., JCTC 17, 4250 (2021), doi:10.1021/acs.jctc.1c00471",
         "license": ""},
        {"item": "g-xTB", "value": "v1.1 and 2.0.1 (H–Lr), q-vSZP basis, EEQ-BC charges, revD4",
         "reference": "Froitzheim, Müller, Hansen, Grimme, ChemRxiv (2025), chemrxiv.org/engage/chemrxiv/article-details/685434533ba0887c335fc974 (v1); 2.0.1: github.com/grimme-lab/g-xtb",
         "license": ""},
        {"item": "GFN-FF", "value": "generic force field with automatic topology",
         "reference": "Spicher, Grimme, Angew. Chem. Int. Ed. 59, 15665 (2020), doi:10.1002/anie.202004239", "license": ""},
        {"item": "Linear algebra", "value": "faer 0.22 (axtb), faer 0.24; OpenBLAS 0.3.32 (laptop) / 0.3.34 (node); Intel MKL 2026.2 (eigensolver benchmarks)"
                  + (f", {mkl_text}" if mkl_text else "") + "; AMD AOCL 5.3 built from source",
         "reference": "faer.veganb.tw; openblas.net; intel.com/oneMKL; github.com/amd (aocl-blis, aocl-libflame)", "license": "MIT; BSD-3; Intel Simplified Software License; BSD-3"},
        {"item": "Constants", "value": "NIST CODATA 2022 (axtb default conventions)",
         "reference": "Mohr et al., CODATA 2022, physics.nist.gov/constants", "license": "public domain"},
    ])


def accuracy(res: dict) -> None:
    r = res["laptop"]
    write("accuracy", [{
        "molecule": x["name"], "atoms": x["nat"], "charge": x["charge"],
        "E_xtb": x["E_xtb"], "E_tblite": x["E_tblite"],
        "E_axtb_xtbref": x["E_axtb_xtbref"], "E_axtb_tbliteref": x["E_axtb_tbliteref"],
        "dG_vs_xtb": x["dG_vs_xtb"], "dG_vs_tblite": x["dG_vs_tblite"],
    } for x in r["accuracy"]])
    write("solvation", [{
        "case": f"{x['name']} · {x['model'].upper()} {x['solvent']}", "molecule": x["name"], "atoms": x["nat"],
        "model": x["model"].upper(), "solvent": x["solvent"], "E_xtb": x["E_xtb"], "E_axtb": x["E_axtb"],
        "dG": x["dG"], "t_xtb": x["t_xtb"], "t_axtb": x["t_axtb"],
    } for x in r["solvation"]])


def xtb_tblite(res: dict) -> None:
    """xtb's GFN2-xTB energy minus tblite's per benchmark molecule, for each xtb version a suite ran
    (accuracy part of run_benchmarks.py, tight SCF): d_671 from a 6.7.1 suite, d_680 from a 6.8.0 one."""
    by = {}
    for m, r in res.items():
        ver = xtb_label(r["meta"].get("xtb", "")) or ""
        key = "d_671" if "6.7.1" in ver else "d_680" if "6.8.0" in ver else None
        if key is None:
            continue
        for x in r["accuracy"]:
            row = by.setdefault(x["name"], {"molecule": x["name"], "atoms": x["nat"], "d_671": None, "d_680": None})
            row[key] = x["E_xtb"] - x["E_tblite"]
    write("xtb_tblite", sorted(by.values(), key=lambda r: r["atoms"]))


def speed(res: dict) -> None:
    timing, hess, tp = [], [], []
    for m, r in res.items():
        timing += [{"id": f"{m}|{x['name']}|{x['threads']}", "machine": m, "molecule": x["name"], "atoms": x["nat"], "threads": x["threads"],
                    "xtb": x["xtb"], "tblite": x["tblite"], "axtb": x["axtb"], "axtb_cli": x["axtb_cli"]} for x in r["timing"]]
        hess += [{"id": f"{m}|{x['name']}|{x['threads']}", "machine": m, "case": f"{x['name']} · {x['threads']} thread{'s' if x['threads'] > 1 else ''}",
                  "molecule": x["name"], "atoms": x["nat"], "threads": x["threads"], "xtb": x["xtb"], "axtb": x["axtb"]}
                 for x in r["hessian"]]
        tp += [{"id": f"{m}|{x['jobs']}x{x['workers']}|{x['label']}", "machine": m, "setup": x["label"].split(", ")[0] if x["label"].startswith("axtb in-process") else x["label"],
                "jobs": x["jobs"], "workers": x["workers"], "batch": f"{x['jobs']} jobs on {x['workers']} workers",
                "wall": x["wall"]} for x in r["throughput"]]
    write("timing", timing)
    write("hessian", hess)
    write("throughput", tp)


def linalg() -> None:
    # generalized and standard symmetric eigenproblems; median over rounds of each round's median
    sources = [("laptop", "main", ROOT / "bench/linalg/results/raw.tsv"),
               ("laptop", "aocl", ROOT / "bench/linalg/results/aocl_raw.tsv"),
               ("node", "main", NODE / "linalg/raw.tsv"),
               ("node", "main", NODE / "linalg/raw_unpinned.tsv")]  # 32/64 threads: unpinned (pinned runs starve faer's pool)
    acc = defaultdict(list)
    for m, sess, path in sources:
        for row in tsv(path):
            pinned_many = m == "node" and path.name == "raw.tsv" and int(row["threads"]) > 8
            unpinned_few = path.name == "raw_unpinned.tsv" and int(row["threads"]) <= 8
            if row["op"] in ("gen", "evd") and row["backend"] in LIBS and not pinned_many and not unpinned_few:
                acc[(m, sess, row["backend"], int(row["threads"]), int(row["n"]), row["op"])].append(float(row["median_ms"]))
    write("linalg", [{
        "machine": m, "session": s, "library": LIBS[b], "threads": t, "n": n,
        "op": "generalized eigenproblem" if op == "gen" else "symmetric eigenproblem", "ms": statistics.median(v),
    } for (m, s, b, t, n, op), v in sorted(acc.items())])
    # throughput mode: P concurrent single-thread processes, median over processes
    acc = defaultdict(list)
    for m, sess, path in [("laptop", "main", ROOT / "bench/linalg/results/throughput.tsv"),
                          ("laptop", "aocl", ROOT / "bench/linalg/results/aocl_throughput.tsv"),
                          ("node", "main", NODE / "linalg/throughput64.tsv"), ("node", "main", NODE / "linalg/throughput128.tsv")]:
        for row in tsv(path):
            if row["op"] in ("gen", "evd") and row["backend"] in LIBS:
                acc[(m, sess, int(row["mode"].removeprefix("concurrent")), row["backend"], int(row["n"]), row["op"])].append(float(row["median_ms"]))
    write("linalg_tp", [{
        "machine": m, "session": sess, "processes": p, "library": LIBS[b], "n": n,
        "op": "generalized eigenproblem" if op == "gen" else "symmetric eigenproblem", "ms": statistics.median(v),
    } for (m, sess, p, b, n, op), v in sorted(acc.items())])


def matrix() -> None:
    path = ROOT / "tests/reference_matrix/results.json"
    r = json.loads(path.read_text())
    cases = r["cases"]
    cases = list(cases.values()) if isinstance(cases, dict) else cases
    check_names = {"xtb": "axtb (xtb mode) vs xtb", "tblite": "axtb (tblite mode) vs tblite",
                   "axtb_vs_xtbmode": "axtb default vs its xtb mode", "hessian": "Hessian vs xtb --hess",
                   "threads": "1 vs 8 threads", "threads_hessian": "1 vs 8 threads, Hessian", "failure": "failure behaviour",
                   "calculator": "Calculator reuse vs fresh", "cli": "command line vs Python",
                   "analytic_ref": "analytic Hessian vs tight reference",
                   "analytic_numref": "analytic Hessian vs tight finite differences",
                   "analytic_xtb": "analytic Hessian vs xtb --hess", "analytic_threads": "1 vs 8 threads, analytic Hessian"}
    cat_names = {"elements": "Elements H–Rn", "structures": "Test structures", "multielement": "Mixed-element molecules",
                 "charges": "Charges", "spin": "Spin states", "open_shell_metals": "Open-shell metals",
                 "solvation": "Implicit solvation", "etemp_acc": "Electronic temperature and accuracy",
                 "hessian": "Hessians", "hessian_analytic": "Analytic Hessians", "threads": "Thread counts", "nodisp": "Dispersion off",
                 "failures": "Invalid or hard input", "api": "Python API and CLI"}
    agg = {}
    ncat = defaultdict(int)
    for c in cases:
        ncat[c["category"]] += 1
    for c in cases:
        for chk, res in c["checks"].items():
            if chk == "tblite_info":
                continue
            k = (c["category"], chk)
            a = agg.setdefault(k, {"id": f"{c['category']}|{chk}", "category": cat_names.get(c["category"], c["category"]),
                                   "check": check_names.get(chk, chk), "category_cases": ncat[c["category"]],
                                   "cases": 0, "pass": 0, "fail": 0, "different_solution": 0, "expected_difference": 0,
                                   "both_failed": 0, "ref_failed": 0, "max_dE": None, "max_dG": None, "max_dq": None})
            a["cases"] += 1
            st = res["status"]
            if st in a:
                a[st] += 1
            elif st != "info":
                a["fail"] += 1
            if st == "pass":
                for key, f in (("dE", "max_dE"), ("dG", "max_dG"), ("dq", "max_dq")):
                    if key in res:
                        v = abs(res[key])
                        a[f] = v if a[f] is None else max(a[f], v)
    order = list(cat_names)
    write("matrix", sorted(agg.values(), key=lambda a: (order.index(a["id"].split("|")[0]), a["check"])))
    # documented differences of the default conventions
    groups = {}
    for c in cases:
        for chk, res in c["checks"].items():
            if res["status"] == "expected_difference":
                note = res.get("note", "").split(": -TS")[0]
                g = groups.setdefault(note, {"reason": note, "cases": 0, "min_dE": None, "max_dE": None})
                g["cases"] += 1
                v = abs(res["dE"])
                g["min_dE"] = v if g["min_dE"] is None else min(g["min_dE"], v)
                g["max_dE"] = v if g["max_dE"] is None else max(g["max_dE"], v)
    write("matrix_expected", list(groups.values()))
    # different SCF solutions
    write("matrix_solutions", [{
        "id": f"{c['id']}|{chk}", "case": c["label"], "check": check_names.get(chk, chk), "dE": res["dE"],
        "lower": (res.get("diagnosis") or {}).get("lower", ""),
    } for c in cases for chk, res in c["checks"].items() if res["status"] == "different_solution"])
    # failure behaviour of each program
    def outcome(o):
        if not o:
            return "—"
        if o["outcome"] == "ok":
            return "returns a result"
        if o["outcome"] == "garbage":
            s = f"returns an energy; {o['why']}"
            return s + ("; then crashes (double free)" if "crashed_after" in o else "")
        return "stops with an error"
    fails = []
    for c in cases:
        if c["category"] != "failures":
            continue
        f = c["checks"]["failure"]
        oc = f.get("outcomes", {})
        fails.append({"case": c["label"], "expected": {"error": "an error", "ok": "a result", "any": "either"}[f.get("expect", "any")],
                      "xtb": outcome(oc.get("xtb")), "tblite": outcome(oc.get("tblite")), "axtb": outcome(oc.get("axtb"))})
    write("matrix_failures", fails)


def conventions() -> None:
    # the "xtb" column: xtb 6.8.0 (conventions "xtb") and 6.7.1 (conventions "xtb-6.7.1")
    write("conventions", [
        {"aspect": "Physical constants", "axtb": "CODATA 2022 (bohr 0.529177210544 Å, Boltzmann constant)",
         "xtb": "6.8.0: radius tables with mctc-lib's CODATA 2018 bohr, as tblite; 6.7.1: with bohr 0.52917726 Å; xtb's Boltzmann constant in both",
         "tblite": "tblite's: CODATA 2018 bohr (mctc-lib); xtb's Boltzmann constant"},
        {"aspect": "D4 coordination-number radii and electronegativities", "axtb": "double precision",
         "xtb": "6.8.0: double precision; 6.7.1: single-precision literals, as in its source", "tblite": "double precision"},
        {"aspect": "D4 cutoffs (two-body / three-body / CN)", "axtb": "72 / 40 / 40 bohr with a smooth 1 bohr switch",
         "xtb": "6.8.0: 50 / 25 / 30 bohr, smoothed (tblite's); 6.7.1: 72 / 40 / 40 bohr, hard", "tblite": "50 / 25 / 30 bohr, smoothed"},
        {"aspect": "Coordination number in energy and gradient", "axtb": "one consistent set",
         "xtb": "6.8.0: as tblite; 6.7.1: energy and gradient use different precision", "tblite": "as tblite"},
        {"aspect": "Integral screening", "axtb": "xtb's formula, no pair cutoff", "xtb": "6.8.0: tblite's threshold; 6.7.1: xtb's formula",
         "tblite": "tblite's"},
        {"aspect": "Starting charges", "axtb": "EEQ", "xtb": "EEQ (both versions)", "tblite": "superposition of atomic densities"},
        {"aspect": "SCC loop", "axtb": "tblite's", "xtb": "6.8.0: tblite's, with its own gradient potentials; 6.7.1: its own (residual test, extra iteration, mixer metric, neglected integrals, gradient from mixed moments)",
         "tblite": "tblite's"},
        {"aspect": "Electronic temperature at or below 0.1 K", "axtb": "Fermi smearing at any T > 0",
         "xtb": "integer (aufbau) occupations, xtb's rule (both versions)", "tblite": "Fermi smearing"},
        {"aspect": "ALPB/GBSA", "axtb": "corrected: fresh Born-radius term (xtb#1504) and the reference-state shift from the solvent's density (xtb#1505)",
         "xtb": "6.8.0: both bugs fixed, xtb's constants; 6.7.1: both bugs kept", "tblite": "corrected (tblite's own model has neither bug)"},
        {"aspect": "Size of the difference to the default", "axtb": "—",
         "xtb": "~1e-8 Eh for ordinary molecules (6.8.0: up to 1e-5 Eh for molecules longer than 25 bohr); 6.7.1: 5–7e-3 Eh in the 'reference' solvation state, up to 3e-3 Eh for heavy–light atom pairs in solvent",
         "tblite": "~1e-8 Eh"},
    ])


def libraries() -> None:
    write("libraries", [
        {"library": "faer (pure Rust)", "license": "MIT", "packaging": "Nothing to ship; static CLI and abi3 wheel",
         "threads": "Per call (Par) on axtb's own rayon pool", "verdict": "Default backend, and the library of the timings on the AMD node; always used below 200 basis functions"},
        {"library": "OpenBLAS", "license": "BSD-3", "packaging": "25–30 MB; C toolchain for static builds; scipy-openblas32 wheel",
         "threads": "Global by default; per-thread openblas_set_num_threads_local", "verdict": "Run-time backend (AXTB_LAPACK): OpenMP or sequential builds; pthreads builds are refused"},
        {"library": "Intel oneMKL", "license": "Proprietary (redistributable)", "packaging": "~300 MB runtime; static only via oneAPI",
         "threads": "mkl_set_num_threads_local; libiomp5 spin-waits", "verdict": "Run-time backend (AXTB_LAPACK), never linked into the LGPL build: the fastest eigensolver on the Intel laptop at every thread count and the library of the laptop's timings; slowest in throughput mode on the AMD node"},
        {"library": "AMD AOCL (BLIS + libFLAME)", "license": "BSD-3", "packaging": "Built from GitHub sources (the prebuilt tarball's EULA restricts use)",
         "threads": "bli_thread_set_num_threads_local", "verdict": "Run-time backend (AXTB_LAPACK with AXTB_LAPACK_BLAS): among the fastest in throughput mode on the AMD node and measured alongside faer in its timings; slower than MKL and OpenBLAS on the Intel laptop"},
        {"library": "nalgebra", "license": "Apache-2.0", "packaging": "Pure Rust",
         "threads": "None", "verdict": "Rejected: 6–9× slower eigensolver, 9e-6 residual at n = 1000"},
        {"library": "ndarray-linalg / lapack crates", "license": "MIT/Apache", "packaging": "Wrappers over a system LAPACK",
         "threads": "Whatever the LAPACK does", "verdict": "Rejected: forces copies; a small FFI layer suffices"},
        {"library": "cuSOLVER (via cudarc)", "license": "NVIDIA EULA", "packaging": "Runtime-loaded CUDA",
         "threads": "GPU", "verdict": "Optional GPU backend (a development experiment, not in this release): worth it on the node for basis sets above ~1000 functions; not for small molecules"},
        {"library": "ELPA / ScaLAPACK", "license": "LGPL / BSD", "packaging": "MPI",
         "threads": "MPI ranks", "verdict": "Out of scope (n ≫ 10⁴)"},
    ])


def program_diffs() -> dict:
    """Measured differences between xtb and tblite (tests/program_differences.py), summarized."""
    path = ROOT / "bench/results/program_diffs.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    if isinstance(raw, dict):  # {"xtb": "6.7.1 (26dd68d)", "tblite": ..., "rows": [...]}
        raw = raw["rows"]
    rows, ev = [], {}
    solv = [r for r in raw if r["topic"] == "solvation reference state"]
    for model in ("ALPB", "GBSA"):
        for state in ("gsolv", "bar1M", "reference"):
            sub = [r for r in solv if r["model"] == model and r["state"] == state and r["xtb_minus_tblite"] is not None]
            if not sub:
                continue
            xs = [r["xtb_minus_tblite"] for r in sub]
            ax = [r["axtb_minus_tblite"] for r in sub if r["axtb_minus_tblite"] is not None]
            rows.append({"id": f"solv|{model}|{state}", "topic": "Implicit solvent, reference state",
                         "case": f"{model}, water, '{state}' state: {len(sub)} solutes (neutral, anion, cation)",
                         "quantity": "E − E(tblite), largest |value|", "xtb": max(xs, key=abs), "tblite": 0.0,
                         "axtb": max(ax, key=abs) if ax else None})
            if state == "reference":
                ev["reference_state"] = max(abs(v) for v in xs)
    for r in raw:
        if r["topic"] == "atom-order dependence":
            rows.append({"id": "csoh", "topic": "Atom order in implicit solvent", "case": r["case"],
                         "quantity": "energy change from reversing the atom order", "xtb": r["xtb"], "tblite": r["tblite"], "axtb": r["axtb"]})
            ev["csoh"] = r["xtb"]
        if r["topic"] == "occupations at low electronic temperature":
            rows.append({"id": f"lowT|{r['case']}", "topic": "Occupations at low electronic temperature", "case": r["case"],
                         "quantity": "E − E(tblite)", "xtb": r["xtb_minus_tblite"], "tblite": 0.0, "axtb": r["axtb_minus_tblite"]})
            if r["case"].endswith("0.1 K"):
                ev["lowT"] = r["xtb_minus_tblite"]
    write("program_diffs", rows)
    xt = json.loads((OUT / "xtb_tblite.json").read_text()) if (OUT / "xtb_tblite.json").exists() else []
    old = [r for r in xt if r["d_671"] is not None]
    if old:
        big = max(old, key=lambda r: abs(r["d_671"]))
        small = [abs(r["d_671"]) for r in old if r["atoms"] <= 30]
        ev["gas_max"] = (big["molecule"], big["atoms"], big["d_671"], min(small, default=None), max(small, default=None))
    new = [abs(r["d_680"]) for r in xt if r["d_680"] is not None]
    if new:
        ev["gas_680"] = max(new)
    return ev


def extra_issues() -> list[dict]:
    """Findings about xtb and the g-xTB program measured by the Hessian and g-xTB work."""
    out = []
    xa = ROOT / "bench/results/xtb_gfn2_hess_acc.json"
    if xa.exists():
        x = json.loads(xa.read_text())
        r = {q["setting"].split(" ")[0]: q for q in x["rows"]}
        acc, xc = r["--acc"], r["$hess"]
        x8p = ROOT / "bench/results/xtb_gfn2_hess_acc_6.8.0.json"
        fixed = ""
        if x8p.exists():
            x8 = {q["setting"].split(" ")[0]: q for q in json.loads(x8p.read_text())["rows"]}
            fixed = (f"; xtb 6.8.0 passes --acc to them (PR #1500): --acc 0.001 is {x8['--acc']['max_dH_vs_analytic']:.1e} Eh/bohr² "
                     f"from the analytic Hessian (6.7.1: {acc.get('max_dH_vs_analytic', float('nan')):.1e})")
        out.append({"program": "xtb 6.7.1", "finding": "For GFN2-xTB, --acc does not reach the displaced SCFs of --hess; only $hess sccacc in an xcontrol file does",
                    "consequence": (f"Caffeine: --acc 0.001 changes the Hessian by {acc['max_dH_vs_acc1']:.1e} Eh/bohr² (reference SCF only); "
                                    f"$hess sccacc=0.001 changes it by {xc['max_dH_vs_acc1']:.1e} and the lowest mode from {x['reference']['lowest_freqs'][0]} "
                                    f"to {xc['lowest_freqs'][0]} cm⁻¹. Default GFN2 Hessians carry ~1e-4 Eh/bohr² of SCF noise"),
                    "in_axtb": "Analytic Hessian: no displaced SCFs",
                    "status": "Documented in docs/validation.md (Hessians); fixed in xtb 6.8.0" + fixed})
    gf = OUT / "gxtb2_hess_freq.json"
    if gf.exists():
        g = json.loads(gf.read_text())
        def f(mol, meth):
            return next((q for q in g if q["molecule"] == mol and q["method"] == meth), None)
        d1, d3 = f("taxol", "program --hess --acc 1"), f("taxol", "program --hess --acc 0.001")
        if d1 and d3:
            out.append({"program": "g-xTB 2.0.1 program", "finding": "--hess at default accuracy (or plain --hess) gives badly wrong frequencies; $hess sccacc has no effect, only --acc",
                        "consequence": (f"Taxol: frequencies off by up to {d1['max_dfreq']:.0f} cm⁻¹ (mean {d1['mean_dfreq']:.0f}) at default accuracy, "
                                        f"{d3['max_dfreq']:.1f} cm⁻¹ at --acc 0.001 (a common tight setting). Displaced SCFs stop at the --acc thresholds and the gradient uses the "
                                        "previous iteration's potentials, so the finite difference amplifies the SCF error ~100×. Plain --hess does not tighten them for these molecules"),
                        "in_axtb": "Analytic g-xTB 2.0.1 Hessian, ~1e-4 cm⁻¹ from tight references", "status": "Documented in docs/gxtb-v2.0.1-spec.md §11.5; not reported upstream"})
    out += [
        {"program": "g-xTB v1.1 program", "finding": "The program differs from the derivation in the paper's supporting information in about ten places, and its Fock matrix is not the exact derivative of its energy",
         "consequence": "Examples: a factor 2 in the repulsion exponent, ATM charge scaling, ACP weights, linear vs square-root CN dependence, the onsite-exchange sign and Fock factor; the AES potential omits dipole–quadrupole and quadrupole–quadrupole terms, so the converged SCF is not stationary",
         "in_axtb": "Program mode reproduces the program (to 1e-12 Eh); the default uses the exact derivative", "status": "Documented in docs/gxtb-v1.1-spec.md; not reported upstream"},
        {"program": "g-xTB v1.1 program", "finding": "Its numerical gradient sets the first atom's gradient to minus the sum of the others, and carries ~2e-5 Eh/bohr of noise at default settings",
         "consequence": "Gradient errors from the finite-difference step and the non-stationary SCF; the printed 'dispersion' energy also omits the three-body term included in the total",
         "in_axtb": "Analytic gradients (exact derivative of the respective energy)", "status": "Documented; not reported upstream"},
        {"program": "g-xTB 2.0.1 program", "finding": "The electron count is not an integer (reference occupations do not sum exactly: water 7.9999999999502), so the HOMO carries a fractional occupation",
         "consequence": "~1e-11 Eh effects; in near-degenerate HOMO sets individual energy terms become eigensolver-dependent at ~1e-10 Eh",
         "in_axtb": "Reproduced in program mode; integer electron count in the default", "status": "Documented in docs/gxtb-v2.0.1-spec.md"},
    ]
    comp = ROOT / "bench/results/solvation_composite.json"
    if comp.exists():
        hf = next((c for c in json.loads(comp.read_text())["cases"] if c["id"] == "hf-stretched"), None)
        if hf:
            out.append({"program": "xtb", "finding": "Broyden mixing keeps every iteration (no history limit), so each iteration costs more than the last",
                        "consequence": (f"A 1000-iteration SCF retry of stretched HF (2 atoms) takes {hf['programs']['seconds']:.0f} s in the xtb programs, retried at higher electronic temperatures; "
                                        "axtb reproducing xtb's loop exactly takes as long"),
                        "in_axtb": "Default: Broyden, then DIIS as fallback, converges the same case at 300 K in 0.2 s (a different electronic temperature, so a different ΔG_solv)",
                        "status": "Documented in README; not reported upstream"})
    vib = ROOT / "bench/results/vibrational.json"
    if vib.exists():
        c = json.loads(vib.read_text())["results"]["gfn2"]["caffeine"]["same_hessian"]
        out.append({"program": "xtb", "finding": "Printed IR intensities come from dipoles of the displaced SCFs at the loose fixed accuracy of --hess",
                    "consequence": (f"Caffeine: intensities up to {c['max_ir_rel_diff_axtb_dipole_derivatives']:.0%} ({c['max_ir_abs_diff_axtb_dipole_derivatives']:.1f} km/mol) "
                                    f"off the exact dipole derivatives on xtb's own Hessian; {c['max_ir_rel_diff_axtb_dipole_derivatives_sccacc_0.001']:.2%} with $hess sccacc=0.001"),
                    "in_axtb": "Dipole derivatives from the SCC response (analytic and response Hessians)", "status": "Documented in README; not reported upstream"})
    gf = ROOT / "bench/results/gfnff_reproduction.json"
    if gf.exists():
        g = json.loads(gf.read_text())["scaling"]["systems"]
        w3 = next((x for x in g if x["natoms"] == 3000), None)
        w5 = next((x for x in g if x["natoms"] == 5100), None)
        if w3 and w5:
            out.append({"program": "xtb", "finding": "GFN-FF allocates a 4-byte array over all pairs of hydrogen-bond acceptors and every hydrogen (considered_ABH, N_AB² × N_H) in every energy evaluation",
                        "consequence": (f"Memory grows as N³: {w3['xtb_grad']['peak_rss_mib'] / 1024:.1f} GiB for a {w3['natoms']}-atom water cluster, and at {w5['natoms']} atoms the array alone would need "
                                        f"{w5['xtb_considered_ABH_gib']:.0f} GiB, so xtb stops with 'insufficient virtual memory'. This limits xtb's GFN-FF to a few thousand atoms"),
                        "in_axtb": f"No dense hydrogen-bond array: {w5['axtb_grad']['peak_rss_mib']:.0f} MiB at {w5['natoms']} atoms",
                        "status": "Reported: grimme-lab/xtb#1515; still in xtb 6.8.0 (8.9 GiB for 1000 waters, 6.7.1: 10.0 GiB)"})
        out += [
            {"program": "xtb", "finding": "GFN-FF's carbonyl hydrogen-bond energy jumps when C=O···H is exactly linear: a torsion about the C=O axis is undefined there and not damped",
             "consequence": "Acetone–water dimer with the water's H on the C=O axis: moving it 0.001 Å off the axis lowers the energy by 6.9e-4 Eh (0.43 kcal/mol); "
                            "nearby Hessians have modes down to −4245 cm⁻¹ (6.8.0: −4031), and at the axis xtb 6.7.1's --hess segfaults (hessian.F90:453); "
                            "6.8.0 has the same jump, and its --hess at the axis runs (no rotation before it) with a −358 cm⁻¹ mode",
             "in_axtb": "Reproduces xtb's energy, so the same jump; at the axis its analytic Hessian is not finite (the energy is not twice differentiable there), its numerical Hessian reproduces xtb 6.8.0", "status": "Reported: grimme-lab/xtb#1516; still in xtb 6.8.0"},
            {"program": "xtb", "finding": "GFN-FF's gradient is not the exact derivative of its energy in two places: the coordination-number derivative uses a truncated √π, and the pyridine-type hydrogen-bond gradient drops part of the lone-pair term",
             "consequence": "Up to 6e-8 Eh/bohr off finite differences of xtb's own energy", "in_axtb": "Reproduced, so energies and gradients match xtb", "status": "Documented in docs/gfnff-spec.md"},
            {"program": "xtb", "finding": "GFN-FF's topology depends on which eigenvectors the eigensolver returns for an exactly degenerate Hückel pair",
             "consequence": "D5h ferrocenium: π bond orders, and the bond parameters built from them, are not invariant to a rotation of the molecule; the one structure of 269 axtb does not reproduce",
             "in_axtb": "Same Hückel step; a degenerate pair gets axtb's eigensolver's vectors", "status": "Documented"},
        ]
    out += [
        {"program": "tblite (also xtb --tblite and the g-xTB 2.0.1 program)", "finding": "The printed molecular quadrupole counts the off-diagonal charge and dipole parts twice",
         "consequence": "The printed tensor is not a tensor: its eigenvalues change when the molecule is rotated (water: −1.37/−0.54/1.91, rotated 45°: −2.11/−0.16/2.26); xtb's own form is rotation invariant",
         "in_axtb": "Reproduced in the tblite and g-xTB program conventions; the real tensor otherwise", "status": "Found; not reported upstream yet"},
        {"program": "xtb 6.7.1", "finding": "Prints the charges and moments it mixes once more after the last SCF iteration, not the converged ones",
         "consequence": "At the default accuracy taxol's printed charges are 5.7e-5 e off the converged ones, and the printed quadrupole up to 1.3e-3", "in_axtb": "Converged values", "status": "Documented"},
        {"program": "g-xTB 2.0.1 program", "finding": "Its analytic gradient is wrong for some bonds lying exactly along a coordinate axis",
         "consequence": "[Gd(H2O)6]3+ and [Lu(H2O)6]3+ with waters on the axes: the y components of the metal and the two axial O atoms are off by 8.0e-5 and 1.4e-4 Eh/bohr; finite differences of the program's own energies agree with axtb, and after a rotation the program agrees too",
         "in_axtb": "Analytic gradient agrees with finite differences in any orientation", "status": "Found in the element survey; not reported upstream yet"},
        {"program": "g-xTB 1.1.0 program", "finding": "For 24 elements without parameters it fails in its first diagonalization yet exits with code 0 and prints no energy",
         "consequence": "A wrapper that trusts the exit code gets no energy and no error", "in_axtb": "Rejected with an error naming the element", "status": "Documented"},
        {"program": "g-xTB 1.1.0 program", "finding": "Its onsite-exchange Fock matrix doubles the off-diagonal population part, which is not rotation covariant",
         "consequence": "Program-convention energies depend on the molecule's orientation by up to ~1e-7 Eh (in the program and in axtb's program mode); axtb's default conventions are invariant to 2e-13",
         "in_axtb": "Reproduced in program mode; exact derivative in the default", "status": "Documented in docs/gxtb-v1.1-spec.md"},
        {"program": "xtb, tblite and g-xTB 1.1.0", "finding": "Each writes its own Molden dialect: Cartesian functions only, unnormalized primitives, and three different normalizations of the components",
         "consequence": "tblite (xtb --gxtb) writes the valence charge in the atomic-number column (O as 6); g-xTB 1.1.0 writes f functions in xtb's internal order instead of Molden's, "
                        "so standard readers show wrong f orbitals; a reader must know which program wrote a file",
         "in_axtb": "Standard Molden: spherical [5D]/[7F] (or Cartesian on request), normalized contractions, atomic numbers", "status": "Documented in README (Molden files)"},
        {"program": "g-xTB 2.0.1 and 1.1.0 programs", "finding": "At a finite electronic temperature 2.0.1 rounds its fractional electron count per spin channel and stops the Fermi-level search at 1.5e-8 electrons; "
                                                         "1.1.0 uses a single-precision Boltzmann constant and computes only the lowest N_α + 10 orbitals",
         "consequence": "Up to 4.5e-8 Eh at 5000 K from the constant alone (EuCl3); slowly converging SCFs (Fe4, TmH3 at 3000–5000 K) end where rounding decides the last digits",
         "in_axtb": "Reproduced in program mode; exact Fermi level and CODATA 2022 kB in the default", "status": "Documented in the g-xTB specs"},
    ]
    return out


def issues(ev: dict) -> None:
    kcal = 627.5094740631  # Eh -> kcal/mol, CODATA 2022 (qcref)
    ref = ev.get("reference_state")
    cs = ev.get("csoh")
    low = ev.get("lowT")
    gm = ev.get("gas_max")
    write("issues", [
        {"program": "xtb", "finding": "OpenMP uses every core by default and nests threads",
         "consequence": "Concurrent runs oversubscribe the CPU unless OMP_NUM_THREADS=1 is set (see throughput)",
         "in_axtb": "One explicit thread count per calculation (default 1); no environment variables", "status": "Known; documented in xtb's own guidance"},
        {"program": "xtb and tblite", "finding": "Stale variable in the Born-radius integral for engulfed spheres",
         "consequence": (f"ALPB/GBSA energies depend on atom order for heavy–light pairs: reversing compressed CsOH changes the energy by {cs:.1e} Eh in both programs"
                         if cs else "ALPB/GBSA energies depend on atom order for heavy–light pairs") + "; the gradient is not the energy's derivative",
         "in_axtb": "Fixed by default and in xtb (6.8.0) mode; kept in xtb-6.7.1 mode",
         "status": "Reported: grimme-lab/xtb#1504 (fixed in xtb 6.8.0), tblite/tblite#430"},
        {"program": "xtb", "finding": "ALPB/GBSA 'reference' state uses ln(ρ·89.24/M²) where the solvent's number density was meant",
         "consequence": (f"xtb is {ref:.2e} Eh ({ref * kcal:.1f} kcal/mol) away from tblite for every solute in water; 5–7e-3 Eh over all solvents"
                         if ref else "3.1 kcal/mol off for water; 5–7e-3 Eh over all solvents"),
         "in_axtb": "Fixed by default, in tblite and xtb (6.8.0) mode; kept in xtb-6.7.1 mode",
         "status": "Reported: grimme-lab/xtb#1505; fixed in xtb 6.8.0 (6.8.0 − tblite in 'reference': 9.9e-9 Eh)"},
        {"program": "xtb", "finding": "D4 coordination number uses single-precision literals for radii and electronegativities",
         "consequence": "~1e-8 relative error in those parameters; the gradient path uses double precision",
         "in_axtb": "Double precision by default and in xtb (6.8.0) mode; reproduced in xtb-6.7.1 mode",
         "status": "Reported: grimme-lab/xtb#1506; fixed in xtb 6.8.0"},
        {"program": "xtb", "finding": "Small-gap Broyden damping tests iter == 0 inside a loop that starts at 1",
         "consequence": "Dead code: the intended damping never happens", "in_axtb": "Not reproduced",
         "status": "Reported: grimme-lab/xtb#1506; removed in xtb 6.8.0"},
        {"program": "xtb", "finding": "Radius tables converted with an older bohr constant (0.52917726 Å)",
         "consequence": "xtb and tblite differ by up to ~1e-8 Eh", "in_axtb": "CODATA 2022 by default; reproduced in each mode",
         "status": "Parametrization detail; xtb 6.8.0 uses mctc-lib's (CODATA 2018) bohr, as tblite"},
        {"program": "xtb 6.7.1 vs tblite", "finding": "Different D4 cutoffs (xtb: 72/40 bohr hard; tblite: 50/25 bohr smoothed), plus the constants above",
         "consequence": (f"The two programs' GFN2-xTB energies differ by {gm[3]:.0e}–{gm[4]:.0e} Eh for molecules up to 30 atoms and more for larger ones: {abs(gm[2]):.1e} Eh "
                         f"({abs(gm[2]) * kcal:.2f} kcal/mol) for {gm[0]} ({gm[1]} atoms)") if gm else "The two programs disagree by ~1e-5 Eh for molecules longer than 25 bohr",
         "in_axtb": "Both reproduced; default 72/40 with a smooth switch",
         "status": "Design difference; xtb 6.8.0 uses tblite's 50/25 bohr smoothed cutoffs" + (
             f" and agrees with tblite to {ev['gas_680']:.0e} Eh on the same molecules" if ev.get("gas_680") is not None else "")},
        {"program": "xtb vs tblite", "finding": "xtb switches to integer (aufbau) occupations at or below 0.1 K; tblite always smears",
         "consequence": (f"O2 singlet at 0.1 K: xtb is {abs(low):.1e} Eh ({abs(low) * kcal:.1f} kcal/mol) {'below' if low < 0 else 'above'} tblite; at 1 K they agree"
                         if low else "Different energies for degenerate frontier orbitals at very low temperature"),
         "in_axtb": "Smearing at any T > 0 by default; xtb's rule in both xtb modes", "status": "Design difference; unchanged in xtb 6.8.0"},
        {"program": "xtb and tblite", "finding": "Invalid electron counts are accepted: more unpaired electrons than electrons, a charge above the electron count, a fractional charge",
         "consequence": "An energy is returned whose atomic charges do not sum to the molecular charge; tblite then crashes with a double free",
         "in_axtb": "Rejected with an error naming the problem", "status": "Not yet reported; unchanged in xtb 6.8.0"},
        {"program": "xtb", "finding": "A negative electronic temperature is accepted silently (aufbau occupations); accuracy 0 or below 1e-4 is clamped to 1e-4 with a warning",
         "consequence": "The run uses other settings than requested", "in_axtb": "Rejected, or used as given", "status": "Not yet reported; unchanged in xtb 6.8.0"},
        {"program": "xtb 6.7.1 release", "finding": "The GFN2 gradient leaves out the H0/S term of AO pairs whose overlap vanishes by symmetry (atoms sharing a Cartesian coordinate)",
         "consequence": "Up to 3.3e-3 Eh/bohr (350 gradient checks of the reference matrix, e.g. guanidinium 2.7e-3); the energy is right. The g-xTB 2.0.1 build (6.7.1 + 127 commits) and 6.8.0 have the fix",
         "in_axtb": "xtb-6.7.1 mode reproduces the fixed program (the g-xTB build)", "status": "Fixed upstream (xtb PR #1208, March 2025); in no 6.7.x release"},
        *extra_issues(),
        {"program": "axtb (own bug, fixed)", "finding": "A starting-charge LU solve used faer's global parallelism, which builds a thread pool with one thread per CPU even when one thread was requested",
         "consequence": "Every axtb process started ~126 idle threads on a cluster node; taxol used 1.9 s of CPU for 0.2 s of wall time", "in_axtb": "Fixed: explicit sequential solve; a test checks that a one-thread run starts no threads", "status": "Fixed"},
        {"program": "faer dependencies", "finding": "private-gemm-x86 0.1.22 and gemm-common 0.18.2 open every cache file of every CPU in /sys at start-up",
         "consequence": "About 5,800 file opens per process on a 128-thread node; with 64 processes starting together each takes ~0.4 s",
         "in_axtb": "Patched copies that read one CPU only, vendored in third_party/ until upstream releases a fix", "status": "Reported: sarah-quinones/gemm#53, sarah-quinones/gemm-x64-v2#8"},
        {"program": "axtb (own bug, fixed)", "finding": "Off-by-one when porting the 1-based Fortran double-factorial table",
         "consequence": "p and d functions unnormalized; invisible in energies, caught by a unit test", "in_axtb": "Fixed; self-overlap now 1 to 1e-9", "status": "Fixed"},
    ])


def learnings(res: dict, ev: dict) -> None:
    lap, node = res.get("laptop"), res.get("node")

    def tpw(r, label, jobs):
        return next(x["wall"] for x in r["throughput"] if x["label"].startswith(label) and x["jobs"] == jobs)
    xc = {r["id"].split(":")[0]: r for r in (json.loads((OUT / "xtb680_changes.json").read_text()) if (OUT / "xtb680_changes.json").exists() else [])}
    rows = [
        {"topic": "Reproducing a program", "lesson": "Matching a reference program to "
         + (f"{xc['xtb-6.8.0']['dE_tight']:.0e} Eh" if "xtb-6.8.0" in xc else "1e-12 Eh") + " means reproducing its constants, not only its equations: "
         + (f"xtb 6.7.1's older bohr constant moves energies by up to {xc['radii']['dE_tight']:.0e} Eh and its single-precision literals by {xc['D4 CN']['dE_tight']:.0e} Eh"
            if "radii" in xc and "D4 CN" in xc else "xtb's older bohr constant and single-precision literals move energies")
         + ", more than a loose comparison would ever show. "
         "A second, independent implementation (tblite) separated program quirks from the method's definition."},
        {"topic": "Different answers are not always bugs", "lesson": "Open-shell metal clusters at low electronic temperature and one test structure have several self-consistent solutions. "
         "Restarting each program from the other's converged moments shows whether a mismatch is a second solution of the same equations "
         "or an implementation error; every mismatch in the matrix was settled this way."},
        {"topic": "Cheap invariance tests find real bugs", "lesson": "Permuting the atoms of CsOH changed xtb's and tblite's ALPB energy"
         + (f" by {ev['csoh']:.1e} Eh. " if ev.get("csoh") else ". ") +
         "That one test led to the Born-radius bug, and reading the same routine found the reference-state error."},
        {"topic": "Input validation", "lesson": "xtb and tblite return energies for impossible electron counts, with atomic charges that do not add up to the molecular charge. "
         "A wrapper cannot see this from the energy alone; axtb checks electron counts before and after filling the orbitals."},
    ]
    if node:
        big = max((x for x in node["throughput"]), key=lambda x: x["jobs"])
        rows.append({"topic": "Threads", "lesson":
                     f"With OpenMP left to its defaults, a batch of {big['jobs']} xtb jobs on {big['workers']} workers took "
                     f"{tpw(node, 'xtb, OMP_NUM_THREADS unset', big['jobs']):.1f} s on the 64-core node instead of "
                     f"{tpw(node, 'xtb, OMP_NUM_THREADS=1', big['jobs']):.1f} s with one thread per job. "
                     "axtb has no implicit threading, so there is no setting to forget."})
    fl = FINAL["laptop"]["dir"] / "gfn2-eg-t1.json"
    if fl.exists():
        sm = {s["program"]: s for s in json.loads(fl.read_text())["summary"] if s["molecule"] == "water"}
        x, a = sm.get("xtb 6.8.0"), sm.get("axtb CLI")
        if x and a and a.get("inner_median"):
            rows.append({"topic": "Start-up dominates small molecules", "lesson":
                         f"A water energy and gradient takes {a['inner_median'] * 1e3:.1f} ms of calculation inside an axtb process, {a['median'] * 1e3:.1f} ms for the whole process, "
                         f"and {x['median'] * 1e3:.0f} ms for the xtb 6.8.0 program (laptop, one thread). For small molecules, how a calculation is started matters more than how fast its code is."})
    # eigensolver libraries, from the linear-algebra datasets
    la = json.loads((OUT / "linalg.json").read_text())
    tp = json.loads((OUT / "linalg_tp.json").read_text())

    def fastest(rows):
        return min(rows, key=lambda r: r["ms"])["library"] if rows else None
    lap_fast = {t: fastest([r for r in la if r["machine"] == "laptop" and r["session"] == "main" and r["threads"] == t
                            and r["n"] == 1000 and r["op"] == "generalized eigenproblem"]) for t in (1, 4, 8)}
    node_tp = [r for r in tp if r["machine"] == "node" and r["processes"] == 64 and r["op"] == "generalized eigenproblem"]
    if node_tp:
        ratio = {}
        for r in node_tp:
            f = next(q["ms"] for q in node_tp if q["library"] == LIBS["faer22"] and q["n"] == r["n"])
            ratio.setdefault(r["library"], []).append(r["ms"] / f)
        slowest = max(ratio, key=lambda k: statistics.mean(ratio[k]))
        best_small = fastest([r for r in node_tp if r["n"] == min(q["n"] for q in node_tp)])
        rows.append({"topic": "Math libraries depend on the CPU", "lesson":
                     (f"For the generalized eigenproblem at n = 1000 on the Intel laptop, {lap_fast[1]} was the fastest library at 1, 4 and 8 threads. "
                      if len(set(lap_fast.values())) == 1 else
                      f"For the generalized eigenproblem at n = 1000 on the Intel laptop, the fastest library was {lap_fast[1]} on 1 thread and {lap_fast[8]} on 8. ") +
                     f"With 64 single-thread processes on the AMD Zen 5 node, {slowest} was the slowest "
                     f"({min(ratio[slowest]):.2f}–{max(ratio[slowest]):.2f}× faer's time) and {best_small} the fastest at n = {min(q['n'] for q in node_tp)}. "
                     "A library choice measured on one CPU does not carry over to another."})
        t64 = {r["n"]: r["ms"] for r in node_tp if r["library"] == LIBS["faer22"]}
        t128 = {r["n"]: r["ms"] for r in tp if r["machine"] == "node" and r["processes"] == 128 and r["op"] == "generalized eigenproblem" and r["library"] == LIBS["faer22"]}
        gain = [(128 / t128[n]) / (64 / t64[n]) for n in t64 if n in t128]
        if gain:
            rows.append({"topic": "SMT", "lesson":
                         f"128 single-thread eigensolver processes on the 64-core node each ran about {statistics.mean(t128[n] / t64[n] for n in t64 if n in t128):.1f}× slower than 64, "
                         f"so filling the SMT threads changed linear-algebra throughput by a factor of {min(gain):.2f}–{max(gain):.2f}."})
    ha = OUT / "hess_accuracy.json"
    if ha.exists():
        acc = json.loads(ha.read_text())
        gas = [r for r in acc if "ALPB" not in r["case"]]
        solv = [r for r in acc if "ALPB" in r["case"]]
        num_g = max(r["max_dfreq"] for r in gas if r["method"] == "axtb finite differences")
        ana = max(r["max_dfreq"] for r in acc if r["method"] == "axtb analytic")
        num_s = max((r for r in solv if r["method"] == "axtb finite differences"), key=lambda r: r["max_dfreq"], default=None)
        rows.append({"topic": "Analytic Hessians are more accurate, not only faster", "lesson":
                     f"Against a tightly converged reference the analytic Hessian's frequencies were within {ana:.0e} cm⁻¹ for every test system; "
                     f"finite differences at their default settings were off by up to {num_g:.1f} cm⁻¹ in the gas phase, from the SCF convergence of each displacement. "
                     + (f"With implicit solvent it is worse ({num_s['max_dfreq']:.0f} cm⁻¹ in the soft modes of {num_s['case'].split(' ')[0]}): the ALPB/GBSA surface term is smooth "
                        "in its first derivatives but not its second, so a finite-difference Hessian depends on the step, while the analytic one is the exact derivative at the point. "
                        if num_s else "")
                     + "For transition-state searches, where the imaginary and low modes decide the outcome, this matters as much as the speed."})
        xa, x8 = ROOT / "bench/results/xtb_gfn2_hess_acc.json", ROOT / "bench/results/xtb_gfn2_hess_acc_6.8.0.json"
        if xa.exists() and x8.exists():
            a6 = {q["setting"].split(" ")[0]: q for q in json.loads(xa.read_text())["rows"]}
            a8 = {q["setting"].split(" ")[0]: q for q in json.loads(x8.read_text())["rows"]}
            rows.append({"topic": "xtb --hess ignored --acc for GFN2-xTB", "lesson":
                         "xtb 6.7.1 runs the displacement SCFs of a GFN2-xTB Hessian at its own fixed SCC accuracy whatever --acc says; only $hess sccacc in an xcontrol file changes it. "
                         f"On caffeine its --hess --acc 0.001 is therefore {a6['--acc']['max_dH_vs_analytic']:.1e} Eh/bohr² from the analytic Hessian and $hess sccacc=0.001 "
                         f"{a6['$hess']['max_dH_vs_analytic']:.1e}, xtb's own finite-difference error. xtb 6.8.0 passes --acc to the displacements (PR #1500): "
                         f"{a8['--acc']['max_dH_vs_analytic']:.1e} at --acc 0.001."})
    cf = OUT / "cli_fix.json"
    if cf.exists():
        allc = json.loads(cf.read_text())
        bigjobs = max(r["jobs"] for r in allc)
        c = {r["setup"]: r for r in allc if r["jobs"] == bigjobs}
        o, e, f = c.get("axtb CLI as benchmarked"), c.get("axtb CLI, thread-pool fix only"), c.get("axtb CLI, thread-pool fix + cache-scan patch")
        if o and e and f:
            rows.append({"topic": "Process start-up on many-core machines", "lesson":
                         f"Two dependencies of faer (private-gemm-x86, gemm-common) read the CPU cache sizes by opening every cache file of every CPU in /sys: "
                         f"thousands of opens per process on a 128-thread node, which the kernel serializes when many processes start together. "
                         f"{o['jobs']} axtb CLI runs on {o['workers']} workers took {o['wall']:.2f} s as benchmarked, {e['wall']:.2f} s with axtb's own thread-pool fix alone, "
                         f"and {f['wall']:.2f} s once the scan read one CPU only. The laptop, with 16 threads, barely shows it."})
    ph = OUT / "phases.json"
    if ph.exists():
        pr = json.loads(ph.read_text())
        node_big = [r for r in pr if r["machine"] == "node" and r["phase"] == "Eigensolver"]
        if node_big:
            big = max(r["atoms"] for r in node_big)
            byt = {r["threads"]: r["seconds"] for r in node_big if r["atoms"] == big}
            if 8 in byt and max(byt) > 8:
                rows.append({"topic": "More threads can be slower", "lesson":
                             f"For the {big}-atom water cluster on the node, faer's eigensolver took {byt[8]:.2f} s on 8 threads and {byt[max(byt)]:.2f} s on {max(byt)}. "
                             "Beyond about 8 threads a single calculation gets slower; on a big node, run many calculations with a few threads each."})
    ss = OUT / "scf_summary.json"
    if ss.exists():
        sm = {r["method"]: r for r in json.loads(ss.read_text())}
        br, di = sm.get(SCF_METHODS["axtb-broyden"]), sm.get(SCF_METHODS["axtb-diis"])
        if br and di:
            rows.append({"topic": "A better SCF converger is not a free win", "lesson":
                         f"On a hard-SCF set (bond breaking, path images, metal clusters, small gaps) DIIS cut failures from {br['failed_300K'] + br['failed_0K']} to "
                         f"{di['failed_300K'] + di['failed_0K']} ({br['failed_300K']} to {di['failed_300K']} at 300 K; the 0 K failures have no stable whole-electron solution for any program). "
                         "But in systems with several SCF solutions it sometimes lands on a different one, once with an implausible −3 charge on an atom, and it lost one structure the reference mixer converges. "
                         "So axtb keeps the reference mixer and uses DIIS only as a fallback: nothing that converges today changes, and rescued results are labelled."})
    gf = OUT / "gxtb2_hess_freq.json"
    if gf.exists():
        g = json.loads(gf.read_text())
        d1 = next((q for q in g if q["molecule"] == "taxol" and q["method"] == "program --hess --acc 1"), None)
        d3 = next((q for q in g if q["molecule"] == "taxol" and q["method"] == "program --hess --acc 0.001"), None)
        if d1:
            rows.append({"topic": "Defaults can be badly wrong, and only an independent implementation shows it", "lesson":
                         f"The g-xTB 2.0.1 program's Hessian at its default accuracy puts taxol's frequencies up to {d1['max_dfreq']:.0f} cm⁻¹ off. Nothing in its output warns; "
                         "it took an independent analytic Hessian, checked against tight references, to see it. The common tighter setting (--acc 0.001) keeps the error "
                         + (f"at {d3['max_dfreq']:.1f} cm⁻¹" if d3 else "near 1 cm⁻¹") + ", but a check on a single small molecule would have suggested that the default was safe."})
    g2 = OUT / "gxtb2.json"
    if g2.exists():
        g = json.loads(g2.read_text())
        worst = max(abs(r["dE"]) for r in g if r.get("dE") is not None)
        rows.append({"topic": "A complete specification makes reimplementation fast", "lesson":
                     f"Once every g-xTB 2.0.1 term had been decoded and each had captured reference values, the implementation took a few hours of agent time: "
                     f"{len(g)} molecules across H–Lr reproduce the program to {worst:.1e} Eh with identical SCF iteration counts. Decoding was the slow part; "
                     "writing the code against per-term reference numbers was not."})
        rows.append({"topic": "The published derivation is not the program", "lesson":
                     "For g-xTB v1.1, the paper's supporting information and the released program disagree in about ten places. Reproducing a method's published numbers "
                     "means following the program; the supporting information is best read as intent."})
    gs = OUT / "gpu_speed.json"
    if gs.exists():
        g = [r for r in json.loads(gs.read_text()) if r["machine"] == "node" and r["n_ao"]]
        big = max(r["n_ao"] for r in g)
        cpu = min(r["seconds"] for r in g if r["n_ao"] == big and r["variant"] == "CPU (faer)")
        f64 = min(r["seconds"] for r in g if r["n_ao"] == big and r["variant"] == "GPU FP64")
        fast = min((r for r in g if r["n_ao"] == big and r["variant"] != "CPU (faer)"), key=lambda r: r["seconds"])
        rows.append({"topic": "GPUs without fast FP64", "lesson":
                     f"At {big} basis functions the RTX PRO 6000 eigensolver made a whole calculation {cpu / f64:.1f}× faster than the node's best CPU run in FP64 "
                     f"and {cpu / fast['seconds']:.1f}× with {fast['variant'].removeprefix('GPU ').lower()} precision. "
                     "Mixed precision helps less than the FP32/FP64 rate gap suggests, because its FP64 refinement steps are matrix multiplies at the card's low FP64 rate."})
    mem = OUT / "gxtb2_hess_memory.json"
    if mem.exists():
        mm = {r["id"]: r for r in json.loads(mem.read_text())}
        t1, w1 = mm.get("taxol 1t"), mm.get("(H2O)100 8t")
        if t1 and w1 and t1["rss_before_mb"]:
            rows.append({"topic": "Memory, not time, limited the g-xTB Hessian", "lesson":
                         f"The first g-xTB 2.0.1 analytic Hessian kept response data for every displacement, so its memory grew as N³: taxol needed {t1['rss_before_mb']:.0f} MB "
                         f"and a {w1['atoms']}-atom water cluster ran out of memory at 12 GB. Contracting each displacement's response as soon as it is made brought taxol to "
                         f"{t1['rss_after_mb']:.0f} MB and the water cluster to {w1['rss_after_mb']:.0f} MB, as for the GFN2-xTB Hessian. "
                         "Speed changes were then kept only if they gained 10%: a preconditioner that saved 7% for 170 lines was dropped."})
    cp = OUT / "composite.json"
    if cp.exists():
        hf = next((r for r in json.loads(cp.read_text()) if r["id"] == "hf-stretched"), None)
        if hf:
            rows.append({"topic": "The retry temperature is part of the result", "lesson":
                         f"A common retry ladder reruns a failed GFN2 solvation pair at a higher electronic temperature. For HF stretched to 3 Å the pair converged only at 5000 K, "
                         f"after {hf['t_programs']:.0f} s of failing 1000-iteration attempts; axtb's default converger finishes it at 300 K in {hf['t_axtb']:.1f} s, "
                         f"and the solvation correction then differs by {abs(hf['dE_default']):.3f} Eh ({abs(hf['dE_default']) * 627.5094740631:.0f} kcal/mol). "
                         "Which rung converged changes the energy, so a path profile mixing rungs is not one surface; record the rung with every result."})
    ml = OUT / "molden.json"
    if ml.exists():
        mo = json.loads(ml.read_text())
        uf6 = lambda r: r["program"] == MOLDEN_PROGRAMS["gxtb11"] and r["molecule"].startswith("UF6")
        rest = max((r["orbital"] for r in mo if not uf6(r) and r["orbital"] is not None), default=None)
        rows.append({"topic": "Comparing orbitals across programs needs overlaps, not coefficients", "lesson":
                     "xtb, tblite and g-xTB 1.1.0 each write Molden files with their own normalization and ordering, so coefficients cannot be compared directly. "
                     "Reading every file with an independent reader and comparing orbitals through the overlap of the two bases (1 − |⟨i|i⟩|, degenerate sets by singular values) "
                     + (f"showed agreement to {rest:.0e} or better per orbital (g-xTB 1.1.0's UF₆ apart, where the two SCFs take different paths) " if rest else "showed close agreement ")
                     + "and found the dialect differences on the way."})
    vc = OUT / "vib_conventions.json"
    if vc.exists():
        v = json.loads(vc.read_text())
        worst = max(v, key=lambda r: r["dG"])
        vb = OUT / "vibrational.json"
        same = max((r["dthermo_same"] for r in json.loads(vb.read_text()) if r["method"].startswith("GFN2")), default=None) if vb.exists() else None
        rows.append({"topic": "Most of a thermochemistry difference can be atomic masses", "lesson":
                     f"On the same Hessian, axtb's default vibrational analysis and xtb's differ by up to {max(r['dfreq'] for r in v):.2f} cm⁻¹ and {worst['dG']:.1e} Eh in G(RRHO) ({worst['molecule']}), "
                     "almost all from the hydrogen mass (1.0080 in current standard atomic weights, 1.00794 in xtb)."
                     + (f" With xtb's masses and projection the two agree to {same:.0e} Eh." if same else "")})
    gs = OUT / "gfnff_scaling.json"
    if gs.exists():
        g = {r["atoms"]: r for r in json.loads(gs.read_text())}
        w3, w5, w9 = g.get(3000), g.get(5100), g.get(9900)
        if w3 and w5 and w9:
            rows.append({"topic": "One array can set a program's size limit", "lesson":
                         f"xtb's GFN-FF needs {w3['xtb_mib'] / 1024:.1f} GiB for a {w3['atoms']}-atom water cluster and fails at {w5['atoms']} atoms, almost all from one hydrogen-bond bookkeeping array that grows as N³. "
                         f"Building the same hydrogen-bond lists from neighbour lists, axtb's GFN-FF needs {w3['axtb_mib']:.0f} MiB at {w3['atoms']} atoms and {w9['axtb_mib'] / 1024:.1f} GiB at {w9['atoms']}, "
                         "with identical energies where both run. Its only dense array is the charge model's matrix, which now sets the time above a few thousand atoms."})
    rows.append({"topic": "Differentiating code is not differentiating the formula", "lesson":
                 "GFN-FF's bend, torsion and hydrogen-bond gradients are 0/0 at exactly linear or planar geometries; xtb guards them. Differentiating the literal guarded code "
                 "gave Hessian errors of up to 0.15 Eh/bohr² for CO₂, hundreds of cm⁻¹ in its frequencies. The Hessian uses algebraically identical smooth forms, while energies and gradients "
                 "keep xtb's exact operations, and finite differences at linear and planar test geometries confirm both."})
    rows.append({"topic": "Bit-for-bit can mean the compiler's arithmetic", "lesson":
                 "Some GFN-FF topology decisions in symmetric molecules are ties decided in the last bit of a vector norm. The Intel compilers inline NORM2 with different fused multiply-add orders, "
                 "and xtb 6.7.1 (ifort) and 6.8.0 (ifx) round it differently, which decides which of several equidistant neighbours comes first. "
                 "Matching each, read from the binary's disassembly, was needed to reproduce NF₃'s out-of-plane term and a nitrogen torsion list."})
    pf, gr = ROOT / "bench/results/gxtb_parity_features.json", OUT / "guess_reuse.json"
    if pf.exists() and gr.exists():
        seqs = json.loads(gr.read_text())
        cut = lambda meths: [1 - r["iter_reuse"] / r["iter_fresh"] for r in seqs if r["method"] in meths and r["iter_fresh"]]
        c2, cx = cut({"GFN2-xTB"}), cut({"g-xTB 2.0.1", "g-xTB v1.1"})
        irc = next((r for r in seqs if "IRC" in r["sequence"]), None)
        sols = json.loads(pf.read_text()).get("solutions", [])
        up = max(((max((x for x in r["dE_reuse_minus_fresh"] if x is not None), default=0), r["molecule"]) for r in sols), default=(None, None))
        if c2 and cx:
            rows.append({"topic": "Restarting the SCF saves iterations but can change the answer", "lesson":
                         f"Starting each SCF from the previous geometry's solution cut GFN2-xTB's iterations per point by {min(c2):.0%}–{max(c2):.0%} along optimizations and random walks, "
                         f"g-xTB's by {min(cx):.0%}–{max(cx):.0%}" + (f" (an IRC from {irc['iter_fresh']:.0f} to {irc['iter_reuse']:.0f} per step)" if irc else "") + ". "
                         "But a restart follows its own SCF solution: recompressing a stretched bond (N2 at 3000 K, F2)"
                         + (f" ended up to {up[0]:.2f} Eh above a fresh start" if up[0] else " ended above a fresh start") +
                         ", metal clusters differed either way, and the programs do the same within one run. Paths through bond breaking should "
                         "check against fresh starts. During development a restarted g-xTB SCF also stopped just at the thresholds with history-dependent gradient noise, which doubled "
                         "a geomeTRIC optimization's steps until restarts converged ten times tighter."})
    write("learnings", rows)


def scope() -> None:
    def load(n):
        f = OUT / f"{n}.json"
        return json.loads(f.read_text()) if f.exists() else []
    eg, hs = load("ft_eg"), load("ft_hess")
    rng = lambda xs: f"{min(xs)}–{max(xs)}" if xs else "?"
    ths = lambda rows: ", ".join(str(t) for t in sorted({r["threads"] for r in rows if r["method"] != "g-xTB v1.1"}))
    timing = (f"Timings of GFN2-xTB and g-xTB energies, gradients and Hessians against xtb 6.8.0 and both g-xTB programs on an Intel laptop and a shared "
              f"64-core AMD node ({rng([r['atoms'] for r in eg + hs])} atoms; energy + gradient on {ths(eg)} threads, Hessians on {ths(hs)}), "
              "axtb with faer and with Intel MKL loaded at run time (and AMD AOCL on the node); earlier GFN2-xTB thread, batch and start-up runs on both machines")
    ffv, ffh, ffa, ffs = load("gfnff_versions"), load("gfnff_hessian"), load("gfnff_accuracy"), load("gfnff_scaling")
    gfnff = (f"GFN-FF topology, charges, energies and gradients against xtb 6.8.0 and 6.7.1 --gfnff on {max((r['structures'] for r in ffv), default='?')} structures "
             f"(H–Rn, charges −3 to +3, metal complexes, hydrogen- and halogen-bonded dimers, up to {max((r['largest'] for r in ffa), default='?')} atoms); "
             f"Hessians on {len(ffh)} structures; time and memory up to {max((r['atoms'] for r in ffs), default='?')} atoms")
    write("scope", [
        {"kind": "Tested", "item": "GFN2-xTB energies, analytic gradients, atomic charges and dipoles against xtb 6.8.0, xtb 6.7.1 (the g-xTB 2.0.1 build) and tblite 0.8.0"},
        {"kind": "Tested", "item": "Elements H–Rn, charges −3 to +3, spin multiplicities up to 11, open-shell transition-metal complexes"},
        {"kind": "Tested", "item": "ALPB (25 solvents) and GBSA (14 solvents) in all three reference states; electronic temperatures 0–5000 K; SCF accuracies 1–1e-6"},
        {"kind": "Tested", "item": "Semi-numerical Hessians and frequencies against xtb --hess; analytic Hessians against tightly converged references (8 benchmark systems, and 45 reference-matrix cases: closed and open shell, charges, transition metals, ALPB and GBSA in several solvents, high electronic temperature, all three conventions) and against xtb --hess; thread-count invariance; invalid inputs"},
        {"kind": "Tested", "item": timing},
        {"kind": "Tested", "item": "A GPU eigensolver (FP64, mixed and adaptive precision) on two GPUs without fast FP64, up to 6000 basis functions; it is kept on a separate branch, not in the main code"},
        {"kind": "Not tested", "item": "How the molecules were chosen: hand-picked test structures, generated water clusters and alkanes, and hand-built metal complexes. "
         "They are not optimized geometries and not a sample of any production workload."},
        {"kind": "Tested", "item": "g-xTB v1.1 and 2.0.1 against their programs for every element H–Lr (169 generated molecules each: hydrides, halides, oxides, aqua and ammine ions, radicals, f-shell high-spin), plus the reference sets"},
        {"kind": "Tested", "item": "Energy differences (conformers, binding, SN2 and HCN → HNC barriers) under program and default g-xTB conventions, on structures optimized with each version"},
        {"kind": "Tested", "item": "SCF restarts (each program's restart files reproduced), bond orders, HOMO–LUMO gaps and molecular quadrupoles against xtb 6.7.1, 6.8.0 and both g-xTB programs; restarts along random walks, geomeTRIC and Sella runs"},
        {"kind": "Tested", "item": "g-xTB 2.0.1 energies and analytic gradients against the 2.0.1 program on its reference set across H–Lr, including f-shell molecules"},
        {"kind": "Tested", "item": "g-xTB 2.0.1 analytic Hessians against tight references and the 2.0.1 program (9 systems including UF6, GdCl3 and ferrocene); memory and time up to a 600-atom water cluster"},
        {"kind": "Tested", "item": "g-xTB 2.0.1 and v1.1 at 300, 1000, 3000 and 5000 K against both programs (59 and 16 molecules), energies, gradients and SCF iterations"},
        {"kind": "Tested", "item": "g-xTB 2.0.1 in ALPB water through the composite correction, against the three xtb runs it replaces (9 structures, including two that need a retry at a higher electronic temperature)"},
        {"kind": "Tested", "item": "Vibrational analysis and quasi-RRHO thermochemistry against xtb 6.7.1 and the g-xTB 2.0.1 program (7 molecules, linear and symmetric tops, a transition state)"},
        {"kind": "Tested", "item": "Molden orbital files against the files of xtb, the g-xTB 2.0.1 program and g-xtb 1.1.0 (23 cases, s to f functions, open shells)"},
        {"kind": "Tested", "item": gfnff},
        {"kind": "Not tested", "item": "The solvation composite in solvents other than water, with GBSA or at other reference states; GFN-FF with implicit solvent or point charges (QM/MM electrostatic embedding); GFN1-xTB, periodic systems, CPCM-X and COSMO solvation, point charges"},
        {"kind": "Not tested", "item": "Geometry optimizations and molecular dynamics (axtb provides energies, gradients and Hessians; an outside optimizer drives them)"},
        {"kind": "Not tested", "item": "GFN2-xTB and g-xTB above 999 atoms (GFN-FF was run up to 9900); CPUs other than these two; macOS, Windows and ARM"},
    ])


GPU_NAO = {"taxol": 299, "water100": 600, "water200": 1200, "water333": 1998, "water667": 4002, "water1000": 6000}  # GFN2 basis functions
GPU_VARIANTS = {"cpu": "CPU (faer)", "cuda-f64": "GPU FP64", "cuda-mixed": "GPU mixed (auto steps)", "cuda-mixed2": "GPU mixed, 2 steps",
                "cuda-mixed3": "GPU mixed, 3 steps", "cuda-adaptive": "GPU adaptive"}


def gpu() -> None:
    """GPU eigensolver backend (a development experiment, not released): bench/gpu/results."""
    base = ROOT / "bench/gpu/results"
    if not base.exists():
        return
    best, acc = {}, {}
    for machine, files in (("node", ["cluster/final-small.jsonl", "cluster/final-big.jsonl", "cluster/bench-acc-imaged.jsonl"]),
                           ("laptop", ["laptop/interleaved.jsonl"])):
        for f in files:
            for line in open(base / f):
                if not line.strip():
                    continue
                r = json.loads(line)
                sysname = r["xyz"].removesuffix(".xyz")
                k = (machine, sysname, r["eig"], r["threads"])
                if f.startswith("cluster/final") or machine == "laptop":
                    best[k] = min(best.get(k, float("inf")), r["seconds"])
                if r["eig"] != "cpu":
                    a = acc.setdefault(r["eig"], {"dE": 0.0, "dg": 0.0, "n": 0, "nmax": 0})
                    a["dE"] = max(a["dE"], abs(r["dE"])); a["dg"] = max(a["dg"], abs(r["dg"])); a["n"] += 1
                    a["nmax"] = max(a["nmax"], GPU_NAO.get(sysname, 0))
    write("gpu_speed", [{"id": f"{m}|{s}|{e}|{t}", "machine": m, "system": s, "n_ao": GPU_NAO.get(s), "variant": GPU_VARIANTS.get(e, e),
                         "threads": t, "seconds": v} for (m, s, e, t), v in sorted(best.items(), key=lambda kv: (kv[0][0], GPU_NAO.get(kv[0][1], 0), kv[0][2], kv[0][3]))])
    write("gpu_accuracy", [{"variant": GPU_VARIANTS.get(e, e), "runs": a["n"], "largest_n_ao": a["nmax"], "max_dE": a["dE"], "max_dG": a["dg"]}
                           for e, a in sorted(acc.items())])
    rows = []
    import re
    for line in open(base / "cluster/throughput.txt"):
        m = re.match(r"(\d+) atoms eig=(\S+) workers=(\d+) ncalc=(\d+): wall ([\d.]+) s, ([\d.]+) calc/s, ([\d.]+) ms", line)
        if m:
            nat = int(m[1])
            rows.append({"id": f"{nat}|{m[2]}|{m[3]}", "molecule": {24: "caffeine", 113: "taxol"}.get(nat, f"{nat} atoms"), "atoms": nat,
                         "variant": GPU_VARIANTS.get(m[2], m[2]), "workers": int(m[3]), "per_second": float(m[6]), "latency_ms": float(m[7])})
    write("gpu_throughput", rows)


PHASE_GROUPS = [("Setup (classical terms, integrals, Cholesky)", ("classical", "integrals", "cholesky")),
                # axtb times the potentials and the Fock matrix as two phases ("scf: potential+fock" before)
                ("Fock build", ("scf: potential+fock", "scf: potentials", "scf: fock")), ("Eigensolver", ("scf: diagonalization",)),
                ("Other SCF (density, charges, energy, mixing)", ("scf: density", "scf: mulliken", "scf: energy+mixing")),
                ("Gradient", ("gradient",))]


def phases() -> None:
    rows = []
    for m, path in (("laptop", ROOT / "bench/results/phases.json"), ("node", NODE / "phases.json")):
        if not path.exists():
            continue
        for r in json.loads(path.read_text()):
            for g, keys in PHASE_GROUPS:
                rows.append({"id": f"{m}|{r['molecule']}|{r['threads']}|{g}", "machine": m, "molecule": r["molecule"], "atoms": r["atoms"],
                             "threads": r["threads"], "phase": g, "seconds": sum(r.get(k, 0.0) for k in keys)})
    write("phases", rows)


def cli_fix() -> None:
    path = NODE / "cli_throughput.json"
    if not path.exists():
        return
    acc = defaultdict(list)
    for r in json.loads(path.read_text()):
        acc[(r["label"], r["jobs"], r["workers"])].append(r["wall"])
    names = {"xtb, OMP_NUM_THREADS=1": "xtb, one thread per job",
             "axtb CLI processes (no env vars), orig": "axtb CLI as benchmarked",
             "axtb CLI processes (no env vars), eeq-fix": "axtb CLI, thread-pool fix only",
             "axtb CLI processes (no env vars), eeq-fix+cache-scan-cpu0": "axtb CLI, thread-pool fix + cache-scan patch"}
    rows = [{"id": f"{k}|{j}x{w}", "setup": names.get(k, k), "jobs": j, "workers": w, "batch": f"{j} jobs on {w} workers",
             "repeats": len(v), "wall": statistics.median(v)} for (k, j, w), v in acc.items()]
    write("cli_fix", rows)
    # the throughput batches on the node predate the start-up fixes: show the CLI as it is now (same batches, same node)
    fixed = {r["batch"]: r["wall"] for r in rows if r["setup"] == "axtb CLI, thread-pool fix + cache-scan patch"}
    tp_path = OUT / "throughput.json"
    if tp_path.exists() and fixed:
        tp = json.loads(tp_path.read_text())
        for r in tp:
            if r["machine"] == "node" and r["setup"].startswith("axtb CLI") and r["batch"] in fixed:
                r["wall"], r["setup"] = fixed[r["batch"]], "axtb CLI processes (with the start-up fixes)"
                r["id"] = f"node|{r['jobs']}x{r['workers']}|{r['setup']}"
        tp_path.write_text(json.dumps(tp, indent=1))


HESS_METHODS = {"axtb analytic": "axtb analytic", "axtb response": "axtb response-based", "axtb numerical": "axtb finite differences",
                "xtb": "xtb --hess"}


def hessian_analytic() -> None:
    """Analytic vs response-based vs finite-difference Hessians and xtb --hess (branch analytic-hessian,
    bench/hessian_benchmark.py)."""
    path = ROOT / "bench/results/hessian_analytic.json"
    if not path.exists():
        return
    d = json.loads(path.read_text())
    times, accs = [], []
    for case, r in d["cases"].items():
        for key, v in r.get("time", {}).items():
            meth, _, thr = key.rpartition(" ")
            if meth in HESS_METHODS and thr.endswith("t"):
                times.append({"id": f"{case}|{meth}|{thr}", "case": case, "atoms": r["natoms"], "method": HESS_METHODS[meth],
                              "threads": int(thr[:-1]), "seconds": v})
        for meth, a in r.get("accuracy", {}).items():
            accs.append({"id": f"{case}|{meth}", "case": case, "atoms": r["natoms"],
                         "method": HESS_METHODS.get(f"axtb {meth}", meth), "max_dH": a.get("max_dH"), "max_dfreq": a.get("max_dfreq"),
                         "asymmetry": r.get("asymmetry", {}).get(meth)})
    write("hess_time", times)
    write("hess_accuracy", accs)


def gxtb() -> None:
    """g-xTB v1.1 in axtb vs the g-xtb 1.1.0 program (tests/gxtb_reproduction.py)."""
    path = ROOT / "bench/results/gxtb_reproduction.json"
    if not path.exists():
        return
    rows = json.loads(path.read_text())["rows"]
    write("gxtb", [{"molecule": r["molecule"], "atoms": r["natoms"], "charge": r["charge"], "unpaired": r["uhf"],
                    "E_program": r["E_program"], "E_axtb": r["E_axtb"], "dE": r["dE"],
                    "iter_program": r["iter_program"], "iter_axtb": r["iter_axtb"],
                    "dE_corrected": r.get("dE_corrected"), "iter_corrected": r.get("iter_corrected")} for r in rows])


def gxtb2() -> None:
    """g-xTB 2.0.1 in axtb vs the 2.0.1 program (tests/gxtb2_reproduction.py)."""
    path = ROOT / "bench/results/gxtb2_reproduction.json"
    if not path.exists():
        return
    rows = json.loads(path.read_text())["rows"]
    write("gxtb2", [{"molecule": r["molecule"], "group": r.get("group"), "atoms": r["natoms"], "charge": r["charge"], "unpaired": r["uhf"],
                     "E_program": r["E_program"], "E_axtb": r.get("E_axtb"), "dE": r.get("dE"),
                     "iter_program": r["iter_program"], "iter_axtb": r.get("iter_axtb"),
                     "max_dterm": r.get("max_dterm"), "max_dgrad": r.get("max_dgrad_tight", r.get("max_dgrad_default")),
                     "dE_default": r.get("dE_default")} for r in rows])


def gxtb2_hessian() -> None:
    """g-xTB 2.0.1 Hessians: axtb analytic/response/numerical vs the 2.0.1 program (tests/gxtb2_hessian.py)."""
    path = ROOT / "bench/results/gxtb2_hessian.json"
    if not path.exists():
        return
    d = json.loads(path.read_text())
    times, freqs = [], []
    for mol, v in d.get("versus_program", {}).items():
        # window 1: program at its default accuracy and all three axtb methods (cases of the
        # interleaved accuracy run alone, e.g. Ni(CO)4, have only window 2)
        for r in v.get("runs", []):
            label, thr = r["what"].rsplit(" ", 1)
            label = {"program --hess": "program --hess, default --acc 1"}.get(label, label)
            times.append({"id": f"{mol}|1|{r['what']}", "molecule": mol, "atoms": v["natoms"], "window": "default", "method": label,
                          "threads": int(thr.rstrip("t")), "seconds": r["seconds"]})
        # window 2: program at --acc 0.001 (a common tight setting) and 1e-4, interleaved with axtb analytic
        pa = v.get("program_acc", {})
        for what, sec in pa.get("time", {}).items():
            label, thr = what.rsplit(" ", 1)
            label = label.replace("program --acc", "program --hess --acc")
            times.append({"id": f"{mol}|2|{what}", "molecule": mol, "atoms": v.get("natoms", pa.get("natoms")), "window": "acc", "method": label,
                          "threads": int(thr.rstrip("t")), "seconds": sec})
        for meth, r in pa.get("vs_reference", {}).items():
            freqs.append({"id": f"{mol}|{meth}", "molecule": mol, "method": meth.replace("program --acc", "program --hess --acc"),
                          "max_dfreq": r.get("max_dfreq"), "mean_dfreq": r.get("mean_dfreq"), "against": "tight reference"})
    for mol, v in d.get("program_accuracy", {}).items():
        for setting, r in v["runs"].items():
            if setting in ("--acc 1", "--acc 0.01"):
                freqs.append({"id": f"{mol}|program {setting}", "molecule": mol, "method": f"program --hess {setting}",
                              "max_dfreq": r.get("max_dfreq"), "mean_dfreq": r.get("mean_dfreq"), "against": "axtb analytic"})
    write("gxtb2_hess_time", times)
    write("gxtb2_hess_freq", freqs)


SCF_METHODS = {"axtb-broyden": "axtb, reference mixer (Broyden)", "axtb-diis": "axtb, DIIS", "axtb-auto": "axtb default (Broyden, then DIIS)",
               "xtb-6.7.1": "xtb 6.7.1", "tblite-0.8.0": "tblite 0.8.0"}


def scf_hard() -> dict:
    """Hard-SCF test set (tests/scf_hard_set.py): convergence per method."""
    path = ROOT / "bench/results/scf_hard.json"
    if not path.exists():
        return {}
    d = json.loads(path.read_text())
    cases = {c["id"]: c for c in d["cases"]}
    runs = d["runs"]
    methods = [m for m in SCF_METHODS if any(r["method"] == m for r in runs)]
    summ = []
    for m in methods:
        rr = [r for r in runs if r["method"] == m]
        ok = [r for r in rr if r["converged"]]
        f300 = sum(1 for r in rr if not r["converged"] and cases[r["case"]]["etemp"] > 0)
        f0 = sum(1 for r in rr if not r["converged"] and cases[r["case"]]["etemp"] == 0)
        its = sorted(r["iterations"] for r in ok)
        summ.append({"method": SCF_METHODS[m], "cases": len(rr), "converged": len(ok), "failed_300K": f300, "failed_0K": f0,
                     "median_iterations": statistics.median(its) if its else None})
    write("scf_summary", summ)
    by = defaultdict(dict)
    for r in runs:
        by[r["case"]][r["method"]] = r
    rows = []
    for cid, rs in by.items():
        conv = [r for r in rs.values() if r["converged"]]
        es = [r["energy"] for r in conv if r.get("energy") is not None]
        spread = (max(es) - min(es)) if len(es) > 1 else 0.0
        if len(conv) == len(rs) and spread < 1e-6:
            continue  # every program converged to the same answer
        c = cases[cid]
        row = {"id": cid, "case": c["label"], "category": c["category"], "etemp": c["etemp"], "energy_spread": spread}
        for m, key in (("axtb-broyden", "broyden"), ("axtb-diis", "diis"), ("axtb-auto", "auto"), ("xtb-6.7.1", "xtb"), ("tblite-0.8.0", "tblite")):
            r = rs.get(m)
            row[key] = None if r is None else (str(r["iterations"]) if r["converged"] else "fails")
        rows.append(row)
    write("scf_cases", sorted(rows, key=lambda r: (r["etemp"], r["category"], r["case"])))
    return {m["method"]: m for m in summ}


def gxtb2_hess_memory() -> None:
    """Peak memory and time of the g-xTB 2.0.1 analytic Hessian before and after the O(N^2) rewrite,
    and the time of each speed change (tests/gxtb2_hessian.py --memory/--speed)."""
    path = ROOT / "bench/results/gxtb2_hessian.json"
    if not path.exists():
        return
    d = json.loads(path.read_text())
    before = d.get("memory_before_fd71faa", {})
    notes = {"killed": "out of memory at a 12 GB limit after 7.6 min", "not run": "not run; estimated ~100 GB"}
    rows = []
    for key, r in d.get("memory", {}).items():
        b = before.get(key, {})
        note = next((v for k, v in notes.items() if k in (b.get("note") or "")), None)
        rows.append({"id": key, "system": key.rsplit(" ", 1)[0], "atoms": r["natoms"], "threads": r["threads"],
                     # the largest case ran in an 8-CPU job on the 64-core node (no host names in the data)
                     "machine": "64-core node" if "machine" in r else "laptop",
                     "rss_before_mb": b.get("peak_rss_mb"), "before_note": note,
                     "rss_after_mb": r["peak_rss_mb"], "rss_single_point_mb": r.get("peak_rss_mb_single_point"),
                     "seconds": r["seconds"]})
    write("gxtb2_hess_memory", rows)
    steps = []
    sp = d.get("speed_per_change_cluster")
    if sp:
        label = {"memory": "O(N²) memory (starting point)", "E": "+ faster response kernel, screened pairs",
                 "E+X": "+ closed-form exchange-kernel Hessian", "E+X+P": "+ shell-charge preconditioner"}
        kept = {"memory": "—", "E": "kept", "E+X": "kept", "E+X+P": "dropped: 7% for 170 lines"}
        base = sp["best_wall"]["memory"]
        for k, sec in sp["best_wall"].items():
            steps.append({"id": f"node|{k}", "machine": "64-core node", "step": label.get(k, k), "seconds": sec,
                          "speedup": base / sec, "decision": kept.get(k, "")})
    lap = d.get("interleaved_laptop_taxol_1t")
    if lap:
        label = {"fd71faa": "before (data stored per displacement)", "memory": "O(N²) memory", "final": "O(N²) memory + both kept speed changes"}
        base = lap["best_wall"]["fd71faa"]
        for k, sec in lap["best_wall"].items():
            steps.append({"id": f"laptop|{k}", "machine": "laptop", "step": label.get(k, k), "seconds": sec, "speedup": base / sec,
                          "decision": "released" if k == "final" else "—"})
    if steps:
        write("gxtb2_hess_steps", steps)


def gxtb_etemp() -> None:
    """g-xTB 2.0.1 and v1.1 at finite electronic temperature vs the programs (tests/gxtb_etemp_reproduction.py)."""
    path = ROOT / "bench/results/gxtb_etemp.json"
    if not path.exists():
        return
    d = json.loads(path.read_text())
    s = d["summary"]
    temps = sorted({r["etemp"] for r in d["v2"]})
    rows = []
    for key, label, prog in (("v2_default", "g-xTB 2.0.1, program's default thresholds", "xtb --gxtb --etemp T"),
                             ("v2_tight", "g-xTB 2.0.1, tight thresholds (--acc 0.0001)", "xtb --gxtb --etemp T --acc 0.0001"),
                             ("v1.1", "g-xTB v1.1", "gxtb -tel T")):
        r = s.get(key)
        if not r:
            continue
        src = d["v2"] if key.startswith("v2") else d["v1.1"]
        rows.append({"id": key, "setting": label, "program": prog, "molecules": len({x["molecule"] for x in src}),
                     "temperatures": ", ".join(f"{t:g}" for t in temps), "runs": r["runs"], "compared": r["compared"],
                     "both_fail": len(r.get("failures_as_program", [])), "rounding_sensitive": len(r.get("rounding_sensitive", {})),
                     "max_dE": r["max_abs_dE"], "max_dgrad": r.get("max_dgrad"),
                     "iteration_mismatches": len(r.get("iteration_mismatches", []))})
    write("gxtb_etemp", rows)


def vibrational() -> None:
    """Vibrational analysis and thermochemistry vs xtb 6.7.1 and the g-xTB 2.0.1 program (tests/vibrational_check.py)."""
    path = ROOT / "bench/results/vibrational.json"
    if not path.exists():
        return
    d = json.loads(path.read_text())
    p680 = ROOT / "bench/results/vibrational_6.8.0.json"
    gfn2_src, gfn2_lab = (json.loads(p680.read_text()), "GFN2-xTB vs xtb 6.8.0") if p680.exists() else (d, "GFN2-xTB vs xtb 6.7.1")
    rows = []
    for mol, r in gfn2_src["results"].get("gfn2", {}).items():
        sh, ah = r["same_hessian"], r["axtb_hessian"]
        rows.append({"id": f"gfn2|{mol}", "method": gfn2_lab, "molecule": mol, "atoms": r["n_atoms"],
                     "point_group": sh["point_group"]["axtb"], "point_group_program": sh["point_group"]["xtb"],
                     "symmetry_number": sh["symmetry_number"]["axtb"],
                     "dfreq_same": sh["max_frequency_diff"], "dthermo_same": max(abs(v) for v in sh["thermo_diff"].values()),
                     "dfreq_own": ah["max_frequency_diff"], "dG_own": abs(ah["thermo_diff"]["g_rrho"])})
    gv = d["results"].get("gxtb-v2.0.1", {})
    for mol, r in gv.items():
        sh, own = r["program_hessian_analysis_vs_printout"], r["axtb_vs_program_hessian"]
        rows.append({"id": f"gxtb2|{mol}", "method": "g-xTB 2.0.1 vs its program", "molecule": mol, "atoms": r["n_atoms"],
                     "point_group": r["axtb_hessian_xtb_thermo_vs_axtb_analysis"]["point_group"]["axtb"],
                     "point_group_program": r["axtb_hessian_xtb_thermo_vs_axtb_analysis"]["point_group"]["xtb"],
                     "symmetry_number": sh["symmetry_number"]["axtb"],
                     "dfreq_same": sh["max_frequency_diff"], "dthermo_same": max(abs(v) for v in sh["thermo_diff"].values()),
                     "dfreq_own": own["max_frequency_diff"], "dG_own": abs(own["g_rrho_diff"])})
    write("vibrational", rows)
    conv = []
    for mol, r in gfn2_src["results"].get("gfn2", {}).items():
        c = r.get("conventions_axtb_minus_xtb")
        if c:
            conv.append({"molecule": mol, "dfreq": c["max_frequency_diff"], "dG": abs(c["g_rrho"]), "dZPVE": abs(c["zpve"])})
    write("vib_conventions", conv)


MOLDEN_PROGRAMS = {"gfn2": "GFN2-xTB (xtb 6.7.1)", "gxtb2": "g-xTB 2.0.1", "gxtb11": "g-xTB 1.1.0"}


def molden() -> None:
    """Molden orbital files from axtb vs the programs' own files (tests/molden_check.py)."""
    path = ROOT / "bench/results/molden.json"
    if not path.exists():
        return
    rows = []
    for r in json.loads(path.read_text()):
        v = r["vs_program_spherical"]
        mol = r["case"].split("/", 1)[1]
        acc = "" if r["axtb_accuracy"] == r["accuracy"] == 1.0 else f" (accuracy {r['accuracy']:g}" + (
            f", axtb {r['axtb_accuracy']:g})" if r["axtb_accuracy"] != r["accuracy"] else ")")
        mh = r.get("mepd_homo")
        rows.append({"id": r["case"], "program": MOLDEN_PROGRAMS.get(r["program"], r["program"]), "molecule": mol.removesuffix("-acc1e-4").removesuffix("-axtb-acc1e-3") + acc,
                     "atoms": r["atoms"], "functions": f"{r['basis_functions']} / {r['cartesian_functions']}", "max_l": "spdf"[r["max_l"]],
                     "spin_sets": r["spin_sets"], "orbital": v["nondegenerate"], "degenerate": v["degenerate_blocks"],
                     "occupied": v["occupied_space"], "energy": v["energy"], "orthonormal": r["orthonormal_axtb_spherical_all"],
                     "mepd_homo": mh["max_diff_rel"] if mh else None})
    write("molden", rows)


def composite() -> None:
    """g-xTB 2.0.1 in ALPB water: axtb's in-process composite vs the three program runs it replaces
    (tests/solvation_composite.py; the `mepd` fields hold the optional private cross-check, empty
    without it)."""
    path = ROOT / "bench/results/solvation_composite.json"
    if not path.exists():
        return
    d = json.loads(path.read_text())
    rows = []
    for c in d["cases"]:
        p, dflt, m = c["axtb_program_mode"], c["axtb_default"], c["mepd"]
        hs = [h for h in c.get("hessian", []) if h["conventions"] == "axtb"]
        rows.append({"id": c["id"], "case": c["label"].split(" (")[0], "atoms": c["natoms"], "charge": c["charge"], "unpaired": c["uhf"],
                     "retry": c["programs"]["retry"], "dG_solv": c["programs"]["solvation_correction"],
                     "dE": p["dE"], "dgrad": p["max_abs_dgrad"], "dE_mepd": m["dE_vs_programs"],
                     "dfreq_hess": hs[0]["max_dfreq_cm"] if hs else None,
                     "t_programs": c["programs"]["seconds"], "t_mepd": m["seconds"],
                     "t_program_mode": p.get("seconds_new_calculator"),
                     "t_axtb": dflt["seconds_new_calculator"], "t_axtb_next": dflt["seconds_next_geometry"],
                     "dE_default": dflt["dE_vs_program_mode"]})
    write("composite", rows)


FT_METHOD = {"gfn2": "GFN2-xTB", "gxtb2": "g-xTB 2.0.1", "gxtb11": "g-xTB v1.1"}
# The final timings (bench/final_timings.py) on both machines. `library` is the requirement's setting on that
# CPU: Intel MKL loaded at run time on the Intel laptop, faer (axtb's default) on the AMD node; the other
# libraries ran alongside.
FINAL = {
    "laptop": {"dir": ROOT / "bench/results/final-laptop", "label": "Laptop", "library": "MKL"},
    "node": {"dir": ROOT / "bench/results/final-9985wx", "label": "64-core node", "library": "faer"},
}
AXTB_LIBS = {"axtb CLI": "faer", "axtb CLI [MKL]": "MKL", "axtb CLI [AOCL]": "AOCL"}  # energy + gradient, default conventions
LIB_NAMES = {"faer": "faer", "MKL": "Intel MKL", "AOCL": "AMD AOCL"}
# the requirement: 1 to 8 threads, axtb at its defaults faster than the reference program at its defaults
REQ_PARTS = {"gfn2": "xtb 6.8.0", "gxtb2": "g-xTB 2.0.1 program"}
REQ_THREADS = (1, 2, 4, 8)
TIGHT = 1.05  # requirement cells below this median speed-up are flagged
# reference program, accurate setting, default setting, axtb; per method and kind
FT_PROGS = {
    ("gfn2", "E+G"): {"xtb 6.8.0": "program"},
    ("gxtb2", "E+G"): {"g-xTB 2.0.1 program": "program"},
    ("gxtb11", "E+G"): {"g-xtb 1.1.0 program": "program"},
    ("gfn2", "Hessian"): {"xtb 6.8.0 --hess --acc 0.001": "program", "xtb 6.8.0 --hess": "program_default", "axtb analytic": "axtb"},
    ("gxtb2", "Hessian"): {"g-xTB 2.0.1 program --hess --acc 0.001": "program", "g-xTB 2.0.1 program --hess": "program_default", "axtb analytic": "axtb"},
    ("gxtb11", "Hessian"): {"g-xtb 1.1.0 program -hess": "program_default", "axtb numerical": "axtb"},
}


def ratio(a, b):
    return a / b if a and b else None


def short(commit: str | None) -> str | None:
    return commit[:7] if commit else None


def eg_cells(d: dict, prog: str) -> dict:
    """Energy + gradient cells of one task file: per molecule the program's median and, per axtb library,
    axtb's median, the median speed-up and the speed-up of every round (program and axtb of the same round)."""
    med = {(s["molecule"], s["program"]): s for s in d["summary"]}
    rounds = defaultdict(dict)
    for r in d["runs"]:
        if r.get("ok"):
            rounds[(r["molecule"], r["round"])][r["program"]] = r["seconds"]
    out = {}
    for mol in dict.fromkeys(s["molecule"] for s in d["summary"]):
        p = med.get((mol, prog))
        if not p:
            continue
        c = {"atoms": p["atoms"], "program": p["median"], "libs": {}}
        for ax, lib in AXTB_LIBS.items():
            a = med.get((mol, ax))
            if not a:
                continue
            rr = [v[prog] / v[ax] for (mm, _), v in rounds.items() if mm == mol and prog in v and ax in v]
            c["libs"][lib] = {"median": a["median"], "speedup": p["median"] / a["median"], "rounds": rr,
                              "other_busy": max(p.get("median_other_busy_threads") or 0, a.get("median_other_busy_threads") or 0)}
        out[mol] = c
    return out


def final_timings() -> None:
    """Single-calculation timings against xtb 6.8.0 and the g-xTB programs on the laptop and the 64-core node
    (bench/final_timings.py): every cell with the requirement's library as `axtb` and the others alongside."""
    rows, head = [], []
    for m, cfg in FINAL.items():
        path = cfg["dir"] / "summary.json"
        if not path.exists():
            continue
        d = json.loads(path.read_text())
        commit = {k: short(t.get("axtb_commit")) for k, t in d.get("tasks", {}).items()}
        cells = defaultdict(dict)
        for r in d["summary"]:
            role = FT_PROGS.get((r["part"], r["kind"]), {}).get(r["program"])
            if role is None and r["kind"] == "E+G" and r["program"] in AXTB_LIBS:
                role = "axtb_" + AXTB_LIBS[r["program"]].lower()
            if role is None:
                continue
            k = (r["part"], r["kind"], r["molecule"], r["threads"])
            cells[k]["atoms"] = r["atoms"]
            cells[k][role] = r["median"]
            cells[k]["task"] = r.get("task")
            cells[k]["other_busy"] = max(cells[k].get("other_busy", 0), r.get("median_other_busy_threads") or 0)
        acc = d.get("hessian_accuracy", {})
        for (part, kind, mol, th), c in cells.items():
            row = {"id": f"{m}|{part}|{kind}|{mol}|{th}", "machine": m, "machine_label": cfg["label"], "method": FT_METHOD[part],
                   "kind": kind, "molecule": mol, "atoms": c["atoms"], "threads": th, "program": c.get("program")}
            if kind == "E+G":
                lib = cfg["library"] if f"axtb_{cfg['library'].lower()}" in c else "faer"
                row["library"] = LIB_NAMES[lib]
                row["axtb"] = c.get(f"axtb_{lib.lower()}")
                row["speedup"] = ratio(c.get("program"), row["axtb"])
                for low in ("faer", "mkl", "aocl"):
                    row[f"axtb_{low}"] = c.get(f"axtb_{low}")
                    row[f"speedup_{low}"] = ratio(c.get("program"), c.get(f"axtb_{low}"))
            else:
                row["library"] = "faer"
                row["program_default"] = c.get("program_default")
                row["axtb"] = c.get("axtb")
                row["speedup"] = ratio(c.get("program") or c.get("program_default"), c.get("axtb"))
                row["speedup_default"] = ratio(c.get("program_default"), c.get("axtb"))
                a = acc.get(part, {}).get(f"{mol} | {th}t", {})
                for lab, role in FT_PROGS[(part, kind)].items():
                    row[f"dfreq_{role}"] = a.get(lab, {}).get("max_dfreq")
            row["other_busy"] = round(c.get("other_busy", 0), 1)
            row["commit"] = commit.get(c.get("task"))
            rows.append(row)
        # headline numbers per machine, method, kind and thread count
        mine = [r for r in rows if r["machine"] == m]
        for meth in FT_METHOD.values():
            for kind in ("Hessian", "E+G"):
                for th in sorted({r["threads"] for r in mine if r["method"] == meth and r["kind"] == kind}):
                    cand = [r for r in mine if r["method"] == meth and r["threads"] == th and r["kind"] == kind and r["speedup"]]
                    if not cand:
                        continue
                    tx = next((r for r in cand if r["molecule"] == "taxol"), None)
                    best, worst = max(cand, key=lambda r: r["speedup"]), min(cand, key=lambda r: r["speedup"])
                    big = [r for r in cand if r["atoms"] >= 20]
                    head.append({"id": f"{m}|{meth}|{kind}|{th}", "machine": m, "machine_label": cfg["label"], "method": meth, "kind": kind,
                                 "threads": th, "taxol_speedup": tx and tx["speedup"], "taxol_speedup_default": tx and tx.get("speedup_default"),
                                 "taxol_axtb": tx and tx["axtb"], "taxol_program": tx and tx["program"],
                                 "taxol_dfreq_program": tx and tx.get("dfreq_program"), "taxol_dfreq_program_default": tx and tx.get("dfreq_program_default"),
                                 "taxol_dfreq_axtb": tx and tx.get("dfreq_axtb"),
                                 "max_speedup": best["speedup"], "max_speedup_molecule": best["molecule"],
                                 "max_speedup_20": max((r["speedup"] for r in big), default=None),
                                 "max_speedup_20_molecule": max(big, key=lambda r: r["speedup"])["molecule"] if big else None,
                                 "min_speedup": worst["speedup"], "min_speedup_molecule": worst["molecule"]})
    order = {"GFN2-xTB": 0, "g-xTB 2.0.1": 1, "g-xTB v1.1": 2}
    rows.sort(key=lambda r: (list(FINAL).index(r["machine"]), order[r["method"]], r["threads"], r["atoms"]))
    write("ft_eg", [r for r in rows if r["kind"] == "E+G"])
    write("ft_hess", [r for r in rows if r["kind"] == "Hessian"])
    write("ft_head", head)
    requirement()
    timing_runs()


def requirement() -> None:
    """The performance requirement, cell by cell: per machine, method, molecule and thread count (1 to 8), the
    median speed-up of axtb at its defaults (the machine's library) over the reference program at its defaults,
    the spread of the per-round speed-ups and a flag below TIGHT; the other libraries and the laptop's second
    session (final-laptop/session2) alongside."""
    long = []
    for m, cfg in FINAL.items():
        lib = cfg["library"]
        for part, prog in REQ_PARTS.items():
            for t in REQ_THREADS:
                f = cfg["dir"] / f"{part}-eg-t{t}.json"
                if not f.exists():
                    continue
                d = json.loads(f.read_text())
                f2 = cfg["dir"] / "session2" / f.name
                s2 = eg_cells(json.loads(f2.read_text()), prog) if f2.exists() else {}
                for mol, c in eg_cells(d, prog).items():
                    L = c["libs"].get(lib)
                    if not L:
                        continue
                    S = s2.get(mol, {}).get("libs", {}).get(lib)
                    long.append({"id": f"{m}|{part}|{mol}|{t}", "machine": m, "machine_label": cfg["label"], "method": FT_METHOD[part],
                                 "reference": prog, "molecule": mol, "atoms": c["atoms"], "threads": t, "library": LIB_NAMES[lib],
                                 "program": c["program"], "axtb": L["median"], "speedup": L["speedup"],
                                 "round_min": min(L["rounds"], default=None), "round_max": max(L["rounds"], default=None), "rounds": len(L["rounds"]),
                                 "faster": L["speedup"] > 1, "flag": L["speedup"] < TIGHT,
                                 **{f"speedup_{low.lower()}": (c["libs"].get(low) or {}).get("speedup") for low in LIB_NAMES},
                                 "s2_speedup": S and S["speedup"], "s2_round_min": S and min(S["rounds"], default=None),
                                 "other_busy": round(L["other_busy"], 1), "commit": short(d["meta"].get("axtb_commit")),
                                 "_rounds": L["rounds"], "_s2": S})
    if not long:
        return

    def summary(rs, id_, **extra):
        lo, hi = min(rs, key=lambda r: r["speedup"]), max(rs, key=lambda r: r["speedup"])
        rd = min((r for r in rs if r["round_min"] is not None), key=lambda r: r["round_min"])
        s2 = [r for r in rs if r["_s2"]]
        s2lo = min(s2, key=lambda r: r["s2_speedup"]) if s2 else None
        return {"id": id_, **extra, "cells": len(rs), "faster": sum(r["faster"] for r in rs), "flagged": sum(r["flag"] for r in rs),
                "flag_below": TIGHT, "min_speedup": lo["speedup"], "min_molecule": lo["molecule"], "min_threads": lo["threads"],
                "min_library": lo["library"], "min_machine": lo["machine_label"], "min_machine_id": lo["machine"], "min_method": lo["method"],
                "min_round_min": lo["round_min"], "min_round_max": lo["round_max"],
                "max_speedup": hi["speedup"], "max_molecule": hi["molecule"], "max_threads": hi["threads"],
                "lowest_round": rd["round_min"], "lowest_round_molecule": rd["molecule"], "lowest_round_threads": rd["threads"],
                "rounds_not_faster": sum(1 for r in rs for v in r["_rounds"] if v <= 1),
                "rounds_total": sum(len(r["_rounds"]) for r in rs),
                "s2_cells": len(s2) or None, "s2_faster": sum(r["s2_speedup"] > 1 for r in s2) if s2 else None,
                "s2_min_speedup": s2lo and s2lo["s2_speedup"], "s2_min_molecule": s2lo and s2lo["molecule"],
                "s2_min_threads": s2lo and s2lo["threads"], "s2_lowest_round": min((r["s2_round_min"] for r in s2), default=None)}
    summ = [summary(long, "all", machine="both", machine_label="both machines", method="both methods", library="—",
                    machines=len({r["machine"] for r in long}), methods=len({r["method"] for r in long}),
                    molecules=len({r["molecule"] for r in long}), thread_counts=len({r["threads"] for r in long}),
                    atoms_min=min(r["atoms"] for r in long), atoms_max=max(r["atoms"] for r in long),
                    threads_min=min(r["threads"] for r in long), threads_max=max(r["threads"] for r in long),
                    rounds=max(r["rounds"] for r in long))]
    for m, cfg in FINAL.items():
        for meth in dict.fromkeys(r["method"] for r in long):
            rs = [r for r in long if r["machine"] == m and r["method"] == meth]
            if rs:
                summ.append(summary(rs, f"{m}|{meth}", machine=m, machine_label=cfg["label"], method=meth, library=rs[0]["library"]))
    grid = {}
    for r in long:
        g = grid.setdefault((r["machine"], r["method"], r["molecule"]), {
            "id": f"{r['machine']}|{r['method']}|{r['molecule']}", "machine": r["machine"], "machine_label": r["machine_label"],
            "method": r["method"], "molecule": r["molecule"], "atoms": r["atoms"], "library": r["library"]})
        n = r["threads"]
        g[f"t{n}"], g[f"t{n}_lo"], g[f"t{n}_hi"], g[f"t{n}_flag"] = r["speedup"], r["round_min"], r["round_max"], r["flag"]
    # every axtb library measured, per machine and method: the libraries the requirement does not use, alongside
    libs = []
    for m, cfg in FINAL.items():
        for meth in dict.fromkeys(r["method"] for r in long):
            rs = [r for r in long if r["machine"] == m and r["method"] == meth]
            for low, name in LIB_NAMES.items():
                vals = [(r, r[f"speedup_{low.lower()}"]) for r in rs if r[f"speedup_{low.lower()}"] is not None]
                if not vals:
                    continue
                lo = min(vals, key=lambda x: x[1])
                below = [r for r, v in vals if v <= 1]
                libs.append({"id": f"{m}|{meth}|{name}", "machine": m, "machine_label": cfg["label"], "method": meth, "library": name,
                             "requirement": name == rs[0]["library"], "cells": len(vals), "faster": sum(v > 1 for _, v in vals),
                             "not_faster": len(below), "min_speedup": lo[1], "min_molecule": lo[0]["molecule"], "min_threads": lo[0]["threads"],
                             "max_speedup": max(v for _, v in vals),
                             "not_faster_molecules": ", ".join(dict.fromkeys(r["molecule"] for r in below)) or None,
                             "not_faster_threads": ", ".join(str(t) for t in sorted({r["threads"] for r in below})) or None})
    write("requirement", [{k: v for k, v in r.items() if not k.startswith("_")} for r in long])
    write("requirement_grid", list(grid.values()))
    write("requirement_summary", summ)
    write("library_summary", libs)


def timing_runs() -> None:
    """How the final timings ran, per machine: commit, dates, rounds, libraries and the load from other work."""
    rows = []
    for m, cfg in FINAL.items():
        sp, mp = cfg["dir"] / "summary.json", cfg["dir"] / "machine.json"
        if not sp.exists():
            continue
        d = json.loads(sp.read_text())
        mach = json.loads(mp.read_text()) if mp.exists() else {}
        tasks = d.get("tasks", {})
        eg = {k: t for k, t in tasks.items() if "-eg-" in k}
        other = {k: t for k, t in tasks.items() if "-eg-" not in k}
        busy_eg, busy_hess, task_busy = [], [], []
        for k in tasks:
            f = cfg["dir"] / f"{k}.json"
            if not f.exists():
                continue
            runs = [r["other_busy_threads"] for r in json.loads(f.read_text())["runs"] if r.get("other_busy_threads") is not None]
            (busy_eg if k in eg else busy_hess).extend(runs)
            if k in eg and runs:
                task_busy.append(statistics.median(runs))
        quota = {}
        for t in tasks.values():
            q = (t.get("pod_cpu_quota") or "").split()
            if len(q) == 2:
                quota[t["threads"]] = round(int(q[0]) / int(q[1]))
        libs = sorted({AXTB_LIBS[s["program"]] for s in d["summary"] if s["program"] in AXTB_LIBS}, key=list(LIB_NAMES).index)
        commits = sorted({short(t.get("axtb_commit")) for t in tasks.values() if t.get("axtb_commit")})
        s2p = cfg["dir"] / "session2" / "summary.json"
        s2 = json.loads(s2p.read_text()).get("tasks", {}) if s2p.exists() else {}
        hhmm = lambda ts: sorted(t["date"][11:16] for t in ts.values() if t.get("date"))
        rows.append({
            "machine": m, "label": cfg["label"], "cpu": mach.get("cpu") or MACHINES.get(m, {}).get("cpu"),
            "cores": mach.get("cores"), "hw_threads": mach.get("hw_threads_visible"),
            "commit": ", ".join(commits) or None, "date": min((t["date"][:10] for t in tasks.values() if t.get("date")), default=None),
            "eg_started": "–".join(dict.fromkeys([hhmm(eg)[0], hhmm(eg)[-1]])) + " UTC" if eg else None,
            "other_started": "–".join(dict.fromkeys([hhmm(other)[0], hhmm(other)[-1]])) + " UTC" if other else None,
            "rounds_eg": max((t["rounds_eg"] for k, t in eg.items()), default=None),
            "rounds_hess": max((t["rounds_hess"] for k, t in other.items() if not k.startswith("gxtb11")), default=None),
            "rounds_v11": max((t["rounds_eg"] for k, t in other.items() if k.startswith("gxtb11")), default=None),
            "library": LIB_NAMES[cfg["library"]], "alongside": ", ".join(LIB_NAMES[x] for x in libs if x != cfg["library"]) or None,
            "other_busy_median": statistics.median(busy_eg) if busy_eg else None,
            "other_busy_task_min": min(task_busy, default=None), "other_busy_task_max": max(task_busy, default=None),
            "other_busy_max": max(busy_eg, default=None),
            "other_busy_hess_median": statistics.median(busy_hess) if busy_hess else None,
            "load1_min": min((min(t["load1_start"], t["load1_end"]) for t in eg.values()), default=None),
            "load1_start_max": max((t["load1_start"] for t in tasks.values()), default=None),
            "load1_max": max((max(t["load1_start"], t["load1_end"]) for t in eg.values()), default=None),
            "pod_cpus_1t": quota.get(1), "pod_cpus_8t": quota.get(8),
            "s2_eg_started": "–".join(dict.fromkeys([hhmm(s2)[0], hhmm(s2)[-1]])) + " UTC" if s2 else None,
            "note": mach.get("note_session"),
        })
    write("ft_runs", rows)


PARITY_PROGRAMS = {"gxtb201": ("g-xTB 2.0.1", "the g-xTB 2.0.1 program", "gxtb"), "gxtb11": ("g-xTB v1.1", "g-xtb 1.1.0", "gxtb"),
                   "xtb680": ("GFN2-xTB", "xtb 6.8.0", "xtb"), "xtb671": ("GFN2-xTB", "xtb 6.7.1", "xtb-6.7.1"),
                   "tblite680": ("GFN2-xTB", "xtb 6.8.0 --tblite", "tblite"), "tblite671": ("GFN2-xTB", "xtb 6.7.1 --tblite", "tblite")}
METHOD_LABEL = {"gfn2": "GFN2-xTB", "gxtb-v2.0.1": "g-xTB 2.0.1", "gxtb-v1.1": "g-xTB v1.1"}


def parity_features() -> None:
    """SCF guess reuse, bond orders, HOMO-LUMO gaps and quadrupoles against the programs (tests/gxtb_parity_features.py)."""
    path = ROOT / "bench/results/gxtb_parity_features.json"
    if not path.exists():
        return
    d = json.loads(path.read_text())
    rows = []
    for r in d.get("sequences", []):
        rows.append({"id": f"{r['method']}|{r['molecule']}|{r['sequence']}", "method": METHOD_LABEL.get(r["method"], r["method"]),
                     "sequence": f"{r['molecule']}: {r['sequence']}", "points": r["points"],
                     "iter_fresh": r["iterations_per_point_fresh"], "iter_reuse": r["iterations_per_point_reuse"],
                     "time_fresh": r["seconds_fresh"], "time_reuse": r["seconds_reuse"]})
    for r in d.get("codec", []):
        if r.get("propagate_wfn"):
            fresh = next((q for q in d["codec"] if not q.get("propagate_wfn") and q["driver"] == r["driver"] and q["method"] == r["method"]), None)
            if fresh:
                rows.append({"id": f"{r['method']}|{r['driver']}", "method": METHOD_LABEL.get(r["method"], r["method"]),
                             "sequence": f"{r['molecule']}: {r['driver']} through qccompute ({fresh['frames']} / {r['frames']} steps)", "points": r["frames"],
                             "iter_fresh": fresh["scf_iterations_per_frame"], "iter_reuse": r["scf_iterations_per_frame"],
                             "time_fresh": None, "time_reuse": None})
    order = ["GFN2-xTB", "g-xTB 2.0.1", "g-xTB v1.1"]
    write("guess_reuse", sorted(rows, key=lambda r: (order.index(r["method"]) if r["method"] in order else 9, r["sequence"])))
    agg = {}
    for r in d.get("properties", {}).get("rows", []):
        meth, prog, key = PARITY_PROGRAMS.get(r["program"], (r["program"], r["program"], None))
        v = r.get(key) or {}
        a = agg.setdefault(r["program"], {"id": r["program"], "method": meth, "program": prog, "molecules": 0,
                                          "bond_orders": None, "gap_eV": None, "quadrupole": None, "quadrupole_printed": None})
        a["molecules"] += 1
        for k, f in (("bond_orders", "bond_order_max_abs_diff"), ("gap_eV", "gap_abs_diff_eV"), ("quadrupole", "quadrupole_max_abs_diff")):
            if v.get(f) is not None:
                a[k] = v[f] if a[k] is None else max(a[k], v[f])
        if v.get("quadrupole_print_precision") is not None:
            a["quadrupole_printed"] = v["quadrupole_print_precision"]
    # xtb --tblite rows only document tblite's quadrupole form (see issues); the table compares each method with its own program
    write("parity_props", [agg[k] for k in PARITY_PROGRAMS if k in agg and not k.startswith("tblite")])


GXTB_STATUS = {"ok": "agrees", "rounding-sensitive": "rounding-sensitive", "program-gradient": "program's gradient wrong",
               "real-difference": "small real difference", "unparametrized": "no parameters in the program"}


def gxtb_elements() -> None:
    """g-xTB v1.1 and 2.0.1 against their programs for every element H-Lr (tests/gxtb_elements.py)."""
    path = ROOT / "bench/results/gxtb_elements.json"
    if not path.exists():
        return
    d = json.loads(path.read_text())
    summ = []
    for v, sm in sorted(d["summary"].items(), key=lambda kv: kv[0] != "2.0.1"):
        st = sm["status"]
        summ.append({"version": f"g-xTB {v}", "molecules": sm["molecules"], "elements": sm["elements"],
                     "agree": st.get("ok", 0), "rounding": st.get("rounding-sensitive", 0),
                     "program_wrong": st.get("program-gradient", 0), "real": st.get("real-difference", 0),
                     "unparametrized": st.get("unparametrized", 0), "max_dE_ok": sm["max_abs_dE_ok"],
                     "max_dG": sm.get("max_dG", sm.get("max_dG_fd_gxtb")),
                     "gradient_against": "the program's analytic gradient" if v == "2.0.1" else "finite differences of tight energies"})
    write("gxtb_elements_summary", summ)
    rows = []
    for r in d["table"]:
        it = r.get("iterations") or [None, None]
        rows.append({"id": f"{r['version']}|{r['Z']}|{r['molecule']}", "version": f"g-xTB {r['version']}", "Z": r["Z"], "element": r["element"],
                     "molecule": r["molecule"], "charge": r["charge"], "unpaired": r["uhf"], "dE": r.get("dE"), "dG": r.get("dG"),
                     "iterations": None if it[0] is None else f"{it[0]} / {it[1]}", "status": GXTB_STATUS.get(r["status"], r["status"]),
                     "note": r.get("note") or None})
    write("gxtb_elements", sorted(rows, key=lambda r: (r["version"] != "g-xTB 2.0.1", r["Z"], r["molecule"])))


def gxtb_energy_differences() -> None:
    """Energy differences under program vs default conventions (tests/gxtb_energy_differences.py)."""
    path = ROOT / "bench/results/gxtb_energy_differences.json"
    if not path.exists():
        return
    d = json.loads(path.read_text())["versions"]
    v1 = {r["quantity"]: r for r in d.get("v1.1", {}).get("rows", [])}
    v2 = {r["quantity"]: r for r in d.get("v2.0.1", {}).get("rows", [])}
    rows = []
    for q in list(v2) + [q for q in v1 if q not in v2]:
        a, b = v2.get(q, {}), v1.get(q, {})
        rows.append({"quantity": q, "v2_kcal": a.get("program_conventions_kcal"), "v2_delta": a.get("default_minus_program_Eh"),
                     "v1_kcal": b.get("program_conventions_kcal"), "v1_delta": b.get("default_minus_program_Eh")})
    write("gxtb_ediff", rows)


def methods() -> None:
    """One row per method axtb implements, from the other datasets (run after them)."""
    def load(n):
        f = OUT / f"{n}.json"
        return json.loads(f.read_text()) if f.exists() else []
    m = load("matrix")
    cases = sum({r["id"].split("|")[0]: r["category_cases"] for r in m}.values())
    worst = lambda c: max((r["max_dE"] for r in m if r["check"] == c and r["max_dE"] is not None), default=None)
    g2, g1, ff = load("gxtb2"), load("gxtb"), load("gfnff_accuracy")
    es = {r["version"]: r for r in load("gxtb_elements_summary")}
    rows = [
        {"method": "GFN2-xTB", "reproduces": "xtb 6.8.0 (and 6.7.1), tblite 0.8.0",
         "energy": "yes", "gradient": "analytic", "hessian": "analytic",
         "extras": "ALPB/GBSA solvation, electronic temperature, SCF restarts, bond orders, gap and quadrupole, vibrational analysis, Molden",
         "validation": f"{cases}-case test matrix; worst passing ΔE {worst('axtb (xtb mode) vs xtb'):.1e} Eh vs xtb, {worst('axtb (tblite mode) vs tblite'):.1e} Eh vs tblite"},
    ]
    if g2:
        rows.append({"method": "g-xTB 2.0.1", "reproduces": "the g-xTB 2.0.1 program",
                     "energy": "yes", "gradient": "analytic", "hessian": "analytic",
                     "extras": "electronic temperature, solvation correction from GFN2-xTB/ALPB, SCF restarts, bond orders, gap and quadrupole, vibrational analysis, Molden",
                     "validation": f"{len(g2)} reference molecules, worst ΔE {max(abs(r['dE']) for r in g2 if r.get('dE') is not None):.1e} Eh with the same SCF iterations"
                                   + (f"; every element H–Lr: {es['g-xTB 2.0.1']['agree']} of {es['g-xTB 2.0.1']['molecules']} molecules agree, the rest rounding-sensitive or a program gradient error" if "g-xTB 2.0.1" in es else "")})
    if g1:
        rows.append({"method": "g-xTB v1.1", "reproduces": "the g-xtb 1.1.0 program",
                     "energy": "yes", "gradient": "analytic", "hessian": "numerical",
                     "extras": "electronic temperature, SCF restarts, bond orders and gap, Molden",
                     "validation": f"{len(g1)} reference molecules, worst ΔE {max(abs(r['dE']) for r in g1 if r.get('dE') is not None):.1e} Eh"
                                   + (f"; every element H–Lr: {es['g-xTB 1.1']['agree']} of {es['g-xTB 1.1']['molecules'] - es['g-xTB 1.1']['unparametrized']} parametrized molecules agree" if "g-xTB 1.1" in es else "")})
    if ff:
        n = sum(r["cases"] for r in ff); ok = sum(r["reproduced"] for r in ff)
        rows.append({"method": "GFN-FF", "reproduces": "xtb 6.8.0 and 6.7.1 (--gfnff)",
                     "energy": "yes", "gradient": "analytic", "hessian": "analytic",
                     "extras": "fixed or reference topology for whole paths, vibrational analysis",
                     "validation": f"{ok} of {n} structures; worst ΔE {max(r['max_dE'] for r in ff):.1e} Eh"})
    write("methods", rows)


def xtb680() -> None:
    """What xtb 6.8.0 changed for GFN2-xTB (tests/xtb680_changes.py) and GFN-FF in 6.8.0 vs axtb's 6.7.1 port."""
    path = ROOT / "bench/results/xtb680_changes.json"
    if path.exists():
        d = json.loads(path.read_text())
        s = d["summary"]
        tight, dflt = s.get("acc 0.0001", {}), s.get("acc 1", {})
        ncases = len(d["cases"]) // max(1, len(s))
        rows = []
        for k in tight:
            a, b = tight[k], dflt.get(k, {})
            rows.append({"id": k, "setting": "axtb's xtb 6.8.0 conventions, as they are" if k == "xtb-6.8.0" else k,
                         "dE_tight": a["max_abs_dE"], "dG_tight": a["max_dG"],
                         "dE_default": b.get("max_abs_dE"), "dG_default": b.get("max_dG"),
                         "iterations_differ": max(a.get("iterations_differ", 0), b.get("iterations_differ", 0)), "cases": ncases})
        write("xtb680_changes", rows)


GFNFF_CATS = {"elements": "Elements H–Rn", "charges": "Charges", "multielement": "Mixed-element molecules", "open_shell_metals": "Metal complexes",
              "structures": "Test structures", "hb-xb-dimers": "Hydrogen- and halogen-bonded dimers", "water": "Water cluster",
              "zwitterions": "Zwitterion in water", "large": "Large (alkane, crambin)"}


def gfnff_name(n: str) -> str:
    if n.startswith("water") and n[5:].isdigit():
        return f"(H2O){n[5:]}"
    return {"crambin-water": "crambin in water"}.get(n, n)


def gfnff() -> None:
    """GFN-FF in axtb vs xtb --gfnff (tests/gfnff_reproduction.py): the test set and Hessians against xtb 6.8.0
    (what axtb's GFN-FF follows), both versions summarized, scaling against 6.7.1 (where it was measured)."""
    path = ROOT / "bench/results/gfnff_reproduction.json"
    path680 = ROOT / "bench/results/gfnff_reproduction_6.8.0.json"
    if not path.exists():
        return
    d671 = json.loads(path.read_text())
    d = json.loads(path680.read_text()) if path680.exists() else d671
    vers = []
    for lab, dd in (("xtb 6.8.0 (axtb default and xtb conventions)", d if path680.exists() else None), ("xtb 6.7.1, the g-xTB 2.0.1 build (conventions xtb-6.7.1)", d671)):
        if dd is None:
            continue
        sm = dd["accuracy"]["summary"]
        vers.append({"version": lab, "structures": sm["cases"], "reproduced": sm["reproduced"], "max_dq": sm["max_charges_diff"],
                     "max_dE": max(sm["max_energy_diff"], sm["max_term_diff"]), "max_dG": sm["max_gradient_diff"],
                     "not_reproduced": ", ".join(k.split("-", 1)[1] for k in sm.get("not_reproduced", {})) or None})
    write("gfnff_versions", vers)
    by = defaultdict(list)
    for c in d["accuracy"]["cases"]:
        by[c["category"]].append(c)
    acc = []
    for cat, cs in by.items():
        ok = [c for c in cs if c["pass"]]
        acc.append({"category": GFNFF_CATS.get(cat, cat), "cases": len(cs), "reproduced": len(ok), "largest": max(c["natoms"] for c in cs),
                    "max_dq": max(c["charges_maxdiff"] for c in ok), "max_dE": max(max(abs(c["energy_diff"]), c["terms_maxdiff"]) for c in ok),
                    "max_dG": max(c["gradient_maxdiff"] for c in ok),
                    "not_reproduced": ", ".join(c["id"].split("-", 1)[1] for c in cs if not c["pass"]) or None})
    order = list(GFNFF_CATS)
    write("gfnff_accuracy", sorted(acc, key=lambda r: next((i for i, k in enumerate(order) if GFNFF_CATS[k] == r["category"]), 99)))
    hs = []
    for c in d["hessian"]["cases"]:
        n, a = c.get("numerical"), c.get("analytic")
        names = {"struct-caffeine": "caffeine", "dimer-water": "water dimer", "dimer-pyridine-HF": "pyridine–HF", "dimer-acetone-water": "acetone–water (H on the C=O axis)",
                 "lin-CO2": "CO2 (linear)", "lin-HCN": "HCN (linear)",
                 "xb-ICl-NH3": "ICl–NH3 (halogen bond)", "glycine-zwitterion-water": "glycine zwitterion with water", "struct-cu-nh3-4": "[Cu(NH3)4]2+",
                 "water-cluster-20": "(H2O)20"}
        ok = a is not None and "error" not in a
        hs.append({"case": names.get(c["id"], c["id"]), "atoms": c["natoms"],
                   "num_dfreq": n and n["frequencies_maxdiff_cm1"], "num_dG": n and abs(n.get("free_energy_diff", n.get("g_rrho_diff", 0))),
                   "ana_dfreq": a["frequencies_maxdiff_cm1"] if ok else None, "ana_dG": abs(a["free_energy_diff"]) if ok else None,
                   "note": "xtb 6.7.1 --hess crashes (segmentation fault)" if "xtb_error" in c else
                           ("energy not twice differentiable here (xtb#1516): axtb's analytic Hessian is not finite; numerical reproduces xtb" if a and not ok else None)})
    write("gfnff_hessian", hs)
    sc = []
    for x in d671["scaling"]["systems"]:
        g, g8, xg = x.get("axtb_grad", {}), x.get("axtb_grad_t8", {}), x.get("xtb_grad", {})
        h1, h8, xh = x.get("axtb_hess_t1"), x.get("axtb_hess_t8"), x.get("xtb_hess")
        sc.append({"id": x["name"], "system": gfnff_name(x["name"]), "atoms": x["natoms"],
                   "xtb_s": xg.get("wall_s") if xg.get("ok") else None, "xtb_mib": xg.get("peak_rss_mib") if xg.get("ok") else None,
                   "xtb_failed": None if xg.get("ok") else "fails: out of memory",
                   "xtb_hb_array_gib": x.get("xtb_considered_ABH_gib"),
                   "axtb_s": g.get("wall_s"), "axtb_s8": g8.get("wall_s"), "axtb_mib": g.get("peak_rss_mib"),
                   "hess_s": h1 and h1["wall_s"], "hess_s8": h8 and h8["wall_s"], "hess_mib": h1 and h1["peak_rss_mib"],
                   "xtb_hess_s": xh and xh["wall_s"], "dE": x.get("energy_diff"), "dG": x.get("gradient_maxdiff"),
                   # 1-minute load before and after each of the system's runs
                   **dict(zip(("load1_min", "load1_max"), (lambda ls: (min(ls), max(ls)) if ls else (None, None))(
                       [v[k][0] for v in x.values() if isinstance(v, dict) for k in ("load_before", "load_after") if v.get(k)])))})
    write("gfnff_scaling", sc)


def main() -> None:
    res = suites()
    machine_rows(res)
    accuracy(res)
    xtb_tblite(res)
    speed(res)
    linalg()
    matrix()
    conventions()
    libraries()
    ev = program_diffs()
    gpu()
    hessian_analytic()
    gxtb()
    gxtb2()
    gxtb2_hessian()
    scf_hard()
    gxtb2_hess_memory()
    gxtb_etemp()
    vibrational()
    molden()
    composite()
    gfnff()
    xtb680()
    gxtb_elements()
    gxtb_energy_differences()
    parity_features()
    final_timings()
    phases()
    cli_fix()
    issues(ev)
    learnings(res, ev)
    scope()
    methods()


if __name__ == "__main__":
    main()
