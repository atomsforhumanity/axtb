#!/usr/bin/env python3
"""GFN-FF in axtb vs xtb 6.8.0 or 6.7.1 (`xtb --gfnff`): accuracy, Hessians, and scaling of
time and memory.

axtb runs in the conventions of the xtb binary given with `--xtb` (default: xtb 6.8.0 from
`$AXTB_XTB680`, see tests/programs.py; the 6.7.1 record ran `--xtb "$AXTB_GXTB2"`) (`xtb` for 6.8.0,
`xtb-6.7.1` for 6.7.x, e.g. the g-xTB 2.0.1 build; `--conventions` overrides), and the
results go to `bench/results/gfnff_reproduction.json` (6.7.1) or
`bench/results/gfnff_reproduction_6.8.0.json` (6.8.0) unless `--out` is given.

`--part accuracy`: every structure of the GFN-FF test set (`tests/gfnff_structures.py`:
organics, charged species -3..+3, zwitterions, metal complexes, open-shell metals, elements
H-Rn, HB/XB dimers, water clusters, crambin with hydrogens) through `xtb coord --gfnff --grad
--json` (one thread per run) and through axtb: the topology (`examples/gfnff_dump`, every
list of xtb's `gfnff_topo` file: neighbors, bonds, angles, torsions, bonded ATM triples,
fragments, HB/XB lists, A-H...B bonds, bond distances; integers exactly, parameters as max
|difference|), and the axtb CLI (`--method gfnff`, JSON request in bohr): EEQ charges, total
and per-term energies, gradient, and the bond list of its `topology` output.

`--part hessian`: `xtb --gfnff --hess` against the CLI's Hessians in xtb's conventions
(numerical, step 0.005 as xtb, and analytic): frequencies, IR intensities, ZPVE and the
total free energy (xtb scales GFN-FF frequencies by 1.03 in its thermochemistry). xtb 6.7.1
rotates the molecule slightly before its numerical Hessian and projects Cartesian
coordinates; 6.8.0 does neither (mass-weighted projection), which the conventions of the
analysis reproduce. `--part topology`: the structures evaluated with the topology of another
geometry (axtb's `topology_positions`, xtb reading the other geometry's `gfnff_topo`).

`--part scaling`: water boxes (liquid density) and crambin (PDB 1CRN, 640 atoms) without and
with a water box, 300 to ~10000 atoms. For each structure, one at a time:

- `xtb coord --gfnff --grad` (one thread, unlimited stack, address space capped at
  `--xtb-mem` GiB so that xtb's dense N^2 / N^3 work arrays fail instead of swapping);
- the axtb GFN-FF energy + analytic gradient (`examples/gfnff_bench`, one thread) and,
  up to `--hess-max` atoms, the analytic Hessian with one and `--threads` threads;
- `xtb --gfnff --hess` (finite differences of gradients) up to `--xtb-hess-max` atoms.

Wall time and peak resident memory come from `/usr/bin/time -v`. Where both programs
finish, the energy and gradient differences are recorded as well.

    cargo build --release -p axtb-gfnff --examples && cargo build --release -p axtb-cli
    python tests/gfnff_reproduction.py [--part accuracy|hessian|topology|scaling|all] [--xtb XTB] [--out OUT]

The output JSON keeps the other parts of an existing file. The accuracy, hessian and topology
parts fail (exit status 1) when a quantity exceeds its limit in `LIMITS` (tests/gate.py) or a
structure is not reproduced, except the known differences; with the conventions of another xtb
version (`--conventions`) the results are a measurement and not gated.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
import gfnff_structures as GS  # noqa: E402
import gfnff_xtb as GX  # noqa: E402
import gate  # noqa: E402
import programs  # noqa: E402

BENCH = ROOT / "target" / "release" / "examples" / "gfnff_bench"
DUMP = ROOT / "target" / "release" / "examples" / "gfnff_dump"
CLI = ROOT / "target" / "release" / "axtb"
# per part: the one structure of the set whose topology differs from xtb's, and the one whose
# analytic Hessian is undefined (docs/validation.md, GFN-FF)
KNOWN_DIFFERENCES = {
    "accuracy": {
        "tm-ferrocenium": "D5h ferrocenium: the Hueckel pi system of each (antiaromatic, cationic) Cp ring has an "
                          "exactly degenerate pair at the Fermi level, filled with integer occupations; its bond orders "
                          "depend on which eigenvectors the eigensolver returns for the pair (xtb: MKL dsyev), so pi bond "
                          "orders and the bond parameters built from them differ (xtb's are not rotation invariant either)",
    },
    "hessian": {
        "dimer-acetone-water": "water H exactly on the C=O axis (grimme-lab/xtb#1516): the HB term's torsions are "
                               "undefined there, so axtb refuses the analytic Hessian (the numerical one reproduces "
                               "xtb 6.8.0's), and xtb 6.7.1's --hess crashes",
    },
}
# The gate (tests/gate.py), per conventions (xtb version) and part; docs/validation.md (GFN-FF and
# Summary) unless "chosen". Hessians: numerical = axtb's numerical Hessian (xtb's step), analytic,
# program = xtb's own `hessian` file through axtb's analysis. Chosen: analytic IR, ZPVE and free
# energy (xtb's finite-difference error, measured 30 km/mol, 3.7e-6 and 3.2e-6 Eh) about 2x;
# program-Hessian frequencies at xtb's 2 printed decimals, its ZPVE and G(RRHO) (the `hessian`
# file's 10 decimals; measured 6.3e-11 and 7.0e-10 Eh with 6.8.0, 1.2e-11 and 1.2e-10 with 6.7.1)
# about 10x; topology_positions charges and gradients (round-off, measured <= 3.0e-14 e and
# 7.6e-15 Eh/bohr) about 10x.
LIMITS = {
    "xtb": {
        "accuracy": {"not reproduced": 0, "max_topology_parameter_diff": 5.0e-11, "max_charges_diff": 2.3e-12,
                     "max_energy_diff": 7.0e-13, "max_term_diff": 8.4e-13, "max_gradient_diff": 4.3e-12},
        "hessian": {"failures": 0, "numerical frequencies_maxdiff_cm1": 0.005, "numerical ir_maxdiff_km_mol": 5e-6,
                    "numerical zpve_diff": 4.8e-13, "numerical free_energy_diff": 4.8e-13,
                    "analytic frequencies_maxdiff_cm1": 0.27, "analytic ir_maxdiff_km_mol": 60.0,
                    "analytic zpve_diff": 8e-6, "analytic free_energy_diff": 7e-6,
                    "program_hessian frequencies_maxdiff_cm1": 0.005, "program_hessian zpve_diff": 1e-9,
                    "program_hessian g_rrho_diff": 1e-8},
        "topology": {"not reproduced": 0, "max_energy_diff": 5.0e-13, "max_charges_diff": 3e-13,
                     "max_gradient_diff": 1e-13},
    },
    "xtb-6.7.1": {
        "accuracy": {"not reproduced": 0, "max_topology_parameter_diff": 5.0e-11, "max_charges_diff": 4.7e-12,
                     "max_energy_diff": 8.4e-13, "max_term_diff": 5.5e-13, "max_gradient_diff": 9.3e-12},
        "hessian": {"failures": 0, "numerical frequencies_maxdiff_cm1": 0.005, "numerical ir_maxdiff_km_mol": 5e-6,
                    "numerical zpve_diff": 2e-11, "numerical free_energy_diff": 2e-11,
                    "analytic frequencies_maxdiff_cm1": 0.3, "analytic ir_maxdiff_km_mol": 60.0,
                    "analytic zpve_diff": 8e-6, "analytic free_energy_diff": 5e-6,
                    "program_hessian frequencies_maxdiff_cm1": 0.005, "program_hessian zpve_diff": 2e-10,
                    "program_hessian g_rrho_diff": 2e-9},
        "topology": {"not reproduced": 0, "max_energy_diff": 4.9e-13, "max_charges_diff": 3e-13,
                     "max_gradient_diff": 1e-13},
    },
}
TOL = {"charges": 1e-10, "energy": 1e-10, "gradient": 1e-9}
HESS_CASES = ["struct-caffeine", "dimer-water", "dimer-pyridine-HF", "dimer-acetone-water", "xb-ICl-NH3",
              "glycine-zwitterion-water", "struct-cu-nh3-4", "water-cluster-20"]
# linear molecules (xtb 6.8.0 projects them; 6.7.1 does not, so they are run against 6.8.0 only)
HESS_LINEAR = ["lin-CO2", "lin-HCN"]
# evaluated with the topology of another geometry (`topology_positions`)
TOPO_CASES = ["struct-caffeine", "glycine-zwitterion-water", "struct-cu-nh3-4", "dimer-pyridine-HF", "struct-ferrocene",
              "xb-ICl-NH3"]


def cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def xtb_version(xtb: str) -> str | None:
    try:
        out = subprocess.run([xtb, "--version"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return None
    m = re.search(r"xtb version (\S+) \((\w+)\)", out)
    return f"{m.group(1)} ({m.group(2)})" if m else None


def timed(cmd, *, cwd=None, stdin=None, env=None, mem_gib=None, timeout=3600) -> dict:
    """Run `cmd` under /usr/bin/time -v; wall time (s), peak RSS (MiB), output."""

    def limits():
        GX._unlimited_stack()
        if mem_gib:
            cap = int(mem_gib * 2**30)
            resource.setrlimit(resource.RLIMIT_AS, (cap, cap))

    load = os.getloadavg()
    t = time.perf_counter()
    try:
        p = subprocess.run(["/usr/bin/time", "-v", *cmd], cwd=cwd, input=stdin, env=env, capture_output=True,
                           text=True, timeout=timeout, preexec_fn=limits)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout after {timeout} s", "wall_s": timeout}
    wall = time.perf_counter() - t
    def field(name):
        m = re.search(rf"{re.escape(name)}: ([\d.]+)", p.stderr)
        return float(m.group(1)) if m else None

    rss = field("Maximum resident set size (kbytes)")
    out = {"ok": p.returncode == 0, "wall_s": round(wall, 3), "peak_rss_mib": round(rss / 1024, 1) if rss else None,
           "cpu_s": round((field("User time (seconds)") or 0) + (field("System time (seconds)") or 0), 2),
           "load_before": [round(x, 2) for x in load], "load_after": [round(x, 2) for x in os.getloadavg()],
           "stdout": p.stdout}
    if p.returncode != 0:
        text = p.stderr.split("\tCommand being timed")[0]
        forrtl = [l.strip() for l in text.splitlines() if "forrtl" in l]
        tail = forrtl or text.strip().splitlines()[-3:]
        out["error"] = f"exit status {p.returncode}: " + " | ".join(tail)
    return out


def run_xtb(xtb, numbers, positions, charge, hess, mem_gib, timeout, workdir):
    tmp = Path(tempfile.mkdtemp(prefix="gfnff_scaling_", dir=workdir))
    try:
        GX.write_coord(tmp / "coord", numbers, positions)
        env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OMP_STACKSIZE": "1G"}
        r = timed([xtb, "coord", "--gfnff", "--chrg", str(int(charge)), "--hess" if hess else "--grad"], cwd=tmp,
                  env=env, mem_gib=mem_gib, timeout=timeout)
        stdout = r.pop("stdout", "")
        if r["ok"]:
            terms, _, _ = GX.parse_stdout(stdout)
            if "total" not in terms:
                r.update(ok=False, error="no total energy in the output")
            else:
                r["energy"] = terms["total"]
                if not hess and (tmp / "gradient").exists():
                    r["gradient"] = GX.parse_gradient(tmp / "gradient", len(numbers))
        elif "error" in r and ("allocat" in stdout.lower() or "memory" in stdout.lower()):
            r["error"] += " | " + " ".join(stdout.strip().splitlines()[-2:])
        return r
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def run_axtb(numbers, positions, charge, mode, threads, timeout, conventions=None):
    req = {"numbers": [int(z) for z in numbers], "positions": np.asarray(positions).tolist(),
           "charge": charge, "threads": threads, "mode": mode}
    if conventions:  # the force field of the xtb compared with (gfnff_bench's default is 6.8.0's)
        req["conventions"] = conventions
    req = json.dumps(req)
    r = timed([str(BENCH)], stdin=req, timeout=timeout)
    stdout = r.pop("stdout", "")
    if r["ok"]:
        out = json.loads(stdout)
        r.update(energy=out["energy"], setup_s=round(out["setup_s"], 3), eval_s=round(out["eval_s"], 3),
                 timings={k: round(v, 3) for k, v in out["timings"]})
        if "gradient" in out:
            r["gradient"] = out["gradient"]
        r["xtb_hb_map"] = out.get("xtb_hb_map")
        if mode == "hess":
            r["max_asymmetry"] = out["max_asymmetry"]
    return r


def composition(numbers) -> dict:
    z = np.asarray(numbers)
    return {GX.SYMBOLS[k - 1]: int(np.sum(z == k)) for k in sorted(set(z.tolist()))}


def systems(max_atoms: int):
    out = [("water100", *GS.water_box(100, seed=0), 0)]
    nums, pos, _ = GX.read_xyz(ROOT / "tests" / "structures" / "large" / "crambin_h.xyz")
    out.append(("crambin", nums, pos, 2))
    for n in (333, 1000, 1700, 3300):
        out.append((f"water{n}", *GS.water_box(n, seed=0), 0))
    out.append(("crambin-water", *GS.solvated_crambin(pad=5.0)))
    out.sort(key=lambda s: len(s[1]))
    return [s for s in out if len(s[1]) <= max_atoms]


def scaling(args) -> dict:
    res = {"cpu": cpu_model(), "logical_cpus": os.cpu_count(), "xtb_version": xtb_version(args.xtb),
           "xtb_address_space_cap_gib": args.xtb_mem, "axtb_threads": [1, args.threads],
           "notes": "one run at a time; xtb with OMP_NUM_THREADS=1 and unlimited stack; axtb energy+gradient "
                    "with one thread; peak RSS, user+system CPU time from /usr/bin/time -v; load_before/after: "
                    "1/5/15-minute load averages of the machine around each run (other work was running)",
           "systems": []}
    for name, numbers, positions, charge in systems(args.max_atoms):
        n = len(numbers)
        entry = {"name": name, "natoms": n, "charge": charge, "composition": composition(numbers)}
        print(f"{name} ({n} atoms)", flush=True)
        ax = run_axtb(numbers, positions, charge, "grad", 1, args.timeout, args.conventions)
        entry["axtb_grad"] = {k: v for k, v in ax.items() if k not in ("gradient", "xtb_hb_map")}
        if ax.get("xtb_hb_map"):
            # xtb allocates considered_ABH(NAB, NAB, NH) as 4-byte logicals in every energy
            # evaluation (gfnff_eg.f90), NAB/NH the non-H/H atoms of its A-H...B bond list
            m = ax["xtb_hb_map"]
            entry["xtb_considered_ABH_gib"] = round(4 * m["nab"] ** 2 * m["nh"] / 2**30, 2)
            entry["xtb_hb_map"] = m
        print("  axtb grad", entry["axtb_grad"].get("wall_s"), "s", entry["axtb_grad"].get("peak_rss_mib"), "MiB",
              flush=True)
        if args.threads > 1:
            a8 = run_axtb(numbers, positions, charge, "grad", args.threads, args.timeout, args.conventions)
            entry[f"axtb_grad_t{args.threads}"] = {k: v for k, v in a8.items() if k not in ("gradient", "xtb_hb_map")}
            print(f"  axtb grad t{args.threads}", a8.get("wall_s"), "s", a8.get("peak_rss_mib"), "MiB", flush=True)
        if not args.skip_xtb:
            xt = run_xtb(args.xtb, numbers, positions, charge, False, args.xtb_mem, args.timeout, args.workdir)
            entry["xtb_grad"] = {k: v for k, v in xt.items() if k != "gradient"}
            print("  xtb grad", xt.get("wall_s"), "s", xt.get("peak_rss_mib"), "MiB", xt.get("error", ""), flush=True)
            if xt["ok"] and ax["ok"]:
                entry["energy_diff"] = abs(xt["energy"] - ax["energy"])
                entry["gradient_maxdiff"] = float(np.max(np.abs(np.array(xt["gradient"]) - np.array(ax["gradient"]))))
        if n <= args.hess_max:
            for th in sorted({1, args.threads}):
                h = run_axtb(numbers, positions, charge, "hess", th, args.timeout, args.conventions)
                entry[f"axtb_hess_t{th}"] = {k: v for k, v in h.items() if k not in ("gradient", "xtb_hb_map")}
                print(f"  axtb hess t{th}", h.get("wall_s"), "s", h.get("peak_rss_mib"), "MiB", flush=True)
        if not args.skip_xtb and n <= args.xtb_hess_max:
            xh = run_xtb(args.xtb, numbers, positions, charge, True, args.xtb_mem, args.xtb_hess_timeout, args.workdir)
            entry["xtb_hess"] = {k: v for k, v in xh.items() if k != "gradient"}
            print("  xtb hess", xh.get("wall_s"), "s", xh.get("peak_rss_mib"), "MiB", xh.get("error", ""), flush=True)
        res["systems"].append(entry)
    return res


def cli(req: dict) -> dict:
    p = subprocess.run([str(CLI), "--json"], input=json.dumps(req), capture_output=True, text=True)
    if p.returncode != 0:
        try:  # the CLI's JSON error
            return {"error": json.loads(p.stdout)["error"]["message"][-500:]}
        except (ValueError, KeyError, TypeError):
            return {"error": (p.stdout + p.stderr).strip()[-500:]}
    return json.loads(p.stdout)


def request(numbers, positions, charge, **extra) -> dict:
    return {"numbers": [int(z) for z in numbers], "positions": np.asarray(positions, float).tolist(),
            "charge": float(charge), "method": "gfnff", **extra}


def accuracy_case(case, xtb: str, workdir, conventions: str) -> dict:
    cid, cat, numbers, positions, charge = case
    out = {"id": cid, "category": cat, "natoms": len(numbers), "charge": charge}
    ref = GX.run_xtb(numbers, positions, charge, workdir=workdir, xtb=xtb)
    if "error" in ref:
        out["xtb_error"] = ref["error"][-300:]
        return out
    try:
        dump = GX.axtb_dump(numbers, positions, charge, exe=str(DUMP), conventions=conventions)
    except RuntimeError as e:
        out["axtb_error"] = str(e)[-300:]
        return out
    ct = GX.compare_topology(ref, dump)
    out["topology_lists_differ"] = sorted(k for k, v in ct.items() if v is False)
    out["topology_parameters_maxdiff"] = max(v for v in ct.values() if not isinstance(v, bool))
    r = cli(request(numbers, positions, charge, gradient=True, conventions=conventions))
    if "error" in r:
        out["axtb_error"] = r["error"]
        return out
    blist = sorted(tuple(sorted((i - 1, j - 1))) for i, j in ref["topology"]["blist"])
    out["cli_bonds_equal"] = sorted(tuple(sorted(b)) for b in r["topology"]["bonds"]) == blist
    terms = GX.compare_terms(ref, r)
    out["energy_diff"] = terms.pop("total")
    out["terms_maxdiff"] = max(terms.values()) if terms else 0.0
    out["charges_maxdiff"] = GX._maxdiff(ref["charges"], r["charges"])
    out["gradient_maxdiff"] = GX._maxdiff(ref["gradient"], r["gradient"])
    out["pass"] = (not out["topology_lists_differ"] and out["cli_bonds_equal"]
                   and out["topology_parameters_maxdiff"] < TOL["charges"]
                   and out["charges_maxdiff"] < TOL["charges"]
                   and max(out["energy_diff"], out["terms_maxdiff"]) < TOL["energy"]
                   and out["gradient_maxdiff"] < TOL["gradient"])
    if cid in KNOWN_DIFFERENCES["accuracy"]:
        out["known_difference"] = KNOWN_DIFFERENCES["accuracy"][cid]
    return out


def _accuracy_job(job):
    return accuracy_case(*job)


def accuracy(args) -> dict:
    from concurrent.futures import ProcessPoolExecutor

    cases = GS.cases(include_large=True)
    jobs = [(c, args.xtb, args.workdir, args.conventions) for c in cases]
    with ProcessPoolExecutor(args.jobs) as ex:
        rows = list(ex.map(_accuracy_job, jobs))
    ok = [r for r in rows if r.get("pass")]
    fail = [r for r in rows if not r.get("pass")]
    for r in fail:
        print("  not reproduced:", r["id"], {k: v for k, v in r.items() if k not in ("id", "known_difference")})

    def worst(key):
        return max((r[key] for r in ok), default=None)

    summary = {
        "cases": len(rows), "reproduced": len(ok),
        "not_reproduced": {r["id"]: r.get("known_difference", "unexpected") for r in fail},
        "tolerances": TOL,
        "max_charges_diff": worst("charges_maxdiff"), "max_energy_diff": worst("energy_diff"),
        "max_term_diff": worst("terms_maxdiff"), "max_gradient_diff": worst("gradient_maxdiff"),
        "max_topology_parameter_diff": worst("topology_parameters_maxdiff"),
        "categories": {c: sum(1 for r in rows if r["category"] == c) for c in sorted({r["category"] for r in rows})},
        "largest": max(r["natoms"] for r in rows),
    }
    print("accuracy:", json.dumps(summary, indent=1))
    return {"xtb_version": xtb_version(args.xtb), "conventions": args.conventions, "summary": summary, "cases": rows}


def check(part: str, result: dict, conventions: str) -> int:
    """The gate of one part (tests/gate.py)."""
    s, rows = result["summary"], result["cases"]
    known = KNOWN_DIFFERENCES.get(part, {})
    seen = {k: v for k, v in known.items() if any(r["id"] == k for r in rows)}
    if part == "accuracy" or part == "topology":
        summary = {"not reproduced": [r["id"] for r in rows if not r.get("pass") and r["id"] not in known]}
        summary |= {k: s[k] for k in LIMITS[conventions][part] if k != "not reproduced"}
    else:  # hessian: xtb's or axtb's errors, except the known ones, and the largest differences
        summary = {"failures": [f"{r['id']} ({k})" for r in rows for k in ("xtb_error", "numerical", "analytic", "program_hessian")
                                if r["id"] not in known and (k in r if k == "xtb_error" else "error" in r.get(k, {}))]}
        summary |= {f"{m} {k}": s[m][k] for m in ("numerical", "analytic", "program_hessian") for k in s[m]}
    return gate.check(summary, LIMITS[conventions][part], seen)


def xtb_hessian(xtb, numbers, positions, charge, workdir, linear=False) -> dict:
    tmp = Path(tempfile.mkdtemp(prefix="gfnff_hess_", dir=workdir))
    try:
        GX.write_coord(tmp / "coord", numbers, positions)
        env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OMP_STACKSIZE": "1G"}
        p = subprocess.run([xtb, "coord", "--gfnff", "--chrg", str(int(charge)), "--hess"], cwd=tmp, env=env,
                           capture_output=True, text=True, preexec_fn=GX._unlimited_stack)
        if p.returncode != 0:
            where = [l.split()[2:5] for l in p.stderr.splitlines() if ".F90" in l or ".f90" in l][:1]
            msg = next((l.strip() for l in p.stderr.splitlines() if "forrtl" in l), f"exit status {p.returncode}")
            return {"error": msg + (f" in {' '.join(where[0])}" if where else "")}
        res = {}
        for key, pat in (("free_energy", "total free energy"), ("zpve", "zero point energy"), ("g_rrho", r"G\(RRHO\) contrib\.")):
            m = re.search(rf"::\s*{pat}\s+(-?\d+\.\d+)\s+Eh", p.stdout)
            res[key] = float(m.group(1)) if m else None
        freqs, ir = [], []
        for line in (tmp / "vibspectrum").read_text().splitlines():
            f = line.split()
            if not f or f[0].startswith(("$", "#")):
                continue
            freqs.append(float(f[-3]))
            ir.append(float(f[-2]))
        # xtb lists the six (linear: five) projected zero modes first
        nz = 5 if linear else 6
        res["frequencies"], res["ir_intensities"] = freqs[nz:], ir[nz:]
        if (tmp / "hessian").exists():
            res["hessian"] = GX.parse_hessian(tmp / "hessian", len(numbers))
        return res
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def rotmol(pos):
    """xtb 6.7.1's `rotmol(n, xyz, 1e-4, 2e-4, 3e-4)` (degrees) before its numerical Hessian."""
    out = np.array(pos, float)
    for (i, j), ang in (((1, 2), 1e-4), ((0, 2), 2e-4), ((0, 1), 3e-4)):
        a = ang * np.pi / 180.0
        xo, yo = out[:, i].copy(), out[:, j].copy()
        out[:, i] = xo * np.cos(a) + yo * np.sin(a)
        out[:, j] = -xo * np.sin(a) + yo * np.cos(a)
    return out


def hessians(args) -> dict:
    rows = []
    cases = {c[0]: c for c in GS.cases(include_large=False)}
    for cid, syms, xyz in (("lin-CO2", "OCO", [[0, 0, -1.17], [0, 0, 0], [0, 0, 1.17]]),
                           ("lin-HCN", "HCN", [[0, 0, -1.07], [0, 0, 0], [0, 0, 1.16]])):
        cases[cid] = (cid, "linear", *GS._mol(syms, xyz), 0)
    hess_cases = HESS_CASES + (HESS_LINEAR if args.conventions != "xtb-6.7.1" else [])
    for cid in hess_cases:
        _, cat, numbers, positions, charge = cases[cid]
        linear = cid in HESS_LINEAR
        ref = xtb_hessian(args.xtb, numbers, positions, charge, args.workdir, linear=linear)
        row = {"id": cid, "natoms": len(numbers), "linear": linear}
        if "error" in ref:
            row["xtb_error"] = ref["error"]
            rows.append(row)
            continue
        if "hessian" in ref:
            # the program's own Hessian (`hessian` file) through axtb's analysis in its conventions,
            # at the structure it was computed for (6.7.1: rotated)
            pos_x = rotmol(positions) if args.conventions == "xtb-6.7.1" else np.asarray(positions, float)
            r = cli({"numbers": [int(z) for z in numbers], "positions": pos_x.tolist(), "analyze_hessian": ref["hessian"].tolist(),
                     "analysis_conventions": args.conventions, "frequency_scale": 1.03})
            if "error" not in r and "frequencies" in r:
                f, fx = np.array(r["frequencies"]), np.array(ref["frequencies"])
                t = r["thermo"]
                row["program_hessian"] = {
                    "frequencies_maxdiff_cm1": float(np.max(np.abs(f - fx))) if f.shape == fx.shape else None,
                    "zpve_diff": abs(t["zpve"] - ref["zpve"]), "g_rrho_diff": abs(t["g_rrho"] - ref["g_rrho"]),
                }
            else:
                row["program_hessian"] = {"error": r.get("error")}
        for method in ("numerical", "analytic"):
            r = cli(request(numbers, positions, charge, hessian=True, hessian_method=method, conventions=args.conventions))
            if "error" in r:
                row[method] = {"error": r["error"]}
                continue
            f, fx = np.array(r["frequencies"]), np.array(ref["frequencies"])
            ir, irx = np.array(r["ir_intensities"]), np.array(ref["ir_intensities"])
            t = r["thermo"]
            row[method] = {
                "frequencies_maxdiff_cm1": float(np.max(np.abs(f - fx))) if f.shape == fx.shape else None,
                "ir_maxdiff_km_mol": float(np.max(np.abs(ir - irx))) if ir.shape == irx.shape else None,
                "zpve_diff": abs(t["zpve"] - ref["zpve"]), "g_rrho_diff": abs(t["g_rrho"] - ref["g_rrho"]),
                "free_energy_diff": abs(t["free_energy"] - ref["free_energy"]),
                "frequency_scale": t["frequency_scale"], "hessian_max_asymmetry": r["hessian_max_asymmetry"],
            }
        print("  hessian", cid, row.get("numerical"), row.get("analytic"), flush=True)
        rows.append(row)

    def worst(method, key):
        v = [r[method][key] for r in rows if method in r and r[method].get(key) is not None]
        return max(v) if v else None

    summary = {m: {k: worst(m, k) for k in ("frequencies_maxdiff_cm1", "ir_maxdiff_km_mol", "zpve_diff",
                                             "free_energy_diff")} for m in ("numerical", "analytic")}
    summary["program_hessian"] = {k: worst("program_hessian", k) for k in ("frequencies_maxdiff_cm1", "zpve_diff", "g_rrho_diff")}
    if args.conventions == "xtb-6.7.1":
        summary["note"] = ("xtb 6.7.1 --hess: central differences (step 0.005 bohr) of the analytic gradient of the slightly "
                           "rotated structure, projected in Cartesian coordinates; axtb numerical uses the same step, analytic is "
                           "the exact derivative of the gradient (it differs from xtb's by xtb's finite-difference error)")
    else:
        summary["note"] = ("xtb 6.8.0 --hess: central differences (step 0.005 bohr) of the analytic gradient of the structure "
                           "as given (no rotation), projected in mass-weighted coordinates (also linear molecules); axtb numerical "
                           "uses the same step, analytic is the exact derivative of the gradient (it differs from xtb's by xtb's "
                           "finite-difference error); program_hessian: xtb's own `hessian` file through axtb's analysis")
    print("hessian:", json.dumps(summary, indent=1))
    return {"summary": summary, "cases": rows}


def topology_case(case, xtb, workdir, conventions, seed) -> dict:
    """The structure displaced by 0.25 bohr (normal noise) and evaluated with the topology of
    the original: xtb reads the original's `gfnff_topo`, axtb gets `topology_positions`."""
    cid, cat, numbers, positions, charge = case
    out = {"id": cid, "natoms": len(numbers)}
    a = GX.run_xtb(numbers, positions, charge, workdir=workdir, xtb=xtb)
    if "error" in a or "topo_bytes" not in a:
        out["xtb_error"] = a.get("error", "no gfnff_topo")[-300:]
        return out
    posb = np.asarray(positions, float) + np.random.default_rng(seed).normal(0, 0.25, np.shape(positions))
    b = GX.run_xtb(numbers, posb, charge, workdir=workdir, xtb=xtb, topo_bytes=a["topo_bytes"])
    if "error" in b:
        out["xtb_error"] = b["error"][-300:]
        return out
    out["xtb_read_topology"] = bool(b.get("topology_read"))
    r = cli(request(numbers, posb, charge, gradient=True, conventions=conventions, topology_positions=np.asarray(positions).tolist()))
    if "error" in r:
        out["axtb_error"] = r["error"]
        return out
    terms = GX.compare_terms(b, r)
    out["energy_diff"] = terms.pop("total")
    out["terms_maxdiff"] = max(terms.values()) if terms else 0.0
    out["charges_maxdiff"] = GX._maxdiff(b["charges"], r["charges"])
    out["gradient_maxdiff"] = GX._maxdiff(b["gradient"], r["gradient"])
    out["pass"] = (out["xtb_read_topology"] and out["charges_maxdiff"] < TOL["charges"]
                   and max(out["energy_diff"], out["terms_maxdiff"]) < TOL["energy"] and out["gradient_maxdiff"] < TOL["gradient"])
    return out


def topology(args) -> dict:
    cases = {c[0]: c for c in GS.cases(include_large=False)}
    rows = [topology_case(cases[cid], args.xtb, args.workdir, args.conventions, 11 + k) for k, cid in enumerate(TOPO_CASES)]
    for r in rows:
        print("  topology_positions", r["id"], {k: v for k, v in r.items() if k != "id"}, flush=True)
    ok = [r for r in rows if r.get("pass")]
    summary = {"cases": len(rows), "reproduced": len(ok), "tolerances": TOL,
               "max_energy_diff": max((max(r["energy_diff"], r["terms_maxdiff"]) for r in ok), default=None),
               "max_charges_diff": max((r["charges_maxdiff"] for r in ok), default=None),
               "max_gradient_diff": max((r["gradient_maxdiff"] for r in ok), default=None)}
    print("topology:", json.dumps(summary, indent=1))
    return {"summary": summary, "cases": rows}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["accuracy", "hessian", "topology", "scaling", "all"], default="all")
    ap.add_argument("--jobs", type=int, default=4, help="parallel xtb/axtb runs (one thread each) for accuracy")
    ap.add_argument("--out", default=None, help="default: bench/results/gfnff_reproduction.json for xtb 6.7.x, "
                    "gfnff_reproduction_6.8.0.json for 6.8.0")
    ap.add_argument("--conventions", default=None, help="axtb's conventions (default: from the xtb version)")
    ap.add_argument("--xtb", default=programs.get(programs.XTB680), help="the xtb binary (default: xtb 6.8.0, $AXTB_XTB680)")
    ap.add_argument("--xtb-mem", type=float, default=10.0, help="address-space cap of xtb runs (GiB)")
    ap.add_argument("--threads", type=int, default=8, help="threads of the second axtb Hessian run")
    ap.add_argument("--max-atoms", type=int, default=10000)
    ap.add_argument("--hess-max", type=int, default=6000, help="largest system for axtb Hessians")
    ap.add_argument("--xtb-hess-max", type=int, default=700, help="largest system for xtb --hess")
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--xtb-hess-timeout", type=float, default=7200)
    ap.add_argument("--skip-xtb", action="store_true")
    ap.add_argument("--workdir", default=None, help="directory for the xtb runs (default: system temp)")
    args = ap.parse_args()
    if not args.xtb:
        programs.skip("pass --xtb or " + programs.reason(programs.XTB680))
    for exe in (BENCH, DUMP, CLI):
        if not exe.exists():
            sys.exit(f"{exe.relative_to(ROOT)} missing: cargo build --release -p axtb-gfnff --examples "
                     "&& cargo build --release -p axtb-cli")
    if args.conventions is None:
        args.conventions = GX.xtb_conventions(args.xtb)
    if args.out is None:
        name = "gfnff_reproduction.json" if args.conventions == "xtb-6.7.1" else "gfnff_reproduction_6.8.0.json"
        args.out = str(ROOT / "bench" / "results" / name)
    out = Path(args.out)
    data = json.loads(out.read_text()) if out.exists() else {}
    data["xtb_version"] = xtb_version(args.xtb)
    data["conventions"] = args.conventions
    gated = {}
    if args.part in ("accuracy", "all"):
        data["accuracy"] = gated["accuracy"] = accuracy(args)
    if args.part in ("hessian", "all"):
        data["hessian"] = gated["hessian"] = hessians(args)
    if args.part in ("topology", "all"):
        data["topology_positions"] = gated["topology"] = topology(args)
    if args.part in ("scaling", "all"):
        data["scaling"] = scaling(args)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {out}")
    if args.conventions != GX.xtb_conventions(args.xtb):
        print(f"conventions {args.conventions} against xtb {data['xtb_version']}: a measurement, not gated")
        return 0
    rc = 0
    for part, result in gated.items():  # the scaling part (timings) has no gate
        print(f"{part}:")
        rc |= check(part, result, args.conventions)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
