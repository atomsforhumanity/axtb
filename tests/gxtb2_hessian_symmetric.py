#!/usr/bin/env python3
"""g-xTB 2.0.1 analytic Hessians: the symmetric response terms (AnalyticOptions::symmetric).

The analytic Hessian used each displacement's coupled-perturbed solution once, so its error
was linear in the solver's tolerance (1e-10 in 0.1.x). With the symmetric form
H - (M + M^T), M_xy = z_x . r_y (z the solutions, r the final residuals), the error is
quadratic and the default tolerance is 1e-6. This script compares, per molecule, three
settings of `crates/gxtb/examples/hessian_symmetric.rs`, all on one tightly converged SCF:

- current: tolerance 1e-10, plain (the 0.1.x analytic Hessian);
- plain 1e-6: tolerance 1e-6 without the symmetric form;
- default: tolerance 1e-6 with the symmetric form.

Accuracy: the largest |dH| (Eh/bohr^2) and |dfrequency| (cm^-1) of each setting against
`current`. Frequencies: eigenvalues of the mass-weighted Hessian (standard atomic weights below)
without the 6 (linear: 5) closest to zero, the translations and rotations, whose square roots
magnify rounding.
Cost: wall seconds of the analytic Hessian (after the SCF) and peak resident memory
(/usr/bin/time), taxol on 1 and 8 threads and (H2O)100 on 8.

Writes bench/results/gxtb2_hessian_symmetric.json.

Usage: python tests/gxtb2_hessian_symmetric.py [--skip-cost]
"""
import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "tests/structures"
R2 = ROOT / "crates/gxtb/tests/reference_v2"
EXAMPLE = ROOT / "target/release/examples/hessian_symmetric"
OUT = ROOT / "bench/results/gxtb2_hessian_symmetric.json"
TIME_V = Path("/usr/bin/time")
HARTREE_TO_CM = 2.1947463136314e7 / 100
AMU_TO_ME = 1.0 / 5.485799090441e-4
# standard atomic weights for the frequency differences
MASS = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "F": 18.998, "Cl": 35.45, "Fe": 55.845,
        "Ni": 58.693, "Cu": 63.546, "Gd": 157.25, "U": 238.03}

# label: (xyz, charge, unpaired electrons, electronic temperature, conventions)
CASES = {
    "water": (S / "water.xyz", 0, 0, 0.0, "axtb"),
    "O2 triplet": (R2 / "reference/dioxygen_triplet/mol.xyz", 0, 2, 0.0, "axtb"),
    "caffeine": (S / "caffeine.xyz", 0, 0, 0.0, "axtb"),
    "[Cu(NH3)4]2+ doublet": (S / "cu-nh3-4.xyz", 2, 1, 0.0, "axtb"),
    "ferrocene": (S / "ferrocene.xyz", 0, 0, 0.0, "axtb"),
    "Ni(CO)4": (R2 / "reference/nickel_tetracarbonyl/mol.xyz", 0, 0, 0.0, "axtb"),
    "UF6": (R2 / "elements/UF6/mol.xyz", 0, 0, 0.0, "axtb"),
    "GdCl3 octet": (R2 / "elements/GdCl3/mol.xyz", 0, 7, 0.0, "axtb"),
    "taxol": (S / "taxol.xyz", 0, 0, 0.0, "axtb"),
    "caffeine, 3000 K": (S / "caffeine.xyz", 0, 0, 3000.0, "axtb"),
    "O2 triplet, 3000 K": (R2 / "reference/dioxygen_triplet/mol.xyz", 0, 2, 3000.0, "axtb"),
    "[Cu(NH3)4]2+, 3000 K": (S / "cu-nh3-4.xyz", 2, 1, 3000.0, "axtb"),
    "ferrocene, 5000 K": (S / "ferrocene.xyz", 0, 0, 5000.0, "axtb"),
    "caffeine, program conventions": (S / "caffeine.xyz", 0, 0, 0.0, "gxtb"),
    "O2 triplet, program conventions": (R2 / "reference/dioxygen_triplet/mol.xyz", 0, 2, 0.0, "gxtb"),
}
SETTINGS = {"current": ("1e-10", "0"), "plain 1e-6": ("1e-6", "0"), "default": ("1e-6", "1")}
COST = [("taxol", S / "taxol.xyz", 1), ("taxol", S / "taxol.xyz", 8), ("(H2O)100", S / "large/water100.xyz", 8)]


def load1() -> float:
    return float(Path("/proc/loadavg").read_text().split()[0])


def run(xyz: Path, charge, uhf, etemp, conv, tol, sym, threads):
    while load1() > 1.5:
        time.sleep(5)
    cmd = [str(TIME_V), "-f", "%M", str(EXAMPLE), str(xyz), str(charge), str(uhf), str(etemp), conv, tol, sym, str(threads)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        sys.exit(f"{xyz.name}: {p.stderr[-800:]}")
    r = json.loads(p.stdout)
    n3 = r["n3"]
    return {"h": np.array(r["hessian"]).reshape(n3, n3), "seconds": sum(r["phases"].values()),
            "phases": r["phases"], "peak_mb": int(p.stderr.split()[-1]) / 1024, "asymmetry": r["max_asymmetry"]}


def frequencies(xyz: Path, h):
    lines = xyz.read_text().split("\n")
    syms = [l.split()[0].capitalize() for l in lines[2:2 + int(lines[0])]]
    m = np.repeat([MASS[s] * AMU_TO_ME for s in syms], 3)
    w = np.linalg.eigvalsh(h / np.sqrt(np.outer(m, m)))
    rigid = 5 if len(syms) == 2 else 6
    vib = np.sort(w[np.argsort(np.abs(w))[rigid:]])
    return np.sign(vib) * np.sqrt(np.abs(vib)) * HARTREE_TO_CM


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-cost", action="store_true")
    args = ap.parse_args()
    subprocess.run(["cargo", "build", "-q", "--release", "-p", "axtb-gxtb", "--example", "hessian_symmetric"], cwd=ROOT, check=True)
    out = {"meta": {"axtb_commit": git("rev-parse", "--short=7", "HEAD"), "dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
                    "cpu": next((l.split(":", 1)[1].strip() for l in Path("/proc/cpuinfo").read_text().splitlines() if l.startswith("model name")), platform.machine()),
                    "accuracy_threads": 8, "settings": {k: {"tol": float(t), "symmetric": s == "1"} for k, (t, s) in SETTINGS.items()}},
           "accuracy": [], "cost": []}
    if args.skip_cost and OUT.exists():
        out["cost"] = json.loads(OUT.read_text()).get("cost", [])
    for label, (xyz, charge, uhf, etemp, conv) in CASES.items():
        res = {k: run(xyz, charge, uhf, etemp, conv, t, s, 8) for k, (t, s) in SETTINGS.items()}
        ref = res["current"]
        fref = frequencies(xyz, ref["h"])
        row = {"case": label, "atoms": int(xyz.read_text().split()[0])}
        for k in ("plain 1e-6", "default"):
            row[f"max_dH_{k}"] = float(np.abs(res[k]["h"] - ref["h"]).max())
            row[f"max_dfreq_{k}"] = float(np.abs(frequencies(xyz, res[k]["h"]) - fref).max())
        row["asymmetry_current"], row["asymmetry_default"] = ref["asymmetry"], res["default"]["asymmetry"]
        out["accuracy"].append(row)
        print(f"{label:32} plain 1e-6 {row['max_dH_plain 1e-6']:.1e} default {row['max_dH_default']:.1e}  "
              f"dfreq default {row['max_dfreq_default']:.1e} cm-1", flush=True)
        OUT.write_text(json.dumps(out, indent=1))
    if not args.skip_cost:
        for label, xyz, threads in COST:
            res = {k: run(xyz, 0, 0, 0.0, "axtb", *SETTINGS[k], threads) for k in ("current", "default")}
            row = {"case": label, "atoms": int(xyz.read_text().split()[0]), "threads": threads,
                   "seconds_current": res["current"]["seconds"], "seconds_default": res["default"]["seconds"],
                   "speedup": res["current"]["seconds"] / res["default"]["seconds"],
                   "peak_mb_current": res["current"]["peak_mb"], "peak_mb_default": res["default"]["peak_mb"],
                   "max_dH": float(np.abs(res["default"]["h"] - res["current"]["h"]).max()),
                   "phases_current": res["current"]["phases"], "phases_default": res["default"]["phases"]}
            out["cost"].append(row)
            print(f"{label:10} {threads} threads: {row['seconds_current']:.1f} -> {row['seconds_default']:.1f} s "
                  f"({row['speedup']:.2f}x), {row['peak_mb_current']:.0f} -> {row['peak_mb_default']:.0f} MB", flush=True)
            OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
