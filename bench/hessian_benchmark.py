#!/usr/bin/env python3
"""Hessian benchmark: axtb analytic / response / numerical Hessians and `xtb --hess`.

For every case it records
- wall time of each axtb method at 1 and N threads (best of `--repeat`), through the
  `axtb` CLI (JSON in/out), and of `xtb --hess` with OMP_NUM_THREADS = threads;
- accuracy of each axtb method (default settings) against a tight reference: the
  Richardson extrapolation (4 H(h/2) - H(h)) / 3 of numerical Hessians with steps 0.002
  and 0.001 bohr and every SCF converged to accuracy 1e-8 (FD error ~h^4); max |dH| and
  the largest harmonic-frequency difference;
- agreement with `xtb --hess --acc 0.001` in xtb conventions (axtb's Hessian projected
  like xtb's, xtb's atomic masses for frequencies).

Usage: python bench/hessian_benchmark.py [--cases water,caffeine] [--threads 8] [--repeat 2]
Environment: AXTB_BIN (default target/release/axtb), AXTB_REFERENCE (reference sources,
for xtb's atomic masses; default ./reference, see reference/README.md), AXTB_GXTB2 (xtb 6.7.1,
the g-xTB 2.0.1 build, for `xtb --hess`; see tests/programs.py; without it, as --no-xtb).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests"))
from validate_tblite import read_xyz  # noqa: E402
from validate_xtb import write_coord  # noqa: E402
import programs  # noqa: E402

AXTB = os.environ.get("AXTB_BIN", str(ROOT / "target/release/axtb"))
# `xtb --hess` of xtb 6.7.1 (the g-xTB 2.0.1 build), whose projection and rotation the comparison
# below reproduces in the xtb-6.7.1 conventions
XTB = programs.get(programs.GXTB2)
REFERENCE = Path(os.environ.get("AXTB_REFERENCE", ROOT / "reference"))
AMU_TO_ME = 1.0 / 5.485799090441e-4      # CODATA 2022 electron mass in u
HARTREE_TO_CM = 2.1947463136314e7 / 100   # CODATA 2022 hartree-inverse meter
S = ROOT / "tests/structures"

# name, file, charge, unpaired electrons, ALPB solvent
CASES = {
    "water": ("water.xyz", 0, 0, None),
    "caffeine": ("caffeine.xyz", 0, 0, None),
    "tyr-xab": ("tyr-xab.xyz", 0, 0, None),
    "taxol": ("taxol.xyz", 0, 0, None),
    "Cu(NH3)4 2+": ("cu-nh3-4.xyz", 2, 1, None),
    "caffeine+ doublet": ("caffeine.xyz", 1, 1, None),
    "caffeine ALPB(water)": ("caffeine.xyz", 0, 0, "water"),
    "taxol ALPB(water)": ("taxol.xyz", 0, 0, "water"),
}


def load(path: Path):
    nums, pos, _ = read_xyz(path)
    return nums, pos


def xtb_masses() -> np.ndarray:
    path = REFERENCE / "xtb-gxtb/src/param/atomicmass.f90"
    if not path.exists():
        programs.skip("needs the xtb source in $AXTB_REFERENCE/xtb-gxtb (reference/README.md) for its atomic masses")
    src = path.read_text()
    body = src[src.index("atomicMass(1:118)"):]
    body = body[body.index("[") + 1:body.index("]")]
    return np.array([float(x.replace("_wp", "")) for x in re.findall(r"[\d.]+_wp", body)])


def tr_projector(pos: np.ndarray) -> np.ndarray:
    """Projector removing translations and rotations (as xtb's `hessian` file)."""
    n = len(pos)
    c = pos - pos.mean(axis=0)
    vecs = []
    for k in range(3):
        t = np.zeros((n, 3)); t[:, k] = 1.0; vecs.append(t.ravel())
    for k in range(3):
        e = np.zeros(3); e[k] = 1.0
        vecs.append(np.cross(e, c).ravel())
    q, r = np.linalg.qr(np.array(vecs).T)
    q = q[:, np.abs(np.diag(r)) > 1e-8]
    return np.eye(3 * n) - q @ q.T


def frequencies(h: np.ndarray, nums, pos, masses) -> np.ndarray:
    """Harmonic frequencies (cm-1, imaginary as negative) of the projected Hessian."""
    p = tr_projector(pos)
    m = np.repeat(masses[np.asarray(nums) - 1] * AMU_TO_ME, 3)
    hm = (p @ h @ p) / np.sqrt(np.outer(m, m))
    w = np.linalg.eigvalsh(0.5 * (hm + hm.T))
    return np.sign(w) * np.sqrt(np.abs(w)) * HARTREE_TO_CM


def run_axtb(nums, pos, charge, uhf, alpb, method, threads, step=None, acc=None, conventions="axtb"):
    req = {"numbers": [int(x) for x in nums], "positions": pos.tolist(), "charge": charge, "uhf": uhf,
           "threads": threads, "conventions": conventions, "hessian": True, "hessian_method": method}
    if alpb:
        req["alpb"] = alpb
    if step is not None:
        req["hessian_step"] = step
    if acc is not None:
        req["hessian_accuracy"] = acc
    out = subprocess.run([AXTB, "--json"], input=json.dumps(req), capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(out.stderr + out.stdout)
    d = json.loads(out.stdout)
    t = dict(d["hessian_timings"])
    return np.array(d["hessian"]), t["total"], d["hessian_max_asymmetry"]


def run_xtb(nums, pos, charge, uhf, alpb, threads, acc=None, sccacc=None):
    """`xtb --hess`; `sccacc` sets the SCC accuracy of the displacements ($hess sccacc,
    default 0.3 in xtb 6.7.1 whatever --acc says)."""
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        write_coord(d / "coord", nums, pos)
        cmd = [XTB, "coord", "--gfn", "2", "--hess", "--chrg", str(charge), "--uhf", str(uhf)]
        if acc is not None:
            cmd += ["--acc", str(acc)]
        if sccacc is not None:
            (d / "xcontrol").write_text(f"$hess\n   sccacc={sccacc}\n$end\n")
            cmd += ["--input", "xcontrol"]
        if alpb:
            cmd += ["--alpb", alpb]
        env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OMP_STACKSIZE="1G")
        t = time.perf_counter()
        out = subprocess.run(cmd, cwd=d, capture_output=True, text=True, env=env)
        wall = time.perf_counter() - t
        if out.returncode or "normal termination of xtb" not in out.stdout + out.stderr:
            raise RuntimeError("xtb failed:\n" + out.stdout[-2000:] + out.stderr[-2000:])
        if sccacc is not None and f"SCC accuracy         :{sccacc:10.5f}" not in out.stdout:
            raise RuntimeError("xtb did not apply $hess sccacc")
        if "Unknown option" in out.stdout + out.stderr:
            raise RuntimeError("xtb ignored an option")
        vals = [float(x) for x in (d / "hessian").read_text().split()[1:]]
    n3 = 3 * len(nums)
    h = np.array(vals).reshape(n3, n3)
    return 0.5 * (h + h.T), wall


def best(f, repeat):
    res = [f() for _ in range(repeat)]
    return min(res, key=lambda r: r[1])


def load_average() -> str:
    return " ".join(Path("/proc/loadavg").read_text().split()[:3])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=",".join(CASES))
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--no-xtb", action="store_true")
    ap.add_argument("--no-reference", action="store_true")
    ap.add_argument("--timing-only", action="store_true", help="timings only (no reference, no xtb comparison)")
    ap.add_argument("--out", default=str(ROOT / "bench/results/hessian.json"))
    args = ap.parse_args()
    if args.timing_only:
        args.no_reference = True
    if not args.no_xtb and not programs.note_skip("the xtb --hess runs", programs.GXTB2):
        args.no_xtb = True
    masses = xtb_masses()
    nt = args.threads
    results = {"machine": platform.processor() or platform.machine(), "threads": nt, "cases": {}}
    try:
        results["xtb_version"] = subprocess.run([XTB, "--version"], capture_output=True, text=True).stdout.strip().splitlines()[-1]
    except Exception:  # noqa: BLE001
        results["xtb_version"] = None
    results["axtb_commit"] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                                            capture_output=True, text=True).stdout.strip()
    for name in args.cases.split(","):
        fname, charge, uhf, alpb = CASES[name]
        nums, pos = load(S / fname)
        r = {"natoms": len(nums), "load_before": load_average(), "time": {}, "asymmetry": {}, "accuracy": {}}
        print(f"== {name} ({len(nums)} atoms), load {r['load_before']}", flush=True)
        hs = {}
        for method in ["analytic", "response", "numerical"]:
            for th in [1, nt]:
                h, t, asym = best(lambda: run_axtb(nums, pos, charge, uhf, alpb, method, th), args.repeat)
                r["time"][f"axtb {method} {th}t"] = t
                r["asymmetry"][method] = asym
                hs[method] = h
                print(f"   axtb {method:9s} {th}t: {t:8.3f} s  (asym before symmetrization {asym:.1e})", flush=True)
        if not args.no_xtb:
            for th in [1, nt]:
                _, t = best(lambda: run_xtb(nums, pos, charge, uhf, alpb, th), args.repeat)
                r["time"][f"xtb {th}t"] = t
                print(f"   xtb --hess      {th}t: {t:8.3f} s", flush=True)
        r["load_after"] = load_average()
        if not args.no_reference:
            n1, _, _ = run_axtb(nums, pos, charge, uhf, alpb, "numerical", nt, step=0.002, acc=1e-8)
            n2, _, _ = run_axtb(nums, pos, charge, uhf, alpb, "numerical", nt, step=0.001, acc=1e-8)
            r["reference_spread"] = float(np.abs(n1 - n2).max())
            if alpb:
                # the ALPB/GBSA surface term is C1 but not C2: finite-difference Hessians
                # with steps >= 2e-4 bohr straddle many switching-function kinks and cannot
                # be extrapolated; use a response Hessian with a 1e-5 bohr step instead
                # (taxol: 5e-5 still straddles a kink, 2e-5..2e-6 agree with the analytic to 1e-10)
                ref, _, _ = run_axtb(nums, pos, charge, uhf, alpb, "response", nt, step=1e-5, acc=1e-8)
                r["reference"] = "response, step 1e-5, acc 1e-8"
                r["numerical_tight_vs_reference"] = float(np.abs(n2 - ref).max())
                print(f"   tight numerical (0.001, acc 1e-8) vs reference: max|dH| {np.abs(n2 - ref).max():.2e}", flush=True)
            else:
                ref = (4.0 * n2 - n1) / 3.0
                r["reference"] = "Richardson numerical 0.002/0.001, acc 1e-8"
            fref = frequencies(ref, nums, pos, masses)
            for method, h in hs.items():
                f = frequencies(h, nums, pos, masses)
                r["accuracy"][method] = {"max_dH": float(np.abs(h - ref).max()), "max_dfreq": float(np.abs(f - fref).max())}
                print(f"   {method:9s} vs tight reference: max|dH| {np.abs(h - ref).max():.2e} Eh/bohr2, "
                      f"max|dfreq| {np.abs(f - fref).max():.3f} cm-1", flush=True)
        if not args.no_xtb and not args.timing_only:
            p = tr_projector(pos)
            r["vs_xtb"] = {}
            # the xtb-6.7.1 conventions: this comparison projects and rotates as xtb 6.7.1 does
            # (the g-xTB 2.0.1 build on PATH); xtb 6.8.0 projects in mass-weighted coordinates and
            # does not rotate (tests/reference_matrix/run.py `xtb_projected`)
            hs_x = {m: run_axtb(nums, pos, charge, uhf, alpb, m, nt, conventions="xtb-6.7.1")[0] for m in ["analytic", "numerical"]}
            for label, kw in [("--acc 0.001", {"acc": 0.001}), ("--acc 0.001 sccacc 0.001", {"acc": 0.001, "sccacc": 0.001})]:
                hx, tx = run_xtb(nums, pos, charge, uhf, alpb, nt, **kw)
                fx = frequencies(hx, nums, pos, masses)
                r["time"][f"xtb {label} {nt}t"] = tx
                for method, h in hs_x.items():
                    hp = p @ h @ p
                    fa = frequencies(h, nums, pos, masses)
                    r["vs_xtb"][f"{method} vs xtb {label}"] = {"max_dH": float(np.abs(hp - hx).max()),
                                                              "max_dfreq": float(np.abs(fa - fx).max())}
                    print(f"   xtb conventions, {method:9s} vs xtb --hess {label}: max|dH| {np.abs(hp - hx).max():.2e}, "
                          f"max|dfreq| {np.abs(fa - fx).max():.3f} cm-1", flush=True)
        results["cases"][name] = r
        Path(args.out).write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
