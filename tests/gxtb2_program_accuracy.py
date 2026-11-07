#!/usr/bin/env python3
"""The 2.0.1 program's numerical Hessian (`xtb --gxtb --hess`) against its own accuracy setting.

The program differentiates gradients from SCFs restarted from the reference wavefunction and
stopped at the calculator's thresholds (`--acc`; `$hess sccacc` does not reach the g-xTB
calculator), and its gradient takes the charge and multipole potential terms from the previous
SCF iteration, so each displaced gradient carries an error of the order of the SCF residual,
magnified ~100x by the central difference (2h = 0.01 bohr). For each case and `--acc` (1, 0.01,
1e-4) this records the wall time and the frequency differences to axtb's analytic Hessian (program
conventions, same geometry), and for the small cases the central differences of the program's own
analytic gradients (a fresh `--grad --acc 1e-4` run per displaced geometry) against axtb.
Appends `program_accuracy` to bench/results/gxtb2_hessian.json.

With `--interleaved` it instead runs, per case and thread count (1 and N), the program's
`--hess --acc A` for each A in `--accs` (default 0.001, a common tight setting,
and 1e-4) interleaved with axtb's analytic Hessian, each run with the load before it, and
reports the frequency errors of every Hessian against axtb's analytic one and against a tight
reference (Richardson numerical, h = 0.002/0.001 bohr, SCF accuracy 1e-6), all in program
conventions at the input geometry; stored as `versus_program[case]["program_acc"]`.

Usage: python tests/gxtb2_program_accuracy.py [--cases "Ni(CO)4,caffeine,taxol"] [--threads 8]
       [--fd "Ni(CO)4"] [--interleaved --accs 0.001,0.0001]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gxtb2_hessian as G  # noqa: E402


def program(xyz_text: str, charge: int, uhf: int, extra: list[str], threads: int, files: list[str]):
    with tempfile.TemporaryDirectory(prefix="gxtb2-acc-") as d:
        d = Path(d)
        (d / "mol.xyz").write_text(xyz_text)
        cmd = [G.GXTB2, "mol.xyz", "--gxtb", "--norestart", "--chrg", str(charge)] + extra
        if uhf:
            cmd += ["--uhf", str(uhf)]
        env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OMP_STACKSIZE="1G", HOME=str(d))
        t = time.perf_counter()
        out = subprocess.run(cmd, cwd=d, capture_output=True, text=True, env=env)
        wall = time.perf_counter() - t
        if "normal termination of xtb" not in out.stdout + out.stderr:
            raise RuntimeError("program failed:\n" + out.stdout[-2000:] + out.stderr[-2000:])
        return wall, {f: (d / f).read_text() for f in files}


def read_hessian(text: str, n3: int) -> np.ndarray:
    h = np.array([float(x) for x in text.split()[1:]]).reshape(n3, n3)
    return 0.5 * (h + h.T)


def read_gradient(text: str, nat: int) -> np.ndarray:
    g = []
    for line in text.splitlines():
        f = line.replace("D", "E").split()
        if len(f) == 3:
            try:
                g.append([float(x) for x in f])
            except ValueError:
                pass
    return np.array(g[-nat:]).ravel()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="Ni(CO)4,caffeine,taxol")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--fd", default="Ni(CO)4,caffeine")
    ap.add_argument("--out", default=str(G.ROOT / "bench/results/gxtb2_hessian.json"))
    ap.add_argument("--interleaved", action="store_true")
    ap.add_argument("--accs", default="0.001,0.0001")
    args = ap.parse_args()
    G.programs.require(G.programs.GXTB2)
    masses = G.xtb_masses()
    out_path = Path(args.out)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    if args.interleaved:
        return interleaved(args, masses, results, out_path)
    results.setdefault("program_accuracy", {})
    for name in filter(None, args.cases.split(",")):
        xyz, charge, uhf = G.CASES[name]
        text = xyz.read_text()
        nums, pos = G.read_xyz(xyz, G.MCTC_AATOAU)
        n3 = 3 * len(nums)
        ha = G.run_axtb(nums, pos, charge, uhf, "analytic", args.threads, conventions="gxtb")[0]
        fa = G.frequencies(ha, nums, pos, masses)
        rec = {"load_before": G.load_average(), "runs": {}}
        for acc in ["1", "0.01", "0.0001"]:
            extra = ["--hess"] + ([] if acc == "1" else ["--acc", acc])
            wall, files = program(text, charge, uhf, extra, args.threads, ["hessian"])
            fx = G.frequencies(read_hessian(files["hessian"], n3), nums, pos, masses)
            d = np.abs(fx - fa)
            rec["runs"][f"--acc {acc}"] = {"seconds": wall, "threads": args.threads, "max_dfreq": float(d.max()), "mean_dfreq": float(d.mean())}
            print(f"{name} program --hess --acc {acc} ({args.threads}t): {wall:8.2f} s, vs axtb analytic max|dfreq| {d.max():.3f} cm-1, "
                  f"mean {d.mean():.3f}", flush=True)
        if name in args.fd.split(","):
            lines = text.splitlines()
            nat = int(lines[0].split()[0])
            syms = [ln.split()[0] for ln in lines[2:2 + nat]]
            ang = np.array([[float(x) for x in ln.split()[1:4]] for ln in lines[2:2 + nat]])
            h = 0.005
            hfd = np.zeros((n3, n3))
            for y in range(n3):
                g = []
                for s in (1.0, -1.0):
                    a = ang.copy()
                    a[y // 3, y % 3] += s * h / G.MCTC_AATOAU
                    t = f"{nat}\n\n" + "".join(f"{e} {x:.15f} {yy:.15f} {z:.15f}\n" for e, (x, yy, z) in zip(syms, a))
                    _, files = program(t, charge, uhf, ["--grad", "--acc", "0.0001"], 1, ["gradient"])
                    g.append(read_gradient(files["gradient"], nat))
                hfd[:, y] = (g[0] - g[1]) / (2 * h)
            hfd = 0.5 * (hfd + hfd.T)
            ffd = G.frequencies(hfd, nums, pos, masses)
            rec["fd_of_program_gradients"] = {"step": h, "max_dH_vs_axtb_analytic": float(np.abs(hfd - ha).max()),
                                              "max_dfreq_vs_axtb_analytic": float(np.abs(ffd - fa).max())}
            print(f"{name} central differences of the program's --grad (h={h}) vs axtb analytic: max|dH| {np.abs(hfd - ha).max():.2e}, "
                  f"max|dfreq| {np.abs(ffd - fa).max():.3f} cm-1", flush=True)
        results["program_accuracy"][name] = rec
        out_path.write_text(json.dumps(results, indent=1))
    return 0


def interleaved(args, masses, results, out_path) -> int:
    results.setdefault("versus_program", {})
    accs = args.accs.split(",")
    for name in filter(None, args.cases.split(",")):
        xyz, charge, uhf = G.CASES[name]
        text = xyz.read_text()
        nums, pos = G.read_xyz(xyz, G.MCTC_AATOAU)
        n3 = 3 * len(nums)
        rec = {"runs": [], "time": {}, "vs_analytic": {}, "vs_reference": {}}
        hx = {}
        ha = None
        print(f"== {name}: program --hess --acc {accs} interleaved with axtb analytic", flush=True)
        for th in sorted({1, args.threads}):
            for what in [f"program --acc {a}" for a in accs[:1]] + ["axtb analytic"] + [f"program --acc {a}" for a in accs[1:]]:
                load = G.load_average()
                if what == "axtb analytic":
                    ha, wall, _, _ = G.run_axtb(nums, pos, charge, uhf, "analytic", th, conventions="gxtb")
                else:
                    acc = what.split()[-1]
                    wall, files = program(text, charge, uhf, ["--hess", "--acc", acc], th, ["hessian"])
                    hx[acc] = read_hessian(files["hessian"], n3)
                rec["runs"].append({"what": f"{what} {th}t", "load_before": load, "seconds": wall})
                rec["time"][f"{what} {th}t"] = wall
                print(f"   {what:22s} {th}t: {wall:8.2f} s  (load {load})", flush=True)
                results["versus_program"].setdefault(name, {})["program_acc"] = rec
                out_path.write_text(json.dumps(results, indent=1))
        load = G.load_average()
        ref, spread = G.reference(nums, pos, charge, uhf, args.threads, conventions="gxtb")
        rec["reference"] = {"what": "Richardson numerical 0.002/0.001 bohr, SCF accuracy 1e-6, program conventions",
                            "spread": spread, "load_before": load}
        fa = G.frequencies(ha, nums, pos, masses)
        fr = G.frequencies(ref, nums, pos, masses)
        da = np.abs(fa - fr)
        rec["vs_reference"]["axtb analytic"] = {"max_dH": float(np.abs(ha - ref).max()), "max_dfreq": float(da.max()), "mean_dfreq": float(da.mean())}
        print(f"   axtb analytic vs tight reference: max|dH| {np.abs(ha - ref).max():.2e}, max|dfreq| {da.max():.4f} cm-1", flush=True)
        p = G.tr_projector(pos)
        for acc, h in hx.items():
            fx = G.frequencies(h, nums, pos, masses)
            d1, d2 = np.abs(fx - fa), np.abs(fx - fr)
            rec["vs_analytic"][f"program --acc {acc}"] = {"max_dH_projected": float(np.abs(p @ ha @ p - h).max()), "max_dfreq": float(d1.max()),
                                                          "mean_dfreq": float(d1.mean())}
            rec["vs_reference"][f"program --acc {acc}"] = {"max_dfreq": float(d2.max()), "mean_dfreq": float(d2.mean())}
            print(f"   program --acc {acc}: vs axtb analytic max|dfreq| {d1.max():.3f} cm-1 (mean {d1.mean():.3f}); "
                  f"vs tight reference max {d2.max():.3f} (mean {d2.mean():.3f})", flush=True)
        results["versus_program"].setdefault(name, {})["program_acc"] = rec
        out_path.write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
