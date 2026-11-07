#!/usr/bin/env python3
"""g-xTB 2.0.1 Hessians: timings and accuracy of axtb's analytic, response and numerical
Hessians, and the 2.0.1 program's `xtb --gxtb --hess`.

For every case it records
- wall time of each axtb method (default settings) at 1 and N threads (best of `--repeat`),
  through the `axtb` CLI (JSON in/out, default conventions), with the system load;
- accuracy against a tight reference, the Richardson extrapolation (4 H(h/2) - H(h)) / 3 of
  numerical Hessians with steps 0.002 and 0.001 bohr and SCFs converged to accuracy 1e-6
  (FD error ~h^4): max |dH| and the largest harmonic-frequency difference (projected Hessian,
  the program's atomic masses); and against a second reference without SCF noise, the
  Richardson extrapolation of response Hessians with steps 2e-4 and 1e-4 bohr (reference SCF
  accuracy 1e-6). Both still see the ~1e-11 Eh steps of tblite's primitive screening (heavy
  elements: ~1e-7 Eh/bohr^2 at these steps), which the analytic Hessian does not;
- for the `--program` cases, the 2.0.1 program's `--ohess` (optimization, then its numerical
  Hessian, OMP_NUM_THREADS=1) against axtb's analytic Hessian at the program's optimized
  geometry in program conventions: max |dH| of the projected Hessians, frequency differences,
  and both against axtb's tight reference there;
- with `--memory name:threads,...`, the peak resident memory (`/usr/bin/time -v`) of the
  analytic Hessian and of the single point alone (default conventions) for the benchmark cases
  and larger systems (C50H102, (H2O)100, (H2O)200); every axtb run records its peak RSS;
- for the `--versus` cases, the head-to-head: the program's `xtb mol.xyz --gxtb --hess` at
  OMP_NUM_THREADS = 1 and N and axtb's analytic, response and numerical Hessians at 1 and N
  threads (program conventions, the same input geometry), interleaved, each run with the
  load (`/proc/loadavg`) before it; frequency differences of each axtb method to the program.

Usage: python tests/gxtb2_hessian.py [--cases water,caffeine] [--threads 8] [--repeat 2]
       [--program water,caffeine] [--versus caffeine,taxol] [--no-reference]
       [--memory "taxol:1,taxol:8,(H2O)100:8"]
Environment: AXTB_BIN (default target/release/axtb), AXTB_GXTB2 (the 2.0.1 program's xtb, see
tests/programs.py; the --program and --versus cases are skipped without it), AXTB_REFERENCE
(reference sources for the program's atomic masses; default ./reference, reference/README.md).
Writes bench/results/gxtb2_hessian.json (--out).
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
import programs  # noqa: E402

SYMBOLS = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
           "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
           "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn "
           "Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr").split()

AXTB = os.environ.get("AXTB_BIN", str(ROOT / "target/release/axtb"))
GXTB2 = programs.get(programs.GXTB2)
REFERENCE = Path(os.environ.get("AXTB_REFERENCE", ROOT / "reference"))
AMU_TO_ME = 1.0 / 5.485799090441e-4      # CODATA 2022 electron mass in u
HARTREE_TO_CM = 2.1947463136314e7 / 100   # CODATA 2022 hartree-inverse meter
AATOAU = 1.0 / 0.529177210544             # CODATA 2022 (axtb conventions)
MCTC_AATOAU = 1.8897261246204404          # mctc-lib (the 2.0.1 program)
S = ROOT / "tests/structures"
R2 = ROOT / "crates/gxtb/tests/reference_v2"

# name: (xyz file, charge, unpaired electrons)
CASES = {
    "water": (S / "water.xyz", 0, 0),
    "O2 triplet": (R2 / "reference/dioxygen_triplet/mol.xyz", 0, 2),
    "caffeine": (S / "caffeine.xyz", 0, 0),
    "Cu(NH3)4 2+": (S / "cu-nh3-4.xyz", 2, 1),
    "ferrocene": (S / "ferrocene.xyz", 0, 0),
    "Ni(CO)4": (R2 / "reference/nickel_tetracarbonyl/mol.xyz", 0, 0),
    "UF6": (R2 / "elements/UF6/mol.xyz", 0, 0),
    "GdCl3 octet": (R2 / "elements/GdCl3/mol.xyz", 0, 7),
    "taxol": (S / "taxol.xyz", 0, 0),
}
# larger systems for the memory measurements only
MEMORY_CASES = dict(CASES, **{
    "C50H102": (S / "large/c50h102.xyz", 0, 0),
    "(H2O)100": (S / "large/water100.xyz", 0, 0),
    "(H2O)200": (S / "large/water200.xyz", 0, 0),
})
TIME_V = Path("/usr/bin/time")
LAST_PEAK_RSS_MB = None


def read_xyz(path: Path, factor: float):
    lines = path.read_text().splitlines()
    nat = int(lines[0].split()[0])
    nums, pos = [], []
    for line in lines[2:2 + nat]:
        f = line.split()
        nums.append(SYMBOLS.index(f[0].capitalize()) + 1)
        pos.append([float(x) * factor for x in f[1:4]])
    return np.array(nums), np.array(pos)


def xtb_masses() -> np.ndarray:
    path = REFERENCE / "xtb-gxtb/src/param/atomicmass.f90"
    if not path.exists():
        programs.skip("needs the xtb source in $AXTB_REFERENCE/xtb-gxtb (reference/README.md) for its atomic masses")
    src = path.read_text()
    body = src[src.index("atomicMass(1:118)"):]
    body = body[body.index("[") + 1:body.index("]")]
    return np.array([float(x.replace("_wp", "")) for x in re.findall(r"[\d.]+_wp", body)])


def tr_projector(pos: np.ndarray) -> np.ndarray:
    """Projector removing translations and rotations (as the program's `hessian` file)."""
    n = len(pos)
    c = pos - pos.mean(axis=0)
    vecs = []
    for k in range(3):
        t = np.zeros((n, 3))
        t[:, k] = 1.0
        vecs.append(t.ravel())
    for k in range(3):
        e = np.zeros(3)
        e[k] = 1.0
        vecs.append(np.cross(e, c).ravel())
    q, r = np.linalg.qr(np.array(vecs).T)
    q = q[:, np.abs(np.diag(r)) > 1e-8]
    return np.eye(3 * n) - q @ q.T


def frequencies(h: np.ndarray, nums, pos, masses) -> np.ndarray:
    """Harmonic frequencies (cm-1, imaginary as negative) of the projected Hessian, ascending,
    without the 5 or 6 zero modes."""
    p = tr_projector(pos)
    m = np.repeat(masses[np.asarray(nums) - 1] * AMU_TO_ME, 3)
    hm = (p @ h @ p) / np.sqrt(np.outer(m, m))
    w = np.linalg.eigvalsh(0.5 * (hm + hm.T))
    f = np.sign(w) * np.sqrt(np.abs(w)) * HARTREE_TO_CM
    nzero = 3 * len(nums) - np.linalg.matrix_rank(p, tol=1e-6)
    order = np.argsort(np.abs(f))
    return np.sort(f[order[nzero:]])


def run_axtb(nums, pos, charge, uhf, method, threads, step=None, acc=None, conventions="axtb", hessian=True):
    """One axtb CLI run; its peak RSS (MB, through /usr/bin/time -v) goes to LAST_PEAK_RSS_MB."""
    global LAST_PEAK_RSS_MB
    req = {"numbers": [int(x) for x in nums], "positions": pos.tolist(), "charge": charge, "uhf": uhf,
           "method": "gxtb-v2.0.1", "threads": threads, "conventions": conventions, "hessian": hessian}
    if hessian:  # the CLI rejects Hessian options in a request without a Hessian
        req["hessian_method"] = method
    if step is not None:
        req["hessian_step"] = step
    if acc is not None:
        req["hessian_accuracy"] = acc
    cmd = ([str(TIME_V), "-v"] if TIME_V.exists() else []) + [AXTB, "--json"]
    t = time.perf_counter()
    out = subprocess.run(cmd, input=json.dumps(req), capture_output=True, text=True)
    wall = time.perf_counter() - t
    if out.returncode:
        raise RuntimeError(out.stderr + out.stdout)
    m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", out.stderr)
    LAST_PEAK_RSS_MB = round(int(m.group(1)) / 1024, 1) if m else None
    d = json.loads(out.stdout)
    if not hessian:
        return None, wall, {}, None
    timings = dict(d["hessian_timings"])
    return np.array(d["hessian"]), wall, timings, d["hessian_max_asymmetry"]


def response_reference(nums, pos, charge, uhf, threads):
    r1 = run_axtb(nums, pos, charge, uhf, "response", threads, step=2e-4, acc=1e-6)[0]
    r2 = run_axtb(nums, pos, charge, uhf, "response", threads, step=1e-4, acc=1e-6)[0]
    return (4.0 * r2 - r1) / 3.0


def reference(nums, pos, charge, uhf, threads, conventions="axtb"):
    n1 = run_axtb(nums, pos, charge, uhf, "numerical", threads, step=0.002, acc=1e-6, conventions=conventions)[0]
    n2 = run_axtb(nums, pos, charge, uhf, "numerical", threads, step=0.001, acc=1e-6, conventions=conventions)[0]
    return (4.0 * n2 - n1) / 3.0, float(np.abs(n1 - n2).max())


def run_program(xyz: Path, charge: int, uhf: int, workdir: Path):
    """The 2.0.1 program's --ohess: optimized geometry (Å), projected Hessian, frequencies, time."""
    (workdir / "mol.xyz").write_text(xyz.read_text())
    cmd = [GXTB2, "mol.xyz", "--gxtb", "--norestart", "--ohess", "--chrg", str(charge)]
    if uhf:
        cmd += ["--uhf", str(uhf)]
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", HOME=str(workdir))
    t = time.perf_counter()
    out = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, env=env)
    wall = time.perf_counter() - t
    if "normal termination of xtb" not in out.stdout + out.stderr:
        raise RuntimeError("program failed:\n" + out.stdout[-2000:] + out.stderr[-2000:])
    vals = [float(x) for x in (workdir / "hessian").read_text().split()[1:]]
    n3 = int(round(np.sqrt(len(vals))))
    h = np.array(vals).reshape(n3, n3)
    text = out.stdout
    block = text[text.rindex("vibrational frequencies (cm"):]
    block = block[:block.index("------")]
    lines = [ln for ln in block.splitlines() if ln.strip().startswith("eigval")]
    freqs = np.array([float(x) for ln in lines for x in re.findall(r"-?\d+\.\d+", ln)])
    step = re.search(r"step length\s*:\s*([\d.]+)", text).group(1)
    sccacc = re.search(r"SCC accuracy\s*:\s*([\d.]+)", text).group(1)
    return workdir / "xtbopt.xyz", 0.5 * (h + h.T), freqs, wall, {"step": float(step), "scc_accuracy": float(sccacc)}


def run_program_hess(xyz: Path, charge: int, uhf: int, threads: int):
    """The 2.0.1 program's `--hess` at the input geometry: projected Hessian and wall time."""
    with tempfile.TemporaryDirectory(prefix="gxtb2-hess-") as d:
        d = Path(d)
        (d / "mol.xyz").write_text(xyz.read_text())
        cmd = [GXTB2, "mol.xyz", "--gxtb", "--norestart", "--hess", "--chrg", str(charge)]
        if uhf:
            cmd += ["--uhf", str(uhf)]
        env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OMP_STACKSIZE="1G", HOME=str(d))
        t = time.perf_counter()
        out = subprocess.run(cmd, cwd=d, capture_output=True, text=True, env=env)
        wall = time.perf_counter() - t
        if "normal termination of xtb" not in out.stdout + out.stderr:
            raise RuntimeError("program failed:\n" + out.stdout[-2000:] + out.stderr[-2000:])
        vals = [float(x) for x in (d / "hessian").read_text().split()[1:]]
    n3 = int(round(np.sqrt(len(vals))))
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
    ap.add_argument("--methods", default="analytic,response,numerical")
    ap.add_argument("--no-reference", default="taxol", help="cases without the tight reference")
    ap.add_argument("--program", default="water,O2 triplet,caffeine,Ni(CO)4,GdCl3 octet")
    ap.add_argument("--versus", default="caffeine,taxol")
    ap.add_argument("--memory", default="", help="name:threads,... peak RSS of the analytic Hessian")
    ap.add_argument("--out", default=str(ROOT / "bench/results/gxtb2_hessian.json"))
    args = ap.parse_args()
    if (args.program or args.versus) and not programs.note_skip("the --program and --versus cases", programs.GXTB2):
        args.program = args.versus = ""
    masses = xtb_masses()
    nt = args.threads
    out_path = Path(args.out)
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    results.update({"machine": platform.processor() or platform.machine(), "threads": nt,
                    "axtb_commit": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                                                  capture_output=True, text=True).stdout.strip(),
                    "reference": "Richardson (4 H(0.001) - H(0.002))/3 of numerical Hessians, SCF accuracy 1e-6",
                    "response_reference": "Richardson (4 H(1e-4) - H(2e-4))/3 of response Hessians, reference SCF accuracy 1e-6"})
    results.setdefault("cases", {})
    results.setdefault("program", {})
    results.setdefault("versus_program", {})
    no_ref = set(filter(None, args.no_reference.split(",")))
    for name in filter(None, args.cases.split(",")):
        xyz, charge, uhf = CASES[name]
        nums, pos = read_xyz(xyz, AATOAU)
        r = {"natoms": len(nums), "load_before": load_average(), "time": {}, "phases": {}, "asymmetry": {}, "accuracy": {},
             "peak_rss_mb": {}}
        print(f"== {name} ({len(nums)} atoms), load {r['load_before']}", flush=True)
        hs = {}
        for method in args.methods.split(","):
            for th in sorted({1, nt}):
                h, t, phases, asym = best(lambda: run_axtb(nums, pos, charge, uhf, method, th), args.repeat)
                r["time"][f"{method} {th}t"] = t
                r["phases"][f"{method} {th}t"] = phases
                r["peak_rss_mb"][f"{method} {th}t"] = LAST_PEAK_RSS_MB
                r["asymmetry"][method] = asym
                hs[method] = h
                print(f"   {method:9s} {th}t: {t:8.3f} s  (asymmetry before symmetrization {asym:.1e})", flush=True)
        r["load_after"] = load_average()
        if name not in no_ref:
            ref, spread = reference(nums, pos, charge, uhf, nt)
            r["reference_spread"] = spread
            rref = response_reference(nums, pos, charge, uhf, nt)
            r["references_difference"] = float(np.abs(ref - rref).max())
            fref = frequencies(ref, nums, pos, masses)
            for method, h in hs.items():
                f = frequencies(h, nums, pos, masses)
                r["accuracy"][method] = {"max_dH": float(np.abs(h - ref).max()), "max_dfreq": float(np.abs(f - fref).max()),
                                         "max_dH_vs_response_reference": float(np.abs(h - rref).max())}
                print(f"   {method:9s} vs tight reference: max|dH| {np.abs(h - ref).max():.2e} Eh/bohr2, "
                      f"max|dfreq| {np.abs(f - fref).max():.2e} cm-1; vs response reference {np.abs(h - rref).max():.2e}", flush=True)
        elif "analytic" in hs:
            for method, h in hs.items():
                if method != "analytic":
                    r["accuracy"][f"{method} vs analytic"] = {"max_dH": float(np.abs(h - hs["analytic"]).max())}
                    print(f"   {method:9s} vs analytic: max|dH| {np.abs(h - hs['analytic']).max():.2e}", flush=True)
        results["cases"][name] = r
        out_path.write_text(json.dumps(results, indent=1))
    results.setdefault("memory", {})
    for item in filter(None, args.memory.split(",")):
        name, th = item.rsplit(":", 1)
        th = int(th)
        xyz, charge, uhf = MEMORY_CASES[name]
        nums, pos = read_xyz(xyz, AATOAU)
        load = load_average()
        run_axtb(nums, pos, charge, uhf, "analytic", th, hessian=False)
        scf_rss = LAST_PEAK_RSS_MB
        _, wall, phases, _ = run_axtb(nums, pos, charge, uhf, "analytic", th)
        rec = {"natoms": len(nums), "threads": th, "load_before": load, "seconds": wall, "phases": phases,
               "peak_rss_mb": LAST_PEAK_RSS_MB, "peak_rss_mb_single_point": scf_rss}
        results["memory"][f"{name} {th}t"] = rec
        print(f"== memory {name} ({len(nums)} atoms) {th}t: peak RSS {LAST_PEAK_RSS_MB} MB (single point {scf_rss} MB), "
              f"{wall:.1f} s (load {load})", flush=True)
        out_path.write_text(json.dumps(results, indent=1))
    for name in filter(None, args.program.split(",")):
        xyz, charge, uhf = CASES[name]
        with tempfile.TemporaryDirectory(prefix="gxtb2-hess-") as d:
            d = Path(d)
            opt, hx, fx_printed, wall, settings = run_program(xyz, charge, uhf, d)
            nums, pos = read_xyz(opt, MCTC_AATOAU)
        p = tr_projector(pos)
        ha, ta, _, _ = run_axtb(nums, pos, charge, uhf, "analytic", 1, conventions="gxtb")
        # axtb's numerical Hessian with the program's step and displacement SCF accuracy
        hn, _, _, _ = run_axtb(nums, pos, charge, uhf, "numerical", nt, step=settings["step"],
                               acc=settings["scc_accuracy"], conventions="gxtb")
        # the same step with tight SCFs: the finite-difference error alone
        ht, _, _, _ = run_axtb(nums, pos, charge, uhf, "numerical", nt, step=settings["step"], acc=1e-6, conventions="gxtb")
        ref, _ = reference(nums, pos, charge, uhf, nt, conventions="gxtb")
        fa = frequencies(ha, nums, pos, masses)
        ft = frequencies(ht, nums, pos, masses)
        fx = frequencies(hx, nums, pos, masses)
        fr = frequencies(ref, nums, pos, masses)
        fn = frequencies(hn, nums, pos, masses)
        rec = {"program_settings": settings, "program_wall_ohess_1t": wall, "axtb_analytic_1t": ta,
               "max_dH_projected": float(np.abs(p @ ha @ p - hx).max()),
               "max_dfreq": float(np.abs(fa - fx).max()),
               "max_dH_projected_axtb_numerical_program_settings": float(np.abs(p @ hn @ p - hx).max()),
               "max_dfreq_axtb_numerical_program_settings": float(np.abs(fn - fx).max()),
               "max_dfreq_axtb_numerical_program_settings_vs_reference": float(np.abs(fn - fr).max()),
               "max_dfreq_axtb_numerical_program_step_tight_scf_vs_reference": float(np.abs(ft - fr).max()),
               "frequencies_program_printed": [float(x) for x in fx_printed],
               "max_dfreq_program_vs_reference": float(np.abs(fx - fr).max()),
               "max_dfreq_analytic_vs_reference": float(np.abs(fa - fr).max()),
               "max_dfreq_program_printed_vs_own_hessian": float(np.abs(np.sort(fx_printed[np.abs(fx_printed) > 1e-3]) - fx).max())
               if len(fx_printed[np.abs(fx_printed) > 1e-3]) == len(fx) else None,
               "frequencies_program": fx.tolist(), "frequencies_axtb": fa.tolist()}
        results["program"][name] = rec
        print(f"== program {name}: projected max|dH| {rec['max_dH_projected']:.2e}, max|dfreq| {rec['max_dfreq']:.3f} cm-1 "
              f"(program vs tight reference {rec['max_dfreq_program_vs_reference']:.3f}, axtb analytic vs reference "
              f"{rec['max_dfreq_analytic_vs_reference']:.1e}); axtb numerical with the program's settings vs program: "
              f"max|dH| {rec['max_dH_projected_axtb_numerical_program_settings']:.2e}, "
              f"max|dfreq| {rec['max_dfreq_axtb_numerical_program_settings']:.3f} (vs reference "
              f"{rec['max_dfreq_axtb_numerical_program_settings_vs_reference']:.3f}; same step, tight SCF "
              f"{rec['max_dfreq_axtb_numerical_program_step_tight_scf_vs_reference']:.3f})", flush=True)
        out_path.write_text(json.dumps(results, indent=1))
    for name in filter(None, args.versus.split(",")):
        xyz, charge, uhf = CASES[name]
        nums, pos = read_xyz(xyz, MCTC_AATOAU)
        rec = {"natoms": len(nums), "conventions": "gxtb (axtb) / program", "runs": [], "time": {}}
        print(f"== versus program: {name} ({len(nums)} atoms)", flush=True)
        hs = {}
        hx = None
        for th in sorted({1, nt}):
            load = load_average()
            hx, wall = run_program_hess(xyz, charge, uhf, th)
            rec["runs"].append({"what": f"program --hess {th}t", "load_before": load, "seconds": wall})
            rec["time"][f"program --hess {th}t"] = wall
            print(f"   program --hess {th}t: {wall:8.2f} s  (load {load})", flush=True)
            for method in args.methods.split(","):
                load = load_average()
                h, wall, phases, asym = run_axtb(nums, pos, charge, uhf, method, th, conventions="gxtb")
                rec["runs"].append({"what": f"axtb {method} {th}t", "load_before": load, "seconds": wall, "phases": phases})
                rec["time"][f"axtb {method} {th}t"] = wall
                hs[method] = h
                print(f"   axtb {method:9s} {th}t: {wall:8.2f} s  (load {load})", flush=True)
        p = tr_projector(pos)
        fx = frequencies(hx, nums, pos, masses)
        rec["frequencies_program"] = fx.tolist()
        rec["vs_program"] = {}
        for method, h in hs.items():
            f = frequencies(h, nums, pos, masses)
            d = np.abs(f - fx)
            rec["vs_program"][method] = {"max_dH_projected": float(np.abs(p @ h @ p - hx).max()), "max_dfreq": float(d.max()),
                                         "mean_dfreq": float(d.mean()), "max_dfreq_above_500": float(d[np.abs(fx) > 500].max())}
            print(f"   {method:9s} vs program: max|dH| {np.abs(p @ h @ p - hx).max():.2e}, max|dfreq| {d.max():.3f} cm-1 "
                  f"(mean {d.mean():.3f})", flush=True)
            if method == "analytic":
                rec["frequencies_axtb_analytic"] = f.tolist()
        results["versus_program"][name] = rec
        out_path.write_text(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
