#!/usr/bin/env python3
"""Final timings of axtb against the programs it replaces, at one thread count:

  gfn2     GFN2-xTB: xtb 6.8.0 and tblite 0.8.0 vs axtb (energy + gradient); xtb 6.8.0
           `--hess` (default, and `--acc 0.001`) vs axtb's analytic Hessian
  gxtb2    g-xTB 2.0.1: the program (`xtb coord --gxtb --grad`, `--hess`, `--hess --acc 0.001`;
           the g-xTB 2.0.1 build of xtb) vs axtb (method "gxtb-v2.0.1"), E+G and Hessians
  gxtb11   g-xTB v1.1: the g-xtb 1.1.0 program (`gxtb -c mol.xyz [-grad|-hess]`) vs axtb
           (method "gxtb-v1.1"); the program's gradient is numerical, so its energy alone
           is timed too; axtb's v1.1 Hessian is semi-numerical
  overhead the axtb CLI at two commits (AXTB_BASE_BIN, AXTB_BIN), plain E+G, interleaved

How it measures:
- Every calculation is a fresh process, one at a time, in a scratch directory of its own:
  the programs, the axtb CLI (JSON request on stdin), and Python processes for tblite and for
  axtb's Python module. `seconds` is the wall time of the process from start to exit;
  `inner_seconds` is, for tblite and axtb's Python module, the calculation inside the process
  (without Python start-up and imports), for the axtb CLI its own `seconds` and, for axtb
  Hessians, axtb's own Hessian time. Before the timings of a method, every program runs once
  on water unrecorded (binaries and libraries into the page cache).
- Programs and axtb alternate: per molecule, `rounds` rounds that run every program once, the
  order rotated from round to round. Summaries give min and median over the rounds.
- Every run records the node's 1-minute load before and after, the average number of busy
  hardware threads on the node while it ran (/proc/stat, node-wide inside a pod) and that
  number minus the run's own CPU use (other work on the node), its CPU time and `max_rss_mb`
  (wait4 rusage). `max_rss_mb` is NOT the program's peak memory: Linux carries the high-water
  RSS of the (v)forked driver into the child at exec, so it is max(driver's peak so far,
  program's peak), i.e. mostly the driver's (it grows with the Hessians it keeps).
- Threads: the programs get OMP_NUM_THREADS = MKL_NUM_THREADS = threads (tblite and the
  Python processes too, which keeps numpy's BLAS at that count); axtb gets the count in its
  request and reads no environment.
- Hessians come with their accuracy against a tight reference: axtb's analytic Hessian with
  SCF accuracy 1e-6 in the program's conventions, projected as that program projects its
  `hessian` file (xtb 6.8.0: mass-weighted; the 6.7.1-based g-xTB 2.0.1 build: Cartesian, about
  the geometric centre, after its tiny `rotmol` rotation), max |dH| (Eh/bohr²) and the largest
  harmonic-frequency difference; axtb's own Hessian against the same in its own conventions;
  and, for molecules up to `richardson_max_atoms`, the tight analytic Hessian against a
  Richardson-extrapolated finite-difference one ((4 H(0.001) - H(0.002)) / 3, tight SCFs).

Usage: final_timings.py --task task.json --out result.json [--scratch DIR]
  task.json: {"id", "parts": ["gfn2", ...], "threads": 1, "eg": [molecules], "hess": [molecules],
              "rounds_eg": 5, "rounds_hess": 3, "max_atoms": {"<program label>": N},
              "richardson_max_atoms": 62, "machine": "64-core node", "overhead_rounds": 10,
              "libs": {"MKL": "<libmkl_rt>", "AOCL": "<libflame>:<libblis-mt>"},
              "programs": ["<program label>", ...]}
  libs: the axtb CLI runs (E+G, both conventions) again with each run-time LAPACK library
  (AXTB_LAPACK, and AXTB_LAPACK_BLAS after the colon), labelled "... [MKL]"; programs: only
  these labels (default: all).
Environment: AXTB_BIN, AXTB_BASE_BIN (overhead), AXTB_XTB680, AXTB_GXTB2, AXTB_GXTB11_DIR (the
1.1.0 program and its parameter files; see tests/programs.py), AXTB_COMMIT (or AXTB_REV_FILE),
PYTHON (with axtb, tblite).
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get("AXTB_ROOT", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(ROOT / "tests"))
import programs  # noqa: E402

AXTB = os.environ.get("AXTB_BIN", str(ROOT / "target/release/axtb"))
AXTB_BASE = os.environ.get("AXTB_BASE_BIN")
XTB680 = programs.get(programs.XTB680)
GXTB2 = programs.get(programs.GXTB2)
GXTB11_DIR = programs.gxtb11_dir()
# the reference program of each part
PART_PROGRAMS = {"gfn2": programs.XTB680, "gxtb2": programs.GXTB2, "gxtb11": programs.GXTB11_DIR}
PY = os.environ.get("PYTHON", sys.executable)

SYMBOLS = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
           "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe").split()
AATOAU = 1.0 / 0.529177210544  # CODATA 2022; every program gets the same bohr coordinates
# xtb's atomic masses (src/param/atomicmass.f90) for the elements used here
MASSES = {1: 1.00794075, 6: 12.01073590, 7: 14.00670321, 8: 15.99940492, 26: 55.84514443}
AMU_TO_ME = 1.0 / 5.485799090441e-4
HARTREE_TO_CM = 2.1947463136314e5

METHODS = {"gfn2": "gfn2", "gxtb2": "gxtb-v2.0.1", "gxtb11": "gxtb-v1.1"}


# ------------------------------------------------------------------------------ molecules
def alkane(n: int):
    """All-trans CnH2n+2 built like tests/structures/large/c50h102.xyz (n = 50 is that file)."""
    atoms = []
    for i in range(n):
        x, y, s = i * 1.24946158, (0.0 if i % 2 == 0 else 0.88303214), (-1.0 if i % 2 == 0 else 1.0)
        atoms += [("C", x, y, 0.0), ("H", x, y + s * 0.51, 0.89), ("H", x, y + s * 0.51, -0.89)]
        if i == 0:
            atoms.append(("H", x - 1.03, y + s * 0.36, 0.0))
        if i == n - 1:
            atoms.append(("H", x + 1.03, y + s * 0.36, 0.0))
    return atoms


def read_xyz_atoms(path: Path):
    lines = path.read_text().splitlines()
    nat = int(lines[0].split()[0])
    return [(ln.split()[0].capitalize(), *map(float, ln.split()[1:4])) for ln in lines[2:2 + nat]]


MOLECULES = {
    "water": lambda: read_xyz_atoms(ROOT / "tests/structures/water.xyz"),
    "ferrocene": lambda: read_xyz_atoms(ROOT / "tests/structures/ferrocene.xyz"),
    "caffeine": lambda: read_xyz_atoms(ROOT / "tests/structures/caffeine.xyz"),
    "C20H42": lambda: alkane(20),
    "taxol": lambda: read_xyz_atoms(ROOT / "tests/structures/taxol.xyz"),
    "C50H102": lambda: read_xyz_atoms(ROOT / "tests/structures/large/c50h102.xyz"),
    "(H2O)100": lambda: read_xyz_atoms(ROOT / "tests/structures/large/water100.xyz"),
    "C100H202": lambda: alkane(100),
    "(H2O)200": lambda: read_xyz_atoms(ROOT / "tests/structures/large/water200.xyz"),
}
SOURCES = {"water": "tests/structures/water.xyz", "ferrocene": "tests/structures/ferrocene.xyz",
           "caffeine": "tests/structures/caffeine.xyz", "C20H42": "generated all-trans (bench/final_timings.py alkane)",
           "taxol": "tests/structures/taxol.xyz", "C50H102": "tests/structures/large/c50h102.xyz",
           "(H2O)100": "tests/structures/large/water100.xyz", "C100H202": "generated all-trans (bench/final_timings.py alkane)",
           "(H2O)200": "tests/structures/large/water200.xyz"}


class Mol:
    def __init__(self, name: str):
        atoms = MOLECULES[name]()
        self.name = name
        self.symbols = [a[0] for a in atoms]
        self.nums = np.array([SYMBOLS.index(s) + 1 for s in self.symbols])
        self.pos = np.array([a[1:4] for a in atoms]) * AATOAU
        self.natoms = len(atoms)
        self.nbf = {}

    def coord(self, pos=None) -> str:
        pos = self.pos if pos is None else pos
        lines = ["$coord"] + [f"{x:22.14f} {y:22.14f} {z:22.14f} {s.lower()}" for s, (x, y, z) in zip(self.symbols, pos)]
        return "\n".join(lines + ["$end", ""])

    def xyz(self) -> str:
        return "\n".join([str(self.natoms), self.name] +
                         [f"{s:<2} {x / AATOAU:18.12f} {y / AATOAU:18.12f} {z / AATOAU:18.12f}"
                          for s, (x, y, z) in zip(self.symbols, self.pos)]) + "\n"

    def request(self, method: str, threads: int, **kw) -> dict:
        return {"numbers": [int(n) for n in self.nums], "positions": self.pos.tolist(), "charge": 0.0, "uhf": None,
                "method": method, "threads": threads, **kw}


# ------------------------------------------------------------------------------ node state
NCPU = sum(1 for ln in open("/proc/stat") if re.match(r"cpu\d", ln))


def cpu_stat():
    with open("/proc/stat") as f:
        v = [int(x) for x in f.readline().split()[1:9]]
    return sum(v), v[3] + v[4]


def load1() -> float:
    return float(open("/proc/loadavg").read().split()[0])


def busy_threads(s0, s1) -> float | None:
    dt = s1[0] - s0[0]
    return None if dt <= 0 else round(NCPU * (dt - (s1[1] - s0[1])) / dt, 1)


# ------------------------------------------------------------------------------ running
class Failed(RuntimeError):
    pass


@functools.cache
def sha256_of(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


ENV_KEYS = ("AXTB_LAPACK", "AXTB_LAPACK_BLAS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")


def provenance(cmd: list, env: dict, pid: int) -> dict:
    """What ran: the command line (long inline scripts abbreviated), the binary's resolved
    path and the start of its sha256, the process id and the library/thread variables."""
    binary = str(Path(shutil.which(str(cmd[0])) or cmd[0]).resolve())
    return {"cmd": [a if len(str(a)) <= 200 else f"<{len(str(a))} characters>" for a in map(str, cmd)], "binary": binary,
            "binary_sha256": sha256_of(binary), "pid": pid, "env": {k: env[k] for k in ENV_KEYS if k in env}}


def run_process(cmd, env, files: dict, stdin_name: str | None, scratch: Path, keep=()):
    """Run cmd in a fresh directory holding `files`; returns timing record and kept outputs."""
    d = Path(tempfile.mkdtemp(dir=scratch))
    try:
        for name, content in files.items():
            if isinstance(content, Path):
                shutil.copy(content, d / name)
            else:
                (d / name).write_text(content)
        fin = open(d / stdin_name) if stdin_name else subprocess.DEVNULL
        fout, ferr = open(d / ".stdout", "w"), open(d / ".stderr", "w")
        l0, s0 = load1(), cpu_stat()
        t0 = time.perf_counter()
        p = subprocess.Popen(cmd, cwd=d, env=env, stdin=fin, stdout=fout, stderr=ferr)
        _, status, ru = os.wait4(p.pid, 0)
        wall = time.perf_counter() - t0
        s1, l1 = cpu_stat(), load1()
        p.returncode = os.waitstatus_to_exitcode(status)
        for f in (fout, ferr) + ((fin,) if stdin_name else ()):
            f.close()
        cpu = ru.ru_utime + ru.ru_stime
        node = busy_threads(s0, s1)
        rec = {"seconds": wall, "cpu_seconds": round(cpu, 4), "cores_used": round(cpu / wall, 2) if wall > 0 else None,
               "max_rss_mb": round(ru.ru_maxrss / 1024, 1), "load1_before": l0, "load1_after": l1,
               "node_busy_threads": node, "other_busy_threads": None if node is None else round(max(node - cpu / wall, 0.0), 1),
               "rc": p.returncode, **provenance(cmd, env, p.pid)}
        out = {"stdout": (d / ".stdout").read_text(errors="replace"), "stderr": (d / ".stderr").read_text(errors="replace")}
        for k in keep:
            out[k] = (d / k).read_text() if (d / k).exists() else None
        return rec, out
    finally:
        shutil.rmtree(d, ignore_errors=True)


def omp_env(threads: int, binary: str | None = None, home: str | None = None) -> dict:
    env = {k: v for k, v in os.environ.items() if k not in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")}
    env.update(OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OMP_STACKSIZE="1G")
    if binary:
        share = Path(binary).resolve().parent.parent / "share/xtb"
        if share.is_dir():
            env["XTBPATH"] = str(share)
    if home:
        env["HOME"] = home
    return env


def axtb_env(lib: str | None = None) -> dict:
    """axtb reads no thread variables; leave the pod's (OMP_NUM_THREADS=1) out of it anyway.
    `lib`: a run-time LAPACK library, "path" or "path:blas" (AXTB_LAPACK, AXTB_LAPACK_BLAS)."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "AXTB_LAPACK", "AXTB_LAPACK_BLAS")}
    if lib:
        path, _, blas = lib.partition(":")
        env["AXTB_LAPACK"] = path
        if blas:
            env["AXTB_LAPACK_BLAS"] = blas
        # the libraries' own dependencies (AOCL's libaoclutils) next to them
        dirs = dict.fromkeys(str(Path(x).parent) for x in (path, blas) if "/" in x)
        env["LD_LIBRARY_PATH"] = os.pathsep.join([*dirs, *filter(None, [env.get("LD_LIBRARY_PATH")])])
    return env


def parse_hessian_file(text: str, n3: int) -> np.ndarray:
    vals = [float(x) for x in text.split()[1:]]
    h = np.array(vals).reshape(n3, n3)
    return 0.5 * (h + h.T)


def xtb_like(binary: str, args: list, mol: Mol, threads: int, scratch: Path, hessian=False):
    """xtb 6.8.0 or the g-xTB 2.0.1 build of xtb, geometry as a Turbomole coord file (bohr)."""
    keep = ("hessian",) if hessian else ()
    rec, out = run_process([binary, "coord", *args], omp_env(threads, binary, home="."), {"coord": mol.coord()}, None,
                           scratch, keep)
    text = out["stdout"] + out["stderr"]
    if "normal termination of xtb" not in text or rec["rc"]:
        raise Failed(f"{Path(binary).name} {' '.join(args)} failed (rc {rec['rc']}): {text[-600:]}")
    e = re.findall(r"TOTAL ENERGY\s+(-?\d+\.\d+)", out["stdout"])
    rec["energy"] = float(e[-1]) if e else None
    it = re.findall(r"convergence criteria satisfied after\s+(\d+)\s+iterations", out["stdout"])
    if it:
        rec["iterations"] = int(it[0])
    h = parse_hessian_file(out["hessian"], 3 * mol.natoms) if hessian else None
    return rec, h


def gxtb11(args: list, mol: Mol, threads: int, scratch: Path, hessian=False):
    """The g-xtb 1.1.0 program: `gxtb -c mol.xyz [args]` with HOME=. and its parameter files."""
    files = {"mol.xyz": mol.xyz(), ".gxtb": GXTB11_DIR / "gxtb_parameters", ".basisq": GXTB11_DIR / "basisq",
             ".eeq": GXTB11_DIR / "eeq"}
    rec, out = run_process([str(GXTB11_DIR / "gxtb"), "-c", "mol.xyz", *args], omp_env(threads, home="."), files, None,
                           scratch)
    m = re.findall(r"^\s*total\s+(-?\d+\.\d+)", out["stdout"], re.M)
    if rec["rc"] or not m:
        raise Failed(f"gxtb 1.1.0 {' '.join(args)} failed (rc {rec['rc']}): {(out['stdout'] + out['stderr'])[-600:]}")
    rec["energy"] = float(m[-1])
    h = None
    if hessian:
        # the program writes no `hessian` file; its frequencies (cm-1) from the output
        block = out["stdout"].split("vibrational frequencies (cm-1)")[-1].split("IR intensities")[0]
        f = np.array([float(x) for ln in block.splitlines() if ln.strip().startswith("eigval")
                      for x in ln.split(":")[1].split()])
        if len(f) != 3 * mol.natoms:
            raise Failed(f"gxtb 1.1.0 -hess: {len(f)} frequencies for {mol.natoms} atoms")
        h = {"freqs": np.sort(f[np.sort(np.argsort(np.abs(f))[6:])])}
    return rec, h


def axtb_cli(req: dict, scratch: Path, binary: str | None = None, lib: str | None = None):
    rec, out = run_process([binary or AXTB, "--json"], axtb_env(lib), {"req.json": json.dumps(req)}, "req.json", scratch)
    if rec["rc"]:
        raise Failed(f"axtb failed (rc {rec['rc']}): {out['stdout'][-600:]} {out['stderr'][-600:]}")
    if "AXTB_LAPACK not used" in out["stderr"]:
        raise Failed(out["stderr"][-600:])
    d = json.loads(out["stdout"])
    rec["request"] = {k: req[k] for k in ("method", "threads", "gradient", "conventions", "hessian") if k in req}
    rec["energy"] = d["energy"]
    rec["iterations"] = d.get("iterations")
    rec["phases"] = dict(d.get("timings", {}))
    rec["inner_seconds"] = d.get("seconds")
    h = None
    if "hessian" in d:
        h = np.array(d["hessian"])
        ht = dict(d.get("hessian_timings") or {})
        rec["hessian_phases"] = ht
        rec["inner_seconds"] = ht.get("total")
    return rec, h, d


TBLITE = r"""
import json, sys, time, numpy as np
from tblite.interface import Calculator
d = json.load(sys.stdin)
t = time.perf_counter()
c = Calculator("GFN2-xTB", np.array(d["numbers"]), np.array(d["positions"]))
c.set("verbosity", 0)
r = c.singlepoint()
g = r.get("gradient")
print(json.dumps({"inner": time.perf_counter() - t, "energy": float(r.get("energy"))}))
"""

AXTB_PY = r"""
import json, sys, time, numpy as np
import axtb
d = json.load(sys.stdin)
kw = {"conventions": d["conventions"]} if d.get("conventions") else {}
t = time.perf_counter()
r = axtb.singlepoint(np.array(d["numbers"]), np.array(d["positions"]), charge=0.0, uhf=None, method=d["method"],
                     gradient=True, threads=d["threads"], **kw)
print(json.dumps({"inner": time.perf_counter() - t, "energy": float(r["energy"]), "iterations": int(r["iterations"])}))
"""


def python_snippet(code: str, req: dict, env: dict, scratch: Path):
    rec, out = run_process([PY, "-c", code], env, {"req.json": json.dumps(req)}, "req.json", scratch)
    if rec["rc"]:
        raise Failed(f"python snippet failed: {out['stderr'][-600:]}")
    d = json.loads(out["stdout"].strip().splitlines()[-1])
    rec["inner_seconds"] = d["inner"]
    rec["energy"] = d["energy"]
    if "iterations" in d:
        rec["iterations"] = d["iterations"]
    return rec


# ------------------------------------------------------------------------------ programs
def eg_programs(part: str, mol: Mol, t: int, scratch: Path, libs: dict | None = None, only: list | None = None):
    """label -> callable returning a run record, for the energy + gradient timings; with `libs`
    the axtb CLI runs again with each run-time LAPACK library, with `only` just those labels."""
    progs = eg_programs_faer(part, mol, t, scratch)
    for name, lib in (libs or {}).items():
        for label in [k for k in progs if k.startswith("axtb CLI")]:
            conv = {"conventions": "xtb"} if label.endswith("xtb conventions") else {}
            progs[f"{label} [{name}]"] = (lambda conv=conv, lib=lib: axtb_cli(mol.request(METHODS[part], t, gradient=True, **conv),
                                                                             scratch, lib=lib)[0])
    return {k: f for k, f in progs.items() if only is None or k in only}


def eg_programs_faer(part: str, mol: Mol, t: int, scratch: Path):
    m = METHODS.get(part)
    if part == "gfn2":
        return {
            "xtb 6.8.0": lambda: xtb_like(XTB680, ["--gfn", "2", "--grad"], mol, t, scratch)[0],
            "tblite 0.8.0": lambda: python_snippet(TBLITE, mol.request(m, t), omp_env(t), scratch),
            "axtb CLI": lambda: axtb_cli(mol.request(m, t, gradient=True), scratch)[0],
            "axtb CLI, xtb conventions": lambda: axtb_cli(mol.request(m, t, gradient=True, conventions="xtb"), scratch)[0],
            "axtb Python": lambda: python_snippet(AXTB_PY, mol.request(m, t), omp_env(t), scratch),
            "axtb Python, xtb conventions": lambda: python_snippet(AXTB_PY, mol.request(m, t, conventions="xtb"), omp_env(t), scratch),
        }
    if part == "gxtb2":
        return {
            "g-xTB 2.0.1 program": lambda: xtb_like(GXTB2, ["--gxtb", "--grad"], mol, t, scratch)[0],
            "axtb CLI": lambda: axtb_cli(mol.request(m, t, gradient=True), scratch)[0],
            "axtb Python": lambda: python_snippet(AXTB_PY, mol.request(m, t), axtb_env(), scratch),
        }
    if part == "gxtb11":
        return {
            "g-xtb 1.1.0 program": lambda: gxtb11(["-grad"], mol, t, scratch)[0],
            "g-xtb 1.1.0 program, energy only": lambda: gxtb11([], mol, t, scratch)[0],
            "axtb CLI": lambda: axtb_cli(mol.request(m, t, gradient=True), scratch)[0],
            "axtb CLI, energy only": lambda: axtb_cli(mol.request(m, t, gradient=False), scratch)[0],
            "axtb Python": lambda: python_snippet(AXTB_PY, mol.request(m, t), axtb_env(), scratch),
        }
    raise ValueError(part)


def hess_programs(part: str, mol: Mol, t: int, scratch: Path):
    """label -> callable returning (run record, Hessian) for the Hessian timings."""
    m = METHODS[part]
    if part == "gfn2":
        return {
            "xtb 6.8.0 --hess": lambda: xtb_like(XTB680, ["--gfn", "2", "--hess"], mol, t, scratch, hessian=True),
            "xtb 6.8.0 --hess --acc 0.001": lambda: xtb_like(XTB680, ["--gfn", "2", "--hess", "--acc", "0.001"], mol, t, scratch,
                                                             hessian=True),
            "axtb analytic": lambda: axtb_cli(mol.request(m, t, hessian=True), scratch)[:2],
        }
    if part == "gxtb2":
        return {
            "g-xTB 2.0.1 program --hess": lambda: xtb_like(GXTB2, ["--gxtb", "--hess"], mol, t, scratch, hessian=True),
            "g-xTB 2.0.1 program --hess --acc 0.001": lambda: xtb_like(GXTB2, ["--gxtb", "--hess", "--acc", "0.001"], mol, t,
                                                                       scratch, hessian=True),
            "axtb analytic": lambda: axtb_cli(mol.request(m, t, hessian=True), scratch)[:2],
        }
    if part == "gxtb11":
        return {
            "g-xtb 1.1.0 program -hess": lambda: gxtb11(["-hess"], mol, t, scratch, hessian=True),
            "axtb numerical": lambda: axtb_cli(mol.request(m, t, hessian=True), scratch)[:2],
        }
    raise ValueError(part)


# ------------------------------------------------------------------------------ Hessian analysis
def rotmol(pos, xrot=1e-4, yrot=2e-4, zrot=3e-4):
    """xtb 6.7.1's `rotmol` (hessian.F90), kept by the g-xTB 2.0.1 build: tiny rotations
    (degrees) of the molecule before `--hess`; xtb 6.8.0 removed it."""
    p = np.array(pos, dtype=float)
    for ax, (i, j) in ((xrot, (1, 2)), (yrot, (0, 2)), (zrot, (0, 1))):
        a = ax * math.pi / 180.0
        xo, yo = p[:, i].copy(), p[:, j].copy()
        p[:, i] = xo * math.cos(a) + yo * math.sin(a)
        p[:, j] = -xo * math.sin(a) + yo * math.cos(a)
    return p


def masses(nums) -> np.ndarray:
    return np.array([MASSES[int(z)] for z in nums])


def proj_mw(h, nums, pos):
    """xtb 6.8.0's `hessian` file: translations and rotations (about the centre of mass)
    projected in mass-weighted coordinates, transformed back."""
    m = masses(nums)
    c = pos - (m[:, None] * pos).sum(axis=0) / m.sum()
    sq = np.repeat(np.sqrt(m), 3)
    vecs = []
    for k in range(3):
        t = np.zeros((len(nums), 3))
        t[:, k] = 1.0
        vecs.append(t.ravel() * sq)
    for k in range(3):
        e = np.zeros(3)
        e[k] = 1.0
        vecs.append(np.cross(e, c).ravel() * sq)
    q, r = np.linalg.qr(np.array(vecs).T)
    q = q[:, np.abs(np.diag(r)) > 1e-8 * np.abs(np.diag(r)).max()]
    pr = np.eye(3 * len(nums)) - q @ q.T
    return (pr @ (h / np.outer(sq, sq)) @ pr) * np.outer(sq, sq)


def proj_cart(h, pos):
    """xtb 6.7.1's (and the g-xTB 2.0.1 build's) `hessian` file: translations and rotations
    about the geometric centre projected from the Cartesian Hessian."""
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
    p = np.eye(3 * n) - q @ q.T
    return p @ h @ p


def freqs(hproj, nums) -> np.ndarray:
    """Harmonic frequencies (cm-1, imaginary negative) of a projected Hessian without its six
    zero modes, ascending."""
    m = np.repeat(masses(nums) * AMU_TO_ME, 3)
    w = np.linalg.eigvalsh(0.5 * (hproj + hproj.T) / np.sqrt(np.outer(m, m)))
    f = np.sign(w) * np.sqrt(np.abs(w)) * HARTREE_TO_CM
    keep = np.sort(np.argsort(np.abs(f))[6:])
    return np.sort(f[keep])


def compare(h, ref, nums) -> dict:
    """h and ref projected the same way: max |dH| and frequency differences."""
    fa, fr = freqs(h, nums), freqs(ref, nums)
    d = np.abs(fa - fr)
    hi = np.abs(fr) > 100.0
    return {"max_dH": float(np.abs(h - ref).max()), "max_dfreq": float(d.max()), "mean_dfreq": float(d.mean()),
            "max_dfreq_above_100": float(d[hi].max()) if hi.any() else None,
            "lowest_freqs": [round(float(x), 2) for x in fa[:3]]}


def hessian_accuracy(part: str, mol: Mol, first: dict, t: int, scratch: Path, richardson: bool) -> dict:
    """Each timed Hessian (first round) against a tight reference."""
    m = METHODS[part]
    nums, pos = mol.nums, mol.pos
    acc = {}

    def cli_h(conv=None, method="analytic", acc_=None, step=None, p=None):
        req = Mol.request(mol, m, t, hessian=True, hessian_method=method, vibrations=False)
        if conv:
            req["conventions"] = conv
        if acc_ is not None:
            req["hessian_accuracy"] = acc_
        if step is not None:
            req["hessian_step"] = step
        if p is not None:
            req["positions"] = p.tolist()
        return axtb_cli(req, scratch)[1]

    if part == "gxtb11":
        def fd_ref(conv):
            n2 = cli_h(conv=conv, method="numerical", step=0.001, acc_=1e-8)
            if not richardson:
                return n2
            n1 = cli_h(conv=conv, method="numerical", step=0.002, acc_=1e-8)
            return (4.0 * n2 - n1) / 3.0
        acc["reference"] = ("axtb v1.1 numerical Hessian, " + ("Richardson (4 H(0.001) - H(0.002))/3" if richardson else "step 0.001 bohr")
                            + ", SCF accuracy 1e-8; in the program's conventions ('gxtb') for the program, in axtb's default "
                            "conventions for axtb's")
        f_prog = freqs(proj_mw(fd_ref("gxtb"), nums, pos), nums)
        r_axtb = proj_mw(fd_ref(None), nums, pos)
        for label, h in first.items():
            if isinstance(h, dict):  # the program's printed frequencies (its own masses and projection)
                d = np.abs(h["freqs"] - f_prog)
                acc[label] = {"max_dfreq": float(d.max()), "mean_dfreq": float(d.mean()),
                              "lowest_freqs": [round(float(x), 2) for x in h["freqs"][:3]],
                              "note": "printed frequencies (no hessian file; step 0.01 bohr) vs the reference's (xtb masses)"}
            elif h is not None:
                acc[label] = compare(proj_mw(h, nums, pos), r_axtb, nums)
        return acc

    prog_conv = "xtb" if part == "gfn2" else "gxtb"
    ref_axtb = cli_h(acc_=1e-6)
    acc["reference"] = (f"axtb analytic Hessian, SCF accuracy 1e-6, in the program's conventions ('{prog_conv}') for the "
                        "program's Hessians, in axtb's default conventions for axtb's")
    project = lambda h: proj_mw(h, nums, pos)  # noqa: E731
    ref_prog = None
    if any(h is not None and not label.startswith("axtb") for label, h in first.items()):
        if part == "gfn2":
            ref_prog = proj_mw(cli_h(conv=prog_conv, acc_=1e-6), nums, pos)
        else:
            # the 6.7.1-based build rotates the molecule slightly (rotmol) and projects Cartesian
            prot = rotmol(pos)
            ref_prog = proj_cart(cli_h(conv=prog_conv, acc_=1e-6, p=prot), prot)
    for label, h in first.items():
        if h is None:
            continue
        if label.startswith("axtb"):
            acc[label] = compare(project(h), project(ref_axtb), nums)
        else:
            acc[label] = compare(h, ref_prog, nums)
    if richardson:
        n1 = cli_h(method="numerical", step=0.002, acc_=1e-8 if part == "gfn2" else 1e-6)
        n2 = cli_h(method="numerical", step=0.001, acc_=1e-8 if part == "gfn2" else 1e-6)
        rich = (4.0 * n2 - n1) / 3.0
        acc["tight analytic vs Richardson finite differences"] = compare(project(ref_axtb), project(rich), nums)
        acc["tight analytic vs Richardson finite differences"]["fd_spread_0.002_0.001"] = float(np.abs(n1 - n2).max())
    return acc


# ------------------------------------------------------------------------------ driver
def check_columns(runs: list) -> dict:
    """Each program label (column) must have run one binary with one library setting, the
    programs and axtb must be different binaries, every run its own process, and the library
    variable must be set exactly for the "[...]" columns. Returns the columns and the checks."""
    cols, pids = {}, []
    for r in runs:
        if not r.get("ok") or r["part"] == "overhead":
            continue
        c = cols.setdefault(f"{r['part']} | {r['program']}", {"binaries": set(), "cmd0": set(), "env": set(), "request": set()})
        c["binaries"].add((r["binary"], r["binary_sha256"]))
        c["cmd0"].add(" ".join(x for x in r["cmd"][1:] if not x.startswith("<")))
        c["env"].add(json.dumps({k: v for k, v in r["env"].items() if k.startswith("AXTB")}, sort_keys=True))
        c["request"].add(json.dumps(r.get("request"), sort_keys=True))
        pids.append(r["pid"])
    one = all(len(c["binaries"]) == 1 and len(c["env"]) == 1 and len(c["request"]) == 1 for c in cols.values())
    prog_bins = {b for k, c in cols.items() if " | axtb" not in k for b in c["binaries"]}
    axtb_bins = {b for k, c in cols.items() if " | axtb" in k for b in c["binaries"]}
    lib_ok = all((("[" in k) == ("AXTB_LAPACK" in next(iter(c["env"])))) for k, c in cols.items())
    checks = {"one_binary_env_request_per_column": one, "programs_and_axtb_different_binaries": not (prog_bins & axtb_bins),
              "every_run_its_own_process": len(pids) == len(set(pids)), "lapack_variable_only_in_library_columns": lib_ok}
    columns = {k: {"binary": sorted(c["binaries"]), "args": sorted(c["cmd0"]), "env": sorted(c["env"]), "request": sorted(c["request"])}
               for k, c in cols.items()}
    return {"columns": columns, "checks": checks}


def summarize(runs: list) -> list:
    groups = {}
    for r in runs:
        if r.get("ok"):
            groups.setdefault((r["part"], r["kind"], r["molecule"], r["program"]), []).append(r)
    out = []
    for (part, kind, mol, prog), rs in groups.items():
        s = [r["seconds"] for r in rs]
        row = {"part": part, "kind": kind, "molecule": mol, "atoms": rs[0]["atoms"], "nbf": rs[0].get("nbf"),
               "threads": rs[0]["threads"], "program": prog, "n": len(s), "min": min(s), "median": statistics.median(s),
               "max": max(s), "median_cores_used": statistics.median(r["cores_used"] or 0 for r in rs),
               "median_max_rss_mb": statistics.median(r["max_rss_mb"] for r in rs),
               "median_other_busy_threads": statistics.median(r["other_busy_threads"] or 0 for r in rs),
               "max_load1": max(max(r["load1_before"], r["load1_after"]) for r in rs)}
        inner = [r["inner_seconds"] for r in rs if r.get("inner_seconds") is not None]
        if inner:
            row["inner_min"], row["inner_median"] = min(inner), statistics.median(inner)
        for key in ("phases", "hessian_phases"):
            ph = [r[key] for r in rs if r.get(key)]
            if ph:
                row[f"{key}_median"] = {k: statistics.median(p.get(k, 0.0) for p in ph) for k in ph[0]}
        its = {r.get("iterations") for r in rs if r.get("iterations") is not None}
        if its:
            row["iterations"] = sorted(its)
        out.append(row)
    return out


def speedups(summary: list) -> list:
    """Reference program median / axtb median (and min / min) per molecule."""
    axtb_of = {"E+G": "axtb CLI", "Hessian": None}
    by = {}
    for r in summary:
        by.setdefault((r["part"], r["kind"], r["molecule"]), {})[r["program"]] = r
    rows = []
    for (part, kind, mol), progs in by.items():
        ax = [p for p in progs if p.startswith("axtb")]
        refs = [p for p in progs if not p.startswith("axtb")]
        if kind == "E+G":  # every axtb CLI run (conventions, LAPACK libraries)
            axrefs = [p for p in ax if p.startswith(axtb_of[kind])]
        else:
            axrefs = [next((p for p in ax if p in ("axtb analytic", "axtb numerical")), None)]
        for axref in axrefs:
            if axref not in progs:
                continue
            a = progs[axref]
            for p in refs:
                b = progs[p]
                rows.append({"part": part, "kind": kind, "molecule": mol, "atoms": a["atoms"], "threads": a["threads"],
                             "program": p, "axtb": axref, "program_median": b["median"], "axtb_median": a["median"],
                             "speedup_median": b["median"] / a["median"], "speedup_min": b["min"] / a["min"]})
    return rows


def versions() -> dict:
    v = {}
    for label, b in (("xtb 6.8.0", XTB680), ("g-xTB 2.0.1 build of xtb", GXTB2)):
        try:
            out = subprocess.run([b, "--version"], capture_output=True, text=True, env=omp_env(1, b)).stdout
            m = re.search(r"xtb version (\S+) \((\w+)\)", out)
            v[label] = f"xtb {m.group(1)} ({m.group(2)[:7]})" if m else "?"
        except (OSError, TypeError):  # TypeError: not there (None)
            v[label] = None
    v["g-xtb 1.1.0"] = "g-xTB 1.1.0 (grimme-lab/g-xtb v1.1.0 release binary)" if GXTB11_DIR else None
    try:
        v["tblite"] = subprocess.run([PY, "-c", "import importlib.metadata as m; print(m.version('tblite'))"], capture_output=True, text=True).stdout.strip()
        v["axtb (Python)"] = subprocess.run([PY, "-c", "import axtb; print(axtb.__version__)"], capture_output=True, text=True).stdout.strip()
    except OSError:
        pass
    return v


def axtb_commit() -> str | None:
    if os.environ.get("AXTB_COMMIT"):
        return os.environ["AXTB_COMMIT"]
    f = os.environ.get("AXTB_REV_FILE")
    if f and Path(f).exists():
        return Path(f).read_text().split()[1][:12]
    r = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short=12", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() or None


def lscpu(key: str) -> str:
    out = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    m = re.search(rf"^{re.escape(key)}:\s*(.+)$", out, re.M)
    return m.group(1).strip() if m else ""


def count_nbf(molden: str) -> int:
    sec = molden.split("[GTO]")[1].split("[MO]")[0]
    return sum(2 * "spdfg".index(m.group(1)) + 1 for m in re.finditer(r"^\s*([spdfg])\s+\d+\s+1\.0", sec, re.M))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scratch", type=Path, default=None)
    a = ap.parse_args()
    task = json.loads(a.task.read_text())
    if not task.get("programs"):  # all programs of the task's parts
        programs.require(*(PART_PROGRAMS[p] for p in task.get("parts", []) if p in PART_PROGRAMS))
    t = int(task["threads"])
    scratch = a.scratch or Path(tempfile.mkdtemp())
    scratch.mkdir(parents=True, exist_ok=True)
    max_atoms = task.get("max_atoms", {})
    res = {"meta": {"task": task.get("id"), "machine": task.get("machine"), "cpu": lscpu("Model name"),
                    "cores": int(lscpu("Core(s) per socket") or 0) * int(lscpu("Socket(s)") or 0),
                    "threads_per_core": int(lscpu("Thread(s) per core") or 0), "hw_threads_visible": NCPU,
                    "pod_cpu_quota": (open("/sys/fs/cgroup/cpu.max").read().strip() if Path("/sys/fs/cgroup/cpu.max").exists() else None),
                    "threads": t, "axtb_commit": axtb_commit(), "versions": versions(),
                    "date": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()), "load1_start": load1(),
                    "rounds_eg": task.get("rounds_eg", 5), "rounds_hess": task.get("rounds_hess", 3),
                    "how": __doc__.split("How it measures:")[1].split("Usage:")[0].strip()},
           "molecules": {}, "runs": [], "hessian_accuracy": {}, "summary": [], "speedups": []}
    mols = {}

    def mol(name):
        if name not in mols:
            mols[name] = Mol(name)
            res["molecules"][name] = {"atoms": mols[name].natoms, "source": SOURCES[name], "nbf": {}}
        return mols[name]

    def nbf(part, mo):
        if part not in mo.nbf:
            try:
                _, _, d = axtb_cli(mo.request(METHODS[part], 1, molden=True), scratch)
                mo.nbf[part] = count_nbf(d["molden"])
            except Failed:
                mo.nbf[part] = None
            res["molecules"][mo.name]["nbf"][METHODS[part]] = mo.nbf[part]
        return mo.nbf[part]

    def save():
        res["summary"] = summarize(res["runs"])
        res["speedups"] = speedups(res["summary"])
        res["meta"]["harness_check"] = check_columns(res["runs"])
        res["meta"]["load1_end"] = load1()
        a.out.write_text(json.dumps(res, indent=1))

    def record(part, kind, mo, label, rnd, fn, meth=None):
        meth = meth or part
        base = {"part": part, "kind": kind, "method": METHODS[meth], "molecule": mo.name, "atoms": mo.natoms,
                "nbf": mo.nbf.get(meth), "threads": t, "program": label, "round": rnd}
        try:
            out = fn()
        except Exception as e:  # noqa: BLE001 - a failed run is recorded, the task goes on
            res["runs"].append({**base, "ok": False, "error": str(e)[-800:]})
            print(f"   FAILED {label}: {str(e)[:300]}", flush=True)
            return None
        rec, h = out if isinstance(out, tuple) else (out, None)
        res["runs"].append({**base, "ok": True, **rec})
        print(f"   r{rnd} {label:40s} {rec['seconds']:10.3f} s  cores {rec['cores_used']}  other {rec['other_busy_threads']}"
              f"  load {rec['load1_before']}", flush=True)
        return h

    def warm_up(progs):
        """One unrecorded run of each program (water): binaries and libraries into the page cache."""
        for label, fn in progs.items():
            try:
                fn()
            except Exception as e:  # noqa: BLE001
                print(f"   warm-up {label} failed: {str(e)[:200]}", flush=True)

    for part in task.get("parts", []):
        if part == "overhead":
            continue
        libs, only = task.get("libs"), task.get("programs")
        warm_up({k: f for k, f in eg_programs(part, mol("water"), t, scratch, libs, only).items() if max_atoms.get(k, 10 ** 9) >= 3})
        for name in task.get("eg", []):
            mo = mol(name)
            nbf(part, mo)
            progs = {k: f for k, f in eg_programs(part, mo, t, scratch, libs, only).items() if mo.natoms <= max_atoms.get(k, 10 ** 9)}
            labels = list(progs)
            print(f"== {part} E+G {name} ({mo.natoms} atoms, {mo.nbf[part]} basis functions), {t} threads", flush=True)
            for rnd in range(task.get("rounds_eg", 5)):
                k = rnd % len(labels)
                for label in labels[k:] + labels[:k]:
                    record(part, "E+G", mo, label, rnd, progs[label])
            save()
        for name in task.get("hess", []):
            mo = mol(name)
            nbf(part, mo)
            progs = {k: f for k, f in hess_programs(part, mo, t, scratch).items() if mo.natoms <= max_atoms.get(k, 10 ** 9)}
            labels = list(progs)
            print(f"== {part} Hessian {name} ({mo.natoms} atoms, {mo.nbf[part]} basis functions), {t} threads", flush=True)
            first = {}
            for rnd in range(task.get("rounds_hess", 3)):
                k = rnd % len(labels)
                for label in labels[k:] + labels[:k]:
                    h = record(part, "Hessian", mo, label, rnd, progs[label])
                    if label not in first and h is not None:
                        first[label] = h
            save()
            if task.get("accuracy", True):
                try:
                    acc = hessian_accuracy(part, mo, first, t, scratch, mo.natoms <= task.get("richardson_max_atoms", 0))
                except Exception as e:  # noqa: BLE001
                    acc = {"error": str(e)[-800:]}
                res["hessian_accuracy"].setdefault(part, {})[name] = acc
                print("   accuracy: " + json.dumps({k: (v if isinstance(v, str) else {kk: vv for kk, vv in v.items() if kk in ("max_dH", "max_dfreq")})
                                                     for k, v in acc.items()}), flush=True)
                save()

    if "overhead" in task.get("parts", []):
        # the axtb CLI at two commits, plain energy + gradient, same molecule set
        bins = {"base": AXTB_BASE, "new": AXTB}
        res["meta"]["overhead_binaries"] = {"base": task.get("base_commit"), "new": res["meta"]["axtb_commit"]}
        w = mol("water")
        warm_up({k: (lambda b=b: axtb_cli(w.request("gfn2", t, gradient=True), scratch, b)) for k, b in bins.items()})
        for meth in task.get("overhead_methods", ["gfn2", "gxtb2"]):
            for name in task.get("overhead_molecules", task.get("eg", [])):
                mo = mol(name)
                nbf(meth, mo)
                print(f"== overhead {meth} {name}", flush=True)
                for rnd in range(task.get("overhead_rounds", 10)):
                    order = ["base", "new"] if rnd % 2 == 0 else ["new", "base"]
                    for which in order:
                        record("overhead", "E+G", mo, f"axtb CLI {which} {METHODS[meth]}", rnd,
                               lambda w=which: axtb_cli(mo.request(METHODS[meth], t, gradient=True), scratch, bins[w])[0], meth)
                save()
    save()
    hc = res["meta"]["harness_check"]
    for k, c in hc["columns"].items():
        print(f"column {k}: {c['binary']} args {c['args']} env {c['env']} request {c['request']}", flush=True)
    print("harness checks:", json.dumps(hc["checks"]), flush=True)
    print("wrote", a.out, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
