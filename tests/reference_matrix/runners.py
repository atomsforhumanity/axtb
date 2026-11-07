"""Runners for the three programs of the reference matrix.

* xtb (`run.py --xtb`, default xtb 6.8.0 from `$AXTB_XTB680`, see tests/programs.py; 6.8.0 is
  compared with axtb's `xtb` conventions, 6.7.1 with `xtb-6.7.1`, chosen by the binary's
  version; `$AXTB_XTB_CONVENTIONS` overrides): geometry as a Turbomole `coord` file in bohr, one scratch
  directory per run, OMP_NUM_THREADS=1, every flag and value as separate argv elements.
  Energy from the `TOTAL ENERGY` line (12 decimals), gradient from the `gradient` file,
  charges from `charges`, dipole from `xtbout.json` (`--json`), converged shell charges and
  atomic multipoles from the binary `xtbrestart`.
* tblite 0.8.0 (Python wheel), always in a child process: tblite aborts the interpreter on
  some inputs ("double free or corruption" after impossible charges), so it must not run
  inside the matrix process. `python runners.py tblite` reads one JSON request on stdin.
* axtb, in process through the Python API (`axtb.singlepoint` / `axtb.Calculator`).

Every runner returns a dict with `ok` (bool) and either results or `error`.
"""

from __future__ import annotations

import json
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PYTHON = str(ROOT / ".venv/bin/python") if (ROOT / ".venv/bin/python").exists() else sys.executable
# xtb's (and tblite's) Boltzmann constant, Eh/K (xtb_mctc_constants::kB)
KB_XTB = 3.166808578545117e-06
XTB_MIN_ACC = 1.0e-4

from structures import SYMBOLS  # noqa: E402

sys.path.insert(0, str(HERE.parent))
import programs  # noqa: E402  (tests/programs.py: the external programs)


def scratch_root() -> Path:
    base = os.environ.get("AXTB_SCRATCH") or os.environ.get("TMPDIR") or tempfile.gettempdir()
    p = Path(base) / "axtb-reference-matrix"
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------------------
# xtb

# the xtb binary and the axtb conventions that reproduce it: `--xtb` of the scripts (passed on to
# their worker processes as AXTB_MATRIX_XTB), else xtb 6.8.0 ($AXTB_XTB680); None: no xtb
XTB = os.environ.get("AXTB_MATRIX_XTB") or programs.get(programs.XTB680)


def xtb_version(binary: str | None = None) -> str:
    """`6.8.0 (4b8c987)`-style version of an xtb binary ("?" if it does not run)."""
    try:
        v = subprocess.run([binary or XTB, "--version"], capture_output=True, text=True,
                           env=dict(os.environ, OMP_NUM_THREADS="1")).stdout
    except (OSError, TypeError):  # TypeError: no xtb (None)
        return "?"
    m = re.search(r"xtb version (\S+) \((\w+)\)", v)
    return f"{m.group(1)} ({m.group(2)[:7]})" if m else "?"


def xtb_conventions(binary: str | None = None) -> str:
    """axtb's conventions for an xtb binary: `xtb` for 6.8.x, `xtb-6.7.1` for 6.7.x."""
    if os.environ.get("AXTB_XTB_CONVENTIONS"):
        return os.environ["AXTB_XTB_CONVENTIONS"]
    return "xtb-6.7.1" if xtb_version(binary).startswith("6.7.") else "xtb"


XTB_CONVENTIONS = xtb_conventions()
# the 6.7.1 release itself (the tag and the commit of the release binaries), not the g-xTB
# 2.0.1 build of 6.7.1 (26dd68d, 127 commits later) that the xtb-6.7.1 conventions reproduce
XTB_671_RELEASE = xtb_version().endswith(("(edcfbbe)", "(26b2801)"))
# another xtb version (`run.py --xtb-other`, passed on as AXTB_MATRIX_XTB_OTHER; optional): its
# converged moments are one more start
# of the different-SCF-solution diagnoses
XTB_OTHER = os.environ.get("AXTB_MATRIX_XTB_OTHER") or None
# xtb 6.7.1 rotates the molecule slightly before a numerical Hessian (`rotmol` in
# hessian.F90) and runs its displacement SCFs at `$hess sccacc` (default 0.3) whatever `--acc`
# says; 6.8.0 does neither: `--acc` (default 0.2 for --hess) reaches the displacements unless
# `$hess sccacc` is given (xtb PR #1500)
XTB_HESS_ROTATES = XTB_CONVENTIONS == "xtb-6.7.1"


def xtb_hess_displacement_acc(acc: float, sccacc: float | None = None) -> float:
    """SCF accuracy of the displacements of `xtb --hess --acc acc` (with `$hess sccacc`)."""
    if sccacc is not None:
        return sccacc
    return 0.3 if XTB_CONVENTIONS == "xtb-6.7.1" else max(acc, XTB_MIN_ACC)


def scrub(text: str, *paths) -> str:
    """`text` with the given paths (and the home directory) replaced by placeholders."""
    for p, name in zip(paths, ("xtb", "<scratch>")):
        text = text.replace(str(p), name)
    return text.replace(str(Path.home()), "~")


def write_coord(path: Path, nums, pos):
    lines = ["$coord"] + [f"{x:24.16f} {y:24.16f} {z:24.16f} {SYMBOLS[n - 1].lower()}"
                          for n, (x, y, z) in zip(nums, pos)]
    path.write_text("\n".join(lines + ["$end", ""]))


def _fortran_records(data: bytes):
    """Split a gfortran sequential unformatted file into records."""
    out, i = [], 0
    while i < len(data):
        (n,) = struct.unpack_from("<i", data, i)
        out.append(data[i + 4:i + 4 + n])
        i += 8 + n
    return out


def read_xtbrestart(path: Path, nat: int):
    recs = _fortran_records(path.read_bytes())
    head = struct.unpack(f"<{len(recs[0]) // 8}q", recs[0])
    qsh = np.frombuffer(recs[1], dtype="<f8").copy()
    res = {"header": list(head), "qsh": qsh.tolist()}
    if len(recs) >= 4:
        res["dpat"] = np.frombuffer(recs[2], dtype="<f8").reshape(nat, 3).tolist()
        res["qpat"] = np.frombuffer(recs[3], dtype="<f8").reshape(nat, 6).tolist()
    return res


def write_xtbrestart(path: Path, nat: int, nel: int, nopen: int, qsh, dpat, qpat, gfn: int = 2):
    def rec(b: bytes) -> bytes:
        return struct.pack("<i", len(b)) + b + struct.pack("<i", len(b))
    head = struct.pack("<6q", gfn, gfn, nat, len(qsh), nel, nopen)
    data = rec(head) + rec(np.asarray(qsh, dtype="<f8").tobytes())
    data += rec(np.asarray(dpat, dtype="<f8").reshape(nat, 3).tobytes())
    data += rec(np.asarray(qpat, dtype="<f8").reshape(nat, 6).tobytes())
    path.write_bytes(data)


def xtb_args(spec: dict, hess: bool = False) -> list[str]:
    acc = max(float(spec.get("accuracy", 1.0)), XTB_MIN_ACC)
    args = ["--gfn", "2", "--hess" if hess else "--grad", "--acc", repr(acc), "--json"]
    q = spec.get("charge", 0)
    args += ["--chrg", str(int(q)) if float(q).is_integer() else repr(float(q))]
    if spec.get("uhf") is not None:
        args += ["--uhf", str(int(spec["uhf"]))]
    if spec.get("etemp", 300.0) != 300.0:
        args += ["--etemp", repr(float(spec["etemp"]))]
    if spec.get("max_iter") is not None:
        args += ["--iterations", str(int(spec["max_iter"]))]
    if spec.get("alpb"):
        args += ["--alpb", spec["alpb"], spec.get("solvation_state", "gsolv")]
    if spec.get("gbsa"):
        args += ["--gbsa", spec["gbsa"], spec.get("solvation_state", "gsolv")]
    return args


def run_xtb(spec: dict, hess: bool = False, restart: dict | None = None, xcontrol: str | None = None,
            timeout: float = 1800.0, binary: str | None = None) -> dict:
    """Run the xtb program (`binary`, default `XTB`) for one case (`spec` = numbers,
    positions, charge, uhf, ...)."""
    nums = np.asarray(spec["numbers"])
    pos = np.asarray(spec["positions"], dtype=float)
    with tempfile.TemporaryDirectory(dir=scratch_root(), prefix="xtb-") as d:
        d = Path(d)
        write_coord(d / "coord", nums, pos)
        cmd = [binary or XTB, "coord"] + xtb_args(spec, hess)
        if xcontrol:
            (d / "xcontrol").write_text(xcontrol)
            cmd += ["--input", "xcontrol"]
        if restart is not None:
            write_xtbrestart(d / "xtbrestart", len(nums), restart["nel"], restart["nopen"],
                             restart["qsh"], restart["dpat"], restart["qpat"])
        env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_STACKSIZE="1G")
        t0 = time.perf_counter()
        try:
            out = subprocess.run(cmd, cwd=d, capture_output=True, text=True, env=env, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": f"timeout after {timeout} s", "cmd": ["xtb"] + cmd[1:]}
        dt = time.perf_counter() - t0
        # xtb's output without the binary's and the scratch directory's paths (results must not
        # hold host paths)
        text = scrub(out.stdout + out.stderr, cmd[0], d)
        # the command as run, without the binary's path
        res: dict = {"cmd": ["xtb"] + cmd[1:], "rc": out.returncode, "seconds": dt}
        warnings = sorted(set(re.findall(r"\[WARNING\][^\n]*", text)))
        if warnings:
            res["warnings"] = warnings
        if out.returncode != 0 or "normal termination" not in text:
            err = [l.strip() for l in text.splitlines() if re.match(r"\s*-\d+-", l) or "[ERROR]" in l]
            res.update(ok=False, error=" | ".join(err)[-600:] or text[-600:])
            return res
        m = re.findall(r"TOTAL ENERGY\s+(-?\d+\.\d+)", out.stdout)
        res["ok"] = True
        res["energy"] = float(m[-1])
        it = re.findall(r"convergence criteria satisfied after\s+(\d+)\s+iterations", out.stdout)
        res["iterations"] = int(it[-1]) if it else None
        gap = re.findall(r"HOMO-LUMO GAP\s+(-?\d+\.\d+)", out.stdout)
        res["gap_ev"] = float(gap[-1]) if gap else None
        if (d / "gradient").exists():
            glines = (d / "gradient").read_text().replace("D", "E").splitlines()
            nat = len(nums)
            res["gradient"] = [[float(x) for x in l.split()] for l in glines[-1 - nat:-1]]
        if (d / "charges").exists():
            res["charges"] = [float(x) for x in (d / "charges").read_text().split()]
        if (d / "xtbout.json").exists():
            js = json.loads((d / "xtbout.json").read_text())
            res["dipole"] = js.get("dipole / a.u.")
            res["occupations"] = js.get("fractional occupation")
            res["nel"] = js.get("number of electrons")
            res["nopen"] = js.get("number of unpaired electrons")
        if (d / "xtbrestart").exists():
            try:
                res["restart"] = read_xtbrestart(d / "xtbrestart", len(nums))
            except Exception as e:  # noqa: BLE001
                res["restart_error"] = str(e)
        gsolv = re.findall(r"-> Gsolv\s+(-?\d+\.\d+)", out.stdout)
        if gsolv:
            res["gsolv"] = float(gsolv[-1])
        if hess and (d / "hessian").exists():
            vals = [float(x) for x in (d / "hessian").read_text().split()[1:]]
            n3 = 3 * len(nums)
            res["hessian"] = np.array(vals).reshape(n3, n3).tolist()
        return res


# ---------------------------------------------------------------------------------------
# tblite (child process)


def run_tblite(spec: dict, timeout: float = 1800.0) -> dict:
    req = {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray) else v) for k, v in spec.items()}
    try:
        out = subprocess.run([PYTHON, str(HERE / "runners.py"), "tblite"], input=json.dumps(req),
                             capture_output=True, text=True, timeout=timeout,
                             env=dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1"))
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout after {timeout} s"}
    lines = [l for l in out.stdout.splitlines() if l.startswith("{")]
    if not lines:
        return {"ok": False, "crashed": True, "rc": out.returncode,
                "error": f"tblite process died (rc {out.returncode}): {out.stderr.strip()[-400:]}"}
    res = json.loads(lines[-1])
    if out.returncode != 0:
        # the result was printed but the interpreter crashed afterwards (tblite cleanup)
        res["crashed_after"] = f"rc {out.returncode}: {out.stderr.strip()[-200:]}"
    return res


def _tblite_param(nodisp: bool):
    from tblite import library as L
    p = L.new_param()
    L.export_gfn2_param(p)
    t = L.new_table()
    L.dump_param(p, t)
    d = L.table_to_dict(t)
    if nodisp:
        d.pop("dispersion")
    return d


def _tblite_child():
    from tblite.interface import Calculator
    req = json.loads(sys.stdin.read())
    nums = np.asarray(req["numbers"])
    pos = np.asarray(req["positions"], dtype=float)
    method = _tblite_param(True) if not req.get("dispersion", True) else "GFN2-xTB"
    try:
        charge = req.get("charge", 0)
        calc = Calculator(method, nums, pos, charge=float(charge), uhf=req.get("uhf"))
        calc.set("verbosity", 0)
        calc.set("accuracy", float(req.get("accuracy", 1.0)))
        calc.set("temperature", float(req.get("etemp", 300.0)) * KB_XTB)
        if req.get("max_iter") is not None:
            calc.set("max-iter", int(req["max_iter"]))
        if req.get("guess") is not None:
            calc.set("guess", int(req["guess"]))
        # tblite names the 1 bar / 1 mol/L state "bar1mol" (xtb and axtb: "bar1M")
        state = {"bar1m": "bar1mol"}.get(req.get("solvation_state", "gsolv").lower(), req.get("solvation_state", "gsolv"))
        if req.get("alpb"):
            calc.add("alpb-solvation", req["alpb"], state)
        if req.get("gbsa"):
            calc.add("gbsa-solvation", req["gbsa"], state)
        t0 = time.perf_counter()
        r = calc.singlepoint()
        res = {"ok": True, "energy": float(r.get("energy")), "gradient": np.asarray(r.get("gradient")).tolist(),
               "charges": np.asarray(r.get("charges")).tolist(), "dipole": np.asarray(r.get("dipole")).tolist(),
               "seconds": time.perf_counter() - t0}
        try:
            res["occupations"] = np.asarray(r.get("orbital-occupations")).tolist()
        except Exception:  # noqa: BLE001
            pass
    except Exception as e:  # noqa: BLE001
        res = {"ok": False, "error": f"{type(e).__name__}: {e}"}
    print(json.dumps(res), flush=True)


# ---------------------------------------------------------------------------------------
# axtb (in process)

AXTB_KEYS = ("charge", "uhf", "accuracy", "etemp", "max_iter", "threads", "dispersion", "alpb", "gbsa",
             "solvation_state")


def axtb_kwargs(spec: dict, conventions: str) -> dict:
    kw = {k: spec[k] for k in AXTB_KEYS if k in spec and spec[k] is not None}
    if "charge" in kw:
        kw["charge"] = float(kw["charge"])
    kw["conventions"] = conventions
    return kw


def _result(r: dict) -> dict:
    out = {"ok": True, "energy": float(r["energy"]), "charges": np.asarray(r["charges"]).tolist(),
           "dipole": np.asarray(r["dipole"]).tolist(), "iterations": int(r["iterations"]),
           "terms": {k: float(v) for k, v in r["terms"].items()}}
    if r.get("gradient") is not None:
        out["gradient"] = np.asarray(r["gradient"]).tolist()
    for k in ("shell_charges", "atomic_dipoles", "atomic_quadrupoles", "occupations_alpha", "occupations_beta",
              "orbital_energies_alpha", "orbital_energies_beta"):
        if k in r and r[k] is not None:
            out[k] = np.asarray(r[k]).tolist()
    return out


def run_axtb(spec: dict, conventions: str = "axtb", gradient: bool = True) -> dict:
    import axtb
    nums = np.asarray(spec["numbers"])
    pos = np.asarray(spec["positions"], dtype=float)
    t0 = time.perf_counter()
    try:
        r = axtb.singlepoint(nums, pos, gradient=gradient, **axtb_kwargs(spec, conventions))
    except BaseException as e:  # noqa: BLE001  (pyo3 PanicException derives from BaseException)
        if isinstance(e, KeyboardInterrupt):
            raise
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    res = _result(r)
    res["seconds"] = time.perf_counter() - t0
    return res


if __name__ == "__main__":
    if sys.argv[1:] == ["tblite"]:
        _tblite_child()
    else:
        sys.exit("usage: runners.py tblite  (JSON request on stdin)")
