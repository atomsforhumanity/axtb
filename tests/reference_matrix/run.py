#!/usr/bin/env python3
"""Reference matrix: every user-facing option of axtb against xtb (6.8.0, or 6.7.1) and
tblite 0.8.0.

    python tests/reference_matrix/run.py [--subset elements,charges,...] [--only ID-substring]
                                         [--jobs N] [--list] [--xtb PATH]

`--xtb` (default: xtb 6.8.0 from `$AXTB_XTB680`, see tests/programs.py) is the xtb binary; its version chooses the
axtb conventions it is compared with: `xtb` for xtb 6.8.0, `xtb-6.7.1` for 6.7.1 (also the
g-xTB 2.0.1 build). Writes `results.json` and `summary.md` next to this file for xtb 6.8.0,
`results-xtb-6.7.1.json` and `summary-xtb-6.7.1.md` for 6.7.1 (merged: cases not run keep
their previous result). Categories and cases are defined in `cases.py`; the programs are run
by `runners.py`. Checks per case (field `checks`):

  xtb               axtb conventions="xtb" (or "xtb-6.7.1") vs the xtb program
  tblite            axtb conventions="tblite" vs tblite
  axtb_vs_xtbmode   axtb default conventions vs axtb xtb conventions (no external reference)
  tblite_info       informational comparison with tblite (currently unused)
  hessian           Calculator.hessian vs `xtb --hess` (projected Hessian, frequencies)
  threads           threads=1 vs threads=8 (energies/gradients to 1e-12)
  threads_hessian   threads=1 vs threads=8 Hessians (bit-identical)
  analytic_ref      analytic Hessian vs a tight response reference, all four conventions
  analytic_numref   analytic Hessian vs Richardson-extrapolated numerical Hessians
  analytic_xtb      analytic Hessian (xtb conventions) vs `xtb --hess` with $hess sccacc=0.001
  analytic_threads  analytic Hessians on 1 vs 8 threads
  failure           failure behaviour of all three programs on invalid/hard input
  calculator        Calculator.singlepoint (wavefunction reuse) vs stateless singlepoint
  cli               the `axtb` command line vs the Python API with the same options

A failed value comparison with xtb is followed by a different-SCF-solution diagnosis: xtb
restarted from axtb's converged moments (written to `xtbrestart`) and axtb started from
xtb's converged moments (`Calculator.set_guess`). If either lands on the other program's
solution, the case is reported as `different_solution`, not as an implementation error.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import multiprocessing as mp
import os
import platform
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cases as C  # noqa: E402
import runners as P  # noqa: E402
import structures as S  # noqa: E402


def output_files(conventions: str) -> tuple[Path, Path]:
    """results and summary files of a run against xtb in `conventions`."""
    if conventions == "xtb":
        return HERE / "results.json", HERE / "summary.md"
    return HERE / f"results-{conventions}.json", HERE / f"summary-{conventions}.md"


RESULTS, SUMMARY = output_files(P.XTB_CONVENTIONS)

# tolerances (Eh, Eh/bohr, e, e·bohr) at tight SCF accuracy
TOL = {
    "xtb": {"E": 1e-7, "G": 1e-7, "q": 1e-6, "dip": 1e-5},
    "tblite": {"E": 1e-9, "G": 1e-8, "q": 1e-7, "dip": 1e-6},
    # corrected default vs xtb mode: README says ~1e-8 Eh for ordinary molecules
    "axtb_vs_xtbmode": {"E": 1e-7, "G": 1e-6, "q": 1e-5, "dip": 1e-4},
    "threads": {"E": 1e-12, "G": 1e-12, "q": 1e-12, "dip": 1e-12},
    "calculator": {"E": 1e-9, "G": 1e-7, "q": 1e-6, "dip": 1e-5},
    "cli": {"E": 1e-10, "G": 1e-9, "q": 1e-8, "dip": 1e-7},
}
# at loose SCF accuracy the converged quantities themselves are only that good: the
# energy threshold is 1e-6·acc and the charge residual 2e-5·acc (1e-4·acc in xtb 6.7.1)
ACC_SCALE = {"E": 1e-6, "G": 1e-4, "q": 3e-4, "dip": 1e-3}


def tolerances(kind: str, case: dict) -> dict:
    base = dict(TOL[kind])
    acc = case.get("accuracy", 1.0)
    if kind == "axtb_vs_xtbmode":
        # size-extensive: xtb 6.7.1's single-precision covalent radii and older bohr
        # constant (and every program's bohr constant vs CODATA 2022) shift every
        # CN-dependent self-energy by ~1e-9 Eh per atom
        base["E"] = max(base["E"], 2e-9 * len(case["numbers"]))
    if kind in ("xtb", "tblite", "axtb_vs_xtbmode"):
        if kind == "xtb":
            acc = max(acc, P.XTB_MIN_ACC)
        if acc > 1e-4:
            for k in base:
                base[k] = max(base[k], ACC_SCALE[k] * acc)
    return base


def diff(a: dict, b: dict) -> dict:
    out = {"dE": a["energy"] - b["energy"]}
    if a.get("gradient") is not None and b.get("gradient") is not None:
        out["dG"] = float(np.abs(np.asarray(a["gradient"]) - np.asarray(b["gradient"])).max())
    if a.get("charges") is not None and b.get("charges") is not None:
        out["dq"] = float(np.abs(np.asarray(a["charges"]) - np.asarray(b["charges"])).max())
    if a.get("dipole") is not None and b.get("dipole") is not None:
        out["ddip"] = float(np.abs(np.asarray(a["dipole"]) - np.asarray(b["dipole"])).max())
    return out


def within(d: dict, tol: dict) -> bool:
    return (abs(d["dE"]) <= tol["E"] and d.get("dG", 0.0) <= tol["G"] and d.get("dq", 0.0) <= tol["q"]
            and d.get("ddip", 0.0) <= tol["dip"])


def compare(new: dict, ref: dict, tol: dict) -> dict:
    if new["ok"] and ref["ok"]:
        d = diff(new, ref)
        d["tol"] = tol
        d["status"] = "pass" if within(d, tol) else "fail"
        return d
    if not new["ok"] and not ref["ok"]:
        return {"status": "both_failed", "error": new.get("error"), "ref_error": ref.get("error")}
    if not new["ok"]:
        return {"status": "axtb_failed", "error": new.get("error")}
    return {"status": "ref_failed", "ref_error": ref.get("error")}


def strip(res: dict) -> dict:
    """Keep results small in results.json."""
    out = {k: v for k, v in res.items() if k not in ("gradient", "restart", "hessian", "shell_charges",
                                                     "atomic_dipoles", "atomic_quadrupoles", "occupations",
                                                     "orbital_energies", "occupations_alpha", "occupations_beta",
                                                     "orbital_energies_alpha", "orbital_energies_beta", "stdout")}
    if "charges" in out and out["charges"] is not None and len(out["charges"]) > 12:
        out["charges"] = None
    return out


# ---------------------------------------------------------------------------------------
# different-SCF-solution diagnosis


def lands_on(d: dict, orig: dict, tol: dict) -> bool:
    """A restarted SCF lands on the other program's solution: every difference is within
    tolerance, or at least 100x smaller than the original discrepancy (needed for
    pathologically slow SCFs, e.g. near-zero gaps, whose converged moments carry the
    convergence noise of the run they come from)."""
    for k, t in (("dE", "E"), ("dG", "G"), ("dq", "q"), ("ddip", "dip")):
        if k in d and abs(d[k]) > max(tol[t], 0.01 * abs(orig.get(k, 0.0))):
            return False
    return True


def diagnose_xtb(case: dict, ax: dict, xt: dict, tol: dict) -> dict:
    import axtb
    out = {}
    spec = dict(case)
    # xtb restarted from axtb's converged moments
    if "restart" in xt and "shell_charges" in ax:
        hdr = xt["restart"]["header"]
        x2 = P.run_xtb(spec, restart={"nel": hdr[4], "nopen": hdr[5], "qsh": ax["shell_charges"],
                                      "dpat": ax["atomic_dipoles"], "qpat": ax["atomic_quadrupoles"]})
        if x2["ok"]:
            out["xtb_from_axtb"] = {"energy": x2["energy"], "iterations": x2.get("iterations"),
                                    **{k: v for k, v in diff(ax, x2).items()}}
        else:
            out["xtb_from_axtb"] = {"error": x2.get("error")}
    # axtb started from xtb's converged moments
    if "restart" in xt and "dpat" in xt["restart"]:
        try:
            calc = axtb.Calculator("gfn2", np.asarray(case["numbers"]), np.asarray(case["positions"]),
                                   **{k: v for k, v in P.axtb_kwargs(case, P.XTB_CONVENTIONS).items() if k != "max_iter"})
            r = xt["restart"]
            calc.set_guess(r["qsh"], np.asarray(r["dpat"]), np.asarray(r["qpat"]))
            a2 = P._result(calc.singlepoint(np.asarray(case["positions"]), gradient=True))
            out["axtb_from_xtb"] = {"energy": a2["energy"], "iterations": a2["iterations"], **diff(a2, xt)}
        except Exception as e:  # noqa: BLE001
            out["axtb_from_xtb"] = {"error": f"{type(e).__name__}: {e}"}
    orig = diff(ax, xt)
    a_ok = "dE" in out.get("xtb_from_axtb", {}) and lands_on(out["xtb_from_axtb"], orig, tol)
    x_ok = "dE" in out.get("axtb_from_xtb", {}) and lands_on(out["axtb_from_xtb"], orig, tol)
    out["axtb_solution_is_xtb_solution"] = a_ok
    out["xtb_solution_is_axtb_solution"] = x_ok
    out["lower"] = "axtb" if ax["energy"] < xt["energy"] else "xtb"
    return out


# ---------------------------------------------------------------------------------------
# checks


def check_xtb(case, store):
    ax = store.setdefault("axtb[xtb]", P.run_axtb(case, P.XTB_CONVENTIONS))
    xt = store.setdefault("xtb", P.run_xtb(case))
    tol = tolerances("xtb", case)
    res = compare(ax, xt, tol)
    if (res["status"] == "fail" and P.XTB_671_RELEASE and abs(res["dE"]) <= tol["E"]
            and res.get("dq", 0.0) <= tol["q"] and res.get("ddip", 0.0) <= tol["dip"]):
        # the xtb-6.7.1 conventions reproduce the g-xTB 2.0.1 build of xtb 6.7.1, which has the
        # fix of xtb PR #1208 (merged after the 6.7.1 release): the release drops the H0/S term
        # of AO pairs whose overlap vanishes by symmetry (two atoms sharing a Cartesian
        # coordinate) from its gradient
        res["status"] = "expected_difference"
        res["note"] = ("xtb 6.7.1 release: gradient without the H0/S term of AO pairs with zero overlap "
                       "(atoms sharing a coordinate), fixed by xtb PR #1208")
    if res["status"] == "fail":
        dg = diagnose_xtb(case, ax, xt, tol)
        res["diagnosis"] = dg
        if dg["axtb_solution_is_xtb_solution"] or dg["xtb_solution_is_axtb_solution"]:
            res["status"] = "different_solution"
    return res


def check_tblite(case, store):
    ax = store.setdefault("axtb[tblite]", P.run_axtb(case, "tblite"))
    tb = store.setdefault("tblite", P.run_tblite(case))
    res = compare(ax, tb, tolerances("tblite", case))
    if tb.get("crashed_after"):
        res["tblite_crashed_after_result"] = tb["crashed_after"]
    if res["status"] == "fail":
        res["diagnosis"] = diagnose_tblite(case, ax, tb, store)
        if res["diagnosis"].get("tblite_solution_is_axtb_solution") or res["diagnosis"].get(
                "axtb_solution_is_tblite_solution"):
            res["status"] = "different_solution"
    return res


def diagnose_tblite(case, ax, tb, store):
    """tblite has no restart interface. Evidence for a different SCF solution:
    (1) axtb (tblite conventions) started from other converged moments (xtb's, axtb's xtb
    mode) lands on tblite's energy and charges; (2) tblite started from its EEQ guess
    instead of zero charges lands on axtb's solution."""
    import axtb
    out = {}
    tol = tolerances("tblite", case)
    starts = {}
    xt = store.get("xtb") or P.run_xtb(case)
    if xt.get("ok") and "restart" in xt and "dpat" in xt["restart"]:
        starts["xtb"] = (xt["restart"]["qsh"], xt["restart"]["dpat"], xt["restart"]["qpat"])
    if P.XTB_OTHER:
        xo = P.run_xtb(case, binary=P.XTB_OTHER)
        if xo.get("ok") and "restart" in xo and "dpat" in xo["restart"]:
            starts[f"xtb {P.xtb_version(P.XTB_OTHER)}"] = (xo["restart"]["qsh"], xo["restart"]["dpat"], xo["restart"]["qpat"])
    axx = store.get("axtb[xtb]") or P.run_axtb(case, P.XTB_CONVENTIONS)
    if axx.get("ok"):
        starts["axtb[xtb]"] = (axx["shell_charges"], axx["atomic_dipoles"], axx["atomic_quadrupoles"])
    for name, (qsh, dp, qp) in starts.items():
        try:
            calc = axtb.Calculator("gfn2", np.asarray(case["numbers"]), np.asarray(case["positions"]),
                                   **{k: v for k, v in P.axtb_kwargs(case, "tblite").items() if k != "max_iter"})
            calc.set_guess(qsh, np.asarray(dp), np.asarray(qp))
            a2 = P._result(calc.singlepoint(np.asarray(case["positions"]), gradient=True))
            d = diff(a2, tb)
            out[f"tbmode_from_{name}"] = {"energy": a2["energy"], **d}
            if lands_on(d, diff(ax, tb), tol):
                out["tblite_solution_is_axtb_solution"] = True
        except Exception as e:  # noqa: BLE001
            out[f"tbmode_from_{name}"] = {"error": f"{type(e).__name__}: {e}"}
    tb2 = P.run_tblite(dict(case, guess=1))
    if tb2.get("ok"):
        d = diff(ax, tb2)
        out["tblite_eeq_guess"] = {"energy": tb2["energy"], **d}
        if lands_on(d, diff(ax, tb), tol):
            out["axtb_solution_is_tblite_solution"] = True
    out["lower"] = "axtb" if ax["energy"] < tb["energy"] else "tblite"
    return out


def extent(case) -> float:
    """Largest interatomic distance (bohr)."""
    p = np.asarray(case["positions"], dtype=float)
    return float(np.sqrt(((p[:, None] - p[None]) ** 2).sum(-1)).max()) if len(p) > 1 else 0.0


# xtb 6.8.0 cuts the CN, repulsion and D4 three-body terms at 25 bohr and the D4 two-body
# term at 50 bohr (tblite's cutoffs); the default conventions keep xtb 6.7.1's 40/72 bohr with
# smooth switches. For molecules longer than 25 bohr the two differ by more than the
# tolerances (taxol, 30 bohr: 9.4e-6 Eh)
LONG_MOLECULE = 25.0
LONG_TOL = {"E": 2e-5, "G": 1e-5}


def check_axtb_vs_xtbmode(case, store):
    ax = store.setdefault("axtb", P.run_axtb(case, "axtb"))
    axx = store.setdefault("axtb[xtb]", P.run_axtb(case, P.XTB_CONVENTIONS))
    tol = tolerances("axtb_vs_xtbmode", case)
    long = P.XTB_CONVENTIONS == "xtb" and extent(case) > LONG_MOLECULE
    if long:
        tol = dict(tol, E=max(tol["E"], LONG_TOL["E"]), G=max(tol["G"], LONG_TOL["G"]))
    res = compare(ax, axx, tol)
    if res["status"] == "pass" and long and not within(res, tolerances("axtb_vs_xtbmode", case)):
        res["status"] = "expected_difference"
        res["note"] = (f"real-space cutoffs: the default's 40/72 bohr vs xtb 6.8.0's 25/50 bohr "
                       f"(molecule {extent(case):.0f} bohr long)")
    if res["status"] == "fail" and ax["ok"]:
        # the default conventions use the CODATA 2022 Boltzmann constant: the electronic
        # free energy -TS changes by -TS·(kB_2022/kB_xtb - 1) to first order
        ts = ax["terms"].get("entropy", 0.0)
        shift = ts * (KB_CODATA2022 / P.KB_XTB - 1.0)
        res["entropy_kB_shift"] = shift
        if abs(res["dE"] - shift) <= max(tol["E"], 0.05 * abs(shift)) and res.get("dG", 0) <= tol["G"]:
            res["status"] = "expected_difference"
            res["note"] = f"CODATA 2022 kB for the electronic temperature: -TS·(kB ratio - 1) = {shift:.2e} Eh"
    if res["status"] == "fail" and ax["ok"] and axx["ok"]:
        # different SCF solutions of the two modes: restart each from the other's moments
        dg = {"axtb_from_xtbmode": restart_axtb(case, "axtb", axx, axx),
              "xtbmode_from_axtb": restart_axtb(case, P.XTB_CONVENTIONS, ax, ax)}
        res["diagnosis"] = dg
        orig = diff(ax, axx)
        if any("dE" in v and lands_on(v, orig, tol) for v in dg.values()):
            res["status"] = "different_solution"
    if res["status"] == "fail" and 0.0 < case.get("etemp", 300.0) <= 0.1:
        res["status"] = "expected_difference"
        res["note"] = "xtb keeps aufbau occupations at or below 0.1 K; the default conventions smear at any T > 0"
    if (res["status"] == "fail" and P.XTB_CONVENTIONS == "xtb-6.7.1" and (case.get("alpb") or case.get("gbsa"))
            and case.get("solvation_state") == "reference"):
        res["status"] = "expected_difference"
        res["note"] = "reference state: mass density of the solvent instead of xtb 6.7.1's rho/M term (grimme-lab/xtb#1505)"
    return res


KB_CODATA2022 = 3.1668115634564e-6


def restart_axtb(case, conv, start, target):
    """axtb in conventions `conv` started from the moments of result `start`, compared with
    result `target` (the other mode's own solution)."""
    import axtb
    try:
        calc = axtb.Calculator("gfn2", np.asarray(case["numbers"]), np.asarray(case["positions"]),
                               **{k: v for k, v in P.axtb_kwargs(case, conv).items() if k != "max_iter"})
        calc.set_guess(start["shell_charges"], np.asarray(start["atomic_dipoles"]), np.asarray(start["atomic_quadrupoles"]))
        r2 = P._result(calc.singlepoint(np.asarray(case["positions"]), gradient=True))
        return {"energy": r2["energy"], **diff(r2, target)}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}


def check_tblite_info(case, store):
    """Informational: tblite's own ALPB/GBSA against axtb (tblite and default conventions)."""
    ax = store.setdefault("axtb[tblite]", P.run_axtb(case, "tblite"))
    tb = store.setdefault("tblite", P.run_tblite(case))
    axd = store.setdefault("axtb", P.run_axtb(case, "axtb"))
    res = compare(ax, tb, tolerances("tblite", case))
    res["status"] = "info"
    if axd["ok"] and tb["ok"]:
        res["axtb_default_minus_tblite"] = axd["energy"] - tb["energy"]
    return res


def rotmol(pos, xrot=1e-4, yrot=2e-4, zrot=3e-4):
    """xtb 6.7.1's `rotmol` (hessian.F90): tiny rotations (degrees) applied before `--hess`;
    6.8.0 removed it (the positions are returned unchanged)."""
    p = np.array(pos, dtype=float)
    if not P.XTB_HESS_ROTATES:
        return p
    for ax, (i, j) in ((xrot, (1, 2)), (yrot, (0, 2)), (zrot, (0, 1))):
        a = ax * math.pi / 180.0
        xo, yo = p[:, i].copy(), p[:, j].copy()
        p[:, i] = xo * math.cos(a) + yo * math.sin(a)
        p[:, j] = -xo * math.sin(a) + yo * math.cos(a)
    return p


def tr_projector(pos):
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


def xtb_masses():
    """xtb's atomic masses (u) by atomic number - 1 (`src/param/atomicmass.f90`, the same in
    6.7.1 and 6.8.0), from axtb's copy of the table (its vibrational analysis in xtb
    conventions)."""
    import axtb
    out = []
    for z in range(1, 87):
        v = axtb.vibrational_analysis([z], np.zeros((1, 3)), np.zeros((3, 3)), conventions="xtb", thermo=False)
        out.append(float(np.asarray(v["masses"])[0]))
    return np.array(out)


def xtb_projected(h, nums, pos, masses):
    """The projected Cartesian Hessian xtb writes to its `hessian` file: 6.7.1 projects the
    translations and rotations about the geometric center from the Cartesian Hessian (not for
    linear molecules); 6.8.0 projects them in mass-weighted coordinates (rotations about the
    center of mass, linear molecules too) and transforms back."""
    lin = is_linear(pos)
    if P.XTB_HESS_ROTATES:  # xtb 6.7.1
        p = np.eye(3 * len(nums)) if lin else tr_projector(pos)
        return p @ h @ p
    m = masses[np.asarray(nums) - 1]
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
    hm = h / np.outer(sq, sq)
    return (pr @ hm @ pr) * np.outer(sq, sq)


def frequencies(h, nums, masses):
    amu_to_me = 1.0 / 5.485799090441e-4
    m = np.repeat(masses[np.asarray(nums) - 1] * amu_to_me, 3)
    w = np.linalg.eigvalsh(h / np.sqrt(np.outer(m, m)))
    return np.sign(w) * np.sqrt(np.abs(w)) * 2.1947463136314e7 / 100


def is_linear(pos):
    c = pos - pos.mean(axis=0)
    return np.linalg.svd(c, compute_uv=False)[1] < 1e-6 if len(pos) > 2 else True


def check_hessian(case, store):
    import axtb
    nums = np.asarray(case["numbers"])
    pos = np.asarray(case["positions"], dtype=float)
    rpos = rotmol(pos)
    masses = xtb_masses()
    out = {}
    lin = is_linear(rpos)
    kw = {k: v for k, v in P.axtb_kwargs(case, P.XTB_CONVENTIONS).items() if k not in ("accuracy",)}
    variants = {
        # as asked: xtb --hess --acc 0.001 (displacement SCFs at 6.7.1's default accu_hess=0.3,
        # at 0.001 in 6.8.0)
        "acc0.001": (dict(case, accuracy=1e-3), None, P.xtb_hess_displacement_acc(1e-3)),
        # tight displacement SCFs on both sides ($hess sccacc)
        "tight": (dict(case, accuracy=1e-4), "$hess\n   sccacc=0.0001\n$end\n", 1e-4),
    }
    for name, (spec, xc, ax_acc) in variants.items():
        xt = P.run_xtb(spec, hess=True, xcontrol=xc)
        if not xt["ok"] or "hessian" not in xt:
            out[name] = {"status": "ref_failed", "ref_error": xt.get("error")}
            continue
        hx = np.asarray(xt["hessian"])
        try:
            calc = axtb.Calculator("gfn2", nums, rpos, accuracy=spec["accuracy"], **kw)
            h = np.asarray(calc.hessian(rpos, step=0.005, accuracy=ax_acc, method="numerical")["hessian"])
        except Exception as e:  # noqa: BLE001
            out[name] = {"status": "axtb_failed", "error": str(e)}
            continue
        hp = xtb_projected(h, nums, rpos, masses)
        d = {"dH": float(np.abs(hp - hx).max()), "Hmax": float(np.abs(hx).max()), "linear": bool(lin)}
        if masses is not None:
            fx, fa = frequencies(hx, nums, masses), frequencies(hp, nums, masses)
            sel = np.abs(fx) > 50
            d["dfreq"] = float(np.abs(fa - fx)[sel].max()) if sel.any() else 0.0
            d["freqs_xtb"] = np.round(np.sort(fx)[-min(6, len(fx)):], 2).tolist()
        tol = {"acc0.001": (2e-4, 1.0), "tight": (2e-6, 0.05)}[name]
        d["tol"] = {"dH": tol[0], "dfreq": tol[1]}
        d["status"] = "pass" if d["dH"] <= tol[0] and d.get("dfreq", 0.0) <= tol[1] else "fail"
        out[name] = d
    st = [v["status"] for v in out.values()]
    return {"status": "pass" if all(s == "pass" for s in st) else "fail", **out}


# Analytic Hessians (Eh/bohr²). The reference is a response Hessian: the same coupled-
# perturbed SCC response, but second derivatives by central differences of the gradient
# functional, Richardson-extrapolated from steps 1e-3/5e-4 bohr (error ~1e-12). In
# implicit solvent the surface term is C¹ but not C² (clamped cubic switching per grid
# point): finite differences straddle its kinks unless the step is tiny, so the solvated
# reference is one 1e-5 step (2e-5..2e-6 agree with the analytic Hessian to 1e-10 for taxol).
HESS_TOL = {
    # Richardson error ~(5e-4)^4 times fifth derivatives plus round-off of the 1e-5 step:
    # observed <= 1.4e-10
    "ref": 1e-8,
    # largest |H_ij - H_ji| of the analytic Hessian before symmetrization (round-off, or the
    # finite-difference stale-r1 Born term of xtb-convention solvation: observed <= 2.3e-12)
    "asym": 1e-10,
    # vs Richardson numerical Hessians (0.002/0.001 bohr, one SCF per displacement at 1e-8);
    # solvated: one 0.001 step, which smooths over the surface kinks (~1.5e-4)
    "numref": 1e-8,
    "numref_solvated": 3e-4,
    "threads": 1e-12,
    # vs `xtb --hess` with $hess sccacc=0.001. axtb's numerical Hessian with xtb's own
    # procedure (0.005 bohr, displacement SCFs at 0.001) must match it to SCF noise; the
    # analytic Hessian differs by xtb's O(h²) truncation (h²/6 · third derivatives, up to
    # ~1e-4 for X-H stretches) and, in implicit solvent, by the finite-difference smoothing
    # of the surface kinks (up to ~1.5e-4, 14 cm⁻¹ in soft caffeine modes). Frequencies
    # above 50 cm⁻¹.
    "xtb_same_procedure": 2e-5,
    "xtb": {"dH": 2e-4, "dfreq": 1.0},
    "xtb_solvated": {"dH": 5e-4, "dfreq": 20.0},
}
XTB_TIGHT_HESS = "$hess\n   sccacc=0.001\n$end\n"


def _solvated(case):
    return bool(case.get("alpb") or case.get("gbsa"))


def _hess_calc(case, conv, pos, threads=None):
    import axtb
    kw = P.axtb_kwargs(case if threads is None else dict(case, threads=threads), conv)
    return axtb.Calculator("gfn2", np.asarray(case["numbers"]), pos, **kw)


def _hessian_compare(h, ref, nums, pos, masses, tol, asym=None):
    lin = is_linear(pos)
    d = {"dH": float(np.abs(h - ref).max()), "Hmax": float(np.abs(ref).max()), "tol": tol}
    if asym is not None:
        d["asym"] = float(asym)
    if masses is not None:
        p = np.eye(3 * len(nums)) if lin else tr_projector(pos)
        fa, fr = frequencies(p @ h @ p, nums, masses), frequencies(p @ ref @ p, nums, masses)
        d["dfreq"] = float(np.abs(fa - fr).max())
    ok = d["dH"] <= tol and (asym is None or asym <= HESS_TOL["asym"])
    d["status"] = "pass" if ok else "fail"
    return d


def check_analytic_ref(case, store):
    """Analytic Hessian vs the response reference, in all four conventions."""
    nums = np.asarray(case["numbers"])
    pos = np.asarray(case["positions"], dtype=float)
    masses = xtb_masses()
    out = {}
    for conv in ("axtb", "xtb", "xtb-6.7.1", "tblite"):
        try:
            calc = _hess_calc(case, conv, pos)
            ha = calc.hessian(pos, method="analytic", accuracy=1e-8)
            if _solvated(case):
                ref = np.asarray(calc.hessian(pos, method="response", step=1e-5, accuracy=1e-8)["hessian"])
            else:
                r1 = np.asarray(calc.hessian(pos, method="response", step=1e-3, accuracy=1e-8)["hessian"])
                r2 = np.asarray(calc.hessian(pos, method="response", step=5e-4, accuracy=1e-8)["hessian"])
                ref = (4.0 * r2 - r1) / 3.0
        except Exception as e:  # noqa: BLE001
            out[conv] = {"status": "axtb_failed", "error": f"{type(e).__name__}: {e}"}
            continue
        out[conv] = _hessian_compare(np.asarray(ha["hessian"]), ref, nums, pos, masses, HESS_TOL["ref"],
                                     asym=ha["hessian_max_asymmetry"])
    st = [v["status"] for v in out.values()]
    return {"status": "pass" if all(s == "pass" for s in st) else "fail", **out}


def check_analytic_numref(case, store):
    """Analytic Hessian vs Richardson-extrapolated numerical Hessians (default conventions)."""
    nums = np.asarray(case["numbers"])
    pos = np.asarray(case["positions"], dtype=float)
    masses = xtb_masses()
    try:
        calc = _hess_calc(case, "axtb", pos)
        ha = np.asarray(calc.hessian(pos, method="analytic", accuracy=1e-8)["hessian"])
        n2 = np.asarray(calc.hessian(pos, method="numerical", step=0.001, accuracy=1e-8)["hessian"])
        if _solvated(case):
            ref, tol = n2, HESS_TOL["numref_solvated"]
        else:
            n1 = np.asarray(calc.hessian(pos, method="numerical", step=0.002, accuracy=1e-8)["hessian"])
            ref, tol = (4.0 * n2 - n1) / 3.0, HESS_TOL["numref"]
    except Exception as e:  # noqa: BLE001
        return {"status": "axtb_failed", "error": f"{type(e).__name__}: {e}"}
    d = _hessian_compare(ha, ref, nums, pos, masses, tol)
    return {"status": d["status"], "numerical": d}


def check_analytic_xtb(case, store):
    """Analytic Hessian (xtb conventions) vs `xtb --hess` with $hess sccacc=0.001."""
    nums = np.asarray(case["numbers"])
    rpos = rotmol(np.asarray(case["positions"], dtype=float))
    masses = xtb_masses()
    xt = P.run_xtb(dict(case, positions=rpos.tolist(), accuracy=1e-3), hess=True, xcontrol=XTB_TIGHT_HESS)
    if not xt["ok"] or "hessian" not in xt:
        return {"status": "ref_failed", "ref_error": xt.get("error")}
    hx = np.asarray(xt["hessian"])
    try:
        calc = _hess_calc(case, P.XTB_CONVENTIONS, rpos)
        h = np.asarray(calc.hessian(rpos, method="analytic", accuracy=1e-6)["hessian"])
        hn = np.asarray(calc.hessian(rpos, method="numerical", step=0.005, accuracy=1e-3)["hessian"])
    except Exception as e:  # noqa: BLE001
        return {"status": "axtb_failed", "error": f"{type(e).__name__}: {e}"}
    lin = is_linear(rpos)
    hp = xtb_projected(h, nums, rpos, masses)
    tol = HESS_TOL["xtb_solvated" if _solvated(case) else "xtb"]
    d = {"dH": float(np.abs(hp - hx).max()), "Hmax": float(np.abs(hx).max()), "linear": bool(lin), "tol": tol,
         # axtb numerical Hessian with xtb's procedure vs xtb: the implementations agree
         "dH_numerical_same_procedure": float(np.abs(xtb_projected(hn, nums, rpos, masses) - hx).max()),
         "tol_same_procedure": HESS_TOL["xtb_same_procedure"]}
    if masses is not None:
        fx, fa = frequencies(hx, nums, masses), frequencies(hp, nums, masses)
        sel = np.abs(fx) > 50
        d["dfreq"] = float(np.abs(fa - fx)[sel].max()) if sel.any() else 0.0
        d["imaginary_xtb_axtb"] = [int((fx < -50).sum()), int((fa < -50).sum())]
    ok = (d["dH"] <= tol["dH"] and d.get("dfreq", 0.0) <= tol["dfreq"]
          and d["dH_numerical_same_procedure"] <= HESS_TOL["xtb_same_procedure"]
          and d.get("imaginary_xtb_axtb", [0, 0])[0] == d.get("imaginary_xtb_axtb", [0, 0])[1])
    d["status"] = "pass" if ok else "fail"
    return {"status": d["status"], "sccacc0.001": d}


def check_analytic_threads(case, store):
    pos = np.asarray(case["positions"], dtype=float)
    hs = {}
    for t in (1, 8):
        calc = _hess_calc(case, "axtb", pos, threads=t)
        hs[t] = np.asarray(calc.hessian(pos, method="analytic")["hessian"])
    d = float(np.abs(hs[1] - hs[8]).max())
    return {"status": "pass" if d <= HESS_TOL["threads"] else "fail", "bit_identical": bool(np.array_equal(hs[1], hs[8])),
            "max_abs_diff": d}


def check_threads(case, store):
    res = {}
    for conv in ("axtb", "xtb", "xtb-6.7.1", "tblite"):
        r1 = P.run_axtb(dict(case, threads=1), conv)
        r8 = P.run_axtb(dict(case, threads=8), conv)
        res[conv] = compare(r8, r1, TOL["threads"])
    res["status"] = "pass" if all(v["status"] == "pass" for v in res.values() if isinstance(v, dict)) else "fail"
    return res


def check_threads_hessian(case, store):
    import axtb
    nums = np.asarray(case["numbers"])
    pos = np.asarray(case["positions"], dtype=float)
    hs = {}
    for t in (1, 8):
        calc = axtb.Calculator("gfn2", nums, pos, **{k: v for k, v in P.axtb_kwargs(dict(case, threads=t), "axtb").items()})
        hs[t] = np.asarray(calc.hessian(pos, step=0.005, accuracy=1e-3, method="numerical")["hessian"])
    same = bool(np.array_equal(hs[1], hs[8]))
    return {"status": "pass" if same else "fail", "bit_identical": same, "max_abs_diff": float(np.abs(hs[1] - hs[8]).max())}


def outcome(res: dict, charge: float) -> dict:
    if not res["ok"]:
        err = res.get("error", "")
        kind = "panic" if "PanicException" in err else ("crash" if res.get("crashed") else "error")
        return {"outcome": kind, "error": err[:300]}
    o = {"outcome": "ok", "energy": res["energy"]}
    if res.get("charges") is not None:
        qsum = float(np.sum(res["charges"]))
        o["charge_sum"] = qsum
        if abs(qsum - charge) > 1e-5 or not math.isfinite(res["energy"]):
            o["outcome"] = "garbage"
            o["why"] = f"charges sum to {qsum:.6f}, molecule charge {charge}"
    if res.get("crashed_after"):
        o["crashed_after"] = res["crashed_after"]
    return o


def check_failure(case, store):
    q = case["charge"]
    progs = {
        "xtb": outcome(store.setdefault("xtb", P.run_xtb(case)), q),
        "tblite": outcome(store.setdefault("tblite", P.run_tblite(case)), q),
        "axtb": outcome(store.setdefault("axtb", P.run_axtb(case, "axtb")), q),
        "axtb[xtb]": outcome(store.setdefault("axtb[xtb]", P.run_axtb(case, P.XTB_CONVENTIONS)), q),
        "axtb[tblite]": outcome(store.setdefault("axtb[tblite]", P.run_axtb(case, "tblite")), q),
    }
    exp = case.get("expect", "any")
    ax = [progs[k]["outcome"] for k in ("axtb", "axtb[xtb]", "axtb[tblite]")]
    if exp == "error":
        ok = all(o == "error" for o in ax)
    elif exp == "ok":
        ok = all(o == "ok" for o in ax)
    else:
        ok = all(o in ("ok", "error") for o in ax)
    res = {"status": "pass" if ok else "fail", "expect": exp, "outcomes": progs}
    refs_bad = [k for k in ("xtb", "tblite") if progs[k]["outcome"] != "ok"]
    res["references_not_ok"] = refs_bad
    return res


def check_calculator(case, store):
    import axtb
    nums = np.asarray(case["numbers"])
    pos0 = np.asarray(case["positions"], dtype=float)
    rng = np.random.default_rng(7)
    kw = P.axtb_kwargs(case, "axtb")
    calc = axtb.Calculator("gfn2", nums, pos0, **kw)
    worst = {"dE": 0.0, "dG": 0.0, "dq": 0.0, "ddip": 0.0}
    iters = []
    pos = pos0.copy()
    for step in range(4):
        r = P._result(calc.singlepoint(pos, gradient=True))
        ref = P.run_axtb(dict(case, positions=pos), "axtb")
        d = diff(r, ref)
        iters.append((r["iterations"], ref["iterations"]))
        for k in worst:
            worst[k] = max(worst[k], abs(d.get(k, 0.0)))
        pos = pos + rng.normal(scale=0.02, size=pos.shape)
    h = calc.hessian(pos0, step=0.005, accuracy=1e-6, method="numerical")
    ref = P.run_axtb(dict(case, positions=pos0), "axtb")
    dh = {"dE": h["energy"] - ref["energy"],
          "dG": float(np.abs(np.asarray(h["gradient"]) - np.asarray(ref["gradient"])).max())}
    tol = TOL["calculator"]
    ok = within(dict(worst, dE=worst["dE"]), tol) and abs(dh["dE"]) <= 1e-8 and dh["dG"] <= 1e-6
    return {"status": "pass" if ok else "fail", **worst, "iterations(calc, fresh)": iters, "hessian_reference": dh,
            "tol": tol}


def check_cli(case, store):
    conv = case["cli_conventions"]
    nums = np.asarray(case["numbers"])
    pos = np.asarray(case["positions"], dtype=float)
    # the bohr (Angstrom) of the CLI's XYZ conversion in these conventions, so that the round trip
    # is exact to ~1e-16: CODATA 2022 in axtb's, mctc-lib's (CODATA 2018) in the program conventions
    bohr = 0.529177210544 if conv == "axtb" else 1.0 / 1.8897261246204404
    with tempfile.TemporaryDirectory(dir=P.scratch_root(), prefix="cli-") as d:
        xyz = Path(d) / "mol.xyz"
        lines = [str(len(nums)), "reference matrix"] + [
            f"{S.SYMBOLS[z - 1]} {x * bohr:.17e} {y * bohr:.17e} {zz * bohr:.17e}" for z, (x, y, zz) in zip(nums, pos)]
        xyz.write_text("\n".join(lines) + "\n")
        args = [str(S.ROOT / "target/release/axtb"), str(xyz), "--grad", "--conventions", conv,
                "--chrg", str(case["charge"]), "--acc", repr(case["accuracy"])]
        if case.get("uhf") is not None:
            args += ["--uhf", str(case["uhf"])]
        if case.get("etemp") is not None:
            args += ["--etemp", repr(case["etemp"])]
        if case.get("threads") is not None:
            args += ["--threads", str(case["threads"])]
        if case.get("dispersion") is False:
            args += ["--nodisp"]
        for m in ("alpb", "gbsa"):
            if case.get(m):
                args += [f"--{m}", case[m]]
        if case.get("solvation_state"):
            args += ["--solvation-state", case["solvation_state"]]
        out = subprocess.run(args, capture_output=True, text=True)
    if out.returncode != 0:
        return {"status": "fail", "error": (out.stderr + out.stdout)[-400:]}
    cli = json.loads(out.stdout)
    cli["ok"] = True
    py = P.run_axtb(case, conv)
    res = compare(cli, py, TOL["cli"])
    res["args"] = args[2:]
    return res


CHECKS = {
    "xtb": check_xtb,
    "tblite": check_tblite,
    "axtb_vs_xtbmode": check_axtb_vs_xtbmode,
    "tblite_info": check_tblite_info,
    "hessian": check_hessian,
    "threads": check_threads,
    "threads_hessian": check_threads_hessian,
    "analytic_ref": check_analytic_ref,
    "analytic_numref": check_analytic_numref,
    "analytic_xtb": check_analytic_xtb,
    "analytic_threads": check_analytic_threads,
    "failure": check_failure,
    "calculator": check_calculator,
    "cli": check_cli,
}


def evaluate(case: dict) -> dict:
    t0 = time.perf_counter()
    store: dict = {}
    out = {k: case[k] for k in ("id", "category", "label", "charge", "uhf", "nel", "tags") if k in case}
    out["natoms"] = len(case["numbers"])
    out["options"] = {k: case[k] for k in ("accuracy", "etemp", "max_iter", "dispersion", "threads", "alpb", "gbsa",
                                           "solvation_state", "expect", "multiplicity") if k in case}
    out["checks"] = {}
    for chk in case["checks"]:
        try:
            out["checks"][chk] = CHECKS[chk](case, store)
        except Exception as e:  # noqa: BLE001
            out["checks"][chk] = {"status": "error", "error": f"{type(e).__name__}: {e}",
                                  "traceback": traceback.format_exc()[-1500:]}
    out["programs"] = {k: strip(v) for k, v in store.items()}
    out["seconds"] = time.perf_counter() - t0
    return out


def _worker(case):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    return evaluate(case)


# ---------------------------------------------------------------------------------------
# summary


def worst_of(rows, key):
    vals = [abs(r[key]) for r in rows if key in r and isinstance(r[key], (int, float))]
    return max(vals) if vals else None


def fmt(x):
    return "—" if x is None else f"{x:.1e}"


def write_summary(results: dict, meta: dict):
    cases = list(results.values())
    cats = list(dict.fromkeys(c["category"] for c in sorted(cases, key=lambda c: list(C.CATEGORIES).index(c["category"]))))
    xconv = meta.get("xtb_conventions", "xtb")
    lines = ["# axtb reference matrix" + ("" if xconv == "xtb" else f" against xtb {meta.get('xtb', '?')}"), "",
             f"Generated {meta['date']} — xtb {meta.get('xtb', '?')} (axtb conventions \"{xconv}\"), "
             f"tblite {meta.get('tblite', '?')}, axtb {meta.get('axtb', '?')} (git {meta.get('git', '?')}).", "",
             "Value comparisons at SCF accuracy 1e-6 (xtb: 1e-4, its minimum). Tolerances: "
             "xtb mode vs xtb |ΔE| ≤ 1e-7 Eh, max|ΔG| ≤ 1e-7 Eh/bohr, |Δq| ≤ 1e-6 e, |Δμ| ≤ 1e-5 e·bohr; "
             "tblite mode vs tblite 1e-9 / 1e-8 / 1e-7 / 1e-6; axtb vs xtb mode 1e-7 / 1e-6 / 1e-5 / 1e-4 "
             "(all scaled up with the SCF accuracy above 1e-4). Statuses: pass, fail, different_solution "
             "(another SCF solution of the same equations, verified by restarting each program from the "
             "other's converged moments), expected_difference (documented fixes of the default conventions), "
             "both_failed (consistent failure), axtb_failed / ref_failed, info (not meant to match).", ""]
    lines += ["## Overview", "", "| category | cases | check | pass | fail | other | worst |ΔE| | worst max|ΔG| | worst |Δq| | worst |Δμ| |",
              "|---|---:|---|---:|---:|---|---:|---:|---:|---:|"]
    for cat in cats:
        cc = [c for c in cases if c["category"] == cat]
        checks = list(dict.fromkeys(k for c in cc for k in c["checks"]))
        for i, chk in enumerate(checks):
            rows = [c["checks"][chk] for c in cc if chk in c["checks"]]
            st = {}
            for r in rows:
                st[r["status"]] = st.get(r["status"], 0) + 1
            other = ", ".join(f"{k} {v}" for k, v in sorted(st.items()) if k not in ("pass", "fail"))
            vals = [r for r in rows if r["status"] in ("pass", "fail", "info", "expected_difference")]
            if chk in ("threads",):
                vals = [v for r in rows for k, v in r.items() if isinstance(v, dict) and "dE" in v]
            if chk in ("hessian", "analytic_ref", "analytic_numref", "analytic_xtb"):
                worst = f"max|ΔH| {fmt(max((r[k]['dH'] for r in rows for k in r if isinstance(r[k], dict) and 'dH' in r[k]), default=None))}, " \
                        f"max|Δν| {fmt(max((r[k]['dfreq'] for r in rows for k in r if isinstance(r[k], dict) and 'dfreq' in r[k]), default=None))} cm⁻¹"
                cols = f"{worst} | | | "
            elif chk in ("threads_hessian", "analytic_threads"):
                cols = f"max|ΔH| {fmt(worst_of(rows, 'max_abs_diff'))} | | | "
            else:
                cols = " | ".join(fmt(worst_of(vals, k)) for k in ("dE", "dG", "dq", "ddip"))
            lines.append(f"| {cat if i == 0 else ''} | {len(cc) if i == 0 else ''} | {chk} | {st.get('pass', 0)} | "
                         f"{st.get('fail', 0)} | {other} | {cols} |")
    lines += [""]
    # expected differences, aggregated by reason
    groups: dict = {}
    for c in cases:
        for chk, r in c["checks"].items():
            if r["status"] == "expected_difference":
                groups.setdefault((chk, r.get("note", "")), []).append(abs(r["dE"]))
    if groups:
        lines += ["## Expected differences (documented fixes of the default conventions)", "",
                  "| check | reason | cases | |ΔE| range |", "|---|---|---:|---|"]
        for (chk, note), v in sorted(groups.items()):
            note = note.split(": -TS")[0]
            lines.append(f"| {chk} | {note} | {len(v)} | {min(v):.1e} – {max(v):.1e} |")
        lines.append("")
    # tblite's own solvation, informational
    info = [(c, c["checks"]["tblite_info"]) for c in cases if "tblite_info" in c["checks"]]
    if info:
        lines += ["## tblite's own ALPB/GBSA vs axtb (informational)", "",
                  "Informational comparison, not a pass/fail check.", "",
                  "| model | state | cases | |ΔE| axtb[tblite]−tblite | max|ΔG| | max|Δq| | |ΔE| axtb−tblite |",
                  "|---|---|---:|---|---:|---:|---|"]
        agg: dict = {}
        for c, r in info:
            o = c["options"]
            key = ("ALPB" if o.get("alpb") else "GBSA", o.get("solvation_state", "gsolv"))
            agg.setdefault(key, []).append(r)
        for (m, st), rows in sorted(agg.items()):
            de = [abs(r["dE"]) for r in rows if "dE" in r]
            dd = [abs(r["axtb_default_minus_tblite"]) for r in rows if "axtb_default_minus_tblite" in r]
            rng = lambda v: f"{min(v):.1e} – {max(v):.1e}" if v else "—"  # noqa: E731
            nfail = sum(1 for r in rows if "dE" not in r)
            lines.append(f"| {m} | {st} | {len(rows)}{f' ({nfail} failed)' if nfail else ''} | {rng(de)} | "
                         f"{fmt(worst_of(rows, 'dG'))} | {fmt(worst_of(rows, 'dq'))} | {rng(dd)} |")
        lines.append("")
    # non-pass details
    lines += ["## Cases that are not a plain pass", ""]
    for cat in cats:
        bad = []
        for c in [c for c in cases if c["category"] == cat]:
            for chk, r in c["checks"].items():
                if r["status"] not in ("pass", "info", "both_failed", "expected_difference"):
                    bad.append((c, chk, r))
        if not bad:
            continue
        lines += [f"### {cat}", "", "| case | check | status | details |", "|---|---|---|---|"]
        for c, chk, r in bad:
            det = []
            for k in ("dE", "dG", "dq", "ddip"):
                if k in r:
                    det.append(f"{k} {r[k]:.1e}")
            if "diagnosis" in r:
                dg = r["diagnosis"]
                if "xtb_from_axtb" in dg and "dE" in dg["xtb_from_axtb"]:
                    det.append(f"xtb from axtb moments: dE {dg['xtb_from_axtb']['dE']:.1e}")
                if "axtb_from_xtb" in dg and "dE" in dg["axtb_from_xtb"]:
                    det.append(f"axtb from xtb moments: dE {dg['axtb_from_xtb']['dE']:.1e}")
                for k, v in dg.items():
                    if k.startswith("tbmode_from_") and isinstance(v, dict) and "dE" in v:
                        det.append(f"axtb[tblite] from {k[12:]} moments: dE {v['dE']:.1e}, dq {v.get('dq', 0):.1e}")
                if "tblite_eeq_guess" in dg and "dE" in dg["tblite_eeq_guess"]:
                    det.append(f"tblite from EEQ guess: dE {dg['tblite_eeq_guess']['dE']:.1e}")
                if "lower" in dg:
                    det.append(f"lower: {dg['lower']}")
            if chk == "failure":
                det.append("; ".join(f"{k}: {v['outcome']}" for k, v in r["outcomes"].items()))
            for k in ("error", "ref_error", "note"):
                if r.get(k):
                    det.append(f"{k}: {str(r[k])[:160]}")
            lines.append(f"| {c['id']} ({c['label']}) | {chk} | {r['status']} | {'; '.join(det).replace('|', '/')} |")
        lines.append("")
    # failure behaviour table
    fc = [c for c in cases if "failure" in c["checks"]]
    if fc:
        lines += ["## Failure behaviour", "", "| case | expect | xtb | tblite | axtb | axtb[xtb] | axtb[tblite] | status |",
                  "|---|---|---|---|---|---|---|---|"]
        for c in fc:
            r = c["checks"]["failure"]
            o = r["outcomes"]
            lines.append(f"| {c['label']} | {r['expect']} | " + " | ".join(o[k]["outcome"] for k in
                         ("xtb", "tblite", "axtb", "axtb[xtb]", "axtb[tblite]")) + f" | {r['status']} |")
        lines.append("")
    SUMMARY.write_text("\n".join(lines))


def metadata() -> dict:
    # no host name or paths in the results
    meta = {"date": _dt.datetime.now().isoformat(timespec="seconds"),
            "cpu": platform.processor() or platform.machine(),
            "xtb": P.xtb_version(), "xtb_conventions": P.XTB_CONVENTIONS}
    if P.XTB_OTHER:
        meta["xtb_other_for_diagnoses"] = P.xtb_version(P.XTB_OTHER)
    try:
        meta["load_average"] = [round(x, 2) for x in os.getloadavg()]
    except OSError:
        pass
    try:
        import importlib.metadata as md
        meta["tblite"] = md.version("tblite")
    except Exception:  # noqa: BLE001
        pass
    try:
        import axtb
        meta["axtb"] = axtb.__version__
        meta["git"] = subprocess.run(["git", "-C", str(S.ROOT), "rev-parse", "--short", "HEAD"], capture_output=True,
                                     text=True).stdout.strip() + ("+dirty" if subprocess.run(
                                         ["git", "-C", str(S.ROOT), "diff", "--quiet", "--", "crates/axtb", "crates/d4",
                                          "crates/solvation", "crates/axtb-py", "crates/axtb-cli"]).returncode else "")
    except Exception:  # noqa: BLE001
        pass
    return meta


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--subset", help="comma-separated categories (" + ", ".join(C.CATEGORIES) + ")")
    ap.add_argument("--only", help="comma-separated substrings of case ids")
    ap.add_argument("--jobs", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)))
    ap.add_argument("--list", action="store_true", help="list the cases and exit")
    ap.add_argument("--fresh", action="store_true", help="do not merge with the previous results.json")
    ap.add_argument("--summary", action="store_true", help="only rewrite summary.md from results.json")
    ap.add_argument("--xtb", help="xtb binary (default: xtb 6.8.0, $AXTB_XTB680)")
    ap.add_argument("--xtb-other", help="another xtb version whose converged moments are one more start of the "
                                        "different-SCF-solution diagnoses (default: none)")
    ap.add_argument("--name", help="write results-NAME.json and summary-NAME.md (default: by the xtb version)")
    ap.add_argument("--outdir", type=Path, help="directory of the results and summary files (default: this one)")
    args = ap.parse_args(argv)
    global RESULTS, SUMMARY
    if args.xtb:
        os.environ["AXTB_MATRIX_XTB"] = args.xtb
        P.XTB = args.xtb
        P.XTB_CONVENTIONS = P.xtb_conventions(args.xtb)
        P.XTB_HESS_ROTATES = P.XTB_CONVENTIONS == "xtb-6.7.1"
        P.XTB_671_RELEASE = P.xtb_version().endswith(("(edcfbbe)", "(26b2801)"))
        RESULTS, SUMMARY = output_files(P.XTB_CONVENTIONS)
    if args.xtb_other:
        os.environ["AXTB_MATRIX_XTB_OTHER"] = args.xtb_other
        P.XTB_OTHER = args.xtb_other
    if args.name:
        RESULTS, SUMMARY = HERE / f"results-{args.name}.json", HERE / f"summary-{args.name}.md"
    if args.outdir:
        args.outdir.mkdir(parents=True, exist_ok=True)
        RESULTS, SUMMARY = args.outdir / RESULTS.name, args.outdir / SUMMARY.name
    if P.XTB is None and not args.summary and not args.list:
        P.programs.skip("the matrix compares with an xtb program: pass --xtb or " + P.programs.reason(P.programs.XTB680))
    print(f"xtb {P.xtb_version()} -> axtb conventions {P.XTB_CONVENTIONS!r}; results in {RESULTS.name}"
          + (f"; diagnoses also start from xtb {P.xtb_version(P.XTB_OTHER)}" if P.XTB_OTHER else ""), flush=True)
    if args.summary:
        data = json.loads(RESULTS.read_text())
        write_summary(data["cases"], data["meta"])
        return 0
    subsets = args.subset.split(",") if args.subset else None
    if subsets:
        unknown = set(subsets) - set(C.CATEGORIES)
        if unknown:
            ap.error(f"unknown subset(s) {sorted(unknown)}")
    todo = C.all_cases(subsets)
    if args.only:
        keys = args.only.split(",")
        todo = [c for c in todo if any(k in c["id"] for k in keys)]
    if args.list:
        for c in todo:
            print(f"{c['id']:50s} {c['category']:12s} {','.join(c['checks']):30s} {c['label']}")
        print(f"{len(todo)} cases")
        return 0
    results = {}
    if RESULTS.exists() and not args.fresh:
        results = json.loads(RESULTS.read_text()).get("cases", {})
    valid = {c["id"] for c in C.all_cases()}
    results = {k: v for k, v in results.items() if k in valid}
    t0 = time.perf_counter()
    done = 0
    ctx = mp.get_context("fork")
    with ctx.Pool(args.jobs, maxtasksperchild=50) as pool:
        for r in pool.imap_unordered(_worker, sorted(todo, key=lambda c: -len(c["numbers"]))):
            results[r["id"]] = r
            done += 1
            st = ",".join(f"{k}:{v['status']}" for k, v in r["checks"].items())
            print(f"[{done}/{len(todo)}] {r['id']:50s} {st}  ({r['seconds']:.1f} s)", flush=True)
    meta = metadata()
    meta["last_run"] = {"subsets": subsets, "only": args.only, "cases": len(todo),
                        "seconds": round(time.perf_counter() - t0, 1)}
    ordered = {c["id"]: results[c["id"]] for c in C.all_cases() if c["id"] in results}
    RESULTS.write_text(json.dumps({"meta": meta, "cases": ordered}, indent=1, default=float))
    write_summary(ordered, meta)
    bad = [(r["id"], k) for r in ordered.values() for k, v in r["checks"].items() if v["status"] in ("fail", "error", "axtb_failed")]
    print(f"\n{len(ordered)} cases in results.json, {len(bad)} failing checks; summary in {SUMMARY}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
