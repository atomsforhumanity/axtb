#!/usr/bin/env python3
"""Vibrational analysis (frequencies, IR intensities, qRRHO thermochemistry) against the
xtb 6.8.0 or 6.7.1 program (GFN2-xTB) and the g-xTB 2.0.1 program (`xtb --gxtb`, the 6.7.1
build).

The xtb binary is `--xtb` (default: xtb 6.8.0 from `$AXTB_XTB680`, see tests/programs.py; the
6.7.1 record and the g-xTB part ran `--xtb "$AXTB_GXTB2"`). axtb's analysis runs in the conventions of
that version: `xtb` for 6.8.0 (no rotation before `--hess`, projection in mass-weighted
coordinates, CODATA 2018 bohr), `xtb-6.7.1` for 6.7.x (the molecule rotated by `rotmol`,
Cartesian projection, bohr 0.52917726 Å). The g-xTB part needs the g-xTB 2.0.1 build (6.7.1)
and is skipped for other binaries.

Molecules: caffeine, CO2 and HCN (linear), benzene, NH3 and CH4 (symmetric), and the SN2
transition state Cl- + CH3Cl (one imaginary mode, located with axtb analytic Hessians as in
bench/ts_example.py). Minima are optimized first with `xtb --opt vtight` (GFN2).

GFN2-xTB, for each molecule:
- **same Hessian**: `xtb --hess` writes its (projected) Hessian; axtb's analysis in the
  version's conventions of that file (6.7.1: at xtb's internally rotated coordinates) is
  compared with the
  program's printout: frequencies and reduced masses (g98.out, 4 decimals), symmetry and
  symmetry number, number of imaginary modes, ZPVE, H(T), T·S, G(RRHO) and G (12 decimals).
- **axtb's Hessian**: the analytic Hessian and dipole derivatives (`--conventions xtb`) with
  the analysis in xtb conventions, against the program's frequencies, IR intensities (summed
  over degenerate modes) and thermochemistry; the differences are those of the Hessians
  (xtb's finite differences run their SCFs at accuracy 0.3).
- **conventions**: the same axtb Hessian analysed in `axtb` (CODATA 2022, standard masses,
  mass-weighted projection) and `xtb` conventions.

g-xTB 2.0.1, for each molecule:
- axtb's analytic Hessian (`gxtb` conventions), projected as xtb projects and written as a
  Turbomole `hessian`, through `xtb thermo` and through axtb's analysis (`xtb` conventions);
- the program's own Hessian (`xtb --gxtb --hess --acc 1e-4`) through axtb's analysis (with
  axtb's analytic dipole derivatives for the IR intensities), against the program's printout,
  and against axtb's Hessian.

Writes bench/results/vibrational.json for xtb 6.7.x and bench/results/vibrational_6.8.0.json
for 6.8.0 (`--out`). Environment: AXTB_BIN (release CLI, default target/release/axtb),
AXTB_SCRATCH (directory for the temporary runs). Every xtb run uses one thread.

The script fails (exit status 1) when a quantity exceeds its limit in `LIMITS` (tests/gate.py),
or a symmetry number or the number of imaginary modes differs.
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
import gate  # noqa: E402
import programs  # noqa: E402

# Per analysis conventions (xtb version); docs/validation.md (Vibrational analysis and
# thermochemistry): the program's own Hessian through axtb's analysis, frequencies <= 5.0e-5 cm-1
# (4 printed decimals), thermochemistry (ZPVE, H(T), T·S, G(RRHO), G) <= 1.4e-10 Eh with 6.8.0
# (the `hessian` file's 10 decimals), 7.7e-11 with 6.7.1, 2.7e-11 for g-xTB, the same point groups
# and symmetry numbers; `xtb thermo` on axtb's g-xTB Hessian: ZPVE and G(RRHO) <= 8e-12 Eh, H and
# T·S to its 6 printed decimals; the g-xTB program's IR intensities from axtb's dipole derivatives
# <= 0.028 km/mol; axtb's own Hessians: GFN2 <= 0.84 cm-1 and 3.2e-5 Eh in G(RRHO) with 6.8.0
# (0.46 and 2.3e-5 with 6.7.1), g-xTB <= 0.35 cm-1 and 5.2e-6 Eh; `axtb` against `xtb` analysis
# conventions <= 0.1 cm-1 and 4.0e-6 Eh. Chosen: GFN2 IR intensities from axtb's dipole derivatives
# on xtb's Hessian at `$hess sccacc=0.001`, relative (docs: 0.07 % for caffeine; measured 0.076 %
# for the SN2 TS), limit 0.1 %.
LIMITS = {
    "xtb": {"GFN2 same Hessian: frequencies": 5.0e-5, "GFN2 same Hessian: thermochemistry": 1.4e-10,
            "GFN2 same Hessian: symmetry or imaginary modes differ": 0, "GFN2 same Hessian at sccacc=0.001: IR relative": 1e-3,
            "GFN2 axtb's Hessian: frequencies": 0.84, "GFN2 axtb's Hessian: G(RRHO)": 3.2e-5,
            "GFN2 axtb vs xtb analysis conventions: frequencies": 0.1, "GFN2 axtb vs xtb analysis conventions: G(RRHO)": 4.0e-6},
    "xtb-6.7.1": {
        "GFN2 same Hessian: frequencies": 5.0e-5, "GFN2 same Hessian: thermochemistry": 7.7e-11,
        "GFN2 same Hessian: symmetry or imaginary modes differ": 0, "GFN2 same Hessian at sccacc=0.001: IR relative": 1e-3,
        "GFN2 axtb's Hessian: frequencies": 0.46, "GFN2 axtb's Hessian: G(RRHO)": 2.3e-5,
        "GFN2 axtb vs xtb analysis conventions: frequencies": 0.1, "GFN2 axtb vs xtb analysis conventions: G(RRHO)": 4.0e-6,
        "g-xTB program Hessian: frequencies": 5.0e-5, "g-xTB program Hessian: thermochemistry": 2.7e-11,
        "g-xTB program Hessian: IR from axtb's dipole derivatives": 0.028,
        "g-xTB xtb thermo on axtb's Hessian: ZPVE, G(RRHO)": 8e-12, "g-xTB xtb thermo on axtb's Hessian: H, T·S": 5e-7,
        "g-xTB symmetry numbers differ": 0, "g-xTB axtb's Hessian vs the program's: frequencies": 0.35,
        "g-xTB axtb's Hessian vs the program's: G(RRHO)": 5.2e-6},
}

AXTB = os.environ.get("AXTB_BIN", str(ROOT / "target/release/axtb"))
XTB = None  # the xtb binary, set from --xtb in main()
CONV = "xtb-6.7.1"  # axtb's conventions for XTB, set in main()
AATOAU = 1.0 / 0.529177210903
ENV = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OMP_STACKSIZE="1G")


def angstrom(atoms):
    return [z for z, _ in atoms], np.array([p for _, p in atoms], dtype=float) * AATOAU


def benzene():
    atoms = []
    for k in range(6):
        a = np.pi / 3 * k
        atoms.append((6, [1.39 * np.cos(a), 1.39 * np.sin(a), 0.0]))
        atoms.append((1, [2.47 * np.cos(a), 2.47 * np.sin(a), 0.0]))
    return atoms


def ammonia():
    atoms = [(7, [0.0, 0.0, 0.0])]
    for k in range(3):
        a = 2 * np.pi / 3 * k
        atoms.append((1, [0.94 * np.cos(a), 0.94 * np.sin(a), -0.38]))
    return atoms


def methane():
    s = 1.09 / np.sqrt(3)
    return [(6, [0.0, 0.0, 0.0]), (1, [s, s, s]), (1, [s, -s, -s]), (1, [-s, s, -s]), (1, [-s, -s, s])]


MINIMA = {
    "caffeine": None,
    "CO2": [(6, [0.0, 0.0, 0.0]), (8, [0.0, 0.0, 1.16]), (8, [0.0, 0.0, -1.16])],
    "HCN": [(1, [0.0, 0.0, -1.065]), (6, [0.0, 0.0, 0.0]), (7, [0.0, 0.0, 1.153])],
    "benzene": benzene(),
    "NH3": ammonia(),
    "CH4": methane(),
}
# SN2 Cl- + CH3Cl, asymmetric guess (Angstrom), as bench/ts_example.py
TS_GUESS = [(6, [0.0, 0.0, 0.0]), (17, [0.0, 0.0, 2.25]), (17, [0.05, 0.0, -2.45]),
            (1, [1.07, 0.0, 0.08]), (1, [-0.535, 0.927, 0.08]), (1, [-0.535, -0.927, 0.08])]


def axtb(req: dict) -> dict:
    out = subprocess.run([AXTB, "--json"], input=json.dumps(req), capture_output=True, text=True)
    if out.returncode:
        raise RuntimeError(out.stderr + out.stdout)
    return json.loads(out.stdout)


def run_xtb(nums, pos, args, charge=0, extra_files=None, inputs=("coord",)):
    """Run xtb in a fresh directory with `coord` (bohr); returns stdout and the files."""
    with tempfile.TemporaryDirectory(dir=os.environ.get("AXTB_SCRATCH")) as d:
        d = Path(d)
        write_coord(d / "coord", nums, pos)
        for name, text in (extra_files or {}).items():
            (d / name).write_text(text)
        cmd = [XTB, *inputs, *args]
        if charge:
            cmd += ["--chrg", str(charge)]
        t = time.perf_counter()
        out = subprocess.run(cmd, cwd=d, capture_output=True, text=True, env=ENV)
        wall = time.perf_counter() - t
        text = out.stdout + out.stderr
        if out.returncode or "normal termination" not in text:
            raise RuntimeError(f"xtb {' '.join(args)} failed:\n" + text[-3000:])
        if "Unknown option" in text:
            raise RuntimeError("xtb ignored an option:\n" + text[-2000:])
        files = {p.name: p.read_text(errors="replace") for p in d.iterdir() if p.is_file() and p.stat().st_size < 50_000_000}
        return out.stdout, files, wall


def read_coord(text):
    lines = text.split("$coord")[1].split("$")[0].strip().splitlines()
    pos = [[float(x) for x in ln.split()[:3]] for ln in lines]
    return np.array(pos)


def optimize(nums, pos, charge=0):
    _, files, _ = run_xtb(nums, pos, ["--gfn", "2", "--opt", "vtight"], charge)
    return read_coord(files["xtbopt.coord"])


def xtb_path_conventions() -> str:
    """axtb's conventions for the xtb binary: `xtb-6.7.1` for xtb 6.7.x (also the g-xTB
    2.0.1 build), `xtb` for 6.8.0."""
    try:
        v = subprocess.run([XTB, "--version"], capture_output=True, text=True).stdout
    except OSError:
        return "xtb"
    m = re.search(r"xtb version (\S+)", v)
    return "xtb-6.7.1" if m and m.group(1).startswith("6.7.") else "xtb"


def rotmol(pos):
    """xtb 6.7.1's `rotmol(n, xyz, 1e-4, 2e-4, 3e-4)` (degrees) before its numerical Hessian;
    6.8.0 does not rotate."""
    if CONV != "xtb-6.7.1":
        return np.array(pos, dtype=float)
    out = pos.copy()
    for (i, j), ang in (((1, 2), 1e-4), ((0, 2), 2e-4), ((0, 1), 3e-4)):
        a = ang * np.pi / 180.0
        xo, yo = out[:, i].copy(), out[:, j].copy()
        out[:, i] = xo * np.cos(a) + yo * np.sin(a)
        out[:, j] = -xo * np.sin(a) + yo * np.cos(a)
    return out


def rotate_dipole_derivatives(dd):
    """Dipole derivatives (3N rows ∂μ/∂R_x) in xtb's rotated frame: D' = R D Rᵀ per atom."""
    if CONV != "xtb-6.7.1":
        return np.asarray(dd).tolist()
    rot = rotmol(np.eye(3)).T
    dd = np.asarray(dd).reshape(-1, 3, 3)  # [atom, coordinate, dipole component]
    return np.einsum("km,amn,jn->ajk", rot, dd.transpose(0, 2, 1), rot).reshape(-1, 3).tolist()


def read_hessian(text, n3):
    vals = np.array([float(x) for x in text.split("$hessian", 1)[1].split("$")[0].split()])
    return vals.reshape(n3, n3)


def write_hessian(h):
    """Turbomole `$hessian` as xtb writes and reads it (fixed format 5x, 5f15.10)."""
    lines = ["$hessian"]
    for row in h:
        for k in range(0, len(row), 5):
            lines.append("     " + "".join(f"{x:15.10f}" for x in row[k:k + 5]))
    return "\n".join(lines + ["$end", ""])


def g98(text):
    def floats(key):
        return [float(x) for m in re.finditer(key + r"\s+--\s+((?:-?\d+\.\d+\s*)+)", text) for x in m.group(1).split()]
    return {"frequencies": floats("Frequencies"), "reduced_masses": floats(r"Red\. masses"), "ir": floats("IR Inten")}


def xtb_thermo(stdout):
    def last(pat):
        return float(re.findall(pat, stdout)[-1])
    out = {
        "point_group": re.findall(r"^(\S*)\s+symmetry found", stdout, re.M)[-1],
        "symmetry_number": int(re.findall(r"rotational number\s+(\d+)", stdout)[-1]),
        "n_imaginary": int(re.findall(r"# imaginary freq\.\s+(\d+)", stdout)[-1]),
        "zpve": last(r"zero point energy\s+(-?\d+\.\d+) Eh"),
        "g_rrho": last(r"G\(RRHO\) contrib\.\s+(-?\d+\.\d+) Eh"),
        "energy": last(r":: total energy\s+(-?\d+\.\d+) Eh"),
        "free_energy": last(r":: total free energy\s+(-?\d+\.\d+) Eh"),
    }
    m = re.findall(r"TOTAL ENTHALPY\s+(-?\d+\.\d+) Eh", stdout)
    row = re.findall(r"^\s*\d+\.\d+\s+(\S+E[-+]\d+)\s+(\S+E[-+]\d+)\s+(\S+E[-+]\d+)\s+(\S+E[-+]\d+)", stdout, re.M)[-1]
    if m:  # 12 decimals
        out["enthalpy"] = float(m[-1]) - out["energy"]
        out["ts"] = out["enthalpy"] - out["g_rrho"]
        out["thermo_digits"] = 12
    else:  # xtb thermo: the table's 6 significant digits
        out["enthalpy"], out["ts"] = float(row[1]), float(row[2])
        out["thermo_digits"] = 6
    return out


def degenerate_groups(freqs, tol=0.5):
    groups, cur = [], [0]
    for i in range(1, len(freqs)):
        if abs(freqs[i] - freqs[cur[-1]]) < tol:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    return groups


def thermo_diff(ax, xt):
    """axtb thermo (CLI JSON) minus the program's printout."""
    return {
        "zpve": ax["zpve"] - xt["zpve"],
        "enthalpy": ax["enthalpy"] - xt["enthalpy"],
        "ts": ax["ts"] - xt["ts"],
        "g_rrho": ax["g_rrho"] - xt["g_rrho"],
        "free_energy": (ax.get("free_energy", ax["g_rrho"]) - xt["free_energy"]),
    }


def ts_search(nums, pos, charge):
    """Eigenvector following with axtb analytic Hessians (bench/ts_example.py)."""
    for _ in range(60):
        d = axtb({"numbers": nums, "positions": pos.tolist(), "charge": charge, "gradient": True, "hessian": True,
                  "accuracy": 1e-3, "vibrations": False})
        g = np.array(d["gradient"]).ravel()
        h = np.array(d["hessian"])
        n = len(nums)
        c = pos - pos.mean(axis=0)
        vecs = []
        for k in range(3):
            t = np.zeros((n, 3)); t[:, k] = 1.0; vecs.append(t.ravel())
            e = np.zeros(3); e[k] = 1.0; vecs.append(np.cross(e, c).ravel())
        q, _ = np.linalg.qr(np.array(vecs).T)
        p = np.eye(3 * n) - q @ q.T
        w, v = np.linalg.eigh(p @ h @ p)
        keep = np.abs(w) > 1e-6
        w, v = w[keep], v[:, keep]
        f = v.T @ (p @ g)
        if np.abs(p @ g).max() < 1e-6:
            return pos
        step = np.zeros_like(f)
        step[0] = f[0] / abs(w[0])
        step[1:] = -f[1:] / np.abs(w[1:])
        dx = v @ step
        nrm = np.linalg.norm(dx)
        if nrm > 0.1:
            dx *= 0.1 / nrm
        pos = pos + dx.reshape(-1, 3)
    raise RuntimeError("TS search did not converge")


def gfn2_case(name, nums, pos, charge):
    n3 = 3 * len(nums)
    stdout, files, wall = run_xtb(nums, pos, ["--gfn", "2", "--hess"], charge)
    xt = xtb_thermo(stdout)
    gx = g98(files["g98.out"])
    hx = read_hessian(files["hessian"], n3)
    # axtb's analytic Hessian and dipole derivatives (xtb conventions)
    d = axtb({"numbers": nums, "positions": pos.tolist(), "charge": charge, "hessian": True,
              "conventions": CONV})
    # same Hessian: axtb's analysis (the version's conventions) of xtb's hessian at its
    # coordinates (6.7.1: rotated), with axtb's dipole derivatives rotated along
    dd_rot = rotate_dipole_derivatives(d["dipole_derivatives"])
    same = axtb({"numbers": nums, "positions": rotmol(pos).tolist(), "analyze_hessian": hx.tolist(),
                 "dipole_derivatives": dd_rot, "analysis_conventions": CONV, "energy": xt["energy"]})
    fa = np.array(same["frequencies"])
    fx = np.array(gx["frequencies"])
    groups = degenerate_groups(fx)
    nondeg = [g[0] for g in groups if len(g) == 1]
    red = np.array(same["reduced_masses"])
    th = same["thermo"]
    ir_x = np.array(gx["ir"])
    ir_s = np.array(same["ir_intensities"])
    gsame = [(float(ir_x[g].sum()), float(ir_s[g].sum())) for g in groups]
    row = {
        "n_atoms": len(nums),
        "xtb_hess_seconds": wall,
        "same_hessian": {
            "n_modes": [len(fa), len(fx)],
            "max_frequency_diff": float(np.abs(fa - fx).max()) if len(fa) == len(fx) else None,
            "max_reduced_mass_diff_nondegenerate": float(np.abs(red[nondeg] - np.array(gx["reduced_masses"])[nondeg]).max()) if nondeg else 0.0,
            "point_group": {"xtb": xt["point_group"], "axtb": th["point_group"]},
            "symmetry_number": {"xtb": xt["symmetry_number"], "axtb": th["symmetry_number"]},
            "n_imaginary_thermo": {"xtb": xt["n_imaginary"], "axtb": th["n_dropped"]},
            "thermo_diff": thermo_diff(th, xt),
            "xtb": {k: xt[k] for k in ("zpve", "enthalpy", "ts", "g_rrho", "free_energy")},
            # xtb's Hessian with axtb's analytic dipole derivatives vs xtb's finite-difference ones
            "max_ir_abs_diff_axtb_dipole_derivatives": float(max(abs(a - x) for x, a in gsame)),
            "max_ir_rel_diff_axtb_dipole_derivatives": float(max((abs(a - x) / x for x, a in gsame if x > 1.0), default=0.0)),
        },
    }
    # xtb runs the displacement SCFs at accuracy 0.3, which leaves noise in its dipole
    # derivatives; with `$hess sccacc=0.001` its intensities converge to axtb's
    out_t, files_t, _ = run_xtb(nums, pos, ["--gfn", "2", "--hess", "--input", "xc.inp"], charge,
                                extra_files={"xc.inp": "$hess\n   sccacc=0.001\n$end\n"})
    gt = g98(files_t["g98.out"])
    ht = read_hessian(files_t["hessian"], n3)
    tight = axtb({"numbers": nums, "positions": rotmol(pos).tolist(), "analyze_hessian": ht.tolist(),
                  "dipole_derivatives": dd_rot, "analysis_conventions": CONV})
    it_x, it_a = np.array(gt["ir"]), np.array(tight["ir_intensities"])
    gt_groups = degenerate_groups(np.array(gt["frequencies"]))
    gtight = [(float(it_x[g].sum()), float(it_a[g].sum())) for g in gt_groups]
    row["same_hessian"]["max_ir_abs_diff_axtb_dipole_derivatives_sccacc_0.001"] = float(max(abs(a - x) for x, a in gtight))
    row["same_hessian"]["max_ir_rel_diff_axtb_dipole_derivatives_sccacc_0.001"] = float(
        max((abs(a - x) / x for x, a in gtight if x > 1.0), default=0.0))
    fa2 = np.array(d["frequencies"])
    ir_a = np.array(d["ir_intensities"])
    gsum = [(float(ir_x[g].sum()), float(ir_a[g].sum())) for g in groups]
    big = [(x, a) for x, a in gsum if x > 1.0]
    row["axtb_hessian"] = {
        "max_frequency_diff": float(np.abs(fa2 - fx).max()),
        "n_imaginary": {"axtb": d["n_imaginary"], "xtb_g98": int((fx < 0).sum())},
        "ir_groups_km_mol": gsum,
        "max_ir_rel_diff": float(max(abs(a - x) / x for x, a in big)) if big else 0.0,
        "max_ir_abs_diff": float(max(abs(a - x) for x, a in gsum)),
        "thermo_diff": thermo_diff(d["thermo"], xt),
        "symmetry_number": d["thermo"]["symmetry_number"],
    }
    # the same axtb Hessian in the axtb conventions
    ax = axtb({"numbers": nums, "positions": pos.tolist(), "analyze_hessian": d["hessian"],
               "dipole_derivatives": d["dipole_derivatives"], "analysis_conventions": "axtb", "energy": d["energy"]})
    row["conventions_axtb_minus_xtb"] = {
        "max_frequency_diff": float(np.abs(np.array(ax["frequencies"]) - fa2).max()),
        "max_ir_rel_diff": float(np.max(np.abs(np.array(ax["ir_intensities"]) - ir_a) / np.maximum(ir_a, 1.0))),
        "zpve": ax["thermo"]["zpve"] - d["thermo"]["zpve"],
        "enthalpy": ax["thermo"]["enthalpy"] - d["thermo"]["enthalpy"],
        "ts": ax["thermo"]["ts"] - d["thermo"]["ts"],
        "g_rrho": ax["thermo"]["g_rrho"] - d["thermo"]["g_rrho"],
        "symmetry_number": [ax["thermo"]["symmetry_number"], d["thermo"]["symmetry_number"]],
    }
    return row


def trproj(nums, pos, h):
    """xtb's projection before it writes `hessian` (not for linear molecules)."""
    n = len(nums)
    c = pos - pos.mean(axis=0)
    f = np.zeros((3 * n, 6))
    for i in range(n):
        x, y, z = c[i]
        f[3 * i:3 * i + 3, :3] = np.eye(3)
        f[3 * i + 1, 3], f[3 * i + 2, 3] = -z, y
        f[3 * i, 4], f[3 * i + 2, 4] = z, -x
        f[3 * i, 5], f[3 * i + 1, 5] = -y, x
    q, r = np.linalg.qr(f)
    q = q[:, np.abs(np.diag(r)) > 1e-8]
    p = np.eye(3 * n) - q @ q.T
    return p @ h @ p


def gxtb_case(name, nums, pos, charge, linear):
    n3 = 3 * len(nums)
    d = axtb({"numbers": nums, "positions": pos.tolist(), "charge": charge, "method": "gxtb-v2.0.1",
              "conventions": "gxtb", "hessian": True})
    h = np.array(d["hessian"])
    hp = h if linear else trproj(nums, pos, h)
    # axtb's Hessian through xtb thermo and through axtb's analysis
    stdout, _, _ = run_xtb(nums, pos, [], 0, extra_files={"hessian": write_hessian(hp)},
                           inputs=("thermo", "coord", "hessian"))
    xt_ours = xtb_thermo(stdout)
    ours = axtb({"numbers": nums, "positions": pos.tolist(), "analyze_hessian": hp.tolist(),
                 "analysis_conventions": "xtb-6.7.1", "energy": 0.0})
    # the program's Hessian
    stdout, files, wall = run_xtb(nums, pos, ["--gxtb", "--hess", "--acc", "1e-4"], charge)
    xt_prog = xtb_thermo(stdout)
    hx = read_hessian(files["hessian"], n3)
    prog = axtb({"numbers": nums, "positions": rotmol(pos).tolist(), "analyze_hessian": hx.tolist(),
                 "dipole_derivatives": rotate_dipole_derivatives(d["dipole_derivatives"]),
                 "analysis_conventions": "xtb-6.7.1", "energy": xt_prog["energy"]})
    gp = g98(files["g98.out"])
    fx = np.array(gp["frequencies"])
    ir_x, ir_a = np.array(gp["ir"]), np.array(prog["ir_intensities"])
    gir = [(float(ir_x[g].sum()), float(ir_a[g].sum())) for g in degenerate_groups(fx)]
    return {
        "n_atoms": len(nums),
        "axtb_hessian_xtb_thermo_vs_axtb_analysis": {
            "thermo_diff": {k: v for k, v in thermo_diff(ours["thermo"], xt_ours).items() if k != "free_energy"},
            "thermo_digits": xt_ours["thermo_digits"],
            "symmetry_number": {"xtb": xt_ours["symmetry_number"], "axtb": ours["thermo"]["symmetry_number"]},
            "point_group": {"xtb": xt_ours["point_group"], "axtb": ours["thermo"]["point_group"]},
            "g_rrho_xtb_thermo": xt_ours["g_rrho"],
        },
        "program_hessian_analysis_vs_printout": {
            "max_frequency_diff": float(np.abs(np.array(prog["frequencies"]) - fx).max()),
            "thermo_diff": thermo_diff(prog["thermo"], xt_prog),
            "symmetry_number": {"xtb": xt_prog["symmetry_number"], "axtb": prog["thermo"]["symmetry_number"]},
            "g_rrho_program": xt_prog["g_rrho"],
            "seconds": wall,
            # the program's Hessian with axtb's analytic dipole derivatives vs its printed IR
            "max_ir_abs_diff_axtb_dipole_derivatives": float(max(abs(a - x) for x, a in gir)),
            "max_ir_rel_diff_axtb_dipole_derivatives": float(max((abs(a - x) / x for x, a in gir if x > 1.0), default=0.0)),
        },
        "axtb_vs_program_hessian": {
            "max_frequency_diff": float(np.abs(np.array(d["frequencies"]) - fx).max()),
            "g_rrho_diff": d["thermo"]["g_rrho"] - xt_prog["g_rrho"],
            "zpve_diff": d["thermo"]["zpve"] - xt_prog["zpve"],
            "analytic_hessian_seconds": dict(d["hessian_timings"])["total"],
        },
    }


def main() -> int:
    global XTB, CONV
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--xtb", default=programs.get(programs.XTB680),
                    help="xtb binary (default: xtb 6.8.0, $AXTB_XTB680; or 6.7.x such as the g-xTB 2.0.1 build)")
    ap.add_argument("--out", default=None, help="default: bench/results/vibrational.json (6.7.x) or vibrational_6.8.0.json")
    ap.add_argument("--only", default=None, help="comma-separated molecule names")
    args = ap.parse_args()
    if not args.xtb:
        programs.skip("pass --xtb or " + programs.reason(programs.XTB680))
    XTB = args.xtb
    CONV = xtb_path_conventions()
    if args.out is None:
        args.out = str(ROOT / "bench/results" / ("vibrational.json" if CONV == "xtb-6.7.1" else "vibrational_6.8.0.json"))
    only = set(args.only.split(",")) if args.only else None
    xtb_version = re.search(r"xtb version (\S+) \((\w+)\)", subprocess.run([XTB, "--version"], capture_output=True, text=True).stdout)
    mols = {}
    for name, atoms in MINIMA.items():
        if only and name not in only:
            continue
        if atoms is None:
            nums, pos, _ = read_xyz(ROOT / "tests/structures/caffeine.xyz")
            nums = [int(z) for z in nums]
        else:
            nums, pos = angstrom(atoms)
        mols[name] = (nums, optimize(nums, pos), 0, name in ("CO2", "HCN"))
    if not only or "SN2-TS" in only:
        nums, pos = angstrom(TS_GUESS)
        mols["SN2-TS"] = (nums, ts_search(nums, pos, -1), -1, False)
    results = {"gfn2": {}, "gxtb-v2.0.1": {}}
    load_before = [round(x, 2) for x in os.getloadavg()]
    summary: dict = {}

    def worst(key, value):  # None (modes that do not pair up) fails
        summary[key] = None if value is None or summary.get(key, 0.0) is None else max(summary.get(key, 0.0), abs(value))

    def differ(key, name, a, b):
        summary.setdefault(key, [])
        if a != b:
            summary[key].append(name)

    for name, (nums, pos, charge, linear) in mols.items():
        r = gfn2_case(name, nums, pos, charge)
        results["gfn2"][name] = r
        s = r["same_hessian"]
        thermo = max(abs(v) for v in s["thermo_diff"].values())
        worst("GFN2 same Hessian: frequencies", s["max_frequency_diff"])
        worst("GFN2 same Hessian: thermochemistry", thermo)
        differ("GFN2 same Hessian: symmetry or imaginary modes differ", name,
               (s["symmetry_number"]["xtb"], s["n_imaginary_thermo"]["xtb"]), (s["symmetry_number"]["axtb"], s["n_imaginary_thermo"]["axtb"]))
        worst("GFN2 same Hessian at sccacc=0.001: IR relative", s["max_ir_rel_diff_axtb_dipole_derivatives_sccacc_0.001"])
        worst("GFN2 axtb's Hessian: frequencies", r["axtb_hessian"]["max_frequency_diff"])
        worst("GFN2 axtb's Hessian: G(RRHO)", r["axtb_hessian"]["thermo_diff"]["g_rrho"])
        worst("GFN2 axtb vs xtb analysis conventions: frequencies", r["conventions_axtb_minus_xtb"]["max_frequency_diff"])
        worst("GFN2 axtb vs xtb analysis conventions: G(RRHO)", r["conventions_axtb_minus_xtb"]["g_rrho"])
        print(f"GFN2 {name:9s} same Hessian: dfreq {s['max_frequency_diff']:.1e} cm-1, thermo {thermo:.1e} Eh, "
              f"sym {s['point_group']['xtb']}/{s['point_group']['axtb']} sigma {s['symmetry_number']['xtb']}/{s['symmetry_number']['axtb']}"
              f" | axtb Hessian: dfreq {r['axtb_hessian']['max_frequency_diff']:.3f} cm-1, IR rel {r['axtb_hessian']['max_ir_rel_diff']:.1e},"
              f" dG(RRHO) {r['axtb_hessian']['thermo_diff']['g_rrho']:+.1e} | axtb-xtb conventions: dfreq "
              f"{r['conventions_axtb_minus_xtb']['max_frequency_diff']:.3f}, dG(RRHO) {r['conventions_axtb_minus_xtb']['g_rrho']:+.1e}")
        if name == "SN2-TS" or CONV != "xtb-6.7.1":
            continue
        g = gxtb_case(name, nums, pos, charge, linear)
        results["gxtb-v2.0.1"][name] = g
        a = g["axtb_hessian_xtb_thermo_vs_axtb_analysis"]
        b = g["program_hessian_analysis_vs_printout"]
        c = g["axtb_vs_program_hessian"]
        worst_a = max(abs(v) for v in a["thermo_diff"].values())
        worst_b = max(abs(v) for v in b["thermo_diff"].values())
        worst("g-xTB program Hessian: frequencies", b["max_frequency_diff"])
        worst("g-xTB program Hessian: thermochemistry", worst_b)
        worst("g-xTB program Hessian: IR from axtb's dipole derivatives", b["max_ir_abs_diff_axtb_dipole_derivatives"])
        worst("g-xTB xtb thermo on axtb's Hessian: ZPVE, G(RRHO)", max(abs(a["thermo_diff"][k]) for k in ("zpve", "g_rrho")))
        worst("g-xTB xtb thermo on axtb's Hessian: H, T·S", max(abs(a["thermo_diff"][k]) for k in ("enthalpy", "ts")))
        differ("g-xTB symmetry numbers differ", name, (a["symmetry_number"]["xtb"], b["symmetry_number"]["xtb"]),
               (a["symmetry_number"]["axtb"], b["symmetry_number"]["axtb"]))
        worst("g-xTB axtb's Hessian vs the program's: frequencies", c["max_frequency_diff"])
        worst("g-xTB axtb's Hessian vs the program's: G(RRHO)", c["g_rrho_diff"])
        print(f"g-xTB {name:8s} xtb thermo on axtb Hessian: {worst_a:.1e} Eh (G(RRHO) {a['g_rrho_xtb_thermo']:.6f}); program Hessian:"
              f" dfreq {b['max_frequency_diff']:.1e}, thermo {worst_b:.1e} Eh | axtb vs program Hessian: dfreq "
              f"{c['max_frequency_diff']:.3f} cm-1, dG(RRHO) {c['g_rrho_diff']:+.1e}")
    ok = gate.check(summary, LIMITS[CONV]) == 0
    gxtb = " and the g-xTB 2.0.1 program" if CONV == "xtb-6.7.1" else ""
    report = {
        "description": f"axtb vibrational analysis (conventions {CONV}) vs xtb {xtb_version.group(1)} (GFN2-xTB){gxtb}; see tests/vibrational_check.py",
        "xtb_version": f"{xtb_version.group(1)} ({xtb_version.group(2)})" if xtb_version else None,
        "analysis_conventions": CONV,
        "machine": platform.processor() or platform.machine(),
        # wall times (xtb_hess_seconds, analytic_hessian_seconds) ran on a shared machine:
        # 1/5/15-minute load averages before and after the runs
        "load_before": load_before,
        "load_after": [round(x, 2) for x in os.getloadavg()],
        "gates": LIMITS[CONV],
        "all_gates_pass": ok,
        "results": results,
    }
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print(f"wrote {args.out}; gates {'pass' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
