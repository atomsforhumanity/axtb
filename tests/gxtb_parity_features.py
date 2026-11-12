#!/usr/bin/env python3
"""g-xTB parity features (docs/gxtb-feature-coverage.md): SCF guess reuse, Mayer-Wiberg
bond orders, the HOMO-LUMO gap and the molecular quadrupole, measured against the programs.

    python tests/gxtb_parity_features.py [--out bench/results/gxtb_parity_features.json]
                                         [--only properties,restart,sequences,codec,solutions,threads]

Needs the axtb Python package (with qccompute, geomeTRIC and Sella for `codec`) and the
programs of tests/programs.py: the xtb 6.7.1 release (`$AXTB_XTB671`), xtb 6.8.0
(`$AXTB_XTB680`), the g-xTB 2.0.1 build of xtb (`$AXTB_GXTB2`) and the g-xtb 1.1.0 program
(`$AXTB_GXTB11_DIR`); the cases of a program that is not there are skipped. Every program runs
with OMP_NUM_THREADS=1 in a
fresh directory on Turbomole `coord` files in bohr, so programs and axtb see the same geometry.
Structures are public (tests/structures, tests/scf_hard, crates/gxtb/tests) or generated.

Sections:

- **properties**: bond orders (the programs' `wbo` file, full precision, pairs above 0.1; the
  1.1.0 program's printout, 3 decimals, pairs above 0.05), the HOMO-LUMO gap (the summary line in
  eV, each program's eV factor) and the molecular quadrupole (xtb's `molmom` "full" line, 3
  decimals; the 2.0.1 program's and xtb `--tblite`'s `--quadrupole` "total" line, 4 decimals)
  of every program vs axtb in the conventions that reproduce it. Water rotated by 45 degrees
  shows that tblite's quadrupole (the 2.0.1 program's) is not a tensor.
- **restart**: the programs' own restart files (2.0.1 `xtbrestart`, 1.1.0 `gxtbrestart`): a run,
  the same geometry again, then a displaced geometry, all in one directory, vs an axtb calculator
  in program conventions with `guess="previous"` (energies and SCF iterations).
- **sequences**: SCF iterations and time with and without guess reuse (default conventions; the
  two calculators take turns point by point, so that both see the same load) along
  MD-like random walks (caffeine, taxol) for g-xTB 2.0.1, v1.1 and GFN2-xTB, energies and
  gradients against fresh starts, and both against a tightly converged fresh sequence (accuracy
  1e-4).
- **codec**: a geomeTRIC minimization and a Sella transition-state search plus IRC (the SN2
  system of bench/ts_example.py) through qccompute and the axtb codec, `propagate_wfn` on and
  off: SCF iterations, wall time, final energies.
- **solutions**: whether reuse lands on another SCF solution: bond-stretch scans in both
  directions (N2, F2, HF, H2O, C2H6) and random walks of metal clusters and complexes, reuse vs
  fresh at every point, at 0 K (the g-xTB default) and 3000 K.
- **threads**: the same reuse sequence on 1 and 4 threads.
- **overhead**: CLI energy + gradient times of the main branch's CLI (AXTB_MAIN) and this one,
  which adds orbital levels, gap, quadrupole and `guess` to every result, and with `--wbo`.

Timings record the 1-minute load average (the machine is shared).

The properties and restart sections fail (exit status 1) when a quantity exceeds its limit in
`LIMITS` (tests/gate.py) or a program run fails; the other sections are measurements.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import axtb

ROOT = Path(__file__).resolve().parents[1]
STRUCT = ROOT / "tests/structures"
HARD = ROOT / "tests/scf_hard"
REF = ROOT / "crates/gxtb/tests"
AATOAU = 1.0 / 0.529177210903
sys.path.insert(0, str(ROOT / "tests"))
import gate  # noqa: E402
import programs  # noqa: E402

# Properties per program, axtb in the conventions that reproduce it: bond orders (`wbo` file; the
# 1.1.0 program's printout, 3 decimals), gap (eV), quadrupole (printout). docs/validation.md, "Bond
# orders, orbital levels, gap and quadrupole", except: the quadrupole limits are the printed
# precision (half the last of 4 or 3 decimals), which that table's "<= 4.9e-5 (4 decimals)" and
# "<= 4.9e-4" mean (xtb 6.8.0 --tblite measured 4.99e-5 on the rotated water); xtb 6.7.1 prints
# the moments it mixed once more after the last iteration, so at --acc 1 its quadrupole is
# 6.2e-4 off for caffeine and 1.3e-3 for taxol (3.0e-4 at --acc 1e-4); the g-xTB 2.0.1 build's
# `xtb --tblite` (tblite671, not in the docs) is chosen from the measured 1.3e-5, 1.2e-4 eV and
# 3.0e-4 (ions; its SCF starts differently from tblite 0.8.0's), about 2x. Every pair a program
# leaves out of its list is below its threshold in axtb too. Restarts (docs/validation.md,
# Summary): g-xTB 2.0.1 and 1.1.0 restarted runs <= 4.3e-11 Eh, the same iterations.
LIMITS = {
    "properties": {
        "gxtb201 bond orders": 9.4e-12, "gxtb201 gap": 3.7e-11, "gxtb201 quadrupole": 5e-5,
        "gxtb11 bond orders": 4.2e-4, "gxtb11 gap": 2.2e-7,
        "xtb680 bond orders": 7.1e-14, "xtb680 gap": 4.7e-13, "xtb680 quadrupole": 5e-4,
        "xtb671 bond orders": 1.8e-9, "xtb671 gap": 1.0e-8, "xtb671 quadrupole": 5e-4,
        "xtb671 quadrupole caffeine": 6.2e-4, "xtb671 quadrupole taxol": 1.3e-3,
        "tblite680 bond orders neutral": 1e-14, "tblite680 bond orders ions": 1.3e-5, "tblite680 gap neutral": 3e-11,
        "tblite680 gap ions": 1.2e-4, "tblite680 quadrupole neutral": 5e-5, "tblite680 quadrupole ions": 3e-4,
        "tblite671 bond orders": 3e-5, "tblite671 gap": 3e-4, "tblite671 quadrupole": 6e-4,
        "unlisted pairs above the program's threshold": 0, "failures": 0,
    },
    "restart": {"max_abs_dE": 4.3e-11, "iteration_mismatches": 0},
}

XTB671 = programs.get(programs.XTB671)
XTB680 = programs.get(programs.XTB680)
GXTB201 = programs.get(programs.GXTB2)
GXTB11 = programs.gxtb11_dir()
# the variable of each program kind (tests/programs.py)
PROGRAM_VAR = {"xtb671": programs.XTB671, "xtb680": programs.XTB680, "tblite671": programs.GXTB2,
               "tblite680": programs.XTB680, "gxtb201": programs.GXTB2, "gxtb11": programs.GXTB11_DIR}
_SKIPPED: set[str] = set()


def have(kind: str) -> bool:
    """Whether the program of `kind` is there (says once on stderr when it is not)."""
    var = PROGRAM_VAR[kind]
    if programs.get(var):
        return True
    if var not in _SKIPPED:
        _SKIPPED.add(var)
        programs.note_skip(f"the cases of {var}", var)
    return False
# eV per Hartree as each program prints the gap: xtb (GFN2-xTB), mctc-lib (tblite, the 2.0.1
# build's tblite mapping), the 1.1.0 program (27.212 in single precision)
EV = {"xtb": 27.21138505, "tblite": 27.211386245988, "gxtb11": float(np.float32(27.212))}
SYMBOLS = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K",
           "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr",
           "Y", "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La",
           "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os",
           "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am",
           "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"]


def load() -> float:
    return round(os.getloadavg()[0], 1)


# ---------------------------------------------------------------------------------------------
# structures


def read_xyz(path: Path, charge: int = 0, uhf: int = 0, name: str | None = None) -> dict:
    lines = path.read_text().splitlines()
    nat = int(lines[0].split()[0])
    head = lines[1]
    if m := re.search(r"charge=(-?\d+)", head):
        charge = int(m.group(1))
    if m := re.search(r"uhf=(\d+)", head):
        uhf = int(m.group(1))
    numbers, pos = [], []
    for line in lines[2:2 + nat]:
        f = line.split()
        numbers.append(SYMBOLS.index(f[0].capitalize()) + 1)
        pos.append([float(x) * AATOAU for x in f[1:4]])
    return {"name": name or path.stem, "numbers": numbers, "positions": np.array(pos), "charge": charge, "uhf": uhf}


def ref_v2(group: str, name: str) -> dict:
    d = REF / "reference_v2" / group / name
    inp = json.loads((d / "input.json").read_text())
    return read_xyz(d / "mol.xyz", inp["charge"], inp["uhf"], name=name)


def o2() -> dict:
    return {"name": "O2 triplet", "numbers": [8, 8], "positions": np.array([[0, 0, 0], [0, 0, 1.21 * AATOAU]]),
            "charge": 0, "uhf": 2}


def rotated(mol: dict, deg: float, name: str) -> dict:
    t = np.radians(deg)
    r = np.array([[np.cos(t), -np.sin(t), 0], [np.sin(t), np.cos(t), 0], [0, 0, 1]])
    return dict(mol, name=name, positions=mol["positions"] @ r.T)


def property_molecules() -> list[dict]:
    water = read_xyz(STRUCT / "water.xyz", name="water")
    return [
        water,
        rotated(water, 45.0, "water rotated 45 deg"),
        read_xyz(STRUCT / "caffeine.xyz", name="caffeine"),
        ref_v2("reference", "acetate"),
        read_xyz(STRUCT / "zncl4.xyz", charge=-2, name="[ZnCl4]2-"),
        read_xyz(STRUCT / "ferrocene.xyz", name="ferrocene"),
        read_xyz(STRUCT / "cu-nh3-4.xyz", charge=2, uhf=1, name="[Cu(NH3)4]2+ doublet"),
        o2(),
        read_xyz(HARD / "open-NO2.xyz", name="NO2 doublet"),
        read_xyz(HARD / "open-CH2-triplet.xyz", name="CH2 triplet"),
        read_xyz(STRUCT / "taxol.xyz", name="taxol"),
        # xtb 6.7.1 prints the moments it mixed once more after the last iteration: at its default
        # accuracy they differ from the converged ones (taxol: charges 6e-5 e)
        dict(read_xyz(STRUCT / "taxol.xyz"), name="taxol, accuracy 1e-4", accuracy=1e-4),
    ]


def write_coord(path: Path, mol: dict) -> None:
    lines = ["$coord"]
    for z, p in zip(mol["numbers"], mol["positions"]):
        lines.append(f"{p[0]:24.16f} {p[1]:24.16f} {p[2]:24.16f} {SYMBOLS[z - 1].lower()}")
    lines += ["$end", ""]
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------------------------
# programs


def run_program(kind: str, mol: dict, workdir: Path, extra: list[str] = ()) -> str:
    """Run a program in `workdir` (kept: restart files stay there) on mol's coordinates."""
    workdir.mkdir(parents=True, exist_ok=True)
    write_coord(workdir / "coord", mol)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", HOME=str(workdir))
    if kind == "gxtb11":
        for f in (".gxtb", ".basisq", ".eeq"):
            if not (workdir / f).exists():
                shutil.copy(GXTB11 / f, workdir / f)
        (workdir / ".CHRG").write_text(f"{mol['charge']}\n")
        if mol["uhf"]:
            (workdir / ".UHF").write_text(f"{mol['uhf']}\n")
        env["HOME"] = "."
        cmd = [str(GXTB11 / "gxtb"), "-c", "coord", *extra]
    else:
        # xtb --tblite: GFN2-xTB through tblite (6.8.0, and the g-xTB build of 6.7.1, which also
        # knows --quadrupole)
        exe = {"xtb671": XTB671, "xtb680": XTB680, "gxtb201": GXTB201, "tblite671": GXTB201, "tblite680": XTB680}[kind]
        cmd = [exe, "coord", "--chrg", str(mol["charge"])]
        if mol["uhf"]:
            cmd += ["--uhf", str(mol["uhf"])]
        if kind == "gxtb201":
            cmd += ["--gxtb", "--quadrupole"]
        if kind in ("tblite671", "tblite680"):
            cmd += ["--tblite", "--quadrupole"]
        cmd += list(extra)
    p = subprocess.run(cmd, cwd=workdir, env=env, capture_output=True, text=True, timeout=7200)
    out = p.stdout + p.stderr
    (workdir / "program.out").write_text(out)
    if p.returncode != 0:
        raise RuntimeError(f"{kind} failed ({p.returncode}): {out[-500:]}")
    return out


def parse_wbo_file(path: Path) -> dict[tuple[int, int], float]:
    out = {}
    for line in path.read_text().splitlines():
        f = line.split()
        if len(f) == 3:
            out[(int(f[0]) - 1, int(f[1]) - 1)] = float(f[2])
    return out


def parse_wbo_gxtb11(text: str) -> dict[tuple[int, int], float] | None:
    """The 1.1.0 program's 'largest Wiberg ... bond orders for each atom' table (3 decimals);
    `None` when it prints none (molecules of more than 20 atoms or so; -wrwbo adds nothing)."""
    out = {}
    if "largest Wiberg" not in text:
        return None
    block = text.split("largest Wiberg", 1)[1].splitlines()[2:]
    for line in block:
        f = line.split()
        if len(f) < 3 or not f[0].isdigit():
            break
        i = int(f[0]) - 1
        rest = f[3:]
        for k in range(0, len(rest) - 2, 3):
            j = int(rest[k + 1]) - 1
            out[(min(i, j), max(i, j))] = float(rest[k + 2])
    return out


def parse_gap(text: str) -> float | None:
    if m := re.search(r"HOMO-LUMO gap\s+(-?\d+\.\d+) eV", text):
        return float(m.group(1))
    if m := re.search(r"gap \(eV\)[^:]*:\s+(-?\d+\.\d+)", text):
        return float(m.group(1))
    return None


def parse_quadrupole(text: str, kind: str) -> list[float] | None:
    if kind in ("xtb671", "xtb680"):
        m = re.search(r"molecular quadrupole \(traceless\):.*?\n\s+full:\s+([^\n]+)", text, re.S)
    else:
        m = re.search(r"Atomic quadrupole moments \(in atomic units\):.*?\n\s+total\s+([^\n]+)", text, re.S)
    return [float(x) for x in m.group(1).split()[:6]] if m else None


def parse_energy_iterations(kind: str, text: str, workdir: Path) -> tuple[float, int]:
    if kind == "gxtb11":
        e = float((workdir / "energy").read_text().split()[2])
        return e, int(re.search(r"(\d+)\s+scf iterations", text).group(1))
    e = float(re.search(r"total energy\s+(-?\d\.\d+E[+-]\d+) Eh", text).group(1)) if kind == "gxtb201" else \
        float(re.search(r"\| TOTAL ENERGY\s+(-?\d+\.\d+) Eh", text).group(1))
    table = text.split("cycle        total energy", 1)[1].split("-" * 60, 2)[1]
    return e, sum(1 for line in table.splitlines() if line.strip() and line.split()[0].isdigit())


# ---------------------------------------------------------------------------------------------
# properties


AXTB_FOR = {
    # program -> (method, conventions) that reproduce it
    "xtb671": ("gfn2", "xtb-6.7.1"), "xtb680": ("gfn2", "xtb"), "tblite671": ("gfn2", "tblite"),
    "tblite680": ("gfn2", "tblite"),
    "gxtb201": ("gxtb-v2.0.1", "gxtb"), "gxtb11": ("gxtb-v1.1", "gxtb"),
}
GAP_EV = {"xtb671": "xtb", "xtb680": "xtb", "tblite671": "tblite", "tblite680": "tblite", "gxtb201": "tblite",
          "gxtb11": "gxtb11"}
QUAD_DECIMALS = {"xtb671": 3, "xtb680": 3, "tblite671": 4, "tblite680": 4, "gxtb201": 4}


def axtb_point(mol: dict, method: str, conventions: str) -> dict:
    return axtb.singlepoint(mol["numbers"], mol["positions"], charge=float(mol["charge"]), uhf=mol["uhf"],
                            method=method, conventions=conventions, bond_orders=True,
                            accuracy=mol.get("accuracy", 1.0))


def compare_properties(kind: str, mol: dict, tmp: Path, extra_conventions: list[str]) -> dict:
    wd = tmp / f"{kind}-{re.sub(r'[^A-Za-z0-9]+', '_', mol['name'])}"
    acc = mol.get("accuracy", 1.0)
    extra = [] if kind == "gxtb11" else ["--norestart"] + ([] if acc == 1.0 else ["--acc", str(acc)])
    text = run_program(kind, mol, wd, extra)
    method, conv = AXTB_FOR[kind]
    row = {"program": kind, "molecule": mol["name"], "natoms": len(mol["numbers"]), "charge": mol["charge"],
           "uhf": mol["uhf"], "method": method, "accuracy": acc}
    convs = [conv] + [c for c in extra_conventions if c != conv]
    for c in convs:
        try:
            r = axtb_point(mol, method, c)
        except ValueError:
            continue
        bo = np.asarray(r["bond_orders"])
        res = {}
        if kind == "gxtb11":
            ref = parse_wbo_gxtb11(text)
            thr = 0.05
            if ref is None:
                res["bond_orders"] = "not printed by the program"
                ref = {}
            else:
                res["bond_order_pairs"] = len(ref)
                res["bond_order_max_abs_diff"] = max(abs(bo[i, j] - v) for (i, j), v in ref.items())
                res["bond_order_precision"] = 5e-4
        else:
            ref = parse_wbo_file(wd / "wbo")
            res["bond_order_pairs"] = len(ref)
            res["bond_order_max_abs_diff"] = max(abs(bo[i, j] - v) for (i, j), v in ref.items())
            thr = 0.1
        # pairs the program leaves out must be below its threshold in axtb too
        listed = set(ref)
        n = len(mol["numbers"])
        if ref:
            res["unlisted_max_bond_order"] = max([bo[i, j] for i in range(n) for j in range(i + 1, n)
                                                  if (i, j) not in listed] or [0.0])
            res["listing_threshold"] = thr
        gap = parse_gap(text)
        if gap is not None and r["homo_lumo_gap"] is not None:
            res["gap_program_eV"] = gap
            res["gap_abs_diff_eV"] = abs(r["homo_lumo_gap"] * EV[GAP_EV[kind]] - gap)
        if kind in QUAD_DECIMALS:
            q = parse_quadrupole(text, kind)
            if q is not None:
                res["quadrupole_program"] = q
                res["quadrupole_axtb"] = [float(x) for x in r["quadrupole"]]
                res["quadrupole_max_abs_diff"] = float(np.max(np.abs(np.asarray(r["quadrupole"]) - q)))
                res["quadrupole_print_precision"] = 0.5 * 10.0 ** -QUAD_DECIMALS[kind]
        row[c] = res
    return row


def section_properties(tmp: Path, conventions_extra: list[str]) -> dict:
    rows = []
    for mol in property_molecules():
        for kind in ("xtb671", "xtb680", "tblite671", "tblite680", "gxtb201", "gxtb11"):
            if kind == "gxtb11" and max(mol["numbers"]) > 86:
                continue
            if "accuracy" in mol and kind not in ("xtb671", "xtb680"):
                continue
            if not have(kind):
                continue

            try:
                row = compare_properties(kind, mol, tmp, conventions_extra if kind.startswith("xtb") else [])
            except Exception as e:  # noqa: BLE001 - recorded
                row = {"program": kind, "molecule": mol["name"], "error": str(e)[-300:]}
            rows.append(row)
            print("properties", kind, mol["name"], {k: v for k, v in row.items() if isinstance(v, dict)}, flush=True)
    # tblite's quadrupole under rotation: eigenvalues of the printed tensors
    out = {"rows": rows}
    try:
        def tensor(q):
            xx, xy, yy, xz, yz, zz = q
            return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])

        rot = {}
        for r in rows:
            if r["molecule"].startswith("water") and "quadrupole_program" in r.get(AXTB_FOR[r["program"]][1], {}):
                ev = np.linalg.eigvalsh(tensor(r[AXTB_FOR[r["program"]][1]]["quadrupole_program"])).tolist()
                rot.setdefault(r["program"], {})[r["molecule"]] = ev
        out["quadrupole_eigenvalues_under_rotation"] = rot
    except Exception as e:  # noqa: BLE001
        out["quadrupole_eigenvalues_under_rotation"] = {"error": str(e)}
    return out


# ---------------------------------------------------------------------------------------------
# restart: the programs' own restart files


def displaced(mol: dict, scale: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    return dict(mol, positions=mol["positions"] + scale * rng.standard_normal(mol["positions"].shape))


def section_restart(tmp: Path) -> list[dict]:
    rows = []
    cases = [("gxtb201", read_xyz(STRUCT / "caffeine.xyz", name="caffeine")),
             ("gxtb201", read_xyz(STRUCT / "water.xyz", name="water")),
             ("gxtb201", o2()),
             ("gxtb201", read_xyz(STRUCT / "cu-nh3-4.xyz", charge=2, uhf=1, name="[Cu(NH3)4]2+ doublet")),
             ("gxtb11", read_xyz(STRUCT / "caffeine.xyz", name="caffeine")),
             ("gxtb11", read_xyz(STRUCT / "water.xyz", name="water")),
             ("gxtb11", o2())]
    for kind, mol in cases:
        if not have(kind):
            continue
        wd = tmp / f"restart-{kind}-{re.sub(r'[^A-Za-z0-9]+', '_', mol['name'])}"
        method = AXTB_FOR[kind][0]
        steps = [("fresh", mol), ("same geometry, restart", mol)]
        steps += [(f"displaced {s} bohr (rms per coordinate), restart", displaced(mol, s, 7)) for s in (0.02, 0.1, 0.3)]
        calc = axtb.Calculator(method, mol["numbers"], mol["positions"], charge=float(mol["charge"]), uhf=mol["uhf"],
                               conventions="gxtb", guess="previous")
        for label, m in steps:
            text = run_program(kind, m, wd)
            e_prog, it_prog = parse_energy_iterations(kind, text, wd)
            r = calc.singlepoint(m["positions"], gradient=False)
            rows.append({"program": kind, "molecule": mol["name"], "step": label, "E_program": e_prog,
                         "iterations_program": it_prog, "E_axtb": r["energy"], "iterations_axtb": r["iterations"],
                         "guess_axtb": r["guess"], "dE": r["energy"] - e_prog})
            print("restart", rows[-1], flush=True)
    return rows


# ---------------------------------------------------------------------------------------------
# sequences: guess reuse in the default conventions


def walk(mol: dict, steps: int, sigma: float, seed: int) -> list[np.ndarray]:
    """An MD-like random walk: every step moves each coordinate by N(0, sigma) bohr."""
    rng = np.random.default_rng(seed)
    pos = mol["positions"].copy()
    out = []
    for _ in range(steps):
        pos = pos + sigma * rng.standard_normal(pos.shape)
        out.append(pos.copy())
    return out



def scf_seconds(timings) -> float:
    """SCF wall time: g-xTB's "scf" total, or the sum of GFN2-xTB's "scf: ..." phases."""
    t = dict(timings)
    return t["scf"] if "scf" in t else sum(v for k, v in t.items() if k.startswith("scf:"))

def run_sequence(mol: dict, method: str, geoms: list[np.ndarray], guess: str, threads: int = 1,
                 gradient: bool = True, accuracy: float = 1.0, etemp: float | None = None) -> list[dict]:
    calc = axtb.Calculator(method, mol["numbers"], geoms[0], charge=float(mol["charge"]), uhf=mol["uhf"],
                           guess=guess, threads=threads, accuracy=accuracy, etemp=etemp)
    out = []
    for g in geoms:
        t0 = time.perf_counter()
        try:
            r = calc.singlepoint(g, gradient=gradient)
        except RuntimeError as e:
            out.append({"error": str(e)[:200], "seconds": time.perf_counter() - t0})
            continue
        out.append({"energy": r["energy"], "iterations": int(r["iterations"]), "guess": r["guess"],
                    "seconds": time.perf_counter() - t0,
                    "scf_seconds": scf_seconds(r["timings"]),
                    "gradient": None if "gradient" not in r else np.asarray(r["gradient"])})
    return out


def run_interleaved(mol: dict, method: str, geoms: list[np.ndarray]) -> tuple[list[dict], list[dict]]:
    """The restart and the fresh sequence, point by point in turn (so that both see the same load
    on a shared machine)."""
    calcs = {g: axtb.Calculator(method, mol["numbers"], geoms[0], charge=float(mol["charge"]), uhf=mol["uhf"], guess=g)
             for g in ("previous", "fresh")}
    out: dict[str, list[dict]] = {"previous": [], "fresh": []}
    for k, g in enumerate(geoms):
        for guess in (("previous", "fresh") if k % 2 == 0 else ("fresh", "previous")):
            t0 = time.perf_counter()
            r = calcs[guess].singlepoint(g, gradient=True)
            out[guess].append({"energy": r["energy"], "iterations": int(r["iterations"]), "guess": r["guess"],
                               "seconds": time.perf_counter() - t0,
                               "scf_seconds": scf_seconds(r["timings"]),
                               "gradient": np.asarray(r["gradient"])})
    return out["previous"], out["fresh"]


def compare_sequences(reuse: list[dict], fresh: list[dict], tight: list[dict] | None = None) -> dict:
    ok = [(a, b) for a, b in zip(reuse, fresh) if "energy" in a and "energy" in b]
    de = [abs(a["energy"] - b["energy"]) for a, b in ok]
    dg = [float(np.max(np.abs(a["gradient"] - b["gradient"]))) for a, b in ok if a["gradient"] is not None]
    it_r = sum(a["iterations"] for a, _ in ok[1:])
    it_f = sum(b["iterations"] for _, b in ok[1:])
    t_r = sum(a["seconds"] for a, _ in ok[1:])
    t_f = sum(b["seconds"] for _, b in ok[1:])
    return {"points": len(reuse), "failed_reuse": sum("error" in a for a in reuse),
            "failed_fresh": sum("error" in b for b in fresh),
            "iterations_reuse": it_r, "iterations_fresh": it_f,
            "iterations_per_point_reuse": it_r / max(len(ok) - 1, 1),
            "iterations_per_point_fresh": it_f / max(len(ok) - 1, 1),
            "seconds_reuse": t_r, "seconds_fresh": t_f,
            "scf_seconds_reuse": sum(a["scf_seconds"] for a, _ in ok[1:]),
            "scf_seconds_fresh": sum(b["scf_seconds"] for _, b in ok[1:]),
            **(_tight_errors(reuse, fresh, tight) if tight else {}),
            "max_abs_dE": max(de, default=None), "max_abs_dgrad": max(dg, default=None),
            "starts_reuse": sorted({a.get("guess") for a in reuse if "guess" in a})}


def _tight_errors(reuse: list[dict], fresh: list[dict], tight: list[dict]) -> dict:
    """Errors of both against a tightly converged fresh sequence (accuracy 1e-4), over the points
    after the first (which both start from the method's own guess)."""
    out = {}
    for label, seq in (("reuse", reuse), ("fresh", fresh)):
        pts = [(a, t) for a, t in zip(seq[1:], tight[1:]) if "energy" in a and "energy" in t]
        out[f"max_abs_dE_vs_tight_{label}"] = max(abs(a["energy"] - t["energy"]) for a, t in pts)
        out[f"max_abs_dgrad_vs_tight_{label}"] = max(float(np.max(np.abs(a["gradient"] - t["gradient"]))) for a, t in pts
                                                     if a["gradient"] is not None)
    return out


def section_sequences() -> list[dict]:
    rows = []
    mols = [(read_xyz(STRUCT / "caffeine.xyz", name="caffeine"), 30), (read_xyz(STRUCT / "taxol.xyz", name="taxol"), 12)]
    for mol, steps in mols:
        for sigma in (0.005, 0.02):
            geoms = walk(mol, steps, sigma, seed=11)
            for method in ("gxtb-v2.0.1", "gxtb-v1.1", "gfn2"):
                l0 = load()
                reuse, fresh = run_interleaved(mol, method, geoms)
                l1 = load()
                tight = run_sequence(mol, method, geoms, "fresh", accuracy=1e-4)
                row = {"molecule": mol["name"], "natoms": len(mol["numbers"]), "method": method,
                       "sequence": f"random walk, {steps} steps, sigma {sigma} bohr per coordinate",
                       "load": [l0, l1], **compare_sequences(reuse, fresh, tight)}
                rows.append(row)
                print("sequence", row, flush=True)
    return rows


# ---------------------------------------------------------------------------------------------
# codec: optimizers through qccompute


SN2_GUESS = [(6, [0.0, 0.0, 0.0]), (17, [0.0, 0.0, 2.25]), (17, [0.05, 0.0, -2.45]), (1, [1.07, 0.0, 0.08]),
             (1, [-0.535, 0.927, 0.08]), (1, [-0.535, -0.927, 0.08])]


def _trajectory_stats(out) -> dict:
    frames = list(out.results.frames)
    its = [f.results.extras.get("scf_iterations") for f in frames if f.results.extras.get("scf_iterations") is not None]
    starts = sorted({f.results.extras.get("scf_guess") for f in frames if f.results.extras.get("scf_guess")})
    energies = [f.results.energy for f in frames if f.results.energy is not None]
    return {"frames": len(frames), "scf_iterations": int(sum(its)), "scf_iterations_per_frame": sum(its) / max(len(its), 1),
            "starts": starts, "final_energy": energies[-1] if energies else None,
            "seconds": out.execution.elapsed if getattr(out, "execution", None) and hasattr(out.execution, "elapsed")
            else None}


def section_codec() -> list[dict]:
    from qccompute import compute
    from qcdata import CalcType, Model, ProgramInput, ProgramSpec, Structure

    rows = []
    caffeine = read_xyz(STRUCT / "caffeine.xyz", name="caffeine")
    start = displaced(caffeine, 0.1, 3)

    def structure(mol: dict, charge: int = 0) -> Structure:
        return Structure(symbols=[SYMBOLS[z - 1] for z in mol["numbers"]], geometry=mol["positions"], charge=charge,
                         multiplicity=1)

    # warm-up (imports, first-call costs), not recorded: water through both drivers
    water = displaced(read_xyz(STRUCT / "water.xyz"), 0.05, 1)
    for prog, ct in (("geometric", CalcType.optimization), ("sella", CalcType.optimization)):
        child = ProgramSpec(program="axtb", calctype=CalcType.gradient, model=Model(method="gxtb-v2.0.1"))
        compute(ProgramInput(program=prog, calctype=ct, structure=structure(water), subprograms=[child],
                             keywords={"steps": 20} if prog == "sella" else {}), propagate_wfn=True, raise_exc=False)

    for method in ("gxtb-v2.0.1", "gxtb-v1.1", "gfn2"):
        for propagate in (False, True):
            child = ProgramSpec(program="axtb", calctype=CalcType.gradient, model=Model(method=method), keywords={})
            inp = ProgramInput(program="geometric", calctype=CalcType.optimization, structure=structure(start),
                               subprograms=[child], keywords={"maxiter": 200})
            l0, t0 = load(), time.perf_counter()
            out = compute(inp, propagate_wfn=propagate, raise_exc=False)
            stats = _trajectory_stats(out)
            stats["wall_seconds"] = time.perf_counter() - t0
            rows.append({"driver": "geomeTRIC minimization", "molecule": "caffeine (displaced 0.1 bohr)", "method": method,
                         "propagate_wfn": propagate, "success": bool(out.success), "load": [l0, load()], **stats})
            print("codec", rows[-1], flush=True)
    # SN2 saddle point (Sella, exact axtb Hessians) and IRC, g-xTB 2.0.1
    sn2 = {"numbers": [z for z, _ in SN2_GUESS], "positions": np.array([p for _, p in SN2_GUESS]) * AATOAU,
           "charge": -1, "uhf": 0}
    for method in ("gxtb-v2.0.1",):
        ts_structure = None
        for propagate in (False, True):
            grad = ProgramSpec(program="axtb", calctype=CalcType.gradient, model=Model(method=method), keywords={})
            hess = ProgramSpec(program="axtb", calctype=CalcType.hessian, model=Model(method=method), keywords={})
            inp = ProgramInput(program="sella", calctype=CalcType.transition_state, structure=structure(sn2, -1),
                               subprograms=[grad, hess], keywords={"fmax": 0.005, "steps": 200})
            l0, t0 = load(), time.perf_counter()
            out = compute(inp, propagate_wfn=propagate, raise_exc=False)
            stats = _trajectory_stats(out)
            stats["wall_seconds"] = time.perf_counter() - t0
            rows.append({"driver": "Sella transition state", "molecule": "SN2 Cl- + CH3Cl", "method": method,
                         "propagate_wfn": propagate, "success": bool(out.success), "load": [l0, load()], **stats})
            print("codec", rows[-1], flush=True)
            if out.success and ts_structure is None:
                ts_structure = out.results.final_structure
        if ts_structure is None:
            continue
        for propagate in (False, True):
            grad = ProgramSpec(program="axtb", calctype=CalcType.gradient, model=Model(method=method), keywords={})
            hess = ProgramSpec(program="axtb", calctype=CalcType.hessian, model=Model(method=method), keywords={})
            inp = ProgramInput(program="sella", calctype=CalcType.irc, structure=ts_structure, subprograms=[grad, hess],
                               keywords={"fmax": 0.01, "steps": 60, "dx": 0.1})
            l0, t0 = load(), time.perf_counter()
            out = compute(inp, propagate_wfn=propagate, raise_exc=False)
            stats = _trajectory_stats(out)
            stats["wall_seconds"] = time.perf_counter() - t0
            rows.append({"driver": "Sella IRC", "molecule": "SN2 Cl- + CH3Cl", "method": method,
                         "propagate_wfn": propagate, "success": bool(out.success), "load": [l0, load()], **stats})
            print("codec", rows[-1], flush=True)
    return rows


# ---------------------------------------------------------------------------------------------
# solutions: does reuse land on another SCF solution?


def stretch(name: str, numbers: list[int], builder, rs: list[float], charge: int = 0, uhf: int = 0) -> tuple[dict, list]:
    geoms = [np.asarray(builder(r)) * AATOAU for r in rs]
    return {"name": name, "numbers": numbers, "positions": geoms[0], "charge": charge, "uhf": uhf}, geoms


def scans() -> list[tuple[dict, list[np.ndarray], str]]:
    rs = [round(1.0 + 0.1 * k, 2) for k in range(31)]  # 1.0 .. 4.0 A

    def diatomic(r):
        return [[0, 0, 0], [0, 0, r]]

    def water(r, angle=104.5):
        t = np.radians(angle / 2)
        return [[0, 0, 0], [r * np.sin(t), 0, r * np.cos(t)], [-r * np.sin(t), 0, r * np.cos(t)]]

    def ethane(rcc, rch=1.09, hcc=111.0):
        t = np.radians(180 - hcc)
        out = [[0, 0, 0], [0, 0, rcc]]
        for k in range(3):
            phi = np.radians(120 * k)
            out.append([rch * np.sin(t) * np.cos(phi), rch * np.sin(t) * np.sin(phi), -rch * np.cos(t)])
            out.append([rch * np.sin(t) * np.cos(phi + np.pi / 3), rch * np.sin(t) * np.sin(phi + np.pi / 3),
                        rcc + rch * np.cos(t)])
        return out

    out = []
    for name, numbers, builder, lo in [("N2", [7, 7], diatomic, 1.0), ("F2", [9, 9], diatomic, 1.3),
                                       ("HF", [9, 1], diatomic, 0.8), ("H2O (both O-H)", [8, 1, 1], water, 0.9),
                                       ("C2H6 (C-C)", [6, 6, 1, 1, 1, 1, 1, 1], ethane, 1.4)]:
        grid = [r for r in rs if r >= lo] if lo >= 1.0 else [lo, 0.9] + rs
        mol, geoms = stretch(name, numbers, builder, grid)
        out.append((mol, geoms, "stretch out"))
        out.append((dict(mol, positions=geoms[-1]), geoms[::-1], "stretch in"))
    return out


def section_solutions() -> list[dict]:
    rows = []
    sequences = scans()
    for f in ("cluster-Fe4", "cluster-Cu5", "cluster-Ni4", "cluster-Pt3", "cluster-Mn2O", "gap-Cr2", "gap-Cr2-stretched",
              "tm-Fe2-aqua-quintet", "tm-Cu-ammine4", "tm-FeCl4-sextet", "gap-ethylene-twisted", "path-sn2-3"):
        mol = read_xyz(HARD / f"{f}.xyz", name=f)
        sequences.append((mol, walk(mol, 15, 0.02, seed=5), "random walk, 15 steps, sigma 0.02 bohr"))
    for mol, geoms, label in sequences:
        for method in ("gxtb-v2.0.1", "gxtb-v1.1"):
            for etemp in (0.0, 3000.0):
                reuse = run_sequence(mol, method, geoms, "previous", gradient=False, etemp=etemp)
                fresh = run_sequence(mol, method, geoms, "fresh", gradient=False, etemp=etemp)
                c = compare_sequences(reuse, fresh)
                diffs = [round(a["energy"] - b["energy"], 9) for a, b in zip(reuse, fresh)
                         if "energy" in a and "energy" in b]
                row = {"molecule": mol["name"], "sequence": label, "method": method, "etemp": etemp, **c,
                       "points_differing_1e-5": sum(abs(d) > 1e-5 for d in diffs),
                       "reuse_lower": sum(d < -1e-5 for d in diffs), "reuse_higher": sum(d > 1e-5 for d in diffs),
                       "dE_reuse_minus_fresh": diffs}
                rows.append(row)
                print("solutions", {k: v for k, v in row.items() if k != "dE_reuse_minus_fresh"}, flush=True)
    return rows


# ---------------------------------------------------------------------------------------------
# overhead: the new outputs on a plain energy + gradient


AXTB_MAIN = os.environ.get("AXTB_MAIN")
AXTB_BIN = os.environ.get("AXTB_BIN", str(ROOT / "target/release/axtb"))


def _best(runs: dict[str, tuple[list[str], dict]], n: int) -> dict[str, float]:
    """Best wall time of each run over `n` rounds; the runs take turns within a round."""
    best = {k: float("inf") for k in runs}
    for _ in range(n):
        for k, (cmd, request) in runs.items():
            t0 = time.perf_counter()
            subprocess.run(cmd, input=json.dumps(request), capture_output=True, text=True, check=True)
            best[k] = min(best[k], time.perf_counter() - t0)
    return best


def section_overhead() -> list[dict]:
    """CLI energy + gradient, best of 15 rounds in which the runs take turns, of the main branch's
    CLI (AXTB_MAIN) and this one (AXTB_BIN), which adds orbital levels, gap, quadrupole and `guess`
    to every result, and this one with bond orders (`--wbo`, opt-in)."""
    rows = []
    for name in ("caffeine", "taxol"):
        mol = read_xyz(STRUCT / f"{name}.xyz", name=name)
        for method in ("gfn2", "gxtb-v2.0.1", "gxtb-v1.1"):
            req = {"numbers": mol["numbers"], "positions": mol["positions"].tolist(), "method": method, "gradient": True}
            l0 = load()
            row = {"molecule": name, "method": method, "best_of": 15}
            runs = {"seconds_new": ([AXTB_BIN, "--json"], req),
                    "seconds_new_bond_orders": ([AXTB_BIN, "--json"], dict(req, bond_orders=True))}
            if AXTB_MAIN:
                runs["seconds_main"] = ([AXTB_MAIN, "--json"], req)
            row.update(_best(runs, 15))
            row["load"] = [l0, load()]
            rows.append(row)
            print("overhead", row, flush=True)
    return rows


# ---------------------------------------------------------------------------------------------
# threads


def section_threads() -> list[dict]:
    rows = []
    for mol in (read_xyz(STRUCT / "caffeine.xyz", name="caffeine"), read_xyz(STRUCT / "taxol.xyz", name="taxol")):
        geoms = walk(mol, 5, 0.01, seed=2)
        for method in ("gxtb-v2.0.1", "gxtb-v1.1"):
            a = run_sequence(mol, method, geoms, "previous", threads=1)
            b = run_sequence(mol, method, geoms, "previous", threads=4)
            rows.append({"molecule": mol["name"], "method": method, "threads": [1, 4],
                         "identical_energies": all(x["energy"] == y["energy"] for x, y in zip(a, b)),
                         "identical_iterations": all(x["iterations"] == y["iterations"] for x, y in zip(a, b)),
                         "max_abs_dE": max(abs(x["energy"] - y["energy"]) for x, y in zip(a, b)),
                         "max_abs_dgrad": max(float(np.max(np.abs(x["gradient"] - y["gradient"]))) for x, y in zip(a, b))})
            print("threads", rows[-1], flush=True)
    return rows


# ---------------------------------------------------------------------------------------------


def program_versions() -> dict:
    out = {}
    for kind, exe in (("xtb671", XTB671), ("xtb680", XTB680), ("gxtb201", GXTB201)):
        try:
            p = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=60)
            m = re.search(r"xtb version (\S+)", p.stdout + p.stderr)
            out[kind] = m.group(1) if m else None
        except (OSError, TypeError):  # TypeError: not there (None)
            out[kind] = None
    out["gxtb11"] = "g-xtb 1.1.0" if GXTB11 else None
    return out


def clean(x):
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items() if k != "gradient"}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    return x


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "bench/results/gxtb_parity_features.json")
    ap.add_argument("--only", default="properties,restart,sequences,codec,solutions,threads,overhead")
    ap.add_argument("--xtb-conventions", default="xtb,xtb-6.7.1",
                    help="GFN2-xTB conventions compared with xtb 6.7.1 and 6.8.0 (those axtb accepts)")
    ap.add_argument("--merge", action="store_true", help="keep the other sections of an existing --out")
    a = ap.parse_args()
    data = json.loads(a.out.read_text()) if a.merge and a.out.exists() else {}
    data["settings"] = {
        "date": _dt.date.today().isoformat(), "axtb": axtb.__version__, "programs": program_versions(),
        "cpu": "i9-11900H (8 cores, 16 threads), shared", "threads": 1,
        "eV_per_Eh": EV,
    }
    sections = a.only.split(",")
    with tempfile.TemporaryDirectory(prefix="axtb-parity-") as tmp:
        tmp = Path(tmp)
        if "properties" in sections:
            data["properties"] = section_properties(tmp, a.xtb_conventions.split(","))
        if "restart" in sections:
            data["restart"] = section_restart(tmp)
    if "sequences" in sections:
        data["sequences"] = section_sequences()
    if "codec" in sections:
        data["codec"] = section_codec()
    if "solutions" in sections:
        data["solutions"] = section_solutions()
    if "threads" in sections:
        data["threads"] = section_threads()
    if "overhead" in sections:
        data["overhead"] = section_overhead()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(clean(data), indent=1) + "\n")
    print("wrote", a.out)
    return check(data, sections)


def check(data: dict, sections: list[str]) -> int:
    """The gate of the properties and restart sections that ran (tests/gate.py)."""
    rc = 0
    rows = data["properties"]["rows"] if "properties" in sections else []
    if rows:
        lim = LIMITS["properties"]
        summary = {"failures": [f"{r['program']} {r['molecule']}" for r in rows if "error" in r],
                   "unlisted pairs above the program's threshold": []}
        for r in rows:
            res = r.get(AXTB_FOR[r["program"]][1], {})
            if res.get("unlisted_max_bond_order", 0.0) >= res.get("listing_threshold", float("inf")):
                summary["unlisted pairs above the program's threshold"].append(f"{r['program']} {r['molecule']}")
            for prop, key in (("bond orders", "bond_order_max_abs_diff"), ("gap", "gap_abs_diff_eV"),
                              ("quadrupole", "quadrupole_max_abs_diff")):
                name = f"{r['program']} {prop}"
                if key not in res:
                    continue
                if f"{name} neutral" in lim:
                    name += " ions" if r["charge"] else " neutral"
                elif f"{name} {r['molecule']}" in lim:
                    name += f" {r['molecule']}"
                summary[name] = max(summary.get(name, 0.0), res[key])
        print("properties:")
        rc |= gate.check(summary, lim)
    restart = data["restart"] if "restart" in sections else []
    if restart:
        print("restart:")
        rc |= gate.check({"max_abs_dE": max(abs(r["dE"]) for r in restart),
                          "iteration_mismatches": [f"{r['program']} {r['molecule']}: {r['step']}" for r in restart
                                                   if r["iterations_axtb"] != r["iterations_program"]]}, LIMITS["restart"])
    return rc


if __name__ == "__main__":
    sys.exit(main())
