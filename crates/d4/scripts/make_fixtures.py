#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate reference fixtures for the axtb-d4 tests from dftd4 4.3.1 and tblite 0.8.0.

Run with the workspace virtualenv (has the dftd4 and tblite wheels)::

    .venv/bin/python crates/d4/scripts/make_fixtures.py

Writes ``crates/d4/tests/fixtures/<name>.txt``; the Rust integration test
``tests/reference.rs`` reads them and prints the deviations.

File format: one record per line, ``key v1 v2 ...`` (floats in Python ``repr``,
i.e. exactly round-tripping doubles; matrices flattened row-major).

Records (all energies Eh, lengths bohr):
  numbers, positions
  d4.cn / d4.q / d4.c6 / d4.alpha       dftd4 get_properties (EEQ charges, cn cutoff 30)
  <set>.cutoff                           disp2 disp3 cn width2 width3 of cutoff set <set>
  <set>.e / <set>.e2 / <set>.e3          dftd4 energy (GFN2 damping) full / s9=0 / s6=s8=0
  <set>.grad                             dftd4 gradient (includes EEQ dq/dR response)
  dqdr                                   central finite differences of the dftd4 EEQ
                                         charges, dq_j/dR_ix stored [i][x][j] (small systems)
  tblite_gfn2.e3                         tblite GFN2 "dispersion energy" (three-body at q=0,
                                         printed with 14 significant digits)
  tblite_gfn2.q                          converged GFN2 Mulliken charges of tblite
  tblite_gfn2.e2_hf                      self-consistent D4 two-body energy at those charges,
                                         obtained from tblite alone via Hellmann-Feynman: the
                                         GFN2 SCC energy is variational and E_2b is linear in
                                         (s6, s8), so E_2b(q*) = dE_tot/dlambda at lambda = 1
                                         with s6 -> lambda s6, s8 -> lambda s8 (5-point
                                         central difference, delta = 0.01, accuracy 1e-4)
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "tests" / "fixtures"
sys.path.insert(0, str(HERE))
from gen_tables import parse_array_constructor, ROOT  # noqa: E402

AA = 1.0 / 0.52917721090380  # Angstrom -> bohr for building geometries only

GFN2 = dict(s6=1.0, s8=2.7, a1=0.52, a2=5.0, s9=5.0)
CUTOFF_SETS = {
    "dftd4": (60.0, 40.0, 30.0, 0.0, 0.0),  # dftd4 defaults
    "tbcut": (50.0, 25.0, 30.0, 0.05, 0.05),  # tblite 0.8.0 GFN2 (smooth_cutoff=0.05)
    "wide": (20.0, 15.0, 30.0, 5.0, 4.0),  # wide switching windows
}

PYYKKO = [
    float(t.replace("_wp", ""))
    for t in parse_array_constructor(
        ROOT / "reference" / "dftd4" / "src" / "dftd4" / "data" / "covrad.f90", "covalent_rad_2009"
    )
]


# ----------------------------------------------------------------- geometries
def water():
    r, a = 0.9572, np.deg2rad(104.52)
    pos = [[0, 0, 0], [r, 0, 0], [r * np.cos(a), r * np.sin(a), 0]]
    return [8, 1, 1], np.array(pos) * AA


def benzene():
    pos, num = [], []
    for k in range(6):
        t = k * np.pi / 3
        pos.append([1.397 * np.cos(t), 1.397 * np.sin(t), 0.0])
        num.append(6)
    for k in range(6):
        t = k * np.pi / 3
        pos.append([2.481 * np.cos(t), 2.481 * np.sin(t), 0.0])
        num.append(1)
    return num, np.array(pos) * AA


def ferrocene():
    num, pos = [26], [[0.0, 0.0, 0.0]]
    rc = 1.42 / (2 * np.sin(np.pi / 5))
    for z in (1.66, -1.66):
        for k in range(5):
            t = 2 * np.pi * k / 5
            num.append(6)
            pos.append([rc * np.cos(t), rc * np.sin(t), z])
        for k in range(5):
            t = 2 * np.pi * k / 5
            num.append(1)
            pos.append([(rc + 1.08) * np.cos(t), (rc + 1.08) * np.sin(t), z + np.sign(z) * 0.04])
    return num, np.array(pos) * AA


def cisplatin():
    num = [78, 17, 17, 7, 7]
    pos = [[0, 0, 0], [2.33, 0, 0], [0, 2.33, 0], [-2.05, 0, 0], [0, -2.05, 0]]
    for n in (np.array([-2.05, 0, 0]), np.array([0, -2.05, 0])):
        u = n / np.linalg.norm(n)
        w = np.cross(u, [0, 0, 1.0])
        for k in range(3):
            t = 2 * np.pi * k / 3
            d = np.cos(t) * w + np.sin(t) * np.array([0, 0, 1.0])
            h = n + 1.01 * (0.334 * u + 0.943 * d)
            num.append(1)
            pos.append(list(h))
    return num, np.array(pos, dtype=float) * AA


def random_molecule(numbers, seed):
    """Grow a connected structure: each atom bonded to a random earlier atom at
    1.05 x the sum of Pyykko radii, rejecting clashes (< 0.95 x sum)."""
    rng = np.random.default_rng(seed)
    pos = [np.zeros(3)]
    for i, z in enumerate(numbers[1:], start=1):
        for _ in range(10000):
            j = int(rng.integers(0, i))
            d = rng.normal(size=3)
            d /= np.linalg.norm(d)
            r = 1.05 * (PYYKKO[z - 1] + PYYKKO[numbers[j] - 1])
            x = pos[j] + r * d
            ok = all(
                np.linalg.norm(x - pos[k]) > 0.95 * (PYYKKO[z - 1] + PYYKKO[numbers[k] - 1])
                for k in range(i)
            )
            if ok:
                pos.append(x)
                break
        else:
            raise RuntimeError("could not place atom")
    return list(numbers), np.array(pos) * AA


def heavy_main_group():
    nums = [83, 53, 53, 53, 82, 17, 17, 17, 17, 86, 54, 9, 9, 52, 1, 1, 81, 35, 51, 6, 1, 1, 1,
            85, 1, 84, 50, 49, 34, 33, 32, 31, 36, 80, 6, 1, 1, 1]
    return random_molecule(nums, 7)


def all_elements():
    return random_molecule(list(range(1, 87)), 11)


def beyond_rn():
    nums = list(range(87, 104)) + list(range(112, 119)) + [1, 6, 8, 1, 1, 7]
    return random_molecule(nums, 5)


def alkane(n=40):
    """Planar zig-zag n-alkane CnH(2n+2), roughly 1.27 n Angstrom long."""
    cc, ch, ang = 1.54, 1.09, np.deg2rad(111.0)
    dx, dy = cc * np.sin(ang / 2), cc * np.cos(ang / 2)
    cs = [np.array([k * dx, (k % 2) * dy, 0.0]) for k in range(n)]
    num, pos = [6] * n, list(cs)
    for k, c in enumerate(cs):
        # in-plane bisector pointing away from the backbone, tilted out of plane
        out = np.array([0.0, -1.0 if k % 2 == 0 else 1.0, 0.0])
        for s in (1.0, -1.0):
            d = 0.58 * out + s * 0.815 * np.array([0, 0, 1.0])
            d /= np.linalg.norm(d)
            num.append(1)
            pos.append(c + ch * d)
    for k, sgn in ((0, -1.0), (n - 1, 1.0)):
        num.append(1)
        pos.append(cs[k] + ch * np.array([sgn * 0.94, (-1.0 if k % 2 == 0 else 1.0) * 0.34, 0.0]))
    return num, np.array(pos) * AA


# name: (builder, finite-difference dq/dR of the EEQ charges, tblite SCC check)
MOLECULES = {
    "water": (water, True, True),
    "benzene": (benzene, True, True),
    "ferrocene": (ferrocene, True, True),
    "cisplatin": (cisplatin, True, True),
    "heavy_main_group": (heavy_main_group, False, True),
    "all_elements": (all_elements, False, False),
    "beyond_rn": (beyond_rn, False, False),
    "alkane_c40": (alkane, False, True),
}


# ----------------------------------------------------------------- references
def fmt(vals) -> str:
    return " ".join(repr(float(v)) for v in np.ravel(vals))


def dftd4_records(numbers, positions, with_dqdr):
    from dftd4.interface import DampingParam, DispersionModel

    numbers = np.asarray(numbers)
    rec = {}
    model = DispersionModel(numbers, positions)
    props = model.get_properties()
    rec["d4.cn"] = props["coordination numbers"]
    rec["d4.q"] = props["partial charges"]
    rec["d4.c6"] = props["c6 coefficients"]
    rec["d4.alpha"] = props["polarizabilities"]
    for name, cut in CUTOFF_SETS.items():
        model = DispersionModel(numbers, positions)
        model.set_realspace_cutoff(cut[0], cut[1], cut[2], cut[3], cut[4])
        rec[f"{name}.cutoff"] = cut
        res = model.get_dispersion(DampingParam(**GFN2), grad=True)
        rec[f"{name}.e"] = [res["energy"]]
        rec[f"{name}.grad"] = res["gradient"]
        p2 = dict(GFN2, s9=0.0)
        rec[f"{name}.e2"] = [model.get_dispersion(DampingParam(**p2), grad=False)["energy"]]
        p3 = dict(GFN2, s6=0.0, s8=0.0)
        rec[f"{name}.e3"] = [model.get_dispersion(DampingParam(**p3), grad=False)["energy"]]
    if with_dqdr:
        nat = len(numbers)
        h = 1e-5
        dqdr = np.zeros((nat, 3, nat))
        m = DispersionModel(numbers, positions)
        for i in range(nat):
            for x in range(3):
                qs = []
                for s in (-2, -1, 1, 2):
                    p = positions.copy()
                    p[i, x] += s * h
                    m.update(p)
                    qs.append(m.get_properties()["partial charges"])
                dqdr[i, x] = (qs[0] - 8 * qs[1] + 8 * qs[2] - qs[3]) / (12 * h)
        rec["dqdr"] = dqdr
    return rec


TBLITE_SNIPPET = r"""
import sys, numpy as np
from tblite.interface import Calculator
d = np.load(sys.argv[1])
calc = Calculator("GFN2-xTB", d["numbers"], d["positions"])
calc.set("verbosity", 2)
calc.set("max-iter", 1)
try:
    calc.singlepoint()
except Exception as e:
    print("tblite exception:", e)
"""


TBLITE_HF_SNIPPET = r"""
import sys, json, numpy as np
from tblite import library
from tblite.interface import Calculator
d = np.load(sys.argv[1])
param = library.new_param(); library.export_gfn2_param(param)
table = library.new_table(); library.dump_param(param, table)
base = library.table_to_dict(table)
out = {}
for lam in (1.0, 0.98, 0.99, 1.01, 1.02):
    par = library.table_to_dict(table)
    par["dispersion"]["d4"]["s6"] = base["dispersion"]["d4"]["s6"] * lam
    par["dispersion"]["d4"]["s8"] = base["dispersion"]["d4"]["s8"] * lam
    calc = Calculator(par, d["numbers"], d["positions"])
    calc.set("verbosity", 0)
    calc.set("accuracy", 1e-4)
    res = calc.singlepoint()
    out[repr(lam)] = [float(res.get("energy")), [float(x) for x in res.get("charges")]]
print("JSON" + json.dumps(out))
"""


def tblite_scc(numbers, positions, tmpdir: Path):
    import json

    f = tmpdir / "mol_hf.npz"
    np.savez(f, numbers=np.asarray(numbers), positions=positions)
    out = subprocess.run(
        [sys.executable, "-I", "-c", TBLITE_HF_SNIPPET, str(f)],
        capture_output=True,
        text=True,
        env=dict(os.environ, OMP_NUM_THREADS="1"),
    )
    line = [l for l in out.stdout.splitlines() if l.startswith("JSON")]
    if not line:
        print(out.stdout, out.stderr)
        raise RuntimeError("tblite SCC run failed")
    data = json.loads(line[0][4:])
    e = {float(k): v[0] for k, v in data.items()}
    h = 0.01
    e2_hf = (e[0.98] - 8 * e[0.99] + 8 * e[1.01] - e[1.02]) / (12 * h)
    return data[repr(1.0)][1], e2_hf


def tblite_e3(numbers, positions, tmpdir: Path):
    if max(numbers) > 86:
        return None
    f = tmpdir / "mol.npz"
    np.savez(f, numbers=np.asarray(numbers), positions=positions)
    out = subprocess.run(
        [sys.executable, "-I", "-c", TBLITE_SNIPPET, str(f)],
        capture_output=True,
        text=True,
        env=dict(os.environ, OMP_NUM_THREADS="1"),
    )
    m = re.search(r"dispersion energy\s+(\S+)\s+Eh", out.stdout + out.stderr)
    if not m:
        print(out.stdout, out.stderr)
        raise RuntimeError("no dispersion energy in tblite output")
    return float(m.group(1))


def main():
    from importlib.metadata import version

    v_d4, v_tb = version("dftd4"), version("tblite")
    print(f"dftd4 {v_d4}, tblite {v_tb}")
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / "axtb_d4_fixtures"
    tmp.mkdir(parents=True, exist_ok=True)
    only = set(sys.argv[1:])
    for name, (builder, with_dqdr, with_scc) in MOLECULES.items():
        if only and name not in only:
            continue
        numbers, positions = builder()
        positions = np.asarray(positions, dtype=float)
        ext = np.ptp(positions, axis=0).max()
        rec = {"numbers": numbers, "positions": positions}
        rec.update(dftd4_records(numbers, positions, with_dqdr))
        e3 = tblite_e3(numbers, positions, tmp)
        if e3 is not None:
            rec["tblite_gfn2.e3"] = [e3]
        if with_scc:
            q, e2_hf = tblite_scc(numbers, positions, tmp)
            rec["tblite_gfn2.q"] = q
            rec["tblite_gfn2.e2_hf"] = [e2_hf]
        with open(OUT / f"{name}.txt", "w") as fh:
            fh.write(f"# {name}: generated by scripts/make_fixtures.py "
                     f"(dftd4 {v_d4}, tblite {v_tb})\n")
            for k, v in rec.items():
                if k == "numbers":
                    fh.write("numbers " + " ".join(str(int(z)) for z in v) + "\n")
                else:
                    fh.write(f"{k} {fmt(v)}\n")
        print(f"{name:18s} nat={len(numbers):4d} extent={ext:7.2f} bohr "
              f"E(dftd4)={rec['dftd4.e'][0]: .12e} tblite.e3={e3}")


if __name__ == "__main__":
    main()
