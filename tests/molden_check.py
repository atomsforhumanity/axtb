#!/usr/bin/env python3
"""axtb's Molden files vs the programs' own: the same orbitals, and files that read back.

For every case the program writes its Molden file in a fresh scratch directory (one thread):

- GFN2-xTB: ``xtb mol.xyz --gfn 2 --molden`` (xtb 6.7.1: the g-xTB 2.0.1 build, ``$AXTB_GXTB2``);
- g-xTB 2.0.1: ``xtb mol.xyz --gxtb --molden`` (the same program; tblite's writer);
- g-xTB v1.1: ``gxtb -c mol.xyz -molden`` (the 1.1.0 program in ``$AXTB_GXTB11_DIR``, with
  ``HOME=.`` and its parameter files).

The programs come from tests/programs.py; cases whose program is not there are skipped.

axtb then runs at the program's geometry (the bohr positions of the program's ``[Atoms]``)
in the conventions that reproduce that program (``xtb``, ``gxtb``) and writes its file with
spherical and with Cartesian functions. The files are read by the small reader below, which
builds every basis function from the file alone: Molden's real solid harmonics written out
as polynomials (normalized numerically, not with axtb's tables), the contraction
coefficients taken for normalized primitives (Molden's convention, axtb's files) or for raw
ones (what xtb and tblite write), and each Cartesian component normalized on its own
(``own``: Molden's convention, axtb's and GFN2 xtb's files) or all with the norm of x^l
(``axis``: tblite's files, i.e. g-xTB 2.0.1's); g-xTB 1.1.0 writes d functions ``own`` and f
functions ``axis`` in another component order (``PROGRAM_READ``). Then, per case:

- **round trip**: the largest deviation of C^T S C from the unit matrix for the occupied
  orbitals of each file (S the overlap of the file's own basis), axtb's spherical and
  Cartesian files and the program's;
- **same orbitals**: the overlaps <axtb|program> through the cross overlap of the two files'
  bases: 1 - |<i|i>| for every orbital whose energy is not degenerate, 1 - the smallest
  singular value of each degenerate block and of the occupied-occupied block (the same
  occupied space); the largest difference of the orbitals' values at random points around
  the molecule (non-degenerate orbitals, sign fixed by the overlap) relative to the largest
  value; the largest difference of the orbital energies and occupations;
- **mepd** (optional; ``$MEPD_DIR``, a checkout of the private mepd workflow): the HOMO on mepd's
  grid from axtb's Cartesian file read with mepd's conventions for Molden-convention files
  (``terachem``: normalized primitives, components normalized on their own) against the HOMO
  from the program's file read with mepd's conventions for that program.

    python tests/molden_check.py [--out bench/results/molden.json] [--only NAME]

Needs the axtb Python package (a wheel built from this tree).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
STRUCT = ROOT / "tests/structures"
REF2 = ROOT / "crates/gxtb/tests/reference_v2"
REF11 = ROOT / "crates/gxtb/tests/reference"
sys.path.insert(0, str(ROOT / "tests"))
import gate  # noqa: E402
import programs  # noqa: E402

# The gate (tests/gate.py), per program (gfn2: xtb 6.7.1, the g-xTB build, at the default
# accuracy; "gfn2 acc 1e-4": both at 1e-4), the larger of axtb's spherical and Cartesian files:
# docs/validation.md (Molden files) for 1 - |<i|i>| (non-degenerate orbitals and degenerate
# blocks), the occupied space, the orbital energies (Eh), the orthonormality of axtb's files and
# mepd's HOMO; chosen about 10x the measured values: the orbitals' values at the random points
# (gfn2 1.5e-7, gxtb2 2.3e-8, gxtb11 8.7e-7, UF6 with g-xTB 1.1.0 5.3e-4) and the occupations
# (g-xTB 2.0.1 7e-10, the others 0), and the program files' orthonormality (5.5e-11, as axtb's).
LIMITS = {
    "orthonormality axtb": 6e-11, "orthonormality program": 6e-11,
    "gxtb2 1-|<i|i>|": 1.1e-9, "gxtb2 occupied space": 6.5e-11, "gxtb2 orbital energies": 3.3e-12,
    "gxtb2 values": 3e-7, "gxtb2 occupations": 7e-9, "gxtb2 mepd HOMO": 2.2e-7,
    "gxtb11 1-|<i|i>|": 6.1e-11, "gxtb11 occupied space": 6.5e-11, "gxtb11 orbital energies": 6.4e-8,
    "gxtb11 values": 1e-5, "gxtb11 occupations": 1e-9,
    "gxtb11 UF6 1-|<i|i>|": 1.1e-7, "gxtb11 UF6 occupied space": 1.1e-10, "gxtb11 UF6 orbital energies": 8.6e-6,
    "gxtb11 UF6 values": 5e-3, "gxtb11 UF6 occupations": 1e-9,
    "gfn2 1-|<i|i>|": 1.1e-10, "gfn2 occupied space": 2.9e-11, "gfn2 orbital energies": 3.0e-10,
    "gfn2 values": 2e-6, "gfn2 occupations": 1e-9, "gfn2 mepd HOMO": 7e-7,
    "gfn2 acc 1e-4 1-|<i|i>|": 1.0e-10, "gfn2 acc 1e-4 occupied space": 1.7e-11, "gfn2 acc 1e-4 orbital energies": 7.0e-10,
    "gfn2 acc 1e-4 values": 2e-6, "gfn2 acc 1e-4 occupations": 1e-9, "gfn2 acc 1e-4 mepd HOMO": 6e-7,
}
KNOWN_DIFFERENCES = {
    "gxtb11/UF6-axtb-acc1e-3": "a diagnosis of gxtb11/UF6 (axtb at SCF accuracy 1e-3): its v1.1 SCF takes 40 "
                               "iterations where the 1.1.0 program takes 50, since which subspace of an exactly "
                               "degenerate set each eigensolver returns in iteration 2 is round-off "
                               "(docs/validation.md, Molden files); only the orthonormality is gated",
}

XTB = programs.get(programs.GXTB2)  # GFN2-xTB and g-xTB 2.0.1
GXTB11_DIR = programs.gxtb11_dir()
MEPD_DIR = Path(os.environ["MEPD_DIR"]) if os.environ.get("MEPD_DIR") else None

L_OF = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}

# Cartesian components in Molden's order
CART = {
    0: [(0, 0, 0)],
    1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
    2: [(2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)],
    3: [(3, 0, 0), (0, 3, 0), (0, 0, 3), (1, 2, 0), (2, 1, 0), (2, 0, 1), (1, 0, 2), (0, 1, 2), (0, 2, 1), (1, 1, 1)],
}
# Molden's spherical functions as polynomials {(a, b, c): weight}, in file order (unnormalized):
# p x, y, z; d 0, +1, -1, +2, -2; f 0, +1, -1, +2, -2, +3, -3 (standard real solid harmonics)
PURE = {
    0: [{(0, 0, 0): 1.0}],
    1: [{(1, 0, 0): 1.0}, {(0, 1, 0): 1.0}, {(0, 0, 1): 1.0}],
    2: [
        {(0, 0, 2): 2.0, (2, 0, 0): -1.0, (0, 2, 0): -1.0},  # 2z^2 - x^2 - y^2
        {(1, 0, 1): 1.0},  # xz
        {(0, 1, 1): 1.0},  # yz
        {(2, 0, 0): 1.0, (0, 2, 0): -1.0},  # x^2 - y^2
        {(1, 1, 0): 1.0},  # xy
    ],
    3: [
        {(0, 0, 3): 2.0, (2, 0, 1): -3.0, (0, 2, 1): -3.0},  # z(2z^2 - 3x^2 - 3y^2)
        {(1, 0, 2): 4.0, (3, 0, 0): -1.0, (1, 2, 0): -1.0},  # x(4z^2 - x^2 - y^2)
        {(0, 1, 2): 4.0, (2, 1, 0): -1.0, (0, 3, 0): -1.0},  # y(4z^2 - x^2 - y^2)
        {(2, 0, 1): 1.0, (0, 2, 1): -1.0},  # z(x^2 - y^2)
        {(1, 1, 1): 1.0},  # xyz
        {(3, 0, 0): 1.0, (1, 2, 0): -3.0},  # x(x^2 - 3y^2)
        {(2, 1, 0): 3.0, (0, 3, 0): -1.0},  # y(3x^2 - y^2)
    ],
}


def dfact(n: int) -> float:
    return 1.0 if n <= 0 else float(n) * dfact(n - 2)


# ------------------------------------------------------------------ reading


def parse_molden(text: str) -> dict:
    """atoms [(Z, xyz bohr)] (Z from the element symbol: tblite writes the valence charge in the
    atomic-number column), shells [(atom, l, alphas, coefs)], pure {l: bool}, mos [{energy,
    spin, occ, coef}] of a Molden file. Spherical flags are read in order with the meaning the
    format description gives them: [5D] and [5D7F] 5D+7F, [5D10F] 5D+10F, [7F] 6D+7F, [9G]."""
    atoms, shells, mos = [], [], []
    pure = {0: False, 1: False, 2: False, 3: False}
    section, atom, cur = None, None, None
    lines = text.splitlines()
    i = 0
    scale = 1.0
    while i < len(lines):
        line = lines[i].strip()
        i += 1
        m = re.match(r"^\[([^\]]+)\](.*)$", line)
        if m:
            section = m.group(1).strip().lower()
            flags = {"5d": (True, True), "5d7f": (True, True), "5d10f": (True, False), "7f": (False, True),
                     "6d": (False, None), "10f": (None, False)}
            if section in flags:
                d, f = flags[section]
                if d is not None:
                    pure[2] = d
                if f is not None:
                    pure[3] = f
            if section == "atoms":
                scale = 1.8897261246257702 if "ang" in m.group(2).lower() else 1.0
            continue
        if not line:
            if section == "gto":
                atom = None
            continue
        f = line.split()
        if section == "atoms":
            atoms.append((SYMBOLS.index(f[0].capitalize()) + 1, np.array([float(x) for x in f[3:6]]) * scale))
        elif section == "gto":
            if atom is None:
                atom = int(f[0]) - 1
                continue
            if f[0].lower() in L_OF:
                l, n = L_OF[f[0].lower()], int(f[1])
                prim = [[float(x.replace("D", "E")) for x in lines[i + k].split()[:2]] for k in range(n)]
                i += n
                shells.append((atom, l, np.array([p[0] for p in prim]), np.array([p[1] for p in prim])))
        elif section == "mo":
            if "=" in line:
                k, v = (x.strip() for x in line.split("=", 1))
                k = k.lower()
                if k == "sym" or cur is None or (k == "ene" and cur["coef"]):
                    cur = {"energy": None, "spin": "alpha", "occ": 0.0, "coef": {}}
                    mos.append(cur)
                if k == "ene":
                    cur["energy"] = float(v.replace("D", "E"))
                elif k == "spin":
                    cur["spin"] = v.lower()
                elif k == "occup":
                    cur["occ"] = float(v.replace("D", "E"))
            else:
                cur["coef"][int(f[0]) - 1] = float(f[1].replace("D", "E"))
    nbf = sum(nfun(l, pure) for _, l, _, _ in shells)
    for mo in mos:
        c = np.zeros(nbf)
        for k, v in mo["coef"].items():
            c[k] = v
        mo["coef"] = c
    return {"atoms": atoms, "shells": shells, "pure": pure, "mos": mos, "nbf": nbf}


def nfun(l: int, pure: dict) -> int:
    return 2 * l + 1 if (l >= 2 and pure.get(l)) else len(CART[l])


def prim_norm(l: int, a):
    return (2 * a / math.pi) ** 0.75 * (4 * a) ** (l / 2) / math.sqrt(dfact(2 * l - 1))


# ------------------------------------------------------------------ integrals


def s1d(la: int, lb: int, a, b, xa: float, xb: float):
    """1D overlap table t[i][j] (arrays over primitive pairs) of (x-A)^i (x-B)^j Gaussians."""
    p = a[:, None] + b[None, :]
    px = (a[:, None] * xa + b[None, :] * xb) / p
    pa, pb = px - xa, px - xb
    h = 0.5 / p
    t = [[None] * (lb + 1) for _ in range(la + 1)]
    t[0][0] = np.sqrt(math.pi / p) * np.exp(-a[:, None] * b[None, :] / p * (xa - xb) ** 2)
    for i in range(la + 1):
        for j in range(lb + 1):
            if i == 0 and j == 0:
                continue
            if i > 0:
                v = pa * t[i - 1][j]
                if i > 1:
                    v = v + (i - 1) * h * t[i - 2][j]
                if j > 0:
                    v = v + j * h * t[i - 1][j - 1]
            else:
                v = pb * t[i][j - 1]
                if j > 1:
                    v = v + (j - 1) * h * t[i][j - 2]
            t[i][j] = v
    return t


def cart_block(sa, sb):
    """Overlap of the Cartesian monomials (Molden order) times the contracted radial parts."""
    (ca, la, aa, wa), (cb, lb, ab, wb) = sa, sb
    tx = [s1d(la, lb, aa, ab, ca[k], cb[k]) for k in range(3)]
    cc = np.outer(wa, wb)
    out = np.empty((len(CART[la]), len(CART[lb])))
    for i, ea in enumerate(CART[la]):
        for j, eb in enumerate(CART[lb]):
            out[i, j] = np.sum(cc * tx[0][ea[0]][eb[0]] * tx[1][ea[1]][eb[1]] * tx[2][ea[2]][eb[2]])
    return out


class Basis:
    """The basis functions of a parsed Molden file as transforms of Cartesian monomials.

    prims: "normalized" (Molden's convention) or "raw" (the coefficients multiply the bare
    exp(-a r^2)); cart: "own" (each Cartesian component normalized) or "axis" (all with the
    norm of x^l), or a dict of these per l; order: Cartesian component orders per l that
    differ from Molden's. Every contracted function is normalized."""

    def __init__(self, mol: dict, prims: str = "normalized", cart="own", order=None):
        self.shells, self.trafo = [], []
        for atom, l, alpha, coef in mol["shells"]:
            conv = cart.get(l, "own") if isinstance(cart, dict) else cart
            w = coef * prim_norm(l, alpha) if prims == "normalized" else coef.copy()
            sh = (mol["atoms"][atom][1], l, alpha, w)
            selfov = cart_block(sh, sh)
            if l >= 2 and mol["pure"].get(l):
                t = np.zeros((2 * l + 1, len(CART[l])))
                for m, poly in enumerate(PURE[l]):
                    for e, v in poly.items():
                        t[m, CART[l].index(e)] = v
                    t[m] /= math.sqrt(t[m] @ selfov @ t[m])
            else:
                d = np.diag(selfov)
                axis = selfov[0, 0]
                t = np.diag(1 / np.sqrt(d if conv == "own" else np.full_like(d, axis)))
                if order and l in order:  # file order -> Molden order of the monomials
                    t = t[[CART[l].index(e) for e in order[l]]]
            self.shells.append(sh)
            self.trafo.append(t)
        self.nbf = sum(t.shape[0] for t in self.trafo)

    def overlap(self, other: "Basis") -> np.ndarray:
        s = np.zeros((self.nbf, other.nbf))
        i0 = 0
        for sa, ta in zip(self.shells, self.trafo):
            j0 = 0
            for sb, tb in zip(other.shells, other.trafo):
                s[i0:i0 + ta.shape[0], j0:j0 + tb.shape[0]] = ta @ cart_block(sa, sb) @ tb.T
                j0 += tb.shape[0]
            i0 += ta.shape[0]
        return s

    def values(self, points: np.ndarray) -> np.ndarray:
        """Basis function values (nbf, npoints)."""
        out = []
        for (center, l, alpha, w), t in zip(self.shells, self.trafo):
            d = points - center
            radial = np.exp(-np.outer((d * d).sum(1), alpha)) @ w
            mono = np.array([d[:, 0] ** e[0] * d[:, 1] ** e[1] * d[:, 2] ** e[2] for e in CART[l]])
            out.append(t @ mono * radial)
        return np.vstack(out)


# ------------------------------------------------------------------ comparisons


def orthonormality(mol: dict, basis: Basis, occupied_only: bool = True) -> float:
    s = basis.overlap(basis)
    worst = 0.0
    for spin in sorted({m["spin"] for m in mol["mos"]}):
        mos = [m for m in mol["mos"] if m["spin"] == spin and (m["occ"] > 1e-8 or not occupied_only)]
        if not mos:
            continue
        c = np.array([m["coef"] for m in mos]).T
        worst = max(worst, float(np.abs(c.T @ s @ c - np.eye(c.shape[1])).max()))
    return worst


def groups(energies, tol=1e-5):
    out, start = [], 0
    for k in range(1, len(energies) + 1):
        if k == len(energies) or energies[k] - energies[k - 1] > tol:
            out.append(list(range(start, k)))
            start = k
    return out


def compare(ours: dict, ob: Basis, theirs: dict, tb: Basis, points: np.ndarray) -> dict:
    """Overlaps, values, energies and occupations of our orbitals vs the program's, per spin.
    A program file with one spin set (GFN2 xtb: spin-summed occupations) is compared with our
    alpha set, its occupations with the sums of ours."""
    sx = ob.overlap(tb)
    vo, vt = ob.values(points), tb.values(points)
    res = {"nondegenerate": 0.0, "degenerate_blocks": 0.0, "occupied_space": 0.0, "values": 0.0, "energy": 0.0,
           "occupation": 0.0, "orbitals_compared": 0}
    spins_t = sorted({m["spin"] for m in theirs["mos"]})
    spins_o = sorted({m["spin"] for m in ours["mos"]})
    for spin in spins_t:
        mt = [m for m in theirs["mos"] if m["spin"] == spin]
        mo = [m for m in ours["mos"] if m["spin"] == (spin if len(spins_o) == 2 else spins_o[0])]
        n = min(len(mt), len(mo))  # xtb leaves out orbitals above 2 Eh, g-xTB 1.1.0 the highest ones
        # degenerate sets from our full list, so that a set the program's file cuts is left out
        sets = [g for g in groups([m["energy"] for m in mo]) if g[-1] < n]
        mt, mo = mt[:n], mo[:n]
        if len(spins_t) == 1:  # spin-summed occupations
            occ_ours = np.zeros(n)
            for s in spins_o:
                occ_ours += np.array([m["occ"] for m in ours["mos"] if m["spin"] == s][:n])
        else:
            occ_ours = np.array([m["occ"] for m in mo]) / (2 if len(spins_o) == 1 else 1)
        ct = np.array([m["coef"] for m in mt]).T
        co = np.array([m["coef"] for m in mo]).T
        ov = co.T @ sx @ ct
        e_t = np.array([m["energy"] for m in mt])
        e_o = np.array([m["energy"] for m in mo])
        res["energy"] = max(res["energy"], float(np.abs(e_t - e_o).max()))
        res["occupation"] = max(res["occupation"], float(np.abs(occ_ours - np.array([m["occ"] for m in mt])).max()))
        res["orbitals_compared"] += n
        val_o, val_t = co.T @ vo, ct.T @ vt
        scale = float(np.abs(val_t).max())
        for g in sets:
            if len(g) == 1:
                k = g[0]
                res["nondegenerate"] = max(res["nondegenerate"], 1 - abs(ov[k, k]))
                sign = math.copysign(1.0, ov[k, k])
                res["values"] = max(res["values"], float(np.abs(sign * val_o[k] - val_t[k]).max()) / scale)
            else:
                sv = np.linalg.svd(ov[np.ix_(g, g)], compute_uv=False)
                res["degenerate_blocks"] = max(res["degenerate_blocks"], float(1 - sv.min()))
        occ = [k for k in range(n) if mt[k]["occ"] > 0.1]
        if occ:
            sv = np.linalg.svd(ov[np.ix_(occ, occ)], compute_uv=False)
            res["occupied_space"] = max(res["occupied_space"], float(1 - sv.min()))
    return res


def points_around(mol: dict, n: int = 2000, seed: int = 7) -> np.ndarray:
    xyz = np.array([p for _, p in mol["atoms"]])
    rng = np.random.default_rng(seed)
    centers = xyz[rng.integers(0, len(xyz), n)]
    return centers + rng.normal(scale=1.5, size=(n, 3))


# ------------------------------------------------------------------ mepd


def load_mepd():
    if MEPD_DIR is None:
        return None
    path = MEPD_DIR / "mepd/orbitals.py"
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location("mepd_orbitals_standalone", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def mepd_homo(mepd, text: str, conv: str):
    """|psi|^2 summed over the HOMO and the orbitals degenerate with it (1e-5 Eh; a degenerate
    HOMO is any combination of its set) on mepd's grid, mepd reading the file with the
    conventions `conv`; the grid volume element, the HOMO energy and the number of orbitals."""
    mol = mepd.parse_molden(text)
    orbs, homo, _ = mepd._frontier(mol["mos"])
    ehomo = orbs[homo]["energy"]
    pick = [o["coef"] for o in orbs if abs(o["energy"] - ehomo) < 1e-5]
    xyz = np.array([p for _, p in mol["atoms"]])
    origin, h, n = mepd.grid_for(xyz, max_points=24 ** 3)
    axes = [origin[i] + h * np.arange(n[i]) for i in range(3)]
    pts = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    v = mepd.evaluate(mol, pick, pts, mepd.D_NORM[conv], mepd.NORMALIZED_PRIMS[conv]).astype(float)
    return (v ** 2).sum(0), h ** 3, ehomo, len(pick)


# ------------------------------------------------------------------ programs

SYMBOLS = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
           "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",
           "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La",
           "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re",
           "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np",
           "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr"]


def xtb_path_conventions() -> str:
    """axtb's conventions for `XTB`: `xtb-6.7.1` for xtb 6.7.x (also the g-xTB 2.0.1 build),
    `xtb` for 6.8.0."""
    import re as _re
    import subprocess as _sp
    try:
        v = _sp.run([XTB, "--version"], capture_output=True, text=True).stdout
    except (OSError, TypeError):
        return "xtb"
    m = _re.search(r"xtb version (\S+)", v)
    return "xtb-6.7.1" if m and m.group(1).startswith("6.7.") else "xtb"


def run_program(kind: str, xyz_text: str, charge: int, uhf: int, workdir: Path, acc: float = 1.0) -> str:
    """The program's molden.input for a molecule (XYZ text, Angstrom)."""
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "mol.xyz").write_text(xyz_text)
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if kind in ("gfn2", "gxtb2"):
        cmd = [XTB, "mol.xyz", "--gfn", "2"] if kind == "gfn2" else [XTB, "mol.xyz", "--gxtb"]
        cmd += ["--molden", "--norestart", "--chrg", str(charge)]
        if uhf:
            cmd += ["--uhf", str(uhf)]
        if acc != 1.0:
            cmd += ["--acc", str(acc)]
    else:
        for f in (".gxtb", ".basisq", ".eeq"):
            shutil.copy(GXTB11_DIR / f, workdir / f)
        (workdir / ".CHRG").write_text(f"{charge}\n")
        if uhf:  # any .UHF file, even 0, makes the 1.1.0 program unrestricted
            (workdir / ".UHF").write_text(f"{uhf}\n")
        env["HOME"] = "."
        cmd = [str(GXTB11_DIR / "gxtb"), "-c", "mol.xyz", "-molden"]
    p = subprocess.run(cmd, cwd=workdir, env=env, capture_output=True, text=True, timeout=3600)
    (workdir / "program.out").write_text(p.stdout + p.stderr)
    out = workdir / "molden.input"
    if p.returncode != 0 or not out.exists():
        raise RuntimeError(f"{kind} failed ({p.returncode}): {(p.stdout + p.stderr)[-800:]}")
    return out.read_text()


# How each program writes its file: coefficients of raw primitives; Cartesian components each
# normalized (xtb), all with the norm of x^l (tblite), or g-xTB 1.1.0's mix: d each normalized,
# f with the norm of x^3 and in the order of xtb's writer (xxx, yyy, zzz, xxy, xxz, xyy, yyz,
# xzz, yzz, xyz) instead of Molden's.
F_XTB = [(3, 0, 0), (0, 3, 0), (0, 0, 3), (2, 1, 0), (2, 0, 1), (1, 2, 0), (0, 2, 1), (1, 0, 2), (0, 1, 2), (1, 1, 1)]
PROGRAM_READ = {"gfn2": ("raw", "own"), "gxtb2": ("raw", "axis"), "gxtb11": ("raw", {2: "own", 3: "axis"}, {3: F_XTB})}
# GFN2-xTB in the conventions of `XTB` (xtb-6.7.1 for the g-xTB 2.0.1 build)
AXTB = {"gfn2": ("gfn2", None), "gxtb2": ("gxtb-v2.0.1", "gxtb"), "gxtb11": ("gxtb-v1.1", "gxtb")}
MEPD_CONV = {"gfn2": "gfn2", "gxtb2": "gxtb"}


def axtb_files(kind: str, mol: dict, charge: int, uhf: int, acc: float = 1.0) -> tuple[str, str]:
    import axtb

    method, conv = AXTB[kind]
    conv = conv or xtb_path_conventions()
    numbers = [z for z, _ in mol["atoms"]]
    pos = np.array([p for _, p in mol["atoms"]])
    kw = dict(charge=charge, uhf=uhf, method=method, conventions=conv, accuracy=acc)
    sph = axtb.singlepoint(numbers, pos, molden=True, **kw)["molden"]
    cart = axtb.singlepoint(numbers, pos, molden="cartesian", **kw)["molden"]
    return sph, cart


def xyz_of(path: Path) -> str:
    return path.read_text()


CASES = [
    # name, program, xyz, charge, uhf[, SCF accuracy of both[, of axtb]]
    ("gfn2/water", "gfn2", STRUCT / "water.xyz", 0, 0),
    ("gfn2/caffeine", "gfn2", STRUCT / "caffeine.xyz", 0, 0),
    ("gfn2/O2-triplet", "gfn2", REF2 / "reference/dioxygen_triplet/mol.xyz", 0, 2),
    ("gfn2/Cu(NH3)4-2+", "gfn2", STRUCT / "cu-nh3-4.xyz", 2, 1),
    ("gfn2/ferrocene", "gfn2", STRUCT / "ferrocene.xyz", 0, 0),
    # the same at the programs' tightest SCF accuracy: the differences above are SCF convergence
    ("gfn2/caffeine-acc1e-4", "gfn2", STRUCT / "caffeine.xyz", 0, 0, 1e-4),
    ("gfn2/ferrocene-acc1e-4", "gfn2", STRUCT / "ferrocene.xyz", 0, 0, 1e-4),
    ("gxtb2/water", "gxtb2", STRUCT / "water.xyz", 0, 0),
    ("gxtb2/caffeine", "gxtb2", STRUCT / "caffeine.xyz", 0, 0),
    ("gxtb2/O2-triplet", "gxtb2", REF2 / "reference/dioxygen_triplet/mol.xyz", 0, 2),
    ("gxtb2/Cu(NH3)4-2+", "gxtb2", STRUCT / "cu-nh3-4.xyz", 2, 1),
    ("gxtb2/ferrocene", "gxtb2", STRUCT / "ferrocene.xyz", 0, 0),
    ("gxtb2/UF6", "gxtb2", REF2 / "elements/UF6/mol.xyz", 0, 0),
    ("gxtb2/GdCl3-octet", "gxtb2", REF2 / "elements/GdCl3/mol.xyz", 0, 7),
    ("gxtb2/CeCl3-doublet", "gxtb2", REF2 / "elements/CeCl3/mol.xyz", 0, 1),
    ("gxtb11/water", "gxtb11", STRUCT / "water.xyz", 0, 0),
    ("gxtb11/caffeine", "gxtb11", STRUCT / "caffeine.xyz", 0, 0),
    ("gxtb11/O2-triplet", "gxtb11", REF2 / "reference/dioxygen_triplet/mol.xyz", 0, 2),
    ("gxtb11/Cu(NH3)4-2+", "gxtb11", STRUCT / "cu-nh3-4.xyz", 2, 1),
    ("gxtb11/ferrocene", "gxtb11", STRUCT / "ferrocene.xyz", 0, 0),
    ("gxtb11/UF6", "gxtb11", REF2 / "elements/UF6/mol.xyz", 0, 0),
    # axtb's v1.1 SCF takes 40 iterations here, the program 50 (total energies 7.5e-9 Eh apart);
    # converged further, axtb's orbitals approach the program's
    ("gxtb11/UF6-axtb-acc1e-3", "gxtb11", REF2 / "elements/UF6/mol.xyz", 0, 0, 1.0, 1e-3),
    ("gxtb11/CeCl3-doublet", "gxtb11", REF2 / "elements/CeCl3/mol.xyz", 0, 1),
]


def have(kind: str) -> bool:
    return (GXTB11_DIR if kind == "gxtb11" else XTB) is not None


def check_case(name, kind, xyz, charge, uhf, scratch: Path, mepd=None, acc: float = 1.0, axtb_acc=None) -> dict:
    prog_text = run_program(kind, xyz_of(xyz), charge, uhf, scratch / name.replace("/", "_"), acc)
    prog = parse_molden(prog_text)
    sph_text, cart_text = axtb_files(kind, prog, charge, uhf, axtb_acc or acc)
    sph, cart = parse_molden(sph_text), parse_molden(cart_text)
    pb = Basis(prog, *PROGRAM_READ[kind])
    sb, cb = Basis(sph), Basis(cart)
    pts = points_around(prog)
    res = {
        "case": name, "program": kind, "accuracy": acc, "axtb_accuracy": axtb_acc or acc, "atoms": len(prog["atoms"]), "basis_functions": sph["nbf"],
        "cartesian_functions": cart["nbf"], "max_l": max(l for _, l, _, _ in sph["shells"]),
        "spin_sets": len({m["spin"] for m in sph["mos"]}),
        "orthonormal_axtb_spherical": orthonormality(sph, sb),
        "orthonormal_axtb_cartesian": orthonormality(cart, cb),
        "orthonormal_axtb_spherical_all": orthonormality(sph, sb, occupied_only=False),
        "orthonormal_program": orthonormality(prog, pb),
        "vs_program_spherical": compare(sph, sb, prog, pb, pts),
        "vs_program_cartesian": compare(cart, cb, prog, pb, pts),
        "spherical_vs_cartesian": compare(cart, cb, sph, sb, pts),
    }
    if mepd is not None and kind in MEPD_CONV:
        ours, dv, e1, k1 = mepd_homo(mepd, cart_text, "terachem")
        theirs, _, e2, k2 = mepd_homo(mepd, prog_text, MEPD_CONV[kind])
        res["mepd_homo"] = {"max_diff_rel": float(np.abs(ours - theirs).max() / np.abs(theirs).max()),
                            "norm_axtb": float(ours.sum() * dv / k1), "norm_program": float(theirs.sum() * dv / k2),
                            "degenerate": k1, "energy_diff": abs(e1 - e2)}
        try:  # mepd refuses spherical files (as of this writing)
            mepd.parse_molden(sph_text)
            res["mepd_reads_spherical"] = True
        except mepd.OrbitalError:
            res["mepd_reads_spherical"] = False
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default=str(ROOT / "bench/results/molden.json"))
    ap.add_argument("--only", default=None, help="regular expression on case names")
    ap.add_argument("--scratch", default=None)
    args = ap.parse_args()
    mepd = load_mepd()
    scratch = Path(args.scratch or tempfile.mkdtemp(prefix="axtb-molden-"))
    results = []
    for name, kind, xyz, charge, uhf, *acc in CASES:
        if args.only and not re.search(args.only, name):
            continue
        if not have(kind):
            print(f"{name}: skipped, " + programs.reason(programs.GXTB11_DIR if kind == "gxtb11" else programs.GXTB2))
            continue
        r = check_case(name, kind, xyz, charge, uhf, scratch, mepd, *acc)
        results.append(r)
        v = r["vs_program_spherical"]
        print(f"{name:22s} nbf {r['basis_functions']:4d}  orth axtb {r['orthonormal_axtb_spherical']:.1e}/"
              f"{r['orthonormal_axtb_cartesian']:.1e} prog {r['orthonormal_program']:.1e}  "
              f"1-|<i|i>| {v['nondegenerate']:.1e} deg {v['degenerate_blocks']:.1e} occ {v['occupied_space']:.1e} "
              f"val {v['values']:.1e} dE {v['energy']:.1e} docc {v['occupation']:.1e}"
              + (f"  mepd {r['mepd_homo']['max_diff_rel']:.1e}" if "mepd_homo" in r else ""), flush=True)
    if args.out and not args.only:
        Path(args.out).write_text(json.dumps(results, indent=1) + "\n")
        print(f"wrote {args.out}")
    return check(results)


def check(results: list[dict]) -> int:
    """The gate (tests/gate.py) over the cases that ran."""
    summary: dict[str, float] = {}

    def worst(name, value):
        summary[name] = max(summary.get(name, 0.0), value)

    for r in results:
        worst("orthonormality axtb", max(r["orthonormal_axtb_spherical"], r["orthonormal_axtb_cartesian"]))
        worst("orthonormality program", r["orthonormal_program"])
        if r["case"] in KNOWN_DIFFERENCES:
            continue
        group = ("gxtb11 UF6" if r["case"] == "gxtb11/UF6" else "gfn2 acc 1e-4" if r["program"] == "gfn2" and r["accuracy"] != 1.0
                 else r["program"])
        for v in (r["vs_program_spherical"], r["vs_program_cartesian"]):
            worst(f"{group} 1-|<i|i>|", max(v["nondegenerate"], v["degenerate_blocks"]))
            for q, key in (("occupied space", "occupied_space"), ("orbital energies", "energy"), ("values", "values"),
                           ("occupations", "occupation")):
                worst(f"{group} {q}", v[key])
        if "mepd_homo" in r:
            worst(f"{group} mepd HOMO", r["mepd_homo"]["max_diff_rel"])
    return gate.check(summary, LIMITS, {c: why for c, why in KNOWN_DIFFERENCES.items() if any(r["case"] == c for r in results)})


if __name__ == "__main__":
    raise SystemExit(main())
