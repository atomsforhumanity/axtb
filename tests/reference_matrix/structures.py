"""Structure generators for the reference matrix.

Every geometry is built here from a few rules instead of typed coordinates:

* bond lengths default to the sum of the Pyykkö-Atsumi covalent radii (metals -10 %), read
  from axtb's generated table `crates/axtb/src/param/element_data.rs` (literal copy of
  mctc-lib's `covrad.f90`), so every element 1-86 gets a sensible bond length;
* ligand-field shapes (linear, bent, trigonal, pyramidal, tetrahedral, square planar,
  trigonal bipyramidal, octahedral) give the bond directions around a centre;
* polyatomic ligands (H2O, NH3, CO, CN, OH) are attached donor-first along the bond
  direction with standard internal geometry;
* small organics and oxo-anions come from Z-matrices with standard bond lengths/angles;
* random MB16-43-style molecules come from a seeded growth procedure (see `mindless`).

All functions return `(numbers, positions_bohr)` as numpy arrays.  Positions are centred
on the centroid so that dipoles of charged species refer to a well-defined origin (both
reference programs and axtb report the dipole about the coordinate origin).
"""

from __future__ import annotations

import math
import re
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
# CODATA 2022 bohr radius (qcref); any value works because every program gets bohr input
AATOAU = 1.0 / 0.529177210544

SYMBOLS = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
           "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
           "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn "
           "Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr Rf Db Sg Bh Hs Mt Ds Rg Cn Nh Fl "
           "Mc Lv Ts Og").split()


def z_of(sym: str) -> int:
    return SYMBOLS.index(sym[0].upper() + sym[1:].lower()) + 1


@lru_cache(maxsize=None)
def covalent_radii() -> np.ndarray:
    """Pyykkö-Atsumi covalent radii in Å (index Z-1) from axtb's generated element table."""
    src = (ROOT / "crates/axtb/src/param/element_data.rs").read_text()
    body = src[src.index("pub const COVALENT_RAD_2009_AA"):]
    body = body[body.index("[", body.index("=")) + 1:body.index("];")]
    vals = np.array([float(x) for x in re.findall(r"\d+\.\d+", body)])
    assert len(vals) == 118
    return vals


def rcov(z: int) -> float:
    return float(covalent_radii()[z - 1])


def bond(za: int, zb: int, scale: float = 1.0) -> float:
    """Bond length in Å: scaled sum of covalent radii."""
    return scale * (rcov(za) + rcov(zb))


# ---------------------------------------------------------------------------------------
# shapes


def _unit(v):
    v = np.asarray(v, dtype=float)
    return v / np.linalg.norm(v)


def shape_vectors(shape: str, n: int | None = None, angle: float | None = None) -> np.ndarray:
    """Unit bond vectors of a coordination shape."""
    s3 = math.sqrt(3.0)
    if shape == "single":
        return np.array([[0.0, 0.0, 1.0]])
    if shape == "linear":
        return np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    if shape == "bent":
        a = math.radians(angle or 104.5) / 2
        return np.array([[math.sin(a), 0.0, math.cos(a)], [-math.sin(a), 0.0, math.cos(a)]])
    if shape == "trigonal":
        return np.array([[1.0, 0.0, 0.0], [-0.5, s3 / 2, 0.0], [-0.5, -s3 / 2, 0.0]])
    if shape == "pyramidal":
        # three bonds with mutual angle `angle` (default 107°)
        a = math.radians(angle or 107.0)
        # cos(a) = cos²θ·(-1/2)·... solve for the polar angle θ of each bond to -z axis
        # for three vectors at azimuth 0/120/240 and polar angle t: cos a = cos²t + sin²t·cos120
        c = (2 * math.cos(a) + 1) / 3
        t = math.acos(math.sqrt(max(c, 0.0)))
        return np.array([[math.sin(t) * math.cos(p), math.sin(t) * math.sin(p), -math.cos(t)]
                         for p in (0, 2 * math.pi / 3, 4 * math.pi / 3)])
    if shape == "tetrahedral":
        return np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]) / s3
    if shape == "square":
        return np.array([[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0]])
    if shape == "tbp":
        return np.array([[0, 0, 1.0], [0, 0, -1.0], [1.0, 0, 0], [-0.5, s3 / 2, 0], [-0.5, -s3 / 2, 0]])
    if shape == "octahedral":
        return np.array([[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0], [0, 0, 1.0], [0, 0, -1.0]])
    raise ValueError(shape)


def default_shape(n: int) -> str:
    return {1: "single", 2: "linear", 3: "trigonal", 4: "tetrahedral", 5: "tbp", 6: "octahedral"}[n]


def _perp(d):
    ref = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    p = np.cross(d, ref)
    return _unit(p)


# polyatomic ligands: donor atom first, the rest in a local frame (d = bond direction
# from the centre to the donor, p/q = perpendicular unit vectors)
def _ligand_atoms(lig: str, d: np.ndarray, twist: float = 0.0):
    p = _perp(d)
    q = np.cross(d, p)
    p, q = math.cos(twist) * p + math.sin(twist) * q, -math.sin(twist) * p + math.cos(twist) * q
    if lig == "H2O":  # H-O-H 104.5°, O-H 0.96 Å, bisector along d (pointing away)
        a = math.radians(104.5) / 2
        return [("O", 0.0 * d), ("H", 0.96 * (math.cos(a) * d + math.sin(a) * p)),
                ("H", 0.96 * (math.cos(a) * d - math.sin(a) * p))]
    if lig == "NH3":  # N-H 1.01 Å, M-N-H 111°
        t = math.radians(180 - 111.0)
        out = [("N", 0.0 * d)]
        for k in range(3):
            ph = 2 * math.pi * k / 3
            out.append(("H", 1.01 * (math.cos(t) * d + math.sin(t) * (math.cos(ph) * p + math.sin(ph) * q))))
        return out
    if lig == "CO":
        return [("C", 0.0 * d), ("O", 1.15 * d)]
    if lig == "CN":
        return [("C", 0.0 * d), ("N", 1.16 * d)]
    if lig == "OH":  # M-O-H 115°
        t = math.radians(180 - 115.0)
        return [("O", 0.0 * d), ("H", 0.97 * (math.cos(t) * d + math.sin(t) * p))]
    return [(lig, 0.0 * d)]


def star(center: str, ligands, shape: str | None = None, r: float | None = None,
         scale: float = 1.0, angle: float | None = None):
    """A centre with ligands (`ligands` = symbol/group or list of them) on a shape.

    `r` (Å) overrides the bond length; otherwise it is `scale` × the covalent-radius sum of
    the centre and the donor atom.
    """
    if isinstance(ligands, str):
        raise TypeError("ligands must be a list")
    n = len(ligands)
    vecs = shape_vectors(shape or default_shape(n), n, angle)
    zc = z_of(center)
    syms, pos = [center], [np.zeros(3)]
    for k, (lig, d) in enumerate(zip(ligands, vecs)):
        atoms = _ligand_atoms(lig, d, twist=0.5 * k)
        rr = r if r is not None else bond(zc, z_of(atoms[0][0]), scale)
        base = rr * d
        for s, off in atoms:
            syms.append(s)
            pos.append(base + off)
    return finish(syms, np.array(pos))


def hydride(center: str, n: int):
    """EH_n with VSEPR shape from the group (lone pairs bend/pyramidalize)."""
    z = z_of(center)
    group = _group(z)
    if n == 1:
        shape = "single"
    elif n == 2:
        shape = "bent" if group == 16 else "linear"
    elif n == 3:
        shape = "pyramidal" if group == 15 else "trigonal"
    else:
        shape = "tetrahedral"
    angle = {16: 92.0 if z > 8 else 104.5, 15: 93.0 if z > 7 else 107.0}.get(group)
    return star(center, ["H"] * n, shape, angle=angle)


def _group(z: int) -> int:
    if z <= 2:
        return {1: 1, 2: 18}[z]
    if z <= 18:
        off = (z - 3) % 8
        return [1, 2, 13, 14, 15, 16, 17, 18][off]
    if z <= 36:
        return z - 18
    if z <= 54:
        return z - 36
    if 57 <= z <= 71:
        return 3
    return z - 68 if z <= 86 else 0


def atom(sym: str):
    return finish([sym], np.zeros((1, 3)))


def diatomic(a: str, b: str, r: float | None = None, scale: float = 1.0):
    rr = r if r is not None else bond(z_of(a), z_of(b), scale)
    return finish([a, b], np.array([[0, 0, 0], [0, 0, rr]], dtype=float))


def cluster(sym: str, shape: str, r: float):
    """Homonuclear cluster with nearest-neighbour distance r (Å)."""
    if shape == "tetrahedron":
        v = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=float)
        v *= r / np.linalg.norm(v[0] - v[1])
    elif shape == "tbp":
        s3 = math.sqrt(3.0)
        eq = np.array([[1.0, 0, 0], [-0.5, s3 / 2, 0], [-0.5, -s3 / 2, 0]]) * (r / s3)
        h = math.sqrt(r * r - (r / s3) ** 2)
        v = np.vstack([eq, [[0, 0, h], [0, 0, -h]]])
    elif shape == "cube":
        v = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=float) * r
        # slight distortion so the cluster is not exactly degenerate by symmetry
        v += 0.03 * r * np.array([[math.sin(1.7 * i + 0.3 * k) for k in range(3)] for i in range(len(v))])
    else:
        raise ValueError(shape)
    return finish([sym] * len(v), v)


# ---------------------------------------------------------------------------------------
# Z-matrix


def zmatrix(rows):
    """Cartesian coordinates from a Z-matrix.

    rows: [(sym,), (sym, i, r), (sym, i, r, j, angle), (sym, i, r, j, angle, k, dihedral)]
    with 0-based reference indices, Å and degrees (natural extension reference frame).
    """
    pos = []
    syms = []
    for row in rows:
        sym = row[0]
        syms.append(sym)
        if len(row) == 1:
            pos.append(np.zeros(3))
            continue
        i, r = row[1], row[2]
        if len(row) == 3:
            pos.append(pos[i] + np.array([0, 0, r]))
            continue
        j, ang = row[3], math.radians(row[4])
        if len(row) == 5:
            # place in the xz plane
            a, b = pos[i], pos[j]
            u = _unit(b - a)
            pos.append(a + r * (math.cos(ang) * u + math.sin(ang) * _perp(u)))
            continue
        k, dih = row[5], math.radians(row[6])
        a, b, c = pos[i], pos[j], pos[k]
        bc = _unit(a - b)
        n = _unit(np.cross(b - c, bc))
        m = np.cross(n, bc)
        d2 = np.array([-r * math.cos(ang), r * math.sin(ang) * math.cos(dih), r * math.sin(ang) * math.sin(dih)])
        pos.append(a + d2[0] * bc + d2[1] * m + d2[2] * n)
    return finish(syms, np.array(pos))


def finish(syms, pos_aa):
    nums = np.array([z_of(s) for s in syms], dtype=int)
    pos = np.asarray(pos_aa, dtype=float)
    pos = pos - pos.mean(axis=0)
    return nums, pos * AATOAU


# ---------------------------------------------------------------------------------------
# named small molecules (standard bond lengths and angles)

TET = 109.47


def molecule(name: str):
    m = MOLECULES[name]
    return m() if callable(m) else m


MOLECULES = {
    "H2": lambda: diatomic("H", "H", 0.74),
    "H2_stretched": lambda: diatomic("H", "H", 1.6),
    "N2_stretched": lambda: diatomic("N", "N", 1.65),
    "F2_stretched": lambda: diatomic("F", "F", 2.0),
    "O2": lambda: diatomic("O", "O", 1.21),
    "NO": lambda: diatomic("N", "O", 1.15),
    "OH": lambda: diatomic("O", "H", 0.97),
    "CN-": lambda: diatomic("C", "N", 1.17),
    "O2-": lambda: diatomic("O", "O", 1.33),
    "O2_2-": lambda: diatomic("O", "O", 1.49),
    "HeH+": lambda: diatomic("He", "H", 0.77),
    "water": lambda: star("O", ["H", "H"], "bent", r=0.96, angle=104.5),
    "H2O+": lambda: star("O", ["H", "H"], "bent", r=1.00, angle=109.0),
    "H3O+": lambda: star("O", ["H", "H", "H"], "pyramidal", r=0.98, angle=112.0),
    "OH-": lambda: diatomic("O", "H", 0.97),
    "NH3": lambda: star("N", ["H"] * 3, "pyramidal", r=1.01, angle=107.0),
    "NH3+": lambda: star("N", ["H"] * 3, "trigonal", r=1.02),
    "NH4+": lambda: star("N", ["H"] * 4, "tetrahedral", r=1.03),
    "CH4": lambda: star("C", ["H"] * 4, "tetrahedral", r=1.09),
    "CH4+": lambda: star("C", ["H"] * 4, "tetrahedral", r=1.12),
    "CH3": lambda: star("C", ["H"] * 3, "trigonal", r=1.08),
    "CH3+": lambda: star("C", ["H"] * 3, "trigonal", r=1.09),
    "CH2": lambda: star("C", ["H", "H"], "bent", r=1.08, angle=134.0),
    "CH2_singlet": lambda: star("C", ["H", "H"], "bent", r=1.11, angle=102.0),
    "NO2": lambda: star("N", ["O", "O"], "bent", r=1.20, angle=134.0),
    "NO2-": lambda: star("N", ["O", "O"], "bent", r=1.25, angle=115.0),
    "NO3-": lambda: star("N", ["O"] * 3, "trigonal", r=1.25),
    "CO2-": lambda: star("C", ["O", "O"], "bent", r=1.25, angle=134.0),
    "CO3_2-": lambda: star("C", ["O"] * 3, "trigonal", r=1.29),
    "SO4_2-": lambda: star("S", ["O"] * 4, "tetrahedral", r=1.49),
    "PO4_3-": lambda: star("P", ["O"] * 4, "tetrahedral", r=1.54),
    "ClO4-": lambda: star("Cl", ["O"] * 4, "tetrahedral", r=1.44),
    "BF4-": lambda: star("B", ["F"] * 4, "tetrahedral", r=1.40),
    "AlF6_3-": lambda: star("Al", ["F"] * 6, "octahedral", r=1.81),
    "SiF6_2-": lambda: star("Si", ["F"] * 6, "octahedral", r=1.70),
    "PF6-": lambda: star("P", ["F"] * 6, "octahedral", r=1.60),
    "MnO4-": lambda: star("Mn", ["O"] * 4, "tetrahedral", r=1.63),
    "CrO4_2-": lambda: star("Cr", ["O"] * 4, "tetrahedral", r=1.66),
    "FeCN6_3-": lambda: star("Fe", ["CN"] * 6, "octahedral", r=1.95),
    "FeCN6_4-": lambda: star("Fe", ["CN"] * 6, "octahedral", r=1.92),
    "ethylene_twisted": lambda: zmatrix([
        ("C",), ("C", 0, 1.42), ("H", 0, 1.08, 1, 121.0), ("H", 0, 1.08, 1, 121.0, 2, 180.0),
        ("H", 1, 1.08, 0, 121.0, 2, 90.0), ("H", 1, 1.08, 0, 121.0, 4, 180.0)]),
    "methanol": lambda: zmatrix([
        ("C",), ("O", 0, 1.43), ("H", 1, 0.96, 0, 108.9), ("H", 0, 1.09, 1, 109.5, 2, 180.0),
        ("H", 0, 1.09, 1, 109.5, 2, 60.0), ("H", 0, 1.09, 1, 109.5, 2, -60.0)]),
    "methylammonium": lambda: zmatrix([
        ("C",), ("N", 0, 1.50), ("H", 1, 1.03, 0, 110.0), ("H", 1, 1.03, 0, 110.0, 2, 120.0),
        ("H", 1, 1.03, 0, 110.0, 2, -120.0), ("H", 0, 1.09, 1, 108.0, 2, 180.0),
        ("H", 0, 1.09, 1, 108.0, 2, 60.0), ("H", 0, 1.09, 1, 108.0, 2, -60.0)]),
    "acetic_acid": lambda: zmatrix([
        ("C",), ("C", 0, 1.50), ("O", 1, 1.21, 0, 126.0), ("O", 1, 1.36, 0, 111.0, 2, 180.0),
        ("H", 3, 0.97, 1, 106.0, 0, 180.0), ("H", 0, 1.09, 1, 109.5, 2, 0.0),
        ("H", 0, 1.09, 1, 109.5, 2, 120.0), ("H", 0, 1.09, 1, 109.5, 2, -120.0)]),
    "acetate": lambda: zmatrix([
        ("C",), ("C", 0, 1.53), ("O", 1, 1.26, 0, 117.0), ("O", 1, 1.26, 0, 117.0, 2, 180.0),
        ("H", 0, 1.09, 1, 109.5, 2, 0.0), ("H", 0, 1.09, 1, 109.5, 2, 120.0),
        ("H", 0, 1.09, 1, 109.5, 2, -120.0)]),
    "guanidinium": lambda: zmatrix([
        ("C",), ("N", 0, 1.33), ("N", 0, 1.33, 1, 120.0), ("N", 0, 1.33, 1, 120.0, 2, 180.0),
        ("H", 1, 1.01, 0, 120.0, 2, 0.0), ("H", 1, 1.01, 0, 120.0, 2, 180.0),
        ("H", 2, 1.01, 0, 120.0, 1, 0.0), ("H", 2, 1.01, 0, 120.0, 1, 180.0),
        ("H", 3, 1.01, 0, 120.0, 1, 0.0), ("H", 3, 1.01, 0, 120.0, 1, 180.0)]),
    "oxalate": lambda: zmatrix([
        ("C",), ("C", 0, 1.56), ("O", 0, 1.26, 1, 117.0), ("O", 0, 1.26, 1, 117.0, 2, 180.0),
        ("O", 1, 1.26, 0, 117.0, 2, 90.0), ("O", 1, 1.26, 0, 117.0, 4, 180.0)]),
    "glycine_zwitterion": lambda: zmatrix([
        ("N",), ("C", 0, 1.49), ("C", 1, 1.53, 0, 111.0), ("O", 2, 1.26, 1, 117.0, 0, 0.0),
        ("O", 2, 1.26, 1, 117.0, 0, 180.0), ("H", 0, 1.03, 1, 110.0, 2, 60.0),
        ("H", 0, 1.03, 1, 110.0, 2, 180.0), ("H", 0, 1.03, 1, 110.0, 2, -60.0),
        ("H", 1, 1.09, 0, 109.0, 2, 120.0), ("H", 1, 1.09, 0, 109.0, 2, -120.0)]),
    "benzene": lambda: _ring("C", 6, 1.39, "H", 1.08),
    "benzene+": lambda: _ring("C", 6, 1.40, "H", 1.08),
    "Fe4": lambda: cluster("Fe", "tetrahedron", 2.30),
    "Cu5": lambda: cluster("Cu", "tbp", 2.40),
    "Na8": lambda: cluster("Na", "cube", 3.40),
    "Li4": lambda: cluster("Li", "tetrahedron", 2.80),
}


def _ring(sym, n, r, hsym, rh):
    R = r / (2 * math.sin(math.pi / n))
    syms, pos = [], []
    for k in range(n):
        a = 2 * math.pi * k / n
        u = np.array([math.cos(a), math.sin(a), 0.0])
        syms.append(sym)
        pos.append(R * u)
    for k in range(n):
        a = 2 * math.pi * k / n
        u = np.array([math.cos(a), math.sin(a), 0.0])
        syms.append(hsym)
        pos.append((R + rh) * u)
    return finish(syms, np.array(pos))


def aqua(metal: str, n: int = 6, scale: float = 1.18, r: float | None = None):
    """[M(H2O)n]: M-O = scale × covalent-radius sum (dative bonds are longer than the
    covalent sum; 1.18 gives 2.0-2.2 Å for first-row transition metals)."""
    shape = {4: "tetrahedral", 6: "octahedral"}[n]
    return star(metal, ["H2O"] * n, shape, r=r, scale=scale)


def ammine(metal: str, n: int = 6, r: float = 1.97):
    shape = {4: "square", 6: "octahedral"}[n]
    return star(metal, ["NH3"] * n, shape, r=r)


# ---------------------------------------------------------------------------------------
# MB16-43-style random molecules ("mindless" molecules, Korth & Grimme, JCTC 2009)


def mindless(seed: int, natoms: int = 16, elements=None, nh: tuple[int, int] = (4, 8),
             max_tries: int = 2000):
    """A random, connected molecule grown atom by atom with covalent bond lengths.

    Heavy atoms are drawn from `elements` (default: H-Cl without noble gases, as in
    MB16-43), hydrogens attach to heavy atoms only. Every new atom bonds to a random
    existing atom at 1.0-1.1 × the covalent-radius sum and must keep ≥ 0.9 × that sum from
    every other atom (H-H ≥ 1.5 Å). The geometry is not optimized: these are hard,
    unusual bonding situations by design (like the original MB16-43 starting points).
    """
    rng = np.random.default_rng(seed)
    if elements is None:
        elements = ["Li", "Be", "B", "C", "N", "O", "F", "Na", "Mg", "Al", "Si", "P", "S", "Cl"]
    n_h = int(rng.integers(nh[0], nh[1] + 1))
    heavy = [str(rng.choice(elements)) for _ in range(natoms - n_h)]
    order = heavy + ["H"] * n_h
    for _ in range(max_tries):
        syms, pos = [], []
        ok = True
        for s in order:
            if not syms:
                syms.append(s)
                pos.append(np.zeros(3))
                continue
            placed = False
            for _try in range(200):
                cands = [i for i, t in enumerate(syms) if t != "H"]
                i = int(rng.choice(cands))
                d = _unit(rng.normal(size=3))
                r = bond(z_of(syms[i]), z_of(s)) * rng.uniform(1.0, 1.1)
                p = pos[i] + r * d
                good = True
                for j, t in enumerate(syms):
                    if j == i:
                        continue
                    dist = np.linalg.norm(pos[j] - p)
                    lim = 1.5 if (s == "H" and t == "H") else 0.9 * bond(z_of(t), z_of(s))
                    if dist < lim:
                        good = False
                        break
                if good:
                    syms.append(s)
                    pos.append(p)
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok:
            return finish(syms, np.array(pos))
    raise RuntimeError("could not grow molecule")


def electrons(nums, charge: int) -> int:
    """GFN2 valence electron count (xtb's ncore table, lanthanides with 3 electrons)."""
    def nval(z):
        if 58 <= z <= 71:
            return 3
        ncore = 0
        for lim, nc in ((2, 0), (10, 2), (18, 10), (29, 18), (36, 28), (47, 36), (54, 46),
                        (71, 54), (79, 68), (86, 78), (103, 86)):
            if z <= lim:
                ncore = nc
                break
        return z - ncore
    return int(sum(nval(int(z)) for z in nums) - charge)


def read_xyz(path: Path):
    lines = path.read_text().splitlines()
    nat = int(lines[0].split()[0])
    m = re.search(r"charge\s*=\s*(-?\d+)", lines[1])
    charge = int(m.group(1)) if m else 0
    syms, pos = [], []
    for line in lines[2:2 + nat]:
        f = line.split()
        syms.append(f[0])
        pos.append([float(x) for x in f[1:4]])
    nums = np.array([z_of(s) for s in syms])
    return nums, np.array(pos) * AATOAU, charge
