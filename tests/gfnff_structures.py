"""Public test set of the GFN-FF port (structures in bohr, with total charges).

Sources (all public or generated):

* `tests/structures/*.xyz`: organics, aromatics, heterocycles (caffeine, taxol, tyrosine
  complex), non-covalent complexes (anthracene dimer, gh106), halogen-bonded CF3Br-benzene,
  [Cu(NH3)4]2+, ferrocene, [ZnCl4]2-;
* `tests/reference_matrix` builders: small molecules of every element H-Rn, random
  MB16-43-style molecules (light and heavy elements), charged species (-3..+3), aqua and
  ammine complexes, open-shell metal complexes;
* `tests/structures/large`: water clusters (H2O)100/200/333, n-C50H102;
* crambin (PDB 1CRN) with hydrogens from Open Babel 3.1.0 (`obabel 1crn.pdb -h -p 7.0`),
  total charge +2 (Arg+ x2, neutral termini and acids as protonated by Open Babel),
  `tests/structures/large/crambin_h.xyz`;
* hydrogen- and halogen-bonded dimers built here (water dimer, formic-acid dimer,
  NH3...HF, pyridine...HF, acetone...HOH, Cl2/Br2/ICl...NH3, CH3I...pyridine-like N,
  CF3Cl...O=CH2) and a glycine zwitterion in a water shell.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests" / "reference_matrix"))
sys.path.insert(0, str(ROOT / "tests"))

import structures as S  # noqa: E402
import cases as C  # noqa: E402
from gfnff_xtb import read_xyz  # noqa: E402

AA = S.AATOAU


def _mol(syms, xyz_aa):
    """`syms`: element symbols, or a string of them ("CClFFF")."""
    if isinstance(syms, str):
        syms = re.findall(r"[A-Z][a-z]?", syms)
    # not centred: the pieces of a complex keep their relative placement
    return np.array([S.z_of(s) for s in syms]), np.asarray(xyz_aa, float) * AA


def _combine(*parts, shift=None):
    nums = np.concatenate([p[0] for p in parts])
    pos = np.concatenate([p[1] + (shift[k] * AA if shift is not None else 0) for k, p in enumerate(parts)])
    return nums, pos


def _place(mol, origin_aa, axis=None):
    nums, pos = mol
    pos = pos - pos.mean(axis=0)
    return nums, pos + np.asarray(origin_aa) * AA


def dimers():
    """Hydrogen- and halogen-bonded dimers (Angstrom geometries, standard bond lengths)."""
    out = []
    w1 = _mol("OHH", [[0, 0, 0], [0.9572, 0, 0], [-0.24, 0.927, 0]])
    w2 = _mol("OHH", [[2.91, 0, 0], [3.23, 0.0, 0.90], [3.23, 0.0, -0.90]])
    out.append(("dimer-water", _combine(w1, w2), 0))
    fa1 = _mol("CHOOH", [[0, 0, 0], [0, 1.09, 0], [1.10, -0.60, 0], [-1.15, -0.62, 0], [-1.85, 0.05, 0]])
    fa2 = _mol("CHOOH", [[-2.67, -2.10, 0], [-2.67, -3.19, 0], [-3.77, -1.50, 0], [-1.52, -1.48, 0], [-0.82, -2.15, 0]])
    out.append(("dimer-formicacid", _combine(fa1, fa2), 0))
    nh3hf = _mol("NHHHFH", [[0, 0, 0], [-0.38, 0.94, 0.0], [-0.38, -0.47, 0.81], [-0.38, -0.47, -0.81],
                           [2.66, 0, 0], [1.74, 0, 0]])
    out.append(("dimer-NH3-HF", nh3hf, 0))
    # pyridine ... HF (N is the acceptor, eg2_rnr case)
    ring = [[1.39 * np.cos(np.pi / 3 * k), 1.39 * np.sin(np.pi / 3 * k), 0.0] for k in range(6)]
    hs = [[2.47 * np.cos(np.pi / 3 * k), 2.47 * np.sin(np.pi / 3 * k), 0.0] for k in range(1, 6)]
    pyr = _mol("NCCCCC" + "HHHHH", ring + hs)
    hf = _mol("HF", [[1.39 + 1.75, 0, 0], [1.39 + 1.75 + 0.92, 0, 0]])
    out.append(("dimer-pyridine-HF", _combine(pyr, hf), 0))
    # acetone ... HOH (carbonyl case eg3)
    acetone = _mol("COCCHHHHHH", [[0, 0, 0], [0, 1.21, 0], [1.29, -0.78, 0], [-1.29, -0.78, 0],
                                  [1.29, -1.42, 0.89], [1.29, -1.42, -0.89], [2.14, -0.10, 0],
                                  [-1.29, -1.42, 0.89], [-1.29, -1.42, -0.89], [-2.14, -0.10, 0]])
    wat = _mol("HOH", [[0, 3.15, 0], [0, 4.10, 0], [0.93, 4.35, 0]])
    out.append(("dimer-acetone-water", _combine(acetone, wat), 0))
    # halogen bonds
    for name, x1, x2, r12, rxb in (("Cl2", "Cl", "Cl", 1.99, 2.73), ("Br2", "Br", "Br", 2.28, 2.80),
                                     ("ICl", "I", "Cl", 2.32, 2.70)):
        d = _mol(x2 + x1 + "NHHH", [[-r12, 0, 0], [0, 0, 0], [rxb, 0, 0], [rxb + 0.38, 0.94, 0.0],
                                     [rxb + 0.38, -0.47, 0.81], [rxb + 0.38, -0.47, -0.81]])
        out.append((f"xb-{name}-NH3", d, 0))
    ch3i = _mol("CIHHH", [[0, 0, 0], [2.14, 0, 0], [-0.36, 1.03, 0], [-0.36, -0.51, 0.89], [-0.36, -0.51, -0.89]])
    nh3 = _mol("NHHH", [[2.14 + 2.95, 0, 0], [5.47, 0.94, 0], [5.47, -0.47, 0.81], [5.47, -0.47, -0.81]])
    out.append(("xb-CH3I-NH3", _combine(ch3i, nh3), 0))
    cf3cl = _mol("CClFFF", [[0, 0, 0], [1.75, 0, 0], [-0.45, 1.27, 0], [-0.45, -0.63, 1.10], [-0.45, -0.63, -1.10]])
    form = _mol("OCHH", [[1.75 + 3.0, 0, 0], [1.75 + 4.21, 0, 0], [6.55, 0.94, 0], [6.55, -0.94, 0]])
    out.append(("xb-CF3Cl-formaldehyde", _combine(cf3cl, form), 0))
    return out


def water_cluster(n: int, seed: int = 0, spacing: float = 3.1):
    """(H2O)n on a jittered cubic grid, each molecule randomly oriented."""
    rng = np.random.default_rng(seed)
    k = int(np.ceil(n ** (1 / 3)))
    nums, pos = [], []
    count = 0
    for ix in range(k):
        for iy in range(k):
            for iz in range(k):
                if count == n:
                    break
                c = np.array([ix, iy, iz]) * spacing + rng.normal(0, 0.15, 3)
                q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
                w = np.array([[0, 0, 0], [0.9572, 0, 0], [-0.2399, 0.9266, 0]]) @ q.T
                for z, p in zip((8, 1, 1), w + c):
                    nums.append(z)
                    pos.append(p)
                count += 1
    pos = np.array(pos)
    return np.array(nums), (pos - pos.mean(axis=0)) * AA


def water_box(n: int, seed: int = 0):
    """(H2O)n at liquid density (3.1 A grid), (numbers, positions in bohr)."""
    return water_cluster(n, seed=seed, spacing=3.1)


def solvated_crambin(pad: float = 5.0, seed: int = 0, clash: float = 2.4):
    """Crambin (`crambin_h.xyz`, charge +2) in a box of water: grid waters (3.1 A spacing,
    random orientations) filling the protein's bounding box plus `pad` A, without those
    closer than `clash` A to a protein atom. Returns (numbers, positions in bohr, charge)."""
    pn, pp, _ = read_xyz(ROOT / "tests" / "structures" / "large" / "crambin_h.xyz")
    pp = pp / AA
    lo, hi = pp.min(axis=0) - pad, pp.max(axis=0) + pad
    k = np.ceil((hi - lo) / 3.1).astype(int)
    rng = np.random.default_rng(seed)
    nums, pos = list(pn), list(pp)
    for ix in range(k[0]):
        for iy in range(k[1]):
            for iz in range(k[2]):
                c = lo + np.array([ix, iy, iz]) * 3.1 + rng.normal(0, 0.15, 3)
                q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
                w = np.array([[0, 0, 0], [0.9572, 0, 0], [-0.2399, 0.9266, 0]]) @ q.T + c
                if np.min(np.linalg.norm(pp[:, None, :] - w[None, :, :], axis=2)) > clash:
                    nums += [8, 1, 1]
                    pos += list(w)
    return np.array(nums), np.array(pos) * AA, 2


def glycine_in_water(nw: int = 20, seed: int = 1):
    gly = S.molecule("glycine_zwitterion")
    wn, wp = water_cluster(nw, seed=seed, spacing=3.1)
    # place the waters around the solute, dropping those that clash
    keep_n, keep_p = list(gly[0]), list(gly[1])
    for m in range(nw):
        idx = slice(3 * m, 3 * m + 3)
        p = wp[idx] + np.array([0.0, 0.0, 0.0])
        d = np.min(np.linalg.norm(np.asarray(keep_p)[:, None, :] - p[None, :, :], axis=2))
        if d > 2.4 * AA:
            keep_n += list(wn[idx])
            keep_p += list(p)
    return np.array(keep_n), np.array(keep_p)


def cases(include_large: bool = True):
    """List of (id, category, numbers, positions_bohr, charge)."""
    out = []
    for c in C.structures():
        out.append((c["id"], "structures", np.array(c["numbers"]), np.array(c["positions"]), c["charge"]))
    for cat in ("elements", "charges", "multielement", "open_shell_metals"):
        for c in C.CATEGORIES[cat]():
            if len(c["numbers"]) < 2 and cat != "charges":
                continue
            out.append((c["id"], cat, np.array(c["numbers"]), np.array(c["positions"]), c["charge"]))
    for name, mol, q in dimers():
        out.append((name, "hb-xb-dimers", mol[0], mol[1], q))
    out.append(("glycine-zwitterion-water", "zwitterions", *glycine_in_water(), 0))
    out.append(("water-cluster-20", "water", *water_cluster(20, seed=3), 0))
    if include_large:
        large = ROOT / "tests" / "structures" / "large"
        for name in ("water100", "water200", "c50h102", "crambin_h"):
            path = large / f"{name}.xyz"
            if path.exists():
                nums, pos, q = read_xyz(path)
                if name == "crambin_h":
                    q = 2
                out.append((name, "large", nums, pos, q))
    return out
