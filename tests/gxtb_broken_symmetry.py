#!/usr/bin/env python3
"""Broken-symmetry unrestricted g-xTB singlets (`broken_symmetry=True`, an axtb extension;
GitHub issue #9): the validation of docs/validation.md, "Broken-symmetry singlets".

    python tests/gxtb_broken_symmetry.py [--only h2,bonds,equilibrium,fd,restarts,radical,default]
        [--out bench/results/gxtb_broken_symmetry.json] [--golden DIR --baseline BIN --new BIN]

Needs the axtb Python package built from this tree (numpy, scipy). The H2 section runs the
g-xTB 2.0.1 program for the triplet at 3 Å when `$AXTB_GXTB2` names it (tests/programs.py);
`default` runs scripts/golden.py and scripts/perf_gate.py on two CLI binaries. Every section
writes its part of the output file (the others are kept), on one thread unless stated. The
script fails when a quantity of the sections that ran exceeds its limit (`LIMITS`,
tests/gate.py).

Sections:

- **h2**: H2 from 0.6 to 5 Å (0.1 Å steps, 0.01 Å around the onset), restricted,
  broken-symmetry and triplet energies with ⟨S²⟩, each point a fresh calculation, both
  versions; where the symmetry first breaks; the triplet at 3 Å against the 2.0.1 program.
- **bonds**: the five bonds of issue #9 (CH3–OCHO, HCO–OCH3 of methyl formate, CH3–COOH of
  acetic acid, H–COOH and HCO–OH of formic acid; built from z-matrices and minimized with
  axtb g-xTB 2.0.1): rigid scans moving the leaving fragment along the bond axis by 0-4 Å
  (0.25 Å steps), restricted, broken symmetry (one calculator along the scan with restarts,
  and a fresh calculation at every point) and triplet; E(r0 + Δ) − E_singlet(r0) in kcal/mol,
  both versions (v1.1 on the 2.0.1 minima).
- **equilibrium**: closed-shell singlets near their minima (the g-xTB 2.0.1 minima of
  tests/structures/energy_differences, the golden corpus' molecules, closed-shell
  reference molecules of crates/gxtb/tests/reference_v2) and O3 and Cr2: |E_BS − E_R|, the
  largest gradient difference, the outcome and ⟨S²⟩.
- **fd**: analytic gradients against five-point differences (h = 1e-3 bohr, SCF accuracy 1e-8,
  each displacement restarted from the reference's state) of the broken-symmetry energy where
  the symmetry is broken (H2 at 2 Å, ethane with C–C at 3 Å, methyl formate with CH3–OCHO
  stretched by 2 Å), both versions; Hessians (2.0.1 analytic and response at their defaults,
  both versions numerical at the default SCF accuracy 0.01 of the displacements and at 1e-5)
  against five-point differences of the analytic gradient.
- **restarts**: SCF iterations per point along the CH3–OCHO scan with restarts (the default)
  and from fresh starts at every point; the first point's cost against a restricted single
  point for taxol.
- **radical**: CH3 + HCO2 3.5 Å apart (from methyl formate, four orientations of the methyl
  group), minimized (scipy L-BFGS-B, max |g| < 4.5e-4 Eh/bohr, geomeTRIC's default) on the
  restricted and on the broken-symmetry singlet: does the pair stay apart?
- **default**: the default results unchanged (golden corpus, bit for bit) and not slower (perf
  gate on the g-xTB cells).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import axtb

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "tests/structures"
R2 = ROOT / "crates/gxtb/tests/reference_v2"
OUT = ROOT / "bench/results/gxtb_broken_symmetry.json"
AA = 1 / 0.529177210544  # bohr per Ångström (CODATA 2022)
KCAL = 627.509474
V2, V1 = "gxtb-v2.0.1", "gxtb-v1.1"
SYMBOLS = ("X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
           "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu "
           "Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr").split()
sys.path.insert(0, str(ROOT / "tests"))


def read_xyz(path: Path):
    lines = path.read_text().splitlines()
    n = int(lines[0])
    z, x = [], []
    for line in lines[2:2 + n]:
        f = line.split()
        z.append(SYMBOLS.index(f[0].capitalize()))
        x.append([float(v) * AA for v in f[1:4]])
    return z, np.array(x)


def zmatrix(rows):
    """Cartesian positions (bohr) of a z-matrix: (symbol, [i, r(Å)], [j, angle], [k, dihedral])."""
    pos, z = [], []
    for row in rows:
        z.append(SYMBOLS.index(row[0]))
        if len(pos) == 0:
            pos.append(np.zeros(3))
            continue
        i, r = row[1]
        if len(pos) == 1:
            pos.append(pos[i] + np.array([0, 0, r]))
            continue
        j, a = row[2]
        a = np.radians(a)
        if len(pos) == 2:
            u = pos[j] - pos[i]
            u /= np.linalg.norm(u)
            perp = np.cross(u, [1.0, 0, 0]) if abs(u[0]) < 0.9 else np.cross(u, [0, 1.0, 0])
            perp /= np.linalg.norm(perp)
            pos.append(pos[i] + r * (np.cos(a) * u + np.sin(a) * perp))
            continue
        k, d = row[3]
        d = np.radians(d)
        b1, b2 = pos[j] - pos[k], pos[i] - pos[j]
        n1 = np.cross(b1, b2)
        n1 /= np.linalg.norm(n1)
        b2u = b2 / np.linalg.norm(b2)
        m = np.cross(n1, b2u)
        v = np.array([-r * np.cos(a), r * np.sin(a) * np.cos(d), r * np.sin(a) * np.sin(d)])
        pos.append(pos[i] + v[0] * b2u + v[1] * m + v[2] * n1)
    return z, np.array(pos) * AA


# minimized with axtb g-xTB 2.0.1 in `bonds` (z-matrices: standard conformers, the methyl cis
# to C=O in methyl formate, the O-H syn to C=O in the acids)
MOLECULES = {
    "methyl formate": [("C",), ("O", [0, 1.20]), ("O", [0, 1.34], [1, 125.0]),
                       ("H", [0, 1.10], [1, 125.0], [2, 180.0]), ("C", [2, 1.44], [0, 115.0], [1, 0.0]),
                       ("H", [4, 1.09], [2, 109.5], [0, 180.0]), ("H", [4, 1.09], [2, 109.5], [0, 60.0]),
                       ("H", [4, 1.09], [2, 109.5], [0, -60.0])],
    "acetic acid": [("C",), ("O", [0, 1.21]), ("O", [0, 1.35], [1, 123.0]), ("H", [2, 0.97], [0, 107.0], [1, 0.0]),
                    ("C", [0, 1.50], [1, 125.0], [2, 180.0]), ("H", [4, 1.09], [0, 109.5], [1, 0.0]),
                    ("H", [4, 1.09], [0, 109.5], [1, 120.0]), ("H", [4, 1.09], [0, 109.5], [1, -120.0])],
    "formic acid": [("C",), ("O", [0, 1.20]), ("O", [0, 1.34], [1, 125.0]), ("H", [2, 0.97], [0, 107.0], [1, 0.0]),
                    ("H", [0, 1.10], [1, 124.0], [2, 180.0])],
}
# bond: molecule, (atom staying, atom leaving), the leaving fragment's atoms, issue #9's columns
# (DFT UKS, g-xTB triplet, g-xTB restricted singlet; kcal/mol at r0 + 4 Å)
BONDS = {
    "CH3–OCHO": ("methyl formate", (2, 4), [4, 5, 6, 7], (105.8, 107.2, 155.5)),
    "HCO–OCH3": ("methyl formate", (2, 0), [0, 1, 3], (106.7, 113.3, 161.5)),
    "CH3–COOH": ("acetic acid", (0, 4), [4, 5, 6, 7], (111.9, 119.6, 174.7)),
    "H–COOH": ("formic acid", (0, 4), [4], (107.6, 116.0, 180.6)),
    "HCO–OH": ("formic acid", (2, 0), [0, 1, 4], (115.6, 123.9, 171.8)),
}


def minimize(calc, x, gtol=1e-5, maxiter=2000):
    """L-BFGS-B on the calculator's surface (Cartesian, bohr); positions, energy, the last result."""
    from scipy.optimize import minimize as smin

    last = {}

    def f(v):
        r = calc.singlepoint(v.reshape(-1, 3), gradient=True)
        last["r"] = r
        return r["energy"], np.asarray(r["gradient"]).reshape(-1)

    res = smin(f, x.reshape(-1), jac=True, method="L-BFGS-B",
               options={"maxiter": maxiter, "gtol": gtol, "ftol": 1e-15, "maxcor": 30})
    xf = res.x.reshape(-1, 3)
    r = calc.singlepoint(xf, gradient=True)
    return xf, r, res.nit


def equilibrium_structures():
    """Minimized methyl formate, acetic acid and formic acid (cached in the output)."""
    out = {}
    for name, zm in MOLECULES.items():
        z, x = zmatrix(zm)
        calc = axtb.Calculator(V2, z, x, accuracy=0.01)
        x, r, nit = minimize(calc, x, gtol=1e-6)
        out[name] = {"numbers": z, "positions": x.tolist(), "energy": r["energy"],
                     "max_gradient": float(np.abs(r["gradient"]).max()), "iterations": nit}
    return out


def section_h2(results):
    out = {}
    grid = sorted(set([round(0.6 + 0.1 * k, 2) for k in range(45)] + [round(1.30 + 0.01 * k, 2) for k in range(21)]
                      + [0.74]))
    for method in (V2, V1):
        rows = []
        for d in grid:
            x = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, d * AA]])
            r = axtb.singlepoint([1, 1], x, method=method)
            b = axtb.singlepoint([1, 1], x, method=method, broken_symmetry=True)
            t = axtb.singlepoint([1, 1], x, method=method, uhf=2)
            bs = b["broken_symmetry"]
            rows.append({"r": d, "restricted": r["energy"], "broken_symmetry": b["energy"], "triplet": t["energy"],
                         "outcome": bs["outcome"], "s2": bs["s2"], "unrestricted": bs["unrestricted_energy"],
                         "unrestricted_s2": bs["unrestricted_s2"]})
        first = next((r["r"] for r in rows if r["outcome"] == "broken"), None)
        at3 = next(r for r in rows if r["r"] == 3.0)
        out[method] = {"points": rows, "first_broken": first,
                       "bs_minus_triplet_3A_kcal": (at3["broken_symmetry"] - at3["triplet"]) * KCAL,
                       "restricted_minus_triplet_3A_kcal": (at3["restricted"] - at3["triplet"]) * KCAL}
    # the triplet at 3 Å against the 2.0.1 program
    x = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0 * AA]])
    prog = {"program_singlet": -0.898613968259, "program_triplet": -0.991205431595, "source": "issue #9 / task"}
    try:
        import programs

        exe = programs.get(programs.GXTB2)
    except Exception:  # noqa: BLE001
        exe = None
    if exe:
        with tempfile.TemporaryDirectory() as d:
            Path(d, "h2.xyz").write_text("2\n\nH 0 0 0\nH 0 0 3.0\n")
            env = {**os.environ, "OMP_NUM_THREADS": "1"}
            for key, extra in (("program_singlet", []), ("program_triplet", ["--uhf", "2"])):
                p = subprocess.run([exe, "h2.xyz", "--gxtb", "--norestart", *extra], cwd=d, capture_output=True,
                                   text=True, env=env)
                line = next(l for l in p.stdout.splitlines() if "TOTAL ENERGY" in l)
                prog[key] = float(line.split()[3])
            prog["source"] = "g-xTB 2.0.1 program (xtb --gxtb --norestart, OMP_NUM_THREADS=1)"
    mctc = 1.8897261246204404  # the program reads XYZ with mctc-lib's factor
    xp = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0 * mctc]])
    prog["axtb_triplet_gxtb_conventions"] = axtb.singlepoint([1, 1], xp, method=V2, uhf=2, conventions="gxtb")["energy"]
    prog["axtb_singlet_gxtb_conventions"] = axtb.singlepoint([1, 1], xp, method=V2, conventions="gxtb")["energy"]
    prog["axtb_triplet_axtb_conventions"] = axtb.singlepoint([1, 1], x, method=V2, uhf=2)["energy"]
    out["program_check_3A"] = prog
    results["h2"] = out


def attempt(f):
    """The result of `f`, or {"error": message} for an SCF that does not converge."""
    try:
        return f()
    except axtb.ScfNotConvergedError as e:
        return {"error": f"scf not converged ({e.iterations} iterations)"}


def bond_frame(x, stay, leave, frag):
    """Positions with the fragment moved by Δ (Å) along the stay→leave bond axis."""
    u = x[leave] - x[stay]
    r0 = np.linalg.norm(u) / AA
    u = u / np.linalg.norm(u)

    def at(delta):
        y = x.copy()
        y[frag] += delta * AA * u
        return y

    return r0, at


def section_bonds(results):
    eq = results.get("structures") or equilibrium_structures()
    results["structures"] = eq
    deltas = [0.25 * k for k in range(17)]
    out = {}
    for method in (V2, V1):
        out[method] = {}
        for bond, (mol, (stay, leave), frag, issue) in BONDS.items():
            z, x = eq[mol]["numbers"], np.array(eq[mol]["positions"])
            r0, at = bond_frame(x, stay, leave, frag)
            e0 = axtb.singlepoint(z, x, method=method)["energy"]
            bs = axtb.Calculator(method, z, x, broken_symmetry=True)
            rows = []
            for d in deltas:
                y = at(d)
                t0 = time.perf_counter()
                b = attempt(lambda: bs.singlepoint(y))
                dt = time.perf_counter() - t0
                rr = attempt(lambda: axtb.singlepoint(z, y, method=method))
                bf = attempt(lambda: axtb.singlepoint(z, y, method=method, broken_symmetry=True))
                t = attempt(lambda: axtb.singlepoint(z, y, method=method, uhf=2))
                rel = lambda r: (r["energy"] - e0) * KCAL if "energy" in r else None  # noqa: E731
                bsi = lambda r, k: r["broken_symmetry"][k] if "broken_symmetry" in r else r.get("error")  # noqa: E731
                rows.append({"delta": d, "restricted": rel(rr), "broken_symmetry": rel(b),
                             "broken_symmetry_fresh": rel(bf), "triplet": rel(t), "outcome": bsi(b, "outcome"),
                             "s2": bsi(b, "s2"), "start": bsi(b, "start"), "fresh_outcome": bsi(bf, "outcome"),
                             "fresh_s2": bsi(bf, "s2"), "fresh_start_etemp": bsi(bf, "start_etemp"),
                             "iterations": [bsi(b, "restricted_iterations"), bsi(b, "unrestricted_iterations")],
                             "seconds": dt})
            last = rows[-1]
            step = lambda a, b, k: None if a[k] is None or b[k] is None else a[k] - b[k]  # noqa: E731
            out[method][bond] = {
                "molecule": mol, "r0": r0, "points": rows,
                "at_4A": {"dft_uks": issue[0], "issue_triplet": issue[1], "issue_restricted": issue[2],
                          "broken_symmetry": last["broken_symmetry"],
                          "broken_symmetry_fresh": last["broken_symmetry_fresh"], "triplet": last["triplet"],
                          "restricted": last["restricted"]},
                "last_step_kcal": {k: step(rows[-1], rows[-2], k)
                                   for k in ("broken_symmetry", "triplet", "restricted")},
                "from_3A_to_4A_kcal": {k: step(rows[-1], rows[-5], k) for k in
                                       ("broken_symmetry", "triplet", "restricted")}}
    results["bonds"] = out


def ref_v2(rel):
    d = R2 / rel
    inp = json.loads((d / "input.json").read_text())
    z, x = read_xyz(d / "mol.xyz")
    return z, x, inp["charge"], inp["uhf"]


def equilibrium_set():
    """(name, numbers, positions, charge) of closed-shell singlets near their minima."""
    out = []
    ed = S / "energy_differences/v2.0.1"
    for f in sorted(ed.glob("*.xyz")):
        if "-irc-" in f.stem:
            continue
        head = f.read_text().splitlines()[1]
        charge = float(head.split("charge")[1].split(",")[0])
        z, x = read_xyz(f)
        out.append((f"{f.stem} (2.0.1 {'TS' if f.stem.endswith('-ts') else 'minimum'})", z, x, charge))
    # (mb33 of the golden corpus has an odd electron count)
    for name, charge in [("water", 0), ("caffeine", 0), ("taxol", 0), ("ferrocene", 0), ("beni-acetone", 0),
                         ("cf3br-benzene", 0), ("tyr-xab", 0), ("s229-cl", 0), ("zncl4", -2), ("anthracene2", 0)]:
        z, x = read_xyz(S / f"{name}.xyz")
        out.append((name, z, x, charge))
    z, x = read_xyz(S / "large/c50h102.xyz")
    out.append(("C50H102", z, x, 0))
    for rel in ["reference/acetate", "reference/ammonium", "reference/benzene", "reference/ethanol",
                "reference/glycine", "reference/hydroxide", "reference/iodomethane", "reference/mesyl_chloride",
                "reference/methane", "elements/Cr_CO_6", "elements/Mo_CO_6", "elements/OsO4", "elements/RuO4",
                "elements/UF6", "elements/WF6", "elements/TiCl4", "elements/PdCl4_2-_", "elements/PtCl4_2-_",
                "elements/BF3", "elements/XeF2", "elements/HgCl2"]:
        z, x, charge, uhf = ref_v2(rel)
        if uhf == 0:
            out.append((rel.split("/")[1], z, x, charge))
    return out


def section_equilibrium(results):
    rows = []
    cases = [(n, z, x, q, "set") for n, z, x, q in equilibrium_set()]
    # known broken-symmetry cases: O3 (experimental geometry 1.272 Å, 116.8°) and Cr2 (1.68 Å)
    a = np.radians(116.8 / 2)
    o3 = np.array([[0, 0, 0], [0, 1.272 * np.sin(a), 1.272 * np.cos(a)], [0, -1.272 * np.sin(a), 1.272 * np.cos(a)]])
    cases += [("O3", [8, 8, 8], o3 * AA, 0, "known"), ("Cr2", [24, 24], np.array([[0, 0, 0], [0, 0, 1.68]]) * AA, 0,
                                                       "known")]
    # far from equilibrium: gh106 of the golden corpus (every bond stretched 1.89 times)
    z, x = read_xyz(S / "gh106.xyz")
    cases.append(("gh106 (stretched)", z, x, 0, "stretched"))
    for name, z, x, q, kind in cases:
        for method in (V2, V1):
            if method == V1 and kind == "set" and len(z) > 60:
                continue  # v1.1 for the small and medium molecules only
            try:
                t0 = time.perf_counter()
                r = axtb.singlepoint(z, x, method=method, charge=q, gradient=True)
                t1 = time.perf_counter()
                b = axtb.singlepoint(z, x, method=method, charge=q, gradient=True, broken_symmetry=True)
                t2 = time.perf_counter()
            except Exception as e:  # noqa: BLE001
                rows.append({"name": name, "method": method, "error": str(e)})
                continue
            info = b["broken_symmetry"]
            rows.append({"name": name, "method": method, "kind": kind, "atoms": len(z), "charge": q,
                         "outcome": info["outcome"], "s2": info["s2"], "pairs": info["frontier_pairs"],
                         "de": b["energy"] - r["energy"],
                         "dg": float(np.abs(np.asarray(b["gradient"]) - np.asarray(r["gradient"])).max()),
                         "unrestricted_minus_restricted": (info["unrestricted_energy"] - info["restricted_energy"])
                         if info["unrestricted_energy"] is not None else None,
                         "unrestricted_s2": info["unrestricted_s2"],
                         "iterations": [info["restricted_iterations"], info["unrestricted_iterations"]],
                         "seconds_restricted": t1 - t0, "seconds_broken_symmetry": t2 - t1})
    sel = [r for r in rows if r.get("kind") == "set" and "error" not in r]
    results["equilibrium"] = {
        "rows": rows,
        "molecules": len({r["name"] for r in sel}),
        "max_abs_de": max(abs(r["de"]) for r in sel),
        "max_abs_dg": max(r["dg"] for r in sel),
        "broken": [(r["name"], r["method"], r["s2"], r["de"] * KCAL) for r in rows
                   if "error" not in r and r["outcome"] == "broken"],
    }


def five_point(f, h):
    return (f(-2 * h) - 8 * f(-h) + 8 * f(h) - f(2 * h)) / (12 * h)


def fd_points():
    eq = equilibrium_structures()
    mf = eq["methyl formate"]
    z, x = mf["numbers"], np.array(mf["positions"])
    _, at = bond_frame(x, 2, 4, [4, 5, 6, 7])
    ethane = [("C",), ("C", [0, 3.0]), ("H", [0, 1.09], [1, 111.0]), ("H", [0, 1.09], [1, 111.0], [2, 120.0]),
              ("H", [0, 1.09], [1, 111.0], [2, -120.0]), ("H", [1, 1.09], [0, 111.0], [2, 180.0]),
              ("H", [1, 1.09], [0, 111.0], [2, 60.0]), ("H", [1, 1.09], [0, 111.0], [2, -60.0])]
    ze, xe = zmatrix(ethane)
    return [("H2 at 2 Å", [1, 1], np.array([[0, 0, 0], [0.1, 0.2, 2.0 * AA]])), ("ethane, C–C 3 Å", ze, xe),
            ("methyl formate, CH3–OCHO +2 Å", z, at(2.0))]


def section_fd(results):
    rows = []
    h = 1e-3
    for name, z, x in fd_points():
        for method in (V2, V1):
            calc = axtb.Calculator(method, z, x, broken_symmetry=True, accuracy=1e-8)
            r0 = calc.singlepoint(x, gradient=True)
            info = r0["broken_symmetry"]
            g = np.asarray(r0["gradient"]).reshape(-1)
            n = g.size
            fd = np.zeros(n)
            hfd = np.zeros((n, n))
            for j in range(n):
                def e(s, j=j):
                    y = x.copy().reshape(-1)
                    y[j] += s
                    return calc.singlepoint(y.reshape(-1, 3))["energy"]

                def gr(s, j=j):
                    y = x.copy().reshape(-1)
                    y[j] += s
                    return np.asarray(calc.singlepoint(y.reshape(-1, 3), gradient=True)["gradient"]).reshape(-1)

                fd[j] = five_point(e, h)
                hfd[:, j] = five_point(gr, h)
            hfd = 0.5 * (hfd + hfd.T)
            row = {"case": name, "method": method, "outcome": info["outcome"], "s2": info["s2"],
                   "max_gradient": float(np.abs(g).max()), "gradient_vs_fd": float(np.abs(g - fd).max()),
                   "hessians": {}}
            # the numerical Hessian at its default SCF accuracy of the displacements (0.01) and at 1e-5
            kinds = [("analytic", None), ("response", None)] if method == V2 else []
            for hm, acc in kinds + [("numerical", None), ("numerical", 1e-5)]:
                c = axtb.Calculator(method, z, x, broken_symmetry=True)
                hh = c.hessian(x, method=hm, accuracy=acc, vibrations=False)
                key = hm if acc is None else f"{hm}, accuracy {acc:g}"
                row["hessians"][key] = {"vs_fd": float(np.abs(np.asarray(hh["hessian"]) - hfd).max()),
                                        "energy_vs_reference": hh["energy"] - r0["energy"]}
            row["max_hessian"] = float(np.abs(hfd).max())
            rows.append(row)
            print(row, flush=True)
    results["fd"] = rows


def section_restarts(results):
    eq = results.get("structures") or equilibrium_structures()
    results["structures"] = eq
    z, x = eq["methyl formate"]["numbers"], np.array(eq["methyl formate"]["positions"])
    _, at = bond_frame(x, 2, 4, [4, 5, 6, 7])
    deltas = [0.25 * k for k in range(17)]
    scan = {}
    for label, guess in (("restarts", None), ("fresh", "fresh")):
        calc = axtb.Calculator(V2, z, x, broken_symmetry=True, guess=guess)
        rows = []
        for d in deltas:
            t0 = time.perf_counter()
            r = calc.singlepoint(at(d), gradient=True)
            dt = time.perf_counter() - t0
            b = r["broken_symmetry"]
            rows.append({"delta": d, "energy": r["energy"], "outcome": b["outcome"], "start": b["start"],
                         "restricted_iterations": b["restricted_iterations"],
                         "unrestricted_iterations": b["unrestricted_iterations"],
                         "total_iterations": (b["restricted_iterations"] or 0) + (b["unrestricted_iterations"] or 0),
                         "seconds": dt})
        scan[label] = rows
    plain = axtb.Calculator(V2, z, x)
    scan["restricted_restarts"] = [{"delta": d, "iterations": plain.singlepoint(at(d))["iterations"]} for d in deltas]
    # taxol: first point and a second (displaced) point, broken symmetry vs restricted
    zt, xt = read_xyz(S / "taxol.xyz")
    rng = np.random.default_rng(1)
    xt2 = xt + rng.normal(scale=0.01, size=xt.shape)
    cost = {}
    for threads in (1, 4):
        times = {"restricted": [], "broken_symmetry": [], "restricted_step2": [], "broken_symmetry_step2": []}
        info = None
        for _ in range(3):
            for label, kw in (("restricted", {}), ("broken_symmetry", {"broken_symmetry": True})):
                calc = axtb.Calculator(V2, zt, xt, threads=threads, **kw)
                t0 = time.perf_counter()
                r = calc.singlepoint(xt, gradient=True)
                t1 = time.perf_counter()
                r2 = calc.singlepoint(xt2, gradient=True)
                t2 = time.perf_counter()
                times[label].append(t1 - t0)
                times[label + "_step2"].append(t2 - t1)
                if label == "broken_symmetry":
                    info = {"first": r["broken_symmetry"], "second": r2["broken_symmetry"]}
                else:
                    info_r = {"first": r["iterations"], "second": r2["iterations"]}
        cost[f"{threads}t"] = {k: statistics.median(v) for k, v in times.items()}
        cost[f"{threads}t"]["broken_symmetry_info"] = info
        cost[f"{threads}t"]["restricted_iterations"] = info_r
        cost[f"{threads}t"]["first_point_ratio"] = cost[f"{threads}t"]["broken_symmetry"] / cost[f"{threads}t"][
            "restricted"]
        cost[f"{threads}t"]["second_point_ratio"] = cost[f"{threads}t"]["broken_symmetry_step2"] / cost[
            f"{threads}t"]["restricted_step2"]
    results["restarts"] = {"ch3_ocho_scan": scan, "taxol": cost}


def section_radical(results):
    eq = results.get("structures") or equilibrium_structures()
    results["structures"] = eq
    z, x = eq["methyl formate"]["numbers"], np.array(eq["methyl formate"]["positions"])
    r0, at = bond_frame(x, 2, 4, [4, 5, 6, 7])
    rows = []
    for k, angle in enumerate((0.0, 60.0, 120.0, 180.0)):
        # the methyl group pulled to 3.5 Å and rotated about the old bond axis' normal
        y = at(3.5 - r0)
        c = y[4].copy()
        axis = np.cross(y[4] - y[2], [0.0, 0.0, 1.0])
        axis /= np.linalg.norm(axis)
        th = np.radians(angle)
        rot = (np.cos(th) * np.eye(3) + np.sin(th) * np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]],
                                                              [-axis[1], axis[0], 0]])
               + (1 - np.cos(th)) * np.outer(axis, axis))
        for i in (5, 6, 7):
            y[i] = c + rot @ (y[i] - c)
        for label, kw in (("restricted", {}), ("broken_symmetry", {"broken_symmetry": True})):
            calc = axtb.Calculator(V2, z, y, **kw)
            try:
                xf, r, nit = minimize(calc, y, gtol=4.5e-4, maxiter=500)
            except axtb.ScfNotConvergedError as e:
                rows.append({"orientation": angle, "surface": label, "error": f"scf not converged: {e}"})
                print(rows[-1], flush=True)
                continue
            d = np.linalg.norm(xf[4] - xf[2]) / AA
            dmin = min(np.linalg.norm(xf[4] - xf[i]) / AA for i in (0, 1, 2, 3))
            rows.append({"orientation": angle, "surface": label, "co_distance": float(d),
                         "closest_c_heavy": float(dmin),
                         "energy": r["energy"], "optimizer_iterations": nit,
                         "max_gradient": float(np.abs(r["gradient"]).max()),
                         "outcome": r.get("broken_symmetry", {}).get("outcome"),
                         "s2": r.get("broken_symmetry", {}).get("s2"),
                         "radical_pair": bool(d > 2.5)})
            print(rows[-1], flush=True)
    results["radical"] = rows


def section_default(results, golden, baseline, new):
    out = {}
    if golden and new:
        p = subprocess.run([sys.executable, str(ROOT / "scripts/golden.py"), "compare", golden, "--axtb", new],
                           capture_output=True, text=True)
        out["golden"] = p.stdout.strip().splitlines()[-1]
        out["golden_diffs"] = [l for l in p.stdout.splitlines() if l.startswith("DIFF")]
    if baseline and new:
        p = subprocess.run([sys.executable, str(ROOT / "scripts/perf_gate.py"), baseline, new, "--only", "gxtb"],
                           capture_output=True, text=True)
        out["perf_gate"] = p.stdout.strip().splitlines()
    results["default"] = out


# the numbers docs/validation.md and the README quote (tests/gate.py); kcal/mol, Eh, Eh/bohr, Eh/bohr²
LIMITS = {
    "h2 2.0.1: |E_BS - E_triplet| at 3 Å (kcal/mol)": 1.7,
    "h2 v1.1: |E_BS - E_triplet| at 3 Å (kcal/mol)": 1.2,
    "h2: |E_BS - E_triplet| at 5 Å, both versions (kcal/mol)": 0.01,
    "h2: axtb triplet at 3 Å (gxtb conventions) vs the 2.0.1 program (Eh)": 1e-11,
    "bonds: |E_BS - E_triplet| at r0 + 4 Å (kcal/mol)": 0.02,
    "bonds: E_BS(r0 + 4 Å) - E_BS(r0 + 3 Å) (kcal/mol)": 0.3,
    "bonds 2.0.1: |restarts - fresh| beyond r0 + 1.5 Å (kcal/mol)": 0.001,
    "bonds: points beyond r0 + 1.5 Å not broken": 0,
    "equilibrium: |E_BS - E_R| (Eh)": 1e-12,
    "equilibrium: |g_BS - g_R| (Eh/bohr)": 1e-12,
    "equilibrium: molecules whose symmetry breaks": 0,
    "fd 2.0.1: gradient vs finite differences (Eh/bohr)": 3e-10,
    "fd v1.1: gradient vs finite differences (Eh/bohr)": 2e-8,
    "fd 2.0.1: analytic Hessian vs finite differences (Eh/bohr²)": 2e-9,
    "fd 2.0.1: response Hessian vs finite differences (Eh/bohr²)": 5e-8,
    "fd: numerical Hessian (accuracy 1e-5) vs finite differences (Eh/bohr²)": 1e-4,
}
KNOWN_DIFFERENCES = {
    "bonds v1.1, restarts vs fresh": "v1.1's convergence test (the program's open-shell thresholds) stops the "
                                     "unrestricted SCF early on these flat surfaces: up to 0.016 kcal/mol, and at "
                                     "HCO–OCH3 + 3 Å a fresh start ends 0.83 kcal/mol high after 14 iterations "
                                     "(accuracy 0.01: the restarts' energy after 922)",
    "fd v1.1 gradient": "v1.1's own restricted and triplet gradients differ from five-point differences by the same "
                        "2e-8 Eh/bohr for methyl formate (5.7e-8 at its minimum): not a broken-symmetry term",
}


def summary(results, sections):
    """The gated quantities of the sections that ran."""
    out = {}
    if "h2" in sections:
        h = results["h2"]
        out["h2 2.0.1: |E_BS - E_triplet| at 3 Å (kcal/mol)"] = abs(h[V2]["bs_minus_triplet_3A_kcal"])
        out["h2 v1.1: |E_BS - E_triplet| at 3 Å (kcal/mol)"] = abs(h[V1]["bs_minus_triplet_3A_kcal"])
        out["h2: |E_BS - E_triplet| at 5 Å, both versions (kcal/mol)"] = max(
            abs(p["broken_symmetry"] - p["triplet"]) * KCAL for m in (V2, V1) for p in h[m]["points"] if p["r"] == 5.0)
        prog = h["program_check_3A"]
        out["h2: axtb triplet at 3 Å (gxtb conventions) vs the 2.0.1 program (Eh)"] = abs(
            prog["axtb_triplet_gxtb_conventions"] - prog["program_triplet"])
    if "bonds" in sections:
        bonds = [o for m in results["bonds"].values() for o in m.values()]
        out["bonds: |E_BS - E_triplet| at r0 + 4 Å (kcal/mol)"] = max(
            abs(o["at_4A"]["broken_symmetry"] - o["at_4A"]["triplet"]) for o in bonds)
        out["bonds: E_BS(r0 + 4 Å) - E_BS(r0 + 3 Å) (kcal/mol)"] = max(
            o["from_3A_to_4A_kcal"]["broken_symmetry"] for o in bonds)
        far = [(o, p) for o in bonds for p in o["points"] if p["delta"] > 1.5]
        out["bonds 2.0.1: |restarts - fresh| beyond r0 + 1.5 Å (kcal/mol)"] = max(
            abs(p["broken_symmetry"] - p["broken_symmetry_fresh"]) for o in results["bonds"][V2].values()
            for p in o["points"] if p["delta"] > 1.5)
        out["bonds: points beyond r0 + 1.5 Å not broken"] = [
            f"{o['molecule']} +{p['delta']}" for o, p in far
            if p["outcome"] != "broken" or p["fresh_outcome"] != "broken"]
    if "equilibrium" in sections:
        e = results["equilibrium"]
        out["equilibrium: |E_BS - E_R| (Eh)"] = e["max_abs_de"]
        out["equilibrium: |g_BS - g_R| (Eh/bohr)"] = e["max_abs_dg"]
        out["equilibrium: molecules whose symmetry breaks"] = [
            f"{r['name']} {r['method']}" for r in e["rows"] if r.get("kind") == "set" and r.get("outcome") == "broken"]
    if "fd" in sections:
        fd = results["fd"]
        out["fd 2.0.1: gradient vs finite differences (Eh/bohr)"] = max(
            r["gradient_vs_fd"] for r in fd if r["method"] == V2)
        out["fd v1.1: gradient vs finite differences (Eh/bohr)"] = max(
            r["gradient_vs_fd"] for r in fd if r["method"] == V1)
        for hm in ("analytic", "response"):
            out[f"fd 2.0.1: {hm} Hessian vs finite differences (Eh/bohr²)"] = max(
                r["hessians"][hm]["vs_fd"] for r in fd if r["method"] == V2)
        out["fd: numerical Hessian (accuracy 1e-5) vs finite differences (Eh/bohr²)"] = max(
            r["hessians"]["numerical, accuracy 1e-05"]["vs_fd"] for r in fd)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="h2,bonds,equilibrium,fd,restarts,radical")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--golden")
    ap.add_argument("--baseline")
    ap.add_argument("--new")
    a = ap.parse_args()
    out = Path(a.out)
    results = json.loads(out.read_text()) if out.exists() else {}
    info = Path("/proc/cpuinfo")
    lines = info.read_text().splitlines() if info.exists() else []
    cpu = next((l.split(":", 1)[1].strip() for l in lines if l.startswith("model name")), platform.machine())
    results["machine"] = {"cpu": cpu, "axtb": axtb.__version__}
    for sec in a.only.split(","):
        t0 = time.perf_counter()
        if sec == "default":
            section_default(results, a.golden, a.baseline, a.new)
        else:
            globals()[f"section_{sec}"](results)
        print(f"{sec}: {time.perf_counter() - t0:.1f} s", flush=True)
        out.write_text(json.dumps(results, indent=1) + "\n")
    import gate

    return gate.check(summary(results, a.only.split(",")), LIMITS, KNOWN_DIFFERENCES)


if __name__ == "__main__":
    sys.exit(main())
