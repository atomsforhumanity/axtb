#!/usr/bin/env python3
"""Energy differences of g-xTB under the program's and axtb's conventions (v1.1 and 2.0.1).

For each g-xTB version, energy differences of public, generated structures are evaluated in
both conventions (`conventions="gxtb"`, which reproduces the program, and the default
`"axtb"`), so that the effect of the corrected defaults on chemistry is visible:

- conformers: n-butane gauche - anti, ethanol OH gauche - anti (minima of each version),
  ferrocene staggered - eclipsed (D5d, the ring-rotation saddle point, and the D5h minimum);
- reactions with transition states: the SN2 identity reaction Cl- + CH3Cl -> ClCH3 + Cl-
  (ion-dipole complex, TS, separated reactants; the TS from the guess of bench/ts_example.py)
  and the isomerization HCN -> HNC (reaction energy and barrier);
- binding energies at fixed geometries: the water dimer (optimized), iodobenzene...acetone and
  ethylammonium perchlorate (tests/structures/beni-acetone.xyz, s229-cl.xyz), with fragments
  as cut from the complex;
- unrelaxed ligand dissociation: Ni(CO)4 -> Ni(CO)3 + CO (the reference geometry of
  crates/gxtb/tests/reference) and [Cu(NH3)4]2+ -> [Cu(NH3)3]2+ + NH3 (tests/structures).

Geometries (bohr) are generated here and optimized with axtb in its default conventions
through qccompute (`--optimize`): minima with geomeTRIC (GAU_VERYTIGHT), transition states
with Sella (order 1, fmax 0.001 eV/A, started from axtb's Hessian), each version on its own
surface; every TS has exactly one imaginary mode below -20 cm-1 and connects the intended
minima (pseudo-IRC: +-0.3 bohr along the mode, then minimization). The optimized structures
are stored in tests/structures/energy_differences/<version>/ (no structure comes from a
research project), and the evaluation (the default) runs the axtb CLI on them:

    python tests/gxtb_energy_differences.py [--optimize] [--out bench/results/gxtb_energy_differences.json]

`--optimize` needs qccompute, qccodec[geometric,sella] and the axtb Python package (with its
qccodec codec) in the interpreter. The script fails (exit status 1) when the default conventions
change an energy difference by more than `LIMITS` allows (tests/gate.py).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests/reference_matrix"))
import structures as S  # noqa: E402

sys.path.insert(0, str(ROOT / "tests"))
import gate  # noqa: E402

AXTB = os.environ.get("AXTB_BIN", str(ROOT / "target/release/axtb"))
GEOM = ROOT / "tests/structures/energy_differences"
VERSIONS = {"v1.1": "gxtb-v1.1", "v2.0.1": "gxtb-v2.0.1"}
# |default - program conventions| of the energy differences: <= 1.1e-5 Eh for v1.1, whose
# default conventions diagonalize another Fock matrix, and <= 4e-11 Eh for 2.0.1 (README, g-xTB;
# docs/gxtb-v1.1-spec.md §4.18, docs/gxtb-v2.0.1-spec.md §12)
LIMITS = {"v1.1 max_abs_change_Eh": 1.1e-5, "v2.0.1 max_abs_change_Eh": 4e-11}
HARTREE_KCAL = 627.5094740631
BOHR_AA = 1.0 / S.AATOAU

# ---------------------------------------------------------------------------------------
# generated start structures: name -> (numbers, positions in bohr, charge, uhf, kind)


def butane(dih):
    rows = [("C",), ("C", 0, 1.53), ("C", 1, 1.53, 0, 112.0), ("C", 2, 1.53, 1, 112.0, 0, dih)]
    rows += [("H", 0, 1.09, 1, 110.5, 2, d) for d in (180.0, 60.0, -60.0)]
    rows += [("H", 1, 1.09, 2, 109.5, 0, d) for d in (121.5, -121.5)]
    rows += [("H", 2, 1.09, 1, 109.5, 3, d) for d in (121.5, -121.5)]
    rows += [("H", 3, 1.09, 2, 110.5, 1, d) for d in (180.0, 60.0, -60.0)]
    return S.zmatrix(rows)


def ethanol(dih):
    rows = [("C",), ("C", 0, 1.52), ("O", 1, 1.43, 0, 108.0), ("H", 2, 0.96, 1, 108.5, 0, dih)]
    rows += [("H", 0, 1.09, 1, 110.0, 2, d) for d in (180.0, 60.0, -60.0)]
    rows += [("H", 1, 1.09, 0, 110.0, 2, d) for d in (120.0, -120.0)]
    return S.zmatrix(rows)


def ferrocene(staggered):
    syms, pos = ["Fe"], [np.zeros(3)]
    for z, off in ((1.66, 0.0), (-1.66, 36.0 if staggered else 0.0)):
        for k in range(5):
            a = math.radians(72.0 * k + off)
            syms.append("C")
            pos.append(np.array([1.208 * math.cos(a), 1.208 * math.sin(a), z]))
        for k in range(5):
            a = math.radians(72.0 * k + off)
            syms.append("H")
            pos.append(np.array([2.285 * math.cos(a), 2.285 * math.sin(a), z]))
    return S.finish(syms, np.array(pos))


def water_dimer():
    a = math.radians(104.5)
    syms = ["O", "H", "H", "O", "H", "H"]
    pos = [[0, 0, 0], [0.96, 0, 0], [0.96 * math.cos(a), 0, 0.96 * math.sin(a)],
           [2.91, 0, 0], [3.25, 0.76, -0.47], [3.25, -0.76, -0.47]]
    return S.finish(syms, np.array(pos, float))


def sn2_ts_guess():
    """The asymmetric guess of bench/ts_example.py (C-Cl 2.25 / 2.45 A)."""
    guess = [("C", [0.0, 0.0, 0.0]), ("Cl", [0.0, 0.0, 2.25]), ("Cl", [0.05, 0.0, -2.45]),
             ("H", [1.07, 0.0, 0.08]), ("H", [-0.535, 0.927, 0.08]), ("H", [-0.535, -0.927, 0.08])]
    return S.finish([s for s, _ in guess], np.array([p for _, p in guess]))


def chloromethane():
    t = math.radians(180.0 - 108.5)
    pos = [[0, 0, 0], [0, 0, 1.79]] + [[1.09 * math.sin(t) * math.cos(p), 1.09 * math.sin(t) * math.sin(p),
                                        -1.09 * math.cos(t)] for p in (0.0, 2.0944, 4.1888)]
    return S.finish(["C", "Cl", "H", "H", "H"], np.array(pos))


def hcn(iso):
    if iso == "HCN":
        return S.finish(["C", "N", "H"], np.array([[0, 0, 0], [1.16, 0, 0], [-1.07, 0, 0]]))
    if iso == "HNC":
        return S.finish(["C", "N", "H"], np.array([[0, 0, 0], [1.17, 0, 0], [2.17, 0, 0]]))
    return S.finish(["C", "N", "H"], np.array([[0, 0, 0], [1.19, 0, 0], [0.55, 1.05, 0]]))  # TS guess


def read_xyz(path: Path):
    lines = path.read_text().splitlines()
    nat = int(lines[0])
    syms = [l.split()[0] for l in lines[2:2 + nat]]
    pos = np.array([[float(x) for x in l.split()[1:4]] for l in lines[2:2 + nat]])
    return np.array([S.z_of(s) for s in syms]), pos * S.AATOAU


# minima and transition states optimized per version
OPTIMIZED = {
    "butane-anti": (lambda: butane(180.0), 0, 0, "min"),
    "butane-gauche": (lambda: butane(65.0), 0, 0, "min"),
    "ethanol-anti": (lambda: ethanol(180.0), 0, 0, "min"),
    "ethanol-gauche": (lambda: ethanol(60.0), 0, 0, "min"),
    "ferrocene-eclipsed": (lambda: ferrocene(False), 0, 0, "min"),
    # D5d is the saddle point of the ring rotation (one imaginary torsion), kept by symmetry
    "ferrocene-staggered": (lambda: ferrocene(True), 0, 0, "symmetric"),
    "water-dimer": (water_dimer, 0, 0, "min"),
    "water": (lambda: S.molecule("water"), 0, 0, "min"),
    "chloromethane": (chloromethane, 0, 0, "min"),
    "chloride": (lambda: S.atom("Cl"), -1, 0, "atom"),
    "sn2-ts": (sn2_ts_guess, -1, 0, "ts"),
    "HCN": (lambda: hcn("HCN"), 0, 0, "min"),
    "HNC": (lambda: hcn("HNC"), 0, 0, "min"),
    "hcn-ts": (lambda: hcn("TS"), 0, 0, "ts"),
}
# the TS's pseudo-IRC ends (minimized) are stored as <ts>-irc-0/1; the SN2 complex is one of them
TS_ENDS = {"sn2-ts": ("sn2-complex", "sn2-complex"), "hcn-ts": ("HCN", "HNC")}


# ---------------------------------------------------------------------------------------
# optimization (qccompute)


def write_xyz(path: Path, nums, pos_bohr, comment: str):
    lines = [str(len(nums)), comment]
    for z, p in zip(nums, pos_bohr):
        lines.append(f"{S.SYMBOLS[int(z) - 1]:2s} " + " ".join(f"{x * BOHR_AA:16.10f}" for x in p))
    path.write_text("\n".join(lines) + "\n")


def optimize_all(version: str):
    from qccompute import compute
    from qcdata import CalcType, Model, ProgramInput, ProgramSpec, Structure

    method = VERSIONS[version]
    model = Model(method=method)
    grad = ProgramSpec(program="axtb", calctype=CalcType.gradient, model=model, keywords={"accuracy": 0.01})
    hess = ProgramSpec(program="axtb", calctype=CalcType.hessian, model=model, keywords={"accuracy": 0.01})
    out_dir = GEOM / version
    out_dir.mkdir(parents=True, exist_ok=True)

    def structure(nums, pos, charge, uhf):
        return Structure(symbols=[S.SYMBOLS[int(z) - 1] for z in nums], geometry=np.asarray(pos).tolist(),
                         charge=charge, multiplicity=uhf + 1)

    def minimize(st):
        out = compute(ProgramInput(program="geometric", calctype=CalcType.optimization, structure=st,
                                   subprograms=[grad], keywords={"convergence_set": "gau_verytight", "maxiter": 500,
                                                                 "coordsys": "tric"}), raise_exc=False)
        if not out.success:
            raise RuntimeError(out.traceback[-2000:] if out.traceback else "geomeTRIC failed")
        return out.results.final_structure

    def frequencies(st):
        out = compute(ProgramInput(program="axtb", calctype=CalcType.hessian, structure=st, model=model,
                                   keywords={"accuracy": 0.01}), raise_exc=True)
        return np.array(out.results.freqs_wavenumber), np.array(out.results.normal_modes_cartesian), out.results.energy

    def energy(st):
        return compute(ProgramInput(program="axtb", calctype=CalcType.energy, structure=st, model=model,
                                    keywords={"accuracy": 1e-4}), raise_exc=True).results.energy

    log = {}
    for name, (build, charge, uhf, kind) in OPTIMIZED.items():
        nums, pos = build()
        st = structure(nums, pos, charge, uhf)
        if kind == "atom":
            final = st
            info = {}
        elif kind in ("min", "symmetric"):
            final = minimize(st)
            fr, _, _ = frequencies(final)
            info = {"lowest_frequencies_cm": fr[:3].round(2).tolist(), "imaginary_below_-20": int((fr < -20).sum())}
            if info["imaginary_below_-20"] > (1 if kind == "symmetric" else 0):
                raise RuntimeError(f"{version} {name}: not a minimum {fr[:3]}")
        else:
            out = compute(ProgramInput(program="sella", calctype=CalcType.transition_state, structure=st,
                                       subprograms=[grad, hess], keywords={"fmax": 0.001, "steps": 1000}),
                          raise_exc=False)
            if not out.success:
                raise RuntimeError(f"{version} {name}: Sella failed\n{(out.traceback or '')[-2000:]}")
            final = out.results.final_structure
            fr, modes, _ = frequencies(final)
            nimag = int((fr < -20).sum())
            if nimag != 1:
                raise RuntimeError(f"{version} {name}: {nimag} imaginary modes {fr[:3]}")
            # pseudo-IRC: +-0.3 bohr on the most displaced atom along the imaginary mode
            mode = modes[0].reshape(-1, 3)
            step = 0.3 * mode / np.linalg.norm(mode, axis=1).max()
            ends = []
            for k, sign in enumerate((1.0, -1.0)):
                g = np.array(final.geometry) + sign * step
                e = minimize(structure(nums, g, charge, uhf))
                efr, _, _ = frequencies(e)
                if (efr < -20).sum():
                    raise RuntimeError(f"{version} {name}: pseudo-IRC end {k} not a minimum {efr[:3]}")
                ends.append(e)
                write_xyz(out_dir / f"{name}-irc-{k}.xyz", nums, np.array(e.geometry),
                          f"{name} pseudo-IRC end {k}, {method} (axtb default conventions), E = {energy(e):.10f} Eh")
            info = {"imaginary_cm": round(float(fr[0]), 2), "next_frequencies_cm": fr[1:3].round(2).tolist()}
            # connectivity: which minima the two ends are (by distances), and their energies
            ends_e = [energy(e) for e in ends]
            if name == "sn2-ts":
                bonded = []
                for e in ends:  # Cl...C distances: one bond (< 2.1 A), one ion-dipole contact (> 2.8 A)
                    d = np.linalg.norm(np.array(e.geometry)[[1, 2]] - np.array(e.geometry)[0], axis=1) * BOHR_AA
                    if not (d.min() < 2.1 and d.max() > 2.8):
                        raise RuntimeError(f"{version} {name}: pseudo-IRC end is not the ion-dipole complex: {d}")
                    bonded.append(int(np.argmin(d)))
                if bonded[0] == bonded[1]:
                    raise RuntimeError(f"{version} {name}: both pseudo-IRC ends bind the same Cl")
                k = int(np.argmin(ends_e))
                write_xyz(out_dir / "sn2-complex.xyz", nums, np.array(ends[k].geometry),
                          f"sn2-complex (pseudo-IRC end of sn2-ts), {method} (axtb default conventions), "
                          f"E = {ends_e[k]:.10f} Eh, charge -1, uhf 0")
                log["sn2-complex"] = {"energy": ends_e[k], "irc_end_energies": ends_e}
                info["connects"] = "ion-dipole complex on both sides (identity reaction)"
            else:
                found = []
                for e, ee in zip(ends, ends_e):
                    g = np.array(e.geometry)
                    iso = "HCN" if np.linalg.norm(g[2] - g[0]) < np.linalg.norm(g[2] - g[1]) else "HNC"
                    if abs(ee - log[iso]["energy"]) > 1e-6:
                        raise RuntimeError(f"{version} {name}: pseudo-IRC end {iso} at {ee}, minimum {log[iso]['energy']}")
                    found.append(iso)
                if sorted(found) != ["HCN", "HNC"]:
                    raise RuntimeError(f"{version} {name}: pseudo-IRC ends {found}")
                info["connects"] = "HCN and HNC (minimized ends within 1e-6 Eh of the minima)"
            info["irc_end_energies"] = ends_e
        e = energy(final)
        write_xyz(out_dir / f"{name}.xyz", nums, np.array(final.geometry),
                  f"{name}, {method} (axtb default conventions), E = {e:.10f} Eh, charge {charge}, uhf {uhf}")
        log[name] = {"energy": e, **info}
        print(version, name, json.dumps(log[name]), flush=True)
    (out_dir / "optimization.json").write_text(json.dumps(log, indent=1))


# ---------------------------------------------------------------------------------------
# evaluation (axtb CLI, both conventions)


def cli_energy(method, conv, nums, pos, charge, uhf=0) -> float:
    req = {"numbers": [int(z) for z in nums], "positions": np.asarray(pos).tolist(), "charge": charge, "uhf": uhf,
           "method": method, "conventions": conv, "accuracy": 1e-4}
    p = subprocess.run([AXTB, "--json"], input=json.dumps(req), capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError((p.stderr + p.stdout).strip()[-300:])
    return json.loads(p.stdout)["energy"]


def stored(version, name):
    """A stored structure: numbers, positions (bohr), charge, uhf."""
    nums, pos = read_xyz(GEOM / version / f"{name}.xyz")
    if name in OPTIMIZED:
        _, charge, uhf, _ = OPTIMIZED[name]
    elif name == "sn2-complex":
        charge, uhf = -1, 0
    else:
        charge, uhf = 0, 0
    return nums, pos, charge, uhf


def quantities(version):
    """(label, [(coefficient, numbers, positions, charge, uhf)]) of every energy difference."""
    def s(name):
        return stored(version, name)

    q = [
        ("n-butane gauche - anti", [(1, *s("butane-gauche")), (-1, *s("butane-anti"))]),
        ("ethanol OH gauche - anti", [(1, *s("ethanol-gauche")), (-1, *s("ethanol-anti"))]),
        ("ferrocene staggered - eclipsed (ring rotation)", [(1, *s("ferrocene-staggered")), (-1, *s("ferrocene-eclipsed"))]),
        ("Cl- + CH3Cl -> [Cl...CH3Cl]-, complexation", [(1, *s("sn2-complex")), (-1, *s("chloromethane")),
                                                         (-1, *s("chloride"))]),
        ("Cl- + CH3Cl SN2 barrier from the complex", [(1, *s("sn2-ts")), (-1, *s("sn2-complex"))]),
        ("Cl- + CH3Cl SN2 TS - separated reactants", [(1, *s("sn2-ts")), (-1, *s("chloromethane")), (-1, *s("chloride"))]),
        ("HCN -> HNC, reaction energy", [(1, *s("HNC")), (-1, *s("HCN"))]),
        ("HCN -> HNC, barrier", [(1, *s("hcn-ts")), (-1, *s("HCN"))]),
        ("water dimer binding", [(1, *s("water-dimer")), (-1, *s("water")), (-1, *s("water"))]),
    ]
    # fixed public complexes, fragments cut from them (unrelaxed)
    for label, path, cut, charges in (
            ("iodobenzene...acetone binding", ROOT / "tests/structures/beni-acetone.xyz", 12, (0, 0)),
            ("ethylammonium perchlorate ion-pair binding", ROOT / "tests/structures/s229-cl.xyz", 11, (1, -1))):
        nums, pos = read_xyz(path)
        q.append((label, [(1, nums, pos, 0, 0), (-1, nums[:cut], pos[:cut], charges[0], 0),
                          (-1, nums[cut:], pos[cut:], charges[1], 0)]))
    nums, pos = read_xyz(ROOT / "crates/gxtb/tests/reference/nickel_tetracarbonyl/mol.xyz")
    # Ni(CO)4: Ni, then C/O pairs; take off the first CO (the C and O nearest Ni first)
    c = [i for i in range(len(nums)) if nums[i] == 6]
    o = [i for i in range(len(nums)) if nums[i] == 8]
    c0 = c[0]
    o0 = min(o, key=lambda i: np.linalg.norm(pos[i] - pos[c0]))
    rest = [i for i in range(len(nums)) if i not in (c0, o0)]
    q.append(("Ni(CO)4 -> Ni(CO)3 + CO (unrelaxed)", [(-1, nums, pos, 0, 0), (1, nums[rest], pos[rest], 0, 0),
                                                       (1, nums[[c0, o0]], pos[[c0, o0]], 0, 0)]))
    nums, pos = read_xyz(ROOT / "tests/structures/cu-nh3-4.xyz")
    n0 = [i for i in range(len(nums)) if nums[i] == 7][0]
    nh3 = [n0] + sorted((i for i in range(len(nums)) if nums[i] == 1),
                        key=lambda i: np.linalg.norm(pos[i] - pos[n0]))[:3]
    rest = [i for i in range(len(nums)) if i not in nh3]
    q.append(("[Cu(NH3)4]2+ -> [Cu(NH3)3]2+ + NH3 (unrelaxed)", [(-1, nums, pos, 2, 1), (1, nums[rest], pos[rest], 2, 1),
                                                                 (1, nums[nh3], pos[nh3], 0, 0)]))
    return q


def evaluate(version):
    method = VERSIONS[version]
    cache = {}
    rows = []
    for label, parts in quantities(version):
        res = {}
        for conv in ("gxtb", "axtb"):
            total = 0.0
            for coef, nums, pos, charge, uhf in parts:
                key = (conv, tuple(nums), np.asarray(pos).round(10).tobytes(), charge, uhf)
                if key not in cache:
                    cache[key] = cli_energy(method, conv, nums, pos, charge, uhf)
                total += coef * cache[key]
            res[conv] = total
        rows.append({"quantity": label, "program_conventions_kcal": res["gxtb"] * HARTREE_KCAL,
                     "default_minus_program_Eh": res["axtb"] - res["gxtb"]})
        print(f"{version} | {label} | {rows[-1]['program_conventions_kcal']:.4f} | {rows[-1]['default_minus_program_Eh']:+.1e} |",
              flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimize", action="store_true")
    ap.add_argument("--versions", default="v1.1,v2.0.1")
    ap.add_argument("--out", type=Path, default=ROOT / "bench/results/gxtb_energy_differences.json")
    a = ap.parse_args()
    versions = a.versions.split(",")
    if a.optimize:
        for v in versions:
            optimize_all(v)
    result = {"axtb_cli": "accuracy 1e-4 (SCF thresholds x 1e-4) in both conventions",
              "geometries": "tests/structures/energy_differences/<version> (optimized with axtb, default conventions)",
              "versions": {}}
    for v in versions:
        rows = evaluate(v)
        opt = GEOM / v / "optimization.json"
        result["versions"][v] = {"method": VERSIONS[v], "rows": rows,
                                 "max_abs_change_Eh": max(abs(r["default_minus_program_Eh"]) for r in rows),
                                 "optimization": json.loads(opt.read_text()) if opt.exists() else None}
    a.out.write_text(json.dumps(result, indent=1))
    print("wrote", a.out)
    return gate.check({f"{v} max_abs_change_Eh": r["max_abs_change_Eh"] for v, r in result["versions"].items()}, LIMITS)


if __name__ == "__main__":
    raise SystemExit(main())
