#!/usr/bin/env python3
"""Golden outputs for refactoring: run a fixed corpus of CLI requests (every method, convention
and feature, a few thread counts and the run-time LAPACK path) and compare outputs bit for bit.

    scripts/golden.py record DIR [--axtb BIN]     run the corpus, write DIR/<case>.json
    scripts/golden.py compare DIR [--axtb BIN]    run it again and compare with DIR
    scripts/golden.py list                        the cases

Outputs are compared exactly (every float, every field) except the timing fields. Golden files
are machine-specific (the matrix-product kernels depend on the CPU), so keep them local.

Trajectory cases (warm starts: each SCF from the previous geometry's wavefunction) run the
dev-only example of axtb-driver (`cargo build --release -p axtb-driver --example
golden_trajectory`) on one calculator: `--trajectory BIN`, by default `BIN-trajectory` next to
the axtb binary if it exists, else `examples/golden_trajectory` in its directory.
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "tests/structures"
R2 = ROOT / "crates/gxtb/tests/reference_v2"
BOHR = 0.529177210544
SYMBOLS = ("X H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr "
           "Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu "
           "Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn").split()
TIMING = {"timings", "hessian_timings", "seconds"}
# a run-time LAPACK for the eigensolver path through the library (optional)
MKL = os.environ.get("AXTB_GOLDEN_LAPACK", "/usr/lib/x86_64-linux-gnu/libmkl_rt.so")


def mol(path: Path, charge=0.0, uhf=None):
    lines = path.read_text().split("\n")
    n = int(lines[0])
    atoms = [l.split() for l in lines[2:2 + n]]
    req = {"numbers": [SYMBOLS.index(a[0].capitalize()) for a in atoms],
           "positions": [[float(x) / BOHR for x in a[1:4]] for a in atoms], "charge": charge}
    if uhf is not None:
        req["uhf"] = uhf
    return req


WATER, CAFF, TAXOL = S / "water.xyz", S / "caffeine.xyz", S / "taxol.xyz"
W100, W333, C50 = S / "large/water100.xyz", S / "large/water333.xyz", S / "large/c50h102.xyz"
O2, CU, FC = R2 / "reference/dioxygen_triplet/mol.xyz", S / "cu-nh3-4.xyz", S / "ferrocene.xyz"
BENI, CF3BR, TYR, IL, GH106, MB33, ZNCL4 = (S / f for f in ("beni-acetone.xyz", "cf3br-benzene.xyz", "tyr-xab.xyz",
                                         "s229-cl.xyz", "gh106.xyz", "mb33.xyz", "zncl4.xyz"))
# hard SCFs (tests/scf_hard_set.py): stretched H2O and F2 (both bonds / the bond at 3 or 3.5 Å, singlets)
# and a seeded random molecule (a doublet)
H2O3, F23, F235, MBH112 = (ROOT / "tests/scf_hard" / f for f in ("scan-H2O-3.0.xyz", "scan-F2-3.0.xyz",
                                                                   "scan-F2-3.5.xyz", "random-mbheavy-112.xyz"))


def corpus():
    """(name, request, extra environment)"""
    c = []
    add = lambda name, base, env=None, **kw: c.append((name, {**base, **kw}, env or {}))
    w, caf, tax = mol(WATER), mol(CAFF), mol(TAXOL)
    o2, cu = mol(O2, 0.0, 2), mol(CU, 2.0, 1)
    # GFN2-xTB
    add("gfn2-water-energy", w, method="gfn2")
    for conv in ("axtb", "xtb", "xtb671", "tblite"):
        add(f"gfn2-caffeine-grad-{conv}", caf, method="gfn2", conventions=conv, gradient=True)
    add("gfn2-taxol-grad-t1", tax, method="gfn2", gradient=True, threads=1)
    add("gfn2-taxol-grad-t4", tax, method="gfn2", gradient=True, threads=4)
    add("gfn2-water100-grad-t1", mol(W100), method="gfn2", gradient=True, threads=1)
    add("gfn2-water100-grad-t4", mol(W100), method="gfn2", gradient=True, threads=4)
    add("gfn2-water100-grad-t4-lapack", mol(W100), {"AXTB_LAPACK": MKL}, method="gfn2", gradient=True, threads=4)
    add("gfn2-taxol-grad-t1-lapack", tax, {"AXTB_LAPACK": MKL}, method="gfn2", gradient=True, threads=1)
    add("gfn2-o2-triplet-grad", o2, method="gfn2", gradient=True)
    add("gfn2-cu-doublet-grad", cu, method="gfn2", gradient=True)
    add("gfn2-caffeine-alpb", caf, method="gfn2", gradient=True, alpb="water")
    add("gfn2-caffeine-alpb-bar1m", caf, method="gfn2", gradient=True, alpb="water", solvation_state="bar1m")
    add("gfn2-caffeine-gbsa", caf, method="gfn2", gradient=True, gbsa="water")
    add("gfn2-caffeine-etemp", caf, method="gfn2", gradient=True, etemp=3000.0)
    add("gfn2-caffeine-nodisp", caf, method="gfn2", gradient=True, dispersion=False)
    for mixer in ("broyden", "diis"):
        add(f"gfn2-caffeine-mixer-{mixer}", caf, method="gfn2", gradient=True, mixer=mixer)
    add("gfn2-caffeine-hessian", caf, method="gfn2", hessian=True)
    add("gfn2-caffeine-hessian-t4", caf, method="gfn2", hessian=True, threads=4)
    add("gfn2-caffeine-hessian-xtb", caf, method="gfn2", hessian=True, conventions="xtb")
    add("gfn2-water-hessian-numerical", w, method="gfn2", hessian=True, hessian_method="numerical")
    add("gfn2-caffeine-hessian-alpb", caf, method="gfn2", hessian=True, alpb="water")
    add("gfn2-caffeine-hessian-gbsa", caf, method="gfn2", hessian=True, gbsa="water")
    add("gfn2-caffeine-hessian-alpb-xtb671", caf, method="gfn2", hessian=True, alpb="water", conventions="xtb671")
    add("gfn2-caffeine-hessian-tblite", caf, method="gfn2", hessian=True, conventions="tblite")
    add("gfn2-taxol-hessian-t4", tax, method="gfn2", hessian=True, threads=4)
    add("gfn2-gh106-grad", mol(GH106), method="gfn2", gradient=True)
    add("gfn2-zncl4-dianion-grad", mol(ZNCL4, -2.0), method="gfn2", gradient=True)
    add("gfn2-beni-acetone-grad", mol(BENI), method="gfn2", gradient=True)
    add("gfn2-water-guess-sad", w, method="gfn2", guess="sad")
    add("gfn2-water-molden", w, method="gfn2", molden=True)
    add("gfn2-caffeine-bond-orders", caf, method="gfn2", bond_orders=True)
    # g-xTB 2.0.1
    add("gxtb2-water-energy", w, method="gxtb-v2.0.1")
    for conv in ("axtb", "gxtb"):
        add(f"gxtb2-caffeine-grad-{conv}", caf, method="gxtb-v2.0.1", conventions=conv, gradient=True)
    add("gxtb2-cu-doublet-grad", cu, method="gxtb-v2.0.1", gradient=True)
    add("gxtb2-o2-triplet-grad", o2, method="gxtb-v2.0.1", gradient=True)
    add("gxtb2-caffeine-etemp", caf, method="gxtb-v2.0.1", gradient=True, etemp=3000.0)
    add("gxtb2-taxol-grad-t1", tax, method="gxtb-v2.0.1", gradient=True, threads=1)
    add("gxtb2-taxol-grad-t4", tax, method="gxtb-v2.0.1", gradient=True, threads=4)
    add("gxtb2-c50h102-grad-t4-lapack", mol(C50), {"AXTB_LAPACK": MKL}, method="gxtb-v2.0.1", gradient=True, threads=4)
    add("gxtb2-caffeine-hessian", caf, method="gxtb-v2.0.1", hessian=True)
    add("gxtb2-caffeine-hessian-t4", caf, method="gxtb-v2.0.1", hessian=True, threads=4)
    add("gxtb2-cu-hessian", cu, method="gxtb-v2.0.1", hessian=True)
    add("gxtb2-water-hessian-response", w, method="gxtb-v2.0.1", hessian=True, hessian_method="response")
    add("gxtb2-water-hessian-numerical", w, method="gxtb-v2.0.1", hessian=True, hessian_method="numerical")
    add("gxtb2-caffeine-composite", caf, method="gxtb-v2.0.1", gradient=True, alpb="water", solvation_correction="gfn2")
    add("gxtb2-water-composite-hessian", w, method="gxtb-v2.0.1", hessian=True, alpb="water", solvation_correction="gfn2")
    add("gxtb2-water-molden", w, method="gxtb-v2.0.1", molden=True)
    add("gxtb2-caffeine-bond-orders", caf, method="gxtb-v2.0.1", bond_orders=True)
    add("gxtb2-mb33-grad", mol(MB33), method="gxtb-v2.0.1", gradient=True)
    add("gxtb2-beni-acetone-hessian", mol(BENI), method="gxtb-v2.0.1", hessian=True)
    # g-xTB v1.1
    for conv in ("axtb", "gxtb"):
        add(f"gxtb11-caffeine-grad-{conv}", caf, method="gxtb-v1.1", conventions=conv, gradient=True)
    add("gxtb11-cu-doublet-grad", cu, method="gxtb-v1.1", gradient=True)
    add("gxtb11-water-hessian", w, method="gxtb-v1.1", hessian=True)
    add("gxtb11-water-hessian-gxtb", w, method="gxtb-v1.1", hessian=True, conventions="gxtb")
    add("gxtb11-caffeine-etemp", caf, method="gxtb-v1.1", gradient=True, etemp=3000.0)
    # GFN-FF
    for conv in ("axtb", "xtb", "xtb671"):
        add(f"gfnff-caffeine-grad-{conv}", caf, method="gfnff", conventions=conv, gradient=True)
    add("gfnff-taxol-grad", tax, method="gfnff", gradient=True)
    add("gfnff-water333-grad-t4", mol(W333), method="gfnff", gradient=True, threads=4)
    add("gfnff-caffeine-hessian", caf, method="gfnff", hessian=True)
    add("gfnff-ferrocene-grad", mol(FC), method="gfnff", gradient=True)
    # GFN-FF Hessians with hydrogen bonds and halogen bonds, in each conventions
    for conv in ("axtb", "xtb", "xtb671"):
        add(f"gfnff-beni-acetone-hessian-{conv}", mol(BENI), method="gfnff", conventions=conv, hessian=True)
        add(f"gfnff-tyr-hessian-{conv}", mol(TYR), method="gfnff", conventions=conv, hessian=True)
    add("gfnff-cf3br-benzene-hessian-xtb671", mol(CF3BR), method="gfnff", conventions="xtb671", hessian=True)
    add("gfnff-water100-hessian", mol(W100), method="gfnff", hessian=True)
    add("gfnff-il-pair-grad", mol(IL), method="gfnff", gradient=True)
    add("gfnff-caffeine-topology-ref", caf, method="gfnff", gradient=True, topology_positions=caf["positions"])
    # vibrational analysis of a given Hessian (identity-like test matrix of water)
    add("analysis-water", {k: w[k] for k in ("numbers", "positions")},
        analyze_hessian=[[0.5 if i == j else 0.01 for j in range(9)] for i in range(9)])
    # errors and refusals
    add("error-negative-uhf", w, uhf=-1)
    add("error-unknown-method", w, method="gfn3")
    add("error-gxtb-alpb", w, method="gxtb-v2.0.1", alpb="water")
    add("error-scf-not-converged", caf, method="gfn2", max_iter=2)
    add("error-zero-threads", w, threads=0)
    add("error-hessian-step-analytic", w, method="gfn2", hessian=True, hessian_step=0.01)
    add("error-gfnff-molden", w, method="gfnff", molden=True)
    # Response Hessians, finite-temperature and open-shell Hessians, g-xTB v1.1's program-mode response
    # gradients (Z-vector GMRES, finite-temperature response)
    # GFN2-xTB: response Hessians (displacement_response), finite temperature, open shell
    add("gfn2-caffeine-hessian-response", caf, method="gfn2", hessian=True, hessian_method="response")
    add("gfn2-caffeine-hessian-response-t4", caf, method="gfn2", hessian=True, hessian_method="response", threads=4)
    add("gfn2-caffeine-hessian-etemp", caf, method="gfn2", hessian=True, etemp=3000.0)
    add("gfn2-caffeine-hessian-response-etemp", caf, method="gfn2", hessian=True, hessian_method="response", etemp=3000.0)
    add("gfn2-o2-hessian", o2, method="gfn2", hessian=True)
    add("gfn2-o2-hessian-etemp", o2, method="gfn2", hessian=True, etemp=5000.0)
    add("gfn2-o2-hessian-response-etemp", o2, method="gfn2", hessian=True, hessian_method="response", etemp=5000.0)
    add("gfn2-cu-hessian-response", cu, method="gfn2", hessian=True, hessian_method="response")
    add("gfn2-taxol-hessian-response-t1", tax, method="gfn2", hessian=True, hessian_method="response", threads=1)
    # g-xTB 2.0.1: response Hessians, finite temperature, open shell
    add("gxtb2-caffeine-hessian-response", caf, method="gxtb-v2.0.1", hessian=True, hessian_method="response")
    add("gxtb2-caffeine-hessian-etemp", caf, method="gxtb-v2.0.1", hessian=True, etemp=3000.0)
    add("gxtb2-caffeine-hessian-response-etemp", caf, method="gxtb-v2.0.1", hessian=True, hessian_method="response",
        etemp=3000.0)
    add("gxtb2-o2-hessian", o2, method="gxtb-v2.0.1", hessian=True)
    add("gxtb2-o2-hessian-etemp", o2, method="gxtb-v2.0.1", hessian=True, etemp=5000.0)
    add("gxtb2-o2-hessian-response-etemp", o2, method="gxtb-v2.0.1", hessian=True, hessian_method="response",
        etemp=5000.0)
    add("gxtb2-cu-hessian-response", cu, method="gxtb-v2.0.1", hessian=True, hessian_method="response")
    # g-xTB v1.1 in the program's conventions: Z-vector (0 K) and finite-temperature response gradients
    add("gxtb11-caffeine-grad-gxtb-etemp", caf, method="gxtb-v1.1", conventions="gxtb", gradient=True, etemp=3000.0)
    add("gxtb11-water-grad-gxtb-etemp", w, method="gxtb-v1.1", conventions="gxtb", gradient=True, etemp=5000.0)
    add("gxtb11-o2-grad-gxtb-etemp", o2, method="gxtb-v1.1", conventions="gxtb", gradient=True, etemp=5000.0)
    add("gxtb11-o2-grad-gxtb", o2, method="gxtb-v1.1", conventions="gxtb", gradient=True)
    add("gxtb11-cu-grad-gxtb", cu, method="gxtb-v1.1", conventions="gxtb", gradient=True)
    add("gxtb11-caffeine-hessian-gxtb", caf, method="gxtb-v1.1", conventions="gxtb", hessian=True, threads=8)
    # singlet O2 (closed shell): a half-filled degenerate pi* pair, strongly fractional at any etemp
    s0 = mol(O2, 0.0, 0)
    for t in (300.0, 5000.0):
        add(f"gfn2-o2s-hessian-etemp{t:.0f}", s0, method="gfn2", hessian=True, etemp=t)
        add(f"gfn2-o2s-hessian-response-etemp{t:.0f}", s0, method="gfn2", hessian=True, hessian_method="response", etemp=t)
        add(f"gxtb2-o2s-hessian-etemp{t:.0f}", s0, method="gxtb-v2.0.1", hessian=True, etemp=t)
        add(f"gxtb2-o2s-hessian-response-etemp{t:.0f}", s0, method="gxtb-v2.0.1", hessian=True, hessian_method="response",
            etemp=t)
        add(f"gxtb11-o2s-grad-gxtb-etemp{t:.0f}", s0, method="gxtb-v1.1", conventions="gxtb", gradient=True, etemp=t)
    add("gxtb11-o2s-hessian-gxtb-etemp5000", s0, method="gxtb-v1.1", conventions="gxtb", hessian=True, etemp=5000.0)
    # SCF paths: mixers, convergence rules, failures
    # GFN2-xTB's mixer ladder (axtb conventions, `auto`): Broyden runs out of iterations (H2O; DIIS restarts
    # once after a stall), Broyden stops on a lost Fermi level after 28 and 14 iterations (F2 at 3 and 3.5 Å;
    # at 3.5 Å DIIS restarts 12 times in 200 iterations, 5 times on a diverging residual), an open shell where
    # DIIS takes 169 iterations (2 stalls)
    h2o3, f23 = mol(H2O3, 0.0, 0), mol(F23, 0.0, 0)
    add("gfn2-h2o-3a-fallback", h2o3, method="gfn2", accuracy=1e-4, gradient=True)
    add("gfn2-f2-3a-fallback", f23, method="gfn2", accuracy=1e-4, gradient=True)
    add("gfn2-f2-3.5a-fallback", mol(F235, 0.0, 0), method="gfn2", accuracy=1e-4, gradient=True)
    add("gfn2-mbheavy112-fallback", mol(MBH112, 0.0, 1), method="gfn2", accuracy=1e-4, gradient=True)
    # xtb 6.7.1's per-atom residual and extra iteration: a hard SCF, an open shell, and the iteration limit
    # reached after the first converged iteration (converged: caffeine converges in 12 iterations)
    add("gfn2-h2o-3a-xtb671", h2o3, method="gfn2", conventions="xtb671", accuracy=1e-4, gradient=True)
    add("gfn2-o2-triplet-xtb671", o2, method="gfn2", conventions="xtb671", gradient=True)
    add("gfn2-caffeine-xtb671-converged-at-limit", caf, method="gfn2", conventions="xtb671", gradient=True, max_iter=11)
    # SCFs that fail: Broyden alone runs out (tblite conventions), the last attempt loses the Fermi level,
    # g-xTB v1.1 and 2.0.1 at their iteration limits, and v1.1 converged in its last iteration (its result is
    # the unmixed iteration after it: caffeine converges in 11 iterations)
    add("error-gfn2-h2o-3a-tblite", h2o3, method="gfn2", conventions="tblite", accuracy=1e-4)
    add("error-gfn2-f2-3a-broyden", f23, method="gfn2", accuracy=1e-4, mixer="broyden")
    add("error-gxtb11-scf-not-converged", caf, method="gxtb-v1.1", max_iter=5)
    add("error-gxtb11-caffeine-converged-at-limit", caf, method="gxtb-v1.1", max_iter=10)
    add("error-gxtb2-scf-not-converged", caf, method="gxtb-v2.0.1", max_iter=5)
    # g-xTB at finite temperature and open shell, energies and gradients, in both conventions
    add("gxtb2-o2-triplet-etemp-grad", o2, method="gxtb-v2.0.1", gradient=True, etemp=5000.0)
    add("gxtb2-cu-doublet-etemp-gxtb", cu, method="gxtb-v2.0.1", conventions="gxtb", etemp=3000.0)
    add("gxtb2-caffeine-etemp-gxtb", caf, method="gxtb-v2.0.1", conventions="gxtb", etemp=3000.0)
    add("gxtb11-o2-triplet-etemp-grad", o2, method="gxtb-v1.1", gradient=True, etemp=5000.0)
    add("gxtb11-cu-doublet-etemp", cu, method="gxtb-v1.1", etemp=3000.0)
    # Warm starts: a structure and two small displacements on one calculator, each SCF from the previous
    # geometry's converged wavefunction (guess "previous": GFN2's moments, g-xTB 2.0.1's state, v1.1's
    # densities); the axtb conventions of 2.0.1 restart at RESTART_ACCURACY, the program's at its own thresholds
    traj = lambda base: {**base, "trajectory": trajectory(base["positions"]), "guess": "previous"}
    for conv in ("axtb", "xtb"):
        add(f"traj-gfn2-caffeine-grad-{conv}", traj(caf), method="gfn2", conventions=conv, gradient=True)
    for conv in ("axtb", "gxtb"):
        add(f"traj-gxtb2-caffeine-grad-{conv}", traj(caf), method="gxtb-v2.0.1", conventions=conv, gradient=True)
        add(f"traj-gxtb11-caffeine-grad-{conv}", traj(caf), method="gxtb-v1.1", conventions=conv, gradient=True)
    # an open shell (2.0.1: both spin densities and the magnetizations carried) and a finite temperature (v1.1
    # in the program's conventions: the eigenvalue cap at N_alpha + 10)
    add("traj-gxtb2-cu-doublet-grad", traj(cu), method="gxtb-v2.0.1", gradient=True)
    add("traj-gxtb11-caffeine-etemp-gxtb", traj(caf), method="gxtb-v1.1", conventions="gxtb", gradient=True,
        etemp=3000.0)
    # Step 16 (the Method trait): the composite's retry ladders (g-xTB at 1000 K, the GFN2 pair at 1000 K,
    # both with 1000 iterations; in the program conventions every rung of the pair starts fresh), a composite
    # failure (an explicit temperature: no g-xTB ladder), the composite of v1.1 (numerical Hessian of the g-xTB
    # part, analytic of the GFN2 halves) and with one Hessian method for all parts
    comp = {"alpb": "water", "solvation_correction": "gfn2"}
    add("gxtb2-caffeine-composite-retries", caf, method="gxtb-v2.0.1", gradient=True, max_iter=5, **comp)
    add("gxtb2-caffeine-composite-retries-gxtb", caf, method="gxtb-v2.0.1", conventions="gxtb", gradient=True,
        max_iter=5, **comp)
    add("gxtb2-water-composite-hessian-retries", w, method="gxtb-v2.0.1", hessian=True, max_iter=3, **comp)
    add("error-gxtb2-caffeine-composite-etemp", caf, method="gxtb-v2.0.1", gradient=True, max_iter=5, etemp=300.0,
        **comp)
    add("gxtb11-water-composite-hessian", w, method="gxtb-v1.1", hessian=True, **comp)
    add("gxtb2-water-composite-hessian-numerical", w, method="gxtb-v2.0.1", hessian=True,
        hessian_method="numerical", **comp)
    # the refusals that move from the driver into the methods, and GFN-FF's numerical Hessian
    add("error-gxtb11-hessian-analytic", w, method="gxtb-v1.1", hessian=True, hessian_method="analytic")
    add("error-gfnff-hessian-response", w, method="gfnff", hessian=True, hessian_method="response")
    add("error-gfnff-hessian-accuracy", w, method="gfnff", hessian=True, hessian_accuracy=0.1)
    add("error-gfnff-bond-orders", w, method="gfnff", bond_orders=True)
    add("gfnff-water-hessian-numerical", w, method="gfnff", hessian=True, hessian_method="numerical")
    # warm starts around Hessians: GFN2 with guess "fresh" (the single point after the Hessian starts from its
    # reference), g-xTB 2.0.1 and v1.1 (the single point after the Hessian starts from the program's start,
    # the next Hessian's reference from the previous state), the composite in both conventions
    add("traj-gfn2-water-hessian-fresh", {**w, "trajectory": trajectory(w["positions"]), "guess": "fresh"},
        method="gfn2", hessian=True)
    add("traj-gfn2-water-hessian", traj(w), method="gfn2", hessian=True)
    add("traj-gxtb2-water-hessian", traj(w), method="gxtb-v2.0.1", hessian=True)
    add("traj-gxtb11-water-hessian", traj(w), method="gxtb-v1.1", hessian=True)
    for conv in ("axtb", "gxtb"):
        # each part as its conventions say (gxtb: every SCF from the programs' starts)
        add(f"traj-gxtb2-water-composite-grad-{conv}", {**w, "trajectory": trajectory(w["positions"])},
            method="gxtb-v2.0.1", conventions=conv, gradient=True, **comp)
    # the retry from the method's own start: water, then the same atoms stretched to 3 Å, where the SCF from
    # water's converged state does not converge within max_iter and the fresh one does (the second response's
    # guess is the method's own start)
    h3 = mol(H2O3, 0.0, 0)
    h3w = [h3["positions"][i] for i in (1, 2, 0)]  # in water.xyz's atom order (H, H, O)
    jump = {**w, "trajectory": [w["positions"], h3w], "guess": "previous"}
    add("traj-gfn2-water-stretch-retry", jump, method="gfn2", accuracy=1e-4, max_iter=60, gradient=True)
    add("traj-gfn2-water-stretch-retry-hessian", jump, method="gfn2", accuracy=1e-4, max_iter=60, hessian=True)
    add("traj-gxtb2-water-stretch-retry", jump, method="gxtb-v2.0.1", max_iter=40, gradient=True)
    add("traj-gxtb2-water-stretch-retry-hessian", jump, method="gxtb-v2.0.1", max_iter=40, hessian=True)
    add("traj-gxtb11-water-stretch-retry", jump, method="gxtb-v1.1", max_iter=60, gradient=True)
    return c


def trajectory(positions):
    """The structure and two small displacements of it (every coordinate by -0.02 to 0.02 bohr per step)."""
    step = lambda k: [[x + 0.01 * k * ((7 * i + 3 * c) % 5 - 2) for c, x in enumerate(r)]
                      for i, r in enumerate(positions)]
    return [step(0), step(1), step(2)]


def run(axtb: str, req: dict, env_extra: dict, trajectory_bin: str = ""):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("AXTB_LAPACK", "OMP_", "MKL_", "RAYON"))}
    env.update(env_extra)
    if "trajectory" in req:
        # one response per geometry, one per line
        req = dict(req)
        text = json.dumps({"trajectory": req.pop("trajectory"), "request": req})
        p = subprocess.run([trajectory_bin], input=text, capture_output=True, text=True, env=env)
        try:
            out = [json.loads(line) for line in p.stdout.splitlines()]
        except json.JSONDecodeError:
            out = {"unparsed_stdout": p.stdout[-2000:]}
        return {"exit_code": p.returncode, "output": strip_timings(out)}
    p = subprocess.run([axtb, "--json"], input=json.dumps(req), capture_output=True, text=True, env=env)
    try:
        out = json.loads(p.stdout)
    except json.JSONDecodeError:
        out = {"unparsed_stdout": p.stdout[-2000:]}
    return {"exit_code": p.returncode, "output": strip_timings(out)}


def trajectory_runner(axtb: str, given: str) -> str:
    """The trajectory runner for the axtb binary `axtb` (see the module documentation)."""
    if given:
        return given
    beside = Path(axtb + "-trajectory")
    return str(beside if beside.exists() else Path(axtb).parent / "examples/golden_trajectory")


def strip_timings(x):
    """The output without its timing fields, at any depth."""
    if isinstance(x, dict):
        return {k: strip_timings(v) for k, v in x.items() if k not in TIMING}
    if isinstance(x, list):
        return [strip_timings(v) for v in x]
    return x


def diff(a, b, path=""):
    """List of (path, description) where a and b differ exactly."""
    if type(a) is not type(b) and not (isinstance(a, (int, float)) and isinstance(b, (int, float))):
        return [(path, f"type {type(a).__name__} vs {type(b).__name__}")]
    if isinstance(a, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append((f"{path}.{k}", "missing in " + ("new" if k not in b else "golden")))
            else:
                out += diff(a[k], b[k], f"{path}.{k}")
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return [(path, f"length {len(a)} vs {len(b)}")]
        flat_a, flat_b = flatten(a), flatten(b)
        if flat_a is not None and flat_b is not None and len(flat_a) == len(flat_b):
            bad = [(x, y) for x, y in zip(flat_a, flat_b) if not (x == y or (isinstance(x, float) and math.isnan(x) and math.isnan(y)))]
            if bad:
                m = max(abs(x - y) for x, y in bad if isinstance(x, (int, float)) and isinstance(y, (int, float)))
                return [(path, f"{len(bad)} of {len(flat_a)} numbers differ, max |d| {m:.3e}")]
            return []
        out = []
        for i, (x, y) in enumerate(zip(a, b)):
            out += diff(x, y, f"{path}[{i}]")
        return out
    if a != b:
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return [(path, f"{a!r} vs {b!r} (|d| {abs(a - b):.3e})")]
        return [(path, f"{str(a)[:80]!r} vs {str(b)[:80]!r}")]
    return []


def flatten(x):
    """All numbers of a nested list of numbers, or None."""
    out = []
    stack = [x]
    while stack:
        v = stack.pop()
        if isinstance(v, list):
            stack.extend(reversed(v))
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            out.append(v)
        else:
            return None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["record", "compare", "list"])
    ap.add_argument("dir", nargs="?")
    ap.add_argument("--axtb", default=str(ROOT / "target/release/axtb"))
    ap.add_argument("--trajectory", default="", help="the trajectory runner (default: see above)")
    ap.add_argument("--only", default="", help="comma-separated case-name substrings")
    a = ap.parse_args()
    cases = corpus()
    if a.only:
        keys = a.only.split(",")
        cases = [c for c in cases if any(k in c[0] for k in keys)]
    if a.mode == "list":
        for name, req, env in cases:
            shown = {k: v for k, v in req.items() if k not in ("numbers", "positions", "trajectory")}
            if "trajectory" in req:
                shown["trajectory"] = f"{len(req['trajectory'])} geometries"
            print(name, json.dumps(shown), env or "")
        return
    d = Path(a.dir)
    sha = hashlib.sha256(Path(a.axtb).read_bytes()).hexdigest()[:16]
    traj = trajectory_runner(a.axtb, a.trajectory)
    if any("trajectory" in req for _, req, _ in cases) and not Path(traj).exists():
        sys.exit(f"the trajectory cases need the runner {traj} (cargo build --release -p axtb-driver --example "
                 "golden_trajectory, or --trajectory BIN)")
    if a.mode == "record":
        d.mkdir(parents=True, exist_ok=True)
        for name, req, env in cases:
            (d / f"{name}.json").write_text(json.dumps(run(a.axtb, req, env, traj)))
            print("recorded", name, flush=True)
        meta_path = d / "_meta.json"
        entry = {"axtb": a.axtb, "sha256": sha}
        if any("trajectory" in req for _, req, _ in cases):
            entry.update(trajectory=traj, trajectory_sha256=hashlib.sha256(Path(traj).read_bytes()).hexdigest()[:16])
        if a.only and meta_path.exists():
            # a partial record adds to the directory: keep the full record's entry, list these cases
            meta = json.loads(meta_path.read_text())
            meta.setdefault("partial", []).append({**entry, "cases": [n for n, _, _ in cases]})
        else:
            meta = {**entry, "cases": len(cases)}
        meta_path.write_text(json.dumps(meta))
        return
    failed = 0
    for name, req, env in cases:
        gold = json.loads((d / f"{name}.json").read_text())
        new = run(a.axtb, req, env, traj)
        ds = diff(gold, new)
        if ds:
            failed += 1
            print(f"DIFF {name}")
            for p, msg in ds[:8]:
                print(f"     {p}: {msg}")
        else:
            print(f"same {name}", flush=True)
    print(f"{len(cases) - failed} of {len(cases)} identical (binary {sha})")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
