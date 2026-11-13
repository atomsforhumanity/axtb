# SPDX-License-Identifier: LGPL-3.0-or-later
"""g-xTB 2.0.1 through the Python package and the qccodec codec.

Run with the extension built from this tree (a wheel built from ``crates/axtb-py``) and the CLI
at ``target/release/axtb`` (or ``$AXTB_BIN``)::

    python -m pytest crates/axtb-py/tests/test_gxtb2.py

The energies and gradients must equal the CLI's bit for bit (the CLI's equal the Rust API's,
see crates/axtb-cli/tests/gxtb.rs); with conventions="gxtb" they reproduce the g-xTB 2.0.1
program (total energy 1e-11 Eh, tight analytic gradient 1e-10 Eh/bohr).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

import axtb

ROOT = Path(__file__).resolve().parents[3]
REF = ROOT / "crates/gxtb/tests/reference_v2"
AXTB_BIN = Path(os.environ.get("AXTB_BIN", ROOT / "target/release/axtb"))
MCTC_AATOAU = 1.8897261246204404  # the 2.0.1 program's Angstrom-to-bohr factor (mctc-lib)
METHOD = "gxtb-v2.0.1"
CASES = [("reference/water", 0, 0), ("reference/dioxygen_triplet", 0, 2), ("reference/tetrachlorocuprate", -2, 1),
         ("elements/GdCl3", 0, 7)]
SYMBOLS = {"H": 1, "C": 6, "N": 7, "O": 8, "Cl": 17, "Cu": 29, "Gd": 64}


def read(rel):
    d = REF / rel
    lines = (d / "mol.xyz").read_text().splitlines()
    nat = int(lines[0])
    numbers, pos = [], []
    for line in lines[2:2 + nat]:
        f = line.split()
        numbers.append(SYMBOLS[f[0]])
        pos.append([float(x) * MCTC_AATOAU for x in f[1:4]])
    total = next(l for l in (d / "xtb.out").read_text().splitlines() if ":: total energy" in l)
    return numbers, np.array(pos), float(total.split()[3]), d


def program_gradient(d, name="gradient_tight"):
    lines = (d / name).read_text().splitlines()
    rows = [l.split() for l in lines[2:] if not l.startswith("$")]
    nat = sum(1 for r in rows if len(r) == 4)
    return np.array([[float(x.replace("D", "E")) for x in r] for r in rows[nat:2 * nat]])


def cli(numbers, pos, charge, uhf, gradient=False, conventions="gxtb", etemp=None):
    req = {"numbers": numbers, "positions": pos.tolist(), "charge": charge, "uhf": uhf, "method": METHOD,
           "gradient": gradient, "conventions": conventions}
    if etemp is not None:
        req["etemp"] = etemp
    out = subprocess.run([str(AXTB_BIN), "--json"], input=json.dumps(req), capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


@pytest.mark.parametrize("rel,charge,uhf", CASES)
def test_python_equals_cli_and_program(rel, charge, uhf):
    numbers, pos, program, d = read(rel)
    r = axtb.singlepoint(numbers, pos, charge=charge, uhf=uhf, method=METHOD, conventions="gxtb")
    assert abs(r["energy"] - program) < 1e-11
    assert abs(sum(r["charges"]) - charge) < 1e-9
    calc = axtb.Calculator(METHOD, numbers, pos, charge=charge, uhf=uhf, threads=2, conventions="gxtb")
    rc = calc.singlepoint(pos, gradient=False)
    assert rc["energy"] == r["energy"]
    assert np.array_equal(rc["charges"], r["charges"])
    # tight analytic gradient against the program's
    t = axtb.singlepoint(numbers, pos, charge=charge, uhf=uhf, method=METHOD, gradient=True, conventions="gxtb",
                         accuracy=1e-4)
    assert np.abs(np.asarray(t["gradient"]) - program_gradient(d)).max() < 1e-10
    g = axtb.singlepoint(numbers, pos, charge=charge, uhf=uhf, method=METHOD, gradient=True, conventions="gxtb")
    assert np.array_equal(calc.singlepoint(pos, gradient=True)["gradient"], g["gradient"])
    if AXTB_BIN.exists():
        c = cli(numbers, pos, charge, uhf, gradient=True)
        assert c["energy"] == g["energy"]
        assert c["terms"] == g["terms"]
        assert np.array_equal(np.asarray(c["gradient"]), np.asarray(g["gradient"]))
    # default conventions: same SCF, CODATA 2022 units, integer electron count, smooth cutoffs
    dflt = axtb.singlepoint(numbers, pos, charge=charge, uhf=uhf, method=METHOD, gradient=True)
    assert abs(dflt["energy"] - program) < 1e-8
    assert abs(sum(dflt["charges"]) - charge) < 1e-8
    if AXTB_BIN.exists():
        c = cli(numbers, pos, charge, uhf, gradient=True, conventions="axtb")
        assert c["energy"] == dflt["energy"]
        assert np.array_equal(np.asarray(c["gradient"]), np.asarray(dflt["gradient"]))


def test_hessian():
    numbers, pos, _, _ = read("reference/water")
    calc = axtb.Calculator(METHOD, numbers, pos, threads=2)
    h = calc.hessian(pos)
    hess = np.asarray(h["hessian"])
    assert hess.shape == (9, 9)
    assert np.array_equal(hess, hess.T)
    assert h["hessian_max_asymmetry"] < 2e-5
    assert np.abs(hess.reshape(9, 3, 3).sum(axis=1)).max() < 1e-5
    serial = axtb.Calculator(METHOD, numbers, pos).hessian(pos)
    assert np.allclose(serial["hessian"], hess, atol=1e-12)
    # the default is analytic; the numerical Hessian (central differences) agrees
    num = np.asarray(calc.hessian(pos, method="numerical")["hessian"])
    assert np.abs(num - hess).max() < 1e-4


def test_errors():
    numbers, pos, _, _ = read("reference/water")
    with pytest.raises(ValueError, match="gxtb-v2.0.1 has no implicit-solvent parameters.*solvation_correction"):
        axtb.singlepoint(numbers, pos, method=METHOD, alpb="water")
    with pytest.raises(ValueError, match="unknown g-xTB conventions"):
        axtb.singlepoint(numbers, pos, method=METHOD, conventions="tblite")
    # v1.1 keeps working next to 2.0.1
    v1 = axtb.singlepoint(numbers, pos, method="gxtb-v1.1")
    v2 = axtb.singlepoint(numbers, pos, method="g-xTB 2.0.1")
    assert v1["energy"] != v2["energy"]


def test_qccodec():
    qccompute = pytest.importorskip("qccompute")
    from qcdata import CalcType, Model, ProgramInput, Structure

    numbers, pos, program, d = read("reference/tetrachlorocuprate")
    names = {v: k for k, v in SYMBOLS.items()}
    structure = Structure(symbols=[names[z] for z in numbers], geometry=pos, charge=-2, multiplicity=2)
    out = qccompute.compute(ProgramInput(program="axtb", calctype=CalcType.energy, structure=structure,
                                         model=Model(method="gxtb-v2.0.1"), keywords={"conventions": "gxtb"}))
    assert out.success, out.logs
    assert abs(out.results.energy - program) < 1e-11
    assert out.results.extras["conventions"] == "gxtb"
    assert "dispersion_three_body" in out.results.extras["energy_terms"]
    out = qccompute.compute(ProgramInput(program="axtb", calctype=CalcType.gradient, structure=structure,
                                         model=Model(method="gxtb-v2.0.1")))
    assert out.success, out.logs
    direct = axtb.singlepoint(numbers, pos, charge=-2, uhf=1, method=METHOD, gradient=True)
    assert np.array_equal(np.asarray(out.results.gradient), np.asarray(direct["gradient"]))
    out = qccompute.compute(ProgramInput(program="axtb", calctype=CalcType.hessian, structure=structure,
                                         model=Model(method="gxtb-v2.0.1"), keywords={"threads": 2}))
    assert out.success, out.logs
    assert np.asarray(out.results.hessian).shape == (15, 15)
    # Fermi smearing (the program's --etemp) through the codec
    out = qccompute.compute(ProgramInput(program="axtb", calctype=CalcType.energy, structure=structure,
                                         model=Model(method="gxtb-v2.0.1"),
                                         keywords={"conventions": "gxtb", "etemp": 1000.0}))
    assert out.success, out.logs
    assert abs(out.results.energy - etemp_reference("tetrachlorocuprate", 1000.0)["total_energy"]) < 1e-11
    # implicit solvent alone is refused with a pointer to the composite (solvation_correction)
    with pytest.raises(Exception, match="solvation_correction"):
        qccompute.compute(ProgramInput(program="axtb", calctype=CalcType.energy, structure=structure,
                                       model=Model(method="gxtb-v2.0.1"), keywords={"alpb": "water"}),
                          raise_exc=True)


def etemp_reference(name, etemp):
    rows = json.loads((REF / "etemp/etemp.json").read_text())["rows"]
    return next(r for r in rows if r["name"] == name and r["etemp"] == etemp)


@pytest.mark.parametrize("rel,charge,uhf", CASES)
def test_etemp_matches_program(rel, charge, uhf):
    """Fermi smearing at 1000 K (the program's --etemp 1000, as the composite's g-xTB retry ladder uses it):
    energies, SCF iterations and analytic gradients as the 2.0.1 program's; the CLI equals the
    Python result bit for bit."""
    numbers, pos, _, _ = read(rel)
    ref = etemp_reference(rel.split("/")[1], 1000.0)
    r = axtb.singlepoint(numbers, pos, charge=charge, uhf=uhf, method=METHOD, gradient=True, conventions="gxtb",
                         etemp=1000.0, max_iter=1000)
    assert abs(r["energy"] - ref["total_energy"]) < 1e-11
    assert r["iterations"] == ref["scf_iterations"]
    assert np.abs(np.asarray(r["gradient"]) - np.asarray(ref["gradient"])).max() < 1e-9
    assert "entropy" in r["terms"]
    if AXTB_BIN.exists():
        c = cli(numbers, pos, charge, uhf, gradient=True, etemp=1000.0)
        assert c["energy"] == r["energy"]
        assert np.array_equal(np.asarray(c["gradient"]), np.asarray(r["gradient"]))


def test_etemp_hessian():
    """Analytic Hessian at 5000 K (stretched N2, fractional occupations) against the numerical
    one, and the same for 1 and 2 threads."""
    d = REF / "etemp/extra/N2_stretched"
    lines = (d / "mol.xyz").read_text().splitlines()
    numbers = [7, 7]
    pos = np.array([[float(x) * MCTC_AATOAU for x in l.split()[1:4]] for l in lines[2:4]])
    one = axtb.Calculator(METHOD, numbers, pos, etemp=5000.0)
    h1 = np.asarray(one.hessian(pos)["hessian"])
    two = axtb.Calculator(METHOD, numbers, pos, etemp=5000.0, threads=2)
    assert np.array_equal(np.asarray(two.hessian(pos)["hessian"]), h1)
    num = np.asarray(one.hessian(pos, method="numerical")["hessian"])
    assert np.abs(num - h1).max() < 1e-5
    # smearing changes the curvature of this stretched bond (5e-5 Eh/bohr^2)
    cold = np.asarray(axtb.Calculator(METHOD, numbers, pos).hessian(pos)["hessian"])
    assert np.abs(cold - h1).max() > 2e-5
