# SPDX-License-Identifier: LGPL-3.0-or-later
"""GFN-FF through the Python package, the CLI and the qccodec codec.

Run with the extension built from this tree (a wheel built from ``crates/axtb-py``) and the CLI
at ``target/release/axtb`` (or ``$AXTB_BIN``)::

    python -m pytest crates/axtb-py/tests/test_gfnff.py

Python results equal the CLI's bit for bit (the CLI's equal the Rust API's, see
crates/axtb-cli/tests/gfnff.rs); in xtb's conventions the Hessian's thermochemistry is
`xtb --gfnff --hess`'s (frequency scale 1.03).
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
AXTB_BIN = Path(os.environ.get("AXTB_BIN", ROOT / "target/release/axtb"))
MCTC_AATOAU = 1.8897261246204404  # xtb's (mctc-lib's) Angstrom-to-bohr factor
SYMBOLS = {"H": 1, "C": 6, "N": 7, "O": 8}
# `xtb caffeine.xyz --gfnff --hess` (xtb 6.8.0): total free energy, ZPVE
XTB_CAFFEINE_G, XTB_CAFFEINE_ZPVE = -4.533411183087, 0.178808030298
# the same with xtb 6.7.1 (the g-xTB 2.0.1 build): its force field and Cartesian projection
XTB671_CAFFEINE_G, XTB671_CAFFEINE_ZPVE = -4.532486509084, 0.178674505263
WATERS = ([8, 1, 1, 8, 1, 1], np.array([[0.0, 0.0, 0.0], [1.8, 0.0, 0.0], [-0.45, 1.75, 0.0],
                                        [9.0, 0.0, 0.0], [10.8, 0.0, 0.0], [8.55, 1.75, 0.0]]))


def caffeine():
    lines = (ROOT / "tests/structures/caffeine.xyz").read_text().splitlines()
    nat = int(lines[0])
    rows = [l.split() for l in lines[2:2 + nat]]
    return [SYMBOLS[r[0]] for r in rows], np.array([[float(x) * MCTC_AATOAU for x in r[1:4]] for r in rows])


def cli(numbers, pos, **extra):
    req = {"numbers": numbers, "positions": pos.tolist(), "method": "gfnff", **extra}
    out = subprocess.run([str(AXTB_BIN), "--json"], input=json.dumps(req), capture_output=True, text=True, check=True)
    return json.loads(out.stdout)


def test_singlepoint_equals_cli():
    numbers, pos = caffeine()
    r = axtb.singlepoint(numbers, pos, method="gfnff", gradient=True)
    c = cli(numbers, pos, gradient=True)
    assert r["energy"] == c["energy"]
    assert np.array_equal(r["gradient"], np.array(c["gradient"]))
    assert np.array_equal(r["charges"], np.array(c["charges"]))
    assert r["terms"] == c["terms"]
    t = r["topology"]
    assert t["bonds"].shape == (len(c["topology"]["bonds"]), 2)
    assert t["bonds"].tolist() == c["topology"]["bonds"]
    assert t["hybridization"].tolist() == c["topology"]["hybridization"]
    assert t["fragments"].tolist() == [0] * len(numbers)
    assert abs(r["charges"].sum()) < 1e-12


def test_fixed_topology():
    numbers, pos = caffeine()
    moved = pos + np.random.default_rng(1).normal(0.0, 0.05, pos.shape)
    calc = axtb.Calculator("gfnff", numbers, pos)
    a = calc.singlepoint(moved, gradient=True)
    b = axtb.singlepoint(numbers, moved, method="gfnff", gradient=True, topology_positions=pos)
    assert a["energy"] == b["energy"]
    assert np.array_equal(a["gradient"], b["gradient"])
    c = cli(numbers, moved, gradient=True, topology_positions=pos.tolist())
    assert a["energy"] == c["energy"]


def test_fragment_charges():
    numbers, pos = WATERS
    r = axtb.singlepoint(numbers, pos, method="gfnff", fragment_charges=[1.0, -1.0])
    assert r["topology"]["fragment_charges"].tolist() == [1.0, -1.0]
    assert r["charges"][:3].sum() == pytest.approx(1.0, abs=1e-10)
    assert r["charges"][3:].sum() == pytest.approx(-1.0, abs=1e-10)
    # default: the total charge on the first fragment
    r = axtb.singlepoint(numbers, pos, method="gfnff", charge=-1.0)
    assert r["charges"][:3].sum() == pytest.approx(-1.0, abs=1e-10)


def test_hessian_thermo_xtb_conventions():
    numbers, pos = caffeine()
    h = axtb.Calculator("gfnff", numbers, pos, conventions="xtb").hessian(pos, method="numerical")
    assert h["hessian_method"] == "numerical" and h["vibrational_conventions"] == "xtb"
    assert h["thermo"]["frequency_scale"] == 1.03
    assert h["thermo"]["free_energy"] == pytest.approx(XTB_CAFFEINE_G, abs=1e-10)
    assert h["thermo"]["zpve"] == pytest.approx(XTB_CAFFEINE_ZPVE, abs=1e-10)
    h = axtb.Calculator("gfnff", numbers, pos, conventions="xtb-6.7.1").hessian(pos, method="numerical")
    assert h["vibrational_conventions"] == "xtb-6.7.1" and h["thermo"]["frequency_scale"] == 1.03
    assert h["thermo"]["free_energy"] == pytest.approx(XTB671_CAFFEINE_G, abs=1e-10)
    assert h["thermo"]["zpve"] == pytest.approx(XTB671_CAFFEINE_ZPVE, abs=1e-10)
    a = axtb.Calculator("gfnff", numbers, pos).hessian(pos)
    assert a["hessian_method"] == "analytic" and a["vibrational_conventions"] == "axtb"
    assert a["thermo"]["frequency_scale"] == 1.0
    assert a["dipole_derivatives"].shape == (3 * len(numbers), 3)
    assert a["hessian_max_asymmetry"] < 1e-12
    c = cli(numbers, pos, hessian=True, vibrations=False)
    assert np.array_equal(a["hessian"], np.array(c["hessian"]))
    assert np.array_equal(a["dipole_derivatives"], np.array(c["dipole_derivatives"]))
    s = axtb.Calculator("gfnff", numbers, pos).hessian(pos, frequency_scale=1.03, analysis_conventions="xtb")
    assert s["thermo"]["frequency_scale"] == 1.03
    # the default (axtb) and xtb conventions share 6.8.0's force field; 6.7.1's differs
    e = {c: axtb.Calculator("gfnff", numbers, pos, conventions=c).singlepoint(pos)["energy"] for c in ("axtb", "xtb", "xtb-6.7.1")}
    assert e["axtb"] == e["xtb"] and 1e-9 < abs(e["xtb"] - e["xtb-6.7.1"]) < 1e-6


def test_rejected_settings():
    numbers, pos = WATERS
    for kw in ({"etemp": 300.0}, {"alpb": "water"}, {"mixer": "diis"}, {"accuracy": 0.1}, {"conventions": "gxtb"}):
        with pytest.raises(ValueError, match="gfnff"):
            axtb.singlepoint(numbers, pos, method="gfnff", **kw)
    with pytest.raises(ValueError, match="orbitals"):
        axtb.singlepoint(numbers, pos, method="gfnff", molden=True)
    with pytest.raises(ValueError, match="GFN-FF settings"):
        axtb.singlepoint(numbers, pos, method="gfn2", fragment_charges=[0.0, 0.0])
    calc = axtb.Calculator("gfnff", numbers, pos)
    with pytest.raises(axtb.InputError):
        calc.hessian(pos, method="response")
    with pytest.raises(axtb.InputError):
        calc.hessian(pos, accuracy=0.1)
    with pytest.raises(axtb.InputError):
        calc.set_guess([], np.zeros((6, 3)), np.zeros((6, 6)))
    with pytest.raises(axtb.InputError, match="threads"):
        axtb.Calculator("gfnff", numbers, pos, threads=0)
    # errors of the Hessian keep their kind: invalid positions are input, not a numerical failure
    bad = pos.copy()
    bad[0, 0] = np.nan
    with pytest.raises(axtb.InputError) as err:
        calc.hessian(bad)
    assert err.value.kind == "input"


def test_qccodec():
    qccompute = pytest.importorskip("qccompute")
    from qcdata import CalcType, Model, ProgramInput, Structure

    numbers, pos = caffeine()
    syms = {v: k for k, v in SYMBOLS.items()}
    structure = Structure(symbols=[syms[z] for z in numbers], geometry=pos)
    out = qccompute.compute(ProgramInput(program="axtb", calctype=CalcType.gradient, structure=structure,
                                         model=Model(method="gfnff")))
    assert out.success, out.logs
    r = axtb.singlepoint(numbers, pos, method="gfnff", gradient=True)
    assert out.results.energy == r["energy"]
    assert np.array_equal(np.asarray(out.results.gradient), r["gradient"])
    assert np.array_equal(np.asarray(out.results.atomic_charges), r["charges"])
    ex = out.results.extras
    assert ex["topology"]["bonds"] == r["topology"]["bonds"].tolist()
    assert ex["energy_terms"]["bond"] == r["terms"]["bond"]
    moved = pos + 0.02
    out = qccompute.compute(ProgramInput(program="axtb", calctype=CalcType.energy,
                                         structure=Structure(symbols=structure.symbols, geometry=moved),
                                         model=Model(method="gfnff"), keywords={"topology_positions": pos.tolist()}))
    assert out.success, out.logs
    assert out.results.energy == axtb.Calculator("gfnff", numbers, pos).singlepoint(moved)["energy"]
    out = qccompute.compute(ProgramInput(program="axtb", calctype=CalcType.hessian, structure=structure,
                                         model=Model(method="gfnff"),
                                         keywords={"conventions": "xtb", "hessian_method": "numerical"}))
    assert out.success, out.logs
    assert out.results.gibbs_free_energy == pytest.approx(XTB_CAFFEINE_G, abs=1e-10)
    assert out.results.extras["thermo"]["frequency_scale"] == 1.03
    assert len(out.results.extras["vibrations"]["dipole_derivatives"]) == 3 * len(numbers)
    with pytest.raises(Exception, match="Unknown axtb keywords"):
        qccompute.compute(ProgramInput(program="axtb", calctype=CalcType.energy, structure=structure,
                                       model=Model(method="gfnff"), keywords={"etemp": 300.0}), raise_exc=True)
