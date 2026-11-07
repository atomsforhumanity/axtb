"""Vibrational analysis through the Python package, the CLI and the qccodec codec.

Run with the extension built from this tree (a wheel built from ``crates/axtb-py``) and the CLI
at ``target/release/axtb`` (or ``$AXTB_BIN``)::

    python -m pytest crates/axtb-py/tests/test_vibrations.py

The comparison with the xtb program (frequencies, thermochemistry, symmetry numbers) is
``tests/vibrational_check.py``; here: the interfaces agree with each other and with the
textbook formulas.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

import axtb

ROOT = Path(__file__).resolve().parents[3]
AXTB_BIN = Path(os.environ.get("AXTB_BIN", ROOT / "target/release/axtb"))
KB = 3.1668115634564e-6  # CODATA 2022 kelvin-hartree relationship

WATER = ([8, 1, 1], np.array([[0.0, 0.0, 0.2214], [0.0, 1.4304, -0.8857], [0.0, -1.4304, -0.8857]]))
CO2 = ([6, 8, 8], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.1953], [0.0, 0.0, -2.1953]]))


def unit(modes):
    return np.allclose(np.linalg.norm(modes.reshape(len(modes), -1), axis=1), 1.0)


def test_calculator_hessian_has_vibrations():
    numbers, pos = WATER
    h = axtb.Calculator("gfn2", numbers, pos).hessian(pos)
    assert h["frequencies"].shape == (3,)
    assert h["normal_modes"].shape == (3, 3, 3) and unit(h["normal_modes"])
    assert h["dipole_derivatives"].shape == (9, 3)
    assert h["ir_intensities"].shape == (3,) and (h["ir_intensities"] > 0).all()
    assert h["vibrational_conventions"] == "axtb" and not h["linear"]
    t = h["thermo"]
    assert (t["point_group"], t["symmetry_number"]) == ("C2v", 2)
    assert t["free_energy"] == pytest.approx(h["energy"] + t["g_rrho"], abs=1e-12)
    assert t["g_rrho"] == pytest.approx(t["enthalpy"] - t["ts"], abs=1e-15)
    assert t["ts"] == pytest.approx(t["temperature"] * t["entropy"], rel=1e-14)
    # the standalone analysis of the same Hessian is the same analysis
    v = axtb.vibrational_analysis(numbers, pos, h["hessian"], dipole_derivatives=h["dipole_derivatives"],
                                  energy=h["energy"])
    assert np.array_equal(v["frequencies"], h["frequencies"])
    assert np.array_equal(v["ir_intensities"], h["ir_intensities"])
    assert v["thermo"]["free_energy"] == t["free_energy"]
    # xtb conventions of the calculation select the xtb analysis
    hx = axtb.Calculator("gfn2", numbers, pos, conventions="xtb").hessian(pos)
    assert hx["vibrational_conventions"] == "xtb"
    assert "frequencies" not in axtb.Calculator("gfn2", numbers, pos).hessian(pos, vibrations=False)


def test_cli_matches_python():
    if not AXTB_BIN.exists():
        pytest.skip("no CLI")
    numbers, pos = WATER
    req = {"numbers": numbers, "positions": pos.tolist(), "hessian": True, "temperature": [250.0, 300.0]}
    d = json.loads(subprocess.run([str(AXTB_BIN), "--json"], input=json.dumps(req), capture_output=True, text=True,
                                  check=True).stdout)
    h = axtb.Calculator("gfn2", numbers, pos).hessian(pos, temperature=[250.0, 300.0])
    assert np.allclose(d["frequencies"], h["frequencies"], rtol=0, atol=1e-9)
    assert d["thermo"]["temperature"] == 300.0 and len(d["thermo"]["points"]) == 2
    assert d["thermo"]["g_rrho"] == pytest.approx(h["thermo"]["g_rrho"], abs=1e-12)
    # analysis only, like `xtb thermo`
    a = json.loads(subprocess.run([str(AXTB_BIN), "--json"], capture_output=True, text=True, check=True,
                                  input=json.dumps({"numbers": numbers, "positions": pos.tolist(),
                                                    "analyze_hessian": d["hessian"], "analysis_conventions": "xtb"})).stdout)
    assert a["vibrational_conventions"] == "xtb" and len(a["frequencies"]) == 3


def test_thermo_options():
    numbers, pos = WATER
    h = axtb.Calculator("gfn2", numbers, pos).hessian(pos, vibrations=False)["hessian"]
    base = axtb.vibrational_analysis(numbers, pos, h)["thermo"]
    # symmetry number: G changes by kT ln(sigma) (rotational entropy -k ln sigma)
    sym1 = axtb.vibrational_analysis(numbers, pos, h, symmetry_number=1)["thermo"]
    assert sym1["g_rrho"] - base["g_rrho"] == pytest.approx(-KB * 298.15 * math.log(2), rel=1e-10)
    # pressure: translational entropy -k ln p
    p2 = axtb.vibrational_analysis(numbers, pos, h, pressure=2 * 101325.0)["thermo"]
    assert p2["g_rrho"] - base["g_rrho"] == pytest.approx(KB * 298.15 * math.log(2), rel=1e-10)
    # several temperatures: the last is reported, all are kept
    many = axtb.vibrational_analysis(numbers, pos, h, temperature=[298.15, 500.0])["thermo"]
    assert many["temperature"] == 500.0 and many["points"][0]["g_rrho"] == pytest.approx(base["g_rrho"], abs=1e-15)
    # rotor cutoff 0: harmonic oscillators only; frequencies here are far above 50 cm-1 anyway
    ho = axtb.vibrational_analysis(numbers, pos, h, rotor_cutoff=0.0)["thermo"]
    assert ho["g_rrho"] == pytest.approx(base["g_rrho"], abs=1e-9)
    # isotopes: D2O frequencies drop by about 1/sqrt(2)
    d2o = axtb.vibrational_analysis(numbers, pos, h, masses=[15.999, 2.014, 2.014])
    assert d2o["frequencies"][2] / axtb.vibrational_analysis(numbers, pos, h)["frequencies"][2] == pytest.approx(0.73, abs=0.02)
    assert "thermo" not in axtb.vibrational_analysis(numbers, pos, h, thermo=False)
    with pytest.raises(ValueError):
        axtb.vibrational_analysis(numbers, pos, h, temperature=0.0)
    with pytest.raises(ValueError):
        axtb.vibrational_analysis(numbers, pos, h[:6, :6])
    with pytest.raises(ValueError):
        axtb.vibrational_analysis(numbers, pos, h, conventions="tblite")


def test_linear_and_gxtb():
    numbers, pos = CO2
    h = axtb.Calculator("gfn2", numbers, pos).hessian(pos)
    assert h["linear"] and h["frequencies"].shape == (4,)
    assert (h["thermo"]["point_group"], h["thermo"]["symmetry_number"]) == ("Dinfh", 2)
    numbers, pos = WATER
    g = axtb.Calculator("gxtb-v2.0.1", numbers, pos, conventions="gxtb").hessian(pos)
    assert g["vibrational_conventions"] == "xtb-6.7.1" and g["frequencies"].shape == (3,)
    assert g["ir_intensities"].shape == (3,) and g["thermo"]["symmetry_number"] == 2
    g1 = axtb.Calculator("gxtb-v1.1", numbers, pos).hessian(pos)
    assert g1["frequencies"].shape == (3,) and g1["ir_intensities"].shape == (3,)


def test_qccodec_fields():
    qccompute = pytest.importorskip("qccompute")
    from qccodec.testing import check_normal_modes
    from qcdata import CalcType, Model, ProgramInput, Structure

    numbers, pos = WATER
    structure = Structure(symbols=["O", "H", "H"], geometry=pos)
    out = qccompute.compute(ProgramInput(program="axtb", calctype=CalcType.hessian, structure=structure,
                                         model=Model(method="gfn2"), keywords={"temperature": [298.15, 350.0]}))
    assert out.success, out.logs
    r = out.results
    assert len(r.freqs_wavenumber) == 3
    assert np.asarray(r.normal_modes_cartesian).shape == (3, 3, 3)
    check_normal_modes(r, ["O", "H", "H"])  # qcdata's convention, qcref masses
    t = r.extras["thermo"]
    assert t["temperature"] == 350.0
    assert r.gibbs_free_energy == pytest.approx(r.energy + t["g_rrho"], abs=1e-10)
    vib = r.extras["vibrations"]
    assert vib["n_imaginary"] == 0 and len(vib["ir_intensities"]) == 3 and len(vib["dipole_derivatives"]) == 9
    out = qccompute.compute(ProgramInput(program="axtb", calctype=CalcType.hessian, structure=structure,
                                         model=Model(method="gxtb-v2.0.1"),
                                         keywords={"conventions": "gxtb", "symmetry_number": 1}))
    assert out.success, out.logs
    assert out.results.extras["thermo"]["symmetry_number"] == 1
    assert out.results.extras["vibrations"]["conventions"] == "xtb-6.7.1"  # the 2.0.1 program is 6.7.1-based
    with pytest.raises(Exception, match="Unknown axtb keywords"):
        qccompute.compute(ProgramInput(program="axtb", calctype=CalcType.hessian, structure=structure,
                                       model=Model(method="gfn2"), keywords={"sthr": 50.0}), raise_exc=True)
