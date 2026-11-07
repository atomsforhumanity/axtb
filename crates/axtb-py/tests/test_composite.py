"""g-xTB in implicit solvent (the composite E_gxtb + [E_GFN2+solv - E_GFN2]) through the Python
package, the qccodec codec and the CLI.

    python -m pytest crates/axtb-py/tests/test_composite.py

Needs the extension built from this tree and the CLI at ``target/release/axtb`` (or
``$AXTB_BIN``). The comparison with the three xtb programs is ``tests/solvation_composite.py``.
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
METHOD = "gxtb-v2.0.1"
# acetate (anion), bohr
A = 1.0 / 0.52917726
NUMBERS = [6, 6, 8, 8, 1, 1, 1]
POS = A * np.array([[0.95727, -0.02507, 0.04245], [2.47592, -0.04629, 0.05497], [3.03349, 0.92888, -0.52062],
                    [2.98681, -1.04103, 0.64211], [0.57548, 0.85802, -0.47879], [0.57813, -0.01187, 1.06868],
                    [0.57813, -0.91714, -0.46504]])
COMPOSITE = {"alpb": "water", "solvation_correction": "gfn2"}


@pytest.mark.parametrize("conventions,gfn2_conventions", [("axtb", "axtb"), ("gxtb", "xtb-6.7.1")])
def test_composite_is_the_sum_of_its_parts(conventions, gfn2_conventions):
    r = axtb.singlepoint(NUMBERS, POS, charge=-1, method=METHOD, gradient=True, conventions=conventions, **COMPOSITE)
    g = axtb.singlepoint(NUMBERS, POS, charge=-1, method=METHOD, gradient=True, conventions=conventions)
    solv = axtb.singlepoint(NUMBERS, POS, charge=-1, gradient=True, conventions=gfn2_conventions, alpb="water")
    gas = axtb.singlepoint(NUMBERS, POS, charge=-1, gradient=True, conventions=gfn2_conventions)
    dg = solv["energy"] - gas["energy"]
    assert abs(r["energy"] - (g["energy"] + dg)) < 1e-12
    assert np.abs(r["gradient"] - (g["gradient"] + solv["gradient"] - gas["gradient"])).max() < 1e-11
    assert r["components"] == pytest.approx({"gxtb": g["energy"], "gfn2_solvated": solv["energy"],
                                             "gfn2_gas": gas["energy"], "solvation_correction": dg}, abs=1e-12)
    assert r["terms"]["solvation_correction"] == r["components"]["solvation_correction"]
    # the charges and moments are g-xTB's
    assert np.array_equal(r["charges"], g["charges"])
    c = r["composite"]
    assert c["definition"] == "E = E[gxtb-v2.0.1] + (E[gfn2 + alpb(water, gsolv)] - E[gfn2])"
    assert (c["conventions"], c["correction_conventions"]) == (conventions, gfn2_conventions)
    assert c["etemp"] == {"gxtb": 0.0, "gfn2": 300.0}
    assert c["reuse_guess"] == (conventions == "axtb")
    p = r["parts"]
    assert set(p) == {"gxtb", "gfn2_solvated", "gfn2_gas"}
    assert p["gfn2_gas"]["mixer_used"] == "broyden" and p["gxtb"]["mixer_used"] is None
    assert all(x["retry"] is None for x in p.values())


def test_rejections():
    with pytest.raises(ValueError, match="solvation_correction"):
        axtb.singlepoint(NUMBERS, POS, charge=-1, method=METHOD, alpb="water")
    with pytest.raises(ValueError, match="solvation_correction"):
        axtb.Calculator(METHOD, NUMBERS, POS, charge=-1, gbsa="water")
    with pytest.raises(ValueError, match="needs a solvent"):
        axtb.singlepoint(NUMBERS, POS, charge=-1, method=METHOD, solvation_correction="gfn2")
    with pytest.raises(axtb.InputError, match="settings of g-xTB in solvent"):
        axtb.singlepoint(NUMBERS, POS, charge=-1, method="gfn2", **COMPOSITE)
    with pytest.raises(ValueError, match="unknown solvation correction"):
        axtb.singlepoint(NUMBERS, POS, charge=-1, method=METHOD, alpb="water", solvation_correction="gfn1")
    with pytest.raises(ValueError, match="dispersion"):
        axtb.singlepoint(NUMBERS, POS, charge=-1, method=METHOD, dispersion=False, **COMPOSITE)


def test_calculator_and_hessian():
    calc = axtb.Calculator(METHOD, NUMBERS, POS, charge=-1, **COMPOSITE)
    a = calc.singlepoint(POS)
    b = calc.singlepoint(POS + 0.01)
    assert a["energy"] != b["energy"]
    h = calc.hessian(POS)
    assert h["hessian"].shape == (21, 21)
    assert h["hessian_method"] == "analytic"
    assert {p["hessian_method"] for p in h["hessian_parts"].values()} == {"analytic"}
    # the response's single point and its parts are those at POS, as without the Hessian
    assert "hessian_method" not in h["parts"]["gxtb"]
    assert abs(h["energy"] - a["energy"]) < 1e-6
    # the vibrational analysis of the composite Hessian, IR from g-xTB's dipole
    assert len(h["frequencies"]) == 15 and h["ir_intensities"] is not None
    with pytest.raises(ValueError, match="GFN2-only"):
        calc.set_guess(np.zeros(1), np.zeros((7, 3)), np.zeros((7, 6)))
    calc.reset_guess()


def test_molden_of_the_composite():
    r = axtb.singlepoint(NUMBERS, POS, charge=-1, method=METHOD, molden=True, **COMPOSITE)
    assert r["molden"].startswith("[Molden Format]")


def test_qccodec():
    qccompute = pytest.importorskip("qccompute")
    from qcdata import CalcType, Model, ProgramInput, Structure

    s = Structure(symbols=["C", "C", "O", "O", "H", "H", "H"], geometry=POS, charge=-1, multiplicity=1)
    out = qccompute.compute(ProgramInput(program="axtb", calctype=CalcType.gradient, structure=s,
                                         model=Model(method=METHOD), keywords={**COMPOSITE, "conventions": "gxtb"}))
    assert out.success, out.logs
    direct = axtb.singlepoint(NUMBERS, POS, charge=-1, method=METHOD, gradient=True, conventions="gxtb", **COMPOSITE)
    assert out.results.energy == direct["energy"]
    ex = out.results.extras
    assert ex["composite"]["correction_conventions"] == "xtb-6.7.1"
    assert set(ex["components"]) == {"gxtb", "gfn2_solvated", "gfn2_gas", "solvation_correction"}
    assert ex["parts"]["gfn2_solvated"]["method"] == "gfn2+alpb"
    out = qccompute.compute(ProgramInput(program="axtb", calctype=CalcType.hessian, structure=s,
                                         model=Model(method=METHOD), keywords=COMPOSITE))
    assert out.success, out.logs
    assert out.results.extras["hessian_parts"]["gxtb"]["hessian_method"] == "analytic"
    for bad, msg in (({"alpb": "water"}, "solvation_correction"), ({"solvation_correction": "gfn2"}, "needs a solvent")):
        with pytest.raises(Exception, match=msg):
            qccompute.compute(ProgramInput(program="axtb", calctype=CalcType.energy, structure=s,
                                           model=Model(method=METHOD), keywords=bad), raise_exc=True)


@pytest.mark.skipif(not AXTB_BIN.exists(), reason="needs the release CLI")
def test_cli_equals_python():
    req = {"numbers": NUMBERS, "positions": POS.tolist(), "charge": -1, "method": METHOD, "gradient": True,
           "conventions": "gxtb", **COMPOSITE}
    out = json.loads(subprocess.run([str(AXTB_BIN), "--json"], input=json.dumps(req), capture_output=True, text=True,
                                    check=True).stdout)
    direct = axtb.singlepoint(NUMBERS, POS, charge=-1, method=METHOD, gradient=True, conventions="gxtb", **COMPOSITE)
    assert out["energy"] == direct["energy"]
    assert np.array_equal(np.asarray(out["gradient"]), direct["gradient"])
    assert out["composite"]["definition"] == direct["composite"]["definition"]
    req.pop("solvation_correction")
    bad = subprocess.run([str(AXTB_BIN), "--json"], input=json.dumps(req), capture_output=True, text=True)
    err = json.loads(bad.stdout)["error"]
    assert bad.returncode == 1 and err["kind"] == "input" and "solvation-correction" in err["message"]
    assert err["message"] == str(pytest.raises(axtb.InputError, axtb.singlepoint, NUMBERS, POS, charge=-1,
                                               method=METHOD, alpb="water").value)
