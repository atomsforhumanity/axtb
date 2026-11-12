"""Broken-symmetry unrestricted g-xTB singlets (``broken_symmetry=True``, an axtb extension)
through the Python package, the qccodec codec and the CLI.

Run with the extension built from this tree and the CLI at ``target/release/axtb`` (or
``$AXTB_BIN``)::

    python -m pytest crates/axtb-py/tests/test_broken_symmetry.py
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
AA = 1 / 0.529177210544
METHODS = ["gxtb-v2.0.1", "gxtb-v1.1"]


def h2(d):
    return [1, 1], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, d * AA]])


@pytest.mark.parametrize("method", METHODS)
def test_singlepoint(method):
    z, x = h2(3.0)
    r = axtb.singlepoint(z, x, method=method, broken_symmetry=True, gradient=True)
    b = r["broken_symmetry"]
    assert b["broken"] and b["outcome"] == "broken" and b["start"] == "homo-lumo" and 0.97 < b["s2"] < 1.0
    assert r["energy"] == b["unrestricted_energy"] and b["restricted_energy"] - r["energy"] > 0.08
    assert list(r["occupations_alpha"][:2]) == [1.0, 0.0] and list(r["occupations_beta"][:2]) == [1.0, 0.0]
    plain = axtb.singlepoint(z, x, method=method)
    assert "broken_symmetry" not in plain and plain["energy"] == b["restricted_energy"]


@pytest.mark.parametrize("method", METHODS)
def test_calculator_restarts(method):
    z, x = h2(3.0)
    calc = axtb.Calculator(method, z, x, broken_symmetry=True)
    first = calc.singlepoint(x)
    assert first["broken_symmetry"]["start"] == "homo-lumo"
    r = calc.singlepoint(h2(2.9)[1])
    assert r["guess"] == "previous" and r["broken_symmetry"]["start"] == "previous"
    assert r["broken_symmetry"]["restricted_energy"] is None
    # the state moves to another calculator; a restricted one starts from its total density
    other = axtb.Calculator(method, z, x)
    other.set_guess_state(calc.guess_state())
    assert other.singlepoint(x)["guess"] == "previous"


def test_refusals():
    z, x = h2(3.0)
    for kw in [dict(method="gfn2"), dict(method="gfnff"), dict(method="gxtb-v2.0.1", uhf=2),
               dict(method="gxtb-v2.0.1", conventions="gxtb"), dict(method="gxtb-v1.1", charge=1.0)]:
        with pytest.raises(axtb.InputError):
            axtb.singlepoint(z, x, broken_symmetry=True, **kw)
        with pytest.raises(axtb.InputError):
            axtb.Calculator(kw.pop("method"), z, x, broken_symmetry=True, **kw)


@pytest.mark.skipif(not AXTB_BIN.exists(), reason="CLI not built")
def test_cli_equals_python():
    z, x = h2(3.0)
    req = {"numbers": z, "positions": x.tolist(), "method": "gxtb-v2.0.1", "broken_symmetry": True, "gradient": True}
    out = json.loads(subprocess.run([str(AXTB_BIN), "--json"], input=json.dumps(req), capture_output=True, text=True,
                                    check=True).stdout)
    py = axtb.singlepoint(z, x, method="gxtb-v2.0.1", broken_symmetry=True, gradient=True)
    assert out["energy"] == py["energy"] and np.array_equal(out["gradient"], py["gradient"])
    assert out["broken_symmetry"] == py["broken_symmetry"]


qcdata = pytest.importorskip("qcdata")
qccompute = pytest.importorskip("qccompute")


def _structure(d=3.0):
    return qcdata.Structure(symbols=["H", "H"], geometry=h2(d)[1], charge=0, multiplicity=1)


def test_codec():
    from qccodec.exceptions import EncoderError
    from qcdata import CalcType, Model, ProgramInput

    from axtb.qccodec import codec, encode

    inp = ProgramInput(program="axtb", calctype=CalcType.gradient, structure=_structure(),
                       model=Model(method="gxtb-v2.0.1"), keywords={"broken_symmetry": True})
    out = qccompute.compute(inp)
    x = out.results.extras
    assert x["broken_symmetry"]["broken"] and x["broken_symmetry"]["s2"] > 0.97
    assert out.results.wavefunction.scf_occupations_a[0] == 1.0
    # the next step starts from the broken-symmetry wavefunction
    nxt = codec.propagate_wfn(out, inp.model_copy(update={"structure": _structure(2.9)}))
    r2 = qccompute.compute(nxt).results.extras
    assert r2["scf_guess"] == "previous" and r2["broken_symmetry"]["start"] == "previous"
    # ... but not a restricted calculation's
    plain = inp.model_copy(update={"keywords": {"guess": x["wavefunction_id"]}})
    r3 = qccompute.compute(plain).results.extras
    assert r3["scf_guess"] != "previous" and "broken symmetry" in r3["scf_guess_note"]
    assert "broken_symmetry" not in r3
    for method, kw in [("gfn2", {}), ("gxtb-v2.0.1", {"broken_symmetry": "yes"})]:
        bad = ProgramInput(program="axtb", calctype=CalcType.energy, structure=_structure(), model=Model(method=method),
                           keywords={"broken_symmetry": True, **kw})
        with pytest.raises(EncoderError):
            encode(bad)
