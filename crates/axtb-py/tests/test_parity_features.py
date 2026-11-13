# SPDX-License-Identifier: LGPL-3.0-or-later
"""SCF guess reuse, Mayer-Wiberg bond orders, orbital levels, HOMO-LUMO gap and the molecular
quadrupole through the Python package, the qccodec codec (incl. ``propagate_wfn``) and the CLI.

    python -m pytest crates/axtb-py/tests/test_parity_features.py

Needs the extension built from this tree and the CLI at ``target/release/axtb`` (or
``$AXTB_BIN``). The comparison with the programs is ``tests/gxtb_parity_features.py``.
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
A = 1.0 / 0.529177210903
# water (tests/structures/water.xyz), bohr
NUMBERS = [1, 1, 8]
POS = A * np.array([[-0.0121648342, 0.0, -0.9323517940], [0.8750977221, 0.0, 0.3219980096],
                    [-0.0543635745, 0.0, 0.0384514415]])
METHODS = ["gfn2", "gxtb-v1.1", "gxtb-v2.0.1"]
KEYS = {"guess", "orbital_energies_alpha", "orbital_energies_beta", "occupations_alpha", "occupations_beta",
        "homo_lumo_gap", "quadrupole"}


@pytest.mark.parametrize("method", METHODS)
def test_singlepoint_properties(method):
    r = axtb.singlepoint(NUMBERS, POS, method=method)
    assert KEYS <= set(r) and "bond_orders" not in r
    assert r["guess"] == ("eeq" if method == "gfn2" else "eeq-bc")
    # closed shell: the same orbitals for both spins, half the occupations each
    assert np.array_equal(r["orbital_energies_alpha"], r["orbital_energies_beta"])
    assert abs(r["occupations_alpha"].sum() - 4.0) < 1e-6 and abs(r["occupations_beta"].sum() - 4.0) < 1e-6
    e = r["orbital_energies_alpha"]
    assert abs(r["homo_lumo_gap"] - (e[4] - e[3])) < 1e-14
    q = r["quadrupole"]
    assert q.shape == (6,) and abs(q[0] + q[2] + q[5]) < 1e-12
    r = axtb.singlepoint(NUMBERS, POS, method=method, bond_orders=True)
    bo = r["bond_orders"]
    assert bo.shape == (3, 3) and np.allclose(bo, bo.T) and np.all(np.diag(bo) == 0.0)
    assert 0.85 < bo[0, 2] < 0.95 and bo[0, 1] < 0.01


def test_open_shell_levels_and_bond_orders():
    o2 = [8, 8], A * np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.21]])
    for method in METHODS:
        r = axtb.singlepoint(*o2, uhf=2, method=method, bond_orders=True)
        assert abs(r["occupations_alpha"].sum() - 7.0) < 1e-6 and abs(r["occupations_beta"].sum() - 5.0) < 1e-6
        if method == "gfn2":  # one set of orbitals
            assert np.array_equal(r["orbital_energies_alpha"], r["orbital_energies_beta"])
        else:
            assert not np.array_equal(r["orbital_energies_alpha"], r["orbital_energies_beta"])
        # Mayer's open-shell bond order 2 sum_s (P^s S)(P^s S): close to 2 for triplet O2
        assert abs(r["bond_orders"][0, 1] - 2.0) < 0.05


@pytest.mark.parametrize("method", ["gxtb-v1.1", "gxtb-v2.0.1"])
def test_guess_reuse(method):
    moved = POS + 0.01
    calc = axtb.Calculator(method, NUMBERS, POS)
    a = calc.singlepoint(POS)
    b = calc.singlepoint(moved, gradient=True)
    assert (a["guess"], b["guess"]) == ("eeq-bc", "previous")
    fresh = axtb.singlepoint(NUMBERS, moved, method=method, gradient=True)
    assert b["iterations"] < fresh["iterations"]
    assert abs(b["energy"] - fresh["energy"]) < 1e-6
    assert np.max(np.abs(b["gradient"] - fresh["gradient"])) < 1e-4
    # program conventions: the program's start every time, unless asked
    calc = axtb.Calculator(method, NUMBERS, POS, conventions="gxtb")
    assert [calc.singlepoint(p)["guess"] for p in (POS, moved)] == ["eeq-bc", "eeq-bc"]
    calc = axtb.Calculator(method, NUMBERS, POS, conventions="gxtb", guess="previous")
    assert [calc.singlepoint(p)["guess"] for p in (POS, POS)] == ["eeq-bc", "previous"]
    calc = axtb.Calculator(method, NUMBERS, POS, guess="eeq-bc")
    assert [calc.singlepoint(p)["guess"] for p in (POS, moved)] == ["eeq-bc", "eeq-bc"]
    # a converged wavefunction handed to another calculator
    src = axtb.Calculator(method, NUMBERS, POS)
    assert src.guess_state() is None
    src.singlepoint(POS)
    state = src.guess_state()
    assert state.method == method and state.natoms == 3
    dst = axtb.Calculator(method, NUMBERS, POS, conventions="gxtb")
    dst.set_guess_state(state)
    assert dst.singlepoint(moved)["guess"] == "previous"
    other = axtb.Calculator("gfn2", NUMBERS, POS)
    with pytest.raises(ValueError, match="cannot start"):
        other.set_guess_state(state)
    calc.reset_guess()


def test_gfn2_guess_option():
    calc = axtb.Calculator("gfn2", NUMBERS, POS)
    assert [calc.singlepoint(p)["guess"] for p in (POS, POS + 0.01)] == ["eeq", "previous"]
    calc = axtb.Calculator("gfn2", NUMBERS, POS, guess="eeq")
    assert [calc.singlepoint(p)["guess"] for p in (POS, POS + 0.01)] == ["eeq", "eeq"]
    with pytest.raises(ValueError, match="guess"):
        axtb.Calculator("gfn2", NUMBERS, POS, guess="huckel")
    with pytest.raises(ValueError, match="guess"):
        axtb.Calculator("gfnff", NUMBERS, POS, guess="previous")


def test_composite_reuses_the_gxtb_part():
    kw = {"alpb": "water", "solvation_correction": "gfn2"}
    calc = axtb.Calculator("gxtb-v2.0.1", NUMBERS, POS, **kw)
    a, b = calc.singlepoint(POS), calc.singlepoint(POS + 0.01)
    assert (a["guess"], b["guess"]) == ("eeq-bc", "previous")
    assert b["composite"]["gxtb_reuse_guess"] and b["composite"]["reuse_guess"]
    calc = axtb.Calculator("gxtb-v2.0.1", NUMBERS, POS, conventions="gxtb", **kw)
    a, b = calc.singlepoint(POS), calc.singlepoint(POS + 0.01)
    assert (a["guess"], b["guess"]) == ("eeq-bc", "eeq-bc")
    assert not b["composite"]["gxtb_reuse_guess"]
    # the state of all three parts moves to another calculator
    src = axtb.Calculator("gxtb-v2.0.1", NUMBERS, POS, **kw)
    src.singlepoint(POS)
    dst = axtb.Calculator("gxtb-v2.0.1", NUMBERS, POS, conventions="gxtb", **kw)
    dst.set_guess_state(src.guess_state())
    assert dst.singlepoint(POS)["guess"] == "previous"


qcdata = pytest.importorskip("qcdata")
qccompute = pytest.importorskip("qccompute")


def _structure():
    return qcdata.Structure(symbols=["H", "H", "O"], geometry=POS, charge=0, multiplicity=1)


@pytest.mark.parametrize("method", METHODS)
def test_codec_fields(method):
    from qcdata import CalcType, Model, ProgramInput

    inp = ProgramInput(program="axtb", calctype=CalcType.gradient, structure=_structure(), model=Model(method=method),
                       keywords={"bond_orders": True})
    out = qccompute.compute(inp)
    r = out.results
    wf = r.wavefunction
    assert len(wf.scf_eigenvalues_a) == r.calcinfo_nbasis == r.calcinfo_nmo
    assert (r.calcinfo_nalpha, r.calcinfo_nbeta, r.calcinfo_natoms) == (4, 4, 3)
    assert abs(sum(wf.scf_occupations_a) - 4.0) < 1e-6
    x = r.extras
    assert x["scf_guess"] in ("eeq", "eeq-bc") and x["homo_lumo_gap"] > 0 and len(x["quadrupole"]) == 6
    assert np.asarray(x["bond_orders"]).shape == (3, 3)
    assert x["wavefunction_id"].startswith("axtb-wfn:")
    # the next input starts from it (qccompute's propagate_wfn hook)
    from axtb.qccodec import codec

    moved = _structure().model_copy(update={"geometry": POS + 0.01})
    nxt = codec.propagate_wfn(out, ProgramInput(program="axtb", calctype=CalcType.gradient, structure=moved,
                                                model=Model(method=method)))
    assert nxt.keywords["guess"] == x["wavefunction_id"]
    r2 = qccompute.compute(nxt).results
    assert r2.extras["scf_guess"] == "previous" and r2.extras["scf_iterations"] < x["scf_iterations"]
    # an unknown wavefunction: the method's own start, and why
    bad = ProgramInput(program="axtb", calctype=CalcType.energy, structure=_structure(), model=Model(method=method),
                       keywords={"guess": "axtb-wfn:0"})
    r3 = qccompute.compute(bad).results
    assert r3.extras["scf_guess"] != "previous" and "not in this process" in r3.extras["scf_guess_note"]


def test_codec_rejects_bad_guess():
    from qccodec.exceptions import EncoderError
    from qcdata import CalcType, Model, ProgramInput

    from axtb.qccodec import encode

    inp = ProgramInput(program="axtb", calctype=CalcType.energy, structure=_structure(), model=Model(method="gfn2"),
                       keywords={"guess": "previous"})
    with pytest.raises(EncoderError, match="guess"):
        encode(inp)


def test_geometric_with_propagate_wfn():
    pytest.importorskip("geometric")
    from qcdata import CalcType, Model, ProgramInput, ProgramSpec

    child = ProgramSpec(program="axtb", calctype=CalcType.gradient, model=Model(method="gxtb-v2.0.1"))
    start = _structure().model_copy(update={"geometry": POS + np.array([[0.1, 0.0, 0.05], [0.0, 0.0, -0.1], [0, 0, 0]])})
    inp = ProgramInput(program="geometric", calctype=CalcType.optimization, structure=start, subprograms=[child])
    out = qccompute.compute(inp, propagate_wfn=True)
    starts = [f.results.extras["scf_guess"] for f in out.results.frames]
    assert starts[0] == "eeq-bc" and set(starts[1:]) == {"previous"}


@pytest.mark.skipif(not AXTB_BIN.exists(), reason="CLI not built")
@pytest.mark.parametrize("method", METHODS)
def test_cli(method):
    req = {"numbers": NUMBERS, "positions": POS.tolist(), "method": method, "bond_orders": True}
    out = json.loads(subprocess.run([str(AXTB_BIN), "--json"], input=json.dumps(req), capture_output=True, text=True,
                                    check=True).stdout)
    assert KEYS <= set(out)
    assert np.asarray(out["bond_orders"]).shape == (3, 3)
    py = axtb.singlepoint(NUMBERS, POS, method=method, bond_orders=True)
    assert np.allclose(out["bond_orders"], py["bond_orders"], atol=1e-12)
    assert np.allclose(out["quadrupole"], py["quadrupole"], atol=1e-12)
