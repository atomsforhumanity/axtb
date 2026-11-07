"""Molden files of converged orbitals (Python package, qccodec codec and CLI).

Run with the extension built from this tree and the CLI at ``target/release/axtb`` (or
``$AXTB_BIN``)::

    python -m pytest crates/axtb-py/tests/test_molden.py

The files are read back by the independent reader of ``tests/molden_check.py`` (Molden's
real solid harmonics written out as polynomials, its own Gaussian overlap integrals):

- every file round-trips: the occupied orbitals are orthonormal under the overlap of the
  file's own basis, for GFN2-xTB, g-xTB v1.1 and 2.0.1, closed and open shells, d and f
  shells, spherical and Cartesian functions;
- the orbitals are the g-xTB 1.1.0 program's (its ``molden.input`` files of the reference set,
  ``crates/gxtb/tests/reference``) and, with ``xtb`` on PATH, those of ``xtb --molden`` and
  ``xtb --gxtb --molden`` (the full comparison is ``tests/molden_check.py``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import axtb

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tests"))
import molden_check as mc  # noqa: E402

REF11 = ROOT / "crates/gxtb/tests/reference"
REF2 = ROOT / "crates/gxtb/tests/reference_v2"
AXTB_BIN = Path(os.environ.get("AXTB_BIN", ROOT / "target/release/axtb"))
SYM = {s: i + 1 for i, s in enumerate(mc.SYMBOLS)}


def molecule(xyz: Path):
    lines = xyz.read_text().splitlines()
    nat = int(lines[0])
    rows = [l.split() for l in lines[2:2 + nat]]
    return [SYM[r[0].capitalize()] for r in rows], np.array([[float(x) for x in r[1:4]] for r in rows]) * 1.8897261246257702


# name, xyz, charge, uhf: closed and open shells, d (Cu, Fe) and f (U, Gd) shells
MOLECULES = [
    ("water", ROOT / "tests/structures/water.xyz", 0, 0),
    ("O2-triplet", REF2 / "reference/dioxygen_triplet/mol.xyz", 0, 2),
    ("Cu(NH3)4-2+", ROOT / "tests/structures/cu-nh3-4.xyz", 2, 1),
    ("ferrocene", ROOT / "tests/structures/ferrocene.xyz", 0, 0),
    ("UF6", REF2 / "elements/UF6/mol.xyz", 0, 0),
    ("CeCl3-doublet", REF2 / "elements/CeCl3/mol.xyz", 0, 1),
    ("GdCl3-octet", REF2 / "elements/GdCl3/mol.xyz", 0, 7),
]
METHODS = ["gfn2", "gxtb-v1.1", "gxtb-v2.0.1"]


def cases():
    for method in METHODS:
        for name, xyz, charge, uhf in MOLECULES:
            if method == "gfn2" and name in ("UF6", "CeCl3-doublet", "GdCl3-octet"):
                continue  # GFN2-xTB has no f functions; its d-metal cases are above
            if method == "gxtb-v1.1" and name == "GdCl3-octet":
                continue  # v1.1 has no Gd parameters
            yield pytest.param(method, xyz, charge, uhf, id=f"{method}-{name}")


@pytest.mark.parametrize("method,xyz,charge,uhf", list(cases()))
def test_round_trip(method, xyz, charge, uhf):
    """Occupied orbitals orthonormal in the file's basis; spherical and Cartesian files hold
    the same orbitals; energies and occupations as the result reports them."""
    numbers, pos = molecule(xyz)
    # two stateless single points: the same SCF (a Calculator would start the second from the first)
    r = axtb.singlepoint(numbers, pos, charge=charge, uhf=uhf, method=method, molden=True)
    rc = axtb.singlepoint(numbers, pos, charge=charge, uhf=uhf, method=method, molden="cartesian")
    sph, cart = mc.parse_molden(r["molden"]), mc.parse_molden(rc["molden"])
    sb, cb = mc.Basis(sph), mc.Basis(cart)
    assert mc.orthonormality(sph, sb) < 1e-8
    assert mc.orthonormality(cart, cb) < 1e-8
    assert mc.orthonormality(sph, sb, occupied_only=False) < 1e-8
    assert [z for z, _ in sph["atoms"]] == numbers
    assert np.allclose([p for _, p in sph["atoms"]], pos, atol=1e-12)
    maxl = max(l for _, l, _, _ in sph["shells"])
    assert all(sph["pure"][l] for l in range(2, maxl + 1)) and not any(cart["pure"].values())
    assert ("[5D]" in r["molden"]) == (maxl >= 2) and ("[7F]" in r["molden"]) == (maxl >= 3)
    assert ("[6D]" in rc["molden"]) == (maxl >= 2) and ("[10F]" in rc["molden"]) == (maxl >= 3)
    # one spin set for closed shells, alpha and beta for open shells
    spins = [m["spin"] for m in sph["mos"]]
    assert sorted(set(spins)) == (["alpha", "beta"] if uhf else ["alpha"])
    assert len(sph["mos"]) == sph["nbf"] * (2 if uhf else 1)
    # g-xTB reports alpha then beta orbitals; GFN2-xTB one set of orbitals (spin-summed
    # occupations) that the file writes twice for an open shell
    energies = np.array([m["energy"] for m in sph["mos"]])
    occ = np.array([m["occ"] for m in sph["mos"]])
    n = sph["nbf"]
    ea, eb = r["orbital_energies_alpha"], r["orbital_energies_beta"]
    oa, ob = r["occupations_alpha"], r["occupations_beta"]
    if uhf:
        assert np.allclose(energies[:n], ea, atol=1e-13) and np.allclose(energies[n:], eb, atol=1e-13)
        if method == "gfn2":
            assert np.array_equal(ea, eb)
            assert np.allclose(occ[:n] + occ[n:], oa + ob, atol=1e-9)
            assert abs(occ[:n].sum() - occ[n:].sum() - uhf) < 1e-6
        else:
            assert np.allclose(occ, np.concatenate([oa, ob]), atol=1e-9)
    else:
        assert np.allclose(energies, ea, atol=1e-13)
        assert np.allclose(occ, oa + ob, atol=1e-9)
    pts = mc.points_around(sph, n=300)
    same = mc.compare(cart, cb, sph, sb, pts)
    assert same["nondegenerate"] < 1e-8 and same["degenerate_blocks"] < 1e-8 and same["values"] < 1e-8


@pytest.mark.parametrize("name,uhf", [("water", 0), ("dioxygen_triplet", 2), ("tetrachlorocuprate", 1),
                                      ("nickel_tetracarbonyl", 0), ("iodomethane", 0)])
def test_same_orbitals_as_gxtb_1_1_0(name, uhf):
    """The orbitals of the g-xTB 1.1.0 program's molden.input (reference set), at its geometry
    and with its conventions."""
    d = REF11 / name
    charge = json.loads((d / "input.json").read_text())["charge"]
    prog = mc.parse_molden((d / "molden.input").read_text())
    numbers = [z for z, _ in prog["atoms"]]
    pos = np.array([p for _, p in prog["atoms"]])
    r = axtb.singlepoint(numbers, pos, charge=charge, uhf=uhf, method="gxtb-v1.1", conventions="gxtb", molden=True)
    ours = mc.parse_molden(r["molden"])
    pb, ob = mc.Basis(prog, *mc.PROGRAM_READ["gxtb11"]), mc.Basis(ours)
    assert mc.orthonormality(prog, pb) < 1e-8
    res = mc.compare(ours, ob, prog, pb, mc.points_around(prog, n=500))
    assert res["orbitals_compared"] > 0
    assert res["nondegenerate"] < 1e-8, res
    assert res["degenerate_blocks"] < 1e-8, res
    assert res["occupied_space"] < 1e-8, res
    assert res["values"] < 1e-5, res
    assert res["energy"] < 1e-8, res
    assert res["occupation"] < 1e-12, res


@pytest.mark.skipif(mc.XTB is None, reason=mc.programs.reason(mc.programs.GXTB2) or "needs the g-xTB 2.0.1 build")
@pytest.mark.parametrize("kind,xyz,charge,uhf", [
    ("gfn2", ROOT / "tests/structures/water.xyz", 0, 0),
    ("gfn2", REF2 / "reference/dioxygen_triplet/mol.xyz", 0, 2),
    ("gxtb2", REF2 / "reference/tetrachlorocuprate/mol.xyz", -2, 1),
    ("gxtb2", REF2 / "elements/UF6/mol.xyz", 0, 0),
])
def test_same_orbitals_as_xtb(kind, xyz, charge, uhf, tmp_path):
    """The orbitals of `xtb --molden` (GFN2-xTB) and `xtb --gxtb --molden` (g-xTB 2.0.1)."""
    res = mc.check_case(f"{kind}/{xyz.parent.name}", kind, xyz, charge, uhf, tmp_path)
    assert res["orthonormal_program"] < 1e-8 and res["orthonormal_axtb_spherical"] < 1e-8
    for key in ("vs_program_spherical", "vs_program_cartesian"):
        v = res[key]
        assert v["occupied_space"] < 1e-8, res
        assert v["degenerate_blocks"] < 1e-6, res
        # GFN2: xtb's SCF and axtb's (xtb conventions) agree to its convergence (1e-6 Eh)
        assert v["nondegenerate"] < (1e-6 if kind == "gfn2" else 1e-8), res
        assert v["energy"] < (1e-5 if kind == "gfn2" else 1e-10), res
        assert v["occupation"] < 1e-8, res


def test_arguments():
    pos = np.array([[0, 0, 0.22], [0, 1.43, -0.89], [0, -1.43, -0.89]])
    r = axtb.singlepoint([8, 1, 1], pos)
    assert "molden" not in r
    assert "molden" not in axtb.singlepoint([8, 1, 1], pos, molden=False)
    assert axtb.singlepoint([8, 1, 1], pos, molden="spherical")["molden"] == axtb.singlepoint([8, 1, 1], pos, molden=True)["molden"]
    with pytest.raises(ValueError):
        axtb.singlepoint([8, 1, 1], pos, molden="6d10f")
    with pytest.raises(ValueError):
        axtb.singlepoint([8, 1, 1], pos, molden=3)
    calc = axtb.Calculator("gxtb-v2.0.1", [8, 1, 1], pos)
    assert "molden" not in calc.singlepoint(pos)
    text = calc.singlepoint(pos, molden=True)["molden"]
    assert text.startswith("[Molden Format]\n[Title]\naxtb ") and "gxtb-v2.0.1" in text.splitlines()[2]


def test_qccodec():
    qccompute = pytest.importorskip("qccompute")
    from qccodec.exceptions import EncoderError
    from qcdata import CalcType, Model, ProgramInput, Structure

    structure = Structure(symbols=["O", "H", "H"], geometry=np.array([[0, 0, 0.22], [0, 1.43, -0.89], [0, -1.43, -0.89]]))
    for method in METHODS:
        for kw in ({"molden": True}, {"molden": "cartesian"}):
            for calctype in (CalcType.energy, CalcType.gradient, CalcType.hessian):
                out = qccompute.compute(ProgramInput(program="axtb", calctype=calctype, structure=structure,
                                                     model=Model(method=method), keywords=kw))
                text = out.results.files["molden.input"]
                mol = mc.parse_molden(text)
                assert mc.orthonormality(mol, mc.Basis(mol)) < 1e-8
                if calctype != CalcType.hessian:  # (a Hessian's single point starts from its reference SCF)
                    ref = axtb.singlepoint([8, 1, 1], structure.geometry, method=method, molden=kw["molden"])
                    assert text == ref["molden"]
        out = qccompute.compute(ProgramInput(program="axtb", calctype=CalcType.energy, structure=structure,
                                             model=Model(method=method)))
        assert "molden.input" not in out.results.files
    with pytest.raises(Exception) as err:
        qccompute.compute(ProgramInput(program="axtb", calctype=CalcType.energy, structure=structure,
                                       model=Model(method="gfn2"), keywords={"molden": "yes"}))
    assert isinstance(err.value, EncoderError) or "molden must be" in str(err.value)


@pytest.mark.skipif(not AXTB_BIN.exists(), reason="needs target/release/axtb")
def test_cli(tmp_path):
    xyz = ROOT / "tests/structures/water.xyz"
    for method in METHODS:
        # `--molden` alone: molden.input in the working directory; `--molden FILE` after the
        # structure; the structure after `--molden` when it is the only file
        subprocess.run([str(AXTB_BIN), str(xyz), "--method", method, "--molden"], cwd=tmp_path, check=True,
                       capture_output=True)
        default = (tmp_path / "molden.input").read_text()
        (tmp_path / "molden.input").unlink()
        out = subprocess.run([str(AXTB_BIN), str(xyz), "--method", method, "--molden", "w.molden"], cwd=tmp_path,
                             check=True, capture_output=True, text=True)
        assert json.loads(out.stdout)["molden_file"] == "w.molden"
        assert (tmp_path / "w.molden").read_text() == default
        subprocess.run([str(AXTB_BIN), "--molden", str(xyz), "--method", method], cwd=tmp_path, check=True,
                       capture_output=True)
        assert (tmp_path / "molden.input").read_text() == default
        subprocess.run([str(AXTB_BIN), str(xyz), "--method", method, "--molden-cartesian"], cwd=tmp_path, check=True,
                       capture_output=True)
        # water has s and p shells only, where both kinds of file are the same
        assert (tmp_path / "molden.input").read_text() == default
        # JSON request: the text in the response
        numbers, pos = molecule(xyz)
        req = {"numbers": numbers, "positions": pos.tolist(), "method": method, "molden": True}
        out = subprocess.run([str(AXTB_BIN), "--json"], input=json.dumps(req), capture_output=True, text=True, check=True)
        text = json.loads(out.stdout)["molden"]
        assert text == axtb.singlepoint(numbers, pos, method=method, molden=True)["molden"]
