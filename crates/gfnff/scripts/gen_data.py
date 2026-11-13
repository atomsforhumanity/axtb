#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate ``crates/gfnff/src/data.rs`` from the xtb Fortran sources of GFN-FF.

Usage::

    python crates/gfnff/scripts/gen_data.py [XTB_SOURCE_DIR] [OUT_FILE]

``XTB_SOURCE_DIR`` is a checkout of xtb (default ``reference/xtb-gxtb`` of the
workspace; the committed file was generated from commit 97f7bf5 of xtb's gxtb
branch). The tables are parsed from the Fortran, never copied by hand:

* ``src/gfnff/gfnff_param.f90``: the element parameters of the published GFN-FF
  (``*_angewChem2020``), Pauling EN, radii, metal/group/normal-CN tables, repz;
* ``src/gfnff/gfnff_rab.f``: the bond-length estimate tables of ``gfnffrab``
  (checked to be identical to the copy in ``gfnffdrab``);
* ``src/gfnff/gfnff_ini.f90``: ``zeff`` and ``c`` of the charge-scaling function zeta.

xtb is compiled with ``-fdefault-real-8`` (GNU) or ``-r8`` (Intel), so every real
literal without a kind is double precision. The digits are emitted verbatim (only the
kind suffix is dropped and a ``d`` exponent becomes ``e``), so rustc rounds the same
decimal string to the same IEEE double as the Fortran compiler.

Tables that other crates of the workspace already hold are not written again; the script
checks that xtb's copies equal them:

* ``src/param/covalentradd3.f90``: the covalent radii in Angstrom of ``axtb_base::elements``
  (``crates/base/src/elements.rs``);
* ``include/param_ref.fh`` and ``src/param/sqrtzr4r2.f90``: dftd4's reference systems and
  ``<r4>/<r2>`` of ``axtb-d4`` (``crates/d4/src/data.rs``), which xtb's D3-type GFN-FF
  dispersion (``newD3Model``) uses, except where xtb's tables are empty: no reference
  systems for Z = 87-111 and ``<r4>/<r2>`` = 0 for Z = 95-111. The force field applies the
  same gaps to axtb-d4's tables.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

NUM = r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][-+]?\d+)?(?:_wp)?"


def literal(tok: str) -> str:
    """Fortran real literal -> Rust f64 literal with the same decimal digits."""
    tok = tok.strip().replace("_wp", "")
    tok = re.sub(r"[dD]", "e", tok)
    if re.fullmatch(r"[-+]?\d+", tok):
        return tok + ".0"
    if tok.startswith("+"):
        tok = tok[1:]
    if tok.startswith("."):
        tok = "0" + tok
    if tok.startswith("-."):
        tok = "-0" + tok[1:]
    m = re.fullmatch(r"([-]?\d+)(e[-+]?\d+)", tok)
    if m:
        tok = f"{m.group(1)}.0{m.group(2)}"
    if tok.endswith("."):
        tok += "0"
    tok = tok.replace(".e", ".0e")
    return tok


def strip_comments(text: str) -> str:
    return "\n".join(re.sub(r"!.*", "", ln) for ln in text.split("\n"))


def array_after(src: str, pattern: str, n: int, ints: bool = False) -> list[str]:
    """Values of the array constructor that follows ``pattern`` (regex), n elements."""
    m = re.search(pattern, src, re.S)
    if m is None:
        raise KeyError(pattern)
    text = strip_comments(src[m.end():])
    start = min(i for i in (text.find("["), text.find("(/")) if i >= 0)
    end_sq, end_sl = text.find("]", start), text.find("/)", start)
    end = min(i for i in (end_sq, end_sl) if i >= 0)
    body = text[start:end].lstrip("[").lstrip("(/").replace("&", " ")
    toks = re.findall(NUM, body)
    if len(toks) != n:
        raise ValueError(f"{pattern}: {len(toks)} values, expected {n}")
    if ints:
        return [str(int(t)) for t in toks]
    return [literal(t) for t in toks]


def data_statement(src: str, name: str, n: int) -> list[str]:
    """Values of a fixed-form ``data name / ... /`` statement (continuation lines start with '.')."""
    m = re.search(rf"^\s+data {name}\s*/", src, re.M)
    if m is None:
        raise KeyError(name)
    text = src[m.end():]
    body = text[: text.index("/")]
    body = "\n".join(re.sub(r"!.*", "", ln) for ln in body.split("\n"))
    body = re.sub(r"^\s*\.", " ", body, flags=re.M)
    toks = re.findall(NUM, body)
    if len(toks) != n:
        raise ValueError(f"data {name}: {len(toks)} values, expected {n}")
    return [literal(t) for t in toks]


def rust_table(path: Path, name: str) -> list[float]:
    """Values of the table ``name`` in a generated Rust file of the workspace (comments removed)."""
    text = path.read_text()
    m = re.search(rf"\b{name}\s*:[^=]*=\s*\[", text)
    if m is None:
        raise KeyError(f"{name} in {path}")
    body = re.sub(r"//[^\n]*", "", text[m.end():text.index("];", m.end())])
    return [float(t) for t in re.findall(NUM, body)]


def same_values(what: str, xtb: list[str], ours: list[float]) -> None:
    if [float(v) for v in xtb] != ours:
        raise ValueError(f"xtb's {what} differs from the workspace's table")


def check_d4(pref: str, r4r2: list[str], d4: Path) -> None:
    """xtb's copies of dftd4's tables equal axtb-d4's, but for the gaps of xtb's tables."""
    refn, refcn, ascale, hcount, refsys, alphaiw, sscale, secaiw = parse_param_ref(pref)
    d4refn = [int(v) for v in rust_table(d4, "REFN")]
    for z in range(1, 119):
        if refn[z - 1] != (0 if 87 <= z <= 111 else d4refn[z - 1]):
            raise ValueError(f"refn of Z={z}: xtb {refn[z - 1]}, dftd4 {d4refn[z - 1]}")
    for name, xtb, ours in (("refcn", refcn, "REFCN"), ("ascale", ascale, "ASCALE"), ("hcount", hcount, "HCOUNT"),
                            ("refsys", refsys, "REFSYS"), ("alphaiw", alphaiw, "ALPHAIW")):
        flat = rust_table(d4, ours)
        n = len(flat) // (118 * 7)  # values per reference: 1, or 23 frequencies
        for z in range(1, 119):
            for ir in range(refn[z - 1]):
                k = ((z - 1) * 7 + ir) * n
                same_values(f"{name} of Z={z}", xtb[z - 1][ir] if n > 1 else [xtb[z - 1][ir]], flat[k:k + n])
    same_values("sscale", sscale, rust_table(d4, "SSCALE"))
    same_values("secaiw", [v for row in secaiw for v in row], rust_table(d4, "SECAIW"))
    ours = rust_table(d4, "R4_OVER_R2")
    same_values("r4Overr2", r4r2, [0.0 if 95 <= z <= 111 else ours[z - 1] for z in range(1, 119)])


def fmt_array(name: str, ty: str, vals: list[str], per_line: int = 5, doc: str = "") -> str:
    out = []
    if doc:
        out += [f"/// {line}" for line in doc.split("\n")]
    out.append(f"pub const {name}: [{ty}; {len(vals)}] = [")
    for i in range(0, len(vals), per_line):
        out.append("    " + ", ".join(vals[i:i + per_line]) + ",")
    out.append("];")
    return "\n".join(out) + "\n"


def parse_param_ref(text: str):
    """D4 reference data of xtb's include/param_ref.fh (data statements)."""
    refn = [0] * 118
    refcn = [["0.0"] * 7 for _ in range(118)]
    ascale = [["0.0"] * 7 for _ in range(118)]
    hcount = [["0.0"] * 7 for _ in range(118)]
    refsys = [[0] * 7 for _ in range(118)]
    alphaiw = [[["0.0"] * 23 for _ in range(7)] for _ in range(118)]
    sscale = ["0.0"] * 17
    secaiw = [["0.0"] * 23 for _ in range(17)]
    text = re.sub(r"!.*", "", text)
    for m in re.finditer(r"data\s+refn\s*\(\s*(\d+)\s*\)\s*/\s*(\d+)\s*/", text):
        refn[int(m.group(1)) - 1] = int(m.group(2))
    for name, store in (("refcn", refcn), ("ascale", ascale), ("hcount", hcount)):
        for m in re.finditer(rf"data\s+{name}\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*/\s*({NUM})\s*/", text):
            store[int(m.group(2)) - 1][int(m.group(1)) - 1] = literal(m.group(3))
    for m in re.finditer(r"data\s+refsys\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*/\s*(\d+)\s*/", text):
        refsys[int(m.group(2)) - 1][int(m.group(1)) - 1] = int(m.group(3))
    for m in re.finditer(r"data\s+alphaiw\s*\(\s*:\s*,\s*(\d+)\s*,\s*(\d+)\s*\)\s*/(.*?)/", text, re.S):
        toks = re.findall(NUM, m.group(3).replace("&", " "))
        assert len(toks) == 23, m.group(0)[:80]
        alphaiw[int(m.group(2)) - 1][int(m.group(1)) - 1] = [literal(t) for t in toks]
    for m in re.finditer(rf"data\s+sscale\s*\(\s*(\d+)\s*\)\s*/\s*({NUM})\s*/", text):
        sscale[int(m.group(1)) - 1] = literal(m.group(2))
    for m in re.finditer(r"data\s+secaiw\s*\(\s*:\s*,\s*(\d+)\s*\)\s*/(.*?)/", text, re.S):
        toks = re.findall(NUM, m.group(2).replace("&", " "))
        assert len(toks) == 23, m.group(0)[:80]
        secaiw[int(m.group(1)) - 1] = [literal(t) for t in toks]
    return refn, refcn, ascale, hcount, refsys, alphaiw, sscale, secaiw


def main() -> None:
    xtb = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "reference" / "xtb-gxtb"
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else ROOT / "crates" / "gfnff" / "src" / "data.rs"
    param = (xtb / "src/gfnff/gfnff_param.f90").read_text()
    rabf = (xtb / "src/gfnff/gfnff_rab.f").read_text()
    ini = (xtb / "src/gfnff/gfnff_ini.f90").read_text()
    covrad = (xtb / "src/param/covalentradd3.f90").read_text()
    r4r2 = (xtb / "src/param/sqrtzr4r2.f90").read_text()
    pref = (xtb / "include/param_ref.fh").read_text()

    parts = [
        "// SPDX-License-Identifier: LGPL-3.0-or-later\n"
        "// @generated by crates/gfnff/scripts/gen_data.py from xtb (LGPL-3.0-or-later):\n"
        "// src/gfnff/gfnff_param.f90, src/gfnff/gfnff_rab.f, src/gfnff/gfnff_ini.f90.\n"
        "// Do not edit by hand.\n"
        "// Literals close to pi and other constants are table values, not constants.\n"
        "#![allow(clippy::approx_constant)]\n"
    ]

    # element parameters of the published force field (Angew. Chem. 2020)
    docs = {
        "chi": "EEQ electronegativity", "gam": "EEQ chemical hardness", "cnf": "EEQ CN dependence of chi",
        "alp": "EEQ charge width", "bond": "bond-stretch prefactor", "repa": "bonded repulsion exponent",
        "repan": "non-bonded repulsion exponent", "angl": "bending prefactor (centre)",
        "angl2": "bending prefactor (ligands)", "tors": "torsion prefactor (central bond)",
        "tors2": "torsion prefactor (outer atoms)",
    }
    for nm, doc in docs.items():
        vals = array_after(param, rf"parameter :: {nm}_angewChem2020\(103\)\s*=", 103)
        parts.append(fmt_array(nm.upper(), "f64", vals, doc=f"GFN-FF {doc} (`{nm}_angewChem2020`), H-Lr."))

    parts.append(fmt_array("EN", "f64", array_after(param, r"parameter :: en\(1:103\)\s*=", 103),
                           doc="Pauling electronegativities of GFN-FF (`en`; W-Au modified)."))
    parts.append(fmt_array("RAD", "f64", array_after(param, r"parameter :: rad\(1:103\)\s*=", 103),
                           doc="Covalent radii in Angstrom (`rad`, Mantina et al.; used in EEQ topology\n"
                               "distances and the HB/XB terms)."))
    parts.append(fmt_array("METAL", "i32", array_after(param, r"parameter :: metal\(103\)\s*=", 103, ints=True),
                           per_line=20, doc="Metal type (`metal`): 0 non-metal, 1 main-group metal, 2 transition metal."))
    parts.append(fmt_array("GROUP", "i32", array_after(param, r"parameter :: group\(103\)\s*=", 103, ints=True),
                           per_line=20, doc="Group (`group`): main groups 1-8, transition metals negative."))
    parts.append(fmt_array("NORMCN", "i32", array_after(param, r"parameter :: normcn\(103\)\s*=", 103, ints=True),
                           per_line=20, doc="Normal coordination number (`normcn`)."))
    parts.append(fmt_array("REPZ", "f64", array_after(param, r"parameter :: repz\(103\)\s*=", 103),
                           per_line=10, doc="Effective nuclear charge of the repulsion (`repz`)."))

    # bond length estimates; gfnffrab and gfnffdrab carry the same tables
    rab_split = rabf.index("subroutine gfnffrab(")
    for nm, doc in (("en", "EN"), ("r0", "base radius"), ("cnfak", "linear CN dependence")):
        a = data_statement(rabf[:rab_split], nm, 103)
        b = data_statement(rabf[rab_split:], nm, 103)
        if a != b:
            raise ValueError(f"gfnffrab and gfnffdrab differ in {nm}")
        parts.append(fmt_array(f"RAB_{nm.upper()}", "f64", a,
                               doc=f"Bond-length estimate `gfnffrab`: {doc} (`{nm}`)."))

    # charge scaling function zeta (D4 type, qref = 0)
    zeta_src = ini[ini.index("pure elemental function zeta"):]
    parts.append(fmt_array("ZETA_ZEFF", "f64", [literal(v) for v in array_after(zeta_src, r"parameter :: zeff\(103\)\s*=", 103, ints=True)],
                           per_line=10, doc="Effective nuclear charges of the zeta function (`zeff`)."))
    parts.append(fmt_array("ZETA_C", "f64", array_after(zeta_src, r"parameter :: c\(1:103\)\s*=", 103),
                           doc="Global hardness of the zeta function (`c`, Ghosh and Islam, x2)."))

    # xtb's D3 radii in Angstrom (before `* aatoau * 4/3`) are mctc-lib's, in axtb-base
    same_values("covalentRadD3", array_after(covrad, r"parameter :: covalentRadD3\(1:118\)\s*=", 118),
                rust_table(ROOT / "crates/base/src/elements.rs", "COVALENT_RAD_2009_AA"))
    # xtb's D4 reference data and r4/r2 are dftd4's, in axtb-d4, with gaps
    check_d4(pref, array_after(r4r2, r"parameter :: r4Overr2\(max_elem\)\s*=", 118), ROOT / "crates/d4/src/data.rs")

    out.write_text("\n".join(parts))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
