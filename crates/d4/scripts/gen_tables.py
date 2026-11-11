#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate ``src/data.rs`` from the dftd4 / mctc-lib Fortran sources.

The tables are parsed from the Fortran (never hand copied).  Every real literal
is emitted with exactly the decimal digits found in the Fortran source, so the
Rust compiler rounds it to the same IEEE double as gfortran does.

Usage (from anywhere)::

    python crates/d4/scripts/gen_tables.py [--dftd4 DIR] [--mctc DIR] [--base FILE] [--out FILE]

Defaults point to ``reference/dftd4`` and ``reference/mctc-lib`` of the
workspace.  The sources used for the committed file are dftd4 v4.3.1
(commit 1107ea7c, identical to upstream main at the time) and mctc-lib
v0.7.0 data modules.

The covalent radii and Pauling electronegativities are not written again: the
crate re-exports them from ``axtb_base::elements`` (``crates/base/src/elements.rs``,
generated from mctc-lib by ``crates/axtb/scripts/gen_data.py``), and this script
checks that dftd4's ``data/covrad.f90`` and ``data/en.f90`` hold the same values.
"""
from __future__ import annotations

import argparse
import math
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]

MAX_ELEM = 118
MAX_REF = 7
NFREQ = 23
NSEC = 17

REAL_RE = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eEdD][-+]?\d+)?(_wp)?$")


def rust_real(tok: str) -> str:
    """Convert a Fortran real literal into an equivalent Rust f64 literal.

    The digits are kept verbatim (only the kind suffix is dropped and a ``d``
    exponent is turned into ``e``) so that both compilers round the same
    decimal string."""
    tok = tok.strip()
    if not REAL_RE.match(tok):
        raise ValueError(f"not a real literal: {tok!r}")
    tok = tok.replace("_wp", "")
    tok = tok.replace("d", "e").replace("D", "e")
    orig = tok
    sign = ""
    if tok[0] in "+-":
        sign, tok = ("-" if tok[0] == "-" else ""), tok[1:]
    mant, _, exp = tok.partition("e")
    if mant.startswith("."):
        mant = "0" + mant
    if "." not in mant:
        mant = mant + ".0"
    elif mant.endswith("."):
        mant = mant + "0"
    out = sign + mant + (("e" + exp) if exp else "")
    # sanity: python parses both identically (correctly rounded decimal)
    assert float(out) == float(orig) and str(float(out))[0] == str(float(orig))[0], (orig, out)
    return out


def strip_comments(text: str) -> str:
    return "\n".join(line.split("!", 1)[0] for line in text.splitlines())


def parse_array_constructor(path: Path, name: str) -> list[str]:
    """Return the literal tokens of ``name(...) = [ ... ]`` (optionally scaled)."""
    text = strip_comments(path.read_text())
    m = re.search(rf"\b{name}\s*\(\s*max_elem\s*\)\s*=", text)
    if not m:
        raise RuntimeError(f"{name} not found in {path}")
    start = text.index("[", m.end())
    end = text.index("]", start)
    body = text[start + 1 : end].replace("&", " ")
    toks = [t.strip() for t in body.split(",") if t.strip()]
    if len(toks) != MAX_ELEM:
        raise RuntimeError(f"{name}: expected {MAX_ELEM} values, got {len(toks)}")
    return toks


def parse_scalar_parameter(path: Path, name: str) -> str:
    text = strip_comments(path.read_text())
    m = re.search(rf"\b{name}\s*=\s*([-+0-9.eEdD_wp]+)", text, re.IGNORECASE)
    if not m:
        raise RuntimeError(f"{name} not found in {path}")
    return m.group(1)


def parse_data_statements(path: Path):
    """Parse all ``data name(idx) / values /`` statements of reference.inc."""
    raw = path.read_text().splitlines()
    # join continuation lines: a line ending in '&' continues on the next one,
    # which itself starts with '&'
    lines: list[str] = []
    buf = ""
    for line in raw:
        line = line.split("!", 1)[0].rstrip()
        if not line.strip():
            continue
        stripped = line.strip()
        if buf:
            if stripped.startswith("&"):
                stripped = stripped[1:]
            buf += " " + stripped
        else:
            buf = stripped
        if buf.endswith("&"):
            buf = buf[:-1]
            continue
        lines.append(buf)
        buf = ""
    if buf:
        lines.append(buf)

    stmt_re = re.compile(r"^data\s+(\w+)\s*\(([^)]*)\)\s*/(.*)/$", re.IGNORECASE)
    out = []
    for line in lines:
        for stmt in line.split(";"):
            stmt = stmt.strip()
            if not stmt:
                continue
            m = stmt_re.match(stmt)
            if not m:
                raise RuntimeError(f"cannot parse data statement: {stmt!r}")
            name = m.group(1).lower()
            idx = [s.strip() for s in m.group(2).split(",")]
            vals = [v.strip() for v in m.group(3).split(",") if v.strip()]
            out.append((name, idx, vals))
    return out


def rust_table(path: Path, name: str) -> list[float]:
    """Values of the table ``name`` in a generated Rust file (comments removed)."""
    text = path.read_text()
    m = re.search(rf"\b{name}\s*:[^=]*=\s*\[", text)
    if not m:
        raise RuntimeError(f"{name} not found in {path}")
    body = re.sub(r"//[^\n]*", "", text[m.end() : text.index("];", m.end())])
    return [float(t) for t in re.findall(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", body)]


def fmt_list(vals, per_line=6, indent="    "):
    lines = []
    for i in range(0, len(vals), per_line):
        lines.append(indent + ", ".join(vals[i : i + per_line]) + ",")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dftd4", type=Path, default=ROOT / "reference" / "dftd4")
    ap.add_argument("--mctc", type=Path, default=ROOT / "reference" / "mctc-lib")
    ap.add_argument("--base", type=Path, default=ROOT / "crates" / "base" / "src" / "elements.rs")
    ap.add_argument("--out", type=Path, default=HERE.parent / "src" / "data.rs")
    args = ap.parse_args()

    d4src = args.dftd4 / "src" / "dftd4"
    ref = parse_data_statements(d4src / "reference.inc")

    zero_ref = lambda: [["0.0"] * MAX_REF for _ in range(MAX_ELEM)]
    tabs = {
        name: zero_ref()
        for name in ("refq", "refh", "clsq", "clsh", "hcount", "ascale", "refcovcn", "refcn")
    }
    refsys = [[0] * MAX_REF for _ in range(MAX_ELEM)]
    refn = [0] * MAX_ELEM
    alphaiw = [[["0.0"] * NFREQ for _ in range(MAX_REF)] for _ in range(MAX_ELEM)]
    sec = {name: ["0.0"] * NSEC for name in ("secq", "sscale", "seccn")}
    secaiw = [["0.0"] * NFREQ for _ in range(NSEC)]
    seen_refn = set()

    for name, idx, vals in ref:
        if name in tabs:
            ir, iz = int(idx[0]), int(idx[1])
            assert len(vals) == 1
            tabs[name][iz - 1][ir - 1] = rust_real(vals[0])
        elif name == "refsys":
            ir, iz = int(idx[0]), int(idx[1])
            refsys[iz - 1][ir - 1] = int(vals[0])
        elif name == "refn":
            iz = int(idx[0])
            refn[iz - 1] = int(vals[0])
            seen_refn.add(iz)
        elif name == "alphaiw":
            assert idx[0] == ":"
            ir, iz = int(idx[1]), int(idx[2])
            assert len(vals) == NFREQ, (name, idx, len(vals))
            alphaiw[iz - 1][ir - 1] = [rust_real(v) for v in vals]
        elif name in sec:
            sec[name][int(idx[0]) - 1] = rust_real(vals[0])
        elif name == "secaiw":
            assert idx[0] == ":"
            assert len(vals) == NFREQ
            secaiw[int(idx[1]) - 1] = [rust_real(v) for v in vals]
        elif name in ("seccnd3", "dftq", "dfth", "pbcq", "pbch", "gffq", "gffh", "eeqbcq", "eeqbch"):
            pass  # not used by the EEQ / GFN2 charge models
        else:
            raise RuntimeError(f"unexpected data statement {name}")

    # every referenced secondary system must be defined
    for iz in range(MAX_ELEM):
        for ir in range(refn[iz]):
            s = refsys[iz][ir]
            if s and all(v == "0.0" for v in secaiw[s - 1]):
                raise RuntimeError(f"secondary system {s} used by Z={iz+1} but undefined")
            if s > NSEC:
                raise RuntimeError("refsys out of range")

    data = d4src / "data"
    covrad = [rust_real(t) for t in parse_array_constructor(data / "covrad.f90", "covalent_rad_2009")]
    en = [rust_real(t) for t in parse_array_constructor(data / "en.f90", "pauling_en")]
    hard = [rust_real(t) for t in parse_array_constructor(data / "hardness.f90", "chemical_hardness")]
    zeff = [rust_real(t) for t in parse_array_constructor(data / "zeff.f90", "effective_nuclear_charge")]
    r4r2 = [rust_real(t) for t in parse_array_constructor(data / "r4r2.f90", "r4_over_r2")]

    # one copy of these element tables, in axtb-base (from mctc-lib, which dftd4 copied)
    for name, base_name, vals in (("covalent_rad_2009", "COVALENT_RAD_2009_AA", covrad),
                                  ("pauling_en", "PAULING_EN", en)):
        if [float(v) for v in vals] != rust_table(args.base, base_name):
            raise RuntimeError(f"dftd4's {name} differs from axtb_base::elements::{base_name}")

    codata = args.mctc / "src" / "mctc" / "io" / "codata2018.f90"
    consts = args.mctc / "src" / "mctc" / "io" / "constants.f90"
    planck = rust_real(parse_scalar_parameter(codata, "Planck_constant"))
    me = rust_real(parse_scalar_parameter(codata, "electron_mass"))
    clight = rust_real(parse_scalar_parameter(codata, "speed_of_light_in_vacuum"))
    alpha = rust_real(parse_scalar_parameter(codata, "fine_structure_constant"))
    pi = rust_real(parse_scalar_parameter(consts, "pi"))
    assert float(pi) == math.pi, pi

    def git_rev(path: Path) -> str:
        try:
            return subprocess.check_output(
                ["git", "-C", str(path), "rev-parse", "--short=12", "HEAD"], text=True
            ).strip()
        except Exception:
            return "unknown"

    o = []
    w = o.append
    w("// SPDX-License-Identifier: LGPL-3.0-or-later")
    w("// @generated by crates/d4/scripts/gen_tables.py -- do not edit by hand.")
    w(f"// Source: dftd4 src/dftd4/{{reference.inc,data/*.f90}} (git {git_rev(args.dftd4)}),")
    w(f"//         mctc-lib src/mctc/io/{{codata2018,constants}}.f90 (git {git_rev(args.mctc)}).")
    w("// Literals carry exactly the decimal digits of the Fortran source.")
    w("#![allow(clippy::excessive_precision, clippy::unreadable_literal)]")
    w("")
    w(f"pub const MAX_ELEM: usize = {MAX_ELEM};")
    w(f"pub const MAX_REF: usize = {MAX_REF};")
    w(f"pub const NFREQ: usize = {NFREQ};")
    w(f"pub const NSEC: usize = {NSEC};")
    w("")
    w("/// CODATA 2018 Planck constant (J s), mctc-lib")
    w(f"pub const PLANCK: f64 = {planck};")
    w("/// CODATA 2018 electron mass (kg), mctc-lib")
    w(f"pub const ELECTRON_MASS: f64 = {me};")
    w("/// CODATA 2018 speed of light (m/s), mctc-lib")
    w(f"pub const SPEED_OF_LIGHT: f64 = {clight};")
    w("/// CODATA 2018 fine structure constant, mctc-lib")
    w(f"pub const FINE_STRUCTURE: f64 = {alpha};")
    w(f"/// pi (mctc-lib writes {pi}, which is this value)")
    w("pub const PI: f64 = std::f64::consts::PI;")
    w("")

    def arr1(name, doc, vals, ty="f64"):
        w(f"/// {doc}")
        w(f"pub static {name}: [{ty}; {len(vals)}] = [")
        w(fmt_list([str(v) for v in vals]))
        w("];")
        w("")

    w("/// Covalent radii (Angstrom), Pyykko & Atsumi 2009 (dftd4 data/covrad.f90, the values of")
    w("/// mctc-lib's; one copy in `axtb_base::elements`)")
    w("pub use axtb_base::elements::COVALENT_RAD_2009_AA as COVALENT_RAD_2009;")
    w("/// Pauling electronegativities (dftd4 data/en.f90, the values of mctc-lib's; one copy in")
    w("/// `axtb_base::elements`)")
    w("pub use axtb_base::elements::PAULING_EN;")
    w("")
    arr1("CHEMICAL_HARDNESS", "Chemical hardnesses (dftd4 data/hardness.f90)", hard)
    arr1("EFFECTIVE_NUCLEAR_CHARGE", "Effective nuclear charges (dftd4 data/zeff.f90)", zeff)
    arr1("R4_OVER_R2", "`<r4>/<r2>` expectation values (dftd4 data/r4r2.f90)", r4r2)
    arr1("REFN", "Number of reference systems per element (reference.inc refn)", refn, "u8")

    def arr2(name, doc, table, ty="f64"):
        w(f"/// {doc}")
        w(f"pub static {name}: [[{ty}; MAX_REF]; MAX_ELEM] = [")
        for iz, row in enumerate(table):
            w(f"    [{', '.join(str(v) for v in row)}], // Z={iz+1}")
        w("];")
        w("")

    arr2("REFQ_GFN2", "Reference charges, GFN2-xTB (reference.inc refq)", tabs["refq"])
    arr2("REFH_GFN2", "Reference hydrogen charges, GFN2-xTB (reference.inc refh)", tabs["refh"])
    arr2("REFQ_EEQ", "Reference charges, EEQ (reference.inc clsq)", tabs["clsq"])
    arr2("REFH_EEQ", "Reference hydrogen charges, EEQ (reference.inc clsh)", tabs["clsh"])
    arr2("HCOUNT", "Number of hydrogen atoms in reference systems (reference.inc hcount)", tabs["hcount"])
    arr2("ASCALE", "Polarizability scaling of reference systems (reference.inc ascale)", tabs["ascale"])
    arr2("REFCOVCN", "Reference coordination numbers, D4 counting (reference.inc refcovcn)", tabs["refcovcn"])
    arr2("REFCN", "Reference coordination numbers, D3 counting (reference.inc refcn)", tabs["refcn"])
    arr2("REFSYS", "Secondary reference system index (reference.inc refsys)", refsys, "u8")

    w("/// Reference dynamic polarizabilities at 23 imaginary frequencies (reference.inc alphaiw)")
    w("pub static ALPHAIW: [[[f64; NFREQ]; MAX_REF]; MAX_ELEM] = [")
    for iz in range(MAX_ELEM):
        w(f"    [ // Z={iz+1}")
        for ir in range(MAX_REF):
            w(f"        [{', '.join(alphaiw[iz][ir])}],")
        w("    ],")
    w("];")
    w("")
    arr1("SECQ", "Secondary system charges (reference.inc secq)", sec["secq"])
    arr1("SSCALE", "Secondary system scaling (reference.inc sscale)", sec["sscale"])
    arr1("SECCN", "Secondary system CN (reference.inc seccn)", sec["seccn"])
    w("/// Secondary system dynamic polarizabilities (reference.inc secaiw)")
    w("pub static SECAIW: [[f64; NFREQ]; NSEC] = [")
    for s in range(NSEC):
        w(f"    [{', '.join(secaiw[s])}],")
    w("];")
    w("")

    args.out.write_text("\n".join(o))
    print(f"wrote {args.out} ({sum(refn)} reference systems)")


if __name__ == "__main__":
    main()
