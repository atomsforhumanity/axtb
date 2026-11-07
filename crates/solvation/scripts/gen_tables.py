#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate the data tables of ``axtb-solvation`` from the xtb Fortran sources.

Everything is parsed from the Fortran (never hand copied):

* ``include/param_alpb_*.fh`` / ``include/param_gbsa_*.fh``: per-solvent
  ALPB/GBSA parameters (only the GFN2-xTB sets are emitted),
* ``src/solv/model.f90``: the solvent-name normalisation (``normalizeSolventName``)
  and the solvent -> parameter-set selection for GFN2 (``loadInternalParam``),
* ``src/param/vdwradd3.f90``: D3 van-der-Waals radii,
* ``src/solv/lebedev.f90``: the Lebedev-Laikov grid generator sequences,
* a handful of scalar constants (unit conversions, model constants).

Every real literal is emitted with exactly the decimal digits of the Fortran
source (only the kind suffix is dropped and a ``d`` exponent becomes ``e``), so
rustc rounds it to the same IEEE double as gfortran.

Usage (from anywhere)::

    python3 crates/solvation/scripts/gen_tables.py [--xtb DIR]

Writes ``src/params.rs`` and ``src/lebedev_data.rs`` of the crate.  The default
source tree is ``reference/xtb-gxtb`` of the workspace (xtb 6.7.1).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CRATE = HERE.parent
ROOT = HERE.parents[2]

NELEM = 94

REAL_RE = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eEdD][-+]?\d+)?(_wp)?$")
REAL_IN_TEXT = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eEdD][-+]?\d+)?(?:_wp)?"


def rust_real(tok: str) -> str:
    """Convert a Fortran real literal into an equivalent Rust f64 literal."""
    tok = tok.strip()
    if not REAL_RE.match(tok):
        raise ValueError(f"not a real literal: {tok!r}")
    tok = tok.replace("_wp", "").replace("d", "e").replace("D", "e")
    mant, _, exp = tok.partition("e")
    sign = ""
    if mant[0] in "+-":
        sign, mant = ("-" if mant[0] == "-" else ""), mant[1:]
    if mant.startswith("."):
        mant = "0" + mant
    if mant.endswith("."):
        mant += "0"
    if "." not in mant:
        mant += ".0"
    return sign + mant + (("e" + exp) if exp else "")


def strip_comments(text: str) -> str:
    out = []
    for line in text.splitlines():
        # no string literal in the parsed regions contains '!'
        i = line.find("!")
        out.append(line if i < 0 else line[:i])
    return "\n".join(out)


def join_continuations(text: str) -> str:
    """Join Fortran free-form continuation lines (trailing/leading '&')."""
    text = re.sub(r"&\s*\n\s*&?", " ", text)
    return text


# --------------------------------------------------------------------------
# solvent parameters
# --------------------------------------------------------------------------

SCALARS = ["epsv", "smass", "rhos", "c1", "rprobe", "gshift", "soset", "alpha"]


def parse_param_sets(include_dir: Path) -> dict[str, dict]:
    sets: dict[str, dict] = {}
    files = sorted(include_dir.glob("param_alpb_*.fh")) + sorted(include_dir.glob("param_gbsa_*.fh"))
    head = re.compile(r"type\(gbsa_parameter\)\s*,\s*parameter\s*::\s*(\w+)\s*=\s*gbsa_parameter\s*\(", re.I)
    for f in files:
        text = strip_comments(f.read_text())
        for m in head.finditer(text):
            name = m.group(1).lower()
            # find the matching closing parenthesis
            depth, i = 1, m.end()
            while depth:
                c = text[i]
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                i += 1
            body = text[m.end(): i - 1].replace("&", " ")
            arrays = re.findall(r"\(/(.*?)/\)", body, re.S)
            scal_text = re.sub(r"\(/(.*?)/\)", "", body, flags=re.S)
            scal = [t.strip() for t in scal_text.split(",") if t.strip()]
            if len(scal) != len(SCALARS) or len(arrays) != 3:
                raise SystemExit(f"{f.name}:{name}: unexpected layout ({len(scal)} scalars, {len(arrays)} arrays)")
            entry = {"file": f.name}
            for k, tok in zip(SCALARS, scal):
                entry[k] = rust_real(tok)
            for k, arr in zip(["gamscale", "sx", "tmp"], arrays):
                vals = [rust_real(t) for t in arr.replace("\n", " ").split(",") if t.strip()]
                if len(vals) != NELEM:
                    raise SystemExit(f"{f.name}:{name}:{k}: {len(vals)} values, expected {NELEM}")
                entry[k] = vals
            if name in sets:
                raise SystemExit(f"duplicate parameter set {name}")
            sets[name] = entry
    return sets


def find_block(text: str, start_pat: str, end_pat: str, start: int = 0) -> tuple[str, int]:
    m = re.compile(start_pat, re.I | re.S).search(text, start)
    if not m:
        raise SystemExit(f"pattern not found: {start_pat}")
    e = re.compile(end_pat, re.I | re.S).search(text, m.end())
    if not e:
        raise SystemExit(f"end pattern not found: {end_pat}")
    return text[m.end(): e.start()], e.start()


def parse_case_list(s: str) -> list[str]:
    return [t.strip().strip("'\"").lower() for t in s.split(",") if t.strip()]


def parse_model(model_f90: Path) -> tuple[list[tuple[str, str]], list[tuple[str, str]], list[tuple[str, str]]]:
    text = join_continuations(strip_comments(model_f90.read_text()))
    # normalizeSolventName
    body, _ = find_block(text, r"subroutine\s+normalizeSolventName", r"end\s+subroutine\s+normalizeSolventName")
    norm: list[tuple[str, str]] = []
    for m in re.finditer(r"case\s*\(([^)]*)\)\s*solvent\s*=\s*'([^']*)'", body, re.I):
        target = m.group(2).lower()
        for alias in parse_case_list(m.group(1)):
            norm.append((alias, target))
    if not norm:
        raise SystemExit("no solvent aliases found in normalizeSolventName")

    load, _ = find_block(text, r"subroutine\s+loadInternalParam", r"end\s+subroutine\s+loadInternalParam")
    case_re = re.compile(r"case\s*\(([^)]*)\)\s*;?\s*param\s*=\s*(\w+)", re.I)

    # GBSA (Still kernel), GFN2 branch: "if (any(level == [2, 0])) then" ... "else"
    still, _ = find_block(load, r"case\s*\(\s*gbKernel%still\s*\)", r"case\s*\(\s*gbKernel%p16\s*\)")
    gfn2_still, _ = find_block(still, r"if\s*\(\s*any\s*\(\s*level\s*==\s*\[\s*2\s*,\s*0\s*\]\s*\)\s*\)\s*then", r"\belse\b")
    gbsa = [(a, m.group(2).lower()) for m in case_re.finditer(gfn2_still) for a in parse_case_list(m.group(1))]

    # ALPB (P16 kernel), GFN2 branch: "if (level == 2) then" ... "elseif (level == 0)"
    p16, _ = find_block(load, r"case\s*\(\s*gbKernel%p16\s*\)", r"end\s+select\s*\n\s*call\s+paramToModel")
    gfn2_p16, _ = find_block(p16, r"if\s*\(\s*level\s*==\s*2\s*\)\s*then", r"else\s*if\s*\(\s*level\s*==\s*0\s*\)")
    alpb = [(a, m.group(2).lower()) for m in case_re.finditer(gfn2_p16) for a in parse_case_list(m.group(1))]
    if not gbsa or not alpb:
        raise SystemExit("could not parse the GFN2 solvent selection tables")
    return norm, gbsa, alpb


# --------------------------------------------------------------------------
# van-der-Waals radii
# --------------------------------------------------------------------------

def parse_vdw_d3(path: Path) -> list[str]:
    text = join_continuations(strip_comments(path.read_text()))
    m = re.search(r"vanDerWaalsRadD3\s*\(\s*1\s*:\s*94\s*\)\s*=\s*aatoau\s*\*\s*\[(.*?)\]", text, re.S)
    if not m:
        raise SystemExit("vanDerWaalsRadD3 not found")
    vals = [rust_real(t) for t in m.group(1).split(",") if t.strip()]
    if len(vals) != NELEM:
        raise SystemExit(f"vanDerWaalsRadD3: {len(vals)} values")
    return vals


# --------------------------------------------------------------------------
# Lebedev grids
# --------------------------------------------------------------------------

def parse_lebedev(path: Path) -> tuple[list[int], dict[int, list[tuple]]]:
    text = strip_comments(path.read_text())
    m = re.search(r"gridSize\s*\(\s*32\s*\)\s*=\s*\[(.*?)\]", join_continuations(text), re.S)
    sizes = [int(t) for t in m.group(1).split(",")]
    grids: dict[int, list[tuple]] = {}
    for sm in re.finditer(r"subroutine\s+ld(\d{4})\s*\(x,\s*w,\s*n\)(.*?)end\s+subroutine", text, re.S | re.I):
        npts = int(sm.group(1))
        cur = {"a": None, "b": None, "v": None}
        seq = []
        for stmt in sm.group(2).splitlines():
            stmt = stmt.strip()
            am = re.match(rf"^([abv])\s*=\s*({REAL_IN_TEXT})$", stmt)
            if am:
                cur[am.group(1)] = rust_real(am.group(2))
                continue
            cm = re.match(r"^call\s+genOh(\d)\s*\(\s*n\s*,\s*x\(1,\s*n\)\s*,\s*w\(n\)\s*,(.*)\)$", stmt, re.I)
            if cm:
                kind = int(cm.group(1))
                args = [t.strip() for t in cm.group(2).split(",")]
                expect = {1: ["v"], 2: ["v"], 3: ["v"], 4: ["a", "v"], 5: ["a", "v"], 6: ["a", "b", "v"]}[kind]
                if args != expect:
                    raise SystemExit(f"ld{npts:04d}: unexpected arguments {args} for genOh{kind}")
                seq.append((kind,) + tuple(cur[k] for k in expect))
        count = sum({1: 6, 2: 12, 3: 8, 4: 24, 5: 24, 6: 48}[g[0]] for g in seq)
        if count != npts:
            raise SystemExit(f"ld{npts:04d}: generators give {count} points")
        grids[npts] = seq
    if sorted(grids) != sorted(sizes):
        raise SystemExit("grid list and generator subroutines disagree")
    return sizes, grids


# --------------------------------------------------------------------------
# scalar constants
# --------------------------------------------------------------------------

def grab(path: Path, name: str, after: str = "") -> str:
    text = strip_comments(path.read_text())
    pat = re.compile(rf"\b{name}\s*=\s*({REAL_IN_TEXT}){after}", re.I)
    m = pat.search(text)
    if not m:
        raise SystemExit(f"constant {name} not found in {path}")
    return rust_real(m.group(1))


def gen_params(xtb: Path, out: Path) -> None:
    sets = parse_param_sets(xtb / "include")
    norm, gbsa_sel, alpb_sel = parse_model(xtb / "src/solv/model.f90")
    vdw = parse_vdw_d3(xtb / "src/param/vdwradd3.f90")

    conv = xtb / "src/mctc/convert.f90"
    const = xtb / "src/mctc/mctc_constants.f90"
    model = xtb / "src/solv/model.f90"
    state = xtb / "src/solv/state.f90"
    gbsa = xtb / "src/solv/gbsa.f90"
    born = xtb / "src/solv/born.f90"
    sasa = xtb / "src/solv/sasa.f90"
    kernel = xtb / "src/solv/kernel.f90"
    inp = xtb / "src/solv/input.F90"
    consts = [
        ("AUTOAA", grab(conv, "autoaa"), "Bohr -> Angstrom (xtb mctc convert.f90, not CODATA)"),
        ("AUTOKCAL", grab(conv, "autokcal"), "Hartree -> kcal/mol (xtb mctc convert.f90)"),
        ("PI", grab(const, "pi"), "pi (xtb mctc_constants.f90)"),
        ("KB", grab(const, "kB"), "Boltzmann constant in Eh/K (xtb mctc_constants.f90)"),
        ("MOLCM3TOAU", grab(model, "molcm3toau"), "g/cm^3 / (g/mol) -> au density conversion (model.f90)"),
        ("SURFACE_TENSION", grab(model, "surfaceTension"), "surface tension unit (model.f90)"),
        ("LRCUT_AA", grab(model, "lrcut", r"\s*\*\s*aatoau"), "Born radii integration cutoff in Angstrom (model.f90)"),
        ("SRCUT_AA", grab(model, "srcut", r"\s*\*\s*aatoau"), "SASA neighbour-list offset in Angstrom (model.f90)"),
        ("REF_DENSITY", grab(state, "refDensity"), "reference density (state.f90)"),
        ("REF_MOLECULAR_MASS", grab(state, "refMolecularMass"), "reference molecular mass (state.f90)"),
        ("IDEAL_GAS_MOL_VOLUME", grab(state, "idealGasMolVolume"), "ideal gas molar volume in L (state.f90)"),
        ("AMBIENT_TEMPERATURE", grab(state, "ambientTemperature"), "ambient temperature in K (state.f90)"),
        ("DEFAULT_TEMPERATURE", grab(inp, "temperature"), "default temperature in K (input.F90)"),
        ("ALPB_ALPHA", grab(gbsa, r"self%alpbet", r"\s*/"), "ALPB alpha constant (gbsa.f90)"),
        ("SMOOTH_W_AA", grab(gbsa, "w", r"\s*\*\s*aatoau"), "SASA smoothing width in Angstrom (gbsa.f90/sasa.f90)"),
        ("TOLSESP", grab(sasa, "tolsesp"), "minimum surface function of a grid point (sasa.f90)"),
        ("ZETA_P16", grab(kernel, "zetaP16"), "P16 kernel zeta (kernel.f90)"),
        ("OBC_ALPHA", grab(born, "alp"), "GBOBC-II alpha (born.f90)"),
        ("OBC_BETA", grab(born, "bet"), "GBOBC-II beta (born.f90)"),
        ("OBC_GAMMA", grab(born, "gam"), "GBOBC-II gamma (born.f90)"),
    ]
    # cross-check the smoothing width of sasa.f90 against gbsa.f90
    if grab(sasa, "w", r"\s*\*\s*aatoau") != dict((c[0], c[1]) for c in consts)["SMOOTH_W_AA"]:
        raise SystemExit("smoothing width differs between gbsa.f90 and sasa.f90")
    default_nang = int(re.search(r"nAng\s*=\s*(\d+)", (xtb / "src/solv/input.F90").read_text()).group(1))

    # which parameter sets are referenced by the GFN2 selection tables
    def used(sel: list[tuple[str, str]]) -> list[str]:
        names: list[str] = []
        for _, p in sel:
            if p not in sets:
                raise SystemExit(f"parameter set {p} referenced in model.f90 but not found")
            if p not in names:
                names.append(p)
        return names

    alpb_sets = used(alpb_sel)
    gbsa_sets = used(gbsa_sel)

    lines: list[str] = []
    w = lines.append
    w("// SPDX-License-Identifier: LGPL-3.0-or-later")
    w("// @generated by crates/solvation/scripts/gen_tables.py -- do not edit by hand.")
    w("// Source: xtb 6.7.1 include/param_{alpb,gbsa}_*.fh, src/solv/{model,state,gbsa,born,sasa,kernel,input}.f90,")
    w("//         src/param/vdwradd3.f90, src/mctc/{convert,mctc_constants}.f90.")
    w("// Literals carry exactly the decimal digits of the Fortran source.")
    w("#![allow(clippy::excessive_precision, clippy::unreadable_literal, dead_code)]")
    w("")
    w(f"/// Number of elements with solvation parameters.")
    w(f"pub const NELEM: usize = {NELEM};")
    w(f"/// Default number of Lebedev grid points for the SASA (input.F90).")
    w(f"pub const DEFAULT_NANG: usize = {default_nang};")
    for name, val, doc in consts:
        w(f"/// {doc}")
        w(f"pub const {name}: f64 = {val};")
    w("")
    w("/// Raw solvent parameter set, field names as in xtb's `gbsa_parameter`.")
    w("#[derive(Debug)]")
    w("pub struct RawParam {")
    w("    /// Name of the Fortran parameter set.")
    w("    pub name: &'static str,")
    w("    /// Dielectric constant.")
    w("    pub epsv: f64,")
    w("    /// Molar mass of the solvent (g/mol).")
    w("    pub smass: f64,")
    w("    /// Density of the solvent (g/cm^3).")
    w("    pub rhos: f64,")
    w("    /// Born radius scaling.")
    w("    pub c1: f64,")
    w("    /// Probe radius (Angstrom).")
    w("    pub rprobe: f64,")
    w("    /// Empirical free energy shift (kcal/mol).")
    w("    pub gshift: f64,")
    w("    /// Born offset (0.1 Angstrom).")
    w("    pub soset: f64,")
    w("    /// Unused by xtb (ALPB alpha is hard coded).")
    w("    pub alpha: f64,")
    w("    /// Surface tension scaling per element.")
    w("    pub gamscale: [f64; NELEM],")
    w("    /// Dielectric descreening per element.")
    w("    pub sx: [f64; NELEM],")
    w("    /// Square root of the H-bond strength per element (kcal/mol)^(1/2).")
    w("    pub tmp: [f64; NELEM],")
    w("}")
    w("")

    def emit_set(var: str, p: str) -> None:
        e = sets[p]
        w(f"/// `{p}` from include/{e['file']}")
        w(f"pub static {var}: RawParam = RawParam {{")
        w(f"    name: \"{p}\",")
        for k in SCALARS:
            w(f"    {k}: {e[k]},")
        for k in ["gamscale", "sx", "tmp"]:
            w(f"    {k}: [")
            vals = e[k]
            for i in range(0, NELEM, 5):
                w("        " + " ".join(v + "," for v in vals[i:i + 5]))
            w("    ],")
        w("};")
        w("")

    ident = {}
    for p in alpb_sets + gbsa_sets:
        ident[p] = p.upper()
        emit_set(ident[p], p)

    w("/// Solvent-name aliases of xtb's `normalizeSolventName` (lowercase input -> normalised name).")
    w("pub static NORMALIZE: &[(&str, &str)] = &[")
    for a, t in norm:
        w(f"    (\"{a}\", \"{t}\"),")
    w("];")
    w("")
    w("/// GFN2-xTB/ALPB selection table of xtb's `loadInternalParam` (normalised name -> set).")
    w("pub static GFN2_ALPB: &[(&str, &RawParam)] = &[")
    for a, p in alpb_sel:
        w(f"    (\"{a}\", &{ident[p]}),")
    w("];")
    w("")
    w("/// GFN2-xTB/GBSA selection table of xtb's `loadInternalParam` (normalised name -> set).")
    w("pub static GFN2_GBSA: &[(&str, &RawParam)] = &[")
    for a, p in gbsa_sel:
        w(f"    (\"{a}\", &{ident[p]}),")
    w("];")
    w("")
    w("/// D3 van-der-Waals radii in Angstrom (xtb `vanDerWaalsRadD3` before the `aatoau` factor).")
    w("pub static VDW_RAD_D3_AA: [f64; NELEM] = [")
    for i in range(0, NELEM, 6):
        w("    " + " ".join(v + "," for v in vdw[i:i + 6]))
    w("];")
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}: {len(alpb_sets)} ALPB and {len(gbsa_sets)} GBSA GFN2 parameter sets, "
          f"{len(norm)} aliases", file=sys.stderr)


def gen_lebedev(xtb: Path, out: Path) -> None:
    sizes, grids = parse_lebedev(xtb / "src/solv/lebedev.f90")
    lines: list[str] = []
    w = lines.append
    w("// SPDX-License-Identifier: LGPL-3.0-or-later")
    w("// @generated by crates/solvation/scripts/gen_tables.py -- do not edit by hand.")
    w("// Source: xtb 6.7.1 src/solv/lebedev.f90 (Lebedev-Laikov generators).")
    w("// Literals carry exactly the decimal digits of the Fortran source.")
    w("#![allow(clippy::excessive_precision, clippy::unreadable_literal)]")
    w("")
    w("use crate::lebedev::Gen::{self, Oh1, Oh2, Oh3, Oh4, Oh5, Oh6};")
    w("")
    w("/// Available grid sizes (xtb `gridSize`).")
    w(f"pub static GRID_SIZE: [usize; {len(sizes)}] = [{', '.join(str(s) for s in sizes)}];")
    w("")
    w("/// Generator sequence of each grid, in the order of `GRID_SIZE`.")
    w(f"pub static GRIDS: [&[Gen]; {len(sizes)}] = [")
    for s in sizes:
        w(f"    // ld{s:04d}")
        w("    &[")
        for g in grids[s]:
            kind, *args = g
            w(f"        Oh{kind}({', '.join(args)}),")
        w("    ],")
    w("];")
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}: {len(sizes)} grids", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xtb", type=Path, default=ROOT / "reference" / "xtb-gxtb", help="xtb source tree")
    ap.add_argument("--out-dir", type=Path, default=CRATE / "src", help="output directory")
    args = ap.parse_args()
    gen_params(args.xtb, args.out_dir / "params.rs")
    gen_lebedev(args.xtb, args.out_dir / "lebedev_data.rs")


if __name__ == "__main__":
    main()
