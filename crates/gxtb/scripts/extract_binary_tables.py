#!/usr/bin/env python3
"""Extract the compiled-in data tables of the g-xTB v1.1.0 binary into Rust.

Usage: python3 crates/gxtb/scripts/extract_binary_tables.py <path to gxtb binary> <output .rs> [<crates/d4/src/data.rs>]

With the third argument the script also checks that the dftd4 reference tables revD4 shares with
axtb_d4 (`SHARED_D4`) are bit-identical in the program and in crates/d4.

The public g-xTB 1.1.0 program (grimme-lab/g-xtb release v1.1.0, GPL-3.0) is a statically
linked, unstripped Intel Fortran executable. Besides the three parameter files it reads at
run time (``.gxtb``, ``.basisq``, ``.eeq``) it carries element tables in its data segment.
This script reads them by symbol name (``nm``) from the ELF file, so every value is
bit-identical to what the program uses (including single-precision literals that the
Fortran source widened to double, e.g. the EEQ-BC average coordination numbers).

Requires binutils ``nm``; no third-party Python packages.
"""

from __future__ import annotations

import struct
import subprocess
import sys

# symbol -> (Rust name, kind, count, doc)
TABLES = [
    ("com_mp_pen_", "PAULING_EN", "f64", 103,
     "Pauling electronegativities (com::pen); actinides set as in multicharge's EEQ-BC 2025 model. "
     "Divided by 3.98 for the EEQ-BC electronegativity-weighted CN."),
    ("com_mp_cnav_", "CN_AVG", "f64", 103,
     "Average coordination numbers (com::cnav), stored by the program as single-precision "
     "literals widened to double. Used by EEQ-BC (charge-width CN normalization)."),
    ("data_vdwrad_mp_vdwrad_", "VDW_RAD_PACKED", "f64", 5356,
     "D3 pairwise van der Waals radii in Angstrom (data_vdwrad::vdwrad), packed lower triangle: "
     "index (max(i,j)*(max(i,j)-1))/2 + min(i,j) - 1 for 1-based atomic numbers i, j. The program "
     "copies them unscaled into parcom::gp3_aesr0."),
    ("ncoord_basq_$RCOV.0.3", "RCOV_BASIS_CN", "f64", 118,
     "Covalent radii in bohr used by the q-vSZP basis coordination number (ncoord_basq)."),
    ("com_mp_rcov_", "RCOV_COM", "f64", 118, "Covalent radii in bohr (com::rcov); role not yet assigned."),
    ("parcom_mp_gp3_gam3_", "GAM3", "f64", 103,
     "Element-wise third-order parameter (parcom::gp3_gam3), compiled in; role per the preprint: "
     "Gamma^(3)_A (unverified)."),
    ("com_mp_gam_", "GAM", "f64", 103, "Element hardness table (com::gam); role not yet assigned."),
    ("com_mp_valel_", "VALEL", "f64", 103, "Valence electron counts (com::valel)."),
    ("com_mp_ipat_", "IP_ATOM", "f64", 103, "Atomic ionization potentials in eV (com::ipat)."),
    ("com_mp_eaat_", "EA_ATOM", "f64", 103, "Atomic electron affinities in eV (com::eaat)."),
    ("com_mp_zeff_shell_", "ZEFF_SHELL", "i32", 412,
     "Aufbau valence-shell electron counts (com::zeff_shell), column-major 103 x 4 (s, p, d, f)."),
    ("com_mp_gpse_", "GPSE", "i32", 103, "Periodic-system group index (com::gpse)."),
    ("bascom_mp_bas_pol_", "BAS_POL", "i32", 412,
     "Polarization-shell flags (bascom::bas_pol), column-major 103 x 4 (shell index in basis order): "
     "1 marks a polarization shell; the MFX kernel averages Hubbard parameters with xi = max + 1."),
    ("com_mp_ipse_", "IPSE", "i32", 103, "Periodic-system period index (com::ipse)."),
    ("com_mp_gmunu_", "GMUNU", "f64", 927,
     "Onsite exchange integrals (com::gmunu), 9 per element in the order sp, pp, sd, pd, dd, sf, pf, df, ff; "
     "the program scales them by 0.15 at start-up (OFX)."),
    ("com_mp_k3_pair_", "K3_PAIR", "f64", 10609,
     "Pair table 103 x 103 (com::k3_pair), symmetric: the D4S pair weighting factors wf(Z_A, Z_B) of the "
     "revD4 Gaussian CN weights (identical to dftd4's `weight_factors` for Z <= 103)."),
    # revD4: the program's own dftd4 copy (module dftd4_reference / dftd4_data)
    ("dftd4_reference_mp_refcn_", "D4_REFCN", "f64", 826,
     "revD4 reference coordination numbers counted (nint) for the number of Gaussian weights per "
     "reference (`set_refgw`), 7 per element (element-major), Z = 1..118. Differs from dftd4 4.3.1's `refcn`."),
    ("dftd4_reference_mp_refcovcn_", "D4_REFCOVCN", "f64", 826,
     "revD4 reference coordination numbers of the Gaussian weights (`set_refcn`), 7 per element, "
     "Z = 1..118; recomputed for the revD4 CN (no electronegativity factor), unlike dftd4 4.3.1's `refcovcn`."),
    ("dftd4_reference_mp_gxtbq_", "D4_REFQ", "f64", 826,
     "revD4 reference charges (`gxtbq`, `set_refq_gxtb`), 7 per element, Z = 1..118."),
    ("dftd4_reference_mp_gxtbh_", "D4_REFH", "f64", 826,
     "revD4 charges of the hydrogen-subtracted secondary systems (`gxtbh`, `set_refalpha_gxtb`), "
     "7 per element, Z = 1..118."),
    ("dftd4_data_tanh_mp_tanh_params_", "D4_TANH", "f64", 472,
     "revD4 charge scaling (`get_tanh_params`), 4 per element (a, b, c, d), Z = 1..118: "
     "alpha(q) ~ a + b tanh(c q + d)."),
    ("dftd4_data_covrad_mp_covalent_rad_d3_", "D4_RCOV", "f64", 118,
     "Covalent radii (bohr) of the revD4 coordination number: 4/3 Pyykko radii converted with "
     "1 A = 1/0.52917726 bohr (dftd4 3.x `covalent_rad_d3`)."),
]


# Unnamed static tables (local SAVE data of a routine): (address, Rust name, kind, count, doc)
RAW_TABLES = [
    (0x3AE4540, "SPIN_W0", "f64", 1030,
     "Reference spin constants W0 (`get_spin_constants_` static table), 10 per element "
     "(element-major, Z = 1..103) in the order ss, sp, sd, sf, pp, pd, pf, dd, df, ff "
     "(`get_spin_constants_$INDX`); float32-rounded. W_ll' = atspin_Z W0."),
]


def load_symbols(binary: str) -> dict[str, tuple[int, int]]:
    out = subprocess.run(["nm", "-S", binary], capture_output=True, text=True, check=True).stdout
    syms = {}
    for line in out.splitlines():
        p = line.split()
        if len(p) == 4:
            syms[p[3]] = (int(p[0], 16), int(p[1], 16))
    return syms


def load_segments(data: bytes) -> list[tuple[int, int, int, int]]:
    phoff, = struct.unpack_from("<Q", data, 0x20)
    phentsize, phnum = struct.unpack_from("<HH", data, 0x36)
    segs = []
    for i in range(phnum):
        p_type, _, off, vaddr, _, filesz, memsz, _ = struct.unpack_from("<IIQQQQQQ", data, phoff + i * phentsize)
        if p_type == 1:
            segs.append((vaddr, off, filesz, memsz))
    return segs


def read(data, segs, addr, n):
    for va, off, fs, ms in segs:
        if va <= addr and addr + n <= va + fs:
            return data[off + addr - va: off + addr - va + n]
    raise KeyError(hex(addr))


# dftd4 tables of the program that revD4 shares with axtb_d4 (crates/d4/src/data.rs, dftd4 4.3.1):
# binary symbol -> (axtb_d4 name, kind, count). They must be identical for crates/gxtb to use axtb_d4's.
SHARED_D4 = {
    "dftd4_reference_mp_alphaiw_": ("ALPHAIW", "d", 23 * 7 * 118),
    "dftd4_reference_mp_ascale_": ("ASCALE", "d", 7 * 118),
    "dftd4_reference_mp_hcount_": ("HCOUNT", "d", 7 * 118),
    "dftd4_reference_mp_secaiw_": ("SECAIW", "d", 23 * 17),
    "dftd4_reference_mp_sscale_": ("SSCALE", "d", 17),
    "dftd4_reference_mp_refn_": ("REFN", "i", 118),
    "dftd4_reference_mp_refsys_": ("REFSYS", "i", 7 * 118),
    "dftd4_data_r4r2_mp_r4_over_r2_": ("R4_OVER_R2", "d", 118),
}


def check_shared_d4(data, segs, syms, d4_data_rs: str) -> None:
    """Fail unless the program's dftd4 tables listed in SHARED_D4 equal axtb_d4's."""
    import re
    src = open(d4_data_rs).read()
    for sym, (name, fmt, count) in SHARED_D4.items():
        m = re.search(r"pub static " + name + r"\s*:[^=]*=\s*\[(.*?)\];\n", src, re.S)
        body = re.sub(r"//[^\n]*", "", m.group(1))
        ours = [float(x) for x in re.findall(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", body)]
        addr, _ = syms[sym]
        prog = struct.unpack(f"<{count}{fmt}", read(data, segs, addr, (8 if fmt == "d" else 4) * count))
        if len(ours) != count or any(float(a) != float(b) for a, b in zip(ours, prog)):
            raise SystemExit(f"{sym} differs from axtb_d4::data::{name}")
    print(f"shared dftd4 tables identical to {d4_data_rs}: {', '.join(v[0] for v in SHARED_D4.values())}")


def main() -> None:
    binary, target = sys.argv[1], sys.argv[2]
    data = open(binary, "rb").read()
    segs = load_segments(data)
    syms = load_symbols(binary)
    if len(sys.argv) > 3:
        check_shared_d4(data, segs, syms, sys.argv[3])
    out = [
        "// @generated by crates/gxtb/scripts/extract_binary_tables.py from the data segment of the",
        "// g-xTB 1.1.0 program (grimme-lab/g-xtb release v1.1.0, GPL-3.0). Do not edit by hand.",
        "#![allow(clippy::excessive_precision, clippy::unreadable_literal, clippy::approx_constant)]",
        "",
    ]
    for sym, name, kind, count, doc in TABLES:
        addr, size = syms[sym]
        width = 4 if kind in ("f32", "i32") else 8
        if size < width * count:
            raise SystemExit(f"{sym}: size {size} < {width * count}")
        raw = read(data, segs, addr, width * count)
        fmt = {"f64": "d", "f32": "f", "i32": "i"}[kind]
        vals = struct.unpack(f"<{count}{fmt}", raw)
        rtype = {"f64": "f64", "f32": "f32", "i32": "i32"}[kind]
        out.append(f"/// {doc}")
        out.append(f"/// Binary symbol `{sym}`.")
        out.append(f"pub static {name}: [{rtype}; {count}] = [")
        per = 4
        for k in range(0, count, per):
            chunk = vals[k:k + per]
            if kind == "i32":
                out.append("    " + ", ".join(str(v) for v in chunk) + ",")
            else:
                out.append("    " + ", ".join(repr(float(v)) if v == v else "f64::NAN" for v in chunk) + ",")
        out.append("];")
        out.append("")
    for addr, name, kind, count, doc in RAW_TABLES:
        vals = struct.unpack(f"<{count}d", read(data, segs, addr, 8 * count))
        out.append(f"/// {doc}")
        out.append(f"/// Binary address 0x{addr:x}.")
        out.append(f"pub static {name}: [f64; {count}] = [")
        for k in range(0, count, 5):
            out.append("    " + ", ".join(repr(float(v)) for v in vals[k:k + 5]) + ",")
        out.append("];")
        out.append("")
    open(target, "w").write("\n".join(out))


if __name__ == "__main__":
    main()
