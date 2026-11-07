#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Run xtb (6.7.1, GFN2-xTB) with ALPB/GBSA solvation and record reference data.

For every case the structure is written as a Turbomole ``coord`` file in bohr
(so no Angstrom->bohr conversion is involved), xtb is run in its own scratch
directory with ``OMP_NUM_THREADS=1``, and the following is recorded in
``tests/data/<case>.ref``:

* atomic numbers and the bohr coordinates (the exact strings given to xtb),
* the converged Mulliken atomic charges in full double precision, obtained
  from the shell charges stored in ``xtbrestart`` and summed per atom like
  xtb's ``qsh2qat`` (cross-checked against the 8-decimal ``charges`` file),
* the solvation energy components printed by xtb (Gsolv, Gelec, Gsasa, Ghb,
  Gshift; 12 decimals).

Usage::

    python3 crates/solvation/scripts/xtb_reference.py --scratch DIR [--xtb XTB] [--only SUBSTR]

The program is ``--xtb``, or else ``$AXTB_GXTB2`` (tests/programs.py): the g-xTB 2.0.1 build of
xtb 6.7.1 (26dd68d), which wrote the committed data. ``xtb`` is never taken from PATH.
"""
from __future__ import annotations

import argparse
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CRATE = HERE.parent
ROOT = HERE.parents[2]
STRUCTURES = ROOT / "tests" / "structures"
DATA = CRATE / "tests" / "data"
sys.path.insert(0, str(ROOT / "tests"))
import programs  # noqa: E402

AUTOAA = 0.52917726  # xtb's value; only used to build the coord file

SYMBOLS = (
    "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
    "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
    "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn"
).split()

# xtb grid keywords -> number of points (set_module.f90, setparam.f90)
NANG = {None: 230, "normal": 230, "tight": 974, "verytight": 2030, "extreme": 5810}

# structure -> (charge, unpaired electrons)
CHARGES = {"cu-nh3-4": (2, 1), "zncl4": (-2, 0)}

# Extra structures (Angstrom) with an atom whose van-der-Waals sphere engulfs
# the descreened sphere of a lighter neighbour with a smaller index: this hits
# the branch of xtb's compute_psi that uses a stale value of r1.
INLINE = {
    "hgh2-compressed": [("H", -1.40, 0.0, 0.0), ("H", 1.40, 0.0, 0.0), ("Hg", 0.0, 0.0, 0.0)],
    "csoh-compressed": [("H", 0.0, 0.0, -0.96), ("O", 0.0, 0.0, 0.0), ("Cs", 0.0, 0.0, 2.08)],
}

ALL_STRUCTURES = [
    "water", "caffeine", "anthracene2", "beni-acetone", "cf3br-benzene", "cu-nh3-4",
    "ferrocene", "gh106", "mb33", "s229-cl", "taxol", "tyr-xab", "zncl4",
]

# one name per GFN2 parameter set, several given as xtb aliases / mixed case
ALPB_SOLVENTS = [
    "acetone", "methylcyanide", "aniline", "benzaldehyde", "benzene", "dichloromethane",
    "chloroform", "cs2", "dioxane", "dmf", "DMSO", "ethanol", "diethylether",
    "ethylacetate", "furan", "hexadecane", "n-hexane", "methanol", "nitromethane",
    "octanol", "phenol", "THF", "toluene", "h2o", "woctanol",
]
GBSA_SOLVENTS = [
    "acetone", "acetonitrile", "benzene", "CH2Cl2", "trichloromethane", "carbondisulfide",
    "dimethylformamide", "dmso", "ether", "methanol", "nhexan", "tetrahydrofuran",
    "toluene", "water",
]


def cases() -> list[dict]:
    out = []
    for s in ALL_STRUCTURES:
        out.append(dict(mol=s, model="alpb", solvent="water", state=None))
    for solv in ALPB_SOLVENTS:
        out.append(dict(mol="caffeine", model="alpb", solvent=solv, state=None))
    for solv in GBSA_SOLVENTS:
        out.append(dict(mol="caffeine", model="gbsa", solvent=solv, state=None))
    for s in ["taxol", "zncl4", "cu-nh3-4", "cf3br-benzene", "beni-acetone", "ferrocene"]:
        out.append(dict(mol=s, model="gbsa", solvent="water", state=None))
    for s, model, solv in [("caffeine", "alpb", "water"), ("zncl4", "alpb", "water"),
                           ("caffeine", "gbsa", "water"), ("s229-cl", "alpb", "toluene"),
                           ("cu-nh3-4", "gbsa", "acetonitrile")]:
        for state in ["bar1M", "reference"]:
            out.append(dict(mol=s, model=model, solvent=solv, state=state))
    out.append(dict(mol="tyr-xab", model="alpb", solvent="dmso", state="gsolv"))
    for s in INLINE:
        out.append(dict(mol=s, model="alpb", solvent="water", state=None))
    # non-default Lebedev grids through an xcontrol file ($gbsa gbsagrid=...)
    for s, model, grid in [("caffeine", "alpb", "tight"), ("cf3br-benzene", "gbsa", "590"),
                           ("zncl4", "alpb", "5810")]:
        out.append(dict(mol=s, model=model, solvent="water", state=None, grid=grid))
    for c in out:
        tag = c["solvent"].lower()
        c["name"] = f"{c['mol']}_{c['model']}_{tag}" + (f"_{c['state']}" if c["state"] else "")
        if c.get("grid"):
            c["name"] += f"_grid{c['grid']}"
    return out


def read_xyz(path: Path) -> tuple[list[int], list[list[float]]]:
    if path.stem in INLINE:
        atoms = INLINE[path.stem]
        return [SYMBOLS.index(a[0]) + 1 for a in atoms], [list(a[1:]) for a in atoms]
    lines = path.read_text().splitlines()
    n = int(lines[0].split()[0])
    nums, pos = [], []
    for line in lines[2:2 + n]:
        t = line.split()
        sym = t[0].capitalize()
        nums.append(int(sym) if sym.isdigit() else SYMBOLS.index(sym) + 1)
        pos.append([float(x) for x in t[1:4]])
    return nums, pos


def shells_per_element(param: Path) -> dict[int, int]:
    out, z = {}, None
    for line in param.read_text().splitlines():
        m = re.match(r"^\$Z=\s*(\d+)", line)
        if m:
            z = int(m.group(1))
            continue
        m = re.match(r"^\s*ao=(\S+)", line)
        if m and z is not None:
            out[z] = len(m.group(1)) // 2
            z = None
    return out


def read_restart(path: Path) -> list[float]:
    data = path.read_bytes()
    recs, i = [], 0
    while i < len(data):
        (n,) = struct.unpack_from("<i", data, i)
        recs.append(data[i + 4:i + 4 + n])
        (n2,) = struct.unpack_from("<i", data, i + 4 + n)
        assert n == n2, "corrupt Fortran record"
        i += 8 + n
    head = struct.unpack("<6q", recs[0])
    nshell = head[3]
    qsh = list(struct.unpack(f"<{nshell}d", recs[1]))
    return qsh


ENERGY_KEYS = {"Gsolv": "gsolv", "Gelec": "gelec", "Gsasa": "gsasa", "Ghb": "ghb", "Gshift": "gshift"}


def run_case(c: dict, xtb: str, scratch: Path, nshell: dict[int, int]) -> str | None:
    nums, pos = read_xyz(STRUCTURES / f"{c['mol']}.xyz")
    wd = scratch / c["name"]
    if wd.exists():
        shutil.rmtree(wd)
    wd.mkdir(parents=True)
    coord_lines = []
    for z, p in zip(nums, pos):
        coord_lines.append([f"{x / AUTOAA:.14f}" for x in p] + [SYMBOLS[z - 1].lower()])
    (wd / "coord").write_text("$coord\n" + "".join(f"{a} {b} {cc} {s}\n" for a, b, cc, s in coord_lines) + "$end\n")
    chrg, uhf = CHARGES.get(c["mol"], (0, 0))
    cmd = [xtb, "coord", f"--{c['model']}", c["solvent"]]
    if c["state"]:
        cmd.append(c["state"])
    cmd += ["--chrg", str(chrg), "--uhf", str(uhf)]
    if c.get("grid"):
        (wd / "xcontrol").write_text(f"$gbsa\n   gbsagrid={c['grid']}\n$end\n")
        cmd += ["--input", "xcontrol"]
    env = dict(os.environ, OMP_NUM_THREADS="1", OMP_STACKSIZE="1G")
    res = subprocess.run(cmd, cwd=wd, env=env, capture_output=True, text=True)
    (wd / "xtb.out").write_text(res.stdout + res.stderr)
    out = res.stdout
    if res.returncode != 0 or "convergence criteria satisfied" not in out:
        print(f"{c['name']}: xtb failed (rc={res.returncode}), skipped", file=sys.stderr)
        return None
    energies = {}
    for key, tag in ENERGY_KEYS.items():
        m = re.search(rf"::\s+(?:->\s+)?{key}\s+(-?\d+\.\d+)\s+Eh", out)
        if not m:
            raise SystemExit(f"{c['name']}: {key} not found in xtb output")
        energies[tag] = m.group(1)
    info = {}
    for key in ["Solvation model", "Parameter file", "Reference state", "Free energy shift", "Interaction kernel",
                "Grid points"]:
        m = re.search(rf"^\s*\*?\s*{key}:?\s+(.*)$", out, re.M)
        info[key] = m.group(1).strip() if m else "?"

    qsh = read_restart(wd / "xtbrestart")
    q, k = [], 0
    for z in nums:
        s = 0.0
        for _ in range(nshell[z]):
            s += qsh[k]
            k += 1
        q.append(s)
    if k != len(qsh):
        raise SystemExit(f"{c['name']}: shell count mismatch ({k} vs {len(qsh)})")
    qfile = [float(x) for x in (wd / "charges").read_text().split()]
    dq = max(abs(a - b) for a, b in zip(q, qfile))
    if dq > 6e-9:
        raise SystemExit(f"{c['name']}: restart charges differ from charges file by {dq}")

    lines = [
        "# generated by crates/solvation/scripts/xtb_reference.py",
        f"# {xtb_version(xtb)}",
        f"# command: {' '.join(['xtb', *cmd[1:]])}",  # the binary's name only, not its path
        f"# xtb: {info['Solvation model']} | {info['Parameter file']} | {info['Reference state']} | "
        f"shift {info['Free energy shift']} | kernel {info['Interaction kernel']}",
        f"case {c['name']}",
        f"model {c['model']}",
        f"solvent {c['solvent']}",
        f"state {c['state'] or 'gsolv'}",
        f"nang {NANG.get(c.get('grid'), c.get('grid') or 230)}",
        f"# xtb grid: {info['Grid points']}",
        f"charge {chrg}",
        f"nat {len(nums)}",
    ]
    for z, (x, y, zz, _), qi in zip(nums, coord_lines, q):
        lines.append(f"atom {z} {x} {y} {zz} {qi!r}")
    for tag in ENERGY_KEYS.values():
        lines.append(f"{tag} {energies[tag]}")
    return "\n".join(lines) + "\n"


_VERSION = None


def xtb_version(xtb: str) -> str:
    global _VERSION
    if _VERSION is None:
        out = subprocess.run([xtb, "--version"], capture_output=True, text=True).stdout
        m = re.search(r"xtb version (\S+ \(\S+\))", out)
        _VERSION = "xtb version " + (m.group(1) if m else "?")
    return _VERSION


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xtb", default=None, help="the xtb program (default: $AXTB_GXTB2, see tests/programs.py)")
    ap.add_argument("--scratch", type=Path, required=True, help="directory for the xtb runs (one subdir per case)")
    ap.add_argument("--only", default=None, help="only cases whose name contains this string")
    args = ap.parse_args()
    xtb = args.xtb or programs.require(programs.GXTB2, what="the ALPB/GBSA reference data")[0]
    xtb_path = shutil.which(xtb)
    if not xtb_path:
        raise SystemExit(f"{xtb}: not found")
    param = Path(xtb_path).resolve().parents[1] / "share" / "xtb" / "param_gfn2-xtb.txt"
    nshell = shells_per_element(param)
    DATA.mkdir(parents=True, exist_ok=True)
    n = 0
    for c in cases():
        if args.only and args.only not in c["name"]:
            continue
        text = run_case(c, xtb_path, args.scratch, nshell)
        if text is None:
            continue
        (DATA / f"{c['name']}.ref").write_text(text)
        n += 1
        print(f"{c['name']}: ok", file=sys.stderr)
    print(f"wrote {n} reference files to {DATA}", file=sys.stderr)


if __name__ == "__main__":
    main()
