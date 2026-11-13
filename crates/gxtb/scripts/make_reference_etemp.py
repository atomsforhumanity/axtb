#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""g-xTB 1.1.0 reference data at finite electronic temperatures (the program's `-tel <K>`).

Usage: python3 crates/gxtb/scripts/make_reference_etemp.py <g-xtb-1.1.0 dir> [--jobs N]

<g-xtb-1.1.0 dir> holds the program `gxtb` and its parameter files (gxtb_parameters, basisq,
eeq). For the molecules of crates/gxtb/tests/reference and the extra finite-temperature cases
of crates/gxtb/tests/reference_v2/etemp/extra (Fe4, stretched N2) at 300, 1000, 3000 and
5000 K, in a scratch directory with HOME=. and the parameter files copied in (charge and
unpaired electrons in .CHRG and .UHF, no gxtbrestart):

    gxtb -c mol.xyz -tel T                        -> total energy (`energy` file), SCF iterations

and for a few molecules at 3000 and 5000 K the program's numerical gradient with the SCF
thresholds of every displaced single point tightened under gdb (as `tight_gradient` in
make_reference.py) -> "gradient_tight". Written to crates/gxtb/tests/reference/etemp.json.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTS = HERE.parent / "tests"
TEMPS = (300.0, 1000.0, 3000.0, 5000.0)
GRADIENT_CASES = {"water", "dioxygen_triplet", "tetrachlorocuprate", "N2_stretched"}
sys.path.insert(0, str(HERE))
from make_reference import tight_gradient  # noqa: E402


def molecules():
    for d in sorted((TESTS / "reference").iterdir()):
        if (d / "input.json").exists():
            yield "reference", d
    for d in sorted((TESTS / "reference_v2/etemp/extra").iterdir()):
        yield "extra", d


def setup(prog: Path, w: Path, d: Path):
    for src, dst in (("gxtb_parameters", ".gxtb"), ("basisq", ".basisq"), ("eeq", ".eeq")):
        shutil.copy(prog / src, w / dst)
    shutil.copy(d / "mol.xyz", w / "mol.xyz")
    inp = json.loads((d / "input.json").read_text())
    if inp["charge"]:
        (w / ".CHRG").write_text(f"{inp['charge']}\n")
    if inp["uhf"]:
        (w / ".UHF").write_text(f"{inp['uhf']}\n")
    return inp


def job(args):
    prog, group, d, temp = args
    prog = Path(prog)
    with tempfile.TemporaryDirectory() as tmp:
        w = Path(tmp)
        inp = setup(prog, w, d)
        env = dict(os.environ, OMP_NUM_THREADS="1", HOME=".")
        p = subprocess.run([str(prog / "gxtb"), "-c", "mol.xyz", "-tel", f"{temp:g}"], cwd=w, env=env,
                           capture_output=True, text=True, timeout=3600)
        out = p.stdout + p.stderr
        m = re.search(r"(\d+)\s+scf iterations", out)
        energy = None
        if (w / "energy").exists():
            energy = float((w / "energy").read_text().split("\n")[1].split()[1])
        row = {"name": d.name, "group": group, "charge": inp["charge"], "uhf": inp["uhf"], "etemp": temp,
               "total_energy": energy, "scf_iterations": int(m.group(1)) if m else None,
               "converged": energy is not None and m is not None}
        if d.name in GRADIENT_CASES and temp >= 3000.0 and row["converged"]:
            for f in ("gxtbrestart", "energy", "gradient"):
                (w / f).unlink(missing_ok=True)
            row["gradient_tight"] = tight_gradient_tel(str(prog / "gxtb"), str(w), temp)
    return row


def tight_gradient_tel(binary, workdir, temp):
    """`tight_gradient` of make_reference.py with `-tel T` appended to the command line."""
    import make_reference as mr
    orig = subprocess.run

    def run(cmd, *a, **k):
        if cmd[:1] == ["gdb"] and "-grad" in cmd:
            cmd = [*cmd, "-tel", f"{temp:g}"]
        return orig(cmd, *a, **k)

    mr.subprocess.run = run
    try:
        text = mr.tight_gradient(binary, workdir)
    finally:
        mr.subprocess.run = orig
    lines = text.splitlines()
    rows = [l.split() for l in lines[2:] if not l.startswith("$")]
    nat = sum(1 for r in rows if len(r) == 4)
    return [[float(x.replace("D", "E")) for x in r] for r in rows[nat:2 * nat]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("program_dir")
    ap.add_argument("--jobs", type=int, default=8)
    a = ap.parse_args()
    jobs = [(a.program_dir, g, d, t) for g, d in molecules() for t in TEMPS]
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(job, jobs))
    doc = {"program": "g-xTB 1.1.0 (grimme-lab/g-xtb v1.1.0)",
           "command": "gxtb -c mol.xyz -tel T [-grad -symthr 0 under gdb with tight SCF thresholds]; HOME=. with "
                      ".gxtb/.basisq/.eeq, OMP_NUM_THREADS=1, no gxtbrestart; charge in .CHRG, unpaired electrons in .UHF",
           "units": "Eh, Eh/bohr", "rows": rows}
    text = json.dumps(doc, separators=(",", ":"))
    text = text.replace(',"rows":[{', ',"rows":[\n{').replace('},{"name"', '},\n{"name"')
    (TESTS / "reference/etemp.json").write_text(text + "\n")
    for r in rows:
        print(f'{r["name"]:22} {r["etemp"]:6g} K  E {r["total_energy"]}  it {r["scf_iterations"]}'
              f'{"  grad" if "gradient_tight" in r else ""}')


if __name__ == "__main__":
    main()
