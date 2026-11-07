#!/usr/bin/env python3
"""g-xTB 2.0.1 reference data at finite electronic temperatures (Fermi smearing, `--etemp`).

Usage: python3 crates/gxtb/scripts/make_reference_v2_etemp.py <xtb 2.0.1 binary> [--jobs N] [--temps 300,1000,...]

For every molecule of crates/gxtb/tests/reference_v2 (groups reference, elements, large) and the
extra finite-temperature cases in reference_v2/etemp/extra (Fe4 tetrahedron at 2.30 Å, N2 at
1.65 Å; both singlets), at every temperature, in a fresh directory:

    OMP_NUM_THREADS=1 xtb mol.xyz --gxtb --norestart --chrg Q [--uhf U] --etemp T --grad

and the same with `--acc 0.0001` (tight: energy change 1e-10 Eh, residual 2e-9). `--uhf` only for
open shells, as in make_reference_v2.py: any `--uhf` switches the program to two spin channels.
Recorded in reference_v2/etemp/etemp.json: the total energy of the summary (12 decimals), the
electronic energy (14 digits), the SCF iterations and every cycle's energy, whether the SCF
converged, and the analytic gradient (`gradient` file, Eh/bohr); the tight run's energy,
iterations and gradient under "tight". Runs are independent and run in parallel (one thread
each). Nothing host-specific is stored.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "tests/reference_v2"
OUT = ROOT / "etemp"
TEMPS = (300.0, 1000.0, 3000.0, 5000.0)


def extra_molecules():
    """Fe4 (regular tetrahedron, Fe-Fe 2.30 Å) and N2 at 1.65 Å, as in tests/reference_matrix."""
    s = 2.30 / (2.0 * 2.0 ** 0.5)
    fe4 = [("Fe", s * x, s * y, s * z) for x, y, z in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1))]
    n2 = [("N", 0.0, 0.0, -0.825), ("N", 0.0, 0.0, 0.825)]
    for name, atoms in (("Fe4", fe4), ("N2_stretched", n2)):
        d = OUT / "extra" / name
        d.mkdir(parents=True, exist_ok=True)
        xyz = f"{len(atoms)}\n{name}\n" + "".join(f"{e:2s} {x:16.10f} {y:16.10f} {z:16.10f}\n" for e, x, y, z in atoms)
        (d / "mol.xyz").write_text(xyz)
        (d / "input.json").write_text(json.dumps({"charge": 0, "uhf": 0}, indent=1) + "\n")


def molecules():
    for group in ("reference", "elements", "large"):
        for d in sorted((ROOT / group).iterdir()):
            if (d / "input.json").exists():
                yield group, d
    for d in sorted((OUT / "extra").iterdir()):
        yield "extra", d


def parse(out: str) -> dict:
    s = {}
    m = re.search(r":: total energy\s+(-?\d+\.\d+) Eh", out)
    s["total_energy"] = float(m.group(1)) if m else None
    m = re.search(r"electronic energy\s+(-?\d\.\d+E[+-]\d+)", out)
    s["electronic_energy"] = float(m.group(1)) if m else None
    cyc = re.findall(r"^\s+(\d+)\s+(-?\d+\.\d+)\s+(-?\d\.\d+E[+-]\d+)\s+(-?\d\.\d+E[+-]\d+)\s*$", out, re.M)
    s["scf_cycles"] = [[int(a), float(b)] for a, b, _, _ in cyc]
    s["scf_iterations"] = len(cyc)
    s["converged"] = "SCF not converged" not in out and "not converged" not in out.lower() and s["total_energy"] is not None
    return s


def read_gradient(path: Path):
    lines = path.read_text().splitlines()
    rows = [l.split() for l in lines[2:] if not l.startswith("$")]
    nat = sum(1 for r in rows if len(r) == 4)
    return [[float(x.replace("D", "E")) for x in r] for r in rows[nat:2 * nat]]


def run(binary, d, q, u, temp, extra):
    with tempfile.TemporaryDirectory() as tmp:
        w = Path(tmp)
        (w / "mol.xyz").write_text((d / "mol.xyz").read_text())
        env = dict(os.environ, OMP_NUM_THREADS="1", OMP_STACKSIZE="1G")
        cmd = [binary, "mol.xyz", "--gxtb", "--norestart", "--chrg", str(q), *(("--uhf", str(u)) if u else ()),
               "--etemp", f"{temp:g}", "--grad", *extra]
        p = subprocess.run(cmd, cwd=w, env=env, capture_output=True, text=True)
        s = parse(p.stdout + p.stderr)
        s["returncode"] = p.returncode
        s["gradient"] = read_gradient(w / "gradient") if (w / "gradient").exists() and s["converged"] else None
    return s


def job(args):
    binary, group, d, temp = args
    inp = json.loads((d / "input.json").read_text())
    q, u = inp["charge"], inp["uhf"]
    s = run(binary, d, q, u, temp, [])
    t = run(binary, d, q, u, temp, ["--acc", "0.0001"])
    s["tight"] = {k: t[k] for k in ("total_energy", "scf_iterations", "converged", "gradient")}
    return {"name": d.name, "group": group, "charge": q, "uhf": u, "etemp": temp, **s}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binary")
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--temps", default=",".join(f"{t:g}" for t in TEMPS))
    a = ap.parse_args()
    extra_molecules()
    temps = [float(t) for t in a.temps.split(",")]
    jobs = [(a.binary, g, d, t) for g, d in molecules() for t in temps]
    # longest first
    jobs.sort(key=lambda j: -int((j[2] / "mol.xyz").read_text().split()[0]))
    with ProcessPoolExecutor(a.jobs) as ex:
        rows = list(ex.map(job, jobs))
    order = {(g, d.name): k for k, (g, d) in enumerate(molecules())}
    rows.sort(key=lambda r: (order[(r["group"], r["name"])], r["etemp"]))
    doc = {"program": "g-xTB 2.0.1 (xtb 6.7.1 + tblite, grimme-lab/g-xtb)",
           "command": "xtb mol.xyz --gxtb --norestart --chrg Q [--uhf U] --etemp T --grad; OMP_NUM_THREADS=1",
           "units": "energies Eh, gradient Eh/bohr", "rows": rows}
    text = json.dumps(doc, indent=None, separators=(",", ":"))
    # one row per line
    text = text.replace(',"rows":[{', ',"rows":[\n{').replace('},{"name"', '},\n{"name"')
    (OUT / "etemp.json").write_text(text + "\n")
    for r in rows:
        t = r["tight"]
        print(f'{r["group"]:9} {r["name"]:22} {r["etemp"]:6g} K  E {r["total_energy"]}  it {r["scf_iterations"]}  conv {r["converged"]}'
              f'  tight E {t["total_energy"]} it {t["scf_iterations"]} conv {t["converged"]}')


if __name__ == "__main__":
    main()
