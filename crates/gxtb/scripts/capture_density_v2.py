#!/usr/bin/env python3
"""Capture the converged density of the g-xTB 2.0.1 program for per-term tests.

Usage: python3 crates/gxtb/scripts/capture_density_v2.py <xtb 2.0.1 binary> <reference_v2 dir> <group/name> ...

Runs `xtb mol.xyz --gxtb --norestart --chrg Q [--uhf U]` (OMP_NUM_THREADS=1, fresh directory) under
gdb and records the arguments of the last call of
`tblite_wavefunction_mulliken_mp_get_mulliken_shell_charges_(bas, S, P, n0sh, qsh)`, i.e. the
density of the last SCF iteration (the one the stored per-term energies `terms` belong to), and the
per-term energies of that run (as make_reference_v2.py). Writes `<dir>/<group>/<name>/density.json`:
nao, nspin, the packed lower triangle of P per spin channel (row by row, tblite AO order, which
equals axtb's), the Mulliken shell charges and reference occupations the program passed, and the
per-term energies. Requires gdb.
"""
from __future__ import annotations
import json, os, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_reference_v2 as mr  # noqa: E402

GDB = mr.TERMS_GDB.replace("run\npython\njson.dump(log, open(\"terms.json\", \"w\"))\nend\n", "") + r'''
python
def rdesc(a):
    base, elen, rank, dims = desc(a)
    n = 1
    for d in dims: n *= d[0]
    return list(struct.unpack("<%dd" % n, rd(base, 8 * n))), [d[0] for d in dims]
MULL = []
class Mull(gdb.Breakpoint):
    def stop(self):
        P, shp = rdesc(arg(2)); n0, _ = rdesc(arg(3))
        MULL.append((arg(4), P, shp, n0))
        if len(MULL) > 2: MULL.pop(0)
        Fin2(arg(4))
        return False
class Fin2(gdb.FinishBreakpoint):
    def __init__(self, a): super().__init__(gdb.newest_frame(), internal=True); self.a = a
    def stop(self):
        q, shq = rdesc(self.a); MULL[-1] = MULL[-1] + (q, shq); return False
    def out_of_scope(self): pass
Mull("tblite_wavefunction_mulliken_mp_get_mulliken_shell_charges_", internal=True)
end
run
python
a, P, shp, n0, q, shq = MULL[-1]
json.dump({"log": log, "P": P, "shapeP": shp, "n0sh": n0, "qsh": q, "shapeq": shq}, open("dens.json", "w"))
end
'''


def main():
    binary, root = sys.argv[1], Path(sys.argv[2])
    for name in sys.argv[3:]:
        d = root / name
        inp = json.loads((d / "input.json").read_text())
        xyz = (d / "mol.xyz").read_text()
        with tempfile.TemporaryDirectory() as tmp:
            w = Path(tmp)
            (w / "mol.xyz").write_text(xyz)
            (w / "dens.gdb").write_text(GDB)
            env = dict(os.environ, OMP_NUM_THREADS="1", OMP_STACKSIZE="1G", HOME=str(w))
            uhf = inp["uhf"]
            cmd = ["gdb", "-q", "-batch", "-x", "dens.gdb", "--args", binary, "mol.xyz", "--gxtb", "--norestart",
                   "--chrg", str(inp["charge"]), *(("--uhf", str(uhf)) if uhf else ())]
            p = subprocess.run(cmd, cwd=w, env=env, capture_output=True, text=True)
            if not (w / "dens.json").exists():
                print(p.stdout[-2000:], p.stderr[-2000:])
                raise SystemExit(f"{name}: capture failed")
            r = json.loads((w / "dens.json").read_text())
        nao = r["shapeP"][0]
        nspin = r["shapeP"][2] if len(r["shapeP"]) > 2 else 1
        P = r["P"]
        packed = []
        for s in range(nspin):
            off = s * nao * nao
            packed.append([P[off + j * nao + i] for i in range(nao) for j in range(i + 1)])
        nsh = r["shapeq"][0]
        out = {"program": "g-xTB 2.0.1", "nao": nao, "nspin": nspin, "density_packed": packed,
               "shell_charges": r["qsh"], "shape_shell_charges": r["shapeq"], "reference_occupations": r["n0sh"][:nsh],
               "terms": mr.terms(r["log"]),
               "note": "density of the last SCF iteration (argument of the last get_mulliken_shell_charges call); "
                       "packed lower triangle, row by row; open shell: spin channels as the program stores them"}
        (d / "density.json").write_text(json.dumps(out))
        print(name, nao, nspin, "terms:", {k: round(v, 12) for k, v in out["terms"].items()})


if __name__ == "__main__":
    main()
