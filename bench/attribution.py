#!/usr/bin/env python3
"""Timing attribution: the programs' own wall-time breakdown (xtb --verbose: classical,
integrals, SCC iterations, gradient; the g-xTB build's tblite timers) next to axtb's phase
timings with faer and with run-time LAPACK libraries, for GFN2-xTB (xtb 6.8.0) and g-xTB 2.0.1
(the g-xTB build of xtb), at the given thread counts. Best of N runs (by total wall time) per
setting, the settings interleaved; one JSON line per (method, molecule, threads, setting) with
the binary, its arguments and the library variables. bench/attribution_table.py turns the lines
into a markdown table.

usage: attribution.py --axtb BIN [--xtb XTB] [--gxtb GXTB] --structures DIR --mols taxol,... --threads 1,8
(--xtb and --gxtb default to $AXTB_XTB680 and $AXTB_GXTB2, tests/programs.py)
                      --rounds 3 [--lib name=path[:blas] ...] [--xyz name=path ...] --out file.jsonl"""
import argparse, json, os, re, shutil, subprocess, sys, tempfile, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
import programs  # noqa: E402

FILES = {"taxol": "taxol.xyz", "C50H102": "large/c50h102.xyz", "(H2O)100": "large/water100.xyz",
         "(H2O)200": "large/water200.xyz", "caffeine": "caffeine.xyz"}
SYM = {"H": 1, "C": 6, "N": 7, "O": 8}
ANG = 1.0 / 0.52917721


def read_xyz(path):
    lines = open(path).read().splitlines()
    nat = int(lines[0])
    nums, pos = [], []
    for l in lines[2:2 + nat]:
        f = l.split()
        nums.append(SYM[f[0]])
        pos.append([float(x) * ANG for x in f[1:4]])
    return nums, pos


def program(binary, args, xyz, threads):
    d = tempfile.mkdtemp()
    shutil.copy(xyz, os.path.join(d, "mol.xyz"))
    env = dict(os.environ, OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads), OMP_STACKSIZE="1G")
    t0 = time.perf_counter()
    out = subprocess.run([binary, "mol.xyz", *args, "--verbose"], cwd=d, env=env, capture_output=True, text=True).stdout
    wall = time.perf_counter() - t0
    shutil.rmtree(d)
    ph = {k: float(v) for k, v in re.findall(r"^ (classical contributions|integral evaluation|iterations|molecular gradient)\s+\.\.\.\s+0 min,\s+([0-9.]+) sec", out, re.M)}
    # the g-xTB build's tblite-style timer: " - name   x.xxx sec"
    ph.update({"tblite: " + k.strip(): float(v) for k, v in re.findall(r"^ - ([a-z ]+?)\s+([0-9.]+) sec", out, re.M)})
    tot = re.findall(r"wall-time:\s+0 d,\s+0 h,\s+0 min,\s+([0-9.]+) sec", out)
    it = re.findall(r"convergence criteria satisfied after\s+(\d+) iterations", out)
    if not it and "cycle        total energy" in out:
        cyc = re.findall(r"^\s+(\d+)\s+-?\d+\.\d+\s+-?\d\.\d+E[-+]\d+\s+\d\.\d+E[-+]\d+\s*$", out, re.M)
        it = [cyc[-1]] if cyc else []
    return {"wall": wall, "internal_total": float(tot[0]) if tot else None, "iterations": int(it[0]) if it else None, "phases": ph}


def axtb(binary, method, nums, pos, threads, env_extra):
    req = {"numbers": nums, "positions": pos, "method": method, "gradient": True, "threads": threads}
    env = {k: v for k, v in os.environ.items() if k not in ("AXTB_LAPACK", "AXTB_LAPACK_BLAS")}
    env.update(env_extra)
    t0 = time.perf_counter()
    r = subprocess.run([binary, "--json"], input=json.dumps(req), capture_output=True, text=True, env=env)
    wall = time.perf_counter() - t0
    d = json.loads(r.stdout)
    return {"wall": wall, "internal_total": d.get("seconds"), "iterations": d.get("iterations"), "phases": d.get("timings"),
            "stderr": r.stderr.strip()[:300]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--axtb", required=True)
    ap.add_argument("--xtb", default=programs.get(programs.XTB680), help="xtb 6.8.0 (default: $AXTB_XTB680)")
    ap.add_argument("--gxtb", default=programs.get(programs.GXTB2), help="the g-xTB 2.0.1 build (default: $AXTB_GXTB2)")
    ap.add_argument("--structures", required=True)
    ap.add_argument("--mols", default="taxol,C50H102,(H2O)100,(H2O)200")
    ap.add_argument("--threads", default="1,8")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--lib", action="append", default=[], help="name=path[:blas]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--methods", default="gfn2,gxtb-v2.0.1")
    ap.add_argument("--xyz", action="append", default=[], help="name=path of a molecule not under --structures")
    a = ap.parse_args()
    if not (a.xtb and a.gxtb):
        programs.require(programs.XTB680, programs.GXTB2)
    for spec in a.xyz:
        name, path = spec.split("=", 1)
        FILES[name] = os.path.abspath(path)
    libs = [("faer", {})]
    for spec in a.lib:
        name, rest = spec.split("=", 1)
        path, _, blas = rest.partition(":")
        env = {"AXTB_LAPACK": path}
        if blas:
            env["AXTB_LAPACK_BLAS"] = blas
        libs.append((name, env))
    out = open(a.out, "a")
    for method, prog, args in (("gfn2", a.xtb, ["--gfn", "2", "--grad"]), ("gxtb-v2.0.1", a.gxtb, ["--gxtb", "--grad"])):
        if method not in a.methods.split(","):
            continue
        for m in a.mols.split(","):
            xyz = os.path.join(a.structures, FILES[m])
            nums, pos = read_xyz(xyz)
            for t in (int(x) for x in a.threads.split(",")):
                # program and axtb settings interleaved, best of rounds
                best = {}
                for r in range(a.rounds):
                    order = [("program", None)] + libs
                    order = order[r % len(order):] + order[:r % len(order)]
                    for name, env in order:
                        res = program(prog, args, xyz, t) if name == "program" else axtb(a.axtb, method, nums, pos, t, env)
                        if name not in best or res["wall"] < best[name]["wall"]:
                            best[name] = res
                libenv = dict(libs)
                for name, res in best.items():
                    rec = {"method": method, "molecule": m, "threads": t, "setting": name, "load1": os.getloadavg()[0],
                           "binary": prog if name == "program" else a.axtb, "args": args if name == "program" else ["--json"],
                           "env": {} if name == "program" else libenv[name], **res}
                    out.write(json.dumps(rec) + "\n")
                    out.flush()
                    print(f"{method:12s} {m:9s} t{t} {name:8s} wall {res['wall']:.3f} it {res['iterations']}", flush=True)


if __name__ == "__main__":
    main()
