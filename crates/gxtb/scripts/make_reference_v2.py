#!/usr/bin/env python3
"""Generate g-xTB 2.0.1 reference data under crates/gxtb/tests/reference_v2/.

Usage: python3 crates/gxtb/scripts/make_reference_v2.py <xtb 2.0.1 binary> <output dir>

The g-xTB 2.0.1 program is xtb 6.7.1 with a modified tblite, statically linked (grimme-lab/g-xtb
release, GPL-3.0); parameters are compiled in. Every run uses a fresh directory with HOME set to it,
`OMP_NUM_THREADS=1` and `--norestart` (otherwise xtb reads `xtbrestart` and starts from the old
wavefunction). `--uhf` is passed only for open shells: any explicit `--uhf` (even 0) switches the
program to an unrestricted calculation. For every molecule:

1. `xtb mol.xyz --gxtb --norestart --chrg Q [--uhf U] --grad`                 -> xtb.out, gradient, energy
2. the same with `--acc 0.0001` (energy 1e-10 Eh, Fock residual 2e-9)           -> gradient_tight, energy_tight
3. the same as 1 under gdb: the energy of every term at the last SCF iteration -> summary.json "terms"

Host names and home paths are removed from the stored output (banner build line, program path).
Requires gdb for step 3 (set NO_GDB=1 to skip). ONLY=name1,name2 regenerates only those molecules
and updates their entries in an existing index.json.
"""
from __future__ import annotations
import json, os, re, shutil, subprocess, sys, tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

TERMS_GDB = r'''
set pagination off
set confirm off
python
import gdb, struct, json
inf = lambda: gdb.selected_inferior()
rd = lambda a, n: bytes(inf().read_memory(a, n))
def u64(a): return struct.unpack("<Q", rd(a, 8))[0]
def i32(a): return struct.unpack("<i", rd(a, 4))[0]
def reg(n): return int(gdb.parse_and_eval("$" + n)) & 0xFFFFFFFFFFFFFFFF
def arg(i):
    regs = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]
    return reg(regs[i]) if i < 6 else u64(reg("rsp") + 8 * (i - 5))
def desc(a):
    base, elen, off, flags, rank = struct.unpack("<5Q", rd(a, 40))
    return base, elen, rank, [struct.unpack("<qqq", rd(a + 48 + 24 * k, 24)) for k in range(rank)]
def dsum(a):
    base, elen, rank, dims = desc(a)
    n = dims[0][0]
    return sum(struct.unpack("<%dd" % n, rd(base, 8 * n)))
NAT = [None]; log = []
def cands():
    out = []
    for i in range(8):
        try:
            base, elen, rank, dims = desc(arg(i))
            if rank == 1 and elen == 8 and dims[0][0] == NAT[0] and dims[0][1] == 8: out.append(arg(i))
        except Exception: pass
    return out
class Fin(gdb.FinishBreakpoint):
    def __init__(self, s): super().__init__(gdb.newest_frame(), internal=True); self.s = s
    def stop(self):
        tag, c, before = self.s
        log.append((tag, [dsum(a) - b for a, b in zip(c, before)])); return False
    def out_of_scope(self): pass
class E(gdb.Breakpoint):
    def stop(self):
        c = cands(); Fin((self.tag, c, [dsum(a) for a in c])); return False
class Mark(gdb.Breakpoint):
    def stop(self):
        if self.tag == "NAT":
            if NAT[0] is None: NAT[0] = i32(arg(1))
        else: log.append((self.tag, []))
        return False
for fn, tag in (("tblite_xtb_singlepoint_mp_xtb_singlepoint_", "NAT"), ("tblite_scf_iterator_mp_next_scf_", "SCF"), ("tblite_xtb_coulomb_mp_get_gradient_", "GRAD")):
    m = Mark(fn, internal=True); m.tag = tag
for fn, tag in {"tblite_coulomb_firstorder_mp_get_energy_": "first_order", "tblite_coulomb_charge_type_mp_get_energy_": "second_order",
                "tblite_coulomb_thirdorder_twobody_mp_get_energy_": "third_order", "tblite_coulomb_fourthorder_mp_get_energy_": "fourth_order",
                "tblite_coulomb_multipole_type_mp_get_energy_": "aes", "tblite_spin_mp_get_energy_": "spin",
                "tblite_disp_d4_mp_get_energy_": "dispersion_sc", "tblite_exchange_type_mp_get_energy_w_overlap_": "exchange",
                "tblite_repulsion_gxtb_mp_get_energy_": "repulsion", "tblite_classical_increment_mp_get_engrad_": "increment",
                "tblite_disp_d4_mp_get_engrad_": "dispersion_nonsc"}.items():
    b = E(fn, internal=True); b.tag = tag
end
run
python
json.dump(log, open("terms.json", "w"))
end
'''


LARGE = ("c50h102", "water100")  # tests/structures/large: cross the 25/50-bohr cutoffs


def molecules():
    import v2_reference_cases as cases, v2_element_cases as elements
    for c in cases.cases():
        yield c["name"], c["xyz"], c["charge"], c["uhf"], "reference"
    for c in elements.cases():
        yield c["name"], c["xyz"], c["charge"], c["uhf"], "elements"
    for name in LARGE:
        yield name, (HERE.parents[2] / "tests/structures/large" / f"{name}.xyz").read_text(), 0, 0, "large"


def scrub(text: str, binary: str) -> str:
    text = text.replace(binary, "xtb")
    text = "\n".join(l for l in text.splitlines()
                     if "compiled by" not in l and not re.search(r"/(home|Users)/", l) and "directory" not in l)
    return text + "\n"


def run(binary, workdir, xyz, charge, uhf, extra, gdb=False):
    for f in ("xtbrestart", "gradient", "energy", "charges", "terms.json"):
        (workdir / f).unlink(missing_ok=True)
    (workdir / "mol.xyz").write_text(xyz)
    env = dict(os.environ, OMP_NUM_THREADS="1", OMP_STACKSIZE="1G", HOME=str(workdir))
    cmd = [binary, "mol.xyz", "--gxtb", "--norestart", "--chrg", str(charge), *(("--uhf", str(uhf)) if uhf else ()), *extra]
    if gdb:
        (workdir / "terms.gdb").write_text(TERMS_GDB)
        cmd = ["gdb", "-q", "-batch", "-x", "terms.gdb", "--args", *cmd]
    p = subprocess.run(cmd, cwd=workdir, env=env, capture_output=True, text=True)
    return p.stdout + p.stderr, p.returncode


def parse(out):
    s = {}
    for lab, key in (("total energy", "total_energy"), ("electronic energy", "electronic_energy"), ("core increment energy", "increment"),
                     ("dispersion energy", "dispersion_nonsc"), ("number of electrons", "nel"), ("integral cutoff", "integral_cutoff"),
                     ("energy convergence", "econv"), ("density convergence", "pconv")):
        m = re.search(lab + r"\s+(-?\d\.\d+E[+-]\d+)", out)
        if m: s[key] = float(m.group(1))
    cyc = re.findall(r"^\s+(\d+)\s+(-?\d+\.\d+)\s+(-?\d\.\d+E[+-]\d+)\s+(-?\d\.\d+E[+-]\d+)\s*$", out, re.M)
    s["scf_cycles"] = [[int(a), float(b), float(c), float(d)] for a, b, c, d in cyc]
    s["scf_iterations"] = len(cyc)
    m = re.search(r"HL-Gap\s+(-?\d+\.\d+) Eh", out); s["gap_eh"] = float(m.group(1)) if m else None
    q = re.search(r"#\s+Z\s+q\s+p\(s\).*?\n-+\n(.*?)\n-+", out, re.S)
    if q: s["charges"] = [float(l.split()[3]) for l in q.group(1).splitlines() if l.strip()]
    return s


def terms(log):
    pre, its, cur, phase = {}, [], None, "pre"
    for tag, v in log:
        if tag == "SCF": cur = {}; its.append(cur); phase = "scf"; continue
        if tag == "GRAD": phase = "grad"; continue
        if phase == "pre": pre[tag] = pre.get(tag, 0.0) + (v[0] if v else 0.0)
        elif phase == "scf" and v: cur[tag] = v[0]
    return {**pre, **(its[-1] if its else {})}


def main():
    binary, outdir = sys.argv[1], Path(sys.argv[2])
    outdir.mkdir(parents=True, exist_ok=True)
    only = [x for x in os.environ.get("ONLY", "").split(",") if x]
    index = json.loads((outdir / "index.json").read_text()) if only and (outdir / "index.json").exists() else []
    for name, xyz, charge, uhf, group in molecules():
        if only and name not in only:
            continue
        index = [e for e in index if e["name"] != name]
        d = outdir / group / re.sub(r"[^A-Za-z0-9_.-]", "_", name)
        d.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory() as tmp:
            w = Path(tmp)
            out, rc = run(binary, w, xyz, charge, uhf, ["--grad"])
            (d / "xtb.out").write_text(scrub(out, binary))
            for f in ("gradient", "energy"):
                if (w / f).exists(): shutil.copy(w / f, d / f)
            s = parse(out); s["returncode"] = rc
            out_t, rc_t = run(binary, w, xyz, charge, uhf, ["--grad", "--acc", "0.0001"])
            st = parse(out_t)
            for f, g in (("gradient", "gradient_tight"), ("energy", "energy_tight")):
                if (w / f).exists(): shutil.copy(w / f, d / g)
            s["tight"] = {k: st.get(k) for k in ("total_energy", "scf_iterations", "econv", "pconv")}
            if not os.environ.get("NO_GDB"):
                run(binary, w, xyz, charge, uhf, [], gdb=True)
                if (w / "terms.json").exists():
                    s["terms"] = terms(json.loads((w / "terms.json").read_text()))
        (d / "mol.xyz").write_text(xyz)
        (d / "input.json").write_text(json.dumps({"charge": charge, "uhf": uhf, "program": "g-xTB 2.0.1 (xtb 6.7.1 + tblite, grimme-lab/g-xtb)",
            "command": "xtb mol.xyz --gxtb --norestart --chrg Q [--uhf U] --grad [--acc 0.0001]; OMP_NUM_THREADS=1, HOME=<workdir>"}, indent=1))
        (d / "summary.json").write_text(json.dumps(s, indent=1))
        index.append({"name": name, "group": group, "charge": charge, "uhf": uhf, "total_energy": s.get("total_energy"),
                      "scf_iterations": s.get("scf_iterations"), "tight_energy": s["tight"].get("total_energy")})
        print(name, s.get("total_energy"), s.get("scf_iterations"), flush=True)
    order = {(g, n): k for k, (n, _, _, _, g) in enumerate(molecules())}
    index.sort(key=lambda e: order.get((e["group"], e["name"]), len(order)))
    (outdir / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
