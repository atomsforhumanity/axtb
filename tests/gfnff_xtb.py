"""Run xtb's GFN-FF and axtb's port on the same structure and compare them.

Used by `tests/gfnff_reproduction.py` (report) and the GFN-FF tests. xtb gets the
coordinates in bohr (Turbomole `coord` file), so both programs see identical numbers;
each xtb run happens in its own fresh directory with one thread and an unlimited stack.

What is read from xtb (6.7.1 or 6.8.0, GFN-FF version 1.0.3 = `angewChem2020_2`):

* `gfnff_topo` (unformatted Fortran file written by `write_restart_gff`): every topology
  array, doubles bit for bit;
* `gfnff_lists.json` (`--json`): neighbor lists, bond-pair matrix, EEQ charges (15 decimals);
* standard output: the per-term energies (12 decimals) and the hybridization table;
* `gradient` (Turbomole format, 14 significant digits), `hessian` (`--hess`).
"""
from __future__ import annotations

import json
import os
import re
import resource
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SYMBOLS = ("H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn "
           "Ga Ge As Se Br Kr Rb Sr Y Zr Nb Mo Tc Ru Rh Pd Ag Cd In Sn Sb Te I Xe Cs Ba La Ce "
           "Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu Hf Ta W Re Os Ir Pt Au Hg Tl Pb Bi Po At Rn "
           "Fr Ra Ac Th Pa U Np Pu Am Cm Bk Cf Es Fm Md No Lr").split()
TERMS = {
    "bond energy": "bond", "angle energy": "angle", "torsion energy": "torsion",
    "repulsion energy": "repulsion", "electrostat energy": "electrostatic",
    "dispersion energy": "dispersion", "HB energy": "hb", "XB energy": "xb",
    "bonded atm energy": "batm", "external energy": "external", "total energy": "total",
}


def read_xyz(path, aatoau: float = 1.0 / 0.529177210544):
    lines = Path(path).read_text().splitlines()
    n = int(lines[0])
    nums, pos = [], []
    for ln in lines[2:2 + n]:
        p = ln.split()
        sym = p[0].capitalize()
        nums.append(int(sym) if sym.isdigit() else SYMBOLS.index(sym) + 1)
        pos.append([float(x) * aatoau for x in p[1:4]])
    m = re.search(r"charge\s*=\s*(-?\d+)", lines[1]) if len(lines) > 1 else None
    return np.array(nums), np.array(pos), (int(m.group(1)) if m else 0)


def write_coord(path, numbers, positions_bohr):
    with open(path, "w") as f:
        f.write("$coord\n")
        for z, p in zip(numbers, positions_bohr):
            f.write(f"{float(p[0])!r:>26} {float(p[1])!r:>26} {float(p[2])!r:>26}  {SYMBOLS[int(z) - 1].lower()}\n")
        f.write("$end\n")


def _unlimited_stack():
    try:
        resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    except (ValueError, OSError):
        soft, hard = resource.getrlimit(resource.RLIMIT_STACK)
        resource.setrlimit(resource.RLIMIT_STACK, (hard, hard))


def records(path):
    data = Path(path).read_bytes()
    pos, out = 0, []
    while pos < len(data):
        (n,) = struct.unpack("<i", data[pos:pos + 4])
        pos += 4
        out.append(data[pos:pos + n])
        pos += n
        (n2,) = struct.unpack("<i", data[pos:pos + 4])
        pos += 4
        assert n == n2
    return out


def parse_topo(path):
    """Arrays of xtb's `gfnff_topo` (molecules: numctr = 1), 1-based indices as in xtb."""
    r = records(path)
    i4 = lambda b: np.frombuffer(b, dtype="<i4")
    version, nat = struct.unpack("<qq", r[0])
    (nangl, ntors, nstors, nathbH, nathbAB, natxbAB, nbatm, nfrag, nsystem, maxsystem) = i4(r[1])
    (nbond_blist, nbond_vbond, nangl_alloc, ntors_alloc, bond_hb_nr, b_max) = i4(r[2])
    numnb, numctr, nbond, itrdim = i4(r[3])
    n = int(nat)
    t = dict(version=int(version), nat=n, nangl=int(nangl), ntors=int(ntors), nstors=int(nstors),
             nathbH=int(nathbH), nathbAB=int(nathbAB), natxbAB=int(natxbAB), nbatm=int(nbatm),
             nfrag=int(nfrag), nbond=int(nbond), bond_hb_nr=int(bond_hb_nr), b_max=int(b_max))
    t["blist"] = i4(r[5]).reshape(nbond, 3)[:, :2].tolist()
    t["alist"] = i4(r[6]).reshape(nangl_alloc, 5)[:nangl, :3].tolist()
    t["tlist"] = i4(r[7]).reshape(ntors_alloc, 8)[:ntors, :5].tolist()
    t["b3list"] = i4(r[8]).reshape(-1, 5)[:nbatm, :3].tolist()
    t["stors"] = i4(r[9]).reshape(-1, 6).tolist() if nstors else []
    t["fraglist"] = i4(r[10]).tolist()
    hl = i4(r[11]).reshape(n, 2)
    t["hbatHl"] = hl[:nathbH, 0].tolist()
    t["hbatABl"] = i4(r[12]).reshape(-1, 2)[:nathbAB].tolist()
    t["xbatABl"] = i4(r[13]).reshape(-1, 5)[:natxbAB, :3].tolist() if natxbAB else []
    ah = i4(r[16]).reshape(-1, 4) if bond_hb_nr else np.zeros((0, 4), int)
    bb = i4(r[17]).reshape(bond_hb_nr, b_max, 2) if bond_hb_nr else np.zeros((0, 0, 2), int)
    bn = i4(r[18]).tolist() if bond_hb_nr else []
    t["bond_hb"] = [{"a": int(ah[k, 0]), "h": int(ah[k, 1]), "b": [int(bb[k, m, 0]) for m in range(bn[k])]}
                    for k in range(bond_hb_nr)]
    t["nr_hb"] = i4(r[19]).tolist()
    f8 = np.frombuffer(r[22], dtype="<f8")
    off = 0

    def take(m):
        nonlocal off
        v = f8[off:off + m]
        off += m
        return v

    t["vangl"] = take(2 * nangl_alloc).reshape(nangl_alloc, 2)[:nangl].tolist()
    t["vtors"] = take(2 * ntors_alloc).reshape(ntors_alloc, 2)[:ntors].tolist()
    t["chieeq"] = take(n).tolist()
    t["gameeq"] = take(n).tolist()
    t["alpeeq"] = take(n).tolist()
    alphanb = take(n * n * (numctr + 1)).reshape(numctr + 1, n, n)[0]  # alphanb(i,j) at [j, i]
    t["alphanb"] = [alphanb[j, i] for i in range(n) for j in range(i + 1)]
    t["qa"] = take(n).tolist()
    take(3 * n)
    t["zetac6"] = take(n * (n + 1) // 2).tolist()
    t["qfrag"] = take(n).tolist()
    t["hbbas"] = take(n).tolist()
    t["hbaci"] = take(n).tolist()
    b = r[23]
    vb = np.frombuffer(b[:24 * nbond], dtype="<f8").reshape(nbond, 3)
    t["vbond"] = vb.tolist()
    o = 24 * nbond
    nb = np.frombuffer(b[o:o + 4 * numnb * n * numctr], dtype="<i4").reshape(numctr, n, numnb)[0]
    t["nb"] = [sorted(nb[i, :nb[i, numnb - 1]].tolist()) for i in range(n)]
    t["cluster_tag"] = [int(nb[i, numnb - 2]) for i in range(n)]
    o += 4 * numnb * n * numctr + 4 * 3 * nbond + 4 * nbond + 24 * nbond
    bp = np.frombuffer(b[o:o + 4 * n * n * numctr], dtype="<i4").reshape(numctr, n, n)[0]
    t["bpair"] = bp.tolist()  # bpair(j, i) at [i][j] (row = centre i)
    return t


def parse_stdout(text: str):
    out = {}
    for line in text.splitlines():
        m = re.match(r"\s*::\s*(.+?)\s{2,}(-?\d+\.\d+)\s+Eh", line)
        if m and m.group(1).strip() in TERMS:
            out[TERMS[m.group(1).strip()]] = float(m.group(2))
    hyb = {}
    table = re.search(r"atom\s+neighbors\s+erfCN.*?\n(.*?)\n\s*\n", text, re.S)
    if table:
        for ln in table.group(1).splitlines():
            p = ln.split()
            if len(p) >= 9 and p[0].isdigit():
                hyb[int(p[0])] = {"nb": int(p[2]), "hyb": int(p[5]), "imet": int(p[6]), "pi": int(p[7]),
                                  "qest": float(p[8])}
    nhb = re.search(r"Number of HB bonds \(bound hydrogen\)\s+(\d+)", text)
    return out, hyb, (int(nhb.group(1)) if nhb else None)


def _fortran_float(x: str) -> float:
    """A Fortran real as printed, also with a three-digit exponent and no exponent letter
    (`-1.8053419329064-111`, xtb 6.8.0 for tiny gradient components)."""
    x = x.replace("D", "E")
    return float(re.sub(r"(?<=[0-9.])([+-]\d{3})$", r"E\1", x))


def parse_gradient(path, n):
    lines = Path(path).read_text().splitlines()
    g = []
    for ln in lines[2 + n:2 + 2 * n]:
        g.append([_fortran_float(x) for x in ln.split()])
    return g


def parse_hessian(path, n):
    vals = []
    for ln in Path(path).read_text().splitlines():
        if ln.startswith("$"):
            continue
        vals.extend(float(x.replace("D", "E")) for x in ln.split())
    return np.array(vals).reshape(3 * n, 3 * n)


def run_xtb(numbers, positions, charge=0, hess=False, workdir=None, xtb="xtb", keep=False, timeout=3600,
            topo_bytes=None):
    """xtb --gfnff on the structure; returns a dict with energies, gradient, charges, topology.
    `topo_bytes`: a `gfnff_topo` file to start from (xtb then reuses that topology)."""
    tmp = Path(tempfile.mkdtemp(prefix="gfnff_", dir=workdir))
    try:
        write_coord(tmp / "coord", numbers, positions)
        if topo_bytes is not None:
            (tmp / "gfnff_topo").write_bytes(topo_bytes)
        env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OMP_STACKSIZE": "1G"}
        args = [xtb, "coord", "--gfnff", "--chrg", str(int(charge)), "--json"]
        args += ["--hess"] if hess else ["--grad"]
        p = subprocess.run(args, cwd=tmp, capture_output=True, text=True, env=env, timeout=timeout,
                           preexec_fn=_unlimited_stack)
        n = len(numbers)
        res = {"returncode": p.returncode}
        terms, hyb, nhb1 = parse_stdout(p.stdout)
        res["terms"], res["hyb"], res["nhb1"] = terms, hyb, nhb1
        if p.returncode != 0 or "total" not in terms:
            res["error"] = (p.stdout + p.stderr)[-3000:]
            return res
        if (tmp / "gfnff_topo").exists():
            res["topology"] = parse_topo(tmp / "gfnff_topo")
            res["topo_bytes"] = (tmp / "gfnff_topo").read_bytes()
            res["topology_read"] = "topology read from file successfully" in p.stdout
        if (tmp / "gfnff_lists.json").exists():
            # xtb prints uninitialized list entries of empty lists as `********`
            lists = json.loads(re.sub(r"\*{3,}", "null", (tmp / "gfnff_lists.json").read_text()))
            res["charges"] = [v[0] for v in lists["eeq"]]
            res["lists"] = {k: lists[k] for k in ("hbl", "hbb", "xb") if k in lists}
        if (tmp / "gradient").exists():
            res["gradient"] = parse_gradient(tmp / "gradient", n)
        if hess and (tmp / "hessian").exists():
            res["hessian"] = parse_hessian(tmp / "hessian", n)
        if keep:
            res["dir"] = str(tmp)
        return res
    finally:
        if not keep:
            shutil.rmtree(tmp, ignore_errors=True)


def xtb_conventions(xtb: str = "xtb") -> str:
    """axtb's conventions for an xtb binary: `xtb-6.7.1` for xtb 6.7.x (also the g-xTB 2.0.1
    build), `xtb` (6.8.0) otherwise."""
    try:
        v = subprocess.run([xtb, "--version"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.TimeoutExpired):
        return "xtb"
    m = re.search(r"xtb version (\S+)", v)
    return "xtb-6.7.1" if m and m.group(1).startswith("6.7.") else "xtb"


def axtb_dump(numbers, positions, charge=0, exe=None, gradient=True, topology=True, threads=1,
              reference_positions=None, conventions="xtb"):
    exe = exe or str(ROOT / "target" / "release" / "examples" / "gfnff_dump")
    req = {"numbers": [int(z) for z in numbers], "positions": [list(map(float, p)) for p in positions],
           "charge": float(charge), "gradient": gradient, "topology": topology, "threads": threads,
           "conventions": conventions,
           "reference_positions": None if reference_positions is None else
           [list(map(float, p)) for p in reference_positions]}
    p = subprocess.run([exe], input=json.dumps(req), capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-2000:])
    return json.loads(p.stdout)


def _maxdiff(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.shape != b.shape:
        return float("inf")
    return float(np.max(np.abs(a - b))) if a.size else 0.0


def compare_topology(ref: dict, ours: dict) -> dict:
    """Differences between xtb's and axtb's topology: integer lists must match exactly,
    reals are reported as max |difference|."""
    t, o = ref["topology"], ours["topology"]
    out = {}
    out["nb"] = t["nb"] == [sorted(x) for x in o["nb"]]
    n = len(t["nb"])
    out["bpair"] = all(t["bpair"][i][j] == o["bpair"][i][j] for i in range(n) for j in range(n))
    for key in ("blist", "alist", "tlist", "b3list", "fraglist", "hbatHl", "hbatABl", "xbatABl", "nr_hb"):
        out[key] = [list(x) if isinstance(x, (list, tuple)) else x for x in t[key]] == \
                   [list(x) if isinstance(x, (list, tuple)) else x for x in o[key]]
    out["stors"] = (not o["stors"] and all(all(v == 0 for v in s) for s in t["stors"])) or t["stors"] == o["stors"]
    out["bond_hb"] = t["bond_hb"] == o["bond_hb"]
    for key in ("qa", "chieeq", "gameeq", "alpeeq", "hbbas", "hbaci", "alphanb", "zetac6", "vbond", "vangl", "vtors"):
        out[key] = _maxdiff(t[key], o[key])
    out["qfrag"] = _maxdiff(t["qfrag"][:len(o["qfrag"])], o["qfrag"])
    return out


def compare_terms(ref: dict, ours: dict) -> dict:
    d = {k: abs(ref["terms"][k] - (ours["energy"] if k == "total" else ours["terms"].get(k, 0.0)))
         for k in ref["terms"]}
    return d
