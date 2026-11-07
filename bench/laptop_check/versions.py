"""Was axtb faster before? GFN2-xTB energy + gradient, one thread: xtb 6.7.1 (release and the g-xTB build), xtb 6.8.0,
and an older and the current axtb build, interleaved with the order rotated; median wall time of whole processes."""
import json, os, statistics, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check import run, program
import final_timings as F
OUT = Path(__file__).resolve().parents[1] / 'results/laptop-check'
# the programs of tests/programs.py; the axtb CLIs of the recorded run: $AXTB_BASE_BIN (the older one) and $AXTB_BIN
P = F.programs
PROGS = {"xtb 6.7.1 release": P.get(P.XTB671), "xtb 6.7.1 (g-xTB build)": P.get(P.GXTB2), "xtb 6.8.0": P.get(P.XTB680)}
AXTBS = {"axtb 6169b18 (Nov 5, earlier report)": os.environ.get('AXTB_BASE_BIN'), "axtb 7bb7c16 (now)": os.environ.get('AXTB_BIN')}

def axtb(binary, mol, th):
    env = {k: v for k, v in os.environ.items() if not k.endswith('_NUM_THREADS')}
    return run([binary, '--json'], env, {}, stdin=json.dumps(mol.request('gfn2', th, gradient=True)))

rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
rows = []
for name in ('caffeine', 'taxol', 'C50H102', '(H2O)100'):
    mol = F.Mol(name)
    jobs = [(k, (lambda b=b: program(b, ['--gfn', '2', '--grad'], mol, 1))) for k, b in PROGS.items() if b] + \
           [(k, (lambda b=b: axtb(b, mol, 1))) for k, b in AXTBS.items() if b]
    for _, f in jobs:
        f()  # warm-up
    t = {k: [] for k, _ in jobs}
    for r in range(rounds):
        for k, f in jobs[r % len(jobs):] + jobs[:r % len(jobs)]:
            t[k].append(f()[0])
    row = {"molecule": name, "atoms": mol.natoms, "load": round(os.getloadavg()[0], 1), **{k: round(statistics.median(v), 4) for k, v in t.items()}}
    rows.append(row); print(row, flush=True)
json.dump({"how": __doc__, "machine": "laptop (Intel i9-11900H)", "threads": 1, "rows": rows}, open(OUT / 'versions.json', 'w'), indent=1)
