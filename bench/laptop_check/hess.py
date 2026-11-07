"""Quiet-laptop Hessian sanity check: axtb analytic vs the programs at --acc 0.001, one thread, one round after a warm-up."""
import json, os, sys
from pathlib import Path
OUT = Path(__file__).resolve().parents[1] / "results/laptop-check"
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from check import run, program, AXTB, XTB680, GXTB
import final_timings as F
def axtb_h(method, mol, th):
    env = {k: v for k, v in os.environ.items() if not k.endswith('_NUM_THREADS')}
    return run([AXTB, '--json'], env, {}, stdin=json.dumps(mol.request(method, th, hessian=True)))
out = []
for label, meth, binary, args, name in [('GFN2-xTB', 'gfn2', XTB680, ['--gfn', '2', '--hess', '--acc', '0.001'], 'caffeine'),
                                        ('GFN2-xTB', 'gfn2', XTB680, ['--gfn', '2', '--hess', '--acc', '0.001'], 'taxol'),
                                        ('g-xTB 2.0.1', 'gxtb-v2.0.1', GXTB, ['--gxtb', '--hess', '--acc', '0.001'], 'caffeine')]:
    mol = F.Mol(name)
    for th in (1, 8):
        p = program(binary, args, mol, th); a = axtb_h(meth, mol, th)
        row = dict(method=label, molecule=name, threads=th, program_s=round(p[0], 2), axtb_s=round(a[0], 3), speedup=round(p[0] / a[0], 1),
                   program_cores=round(p[1], 2), axtb_cores=round(a[1], 2), load=os.getloadavg()[0])
        out.append(row); print(row, flush=True)
json.dump(out, open(OUT / 'hess.json', 'w'), indent=1)
