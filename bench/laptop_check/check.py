"""Quiet-laptop sanity check: axtb CLI vs the programs, energy + gradient, interleaved; wall time and cores used."""
import json, os, resource, subprocess, sys, tempfile, time, statistics
from pathlib import Path
OUT = Path(__file__).resolve().parents[1] / 'results/laptop-check'
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import final_timings as F
# the programs of tests/programs.py ($AXTB_XTB680, $AXTB_GXTB2) and the CLI ($AXTB_BIN), as final_timings.py
AXTB, XTB680, GXTB = F.AXTB, F.XTB680, F.GXTB2
F.programs.require(F.programs.XTB680, F.programs.GXTB2)

def run(cmd, env, files, stdin=None):
    with tempfile.TemporaryDirectory() as d:
        for k, v in files.items():
            Path(d, k).write_text(v)
        r0 = resource.getrusage(resource.RUSAGE_CHILDREN); t = time.perf_counter()
        p = subprocess.run(cmd, cwd=d, env=env, input=stdin, capture_output=True, text=True)
        wall = time.perf_counter() - t; r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
        cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
        if p.returncode:
            raise SystemExit(f'{cmd[0]} failed: {p.stderr[-400:]} {p.stdout[-400:]}')
        return wall, cpu / wall

def program(binary, args, mol, th):
    env = F.omp_env(th, binary, home='.')
    return run([binary, 'coord', *args], env, {'coord': mol.coord()})

def axtb(method, mol, th):
    env = {k: v for k, v in os.environ.items() if not k.endswith('_NUM_THREADS')}
    req = mol.request(method, th, gradient=True)
    return run([AXTB, '--json'], env, {}, stdin=json.dumps(req))

def main():
    cases = [('g-xTB 2.0.1', 'gxtb-v2.0.1', GXTB, ['--gxtb', '--grad'], m) for m in ('taxol', 'C50H102', '(H2O)100')] + \
            [('GFN2-xTB', 'gfn2', XTB680, ['--gfn', '2', '--grad'], m) for m in ('taxol', '(H2O)100', '(H2O)200')]
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    out = []
    for label, meth, binary, args, name in cases:
        mol = F.Mol(name)
        for th in (1, 8):
            program(binary, args, mol, th); axtb(meth, mol, th)  # warm-up
            P, A = [], []
            for r in range(rounds):
                order = [('p', lambda: program(binary, args, mol, th)), ('a', lambda: axtb(meth, mol, th))]
                for k, f in (order if r % 2 == 0 else order[::-1]):
                    (P if k == 'p' else A).append(f())
            pm, am = statistics.median(w for w, _ in P), statistics.median(w for w, _ in A)
            row = dict(method=label, molecule=name, threads=th, program_s=round(pm, 3), axtb_s=round(am, 3), speedup=round(pm / am, 2),
                       program_cores=round(statistics.median(c for _, c in P), 2), axtb_cores=round(statistics.median(c for _, c in A), 2),
                       load=os.getloadavg()[0])
            out.append(row); print(row, flush=True)
    json.dump(out, open(OUT / 'eg.json', 'w'), indent=1)

if __name__ == "__main__":
    main()
