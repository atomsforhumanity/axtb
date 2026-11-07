#!/usr/bin/env python3
"""Side-by-side timing attribution table from attrib.py output: per method, molecule and
thread count the program's wall time and --verbose breakdown (integrals+classical, SCC
iterations, gradient) next to axtb's (setup, SCF iterations incl. diagonalization, gradient)
for every eigensolver setting. Markdown.

usage: attribution_table.py attribution.jsonl"""
import json, sys
from collections import OrderedDict

rows = [json.loads(l) for l in open(sys.argv[1])]
by = OrderedDict()
for r in rows:
    by.setdefault((r["method"], r["molecule"], r["threads"]), {})[r["setting"]] = r


def ax_phases(r):
    ph = dict(r["phases"] or [])
    if r is None:
        return None
    if "setup" in ph:  # g-xTB 2.0.1
        setup = ph.get("setup", 0.0)
        scf = ph.get("scf", 0.0)
        grad = ph.get("gradient", 0.0)
    else:  # GFN2-xTB
        setup = ph.get("classical", 0.0) + ph.get("integrals", 0.0)
        scf = sum(v for k, v in ph.items() if k.startswith("scf:")) + ph.get("cholesky", 0.0)
        grad = ph.get("gradient", 0.0)
    diag = ph.get("scf: diagonalization", 0.0)
    return setup, scf, diag, grad


settings = []
for v in by.values():
    for s in v:
        if s != "program" and s not in settings:
            settings.append(s)
hdr = "| method | molecule | thr | program: wall (setup / SCC iter. / gradient), it | " + " | ".join(f"axtb {s}: wall (setup / SCF [diag] / gradient), it" for s in settings) + " |"
print(hdr)
print("|" + "---|" * (4 + len(settings)))
for (meth, mol, t), v in by.items():
    p = v.get("program")
    if p:
        ph = p["phases"]
        if "tblite: scc" in ph:
            setup = sum(ph.get("tblite: " + k, 0) for k in ("increment", "repulsion", "dispersion", "coulomb", "exchange", "hamiltonian", "acp"))
            pc = f"{p['wall']:.3f} ({setup:.3f} / {ph['tblite: scc']:.3f} / {ph.get('tblite: gradient', 0):.3f}), {p['iterations']}"
        else:
            pc = f"{p['wall']:.3f} ({ph.get('integral evaluation', 0) + ph.get('classical contributions', 0):.3f} / {ph.get('iterations', 0):.3f} / {ph.get('molecular gradient', 0):.3f}), {p['iterations']}"
    else:
        pc = ""
    cells = []
    for s in settings:
        r = v.get(s)
        if not r:
            cells.append("")
            continue
        setup, scf, diag, grad = ax_phases(r)
        cells.append(f"{r['wall']:.3f} ({setup:.3f} / {scf:.3f} [{diag:.3f}] / {grad:.3f}), {r['iterations']}")
    print(f"| {meth} | {mol} | {t} | {pc} | " + " | ".join(cells) + " |")
