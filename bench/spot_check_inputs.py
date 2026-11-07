"""Inputs for bench/spot_check.sh: a Turbomole coord file and axtb JSON requests per cell, written
the way bench/final_timings.py writes them.  usage: spot_check_inputs.py OUTDIR"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "bench"))
import final_timings as F  # noqa: E402

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
CELLS = [("gfn2", "(H2O)100", 1), ("gfn2", "(H2O)200", 8), ("gxtb2", "taxol", 4)]
names = {"(H2O)100": "w100", "(H2O)200": "w200", "taxol": "taxol"}
for part, mol, t in CELLS:
    m = F.Mol(mol)
    tag = f"{part}-{names[mol]}-t{t}"
    (out / f"{names[mol]}.coord").write_text(m.coord())
    (out / f"{tag}.json").write_text(json.dumps(m.request(F.METHODS[part], t, gradient=True)))
    print(tag)
