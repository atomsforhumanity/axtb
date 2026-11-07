import json
from pathlib import Path
REF = Path(__file__).resolve().parents[1] / "tests"
def cases():
    for d in sorted(p for p in (REF / "reference").iterdir() if p.is_dir()):
        s = json.loads((d / "summary.json").read_text()); i = json.loads((d / "input.json").read_text())
        yield {"name": d.name, "xyz": (d / "mol.xyz").read_text(), "charge": i["charge"], "uhf": i["uhf"],
               "E11": s["total_energy"], "it11": s["scf_iterations"], "summary11": s}
    # the element-coverage molecules of reference_extra.json are 2.0.1 cases of their own
    # (v2_element_cases.stage3)
    for m in json.loads((REF / "reference_extra.json").read_text())["molecules"]:
        if m.get("source") == "tests/gxtb_elements.py":
            continue
        yield {"name": m["name"], "xyz": m["xyz"], "charge": m["charge"], "uhf": m["uhf"],
               "E11": m["total_energy"], "it11": m["scf_iterations"]}
