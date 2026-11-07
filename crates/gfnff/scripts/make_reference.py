#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Write the xtb reference data of the GFN-FF tests: `crates/gfnff/tests/reference-xtb-6.7.1/*.json`
from xtb 6.7.1 (the g-xTB 2.0.1 build, 26dd68d) and `crates/gfnff/tests/reference-xtb-6.8.0/`
from xtb 6.8.0 (the directory follows the version of `--xtb`).

Every structure comes from `tests/gfnff_structures.py`; the JSON files hold the structure
(bohr), xtb's per-term energies, gradient, EEQ charges and the topology arrays read from
`gfnff_topo`, so `cargo test` checks the port without xtb.

    python crates/gfnff/scripts/make_reference.py --xtb "$AXTB_GXTB2"    # xtb 6.7.1
    python crates/gfnff/scripts/make_reference.py --xtb "$AXTB_XTB680"   # xtb 6.8.0

`--xtb` is required: the script never takes `xtb` from PATH (tests/programs.py).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "tests"))
import gfnff_structures as GS  # noqa: E402
import gfnff_xtb as G  # noqa: E402

TESTS = ROOT / "crates" / "gfnff" / "tests"
CASES = [
    "struct-caffeine", "struct-taxol", "struct-cf3br-benzene", "struct-cu-nh3-4", "struct-ferrocene",
    "struct-zncl4", "struct-tyr-xab", "dimer-water", "dimer-pyridine-HF", "dimer-acetone-water",
    "dimer-formicacid", "xb-ICl-NH3", "xb-CF3Cl-formaldehyde", "glycine-zwitterion-water", "chg-3-PO4_3-",
    "chg+1-guanidinium", "chg-2-O2_2-", "mb16-03", "mbheavy-105", "el-57-La", "el-07-N-alt",
    "water-cluster-20",
]
TOPO_KEYS = ("nb", "blist", "alist", "tlist", "b3list", "fraglist", "hbatHl", "hbatABl", "xbatABl", "nr_hb",
             "bond_hb", "qa", "chieeq", "gameeq", "alpeeq", "hbbas", "hbaci", "vbond", "vangl", "vtors", "qfrag")


def entry(cid, nums, pos, q, ref, program, positions_eval=None):
    t = ref["topology"]
    return {
        "id": cid,
        "numbers": [int(z) for z in nums],
        "positions": np.asarray(pos, float).tolist(),
        "charge": q,
        "positions_eval": None if positions_eval is None else np.asarray(positions_eval, float).tolist(),
        "terms": ref["terms"],
        "gradient": ref["gradient"],
        "charges": ref["charges"],
        "topology": {k: t[k] for k in TOPO_KEYS},
        "program": program,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--xtb", required=True,
                    help="the xtb program: $AXTB_GXTB2 (6.7.1) or $AXTB_XTB680 (6.8.0), see tests/programs.py")
    args = ap.parse_args()
    version = re.search(r"xtb version (\S+) \((\w+)\)", subprocess.run([args.xtb, "--version"], capture_output=True, text=True).stdout)
    program = f"xtb {version.group(1)} ({version.group(2)}) --gfnff"
    out = TESTS / f"reference-xtb-{version.group(1)}"
    out.mkdir(parents=True, exist_ok=True)
    allc = {c[0]: c for c in GS.cases(include_large=False)}
    for cid in CASES:
        _, _, nums, pos, q = allc[cid]
        ref = G.run_xtb(nums, pos, q, xtb=args.xtb)
        if "error" in ref:
            raise RuntimeError(f"{cid}: {ref['error']}")
        (out / f"{cid}.json").write_text(json.dumps(entry(cid, nums, pos, q, ref, program)))
        print("wrote", cid)
    # fixed topology: caffeine and the glycine zwitterion displaced, topology of the original
    rng = np.random.default_rng(11)
    for cid in ("struct-caffeine", "glycine-zwitterion-water"):
        _, _, nums, pos, q = allc[cid]
        a = G.run_xtb(nums, pos, q, xtb=args.xtb)
        posb = pos + rng.normal(0, 0.25, pos.shape)
        b = G.run_xtb(nums, posb, q, xtb=args.xtb, topo_bytes=a["topo_bytes"])
        assert b["topology_read"]
        e = entry(cid + "-fixedtopo", nums, pos, q, b, program, positions_eval=posb)
        (out / f"{cid}-fixedtopo.json").write_text(json.dumps(e))
        print("wrote", cid + "-fixedtopo")


if __name__ == "__main__":
    main()
