#!/usr/bin/env python3
"""Does `--acc` reach the displaced SCFs of xtb's GFN2-xTB `--hess`? Runs `xtb --gfn 2 --hess`
on caffeine at --acc 1.0, at --acc 0.001, with an xcontrol `$hess sccacc=0.001`, and with no
accuracy setting at all, and compares the Hessians and lowest frequencies; also each Hessian
against axtb's analytic one (xtb conventions of that version, projected as that version
projects its `hessian` file), which has no displaced SCFs.

xtb 6.7.1 runs the displacement SCFs at `$hess sccacc` (default 0.3) whatever `--acc` says;
xtb 6.8.0 (PR #1500) uses `--acc` when it is given, `$hess sccacc` when that is given, and
otherwise the accuracy of the `--hess` run, which now defaults to 0.2 (the reference SCF too).

    python tests/xtb_hess_acc.py [--xtb PATH] [--out bench/results/xtb_gfn2_hess_acc_6.8.0.json]

`--xtb` defaults to xtb 6.8.0 (`$AXTB_XTB680`, see tests/programs.py); the 6.7.1 record ran the
g-xTB 2.0.1 build (`--xtb "$AXTB_GXTB2"`).

The default output name carries the xtb version: `xtb_gfn2_hess_acc.json` is the 6.7.1 record
(the g-xTB 2.0.1 build), `xtb_gfn2_hess_acc_6.8.0.json` xtb 6.8.0's.

The script fails (exit status 1) when an xtb Hessian is further from axtb's analytic one than its
limit in `LIMITS` (tests/gate.py), or the comparison fails.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tests/reference_matrix"))
sys.path.insert(0, str(ROOT / "tests"))
import gate  # noqa: E402
import programs  # noqa: E402

# max |H_xtb - H_analytic| (Eh/bohr^2) per setting, xtb's finite-difference error, as
# docs/validation.md (Hessians) quotes it for each version
LIMITS = {
    "6.8": {"--acc 1.0": 1.3e-3, "--acc 0.001": 3.7e-5, "$hess sccacc=0.001 (xcontrol)": 3.7e-5, "no accuracy setting": 2.7e-4},
    "6.7": {"--acc 1.0": 2.2e-4, "--acc 0.001": 2.2e-4, "$hess sccacc=0.001 (xcontrol)": 3.7e-5, "no accuracy setting": 2.2e-4},
}


def run(xtb: str, xyz: Path, extra: list[str], xcontrol: str | None = None):
    with tempfile.TemporaryDirectory(dir=os.environ.get("AXTB_SCRATCH")) as tmp:
        t = Path(tmp)
        (t / "mol.xyz").write_text(xyz.read_text())
        args = [xtb, "mol.xyz", "--gfn", "2", "--hess", *extra]
        if xcontrol:
            (t / "xc.inp").write_text(xcontrol)
            args += ["--input", "xc.inp"]
        out = subprocess.run(args, cwd=t, capture_output=True, text=True, env={**os.environ, "OMP_NUM_THREADS": "1"})
        h = np.array([float(x) for x in (t / "hessian").read_text().split()[1:]])
        block = out.stdout.split("projected vibrational frequencies")[-1]
        vals = [float(x) for ln in block.splitlines() if ln.strip().startswith("eigval :") for x in ln.split(":")[1].split()]
        low = [v for v in vals if abs(v) > 1e-3]
        ver = re.search(r"xtb version (\S+) \((\w+)\)", out.stdout)
        acc = re.findall(r"accuracy\s+(\S+)", out.stdout)
        # the molecule as xtb read it (6.7.1 rotates it slightly before the Hessian)
        return h, low, f"{ver.group(1)} ({ver.group(2)[:7]})" if ver else "unknown", acc[0] if acc else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--xtb", default=programs.get(programs.XTB680), help="the xtb binary (default: xtb 6.8.0, $AXTB_XTB680)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if not a.xtb:
        programs.skip("pass --xtb or " + programs.reason(programs.XTB680))
    xyz = ROOT / "tests/structures/caffeine.xyz"
    h1, f1, ver, acc1 = run(a.xtb, xyz, ["--acc", "1.0"])
    h2, f2, _, _ = run(a.xtb, xyz, ["--acc", "0.001"])
    h3, f3, _, acc3 = run(a.xtb, xyz, [], "$hess\n   sccacc=0.001\n$end\n")
    h4, f4, _, acc4 = run(a.xtb, xyz, [])
    rows = [
        {"setting": "--acc 0.001", "max_dH_vs_acc1": float(np.abs(h2 - h1).max()), "lowest_freqs": f2[:3]},
        {"setting": "$hess sccacc=0.001 (xcontrol)", "max_dH_vs_acc1": float(np.abs(h3 - h1).max()), "lowest_freqs": f3[:3],
         "printed_accuracy": acc3},
        {"setting": "no accuracy setting", "max_dH_vs_acc1": float(np.abs(h4 - h1).max()), "lowest_freqs": f4[:3],
         "printed_accuracy": acc4},
    ]
    # each Hessian against the exact (analytic) one, projected the way this xtb version
    # projects its `hessian` file
    try:
        os.environ["AXTB_MATRIX_XTB"] = a.xtb
        import axtb
        import runners as P
        import run as R
        lines = xyz.read_text().splitlines()
        nat = int(lines[0])
        sym = [ln.split()[0] for ln in lines[2:2 + nat]]
        nums = np.array([R.S.SYMBOLS.index(s.capitalize()) + 1 for s in sym])
        pos = np.array([[float(x) for x in ln.split()[1:4]] for ln in lines[2:2 + nat]]) / 0.52917721090380
        pos = R.rotmol(pos)
        calc = axtb.Calculator("gfn2", nums, pos, conventions=P.XTB_CONVENTIONS)
        ha = R.xtb_projected(np.asarray(calc.hessian(pos, method="analytic", accuracy=1e-6)["hessian"]), nums, pos,
                             R.xtb_masses()).ravel()
        for r, h in zip([{"setting": "--acc 1.0"}] + rows, (h1, h2, h3, h4)):
            r["max_dH_vs_analytic"] = float(np.abs(h - ha).max())
        ref_vs_analytic = float(np.abs(h1 - ha).max())
    except Exception as e:  # noqa: BLE001
        print("no analytic comparison:", e)
        ref_vs_analytic = None
    version = ver.split()[0]
    out = a.out or ROOT / ("bench/results/xtb_gfn2_hess_acc.json" if version.startswith("6.7.")
                           else f"bench/results/xtb_gfn2_hess_acc_{version}.json")
    out.write_text(json.dumps({"program": f"xtb {ver}", "xtb_version": version, "method": "GFN2-xTB",
                               "molecule": "caffeine (tests/structures, not optimized)",
                               "reference": {"setting": "--acc 1.0", "lowest_freqs": f1[:3], "printed_accuracy": acc1,
                                             "max_dH_vs_analytic": ref_vs_analytic},
                               "rows": rows, "load_average": [round(x, 2) for x in os.getloadavg()]}, indent=1) + "\n")
    print(json.dumps(rows, indent=1))
    print("wrote", out)
    limits = LIMITS[version[:3]]
    # None (a failure) without the analytic comparison
    summary = {"--acc 1.0": ref_vs_analytic} | {r["setting"]: r.get("max_dH_vs_analytic") for r in rows}
    return gate.check(summary, limits)


if __name__ == "__main__":
    raise SystemExit(main())
