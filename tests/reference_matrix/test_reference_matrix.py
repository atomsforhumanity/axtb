"""Fast CI subset of the reference matrix.

    python -m pytest tests/reference_matrix -q

Runs a representative case of every category through the same checks and tolerances as
`run.py` (which runs the full matrix and writes results.json/summary.md). Cases that need
a reference program are skipped when it is not there: xtb 6.8.0 (`$AXTB_XTB680`, see
tests/programs.py; compared with axtb's `xtb` conventions) and the tblite wheel.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import cases as C  # noqa: E402
import run as R  # noqa: E402

FAST = [
    # elements across the periodic table (main group, TM, 4d/5d, lanthanide, heavy p block)
    "el-01-H", "el-02-He", "el-08-O", "el-26-Fe", "el-29-Cu-alt", "el-46-Pd", "el-54-Xe", "el-64-Gd", "el-71-Lu",
    "el-77-Ir", "el-83-Bi", "el-86-Rn", "mb16-03", "mbheavy-101",
    # charges -3 .. +3 and charged radicals
    "chg-3-PO4_3-", "chg-2-SO4_2-", "chg-1-acetate", "chg-1-O2-", "chg+1-NH4+", "chg+2-aqua-Mg", "chg+3-aqua-Al",
    "chg+3-aqua-Fe-hs", "chg+1-ion-Na",
    # multiplicities and the default spin
    "spin-H-atom", "spin-NO2-default", "spin-O2-triplet", "spin-CH2-triplet", "spin-Cr-aqua-quartet",
    "spin-Fe2-aqua-quintet", "spin-Mn-aqua-sextet",
    # solvation: neutral, cation, anion, radical anion; ALPB and GBSA; all states
    "solv-alpb-water-gsolv-acetate", "solv-alpb-water-bar1M-methylammonium", "solv-alpb-dmso-reference-O2-",
    "solv-gbsa-water-reference-acetate", "solv-gbsa-thf-bar1M-acetic_acid", "solv-alpb-water-gsolv-Mg_aqua",
    "solv-alpb-alias-n-hexane",
    # electronic temperature and accuracy on small-gap systems
    "eta-Fe4-T5000-acc1e-06", "eta-Cu5-T1000-acc0.01", "eta-Na8-T0-acc1e-06", "eta-N2-stretched-T300-acc1",
    # dispersion off vs tblite
    "nodisp-acetate", "nodisp-Fe3-aqua",
    # failure behaviour
    "fail-scf-maxiter3", "fail-Z87", "fail-uhf-parity", "fail-uhf-too-many", "fail-charge-negative-nel",
    "fail-charge-too-negative", "fail-acc-zero", "fail-unknown-solvent",
    # API surfaces and threads
    "api-cli-charged-uhf-xtb", "api-cli-alpb-bar1M-axtb", "api-calculator-sequence", "thr-Fe4-T5000",
    "thr-hess-NO2", "hess-NO2",
    # analytic Hessians: closed and open shell, solvated, fractional occupations, a metal
    "hana-water", "hana-NO2-doublet", "hana-acetate-alpb-water", "hana-ethylene-twisted-T3000",
    "hana-Cu-NH3-4-2+",
]

OK = {"pass", "both_failed", "info", "expected_difference"}
NEEDS = {"xtb": "xtb", "hessian": "xtb", "analytic_xtb": "xtb", "failure": "xtb", "tblite": "tblite",
         "tblite_info": "tblite"}


def _have(prog: str) -> bool:
    if prog == "xtb":
        return R.P.XTB is not None
    return importlib.util.find_spec("tblite") is not None


_ALL = {c["id"]: c for c in C.all_cases()}


@pytest.mark.parametrize("cid", FAST)
def test_case(cid):
    case = _ALL[cid]
    missing = [NEEDS[k] for k in case["checks"] if k in NEEDS and not _have(NEEDS[k])]
    if missing:
        pytest.skip(f"needs {', '.join(sorted(set(missing)))}")
    if "cli" in case["checks"] and not (R.S.ROOT / "target/release/axtb").exists():
        pytest.skip("needs target/release/axtb (cargo build --release -p axtb-cli)")
    res = R.evaluate(case)
    # the default conventions may converge where the program conventions, like the program,
    # do not (the DIIS fallback; mbheavy-101 since 2026-11-06, as mbheavy-112/113 in the full
    # matrix): `ref_failed` of that comparison is not a defect; the `xtb` check still compares
    # the program conventions with the program
    bad = {k: v for k, v in res["checks"].items()
           if v["status"] not in OK and not (k == "axtb_vs_xtbmode" and v["status"] == "ref_failed")}
    assert not bad, f"{cid}: {bad}"


def test_case_ids_are_valid():
    assert set(FAST) <= set(_ALL)
