#!/usr/bin/env python3
"""The pass/fail rule of the validation scripts: every measured quantity against its limit.

A script collects its results in a summary (quantity -> the largest deviation, a count, or the
list of failing cases, which counts its entries) and ends with `return gate.check(summary,
LIMITS, KNOWN_DIFFERENCES)`. The limits are the numbers that docs/validation.md, the README and
the specs quote, which give two significant digits, so a value passes when it is at most its
limit at two digits; a count passes when it is at most its limit. The known differences (case
-> reason) are the cases the docs explain; the script leaves them out of the summary, and they
are printed, not gated.

    python tests/gate.py SCRIPT...   # the LIMITS and KNOWN_DIFFERENCES of each script (check.sh list)
"""

from __future__ import annotations

import ast
import sys
import textwrap
from pathlib import Path


def within(x: float | int | None, limit: float) -> bool:
    """A value (two significant digits) or count at most its limit; None (not measured) is not."""
    return x is not None and (float(f"{x:.2g}") if isinstance(x, float) else x) <= limit


def check(summary: dict, limits: dict, known: dict[str, str] | None = None) -> int:
    """Print every quantity next to its limit and the known differences; 1 if any quantity is
    over its limit or missing (None), else 0."""
    failed = []
    for name, value in summary.items():
        limit = limits[name]
        cases = value if isinstance(value, (list, tuple, set, dict)) else None
        x = len(cases) if cases is not None else value
        ok = within(x, limit)
        shown = "-" if x is None else f"{x:.3g}" if isinstance(x, float) else str(x)
        print(f"  {name:56s} {shown:>9s}  limit {limit:<8g} {'ok' if ok else 'FAIL'}"
              + (f"  {', '.join(map(str, cases))}" if cases else ""))
        if not ok:
            failed.append(name)
    for case, why in (known or {}).items():
        print(f"  known difference (not gated): {case}: {why}")
    print(f"gate {'FAIL: ' + ', '.join(failed) if failed else 'PASS'} ({len(summary)} quantities)")
    return 1 if failed else 0


def describe(path: Path) -> str:
    """The LIMITS and KNOWN_DIFFERENCES tables of a script, read without importing it."""
    tables = {}
    for node in ast.parse(path.read_text()).body:
        target = node.targets[0] if isinstance(node, ast.Assign) else getattr(node, "target", None)
        if isinstance(target, ast.Name) and target.id in ("LIMITS", "KNOWN_DIFFERENCES") and node.value is not None:
            tables[target.id] = ast.literal_eval(node.value)

    def text(d):  # a nested table per program version or part; a known difference's value is its reason
        return ", ".join(f"[{k}] {text(v)}" if isinstance(v, dict) else k if isinstance(v, str) else f"{k} ≤ {v:g}"
                         for k, v in d.items())

    if "LIMITS" not in tables:
        return "limits: the script's own checks"
    known = tables.get("KNOWN_DIFFERENCES")
    return f"limits: {text(tables['LIMITS'])}" + (f"; known differences: {text(known)}" if known else "")


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        print(textwrap.fill(describe(Path(arg)), 118, initial_indent="    ", subsequent_indent="      "))
