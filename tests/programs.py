"""The external programs of axtb's validation scripts and benchmarks, one environment variable each.

| variable | program |
|---|---|
| ``AXTB_XTB680`` | xtb 6.8.0, the grimme-lab/xtb v6.8.0 release binary (``xtb version 6.8.0 (4b8c987)``): GFN2-xTB and GFN-FF in conventions ``"xtb"`` |
| ``AXTB_GXTB2`` | the g-xTB 2.0.1 build of xtb from the grimme-lab/g-xtb 2.0.1 release (``xtb version 6.7.1 (26dd68d)``): ``xtb --gxtb``, and GFN2-xTB and GFN-FF in conventions ``"xtb-6.7.1"`` |
| ``AXTB_XTB671`` | the xtb 6.7.1 release (``edcfbbe``; e.g. the copy that ships with ORCA 6.0), which differs from the g-xTB build in GFN2-xTB gradients |
| ``AXTB_GXTB11_DIR`` | a directory with the g-xTB 1.1.0 program of the grimme-lab/g-xtb v1.1.0 release and its parameter files: ``gxtb``, ``gxtb_parameters``, ``basisq``, ``eeq`` |

A variable holds a path (or, for the binaries, a command name on ``PATH``). Nothing else is
searched: no default location, and not ``xtb`` on ``PATH``, so that a script never compares with
another version than the one it names. When a variable is not set, or names nothing usable,
the cases that need the program are skipped with a message that names the variable; a script
that needs it for everything exits with status 77 (``SKIP``), which ``scripts/check.sh`` reports
as skipped.

``python tests/programs.py`` prints what each variable resolves to; ``--have VAR`` answers with
the exit status.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

XTB680 = "AXTB_XTB680"
GXTB2 = "AXTB_GXTB2"
XTB671 = "AXTB_XTB671"
GXTB11_DIR = "AXTB_GXTB11_DIR"

DESCRIPTION = {
    XTB680: "xtb 6.8.0 (the release binary)",
    GXTB2: "the g-xTB 2.0.1 build of xtb (xtb 6.7.1 26dd68d)",
    XTB671: "the xtb 6.7.1 release (edcfbbe)",
    GXTB11_DIR: "the g-xTB 1.1.0 program with its parameter files (gxtb, gxtb_parameters, basisq, eeq)",
}
GXTB11_FILES = ("gxtb", "gxtb_parameters", "basisq", "eeq")
SKIP = 77  # exit status of a script whose programs are missing


def get(var: str) -> str | None:
    """The program (an absolute path) that ``var`` names, or ``None``."""
    value = os.environ.get(var, "").strip()
    if not value:
        return None
    if var == GXTB11_DIR:
        d = Path(value).expanduser()
        return str(d.resolve()) if all((d / f).is_file() for f in GXTB11_FILES) else None
    found = shutil.which(os.path.expanduser(value))
    return str(Path(found).resolve()) if found else None


def gxtb11_dir() -> Path | None:
    """The g-xTB 1.1.0 directory as a ``Path``, or ``None``."""
    d = get(GXTB11_DIR)
    return Path(d) if d else None


def missing(*vars: str) -> list[str]:
    """The variables among ``vars`` that name no usable program."""
    return [v for v in vars if get(v) is None]


def reason(*vars: str) -> str:
    """Why cases that need ``vars`` are skipped (empty when nothing is missing)."""
    lack = missing(*vars)
    if not lack:
        return ""
    why = []
    for v in lack:
        state = "not set" if not os.environ.get(v, "").strip() else f"{os.environ[v]!r} is not usable"
        why.append(f"{DESCRIPTION[v]}: set {v} ({state})")
    return "needs " + "; ".join(why)


def require(*vars: str, what: str = "") -> list[str]:
    """The programs of ``vars``, or exit with ``SKIP`` and a message naming what is missing."""
    if missing(*vars):
        skip(f"{what + ': ' if what else ''}{reason(*vars)}")
    return [get(v) for v in vars]


def skip(message: str) -> None:
    """Exit with ``SKIP`` after saying why."""
    script = Path(sys.argv[0]).name if sys.argv and sys.argv[0] else "script"
    print(f"SKIPPED {script}: {message}", file=sys.stderr)
    raise SystemExit(SKIP)


def note_skip(cases: str, *vars: str) -> bool:
    """``True`` when ``vars`` are all there; otherwise say on stderr that ``cases`` are skipped
    and return ``False``."""
    if missing(*vars):
        print(f"skipping {cases}: {reason(*vars)}", file=sys.stderr)
        return False
    return True


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--have":  # exit status: whether the program is there
        return 0 if get(sys.argv[2]) else 1
    for var in (XTB680, GXTB2, XTB671, GXTB11_DIR):
        print(f"{var:16} {get(var) or '-- ' + (reason(var).removeprefix('needs ') or 'missing')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
