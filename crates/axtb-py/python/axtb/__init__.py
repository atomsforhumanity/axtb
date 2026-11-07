"""axtb: extended tight binding (GFN2-xTB; g-xTB v1.1 and 2.0.1) and GFN-FF in-process.

All quantities are in atomic units (bohr, Hartree). Calculations run on the number of
threads you pass (default 1) and never spawn more, so many can run side by side. Results are
dicts with the keys of the CLI's JSON response (``program``, ``version``, ``method``,
``conventions``, ``energy``, ``gradient``, ...; arrays as numpy arrays); a key is present when
it was computed. Settings that do not apply to the method raise ``axtb.InputError``.

>>> import numpy as np, axtb
>>> pos = [[0, 0, 0.22], [0, 1.43, -0.89], [0, -1.43, -0.89]]          # any array-like, bohr
>>> r = axtb.singlepoint([8, 1, 1], pos, method="gfn2", gradient=True)
>>> r["energy"], r["gradient"].shape
>>> axtb.singlepoint([8, 1, 1], pos, method="gxtb-v2.0.1", gradient=True)["gradient"]
>>> r = axtb.singlepoint([8, 1, 1], pos, method="gfnff", gradient=True)
>>> r["charges"], r["topology"]["bonds"]

Hessians come with the single point at the same geometry and their vibrational analysis
(frequencies in cm⁻¹, normal modes, IR intensities, quasi-RRHO thermochemistry);
``vibrational_analysis`` analyses a Hessian from any source:

>>> h = axtb.Calculator("gfn2", [8, 1, 1], pos).hessian(pos)
>>> h["frequencies"], h["ir_intensities"], h["thermo"]["g_rrho"], h["thermo"]["free_energy"]
>>> axtb.vibrational_analysis([8, 1, 1], pos, h["hessian"], conventions="xtb")["thermo"]["zpve"]

Errors: ``axtb.InputError`` (a ``ValueError``) for invalid input, elements without
parameters and structures GFN-FF cannot set up; ``axtb.ScfNotConvergedError`` (a
``RuntimeError``) with ``iterations``, ``energy_change`` and ``residual``; ``RuntimeError``
for numerical failures. Each carries ``kind`` (``"input"``, ``"unsupported_element"``,
``"topology"``, ``"scf_not_converged"``, ``"numerical"``).

The x86-64 Linux builds need an x86-64-v3 CPU (AVX2, FMA, BMI2, ...). ``import axtb`` checks the
CPU flags in ``/proc/cpuinfo`` before it loads the extension module and raises ``ImportError``
on a CPU without them. Where ``/proc/cpuinfo`` cannot be read or lists no flags (other
operating systems, some emulators) the check is skipped, and on such a CPU the import would end
the process with "Illegal instruction". User-mode QEMU shows the host's ``/proc/cpuinfo``, not
the emulated CPU's. A build with ``RUSTFLAGS="-C target-cpu=x86-64"`` runs on any x86-64 CPU and
skips the check.
"""

import platform as _platform
import sys as _sys

# The CLI's message (axtb_driver's CPU_MISMATCH in crates/driver/src/lib.rs; a test compares them).
_CPU_MISMATCH = (
    "this axtb was built for x86-64-v3 CPUs (AVX2, FMA, BMI2: Intel since Haswell (2013), AMD "
    "since Excavator (2015); not many low-end Intel Atom, Celeron and Pentium N-series CPUs), and "
    "this CPU (or its operating system) does not support them. Build axtb from source for any "
    'x86-64 CPU with RUSTFLAGS="-C target-cpu=x86-64" (CLI: cargo install axtb-cli; '
    "Python: pip install --no-binary axtb axtb)"
)

# x86-64-v3 as Linux lists it in /proc/cpuinfo; LZCNT is "abm". The kernel lists AVX (and
# everything built on it) only when it saves the AVX registers (OSXSAVE, which it does not list).
_V3_FLAGS = ("avx", "avx2", "fma", "bmi1", "bmi2", "f16c", "movbe", "xsave")


def _missing_cpu_flags(lines):
    """The x86-64-v3 flags missing from the first ``flags`` line of ``/proc/cpuinfo``'s
    ``lines``, or ``None`` when there is no such line."""
    for line in lines:
        key, sep, value = line.partition(":")
        if sep and key.strip() == "flags":
            flags = set(value.split())
            missing = [f for f in _V3_FLAGS if f not in flags]
            if not flags & {"abm", "lzcnt"}:
                missing.append("lzcnt")
            return missing
    return None


def _check_cpu():
    """``ImportError`` on a Linux x86-64 CPU without what this build assumes, before the
    extension module (and its first AVX instruction) loads."""
    try:
        from ._build import X86_64_V3  # written by build.rs
    except ImportError:
        X86_64_V3 = True  # the default build
    if not X86_64_V3 or not _sys.platform.startswith("linux") or _platform.machine() != "x86_64":
        return
    try:
        with open("/proc/cpuinfo", encoding="ascii", errors="replace") as f:
            missing = _missing_cpu_flags(f)
    except OSError:
        return
    if missing:
        raise ImportError(_CPU_MISMATCH)


_check_cpu()

from ._axtb import (  # noqa: E402  (after the CPU check)
    Calculator,
    InputError,
    ScfNotConvergedError,
    ScfState,
    __version__,
    canonical_method,
    gxtb_solvent_hint,
    singlepoint,
    vibrational_analysis,
)

__all__ = [
    "Calculator",
    "InputError",
    "ScfNotConvergedError",
    "ScfState",
    "__version__",
    "canonical_method",
    "gxtb_solvent_hint",
    "singlepoint",
    "vibrational_analysis",
]
