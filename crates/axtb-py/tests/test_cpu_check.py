# SPDX-License-Identifier: LGPL-3.0-or-later
"""The CPU check of ``import axtb`` (``axtb/__init__.py``): which ``/proc/cpuinfo`` flags it
needs, and that its message is the CLI's.

    python -m pytest crates/axtb-py/tests/test_cpu_check.py
"""

from __future__ import annotations

import re
from pathlib import Path

import axtb

ROOT = Path(__file__).resolve().parents[3]

# /proc/cpuinfo flags of an Intel Haswell and a Nehalem CPU (the parts that matter)
HASWELL = ("fpu vme de pse tsc msr pae mce cx8 apic sep mtrr sse sse2 ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt "
           "aes xsave avx f16c rdrand lahf_lm abm fsgsbase bmi1 avx2 smep bmi2 erms invpcid")
NEHALEM = "fpu vme de pse tsc msr pae mce cx8 apic sep mtrr sse sse2 ssse3 cx16 sse4_1 sse4_2 popcnt lahf_lm"


def cpuinfo(flags: str) -> list[str]:
    return ["processor\t: 0", "model name\t: test", f"flags\t\t: {flags}", "", "processor\t: 1", "flags\t\t: none"]


def test_x86_64_v3_flags():
    assert axtb._missing_cpu_flags(cpuinfo(HASWELL)) == []
    assert axtb._missing_cpu_flags(cpuinfo(HASWELL.replace("abm", "lzcnt"))) == []
    missing = axtb._missing_cpu_flags(cpuinfo(NEHALEM))
    assert missing == ["avx", "avx2", "fma", "bmi1", "bmi2", "f16c", "movbe", "xsave", "lzcnt"]
    for flag in ("avx2", "fma", "bmi2", "movbe"):
        assert axtb._missing_cpu_flags(cpuinfo(HASWELL.replace(f" {flag} ", " "))) == [flag]


def test_no_flags_line_skips_the_check():
    # e.g. an ARM host's /proc/cpuinfo under user-mode emulation
    assert axtb._missing_cpu_flags(["processor\t: 0", "Features\t: fp asimd evtstrm"]) is None
    assert axtb._missing_cpu_flags([]) is None


def test_message_is_the_clis():
    src = (ROOT / "crates/driver/src/lib.rs").read_text()
    m = re.search(r'const CPU_MISMATCH: &std::ffi::CStr = c"(.*?)";\n', src, re.S)
    assert m, "CPU_MISMATCH not found in crates/driver/src/lib.rs"
    rust = re.sub(r"\\\n\s*", "", m.group(1)).replace('\\"', '"')
    assert axtb._CPU_MISMATCH == rust
