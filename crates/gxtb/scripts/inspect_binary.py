#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Inspect the unstripped g-xTB 1.1.0 program (tooling for decoding further terms).

Usage:
  inspect_binary.py <gxtb binary> consts <routine>     constants loaded rip-relative by a routine
  inspect_binary.py <gxtb binary> users <symbol prefix> routines that reference module arrays
  inspect_binary.py <gxtb binary> disasm <routine>     Intel-syntax disassembly of one routine
  inspect_binary.py <gxtb binary> f64 <symbol> <n>     first n doubles of a data symbol

Needs binutils (nm, objdump). Module arrays are named `<module>_mp_<array>_` (Intel
Fortran); arrays indexed by element are addressed as [index*8 + base - 8], so references up
to 64 bytes before an array are attributed to it (adjacent arrays can give false positives).
Run-time values are best read with gdb, see make_reference.py.
"""

from __future__ import annotations

import bisect
import re
import struct
import subprocess
import sys


def symbols(binary):
    out = subprocess.run(["nm", "-n", "-S", binary], capture_output=True, text=True, check=True).stdout
    syms = []
    for line in out.splitlines():
        p = line.split()
        if len(p) == 4:
            syms.append((int(p[0], 16), int(p[1], 16), p[2], p[3]))
    return syms


def disasm(binary, start, stop):
    return subprocess.run(["objdump", "-d", "--no-show-raw-insn", "-M", "intel", f"--start-address={start}",
                           f"--stop-address={stop}", binary], capture_output=True, text=True, check=True).stdout


def reader(binary):
    data = open(binary, "rb").read()
    phoff, = struct.unpack_from("<Q", data, 0x20)
    phentsize, phnum = struct.unpack_from("<HH", data, 0x36)
    segs = []
    for i in range(phnum):
        t, _, off, va, _, fs, ms, _ = struct.unpack_from("<IIQQQQQQ", data, phoff + i * phentsize)
        if t == 1:
            segs.append((va, off, fs, ms))

    def read(addr, n):
        for va, off, fs, ms in segs:
            if va <= addr < va + ms:
                return data[off + addr - va: off + addr - va + n] if addr - va + n <= fs else b"\0" * n
        raise KeyError(hex(addr))
    return read


def main():
    binary, cmd = sys.argv[1], sys.argv[2]
    syms = symbols(binary)
    by_name = {s[3]: s for s in syms}
    read = reader(binary)
    if cmd in ("consts", "disasm"):
        a, size, _, _ = by_name[sys.argv[3]]
        text = disasm(binary, a, a + size)
        if cmd == "disasm":
            print(text)
            return
        for line in text.splitlines():
            m = re.search(r"# ([0-9a-f]+) <([^>]*)>", line)
            if m:
                addr = int(m.group(1), 16)
                try:
                    val = struct.unpack("<d", read(addr, 8))[0]
                except KeyError:
                    val = None
                print(line.split(":")[0].strip(), f"{m.group(2):40s}", repr(val))
    elif cmd == "users":
        prefix = sys.argv[3]
        arrays = sorted((s[0], s[1], s[3]) for s in syms if s[3].startswith(prefix) and s[2] in "bBdD")
        starts = [x[0] for x in arrays]
        funcs = [(s[0], s[1], s[3]) for s in syms if s[2] in "tT" and 0x400000 <= s[0] < 0x600000]
        lo, hi = min(f[0] for f in funcs), max(f[0] + f[1] for f in funcs)
        users, cur = {}, None
        for line in disasm(binary, lo, hi).splitlines():
            m = re.match(r"^[0-9a-f]+ <(.*)>:", line)
            if m:
                cur = m.group(1)
                continue
            vals = [int(x, 16) for x in re.findall(r"0x([0-9a-f]{6,7})\b", line)]
            vals += [int(x, 16) for x in re.findall(r"# ([0-9a-f]{6,7}) <", line)]
            for v in vals:
                i = bisect.bisect_right(starts, v + 64) - 1
                for j in (i, i + 1):
                    if 0 <= j < len(arrays) and arrays[j][0] - 64 <= v < arrays[j][0] + arrays[j][1]:
                        users.setdefault(arrays[j][2], set()).add(cur)
        for name in sorted(users):
            print(f"{name:36s}", " ".join(sorted(users[name])))
    elif cmd == "f64":
        a = by_name[sys.argv[3]][0]
        n = int(sys.argv[4])
        print(list(struct.unpack(f"<{n}d", read(a, 8 * n))))


if __name__ == "__main__":
    main()
