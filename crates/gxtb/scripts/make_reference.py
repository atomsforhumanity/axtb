#!/usr/bin/env python3
# SPDX-License-Identifier: LGPL-3.0-or-later
"""Generate the g-xTB 1.1.0 reference data under crates/gxtb/tests/reference/.

Usage: python3 crates/gxtb/scripts/make_reference.py <g-xtb-1.1.0 dir> <geometry dir> <output dir>

<g-xtb-1.1.0 dir> holds the program `gxtb` and its parameter files (gxtb_parameters,
basisq, eeq). <geometry dir> holds <name>.xyz for the molecules generated with Open Babel
(--gen3d); the small hand-made ones are defined below.

For every molecule the script runs, in a scratch directory with HOME=. and the parameter
files copied in (the program reads ~/.gxtb, ~/.basisq, ~/.eeq and truncates long paths):

1. `gxtb -c mol.xyz -molden`  -> gxtb.out, molden.input, energy
2. `gxtb -c mol.xyz -grad -symthr 0` -> gxtb_grad.out, gradient (numerical central
   differences; symmetry use is switched off because the program aborts on some point
   groups, e.g. D6h benzene)
3. the same single point under gdb, stopped at the entry of `scf_`, dumping the program's
   module arrays (bascom: q_eeq, cn_eeq, qeff, xyz, prim_exp, prim_cnt, at) at full
   precision -> intermediates.json

4. `gxtb -c mol.xyz -grad -symthr 0` under gdb with the SCF thresholds overwritten
   (`tight_gradient`) -> gradient_tight

Set INTERMEDIATES_ONLY=1 to refresh only the gdb dumps (intermediates.json, h0.json, scf.json,
kernels.json, fock.json); DISP_ONLY=1 to add only the revD4 energies (disp2, disp3) to
kernels.json; FOCK_ONLY=1 to write only fock.json; TIGHT_GRADIENT_ONLY=1 to write only
gradient_tight.

Charge and unpaired electrons are passed as the program expects them, in the files .CHRG
and .UHF of the working directory; they are also recorded in input.json. Each run starts
without a restart file (gxtbrestart), so the SCF starts from the program's own guess.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile

HANDMADE = {
    "water": ("3\nwater\nO 0.000000 0.000000 0.119262\nH 0.000000 0.763239 -0.477047\nH 0.000000 -0.763239 -0.477047\n", 0, 0),
    "hydroxide": ("2\nOH-\nO 0.0 0.0 0.0\nH 0.0 0.0 0.97\n", -1, 0),
    "dioxygen_triplet": ("2\nO2 triplet\nO 0.0 0.0 0.0\nO 0.0 0.0 1.21\n", 0, 2),
    "nickel_tetracarbonyl": None,  # built below
    "tetrachlorocuprate": ("5\n[CuCl4]2- doublet\nCu 0.0 0.0 0.0\nCl 2.26 0.0 0.0\nCl -2.26 0.0 0.0\nCl 0.0 2.26 0.0\nCl 0.0 -2.26 0.0\n", -2, 1),
}
OBABEL = {  # name: (charge, uhf); geometries from Open Babel --gen3d
    "methane": (0, 0), "ethanol": (0, 0), "benzene": (0, 0), "acetate": (-1, 0), "ammonium": (1, 0),
    "mesyl_chloride": (0, 0), "iodomethane": (0, 0), "glycine": (0, 0), "silane_phosphine": (0, 0),
}


def ni_co4() -> str:
    d = 3 ** -0.5
    dirs = [(d, d, d), (d, -d, -d), (-d, d, -d), (-d, -d, d)]
    lines = ["Ni 0.0 0.0 0.0"]
    for v in dirs:
        lines.append("C %.6f %.6f %.6f" % tuple(1.84 * x for x in v))
        lines.append("O %.6f %.6f %.6f" % tuple(2.98 * x for x in v))
    return "9\nNi(CO)4\n" + "\n".join(lines) + "\n"


# Intel Fortran rank-r array descriptor: base, elem_len, offset, flags, rank, reserved,
# then (extent, stride, lbound) per dimension. Addresses from `nm` of the 1.1.0 binary.
DESCRIPTORS = {
    "q_eeq": (0x3A65920, 1, "d"), "cn_eeq": (0x3A658C0, 1, "d"), "qeff": (0x3A65860, 1, "d"),
    "xyz": (0x3A65AA0, 2, "d"), "prim_exp": (0x3A65B60, 1, "d"), "prim_cnt": (0x3A65B00, 1, "d"),
    "at": (0x3A65FE0, 1, "i"),
}


# Scalars read at a code address inside scf_ (after the call that computes them):
# name -> (breakpoint address, gdb expression). 0x58354f follows `call repuls_`, whose
# second argument is the stack slot rbp-0x110.
SCALARS = {"repulsion_energy": (0x58354F, "*(double*)($rbp-0x110)")}


# H0: inside scf_, the packed (lower triangle, program AO order) array at rbp-0xd0 holds the
# diatomic-frame scaled overlap right after `call sintm_diat_` (0x583570) and H0 after the
# assembly loop (0x58385e) and H0 + ACP after `call calcvacp_` (0x583871); rbp-0x190 holds the
# AO levels lev - lev_cn*CN.
H0_STOPS = (0x583570, 0x58385E, 0x583871)


def h0_dump(binary, workdir):
    env = dict(os.environ, OMP_NUM_THREADS="1", HOME=".")

    def gdb(cmds):
        open(os.path.join(workdir, "gdb.cmd"), "w").write("\n".join(["set pagination off", *cmds, "kill"]) + "\n")
        p = _fresh(workdir) or subprocess.run(["gdb", "-batch", "-x", "gdb.cmd", "--args", binary, "-c", "mol.xyz"], cwd=workdir, env=env,
                           capture_output=True, text=True, timeout=600)
        os.remove(os.path.join(workdir, "gdb.cmd"))
        return p.stdout

    out = gdb([f"break *0x{H0_STOPS[0]:x}", "run", 'printf "NDIM %d\\n", *(int*)0x3be21c0'])
    n = int(re.search(r"NDIM (\d+)", out).group(1))
    npk = n * (n + 1) // 2
    gdb([f"break *0x{H0_STOPS[0]:x}", f"break *0x{H0_STOPS[1]:x}", f"break *0x{H0_STOPS[2]:x}", "run",
         f"dump binary memory st.bin *(long*)($rbp-0xd0) *(long*)($rbp-0xd0)+{8 * npk}", "continue",
         f"dump binary memory h0.bin *(long*)($rbp-0xd0) *(long*)($rbp-0xd0)+{8 * npk}",
         f"dump binary memory hd.bin *(long*)($rbp-0x190) *(long*)($rbp-0x190)+{8 * n}", "continue",
         f"dump binary memory ha.bin *(long*)($rbp-0xd0) *(long*)($rbp-0xd0)+{8 * npk}"])
    res = {"nsao": n}
    for f, key, k in (("st.bin", "stilde_packed", npk), ("h0.bin", "h0_packed", npk), ("hd.bin", "levels", n),
                      ("ha.bin", "h0_acp_packed", npk)):
        res[key] = list(struct.unpack("<%dd" % k, open(os.path.join(workdir, f), "rb").read()))
        os.remove(os.path.join(workdir, f))
    return res


def scf_dump(binary, workdir, nsao):
    """Converged density and overlap at the entry of `wiberg_` (after the SCF): the packed
    (program AO order) overlap S (second argument), the packed spin densities at rbp-0x1608
    (total density for closed shells; alpha then beta for open shells) and the Mulliken shell
    (rbp-0x4a8) and atomic (rbp-0x4f0) charges relative to the reference occupations."""
    env = dict(os.environ, OMP_NUM_THREADS="1", HOME=".")
    npk = nsao * (nsao + 1) // 2
    cmds = ["set pagination off", "break wiberg_", "run", 'printf "NSPIN %d\\n", *(int*)0x3be12e8',
            f"dump binary memory s.bin $rsi $rsi+{8 * npk}",
            f"dump binary memory p.bin *(long*)($rbp-0x1608) *(long*)($rbp-0x1608)+{16 * npk}",
            "dump binary memory qs.bin *(long*)($rbp-0x4a8) *(long*)($rbp-0x4a8)+8*(*(int*)0x3be12d8)",
            "dump binary memory qa.bin *(long*)($rbp-0x4f0) *(long*)($rbp-0x4f0)+8*(*(int*)0x3bcad40)", "kill"]
    open(os.path.join(workdir, "gdb.cmd"), "w").write("\n".join(cmds) + "\n")
    p = _fresh(workdir) or subprocess.run(["gdb", "-batch", "-x", "gdb.cmd", "--args", binary, "-c", "mol.xyz"], cwd=workdir, env=env,
                       capture_output=True, text=True, timeout=600)
    os.remove(os.path.join(workdir, "gdb.cmd"))
    nspin = int(re.search(r"NSPIN (\d+)", p.stdout).group(1))
    s_pk = list(struct.unpack("<%dd" % npk, open(os.path.join(workdir, "s.bin"), "rb").read()))
    p_pk = list(struct.unpack("<%dd" % (2 * npk), open(os.path.join(workdir, "p.bin"), "rb").read()))[: nspin * npk]
    rd = lambda f: list(struct.unpack("<%dd" % (os.path.getsize(os.path.join(workdir, f)) // 8), open(os.path.join(workdir, f), "rb").read()))
    q_shell, q_atom = rd("qs.bin"), rd("qa.bin")
    for f in ("s.bin", "p.bin", "qs.bin", "qa.bin"):
        os.remove(os.path.join(workdir, f))
    return {"nsao": nsao, "nspin": nspin, "s_packed": s_pk, "p_packed": p_pk, "shell_charges": q_shell, "atom_charges": q_atom}


def kernel_dump(binary, workdir, nsao):
    """Kernels set by `setgab_lrao_` (read after the call at 0x58353a): packed shell kernel
    gamma2 (rbp-0x538), packed shell array rbp-0x580 (= -1/2 gamma2^2, third-order use), full AO
    matrix of the Mulliken-exchange kernel (rbp-0x1148); and the energies of the final SCF
    iteration read after the last `call electro_` (0x5897ca): ES2+3 (rbp-0xf8), ES1 (rbp-0xf0),
    OFX (rbp-0x100), MFX exchange (rbp-0x108)."""
    env = dict(os.environ, OMP_NUM_THREADS="1", HOME=".")

    def gdb(cmds):
        open(os.path.join(workdir, "gdb.cmd"), "w").write("\n".join(["set pagination off", *cmds, "kill"]) + "\n")
        p = _fresh(workdir) or subprocess.run(["gdb", "-batch", "-x", "gdb.cmd", "--args", binary, "-c", "mol.xyz"], cwd=workdir, env=env,
                           capture_output=True, text=True, timeout=600)
        os.remove(os.path.join(workdir, "gdb.cmd"))
        return p.stdout

    out = gdb(["break *0x58353a", "run", 'printf "NSH %d\\n", *(int*)0x3be12d8',
               "dump binary memory g2.bin *(long*)($rbp-0x538) *(long*)($rbp-0x538)+8*(*(int*)0x3be12d8)*(*(int*)0x3be12d8+1)/2",
               "dump binary memory g3.bin *(long*)($rbp-0x580) *(long*)($rbp-0x580)+8*(*(int*)0x3be12d8)*(*(int*)0x3be12d8+1)/2",
               f"dump binary memory gx.bin *(long*)($rbp-0x1148) *(long*)($rbp-0x1148)+{8 * nsao * nsao}"])
    nsh = int(re.search(r"NSH (\d+)", out).group(1))
    rd = lambda f, k: list(struct.unpack("<%dd" % k, open(os.path.join(workdir, f), "rb").read()))
    res = {"nsh": nsh, "gamma2_packed": rd("g2.bin", nsh * (nsh + 1) // 2),
           "half_gamma2_sq_packed": rd("g3.bin", nsh * (nsh + 1) // 2), "gamma_mfx": rd("gx.bin", nsao * nsao)}
    for f in ("g2.bin", "g3.bin", "gx.bin"):
        os.remove(os.path.join(workdir, f))
    out = gdb(["break *0x5897ca", "run", "ignore 1 100000", "continue", "info breakpoints"])
    hits = int(re.search(r"already hit (\d+) time", out).group(1))
    cmd = ["break *0x5897ca", "run"] + ([f"continue {hits - 1}"] if hits > 1 else []) + [
        'printf "E %.17g %.17g %.17g %.17g\\n", *(double*)($rbp-0xf8), *(double*)($rbp-0xf0), *(double*)($rbp-0x100), *(double*)($rbp-0x108)',
        f"dump binary memory g3s.bin *(long*)($rbp-0x6a0) *(long*)($rbp-0x6a0)+{8 * nsh}",
        f"dump binary memory g4s.bin *(long*)($rbp-0x850) *(long*)($rbp-0x850)+{8 * nsh}"]
    e = [float(x) for x in re.search(r"^E (.*)$", gdb(cmd), re.M).group(1).split()]
    res["shell_gamma3"] = rd("g3s.bin", nsh)
    res["shell_rbp850"] = rd("g4s.bin", nsh)
    for f in ("g3s.bin", "g4s.bin"):
        os.remove(os.path.join(workdir, f))
    res["final_energies"] = {"es2_3": e[0], "es1": e[1], "ofx": e[2], "mfx": e[3], "scf_iterations": hits}
    # anisotropic electrostatics: rbp-0xe0 after `call aniso_electro_` (0x5898fa); spin
    # polarization: rbp-0x1c8 after `call energy_os_` (0x5899c6, open shells only)
    cmd = ["break *0x5898fa", "run"] + ([f"continue {hits - 1}"] if hits > 1 else []) + [
        'printf "A %.17g\\n", *(double*)($rbp-0xe0)']
    res["final_energies"]["aes"] = float(re.search(r"^A (\S+)$", gdb(cmd), re.M).group(1))
    # AES inputs of the final iteration at `call aniso_electro_` (0x5898f5): atomic dipoles
    # (3 x nat, rbp-0xd28), traceless quadrupoles (6 x nat: xx, xy, yy, xz, yz, zz; rbp-0xd88)
    # and the damped kernels g3, g5, g7, g9 (nat x nat; rbp-0xba8, -0xc08, -0xf08, -0xf68)
    nat = int(re.search(r"NAT (\d+)", gdb(["break scf_", "run", 'printf "NAT %d\\n", *(int*)0x3bcad40'])).group(1))
    regions = (("dip", "0xd28", 3 * nat), ("qp", "0xd88", 6 * nat), ("g3", "0xba8", nat * nat), ("g5", "0xc08", nat * nat),
               ("g7", "0xf08", nat * nat), ("g9", "0xf68", nat * nat))
    gdb(["break *0x5898f5", "run"] + ([f"continue {hits - 1}"] if hits > 1 else []) +
        [f"dump binary memory {n}.bin *(long*)($rbp-{o}) *(long*)($rbp-{o})+{8 * c}" for n, o, c in regions])
    res["aes"] = {n: rd(n + ".bin", c) for n, o, c in regions}
    for n, o, c in regions:
        os.remove(os.path.join(workdir, n + ".bin"))
    res["final_energies"].update(disp_dump(binary, workdir, hits))
    out = gdb(["break *0x5899c6", "run", "ignore 1 100000", "continue", "info breakpoints"])
    m = re.search(r"already hit (\d+) time", out)
    if m:
        h = int(m.group(1))
        cmd = ["break *0x5899c6", "run"] + ([f"continue {h - 1}"] if h > 1 else []) + [
            'printf "S %.17g\\n", *(double*)($rbp-0x1c8)']
        res["final_energies"]["spin"] = float(re.search(r"^S (\S+)$", gdb(cmd), re.M).group(1))
    return res


def disp_dump(binary, workdir, hits):
    """revD4 energies at full precision: the three-body (ATM) energy at rbp-0x120 after
    `call tblite_disp_d4_mp_get_dispersion_nonsc_` (0x5831d1, once before the SCF) and the
    two-body energy of the final SCF iteration at rbp-0xd8 after `call
    tblite_disp_d4_mp_get_energy_` (0x589977). Printed "dispersion" = disp2 + disp3."""
    env = dict(os.environ, OMP_NUM_THREADS="1", HOME=".")
    cmds = ["set pagination off", "break *0x5831d1", "break *0x589977", "run",
            'printf "D3 %.17g\\n", *(double*)($rbp-0x120)', "delete 1", "continue"]
    cmds += ([f"continue {hits - 1}"] if hits > 1 else []) + ['printf "D2 %.17g\\n", *(double*)($rbp-0xd8)', "kill"]
    open(os.path.join(workdir, "gdb.cmd"), "w").write("\n".join(cmds) + "\n")
    p = _fresh(workdir) or subprocess.run(["gdb", "-batch", "-x", "gdb.cmd", "--args", binary, "-c", "mol.xyz"], cwd=workdir, env=env,
                       capture_output=True, text=True, timeout=600)
    os.remove(os.path.join(workdir, "gdb.cmd"))
    return {"disp3": float(re.search(r"^D3 (\S+)$", p.stdout, re.M).group(1)),
            "disp2": float(re.search(r"^D2 (\S+)$", p.stdout, re.M).group(1))}


def fock_dump(binary, workdir):
    """Fock matrices as built from a density, before the mixing: at the call of `diisfock_`
    in `scf_` (0x588e43) the packed (program AO order) Fock matrices per spin at [rbp-0xa88]
    and the spin densities they were built from at [rbp-0x1608] (closed shell: total
    density). Kept for the first iteration (start density) and the last call."""
    env = dict(os.environ, OMP_NUM_THREADS="1", HOME=".")
    py = [
        "python",
        "import gdb, struct, json",
        "res = []",
        "def rd(a, k): return list(struct.unpack('<%dd' % k, bytes(gdb.selected_inferior().read_memory(a, 8 * k))))",
        "def i32(a): return struct.unpack('<i', bytes(gdb.selected_inferior().read_memory(a, 4)))[0]",
        "def u64(a): return struct.unpack('<Q', bytes(gdb.selected_inferior().read_memory(a, 8)))[0]",
        "class B(gdb.Breakpoint):",
        "    def stop(self):",
        "        rbp = int(gdb.parse_and_eval('$rbp')) & 0xffffffffffffffff",
        "        n = i32(0x3be21c0); ns = i32(0x3be12e8); npk = n * (n + 1) // 2",
        "        res.append({'nsao': n, 'nspin': ns, 'F': rd(u64(rbp - 0xa88), ns * npk), 'P': rd(u64(rbp - 0x1608), ns * npk)})",
        "        return False",
        "B('*0x588e43')",
        "end",
        "run",
        "python",
        "json.dump({'first': res[0], 'last': res[-1], 'calls': len(res)}, open('fock_dump.json', 'w'))",
        "end",
    ]
    open(os.path.join(workdir, "gdb.cmd"), "w").write("\n".join(["set pagination off", *py]) + "\n")
    _fresh(workdir) or subprocess.run(["gdb", "-batch", "-x", "gdb.cmd", "--args", binary, "-c", "mol.xyz"], cwd=workdir, env=env,
                                      capture_output=True, text=True, timeout=600)
    os.remove(os.path.join(workdir, "gdb.cmd"))
    res = json.load(open(os.path.join(workdir, "fock_dump.json")))
    os.remove(os.path.join(workdir, "fock_dump.json"))
    return res


# SCF convergence test in scf_ (1.1.0 binary): at 0x589c1b the energy-change threshold is at
# rbp-0x2a80 and the RMS density-change threshold at rbp-0x2ad0 (defaults 1e-8 Eh, 2e-5;
# open shell 4e-7 Eh, 5e-5).
SCF_TEST = 0x589C1B
TIGHT_THRESHOLDS = (1e-12, 1e-10)


def tight_gradient(binary, workdir):
    """The program's numerical gradient (`-grad -symthr 0`) with the SCF thresholds of every
    single point overwritten to TIGHT_THRESHOLDS, so that its central differences carry no
    SCF noise (with the defaults the noise reaches ~2e-5 Eh/bohr, because the energy is not
    stationary in the density). Returns the `gradient` file."""
    env = dict(os.environ, OMP_NUM_THREADS="1", HOME=".")
    et, pt = TIGHT_THRESHOLDS
    cmds = [
        "set pagination off",
        "python",
        "import gdb, struct",
        "class Tight(gdb.Breakpoint):",
        "    def stop(self):",
        "        rbp = int(gdb.parse_and_eval('$rbp'))",
        f"        gdb.selected_inferior().write_memory(rbp - 0x2a80, struct.pack('<d', {et!r}))",
        f"        gdb.selected_inferior().write_memory(rbp - 0x2ad0, struct.pack('<d', {pt!r}))",
        "        return False",
        f"Tight('*0x{SCF_TEST:x}')",
        "end",
        "run",
    ]
    open(os.path.join(workdir, "gdb.cmd"), "w").write("\n".join(cmds) + "\n")
    _fresh(workdir)
    p = subprocess.run(["gdb", "-batch", "-x", "gdb.cmd", "--args", binary, "-c", "mol.xyz", "-grad", "-symthr", "0"],
                       cwd=workdir, env=env, capture_output=True, text=True, timeout=3600)
    os.remove(os.path.join(workdir, "gdb.cmd"))
    if "exited normally" not in p.stdout:
        raise RuntimeError("tight gradient run failed:\n" + p.stdout[-2000:] + p.stderr[-2000:])
    return open(os.path.join(workdir, "gradient")).read()


def _fresh(workdir):
    """Remove restart and side files so that every run starts the SCF from the program's own
    guess (a `gxtbrestart` left by an earlier run changes the SCF path and the converged
    density at the 1e-9 level)."""
    for f in ("gxtbrestart", "energy", "gradient", ".data", "molden.input"):
        if os.path.exists(os.path.join(workdir, f)):
            os.remove(os.path.join(workdir, f))


def run(binary, workdir, args, outname):
    env = dict(os.environ, OMP_NUM_THREADS="1", HOME=".")
    for f in ("gxtbrestart", "energy", "gradient", ".data", "molden.input"):
        if os.path.exists(os.path.join(workdir, f)):
            os.remove(os.path.join(workdir, f))
    p = subprocess.run([binary, "-c", "mol.xyz", *args], cwd=workdir, env=env, capture_output=True, text=True, timeout=3600)
    open(os.path.join(workdir, outname), "w").write(p.stdout + p.stderr)
    if p.returncode != 0:
        raise RuntimeError(f"{args}: exit {p.returncode}")


def gdb_dump(binary, workdir):
    env = dict(os.environ, OMP_NUM_THREADS="1", HOME=".")

    def gdb(cmds):
        open(os.path.join(workdir, "gdb.cmd"), "w").write("\n".join(["set pagination off", "break scf_", "run", *cmds, "kill"]) + "\n")
        _fresh(workdir) or subprocess.run(["gdb", "-batch", "-x", "gdb.cmd", "--args", binary, "-c", "mol.xyz"], cwd=workdir, env=env,
                       capture_output=True, text=True, timeout=600)
        os.remove(os.path.join(workdir, "gdb.cmd"))

    gdb([f"dump binary memory {n}.desc 0x{a:x} 0x{a + 48 + 24 * r:x}" for n, (a, r, _) in DESCRIPTORS.items()])
    cmds, shapes = [], {}
    for n, (a, r, t) in DESCRIPTORS.items():
        raw = open(os.path.join(workdir, n + ".desc"), "rb").read()
        d = struct.unpack("<%dq" % (6 + 3 * r), raw)
        ext = [d[6 + 3 * k] for k in range(r)]
        nel = 1
        for e in ext:
            nel *= e
        shapes[n] = (ext, d[1], t)
        cmds.append(f"dump binary memory {n}.bin 0x{d[0]:x} 0x{d[0] + d[1] * nel:x}")
    gdb(cmds)
    out = {}
    for n, (addr, expr) in SCALARS.items():
        open(os.path.join(workdir, "gdb.cmd"), "w").write(
            f"set pagination off\nbreak *0x{addr:x}\nrun\nprintf \"VALUE %.17g\\n\", {expr}\nkill\n")
        p = _fresh(workdir) or subprocess.run(["gdb", "-batch", "-x", "gdb.cmd", "--args", binary, "-c", "mol.xyz"], cwd=workdir, env=env,
                           capture_output=True, text=True, timeout=600)
        os.remove(os.path.join(workdir, "gdb.cmd"))
        out[n] = float(re.search(r"VALUE (\S+)", p.stdout).group(1))
    for n, (ext, elen, t) in shapes.items():
        raw = open(os.path.join(workdir, n + ".bin"), "rb").read()
        fmt = "d" if t == "d" else ("i" if elen == 4 else "q")
        vals = list(struct.unpack("<%d%s" % (len(raw) // elen, fmt), raw))
        out[n] = {"shape": ext, "data": vals}
        os.remove(os.path.join(workdir, n + ".bin"))
        os.remove(os.path.join(workdir, n + ".desc"))
    return out


NUM = r"[-+]?\d+\.\d+(?:[EeDd][-+]?\d+)?"


def summarize(text):
    s = {}
    m = re.search(r"CN\s+q_CN\s+CN\(basis\)\s+q\n(.*?)\n\s*\n", text, re.S)
    if m:
        rows = [l.split() for l in m.group(1).splitlines() if l.strip()]
        s["eeq_table"] = {"cn": [float(r[2]) for r in rows], "qloc": [float(r[3]) for r in rows],
                          "cn_basis": [float(r[4]) for r in rows], "q": [float(r[5]) for r in rows]}
    m = re.search(r"dipole moment.*?\n\s*(" + NUM + r")\s+(" + NUM + r")\s+(" + NUM + ")", text)
    if m:
        s["eeq_dipole_au"] = [float(x) for x in m.groups()]
    m = re.search(r"at\s+q\s+CN\s+q change.*?\n(.*?)\n ncao", text, re.S)
    if m:
        rows = [l.split() for l in m.group(1).splitlines() if l.strip()]
        s["qvszp_table"] = {k: [float(r[i]) for r in rows] for i, k in
                            enumerate(["z", "q", "cn", "q_change", "cn_change", "qcn_change", "qeff"])}
    for key in ("ncao", "nsao", "npr", "nel", "nalpha", "nbeta", "nopen", "acpsao"):
        m = re.search(r"^\s*%s\s+(-?\d+)" % key, text, re.M)
        if m:
            s[key] = int(m.group(1))
    terms = {}
    for key in ["ES1 (charge SIE)", "ES2+3", "ES multipole", "ES total", "Ex (Mulliken)", "Espinpol",
                "electronic", "atomic core increments", "dispersion", "nuclear repulsion", "gap (eV)"]:
        m = re.search(re.escape(key) + r"\s*:\s*(" + NUM + ")", text)
        if m:
            terms[key] = float(m.group(1))
    m = re.search(r"^total\s+(" + NUM + ")", text, re.M)
    if m:
        terms["total"] = float(m.group(1))
    s["energy_terms"] = terms
    m = re.search(r"(\d+)\s+scf iterations", text)
    if m:
        s["scf_iterations"] = int(m.group(1))
    return s


def main():
    refdir, geomdir, outdir = sys.argv[1:4]
    binary = os.path.join(refdir, "gxtb")
    mols = {}
    for name, v in HANDMADE.items():
        mols[name] = v if v is not None else (ni_co4(), 0, 0)
    for name, (c, u) in OBABEL.items():
        mols[name] = (open(os.path.join(geomdir, name + ".xyz")).read(), c, u)
    only = sys.argv[4].split(",") if len(sys.argv) > 4 else None
    for name, (xyz, charge, uhf) in mols.items():
        if only and name not in only:
            continue
        print(name, flush=True)
        work = tempfile.mkdtemp()
        try:
            open(os.path.join(work, "mol.xyz"), "w").write(xyz)
            if charge:
                open(os.path.join(work, ".CHRG"), "w").write(f"{charge}\n")
            if uhf:
                open(os.path.join(work, ".UHF"), "w").write(f"{uhf}\n")
            for src, dst in (("gxtb_parameters", ".gxtb"), ("basisq", ".basisq"), ("eeq", ".eeq")):
                shutil.copy(os.path.join(refdir, src), os.path.join(work, dst))
            dest = os.path.join(outdir, name)
            os.makedirs(dest, exist_ok=True)
            if os.environ.get("TIGHT_GRADIENT_ONLY"):
                open(os.path.join(dest, "gradient_tight"), "w").write(tight_gradient(binary, work))
                continue
            if os.environ.get("FOCK_ONLY"):
                json.dump(fock_dump(binary, work), open(os.path.join(dest, "fock.json"), "w"))
                continue
            if os.environ.get("DISP_ONLY"):
                kpath = os.path.join(dest, "kernels.json")
                k = json.load(open(kpath))
                k["final_energies"].update(disp_dump(binary, work, k["final_energies"]["scf_iterations"]))
                json.dump(k, open(kpath, "w"))
                continue
            if os.environ.get("INTERMEDIATES_ONLY"):
                json.dump(gdb_dump(binary, work), open(os.path.join(dest, "intermediates.json"), "w"))
                h0 = h0_dump(binary, work)
                json.dump(h0, open(os.path.join(dest, "h0.json"), "w"))
                json.dump(scf_dump(binary, work, h0["nsao"]), open(os.path.join(dest, "scf.json"), "w"))
                json.dump(kernel_dump(binary, work, h0["nsao"]), open(os.path.join(dest, "kernels.json"), "w"))
                json.dump(fock_dump(binary, work), open(os.path.join(dest, "fock.json"), "w"))
                continue
            run(binary, work, ["-molden"], "gxtb.out")
            for f in ("gxtb.out", "molden.input", "energy"):
                shutil.copy(os.path.join(work, f), os.path.join(dest, f))
            run(binary, work, ["-grad", "-symthr", "0"], "gxtb_grad.out")
            for f in ("gxtb_grad.out", "gradient"):
                shutil.copy(os.path.join(work, f), os.path.join(dest, f))
            inter = gdb_dump(binary, work)
            shutil.copy(os.path.join(work, "mol.xyz"), os.path.join(dest, "mol.xyz"))
            summary = summarize(open(os.path.join(work, "gxtb.out")).read())
            etot = float(open(os.path.join(work, "energy")).read().split("\n")[1].split()[1])
            summary["total_energy"] = etot
            json.dump({"charge": charge, "uhf": uhf, "program": "g-xTB 1.1.0 (grimme-lab/g-xtb v1.1.0)",
                       "command": "gxtb -c mol.xyz -molden | -grad -symthr 0; HOME=. with .gxtb/.basisq/.eeq, OMP_NUM_THREADS=1"},
                      open(os.path.join(dest, "input.json"), "w"), indent=1)
            json.dump(summary, open(os.path.join(dest, "summary.json"), "w"), indent=1)
            json.dump(inter, open(os.path.join(dest, "intermediates.json"), "w"))
            h0 = h0_dump(binary, work)
            json.dump(h0, open(os.path.join(dest, "h0.json"), "w"))
            json.dump(scf_dump(binary, work, h0["nsao"]), open(os.path.join(dest, "scf.json"), "w"))
            json.dump(kernel_dump(binary, work, h0["nsao"]), open(os.path.join(dest, "kernels.json"), "w"))
            json.dump(fock_dump(binary, work), open(os.path.join(dest, "fock.json"), "w"))
            open(os.path.join(dest, "gradient_tight"), "w").write(tight_gradient(binary, work))
        finally:
            shutil.rmtree(work)


if __name__ == "__main__":
    main()
