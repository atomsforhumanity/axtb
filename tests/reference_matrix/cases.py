"""Case definitions of the reference matrix.

A case is a dict:

    id, category, label           unique name, category (= --subset name), description
    numbers, positions            atomic numbers, positions in bohr (generated, structures.py)
    charge, uhf                   total charge (int), unpaired electrons (None = default)
    accuracy, etemp, max_iter, dispersion, threads, alpb, gbsa, solvation_state
                                  options as accepted by axtb.singlepoint (defaults otherwise)
    checks                        which comparisons to run (see run.py `CHECKS`)
    tags                          free-form labels (open-shell, small-gap, ...)
    expect                        failure cases: what a correct program does ("error" or "ok")

Add a category by writing a generator below and listing it in `CATEGORIES`.
"""

from __future__ import annotations

import numpy as np

import structures as S

TIGHT = 1.0e-6  # SCF accuracy for value comparisons (xtb is clamped to its minimum 1e-4)
SP_CHECKS = ("xtb", "tblite", "axtb_vs_xtbmode")


def case(cid, category, label, mol, charge=0, uhf=None, checks=SP_CHECKS, tags=(), **opts):
    nums, pos = mol
    c = {"id": cid, "category": category, "label": label, "numbers": [int(z) for z in nums],
         "positions": np.asarray(pos, dtype=float).tolist(),
         "charge": int(charge) if float(charge).is_integer() else float(charge), "uhf": uhf,
         "accuracy": TIGHT, "checks": list(checks), "tags": list(tags)}
    c.update(opts)
    nel = S.electrons(nums, charge)
    c["nel"] = nel
    if "expect" not in c and nel >= 0:
        u = uhf if uhf is not None else nel % 2
        c["multiplicity"] = u + 1
        if u > 0 and "open-shell" not in c["tags"]:
            c["tags"].append("open-shell")
    return c


# ---------------------------------------------------------------------------------------
# elements: every element 1-86 in at least one small molecule


def _lncl3(sym):
    return S.star(sym, ["Cl"] * 3, "trigonal")


# (builder, charge, uhf, label)
ELEMENT_MOLECULES = {
    "H": (lambda: S.molecule("H2"), 0, None, "H2"),
    "He": (lambda: S.molecule("HeH+"), 1, None, "HeH+"),
    "Li": (lambda: S.hydride("Li", 1), 0, None, "LiH"),
    "Be": (lambda: S.hydride("Be", 2), 0, None, "BeH2"),
    "B": (lambda: S.hydride("B", 3), 0, None, "BH3"),
    "C": (lambda: S.hydride("C", 4), 0, None, "CH4"),
    "N": (lambda: S.hydride("N", 3), 0, None, "NH3"),
    "O": (lambda: S.hydride("O", 2), 0, None, "H2O"),
    "F": (lambda: S.hydride("F", 1), 0, None, "HF"),
    "Ne": (lambda: S.diatomic("Ne", "H"), 1, None, "NeH+"),
    "Na": (lambda: S.hydride("Na", 1), 0, None, "NaH"),
    "Mg": (lambda: S.hydride("Mg", 2), 0, None, "MgH2"),
    "Al": (lambda: S.hydride("Al", 3), 0, None, "AlH3"),
    "Si": (lambda: S.hydride("Si", 4), 0, None, "SiH4"),
    "P": (lambda: S.hydride("P", 3), 0, None, "PH3"),
    "S": (lambda: S.hydride("S", 2), 0, None, "H2S"),
    "Cl": (lambda: S.hydride("Cl", 1), 0, None, "HCl"),
    "Ar": (lambda: S.diatomic("Ar", "H"), 1, None, "ArH+"),
    "K": (lambda: S.hydride("K", 1), 0, None, "KH"),
    "Ca": (lambda: S.hydride("Ca", 2), 0, None, "CaH2"),
    "Sc": (lambda: S.star("Sc", ["Cl"] * 3, "trigonal"), 0, None, "ScCl3"),
    "Ti": (lambda: S.star("Ti", ["Cl"] * 4, "tetrahedral"), 0, None, "TiCl4"),
    "V": (lambda: S.star("V", ["Cl"] * 4, "tetrahedral"), 0, 1, "VCl4 (d1 doublet)"),
    "Cr": (lambda: S.star("Cr", ["CO"] * 6, "octahedral", r=1.91), 0, None, "Cr(CO)6"),
    "Mn": (lambda: S.molecule("MnO4-"), -1, None, "MnO4-"),
    "Fe": (lambda: S.star("Fe", ["CO"] * 5, "tbp", r=1.81), 0, None, "Fe(CO)5"),
    "Co": (lambda: S.star("Co", ["CO"] * 4, "tetrahedral", r=1.75), -1, None, "Co(CO)4-"),
    "Ni": (lambda: S.star("Ni", ["CO"] * 4, "tetrahedral", r=1.83), 0, None, "Ni(CO)4"),
    "Cu": (lambda: S.star("Cu", ["Cl"] * 2, "linear", r=2.10), -1, None, "CuCl2-"),
    "Zn": (lambda: S.star("Zn", ["Cl"] * 2, "linear"), 0, None, "ZnCl2"),
    "Ga": (lambda: S.hydride("Ga", 3), 0, None, "GaH3"),
    "Ge": (lambda: S.hydride("Ge", 4), 0, None, "GeH4"),
    "As": (lambda: S.hydride("As", 3), 0, None, "AsH3"),
    "Se": (lambda: S.hydride("Se", 2), 0, None, "H2Se"),
    "Br": (lambda: S.hydride("Br", 1), 0, None, "HBr"),
    "Kr": (lambda: S.star("Kr", ["F"] * 2, "linear", r=1.89), 0, None, "KrF2"),
    "Rb": (lambda: S.hydride("Rb", 1), 0, None, "RbH"),
    "Sr": (lambda: S.hydride("Sr", 2), 0, None, "SrH2"),
    "Y": (lambda: S.star("Y", ["Cl"] * 3, "trigonal"), 0, None, "YCl3"),
    "Zr": (lambda: S.star("Zr", ["Cl"] * 4, "tetrahedral"), 0, None, "ZrCl4"),
    "Nb": (lambda: S.star("Nb", ["Cl"] * 5, "tbp"), 0, None, "NbCl5"),
    "Mo": (lambda: S.star("Mo", ["CO"] * 6, "octahedral", r=2.06), 0, None, "Mo(CO)6"),
    "Tc": (lambda: S.star("Tc", ["O"] * 4, "tetrahedral", r=1.71), -1, None, "TcO4-"),
    "Ru": (lambda: S.star("Ru", ["O"] * 4, "tetrahedral", r=1.70), 0, None, "RuO4"),
    "Rh": (lambda: S.star("Rh", ["Cl"] * 6, "octahedral", r=2.35), -3, None, "RhCl6(3-)"),
    "Pd": (lambda: S.star("Pd", ["Cl"] * 4, "square", r=2.31), -2, None, "PdCl4(2-)"),
    "Ag": (lambda: S.star("Ag", ["Cl"] * 2, "linear", r=2.30), -1, None, "AgCl2-"),
    "Cd": (lambda: S.star("Cd", ["Cl"] * 2, "linear"), 0, None, "CdCl2"),
    "In": (lambda: S.hydride("In", 3), 0, None, "InH3"),
    "Sn": (lambda: S.hydride("Sn", 4), 0, None, "SnH4"),
    "Sb": (lambda: S.hydride("Sb", 3), 0, None, "SbH3"),
    "Te": (lambda: S.hydride("Te", 2), 0, None, "H2Te"),
    "I": (lambda: S.hydride("I", 1), 0, None, "HI"),
    "Xe": (lambda: S.star("Xe", ["F"] * 2, "linear", r=1.98), 0, None, "XeF2"),
    "Cs": (lambda: S.hydride("Cs", 1), 0, None, "CsH"),
    "Ba": (lambda: S.hydride("Ba", 2), 0, None, "BaH2"),
    "La": (lambda: _lncl3("La"), 0, None, "LaCl3"),
    **{ln: ((lambda ln=ln: _lncl3(ln)), 0, None, f"{ln}Cl3")
       for ln in "Ce Pr Nd Pm Sm Eu Gd Tb Dy Ho Er Tm Yb Lu".split()},
    "Hf": (lambda: S.star("Hf", ["Cl"] * 4, "tetrahedral"), 0, None, "HfCl4"),
    "Ta": (lambda: S.star("Ta", ["Cl"] * 5, "tbp"), 0, None, "TaCl5"),
    "W": (lambda: S.star("W", ["F"] * 6, "octahedral", r=1.83), 0, None, "WF6"),
    "Re": (lambda: S.star("Re", ["O"] * 4, "tetrahedral", r=1.72), -1, None, "ReO4-"),
    "Os": (lambda: S.star("Os", ["O"] * 4, "tetrahedral", r=1.71), 0, None, "OsO4"),
    "Ir": (lambda: S.star("Ir", ["Cl"] * 6, "octahedral", r=2.36), -3, None, "IrCl6(3-)"),
    "Pt": (lambda: S.star("Pt", ["Cl"] * 4, "square", r=2.31), -2, None, "PtCl4(2-)"),
    "Au": (lambda: S.star("Au", ["Cl"] * 2, "linear", r=2.26), -1, None, "AuCl2-"),
    "Hg": (lambda: S.star("Hg", ["Cl"] * 2, "linear", r=2.29), 0, None, "HgCl2"),
    "Tl": (lambda: S.hydride("Tl", 3), 0, None, "TlH3"),
    "Pb": (lambda: S.hydride("Pb", 4), 0, None, "PbH4"),
    "Bi": (lambda: S.hydride("Bi", 3), 0, None, "BiH3"),
    "Po": (lambda: S.hydride("Po", 2), 0, None, "H2Po"),
    "At": (lambda: S.hydride("At", 1), 0, None, "HAt"),
    "Rn": (lambda: S.star("Rn", ["F"] * 2, "linear", r=2.08), 0, None, "RnF2"),
}

# a second environment for many elements: halides/oxides of the p block and aqua ions
ELEMENT_ALT = {
    "Li": (lambda: S.aqua("Li", 4, scale=1.10), 1, None, "[Li(H2O)4]+"),
    "Na": (lambda: S.aqua("Na", 6, scale=1.18), 1, None, "[Na(H2O)6]+"),
    "K": (lambda: S.aqua("K", 6, scale=1.15), 1, None, "[K(H2O)6]+"),
    "Be": (lambda: S.aqua("Be", 4, scale=1.04), 2, None, "[Be(H2O)4]2+"),
    "Ca": (lambda: S.aqua("Ca", 6, scale=1.10), 2, None, "[Ca(H2O)6]2+"),
    "Sr": (lambda: S.aqua("Sr", 6, scale=1.13), 2, None, "[Sr(H2O)6]2+"),
    "Ba": (lambda: S.aqua("Ba", 6, scale=1.15), 2, None, "[Ba(H2O)6]2+"),
    "Sc": (lambda: S.aqua("Sc", 6, scale=1.08), 3, None, "[Sc(H2O)6]3+"),
    "Ti": (lambda: S.aqua("Ti", 6, scale=1.13), 3, 1, "[Ti(H2O)6]3+ (d1)"),
    "Co": (lambda: S.aqua("Co"), 2, 3, "[Co(H2O)6]2+ (d7 quartet)"),
    "Ni": (lambda: S.aqua("Ni"), 2, 2, "[Ni(H2O)6]2+ (d8 triplet)"),
    "Cu": (lambda: S.aqua("Cu"), 2, 1, "[Cu(H2O)6]2+ (d9 doublet)"),
    "Zn": (lambda: S.aqua("Zn"), 2, None, "[Zn(H2O)6]2+"),
    "Ga": (lambda: S.aqua("Ga", 6, scale=1.13), 3, None, "[Ga(H2O)6]3+"),
    "In": (lambda: S.aqua("In", 6, scale=1.12), 3, None, "[In(H2O)6]3+"),
    "Cd": (lambda: S.aqua("Cd", 6, scale=1.24), 2, None, "[Cd(H2O)6]2+"),
    "Hg": (lambda: S.aqua("Hg", 6, scale=1.25), 2, None, "[Hg(H2O)6]2+"),
    "Y": (lambda: S.aqua("Y", 6, scale=1.13), 3, None, "[Y(H2O)6]3+"),
    "La": (lambda: S.aqua("La", 6, scale=1.13), 3, None, "[La(H2O)6]3+"),
    "Gd": (lambda: S.aqua("Gd", 6, scale=1.13), 3, None, "[Gd(H2O)6]3+"),
    "Lu": (lambda: S.aqua("Lu", 6, scale=1.13), 3, None, "[Lu(H2O)6]3+"),
    "Ru": (lambda: S.ammine("Ru", 6, r=2.10), 2, None, "[Ru(NH3)6]2+"),
    "Rh": (lambda: S.ammine("Rh", 6, r=2.07), 3, None, "[Rh(NH3)6]3+"),
    "Pt": (lambda: S.ammine("Pt", 4, r=2.05), 2, None, "[Pt(NH3)4]2+"),
    "Pd": (lambda: S.ammine("Pd", 4, r=2.05), 2, None, "[Pd(NH3)4]2+"),
    "Ir": (lambda: S.ammine("Ir", 6, r=2.10), 3, None, "[Ir(NH3)6]3+"),
    "Os": (lambda: S.ammine("Os", 6, r=2.12), 2, None, "[Os(NH3)6]2+"),
    "W": (lambda: S.star("W", ["CO"] * 6, "octahedral", r=2.06), 0, None, "W(CO)6"),
    "B": (lambda: S.star("B", ["F"] * 3, "trigonal", r=1.31), 0, None, "BF3"),
    "C": (lambda: S.star("C", ["F"] * 4, "tetrahedral", r=1.32), 0, None, "CF4"),
    "N": (lambda: S.star("N", ["F"] * 3, "pyramidal", r=1.37, angle=102.0), 0, None, "NF3"),
    "O": (lambda: S.star("O", ["F"] * 2, "bent", r=1.41, angle=103.0), 0, None, "OF2"),
    "Al": (lambda: S.star("Al", ["Cl"] * 3, "trigonal"), 0, None, "AlCl3"),
    "Si": (lambda: S.star("Si", ["Cl"] * 4, "tetrahedral"), 0, None, "SiCl4"),
    "P": (lambda: S.star("P", ["Cl"] * 3, "pyramidal", angle=100.0), 0, None, "PCl3"),
    "S": (lambda: S.star("S", ["O"] * 3, "trigonal", r=1.42), 0, None, "SO3"),
    "Ge": (lambda: S.star("Ge", ["Cl"] * 4, "tetrahedral"), 0, None, "GeCl4"),
    "As": (lambda: S.star("As", ["Cl"] * 3, "pyramidal", angle=98.0), 0, None, "AsCl3"),
    "Se": (lambda: S.star("Se", ["O"] * 2, "bent", r=1.61, angle=114.0), 0, None, "SeO2"),
    "Br": (lambda: S.star("Br", ["F"] * 3, "trigonal"), 0, None, "BrF3 (planar)"),
    "Sn": (lambda: S.star("Sn", ["Cl"] * 4, "tetrahedral"), 0, None, "SnCl4"),
    "Sb": (lambda: S.star("Sb", ["Cl"] * 3, "pyramidal", angle=97.0), 0, None, "SbCl3"),
    "Te": (lambda: S.star("Te", ["Cl"] * 2, "bent", angle=98.0), 0, None, "TeCl2"),
    "I": (lambda: S.star("I", ["Cl"] * 2, "linear"), -1, None, "ICl2-"),
    "Tl": (lambda: S.diatomic("Tl", "Cl", 2.49), 0, None, "TlCl"),
    "Pb": (lambda: S.star("Pb", ["Cl"] * 4, "tetrahedral"), 0, None, "PbCl4"),
    "Bi": (lambda: S.star("Bi", ["Cl"] * 3, "pyramidal", angle=97.0), 0, None, "BiCl3"),
    "Po": (lambda: S.star("Po", ["Cl"] * 2, "bent", angle=100.0), 0, None, "PoCl2"),
    "At": (lambda: S.diatomic("At", "Cl"), 0, None, "AtCl"),
    "He": (lambda: S.atom("He"), 0, None, "He atom"),
    "Ne": (lambda: S.diatomic("Ne", "Ne", 3.1), 0, None, "Ne2 (vdW)"),
    "Ar": (lambda: S.diatomic("Ar", "Ar", 3.8), 0, None, "Ar2 (vdW)"),
    "Kr": (lambda: S.atom("Kr"), 0, None, "Kr atom"),
}


def elements():
    out = []
    for z in range(1, 87):
        sym = S.SYMBOLS[z - 1]
        build, charge, uhf, label = ELEMENT_MOLECULES[sym]
        out.append(case(f"el-{z:02d}-{sym}", "elements", label, build(), charge, uhf, tags=[f"Z={z}"]))
    for sym, (build, charge, uhf, label) in ELEMENT_ALT.items():
        z = S.z_of(sym)
        out.append(case(f"el-{z:02d}-{sym}-alt", "elements", label, build(), charge, uhf, tags=[f"Z={z}"]))
    return out


# ---------------------------------------------------------------------------------------
# multi-element: random MB16-43-style molecules, light and heavy


HEAVY_POOL = ["B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Ti", "Fe", "Cu", "Zn", "Ge", "As", "Se", "Br",
              "Zr", "Mo", "Ag", "Sn", "Sb", "Te", "I", "La", "Gd", "W", "Pt", "Au", "Hg", "Pb", "Bi"]


def multielement():
    out = []
    for seed in range(16):
        mol = S.mindless(seed)
        nel = S.electrons(mol[0], 0)
        out.append(case(f"mb16-{seed:02d}", "multielement", "MB16-43-style random molecule (H-Cl)", mol,
                        0, nel % 2 or None, tags=["random"]))
    for seed in range(100, 116):
        mol = S.mindless(seed, natoms=12, elements=HEAVY_POOL, nh=(3, 6))
        nel = S.electrons(mol[0], 0)
        out.append(case(f"mbheavy-{seed}", "multielement", "random molecule with heavy elements", mol,
                        0, nel % 2 or None, tags=["random", "heavy"]))
    return out


# ---------------------------------------------------------------------------------------
# charges -3 .. +3 (closed shell and radicals)

CHARGED = [
    ("PO4_3-", -3, None, "PO4(3-)"), ("AlF6_3-", -3, None, "AlF6(3-)"), ("FeCN6_3-", -3, 1, "[Fe(CN)6]3- (low-spin d5)"),
    ("SO4_2-", -2, None, "SO4(2-)"), ("CO3_2-", -2, None, "CO3(2-)"), ("oxalate", -2, None, "C2O4(2-)"),
    ("SiF6_2-", -2, None, "SiF6(2-)"), ("O2_2-", -2, None, "O2(2-) peroxide"), ("CrO4_2-", -2, None, "CrO4(2-)"),
    ("acetate", -1, None, "acetate"), ("OH-", -1, None, "OH-"), ("CN-", -1, None, "CN-"),
    ("NO3-", -1, None, "NO3-"), ("BF4-", -1, None, "BF4-"), ("PF6-", -1, None, "PF6-"),
    ("ClO4-", -1, None, "ClO4-"), ("NO2-", -1, None, "NO2-"),
    ("O2-", -1, 1, "O2- superoxide (radical)"), ("CO2-", -1, 1, "CO2- (radical)"),
    ("NH4+", 1, None, "NH4+"), ("H3O+", 1, None, "H3O+"), ("methylammonium", 1, None, "CH3NH3+"),
    ("guanidinium", 1, None, "guanidinium"), ("CH3+", 1, None, "CH3+"),
    ("H2O+", 1, 1, "H2O+ (radical)"), ("NH3+", 1, 1, "NH3+ (radical)"), ("CH4+", 1, 1, "CH4+ (radical)"),
    ("benzene+", 1, 1, "C6H6+ (radical)"),
]

BARE_IONS = [("F", -1), ("Cl", -1), ("Br", -1), ("I", -1), ("Li", 1), ("Na", 1), ("K", 1), ("Mg", 2), ("Ca", 2),
             ("Al", 3)]

AQUA_IONS = [  # metal, charge, uhf (high spin), label
    ("Mg", 2, None, "[Mg(H2O)6]2+"), ("Al", 3, None, "[Al(H2O)6]3+"), ("Fe", 2, 4, "[Fe(H2O)6]2+ quintet"),
    ("Fe", 3, 5, "[Fe(H2O)6]3+ sextet"), ("Cr", 3, 3, "[Cr(H2O)6]3+ quartet"), ("Mn", 2, 5, "[Mn(H2O)6]2+ sextet"),
    ("V", 3, 2, "[V(H2O)6]3+ triplet"),
]


def charges():
    out = []
    for name, q, uhf, label in CHARGED:
        out.append(case(f"chg{q:+d}-{name}", "charges", label, S.molecule(name), q, uhf, tags=[f"charge={q:+d}"]))
    for sym, q in BARE_IONS:
        out.append(case(f"chg{q:+d}-ion-{sym}", "charges", f"bare {sym}{q:+d} ion", S.atom(sym), q, None,
                        tags=[f"charge={q:+d}", "atom"]))
    for metal, q, uhf, label in AQUA_IONS:
        out.append(case(f"chg{q:+d}-aqua-{metal}{'-hs' if uhf else ''}", "charges", label, S.aqua(metal), q, uhf,
                        tags=[f"charge={q:+d}"]))
    out.append(case("chg+3-CoNH3_6", "charges", "[Co(NH3)6]3+ low spin", S.ammine("Co", 6, r=1.97), 3, None,
                    tags=["charge=+3"]))
    out.append(case("chg+2-CuNH3_4", "charges", "[Cu(NH3)4]2+ doublet", S.ammine("Cu", 4, r=2.03), 2, 1,
                    tags=["charge=+2"]))
    return out


# ---------------------------------------------------------------------------------------
# multiplicities


def spin():
    out = []
    sp = [
        ("H-atom", S.atom("H"), 0, None, "H atom, default uhf (odd)"),
        ("H-atom-uhf1", S.atom("H"), 0, 1, "H atom doublet"),
        ("C-atom-triplet", S.atom("C"), 0, 2, "C atom triplet"),
        ("N-atom-quartet", S.atom("N"), 0, 3, "N atom quartet"),
        ("O-atom-triplet", S.atom("O"), 0, 2, "O atom triplet"),
        ("CH3", S.molecule("CH3"), 0, 1, "CH3 radical"),
        ("CH3-default", S.molecule("CH3"), 0, None, "CH3 radical, default uhf"),
        ("OH", S.molecule("OH"), 0, 1, "OH radical"),
        ("NO", S.molecule("NO"), 0, 1, "NO radical"),
        ("NO2", S.molecule("NO2"), 0, 1, "NO2 radical"),
        ("NO2-default", S.molecule("NO2"), 0, None, "NO2 radical, default uhf"),
        ("H2O-default", S.molecule("water"), 0, None, "water, default uhf (even)"),
        ("H2O-uhf0", S.molecule("water"), 0, 0, "water, explicit singlet"),
        ("O2-triplet", S.molecule("O2"), 0, 2, "O2 triplet"),
        ("O2-singlet", S.molecule("O2"), 0, 0, "O2 singlet (closed-shell occupation)"),
        ("O2-quintet", S.molecule("O2"), 0, 4, "O2 quintet"),
        ("CH2-triplet", S.molecule("CH2"), 0, 2, "CH2 triplet"),
        ("CH2-singlet", S.molecule("CH2_singlet"), 0, 0, "CH2 singlet"),
        ("H2-triplet", S.molecule("H2"), 0, 2, "H2 triplet"),
        ("Cr-aqua-quartet", S.aqua("Cr"), 3, 3, "[Cr(H2O)6]3+ quartet"),
        ("Cr-aqua-doublet", S.aqua("Cr"), 3, 1, "[Cr(H2O)6]3+ doublet"),
        ("Fe2-aqua-quintet", S.aqua("Fe"), 2, 4, "[Fe(H2O)6]2+ quintet"),
        ("Fe2-aqua-triplet", S.aqua("Fe"), 2, 2, "[Fe(H2O)6]2+ triplet"),
        ("Fe2-aqua-singlet", S.aqua("Fe"), 2, 0, "[Fe(H2O)6]2+ singlet"),
        ("Fe3-aqua-sextet", S.aqua("Fe"), 3, 5, "[Fe(H2O)6]3+ sextet"),
        ("Fe3-aqua-doublet", S.aqua("Fe"), 3, 1, "[Fe(H2O)6]3+ doublet"),
        ("Mn-aqua-sextet", S.aqua("Mn"), 2, 5, "[Mn(H2O)6]2+ sextet"),
        ("Mn-aqua-quartet", S.aqua("Mn"), 2, 3, "[Mn(H2O)6]2+ quartet"),
        ("Mn-aqua-default", S.aqua("Mn"), 2, None, "[Mn(H2O)6]2+ default uhf"),
        ("Co-aqua-quartet", S.aqua("Co"), 2, 3, "[Co(H2O)6]2+ quartet"),
        ("Ni-aqua-triplet", S.aqua("Ni"), 2, 2, "[Ni(H2O)6]2+ triplet"),
        ("FeCl4-sextet", S.star("Fe", ["Cl"] * 4, "tetrahedral", r=2.19), -1, 5, "FeCl4- sextet"),
        ("Fe4-uhf4", S.molecule("Fe4"), 0, 4, "Fe4 cluster quintet"),
        ("Cu5-default", S.molecule("Cu5"), 0, None, "Cu5 cluster, default uhf (odd)"),
    ]
    for cid, mol, q, uhf, label in sp:
        out.append(case(f"spin-{cid}", "spin", label, mol, q, uhf))
    return out


# ---------------------------------------------------------------------------------------
# hard open-shell transition-metal systems (several SCF solutions are possible)


def open_shell_metals():
    fc = S.read_xyz(S.ROOT / "tests/structures/ferrocene.xyz")
    sys_ = [
        ("Fe-atom-quintet", S.atom("Fe"), 0, 4), ("Mn-atom-sextet", S.atom("Mn"), 0, 5),
        ("Cr-atom-septet", S.atom("Cr"), 0, 6), ("Cu-atom", S.atom("Cu"), 0, None), ("Ni-atom-triplet", S.atom("Ni"), 0, 2),
        ("Fe2-septet", S.diatomic("Fe", "Fe", 2.02), 0, 6), ("Fe2-singlet", S.diatomic("Fe", "Fe", 2.02), 0, 0),
        ("Cr2", S.diatomic("Cr", "Cr", 1.68), 0, 0), ("Cr2-stretched", S.diatomic("Cr", "Cr", 2.50), 0, 0),
        ("Mn2-undectet", S.diatomic("Mn", "Mn", 2.60), 0, 10), ("Co2-quintet", S.diatomic("Co", "Co", 2.00), 0, 4),
        ("Ni2-triplet", S.diatomic("Ni", "Ni", 2.15), 0, 2),
        ("TiO-triplet", S.diatomic("Ti", "O", 1.62), 0, 2), ("VO-quartet", S.diatomic("V", "O", 1.59), 0, 3),
        ("CrO-quintet", S.diatomic("Cr", "O", 1.62), 0, 4), ("MnO-sextet", S.diatomic("Mn", "O", 1.65), 0, 5),
        ("FeO-quintet", S.diatomic("Fe", "O", 1.62), 0, 4), ("CuO-doublet", S.diatomic("Cu", "O", 1.72), 0, 1),
        ("ferrocenium", (fc[0], fc[1]), 1, 1),
        ("FeCO4-triplet", S.star("Fe", ["CO"] * 4, "tetrahedral", r=1.82), 0, 2),
        ("CoCl4-quartet", S.star("Co", ["Cl"] * 4, "tetrahedral", r=2.28), -2, 3),
        ("NiCl4-triplet", S.star("Ni", ["Cl"] * 4, "tetrahedral", r=2.27), -2, 2),
        ("MnCl4-sextet", S.star("Mn", ["Cl"] * 4, "tetrahedral", r=2.37), -2, 5),
        ("Mn3", S.star("Mn", ["Mn", "Mn"], "bent", r=2.70, angle=60.0), 0, None),
        ("Co4", S.cluster("Co", "tetrahedron", 2.30), 0, None), ("Ni4", S.cluster("Ni", "tetrahedron", 2.35), 0, None),
        ("Pt3", S.star("Pt", ["Pt", "Pt"], "bent", r=2.55, angle=60.0), 0, None),
        ("Au4", S.cluster("Au", "tetrahedron", 2.75), 0, None),
        ("Fe4-quintet-T1000", S.molecule("Fe4"), 0, 4),
    ]
    out = []
    for cid, mol, q, uhf in sys_:
        opts = {"etemp": 1000.0} if cid.endswith("T1000") else {}
        out.append(case(f"tm-{cid}", "open_shell_metals", cid.replace("-", " "), mol, q, uhf, tags=["hard"], **opts))
    return out


# ---------------------------------------------------------------------------------------
# solvation: every GFN2 ALPB/GBSA solvent, all reference states

ALPB_SOLVENTS = ["acetone", "acetonitrile", "aniline", "benzaldehyde", "benzene", "ch2cl2", "chcl3", "cs2",
                 "dioxane", "dmf", "dmso", "ether", "ethanol", "ethylacetate", "furane", "hexadecane",
                 "hexane", "methanol", "nitromethane", "octanol", "phenol", "thf", "toluene", "water",
                 "woctanol"]
GBSA_SOLVENTS = ["acetone", "acetonitrile", "benzene", "ch2cl2", "chcl3", "cs2", "dmf", "dmso", "ether",
                 "hexane", "methanol", "thf", "toluene", "water"]
STATES = ["gsolv", "bar1M", "reference"]
SOLV_SPECIES = [
    ("acetic_acid", 0, None, "acetic acid"), ("methylammonium", 1, None, "CH3NH3+"),
    ("acetate", -1, None, "acetate"), ("O2-", -1, 1, "O2- (radical anion)"),
]


def solvation():
    out = []
    for model, solvents in (("alpb", ALPB_SOLVENTS), ("gbsa", GBSA_SOLVENTS)):
        for solvent in solvents:
            for state in STATES:
                for name, q, uhf, label in SOLV_SPECIES:
                    out.append(case(f"solv-{model}-{solvent}-{state}-{name}", "solvation",
                                    f"{label} in {solvent} ({model.upper()}, {state})", S.molecule(name), q, uhf,
                                    checks=("xtb", "tblite", "axtb_vs_xtbmode"), tags=[model, solvent, state],
                                    **{model: solvent, "solvation_state": state}))
    # a dication and a trication in water, and aliases of solvent names
    for model in ("alpb", "gbsa"):
        for state in STATES:
            out.append(case(f"solv-{model}-water-{state}-Mg_aqua", "solvation",
                            f"[Mg(H2O)6]2+ in water ({model.upper()}, {state})", S.aqua("Mg"), 2, None,
                            checks=("xtb", "axtb_vs_xtbmode"), tags=[model, "water", state],
                            **{model: "water", "solvation_state": state}))
        out.append(case(f"solv-{model}-water-gsolv-Fe3_aqua", "solvation",
                        f"[Fe(H2O)6]3+ sextet in water ({model.upper()})", S.aqua("Fe"), 3, 5,
                        checks=("xtb", "axtb_vs_xtbmode"), tags=[model, "water", "gsolv"],
                        **{model: "water", "solvation_state": "gsolv"}))
    for model, alias in (("alpb", "h2o"), ("alpb", "n-hexane"), ("alpb", "dichlormethane"), ("alpb", "chloroform"),
                         ("gbsa", "h2o"), ("gbsa", "nhexan"), ("gbsa", "CHCl3")):
        out.append(case(f"solv-{model}-alias-{alias}", "solvation", f"acetate in '{alias}' ({model.upper()} alias)",
                        S.molecule("acetate"), -1, None, checks=("xtb", "axtb_vs_xtbmode"), tags=[model, "alias"],
                        **{model: alias, "solvation_state": "gsolv"}))
    return out


# ---------------------------------------------------------------------------------------
# electronic temperature x accuracy on small-gap systems

SMALL_GAP = [
    ("Fe4", S.molecule, "Fe4", 0, None, "Fe4 cluster"),
    ("Cu5", S.molecule, "Cu5", 0, None, "Cu5 cluster (doublet)"),
    ("Na8", S.molecule, "Na8", 0, None, "Na8 cluster"),
    ("N2-stretched", S.molecule, "N2_stretched", 0, None, "N2 at 1.65 Å"),
    ("ethylene-twisted", S.molecule, "ethylene_twisted", 0, None, "90°-twisted ethylene"),
    ("NO2", S.molecule, "NO2", 0, 1, "NO2 radical"),
]


def etemp_acc():
    out = []
    for cid, fn, arg, q, uhf, label in SMALL_GAP:
        mol = fn(arg)
        for et in (0.0, 300.0, 1000.0, 5000.0):
            for acc in (1.0, 1e-2, 1e-4, 1e-6):
                out.append(case(f"eta-{cid}-T{int(et)}-acc{acc:g}", "etemp_acc", f"{label}, {et:g} K, acc {acc:g}",
                                mol, q, uhf, tags=["small-gap", f"T={et:g}", f"acc={acc:g}"],
                                accuracy=acc, etemp=et))
    # xtb smears only above 0.1 K: degenerate frontier orbitals (singlet O2) show it
    for et in (0.1, 0.11, 1.0, 300.0):
        out.append(case(f"eta-O2-singlet-T{et:g}", "etemp_acc", f"O2 singlet (degenerate π*), {et:g} K",
                        S.molecule("O2"), 0, 0, tags=["degenerate", f"T={et:g}"], etemp=et))
    return out


# ---------------------------------------------------------------------------------------
# Hessians vs xtb --hess (neutral, charged, open-shell, solvated)


def hessian():
    hc = ("hessian",)
    return [
        case("hess-water", "hessian", "water", S.molecule("water"), 0, None, checks=hc),
        case("hess-NH4+", "hessian", "NH4+ (cation)", S.molecule("NH4+"), 1, None, checks=hc),
        case("hess-acetate", "hessian", "acetate (anion)", S.molecule("acetate"), -1, None, checks=hc),
        case("hess-NO2", "hessian", "NO2 (doublet)", S.molecule("NO2"), 0, 1, checks=hc),
        case("hess-CH2-triplet", "hessian", "CH2 triplet", S.molecule("CH2"), 0, 2, checks=hc),
        case("hess-O2-", "hessian", "O2- (radical anion)", S.molecule("O2-"), -1, 1, checks=hc),
        case("hess-acetate-alpb", "hessian", "acetate in ALPB water", S.molecule("acetate"), -1, None, checks=hc,
             alpb="water"),
    ]


# ---------------------------------------------------------------------------------------
# analytic Hessians: vs a tight response/numerical reference in all three conventions, and
# vs `xtb --hess` with tightened displacement SCFs ($hess sccacc)


def hessian_analytic():
    ref = ("analytic_ref", "analytic_xtb")

    def struct(name):
        nums, pos, _ = S.read_xyz(S.ROOT / "tests/structures" / f"{name}.xyz")
        return nums, pos

    # id, molecule, charge, uhf, options, extra checks, tags
    sys_ = [
        # closed shell, neutral
        ("water", S.molecule("water"), 0, None, {}, ("analytic_numref", "analytic_threads"), []),
        ("NH3", S.molecule("NH3"), 0, None, {}, (), []),
        ("CH4", S.molecule("CH4"), 0, None, {}, (), []),
        ("methanol", S.molecule("methanol"), 0, None, {}, ("analytic_numref",), []),
        ("acetic_acid", S.molecule("acetic_acid"), 0, None, {}, (), []),
        ("benzene", S.molecule("benzene"), 0, None, {}, (), []),
        ("caffeine", struct("caffeine"), 0, None, {}, ("analytic_numref", "analytic_threads"), []),
        # open shells
        ("NO2-doublet", S.molecule("NO2"), 0, 1, {}, ("analytic_numref",), ["open-shell"]),
        ("CH2-triplet", S.molecule("CH2"), 0, 2, {}, (), ["open-shell"]),
        ("O2-triplet", S.molecule("O2"), 0, 2, {}, ("analytic_numref",), ["open-shell", "linear"]),
        ("CH3-doublet", S.molecule("CH3"), 0, 1, {}, (), ["open-shell"]),
        ("NO-doublet", S.molecule("NO"), 0, 1, {}, (), ["open-shell", "linear"]),
        ("benzene+-doublet", S.molecule("benzene+"), 1, 1, {}, (), ["open-shell", "charged"]),
        # charged
        ("NH4+", S.molecule("NH4+"), 1, None, {}, (), ["charged"]),
        ("H3O+", S.molecule("H3O+"), 1, None, {}, (), ["charged"]),
        ("acetate", S.molecule("acetate"), -1, None, {}, ("analytic_numref",), ["charged"]),
        ("CO3_2-", S.molecule("CO3_2-"), -2, None, {}, (), ["charged"]),
        ("PO4_3-", S.molecule("PO4_3-"), -3, None, {}, (), ["charged"]),
        ("guanidinium", S.molecule("guanidinium"), 1, None, {}, (), ["charged"]),
        ("Mg-aqua-2+", S.aqua("Mg"), 2, None, {}, (), ["charged"]),
        # transition metals
        ("ferrocene", struct("ferrocene"), 0, None, {}, ("analytic_numref",), ["metal"]),
        ("Cu-NH3-4-2+", struct("cu-nh3-4"), 2, 1, {}, ("analytic_numref", "analytic_threads"),
         ["metal", "open-shell", "charged"]),
        ("Fe-aqua-3+-sextet", S.aqua("Fe"), 3, 5, {}, (), ["metal", "open-shell", "charged"]),
        ("NiCO4", S.star("Ni", ["CO"] * 4, "tetrahedral", r=1.84), 0, None, {}, (), ["metal"]),
        ("CrCO6", S.star("Cr", ["CO"] * 6, "octahedral", r=1.91), 0, None, {}, (), ["metal"]),
        ("MnO4-", S.molecule("MnO4-"), -1, None, {}, (), ["metal", "charged"]),
        ("ZnCl4_2-", struct("zncl4"), -2, None, {}, (), ["metal", "charged"]),
        ("CoCl4_2--quartet", S.star("Co", ["Cl"] * 4, "tetrahedral", r=2.28), -2, 3, {}, (),
         ["metal", "open-shell", "charged"]),
        ("TiCl4", S.star("Ti", ["Cl"] * 4, "tetrahedral"), 0, None, {}, (), ["metal"]),
        # implicit solvation
        ("acetate-alpb-water", S.molecule("acetate"), -1, None, {"alpb": "water"}, ("analytic_numref",),
         ["solvated", "charged"]),
        ("methylammonium-alpb-water-bar1M", S.molecule("methylammonium"), 1, None,
         {"alpb": "water", "solvation_state": "bar1M"}, (), ["solvated", "charged"]),
        ("caffeine-alpb-water", struct("caffeine"), 0, None, {"alpb": "water"}, (), ["solvated"]),
        ("O2--alpb-dmso", S.molecule("O2-"), -1, 1, {"alpb": "dmso", "solvation_state": "reference"}, (),
         ["solvated", "open-shell", "charged", "linear"]),
        ("benzene-alpb-toluene", S.molecule("benzene"), 0, None, {"alpb": "toluene"}, (), ["solvated"]),
        ("Mg-aqua-2+-alpb-water", S.aqua("Mg"), 2, None, {"alpb": "water"}, (), ["solvated", "charged"]),
        ("acetic_acid-gbsa-thf", S.molecule("acetic_acid"), 0, None, {"gbsa": "thf"}, (), ["solvated"]),
        ("glycine-gbsa-water", S.molecule("glycine_zwitterion"), 0, None, {"gbsa": "water"}, (), ["solvated"]),
        ("acetate-gbsa-acetonitrile", S.molecule("acetate"), -1, None, {"gbsa": "acetonitrile"}, (),
         ["solvated", "charged"]),
        # high electronic temperature: fractional occupations
        ("ethylene-twisted-T300", S.molecule("ethylene_twisted"), 0, None, {"etemp": 300.0}, ("analytic_numref",),
         ["fractional"]),
        ("ethylene-twisted-T3000", S.molecule("ethylene_twisted"), 0, None, {"etemp": 3000.0}, (),
         ["fractional"]),
        ("O2-singlet-T300", S.molecule("O2"), 0, 0, {"etemp": 300.0}, (), ["fractional", "linear"]),
        ("N2-stretched-T5000", S.molecule("N2_stretched"), 0, None, {"etemp": 5000.0}, (), ["fractional", "linear"]),
        ("Fe4-T5000", S.molecule("Fe4"), 0, None, {"etemp": 5000.0}, (), ["fractional", "metal"]),
        ("Cu5-T1000", S.molecule("Cu5"), 0, None, {"etemp": 1000.0}, (), ["fractional", "metal"]),
        ("Na8-T2000", S.molecule("Na8"), 0, None, {"etemp": 2000.0}, (), ["fractional"]),
    ]
    out = []
    for cid, mol, q, uhf, opts, more, tags in sys_:
        out.append(case(f"hana-{cid}", "hessian_analytic", cid.replace("-", " "), mol, q, uhf,
                        checks=ref + tuple(more), tags=list(tags), **opts))
    return out


# ---------------------------------------------------------------------------------------
# thread invariance (energies/gradients to 1e-12, Hessians bit-identical)


def threads():
    tc = ("threads",)
    out = []
    for name in ("caffeine", "taxol", "ferrocene", "cu-nh3-4"):
        nums, pos, q = S.read_xyz(S.ROOT / "tests/structures" / f"{name}.xyz")
        nel = S.electrons(nums, q)
        out.append(case(f"thr-{name}", "threads", f"{name} 1 vs 8 threads", (nums, pos), q, nel % 2 or None,
                        checks=tc))
    out.append(case("thr-Fe3-aqua-alpb", "threads", "[Fe(H2O)6]3+ sextet, ALPB water", S.aqua("Fe"), 3, 5,
                    checks=tc, alpb="water"))
    out.append(case("thr-Fe4-T5000", "threads", "Fe4 cluster at 5000 K", S.molecule("Fe4"), 0, None, checks=tc,
                    etemp=5000.0))
    for name in ("water", "NO2"):
        mol = S.molecule(name)
        uhf = 1 if name == "NO2" else None
        out.append(case(f"thr-hess-{name}", "threads", f"{name} Hessian 1 vs 8 threads", mol, 0, uhf,
                        checks=("threads_hessian",)))
    return out


# ---------------------------------------------------------------------------------------
# dispersion off vs tblite without its dispersion section


def nodisp():
    out = []
    mols = [("caffeine", None), ("ferrocene", None), ("zncl4", None), ("cf3br-benzene", None)]
    for name, _ in mols:
        nums, pos, q = S.read_xyz(S.ROOT / "tests/structures" / f"{name}.xyz")
        nel = S.electrons(nums, q)
        out.append(case(f"nodisp-{name}", "nodisp", f"{name}, dispersion off", (nums, pos), q, nel % 2 or None,
                        checks=("tblite",), dispersion=False))
    for cid, mol, q, uhf in [("acetate", S.molecule("acetate"), -1, None), ("Fe3-aqua", S.aqua("Fe"), 3, 5),
                             ("NO2", S.molecule("NO2"), 0, 1), ("PtCl4", S.star("Pt", ["Cl"] * 4, "square", r=2.31), -2,
                                                                None),
                             ("LaCl3", _lncl3("La"), 0, None), ("XeF2", S.star("Xe", ["F"] * 2, "linear", r=1.98), 0,
                                                                  None)]:
        out.append(case(f"nodisp-{cid}", "nodisp", f"{cid}, dispersion off", mol, q, uhf, checks=("tblite",),
                        dispersion=False))
    return out


# ---------------------------------------------------------------------------------------
# failure behaviour


def failures():
    fc = ("failure",)
    w = S.molecule("water")
    h2 = S.molecule("H2")
    out = [
        case("fail-scf-maxiter3", "failures", "water, max 3 SCF iterations", w, 0, None, checks=fc, max_iter=3,
             expect="error"),
        case("fail-scf-Fe4-T0", "failures", "Fe4 cluster at 0 K (charge sloshing)", S.molecule("Fe4"), 0, None,
             checks=fc, etemp=0.0, max_iter=60, expect="any"),
        case("fail-Z87", "failures", "FrH (Z=87, no GFN2 parameters)", S.diatomic("Fr", "H"), 0, None, checks=fc,
             expect="error"),
        case("fail-Z92", "failures", "UF6 (Z=92, no GFN2 parameters)", S.star("U", ["F"] * 6, "octahedral", r=2.0), 0,
             None, checks=fc, expect="error"),
        case("fail-Z118", "failures", "OgH (Z=118)", S.diatomic("Og", "H", 2.0), 0, None, checks=fc,
             expect="error"),
        case("fail-uhf-parity", "failures", "water with uhf=1 (parity mismatch)", w, 0, 1, checks=fc,
             expect="error"),
        case("fail-uhf0-odd", "failures", "NO2 with uhf=0 (odd electrons)", S.molecule("NO2"), 0, 0, checks=fc,
             expect="error"),
        case("fail-uhf-too-many", "failures", "H2 with uhf=4 (more unpaired than electrons)", h2, 0, 4, checks=fc,
             expect="error"),
        case("fail-uhf-gt-nao", "failures", "O2 with uhf=10 (alpha electrons exceed orbitals)",
             S.molecule("O2"), 0, 10, checks=fc, expect="error"),
        case("fail-charge-negative-nel", "failures", "H2 with charge +3 (negative electron count)", h2, 3, None,
             checks=fc, expect="error"),
        case("fail-charge-zero-nel", "failures", "H2 with charge +2 (no electrons)", h2, 2, None, checks=fc,
             expect="ok"),
        case("fail-charge-water+9", "failures", "water with charge +9 (negative electron count)", w, 9, None,
             checks=fc, expect="error"),
        case("fail-charge-too-negative", "failures", "H2 with charge -5 (more electrons than spin orbitals)", h2, -5,
             None, checks=fc, expect="error"),
        case("fail-charge-water-30", "failures", "water with charge -30", w, -30, None, checks=fc, expect="error"),
        case("fail-etemp-negative", "failures", "water with etemp -5 K", w, 0, None, checks=fc, etemp=-5.0,
             expect="error"),
        case("fail-acc-zero", "failures", "water with accuracy 0", w, 0, None, checks=fc, accuracy=0.0,
             expect="error"),
        case("fail-acc-too-small", "failures", "water with accuracy 1e-12 (xtb clamps to 1e-4)", w, 0, None,
             checks=fc, accuracy=1e-12, max_iter=60, expect="any"),
        case("fail-fractional-charge", "failures", "H2 with charge +0.5 (a non-integer charge is an input error)", h2,
             0.5, None, checks=fc, expect="error"),
        case("fail-unknown-solvent", "failures", "water in ALPB 'octane' (not parametrized)", w, 0, None, checks=fc,
             alpb="octane", expect="error"),
        case("fail-gbsa-only-alpb-solvent", "failures", "water in GBSA 'aniline' (ALPB-only solvent)", w, 0, None,
             checks=fc, gbsa="aniline", expect="error"),
        case("fail-overlapping-atoms", "failures", "two H atoms at the same position", (
            np.array([1, 1, 8]), np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.8]])), 0, None, checks=fc,
             expect="error"),
    ]
    return out


# ---------------------------------------------------------------------------------------
# API surfaces: Calculator vs singlepoint, CLI flags vs Python options


def api():
    out = []
    out.append(case("api-calculator-sequence", "api", "Calculator.singlepoint on a displaced sequence vs singlepoint",
                    S.molecule("acetate"), -1, None, checks=("calculator",), alpb="water"))
    out.append(case("api-calculator-open-shell", "api", "Calculator, [Fe(H2O)6]3+ sextet", S.aqua("Fe"), 3, 5,
                    checks=("calculator",)))
    cli = [
        ("default", S.molecule("acetic_acid"), 0, None, {}),
        ("charged-uhf", S.molecule("O2-"), -1, 1, {}),
        ("acc-etemp", S.molecule("Cu5"), 0, None, {"accuracy": 0.01, "etemp": 1000.0}),
        ("nodisp-xtb", S.molecule("acetate"), -1, None, {"dispersion": False}),
        ("alpb-bar1M", S.molecule("methylammonium"), 1, None, {"alpb": "methanol", "solvation_state": "bar1M"}),
        ("gbsa-reference", S.molecule("acetate"), -1, None, {"gbsa": "dmso", "solvation_state": "reference"}),
        ("threads", S.molecule("acetic_acid"), 0, None, {"threads": 4}),
    ]
    for cid, mol, q, uhf, opts in cli:
        for conv in ("axtb", "xtb", "tblite"):
            out.append(case(f"api-cli-{cid}-{conv}", "api", f"CLI flags vs Python API ({cid}, {conv})", mol, q, uhf,
                            checks=("cli",), cli_conventions=conv, **opts))
    return out


# ---------------------------------------------------------------------------------------
# the repository's validation structures (tests/structures/*.xyz, up to taxol)


def structures():
    out = []
    for path in sorted((S.ROOT / "tests/structures").glob("*.xyz")):
        nums, pos, q = S.read_xyz(path)
        nel = S.electrons(nums, q)
        # gh106 is written in bohr under an Å header: read as Å it is a blown-up, metallic
        # molecule (gap 0.005 eV) whose SCF needs ~250 iterations and has a residual noise
        # floor of ~7e-10, so 1e-6 accuracy is unreachable; it runs at xtb's 1e-4
        opts = {"accuracy": 1e-4, "max_iter": 1000} if path.stem == "gh106" else {}
        out.append(case(f"struct-{path.stem}", "structures", f"tests/structures/{path.name}", (nums, pos), q,
                        nel % 2 or None, **opts))
    return out


CATEGORIES = {
    "elements": elements,
    "structures": structures,
    "multielement": multielement,
    "charges": charges,
    "spin": spin,
    "open_shell_metals": open_shell_metals,
    "solvation": solvation,
    "etemp_acc": etemp_acc,
    "hessian": hessian,
    "hessian_analytic": hessian_analytic,
    "threads": threads,
    "nodisp": nodisp,
    "failures": failures,
    "api": api,
}


def all_cases(subsets=None):
    out = []
    for name, fn in CATEGORIES.items():
        if subsets and name not in subsets:
            continue
        out.extend(fn())
    ids = [c["id"] for c in out]
    assert len(ids) == len(set(ids)), "duplicate case ids"
    return out
