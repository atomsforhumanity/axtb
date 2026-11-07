"""Small test molecules covering further elements (rough textbook geometries, not minima)."""
import math
def xyz(name, atoms):
    return f"{len(atoms)}\n{name}\n" + "".join(f"{s} {x:.6f} {y:.6f} {z:.6f}\n" for s, x, y, z in atoms)
def diatomic(a, b, r): return [(a, 0, 0, 0), (b, 0, 0, r)]
def linear3(c, l, r): return [(c, 0, 0, 0), (l, 0, 0, r), (l, 0, 0, -r)]
def trig(c, l, r): return [(c, 0, 0, 0)] + [(l, r * math.cos(2 * math.pi * k / 3), r * math.sin(2 * math.pi * k / 3), 0) for k in range(3)]
def tetra(c, l, r):
    d = r / math.sqrt(3); return [(c, 0, 0, 0), (l, d, d, d), (l, d, -d, -d), (l, -d, d, -d), (l, -d, -d, d)]
def octa(c, l, r): return [(c, 0, 0, 0)] + [(l, *v) for v in ((r,0,0),(-r,0,0),(0,r,0),(0,-r,0),(0,0,r),(0,0,-r))]
def sqpl(c, l, r): return [(c, 0, 0, 0)] + [(l, *v) for v in ((r,0,0),(-r,0,0),(0,r,0),(0,-r,0))]
def bent(c, l, r, ang):
    a = math.radians(ang) / 2; return [(c, 0, 0, 0), (l, r * math.sin(a), 0, r * math.cos(a)), (l, -r * math.sin(a), 0, r * math.cos(a))]
def pyr(c, l, r, ang):
    # three ligands at bond angle ang
    t = math.radians(ang); cz = math.sqrt(max(0.0, 1 - (2 / 3) * (1 - math.cos(t)) * 2 / 2))
    s = math.sqrt(2 * (1 - math.cos(t)) / 3)
    z = -math.sqrt(max(0.0, 1 - s * s))
    return [(c, 0, 0, 0)] + [(l, r * s * math.cos(2 * math.pi * k / 3), r * s * math.sin(2 * math.pi * k / 3), r * z) for k in range(3)]
def mco6(m, rmc, rco):
    out = [(m, 0, 0, 0)]
    for v in ((1,0,0),(-1,0,0),(0,1,0),(0,-1,0),(0,0,1),(0,0,-1)):
        out.append(("C", *(rmc * x for x in v))); out.append(("O", *((rmc + rco) * x for x in v)))
    return out
MOLS = {  # name: (atoms, charge, uhf)
    "LiF": (diatomic("Li", "F", 1.564), 0, 0),
    "BF3": (trig("B", "F", 1.313), 0, 0),
    "NaCl": (diatomic("Na", "Cl", 2.361), 0, 0),
    "AlCl3": (trig("Al", "Cl", 2.06), 0, 0),
    "HBr": (diatomic("H", "Br", 1.414), 0, 0),
    "H2Se": (bent("Se", "H", 1.46, 91.0), 0, 0),
    "SnH4": (tetra("Sn", "H", 1.711), 0, 0),
    "XeF2": (linear3("Xe", "F", 1.977), 0, 0),
    "KCl": (diatomic("K", "Cl", 2.667), 0, 0),
    "CaCl2": (linear3("Ca", "Cl", 2.48), 0, 0),
    "TiCl4": (tetra("Ti", "Cl", 2.17), 0, 0),
    "Cr(CO)6": (mco6("Cr", 1.916, 1.141), 0, 0),
    "ZnCl2": (linear3("Zn", "Cl", 2.07), 0, 0),
    "AgCl": (diatomic("Ag", "Cl", 2.28), 0, 0),
    "PtCl4(2-)": (sqpl("Pt", "Cl", 2.31), -2, 0),
    "AuCl": (diatomic("Au", "Cl", 2.199), 0, 0),
    "HgCl2": (linear3("Hg", "Cl", 2.25), 0, 0),
    "PbH4": (tetra("Pb", "H", 1.75), 0, 0),
    "BiH3": (pyr("Bi", "H", 1.78, 90.5), 0, 0),
    "WF6": (octa("W", "F", 1.832), 0, 0),
    "LaCl3": (trig("La", "Cl", 2.62), 0, 0),
    "CeCl3": (trig("Ce", "Cl", 2.57), 0, 1),
    "EuCl3": (trig("Eu", "Cl", 2.52), 0, 6),
    "GdCl3": (trig("Gd", "Cl", 2.49), 0, 7),
    "LuCl3": (trig("Lu", "Cl", 2.42), 0, 0),
    "ThCl4": (tetra("Th", "Cl", 2.58), 0, 0),
    "UF6": (octa("U", "F", 1.996), 0, 0),
    "NpF6": (octa("Np", "F", 1.98), 0, 1),
    # stage 2: more f elements (open f shells, lanthanide hydrides, an actinide oxide)
    "LaH3": (pyr("La", "H", 2.14, 116.0), 0, 0),
    "GdH3": (pyr("Gd", "H", 2.05, 116.0), 0, 7),
    "TmH3": (pyr("Tm", "H", 2.0, 116.0), 0, 2),
    "ErF3": (trig("Er", "F", 2.02), 0, 3),
    "TbCl3": (trig("Tb", "Cl", 2.46), 0, 6),
    "YbCl3": (trig("Yb", "Cl", 2.43), 0, 1),
    "AmCl3": (trig("Am", "Cl", 2.55), 0, 6),
    "PuO2": (linear3("Pu", "O", 1.82), 0, 4),
}
def stage3():
    """Element coverage (tests/gxtb_elements.py): generated molecules of the elements and
    environments the cases above leave out, named by their formula."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))
    import gxtb_elements as G
    mols = {m["id"]: m for m in G.molecules()}
    for cid in G.REFERENCE_IDS_V2:
        m = mols[cid]
        yield {"name": m["label"].split(" (")[0], "xyz": G.xyz_text(m), "charge": m["charge"], "uhf": m["uhf"]}
def cases():
    for n, (a, q, u) in MOLS.items():
        yield {"name": n, "xyz": xyz(n, a), "charge": q, "uhf": u}
    yield from stage3()
