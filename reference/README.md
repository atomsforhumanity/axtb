# reference/: upstream sources and programs (not in the repository)

The table generators, a few validation scripts and the specifications read upstream sources and
programs from this directory. Everything in it except this file is git-ignored. Clone or unpack
what you need here; nothing is required to build axtb or to run `scripts/check.sh fast`.

| directory | upstream | version or commit | read by |
|---|---|---|---|
| `xtb-gxtb/` | [thfroitzheim/xtb](https://github.com/thfroitzheim/xtb), branch `gxtb`: the xtb side of the g-xTB 2.0.1 build (xtb 6.7.1, LGPL-3.0) | `97f7bf5` | `crates/axtb/scripts/gen_data.py` (xtb 6.7.1 tables), `crates/gfnff/scripts/gen_data.py`, `crates/solvation/scripts/gen_tables.py`; xtb's atomic masses (`src/param/atomicmass.f90`) for `tests/validate_hessian.py`, `tests/gxtb2_hessian.py`, `bench/hessian_benchmark.py`, `bench/ts_example.py`; `docs/gfnff-spec.md`, `docs/gxtb-v2.0.1-spec.md` |
| `xtb/` | [grimme-lab/xtb](https://github.com/grimme-lab/xtb) | tag `v6.8.0` (`4b8c987`) | `crates/axtb/scripts/gen_data.py` (`XTB_*` tables); `docs/gfnff-spec.md` §11 |
| `tblite-main/` | [tblite/tblite](https://github.com/tblite/tblite) | `a4b07ac` (main when the tables were generated) | `crates/axtb/scripts/gen_data.py` (GFN2-xTB parameters, Slater expansions) |
| `mctc-lib/` | [grimme-lab/mctc-lib](https://github.com/grimme-lab/mctc-lib) | `73fc4ca` (v0.7.0 data modules) | `crates/axtb/scripts/gen_data.py`, `crates/d4/scripts/gen_tables.py` |
| `multicharge/` | [grimme-lab/multicharge](https://github.com/grimme-lab/multicharge) | `3414b5f` | `crates/axtb/scripts/gen_data.py` (EEQ parameters); `docs/gxtb-v1.1-spec.md` (EEQ-BC) |
| `dftd4/` | [dftd4/dftd4](https://github.com/dftd4/dftd4) | v4.3.1 (the tables came from `1107ea7`, main at the time, with the same data) | `crates/d4/scripts/gen_tables.py` (`crates/d4/src/data.rs` records the commit) |
| `tblite-gxtb/` | [thfroitzheim/tblite](https://github.com/thfroitzheim/tblite), branch `g-xtb` | `e9abc39` | read for `docs/gxtb-v1.1-spec.md` and `docs/gxtb-v2.0.1-spec.md`; no script |
| `simple-dftd3/` | [dftd3/simple-dftd3](https://github.com/dftd3/simple-dftd3) | `84cb7cd` | the origin of several test structures (`tests/structures/README.md`); no script |
| `gxtb-v1.1.0/` | [grimme-lab/g-xtb](https://github.com/grimme-lab/g-xtb) release v1.1.0 (GPL-3.0): `gxtb`, `gxtb_parameters`, `basisq`, `eeq` | v1.1.0 | the g-xTB 1.1.0 program: set `AXTB_GXTB11_DIR` to this directory (`tests/programs.py`) |
| `gxtb-si/` | the supporting information of Froitzheim, Müller, Hansen, Grimme, ChemRxiv 2025, doi [10.26434/chemrxiv-2025-bjxvt](https://doi.org/10.26434/chemrxiv-2025-bjxvt) | — | `docs/gxtb-v2.0.1-spec.md` §7; not redistributable |

For example:

```bash
git clone -b gxtb https://github.com/thfroitzheim/xtb reference/xtb-gxtb && git -C reference/xtb-gxtb checkout 97f7bf5
git clone -b v6.8.0 --depth 1 https://github.com/grimme-lab/xtb reference/xtb
git clone https://github.com/dftd4/dftd4 reference/dftd4 && git -C reference/dftd4 checkout v4.3.1
```

The commits of `tblite-main`, `mctc-lib`, `multicharge`, `tblite-gxtb` and `simple-dftd3` are
the checkouts the committed tables were generated from; the scripts copy literals from the
Fortran sources, and a newer upstream commit gives the same tables unless the parameters
changed (the generators' output then shows up as a diff of the committed files).

The reference programs themselves (xtb 6.8.0, the g-xTB 2.0.1 build of xtb, the xtb 6.7.1
release, the g-xTB 1.1.0 program) can live anywhere; `tests/programs.py` finds them through
`AXTB_XTB680`, `AXTB_GXTB2`, `AXTB_XTB671` and `AXTB_GXTB11_DIR`. Scripts that read xtb's
atomic masses take this directory from `AXTB_REFERENCE` (default: `reference/` of the checkout).
