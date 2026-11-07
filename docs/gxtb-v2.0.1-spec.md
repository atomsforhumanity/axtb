# g-xTB 2.0.1 specification: delta to v1.1 (for axtb)

What the g-xTB 2.0.1 program (`xtb --gxtb`) computes, written as a **delta to
`docs/gxtb-v1.1-spec.md`** (section numbers below follow it; "v1.1 §x" refers to that document).
Every term was first decoded from the program and checked in a Python model against values
read from the running program, then implemented in `crates/gxtb/src/v2` (method
`"gxtb-v2.0.1"`); its status, the reproduction numbers and the details found while
implementing are in §10.

**How 2.0.1 was reproduced.** The source code of g-xTB 2.0.1 is not public; its release
binaries are distributed under GPL-3.0. The parameters and the details of the terms in this
document were obtained from the released binary: from its outputs, and by inspecting the
running program with a debugger (gdb) at the capture points listed in Appendix A, together with
the disassembly of the release's object files (Sources, below). axtb's results are checked
against the program's outputs (§5, §10).

**Headline.** 2.0.1 is the same method family as v1.1 (same energy terms, same q-vSZP basis,
same reference-data machinery), but **every element parameter is refitted, all 103 elements are
parametrised (v1.1: 79), most global constants changed, and there are three new terms** (an
H0 dipole/anisotropy correction, an exchange bond-order correction, charge-dependent onsite
exchange integrals) plus new functional forms in EEQ-BC, repulsion (now self-consistent), third
and fourth order, AES, onsite exchange and revD4 damping. The SCF is the tblite one (Fock-matrix
DIIS). Unlike 1.1.0, **the program's Fock matrix is the exact derivative of its energy** and the
program has analytic gradients. Total energies differ from 1.1.0 by up to 2e-2 Eh (water by
chance only 5.2e-5).

**Sources**

- Program: g-xTB 2.0.1 = xtb 6.7.1 (26dd68d, built 2026-05-14) with a private tblite fork
  (`thfroitzheim/save_tblite`), grimme-lab/g-xtb release, GPL-3.0 (binary `bin/xtb`, library `lib/libxtb.a`). Statically linked Intel Fortran, unstripped. The release also
  ships `lib/libxtb.a` with one object per Fortran source file; `objdump -dr` of those objects
  gives disassembly **with relocations** (callee names, module arrays, literal pools), which is
  how this document was decoded.
- Public code (read, licences respected): the xtb side of 2.0.1 (thfroitzheim/xtb branch
  `gxtb`, `reference/xtb-gxtb`, LGPL), public tblite main and the thfroitzheim tblite branches
  `g-xtb`, `gp3`, `mulliken-FX` (LGPL; early 2024 versions of q-vSZP, repulsion and Mulliken
  exchange used as templates), multicharge, mctc-lib, dftd4 (incl. the public thfroitzheim
  `dftd5` branch for the structure of the restructured damping code). No code was copied.
- Paper and SI: Froitzheim, Müller, Hansen, Grimme, ChemRxiv 2025 (doi
  10.26434/chemrxiv-2025-bjxvt) and its 110-page SI (`reference/gxtb-si/`, git-ignored). **The
  SI describes the v1.x method, not 2.0.1**: none of the 2.0.1 additions appear in it ("nuclear
  gradients will be part of a future version"; revD4 "still subject to changes", p. 83). §7
  cross-checks the v1.1 specification against the SI.

**Confidence labels** (as in v1.1, with [C] adapted to the decoding)

| Label | Meaning |
|---|---|
| **[C]** | Reproduced numerically: a Python model (parameters extracted from the binary, formula decoded) equals values read from the running 2.0.1 program with gdb, to ~1e-15 unless stated; molecules named. |
| **[D]** | Decoded from disassembly, call arguments or data, not reproduced numerically. |
| **[P]** | From the paper/SI. |
| **[U]** | Unknown. |

---

## 0. Differences at a glance

| term (v1.1 §) | 1.1.0 | 2.0.1 | label |
|---|---|---|---|
| program, I/O (§1) | `gxtb -c`, parameter files, numerical gradient | `xtb --gxtb`, compiled-in parameters, analytic gradient | [C] |
| Å → bohr (§1.2) | 1/0.52917726_f32 | 1.8897261246204404 (mctc-lib) | [C] |
| element coverage | 79 elements (no 59–69, 90, 91, 93–103) | all 103 | [C] |
| electron count (§4.17) | Σ Z_l (integers) | Σ n^ref_l (doubles, not integral: water 7.9999999999502) | [C] |
| g-xTB CN (§2) | erf 1.8217544049, Rc = (r_A+r_B)/2 | erf **2.068**, Rc = r_A + r_B, 25 bohr cutoff | [C] |
| √CN everywhere | √CN | **√(CN + 1e-12) − 1e-6** (q_eff, repulsion α, μ⁽¹⁾, U⁽²⁾, AES dipole scale) | [C] |
| EEQ-BC (§4.1) | eeqbc2025 parameters | refitted; **no kqeta term**; kcnrad 0.14→**0.3**; capacitance steepness 0.6→**0.75**; per-element vdW-radius scale | [C] |
| q-vSZP basis (§4.2) | `basisq` | **bit-identical tables**; q_eff with smoothed √CN | [C] |
| increments (§4.3) | Σ at_A, H = He = 0 | refitted; **He ≠ 0** | [C] |
| repulsion (§4.4) | EEQ-BC charges, fixed; 2αα/(α+α); k/Rⁿ | **SCF Mulliken charges (self-consistent)**; αα/(α+α); added Gaussian; Rⁿ terms in units of a vdW pair radius; per-element k1 | [C] |
| H0 (§4.5) | linear Π, δ-scale 5 | **new R² term in Π**, H radius override, d–f ×1.5, per-element δ, refitted | [C] (d–f [D]) |
| H0 anisotropy (new, §4.5.1) | — | **ΔH = −½[a_B·D^B + a_A·D^A]**, a_A from a damped sum of bond vectors | [C] |
| ACP (§4.6) | w = c(1 + k√CN), s,p,d,f everywhere | **w = c (no CN)**, element-specific projector sets | [C] |
| first order (§4.7) | k_dis 0.012; offsite term here | 0.0125; offsite term moved to E(2) (same value) | [C] |
| second order (§4.8) | k = 0.33 | k = 0.294621155 | [C] |
| AES (§4.9) | constants 0.4667/0.2287/0.0523/−0.003, 0.45/1.05; R⁰ = D3 radius (Å number) | new constants; R⁰ scaled per element; **CN-scaled atomic dipoles**; dQ and QQ potentials included | [C] |
| spin (§4.10) | W = k^W W⁰ | same form; W⁰ identical, k^W refitted | [C] |
| third order (§4.11) | ⅓ Σ q_l q_l' q_A Γ_A τ; 1.6/1.15; onsite −½η² | **⅓ Σ q_l² Γ_l (τq)_l**, Γ_l shell-scaled; 2.3/1.3; onsite −0.2093327496 η² | [C] |
| fourth order (§4.12) | Σ (0.021/4) q_l⁴ | **(1/24) Σ 0.036 k_l q_l⁴** | [C] |
| MFX (§4.13) | ω 0.2347, k^x1 −0.596, k^x2 0.214, G, 1.39 | same form; ω 0.2, k^x1 −0.296, k^x2 0.035, new G, 1.3827; R⁰ scaled | [C] |
| bond-order correction (new, §4.13.1) | — | **Σ_{A<B} K_AB · B^Mayer_AB** | [C] |
| onsite exchange (§4.14) | dual-density form, Λ = 0.15 Λ⁰ | **Mulliken form, Λ charge dependent** Λ = 0.15 Λ⁰[1 − ½(k_l q_l + k_l' q_l')] | [C] |
| revD4 (§4.15) | BJ (a1, s8), zero-damped ATM s9 1.75 | same C6 model, new g-xTB reference charges; **screened BJ** two- and three-body, s8 = 0, s9 1.826 | [C] |
| Fock (§4.16) | two non-exact parts (AES dQ/QQ, OFX ×2) | **exact derivative** for every term | [C] |
| SCF (§4.17) | damping 0.1 → Pulay (packed, per spin), extra final iteration | **tblite**: none → damping 0.2 → Pulay on full Fock (joint spins), no extra iteration | [C] |
| gradient (§4.18) | numerical | analytic, equals finite differences of the energy to 1e-10 | [C] |

---

## 1. The program

### 1.1 Running it [C]

```
cd <workdir> && OMP_NUM_THREADS=1 HOME=<workdir> xtb mol.xyz --gxtb --norestart [--chrg Q] [--uhf U] [--grad] [--acc A]
```

- `--gxtb` selects tblite's `new_gxtb_calculator` (xtb side public: `reference/xtb-gxtb`,
  `src/tblite/calculator.F90`). Electronic temperature **0 K** unless `--etemp` is given (then
  Fermi smearing, §4.17.1; `--etemp 300` moves water by −2.1e-11 Eh: the fractional electron
  count is rounded). **Any `--uhf`, even `--uhf 0`,
  switches to two spin channels** (also an odd electron count); closed-shell energies are the
  same either way (water identical). Start guess: the calculator default `eeqbc` (§4.17). At most 250 SCF
  iterations. `--acc A` scales the convergence thresholds (1e-6·A Eh, 2e-5·A) and the integral
  cutoff.
- **`--norestart` is required for reference runs**: xtb writes `xtbrestart` and otherwise starts
  the next run from it (converging in 2 cycles; §4.17.2).
- No parameter files: everything is compiled in (§3).
- Outputs: printout (§1.3), `gradient` (Turbomole format, analytic, §4.18), `energy`, `charges`,
  `wbo`, `xtbtopo.mol`, `xtbrestart`. The printout contains the program path and the build host;
  `make_reference_v2.py` strips both.
- CPU dispatch: hot routines exist as `.a` (AVX-512; runs on the reference laptop, i9-11900H),
  `.V` (AVX2) and `.A` (generic). The `.a` code was read.

### 1.2 Conventions [C]

- Å → bohr: **1.8897261246204404** = mctc-lib `aatoau` = 1/(ħ/(m_e c α)·10¹⁰), CODATA 2018
  constants folded in double (read from the program's `mol%xyz` for H2 at 1 Å). This is
  `axtb::units::MCTC_AATOAU`. Kernels evaluated with CODATA 2018's 0.529177210903 instead deviate
  by ~1e-12 relative. autoaa = 0.5291772109044924 appears as a literal throughout.
- tblite conventions: spherical AOs in tblite order, Mulliken shell charges q_l = n⁰_l −
  population, atomic charges q_A = Σ_l q_l, potentials in the (charge, magnetisation)
  representation, both spin channels converted with (h_c + h_m, h_c − h_m) [D].
- **Electron count**: N = Σ_A Σ_l n^ref_l(Z_A) − Q with the compiled-in reference occupations
  (`ps_reference_occ`, §3.3), which are doubles whose per-element sums are integral only to
  ~1e-10: water N = 7.9999999999502 (printed "number of electrons"). With 0 K filling the HOMO of
  each channel carries 1 − 2.5e-11 (§4.17).

### 1.3 Total energy as printed [C]

```
core increment energy        Σ_A ΔE^incr_A                          (before the SCF)
dispersion energy            revD4 three-body (ATM) energy           (before the SCF)
cycle  total energy  energy error  density error                     (§4.17)
electronic energy            all self-consistent terms, INCLUDING repulsion and revD4 two-body
total energy                 = increments + three-body + electronic
```

electronic = Tr(H P) + E(1) + E(2) + E(3) + E(4) + E_AES + E_spin + E_x + E_disp,2b + E_rep,
H = H0 (incl. the anisotropy term) + ACP. There is no per-term printout (`--verbose` adds
nothing). Per-term energies of the last SCF iteration were read with gdb (finish breakpoints on
each container's `get_energy`, difference of the summed per-atom energy array) and are stored as
`terms` in `tests/reference_v2/*/*/summary.json`.

### 1.4 Routine map [D]

`tblite_xtb_gxtb::new_gxtb_calculator` (private `src/tblite/xtb/gxtb.f90`) calls, in order:
`add_charge_model` (multicharge `new_eeqbc2025_model`), `add_basis` (q-vSZP, SCF and H0
bases), `add_acp`, `add_ncoord` (g-xTB CN), `add_increment`, `add_hamiltonian`
(`new_gxtb_h0spec`), `add_repulsion` (`tblite_repulsion_gxtb`), `add_dispersion`
(`tblite_disp_d4::new_d4srev_dispersion`, dftd4 model `d4srev`), `add_coulomb` (one
`tb_coulomb` container: `onsite_firstorder`, `effective_coulomb`, `twobody_thirdorder`,
`onsite_fourthorder`, `gxtb_multipole`), `add_exchange` (`tblite_exchange_fock`, 23 arguments),
`add_spin_polarization` (`tblite_spin`), `add_iterator` (mixers). Parameter accessors in the same
module: `get_selfenergy`, `get_cnshift`, `get_hscale`, `get_shpoly2`, `get_shpoly4`,
`get_diat_scale`, `get_rad`, `get_rvdw`, `get_anisotropy`, `get_reference_occ`,
`get_reference_shell_charge`, `get_shell_ipea`, `get_shell_hardness`, `get_hubbard_derivs`,
`get_hubbard_second_derivs`, `get_shell_hardness_exchange`,
`get_exchange_averaging_exponents`, `get_onecenter_exchange_ints`,
`get_offdiagonal_exchange_scaling`, `get_charge_exchange_fraction_scaling`,
`get_spin_constants`, and `export_gxtb_param` (TOML export; not reachable from the xtb command
line). Driver `tblite_xtb_singlepoint`: `get_selfenergy` → `get_anisotropy` → `get_hamiltonian`
→ `get_acp` → SCF loop (`iterator%update`, `next_scf`, `check_convergence`) → analytic gradient
(`get_hamiltonian_gradient`, `get_anisotropy_gradient`, `get_acp_gradient`, container gradients).

---

## 2. Coordination numbers

| CN | 2.0.1 formula | used by | change vs v1.1 §2 | status |
|---|---|---|---|---|
| EEQ-BC CN | Σ_B ½[1 + erf(−2 (R − Rc)/Rc^0.75)], Rc = cov_A + cov_B (`eeqbc_cov_radii`, already halved), 25 bohr cutoff | EEQ-BC | radii refitted (no element equals v1.1 `eeq` column 4 / 2); cutoff | [C] |
| basis CN | Σ_B ½[1 + erf(−3.75 (R − Rc)/Rc)], Rc = R_A + R_B (`qvszp_cov_radii` = v1.1 `ncoord_basq$RCOV`, bit-identical), 25 bohr | q_eff | cutoff only | [C] |
| g-xTB CN | Σ_B ½[1 + erf(**−2.068** (R − Rc)/Rc)], **Rc = rcov_A + rcov_B**, rcov = `pa_cn_rcov` (bohr), 25 bohr | H0 self-energies (raw CN); repulsion α, μ⁽¹⁾, U⁽²⁾, AES dipole scale (smoothed √CN) | steepness, radius sum, refitted radii, cutoff; no longer used by ACP | [C] water, O2, mesyl chloride, Ni(CO)4, [CuCl4]²⁻ 2e-16 |
| revD4 CN | Σ_B ½[1 + erf(−7.5 (R − Rc)/Rc)], Rc = R_A + R_B (dftd4 `covalent_rad_d3`), 30 bohr | revD4 weights | radii now 4/3 Pyykkö × 1/0.529177210903 (CODATA 2018; v1.1 1/0.52917726), relative change 2.2e-7 | [C] |

**Smoothed square root** [C]: every √CN of v1.1 becomes f(CN) = √(CN + 1e-12) − 1e-6 (literals
in the routines' literal pools), which is differentiable at CN = 0. The H0 self-energy keeps the
raw CN.

---

## 3. Parameters

All arrays were extracted at full precision from the binary's data section by symbol name and
are stored in `crates/gxtb/tests/reference_v2/params/{hamiltonian,classical_disp,
electrostatics_spin,exchange}.json` (with per-array diffs against v1.1). Naming in module
`tblite_xtb_gxtb`: `pa_*` per element (103), `ps_*` per shell × element (4 × 103, Fortran
order, shells in q-vSZP basis order), `pg_*` per angular momentum (4). Every element array is
non-zero for Z = 1–103.

### 3.1 Element arrays → roles [C unless stated]

| 2.0.1 array | role (§) | v1.1 counterpart | status |
|---|---|---|---|
| `pa_nshell` | shells used per element (1: H, He; 2: Li–Na, K, Zn, Rb, Cd, Cs, Hg, Fr; 3: Mg–Ar, Ca–Cu, Ga–Kr, Sr–Ag, In–Xe, Ba, La, Hf–Au, Tl–Rn, Ra; 4: Ce–Lu, Ac–Lr) | `basisq` shell count | identical where v1.1 has the element |
| `pa_cn_rcov` | g-xTB CN radius (§2) | `rad` (L1 C5) | refitted (median 8 %), now summed |
| `ps_h0_selfenergy`, `ps_h0_selfenergy_cn` | H0 levels se − se_cn·CN (§4.5) | −`lev`, −`lev_cn` | refitted (16 %, 62 %), sign convention flipped |
| `ps_h0_qvszp_exp_scal` | H0-basis exponent scale (§4.5) | `pln` | refitted (4 %) |
| `pa_h0_qvszp_k{0,2,3}_scal` | H0-basis q_eff weights (§4.5) | `kqvszp(1:3)` | refitted |
| `pa_h0_shpoly2` | distance-polynomial k^shp (§4.5) | `poly` | refitted (35 %) |
| `ps_h0_diat_scale` (3 × 103) | σ, π, **δ** diatomic-frame scales (§4.5) | `h0k(1:2)`, δ = 5 | refitted; δ column new (3.5, 4.0, 4.5) |
| `pa_h0_dip_scale` | anisotropy strength (§4.5.1) | — | **new** |
| `pa_rvdw_scale` | vdW pair-radius scale (§3.2) | — | **new** (default 0.5291772109044924) |
| `pa_nacp`, `pa_l_acp`, `ps_acp_level`, `ps_acp_exp` | ACP projectors (§4.6) | `acp(1,l)`, `acp(2,l)` (all s, p, d, f) | refitted (42 %, 23 %); sets new; `acpcn` removed |
| `pa_increment` | ΔE^incr (§4.3) | `at` | refitted, He ≠ 0 |
| `pa_rep_alpha`, `_roffset`, `_zeff`, `_cn`, `_q`, `_k1` | repulsion (§4.4) | `rep(1:5)`, glob 4/9 | refitted; `_k1` per element new; `rep(6)` removed |
| `ps_tb1_ipea`, `pa_tb1_ipea_cn` | μ⁰_l, k^(1),CN (§4.7) | `gam1`, `e1cn` | refitted |
| `ps_tb1_zeffsh` | Z_l (§4.7) | `com::zeff_shell` | identical |
| `ps_reference_occ` | n^ref_l (§4.7, §4.17) | `shellocc_ref_` | identical to float32 rounding for 74 elements; **new for Ba, Ce–Lu, Th–Lr** (29 elements, up to 0.92 e; f occupations non-integer) |
| `ps_tb2_shell_hubbard`, `pa_tb2_hubbard_cn` | U⁽²⁾ (§4.8) | `gam2`, `g2cn` | refitted |
| `pa_hubbard_parameter` | element hardness, factor of U⁽²⁾ and U^x | `com::gam` | same values in double precision (v1.1 float32, ≤ 2.5e-8) |
| `pa_tb3_hubbard_derivs` | Γ_A (§4.11) | `gp3_gam3` | refitted |
| `pa_aes_dip_scale` | AES dipole CN scale (§4.9) | — | **new** |
| `pa_wll_scale` | k^W (§4.10) | `atspin` | refitted |
| `ps_fock_shell_hubbard` | U^x_l / hardness (§4.13) | `fam` | refitted (30 %) |
| `ps_fock_avg_exp` | ξ_l (§4.13) | `bas_pol` + 1 | identical except Ar, Kr, Xe, Rn d: 1 (v1.1 2) |
| `pa_fock_cscale`, `pa_fock_crad` | bond-order correction (§4.13.1) | — | **new** |
| `pa_cn_average` | only read by `export_gxtb_param`; = EEQ-BC `avg_cn` | — | unused by the energy [D] |

### 3.2 Global constants

| symbol (2.0.1) | value | v1.1 | used in |
|---|---|---|---|
| `pg_h0_kshell` | 0.7129720394, 0.7981384369, 1.425, 2.85 | glob(1), glob(2), 1.4, 2.85 | H0 K_l |
| `pg_h0_shpoly2` | 1, 1.5, 2, 2.5 | 1, glob(3) = 1.1995, 2, 2 | H0 F_l |
| H0 R² coefficient | 0.0804696812 | — | §4.5 |
| H0 d–f kshell factor | 1.5 | — | §4.5 [D] |
| anisotropy steepness | 1.5 | — | §4.5.1 |
| g-xTB CN | 2.068 | 1.8217544049 | §2 |
| EEQ-BC kcnrad, kbc, norm_exp, kcn, cutoff | 0.3, 0.75, 0.75, 2, 25 | 0.14, 0.6, 0.75, 2, — | §4.1 |
| repulsion k_s, k_a, k3, k4, kpen(Z ≤ 2), kpen(Z > 2), exponents | 0.0046511298, 0.73, 0.011095539524126988, 0.011607795128002491, 0.012098131381864387, 0.008544252691968662, 1.5 and 2 | glob 13–15, 4, 9 | §4.4 |
| first order k_dis, k_x, k_s | 0.0125 (passed as 0.025 × ½), 1.0, 0.666666666 | 0.012, 1, 0.666666666 | §4.7 |
| second-order k | 0.294621155 | glob(12) = 0.3300126723 | §4.8 |
| `pg_tb3_kshell`, third order k^(3), k^(3),x, onsite | (1, 1.15, 1.3, 1.45); 2.3; 1.3; −0.2093327496 | (1, 1, 1, 1); 1.6; 1.15; −0.5 | §4.11 |
| `pg_tb4_kshell`, Γ⁽⁴⁾ | (1, 1.15, 1.3, 1.45); 0.036 (with 1/24) | 0.021 (with 1/4) | §4.12 |
| AES k3, k5, k7, k9 / steepness | 0.3405910191, 0.1691310614, 0.074034339, −0.02 / 0.5, 1, 1, 1 | 0.4667, 0.2287, 0.0523, −0.003 / 0.45, 1.05, 1.05, 1.05 | §4.9 |
| MFX α, 1 − α, ω | 0.15, 0.85, 0.2 | 0.15, 0.85, 0.2347047181 | §4.13 |
| MFX k^x1, k^x2 | −0.2960502355, 0.0349817377 (stored as 0.018511538388678535 = ×autoaa) | −0.5959929766, 0.2140456651 | §4.13 |
| `pg_fock_offdiag_l` (G_l), same-shell factor | 0.05, 2.5, 2.5, 2.5; 1.3826597204 | glob(5), glob(6) ×3; 1.39 | §4.13 |
| `pg_fock_kq` | 1.1, 0.55, 0.275, 0.1375 | — | §4.14 |
| revD4 s6, s8, s9, a1, a2, a3, a4, α | 1, **0**, 1.826410691, 0, 8.5302496868, 0.5395219242, 0.654, 16 (unused) | 1, 0.3042947280, 1.75, 1.2154627292, 0, —, —, 16 | §4.15 |
| revD4 cutoffs CN / two-body / three-body (bohr) | 30 / 50 / 25 | 30 / 60 / 20 | §4.15 [D] |
| SCF damping, DIIS memory, thresholds | 0.2; 7; 1e-6·acc Eh, 2e-5·acc | 0.1; 7 (8); 1e-8, 2e-5 | §4.17 |

**Shared vdW pair radius** [C]. Four terms use R̄_AB = ½(s_A + s_B)·R^D3_AB with R^D3_AB the
mctc-lib D3 pair radius in bohr (`mctc_data_vdwrad_mp_vdwrad_`, extract bit-exactly) and
s = `pa_rvdw_scale`. The default s = autoaa makes R̄ numerically the radius in Å, i.e. v1.1's
"Å number in a bohr formula"; fitted elements deviate (O 0.5715113878, Fe 0.582...). Users:
H0 anisotropy (§4.5.1), AES (§4.9), repulsion (§4.4), MFX and bond-order correction (§4.13).
EEQ-BC has its own scale (§4.1).

### 3.3 Compiled-in tables [C]

- q-vSZP (`tblite_basis_qvszp_mp_{exponents,coefficients,coefficients_env,n_prim,ang_shell,
  nshell,p_k0..p_k3,qvszp_cov_radii}_`): **bit-identical** to v1.1 `basisq` and
  `ncoord_basq$RCOV` for every shell used (p_k0 = 1, p_k1 = k_q2, p_k2 = k_cn, p_k3 = k_qcn). The
  module also holds polarization shells for H–Ne (`nshell_`(H) = 2) that g-xTB does not use.
- Onsite exchange integrals `tblite_data_onecxints` (103 × 10, packed (l,l') incl. ss = 0):
  bit-identical to v1.1 `com::gmunu`.
- Spin constants `tblite_data_spin::spin_constants_wb97mv` (103 × 10): v1.1's float32 `SPIN_W0`
  values in double precision (max difference 6.7e-9).
- EEQ-BC `multicharge_param_eeqbc2025_mp_eeqbc_{chi,eta,rad,kcnchi,kqchi,cap,cov_radii,avg_cn,
  rvdw_scale}_`: newer than public multicharge, all refitted; `avg_cn` same decimals as v1.1
  `com::cnav` but true doubles.
- dftd4 (`dftd4_model_reference_d4srev`, `dftd4_data_*`): alphaiw, ascale, hcount, refcn,
  refcovcn, refn, refsys, secaiw, sscale, tanh_params, r4_over_r2, the frequency grid and the
  pair weight factors (= v1.1 `com::k3_pair`) are identical to v1.1; **`gxtbq` and `gxtbh` differ
  in 272 of 826 entries** (max 2.4 e and 0.667); `covalent_rad_d3` changed by the CODATA
  conversion; chemical_hardness(H) back to 0.47259288 (unused).
- `ps_reference_occ`: §3.1.

---

## 4. Terms

Notation as in v1.1. f(CN) = √(CN + 1e-12) − 1e-6 with the g-xTB CN of §2 unless stated.
Distances in bohr.

### 4.1 EEQ-BC charges [C water, mesyl chloride: 7e-17]

As v1.1 §4.1 with three formula changes and new parameters (`add_charge_model` →
multicharge `new_eeqbc2025_model`, constants 0.3, 0.75, 25, 2, 0.75 passed):

- A_AA = (η_A + √(2/π)/w_A) C_AA + 1: **the local-charge hardness term kqeta·q^loc is gone**.
- w_A = rad_A (1 − **0.3** CN_A / CNavg_A^0.75).
- c_AB = √(cap_A cap_B) ½[1 + erf(−**0.75** (R − R^vdW_AB)/R^vdW_AB)] with
  R^vdW_AB = R^D3_AB(bohr) × 0.26458860545224622 × (s_A + s_B), s = `eeqbc_rvdw_scale`
  (= autoaa × arithmetic mean, an Å number).
- Unchanged: CN (§2), q^loc with Pauling EN/3.98 and the actinide overrides, x = −χ + kcnchi CN +
  kqchi q^loc, the Maxwell capacitance matrix.

Water O −0.31296 (v1.1 −0.27826). Used for q_eff (§4.2) and the start guess (§4.17); **no
longer used by the repulsion** (§4.4).

### 4.2 q-vSZP basis [C water, mesyl chloride: q_eff 1e-16 over 38 captured calls]

Tables bit-identical (§3.3); basis CN §2. Only change:

q_eff = k0 (q − k1 q²) + k2 f(CN_basis) + k3 q CN_basis (`get_qeff`; v1.1: √CN). The shift is
−k2·1e-6. H0 basis (`add_basis`): exponents × `ps_h0_qvszp_exp_scal`, k0, k2, k3 ×
`pa_h0_qvszp_k{0,2,3}_scal`, k1 unscaled, the same q_eff formula (v1.1 structure, §4.5).

### 4.3 Atomic increments [C 21 molecules, ≤ 9e-13 Eh]

E^incr = Σ_A `pa_increment`(Z_A). Refitted; **He = −0.15208134 Eh** (v1.1 and SI §1.5: zero
for H and He); H = 0.

### 4.4 Repulsion — now self-consistent [C 17 molecules incl. Cu, Ni, Zn, Fe, I, Gd, U: 1.5e-12 Eh]

E_rep = Σ_{A<B, R ≤ 25} z_A z_B · [k_s e^{−k_a α_AB (R + R0_AB)²} + e^{−α_AB (R + R0_AB)^{3/2}}]
· [1 + k1_AB/R + kpen_AB (R̄_AB/R)² + k3 (R̄_AB/R)³ + k4 (R̄_AB/R)⁴]

- z_A = `pa_rep_zeff`_A (1 − `pa_rep_q`_A q_A) with **q = the SCF Mulliken atomic charges of the
  current iteration** (v1.1: EEQ-BC charges and an extra quadratic term). The container is
  evaluated every SCF iteration; its potential V_A = ∂E/∂q_A enters the Fock matrix (exact, to
  5e-18) and its energy is part of the printed electronic energy.
- α_A = `pa_rep_alpha` (1 + `pa_rep_cn` f(CN_A)); **α_AB = α_A α_B/(α_A + α_B)** (v1.1 had a
  factor 2; the SI Eq. 55 has none).
- R0_AB = √(`pa_rep_roffset`_A `pa_rep_roffset`_B) (unchanged).
- k1_AB = ½(k1_A + k1_B), k1 = `pa_rep_k1` per element (v1.1: two globals by period).
- kpen_AB = ½(kpen_A + kpen_B), kpen = 0.012098131381864387 (Z ≤ 2), 0.008544252691968662
  (Z > 2); k3, k4, k_s = 0.0046511298, k_a = 0.73 global (§3.2); R̄ §3.2.

### 4.5 Extended Hückel Hamiltonian [C water, mesyl chloride, Ni(CO)4: H0 2.2e-16]

- **Levels**: se_l,A = `ps_h0_selfenergy` − `ps_h0_selfenergy_cn` · CN_A (raw g-xTB CN).
- **Shell factor** K(l,l') = ½(k_l + k_l'), k = `pg_h0_kshell`; **d–f pairs × 1.5** [D].
- **Radii** (`get_rad`): rad(Z) = `covalent_rad_d3`(Z)·0.75/aatoau·1.889725949 (Pyykkö radius in
  Å × 1.889725949, evaluated in this order), overrides H 0.7615595574470001 (0.403 Å), F
  0.9448629745 (0.5 Å), Ne 1.15273282889 (0.61 Å). v1.1 `com::rcov` for most elements; H
  differs (v1.1 0.7899).
- **Distance polynomial**: x = R/(rad_A + rad_B),
  π_A,l = 1 + 2 kp_A F_l x + 0.0804696812 kp_A F_l R x, kp = `pa_h0_shpoly2`,
  F = `pg_h0_shpoly2` = (1, 1.5, 2, 2.5). The linear term equals v1.1's
  (2kp R/(r_A + r_B) = kp R/((r_A + r_B)/2)); the R²/(rad_A + rad_B) term is new.
- **Off-diagonal** (A ≠ B): H_μν = ½(se_μ + se_ν) K π_μ π_ν S̃_μν + ΔH^aniso_μν, S̃ = H0-basis
  overlap scaled in the diatomic frame with k_σ, k_π, **k_δ** = harmonic means 2/(1/k_A + 1/k_B)
  of `ps_h0_diat_scale`(1..3) (0 if either is 0: H and He have no π scale); **δ per element**
  (v1.1: 5). The unscaled H0-basis overlap is the v1.1 construction ([C] s–s blocks, [D] p, d);
  f blocks unscaled per SI (untested, no f element run) [U].
- **Onsite**: H_μν = se_μ δ_μν + ΔH^aniso_μν (the anisotropy adds s–p mixing within an atom).

#### 4.5.1 New: anisotropy (atomic dipole shift) term [C water, mesyl chloride, Ni(CO)4: a_A 1.3e-18, H0 2.2e-16]

ΔH^aniso_μν = −½ [a_B · D^B_μν + a_A · D^A_μν],  μ ∈ A, ν ∈ B (A = B included),

a_A = Σ_{B≠A} s_AB · ½[1 + erf(−1.5 (R_AB − R̄_AB))] · (R_B − R_A),
s_AB = ½(`pa_h0_dip_scale`_A + `pa_h0_dip_scale`_B),

with D^A_μν = ⟨μ|r − R_A|ν⟩ the dipole integrals of the SCF (q-vSZP) basis about atom A
(tblite `dpint` and its `shift_operator` copy) and R̄ the shared pair radius (§3.2). Water:
a_O = (0, 0, −0.0021648807), O s–p_z onsite element 0.0014947595 Eh. Geometry-only (no density
dependence), so it is part of H; its gradient needs dipole-integral derivatives on both centres
(the program has `get_anisotropy_gradient`). Not in v1.1 or the SI.

### 4.6 Atomic correction potentials [C water, mesyl chloride, Ni(CO)4: 4.4e-16]

H^ACP = S_pvᵀ diag(w) S_pv, S_pv = overlap of normalized single-Gaussian projector functions
with the SCF AOs, **w = `ps_acp_level`(k, Z) (no CN factor)**, exponent `ps_acp_exp`(k, Z),
`pa_nacp`(Z) projectors with l = `pa_l_acp`(k, Z): (s, p, d, p) for H and Li–Ne, (s, p, d) for
He, Na, K, Rb, Cs, Po, At, Fr, (s, p, d, f) otherwise (v1.1: s, p, d, f everywhere with
w = c(1 + acpcn √CN)). Primitive screening and pair cutoff not decoded [U] (v1.1: 20,
R² ≤ 1600).

### 4.7 First order [C water, O2, [CuCl4]²⁻: 3e-16 Eh per atom]

E(1) = Σ_l μ_l [1 + `pa_tb1_ipea_cn` f(CN_A)] · {1 + 0.0125 [erf(q_A − 0.666666666) + erf(q_A + 0.666666666)]} q_l,
μ_l = `ps_tb1_ipea`. The offsite cross term of v1.1 §4.7 (−Σ Δρ⁰ γ2 q) is **booked in the
second-order container** in 2.0.1 with the same value (§4.8).

### 4.8 Second order [C water, O2, [CuCl4]²⁻: γ bit-identical, energy 1e-15 Eh]

γ_ij = 1/(R + e^{−0.294621155 R}/η_ij), η_ij = 2/(1/U'_i + 1/U'_j),
U'_l = `ps_tb2_shell_hubbard`_l · `pa_hubbard_parameter`_Z · [1 + `pa_tb2_hubbard_cn` f(CN_A)],
γ_ii = U'_i.

E(2) = ½ Σ_ij q_i γ_ij q_j + Σ_i q_i Σ_{j: A(j)≠A(i)} γ_ij r_j, r_l = `ps_reference_occ` −
`ps_tb1_zeffsh` (= −Δρ⁰ of v1.1). The program books the cross term to atoms differently in open
shells; only the total is defined.

### 4.9 Anisotropic electrostatics [C water, O2, [CuCl4]²⁻: 9e-16 Eh]

Pair formula of v1.1 §4.9 (q–d, q–Q, d–d, d–Q, Q–Q with ⅓, no onsite terms, Mulliken moments),
with:

- g_n = ½ k_n [1 − erf(−s_n (R − R̄_AB))]/Rⁿ, k = (0.3405910191, 0.1691310614, 0.074034339,
  −0.02), s = (0.5, 1, 1, 1) for n = 3, 5, 7, 9; R̄ §3.2.
- **Dipoles scaled**: d'_A = [1 + `pa_aes_dip_scale`_A f(CN_A)] d_A in every term with a dipole
  (quadrupoles unscaled).
- No onsite (AXC) kernel (`get_kernel_energy` never called). Potentials of all five parts are
  included and exact (§4.16).

### 4.10 Spin polarization [C O2, [CuCl4]²⁻: 4e-16 Eh]

Unchanged form: E_spin = ½ Σ_A Σ_{ll'∈A} m_l W_ll'(A) m_l', W = `pa_wll_scale`_A · W⁰ (W⁰ §3.3).

### 4.11 Third order [C water, O2, [CuCl4]²⁻: τ 4e-17, energy 4e-16 Eh]

E(3) = ⅓ Σ_i q_i² Γ_i Σ_j τ_ij q_j, **Γ_l = `pa_tb3_hubbard_derivs`_A · kshell_l**,
kshell = (1, 1.15, 1.3, 1.45) (v1.1: ⅓ Σ q_l q_l' q_A Γ_A τ). Kernel with η_ij of §4.8:
τ_ij = −0.2093327496 η_ij² within an atom (incl. i = j); between atoms, x = R/η_ij,
τ_ij = 2.3 x (1 − 1.3 x/2) e^{−1.3 x} (v1.1: 1.6, 1.15, onsite −½η²).

### 4.12 Fourth order [C]

E(4) = (1/24) Σ_l 0.036 · kshell_l · q_l⁴, kshell = (1, 1.15, 1.3, 1.45) (v1.1: Σ (0.021/4) q_l⁴).

### 4.13 Mulliken-approximated range-separated exchange [C water, O2, mesyl chloride, Ni(CO)4, [CuCl4]²⁻: E 9e-16 Eh, F 2e-16]

Kernel (shell pairs, expanded to AO blocks), same form as v1.1 §4.13:

γ^x_ll'(A,B) = (0.15 + 0.85 erf(0.2 R)) / (R + e^{−(k^x1 + k^x2 R̄_AB) R}/(η̃_ll' g_ll')),

k^x1 = −0.2960502355, k^x2 R̄ = 0.0349817377 R̄ (the program multiplies 0.018511538388678535 by
½(s_A + s_B)·aatoau·R^D3 in bohr; reproduce this order); η̃ = general average
2^{ξ−1}(U_i U_j)^{ξ/2}/(U_i + U_j)^{ξ−1} with **ξ = max(ξ_l, ξ_l')** (`ps_fock_avg_exp`),
U^x_l = `ps_fock_shell_hubbard`_l · `pa_hubbard_parameter`_Z; g_ll' = √(G_l G_l'),
G = (0.05, 2.5, 2.5, 2.5), same shell of the same atom 1.3826597204; R = 0 within an atom.

Energy (closed shell, total P): E_MFX = −⅛ Σ_μν γ^x_μν [(PS)_μν (SP)_μν + P_μν (SPS)_μν].
Open shell: E = Σ_σ 2 E_closed(P^σ) (the v1.1 per-spin convention). The program evaluates
E_x = ½ Tr(F_x P) per atom by row (exact, E_x is quadratic in P at fixed kernels).

#### 4.13.1 New: bond-order correction [C water: kernel 2e-18]

E_bo = ½ Σ_μν K_A(μ)B(ν) (PS)_μν (SP)_μν = Σ_{A<B} K_AB B_AB, B_AB the Mayer bond order
Σ_{μ∈A,ν∈B} (PS)_μν (PS)_νμ,
K_AB = c̄_AB · ½[1 + erf(−(R − r̄_AB)/R̄_AB)] (A ≠ B), c̄ = ½(`pa_fock_cscale`_A +
`pa_fock_cscale`_B) (all negative), r̄ = √(`pa_fock_crad`_A `pa_fock_crad`_B) (program form:
erf(−aatoau (R − r̄)/(aatoau R̄))). Open shell: per spin as MFX. Water −0.0248 Eh.

The program's `wbo` file and "Wiberg/Mayer (AO) data" printout (tblite's post-processing,
`get_mayer_bond_orders`) are the same Mayer bond orders of the converged density: B_AB as above
for a closed shell, 2 Σ_σ Σ (P^σ S)_μν (P^σ S)_νμ for two spin channels, no distance cutoff;
the file lists pairs above 0.1 (`print_wbofile`). The printout labels every atom with the symbol
of the atom whose index is its species index (`mol%sym(mol%id)`), so heavy atoms after the first
species appear as "H"; the atomic numbers and the file are right. axtb `bond_orders` [C: ≤ 1e-12
against the file, `tests/gxtb_parity_features.py`]. `--quadrupole` prints tblite's
`get_molecular_quadrupole_moment` about the coordinate origin: the atomic quadrupoles plus
3/2(q r r + r d + d r) − 1/2(q r² + 2 r·d) on the diagonal, but 3 (not 3/2) times the charge and
dipole parts off the diagonal, so the printed "total" is not a tensor (its eigenvalues change
when the molecule is rotated); axtb reproduces it in program conventions and gives the traceless
tensor otherwise (`axtb::properties::QuadrupoleForm`). The HOMO–LUMO gap in the summary is that
of the alpha channel (mctc-lib's eV, 27.211386245988).

### 4.14 Onsite exchange — Mulliken form, charge-dependent integrals [C, same molecules as §4.13]

Λ_ll'(A) = 0.15 Λ⁰_ll'(Z_A) [1 − ½(kq_l q_l + kq_l' q_l')], kq = `pg_fock_kq` = (1.1, 0.55,
0.275, 0.1375), q_l the total SCF Mulliken shell charges (n⁰ = `ps_reference_occ`), Λ⁰ §3.3
(Λ⁰_ss = 0).

E_ons = −(1/16) Σ_A Σ_{κ,λ∈A} Λ_κλ [(PS)²_κλ + P_κλ (SPS)_κλ + P_κκ (SPS)_λλ + (SP)_κκ (SP)_λλ]
      + ¼ Σ_{shells s, l ≥ 1} r_s Σ_{κ,λ∈s} [(PS)_κλ (SP)_κλ + P_κλ (SPS)_κλ],
r_s = Λ_ll/(2(2l + 1)) over the whole shell block (closed shell; open shell Σ_σ 2 E(P^σ) with
total charges in Λ). Replaces v1.1's dual-density OFX (water −0.0917 Eh vs −0.0046 Eh with the
v1.1 expression at the same density). The charge dependence gives a Mulliken shell potential
v_l = ∂E_x/∂q_l (charge channel only).

### 4.15 revD4 dispersion [C 13 molecules incl. GdCl3, UF6: C6 1.6e-15 rel., E2 1.5e-15 Eh, E3 6e-18 Eh]

C6 model unchanged from v1.1 §4.15 (revD4 CN, pair-specific Gaussian weights with
`weight_factors`, tanh charge scaling n(q)/n(qref), hydrogen-subtracted reference
polarizabilities, Casimir–Polder C6) with the new `gxtbq`/`gxtbh` tables (§3.3). Damping
(`new_damping(2, 2)`: screened rational for two- and three-body terms; differs from the public
`dftd5` branch):

- ρ_AB = √(3 r4r2_A r4r2_B), reff_AB = a2 · ½[1 + erf(−a3 (R − a4 ρ_AB))] (a1 = 0).
- Two-body (SCF Mulliken charges, every iteration, exact potential):
  E2 = −Σ_{A<B, R ≤ 50} C6_AB(q) · s6/(R + reff_AB)⁶ (s8 = 0: no C8 term).
- Three-body (once before the SCF, C6 at **q = 0**, printed "dispersion energy"):
  E3 = −Σ_{triples, R ≤ 25} c9 · s9 Π_pairs (R/(R + reff))³ · (3 cos α cos β cos γ + 1)/(R_AB R_AC R_BC)³,
  c9 = −√|C6_AB C6_AC C6_BC|, s9 = 1.826410691.

### 4.16 Fock matrices [C]

Every term's potential is the exact derivative of its energy: per-term potentials vs finite
differences of the Python energies (water, O2; exchange also mesyl chloride, Ni(CO)4,
[CuCl4]²⁻) agree to FD noise (≤ 1e-10); H0, anisotropy and ACP are linear in P. This includes
the AES dipole–quadrupole and quadrupole–quadrupole potentials and the onsite exchange
population part that the 1.1.0 program got wrong (v1.1 §4.16). Charge-dependent terms
(E(1)–E(4), AES charge part, repulsion, revD4 two-body, Λ(q)) enter as Mulliken
atom/shell potentials −½(V_μ + V_ν)S_μν; exchange as F_x = ∂E_x/∂P at fixed kernels plus its
shell potential; spin in the magnetization channel. Consequently the program's SCF solution is
stationary and its analytic gradient needs no response term.

### 4.17 SCF [C: mixer reproduced from captured vectors to 1e-15 (water, O2, [CuCl4]²⁻, caffeine), 1.6e-9 relative for AuCl (30 iterations)]

- **Start**: `eeqbc_guess`: EEQ-BC atomic charges (§4.1), shell charges by `shell_partition`
  q_l = n⁰_l/n⁰_A · q_A [D, public tblite `guess.f90`, call sequence seen in gdb]; no density
  matrix and no multipoles in iteration 1 (zero-initialised).
- **Iteration k**: Fock matrix F_k from the current input (charges, multipoles, density of
  P_{k−1}); mixing → F̃_k; diagonalise F̃_k (generalized, S); occupations; P_k; Mulliken shell
  charges, atomic dipoles and quadrupoles of P_k; energy E_k = E[P_k]. **No extra unmixed final
  iteration** (v1.1 had one).
- **Mixing** on the full Fock matrix (nspin·nao² elements, both spin channels in one vector, so one
  set of DIIS coefficients for α and β):
  k = 1: F̃_1 = F_1; k = 2, 3: F̃_k = F̃_{k−1} + 0.2 (F_k − F̃_{k−1})
  (`create_simple_input`, damping 0.2); k ≥ 4: Pulay DIIS (`create_diis_input`, memory 7) with
  residuals r_j = F_j − F̃_{j−1}, history j = 3..k (last 7), c minimising |Σ c_j r_j|² with Σ c_j
  = 1 (bordered B matrix, LU `getrf/getrs`), dot products over the full vector, F̃_k = Σ c_j F_j.
- **Switch to DIIS** [C, disassembly of `tblite_scf_iterator_mp_update_` and gdb]: the
  iterator holds the two mixers as stages; after iteration k it moves to the DIIS stage when
  k + 1 ≥ its start (4) **or** |E_k − E_{k−1}| < its energy threshold **1e-12 Eh** (E_0 = 0;
  both values fixed, the same at every `--acc` and `--etemp`). Normally the first condition
  decides (DIIS from iteration 4). When the energy stalls earlier, DIIS starts one iteration
  early: for H2 the density is fixed by symmetry from iteration 1, E_2 = E_1, iteration 3 is
  already DIIS (one stored vector: F̃_3 = F_3), the residual of iteration 4 is 3e-17 and the
  program stops after 4 iterations (with damping in iteration 3, 5). `v2/scf.rs` follows
  (`DIIS_SWITCH_ENERGY`; test `scf_paths::h2_v2_early_diis`).
- **"density error"** printed = RMS of r_k over the full vector, √(Σ r_k²/(nspin·nao²)) (0 at
  k = 1).
- **Convergence** after iteration k: |E_k − E_{k−1}| < 1e-6·acc Eh and (k > 1) error < 2e-5·acc
  (`check_convergence`), **with sticky flags** [C]: the routine only ever sets the
  energy and the residual flag (disassembly: it stores −1 when a criterion holds and otherwise
  keeps the caller's value), so each criterion counts once it has held in any iteration; the
  two need not hold in the same iteration (EuCl3 converges in iteration 13 with residual
  2.5e-5 because iteration 12 had 1.5e-5).
- **Occupations at 0 K** (`get_aufbau_filling`) [D, consistent with the printed electron count]:
  per channel n_σ = (N ± U)/2 (real, §1.2), occ(1..⌊n_σ⌋) = 1, occ(⌊n_σ⌋ + 1) = n_σ − ⌊n_σ⌋;
  closed shell: one channel with doubled occupations. `--etemp` > 0: Fermi filling (§4.17.1).

#### 4.17.1 Finite electronic temperature (`--etemp T`) [C: 236 runs, §10]

The 2.0.1 program's default for g-xTB is 0 K (xtb `calculator.F90`: `if (.not.
input%custom_etemp) calc%etemp = 0`); `--etemp T` sets kT = T · 3.166808578545117e-6 Eh (xtb's
`kt` literal, the same as tblite's and xtb's GFN2 kB). Decoded from the binary
(`tblite_wavefunction_fermi_mp_get_fermi_filling_` and `get_fermi_filling__`, the `.a` variant
disassembled; constants read from `.rodata`):

- **Per electron spin**, on the eigenvalues of its channel (closed shell: both electron spins
  fill the one channel's levels, the occupations are summed for the density): Aufbau filling
  first (`get_aufbau_filling`), which also sets the HOMO index h = ⌊n_σ⌋, +1 if mod(n_σ, 1) > ½;
  then, **if h > 0 and kT > 0**, Newton steps for the Fermi level with the **integer target h**,
  not n_σ: the program's fractional electron count (Σ n⁰ − Q, §1.2) is rounded per channel at
  every finite temperature (water: 4 + 4 electrons instead of 2 × 3.99999999997510, so any
  `--etemp` lowers its energy by 2.1e-11 Eh even where no orbital is fractionally occupied). This
  is the older tblite routine (current tblite targets n_σ with a tighter threshold). Channels
  with h ≤ 0 keep the Aufbau occupations.
- **Newton steps** (`get_fermi_filling__`): μ₀ = ½ (ε_h + ε_min(h+1, n)); per step, for every
  level x = (ε_i − μ)/kT, f_i = 1/(eˣ + 1) and f'_i = eˣ / (kT (eˣ + 1)²) when x < 50, else
  f_i = f'_i = 0 (no lower cutoff); Δ = h − Σ f_i; μ += Δ / Σ f'_i if Σ f'_i > √tiny
  (1.49e-154), else unchanged; stop when |Δ| ≤ √ε = 2⁻²⁶ ≈ 1.49e-8 (checked after the update, so
  the occupations are those of the previous μ) or after 200 steps. The electron count is
  therefore exact only to 1.5e-8 per channel; in practice the last step leaves ~1e-9. The
  AVX-512 build sums the first 4⌊n/4⌋ levels in four lanes (reduced (l0 + l2) + (l1 + l3)) with
  f' = eˣ / ((eˣ + 1)((eˣ + 1) kT)), the remainder in order with fused multiply-adds
  (kT eˣ + kT, then eˣ·K + K); `exp` is Intel's (SVML for the vector part). The returned Fermi
  level is ½ μ (not used further).
- **Entropy** (`tblite_scf_iterator_mp_next_density_`, `get_electronic_entropy`): −TS =
  kT Σ_σ Σ_i ln(f^f (1 − f)^(1 − f)) per electron spin (≤ 0), divided evenly over the atoms'
  electronic energies (`energies(:) = ts / nat` before the H0 and container energies are added):
  it is part of every printed SCF cycle energy, the convergence test |ΔE|, the printed
  "electronic energy" and the total. There is no separate printout of −TS.
- **Gradient**: the same formula as at 0 K with the smeared occupations (W = Σ f ε c cᵀ of the
  last diagonalization); no occupation response (the free energy is stationary in f for the
  exact μ; the program's μ is exact to its 1.5e-8 count threshold).
- **Sensitivity**: with smeared occupations the density depends continuously on the orbital
  energies, so last-bit differences of the eigensolver (MKL vs faer) and of `exp` (Intel vs
  glibc) reach the SCF. On SCFs that converge slowly under the program's DIIS this decides the
  last digits: perturbing kT by one ulp in our SCF moves Fe4 (1000–5000 K), TmH3 (3000, 5000 K)
  and ErF3 (5000 K) at the default thresholds by as much as their difference to the program
  (1e-9 to 2e-4 Eh, iterations ±20), and the tight-threshold iteration counts of Fe4 (1000 K),
  CeCl3 (5000 K), ErF3 (3000, 5000 K) and stretched N2 (5000 K) by as much (energies equal to
  1e-12 Eh). Some of these end at the 250-iteration limit for one of kT, kT ± 1 ulp: TmH3
  5000 K (default thresholds: 157 iterations or none within 250, depending on the round-off;
  the program: 158), Fe4 1000 K (tight: 75-250 iterations), EuCl3 5000 K (tight: 102-250
  iterations; the program fails).
- No level shift; integral cutoff min(√(2·intcut/α_min), 40) bohr (tblite `get_cutoff`, printed
  26.9 bohr for water at acc = 1) [C: C50H102 crosses it]; primitive screening
  a_i a_j R²/(a_i + a_j) > 25 for the SCF-basis, H0-basis and ACP integrals [C: the
  reproduction holds to 1e-12 with 25 everywhere].

#### 4.17.2 Restart (`xtbrestart`) [C: caffeine, water, O2 triplet, [Cu(NH3)4]²⁺ doublet; same geometry and 0.02/0.1/0.3 bohr rms displacements; energies ≤ 4.3e-11 Eh, identical SCF iterations]

xtb's `dumpRestart` (`src/tblite/restart.F90`, public) writes, as format version 3: the atom and
shell counts, electrons and unpaired electrons, then the wavefunction's shell charges `qsh`
(charge and magnetization channels), atomic dipoles `dpat` and quadrupoles `qpat`, and the
density matrices (per spin). `loadRestart` reads them back when the atom and shell counts match;
the singlepoint recomputes the atomic charges from `qsh` and starts the SCF from that state
instead of the EEQ-BC guess: iteration 1 builds F_1 from the restart's charges, multipoles,
magnetizations and densities (the q-vSZP basis is the new geometry's; the density matrices are
taken over as they are). The mixing is unchanged (damping in iterations 2 and 3, then DIIS), and
with E_0 = 0 the earliest convergence is iteration 2 (2 cycles at the same geometry). The same
state is carried between the steps of an optimization inside one program run.
`v2::scf::Setup::scf_from` (from a `GuessState`; the calculator's `Guess::Previous`,
`calculator.rs`) reproduces it: a restarted caffeine takes 2 cycles and lands 7.0e-8 Eh below the
fresh run at the default accuracy, as the program.

### 4.18 Gradients, conventions, Hessian [C]

- The program's analytic gradient (`--grad`) equals Richardson-extrapolated central differences
  (h = 0.002, 0.001 Å) of its own tight energies (`--acc 1e-4`) to 1.3e-10 Eh/bohr (water),
  9.5e-11 (O2 triplet), 3e-8 (mesyl chloride, finite-difference noise); plain central
  differences at h = 0.0005 Å to ≤ 4e-7 for Ni(CO)4 and [CuCl4]²⁻ (`reference_v2/
  gradient_check.json`). With §4.16 this makes the program's gradient a direct reference for
  ours.
- **Conventions.** Because the 2.0.1 Fock matrix is exact, the program and the corrected default
  share one Fock matrix and SCF. The remaining program-only artefacts are: the non-integral
  electron count (§1.2; ~1e-11 Eh), the mctc Å→bohr factor (~1e-12 relative), float literals
  such as 0.666666666 and k^x2 stored as a product, and hard cutoffs (CN and repulsion 25 bohr,
  revD4 50/25 bohr, integral cutoff) that make the energy discontinuous for large systems (§9).
- **The default conventions keep tblite's primitive screening.** Every integral drops primitive pairs with a_i a_j R²/(a_i + a_j) > 25, in both
  conventions. Over a primitive pair crossing that threshold, the energy steps by ~1e-11 Eh. That
  puts ~1e-7 Eh/bohr² of noise into any *finite-difference* Hessian of heavy-element systems
  (UF6, Fe, Ni compounds; §11.4). We keep it in the default because:
  1. it is part of how the method was parametrized and is evaluated by the program, so energies
     stay comparable to published g-xTB numbers;
  2. its effect on energies and gradients (~1e-11 Eh) is far below any chemical relevance;
  3. the default Hessian is analytic, so it does not see the steps.

  Only the response and numerical Hessians are affected, and only at the 1e-7 level. Revisit if
  finite-difference Hessians of heavy-element systems ever become the default, or if a tighter
  screening proves cheap.
- Hessian: the program's `--hess` is numerical (§11.5). Ours: analytic (the default),
  response or numerical (§11).
- **The program's gradient formula** [C]: its Mulliken charge, spin and multipole
  potential terms (the potentials contracted with the overlap and multipole-integral
  derivatives) use the potential of the last Fock build, i.e. the charges and moments of
  P_{k−1}, while everything else uses P_k. At self-consistency this is the same gradient; at
  the default thresholds it differs from the P_k-only formula by up to ~1.6e-5 Eh/bohr (the
  size of the SCF residual effect; both are equally far from the converged gradient). axtb's
  program mode uses the program's formula and reproduces its gradients to ≤ 5e-11 Eh/bohr at
  any threshold; the default conventions use P_k throughout.
- **The program's gradient is wrong for some bonds along a coordinate axis** [C,
  `tests/gxtb_elements.py`]: for [Gd(H2O)6]³⁺ and [Lu(H2O)6]³⁺ with the M–O bonds on the x, y
  and z axes, the program's analytic gradient (default and `--acc 0.0001`) differs from axtb's
  by 8.0e-5 and 1.4e-4 Eh/bohr, only in the y components of the metal and the two axial (±z)
  oxygens (the differences sum to zero), while the energies agree to 3e-11 Eh and the SCF
  iterations are the same. Five-point differences of the program's own energies follow axtb
  (Gd: 5.967e-4 at O(+z) against the program's 5.170e-4 and axtb's 5.967e-4), and after a
  rotation of the input both gradients agree to 1e-15 while axtb's does not move (1e-10). So
  the program evaluates some derivative incorrectly for a bond exactly along z (likely a
  frame or anisotropy term with an axis special case; not decoded). Not reproduced (axtb's
  gradient stays the derivative of the energy); regression test
  `scf_paths::gd_aqua_v2_program_gradient_axis` with the program's displaced energies in
  `tests/reference_elements/gd_aqua_v2_gradient.json`. No other molecule of the element survey
  shows it (UF6, WF6 and the other aqua ions agree to ≤ 1.8e-9).

---

## 5. Reference data set (`crates/gxtb/tests/reference_v2/`)

Generated by `crates/gxtb/scripts/make_reference_v2.py <xtb 2.0.1> <outdir>` (geometries from
`v2_reference_cases.py`, which reads `tests/reference*`, and `v2_element_cases.py`):

| path | content |
|---|---|
| `reference/<mol>/` (19: the 14 + 5 v1.1 reference molecules), `elements/<mol>/` (36: LiF, BF3, NaCl, AlCl3, HBr, H2Se, SnH4, XeF2, KCl, CaCl2, TiCl4, Cr(CO)6, ZnCl2, AgCl, [PtCl4]²⁻, AuCl, HgCl2, PbH4, BiH3, WF6, LaCl3, CeCl3 (d), EuCl3 (sept.), GdCl3 (oct.), LuCl3, ThCl4, UF6, NpF6 (d); added while implementing: LaH3, GdH3 (oct.), TmH3 (trip.), ErF3 (quart.), TbCl3 (sept.), YbCl3 (d), AmCl3 (sept.), PuO2 (quint.); rough geometries, not minima; added in stage 3, generated by
`tests/gxtb_elements.py` (`REFERENCE_IDS_V2`, element coverage): H2, BeH2, MgH2, ScCl3,
Co(CO)4⁻, KrF2, RbH, YCl3, Mo(CO)6, RuO4, [PdCl4]²⁻, HfCl4, OsO4, [IrCl6]³⁻, RnF2, FrH, RaH2,
AcCl3, PaCl5, PrCl3 (trip.), DyCl3 (sext.), CmCl3 (oct.), NoCl2, LrCl3, [Co(H2O)6]²⁺ (quart.),
[Mn(H2O)6]²⁺ (sext.), CH3 radical) and `large/<mol>/` (C50H102, 148 bohr long, and (H2O)100 from `tests/structures/large`, which cross the 25/50-bohr cutoffs) | `mol.xyz`, `input.json`, `xtb.out` (scrubbed), `energy`, `gradient` (default `--grad`), `energy_tight`, `gradient_tight` (`--acc 0.0001`; absent for TmH3, whose tight SCF does not converge in the program's 250 iterations), `summary.json` (parsed printout: energies, electron count, cutoff, thresholds, every SCF cycle, gap, charges, tight energy/iterations, and `terms`: per-term energies of the last iteration read with gdb) |
| `*/<mol>/density.json` (12 molecules: water, methane, hydroxide, O2, mesyl chloride, Ni(CO)4, [CuCl4]²⁻, ferrocene, AuCl, CeCl3, GdCl3, UF6) | the program's density of the last SCF iteration (packed, per spin channel), its Mulliken shell charges and per-term energies (`scripts/capture_density_v2.py`): every term is tested at the program's own density |
| `index.json` | name, charge, uhf, energy, iterations, tight energy |
| `params/*.json` | all 2.0.1 parameter arrays at full precision, constants, diffs vs v1.1, Python check summaries |
| `v1.1_vs_v2.0.1.json` | total energies and iterations of both programs (§6) |
| `gradient_check.json` | §4.18 |
| `../reference_elements/gd_aqua_v2_gradient.json` | [Gd(H2O)6]³⁺: the program's tight gradient and its energies at ±h, ±2h along the components its gradient gets wrong (§4.18) |

`ONLY=name1,name2 make_reference_v2.py …` regenerates single molecules. The capture points
used in the decoding are listed in Appendix A; `params/*.json` also hold the check summaries
(`verification`, `formulas`).

---

## 6. 2.0.1 versus 1.1.0, numbers

Same geometries, `OMP_NUM_THREADS=1`, no restart files (laptop, load < 3).

| molecule | E 1.1.0 (Eh) | E 2.0.1 (Eh) | 2.0.1 − 1.1.0 | SCF iterations 1.1.0 / 2.0.1 |
|---|---|---|---|---|
| acetate | -228.548515555 | -228.551501119 | -2.99e-03 | 10 / 9 |
| ammonium | -56.883840833 | -56.885718912 | -1.88e-03 | 9 / 8 |
| benzene | -232.229427588 | -232.227094624 | +2.33e-03 | 9 / 7 |
| dioxygen triplet | -150.341021159 | -150.340838951 | +1.82e-04 | 9 / 7 |
| ethanol | -155.031990344 | -155.031457206 | +5.33e-04 | 10 / 8 |
| glycine | -284.455651437 | -284.456227574 | -5.76e-04 | 12 / 9 |
| hydroxide | -75.803071016 | -75.798360266 | +4.71e-03 | 9 / 9 |
| iodomethane | -337.585583385 | -337.589587243 | -4.00e-03 | 9 / 8 |
| mesyl chloride | -1048.747710659 | -1048.743945835 | +3.76e-03 | 10 / 9 |
| methane | -40.494236628 | -40.494435440 | -1.99e-04 | 8 / 8 |
| nickel tetracarbonyl | -1961.713559308 | -1961.728192381 | -1.46e-02 | 11 / 9 |
| silane phosphine | -633.797490699 | -633.792065286 | +5.43e-03 | 11 / 8 |
| tetrachlorocuprate | -3481.408103773 | -3481.396112835 | +1.20e-02 | 10 / 9 |
| water | -76.437375752 | -76.437427653 | -5.19e-05 | 9 / 9 |
| caffeine | -680.410770589 | -680.419231528 | -8.46e-03 | 12 / 9 |
| ferrocene | -1650.717786822 | -1650.731826007 | -1.40e-02 | 12 / 9 |
| cf3br-benzene | -3143.957853651 | -3143.956680036 | +1.17e-03 | 11 / 9 |
| zncl4 (2−) | -3620.378994842 | -3620.362891681 | +1.61e-02 | 8 / 9 |
| cu-nh3-4 (2+, doublet) | -1866.112415024 | -1866.120130573 | -7.72e-03 | 11 / 9 |

Further elements (`elements/`): differences −2.2e-2 to +2.2e-2 Eh where 1.1.0 runs (LiF −1.6e-2,
XeF2 −1.8e-2, Cr(CO)6 −1.7e-2, [PtCl4]²⁻ +2.2e-2, CeCl3 −2.0e-2, UF6 −2.2e-2, others ≤ 1.1e-2);
1.1.0 fails for EuCl3, GdCl3, ThCl4, NpF6 (no parameters), 2.0.1 converges in 9–15 iterations;
AuCl takes 30 iterations in 2.0.1 (10 in 1.1.0), UF6 50 in 1.1.0 (9 in 2.0.1). Full table in
`reference_v2/v1.1_vs_v2.0.1.json`.

Energy components at convergence (Eh; 2.0.1 / 1.1.0; v1.1 values from `tests/reference/*/kernels.json`):

| molecule | E(1) | E(2)+E(3)+E(4) | AES | exchange (1.1: MFX+OFX) | revD4 two-body | repulsion | increments | Tr(HP) |
|---|---|---|---|---|---|---|---|---|
| water | 0.0879 / 0.0626 | 0.0500 / 0.0625 | 0.0042 / 0.0039 | -1.244 / -1.534 | -0.00004 / -0.00078 | 0.3227 / 0.3053 | -69.427 / -69.018 | -6.231 / -6.318 |
| benzene | -0.0688 / -0.0314 | 0.0192 / 0.0300 | 0.0036 / -0.0110 | -3.122 / -3.033 | -0.01093 / -0.01592 | 2.3105 / 2.4606 | -209.205 / -209.627 | -22.155 / -22.001 |
| glycine | 0.0897 / 0.1172 | 0.0942 / 0.1115 | 0.0110 / 0.0064 | -4.298 / -4.790 | -0.00790 / -0.01260 | 1.5218 / 1.5022 | -258.879 / -258.212 | -22.988 / -23.178 |
| dioxygen triplet | -0.0355 / 0.1063 | 0.0016 / 0.0050 | 0.0015 / 0.0038 | -2.207 / -2.908 | -0.00006 / -0.00075 | 0.2262 / 0.1633 | -138.855 / -138.036 | -9.426 / -9.639 |
| mesyl chloride | 0.0919 / 0.2063 | 0.1088 / 0.0996 | 0.0108 / 0.0133 | -4.490 / -5.237 | -0.00775 / -0.00867 | 1.2815 / 1.2404 | -1022.590 / -1021.749 | -23.150 / -23.313 |
| nickel tetracarbonyl | -0.4844 / -0.1748 | 0.0529 / 0.0462 | -0.0141 / 0.0040 | -7.690 / -8.837 | -0.00745 / -0.00772 | 1.9924 / 1.9133 | -1921.203 / -1920.023 | -34.375 / -34.635 |
| tetrachlorocuprate | 0.5891 / 0.6375 | 0.4617 / 0.4162 | -0.0006 / -0.0031 | -6.089 / -5.851 | -0.00633 / -0.00350 | 0.1537 / 0.1204 | -3455.374 / -3455.135 | -21.131 / -21.588 |

(7 of the 14 molecules with v1.1 term data shown; per-term 2.0.1 energies of all 47 molecules are in
`reference_v2/*/*/summary.json`. The 2.0.1 E(1) excludes the offsite cross term, which 2.0.1 books in E(2).) Every component moves by 1e-3–1 Eh: the differences are a refit plus form
changes in every term, not a local change, so **no 2.0.1 term can be reused from v1.1 without
new parameters**; most formulas and all integral machinery can.

---

## 7. SI cross-check of the v1.1 specification (side task)

The SI describes the preprint (v1.x) method; it gives functional forms but almost no parameter
values. Items of `docs/gxtb-v1.1-spec.md` that deferred to the SI or carry [P]/[D]/[U], and the
v1.1 reconstructions the SI bears on:

**Confirmed by the SI**
- Reference occupations: SI Table 1 (H–Ar) equals v1.1's `shellocc_ref_` truncated to three
  decimals for all 18 elements (2.0.1 unchanged for these).
- q_eff structure and the H0-basis scaling of exponents, k0, k2, k3 (Eq. 28, p. 22); no
  polarization shells for H, He, B–Ne.
- H0: averaged Wolfsberg factor and levels (Eq. 64), levels linear in the raw CN (Eq. 65),
  linear distance polynomial with averaged radii (Eq. 67), f blocks not scaled in the diatomic
  frame (p. 25); g-xTB CN definition (Eq. 47).
- Increments (Eqs. 49–50), repulsion offset as geometric mean, √CN in α, k^pen,1 split H/He vs
  rest, exponent 1.5, EEQ-BC charges in the v1.1 repulsion (Eqs. 53–56).
- First-order f(q) (Eq. 83) and the offsite cross term (Eqs. 85–86); γ2 kernel (Eq. 101);
  atomic-charge third order (Eq. 129); τ functional form (Eq. 133); AES energy incl. ⅓ on Q–Q
  (Eq. 116) and damping form (Eq. 117); spin form, W⁰ order and ωB97M-V origin (Eqs. 120–121,
  §2.1); Z_l (Eq. 85).
- **AES Fock without the d–Q and Q–Q potentials** (§1.11.2, p. 62: "under development"): the
  1.1.0 program's omission (v1.1 §4.16) is the documented v1.x design, not a bug.
- MFX averaging function and ξ rule (Eq. 150), k^x1, k^x2 with D3 radii (Eq. 149), MFX energy
  and exact Fock (Eqs. 151, 153), Λ = αΛ⁰ with nine integral types (Eq. 155, §2.2).
- revD4: BJ with a1 only (Eq. 170), zero-damped ATM with exponent 16 (Eq. 171), C9 = √(C6C6C6)
  (Eq. 163), D4S pair factors (Eq. 165), Pyykkö CN radii; the tanh parameters satisfy b = −a for
  all 103 elements and d = atanh((1 − a)/b) to 3.6e-9 (Eqs. 167–169), explaining v1.1 §3.4.
- SCF: DIIS on the full Fock matrix with optional level shift (§1.1.4, p. 18; details deferred
  to a future paper). Fock = ∂E/∂P (Eqs. 21–22): the method's intent is the exact derivative,
  which supports `conventions = "axtb"` as the default.

**Contradicted by the SI (the 1.1.0 program, reproduced to 1e-12, differs)**
- Repulsion α_AB: SI Eq. 55 has no factor 2 (program has it; 2.0.1 follows the SI).
- revD4 ATM charge scaling: SI p. 84 says unity; both programs use n(0)/n(qref) (ζ = 1 would raise
  water's E_ATM by 18 %).
- ACP: SI squares the coefficient and uses c0(1 + k CN/CN_avg) with one global k (Eqs. 78–80) and
  projector sets s, p, d (+f for H, B–F, TMs, LNs, ANs); the program uses a signed linear weight
  with element-wise √CN and all four projectors (2.0.1 matches neither).
- δ scaling of the diatomic frame: SI Eq. 31 is element-wise; the 1.1.0 program uses 5 (2.0.1
  follows the SI).
- CN dependence of μ⁽¹⁾ and U⁽²⁾: SI Eqs. 84, 102 are linear in CN; both programs use √CN.
- Third-order τ argument: SI x = Ū R with the arithmetic mean of U; both programs use R/η.
  The onsite τ (−½η²) is not covered by the SI.
- Fourth order: SI Eq. 140b is atomic (1/24 Γ q_A⁴); 1.1.0 is shell-resolved with 0.021/4
  (2.0.1: 1/24, still shell-resolved).
- MFX kernel: SI Eq. 149 has no off-diagonal factor g (no G_l, no 1.39).
- OFX: SI Eq. 155 has no overall minus sign and no spin sum; its Fock (Eq. 159) gives half the
  exact population derivative, the 1.1.0 program twice (off the diagonal) — neither exact.
- Repulsion charge sign convention (Eq. 53 +k^{q,2}q² vs stored −rep(6)): convention only.
- Elements: the SI claims H–Lr; the public v1.1 parameter file has 79 elements (2.0.1: 103).

**Not addressed by the SI** (program internals, still open in v1.1): routine map (§1.4), the
printed D3-type CN, `com::valel/ipat/eaat` [U], the [CuCl4]²⁻ d–d AES Fock deviation [U],
degenerate occupations (`occu_` 0.99) [U], the program's Hessian projection [U], all §3.5
literal values (the SI names the parameters but gives no numbers).

No v1.1 item needs changing: every contradiction is a place where the SI describes a different
variant than the 1.1.0 program, which v1.1 reproduces to ~1e-12. **The SI is not needed for any
remaining 2.0.1 term** (it does not describe 2.0.1's additions); all 2.0.1 terms were decoded
from the program.

---

## 8. Open questions and how they were resolved

1. H0-basis overlap p/d/f blocks and the d–f ×1.5 kshell factor: **resolved** [C]. Tr(HP) at
   the program's density (electronic energy minus all other terms) agrees to print precision
   (≤ 6e-13 Eh) for CeCl3, GdCl3 and UF6 (f shells) as for the d-block molecules; f blocks stay
   unscaled in the diatomic frame (tblite `max_diat_l = 2`).
2. Primitive screening and pair cutoffs: **resolved** [C] — 25 for every integral (SCF basis,
   H0 basis, ACP projectors) and tblite's `get_cutoff` pair cutoff; C50H102 crosses it.
3. revD4 (50/25 bohr) and CN/repulsion (25 bohr) cutoffs: **resolved** [C] by C50H102 (148 bohr)
   and (H2O)100: program-mode totals 2e-13 / 2e-11 Eh, terms 4e-13 / 2e-11 Eh, same iterations.
4. Start guess and iteration 1: **resolved** [C] — the printed energies of every SCF cycle are
   reproduced to their print precision (9–11 decimals) from iteration 1 on.
5. Iteration counts of hard SCFs: **resolved** — an unblocked partially pivoted LU of the bordered
   DIIS system reproduces every count (AuCl 30, TbCl3 31, CeCl3 tight 45); the remaining
   undocumented detail was the sticky convergence flags (§4.17).
6. Program gradients: decoded where it matters (§4.18: the potential terms use the last Fock
   build's charges); reproduced to ≤ 5e-11 Eh/bohr.
7. `--etemp` (Fermi smearing): **resolved** [C] (§4.17.1, §10). Restart files: **resolved** [C]
   (§4.17.2).
8. The program's analytic gradient for bonds exactly along a coordinate axis ([Gd(H2O)6]³⁺,
   [Lu(H2O)6]³⁺; §4.18): which term the program gets wrong is open (a program bug; axtb keeps
   the derivative of the energy).

---

## 9. Implementation design (method `"gxtb-v2.0.1"`; the result is in §10)

**Structure.** A second parameter/model set in `crates/gxtb` rather than a new crate: a
`Version` (or `Model`) enum carried by `Setup`, with v1.1 and v2.0.1 parameter tables
(`data_v2.rs` generated by an `extract_binary_tables_v2.py` from the 2.0.1 binary, as for v1.1)
and per-term branches where forms differ. Calculator, CLI, Python and qccodec gain the method
name `gxtb-v2.0.1` (aliases `gxtb-2.0.1`, `g-xTB 2.0`); conventions as for v1.1:

- `conventions = "gxtb"` (program mode): reproduce the program (fractional electron count from
  the stored occupations, mctc Å→bohr, literals as stored, program cutoffs, tblite SCF exactly);
  target ~1e-12 Eh and identical iteration counts on `reference_v2`.
- `conventions = "axtb"` (default, corrected): the same Fock matrix (it is already exact) and
  SCF; integral electron count (N = Σ Z_l − Q), axtb's unit convention, and smooth or
  removed hard cutoffs where they cause discontinuities (axtb switches them off smoothly over
  1 bohr beyond the program's radii, §10; the effect is only for pairs beyond 25 bohr).
  Expected differences to the program ~1e-11 Eh for small molecules.
- No Z-vector response is needed in either mode (exact Fock); v1.1's `response.rs` stays v1.1-only.
- Analytic gradients for every term (Hessians: §11). Validation: finite differences of our tight energies and the
  program's analytic `gradient`/`gradient_tight` (§4.18: the program's gradient is exact to
  ~1e-10, so ours should agree to ~1e-9 with tight SCF).

---

## 10. Implementation status (`crates/gxtb/src/v2`)

Method `"gxtb-v2.0.1"` (aliases `gxtb-2.0.1`, `gxtbv201`, `g-xTB 2.0`) in Rust
(`axtb_gxtb::Calculator::new(Version::V2_0_1, …)`, the default version, or `v2::scf::Setup` +
`scf` + `gradient`), the axtb CLI (`--method`), Python (`method=`) and qccodec
(`Model(method="gxtb-v2.0.1")`): energies, analytic gradients and Hessians
(`Calculator::hessian`: analytic, the default, response or numerical; §11), with `accuracy`,
`etemp` (Fermi smearing, default 0 K as in the program; §4.17.1, §11.6), `max_iter`, `threads`,
`conventions` and the mixer settings `damping`, `diis_size`, `energy_tol` and `residual_tol` (the
Fock residual; `ScfOptions` refuses v1.1's `density_tol` and `transition_iteration`); the
GFN2-only settings (solvation, dispersion off, the GFN2 mixer) are rejected. v1.1 (`"gxtb-v1.1"`): numerical Hessians only; its `etemp` reproduces the
1.1.0 program's `-tel` (`docs/gxtb-v1.1-spec.md` §4.17).

| module | content |
|---|---|
| `data_v2.rs` | every 2.0.1 parameter array, generated by `scripts/make_data_v2.py` from `params/*.json` (checks the tables shared with v1.1 bit for bit) |
| `v2/mod.rs` | `Model`: the conventions' constants (Ångström factor, covalent radii, cutoff factors), `read_xyz` |
| `v2/cn.rs`, `v2/eeqbc.rs`, `v2/basis.rs` | g-xTB and basis CNs, EEQ-BC (charges and adjoint gradient), q_eff and the SCF/H0 bases, integrals with tblite's screening |
| `v2/h0.rs` | self-energies, H0 with the new polynomial and per-element δ scaling, anisotropy, ACP |
| `v2/coulomb.rs`, `v2/repulsion.rs`, `v2/disp.rs`, `v2/exchange.rs` | E(1)–E(4), AES with scaled dipoles, spin; self-consistent repulsion; revD4 with screened damping; MFX + bond-order correction + charge-dependent onsite exchange (energy, Fock, shell potential, dE/dS) |
| `v2/scf.rs` | setup, energies and Fock matrices of a density, tblite's SCF (eeqbc guess, damping 0.2, DIIS 7 on the full Fock matrix, 0 K Aufbau with fractional remainder or Fermi smearing, sticky convergence flags) |
| `v2/fermi.rs` | occupations at a finite electronic temperature: the program's filling (§4.17.1) and the exact Fermi level of the default conventions; entropy term −TS |
| `v2/gradient.rs` | analytic gradient: explicit derivatives of every term plus chain rules; no response term |

**Conventions.** `gxtb` (program mode): mctc-lib's Ångström factor (1.8897261246204404) for the
XYZ input and every Ångström table, the fractional electron count Σ n⁰ − Q, the hard cutoffs
(CNs 25, repulsion 25, revD4 CN 30 / two-body 50 / three-body 25 bohr, the integral pair
cutoff), the program's gradient formula (§4.18). `axtb` (default): the same energy
expression, Fock matrix and SCF with CODATA 2022 units (Å factor and the Ångström tables:
D3 pair radii, Pyykkö radii; the unit literals autoaa/aatoau of k^x2, R0, K^bo and the EEQ-BC
radii), an integer electron count Σ Z_l − Q, and each hard cutoff replaced by a quintic
smoothstep from the program's radius to 1 bohr beyond it (no integral pair cutoff); its
gradient takes every term from the final density. Parameter literals (0.666666666, the
`pa_rvdw_scale` defaults, 1.889725949 of the H0 radii, the smoothed √CN) are the method's and
kept in both.

**Reproduction** (`tests/reference_v2.rs`, `tests/gradient_v2.rs`,
`tests/gxtb2_reproduction.py` → `bench/results/gxtb2_reproduction.json`; program mode, 84
molecules H–Lr, 23 with lanthanides or actinides, two of them longer than 50 bohr):

| quantity | result |
|---|---|
| every term at the program's density (12 molecules incl. CeCl3, GdCl3, UF6, AuCl, 4 open shells) | ≤ 1.3e-14 Eh; electronic energy to print precision; Mulliken charges 4e-15 |
| total energy (12-decimal printout) | ≤ 1.8e-12 Eh for 83 molecules; 9.1e-12 Eh for (H2O)100 (7644 Eh, 1.2e-15 relative) |
| SCF iterations, default and tight thresholds | identical for all 84 (TmH3's tight SCF fails in both at 250 iterations) |
| per-term energies after the SCF | ≤ 9.1e-12 Eh except TbCl3 2.6e-11: one orbital of a (near-)degenerate HOMO set carries the program's fractional remainder (Σ n⁰ is not an integer), so the unconverged part of the final density depends on the eigensolver and the summation order at round-off level (methane, hydroxide, AuCl move between 1e-12 and 2e-10 with them); the totals agree to ≤ 9e-13 |
| analytic gradient vs the program's, default thresholds | ≤ 2e-11 Eh/bohr |
| analytic gradient vs the program's, tight thresholds | ≤ 6e-11 Eh/bohr except CeCl3 4.0e-9 (f¹, seven near-degenerate f orbitals; 45 iterations: the summation order of the density matrix alone moves this gradient by up to 6e-8 Eh/bohr at the same energy and iterations) |
| default conventions vs finite differences of their own energy (5-point, h = 5e-4 bohr) | ≤ 6e-10 Eh/bohr (water, O2, mesyl chloride, [CuCl4]²⁻, Ni(CO)4); smooth cutoffs 3e-11 |

**Default vs program conventions**: total energies differ by ≤ 8.7e-10 Eh for the 82 small
molecules ([Co(H2O)6]²⁺) with identical iteration counts; binding energies (iodobenzene···acetone,
CF3Br···benzene from `tests/structures`) by ≤ 1.7e-11 Eh, 2 % expansions of eight molecules by
≤ 8.7e-10 Eh (2.3e-8 Eh for C50H102); C50H102 −5.3e-8 Eh and (H2O)100 −1.9e-6 Eh, almost all from
the revD4 three-body tail beyond 25 bohr, which the program cuts and the smooth switch keeps to
26 bohr.

**Cost** (laptop, i9-11900H, whole processes, energy + gradient, 1 / 8 threads): C50H102 (152
atoms) 0.34 / 0.13 s vs the program's 0.64 / 0.24 s; (H2O)100 2.3 / 0.65 s vs 3.9 / 1.1 s. What
made it fast and the node's numbers: `bench/README.md`, "g-xTB 2.0.1 performance".

**Finite electronic temperature** (`ScfOptions::etemp`, kelvin; 0 by default as in the program;
`tests/etemp_v2.rs`, data `tests/reference_v2/etemp/` from `scripts/make_reference_v2_etemp.py`:
57 of the reference molecules plus Fe4 (tetrahedron, 2.30 Å) and N2 at 1.65 Å, both singlets, at
300, 1000, 3000 and 5000 K, the program's default and tight thresholds, 472 program runs). Program mode
fills as the program (§4.17.1: integer target per channel, its Newton steps and thresholds,
kB = 3.166808578545117e-6 Eh/K) and adds −TS to the electronic energy; the default conventions
fill the integer electron count with the exact Fermi level (Newton with bisection safeguard, to
round-off) and CODATA 2022 kB (3.1668115634564e-6 Eh/K). The gradient is the 0 K formula with
the smeared occupations in both (no occupation response: the free energy is stationary in the
occupations).

| quantity (program mode) | result |
|---|---|
| total energy, default thresholds (every converged run that is not rounding-sensitive: 229 of 236) | ≤ 1.8e-12 Eh; (H2O)100 1.1e-11 Eh (1.4e-15 relative, as at 0 K) |
| printed electronic energy (incl. −TS) | ≤ 2.6e-12 Eh |
| SCF iterations, default thresholds | identical; Fe4 at 300 K fails in both (250 iterations) |
| analytic gradient vs the program's, default thresholds | ≤ 4.7e-11 Eh/bohr |
| tight thresholds (`--acc 0.0001`) | energies ≤ 2.0e-12 Eh for every run that converges ((H2O)100 9.1e-12 Eh); gradients ≤ 2.9e-11 Eh/bohr; failures (TmH3 at every T, Fe4 at 300, 3000 and 5000 K) as the program's; EuCl3 at 5000 K is rounding-sensitive: the program stops at 250 iterations with the residual just above the threshold, axtb converges (in 102-250 iterations as kT moves by ± 1 ulp or the summation order changes) |
| rounding-sensitive runs (§4.17.1) | default thresholds: Fe4 1000/3000/5000 K (1.5e-6, 2.5e-6, 4.8e-7 Eh; 33 vs 32 iterations at 3000 K), TmH3 3000 K (1.4e-6 Eh, 75 vs 72 iterations), TmH3 5000 K (no convergence in 250 iterations; the program: 158), ErF3 5000 K (5.7e-9 Eh); tight: iteration counts of CeCl3 5000 K (75 vs 51), ErF3 3000 K (29 vs 31) and ErF3 5000 K (78 vs 79) differ (energies ≤ 5e-13 Eh), Fe4 1000 K ends at 250 iterations (residual 3e-9) where the program converges, EuCl3 5000 K as above. A one-ulp change of kT in our SCF moves each by as much. |
| default conventions vs finite differences of their free energy (5-point, h = 5e-4 bohr, 5000 K) | 1.1e-11 (stretched N2), 7.6e-10 ([CuCl4]²⁻), 3.0e-10 (CeCl3, −TS = −0.039 Eh) Eh/bohr |

**kB of the default conventions**: CODATA 2022 kB is 9.43e-7 relative larger than the program's;
the free energy changes by that fraction of −TS (`tests/gxtb_etemp_reproduction.py`: the default
conventions at T against T·kB_program/kB_CODATA): ≤ 3e-11 Eh where −TS < 1e-4 Eh, 1.3e-9 Eh for
GdH3 at 5000 K (−TS = −1.3e-3 Eh), 1.3e-8 Eh for ErF3 at 3000 K, 3.7e-8 Eh for CeCl3 and 4.5e-8 Eh
for EuCl3 at 5000 K (−TS ≈ −0.04 Eh). Otherwise the default conventions differ from the program
at finite T as at 0 K (units, cutoffs; (H2O)100 1.9e-6 Eh, C50H102 5.2e-8 Eh, the revD4 tail),
and the exact Fermi level adds up to 2.9e-8 Eh (TbCl3 at 3000 K; ≤ 7e-9 for the others): the
program's electron count is exact only to its 2⁻²⁶ threshold, so its energy carries μ·δN
(`bench/results/gxtb_etemp.json`).

Analytic and response Hessians at a finite electronic temperature: §11.6.

Restarts: `Calculator` starts each SCF from the previous geometry's converged state in the
default conventions and, with `guess = Previous`, reproduces the program's `xtbrestart` in its
own (§4.17.2).

---

## 11. Hessians (`v2/response.rs`, `v2/hessian.rs`, `v2/tangent.rs`, `v2/analytic.rs`, `v2/explicit.rs`)

Three methods (`Calculator::hessian`, CLI `--hess-method`, Python `Calculator.hessian(method=…)`,
qccodec `hessian_method`): **analytic** (the 2.0.1 default), **response** and **numerical** (central
differences of analytic gradients, the only method of v1.1). The 2.0.1 program's own `--hess` is
numerical (§11.5).

### 11.1 Coupled-perturbed response of the full density

The Fock matrix is the exact derivative of the energy (§4.16), so the response of the converged SCF to
a perturbation with first-order Fock matrix F^(y) (at fixed AO density) and overlap S^y is

    dF = F^(y) + K[dP],   dP = dP^S + chi0[dF],

with the orthonormality part dP^S = −C (S̃ ∘ D2) Cᵀ, the non-interacting response
chi0[X] = C (CᵀXC ∘ D1) Cᵀ (S̃ = Cᵀ S^y C; D1, D2, D3 the divided differences of f, fε, fε² between
orbitals per spin channel; at 0 K the occupations, incl. the program's fractional remainder, are
fixed) and dW = C (CᵀdFC ∘ D2 − S̃ ∘ D3) Cᵀ. At a finite electronic temperature (§11.6) the
occupations follow the levels: D1, D2, D3 take the derivatives n' = dn/dε on the diagonal and between
degenerate orbitals, and each channel's Fermi level shifts to keep its electron count. Unlike GFN2, the kernel K = d²E/dPdP cannot be reduced to
shell charges and atomic multipoles: the Mulliken-approximated exchange, the bond-order correction
(quadratic in P) and the charge-dependent onsite exchange depend on the whole density matrix.
`Setup::fock_response` is the exact linearization of every Fock contribution (E1–E4 Hessian, AES,
spin, repulsion and revD4 charge kernels, exchange at fixed q plus its Λ(q) and shell-potential
couplings). In the space of orbital pairs with D1 < 0 the equations become the symmetric positive
definite system (1 + d^½ K̃ d^½) z = d^½ b (d = −D1), solved by conjugate gradients per perturbation
(tolerance 1e-10 relative; 15–26 iterations). Per kernel application the exchange's Fock change takes
four matrix products (dP S, S dP S, S X and (B + S X/2) S, whose sum with its transpose is
B S + (B S)ᵀ + S X S), its onsite (Λ(q), r(q)) terms run over the atom blocks only, and the MO
transforms form only the response columns (from the first orbital pair with d ≠ 0: the virtuals at
0 K). A preconditioner that inverts the operator's shell-charge part exactly (1 + ½ Uᵀ K_q U, with U
the Mulliken shell charges of the density of z and K_q = d²E/dq dq, by the Woodbury identity with an
nsh × nsh matrix) cuts the iterations to 12–14 (taxol 19 → 13; the bulk of the spectrum is the
exchange: adding the multipoles, or the exact diagonal of the rest, saves at most one more), but each
application costs two n_o × 2n × n_v products; it made the taxol Hessian only 7 % faster (one thread,
quiet cluster node; 12–13 % with the node loaded) and was left out.

### 11.2 Response Hessian

Column y is the central difference of the analytic gradient functional g(R ± h e_y; P ± h dP^y,
W ± h dW^y) (the wavefunction extrapolated by the exact response; no SCF per displacement; the
second-order extrapolation errors cancel). F^(y), S^y come from the same two displaced setups at fixed
density. Default h = 2e-4 bohr; error O(h²).

### 11.3 Analytic Hessian

    H_xy = E^xy|_{P,W} + Σ_s Tr(F^(x)_s dP^y_s) − Tr(S^x dW^y)

- **Tangents** (`tangent.rs`): every geometry dependence of S, the dipole and quadrupole integrals and
  H0 + ACP factors into local blocks (rows of one atom's AOs): the displacement of an atom, its q_eff,
  q_eff0 and g-xTB CN; the Cartesian derivative is their sum with the Jacobians dq_eff/dR (EEQ-BC
  charges and basis CN), dq_eff0/dR, dCN/dR, plus a low-rank term for the ACP projectors of the
  displaced atom. The anisotropy term enters as a dipole potential.
- **F^(x)** (`analytic.rs`): the tangent-linear Fock matrix: integral derivatives with the reference
  potentials; potential changes = K b^x + c^x with b^x the SCC-variable change at fixed P and c^x the
  explicit potential derivative at fixed SCC variables; the exchange's overlap derivatives (E_x is a
  homogeneous quadratic in S at fixed P, so its Fock matrix, shell potential, dE/dS and dE/dG tangents
  are exact products), its kernel (MFX, bond order) derivatives and its Λ(q) change.
- **Explicit part** at fixed P, W: b^x·K b^y + c^x·b^y + b^x·c^y (SCC couplings, K = d²E/ds ds), the
  exchange's d²E_x/dS dS[S^x, S^y] and overlap–kernel cross terms, the integral second derivatives
  contracted with the gradient's weights (overlap weight incl. −W, dipole/quadrupole potentials, P) in
  local variables chained with the Jacobians, the second derivatives of the chains (EEQ-BC charges,
  basis and g-xTB CNs, the q_eff formula), and the kernel terms at fixed SCC variables.
- **Finite-difference parts** (deterministic, no SCF, local): second derivatives of each SCF-basis
  integral pair and each H0 pair (diatomic frame, distance polynomial, self-energies) by 4-point
  differences of the pair's analytic gradient over its local variables (bond vector, q_eff, q_eff0,
  CN); the ACP projector overlaps' second derivatives likewise per (projector, atom) pair; the kernel
  terms (first to fourth order, AES, repulsion, revD4 two- and three-body, the anisotropy vectors)
  and c^x by 4-point differences of their analytic gradient and potentials over the Cartesian
  displacements at fixed SCC variables (the kernels are rebuilt, the integrals are not); the EEQ-BC
  and CN second derivatives by 4-point differences of their analytic adjoint gradients. Steps 1e-3
  bohr (and units of charge); errors O(h⁴). Pairs whose primitive pairs are all screened at every
  displaced geometry (a_i a_j (R − 3h)²/(a_i + a_j) > 25) give exactly zero and are skipped. The
  exchange kernels (MFX shell kernel, bond-order kernel) at the fixed weights dE/dG, dE/dK are pair
  functions of the distance and have a closed-form Hessian (G'' u uᵀ + G'/R (1 − u uᵀ)); it replaced
  their differences (changes ≤ 1e-13 Eh/bohr²). The analytic second derivatives of the integral
  pairs (`multipole_hess`) were tried for the positions and left out: they move the heavy-element
  Hessians by the differences' screening noise (UF6 2e-7, caffeine 2e-9 Eh/bohr²), more than
  round-off.
- **Contraction and memory** (`Analytic::response_column`, `hessian::AnalyticOptions`): the response
  term Σ_s Tr(F^(x)_s dP^y_s) − Tr(S^x dW^y) is contracted per response y for every x at once without
  forming any F^(x): it is the derivative of the gradient functional g_x(P, W) along (dP^y, dW^y).
  The Fock tangent at the reference potentials (H0 + ACP rows, Mulliken shell potential × overlap
  rows, multipole potentials × dipole and quadrupole rows and columns) is folded into one set of local
  rows per atom displacement and per q_eff (q_eff0, CN: their H0 rows), contracted with dP + dPᵀ,
  plus the ACP projectors of the displaced atom (one product with the projector columns); the
  overlap rows take the exchange's dE/dS tangent along dP (the transpose of its overlap-tangent Fock
  matrix), the spin potential times the magnetization density and −dW; the MFX and bond-order kernel
  derivatives take the tangent of dE/dG, dE/dK along dP; the SCC-variable potentials c^x + K b^x and
  the exchange's charge coupling (Λ(q)) act through the Mulliken analysis of dP^y and the change of
  the exchange shell potential, summed for all pairs in one matrix product with the SCC couplings.
  b^x of every displacement comes from per-block moment vectors and the q_eff Jacobian; the dipole,
  quadrupole and remaining H0 blocks of the tangents are then dropped. Each displacement is one task
  (F^(x), S^x, the response, the contraction), after which its n×n matrices are freed; blocks of 64
  tasks run in parallel and are summed in order. Memory is O(N²) (local rows ~10 n², vectors of the
  SCC layout, the Hessian) plus a few n×n matrices per thread, and the result depends neither on the
  block size nor on the thread count. The ACP part of the integral local Hessian (one dense block
  over each projector's variables) is summed in chunks of projectors.

### 11.4 Accuracy and cost

`tests/gxtb2_hessian.py` → `bench/results/gxtb2_hessian.json` (laptop i9-11900H, loads recorded per
run). Max |ΔH| (Eh/bohr²) of the default settings against two independent references: (a) the
Richardson extrapolation of numerical Hessians (h = 0.002/0.001 bohr, SCF accuracy 1e-6) and (b) the
Richardson extrapolation of response Hessians (h = 2e-4/1e-4, no SCF noise). Both references still
see tblite's primitive screening (pairs with a_i a_j R²/(a_i + a_j) > 25 dropped), which steps the
energy surface by ~1e-11 Eh: heavy-element references (Fe, Ni, U) scatter by up to ~1e-7, and the
two references then differ from each other by that much; the analytic Hessian is free of it.

| molecule | analytic vs (a) / (b) | response vs (a) | numerical vs (a) |
|---|---|---|---|
| water | 1.6e-10 / 4.0e-12 | 3.2e-8 | 1.9e-5 |
| O2 triplet | 2.7e-12 / 5.7e-12 | 4.3e-8 | 3.1e-5 |
| caffeine | 1.3e-9 / 2.0e-8 | 3.9e-8 | 2.7e-5 |
| [Cu(NH3)4]²⁺ doublet | 1.7e-8 / 7.8e-11 | 3.1e-8 | 2.6e-5 |
| ferrocene | 1.1e-7 / 1.4e-10 | 1.1e-7 | 6.2e-5 |
| Ni(CO)4 | 6.0e-8 / 2.6e-11 | 5.9e-8 | 7.2e-6 |
| UF6 | 5.3e-9 / 5.1e-7 | 7.1e-7 | 1.5e-5 |
| GdCl3 octet | 7.6e-9 / 5.3e-11 | 9.2e-9 | 2.7e-6 |

The analytic Hessian agrees with one of the two references to ≤ 5.3e-9 for every molecule (with
tight SCFs in `tests/hessian_v2.rs`, both conventions: ≤ 7.6e-9); asymmetry before symmetrization
≤ 2.4e-11. Its error is first order in the reference SCF residual, hence the default reference
accuracy 1e-5 (1e-3 leaves 1.8e-7 on caffeine at the same cost). The response Hessian's O(h²)
error is ~3e-8 at h = 2e-4; the default numerical Hessian (h = 0.005, displacement SCFs at
accuracy 0.01) is off by 3e-6 to 6e-5 (frequencies up to 0.3 cm⁻¹).

**Cost** (wall seconds, laptop i9-11900H, interleaved with the program, load 1–2.6 before each
run; program conventions for the head-to-head, `tests/gxtb2_hessian.py --versus`,
`tests/gxtb2_program_accuracy.py --interleaved`):

| | caffeine 1 / 8 threads | taxol 1 / 8 threads |
|---|---|---|
| axtb analytic (fd71faa) | 1.7 / 0.42 | 152 / 36 |
| axtb analytic (03fcd95)² | 1.6 / 0.44 | 137 / 33 |
| axtb response | 5.7 / 0.94 | 500 / 89 |
| axtb numerical | 6.9 / 1.14 | 730 / 127 |
| program `--hess` (default `--acc 1`) | 6.6 / 1.05 | 450 / 80 |
| program `--hess --acc 0.001` (a common tight setting) | 6.9 / 1.07 | 534 / 95 |
| program `--hess --acc 1e-4` | 7.0 / 1.13 | 581 / 102 |

² a later session with other jobs on the laptop (`tests/gxtb2_hessian.py`, load 4.4; taxol on one
thread from three interleaved rounds with fd71faa, load 2.6–11 before each run: fd71faa 168 s,
03fcd95 137 s best of 3, 1.23×; per round 1.18–1.56×; user CPU 166 vs 122 s).

Taxol analytic phases (one thread), fd71faa: tangents 1.4 s, perturbations F^(x) etc. 33 s,
coupled-perturbed response 105 s (15–26 CG iterations of a 13 ms kernel per displacement), explicit
integral terms 17 s; 03fcd95 (load 4.4): 0.9, 28, 99 and 20 s.

**Speed per change** (taxol, one thread, cluster node with a 9985WX, one 96-CPU pod, three
interleaved rounds with the order rotated, best wall time; the node's load from other jobs rose
from 2 to 35 during the rounds; `speed_per_change_cluster` in the results file):

| build | wall | response | perturbations | explicit |
|---|---|---|---|---|
| O(N²) memory (59cc1a4) | 119.1 s | 80.4 | 23.0 | 13.3 |
| + response kernel (four exchange products, block-diagonal onsite terms, response columns only, faster AES and moments) and screened pairs | 77.1 s | 44.6 | 18.5 | 11.9 |
| + closed-form exchange-kernel Hessian (03fcd95) | 73.4 s | 44.5 | 15.6 | 11.0 |
| + Woodbury shell-charge preconditioner (not kept) | 68.2 s | 36.1 | 15.8 | 14.0 |

Per round the kernel changes gave 1.24–1.66×, the closed form 2.4–4.8 % (the perturbations 3 s
less in every round), the preconditioner 7 % in the quietest round and 12–13 % in the loaded ones.
Hessians: the kernel changes and the closed form agree with fd71faa to ≤ 4.3e-12 Eh/bohr² (benchmark
set, both conventions, 0–5000 K), the preconditioned solver to ≤ 2.5e-11.

**Memory** (peak resident set from `/usr/bin/time -v`, default conventions, through the CLI, laptop
except (H2O)200; `tests/gxtb2_hessian.py --memory`, `bench/results/gxtb2_hessian.json`). Before
(fd71faa) the analytic Hessian kept F^(x) and S^x of every displacement (O(N³)) and the ACP part of
the integral local Hessian held one dense block per projector at once (0.7 GB for taxol); after
(03fcd95; §11.3, "Contraction and memory"):

| | atoms / AOs | threads | before | after | single point |
|---|---|---|---|---|---|
| caffeine | 24 / 66 | 1 / 8 | 29 / 35 MB | 14 / 19 MB | 10 MB |
| taxol | 113 / 299 | 1 / 8 | 1358 / 1441 MB | 139 / 235 MB | 49 MB |
| C50H102 | 152 / 302 | 8 | 2694 MB | 269 MB | 52 MB |
| (H2O)100 | 300 / 600 | 8 | killed at a 12 GB limit (7.6 min) | 992 MB (423 s) | 176 MB |
| (H2O)200 | 600 / 1200 | 8 | ≈ 100 GB (estimate, not run) | 4569 MB (36 min)¹ | 664 MB¹ |

¹ Cluster node (AMD Threadripper PRO 9985WX), 8-CPU pod, commit 59cc1a4.

What remains: the local rows (~10 n²), four vectors of the SCC layout per displacement and the
Hessian (O(N²)); a few tens of n×n matrices per thread for the displacement in flight (~25 n²:
70 MB per thread for (H2O)100, 0.3 GB for (H2O)200, most of its 4.6 GB at 8 threads); and,
briefly while the rows are built, the full tangent blocks (~80 n²).

### 11.5 The 2.0.1 program's `--hess`

The program's Hessian is numerical (xtb `numhess` → `calculator%hessian`): central differences
(h = 0.005 bohr) of the gradients of SCFs restarted from the reference wavefunction and stopped at
the calculator's thresholds (`--acc`; `$hess sccacc` does not reach the g-xTB calculator: sccacc
0.3, 0.01 and 1e-4 give bit-identical Hessians). Its gradient takes the Mulliken and multipole
potential terms from the previous SCF iteration's charges (§4.18), so each displaced gradient errs
by about the SCF residual, which the central difference magnifies by 1/(2h) = 100. Largest (mean)
frequency difference to axtb's analytic Hessian, which matches the tight references to ≤ 1e-5 cm⁻¹:

| program setting | Ni(CO)4 | caffeine | taxol |
|---|---|---|---|
| `--acc 1` (default) | 29 (11) | 104 (30) | 149 (78) |
| `--acc 0.01` | 1.5 (0.32) | 4.3 (0.29) | 9.9 (1.4) |
| `--acc 0.001` (a common tight setting) | | 0.54 (0.03) | 1.07 (0.14) |
| `--acc 1e-4` | 0.055 (0.020) | 0.35 (0.021) | 0.28 (0.016) |

(water 0.25, O2 0.03 cm⁻¹ at the default; GdCl3 92 cm⁻¹ at its optimized geometry). At 1e-4 what
remains is the truncation error of h = 0.005 bohr. Central differences of the program's own analytic gradients from
fresh, tight runs (`--grad --acc 1e-4`) agree with axtb's analytic Hessian to 7e-6 Eh/bohr²
(Ni(CO)4, 0.05 cm⁻¹; caffeine 2.7e-5, 0.32 cm⁻¹, the h = 0.005 truncation), so both programs describe
the same surface; the program's default `--hess` does not resolve it. axtb's analytic Hessian agrees
with the tight reference (program conventions) to 1.3e-9 Eh/bohr² for caffeine and 8.2e-8 for taxol
(≤ 1e-4 cm⁻¹). Against the program at `--acc 0.001` it is 4.1× (caffeine) and 3.4× (taxol)
faster on one thread, 2.5× and 2.7× on eight, and exact where the program's frequencies are off by
up to 0.54 and 1.07 cm⁻¹. The program's printed frequencies equal those of its `hessian` file with its
atomic masses (≤ 5e-3 cm⁻¹); it rotates the molecule by 1e-4 rad before the Hessian (frequencies
unchanged).

---


### 11.6 Finite electronic temperature

With Fermi occupations n_p = n_max f((ε_p − μ)/kT) the free energy E − TS is stationary in the
occupations, so the gradient keeps its 0 K form (§4.17.1) and the Hessian is its derivative with the
response of the occupations. In the orbital basis of a channel, with x = CᵀdFC and S̃ = CᵀS^yC,

    dn_p = n'_p (x_pp − ε_p S̃_pp − dμ),   dμ = Σ_p n'_p (x_pp − ε_p S̃_pp) / Σ_p n'_p,
    n'_p = −n_p (1 − n_p/n_max)/kT,

which the response (`response.rs`, `Channel`) takes up as: D1_pp = n'_p and D1_pq = ½(n'_p + n'_q)
for degenerate pairs (D2, D3 follow: n + εn', 2nε + ε²n'); dP^S gains n'_p Σ n'_r ε_r S̃_rr / Σ n'
on the diagonal and dW gains −n'_p ε_p dμ; χ0 loses n'n'ᵀ/Σn' on the diagonal, which in the
symmetric system (1 + d^½ K̃ d^½) z = d^½ b gives the diagonal pairs d^½ = (−n'/2)^½ and the
projector 1 − vvᵀ, v_p = (n'_p/Σn')^½ (applied before and after the kernel). Closed shells use
n_max = 2 with the one channel's Fermi level; program mode uses the same formulas with its
occupations (its electron count is exact to its 2⁻²⁶ threshold).

Accuracy (`tests/etemp_hessian_v2.rs`, Richardson reference of analytic-gradient differences with
h = 2e-3/1e-3 bohr, SCF 1e-12 Eh / 5e-13; both conventions): water, caffeine, O2 (triplet),
[CuCl4]²⁻ and Ni(CO)4 at 1000 and 3000 K ≤ 2.8e-9 Eh/bohr² (analytic) and ≤ 4.3e-8 (response);
stretched N2 at 3000/5000 K and AgCl at 5000 K (−TS = −4.4e-5 and −3.6e-4 Eh) ≤ 7.5e-10 (analytic).
Without the occupation response the AgCl Hessian would be off by 9.1e-5 Eh/bohr². CeCl3 has no
stable SCF solution to differentiate (0 K: no convergence; 1000–5000 K: the electronic Hessian is
not positive definite, so the response refuses it; use the numerical Hessian).

## 12. Element coverage and energy differences

**Every element H–Lr** (`tests/gxtb_elements.py`, `bench/results/gxtb_elements.json`): 169
generated molecules (the GFN2 reference matrix's element molecules with high-spin f^n states,
actinides 87–103 added the same way, and a second environment for many elements: aqua and
ammine ions, halides, oxides, radicals), the program (`xtb --gxtb --grad`, default and
`--acc 0.0001`) against axtb in program conventions:

- 155 agree: total energies ≤ 4.9e-11 Eh, the same SCF iterations, the printed terms (core
  increments, three-body dispersion, electronic energy) ≤ 1.1e-12 Eh, analytic gradients
  ≤ 2.3e-10 Eh/bohr (default; NdCl3, an open f shell, 2.5e-9) and ≤ 2.1e-10 (tight); the default
  conventions are ≤ 9.0e-10 Eh from the program. This includes H2, which needed the early DIIS
  switch (§4.17; before it, 5 iterations against the program's 4).
- 12 are rounding-sensitive, diagnosed by rerunning both programs on a translated copy with the
  atoms in reverse order and, for gradient mismatches, on a rotated copy (2.0.1 is rotation
  invariant to ~1e-13 Eh, so both only change the round-off): open f shells of the LnCl3 and
  AnCl3 at their symmetric geometries (Ce, Gd, Tb, Er, Tm, Cf, Fm: energies ≤ 6.1e-10 Eh apart,
  gradients 1.9e-8 to 1.3e-6 Eh/bohr; the SCF breaks the symmetry by round-off, axtb's own
  gradient moves as much under a rotation, and the energy has a cusp there: ErCl3's FD
  derivative along a symmetry-forbidden direction is 0, axtb's analytic one 1.1e-6),
  partly filled degenerate sets (VCl4 d¹, UCl4 and PuF6 f²: energies 1.3e-8 to 4.6e-6 Eh,
  gradients to 5e-3), BkCl3 (45 vs 59 iterations, 8.1e-9 Eh) and HeH⁺ (9 vs 12 iterations and a
  gradient 1.2e-5 Eh/bohr apart at the default thresholds, 11 vs 10 on the shifted copy; the
  tight runs agree).
- 2 are the program's gradient error for bonds along an axis, [Gd(H2O)6]³⁺ and [Lu(H2O)6]³⁺
  (§4.18).
- Tight runs differ in addition for NeH⁺ and [Cu(H2O)6]²⁺ (iteration counts that move with
  the round-off in both programs: 10/11 and 89/83 in the program) and TmCl3 (the program
  converges in 98 iterations, fails like axtb on the perturbed copies).

**Energy differences, default − program conventions** (`tests/gxtb_energy_differences.py`,
`bench/results/gxtb_energy_differences.json`; SCF accuracy 1e-4). Minima and saddle points of
2.0.1 itself (default conventions; geomeTRIC GAU_VERYTIGHT, Sella fmax 0.001 eV/Å from the
analytic Hessian, one imaginary mode and a pseudo-IRC to the intended minima), stored in
`tests/structures/energy_differences/v2.0.1/`:

| quantity | `gxtb` (kcal/mol) | change with `axtb` (Eh) |
|---|---|---|
| n-butane gauche − anti | 0.4892 | −1.8e-12 |
| ethanol OH gauche − anti | −0.0344 | −2.1e-12 |
| ferrocene staggered (D5d, ring-rotation saddle) − eclipsed | 0.1800 | +4.5e-13 |
| Cl⁻ + CH3Cl → [Cl···CH3Cl]⁻, complexation | −9.2688 | +2.9e-11 |
| Cl⁻ + CH3Cl SN2 identity reaction, barrier from the complex (TS 443i cm⁻¹) | 12.9874 | −1.1e-11 |
| same, TS − separated reactants | 3.7186 | +1.8e-11 |
| HCN → HNC, reaction energy | 7.1442 | +3.2e-12 |
| HCN → HNC, barrier (TS 1050i cm⁻¹) | 51.7573 | −3.7e-11 |
| water dimer binding | −4.8961 | +2.5e-12 |
| iodobenzene···acetone binding (`tests/structures/beni-acetone.xyz`) | −2.9485 | +9.5e-12 |
| ethylammonium perchlorate ion-pair binding (`tests/structures/s229-cl.xyz`) | −103.2545 | −9.0e-12 |
| Ni(CO)4 → Ni(CO)3 + CO (unrelaxed) | 32.8744 | +8.5e-14 |
| [Cu(NH3)4]²⁺ → [Cu(NH3)3]²⁺ + NH3 (unrelaxed) | 47.4586 | +7.8e-12 |

So for molecules of this size the corrected default conventions change energy differences by
≤ 4e-11 Eh (g-xTB v1.1: ≤ 1.1e-5 Eh, `docs/gxtb-v1.1-spec.md` §4.18); `tests/gxtb2_reproduction.py`
has the larger systems (2.3e-8 Eh for a 2 % expansion of C50H102, 1.9e-6 Eh in the total of
(H2O)100).

## Appendix A. Capture points used in the decoding (2.0.1 binary, gdb)

Intel Fortran passes every argument by reference; assumed-shape arrays as a pointer to a
descriptor (base, element length, offset, flags, rank, reserved, then extent, stride in bytes,
lower bound per dimension). Breakpoints at the dispatcher symbol (no suffix) catch every variant.

| quantity | where |
|---|---|
| coordinates (bohr) | `tblite_xtb_singlepoint_mp_xtb_singlepoint_`, arg 1 = `structure_type`; nat = first int; `xyz` descriptor at +232 |
| EEQ-BC charges, q_eff, cgto k's | `tblite_basis_qvszp_mp_get_qeff_` (charges as its `qat` argument) |
| g-xTB CN, self-energies | `tblite_xtb_h0_mp_get_selfenergy_` arguments |
| H0 parameters | `tblite_xtb_h0_mp_get_hamiltonian_` args (mol, trans, list, bas, H0-basis cgtos, h0, selfenergy, aniso(3,nat), overlap, dpint, qpint, hamiltonian); h0 object (arg 6): +0x0 selfenergy, +0x60 kcn, +0x1e0 hscale, +0x270 shpoly (zero), +0x2d0 shpoly2, +0x330 shpoly4, +0x390 rad, +0x3d8 pair dip scale, +0x438 pair R̄, +0x498 anisotropy steepness, +0x4a0 refocc, +0x500 kσ, +0x560 kπ, +0x5c0 kδ |
| anisotropy vectors | `tblite_xtb_h0_mp_get_anisotropy_(h0, mol, aniso)` output |
| ACP | `tblite_acp_type_mp_get_acp_` (S_pv via `get_pv_overlap_0d`) |
| repulsion | `get_scaled_alpha`, `get_repulsion_matrix`, `tblite_repulsion_gxtb_mp_get_energy_` |
| revD4 | `tblite_disp_d4_mp_new_d4srev_dispersion_` (self, mol, 2, 2, s6, s8, a1, a2, a3, a4, s9, 7.5), `dftd4_model_d4srev` `get_2b_coeffs`, `tblite_disp_d4_mp_get_dispersion_matrix_` (cutoff 50), `get_dispersion3` (25) |
| Coulomb constructors | `new_onsite_firstorder`, `new_effective_coulomb`, `new_twobody_thirdorder`, `new_onsite_fourthorder`, `new_gxtb_multipole` arguments (§3.2 constants) |
| exchange | `tblite_exchange_fock_mp_new_exchange_fock_` (23 args); exchange cache: +0x140 MFX shell kernel, +0x1a0 Λ(l,l',atom), +0x218 dΛ/dq, +0x290 r, +0x2f0 dr/dq, +0x350 K^bo, +0x3b0 F_x (nao,nao,nspin), +0x428 v_sh |
| per-term energies | finish breakpoints on each container's `get_energy` (`tblite_coulomb_*`, `tblite_spin_mp_get_energy_`, `tblite_disp_d4_mp_get_energy_`, `tblite_exchange_type_mp_get_energy_w_overlap_`, `tblite_repulsion_gxtb_mp_get_energy_`; pre-SCF `tblite_classical_increment_mp_get_engrad_`, `tblite_disp_d4_mp_get_engrad_`): difference of the summed per-atom energy argument (`make_reference_v2.py`) |
| SCF mixer | `tblite_scf_mixer_type_mp_{set,diff,get}_1d_` arg 2 (vector descriptor); mixer calls `tblite_scf_mixer_{simple,diis}_mp_next_` |
| guess | `tblite_wavefunction_guess_mp_eeqbc_guess_qsh_`, `shell_partition_` |
| occupations | `tblite_wavefunction_fermi_mp_get_aufbau_filling_(nel, homo, occ)` |
| Mulliken charges, density | `tblite_wavefunction_mulliken_mp_get_mulliken_shell_charges_(bas, S, P, n0sh, qsh)` |
