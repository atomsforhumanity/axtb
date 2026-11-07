# g-xTB v1.1 specification (for axtb)

What the public g-xTB 1.1.0 program computes, term by term, with every entry of its three
parameter files mapped to the term that uses it, plus its Fock matrices and SCF procedure.
Written for `crates/gxtb` (package `axtb-gxtb`), which implements all of it: single points
reproduce the program's total energies to ~1e-12 Eh (§4.17, §7), and analytic gradients
(which the program does not have) are exact derivatives of the energy (§4.18). By default
(`conventions = "axtb"`) the SCF diagonalizes the exact derivative of the energy;
`conventions = "gxtb"` uses the program's Fock matrix and reproduces its numbers (§4.18).

**Sources**

- Preprint: T. Froitzheim, M. Müller, A. Hansen, S. Grimme, "g-xTB: A General-Purpose
  Extended Tight-Binding Electronic Structure Method For the Elements H to Lr (Z=1–103)",
  ChemRxiv 2025, doi:10.26434/chemrxiv-2025-bjxvt (main text, 49 pp.). Equation numbers
  below ("P-Eq. n") refer to it. **The SI (full derivation, spin constants, exchange
  integrals) was not available** (ChemRxiv blocks scripted downloads); where the main text
  defers to the SI, this document says so.
- Program: g-xTB 1.1.0, grimme-lab/g-xtb release v1.1.0 (GPL-3.0), in
  `reference/gxtb-v1.1.0/`. It is a statically linked, **unstripped** Intel Fortran
  executable ("preliminary g-xTB(EEQ BC) version: Tue Aug 12 2025"). Its symbol table names
  the routines (`repuls_`, `setgab_lrao_`, `calcvacp_`, ...) and the module arrays that hold
  the parameters (`parcom::gp3_*`, `eeqcom::eeq_*`, `com::*`). The facts below come from (a)
  the read statements of `rdpar_`, `rdbas_`, `rdpar_eeq_`; (b) which routine reads which
  array; (c) disassembly of individual routines; (d) full-precision values read from the
  running program with gdb; (e) single-parameter perturbations of the input files.
- Public code: multicharge (`eeqbc2025`, Apache-2.0) and mctc-lib (`ncoord/erf_en`), which
  match the program's EEQ-BC exactly; tblite branches of thfroitzheim (`diat_trafo.f90`).
  github.com/atomag/gxtb-torch (no licence) was only looked at: its parameter mapping is
  self-declared guesswork and disagrees with the program in most entries; nothing was taken
  from it.

**Confidence labels**

| Label | Meaning |
|---|---|
| **[C]** | Confirmed: formula decoded and reproduced numerically against the program (to ~1e-15 unless stated); covered by `crates/gxtb/tests/reference.rs`. |
| **[D]** | Decoded from the program (read statement, array usage, disassembly or memory dump), not yet reproduced numerically. |
| **[P]** | From the preprint; the parameter assignment is inferred from which routine reads the array and how the energy responds to perturbing it. |
| **[U]** | Unknown. |

---

## 1. The program

### 1.1 Running it

```
cd <workdir> && OMP_NUM_THREADS=1 HOME=<dir with .gxtb .basisq .eeq> gxtb -c mol.xyz [options]
```

- Parameter files are read from `$HOME/.gxtb`, `$HOME/.basisq`, `$HOME/.eeq` (a `GXTBHOME`
  variable also exists). Long paths are truncated: `HOME=.` with the three files copied into
  the working directory always works.
- **Charge and spin are not command-line options**: the program reads the integer charge
  from the file `.CHRG` and the number of unpaired electrons from `.UHF` in the working
  directory (`-chrg` would be parsed as `-c`).
- Options (from the binary's string table): `-c <file>` coordinates (xyz, also TURBOMOLE
  `coord`), `-grad` (numerical gradient, file `gradient`), `-hess`, `-raman`, `-molden`
  (writes `molden.input` with the q-vSZP basis and MOs), `-eeq` (EEQ-BC charges only),
  `-tel <K>` (Fermi smearing, default 0 K), `-maxiter`, `-diisgen`, `-symthr <thr>` (symmetry
  threshold of the numerical gradient; `-symthr 0` switches symmetry off, which is needed
  because the symmetry code aborts for some point groups, e.g. D6h benzene), `-tmocc`,
  `-wrwbo`, `-desy`, `-fit`.
- Outputs: `energy` (total energy, 14 digits), `gradient` (TURBOMOLE format), `gxtbrestart`
  (read on the next run if present: starts the SCF from the old density, §4.17; converged
  energies agree to ~1e-9 Eh, the gap printout to ~1e-7 eV), `.data`. The printout has the
  Mayer bond orders ("largest Wiberg (=(PS)*(SP)) bond orders for each atom", 3 decimals, pairs
  above 0.05; open shells 2 Σ_σ (P^σ S)(P^σ S)) and the gap ("gap (eV)", open shell
  "alpha->alpha": the alpha channel's, with 27.212 eV/Eh in single precision); no quadrupole.
  `-wrwbo` writes no file.
- The gradient is **numerical**: central differences with a step of 0.0075 bohr (constants
  in `numgrad_`), optionally symmetry-reduced, for every atom but the **first, whose
  gradient is minus the sum of the others** (translational invariance). Its error is the
  h² truncation (up to 2.2e-5 Eh/bohr in the reference set) plus SCF noise of up to 2e-5
  Eh/bohr with the default thresholds: the energy is not stationary in the density
  (§4.16), so its error is first order in the density error. With the SCF thresholds
  tightened through gdb (§5, `gradient_tight`) the program's central differences equal
  ours to ≤ 3e-11 Eh/bohr (§4.18).

### 1.2 Conventions [C]

- **Ångström to bohr**: coordinates are divided by `0.52917726` *as a single-precision
  literal* widened to double (0.5291772484779358), i.e. 1 Å = 1.889725990443248 bohr. The
  list-directed read occasionally rounds a decimal 1 ulp away from Rust's parser (max 1.1e-16
  bohr in the reference set).
- Atomic units (bohr, Hartree) elsewhere; the D3 van der Waals pair radii used by EEQ-BC are
  kept in Ångström (§4.1).
- Several compiled-in element tables were written as default-real (single-precision)
  literals and are therefore float32-rounded doubles (e.g. the EEQ-BC average CNs,
  `com::cnav`). `crates/gxtb/src/data.rs` stores the program's exact bit patterns.
- Basis: the program stores primitives per **Cartesian** AO (atom by atom, shell by shell;
  d = 6, f = 10 Cartesian components; `ncao` Cartesian vs `nsao` spherical AOs printed) with
  coefficients `(c0 + c1 q_eff) * N_prim` where N_prim normalizes the x^l primitive; the
  contracted function is *not* normalized in that array. The integral code divides by the
  contraction norms (`intcom::norm`, e.g. 1.1822612154 for the O s shell of water) and all
  matrices are in **normalized spherical AOs** [C: S, H0, ACP reproduced element by element].
- **AO order of the program's matrices** [C]: atom by atom, shell by shell (basis order),
  p = (x, y, z), d = (x²−y², z², xy, xz, yz) (axtb offsets m = +2, 0, −2, +1, −1). Matrices
  are stored packed (lower triangle, row by row). `molden.input` writes d/f Cartesian.
- **Primitive screening** [C]: pairs with a_i a_j R²/(a_i + a_j) above 25 are skipped in the
  SCF-basis overlap (and its multipoles, presumably), above 20 in the H0 (`sintm_diat_`) and
  ACP (`calcvacp_`) overlaps. Atom pairs beyond R² = 1600 bohr² are skipped in H0/ACP.
- **Self distance** [C]: the program's packed distance array has 1e-12 bohr on the diagonal and
  the second-order and MFX kernels evaluate same-atom elements with R = 1e-12 (changes onsite
  values by ~1e-12 relative).
- Wavefunction: restricted closed shell when `.UHF` is absent/0, unrestricted otherwise
  ("UKS ?" in the output); Aufbau occupation at 0 K (P-Sec. 2.1).

### 1.3 Total energy as printed

```
ES1 (charge SIE)        first-order term E(1)                         [C]
ES2+3                   E(2) + E(3) + E(4)                            [C]
ES multipole            anisotropic electrostatics E(AES)             [C]
ES total                = ES1 + ES2+3 + ES multipole
Ex (Mulliken)           Mulliken-approximated range-separated exchange [C]
Espinpol                spin polarization (open shell only)           [C]
electronic              Tr((H0 + ACP) P) + ES total + Ex + E_OFX + Espinpol  [C]
atomic core increments  sum of increments                              [C]
dispersion              revD4 two-body part only (SCF charges)         [C]
nuclear repulsion       semi-classical repulsion                      [C]
total                   electronic + increments + dispersion + E_ATM + repulsion [C]
```

The revD4 three-body (ATM) energy E_ATM is part of the total but **not printed**: "total"
minus the printed terms is E_ATM (e.g. 1.6e-4 Eh for glycine). Full-precision values of the
two parts are `disp2`, `disp3` in `tests/reference/*/kernels.json`.

The onsite exchange correction E_OFX is not printed; it is the difference
electronic − Tr((H0+ACP)P) − ES total − Ex − Espinpol (checked with full-precision values read
from the program for 7 molecules, agreement 1e-8 = print precision). Full-precision values of
ES1, ES2+3, Ex and E_OFX of the final SCF iteration are in `tests/reference/*/kernels.json`.

P-Eq. 6 decomposes the energy as
E = E^incr + E^rep + E^EHT + E^ACP + E^(1) + E^(2) + E^AES + E^spin + E^(3) + E^(4)
+ E^lr,MFX + E^OFX + E^revD4.

### 1.4 Routine map [D]

Routine → term, from symbol names and from which arrays each routine reads (§3):

| Routine | Term |
|---|---|
| `rdpar_`, `rdbas_`, `rdpar_eeq_` | read `.gxtb`, `.basisq`, `.eeq` |
| `eeq_bc_` (+ `ncoord_eeq_`, `ncoord_basq_`) | EEQ-BC charges, basis CN |
| `setupbas_`, `setupbas0_` | q-vSZP basis (prints the "q-vSZP AO setup" table) |
| `modbas_` | modified q-vSZP basis for H0 (`gp3_pln`, `gp3_kqvszp`) |
| `sintm_diat_`, `sintm2_diat_`, `diat_scale_matrix_`, `tblite_integral_diat_trafo_*` | diatomic-frame scaled overlap for H0 (`sintm_diat_` is the one used by `scf_`) |
| `rfactor_` | H0 distance polynomial Π (`gp3_poly`, `com::rcov`) |
| `calcvacp_`, `setupcbas0_`, `setupcbas_`, `set_acp_prim_`, `csint_` | atomic correction potentials |
| `set1espot_`, `electro_` | first-order term |
| `setgab_lrao_`, `setespot_` | second-order kernel γ(2) and long-range MFX kernel |
| `set3espot_` | third-order term |
| `xtb_aespot_*`, `addanisotropich1_`, `count_dpint_`, `count_qpint_` | anisotropic electrostatics |
| `get_spin_constants_`, `setospot_`, `energy_os_`, `spin_contamination_` | spin polarization |
| `setfock_mat_`, `setfock_mat4_` | Fock build incl. Mulliken-approximated exchange |
| `setfock_indo_` | onsite exchange correction (OFX) |
| `set_ref_pop_`, `shellocc_ref_`, `guess_psh_`, `guess_p_` | reference shell populations, initial guess |
| `tblite_disp_d4_mp_new_d4_dispersion_` (+ `dftd4_model_mp_new_d4_model_` with `ref = 3`) | revD4 model setup: damping parameters, gxtb reference set (§4.15) |
| `tblite_disp_d4_mp_update_` (+ `dftd4_ncoord_mp_ncoord_derf_`) | revD4 CN and damped pair factors |
| `dftd4_model_mp_weight_references_pair_` | pair-specific (D4S) Gaussian weights with the tanh charge scaling (inlined `new_zeta`) |
| `tblite_disp_d4_mp_get_dispersion_nonsc_` (+ `dftd4_damping_atm_mp_get_atm_dispersion_`) | ATM three-body energy, once before the SCF |
| `tblite_disp_d4_mp_get_energy_`, `get_potential_` | two-body energy and dE/dq every SCF iteration |
| `xtb_aespot_mp_setvsdq_`, `addanisotropich1_` | AES potential (charge-dipole, charge-quadrupole, dipole-dipole only) and its Fock contribution |
| `guess_psh_`, `guess_p_` | start density (§4.17) |
| `diisfock_` | Fock mixing: damping, transition step, Pulay mixing (§4.17) |
| `solve_` (+ `occu_`, `dmat_`) | diagonalization (DSYGVX/DSYEVD), Aufbau occupation, density |
| `repuls_` | repulsion |
| `scf_` (81 kB) | driver; contains inline the g-xTB CN, H0 diagonal, increments |
| `numgrad_` | numerical gradient |
| `gp3_interpol_`, `checkpar_` | parameter interpolation / check (fit mode; see §3.5) |

---

## 2. Coordination numbers

| CN | Formula | Used by | Status |
|---|---|---|---|
| EEQ-BC CN | Σ_B ½[1+erf(−2 (R−Rc)/Rc^0.75)], Rc = (rcov2_A+rcov2_B)/2 (eeq col 4) | EEQ-BC (χ, width) | [C] |
| EEQ-BC local charge | Σ_B (EN_B−EN_A)/3.98 · ½[1+erf(−2 (R−Rc)/Rc^0.75)] + Q/N | EEQ-BC (χ, η) | [C] |
| basis CN | Σ_B ½[1+erf(−3.75 (R−Rc)/Rc)], Rc = R_A+R_B (`ncoord_basq$RCOV`, bohr) | q-vSZP q_eff | [C] |
| g-xTB CN | Σ_B ½[1+erf(glob(11) (R−Rc)/Rc)], Rc = (rad_A+rad_B)/2, glob(11) = −1.8217544049 | H0 levels (CN), repulsion, second-order Hubbard parameters, ACP weights (all √CN) | [C] (E_rep, H0, ACP, γ2 reproduced) |
| revD4 CN | Σ_B ½[1+erf(−7.5 (R−Rc)/Rc)], Rc = R_A+R_B (dftd4 3.x `covalent_rad_d3`, bohr), no electronegativity factor, pairs with R > 30 bohr skipped (`dftd4_ncoord_mp_ncoord_derf_`) | revD4 weights | [C] |
| D3-type CN | `ncoord_d3erf_` | printed "CN" column of the final table (after the SCF) | [D] |

All sums run over all pairs B ≠ A without cutoff. In `scf_` the g-xTB CN array is used raw
for the H0 diagonal and then **replaced by √CN** before it is passed to `setgab_lrao_` and
`repuls_` [C].

---

## 3. Parameter files

### 3.1 `gxtb_parameters` (`~/.gxtb`) [D: layout from `rdpar_`]

Two global lines of 10 numbers (`gp3_glob(1:20)`), then one 10-line block per element:

```
Z
L1: rep(1:5) rad g2cn poly e1cn atspin                         (10 numbers)
L2: lev(1:4)         L3: lev_cn(1:4)      L4: pln(1:4)
L5: fam(1:4)         L6: gam2(1:4)        L7: gam1(1:4)         (4 numbers each, per shell)
L8: h0k(1:2) kqvszp(1:3) at rep(6) acpcn                        (8 numbers)
L9: acp(1,1:4) acp(2,1:4)                                       (8 numbers)
```

Per-shell columns follow the element's shells in q-vSZP basis order (`basisq`), e.g.
O: s, p; S: s, p, d; Fe: s, p, d; Ce: s, p, d, f. Unused columns are zero.

**Elements present: Z = 1–58, 70–89, 92 (79 elements).** Z = 59–69, 90, 91, 93–103 are
absent in the public file; the program then runs with their parameters zero (verified
with gdb for Z = 59–69, 90, 91, 93–103: `gp3_lev` and `gp3_rep` stay 0; the SCF of GdCl3 fails in the first diagonalization).
`gp3_interpol_` (interpolation over Z) is not applied in normal runs. So v1.1 is usable for
79 elements only, despite the preprint's Z = 1–103.

**Global parameters** (k = index in `gp3_glob`, file line/column 0-based):

| k | file | value | read by | role | status |
|---|---|---|---|---|---|
| 1 | L1 C0 | 0.7188463521 | `scf_` | K_s, H0 off-diagonal shell factor for s (`scf_$KLL(0)`) | [C] |
| 2 | L1 C1 | 0.8284173386 | `scf_`, `sintm2_diat_` | K_p (`scf_$KLL(1)`) | [C] |
| 3 | L1 C2 | 1.1995319600 | `scf_` | F_p, H0 distance-polynomial factor for p (`scf_$RFL(1)`) | [C] |
| 4 | L1 C3 | 0.2599875891 | `repuls_` | k^pen,1 for Z ≥ 3 | [C] |
| 5 | L1 C4 | 0.0788775224 | `setgab_lrao_` | G_s of the MFX kernel (`GLL1(1)`) | [C] |
| 6 | L1 C5 | 1.7995847408 | `setgab_lrao_` | G_p = G_d = G_f of the MFX kernel | [C] |
| 7 | L1 C6 | −0.5959929766 | `setgab_lrao_` | k^MFX,1 | [C] |
| 8 | L1 C7 | 0.2140456651 | `setgab_lrao_` | k^MFX,2 (times R⁰_AB in Å) | [C] |
| 9 | L1 C8 | −0.3191336623 | `repuls_` | k^pen,1 for H, He | [C] |
| 10 | L1 C9 | 1.2154627292 | `scf_` → `new_d4_dispersion_` | revD4 BJ damping a1 (a2 = 0) | [C] |
| 11 | L2 C0 | −1.8217544049 | `scf_`, `ncoord_scf_` | g-xTB CN steepness | [C] |
| 12 | L2 C1 | 0.3300126723 | `setgab_lrao_` | k^(2),x of the second-order kernel | [C] |
| 13 | L2 C2 | −0.0276743359 | `repuls_` | k^pen,2 | [C] |
| 14 | L2 C3 | 0.1057773939 | `repuls_` | k^pen,3 | [C] |
| 15 | L2 C4 | 0.1682969331 | `repuls_` | k^pen,4 | [C] |
| 16 | L2 C5 | 0.4667106684 | `xtb_aespot_mmomgaberf_` | k3 of the AES charge–dipole kernel | [C] |
| 17 | L2 C6 | 0.2287119112 | `xtb_aespot_mmomgaberf_` | k5 of the AES kernels (q–Q, d–d) | [C] |
| 18 | L2 C7 | 0.0522538831 | `xtb_aespot_mmomgaberf_` | k7 of the AES dipole–quadrupole kernel | [C] |
| 19 | L2 C8 | 0.2347047181 | `setgab_lrao_` | ω of the MFX kernel (`omega_fx` is 0) | [C] |
| 20 | L2 C9 | 0.3042947280 | `scf_` → `new_d4_dispersion_` | revD4 s8 | [C] |

The MFX admixture α is `alpha_fx` if > 0, else the constant **0.15** (ωB97M-V's
short-range exact exchange); `alpha_fx` is 0 in normal runs [C].

**Element block** (`ElementParams` in `crates/gxtb/src/params.rs`):

| file | array | role | status |
|---|---|---|---|
| L1 C0 | rep(1) | repulsion α⁰_A | [C] |
| L1 C1 | rep(2) | repulsion R⁰_A (R⁰_AB = √(R⁰_A R⁰_B)) | [C] |
| L1 C2 | rep(3) | repulsion Z^eff,0_A | [C] |
| L1 C3 | rep(4) | repulsion k^CN_A (α_A ∝ 1 + k √CN) | [C] |
| L1 C4 | rep(5) | repulsion k^q_A (linear charge term) | [C] |
| L8 C6 | rep(6) | repulsion k^q2_A (quadratic, enters with − sign) | [C] |
| L1 C5 | rad | g-xTB CN radius (bohr); also read by H0 code in `scf_` | [C] (CN) |
| L1 C6 | g2cn | k^(2),U_A: U_l = gam2_l · gam_Z · (1 + g2cn √CN) | [C] |
| L1 C7 | poly | k^shp_A of the H0 distance polynomial | [C] |
| L1 C8 | e1cn | k^(1),CN_A of the first-order chemical potential (P-Eq. 27), μ_l = gam1_l (1 + e1cn √CN) | [C] |
| L1 C9 | atspin | k^W_A spin scaling (P-Eq. 35); affects only Espinpol, open shell only | [C] |
| L2 | lev(l) | H0 level: H_ii = −(lev_l − lev_cn_l · CN_A) | [C] |
| L3 | lev_cn(l) | CN dependence of the level (raw g-xTB CN) | [C] |
| L4 | pln(l) | exponent scaling of the H0 basis | [C] |
| L5 | fam(l) | U^x_l = fam_l · gam_Z (MFX Hubbard parameter) | [C] |
| L6 | gam2(l) | second-order Hubbard parameter, scaled by the element hardness `com::gam` | [C] |
| L7 | gam1(l) | μ^(1),0_l first-order chemical potential (P-Eq. 27) | [C] |
| L8 C0 | h0k(1) | σ scaling: k_σ = 2/(1/h0k1_A + 1/h0k1_B) | [C] |
| L8 C1 | h0k(2) | π scaling: k_π = 2/(1/h0k2_A + 1/h0k2_B) (δ: constant 5) | [C] |
| L8 C2–C4 | kqvszp(1:3) | weights of the three q_eff terms in the H0 basis | [C] |
| L8 C5 | at | atomic increment ΔE^incr_A (Eh) | [C] |
| L8 C7 | acpcn | ACP weight factor 1 + acpcn √CN | [C] |
| L9 C0–3 | acp(1, l) | ACP weight c_{A,l}, l = s, p, d, f (signed, linear) | [C] |
| L9 C4–7 | acp(2, l) | ACP projector exponent ξ_{A,l} | [C] |

Every element has all four ACP projectors in the file (the program builds 16 spherical
projector functions per atom: water has `acpsao 48`).

### 3.2 `basisq` (`~/.basisq`) [C]

Blocks separated by `*` lines, one per element 1–103:

```
Z  k_cn  k_q2  k_qcn
nprim l            (l = s, p, d, f)
alpha  c0  c1      (nprim lines)
...
```

- Header: q_eff = q − k_q2·q² + k_cn·√CN_basis + k_qcn·q·CN_basis (§4.2). The program's
  printout calls this "q+aq^2+bCN^0.5+cqCN" with a = −k_q2.
- Contraction: c_i = c0_i + c1_i·q_eff for primitive exp(−α_i r²).
- The file is already the g-xTB variant: no polarization shells for H, He, B–Ne (P-Sec. 2.1);
  third-row and heavier main-group elements carry d shells (e.g. Cl, S), transition metals
  s, p, d. Up to 8 primitives per shell in this file.

### 3.3 `eeq` (`~/.eeq`) [C]

103 lines of 10 numbers (the last two are unused zeros), then a date line
("Sun Dec 15 12:18:58 CET 2024"). Column → `eeqcom` array → multicharge `eeqbc2025` name:

| col | array | multicharge | meaning |
|---|---|---|---|
| 0 | eeq_xi | eeqbc_chi | electronegativity χ |
| 1 | eeq_gam2 | eeqbc_eta | hardness η |
| 2 | eeq_alp | eeqbc_rad | Gaussian charge width |
| 3 | eeq_cn1 | eeqbc_kcnchi | CN scaling of χ |
| 4 | eeq_rad | 2 × eeqbc_cov_radii | CN radius (bohr); Rc = (col4_A + col4_B)/2 |
| 5 | eeq_gamq | eeqbc_kqeta | local-charge scaling of η |
| 6 | eeq_cn2 | eeqbc_kqchi | local-charge scaling of χ |
| 7 | eeq_boca | eeqbc_cap | bond capacitance |

All 8 × 103 values are identical to multicharge's `eeqbc2025` (checked column by column). The
average CNs (`eeqbc_avg_cn`) are not in the file; the program has them compiled in
(`com::cnav`, float32-rounded).

### 3.4 Compiled-in tables [D]

Extracted bit-exactly by `crates/gxtb/scripts/extract_binary_tables.py` into
`crates/gxtb/src/data.rs`:

| symbol | size | use |
|---|---|---|
| `com::pen` | 103 | Pauling EN with multicharge's actinide values; EEQ-BC local charge [C] |
| `com::cnav` | 103 | EEQ-BC average CN (float32-rounded) [C] |
| `data_vdwrad::vdwrad` | 5356 | D3 pair radii in Å (packed); EEQ-BC capacitance [C]; copied unscaled into `gp3_aesr0` as R⁰_AB (Å) of the AES damping (P-Eq. 33) and the MFX kernel (P-Eq. 42) [C] |
| `ncoord_basq$RCOV` | 118 | basis-CN radii (bohr) [C] |
| `com::rcov` | 118 | radii (bohr) of the H0 distance polynomial, Rc_AB = (R_A + R_B)/2 [C] |
| `parcom::gp3_gam3` | 103 | Γ_A of the third-order term (compiled in, float32) [C] |
| `com::zeff_shell` | 103×4 int | Aufbau valence electrons per shell Z_l; first-order Δρ⁰_l = Z_l − n^ref_l [C] |
| static table in `get_spin_constants_` | 103×10 | reference spin constants W⁰ (float32), order ss, sp, sd, sf, pp, pd, pf, dd, df, ff (`SPIN_W0`) [C] |
| `com::gam` | 103 | element hardness: factor of both Hubbard parameter sets (U^(2), U^x) [C] |
| `bascom::bas_pol` | 103×4 int | polarization-shell flags (d of main-group elements, p of Ce, ...): ξ = 2 in the MFX averaging [C] |
| `shellocc_ref_` (function) | 103×4 | reference shell occupations (tabulated via gdb into `src/data_runtime.rs`); Mulliken shell charge = n^ref_l − population [C] |
| `com::ipse`, `com::gpse` | 103 int | period / group; period selects k^pen,1 in `repuls_` [C] |
| `com::gmunu` | 103×9 | onsite exchange integrals of OFX per l pair (sp, pp, sd, pd, dd, sf, pf, df, ff); `MAIN__` multiplies them by 0.15 at start-up [C] |
| `com::k3_pair` | 103×103 | revD4 pair weighting factors wf(Z_A, Z_B) of the D4S Gaussian CN weights; symmetric, identical to dftd4 4.x `weight_factors` (Z ≤ 103) [C] |
| `com::valel`, `com::ipat`, `com::eaat` | | read only by `MAIN__`/fit code or role unknown [U] |

The program also embeds its own dftd4 (a 3.x-era version with g-xTB additions). Its tables
`alphaiw`, `ascale`, `hcount`, `secaiw`, `sscale`, `refn`, `refsys` and `r4_over_r2` are
bit-identical to dftd4 4.3.1's (`crates/d4/src/data.rs`; `extract_binary_tables.py` checks
this) and are taken from `axtb_d4`. Specific to the program, extracted into
`src/data.rs` [C]:

| symbol | Rust | use |
|---|---|---|
| `dftd4_reference_mp_refcovcn_` | `D4_REFCOVCN` | reference CNs of the Gaussian weights; recomputed for the revD4 CN, differ from dftd4 4.3.1 |
| `dftd4_reference_mp_refcn_` | `D4_REFCN` | reference CNs counted (nint, capped at 19) for the number of Gaussians per reference |
| `dftd4_reference_mp_gxtbq_` | `D4_REFQ` | reference charges |
| `dftd4_reference_mp_gxtbh_` | `D4_REFH` | charges of the hydrogen-subtracted secondary systems |
| `dftd4_data_tanh_mp_tanh_params_` | `D4_TANH` | 4 per element (a, b, c, d) of the charge function a + b tanh(c q + d) (b = −a for the elements checked) |
| `dftd4_data_covrad_mp_covalent_rad_d3_` | `D4_RCOV` | CN radii: 4/3 Pyykkö radii with 1 Å = 1/0.52917726 bohr (not mctc-lib's CODATA value) |

(`chemical_hardness` of H is 0.2 instead of dftd4's 0.47259288; revD4 does not use the
hardness.)

### 3.5 Hard-coded constants found so far [D]

EEQ-BC: k_cn = 2.0, norm_exp = 0.75, k_bc = 0.6, k_cnrad = 0.14, EN norm 3.98 [C].
Basis CN: 3.75 [C]. Repulsion: exponent 3/2 (as √x·x), factor 2 in α_AB [C].
`setgab_lrao_`: 0.15, 2.0, −1.15, −0.5, −1.84, 0.5, 3.2, 1.39.
`set1espot_`/`electro_`: 0.976, 1.024, ±0.666666666, 0.5, 0.25, 2/√π, 3.0.
`set3espot_`: 1/3. `mmomgaberf_`: 0.5, 200, −0.003, 1.0. `sintm_diat_`, `calcvacp_`: R² cutoff
1600 bohr² (40 bohr). `numgrad_`: step 0.0075 bohr.
revD4 (`new_d4_dispersion_` arguments and `tblite_disp_d4` defaults): s6 = 1, s9 = 1.75, a2 = 0,
α_ATM = 16, `ga` = `gc` = 0 (unused by the tanh scaling), wf = 6 (unused: pair factors), cutoffs
CN 30, two-body 60, three-body 20 bohr (hard) [C].
SCF (`scf_`): damping α = 0.1 (single-precision literal, 0.10000000149), transition iteration
3 (open shell 4), first Pulay iteration 4 (5), at most 7 (8) stored Fock matrices,
convergence |ΔE| < 1e-8 Eh and ΔP_rms < 2e-5 (open shell 4e-7 Eh, 5e-5), at most 1000
iterations (`-maxiter`), `-diisgen` overrides the number of stored matrices; level shift
`bascom::lshift` = 0 [C].

---

## 4. Terms

### 4.1 EEQ-BC charges [C]

Model of T. Froitzheim et al., J. Chem. Phys. 162, 214109 (2025); identical to multicharge
`eeqbc2025`. Distances R_AB in bohr; Rc_AB = (col4_A + col4_B)/2; EN_A = `pen`_A / 3.98.

- count(R) = ½[1 + erf(−2 (R − Rc)/Rc^0.75)];  CN_A = Σ_{B≠A} count(R_AB).
- q^loc_A = Σ_{B≠A} (EN_B − EN_A) count(R_AB) + Q/N.
- c_AB = √(cap_A cap_B) · ½[1 + erf(−0.6 (R_AB − R^vdW_AB)/R^vdW_AB)] with R^vdW_AB the D3 pair
  radius **in Ångström** (R_AB stays in bohr; same in multicharge, which converts the radii
  back to Å).
- Maxwell matrix C (N+1)×(N+1): C_AB = −c_AB, C_AA = Σ_B c_AB, C_{N+1,N+1} = 1.
- x_A = −χ_A + kcnchi_A CN_A + kqchi_A q^loc_A, x_{N+1} = Q; b = C x.
- w_A = rad_A (1 − 0.14 CN_A / CNavg_A^0.75).
- A_AB = erf(R_AB/√(w_A² + w_B²))/R_AB · C_AB (A ≠ B);
  A_AA = (η_A + kqeta_A q^loc_A + √(2/π)/w_A) C_AA + 1; A_{A,N+1} = A_{N+1,A} = 1, A_{N+1,N+1} = 0.
- (q, λ) = A⁻¹ b (the program uses DSYSV).

The program prints CN, q^loc ("q_CN"), the basis CN ("CN(basis)") and q, and the EEQ dipole
Σ q_A R_A. Role in g-xTB: q-vSZP effective charge (§4.2), repulsion charges (§4.5); the
revD4 charges are the SCF Mulliken charges (ATM: q = 0), not EEQ-BC [C].

Reproduction: charges 8e-16, printed CN/q_CN/dipole within print precision, 14 molecules.

### 4.2 q-vSZP basis [C]

M. Müller, A. Hansen, S. Grimme, J. Chem. Phys. 158, 014103 (2023), g-xTB variant (P-Eqs. 9–11).

- CN_basis,A = Σ_{B≠A} ½[1 + erf(−3.75 (R_AB − Rc)/Rc)], Rc = R_A + R_B (`ncoord_basq$RCOV`).
- q_eff,A = q_A − k_q2 q_A² + k_cn √CN_A + k_qcn q_A CN_A (q = EEQ-BC). Corresponds to P-Eq. 11 with
  k⁰ = 1 for the SCF basis.
- Shell of atom A: φ = Σ_i (c0_i + c1_i q_eff,A) N_l(α_i) x^l exp(−α_i r²), normalized as a whole
  (N_l normalizes the x^l primitive: (2α/π)^{3/4} (4α)^{l/2} / √((2l−1)!!)).

`QvszpBasis::new` builds it; `to_axtb()` gives an `axtb::basis::Basis` (each atom its own
species). Reproduction: basis CN 4e-16, q_eff 7e-16, internal primitive coefficients 1.3e-15
(relative), molden-normalized coefficients 5e-15, 14 molecules incl. S, Cl, Si, P, I, Ni, Cu.

### 4.3 Atomic increments [C]

E^incr = Σ_A at_A (L8 C5). Zero for H and He (P-Eq. 13). Reproduced exactly.

### 4.4 Repulsion [C]

Decoded from `repuls_` and reproduced to 4e-16 Eh (14 molecules):

E^rep = Σ_{A<B} Z_A Z_B exp(−α_AB (R_AB + R⁰_AB)^{3/2}) ·
[1 + (k1_A + k1_B)/(2R_AB) + k2/R_AB² + k3/R_AB³ + k4/R_AB⁴]

- Z_A = rep3_A (1 − rep5_A q_A − rep6_A q_A²), q = EEQ-BC charges.
- α_A = rep1_A (1 + rep4_A √CN_A), g-xTB CN; **α_AB = 2 α_A α_B/(α_A + α_B)** (P-Eq. 17 omits the 2).
- R⁰_AB = √(rep2_A rep2_B) (P-Eq. 16 writes R⁰_AB).
- k1 = glob(9) for period 1 (H, He), glob(4) otherwise (`com::ipse`); k2, k3, k4 = glob(13), glob(14), glob(15).
- k_exp = 3/2 is hard-coded.

### 4.5 Extended Hückel Hamiltonian [C]

Decoded from `scf_`, `modbas_`, `sintm_diat_` and reproduced element by element (14
molecules incl. Si, P, S, Cl, I, Ni, Cu d shells): levels 1e-16, scaled overlap 1e-15, H0
3e-16 (`crates/gxtb/src/h0.rs`, test `h0_matches_program`).

- **H0 basis** (`modbas_`): per atom the q-vSZP shells with exponents α'_i = pln_l α_i and
  coefficients c0_i + c1_i q'_eff, normalized, where
  q'_eff = k̃⁰ (q − k_q2 q²) + k̃² k_cn √CN_basis + k̃³ k_qcn q CN_basis, i.e. the three terms of
  the SCF-basis effective charge (§4.2) reweighted by `kqvszp` = (k̃⁰, k̃², k̃³).
- **Scaled overlap S̃** (`sintm_diat_`, primitive cutoff 20): for A ≠ B the overlap block of the
  H0 basis is rotated to the diatomic frame (bond along z) and its σ (m = 0), π (|m| = 1) and
  δ (|m| = 2) parts are scaled by k_σ = 2/(1/h0k1_A + 1/h0k1_B), k_π = 2/(1/h0k2_A + 1/h0k2_B)
  and k_δ = 5 (constant); blocks with f functions stay unscaled (program's
  `scale_diatomic_frame`; no f shell in the reference set). Implemented with bond-frame
  projectors, which is equivalent to the program's rotate–scale–rotate back.
- **Levels**: H_ii = −(lev_l − lev_cn_l · CN_A), g-xTB CN (not √CN).
- **Off-diagonal**: H_ij = ¼ (H_ii + H_jj)(K_li + K_lj) π_i π_j S̃_ij for A ≠ B, zero within an
  atom, with π_i = 1 + poly_A F_li R_AB/Rc_AB, Rc_AB = (rcov_A + rcov_B)/2 (`com::rcov`),
  K = (glob(1), glob(2), 1.4, 2.85) and F = (1, glob(3), 2, 2) for l = s, p, d, f.

### 4.6 Atomic correction potentials [C]

Reproduced to 2e-15 (`acp.rs`, test `acp_matches_program`). Every atom carries normalized
single-Gaussian projector shells for l = s, p, d, f with exponent acp(2,l); with S_μr the
overlap between SCF q-vSZP AOs and projector functions (primitive cutoff 20, pairs within
R² ≤ 1600 bohr²),
H^ACP_μν = Σ_r w_r S_μr S_νr,  w_{A,l} = acp(1,l) · (1 + acpcn_A √CN_A)  (g-xTB CN).
The weight is signed and linear (not squared as P-Eqs. 22–24 suggest) and uses √CN without
CN^avg. H0 + H^ACP is the one-electron matrix of the SCF.

### 4.7 First-order term [C]

Reproduced at the program's converged density to 9e-15 Eh (`es.rs`, 14 molecules):
E(1) = Σ_l μ_l q_l f(q_A) − Σ_{l∈A} Σ_{l'∈B≠A} Δρ⁰_l' γ2_ll' q_l,
μ_l = gam1_l (1 + e1cn_A √CN_A), f(q) = 1 + 0.012 [erf(q − 2/3) + erf(q + 2/3)]
(k^(1),dis = 0.012, k^(1),x = 1, k^(1),s = 2/3 hard-coded), Δρ⁰_l = Z_l − n^ref_l
(`com::zeff_shell` minus the reference occupation), shell charges q_l and atomic charges
q_A = Σ_l q_l, γ2 the second-order kernel.

### 4.8 Second-order term [C]

Shell kernel (`setgab_lrao_`, reproduced to 2e-16, `kernels.rs`):
γ2_ij = 1/(R + e^{−glob(12) R}/η),  η = 2/(1/U_i + 1/U_j),  U_l = gam2_l · gam_Z · (1 + g2cn_Z √CN),
with R = 1e-12 within an atom. Shell charges q_l = n^ref_l − Mulliken population
(reproduced to 5e-15). E(2) = ½ Σ q_i γ2_ij q_j; with E(3) and E(4) it reproduces "ES2+3" to
5e-15 Eh at the program's density.

### 4.9 Anisotropic electrostatics [C]

Decoded from the program's `xtb_aespot` module (`mmompop`, `mmomgaberf`, `aniso_electro`) and
reproduced at its converged density (`aes.rs`, 14 molecules): kernels 5e-16 (relative),
atomic moments 1.2e-13, E_AES 1.1e-15 Eh.

- Atomic moments (GFN2-xTB style, total density, Mulliken partition, primitive cutoff 25):
  d_A = −Σ_{μ∈A,ν} P_μν ⟨μ|(r−R_A)|ν⟩, Q_A = −Σ_{μ∈A,ν} P_μν ⟨μ|3/2 (r−R_A)(r−R_A) − 1/2 |r−R_A|²|ν⟩
  (traceless; components xx, xy, yy, xz, yz, zz).
- Kernels for atom pairs (R in bohr, R⁰_AB the D3 pair radius in Å, zero beyond R = 200):
  g3 = glob(16)/2 · [1 − erf(−0.45 (R − R⁰))]/R³,
  g_n = k_n/2 · [1 − erf(−1.05 (R − R⁰))]/Rⁿ for n = 5, 7, 9 with k5 = glob(17), k7 = glob(18),
  k9 = −0.003 (the steepnesses 0.45 and 1.05 are arguments set in `scf_`).
- Energy (pairs A > B, r = R_B − R_A, atomic Mulliken charges q, no onsite terms):
  E = Σ [e_qd g3 + (e_qQ + e_dd) g5 − e_dQ g7 + ⅓ e_QQ g9] with
  e_qd = q_B (d_A·r) − q_A (d_B·r), e_qQ = q_B rᵀQ_A r + q_A rᵀQ_B r,
  e_dd = (d_A·d_B) r² − 3 (d_A·r)(d_B·r),
  e_dQ = 5(rᵀQ_A r)(d_B·r) − 2(rᵀQ_A d_B) r² − [5(d_A·r)(rᵀQ_B r) − 2(rᵀQ_B d_A) r²],
  e_QQ = 35(rᵀQ_A r)(rᵀQ_B r) − 20(Q_A r)·(Q_B r) r² + 3 Tr(Q_A Q_B) r⁴ (P-Eq. 32).
  The dipole–quadrupole and quadrupole–quadrupole parts are not in the Fock matrix
  (`setvsdq_` builds the potential of the q–d, q–Q and d–d parts only, §4.16) [C].

### 4.10 Spin polarization [C]

E^spin = ½ Σ_A Σ_{l,l'∈A} m_l m_l' k^W_A W⁰_{ll'}(Z), m_l = Mulliken shell magnetization
(α − β population), k^W_A = atspin, W⁰ a compiled-in float32 table (10 per element, order ss,
sp, sd, sf, pp, pd, pf, dd, df, ff). Reproduced to 1e-17 Eh at the program's density (O2
triplet, [CuCl4]²⁻ doublet).

### 4.11 Third-order term [C]

E(3) = ⅓ Σ_{l,l'} q_l q_l' q_{A(l)} Γ_{A(l)} τ_ll', Γ_A = `gp3_gam3` (compiled in; the same for
all shells of an atom), with the kernel built by `setgab_lrao_`: within an atom
τ = −½ γ2², between atoms τ = ½ R e^{−1.15 x}(3.2 − 1.84 x)/η with x = R/η (η as in γ2; the
literals replace the preprint's k^(3), k^(3),x). Reproduced: τ to 1e-14, energy (with E(2),
E(4)) to 5e-15 Eh.

### 4.12 Fourth-order term [C]

E(4) = Σ_l (k4/4) q_l⁴ with k4 = 0.021 (a constant per-shell array) — shell-resolved, unlike
P-Eq. 40's atomic form.

### 4.13 Mulliken-approximated range-separated Fock exchange [C]

Kernel (`setgab_lrao_`, reproduced to 6e-17) per shell pair, expanded to all AO pairs:
γ^x_ij = (α + (1−α) erf(ω R)) / (R + e^{−(k^x1 + k^x2 R⁰_AB) R}/(η̃_ij g_ij)),
α = 0.15, ω = glob(19), k^x1 = glob(7), k^x2 = glob(8), R⁰_AB = D3 pair radius in Å, R = 1e-12
within an atom; η̃ = 2^{ξ−1}(U^x_i U^x_j)^{ξ/2}/(U^x_i + U^x_j)^{ξ−1} (`uwe_av_`) with
U^x_l = fam_l · gam_Z and ξ = 2 if either shell is a polarization shell (`bas_pol`), else 1;
g_ij = 1.39 for the same shell of the same atom, else √(G_li G_lj) with G = (glob(5), glob(6),
glob(6), glob(6)).
Energy (`setfock_mat_`, reproduced at the program's converged density to 1e-14 Eh, closed and
open shell):
E^x = −¼ Σ_σ Σ_μν γ^x_μν [(P^σS)_μν (SP^σ)_μν + P^σ_μν (SP^σS)_μν]
(= −⅛[…] with the total density for closed shells), printed as "Ex (Mulliken)".

### 4.14 Onsite exchange correction (OFX) [C]

Reproduced at the program's density to 3e-16 Eh (`exchange.rs`, incl. d shells and UHF):
E^OFX = −Σ_σ Σ_A [Σ_{κ≠λ∈A} Λ_κλ (P̃^σ_κλ)² + Σ_{shells s∈A, l_s≥1} Σ_{κ,λ∈s}
((2l+1)δ_κλ − 1)/(2l+1) Λ_κλ P̃^σ_κκ P̃^σ_λλ],
P̃^σ = ½(P^σS + SP^σ) (closed shell: P^σ = P/2), Λ_κλ = 0.15 · gmunu(index(l_κ, l_λ), Z) with
the index sp 1, pp 2, sd 3, pd 4, dd 5, sf 6, pf 7, df 8, ff 9 (`setfock_indo_$IPOINT`); s–s
pairs have none. Not printed separately; part of "electronic".

### 4.15 revD4 dispersion [C]

Decoded from the program's embedded dftd4/tblite (`new_d4_model` with `ref = 3`,
`weight_references_pair`, `set_refalpha_gxtb`, `tblite_disp_d4`), implemented in `disp.rs` on
top of `axtb_d4`, reproduced to 3e-18 Eh (two-body, at the program's charges) and 5e-20 Eh
(ATM), 14 molecules. A D4S-type model (pair-specific Gaussian weights) with a tanh charge
scaling ("new atomic charge scaling function", P-Sec. 2.10) and one BJ parameter fewer.

- **CN**: revD4 CN of §2 (erf, steepness 7.5, no EN factor, R_A + R_B from `D4_RCOV`, 30 bohr).
- **Charge function** of element Z: n_Z(q) = a + b tanh(c q + d) (`D4_TANH`); the scaling
  is n_Z(q)/n_Z(q_ref) (`new_zeta`), no Z_eff offset.
- **Reference polarizabilities** (`set_refalpha_gxtb`): for reference r of element Z with
  secondary system s = refsys(r, Z) (an element number),
  α_r(iω) = max(0, ascale_r [αref_r(iω) − hcount_r · secaiw_s(iω) · (sscale_s ζ_r)]),
  ζ_r = n_s(h_r)/n_s(0) with h_r = `D4_REFH`; note the product order secaiw · (sscale · ζ).
- **Reference C6**: Casimir–Polder integration of α_r α_r' (3/π, dftd4's 23-point trapezoid
  grid), `axtb_d4` (`D4Model::with_reference_data`).
- **Weights of reference r of atom A in the pair AB** (`weight_references_pair`):
  w_r(A,B) = n_A(q_A)/n_A(qref_r) · Σ_{g=1..ngw_r} exp(−g · wf_AB (CN_A − CN_r)²) / N_A(B),
  N_A(B) the same sum over all references, wf_AB = `com::k3_pair`(Z_A, Z_B), CN_r =
  `D4_REFCOVCN`, qref_r = `D4_REFQ`, ngw_r = c(c+1)/2 with c the number of references sharing
  nint(`D4_REFCN`) (plus one for nint = 0). If the normalized weight is NaN/infinite, the
  reference with the largest CN_r gets 1 (dftd4's rule).
- **C6_AB** = Σ_rr' w_r(A,B) w_r'(B,A) C6ref_rr'. Since q enters only through n_A(q_A),
  C6_AB(q) = n_A(q_A) n_B(q_B) K_AB with a geometry-only K_AB (used by `RevD4`).
- **Two-body** (rational damping): E2 = −Σ_{A>B} C6_AB [s6/(R⁶ + R0⁶) + s8 rr_AB/(R⁸ + R0⁸)],
  rr_AB = 3 r4r2_A r4r2_B (dftd4 `sqrt_z_r4_over_r2`), R0 = a1 √rr_AB + a2, s6 = 1, s8 = glob(20),
  a1 = glob(10), a2 = 0, pairs within 60 bohr; q = the SCF Mulliken atomic charges (printed
  "dispersion"; evaluated every iteration, potential dE/dq_A enters the Fock matrix).
- **Three-body** (ATM, zero damping): dftd4's kernel with s9 = 1.75, α = 16, R0 as above,
  triples with all distances ≤ 20 bohr, C6 at q = 0; evaluated once before the SCF, not
  printed, part of the total (§1.3).

### 4.16 Fock matrices [C]

`fock.rs`; checked against the program's Fock matrices (built at `diisfock_` entry) at the
start density and the last iteration: 7e-15 for 13 molecules, 1.5e-9 for [CuCl4]²⁻ (below).
F^σ = ∂E/∂P^σ (closed shell ∂E/∂P with the total density), packed per spin in the program:

- H0 + ACP (both spins).
- Shell-charge potentials of E(1)–E(4) and the atomic potentials of revD4 (dE/dq_A) and of the
  AES charge part: F_μν += −½ (V_μ + V_ν) S_μν (both spins).
- AES (`setvsdq_` + `addanisotropich1_`, the potentials of xtb's GFN2 `setvsdq`):
  equivalent to differentiating only the charge–dipole (g3), charge–quadrupole and
  dipole–dipole (g5) parts of E_AES; the dipole–quadrupole (g7) and quadrupole–quadrupole
  (g9) energies have **no potential** (verified: the program's AES Fock contribution
  reproduced to 4e-17).
- Spin polarization: F^α += ½(w_μ + w_ν) S_μν, F^β −= the same, w_l = ∂E_spin/∂m_l.
- Mulliken exchange (`setfock_mat_`): exact derivative,
  F^σ = −¼ [(γ∘SP)S + S(γ∘PS) + γ∘(SPS) + S(γ∘P)S] (closed shell −⅛, total P) [4e-16].
- Onsite exchange (`setfock_indo_`): with the dual density P̃ and Y = ∂E/∂P̃ for the
  exchange-type pairs, F = ½(YS + SY) + D where the population part y_κ = ∂E/∂P̃_κκ enters as
  D_κκ = y_κ, D_κλ = (y_κ + y_λ) S_κλ (κ ≠ λ) — **twice the derivative off the diagonal** (the
  program omits the ½ of the symmetrization; reproduced to 4e-17).

Because of the last two items the program's SCF solution is not stationary for its energy
(important for analytic gradients). All other parts are exact derivatives (checked by
finite differences of our energy). `crates/gxtb` builds this Fock matrix with
`Conventions::Gxtb` and the exact derivative (full AES potentials, OFX population part with
the ½) with `Conventions::Axtb`, the default (§4.18).

Known deviation [U]: for the Cl–Cl d–d AO pairs at 8.5 bohr in [CuCl4]²⁻ the program's AES
Fock elements (~1e-8) differ from ours by up to 1e-9 although its atomic moments, overlap and
energies agree; independent of our primitive screening (25–1000), so the program apparently
screens or approximates its long-range d–d multipole integrals (`intcom::d_at`/`qint_at`,
built in `MAIN__`). Effect: orbital energies 9e-10 Eh, charges 2e-11, total energy < 1e-13 Eh.
The element survey (§5.1) shows the same for the farthest pairs of larger complexes: in
[Ca(H2O)6]²⁺ the H–H s–s elements between trans waters (11.27–11.58 bohr, S ≈ 1e-6) differ by
2.2e-8 at the start density and 1.6e-7 at convergence, smoothly decreasing with R, and nothing
below 10.3 bohr; the converged densities differ by 2.5e-7, the printed "Ex (Mulliken)" by
1.3e-8 Eh (6e-9–1.7e-8 for [Ca(H2O)6]²⁺, [Ba(H2O)6]²⁺, [La(H2O)6]³⁺, [Rh(NH3)6]³⁺), the
totals by ≤ 1e-11 Eh (at the program's density every term of ours equals the program's to
1e-14). Ruled out: the multipole-integral primitive screening (8–25), and xtb's `setdqlist`
pair lists (decoded: `scf_` calls `xtb_aespot_mp_setdqlist_` with 0.01 × 1e-9, keeping pairs
with |d|² or ‖Q‖²_F above 1e-11; applying them changes [CuCl4]²⁻ from 1.5e-9 to 9e-10 but leaves
the H–H elements, whose integrals pass the test, unchanged).

### 4.17 SCF [C]

Decoded from `scf_`, `guess_psh_`, `guess_p_`, `diisfock_`, `solve_`; `scf.rs` follows the
program iteration by iteration (energies agree to ~1e-14 Eh in every iteration).

- **Electrons**: Z_A = Σ_l `com::zeff_shell`; N = Σ Z_A − charge; N_β = (N − uhf)/2. Closed
  shell (one spin channel) when uhf = 0, else unrestricted (two channels).
- **Start density**: diagonal, P_μμ = f · n^ref_l (Z_A − q_A)/Z_A / (2l+1) with the reference
  shell occupations n^ref (`shellocc_ref_`), the EEQ-BC charges q_A, f = 1 (closed shell) or ½
  per spin. Not normalized to N exactly (n^ref are float32-valued: +1.7e-7 e for [CuCl4]²⁻).
- **Iteration k** = 1, 2, …: Fock matrix F_k from P_{k−1}; mixing → F̃_k; diagonalization of
  F̃_k (generalized eigenproblem with S); Aufbau occupation (lowest N_α, N_β orbitals,
  closed shell doubly occupied, 0 K); P_k; energy E_k; ΔP_rms = √(Σ_{σ, packed} (P_k − P_{k−1})² / npk),
  npk = n(n+1)/2.
- **Mixing** (`diisfock_`; F̃_0 = 0; closed-shell iteration numbers, open shell one later):
  iterations 1–2: F̃_k = F̃_{k−1} + α (F_k − F̃_{k−1}), α = 0.1 (single precision);
  iteration 3 (transition): F̃_k = 2 F̃_{k−1} + α (F_k − F̃_{k−1}) and the pair
  (F_k, r_k = F_k − F̃_{k−1}) is stored; iterations ≥ 4: store (F_k, r_k), keep the last 7
  (open shell 8), F̃_k = Σ_i c_i F_i with Σ c_i = 1 minimizing |Σ c_i r_i|² (dot products over
  the packed lower triangles; separately for each spin). Only the direction of F̃ matters
  (the scale of the damped matrices is not 1).
- **Convergence**: after iteration k with |E_k − E_{k−1}| < 1e-8 Eh and ΔP_rms < 2e-5 (open
  shell 4e-7 and 5e-5; E_0 = 0), the program does **one more iteration with the unmixed
  F_{k+1}** and reports E_{k+1}, its orbital energies (eigenvalues of F_{k+1}), the density
  P_{k+1} and its Mulliken charges. "N scf iterations" counts this last one.
- No level shift (`lshift` = 0; it would be switched off when |ΔE| < 1e-6), no Fermi smearing
  at the default 0 K; `fpcom_` (the commutator) is not used on this path.
- **Restart** (`gxtbrestart`) [C: caffeine, water, O2 triplet; same geometry and 0.02/0.1/0.3
  bohr rms displacements; energies ≤ 7e-12 Eh, identical iterations]: the file holds the packed
  spin densities and the MO coefficients of the last iteration. The next run starts from the
  densities as they are (no reorthonormalization in the new q-vSZP basis: the first two
  iterations' energies and ΔP_rms equal those of axtb's start from the raw densities to all
  printed digits), with F̃_0 = 0 and the damping of iterations 1–2 as usual, but **without the
  transition step**: iteration 3 (open shell 4) stores its pair and already takes the Pulay
  combination of the stored pairs (its own only, F̃_3 = F_3). With E_0 = 0 a restart at the same
  geometry takes 3 iterations (converged after 2, plus the unmixed one). Decoded by comparing
  every iteration of restarted runs; with the transition step, caffeine 0.02 bohr away took 11
  iterations where the program took 7. axtb: `Setup::scf_from`, `Guess::Previous`.
- **Finite electronic temperature** (`-tel <K>`; decoded from `solve_`, `occu_`, `fermismear_`,
  `scf_`) [C: 14 molecules + stretched N2 at 300/1000/3000/5000 K, `tests/etemp.rs`]: when
  T > 1e-6 K, `solve_` smears every spin channel that has occupied and virtual orbitals
  (closed shell: the alpha and the beta electrons on the one set of levels) after `occu_`'s
  Aufbau filling, with the **integer** electron count N_σ of the channel:
  - kB = xtb's 3.166808578545117e-6 Eh/K **rounded to single precision**
    (3.1668084830016596e-6, a `real(sp)` literal in `fermismear_`);
  - **only the lowest min(N_α + 10, n) eigenpairs are computed** (DSYGVX, il = 1, iu = N_α + 10,
    for both spins); the other eigenvalues are set to 100 Eh, so those levels stay empty at any
    temperature (at 0 K this has no effect);
  - the levels are those of the matrix the SCF diagonalizes, i.e. **the damped Fock matrices of
    iterations 1–3 with their scale** (F̃_1 = 0.1 F_1, …, §4.17 mixing): the first iterations
    smear with scaled orbital energies (water at 3000 K: iteration 1 is 0.06 Eh above the 0 K
    run's), so the SCF path differs from the 0 K one even where the converged occupations are
    integers;
  - Newton steps from μ = ½ (ε_N + ε_N+1): f = 1/(1 + eˣ), f' = eˣ/((kT (1 + eˣ))(1 + eˣ)) for
    x = (ε − μ)/kT < 50, else 0; μ += (N − Σf)/(Σf' + 1e-14); stop when |N − Σf| ≤ 1e-9 (the
    occupations are those of the previous μ) or after 200 steps (two SSE2 lanes, then the
    remainder);
  - −TS = kT Σ [f ln f + (1 − f) ln(1 − f)] over the levels with f > 1e-9 and 1 − f > 1e-9, per
    electron spin, added to the electronic energy of every iteration (`scf_`: the electronic
    energy is the sum of the terms plus −TS), so it is in the printed SCF energies, the
    convergence test and the total. Nothing about it is printed besides "Tel(K)".

  Reproduction: total energies (`energy` file, 14 decimals) ≤ 2e-11 Eh with identical SCF
  iterations for all 60 runs of the 14 reference molecules and stretched N2; Fe4 (45–941
  iterations, or none) is rounding-sensitive. Program-mode analytic gradients: the program's
  numerical gradients (tight SCF) are our central differences with its step (≤ 7e-11 Eh/bohr,
  [CuCl4]²⁻ 2e-9), the analytic gradient their Richardson extrapolation (≤ 2e-8); against
  5-point differences of the free energy at 5000 K ≤ 2.2e-9 (the program's 1e-9 electron
  threshold roughens its energy at that level). The response (`response.rs`, `fermi_gradient`)
  generalizes the Z-vector to fractional occupations: divided differences of n and n·ε between
  all orbital pairs, derivatives n' on degenerate pairs, the Fermi-level shift of every channel,
  and the levels the entropy leaves out (their occupations are not variational).
  The default conventions smear every level with the exact Fermi level and CODATA 2022 kB, so
  their gradient needs no response (≤ 9e-10 Eh/bohr against finite differences).

Reproduction (`Conventions::Gxtb`; tests `scf_matches_program`,
`extra_singlepoints_match_program`): total
energies of the 14 reference molecules to 4.6e-13 Eh with the same iteration counts, all
full-precision energy terms to 4e-12 Eh, Mulliken charges 1.7e-11, molden orbital energies
1.5e-14 Eh (9e-10 for [CuCl4]²⁻); caffeine, ferrocene, CF3Br-benzene, [ZnCl4]²⁻ and
[Cu(NH3)4]²⁺ (doublet) to 7e-12 Eh, taxol (113 atoms, 0.8 s) to 9e-12 Eh; 22 generated
element molecules (§5.1) to 7e-12 Eh. Not reproducible: SCFs that do not settle (neutral ZnCl4
oscillates for 286 iterations in the program), and exactly degenerate, partially filled HOMO
sets, below.

**Occupations** (`occu_`, decoded [C]): xtb's `occu`, plain integer Aufbau — N_α and N_β
lowest orbitals occupied (closed shell doubly), no averaging over degenerate levels; its 0.99
only locates the HOMO index (`focc > 0.99`). So when a matrix of the SCF has an **exactly
degenerate HOMO set that is only partly filled**, the occupied subspace is whichever the
eigensolver returns, i.e. round-off, and no reimplementation can follow the program there.
UF6 (O_h; `tests/reference_v2/elements/UF6`): the damped Fock matrix of iteration 2 has a
three-fold HOMO set (ε = −0.1022481 Eh to 1e-16) holding four electrons. Our energies of the
program's densities of iterations 1 and 2 (read under gdb) equal its printed −1075.44639916 and
−1039.62562161 Eh, our Fock matrices at them its own to 2.5e-14, and its iteration-2 density
equals ours to 2e-14 outside the degenerate set; inside, the same four electrons sit in another
two-dimensional subspace (occupations 1.60/0.45/1.95 in our orbitals instead of 2/2/0;
our own choice gives −1039.6465 Eh, and a 1e-15 change of the matrix another one, −1039.600).
From there the SCF paths part (the energy jumps by tens of Eh several times in both) and meet
on the same converged state: 50 iterations in the program, 40 in axtb, totals 1.1e-10 Eh
apart.
Test `scf_paths::uf6_v11_degenerate_iteration_two` (program densities in
`tests/reference_elements/uf6_v1.1_iterations.json`). The element survey (§5.1) has the same
mechanism in VCl4 (d¹, e set), UCl4 (f²) and YbCl2, and round-off-sized differences elsewhere.

**Orientation** [C]: the program's SCF solution, and with it axtb's in program conventions,
depends on the orientation of the molecule by up to ~1e-7 Eh at any SCF threshold
([Ca(H2O)6]²⁺ rotated: 9.76e-8 Eh at accuracy 1 to 1e-6, the program likewise), because the
onsite-exchange Fock matrix doubles the population part off the diagonal (§4.16), which is not
rotation covariant. The default conventions (exact Fock matrix) are invariant to 2e-13 Eh.

### 4.18 Conventions and analytic gradients [C]

**Conventions** (`scf::Conventions`, `conventions` in the CLI, Python and qccodec):

- `axtb` (default): the SCF diagonalizes F = ∂E/∂P exactly (AES potentials of all five
  parts; OFX population part ½(y_κ + y_λ)S_κλ off the diagonal). Same energy expression,
  same SCF procedure and thresholds; the converged density is stationary.
- `gxtb`: the program's Fock matrix (§4.16); reproduces the program (§4.17).

What `axtb` changes (both SCFs with the default thresholds at the same geometries): total
energies are lower by 8e-9 to 4e-5 Eh for the 14 + 5 reference molecules
(median ~8e-6), except ferrocene (−2.3e-4 Eh), with the same or
1–2 fewer SCF iterations (e.g. caffeine 12 → 11, ferrocene 12 → 10, glycine 12 → 10). Energy
differences move much less (`python tests/gxtb_energy_differences.py`,
`bench/results/gxtb_energy_differences.json`; SCF accuracy 1e-4 in both conventions).
Conformers, the water dimer and the reactions are minima and first-order saddle points of
g-xTB v1.1 itself (default conventions), optimized through qccompute with geomeTRIC
(GAU_VERYTIGHT) and Sella (fmax 0.001 eV/Å from axtb's Hessian; one imaginary mode, connected
to the intended minima by a pseudo-IRC); all structures are generated or public and stored in
`tests/structures/energy_differences/v1.1/`:

| quantity | `gxtb` (kcal/mol) | change with `axtb` (Eh) |
|---|---|---|
| n-butane gauche − anti | 0.6109 | +5.8e-6 |
| ethanol OH gauche − anti | −0.3432 | +6.0e-7 |
| ferrocene staggered (D5d, ring-rotation saddle) − eclipsed | 0.1845 | +1.6e-7 |
| Cl⁻ + CH3Cl → [Cl···CH3Cl]⁻, complexation | −9.4330 | −6.2e-6 |
| Cl⁻ + CH3Cl SN2 identity reaction, barrier from the complex (TS 445i cm⁻¹) | 14.9163 | +7.0e-6 |
| same, TS − separated reactants | 5.4833 | +8.0e-7 |
| HCN → HNC, reaction energy | 6.7835 | −8.0e-7 |
| HCN → HNC, barrier (TS 1057i cm⁻¹) | 54.1483 | −5.6e-6 |
| water dimer binding | −4.7713 | −2.6e-7 |
| iodobenzene···acetone binding (`tests/structures/beni-acetone.xyz`) | −2.5036 | +3.9e-7 |
| ethylammonium perchlorate ion-pair binding (`tests/structures/s229-cl.xyz`) | −104.6248 | −9.4e-7 |
| Ni(CO)4 → Ni(CO)3 + CO (unrelaxed) | 34.5156 | −1.1e-5 |
| [Cu(NH3)4]²⁺ → [Cu(NH3)3]²⁺ + NH3 (unrelaxed) | 48.7502 | −2.3e-6 |

(The last four rows, at fixed public structures, are as in the earlier version of this table,
where the ion pair was mislabeled Me2NH2⁺ ClO4⁻; the conformer, water-dimer and reaction rows
are new.) Program-convention differences also carry the orientation dependence of §4.17
(~1e-7 Eh); the structures fix the orientation.

**Gradient** (`Setup::gradient`, `gradient.rs` + `response.rs`): with the SCF orbitals C,
energies ε and occupations n (eigenvectors of the diagonalized Fock matrix),

dE/dR = ∂E/∂R|_P − Tr(W dS/dR) + Tr((F − F_used) dP/dR),  W = Σ_σ C n ε Cᵀ.

- ∂E/∂R|_P differentiates everything that depends on R: the SCF-basis overlap and multipole
  integrals (centres and the q_eff-dependent contraction; overlap and both atoms' moments
  from one integral-derivative call per shell pair), the H0 basis and diatomic-frame scaling
  (bond-direction derivatives of the frame projectors), the ACP projector overlaps, the
  g-xTB CN (levels, Hubbard parameters, ACP weights, repulsion, ES1), the kernels (γ2, τ,
  MFX, AES g3–g9), revD4 (pair weights through its CN, damping, three-body) and the
  repulsion; then the chain rules q_eff → EEQ-BC charges and basis CN, the EEQ-BC charges by
  the adjoint of the charge equations (A w = ∂E/∂q), and all CNs → R. dS/dR is the total
  overlap derivative (centres and contraction).
- `axtb`: F_used = F, the last term vanishes.
- `gxtb`: F − F_used = F_AES(dQ, QQ) − F_x (F_x the extra half of the OFX population part,
  linear in P). The last term is a Z-vector response: Aᵀz = g with
  (AU)_ai = (ε_a − ε_i)U_ai + [C_vᵀ δF_used(n(C_v U C_oᵀ + h.c.)) C_o]_ai,
  g = 2n C_vᵀ(F − F_used)C_o, solved by restarted GMRES with a diagonal preconditioner
  (~25 iterations to 1e-9); δF_used(D) is the exchange part (linear) plus the potentials
  differentiated by central differences of the cheap charge/moment potential functions;
  Aᵀ uses the symmetry of the Hessian of E' = E − E_dQ − E_QQ and the explicit adjoint of F_x.
  The z-dependent terms need Tr(Z ∂F_used/∂R|_P), the directional derivative of the explicit
  gradient of E' along Z = sym(C_v z C_oᵀ): the integral derivatives are linear in their
  weights, so one pass with combined weights; weights and cheap terms by central
  differences (exact up to third order in P).

Validation (`tests/gradient.rs`): five-point finite differences (h = 5e-4 bohr) of tight
SCF energies (1e-13 Eh / 1e-11) for all 14 reference molecules: ≤ 5.2e-10 Eh/bohr with
`gxtb` (≤ 6e-10 with `axtb` for water, O2, [CuCl4]²⁻, mesyl chloride, Ni(CO)4, glycine,
benzene), plus H atom, H2⁺, He2, LiH triplet, CH3 radical ≤ 2e-9. Against the program (`gxtb`): its tight-threshold central differences
equal ours at the same step to ≤ 3e-11 (all atoms but the first, which it sets to minus the
sum of the others); the analytic gradient equals their Richardson extrapolation (h, h/2) to
≤ 3.5e-8 and the program's default `gradient` file to ≤ 4.1e-5 (its truncation and SCF
noise). With the default SCF thresholds the gradient is within 3e-6 Eh/bohr of the tight
one (either convention; first order in the density error), with `accuracy = 0.01` within
2e-8.

Cost (one thread, laptop): caffeine SCF 15 ms, gradient 38 ms (`axtb`) / 52 ms (`gxtb`);
taxol (113 atoms, 299 AOs) SCF 0.54 s, gradient 0.58 s (`axtb`) / 1.1 s (`gxtb`, half of it
the Z-vector equations).

**Hessian** (`Calculator::numerical_hessian`, or `Calculator::hessian` with the numerical method): central differences of analytic gradients (step 0.005
bohr, SCF accuracy 0.01, displacements in parallel), as for GFN2; matches second
differences of tight energies to 6e-6 Eh/bohr² (water). The program's `-hess` frequencies of
the (non-stationary) reference water differ from ours by 0.4/7/0.15 cm⁻¹ (bend, symmetric,
antisymmetric stretch), also with tight SCF thresholds; its Hessian/projection at
non-stationary geometries was not decoded [U].

---

## 5. Reference data set

`crates/gxtb/tests/reference/<molecule>/`, generated by `crates/gxtb/scripts/make_reference.py`
(program 1.1.0, `OMP_NUM_THREADS=1`, no restart file):

| file | content |
|---|---|
| `mol.xyz`, `input.json` | geometry (Å), charge, unpaired electrons |
| `gxtb.out`, `molden.input`, `energy` | `-molden` single point: full printout, basis + MOs + orbital energies, total energy (14 digits) |
| `gxtb_grad.out`, `gradient` | `-grad -symthr 0`: numerical gradient (central differences, h = 0.0075 bohr; first atom = −Σ others) |
| `gradient_tight` | the same with the SCF thresholds of every single point overwritten to 1e-12 Eh / 1e-10 under gdb (energy threshold at rbp−0x2a80, density threshold at rbp−0x2ad0, breakpoint 0x589c1b in `scf_`; `make_reference.py`, `TIGHT_GRADIENT_ONLY=1`) |
| `summary.json` | parsed printout: EEQ table, q-vSZP table, ncao/nsao/npr, electron counts, energy terms (8 decimals), SCF iterations |
| `intermediates.json` | full precision from the program's memory at the entry of `scf_`: EEQ-BC charges, basis CN, q_eff, coordinates (bohr), primitive exponents and coefficients per Cartesian AO, atomic numbers; and E_rep read after `repuls_` |

Molecules (14): water, methane, ethanol, benzene, glycine, acetate (−1), hydroxide (−1),
ammonium (+1), dioxygen triplet (UHF 2), mesyl chloride CH3SO2Cl (S, Cl), iodomethane (I),
silylphosphine SiH3PH2 (Si, P), Ni(CO)4 (TM, closed shell), [CuCl4]²⁻ (TM, doublet, −2).
Geometries: Open Babel `--gen3d` (MMFF) or hand-built; they need not be minima.

Further files, all read from the program's memory with gdb in fresh runs (no restart file):

| file | content |
|---|---|
| `h0.json` | packed (program AO order) diatomic-frame scaled overlap, H0, H0 + ACP, and the AO levels lev − lev_cn·CN |
| `scf.json` | converged packed spin densities (closed shell: total; open shell: α then β), packed SCF overlap, Mulliken shell and atomic charges |
| `kernels.json` | the AES inputs of the final iteration (atomic dipoles and quadrupoles, kernels g3–g9; key `aes`), γ2 and the third-order kernel τ (packed shells; key `half_gamma2_sq_packed`, which is −½γ2² only within atoms), MFX kernel (full AO matrix), per-shell Γ and k4 arrays, and the final-iteration energies ES1, ES2+3, E_OFX, Ex, E_AES, (open shell) E_spin, revD4 two-body `disp2` and three-body `disp3` at full precision |
| `fock.json` | packed Fock matrices per spin before the mixing (`diisfock_` entry) and the densities they were built from, for the first iteration (start density) and the last call |

`tests/reference_extra.json`: five more molecules (geometries of the repository's
`tests/structures`, with charge and spin) with the program's total energies and iteration
counts, and 22 generated element molecules of the element survey (§5.1; `source`
`tests/gxtb_elements.py`, written by its `--write-reference`). `tests/reference_elements/`:
the program's densities of UF6's first two SCF iterations (§4.17) and the 2.0.1 program's
energies around [Gd(H2O)6]³⁺ (2.0.1 spec §4.18).

`make_reference.py` documents the breakpoint addresses (`H0_STOPS`, `SCALARS`, `DESCRIPTORS`,
the `wiberg_` entry, the call after `electro_`, `disp_dump`, `fock_dump`); new quantities are
added the same way (`DISP_ONLY=1`, `FOCK_ONLY=1` refresh single files). The refdir argument
must be an absolute path (gdb runs in a scratch directory).

Useful gdb stops in `scf_` (1.1.0 binary): 0x5830ec after `new_d4_dispersion_` (dispersion
object at rbp−0x29d0, its `d4_model` at +0x348 with descriptors wf +0x20, zeff +0x80, eta +0xc8,
en +0x110, rcov +0x158, r4r2 +0x1a0, nref +0x1e8, ngw +0x230, refcn +0x290, refq +0x2f0, aiw
+0x350, c6 +0x3c8; damping s6, s8, s9, a1, a2, alp at +0x7a0, cutoffs at +0x7d0); 0x5831d1 E_ATM
at rbp−0x120; 0x589977 two-body energy at rbp−0xd8 (`get_energy_` gets the SCF atomic charges
at rbp−0x4f0); 0x588e43 call `diisfock_` (args: iteration, max. stored matrices, α, first Pulay
iteration, transition iteration, 1000, stored count, S, mixed Fock in/out, P, history arrays);
0x58916d call `solve_` (Fock matrix = 3rd stack argument); 0x589bf9 end of iteration (energy at
*[rbp−0x3f8], converged flag rbp−0x21a0, ΔP_rms rbp−0x3f0, thresholds rbp−0x2a80/−0x2ad0).
gdb's Python API makes these dumps easy (the Intel Fortran array descriptor: base, element
length, offset, flags, rank, reserved, then extent, stride (bytes), lower bound per dimension).
`extract_runtime_tables.py` tabulates `shellocc_ref_` by calling it under gdb.

### 5.1 Element coverage (`tests/gxtb_elements.py`)

Every element H–Lr in one generated molecule (the GFN2 reference matrix's `elements` category,
high-spin f^n states because g-xTB treats f shells explicitly, actinides added the same way) and
a second environment for many (aqua and ammine ions, halides, oxides, radicals): 169 molecules,
run through the 1.1.0 program (`gxtb -c`, `-grad -symthr 0`) and axtb in program conventions
(results with 2.0.1 in `bench/results/gxtb_elements.json`).

- 29 molecules of the 24 elements without parameters (Z = 59–69, 90, 91, 93–103, §3.1): the
  program fails in its first diagonalization ("(sygvr) failed to solve eigenvalue problem",
  with exit code 0 and no energy), axtb refuses them.
- 130 of the other 140 agree: total energies ≤ 2.8e-12 Eh, the same SCF iterations, printed
  terms within their 8 decimals (≤ 5.0e-9 Eh).
- 6 are rounding-sensitive (diagnosed by rerunning both programs on a translated copy with the
  atoms in reverse order, which changes only the round-off): UF6 (§4.17) and VCl4 (d¹), UCl4
  (f²) and YbCl2, whose SCFs pass through or end in exactly degenerate, partly filled sets
  (energies 4.5e-9 to 1.7e-5 Eh apart, 37–108 iterations; axtb's own result moves as much under
  the perturbation), and HeH⁺ and the CH3 radical (printed terms 7e-9 / 9e-7 apart, totals
  1.3e-9 Eh at most).
- 4 differ in the printed "Ex (Mulliken)" only, by 6.5e-9 to 1.7e-8 Eh ([Ca(H2O)6]²⁺,
  [Ba(H2O)6]²⁺, [La(H2O)6]³⁺, [Rh(NH3)6]³⁺; totals ≤ 9e-12 Eh, the same iterations): the far-pair
  Fock elements of §4.16 [U], stable under the perturbation.
- Gradients: the program's are numerical with its default SCF thresholds, axtb's analytic ones
  differ from them by 4e-6 Eh/bohr (median; up to 4e-3 for the open shells with soft or
  degenerate SCFs). So axtb's gradient (both conventions) is checked against five-point
  differences of its own energies at SCF accuracy 1e-7 (h = 1e-3 bohr, also 4e-3 above 1e-8):
  129 of 136 molecules ≤ 1e-8 Eh/bohr (median 1.6e-10). [Os(NH3)6]²⁺ shows an energy step of
  1.3e-9 Eh (error ∝ 1/h: a primitive pair crossing the integral screening), six more 1.6e-8 to
  8.8e-8 (BrF3, CuCl2⁻, AuCl2⁻, the Cr, Fe and Cd aqua ions) with the same error in both
  conventions, whose gradients are computed differently (only `gxtb` has the Z-vector term),
  so from the energies, not a gradient formula. The tight SCF does not converge for VCl4,
  UCl4, YbCl2 and [Cu(H2O)6]²⁺ (displaced geometries).
- Orientation: the program's (and program-mode) results depend on the orientation at the 1e-7
  Eh level (§4.17), so the diagnosis uses a translation and reordering, not a rotation.

## 6. Open questions and how to resolve them

1. Analytic gradients: done (§4.18), with a Z-vector response for the program's
   non-stationary SCF.
2. Fock elements of far AO pairs (§4.16): Cl–Cl d–d at 8.5 bohr in [CuCl4]²⁻ (1e-9), H–H at
   11.3 bohr in [Ca(H2O)6]²⁺ (1.6e-7; printed Ex 1.3e-8 Eh, totals 1e-11). `setdqlist_` is
   decoded and not the cause; next: compare the program's AES and exchange Fock contributions
   separately for those pairs under gdb (`addanisotropich1_`, `setfock_mat_`).
3. Degenerate occupations (`occu_`: integer Aufbau, decoded; exactly degenerate, partly filled
   sets are round-off, §4.17); Fermi smearing (`fermismear_`, `-tel`): **done** (§4.17).
4. Elements 59–69, 90, 91, 93–103 are absent from the parameter file (§3.1).

## 7. Implementation status in `crates/gxtb` and tooling

| Module | Content | Validation against the program (14 molecules) |
|---|---|---|
| `params` | parsers for `gxtb_parameters`, `basisq`, `eeq` | unit test |
| `data`, `data_runtime` | compiled-in tables (static data; `shellocc_ref_` via gdb) | bit-exact by construction |
| `eeqbc` | EEQ-BC charges, CN, local charges | charges 8e-16 |
| `qvszp` | basis CN, q_eff, normalized per-atom shells | CN 4e-16, q_eff 7e-16, coefficients 1.3e-15 rel. |
| `classical` | g-xTB CN, repulsion, increments | E_rep 4e-16 Eh |
| `ints` | SCF-basis overlap, Mulliken shell charges | S 1.7e-15 (incl. d); shell charges 5e-15 at the program's density |
| `h0` | H0 basis, diatomic-frame scaled overlap, H0 | S̃ 1.1e-15, H0 3e-16 |
| `acp` | atomic correction potentials | H0 + ACP 1.8e-15 |
| `kernels` | γ2, third-order τ and MFX kernels | 2e-16 / 1e-14 / 6e-17 |
| `exchange` | Mulliken-approximated exchange and onsite exchange (OFX) energies | 1.1e-14 / 3e-16 Eh at the program's density (RHF and UHF) |
| `es` | first- to fourth-order energies, spin polarization | ES1 9e-15, ES2+3 5e-15, E_spin 1e-17 Eh at the program's density |
| `aes` | atomic multipole moments, AES kernels and energy | moments 1.2e-13, E_AES 1.1e-15 Eh at the program's density |
| `disp` | revD4 (on `axtb_d4`: reference C6, damped pair list, ATM kernel) | two-body 3e-18, ATM 5e-20 Eh at the program's charges |
| `fock` | Fock matrices of all terms | 7e-15 at the program's densities (1.5e-9 for [CuCl4]²⁻, §4.16) |
| `scf` | setup, energy/Fock evaluation, start density, the program's mixing and convergence, `singlepoint()` | total energies 4.6e-13 Eh, terms 4e-12 Eh, charges 1.7e-11, orbital energies 1.5e-14 Eh, same iteration counts; 5 more molecules 7e-12 Eh |
| `read_xyz` | the program's Å→bohr convention | ≤ 1.1e-16 bohr |
| `gradient`, `response` | analytic gradients: explicit derivatives and chain rules; Z-vector response of the program's Fock matrix | finite differences ≤ 5.2e-10 Eh/bohr; program central differences ≤ 3e-11 (§4.18) |
| `calculator` | `Calculator` (method `"gxtb-v1.1"` in the axtb CLI, Python package and qccodec; `Calculator::new(Version::V1_1, …)`), conventions, semi-numerical Hessian | bit-identical to `Setup::scf`/`gradient`; threads tests |

`crates/d4` hooks used by `disp` (backwards compatible, tested in `crates/d4/src/tests.rs`):
`D4Model::with_reference_data` (replace the reference systems of every element and recompute
the reference C6), `D4Cache::pairs` (damped two-body pair factors) and
`D4Dispersion::three_body_c6` (ATM with a caller-supplied C6 matrix).

Scripts (`crates/gxtb/scripts/`): `extract_binary_tables.py` (→ `src/data.rs`; with
`crates/d4/src/data.rs` as third argument it checks the shared dftd4 tables),
`extract_runtime_tables.py` (→ `src/data_runtime.rs`), `make_reference.py`
(→ `tests/reference/`), `inspect_binary.py` (constants of a routine, users of module arrays,
disassembly).

Remaining: faster gradients (overlap-only
derivatives for the H0 and ACP integrals instead of the multipole kernel; a better
preconditioner for the Z-vector equations).
