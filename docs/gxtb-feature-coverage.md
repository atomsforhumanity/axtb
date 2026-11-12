# g-xTB 2.0.1 feature coverage: the program and axtb

Which features of the g-xTB 2.0.1 program (`xtb --gxtb`, an xtb 6.7.1 build) work with g-xTB,
and which of them axtb provides. Most of xtb's options belong to GFN2-xTB, GFN1-xTB or GFN-FF,
so every option was run with `--gxtb` and checked for what it actually did.

**How it was tested (2026-11-05).** The program was the g-xTB 2.0.1 build of xtb, `xtb version
6.7.1 (26dd68d)`, built 2026-05-14, sha256 `1b4e30b68ed4e88b…`. It ran on an i9-11900H with
`OMP_NUM_THREADS=1`, one fresh directory per run, and the geometry first: `xtb mol.xyz --gxtb
[options] [--input xc.inp]`. Every option in `xtb --help` and the xcontrol blocks used with
g-xTB were tried on `tests/structures/water.xyz` and `caffeine.xyz`. For each run we checked:
- the exit code;
- `Unknown option` warnings and the `-n-` error chain;
- the total energy against the plain g-xTB single point (water −76.437387426 Eh, caffeine
  −680.419231528 Eh; GFN2 gives −5.070 and −42.154 Eh, so a fallback to another method shows
  in the energy scale);
- the files written, and the printout or file content the option should produce.

GFN2 control runs (no `--gxtb`) separated g-xTB-specific behaviour from the general xtb kind.
The public xtb side of the build (`reference/xtb-gxtb`, mainly `src/prog/main.F90` and
`src/tblite/calculator.F90`) explains several results. Numbers from earlier work cite
`docs/gxtb-v2.0.1-spec.md` ("spec").

**Columns**

- **Program with `--gxtb`** starts with the outcome:
  - **Works**: does what it says;
  - **Caveats**: runs, but g-xTB-specific problems apply;
  - **Error**: stops with an error;
  - **Ignored**: silently ignored, or falls back to another method;
  - **N/A**: only for other methods.
- **axtb**: **Has**, **Partial**, **No**, or n/a (dates: when axtb gained it).
- **Category**:
  - **ES**: electronic-structure property or setting; belongs in axtb.
  - **Derived**: analysis built on axtb results, such as thermochemistry from the Hessian.
  - **Driver**: belongs in a driver (qccompute and the optimizers, MD or reaction-path programs
    it runs). The cell says whether such a path exists today.
  - **Upstream: none**: g-xTB doesn't support it upstream either.
  - **Control**: program control or I/O with no axtb equivalent needed.

## 1. Method, charge, spin and SCF settings

| Feature | Program with `--gxtb` | axtb | Category |
|---|---|---|---|
| Single point (`--sp`, `--scc`, the default) | **Works.** Water −76.437387426460, caffeine −680.419231527755 Eh. Writes `charges`, `wbo`, `xtbtopo.mol` and `xtbrestart` by default. | **Has.** `gxtb-v2.0.1` in Rust, CLI, Python and qccodec. In `gxtb` conventions, energies agree with the program to ≤ 1.6e-12 Eh on 57 molecules (README). | ES |
| Gradient (`--grad`; files `gradient`, `energy`, `<name>.engrad`) | **Works.** | **Has.** Analytic, ≤ 5e-11 Eh/bohr vs the program. Returned in memory or JSON, no files. | ES |
| Charge and spin (`-c/--chrg`, `-u/--uhf`, `.CHRG`/`.UHF` files, `$chrg`/`$spin`) | **Works.** All four routes give the water cation −75.962912048 Eh. Any `--uhf`, even 0, switches to two spin channels; an odd electron count does so too (`calculator.F90`). | **Has** `charge`/`uhf`, and multiplicity through qccodec. The files and xcontrol routes are n/a. | ES |
| Open shells: g-xTB's own spin-polarization term | **Works** (spec §4.10). | **Has.** | ES |
| `--spinpol` | **Caveats: double counting.** It adds tblite's generic GFN spin-polarization container on top of g-xTB's own spin term (`calculator.F90`). Water triplet −75.862275 → −75.894651 Eh (−20.3 kcal/mol); caffeine triplet −12.5 kcal/mol. Singlets are unchanged. | No. axtb should reject it. | Upstream: none (not part of the parametrization) |
| Electronic temperature (`--etemp`) | **Works.** Fermi smearing. **For g-xTB the default is 0 K** unless `--etemp` is given (`calculator.F90`). Caffeine at 3000 K moves by −1.3e-9 Eh, but stretched or metal systems move by tens of kcal/mol. SCF retry ladders (1000/3000 K) and nanoreactor MD (3000 K) use it. | **Done.** `etemp` in Rust, the CLI (`--etemp`), Python and qccodec, 0 K by default; program mode reproduces `--etemp` (energies, SCF iterations and analytic gradients at 300–5000 K; quirks: the fractional electron count is rounded, the Fermi search stops at 1.5e-8 electrons), analytic and response Hessians include the occupation response (spec §4.17.1, §11.6). v1.1's `-tel` too. | ES |
| `$scc temp=`, `$scc maxiterations=` (xcontrol) | **Ignored.** Energies are identical to the default, and `maxiterations=3` still converges. Only the command-line `--etemp`/`--iterations`/`--acc` reach the tblite calculator (`main.F90`). The GFN2 control honors both: `$scc temp=3000` equals `--etemp 3000`, and `maxiterations=3` fails. | n/a (`etemp`, `max_iter` keywords) | Control |
| SCF accuracy (`-a/--acc`) | **Works.** Scales the convergence thresholds (1e-6·A Eh, 2e-5·A e) and the integral cutoff. Caffeine: `--acc 0.01` −8.7e-8 Eh, `--acc 10` +4.0e-7 Eh. | **Has** (`accuracy`, same meaning in `gxtb` conventions) | ES |
| SCF iteration limit (`--iterations`) | **Works.** `--iterations 3` gives exit 128, "SCF not converged in 3 cycles". | **Has** (`max_iter`) | ES |
| Wavefunction restart (`xtbrestart`, `--[no]restart`) | **Works.** A second caffeine run in the same directory reads it: 2 SCF cycles instead of 9, energy −7.0e-8 Eh different at the default `--acc`. `xtbrestart` holds the shell charges and magnetizations, atomic dipoles and quadrupoles and the density matrices (format 3, `src/tblite/restart.F90`); the SCF starts from them at any geometry of the same molecule, and an optimization inside one run carries the same state between steps. The 1.1.0 program's `gxtbrestart` holds the packed densities and MO coefficients; it restarts from the densities and skips the transition step of its Fock mixing (spec v1.1 §4.17). | **Done** (2026-11-05). `axtb_gxtb::Calculator` (v1.1, 2.0.1, also the composite's g-xTB part) starts every SCF from the previous geometry's converged state in the default conventions, retrying once from the EEQ-BC start if that SCF fails; `conventions="gxtb"` keeps the program's fresh start, and `guess="previous"` there reproduces the programs' restarts exactly: energies ≤ 4.3e-11 Eh and identical SCF iterations, for the same geometry (2.0.1: 2 cycles, 1.1.0: 3) and 0.02–0.3 bohr displacements, closed and open shells (caffeine, water, triplet O2, [Cu(NH3)4]²⁺). Results report `guess` (`previous`/`eeq-bc`); `guess_state()`/`set_guess_state()` move a state between calculators; qccompute's `propagate_wfn=True` carries it between geomeTRIC/Sella steps through the codec. Saved (README "Restarts"): 10–45 % of the SCF iterations along random walks of caffeine and taxol, 26 % in a geomeTRIC minimization of caffeine (154 instead of 207, 25 instead of 23 steps), 43 % along a Sella IRC (5.0 instead of 8.9 per point); a whole g-xTB step gets only 0–7 % faster, since the SCF is a quarter to half of an energy + gradient. In the default conventions a restarted 2.0.1 SCF converges to 0.1 × the thresholds (at the program's thresholds its history-dependent gradient errors made geomeTRIC take 48 steps). | ES (SCF guess reuse) |
| Electric field (`--efield x,y,z`, `.EFIELD` file) | **Works.** **Units are V/Å** (`efield*vatoau`). Water, 0.01 V/Å along ±x: ∓1.411e-4 Eh, exactly −μ·F with μx = 0.7257 e·bohr; along z: +9.97e-5 Eh. | No | ES |
| Method switches combined with `--gxtb` (`--gfn N`, `--gfnff`, `--tblite`, `--ptb`) | **N/A, order-dependent.** `--gxtb --gfn 1` runs GFN1 (−5.768 Eh), while `--gfn 1 --gxtb` runs g-xTB. `--gfnff --gxtb` runs GFN-FF. `--gxtb --gfnff`, `--gxtb --tblite` and `--gxtb --ptb` run g-xTB. | n/a: one method per call | Control |
| `--ptb`, `--raman`, `--alpha` | **Error or N/A.** PTB fails to construct its calculator in this build, with or without `--gxtb`. `--raman` stops with "Raman activities are not implemented with this method" for GFN2 as well. `--alpha` prints nothing for GFN2 either. | No | Upstream: none |
| `--ceh` (CEH charges) | **N/A.** tblite's CEH guess model; `ceh.charges` is identical with and without `--gxtb`, so it isn't g-xTB. | No | Upstream: none (not g-xTB) |
| `--vparam`, `--xparam` | **N/A.** Parameter files for other models; `--vparam` stops with an error, `--xparam` is "not used". | n/a | Upstream: none |
| Periodic input (POSCAR, genFormat) | **Caveats: wrong energies.** Runs and writes `gradlatt`, but water in a 12/20/30 Å cubic box gives −72.9534/−72.9540/−72.9541 Eh, 3.48 Eh above the isolated molecule (−76.4374) and not approaching it. GFN2 refuses PBC ("Multipoles not available with PBC"). | No (molecules only) | Upstream: none (effectively) |
| Elements | H–Lr (103) | **Has** H–Lr | ES |
| Input formats (xyz, Turbomole `coord`, sdf/mol, PDB, gen, POSCAR via mctc-lib) | **Works** | **Partial.** The CLI reads XYZ only; Python takes arrays, and the codec takes a qcdata `Structure`. | Control (I/O) |
| Threads (`-P/--parallel`) | **Works** (OpenMP) | **Has** (`threads`; no implicit threading) | Control |

## 2. Implicit solvation and embedding

| Feature | Program with `--gxtb` | axtb | Category |
|---|---|---|---|
| `--alpb`, `-g/--gbsa` | **Error.** Exit 128, "Could not construct new calculator". | axtb rejects solvation for g-xTB; it **has** ALPB/GBSA for GFN2 (25/14 solvents, analytic Hessians). | Upstream: none |
| `--cpcmx` | **Ignored.** Gas-phase energy to the last digit, no `Gsolv` line; `--strict` doesn't catch it. | n/a (no CPCM-X) | Upstream: none |
| `--cosmo`, `--tmcosmo`, `--gbe`, `--gb` | **Caveats: unparametrized.** Runs and changes the energy. Water: COSMO +1.62 kcal/mol (wrong sign), GBE −4.08, GB −4.87. Caffeine: −4.17/−2.69/−2.91. GFN2 ALPB water gives −12.1; experiment −6.3 kcal/mol. `--tmcosmo` writes no `.cosmo` file. Don't use. | No | Upstream: none |
| **Composite solvation**: g-xTB + [GFN2/ALPB − GFN2] for E, gradient and Hessian | Not a program feature: run as three xtb programs per gradient (`xtb --gxtb`, `xtb --alpb water`, `xtb`). | **Has** (2026-11-05). `solvation_correction="gfn2"` with `alpb`/`gbsa` on a g-xTB method (Python, CLI, qccodec; `axtb_gxtb::CompositeCalculator`), energies, gradients and analytic Hessians in one process, with the settings, temperature pairing, retry ladders and zero check of the three-program workflow. `conventions="gxtb"` reproduces the three programs to ≤ 9.5e-11 Eh and ≤ 2.8e-10 Eh/bohr, retry ladder included (`tests/solvation_composite.py`). A g-xTB method with `alpb` alone is rejected with a pointer to it. README "g-xTB in implicit solvent". | ES (beyond parity) |
| Point-charge embedding (`$embedding`) | **Ignored.** The energy is unchanged and no `pcgrad` is written. The GFN2 control changes −5.0702 → −5.0803 Eh and writes `pcgrad`. | No (neither method) | Upstream: none. Electrostatic embedding for QM/MM would go beyond the program. |

## 3. Properties and output

| Feature | Program with `--gxtb` | axtb | Category |
|---|---|---|---|
| Mulliken atomic charges and shell populations (default printout, `charges`; `--pop` adds nothing) | **Works** | **Has.** `charges` everywhere, `shell_charges` in Python. | ES |
| Dipole moment (default printout; `--dipole` adds nothing) | **Works** | **Has** (`dipole`; codec `scf_dipole_moment`) | ES |
| Molecular quadrupole (`--quadrupole`) | **Caveats: not a tensor.** Prints tblite's `get_molecular_quadrupole_moment` about the coordinate origin ("total" line, 4 decimals): the atomic quadrupoles plus the charge and dipole parts, whose off-diagonal elements it takes twice (3 instead of 3/2, while the atomic quadrupoles use 3/2). The printed total changes its eigenvalues when the molecule is rotated (water: −1.354/−0.570/1.924, rotated by 45° −2.070/−0.196/2.265). xtb's GFN2-xTB printout (`molmom`, "full", 3 decimals) is the traceless tensor; `xtb --tblite --quadrupole` has tblite's error too. | **Done** (2026-11-05), every SCF method: `quadrupole` (xx, xy, yy, xz, yz, zz, e·bohr², about the origin) in Python, the CLI JSON and the codec (`extras['quadrupole']`; qcdata has no field); Rust `axtb::properties::molecular_quadrupole`. Program conventions print the program's form: g-xTB 2.0.1 `gxtb` and GFN2 `tblite` tblite's, GFN2 `xtb`/`xtb-6.7.1` xtb's; the default conventions the traceless tensor. Agreement with the printouts within their last digit: 2.0.1 ≤ 4.9e-5 on 12 molecules (water, rotated water, caffeine, acetate, [ZnCl4]²⁻, ferrocene, [Cu(NH3)4]²⁺, O2, NO2, CH2, taxol); xtb 6.8.0 ≤ 5.0e-4; xtb 6.7.1 ≤ 5e-4 at `--acc 1e-4` (at `--acc 1` it prints the moments it mixes once more after the last iteration: taxol 1.3e-3). | ES |
| Orbital energies, occupations, HOMO–LUMO gap (default printout) | **Works.** The summary's gap is the alpha channel's, also for open shells (mctc-lib's 27.211386245988 eV/Eh; xtb's GFN2-xTB 27.21138505; the 1.1.0 program "gap (eV) alpha->alpha", 27.212 in single precision). | **Done** (2026-11-05), every SCF method: per-spin `orbital_energies_alpha/_beta`, `occupations_alpha/_beta` and `homo_lumo_gap` (Eh) in Python and the CLI JSON; the codec fills qcdata's `wavefunction` (`scf_eigenvalues_a/b`, `scf_occupations_a/b`) and `calcinfo_nbasis/nmo/nalpha/nbeta/natoms`, the gap in `extras['homo_lumo_gap']`. Program conventions use the programs' alpha-channel gap, the default conventions the gap over both spins. Gaps vs the printouts: 2.0.1 ≤ 3.7e-11 eV (12 molecules), 1.1.0 ≤ 2.2e-7 eV (8 printed digits), xtb 6.8.0 ≤ 5e-13 eV, 6.7.1 ≤ 1e-8 eV. | ES |
| Mayer–Wiberg bond orders (`wbo` file and printout, on by default; `--wbo` adds nothing) and the `xtbtopo.mol` topology | **Works.** Quirk: the printout labels atoms with the symbol of the atom whose index is their species index (`mol%sym(mol%id)`), so heavy atoms appear as "H" (`3 8 H` for O); the `wbo` file (pairs above 0.1, full precision) is right. Open shells: 2 Σ_σ (P^σ S)(P^σ S) (tblite; xtb's GFN2-xTB the same, with pairs ≥ 10 bohr apart set to zero). The 1.1.0 program prints them (3 decimals, pairs above 0.05) only for small molecules (not for caffeine, 24 atoms; `-wrwbo` writes nothing). | **Done** (2026-11-05), every SCF method: the N×N matrix on request, Python `singlepoint(..., bond_orders=True)` → `bond_orders`, CLI `--wbo` (JSON `"bond_orders": true`), codec keyword `bond_orders` → `extras['bond_orders']` (qcdata has no field); Rust `Want::bond_orders`, `Electronic::bond_orders`. Against the `wbo` files: 2.0.1 ≤ 1.1e-11 on 11 molecules (closed and open shells, anions, a cation, ferrocene, taxol), xtb 6.8.0 ≤ 7e-14, 6.7.1 ≤ 1.2e-9, the 1.1.0 printout ≤ 4.1e-4 (its 3 decimals); every pair a program leaves out is below its threshold in axtb too. No `xtbtopo.mol`. | ES |
| Molden file (`--molden`) | **Works.** Writes `molden.input` with each atom's charge-adapted q-vSZP contraction and the MOs, in tblite's conventions: Cartesian functions (no `[6D]`/`[10F]` flags; Molden's default) whose components all have the norm of x^l, coefficients of unnormalized primitives, and the valence charge in the atomic-number column (O as 6). | **Has** (2026-11-05), for every method: Python `singlepoint(..., molden=True)` (`"cartesian"` for Cartesian functions), CLI `--molden [FILE]`, qccodec keyword `molden` (`results.files["molden.input"]`), Rust `Want::orbitals` + `ToMolden::to_molden`. Standard Molden conventions (spherical `[7F]`/`[5D]` by default, normalized primitives). Same orbitals as the program's file: 1 − \|⟨axtb\|program⟩\| ≤ 1.1e-9, orbital energies ≤ 2.7e-12 Eh (water, caffeine, O2 triplet, [Cu(NH3)4]2+, ferrocene, UF6, GdCl3, CeCl3; `tests/molden_check.py`, README "Molden files"). Readers that take only Cartesian files read `molden="cartesian"`. | ES |
| `--json` (`xtbout.json`) | **Ignored.** No file; GFN2 writes one. | n/a (CLI JSON output) | Control |
| `--lmo`, `--fod`, `--esp`, `--stm` | **Ignored.** No printout, no files. GFN2 writes `lmocent.coord`/`xtblmoinfo`, `fod`/`fod.cub`, `xtb_esp.*`, `stm.dat`. | No | Upstream: none |
| Vertical IP/EA, electrophilicity (`--vip`, `--vea`, `--vipea`, `--vomega`) | **Caveats: wrong.** A ΔSCF with g-xTB, but the ion is a spin-restricted single-channel calculation, and the GFN1-IPEA empirical shift (4.8455 eV, `global_ipeashift` "for all models") is subtracted. vIP: water 10.30 eV, caffeine 4.68 eV. Two g-xTB single points (`--chrg 1 --uhf 1`) give 12.91 and 8.23 eV. Caffeine vEA −6.81 eV, ω 0.05 eV. | **Partial.** The ingredients exist (single points at any charge and uhf); there is no helper. | Derived |
| Fukui indices (`--vfukui`) | **Works.** f(+), f(−), f(0) from Mulliken charges at N and N±1. | **Partial** (charges at three charge states; no helper) | Derived |
| `--cma` (principal-axes orientation) | **Works.** The dipole is printed in the new frame. | n/a (geometry preprocessing, caller or qcinf) | Control |

## 4. Hessians, frequencies and thermochemistry

| Feature | Program with `--gxtb` | axtb | Category |
|---|---|---|---|
| Hessian (`--hess`: `hessian`, `vibspectrum`, `g98.out`) | **Caveats.** Numerical central differences. At the default `--acc 1`, frequencies are off by up to 104 cm⁻¹ (caffeine) and 149 cm⁻¹ (taxol) (spec §11.5), so `--acc 0.001` is needed. Caffeine took 6.2 s (default) / 7.0 s (`--acc 0.001`) on one thread. `--ohess tight` at the default acc printed "Hessian element … is not symmetric". | **Has, better.** Analytic (default), response and numerical. ≤ 8e-9 Eh/bohr² vs tight references (taxol 8e-8); 3.4–4.1× faster than the program at `--acc 0.001`, which is still off by up to 1.1 cm⁻¹ (spec §11.5). | ES |
| `$hess` block (`step`, `scale`, `sccacc`) | **Works partly.** `step=0.001` changes the Hessian; `sccacc` doesn't reach the g-xTB calculator (spec §11.5). | **Has** (`hessian_step`, `hessian_accuracy`) | ES |
| Harmonic frequencies and normal modes (printout, `vibspectrum`, `g98.out`) | **Works.** The rigid-body block is projected out. | **Has** (2026-11-05), for every method with a Hessian: `axtb::vibrations`, Python `Calculator.hessian` and `axtb.vibrational_analysis` (any Hessian), CLI `--hess`, codec `freqs_wavenumber`/`normal_modes_cartesian` (unit-length Cartesian modes, checked with `qccodec.testing.check_normal_modes`), reduced masses, force constants and `n_imaginary`. `xtb-6.7.1` analysis conventions (the default for `conventions="gxtb"`: the 2.0.1 program is a 6.7.1-based xtb build) reproduce the program's projection (also for linear molecules) to its printed digits: on the program's own `--acc 1e-4` Hessian ≤ 5e-5 cm⁻¹ (caffeine, CO2, HCN, benzene, NH3, CH4; `tests/vibrational_check.py`, README "Vibrational analysis"); axtb's analytic Hessian vs the program's ≤ 0.35 cm⁻¹. | Derived |
| IR intensities (`vibspectrum` column, from dipoles at displaced geometries) | **Works.** Water 97.3/19.3/28.7 km/mol. | **Has** (2026-11-05): `ir_intensities` (km/mol) with `dipole_derivatives` from the analytic Hessian's coupled-perturbed density response (2.0.1; ≤ 1e-8 e vs finite differences of converged dipoles), the response Hessian's extrapolated densities, or the displaced dipoles of numerical Hessians (v1.1, 2.0.1). On the program's own `--acc 1e-4` Hessian, axtb's dipole derivatives give its printed intensities to ≤ 0.028 km/mol (≤ 0.03 %; caffeine, CO2, HCN, benzene, NH3, CH4). For GFN2-xTB the program's intensities match axtb's on the same Hessian to 0.07 % once its displacement SCFs run at `$hess sccacc=0.001` (its default 0.3 leaves up to 43 % on weak caffeine bands). | Derived |
| Thermochemistry (RRHO/qRRHO: ZPVE, H, S, G(T), `TOTAL FREE ENERGY`, `G(RRHO) contrib.`; `$thermo temp/sthr/imagthr/scale`; the `xtb thermo mol.xyz hessian` subcommand) | **Works.** `$thermo temp=500 sthr=100 scale=0.95` is honored. `xtb thermo` on a g-xTB `hessian` file reproduces the `--hess` run's G(RRHO) (0.151263 Eh, caffeine) and needs no g-xTB. | **Has** (2026-11-05): xtb's quasi-RRHO with its options (temperatures, pressure, `rotor_cutoff` = `sthr`, `imaginary_cutoff` = `ithr`, `frequency_scale`, symmetry number from the detected point group or given, `symmetry_threshold` = `desy`); ZPVE, H(T), S, T·S, G(RRHO), G = E + G(RRHO) (codec `gibbs_free_energy`, `extras['thermo']`); `analyze_hessian` CLI requests and `axtb.vibrational_analysis` replace `xtb thermo`. On the program's Hessians the `xtb` conventions give its ZPVE, H(T), T·S, G(RRHO) and G to ≤ 3e-11 Eh with identical point groups and symmetry numbers; `xtb thermo` on axtb's analytic Hessians agrees to ≤ 8e-12 Eh. Caffeine G(RRHO): axtb analytic 0.150578 Eh, program `--acc 1e-4` 0.150572, default `--acc 1` 0.151263. | Derived |
| Opt + Hessian (`--ohess [level]`) | **Works.** Caffeine 7.2 s. | Through qccompute: a geomeTRIC/Sella optimization with axtb gradients, then an axtb Hessian. | Driver (path exists) |
| Biased Hessian, SPH (`--bhess`; `hessian_sph`) | **Works** (optimizes first, then the biased Hessian) | No | Derived |
| Geometry displaced along an imaginary mode (`xtbhess.xyz`) | **Works** | No (the normal modes and frequencies to build it are returned) | Derived |

## 5. Drivers

| Feature | Program with `--gxtb` | axtb | Category |
|---|---|---|---|
| Optimization (`--opt crude…extreme`, `--cycles`, `$opt engine=rf/lbfgs/inertial`) | **Works.** ANCopt, L-BFGS and FIRE all converge. Caffeine: normal 0.34 s, tight 1.0 s, FIRE 1.8 s. With `--cycles 2`: "FAILED TO CONVERGE", exit 0. | Through qccompute: geomeTRIC or Sella with the axtb codec as the gradient (and exact Hessian) child. | Driver (path exists) |
| Constraints (`$constrain`: distance/angle/dihedral/atoms, force constant) | **Works**, as harmonic restraints. Caffeine C1–N2 1.437 → 1.587 Å for a 1.60 Å target at force constant 1; dihedral −1.5 → 59.7° for 60°; water angle 104.5 → 112.1° for 120° at the default force constant. | geomeTRIC `constraints` (exact; `set` targets in bohr), Sella `fix_bond/fix_angle/fix_dihedral` (exact). | Driver (path exists) |
| Fixed atoms (`$fix atoms:`) | **Works.** Fixed atoms don't move (0.0 Å). | geomeTRIC `freeze`, Sella constraints | Driver (path exists) |
| Scan (`$scan` with `$constrain`) | **Works** (`xtbscan.log`) | **No codec path.** qcdata defines `CalcType.scan`, but no qccodec codec implements it, and the geomeTRIC codec refuses scans. Workaround: a Python loop of constrained optimizations. | Driver (partial) |
| Wall potential (`$wall`, logfermi sphere) | **Works.** Water, 1 bohr sphere: +0.00871 Eh "add. restraining". | No wall in the geomeTRIC/Sella optimizations; an MD driver can add one around axtb's gradients. | Driver (partial) |
| MD (`--md`, `--omd`, `$md`: temp/time/step/dump/shake/hmass/nvt/restart) | **Works.** Caffeine 0.05 ps took 5.2 s (≈ 100 s/ps, one core); the trajectory energies are on the g-xTB scale; SHAKE runs. Quirk (all of xtb): the printed `<Epot>` average divides by n+1 (−673.65 vs −680.39 Eh for 100 steps). | Any MD driver that takes gradients from qccompute (program `axtb`); none was tried here. | Driver (path exists) |
| Metadynamics (`--metadyn`, `$metadyn`) and `--metaopt` | **Works.** Caffeine 0.05 ps metadyn 4.6 s; metaopt writes `xtbmeta.log`. | **No path.** CREST 3 runs g-xTB only through the external xtb binary. | Driver (missing) |
| Reaction path (`--path FILE`, `$path`; RMSD push/pull) | **Works.** Caffeine 57 s; writes `xtbpath_ts.xyz`. | the geomeTRIC NEB codec, and reaction-path programs that take axtb gradients | Driver (path exists, better) |
| Mode following (`--modef N`) | **Works**, but only after a `--hess` run with `$write modef=true` (writes `xtb_normalmodes`; molecules > 3 atoms). Otherwise exit 128, "Hessian not found! run -hess first!". | Sella transition-state searches with axtb Hessians cover the use | Driver (equivalent exists) |
| TS optimization and IRC | **None** in the program for g-xTB (only `--modef`/`--path`) | qccompute Sella/geomeTRIC `transition_state`/`irc`; axtb supplies analytic Hessians (`bench/ts_example.py`) | Driver (path exists) |
| Docking (`xtb dock`, aISS) | **Ignored, falls back.** `--gxtb` is an "Unknown option" for the dock subcommand (exit 0), and docking runs at its default levels without g-xTB. | n/a | Upstream: none |
| ONIOM (`--oniom inner:outer LIST`) | **Ignored.** With `--gxtb`, the energy is pure g-xTB (GFN2 control: writes `inner_region.xyz`); g-xTB isn't an allowed inner method. | n/a | Upstream: none |

## 6. Program control

| Feature | Program with `--gxtb` | axtb | Category |
|---|---|---|---|
| `-I/--input` (xcontrol), `--namespace` (prefixes every file), `--[no]copy`, `--define`, `-v` (adds only a constraints debug section), `-s`, `--ceasefiles` (no files), `--citation`, `--license` | **Works** | n/a (in-process API; JSON results) | Control |
| `--strict` | **Works only partly.** "Unknown option" stays a warning: `--strict --foo` exits 0. | axtb's CLI and codec reject unknown options and keywords | Control |

## Summary

axtb covers g-xTB 2.0.1's core:
- energies and gradients, matching the program to its print precision;
- charges, dipole, quadrupole, bond orders, orbital energies, the gap and open shells;
- Hessians that are faster and more accurate than the program's numerical `--hess`, with the
  vibrational analysis, IR intensities and thermochemistry;
- the electronic temperature, restarts and Molden files;
- implicit solvent through the GFN2-xTB composite, which the program does not have.

Optimizations, constraints, paths, transition states, IRCs and MD run in drivers (qccompute with
geomeTRIC or Sella, or any program that takes axtb's gradients and Hessians).

**Not in axtb yet:**
- the electric field (`--efield`, in V/Å as the program takes it);
- the SPH biased Hessian (`--bhess`), which matters for thermochemistry at non-stationary
  geometries;
- IP/EA/Fukui helpers built from charge-state single points (without the program's `--vip`
  shift);
- input formats other than XYZ in the CLI.

**Not planned** (not g-xTB upstream, or wrong there):
- `--spinpol`: double counts spin polarization; reject it explicitly.
- COSMO, GB, GBE and CPCM-X for g-xTB.
- Periodic boundaries.
- CEH, PTB, Raman, `--alpha`, LMO, FOD, ESP, STM.
- Point-charge embedding: only with a validation plan, since g-xTB has no published embedding.

**Gaps of the driver layer (not axtb):**
- a scan calculation type in qccodec (geomeTRIC or Sella constrained-optimization loop);
- a wall or bias potential for optimizers;
- RMSD-biased metadynamics (CREST reaches g-xTB only through the xtb binary).

**Program behaviour worth knowing** (found on the way):
- `$scc temp`/`maxiterations` are silently ignored for g-xTB; use the command-line flags.
- `--spinpol` changes open-shell g-xTB energies.
- `--vip`/`--vea` apply GFN1's 4.85 eV shift.
- `--efield` is in V/Å.
- `--oniom` and `xtb dock --gxtb` silently run without g-xTB.
- `--json`, `--lmo`, `--fod`, `--esp`, `--stm` are ignored.
- Periodic g-xTB energies are 3.5 Eh off.
- `--strict` doesn't make unknown options fatal.
- `--modef` needs `$write modef=true` in the Hessian run.
- `--quadrupole` (tblite's form, also `xtb --tblite`) doubles the off-diagonal charge and dipole
  parts: the printed total is not a tensor and changes with the orientation.
- The bond-order printout shows wrong element symbols (O as "H"); read the `wbo` file.
- The 1.1.0 program prints bond orders only for small molecules (not for caffeine).
- xtb 6.7.1 prints the charges and moments it mixed once more after the last iteration, not the
  converged ones (taxol at `--acc 1`: 5.7e-5 e).
- Restarts (`xtbrestart`, in-run optimization or MD steps) can follow another SCF solution than a
  fresh start on stretched bonds and metal clusters, at 0 K and 3000 K (N2 recompressed from 4 Å
  at 3000 K: up to 0.30 Eh above it).
