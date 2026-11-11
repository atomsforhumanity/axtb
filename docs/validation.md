# Validation

How axtb's results compare with the reference programs: GFN2-xTB with xtb 6.8.0, xtb 6.7.1
(the g-xTB 2.0.1 build, and the 6.7.1 release) and tblite 0.8.0; g-xTB with the g-xTB 1.1.0
and 2.0.1 programs; GFN-FF with xtb 6.8.0 and 6.7.1. Every number comes from a script in
`tests/` or `bench/` and a result file in `bench/results/` (or `tests/reference_matrix/`),
named with it. The README describes the methods and conventions these results refer to, and its
[Checks](../README.md#checks) section how to rerun them (`scripts/check.sh programs`) and what
each needs. The optional cross-checks against the private mepd workflow run only with
`MEPD_DIR` (`tests/molden_check.py`) or `MEPD_PYTHON` (`tests/solvation_composite.py`) set.

## Summary

| Method / feature | State | Validated against |
|---|---|---|
| GFN2-xTB energy + analytic gradient (incl. self-consistent D4, multipole electrostatics) | done | xtb 6.8.0 (`conventions="xtb"`): energies ≤ 5e-13 Eh, gradients ≤ 4e-14 Eh/bohr, same SCC iterations, at `--acc 1e-4` and the default `--acc 1`; xtb 6.7.1 (`conventions="xtb-6.7.1"`) to ≤ 1e-8 Eh; tblite 0.8.0 to ~1e-13 Eh (see [xtb 6.8.0](../README.md#xtb-680-and-671) and the `bench/` report) |
| Analytic Hessian (coupled-perturbed SCC response; default) | done: all terms analytic (H0/overlap/multipole integrals, ES2/ES3/AES, D4 incl. ATM, repulsion, ALPB/GBSA) | finite differences term by term; extrapolated numerical Hessians to ≤ 1e-10 Eh/bohr² (gas), see [Hessians](#hessians) |
| Response Hessian (finite differences of the gradient functional with the response-extrapolated wavefunction) and semi-numerical Hessian | done | as above; `xtb --hess`: frequencies within 1 cm⁻¹ (xtb's FD noise) |
| Python API + qccompute codec (`program="axtb"`) | done | — |
| ALPB/GBSA implicit solvation (GFN2; 25 ALPB / 14 GBSA solvents, 3 reference states) | done | xtb 6.8.0: full SCF ≤ 5e-13 Eh in all three states; xtb 6.7.1: components to all 12 printed digits, full SCF ≤ 3e-10 Eh; tblite 0.8.0 ≤ 1.3e-10 Eh |
| g-xTB v1.1 (public parameters) energies, analytic gradients and semi-numerical Hessians, also at a finite electronic temperature (`etemp`, the program's `-tel`), method `"gxtb-v1.1"` in Rust (`axtb_gxtb::Calculator::new(Version::V1_1, …)`), CLI, Python and qccodec | done (`crates/gxtb`, spec in `docs/gxtb-v1.1-spec.md`) | g-xTB 1.1.0 program (`conventions="gxtb"`): total energies of 41 molecules to ≤ 1e-11 Eh, same SCF iterations; its numerical gradients to ≤ 3e-11 Eh/bohr at equal step; gradients vs finite differences ≤ 6e-10 Eh/bohr; every element with parameters (79; 140 molecules, `tests/gxtb_elements.py`): 130 agree to ≤ 2.8e-12 Eh with the same iterations, 6 are rounding-sensitive (degenerate, partly filled sets: UF6, VCl4, UCl4, YbCl2), 4 differ in a printed term by ≤ 1.7e-8 Eh (far-pair Fock elements, spec §4.16) |
| g-xTB 2.0.1 energies, analytic gradients and analytic (default), response and numerical Hessians, also at a finite electronic temperature (`etemp`, the program's `--etemp`), method `"gxtb-v2.0.1"` in Rust (`axtb_gxtb::Calculator::new(Version::V2_0_1, …)`, the default version), CLI, Python and qccodec | done (`crates/gxtb/src/v2`, decoded from the program: `docs/gxtb-v2.0.1-spec.md`; Hessians §11) | g-xTB 2.0.1 program (`conventions="gxtb"`), 84 molecules H–Lr (23 with lanthanides or actinides) incl. two longer than 50 bohr: total energies ≤ 2.1e-12 Eh (9.1e-12 for (H2O)100), same SCF iterations, per-term energies ≤ 9.1e-12 Eh except TbCl3 2.6e-11 (an orbital of a near-degenerate HOMO set carries the program's fractional electron count, so the terms of the last, unconverged density follow the round-off while the totals agree; spec §10); every element H–Lr (169 molecules, `tests/gxtb_elements.py`): 155 agree to ≤ 4.9e-11 Eh with the same iterations and gradients ≤ 2.3e-10 Eh/bohr (NdCl3 2.5e-9), 12 are rounding-sensitive (open f shells at symmetric geometries, degenerate sets), 2 hit a program gradient error for bonds along an axis (spec §4.18); its analytic gradients ≤ 2e-11 Eh/bohr (default) and ≤ 6e-11 (tight; CeCl3 4.0e-9: 45 iterations around seven near-degenerate f orbitals, where the summation order of the density alone moves it by up to 6e-8); default conventions vs finite differences ≤ 6e-10 Eh/bohr; analytic Hessians vs tight references ≤ 8e-9 Eh/bohr² (water, O2 triplet, caffeine, [Cu(NH3)4]2+, ferrocene, Ni(CO)4, UF6, GdCl3; taxol 8e-8), response ≤ 4e-8; taxol analytic Hessian 95.6 s (1 thread) / 19.8 s (8) on the laptop vs the program's `--hess --acc 0.001` 533 / 95.5 s (`bench/results/final-laptop/`; 64-core node: 53.9 / 8.8 s vs 445 / 60.2 s), whose frequencies are off by up to 1.1 cm⁻¹ (default `--acc 1`: 149 cm⁻¹; spec §11.5); analytic Hessian memory O(N²) (`bench/results/gxtb2_hessian.json`, node): taxol 145 MB on 1 thread, (H2O)100 1.0 GB and (H2O)200 4.2 GB on 8. After 0.1.1 the response terms take a symmetric form (default; spec §11.3): the Hessian equals that of the 0.1.x solve to ≤ 2.2e-11 Eh/bohr² (15 cases, frequencies ≤ 2.5e-7 cm⁻¹) in 1.33-1.37x less time for taxol and (H2O)100, and keeps 2 · 3N · n_occ · n_virt single-precision numbers (taxol +53 MB, (H2O)100 +0.67 GB; `bench/results/gxtb2_hessian_symmetric.json`, laptop) |
| g-xTB in implicit solvent: the composite E_gxtb + [E_GFN2+ALPB/GBSA − E_GFN2] for energies, gradients and analytic Hessians in one process (`axtb_gxtb::CompositeCalculator`; `solvation_correction="gfn2"` in Python, CLI and qccodec) | done (`crates/gxtb/src/composite.rs`, see [g-xTB in implicit solvent](#g-xtb-in-implicit-solvent)) | the three xtb programs it replaces (`xtb --gxtb`, `xtb --alpb water`, `xtb`; `conventions="gxtb"`), 9 molecules incl. a charged species, an ion pair, a path image and two retry-ladder cases: energies ≤ 9.7e-11 Eh, gradients ≤ 5.3e-10 Eh/bohr, same retry rung; mepd's own `SolvationCorrection` ≤ 9.1e-12 Eh; analytic Hessian vs finite differences of the composite gradient ≤ 6e-8 Eh/bohr² |
| Molden files of the converged orbitals (all three methods; spherical or Cartesian functions; Rust, Python, CLI `--molden`, qccodec `molden` keyword) | done (`crates/axtb/src/molden.rs`, see [Molden files](#molden-files)) | the programs' own `molden.input` (`tests/molden_check.py`, 23 cases): same orbitals as g-xTB 2.0.1 (`xtb --gxtb --molden`; 1 − \|⟨axtb\|program⟩\| ≤ 1.1e-9 per orbital, orbital energies ≤ 3.3e-12 Eh, incl. UF6, GdCl3, CeCl3, ferrocene, [Cu(NH3)4]2+), g-xTB 1.1.0 (≤ 6e-11) and GFN2 `xtb --molden` (≤ 1e-10 at `--acc 1e-4`); every file reads back with orthonormal orbitals (≤ 6e-11) |
| Vibrational analysis of any Hessian: harmonic frequencies, normal modes, reduced masses, IR intensities (dipole derivatives from the response: GFN2-xTB, g-xTB 2.0.1; from displaced dipoles for numerical Hessians), quasi-RRHO thermochemistry with xtb's options and point-group symmetry numbers; Rust (`axtb::vibrations`), Python (`Calculator.hessian`, `axtb.vibrational_analysis`), CLI `--hess`, qccodec (`freqs_wavenumber`, `normal_modes_cartesian`, `gibbs_free_energy`) | done (see [Vibrational analysis](#vibrational-analysis-and-thermochemistry)) | `xtb --hess` of xtb 6.8.0 and 6.7.1 (GFN2) and `xtb --gxtb --hess` on their own Hessians (`tests/vibrational_check.py`: caffeine, CO2, HCN, benzene, NH3, CH4, SN2 TS): frequencies ≤ 5e-5 cm⁻¹ (their 4 printed decimals), ZPVE, H(T), T·S, G(RRHO) and G ≤ 8e-11 Eh (6.8.0: ≤ 1.4e-10, the `hessian` file's 10 decimals; against 6.8.0's `xtb thermo` on that file 2.4e-13), same point groups and symmetry numbers; `xtb thermo` on axtb's g-xTB Hessians ≤ 8e-12 Eh; dipole derivatives vs finite differences ≤ 1e-7 e (analytic, response), ≤ 3e-7 e (numerical) |
| GFN-FF energies, analytic gradients and analytic (default) or numerical Hessians with dipole derivatives; topology (bonds, hybridization, fragments, topology charges) set up once and kept fixed, or taken from a reference structure; method `"gfnff"` in Rust (`axtb_gfnff::Calculator`), CLI, Python and qccodec | done (`crates/gfnff`, mapping to xtb's Fortran in `docs/gfnff-spec.md`; see [GFN-FF](#gfn-ff)) | xtb 6.8.0 `--gfnff` (default and `xtb` conventions) and xtb 6.7.1 (`xtb-6.7.1`) on 269 structures each (H-Rn, charges -3...+3, metal complexes, zwitterions, HB/XB dimers, water clusters, crambin with H): topology identical (every list of xtb's `gfnff_topo`) except D5h ferrocenium (degenerate Hückel pair), EEQ charges ≤ 5e-12 e, energies (each term) ≤ 9e-13 Eh, gradients ≤ 1e-11 Eh/bohr; `xtb --gfnff --hess` free energies and ZPVE ≤ 5e-13 Eh (6.8.0; 6.7.1: 2e-11; numerical Hessian, same step), frequencies ≤ 0.3 cm⁻¹ with the analytic Hessian (xtb's finite-difference error); analytic Hessian vs finite differences ≤ 1e-8 Eh/bohr² incl. linear and planar structures; (H2O)1700 (5100 atoms), where xtb runs out of 10 GiB, in 15 s and 461 MiB for energy + gradient (1 thread) |
| Restarts (each SCF from the previous geometry's state; g-xTB in the default conventions, and the programs' `xtbrestart`/`gxtbrestart` on request), Mayer–Wiberg bond orders, per-spin orbital levels, HOMO–LUMO gap and the molecular quadrupole for every SCF method in Rust, Python, CLI and qccodec (`wavefunction` fields, `propagate_wfn`) | done (see [Restarts](../README.md#restarts-the-previous-geometrys-wavefunction) and [Bond orders](#bond-orders-orbital-levels-gap-and-quadrupole)) | g-xTB 2.0.1 and 1.1.0 restarted runs: energies ≤ 4.3e-11 Eh, same iterations (0–0.3 bohr displacements); bond orders vs the `wbo` files: 2.0.1 ≤ 9.4e-12, xtb 6.8.0 ≤ 7e-14, 6.7.1 ≤ 2e-9; gaps and quadrupoles of 2.0.1 and xtb 6.8.0 to the printed digits (11 molecules, closed and open shells, ions) |

## GFN2-xTB against xtb 6.8.0, xtb 6.7.1 and tblite

Validation against xtb 6.8.0: the reference matrix (all 991 cases, `summary.md`) has 0 failing
checks. 860 single-point xtb checks pass: energies ≤ 7.1e-12 Eh and gradients ≤ 2.2e-9
Eh/bohr in the categories run at xtb's tightest accuracy 1e-4 (elements, charges, spins,
open-shell metals, 483 ALPB/GBSA cases in all three reference states; the gradient difference
is xtb's convergence there), and ≤ 4.6e-10 Eh and 6.8e-7 Eh/bohr in `etemp_acc` (0–5000 K,
accuracies 1–1e-6; small-gap Fe4 and twisted ethylene at accuracy 1). The others are both
programs failing (12) and different SCF solutions of pathological cases (5: Cu5 at 0 K, `gh106`,
each confirmed by restarting each program from the other's moments). 59 of the 63 cases run
at equal accuracy take the same SCC iterations (the other four, small-gap SCFs at the
threshold, ±1), the failure behaviour is xtb's, and all 52 Hessian cases pass: `xtb --hess
--acc 0.001` against axtb's numerical Hessian with the same procedure ≤ 2.6e-9 Eh/bohr² (6.7.1:
1.3e-4, from its displacements at 0.3), against the analytic Hessian ≤ 2.6e-4 (xtb's
finite-difference error). The matrix against the g-xTB build (`summary-xtb-6.7.1.md`, the
`xtb-6.7.1` conventions) has 0 failing checks too.

**The 6.7.1 release itself** (`edcfbbe`, e.g. the copy that ships with ORCA 6.0) is not the
g-xTB build: its GFN2 gradient leaves out the H0/S term of AO pairs whose overlap vanishes by
symmetry, i.e. of atoms that share a Cartesian coordinate (fixed by xtb PR #1208, March 2025,
before the g-xTB build). Against it the `xtb-6.7.1` conventions agree in energies, charges and
dipoles, but 350 gradient checks of the reference matrix differ, by up to 3.3e-3 Eh/bohr
(guanidinium 2.7e-3, [Cu(NH3)4]²⁺ 6.5e-5; `summary-xtb-6.7.1-release.md`).

The reference matrix (`tests/reference_matrix/`, results in its `summary.md`) covers
elements 1-86, charges -3...+3, singlet to undectet spin states, all 25 ALPB and 14 GBSA
solvents in all three reference states, electronic temperatures 0-5000 K with accuracies
1-1e-6, Hessians, thread invariance, dispersion off, the CLI and failure behaviour. Its
`hessian_analytic` category (45 cases: closed and open shells, charges -3...+3, transition
metals, ALPB/GBSA in several solvents and states, 300-5000 K) checks the analytic Hessian in
all three conventions against tight response and numerical references and against
`xtb --hess` run with `$hess sccacc=0.001`.

### What xtb 6.8.0 changed for GFN2-xTB

For each behaviour of 6.7.1 that axtb reproduces, the table gives what 6.8.0 does (from the
v6.8.0 source) and what the 6.7.1 behaviour would cost against the 6.8.0 program:
`tests/xtb680_changes.py` runs axtb's 6.8.0 conventions with that one setting put back to 6.7.1's (`examples/conventions_ablation`) on ten
cases (water, caffeine, ferrocene, [Cu(NH3)4]²⁺, taxol, caffeine in ALPB/GBSA water in the
three reference states, compressed CsOH in both atom orders) at `--acc 1e-4` and `--acc 1`
(largest |ΔE| in Eh and max |ΔG| in Eh/bohr; `bench/results/xtb680_changes.json`). The 6.8.0
conventions themselves: ≤ 4.8e-13 Eh, ≤ 3.9e-14 Eh/bohr, the same iterations in all 20 runs.

| xtb 6.7.1 behaviour | in xtb 6.8.0 | 6.7.1's behaviour vs the 6.8.0 program |
|---|---|---|
| radius tables (atomic radii and the GFN2 CN radii; for D4 see the next row) converted with bohr = 0.52917726 Å | **fixed**: `convert.f90` takes mctc-lib's CODATA 2018 bohr, 0.529177210904492434 Å, so the tables equal tblite's to the last bit (only the operation order differs) | 1.2e-7 Eh (taxol; caffeine 3.2e-8), 4.3e-9 Eh/bohr |
| single-precision literals (and the older bohr) in the D4 coordination number and the EEQ start (grimme-lab/xtb#1506) | **fixed**: `_wp` literals (`disp/ncoord.f90`) | 6.1e-10 Eh, 7.3e-9 Eh/bohr |
| cutoffs: D4 two-body 72 bohr and three-body 40 bohr, hard; CN and repulsion 40 bohr | **changed** to tblite's: D4 50/25 bohr with 0.05 bohr smooth switches, D4 CN 30 bohr, GFN2 CN and repulsion 25 bohr | 9.4e-6 Eh, 3.1e-6 Eh/bohr (taxol, 30 bohr long; no effect on the smaller molecules) |
| integral screening max(20, 25 − 10 log₁₀ acc), overlaps below 1e-8·acc zeroed | **changed** to tblite's (25 for acc ≤ 1, no overlap neglect, onsite overlaps from the contracted functions) | 3.8e-11 Eh |
| dead small-gap Broyden damping (`iter == 0` in a loop from 1, #1506) | **removed** | — |
| stale `r1` in the Born-radius integral (#1504) | **fixed**: `born.f90` sets `r1` for every pair | compressed CsOH: order dependence 2.9e-3 Eh → 0 (tblite still 2.9e-3, tblite/tblite#430) |
| ALPB/GBSA `reference` state with ln(ρ·89.24/M²) (#1505) | **fixed**: the solvent's mass density | 5.0e-3 Eh; 6.8.0 − tblite in `reference`: −5.0e-3 → 9.9e-9 Eh (xtb's kB and 24.79 L) |
| SCC 1, EEQ start | **unchanged** (with 6.8.0's radii) | — |
| SCC 2, residual sqrt(Σdq²/N_atoms) < 1e-4·acc | **changed** to tblite's RMS < 2e-5·acc | 1.6e-8 Eh, 9.4e-6 Eh/bohr at acc 1; one iteration fewer in 4 (acc 1e-4) and 5 (acc 1) of the 10 cases |
| SCC 3, one more iteration after convergence | **removed** | 4.9e-9 Eh, 6.2e-6 Eh/bohr; +1 iteration everywhere |
| SCC 4, unscaled quadrupoles in the mixer | **changed** to tblite's √2 | 1.0e-10 Eh, 3.4e-7 Eh/bohr |
| SCC 5, neglected multipole integrals | **removed** (threshold 0) | 1.1e-10 Eh, 6.3e-8 Eh/bohr |
| SCC 6, gradient from the moments mixed after the last iteration | **changed**, differently from tblite: output moments, AES potentials recomputed from them, isotropic shifts of the last input | 1.4e-6 Eh/bohr; tblite's choice (all last-input potentials) would be 3.9e-6 Eh/bohr off |
| single atoms: simple damping instead of Broyden (`broy = mol%n > 1`) | **changed**: Broyden for atoms too | iteration counts only (C atom at acc 1: 45 in the 6.7.1 program, 4 in `xtb-6.7.1`, which does not reproduce this; energies 4e-13 Eh apart); 6.8.0: 3 and 3 |
| aufbau occupations at ≤ 0.1 K | **unchanged** (`if (et > 0.1)`) | O2 singlet at 0.1 K: −2.52e-3 Eh vs tblite in both |
| invalid electron counts accepted (more unpaired than electrons, charge above the electron count, fractional charge) | **unchanged**: the matrix's failure table is identical | — |
| `--acc` below 1e-4 clamped to 1e-4 (with a warning), negative `--etemp` accepted silently (aufbau) | **unchanged** | — |
| `--acc` does not reach the displaced SCFs of `--hess` (only `$hess sccacc`) | **fixed** (xtb PR #1500): `--acc` reaches them unless `$hess sccacc` is given; plain `--hess` now runs at accuracy 0.2 | see [Hessians](#hessians) |
| `--hess` rotates the molecule slightly (`rotmol`) and projects about the geometric centre | **changed** (PRs #1426, #1435): no rotation; the `hessian` file is projected in mass-weighted coordinates (also for linear molecules) and transformed back | the reference matrix compares accordingly (`xtb_projected`); the `xtb` analysis conventions project as 6.8.0 (see [Vibrational analysis](#vibrational-analysis-and-thermochemistry)) |

### xtb's SCC loop

The `xtb-6.7.1` and `xtb` conventions run the programs' own SCC loops (README, "xtb's SCC
loop"). The result, per program call at `--acc 1` against the 6.7.1 program (water, caffeine, acetate,
water cation, Na⁺ acetate, an SN2 path image; GFN2 and GFN2/ALPB water; the GFN2 halves of
`tests/solvation_composite.py` in program mode): energies ≤ 5.5e-10 Eh, the static difference
the 6.7.1 conventions also have at tight accuracy (caffeine 5.5e-10 Eh, mostly D4). Against
xtb 6.8.0 (`conventions="xtb"`, `tests/validate_xtb.py`, the 13 test structures, at `--acc 1`
and `--acc 1e-4`, gas phase and ALPB water): energies ≤ 5.0e-13 Eh, gradients ≤ 3.6e-14
Eh/bohr, the same SCC iterations; only the pathological `gh106` differs (166 vs 161
iterations at `--acc 1`, 6.6e-10 Eh; 194 vs 193 at `--acc 1e-4`; in ALPB water xtb itself
stops with an error on it). Molden files at the default accuracy match the program's
orbitals as well as at `--acc 1e-4` (see [Molden files](#molden-files)).

On the reference matrix against xtb 6.7.1 (the g-xTB 2.0.1 build; all 991 cases, 0 failing
checks, `tests/reference_matrix/summary-xtb-6.7.1.md`), with the 6.7.1 SCC loop the largest
|ΔG| in `etemp_acc`, on the cases where both programs reach the same solution, is 1.3e-7
Eh/bohr at accuracy 1 and 5.7e-9 at 0.01. Two pathological SCFs take another path, and neither is
reproducible at this level:
- **Cu5 at 0 K.** It has two solutions 3.0e-4 Eh apart with degenerate frontier orbitals, and
  which one each program reaches depends on accuracy, in xtb too. At accuracies ≤ 0.01, axtb
  in xtb conventions now lands on the other solution than xtb; the restart diagnosis confirms
  both.
- **mbheavy-113.** This random heavy-element molecule has a 0.07 eV gap and residuals above 1
  for 100 iterations. The iterations agree to 1e-7 Eh through iteration 13 and then diverge:
  xtb stops unconverged at 250, axtb converges at 189.

### SCF convergence

On the hard-SCF set (`tests/scf_hard_set.py`: 104 cases of stretched bonds, reaction paths,
metal clusters, small gaps, open shells and transition-metal complexes at 300 K and 0 K,
accuracy 1e-4, 250 iterations per attempt; `bench/results/scf_hard.json`) Broyden fails on 20
cases, xtb 6.7.1 and tblite 0.8.0 on 19 each, DIIS and `auto` on 10. `auto` gives Broyden's
results on the 84 cases Broyden converges and rescues 10 of its 20 failures (stretched HF, H2O
and F2, two random molecules) with DIIS, after up to 250 Broyden iterations. The remaining ten
failures are 0 K cases whose solution has degenerate, fractionally occupied frontier orbitals
at any temperature above 0 K (Fe4, Cr2, twisted ethylene, square cyclobutadiene, N2 at 2.5-4
Å): no aufbau fixed point exists, so no mixer converges them without changing the occupations.
On the reference matrix (against xtb 6.8.0) the default conventions' fallback converges three
random molecules that Broyden in the xtb conventions does not (mbheavy-101, -112, -113). DIIS
alone is about as fast as Broyden (same median iterations on both sets) but lands on another
solution in some cases with several (e.g. the pathological `gh106`).

## Restarts

Along paths, default conventions, one thread (`python tests/gxtb_parity_features.py`,
`bench/results/gxtb_parity_features.json`; axtb 3499913 on the i9-11900H, started at a 1-minute
load below 1.5, recorded per row). Random walks move every coordinate by N(0, σ) bohr per step;
errors are against fresh SCFs at accuracy 1e-4 over the restarted points:

| molecule (points) | method | σ (bohr) | SCF iterations per point, fresh → restart | SCF time (s) | time of all points (s) | max error vs converged, fresh / restart: energy (Eh); gradient (Eh/bohr) | load |
|---|---|---|---|---|---|---|---|
| caffeine (30) | g-xTB 2.0.1 | 0.005 | 9.0 → 6.7 | 0.13 → 0.10 | 0.39 → 0.36 | 9e-08 / 1e-08; 7e-05 / 4e-05 | 1–1 |
| caffeine (30) | g-xTB v1.1 | 0.005 | 11.0 → 7.9 | 0.24 → 0.17 | 0.92 → 0.85 | 7e-11 / 4e-10; 3e-06 / 4e-06 | 1–1 |
| caffeine (30) | GFN2-xTB | 0.005 | 11.0 → 6.5 | 0.12 → 0.07 | 0.17 → 0.12 | 6e-10 / 4e-09; 3e-06 / 7e-06 | 1–1 |
| caffeine (30) | g-xTB 2.0.1 | 0.02 | 9.0 → 8.0 | 0.14 → 0.13 | 0.42 → 0.41 | 1e-07 / 2e-09; 8e-05 / 1e-05 | 1–1 |
| caffeine (30) | g-xTB v1.1 | 0.02 | 11.0 → 9.0 | 0.26 → 0.21 | 0.99 → 0.95 | 9e-11 / 2e-10; 3e-06 / 4e-06 | 1–1 |
| caffeine (30) | GFN2-xTB | 0.02 | 11.0 → 7.7 | 0.12 → 0.08 | 0.17 → 0.13 | 8e-10 / 4e-09; 5e-06 / 8e-06 | 1–1 |
| taxol (12) | g-xTB 2.0.1 | 0.005 | 9.0 → 6.8 | 1.62 → 1.24 | 3.39 → 3.01 | 1e-07 / 3e-08; 4e-05 / 5e-05 | 1–1 |
| taxol (12) | g-xTB v1.1 | 0.005 | 11.0 → 9.0 | 4.27 → 3.47 | 8.63 → 7.88 | 9e-11 / 5e-10; 2e-06 / 7e-06 | 1–1 |
| taxol (12) | GFN2-xTB | 0.005 | 11.9 → 6.9 | 1.18 → 0.68 | 1.39 → 0.89 | 1e-08 / 2e-08; 7e-06 / 7e-06 | 1–1 |
| taxol (12) | g-xTB 2.0.1 | 0.02 | 9.0 → 8.7 | 1.64 → 1.58 | 3.43 → 3.35 | 2e-07 / 1e-08; 4e-05 / 3e-05 | 1–1 |
| taxol (12) | g-xTB v1.1 | 0.02 | 11.0 → 10.2 | 4.30 → 3.99 | 8.72 → 8.40 | 5e-10 / 3e-10; 6e-06 / 5e-06 | 1–1 |
| taxol (12) | GFN2-xTB | 0.02 | 12.0 → 8.7 | 1.13 → 0.82 | 1.33 → 1.02 | 1e-08 / 2e-08; 6e-06 / 8e-06 | 1–1 |

Through the codec with qccompute (`propagate_wfn` off → on; SCF iterations of all steps):

| driver | method | steps | SCF iterations | final energy difference |
|---|---|---|---|---|
| geomeTRIC minimization, caffeine (0.1 bohr from its minimum) | g-xTB 2.0.1 | 23 → 25 | 207 → 154 | 2.8e-7 Eh |
| geomeTRIC minimization, caffeine | g-xTB v1.1 | 49 → 49 | 538 → 343 | 2.8e-9 Eh |
| geomeTRIC minimization, caffeine | GFN2-xTB | 14 → 14 | 154 → 93 | 1.0e-10 Eh |
| Sella saddle point (exact axtb Hessians), SN2 Cl⁻ + CH3Cl (`bench/ts_example.py`'s guess) | g-xTB 2.0.1 | 5 → 5 | 40 → 35 | 3.5e-9 Eh |
| Sella IRC from that saddle point, both directions | g-xTB 2.0.1 | 76 → 76 | 676 → 383 | 5.7e-9 Eh |

Restarts save 3–42 % of the SCF iterations per point and 4–42 % of the SCF time at these step
sizes (most for GFN2-xTB and the smaller steps), and 43 % of the iterations along the IRC. A whole
g-xTB step gains less: the SCF is a quarter to half of an energy + gradient (caffeine 34 %, taxol
48 % for 2.0.1, without the setup of the charge-adapted basis and the integrals), so the time of
all points drops by 2–11 %; GFN2-xTB, whose SCF is three quarters of the time or more, gains
23–36 %. The two sequences of a row take turns point by point, so both see the
same load. Through qccompute, wall times are dominated by the drivers and the machine's load
and are in the results only for completeness.

**Different SCF solutions.** Where g-xTB has several SCF solutions, a restart follows the one it
started on, where a fresh start may land on another, so the two can differ by more than the
SCF thresholds (`solutions` in the results: bond-stretch scans 0.8–4 Å in 0.1 Å steps, both
directions, and 15-step random walks of σ 0.02 bohr, at 0 K and 3000 K, v1.1 and 2.0.1):

- none on ordinary molecules: ethane's C–C scan, the SN2 path image, [Cu(NH3)4]²⁺, [FeCl4]⁻,
  twisted ethylene, Cr2 (≤ 5e-8 Eh, 2.0.1);
- stretched bonds: F2 from 1.7 Å, HF and H2O from 2.6 Å, N2 at 3000 K from 1.9 Å (none at
  0 K). A restart that comes in from the dissociated side stays on its solution until it
  collapses: N2 at 3000 K (2.0.1), recompressed from 4 Å, is up to 0.30 Eh above the fresh start
  before it returns to it at 1.8 Å; F2 at 0 K up to 0.32 Eh, back at 1.6 Å. Going out from the
  bonded side the restart is often lower (HF v1.1 −0.12 Eh, N2 at 3000 K −0.08 Eh beyond
  3.5 Å): it keeps the lower solution that a fresh EEQ-BC start misses;
- metal clusters (Fe4, Cu5, Ni4, Mn2O, Pt3 v1.1, Fe²⁺ aqua v1.1): differences of 1e-5 to 9e-2 Eh,
  either sign, at 0 K and 3000 K.

The programs behave the same way within a run (optimization or MD steps carry the state). For
dynamics or paths through bond breaking and forming, compare with fresh starts (`guess="eeq-bc"`)
at the critical points; neither start finds the lowest solution everywhere.

The thread count does not change the sequences: the same restart sequences of caffeine and taxol
on 1 and 4 threads give identical energies, gradients and iterations (in general results may
differ between thread counts by round-off; see [Design rules](../README.md#design-rules)).

## Hessians

When comparing with `xtb --hess`: xtb 6.7.1 runs the displacement SCFs of a GFN2 `--hess` at SCC
accuracy 0.3 regardless of `--acc` (only `$hess sccacc=` changes it), which leaves 2e-4–6e-4
Eh/bohr² noise in its Hessian. xtb 6.8.0 (PR #1500) passes `--acc` to the displacements unless
`$hess sccacc` is given, and a plain `--hess` runs the reference and the displacements at
accuracy 0.2. On caffeine (`tests/xtb_hess_acc.py`,
`bench/results/xtb_gfn2_hess_acc_6.8.0.json`; xtb's Hessian against axtb's analytic one,
projected as each version projects): 6.8.0 `--acc 0.001` 3.7e-5 Eh/bohr² (6.7.1: 2.2e-4, the
same as `--acc 1`), `$hess sccacc=0.001` 3.7e-5 in both, plain `--hess` 2.7e-4 (6.7.1: 2.2e-4);
an explicit `--acc 1.0` is now worse than before, 1.3e-3 (lowest mode −16.6 instead of −27.6
cm⁻¹), since its displacements run at 1.0 instead of 0.3. 6.8.0 also no longer rotates the
molecule before `--hess`, and projects the Hessian it writes in mass-weighted coordinates.

Timings on the 64-core AMD Threadripper PRO 9985WX node, in a 16-CPU pod of the shared node
(`bench/hessian_benchmark.py`, axtb 3499913, best of 2, all results in
`bench/results/hessian.json`; `xtb --hess` is the g-xTB 2.0.1 build of xtb 6.7.1; seconds, 1 / 8
threads):

| molecule | analytic | response | numerical | `xtb --hess` |
|---|---|---|---|---|
| caffeine (24) | 0.019 / 0.016 | 0.12 / 0.029 | 0.48 / 0.074 | 1.02 / 0.18 |
| tyr-xab (33) | 0.034 / 0.024 | 0.25 / 0.051 | 1.00 / 0.14 | 2.32 / 0.55 |
| taxol (113) | 1.44 / 0.52 | 9.8 / 1.5 | 39.7 / 5.2 | 93.5 / 12.7 |
| taxol, ALPB water | 1.29 / 0.42 | 9.1 / 1.4 | 41.7 / 5.3 | 103 / 15.7 |

Analytic vs a Richardson-extrapolated tight numerical Hessian: ≤ 1.3e-8 Eh/bohr² for all
gas-phase cases (frequencies identical to 0.001 cm⁻¹); the default numerical Hessian is off
by 2–4e-5 (up to 0.7 cm⁻¹).

## Vibrational analysis and thermochemistry

Dipole derivatives (IR intensities): they agree with central differences of converged dipoles to
≤ 1e-7 e (analytic and response; numerical ≤ 3e-7 e; closed and open shells, anions; caffeine
6e-7 e against a 0.001 bohr numerical Hessian). xtb's printed IR intensities carry the noise of
its displacement SCFs (accuracy 0.3): on the program's own Hessian, axtb's dipole derivatives
give intensities up to 43 % (2.6 km/mol) different on caffeine, 0.07 % (0.012 km/mol) once xtb
runs them at `$hess sccacc=0.001`. The g-xTB 2.0.1 program at `--acc 1e-4` prints intensities
that axtb's dipole derivatives reproduce on its Hessian to ≤ 0.028 km/mol (≤ 0.03 %).

On the same Hessian `axtb` and `xtb` differ by ≤ 0.1 cm⁻¹ in frequencies and by ≤ 4.0e-6 Eh in
G(RRHO) (caffeine; ≤ 2.1e-6 Eh for the other six), almost all from the masses (H 1.0080 vs
xtb's 1.00794).
`tests/vibrational_check.py` runs `xtb --hess` (6.8.0: `bench/results/vibrational_6.8.0.json`;
the 6.7.1 g-xTB build: `bench/results/vibrational.json`) and `xtb --gxtb --hess --acc 1e-4` (the
g-xTB build) on caffeine, CO2, HCN, benzene, NH3, CH4 and the SN2 transition state and feeds
their `hessian` files to axtb's analysis in that version's conventions (6.7.1: at its slightly
rotated coordinates):

| | frequencies | ZPVE, H(T), T·S, G(RRHO), G | point group, symmetry number |
|---|---|---|---|
| GFN2-xTB, xtb 6.8.0's Hessian (`xtb`) | ≤ 5.0e-5 cm⁻¹ (4 printed decimals) | ≤ 1.4e-10 Eh (caffeine; the others ≤ 1.3e-11) | identical: c1 1, din 2, cin 1, d6h 12, c3v 3, td 12, d3h 6 |
| GFN2-xTB, xtb 6.7.1's Hessian (`xtb-6.7.1`) | ≤ 5.0e-5 cm⁻¹ | ≤ 7.7e-11 Eh | identical |
| g-xTB 2.0.1, the program's Hessian | ≤ 5.0e-5 cm⁻¹ | ≤ 2.7e-11 Eh | identical |
| g-xTB 2.0.1, axtb's analytic Hessian through `xtb thermo` | — | ZPVE, G(RRHO) ≤ 8e-12 Eh (H, T·S to its 6 printed digits) | identical |

The `hessian` file holds 10 decimals, which limits the "same Hessian" comparison: `xtb thermo`
of 6.8.0 on the file of its own `--gfn 2 --hess` of caffeine prints a G(RRHO) 9.6e-11 Eh away
from the `--hess` printout, and axtb's `xtb` analysis of that file reproduces the `xtb thermo`
printout to 2.4e-13 Eh (the `xtb-6.7.1` analysis: 2.6e-10). Away from minima (reference-matrix
Hessian cases with gradients up to 4e-2 Eh/bohr: acetate, [Mg(H2O)6]²⁺ in ALPB water, triplet
CH2, CO3²⁻), axtb's numerical Hessian (`--acc 0.001`) analysed in `xtb` conventions reproduces
6.8.0's `--hess --acc 0.001` G(RRHO) to ≤ 1.4e-12 Eh.

With axtb's own Hessians the differences are those of the Hessians: GFN2-xTB (analytic vs xtb's
finite differences) ≤ 0.46 cm⁻¹ and ≤ 2.3e-5 Eh in G(RRHO) (caffeine, 6.7.1; 6.8.0, whose plain
`--hess` runs its SCFs at accuracy 0.2: ≤ 0.84 cm⁻¹ and 3.2e-5 Eh); g-xTB 2.0.1 (analytic vs
the program at `--acc 1e-4`) ≤ 0.35 cm⁻¹ and ≤ 5.2e-6 Eh. Caffeine's g-xTB G(RRHO) is 0.150578
Eh with axtb's analytic Hessian, 0.150572 with the program's at `--acc 1e-4` and 0.151263 at
its default `--acc 1` (inaccurate Hessian, +0.43 kcal/mol). The analysis takes 0.5 ms for
caffeine and 19 ms for taxol on one thread of the i9-11900H, symmetry detection included
(`axtb.vibrational_analysis` on GFN2-xTB analytic Hessians, best of 7, axtb 3499913;
`bench/results/final-laptop/vibrational_analysis_time.json`).

## g-xTB at finite electronic temperature

`etemp` for g-xTB v1.1 and 2.0.1 (README, "g-xTB"). Against the 2.0.1 program at
300/1000/3000/5000 K (57 molecules, Fe4, stretched N2; the solvation composite's g-xTB retry
ladder runs 1000 and 3000 K): energies ≤ 1.8e-12 Eh ((H2O)100 1.1e-11), same SCF iterations,
analytic gradients ≤ 5e-11 Eh/bohr; against 1.1.0: energies ≤ 2e-11 Eh, same iterations. A few
slowly converging SCFs (Fe4, TmH3 at 3000/5000 K) end where rounding decides the last digits, or
whether they converge within 250 iterations at all, in either program (spec §4.17.1).
Finite-temperature Hessians agree with tight references to ≤ 3e-9 Eh/bohr² (spec §11.6);
`bench/results/gxtb_etemp.json` from `python tests/gxtb_etemp_reproduction.py`.

## g-xTB in implicit solvent

**Validation** (`python tests/solvation_composite.py` →
`bench/results/solvation_composite.json`; axtb 3499913 on the i9-11900H, one thread, xtb 6.7.1
g-xTB build). The cases are water, caffeine, acetate, the water radical cation, an Na⁺ acetate
contact ion pair, an SN2 path image (the `bench/ts_example.py` guess), F2 and HF at 3.0 Å
(whose GFN2 pairs need the 1000 K and 5000 K retries), and taxol, all in ALPB water. Each is
compared with the three xtb programs, run with the same retry ladder. Two rows compare with the
composite's earlier Python implementation in the private mepd workflow (`SolvationCorrection`),
which cannot be rerun outside it.

| check (`conventions="gxtb"`) | result |
|---|---|
| energy vs the sum of the three programs | ≤ 9.7e-11 Eh (taxol; ≤ 4.7e-11 for the others) |
| gradient vs the programs | ≤ 5.3e-10 Eh/bohr (F2 at 3.0 Å; taxol 2.8e-10, ≤ 6.7e-11 for the others) |
| retry rung | the same as the programs' (F2: 1000 K, HF: 5000 K, both halves) |
| mepd's own `SolvatedEngine`/`SolvationCorrection` vs the programs | ≤ 9.1e-12 Eh (mepd reads 11 decimals) |
| analytic composite Hessian vs central differences of the composite gradient (both conventions) | ≤ 5.6e-8 Eh/bohr² (caffeine; ≤ 1.1e-8 for the others), frequencies ≤ 2.5e-4 cm⁻¹ |

The Hessian check takes the better of the steps 1e-4 and 2e-5 bohr. The ALPB surface term is
C¹ but not C², and at 1e-4 caffeine's solvated half crosses its switching kinks (3.2e-6
Eh/bohr², all from the GFN2/ALPB part; see [Hessians](#hessians)).

Time for one composite gradient (best of 3; the programs' time includes process start and
retries; axtb in default conventions, new calculator / next geometry on the same calculator):

| molecule | 3 xtb programs | mepd `SolvationCorrection` path | axtb composite |
|---|---|---|---|
| water | 26 ms | 28 ms | 1.1 / 0.9 ms |
| caffeine | 103 ms | 103 ms | 27 / 19 ms |
| taxol (113 atoms) | 1.46 s | 1.46 s | 0.58 / 0.38 s (g-xTB 0.25 s, GFN2 halves 0.14 + 0.14 s) |
| F2 at 3.0 Å (one retry) | 118 ms | 204 ms | 0.75 ms (program mode) |

Small molecules gain most, since every program call costs about 9 ms to start. For large ones
the g-xTB part takes the largest share (taxol: 0.25 of 0.58 s). Hessians are
analytic, where the three-program workflow differentiates the composite gradient numerically (6N gradients of three
programs each). The failing rungs of a retry ladder cost about as much as in the programs (HF:
40 s vs 47 s), because Broyden mixing with full history is O(n³) per iteration for 1000 iterations.
In the default conventions the GFN2 halves are not the programs: the `auto` mixer converges
stretched HF at 300 K without retries, and that ΔG_solv differs from the programs' 5000 K value by
0.029 Eh. On the other molecules the default and program modes differ by ≤ 1.3e-8 Eh for the
small ones (F2 6.8e-10) and 5.1e-7 Eh for taxol: the corrected constants, solvation and cutoffs
of the default conventions.

## GFN-FF

Validation (`tests/gfnff_reproduction.py --xtb <binary>`; `bench/results/gfnff_reproduction_6.8.0.json`
and, for 6.7.1, `bench/results/gfnff_reproduction.json`):

| | against xtb 6.8.0 (`xtb`) | against the 6.7.1 g-xTB build (`xtb-6.7.1`) |
|---|---|---|
| 269 structures: topology lists, CLI bonds | identical in 268 | identical in 268 |
| topology parameters, EEQ charges | ≤ 5.0e-11, ≤ 2.3e-12 e | ≤ 5.0e-11, ≤ 4.7e-12 e |
| energies (every term), gradients | ≤ 7.0e-13 Eh (terms ≤ 8.4e-13), ≤ 4.3e-12 Eh/bohr | ≤ 8.4e-13 Eh (terms ≤ 5.5e-13), ≤ 9.3e-12 Eh/bohr |
| 6 structures displaced and evaluated with the original's topology (`topology_positions`; xtb reading `gfnff_topo`) | ≤ 5.0e-13 Eh | ≤ 4.9e-13 Eh |
| `--hess` (8 structures; 6.8.0: also linear CO2 and HCN): axtb numerical Hessian in the version's analysis | frequencies ≤ 0.005 cm⁻¹ (xtb prints 0.01), IR ≤ 5e-6 km/mol, ZPVE and free energy ≤ 4.8e-13 Eh | frequencies ≤ 0.005 cm⁻¹, IR ≤ 1.3e-4 km/mol (the water cluster and the glycine zwitterion with water; the others ≤ 5.7e-6), ZPVE and free energy ≤ 2e-11 Eh |
| analytic Hessian | ≤ 0.27 cm⁻¹ (xtb's finite-difference error) | ≤ 0.3 cm⁻¹ |

D5h ferrocenium is the exception in both (its degenerate Hückel pair, `docs/gfnff-spec.md` §7).
The `axtb` and `xtb` conventions give the same force field. On the
acetone–water dimer of grimme-lab/xtb#1516 (water H exactly on the C=O axis), where xtb 6.7.1's
`--hess` crashed and 6.8.0's runs, the analytic Hessian is undefined (the HB term's R–C=O···H
torsions are undefined on the axis, where the energy jumps) and refused with a numerical error
that names the hydrogen bond and points to the numerical Hessian, which reproduces 6.8.0's (whose lowest mode there is −358 cm⁻¹). xtb 6.8.0 still allocates the
N_AB² × N_H hydrogen-bond array (grimme-lab/xtb#1515; 8.9 GiB peak for 1000 waters, 6.7.1:
10.0 GiB) and still has the energy jump at a linear C=O···H hydrogen bond (#1516: −6.87e-4 Eh
0.001 Å off the axis, as in 6.7.1).

The official 6.8.0 binary is built for two instruction sets and picks one at run time (Intel's
`-ax` dispatch: an AVX2 path, which the validation above ran on an Intel CPU, and a baseline
SSE2 path without fused multiply-adds, which Intel's dispatchers use on non-Intel CPUs). The
tie-breaking distances follow the AVX2 path; on a CPU that runs the baseline path, xtb 6.8.0
itself can order exactly equidistant neighbors differently (not tested here).

### Time and memory

Memory grows linearly with the number of atoms except for the dense EEQ matrix ((N +
fragments)², 1.4 GB at 9900 water atoms); xtb keeps several N x N arrays and an NAB² x NH
logical array of the hydrogen-bond atoms in every energy evaluation (7.5 GiB at 3000 water
atoms, 37 GiB at 5100). Wall time and peak memory (`/usr/bin/time -v`) on the i9-11900H, axtb
3499913 in the `xtb-6.7.1` conventions against xtb 6.7.1 with one thread and its address space
capped at 10 GiB (started at a 1-minute load below 1.5, which these runs raised to at most 4;
recorded per run in `bench/results/gfnff_reproduction.json`, from `python
tests/gfnff_reproduction.py --part scaling --xtb <the g-xTB build>`; xtb 6.8.0 still allocates
the hydrogen-bond array below, 8.9 GiB peak for 1000 waters,
`bench/results/gfnff_xtb680.json`):

| system (atoms) | xtb energy + gradient | axtb energy + gradient, 1 / 8 threads | axtb analytic Hessian, 1 / 8 threads | `xtb --hess` |
|---|---|---|---|---|
| (H2O)100 (300) | 0.08 s, 64 MiB | 0.03 / 0.03 s, 11 MiB | 0.33 / 0.09 s, 28 MiB | 40 s, 114 MiB |
| crambin (640) | 0.46 s, 134 MiB | 0.10 / 0.08 s, 15 MiB | 1.6 / 0.35 s, 67 MiB | 209 s, 334 MiB |
| (H2O)333 (999) | 0.77 s, 542 MiB | 0.33 / 0.24 s, 39 MiB | 3.3 / 0.85 s, 171 MiB | — |
| (H2O)1000 (3000) | 13 s, 9.0 GiB | 3.8 / 1.8 s, 190 MiB | 37 / 8.0 s, 1.1 GiB | — |
| (H2O)1700 (5100) | out of memory (> 36 GiB) | 14 / 4.8 s, 461 MiB | 134 / 28 s, 3.1 GiB | — |
| crambin in water (5311) | out of memory (> 33 GiB) | 20 / 6.1 s, 458 MiB | 149 / 32 s, 3.3 GiB | — |
| (H2O)3300 (9900) | out of memory (> 267 GiB) | 89 / 23 s, 1.5 GiB | — | — |

Where both finish, energies agree to 1.3e-11 Eh and gradients to 5.3e-14 Eh/bohr. Above ~3000
atoms the dense EEQ solves (two in the setup, one per energy) dominate axtb's time.

### What xtb 6.8.0 changed in GFN-FF

xtb 6.8.0 (`4b8c987`) and the 6.7.1-based g-xTB build (`26dd68d`) compute GFN-FF differently in
five places; four come from the source (two in `src/gfnff`, two in code GFN-FF shares with the
rest of xtb), one from the compiler. Each is a setting of the conventions' table (crate-private
`ConventionParams` in `crates/gfnff/src/conventions.rs`), and
the lanthanide reparametrization and actinide extension (PR #1103) and the other GFN-FF PRs of
the 6.8.0 release notes up to #1362 are already in the g-xTB build:

| | xtb 6.7.1 (g-xTB build) | xtb 6.8.0 | effect (6.7.1's behaviour against the 6.8.0 program, 269 structures) |
|---|---|---|---|
| covalent radii `covalentRadD3 · aatoau · 4/3` (CN, HB CN, bend and torsion damping) | bohr 0.52917726 Å | mctc-lib's CODATA 2018 bohr (PR #1478, all of xtb) | ~1e-7 relative in every CN; energies ~1e-9 Eh |
| pi bonds in the bond and torsion setup | Hückel bond order > 0 | > 1e3·ε (PR #1494) | round-off bond orders (~1e-15) no longer make pi-bond parameters |
| `nbondmat`: neighbors that do not list the atom back (one-sided bonds, e.g. of metals) | removed by shifting the original list once per neighbor, which drops the wrong entries for several | all removed (PR #1494) | bond counts between atom pairs (1-2/1-3/1-4 scaling) of such atoms |
| Fermi smearing of the Hückel step (4000 K, `fermismear`) | Fermi level to 1e-9 electrons | to min(√ε, 1e5·ε) = 2.2e-11 electrons, derivative f(1−f)/kT, clamped beyond 50 kT (PR #1474, shared with GFN2-xTB) | pi bond orders up to ~1e-9, topology parameters up to 1e-9, energies up to 5e-11 Eh |
| `NORM2` of the distances whose order decides between equidistant neighbors (out-of-plane terms, `jth_nb`, `nn_nearest_noM`) | ifort's fused multiply-adds | ifx's (oneAPI 2025; found in the binary's AVX2 code path) | the out-of-plane term of N in `el-07-N-alt` (an NH3-like centre with three equal bonds): 2.2e-3 Eh/bohr |

The rest of the GFN-FF changes in 6.8.0 leave molecules' results alone: printout switches
(#1438), `gfnff_topo` only written for restarts (#1467; xtb still writes it by default), heap
instead of stack arrays and OpenMP reductions as critical sections (#1428), the bond list of the
harmonic 2020 variant (#1425), periodic-only code and the fragmentation of the optimizer's
model Hessian. Together, 6.7.1's force field (the `xtb-6.7.1` conventions) differs from the 6.8.0
program on 206 of the 268 structures by more than 1e-10 Eh (median 1.4e-9, largest 9.0e-7 Eh),
charges ≤ 3.6e-7 e, with one torsion list different (`tests/gfnff_reproduction.py --conventions
xtb-6.7.1 --part accuracy` against xtb 6.8.0 → `bench/results/gfnff_xtb680.json`).

## Molden files

`tests/molden_check.py` reads both kinds of file with an independent reader (Molden's solid
harmonics written out as polynomials, its own Gaussian overlap integrals) and compares the
orbitals through the overlaps of the two files' bases (`bench/results/molden.json`; 1 − |⟨i|i⟩|
per non-degenerate orbital, 1 − the smallest singular value per degenerate set and for the
occupied space, values at 2000 random points, orbital energies), with each program in the
conventions that reproduce it:

| program | molecules | 1 − \|⟨i\|i⟩\| | occupied space | orbital energies (Eh) |
|---|---|---|---|---|
| g-xTB 2.0.1 (`xtb --gxtb --molden`) | water, caffeine, O2 triplet, [Cu(NH3)4]2+, ferrocene, UF6, GdCl3 (octet), CeCl3 (doublet) | ≤ 1.1e-9 | ≤ 6.5e-11 | ≤ 3.3e-12 |
| g-xTB 1.1.0 (`gxtb -molden`) | water, caffeine, O2 triplet, [Cu(NH3)4]2+, ferrocene, CeCl3 | ≤ 6.1e-11 | ≤ 6.5e-11 | ≤ 6.4e-8 |
| g-xTB 1.1.0 | UF6 | 1.1e-7 | 1.1e-10 | 8.6e-6 |
| GFN2-xTB (`xtb --gfn 2 --molden`), default accuracy | water, caffeine, O2 triplet, [Cu(NH3)4]2+, ferrocene | ≤ 1.1e-10 | ≤ 2.9e-11 | ≤ 3.0e-10 |
| GFN2-xTB, `--acc 1e-4` (both) | caffeine, ferrocene | ≤ 1.0e-10 | ≤ 1.7e-11 | ≤ 7.0e-10 |

The remaining differences are SCF convergence (GFN2-xTB: the g-xTB build's xtb 6.7.1 and axtb's
`xtb-6.7.1` conventions). axtb's GFN2 SCF in those conventions follows the program's SCC loop
(see [xtb's SCC loop](../README.md#xtbs-scc-loop-conventionsxtb-and-xtb-671)), so at the default
accuracy it agrees as closely as at `--acc 1e-4`; with tblite's loop it would agree only to the
SCF thresholds. For UF6 axtb's v1.1 SCF
takes 40 iterations where the 1.1.0 program takes 50 (total energies 1.1e-10 Eh apart): in
iteration 2 both fill two of three exactly degenerate orbitals, and which two-dimensional
subspace each eigensolver returns is round-off, so the paths part there and meet again at the
same state within the SCF thresholds (`docs/gxtb-v1.1-spec.md` §4.17). Every file, axtb's
spherical and Cartesian ones and the programs', reads back with orthonormal occupied orbitals
(C^T S C = 1 to ≤ 6e-11 for axtb's files). The orbital viewer of the private mepd workflow
(`orbitals.py`; not reproducible outside it) reads only Cartesian files: it reads axtb's
`molden="cartesian"` files with its `terachem` settings (normalized primitives, components
normalized on their own), and the HOMO density on its grid then equals the one from the
program's file read with its `gfn2`/`gxtb` settings (to ≤ 2.2e-7 relative for g-xTB 2.0.1; for
GFN2-xTB ≤ 7e-7 at the default accuracy and 6e-7 at `--acc 1e-4`).

## Bond orders, orbital levels, gap and quadrupole

Agreement with the programs (`python tests/gxtb_parity_features.py`,
`bench/results/gxtb_parity_features.json`): water, water rotated by 45°, caffeine, acetate,
[ZnCl4]²⁻, ferrocene, [Cu(NH3)4]²⁺ (doublet), O2 (triplet), NO2 (doublet), CH2 (triplet) and
taxol, each program on the same bohr coordinates, axtb in the conventions that reproduce it:

| program | conventions | bond orders (`wbo` file) | gap (printed eV) | quadrupole (printed) |
|---|---|---|---|---|
| g-xTB 2.0.1, `xtb --gxtb --quadrupole` | `gxtb` | ≤ 9.4e-12 | ≤ 3.7e-11 eV | ≤ 4.9e-5 (4 decimals) |
| g-xTB 1.1.0 | `gxtb` | ≤ 4.2e-4 (3 printed decimals; 8 molecules) | ≤ 2.2e-7 eV (8 decimals) | — (not printed) |
| xtb 6.8.0 | `xtb` | ≤ 7.1e-14 | ≤ 4.7e-13 eV | ≤ 5.0e-4 (3 decimals) |
| xtb 6.7.1 | `xtb-6.7.1` | ≤ 1.8e-9 | ≤ 1.0e-8 eV | ≤ 4.9e-4, but caffeine 6.2e-4 and taxol 1.3e-3 at `--acc 1` (3.0e-4 at `--acc 1e-4`) |
| xtb 6.8.0 `--tblite --quadrupole` | `tblite` | neutral ≤ 1e-14, ions ≤ 1.3e-5 | neutral ≤ 3e-11 eV, ions ≤ 1.2e-4 eV | neutral ≤ 4.9e-5, ions ≤ 3e-4 |

Every pair a program leaves out of its list is below its threshold in axtb too. xtb 6.7.1 prints
the moments it mixed once more after the last SCF iteration (its gradient potentials), so at
`--acc 1` its printed charges differ from the converged ones by up to 5.7e-5 e on taxol and its
quadrupole by 1.3e-3; at `--acc 1e-4` both agree within the printed digits (3.0e-4). The ions
through `xtb --tblite` start their SCF differently from tblite 0.8.0, which the `tblite`
conventions reproduce, and differ at the level of the SCF thresholds.

Cost (CLI energy + gradient, `bench/results/gxtb_parity_features.json`, best of 15): `--wbo` adds
little, −0.5 % to +2.3 % for caffeine and taxol with all three methods (1-minute load 2.4–3.7;
`overhead` in the results).

## Running the scripts by hand

`scripts/check.sh programs` runs the gates below with their outputs in `target/check/programs/`;
run by hand, most scripts write the result files in `bench/results/` that this page quotes. Each
script fails (exit status 1) when a quantity exceeds the limit that this page or the specs quote
for it, at the two significant digits they give, or a calculation fails, except the known
differences explained here; the limits and known differences are in the script (`LIMITS`,
`KNOWN_DIFFERENCES`, `tests/gate.py`), and `scripts/check.sh list` prints them.
The programs come from `AXTB_XTB680`, `AXTB_GXTB2`, `AXTB_XTB671` and `AXTB_GXTB11_DIR`
(`tests/programs.py`); `--xtb PATH` selects another xtb where a script takes it.

```
cargo test --release -p axtb --test consistency   # default conventions: FD gradients, invariances
python tests/validate_tblite.py          # tblite conventions vs tblite (energies, gradients, charges)
python tests/validate_xtb.py [--xtb PATH] [--alpb water]  # xtb (6.8.0) or xtb-6.7.1 conventions vs that xtb program
python tests/validate_hessian.py         # Hessians/frequencies vs xtb --hess
python tests/reference_matrix/run.py [--xtb PATH] [--subset elements,solvation,...]  # every option vs xtb (6.8.0: summary.md; 6.7.1: summary-xtb-6.7.1.md) and tblite
python tests/xtb680_changes.py  # each 6.7.1 behaviour put back into the 6.8.0 conventions, vs xtb 6.8.0
python tests/xtb_hess_acc.py [--xtb PATH]   # what --acc and $hess sccacc do to xtb --hess (per version)
python tests/program_differences.py [--xtb PATH]  # xtb vs tblite: solvation states, atom order, 0.1 K (per version)
python -m pytest tests/reference_matrix  # fast subset of the matrix
python tests/reference_matrix/cli_dump.py target/release/axtb after.jsonl  # every case through the CLI; --compare before.jsonl after.jsonl (bytes), --compare-numbers (numbers only)
python tests/scf_hard_set.py             # SCF convergence on hard cases vs xtb/tblite (bench/results/scf_hard.json)
cargo test --release -p axtb-gxtb        # g-xTB v1.1 and 2.0.1 against their programs (reference data in crates/gxtb/tests)
python tests/gxtb2_reproduction.py       # g-xTB 2.0.1: both conventions vs the program
python tests/gxtb_etemp_reproduction.py  # g-xTB 2.0.1 and v1.1 at 300-5000 K vs the programs (--etemp, -tel)
python tests/gxtb_elements.py            # g-xTB both versions, every element H-Lr vs the programs (bench/results/gxtb_elements.json)
python tests/gxtb_energy_differences.py  # g-xTB energy differences, default vs program conventions [--optimize: rebuild the structures]
python tests/solvation_composite.py      # g-xTB in solvent vs the three xtb programs (bench/results/solvation_composite.json)
python bench/run_benchmarks.py           # full accuracy + performance suite
cargo test --release -p axtb --lib       # incl. response/Hessian validation against finite differences
python bench/hessian_benchmark.py        # Hessian timings and accuracy, all methods vs xtb --hess
python bench/ts_example.py               # SN2 saddle point with analytic Hessians; one imaginary mode
python tests/molden_check.py             # Molden files vs xtb --molden, xtb --gxtb --molden, gxtb -molden
python tests/vibrational_check.py        # frequencies, IR, thermochemistry vs xtb --hess / xtb thermo (GFN2, g-xTB 2.0.1)
python -m pytest crates/axtb-py/tests/test_vibrations.py  # vibrational analysis: Python, CLI and codec
python -m pytest crates/axtb-py/tests/test_molden.py  # Molden round trips, g-xTB 1.1.0 files, codec, CLI
cargo test --release -p axtb-gfnff       # GFN-FF topology, charges, energies, gradients vs xtb; Hessians vs FD
python tests/gfnff_reproduction.py       # GFN-FF: test set, Hessians and scaling (timings, not gated) vs xtb --gfnff
python -m pytest crates/axtb-py/tests/test_gfnff.py   # GFN-FF through Python, CLI and codec
cargo test --release -p axtb-gxtb --test parity_features  # restarts as the g-xTB programs', bond orders, gap, quadrupole
python tests/gxtb_parity_features.py     # bond orders, gap, quadrupole vs every program; restarts; guess reuse along paths and optimizers
python -m pytest crates/axtb-py/tests/test_parity_features.py  # the same features through Python, codec (propagate_wfn) and CLI
```
