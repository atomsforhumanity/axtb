# axtb reference matrix against xtb 6.7.1 (edcfbbe)

Generated 2026-11-07T04:22:07 — xtb 6.7.1 (edcfbbe) (axtb conventions "xtb-6.7.1"), tblite 0.8.0, axtb 0.1.0 (git 3499913).

Value comparisons at SCF accuracy 1e-6 (xtb: 1e-4, its minimum). Tolerances: xtb mode vs xtb |ΔE| ≤ 1e-7 Eh, max|ΔG| ≤ 1e-7 Eh/bohr, |Δq| ≤ 1e-6 e, |Δμ| ≤ 1e-5 e·bohr; tblite mode vs tblite 1e-9 / 1e-8 / 1e-7 / 1e-6; axtb vs xtb mode 1e-7 / 1e-6 / 1e-5 / 1e-4 (all scaled up with the SCF accuracy above 1e-4). Statuses: pass, fail, different_solution (another SCF solution of the same equations, verified by restarting each program from the other's converged moments), expected_difference (documented fixes of the default conventions), both_failed (consistent failure), axtb_failed / ref_failed, info (not meant to match).

## Overview

| category | cases | check | pass | fail | other | worst |ΔE| | worst max|ΔG| | worst |Δq| | worst |Δμ| |
|---|---:|---|---:|---:|---|---:|---:|---:|---:|
| elements | 139 | xtb | 112 | 0 | expected_difference 26, ref_failed 1 | 1.2e-09 | 1.6e-03 | 2.0e-08 | 1.2e-08 |
|  |  | tblite | 139 | 0 |  | 5.3e-14 | 1.6e-11 | 2.6e-10 | 9.4e-11 |
|  |  | axtb_vs_xtbmode | 139 | 0 |  | 4.3e-08 | 2.7e-08 | 1.1e-07 | 1.1e-07 |
| structures | 13 | xtb | 11 | 1 | expected_difference 1 | 3.0e-08 | 6.5e-05 | 1.9e-07 | 1.4e-05 |
|  |  | tblite | 12 | 0 | different_solution 1 | 8.5e-14 | 6.8e-13 | 2.7e-11 | 5.6e-12 |
|  |  | axtb_vs_xtbmode | 12 | 0 | different_solution 1 | 1.2e-07 | 6.1e-09 | 1.7e-07 | 8.9e-08 |
| multielement | 32 | xtb | 28 | 0 | both_failed 3, ref_failed 1 | 1.0e-09 | 2.0e-08 | 6.0e-09 | 7.7e-08 |
|  |  | tblite | 29 | 0 | both_failed 3 | 6.4e-14 | 3.9e-13 | 2.0e-11 | 2.7e-10 |
|  |  | axtb_vs_xtbmode | 29 | 0 | both_failed 1, ref_failed 2 | 3.1e-08 | 6.1e-08 | 1.6e-07 | 4.1e-07 |
| charges | 47 | xtb | 36 | 0 | expected_difference 11 | 3.7e-09 | 2.7e-03 | 5.3e-09 | 4.9e-09 |
|  |  | tblite | 47 | 0 |  | 5.0e-14 | 6.1e-12 | 1.7e-10 | 5.3e-11 |
|  |  | axtb_vs_xtbmode | 47 | 0 |  | 3.5e-08 | 7.1e-09 | 7.4e-08 | 1.1e-08 |
| spin | 34 | xtb | 22 | 0 | expected_difference 12 | 6.8e-10 | 3.3e-03 | 5.4e-09 | 1.0e-08 |
|  |  | tblite | 34 | 0 |  | 6.4e-14 | 2.8e-12 | 1.7e-10 | 1.7e-10 |
|  |  | axtb_vs_xtbmode | 34 | 0 |  | 2.0e-08 | 2.4e-08 | 4.4e-08 | 1.1e-08 |
| open_shell_metals | 29 | xtb | 29 | 0 |  | 7.0e-10 | 5.1e-09 | 5.5e-09 | 1.1e-08 |
|  |  | tblite | 29 | 0 |  | 2.5e-14 | 8.8e-13 | 7.5e-11 | 2.8e-10 |
|  |  | axtb_vs_xtbmode | 29 | 0 |  | 2.5e-08 | 1.2e-08 | 1.7e-07 | 3.0e-08 |
| solvation | 483 | xtb | 183 | 0 | expected_difference 300 | 2.6e-10 | 2.4e-03 | 5.4e-09 | 9.4e-09 |
|  |  | tblite | 468 | 0 |  | 1.3e-10 | 5.4e-11 | 2.8e-10 | 6.2e-10 |
|  |  | axtb_vs_xtbmode | 325 | 0 | expected_difference 158 | 6.9e-03 | 1.1e-08 | 8.2e-08 | 8.9e-08 |
| etemp_acc | 100 | xtb | 88 | 0 | both_failed 8, different_solution 4 | 9.0e-10 | 3.0e-07 | 2.3e-05 | 5.7e-05 |
|  |  | tblite | 88 | 0 | both_failed 8, different_solution 4 | 4.0e-11 | 1.2e-06 | 3.4e-05 | 1.0e-04 |
|  |  | axtb_vs_xtbmode | 84 | 0 | both_failed 8, different_solution 5, expected_difference 3 | 2.6e-07 | 8.2e-07 | 2.1e-05 | 4.4e-05 |
| hessian | 7 | hessian | 7 | 0 |  | max|ΔH| 1.3e-04, max|Δν| 9.1e-01 cm⁻¹ | | |  |
| hessian_analytic | 45 | analytic_ref | 45 | 0 |  | max|ΔH| 1.4e-10, max|Δν| 5.9e-05 cm⁻¹ | | |  |
|  |  | analytic_xtb | 45 | 0 |  | max|ΔH| 2.6e-04, max|Δν| 6.6e+00 cm⁻¹ | | |  |
|  |  | analytic_numref | 10 | 0 |  | max|ΔH| 3.0e-06, max|Δν| 1.6e-02 cm⁻¹ | | |  |
|  |  | analytic_threads | 3 | 0 |  | max|ΔH| 0.0e+00 | | |  |
| threads | 8 | threads | 6 | 0 |  | 0.0e+00 | 0.0e+00 | 0.0e+00 | 0.0e+00 |
|  |  | threads_hessian | 2 | 0 |  | max|ΔH| 0.0e+00 | | |  |
| nodisp | 10 | tblite | 10 | 0 |  | 2.5e-14 | 1.5e-12 | 8.3e-11 | 4.4e-11 |
| failures | 21 | failure | 21 | 0 |  | — | — | — | — |
| api | 23 | calculator | 2 | 0 |  | 2.5e-14 | 2.0e-12 | 4.3e-11 | 1.0e-10 |
|  |  | cli | 21 | 0 |  | 1.8e-14 | 2.6e-15 | 2.1e-14 | 4.0e-14 |

## Expected differences (documented fixes of the default conventions)

| check | reason | cases | |ΔE| range |
|---|---|---:|---|
| axtb_vs_xtbmode | CODATA 2022 kB for the electronic temperature | 3 | 2.6e-07 – 2.6e-07 |
| axtb_vs_xtbmode | reference state: mass density of the solvent instead of xtb 6.7.1's rho/M term (grimme-lab/xtb#1505) | 158 | 5.0e-03 – 6.9e-03 |
| xtb | xtb 6.7.1 release: gradient without the H0/S term of AO pairs with zero overlap (atoms sharing a coordinate), fixed by xtb PR #1208 | 350 | 1.2e-12 – 4.7e-10 |

## Cases that are not a plain pass

### elements

| case | check | status | details |
|---|---|---|---|
| el-39-Y-alt ([Y(H2O)6]3+) | xtb | ref_failed | ref_error: [ERROR] Program stopped due to fatal error / -3- Single point calculation terminated / -2- xtb_calculator_singlepoint: Electronic structure method terminated /  |

### structures

| case | check | status | details |
|---|---|---|---|
| struct-gh106 (tests/structures/gh106.xyz) | xtb | fail | dE 3.0e-08; dG 9.6e-08; dq 1.9e-07; ddip 1.4e-05; xtb from axtb moments: dE 3.0e-08; axtb from xtb moments: dE 3.0e-08; lower: xtb |
| struct-gh106 (tests/structures/gh106.xyz) | tblite | different_solution | dE 2.3e-04; dG 2.6e-04; dq 1.8e-03; ddip 1.6e-02; axtb[tblite] from xtb moments: dE 4.1e-04, dq 1.8e-03; axtb[tblite] from xtb 6.7.1 (26dd68d) moments: dE -7.1e-14, dq 1.6e-08; axtb[tblite] from axtb[xtb] moments: dE 4.1e-04, dq 1.8e-03; tblite from EEQ guess: dE -1.8e-04; lower: tblite |
| struct-gh106 (tests/structures/gh106.xyz) | axtb_vs_xtbmode | different_solution | dE -2.0e-04; dG 2.5e-04; dq 1.6e-03; ddip 5.9e-02 |

### multielement

| case | check | status | details |
|---|---|---|---|
| mbheavy-101 (random molecule with heavy elements) | axtb_vs_xtbmode | ref_failed | ref_error: ScfNotConvergedError: SCF not converged in 250 iterations (last energy change -1.017e-2 Eh, residual 1.384e0) |
| mbheavy-112 (random molecule with heavy elements) | axtb_vs_xtbmode | ref_failed | ref_error: ScfNotConvergedError: SCF not converged in 250 iterations (last energy change -3.405e-3 Eh, residual 1.659e-1) |
| mbheavy-113 (random molecule with heavy elements) | xtb | ref_failed | ref_error: [ERROR] Program stopped due to fatal error / -3- Single point calculation terminated / -2- xtb_calculator_singlepoint: Electronic structure method terminated /  |

### etemp_acc

| case | check | status | details |
|---|---|---|---|
| eta-Cu5-T0-acc1 (Cu5 cluster (doublet), 0 K, acc 1) | xtb | different_solution | dE 3.0e-04; dG 1.5e-02; dq 1.1e-01; ddip 5.4e-01; xtb from axtb moments: dE -6.7e-10; axtb from xtb moments: dE -7.5e-10; lower: xtb |
| eta-Cu5-T0-acc1 (Cu5 cluster (doublet), 0 K, acc 1) | tblite | different_solution | dE 3.0e-04; dG 3.3e-02; dq 2.3e-01; ddip 1.1e+00; axtb[tblite] from xtb moments: dE 7.3e-12, dq 1.8e-01; axtb[tblite] from xtb 6.7.1 (26dd68d) moments: dE 6.7e-13, dq 6.9e-06; axtb[tblite] from axtb[xtb] moments: dE 3.0e-04, dq 2.3e-01; tblite from EEQ guess: dE -2.6e-10; lower: tblite |
| eta-Cu5-T0-acc1 (Cu5 cluster (doublet), 0 K, acc 1) | axtb_vs_xtbmode | different_solution | dE -3.0e-04; dG 2.2e-02; dq 1.1e-01; ddip 5.4e-01 |
| eta-Cu5-T0-acc0.01 (Cu5 cluster (doublet), 0 K, acc 0.01) | xtb | different_solution | dE -3.0e-04; dG 1.8e-02; dq 1.1e-01; ddip 4.0e-01; xtb from axtb moments: dE -7.0e-10; axtb from xtb moments: dE -7.0e-10; lower: axtb |
| eta-Cu5-T0-acc0.01 (Cu5 cluster (doublet), 0 K, acc 0.01) | tblite | different_solution | dE 3.0e-04; dG 3.3e-02; dq 2.3e-01; ddip 1.1e+00; axtb[tblite] from xtb moments: dE 3.0e-04, dq 1.1e-01; axtb[tblite] from xtb 6.7.1 (26dd68d) moments: dE 3.0e-04, dq 1.1e-01; axtb[tblite] from axtb[xtb] moments: dE -1.1e-14, dq 1.4e-07; tblite from EEQ guess: dE -2.5e-14; lower: tblite |
| eta-Cu5-T0-acc0.01 (Cu5 cluster (doublet), 0 K, acc 0.01) | axtb_vs_xtbmode | different_solution | dE 3.1e-09; dG 3.8e-02; dq 1.8e-01; ddip 9.4e-01 |
| eta-Cu5-T0-acc0.0001 (Cu5 cluster (doublet), 0 K, acc 0.0001) | xtb | different_solution | dE -3.0e-04; dG 1.8e-02; dq 1.1e-01; ddip 4.0e-01; xtb from axtb moments: dE -7.0e-10; axtb from xtb moments: dE -7.0e-10; lower: axtb |
| eta-Cu5-T0-acc0.0001 (Cu5 cluster (doublet), 0 K, acc 0.0001) | tblite | different_solution | dE 3.0e-04; dG 3.3e-02; dq 2.3e-01; ddip 1.1e+00; axtb[tblite] from xtb moments: dE 3.0e-04, dq 1.1e-01; axtb[tblite] from xtb 6.7.1 (26dd68d) moments: dE 3.0e-04, dq 1.1e-01; axtb[tblite] from axtb[xtb] moments: dE -7.1e-15, dq 4.3e-09; tblite from EEQ guess: dE -1.8e-14; lower: tblite |
| eta-Cu5-T0-acc0.0001 (Cu5 cluster (doublet), 0 K, acc 0.0001) | axtb_vs_xtbmode | different_solution | dE 3.1e-09; dG 3.8e-02; dq 1.8e-01; ddip 9.4e-01 |
| eta-Cu5-T0-acc1e-06 (Cu5 cluster (doublet), 0 K, acc 1e-06) | xtb | different_solution | dE -3.0e-04; dG 1.8e-02; dq 1.1e-01; ddip 4.0e-01; xtb from axtb moments: dE -7.0e-10; axtb from xtb moments: dE -7.0e-10; lower: axtb |
| eta-Cu5-T0-acc1e-06 (Cu5 cluster (doublet), 0 K, acc 1e-06) | tblite | different_solution | dE 3.0e-04; dG 3.3e-02; dq 2.3e-01; ddip 1.1e+00; axtb[tblite] from xtb moments: dE 3.0e-04, dq 1.1e-01; axtb[tblite] from xtb 6.7.1 (26dd68d) moments: dE 3.0e-04, dq 1.1e-01; axtb[tblite] from axtb[xtb] moments: dE -1.8e-14, dq 1.0e-11; tblite from EEQ guess: dE -2.1e-14; lower: tblite |
| eta-Cu5-T0-acc1e-06 (Cu5 cluster (doublet), 0 K, acc 1e-06) | axtb_vs_xtbmode | different_solution | dE 3.1e-09; dG 3.8e-02; dq 1.8e-01; ddip 9.4e-01 |
| eta-O2-singlet-T0.1 (O2 singlet (degenerate π*), 0.1 K) | axtb_vs_xtbmode | different_solution | dE 2.5e-03; dG 4.2e-04; dq 4.4e-11; ddip 8.9e-16 |

## Failure behaviour

| case | expect | xtb | tblite | axtb | axtb[xtb] | axtb[tblite] | status |
|---|---|---|---|---|---|---|---|
| water, max 3 SCF iterations | error | error | error | error | error | error | pass |
| Fe4 cluster at 0 K (charge sloshing) | any | error | error | error | error | error | pass |
| FrH (Z=87, no GFN2 parameters) | error | error | error | error | error | error | pass |
| UF6 (Z=92, no GFN2 parameters) | error | error | error | error | error | error | pass |
| OgH (Z=118) | error | error | error | error | error | error | pass |
| water with uhf=1 (parity mismatch) | error | ok | error | error | error | error | pass |
| NO2 with uhf=0 (odd electrons) | error | ok | ok | error | error | error | pass |
| H2 with uhf=4 (more unpaired than electrons) | error | ok | ok | error | error | error | pass |
| O2 with uhf=10 (alpha electrons exceed orbitals) | error | garbage | garbage | error | error | error | pass |
| H2 with charge +3 (negative electron count) | error | garbage | garbage | error | error | error | pass |
| H2 with charge +2 (no electrons) | ok | ok | ok | ok | ok | ok | pass |
| water with charge +9 (negative electron count) | error | garbage | garbage | error | error | error | pass |
| H2 with charge -5 (more electrons than spin orbitals) | error | error | garbage | error | error | error | pass |
| water with charge -30 | error | error | garbage | error | error | error | pass |
| water with etemp -5 K | error | ok | ok | error | error | error | pass |
| water with accuracy 0 | error | ok | error | error | error | error | pass |
| water with accuracy 1e-12 (xtb clamps to 1e-4) | any | ok | error | error | error | error | pass |
| H2 with charge +0.5 (a non-integer charge is an input error) | error | garbage | garbage | error | error | error | pass |
| water in ALPB 'octane' (not parametrized) | error | error | error | error | error | error | pass |
| water in GBSA 'aniline' (ALPB-only solvent) | error | error | error | error | error | error | pass |
| two H atoms at the same position | error | error | error | error | error | error | pass |
