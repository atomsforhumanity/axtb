# GFN-FF in axtb: specification and mapping to xtb

GFN-FF (S. Spicher, S. Grimme, Angew. Chem. Int. Ed. 59, 15665 (2020),
doi:10.1002/anie.202004239) as implemented in `crates/gfnff` (crate `axtb-gfnff`,
LGPL-3.0-or-later), method `"gfnff"` in the CLI, Python and qccodec. It is a port of xtb's
implementation, term by term and operation by operation, and reproduces the xtb 6.8.0 program
(`xtb --gfnff`; conventions `axtb`, the default, and `xtb`) and the xtb 6.7.1 program (the
g-xTB 2.0.1 build; conventions `xtb-6.7.1`): topology identical, EEQ charges to 5e-12 e,
energies (every term) to 1e-12 Eh and gradients to 1e-11 Eh/bohr (§9). On top of xtb it has
analytic Hessians with dipole derivatives, linear-memory pair loops (xtb fails above ~3000
atoms of water on 10 GiB) and deterministic threading.

**Two xtb versions.** The port was written against 6.7.1 (the reference binary below) and
follows 6.8.0 through one table of settings, `conventions::ConventionParams` (§11): the bohr of
the covalent radii, the pi-bond threshold, the removal of one-sided neighbors in `nbondmat`,
the Fermi smearing of the Hückel step, and the rounding of the distances that decide ties.
`Conventions::Xtb680` (default) and `Conventions::Xtb671` select them; the `Topology` carries
its conventions, so a topology and its energies always belong to one version.

**Sources.** The Fortran of xtb's `src/gfnff` (the `gxtb` branch of xtb, commit 97f7bf5,
LGPL-3.0, in `reference/xtb-gxtb`, git-ignored; GFN-FF there is unchanged from xtb 6.7.1) was
read and ported, and compared with tag v6.8.0 (4b8c987) for §11. The reference programs are
the xtb 6.7.1 binary (26dd68d, Intel Fortran ifort, `-r8`, MKL) and the official xtb 6.8.0
binary (4b8c987, Intel ifx from oneAPI 2025, MKL). Where the source and the binary could differ (compiler-dependent rounding), the
binary was followed; the only such case is the Intel inlining of `NORM2` (§7). Parameter
tables are generated, not typed: `crates/gfnff/scripts/gen_data.py <xtb source> src/data.rs`
reads `gfnff_param.f90`, `gfnff_rab.f` and `gfnff_ini.f90`, and checks that xtb's copies of
data other crates hold (`covalentradd3.f90`, `sqrtzr4r2.f90`, `include/param_ref.fh`) equal
them (§3).

**Scope.** Molecules (no periodic boundary conditions), gas phase (no GBSA/ALPB), no
external field. Everything xtb does for a molecule with `--gfnff` is there, including the
topology setup for every element H-Lr, metal complexes, charged fragments and the HB/XB
machinery.

## 1. Units, precision, conventions

Bohr and Hartree throughout. xtb compiles with `-r8`, so every unsuffixed Fortran literal is
a double; the port writes the same literals. The few single-precision quantities of xtb are
single precision here too: the topological distance matrix `rabd` of the charge model
(Floyd-Warshall in `real(sp)`, cutoff 13) and the `tdist_thr` comparison. Constants are
xtb's: `autoaa = 0.52917726` inside GFN-FF (`param.rs`), `pi = 3.1415926535897932384626433832795029`
(`f64::consts::PI`), the truncated `sqrtpi = 1.77245385091` of the CN derivative, `kB` and
the eV factor of the Hückel step.

The CLI's `conventions` `"axtb"` (default) and `"xtb"` give 6.8.0's force field and `"xtb-6.7.1"`
6.7.1's; `"xtb"` and `"xtb-6.7.1"` read XYZ files with mctc-lib's Ångström factor (as `xtb
mol.xyz`) instead of CODATA 2022 and make the vibrational analysis that xtb version's (masses,
constants, projection, and the GFN-FF frequency scale 1.03 of xtb's thermochemistry,
`property.F90`). JSON and Python input is in bohr and needs no factor.

## 2. Code map

| Rust (`crates/gfnff/src`) | Fortran (`src/gfnff`) | Content |
|---|---|---|
| `data.rs` (generated) | `gfnff_param.f90`, `gfnff_rab.f`, `gfnff_ini.f90` | element parameters (§3) |
| `param.rs` | `gfnff_param.f90` (`newGFNFFGenerator`, `gfnff_set_param`, `gfnff_thresholds`), `gfnff_rab.f` | global constants, derived parameters, cutoffs |
| `setup.rs`, `setup_util.rs` | `gfnff_ini.f90`, `gfnff_ini2.F90`, `getring36.f90`, `mrec.f90`, `gfnff_qm.f90` | topology (§4) |
| `topology.rs` | `topology.f90` (`TGFFTopology`), `neighbor.f90` (`TNeigh`) | the topology data |
| `cn.rs` | `gfnff_eg.f90` (`gfnff_dlogcoord`, `dncoord_erf`, `erfCount`) | coordination numbers (the counting function and the cap are `axtb_base::ncoord`'s `XtbErf::GFNFF`, `log_cap`) |
| `energy.rs` | `gfnff_eg.f90` (`gfnff_eg`, `goed_gfnff`), `gdisp0.f90` | energy, gradient, EEQ, dispersion, HB lists (§5) |
| `terms.rs` | `gfnff_eg.f90` (`egbond*`, `egbend*`, `egtors*`, `abhgfnff_eg*`, `rbxgfnff_eg`, `batmgfnff_eg`, `sTors_eg`) | local terms, generic over `f64`/`Dual` |
| `geom.rs` | `gfnff_helpers` / `math.f` (`bangl`, `valijklff`, `dphidr`, `omega`, `domegadr`, `impsc`, `vecnorm`) | geometry helpers |
| `cells.rs` | — | linked cells replacing xtb's all-pairs loops |
| (`axtb_base::linalg`) | LAPACK `dgetrf`/`dgetrs`/`dsytrf`, `dsyev` (MKL) | `Lu` and `sym_eig`: faer's low-level LU and eigensolver |
| `hessian.rs` | — (xtb: finite differences) | analytic Hessian, dipole derivatives (§6) |
| `calculator.rs` | `calculator.f90` | fixed-topology calculator, threads |
| `conventions.rs` | — | xtb 6.8.0 vs 6.7.1: the settings that differ (§11) |

Every module is crate-private. The public API is `Calculator` (`new`, `singlepoint`, `hessian`,
`numerical_hessian`, `topology`) with its `Options` and `Conventions`, the results `Results`,
`Energies` and `Hessian`, the read-only `Topology` (with `Bond`, `Angle`, `Torsion`, `BondHb`,
`BondDistance`), `Error`, and `METHOD`/`is_method` for the front ends.

## 3. Parameters

`data.rs` holds xtb's tables literally: EEQ (`chi`, `gam`, `cnf`, `alp`), bond (`bond`),
repulsion (`repa`, `repan`, `repz`), angle (`angl`, `angl2`), torsion (`tors`, `tors2`),
Pauling EN, radii (`rad`), metal and group classification, CN normalization, the bond-length
model of `gfnffrab` (`rab_r0`, `rab_en`, `rab_cnfak`; the copies in `gfnffrab` and
`gfnffdrab` were checked to be identical) and the `zeta` charge-scaling constants. The D3/D4
data that xtb copies from other libraries come from the workspace crates that hold them: the
covalent radii of `covalentRadD3` are mctc-lib's (`axtb_base::elements`), the D4 reference
systems (reference CNs, polarizabilities `alphaiw`, `ascale`, `refsys`, `hcount`, `sscale`,
`secaiw`), their Casimir-Polder integration to C6 and `r4_over_r2` are dftd4's (`axtb-d4`),
with the gaps of xtb's tables applied to them: no reference systems for Z = 87-111 (so the
C6 of Fr-Lr is zero) and `r4_over_r2` = 0 for Z = 95-111. `r4_over_r2` enters only together
with a pair's C6, so its zeros change no result; they are kept to keep xtb's parameters. The
generator checks every one of these values against xtb's source.
`param.rs::Gen` is `newGFNFFGenerator` (all global constants: `cnmax`, `linthr`, `fcthr`,
`tdist_thr`, `rqshrink`, `hqpi`, `hbacut`, `xbacut`, the bond/angle/torsion scaling
factors, ...), `Param::new` is `gfnff_set_param` (HB/XB acidity and basicity tables,
`zb3atm`, D3 cutoff radii `d3r0`, `rcov = covrad_d3 * aatoau * 4/3`, `sqrt(Z r4/r2)`), and
`Thresholds` is `gfnff_thresholds(accuracy)` (accuracy 0.1, 2.0 above 10000 atoms as in
`calculator.f90`; squared-distance cutoffs CN 150, repulsion 500, dispersion 2500, HB1 250,
HB2 450 bohr² at 0.1).

## 4. Topology setup (`setup.rs::setup` = `gfnff_ini`)

In xtb's order:

| Step | Rust | Fortran | Notes |
|---|---|---|---|
| distances, `rab`, `sqrab` | `setup` (on demand) | `gfnff_ini.f90` "distances and bonds" | no N x N arrays |
| neighbor lists `nbf`, `nb`, `nbm` (full, normal, no-metal), metallic character, cluster tags | `gfnff_neigh`, `adjust_nb_lnh_anh` | `gfnff_neigh`, `getnb`, `nn_nearest_noM` (`gfnff_ini2.F90`), `adjust_NB_LnH_AnH` | element factors `fat`, `rqshrink`; `jth_nb` with Intel's `NORM2` (§7) |
| hybridization, special tags (carbenes, nitro N, eta) | `gfnff_neigh` | `gfnff_neigh` | |
| rings | `setup_util::getring36`, `ssort`, `ringsatom`, `ringsbond`, `ringsbend`, `ringstors`, `ringstorl` | `getring36`, `ssort` (`gfnff_ini2.F90`), `rings*` | literal DFS, at most 19 rings per atom, xtb's early exit and `ringsbend` quirk |
| bond pair distances `bpair` (1, 2, 3 bonds, else 5) | `nbondmat` | `nbondmat`, `pairsbond` | sparse map; xtb's tagging order reproduced (it drops one entry when an atom has several, kept literally) |
| topological distances | `topo_distances` | Floyd-Warshall in `gfnff_ini` | single precision, cutoff 13; sparse merge instead of the dense N³ loop, same numbers |
| fragments | `setup_util::mrec` | `mrecgff` | |
| fragment charges | `setup` | `gfnff_input` (`.CHRG` second line) | total charge on the first fragment unless `fragment_charges`; given ones must be one per fragment and add up to the total charge, else an input error (xtb ignores extra charges and zeroes inconsistent ones) |
| topology charges (two `qloop` passes) | `goedeckera` | `goedeckera` | EEQ with fragment constraints, dense LU (faer low-level `partial_pivoting`), the CN of the setup; neighbor lists rebuilt between passes |
| pi systems, Hückel bond orders | `huckel`, `qmsolve`, `occu`, `fermismear` | "Hueckel setup", `gfnffqmsolve`, `occu`, `fermismear` (`gfnff_qm.f90`) | iterative, Fermi smearing at 4000 K, eigenvalues scaled by 0.1 x 27.2113957, integer occupations for biradicals; exactly equal occupations give P = f·I (xtb's LAPACK returns exact zeros there) |
| EEQ parameters, xi correction | `setup` | "setup xi correction for EEQ" | atomic `chieeq`, `gameeq`, `alpeeq` |
| bonds `vbond` | `bond_params` | "bonds" (`gfnff_ini.f90` ~1200-1430) | ring, pi-bond-order, metal, hypervalent and HB scaling; `btyp` |
| A-H...B coupled bonds | `bond_hb_setup` | `bond_hbset`, `bond_hb_AHB_set` (`gfnff_ini2.F90`) | `nr_hb` carried over as in xtb |
| bends | `angle_setup` | "bend" (~1476-1808) | per-element cases, rings, metals |
| torsions, out-of-plane | `torsion_setup` | "torsion" (~1810-2035), "out-of-plane" (~2037), `chktors` | `chktors` with xtb's 3.1415926 |
| triple-bond torsions | `stors_setup` | `specialTorsList` | xtb's index quirks |
| bonded ATM triples | `setup` | "bend and tors" / `b3list` | |
| non-bonded exponents, `zetac6` | `Topology::alphanb`, `topology.zeta` | "non bonded pair exponents" | stored per atom, evaluated per pair |
| HB basicity/acidity, HB/XB atom lists | `setup` | "make list of HB donor basicity/acidity", "make list of ABs for HAB" | `hb_h`, `hb_ab`, `xb_ax`, `xb_b` |

The result (`Topology`) can be compared field by field with xtb's binary `gfnff_topo` file
(24 unformatted records; `tests/gfnff_xtb.py::parse_topo`).

## 5. Energy and gradient (`energy.rs::evaluate` = `gfnff_eg`)

Every term follows xtb's expression and summation order within a pair or tuple; pair loops
visit only pairs inside xtb's cutoff (cell lists), in xtb's order. Gradients are xtb's
analytic ones (including where they are not the exact derivative of the energy, §7).

| Term (`Energies` field) | Rust | Fortran | Form |
|---|---|---|---|
| CN (log of erf CN, cap `cnmax`) | `cn.rs` | `gfnff_dlogcoord`, `erfCount` | dCN with xtb's truncated sqrt(pi) |
| bond-length model r0(CN) | `energy::bond_par`, `param::rab_ff`, `param::rab_scale` | `gfnffdrab` | per bond, CN derivatives chained |
| `bond` | `terms::bond` | `egbond`, `egbond_hb` | k_pref exp(-k_steep (r - r0)²) with k_pref < 0; for A-H bonds with H-bond partners the steepness is scaled by 1 - c·CN_HB(H) (`dncoord_erf`, `hb_cn`) |
| `repulsion` (non-bonded) | `terms::rep_nonbonded` | `gfnff_eg` "REP non-bonded" | z_i z_j exp(-alphanb r^1.5)/r over non-bonded pairs |
| `repulsion` (bonded) | `terms::rep_bonded` | `gfnff_eg` "bonded REP" | |
| `electrostatic` | `energy::eeq` | `goed_gfnff` | EEQ with fragment constraints: bordered N + nfrag system, dense LU; gradient with dq/dCN (`qtmp`) |
| `dispersion` | `terms::disp_pair`, `DispModel` | `d3_gradient`, `weight_references`, `get_atomic_c6` (`gdisp0.f90`) | D3-type pairwise with D4 reference polarizabilities (Casimir-Polder on xtb's 23-point grid), BJ damping, charge scaling `zetac6` |
| `angle` | `terms::bend` | `egbend` | k (cos θ - cos θ0)², (θ - θ0)² for θ0 ≥ π - 1e-6; distance damping `gfnffdampa` |
| `torsion` | `terms::torsion` | `egtors` | k (1 + cos(n(φ - φ0) + π)); out-of-plane via `omega`/`domegadr`; damping `gfnffdampt` |
| `torsion` (triple bonds) | `terms::stors` | `sTors_eg` | xtb applies the last list entry once per entry; kept |
| `batm` | `terms::batm` | `batmgfnff_eg` | bonded three-body ATM with topology charges |
| `hb` (case 1, H not bonded to A) | `terms::hb1` | `abhgfnff_eg1` | |
| `hb` (case 2) | `terms::hb2` | `abhgfnff_eg2new` | B neighbors' out-of-line damping |
| `hb` (pyridine-type N) | `terms::hb2_rnr` | `abhgfnff_eg2_rnr` | lone-pair position from the ring neighbors |
| `hb` (carbonyl, nitro) | `terms::hb3` | `abhgfnff_eg3` | bend H...O=C and torsions R-C=O...H (`egbend_nci_mul`, `egtors_nci_mul`) |
| `xb` | `terms::xb` | `rbxgfnff_eg` | halogen, chalcogen and pnictogen bonds |
| `external` | — | `gfnff_eg` "external stuff" | always 0 (no field) |

HB and XB pair lists (`energy::hb_lists`) are rebuilt at every geometry from the topology's
candidate atoms with xtb's distance thresholds (`gfnff_hbset0`); xtb freezes them between its
own list updates during an optimization, so results along an xtb optimization can differ
from a fresh evaluation in the same way that xtb's own restart does.

## 6. Hessian (`hessian.rs`)

xtb computes GFN-FF Hessians by central differences (step 0.005 bohr) of the analytic
gradient. axtb differentiates the gradient code itself:

- **Columns**: one column (Cartesian coordinate x_k) is the forward-mode derivative of the
  gradient (dual numbers, `real::Dual`; every kernel in `terms.rs` is generic over the scalar
  type), restricted to the terms x_k reaches: terms containing atom k, terms whose CN or
  HB-CN involves k (with the CN chain rule), dispersion pairs of those atoms (dual C6 through
  the CN weights). The EEQ charges are held fixed in the columns.
- **Charge response**: with R = ∂(∂E/∂q)/∂x and the bordered EEQ matrix M (LU factors from
  the energy evaluation), H += -Rᵀ [M⁻¹R]_q, solved in blocks of 384 columns with one GEMM per
  block.
- **Dipole derivatives** (for IR intensities): ∂μ/∂x_k = q_k e_c + Σ_i x_i ∂q_i/∂x_k from the
  same response.
- **Singular points of xtb's formulas**: bends and torsions are written through the angle
  (acos, 1/sin θ) and the HB/XB out-of-line factor through expo/(r_AH + r_BH - r_AB); all
  are 0/0 at linear bends, planar torsions and linear A-H...B or A-X...B, where the gradient
  is smooth but the derivative of the literal expressions is not. The dual path therefore
  uses algebraically identical forms (`bend_smooth`, `torsion_smooth`, `stors_smooth`,
  cos(φ) in the carbonyl-HB torsions, (cut/radab)/r_AB for the out-of-line factor;
  `Real::DUAL`); energies and gradients keep xtb's operations. Torsions with sin(nφ0) ≠ 0
  are not differentiable at planarity and keep the literal kernel.
- The result is symmetrized (`max_asymmetry` reports the largest |H_ij - H_ji|, ≤ 1e-13).
  Where xtb's gradient is not the derivative of its energy (§7), this is the derivative of
  the gradient, i.e. what finite differences of xtb's gradient converge to.
- **Where it is undefined**: with an H exactly on the C=O axis of a carbonyl acceptor (the
  acetone...water dimer of grimme-lab/xtb#1516), the carbonyl HB term's R-C=O...H torsions
  are undefined and the energy jumps; the analytic Hessian there is not finite, and
  `Calculator::hessian` returns a numerical error naming the bond (`hessian::not_finite`)
  instead of NaN. The numerical Hessian is defined there.
- `Calculator::numerical_hessian(step)` gives xtb's finite-difference Hessian (same step,
  symmetrized, dipole derivatives from the displaced EEQ dipoles); it reproduces
  `xtb --gfnff --hess` thermochemistry to 1e-11 Eh.

Columns run in parallel (one task per column, written into its own slice); memory is the
3N x 3N result plus R (N + nfrag rows), so a 5000-atom Hessian needs ~3.2 GB. The numerical
Hessian also writes each column straight into its 3N x 3N result.

## 7. xtb behaviors reproduced on purpose

- **Intel `NORM2`**: ifort (6.7.1) inlines `norm2` of a 3-vector as
  `sqrt(fma(z,z,fma(x,x,y*y)))` (`jth_nb`, `nn_nearest_noM`) and the generic array version as
  `sqrt(fma(z,z,fma(y,y,x*x)))` (the out-of-plane neighbor sort in `gfnff_ini`), found in the
  binary's disassembly; it decides ties in symmetric structures (NF3's out-of-plane term).
  ifx (6.8.0, AVX2 path): `jth_nb` `sqrt(fma(z,z,fma(y,y,x*x)))` for the neighbors its
  vectorized loop takes four at a time and `sqrt(fma(x,x,y*y) + z*z)` for the rest,
  `nn_nearest_noM` `sqrt(fma(z,z,x*x+y*y))`, the out-of-plane sort `sqrt((x*x+y*y)+z*z)`
  (`conventions::TieNorms`).
- `create_derfCN` uses sqrt(pi) truncated to 1.77245385091 in the CN derivative, so the
  gradient is not exactly the derivative of the energy (~1e-11 relative).
- `abhgfnff_eg2_rnr` keeps only the A and LP parts of the A...LP-B out-of-line gradient
  (the B part is dropped): xtb's gradient differs from finite differences of its energy by
  up to ~6e-8 Eh/bohr there; reproduced.
- With no B neighbors `eg2_rnr` multiplies uninitialized array elements; taken as 1.
- `ringsbend` keeps the third atom's ring list in xtb's (inconsistent) way; `getring36`
  stops after the first non-skipped member, at most 19 rings.
- `nbondmat`'s tagging loses an entry for atoms with several equivalent ones.
- `sTors_eg` applies the last entry of the triple-bond torsion list once per entry.
- `nr_hb` of an A-H bond carries over between bonds as in `bond_hb_AHB_set`.
- Hückel: eigenvalues x 0.1 x 27.2113957 (eV per Hartree of the pi Hamiltonian), Fermi
  smearing at 4000 K, integer-occupation fallback for biradicals.
- xtb prints undefined list entries as `********` in `gfnff_lists.json`; the comparison
  scripts read them as null.

Known difference: D5h ferrocenium (`tm-ferrocenium`). Each cationic Cp ring has an exactly
degenerate Hückel pair at the Fermi level with integer occupations, so the bond orders depend
on which eigenvectors the eigensolver returns for the pair (xtb: MKL `dsyev`); the topology
and energy differ from xtb's (xtb's result is not rotation invariant either).

## 8. Memory, threads, fixed topology

xtb keeps dense N x N arrays (`sqrab`, `dcn`, `eeqtmp`, `d3list`, `alphanb`, `zetac6`,
`bpair`, `hbatABl`) and, in every energy evaluation, `considered_ABH(NAB, NAB, NH)`
(`gfnff_eg.f90`; 4-byte logicals over the A/B and H atoms of the A-H...B bond list). For
water that is 4 · (N/3)² · (2N/3) bytes: 7.5 GiB at 3000 atoms, 37 GiB at 5100. axtb replaces
all of them by pair lists and cell-list searches; the only dense matrix is the EEQ system
((N + nfrag)², ~1.4 GB at 9900 water atoms), factorized once per geometry.

Threads: `Options::threads` (at least 1) creates one rayon pool of that size per calculator
(none for 1); pair loops run over 64 fixed interleaved chunks summed in order, dense algebra
uses faer's low-level routines with an explicit `Par`, so results repeat bit for bit on a given
thread count. Between thread counts the LU factorizations of the EEQ systems differ by
round-off above ~100 atoms (faer's parallel kernels split them by thread count): C50H102,
(H2O)200, (H2O)333 and crambin on 1, 2, 4 and 8 threads gave topology charges ≤ 3e-12 e,
energies ≤ 1.2e-12 Eh, gradients ≤ 9e-14 Eh/bohr and analytic Hessians ≤ 3e-13 Eh/bohr² apart;
the 24 reference structures (up to 113 atoms) gave identical bits.

Fixed topology: a `Calculator` keeps the topology of its first structure (or of
`topology_positions`) for every later geometry, as xtb does within an optimization
(`gfnff_topo`); the CLI, Python and qccodec expose it, and results carry the topology (bonds,
pi bond orders, hybridization, fragments, fragment and topology charges).

Input and errors (`Error`): positions must be finite and no two atoms closer than 0.1 bohr
(the setup, every single point and both Hessians); the total and fragment charges must lie
within the total nuclear charge (±), with one fragment charge per fragment, adding up to the
total charge; `threads` at least 1; the accuracy in (0, 10]; a finite, positive
finite-difference step. These are `Error::Input` (`Error::UnsupportedElement` for elements
beyond Lr), structures the setup cannot handle (more than 40 neighbors, ...) `Error::Topology`.
No NaN or infinity leaves the crate: a non-finite energy, gradient, charge, Hessian or dipole
derivative is an `Error::Numerical`.

## 9. Validation

`crates/gfnff/tests/reference.rs` (24 cases from each xtb version, `scripts/make_reference.py`:
`tests/reference-xtb-6.8.0/` and `tests/reference-xtb-6.7.1/`): every topology list identical,
real topology parameters, charges and energies to 1e-10, gradients to 1e-9 (achieved, both
versions: charges 8e-14 e, energies 5e-13 Eh, gradients 4e-14 Eh/bohr), including two
evaluations with a topology from another geometry against xtb reading that geometry's
`gfnff_topo`. `crates/gfnff/tests/hessian.rs`: analytic vs Richardson-extrapolated finite
differences (≤ 1e-8 at displaced structures and at exactly linear/planar ones), dipole
derivatives, thread counts (1 vs 4 threads; the same bits twice on 3 threads for (H2O)200).
`crates/gfnff/tests/validation.rs`: the input errors (fragment charges, positions, threads,
settings) and the analytic Hessian of the linear C=O...H case. `crates/axtb-cli/tests/gfnff.rs`:
CLI equals the Rust API bit for bit, XYZ files with `--conventions xtb` reproduce
`xtb mol.xyz --gfnff` (taxol 5e-13 Eh), xtb's `--hess` thermochemistry.

`tests/gfnff_reproduction.py --xtb <binary>` writes `bench/results/gfnff_reproduction_6.8.0.json`
(xtb 6.8.0, conventions `xtb`) and `bench/results/gfnff_reproduction.json` (the 6.7.1 g-xTB build,
conventions `xtb-6.7.1`). The numbers below are 6.7.1's; against 6.8.0 the same 268 structures
reproduce with charges ≤ 2.3e-12 e, energies ≤ 7.0e-13 Eh (terms 8.4e-13), gradients ≤ 4.3e-12
Eh/bohr, `--hess` thermochemistry ≤ 4.8e-13 Eh (also linear CO2 and HCN, which 6.8.0 projects),
and `topology_positions` (6 structures, both versions) ≤ 5e-13 Eh:

- `accuracy`: 269 structures (elements H-Rn, charges -3...+3, open-shell metals, random
  MB16-43-style molecules, HB/XB dimers, zwitterions, water clusters, n-C50H102, crambin with
  H, 640 atoms) through xtb and axtb: 268 with every topology list identical, real topology
  parameters ≤ 5e-11, charges ≤ 4.7e-12 e, energies ≤ 8.4e-13 Eh (each term ≤ 5.5e-13),
  gradients ≤ 9.3e-12 Eh/bohr (an ill-conditioned EEQ system, mb16-08; all others ≤ 7e-14);
  the exception is ferrocenium (§7);
- `hessian`: against `xtb --gfnff --hess`, the numerical Hessian in xtb's conventions gives
  frequencies within xtb's printed 0.01 cm⁻¹, IR intensities ≤ 1.2e-4 km/mol, ZPVE and free
  energies ≤ 2e-11 Eh; the analytic Hessian is within 0.3 cm⁻¹ (xtb's step-0.005 truncation
  error; IR intensities of degenerate modes depend on the basis of their subspace). xtb 6.7.1
  segfaults (`hessian.F90:454`, zero-mode bookkeeping) on the acetone...water dimer, whose water
  H sits exactly on the C=O axis; xtb 6.8.0 runs it and the numerical Hessian reproduces it,
  but the analytic Hessian is undefined there (a numerical error naming the bond, §6): the
  carbonyl HB term's R-C=O...H torsions are undefined on the axis, where the energy jumps
  (grimme-lab/xtb#1516);
- `scaling`: wall time and peak memory of xtb and axtb, 300 to 9900 atoms, with the machine
  load recorded next to each run. xtb (10 GiB address-space cap) needs 9.0 GiB at 3000 water
  atoms and stops with `forrtl: severe (41): insufficient virtual memory` at 5100 (water and
  solvated crambin); axtb needs 460 MiB for energy + gradient and 3.1 GiB for the analytic
  Hessian at 5100 atoms, 1.5 GiB for energy + gradient at 9900.

## 10. Not implemented

Periodic boundary conditions, GBSA/ALPB solvation with GFN-FF, electric fields, the
`--gfnff` options for PDB residue charges (`.CHRG` from residues), xtb's single-precision
EEQ path above 10000 atoms, and xtb's freezing of HB lists between list updates during its
own optimizations.

## 11. xtb 6.8.0 and 6.7.1

From the diff of v6.8.0 (4b8c987) against the g-xTB build (26dd68d), for `src/gfnff` and the
code GFN-FF calls elsewhere in xtb, checked by reproducing both binaries (§9). Rows are the
fields of `ConventionParams`:

| `ConventionParams` | Fortran | xtb 6.7.1 | xtb 6.8.0 |
|---|---|---|---|
| `rcov_aatoau` | `param/covalentradd3.f90` `covalentRadD3 = [...] * aatoau * 4/3`, `mctc/convert.f90` | `autoaa = 0.52917726` | `autoaa = 5.29177210904492434e-1` (mctc-lib's CODATA 2018 value; PR #1478) |
| `pibo_threshold` | `gfnff_ini.f90`, bond loop (`shift`, `fpi`, `btyp`) and torsion loop (`f2`) | `pibo(i) > 0` | `pibo(i) > pibo_thr = 1e3·epsilon` (PR #1494) |
| `unpaired_neighbors` | `gfnff_ini2.F90` `nbondmat_pbc` (reduced list `nbr` before the bond-pair counts) | `nbr(l,i) = nb(l+1,i)` from position j of the original list, once per unpaired neighbor | `nbr(l,i) = nbr(l+1,i)` from the current position `inb` (PR #1494) |
| `fermi_smearing` | `scc_core.f90` `fermismear` (called by `gfnffqmsolve` at 4000 K) | Newton steps until the count is within 1e-9; derivative e^x/(kT(e^x+1)²); x < 50 for f > 0 | count within min(√ε, 1e5·ε); derivative f(1−f)/kT; f = 0 above x = 50, 1 below −50; aufbau start; stops at a vanishing derivative (PR #1474) |
| `tie_norms` | `NORM2` inlined by the compiler (§7) | ifort | ifx 2025 |

Unchanged for molecules: printout (`print_setup`, PR #1438), `gfnff_topo` written for restarts
(#1467; still written by a default run), allocatable work arrays and OpenMP reductions as
critical sections (#1428; the CN loops sum the per-thread arrays into zero-initialized ones,
the same numbers with one thread), the harmonic 2020 bond list (`neigh%blist`, #1425; version
`harmonic2020` only), `get_damat_3d`/`ncoordLatP` (periodic), `frag_hess.f90` (fragmentation of
the optimizer's model Hessian). Of the GFN-FF PRs in the 6.8.0 release notes, #1103 (lanthanides
and actinides), #1105, #1201, #1202, #1204, #1222, #1287, #1288, #1289, #1350 and #1362 are
already in the g-xTB build; #1450 changes `get_bonds` of xtb's topology module, not GFN-FF.

A `Topology` records its conventions, `Model::new` takes the covalent radii from them, and the
HB lists of the energy use the topology's tie rounding for `jth_nb`. With `Xtb671` every
structure of the 269-structure set gives the former port's results bit for bit.
