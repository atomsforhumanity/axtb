# Roadmap

What axtb plans to add after 0.3.0, in order, and the rules that decide what goes in. Nothing
here has a date: each feature is built when a project needs it, in this order unless a need
changes it.

## What goes into axtb

- **Calculations that AFH needs go into its own programs.** If a needed feature is an
  electronic-structure feature of a method axtb has, it goes into axtb. Mapping inputs to an
  external program through a qccodec codec (for example `xtb --efield`) is only a fallback, if a
  need comes up before axtb has the feature.
- **Drivers stay outside.** Optimizations, transition-state searches, IRCs, scans and reaction
  paths run in drivers (qccompute with geomeTRIC or Sella, reaction-path programs) that take
  axtb's gradients and Hessians. Molecular dynamics isn't planned in axtb either.
- **Shared inputs come from qcdata.** External fields and point charges use qcdata's shared
  input when it exists. A program that can't model an input refuses it; it never ignores it.

## Planned, in order

| # | feature | methods | what it reproduces | notes |
|---|---|---|---|---|
| 1 | Biased Hessian (`--bhess`): vibrational analysis and thermochemistry at a geometry that is not a stationary point | GFN2-xTB first, then the others | `xtb --bhess` where its numbers are sound (see below) | |
| 2 | Uniform external electric field and point-charge embedding: energy, gradient (including the gradient on the charges) and Hessian | GFN2-xTB | `xtb --efield` (V/Å), `$embedding` | The field enters the SCF and the response, so Hessians stay analytic. Its potential is −F·r about the input's origin, with no recentering. The record states the field, the origin and that every result includes it. In a `Session` the field is fixed when the session opens. Point charges may change per request, but their number is fixed. |
| 3 | The same field, natively in g-xTB | g-xTB 2.0.1, v1.1 | `xtb --gxtb --efield` (program conventions); the same definition in axtb's conventions | Replaces the workaround of adding GFN2-xTB's E(F) − E(0) to g-xTB. The g-xTB program ignores point charges (docs/gxtb-feature-coverage.md), so g-xTB embedding has no reference to reproduce and would go beyond the program. |
| 4 | GFN1-xTB | GFN1-xTB | xtb 6.8.0 and 6.7.1 | Also the IPEA parametrization for vertical IPs and EAs. |
| 5 | CPCM-X implicit solvation | GFN2-xTB | xtb's `--cpcmx` | g-xTB has no CPCM-X upstream: the program silently runs in the gas phase. axtb keeps refusing it for g-xTB. |

## The biased Hessian: a sound result first

xtb's `--bhess` (single-point Hessian; Spicher and Grimme, J. Chem. Theory Comput. 2021, 17,
1701, doi:10.1021/acs.jctc.0c01306) adds an RMSD bias potential so that the geometry becomes a
stationary point. It then computes the Hessian there and applies the paper's corrections.

axtb's version aims for the **physically sound result**. It matches xtb only where xtb's numbers
are credible. Where axtb differs, the docs and docs/validation.md say how and why, with
evidence, for example:
- a setting or tolerance of xtb's that leaves the result unconverged;
- a step that depends on the frame or the numerical path.

As elsewhere in axtb, a program-conventions mode can reproduce xtb's numbers exactly, when that
is useful for checking. The default is the sound result.
