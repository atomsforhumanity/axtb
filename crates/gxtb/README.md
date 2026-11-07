# axtb-gxtb

g-xTB (T. Froitzheim, M. Müller, A. Hansen, S. Grimme, ChemRxiv 2025,
doi:10.26434/chemrxiv-2025-bjxvt) in Rust, in two versions:

- **2.0.1**, the version of the grimme-lab/g-xtb 2.0.1 release (`xtb --gxtb`): energies,
  analytic gradients, analytic (default), response and numerical Hessians, finite electronic
  temperature;
- **v1.1**, the public preprint version as the g-xTB 1.1.0 program implements it: energies,
  analytic gradients, semi-numerical Hessians, finite electronic temperature.

In the program conventions (`Conventions::Gxtb`) it reproduces each program's numbers to the
last digits; the default `Conventions::Axtb` uses CODATA 2022 units and, for v1.1, the exact
derivative of the energy as the Fock matrix. `CompositeCalculator` adds GFN2-xTB's ALPB/GBSA
solvation energy (g-xTB has no implicit solvent of its own) in one process. Molecules only.

Part of [axtb](https://github.com/atomsforhumanity/axtb); the specifications
([docs/gxtb-v1.1-spec.md](https://github.com/atomsforhumanity/axtb/blob/main/docs/gxtb-v1.1-spec.md),
[docs/gxtb-v2.0.1-spec.md](https://github.com/atomsforhumanity/axtb/blob/main/docs/gxtb-v2.0.1-spec.md))
document every term and where it comes from.

License: GPL-3.0-only (`COPYING`). The crate embeds the parameter files of the g-xTB 1.1.0
release (`param/`) and tables taken from the g-xTB 1.1.0 and 2.0.1 programs (grimme-lab/g-xtb,
GPL-3.0). `NOTICE` credits the upstream work.
