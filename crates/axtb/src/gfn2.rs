//! GFN2-xTB parametrization (C. Bannwarth, S. Ehlert, S. Grimme, JCTC 15, 1652 (2019)).

use crate::basis::{slater_to_gauss, Basis, StoExpansion};
use crate::coulomb::{DampedMultipole, EffectiveCoulomb, OnsiteThirdOrder};
use crate::error::{Error, Result};
use crate::hamiltonian::H0Params;
use crate::molecule::Molecule;
use crate::ncoord::Ncoord;
use crate::param::gfn2_data::*;
use crate::repulsion::Repulsion;
use crate::xtb::XtbModel;
use crate::ConventionParams;

pub const MAX_ELEM: usize = 86;

const WEXP: f64 = 0.5;
const ENSCALE: f64 = 2.0e-2;
const KDIAG: [f64; 5] = [1.85, 2.23, 2.23, 2.23, 2.23];
const SHELL_HUBBARD_DERIVS: [f64; 5] = [1.0, 0.5, 0.25, 0.25, 0.25];
const GEXP: f64 = 2.0;

fn kshell(k: usize, l: usize) -> f64 {
    if (k == 2 && (l == 0 || l == 1)) || (l == 2 && (k == 0 || k == 1)) {
        2.0
    } else {
        (KDIAG[l] + KDIAG[k]) / 2.0
    }
}

/// Build the GFN2-xTB model for a molecule with the settings of a convention
/// ([`Conventions::params`](crate::Conventions::params)).
pub fn build(mol: &Molecule, acc: f64, params: &ConventionParams) -> Result<XtbModel> {
    if let Some(&z) = mol.species.iter().find(|&&z| z > MAX_ELEM) {
        return Err(Error::UnsupportedElement { method: "GFN2-xTB", z });
    }
    let nid = mol.nid();
    let zs = &mol.species;

    // basis
    let mut cgto = Vec::with_capacity(nid);
    for &z in zs {
        let iz = z - 1;
        let mut shells = Vec::new();
        for ish in 0..NSHELL[iz] {
            shells.push(slater_to_gauss(
                NUMBER_OF_PRIMITIVES[iz][ish],
                PRINCIPAL_QUANTUM_NUMBER[iz][ish],
                ANG_SHELL[iz][ish],
                SLATER_EXPONENT[iz][ish],
                true,
                StoExpansion::Gfn2,
            )?);
        }
        cgto.push(shells);
    }
    let mut basis = Basis::new(mol, cgto, 1.0);
    // integral screening: primitive pairs (xtb: no upper clip at high accuracy) and, in
    // tblite, atom pairs beyond a distance
    basis.intcut = (params.primitive_cutoff)(acc);
    let pair_cutoff = (params.pair_cutoff)(&basis, acc);
    let en = params.pauling_en;
    let ncoord = Ncoord::with_radii(mol, params.cn_rcov, params.cn_cutoff);

    // core Hamiltonian
    let per_shell = |f: &dyn Fn(usize, usize) -> f64| -> Vec<Vec<f64>> {
        zs.iter().map(|&z| (0..NSHELL[z - 1]).map(|ish| f(z - 1, ish)).collect()).collect()
    };
    let ang = |iz: usize, ish: usize| ANG_SHELL[iz][ish];
    let selfenergy = per_shell(&|iz, ish| P_SELFENERGY[iz][ish]);
    let kcn = per_shell(&|iz, ish| P_KCN[iz][ang(iz, ish)]);
    let shpoly = per_shell(&|iz, ish| P_SHPOLY[iz][ang(iz, ish)]);
    let refocc = per_shell(&|iz, ish| REFERENCE_OCC[iz][ang(iz, ish)]);
    let rad = zs.iter().map(|&z| (params.atomic_rad)(z)).collect();
    let mut hscale = vec![vec![Vec::new(); nid]; nid];
    for (isp, &zi) in zs.iter().enumerate() {
        for (jsp, &zj) in zs.iter().enumerate() {
            let den = (en(zi) - en(zj)).powi(2);
            let (iz, jz) = (zi - 1, zj - 1);
            let mut block = vec![vec![0.0; NSHELL[jz]]; NSHELL[iz]];
            for ish in 0..NSHELL[iz] {
                let il = ang(iz, ish);
                for jsh in 0..NSHELL[jz] {
                    let jl = ang(jz, jsh);
                    let (za, zb) = (SLATER_EXPONENT[iz][ish], SLATER_EXPONENT[jz][jsh]);
                    let zij = (2.0 * (za * zb).sqrt() / (za + zb)).powf(WEXP);
                    let enp = 1.0 + ENSCALE * den;
                    let km = kshell(jl, il) * enp;
                    block[ish][jsh] = zij * km;
                }
            }
            hscale[isp][jsp] = block;
        }
    }
    let h0 = H0Params { selfenergy, kcn, shpoly, refocc, rad, hscale };

    // electrostatics
    let hardness = per_shell(&|iz, ish| HUBBARD_PARAMETER[iz] * SHELL_HUBBARD[iz][ang(iz, ish)]);
    let es2 = EffectiveCoulomb::new(&hardness, GEXP);
    let es3 =
        OnsiteThirdOrder { gamma: per_shell(&|iz, ish| P_HUBBARD_DERIVS[iz] * SHELL_HUBBARD_DERIVS[ang(iz, ish)]) };
    let aes2 = DampedMultipole::new(
        zs.iter().map(|&z| P_DKERNEL[z - 1]).collect(),
        zs.iter().map(|&z| P_QKERNEL[z - 1]).collect(),
        zs.iter().map(|&z| P_RAD[z - 1]).collect(),
        zs.iter().map(|&z| P_VCN[z - 1]).collect(),
        ncoord.clone(),
    );
    let alpha: Vec<f64> = zs.iter().map(|&z| REP_ALPHA[z - 1]).collect();
    let zeff: Vec<f64> = zs.iter().map(|&z| REP_ZEFF[z - 1]).collect();
    let repulsion = Repulsion::new(mol, &alpha, &zeff, 1.5, 1.0, 1.0).with_cutoff(params.repulsion_cutoff);

    Ok(XtbModel { basis, h0, ncoord, repulsion, es2, es3: Some(es3), aes2: Some(aes2), pair_cutoff, params: *params })
}
