//! Fixed parameters, thresholds and element helpers of GFN-FF.
//!
//! Port of `newGFNFFGenerator` and `gfnff_set_param` (xtb `src/gfnff/gfnff_param.f90`),
//! `gfnff_thresholds`, `itabrow6`, `zeta` and the factors of `gfnffrab`. xtb compiles with
//! `-fdefault-real-8`/`-r8`, so every literal of the Fortran is a double; the digits are
//! kept as written there. Element tables are generated (`data.rs`).

use crate::data;

/// Bohr radius in Angstrom written into GFN-FF's own setup code (`gfnff_ini`'s `bohr`, the
/// topological distances of the charge model; unchanged in xtb 6.8.0). The covalent radii
/// take the bohr of the conventions ([`crate::conventions::ConventionParams::rcov_aatoau`]).
pub const AUTOAA: f64 = 0.52917726;
/// Hartree to eV of xtb (`autoev`).
pub const AUTOEV: f64 = 27.21138505;
/// Boltzmann constant in Eh/K of xtb (`kB`; from `axtb-base`).
pub use axtb_base::units::reference::KB_XTB_TBLITE as KB;
/// pi (xtb writes 3.1415926535897932384626433832795029, which is this value).
pub const PI: f64 = std::f64::consts::PI;
/// sqrt(pi) of `xtb_mctc_constants`.
pub fn sqrtpi() -> f64 {
    PI.sqrt()
}
/// sqrt(2/pi) as written in xtb's EEQ (`tsqrt2pi` of `goed_gfnff` and `goedeckera`).
pub const TSQRT2PI: f64 = 0.797884560802866;

/// Number of elements with GFN-FF parameters (H-Lr).
pub const MAX_ELEM: usize = 103;

/// H-H factors of the non-bonded repulsion exponent (`hhfac`, `hh13rep`, `hh14rep` of
/// `newGFNFFGenerator`): every H-H pair, and 1-3 and 1-4 pairs on top.
pub const HHFAC: f64 = 0.6290;
pub const HH13REP: f64 = 1.4580;
pub const HH14REP: f64 = 0.7080;

/// Constants of the topology generator (`TGFFGenerator`, `newGFNFFGenerator`).
#[derive(Clone, Debug)]
pub struct Gen {
    pub cnmax: f64,
    pub linthr: f64,
    pub fcthr: f64,
    /// single precision in xtb (`real(sp)`)
    pub tdist_thr: f32,
    pub rthr: f64,
    pub rthr2: f64,
    pub rqshrink: f64,
    pub hqabthr: f64,
    pub qabthr: f64,
    pub srb1: f64,
    pub srb2: f64,
    pub srb3: f64,
    pub qrepscal: f64,
    pub nrepscal: f64,
    /// `bstren(1..9)`, index 0 unused
    pub bstren: [f64; 10],
    pub qfac_ben: f64,
    pub qfac_tor: f64,
    pub fr3: f64,
    pub fr4: f64,
    pub fr5: f64,
    pub fr6: f64,
    /// `torsf(1..8)`, index 0 unused (`torsf(4)` is never set in xtb)
    pub torsf: [f64; 9],
    pub fbs1: f64,
    pub batmscal: f64,
    pub mchishift: f64,
    pub rabshift: f64,
    pub rabshifth: f64,
    pub hyper_shift: f64,
    pub hshift3: f64,
    pub hshift4: f64,
    pub hshift5: f64,
    pub metal1_shift: f64,
    pub metal2_shift: f64,
    pub metal3_shift: f64,
    pub eta_shift: f64,
    /// `qfacbm(0:4)`
    pub qfacbm: [f64; 5],
    pub qfacbm0: f64,
    pub rfgoed1: f64,
    pub htriple: f64,
    pub hueckelp2: f64,
    pub hueckelp3: f64,
    /// `hdiag(1..17)`, index 0 unused; unset entries are 0
    pub hdiag: [f64; 18],
    /// `hoffdiag(1..17)`, index 0 unused; unset entries are 0
    pub hoffdiag: [f64; 18],
    pub hiter: f64,
    pub hueckelp: f64,
    pub bzref: f64,
    pub bzref2: f64,
    pub pilpf: f64,
    pub maxhiter: f64,
    pub d3a1: f64,
    pub d3a2: f64,
    pub split0: f64,
    pub split1: f64,
    pub fringbo: f64,
    pub aheavy3: f64,
    pub aheavy4: f64,
    /// `bsmat(0:3,0:3)`, `bsmat[a][b]` = Fortran `bsmat(a,b)`
    pub bsmat: [[f64; 4]; 4],
}

impl Default for Gen {
    fn default() -> Self {
        let mut g = Gen {
            cnmax: 4.4,
            linthr: 160.,
            fcthr: 1.0e-3,
            tdist_thr: 12.,
            rthr: 1.25,
            rthr2: 1.00,
            rqshrink: 0.23,
            hqabthr: 0.01,
            qabthr: 0.10,
            srb1: 0.3731,
            srb2: 0.3171,
            srb3: 0.2538,
            qrepscal: 0.3480,
            nrepscal: -0.1270,
            bstren: [0.0; 10],
            qfac_ben: -0.54,
            qfac_tor: 12.0e0,
            fr3: 0.3,
            fr4: 1.0,
            fr5: 1.5,
            fr6: 5.7,
            torsf: [0.0; 9],
            fbs1: 0.50,
            batmscal: 0.30e0,
            mchishift: -0.09e0,
            rabshift: -0.110,
            rabshifth: -0.050,
            hyper_shift: 0.03,
            hshift3: -0.11,
            hshift4: -0.11,
            hshift5: -0.06,
            metal1_shift: 0.2,
            metal2_shift: 0.15,
            metal3_shift: 0.05,
            eta_shift: 0.040,
            qfacbm: [0.0; 5],
            qfacbm0: 0.047,
            rfgoed1: 1.175,
            htriple: 1.45e0,
            hueckelp2: 1.00e0,
            hueckelp3: -0.24e0,
            hdiag: [0.0; 18],
            hoffdiag: [0.0; 18],
            hiter: 0.700e0,
            hueckelp: 0.340e0,
            bzref: 0.370e0,
            bzref2: 0.315e0,
            pilpf: 0.530e0,
            maxhiter: 5.0,
            d3a1: 0.58e0,
            d3a2: 4.80e0,
            split0: 0.670e0,
            split1: 0.0,
            fringbo: 0.020e0,
            aheavy3: 89.,
            aheavy4: 100.,
            bsmat: [[-999.; 4]; 4],
        };
        g.bstren[1] = 1.00e0; // single bond
        g.bstren[2] = 1.24e0; // double bond
        g.bstren[3] = 1.98e0; // triple bond
        g.bstren[4] = 1.22e0; // hyperval bond
        g.bstren[5] = 1.00e0; // M-X
        g.bstren[6] = 0.78e0; // M eta
        g.bstren[7] = 3.40e0; // M-M
        g.bstren[8] = 3.40e0; // M-M
        g.torsf[1] = 1.00; // single bond
        g.torsf[2] = 1.18; // pi bond
        g.torsf[3] = 1.05; // improper
        g.torsf[5] = 0.50; // pi part improper
        g.torsf[6] = -0.90; // extra sp3 C
        g.torsf[7] = 0.70; // extra sp3 N
        g.torsf[8] = -2.00; // extra sp3 O
        g.qfacbm[0] = 1.0e0;
        g.qfacbm[1] = -0.2e0;
        g.qfacbm[2] = -0.2e0;
        g.qfacbm[3] = 0.70e0;
        g.qfacbm[4] = 0.50e0;
        g.hdiag[5] = -0.5e0;
        g.hdiag[6] = 0.00e0;
        g.hdiag[7] = 0.14e0;
        g.hdiag[8] = -0.38e0;
        g.hdiag[9] = -0.29e0;
        g.hdiag[16] = -0.30e0;
        g.hdiag[17] = -0.30e0;
        g.hoffdiag[5] = 0.5e0;
        g.hoffdiag[6] = 1.00e0;
        g.hoffdiag[7] = 0.66e0;
        g.hoffdiag[8] = 1.10e0;
        g.hoffdiag[9] = 0.23e0;
        g.hoffdiag[16] = 0.60e0;
        g.hoffdiag[17] = 1.00e0;
        g.split1 = 1.0e0 - g.split0;
        g.bsmat[0][0] = g.bstren[1];
        g.bsmat[3][0] = g.bstren[1];
        g.bsmat[3][3] = g.bstren[1];
        g.bsmat[2][2] = g.bstren[2];
        g.bsmat[1][1] = g.bstren[3];
        g.bsmat[1][0] = g.split0 * g.bstren[1] + g.split1 * g.bstren[3];
        g.bsmat[3][1] = g.split0 * g.bstren[1] + g.split1 * g.bstren[3];
        g.bsmat[2][1] = g.split0 * g.bstren[2] + g.split1 * g.bstren[3];
        g.bsmat[2][0] = g.split0 * g.bstren[1] + g.split1 * g.bstren[2];
        g.bsmat[3][2] = g.split0 * g.bstren[1] + g.split1 * g.bstren[2];
        g.bstren[9] = 0.5 * (g.bstren[7] + g.bstren[8]);
        g
    }
}

/// Global parameters of the force field (`TGFFData`, set in `gfnff_set_param`).
#[derive(Clone, Debug)]
pub struct Param {
    pub cnmax: f64,
    pub atcuta: f64,
    pub atcutt: f64,
    pub repscalb: f64,
    pub repscaln: f64,
    pub hbacut: f64,
    pub hbscut: f64,
    pub xbacut: f64,
    pub xbscut: f64,
    pub hbsf: f64,
    pub hbst: f64,
    pub xbsf: f64,
    pub xbst: f64,
    pub hbalp: f64,
    pub hblongcut: f64,
    pub hblongcut_xb: f64,
    pub hbabmix: f64,
    pub hbnbcut: f64,
    pub tors_hb: f64,
    pub bend_hb: f64,
    pub vbond_scale: f64,
    pub xhaci_globabh: f64,
    pub xhaci_coh: f64,
    pub xhaci_glob: f64,
    /// dispersion scaling (`set%dispscale`, default 1)
    pub dispscale: f64,
    /// HB/XB basicity per element (index Z-1)
    pub xhbas: [f64; MAX_ELEM],
    /// HB acidity per element
    pub xhaci: [f64; MAX_ELEM],
    /// XB acidity per element
    pub xbaci: [f64; MAX_ELEM],
    /// bonded ATM prefactor per element
    pub zb3atm: [f64; MAX_ELEM],
    /// D3 R0^2 per element pair, packed lower triangle `lin(i,j)` (0-based)
    pub d3r0: Vec<f64>,
    /// D3 covalent radii (bohr, x4/3) per element
    pub rcov: [f64; 118],
    /// sqrt(Z `<r4>/<r2>`) per element
    pub sqrtzr4r2: [f64; 118],
}

impl Param {
    /// The parameters of `gfnff_set_param` (with the covalent radii of `conv`).
    pub fn new(gen: &Gen, conv: &crate::conventions::ConventionParams) -> Self {
        let mut p = Param {
            cnmax: 4.4,
            atcuta: 0.595e0,
            atcutt: 0.505e0,
            repscalb: 1.7583,
            repscaln: 0.4270,
            hbacut: 49.,
            hbscut: 22.,
            xbacut: 70.,
            xbscut: 5.,
            hbsf: 1.,
            hbst: 15.,
            xbsf: 0.03,
            xbst: 15.,
            hbalp: 6.,
            hblongcut: 85.,
            hblongcut_xb: 70.,
            hbabmix: 0.80,
            hbnbcut: 11.20,
            tors_hb: 0.94,
            bend_hb: 0.20,
            vbond_scale: 0.9,
            xhaci_globabh: 0.268,
            xhaci_coh: 0.350,
            xhaci_glob: 1.50,
            dispscale: 1.0,
            xhbas: [0.0; MAX_ELEM],
            xhaci: [0.0; MAX_ELEM],
            xbaci: [0.0; MAX_ELEM],
            zb3atm: [0.0; MAX_ELEM],
            d3r0: vec![0.0; MAX_ELEM * (MAX_ELEM + 1) / 2],
            rcov: [0.0; 118],
            sqrtzr4r2: [0.0; 118],
        };
        let z = |n: usize| n - 1;
        p.xhbas[z(6)] = 0.80e0;
        p.xhbas[z(7)] = 1.68e0;
        p.xhbas[z(8)] = 0.67e0;
        p.xhbas[z(9)] = 0.52e0;
        p.xhbas[z(14)] = 4.0e0;
        p.xhbas[z(15)] = 3.5e0;
        p.xhbas[z(16)] = 2.0e0;
        p.xhbas[z(17)] = 1.5e0;
        p.xhbas[z(35)] = 1.5e0;
        p.xhbas[z(53)] = 1.9e0;
        p.xhbas[z(33)] = p.xhbas[z(15)];
        p.xhbas[z(34)] = p.xhbas[z(16)];
        p.xhbas[z(51)] = p.xhbas[z(15)];
        p.xhbas[z(52)] = p.xhbas[z(16)];
        p.xhaci[z(6)] = 0.75;
        p.xhaci[z(7)] = p.xhaci_glob + 0.1;
        p.xhaci[z(8)] = p.xhaci_glob;
        p.xhaci[z(9)] = p.xhaci_glob;
        p.xhaci[z(15)] = p.xhaci_glob;
        p.xhaci[z(16)] = p.xhaci_glob;
        p.xhaci[z(17)] = p.xhaci_glob + 1.0;
        p.xhaci[z(35)] = p.xhaci_glob + 1.0;
        p.xhaci[z(53)] = p.xhaci_glob + 1.0;
        p.xbaci[z(15)] = 1.0e0;
        p.xbaci[z(16)] = 1.0e0;
        p.xbaci[z(17)] = 0.5e0;
        p.xbaci[z(33)] = 1.2e0;
        p.xbaci[z(34)] = 1.2e0;
        p.xbaci[z(35)] = 0.9e0;
        p.xbaci[z(51)] = 1.2e0;
        p.xbaci[z(52)] = 1.2e0;
        p.xbaci[z(53)] = 1.2e0;
        for i in 0..118 {
            // covalentRadD3 = r * aatoau * 4/3 with the conventions' bohr
            p.rcov[i] = data::COVRAD_D3_AA[i] * conv.rcov_aatoau * 4.0 / 3.0;
            // sqrt(0.5*(r4Overr2*sqrt(Z)))
            p.sqrtzr4r2[i] = (0.5 * (data::R4_OVER_R2[i] * ((i + 1) as f64).sqrt())).sqrt();
        }
        let mut k = 0;
        for i in 0..MAX_ELEM {
            let dum = (i + 1) as f64;
            p.zb3atm[i] = -dum * gen.batmscal.powf(1.0 / 3.0);
            for j in 0..=i {
                let dum = p.sqrtzr4r2[i] * p.sqrtzr4r2[j] * 3.0;
                let r = gen.d3a1 * dum.sqrt() + gen.d3a2;
                p.d3r0[k] = r * r;
                k += 1;
            }
        }
        p.zb3atm[0] = -0.25 * gen.batmscal.powf(1.0 / 3.0);
        p
    }
}

/// Cutoffs (squared distances, bohr²) from the accuracy (`gfnff_thresholds`).
#[derive(Clone, Copy, Debug)]
pub struct Thresholds {
    pub disp: f64,
    pub cn: f64,
    pub rep: f64,
    pub hb1: f64,
    pub hb2: f64,
}

impl Thresholds {
    pub fn new(accuracy: f64) -> Self {
        let l = accuracy.log10();
        Thresholds {
            disp: 1500.0 - l * 1000.0,
            cn: 100.0 - l * 50.0,
            rep: 400.0 - l * 100.0,
            hb1: 200.0 - l * 50.0,
            hb2: 400.0 - l * 50.0,
        }
    }
}

/// Packed lower-triangle index of the element pair (0-based atomic numbers - 1).
#[inline]
pub fn lin(i: usize, j: usize) -> usize {
    let (a, b) = if i > j { (i, j) } else { (j, i) };
    a * (a + 1) / 2 + b
}

/// Row of the periodic table (`itabrow6`), Z = atomic number.
pub fn itabrow6(z: usize) -> usize {
    match z {
        1..=2 => 1,
        3..=10 => 2,
        11..=18 => 3,
        19..=36 => 4,
        37..=54 => 5,
        z if z > 54 => 6,
        _ => 0,
    }
}

#[inline]
pub fn metal(z: usize) -> i32 {
    data::METAL[z - 1]
}
#[inline]
pub fn group(z: usize) -> i32 {
    data::GROUP[z - 1]
}
#[inline]
pub fn normcn(z: usize) -> i32 {
    data::NORMCN[z - 1]
}
#[inline]
pub fn en(z: usize) -> f64 {
    data::EN[z - 1]
}
#[inline]
pub fn rad(z: usize) -> f64 {
    data::RAD[z - 1]
}
#[inline]
pub fn repz(z: usize) -> f64 {
    data::REPZ[z - 1]
}

/// Possible pi atoms (B, C, N, O, F, S, Cl).
pub fn pilist(z: usize) -> bool {
    matches!(z, 5 | 6 | 7 | 8 | 9 | 16 | 17)
}
/// N, O, F, S, Cl.
pub fn nofs(z: usize) -> bool {
    matches!(z, 7 | 8 | 9 | 16 | 17)
}
/// X in A-X...B halogen/chalcogen/pnictogen bonds.
pub fn xatom(z: usize) -> bool {
    matches!(z, 17 | 35 | 53 | 16 | 34 | 52 | 15 | 33 | 51)
}

/// Charge scaling function of the dispersion (`zeta` in `gfnff_ini`), qref = 0.
pub fn zeta(z: usize, q: f64) -> f64 {
    let zeff = data::ZETA_ZEFF[z - 1];
    let c = data::ZETA_C[z - 1];
    let qmod = zeff + q;
    if qmod < 0.0 {
        (3.0f64).exp()
    } else {
        (3.0 * (1.0 - (c * (1.0 - zeff / qmod)).exp())).exp()
    }
}

/// `p(6,2)` of `gfnffrab`: EN polynomial parameters x 10^3, by table row.
const RAB_P: [[f64; 2]; 6] = [
    [29.84522887, -8.87843763],
    [-1.70549806, 2.10878369],
    [6.54013762, 0.08009374],
    [6.39169003, -0.85808076],
    [6.00000000, -1.15000000],
    [5.60000000, -1.30000000],
];

/// `scaleF` of `gfnffrab`: F/Cl/Br/I bonds to lanthanides and actinides.
pub fn rab_scale(zi: usize, zj: usize) -> f64 {
    let lnan = |z: usize| (57..=71).contains(&z) || (89..=103).contains(&z);
    let f = |x: usize| match x {
        9 => Some(0.93533568),
        17 => Some(1.0190114),
        35 => Some(1.0425532),
        53 => Some(1.0551948),
        _ => None,
    };
    // the Fortran assigns the four halogen blocks in order (F, Cl, Br, I); the blocks
    // never overlap, so the order does not matter
    if lnan(zi) {
        if let Some(s) = f(zj) {
            return s;
        }
    }
    if lnan(zj) {
        if let Some(s) = f(zi) {
            return s;
        }
    }
    1.0
}

/// Prefactors of the bond-length estimate of the pair: `(ff*scaleF, k1, k2)` parts,
/// returned as `(fac)` so that `rab = (ra + rb [+ shift]) * fac` with
/// `fac = ff * scaleF(ati,atj)` exactly as `gfnffrab`/`gfnffdrab` evaluate it.
#[inline]
pub fn rab_ff(zi: usize, zj: usize) -> f64 {
    let ir = itabrow6(zi) - 1;
    let jr = itabrow6(zj) - 1;
    let den = (data::RAB_EN[zi - 1] - data::RAB_EN[zj - 1]).abs();
    let k1 = 0.005e0 * (RAB_P[ir][0] + RAB_P[jr][0]);
    let k2 = 0.005e0 * (RAB_P[ir][1] + RAB_P[jr][1]);
    1.0e0 - k1 * den - k2 * den.powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn thresholds_default() {
        let t = Thresholds::new(0.1);
        assert_eq!(t.cn, 150.0);
        assert_eq!(t.rep, 500.0);
        assert_eq!(t.disp, 2500.0);
        assert_eq!(t.hb1, 250.0);
        assert_eq!(t.hb2, 450.0);
    }

    /// axtb-base's covalent radii (Å) equal xtb's `covalentRadD3` literals bit for bit.
    #[test]
    fn base_covalent_radii_equal_the_gfnff_copy() {
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&axtb_base::elements::COVALENT_RAD_2009_AA), bits(&data::COVRAD_D3_AA));
    }
}
