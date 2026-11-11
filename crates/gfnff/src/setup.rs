//! GFN-FF topology setup: a port of xtb's `gfnff_ini` (`src/gfnff/gfnff_ini.f90`) and
//! `gfnff_neigh` (`gfnff_ini2.F90`) for molecules.
//!
//! The order of every loop and every floating-point operation follows the Fortran, so the
//! topology (neighbor lists, hybridization, rings, pi systems, charges, force constants)
//! is that of xtb. Differences in data layout: no N x N arrays (the topological distance
//! matrix of the charge model is kept sparse, `bpair` is a map, pair lists are generated
//! with cell lists), and the pi-system charge estimate solves the EEQ once per charged
//! fragment instead of once per pi system (same numbers).

use crate::cells::{dist2, CellList};
use crate::cn::LogCn;
use crate::conventions::{ConventionParams, Conventions, FermiSmearing, UnpairedNeighbors};
use crate::data;
use crate::error::{Error, Result};
use crate::geom;
use crate::linalg;
use crate::param::{
    self, en, group, metal, nofs, normcn, pilist, rad, repz, xatom, Gen, Param, Thresholds, PI, TSQRT2PI,
};
use crate::real::erf;
use crate::setup_util::*;
use crate::topology::*;
use faer::{Mat, Par};
use std::collections::HashMap;

/// Options of the topology setup.
#[derive(Clone, Debug, Default)]
pub struct SetupOptions {
    /// Accuracy of the cutoffs; default: xtb's (0.1, or 2.0 above 10000 atoms).
    pub accuracy: Option<f64>,
    /// Charges of the molecular fragments (see [`crate::Options::fragment_charges`]).
    pub fragment_charges: Option<Vec<f64>>,
    /// Parallelism of the dense linear algebra.
    pub par: Option<Par>,
    /// xtb version whose GFN-FF is set up (default: 6.8.0).
    pub conventions: Conventions,
}

/// `fat` element factors of `gfnff_neigh`.
fn fat(z: usize) -> f64 {
    match z {
        1 => 1.02,
        4 => 1.03,
        5 => 1.02,
        8 => 1.02,
        9 => 1.05,
        10 => 1.10,
        11 => 1.01,
        12 => 1.02,
        15 => 0.97,
        18 => 1.10,
        19 => 1.02,
        20 => 1.02,
        38 => 1.02,
        34 => 0.99,
        50 => 1.01,
        51 => 0.99,
        52 => 0.95,
        53 => 0.98,
        56 => 1.02,
        76 => 1.02,
        82 => 1.06,
        83 => 0.95,
        _ => 1.0,
    }
}

const RABD_CUTOFF: f32 = 13.0;

struct Ctx<'a> {
    n: usize,
    at: &'a [usize],
    xyz: &'a [[f64; 3]],
    gen: Gen,
    param: Param,
    conv: ConventionParams,
    cells: CellList,
    par: Par,
    // evolving state
    qa: Vec<f64>,
    hyb: Vec<i32>,
    itag: Vec<i32>,
    nbf: Nb,
    nb: Nb,
    nbm: Nb,
    tag: Vec<bool>,
    mchar: Vec<f64>,
    imetal: Vec<i32>,
    nfrag: usize,
    fraglist: Vec<usize>,
    qfrag: Vec<f64>,
}

/// Atoms closer than this (bohr) are an input error (the limit of `axtb::Molecule` too).
pub(crate) const MIN_DISTANCE: f64 = 0.1;

/// Check the positions (bohr) of a structure of `n` atoms: one finite position per atom and
/// no two atoms closer than [`MIN_DISTANCE`].
pub(crate) fn check_positions(positions: &[[f64; 3]], n: usize) -> Result<()> {
    if positions.len() != n {
        return Err(Error::Input(format!("expected {n} positions, got {}", positions.len())));
    }
    if let Some(i) = positions.iter().position(|p| p.iter().any(|x| !x.is_finite())) {
        return Err(Error::Input(format!("the position of atom {} is not finite: {:?}", i + 1, positions[i])));
    }
    let cells = CellList::new(positions, 1.0);
    let mut cand = Vec::new();
    for i in 0..n {
        cells.candidates(i, MIN_DISTANCE, |j| j < i, &mut cand);
        for &j in &cand {
            let r = dist2(positions[i], positions[j]).sqrt();
            if r < MIN_DISTANCE {
                return Err(Error::Input(format!(
                    "atoms {} and {} are {r:.3e} bohr apart, closer than {MIN_DISTANCE} bohr",
                    j + 1,
                    i + 1
                )));
            }
        }
    }
    Ok(())
}

/// Build the GFN-FF topology of a molecule (positions in bohr).
pub(crate) fn setup(numbers: &[usize], positions: &[[f64; 3]], charge: f64, opts: &SetupOptions) -> Result<Topology> {
    let n = numbers.len();
    if n == 0 {
        return Err(Error::Input("no atoms".into()));
    }
    if let Some(&z) = numbers.iter().find(|&&z| z == 0 || z > param::MAX_ELEM) {
        return Err(Error::UnsupportedElement { method: "GFN-FF", z });
    }
    check_positions(positions, n)?;
    // charges beyond the nuclear charge have no meaning (and would overflow the pi-electron
    // counts of the setup)
    let zsum = numbers.iter().sum::<usize>() as f64;
    if charge.is_nan() || charge.abs() > zsum {
        return Err(Error::Input(format!("the charge must be within ±{zsum} (the nuclear charge), not {charge}")));
    }
    let accuracy = opts.accuracy.unwrap_or(if n > 10000 { 2.0 } else { 0.1 });
    // above 10^1.5 the dispersion cutoff of `Thresholds` turns negative
    if !(accuracy > 0.0 && accuracy <= 10.0) {
        return Err(Error::Input(format!(
            "the GFN-FF accuracy must be in (0, 10], not {accuracy} (xtb: 0.1, and 2.0 above 10000 atoms)"
        )));
    }
    let thr = Thresholds::new(accuracy);
    let gen = Gen::default();
    let conv = opts.conventions.params();
    let param = Param::new(&gen, &conv);
    let cells = CellList::new(positions, 8.0);
    let mut c = Ctx {
        n,
        at: numbers,
        xyz: positions,
        gen,
        param,
        conv,
        cells,
        par: opts.par.unwrap_or(Par::Seq),
        qa: vec![0.0; n],
        hyb: vec![0; n],
        itag: vec![0; n],
        nbf: vec![Vec::new(); n],
        nb: vec![Vec::new(); n],
        nbm: vec![Vec::new(); n],
        tag: vec![false; n],
        mchar: vec![0.0; n],
        imetal: vec![0; n],
        nfrag: 0,
        fraglist: vec![0; n],
        qfrag: vec![0.0; n],
    };

    // fragment charges (gfnff_input): the total charge on fragment 1 unless given. Given ones
    // must add up to the total charge (xtb's test; xtb zeroes them otherwise) and be one per
    // fragment (xtb ignores extra ones; checked once the fragments are known).
    c.qfrag[0] = charge;
    if let Some(fc) = &opts.fragment_charges {
        if fc.iter().any(|q| !q.is_finite() || q.abs() > zsum) {
            return Err(Error::Input(format!(
                "fragment charges must be within ±{zsum} (the nuclear charge), not {fc:?}"
            )));
        }
        let sum: f64 = fc.iter().sum();
        if (charge - sum).abs() >= f64::EPSILON.sqrt() {
            return Err(Error::Input(format!(
                "the fragment charges {fc:?} add up to {sum}, not to the total charge {charge}"
            )));
        }
        if fc.len() > n {
            return Err(Error::Input(format!("{} fragment charges for {n} atoms", fc.len())));
        }
        c.qfrag[..fc.len()].copy_from_slice(fc);
    }

    // coordination number and metallic character
    let lcn = LogCn::new(numbers, positions, thr.cn, &c.param, Some(&c.cells));
    c.mchar = lcn.metal_char(numbers);

    let mut adj_lnan = numbers.iter().any(|&z| (57..=71).contains(&z) || (89..=103).contains(&z));

    // the two passes of the topology/charge loop
    let mut qloop_count = 0;
    let mut bpair = BondDistance::new(n);
    let mut piadr_list: Vec<usize> = Vec::new(); // pi candidates (atoms), 'piadr(k)'
    let mut piadr2: Vec<usize> = vec![0; n]; // atom -> 1-based candidate index
    let mut pimvec: Vec<usize> = Vec::new();
    let mut picount = 0;
    let mut dxi = vec![0.0; n];
    let mut ipis: Vec<i64> = Vec::new();
    let mut blist: Vec<(usize, usize)> = Vec::new(); // (j, i) = (blist(1), blist(2))
    let mut chieeq = vec![0.0; n];
    let mut gameeq = vec![0.0; n];
    let mut alpeeq = vec![0.0; n];
    while (qloop_count < 2 && c.gen.rqshrink > 1.0e-3) || adj_lnan {
        gfnff_neigh(&mut c, qloop_count > 0)?;
        if adj_lnan && qloop_count != 0 {
            adjust_nb_lnh_anh(&mut c);
            adj_lnan = false;
        }
        for i in 0..n {
            c.imetal[i] = metal(numbers[i]);
            if c.nb[i].len() <= 4 && group(numbers[i]) > 3 {
                c.imetal[i] = 0;
            }
        }
        // bond list: every neighbor pair once, in order of the first atom that lists it
        blist.clear();
        {
            let mut seen: std::collections::HashSet<(usize, usize)> = std::collections::HashSet::new();
            for i in 0..n {
                for &l in &c.nb[i] {
                    if seen.insert((l, i)) {
                        seen.insert((i, l));
                        blist.push((l, i));
                    }
                }
            }
        }
        // pi atom candidates
        piadr_list.clear();
        piadr2 = vec![0; n];
        for i in 0..n {
            let mut piat = (c.hyb[i] == 1 || c.hyb[i] == 2) && pilist(numbers[i]);
            let mut kk = 0;
            for &jj in &c.nb[i] {
                if numbers[i] == 8 && numbers[jj] == 16 && c.hyb[jj] == 5 {
                    piat = false;
                    continue;
                }
                if c.hyb[jj] == 1 || c.hyb[jj] == 2 {
                    kk += 1;
                }
            }
            let picon = kk > 0 && nofs(numbers[i]);
            if numbers[i] == 7 && c.nb[i].len() > 3 {
                continue;
            }
            if numbers[i] == 16 && c.hyb[i] == 5 {
                continue;
            }
            if picon || piat {
                piadr_list.push(i);
                piadr2[i] = piadr_list.len();
            }
        }
        let npiall = piadr_list.len();
        let mut nbpi: Nb = vec![Vec::new(); npiall];
        for i in 0..n {
            if piadr2[i] == 0 {
                continue;
            }
            let ii = piadr2[i] - 1;
            for &k in &c.nb[i] {
                if piadr2[k] > 0 {
                    nbpi[ii].push(piadr2[k] - 1);
                }
            }
        }
        let (pc, pv) = mrec(&nbpi);
        picount = if npiall == 0 { 0 } else { pc };
        pimvec = pv;

        // xi corrections of the EEQ
        dxi = vec![0.0; n];
        for i in 0..n {
            let ati = numbers[i];
            let nn = c.nb[i].len();
            if nn == 0 {
                continue;
            }
            let ip = piadr2[i];
            let ji = jth_nb(&c.nb[i], positions, i, 1, c.conv.tie_norms).unwrap();
            let mut nh = 0;
            let mut nm = 0;
            for &j in &c.nb[i] {
                if numbers[j] == 1 {
                    nh += 1;
                }
                if c.imetal[j] != 0 {
                    nm += 1;
                }
            }
            if ati == 5 {
                dxi[i] += nh as f64 * 0.015;
            }
            if ati == 6 && nn == 2 && c.itag[i] == 1 {
                dxi[i] = -0.15;
            }
            if ati == 6 && nn == 1 && numbers[ji] == 8 && c.nb[ji].len() == 1 {
                dxi[ji] = 0.15;
            }
            if ati == 8 && nn == 1 && ip != 0 && numbers[ji] == 7 && piadr2[ji] != 0 {
                dxi[i] = 0.05;
            }
            if ati == 8 && nn == 2 && nh == 2 {
                dxi[i] = -0.02;
            }
            if group(ati) == 6 && nn > 2 {
                dxi[i] += nn as f64 * 0.005;
            }
            if ati == 8 || ati == 16 {
                dxi[i] -= nh as f64 * 0.005;
            }
            if group(ati) == 7 && ati > 9 && nn > 1 {
                if nm == 0 {
                    dxi[i] -= nn as f64 * 0.021;
                } else {
                    dxi[i] += nn as f64 * 0.05;
                }
            }
        }
        // EEQ parameters for the topology charges (CN = number of neighbors)
        for i in 0..n {
            let ati = numbers[i];
            let dum = (c.nb[i].len() as f64).min(c.gen.cnmax);
            chieeq[i] = -data::CHI[ati - 1] + dxi[i] + data::CNF[ati - 1] * dum.sqrt();
            gameeq[i] = data::GAM[ati - 1];
            if c.imetal[i] == 2 {
                chieeq[i] -= c.gen.mchishift;
            }
            alpeeq[i] = data::ALP[ati - 1].powi(2);
        }
        if qloop_count == 1 {
            bpair = nbondmat(&c.nb, c.conv.unpaired_neighbors);
        }
        // topological distances (Floyd-Warshall in single precision, as xtb)
        let rabd = topo_distances(&c);
        // fragments
        if c.nfrag <= 1 {
            let (nf, fl) = mrec(&c.nbf);
            c.nfrag = nf;
            c.fraglist = fl;
        }
        // topology charges
        c.qa = goedeckera(&c, &rabd, &chieeq, &gameeq, &alpeeq, &c.qfrag)?;
        // charge of the pi systems
        if picount > 0 && qloop_count > 0 {
            ipis = vec![0; picount];
            let qtmp = c.qa.clone();
            let mut qah = c.qa.clone();
            qheavy(numbers, &c.nb, &mut qah);
            let mut cache: HashMap<usize, Vec<f64>> = HashMap::new();
            for pis in 1..=picount {
                let k0 = (0..npiall).find(|&k| pimvec[k] == pis).unwrap();
                let ifrag = c.fraglist[piadr_list[k0]];
                let dum2 = c.qfrag[ifrag - 1];
                let qneutral = if dum2 == 0.0 {
                    // identical system, identical charges
                    qah.clone()
                } else {
                    if let std::collections::hash_map::Entry::Vacant(slot) = cache.entry(ifrag) {
                        let mut qf = c.qfrag.clone();
                        qf[ifrag - 1] = 0.0;
                        let mut q = goedeckera(&c, &rabd, &chieeq, &gameeq, &alpeeq, &qf)?;
                        qheavy(numbers, &c.nb, &mut q);
                        slot.insert(q);
                    }
                    cache[&ifrag].clone()
                };
                let mut dum = 0.0;
                for k in 0..npiall {
                    if pimvec[k] == pis {
                        let a = piadr_list[k];
                        dum += qah[a] - qneutral[a];
                    }
                }
                dum *= 1.1;
                ipis[pis - 1] = dum.round() as i64;
            }
            c.qa = qtmp;
        }
        qloop_count += 1;
    }
    if let Some(fc) = &opts.fragment_charges {
        if fc.len() != c.nfrag {
            let s = |k: usize| if k == 1 { "" } else { "s" };
            return Err(Error::Input(format!(
                "{} fragment charge{} for {} fragment{} (the bonded molecules of the structure, in the order of \
                 their first atoms)",
                fc.len(),
                s(fc.len()),
                c.nfrag,
                s(c.nfrag)
            )));
        }
    }

    // charge-dependent EEQ hardness and the final atomic EEQ parameters
    let pi_buggy = |a: usize| a < piadr_list.len(); // xtb indexes the candidate list by atom here
    let pi2 = |a: usize| piadr2[a] != 0;
    let mut dgam = vec![0.0; n];
    for i in 0..n {
        let z = numbers[i];
        let mut ff = 0.0;
        if z == 1 {
            ff = -0.08;
        }
        if z == 5 {
            ff = -0.05;
        }
        if z == 6 {
            ff = -0.27;
            if c.hyb[i] < 3 {
                ff = -0.45;
            }
            if c.hyb[i] < 2 {
                ff = -0.34;
            }
        }
        if z == 7 {
            ff = -0.13;
            if pi_buggy(i) {
                ff = -0.14;
            }
            if amide(numbers, &c.hyb, &c.nb, &pi_buggy, i) {
                ff = -0.16;
            }
        }
        if z == 8 {
            ff = -0.15;
            if c.hyb[i] < 3 {
                ff = -0.08;
            }
        }
        if z == 9 {
            ff = 0.10;
        }
        if z > 10 {
            ff = -0.02;
        }
        if z == 17 {
            ff = -0.02;
        }
        if z == 35 {
            ff = -0.11;
        }
        if z == 53 {
            ff = -0.07;
        }
        if c.imetal[i] == 1 {
            ff = -0.08;
        }
        if c.imetal[i] == 2 {
            ff = -0.9;
        }
        if group(z) == 8 {
            ff = 0.0;
        }
        dgam[i] = c.qa[i] * ff;
    }
    for i in 0..n {
        let z = numbers[i];
        chieeq[i] = -data::CHI[z - 1] + dxi[i];
        gameeq[i] = data::GAM[z - 1] + dgam[i];
        if amide_h(numbers, &c.hyb, &c.nb, &pi2, i) {
            chieeq[i] -= 0.02;
        }
        let mut ff = 0.0;
        if z == 6 {
            ff = 0.09;
        }
        if z == 7 {
            ff = -0.21;
        }
        if group(z) == 6 {
            ff = -0.03;
        }
        if group(z) == 7 {
            ff = 0.50;
        }
        if c.imetal[i] == 1 {
            ff = 0.3;
        }
        if c.imetal[i] == 2 {
            ff = -0.1;
        }
        alpeeq[i] = (data::ALP[z - 1] + ff * c.qa[i]).powi(2);
    }

    // rings (no-metal neighbor list)
    let rings: Vec<AtomRings> = (0..n).map(|i| getring36(&c.nbm, &c.tag, i)).collect();

    // bonded ATM triples
    let mut batm = Vec::new();
    for i in 0..n {
        for j in bpair.with_value(i, 3) {
            if j >= i {
                continue;
            }
            for &k in &c.nb[j] {
                if i == k || j == k {
                    continue;
                }
                batm.push([i, j, k]);
            }
            for &k in &c.nb[i] {
                if i == k || j == k {
                    continue;
                }
                batm.push([i, j, k]);
            }
        }
    }
    if batm.len() > 1000 * n {
        return Err(Error::Topology("overflow in the bonded ATM list".into()));
    }

    // non-bonded repulsion exponents and dispersion charge scaling
    let mut rep_f = vec![0.0; n];
    let mut zeta = vec![0.0; n];
    for i in 0..n {
        let z = numbers[i];
        let fn_ = 1.0 + c.gen.nrepscal / (1.0 + (c.nb[i].len() as f64).powi(2));
        rep_f[i] = data::REPAN[z - 1] * (1.0 + c.qa[i] * c.gen.qrepscal) * fn_;
        zeta[i] = param::zeta(z, c.qa[i]);
    }

    // HB basicity
    let mut hbbas = vec![1.0; n];
    for i in 0..n {
        let nn = c.nb[i].len();
        let z = numbers[i];
        hbbas[i] = c.param.xhbas[z - 1];
        if z == 6 && nn == 2 && c.itag[i] == 1 {
            hbbas[i] = 1.46;
        }
        if !(z == 8 && nn == 1) {
            continue;
        }
        let nbz = numbers[c.nb[i][0]];
        if nbz == 6 {
            hbbas[i] = 0.68;
        }
        if nbz == 7 {
            hbbas[i] = 0.47;
        }
    }
    // HB acidity
    let mut hbaci: Vec<f64> = (0..n).map(|i| c.param.xhaci[numbers[i] - 1]).collect();
    for i in 0..n {
        let Some(nn) = jth_nb(&c.nb[i], positions, i, 1, c.conv.tie_norms) else { continue };
        hbaci[i] = c.param.xhaci[numbers[i] - 1];
        if amide_h(numbers, &c.hyb, &c.nb, &pi2, i) {
            hbaci[nn] *= 0.80;
        }
    }
    // H atoms of hydrogen bonds
    let mut hb_h = Vec::new();
    for i in 0..n {
        if numbers[i] != 1 || c.hyb[i] == 1 {
            continue;
        }
        let mut ff = c.gen.hqabthr;
        let Some(j) = jth_nb(&c.nb[i], positions, i, 1, c.conv.tie_norms) else { continue };
        if numbers[j] > 10 {
            ff -= 0.20;
        }
        if numbers[j] == 6 && c.hyb[j] == 3 {
            ff += 0.05;
        }
        if c.qa[i] > ff {
            hb_h.push(i);
        }
    }
    // A/B atoms of hydrogen bonds (per-atom part of hbatABl)
    let hb_ab: Vec<bool> = (0..n)
        .map(|i| {
            if numbers[i] == 6 && piadr2[i] == 0 {
                return false;
            }
            let mut ff = c.gen.qabthr;
            if numbers[i] > 10 {
                ff += 0.2;
            }
            c.qa[i] <= ff
        })
        .collect();
    // A-X bonds of XBs
    let mut xb_ax = Vec::new();
    for i in 0..n {
        for &ix in &c.nb[i] {
            if xatom(numbers[ix]) {
                if numbers[ix] == 16 && c.nb[ix].len() > 2 {
                    continue;
                }
                xb_ax.push((i, ix));
            }
        }
    }

    // Hueckel for the pi systems
    let mut pibo_bond = vec![0.0; blist.len()];
    let mut pbo: HashMap<(usize, usize), f64> = HashMap::new();
    let mut piflag = vec![false; n];
    if picount > 0 {
        huckel(&c, &blist, &piadr_list, &pimvec, picount, &ipis, &mut pibo_bond, &mut pbo, &mut piflag)?;
    }
    let pi = |a: usize| piflag[a];

    // sp2 -> sp3 for non-pi three-coordinated group-4 atoms that are not planar
    for i in 0..n {
        if c.hyb[i] == 2 && !piflag[i] && c.nb[i].len() == 3 && group(numbers[i]) == 4 {
            let (jj, kk, ll) = (c.nb[i][0], c.nb[i][1], c.nb[i][2]);
            let phi = geom::omega(positions[i], positions[jj], positions[kk], positions[ll]);
            if phi.abs() * 180.0 / PI > 40.0 {
                c.hyb[i] = 3;
            }
        }
    }
    let amide_flags: Vec<bool> = (0..n).map(|i| amide(numbers, &c.hyb, &c.nb, &pi, i)).collect();

    // bonds
    let mut bonds = Vec::with_capacity(blist.len());
    for (bi, &(jj, ii)) in blist.iter().enumerate() {
        bonds.push(bond_params(&c, &rings, &pi, jj, ii, pibo_bond[bi]));
    }

    // bonds coupled to hydrogen bonds
    let bond_hb = bond_hb_setup(&c, &bpair, &hb_h, &hb_ab, &hbbas, &hbaci, thr.hb1, &mut bonds);

    // angles
    let angles = angle_setup(&c, &rings, &pi, &pbo);

    // torsions and out-of-plane terms
    let torsions = torsion_setup(&c, &rings, &pi, &pbo, &bonds)?;

    // triple-bond torsions
    let stors = stors_setup(&c);

    // B atoms of XBs (the B condition of xtb's `xbatABl`)
    let pi_cand: Vec<bool> = piadr2.iter().map(|&v| v != 0).collect();
    let xb_b = (0..n)
        .map(|j| {
            let z = numbers[j];
            c.param.xhbas[z - 1] >= 1.0e-6 && !(group(z) == 4 && (!pi_cand[j] || c.qa[j] > 0.05))
        })
        .collect();

    Ok(Topology {
        numbers: numbers.to_vec(),
        charge,
        accuracy,
        conventions: opts.conventions,
        nb: c.nb.clone(),
        hyb: c.hyb.clone(),
        itag: c.itag.clone(),
        imetal: c.imetal.clone(),
        pi: piflag,
        pi_cand,
        amide: amide_flags,
        bpair,
        bonds,
        angles,
        torsions,
        batm,
        stors,
        nfrag: c.nfrag,
        fraglist: c.fraglist.clone(),
        qfrag: c.qfrag[..c.nfrag.max(1)].to_vec(),
        qa: c.qa.clone(),
        chieeq,
        gameeq,
        alpeeq,
        rep_f,
        zeta,
        hbbas,
        hbaci,
        hb_h,
        hb_ab,
        xb_ax,
        xb_b,
        bond_hb,
        cn_setup: lcn.cn,
        mchar: c.mchar.clone(),
    })
}

/// `gfnff_neigh`: neighbor lists (full, without highly coordinated atoms, without metals),
/// cluster tags and hybridization.
fn gfnff_neigh(c: &mut Ctx, nb_call: bool) -> Result<()> {
    let n = c.n;
    let at = c.at;
    let xyz = c.xyz;
    let fq = c.gen.rqshrink;
    let f_in = c.gen.rthr;
    let f2_in = c.gen.rthr2;
    // estimated bond lengths with normal CN, shrunk/expanded with the charges
    let rtmp = |i: usize, j: usize| -> f64 {
        // i is the outer (larger) index of the Fortran loop
        let (i, j) = if i >= j { (i, j) } else { (j, i) };
        let (ai, aj) = (at[i], at[j]);
        // gfnffrab: ra of the outer atom i, rb of j
        let ra = data::RAB_R0[ai - 1] + data::RAB_CNFAK[ai - 1] * normcn(ai) as f64;
        let rb = data::RAB_R0[aj - 1] + data::RAB_CNFAK[aj - 1] * normcn(aj) as f64;
        let mut v = (ra + rb) * param::rab_ff(ai, aj) * param::rab_scale(ai, aj);
        let mut f1 = fq;
        if metal(ai) > 0 {
            f1 *= 2.0;
        }
        let mut f2 = fq;
        if metal(aj) > 0 {
            f2 *= 2.0;
        }
        v = v - c.qa[i] * f1 - c.qa[j] * f2;
        v * fat(ai) * fat(aj)
    };
    // largest possible bond distance for the cell search
    let qmax = c.qa.iter().fold(0.0f64, |m, &q| m.max(q.abs()));
    let mut zs: Vec<usize> = at.to_vec();
    zs.sort_unstable();
    zs.dedup();
    let mut rmax: f64 = 0.0;
    for &a in &zs {
        for &b in &zs {
            let ra = data::RAB_R0[a - 1] + data::RAB_CNFAK[a - 1].abs() * normcn(a) as f64;
            let rb = data::RAB_R0[b - 1] + data::RAB_CNFAK[b - 1].abs() * normcn(b) as f64;
            let v =
                ((ra + rb) * param::rab_ff(a, b).abs() * param::rab_scale(a, b) + 4.0 * fq * qmax) * fat(a) * fat(b);
            rmax = rmax.max(v);
        }
    }
    let cutoff = rmax * f_in * (f2_in + 0.025) * (f2_in + 0.025) * 1.0001 + 1.0e-6;

    // fillnb for the three cases
    let mut cand = Vec::new();
    let mut pairs: Vec<Vec<(usize, f64, f64)>> = vec![Vec::new(); n]; // (j, dist, rco)
    for i in 0..n {
        c.cells.candidates(i, cutoff, |j| j != i, &mut cand);
        for &j in &cand {
            let d = [xyz[i][0] - (xyz[j][0] + 0.0), xyz[i][1] - (xyz[j][1] + 0.0), xyz[i][2] - (xyz[j][2] + 0.0)];
            let dist = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
            if dist <= 0.0 || dist > cutoff {
                continue;
            }
            pairs[i].push((j, dist, rtmp(i, j)));
        }
    }
    let fill = |icase: i32, nbf: &Nb, mchar: &[f64]| -> Nb {
        let mut out: Nb = vec![Vec::new(); n];
        for i in 0..n {
            for &(j, dist, rco) in &pairs[i] {
                let mut fm = 1.0;
                if icase == 1 {
                    if metal(at[i]) == 2 {
                        fm *= f2_in;
                    }
                    if metal(at[j]) == 2 {
                        fm *= f2_in;
                    }
                    if metal(at[i]) == 1 {
                        fm *= f2_in + 0.025;
                    }
                    if metal(at[j]) == 1 {
                        fm *= f2_in + 0.025;
                    }
                }
                if icase == 2 {
                    let nnfi = nbf[i].len() as i32;
                    let nnfj = nbf[j].len() as i32;
                    let hc_i = if group(at[i]) <= 2 { 4 } else { 6 };
                    if nnfi > hc_i {
                        continue;
                    }
                    let hc_j = if group(at[j]) <= 2 { 4 } else { 6 };
                    if nnfj > hc_j {
                        continue;
                    }
                }
                if icase == 3 {
                    let nnfi = nbf[i].len() as i32;
                    let nnfj = nbf[j].len() as i32;
                    if mchar[i] > 0.25 || metal(at[i]) > 0 {
                        continue;
                    }
                    if mchar[j] > 0.25 || metal(at[j]) > 0 {
                        continue;
                    }
                    if nnfi > normcn(at[i]) && at[i] > 10 {
                        continue;
                    }
                    if nnfj > normcn(at[j]) && at[j] > 10 {
                        continue;
                    }
                }
                if dist < fm * f_in * rco {
                    out[i].push(j);
                }
            }
        }
        out
    };
    let empty: Nb = vec![Vec::new(); n];
    c.nbf = fill(1, &empty, &c.mchar);
    c.nb = fill(2, &c.nbf, &c.mchar);
    c.nbm = fill(3, &c.nbf, &c.mchar);
    for i in 0..n {
        if c.nbf[i].len() > 40 {
            return Err(Error::Topology(format!("atom {} has more than 40 neighbors", i + 1)));
        }
    }

    c.itag = vec![0; n];
    c.tag = vec![false; n];
    for i in 0..n {
        if at[i] < 11 && c.nbf[i].len() > 2 {
            for &kk in &c.nbf[i] {
                if metal(at[kk]) != 0 || c.nb[kk].len() > 4 {
                    c.tag[i] = true;
                }
            }
        }
    }

    // hybridization
    let mut nbdum: Nb = vec![Vec::new(); n];
    for i in 0..n {
        let ati = at[i];
        let numnbf = c.nbf[i].len();
        let numnbm = c.nbm[i].len();
        let mut etacoord = false;
        if ati <= 10 {
            if ati == 6 && numnbf >= 4 && numnbm == 3 {
                etacoord = true;
            }
            if ati == 6 && numnbf == 3 && numnbm == 2 {
                etacoord = true;
            }
            let mut nm = 0;
            let mut im = usize::MAX;
            for &kk in &c.nbf[i] {
                if metal(at[kk]) != 0 {
                    nm += 1;
                    im = kk;
                }
            }
            if nm == 0 {
                etacoord = false;
            } else if nm == 1 {
                let mut ncm = 0;
                for &kk in &c.nbf[i] {
                    if kk != im {
                        for &l in &c.nbf[kk] {
                            if l == im {
                                ncm += 1;
                            }
                        }
                    }
                }
                if ncm == 0 {
                    etacoord = false;
                }
            }
        }
        if etacoord {
            c.itag[i] = -1;
            nbdum[i] = c.nbm[i].clone();
        } else {
            nbdum[i] = c.nbf[i].clone();
        }
    }
    let lintr = c.gen.linthr;
    for i in 0..n {
        let ati = at[i];
        let mut hyb = 0;
        let numnbm = c.nbm[i].len() as i32;
        let numnbf = c.nbf[i].len() as i32;
        let numnb = c.nb[i].len() as i32;
        let nbdiff = numnbf - numnb;
        let nbmdiff = numnbf - numnbm;
        let nb20i = nbdum[i].len() as i32;
        let mut nh = 0;
        for &j in &nbdum[i] {
            if at[j] == 1 {
                nh += 1;
            }
        }
        let g = group(ati);
        if g == 1 {
            if nb20i == 2 {
                hyb = 1;
            }
            if nb20i > 2 {
                hyb = 3;
            }
            if nb20i > 4 {
                hyb = 0;
            }
        }
        if g == 2 {
            if nb20i == 2 {
                hyb = 1;
            }
            if nb20i > 2 {
                hyb = 3;
            }
            if nb20i > 4 {
                hyb = 0;
            }
        }
        if g == 3 {
            if nb20i > 4 {
                hyb = 3;
            }
            if nb20i > 4 && ati > 10 && nbdiff == 0 {
                hyb = 5;
            }
            if nb20i == 4 {
                hyb = 3;
            }
            if nb20i == 3 {
                hyb = 2;
            }
            if nb20i == 2 {
                hyb = 1;
            }
        }
        if g == 4 {
            if nb20i >= 4 {
                hyb = 3;
            }
            if nb20i > 4 && ati > 10 && nbdiff == 0 {
                hyb = 5;
            }
            if nb20i == 3 {
                hyb = 2;
            }
            if nb20i == 2 {
                let (a, b) = (nbdum[i][0], nbdum[i][1]);
                let phi = geom::bangl(xyz[a], xyz[i], xyz[b]);
                if phi * 180.0 / PI < 150.0 {
                    hyb = 2;
                    c.itag[i] = 1;
                } else {
                    hyb = 1;
                }
                if c.qa[i] < -0.4 {
                    hyb = 2;
                    c.itag[i] = 0;
                }
            }
            if nb20i == 1 {
                hyb = 1;
            }
        }
        if g == 5 {
            if nb20i >= 4 {
                hyb = 3;
            }
            if nb20i > 4 && ati > 10 && nbdiff == 0 {
                hyb = 5;
            }
            if nb20i == 3 {
                hyb = 3;
            }
            if nb20i == 3 && ati == 7 {
                let (mut kk, mut ll, mut nn) = (0, 0, 0);
                for &jj in nbdum[i].iter().take(3) {
                    if at[jj] == 8 && c.nb[jj].len() == 1 {
                        kk += 1;
                    }
                    if at[jj] == 5 && c.nb[jj].len() == 4 {
                        ll += 1;
                    }
                    if at[jj] == 16 && c.nb[jj].len() == 4 {
                        nn += 1;
                    }
                }
                if nn == 1 && ll == 0 && kk == 0 {
                    hyb = 3;
                }
                if ll == 1 && nn == 0 {
                    hyb = 2;
                }
                if kk >= 1 {
                    hyb = 2;
                    c.itag[i] = 1;
                }
                if nbmdiff > 0 && nn == 0 {
                    hyb = 2;
                }
            }
            if nb20i == 2 {
                hyb = 2;
                let (jj, kk) = (nbdum[i][0], nbdum[i][1]);
                let phi = geom::bangl(xyz[jj], xyz[i], xyz[kk]);
                if nbdum[jj].len() == 1 && at[jj] == 6 {
                    hyb = 1;
                }
                if nbdum[kk].len() == 1 && at[kk] == 6 {
                    hyb = 1;
                }
                if nbdum[jj].len() == 1 && at[jj] == 7 {
                    hyb = 1;
                }
                if nbdum[kk].len() == 1 && at[kk] == 7 {
                    hyb = 1;
                }
                if metal(at[jj]) > 0 {
                    hyb = 1;
                }
                if metal(at[kk]) > 0 {
                    hyb = 1;
                }
                if at[jj] == 7 && at[kk] == 7 && nbdum[jj].len() <= 2 && nbdum[kk].len() <= 2 {
                    hyb = 1;
                }
                if phi * 180.0 / PI > lintr {
                    hyb = 1;
                }
            }
            if nb20i == 1 {
                hyb = 1;
            }
        }
        if g == 6 {
            if nb20i >= 3 {
                hyb = 3;
            }
            if nb20i > 3 && ati > 10 && nbdiff == 0 {
                hyb = 5;
            }
            if nb20i == 2 {
                hyb = 3;
            }
            if nb20i == 2 && nbmdiff > 0 {
                // CN of the closest non-metal neighbor (neigh%nb list)
                let mut rmin = 1.0e42;
                let mut jmin = usize::MAX;
                for &jj in &c.nb[i] {
                    if metal(at[jj]) != 0 {
                        continue;
                    }
                    let dist = c.conv.tie_norms.nearest([
                        xyz[i][0] - xyz[jj][0],
                        xyz[i][1] - xyz[jj][1],
                        xyz[i][2] - xyz[jj][2],
                    ]);
                    if dist < rmin {
                        rmin = dist;
                        jmin = jj;
                    }
                }
                let j = if jmin != usize::MAX { c.nb[jmin].len() } else { 0 };
                if j == 3 {
                    hyb = 2;
                }
                if j == 4 {
                    hyb = 3;
                }
            }
            if nb20i == 1 {
                hyb = 2;
            }
            if nb20i == 1 && nbdiff == 0 && !c.nb[i].is_empty() {
                let k = c.nb[i][0];
                if c.nb[k].len() == 1 {
                    hyb = 1;
                }
            }
        }
        if g == 7 {
            if nb20i == 2 {
                hyb = 1;
            }
            if nb20i > 2 && ati > 10 {
                hyb = 5;
            }
        }
        if g == 8 {
            hyb = 0;
            if nb20i > 0 && ati > 2 {
                hyb = 5;
            }
        }
        if g <= 0 {
            let mut nni = nb20i;
            if nh != 0 && nh != nni {
                nni -= nh;
            }
            if nni <= 2 {
                hyb = 1;
            }
            if nni <= 2 && g <= -6 {
                hyb = 2;
            }
            if nni == 3 {
                hyb = 2;
            }
            if nni == 4 && g > -7 {
                hyb = 3;
            }
            if nni == 4 && g <= -7 {
                hyb = 3;
            }
            if nni == 5 && g == -3 {
                hyb = 3;
            }
        }
        c.hyb[i] = hyb;
    }
    c.nb = nbdum;

    let mut nhigh = 0;
    for i in 0..n {
        if c.nb[i].len() > 12 {
            nhigh += 1;
        }
        for k in 0..c.nb[i].len() {
            let kk = c.nb[i][k];
            if at[kk] == 6 && at[i] == 6 && c.itag[i] == 1 && c.itag[kk] == 1 {
                c.itag[i] = 0;
                c.itag[kk] = 0;
            }
        }
    }
    if nhigh as f64 / n as f64 > 0.3 && nb_call {
        return Err(Error::Topology("too many atoms with extreme high CN".into()));
    }
    Ok(())
}

/// Remove H neighbors of lanthanides/actinides with a topology charge above -0.0281.
fn adjust_nb_lnh_anh(c: &mut Ctx) {
    const QTHR: f64 = -0.0281;
    for i in 0..c.n {
        let z = c.at[i];
        if !((57..=71).contains(&z) || (89..=103).contains(&z)) {
            continue;
        }
        let list = c.nb[i].clone();
        let mut keep = Vec::new();
        for inb in list {
            if c.at[inb] == 1 && c.qa[inb] > QTHR {
                c.nb[inb].retain(|&x| x != i);
            } else {
                keep.push(inb);
            }
        }
        c.nb[i] = keep;
    }
}

/// `nbondmat_pbc` for molecules: number of bonds (1-3, else 5) between atoms over the
/// neighbor list with unpaired bonds removed (and restored as 1 afterwards).
fn nbondmat(nb: &Nb, unpaired: UnpairedNeighbors) -> BondDistance {
    let n = nb.len();
    // reduced list without the unpaired neighbors (xtb 6.8.0); xtb 6.7.1 removed them by
    // shifting the original list, which drops the wrong entry when an atom has several of
    // them; reproduced literally
    let mut nbr: Nb = nb.clone();
    let mut tmpp: Vec<(usize, usize)> = Vec::new(); // (k, i)
    for i in 0..n {
        let mut cnt = nb[i].len();
        let mut arr: Vec<usize> = nb[i].clone();
        let mut inb = 0; // position of neighbor j in the reduced list (6.8.0)
        for j in 0..nb[i].len() {
            let k = nb[i][j];
            let hasnb = nb[k].contains(&i);
            if !hasnb {
                match unpaired {
                    UnpairedNeighbors::ShiftedOriginal => {
                        // nbr(l,i) = nb(l+1,i) for l = j.. (from the original list)
                        for l in j..arr.len() {
                            arr[l] = if l + 1 < nb[i].len() { nb[i][l + 1] } else { 0usize.wrapping_sub(1) };
                        }
                    }
                    UnpairedNeighbors::Removed => {
                        arr.remove(inb);
                    }
                }
                cnt -= 1;
                tmpp.push((k, i));
            } else {
                inb += 1;
            }
        }
        nbr[i] = arr.into_iter().take(cnt).collect();
    }
    let mut bp = BondDistance::new(n);
    for i in 0..n {
        let mut dist: HashMap<usize, u8> = HashMap::new();
        for &j in &nbr[i] {
            dist.insert(j, 1);
        }
        for cval in 1..=2u8 {
            let mut front: Vec<usize> = dist.iter().filter(|(_, &v)| v == cval).map(|(&k, _)| k).collect();
            front.sort_unstable();
            for inew in front {
                for &j in &nbr[inew] {
                    if dist.contains_key(&j) {
                        continue;
                    }
                    if j == i {
                        continue;
                    }
                    dist.insert(j, cval + 1);
                }
            }
        }
        for (j, v) in dist {
            if j != i {
                bp.set(j, i, v);
            }
        }
    }
    for (k, i) in tmpp {
        bp.set(k, i, 1);
        bp.set(i, k, 1);
    }
    bp
}

/// Topological distances (Angstrom, f32) between atoms connected within 12 Angstrom of
/// summed covalent radii: xtb's single-precision Floyd-Warshall restricted to the entries
/// that can matter (entries above `tdist_thr` never act as intermediates and end up as the
/// 13 Angstrom default, so they are not stored). Rows are sorted by column.
fn topo_distances(c: &Ctx) -> Vec<Vec<(u32, f32)>> {
    let n = c.n;
    let thr = c.gen.tdist_thr;
    let mut r: Vec<Vec<(u32, f32)>> = vec![Vec::new(); n];
    {
        let mut m: Vec<HashMap<u32, f32>> = vec![HashMap::new(); n];
        for i in 0..n {
            m[i].insert(i as u32, 0.0);
            for &j in &c.nb[i] {
                let v = (rad(c.at[i]) + rad(c.at[j])) as f32;
                m[j].insert(i as u32, v);
                m[i].insert(j as u32, v);
            }
        }
        for i in 0..n {
            let mut row: Vec<(u32, f32)> =
                m[i].iter().filter(|(_, &v)| v <= RABD_CUTOFF).map(|(&a, &v)| (a, v)).collect();
            row.sort_unstable_by_key(|e| e.0);
            r[i] = row;
        }
    }
    let mut merged: Vec<(u32, f32)> = Vec::new();
    for k in 0..n {
        let rowk: Vec<(u32, f32)> = r[k].iter().copied().filter(|e| e.1 <= thr).collect();
        for &(i, rik) in &rowk {
            // candidates rik + r(k,j), sorted by j; merge into row i keeping minima
            let old = std::mem::take(&mut r[i as usize]);
            merged.clear();
            let (mut a, mut b) = (0, 0);
            while a < old.len() || b < rowk.len() {
                let take_old = b >= rowk.len() || (a < old.len() && old[a].0 < rowk[b].0);
                let take_new = a >= old.len() || (b < rowk.len() && rowk[b].0 < old[a].0);
                if take_old {
                    merged.push(old[a]);
                    a += 1;
                } else if take_new {
                    let s = rik + rowk[b].1;
                    if s <= thr {
                        merged.push((rowk[b].0, s));
                    }
                    b += 1;
                } else {
                    let s = rik + rowk[b].1;
                    let v = if old[a].1 > s { s } else { old[a].1 };
                    merged.push((old[a].0, v));
                    a += 1;
                    b += 1;
                }
            }
            r[i as usize] = merged.clone();
        }
    }
    for row in r.iter_mut() {
        row.retain(|e| e.1 <= thr);
    }
    r
}

/// Topology EEQ (`goedeckera`): charges from the topological distances, with one charge
/// constraint per fragment.
fn goedeckera(
    c: &Ctx,
    rabd: &[Vec<(u32, f32)>],
    chieeq: &[f64],
    gameeq: &[f64],
    alpeeq: &[f64],
    qfrag: &[f64],
) -> Result<Vec<f64>> {
    let n = c.n;
    let nf = c.nfrag;
    let m = n + nf;
    let far = c.gen.rfgoed1 * (RABD_CUTOFF as f64) / param::AUTOAA;
    let mut a = Mat::<f64>::zeros(m, m);
    let mut x = vec![0.0; m];
    for i in 0..n {
        x[i] = chieeq[i];
        a[(i, i)] = gameeq[i] + TSQRT2PI / alpeeq[i].sqrt();
    }
    for i in 0..n {
        let row = &rabd[i];
        let mut p = 0;
        for j in 0..i {
            while p < row.len() && (row[p].0 as usize) < j {
                p += 1;
            }
            let rij = if p < row.len() && row[p].0 as usize == j {
                c.gen.rfgoed1 * (row[p].1 as f64) / param::AUTOAA
            } else {
                far
            };
            let gammij = 1.0 / (alpeeq[i] + alpeeq[j]).sqrt();
            let tmp = erf(gammij * rij) / rij;
            a[(j, i)] = tmp;
            a[(i, j)] = tmp;
        }
    }
    for f in 0..nf {
        x[n + f] = qfrag[f];
        for j in 0..n {
            if c.fraglist[j] == f + 1 {
                a[(n + f, j)] = 1.0;
                a[(j, n + f)] = 1.0;
            }
        }
    }
    let lu = linalg::Lu::new(a, c.par)?;
    let sol = lu.solve_vec(&x);
    let mut q: Vec<f64> = sol[..n].to_vec();
    if n == 1 {
        q[0] = qfrag[0];
    }
    Ok(q)
}

/// Iterative Hueckel calculation of every pi system: pi bond orders.
fn huckel(
    c: &Ctx,
    blist: &[(usize, usize)],
    piadr: &[usize],
    pimvec: &[usize],
    picount: usize,
    ipis: &[i64],
    pibo: &mut [f64],
    pbo: &mut HashMap<(usize, usize), f64>,
    piflag: &mut [bool],
) -> Result<()> {
    let n = c.n;
    let at = c.at;
    let g = &c.gen;
    let mut itmp = vec![false; n];
    for pis in 1..=picount {
        let mut npi = 0;
        let mut nelpi: i64 = 0;
        let mut piadr3: Vec<usize> = Vec::new();
        let mut piadr4 = vec![0usize; n];
        let mut piel = vec![0i64; n];
        for k in 0..piadr.len() {
            if pimvec[k] != pis {
                continue;
            }
            npi += 1;
            let a = piadr[k];
            let ati = at[a];
            let hybi = c.hyb[a];
            let ii = nelpi;
            if ati == 5 && hybi == 1 {
                nelpi += 1;
            }
            if ati == 6 && c.itag[a] != 1 {
                nelpi += 1;
            }
            if ati == 7 && hybi == 2 && c.itag[a] == 1 {
                nelpi += 1;
            }
            if ati == 7 && hybi <= 2 {
                nelpi += 1;
            }
            if ati == 7 && hybi == 3 {
                nelpi += 2;
            }
            if ati == 8 && hybi == 1 {
                nelpi += 1;
            }
            if ati == 8 && hybi == 2 {
                nelpi += 1;
            }
            if ati == 8 && hybi == 3 {
                nelpi += 2;
            }
            if ati == 9 && hybi != 1 {
                nelpi += 2;
            }
            if ati == 9 && hybi == 1 {
                nelpi += 3;
            }
            if ati == 16 && hybi == 1 {
                nelpi += 1;
            }
            if ati == 16 && hybi == 2 {
                nelpi += 1;
            }
            if ati == 16 && hybi == 3 {
                nelpi += 2;
            }
            if ati == 17 && hybi == 0 {
                nelpi += 2;
            }
            if ati == 17 && hybi == 1 {
                nelpi += 3;
            }
            piadr3.push(a);
            piadr4[a] = npi;
            piel[a] = nelpi - ii;
            if piel[a] > 2 {
                piel[a] = 2;
            }
        }
        let ipi = ipis.get(pis - 1).copied().unwrap_or(0);
        nelpi -= ipi;
        if npi < 2 || nelpi < 1 {
            continue;
        }
        let mut eold = 0.0;
        let mut pold = Mat::<f64>::from_fn(npi, npi, |_, _| 2.0e0 / 3.0e0);
        let mut api = Mat::<f64>::zeros(npi, npi);
        let mut apisave = Mat::<f64>::zeros(npi, npi);
        let mut pisip = 0.0;
        let maxit = g.maxhiter.round() as usize;
        for _ in 0..maxit {
            api.fill(0.0);
            for i in 0..npi {
                let ii = piadr3[i];
                api[(i, i)] = g.hdiag[at[ii]] + c.qa[ii] * g.hueckelp3 - (piel[ii] - 1) as f64 * g.pilpf;
            }
            for &(jj, ii) in blist {
                let ia = piadr4[ii];
                let ja = piadr4[jj];
                if ia > 0 && ja > 0 {
                    let (ia, ja) = (ia - 1, ja - 1);
                    let d = [c.xyz[ii][0] - c.xyz[jj][0], c.xyz[ii][1] - c.xyz[jj][1], c.xyz[ii][2] - c.xyz[jj][2]];
                    let mut dum = 1.0e-9 * (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                    dum = (g.hoffdiag[at[ii]] * g.hoffdiag[at[jj]]).sqrt() - dum;
                    let mut dum2 = g.hiter;
                    if c.hyb[ii] == 1 {
                        dum2 *= g.htriple;
                    }
                    if c.hyb[jj] == 1 {
                        dum2 *= g.htriple;
                    }
                    let v = -(dum * (1.0 - dum2 * (2.0e0 / 3.0e0 - pold[(ja, ia)])));
                    api[(ja, ia)] = v;
                    api[(ia, ja)] = v;
                }
            }
            apisave.copy_from(&api);
            let (p, eel, occ, eps) = qmsolve(&api, nelpi as usize, 4000.0, c.conv.fermi_smearing, c.par)?;
            for i in 0..npi {
                if occ[i] > 0.5 {
                    pisip = eps[i];
                }
            }
            api = p;
            if (eel - eold).abs() < 1.0e-4 {
                break;
            }
            pold.copy_from(&api);
            eold = eel;
        }
        if pisip > 0.40 {
            nelpi -= 1;
            let (p, _, _, _) = qmsolve(&apisave, nelpi.max(0) as usize, 4000.0, c.conv.fermi_smearing, c.par)?;
            api = p;
        }
        for (bi, &(jj, ii)) in blist.iter().enumerate() {
            let ja = piadr4[jj];
            let ia = piadr4[ii];
            if ia > 0 && ja > 0 {
                let v = api[(ja - 1, ia - 1)];
                pibo[bi] = v;
                pbo.insert((ii.max(jj), ii.min(jj)), v);
                itmp[ii] = true;
                itmp[jj] = true;
            }
        }
    }
    piflag.copy_from_slice(&itmp);
    Ok(())
}

/// Density matrix, electronic energy, occupations and orbital energies of `qmsolve`.
type QmSolution = (Mat<f64>, f64, Vec<f64>, Vec<f64>);

/// `gfnffqmsolve` without overlap: diagonalize, Fermi-smeared occupations at `et` K
/// (eigenvalues scaled by 0.1 x 27.2113957), density matrix.
fn qmsolve(a: &Mat<f64>, nel: usize, et: f64, fermi: FermiSmearing, par: Par) -> Result<QmSolution> {
    let ndim = a.nrows();
    let (mut e, u) = linalg::symmetric_eigen(a.as_ref(), par)?;
    for v in e.iter_mut() {
        *v = *v * 0.1 * 27.2113957;
    }
    let mut focc = vec![0.0; ndim];
    if et > 1.0e-3 {
        let (ihomoa, ihomob, mut focca, mut foccb) = occu(ndim, nel);
        if ihomoa <= ndim && ihomoa > 0 {
            fermismear(ndim, ihomoa, et, &e, &mut focca, fermi);
        } else {
            focca = vec![0.0; ndim];
        }
        if ihomob <= ndim && ihomob > 0 {
            fermismear(ndim, ihomob, et, &e, &mut foccb, fermi);
        } else {
            foccb = vec![0.0; ndim];
        }
        for i in 0..ndim {
            focc[i] = focca[i] + foccb[i];
        }
        if ihomoa < ndim && ihomoa >= 1 && (focc[ihomoa - 1] - focc[ihomoa]).abs() < 1.0e-4 {
            // perfect biradical: integer occupations
            focc = vec![0.0; ndim];
            for v in focc.iter_mut().take(nel / 2) {
                *v = 2.0;
            }
        }
    } else {
        for v in focc.iter_mut().take(nel / 2) {
            *v = 2.0;
        }
        if 2 * (nel / 2) != nel && nel / 2 < ndim {
            focc[nel / 2] = 1.0;
        }
    }
    let mut eel = 0.0;
    for i in 0..ndim {
        eel += focc[i] * e[i];
    }
    if focc.iter().all(|&f| f == focc[0]) {
        // all orbitals equally occupied (e.g. a filled pi "system" of lone pairs):
        // P = f I exactly; the rounding noise of C f C^T would otherwise decide the sign
        // of zero bond orders (xtb's LAPACK gives exact zeros here)
        let p = Mat::<f64>::from_fn(ndim, ndim, |r, c| if r == c { focc[0] } else { 0.0 });
        return Ok((p, eel, focc, e));
    }
    // P = C diag(focc) C^T
    let mut cw = Mat::<f64>::zeros(ndim, ndim);
    for m in 0..ndim {
        for i in 0..ndim {
            cw[(i, m)] = u[(i, m)] * focc[m];
        }
    }
    let mut p = Mat::<f64>::zeros(ndim, ndim);
    faer::linalg::matmul::matmul(p.as_mut(), faer::Accum::Replace, cw.as_ref(), u.transpose(), 1.0, Par::Seq);
    Ok((p, eel, focc, e))
}

/// `occu` with nopen = 0: (ihomoa, ihomob, alpha and beta occupations).
fn occu(ndim: usize, nel: usize) -> (usize, usize, Vec<f64>, Vec<f64>) {
    // xtb fills focc(1:nel); only the first ndim entries are read
    let mut focc = vec![0i64; ndim];
    if nel.is_multiple_of(2) {
        let ihomo = nel / 2;
        for v in focc.iter_mut().take(ihomo) {
            *v = 2;
        }
    } else {
        let na = nel / 2 + 1;
        let nb = nel / 2;
        for v in focc.iter_mut().take(na) {
            *v += 1;
        }
        for v in focc.iter_mut().take(nb) {
            *v += 1;
        }
    }
    let mut focca = vec![0.0; ndim];
    let mut foccb = vec![0.0; ndim];
    for i in 0..ndim {
        if focc[i] == 2 {
            focca[i] = 1.0;
            foccb[i] = 1.0;
        }
        if focc[i] == 1 {
            focca[i] = 1.0;
        }
    }
    let mut ihomoa = 0;
    let mut ihomob = 0;
    for i in 0..ndim {
        if focca[i] > 0.99 {
            ihomoa = i + 1;
        }
        if foccb[i] > 0.99 {
            ihomob = i + 1;
        }
    }
    (ihomoa, ihomob, focca, foccb)
}

/// `fermismear` of xtb (occupations only; `nel` ≥ 1), of the version `fermi`.
fn fermismear(norbs: usize, nel: usize, t: f64, eig: &[f64], occ: &mut [f64], fermi: FermiSmearing) {
    if fermi == FermiSmearing::Xtb680 {
        return fermismear_680(norbs, nel, t, eig, occ);
    }
    let boltz = param::KB * param::AUTOEV;
    let bkt = boltz * t;
    let thr = 1e-9;
    if nel + 1 > norbs {
        // occupations stay as given
        return;
    }
    let mut e_fermi = 0.5 * (eig[nel - 1] + eig[nel]);
    let occt = nel as f64;
    for _ in 0..200 {
        let mut total_number = 0.0;
        let mut total_dfermi = 0.0;
        for i in 0..norbs {
            let mut fermifunct = 0.0;
            let dfermifunct;
            if (eig[i] - e_fermi) / bkt < 50.0 {
                fermifunct = 1.0 / (((eig[i] - e_fermi) / bkt).exp() + 1.0);
                dfermifunct =
                    ((eig[i] - e_fermi) / bkt).exp() / (bkt * (((eig[i] - e_fermi) / bkt).exp() + 1.0).powi(2));
            } else {
                dfermifunct = 0.0;
            }
            occ[i] = fermifunct;
            total_number += fermifunct;
            total_dfermi += dfermifunct;
        }
        let change_fermi = (occt - total_number) / total_dfermi;
        e_fermi += change_fermi;
        if (occt - total_number).abs() <= thr {
            break;
        }
    }
}

/// xtb 6.8.0's `fermismear` (occupations only; `nel` ≥ 1): aufbau occupations to start with,
/// the Newton search stops at min(√ε, 1e5 ε) electrons or a vanishing derivative.
fn fermismear_680(norbs: usize, nel: usize, t: f64, eig: &[f64], occ: &mut [f64]) {
    let boltz = param::KB * param::AUTOEV;
    let bkt = boltz * t;
    let thr = f64::EPSILON.sqrt().min(1.0e5 * f64::EPSILON);
    let sqrttiny = f64::MIN_POSITIVE.sqrt();
    for (i, o) in occ.iter_mut().enumerate().take(norbs) {
        *o = if i < nel { 1.0 } else { 0.0 };
    }
    if nel + 1 > norbs {
        return;
    }
    let mut e_fermi = 0.5 * (eig[nel - 1] + eig[nel]);
    let occt = nel as f64;
    for _ in 0..200 {
        if t <= 0.0 {
            break;
        }
        let mut total_number = 0.0;
        let mut total_dfermi = 0.0;
        for i in 0..norbs {
            let arg = (eig[i] - e_fermi) / bkt;
            let (fermifunct, dfermifunct) = if arg > 50.0 {
                (0.0, 0.0)
            } else if arg < -50.0 {
                (1.0, 0.0)
            } else {
                let f = 1.0 / (arg.exp() + 1.0);
                (f, f * (1.0 - f) / bkt)
            };
            occ[i] = fermifunct;
            total_number += fermifunct;
            total_dfermi += dfermifunct;
        }
        if (occt - total_number).abs() <= thr {
            break;
        }
        if total_dfermi > sqrttiny {
            e_fermi += (occt - total_number) / total_dfermi;
        } else {
            break;
        }
    }
}

/// Parameters of one bond (the bond loop of `gfnff_ini`, with its local bond type `bbtyp`).
fn bond_params(c: &Ctx, rings: &[AtomRings], pi: &dyn Fn(usize) -> bool, jj: usize, ii: usize, pibo: f64) -> Bond {
    let g = &c.gen;
    let at = c.at;
    let hyb = &c.hyb;
    let nni = c.nb[ii].len();
    let nnj = c.nb[jj].len();
    let ia = at[ii];
    let ja = at[jj];
    let ringsb = ringsbond(rings, ii, jj);
    let mut shift = 0.0;
    let mut fxh = 1.0;
    let mut ringf = 1.0;
    let fqq;
    let mut fpi = 1.0;
    let mut fheavy = 1.0;
    let mut fcn = 1.0;
    let mut fsrb2 = g.srb2;
    let mut bridge = false;
    let mut btyp = 1;
    if hyb[ii] == 2 && hyb[jj] == 2 {
        btyp = 2;
    }
    if hyb[ii] == 3 && hyb[jj] == 2 && ia == 7 {
        btyp = 2;
    }
    if hyb[jj] == 3 && hyb[ii] == 2 && ja == 7 {
        btyp = 2;
    }
    if hyb[ii] == 1 || hyb[jj] == 1 {
        btyp = 3;
    }
    if (group(ia) == 7 || ia == 1) && hyb[ii] == 1 {
        btyp = 3;
        bridge = true;
    }
    if (group(ja) == 7 || ja == 1) && hyb[jj] == 1 {
        btyp = 3;
        bridge = true;
    }
    if hyb[ii] == 5 || hyb[jj] == 5 {
        btyp = 4;
    }
    if c.imetal[ii] > 0 || c.imetal[jj] > 0 {
        btyp = 5;
    }
    if c.imetal[ii] == 2 && c.imetal[jj] == 2 {
        btyp = 7;
    }
    if c.imetal[jj] == 2 && c.itag[ii] == -1 && pi(ii) {
        btyp = 6;
    }
    if c.imetal[ii] == 2 && c.itag[jj] == -1 && pi(jj) {
        btyp = 6;
    }
    let mut bbtyp = btyp;
    let bstrength;
    if bbtyp < 5 {
        let hybi = hyb[ii].max(hyb[jj]);
        let hybj = hyb[ii].min(hyb[jj]);
        let mut bs = if hybi == 5 || hybj == 5 { g.bstren[4] } else { g.bsmat[hybi as usize][hybj as usize] };
        if hybi == 3 && hybj == 2 && (ia == 7 || ja == 7) {
            bs = g.bstren[2] * 1.04;
        }
        if bridge {
            if group(ia) == 7 {
                bs = g.bstren[1] * 0.50e0;
            }
            if group(ja) == 7 {
                bs = g.bstren[1] * 0.50e0;
            }
            if ia == 1 || ia == 9 {
                bs = g.bstren[1] * 0.30e0;
            }
            if ja == 1 || ja == 9 {
                bs = g.bstren[1] * 0.30e0;
            }
        }
        if bbtyp == 4 {
            shift = g.hyper_shift;
        }
        if ia == 1 || ja == 1 {
            shift = g.rabshifth;
        }
        if ia == 9 && ja == 9 {
            shift = 0.22;
        }
        if hyb[ii] == 3 && hyb[jj] == 0 {
            shift -= 0.022;
        }
        if hyb[ii] == 0 && hyb[jj] == 3 {
            shift -= 0.022;
        }
        if hyb[ii] == 1 && hyb[jj] == 0 {
            shift += 0.14;
        }
        if hyb[ii] == 0 && hyb[jj] == 1 {
            shift += 0.14;
        }
        if ia == 1 && ja == 6 {
            if ringsatom(rings, jj) == 3 {
                fxh = 1.05;
            }
            if ctype(at, &c.nb, pi, jj) == 1 {
                fxh = 0.95;
            }
        }
        if ia == 6 && ja == 1 {
            if ringsatom(rings, ii) == 3 {
                fxh = 1.05;
            }
            if ctype(at, &c.nb, pi, ii) == 1 {
                fxh = 0.95;
            }
        }
        if ia == 1 && ja == 5 {
            fxh = 1.10;
        }
        if ja == 1 && ia == 5 {
            fxh = 1.10;
        }
        if ia == 1 && ja == 7 {
            fxh = 1.06;
        }
        if ja == 1 && ia == 7 {
            fxh = 1.06;
        }
        if ia == 1 && ja == 8 {
            fxh = 0.93;
        }
        if ja == 1 && ia == 8 {
            fxh = 0.93;
        }
        if bbtyp == 3 && ia == 6 && ja == 8 {
            bs = g.bstren[3] * 0.90e0;
        }
        if bbtyp == 3 && ia == 8 && ja == 6 {
            bs = g.bstren[3] * 0.90e0;
        }
        if bbtyp == 3 && (hyb[ii] == 0 || hyb[jj] == 0) {
            bbtyp = 1;
        }
        if bbtyp == 3 && (hyb[ii] == 3 || hyb[jj] == 3) {
            bbtyp = 1;
        }
        if bbtyp == 3 && (hyb[ii] == 2 || hyb[jj] == 2) {
            bbtyp = 2;
        }
        if pibo > c.conv.pibo_threshold {
            shift = g.hueckelp * (g.bzref - pibo);
            if bbtyp != 3 && pibo > 0.1 {
                btyp = 2;
                // xtb also sets bbtyp = 2, which is only printed
            }
            fpi = 1.0 - g.hueckelp2 * (g.bzref2 - pibo);
        }
        if ia > 10 && ja > 10 {
            fcn /= 1.0 + 0.007 * (nni as f64).powi(2);
            fcn /= 1.0 + 0.007 * (nnj as f64).powi(2);
        }
        let qafac = c.qa[ii] * c.qa[jj] * 70.0e0;
        fqq = 1.0 + g.qfacbm0 / (1.0 + (15.0 * qafac).exp());
        bstrength = bs;
    } else {
        shift = 0.0;
        let mut bs = g.bstren[bbtyp as usize];
        if bbtyp == 7 {
            let (ri, rj) = (param::itabrow6(ia), param::itabrow6(ja));
            if ri > 4 && rj > 4 {
                bs = g.bstren[8];
            }
            if ri == 4 && rj > 4 {
                bs = g.bstren[9];
            }
            if rj == 4 && ri > 4 {
                bs = g.bstren[9];
            }
            let mut dum = 2.0e0 * c.mchar[ii] + 2.0e0 * c.mchar[jj];
            dum = dum.min(0.5e0);
            bs *= 1.0e0 - dum;
        }
        let mtyp = |a: usize, z: usize| -> usize {
            let mut m = 0;
            if group(z) == 1 {
                m = 1;
            }
            if group(z) == 2 {
                m = 2;
            }
            if group(z) > 2 && c.imetal[a] == 1 {
                m = 3;
            }
            if c.imetal[a] == 2 {
                m = 4;
            }
            m
        };
        let mtyp1 = mtyp(ii, ia);
        let mtyp2 = mtyp(jj, ja);
        let qafac = c.qa[ii] * c.qa[jj] * 25.0e0;
        let dum = 1.0 / (1.0 + (15.0 * qafac).exp());
        fqq = 1.0e0 + dum * (g.qfacbm[mtyp1] + g.qfacbm[mtyp2]) * 0.5;
        let (mi, mj) = (c.imetal[ii], c.imetal[jj]);
        if mi == 2 && ja > 10 {
            fheavy = 0.65e0;
        }
        if mj == 2 && ia > 10 {
            fheavy = 0.65e0;
        }
        if mi == 2 && ja == 15 {
            fheavy = 1.60e0;
        }
        if mj == 2 && ia == 15 {
            fheavy = 1.60e0;
        }
        if mi == 2 && group(ja) == 6 {
            fheavy = 0.85e0;
        }
        if mj == 2 && group(ia) == 6 {
            fheavy = 0.85e0;
        }
        if mi == 2 && group(ja) == 7 {
            fheavy = 1.30e0;
        }
        if mj == 2 && group(ia) == 7 {
            fheavy = 1.30e0;
        }
        if mi == 2 && ja == 1 && param::itabrow6(ia) <= 5 {
            fxh = 0.80e0;
        }
        if mj == 2 && ia == 1 && param::itabrow6(ja) <= 5 {
            fxh = 0.80e0;
        }
        if mi == 2 && ja == 1 && param::itabrow6(ia) > 5 {
            fxh = 1.00e0;
        }
        if mj == 2 && ia == 1 && param::itabrow6(ja) > 5 {
            fxh = 1.00e0;
        }
        if mi == 1 && ja == 1 {
            fxh = 1.20e0;
        }
        if mj == 1 && ia == 1 {
            fxh = 1.20e0;
        }
        if mj == 2 && hyb[ii] == 1 {
            if ia == 6 {
                fpi = 1.5e0;
                shift = -0.45e0;
            }
            if ia == 7 && nni != 1 {
                fpi = 0.4e0;
                shift = 0.47e0;
            }
        }
        if mi == 2 && hyb[jj] == 1 {
            if ja == 6 {
                fpi = 1.5e0;
                shift = -0.45e0;
            }
            if ja == 7 && nnj != 1 {
                fpi = 0.4e0;
                shift = 0.47e0;
            }
        }
        if mi == 2 {
            shift += g.metal2_shift;
        }
        if mj == 2 {
            shift += g.metal2_shift;
        }
        if mi == 1 && group(ia) <= 2 {
            shift += g.metal1_shift;
        }
        if mj == 1 && group(ja) <= 2 {
            shift += g.metal1_shift;
        }
        if mtyp1 == 3 {
            shift += g.metal3_shift;
        }
        if mtyp2 == 3 {
            shift += g.metal3_shift;
        }
        if bbtyp == 6 && metal(ia) == 2 {
            shift += g.eta_shift * nni as f64;
        }
        if bbtyp == 6 && metal(ja) == 2 {
            shift += g.eta_shift * nnj as f64;
        }
        if mtyp1 > 0 && mtyp1 < 3 {
            fcn /= 1.0e0 + 0.100 * (nni as f64).powi(2);
        }
        if mtyp2 > 0 && mtyp2 < 3 {
            fcn /= 1.0e0 + 0.100 * (nnj as f64).powi(2);
        }
        if mtyp1 == 3 {
            fcn /= 1.0e0 + 0.030 * (nni as f64).powi(2);
        }
        if mtyp2 == 3 {
            fcn /= 1.0e0 + 0.030 * (nnj as f64).powi(2);
        }
        if mtyp1 == 4 {
            fcn /= 1.0e0 + 0.036 * (nni as f64).powi(2);
        }
        if mtyp2 == 4 {
            fcn /= 1.0e0 + 0.036 * (nnj as f64).powi(2);
        }
        if mtyp1 == 4 || mtyp2 == 4 {
            fsrb2 = -g.srb2 * 0.22;
        } else {
            fsrb2 = g.srb2 * 0.28;
        }
        bstrength = bs;
    }
    if ia > 10 && ja > 10 {
        shift += g.hshift3;
        if ia > 18 {
            shift += g.hshift4;
        }
        if ja > 18 {
            shift += g.hshift4;
        }
        if ia > 36 {
            shift += g.hshift5;
        }
        if ja > 36 {
            shift += g.hshift5;
        }
    }
    let v1 = g.rabshift + shift;
    if ringsb > 0 {
        ringf = 1.0e0 + g.fringbo * (6.0e0 - ringsb as f64).powi(2);
    }
    let v2 = g.srb1 * (1.0e0 + fsrb2 * (en(ia) - en(ja)).powi(2) + g.srb3 * bstrength);
    let v3 = -(data::BOND[ia - 1] * data::BOND[ja - 1] * ringf * bstrength * fqq * fheavy * fpi * fxh * fcn);
    Bond { j: jj, i: ii, vbond: [v1, v2, v3], nr_hb: 0, pibo, btyp }
}

/// Hydrogen-bond partners of N-H and O-H bonds at the setup geometry (`bond_hbset`,
/// `bond_hb_AHB_set*`); sets `nr_hb` of the bonds.
fn bond_hb_setup(
    c: &Ctx,
    bpair: &BondDistance,
    hb_h: &[usize],
    hb_ab: &[bool],
    hbbas: &[f64],
    hbaci: &[f64],
    hbthr1: f64,
    bonds: &mut [Bond],
) -> Vec<BondHb> {
    let at = c.at;
    let xyz = c.xyz;
    // (A, B, H) triples in xtb's order: AB pairs (i > j), then H atoms
    let mut hbl: Vec<(usize, usize, usize)> = Vec::new();
    if !hb_h.is_empty() {
        // xtb loops over every HB hydrogen for every A/B pair; only those bonded to A or B
        // (bpair = 1) add a triple, so visit just those, in hb_h order: h_of[x] = positions k
        // in hb_h with bpair(x, hb_h[k]) = 1
        let mut h_of: Vec<Vec<usize>> = vec![Vec::new(); c.n];
        for (k, &nh) in hb_h.iter().enumerate() {
            for x in bpair.with_value(nh, 1) {
                h_of[x].push(k);
            }
        }
        let mut hs: Vec<usize> = Vec::new();
        let mut cand = Vec::new();
        for i in 0..c.n {
            if !hb_ab[i] {
                continue;
            }
            c.cells.candidates(i, hbthr1.sqrt() * (1.0 + 1e-12) + 1e-9, |j| j < i && hb_ab[j], &mut cand);
            for &j in &cand {
                if hbbas[i] * hbaci[j] < 1.0e-6 && hbaci[i] * hbbas[j] < 1.0e-6 {
                    continue;
                }
                let d = [xyz[i][0] - xyz[j][0], xyz[i][1] - xyz[j][1], xyz[i][2] - xyz[j][2]];
                let rab = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().powi(2);
                if rab > hbthr1 {
                    continue;
                }
                let ijnonbond = bpair.get(j, i) != 1;
                if !ijnonbond {
                    continue;
                }
                hs.clear();
                hs.extend_from_slice(&h_of[i]);
                hs.extend_from_slice(&h_of[j]);
                hs.sort_unstable();
                hs.dedup();
                for &k in &hs {
                    let nh = hb_h[k];
                    if bpair.get(i, nh) == 1 {
                        hbl.push((i, j, nh));
                    }
                    if bpair.get(j, nh) == 1 {
                        hbl.push((j, i, nh));
                    }
                }
            }
        }
    }
    // group by A-H bond in bond-list order
    let mut by_ah: HashMap<(usize, usize), Vec<usize>> = HashMap::new();
    for (k, &(a, _, h)) in hbl.iter().enumerate() {
        by_ah.entry((a, h)).or_default().push(k);
    }
    let mut out: Vec<BondHb> = Vec::new();
    let mut b_count = 1usize;
    for bond in bonds.iter_mut() {
        let (jj, ii) = (bond.j, bond.i);
        let (hb_h_at, hb_a) = if at[ii] == 1 {
            (ii, jj)
        } else if at[jj] == 1 {
            (jj, ii)
        } else {
            continue;
        };
        if !(at[hb_a] == 7 || at[hb_a] == 8) {
            continue;
        }
        let mut entry: Option<BondHb> = None;
        if let Some(ks) = by_ah.get(&(hb_a, hb_h_at)) {
            for &k in ks {
                let b = hbl[k].1;
                if at[b] == 7 || at[b] == 8 {
                    match entry.as_mut() {
                        Some(e) => {
                            e.b.push(b);
                            b_count += 1;
                        }
                        None => {
                            entry = Some(BondHb { a: hb_a, h: hb_h_at, b: vec![b] });
                            b_count = 1;
                        }
                    }
                }
            }
        }
        if let Some(e) = entry {
            out.push(e);
        }
        bond.nr_hb = b_count;
    }
    out
}

/// Bending terms (the angle loop of `gfnff_ini`).
fn angle_setup(
    c: &Ctx,
    rings: &[AtomRings],
    pi: &dyn Fn(usize) -> bool,
    pbo: &HashMap<(usize, usize), f64>,
) -> Vec<Angle> {
    let g = &c.gen;
    let at = c.at;
    let hyb = &c.hyb;
    let xyz = c.xyz;
    let pboget = |a: usize, b: usize| pbo.get(&(a.max(b), a.min(b))).copied().unwrap_or(0.0);
    let mut out = Vec::new();
    for i in 0..c.n {
        let nn = c.nb[i].len();
        if nn <= 1 || nn > 6 {
            continue;
        }
        let ii = i;
        let ati = at[i];
        for j in 0..nn {
            for k in 0..j {
                let jj = c.nb[i][j];
                let kk = c.nb[i][k];
                let atj = at[jj];
                let atk = at[kk];
                let fijk = data::ANGL[ati - 1] * data::ANGL2[atj - 1] * data::ANGL2[atk - 1];
                if fijk < g.fcthr {
                    continue;
                }
                let phi = geom::bangl(xyz[jj], xyz[i], xyz[kk]);
                if metal(ati) > 0 && phi * 180.0 / PI < 60.0 {
                    continue;
                }
                let mut feta = 1.0e0;
                if c.imetal[ii] == 2 && c.itag[jj] == -1 && pi(jj) {
                    feta = 0.3e0;
                }
                if c.imetal[ii] == 2 && c.itag[kk] == -1 && pi(kk) {
                    feta *= 0.3e0;
                }
                let cnt = |f: &dyn Fn(usize) -> bool| -> i32 { f(jj) as i32 + f(kk) as i32 };
                let nh = cnt(&|x| at[x] == 1);
                let no = cnt(&|x| at[x] == 8);
                let nsi = cnt(&|x| at[x] == 14);
                let nc = cnt(&|x| at[x] == 6);
                let nmet = cnt(&|x| metal(at[x]) != 0);
                let npi = cnt(&|x| pi(x));
                let ringsb = ringsbend(rings, ii, jj, kk);
                // xtb: (hyb(ii) == 1 .or. hyb(jj) == 1) .or. (hyb(ii) == 1 .or. hyb(kk) == 1)
                let mut triple = hyb[ii] == 1 || hyb[jj] == 1 || hyb[kk] == 1;
                let fqq = if c.imetal[ii] == 0 && c.imetal[jj] == 0 && c.imetal[kk] == 0 {
                    1.0e0 - (c.qa[ii] * c.qa[jj] + c.qa[ii] * c.qa[kk]) * g.qfac_ben
                } else {
                    1.0e0 - (c.qa[ii] * c.qa[jj] + c.qa[ii] * c.qa[kk]) * g.qfac_ben * 2.5
                };
                let mut f2 = 1.0e0;
                let deg = phi * 180.0 / PI;
                let mut r0 = 100.0;
                if hyb[i] == 1 {
                    r0 = 180.;
                }
                if hyb[i] == 2 {
                    r0 = 120.;
                }
                if hyb[i] == 3 {
                    r0 = 109.5;
                }
                if hyb[i] == 3 && at[i] > 10 {
                    if nn <= 3 {
                        r0 = g.aheavy3;
                    }
                    if nn >= 4 {
                        r0 = g.aheavy4;
                    }
                    if nn == 4 && group(ati) == 5 {
                        r0 = 109.5;
                    }
                    if nn == 4 && group(ati) == 4 && ati > 49 {
                        r0 = 109.5;
                    }
                    if group(ati) == 4 {
                        r0 -= nh as f64 * 5.;
                    }
                    if group(ati) == 5 {
                        r0 -= nh as f64 * 5.;
                    }
                    if group(ati) == 6 {
                        r0 -= nh as f64 * 5.;
                    }
                }
                if hyb[i] == 5 {
                    r0 = 90.;
                    f2 = 0.11;
                    if deg > g.linthr {
                        r0 = 180.;
                    }
                }
                if ati == 5 {
                    if hyb[i] == 3 {
                        r0 = 115.;
                    }
                    if hyb[i] == 2 {
                        r0 = 115.;
                    }
                }
                if ati == 6 {
                    if hyb[i] == 3 && nh == 2 {
                        r0 = 108.6;
                    }
                    if hyb[i] == 3 && no == 1 {
                        r0 = 108.5;
                    }
                    if hyb[i] == 2 && no == 2 {
                        r0 = 122.;
                    }
                    if hyb[i] == 2 && no == 1 {
                        f2 = 0.7;
                    }
                    if hyb[i] == 1 && no == 2 {
                        triple = false;
                        f2 = 2.0;
                    }
                    if hyb[i] == 3 && nn > 4 && deg > g.linthr {
                        r0 = 180.;
                    }
                }
                if ati == 8 && nn == 2 {
                    r0 = 104.5;
                    if nh == 2 {
                        r0 = 100.;
                        f2 = 1.20;
                    }
                    r0 += 7. * nsi as f64;
                    r0 += 14. * nmet as f64;
                    if npi == 2 {
                        r0 = 109.;
                    }
                    if nmet > 0 && deg > g.linthr {
                        r0 = 180.;
                        f2 = 0.3;
                    }
                }
                if ati == 7 && nn == 2 {
                    f2 = 1.4;
                    r0 = 115.;
                    if ringsb != 0 {
                        r0 = 105.;
                    }
                    if at[kk] == 8 || at[jj] == 8 {
                        r0 = 103.;
                    }
                    if at[kk] == 9 || at[jj] == 9 {
                        r0 = 102.;
                    }
                    if hyb[i] == 1 {
                        r0 = 180.;
                    }
                    if c.imetal[jj] == 2 && hyb[i] == 1 && at[kk] == 7 {
                        r0 = 135.;
                    }
                    if c.imetal[kk] == 2 && hyb[i] == 1 && at[jj] == 7 {
                        r0 = 135.;
                    }
                }
                if ati == 7 && hyb[i] == 3 {
                    if npi > 0 {
                        if amide(at, hyb, &c.nb, pi, i) {
                            r0 = 115.;
                            f2 = 1.2e0;
                        } else {
                            let sumppi = pboget(ii, jj) + pboget(ii, kk);
                            r0 = 113.;
                            f2 = 1.0e0 - sumppi * 0.7e0;
                        }
                    } else {
                        r0 = 104.;
                        f2 = 0.40;
                        f2 += nh as f64 * 0.19;
                        f2 += no as f64 * 0.25;
                        f2 += nc as f64 * 0.01;
                    }
                }
                if ringsb == 3 {
                    r0 = 82.;
                }
                if ringsb == 4 {
                    r0 = 96.;
                }
                if ringsb == 5 && ati == 6 {
                    r0 = 109.;
                }
                if ringsb == 0 {
                    let idum = ringsatom(rings, i);
                    if idum == 3 {
                        let ringsj = ringsatom(rings, jj);
                        let ringsk = ringsatom(rings, kk);
                        if ringsj + ringsk == 102 {
                            r0 += 4.0e0;
                        }
                    }
                }
                if triple {
                    f2 = 0.60e0;
                    if atj == 7 || atk == 7 {
                        f2 = 1.00e0;
                    }
                    if (c.imetal[jj] == 2 || c.imetal[kk] == 2) && deg > g.linthr {
                        if ati == 6 && atj == 6 {
                            f2 = 3.;
                        }
                        if ati == 6 && atk == 6 {
                            f2 = 3.;
                        }
                        if ati == 6 && atj == 7 {
                            f2 = 3.;
                        }
                        if ati == 6 && atk == 7 {
                            f2 = 3.;
                        }
                        if ati == 6 && group(atj) == 6 {
                            f2 = 14.;
                        }
                        if ati == 6 && group(atk) == 6 {
                            f2 = 14.;
                        }
                        if ati == 7 && atj == 7 {
                            f2 = 10.;
                        }
                        if ati == 7 && atj == 6 {
                            f2 = 10.;
                        }
                        if ati == 7 && atk == 6 {
                            f2 = 10.;
                        }
                        if ati == 7 && atj == 8 {
                            r0 = 180.;
                            f2 = 12.;
                        }
                        if ati == 7 && atk == 8 {
                            r0 = 180.;
                            f2 = 12.;
                        }
                    }
                }
                if group(ati) == 4 && nn == 2 && c.itag[i] == 1 {
                    if ati == 6 {
                        r0 = 145.;
                    }
                    if ati > 6 {
                        r0 = 90.;
                    }
                }
                if group(ati) == 6 && nn == 4 && no >= 1 {
                    r0 = 115.;
                }
                if group(ati) == 7 && hyb[i] == 1 {
                    if ati == 9 {
                        r0 = 90.;
                    }
                    if ati == 17 {
                        r0 = 90.;
                    }
                    if ati == 35 {
                        r0 = 90.;
                    }
                    if ati == 53 {
                        r0 = 90.;
                    }
                    if ati > 9 && deg > g.linthr {
                        r0 = 180.;
                    }
                    f2 = 0.6 / (ati as f64).powf(0.15);
                }
                if hyb[i] == 3 && group(ati) == 4 && ati > 32 && c.qa[i] > 0.4 {
                    if deg > 140. {
                        r0 = 180.;
                    }
                    if deg < 100. {
                        r0 = 90.;
                    }
                    f2 = 1.0;
                }
                if c.imetal[ii] > 0 {
                    if hyb[i] == 0 {
                        r0 = 90.;
                        f2 = 1.35;
                    }
                    if hyb[i] == 1 {
                        r0 = 180.;
                    }
                    if hyb[i] == 2 {
                        r0 = 120.;
                    }
                    if hyb[i] == 3 {
                        r0 = 109.5;
                    }
                    if deg > g.linthr {
                        r0 = 180.;
                    }
                }
                let fn_ = 1.0e0 - 2.36e0 / (nn as f64).powi(2);
                let phi0 = r0 * PI / 180.;
                let fbsmall = 1.0e0 - g.fbs1 * (-(0.64 * (phi0 - PI).powi(2))).exp();
                let fc = fijk * fqq * f2 * fn_ * fbsmall * feta;
                out.push(Angle { i: ii, j: jj, k: kk, phi0, fc });
            }
        }
    }
    out
}

/// The truncated pi of xtb's near-linear torsion test (`chktors`, `3.1415926`), kept so the
/// same torsions are set up.
#[allow(clippy::approx_constant)]
const PI_CHKTORS: f64 = 3.1415926e0;

/// Torsions and out-of-plane terms (the torsion loops of `gfnff_ini`).
fn torsion_setup(
    c: &Ctx,
    rings: &[AtomRings],
    pi: &dyn Fn(usize) -> bool,
    pbo: &HashMap<(usize, usize), f64>,
    bonds: &[Bond],
) -> Result<Vec<Torsion>> {
    let g = &c.gen;
    let at = c.at;
    let hyb = &c.hyb;
    let xyz = c.xyz;
    let pboget = |a: usize, b: usize| pbo.get(&(a.max(b), a.min(b))).copied().unwrap_or(0.0);
    let tors = |z: usize| data::TORS[z - 1];
    let tors2 = |z: usize| data::TORS2[z - 1];
    let mut out = Vec::new();
    for b in bonds {
        let jj = b.j;
        let ii = b.i;
        let nni = c.nb[ii].len();
        let nnj = c.nb[jj].len();
        let btyp = b.btyp;
        if btyp == 3 || btyp == 6 {
            continue;
        }
        let mut fij = tors(at[ii]) * tors(at[jj]);
        if fij < g.fcthr {
            continue;
        }
        if tors(at[ii]) < 0.0 || tors(at[jj]) < 0.0 {
            continue;
        }
        if metal(at[ii]) > 1 && nni > 4 {
            continue;
        }
        if metal(at[jj]) > 1 && nnj > 4 {
            continue;
        }
        let fqq = 1.0e0 + (c.qa[ii] * c.qa[jj]).abs() * g.qfac_tor;
        let ringsb = ringsbond(rings, ii, jj);
        let lring = ringsb > 0;
        let sp3ij = hyb[ii] == 3 && hyb[jj] == 3;
        let nhi = 1 + c.nb[ii].iter().filter(|&&x| at[x] == 1).count();
        let nhj = 1 + c.nb[jj].iter().filter(|&&x| at[x] == 1).count();
        fij *= (nhi as f64 * nhj as f64).powf(0.07);
        if alpha_co(at, hyb, &c.nb, pi, ii, jj) {
            fij *= 1.3e0;
        }
        if amide(at, hyb, &c.nb, pi, ii) && hyb[jj] == 3 && at[jj] == 6 {
            fij *= 1.3e0;
        }
        if amide(at, hyb, &c.nb, pi, jj) && hyb[ii] == 3 && at[ii] == 6 {
            fij *= 1.3e0;
        }
        if btyp == 4 {
            fij *= 0.2e0;
        }
        for &kk in &c.nb[ii] {
            if kk == jj {
                continue;
            }
            for &ll in &c.nb[jj] {
                if ll == ii {
                    continue;
                }
                if ll == kk {
                    continue;
                }
                // chktors: skip near-linear arrangements
                let phi1 = geom::bangl(xyz[jj], xyz[ii], xyz[kk]);
                if phi1 * 180. / PI_CHKTORS > 170.0e0 {
                    continue;
                }
                let phi2 = geom::bangl(xyz[ii], xyz[jj], xyz[ll]);
                if phi2 * 180. / PI_CHKTORS > 170.0e0 {
                    continue;
                }
                let mut fkl = tors2(at[kk]) * tors2(at[ll]);
                if at[kk] == 7 && !pi(kk) {
                    fkl = tors2(at[kk]) * tors2(at[ll]) * 0.5;
                }
                if at[ll] == 7 && !pi(ll) {
                    fkl = tors2(at[kk]) * tors2(at[ll]) * 0.5;
                }
                if fkl < g.fcthr {
                    continue;
                }
                if tors(at[kk]) < 0.0 || tors(at[ll]) < 0.0 {
                    continue;
                }
                let mut f1 = g.torsf[1];
                let mut f2 = 0.0e0;
                fkl *= (c.nb[kk].len() as f64 * c.nb[ll].len() as f64).powf(-0.14);
                let mut nrot;
                let mut phi;
                if lring {
                    let rings4 = if ringsb > 3 { ringstors(rings, ii, jj, kk, ll) } else { 3 };
                    nrot = 1;
                    if btyp == 2 {
                        nrot = 2;
                    }
                    phi = 0.0;
                    if btyp == 1 && rings4 > 0 {
                        let ringl = ringstorl(rings, ii, jj, kk, ll);
                        let notpicon = !pi(kk) && !pi(ll);
                        if rings4 == 3 && notpicon {
                            nrot = 1;
                            phi = 0.0e0;
                            f1 = g.fr3;
                        }
                        if rings4 == 4 && ringl == rings4 && notpicon {
                            nrot = 6;
                            phi = 30.0e0;
                            f1 = g.fr4;
                        }
                        if rings4 == 5 && ringl == rings4 && notpicon {
                            nrot = 6;
                            phi = 30.0e0;
                            f1 = g.fr5;
                        }
                        if rings4 == 6 && ringl == rings4 && notpicon {
                            nrot = 3;
                            phi = 60.0e0;
                            f1 = g.fr6;
                        }
                    }
                    if rings4 == 0 && btyp == 1 && c.nb[kk].len() == 1 && c.nb[ll].len() == 1 {
                        nrot = 6;
                        phi = 30.0e0;
                        f1 = 0.30;
                    }
                    if btyp == 2
                        && ringsb == 5
                        && at[ii] * at[jj] == 42
                        && (amide(at, hyb, &c.nb, pi, ii) || amide(at, hyb, &c.nb, pi, jj))
                    {
                        f1 = 5.;
                    }
                } else {
                    phi = 180.0e0;
                    nrot = 1;
                    if hyb[ii] == 3 && hyb[jj] == 3 {
                        nrot = 3;
                    }
                    if btyp == 2 {
                        nrot = 2;
                    }
                    if pi(ii) && (!pi(jj) && hyb[jj] == 3) {
                        f1 = 0.5e0;
                        if at[ii] == 7 {
                            f1 = 0.2e0;
                        }
                        phi = 180.0e0;
                        nrot = 3;
                    }
                    if pi(jj) && (!pi(ii) && hyb[ii] == 3) {
                        f1 = 0.5e0;
                        if at[jj] == 7 {
                            f1 = 0.2e0;
                        }
                        phi = 180.0e0;
                        nrot = 3;
                    }
                }
                if hyb[ii] == 3 && hyb[jj] == 3 {
                    let (gi, gj) = (group(at[ii]), group(at[jj]));
                    if gi == 5 && gj == 5 {
                        nrot = 3;
                        phi = 60.0e0;
                        f1 = 3.0e0;
                    }
                    if (gi == 5 && gj == 6) || (gi == 6 && gj == 5) {
                        nrot = 2;
                        phi = 90.0e0;
                        f1 = 1.0e0;
                        if at[ii] >= 15 && at[jj] >= 15 {
                            f1 = 20.0e0;
                        }
                    }
                    if gi == 6 && gj == 6 {
                        nrot = 2;
                        phi = 90.0e0;
                        f1 = 5.0e0;
                        if at[ii] >= 16 && at[jj] >= 16 {
                            f1 = 25.0e0;
                        }
                    }
                }
                if b.pibo > c.conv.pibo_threshold {
                    f2 = b.pibo * (-(2.5e0 * (1.24e0 - b.pibo).powi(14))).exp();
                    if !pi(kk) && at[kk] > 10 {
                        f2 *= 1.3;
                    }
                    if !pi(ll) && at[ll] > 10 {
                        f2 *= 1.3;
                    }
                    f1 *= 0.55;
                }
                if hyb[kk] == 5 || hyb[ll] == 5 {
                    fkl *= 1.5;
                }
                let fctot = (f1 + 10.0e0 * g.torsf[2] * f2) * fqq * fij * fkl;
                if fctot > g.fcthr {
                    out.push(Torsion { atoms: [ll, ii, jj, kk], nrot, phi0: phi * PI / 180.0e0, fc: fctot });
                }
                let sp3kl = hyb[kk] == 3 && hyb[ll] == 3;
                if sp3kl && sp3ij && !lring && btyp < 5 {
                    let mut ff = g.torsf[6];
                    if at[ii] == 7 || at[jj] == 7 {
                        ff = g.torsf[7];
                    }
                    if at[ii] == 8 || at[jj] == 8 {
                        ff = g.torsf[8];
                    }
                    out.push(Torsion { atoms: [ll, ii, jj, kk], nrot: 1, phi0: PI, fc: ff * fij * fkl * fqq });
                }
            }
        }
    }
    // out-of-plane terms
    for i in 0..c.n {
        if c.nb[i].len() != 3 {
            continue;
        }
        if !pi(i) && at[i] != 7 {
            continue;
        }
        let (jj, kk, ll) = (c.nb[i][0], c.nb[i][1], c.nb[i][2]);
        let dist = |a: usize| -> f64 {
            c.conv.tie_norms.out_of_plane([xyz[i][0] - xyz[a][0], xyz[i][1] - xyz[a][1], xyz[i][2] - xyz[a][2]])
        };
        let mut sdum3 = [dist(jj), dist(kk), dist(ll)];
        let mut ind3 = [jj, kk, ll];
        ssort(&mut sdum3, &mut ind3);
        let [jj, kk, ll] = ind3;
        if !pi(i) && at[i] == 7 {
            let r0 = 80.0e0;
            let ff = 0.60e0;
            let mut fc = 0.0e0;
            for &idum in &c.nb[i] {
                fc += ff * repz(at[idum]).sqrt();
            }
            out.push(Torsion { atoms: [i, jj, kk, ll], nrot: -1, phi0: r0 * PI / 180., fc });
        } else {
            let mut ncarbo = 0;
            let mut nf = 0;
            for &idum in &c.nb[i] {
                if at[idum] == 8 || at[idum] == 16 {
                    ncarbo += 1;
                }
                if group(at[idum]) == 7 {
                    nf += 1;
                }
            }
            let fqq = 1.0e0 + c.qa[i] * 5.0e0;
            let sumppi = pboget(i, jj) + pboget(i, kk) + pboget(i, ll);
            let f2 = 1.0e0 - sumppi * g.torsf[5];
            let mut fc = g.torsf[3] * f2 * fqq;
            if at[i] == 5 && ncarbo > 0 {
                fc *= 38.;
            }
            if at[i] == 6 && ncarbo > 0 {
                fc *= 38.;
            }
            if at[i] == 6 && nf > 0 && ncarbo == 0 {
                fc *= 10.;
            }
            if at[i] == 7 && ncarbo > 0 {
                fc = fc * 10. / f2;
            }
            out.push(Torsion { atoms: [i, jj, kk, ll], nrot: 0, phi0: 0.0e0, fc });
        }
    }
    Ok(out)
}

/// `specialTorsList`: torsions around C-C triple bonds between sp2 carbons (with xtb's
/// index quirks).
fn stors_setup(c: &Ctx) -> Vec<[usize; 6]> {
    let at = c.at;
    let xyz = c.xyz;
    let nb = &c.nb;
    // count first
    let mut nn = 0;
    for i in 0..c.n {
        if at[i] == 6 && nb[i].len() == 2 {
            for j in 1..=2 {
                if let Some(nbi) = jth_nb(&nb[i], xyz, i, j, c.conv.tie_norms) {
                    if at[nbi] == 6 && nb[nbi].len() == 2 {
                        nn += 1;
                    }
                }
            }
        }
    }
    nn /= 2;
    let mut out = Vec::new();
    if nn == 0 {
        return out;
    }
    // indices as in the Fortran (1-based, -1 = unset); jj, kk, ii, ll persist across atoms
    let mut ii: i64 = -1;
    let mut jj: i64 = -1;
    let mut kk: i64 = -1;
    let mut ll: i64 = -1;
    let nbcount = |a: usize| nb[a].len();
    'atoms: for i in 0..c.n {
        if !(at[i] == 6 && nbcount(i) == 2) {
            continue;
        }
        for j in 1..=2 {
            let Some(nbi) = jth_nb(&nb[i], xyz, i, j, c.conv.tie_norms) else { continue };
            if !(at[nbi] == 6 && nbcount(nbi) == 2) {
                continue;
            }
            let d = [xyz[i][0] - xyz[nbi][0], xyz[i][1] - xyz[nbi][1], xyz[i][2] - xyz[nbi][2]];
            if (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt() > 2.37 {
                continue;
            }
            for k in 1..=2 {
                if let Some(nbk) = jth_nb(&nb[i], xyz, i, k, c.conv.tie_norms) {
                    if nbk != nbi && at[nbk] == 6 {
                        jj = nbk as i64;
                    }
                }
            }
            for k in 1..=2 {
                if let Some(nbk) = jth_nb(&nb[nbi], xyz, nbi, k, c.conv.tie_norms) {
                    if nbk != i && at[nbk] == 6 {
                        kk = nbk as i64;
                    }
                }
            }
            if jj == -1 || kk == -1 {
                continue 'atoms;
            }
            let (ju, ku) = (jj as usize, kk as usize);
            if c.hyb[ju] == 2 && c.hyb[ku] == 2 && at[ju] == 6 && at[ku] == 6 {
                let mut iiok = false;
                let mut llok = false;
                // xtb tests hyb/at/nb of the loop counter k (1-based atom number k)
                for k in 1..=nbcount(ju) {
                    let nbk = jth_nb(&nb[ju], xyz, ju, k, c.conv.tie_norms);
                    let kat = k - 1;
                    if let Some(nbk) = nbk {
                        if c.hyb[kat] == 2 && at[kat] == 6 && nbcount(kat) == 3 && nbk != i {
                            ii = nbk as i64;
                            iiok = true;
                        }
                    }
                }
                for k in 1..=nbcount(ku) {
                    // xtb takes the neighbors of jj here as well
                    let nbk = jth_nb(&nb[ju], xyz, ju, k, c.conv.tie_norms);
                    let kat = k - 1;
                    if let Some(nbk) = nbk {
                        if c.hyb[kat] == 2 && at[kat] == 6 && nbcount(i) == 3 && nbk != nbi {
                            ll = nbk as i64;
                            llok = true;
                        }
                    }
                }
                if nbi > i && iiok && llok {
                    out.push([ii as usize, ju, i, nbi, ku, ll as usize]);
                }
            }
        }
    }
    // xtb allocates nn entries; unused ones stay zero
    out.truncate(nn);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nbondmat_unpaired_neighbors() {
        // atom 0 lists 1, 2, 3, 4 as neighbors, but 1 and 2 do not list 0 (one-sided bonds, as
        // of a metal); 3 and 4 do, and 5 hangs on 4
        let rest: Nb = vec![vec![], vec![], vec![0], vec![0, 5], vec![4]];
        let with = |first: Vec<usize>| -> Nb { std::iter::once(first).chain(rest.iter().cloned()).collect() };
        for order in [vec![1, 2, 3, 4], vec![3, 1, 4, 2], vec![1, 3, 2, 4]] {
            // xtb 6.8.0 removes both unpaired neighbors: 0-3 and 0-4 bonded, 0-5 two bonds apart
            let b = nbondmat(&with(order.clone()), UnpairedNeighbors::Removed);
            assert_eq!((b.get(3, 0), b.get(4, 0), b.get(5, 0)), (1, 1, 2), "{order:?}");
            // the unpaired bonds count as bonds (xtb's `tmpp`)
            assert_eq!((b.get(1, 0), b.get(2, 0)), (1, 1));
        }
        // xtb 6.7.1 shifts the original list down once per unpaired neighbor and keeps as many
        // entries as there are paired ones: [1,3,2,4] -> [3,2,4] -> [3,2,4] -> [3, 2], so atom 4
        // is lost and 5 is not reached from 0; [3,1,4,2] -> [3,4,2] -> [3, 4] happens to be right
        let b = nbondmat(&with(vec![1, 3, 2, 4]), UnpairedNeighbors::ShiftedOriginal);
        assert_eq!((b.get(3, 0), b.get(4, 0), b.get(5, 0)), (1, 5, 5));
        let b = nbondmat(&with(vec![3, 1, 4, 2]), UnpairedNeighbors::ShiftedOriginal);
        assert_eq!((b.get(3, 0), b.get(4, 0), b.get(5, 0)), (1, 1, 2));
    }

    #[test]
    fn fermismear_versions() {
        // both versions conserve the electron count, 6.8.0 to min(sqrt eps, 1e5 eps) = 2.2e-11
        // electrons, 6.7.1 to 1e-9
        let eig = [-2.0, -1.0, -0.2, 1.0, 2.0];
        for (fermi, thr) in [(FermiSmearing::Xtb671, 1e-9), (FermiSmearing::Xtb680, 2.3e-11)] {
            let mut occ = vec![1.0, 1.0, 0.0, 0.0, 0.0];
            fermismear(5, 2, 4000.0, &eig, &mut occ, fermi);
            let n: f64 = occ.iter().sum();
            assert!((n - 2.0).abs() <= thr, "{fermi:?}: {n}");
            assert!(occ.windows(2).all(|w| w[0] >= w[1]) && occ[2] > 0.0 && occ[2] < 0.5, "{occ:?}");
        }
        // no orbital above the electrons: aufbau occupations
        let mut occ = vec![0.0; 2];
        fermismear(2, 2, 4000.0, &[-1.0, -0.5], &mut occ, FermiSmearing::Xtb680);
        assert_eq!(occ, vec![1.0, 1.0]);
    }
}
