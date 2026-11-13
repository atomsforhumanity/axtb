// SPDX-License-Identifier: LGPL-3.0-or-later
//! Energy and analytic gradient of GFN-FF at one geometry with a fixed topology: a port
//! of xtb's `gfnff_eg` (`src/gfnff/gfnff_eg.f90`) for molecules.
//!
//! Pair loops use cell lists and xtb's cutoffs; dense N x N arrays of xtb (`dcn`, `sqrab`,
//! `eeqtmp`, `d3list`, `grab0`, `considered_ABH`) are replaced by pair lists, so memory
//! grows linearly with the number of atoms except for the EEQ matrix. Parallel loops run
//! over a fixed number of interleaved atom chunks whose results are added in chunk order,
//! so their sums do not depend on the number of threads (the EEQ solve does in its last
//! bits; see the crate documentation).

use crate::cells::CellList;
use crate::cn::LogCn;
use crate::data;
use crate::error::{Error, Result};
use crate::param::{self, Gen, Param, Thresholds, TSQRT2PI};
use crate::real::{erf, V3};
use crate::setup_util::jth_nb;
use crate::terms::{self, BondPar, HbPar};
use crate::topology::Topology;
use axtb_base::eeq::{ChargeEquations, ChargeSolver, Constraints};
use axtb_base::threads::Threads;
use axtb_d4::data::{MAX_REF, NFREQ};
use faer::Par;

/// Number of interleaved chunks of the parallel pair loops (fixed, so that their sums do not
/// depend on the thread count).
pub const NCHUNK: usize = 64;

/// Energy terms as xtb prints them (Eh).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Energies {
    pub bond: f64,
    pub angle: f64,
    pub torsion: f64,
    pub repulsion: f64,
    pub electrostatic: f64,
    pub dispersion: f64,
    pub hb: f64,
    pub xb: f64,
    pub batm: f64,
    pub external: f64,
    pub total: f64,
}

impl Energies {
    pub fn named(&self) -> Vec<(&'static str, f64)> {
        vec![
            ("bond", self.bond),
            ("angle", self.angle),
            ("torsion", self.torsion),
            ("repulsion", self.repulsion),
            ("electrostatic", self.electrostatic),
            ("dispersion", self.dispersion),
            ("hb", self.hb),
            ("xb", self.xb),
            ("batm", self.batm),
            ("external", self.external),
        ]
    }
}

/// D3-type dispersion model of GFN-FF (`newD3Model`): dftd4's reference polarizabilities
/// without charge scaling, integrated to C6 per reference pair. The reference data and the
/// Casimir-Polder integration are axtb-d4's (dftd4 4.3.1, the tables xtb carries in
/// `include/param_ref.fh`), with xtb's gap: no references for Fr-Rg (`d4_nref`).
#[derive(Clone, Debug)]
pub struct DispModel {
    /// per element (Z-1): reference CNs
    pub refcn: Vec<Vec<f64>>,
    /// c6[(zi, zj)] as nref_i x nref_j row-major, only for elements present
    pub c6: std::collections::HashMap<(usize, usize), Vec<f64>>,
}

/// Number of D4 reference systems of element `z` in xtb's GFN-FF: dftd4's, except none for
/// Fr-Rg (Z = 87-111), which xtb's tables leave empty (dftd4 4.3.1 has references there; of
/// these elements GFN-FF takes Fr-Lr, whose C6 is therefore zero).
fn d4_nref(z: usize) -> usize {
    if (87..=111).contains(&z) {
        0
    } else {
        axtb_d4::data::REFN[z - 1] as usize
    }
}

impl DispModel {
    pub fn new(numbers: &[usize]) -> Self {
        let mut zs: Vec<usize> = numbers.to_vec();
        zs.sort_unstable();
        zs.dedup();
        let nref: Vec<usize> = zs.iter().map(|&z| d4_nref(z)).collect();
        let alpha: Vec<[[f64; NFREQ]; MAX_REF]> = zs
            .iter()
            .zip(&nref)
            .map(|(&z, &n)| if n > 0 { axtb_d4::reference_alpha(z, |_, _| 1.0) } else { [[0.0; NFREQ]; MAX_REF] })
            .collect();
        let mut refcn = vec![Vec::new(); 118];
        for (&z, &n) in zs.iter().zip(&nref) {
            refcn[z - 1] = axtb_d4::data::REFCN[z - 1][..n].to_vec();
        }
        // the species ascend in Z, so each product is formed with the larger element first, as
        // in xtb
        let c6ref = axtb_d4::reference_c6_table(&nref, &alpha);
        let nsp = zs.len();
        let mut c6 = std::collections::HashMap::new();
        for (isp, &zi) in zs.iter().enumerate() {
            for (jsp, &zj) in zs.iter().enumerate() {
                let block = &c6ref[isp * nsp + jsp];
                let tab: Vec<f64> = (0..nref[isp]).flat_map(|ii| block[ii][..nref[jsp]].iter().copied()).collect();
                c6.insert((zi, zj), tab);
            }
        }
        DispModel { refcn, c6 }
    }

    /// Reference weights and their CN derivatives (`weight_references_d4`, wf = 4).
    pub fn weights(&self, z: usize, cn: f64) -> (Vec<f64>, Vec<f64>) {
        let wf = 4.0;
        let refcn = &self.refcn[z - 1];
        let wcn = |r: f64| {
            let val = -wf * (cn - r).powi(2);
            if val < -200.0 {
                0.0
            } else {
                val.exp()
            }
        };
        let mut norm = 0.0;
        let mut dnorm = 0.0;
        for &r in refcn {
            let gw = wcn(r);
            norm += gw;
            dnorm += 2.0 * wf * (r - cn) * gw;
        }
        norm = if norm > 1e-80 { 1.0 / norm } else { 0.0 };
        let maxcn = refcn.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut gw = Vec::with_capacity(refcn.len());
        let mut dgw = Vec::with_capacity(refcn.len());
        for &r in refcn {
            let expw = wcn(r);
            let expd = 2.0 * wf * (r - cn) * expw;
            let mut gwk = expw * norm;
            if norm == 0.0 {
                gwk = if (maxcn - r).abs() < 1e-12 { 1.0 } else { 0.0 };
            }
            gw.push(gwk);
            dgw.push(expd * norm - expw * dnorm * norm.powi(2));
        }
        (gw, dgw)
    }
}

/// Everything fixed for a topology besides the topology itself.
#[derive(Clone, Debug)]
pub struct Model {
    pub param: Param,
    pub hb: HbPar,
    pub thr: Thresholds,
    pub disp: DispModel,
    /// H atoms of `hb_h` with bpair(a, H) = 1 for every atom a
    pub bonded_h: Vec<Vec<usize>>,
    /// per-atom flags
    pub is_hb_h: Vec<bool>,
    /// bonds that use the HB-coupled stretch (index into topo.bonds) -> (A, H)
    pub bond_hb_ah: Vec<Option<(usize, usize)>>,
}

impl Model {
    pub fn new(topo: &Topology) -> Self {
        let param = Param::new(&Gen::default(), &topo.conventions.params());
        let thr = Thresholds::new(topo.accuracy);
        let hb = HbPar {
            hbacut: param.hbacut,
            hbscut: param.hbscut,
            hbalp: param.hbalp,
            hblongcut: param.hblongcut,
            hbabmix: param.hbabmix,
            hbnbcut: param.hbnbcut,
            xhaci_globabh: param.xhaci_globabh,
            xhaci_coh: param.xhaci_coh,
            tors_hb: param.tors_hb,
            bend_hb: param.bend_hb,
        };
        let n = topo.natoms();
        let mut is_hb_h = vec![false; n];
        for &h in &topo.hb_h {
            is_hb_h[h] = true;
        }
        // bpair(a, H) = 1: a and H list each other, or one lists the other (restored half bonds)
        let mut rev: Vec<Vec<usize>> = vec![Vec::new(); n];
        for a in 0..n {
            for &b in &topo.nb[a] {
                rev[b].push(a);
            }
        }
        let mut bonded_h = vec![Vec::new(); n];
        for &h in &topo.hb_h {
            let mut cands: Vec<usize> = topo.nb[h].clone();
            cands.extend_from_slice(&rev[h]);
            cands.sort_unstable();
            cands.dedup();
            for a in cands {
                if topo.bpair.get(a, h) == 1 {
                    bonded_h[a].push(h);
                }
            }
        }
        for l in bonded_h.iter_mut() {
            l.sort_unstable();
            l.dedup();
        }
        let mut bond_hb_ah = vec![None; topo.bonds.len()];
        for (k, b) in topo.bonds.iter().enumerate() {
            if b.nr_hb >= 1 {
                let (h, a) = if topo.numbers[b.i] == 1 { (b.i, b.j) } else { (b.j, b.i) };
                bond_hb_ah[k] = Some((a, h));
            }
        }
        Model { disp: DispModel::new(&topo.numbers), param, hb, thr, bonded_h, is_hb_h, bond_hb_ah }
    }
}

/// Hydrogen-bond and halogen-bond lists of one geometry (`gfnff_hbset`).
#[derive(Clone, Debug, Default)]
pub struct HbLists {
    /// (A, B, H) with H not bonded to A
    pub hb1: Vec<[usize; 3]>,
    /// (A, B, H) with H bonded to A
    pub hb2: Vec<[usize; 3]>,
    /// (A, B, X)
    pub xb: Vec<[usize; 3]>,
}

/// The H-bond triplets (A, B, H) of `gfnff_hbset` (and the halogen bonds) at `xyz`. Per
/// acceptor-capable atom i (in order) its partners j < i (ascending) and, per pair, the
/// candidate H in ascending order: the H near i, the H bonded to i and the H bonded to j. With
/// `threads` parallel the atoms i run in fixed chunks whose lists are joined in order: the
/// same lists in the same order on any thread count.
pub fn hb_lists(topo: &Topology, model: &Model, xyz: &[[f64; 3]], cells: &CellList, threads: &Threads) -> HbLists {
    let n = topo.natoms();
    let thr1 = model.thr.hb1;
    let thr2 = model.thr.hb2;
    let mut out = HbLists::default();
    let d2 = |a: usize, b: usize| {
        let d = [xyz[a][0] - xyz[b][0], xyz[a][1] - xyz[b][1], xyz[a][2] - xyz[b][2]];
        d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
    };
    if !topo.hb_h.is_empty() {
        // the triplets of the atoms i in `range`: (hb1, hb2)
        let atoms = |range: std::ops::Range<usize>| {
            let (mut hb1, mut hb2) = (Vec::new(), Vec::new());
            let mut cand = Vec::new();
            let mut hnear: Vec<usize> = Vec::new();
            // d2(H, i) of the H in `hnear`
            let mut rnear: Vec<f64> = Vec::new();
            let mut bonded: Vec<usize> = Vec::new();
            for i in range {
                if !topo.hb_ab[i] {
                    continue;
                }
                cells.candidates(i, thr1.sqrt() * (1.0 + 1e-12) + 1e-9, |j| j < i && topo.hb_ab[j], &mut cand);
                let mut hnear_done = false;
                for &j in &cand {
                    if topo.hbbas[i] * topo.hbaci[j] < 1.0e-6 && topo.hbaci[i] * topo.hbbas[j] < 1.0e-6 {
                        continue;
                    }
                    let rab = d2(i, j);
                    if rab > thr1 {
                        continue;
                    }
                    if !hnear_done {
                        // ascending (`CellList::candidates`)
                        cells.candidates(i, thr2.sqrt() * (1.0 + 1e-12) + 1e-9, |h| model.is_hb_h[h], &mut hnear);
                        rnear.clear();
                        rnear.extend(hnear.iter().map(|&h| d2(h, i)));
                        hnear_done = true;
                    }
                    let ijnonbond = topo.bpair.get(j, i) != 1;
                    // the H bonded to i or j, ascending and unique (each list is sorted)
                    bonded.clear();
                    bonded.extend_from_slice(&model.bonded_h[i]);
                    bonded.extend_from_slice(&model.bonded_h[j]);
                    bonded.sort_unstable();
                    bonded.dedup();
                    // the union with `hnear`, ascending: an H bonded to i or j takes the checks
                    // of the bonded H; any other H can only make an hb1 triplet, whose test
                    // (rab + rih) + rjh < thr2 fails when rab + rih alone does not pass (rjh >= 0)
                    let (mut a, mut b) = (0, 0);
                    while a < hnear.len() || b < bonded.len() {
                        let next_near = hnear.get(a).copied().unwrap_or(usize::MAX);
                        let next_bonded = bonded.get(b).copied().unwrap_or(usize::MAX);
                        if next_bonded <= next_near {
                            let nh = next_bonded;
                            b += 1;
                            if next_near == nh {
                                a += 1;
                            }
                            let rih = d2(nh, i);
                            let rjh = d2(nh, j);
                            if topo.bpair.get(i, nh) == 1 && ijnonbond {
                                hb2.push([i, j, nh]);
                                continue;
                            }
                            if topo.bpair.get(j, nh) == 1 && ijnonbond {
                                hb2.push([j, i, nh]);
                                continue;
                            }
                            if rab + rih + rjh < thr2 {
                                hb1.push([i, j, nh]);
                            }
                        } else {
                            let (nh, rih) = (next_near, rnear[a]);
                            a += 1;
                            let s = rab + rih;
                            if s.partial_cmp(&thr2) != Some(std::cmp::Ordering::Less) {
                                continue;
                            }
                            if s + d2(nh, j) < thr2 {
                                hb1.push([i, j, nh]);
                            }
                        }
                    }
                }
            }
            (hb1, hb2)
        };
        for (hb1, hb2) in threads.install(|| axtb_base::threads::map_chunks(n, threads.parallel(), atoms)) {
            out.hb1.extend(hb1);
            out.hb2.extend(hb2);
        }
    }
    if !topo.xb_ax.is_empty() {
        let mut cand = Vec::new();
        for &(a, x) in &topo.xb_ax {
            cells.candidates(a, thr2.sqrt() * (1.0 + 1e-12) + 1e-9, |j| topo.xb_b[j], &mut cand);
            for &j in &cand {
                if j == a || j == x {
                    continue;
                }
                if topo.bpair.get(x, j) <= 3 {
                    continue;
                }
                let d = [xyz[j][0] - xyz[a][0], xyz[j][1] - xyz[a][1], xyz[j][2] - xyz[a][2]];
                let rab = d[0] * d[0] + d[1] * d[1] + d[2] * d[2];
                if rab > thr2 {
                    continue;
                }
                out.xb.push([a, j, x]);
            }
        }
    }
    out
}

/// HB coordination number of the H atoms of HB-coupled bonds (`dncoord_erf`): per H the
/// value and, per (entry, B), the vector `dtmp * r_HB / r` (d cn_H / d x_B).
#[derive(Clone, Debug, Default)]
pub struct HbCn {
    pub cn: Vec<f64>,
    /// for every bond_hb entry, per B: d cn(H) / d x_B
    pub dcn_b: Vec<Vec<[f64; 3]>>,
    /// d cn(H) / d x_H (sum over all entries with this H)
    pub dcn_h: Vec<[f64; 3]>,
}

pub const HB_KN: f64 = 27.5;
pub const HB_RCOV_SCAL: f64 = 1.78;
pub const HB_HLFOSQRTPI: f64 = 1.0 / 1.77245385091;

pub fn hb_cn(topo: &Topology, param: &Param, xyz: &[[f64; 3]]) -> HbCn {
    let n = topo.natoms();
    let mut cn = vec![0.0; n];
    let mut dcn_h = vec![[0.0; 3]; n];
    let mut dcn_b = Vec::with_capacity(topo.bond_hb.len());
    for e in &topo.bond_hb {
        let h = e.h;
        let mut v = Vec::with_capacity(e.b.len());
        for &b in &e.b {
            let rij = [xyz[b][0] - xyz[h][0], xyz[b][1] - xyz[h][1], xyz[b][2] - xyz[h][2]];
            let r2 = rij[0].powi(2) + rij[1].powi(2) + rij[2].powi(2);
            if r2 > 900.0e0 {
                v.push([0.0; 3]);
                continue;
            }
            let r = r2.sqrt();
            let rcovij = HB_RCOV_SCAL * (param.rcov[topo.numbers[h] - 1] + param.rcov[topo.numbers[b] - 1]);
            let tmp = 0.5 * (1.0 + erf(-HB_KN * (r - rcovij) / rcovij));
            let dtmp =
                -(HB_HLFOSQRTPI * HB_KN * (-(HB_KN.powi(2) * (r - rcovij).powi(2) / rcovij.powi(2))).exp() / rcovij);
            cn[h] += tmp;
            cn[b] += tmp;
            let d = [dtmp * rij[0] / r, dtmp * rij[1] / r, dtmp * rij[2] / r];
            for c in 0..3 {
                dcn_h[h][c] -= d[c];
            }
            v.push(d);
        }
        dcn_b.push(v);
    }
    HbCn { cn, dcn_b, dcn_h }
}

/// Result of one evaluation.
#[derive(Clone, Debug)]
pub struct Eval {
    pub energies: Energies,
    pub gradient: Option<Vec<[f64; 3]>>,
    /// EEQ charges at this geometry
    pub charges: Vec<f64>,
    /// numbers of HB/XB terms
    pub nhb1: usize,
    pub nhb2: usize,
    pub nxb: usize,
    pub timings: Vec<(&'static str, f64)>,
}

fn add_grad(g: &mut [[f64; 3]], other: &[[f64; 3]]) {
    for (a, b) in g.iter_mut().zip(other) {
        a[0] += b[0];
        a[1] += b[1];
        a[2] += b[2];
    }
}

#[inline]
fn addv(g: &mut [f64; 3], v: V3<f64>) {
    g[0] += v[0];
    g[1] += v[1];
    g[2] += v[2];
}

/// Parameters of the stretch of bond k.
pub fn bond_par(topo: &Topology, k: usize) -> BondPar {
    let b = &topo.bonds[k];
    let (zi, zj) = (topo.numbers[b.i], topo.numbers[b.j]);
    BondPar {
        r0i: data::RAB_R0[zi - 1],
        cnfi: data::RAB_CNFAK[zi - 1],
        r0j: data::RAB_R0[zj - 1],
        cnfj: data::RAB_CNFAK[zj - 1],
        shift: b.vbond[0],
        ff: param::rab_ff(zi, zj),
        scale: param::rab_scale(zi, zj),
        steep: b.vbond[1],
        pref: b.vbond[2],
    }
}

/// Intermediate quantities of an evaluation that the Hessian needs.
pub struct State {
    pub lists: HbLists,
    pub lcn: LogCn,
    pub hbcn: HbCn,
    pub charges: Vec<f64>,
    /// The factorized bordered EEQ matrix
    pub eeq: ChargeSolver,
    /// dE/dCN of every atom (all CN-dependent terms)
    pub w_cn: Vec<f64>,
}

/// Energy (and gradient) at `xyz` (bohr).
pub fn evaluate(topo: &Topology, model: &Model, xyz: &[[f64; 3]], want_grad: bool, threads: &Threads) -> Result<Eval> {
    Ok(evaluate_full(topo, model, xyz, want_grad, false, threads)?.0)
}

/// Energy (and gradient) at `xyz`, optionally keeping the intermediate [`State`].
pub fn evaluate_full(
    topo: &Topology,
    model: &Model,
    xyz: &[[f64; 3]],
    want_grad: bool,
    keep_state: bool,
    threads: &Threads,
) -> Result<(Eval, Option<State>)> {
    let n = topo.natoms();
    let at = &topo.numbers;
    let p = &model.param;
    let thr = &model.thr;
    let mut timings = Vec::new();
    let mut t = std::time::Instant::now();
    let mut lap = |name: &'static str, t: &mut std::time::Instant| {
        timings.push((name, t.elapsed().as_secs_f64()));
        *t = std::time::Instant::now();
    };
    let mut e = Energies::default();
    let mut grad = vec![[0.0f64; 3]; n];
    let cells = CellList::new(xyz, 8.0);
    let nch = NCHUNK.min(n.max(1));

    // HB/XB lists
    let lists = hb_lists(topo, model, xyz, &cells, threads);
    lap("hb lists", &mut t);

    // non-bonded repulsion
    {
        let repthr = thr.rep;
        let rc = repthr.sqrt() * (1.0 + 1e-12) + 1e-9;
        let parts = threads.map(nch, |c| {
            let mut gl = if want_grad { vec![[0.0; 3]; n] } else { Vec::new() };
            let mut er = 0.0;
            let mut cand = Vec::new();
            let mut i = c;
            while i < n {
                cells.candidates(i, rc, |j| j <= i, &mut cand);
                for &j in &cand {
                    let d = [xyz[i][0] - xyz[j][0], xyz[i][1] - xyz[j][1], xyz[i][2] - xyz[j][2]];
                    let r2 = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().powi(2);
                    if r2 > repthr || r2 < 1.0e-8 {
                        continue;
                    }
                    if topo.bpair.get(i, j) == 1 {
                        continue;
                    }
                    let zz = param::repz(at[i]) * param::repz(at[j]) * p.repscaln * 1.0;
                    let (ev, gi) = terms::rep_nonbonded(xyz[i], xyz[j], topo.alphanb(i, j), zz);
                    er += ev;
                    if want_grad {
                        addv(&mut gl[i], gi);
                        addv(&mut gl[j], [-gi[0], -gi[1], -gi[2]]);
                    }
                }
                i += nch;
            }
            (er, gl)
        });
        for (er, gl) in parts {
            e.repulsion += er;
            if want_grad {
                add_grad(&mut grad, &gl);
            }
        }
    }
    lap("repulsion", &mut t);

    // coordination numbers
    let lcn = LogCn::new(at, xyz, thr.cn, p, Some(&cells));
    let cn = &lcn.cn;
    let any_hb_bond = topo.bonds.iter().any(|b| b.nr_hb > 0);
    let hbcn = if any_hb_bond { hb_cn(topo, p, xyz) } else { HbCn::default() };
    let mut w_cn = vec![0.0; n]; // dE/dCN accumulated over terms
    lap("cn", &mut t);

    // EEQ charges and electrostatics
    let (q, es, eeq_solver) = eeq(topo, xyz, cn, threads.par())?;
    e.electrostatic = es;
    lap("eeq solve", &mut t);
    if want_grad {
        let parts = threads.map(nch, |c| {
            let mut gl = vec![[0.0; 3]; n];
            let mut i = c;
            while i < n {
                for j in 0..i {
                    let r3 = terms::eeq_pair_grad(xyz[i], xyz[j], topo.alpeeq[i], topo.alpeeq[j], q[i], q[j]);
                    addv(&mut gl[i], r3);
                    addv(&mut gl[j], [-r3[0], -r3[1], -r3[2]]);
                }
                i += nch;
            }
            gl
        });
        for gl in parts {
            add_grad(&mut grad, &gl);
        }
        for i in 0..n {
            let qtmp = q[i] * data::CNF[at[i] - 1] / (2.0e0 * cn[i].sqrt() + 1.0e-16);
            w_cn[i] -= qtmp;
        }
    }
    lap("eeq gradient", &mut t);

    // dispersion
    {
        let dispthr = thr.disp;
        let rc = dispthr.sqrt() * (1.0 + 1e-12) + 1e-9;
        let weights: Vec<(Vec<f64>, Vec<f64>)> = (0..n).map(|i| model.disp.weights(at[i], cn[i])).collect();
        let parts = threads.map(nch, |c| {
            let mut gl = if want_grad { vec![[0.0; 3]; n] } else { Vec::new() };
            let mut wl = if want_grad { vec![0.0; n] } else { Vec::new() };
            let mut ed = 0.0;
            let mut cand = Vec::new();
            let mut i = c;
            while i < n {
                cells.candidates(i, rc, |j| j < i, &mut cand);
                let (gwi, dgwi) = &weights[i];
                for &j in &cand {
                    let d = [xyz[i][0] - xyz[j][0], xyz[i][1] - xyz[j][1], xyz[i][2] - xyz[j][2]];
                    let r2 = d[0].powi(2) + d[1].powi(2) + d[2].powi(2);
                    if r2 >= dispthr {
                        continue;
                    }
                    let (gwj, dgwj) = &weights[j];
                    let tab = &model.disp.c6[&(at[i], at[j])];
                    let nj = gwj.len();
                    let (mut c6, mut dc6i, mut dc6j) = (0.0, 0.0, 0.0);
                    for ii in 0..gwi.len() {
                        for jj in 0..nj {
                            let refc6 = tab[ii * nj + jj];
                            c6 += gwi[ii] * gwj[jj] * refc6;
                            dc6i += dgwi[ii] * gwj[jj] * refc6;
                            dc6j += gwi[ii] * dgwj[jj] * refc6;
                        }
                    }
                    let r4r2ij = 3.0 * p.sqrtzr4r2[at[i] - 1] * p.sqrtzr4r2[at[j] - 1];
                    let r0 = p.d3r0[param::lin(at[i] - 1, at[j] - 1)];
                    let zeta = topo.zeta[i] * topo.zeta[j];
                    let (ev, gi, wi, wj) =
                        terms::disp_pair(xyz[i], xyz[j], c6, dc6i, dc6j, r4r2ij, r0, zeta, p.dispscale);
                    ed += ev;
                    if want_grad {
                        addv(&mut gl[i], gi);
                        addv(&mut gl[j], [-gi[0], -gi[1], -gi[2]]);
                        wl[i] += wi;
                        wl[j] += wj;
                    }
                }
                i += nch;
            }
            (ed, gl, wl)
        });
        for (ed, gl, wl) in parts {
            e.dispersion += ed;
            if want_grad {
                add_grad(&mut grad, &gl);
                for k in 0..n {
                    w_cn[k] += wl[k];
                }
            }
        }
    }
    lap("dispersion", &mut t);

    // bonds (stretch and bonded repulsion)
    {
        let t1 = 1.0 - p.vbond_scale;
        for (k, b) in topo.bonds.iter().enumerate() {
            let bp = bond_par(topo, k);
            let hb = model.bond_hb_ah[k].map(|(_, h)| (hbcn.cn[h], t1));
            let o = terms::bond(xyz[b.i], xyz[b.j], cn[b.i], cn[b.j], &bp, hb);
            e.bond += o.e;
            if want_grad {
                addv(&mut grad[b.i], o.gi);
                addv(&mut grad[b.j], o.gj);
                w_cn[b.i] += o.dcni;
                w_cn[b.j] += o.dcnj;
                if let Some((a, h)) = model.bond_hb_ah[k] {
                    // the CN gradient of H for every bond_hb entry of this A-H pair
                    for ent in &topo.bond_hb {
                        if ent.a == a && ent.h == h {
                            let zz = o.dhb;
                            for c in 0..3 {
                                grad[h][c] += hbcn.dcn_h[h][c] * zz;
                            }
                            for &bb in &ent.b {
                                // xtb subtracts hb_dcn(:,B,H) = -(d cn_H / d x_B)
                                let v = hbcn_dcn_bh(topo, &hbcn, bb, h);
                                for c in 0..3 {
                                    grad[bb][c] -= v[c] * zz;
                                }
                            }
                        }
                    }
                }
            }
            let (zi, zj) = (at[b.i], at[b.j]);
            let alpha = (data::REPA[zi - 1] * data::REPA[zj - 1]).sqrt();
            let repab = param::repz(zi) * param::repz(zj) * p.repscalb;
            let (er, gi) = terms::rep_bonded(xyz[b.i], xyz[b.j], alpha, repab);
            e.repulsion += er;
            if want_grad {
                addv(&mut grad[b.i], gi);
                addv(&mut grad[b.j], [-gi[0], -gi[1], -gi[2]]);
            }
        }
    }
    lap("bonds", &mut t);

    // CN chain rule of all CN-dependent terms
    if want_grad {
        lcn.chain(&w_cn, 1.0, &mut grad);
    }

    // bends
    for a in &topo.angles {
        let rc = [p.rcov[at[a.i] - 1], p.rcov[at[a.j] - 1], p.rcov[at[a.k] - 1]];
        let (ev, g) = terms::bend(xyz[a.i], xyz[a.j], xyz[a.k], a.phi0, a.fc, rc, p.atcuta);
        e.angle += ev;
        if want_grad {
            addv(&mut grad[a.i], g[0]);
            addv(&mut grad[a.j], g[1]);
            addv(&mut grad[a.k], g[2]);
        }
    }
    // torsions and out-of-plane terms
    for tq in &topo.torsions {
        let idx = tq.atoms;
        let x = [xyz[idx[0]], xyz[idx[1]], xyz[idx[2]], xyz[idx[3]]];
        let rc = idx.map(|k| p.rcov[at[k] - 1]);
        let (ev, g) = terms::torsion(x, tq.nrot, tq.phi0, tq.fc, rc, p.atcutt);
        e.torsion += ev;
        if want_grad {
            for k in 0..4 {
                addv(&mut grad[idx[k]], g[k]);
            }
        }
    }
    // triple-bond torsions: xtb evaluates the last entry once per entry
    if let Some(last) = topo.stors.last() {
        let m = topo.stors.len();
        if !last.contains(&usize::MAX) {
            let c = [last[0], last[1], last[4], last[5]];
            let (ev, g) = terms::stors([xyz[c[0]], xyz[c[1]], xyz[c[2]], xyz[c[3]]]);
            for _ in 0..m {
                e.torsion += ev;
                if want_grad {
                    for k in 0..4 {
                        addv(&mut grad[c[k]], g[k]);
                    }
                }
            }
        }
    }
    lap("bend/torsion", &mut t);

    // bonded ATM
    for tr in &topo.batm {
        let [i, j, k] = *tr;
        let c9 = terms::batm_c9(
            topo.qa[i],
            topo.qa[j],
            topo.qa[k],
            p.zb3atm[at[i] - 1],
            p.zb3atm[at[j] - 1],
            p.zb3atm[at[k] - 1],
        );
        let (ev, g) = terms::batm(xyz[i], xyz[j], xyz[k], c9);
        e.batm += ev;
        if want_grad {
            addv(&mut grad[i], g[0]);
            addv(&mut grad[j], g[1]);
            addv(&mut grad[k], g[2]);
        }
    }
    lap("batm", &mut t);

    // hydrogen bonds
    let qs = |z: f64, q: f64| terms::qscale(z, p.hbsf, q);
    for tr in &lists.hb1 {
        let [a, b, h] = *tr;
        let ca = [topo.hbbas[a], topo.hbaci[a]];
        let cb = [topo.hbbas[b], topo.hbaci[b]];
        let radab = param::rad(at[a]) + param::rad(at[b]);
        if let Some((ev, g)) = terms::hb1(
            xyz[a],
            xyz[b],
            xyz[h],
            ca,
            cb,
            qs(-p.hbst, topo.qa[a]),
            qs(-p.hbst, topo.qa[b]),
            qs(p.hbst, topo.qa[h]),
            radab,
            &model.hb,
        ) {
            e.hb += ev;
            if want_grad {
                addv(&mut grad[a], g[0]);
                addv(&mut grad[b], g[1]);
                addv(&mut grad[h], g[2]);
            }
        }
    }
    for tr in &lists.hb2 {
        let [a, b, h] = *tr;
        if let Some((ev, atoms, g)) = hb2_term(topo, model, xyz, a, b, h) {
            e.hb += ev;
            if want_grad {
                for (k, &idx) in atoms.iter().enumerate() {
                    addv(&mut grad[idx], g[k]);
                }
            }
        }
    }
    // halogen bonds
    for tr in &lists.xb {
        let [a, b, x] = *tr;
        let radab = param::rad(at[a]) + param::rad(at[b]);
        let cx = p.xbaci[at[x] - 1];
        let qb = terms::qscale(-p.xbst, p.xbsf, topo.qa[b]);
        let qx = terms::qscale(p.xbst, p.xbsf, topo.qa[x]);
        if let Some((ev, g)) = terms::xb(xyz[a], xyz[b], xyz[x], cx, qb, qx, radab, p) {
            e.xb += ev;
            if want_grad {
                addv(&mut grad[a], g[0]);
                addv(&mut grad[b], g[1]);
                addv(&mut grad[x], g[2]);
            }
        }
    }
    lap("hb/xb", &mut t);

    e.total =
        e.electrostatic + e.dispersion + e.repulsion + e.bond + e.angle + e.torsion + e.hb + e.xb + e.batm + e.external;
    let ev = Eval {
        energies: e,
        gradient: if want_grad { Some(grad) } else { None },
        charges: q.clone(),
        nhb1: lists.hb1.len(),
        nhb2: lists.hb2.len(),
        nxb: lists.xb.len(),
        timings,
    };
    let state = if keep_state { Some(State { lists, lcn, hbcn, charges: q, eeq: eeq_solver, w_cn }) } else { None };
    Ok((ev, state))
}

/// `hb_dcn(:,B,H)` of xtb negated: sum over all bond_hb entries with this (B, H) of d cn_H / d x_B.
fn hbcn_dcn_bh(topo: &Topology, hbcn: &HbCn, b: usize, h: usize) -> [f64; 3] {
    let mut v = [0.0; 3];
    for (ei, ent) in topo.bond_hb.iter().enumerate() {
        if ent.h != h {
            continue;
        }
        for (bi, &bb) in ent.b.iter().enumerate() {
            if bb == b {
                for c in 0..3 {
                    v[c] -= hbcn.dcn_b[ei][bi][c];
                }
            }
        }
    }
    // xtb stores -dtmp*rij/r and subtracts it; return d cn_H/d x_B with xtb's sign so
    // that `grad[B] -= v * zz` reproduces `g(:,hbB) = g(:,hbB) - hb_dcn(:,hbB,hbH)*zz`
    v
}

/// Which case-2/3 HB kernel an (A, B, H) triple uses, with its extra atoms.
#[derive(Clone, Debug)]
pub enum Hb2Kind {
    /// carbonyl/nitro acceptor O=C: C and the other neighbors R of C
    Carbonyl { c: usize, rs: Vec<usize> },
    /// pyridine-type N with two neighbors (sorted by distance)
    Rnr { n1: usize, n2: usize },
    /// any other acceptor with its neighbors (sorted by distance)
    Default { nbs: Vec<usize> },
}

/// Selects the case-2/3 HB kernel of (A, B, H) as `gfnff_eg` does.
pub fn hb2_kind(topo: &Topology, xyz: &[[f64; 3]], b: usize) -> Hb2Kind {
    let norms = topo.conventions.params().tie_norms;
    let at = &topo.numbers;
    let nbk = topo.nb[b].len();
    let mut nbnbk = 0;
    let mut atnb = 0;
    let mut cnb = usize::MAX;
    if at[b] == 8 && nbk == 1 {
        if let Some(c) = jth_nb(&topo.nb[b], xyz, b, 1, norms) {
            nbnbk = topo.nb[c].len();
            atnb = at[c];
            cnb = c;
        }
    }
    if at[b] == 8 && nbk == 1 && (atnb == 6 || atnb == 7) && nbnbk > 1 {
        Hb2Kind::Carbonyl { c: cnb, rs: topo.nb[cnb].iter().copied().filter(|&r| r != b).collect() }
    } else if at[b] == 7 && nbk == 2 {
        Hb2Kind::Rnr {
            n1: jth_nb(&topo.nb[b], xyz, b, 1, norms).unwrap(),
            n2: jth_nb(&topo.nb[b], xyz, b, 2, norms).unwrap(),
        }
    } else {
        Hb2Kind::Default { nbs: (1..=nbk).map(|k| jth_nb(&topo.nb[b], xyz, b, k, norms).unwrap()).collect() }
    }
}

impl Hb2Kind {
    /// Atoms of the term in the order of the kernel's gradient output.
    pub fn atoms(&self, a: usize, b: usize, h: usize) -> Vec<usize> {
        match self {
            Hb2Kind::Carbonyl { c, rs } => {
                let mut v = vec![a, b, h, *c];
                v.extend(rs.iter().copied());
                v
            }
            Hb2Kind::Rnr { n1, n2 } => vec![a, b, h, *n1, *n2],
            Hb2Kind::Default { nbs } => {
                let mut v = vec![a, b, h];
                v.extend(nbs.iter().copied());
                v
            }
        }
    }
}

/// Evaluates the case-2/3 HB kernel with positions `x` of [`Hb2Kind::atoms`].
pub fn hb2_eval<T: crate::real::Real>(
    topo: &Topology,
    model: &Model,
    kind: &Hb2Kind,
    a: usize,
    b: usize,
    h: usize,
    x: &[V3<T>],
) -> Option<(T, Vec<V3<T>>)> {
    let at = &topo.numbers;
    let p = &model.param;
    let ca = [topo.hbbas[a], topo.hbaci[a]];
    let cb = [topo.hbbas[b], topo.hbaci[b]];
    let radab = param::rad(at[a]) + param::rad(at[b]);
    let qa = terms::qscale(-p.hbst, p.hbsf, topo.qa[a]);
    let qb = terms::qscale(-p.hbst, p.hbsf, topo.qa[b]);
    let qh = terms::qscale(p.hbst, p.hbsf, topo.qa[h]);
    match kind {
        Hb2Kind::Carbonyl { .. } => terms::hb3(x[0], x[1], x[2], x[3], &x[4..], ca, cb, qa, qb, qh, radab, &model.hb),
        Hb2Kind::Rnr { .. } => {
            terms::hb2_rnr(x[0], x[1], x[2], [x[3], x[4]], param::repz(at[b]), ca, cb, qa, qb, qh, radab, &model.hb)
        }
        Hb2Kind::Default { nbs } => {
            terms::hb2(x[0], x[1], x[2], &x[3..], at[b] == 7 && nbs.len() == 1, ca, cb, qa, qb, qh, radab, &model.hb)
        }
    }
}

/// Selects and evaluates the case-2/3 HB term of (A, B, H) as `gfnff_eg` does.
pub fn hb2_term(
    topo: &Topology,
    model: &Model,
    xyz: &[[f64; 3]],
    a: usize,
    b: usize,
    h: usize,
) -> Option<(f64, Vec<usize>, Vec<V3<f64>>)> {
    let kind = hb2_kind(topo, xyz, b);
    let atoms = kind.atoms(a, b, h);
    let x: Vec<V3<f64>> = atoms.iter().map(|&k| xyz[k]).collect();
    let (ev, g) = hb2_eval(topo, model, &kind, a, b, h, &x)?;
    Some((ev, atoms, g))
}

/// Geometry-dependent EEQ (`goed_gfnff`): charges, electrostatic energy and the factorized
/// bordered EEQ matrix (one charge constraint per fragment; kept for the charge response of the
/// Hessian).
pub fn eeq(topo: &Topology, xyz: &[[f64; 3]], cn: &[f64], par: Par) -> Result<(Vec<f64>, f64, ChargeSolver)> {
    let n = topo.natoms();
    let at = &topo.numbers;
    let nf = topo.nfrag;
    let eqs = ChargeEquations::new(
        n,
        Constraints::Fragments { fragment: &topo.fraglist, count: nf },
        |i| TSQRT2PI / topo.alpeeq[i].sqrt() + topo.gameeq[i],
        |i, j| {
            let d = [xyz[i][0] - xyz[j][0], xyz[i][1] - xyz[j][1], xyz[i][2] - xyz[j][2]];
            let r = (d[0].powi(2) + d[1].powi(2) + d[2].powi(2)).sqrt();
            let gammij = 1. / (topo.alpeeq[i] + topo.alpeeq[j]).sqrt();
            let tmp = erf(gammij * r);
            tmp / r
        },
    );
    let solver = eqs.try_factorize(par).ok_or_else(|| Error::Numerical("singular EEQ matrix".into()))?;
    let mut x: Vec<f64> = (0..n).map(|i| topo.chieeq[i] + data::CNF[at[i] - 1] * cn[i].sqrt()).collect();
    x.extend_from_slice(&topo.qfrag[..nf]);
    let mut q = solver.charges(&x);
    if n == 1 {
        q[0] = topo.charge;
    }
    let mut es = 0.0;
    for i in 0..n {
        for j in 0..i {
            let d = [xyz[i][0] - xyz[j][0], xyz[i][1] - xyz[j][1], xyz[i][2] - xyz[j][2]];
            let r = (d[0].powi(2) + d[1].powi(2) + d[2].powi(2)).sqrt();
            let gammij = 1. / (topo.alpeeq[i] + topo.alpeeq[j]).sqrt();
            let tmp = erf(gammij * r);
            es += q[i] * q[j] * tmp / r;
        }
        es = es - q[i] * (topo.chieeq[i] + data::CNF[at[i] - 1] * cn[i].sqrt())
            + q[i] * q[i] * 0.5e0 * (topo.gameeq[i] + TSQRT2PI / topo.alpeeq[i].sqrt());
    }
    Ok((q, es, solver))
}

impl DispModel {
    /// [`DispModel::weights`] for any scalar type.
    pub fn weights_t<T: crate::real::Real>(&self, z: usize, cn: T) -> (Vec<T>, Vec<T>) {
        let wf = 4.0;
        let refcn = &self.refcn[z - 1];
        let wcn = |r: f64| -> T {
            let val = (cn - r).powi(2) * (-wf);
            if val.val() < -200.0 {
                T::cst(0.0)
            } else {
                val.exp()
            }
        };
        let mut norm = T::cst(0.0);
        let mut dnorm = T::cst(0.0);
        for &r in refcn {
            let gw = wcn(r);
            norm += gw;
            dnorm += cn.rsub(r) * (2.0 * wf) * gw;
        }
        let zero_norm = norm.val().is_nan() || norm.val() <= 1e-80;
        let norm = if zero_norm { T::cst(0.0) } else { norm.rdiv(1.0) };
        let maxcn = refcn.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut gw = Vec::with_capacity(refcn.len());
        let mut dgw = Vec::with_capacity(refcn.len());
        for &r in refcn {
            let expw = wcn(r);
            let expd = cn.rsub(r) * (2.0 * wf) * expw;
            let mut gwk = expw * norm;
            if zero_norm {
                gwk = T::cst(if (maxcn - r).abs() < 1e-12 { 1.0 } else { 0.0 });
            }
            gw.push(gwk);
            dgw.push(expd * norm - expw * dnorm * norm.powi(2));
        }
        (gw, dgw)
    }

    /// C6 and its derivatives with respect to both CNs from the reference weights.
    pub fn c6_t<T: crate::real::Real>(
        &self,
        zi: usize,
        zj: usize,
        wi: &(Vec<T>, Vec<T>),
        wj: &(Vec<T>, Vec<T>),
    ) -> (T, T, T) {
        let tab = &self.c6[&(zi, zj)];
        let nj = wj.0.len();
        let (mut c6, mut dc6i, mut dc6j) = (T::cst(0.0), T::cst(0.0), T::cst(0.0));
        for ii in 0..wi.0.len() {
            for jj in 0..nj {
                let refc6 = tab[ii * nj + jj];
                c6 += wi.0[ii] * wj.0[jj] * refc6;
                dc6i += wi.1[ii] * wj.0[jj] * refc6;
                dc6j += wi.0[ii] * wj.1[jj] * refc6;
            }
        }
        (c6, dc6i, dc6j)
    }
}
