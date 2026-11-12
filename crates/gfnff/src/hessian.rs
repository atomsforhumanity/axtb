//! Analytic Hessian of GFN-FF: the exact derivative of xtb's analytic gradient.
//!
//! Every column (one Cartesian coordinate of one atom) is the forward-mode derivative
//! (dual numbers, `real::Dual`) of the gradient code of `energy.rs`, restricted to the
//! terms that coordinate influences: terms containing the atom, terms whose coordination
//! numbers change (its CN neighbors), the dispersion partners of those, the CN and HB-CN
//! chain rules. The EEQ charges are held fixed in the columns; their response is added
//! analytically: with R = d(dE/dq)/dx and the bordered EEQ matrix M (charge constraints),
//! H_resp = -R^T [M^-1 R]_q, a dense block solved with the LU factors of the energy
//! evaluation. Columns are independent and written by one task each; the charge response
//! uses faer's parallel kernels (see the crate documentation on thread counts).
//!
//! Where xtb's gradient is not the exact derivative of its energy (truncated sqrt(pi) of
//! the CN derivatives, the pyridine-type HB term), this Hessian is the derivative of the
//! gradient, i.e. what finite differences of xtb's or axtb's gradient converge to; it is
//! symmetrized at the end (`max_asymmetry` reports how far it was from symmetric).

use crate::data;
use crate::energy::{self, bond_par, hb2_eval, hb2_kind, Hb2Kind, Model, State};
use crate::error::{Error, Result};
use crate::param;
use crate::real::{Dual, Real, V3};
use crate::terms;
use crate::topology::Topology;
use axtb_base::fd;
use axtb_base::ncoord::{dlog_cap, log_cap, XtbErf};
use axtb_base::threads::{for_slices, Threads};
use faer::Mat;

/// A nuclear Hessian.
#[derive(Clone, Debug)]
pub struct Hessian {
    /// Energy at the geometry (Eh).
    pub energy: f64,
    /// Gradient at the geometry (Eh/bohr).
    pub gradient: Vec<[f64; 3]>,
    /// Row-major 3N x 3N matrix of second derivatives (Eh/bohr²), symmetrized.
    pub hessian: Vec<f64>,
    /// Largest |H_ij - H_ji| before symmetrization.
    pub max_asymmetry: f64,
    /// d(dipole)/d(x_k) of every Cartesian coordinate k (e, dipole of the EEQ charges about
    /// the origin), 3N entries.
    pub dipole_derivatives: Vec<[f64; 3]>,
    /// Wall time of the steps (s).
    pub timings: Vec<(&'static str, f64)>,
}

/// Dispersion reference weights of an atom and their CN derivatives.
type Weights = (Vec<Dual>, Vec<Dual>);

struct Ctx<'a> {
    topo: &'a Topology,
    model: &'a Model,
    xyz: &'a [[f64; 3]],
    st: &'a State,
    /// CN pairs of every atom (indices into st.lcn.pairs)
    cn_pairs: Vec<Vec<usize>>,
    bonds_of: Vec<Vec<usize>>,
    /// HB-coupled bonds whose HB-CN involves an atom as B or H
    hbbonds_of: Vec<Vec<usize>>,
    angles_of: Vec<Vec<usize>>,
    tors_of: Vec<Vec<usize>>,
    batm_of: Vec<Vec<usize>>,
    hb1_of: Vec<Vec<usize>>,
    hb2: Vec<(Hb2Kind, Vec<usize>)>,
    hb2_of: Vec<Vec<usize>>,
    xb_of: Vec<Vec<usize>>,
    /// d(dlogCN/dCN_raw)/dCN_raw of every atom
    ddlog: Vec<f64>,
    /// dispersion weights of every atom (zero-derivative duals)
    wts_dual: Vec<Weights>,
    /// (entry, B index) of every bond_hb entry per H
    hb_entries_of_h: Vec<Vec<(usize, usize)>>,
}

fn index<F: Fn(usize) -> Vec<usize>>(n: usize, m: usize, atoms: F) -> Vec<Vec<usize>> {
    let mut out = vec![Vec::new(); n];
    for t in 0..m {
        let mut a = atoms(t);
        a.sort_unstable();
        a.dedup();
        for k in a {
            out[k].push(t);
        }
    }
    out
}

impl<'a> Ctx<'a> {
    fn new(topo: &'a Topology, model: &'a Model, xyz: &'a [[f64; 3]], st: &'a State) -> Self {
        let n = topo.natoms();
        let mut cn_pairs = vec![Vec::new(); n];
        for (k, p) in st.lcn.pairs.iter().enumerate() {
            cn_pairs[p.i as usize].push(k);
            cn_pairs[p.j as usize].push(k);
        }
        let bonds_of = index(n, topo.bonds.len(), |k| vec![topo.bonds[k].i, topo.bonds[k].j]);
        let mut hbbonds_of = vec![Vec::new(); n];
        for (k, ah) in model.bond_hb_ah.iter().enumerate() {
            if let Some((_, h)) = ah {
                let mut atoms = vec![*h];
                for e in &topo.bond_hb {
                    if e.h == *h {
                        atoms.extend(e.b.iter().copied());
                    }
                }
                atoms.sort_unstable();
                atoms.dedup();
                for a in atoms {
                    hbbonds_of[a].push(k);
                }
            }
        }
        let angles_of = index(n, topo.angles.len(), |k| vec![topo.angles[k].i, topo.angles[k].j, topo.angles[k].k]);
        let tors_of = index(n, topo.torsions.len(), |k| topo.torsions[k].atoms.to_vec());
        let batm_of = index(n, topo.batm.len(), |k| topo.batm[k].to_vec());
        let hb1_of = index(n, st.lists.hb1.len(), |k| st.lists.hb1[k].to_vec());
        let hb2: Vec<(Hb2Kind, Vec<usize>)> = st
            .lists
            .hb2
            .iter()
            .map(|&[a, b, h]| {
                let kind = hb2_kind(topo, xyz, b);
                let atoms = kind.atoms(a, b, h);
                (kind, atoms)
            })
            .collect();
        let hb2_of = index(n, hb2.len(), |k| hb2[k].1.clone());
        let xb_of = index(n, st.lists.xb.len(), |k| st.lists.xb[k].to_vec());
        let wts_dual = (0..n)
            .map(|i| {
                let (g, d) = model.disp.weights(topo.numbers[i], st.lcn.cn[i]);
                (g.iter().map(|&v| Dual::new(v, 0.0)).collect(), d.iter().map(|&v| Dual::new(v, 0.0)).collect())
            })
            .collect();
        let mut hb_entries_of_h = vec![Vec::new(); n];
        for (ei, e) in topo.bond_hb.iter().enumerate() {
            for bi in 0..e.b.len() {
                hb_entries_of_h[e.h].push((ei, bi));
            }
        }
        Ctx {
            topo,
            model,
            xyz,
            st,
            cn_pairs,
            bonds_of,
            hbbonds_of,
            angles_of,
            tors_of,
            batm_of,
            hb1_of,
            hb2,
            hb2_of,
            xb_of,
            wts_dual,
            ddlog: (0..n).map(|a| dlog_cap(Dual::new(st.lcn.cn_raw[a], 1.0), model.param.cnmax).d).collect(),
            hb_entries_of_h,
        }
    }

    /// Position of atom i as a dual number seeded at (k, c).
    #[inline]
    fn xd(&self, i: usize, k: usize, c: usize) -> V3<Dual> {
        let p = self.xyz[i];
        let mut v = [Dual::new(p[0], 0.0), Dual::new(p[1], 0.0), Dual::new(p[2], 0.0)];
        if i == k {
            v[c].d = 1.0;
        }
        v
    }

    fn rcovij_hb(&self, h: usize, b: usize) -> f64 {
        let p = &self.model.param;
        energy::HB_RCOV_SCAL * (p.rcov[self.topo.numbers[h] - 1] + p.rcov[self.topo.numbers[b] - 1])
    }

    /// HB coordination number of H as a dual (value of the evaluation, derivative at (k, c)).
    fn hbcn_dual(&self, h: usize, k: usize, c: usize) -> Dual {
        let mut d = 0.0;
        for &(ei, bi) in &self.hb_entries_of_h[h] {
            let b = self.topo.bond_hb[ei].b[bi];
            if h != k && b != k {
                continue;
            }
            if let Some((t, _)) = terms::hbcn_pair(self.xd(h, k, c), self.xd(b, k, c), self.rcovij_hb(h, b)) {
                d += t.d;
            }
        }
        Dual::new(self.st.hbcn.cn[h], d)
    }

    /// d(gradient)/d x_{k,c} without the EEQ charge response.
    fn column(&self, k: usize, c: usize) -> Vec<f64> {
        let topo = self.topo;
        let model = self.model;
        let st = self.st;
        let xyz = self.xyz;
        let at = &topo.numbers;
        let p = &model.param;
        let n = topo.natoms();
        let mut g = vec![0.0; 3 * n];
        let add = |g: &mut Vec<f64>, a: usize, v: V3<Dual>, s: f64| {
            g[3 * a] += s * v[0].d;
            g[3 * a + 1] += s * v[1].d;
            g[3 * a + 2] += s * v[2].d;
        };

        // coordination numbers that move with x_{k,c}
        let mut rawd = vec![0.0; n];
        let mut affected = vec![k];
        for &pi in &self.cn_pairs[k] {
            let pr = &st.lcn.pairs[pi];
            let (i, j) = (pr.i as usize, pr.j as usize);
            let r0 = p.rcov[at[i] - 1] + p.rcov[at[j] - 1];
            let xi = self.xd(i, k, c);
            let xj = self.xd(j, k, c);
            let r = crate::real::norm2(crate::real::sub(xi, xj));
            let e = XtbErf::GFNFF.count_of(r, r0);
            rawd[i] += e.d;
            rawd[j] += e.d;
            // partners whose CN moves by less than 1e-14 per bohr are left out (their
            // contributions are below 1e-14 Eh/bohr²; the erf count of distant pairs
            // has derivatives down to 1e-90)
            let other = if i == k { j } else { i };
            if e.d.abs() > 1.0e-14 {
                affected.push(other);
            } else {
                rawd[other] = 0.0;
            }
        }
        let mut in_a = vec![false; n];
        for &a in &affected {
            in_a[a] = true;
        }
        let cn_dual = |a: usize| -> Dual {
            if in_a[a] {
                log_cap(Dual::new(st.lcn.cn_raw[a], rawd[a]), p.cnmax)
            } else {
                Dual::new(st.lcn.cn[a], 0.0)
            }
        };
        let mut wd = vec![0.0; n]; // d(dE/dCN)/dx_{k,c}

        // non-bonded repulsion of pairs with k
        {
            let repthr = model.thr.rep;
            for b in 0..n {
                if b == k {
                    continue;
                }
                let (i, j) = if k > b { (k, b) } else { (b, k) };
                let d = [xyz[i][0] - xyz[j][0], xyz[i][1] - xyz[j][1], xyz[i][2] - xyz[j][2]];
                let r2 = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt().powi(2);
                if r2 > repthr || r2 < 1.0e-8 || topo.bpair.get(i, j) == 1 {
                    continue;
                }
                let zz = param::repz(at[i]) * param::repz(at[j]) * p.repscaln * 1.0;
                let (_, gi) = terms::rep_nonbonded(self.xd(i, k, c), self.xd(j, k, c), topo.alphanb(i, j), zz);
                add(&mut g, i, gi, 1.0);
                add(&mut g, j, gi, -1.0);
            }
        }

        // EEQ pair gradient at fixed charges
        for b in 0..n {
            if b == k {
                continue;
            }
            let (i, j) = if k > b { (k, b) } else { (b, k) };
            let gi = terms::eeq_pair_grad(
                self.xd(i, k, c),
                self.xd(j, k, c),
                topo.alpeeq[i],
                topo.alpeeq[j],
                st.charges[i],
                st.charges[j],
            );
            add(&mut g, i, gi, 1.0);
            add(&mut g, j, gi, -1.0);
        }
        // CN dependence of the EEQ electronegativities (chain term weights)
        for &a in &affected {
            let cna = cn_dual(a);
            let w = -((cna.sqrt() * 2.0e0 + 1.0e-16).rdiv(st.charges[a] * data::CNF[at[a] - 1]));
            wd[a] += w.d;
        }

        // dispersion pairs with an affected atom
        {
            let dispthr = model.thr.disp;
            let wa: Vec<(usize, Weights)> =
                affected.iter().map(|&a| (a, model.disp.weights_t(at[a], cn_dual(a)))).collect();
            let mut widx = vec![usize::MAX; n];
            for (m, (a, _)) in wa.iter().enumerate() {
                widx[*a] = m;
            }
            let wref = |a: usize| -> &Weights {
                if widx[a] != usize::MAX {
                    &wa[widx[a]].1
                } else {
                    &self.wts_dual[a]
                }
            };
            for (m, &(a, _)) in wa.iter().enumerate() {
                if widx[a] != m {
                    continue;
                }
                for b in 0..n {
                    if b == a || (in_a[b] && b < a) {
                        continue;
                    }
                    let (i, j) = if a > b { (a, b) } else { (b, a) };
                    let d = [xyz[i][0] - xyz[j][0], xyz[i][1] - xyz[j][1], xyz[i][2] - xyz[j][2]];
                    let r2 = d[0].powi(2) + d[1].powi(2) + d[2].powi(2);
                    if r2 >= dispthr {
                        continue;
                    }
                    let (wi, wj) = (wref(i), wref(j));
                    let (c6, dc6i, dc6j) = model.disp.c6_t(at[i], at[j], wi, wj);
                    let r4r2ij = 3.0 * p.sqrtzr4r2[at[i] - 1] * p.sqrtzr4r2[at[j] - 1];
                    let r0 = p.d3r0[param::lin(at[i] - 1, at[j] - 1)];
                    let zeta = topo.zeta[i] * topo.zeta[j];
                    let (_, gi, w1, w2) = terms::disp_pair(
                        self.xd(i, k, c),
                        self.xd(j, k, c),
                        c6,
                        dc6i,
                        dc6j,
                        r4r2ij,
                        r0,
                        zeta,
                        p.dispscale,
                    );
                    add(&mut g, i, gi, 1.0);
                    add(&mut g, j, gi, -1.0);
                    wd[i] += w1.d;
                    wd[j] += w2.d;
                }
            }
        }

        // bonds: stretch (with CN and HB-CN dependence) and bonded repulsion
        {
            let t1 = 1.0 - p.vbond_scale;
            let mut todo: Vec<usize> = Vec::new();
            for &a in &affected {
                todo.extend_from_slice(&self.bonds_of[a]);
            }
            todo.extend_from_slice(&self.hbbonds_of[k]);
            todo.sort_unstable();
            todo.dedup();
            for bk in todo {
                let b = &topo.bonds[bk];
                let bp = bond_par(topo, bk);
                let hb = model.bond_hb_ah[bk].map(|(_, h)| (self.hbcn_dual(h, k, c), t1));
                let o = terms::bond(self.xd(b.i, k, c), self.xd(b.j, k, c), cn_dual(b.i), cn_dual(b.j), &bp, hb);
                add(&mut g, b.i, o.gi, 1.0);
                add(&mut g, b.j, o.gj, 1.0);
                wd[b.i] += o.dcni.d;
                wd[b.j] += o.dcnj.d;
                if let Some((a, h)) = model.bond_hb_ah[bk] {
                    // HB-CN chain: g(H) += zz dcn_h(H); g(B) -= zz hb_dcn(B,H)
                    let zz = o.dhb;
                    // d cn_H / d x for all entries with H (H part) and per B
                    let mut vh = [Dual::new(0.0, 0.0); 3];
                    let mut vb: Vec<(usize, V3<Dual>)> = Vec::new();
                    for &(ei, bi) in &self.hb_entries_of_h[h] {
                        let bb = topo.bond_hb[ei].b[bi];
                        let v = match terms::hbcn_pair(self.xd(h, k, c), self.xd(bb, k, c), self.rcovij_hb(h, bb)) {
                            Some((_, v)) => v,
                            None => [Dual::new(0.0, 0.0); 3],
                        };
                        for cc in 0..3 {
                            vh[cc] = -v[cc] + vh[cc];
                        }
                        vb.push((bb, v));
                    }
                    for ent in &topo.bond_hb {
                        if !(ent.a == a && ent.h == h) {
                            continue;
                        }
                        let gh = [zz * vh[0], zz * vh[1], zz * vh[2]];
                        add(&mut g, h, gh, 1.0);
                        for &bb in &ent.b {
                            // hb_dcn(:,B,H) = sum over entries with (B, H) of -v
                            let mut hv = [Dual::new(0.0, 0.0); 3];
                            for &(b2, v) in &vb {
                                if b2 == bb {
                                    for cc in 0..3 {
                                        hv[cc] = -v[cc] + hv[cc];
                                    }
                                }
                            }
                            let gb = [zz * hv[0], zz * hv[1], zz * hv[2]];
                            add(&mut g, bb, gb, -1.0);
                        }
                    }
                }
                if b.i == k || b.j == k {
                    let (zi, zj) = (at[b.i], at[b.j]);
                    let alpha = (data::REPA[zi - 1] * data::REPA[zj - 1]).sqrt();
                    let repab = param::repz(zi) * param::repz(zj) * p.repscalb;
                    let (_, gi) = terms::rep_bonded(self.xd(b.i, k, c), self.xd(b.j, k, c), alpha, repab);
                    add(&mut g, b.i, gi, 1.0);
                    add(&mut g, b.j, gi, -1.0);
                }
            }
        }

        // CN chain rule: sum_p s_p d_p with s_p = dlog_i W_i + dlog_j W_j
        for pr in st.lcn.pairs.iter() {
            let (i, j) = (pr.i as usize, pr.j as usize);
            let moving = i == k || j == k;
            let ds = rawd[i] != 0.0 || rawd[j] != 0.0 || wd[i] != 0.0 || wd[j] != 0.0;
            if !moving && !ds {
                continue;
            }
            let (wi, wj) = (st.w_cn[i], st.w_cn[j]);
            let s = st.lcn.dlog[i] * wi + st.lcn.dlog[j] * wj;
            let sd = self.ddlog[i] * rawd[i] * wi
                + st.lcn.dlog[i] * wd[i]
                + self.ddlog[j] * rawd[j] * wj
                + st.lcn.dlog[j] * wd[j];
            if moving {
                let xi = self.xd(i, k, c);
                let xj = self.xd(j, k, c);
                let d = crate::real::sub(xi, xj);
                let r = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt();
                let r0 = p.rcov[at[i] - 1] + p.rcov[at[j] - 1];
                let der = XtbErf::GFNFF.dcount_of(r, r0);
                for cc in 0..3 {
                    let dv = der * (xj[cc] - xi[cc]) / r;
                    let v = sd * dv.v + s * dv.d;
                    g[3 * j + cc] += v;
                    g[3 * i + cc] -= v;
                }
            } else {
                for cc in 0..3 {
                    let v = sd * pr.d[cc];
                    g[3 * j + cc] += v;
                    g[3 * i + cc] -= v;
                }
            }
        }

        // bends, torsions, bonded ATM
        for &ai in &self.angles_of[k] {
            let a = &topo.angles[ai];
            let rc = [p.rcov[at[a.i] - 1], p.rcov[at[a.j] - 1], p.rcov[at[a.k] - 1]];
            let (_, gg) = terms::bend_smooth(
                self.xd(a.i, k, c),
                self.xd(a.j, k, c),
                self.xd(a.k, k, c),
                a.phi0,
                a.fc,
                rc,
                p.atcuta,
            );
            add(&mut g, a.i, gg[0], 1.0);
            add(&mut g, a.j, gg[1], 1.0);
            add(&mut g, a.k, gg[2], 1.0);
        }
        for &ti in &self.tors_of[k] {
            let tq = &topo.torsions[ti];
            let idx = tq.atoms;
            let x = idx.map(|a| self.xd(a, k, c));
            let rc = idx.map(|a| p.rcov[at[a] - 1]);
            let (_, gg) = terms::torsion_smooth(x, tq.nrot, tq.phi0, tq.fc, rc, p.atcutt);
            for m in 0..4 {
                add(&mut g, idx[m], gg[m], 1.0);
            }
        }
        if let Some(last) = topo.stors.last() {
            let cs = [last[0], last[1], last[4], last[5]];
            if cs.contains(&k) {
                let (_, gg) = terms::stors_smooth(cs.map(|a| self.xd(a, k, c)));
                for _ in 0..topo.stors.len() {
                    for m in 0..4 {
                        add(&mut g, cs[m], gg[m], 1.0);
                    }
                }
            }
        }
        for &bi in &self.batm_of[k] {
            let [i, j, l] = topo.batm[bi];
            let c9 = terms::batm_c9(
                topo.qa[i],
                topo.qa[j],
                topo.qa[l],
                p.zb3atm[at[i] - 1],
                p.zb3atm[at[j] - 1],
                p.zb3atm[at[l] - 1],
            );
            let (_, gg) = terms::batm(self.xd(i, k, c), self.xd(j, k, c), self.xd(l, k, c), c9);
            add(&mut g, i, gg[0], 1.0);
            add(&mut g, j, gg[1], 1.0);
            add(&mut g, l, gg[2], 1.0);
        }

        // hydrogen and halogen bonds
        let qs = |z: f64, q: f64| terms::qscale(z, p.hbsf, q);
        for &hi in &self.hb1_of[k] {
            let [a, b, h] = st.lists.hb1[hi];
            let ca = [topo.hbbas[a], topo.hbaci[a]];
            let cb = [topo.hbbas[b], topo.hbaci[b]];
            let radab = param::rad(at[a]) + param::rad(at[b]);
            if let Some((_, gg)) = terms::hb1(
                self.xd(a, k, c),
                self.xd(b, k, c),
                self.xd(h, k, c),
                ca,
                cb,
                qs(-p.hbst, topo.qa[a]),
                qs(-p.hbst, topo.qa[b]),
                qs(p.hbst, topo.qa[h]),
                radab,
                &model.hb,
            ) {
                add(&mut g, a, gg[0], 1.0);
                add(&mut g, b, gg[1], 1.0);
                add(&mut g, h, gg[2], 1.0);
            }
        }
        for &hi in &self.hb2_of[k] {
            let [a, b, h] = st.lists.hb2[hi];
            let (kind, atoms) = &self.hb2[hi];
            let x: Vec<V3<Dual>> = atoms.iter().map(|&m| self.xd(m, k, c)).collect();
            if let Some((_, gg)) = hb2_eval(topo, model, kind, a, b, h, &x) {
                for (m, &idx) in atoms.iter().enumerate() {
                    add(&mut g, idx, gg[m], 1.0);
                }
            }
        }
        for &xi in &self.xb_of[k] {
            let [a, b, x] = st.lists.xb[xi];
            let radab = param::rad(at[a]) + param::rad(at[b]);
            let cx = p.xbaci[at[x] - 1];
            let qb = terms::qscale(-p.xbst, p.xbsf, topo.qa[b]);
            let qx = terms::qscale(p.xbst, p.xbsf, topo.qa[x]);
            if let Some((_, gg)) = terms::xb(self.xd(a, k, c), self.xd(b, k, c), self.xd(x, k, c), cx, qb, qx, radab, p)
            {
                add(&mut g, a, gg[0], 1.0);
                add(&mut g, b, gg[1], 1.0);
                add(&mut g, x, gg[2], 1.0);
            }
        }
        g
    }

    /// R = d(dE/dq)/dx (N x 3N) of the gradient code: the pair term and the CN dependence
    /// of the electronegativities.
    fn eeq_r(&self) -> Mat<f64> {
        let topo = self.topo;
        let st = self.st;
        let xyz = self.xyz;
        let n = topo.natoms();
        let q = &st.charges;
        let mut r = Mat::<f64>::zeros(n + topo.nfrag, 3 * n);
        for i in 0..n {
            for j in 0..i {
                let d = [xyz[i][0] - xyz[j][0], xyz[i][1] - xyz[j][1], xyz[i][2] - xyz[j][2]];
                let r2 = d[0].powi(2) + d[1].powi(2) + d[2].powi(2);
                let rab = r2.sqrt();
                let gammij = 1. / (topo.alpeeq[i] + topo.alpeeq[j]).sqrt();
                let dd = terms::eeq_kernel_dd(r2, rab, gammij);
                for c in 0..3 {
                    let t = dd * d[c];
                    // d A_ij / d x_i = t, d A_ij / d x_j = -t
                    r[(i, 3 * i + c)] += t * q[j];
                    r[(i, 3 * j + c)] -= t * q[j];
                    r[(j, 3 * j + c)] -= t * q[i];
                    r[(j, 3 * i + c)] += t * q[i];
                }
            }
        }
        // -d chi_a/dx = -cnf_a/(2 sqrt(cn_a) + 1e-16) dCN_a/dx
        let at = &topo.numbers;
        let f: Vec<f64> = (0..n).map(|a| data::CNF[at[a] - 1] / (2.0e0 * st.lcn.cn[a].sqrt() + 1.0e-16)).collect();
        for pr in &st.lcn.pairs {
            let (i, j) = (pr.i as usize, pr.j as usize);
            for c in 0..3 {
                let vi = st.lcn.dlog[i] * pr.d[c];
                let vj = st.lcn.dlog[j] * pr.d[c];
                r[(i, 3 * j + c)] -= f[i] * vi;
                r[(i, 3 * i + c)] += f[i] * vi;
                r[(j, 3 * j + c)] -= f[j] * vj;
                r[(j, 3 * i + c)] += f[j] * vj;
            }
        }
        r
    }
}

/// Analytic Hessian at `xyz` (bohr).
pub fn hessian(topo: &Topology, model: &Model, xyz: &[[f64; 3]], threads: &Threads) -> Result<Hessian> {
    let n = topo.natoms();
    let n3 = 3 * n;
    let mut timings = Vec::new();
    let t0 = std::time::Instant::now();
    let (ev, st) = energy::evaluate_full(topo, model, xyz, true, true, threads)?;
    let st = st.unwrap();
    timings.push(("energy+gradient", t0.elapsed().as_secs_f64()));
    let t1 = std::time::Instant::now();
    let ctx = Ctx::new(topo, model, xyz, &st);
    // columns written straight into one column-major buffer (column col = chunk col)
    let mut hbuf = vec![0.0f64; n3 * n3];
    let fill = |col: usize, chunk: &mut [f64]| chunk.copy_from_slice(&ctx.column(col / 3, col % 3));
    threads.install(|| for_slices(&mut hbuf, n3.max(1), threads.parallel(), fill));
    timings.push(("columns", t1.elapsed().as_secs_f64()));
    let t2 = std::time::Instant::now();
    // EEQ charge response: H -= R^T [M^-1 R]_q
    let mut dipd = vec![[0.0f64; 3]; n3];
    {
        let mut h = faer::MatMut::from_column_major_slice_mut(&mut hbuf, n3, n3);
        let r = ctx.eeq_r();
        let par = threads.par();
        let block = 384;
        let mut start = 0;
        while start < n3 {
            let w = block.min(n3 - start);
            let mut x = r.as_ref().subcols(start, w).to_owned();
            st.eeq.solve_in_place(x.as_mut());
            let rt = r.as_ref().subrows(0, n).transpose();
            let xq = x.as_ref().subrows(0, n);
            faer::linalg::matmul::matmul(h.as_mut().subcols_mut(start, w), faer::Accum::Add, rt, xq, -1.0, par);
            // dipole derivatives: d(sum_i q_i x_i)/dx_col with dq/dx = -[M^-1 R]_q
            for m in 0..w {
                let col = start + m;
                let (k, c) = (col / 3, col % 3);
                let mut d = [0.0; 3];
                d[c] = st.charges[k];
                for i in 0..n {
                    let dq = -x[(i, m)];
                    for e in 0..3 {
                        d[e] += xyz[i][e] * dq;
                    }
                }
                dipd[col] = d;
            }
            start += w;
        }
    }
    timings.push(("eeq response", t2.elapsed().as_secs_f64()));
    // symmetrize in place (the buffer is then the same in row- and column-major order)
    let maxasym = fd::symmetrize(&mut hbuf, n3);
    if hbuf.iter().any(|v| !v.is_finite()) {
        return Err(not_finite(&ctx, &hbuf));
    }
    Ok(Hessian {
        energy: ev.energies.total,
        gradient: ev.gradient.unwrap(),
        hessian: hbuf,
        max_asymmetry: maxasym,
        dipole_derivatives: dipd,
        timings,
    })
}

/// The error for an analytic Hessian `h` with non-finite entries: the atoms of those rows
/// and, if there is one, the hydrogen bond along a C=O (or N=O) axis that causes them: the
/// carbonyl HB term's R-C=O...H torsions are undefined there, where the energy jumps
/// (grimme-lab/xtb#1516), and the numerical Hessian is defined.
fn not_finite(ctx: &Ctx, h: &[f64]) -> Error {
    let n3 = 3 * ctx.topo.natoms();
    let atoms: Vec<String> = (0..n3 / 3)
        .filter(|&a| h[3 * a * n3..3 * (a + 1) * n3].iter().any(|v| !v.is_finite()))
        .map(|a| (a + 1).to_string())
        .collect();
    let (xyz, at) = (ctx.xyz, &ctx.topo.numbers);
    let linear: Vec<String> = ctx
        .hb2
        .iter()
        .zip(&ctx.st.lists.hb2)
        .filter_map(|((kind, _), &[_, b, h])| {
            let Hb2Kind::Carbonyl { c, .. } = kind else { return None };
            let (co, oh) = (crate::real::sub(xyz[b], xyz[*c]), crate::real::sub(xyz[h], xyz[b]));
            // sine of the angle between C=O and O...H
            let sin =
                crate::real::norm2(crate::real::cross(co, oh)) / (crate::real::norm2(co) * crate::real::norm2(oh));
            let x = if at[*c] == 7 { "N" } else { "C" };
            (sin < 1e-6).then(|| format!("{x}{}=O{}...H{}", c + 1, b + 1, h + 1))
        })
        .collect();
    let atoms = atoms.join(", ");
    Error::Numerical(if linear.is_empty() {
        format!("the analytic Hessian is not finite (rows of atoms {atoms}); try the numerical Hessian")
    } else {
        format!(
            "the analytic Hessian is not finite (rows of atoms {atoms}): the hydrogen bond {} lies on the C=O axis, \
             where GFN-FF's R-C=O...H torsions are undefined (grimme-lab/xtb#1516); use the numerical Hessian, \
             which is defined there",
            linear.join(", ")
        )
    })
}

/// Hessian by central differences (`step` bohr) of the analytic gradient with the fixed
/// topology, as `xtb --hess` (step 0.005) computes it: symmetrized, with the dipole
/// derivatives from the EEQ dipoles of the displaced structures. 6N gradient evaluations
/// ([`fd::numerical_hessian`], each coordinate moved to x + h and x - h), parallel over the
/// displacements, each column written straight into the result.
pub fn numerical_hessian(
    topo: &Topology,
    model: &Model,
    xyz: &[[f64; 3]],
    step: f64,
    threads: &Threads,
) -> Result<Hessian> {
    let t0 = std::time::Instant::now();
    let reference = energy::evaluate(topo, model, xyz, true, threads)?;
    let mut timings = vec![("energy+gradient", t0.elapsed().as_secs_f64())];
    let t1 = std::time::Instant::now();
    // gradient and EEQ dipole (about the origin) of a displaced structure
    let point = |x: &[[f64; 3]]| -> Result<(Vec<[f64; 3]>, [f64; 3])> {
        let e = energy::evaluate(topo, model, x, true, &Threads::sequential())?;
        let mut d = [0.0; 3];
        for (xi, qi) in x.iter().zip(&e.charges) {
            for k in 0..3 {
                d[k] += xi[k] * qi;
            }
        }
        Ok((e.gradient.unwrap(), d))
    };
    let parallel = threads.parallel();
    let nh = threads.install(|| fd::numerical_hessian(xyz, step, fd::Displacement::FromReference, parallel, point))?;
    timings.push(("displacements", t1.elapsed().as_secs_f64()));
    Ok(Hessian {
        energy: reference.energies.total,
        gradient: reference.gradient.unwrap(),
        hessian: nh.hessian,
        max_asymmetry: nh.max_asymmetry,
        dipole_derivatives: nh.dipole_derivatives,
        timings,
    })
}
