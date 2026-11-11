//! Error-function coordination number with logarithmic cap (`gfnff_dlogcoord`).
//!
//! The counting function ([`XtbErf::GFNFF`]: xtb's erf count with kn = −7.5 and its derivative
//! with xtb's truncated √π), the logarithmic cap ([`log_cap`], [`dlog_cap`] at `cnmax`) and the
//! pair test (r > √cnthr skipped, tested on the distance) are `axtb-base`'s
//! (`axtb_base::ncoord`). xtb stores the derivative as a dense `dcn(3,n,n)`; here it is kept as
//! the list of contributing pairs, found through the cell list, which gives the same chain-rule
//! products.

use crate::cells::CellList;
use crate::param::Param;
use axtb_base::ncoord::{dlog_cap, log_cap, CnModel, Cutoff, XtbErf};

#[derive(Clone, Copy, Debug)]
pub struct CnPair {
    /// outer atom (larger index)
    pub i: u32,
    /// inner atom
    pub j: u32,
    /// `derivative * (x_j - x_i) / r`: d(erf count)/d x_j
    pub d: [f64; 3],
}

#[derive(Clone, Debug, Default)]
pub struct LogCn {
    /// capped logarithmic CN
    pub cn: Vec<f64>,
    /// raw error-function CN
    pub cn_raw: Vec<f64>,
    /// dlogCN/dCN_raw
    pub dlog: Vec<f64>,
    /// pairs within the cutoff, ordered (i ascending, j ascending), i > j
    pub pairs: Vec<CnPair>,
}

impl LogCn {
    /// CN of all atoms. `thr2` is the squared cutoff (`cnthr`); `numbers` are atomic numbers.
    pub fn new(numbers: &[usize], positions: &[[f64; 3]], thr2: f64, param: &Param, cells: Option<&CellList>) -> Self {
        let n = numbers.len();
        let thr = thr2.sqrt();
        let model =
            CnModel::new(XtbErf::GFNFF, |i: usize, j: usize| param.rcov[numbers[i] - 1] + param.rcov[numbers[j] - 1])
                .with_cutoff(Cutoff::Distance(thr));
        let mut pairs = Vec::new();
        // raw CN; the candidates are ascending, so every atom sums its partners in ascending
        // order, as xtb's loop does
        let mut cn_raw = vec![0.0; n];
        let mut cand = Vec::new();
        for i in 1..n {
            let xi = positions[i];
            match cells {
                Some(c) => c.candidates(i, thr * (1.0 + 1e-12) + 1e-9, |j| j < i, &mut cand),
                None => {
                    cand.clear();
                    cand.extend(0..i);
                }
            }
            for &j in cand.iter() {
                let Some((_, r2)) = model.pair(positions, i, j) else {
                    continue;
                };
                let r = r2.sqrt();
                let e = model.pair_count(i, j, r);
                let der = model.pair_dcount(i, j, r);
                let xj = positions[j];
                let v = [der * (xj[0] - xi[0]) / r, der * (xj[1] - xi[1]) / r, der * (xj[2] - xi[2]) / r];
                pairs.push(CnPair { i: i as u32, j: j as u32, d: v });
                cn_raw[i] += e;
                cn_raw[j] += e;
            }
        }
        let cn = cn_raw.iter().map(|&c| log_cap(c, param.cnmax)).collect();
        let dlog = cn_raw.iter().map(|&c| dlog_cap(c, param.cnmax)).collect();
        LogCn { cn, cn_raw, dlog, pairs }
    }

    /// `g += alpha * sum_a w_a dCN_a/dx` (the `mctc_gemv(dcn, w, g, alpha)` of xtb).
    pub fn chain(&self, w: &[f64], alpha: f64, g: &mut [[f64; 3]]) {
        for p in &self.pairs {
            let (i, j) = (p.i as usize, p.j as usize);
            let s = alpha * (self.dlog[i] * w[i] + self.dlog[j] * w[j]);
            for c in 0..3 {
                let v = s * p.d[c];
                g[j][c] += v;
                g[i][c] -= v;
            }
        }
    }

    /// Metallic character estimate of the setup: `exp(-0.005 EN^8) * sum_j |dCN_i/dx_j| / (CN_i + 1)`.
    pub fn metal_char(&self, numbers: &[usize]) -> Vec<f64> {
        let n = self.cn.len();
        let mut by_atom: Vec<Vec<(u32, usize)>> = vec![Vec::new(); n];
        for (k, p) in self.pairs.iter().enumerate() {
            by_atom[p.i as usize].push((p.j, k));
            by_atom[p.j as usize].push((p.i, k));
        }
        let mut out = vec![0.0; n];
        for a in 0..n {
            let list = &mut by_atom[a];
            list.sort_unstable();
            let dl = self.dlog[a];
            // self derivative dcn(:,a,a), accumulated in ascending partner order
            let mut selfd = [0.0f64; 3];
            for &(_, k) in list.iter() {
                let p = &self.pairs[k];
                for c in 0..3 {
                    // pair (i outer, j inner), d = d(erf)/dx_j
                    if p.i as usize == a {
                        selfd[c] += -(dl * p.d[c]);
                    } else {
                        selfd[c] += dl * p.d[c];
                    }
                }
            }
            // sum over j = 1..n of |dcn(:,j,a)| (zero outside the cutoff), self term in place
            let mut dum2 = 0.0;
            let selfn = (selfd[0] * selfd[0] + selfd[1] * selfd[1] + selfd[2] * selfd[2]).sqrt();
            let mut self_done = false;
            for &(b, k) in list.iter() {
                if !self_done && b as usize > a {
                    dum2 += selfn;
                    self_done = true;
                }
                let p = &self.pairs[k];
                let v = if p.i as usize == a {
                    [dl * p.d[0], dl * p.d[1], dl * p.d[2]]
                } else {
                    [-(dl * p.d[0]), -(dl * p.d[1]), -(dl * p.d[2])]
                };
                dum2 += (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
            }
            if !self_done {
                dum2 += selfn;
            }
            let en = crate::param::en(numbers[a]);
            out[a] = (-0.005e0 * en.powi(8)).exp() * dum2 / (self.cn[a] + 1.0e0);
        }
        out
    }
}
