// SPDX-License-Identifier: LGPL-3.0-or-later
//! Smoothed solvent accessible surface area (port of xtb `src/solv/sasa.f90`).
//!
//! The surface derivative d sasa_i / d R_j is stored sparsely: for atom i the
//! self term and one entry per SASA neighbour (same neighbour order as xtb).

/// Smoothing function constants (xtb `ah0`, `ah1`, `ah3`).
#[derive(Clone, Copy, Debug)]
pub(crate) struct Smooth {
    pub ah0: f64,
    pub ah1: f64,
    pub ah3: f64,
}

/// Sparse storage of the surface derivatives.
#[derive(Clone, Debug, Default)]
pub(crate) struct SurfaceDeriv {
    /// d sasa_i / d R_i
    pub self_grad: Vec<[f64; 3]>,
    /// CSR row pointer into `nb_idx` / `nb_grad` (len nat+1)
    pub nb_ptr: Vec<usize>,
    /// neighbour atom j
    pub nb_idx: Vec<usize>,
    /// d sasa_i / d R_j
    pub nb_grad: Vec<[f64; 3]>,
}

/// Build the SASA neighbour lists in xtb order: for each atom the neighbours
/// appear in ascending index order.
///
/// xtb uses all pairs with r < srcut = 2 (w + max R_sa) + 2 AA.  A neighbour j
/// can only affect the surface of atom i if r_ij < R_sa,i + R_sa,j + w (the
/// grid points of i lie on the sphere of radius R_sa,i, and j's smoothing
/// function is 1 beyond R_sa,j + w), so such pairs are dropped as well; the
/// margin keeps the result bitwise identical to the unfiltered list.
#[inline(never)]
pub(crate) fn neighbour_list(xyz: &[[f64; 3]], srcut: f64, vdwsa: &[f64], w: f64, deriv: &mut SurfaceDeriv) {
    let nat = xyz.len();
    let srcut2 = srcut * srcut;
    let near = |i: usize, j: usize| {
        let d2 = dist2(xyz, i, j);
        let rmax = vdwsa[i] + vdwsa[j] + w + 1.0e-6;
        d2 < srcut2 && d2 < rmax * rmax
    };
    let ptr = &mut deriv.nb_ptr;
    ptr.clear();
    ptr.resize(nat + 1, 0);
    // count
    for i in 0..nat {
        for j in 0..i {
            if near(i, j) {
                ptr[i + 1] += 1;
                ptr[j + 1] += 1;
            }
        }
    }
    for i in 0..nat {
        ptr[i + 1] += ptr[i];
    }
    let total = ptr[nat];
    deriv.nb_idx.clear();
    deriv.nb_idx.resize(total, 0);
    deriv.nb_grad.clear();
    deriv.nb_grad.resize(total, [0.0; 3]);
    deriv.self_grad.clear();
    deriv.self_grad.resize(nat, [0.0; 3]);
    // fill: ascending order per atom
    let mut fill: Vec<usize> = ptr[..nat].to_vec();
    for i in 0..nat {
        for j in 0..i {
            if near(i, j) {
                deriv.nb_idx[fill[i]] = j;
                fill[i] += 1;
                deriv.nb_idx[fill[j]] = i;
                fill[j] += 1;
            }
        }
    }
    // the second loop fills atom i's list with j < i first and later with
    // larger indices, i.e. ascending order, as in xtb's pair-list traversal.
}

#[inline(always)]
fn dist2(xyz: &[[f64; 3]], i: usize, j: usize) -> f64 {
    let x = xyz[i][0] - xyz[j][0];
    let y = xyz[i][1] - xyz[j][1];
    let z = xyz[i][2] - xyz[j][2];
    x * x + y * y + z * z
}

/// Scratch space for [`compute_numsa`].
#[derive(Clone, Debug, Default)]
pub(crate) struct SasaScratch {
    grdi: Vec<usize>,
    grds: Vec<[f64; 3]>,
    grads: Vec<[f64; 3]>,
    /// neighbour data of the current atom, structure of arrays
    nbx: Vec<f64>,
    nby: Vec<f64>,
    nbz: Vec<f64>,
    nbt1: Vec<f64>,
    nbt2: Vec<f64>,
    nbr: Vec<f64>,
    tj2: Vec<f64>,
}

/// Surface areas (xtb `compute_numsa`) and their derivatives.
///
/// Same arithmetic as xtb.  For speed, the squared distances of a grid point
/// to all neighbours are computed first (vectorisable) and a point buried by
/// any neighbour is skipped right away: in xtb such a point gets zero weight
/// whatever the neighbour order, so the result is unchanged.
#[inline(never)]
pub(crate) fn compute_numsa(
    xyz: &[[f64; 3]],
    vdwsa: &[f64],
    wrp: &[f64],
    trj2: &[[f64; 2]],
    sm: Smooth,
    tolsesp: f64,
    ang_grid: &[[f64; 3]],
    ang_weight: &[f64],
    sasa: &mut [f64],
    deriv: &mut SurfaceDeriv,
    scratch: &mut SasaScratch,
) {
    let nat = xyz.len();
    let maxnb = (0..nat).map(|i| deriv.nb_ptr[i + 1] - deriv.nb_ptr[i]).max().unwrap_or(0);
    let sc = scratch;
    sc.grdi.clear();
    sc.grdi.resize(maxnb, 0);
    sc.grds.clear();
    sc.grds.resize(maxnb, [0.0; 3]);
    sc.grads.clear();
    sc.grads.resize(maxnb, [0.0; 3]);
    for v in [&mut sc.nbx, &mut sc.nby, &mut sc.nbz, &mut sc.nbt1, &mut sc.nbt2, &mut sc.nbr, &mut sc.tj2] {
        v.clear();
        v.resize(maxnb, 0.0);
    }

    for iat in 0..nat {
        let (p0, p1) = (deriv.nb_ptr[iat], deriv.nb_ptr[iat + 1]);
        let nno = p1 - p0;
        for (slot, &ia) in deriv.nb_idx[p0..p1].iter().enumerate() {
            sc.nbx[slot] = xyz[ia][0];
            sc.nby[slot] = xyz[ia][1];
            sc.nbz[slot] = xyz[ia][2];
            sc.nbt1[slot] = trj2[ia][0];
            sc.nbt2[slot] = trj2[ia][1];
            sc.nbr[slot] = vdwsa[ia];
        }
        let (nbx, nby, nbz) = (&sc.nbx[..nno], &sc.nby[..nno], &sc.nbz[..nno]);
        let (nbt1, nbt2, nbr) = (&sc.nbt1[..nno], &sc.nbt2[..nno], &sc.nbr[..nno]);
        let tj2v = &mut sc.tj2[..nno];
        let rsas = vdwsa[iat];
        let wr = wrp[iat];
        let xyza = xyz[iat];
        // local gradient: position 0..nno for neighbours (by list slot)
        let grads = &mut sc.grads[..nno];
        grads.iter_mut().for_each(|g| *g = [0.0; 3]);
        let mut gself = [0.0f64; 3];
        let mut sasai = 0.0f64;
        for (ip, p) in ang_grid.iter().enumerate() {
            let xyzp = [xyza[0] + rsas * p[0], xyza[1] + rsas * p[1], xyza[2] + rsas * p[2]];
            let mut buried = false;
            for slot in 0..nno {
                let (dx, dy, dz) = (xyzp[0] - nbx[slot], xyzp[1] - nby[slot], xyzp[2] - nbz[slot]);
                let tj2 = dx * dx + dy * dy + dz * dz;
                tj2v[slot] = tj2;
                buried |= tj2 <= nbt1[slot];
            }
            if buried {
                continue;
            }
            // atomic surface function at the grid point (xtb compute_w_sp)
            let mut nni = 0usize;
            let mut sasap = 1.0f64;
            for slot in 0..nno {
                let tj2 = tj2v[slot];
                if tj2 < nbt2[slot] {
                    let tj = [xyzp[0] - nbx[slot], xyzp[1] - nby[slot], xyzp[2] - nbz[slot]];
                    let sqtj = tj2.sqrt();
                    let uj = sqtj - nbr[slot];
                    let ah3uj2 = sm.ah3 * uj * uj;
                    let mut dsasaij = sm.ah1 + 3.0 * ah3uj2;
                    let mut sasaij = sm.ah0 + (sm.ah1 + ah3uj2) * uj;
                    sasaij = fmin(sasaij, 1.0);
                    sasaij = fmax(0.0, sasaij);
                    if sasaij < f64::MIN_POSITIVE {
                        sasap = 0.0;
                        break;
                    }
                    sasap *= sasaij;
                    dsasaij /= sasaij * sqtj;
                    sc.grdi[nni] = slot;
                    sc.grds[nni] = [dsasaij * tj[0], dsasaij * tj[1], dsasaij * tj[2]];
                    nni += 1;
                }
            }
            if sasap > tolsesp {
                let wsa = ang_weight[ip] * wr * sasap;
                sasai += wsa;
                for jj in 0..nni {
                    let slot = sc.grdi[jj];
                    let g = sc.grds[jj];
                    let drjj = [wsa * g[0], wsa * g[1], wsa * g[2]];
                    for k in 0..3 {
                        gself[k] += drjj[k];
                        grads[slot][k] -= drjj[k];
                    }
                }
            }
        }
        sasa[iat] = sasai;
        deriv.self_grad[iat] = gself;
        deriv.nb_grad[p0..p1].copy_from_slice(grads);
    }
}

/// Fortran `min(a, b)` for the values occurring here (no NaN handling needed).
#[inline(always)]
fn fmin(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

#[inline(always)]
fn fmax(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

impl SurfaceDeriv {
    /// grad += sum_i c_i * d sasa_i / d R
    pub(crate) fn contract(&self, c: &[f64], grad: &mut [[f64; 3]]) {
        let nat = c.len();
        for i in 0..nat {
            let ci = c[i];
            if ci == 0.0 {
                continue;
            }
            let g = self.self_grad[i];
            for k in 0..3 {
                grad[i][k] += g[k] * ci;
            }
            for p in self.nb_ptr[i]..self.nb_ptr[i + 1] {
                let j = self.nb_idx[p];
                let g = self.nb_grad[p];
                for k in 0..3 {
                    grad[j][k] += g[k] * ci;
                }
            }
        }
    }
}

/// Straightforward port of xtb's `compute_numsa` (dense derivative, full
/// `srcut` neighbour list, no shortcuts) to check the optimised version.
#[cfg(test)]
pub(crate) fn compute_numsa_reference(
    xyz: &[[f64; 3]],
    srcut: f64,
    vdwsa: &[f64],
    wrp: &[f64],
    trj2: &[[f64; 2]],
    sm: Smooth,
    tolsesp: f64,
    ang_grid: &[[f64; 3]],
    ang_weight: &[f64],
) -> (Vec<f64>, Vec<Vec<[f64; 3]>>) {
    let nat = xyz.len();
    let mut nnlists: Vec<Vec<usize>> = vec![Vec::new(); nat];
    for i in 0..nat {
        for j in 0..i {
            if dist2(xyz, i, j) < srcut * srcut {
                nnlists[i].push(j);
                nnlists[j].push(i);
            }
        }
    }
    let mut sasa = vec![0.0; nat];
    let mut dsdrt = vec![vec![[0.0; 3]; nat]; nat]; // dsdrt[i][j] = d sasa_i / d R_j
    for iat in 0..nat {
        let mut grads = vec![[0.0f64; 3]; nat];
        let mut sasai = 0.0;
        for ip in 0..ang_grid.len() {
            let xyzp: [f64; 3] = std::array::from_fn(|k| xyz[iat][k] + vdwsa[iat] * ang_grid[ip][k]);
            let mut sasap = 1.0;
            let mut grd: Vec<(usize, [f64; 3])> = Vec::new();
            for &ia in &nnlists[iat] {
                let tj: [f64; 3] = std::array::from_fn(|k| xyzp[k] - xyz[ia][k]);
                let tj2 = tj[0] * tj[0] + tj[1] * tj[1] + tj[2] * tj[2];
                if tj2 < trj2[ia][1] {
                    if tj2 <= trj2[ia][0] {
                        sasap = 0.0;
                        break;
                    }
                    let sqtj = tj2.sqrt();
                    let uj = sqtj - vdwsa[ia];
                    let ah3uj2 = sm.ah3 * uj * uj;
                    let mut dsasaij = sm.ah1 + 3.0 * ah3uj2;
                    let mut sasaij = sm.ah0 + (sm.ah1 + ah3uj2) * uj;
                    sasaij = sasaij.min(1.0).max(0.0);
                    if sasaij < f64::MIN_POSITIVE {
                        sasap = 0.0;
                        break;
                    }
                    sasap *= sasaij;
                    dsasaij /= sasaij * sqtj;
                    grd.push((ia, [dsasaij * tj[0], dsasaij * tj[1], dsasaij * tj[2]]));
                }
            }
            if sasap > tolsesp {
                let wsa = ang_weight[ip] * wrp[iat] * sasap;
                sasai += wsa;
                for (j, g) in grd {
                    for k in 0..3 {
                        grads[iat][k] += wsa * g[k];
                        grads[j][k] -= wsa * g[k];
                    }
                }
            }
        }
        sasa[iat] = sasai;
        dsdrt[iat] = grads;
    }
    (sasa, dsdrt)
}
