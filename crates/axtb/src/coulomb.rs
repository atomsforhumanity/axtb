//! Electrostatics of the GFN-xTB methods: isotropic second order (shell-resolved γ
//! kernel), on-site third order, and damped anisotropic multipole electrostatics (GFN2).

use crate::molecule::{dot3, norm2_sq, sub3, Molecule};
use crate::ncoord::Ncoord;

/// Isotropic second-order electrostatics with the generalized Mataga–Nishimoto–Ohno–Klopman
/// kernel `γ = 1 / (r^g + η^-g)^(1/g)`, shell resolved.
#[derive(Clone, Debug)]
pub struct EffectiveCoulomb {
    gexp: f64,
    /// Averaged hardness `eta[isp][jsp][ish][jsh]`.
    eta: Vec<Vec<Vec<Vec<f64>>>>,
}

impl EffectiveCoulomb {
    /// `hardness[isp][ish]` are the shell hardnesses of each species; a shell pair has their
    /// arithmetic mean (GFN2-xTB).
    pub fn new(hardness: &[Vec<f64>], gexp: f64) -> Self {
        let eta = hardness
            .iter()
            .map(|hi| {
                hardness
                    .iter()
                    .map(|hj| hi.iter().map(|&a| hj.iter().map(|&b| 0.5 * (a + b)).collect()).collect())
                    .collect()
            })
            .collect();
        Self { gexp, eta }
    }

    /// Dense γ matrix over all shells (row-major `nsh x nsh`).
    pub fn matrix(&self, mol: &Molecule, ish_at: &[usize], nsh_at: &[usize], nsh: usize) -> Vec<f64> {
        let mut a = vec![0.0; nsh * nsh];
        let g = self.gexp;
        for iat in 0..mol.nat() {
            let isp = mol.id[iat];
            let ii = ish_at[iat];
            for jat in 0..=iat {
                let jsp = mol.id[jat];
                let jj = ish_at[jat];
                let r1 = norm2_sq(sub3(mol.positions[jat], mol.positions[iat])).sqrt();
                let r1g = r1.powf(g);
                for ish in 0..nsh_at[iat] {
                    let jmax = if iat == jat { ish + 1 } else { nsh_at[jat] };
                    for jsh in 0..jmax {
                        let eta = self.eta[isp][jsp][ish][jsh];
                        let val = if iat == jat { eta } else { 1.0 / (r1g + eta.powf(-g)).powf(1.0 / g) };
                        a[(ii + ish) * nsh + jj + jsh] = val;
                        a[(jj + jsh) * nsh + ii + ish] = val;
                    }
                }
            }
        }
        a
    }

    /// Add `sum_{A<B} q q dγ/dR` to the gradient.
    pub fn add_gradient(&self, mol: &Molecule, ish_at: &[usize], nsh_at: &[usize], qsh: &[f64], grad: &mut [[f64; 3]]) {
        let g = self.gexp;
        for iat in 0..mol.nat() {
            let isp = mol.id[iat];
            let ii = ish_at[iat];
            for jat in 0..iat {
                let jsp = mol.id[jat];
                let jj = ish_at[jat];
                let vec = sub3(mol.positions[iat], mol.positions[jat]);
                let r1 = norm2_sq(vec).sqrt();
                let mut dg = 0.0;
                for ish in 0..nsh_at[iat] {
                    for jsh in 0..nsh_at[jat] {
                        let eta = self.eta[isp][jsp][ish][jsh];
                        let mut dtmp = 1.0 / (r1.powf(g) + eta.powf(-g));
                        dtmp = -r1.powf(g - 2.0) * dtmp * dtmp.powf(1.0 / g);
                        dg += dtmp * qsh[ii + ish] * qsh[jj + jsh];
                    }
                }
                for k in 0..3 {
                    grad[iat][k] += dg * vec[k];
                    grad[jat][k] -= dg * vec[k];
                }
            }
        }
    }
}

impl EffectiveCoulomb {
    /// `dγ/dr / r` for a shell pair at distance `r1` (`dγ/dR_i = value * (R_i - R_j)`).
    #[inline]
    fn dgamma_r(&self, r1: f64, eta: f64) -> f64 {
        let g = self.gexp;
        let dtmp = 1.0 / (r1.powf(g) + eta.powf(-g));
        -r1.powf(g - 2.0) * dtmp * dtmp.powf(1.0 / g)
    }

    /// `d²γ/dr²` and `dγ/dr` for a shell pair.
    #[inline]
    fn d2gamma(&self, r1: f64, eta: f64) -> (f64, f64) {
        // γ = (r^g + η^-g)^(-1/g); u = r^g + η^-g
        let g = self.gexp;
        let u = r1.powf(g) + eta.powf(-g);
        let du = g * r1.powf(g - 1.0);
        let d2u = g * (g - 1.0) * r1.powf(g - 2.0);
        let d1 = -u.powf(-1.0 / g - 1.0) / g * du;
        let d2 = (1.0 / g) * (1.0 / g + 1.0) * u.powf(-1.0 / g - 2.0) * du * du - u.powf(-1.0 / g - 1.0) / g * d2u;
        (d2, d1)
    }

    /// Derivative of the shell potentials `V_a = sum_b γ_ab q_b` with respect to the nuclear
    /// positions at fixed charges, added to `dv` (row-major `nsh x 3 nat`).
    pub fn potential_derivative(
        &self,
        mol: &Molecule,
        ish_at: &[usize],
        nsh_at: &[usize],
        qsh: &[f64],
        dv: &mut [f64],
    ) {
        let n3 = 3 * mol.nat();
        for iat in 0..mol.nat() {
            let isp = mol.id[iat];
            let ii = ish_at[iat];
            for jat in 0..iat {
                let jsp = mol.id[jat];
                let jj = ish_at[jat];
                let vec = sub3(mol.positions[iat], mol.positions[jat]);
                let r1 = norm2_sq(vec).sqrt();
                for ish in 0..nsh_at[iat] {
                    for jsh in 0..nsh_at[jat] {
                        let dg = self.dgamma_r(r1, self.eta[isp][jsp][ish][jsh]);
                        let (a, b) = (ii + ish, jj + jsh);
                        for k in 0..3 {
                            let d = dg * vec[k];
                            dv[a * n3 + 3 * iat + k] += d * qsh[b];
                            dv[a * n3 + 3 * jat + k] -= d * qsh[b];
                            dv[b * n3 + 3 * iat + k] += d * qsh[a];
                            dv[b * n3 + 3 * jat + k] -= d * qsh[a];
                        }
                    }
                }
            }
        }
    }

    /// Second derivatives `sum_{A<B} q q d²γ/dR dR` at fixed charges, added to `h`
    /// (row-major `3 nat x 3 nat`).
    pub fn add_hessian(&self, mol: &Molecule, ish_at: &[usize], nsh_at: &[usize], qsh: &[f64], h: &mut [f64]) {
        let n3 = 3 * mol.nat();
        for iat in 0..mol.nat() {
            let isp = mol.id[iat];
            let ii = ish_at[iat];
            for jat in 0..iat {
                let jsp = mol.id[jat];
                let jj = ish_at[jat];
                let vec = sub3(mol.positions[iat], mol.positions[jat]);
                let r1 = norm2_sq(vec).sqrt();
                let (mut e2, mut e1) = (0.0, 0.0);
                for ish in 0..nsh_at[iat] {
                    for jsh in 0..nsh_at[jat] {
                        let (d2, d1) = self.d2gamma(r1, self.eta[isp][jsp][ish][jsh]);
                        let qq = qsh[ii + ish] * qsh[jj + jsh];
                        e2 += d2 * qq;
                        e1 += d1 * qq;
                    }
                }
                add_radial_hessian(h, n3, iat, jat, vec, r1, e1, e2);
            }
        }
    }
}

/// Add the Cartesian Hessian of a pair term `E(r)` with `dE/dr = e1`, `d²E/dr² = e2`
/// (`vec = R_i - R_j`, `r1 = |vec|`) to `h` (row-major `n3 x n3`).
#[inline]
pub(crate) fn add_radial_hessian(
    h: &mut [f64],
    n3: usize,
    iat: usize,
    jat: usize,
    vec: [f64; 3],
    r1: f64,
    e1: f64,
    e2: f64,
) {
    let u = [vec[0] / r1, vec[1] / r1, vec[2] / r1];
    for a in 0..3 {
        for b in 0..3 {
            let d = if a == b { 1.0 } else { 0.0 };
            let blk = e2 * u[a] * u[b] + e1 / r1 * (d - u[a] * u[b]);
            h[(3 * iat + a) * n3 + 3 * iat + b] += blk;
            h[(3 * jat + a) * n3 + 3 * jat + b] += blk;
            h[(3 * iat + a) * n3 + 3 * jat + b] -= blk;
            h[(3 * jat + a) * n3 + 3 * iat + b] -= blk;
        }
    }
}

/// On-site third-order term `E = sum_sh Γ_sh q_sh^3 / 3`.
#[derive(Clone, Debug)]
pub struct OnsiteThirdOrder {
    /// Hubbard derivative per species and shell.
    pub gamma: Vec<Vec<f64>>,
}

impl OnsiteThirdOrder {
    pub fn energy(&self, mol: &Molecule, ish_at: &[usize], qsh: &[f64], energies: &mut [f64]) {
        for iat in 0..mol.nat() {
            let ii = ish_at[iat];
            for (ish, &gam) in self.gamma[mol.id[iat]].iter().enumerate() {
                energies[iat] += qsh[ii + ish].powi(3) * gam / 3.0;
            }
        }
    }

    pub fn potential(&self, mol: &Molecule, ish_at: &[usize], qsh: &[f64], vsh: &mut [f64]) {
        for iat in 0..mol.nat() {
            let ii = ish_at[iat];
            for (ish, &gam) in self.gamma[mol.id[iat]].iter().enumerate() {
                vsh[ii + ish] += qsh[ii + ish].powi(2) * gam;
            }
        }
    }

    /// Diagonal second derivatives `2 Γ q` with respect to the shell charges, added to `kdiag`.
    pub fn kernel_diagonal(&self, mol: &Molecule, ish_at: &[usize], qsh: &[f64], kdiag: &mut [f64]) {
        for iat in 0..mol.nat() {
            let ii = ish_at[iat];
            for (ish, &gam) in self.gamma[mol.id[iat]].iter().enumerate() {
                kdiag[ii + ish] += 2.0 * qsh[ii + ish] * gam;
            }
        }
    }
}

/// Damped anisotropic multipole electrostatics of GFN2-xTB: charge–dipole, dipole–dipole
/// and charge–quadrupole interactions plus on-site multipole kernels.
#[derive(Clone, Debug)]
pub struct DampedMultipole {
    pub kdmp3: f64,
    pub kdmp5: f64,
    /// On-site dipole and quadrupole kernels per species.
    pub dkernel: Vec<f64>,
    pub qkernel: Vec<f64>,
    pub shift: f64,
    pub kexp: f64,
    pub rmax: f64,
    /// Base damping radius and valence CN per species.
    pub rad: Vec<f64>,
    pub vcn: Vec<f64>,
    pub ncoord: Ncoord,
}

/// Geometry-dependent data of the multipole term.
pub struct MultipoleCache {
    /// Damping radius and the negative of its CN derivative per atom (tblite's sign
    /// convention, matched by the gradient code).
    pub mrad: Vec<f64>,
    pub dmrdcn: Vec<f64>,
    /// Per unordered pair (i > j, packed i*(i-1)/2 + j): f3/r^3 and f5/r^5.
    g3: Vec<f64>,
    g5: Vec<f64>,
}

#[inline]
fn pair_index(i: usize, j: usize) -> usize {
    i * (i - 1) / 2 + j
}

/// Off-diagonal weight of packed symmetric quadrupoles (xy, xz, yz count twice).
const QSCALE: [f64; 6] = [1.0, 2.0, 1.0, 2.0, 2.0, 1.0];

impl DampedMultipole {
    /// On-site kernels, damping radii and valence CNs per species, and the coordination
    /// number of the damping radii.
    pub fn new(dkernel: Vec<f64>, qkernel: Vec<f64>, rad: Vec<f64>, vcn: Vec<f64>, ncoord: Ncoord) -> Self {
        Self { kdmp3: 3.0, kdmp5: 4.0, dkernel, qkernel, shift: 1.2, kexp: 4.0, rmax: 5.0, rad, vcn, ncoord }
    }

    pub fn update(&self, mol: &Molecule, cn: &[f64]) -> MultipoleCache {
        let nat = mol.nat();
        let mut mrad = vec![0.0; nat];
        let mut dmrdcn = vec![0.0; nat];
        for i in 0..nat {
            let isp = mol.id[i];
            let arg = cn[i] - self.vcn[isp] - self.shift;
            let t1 = (-self.kexp * arg).exp();
            let t2 = (self.rmax - self.rad[isp]) / (1.0 + t1);
            mrad[i] = self.rad[isp] + t2;
            dmrdcn[i] = -t2 * self.kexp * t1 / (1.0 + t1);
        }
        let npair = nat * nat.saturating_sub(1) / 2;
        let mut g3 = vec![0.0; npair];
        let mut g5 = vec![0.0; npair];
        for i in 1..nat {
            for j in 0..i {
                let r1 = norm2_sq(sub3(mol.positions[i], mol.positions[j])).sqrt();
                let g1 = 1.0 / r1;
                let g3v = g1 * g1 * g1;
                let g5v = g3v * g1 * g1;
                let rr = 0.5 * (mrad[j] + mrad[i]) * g1;
                let fdmp3 = 1.0 / (1.0 + 6.0 * rr.powf(self.kdmp3));
                let fdmp5 = 1.0 / (1.0 + 6.0 * rr.powf(self.kdmp5));
                g3[pair_index(i, j)] = g3v * fdmp3;
                g5[pair_index(i, j)] = g5v * fdmp5;
            }
        }
        MultipoleCache { mrad, dmrdcn, g3, g5 }
    }

    /// Potentials from the multipole term: `vat += dE/dq`, `vdp += dE/dμ`, `vqp += dE/dΘ`.
    /// Every atom adds its pair terms in the order of the other atom. The atoms are split into
    /// blocks: the pairs inside a block are computed once for both atoms (in the order of the
    /// pair loop i > j, which is that order), the pairs with atoms outside it for the block's
    /// atom alone. One block on one thread (each pair once); with `parallel`, blocks on the
    /// current rayon pool. The terms of the second atom of a pair are those of the first with
    /// R_a - R_b negated, which is exact, so the blocks change no bit: the same result for any
    /// thread count.
    pub fn potential(
        &self,
        mol: &Molecule,
        cache: &MultipoleCache,
        qat: &[f64],
        dpat: &[[f64; 3]],
        qpat: &[[f64; 6]],
        vat: &mut [f64],
        vdp: &mut [[f64; 3]],
        vqp: &mut [[f64; 6]],
        parallel: bool,
    ) {
        let nat = mol.nat();
        let pos = &mol.positions;
        type Acc = (f64, [f64; 3], [f64; 6]);
        // the geometry of the pair (a, b): w = R_a - R_b, f3/r^3, f5/r^5, |w|^2 and the packed
        // quadrupole weights QSCALE_k (w w)_k f5/r^5, the same for (b, a)
        let geometry = |a: usize, b: usize| {
            let p = pair_index(a.max(b), a.min(b));
            let w = sub3(pos[a], pos[b]);
            let g5 = cache.g5[p];
            let tc = [w[0] * w[0], w[0] * w[1], w[1] * w[1], w[0] * w[2], w[1] * w[2], w[2] * w[2]];
            (w, cache.g3[p], g5, norm2_sq(w), std::array::from_fn::<f64, 6, _>(|k| QSCALE[k] * tc[k] * g5))
        };
        // the terms of atom a from atom b, w = R_a - R_b
        let terms = |b: usize, w: [f64; 3], g3: f64, g5: f64, r2: f64, t: &[f64; 6], (va, vd, vq): &mut Acc| {
            // charge-dipole: E = q_a μ_b·w g3 - q_b μ_a·w g3
            let mb_w = dot3(dpat[b], w);
            *va += mb_w * g3;
            for k in 0..3 {
                vd[k] -= w[k] * g3 * qat[b];
            }
            // dipole-dipole: E = μ_a·(I r^2 - 3 w w)/r^5 f5 μ_b
            for k in 0..3 {
                vd[k] += g5 * (r2 * dpat[b][k] - 3.0 * w[k] * mb_w);
            }
            // charge-quadrupole: E = q_b Θ_a:ww g5 + q_a Θ_b:ww g5 (packed, off-diagonals x2)
            let mut thb = 0.0;
            for k in 0..6 {
                vq[k] += t[k] * qat[b];
                thb += t[k] * qpat[b][k];
            }
            *va += thb;
        };
        let one_sided = |a: usize, b: usize, acc: &mut Acc| {
            let (w, g3, g5, r2, t) = geometry(a, b);
            terms(b, w, g3, g5, r2, &t, acc);
        };
        // the atoms a0..a0 + acc.len(): the atoms before the block, the pairs inside it, the
        // atoms after it
        let block = |a0: usize, acc: &mut [Acc]| {
            let a1 = a0 + acc.len();
            for (a, v) in (a0..a1).zip(acc.iter_mut()) {
                let mut x = *v;
                (0..a0).for_each(|b| one_sided(a, b, &mut x));
                *v = x;
            }
            for i in a0 + 1..a1 {
                let (before, rest) = acc.split_at_mut(i - a0);
                let mut x = rest[0];
                for (j, y) in (a0..i).zip(before.iter_mut()) {
                    let (w, g3, g5, r2, t) = geometry(i, j);
                    terms(j, w, g3, g5, r2, &t, &mut x);
                    terms(i, [-w[0], -w[1], -w[2]], g3, g5, r2, &t, y);
                }
                rest[0] = x;
            }
            // b outermost: the block's pairs of a row are contiguous in the packed g3, g5
            for b in a1..nat {
                for (a, v) in (a0..a1).zip(acc.iter_mut()) {
                    one_sided(a, b, v);
                }
            }
        };
        let mut out: Vec<Acc> = (0..nat).map(|a| (vat[a], vdp[a], vqp[a])).collect();
        if parallel {
            use rayon::prelude::*;
            // about four blocks per thread (of at least 8 atoms), for an even load
            let size = nat.div_ceil(4 * rayon::current_num_threads()).max(8);
            out.par_chunks_mut(size).enumerate().for_each(|(c, acc)| block(c * size, acc));
        } else {
            block(0, &mut out);
        }
        for (i, v) in out.into_iter().enumerate() {
            (vat[i], vdp[i], vqp[i]) = v;
        }
        for i in 0..nat {
            let isp = mol.id[i];
            for k in 0..3 {
                vdp[i][k] += 2.0 * self.dkernel[isp] * dpat[i][k];
            }
            for k in 0..6 {
                vqp[i][k] += 2.0 * self.qkernel[isp] * qpat[i][k] * QSCALE[k];
            }
        }
    }

    /// Second derivatives of the multipole energy with respect to the moments in the atomic
    /// layout `[q (nat) | μ (3 nat) | Θ (6 nat, packed)]`, added to `k` (row-major,
    /// `10 nat` square). The energy is bilinear in the moments, so this is its full kernel.
    pub fn kernel(&self, mol: &Molecule, cache: &MultipoleCache, k: &mut [f64]) {
        let nat = mol.nat();
        let n = 10 * nat;
        let id = |i: usize, c: usize| nat + 3 * i + c;
        let iq = |i: usize, c: usize| 4 * nat + 6 * i + c;
        let mut add = |a: usize, b: usize, v: f64| {
            k[a * n + b] += v;
            k[b * n + a] += v;
        };
        for i in 1..nat {
            for j in 0..i {
                let p = pair_index(i, j);
                let (g3, g5) = (cache.g3[p], cache.g5[p]);
                let v = sub3(mol.positions[i], mol.positions[j]);
                let r2 = norm2_sq(v);
                for c in 0..3 {
                    add(i, id(j, c), v[c] * g3);
                    add(j, id(i, c), -v[c] * g3);
                    for d in 0..3 {
                        let delta = if c == d { r2 } else { 0.0 };
                        add(id(i, c), id(j, d), g5 * (delta - 3.0 * v[c] * v[d]));
                    }
                }
                let tc = [v[0] * v[0], v[0] * v[1], v[1] * v[1], v[0] * v[2], v[1] * v[2], v[2] * v[2]];
                for c in 0..6 {
                    let t = QSCALE[c] * tc[c] * g5;
                    add(j, iq(i, c), t);
                    add(i, iq(j, c), t);
                }
            }
        }
        for i in 0..nat {
            let isp = mol.id[i];
            for c in 0..3 {
                k[id(i, c) * n + id(i, c)] += 2.0 * self.dkernel[isp];
            }
            for c in 0..6 {
                k[iq(i, c) * n + iq(i, c)] += 2.0 * self.qkernel[isp] * QSCALE[c];
            }
        }
    }

    /// Derivatives of the potentials (`vat`, `vdp`, `vqp` in the layout of
    /// [`kernel`](Self::kernel)) with respect to the nuclear positions at fixed moments:
    /// the explicit dependence at fixed coordination numbers is added to `dv` (row-major
    /// `10 nat x 3 nat`), the dependence through the damping radii to `dvcn` (row-major
    /// `10 nat x nat`, derivative with respect to each atom's CN).
    pub fn potential_derivative(
        &self,
        mol: &Molecule,
        cache: &MultipoleCache,
        qat: &[f64],
        dpat: &[[f64; 3]],
        qpat: &[[f64; 6]],
        dv: &mut [f64],
        dvcn: &mut [f64],
    ) {
        let nat = mol.nat();
        let n3 = 3 * nat;
        let id = |i: usize, c: usize| nat + 3 * i + c;
        let iq = |i: usize, c: usize| 4 * nat + 6 * i + c;
        let (ka, kb) = (self.kdmp3, self.kdmp5);
        for i in 1..nat {
            for j in 0..i {
                let v = sub3(mol.positions[i], mol.positions[j]);
                let r2 = norm2_sq(v);
                let r = r2.sqrt();
                let rho = 0.5 * (cache.mrad[i] + cache.mrad[j]);
                let x = rho / r;
                let f3 = 1.0 / (1.0 + 6.0 * x.powf(ka));
                let f5 = 1.0 / (1.0 + 6.0 * x.powf(kb));
                let r4 = r2 * r2;
                let r6 = r4 * r2;
                let g3 = f3 / (r2 * r);
                let g5 = f5 / (r4 * r);
                let dg3dr = (6.0 * ka * x.powf(ka) * f3 * f3 - 3.0 * f3) / r4;
                let dg3drho = -6.0 * ka * x.powf(ka - 1.0) * f3 * f3 / r4;
                let dg5dr = (6.0 * kb * x.powf(kb) * f5 * f5 - 5.0 * f5) / r6;
                let dg5drho = -6.0 * kb * x.powf(kb - 1.0) * f5 * f5 / r6;
                let (qi, qj) = (qat[i], qat[j]);
                let (mi, mj) = (dpat[i], dpat[j]);
                let (ti, tj) = (qpat[i], qpat[j]);
                let mi_v = dot3(mi, v);
                let mj_v = dot3(mj, v);
                let tc = [v[0] * v[0], v[0] * v[1], v[1] * v[1], v[0] * v[2], v[1] * v[2], v[2] * v[2]];
                let vtv = |t: &[f64; 6]| (0..6).map(|c| QSCALE[c] * tc[c] * t[c]).sum::<f64>();
                let tv = |t: &[f64; 6]| {
                    [
                        t[0] * v[0] + t[1] * v[1] + t[3] * v[2],
                        t[1] * v[0] + t[2] * v[1] + t[4] * v[2],
                        t[3] * v[0] + t[4] * v[1] + t[5] * v[2],
                    ]
                };
                let (vtvi, vtvj) = (vtv(&ti), vtv(&tj));
                let (tvi, tvj) = (tv(&ti), tv(&tj));
                // dependence through the damping radius: the potential with g -> dg/dρ
                {
                    let (h3, h5) = (dg3drho, dg5drho);
                    let mut w = |row: usize, val: f64| {
                        // `dmrdcn` holds -dR/dCN (tblite's sign convention)
                        dvcn[row * nat + i] -= val * 0.5 * cache.dmrdcn[i];
                        dvcn[row * nat + j] -= val * 0.5 * cache.dmrdcn[j];
                    };
                    w(i, mj_v * h3 + vtvj * h5);
                    w(j, -mi_v * h3 + vtvi * h5);
                    for c in 0..3 {
                        w(id(i, c), -v[c] * h3 * qj + h5 * (r2 * mj[c] - 3.0 * v[c] * mj_v));
                        w(id(j, c), v[c] * h3 * qi + h5 * (r2 * mi[c] - 3.0 * v[c] * mi_v));
                    }
                    for c in 0..6 {
                        w(iq(i, c), QSCALE[c] * tc[c] * h5 * qj);
                        w(iq(j, c), QSCALE[c] * tc[c] * h5 * qi);
                    }
                }
                // explicit dependence on R_i (R_j: negative)
                for a in 0..3 {
                    let d3 = dg3dr * v[a] / r;
                    let d5 = dg5dr * v[a] / r;
                    let mut dtc = [0.0; 6];
                    match a {
                        0 => dtc = [2.0 * v[0], v[1], 0.0, v[2], 0.0, 0.0],
                        1 => dtc = [0.0, v[0], 2.0 * v[1], 0.0, v[2], 0.0],
                        _ => {
                            dtc[3] = v[0];
                            dtc[4] = v[1];
                            dtc[5] = 2.0 * v[2];
                        }
                    }
                    let mut w = |row: usize, val: f64| {
                        dv[row * n3 + 3 * i + a] += val;
                        dv[row * n3 + 3 * j + a] -= val;
                    };
                    w(i, mj[a] * g3 + mj_v * d3 + 2.0 * tvj[a] * g5 + vtvj * d5);
                    w(j, -mi[a] * g3 - mi_v * d3 + 2.0 * tvi[a] * g5 + vtvi * d5);
                    for c in 0..3 {
                        let dac = if a == c { 1.0 } else { 0.0 };
                        w(
                            id(i, c),
                            -dac * g3 * qj - v[c] * d3 * qj
                                + d5 * (r2 * mj[c] - 3.0 * v[c] * mj_v)
                                + g5 * (2.0 * v[a] * mj[c] - 3.0 * dac * mj_v - 3.0 * v[c] * mj[a]),
                        );
                        w(
                            id(j, c),
                            dac * g3 * qi
                                + v[c] * d3 * qi
                                + d5 * (r2 * mi[c] - 3.0 * v[c] * mi_v)
                                + g5 * (2.0 * v[a] * mi[c] - 3.0 * dac * mi_v - 3.0 * v[c] * mi[a]),
                        );
                    }
                    for c in 0..6 {
                        let t = QSCALE[c] * (dtc[c] * g5 + tc[c] * d5);
                        w(iq(i, c), t * qj);
                        w(iq(j, c), t * qi);
                    }
                }
            }
        }
    }

    /// Multipole energy per atom (pair terms split evenly, on-site kernels on the atom).
    pub fn energy(
        &self,
        mol: &Molecule,
        cache: &MultipoleCache,
        qat: &[f64],
        dpat: &[[f64; 3]],
        qpat: &[[f64; 6]],
        energies: &mut [f64],
        parallel: bool,
    ) {
        let nat = mol.nat();
        let pair = |i: usize, j: usize| -> f64 {
            let p = pair_index(i, j);
            let (g3, g5) = (cache.g3[p], cache.g5[p]);
            let v = sub3(mol.positions[i], mol.positions[j]);
            let r2 = norm2_sq(v);
            let mi_v = dot3(dpat[i], v);
            let mj_v = dot3(dpat[j], v);
            let mut e = (qat[i] * mj_v - qat[j] * mi_v) * g3;
            e += g5 * (r2 * dot3(dpat[i], dpat[j]) - 3.0 * mi_v * mj_v);
            let tc = [v[0] * v[0], v[0] * v[1], v[1] * v[1], v[0] * v[2], v[1] * v[2], v[2] * v[2]];
            for k in 0..6 {
                e += QSCALE[k] * tc[k] * g5 * (qat[j] * qpat[i][k] + qat[i] * qpat[j][k]);
            }
            e
        };
        if parallel {
            // the pair energies (packed like g3), then each atom's sum in the pair loop's order
            use rayon::prelude::*;
            let e: Vec<f64> = (1..nat).into_par_iter().flat_map_iter(|i| (0..i).map(move |j| pair(i, j))).collect();
            let sums: Vec<f64> = (0..nat)
                .into_par_iter()
                .with_min_len(8)
                .map(|i| {
                    let mut s = energies[i];
                    for j in 0..i {
                        s += 0.5 * e[pair_index(i, j)];
                    }
                    for r in i + 1..nat {
                        s += 0.5 * e[pair_index(r, i)];
                    }
                    s
                })
                .collect();
            energies[..nat].copy_from_slice(&sums);
        } else {
            for i in 1..nat {
                for j in 0..i {
                    let e = pair(i, j);
                    energies[i] += 0.5 * e;
                    energies[j] += 0.5 * e;
                }
            }
        }
        for i in 0..nat {
            let isp = mol.id[i];
            energies[i] += self.dkernel[isp] * dot3(dpat[i], dpat[i]);
            let mut qq = 0.0;
            for k in 0..6 {
                qq += QSCALE[k] * qpat[i][k] * qpat[i][k];
            }
            energies[i] += self.qkernel[isp] * qq;
        }
    }

    /// Damping radius and its first and second derivatives with respect to the CN.
    fn radius_derivs(&self, isp: usize, cn: f64) -> (f64, f64, f64) {
        let arg = cn - self.vcn[isp] - self.shift;
        let t1 = (-self.kexp * arg).exp();
        let a = self.rmax - self.rad[isp];
        let m = self.rad[isp] + a / (1.0 + t1);
        let d1 = a * self.kexp * t1 / ((1.0 + t1) * (1.0 + t1));
        let d2 = -a * self.kexp * self.kexp * t1 * (1.0 - t1) / ((1.0 + t1) * (1.0 + t1) * (1.0 + t1));
        (m, d1, d2)
    }

    /// Second derivatives of the multipole energy with respect to the nuclear positions at
    /// fixed moments, including the CN dependence of the damping radii (`cn` and its
    /// Jacobian `jac`, row-major `nat x 3 nat`, of `self.ncoord`). Added to `hess`
    /// (row-major `3 nat x 3 nat`).
    pub fn add_hessian(
        &self,
        mol: &Molecule,
        cache: &MultipoleCache,
        cn: &[f64],
        jac: &[f64],
        qat: &[f64],
        dpat: &[[f64; 3]],
        qpat: &[[f64; 6]],
        hess: &mut [f64],
    ) {
        let nat = mol.nat();
        let n3 = 3 * nat;
        let mut xr = vec![0.0; n3 * nat];
        let mut emm = vec![0.0; nat * nat];
        let mut em = vec![0.0; nat];
        // g = f(ρ/r) r^-n with f = 1/(1 + 6 x^a): value and derivatives in r and ρ
        let damp = |n: f64, a: f64, r: f64, rho: f64| -> [f64; 6] {
            let x = rho / r;
            let f = 1.0 / (1.0 + 6.0 * x.powf(a));
            let f1 = -6.0 * a * x.powf(a - 1.0) * f * f;
            let f2 = -6.0 * a * (a - 1.0) * x.powf(a - 2.0) * f * f + 72.0 * a * a * x.powf(2.0 * a - 2.0) * f * f * f;
            let rn = r.powf(-n);
            let (rn1, rn2) = (rn / r, rn / (r * r));
            [
                f * rn,
                -(x * f1 + n * f) * rn1,
                f1 * rn1,
                (2.0 * (n + 1.0) * x * f1 + x * x * f2 + n * (n + 1.0) * f) * rn2,
                -((n + 1.0) * f1 + x * f2) * rn2,
                f2 * rn2,
            ]
        };
        for i in 1..nat {
            for j in 0..i {
                let v = sub3(mol.positions[i], mol.positions[j]);
                let r2 = norm2_sq(v);
                let r = r2.sqrt();
                let rho = 0.5 * (cache.mrad[i] + cache.mrad[j]);
                // [g, g_r, g_ρ, g_rr, g_rρ, g_ρρ]
                let g3 = damp(3.0, self.kdmp3, r, rho);
                let g5 = damp(5.0, self.kdmp5, r, rho);
                let (qi, qj) = (qat[i], qat[j]);
                let (mi, mj) = (dpat[i], dpat[j]);
                let a = [qi * mj[0] - qj * mi[0], qi * mj[1] - qj * mi[1], qi * mj[2] - qj * mi[2]];
                let full = |t: &[f64; 6]| [[t[0], t[1], t[3]], [t[1], t[2], t[4]], [t[3], t[4], t[5]]];
                let (ti, tj) = (full(&qpat[i]), full(&qpat[j]));
                let mimj = dot3(mi, mj);
                let mut mb = [[0.0; 3]; 3];
                for k in 0..3 {
                    for l in 0..3 {
                        mb[k][l] = -1.5 * (mi[k] * mj[l] + mj[k] * mi[l]) + qj * ti[k][l] + qi * tj[k][l];
                    }
                    mb[k][k] += mimj;
                }
                let av = dot3(a, v);
                let mv = [dot3(mb[0], v), dot3(mb[1], v), dot3(mb[2], v)];
                let vmv = dot3(v, mv);
                let u = [v[0] / r, v[1] / r, v[2] / r];
                let dk = |g: &[f64; 6], k: usize| g[1] * u[k];
                let dkl = |g: &[f64; 6], k: usize, l: usize| {
                    let d = if k == l { 1.0 } else { 0.0 };
                    g[3] * u[k] * u[l] + g[1] * (d - u[k] * u[l]) / r
                };
                for k in 0..3 {
                    for l in 0..3 {
                        let hkl = a[k] * dk(&g3, l)
                            + a[l] * dk(&g3, k)
                            + av * dkl(&g3, k, l)
                            + 2.0 * mb[k][l] * g5[0]
                            + 2.0 * mv[k] * dk(&g5, l)
                            + 2.0 * mv[l] * dk(&g5, k)
                            + vmv * dkl(&g5, k, l);
                        hess[(3 * i + k) * n3 + 3 * i + l] += hkl;
                        hess[(3 * j + k) * n3 + 3 * j + l] += hkl;
                        hess[(3 * i + k) * n3 + 3 * j + l] -= hkl;
                        hess[(3 * j + k) * n3 + 3 * i + l] -= hkl;
                    }
                    let evr = a[k] * g3[2] + av * g3[4] * u[k] + 2.0 * mv[k] * g5[2] + vmv * g5[4] * u[k];
                    for c in [i, j] {
                        xr[(3 * i + k) * nat + c] += 0.5 * evr;
                        xr[(3 * j + k) * nat + c] -= 0.5 * evr;
                    }
                }
                let err = av * g3[5] + vmv * g5[5];
                let er = av * g3[2] + vmv * g5[2];
                emm[i * nat + i] += 0.25 * err;
                emm[j * nat + j] += 0.25 * err;
                emm[i * nat + j] += 0.25 * err;
                emm[j * nat + i] += 0.25 * err;
                em[i] += 0.5 * er;
                em[j] += 0.5 * er;
            }
        }
        // chain rule through the damping radii m(CN(R))
        let (mut d1, mut d2) = (vec![0.0; nat], vec![0.0; nat]);
        for c in 0..nat {
            let (_, a, b) = self.radius_derivs(mol.id[c], cn[c]);
            d1[c] = a;
            d2[c] = b;
        }
        // Y = X diag(m') + Jᵀ diag(m') Emm diag(m') / 2 ... assembled as XJ + (XJ)ᵀ + Jᵀ B J
        let mut bmat = vec![0.0; nat * nat];
        for c in 0..nat {
            for d in 0..nat {
                bmat[c * nat + d] = d1[c] * emm[c * nat + d] * d1[d];
            }
            bmat[c * nat + c] += em[c] * d2[c];
        }
        // BJ (nat x n3)
        let mut bj = vec![0.0; nat * n3];
        for c in 0..nat {
            for d in 0..nat {
                let w = bmat[c * nat + d];
                if w != 0.0 {
                    for y in 0..n3 {
                        bj[c * n3 + y] += w * jac[d * n3 + y];
                    }
                }
            }
        }
        let mut xj = vec![0.0; n3 * n3];
        for x in 0..n3 {
            for c in 0..nat {
                let w = xr[x * nat + c] * d1[c];
                if w != 0.0 {
                    for y in 0..n3 {
                        xj[x * n3 + y] += w * jac[c * n3 + y];
                    }
                }
            }
        }
        for x in 0..n3 {
            for y in 0..n3 {
                let mut jbj = 0.0;
                for c in 0..nat {
                    jbj += jac[c * n3 + x] * bj[c * n3 + y];
                }
                hess[x * n3 + y] += xj[x * n3 + y] + xj[y * n3 + x] + jbj;
            }
        }
        let w: Vec<f64> = (0..nat).map(|c| em[c] * d1[c]).collect();
        self.ncoord.add_hessian(mol, &w, hess);
    }

    /// Explicit nuclear gradient at fixed moments, including the CN dependence of the
    /// damping radii.
    pub fn add_gradient(
        &self,
        mol: &Molecule,
        cache: &MultipoleCache,
        qat: &[f64],
        dpat: &[[f64; 3]],
        qpat: &[[f64; 6]],
        grad: &mut [[f64; 3]],
    ) {
        let nat = mol.nat();
        let (kdmp3, kdmp5) = (self.kdmp3, self.kdmp5);
        let mut dedr = vec![0.0; nat];
        for iat in 1..nat {
            for jat in 0..iat {
                let rr = 0.5 * (cache.mrad[iat] + cache.mrad[jat]);
                let vec = sub3(mol.positions[jat], mol.positions[iat]);
                let r2 = norm2_sq(vec);
                let r1 = r2.sqrt();
                let g1 = 1.0 / r1;
                let g3 = g1 * g1 * g1;
                let g5 = g3 * g1 * g1;
                let g7 = g5 * g1 * g1;
                let fdmp3 = 1.0 / (1.0 + 6.0 * (rr * g1).powf(kdmp3));
                let ddmp3 = -3.0 * g5 * fdmp3 - kdmp3 * fdmp3 * (fdmp3 - 1.0) * g5;
                let fdmp5 = 1.0 / (1.0 + 6.0 * (rr * g1).powf(kdmp5));
                let ddmp5 = -5.0 * fdmp5 - kdmp5 * (fdmp5 * fdmp5 - fdmp5);
                let (qi, qj) = (qat[iat], qat[jat]);
                let (di, dj) = (dpat[iat], dpat[jat]);

                let dpiqj = dot3(vec, di) * qj;
                let qidpj = dot3(vec, dj) * qi;
                let mut fddr = 3.0 * (dpiqj - qidpj) * kdmp3 * fdmp3 * g3 * (fdmp3 / rr) * (rr * g1).powf(kdmp3);
                let mut dg = [0.0; 3];
                for k in 0..3 {
                    dg[k] = -ddmp3 * vec[k] * (dpiqj - qidpj) + fdmp3 * g3 * (qi * dj[k] - qj * di[k]);
                }

                let dpidpj = dot3(dj, di);
                let dpiv = dot3(di, vec);
                let dpjv = dot3(dj, vec);
                let edd = dpidpj * r2 - 3.0 * dpjv * dpiv;
                fddr += 3.0 * edd * kdmp5 * fdmp5 * g5 * (fdmp5 / rr) * (rr * g1).powf(kdmp5);
                for k in 0..3 {
                    dg[k] += -2.0 * fdmp5 * g5 * dpidpj * vec[k] + 3.0 * fdmp5 * g5 * (dpiv * dj[k] + dpjv * di[k])
                        - edd * ddmp5 * g7 * vec[k];
                }

                let (ti, tj) = (qpat[iat], qpat[jat]);
                let eq = 2.0 * (qj * ti[1] + tj[1] * qi) * vec[0] * vec[1]
                    + 2.0 * (qj * ti[3] + tj[3] * qi) * vec[0] * vec[2]
                    + 2.0 * (qj * ti[4] + tj[4] * qi) * vec[1] * vec[2]
                    + (qj * ti[0] + tj[0] * qi) * vec[0] * vec[0]
                    + (qj * ti[2] + tj[2] * qi) * vec[1] * vec[1]
                    + (qj * ti[5] + tj[5] * qi) * vec[2] * vec[2];
                fddr += eq * 3.0 * kdmp5 * fdmp5 * g5 * fdmp5 / rr * (rr * g1).powf(kdmp5);
                let tvj = [
                    vec[0] * tj[0] + vec[1] * tj[1] + vec[2] * tj[3],
                    vec[0] * tj[1] + vec[1] * tj[2] + vec[2] * tj[4],
                    vec[0] * tj[3] + vec[1] * tj[4] + vec[2] * tj[5],
                ];
                let tvi = [
                    vec[0] * ti[0] + vec[1] * ti[1] + vec[2] * ti[3],
                    vec[0] * ti[1] + vec[1] * ti[2] + vec[2] * ti[4],
                    vec[0] * ti[3] + vec[1] * ti[4] + vec[2] * ti[5],
                ];
                for k in 0..3 {
                    dg[k] +=
                        -eq * ddmp5 * g7 * vec[k] - 2.0 * fdmp5 * g5 * qi * tvj[k] - 2.0 * fdmp5 * g5 * qj * tvi[k];
                }

                dedr[iat] += fddr;
                dedr[jat] += fddr;
                for k in 0..3 {
                    grad[iat][k] += dg[k];
                    grad[jat][k] -= dg[k];
                }
            }
        }
        for i in 0..nat {
            dedr[i] *= cache.dmrdcn[i];
        }
        self.ncoord.add_gradient(mol, &dedr, grad);
    }
}
