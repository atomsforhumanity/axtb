// SPDX-License-Identifier: LGPL-3.0-or-later
//! AO-matrix assembly from shell-pair blocks: a visitor over the atom pairs and shell pairs of
//! a [`Basis`] with the integral screening ([`ShellPairs`]), and the overlap, dipole and
//! quadrupole matrices built on it ([`AoMatrices`]).
//!
//! Screening (written once from the GFN2-xTB core Hamiltonian build, tblite's conventions):
//! atom pairs farther apart than a real-space cutoff are skipped, and with exponent screening
//! so is every shell pair (and atom pair) whose smallest primitive exponents a, b give
//! a b R²/(a + b) above the primitive cutoff `intcut` ([`pair_screened`], grimme-lab/xtb#1465):
//! the integral kernels skip each of its primitive pairs, so its blocks and their derivatives
//! are exactly zero (all +0.0) and skipping them changes no result.
//!
//! The visitor leaves the loops over atoms, and so the parallel structure, to its caller: a
//! builder asks for the atom pairs it wants ([`ShellPairs::pair`]) and visits their shell
//! pairs ([`ShellPairs::for_each`]) with a closure (monomorphized; nothing is allocated).
//! [`AoMatrices::build`] evaluates each unordered atom pair once, around the ket atom, and
//! gets the mirrored block from [`shift_to_bra`].

use crate::basis::{nsph, Basis, Cgto};
use crate::integrals::{multipole, pair_screened, shift_to_bra, Scratch, MBLK};
use crate::molecule::{norm2_sq, sub3, Molecule};
use faer::Mat;
use rayon::prelude::*;

/// An atom pair of [`ShellPairs`]: ket atom `i`, bra atom `j`, `vec` = R_i - R_j (the `vec` of
/// the integral kernels) and `r2` = |vec|².
#[derive(Clone, Copy, Debug)]
pub struct AtomPair {
    pub i: usize,
    pub j: usize,
    pub vec: [f64; 3],
    pub r2: f64,
}

impl AtomPair {
    /// Atom `i` with itself (R = 0: never beyond a cutoff, never screened).
    pub const fn onsite(i: usize) -> Self {
        Self { i, j: i, vec: [0.0; 3], r2: 0.0 }
    }
}

/// A shell pair of [`ShellPairs::for_each`]: shell `ish` of atom i (ket, `ci`) and `jsh` of atom
/// j (bra, `cj`), their indices in the basis (`si`, `sj`) and their first AOs (`ii`, `jj`).
#[derive(Clone, Copy, Debug)]
pub struct ShellPair<'a> {
    pub ish: usize,
    pub jsh: usize,
    pub si: usize,
    pub sj: usize,
    pub ii: usize,
    pub jj: usize,
    pub ci: &'a Cgto,
    pub cj: &'a Cgto,
}

/// The atom pairs and shell pairs of a basis within a real-space cutoff, with or without the
/// exponent screening (module docs).
#[derive(Clone, Copy, Debug)]
pub struct ShellPairs<'a> {
    pos: &'a [[f64; 3]],
    bas: &'a Basis,
    cutoff2: f64,
    intcut: f64,
    screen: bool,
}

impl<'a> ShellPairs<'a> {
    /// Pairs of `bas` for `mol` within `cutoff` (bohr; infinite for none), skipping shell and
    /// atom pairs beyond the primitive cutoff `intcut` (the kernels' threshold, >= 0).
    pub fn screened(mol: &'a Molecule, bas: &'a Basis, cutoff: f64, intcut: f64) -> Self {
        Self { pos: &mol.positions, bas, cutoff2: cutoff * cutoff, intcut, screen: true }
    }

    /// Every shell pair of the atom pairs of `bas` for `mol` within `cutoff` (bohr; infinite for
    /// none).
    pub fn unscreened(mol: &'a Molecule, bas: &'a Basis, cutoff: f64) -> Self {
        Self { pos: &mol.positions, bas, cutoff2: cutoff * cutoff, intcut: f64::INFINITY, screen: false }
    }

    /// The basis.
    #[inline]
    pub fn basis(&self) -> &'a Basis {
        self.bas
    }

    /// The pair of ket atom `i` and bra atom `j`, or `None` if it is beyond the cutoff or (with
    /// screening) all its shell pairs are screened. An atom with itself is never skipped.
    #[inline]
    pub fn pair(&self, i: usize, j: usize) -> Option<AtomPair> {
        self.pair_with(i, j, self.bas.min_alpha_at[j])
    }

    /// The pair of ket atom `i` and a bra atom `j` whose shells (outside the basis, see
    /// [`ShellPairs::for_each_against`]) have the smallest exponent `amin_j`.
    #[inline]
    pub fn pair_with(&self, i: usize, j: usize, amin_j: f64) -> Option<AtomPair> {
        let vec = sub3(self.pos[i], self.pos[j]);
        let r2 = norm2_sq(vec);
        if r2 > self.cutoff2 || (self.screen && pair_screened(self.bas.min_alpha_at[i], amin_j, r2, self.intcut)) {
            return None;
        }
        Some(AtomPair { i, j, vec, r2 })
    }

    /// `f` for every shell pair of atom pair `p` that is not screened: shells of atom i in the
    /// outer loop, of atom j in the inner one.
    #[inline]
    pub fn for_each(&self, p: &AtomPair, mut f: impl FnMut(&ShellPair<'a>)) {
        let bas = self.bas;
        let (is, js) = (bas.ish_at[p.i], bas.ish_at[p.j]);
        for (ish, ci) in bas.shells(p.i).iter().enumerate() {
            let si = is + ish;
            for (jsh, cj) in bas.shells(p.j).iter().enumerate() {
                let sj = js + jsh;
                if self.screen && pair_screened(bas.min_alpha_sh[si], bas.min_alpha_sh[sj], p.r2, self.intcut) {
                    continue;
                }
                f(&ShellPair { ish, jsh, si, sj, ii: bas.iao_sh[si], jj: bas.iao_sh[sj], ci, cj });
            }
        }
    }

    /// `f` for every shell of atom i of pair `p` against one shell `cj` outside the basis on
    /// atom j (a projector, say; `jsh`, `sj` and `jj` are 0), without the screened ones.
    #[inline]
    pub fn for_each_against<'b>(&self, p: &AtomPair, cj: &'b Cgto, mut f: impl FnMut(&ShellPair<'b>))
    where
        'a: 'b,
    {
        let bas = self.bas;
        let is = bas.ish_at[p.i];
        let amin_j = cj.min_alpha();
        for (ish, ci) in bas.shells(p.i).iter().enumerate() {
            let si = is + ish;
            if self.screen && pair_screened(bas.min_alpha_sh[si], amin_j, p.r2, self.intcut) {
                continue;
            }
            f(&ShellPair { ish, jsh: 0, si, sj: 0, ii: bas.iao_sh[si], jj: 0, ci, cj });
        }
    }
}

/// Overlap, dipole and traceless quadrupole matrices of a basis (n x n, column-major), moments
/// around the atom of the column AO: `dip[k][(mu, nu)] = <mu| (r - R_atom(nu))_k |nu>`.
pub struct AoMatrices {
    pub overlap: Mat<f64>,
    pub dip: [Mat<f64>; 3],
    pub quad: [Mat<f64>; 6],
}

impl AoMatrices {
    /// S, D and Q of `bas` for `mol` with the screening of [`ShellPairs::screened`] (atom pair
    /// cutoff `cutoff`, primitive cutoff `bas.intcut`).
    ///
    /// Each unordered atom pair is evaluated once (ket: the larger atom index) and its mirrored
    /// block comes from [`shift_to_bra`]. With `parallel`, atoms are distributed over the
    /// current rayon pool; every matrix element is written by exactly one atom task (the larger
    /// atom index of its pair), so the result does not depend on the thread count.
    pub fn build(mol: &Molecule, bas: &Basis, cutoff: f64, parallel: bool) -> Self {
        let mats = assemble::<10, _, _>(mol, bas, cutoff, parallel, |_: &AtomPair| |_: &ShellPair| 0.0);
        Self::from_mats(mats.into_iter())
    }

    /// [`AoMatrices::build`] together with a scaled overlap matrix H, `H_mu,nu = h S_mu,nu` with a
    /// factor h for every shell pair, in the same pass: `scale(pair)` gives the factor function
    /// of an atom pair (called once per atom pair, the on-site pair `i == j` included), which
    /// gives the factor of each of its shell pairs. The mirrored block of H takes the same
    /// factor. (The core Hamiltonian of the GFN-xTB methods is such a matrix.)
    pub fn build_with_scaled<F, G>(
        mol: &Molecule,
        bas: &Basis,
        cutoff: f64,
        parallel: bool,
        scale: F,
    ) -> (Self, Mat<f64>)
    where
        F: Fn(&AtomPair) -> G + Sync,
        G: FnMut(&ShellPair) -> f64,
    {
        let mut mats = assemble::<11, _, _>(mol, bas, cutoff, parallel, scale).into_iter();
        let m = Self::from_mats(&mut mats);
        (m, mats.next().unwrap())
    }

    /// S, D and Q from the first ten matrices of `mats`.
    fn from_mats(mut mats: impl Iterator<Item = Mat<f64>>) -> Self {
        let overlap = mats.next().unwrap();
        let dip = std::array::from_fn(|_| mats.next().unwrap());
        let quad = std::array::from_fn(|_| mats.next().unwrap());
        Self { overlap, dip, quad }
    }
}

/// The matrices S, D (3), Q (6) and, for N = 11, the scaled overlap of [`AoMatrices`].
fn assemble<const N: usize, F, G>(mol: &Molecule, bas: &Basis, cutoff: f64, parallel: bool, scale: F) -> [Mat<f64>; N]
where
    F: Fn(&AtomPair) -> G + Sync,
    G: FnMut(&ShellPair) -> f64,
{
    let nao = bas.nao;
    // zeroed by the pool's threads: one thread would take all first-touch page faults
    let par = if parallel { faer::Par::rayon(0) } else { faer::Par::Seq };
    let mut mats: [Mat<f64>; N] = std::array::from_fn(|_| axtb_base::linalg::zeros(nao, nao, par));
    {
        let w = MatWriter::new(mats.each_mut());
        let pairs = ShellPairs::screened(mol, bas, cutoff, bas.intcut);
        let task = |scr: &mut Scratch, iat: usize| atom_blocks(&pairs, iat, &w, &scale, scr);
        if parallel {
            (0..mol.nat()).into_par_iter().for_each_init(Scratch::default, task);
        } else {
            let mut scr = Scratch::default();
            (0..mol.nat()).for_each(|iat| task(&mut scr, iat));
        }
    }
    mats
}

/// All blocks owned by atom `iat`: its pairs with every `jat < iat` (both orientations) and its
/// on-site block. Matrix order of `w`: S, D (3), Q (6), then the scaled overlap for N = 11.
#[inline]
fn atom_blocks<const N: usize, F, G>(pairs: &ShellPairs, iat: usize, w: &MatWriter<'_, N>, scale: &F, scr: &mut Scratch)
where
    F: Fn(&AtomPair) -> G,
    G: FnMut(&ShellPair) -> f64,
{
    let intcut = pairs.basis().intcut;
    let mut s = [0.0; MBLK];
    let mut d = [[0.0; 3]; MBLK];
    let mut q = [[0.0; 6]; MBLK];
    // SAFETY (all `w.set` calls): the AO indices are those of atoms iat and jat <= iat of the
    // basis the matrices were made for (in bounds), and element (mu, nu) with
    // atom(mu) <= atom(nu) = iat or atom(nu) <= atom(mu) = iat is written only by this task.
    for jat in 0..iat {
        let Some(p) = pairs.pair(iat, jat) else { continue };
        let mut h = scale(&p);
        pairs.for_each(&p, |sp| {
            multipole(sp.cj, sp.ci, p.vec, intcut, &mut s, &mut d, &mut q, scr);
            let hij = h(sp);
            let nj = nsph(sp.cj.l);
            for iao in 0..nsph(sp.ci.l) {
                for jao in 0..nj {
                    let ij = jao + nj * iao;
                    let (mu, nu) = (sp.jj + jao, sp.ii + iao);
                    let (dj, qj) = shift_to_bra(p.vec, s[ij], d[ij], q[ij]);
                    unsafe {
                        w.set(0, mu, nu, s[ij]);
                        w.set(0, nu, mu, s[ij]);
                        for k in 0..3 {
                            w.set(1 + k, mu, nu, d[ij][k]);
                            w.set(1 + k, nu, mu, dj[k]);
                        }
                        for k in 0..6 {
                            w.set(4 + k, mu, nu, q[ij][k]);
                            w.set(4 + k, nu, mu, qj[k]);
                        }
                        if N > 10 {
                            w.set(10, mu, nu, s[ij] * hij);
                            w.set(10, nu, mu, s[ij] * hij);
                        }
                    }
                }
            }
        });
    }
    // on-site: no cutoff, and the exponent screening never applies at R = 0
    let p = AtomPair::onsite(iat);
    let mut h = scale(&p);
    pairs.for_each(&p, |sp| {
        multipole(sp.cj, sp.ci, p.vec, intcut, &mut s, &mut d, &mut q, scr);
        let hij = h(sp);
        let nj = nsph(sp.cj.l);
        for iao in 0..nsph(sp.ci.l) {
            for jao in 0..nj {
                let ij = jao + nj * iao;
                let (mu, nu) = (sp.jj + jao, sp.ii + iao);
                unsafe {
                    w.set(0, mu, nu, s[ij]);
                    for k in 0..3 {
                        w.set(1 + k, mu, nu, d[ij][k]);
                    }
                    for k in 0..6 {
                        w.set(4 + k, mu, nu, q[ij][k]);
                    }
                    if N > 10 {
                        w.set(10, mu, nu, s[ij] * hij);
                    }
                }
            }
        }
    });
}

/// Writes into N n x n column-major matrices from several threads at once, for the lifetime
/// `'a` of the exclusive borrows it was made from (nothing else reads or writes the matrices
/// meanwhile).
struct MatWriter<'a, const N: usize> {
    ptr: [*mut f64; N],
    stride: [isize; N],
    n: usize,
    _mats: std::marker::PhantomData<&'a mut Mat<f64>>,
}

// SAFETY: the writer only hands out writes through `set`, whose contract makes concurrent
// calls touch disjoint elements
unsafe impl<const N: usize> Send for MatWriter<'_, N> {}
unsafe impl<const N: usize> Sync for MatWriter<'_, N> {}

impl<'a, const N: usize> MatWriter<'a, N> {
    fn new(mats: [&'a mut Mat<f64>; N]) -> Self {
        let n = mats[0].nrows();
        let mut ptr = [std::ptr::null_mut(); N];
        let mut stride = [0; N];
        for (k, m) in mats.into_iter().enumerate() {
            assert!(m.nrows() == n && m.ncols() == n && m.row_stride() == 1);
            stride[k] = m.col_stride();
            ptr[k] = m.as_ptr_mut();
        }
        Self { ptr, stride, n, _mats: std::marker::PhantomData }
    }

    /// Set element (i, j) of matrix `m` to `v`.
    ///
    /// # Safety
    /// `m < N` and `i, j < n` (checked in debug builds only), and no other thread writes
    /// element (i, j) of matrix `m` while this call runs. `atom_blocks` keeps the latter by
    /// letting the task of atom `iat` write only the blocks of its pairs with atoms `jat <=
    /// iat` (both orientations): every element has exactly one writer.
    #[inline]
    unsafe fn set(&self, m: usize, i: usize, j: usize, v: f64) {
        debug_assert!(m < N && i < self.n && j < self.n);
        *self.ptr[m].offset(i as isize + j as isize * self.stride[m]) = v;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::ao_offsets;
    use crate::gfn2;

    fn molecule() -> Molecule {
        let numbers = [8, 1, 1, 6, 26, 1];
        let pos = [
            [0.1, 0.2, 0.22],
            [0.3, 1.43, -0.9],
            [-0.2, -1.4, -0.85],
            [2.9, 0.4, 1.7],
            [-2.6, 1.1, -1.9],
            [30.0, 25.0, -20.0],
        ];
        Molecule::new(&numbers, &pos, 0.0, None).unwrap()
    }

    /// The same shells per species and per atom give the same index maps and screening data.
    #[test]
    fn atom_shells_give_the_species_index_maps() {
        let mol = molecule();
        let bas = gfn2::build(&mol, 1.0, &crate::Conventions::Tblite.params()).unwrap().basis;
        let shells: Vec<Vec<Cgto>> = (0..mol.nat()).map(|i| bas.shells(i).to_vec()).collect();
        let per_atom = Basis::from_atom_shells(&mol, shells.clone(), bas.intcut);
        assert_eq!(per_atom.id, (0..mol.nat()).collect::<Vec<_>>());
        assert_eq!(bas.id, mol.id);
        assert_eq!(per_atom.ish_at, bas.ish_at);
        assert_eq!(per_atom.nsh_at, bas.nsh_at);
        assert_eq!(per_atom.iao_at, bas.iao_at);
        assert_eq!(per_atom.nao_at, bas.nao_at);
        assert!((0..mol.nat()).all(|i| bas.iao_at[i] == bas.iao_sh[bas.ish_at[i]]));
        assert_eq!(bas.nao_at.iter().sum::<usize>(), bas.nao);
        let listed: Vec<(usize, usize)> = per_atom.shell_iter().map(|(i, ish, _)| (i, ish)).collect();
        assert_eq!(listed, (0..bas.nsh).map(|s| (bas.sh2at[s], s - bas.ish_at[bas.sh2at[s]])).collect::<Vec<_>>());
        assert_eq!(per_atom.sh2at, bas.sh2at);
        assert_eq!(per_atom.iao_sh, bas.iao_sh);
        assert_eq!(per_atom.nao_sh, bas.nao_sh);
        assert_eq!(per_atom.ao2at, bas.ao2at);
        assert_eq!(per_atom.ao2sh, bas.ao2sh);
        assert_eq!((per_atom.nsh, per_atom.nao, per_atom.maxl), (bas.nsh, bas.nao, bas.maxl));
        assert_eq!(per_atom.min_alpha.to_bits(), bas.min_alpha.to_bits());
        assert_eq!(per_atom.min_alpha_sh, bas.min_alpha_sh);
        assert_eq!(per_atom.min_alpha_at, bas.min_alpha_at);
        let first = ao_offsets(&shells);
        for i in 0..mol.nat() {
            for (ish, &f) in first[i].iter().enumerate() {
                assert_eq!(f, bas.iao_sh[bas.ish_at[i] + ish]);
            }
        }
    }

    fn same(a: &Mat<f64>, b: &Mat<f64>) -> bool {
        a.nrows() == b.nrows()
            && (0..a.ncols()).all(|j| (0..a.nrows()).all(|i| a[(i, j)].to_bits() == b[(i, j)].to_bits()))
    }

    fn all(m: &AoMatrices) -> Vec<&Mat<f64>> {
        std::iter::once(&m.overlap).chain(m.dip.iter()).chain(m.quad.iter()).collect()
    }

    /// S, D, Q: the same bits per species and per atom, serial and parallel, with and without
    /// the scaled overlap; screening and the cutoff only drop blocks that are exactly zero.
    #[test]
    fn ao_matrices_are_the_same_for_every_path() {
        let mol = molecule();
        let bas = gfn2::build(&mol, 1.0, &crate::Conventions::Tblite.params()).unwrap().basis;
        let shells: Vec<Vec<Cgto>> = (0..mol.nat()).map(|i| bas.shells(i).to_vec()).collect();
        let per_atom = Basis::from_atom_shells(&mol, shells, bas.intcut);
        let reference = AoMatrices::build(&mol, &bas, 40.0, false);
        let (scaled, h) =
            AoMatrices::build_with_scaled(&mol, &per_atom, 40.0, true, |_: &AtomPair| |_: &ShellPair| 2.0);
        let unscreened = AoMatrices::build(&mol, &Basis { intcut: f64::INFINITY, ..bas.clone() }, f64::INFINITY, false);
        for other in
            [AoMatrices::build(&mol, &per_atom, 40.0, false), AoMatrices::build(&mol, &bas, 40.0, true), scaled]
        {
            assert!(all(&reference).iter().zip(all(&other)).all(|(a, b)| same(a, b)));
        }
        let s = &reference.overlap;
        assert!((0..bas.nao).all(|j| (0..bas.nao).all(|i| h[(i, j)].to_bits() == (2.0 * s[(i, j)]).to_bits())));
        // the far hydrogen is beyond the primitive cutoff of every other atom
        let far = bas.iao_sh[bas.ish_at[5]];
        assert!((0..far).all(|i| s[(i, far)].to_bits() == 0 && s[(far, i)].to_bits() == 0));
        assert!((s[(far, far)] - 1.0).abs() < 1e-10);
        assert!(unscreened.overlap[(0, far)] != 0.0);
    }

    /// The visitor without screening visits every shell pair within the cutoff.
    #[test]
    fn unscreened_pairs_visit_every_shell_pair() {
        let mol = molecule();
        let bas = gfn2::build(&mol, 1.0, &crate::Conventions::Tblite.params()).unwrap().basis;
        let (all, screened) = (
            ShellPairs::unscreened(&mol, &bas, f64::INFINITY),
            ShellPairs::screened(&mol, &bas, f64::INFINITY, bas.intcut),
        );
        let count = |sp: &ShellPairs| {
            let mut n = 0;
            for i in 0..mol.nat() {
                for j in 0..mol.nat() {
                    if let Some(p) = sp.pair(i, j) {
                        sp.for_each(&p, |_| n += 1);
                    }
                }
            }
            n
        };
        assert_eq!(count(&all), bas.nsh * bas.nsh);
        assert!(count(&screened) < bas.nsh * bas.nsh);
        assert!(ShellPairs::unscreened(&mol, &bas, 1.0).pair(0, 3).is_none());
    }
}
