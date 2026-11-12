//! Properties of a converged SCF that the reference programs print: Mayer–Wiberg bond orders,
//! the molecular quadrupole moment and the HOMO–LUMO gap. Shared by GFN2-xTB and g-xTB.
//!
//! - **Bond orders** (xtb `get_wiberg`/`get_unrestricted_wiberg`, tblite
//!   `get_mayer_bond_orders`): B_AB = Σ_{μ∈A, ν∈B} (PS)_μν (PS)_νμ with the total density of a
//!   closed shell, and B_AB = 2 Σ_σ Σ_{μ∈A, ν∈B} (P^σ S)_μν (P^σ S)_νμ with the alpha and beta
//!   densities of an open shell (Mayer's open-shell form; equal to the closed-shell value for
//!   P^α = P^β = P/2). Both programs use these formulas; xtb's GFN2-xTB (6.7.1 and 6.8.0) sets
//!   pairs 10 bohr or more apart to zero ([`cut_bond_orders`]), tblite (and so the g-xTB 2.0.1
//!   program) does not.
//! - **Molecular quadrupole** about the coordinate origin, traceless, from the atomic charges,
//!   dipoles and quadrupoles (the programs' "full" quadrupole). [`QuadrupoleForm::Traceless`]
//!   is Θ = Σ_A [θ_A + 3/2 (q r r + r d + d r) − 1/2 (q r² + 2 r·d) 1] (Buckingham's convention,
//!   the one of the atomic quadrupoles θ_A: xtb's `molmom`). [`QuadrupoleForm::Tblite`] is
//!   tblite's `get_molecular_quadrupole_moment`, which takes the off-diagonal elements of the
//!   charge and dipole parts twice (3 instead of 3/2), so its result is not a tensor (it depends
//!   on the orientation of the molecule); the g-xTB 2.0.1 program prints it.
//! - **HOMO–LUMO gap** from the orbital energies and occupations of the spin channels.

use faer::{Mat, MatRef, Par};

pub use axtb_base::orbitals::OrbitalLevels;

/// The alpha and beta channels of `levels` (qcdata's `scf_eigenvalues_a/b`,
/// `scf_occupations_a/b`): a closed shell's one channel twice with half its occupations.
pub fn alpha_beta(levels: &[OrbitalLevels]) -> [OrbitalLevels; 2] {
    match levels {
        [one] => {
            let half = OrbitalLevels {
                energies: one.energies.clone(),
                occupations: one.occupations.iter().map(|o| 0.5 * o).collect(),
            };
            [half.clone(), half]
        }
        [a, b, ..] => [a.clone(), b.clone()],
        [] => [OrbitalLevels::default(), OrbitalLevels::default()],
    }
}

/// Mayer–Wiberg bond orders (row-major N x N, zero diagonal) from the overlap `s` and the
/// density matrices: one (the total density of a closed shell) or two (the alpha and beta
/// densities; each spin's sum is doubled). `ao_atom` maps every AO to its atom (AOs ordered by
/// atom).
pub fn mayer_bond_orders(
    nat: usize,
    ao_atom: &[usize],
    s: MatRef<'_, f64>,
    densities: &[MatRef<'_, f64>],
    par: Par,
) -> Vec<f64> {
    let n = ao_atom.len();
    let scale = if densities.len() == 2 { 2.0 } else { 1.0 };
    let mut bo = vec![0.0; nat * nat];
    let mut ps = Mat::<f64>::zeros(n, n);
    for p in densities {
        faer::linalg::matmul::matmul(ps.as_mut(), faer::Accum::Replace, *p, s, 1.0, par);
        for mu in 0..n {
            let a = ao_atom[mu];
            for nu in 0..mu {
                let b = ao_atom[nu];
                if a != b {
                    bo[a * nat + b] += ps[(mu, nu)] * ps[(nu, mu)];
                }
            }
        }
    }
    for a in 0..nat {
        for b in 0..a {
            let v = scale * bo[a * nat + b];
            bo[a * nat + b] = v;
            bo[b * nat + a] = v;
        }
    }
    bo
}

/// Zero the bond orders (N x N) of atom pairs at least `max_distance` (bohr) apart, as xtb's
/// `get_wiberg` does for 10 bohr.
pub fn cut_bond_orders(bo: &mut [f64], positions: &[[f64; 3]], max_distance: f64) {
    let nat = positions.len();
    for a in 0..nat {
        for b in 0..nat {
            let r2: f64 = (0..3).map(|k| (positions[a][k] - positions[b][k]).powi(2)).sum();
            if r2 >= max_distance * max_distance {
                bo[a * nat + b] = 0.0;
            }
        }
    }
}

/// Alpha and beta density matrices of one set of orbitals (`coeff` columns) with the alpha and
/// beta occupations (GFN2-xTB's open shells).
pub fn spin_densities(coeff: MatRef<'_, f64>, occ_a: &[f64], occ_b: &[f64], par: Par) -> [Mat<f64>; 2] {
    let n = coeff.nrows();
    let mut pa = Mat::<f64>::zeros(n, n);
    let mut pb = Mat::<f64>::zeros(n, n);
    crate::linalg::weighted_outer(coeff, occ_a, &mut pa, par);
    crate::linalg::weighted_outer(coeff, occ_b, &mut pb, par);
    [pa, pb]
}

pub use axtb_base::method::QuadrupoleForm;

/// The form GFN2-xTB prints in its conventions: tblite's in the tblite conventions, the
/// traceless tensor otherwise (xtb's `molmom`).
impl From<crate::Conventions> for QuadrupoleForm {
    fn from(conventions: crate::Conventions) -> Self {
        match conventions {
            crate::Conventions::Tblite => QuadrupoleForm::Tblite,
            _ => QuadrupoleForm::Traceless,
        }
    }
}

/// Molecular quadrupole (xx, xy, yy, xz, yz, zz; e·bohr², traceless) about the coordinate
/// origin from the atomic charges `qat`, dipoles `dpat` and quadrupoles `qpat` (the same
/// packing) at `positions` (bohr).
pub fn molecular_quadrupole(
    positions: &[[f64; 3]],
    qat: &[f64],
    dpat: &[[f64; 3]],
    qpat: &[[f64; 6]],
    form: QuadrupoleForm,
) -> [f64; 6] {
    let mut out = [0.0; 6];
    match form {
        QuadrupoleForm::Traceless => {
            // xtb `molmom`: charge part q r_j r_k and dipole part d_k r_j + d_j r_k, each made
            // traceless with 3/2 M - 1/2 tr(M), plus the atomic quadrupoles
            let (mut tma, mut tmb, mut tmc) = ([0.0; 6], [0.0; 6], [0.0; 6]);
            for (i, r) in positions.iter().enumerate() {
                let (q, d) = (qat[i], dpat[i]);
                for j in 0..3 {
                    for k in 0..=j {
                        let l = lin(k, j);
                        tma[l] += r[j] * r[k] * q;
                        tmb[l] += d[k] * r[j] + d[j] * r[k];
                    }
                }
                for l in 0..6 {
                    tmc[l] += qpat[i][l];
                }
            }
            for t in [&mut tma, &mut tmb] {
                let tr = 0.5 * (t[0] + t[2] + t[5]);
                for v in t.iter_mut() {
                    *v *= 1.5;
                }
                for l in [0, 2, 5] {
                    t[l] -= tr;
                }
            }
            for l in 0..6 {
                out[l] = tma[l] + tmb[l] + tmc[l];
            }
        }
        QuadrupoleForm::Tblite => {
            for (i, r) in positions.iter().enumerate() {
                let (q, d) = (qat[i], dpat[i]);
                let v = [r[0] * q, r[1] * q, r[2] * q];
                let mut c = [0.0; 6];
                c[0] = r[0] * (v[0] + 2.0 * d[0]);
                c[2] = r[1] * (v[1] + 2.0 * d[1]);
                c[5] = r[2] * (v[2] + 2.0 * d[2]);
                c[1] = r[0] * (v[1] + d[1]) + d[0] * r[1];
                c[3] = r[0] * (v[2] + d[2]) + d[0] * r[2];
                c[4] = r[1] * (v[2] + d[2]) + d[1] * r[2];
                let tr = 0.5 * (c[0] + c[2] + c[5]);
                c[0] = 1.5 * c[0] - tr;
                c[1] *= 3.0;
                c[2] = 1.5 * c[2] - tr;
                c[3] *= 3.0;
                c[4] *= 3.0;
                c[5] = 1.5 * c[5] - tr;
                for l in 0..6 {
                    out[l] += qpat[i][l] + c[l];
                }
            }
        }
    }
    out
}

/// Packed index of the symmetric pair (k <= j) in the order xx, xy, yy, xz, yz, zz.
fn lin(k: usize, j: usize) -> usize {
    j * (j + 1) / 2 + k
}

/// How the HOMO–LUMO gap is defined (see [`homo_lumo_gap`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GapRule {
    /// Over both spin channels: the lowest unoccupied minus the highest occupied level of any
    /// channel (one set of orbitals, closed shells and GFN2-xTB's open shells: the levels above
    /// and at the alpha HOMO).
    #[default]
    Overall,
    /// The alpha channel only (xtb and tblite, so the g-xTB 2.0.1 program, print this).
    Alpha,
}

/// Index of the HOMO of a channel with `nel` electrons (0-based), as xtb and tblite count it:
/// floor(nel), one more when the fractional part exceeds 1/2; `None` without electrons.
fn homo_index(nel: f64) -> Option<usize> {
    let n = nel.floor();
    let k = if nel - n > 0.5 { n + 1.0 } else { n };
    (k >= 1.0).then(|| k as usize - 1)
}

/// HOMO–LUMO gap (Eh) of the spin channels `levels` (one: closed shell or one set of orbitals
/// with the total occupations; two: alpha and beta) with `nalpha` and `nbeta` electrons. The
/// HOMO of a channel is its Aufbau level (the `nel`-th), independent of
/// smearing. One set of orbitals (`levels.len() == 1`): the gap above the alpha HOMO. `None`
/// when a channel has no HOMO or no LUMO.
pub fn homo_lumo_gap(levels: &[OrbitalLevels], nalpha: f64, nbeta: f64, rule: GapRule) -> Option<f64> {
    let channel = |e: &[f64], nel: f64| -> Option<(f64, f64)> {
        let h = homo_index(nel)?;
        Some((*e.get(h)?, *e.get(h + 1)?))
    };
    match levels {
        [one] => channel(&one.energies, nalpha.max(nbeta)).map(|(h, l)| l - h),
        [a, b] => {
            let ga = channel(&a.energies, nalpha);
            if rule == GapRule::Alpha {
                return ga.map(|(h, l)| l - h);
            }
            let gb = channel(&b.energies, nbeta);
            match (ga, gb) {
                (Some((ha, la)), Some((hb, lb))) => Some(la.min(lb) - ha.max(hb)),
                (Some((h, l)), None) | (None, Some((h, l))) => {
                    // a channel without electrons: its lowest level is unoccupied
                    let other = if nalpha < 1.0 { &a.energies } else { &b.energies };
                    Some(l.min(other.first().copied().unwrap_or(l)) - h)
                }
                (None, None) => None,
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quadrupole_forms_agree_on_diagonal_and_traceless_is_covariant() {
        // two point charges and dipoles; rotate by 90 degrees about z and compare
        let pos = [[0.3, -0.2, 0.5], [-1.1, 0.7, 0.2]];
        let q = [0.4, -0.4];
        let d = [[0.1, 0.0, -0.2], [0.05, 0.3, 0.0]];
        let qp = [[0.1, 0.02, -0.05, 0.03, 0.0, -0.05], [0.0; 6]];
        let a = molecular_quadrupole(&pos, &q, &d, &qp, QuadrupoleForm::Traceless);
        let b = molecular_quadrupole(&pos, &q, &d, &qp, QuadrupoleForm::Tblite);
        for l in [0, 2, 5] {
            assert!((a[l] - b[l]).abs() < 1e-14);
        }
        assert!((a[0] + a[2] + a[5]).abs() < 1e-14);
        // rotation x -> y, y -> -x: Θ'_xx = Θ_yy, Θ'_xy = -Θ_xy, Θ'_xz = -Θ_yz, Θ'_yz = Θ_xz
        let rot = |v: [f64; 3]| [-v[1], v[0], v[2]];
        let rq = |t: [f64; 6]| [t[2], -t[1], t[0], -t[4], t[3], t[5]];
        let pos2: Vec<[f64; 3]> = pos.iter().map(|&p| rot(p)).collect();
        let d2: Vec<[f64; 3]> = d.iter().map(|&p| rot(p)).collect();
        let qp2: Vec<[f64; 6]> = qp.iter().map(|&t| rq(t)).collect();
        let a2 = molecular_quadrupole(&pos2, &q, &d2, &qp2, QuadrupoleForm::Traceless);
        let want = rq(a);
        for l in 0..6 {
            assert!((a2[l] - want[l]).abs() < 1e-14, "{l}: {} vs {}", a2[l], want[l]);
        }
    }

    #[test]
    fn gap_rules() {
        let one = OrbitalLevels { energies: vec![-0.5, -0.3, 0.1, 0.4], occupations: vec![2.0, 2.0, 0.0, 0.0] };
        assert!((homo_lumo_gap(std::slice::from_ref(&one), 2.0, 2.0, GapRule::Overall).unwrap() - 0.4).abs() < 1e-15);
        let a = OrbitalLevels { energies: vec![-0.6, -0.4, -0.2, 0.3], occupations: vec![1.0, 1.0, 1.0, 0.0] };
        let b = OrbitalLevels { energies: vec![-0.55, -0.35, 0.0, 0.35], occupations: vec![1.0, 1.0, 0.0, 0.0] };
        let both = [a, b];
        assert!((homo_lumo_gap(&both, 3.0, 2.0, GapRule::Alpha).unwrap() - 0.5).abs() < 1e-15);
        assert!((homo_lumo_gap(&both, 3.0, 2.0, GapRule::Overall).unwrap() - 0.2).abs() < 1e-15);
        assert_eq!(homo_index(0.4), None);
        assert_eq!(homo_index(0.6), Some(0));
        assert_eq!(homo_index(3.0), Some(2));
    }
}
