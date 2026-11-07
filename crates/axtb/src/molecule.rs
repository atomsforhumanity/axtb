//! Molecular structure.

use crate::error::{Error, Result};

/// Smallest distance (bohr) between two atoms that [`Molecule::new`] and
/// [`Molecule::set_positions`] accept, for every method. Closer atoms are a broken structure
/// (a duplicated atom, a unit mix-up), not chemistry: the overlap matrix turns singular and
/// pair terms diverge. 0.1 bohr is far below any bond (H2: 1.4 bohr) and below anything an
/// optimizer or a finite-difference step makes of a sound structure.
pub const MIN_ATOM_DISTANCE: f64 = 0.1;

/// A molecule: atomic numbers, Cartesian positions in bohr, total charge and the number of
/// unpaired electrons.
#[derive(Clone, Debug)]
pub struct Molecule {
    /// Atomic number of every atom.
    pub numbers: Vec<usize>,
    /// Positions in bohr.
    pub positions: Vec<[f64; 3]>,
    /// Total molecular charge.
    pub charge: f64,
    /// Number of unpaired electrons (multiplicity - 1); `None` lets the method choose
    /// (0 for an even and 1 for an odd electron count).
    pub uhf: Option<usize>,
    /// Species index of every atom into `species`.
    pub id: Vec<usize>,
    /// Atomic number of every distinct species, in order of first appearance.
    pub species: Vec<usize>,
}

impl Molecule {
    /// Build a molecule from atomic numbers and bohr positions.
    pub fn new(numbers: &[usize], positions: &[[f64; 3]], charge: f64, uhf: Option<usize>) -> Result<Self> {
        if numbers.len() != positions.len() {
            return Err(Error::Input(format!("{} atomic numbers but {} positions", numbers.len(), positions.len())));
        }
        if numbers.is_empty() {
            return Err(Error::Input("molecule has no atoms".into()));
        }
        if let Some(&z) = numbers.iter().find(|&&z| z == 0 || z > 118) {
            return Err(Error::Input(format!("invalid atomic number {z}")));
        }
        if !charge.is_finite() {
            return Err(Error::Input(format!("the charge must be a finite number, got {charge}")));
        }
        check_positions(positions)?;
        let mut species = Vec::new();
        let mut id = Vec::with_capacity(numbers.len());
        for &z in numbers {
            let isp = match species.iter().position(|&s| s == z) {
                Some(i) => i,
                None => {
                    species.push(z);
                    species.len() - 1
                }
            };
            id.push(isp);
        }
        Ok(Self { numbers: numbers.to_vec(), positions: positions.to_vec(), charge, uhf, id, species })
    }

    /// Number of atoms.
    #[inline]
    pub fn nat(&self) -> usize {
        self.numbers.len()
    }

    /// Number of distinct species.
    #[inline]
    pub fn nid(&self) -> usize {
        self.species.len()
    }

    /// Replace the positions (bohr), keeping everything else; checked as in
    /// [`Molecule::new`] (after an error the positions are those of before).
    pub fn set_positions(&mut self, positions: &[[f64; 3]]) -> Result<()> {
        if positions.len() != self.nat() {
            return Err(Error::Input(format!("{} positions for {} atoms", positions.len(), self.nat())));
        }
        check_positions(positions)?;
        self.positions.copy_from_slice(positions);
        Ok(())
    }
}

/// Finite positions, no two atoms closer than [`MIN_ATOM_DISTANCE`]. The atoms are swept in
/// the order of their x coordinates, comparing only pairs closer than that in x.
fn check_positions(positions: &[[f64; 3]]) -> Result<()> {
    if let Some(i) = positions.iter().position(|r| r.iter().any(|x| !x.is_finite())) {
        return Err(Error::Input(format!("atom {}: position {:?} is not finite", i + 1, positions[i])));
    }
    let mut order: Vec<usize> = (0..positions.len()).collect();
    order.sort_by(|&a, &b| positions[a][0].total_cmp(&positions[b][0]));
    for (k, &i) in order.iter().enumerate() {
        for &j in &order[k + 1..] {
            if positions[j][0] - positions[i][0] >= MIN_ATOM_DISTANCE {
                break;
            }
            let r = norm2_sq(sub3(positions[i], positions[j])).sqrt();
            if r < MIN_ATOM_DISTANCE {
                return Err(Error::Input(format!(
                    "atoms {} and {} are {r:.3e} bohr apart, closer than {MIN_ATOM_DISTANCE} bohr",
                    i.min(j) + 1,
                    i.max(j) + 1
                )));
            }
        }
    }
    Ok(())
}

/// Difference vector `a - b`.
#[inline]
pub fn sub3(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

/// Squared norm.
#[inline]
pub fn norm2_sq(v: [f64; 3]) -> f64 {
    v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
}

/// Dot product.
#[inline]
pub fn dot3(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The sweep over x finds a close pair exactly when a loop over all pairs does.
    #[test]
    fn close_atoms_found_as_by_all_pairs() {
        let mut seed = 99u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        let (mut ok, mut bad) = (0, 0);
        for trial in 0..200 {
            let n = 2 + trial % 60;
            let side = 0.4 + 0.02 * trial as f64;
            let p: Vec<[f64; 3]> = (0..n).map(|_| [rnd() * side, rnd() * side, rnd() * side]).collect();
            let close = (0..n).any(|i| (0..i).any(|j| norm2_sq(sub3(p[i], p[j])).sqrt() < MIN_ATOM_DISTANCE));
            assert_eq!(check_positions(&p).is_err(), close, "trial {trial}");
            if close {
                bad += 1;
            } else {
                ok += 1;
            }
        }
        assert!(ok > 20 && bad > 20, "{ok} {bad}");
    }
}
