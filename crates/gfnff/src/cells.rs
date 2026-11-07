//! Linked-cell search for atom pairs within a cutoff (O(N) for dense systems).
//!
//! xtb's GFN-FF loops over all atom pairs (and keeps N x N arrays); every pair loop here
//! visits only the pairs that can pass the term's own cutoff test, which is still
//! applied with xtb's exact comparison. Candidate lists come out sorted by atom index,
//! i.e. in the order of xtb's loops.

use std::collections::HashMap;

/// Atoms binned into cubic cells.
pub struct CellList {
    cell: f64,
    dims: [i64; 3],
    /// atoms of each cell (dense grid, x fastest), ascending
    grid: Vec<Vec<u32>>,
    /// sparse fallback for very elongated systems
    sparse: Option<HashMap<[i64; 3], Vec<u32>>>,
    coords: Vec<[i64; 3]>,
}

impl CellList {
    /// Bin `positions` into cubic cells of edge `cell` (bohr).
    pub fn new(positions: &[[f64; 3]], cell: f64) -> Self {
        let cell = cell.max(1.0);
        let mut lo = [f64::INFINITY; 3];
        let mut hi = [f64::NEG_INFINITY; 3];
        for p in positions {
            for d in 0..3 {
                lo[d] = lo[d].min(p[d]);
                hi[d] = hi[d].max(p[d]);
            }
        }
        if positions.is_empty() {
            lo = [0.0; 3];
            hi = [0.0; 3];
        }
        // at most 1e12 cells per axis (no integer overflow for absurd extents); clamping the
        // cell indices of atoms beyond that keeps every pair within reach
        let dims = [0, 1, 2].map(|d| (((hi[d] - lo[d]) / cell).min(1e12).floor() as i64 + 1).max(1));
        let coords: Vec<[i64; 3]> = positions
            .iter()
            .map(|p| [0, 1, 2].map(|d| (((p[d] - lo[d]) / cell).floor() as i64).clamp(0, dims[d] - 1)))
            .collect();
        let ncell = dims[0].saturating_mul(dims[1]).saturating_mul(dims[2]);
        if ncell <= 4 * positions.len() as i64 + 64 {
            let mut grid = vec![Vec::new(); ncell as usize];
            for (i, c) in coords.iter().enumerate() {
                grid[(c[0] + dims[0] * (c[1] + dims[1] * c[2])) as usize].push(i as u32);
            }
            CellList { cell, dims, grid, sparse: None, coords }
        } else {
            let mut map: HashMap<[i64; 3], Vec<u32>> = HashMap::new();
            for (i, c) in coords.iter().enumerate() {
                map.entry(*c).or_default().push(i as u32);
            }
            CellList { cell, dims, grid: Vec::new(), sparse: Some(map), coords }
        }
    }

    fn cell_atoms(&self, c: [i64; 3]) -> &[u32] {
        if let Some(map) = &self.sparse {
            map.get(&c).map(|v| v.as_slice()).unwrap_or(&[])
        } else {
            &self.grid[(c[0] + self.dims[0] * (c[1] + self.dims[1] * c[2])) as usize]
        }
    }

    /// Candidates j (ascending) whose cell lies within `cutoff` of atom `i`'s cell; the
    /// caller tests the distance. `filter(j)` selects which j to keep.
    pub fn candidates(&self, i: usize, cutoff: f64, mut filter: impl FnMut(usize) -> bool, out: &mut Vec<usize>) {
        out.clear();
        let reach = (cutoff / self.cell).ceil() as i64;
        let c = self.coords[i];
        for dz in -reach..=reach {
            let z = c[2] + dz;
            if z < 0 || z >= self.dims[2] {
                continue;
            }
            for dy in -reach..=reach {
                let y = c[1] + dy;
                if y < 0 || y >= self.dims[1] {
                    continue;
                }
                for dx in -reach..=reach {
                    let x = c[0] + dx;
                    if x < 0 || x >= self.dims[0] {
                        continue;
                    }
                    for &j in self.cell_atoms([x, y, z]) {
                        if filter(j as usize) {
                            out.push(j as usize);
                        }
                    }
                }
            }
        }
        out.sort_unstable();
    }
}

/// Squared distance.
#[inline]
pub fn dist2(a: [f64; 3], b: [f64; 3]) -> f64 {
    let d = [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
    d[0] * d[0] + d[1] * d[1] + d[2] * d[2]
}
