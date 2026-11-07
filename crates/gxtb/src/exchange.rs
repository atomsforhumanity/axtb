//! Mulliken-approximated range-separated Fock exchange (P-Eq. 41), as evaluated by the
//! 1.1.0 program (`setfock_mat_`):
//! E = -1/4 sum_sigma sum_{mu nu} gamma_{mu nu} [(P^s S)_{mu nu} (S P^s)_{mu nu} + P^s_{mu nu} (S P^s S)_{mu nu}],
//! i.e. -1/8 [...] with the total density for closed shells. Matrices are full, row-major,
//! in one AO order; `gamma` is the AO-expanded MFX kernel.

/// Exchange energy for spin densities `p` (one matrix for a closed shell = total density).
pub fn mfx_energy(n: usize, s: &[f64], p: &[Vec<f64>], gamma: &[f64]) -> f64 {
    let mm = |a: &[f64], b: &[f64]| {
        let mut c = vec![0.0; n * n];
        for i in 0..n {
            for k in 0..n {
                let aik = a[i * n + k];
                if aik == 0.0 {
                    continue;
                }
                for j in 0..n {
                    c[i * n + j] += aik * b[k * n + j];
                }
            }
        }
        c
    };
    let closed = p.len() == 1;
    let mut e = 0.0;
    for ps in p {
        let psm = mm(ps, s);
        let sp = mm(s, ps);
        let sps = mm(&sp, s);
        let mut t = 0.0;
        for k in 0..n * n {
            t += gamma[k] * (psm[k] * sp[k] + ps[k] * sps[k]);
        }
        e += if closed { -0.125 * t } else { -0.25 * t };
    }
    e
}

/// Onsite exchange integral index (1-based into the 9 per-element entries of `com::gmunu`)
/// of an angular-momentum pair (`setfock_indo_$IPOINT`): sp 1, pp 2, sd 3, pd 4, dd 5,
/// sf 6, pf 7, df 8, ff 9; s-s has none.
pub fn gmunu_index(la: usize, lb: usize) -> Option<usize> {
    match (la.min(lb), la.max(lb)) {
        (0, 1) => Some(1),
        (1, 1) => Some(2),
        (0, 2) => Some(3),
        (1, 2) => Some(4),
        (2, 2) => Some(5),
        (0, 3) => Some(6),
        (1, 3) => Some(7),
        (2, 3) => Some(8),
        (3, 3) => Some(9),
        _ => None,
    }
}

/// Exchange-integral scaling applied by the program at start-up (alpha of the reference
/// functional's short-range exact exchange).
pub const OFX_SCALE: f64 = 0.15;

/// Onsite exchange correction (P-Eq. 43, `setfock_indo_`):
/// E = -sum_sigma sum_A [ sum_{k != l in A} L_kl Pt_kl^2
///       + sum_{shells s in A, l_s >= 1} sum_{k,l in s} ((2l+1) d_kl - 1)/(2l+1) L_kl Pt_kk Pt_ll ],
/// with the dual density Pt = (P^s S + S P^s)/2 per spin (closed shell: P^s = P/2) and
/// L = 0.15 * gmunu(index(l_k, l_l), Z). `ao` lists (atom, shell, l) per AO.
pub fn ofx_energy(numbers: &[usize], ao: &[(usize, usize, usize)], s: &[f64], p: &[Vec<f64>]) -> f64 {
    let n = ao.len();
    let closed = p.len() == 1;
    let mut e = 0.0;
    for ps in p {
        // spin density and its dual
        let scale = if closed { 0.5 } else { 1.0 };
        let mut pt = vec![0.0; n * n];
        for i in 0..n {
            for j in 0..n {
                if ao[i].0 != ao[j].0 {
                    continue;
                }
                let mut v = 0.0;
                for k in 0..n {
                    v += ps[i * n + k] * s[k * n + j] + s[i * n + k] * ps[k * n + j];
                }
                pt[i * n + j] = 0.5 * scale * v;
            }
        }
        let mut t = 0.0;
        for i in 0..n {
            for j in 0..n {
                let (ai, si, li) = ao[i];
                let (aj, sj, lj) = ao[j];
                if ai != aj {
                    continue;
                }
                let Some(k) = gmunu_index(li, lj) else { continue };
                let lam = OFX_SCALE * crate::data::GMUNU[(k - 1) + 9 * (numbers[ai] - 1)];
                if i != j {
                    t += lam * pt[i * n + j] * pt[i * n + j];
                }
                if si == sj && li >= 1 {
                    let f = (if i == j { (2 * li + 1) as f64 } else { 0.0 } - 1.0) / (2 * li + 1) as f64;
                    t += f * lam * pt[i * n + i] * pt[j * n + j];
                }
            }
        }
        e -= if closed { 2.0 * t } else { t };
    }
    e
}
