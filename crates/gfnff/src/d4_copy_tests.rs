//! Check before the switch (removed with the copied tables in the next commit): axtb-d4's
//! dftd4 tables, masked as xtb's GFN-FF has them (no references for Fr-Rg, `<r4>/<r2>` zero
//! for Am-Rg), give the tables, polarizabilities, reference C6 and `sqrt(Z r4/r2)` of the
//! copies in `data.rs` bit for bit.

use crate::data;
use crate::energy::DispModel;
use axtb_d4::data::{MAX_REF, NFREQ};
use std::collections::HashMap;

fn nref(z: usize) -> usize {
    if (87..=111).contains(&z) {
        0
    } else {
        axtb_d4::data::REFN[z - 1] as usize
    }
}

fn sqrt_z_r4_over_r2(z: usize) -> f64 {
    if (95..=111).contains(&z) {
        0.0
    } else {
        axtb_d4::r4r2_value(z)
    }
}

/// `DispModel::new` on axtb-d4, as the next commit writes it.
fn new_disp_model(numbers: &[usize]) -> DispModel {
    let mut zs: Vec<usize> = numbers.to_vec();
    zs.sort_unstable();
    zs.dedup();
    let nref: Vec<usize> = zs.iter().map(|&z| nref(z)).collect();
    let alpha: Vec<[[f64; NFREQ]; MAX_REF]> = zs
        .iter()
        .zip(&nref)
        .map(|(&z, &n)| if n > 0 { axtb_d4::reference_alpha(z, |_, _| 1.0) } else { [[0.0; NFREQ]; MAX_REF] })
        .collect();
    let mut refcn = vec![Vec::new(); 118];
    for (&z, &n) in zs.iter().zip(&nref) {
        refcn[z - 1] = axtb_d4::data::REFCN[z - 1][..n].to_vec();
    }
    let c6ref = axtb_d4::reference_c6_table(&nref, &alpha);
    let nsp = zs.len();
    let mut c6 = HashMap::new();
    for (isp, &zi) in zs.iter().enumerate() {
        for (jsp, &zj) in zs.iter().enumerate() {
            let block = &c6ref[isp * nsp + jsp];
            let tab: Vec<f64> = (0..nref[isp]).flat_map(|ii| block[ii][..nref[jsp]].iter().copied()).collect();
            c6.insert((zi, zj), tab);
        }
    }
    DispModel { refcn, c6 }
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

#[test]
fn reference_tables_equal_with_mask() {
    for z in 1..=118 {
        let n = nref(z);
        assert_eq!(n, data::D4_REFN[z - 1], "refn Z={z}");
        for ir in 0..n {
            assert_eq!(axtb_d4::data::REFCN[z - 1][ir].to_bits(), data::D4_REFCN[z - 1][ir].to_bits(), "Z={z}");
            assert_eq!(axtb_d4::data::ASCALE[z - 1][ir].to_bits(), data::D4_ASCALE[z - 1][ir].to_bits(), "Z={z}");
            assert_eq!(axtb_d4::data::HCOUNT[z - 1][ir].to_bits(), data::D4_HCOUNT[z - 1][ir].to_bits(), "Z={z}");
            assert_eq!(axtb_d4::data::REFSYS[z - 1][ir] as usize, data::D4_REFSYS[z - 1][ir], "Z={z}");
            // every reference has a secondary system (xtb's loop would fail on 0)
            assert_ne!(data::D4_REFSYS[z - 1][ir], 0, "Z={z}");
            assert_eq!(bits(&axtb_d4::data::ALPHAIW[z - 1][ir]), bits(&data::D4_ALPHAIW[z - 1][ir]), "Z={z}");
        }
    }
    assert_eq!(bits(&axtb_d4::data::SSCALE), bits(&data::D4_SSCALE));
    for s in 0..17 {
        assert_eq!(bits(&axtb_d4::data::SECAIW[s]), bits(&data::D4_SECAIW[s]));
    }
}

#[test]
fn polarizabilities_and_c6_equal() {
    // the polarizabilities as the copied code forms them
    for z in 1..=118 {
        let ia = z - 1;
        let new = axtb_d4::reference_alpha(z, |_, _| 1.0);
        assert_eq!(nref(z), data::D4_REFN[ia]);
        for j in 0..data::D4_REFN[ia] {
            let is = data::D4_REFSYS[ia][j] - 1;
            for f in 0..23 {
                let sec = data::D4_SSCALE[is] * data::D4_SECAIW[is][f];
                let old =
                    (data::D4_ASCALE[ia][j] * (data::D4_ALPHAIW[ia][j][f] - data::D4_HCOUNT[ia][j] * sec)).max(0.0);
                assert_eq!(new[j][f].to_bits(), old.to_bits(), "Z={z} ref {j} freq {f}");
            }
        }
    }
    // the whole model: every element of GFN-FF at once, and a few subsets (other species order)
    let all: Vec<usize> = (1..=103).collect();
    for numbers in [all, vec![8, 1, 1], vec![86, 87, 6, 1, 103, 26], vec![55, 92, 17]] {
        let old = DispModel::new(&numbers);
        let new = new_disp_model(&numbers);
        assert_eq!(old.refcn.len(), new.refcn.len());
        for (a, b) in old.refcn.iter().zip(&new.refcn) {
            assert_eq!(bits(a), bits(b));
        }
        assert_eq!(old.c6.len(), new.c6.len());
        for (key, tab) in &old.c6 {
            assert_eq!(bits(tab), bits(&new.c6[key]), "C6 of {key:?}");
        }
    }
}

#[test]
fn r4_over_r2_equal_with_mask() {
    for z in 1..=118 {
        let masked = if (95..=111).contains(&z) { 0.0 } else { axtb_d4::data::R4_OVER_R2[z - 1] };
        assert_eq!(masked.to_bits(), data::R4_OVER_R2[z - 1].to_bits(), "Z={z}");
        let old = (0.5 * (data::R4_OVER_R2[z - 1] * (z as f64).sqrt())).sqrt();
        assert_eq!(sqrt_z_r4_over_r2(z).to_bits(), old.to_bits(), "Z={z}");
    }
    // xtb's table differs from dftd4's exactly there
    for z in 95..=111 {
        assert_ne!(axtb_d4::data::R4_OVER_R2[z - 1], 0.0, "Z={z}");
    }
}
