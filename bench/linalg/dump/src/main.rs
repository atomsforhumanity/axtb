//! Usage: linalg-dump <xyz> <ncopies> <outdir>
//! Builds a cluster of randomly rotated copies of the molecule on a cubic grid
//! (3 Å gaps) and writes GFN2-xTB H0 and S (column-major f64) plus the dimension.
use axtb::{io::read_xyz, units::AATOAU, Conventions, Molecule};
use std::io::Write;

fn rot(seed: u64) -> [[f64; 3]; 3] {
    // random unit quaternion from a tiny LCG
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let mut u = || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let (u1, u2, u3) = (u(), u(), u());
    let tau = std::f64::consts::TAU;
    let (a, b, c, d) = (
        (1.0 - u1).sqrt() * (tau * u2).sin(),
        (1.0 - u1).sqrt() * (tau * u2).cos(),
        u1.sqrt() * (tau * u3).sin(),
        u1.sqrt() * (tau * u3).cos(),
    );
    let (w, x, y, z) = (a, b, c, d);
    [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let (z0, p0) = read_xyz(&std::fs::read_to_string(&a[1]).unwrap(), AATOAU).unwrap();
    let ncopy: usize = a[2].parse().unwrap();
    let out = std::path::Path::new(&a[3]);
    std::fs::create_dir_all(out).unwrap();
    let nat0 = z0.len();
    let com: [f64; 3] = std::array::from_fn(|k| p0.iter().map(|r| r[k]).sum::<f64>() / nat0 as f64);
    let rmax = p0
        .iter()
        .map(|r| ((r[0] - com[0]).powi(2) + (r[1] - com[1]).powi(2) + (r[2] - com[2]).powi(2)).sqrt())
        .fold(0.0, f64::max);
    let spacing = 2.0 * rmax + 3.0 / 0.529177; // bohr
    let side = (ncopy as f64).cbrt().ceil() as usize;
    let (mut zs, mut ps) = (vec![], vec![]);
    for c in 0..ncopy {
        let (ix, iy, iz) = (c % side, (c / side) % side, c / (side * side));
        let off = [ix as f64 * spacing, iy as f64 * spacing, iz as f64 * spacing];
        let m = rot(c as u64 + 1);
        for (z, r) in z0.iter().zip(&p0) {
            let v = [r[0] - com[0], r[1] - com[1], r[2] - com[2]];
            zs.push(*z);
            ps.push(std::array::from_fn::<f64, 3, _>(|i| m[i][0] * v[0] + m[i][1] * v[1] + m[i][2] * v[2] + off[i]));
        }
    }
    // the cluster geometry too (Å), for timing full axtb runs on the same system
    let mut xyz = format!("{}\n{} copies of {}\n", zs.len(), ncopy, a[1]);
    for (z, r) in zs.iter().zip(&ps) {
        xyz += &format!(
            "{} {:.8} {:.8} {:.8}\n",
            axtb::io::symbol(*z).unwrap(),
            r[0] * axtb::units::AUTOAA,
            r[1] * axtb::units::AUTOAA,
            r[2] * axtb::units::AUTOAA
        );
    }
    std::fs::write(out.join("cluster.xyz"), xyz).unwrap();
    let mol = Molecule::new(&zs, &ps, 0.0, None).unwrap();
    let (h0, s) = axtb::gfn2_core_matrices(&mol, Conventions::Axtb).unwrap();
    let n = h0.nrows();
    for (name, m) in [("h0.bin", &h0), ("s.bin", &s)] {
        let mut f = std::io::BufWriter::new(std::fs::File::create(out.join(name)).unwrap());
        for j in 0..n {
            for i in 0..n {
                f.write_all(&m[(i, j)].to_le_bytes()).unwrap();
            }
        }
    }
    std::fs::write(out.join("dim.txt"), format!("{n}\n")).unwrap();
    eprintln!("nat={} nao={} (per copy {})", zs.len(), n, n / ncopy);
}
