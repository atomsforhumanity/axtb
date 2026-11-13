// SPDX-License-Identifier: LGPL-3.0-or-later
//! A toy model: an intermediate of the positions, a second one that depends on it (two
//! outputs), and a moment term that reads the second and the atomic charges and dipoles.

use super::*;
use crate::molecule::{norm2_sq, sub3, Molecule};
use axtb_base::fd::{self as bfd, Stencil};
use std::sync::Arc;

/// X_A = Σ_B exp(-r_AB² / 4).
struct ToyCn;

impl Intermediate for ToyCn {
    fn name(&self) -> &'static str {
        "toy cn"
    }
    fn compute(&self, mol: &Molecule, _: &Values) -> Result<Vec<Vec<f64>>> {
        let n = mol.nat();
        let mut x = vec![0.0; n];
        for i in 0..n {
            for j in 0..i {
                let e = (-norm2_sq(sub3(mol.positions[i], mol.positions[j])) / 4.0).exp();
                x[i] += e;
                x[j] += e;
            }
        }
        Ok(vec![x])
    }
    fn chain(&self, geo: &Geometry<'_>, w: &[&[f64]], chain: &mut Chain) -> Result<()> {
        let pos = &geo.mol.positions;
        for i in 0..pos.len() {
            for j in 0..i {
                let v = sub3(pos[i], pos[j]);
                let f = (w[0][i] + w[0][j]) * (-norm2_sq(v) / 4.0).exp() * -0.5;
                for k in 0..3 {
                    chain.g[i][k] += f * v[k];
                    chain.g[j][k] -= f * v[k];
                }
            }
        }
        Ok(())
    }
}

/// Y0_A = 2 X_A, Y1_A = X_A² of the toy CN.
struct ToyQ(Slot);

impl Intermediate for ToyQ {
    fn name(&self) -> &'static str {
        "toy q"
    }
    fn outputs(&self) -> usize {
        2
    }
    fn compute(&self, _: &Molecule, values: &Values) -> Result<Vec<Vec<f64>>> {
        let x = values.get(self.0);
        Ok(vec![x.iter().map(|x| 2.0 * x).collect(), x.iter().map(|x| x * x).collect()])
    }
    fn chain(&self, geo: &Geometry<'_>, w: &[&[f64]], chain: &mut Chain) -> Result<()> {
        let x = geo.value(self.0);
        let up = chain.slot(self.0);
        for a in 0..x.len() {
            up[a] += 2.0 * w[0][a] + 2.0 * x[a] * w[1][a];
        }
        Ok(())
    }
    fn jacobian(&self, geo: &Geometry<'_>, deps: &Jacobians) -> Result<Vec<Vec<f64>>> {
        let (x, jx) = (geo.value(self.0), deps.get(self.0));
        let n3 = 3 * x.len();
        let j0 = jx.iter().map(|v| 2.0 * v).collect();
        let j1 = (0..jx.len()).map(|e| 2.0 * x[e / n3] * jx[e]).collect();
        Ok(vec![j0, j1])
    }
}

/// E = Σ_A [α q_A²/2 + β q_A Y0_A + γ μ_A · R_A + δ Y1_A].
struct ToyTerm {
    y0: Slot,
    y1: Slot,
    slots: [Slot; 2],
}

const ALPHA: f64 = 0.7;
const BETA: f64 = -0.3;
const GAMMA: f64 = 0.2;
const DELTA: f64 = 0.05;

impl MomentTerm for ToyTerm {
    type Cache = ();
    fn names(&self) -> &[&'static str] {
        &["toy"]
    }
    fn fixed_names(&self) -> &[&'static str] {
        &["toy fixed"]
    }
    fn needs(&self) -> Needs {
        Needs { atom: true, dipole: true, ..Needs::default() }
    }
    fn slots(&self) -> &[Slot] {
        &self.slots
    }
    fn update(&self, _: &Geometry<'_>) -> Result<()> {
        Ok(())
    }
    fn fixed_energy(&self, geo: &Geometry<'_>, _: &(), e: &mut TermEnergy<'_>, chain: Option<&mut Chain>) {
        e.pieces[0] = geo.value(self.y1).iter().map(|y| DELTA * y).sum();
        if let Some(ch) = chain {
            ch.slot(self.y1).iter_mut().for_each(|w| *w += DELTA);
        }
    }
    fn energy(&self, geo: &Geometry<'_>, _: &(), s: &SccMoments<'_>, e: &mut TermEnergy<'_>) {
        let y0 = geo.value(self.y0);
        e.pieces[0] = (0..s.qat.len())
            .map(|a| {
                let q = s.qat[a];
                let r = geo.mol.positions[a];
                0.5 * ALPHA * q * q + BETA * q * y0[a] + GAMMA * (0..3).map(|k| s.dip[a][k] * r[k]).sum::<f64>()
            })
            .sum();
    }
    fn potential(&self, geo: &Geometry<'_>, _: &(), s: &SccMoments<'_>, v: &mut Potentials) {
        let y0 = geo.value(self.y0);
        for a in 0..s.qat.len() {
            v.atom[a] += ALPHA * s.qat[a] + BETA * y0[a];
            for k in 0..3 {
                v.dipole[a][k] += GAMMA * geo.mol.positions[a][k];
            }
        }
    }
    fn gradient(&self, _: &Geometry<'_>, _: &(), s: &SccMoments<'_>, chain: &mut Chain) {
        for a in 0..s.qat.len() {
            for k in 0..3 {
                chain.g[a][k] += GAMMA * s.dip[a][k];
            }
            chain.slot(self.y0)[a] += BETA * s.qat[a];
        }
    }
}

struct Toy {
    mol: Molecule,
    layout: ChainLayout,
    term: ToyTerm,
    qat: Vec<f64>,
    dip: Vec<[f64; 3]>,
}

fn toy() -> Toy {
    let pos = [[0.0, 0.0, 0.0], [1.4, 0.2, -0.1], [-0.3, 1.7, 0.4], [0.9, -0.8, 1.6]];
    let mol = Molecule::new(&[6, 1, 8, 1], &pos, 0.0, None).unwrap();
    let mut layout = ChainLayout::new();
    let x = layout.register(Arc::new(ToyCn))[0];
    let q = layout.register(Arc::new(ToyQ(x)));
    let term = ToyTerm { y0: q[0], y1: q[1], slots: [q[0], q[1]] };
    let qat = vec![0.2, -0.1, -0.3, 0.2];
    let dip = vec![[0.1, 0.0, -0.2], [0.0, 0.05, 0.0], [0.3, -0.1, 0.1], [-0.05, 0.2, 0.0]];
    Toy { mol, layout, term, qat, dip }
}

impl Toy {
    fn moments(&self) -> SccMoments<'_> {
        SccMoments { qsh: &[], qat: &self.qat, dip: &self.dip, quad: &[[0.0; 6]; 4], mag: None }
    }

    /// Energy and analytic gradient at `mol`.
    fn energy_gradient(&self, mol: &Molecule) -> (f64, Vec<[f64; 3]>) {
        let values = self.layout.compute(mol).unwrap();
        let geo = Geometry { mol, layout: &self.layout, values: &values, parallel: false };
        let (mut p, mut f) = ([0.0], [0.0]);
        let mut chain = Chain::new(&self.layout, mol.nat());
        self.term.fixed_energy(&geo, &(), &mut TermEnergy { pieces: &mut f, atoms: None }, Some(&mut chain));
        self.term.energy(&geo, &(), &self.moments(), &mut TermEnergy { pieces: &mut p, atoms: None });
        self.term.gradient(&geo, &(), &self.moments(), &mut chain);
        (p[0] + f[0], chain.finish(&geo).unwrap())
    }
}

#[test]
fn chain_rules_give_the_gradient() {
    let t = toy();
    let (_, g) = t.energy_gradient(&t.mol);
    for x in 0..3 * t.mol.nat() {
        let fd = bfd::derivative(Stencil::Central4, 1e-3, |d| {
            let mut m = t.mol.clone();
            m.positions[x / 3][x % 3] += d;
            t.energy_gradient(&m).0
        });
        assert!((fd - g[x / 3][x % 3]).abs() < 1e-9, "coordinate {x}: {fd} vs {}", g[x / 3][x % 3]);
    }
}

#[test]
fn jacobians_follow_the_dependencies() {
    let t = toy();
    let values = t.layout.compute(&t.mol).unwrap();
    let geo = Geometry { mol: &t.mol, layout: &t.layout, values: &values, parallel: false };
    let jac = t.layout.jacobians(&geo).unwrap();
    let n3 = 3 * t.mol.nat();
    for out in 0..2 {
        let s = t.term.slots[out];
        for x in 0..n3 {
            let col = bfd::derivative(Stencil::Central4, 1e-3, |d| {
                let mut m = t.mol.clone();
                m.positions[x / 3][x % 3] += d;
                t.layout.compute(&m).unwrap().get(s).to_vec()
            });
            for a in 0..t.mol.nat() {
                assert!((col[a] - jac.get(s)[a * n3 + x]).abs() < 1e-9);
            }
        }
    }
}

#[test]
fn finite_difference_defaults() {
    let t = toy();
    let nat = t.mol.nat();
    let n3 = 3 * nat;
    let values = t.layout.compute(&t.mol).unwrap();
    let geo = Geometry { mol: &t.mol, layout: &t.layout, values: &values, parallel: false };
    let s = t.moments();
    // kernel: α on the charge diagonal of the multipole block, nothing else
    let mut k = Kernel { multipole: vec![0.0; 100 * nat * nat], ..Kernel::default() };
    t.term.kernel(&geo, &(), &s, &mut k);
    for a in 0..10 * nat {
        for b in 0..10 * nat {
            let want = if a == b && a < nat { ALPHA } else { 0.0 };
            assert!((k.multipole[a * 10 * nat + b] - want).abs() < 1e-9);
        }
    }
    // kernel times a vector: α δq on the charges
    let dq = vec![1.0, -0.5, 0.25, 2.0];
    let zero3 = vec![[0.0; 3]; nat];
    let ds = SccMoments { qsh: &[], qat: &dq, dip: &zero3, quad: &[[0.0; 6]; 4], mag: None };
    let mut dv = Potentials::new(0, nat, false);
    t.term.kernel_apply(&geo, &(), &s, &ds, &mut dv);
    for a in 0..nat {
        assert!((dv.atom[a] - ALPHA * dq[a]).abs() < 1e-9);
    }
    // potential derivatives and Hessian against differences of the analytic potential and
    // gradient with a larger step
    let mut pd = PotentialDerivative::new(0, 10 * nat, n3);
    let jac = t.layout.jacobians(&geo).unwrap();
    let mut h = vec![0.0; n3 * n3];
    t.term.second_derivatives(&geo, &(), &s, Some(&mut pd), Some(&mut Chain2 { h: &mut h, jacobians: &jac })).unwrap();
    let jy0 = jac.get(t.term.y0);
    for a in 0..nat {
        for x in 0..n3 {
            assert!((pd.atomic[a * n3 + x] - BETA * jy0[a * n3 + x]).abs() < 1e-7);
            for c in 0..3 {
                let want = if x == 3 * a + c { GAMMA } else { 0.0 };
                assert!((pd.atomic[(nat + 3 * a + c) * n3 + x] - want).abs() < 1e-7);
            }
        }
    }
    for x in 0..n3 {
        let col = bfd::derivative(Stencil::Central4, 1e-3, |d| {
            let mut m = t.mol.clone();
            m.positions[x / 3][x % 3] += d;
            t.energy_gradient(&m).1
        });
        for y in 0..n3 {
            assert!((col[y / 3][y % 3] - h[y * n3 + x]).abs() < 1e-6, "H[{y}][{x}]");
        }
    }
}
