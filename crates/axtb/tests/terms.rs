//! A term written outside the crate with the public API of `axtb::terms` only: a uniform
//! electric field acting on the molecular dipole, E = -F · (Σ_A q_A R_A + μ_A), added to a
//! GFN2-xTB calculator with `push_term`. Its kernel and second derivatives are the traits'
//! finite-difference defaults, so the analytic and response Hessians exercise them.

use axtb::terms::{Chain, Geometry, MomentTerm, Needs, Potentials, SccMoments, TermEnergy};
use axtb::{Calculator, HessianMethod, HessianOptions, Method, Options};

/// A uniform electric field (atomic units).
struct Field([f64; 3]);

impl MomentTerm for Field {
    type Cache = ();

    fn names(&self) -> &[&'static str] {
        &["field"]
    }

    fn needs(&self) -> Needs {
        Needs { atom: true, dipole: true, ..Needs::default() }
    }

    fn update(&self, _geo: &Geometry<'_>) -> axtb::Result<()> {
        Ok(())
    }

    fn energy(&self, geo: &Geometry<'_>, _c: &(), s: &SccMoments<'_>, e: &mut TermEnergy<'_>) {
        e.pieces[0] = -dot(self.0, dipole(geo, s));
    }

    fn potential(&self, geo: &Geometry<'_>, _c: &(), s: &SccMoments<'_>, v: &mut Potentials) {
        for a in 0..s.qat.len() {
            v.atom[a] -= dot(self.0, geo.mol.positions[a]);
            for k in 0..3 {
                v.dipole[a][k] -= self.0[k];
            }
        }
    }

    fn gradient(&self, _geo: &Geometry<'_>, _c: &(), s: &SccMoments<'_>, chain: &mut Chain) {
        for a in 0..s.qat.len() {
            for k in 0..3 {
                chain.g[a][k] -= s.qat[a] * self.0[k];
            }
        }
    }
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn dipole(geo: &Geometry<'_>, s: &SccMoments<'_>) -> [f64; 3] {
    let mut mu = [0.0; 3];
    for a in 0..s.qat.len() {
        for k in 0..3 {
            mu[k] += s.qat[a] * geo.mol.positions[a][k] + s.dip[a][k];
        }
    }
    mu
}

const FIELD: [f64; 3] = [0.004, -0.002, 0.006];

/// Formaldehyde (bohr), a polar molecule.
fn molecule() -> (Vec<usize>, Vec<[f64; 3]>) {
    let pos = vec![[0.0, 0.0, -1.14], [0.0, 0.0, 1.14], [0.0, 1.77, -2.24], [0.0, -1.77, -2.24]];
    (vec![6, 8, 1, 1], pos)
}

fn calculator(field: Option<[f64; 3]>) -> Calculator {
    let (num, pos) = molecule();
    let mut o = Options::default();
    o.scf.accuracy = 1e-6;
    let mut calc = Calculator::new(Method::Gfn2, &num, &pos, 0.0, None, o).unwrap();
    if let Some(f) = field {
        calc.push_term(Field(f)).unwrap();
    }
    calc
}

#[test]
fn field_energy_and_decomposition() {
    let (_, pos) = molecule();
    let e0 = calculator(None).singlepoint(&pos, false).unwrap();
    let sp = calculator(Some(FIELD)).singlepoint(&pos, false).unwrap();
    let minus = calculator(Some(FIELD.map(|f| -f))).singlepoint(&pos, false).unwrap();
    // the piece is -F · μ of the converged moments
    let piece = sp.terms.other.iter().find(|(n, _)| *n == "field").expect("the field's piece").1;
    assert!((piece + dot(FIELD, sp.dipole)).abs() < 1e-12, "piece {piece} vs -F.mu {}", -dot(FIELD, sp.dipole));
    // the odd part of E(F) is -F · μ(0) to third order in the field
    let odd = 0.5 * (sp.energy - minus.energy);
    assert!((odd + dot(FIELD, e0.dipole)).abs() < 1e-6, "odd part {odd} vs -F.mu0 {}", -dot(FIELD, e0.dipole));
    // and the even part is the polarization energy, negative
    assert!(sp.energy + minus.energy - 2.0 * e0.energy < 0.0);
    assert!(sp.terms.other.len() == 1 && e0.terms.other.is_empty());
}

#[test]
fn field_gradient_against_differences() {
    let (_, pos) = molecule();
    let mut calc = calculator(Some(FIELD));
    let g = calc.singlepoint(&pos, true).unwrap().gradient.unwrap();
    let h = 1e-4;
    for a in 0..pos.len() {
        for k in 0..3 {
            let mut p = pos.clone();
            p[a][k] += h;
            let ep = calc.singlepoint(&p, false).unwrap().energy;
            p[a][k] -= 2.0 * h;
            let em = calc.singlepoint(&p, false).unwrap().energy;
            let fd = (ep - em) / (2.0 * h);
            assert!((fd - g[a][k]).abs() < 1e-6, "atom {a} direction {k}: analytic {} differences {fd}", g[a][k]);
        }
    }
}

#[test]
fn field_hessians_through_the_trait_defaults() {
    let (_, pos) = molecule();
    let n3 = 3 * pos.len();
    let hessian = |method| {
        let opts = HessianOptions { method, accuracy: Some(1e-8), ..HessianOptions::default() };
        calculator(Some(FIELD)).hessian(&pos, &opts).unwrap().hessian
    };
    let analytic = hessian(HessianMethod::Analytic);
    let response = hessian(HessianMethod::Response);
    let numerical = hessian(HessianMethod::Numerical);
    let scale = analytic.iter().fold(0.0f64, |m, x| m.max(x.abs()));
    for i in 0..n3 * n3 {
        assert!(
            (analytic[i] - response[i]).abs() < 1e-5 * scale,
            "H[{i}]: analytic {} response {}",
            analytic[i],
            response[i]
        );
        assert!(
            (analytic[i] - numerical[i]).abs() < 1e-4 * scale,
            "H[{i}]: analytic {} numerical {}",
            analytic[i],
            numerical[i]
        );
    }
    // the field changes the Hessian (through the charges' and dipoles' response)
    let opts = HessianOptions { accuracy: Some(1e-8), ..HessianOptions::default() };
    let h0 = calculator(None).hessian(&pos, &opts).unwrap().hessian;
    assert!(h0.iter().zip(&analytic).any(|(a, b)| (a - b).abs() > 1e-6 * scale));
}
