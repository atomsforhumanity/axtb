//! The energy terms of g-xTB 2.0.1 as core terms ([`axtb::terms`]). The setup ([`Setup`]) is
//! the model at one geometry: it holds the terms' caches (`disp`, ...) and calls the terms in
//! the order of the program's arithmetic (its energy sum, its shell-potential fold, its
//! gradient order), which the core leaves to the model. Their energies are totals (g-xTB sums
//! no per-atom energies).
//!
//! [`Setup`]: super::scf::Setup

use super::coulomb::{
    aes_kernel, first_order_g, gamma_kernel, spin_energy, spin_potential, tau_kernel, Coulomb, Shell,
};
use super::disp::RevD4;
use super::gradient::{add_pair, chain_rows};
use super::intermediates::Slots;
use super::repulsion::Repulsion;
use super::{sqrt_cn, Model};
use crate::data_v2::{AES_DIP_SCALE, HUBBARD, HUBBARD_CN, INCREMENT, IPEA, IPEA_CN, SHELL_HUBBARD};
use axtb::molecule::{norm2_sq, sub3};
use axtb::terms::{Chain, Chain2, Geometry, GeometryTerm, MomentTerm, Needs, Potentials, SccMoments, Slot, TermEnergy};
use axtb::Result;

/// The terms of a 2.0.1 model.
#[derive(Clone, Debug)]
pub struct Terms {
    pub increments: Increments,
    pub iso: Isotropic,
    pub aes: Aes,
    pub spin: Spin,
    pub rep: RepulsionTerm,
    pub disp: RevD4Term,
}

impl Terms {
    /// The terms of `model` for the molecule with the SCF-basis shells `shells`
    /// ([`crate::fock::shell_list`]) and the intermediates `slots`.
    pub fn new(model: &Model, shells: &[Shell], slots: &Slots) -> Self {
        Self {
            increments: Increments,
            iso: Isotropic { model: *model, shells: shells.to_vec(), cn: [slots.cn] },
            aes: Aes { model: *model, shells: shells.to_vec(), cn: [slots.cn] },
            spin: Spin { shells: shells.to_vec() },
            rep: RepulsionTerm { model: *model, cn: [slots.cn] },
            disp: RevD4Term { model: *model, cnd4: [slots.cnd4] },
        }
    }
}

/// The atomic energy increments: a constant per element.
#[derive(Clone, Copy, Debug)]
pub struct Increments;

impl GeometryTerm for Increments {
    type Cache = ();

    fn names(&self) -> &[&'static str] {
        &["increments"]
    }

    fn update(&self, _geo: &Geometry<'_>) -> Result<()> {
        Ok(())
    }

    fn energy(&self, geo: &Geometry<'_>, _c: &(), e: &mut TermEnergy<'_>, _chain: Option<&mut Chain>) {
        e.pieces[0] = geo.mol.numbers.iter().map(|&z| INCREMENT[z - 1]).sum();
    }

    fn add_hessian(&self, _geo: &Geometry<'_>, _c: &(), _h: &mut Chain2<'_>) -> Result<()> {
        Ok(())
    }
}

/// revD4 ([`super::disp`]): the two-body energy of the SCF charges (its energy and potential in
/// one pass) and the three-body energy at zero charges (the moment-independent part, once per
/// geometry); both depend on the revD4 coordination number.
#[derive(Clone, Debug)]
pub struct RevD4Term {
    model: Model,
    /// The revD4 CN.
    cnd4: [Slot; 1],
}

impl MomentTerm for RevD4Term {
    type Cache = RevD4;

    fn names(&self) -> &[&'static str] {
        &["dispersion"]
    }

    fn fixed_names(&self) -> &[&'static str] {
        &["dispersion_three_body"]
    }

    fn needs(&self) -> Needs {
        Needs { atom: true, ..Needs::default() }
    }

    fn slots(&self) -> &[Slot] {
        &self.cnd4
    }

    fn update(&self, geo: &Geometry<'_>) -> Result<RevD4> {
        RevD4::with_cn_par(&self.model, geo.mol, geo.value(self.cnd4[0]).to_vec(), geo.parallel)
    }

    /// The three-body energy and, with `chain`, its gradient: positions and dE/dCN (revD4).
    fn fixed_energy(&self, geo: &Geometry<'_>, c: &RevD4, e: &mut TermEnergy<'_>, chain: Option<&mut Chain>) {
        e.pieces[0] = match chain {
            Some(ch) => {
                let (e3, d3) = c.three_body_par(&self.model, geo.mol, Some(&mut ch.g), geo.parallel);
                let dcn = ch.slot(self.cnd4[0]);
                for (w, d) in dcn.iter_mut().zip(d3) {
                    *w += d;
                }
                e3
            }
            None => c.three_body_par(&self.model, geo.mol, None, geo.parallel).0,
        };
    }

    fn energy(&self, geo: &Geometry<'_>, c: &RevD4, s: &SccMoments<'_>, e: &mut TermEnergy<'_>) {
        e.pieces[0] = c.two_body_par(s.qat, geo.parallel).0;
    }

    fn potential(&self, geo: &Geometry<'_>, c: &RevD4, s: &SccMoments<'_>, v: &mut Potentials) {
        let (_, vd) = c.two_body_par(s.qat, geo.parallel);
        add(&mut v.atom, &vd);
    }

    fn energy_and_potential(
        &self,
        geo: &Geometry<'_>,
        c: &RevD4,
        s: &SccMoments<'_>,
        e: &mut TermEnergy<'_>,
        v: &mut Potentials,
    ) {
        let (e2, vd) = c.two_body_par(s.qat, geo.parallel);
        e.pieces[0] = e2;
        add(&mut v.atom, &vd);
    }

    /// The two-body gradient at fixed charges: positions and dE/dCN (revD4).
    fn gradient(&self, geo: &Geometry<'_>, c: &RevD4, s: &SccMoments<'_>, chain: &mut Chain) {
        let d2 = c.two_body_gradient_par(&self.model, geo.mol, s.qat, &mut chain.g, geo.parallel);
        add(chain.slot(self.cnd4[0]), &d2);
    }

    fn kernel_apply(
        &self,
        _geo: &Geometry<'_>,
        c: &RevD4,
        s: &SccMoments<'_>,
        ds: &SccMoments<'_>,
        dv: &mut Potentials,
    ) {
        add(&mut dv.atom, &c.two_body_hessian_vec(s.qat, ds.qat));
    }
}

/// Zero potentials of the moments `t` reads, the other slots empty: a term's potential is zero
/// in every other moment, and the setup's per-term buffers (made per call) stay small.
pub(crate) fn buffer(t: &impl MomentTerm, nsh: usize, nat: usize) -> Potentials {
    let n = t.needs();
    let len = |on: bool, len: usize| if on { len } else { 0 };
    Potentials {
        shell: vec![0.0; len(n.shell, nsh)],
        atom: vec![0.0; len(n.atom, nat)],
        dipole: vec![[0.0; 3]; len(n.dipole, nat)],
        quadrupole: vec![[0.0; 6]; len(n.quadrupole, nat)],
        magnetization: vec![0.0; len(n.magnetization, nsh)],
    }
}

/// `out += x` element by element. 2.0.1's terms compute their potentials and dE/dCN into fresh
/// vectors, as before, and add them to buffers the setup zeroed: the same values, except that a
/// -0 becomes +0. The setup folds each potential buffer into a shell's sum of the exchange,
/// AES, repulsion, dispersion and isotropic potentials, which can differ only in the sign of a
/// zero, and only where every one of them is exactly zero; the dE/dCN of the two- and
/// three-body parts meet in a zeroed slot, where 0 + (a + b) and (0 + a) + b agree for all a, b,
/// signed zeros included, and the revD4 CN's chain rule skips zero weights.
fn add(out: &mut [f64], x: &[f64]) {
    for (o, v) in out.iter_mut().zip(x) {
        *o += v;
    }
}

/// First- to fourth-order isotropic electrostatics ([`super::coulomb`]): the four energies from
/// one pass over the shells, the shell potential, the gradient at fixed charges (kernels and
/// the CN-dependent parameters, dE/dCN of the g-xTB CN), the kernel product. The cache is the
/// setup's [`Coulomb`] (shared with [`Aes`]).
#[derive(Clone, Debug)]
pub struct Isotropic {
    model: Model,
    shells: Vec<Shell>,
    cn: [Slot; 1],
}

impl MomentTerm for Isotropic {
    type Cache = Coulomb;

    fn names(&self) -> &[&'static str] {
        &["first_order", "second_order", "third_order", "fourth_order"]
    }

    fn needs(&self) -> Needs {
        Needs { shell: true, ..Needs::default() }
    }

    fn slots(&self) -> &[Slot] {
        &self.cn
    }

    fn update(&self, geo: &Geometry<'_>) -> Result<Coulomb> {
        Ok(Coulomb::new_par(&self.model, geo.mol, &self.shells, geo.value(self.cn[0]), geo.parallel))
    }

    fn energy(&self, geo: &Geometry<'_>, c: &Coulomb, s: &SccMoments<'_>, e: &mut TermEnergy<'_>) {
        e.pieces.copy_from_slice(&c.isotropic_energies(s.qsh, geo.parallel));
    }

    fn potential(&self, geo: &Geometry<'_>, c: &Coulomb, s: &SccMoments<'_>, v: &mut Potentials) {
        add(&mut v.shell, &c.isotropic_potential(s.qsh, geo.parallel));
    }

    fn gradient(&self, geo: &Geometry<'_>, c: &Coulomb, s: &SccMoments<'_>, chain: &mut Chain) {
        add_isotropic(geo, self.cn[0], c, s.qsh, chain);
    }

    fn kernel_apply(
        &self,
        _geo: &Geometry<'_>,
        c: &Coulomb,
        s: &SccMoments<'_>,
        ds: &SccMoments<'_>,
        dv: &mut Potentials,
    ) {
        add(&mut dv.shell, &c.isotropic_hessian_vec(s.qsh, ds.qsh));
    }
}

/// Anisotropic electrostatics with CN-scaled atomic dipoles ([`super::coulomb`]). Its gradient
/// needs dE/d(c d), the potential of the scaled dipoles: the trait's `gradient` computes it
/// again; the setup, which has it from the potential pass at the same state, uses
/// [`Aes::potential_scaled`] and [`Aes::gradient_with`] instead (the same bits). The cache is
/// the setup's [`Coulomb`].
#[derive(Clone, Debug)]
pub struct Aes {
    model: Model,
    shells: Vec<Shell>,
    cn: [Slot; 1],
}

impl Aes {
    /// [`MomentTerm::potential`], returning dE/d(c d) (the dipole potential before the scaling).
    pub fn potential_scaled(
        &self,
        geo: &Geometry<'_>,
        c: &Coulomb,
        s: &SccMoments<'_>,
        v: &mut Potentials,
    ) -> Vec<[f64; 3]> {
        let (vs, vd_scaled, vq) = c.aes_potentials_raw(geo.mol, s.qat, s.dip, s.quad, geo.parallel);
        add(&mut v.atom, &vs);
        add(v.dipole.as_flattened_mut(), c.scaled_dipoles(&vd_scaled).as_flattened());
        add(v.quadrupole.as_flattened_mut(), vq.as_flattened());
        vd_scaled
    }

    /// [`MomentTerm::gradient`] with dE/d(c d) of the moments `s` given.
    pub fn gradient_with(
        &self,
        geo: &Geometry<'_>,
        c: &Coulomb,
        s: &SccMoments<'_>,
        vd_scaled: &[[f64; 3]],
        chain: &mut Chain,
    ) {
        add_aes(geo, &self.model, self.cn[0], c, s.qat, s.dip, s.quad, vd_scaled, chain);
    }
}

impl MomentTerm for Aes {
    type Cache = Coulomb;

    fn names(&self) -> &[&'static str] {
        &["aes"]
    }

    fn needs(&self) -> Needs {
        Needs { atom: true, dipole: true, quadrupole: true, ..Needs::default() }
    }

    fn slots(&self) -> &[Slot] {
        &self.cn
    }

    fn update(&self, geo: &Geometry<'_>) -> Result<Coulomb> {
        Ok(Coulomb::new_par(&self.model, geo.mol, &self.shells, geo.value(self.cn[0]), geo.parallel))
    }

    fn energy(&self, geo: &Geometry<'_>, c: &Coulomb, s: &SccMoments<'_>, e: &mut TermEnergy<'_>) {
        e.pieces[0] = c.aes_energy(geo.mol, s.qat, s.dip, s.quad, geo.parallel);
    }

    fn potential(&self, geo: &Geometry<'_>, c: &Coulomb, s: &SccMoments<'_>, v: &mut Potentials) {
        self.potential_scaled(geo, c, s, v);
    }

    fn gradient(&self, geo: &Geometry<'_>, c: &Coulomb, s: &SccMoments<'_>, chain: &mut Chain) {
        let (_, vd_scaled, _) = c.aes_potentials_raw(geo.mol, s.qat, s.dip, s.quad, geo.parallel);
        self.gradient_with(geo, c, s, &vd_scaled, chain);
    }

    /// The energy is bilinear in the moments: the potentials of δs (one thread: the response
    /// runs its perturbations in parallel).
    fn kernel_apply(
        &self,
        geo: &Geometry<'_>,
        c: &Coulomb,
        _s: &SccMoments<'_>,
        ds: &SccMoments<'_>,
        dv: &mut Potentials,
    ) {
        let (vs, vd, vq) = c.aes_potentials(geo.mol, ds.qat, ds.dip, ds.quad, false);
        add(&mut dv.atom, &vs);
        add(dv.dipole.as_flattened_mut(), vd.as_flattened());
        add(dv.quadrupole.as_flattened_mut(), vq.as_flattened());
    }
}

/// Spin polarization of the shell magnetizations (open shells; no geometry dependence).
#[derive(Clone, Debug)]
pub struct Spin {
    shells: Vec<Shell>,
}

impl MomentTerm for Spin {
    type Cache = ();

    fn names(&self) -> &[&'static str] {
        &["spin"]
    }

    fn needs(&self) -> Needs {
        Needs { magnetization: true, ..Needs::default() }
    }

    fn update(&self, _geo: &Geometry<'_>) -> Result<()> {
        Ok(())
    }

    /// Zero for a closed shell (no magnetizations).
    fn energy(&self, geo: &Geometry<'_>, _c: &(), s: &SccMoments<'_>, e: &mut TermEnergy<'_>) {
        e.pieces[0] = s.mag.map_or(0.0, |m| spin_energy(geo.mol, &self.shells, m));
    }

    fn potential(&self, geo: &Geometry<'_>, _c: &(), s: &SccMoments<'_>, v: &mut Potentials) {
        if let Some(m) = s.mag {
            add(&mut v.magnetization, &spin_potential(geo.mol, &self.shells, m));
        }
    }

    fn gradient(&self, _geo: &Geometry<'_>, _c: &(), _s: &SccMoments<'_>, _chain: &mut Chain) {}

    fn kernel_apply(&self, geo: &Geometry<'_>, _c: &(), _s: &SccMoments<'_>, ds: &SccMoments<'_>, dv: &mut Potentials) {
        if let Some(m) = ds.mag {
            add(&mut dv.magnetization, &spin_potential(geo.mol, &self.shells, m));
        }
    }

    fn second_derivatives(
        &self,
        _geo: &Geometry<'_>,
        _c: &(),
        _s: &SccMoments<'_>,
        _dv: Option<&mut axtb::terms::PotentialDerivative>,
        _h: Option<&mut Chain2<'_>>,
    ) -> Result<()> {
        Ok(())
    }
}

/// The self-consistent repulsion ([`super::repulsion`]): the geometry part M_AB depends on the
/// g-xTB CN; energy, potential, gradient (positions and dE/dCN) and kernel product at the SCF
/// charges.
#[derive(Clone, Debug)]
pub struct RepulsionTerm {
    model: Model,
    cn: [Slot; 1],
}

impl MomentTerm for RepulsionTerm {
    type Cache = Repulsion;

    fn names(&self) -> &[&'static str] {
        &["repulsion"]
    }

    fn needs(&self) -> Needs {
        Needs { atom: true, ..Needs::default() }
    }

    fn slots(&self) -> &[Slot] {
        &self.cn
    }

    fn update(&self, geo: &Geometry<'_>) -> Result<Repulsion> {
        Ok(Repulsion::new_par(&self.model, geo.mol, geo.value(self.cn[0]), geo.parallel))
    }

    fn energy(&self, geo: &Geometry<'_>, c: &Repulsion, s: &SccMoments<'_>, e: &mut TermEnergy<'_>) {
        e.pieces[0] = c.energy_par(s.qat, geo.parallel);
    }

    fn potential(&self, geo: &Geometry<'_>, c: &Repulsion, s: &SccMoments<'_>, v: &mut Potentials) {
        add(&mut v.atom, &c.potential_par(s.qat, geo.parallel));
    }

    fn gradient(&self, geo: &Geometry<'_>, c: &Repulsion, s: &SccMoments<'_>, chain: &mut Chain) {
        let cn = geo.value(self.cn[0]);
        let (g, dcn) = chain.g_and_slot(self.cn[0]);
        c.gradient_par(&self.model, geo.mol, cn, s.qat, g, dcn, geo.parallel);
    }

    fn kernel_apply(
        &self,
        _geo: &Geometry<'_>,
        c: &Repulsion,
        _s: &SccMoments<'_>,
        ds: &SccMoments<'_>,
        dv: &mut Potentials,
    ) {
        add(&mut dv.atom, &c.potential_delta(ds.qat));
    }
}

/// First- to fourth-order terms at fixed shell charges: kernels (distances, Hubbard
/// parameters through the CN) and the CN-dependent first-order potentials.
pub(crate) fn add_isotropic(geo: &Geometry<'_>, cn_slot: Slot, c: &Coulomb, q: &[f64], ch: &mut Chain) {
    let mol = geo.mol;
    let cn_values = geo.value(cn_slot);
    let sh = &c.shells;
    let ns = sh.len();
    let qa = c.atom_charges(q);
    let cn = cn_slot;
    let mut du = vec![0.0; ns];
    let dcn = ch.slot(cn);
    for (i, &(iat, ish, _)) in sh.iter().enumerate() {
        let z = mol.numbers[iat] - 1;
        let df = sqrt_cn(cn_values[iat]).1;
        du[i] = SHELL_HUBBARD[z][ish] * HUBBARD[z] * HUBBARD_CN[z] * df;
        let g = first_order_g(qa[iat]).0;
        dcn[iat] += IPEA[z][ish] * IPEA_CN[z] * df * g * q[i];
    }
    let u = &c.u;
    let acc = chain_rows(ns, ch, geo.parallel, |i, ch| {
        let (g, dcn) = ch.g_and_slot(cn);
        for j in 0..=i {
            let (ia, ja) = (sh[i].0, sh[j].0);
            let same = ia == ja;
            // dE/dgamma_ij and dE/dtau_ij of the symmetric pair
            let (wg, wt) = if i == j {
                (0.5 * q[i] * q[i], q[i] * q[i] * q[i] * c.gam3[i] / 3.0)
            } else {
                let mut wg = q[i] * q[j];
                if !same {
                    wg += q[i] * c.rref[j] + q[j] * c.rref[i];
                }
                (wg, (q[i] * q[i] * c.gam3[i] * q[j] + q[j] * q[j] * c.gam3[j] * q[i]) / 3.0)
            };
            let r = if same { 0.0 } else { norm2_sq(sub3(mol.positions[ia], mol.positions[ja])).sqrt() };
            let eta = 2.0 / (1.0 / u[i] + 1.0 / u[j]);
            let (_, dgdr, dgde) = gamma_kernel(r, eta);
            let (_, dtdr, dtde) = tau_kernel(r, eta, same);
            let dede = wg * dgde + wt * dtde;
            let s2 = (u[i] + u[j]) * (u[i] + u[j]);
            dcn[ia] += dede * 2.0 * u[j] * u[j] / s2 * du[i];
            dcn[ja] += dede * 2.0 * u[i] * u[i] / s2 * du[j];
            if !same {
                let dedr = wg * dgdr + wt * dtdr;
                let v = sub3(mol.positions[ia], mol.positions[ja]);
                add_pair(g, ia, ja, dedr / r, v);
            }
        }
    });
    ch.add(&acc);
}

/// Anisotropic electrostatics at fixed charges and moments: kernels (explicit distance
/// vectors) and the CN-dependent dipole scaling (`vd_scaled` = dE/d(c d)).
#[allow(clippy::too_many_arguments)]
pub(crate) fn add_aes(
    geo: &Geometry<'_>,
    model: &Model,
    cn_slot: Slot,
    c: &Coulomb,
    qa: &[f64],
    dip: &[[f64; 3]],
    qp: &[[f64; 6]],
    vd_scaled: &[[f64; 3]],
    ch: &mut Chain,
) {
    let mol = geo.mol;
    let cn_values = geo.value(cn_slot);
    let n = mol.nat();
    let d = c.scaled_dipoles(dip);
    let dcn = ch.slot(cn_slot);
    for i in 0..n {
        let z = mol.numbers[i] - 1;
        let dc = AES_DIP_SCALE[z] * sqrt_cn(cn_values[i]).1;
        dcn[i] += dc * (0..3).map(|x| dip[i][x] * vd_scaled[i][x]).sum::<f64>();
    }
    let dot = |a: [f64; 3], b: [f64; 3]| a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    let qm = |q: &[f64; 6]| [[q[0], q[1], q[3]], [q[1], q[2], q[4]], [q[3], q[4], q[5]]];
    let mv = |m: &[[f64; 3]; 3], v: [f64; 3]| [dot(m[0], v), dot(m[1], v), dot(m[2], v)];
    let q = qa;
    let acc = chain_rows(n, ch, geo.parallel, |i, ch| {
        for j in 0..i {
            let r = sub3(mol.positions[j], mol.positions[i]);
            let r2 = dot(r, r);
            let rr = r2.sqrt();
            let r0 = model.rvdw(mol.numbers[i], mol.numbers[j]);
            let (g3, dg3) = aes_kernel(0, rr, r0);
            let (g5, dg5) = aes_kernel(1, rr, r0);
            let (g7, dg7) = aes_kernel(2, rr, r0);
            let (g9, dg9) = aes_kernel(3, rr, r0);
            let (qi, qj) = (qm(&qp[i]), qm(&qp[j]));
            let (di, dj) = (d[i], d[j]);
            let (qir, qjr) = (mv(&qi, r), mv(&qj, r));
            let (rqir, rqjr) = (dot(r, qir), dot(r, qjr));
            let (dir, djr) = (dot(di, r), dot(dj, r));
            let e_qd = q[j] * dir - q[i] * djr;
            let grad_qd = [0, 1, 2].map(|c| q[j] * di[c] - q[i] * dj[c]);
            let e_qq = q[j] * rqir + q[i] * rqjr;
            let grad_qq = [0, 1, 2].map(|c| 2.0 * (q[j] * qir[c] + q[i] * qjr[c]));
            let e_dd = dot(di, dj) * r2 - 3.0 * dir * djr;
            let grad_dd = [0, 1, 2].map(|c| 2.0 * dot(di, dj) * r[c] - 3.0 * (djr * di[c] + dir * dj[c]));
            let (qidj, qjdi) = (mv(&qi, dj), mv(&qj, di));
            let (rqidj, rqjdi) = (dot(r, qidj), dot(r, qjdi));
            let e_dq = 5.0 * rqir * djr - 2.0 * rqidj * r2 - (5.0 * dir * rqjr - 2.0 * rqjdi * r2);
            let grad_dq = [0, 1, 2].map(|c| {
                5.0 * (2.0 * qir[c] * djr + rqir * dj[c])
                    - 2.0 * (qidj[c] * r2 + 2.0 * rqidj * r[c])
                    - 5.0 * (di[c] * rqjr + 2.0 * dir * qjr[c])
                    + 2.0 * (qjdi[c] * r2 + 2.0 * rqjdi * r[c])
            });
            let mut tr = 0.0;
            for a in 0..3 {
                for b in 0..3 {
                    tr += qi[a][b] * qj[a][b];
                }
            }
            let qiqjr = mv(&qi, qjr);
            let qjqir = mv(&qj, qir);
            let qrqr = dot(qir, qjr);
            let e_q2 = 35.0 * rqir * rqjr - 20.0 * qrqr * r2 + 3.0 * tr * r2 * r2;
            let grad_q2 = [0, 1, 2].map(|c| {
                35.0 * (2.0 * qir[c] * rqjr + 2.0 * qjr[c] * rqir)
                    - 20.0 * ((qiqjr[c] + qjqir[c]) * r2 + 2.0 * qrqr * r[c])
                    + 12.0 * tr * r2 * r[c]
            });
            let dedr = e_qd * dg3 + (e_qq + e_dd) * dg5 - e_dq * dg7 + e_q2 * dg9 / 3.0;
            let mut gv = [0.0; 3];
            for c in 0..3 {
                gv[c] = g3 * grad_qd[c] + g5 * (grad_qq[c] + grad_dd[c]) - g7 * grad_dq[c]
                    + g9 * grad_q2[c] / 3.0
                    + dedr * r[c] / rr;
            }
            add_pair(&mut ch.g, j, i, 1.0, gv);
        }
    });
    ch.add(&acc);
}

#[cfg(test)]
mod tests {
    use super::super::scf::Setup;
    use crate::scf::{Conventions, ScfOptions};
    use axtb::terms::{Chain, MomentTerm, Potentials};
    use axtb::Molecule;

    /// The bits of a chain: the direct gradient and the weights of every intermediate.
    fn bits(setup: &Setup, c: &Chain) -> Vec<u64> {
        let s = setup.slots;
        let mut out: Vec<u64> = c.g.iter().flatten().map(|x| x.to_bits()).collect();
        for slot in [s.qeff, s.qeff0, s.cn, s.cnb, s.cnd4, s.eeq] {
            out.extend(c.weights(slot).iter().map(|x| x.to_bits()));
        }
        out
    }

    /// The setup's AES gradient (with dE/d(c d) from its potential pass) and the trait's (which
    /// computes it again) agree bit for bit, at the converged state of a polar molecule, on one
    /// and on several threads.
    #[test]
    fn aes_gradient_paths_agree() {
        let pos = [[0.0, 0.0, -1.14], [0.0, 0.0, 1.14], [0.0, 1.77, -2.24], [0.0, -1.77, -2.24]];
        let mol = Molecule::new(&[6, 8, 1, 1], &pos, 0.0, None).unwrap();
        let mut setup = Setup::new(&mol, super::Model::v2_0_1(Conventions::Axtb)).unwrap();
        let r = setup.scf(&ScfOptions::default()).unwrap();
        for threads in [1, 2] {
            setup.set_threads(threads).unwrap();
            let (tg, s, aes) = (setup.term_geometry(), r.state.moments(), &setup.terms.aes);
            let mut v = Potentials::new(setup.shells.len(), mol.nat(), false);
            let vd_scaled = aes.potential_scaled(&tg, &setup.coulomb, &s, &mut v);
            let mut w = Potentials::new(setup.shells.len(), mol.nat(), false);
            aes.potential(&tg, &setup.coulomb, &s, &mut w);
            assert_eq!(v, w);
            let (mut a, mut b) = (Chain::new(&setup.layout, mol.nat()), Chain::new(&setup.layout, mol.nat()));
            aes.gradient_with(&tg, &setup.coulomb, &s, &vd_scaled, &mut a);
            aes.gradient(&tg, &setup.coulomb, &s, &mut b);
            assert_eq!(bits(&setup, &a), bits(&setup, &b));
            assert!(a.g.iter().flatten().any(|&x| x != 0.0));
        }
    }
}
