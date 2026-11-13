//! The chain of intermediates: geometry-dependent quantities that several terms differentiate
//! through (coordination numbers, model charges), each with its chain rule applied once at the
//! end of a gradient.

use super::sinks::Geometry;
use crate::molecule::Molecule;
use crate::Result;
use std::sync::Arc;

/// One output of an intermediate registered in a [`ChainLayout`]: its values (one per atom),
/// its slot of dE/dX in a [`Chain`] and its Jacobian in [`Jacobians`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Slot(usize);

impl Slot {
    /// The index of this slot in the layout (outputs counted in registration order).
    pub fn index(self) -> usize {
        self.0
    }
}

/// A geometry-dependent quantity with one value per atom per output, which terms differentiate
/// through: a coordination number, model charges, effective charges that depend on both.
///
/// An intermediate either depends on the positions alone or on the values of intermediates
/// registered before it ([`ChainLayout::register`]); [`chain`](Self::chain) contracts dE/dX
/// into the gradient and into the weights of those intermediates.
pub trait Intermediate: Send + Sync {
    /// Name, unique within a layout ([`ChainLayout::find`]).
    fn name(&self) -> &'static str;

    /// Number of outputs (vectors of one value per atom); 1 by default.
    fn outputs(&self) -> usize {
        1
    }

    /// The outputs at the positions of `mol`, from the values of the intermediates registered
    /// before this one (in `values`).
    fn compute(&self, mol: &Molecule, values: &Values) -> Result<Vec<Vec<f64>>>;

    /// Add `Σ_k Σ_A w[k][A] dX_k,A/dR`: the position dependence to `chain.g`, the dependence on
    /// earlier intermediates to their slots in `chain` (this intermediate's own slots are zero
    /// while it is applied).
    fn chain(&self, geo: &Geometry<'_>, w: &[&[f64]], chain: &mut Chain) -> Result<()>;

    /// The Jacobians dX_k/dR, one row-major `nat x 3N` matrix per output, given the Jacobians of
    /// the intermediates registered before it (`deps`). The default contracts unit weights with
    /// [`chain`](Self::chain), atom by atom, which is right for an intermediate that depends on
    /// the positions alone; one that depends on others overrides it.
    fn jacobian(&self, geo: &Geometry<'_>, _deps: &Jacobians) -> Result<Vec<Vec<f64>>> {
        let nat = geo.mol.nat();
        let n3 = 3 * nat;
        let mut out = Vec::with_capacity(self.outputs());
        for k in 0..self.outputs() {
            let mut jac = vec![0.0; nat * n3];
            let zero = vec![0.0; nat];
            for a in 0..nat {
                let mut unit = vec![0.0; nat];
                unit[a] = 1.0;
                let w: Vec<&[f64]> = (0..self.outputs()).map(|o| if o == k { &unit[..] } else { &zero[..] }).collect();
                let mut c = Chain::new(geo.layout, nat);
                self.chain(geo, &w, &mut c)?;
                jac[a * n3..(a + 1) * n3].copy_from_slice(c.g.as_flattened());
            }
            out.push(jac);
        }
        Ok(out)
    }
}

/// The intermediates of a model, in registration order: an intermediate after those it depends
/// on. [`Chain::finish`] applies them in the reverse order, so each one has received all of its
/// weights when it is applied.
#[derive(Clone, Default)]
pub struct ChainLayout {
    list: Vec<Arc<dyn Intermediate>>,
    /// First slot of each intermediate.
    first: Vec<usize>,
    slots: usize,
}

impl std::fmt::Debug for ChainLayout {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_list().entries(self.list.iter().map(|x| x.name())).finish()
    }
}

impl ChainLayout {
    /// A layout without intermediates.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register `x` after the intermediates it depends on; returns the slots of its outputs.
    pub fn register(&mut self, x: Arc<dyn Intermediate>) -> Vec<Slot> {
        assert!(self.find(x.name()).is_none(), "intermediate {:?} registered twice", x.name());
        let first = self.slots;
        self.slots += x.outputs();
        self.first.push(first);
        self.list.push(x);
        (first..self.slots).map(Slot).collect()
    }

    /// The first output of the intermediate named `name`.
    pub fn find(&self, name: &str) -> Option<Slot> {
        self.list.iter().position(|x| x.name() == name).map(|i| Slot(self.first[i]))
    }

    /// Number of slots (outputs of all intermediates).
    pub fn slots(&self) -> usize {
        self.slots
    }

    /// The intermediate of slot `s` and the output of `s` within it.
    pub fn intermediate(&self, s: Slot) -> (&dyn Intermediate, usize) {
        let i = self.owner(s.0);
        (self.list[i].as_ref(), s.0 - self.first[i])
    }

    fn owner(&self, slot: usize) -> usize {
        assert!(slot < self.slots, "slot {slot} of a layout with {} slots", self.slots);
        self.first.iter().rposition(|&f| f <= slot).expect("slot of a registered intermediate")
    }

    fn range(&self, i: usize) -> std::ops::Range<usize> {
        self.first[i]..self.first[i] + self.list[i].outputs()
    }

    /// The values of every intermediate at the positions of `mol`, in registration order.
    pub fn compute(&self, mol: &Molecule) -> Result<Values> {
        let mut values = Values { v: vec![Vec::new(); self.slots] };
        for (i, x) in self.list.iter().enumerate() {
            let out = x.compute(mol, &values)?;
            assert_eq!(out.len(), x.outputs(), "intermediate {:?}: outputs", x.name());
            for (s, v) in self.range(i).zip(out) {
                values.v[s] = v;
            }
        }
        Ok(values)
    }

    /// The Jacobians of every intermediate at `geo`, in registration order.
    pub fn jacobians(&self, geo: &Geometry<'_>) -> Result<Jacobians> {
        let mut jac = Jacobians { j: vec![Vec::new(); self.slots] };
        for (i, x) in self.list.iter().enumerate() {
            let out = x.jacobian(geo, &jac)?;
            for (s, v) in self.range(i).zip(out) {
                jac.j[s] = v;
            }
        }
        Ok(jac)
    }
}

/// The values of a layout's intermediates at one geometry (one vector per slot, one value per
/// atom).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Values {
    v: Vec<Vec<f64>>,
}

impl Values {
    /// No values (the layout's slots unset).
    pub fn new(layout: &ChainLayout) -> Self {
        Self { v: vec![Vec::new(); layout.slots()] }
    }

    /// The values of slot `s` (empty if not set).
    #[inline]
    pub fn get(&self, s: Slot) -> &[f64] {
        &self.v[s.0]
    }

    /// Set the values of slot `s` (e.g. from a model's own setup).
    pub fn set(&mut self, s: Slot, v: Vec<f64>) {
        self.v[s.0] = v;
    }
}

/// The Jacobians dX/dR of a layout's intermediates at one geometry: per slot, row-major
/// `nat x 3N`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Jacobians {
    j: Vec<Vec<f64>>,
}

impl Jacobians {
    /// No Jacobians (the layout's slots unset).
    pub fn new(layout: &ChainLayout) -> Self {
        Self { j: vec![Vec::new(); layout.slots()] }
    }

    /// The Jacobian of slot `s` (empty if not set).
    #[inline]
    pub fn get(&self, s: Slot) -> &[f64] {
        &self.j[s.0]
    }

    /// Set the Jacobian of slot `s`.
    pub fn set(&mut self, s: Slot, j: Vec<f64>) {
        self.j[s.0] = j;
    }
}

/// The derivatives of an energy before the chain rules of its intermediates: the direct
/// position derivative `g` (Eh/bohr) and dE/dX of every slot of the layout (zero to start,
/// one value per atom; the slots in one vector, so that a chunk's accumulator is two
/// allocations whatever the layout).
#[derive(Clone, Debug, PartialEq)]
pub struct Chain {
    pub g: Vec<[f64; 3]>,
    w: Vec<f64>,
}

impl Chain {
    /// Zero derivatives of `nat` atoms for the slots of `layout`.
    pub fn new(layout: &ChainLayout, nat: usize) -> Self {
        Self { g: vec![[0.0; 3]; nat], w: vec![0.0; layout.slots() * nat] }
    }

    /// Zero derivatives of the same shape (a chunk's accumulator, added back with
    /// [`add`](Self::add)).
    pub fn zeroed_like(&self) -> Self {
        Self { g: vec![[0.0; 3]; self.g.len()], w: vec![0.0; self.w.len()] }
    }

    #[inline]
    fn range(&self, s: Slot) -> std::ops::Range<usize> {
        let nat = self.g.len();
        s.0 * nat..(s.0 + 1) * nat
    }

    /// dE/dX of slot `s`, to add to.
    #[inline]
    pub fn slot(&mut self, s: Slot) -> &mut [f64] {
        let r = self.range(s);
        &mut self.w[r]
    }

    /// The direct gradient and dE/dX of slot `s` together, for code that accumulates both in
    /// one pass.
    #[inline]
    pub fn g_and_slot(&mut self, s: Slot) -> (&mut [[f64; 3]], &mut [f64]) {
        let r = self.range(s);
        (&mut self.g, &mut self.w[r])
    }

    /// The direct gradient and dE/dX of the distinct slots `s` together.
    pub fn g_and_slots<const N: usize>(&mut self, s: [Slot; N]) -> (&mut [[f64; 3]], [&mut [f64]; N]) {
        let nat = self.g.len();
        let mut rest: &mut [f64] = &mut self.w;
        let mut parts: Vec<Option<&mut [f64]>> = Vec::with_capacity(rest.len() / nat.max(1));
        while rest.len() >= nat && nat > 0 {
            let (a, b) = std::mem::take(&mut rest).split_at_mut(nat);
            parts.push(Some(a));
            rest = b;
        }
        let w = s.map(|s| {
            if nat == 0 { Some(&mut [][..]) } else { parts[s.0].take() }.expect("distinct slots of the layout")
        });
        (&mut self.g, w)
    }

    /// dE/dX of slot `s`.
    #[inline]
    pub fn weights(&self, s: Slot) -> &[f64] {
        &self.w[self.range(s)]
    }

    /// Add the derivatives of `o` (every element, in place).
    pub fn add(&mut self, o: &Chain) {
        for (a, b) in self.g.iter_mut().zip(&o.g) {
            for x in 0..3 {
                a[x] += b[x];
            }
        }
        for (x, y) in self.w.iter_mut().zip(&o.w) {
            *x += y;
        }
    }

    /// The gradient: every intermediate applied once, in the reverse registration order, each
    /// pushing its weights into `g` and into the slots of the intermediates it depends on (its
    /// own slots are zeroed while it is applied). An intermediate whose weights are all exactly
    /// zero is skipped (its contribution would be zeros, which can only turn a -0 of `g` into
    /// +0).
    pub fn finish(mut self, geo: &Geometry<'_>) -> Result<Vec<[f64; 3]>> {
        let layout = geo.layout;
        for i in (0..layout.list.len()).rev() {
            let slots: Vec<Slot> = layout.range(i).map(Slot).collect();
            if slots.iter().all(|&s| self.weights(s).iter().all(|&x| x == 0.0)) {
                continue;
            }
            let own: Vec<Vec<f64>> = slots
                .iter()
                .map(|&s| {
                    let w = self.weights(s).to_vec();
                    self.slot(s).fill(0.0);
                    w
                })
                .collect();
            let w: Vec<&[f64]> = own.iter().map(|v| v.as_slice()).collect();
            layout.list[i].chain(geo, &w, &mut self)?;
        }
        Ok(self.g)
    }
}

/// Second derivatives of an energy with respect to the nuclear positions: the Hessian to add
/// to (row-major 3N x 3N) and the Jacobians of the model's intermediates at this geometry.
pub struct Chain2<'a> {
    pub h: &'a mut [f64],
    pub jacobians: &'a Jacobians,
}
