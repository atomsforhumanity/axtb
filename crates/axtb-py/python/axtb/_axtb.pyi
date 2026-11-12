"""Type stubs of the extension module ``axtb._axtb`` (see the package docs and the README)."""

from typing import Any, Literal, Sequence

import numpy as np
import numpy.typing as npt

__version__: str

ArrayLike2 = npt.ArrayLike
"""Any array-like of shape (N, 3) (or the shape the argument names), converted to float64."""

Result = dict[str, Any]
"""A response: the keys of the CLI's JSON (``program``, ``version``, ``method``,
``conventions``, ``eigensolver``, ``energy``, ``gradient``, ``charges``, ...), arrays as numpy
arrays; a key is present when it was computed."""

class InputError(ValueError):
    """Invalid input; ``kind``: ``"input"``, ``"unsupported_element"`` (with ``z``) or
    ``"topology"``."""

    kind: str

class ScfNotConvergedError(RuntimeError):
    """The SCF did not converge (``kind`` ``"scf_not_converged"``)."""

    kind: str
    iterations: int
    energy_change: float
    residual: float

class ScfState:
    """A converged wavefunction (``Calculator.guess_state()``); opaque."""

    @property
    def method(self) -> str: ...
    @property
    def natoms(self) -> int: ...

class Calculator:
    """A method bound to one molecule; reuses its setup and the previous wavefunction."""

    def __init__(
        self,
        method: str,
        numbers: Sequence[int],
        positions: ArrayLike2,
        *,
        charge: float = 0.0,
        uhf: int | None = None,
        accuracy: float | None = None,
        etemp: float | None = None,
        max_iter: int | None = None,
        threads: int = 1,
        dispersion: bool | None = None,
        conventions: str | None = None,
        alpb: str | None = None,
        gbsa: str | None = None,
        solvation_state: str | None = None,
        mixer: str | None = None,
        solvation_correction: str | None = None,
        correction_conventions: str | None = None,
        fragment_charges: Sequence[float] | None = None,
        topology_positions: ArrayLike2 | None = None,
        guess: str | None = None,
        scf_trace: bool = False,
        broken_symmetry: bool = False,
    ) -> None: ...
    @property
    def method(self) -> str: ...
    @property
    def conventions(self) -> str: ...
    def singlepoint(
        self,
        positions: ArrayLike2,
        gradient: bool = False,
        molden: bool | Literal["spherical", "cartesian"] | None = None,
        bond_orders: bool = False,
    ) -> Result: ...
    def hessian(
        self,
        positions: ArrayLike2,
        step: float | None = None,
        accuracy: float | None = None,
        method: str | None = None,
        *,
        vibrations: bool = True,
        temperature: float | Sequence[float] | None = None,
        pressure: float | None = None,
        rotor_cutoff: float | None = None,
        imaginary_cutoff: float | None = None,
        frequency_scale: float | None = None,
        symmetry_number: int | None = None,
        symmetry_threshold: float | None = None,
        masses: Sequence[float] | None = None,
        thermo: bool = True,
        analysis_conventions: str | None = None,
        molden: bool | Literal["spherical", "cartesian"] | None = None,
        bond_orders: bool = False,
    ) -> Result: ...
    def guess_state(self) -> ScfState | None: ...
    def set_guess_state(self, state: ScfState) -> None: ...
    def reset_guess(self, gxtb_only: bool = False) -> None: ...
    def set_guess(
        self,
        shell_charges: Sequence[float] | np.ndarray,
        atomic_dipoles: ArrayLike2,
        atomic_quadrupoles: ArrayLike2,
    ) -> None: ...

def singlepoint(
    numbers: Sequence[int],
    positions: ArrayLike2,
    *,
    charge: float = 0.0,
    uhf: int | None = None,
    method: str = "gfn2",
    gradient: bool = False,
    accuracy: float | None = None,
    etemp: float | None = None,
    max_iter: int | None = None,
    threads: int = 1,
    dispersion: bool | None = None,
    conventions: str | None = None,
    alpb: str | None = None,
    gbsa: str | None = None,
    solvation_state: str | None = None,
    mixer: str | None = None,
    molden: bool | Literal["spherical", "cartesian"] | None = None,
    solvation_correction: str | None = None,
    correction_conventions: str | None = None,
    fragment_charges: Sequence[float] | None = None,
    topology_positions: ArrayLike2 | None = None,
    bond_orders: bool = False,
    scf_trace: bool = False,
    broken_symmetry: bool = False,
) -> Result: ...
def vibrational_analysis(
    numbers: Sequence[int],
    positions: ArrayLike2,
    hessian: ArrayLike2,
    *,
    dipole_derivatives: ArrayLike2 | None = None,
    energy: float | None = None,
    conventions: str | None = None,
    masses: Sequence[float] | None = None,
    temperature: float | Sequence[float] | None = None,
    pressure: float | None = None,
    rotor_cutoff: float | None = None,
    imaginary_cutoff: float | None = None,
    frequency_scale: float | None = None,
    symmetry_number: int | None = None,
    symmetry_threshold: float | None = None,
    thermo: bool = True,
) -> Result: ...
def canonical_method(name: str) -> str: ...
def gxtb_solvent_hint(method: str) -> str: ...
