# SPDX-License-Identifier: LGPL-3.0-or-later
"""qccodec codec for axtb: GFN2-xTB, GFN-FF and g-xTB (v1.1, 2.0.1) energies, gradients
and Hessians, in-process.

Installed with axtb as the ``qccodec.codecs`` entry point ``axtb``, so qccompute finds it
without any change to qccodec::

    from qccompute import compute
    from qcdata import CalcType, Model, ProgramInput, Structure

    inp = ProgramInput(program="axtb", calctype=CalcType.gradient,
                       structure=Structure.open("mol.xyz"), model=Model(method="gfn2"))
    out = compute(inp)

Inputs map to ``axtb.Calculator`` as follows (the README's "Use" section lists every setting):

- ``model.method``: ``gfn2`` (GFN2-xTB), ``gfnff`` (GFN-FF), ``gxtb-v1.1`` or ``gxtb-v2.0.1``
  (g-xTB), any spelling ``axtb.canonical_method`` accepts (case and anything but letters and
  digits are ignored). Models have no basis set.
- ``structure.charge`` is the total charge and ``structure.multiplicity - 1`` the number of
  unpaired electrons (GFN-FF: no effect). An odd electron count with multiplicity 1 is
  rejected rather than silently changed.
- ``keywords`` (all optional; ``None`` or absent means the method's default, which the codec
  does not repeat). They are the ``axtb.Calculator`` arguments and the analysis settings of
  its ``hessian``, checked per method; an unknown keyword or a wrong type raises
  ``EncoderError``:

  - every method: ``threads`` (default 1; declared to qccompute's thread budget),
    ``conventions``, ``hessian_method``, ``hessian_step`` and the vibrational analysis of
    Hessians: ``vibrations`` (``False`` skips it), ``analysis_conventions``, ``masses``,
    ``thermo`` (``False`` skips the thermochemistry), ``temperature`` (K, a number or a list;
    results at the last), ``pressure``, ``rotor_cutoff``, ``imaginary_cutoff``,
    ``frequency_scale``, ``symmetry_number``, ``symmetry_threshold``. The Hessian and analysis
    keywords apply to ``hessian`` calculations only, so that one keyword set serves an
    optimizer's gradients and its Hessians;
  - SCF methods (GFN2-xTB, g-xTB): ``accuracy``, ``etemp``, ``max_iter``,
    ``hessian_accuracy``, ``molden`` (``True``/``"spherical"`` or ``"cartesian"``: a Molden file
    of the converged orbitals in the results' files as ``molden.input``), ``bond_orders``
    (``extras['bond_orders']``) and ``guess`` (set by ``propagate_wfn``, below);
  - GFN2-xTB: ``mixer``, ``dispersion`` (``False`` for validation), ``alpb`` or ``gbsa`` (a
    solvent) with ``solvation_state``;
  - g-xTB in implicit solvent (the solvation composite E = E(g-xTB) + [E(GFN2-xTB + ALPB/GBSA)
    - E(GFN2-xTB)]): ``solvation_correction="gfn2"`` with ``alpb`` or ``gbsa``,
    ``solvation_state``, ``correction_conventions`` and ``mixer`` (the GFN2 halves'); ``alpb``
    or ``gbsa`` alone are refused with ``axtb.gxtb_solvent_hint``'s explanation;
  - GFN-FF: ``fragment_charges`` (one per fragment, in the order of their first atoms, summing
    to the total charge) and ``topology_positions`` (bohr, N x 3: the structure is evaluated
    with the topology of that one).
- ``cmdline_args`` and ``files`` are not accepted.

Calculation types: ``energy``, ``gradient``, ``hessian`` (with the single point at the same
geometry and its vibrational analysis, as ``axtb.Calculator.hessian``). Results are in the
input's frame and atomic units: ``energy``, ``gradient``, ``hessian``; ``atomic_charges``
(Mulliken; GFN-FF: EEQ); the dipole moment (e·bohr) in ``scf_dipole_moment`` (GFN-FF:
``extras['dipole_moment']``). Hessians also fill ``freqs_wavenumber``,
``normal_modes_cartesian`` and ``gibbs_free_energy`` (energy + quasi-RRHO G(RRHO), xtb's
``TOTAL FREE ENERGY``), with ``extras['vibrations']`` (reduced masses, force constants, IR
intensities, number of imaginary modes, dipole derivatives), ``extras['thermo']``,
``extras['hessian_method']`` and ``extras['hessian_max_asymmetry']``. ``extras`` also hold
``conventions`` (canonical), ``energy_terms``, ``eigensolver`` (SCF methods), the composite's
``composite``, ``components``, ``parts`` (and ``hessian_parts``), GFN-FF's ``topology``,
``n_hbonds`` and ``n_xbonds``. A failed calculation raises ``CalculationError`` with axtb's
message (an ``axtb.InputError`` or ``axtb.ScfNotConvergedError`` inside).

Every SCF method (GFN2-xTB, g-xTB, the composite's g-xTB part) also fills:

- ``wavefunction``: ``scf_eigenvalues_a``/``_b`` and ``scf_occupations_a``/``_b`` (Eh; a closed
  shell's orbitals for both spins with half its occupations; GFN2-xTB's open shells one set of
  orbitals with the alpha and beta occupations; g-xTB's open shells the orbitals of each spin
  channel), and ``calcinfo_natoms``, ``calcinfo_nbasis``, ``calcinfo_nmo``, ``calcinfo_nalpha``,
  ``calcinfo_nbeta`` (the electron counts only when they are integers);
- ``extras['homo_lumo_gap']`` (Eh; ``None`` without a LUMO), ``extras['quadrupole']`` (the
  molecular quadrupole about the origin, xx, xy, yy, xz, yz, zz in e·bohr²; qcdata has no field
  for it), ``extras['scf_iterations']``, ``extras['scf_guess']`` (where the SCF started:
  ``"eeq"``/``"sad"``, ``"eeq-bc"`` or ``"previous"``; ``scf_guess_note`` says why a given
  wavefunction was not used) and, for GFN2-xTB, ``extras['mixer_used']`` and
  ``extras['scf_attempts']``.

Wavefunction reuse: with qccompute's ``propagate_wfn=True`` (geomeTRIC and Sella children)
every step starts its SCF from the previous step's converged wavefunction, in every
conventions (the program conventions then reproduce the programs' restart: g-xTB's
``xtbrestart``/``gxtbrestart``, xtb's in-run restarts). The wavefunctions stay in the
process's memory (the 8 most recent); a result carries only their id,
``extras['wavefunction_id']`` (also the file ``axtb.wavefunction`` qccompute collects), which
``propagate_wfn`` puts into the next input's keyword ``guess``. A ``guess`` this process does
not hold, or that belongs to another calculation (method, atoms, charge, spin, conventions,
solvent), starts from the method's own guess. Without ``propagate_wfn`` every call is
independent (``guess`` absent or ``"fresh"``).
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Callable

from qccodec.codec import Codec
from qccodec.exceptions import CalculationError, EncoderError
from qccodec.native import PythonInput, PythonOutput
from qcdata import CalcType, ProgramInput, ProgramOutput, Provenance, SinglePointData, Wavefunction

from . import _axtb

PROGRAM = "axtb"
GXTB_METHODS = frozenset({"gxtb-v1.1", "gxtb-v2.0.1"})
# the vibrational analysis of Hessians (every method)
VIB_KEYWORDS = frozenset(
    {"vibrations", "analysis_conventions", "masses", "thermo", "temperature", "pressure", "rotor_cutoff",
     "imaginary_cutoff", "frequency_scale", "symmetry_number", "symmetry_threshold"}
)
COMMON_KEYWORDS = frozenset({"threads", "conventions", "hessian_method", "hessian_step"}) | VIB_KEYWORDS
SCF_KEYWORDS = COMMON_KEYWORDS | {"accuracy", "etemp", "max_iter", "hessian_accuracy", "molden", "bond_orders", "guess"}
KEYWORDS = {
    "gfn2": SCF_KEYWORDS | {"mixer", "dispersion", "alpb", "gbsa", "solvation_state"},
    "gfnff": COMMON_KEYWORDS | {"fragment_charges", "topology_positions"},
    "gxtb": SCF_KEYWORDS,
    # g-xTB in solvent: the solvation composite
    "composite": SCF_KEYWORDS | {"solvation_correction", "correction_conventions", "alpb", "gbsa", "solvation_state",
                                 "mixer"},
}
# keywords that are not `axtb.Calculator` arguments
HESSIAN_ARGS = {"hessian_method": "method", "hessian_step": "step", "hessian_accuracy": "accuracy"}
CALL_KEYWORDS = VIB_KEYWORDS | set(HESSIAN_ARGS) | {"molden", "bond_orders", "guess"}
# converged wavefunctions of this process's calculations, for `propagate_wfn` (most recent last)
WAVEFUNCTION_PREFIX = "axtb-wfn:"
MAX_WAVEFUNCTIONS = 8
# written into the calculation directory: the id of the wavefunction kept in memory
WAVEFUNCTION_FILE = "axtb.wavefunction"
_WAVEFUNCTIONS: OrderedDict[str, tuple[tuple[Any, ...], Any]] = OrderedDict()
_WAVEFUNCTIONS_LOCK = threading.Lock()
# file name of the Molden file in the results (as `xtb --molden` names it)
MOLDEN_FILE = "molden.input"


def _number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _integer(v: Any) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _numbers(v: Any) -> bool:
    try:
        import numpy as np

        a = np.asarray(v)
    except (TypeError, ValueError):
        return False
    return a.ndim == 1 and (a.size == 0 or np.issubdtype(a.dtype, np.number)) and a.dtype != bool


def _rows3(v: Any) -> bool:
    try:
        import numpy as np

        a = np.asarray(v)
    except (TypeError, ValueError):
        return False
    return a.ndim == 2 and a.shape[1] == 3 and np.issubdtype(a.dtype, np.number) and a.dtype != bool


# the type of every keyword: (description, check)
TYPES: dict[str, tuple[str, Callable[[Any], bool]]] = {
    **{k: ("a number", _number) for k in ("accuracy", "etemp", "hessian_step", "hessian_accuracy", "pressure",
                                          "rotor_cutoff", "imaginary_cutoff", "frequency_scale",
                                          "symmetry_threshold")},
    **{k: ("an integer", _integer) for k in ("max_iter", "threads", "symmetry_number")},
    **{k: ("a string", lambda v: isinstance(v, str)) for k in ("conventions", "hessian_method", "mixer", "alpb", "gbsa",
                                                               "solvation_state", "solvation_correction",
                                                               "correction_conventions", "analysis_conventions",
                                                               "guess")},
    **{k: ("True or False", lambda v: isinstance(v, bool)) for k in ("bond_orders", "dispersion", "vibrations",
                                                                    "thermo")},
    "molden": ("True, False, 'spherical' or 'cartesian'", lambda v: isinstance(v, bool) or v in ("spherical",
                                                                                               "cartesian")),
    "temperature": ("a number or a list of numbers", lambda v: _number(v) or _numbers(v)),
    "masses": ("a list of numbers", _numbers),
    "fragment_charges": ("a list of numbers", _numbers),
    "topology_positions": ("N x 3 numbers (bohr)", _rows3),
}


def _kind(method: str, keywords: dict[str, Any]) -> str:
    """The keyword set of a method: `gfn2`, `gfnff`, `gxtb` or `composite` (g-xTB in solvent)."""
    if method in GXTB_METHODS:
        return "composite" if keywords.get("solvation_correction") is not None else "gxtb"
    return method


def encode(inp: ProgramInput) -> PythonInput:
    """Validate the input and package one axtb call."""
    if inp.model is None:
        raise EncoderError("axtb requires a model, e.g. {'method': 'gfn2'}.")
    if inp.model.basis:
        raise EncoderError(f"axtb models have no basis set; remove basis {inp.model.basis!r}.")
    try:
        method = _axtb.canonical_method(inp.model.method)
    except ValueError as e:
        raise EncoderError(str(e)) from None
    if inp.cmdline_args or inp.files:
        raise EncoderError("axtb runs in Python; cmdline_args and files are unused.")
    keywords = {k: v for k, v in inp.keywords.items() if v is not None}
    kind = _kind(method, keywords)
    if kind == "gxtb" and {"alpb", "gbsa", "solvation_state", "correction_conventions"} & set(keywords):
        raise EncoderError(_axtb.gxtb_solvent_hint(method) + ".")
    if "solvation_correction" in keywords and method not in GXTB_METHODS:
        raise EncoderError("solvation_correction is for g-xTB methods; GFN2-xTB has its own ALPB/GBSA (alpb or gbsa "
                           "alone).")
    if kind == "composite" and not {"alpb", "gbsa"} & set(keywords):
        raise EncoderError("solvation_correction needs a solvent: the keyword alpb or gbsa (e.g. alpb='water').")
    allowed = KEYWORDS[kind]
    unknown = sorted(set(keywords) - allowed)
    if unknown:
        raise EncoderError(f"Unknown axtb keywords {unknown} for {method}; supported: {sorted(allowed)}.")
    for k, v in keywords.items():
        what, ok = TYPES[k]
        if not ok(v):
            raise EncoderError(f"axtb keyword {k} must be {what}, not {v!r}.")
    threads = keywords.get("threads", 1)
    if threads < 1:
        raise EncoderError("threads must be at least 1.")
    guess = keywords.get("guess")
    if not (guess is None or guess == "fresh" or guess.startswith(WAVEFUNCTION_PREFIX)):
        raise EncoderError(
            f"guess must be 'fresh' (the method's own start) or a wavefunction id from a previous axtb result "
            f"(extras['wavefunction_id'], as qccompute's propagate_wfn=True passes it), not {guess!r}."
        )
    structure = inp.require_structure()
    return PythonInput(
        function=run_axtb,
        kwargs={
            "method": method,
            "calctype": inp.calctype.value,
            "numbers": list(structure.atomic_numbers),
            "positions": structure.geometry.tolist(),
            "charge": float(structure.charge),
            "uhf": int(structure.multiplicity) - 1,
            "keywords": keywords,
        },
        threads=threads,
        wavefunction_files=() if method == "gfnff" else (WAVEFUNCTION_FILE,),
    )


def _plain(value: Any) -> Any:
    """Numpy arrays and scalars as JSON-ready Python values."""
    import numpy as np

    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def run_axtb(
    *, method: str, calctype: str, numbers: list[int], positions: list[list[float]], charge: float, uhf: int,
    keywords: dict[str, Any],
) -> dict[str, Any]:
    """One axtb calculation; returns a snapshot in atomic units."""
    import numpy as np

    import axtb

    pos = np.asarray(positions, dtype=float)
    # the method's settings exactly as given; the rest are arguments of the call
    settings = {k: v for k, v in keywords.items() if k not in CALL_KEYWORDS}
    calc = axtb.Calculator(method, numbers, pos, charge=charge, uhf=uhf, **settings)
    scf = method != "gfnff"
    # a previous calculation's wavefunction (qccompute's propagate_wfn); None: the method's start
    key = (calc.method, tuple(int(z) for z in numbers), charge, uhf, calc.conventions,
           keywords.get("alpb"), keywords.get("gbsa"))
    note = _start_from(calc, keywords.get("guess"), key) if scf else None
    call = {k: keywords[k] for k in ("molden", "bond_orders") if k in keywords}
    if calctype == CalcType.hessian.value:
        call.update({HESSIAN_ARGS.get(k, k): v for k, v in keywords.items() if k in VIB_KEYWORDS or k in HESSIAN_ARGS})
        r = calc.hessian(pos, **call)
    else:
        r = calc.singlepoint(pos, gradient=calctype == CalcType.gradient.value, **call)
    out: dict[str, Any] = {"version": r["version"], "energy": float(r["energy"])}
    if "gradient" in r and calctype != CalcType.energy.value:
        out["gradient"] = np.asarray(r["gradient"]).tolist()
    out["atomic_charges"] = np.asarray(r["charges"]).tolist()
    extras: dict[str, Any] = {"conventions": r["conventions"], "energy_terms": dict(r["terms"])}
    out["extras"] = extras
    if "hessian" in r:
        out["hessian"] = np.asarray(r["hessian"]).tolist()
        extras.update(hessian_max_asymmetry=float(r["hessian_max_asymmetry"]), hessian_method=r["hessian_method"])
        out.update(_vibrations(r))
    for k in ("composite", "components", "parts", "hessian_parts", "topology"):
        if k in r:
            extras[k] = _plain(r[k])
    if not scf:
        extras.update(dipole_moment=np.asarray(r["dipole"]).tolist(), n_hbonds=int(r["n_hbonds"]),
                      n_xbonds=int(r["n_xbonds"]))
        return out
    out["scf_dipole_moment"] = np.asarray(r["dipole"]).tolist()
    extras.update(scf_iterations=int(r["iterations"]), eigensolver=r["eigensolver"], scf_guess=r["guess"],
                  homo_lumo_gap=r["homo_lumo_gap"], quadrupole=np.asarray(r["quadrupole"]).tolist())
    if "mixer_used" in r:
        extras.update(mixer_used=r["mixer_used"], scf_attempts=_plain(r["scf_attempts"]))
    if note:
        extras["scf_guess_note"] = note
    if "bond_orders" in r:
        extras["bond_orders"] = np.asarray(r["bond_orders"]).tolist()
    out["wavefunction"] = {
        "scf_eigenvalues_a": np.asarray(r["orbital_energies_alpha"]).tolist(),
        "scf_eigenvalues_b": np.asarray(r["orbital_energies_beta"]).tolist(),
        "scf_occupations_a": np.asarray(r["occupations_alpha"]).tolist(),
        "scf_occupations_b": np.asarray(r["occupations_beta"]).tolist(),
    }
    nalpha, nbeta = float(np.sum(r["occupations_alpha"])), float(np.sum(r["occupations_beta"]))
    out["calcinfo"] = {"calcinfo_natoms": len(numbers), "calcinfo_nbasis": len(r["orbital_energies_alpha"]),
                       "calcinfo_nmo": len(r["orbital_energies_alpha"])}
    if abs(nalpha - round(nalpha)) < 1e-6 and abs(nbeta - round(nbeta)) < 1e-6:
        out["calcinfo"].update(calcinfo_nalpha=round(nalpha), calcinfo_nbeta=round(nbeta))
    state = calc.guess_state()
    if state is not None:
        wid = _keep_wavefunction(key, state)
        extras["wavefunction_id"] = wid
        # qccompute collects it as the calculation's wavefunction file (collect_wfn)
        Path(WAVEFUNCTION_FILE).write_text(wid + "\n")
    if "molden" in r:
        out["files"] = {MOLDEN_FILE: r["molden"]}
    return out


def _keep_wavefunction(key: tuple[Any, ...], state: Any) -> str:
    """Keep a converged wavefunction in this process (the most recent MAX_WAVEFUNCTIONS); its id."""
    wid = WAVEFUNCTION_PREFIX + uuid.uuid4().hex
    with _WAVEFUNCTIONS_LOCK:
        _WAVEFUNCTIONS[wid] = (key, state)
        while len(_WAVEFUNCTIONS) > MAX_WAVEFUNCTIONS:
            _WAVEFUNCTIONS.popitem(last=False)
    return wid


def _start_from(calc: Any, guess: str | None, key: tuple[Any, ...]) -> str | None:
    """Start `calc` from the wavefunction `guess` names, if this process still has it and it
    belongs to the same calculation; otherwise the calculation starts from the method's own
    guess and the returned note says why."""
    if guess is None or guess == "fresh":
        return None
    with _WAVEFUNCTIONS_LOCK:
        entry = _WAVEFUNCTIONS.get(guess)
    if entry is None:
        return f"wavefunction {guess} is not in this process (other process, or more than {MAX_WAVEFUNCTIONS} ago)"
    if entry[0] != key:
        return f"wavefunction {guess} belongs to another calculation (method, atoms, charge, spin, conventions, solvent)"
    calc.set_guess_state(entry[1])
    return None


def propagate_wfn(output: ProgramOutput, inp: ProgramInput) -> ProgramInput:
    """Start the next calculation's SCF from a previous output's converged wavefunction.

    qccompute calls this for the children of an optimizer (geomeTRIC, Sella) with
    ``propagate_wfn=True``. The wavefunction stays in this process's memory (the most recent
    ``MAX_WAVEFUNCTIONS``); the output carries only its id (``extras['wavefunction_id']``), which
    becomes the next input's keyword ``guess``. An output without one leaves the input as it is.
    """
    wid = (output.results.extras or {}).get("wavefunction_id") if output.results is not None else None
    if not wid:
        return inp
    values = inp.model_dump()
    values["keywords"]["guess"] = wid
    return ProgramInput.model_validate(values)


def _vibrations(r: dict[str, Any]) -> dict[str, Any]:
    """Snapshot entries of a Hessian response's vibrational analysis."""
    out: dict[str, Any] = {}
    vib: dict[str, Any] = {"dipole_derivatives": _plain(r.get("dipole_derivatives"))}
    if "frequencies" in r:
        out["frequencies"] = _plain(r["frequencies"])
        out["normal_modes"] = _plain(r["normal_modes"])
        vib.update(
            conventions=r["vibrational_conventions"], reduced_masses=_plain(r["reduced_masses"]),
            force_constants=_plain(r["force_constants"]), ir_intensities=_plain(r.get("ir_intensities")),
            n_imaginary=int(r["n_imaginary"]), linear=bool(r["linear"]), masses=_plain(r["masses"]),
        )
        if "thermo" in r:
            out["thermo"] = _plain(r["thermo"])
    out["vibrations"] = vib
    return out


def decode(inp: ProgramInput, out: PythonOutput) -> SinglePointData:
    """Turn the snapshot into qcdata results; the calculation type decides the fields."""
    snapshot = out.value or {}
    provenance = Provenance(program=PROGRAM, program_version=snapshot.get("version", _axtb.__version__))
    if out.error is not None or not snapshot:
        reason = f"{type(out.error).__name__}: {out.error}" if out.error is not None else "no results were returned"
        raise CalculationError(f"axtb failed: {reason}", data=SinglePointData(provenance=provenance))
    fields: dict[str, Any] = {
        "energy": snapshot["energy"],
        "atomic_charges": snapshot.get("atomic_charges"),
        "scf_dipole_moment": snapshot.get("scf_dipole_moment"),
        "extras": snapshot.get("extras", {}),
    }
    if snapshot.get("wavefunction") is not None:
        fields["wavefunction"] = Wavefunction(**snapshot["wavefunction"])
    fields.update(snapshot.get("calcinfo", {}))
    if inp.calctype in (CalcType.gradient, CalcType.hessian):
        fields["gradient"] = snapshot.get("gradient")
    if inp.calctype == CalcType.hessian:
        fields["hessian"] = snapshot.get("hessian")
        fields["extras"] = {**fields["extras"], "vibrations": snapshot.get("vibrations", {})}
        if snapshot.get("frequencies") is not None:
            fields["freqs_wavenumber"] = snapshot["frequencies"]
            fields["normal_modes_cartesian"] = snapshot["normal_modes"] or None
        thermo = snapshot.get("thermo")
        if thermo is not None:
            fields["gibbs_free_energy"] = thermo.get("free_energy")
            fields["extras"]["thermo"] = thermo
    if snapshot.get("files"):
        fields["files"] = dict(snapshot["files"])
    return SinglePointData(provenance=provenance, **fields)


codec = Codec(PROGRAM, calctypes={CalcType.energy, CalcType.gradient, CalcType.hessian}, encode=encode, decode=decode,
              propagate_wfn=propagate_wfn)
