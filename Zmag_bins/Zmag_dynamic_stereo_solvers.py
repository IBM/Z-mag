"""Factory and thin wrappers for stereographic **dynamic** solvers.

* :class:`StereoDemagSolver` — :class:`~Zmag_bins.Zmag_dynamic_stereo_demag.Z_sim_dynamic_stereo_demag`
* :class:`StereoNoDemagSolver` — :class:`~Zmag_bins.Zmag_dynamic_stereo_nodemag.Z_sim_dynamic_stereo_nodemag`

:func:`create_stereo_solver` is the recommended entry point. Wrappers forward attribute
access to :attr:`StereoSolverBase.impl`."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any

from Zmag_bins.Zmag_dynamic_stereo_demag import Z_sim_dynamic_stereo_demag
from Zmag_bins.Zmag_dynamic_stereo_nodemag import Z_sim_dynamic_stereo_nodemag

__all__ = [
    "StereoSolverBase",
    "StereoDemagSolver",
    "StereoNoDemagSolver",
    "create_stereo_solver",
    "Z_sim_dynamic_stereo_demag",
    "Z_sim_dynamic_stereo_nodemag",
]


class StereoSolverBase(ABC):
    """
    Abstract facade over dynamic stereographic solver implementations.

    Deep-copies ``args_init_sim`` and forwards attributes to :attr:`impl`.
    """

    def __init__(self, args_init_sim: dict[str, Any]):
        """
        Deep-copy initialization dict and construct the concrete solver.

        Parameters
        ----------
        args_init_sim : dict
            Simulation setup dictionary for the underlying class.
        """
        self._args_init_sim = deepcopy(args_init_sim)
        self._solver = self._build_solver(self._args_init_sim)

    @abstractmethod
    def _build_solver(self, args_init_sim: dict[str, Any]):
        """Instantiate the model-specific solver implementation."""
        raise NotImplementedError

    @property
    def impl(self):
        """Underlying ``Z_sim_dynamic_stereo_*`` instance."""
        return self._solver

    def __getattr__(self, name: str):
        """Delegate attribute access to :attr:`impl`."""
        return getattr(self._solver, name)


class StereoDemagSolver(StereoSolverBase):
    """
    Wrapper for :class:`~Zmag_bins.Zmag_dynamic_stereo_demag.Z_sim_dynamic_stereo_demag`.

    Requires ``demag`` in ``energy_terms``.
    """

    def _build_solver(self, args_init_sim: dict[str, Any]):
        terms = list(args_init_sim.get("energy_terms", []))
        if "demag" not in terms:
            raise ValueError("StereoDemagSolver requires 'demag' in energy_terms.")
        return Z_sim_dynamic_stereo_demag(args_init_sim)


class StereoNoDemagSolver(StereoSolverBase):
    """
    Wrapper for :class:`~Zmag_bins.Zmag_dynamic_stereo_nodemag.Z_sim_dynamic_stereo_nodemag`.

    Strips ``demag`` from ``energy_terms`` if present.
    """

    def _build_solver(self, args_init_sim: dict[str, Any]):
        return Z_sim_dynamic_stereo_nodemag(args_init_sim)


def create_stereo_solver(args_init_sim: dict[str, Any], model: str = "demag") -> StereoSolverBase:
    """
    Build a stereographic dynamic solver wrapper.

    Parameters
    ----------
    args_init_sim : dict
        Simulation setup dictionary.
    model : str
        ``"demag"`` or ``"nodemag"``.

    Returns
    -------
    StereoSolverBase
    Delegating wrapper around the concrete solver.

    Raises
    ------
    ValueError
        If ``model`` is unrecognized.
    """

    mode = model.strip().lower()
    if mode == "demag":
        return StereoDemagSolver(args_init_sim)
    if mode == "nodemag":
        return StereoNoDemagSolver(args_init_sim)
    raise ValueError(f"Unknown model='{model}'. Use 'demag' or 'nodemag'.")
