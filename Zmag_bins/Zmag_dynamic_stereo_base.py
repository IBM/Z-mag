r"""Shared core for stereographic **dynamic** micromagnetics.

This module factors all model-independent logic used by the demag and nodemag dynamic
solvers. The Landau--Lifshitz--Gilbert (LLG) equation is discretized in stereographic chart
coordinates :math:`(q_A,q_B)` with a two-chart atlas so that :math:`|\mathbf{m}|=1` is
intrinsic to the parametrization.

Module-level API
----------------
* :data:`DEFAULT_CFL_SETTINGS` — defaults merged by :func:`apply_cfl_settings`
* :data:`DEFAULT_CONVERGENCE_SETTINGS` — defaults merged by :func:`merge_convergence_settings`
* :func:`save_checkpoint_atlas_state` / :func:`restore_checkpoint_atlas_state` — npz atlas + BDF history I/O

Class :class:`Z_sim_dynamic_stereo_base`
----------------------------------------
Hosts :meth:`Z_sim_dynamic_stereo_base.solve_problem`, atlas masks and weights, BDF time
differencing, adaptive timestep control (CFL cap, embedded BDF error estimate, thermodynamic
caps), acceptance gates, continuation ramps, and shared field writers.

The per-timestep **acceptance pipeline** is orchestrated by
:meth:`Z_sim_dynamic_stereo_base.solve_problem` and factored into private helpers
(see class docstring *Code map*). Algorithm details and gate labels (A)–(F) are in
:doc:`theory_algorithms`.

Gate → method (quick reference)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
======  ==========================================
Label   Method
======  ==========================================
(A)     :meth:`~Z_sim_dynamic_stereo_base.Z_sim_dynamic_stereo_base._run_nonlinear_solve`
(B)     :meth:`~Z_sim_dynamic_stereo_base.Z_sim_dynamic_stereo_base._validate_algebraic_solve`
(C)     :meth:`~Z_sim_dynamic_stereo_base.Z_sim_dynamic_stereo_base._refresh_attempt_energies`
(D)     :meth:`~Z_sim_dynamic_stereo_base.Z_sim_dynamic_stereo_base._apply_residual_gate`
(E)     :meth:`~Z_sim_dynamic_stereo_base.Z_sim_dynamic_stereo_base._apply_thermodynamic_gate`
(F)     :meth:`~Z_sim_dynamic_stereo_base.Z_sim_dynamic_stereo_base._apply_closure_gate`
======  ==========================================

One attempt runs :meth:`~Z_sim_dynamic_stereo_base.Z_sim_dynamic_stereo_base._run_accept_attempt`;
rejections call :meth:`~Z_sim_dynamic_stereo_base.Z_sim_dynamic_stereo_base._handle_step_rejection`;
accepted steps call :meth:`~Z_sim_dynamic_stereo_base.Z_sim_dynamic_stereo_base._finalize_accepted_step`.

Subclasses supply mixed FEM spaces, SNES residuals, and magnetostatics coupling."""

from __future__ import annotations

from typing import Any
from dataclasses import dataclass
from functools import partial
import re
import os
import csv

import numpy as np
import time
import ufl
from basix.ufl import mixed_element
import adios4dolfinx as a4dx
from dolfinx import fem, io, la, mesh, cpp
from mpi4py import MPI
from petsc4py import PETSc

from Zmag_bins.tools import compute_Hzee_coil

DEFAULT_CONVERGENCE_SETTINGS: dict[str, float | int] = {
    "physical_tol_mdot_rms": 1e-4,
    "physical_tol_edot": 1e-5,
    "steady_steps": 5,
    "stagnation_tol": 1e-4,
    "stagnation_steps": 5,
}

DEFAULT_CFL_SETTINGS: dict[str, float | int] = {
    "CFL": 0.5,
    "dt_min": 1e-7,
    "dt_max": 1e-2,
    "smooth_factor": 0.6,
    "max_growth": 1.2,
    "min_shrink": 0.5,
    "error_rel_tol": 1e-2,
    "error_abs_tol": 1e-4,
    "error_headroom": 0.90,
    "max_retry_thermo": 20,
}

# Stereographic LLG maps S^2 into R^3; magnetization and magnetic fields are 3-vectors
# independent of the mesh topological dimension (e.g. thin-film meshes are often 3D).
SPIN_DIM = 3


def parent_dof_map_from_collapse(collapse_map: Any) -> np.ndarray:
    """Flatten a DOLFINx ``collapse()`` parent DOF map to a 1-D ``int32`` array.

    DOLFINx 0.9 returned a flat sequence of parent DOF indices. DOLFINx 0.10+
    returns a list of arrays (one per collapsed block). Indexing the raw list
    with ``map[:n_owned]`` is wrong for the latter and yields invalid fieldsplit
    index sets (PETSc SEGV in the LU factor).
    """
    if isinstance(collapse_map, (list, tuple)):
        if len(collapse_map) == 0:
            return np.array([], dtype=np.int32)
        first = collapse_map[0]
        if isinstance(first, (np.ndarray, list, tuple)):
            return np.concatenate(
                [np.asarray(block, dtype=np.int32).ravel() for block in collapse_map]
            )
        return np.asarray(collapse_map, dtype=np.int32).ravel()
    return np.asarray(collapse_map, dtype=np.int32).ravel()


def validate_spin_vector(vec: Any, name: str) -> np.ndarray:
    """Return a length-:data:`SPIN_DIM` finite vector or raise ``ValueError``."""
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    if arr.size != SPIN_DIM:
        raise ValueError(f"{name} must be a length-{SPIN_DIM} vector, got length {arr.size}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def init_stereographic_chart_basis(
    args_init_sim: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build orthonormal stereographic projection pole and tangent basis in R^3.

    Parameters
    ----------
    args_init_sim : dict
        May contain ``projection_pole`` (length-3, nonzero).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(pole, e1, e2)`` unit vectors.
    """
    pole_input = args_init_sim.get("projection_pole", (0, -1, 0))
    pole = validate_spin_vector(pole_input, "projection_pole")
    norm = np.linalg.norm(pole)
    if norm < 1e-12:
        raise ValueError("projection_pole must be nonzero")
    pole = pole / norm
    seed = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(np.dot(seed, pole)) > 0.9:
        seed = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    e1 = seed - np.dot(seed, pole) * pole
    e1_norm = np.linalg.norm(e1)
    if e1_norm < 1e-12:
        raise ValueError(
            "projection_pole is degenerate; cannot construct stereographic tangent basis"
        )
    e1 = e1 / e1_norm
    e2 = np.cross(pole, e1)
    e2 = e2 / np.linalg.norm(e2)
    return pole, e1, e2


def merge_convergence_settings(args_init_dyn: dict[str, Any]) -> dict[str, float | int]:
    """
    Merge user ``convergence_settings`` with :data:`DEFAULT_CONVERGENCE_SETTINGS`.

    Parameters
    ----------
    args_init_dyn : dict
        Dynamic configuration dictionary.

    Returns
    -------
    dict
    Merged settings (``physical_tol_mdot_rms``, ``physical_tol_edot``, ``steady_steps``, ...).

    Raises
    ------
    ValueError
        If ``convergence_settings`` is present but not a ``dict``.
    """
    user = args_init_dyn.get("convergence_settings", {})
    if user is None:
        user = {}
    if not isinstance(user, dict):
        raise ValueError("[init_dynamic] convergence_settings must be a dict when provided.")
    return {**DEFAULT_CONVERGENCE_SETTINGS, **user}


def apply_cfl_settings(solver: Any, args_init_dyn: dict[str, Any]) -> None:
    """
    Attach adaptive timestep attributes on a solver instance.

    Merges ``args_init_dyn['cfl_settings']`` with :data:`DEFAULT_CFL_SETTINGS` for use in
    :meth:`Z_sim_dynamic_stereo_base._update_timestep_cfl`.

    Parameters
    ----------
    solver : Z_sim_dynamic_stereo_base
        Solver during ``init_dynamic`` setup.
    args_init_dyn : dict
        Must contain a ``cfl_settings`` sub-dictionary.

    Raises
    ------
    ValueError
        If ``cfl_settings`` is missing or not a ``dict``.
    """
    user_cfl = args_init_dyn.get("cfl_settings")
    if user_cfl is None:
        raise ValueError("[init_dynamic]: 'cfl_settings' is required")
    if not isinstance(user_cfl, dict):
        raise ValueError("[init_dynamic]: 'cfl_settings' must be a dict.")
    cfl_merged = {**DEFAULT_CFL_SETTINGS, **user_cfl}
    solver.CFL = cfl_merged["CFL"]
    solver.dt_min = cfl_merged["dt_min"]
    solver.dt_max = cfl_merged["dt_max"]
    solver.smooth_factor = cfl_merged["smooth_factor"]
    solver.max_growth = cfl_merged["max_growth"]
    solver.min_shrink = cfl_merged["min_shrink"]
    solver.error_rel_tol = cfl_merged["error_rel_tol"]
    solver.error_abs_tol = cfl_merged["error_abs_tol"]
    solver.error_headroom = cfl_merged["error_headroom"]
    solver.max_retry_thermo = int(cfl_merged["max_retry_thermo"])


def convergence_tolerances(convergence_settings: dict[str, Any]) -> tuple[float, float, int]:
    """
    Extract steady-state stopping thresholds.

    Parameters
    ----------
    convergence_settings : dict
        Merged convergence dictionary.

    Returns
    -------
    tuple[float, float, int]
    ``(physical_tol_mdot_rms, physical_tol_edot, steady_steps)``.
    """
    conv = convergence_settings
    return (
        float(conv["physical_tol_mdot_rms"]),
        float(conv["physical_tol_edot"]),
        int(conv.get("steady_steps", 5)),
    )


def save_checkpoint_atlas_state(solver: Any, base: str) -> None:
    """
    Write stereographic atlas state and BDF history to npz files.

    Writes ``base_qA.npz``, ``base_qB.npz``, optional BDF history files, and atlas metadata.
    """
    solver._save_func_npz(solver.qA, base + "_qA.npz")
    solver._save_func_npz(solver.qA_prev, base + "_qA_prev.npz")
    solver._save_func_npz(solver.qA_prev_prev, base + "_qA_prev_prev.npz")
    solver._save_func_npz(solver.qB, base + "_qB.npz")
    solver._save_func_npz(solver.qB_prev, base + "_qB_prev.npz")
    solver._save_func_npz(solver.qB_prev_prev, base + "_qB_prev_prev.npz")
    solver._save_func_npz(solver._node_chart_func, base + "_node_chart.npz")
    solver._save_func_npz(solver.chart_ind, base + "_chart_ind.npz")


def _load_optional_q_history(solver: Any, path: str, target: fem.Function) -> bool:
    try:
        solver._load_func_npz(target, path)
        target.x.scatter_forward()
        return True
    except (FileNotFoundError, OSError):
        return False


def _checkpoint_base_path(filename: str) -> str:
    return filename[:-3] if filename.endswith(".bp") else filename


def _state_monitor_get_or_nan(state_monitor: dict[str, Any], key: str) -> float:
    """Fetch a float-like state value, mapping None → NaN."""
    v = None if state_monitor is None else state_monitor.get(key, None)
    if v is None:
        return np.nan
    return float(v)


def _state_monitor_get_or_default_int(
    state_monitor: dict[str, Any], key: str, default: int
) -> int:
    """Fetch an int state value, mapping None → default."""
    v = None if state_monitor is None else state_monitor.get(key, None)
    if v is None:
        return int(default)
    return int(v)


def save_checkpoint_time_state(solver: Any, base: str) -> None:
    """
    Write time-integration/controller state to ``base + '_time_state.npz'``.

    The dynamic checkpoint test expects the solver to be able to *continue* the
    same adaptive/BDF/thermo controllers after restoring a checkpoint. The atlas
    state alone is insufficient because gate budgets (EMAs) and BDF coefficients
    depend on solver history.
    """
    sm = getattr(solver, "state_monitor", None) or {}

    # BDF state for qdot reconstruction (depends on the dt ratio through _update_bdf_coefficients()).
    bdf_active_order = int(getattr(solver, "_bdf_active_order", 1))
    bdf_steps_since_restart = int(getattr(solver, "_bdf_steps_since_restart", 0))
    if hasattr(solver, "_dt_prev_value"):
        dt_prev_value = float(solver._dt_prev_value)
    else:
        # Fallback: best-effort from current dt constant.
        dt_prev_value = float(getattr(solver.dt, "value", solver.dt))  # type: ignore[attr-defined]

    # EMA/budget inputs for thermo and residual gates.
    thermo_defect_ema = _state_monitor_get_or_nan(sm, "thermo_defect_ema")
    thermo_phys_delta_ema = _state_monitor_get_or_nan(sm, "thermo_phys_delta_ema")
    accepted_residual_ema = _state_monitor_get_or_nan(sm, "accepted_residual_ema")
    accepted_residual_ema_slow = _state_monitor_get_or_nan(sm, "accepted_residual_ema_slow")

    # CFL/controller inputs.
    be_recovery_steps_remaining = _state_monitor_get_or_default_int(
        sm, "be_recovery_steps_remaining", default=0
    )
    last_temporal_error_norm = _state_monitor_get_or_nan(sm, "last_temporal_error_norm")

    thermo_dt_cap = _state_monitor_get_or_nan(sm, "thermo_dt_cap")
    thermo_dt_cap_cooldown_steps = _state_monitor_get_or_default_int(
        sm, "thermo_dt_cap_cooldown_steps", default=0
    )

    nonthermo_dt_cap = _state_monitor_get_or_nan(sm, "nonthermo_dt_cap")
    nonthermo_dt_cap_cooldown_steps = _state_monitor_get_or_default_int(
        sm, "nonthermo_dt_cap_cooldown_steps", default=0
    )

    continuation_transition_dt_cap = _state_monitor_get_or_nan(sm, "continuation_transition_dt_cap")
    continuation_transition_cooldown_steps = _state_monitor_get_or_default_int(
        sm, "continuation_transition_cooldown_steps", default=0
    )

    nonthermo_clean_accepted_steps = _state_monitor_get_or_default_int(
        sm, "nonthermo_clean_accepted_steps", default=0
    )
    nonlinear_retry_cooldown_steps = _state_monitor_get_or_default_int(
        sm, "nonlinear_retry_cooldown_steps", default=0
    )

    # Outer-split convergence cap depends on prior phi-defect magnitude (demag model).
    last_outer_split_phi_defect = _state_monitor_get_or_nan(sm, "last_outer_split_phi_defect")

    continuation_transition_just_triggered = int(
        bool(sm.get("continuation_transition_just_triggered", False))
    )
    continuation_reconcile_steps_remaining = _state_monitor_get_or_default_int(
        sm, "continuation_reconcile_steps_remaining", default=0
    )
    continuation_transition_source = str(sm.get("continuation_transition_source", "checkpoint"))

    np.savez_compressed(
        base + "_time_state.npz",
        time_state_version=np.array(1, dtype=np.int32),
        bdf_active_order=np.array(bdf_active_order, dtype=np.int32),
        bdf_steps_since_restart=np.array(bdf_steps_since_restart, dtype=np.int32),
        dt_prev_value=np.array(dt_prev_value, dtype=np.float64),
        be_recovery_steps_remaining=np.array(be_recovery_steps_remaining, dtype=np.int32),
        last_temporal_error_norm=np.array(last_temporal_error_norm, dtype=np.float64),
        thermo_defect_ema=np.array(thermo_defect_ema, dtype=np.float64),
        thermo_phys_delta_ema=np.array(thermo_phys_delta_ema, dtype=np.float64),
        accepted_residual_ema=np.array(accepted_residual_ema, dtype=np.float64),
        accepted_residual_ema_slow=np.array(accepted_residual_ema_slow, dtype=np.float64),
        thermo_dt_cap=np.array(thermo_dt_cap, dtype=np.float64),
        thermo_dt_cap_cooldown_steps=np.array(thermo_dt_cap_cooldown_steps, dtype=np.int32),
        nonthermo_dt_cap=np.array(nonthermo_dt_cap, dtype=np.float64),
        nonthermo_dt_cap_cooldown_steps=np.array(nonthermo_dt_cap_cooldown_steps, dtype=np.int32),
        continuation_transition_dt_cap=np.array(continuation_transition_dt_cap, dtype=np.float64),
        continuation_transition_cooldown_steps=np.array(
            continuation_transition_cooldown_steps, dtype=np.int32
        ),
        nonthermo_clean_accepted_steps=np.array(nonthermo_clean_accepted_steps, dtype=np.int32),
        nonlinear_retry_cooldown_steps=np.array(nonlinear_retry_cooldown_steps, dtype=np.int32),
        last_outer_split_phi_defect=np.array(last_outer_split_phi_defect, dtype=np.float64),
        continuation_transition_just_triggered=np.array(
            continuation_transition_just_triggered, dtype=np.int32
        ),
        continuation_reconcile_steps_remaining=np.array(
            continuation_reconcile_steps_remaining, dtype=np.int32
        ),
        continuation_transition_source=np.array(continuation_transition_source, dtype=object),
    )


def restore_checkpoint_time_state(solver: Any, filename: str) -> None:
    """
    Restore time/controller state from ``*_time_state.npz`` if present.

    Safe to call even when the file is missing (older checkpoints).
    """
    base = _checkpoint_base_path(filename)
    path = base + "_time_state.npz"
    if not os.path.exists(path):
        return

    data = np.load(path, allow_pickle=True)
    # BDF state
    solver._bdf_active_order = int(data["bdf_active_order"].item())
    solver._bdf_steps_since_restart = int(data["bdf_steps_since_restart"].item())
    solver._dt_prev_value = float(data["dt_prev_value"].item())

    # Mark so _init_solve_problem_runtime and init_dynamic can preserve restored state.
    solver._time_checkpoint_restored = True

    sm = getattr(solver, "state_monitor", None)
    if sm is None:
        sm = {}
        solver.state_monitor = sm

    # Helper: map NaN → None for EMAs/caps.
    def _nan_to_none(x: float) -> float | None:
        return None if (not np.isfinite(float(x))) else float(x)

    sm["be_recovery_steps_remaining"] = int(data["be_recovery_steps_remaining"].item())
    sm["last_temporal_error_norm"] = _nan_to_none(data["last_temporal_error_norm"].item())
    sm["thermo_defect_ema"] = _nan_to_none(data["thermo_defect_ema"].item())
    sm["thermo_phys_delta_ema"] = _nan_to_none(data["thermo_phys_delta_ema"].item())
    sm["accepted_residual_ema"] = _nan_to_none(data["accepted_residual_ema"].item())
    sm["accepted_residual_ema_slow"] = _nan_to_none(data["accepted_residual_ema_slow"].item())

    sm["thermo_dt_cap"] = _nan_to_none(data["thermo_dt_cap"].item())
    sm["thermo_dt_cap_cooldown_steps"] = int(data["thermo_dt_cap_cooldown_steps"].item())
    sm["nonthermo_dt_cap"] = _nan_to_none(data["nonthermo_dt_cap"].item())
    sm["nonthermo_dt_cap_cooldown_steps"] = int(data["nonthermo_dt_cap_cooldown_steps"].item())

    sm["continuation_transition_dt_cap"] = _nan_to_none(data["continuation_transition_dt_cap"].item())
    sm["continuation_transition_cooldown_steps"] = int(
        data["continuation_transition_cooldown_steps"].item()
    )

    sm["nonthermo_clean_accepted_steps"] = int(data["nonthermo_clean_accepted_steps"].item())
    sm["nonlinear_retry_cooldown_steps"] = int(data["nonlinear_retry_cooldown_steps"].item())

    sm["last_outer_split_phi_defect"] = _nan_to_none(data["last_outer_split_phi_defect"].item())

    sm["continuation_transition_just_triggered"] = bool(
        int(data["continuation_transition_just_triggered"].item())
    )
    sm["continuation_reconcile_steps_remaining"] = int(
        data["continuation_reconcile_steps_remaining"].item()
    )
    sm["continuation_transition_source"] = str(
        data["continuation_transition_source"].item()
    )


def _load_atlas_metadata_from_checkpoint(solver: Any, base: str) -> None:
    try:
        solver._load_func_npz(solver._node_chart_func, base + "_node_chart.npz")
        n_nodes = min(len(solver._node_chart), len(solver._node_chart_func.x.array))
        solver._node_chart[:n_nodes] = np.round(solver._node_chart_func.x.array[:n_nodes]).astype(
            np.int32
        )
        if n_nodes < len(solver._node_chart):
            solver._node_chart[n_nodes:] = 0
    except (FileNotFoundError, OSError):
        pass

    try:
        solver._load_func_npz(solver.chart_ind, base + "_chart_ind.npz")
        solver.chart_ind.x.scatter_forward()
    except (FileNotFoundError, OSError):
        pass


def restore_checkpoint_atlas_state(solver: Any, filename: str) -> None:
    """
    Restore atlas chart fields and BDF history in-place on ``solver``.

    Loads ``qA``, ``qB``, both BDF histories, and atlas metadata directly into the
    solver's live fields. Sets ``solver._atlas_checkpoint_restored`` so
    :meth:`Z_sim_dynamic_stereo_base._bootstrap_chart_state_for_time_loop` preserves
    the restored history instead of cold-start regeneration.

    Parameters
    ----------
    solver : Z_sim_dynamic_stereo_base
        Solver with allocated chart spaces.
    filename : str
        Checkpoint stem; ``_qA.npz`` and ``_qB.npz`` must exist.

    Raises
    ------
    FileNotFoundError
        If required npz files are missing.

    Notes
    -----
    Missing BDF history files fall back to repeating the loaded current chart state.
    Sets ``_qB_ready`` and ``_dual_charts_initialized`` so magnetization outputs and
    cold-start guards treat the restored charts as authoritative.  DG0 atlas weights
    are rebuilt immediately via :meth:`_rehydrate_atlas_state_from_restored_q`.
    Demag ``phi`` (when present) is restored as a warm start only; after remesh the
    live BEM generally differs, so demag solvers re-equilibrate ``phi`` from the
    restored magnetization in ``init_dynamic`` before seeding the thermo gate.
    """
    base = _checkpoint_base_path(filename)

    solver._load_func_npz(solver.qA, base + "_qA.npz")
    solver.qA.x.scatter_forward()
    solver._load_func_npz(solver.qB, base + "_qB.npz")
    solver.qB.x.scatter_forward()
    solver._qB_ready = True
    solver._dual_charts_initialized = True

    if not _load_optional_q_history(solver, base + "_qA_prev.npz", solver.qA_prev):
        solver.qA_prev.x.array[:] = solver.qA.x.array[:]
        solver.qA_prev.x.scatter_forward()
    if not _load_optional_q_history(solver, base + "_qA_prev_prev.npz", solver.qA_prev_prev):
        solver.qA_prev_prev.x.array[:] = solver.qA_prev.x.array[:]
        solver.qA_prev_prev.x.scatter_forward()

    if not _load_optional_q_history(solver, base + "_qB_prev.npz", solver.qB_prev):
        solver.qB_prev.x.array[:] = solver.qB.x.array[:]
        solver.qB_prev.x.scatter_forward()
    if not _load_optional_q_history(solver, base + "_qB_prev_prev.npz", solver.qB_prev_prev):
        solver.qB_prev_prev.x.array[:] = solver.qB_prev.x.array[:]
        solver.qB_prev_prev.x.scatter_forward()

    _load_atlas_metadata_from_checkpoint(solver, base)
    if hasattr(solver, "_rehydrate_atlas_state_from_restored_q"):
        solver._rehydrate_atlas_state_from_restored_q()
    solver._demag_phi_checkpoint_restored = False
    if hasattr(solver, "_restore_demag_phi_from_checkpoint"):
        solver._demag_phi_checkpoint_restored = bool(solver._restore_demag_phi_from_checkpoint(base))
    solver._atlas_checkpoint_restored = True
    restore_checkpoint_time_state(solver, filename)


def checkpoint_atlas_field_norms(solver: Any, filename: str) -> dict[str, float]:
    """Return owned L2 norms of atlas checkpoint fields without mutating solver state."""
    base = _checkpoint_base_path(filename)
    owned = solver.V_q.dofmap.index_map.size_local * solver.V_q.dofmap.index_map_bs

    def _norm_from_npz(path: str) -> float:
        buf = fem.Function(solver.V_q, name="checkpoint_norm_buf")
        solver._load_func_npz(buf, path)
        local = buf.x.array[:owned]
        return float(np.sqrt(solver.domain.comm.allreduce(np.dot(local, local), op=MPI.SUM)))

    norms = {
        "qA": _norm_from_npz(base + "_qA.npz"),
        "qA_prev": 0.0,
        "qA_prev_prev": 0.0,
        "qB": _norm_from_npz(base + "_qB.npz"),
        "qB_prev": 0.0,
        "qB_prev_prev": 0.0,
    }
    for key, suffix in (
        ("qA_prev", "_qA_prev.npz"),
        ("qA_prev_prev", "_qA_prev_prev.npz"),
        ("qB_prev", "_qB_prev.npz"),
        ("qB_prev_prev", "_qB_prev_prev.npz"),
    ):
        try:
            norms[key] = _norm_from_npz(base + suffix)
        except (FileNotFoundError, OSError):
            pass
    return norms


print_flush = partial(print, flush=True)


class Z_sim_dynamic_stereo_base:
    r"""
    Shared Landau--Lifshitz--Gilbert time integrator on a two-chart stereographic atlas.

    Instantiate via :class:`~Zmag_bins.Zmag_dynamic_stereo_demag.Z_sim_dynamic_stereo_demag`
    or :class:`~Zmag_bins.Zmag_dynamic_stereo_nodemag.Z_sim_dynamic_stereo_nodemag`.

    Physical model
    --------------
    Chart coordinates :math:`q_A` (pole :math:`+1`) and :math:`q_B` (pole :math:`-1`) represent
    magnetization on :math:`S^2`. DG0 masks ``chiA/chiB/chiAB`` and weights ``wA/wB`` blend
    contributions cell-wise. The blended Cartesian direction used in diagnostics is reconstructed
    from both charts.

    Time stepping
    -------------
    :meth:`solve_problem` runs a retry loop per step: update continuation ramps, call subclass
    :meth:`solve_weak_form`, apply residual / thermodynamic / atlas gates, then adapt
    :math:`\Delta t^\ast` on acceptance.

    Code map (this class)
    ---------------------
    Methods are grouped in the source file by section banners. Within
    :meth:`solve_problem`, the private pipeline is:

    * **Setup** — ``_init_solve_problem_runtime``, ``_open_timestep``,
      ``_run_timestep_preamble``
    * **One attempt** — ``_run_accept_attempt`` (gates A–F below)
    * **On failure** — ``_handle_step_rejection`` (rollback, ``dt`` cut, SNES mode)
    * **On success** — ``_finalize_accepted_step`` (history, CFL, I/O, stop test)

    **Gates A–F** (labels match :doc:`theory_algorithms`):

    ====  ==============================  ========================================
    Gate  Method                          Role
    ====  ==============================  ========================================
    A     ``_run_nonlinear_solve``        BDF step + outer split → trial ``q``
    B     ``_validate_algebraic_solve``     SNES/KSP + outer-split convergence
    C     ``_refresh_attempt_energies``   Atlas radius guard + energy snapshot
    D     ``_apply_residual_gate``        D1 backtrack, D2 merit, D3 soft accept
    E     ``_apply_thermodynamic_gate``   Dissipation inequality; E1 backtrack
    F     ``_apply_closure_gate``         Inactive-chart closure + jump check
    ====  ==============================  ========================================

    **Supporting blocks** (other section banners in source):

    * **Mesh / I/O** — ``check_and_save_mesh_partition``, ``load_mesh_partition``, XDMF writers
    * **Atlas** — ``_atlas_update_masks_from_qA`` … ``_apply_post_accept_atlas_closure``
    * **State blend / rollback** — ``_set_state_from_backup_trial_blend``, ``_rollback_to_step_backup``
    * **Dual-chart init** — ``_initialize_and_smooth_q_from_m``, ``_cart_to_stereo_dual_numpy``,
      ``_bootstrap_chart_state_for_time_loop``, ``restore_checkpoint_atlas_state``
    * **Backtracking searches** — ``_residual_backtracking_search``, ``_thermo_backtracking_search``
    * **Thermo assembly** — ``_evaluate_thermodynamic_residual`` (with gate E)
    * **Timestep control** — ``_update_timestep_cfl``, ``_compute_temporal_error_norm``

    Subclasses implement ``init_dynamic``, ``solve_weak_form``, ``compute_energy``, and
    model-specific I/O hooks.  Shared initialization, checkpoint restore, physical-energy
    evaluation, per-step backup/rollback, and ``destroy_dynamic`` live on this base class.
    """

    # ==================================================================
    # Mesh, materials, and external field setup
    # ==================================================================

    def check_and_save_mesh_partition(self, comm, dmain, c_tags, f_tags, results_dir):
        """
        Validate MPI mesh partition balance and export ``domain.xdmf``.

        Fails if any rank owns zero cells. Rank 0 prints load-balance statistics.

        Parameters
        ----------
        comm : mpi4py.MPI.Comm
            Mesh communicator.
        dmain : dolfinx.mesh.Mesh
            Distributed mesh.
        c_tags, f_tags : dolfinx.cpp.mesh.MeshTags_int32
            Cell and facet tags.
        results_dir : str
            Output directory for ``domain.xdmf``.
        """
        num_local_cells = dmain.topology.index_map(dmain.topology.dim).size_local
        if num_local_cells == 0:
            raise RuntimeError(f"Error: Rank {comm.rank} owns 0 cells.")
        all_num_cells = comm.gather(num_local_cells, root=0)
        if comm.rank == 0:
            all_num_cells = np.array(all_num_cells, dtype=np.int64)
            num_ranks = len(all_num_cells)
            total_cells = np.sum(all_num_cells)
            avg_cells = total_cells / num_ranks
            min_cells = np.min(all_num_cells)
            max_cells = np.max(all_num_cells)
            imbalance = (max_cells - min_cells) / avg_cells * 100.0
            print_flush("\n" + "=" * 65)
            print_flush(f"{' Mesh Partitioning Report ':=^65}")
            print_flush(f" Total number of ranks       : {num_ranks}")
            print_flush(f" Total number of cells       : {total_cells}")
            print_flush(f" Average cells per rank      : {avg_cells:.1f}")
            print_flush(f" Min / Max cells per rank    : {min_cells} / {max_cells}")
            print_flush(f" Load imbalance              : {imbalance:.2f}%")
            print_flush("=" * 65 + "\n")
        comm.Barrier()
        with io.XDMFFile(comm, results_dir + "/domain.xdmf", "w") as xdmf:
            xdmf.write_mesh(dmain)
            c_tags.name = "cell_tags"
            f_tags.name = "facet_tags"
            xdmf.write_meshtags(c_tags, dmain.geometry)
            xdmf.write_meshtags(f_tags, dmain.geometry)
            tdim = dmain.topology.dim
            imap = dmain.topology.index_map(tdim)
            num_local_cells = imap.size_local
            local_cell_indices = np.arange(num_local_cells, dtype=np.int32)
            rank_array = np.full(num_local_cells, comm.Get_rank(), dtype=np.int32)
            partition_tags = mesh.meshtags(dmain, tdim, local_cell_indices, rank_array)
            partition_tags.name = "partition_tags"
            xdmf.write_meshtags(partition_tags, dmain.geometry)
            comm.Barrier()

    def load_mesh_partition(self, comm, results_dir):
        """
        Reload mesh and meshtags written by :meth:`check_and_save_mesh_partition`.

        Parameters
        ----------
        comm : mpi4py.MPI.Comm
        results_dir : str
            Directory containing ``domain.xdmf``.

        Returns
        -------
        tuple
            ``(domain, cell_tags, facet_tags)``.
        """
        xdmf_path = results_dir + "/domain.xdmf"
        with io.XDMFFile(comm, xdmf_path, "r") as xdmf:
            domain = xdmf.read_mesh(name="mesh")
            tdim = domain.topology.dim
            fdim = tdim - 1
            domain.topology.create_connectivity(fdim, tdim)
            domain.topology.create_connectivity(tdim, 0)
            domain.topology.create_connectivity(fdim, 0)
            cell_tags = xdmf.read_meshtags(domain, name="cell_tags")
            facet_tags = xdmf.read_meshtags(domain, name="facet_tags")
        return domain, cell_tags, facet_tags

    def set_mesh(
        self,
        mesh: mesh.Mesh,
        cell_tags: cpp.mesh.MeshTags_int32,
        facet_tags: cpp.mesh.MeshTags_int32,
        surface_tag: int,
    ):
        """
        Attach mesh, tags, and build FEM spaces for the mixed formulation.

        Calls ``_init_after_mesh`` to allocate ``V_q``, ``V_phi`` (demag only),
        mixed ``V_mixed``, atlas DG0 fields, measures, and mesh-size field.

        Parameters
        ----------
        mesh : dolfinx.mesh.Mesh
        cell_tags, facet_tags : dolfinx.cpp.mesh.MeshTags_int32
        surface_tag : int
            Facet tag for the exterior magnetostatic boundary (demag).
        """
        self.domain = mesh
        self.cell_tags = cell_tags
        self.facet_tags = facet_tags
        self._init_after_mesh(surface_tag)

    def set_materials(self, materials: list[dict]):
        """
        Interpolate piecewise-constant material data onto FEM coefficient fields.

        Parameters
        ----------
        materials : list[dict]
            One dict per cell tag with keys ``tag``, ``Ms``, ``Aex``, ``K``, ``Q1``,
            ``K2K1_ratio``, and optional ``anisotropy_axis``.
        """
        self._interpolate_materials_properties(materials)

    def set_external_field(self, field_input):
        """
        Set the non-dimensional Zeeman field ``h_zee``.

        Requires ``"zeeman"`` in ``energy_terms``.

        Parameters
        ----------
        field_input : dict or array-like
            Constant 3-vector or Biot--Savart coil parameter dictionary.

        Raises
        ------
        TypeError
            If the input type is unsupported.
        AssertionError
            If Zeeman is not an active energy term.
        """
        assert "zeeman" in self.energy_terms
        if isinstance(field_input, dict):
            args_coil_param = field_input
            hfield = compute_Hzee_coil.HFieldCoilBiotSavart(
                args_coil_param["wire_path"],
                args_coil_param["N_turns"],
                args_coil_param["current"],
                args_coil_param["l_ex"],
                args_coil_param["Ms_ref"],
                args_coil_param["size_scale_fac"],
            )
            self.h_zee = fem.Function(self.V_hzee)
            self.h_zee.interpolate(lambda x: hfield.compute_non_dim_field(x, args_coil_param["current"]))
            self.h_zee.x.scatter_forward()
            if args_coil_param["write_Hzee_to_file"]:
                outdir = self.path_dirs["results_dir"]
                self.h_zee_vtx = io.VTXWriter(
                    self.domain.comm, outdir + "/HzeeField_nonDim.bp", [self.h_zee], engine="BP4"
                )
                self.h_zee_vtx.write(0)
            return
        if isinstance(field_input, (list, tuple, np.ndarray)):
            zeeman_vec = validate_spin_vector(field_input, "Zeeman field")
            self.h_zee = fem.Function(self.V_hzee)
            self.h_zee.interpolate(lambda x: np.tile(zeeman_vec[:, None], x.shape[1]))
            self.h_zee.x.scatter_forward()
            return
        raise TypeError(f"Unsupported type {type(field_input)} for external field input")

    @staticmethod
    # ==================================================================
    # Stereographic chart operators (UFL)
    # ==================================================================

    def _sigma2(q):
        # Stereographic radius factor sigma^2 = (2/(1+r^2))^2; appears in P_mu and L_mu.
        r2 = q[0] ** 2 + q[1] ** 2
        return (2 / (1 + r2)) ** 2

    def P_mu_gamma(self, q):
        r"""
        Stereographic exchange metric :math:`P_\mu = \mathrm{diag}(\sigma^2,\sigma^2)`.

        Parameters
        ----------
        q : ufl.core.expr.Expr
            Chart coordinates.

        Returns
        -------
        ufl.Matrix
        2×2 metric with :math:`\sigma^2=(2/(1+r^2))^2`.
        """
        s2 = self._sigma2(q)
        return ufl.as_matrix([[s2, 0], [0, s2]])

    def Q_mu(self, q, grad_q):
        r"""
        Nonlinear exchange coupling :math:`Q_\mu(q,\nabla q)` in the stereographic weak form.

        Parameters
        ----------
        q, grad_q
        UFL chart value and gradient.

        Returns
        -------
        ufl.Vector
        Length-2 UFL vector contracted with test functions in exchange terms.
        """
        u, v = q[0], q[1]
        r2 = u**2 + v**2
        s2 = (2 / (1 + r2)) ** 2
        grad_q_sq = ufl.dot(grad_q[0, :], grad_q[0, :]) + ufl.dot(grad_q[1, :], grad_q[1, :])
        factor = -4 * s2 / (1 + r2) * grad_q_sq
        return ufl.as_vector([factor * u, factor * v])

    # ==================================================================
    # Continuation ramps (demag / Zeeman)
    # ==================================================================

    def _mark_continuation_transition(self, source: str, current_step: int):
        """Mark a continuation-regime transition for temporary stabilization."""
        if self.state_monitor is None:
            return
        cd_key = "continuation_transition_cooldown_steps"
        self.state_monitor[cd_key] = max(
            int(self.state_monitor.get(cd_key, 0)),
            int(self._continuation_transition_cooldown_default),
        )
        self.state_monitor["continuation_transition_source"] = str(source)
        self.state_monitor["continuation_transition_just_triggered"] = True
        if self.domain.comm.rank == 0:
            print_flush(f"\n[Continuation Transition] {source} ramp completed at step {current_step}")

    def update_demag_ramp_factor(self):
        """Update demag ramp factor with transition marker on completion."""
        if self.demag_ramp_factor is None:
            return
        current_step = self.step_now
        if current_step < self.demag_ramp_delay:
            self.demag_ramp_factor = self.demag_start_fraction
            return
        ramp_step = current_step - self.demag_ramp_delay
        if ramp_step < self.demag_ramp_N and not self._demag_ramp_done:
            x = ramp_step / max(self.demag_ramp_N - 1, 1)
            smoothstep = 3 * x**2 - 2 * x**3
            self.demag_ramp_factor = self.demag_start_fraction + (1.0 - self.demag_start_fraction) * smoothstep
        else:
            self.demag_ramp_factor = 1.0
            if not self._demag_ramp_done:
                self._demag_ramp_done = True
                if self.domain.comm.rank == 0:
                    print_flush(f"\n[Demag Ramp] Complete at step {current_step}")
                self._mark_continuation_transition("demag", current_step)

    def update_zeeman_ramp_factor(self):
        """Update Zeeman ramp factor with transition marker on completion."""
        if self.zeeman_ramp_factor is None:
            return
        current_step = self.step_now
        if not self._zeeman_field_stored:
            self.h_zee.x.scatter_forward()
            self.h_zee_original = self.h_zee.x.array.copy()
            self.h_zee.x.array[:] = self.zeeman_start_fraction * self.h_zee_original
            self.h_zee.x.scatter_forward()
            self._zeeman_field_stored = True
        if current_step < self.zeeman_ramp_delay:
            self.zeeman_ramp_factor = self.zeeman_start_fraction
            self.h_zee.x.array[:] = self.zeeman_start_fraction * self.h_zee_original
            self.h_zee.x.scatter_forward()
            return
        ramp_step = current_step - self.zeeman_ramp_delay
        if ramp_step < self.zeeman_ramp_N and not self._zeeman_ramp_done:
            x = ramp_step / max(self.zeeman_ramp_N - 1, 1)
            smoothstep = 3 * x**2 - 2 * x**3
            self.zeeman_ramp_factor = self.zeeman_start_fraction + (1.0 - self.zeeman_start_fraction) * smoothstep
            self.h_zee.x.array[:] = self.zeeman_ramp_factor * self.h_zee_original
            self.h_zee.x.scatter_forward()
        else:
            self.zeeman_ramp_factor = 1.0
            self.h_zee.x.array[:] = self.h_zee_original
            self.h_zee.x.scatter_forward()
            if not self._zeeman_ramp_done:
                self._zeeman_ramp_done = True
                if self.domain.comm.rank == 0:
                    print_flush(f"\n[Zeeman Ramp] Complete at step {current_step}")
                self._mark_continuation_transition("zeeman", current_step)

    def stereo_to_cart_chart(self, q, sign):
        r"""
        Map stereographic chart coordinates to a unit Cartesian magnetization.

        Uses the pole and tangent basis from ``args_init_sim``. Chart B uses ``sign=-1``.

        Parameters
        ----------
        q : ufl.core.expr.Expr
            Chart coordinates :math:`(u,v)`.
        sign : float
            ``+1`` (chart A) or ``-1`` (chart B).

        Returns
        -------
        ufl.Vector
        :math:`\mathbf{m}(q,\mathrm{sign})` on the unit sphere.
        """
        u, v = q[0], q[1]
        r2 = u**2 + v**2
        denom = 1 + r2
        omr2 = 1 - r2
        mx = (-sign * self._pole_x * omr2 + 2 * u * self._e1_x + 2 * v * self._e2_x) / denom
        my = (-sign * self._pole_y * omr2 + 2 * u * self._e1_y + 2 * v * self._e2_y) / denom
        mz = (-sign * self._pole_z * omr2 + 2 * u * self._e1_z + 2 * v * self._e2_z) / denom
        return ufl.as_vector([mx, my, mz])

    def L_mu_gamma_chart(self, q, sign):
        """Chart-specific L matrix with fixed orientation sign."""
        s2 = self._sigma2(q)
        s2_reg = ufl.max_value(s2, self._sigma2_floor)
        return ufl.as_matrix(
            [
                [self.alpha * s2_reg, sign * s2_reg],
                [-sign * s2_reg, self.alpha * s2_reg],
            ]
        )

    def A_mu_i_chart(self, q, sign):
        """Chart-specific Jacobian dm/dq with fixed sign."""
        u, v = q[0], q[1]
        r2 = u**2 + v**2
        f = 2 / (1 + r2) ** 2

        s2u = sign * 2 * u
        s2v = sign * 2 * v
        a = 1 + v**2 - u**2
        b = -2 * u * v
        c = 1 + u**2 - v**2

        col_u_x = f * (s2u * self._pole_x + a * self._e1_x + b * self._e2_x)
        col_u_y = f * (s2u * self._pole_y + a * self._e1_y + b * self._e2_y)
        col_u_z = f * (s2u * self._pole_z + a * self._e1_z + b * self._e2_z)
        col_v_x = f * (s2v * self._pole_x + b * self._e1_x + c * self._e2_x)
        col_v_y = f * (s2v * self._pole_y + b * self._e1_y + c * self._e2_y)
        col_v_z = f * (s2v * self._pole_z + b * self._e1_z + c * self._e2_z)

        return ufl.as_matrix(
            [[col_u_x, col_v_x], [col_u_y, col_v_y], [col_u_z, col_v_z]]
        )

    # ==================================================================
    # Atlas masks, weights, and closure
    # ==================================================================

    def _atlas_update_masks_from_qA(self):
        """Update DG0 atlas masks from symmetric atlas state (qA + qB)."""
        dofmap = self.V_q.dofmap
        n_owned = dofmap.index_map.size_local
        n_total = n_owned + dofmap.index_map.num_ghosts
        n_cells = self._num_local_cells

        mA = self._m_from_q_numpy(self.qA.x.array[: n_total * 2], np.zeros(n_total, dtype=np.int32))
        mB = self._m_from_q_numpy(self.qB.x.array[: n_total * 2], np.ones(n_total, dtype=np.int32))
        # Weighted Cartesian blend for the mask signal (not 0.5*(mA+mB)): at a pole,
        # q=(0,0) on one chart is the antipode on the other, so an unweighted average
        # falsely reports equatorial overlap and seeds spurious wB (or wA).
        wa_node, wb_node = self._compute_nodal_atlas_weights(n_total)
        mdotp = self._atlas_mdot_signal_nodes(mA, mB, n_total)

        tau = max(float(self.chart_hysteresis_tau), 0.0)
        overlap = max(float(self.atlas_overlap_tau), 0.0)

        # Needed for both diagnostics and automatic thermo operator budget.
        collect_mask_stats = True
        if collect_mask_stats:
            old_chiA = np.array(self.chiA.x.array[:n_cells], copy=True)
            old_chiB = np.array(self.chiB.x.array[:n_cells], copy=True)
            old_chiAB = np.array(self.chiAB.x.array[:n_cells], copy=True)

        chiA = np.zeros(n_cells, dtype=self.chiA.x.array.dtype)
        chiB = np.zeros(n_cells, dtype=self.chiB.x.array.dtype)
        chiAB = np.zeros(n_cells, dtype=self.chiAB.x.array.dtype)
        wA = np.zeros(n_cells, dtype=self.wA.x.array.dtype)
        wB = np.zeros(n_cells, dtype=self.wB.x.array.dtype)
        old_wA = np.array(self.wA.x.array[:n_cells], copy=True)
        band = max(tau + overlap, 1e-12)

        for c in range(n_cells):
            cdofs = self._cell_dofs_local[c]
            s = float(np.mean(mdotp[cdofs]))
            if s > band:
                wa_raw, wb_raw = 0.0, 1.0
            elif s < -band:
                wa_raw, wb_raw = 1.0, 0.0
            else:
                # Smooth overlap interpolation to reduce operator jumps.
                wa_raw = 0.5 * (1.0 - s / band)
                wa_raw = min(1.0, max(0.0, wa_raw))
                wb_raw = 1.0 - wa_raw
            rho = self.atlas_weight_relaxation
            wa = (1.0 - rho) * float(old_wA[c]) + rho * float(wa_raw)
            wa = min(1.0, max(0.0, wa))
            wb = 1.0 - wa
            wA[c], wB[c] = wa, wb
            hasA = wa > 1e-12
            hasB = wb > 1e-12
            chiA[c] = 1.0 if hasA else 0.0
            chiB[c] = 1.0 if hasB else 0.0
            chiAB[c] = 1.0 if (hasA and hasB) else 0.0

        self.chiA.x.array[:n_cells] = chiA
        self.chiB.x.array[:n_cells] = chiB
        self.chiAB.x.array[:n_cells] = chiAB
        self.wA.x.array[:n_cells] = wA
        self.wB.x.array[:n_cells] = wB
        self.chiA.x.scatter_forward()
        self.chiB.x.scatter_forward()
        self.chiAB.x.scatter_forward()
        self.wA.x.scatter_forward()
        self.wB.x.scatter_forward()
        # Nodal atlas weight averaging depends on current DG0 weights.
        # Invalidate cache so diagnostics/outputs/demag computations use
        # up-to-date chart activity after atlas closure updates.
        self._atlas_nodal_weights_cache = None

        # Mask stats are diagnostics-only; skip in hot path when disabled.
        if collect_mask_stats:
            switched_local = int(
                np.count_nonzero(np.abs(chiA - old_chiA) > 0.5)
                + np.count_nonzero(np.abs(chiB - old_chiB) > 0.5)
                + np.count_nonzero(np.abs(chiAB - old_chiAB) > 0.5)
            )
            count_A_local = int(np.count_nonzero(chiA > 0.5))
            count_B_local = int(np.count_nonzero(chiB > 0.5))
            count_AB_local = int(np.count_nonzero(chiAB > 0.5))
            comm = self.domain.comm
            self._atlas_mask_stats = {
                "cells_local": int(n_cells),
                "cells_global": int(comm.allreduce(n_cells, op=MPI.SUM)),
                "switched_flags_global": int(comm.allreduce(switched_local, op=MPI.SUM)),
                "chiA_cells_global": int(comm.allreduce(count_A_local, op=MPI.SUM)),
                "chiB_cells_global": int(comm.allreduce(count_B_local, op=MPI.SUM)),
                "chiAB_cells_global": int(comm.allreduce(count_AB_local, op=MPI.SUM)),
                "avg_wA_global": float(
                    comm.allreduce(float(np.sum(wA)), op=MPI.SUM)
                    / max(1, comm.allreduce(n_cells, op=MPI.SUM))
                ),
                "avg_wB_global": float(
                    comm.allreduce(float(np.sum(wB)), op=MPI.SUM)
                    / max(1, comm.allreduce(n_cells, op=MPI.SUM))
                ),
            }

    def _bootstrap_chart_state_for_time_loop(self):
        """
        Prepare chart/mixed state for time stepping.

        On atlas checkpoint restart, preserve restored ``qA/qB`` and BDF history.
        On cold start, rebuild a coherent dual-chart state and initialize BDF history.
        """
        if getattr(self, "_atlas_checkpoint_restored", False):
            # History and charts were loaded from checkpoint; do not overwrite.
            self._sync_mixed_q_from_state()
            self._atlas_checkpoint_restored = False
            return

        if not getattr(self, "_dual_charts_initialized", False):
            # No set_init_q_* call: rebuild both charts from current Cartesian m.
            self._sync_dual_charts_from_cartesian()
        # Cold-start BDF history: q(n)=q(n-1)=q(n-2) at t=0.
        # qA_prev/qB_prev were already set in _write_dual_q_to_state during init.
        self.qA_prev_prev.x.array[:] = self.qA.x.array[:]
        self.qA_prev_prev.x.scatter_forward()
        self.qB_prev_prev.x.array[:] = self.qB.x.array[:]
        self.qB_prev_prev.x.scatter_forward()
        self._sync_mixed_q_from_state()

    def _compute_nodal_atlas_weights(self, n_total):
        """Average DG0 atlas weights onto q nodes and renormalize."""
        cache = getattr(self, "_atlas_nodal_weights_cache", None)
        if cache is not None and int(cache.get("n_total", -1)) == int(n_total):
            return cache["wa"], cache["wb"]

        wa_sum = np.zeros(n_total, dtype=np.float64)
        wb_sum = np.zeros(n_total, dtype=np.float64)
        cnt = np.zeros(n_total, dtype=np.int32)
        for c in range(self._num_local_cells):
            cdofs = self._cell_dofs_local[c]
            wa_c = float(self.wA.x.array[c])
            wb_c = float(self.wB.x.array[c])
            wa_sum[cdofs] += wa_c
            wb_sum[cdofs] += wb_c
            cnt[cdofs] += 1

        wa_node = np.ones(n_total, dtype=np.float64)
        wb_node = np.zeros(n_total, dtype=np.float64)
        has_cells = cnt > 0
        if np.any(has_cells):
            wa_node[has_cells] = wa_sum[has_cells] / np.maximum(cnt[has_cells], 1)
            wb_node[has_cells] = wb_sum[has_cells] / np.maximum(cnt[has_cells], 1)
            s = wa_node[has_cells] + wb_node[has_cells]
            wa_node[has_cells] = wa_node[has_cells] / np.maximum(s, 1e-12)
            wb_node[has_cells] = 1.0 - wa_node[has_cells]
        self._atlas_nodal_weights_cache = {
            "n_total": int(n_total),
            "wa": wa_node,
            "wb": wb_node,
        }
        return wa_node, wb_node

    def _reassign_charts_multipass(self, max_passes=3, threshold=0.0):
        """Node reassignment is not used in qA/qB mode."""
        return 0

    def check_q_magnitude(self, silent=False):
        r"""
        Report the global maximum effective chart radius.

        Computes :math:`\min(r_A,r_B)` per node with :math:`r=|q|`, then MPI-reduces the maximum.
        Used as an atlas stability diagnostic.

        Parameters
        ----------
        silent : bool, optional
            Suppress rank-0 printing when ``True``.

        Returns
        -------
        float
        Global maximum of the per-node minimum chart radius.
        """
        qA = self.qA.x.array
        qB = self.qB.x.array
        rA2 = qA[0::2] ** 2 + qA[1::2] ** 2
        rB2 = qB[0::2] ** 2 + qB[1::2] ** 2
        r2 = np.minimum(rA2, rB2)
        r2_max_local = np.max(r2) if len(r2) > 0 else 0.0
        r2_max = self.domain.comm.allreduce(r2_max_local, op=MPI.MAX)
        r_max = float(np.sqrt(r2_max))
        if not silent and self.domain.comm.rank == 0 and self.printout:
            print_flush(f"\n  effective max |q| = {r_max:.4f}  (atlas qA/qB)")
        return r_max

    def _enforce_atlas_radius_guard(self, silent=True):
        """
        Enforce atlas stereographic radius limits on the current ``qA``/``qB``.

        Called after the SNES trial and again whenever backtracking changes the
        dynamic state (residual or thermodynamic line search).
        """
        r_max_attempt = self.check_q_magnitude(silent=silent)
        atlas_guard = float(self.atlas_qmax_guard)
        atlas_hard_reject = atlas_guard * float(self._atlas_qmax_guard_soft_factor)
        if r_max_attempt > atlas_hard_reject:
            raise RuntimeError(
                "Atlas radius guard failed: "
                f"effective max |q|={r_max_attempt:.3e} exceeds "
                f"hard_guard={atlas_hard_reject:.3e} "
                f"(base_guard={atlas_guard:.3e})"
            )
        if (
            r_max_attempt > atlas_guard
            and self.domain.comm.rank == 0
            and self._debug_logs
        ):
            print_flush(
                f"    [Atlas Guard Soft] max |q|={r_max_attempt:.3e} "
                f"above base_guard={atlas_guard:.3e} but below hard_guard={atlas_hard_reject:.3e}"
            )
        return r_max_attempt

    def _chart_consistency_metrics(self):
        """Atlas mode does not use node-vs-cell single-chart consistency."""
        n_owned = self.V_q.dofmap.index_map.size_local
        n_global = self.domain.comm.allreduce(n_owned, op=MPI.SUM)
        return 0, 1, 0, n_global

    def _reconstruct_inactive_chart_from_active(self):
        """
        Smoothly reconstruct both charts from blended Cartesian m.

        Avoids hard active/inactive switching: each chart is relaxed toward its
        reconstructed value with complementary nodal weight.
        """
        n_owned = self.V_q.dofmap.index_map.size_local
        n_total = n_owned + self.V_q.dofmap.index_map.num_ghosts
        if n_total <= 0:
            return

        wa_node, wb_node = self._compute_nodal_atlas_weights(n_total)

        qA_all = self.qA.x.array[: n_total * 2]
        qB_all = self.qB.x.array[: n_total * 2]
        mA = self._m_from_q_numpy(qA_all, np.zeros(n_total, dtype=np.int32))
        mB = self._m_from_q_numpy(qB_all, np.ones(n_total, dtype=np.int32))
        m_blend = wa_node[:, None] * mA + wb_node[:, None] * mB
        m_norm = np.linalg.norm(m_blend, axis=1)
        m_norm = np.maximum(m_norm, 1e-15)
        m_blend = m_blend / m_norm[:, None]

        # Rebuild both charts from trusted Cartesian m (same pole-safe path as init).
        # At exact poles: complementary chart -> q=0; singular chart -> R_cap * direction.
        uA_rec, vA_rec, uB_rec, vB_rec = self._cart_to_stereo_dual_numpy(
            m_blend, prev_qA=qA_all, prev_qB=qB_all
        )

        # Update owned nodes with complementary relaxation factors.
        wb_owned = np.clip(wb_node[:n_owned], 0.0, 1.0)
        wa_owned = np.clip(wa_node[:n_owned], 0.0, 1.0)
        idx = np.arange(n_owned, dtype=np.int64)
        self.qA.x.array[2 * idx] = (1.0 - wb_owned) * self.qA.x.array[2 * idx] + wb_owned * uA_rec[:n_owned]
        self.qA.x.array[2 * idx + 1] = (
            (1.0 - wb_owned) * self.qA.x.array[2 * idx + 1] + wb_owned * vA_rec[:n_owned]
        )
        self.qB.x.array[2 * idx] = (1.0 - wa_owned) * self.qB.x.array[2 * idx] + wa_owned * uB_rec[:n_owned]
        self.qB.x.array[2 * idx + 1] = (
            (1.0 - wa_owned) * self.qB.x.array[2 * idx + 1] + wa_owned * vB_rec[:n_owned]
        )

        self.qA.x.scatter_forward()
        self.qB.x.scatter_forward()

    def _rehydrate_atlas_state_from_restored_q(self):
        """Rebuild atlas masks/weights immediately after restoring qA/qB."""
        if not hasattr(self, "wA"):
            return
        n_total = self.V_q.dofmap.index_map.size_local + self.V_q.dofmap.index_map.num_ghosts
        if n_total <= 0:
            return
        # Checkpoints keep qA/qB history but not DG0 atlas weights. Rebuild the
        # operational atlas state immediately so restart outputs use the same
        # physical blend as the restored chart state.
        m_restore = self._blend_m_from_qA_qB(
            self.qA.x.array[: n_total * 2],
            self.qB.x.array[: n_total * 2],
        )
        self._seed_atlas_masks_from_m_numpy(m_restore, full_relax=True)

    def _atlas_closure_metrics(self):
        """Compute lightweight global closure indicators for diagnostics."""
        qA = self.qA.x.array
        qB = self.qB.x.array
        n_total = min(len(qA), len(qB)) // 2
        if n_total <= 0:
            return {
                "closure_overlap_rms": 0.0,
                "closure_exclusive_rms": 0.0,
                "sigma2_min_active": 0.0,
                "overlap_cell_fraction": 0.0,
            }

        mA = self._m_from_q_numpy(qA[: 2 * n_total], np.zeros(n_total, dtype=np.int32))
        mB = self._m_from_q_numpy(qB[: 2 * n_total], np.ones(n_total, dtype=np.int32))
        dm = mA - mB
        err_m2 = np.sum(dm * dm, axis=1)
        uA = qA[0 : 2 * n_total : 2]
        vA = qA[1 : 2 * n_total : 2]
        uB = qB[0 : 2 * n_total : 2]
        vB = qB[1 : 2 * n_total : 2]
        rA2 = uA * uA + vA * vA
        rB2 = uB * uB + vB * vB
        s2A = (2.0 / (1.0 + rA2)) ** 2
        s2B = (2.0 / (1.0 + rB2)) ** 2

        sum_overlap_local = 0.0
        cnt_overlap_local = 0
        sum_exclusive_local = 0.0
        cnt_exclusive_local = 0
        s2_min_local = float("inf")
        n_overlap_cells_local = 0

        for c in range(self._num_local_cells):
            cdofs = self._cell_dofs_local[c]
            chiA = float(self.chiA.x.array[c])
            chiB = float(self.chiB.x.array[c])
            chiAB = float(self.chiAB.x.array[c])
            if chiAB > 0.5:
                n_overlap_cells_local += 1
                for d in cdofs:
                    if d < n_total:
                        sum_overlap_local += float(err_m2[d])
                        cnt_overlap_local += 1
            elif chiA > 0.5 and chiB < 0.5:
                for d in cdofs:
                    if d < n_total:
                        sum_exclusive_local += float(err_m2[d])
                        cnt_exclusive_local += 1
            elif chiB > 0.5 and chiA < 0.5:
                for d in cdofs:
                    if d < n_total:
                        sum_exclusive_local += float(err_m2[d])
                        cnt_exclusive_local += 1

            for d in cdofs:
                if d < n_total:
                    if chiA > 0.5:
                        s2_min_local = min(s2_min_local, float(s2A[d]))
                    if chiB > 0.5:
                        s2_min_local = min(s2_min_local, float(s2B[d]))

        comm = self.domain.comm
        sum_overlap = comm.allreduce(sum_overlap_local, op=MPI.SUM)
        cnt_overlap = comm.allreduce(cnt_overlap_local, op=MPI.SUM)
        sum_exclusive = comm.allreduce(sum_exclusive_local, op=MPI.SUM)
        cnt_exclusive = comm.allreduce(cnt_exclusive_local, op=MPI.SUM)
        overlap_cells = comm.allreduce(n_overlap_cells_local, op=MPI.SUM)
        all_cells = comm.allreduce(self._num_local_cells, op=MPI.SUM)
        s2_min_global = comm.allreduce(s2_min_local, op=MPI.MIN)
        if not np.isfinite(s2_min_global):
            s2_min_global = 0.0

        return {
            "closure_overlap_rms": float(np.sqrt(sum_overlap / max(1, cnt_overlap))),
            "closure_exclusive_rms": float(np.sqrt(sum_exclusive / max(1, cnt_exclusive))),
            "sigma2_min_active": float(s2_min_global),
            "overlap_cell_fraction": float(overlap_cells / max(1, all_cells)),
        }

    def _apply_post_accept_atlas_closure(self, update_monitor=True, apply_closure=True):
        """Atlas closure/monitor pass with optional spike-triggered extra passes."""
        if apply_closure:
            # Baseline closure pass.
            self._atlas_update_masks_from_qA()
            self._reconstruct_inactive_chart_from_active()
            self._atlas_update_masks_from_qA()

        closure = self._atlas_closure_metrics()
        overlap_rms = float(closure.get("closure_overlap_rms", 0.0))
        extra_passes = 0

        ema_prev = self.state_monitor.get("atlas_overlap_rms_ema", None)
        abs_tol = max(float(self.atlas_closure_overlap_abs_tol), 0.0)
        spike_factor = max(float(self.atlas_closure_overlap_spike_factor), 1.0)
        threshold = max(abs_tol, spike_factor * float(ema_prev)) if ema_prev is not None else abs_tol

        if apply_closure and self.atlas_closure_monitor_enable and overlap_rms > threshold:
            for _ in range(max(0, int(self.atlas_closure_extra_passes_max))):
                self._reconstruct_inactive_chart_from_active()
                self._atlas_update_masks_from_qA()
                extra_passes += 1
                closure = self._atlas_closure_metrics()
                overlap_rms = float(closure.get("closure_overlap_rms", 0.0))
                if overlap_rms <= threshold:
                    break

        if update_monitor:
            beta = float(min(max(self.atlas_closure_overlap_ema_beta, 0.0), 0.999))
            ema_new = overlap_rms if ema_prev is None else (beta * float(ema_prev) + (1.0 - beta) * overlap_rms)
            self.state_monitor["atlas_overlap_rms_ema"] = float(ema_new)
            self.state_monitor["atlas_overlap_rms_last"] = float(overlap_rms)
            self.state_monitor["atlas_overlap_extra_passes_last"] = int(extra_passes)
            if self.domain.comm.rank == 0 and extra_passes > 0:
                print_flush(
                    "    [Atlas Closure Monitor] "
                    f"overlap_rms={overlap_rms:.3e}, threshold={threshold:.3e}, "
                    f"extra_passes={extra_passes}"
                )
        return closure

    def _sync_mixed_q_from_state(self):
        """Keep mixed q subfields aligned with current qA/qB functions.

        Rule for 0.11: if the source is already a ``Function``, interpolate it
        directly into the target (or mixed ``sub``). Use ``fem.Expression`` only
        for UFL forms (gradients, blended fields, etc.), not to wrap Functions.
        """
        self.u_mixed.sub(0).interpolate(self.qA)
        self.u_mixed.sub(1).interpolate(self.qB)
        self.u_mixed.x.scatter_forward()

    def _refresh_qdot_from_current_state(self):
        """Recompute qA_dot and qB_dot from current qA/qB and BDF history."""
        self._update_q_dot_from_bdf_history()

    # ==================================================================
    # Outer-split helpers (demag and nodemag)
    # ==================================================================

    def _outer_split_pack_state_owned(self, n_q_owned):
        """Pack owned qA/qB state for outer fixed-point acceleration."""
        x = np.empty(2 * n_q_owned, dtype=np.float64)
        x[:n_q_owned] = self.qA.x.array[:n_q_owned]
        x[n_q_owned:] = self.qB.x.array[:n_q_owned]
        return x

    def _outer_split_unpack_state_owned(self, x_owned, n_q_owned):
        """Unpack owned qA/qB accelerated state back into FE functions."""
        self.qA.x.array[:n_q_owned] = x_owned[:n_q_owned]
        self.qB.x.array[:n_q_owned] = x_owned[n_q_owned:]
        self.qA.x.scatter_forward()
        self.qB.x.scatter_forward()

    def _outer_split_anderson_candidate(self, x_hist, g_hist):
        """
        Compute Anderson-mixed candidate for fixed-point map x -> g(x).

        Uses constrained residual minimization on recent history:
            min ||sum_i alpha_i (g_i - x_i)||  s.t. sum_i alpha_i = 1
        and returns x_acc = sum_i alpha_i g_i.
        """
        m = len(g_hist)
        if m < 2:
            return g_hist[-1]

        F_local = np.vstack([g_hist[i] - x_hist[i] for i in range(m)])  # shape (m, n_local)
        G_local = F_local @ F_local.T
        G = np.zeros_like(G_local)
        self.domain.comm.Allreduce(G_local, G, op=MPI.SUM)

        reg = 1e-12 * max(1.0, float(np.trace(G)) / max(1, m))
        G = G + reg * np.eye(m, dtype=np.float64)
        KKT = np.zeros((m + 1, m + 1), dtype=np.float64)
        KKT[:m, :m] = 2.0 * G
        KKT[:m, m] = 1.0
        KKT[m, :m] = 1.0
        rhs = np.zeros(m + 1, dtype=np.float64)
        rhs[m] = 1.0
        try:
            sol = np.linalg.solve(KKT, rhs)
            alpha = sol[:m]
        except np.linalg.LinAlgError:
            alpha, *_ = np.linalg.lstsq(KKT, rhs, rcond=None)
            alpha = alpha[:m]

        alpha_sum = float(np.sum(alpha))
        if not np.isfinite(alpha_sum) or abs(alpha_sum) < 1e-14:
            return g_hist[-1]
        alpha = alpha / alpha_sum
        x_acc = np.zeros_like(g_hist[-1])
        for i in range(m):
            x_acc += float(alpha[i]) * g_hist[i]
        return x_acc

    # ==================================================================
    # State backup, trial blend, and acceptance prep
    # ==================================================================

    def _set_state_from_backup_trial_blend(self, lam, qA_trial, qB_trial):
        r"""
        Convex blend of chart state toward the SNES trial (state-space line search).

        .. math::
            q \leftarrow q^n + \lambda\,(q^{\mathrm{trial}} - q^n)

        Used by residual and thermo backtracking. Does **not** change ``dt`` or re-run SNES;
        refreshes ``q̇`` from BDF history, syncs ``u_mixed``, and (demag) refreshes ``phi``
        from the blended ``q``. See :doc:`theory_algorithms` (*State backtracking*).
        """
        lam = float(lam)
        self.qA.x.array[:] = self._qA_backup + lam * (qA_trial - self._qA_backup)
        self.qB.x.array[:] = self._qB_backup + lam * (qB_trial - self._qB_backup)
        self.qA.x.scatter_forward()
        self.qB.x.scatter_forward()

        # Thermo acceptance is evaluated on the raw dynamic state.
        # Atlas closure is applied only after a step is accepted.
        if "demag" in self.energy_terms:
            # Keep phi/state closure consistent with damped q.
            self._refresh_demag_state_from_current_q()
        self._sync_mixed_q_from_state()
        self._refresh_qdot_from_current_state()

    def _rollback_to_step_backup(self):
        """
        Restore dynamic state from the per-step backup before a retry.

        Called when an attempt is **rejected** (after ``RuntimeError``). Contrasts with
        backtracking, which accepts a damped trial without shrinking ``dt``.
        """
        if "demag" in self.energy_terms:
            self._rollback_state(self._qA_backup, self._qA_prev_backup, self._phi_backup)
        else:
            self._rollback_state(self._qA_backup, self._qA_prev_backup)

    def _prepare_state_for_acceptance(self, apply_closure: bool):
        """
        Synchronize auxiliary fields for acceptance checks.

        Parameters
        ----------
        apply_closure:
            If ``False`` (default during dynamic gates): keep the raw SNES/backtracked
            charts; refresh ``q̇``, mixed vector, and demag ``phi`` only.
            If ``True`` (after gates pass): run atlas reconstruction
            (``_reconstruct_inactive_chart_from_active``) then sync aux fields.
            Closure energy jump is measured immediately after this call.
        """
        self._apply_post_accept_atlas_closure(update_monitor=False, apply_closure=bool(apply_closure))
        if "demag" in self.energy_terms:
            self._refresh_demag_state_from_current_q()
        self._sync_mixed_q_from_state()
        self._refresh_qdot_from_current_state()

    def _compute_energy_quiet(self):
        """Compute energies without per-attempt console spam during retries."""
        old_printout = self.printout
        try:
            self.printout = False
            return self.compute_energy()
        finally:
            self.printout = old_printout

    def _compute_acceptance_energies(self):
        """
        Split-chart and physical energies on the current dynamic state.

        Used after acceptance-stage state changes (e.g. residual backtracking)
        so thermo and closure gates evaluate the same ``q`` they act on.
        """
        energy_components = self._compute_energy_quiet()
        energy_components_physical = self._compute_energy_physical_cartesian()
        return (
            energy_components,
            float(energy_components[-1]),
            energy_components_physical,
            float(energy_components_physical[-1]),
        )

    def _compute_energy_physical_cartesian(self):
        r"""
        Chart-invariant physical energy evaluated from blended Cartesian :math:`\mathbf{m}`.

        Used by thermodynamic and closure gates so atlas chart representation changes
        are not mistaken for physical energy increments.  Magnetization is taken from
        :meth:`_interpolate_m_cart_numpy` (atlas-weighted blend), not from a single chart.

        Returns
        -------
        numpy.ndarray
            ``[E_ex, E_ua, E_ca, E_deg, E_zee, E_tot]`` in consistent units.
        """
        self._interpolate_m_cart_numpy(target=self._thermo_m_curr)
        m = self._thermo_m_curr
        g = ufl.grad(m)

        W_ex = PETSc.ScalarType(0.0)
        W_ua = PETSc.ScalarType(0.0)
        W_ca = PETSc.ScalarType(0.0)
        W_deg = PETSc.ScalarType(0.0)
        W_zee = PETSc.ScalarType(0.0)
        W = PETSc.ScalarType(0.0)

        if "exchange" in self.energy_terms:
            W_ex = self.Aex * (
                g[0, 0] * g[0, 0]
                + g[0, 1] * g[0, 1]
                + g[0, 2] * g[0, 2]
                + g[1, 0] * g[1, 0]
                + g[1, 1] * g[1, 1]
                + g[1, 2] * g[1, 2]
                + g[2, 0] * g[2, 0]
                + g[2, 1] * g[2, 1]
                + g[2, 2] * g[2, 2]
            )
            W += W_ex

        if "uniaxial_anisotropy" in self.energy_terms:
            W_ua = 0.5 * self.Q1 * self.K * (1 - ufl.dot(self.axis_uni_anisotropy, m) ** 2)
            W += W_ua

        if "cubic_anisotropy" in self.energy_terms:
            f1 = m[0] ** 2 * m[1] ** 2 + m[1] ** 2 * m[2] ** 2 + m[2] ** 2 * m[0] ** 2
            f2 = m[0] ** 2 * m[1] ** 2 * m[2] ** 2
            W_ca = 0.5 * self.Q1 * self.K * (f1 + self.K2K1_ratio * f2)
            W += W_ca

        # Demag term only exists in the demag solver (nodemag leaves W_deg = 0).
        if "demag" in self.energy_terms:
            h_deg_expr = -ufl.grad(self.u_mixed.sub(2))
            W_deg = -0.5 * self.demag_ramp_constant * self.Ms * ufl.inner(h_deg_expr, m)
            W += W_deg

        if "zeeman" in self.energy_terms:
            # h_zee is already ramped in update_zeeman_ramp_factor().
            W_zee = -self.Ms * ufl.inner(self.h_zee, m)
            W += W_zee

        ex = self.domain.comm.allreduce(fem.assemble_scalar(fem.form(W_ex * self.dv)), op=MPI.SUM)
        ua = self.domain.comm.allreduce(fem.assemble_scalar(fem.form(W_ua * self.dv)), op=MPI.SUM)
        ca = self.domain.comm.allreduce(fem.assemble_scalar(fem.form(W_ca * self.dv)), op=MPI.SUM)
        deg = self.domain.comm.allreduce(fem.assemble_scalar(fem.form(W_deg * self.dv)), op=MPI.SUM)
        zee = self.domain.comm.allreduce(fem.assemble_scalar(fem.form(W_zee * self.dv)), op=MPI.SUM)
        tot = self.domain.comm.allreduce(fem.assemble_scalar(fem.form(W * self.dv)), op=MPI.SUM)
        return np.array([ex, ua, ca, deg, zee, tot], dtype=np.float64)

    def _backup_solution(self):
        """
        Snapshot dynamic state at the last accepted step before a new attempt.

        Captures both chart fields, full BDF history, and (demag only) the mixed
        potential so :meth:`_rollback_to_step_backup` can restore a rejected trial.
        """
        self._qA_backup = self.qA.x.array.copy()
        self._qA_prev_backup = self.qA_prev.x.array.copy()
        self._qA_prev_prev_backup = self.qA_prev_prev.x.array.copy()
        self._qB_backup = self.qB.x.array.copy()
        self._qB_prev_backup = self.qB_prev.x.array.copy()
        self._qB_prev_prev_backup = self.qB_prev_prev.x.array.copy()
        # Thermo gate evaluates blended m with the backed-up qB state.
        self._qB_backup_for_thermo = self._qB_backup.copy()
        self._bdf_active_order_backup = self._bdf_active_order
        self._bdf_steps_since_restart_backup = self._bdf_steps_since_restart
        self._dt_prev_value_backup = self._dt_prev_value
        if "demag" in self.energy_terms:
            phi_mixed = self.u_mixed.sub(2)
            self._phi_backup = phi_mixed.x.array.copy()

    def _rollback_state(self, qA_bk, qA_prev_bk, phi_bk=None):
        """
        Restore chart/BDF state from a per-step backup.

        Parameters
        ----------
        qA_bk, qA_prev_bk : array_like
            Backed-up current and previous ``qA`` DoF vectors.
        phi_bk : array_like, optional
            Backed-up demag potential DoFs; restored only when provided.
        """
        self.qA.x.array[:] = qA_bk
        self.qA_prev.x.array[:] = qA_prev_bk
        self.qA.x.scatter_forward()
        self.qA_prev.x.scatter_forward()
        if hasattr(self, "_qA_prev_prev_backup"):
            self.qA_prev_prev.x.array[:] = self._qA_prev_prev_backup
            self.qA_prev_prev.x.scatter_forward()

        if hasattr(self, "_qB_backup"):
            self.qB.x.array[:] = self._qB_backup
            self.qB_prev.x.array[:] = self._qB_prev_backup
            self.qB_prev_prev.x.array[:] = self._qB_prev_prev_backup
            self.qB.x.scatter_forward()
            self.qB_prev.x.scatter_forward()
            self.qB_prev_prev.x.scatter_forward()
            self._qB_backup_for_thermo = self._qB_backup.copy()

        if phi_bk is not None:
            phi_mixed = self.u_mixed.sub(2)
            phi_mixed.x.array[:] = phi_bk
            phi_mixed.x.scatter_forward()
            self.u_mixed.x.scatter_forward()

        if hasattr(self, "_bdf_active_order_backup"):
            self._bdf_active_order = self._bdf_active_order_backup
            self._bdf_steps_since_restart = self._bdf_steps_since_restart_backup
            self._dt_prev_value = self._dt_prev_value_backup

    def _rebase_thermo_baseline_for_control_transition(self):
        """
        Rebase thermo accepted baseline after a continuation control jump.

        This keeps the thermo gate consistent with the *current* model
        parameters (demag/zeeman ramp state) before evaluating dynamic
        dissipation in the new continuation regime.
        """
        split_components = self._compute_energy_quiet()
        split_total = float(split_components[-1])
        physical_components = self._compute_energy_physical_cartesian()
        physical_total = float(physical_components[-1])
        self.state_monitor["thermo_last_energy"] = (
            split_total if self.thermo_use_split_energy else physical_total
        )
        self.state_monitor["thermo_last_energy_physical"] = physical_total
        self.state_monitor["thermo_last_components"] = np.array(split_components, dtype=np.float64)
        self.state_monitor["thermo_last_components_physical"] = np.array(
            physical_components, dtype=np.float64
        )

    # ==================================================================
    # State backtracking (residual gate D1, thermodynamic gate E1)
    # ==================================================================

    def _thermo_backtracking_search(
        self,
        step,
        qA_trial,
        qB_trial,
        prev_energy,
        prev_components,
        prev_components_physical,
        zeeman_prev,
        demag_ramp_prev,
        effective_gate,
    ):
        r"""
        Physical energy line search before cutting ``dt`` on thermo failure.

        Tries :math:`\lambda = \texttt{shrink}, \texttt{shrink}^2, \ldots` along
        ``_set_state_from_backup_trial_blend``. Each trial must pass the residual gate
        (if active) and the thermodynamic inequality
        :math:`\Delta E \le \max(0,-W_{\mathrm{ext}}) + B` (plus optional physical guard).
        Recomputes energies on every trial. Returns metrics dict or ``None``.
        """
        lam = 1.0
        shrink = float(min(max(self.thermo_backtracking_shrink, 0.05), 0.95))
        min_lam = float(max(self.thermo_backtracking_min_lambda, 1e-8))
        max_iter = max(1, int(self.thermo_backtracking_max_iter))
        prev_energy_chart = (
            float(prev_components[-1])
            if prev_components is not None and len(prev_components) > 0
            else float(prev_energy)
        )
        prev_demag_phys = (
            float(prev_components_physical[3])
            if prev_components_physical is not None and len(prev_components_physical) > 3
            else None
        )

        for _ in range(max_iter):
            lam *= shrink
            if lam < min_lam:
                break
            self._set_state_from_backup_trial_blend(lam, qA_trial, qB_trial)

            residual_norm = self._compute_residual_norm()
            if effective_gate is not None and residual_norm > float(effective_gate):
                continue

            energy_components = self._compute_energy_quiet()
            energy_components_physical = self._compute_energy_physical_cartesian()
            current_energy_total = float(energy_components[-1])
            current_energy_total_physical = float(energy_components_physical[-1])
            current_energy_gate = (
                current_energy_total
                if self.thermo_use_split_energy
                else current_energy_total_physical
            )
            thermo_residual, thermo_budget, dissipation, external_work = self._evaluate_thermodynamic_residual(
                prev_energy=float(prev_energy),
                curr_energy=float(current_energy_gate),
                chart_energy_delta=float(current_energy_total - prev_energy_chart),
                prev_qA=self._qA_backup,
                dt_val=float(self.dt.value),
                nonlinear_residual=float(residual_norm),
                zeeman_prev=zeeman_prev,
                demag_ramp_prev=demag_ramp_prev,
                prev_demag_energy_physical=prev_demag_phys,
            )
            delta_energy = float(current_energy_gate - prev_energy)
            allowed_increase = float(max(0.0, -external_work) + thermo_budget)
            if delta_energy > allowed_increase:
                continue

            delta_energy_phys = None
            if self.thermo_enable_physical_guard:
                prev_energy_phys = float(
                    self.state_monitor.get("thermo_last_energy_physical", current_energy_total_physical)
                )
                delta_energy_phys = float(current_energy_total_physical - prev_energy_phys)
                phys_scale = max(1.0, abs(prev_energy_phys), abs(current_energy_total_physical))
                allowed_phys_base = allowed_increase + float(self.thermo_physical_guard_rel_floor) * phys_scale
                phys_delta_ema = self.state_monitor.get("thermo_phys_delta_ema", None)
                adaptive_phys_floor = 0.0
                if (
                    phys_delta_ema is not None
                    and step >= int(self.thermo_physical_guard_warmup_steps)
                    and np.isfinite(float(phys_delta_ema))
                ):
                    adaptive_phys_floor = float(self.thermo_physical_guard_ema_factor) * float(
                        phys_delta_ema
                    )
                adaptive_phys_floor = max(float(self.thermo_physical_guard_abs_floor), adaptive_phys_floor)
                allowed_phys = float(allowed_phys_base + adaptive_phys_floor)
                if delta_energy_phys > allowed_phys:
                    continue

            return {
                "lambda": float(lam),
                "residual_norm": float(residual_norm),
                "energy_components": energy_components,
                "energy_components_physical": np.array(energy_components_physical, dtype=np.float64),
                "energy_total": float(current_energy_total),
                "energy_total_physical": float(current_energy_total_physical),
                "thermo_residual": float(thermo_residual),
                "delta_energy_phys": float(delta_energy_phys) if delta_energy_phys is not None else None,
            }
        return None

    def _residual_backtracking_search(self, qA_trial, qB_trial, effective_gate):
        r"""
        Algebraic residual line search before merit/soft gate or ``dt`` cut.

        SNES may converge while ``\|F(q)\|`` exceeds the EMA-based ``effective_gate``.
        Searches :math:`\lambda = \texttt{shrink}, \texttt{shrink}^2, \ldots` (first try
        :math:`\lambda=\texttt{shrink}`, never retests :math:`\lambda=1`) until
        ``\|F(q(\lambda))\| \le`` gate. On success the damped state is left in place;
        the caller must refresh energies via ``_compute_acceptance_energies``. On failure
        the SNES trial (:math:`\lambda=1`) is restored so merit/soft gates (D2/D3) evaluate
        the full update. Returns dict or ``None``.
        """
        if effective_gate is None or not bool(self._residual_backtracking_enable):
            return None
        gate = float(effective_gate)
        if not np.isfinite(gate) or gate <= 0.0:
            return None

        lam = 1.0
        shrink = float(min(max(self._residual_backtracking_shrink, 0.05), 0.95))
        min_lam = float(max(self._residual_backtracking_min_lambda, 1e-8))
        max_iter = max(1, int(self._residual_backtracking_max_iter))
        state_touched = False
        for _ in range(max_iter):
            lam *= shrink
            if lam < min_lam:
                break
            self._set_state_from_backup_trial_blend(lam, qA_trial, qB_trial)
            state_touched = True
            residual_norm = float(self._compute_residual_norm())
            if residual_norm <= gate:
                return {"lambda": float(lam), "residual_norm": residual_norm}
        if state_touched:
            self._set_state_from_backup_trial_blend(1.0, qA_trial, qB_trial)
        return None

    def _evaluate_residual_merit(self, residual_before, residual_after, effective_gate):
        """
        Merit-style residual acceptance around gate failures.

        Returns dict with:
          - accepted: bool
          - limit: relaxed residual ceiling (None if not used)
          - ratio_to_gate, contraction, rho: diagnostics (optional)
        """
        if effective_gate is None:
            return {"accepted": True, "limit": None}
        gate = float(effective_gate)
        if not np.isfinite(gate) or gate <= 0.0:
            return {"accepted": True, "limit": None}
        r_after = float(residual_after)
        if r_after <= gate:
            return {"accepted": True, "limit": gate}
        if not bool(self._residual_merit_enable):
            return {
                "accepted": False,
                "limit": gate,
                "ratio_to_gate": r_after / max(gate, 1e-30),
            }
        r_before = float(residual_before)
        if (not np.isfinite(r_before)) or r_before <= 0.0:
            return {
                "accepted": False,
                "limit": gate,
                "ratio_to_gate": r_after / max(gate, 1e-30),
            }

        ratio_to_gate = r_after / max(gate, 1e-30)
        contraction = r_after / max(r_before, 1e-30)
        predicted_reduction = max(r_before - gate, 1e-14)
        actual_reduction = max(0.0, r_before - r_after)
        rho = actual_reduction / predicted_reduction

        relaxed_limit = max(
            float(gate) * float(self._residual_merit_gate_relax),
            float(self.nonlinear_residual_gate_floor) * float(self._residual_merit_floor_mult),
        )
        accepted = (
            r_after <= relaxed_limit
            and contraction <= float(self._residual_merit_required_contraction)
            and rho >= float(self._residual_merit_required_rho)
        )
        return {
            "accepted": bool(accepted),
            "limit": float(relaxed_limit),
            "ratio_to_gate": float(ratio_to_gate),
            "contraction": float(contraction),
            "rho": float(rho),
        }

    def _is_snes_reason_acceptable(self, snes_reason: int) -> bool:
        """Reason-aware nonlinear acceptance filter."""
        if snes_reason <= 0:
            return False
        if not self.strict_snes_reason_check:
            return True
        return int(snes_reason) not in self.disallowed_snes_reasons

    # ==================================================================
    # Physical stopping criteria and SNES globalization
    # ==================================================================

    def _check_physical_convergence(self, field_prev_qA, field_prev_qB, current_energy):
        """Check physical steady state using rate-based chart-invariant metrics."""
        st_mon = self.state_monitor if self.state_monitor is not None else {}
        tol_mdot, tol_edot, steady_steps = convergence_tolerances(self.convergence_settings)

        steady_counter = st_mon.get("steady_counter", 0)
        last_energy = st_mon.get("last_energy", None)
        dt_step = max(float(self.dt.value), 1e-15)

        n_owned = self.V_q.dofmap.index_map.size_local
        n_total = n_owned + self.V_q.dofmap.index_map.num_ghosts
        m_prev = self._blend_m_from_qA_qB(
            field_prev_qA[: n_total * 2], field_prev_qB[: n_total * 2]
        )
        m_curr = self._blend_m_from_qA_qB(
            self.qA.x.array[: n_total * 2], self.qB.x.array[: n_total * 2]
        )
        dm_owned = m_curr[:n_owned] - m_prev[:n_owned]
        local_sq = np.sum(dm_owned**2) if len(dm_owned) > 0 else 0.0
        global_sq = self.domain.comm.allreduce(local_sq, op=MPI.SUM)
        global_n_owned = self.domain.comm.allreduce(n_owned, op=MPI.SUM)
        dm_rms = np.sqrt(global_sq / max(global_n_owned, 1))
        mdot_rms = dm_rms / dt_step
        energy_change = 0.0
        edot_abs = np.inf
        edot_scaled = np.inf
        if last_energy is not None:
            energy_change = abs(current_energy - last_energy)
            edot_abs = energy_change / dt_step
            energy_scale = max(abs(float(current_energy)), abs(float(last_energy)), 1.0)
            edot_scaled = edot_abs / max(energy_scale, 1e-30)

        if mdot_rms < tol_mdot and edot_scaled < tol_edot:
            steady_counter += 1
            if self.domain.comm.rank == 0:
                print_flush(
                    f"\n    [Steady state counter {steady_counter}/{steady_steps}]. "
                    f"||m_dot||_rms={mdot_rms:.2e}, |dE/dt|/max(|E|,1)={edot_scaled:.2e}"
                )
        else:
            steady_counter = 0

        if self.state_monitor is not None:
            self.state_monitor["steady_counter"] = steady_counter
            self.state_monitor["last_energy"] = current_energy
            self.state_monitor["last_physical_norm"] = float(mdot_rms)
            self.state_monitor["last_energy_change"] = float(edot_abs if np.isfinite(edot_abs) else 0.0)
            self.state_monitor["last_energy_change_scaled"] = float(edot_scaled if np.isfinite(edot_scaled) else 0.0)
            self.state_monitor["last_physical_field_label"] = "m_dot_rms"

        if steady_counter >= steady_steps:
            if self.domain.comm.rank == 0:
                print_flush(f"\n  === Physical steady state at t={self.t:.3e} ===")
            return True
        return False

    def _distributed_percentile_owned(self, values, quantile, n_bins=256):
        """MPI-safe percentile estimate from owned local values via histogram reduction."""
        q = float(min(max(quantile, 0.0), 1.0))
        vals = np.asarray(values, dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        vals = vals[vals >= 0.0]

        local_count = np.array([vals.size], dtype=np.int64)
        global_count = np.array([0], dtype=np.int64)
        self.domain.comm.Allreduce(local_count, global_count, op=MPI.SUM)
        if int(global_count[0]) == 0:
            return 0.0

        local_max = np.array([float(np.max(vals)) if vals.size > 0 else 0.0], dtype=np.float64)
        global_max = np.array([0.0], dtype=np.float64)
        self.domain.comm.Allreduce(local_max, global_max, op=MPI.MAX)
        vmax = float(global_max[0])
        if vmax <= 0.0:
            return 0.0

        edges = np.linspace(0.0, vmax, int(n_bins) + 1, dtype=np.float64)
        local_hist, _ = np.histogram(vals, bins=edges)
        global_hist = np.zeros_like(local_hist, dtype=np.int64)
        self.domain.comm.Allreduce(local_hist.astype(np.int64), global_hist, op=MPI.SUM)

        total = int(np.sum(global_hist))
        if total <= 0:
            return 0.0
        target = q * (total - 1)
        cdf = np.cumsum(global_hist)
        idx = int(np.searchsorted(cdf, target, side="right"))
        idx = min(max(idx, 0), len(global_hist) - 1)
        cdf_prev = int(cdf[idx - 1]) if idx > 0 else 0
        bin_count = int(global_hist[idx])
        if bin_count <= 0:
            return float(edges[idx])
        frac = (target - cdf_prev) / max(bin_count, 1)
        frac = float(min(max(frac, 0.0), 1.0))
        return float(edges[idx] + frac * (edges[idx + 1] - edges[idx]))

    def _atlas_speed_expr(self):
        """Atlas-weighted physical speed proxy from both chart time derivatives."""
        m_t_A = ufl.dot(self.A_mu_i_chart(self.qA, +1.0), self.qA_dot)
        m_t_B = ufl.dot(self.A_mu_i_chart(self.qB, -1.0), self.qB_dot)
        return ufl.sqrt(
            ufl.max_value(
                self.wA * ufl.inner(m_t_A, m_t_A) + self.wB * ufl.inner(m_t_B, m_t_B),
                PETSc.ScalarType(0.0),
            )
        )

    def _update_norm_q_dot_field(self):
        """Populate ``norm_q_dot`` with the atlas-blended chart speed."""
        if not hasattr(self, "V_norm_q_dot"):
            self.V_norm_q_dot = fem.functionspace(self.domain, ("CG", 1))
            self.norm_q_dot = fem.Function(self.V_norm_q_dot)
        self.norm_q_dot.interpolate(
            fem.Expression(self._atlas_speed_expr(), self.V_norm_q_dot.element.interpolation_points)
        )
        self.norm_q_dot.x.scatter_forward()

    # ==================================================================
    # Adaptive timestep control (CFL cap + embedded BDF error)
    # ==================================================================

    def _update_timestep_cfl(self):
        """Update timestep: primary error control with CFL safety cap."""
        dt_old = float(self.dt.value)

        self._update_norm_q_dot_field()

        n_owned = (
            self.V_norm_q_dot.dofmap.index_map.size_local
            * self.V_norm_q_dot.dofmap.index_map_bs
        )
        speed_q99 = self._distributed_percentile_owned(self.norm_q_dot.x.array[:n_owned], 0.99)
        self.state_monitor["last_cfl_speed_q99"] = float(speed_q99)

        eps = 1e-6
        dt_cfl_cap = float(self.CFL * self.h_min / max(speed_q99, eps))
        dt_cfl_cap = max(self.dt_min, min(dt_cfl_cap, self.dt_max))

        err_norm = self.state_monitor.get("last_temporal_error_norm", None)
        if err_norm is None or not np.isfinite(err_norm) or err_norm <= 0.0:
            dt_error = self.dt_max
        else:
            # LTE-like controller: error ~ O(dt^(p+1)), p=2 for active BDF2.
            p = 2 if self._bdf_active_order == 2 else 1
            exponent = 1.0 / float(p + 1)
            factor = self.error_headroom * (float(err_norm) ** (-exponent))
            factor = float(min(max(factor, self.min_shrink), self.max_growth))
            dt_error = max(self.dt_min, min(dt_old * factor, self.dt_max))

        thermo_cap = self.state_monitor.get("thermo_dt_cap", self.dt_max)
        if thermo_cap is None or (not np.isfinite(float(thermo_cap))) or float(thermo_cap) <= 0.0:
            thermo_cap = self.dt_max
        thermo_cap = float(max(self.dt_min, min(float(thermo_cap), self.dt_max)))
        cooldown = int(self.state_monitor.get("thermo_dt_cap_cooldown_steps", 0))
        if cooldown > 0:
            # Keep cap active for a short horizon, but relax slowly.
            thermo_cap = min(self.dt_max, max(thermo_cap, dt_old * self._thermo_cap_growth_per_step))
            self.state_monitor["thermo_dt_cap"] = float(thermo_cap)
            self.state_monitor["thermo_dt_cap_cooldown_steps"] = cooldown - 1
        else:
            self.state_monitor["thermo_dt_cap"] = float(self.dt_max)
            self.state_monitor["thermo_dt_cap_cooldown_steps"] = 0
            thermo_cap = self.dt_max

        nonthermo_cap = self.state_monitor.get("nonthermo_dt_cap", self.dt_max)
        if (
            nonthermo_cap is None
            or (not np.isfinite(float(nonthermo_cap)))
            or float(nonthermo_cap) <= 0.0
        ):
            nonthermo_cap = self.dt_max
        nonthermo_cap = float(max(self.dt_min, min(float(nonthermo_cap), self.dt_max)))
        nonthermo_cooldown = int(self.state_monitor.get("nonthermo_dt_cap_cooldown_steps", 0))
        if nonthermo_cooldown > 0:
            # Keep cap active after nonlinear gate rejections to reduce
            # repeated regrow->reject cycles.
            clean_steps = int(self.state_monitor.get("nonthermo_clean_accepted_steps", 0))
            recovery_growth = float(self._nonthermo_cap_growth_per_step)
            if clean_steps >= int(self._nonthermo_recovery_clean_steps):
                recovery_growth = max(
                    recovery_growth, float(self._nonthermo_cap_recovery_growth_per_step)
                )
            nonthermo_cap = min(
                self.dt_max, max(nonthermo_cap, dt_old * recovery_growth)
            )
            self.state_monitor["nonthermo_dt_cap"] = float(nonthermo_cap)
            self.state_monitor["nonthermo_dt_cap_cooldown_steps"] = nonthermo_cooldown - 1
        else:
            self.state_monitor["nonthermo_dt_cap"] = float(self.dt_max)
            self.state_monitor["nonthermo_dt_cap_cooldown_steps"] = 0
            nonthermo_cap = self.dt_max

        continuation_cap = self.state_monitor.get("continuation_transition_dt_cap", self.dt_max)
        if (
            continuation_cap is None
            or (not np.isfinite(float(continuation_cap)))
            or float(continuation_cap) <= 0.0
        ):
            continuation_cap = self.dt_max
        continuation_cap = float(max(self.dt_min, min(float(continuation_cap), self.dt_max)))
        continuation_cooldown = int(
            self.state_monitor.get("continuation_transition_cooldown_steps", 0)
        )
        if continuation_cooldown > 0:
            continuation_cap = min(
                self.dt_max,
                max(
                    continuation_cap,
                    dt_old * float(self._continuation_transition_cap_growth_per_step),
                ),
            )
            self.state_monitor["continuation_transition_dt_cap"] = float(continuation_cap)
            self.state_monitor["continuation_transition_cooldown_steps"] = continuation_cooldown - 1
        else:
            self.state_monitor["continuation_transition_dt_cap"] = float(self.dt_max)
            self.state_monitor["continuation_transition_cooldown_steps"] = 0
            continuation_cap = self.dt_max

        dt_target = min(
            dt_error,
            dt_cfl_cap,
            thermo_cap,
            nonthermo_cap,
            continuation_cap,
            self.dt_max,
        )
        dt_bounded = max(self.dt_min, min(dt_target, self.dt_max))
        dt_smooth = self.smooth_factor * dt_bounded + (1.0 - self.smooth_factor) * dt_old

        nl_cooldown = int(self.state_monitor.get("nonlinear_retry_cooldown_steps", 0))
        if nl_cooldown > 0:
            growth_cap = min(self.max_growth, self._nonlinear_retry_growth_cap)
            self.state_monitor["nonlinear_retry_cooldown_steps"] = nl_cooldown - 1
        else:
            growth_cap = self.max_growth
        dt_rate_limited = max(min(dt_smooth, dt_old * growth_cap), dt_old * self.min_shrink)
        self.state_monitor["last_dt_error_target"] = float(dt_error)
        self.state_monitor["last_dt_cfl_cap"] = float(dt_cfl_cap)
        self.state_monitor["last_dt_thermo_cap"] = float(thermo_cap)
        self.state_monitor["last_dt_nonthermo_cap"] = float(nonthermo_cap)
        self.state_monitor["last_dt_continuation_cap"] = float(continuation_cap)
        self.state_monitor["last_dt_target_pre_smooth"] = float(dt_target)
        self.dt.value = self.domain.comm.bcast(dt_rate_limited, root=0)

    def _set_snes_globalization_mode(self, mode):
        """Switch SNES globalization mode only when needed."""
        target = self._snes_type_default if mode == "default" else self._snes_type_retry
        if getattr(self, "_snes_current_type", None) != target:
            self.solver_dynamic.setType(target)
            self._snes_current_type = target

    def _compute_temporal_error_norm(self):
        """
        Compute a variable-step-consistent BDF2 temporal defect norm in Cartesian m.

        Uses the difference between BDF2 and BE time derivatives, then rescales
        to an m-like defect (dt * derivative defect) for robust normalization.
        Uses owned qA/qB DOFs only to avoid duplicate ghost contributions.
        """
        if self._bdf_active_order != 2:
            return None

        n_owned = self.V_q.dofmap.index_map.size_local * self.V_q.dofmap.index_map_bs
        if n_owned <= 0:
            return None

        n_nodes_owned = self.V_q.dofmap.index_map.size_local
        n_q_owned = n_nodes_owned * 2
        qA = self.qA.x.array[:n_q_owned]
        qA_nm1 = self.qA_prev.x.array[:n_q_owned]
        qA_nm2 = self.qA_prev_prev.x.array[:n_q_owned]
        qB = self.qB.x.array[:n_q_owned]
        qB_nm1 = self.qB_prev.x.array[:n_q_owned]
        qB_nm2 = self.qB_prev_prev.x.array[:n_q_owned]

        m_n = self._blend_m_from_qA_qB(qA, qB)
        m_nm1 = self._blend_m_from_qA_qB(qA_nm1, qB_nm1)
        m_nm2 = self._blend_m_from_qA_qB(qA_nm2, qB_nm2)

        dt_n = max(float(self.dt.value), 1e-15)
        c0 = float(self._bdf_c0.value)
        c1 = float(self._bdf_c1.value)
        c2 = float(self._bdf_c2.value)
        m_dot_bdf2 = (c0 * m_n + c1 * m_nm1 + c2 * m_nm2) / dt_n
        m_dot_be = (m_n - m_nm1) / dt_n
        err_m = dt_n * (m_dot_bdf2 - m_dot_be)
        den_m = self.error_abs_tol + self.error_rel_tol * np.maximum.reduce(
            [np.abs(m_n), np.abs(m_nm1), np.abs(m_nm2)]
        )

        local_sum = np.array(
            [
                float(np.sum((err_m / den_m) ** 2)),
                float(err_m.size),
            ],
            dtype=np.float64,
        )
        global_sum = np.zeros(2, dtype=np.float64)
        self.domain.comm.Allreduce(local_sum, global_sum, op=MPI.SUM)
        if global_sum[1] <= 0.0:
            return None
        return float(np.sqrt(global_sum[0] / global_sum[1]))

    # ------------------------------------------------------------------
    # solve_problem helpers
    #
    # Per step: ``_run_timestep_preamble`` → retry loop
    # (``_run_accept_attempt`` or ``_handle_step_rejection``) →
    # ``_finalize_accepted_step``.
    # ------------------------------------------------------------------

    # --- Per-attempt state records (private dataclasses) ---

    @dataclass
    class _StepPreambleState:
        """Per-step context computed before the inner retry loop."""

        zeeman_prev: Any
        demag_ramp_prev: float | None
        thermo_guard_active: bool
        transition_reconcile_active: bool
        transition_rebased: bool

    @dataclass
    class _AcceptAttemptState:
        """Mutable workspace for one inner-loop acceptance attempt (gates A–F)."""

        qA_trial: np.ndarray | None = None
        qB_trial: np.ndarray | None = None
        residual_norm_before: float = 0.0
        residual_norm: float = 0.0
        ksp_iters: int = 0
        snes_iters: int = 0
        effective_gate: float | None = None
        accepted_backtracking_lambda: float = 1.0
        residual_soft_violation: bool = False
        residual_soft_ratio: float = 1.0
        energy_components: Any = None
        current_energy_total: float = 0.0
        energy_components_physical: Any = None
        current_energy_total_physical: float = 0.0
        r_max_attempt: float = 0.0
        accepted_thermo_residual: float | None = None
        accepted_delta_energy_phys: float | None = None
        accepted_closure_jump_energy: float = 0.0

    @dataclass
    class _RetryLoopState:
        """Counters and caps for the fixed-``dt`` inner retry loop at one step."""

        retry_count: int = 0
        thermo_retry_count: int = 0
        nonthermo_retry_count: int = 0
        outer_retry_count: int = 0
        outer_iter_cap: int = 1
        disallowed_reason_same_dt_recoveries: int = 0

    @dataclass
    class _AcceptedStepData:
        """Immutable snapshot passed to :meth:`_finalize_accepted_step` after success."""

        energy_components: Any
        energy_total: float
        energy_total_physical: float
        energy_components_physical: np.ndarray
        residual_norm: float
        ksp_iters: int
        snes_iters: int
        thermo_residual: float | None
        delta_energy_phys: float | None
        backtracking_lambda: float
        closure_jump_energy: float

    # --- One-time init and per-step preamble ---

    def _init_solve_problem_runtime(self):
        """One-time setup before the main step loop in :meth:`solve_problem`.

        Seeds the thermodynamic baseline from the current energy. On demag
        restarts, ``init_dynamic`` must already have re-equilibrated ``phi``
        with the live BEM so this baseline does not embed a remesh/operator
        jump that the first dynamical step would then be billed for.
        """
        self._open_xdmf_time_only_outputs()

        self.timestep_stagnation_counter = 0
        self.prev_timestep_residual = None
        self._backup_solution()
        self._ensure_thermo_buffers()

        init_energy_components = self.compute_energy()
        init_energy_total = init_energy_components[-1]
        init_energy_physical = self._compute_energy_physical_cartesian()
        init_energy_total_physical = init_energy_physical[-1]
        init_thermo_gate_energy = (
            init_energy_total if self.thermo_use_split_energy else init_energy_total_physical
        )
        self.state_monitor["thermo_last_energy"] = init_thermo_gate_energy
        self.state_monitor["thermo_last_energy_physical"] = init_energy_total_physical
        self.state_monitor["thermo_last_components"] = np.array(init_energy_components, dtype=np.float64)
        self.state_monitor["thermo_last_components_physical"] = np.array(
            init_energy_physical, dtype=np.float64
        )
        # These are unused by the thermo gate budgets directly, but we still reset
        # them because they are only diagnostic/optional state.
        self.state_monitor["thermo_energy_nm1"] = None
        self.state_monitor["thermo_energy_nm2"] = None
        self.state_monitor["thermo_dt_nm1"] = None

        time_restored = bool(getattr(self, "_time_checkpoint_restored", False))
        if not time_restored:
            # Default cold-start controller history.
            self.state_monitor["thermo_defect_ema"] = None
            self.state_monitor["thermo_phys_delta_ema"] = None
            self.state_monitor["be_recovery_steps_remaining"] = 0
            self.state_monitor["last_nonlinear_retry_mode"] = None
            self.state_monitor["accepted_residual_ema"] = None
            self.state_monitor["accepted_residual_ema_slow"] = None
            self.state_monitor["nonthermo_clean_accepted_steps"] = 0
            self.state_monitor["continuation_transition_cooldown_steps"] = 0
            self.state_monitor["continuation_transition_dt_cap"] = float(self.dt_max)
            self.state_monitor["continuation_transition_just_triggered"] = False
            self.state_monitor["continuation_transition_source"] = None
            self.state_monitor["continuation_reconcile_steps_remaining"] = 0
        else:
            # Preserve restored values. Only ensure required keys exist.
            self.state_monitor.setdefault("thermo_defect_ema", None)
            self.state_monitor.setdefault("thermo_phys_delta_ema", None)
            self.state_monitor.setdefault("be_recovery_steps_remaining", 0)
            self.state_monitor.setdefault("last_nonlinear_retry_mode", None)
            self.state_monitor.setdefault("accepted_residual_ema", None)
            self.state_monitor.setdefault("accepted_residual_ema_slow", None)
            self.state_monitor.setdefault("nonthermo_clean_accepted_steps", 0)
            self.state_monitor.setdefault("continuation_transition_cooldown_steps", 0)
            self.state_monitor.setdefault("continuation_transition_dt_cap", float(self.dt_max))
            self.state_monitor.setdefault("continuation_transition_just_triggered", False)
            self.state_monitor.setdefault("continuation_transition_source", None)
            self.state_monitor.setdefault("continuation_reconcile_steps_remaining", 0)
        self.state_monitor["last_closure_jump_energy"] = 0.0
        self.state_monitor["last_closure_jump_warn"] = 0
        self.state_monitor["last_residual_merit_accept"] = 0
        self.state_monitor["last_residual_merit_ratio"] = 1.0
        self.state_monitor["last_residual_merit_contraction"] = 1.0
        self.state_monitor["last_residual_merit_rho"] = 1.0

        if "zeeman" in self.energy_terms and hasattr(self.h_zee, "x"):
            self._h_zee_prev_accepted = self.h_zee.x.array.copy()
        else:
            self._h_zee_prev_accepted = None
        if "demag" in self.energy_terms:
            self._demag_ramp_prev_accepted = float(self.demag_ramp_constant.value)
        else:
            self._demag_ramp_prev_accepted = None

    def _open_timestep(self, step: int):
        """Per-step header: BDF recovery window and optional console output."""
        self._set_snes_globalization_mode("default")
        be_recovery_left = int(self.state_monitor.get("be_recovery_steps_remaining", 0))
        if be_recovery_left > 0:
            self._bdf_active_order = 1
            self._bdf_steps_since_restart = 0
            self.state_monitor["be_recovery_steps_remaining"] = be_recovery_left - 1
        if self.domain.comm.rank == 0 and self.printout:
            order_tag = f"BDF{self._bdf_active_order}"
            print_flush(
                f"\n~~~ Step = {step} : t= {self.t:.4e}, dt = {float(self.dt.value):.4e}  [{order_tag}]"
            )

    def _run_timestep_preamble(self, step: int) -> _StepPreambleState:
        """Continuation ramps, transition guard, magnetostatics, thermo/reconcile flags.

        Called once per outer step index before the inner retry loop. When a demag or
        Zeeman ramp completes, may shrink ``dt``, force BDF1, suspend the thermodynamic
        gate briefly (``thermo_guard_active=False``), and rebase thermodynamic baselines.
        """
        zeeman_prev = None
        if self._h_zee_prev_accepted is not None:
            zeeman_prev = self._h_zee_prev_accepted.copy()
        demag_ramp_prev = (
            float(self._demag_ramp_prev_accepted)
            if self._demag_ramp_prev_accepted is not None
            else None
        )

        self.update_demag_ramp_factor()
        self.update_zeeman_ramp_factor()
        transition_rebased = False
        if bool(self.state_monitor.get("continuation_transition_just_triggered", False)):
            dt_before = float(self.dt.value)
            dt_shrunk = max(
                self.dt_min,
                min(
                    dt_before,
                    dt_before * float(self._continuation_transition_dt_shrink_factor),
                ),
            )
            self.dt.value = dt_shrunk
            cd_cap = min(
                self.dt_max,
                max(
                    dt_shrunk,
                    dt_shrunk * float(self._continuation_transition_dt_cap_relax),
                ),
            )
            self.state_monitor["continuation_transition_dt_cap"] = float(cd_cap)
            self.state_monitor["thermo_dt_cap"] = min(
                float(self.state_monitor.get("thermo_dt_cap", self.dt_max)),
                float(cd_cap),
            )
            self.state_monitor["nonthermo_dt_cap"] = min(
                float(self.state_monitor.get("nonthermo_dt_cap", self.dt_max)),
                float(cd_cap),
            )
            self._bdf_active_order = 1
            self._bdf_steps_since_restart = 0
            self.state_monitor["be_recovery_steps_remaining"] = max(
                int(self.state_monitor.get("be_recovery_steps_remaining", 0)),
                max(0, int(self._continuation_transition_be_recovery_steps) - 1),
            )
            self._rebase_thermo_baseline_for_control_transition()
            transition_rebased = True
            self.state_monitor["continuation_reconcile_steps_remaining"] = max(
                int(self.state_monitor.get("continuation_reconcile_steps_remaining", 0)),
                int(self._continuation_transition_reconcile_steps),
            )
            self.state_monitor["continuation_transition_just_triggered"] = False
            if self.domain.comm.rank == 0 and self.printout:
                src = self.state_monitor.get("continuation_transition_source", "ramp")
                print_flush(
                    f"[Continuation Guard] {src}: dt {dt_before:.3e} -> {dt_shrunk:.3e}, "
                    f"BE recovery={int(self._continuation_transition_be_recovery_steps)}"
                )

        self.solve_magnetostatics(step)

        thermo_guard_active = self._thermo_check_active()
        transition_reconcile_active = (
            int(self.state_monitor.get("continuation_reconcile_steps_remaining", 0)) > 0
        )
        if transition_reconcile_active:
            thermo_guard_active = False
            if self.domain.comm.rank == 0 and self.printout:
                src = self.state_monitor.get("continuation_transition_source", "transition")
                print_flush(
                    f"[Continuation Reconcile] {src}: thermo gate suspended for this step"
                )
        if transition_rebased:
            if zeeman_prev is not None and hasattr(self.h_zee, "x"):
                zeeman_prev = self.h_zee.x.array.copy()
            if demag_ramp_prev is not None:
                demag_ramp_prev = float(self.demag_ramp_constant.value)

        return self._StepPreambleState(
            zeeman_prev=zeeman_prev,
            demag_ramp_prev=demag_ramp_prev,
            thermo_guard_active=thermo_guard_active,
            transition_reconcile_active=transition_reconcile_active,
            transition_rebased=transition_rebased,
        )

    def _initial_outer_iter_cap(self) -> int:
        outer_iter_cap = max(1, int(self.outer_split_max_iter))
        if "demag" in self.energy_terms:
            prev_phi_def = float(self.state_monitor.get("last_outer_split_phi_defect", 0.0))
            if prev_phi_def > float(self.outer_split_phi_growth_trigger):
                outer_iter_cap = max(outer_iter_cap, int(self.outer_split_min_iter_on_growth))
        return outer_iter_cap

    # --- Gates A–F: one inner-loop acceptance attempt ---
    # See :meth:`_run_accept_attempt` for call order.

    def _refresh_attempt_energies(self, att: _AcceptAttemptState) -> None:
        """Atlas guard plus chart/physical energy snapshot for gate evaluation."""
        att.r_max_attempt = self._enforce_atlas_radius_guard(silent=True)
        att.energy_components = self._compute_energy_quiet()
        att.current_energy_total = float(att.energy_components[-1])
        att.energy_components_physical = self._compute_energy_physical_cartesian()
        att.current_energy_total_physical = float(att.energy_components_physical[-1])

    def _pack_accepted_step(self, att: _AcceptAttemptState) -> _AcceptedStepData:
        """Collect per-attempt diagnostics into an immutable accepted-step record."""
        return self._AcceptedStepData(
            energy_components=att.energy_components,
            energy_total=att.current_energy_total,
            energy_total_physical=att.current_energy_total_physical,
            energy_components_physical=np.array(att.energy_components_physical, dtype=np.float64),
            residual_norm=att.residual_norm,
            ksp_iters=att.ksp_iters,
            snes_iters=att.snes_iters,
            thermo_residual=att.accepted_thermo_residual,
            delta_energy_phys=att.accepted_delta_energy_phys,
            backtracking_lambda=att.accepted_backtracking_lambda,
            closure_jump_energy=att.accepted_closure_jump_energy,
        )

    def _run_accept_attempt(
        self,
        step: int,
        preamble: _StepPreambleState,
        retry: _RetryLoopState,
        att: _AcceptAttemptState,
    ) -> _AcceptedStepData:
        r"""Run acceptance gates (A)–(F) for one inner-loop attempt.

        Call order (matches :doc:`theory_algorithms`):

        ====  =================================  ======================================
        Gate  Method                             Notes
        ====  =================================  ======================================
        A     :meth:`_run_nonlinear_solve`       ``solve_weak_form``; saves trial ``q``
        B     :meth:`_validate_algebraic_solve`  SNES/KSP reason; outer-split check
        C     :meth:`_refresh_attempt_energies`  ``_enforce_atlas_radius_guard`` + energies
        —     :meth:`_compute_effective_residual_gate`  builds :math:`g_{\mathrm{res}}`
        —     :meth:`_enforce_snes_reason_on_residual`  optional reason filter (B)
        D     :meth:`_apply_residual_gate`       D1/D2/D3 substages
        E     :meth:`_apply_thermodynamic_gate`  skipped if preamble disables thermo
        F     :meth:`_apply_closure_gate`        atlas closure + jump gate
        —     :meth:`_apply_residual_soft_accept_policy`  dt cap after D3 soft accept
        ====  =================================  ======================================

        Raises :class:`RuntimeError` on hard rejection (caught by :meth:`solve_problem`
        and routed to :meth:`_handle_step_rejection`). Returns a packed record on success.
        """
        # Gate (A): nonlinear solve — BDF weak form, optional outer split (demag).
        split_info = self._run_nonlinear_solve(retry.outer_iter_cap, att)
        # Gate (B): algebraic validity — SNES/KSP reasons, outer-split convergence.
        snes_reason, att.ksp_iters, att.snes_iters, att.residual_norm = (
            self._validate_algebraic_solve(split_info)
        )
        # Gate (C): atlas radius guard and energy snapshot for later gates.
        self._refresh_attempt_energies(att)
        att.effective_gate = self._compute_effective_residual_gate(step)
        self._enforce_snes_reason_on_residual(
            snes_reason, att.residual_norm, att.effective_gate
        )
        # Gate (D): residual — D1 backtrack, D2 merit accept, or D3 soft violation.
        self._apply_residual_gate(att)
        # Gate (E): thermodynamic dissipation; E1 backtrack if enabled (may be skipped).
        if preamble.thermo_guard_active:
            self._apply_thermodynamic_gate(step, preamble, att, retry)
        # Gate (F): reconcile inactive chart; check |ΔE_closure| across closure.
        self._apply_closure_gate(preamble, att)
        # After D3 soft accept: cap nonthermo dt growth for upcoming steps.
        self._apply_residual_soft_accept_policy(att)
        return self._pack_accepted_step(att)

    def _run_nonlinear_solve(self, outer_iter_cap: int, att: _AcceptAttemptState) -> dict:
        """Nonlinear outer-split solve and trial-state sync (gate A)."""
        att.residual_norm_before = float(self._compute_residual_norm())
        split_info = self.solve_weak_form(max_outer_override=outer_iter_cap)
        att.qA_trial = self.qA.x.array.copy()
        att.qB_trial = self.qB.x.array.copy()
        self._prepare_state_for_acceptance(apply_closure=False)
        return split_info

    def _validate_algebraic_solve(self, split_info: dict) -> tuple[int, int, int, float]:
        """SNES/KSP convergence and outer-split validity (gate B)."""
        snes_reason = int(self.solver_dynamic.getConvergedReason())
        ksp_reason = int(self.solver_dynamic.getKSP().getConvergedReason())
        convergence_reason = min(snes_reason, ksp_reason)
        if convergence_reason < 0:
            raise RuntimeError(
                f"SNES/KSP diverged (snes_reason={snes_reason}, ksp_reason={ksp_reason})"
            )
        if self.enforce_outer_split_acceptance and not bool(split_info.get("converged", False)):
            split_msg = (
                "Outer split not converged: "
                f"iters={int(split_info.get('iters', 0))}, "
                f"q_defect={float(split_info.get('q_defect', 0.0)):.3e}, "
                f"q_defect_abs={float(split_info.get('q_defect_abs', 0.0)):.3e}"
            )
            if "demag" in self.energy_terms:
                split_msg += (
                    f", phi_defect={float(split_info.get('phi_defect', 0.0)):.3e}, "
                    f"phi_defect_abs={float(split_info.get('phi_defect_abs', 0.0)):.3e}"
                )
            raise RuntimeError(split_msg)

        residual_norm = float(self._compute_residual_norm())
        ksp_iters = self.solver_dynamic.getKSP().getIterationNumber()
        snes_iters = self.solver_dynamic.getIterationNumber()
        return snes_reason, ksp_iters, snes_iters, residual_norm

    def _compute_effective_residual_gate(self, step: int) -> float | None:
        """Build the residual acceptance threshold for gate (D)."""
        residual_ema = self.state_monitor.get("accepted_residual_ema", None)
        residual_ema_slow = self.state_monitor.get("accepted_residual_ema_slow", None)
        dynamic_gate = None
        if (
            residual_ema is not None
            and step >= int(self.nonlinear_residual_gate_warmup_steps)
            and np.isfinite(float(residual_ema))
        ):
            dynamic_gate_fast = max(
                float(self.nonlinear_residual_gate_floor),
                float(self.nonlinear_residual_ema_factor) * float(residual_ema),
            )
            dynamic_gate = dynamic_gate_fast
            if (
                residual_ema_slow is not None
                and np.isfinite(float(residual_ema_slow))
                and float(residual_ema_slow) > 0.0
            ):
                dynamic_gate_slow = max(
                    float(self.nonlinear_residual_gate_floor),
                    float(self.nonlinear_residual_ema_factor) * float(residual_ema_slow),
                )
                dynamic_gate_floor = max(
                    float(self.nonlinear_residual_gate_floor),
                    float(dynamic_gate_slow)
                    / max(float(self._nonlinear_residual_gate_deflation_cap), 1.0),
                )
                dynamic_gate_cap = max(
                    float(self.nonlinear_residual_gate_floor),
                    float(self.nonlinear_residual_ema_factor)
                    * float(self._nonlinear_residual_gate_inflation_cap)
                    * float(residual_ema_slow),
                )
                dynamic_gate = min(
                    max(dynamic_gate_fast, dynamic_gate_floor),
                    dynamic_gate_cap,
                )
        abs_gate = None
        if (
            np.isfinite(float(self.nonlinear_residual_max_accept))
            and float(self.nonlinear_residual_max_accept) > 0.0
        ):
            abs_gate = float(self.nonlinear_residual_max_accept)
        effective_gate = dynamic_gate
        if abs_gate is not None:
            effective_gate = abs_gate if effective_gate is None else min(effective_gate, abs_gate)
        if effective_gate is not None:
            dt_ratio = float(self.dt.value) / max(float(self.dt_min), 1e-30)
            relax_max = max(1.0, float(self.nonlinear_residual_dt_relax_max))
            if dt_ratio >= 5.0:
                dt_relax = 1.0
            else:
                frac = min(max((5.0 - dt_ratio) / 4.0, 0.0), 1.0)
                dt_relax = 1.0 + frac * (relax_max - 1.0)
            effective_gate = float(effective_gate) * float(dt_relax)
        self.state_monitor["last_residual_gate"] = (
            float(effective_gate) if effective_gate is not None else None
        )
        return effective_gate

    def _enforce_snes_reason_on_residual(
        self, snes_reason: int, residual_norm: float, effective_gate: float | None
    ) -> None:
        """Optional SNES-reason filter after algebraic solve (gate B)."""
        if self._is_snes_reason_acceptable(snes_reason):
            return
        dt_ratio = float(self.dt.value) / max(float(self.dt_min), 1e-30)
        reason_relax = float(max(self.snes_disallowed_residual_factor, 1.0))
        if dt_ratio <= 2.0:
            reason_relax *= float(max(self.snes_disallowed_dtmin_relax, 1.0))
        residual_ema_any = self.state_monitor.get("accepted_residual_ema", None)
        disallowed_tol = (
            float(effective_gate)
            if effective_gate is not None
            else (
                max(
                    10.0 * float(self.nonlinear_residual_gate_floor),
                    50.0 * float(residual_ema_any),
                )
                if (
                    residual_ema_any is not None
                    and np.isfinite(float(residual_ema_any))
                    and float(residual_ema_any) > 0.0
                )
                else max(1e-3, 10.0 * float(self.nonlinear_residual_gate_floor))
            )
        )
        disallowed_tol *= reason_relax
        if int(snes_reason) == 4:
            disallowed_tol = max(disallowed_tol, 5e-2)
        if residual_norm > disallowed_tol:
            raise RuntimeError(
                "SNES converged for disallowed reason with large residual: "
                f"reason={snes_reason}, residual={residual_norm:.3e}, "
                f"tol={disallowed_tol:.3e}"
            )

    def _apply_residual_gate(self, att: _AcceptAttemptState) -> None:
        """Residual gate (D): D1 backtrack, D2 merit accept, or D3 soft violation.

        No-op when ``att.residual_norm <= att.effective_gate``. On D1 success the damped
        state is left in place and energies are refreshed. On D1 failure the SNES trial
        (λ=1) is restored before D2/D3. D3 sets ``att.residual_soft_violation`` without
        raising; :meth:`_apply_residual_soft_accept_policy` caps ``dt`` after accept.
        """
        if att.effective_gate is None or att.residual_norm <= float(att.effective_gate):
            return
        rb_info = self._residual_backtracking_search(
            qA_trial=att.qA_trial,
            qB_trial=att.qB_trial,
            effective_gate=att.effective_gate,
        )
        if rb_info is not None:
            att.accepted_backtracking_lambda = min(
                float(att.accepted_backtracking_lambda),
                float(rb_info["lambda"]),
            )
            att.residual_norm = float(rb_info["residual_norm"])
            self._prepare_state_for_acceptance(apply_closure=False)
            (
                att.energy_components,
                att.current_energy_total,
                att.energy_components_physical,
                att.current_energy_total_physical,
            ) = self._compute_acceptance_energies()
            att.r_max_attempt = self._enforce_atlas_radius_guard(silent=True)
            return

        self._prepare_state_for_acceptance(apply_closure=False)
        (
            att.energy_components,
            att.current_energy_total,
            att.energy_components_physical,
            att.current_energy_total_physical,
        ) = self._compute_acceptance_energies()
        merit = self._evaluate_residual_merit(
            residual_before=att.residual_norm_before,
            residual_after=att.residual_norm,
            effective_gate=att.effective_gate,
        )
        if bool(merit.get("accepted", False)):
            self.state_monitor["last_residual_merit_accept"] = 1
            self.state_monitor["last_residual_merit_ratio"] = float(merit.get("ratio_to_gate", 1.0))
            self.state_monitor["last_residual_merit_contraction"] = float(
                merit.get("contraction", 1.0)
            )
            self.state_monitor["last_residual_merit_rho"] = float(merit.get("rho", 1.0))
            if self.domain.comm.rank == 0 and self._debug_logs:
                print_flush(
                    "    [Residual Merit Accept] "
                    f"residual={att.residual_norm:.3e}, gate={float(att.effective_gate):.3e}, "
                    f"limit={float(merit.get('limit', att.effective_gate)):.3e}, "
                    f"ratio={float(merit.get('ratio_to_gate', 1.0)):.2f}, "
                    f"contraction={float(merit.get('contraction', 1.0)):.3e}, "
                    f"rho={float(merit.get('rho', 1.0)):.2f}"
                )
            return

        self.state_monitor["last_residual_merit_accept"] = 0
        self.state_monitor["last_residual_merit_ratio"] = float(merit.get("ratio_to_gate", np.inf))
        self.state_monitor["last_residual_merit_contraction"] = float(
            merit.get("contraction", np.inf)
        )
        self.state_monitor["last_residual_merit_rho"] = float(merit.get("rho", -np.inf))
        att.residual_soft_violation = True
        att.residual_soft_ratio = float(
            merit.get(
                "ratio_to_gate",
                float(att.residual_norm) / max(float(att.effective_gate), 1e-30),
            )
        )
        if self.domain.comm.rank == 0 and self._debug_logs:
            print_flush(
                "    [Residual Gate Soft] "
                f"residual={att.residual_norm:.3e}, gate={float(att.effective_gate):.3e}, "
                f"ratio={att.residual_soft_ratio:.2f}"
            )

    # ==================================================================
    # Thermodynamic residual assembly (gate E / E1)
    #
    # Used by :meth:`_apply_thermodynamic_gate` and :meth:`_thermo_backtracking_search`.
    # ==================================================================

    def _evaluate_thermodynamic_residual(
        self,
        prev_energy,
        curr_energy,
        chart_energy_delta,
        prev_qA,
        dt_val,
        nonlinear_residual,
        zeeman_prev=None,
        demag_ramp_prev=None,
        prev_demag_energy_physical=None,
    ):
        r"""
        Evaluate thermodynamic residual and acceptance budget for one trial state.

        Forms :math:`R = \Delta E + D_q + W_{\mathrm{ext}}` where :math:`D_q` uses the
        current ``qA``/``qB`` and ``qA_dot``/``qB_dot`` (form-consistent damping). The gate
        compares :math:`\Delta E` to ``max(0, -W_ext) + budget``; ``budget`` adds solver-noise
        floors from ``nonlinear_residual``, defect EMA, and (demag) ramp terms. See
        :doc:`theory_algorithms` (thermodynamic gate, :eq:`eq:thermo_residual`).
        """
        self._ensure_thermo_buffers()
        dt_safe = max(float(dt_val), 1e-15)

        # Form-consistent q-space dissipation.
        s2A_reg = ufl.max_value(self._sigma2(self.qA), self._sigma2_floor)
        s2B_reg = ufl.max_value(self._sigma2(self.qB), self._sigma2_floor)
        diss_q_local = fem.assemble_scalar(
            fem.form(
                self.alpha
                * self.Ms
                * (
                    self.wA * s2A_reg * ufl.inner(self.qA_dot, self.qA_dot)
                    + self.wB * s2B_reg * ufl.inner(self.qB_dot, self.qB_dot)
                )
                * self.dv
            )
        )
        dissipation_q = dt_safe * float(self.domain.comm.allreduce(diss_q_local, op=MPI.SUM))

        # Cartesian dissipation is diagnostics-only in atlas mode.
        diagnostics_active = bool(self.enable_thermo_diagnostics)
        dissipation_m = 0.0
        if diagnostics_active:
            self._interpolate_m_cart_numpy(target=self._thermo_m_curr)
            self._set_m_from_state_numpy(prev_qA, self._thermo_m_prev)
            self._thermo_dm_dt.x.array[:] = (
                self._thermo_m_curr.x.array[:] - self._thermo_m_prev.x.array[:]
            ) / dt_safe
            self._thermo_dm_dt.x.scatter_forward()
            diss_m_local = fem.assemble_scalar(
                fem.form(ufl.inner(self._thermo_dm_dt, self._thermo_dm_dt) * self.dv)
            )
            dissipation_m = float(self.alpha) * dt_safe * float(
                self.domain.comm.allreduce(diss_m_local, op=MPI.SUM)
            )

        external_work = 0.0
        if (
            "zeeman" in self.energy_terms
            and zeeman_prev is not None
            and hasattr(self.h_zee, "x")
            and hasattr(self, "_thermo_dh_zee")
        ):
            if not diagnostics_active:
                # Thermo external work uses m_curr; compute it only when needed.
                self._interpolate_m_cart_numpy(target=self._thermo_m_curr)
            self._thermo_dh_zee.x.array[:] = self.h_zee.x.array[:] - zeeman_prev[:]
            self._thermo_dh_zee.x.scatter_forward()
            external_work = self.domain.comm.allreduce(
                fem.assemble_scalar(
                    fem.form(self.Ms * ufl.inner(self._thermo_dh_zee, self._thermo_m_curr) * self.dv)
                ),
                op=MPI.SUM,
            )

        delta_energy = curr_energy - prev_energy
        residual = delta_energy + dissipation_q + external_work

        eps = np.finfo(np.float64).eps
        budget = 128.0 * eps * max(1.0, abs(prev_energy), abs(curr_energy))
        budget += dt_safe * abs(float(nonlinear_residual))
        ema_floor = 0.0
        thermo_def_ema = self.state_monitor.get("thermo_defect_ema", None)
        step_now = int(getattr(self, "step_now", 0))
        if (
            thermo_def_ema is not None
            and step_now >= int(self.thermo_defect_ema_warmup_steps)
            and np.isfinite(float(thermo_def_ema))
        ):
            ema_floor = max(0.0, float(self.thermo_defect_ema_factor) * float(thermo_def_ema))
            budget += ema_floor
        if "demag" in self.energy_terms:
            # Demag-dominated problems carry large absolute energies; include a
            # relative floor to avoid false rejections from linear-solve noise.
            budget += float(self.thermo_demag_energy_rel_floor) * max(
                1.0, abs(prev_energy), abs(curr_energy)
            )
        ramp_budget = 0.0
        if (
            "demag" in self.energy_terms
            and demag_ramp_prev is not None
            and prev_demag_energy_physical is not None
        ):
            lam_prev = float(demag_ramp_prev)
            lam_curr = float(self.demag_ramp_constant.value)
            dlam = abs(lam_curr - lam_prev)
            # Continuation forcing injects/removes demag energy as lambda varies.
            # Account for this explicitly to avoid misclassifying forced steps as
            # thermodynamic violations.
            # prev_demag_energy_physical already includes lambda scaling.
            # Recover an unramped estimate before applying |Δlambda|.
            demag_unramped = abs(float(prev_demag_energy_physical))
            if abs(lam_prev) > 1e-12:
                demag_unramped /= abs(lam_prev)
            ramp_budget = dlam * demag_unramped
            budget += ramp_budget

        self._thermo_last_diss_q_form = float(dissipation_q)
        self._thermo_last_diss_m_cart = float(dissipation_m)
        self._thermo_last_op_budget = 0.0
        self._thermo_last_ema_floor = float(ema_floor)
        # Representation-change contribution (atlas chart energy - physical energy).
        self._thermo_last_switch_work = float(chart_energy_delta - delta_energy)
        self._thermo_last_ramp_budget = float(ramp_budget)
        self._atlas_thermo_metrics = {}

        return residual, budget, dissipation_q, external_work

    def _log_thermo_diagnostics(
        self,
        step,
        retry_count,
        accepted_prev_energy,
        accepted_prev_components,
        current_components,
        thermo_residual,
        thermo_budget,
        dissipation,
        external_work,
        delta_energy,
        allowed_increase,
        switched_after_accept,
        r_max_attempt,
    ):
        """Base diagnostics plus atlas-specific thermo/closure metrics."""
        mismatch_pairs, owned_refs, mismatch_nodes, owned_nodes = self._chart_consistency_metrics()
        if self.domain.comm.rank == 0:
            labels = self._energy_component_labels()
            curr = np.asarray(current_components, dtype=np.float64)
            prev = None if accepted_prev_components is None else np.asarray(accepted_prev_components, dtype=np.float64)
            pair_rate = (mismatch_pairs / max(owned_refs, 1)) * 100.0
            node_rate = (mismatch_nodes / max(owned_nodes, 1)) * 100.0
            print_flush(
                f"\n    [Thermo Diagnostics] step={step}, retry={retry_count}, "
                f"ΔE={delta_energy:.3e}, allowed={allowed_increase:.3e}, "
                f"R={thermo_residual:.3e}, budget={thermo_budget:.3e}, "
                f"D={dissipation:.3e}, Wext={external_work:.3e}"
            )
            print_flush(
                f"      chart churn: switched_nodes={switched_after_accept}, max|q|={r_max_attempt:.4f}"
            )
            print_flush(
                f"      chart consistency: mismatch_pairs={mismatch_pairs}/{owned_refs} ({pair_rate:.2f}%), "
                f"mismatch_nodes={mismatch_nodes}/{owned_nodes} ({node_rate:.2f}%)"
            )
            comp_line = " | ".join(f"{k}={v:.4e}" for k, v in zip(labels, curr))
            print_flush(f"      energies(curr): {comp_line}")
            if prev is not None and len(prev) == len(curr):
                dcomp = curr - prev
                dcomp_line = " | ".join(f"Δ{k}={v:.4e}" for k, v in zip(labels, dcomp))
                print_flush(f"      energies(delta from accepted): {dcomp_line}")
                print_flush(f"      previous accepted total={accepted_prev_energy:.4e}")
        # IMPORTANT: closure metrics use MPI collectives, so compute on all ranks.
        closure = self._atlas_closure_metrics()
        self._atlas_thermo_metrics = closure
        if self.domain.comm.rank != 0:
            return

        d_q = float(getattr(self, "_thermo_last_diss_q_form", 0.0))
        d_m = float(getattr(self, "_thermo_last_diss_m_cart", 0.0))
        print_flush(
            "      atlas thermo: "
            f"D_q_form={d_q:.3e}, D_m_cart={d_m:.3e}, "
            f"ΔD={abs(d_q - d_m):.3e}"
        )
        print_flush(
            "      atlas budget/switch: "
            f"op_budget={float(getattr(self, '_thermo_last_op_budget', 0.0)):.3e}, "
            f"ema_floor={float(getattr(self, '_thermo_last_ema_floor', 0.0)):.3e}, "
            f"ramp_budget={float(getattr(self, '_thermo_last_ramp_budget', 0.0)):.3e}, "
            f"W_switch={float(getattr(self, '_thermo_last_switch_work', 0.0)):.3e}"
        )
        print_flush(
            "      atlas closure: "
            f"overlap_rms={closure.get('closure_overlap_rms', 0.0):.3e}, "
            f"exclusive_rms={closure.get('closure_exclusive_rms', 0.0):.3e}, "
            f"sigma2_min_active={closure.get('sigma2_min_active', 0.0):.3e}, "
            f"overlap_frac={closure.get('overlap_cell_fraction', 0.0):.3f}"
        )
        mask_stats = getattr(self, "_atlas_mask_stats", None)
        if mask_stats is not None and self._debug_logs:
            print_flush(
                "      atlas masks: "
                f"chiA={mask_stats.get('chiA_cells_global', 0)}, "
                f"chiB={mask_stats.get('chiB_cells_global', 0)}, "
                f"chiAB={mask_stats.get('chiAB_cells_global', 0)}, "
                f"switched_flags={mask_stats.get('switched_flags_global', 0)}, "
                f"avg_wA={mask_stats.get('avg_wA_global', 0.0):.3f}, "
                f"avg_wB={mask_stats.get('avg_wB_global', 0.0):.3f}"
            )


    def _commit_thermo_backtrack(self, att: _AcceptAttemptState, bt_info: dict) -> None:
        att.accepted_backtracking_lambda = float(bt_info["lambda"])
        att.residual_norm = float(bt_info["residual_norm"])
        att.energy_components = bt_info["energy_components"]
        att.energy_components_physical = bt_info["energy_components_physical"]
        att.current_energy_total = float(bt_info["energy_total"])
        att.current_energy_total_physical = float(bt_info["energy_total_physical"])
        att.accepted_thermo_residual = float(bt_info["thermo_residual"])
        att.accepted_delta_energy_phys = bt_info["delta_energy_phys"]
        self.state_monitor["last_thermo_backtracking_lambda"] = att.accepted_backtracking_lambda
        att.r_max_attempt = self._enforce_atlas_radius_guard(silent=True)

    def _apply_thermodynamic_gate(
        self,
        step: int,
        preamble: _StepPreambleState,
        att: _AcceptAttemptState,
        retry: _RetryLoopState,
    ) -> None:
        """Thermodynamic dissipation gate (E) with optional E1 backtracking.

        Skipped by the caller when ``preamble.thermo_guard_active`` is false (ramps,
        continuation reconcile). Evaluates :meth:`_evaluate_thermodynamic_residual` and
        may call :meth:`_thermo_backtracking_search` / :meth:`_commit_thermo_backtrack`.
        Physical-energy guard uses the same E1 path on failure.
        """
        current_energy_gate = (
            float(att.current_energy_total)
            if self.thermo_use_split_energy
            else float(att.current_energy_total_physical)
        )
        prev_energy = self.state_monitor.get("thermo_last_energy", current_energy_gate)
        prev_components = self.state_monitor.get("thermo_last_components", None)
        prev_components_physical = self.state_monitor.get("thermo_last_components_physical", None)
        prev_energy_chart = (
            float(prev_components[-1])
            if prev_components is not None and len(prev_components) > 0
            else float(att.current_energy_total)
        )
        prev_demag_phys = (
            float(prev_components_physical[3])
            if prev_components_physical is not None and len(prev_components_physical) > 3
            else None
        )
        thermo_residual, thermo_budget, dissipation, external_work = self._evaluate_thermodynamic_residual(
            prev_energy=prev_energy,
            curr_energy=current_energy_gate,
            chart_energy_delta=float(att.current_energy_total - prev_energy_chart),
            prev_qA=self._qA_backup,
            dt_val=float(self.dt.value),
            nonlinear_residual=att.residual_norm,
            zeeman_prev=preamble.zeeman_prev,
            demag_ramp_prev=preamble.demag_ramp_prev,
            prev_demag_energy_physical=prev_demag_phys,
        )
        delta_energy = current_energy_gate - prev_energy
        allowed_increase = max(0.0, -external_work) + thermo_budget
        if delta_energy > allowed_increase:
            thermo_msg = (
                "Thermodynamic check failed: "
                f"ΔE={delta_energy:.3e} > allowed={allowed_increase:.3e} "
                f"(R={thermo_residual:.3e}, D={dissipation:.3e}, Wext={external_work:.3e})"
            )
            bt_info = None
            if self.thermo_backtracking_enable:
                bt_info = self._thermo_backtracking_search(
                    step=step,
                    qA_trial=att.qA_trial,
                    qB_trial=att.qB_trial,
                    prev_energy=float(prev_energy),
                    prev_components=prev_components,
                    prev_components_physical=prev_components_physical,
                    zeeman_prev=preamble.zeeman_prev,
                    demag_ramp_prev=preamble.demag_ramp_prev,
                    effective_gate=att.effective_gate,
                )
            if bt_info is None:
                raise RuntimeError(thermo_msg)
            self._commit_thermo_backtrack(att, bt_info)

        if self.thermo_enable_physical_guard:
            prev_energy_phys = float(
                self.state_monitor.get(
                    "thermo_last_energy_physical", att.current_energy_total_physical
                )
            )
            delta_energy_phys = float(att.current_energy_total_physical) - prev_energy_phys
            phys_scale = max(
                1.0, abs(prev_energy_phys), abs(float(att.current_energy_total_physical))
            )
            allowed_phys_base = (
                allowed_increase + float(self.thermo_physical_guard_rel_floor) * phys_scale
            )
            phys_delta_ema = self.state_monitor.get("thermo_phys_delta_ema", None)
            adaptive_phys_floor = 0.0
            if (
                phys_delta_ema is not None
                and step >= int(self.thermo_physical_guard_warmup_steps)
                and np.isfinite(float(phys_delta_ema))
            ):
                adaptive_phys_floor = float(self.thermo_physical_guard_ema_factor) * float(
                    phys_delta_ema
                )
            adaptive_phys_floor = max(float(self.thermo_physical_guard_abs_floor), adaptive_phys_floor)
            allowed_phys = allowed_phys_base + adaptive_phys_floor
            if delta_energy_phys > allowed_phys:
                phys_msg = (
                    "Physical thermo guard failed: "
                    f"ΔE_phys={delta_energy_phys:.3e} > allowed={allowed_phys:.3e} "
                    f"(base={allowed_phys_base:.3e}, adaptive={adaptive_phys_floor:.3e})"
                )
                bt_info = None
                if self.thermo_backtracking_enable:
                    bt_info = self._thermo_backtracking_search(
                        step=step,
                        qA_trial=att.qA_trial,
                        qB_trial=att.qB_trial,
                        prev_energy=float(prev_energy),
                        prev_components=prev_components,
                        prev_components_physical=prev_components_physical,
                        zeeman_prev=preamble.zeeman_prev,
                        demag_ramp_prev=preamble.demag_ramp_prev,
                        effective_gate=att.effective_gate,
                    )
                if bt_info is None:
                    raise RuntimeError(phys_msg)
                self._commit_thermo_backtrack(att, bt_info)
            else:
                att.accepted_delta_energy_phys = float(delta_energy_phys)

        should_log_diag = (
            self.enable_thermo_diagnostics
            and (delta_energy > allowed_increase or self.thermo_diag_log_accepted)
        )
        if should_log_diag:
            self._log_thermo_diagnostics(
                step=step,
                retry_count=retry.retry_count,
                accepted_prev_energy=prev_energy,
                accepted_prev_components=prev_components,
                current_components=att.energy_components,
                thermo_residual=thermo_residual,
                thermo_budget=thermo_budget,
                dissipation=dissipation,
                external_work=external_work,
                delta_energy=delta_energy,
                allowed_increase=allowed_increase,
                switched_after_accept=0,
                r_max_attempt=att.r_max_attempt,
            )
        att.accepted_thermo_residual = thermo_residual

    def _apply_closure_gate(
        self, preamble: _StepPreambleState, att: _AcceptAttemptState
    ) -> None:
        """Atlas closure and representation-jump check (gate F).

        Applies inactive-chart reconciliation via :meth:`_prepare_state_for_acceptance`
        with ``apply_closure=True``. Raises on excessive :math:`|\\Delta E_{\\mathrm{closure}}|`
        unless continuation reconcile is active (soft warn only).
        """
        energy_gate_pre_closure = (
            float(att.current_energy_total)
            if self.thermo_use_split_energy
            else float(att.current_energy_total_physical)
        )
        self._prepare_state_for_acceptance(apply_closure=True)
        att.energy_components = self._compute_energy_quiet()
        att.current_energy_total = float(att.energy_components[-1])
        att.energy_components_physical = self._compute_energy_physical_cartesian()
        att.current_energy_total_physical = float(att.energy_components_physical[-1])
        energy_gate_post_closure = (
            float(att.current_energy_total)
            if self.thermo_use_split_energy
            else float(att.current_energy_total_physical)
        )
        closure_jump = float(energy_gate_post_closure - energy_gate_pre_closure)
        closure_jump_abs = abs(closure_jump)
        closure_scale = max(
            1.0,
            abs(energy_gate_pre_closure),
            abs(energy_gate_post_closure),
        )
        closure_jump_tol = max(
            float(self._atlas_closure_jump_abs_floor),
            float(self._atlas_closure_jump_rel_tol) * closure_scale,
        )
        att.accepted_closure_jump_energy = closure_jump
        self.state_monitor["last_closure_jump_energy"] = float(closure_jump)
        self.state_monitor["last_closure_jump_warn"] = 0
        if (
            closure_jump_abs
            > closure_jump_tol * float(self._atlas_closure_jump_reject_factor)
            and not preamble.transition_reconcile_active
        ):
            raise RuntimeError(
                "Atlas closure representation jump failed: "
                f"|ΔE_closure|={closure_jump_abs:.3e} > tol={closure_jump_tol:.3e}"
            )
        if closure_jump_abs > closure_jump_tol and not preamble.transition_reconcile_active:
            self.state_monitor["last_closure_jump_warn"] = 1
            if self.domain.comm.rank == 0 and self._debug_logs:
                print_flush(
                    "    [Closure Jump Soft] "
                    f"|ΔE_closure|={closure_jump_abs:.3e} "
                    f"near tol={closure_jump_tol:.3e}"
                )

    def _apply_residual_soft_accept_policy(self, att: _AcceptAttemptState) -> None:
        """Post-acceptance dt caps after residual soft violation (D3)."""
        if not att.residual_soft_violation:
            return
        soft_cap_factor = 0.5 if att.residual_soft_ratio > 2.0 else 0.75
        soft_cap = max(float(self.dt_min), float(self.dt.value) * soft_cap_factor)
        self.state_monitor["nonthermo_dt_cap"] = min(
            float(self.state_monitor.get("nonthermo_dt_cap", self.dt_max)),
            float(soft_cap),
        )
        self.state_monitor["nonthermo_dt_cap_cooldown_steps"] = max(
            int(self.state_monitor.get("nonthermo_dt_cap_cooldown_steps", 0)),
            int(self._nonthermo_cap_cooldown_default),
        )
        self.state_monitor["be_recovery_steps_remaining"] = max(
            int(self.state_monitor.get("be_recovery_steps_remaining", 0)),
            1,
        )
        self.state_monitor["nonthermo_clean_accepted_steps"] = 0

    # --- Rejection handler and post-accept bookkeeping ---

    def _handle_step_rejection(
        self, step: int, exc: RuntimeError, retry: _RetryLoopState
    ) -> None:
        """Classify acceptance failure, restore backup state, adjust ``dt`` / retry policy.

        Parses the exception message to distinguish thermodynamic, outer-split, SNES-reason,
        and generic nonlinear failures. May shrink ``dt``, escalate ``retry.outer_iter_cap``,
        switch SNES globalization mode, or fall back BDF2→BDF1. Re-raises when retry limits
        or ``dt_min`` are exhausted.
        """
        # --- Rejection handler: classify failure, rollback, maybe cut dt ---
        err_msg = str(exc)
        is_outer_split_rejection = "Outer split not converged" in err_msg
        is_thermo_rejection = (
            "Thermodynamic check failed" in err_msg
            or "Physical thermo guard failed" in err_msg
        )
        is_disallowed_reason_rejection = (
            "SNES converged for disallowed reason with large residual" in err_msg
        )
        retry.retry_count += 1
        if is_thermo_rejection:
            retry.thermo_retry_count += 1
        elif is_outer_split_rejection:
            retry.outer_retry_count += 1
            self.state_monitor["last_nonlinear_retry_mode"] = None
        else:
            retry.nonthermo_retry_count += 1
            # Reason-aware globalization policy:
            # -6 (line search): first retry keeps line-search with
            # reduced dt; only escalate to trust-region if repeated.
            # -11 (trust-region delta): switch back to line-search.
            # Unknown: alternate to avoid mode lock-in.
            selected_mode = None
            if ("reason=-6" in err_msg) or ("snes_reason=-6" in err_msg):
                selected_mode = "default" if retry.nonthermo_retry_count == 1 else "retry"
                self._set_snes_globalization_mode(selected_mode)
            elif ("reason=-11" in err_msg) or ("snes_reason=-11" in err_msg):
                selected_mode = "default"
                self._set_snes_globalization_mode(selected_mode)
            else:
                mode = "retry" if (retry.nonthermo_retry_count % 2 == 1) else "default"
                selected_mode = mode
                self._set_snes_globalization_mode(selected_mode)
            self.state_monitor["last_nonlinear_retry_mode"] = selected_mode
            self.state_monitor["nonlinear_retry_cooldown_steps"] = int(
                self._nonlinear_retry_growth_cooldown_steps
            )

        dt_old = float(self.dt.value)
        if not is_thermo_rejection:
            self.state_monitor["nonthermo_clean_accepted_steps"] = 0
        # For disallowed SNES converged reasons (typically reason=4),
        # try one same-dt recovery (with mode switch and optional
        # extra outer stage) before cutting dt.
        if (
            is_disallowed_reason_rejection
            and retry.disallowed_reason_same_dt_recoveries < 1
        ):
            retry.disallowed_reason_same_dt_recoveries += 1
            old_cap = int(retry.outer_iter_cap)
            if retry.outer_iter_cap < int(self.outer_split_max_iter_hard):
                retry.outer_iter_cap = min(
                    int(self.outer_split_max_iter_hard),
                    int(retry.outer_iter_cap) + max(1, int(self.outer_split_iter_escalation)),
                )
            prev_mode = self.state_monitor.get("last_nonlinear_retry_mode", "default")
            next_mode = "retry" if prev_mode != "retry" else "default"
            self._set_snes_globalization_mode(next_mode)
            self.state_monitor["last_nonlinear_retry_mode"] = next_mode
            self.state_monitor["nonlinear_retry_cooldown_steps"] = int(
                self._nonlinear_retry_growth_cooldown_steps
            )
            self._rollback_to_step_backup()
            if self.domain.comm.rank == 0:
                print_flush(
                    f"\n    [Nonlinear Recovery] {exc}. Retry {retry.retry_count} "
                    f"(outer={retry.outer_retry_count}, thermo={retry.thermo_retry_count}, other={retry.nonthermo_retry_count}): "
                    f"dt unchanged {dt_old:.3e}, mode={next_mode}, "
                    f"retry.outer_iter_cap {old_cap} -> {int(retry.outer_iter_cap)}"
                )
            return
        # Outer fixed-point nonconvergence is not an SNES failure.
        # First escalate outer iterations before shrinking dt.
        if is_outer_split_rejection and self.enforce_outer_split_acceptance:
            if (
                retry.outer_retry_count <= max(1, int(self.outer_split_retry_before_dt_cut))
                and retry.outer_iter_cap < int(self.outer_split_max_iter_hard)
            ):
                old_cap = int(retry.outer_iter_cap)
                retry.outer_iter_cap = min(
                    int(self.outer_split_max_iter_hard),
                    int(retry.outer_iter_cap) + max(1, int(self.outer_split_iter_escalation)),
                )
                self._rollback_to_step_backup()
                if self.domain.comm.rank == 0:
                    print_flush(
                        f"\n    [Outer Split Retry] {exc}. Retry {retry.retry_count} "
                        f"(outer={retry.outer_retry_count}, thermo={retry.thermo_retry_count}, other={retry.nonthermo_retry_count}): "
                        f"increasing retry.outer_iter_cap {old_cap} -> {retry.outer_iter_cap} (dt unchanged {dt_old:.3e})"
                    )
                return
        # Thermo rejection uses overshoot-aware shrink; non-thermo
        # failures keep conservative halving for robustness.
        shrink_factor = self.min_shrink if is_thermo_rejection else 0.5
        if is_thermo_rejection:
            # Parse "ΔE=... > allowed=..." from the rejection message.
            m = re.search(
                r"ΔE=([0-9eE+\-\.]+)\s*>\s*allowed=([0-9eE+\-\.]+)",
                err_msg,
            )
            if m:
                delta_e = float(m.group(1))
                allowed_e = max(abs(float(m.group(2))), 1e-30)
                ratio = max(delta_e / allowed_e, 1.0)
                # If overshoot is large, shrink more aggressively.
                # If overshoot is mild, avoid over-shrinking.
                shrink_factor = min(self.min_shrink, 0.90 / ratio)
                self.state_monitor["last_thermo_overshoot_ratio"] = float(ratio)
        shrink_factor = float(min(max(shrink_factor, 0.05), 0.95))
        dt_new = max(dt_old * shrink_factor, self.dt_min)
        same_dt_retry = False

        if is_thermo_rejection:
            # Keep a temporary cap on future proposals to avoid
            # immediate regrowth into the same thermo rejection zone.
            thermo_cap = min(self.dt_max, dt_new * self._thermo_cap_relax_factor)
            self.state_monitor["thermo_dt_cap"] = float(thermo_cap)
            self.state_monitor["thermo_dt_cap_cooldown_steps"] = int(
                self._thermo_cap_cooldown_default
            )
            self.state_monitor["be_recovery_steps_remaining"] = max(
                int(self.state_monitor.get("be_recovery_steps_remaining", 0)),
                self._be_recovery_steps_on_thermo,
            )
        elif not is_outer_split_rejection:
            # Non-thermo rejections usually mark a nonlinear
            # globalization window. Keep dt regrowth bounded
            # for a short horizon to reduce reject oscillations.
            self.state_monitor["nonthermo_dt_cap"] = float(dt_new)
            self.state_monitor["nonthermo_dt_cap_cooldown_steps"] = int(
                self._nonthermo_cap_cooldown_default
            )

        if dt_new >= dt_old * (1.0 - 1e-14):
            at_dt_min = dt_old <= (self.dt_min * (1.0 + 1e-14))
            if at_dt_min:
                # At dt_min we may still recover by switching SNES mode
                # or BDF order; allow bounded same-dt retries.
                same_dt_retry = True
                dt_new = dt_old
            else:
                raise RuntimeError(
                    f"Step {step}: rejection at dt_min={self.dt_min:.3e}; cannot reduce further. "
                    f"Last error: {err_msg}"
                ) from exc

        if is_thermo_rejection and retry.thermo_retry_count > self.max_retry_thermo:
            raise RuntimeError(
                f"Step {step}: thermodynamic rejection persisted after "
                f"{retry.thermo_retry_count} retries. Last error: {err_msg}"
            ) from exc
        if is_outer_split_rejection and retry.outer_retry_count > self.max_retry:
            raise RuntimeError(
                f"Step {step}: outer split did not converge after "
                f"{retry.outer_retry_count} retries. Last error: {err_msg}"
            ) from exc
        if (not is_thermo_rejection) and (not is_outer_split_rejection) and retry.nonthermo_retry_count > self.max_retry:
            raise RuntimeError(
                f"Step {step}: nonlinear/linear solve failed after "
                f"{retry.nonthermo_retry_count} retries. Last error: {err_msg}"
            ) from exc

        if self.domain.comm.rank == 0:
            retry_mode = self.state_monitor.get("last_nonlinear_retry_mode", None)
            mode_txt = f", mode={retry_mode}" if retry_mode is not None else ""
            dt_tag = " (dt_min retry)" if same_dt_retry else ""
            print_flush(
                f"\n    [Step Rejection] {exc}. Retry {retry.retry_count} "
                f"(outer={retry.outer_retry_count}, thermo={retry.thermo_retry_count}, other={retry.nonthermo_retry_count}): "
                f"dt {dt_old:.3e} -> {dt_new:.3e}{mode_txt}{dt_tag}"
            )
        self.dt.value = dt_new
        self._rollback_to_step_backup()
        # Keep BDF2 for a first thermo rejection to reduce
        # unnecessary order toggling. Fall back on repeated thermo
        # rejections or any non-thermo solver failure.
        fallback_bdf = (not is_thermo_rejection) or (retry.thermo_retry_count >= 2)
        if self._bdf_active_order == 2 and fallback_bdf:
            self._bdf_active_order = 1
            self._bdf_steps_since_restart = 0
            if self.domain.comm.rank == 0 and self._debug_logs:
                print_flush("    [BDF] Falling back to order 1 after step rejection")


    def _finalize_accepted_step(
        self,
        step: int,
        preamble: _StepPreambleState,
        accepted_step: _AcceptedStepData,
    ) -> bool:
        """Post-accept bookkeeping after a successful step.

        Advances ``t``, updates BDF order promotion, residual/thermo EMAs, thermodynamic
        history, writes time-series output and optional checkpoints, and calls
        :meth:`_update_timestep_cfl`. Returns ``True`` when physical steady-state
        convergence requests stopping the outer loop.
        """
        stop_after_accept = False
        self._bdf_steps_since_restart += 1
        if (
            self.time_order >= 2
            and self._bdf_active_order == 1
            and self._bdf_steps_since_restart >= 2
        ):
            self._bdf_active_order = 2
            if self.domain.comm.rank == 0 and self._debug_logs:
                print_flush(f"\n    [BDF] Switching to order 2 at step {step}")

        self._dt_prev_value = float(self.dt.value)

        if self.domain.comm.rank == 0:
            split_iters = int(self.state_monitor.get("last_outer_split_iters", 0))
            and_used = int(self.state_monitor.get("last_outer_split_anderson_used", 0))
            q_def = float(self.state_monitor.get("last_outer_split_q_defect", 0.0))
            msg = (
                f"\n    Step {step} accepted: residual={accepted_step.residual_norm:.3e}, "
                f"dt={float(self.dt.value):.3e}, SNES={accepted_step.snes_iters}, KSP={accepted_step.ksp_iters}, "
                f"outer={split_iters}, anderson={and_used}, "
                f"q_def={q_def:.3e}"
            )
            if "demag" in self.energy_terms:
                phi_def = float(self.state_monitor.get("last_outer_split_phi_defect", 0.0))
                msg += f", phi_def={phi_def:.3e}"
            if accepted_step.backtracking_lambda < 0.999:
                msg += f", bt_lambda={accepted_step.backtracking_lambda:.3e}"
            print_flush(msg)

        # Atlas closure has already been applied inside the accepted attempt.
        energy_components_post = accepted_step.energy_components
        energy_total_post = accepted_step.energy_total

        # Update closure monitor/telemetry on the accepted state.
        self._apply_post_accept_atlas_closure(update_monitor=True, apply_closure=False)
        closure_delta_e = float(accepted_step.closure_jump_energy)
        if self.domain.comm.rank == 0 and self.enable_thermo_diagnostics:
            print_flush(
                f"    [Atlas Closure] ΔE_post_closure={closure_delta_e:.3e}"
            )

        if self.prev_timestep_residual is not None:
            conv = self.convergence_settings
            rel_change = abs(accepted_step.residual_norm - self.prev_timestep_residual) / max(
                self.prev_timestep_residual, 1e-15
            )
            if rel_change < conv.get("stagnation_tol", 1e-4):
                self.timestep_stagnation_counter += 1
            else:
                self.timestep_stagnation_counter = 0
            if self.timestep_stagnation_counter >= conv.get("stagnation_steps", 5):
                if self.domain.comm.rank == 0:
                    last_phys = self.state_monitor.get("last_physical_norm")
                    last_dE = self.state_monitor.get("last_energy_change")
                    phys_label = self.state_monitor.get("last_physical_field_label", "state")
                    phys_txt = (
                        f"last ||{phys_label}||={last_phys:.2e}"
                        if last_phys is not None
                        else "last ||physical_rate||=n/a"
                    )
                    dE_txt = (
                        f"last |dE/dt|={last_dE:.2e}"
                        if last_dE is not None
                        else "last |dE/dt|=n/a"
                    )
                    print_flush(
                        f"\n  *** STAGNATION warning "
                        f"(residual rel. change < {conv.get('stagnation_tol', 1e-4):.1e} "
                        f"for {conv.get('stagnation_steps', 5)} consecutive steps); "
                        f"dt={float(self.dt.value):.3e}; {phys_txt}; {dE_txt}; "
                        f"continuing until physical convergence."
                    )
                # Residual plateau alone is not a robust physical stop condition.
                self.timestep_stagnation_counter = 0
        self.prev_timestep_residual = accepted_step.residual_norm
        prev_ema = self.state_monitor.get("accepted_residual_ema", None)
        beta = float(min(max(self.nonlinear_residual_ema_beta, 0.0), 0.999))
        self.state_monitor["accepted_residual_ema"] = (
            float(accepted_step.residual_norm)
            if prev_ema is None
            else (beta * float(prev_ema) + (1.0 - beta) * float(accepted_step.residual_norm))
        )
        prev_ema_slow = self.state_monitor.get("accepted_residual_ema_slow", None)
        beta_slow = float(min(max(self._nonlinear_residual_slow_ema_beta, 0.0), 0.9999))
        self.state_monitor["accepted_residual_ema_slow"] = (
            float(accepted_step.residual_norm)
            if prev_ema_slow is None
            else (beta_slow * float(prev_ema_slow) + (1.0 - beta_slow) * float(accepted_step.residual_norm))
        )
        self.state_monitor["nonthermo_clean_accepted_steps"] = (
            int(self.state_monitor.get("nonthermo_clean_accepted_steps", 0)) + 1
        )
        if accepted_step.delta_energy_phys is not None:
            prev_phys_ema = self.state_monitor.get("thermo_phys_delta_ema", None)
            phys_abs = abs(float(accepted_step.delta_energy_phys))
            self.state_monitor["thermo_phys_delta_ema"] = (
                phys_abs if prev_phys_ema is None else (0.8 * float(prev_phys_ema) + 0.2 * phys_abs)
            )
        temporal_err_norm = self._compute_temporal_error_norm()
        self.state_monitor["last_temporal_error_norm"] = (
            float(temporal_err_norm) if temporal_err_norm is not None else None
        )

        self.t += float(self.dt.value)
        self.check_q_magnitude()

        # Physical steady-state detection should use physical cartesian
        # energy to avoid atlas-closure representation jumps contaminating
        # |dE/dt| near equilibrium.
        if self._check_physical_convergence(
            self._qA_backup,
            self._qB_backup,
            accepted_step.energy_total_physical,
        ):
            if self.domain.comm.rank == 0:
                print_flush("\n Physical equilibrium reached.")
            stop_after_accept = True

        if preamble.thermo_guard_active:
            prev_last = self.state_monitor.get(
                "thermo_last_energy",
                (accepted_step.energy_total if self.thermo_use_split_energy else accepted_step.energy_total_physical),
            )
            self.state_monitor["thermo_energy_nm2"] = self.state_monitor.get("thermo_energy_nm1")
            self.state_monitor["thermo_energy_nm1"] = prev_last
            self.state_monitor["thermo_dt_nm1"] = float(self.dt.value)
            if accepted_step.thermo_residual is not None:
                prev_ema = self.state_monitor.get("thermo_defect_ema")
                abs_r = abs(float(accepted_step.thermo_residual))
                self.state_monitor["thermo_defect_ema"] = (
                    abs_r if prev_ema is None else (0.8 * float(prev_ema) + 0.2 * abs_r)
                )
        else:
            self.state_monitor["thermo_energy_nm2"] = None
            self.state_monitor["thermo_energy_nm1"] = None
            self.state_monitor["thermo_dt_nm1"] = None
            self.state_monitor["thermo_defect_ema"] = None
            self.state_monitor["thermo_phys_delta_ema"] = None

        energy_post_physical = self._compute_energy_physical_cartesian()
        self.state_monitor["thermo_last_energy"] = (
            float(energy_total_post) if self.thermo_use_split_energy else float(energy_post_physical[-1])
        )
        self.state_monitor["thermo_last_energy_physical"] = float(energy_post_physical[-1])
        self.state_monitor["thermo_last_components"] = np.array(
            energy_components_post, dtype=np.float64
        )
        self.state_monitor["thermo_last_components_physical"] = np.array(
            energy_post_physical, dtype=np.float64
        )
        if preamble.transition_reconcile_active:
            self.state_monitor["continuation_reconcile_steps_remaining"] = max(
                0,
                int(self.state_monitor.get("continuation_reconcile_steps_remaining", 0)) - 1,
            )
        if self._h_zee_prev_accepted is not None and hasattr(self.h_zee, "x"):
            self._h_zee_prev_accepted = self.h_zee.x.array.copy()
        if self._demag_ramp_prev_accepted is not None:
            self._demag_ramp_prev_accepted = float(self.demag_ramp_constant.value)

        self._backup_solution()

        if hasattr(self, "energies"):
            energy_update = np.hstack([self.t, float(self.dt.value), energy_components_post])
            self.energies = np.vstack([self.energies, energy_update])
        else:
            self.energies = np.hstack([self.t, float(self.dt.value), energy_components_post])
            self.energies = self.energies.reshape(1, -1)

        for target in self._time_write_targets:
            self.write_output(target)
        if "for_vacuum_dir" in self.path_dirs and "demag" in self.energy_terms:
            self.write_phi1_boundary(
                f"{self.path_dirs['for_vacuum_dir']}/phi1_boundary_timestep_{step}.bp"
            )

        self._update_timestep_cfl()

        checkpoint_dir = self.path_dirs["checkpoint_dir"]
        if step % self.checkpoint_every == 0:
            dt_this_step = self.energies[-1, 1]
            self.energies[-1, 1] = float(self.dt.value)
            self.write_energies(f"{checkpoint_dir}/checkpoint_energies.txt")
            if "results_dir" in self.path_dirs:
                self.write_energies(f"{self.path_dirs['results_dir']}/results_energies.txt")
            self.energies[-1, 1] = dt_this_step
            self.write_checkpoint(f"{checkpoint_dir}/checkpoint_timestep_{step}.bp")
            for xdmf_target in getattr(self, "_xdmf_checkpoint_targets", []):
                self.write_xdmf_checkpoint(
                    xdmf_target,
                    f"{checkpoint_dir}/checkpoint_{xdmf_target}_step_{step}.xdmf",
                )
        return stop_after_accept

    # ==================================================================
    # Public driver: solve_problem
    # ==================================================================

    def solve_problem(self):
        r"""
        Advance the simulation from ``step_start`` to ``step_end``.

        **High-level loop** (each integer ``step``):

        1. **Preamble** — update demag/Zeeman ramps; optional continuation transition
           (shrink ``dt``, force BDF1, suspend thermo briefly); ``solve_magnetostatics``.
        2. **Retry loop** — repeat until the step is accepted or retry limits fire.
        3. **Post-accept** — advance ``t`` by full ``dt``, update BDF history, CFL controller,
           I/O, checkpoints; test physical steady-state stop.

        **One acceptance attempt** is :meth:`_run_accept_attempt`, which runs gates
        in order: (A) :meth:`_run_nonlinear_solve`, (B) :meth:`_validate_algebraic_solve`,
        (C) :meth:`_refresh_attempt_energies`, (D) :meth:`_apply_residual_gate`,
        (E) :meth:`_apply_thermodynamic_gate`, (F) :meth:`_apply_closure_gate`.
        See the table in :meth:`_run_accept_attempt` for substages and side helpers.

        **Backtracking** (both kinds) damps state along
        :math:`q^n + \lambda(q^{\mathrm{trial}}-q^n)` at **full** ``dt`` without re-solving;
        see :meth:`_set_state_from_backup_trial_blend`. **Residual** backtracking failure
        restores the SNES trial and falls through to merit/soft gates (D2/D3); only
        **thermodynamic** backtracking failure raises ``RuntimeError``, rolls back via
        :meth:`_rollback_to_step_backup`, and may cut ``dt`` (unless same-``dt`` recovery applies).

        **Rejected steps** shrink ``dt`` (thermo-aware or halving), may escalate outer-split
        cap or switch SNES globalization, and may fall back BDF2→BDF1.

        See Also
        --------
        :doc:`theory_algorithms` — full algorithm reference.
        :meth:`solve_weak_form` — nonlinear solve (subclass).
        :meth:`_update_timestep_cfl` — ``dt`` selection after accept.

        Notes
        -----
        All ranks participate in atlas and gate MPI reductions.
        """
        # One-time setup: backup u^n, open optional XDMF writers, seed thermo/residual
        # monitors and continuation state used by the gates below.
        self._init_solve_problem_runtime()

        # Outer driver: each integer ``step`` seeks one accepted state at t^{n+1}.
        for step in range(self.step_start, self.step_end):
            stop_after_accept = False
            self.step_now = step

            # --- Preamble (once per step, before the inner retry loop) ---
            # BDF recovery window, console header; demag/Zeeman ramp updates;
            # optional continuation transition (shrink dt*, force BDF1, suspend
            # thermodynamic gate briefly); subclass magnetostatics solve.
            self._open_timestep(step)
            preamble = self._run_timestep_preamble(step)

            # --- Inner retry loop at fixed dt* ---
            # Each attempt begins from the step backup (_qA_backup/_qB_backup).
            # On rejection, _handle_step_rejection restores that backup and may
            # cut dt*, escalate outer-split iterations, or switch SNES mode.
            accepted = False
            accepted_step: _AcceptedStepData | None = None
            retry = self._RetryLoopState(outer_iter_cap=self._initial_outer_iter_cap())
            # Carry smallest residual/thermo backtracking lambda across retries at
            # the same dt* so a prior damped state is not forgotten.
            accepted_backtracking_lambda = 1.0
            self.state_monitor["last_thermo_backtracking_lambda"] = 1.0
            while not accepted:
                try:
                    att = self._AcceptAttemptState()
                    att.accepted_backtracking_lambda = accepted_backtracking_lambda

                    # --- Acceptance attempt: gates (A)–(F) via _run_accept_attempt ---
                    # (A) Nonlinear BDF step (+ outer split in demag) → trial q.
                    # (B) SNES/KSP convergence and outer-split defect check.
                    # (C) Atlas radius guard + chart/physical energy snapshot.
                    # (D) Residual gate: D1 line search, D2 merit, or D3 soft accept.
                    # (E) Thermodynamic dissipation (skipped during ramps/reconcile);
                    #     E1 backtracking on failure.
                    # (F) Inactive-chart closure and representation-jump check.
                    # D3 soft accept and some closure warnings do not raise; they
                    # only tighten future dt* caps after the step is accepted.
                    accepted_step = self._run_accept_attempt(step, preamble, retry, att)
                    accepted_backtracking_lambda = att.accepted_backtracking_lambda
                    accepted = True
                except RuntimeError as e:
                    # Hard rejection from (B), (E), (E1), or (F): roll back to
                    # backup, adjust retry policy, and either continue at smaller
                    # dt* or re-raise when retry/dt_min limits are hit.
                    self._handle_step_rejection(step, e, retry)
                    continue

            # --- Post-accept (once per accepted step) ---
            # Advance t by full dt*, promote BDF order when eligible, update
            # residual/thermo EMAs and history, write fields/checkpoints, and
            # select the next dt* proposal (_update_timestep_cfl). May return
            # True to stop the simulation on physical steady-state convergence.
            stop_after_accept = self._finalize_accepted_step(step, preamble, accepted_step)
            if stop_after_accept:
                break

        # Flush energy table if the run ended between checkpoint intervals.
        if "results_dir" in self.path_dirs and hasattr(self, "energies"):
            self.write_energies(f"{self.path_dirs['results_dir']}/results_energies.txt")

    # ==================================================================
    # XDMF time-series writers
    # ==================================================================

    def _xdmf_time_enabled(self, target: str) -> bool:
        return bool(self._xdmf_time_all_targets or (target in self._xdmf_time_targets))

    def _open_xdmf_time_writer(self, target: str, filename: str) -> None:
        if not self._xdmf_time_enabled(target):
            return
        if target in self._xdmf_time_writers:
            self._xdmf_time_writers[target].close()
            del self._xdmf_time_writers[target]
        xw = io.XDMFFile(self.domain.comm, filename + ".xdmf", "w")
        xw.write_mesh(self.domain)
        self._xdmf_time_writers[target] = xw

    def _write_xdmf_time_output(self, target: str, f_out: fem.Function) -> None:
        xw = self._xdmf_time_writers.get(target)
        if xw is not None:
            xw.write_function(f_out, self.t)

    def _close_xdmf_time_writer(self, target: str) -> None:
        xw = self._xdmf_time_writers.pop(target, None)
        if xw is not None:
            xw.close()

    def _open_xdmf_time_only_outputs(self) -> None:
        """Open XDMF-only writers for targets not present in BP write_target."""
        results_dir = self.path_dirs.get("results_dir", None)
        if not results_dir:
            return
        for target in self._time_write_targets:
            if target in self.write_target:
                continue
            if not self._xdmf_time_enabled(target):
                continue
            filename = os.path.join(results_dir, f"results_evol_{target}")
            self._open_xdmf_time_writer(target, filename)

    # ==================================================================
    # Cartesian diagnostics and magnetization interpolation
    # ==================================================================

    def _blend_m_from_qA_qB(self, qA_arr, qB_arr, wa_node=None, wb_node=None):
        """
        Reconstruct a single Cartesian direction from both charts.

        When nodal atlas weights are supplied, uses the weighted blend (same rule as
        closure).  Otherwise picks the reconstruction from the chart with smaller
        ``|q|`` (minimum-radius rule), which is the round-trip used during init noise
        and smoothing when weights are not yet needed.
        """
        uA = qA_arr[0::2]
        vA = qA_arr[1::2]
        uB = qB_arr[0::2]
        vB = qB_arr[1::2]
        rA2 = uA**2 + vA**2
        rB2 = uB**2 + vB**2
        n = len(uA)
        mA = self._m_from_q_numpy(qA_arr, np.zeros(n, dtype=np.int32))
        mB = self._m_from_q_numpy(qB_arr, np.ones(n, dtype=np.int32))
        if wa_node is not None and wb_node is not None:
            wa = np.asarray(wa_node[:n], dtype=np.float64)
            wb = np.asarray(wb_node[:n], dtype=np.float64)
            wsum = np.maximum(wa + wb, 1e-12)
            wa = wa / wsum
            wb = wb / wsum
            m = wa[:, None] * mA + wb[:, None] * mB
            m_norm = np.linalg.norm(m, axis=1)
            m_norm = np.maximum(m_norm, 1e-15)
            m = m / m_norm[:, None]
        else:
            chooseA = rA2 <= rB2
            m = np.where(chooseA[:, None], mA, mB)
        return m

    def _interpolate_m_cart_numpy(self, q_func=None, target=None):
        """Atlas-aware Cartesian m interpolation used by diagnostics/outputs."""
        if target is None:
            target = self.m_cart
        n_owned = self.V_q.dofmap.index_map.size_local
        n_total = n_owned + self.V_q.dofmap.index_map.num_ghosts

        # If a specific q field is requested, project that field directly.
        if q_func is not None:
            if q_func is self.qB:
                node_chart = np.ones(n_total, dtype=np.int32)
            elif q_func is self.qA:
                node_chart = np.zeros(n_total, dtype=np.int32)
            else:
                node_chart = np.zeros(n_total, dtype=np.int32)
            m_values = self._m_from_q_numpy(q_func.x.array[: n_total * 2], node_chart)
        else:
            # Before init_dynamic (or if qB was not restored), avoid using
            # uninitialized qB in the blend to prevent fake uniform -Y output.
            if not getattr(self, "_qB_ready", False):
                m_values = self._m_from_q_numpy(
                    self.qA.x.array[: n_total * 2], np.zeros(n_total, dtype=np.int32)
                )
            else:
                wa_node, wb_node = self._compute_nodal_atlas_weights(n_total)
                m_values = self._blend_m_from_qA_qB(
                    self.qA.x.array[: n_total * 2],
                    self.qB.x.array[: n_total * 2],
                    wa_node=wa_node,
                    wb_node=wb_node,
                )
        target.x.array[: n_owned * SPIN_DIM] = m_values[:n_owned].flatten()
        target.x.scatter_forward()

    def _set_m_from_state_numpy(self, q_arr, target):
        """Atlas-aware previous-state Cartesian m for thermo checks."""
        n_owned = self.V_q.dofmap.index_map.size_local
        n_total = n_owned + self.V_q.dofmap.index_map.num_ghosts
        qB_ref = getattr(self, "_qB_backup_for_thermo", None)
        if qB_ref is None or len(qB_ref) < n_total * 2:
            qB_ref = self.qB.x.array
        wa_node, wb_node = self._compute_nodal_atlas_weights(n_total)
        m_values = self._blend_m_from_qA_qB(
            q_arr[: n_total * 2],
            qB_ref[: n_total * 2],
            wa_node=wa_node,
            wb_node=wb_node,
        )
        target.x.array[: n_owned * SPIN_DIM] = m_values[:n_owned].flatten()
        target.x.scatter_forward()

    # ==================================================================
    # Materials interpolation and q initialization
    # ==================================================================

    def _interpolate_materials_properties(self, materials: list[dict]):
        V_dg0 = fem.functionspace(self.domain, ("DG", 0))
        V_dg0_vec = fem.functionspace(self.domain, ("DG", 0, (SPIN_DIM,)))
        self.Q1 = fem.Function(V_dg0)
        self.Ms = fem.Function(V_dg0)
        self.Aex = fem.Function(V_dg0)
        self.K = fem.Function(V_dg0)
        self.axis_uni_anisotropy = fem.Function(V_dg0_vec)
        self.K2K1_ratio = fem.Function(V_dg0)
        for mat in materials:
            cells_index = self.cell_tags.find(mat["tag"])
            self.Q1.x.array[cells_index] = float(mat["Q1"])
            self.Ms.x.array[cells_index] = float(mat["Ms"])
            self.Aex.x.array[cells_index] = float(mat["Aex"])
            self.K.x.array[cells_index] = float(mat["K"])
            self.K2K1_ratio.x.array[cells_index] = float(mat["K2K1_ratio"])
            if "uniaxial_anisotropy" in self.energy_terms and float(mat["K"]) != 0.0:
                axis = mat.get("anisotropy_axis", None)
                if axis is None:
                    raise Exception("Provide key anisotropy_axis in the material dictionary")
                axis_arr = validate_spin_vector(
                    axis, f"anisotropy_axis for material tag {mat['tag']}"
                )
                for j in cells_index:
                    self.axis_uni_anisotropy.x.array[
                        SPIN_DIM * j : SPIN_DIM * (j + 1)
                    ] = axis_arr
        self.Q1.x.scatter_forward()
        self.Ms.x.scatter_forward()
        self.Aex.x.scatter_forward()
        self.K.x.scatter_forward()
        self.axis_uni_anisotropy.x.scatter_forward()
        self.K2K1_ratio.x.scatter_forward()

    def _cart_to_stereo_numpy(self, m_vectors, chart=0):
        """Forward stereographic projection for one chart (raw formula only).

        Called only from :meth:`_cart_to_stereo_dual_numpy`.  Near-pole and exact-pole
        repair (origin on the complementary chart, finite cap on the singular chart) is
        applied there via :meth:`_cap_singular_chart`.
        """
        m_vectors = np.asarray(m_vectors, dtype=np.float64)
        if m_vectors.ndim == 1:
            m_vectors = m_vectors.reshape(1, 3)
        pole = self._pole_np
        e1 = self._e1_np
        e2 = self._e2_np
        s = 1.0 if chart == 0 else -1.0
        m_dot_pole = np.clip(m_vectors @ pole, -1.0, 1.0)
        denom = 1.0 - s * m_dot_pole
        tol = self._stereo_exact_pole_tol()
        # Avoid division blow-up at the projection pole; at m=±P the numerator is zero
        # so (u,v)=(0,0) here and the dual path assigns the authoritative coordinates.
        safe_denom = np.where(np.abs(denom) <= tol, 1.0, denom)
        u = (m_vectors @ e1) / safe_denom
        v = (m_vectors @ e2) / safe_denom
        return u, v

    def _m_from_q_numpy(self, q_arr, node_chart):
        u = q_arr[0::2]
        v = q_arr[1::2]
        r2 = u**2 + v**2
        denom = 1 + r2
        sign = np.where(node_chart == 0, 1.0, -1.0)
        pole = self._pole_np
        e1 = self._e1_np
        e2 = self._e2_np
        m = np.empty((len(u), SPIN_DIM), dtype=np.float64)
        for j in range(SPIN_DIM):
            m[:, j] = (-sign * pole[j] * (1 - r2) + 2 * u * e1[j] + 2 * v * e2[j]) / denom
        return m

    # ------------------------------------------------------------------
    # Dual-chart initialization (dynamic solvers only; see theory_dynamics)
    # ------------------------------------------------------------------

    def _stereo_pole_singularity_eps(self) -> float:
        """Near-pole band half-width derived from ``atlas_overlap_tau``.

        Used for atlas-style routing during init noise (active-chart selection) and
        for defining ``R_cap`` together with :meth:`_stereo_exact_pole_tol`.
        """
        overlap = float(getattr(self, "atlas_overlap_tau", 0.05))
        return max(1e-8, min(0.1, 0.5 * overlap))

    def _stereo_exact_pole_tol(self) -> float:
        """Floating-point tolerance for detecting exact projection-pole coincidence."""
        return max(1e-12, 64.0 * np.finfo(np.float64).eps)

    def _stereo_pole_cap_radius(self) -> float:
        """Finite cap radius ``R_cap`` for the singular chart at exact poles.

        Matches the stereographic radius at the edge of the near-pole neighborhood
        defined by ``max(eps_pole, eps_exact)`` where ``eps_pole`` comes from
        :meth:`_stereo_pole_singularity_eps`.
        """
        eps = max(self._stereo_pole_singularity_eps(), self._stereo_exact_pole_tol())
        return float(np.sqrt((2.0 - eps) / eps))

    def _normalized_q_direction(self, q_flat, n_nodes):
        """Return per-node unit directions for chart coordinates when available."""
        if q_flat is None:
            return None
        q_flat = np.asarray(q_flat, dtype=np.float64)
        if q_flat.size < 2 * n_nodes:
            return None
        dirs = np.zeros((n_nodes, 2), dtype=np.float64)
        u = q_flat[0 : 2 * n_nodes : 2]
        v = q_flat[1 : 2 * n_nodes : 2]
        r = np.sqrt(u * u + v * v)
        mask = r > 1e-14
        if np.any(mask):
            dirs[mask, 0] = u[mask] / r[mask]
            dirs[mask, 1] = v[mask] / r[mask]
        return dirs

    def _cap_singular_chart(self, u, v, singular_mask, prev_q_flat=None):
        """
        Store an exact-pole singular chart at finite radius ``R_cap``.

        The complementary chart already holds the pole at ``q = 0``; this routine
        prevents the inactive chart from collapsing to the origin.  Direction in the
        ``(u, v)`` plane is taken from ``prev_q_flat`` when available (approach path
        during closure or init round-trips); otherwise ``(+e1, 0)`` is used.
        """
        if not np.any(singular_mask):
            return u, v
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        dirs = self._normalized_q_direction(prev_q_flat, len(u))
        cap = self._stereo_pole_cap_radius()
        if dirs is None:
            dirs = np.zeros((len(u), 2), dtype=np.float64)
        fallback = np.sqrt(dirs[:, 0] * dirs[:, 0] + dirs[:, 1] * dirs[:, 1]) <= 1e-14
        dirs[fallback, 0] = 1.0
        dirs[fallback, 1] = 0.0
        u[singular_mask] = cap * dirs[singular_mask, 0]
        v[singular_mask] = cap * dirs[singular_mask, 1]
        return u, v

    def _cart_to_stereo_dual_numpy(self, m_vectors, prev_qA=None, prev_qB=None):
        """
        Project Cartesian ``m`` onto both charts with exact-pole repair.

        Algorithm per node:

        1. Normalize to unit ``m`` (degenerate input → ``-P``).
        2. Evaluate both forward stereographic maps.
        3. At ``m = +P`` (``m·P >= 1 - tol``): set ``q_B = 0``, cap ``q_A`` at ``R_cap``.
        4. At ``m = -P``: set ``q_A = 0``, cap ``q_B`` at ``R_cap``.
        5. Away from exact poles: leave forward-map values unchanged.

        Near poles (not exact) both coordinates are returned as evaluated; atlas
        weights suppress the ill-conditioned chart during assembly.

        Parameters
        ----------
        prev_qA, prev_qB : array-like, optional
            Flat chart arrays used only to inherit singular-chart direction at exact poles.
        """
        m_vectors = np.asarray(m_vectors, dtype=np.float64)
        if m_vectors.ndim == 1:
            m_vectors = m_vectors.reshape(1, 3)
        norms = np.linalg.norm(m_vectors, axis=1)
        safe = norms > 1e-15
        m_unit = np.array(m_vectors, copy=True)
        if np.any(safe):
            m_unit[safe] = (m_unit[safe].T / norms[safe]).T
        if np.any(~safe):
            m_unit[~safe, :] = -self._pole_np
        mdot = np.clip(m_unit @ self._pole_np, -1.0, 1.0)
        uA, vA = self._cart_to_stereo_numpy(m_unit, chart=0)
        uB, vB = self._cart_to_stereo_numpy(m_unit, chart=1)
        tol = self._stereo_exact_pole_tol()
        # Exact poles are the only truly singular states. There the chart whose
        # pole is not being hit is fixed to its origin and becomes authoritative.
        chart_a_singular = mdot >= 1.0 - tol
        chart_b_singular = mdot <= -1.0 + tol
        if np.any(chart_a_singular):
            uB = np.asarray(uB, dtype=np.float64)
            vB = np.asarray(vB, dtype=np.float64)
            uB[chart_a_singular] = 0.0
            vB[chart_a_singular] = 0.0
        if np.any(chart_b_singular):
            uA = np.asarray(uA, dtype=np.float64)
            vA = np.asarray(vA, dtype=np.float64)
            uA[chart_b_singular] = 0.0
            vA[chart_b_singular] = 0.0
        # Singular chart at an exact pole: finite radius, direction from history when
        # available (approach path) for interpretable storage/output.
        uA, vA = self._cap_singular_chart(uA, vA, chart_a_singular, prev_qA)
        uB, vB = self._cap_singular_chart(uB, vB, chart_b_singular, prev_qB)
        return uA, vA, uB, vB

    def _write_dual_q_to_state(self, uA, vA, uB, vB):
        """Write owned chart DOFs, seed BDF history slots, and sync ``u_mixed``."""
        n_owned = self.V_q.dofmap.index_map.size_local
        self.qA.x.array[: n_owned * 2 : 2] = uA[:n_owned]
        self.qA.x.array[1 : n_owned * 2 : 2] = vA[:n_owned]
        self.qB.x.array[: n_owned * 2 : 2] = uB[:n_owned]
        self.qB.x.array[1 : n_owned * 2 : 2] = vB[:n_owned]
        self.qA.x.scatter_forward()
        self.qB.x.scatter_forward()
        if hasattr(self, "qA_prev"):
            self.qA_prev.x.array[:] = self.qA.x.array[:]
            self.qA_prev.x.scatter_forward()
        if hasattr(self, "qA_prev_prev"):
            self.qA_prev_prev.x.array[:] = self.qA.x.array[:]
            self.qA_prev_prev.x.scatter_forward()
        if hasattr(self, "qB_prev"):
            self.qB_prev.x.array[:] = self.qB.x.array[:]
            self.qB_prev.x.scatter_forward()
        if hasattr(self, "qB_prev_prev"):
            self.qB_prev_prev.x.array[:] = self.qB.x.array[:]
            self.qB_prev_prev.x.scatter_forward()
        self._qB_ready = True
        self._dual_charts_initialized = True
        # Keep SNES initial guess aligned with both chart fields (not qB only).
        if hasattr(self, "u_mixed"):
            self._sync_mixed_q_from_state()

    def _sync_dual_charts_from_cartesian(self, m_vectors=None):
        """
        Rebuild both charts from a trusted Cartesian magnetization field.

        If ``m_vectors`` is omitted, forms ``m`` from the current ``qA/qB`` using
        atlas nodal weights when available, otherwise from chart A alone.  Passes
        the current chart state as ``prev_q*`` so exact-pole singular directions are
        preserved.  Used by cold-start bootstrap and post-acceptance closure.
        """
        n_owned = self.V_q.dofmap.index_map.size_local
        n_total = n_owned + self.V_q.dofmap.index_map.num_ghosts
        if m_vectors is None:
            if getattr(self, "_qB_ready", False) and hasattr(self, "wA"):
                wa_node, wb_node = self._compute_nodal_atlas_weights(n_total)
                m_vectors = self._blend_m_from_qA_qB(
                    self.qA.x.array[: n_total * 2],
                    self.qB.x.array[: n_total * 2],
                    wa_node=wa_node,
                    wb_node=wb_node,
                )
            else:
                m_vectors = self._m_from_q_numpy(
                    self.qA.x.array[: n_total * 2], np.zeros(n_total, dtype=np.int32)
                )
        prev_qA = self.qA.x.array[: n_total * 2].copy() if hasattr(self, "qA") else None
        prev_qB = self.qB.x.array[: n_total * 2].copy() if hasattr(self, "qB") else None
        uA, vA, uB, vB = self._cart_to_stereo_dual_numpy(m_vectors, prev_qA=prev_qA, prev_qB=prev_qB)
        self._write_dual_q_to_state(uA, vA, uB, vB)

    def _seed_atlas_masks_from_m_numpy(self, m_vectors, full_relax: bool = True):
        """
        Assign cell-wise atlas weights from cell-averaged ``m·P``.

        At initialization (``full_relax=True``) weights are set directly from the
        prescribed ``m_trusted`` field.  During time stepping (``full_relax=False``)
        new weights are relaxed toward the previous state to avoid abrupt jumps.
        """
        if not hasattr(self, "wA"):
            return
        m_vectors = np.asarray(m_vectors, dtype=np.float64)
        if m_vectors.ndim == 1:
            m_vectors = m_vectors.reshape(1, 3)
        mdot = m_vectors @ self._pole_np
        n_cells = self._num_local_cells
        tau = max(float(getattr(self, "chart_hysteresis_tau", 0.0)), 0.0)
        overlap = max(float(getattr(self, "atlas_overlap_tau", 0.05)), 0.0)
        # Exclusive-chart band: |m·P| > tau + overlap.
        band = max(tau + overlap, 1e-12)

        chiA = np.zeros(n_cells, dtype=self.chiA.x.array.dtype)
        chiB = np.zeros(n_cells, dtype=self.chiB.x.array.dtype)
        chiAB = np.zeros(n_cells, dtype=self.chiAB.x.array.dtype)
        wA = np.zeros(n_cells, dtype=self.wA.x.array.dtype)
        wB = np.zeros(n_cells, dtype=self.wB.x.array.dtype)

        for c in range(n_cells):
            cdofs = self._cell_dofs_local[c]
            s = float(np.mean(mdot[cdofs]))
            if s > band:
                # Near +P: chart B is well conditioned.
                wa_raw, wb_raw = 0.0, 1.0
            elif s < -band:
                # Near -P: chart A is well conditioned.
                wa_raw, wb_raw = 1.0, 0.0
            else:
                # Equatorial overlap: linear blend between charts.
                wa_raw = 0.5 * (1.0 - s / band)
                wa_raw = min(1.0, max(0.0, wa_raw))
                wb_raw = 1.0 - wa_raw
            wA[c], wB[c] = wa_raw, wb_raw
            chiA[c] = 1.0 if wa_raw > 1e-12 else 0.0
            chiB[c] = 1.0 if wb_raw > 1e-12 else 0.0
            chiAB[c] = 1.0 if (wa_raw > 1e-12 and wb_raw > 1e-12) else 0.0

        if not full_relax:
            # During time stepping, blend toward new weights to avoid abrupt jumps.
            # At initialization full_relax=True assigns weights directly.
            rho = float(getattr(self, "atlas_weight_relaxation", 0.35))
            old_wA = np.array(self.wA.x.array[:n_cells], copy=True)
            wA = (1.0 - rho) * old_wA + rho * wA
            wB = 1.0 - wA

        self.chiA.x.array[:n_cells] = chiA
        self.chiB.x.array[:n_cells] = chiB
        self.chiAB.x.array[:n_cells] = chiAB
        self.wA.x.array[:n_cells] = wA
        self.wB.x.array[:n_cells] = wB
        self.chiA.x.scatter_forward()
        self.chiB.x.scatter_forward()
        self.chiAB.x.scatter_forward()
        self.wA.x.scatter_forward()
        self.wB.x.scatter_forward()
        if hasattr(self, "_atlas_nodal_weights_cache"):
            self._atlas_nodal_weights_cache = None
        self._atlas_masks_seeded = True

    def _perturb_dual_chart_noise(self, m_vectors, uA, vA, uB, vB, std_u, std_v):
        """
        Add Gaussian noise on the geometrically active chart(s), then re-project.

        Noise is routed by ``m·P`` relative to :meth:`_stereo_pole_singularity_eps`.
        The perturbed charts are blended to one Cartesian direction and passed through
        :meth:`_cart_to_stereo_dual_numpy` so both charts remain coherent.
        """
        m_vectors = np.asarray(m_vectors, dtype=np.float64)
        mdot = m_vectors @ self._pole_np
        eps = self._stereo_pole_singularity_eps()
        # Route noise to the well-conditioned chart at each node:
        #   m·P < -eps  -> chart A active
        #   m·P > +eps  -> chart B active
        #   |m·P| <= eps -> equatorial overlap: perturb both charts
        use_b = mdot > eps
        use_a = mdot < -eps
        overlap = ~(use_a | use_b)
        n = len(uA)
        noise_u = np.random.normal(0.0, std_u, size=n)
        noise_v = np.random.normal(0.0, std_v, size=n)
        uA = uA + np.where(use_a | overlap, noise_u, 0.0)
        vA = vA + np.where(use_a | overlap, noise_v, 0.0)
        uB = uB + np.where(use_b | overlap, noise_u, 0.0)
        vB = vB + np.where(use_b | overlap, noise_v, 0.0)
        qA_tmp = np.empty(n * 2, dtype=np.float64)
        qB_tmp = np.empty(n * 2, dtype=np.float64)
        qA_tmp[0::2] = uA
        qA_tmp[1::2] = vA
        qB_tmp[0::2] = uB
        qB_tmp[1::2] = vB
        # Chart noise does not commute with projection: recover a single physical
        # direction, then dual-project again so both charts stay coherent.
        m_noisy = self._blend_m_from_qA_qB(qA_tmp, qB_tmp)
        return self._cart_to_stereo_dual_numpy(m_noisy, prev_qA=qA_tmp, prev_qB=qB_tmp)

    def _ensure_node_vectors_with_ghosts(self, m_vectors):
        """
        Make sure a per-node array includes ghost DOFs, not just owned DOFs.

        In MPI, each rank owns some mesh nodes and also stores copies of a few
        neighboring nodes owned by other ranks (ghosts). Atlas seeding walks
        cells and looks up ``m`` at every DOF of each cell, including those
        ghost indices. Init helpers such as ``set_init_q_from_m_function`` only
        evaluate ``m`` on owned nodes, so the array is too short and would
        IndexError on partition-boundary cells unless we fill the ghosts here.

        Owned values are copied into a DOLFINx vector, then ``scatter_forward``
        copies each owned value to the ranks that need it as a ghost.
        """
        imap = self.V_q.dofmap.index_map
        n_owned = imap.size_local
        n_total = n_owned + imap.num_ghosts
        m = np.asarray(m_vectors, dtype=np.float64)
        if m.ndim == 1:
            m = m.reshape(1, 3)
        # Already long enough (e.g. after Helmholtz, which returns owned+ghost).
        if m.shape[0] >= n_total:
            return np.array(m[:n_total], copy=True)
        if m.shape[0] < n_owned:
            raise ValueError(
                f"Expected at least {n_owned} owned node vectors, got {m.shape[0]}"
            )
        # Block size 3: one Cartesian magnetization vector per mesh node.
        buf = la.vector(imap, 3, dtype=np.float64)
        buf.array[: n_owned * 3] = m[:n_owned].reshape(-1)
        # Send owned entries to neighboring ranks that store them as ghosts.
        buf.scatter_forward()
        return np.array(buf.array[: n_total * 3].reshape(n_total, 3), copy=True)

    def _smooth_scalar_field_on_q_nodes(self, init_arr, smoothing_len, smoothing_iters, mesh_adaptive=True):
        """
        Spatially smooth one scalar chart component (``u`` or ``v``) on q-nodes.

        Applies a Helmholtz filter repeatedly:

            (I - ℓ² Δ) u_{k+1} = u_k

        with length scale ``ℓ = smoothing_len`` (or ``smoothing_len * h`` when
        ``mesh_adaptive`` is True). Each pass damps high spatial frequencies.
        Call once per component (``uA``, ``vA``, ``uB``, ``vB``). Chart-space
        smoothing breaks ``|m|=1``, so the caller must rebuild Cartesian ``m``
        and re-project onto both charts afterwards.

        Implementation note (MPI): build **one** PETSc linear solver, reuse it
        for every smoothing pass, and destroy Mat/KSP/Vec explicitly on every
        rank before returning. Creating a new solver each pass and relying on
        Python's garbage collector to call ``destroy`` is unsafe across nodes:
        PETSc teardown is collective, but GC can run at different times on
        different ranks and hang the job.
        """
        # Collapsed scalar subspace of V_q (one DOF per chart coordinate).
        u_space, _ = self.V_q.sub(0).collapse()
        u_func = fem.Function(u_space)
        bmap = u_space.dofmap.index_map
        local_nodes = bmap.size_local
        block_size = u_space.dofmap.bs
        # Seed owned DOFs from the input array, then sync ghosts.
        for local_node in range(local_nodes):
            if local_node < len(init_arr):
                u_func.x.array[local_node * block_size] = init_arr[local_node]
        u_func.x.scatter_forward()

        u_trial = ufl.TrialFunction(u_func.function_space)
        v_test = ufl.TestFunction(u_func.function_space)
        if mesh_adaptive:
            # ℓ = smoothing_len * local mesh size h (DG0 cell field).
            h_field = self._compute_mesh_size_field()
            smoothing_coefficient = smoothing_len * h_field
            a = (
                ufl.inner(u_trial, v_test) * self.dv
                + smoothing_coefficient**2 * ufl.inner(ufl.grad(u_trial), ufl.grad(v_test)) * self.dv
            )
        else:
            # ℓ = smoothing_len (constant, in mesh length units).
            a = ufl.inner(u_trial, v_test) * self.dv + smoothing_len**2 * ufl.inner(
                ufl.grad(u_trial), ufl.grad(v_test)
            ) * self.dv

        # One LinearProblem for all passes. The RHS is L(v) = (u_func, v); each
        # solve writes into u_func, so the next pass uses the updated field.
        problem = fem.petsc.LinearProblem(
            a,
            ufl.inner(u_func, v_test) * self.dv,
            bcs=[],
            u=u_func,
            petsc_options_prefix="zmag_smooth_q_",
            petsc_options={"ksp_type": "cg", "pc_type": "jacobi"},
        )
        try:
            for _ in range(int(smoothing_iters)):
                problem.solve()
        finally:
            # Tear down PETSc objects now, on all ranks together. Disable
            # LinearProblem.__del__ so the GC does not destroy them again later.
            problem._solver.destroy()
            problem._A.destroy()
            problem._b.destroy()
            problem._x.destroy()
            problem.__del__ = lambda *args, **kwargs: None  # noqa: E731
        # Owned + ghost values; caller may index either when rebuilding m.
        return u_func.x.array.copy()

    def _initialize_and_smooth_q_from_m(
        self, m_vectors, smoothing_len, smoothing_iters, mesh_adaptive=True, std_u=0.0, std_v=0.0
    ):
        """
        Initialize qA/qB from a prescribed Cartesian magnetization field.

        Pipeline (Cartesian ``m`` is always the source of truth):

        1. Normalize inputs → ``m_trusted`` (degenerate → ``-P``).
        2. Dual-project via :meth:`_cart_to_stereo_dual_numpy`.
        3. Optional Gaussian noise on active charts + round-trip re-projection.
        4. Optional Helmholtz smoothing per scalar component + round-trip.
        5. Fill ghost DOFs on ``m_trusted`` if needed, then seed atlas masks/weights
           from that field (not from a provisional ``q`` blend).
        6. Write chart state and BDF history via :meth:`_write_dual_q_to_state`.

        At exact poles the well-conditioned chart is stored at ``q = 0`` and the
        singular chart at ``|q| = R_cap`` with direction inherited from the
        round-trip ``prev_q*`` buffers when noise or smoothing is active.
        """
        m_vectors = np.asarray(m_vectors, dtype=np.float64)
        if m_vectors.ndim == 1:
            m_vectors = m_vectors.reshape(1, 3)
        norms = np.linalg.norm(m_vectors, axis=1)
        safe = norms > 1e-15
        m_trusted = np.array(m_vectors, copy=True)
        if np.any(safe):
            m_trusted[safe] = (m_trusted[safe].T / norms[safe]).T
        if np.any(~safe):
            # Chart-A origin (-P) is the deterministic fallback for zero inputs.
            m_trusted[~safe, :] = -self._pole_np

        # --- Step 1: dual projection from trusted Cartesian m ---
        uA, vA, uB, vB = self._cart_to_stereo_dual_numpy(m_trusted)

        # --- Step 2 (optional): Gaussian noise on active chart(s) ---
        if std_u > 0.0 or std_v > 0.0:
            uA, vA, uB, vB = self._perturb_dual_chart_noise(
                m_trusted, uA, vA, uB, vB, std_u, std_v
            )
            n_q = len(uA)
            qA_tmp = np.empty(n_q * 2, dtype=np.float64)
            qB_tmp = np.empty(n_q * 2, dtype=np.float64)
            qA_tmp[0::2] = uA
            qA_tmp[1::2] = vA
            qB_tmp[0::2] = uB
            qB_tmp[1::2] = vB
            # Keep atlas seeding tied to a single physical direction after noise.
            m_trusted = self._blend_m_from_qA_qB(qA_tmp, qB_tmp)
            uA, vA, uB, vB = self._cart_to_stereo_dual_numpy(m_trusted, prev_qA=qA_tmp, prev_qB=qB_tmp)

        # --- Step 3 (optional): Helmholtz relaxation in chart space ---
        if smoothing_iters > 0 and smoothing_len > 0.0:
            # Smooth each scalar component independently (fast, but breaks the
            # sphere constraint in chart space).
            uA = self._smooth_scalar_field_on_q_nodes(uA, smoothing_len, smoothing_iters, mesh_adaptive)
            vA = self._smooth_scalar_field_on_q_nodes(vA, smoothing_len, smoothing_iters, mesh_adaptive)
            uB = self._smooth_scalar_field_on_q_nodes(uB, smoothing_len, smoothing_iters, mesh_adaptive)
            vB = self._smooth_scalar_field_on_q_nodes(vB, smoothing_len, smoothing_iters, mesh_adaptive)
            n_total = self.V_q.dofmap.index_map.size_local + self.V_q.dofmap.index_map.num_ghosts
            qA_tmp = np.empty(n_total * 2, dtype=np.float64)
            qB_tmp = np.empty(n_total * 2, dtype=np.float64)
            qA_tmp[0::2] = uA[:n_total]
            qA_tmp[1::2] = vA[:n_total]
            qB_tmp[0::2] = uB[:n_total]
            qB_tmp[1::2] = vB[:n_total]
            # Round-trip: smoothed charts -> one Cartesian m -> dual charts again.
            # prev_qA/prev_qB preserve singular-chart direction if m sits on +/-P.
            m_trusted = self._blend_m_from_qA_qB(qA_tmp, qB_tmp)
            uA, vA, uB, vB = self._cart_to_stereo_dual_numpy(m_trusted, prev_qA=qA_tmp, prev_qB=qB_tmp)

        # --- Step 4: atlas seeding and state write ---
        # Seed wA/wB from the trusted Cartesian field (hemisphere near poles),
        # not from a provisional q blend. m_trusted may still be owned-only here
        # (smoothing off, or entry points that only evaluate owned nodes), so
        # fill ghost DOFs before cell-wise averages look them up.
        m_trusted = self._ensure_node_vectors_with_ghosts(m_trusted)
        self._seed_atlas_masks_from_m_numpy(m_trusted)
        # Write owned chart DOFs, sync ghosts, and seed BDF history / u_mixed.
        self._write_dual_q_to_state(uA, vA, uB, vB)

    def _atlas_mdot_signal_nodes(self, mA, mB, n_total):
        """Weighted ``m·P`` per node for atlas mask updates."""
        wa_node, wb_node = self._compute_nodal_atlas_weights(n_total)
        wa = np.asarray(wa_node[: len(mA)], dtype=np.float64)
        wb = np.asarray(wb_node[: len(mA)], dtype=np.float64)
        wsum = np.maximum(wa + wb, 1e-12)
        wa = wa / wsum
        wb = wb / wsum
        m_signal = wa[:, None] * mA + wb[:, None] * mB
        m_norm = np.linalg.norm(m_signal, axis=1)
        m_signal = m_signal / np.maximum(m_norm[:, None], 1e-15)
        return m_signal @ self._pole_np

    # ==================================================================
    # BDF history, algebraic residual, effective field
    # ==================================================================

    def _update_bdf_coefficients(self):
        if self._bdf_active_order == 1:
            self._bdf_c0.value = 1.0
            self._bdf_c1.value = -1.0
            self._bdf_c2.value = 0.0
        elif self._bdf_active_order == 2:
            dt_n = float(self.dt.value)
            dt_nm1 = self._dt_prev_value
            if dt_nm1 <= 0:
                self._bdf_c0.value = 1.0
                self._bdf_c1.value = -1.0
                self._bdf_c2.value = 0.0
                return
            r = dt_n / dt_nm1
            self._bdf_c0.value = (1 + 2 * r) / (1 + r)
            self._bdf_c1.value = -(1 + r)
            self._bdf_c2.value = r**2 / (1 + r)

    def _shift_bdf_history_for_step(self):
        """
        Advance chart history at the start of a nonlinear solve attempt.

        Shifts the accepted state ``q(n) -> q(n-1) -> q(n-2)`` for both ``qA`` and
        ``qB`` before the trial update writes ``q(n+1)``.  Rejected attempts restore
        the pre-shift backup via :meth:`_rollback_to_step_backup`.
        """
        self.qA_prev_prev.x.array[:] = self.qA_prev.x.array[:]
        self.qA_prev.x.array[:] = self.qA.x.array[:]
        self.qA_prev_prev.x.scatter_forward()
        self.qA_prev.x.scatter_forward()

        self.qB_prev_prev.x.array[:] = self.qB_prev.x.array[:]
        self.qB_prev.x.array[:] = self.qB.x.array[:]
        self.qB_prev_prev.x.scatter_forward()
        self.qB_prev.x.scatter_forward()

    def _update_q_dot_from_bdf_history(self):
        """Recompute ``qA_dot`` and ``qB_dot`` from current charts and BDF history."""
        dt_safe = max(float(self.dt.value), 1e-15)
        c0 = float(self._bdf_c0.value)
        c1 = float(self._bdf_c1.value)
        c2 = float(self._bdf_c2.value)
        self.qA_dot.x.array[:] = (
            c0 * self.qA.x.array + c1 * self.qA_prev.x.array + c2 * self.qA_prev_prev.x.array
        ) / dt_safe
        self.qA_dot.x.scatter_forward()
        if hasattr(self, "qB_dot"):
            self.qB_dot.x.array[:] = (
                c0 * self.qB.x.array + c1 * self.qB_prev.x.array + c2 * self.qB_prev_prev.x.array
            ) / dt_safe
            self.qB_dot.x.scatter_forward()

    def _compute_residual_norm(self):
        # Ensure residual norm corresponds to the current iterate/state.
        if hasattr(self, "_snes_problem") and self._snes_problem is not None:
            self._snes_problem.F(self.solver_dynamic, self.u_mixed.x.petsc_vec, self.b)
        return self.b.norm()

    def _exchange_field_cart_chart(self, q, sign):
        m = self.stereo_to_cart_chart(q, sign)
        return ufl.div(ufl.grad(m))

    def _uniaxial_field_cart_chart(self, q, sign):
        m = self.stereo_to_cart_chart(q, sign)
        return self.Q1 * ufl.inner(m, self.axis_uni_anisotropy) * self.axis_uni_anisotropy

    def _cubic_anisotropy_field_cart(self, m):
        k1 = self.Q1 * self.K
        k2 = k1 * self.K2K1_ratio
        return ufl.as_vector(
            [
                2 * k1 * m[0] * (m[1] ** 2 + m[2] ** 2) + 4 * k2 * m[0] * m[1] ** 2 * m[2] ** 2,
                2 * k1 * m[1] * (m[0] ** 2 + m[2] ** 2) + 4 * k2 * m[1] * m[0] ** 2 * m[2] ** 2,
                2 * k1 * m[2] * (m[0] ** 2 + m[1] ** 2) + 4 * k2 * m[2] * m[0] ** 2 * m[1] ** 2,
            ]
        )

    def _atlas_blended_effective_field_expr(self):
        """Atlas-weighted micromagnetic effective field in Cartesian space."""
        mA = self.stereo_to_cart_chart(self.qA, +1.0)
        mB = self.stereo_to_cart_chart(self.qB, -1.0)
        terms = []

        if "exchange" in self.energy_terms:
            terms.append(self.wA * self._exchange_field_cart_chart(self.qA, +1.0))
            terms.append(self.wB * self._exchange_field_cart_chart(self.qB, -1.0))
        if "uniaxial_anisotropy" in self.energy_terms:
            terms.append(self.wA * self._uniaxial_field_cart_chart(self.qA, +1.0))
            terms.append(self.wB * self._uniaxial_field_cart_chart(self.qB, -1.0))
        if "cubic_anisotropy" in self.energy_terms:
            terms.append(self.wA * self._cubic_anisotropy_field_cart(mA))
            terms.append(self.wB * self._cubic_anisotropy_field_cart(mB))
        if "demag" in self.energy_terms:
            terms.append(self.h_deg)
        if "zeeman" in self.energy_terms:
            terms.append(self.h_zee)

        if not terms:
            return ufl.as_vector([0.0, 0.0, 0.0])

        h_eff = terms[0]
        for term in terms[1:]:
            h_eff += term
        return h_eff

    def effective_field(self, target=None):
        r"""
        Assemble the atlas-weighted effective field :math:`\mathbf{H}_{\mathrm{eff}}`.

        Exchange and anisotropy contributions are blended with ``wA``/``wB`` from both
        charts; demagnetizing and Zeeman terms use the solver's shared field state.
        """
        h_eff = target if target is not None else fem.Function(self.V_h)
        h_eff.interpolate(
            fem.Expression(
                self._atlas_blended_effective_field_expr(),
                self.V_h.element.interpolation_points,
            )
        )
        h_eff.x.scatter_forward()
        return h_eff

    # ==================================================================
    # Thermo buffers, energy labels, checkpoint I/O
    # ==================================================================

    def _ensure_thermo_buffers(self):
        if not hasattr(self, "_thermo_m_curr"):
            self._thermo_m_curr = fem.Function(self.V_m, name="thermo_m_curr")
            self._thermo_m_prev = fem.Function(self.V_m, name="thermo_m_prev")
            self._thermo_dm_dt = fem.Function(self.V_m, name="thermo_dm_dt")
        if "zeeman" in self.energy_terms and hasattr(self.h_zee, "x") and not hasattr(self, "_thermo_dh_zee"):
            self._thermo_dh_zee = fem.Function(self.V_hzee, name="thermo_dh_zee")

    def _thermo_check_active(self):
        demag_ramp_active = self.demag_ramp_factor is not None and not getattr(self, "_demag_ramp_done", True)
        return not demag_ramp_active

    def _energy_component_labels(self):
        labels = []
        if "exchange" in self.energy_terms:
            labels.append("exchange")
        if "uniaxial_anisotropy" in self.energy_terms:
            labels.append("uniaxial")
        if "cubic_anisotropy" in self.energy_terms:
            labels.append("cubic")
        if "demag" in self.energy_terms:
            labels.append("demag")
        if "zeeman" in self.energy_terms:
            labels.append("zeeman")
        labels.append("total")
        return labels

    def _save_func_npz(self, func, filepath):
        comm = self.domain.comm
        imap = func.function_space.dofmap.index_map
        bs = func.function_space.dofmap.index_map_bs
        n_owned = imap.size_local
        coords = func.function_space.tabulate_dof_coordinates()[:n_owned]
        values = func.x.array[: n_owned * bs].copy().reshape(n_owned, bs)
        all_coords = comm.gather(coords, root=0)
        all_values = comm.gather(values, root=0)
        if comm.rank == 0:
            all_coords = np.concatenate(all_coords, axis=0)
            all_values = np.concatenate(all_values, axis=0)
            key = np.round(all_coords, decimals=12)
            order = np.lexsort((key[:, 2], key[:, 1], key[:, 0]))
            np.savez_compressed(filepath, coords=all_coords[order], values=all_values[order])

    def _load_func_npz(self, func, filepath):
        from scipy.spatial import cKDTree

        comm = self.domain.comm
        imap = func.function_space.dofmap.index_map
        bs = func.function_space.dofmap.index_map_bs
        n_owned = imap.size_local
        data = np.load(filepath)
        saved_coords = data["coords"]
        saved_values = data["values"]
        tree = cKDTree(saved_coords)
        local_coords = func.function_space.tabulate_dof_coordinates()[:n_owned]
        distances, indices = tree.query(local_coords)
        max_dist = comm.allreduce(float(np.max(distances)) if len(distances) > 0 else 0.0, op=MPI.MAX)
        if max_dist > 1e-10:
            raise RuntimeError(f"Coordinate mismatch — max distance = {max_dist:.3e}")
        func.x.array[: n_owned * bs] = saved_values[indices].flatten()
        func.x.scatter_forward()

    def write_energies(self, filename: str):
        r"""
        Write a tab-separated energy log on rank 0.

        Rows follow ``self.energies`` (time, :math:`\Delta t^\ast`, active terms, total).

        Parameters
        ----------
        filename : str
            Output path for the energy table.
        """
        if self.domain.comm.rank == 0:
            if hasattr(self, "energies") and np.array(self.energies).ndim > 1:
                with open(filename, "w") as txt:
                    writer = csv.writer(txt, delimiter="\t")
                    header = self.energy_terms.copy()
                    header.insert(0, "time")
                    header.insert(1, "delta_t")
                    header.append("total")
                    # Guard against accidental column mismatch; keep output
                    # readable instead of silently mislabeling columns.
                    n_cols = int(self.energies.shape[1])
                    if len(header) != n_cols:
                        header = ["time", "delta_t"] + [
                            f"energy_{i}" for i in range(max(0, n_cols - 3))
                        ] + ["total"]
                    writer.writerow(header)
                    for row in np.arange(self.energies.shape[0]):
                        writer.writerow(self.energies[row, :])

    def _compute_mesh_size_field(self):
        """Compute local mesh-size field in DG0."""
        V_dg0 = fem.functionspace(self.domain, ("DG", 0))
        h_field = fem.Function(V_dg0, name="mesh_size")
        num_local_cells = self.domain.topology.index_map(self.domain.topology.dim).size_local
        if num_local_cells > 0:
            h_values = self.domain.h(self.domain.topology.dim, np.arange(num_local_cells))
            h_field.x.array[:num_local_cells] = h_values
        h_field.x.scatter_forward()
        return h_field

    # ------------------------------------------------------------------
    # Public chart initialization API (shared by demag and nodemag)
    # ------------------------------------------------------------------

    def set_init_q_from_external_field(
        self, std_u, std_v, smoothing_len, smoothing_iters, material_aware=False, zeeman_bias=0.3
    ):
        r"""
        Initialize charts from the Zeeman field direction plus Gaussian noise.

        Requires ``set_external_field`` and the ``zeeman`` energy term.  The field
        direction is preserved at every node, including near the projection pole
        ``P``: chart B is used when ``m·P > 0``, chart A when ``m·P < 0`` (see
        :ref:`dual-chart-initialization`).

        Parameters
        ----------
        std_u, std_v : float
            Noise standard deviations on the **active** chart coordinates per node.
        smoothing_len, smoothing_iters : float, int
            Helmholtz smoothing controls after projection.
        material_aware, zeeman_bias : optional
            Reserved API parameters (currently unused).
        """
        assert "zeeman" in self.energy_terms
        np.random.seed(123 * self.domain.comm.rank)
        local_size = self.h_zee.x.petsc_vec.getLocalSize()
        h_array = self.h_zee.x.array[:local_size]
        num_dofs = local_size // SPIN_DIM
        h_vectors = h_array.reshape(num_dofs, SPIN_DIM)
        norms = np.linalg.norm(h_vectors, axis=1)
        min_norm_threshold = 1e-10
        zero_field_mask = norms < min_norm_threshold
        norms_safe = np.maximum(norms, min_norm_threshold)
        h_unit = h_vectors / norms_safe[:, np.newaxis]
        # Zero field: antipode -P (chart-A origin, well-conditioned for projection).
        h_unit[zero_field_mask] = -self._pole_np
        _ = material_aware, zeeman_bias
        self._initialize_and_smooth_q_from_m(
            h_unit,
            smoothing_len,
            smoothing_iters,
            mesh_adaptive=True,
            std_u=std_u,
            std_v=std_v,
        )

    def set_init_q_from_angles(
        self, theta0_target, theta1_target, std_u, std_v, smoothing_len, smoothing_iters
    ):
        r"""
        Initialize uniform magnetization from spherical angles ``(theta0, theta1)`` plus noise.

        Uses dual-chart projection; noise is applied on the active chart per node.

        Parameters
        ----------
        theta0_target, theta1_target : float
            Spherical angles defining the target direction.
        std_u, std_v, smoothing_len, smoothing_iters
            Noise and smoothing parameters.
        """
        np.random.seed(123 * self.domain.comm.rank)
        u_space, _ = self.V_q.sub(0).collapse()
        num_dofs = u_space.dofmap.index_map.size_local
        mx_target = np.sin(theta0_target) * np.cos(theta1_target)
        my_target = np.sin(theta0_target) * np.sin(theta1_target)
        mz_target = np.cos(theta0_target)
        m_target = np.tile(
            np.array([mx_target, my_target, mz_target], dtype=np.float64), (num_dofs, 1)
        )
        self._initialize_and_smooth_q_from_m(
            m_target,
            smoothing_len,
            smoothing_iters,
            mesh_adaptive=True,
            std_u=std_u,
            std_v=std_v,
        )

    def set_init_q_from_theta_function(self, eval_theta_init, smoothing_len, smoothing_iters):
        r"""
        Initialize from a spatially varying angle field ``eval_theta_init(x)``.

        Cartesian directions are built from ``(theta0, theta1)`` and projected onto
        both charts (pole-aware routing).

        Parameters
        ----------
        eval_theta_init : callable
            Returns ``(theta0, theta1)`` arrays for node coordinates ``x`` of shape ``(3, N)``.
        smoothing_len, smoothing_iters : float, int
            Post-projection smoothing.
        """
        u_space, _ = self.V_q.sub(0).collapse()
        dof_coords = u_space.tabulate_dof_coordinates()[: u_space.dofmap.index_map.size_local]
        theta0_arr, theta1_arr = eval_theta_init(dof_coords.T)
        mx = np.sin(theta0_arr) * np.cos(theta1_arr)
        my = np.sin(theta0_arr) * np.sin(theta1_arr)
        mz = np.cos(theta0_arr)
        m_vectors = np.column_stack([mx, my, mz])
        self._initialize_and_smooth_q_from_m(
            m_vectors, smoothing_len, smoothing_iters, mesh_adaptive=True
        )

    def set_init_q_from_m_function(self, eval_m_init, smoothing_len, smoothing_iters):
        r"""
        Initialize by projecting ``eval_m_init(x)`` onto both atlas charts.

        Accepts any magnetization field, including ``m = ±P`` and directions close to
        the projection pole.  Vectors are normalized per node inside
        :meth:`_initialize_and_smooth_q_from_m`; degenerate inputs default to ``-P``
        (chart-A origin).  At exact poles the well-conditioned chart is stored at
        ``q = 0`` and the singular chart at ``|q| = R_cap``.

        Parameters
        ----------
        eval_m_init : callable
            Returns ``(mx, my, mz)`` arrays for node coordinates ``x`` of shape ``(3, N)``.
        smoothing_len, smoothing_iters : float, int
            Post-projection smoothing.
        """
        u_space, _ = self.V_q.sub(0).collapse()
        dof_coords = u_space.tabulate_dof_coordinates()[: u_space.dofmap.index_map.size_local]
        mx, my, mz = eval_m_init(dof_coords.T)
        m_vectors = np.column_stack([mx, my, mz]).astype(np.float64, copy=False)
        self._initialize_and_smooth_q_from_m(
            m_vectors, smoothing_len, smoothing_iters, mesh_adaptive=True
        )

    # ------------------------------------------------------------------
    # Checkpoint restore and dynamic teardown (shared by demag / nodemag)
    # ------------------------------------------------------------------

    def restore_checkpoint_atlas_state(self, filename: str) -> None:
        """
        Restore atlas chart fields, BDF history, and optional demag potential.

        Delegates to the module-level :func:`restore_checkpoint_atlas_state` loader,
        which also rehydrates DG0 atlas weights and sets restart flags.

        Parameters
        ----------
        filename : str
            Checkpoint stem (``_qA.npz`` / ``_qB.npz`` siblings required).
        """
        restore_checkpoint_atlas_state(self, filename)

    def destroy_dynamic(self):
        """
        Release SNES/KSP objects and close XDMF time-series writers.

        Safe to call at the end of a simulation segment or before process exit.
        """
        if hasattr(self, "_xdmf_time_writers"):
            for target in list(self._xdmf_time_writers.keys()):
                self._close_xdmf_time_writer(target)
        if hasattr(self, "solver_dynamic") and self.solver_dynamic is not None:
            self.solver_dynamic.destroy()
        if hasattr(self, "b") and self.b is not None:
            self.b.destroy()
        if hasattr(self, "J") and self.J is not None:
            self.J.destroy()
        if hasattr(self, "P") and self.P is not None:
            self.P.destroy()

