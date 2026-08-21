r"""Dynamic stereographic LLG with self-consistent demagnetizing field.

:class:`Z_sim_dynamic_stereo_demag` extends
:class:`~Zmag_bins.Zmag_dynamic_stereo_base.Z_sim_dynamic_stereo_base` with mixed
unknowns :math:`(q_A,q_B,\varphi)` and hybrid FEM--BEM magnetostatics
(:math:`\mathbf{H}_{\mathrm{d}}=-\nabla\varphi`).

The weak form uses Gilbert damping in chart space, the stereographic exchange metric
:math:`P_\mu`, atlas-weighted energies, and a Poisson subsystem for :math:`\varphi`. Each
accepted timestep advances BDF history, updates DG0 atlas weights, and may apply chart
closure near region boundaries.

Typical workflow: ``set_mesh`` → ``set_materials`` → ``set_external_field`` →
``set_magnetostatics`` → ``set_BEMcpp`` → chart initialization → ``init_dynamic`` →
``solve_problem``.

Use :func:`~Zmag_bins.Zmag_dynamic_stereo_solvers.create_stereo_solver` with
``model="demag"``. ``energy_terms`` must include ``"demag"``."""


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

from Zmag_bins.Zmag_dynamic_stereo_base import (
    Z_sim_dynamic_stereo_base,
    SPIN_DIM,
    apply_cfl_settings,
    convergence_tolerances,
    init_stereographic_chart_basis,
    merge_convergence_settings,
    parent_dof_map_from_collapse,
    save_checkpoint_atlas_state,
    save_checkpoint_time_state,
    _checkpoint_base_path,
)
from Zmag_bins.Zmag_magnetostatic import Z_sim_magnetostatic
from Zmag_bins.tools import compute_Hzee_coil
from bem_assembler.bem_assembler import BEMAssembler


print_flush = partial(print, flush=True)


class Z_sim_dynamic_stereo_demag(Z_sim_dynamic_stereo_base):
    """
    Stereographic LLG with mixed unknowns ``(q_A, q_B, phi)`` and magnetostatics.

    Inherits the adaptive time loop from
    :class:`~Zmag_bins.Zmag_dynamic_stereo_base.Z_sim_dynamic_stereo_base` and adds
    Fredkin--Koehler magnetostatics plus optional C++ BEM acceleration.
    """

    def __init__(self, args_init_sim: dict) -> None:
        """
        Initialize demag solver configuration and stereographic chart basis.

        Parameters
        ----------
        args_init_sim : dict
            Requires ``alpha``, ``energy_terms`` (including ``demag``), ``NeumannBC``,
            ``printout``, ``path_dirs``. Optional: ``projection_pole``, ``sigma2_floor``.

        Raises
        ------
        ValueError
            If ``demag`` is missing from ``energy_terms``.
        """
        self.alpha = args_init_sim["alpha"]
        self.energy_terms = args_init_sim["energy_terms"]
        self.NeumannBC = args_init_sim["NeumannBC"]
        self.printout = args_init_sim["printout"]
        self.path_dirs = args_init_sim["path_dirs"]

        self._pole_np, self._e1_np, self._e2_np = init_stereographic_chart_basis(args_init_sim)
        self._sigma2_floor = args_init_sim.get("sigma2_floor", 0.25)

        if "demag" not in self.energy_terms:
            raise ValueError(
                f"[__init__] Mixed formulation requires 'demag' in energy_terms. Got: {self.energy_terms}"
            )

    def _init_after_mesh(self, surface_tag: int):
        """Initialize qA/qB/phi mixed spaces and fields."""
        degree = 1

        # Primary spaces
        self.V_q = fem.functionspace(self.domain, ("CG", degree, (2,)))
        self.V_phi = fem.functionspace(self.domain, ("CG", degree))
        self.V_m = fem.functionspace(self.domain, ("CG", degree, (SPIN_DIM,)))
        self.V_h = fem.functionspace(self.domain, ("DG", degree, (SPIN_DIM,)))
        self.V_scaled = fem.functionspace(self.domain, ("DG", degree, (SPIN_DIM,)))
        self.V_hzee = fem.functionspace(self.domain, ("CG", degree, (SPIN_DIM,)))

        # Mixed space [qA, qB, phi]
        element_q = self.V_q.ufl_element()
        element_phi = self.V_phi.ufl_element()
        mixed_el = mixed_element([element_q, element_q, element_phi])
        self.V_mixed = fem.functionspace(self.domain, mixed_el)

        self.u_mixed = fem.Function(self.V_mixed, name="mixed_solution_qAqB")
        self.qA_new_mixed, self.qB_new_mixed, self.phi_mixed = ufl.split(self.u_mixed)
        self.v_mixed = ufl.TestFunction(self.V_mixed)
        self.du_mixed = ufl.TrialFunction(self.V_mixed)
        self.v_qA, self.v_qB, self.v_phi = ufl.split(self.v_mixed)

        # Canonical chart fields.
        self.qA = fem.Function(self.V_q, name="qA")
        self.qA_prev = fem.Function(self.V_q, name="qA_prev")
        self.qA_prev_prev = fem.Function(self.V_q, name="qA_prev_prev")
        self.qA_dot = fem.Function(self.V_q, name="qA_dot")

        self.qB = fem.Function(self.V_q, name="qB")
        self.qB_prev = fem.Function(self.V_q, name="qB_prev")
        self.qB_prev_prev = fem.Function(self.V_q, name="qB_prev_prev")
        self.qB_dot = fem.Function(self.V_q, name="qB_dot")
        # qB becomes trustworthy once dual-chart initialization or checkpoint
        # restore has rebuilt a coherent qA/qB pair.
        self._qB_ready = False

        self.m_cart = fem.Function(self.V_m, name="magnetization_cart")
        self.M_scaled = fem.Function(self.V_scaled, name="magnetization_cart_scaled")
        self.Hd_scaled = fem.Function(self.V_scaled, name="demag_scaled")
        self.B_scaled = fem.Function(self.V_scaled, name="induction_scaled")
        self.h_deg = fem.Function(self.V_h, name="demag_field")

        # Atlas activity diagnostics (cell-wise DG0 masks/weights).
        V_dg0_chart = fem.functionspace(self.domain, ("DG", 0))
        self.chart_ind = fem.Function(V_dg0_chart, name="chart_indicator")
        self.chart_ind.x.array[:] = 0.0
        self.chart_ind.x.scatter_forward()

        # Atlas masks
        self.chiA = fem.Function(V_dg0_chart, name="chiA")
        self.chiB = fem.Function(V_dg0_chart, name="chiB")
        self.chiAB = fem.Function(V_dg0_chart, name="chiAB")
        self.wA = fem.Function(V_dg0_chart, name="wA")
        self.wB = fem.Function(V_dg0_chart, name="wB")
        self.chiA.x.array[:] = 1.0
        self.chiB.x.array[:] = 0.0
        self.chiAB.x.array[:] = 0.0
        self.wA.x.array[:] = 1.0
        self.wB.x.array[:] = 0.0
        self.chiA.x.scatter_forward()
        self.chiB.x.scatter_forward()
        self.chiAB.x.scatter_forward()
        self.wA.x.scatter_forward()
        self.wB.x.scatter_forward()
        self._atlas_nodal_weights_cache = None

        # Pole/basis constants (same as base class)
        self._pole_x = fem.Constant(self.domain, PETSc.ScalarType(self._pole_np[0]))
        self._pole_y = fem.Constant(self.domain, PETSc.ScalarType(self._pole_np[1]))
        self._pole_z = fem.Constant(self.domain, PETSc.ScalarType(self._pole_np[2]))
        self._e1_x = fem.Constant(self.domain, PETSc.ScalarType(self._e1_np[0]))
        self._e1_y = fem.Constant(self.domain, PETSc.ScalarType(self._e1_np[1]))
        self._e1_z = fem.Constant(self.domain, PETSc.ScalarType(self._e1_np[2]))
        self._e2_x = fem.Constant(self.domain, PETSc.ScalarType(self._e2_np[0]))
        self._e2_y = fem.Constant(self.domain, PETSc.ScalarType(self._e2_np[1]))
        self._e2_z = fem.Constant(self.domain, PETSc.ScalarType(self._e2_np[2]))

        # Nodal chart indicators for atlas utilities and checkpoint I/O.
        n_nodes = (
            self.V_q.dofmap.index_map.size_local
            + self.V_q.dofmap.index_map.num_ghosts
        )
        self._node_chart = np.zeros(n_nodes, dtype=np.int32)
        u_sub, _ = self.V_q.sub(0).collapse()
        self._node_chart_func = fem.Function(u_sub, name="node_chart")
        self._node_chart_func.x.array[:] = 0.0
        self._node_chart_func.x.scatter_forward()

        # Measures
        self.dv = ufl.Measure(
            "dx", self.domain, subdomain_data=self.cell_tags, metadata={"quadrature_degree": 4}
        )
        self.ds = ufl.Measure(
            "ds",
            self.domain,
            subdomain_data=self.facet_tags,
            subdomain_id=surface_tag,
            metadata={"quadrature_degree": 4},
        )
        self.normal = ufl.FacetNormal(self.domain)

        # Keep base-class runtime attributes expected by CFL and output logic.
        self.surface_tag = surface_tag
        if self.domain.comm.rank == 0:
            num_global_cells = self.domain.topology.index_map(self.domain.topology.dim).size_global
            assert num_global_cells >= self.domain.comm.size, (
                f"Number of cells {num_global_cells} insufficient for {self.domain.comm.size} cores"
            )
        num_local_cells = self.domain.topology.index_map(self.domain.topology.dim).size_local
        h_values = self.domain.h(self.domain.topology.dim, np.arange(num_local_cells))
        h_min_local = min(h_values) if len(h_values) > 0 else float("inf")
        self.h_min = self.domain.comm.allreduce(h_min_local, op=MPI.MIN)
        self._num_local_cells = num_local_cells
        self._cell_dofs_local = [self.V_q.dofmap.cell_dofs(c) for c in range(num_local_cells)]

    def init_dynamic(self, args_init_dyn: dict):
        """
        Configure time integration, SNES, BDF history, ramps, and I/O.

        Call after mesh setup, materials, chart initialization, magnetostatics, and BEM assembly.

        Parameters
        ----------
        args_init_dyn : dict
            Requires ``t_start``, ``dt``, ``step_start``, ``step_end``, ``first_run``,
            ``checkpoint_every``, ``write_target``, and ``cfl_settings``.
            See :ref:`configuration` in the manual.

        Raises
        ------
        ValueError
            On invalid configuration keys or values.
        FileNotFoundError
            If required ``path_dirs`` entries are missing.
        KeyError
            If a required ``path_dirs`` key is absent.
        """
        t0_init = time.time()
        # Validate runtime output directories (directory layout is managed by launcher).
        required_dirs = ["results_dir", "checkpoint_dir"]
        for key in required_dirs:
            if key not in self.path_dirs:
                raise KeyError(f"Missing required path_dirs key: '{key}'")
            if not os.path.isdir(self.path_dirs[key]):
                raise FileNotFoundError(
                    f"Required directory does not exist: path_dirs['{key}']={self.path_dirs[key]}"
                )
        if "for_vacuum_dir" in self.path_dirs and self.path_dirs["for_vacuum_dir"]:
            if not os.path.isdir(self.path_dirs["for_vacuum_dir"]):
                raise FileNotFoundError(
                    f"for_vacuum_dir is set but missing: {self.path_dirs['for_vacuum_dir']}"
                )

        self.t = args_init_dyn["t_start"]
        self.dt = fem.Constant(self.domain, PETSc.ScalarType(args_init_dyn["dt"]))
        self.checkpoint_step = args_init_dyn["step_start"]
        self.first_run = args_init_dyn["first_run"]
        self.step_start = self.checkpoint_step if self.first_run else (self.checkpoint_step + 1)
        self.step_end = args_init_dyn["step_end"]
        self.max_retry = args_init_dyn.get("max_retry", 5)
        self.checkpoint_every = args_init_dyn["checkpoint_every"]
        self.write_target = args_init_dyn["write_target"]
        self.use_mf = args_init_dyn.get("use_mf", False)
        # JIT compilation settings (helps avoid cluster timeout/cache contention).
        self._jit_options = {}
        jit_timeout = float(args_init_dyn.get("jit_timeout_seconds", 1200.0))
        if np.isfinite(jit_timeout) and jit_timeout > 0.0:
            self._jit_options["timeout"] = max(1, int(round(jit_timeout)))
        jit_cache_base = args_init_dyn.get("jit_cache_dir", None)
        if jit_cache_base is None and "results_dir" in self.path_dirs:
            jit_cache_base = os.path.join(self.path_dirs["results_dir"], "jit_cache")
        if jit_cache_base:
            rank_cache = os.path.join(str(jit_cache_base), f"rank_{self.domain.comm.rank}")
            os.makedirs(rank_cache, exist_ok=True)
            self._jit_options["cache_dir"] = rank_cache

        # Demag ramp
        if "demag" in self.energy_terms and "demag_ramp_N" in args_init_dyn:
            self.demag_ramp_N = args_init_dyn["demag_ramp_N"]
            self.demag_ramp_delay = args_init_dyn.get("demag_ramp_delay", 0)
            self.demag_start_fraction = args_init_dyn.get("demag_start_fraction", 0.0)
            if self.demag_start_fraction <= 0.0:
                raise ValueError("[init_dynamic] demag_start_fraction must be > 0 for mixed formulation.")
            self.demag_ramp_factor = self.demag_start_fraction
            self._demag_ramp_done = False
            if self.domain.comm.rank == 0:
                print_flush(
                    f"\n[Demag Ramp] {self.demag_ramp_N} steps "
                    f"({self.demag_start_fraction*100:.0f}% -> 100%)"
                )
        else:
            self.demag_ramp_factor = None
            self.demag_ramp_delay = 0
            self.demag_start_fraction = 0.0
            self._demag_ramp_done = True

        initial_ramp = self.demag_ramp_factor if self.demag_ramp_factor is not None else 1.0
        self.demag_ramp_constant = fem.Constant(self.domain, PETSc.ScalarType(initial_ramp))

        # Zeeman ramp
        if "zeeman" in self.energy_terms and "zeeman_ramp_N" in args_init_dyn:
            self.zeeman_ramp_N = args_init_dyn["zeeman_ramp_N"]
            self.zeeman_ramp_delay = args_init_dyn.get("zeeman_ramp_delay", 0)
            self.zeeman_start_fraction = args_init_dyn.get("zeeman_start_fraction", 0.0)
            self.zeeman_ramp_factor = self.zeeman_start_fraction
            self._zeeman_ramp_done = False
            self._zeeman_field_stored = False
            if self.domain.comm.rank == 0:
                print_flush(
                    f"\n[Zeeman Ramp] {self.zeeman_ramp_N} steps "
                    f"({self.zeeman_start_fraction*100:.0f}% -> 100%)"
                )
        else:
            self.zeeman_ramp_factor = None
            self.zeeman_ramp_delay = 0
            self.zeeman_start_fraction = 0.0
            self._zeeman_ramp_done = True
            self._zeeman_field_stored = False

        self.convergence_settings = merge_convergence_settings(args_init_dyn)
        apply_cfl_settings(self, args_init_dyn)
        # Internal globalization policy (kept non-user-facing for a clean API).
        self._nonlinear_retry_growth_cooldown_steps = 3
        self._nonlinear_retry_growth_cap = 1.05
        # Residual-gate robustness:
        # - keep fast EMA for responsiveness
        # - use slow EMA to bound both inflation and over-deflation
        self._nonlinear_residual_slow_ema_beta = 0.95
        self._nonlinear_residual_gate_inflation_cap = 4.0
        # Symmetric two-timescale guard: avoid over-deflation of the fast gate.
        self._nonlinear_residual_gate_deflation_cap = 2.0
        # Merit-function acceptance for residual gate globalization.
        self._residual_merit_enable = True
        self._residual_merit_gate_relax = 3.5
        self._residual_merit_required_contraction = 0.30
        self._residual_merit_required_rho = 0.70
        self._residual_merit_floor_mult = 20.0
        # Continuation-transition stabilization (problem-agnostic): when a
        # homotopy ramp completes, briefly enforce conservative dt growth and
        # BE recovery to avoid transition shocks.
        self._continuation_transition_cooldown_default = 8
        self._continuation_transition_dt_shrink_factor = 0.5
        self._continuation_transition_dt_cap_relax = 1.10
        self._continuation_transition_cap_growth_per_step = 1.02
        self._continuation_transition_be_recovery_steps = 3
        # After a continuation-control jump, run a short reconciliation stage
        # before re-enabling strict thermo dissipation gating.
        self._continuation_transition_reconcile_steps = 1
        # Representation (atlas closure) energy jump tolerance.
        self._atlas_closure_jump_rel_tol = 5e-3
        self._atlas_closure_jump_abs_floor = 1e-8
        self._atlas_closure_jump_reject_factor = 1.25
        # Atlas radius guard: keep hard safety, but avoid rejecting tiny
        # near-threshold excursions caused by representation noise.
        self._atlas_qmax_guard_soft_factor = 1.02
        # Non-thermo (solver/gate) dt-cap memory to avoid reject/regrow oscillations.
        self._nonthermo_cap_cooldown_default = 8
        self._nonthermo_cap_growth_per_step = 1.01
        self._nonthermo_cap_recovery_growth_per_step = 1.20
        self._nonthermo_recovery_clean_steps = 6
        # Thermo-aware dt cap memory: prevents repeated overshoot/reject cycles.
        self._thermo_cap_cooldown_default = 40
        self._thermo_cap_relax_factor = 1.20
        self._thermo_cap_growth_per_step = 1.01
        self.state_monitor = {}

        chart_settings = args_init_dyn.get("chart_settings", {})
        self.chart_hysteresis_tau = args_init_dyn.get(
            "chart_hysteresis_tau", chart_settings.get("hysteresis_tau", 0.0)
        )
        self.atlas_overlap_tau = float(
            args_init_dyn.get("atlas_overlap_tau", chart_settings.get("atlas_overlap_tau", 0.05))
        )
        # Smooth DG0 weight transitions to reduce representation-driven energy jumps.
        self.atlas_weight_relaxation = float(
            args_init_dyn.get("atlas_weight_relaxation", chart_settings.get("weight_relaxation", 0.35))
        )
        self.atlas_weight_relaxation = float(min(max(self.atlas_weight_relaxation, 0.0), 1.0))
        # Hard safety for atlas manifold drift; keep conservative but not over-restrictive.
        self.atlas_qmax_guard = float(args_init_dyn.get("atlas_qmax_guard", 1.01))
        self._atlas_mask_stats = None
        self._atlas_thermo_metrics = {}
        self._thermo_last_diss_q_form = 0.0
        self._thermo_last_diss_m_cart = 0.0
        self._thermo_last_op_budget = 0.0
        self._thermo_last_ema_floor = 0.0
        self._thermo_last_switch_work = 0.0
        self._thermo_last_ramp_budget = 0.0

        # Physical acceptance and recovery controls.
        self.nonlinear_residual_max_accept = float(
            args_init_dyn.get("nonlinear_residual_max_accept", np.inf)
        )
        self.nonlinear_residual_ema_factor = float(
            args_init_dyn.get("nonlinear_residual_ema_factor", 8.0)
        )
        self.nonlinear_residual_ema_beta = float(
            args_init_dyn.get("nonlinear_residual_ema_beta", 0.8)
        )
        self.nonlinear_residual_gate_warmup_steps = int(
            args_init_dyn.get("nonlinear_residual_gate_warmup_steps", 5)
        )
        self.nonlinear_residual_gate_floor = float(
            args_init_dyn.get("nonlinear_residual_gate_floor", 1e-6)
        )
        self.nonlinear_residual_dt_relax_max = float(
            args_init_dyn.get("nonlinear_residual_dt_relax_max", 2.0)
        )
        self._residual_backtracking_enable = True
        self._residual_backtracking_max_iter = 6
        self._residual_backtracking_shrink = 0.5
        self._residual_backtracking_min_lambda = 1.0 / 64.0
        self._be_recovery_steps_on_thermo = int(args_init_dyn.get("be_recovery_steps_on_thermo", 8))
        # Explicit operator-split controls (outer fixed-point loop).
        self.outer_split_max_iter = int(args_init_dyn.get("outer_split_max_iter", 2))
        self.outer_split_tol_q = float(args_init_dyn.get("outer_split_tol_q", 5e-3))
        self.outer_split_tol_phi = float(args_init_dyn.get("outer_split_tol_phi", 1e-3))
        self.outer_split_tol_q_abs = float(args_init_dyn.get("outer_split_tol_q_abs", 1e-8))
        self.outer_split_tol_phi_abs = float(args_init_dyn.get("outer_split_tol_phi_abs", 1e-9))
        self.outer_split_max_iter_hard = int(
            args_init_dyn.get("outer_split_max_iter_hard", max(4, self.outer_split_max_iter))
        )
        self.outer_split_iter_escalation = int(args_init_dyn.get("outer_split_iter_escalation", 1))
        self.outer_split_retry_before_dt_cut = int(
            args_init_dyn.get("outer_split_retry_before_dt_cut", 2)
        )
        # Optional Anderson acceleration for outer fixed-point (qA,qB) map.
        self.outer_split_anderson_enable = bool(
            args_init_dyn.get("outer_split_anderson_enable", True)
        )
        self.outer_split_anderson_depth = int(
            args_init_dyn.get("outer_split_anderson_depth", 3)
        )
        self.outer_split_anderson_start_iter = int(
            args_init_dyn.get("outer_split_anderson_start_iter", 0)
        )
        self.outer_split_anderson_damping = float(
            args_init_dyn.get("outer_split_anderson_damping", 0.85)
        )
        self.outer_split_min_iter_with_accel = int(
            args_init_dyn.get("outer_split_min_iter_with_accel", 2)
        )
        self.outer_split_phi_growth_trigger = float(
            args_init_dyn.get("outer_split_phi_growth_trigger", 5e-2)
        )
        self.outer_split_min_iter_on_growth = int(
            args_init_dyn.get("outer_split_min_iter_on_growth", 3)
        )
        self.enforce_outer_split_acceptance = bool(
            args_init_dyn.get("enforce_outer_split_acceptance", True)
        )
        # PETSc reason=4 is SNORM_RELATIVE (step-size-only). Keep strict
        # reason-code filtering opt-in; default policy is residual/split/thermo.
        self.disallowed_snes_reasons = tuple(
            int(v) for v in args_init_dyn.get("disallowed_snes_reasons", [])
        )
        self.strict_snes_reason_check = bool(args_init_dyn.get("strict_snes_reason_check", False))
        self.snes_disallowed_residual_factor = float(
            args_init_dyn.get("snes_disallowed_residual_factor", 1.5)
        )
        self.snes_disallowed_dtmin_relax = float(
            args_init_dyn.get("snes_disallowed_dtmin_relax", 2.0)
        )
        # Thermo gate model: split-consistent energy by default, physical energy
        # retained as secondary monitor.
        self.thermo_use_split_energy = bool(args_init_dyn.get("thermo_use_split_energy", True))
        self.thermo_demag_energy_rel_floor = float(
            args_init_dyn.get("thermo_demag_energy_rel_floor", 1e-6)
        )
        self.thermo_defect_ema_factor = float(
            args_init_dyn.get("thermo_defect_ema_factor", 4.0)
        )
        self.thermo_defect_ema_warmup_steps = int(
            args_init_dyn.get("thermo_defect_ema_warmup_steps", 5)
        )
        self.thermo_enable_physical_guard = bool(
            args_init_dyn.get("thermo_enable_physical_guard", True)
        )
        self.thermo_physical_guard_ema_factor = float(
            args_init_dyn.get("thermo_physical_guard_ema_factor", 8.0)
        )
        self.thermo_physical_guard_warmup_steps = int(
            args_init_dyn.get("thermo_physical_guard_warmup_steps", 5)
        )
        self.thermo_physical_guard_abs_floor = float(
            args_init_dyn.get("thermo_physical_guard_abs_floor", 0.0)
        )
        self.thermo_physical_guard_rel_floor = float(
            args_init_dyn.get("thermo_physical_guard_rel_floor", 1e-6)
        )
        # Thermo-aware state backtracking (line search in state space) before dt cut.
        self.thermo_backtracking_enable = bool(
            args_init_dyn.get("thermo_backtracking_enable", True)
        )
        self.thermo_backtracking_max_iter = int(
            args_init_dyn.get("thermo_backtracking_max_iter", 8)
        )
        self.thermo_backtracking_shrink = float(
            args_init_dyn.get("thermo_backtracking_shrink", 0.5)
        )
        self.thermo_backtracking_min_lambda = float(
            args_init_dyn.get("thermo_backtracking_min_lambda", 1.0 / 256.0)
        )
        # Atlas closure quality monitor with optional adaptive extra passes.
        self.atlas_closure_monitor_enable = bool(
            args_init_dyn.get("atlas_closure_monitor_enable", True)
        )
        self.atlas_closure_overlap_ema_beta = float(
            args_init_dyn.get("atlas_closure_overlap_ema_beta", 0.8)
        )
        self.atlas_closure_overlap_spike_factor = float(
            args_init_dyn.get("atlas_closure_overlap_spike_factor", 4.0)
        )
        self.atlas_closure_overlap_abs_tol = float(
            args_init_dyn.get("atlas_closure_overlap_abs_tol", 1e-3)
        )
        self.atlas_closure_extra_passes_max = int(
            args_init_dyn.get("atlas_closure_extra_passes_max", 2)
        )
        # Cache nodal atlas weights until masks are updated again.
        self._atlas_nodal_weights_cache = None

        diag_settings = args_init_dyn.get("diagnostic_settings", {})
        self.enable_thermo_diagnostics = bool(
            args_init_dyn.get(
                "enable_thermo_diagnostics",
                diag_settings.get("enable_thermo_diagnostics", False),
            )
        )
        self.thermo_diag_log_accepted = bool(
            args_init_dyn.get(
                "thermo_diag_log_accepted",
                diag_settings.get("thermo_diag_log_accepted", False),
            )
        )

        self.time_order = args_init_dyn.get("time_order", 1)
        if self.time_order not in (1, 2):
            raise ValueError(f"[init_dynamic] time_order must be 1 or 2, got {self.time_order}")
        # Logging mode: "concise" (default) or "debug".
        self.log_verbosity = str(args_init_dyn.get("log_verbosity", "concise")).strip().lower()
        self._debug_logs = self.log_verbosity == "debug"
        self._bdf_c0 = fem.Constant(self.domain, PETSc.ScalarType(1.0))
        self._bdf_c1 = fem.Constant(self.domain, PETSc.ScalarType(-1.0))
        self._bdf_c2 = fem.Constant(self.domain, PETSc.ScalarType(0.0))
        if not bool(getattr(self, "_time_checkpoint_restored", False)):
            self._bdf_active_order = 1
            self._bdf_steps_since_restart = 0
            self._dt_prev_value = float(args_init_dyn["dt"])
        if bool(getattr(self, "_time_checkpoint_restored", False)):
            self._update_bdf_coefficients()

        if self.domain.comm.rank == 0:
            order_str = "BE" if self.time_order == 1 else "BE->BDF2 (auto-switch after 2 steps)"
            tol_mdot, tol_edot, steady_steps = convergence_tolerances(self.convergence_settings)
            print_flush(f"\n[Time Integration] Order = {self.time_order} ({order_str})")
            print_flush(
                f"[Timestep Controller] mode=error-controlled (CFL-capped), CFL={self.CFL:.3f}, "
                f"dt_min={self.dt_min:.1e}, dt_max={self.dt_max:.1e}, "
                f"smooth={self.smooth_factor:.2f}, growth<={self.max_growth:.2f}, "
                f"shrink>={self.min_shrink:.2f}"
            )
            print_flush(
                f"[Error Controller] rel_tol={self.error_rel_tol:.1e}, "
                f"abs_tol={self.error_abs_tol:.1e}, headroom={self.error_headroom:.2f}, "
                f"max_retry_thermo={self.max_retry_thermo}"
            )
            if self._debug_logs:
                print_flush(
                    "[Timestep Metric] dt = min(dt_error, dt_CFL_cap, dt_thermo_cap); "
                    "dt_CFL_cap from Q99(|m_t|)"
                )
            print_flush(
                f"[Stopping Criteria] ||m_dot||_rms<{tol_mdot:.1e}, |dE/dt|/max(|E|,1)<{tol_edot:.1e}, "
                f"steady_steps={steady_steps}"
            )
            print_flush(
                f"[Atlas Activity] hysteresis_tau={self.chart_hysteresis_tau:.3f}, "
                f"overlap_tau={self.atlas_overlap_tau:.3f}, "
                f"weight_relax={self.atlas_weight_relaxation:.2f}, "
                f"qmax_guard={self.atlas_qmax_guard:.4f}"
            )
            print_flush(
                f"[Acceptance Gate] residual_max={self.nonlinear_residual_max_accept:.2e}, "
                f"ema_factor={self.nonlinear_residual_ema_factor:.2f}, "
                f"warmup_steps={self.nonlinear_residual_gate_warmup_steps}, "
                f"dt_relax_max={self.nonlinear_residual_dt_relax_max:.2f}, "
                f"BE_recovery_steps={self._be_recovery_steps_on_thermo}"
            )
            print_flush(
                f"[Outer Split] max_iter={self.outer_split_max_iter}, "
                f"max_iter_hard={self.outer_split_max_iter_hard}, "
                f"tol_q={self.outer_split_tol_q:.1e}, tol_phi={self.outer_split_tol_phi:.1e}, "
                f"tol_q_abs={self.outer_split_tol_q_abs:.1e}, tol_phi_abs={self.outer_split_tol_phi_abs:.1e}, "
                f"enforce={self.enforce_outer_split_acceptance}"
            )
            print_flush(
                f"[Outer Split Accel] anderson={self.outer_split_anderson_enable}, "
                f"depth={self.outer_split_anderson_depth}, start_iter={self.outer_split_anderson_start_iter}, "
                f"damping={self.outer_split_anderson_damping:.2f}, "
                f"min_iter={self.outer_split_min_iter_with_accel}, "
                f"phi_growth_trigger={self.outer_split_phi_growth_trigger:.2e}"
            )
            print_flush(
                f"[SNES Convergence Gate] strict={self.strict_snes_reason_check}, "
                f"disallowed_reasons={self.disallowed_snes_reasons}, "
                f"residual_factor={self.snes_disallowed_residual_factor:.2f}, "
                f"dtmin_relax={self.snes_disallowed_dtmin_relax:.2f}"
            )
            print_flush(
                f"[Thermo Gate] use_split_energy={self.thermo_use_split_energy}, "
                f"demag_rel_floor={self.thermo_demag_energy_rel_floor:.1e}, "
                f"defect_ema_factor={self.thermo_defect_ema_factor:.2f}, "
                f"physical_guard={self.thermo_enable_physical_guard}, "
                f"guard_ema_factor={self.thermo_physical_guard_ema_factor:.2f}, "
                f"guard_warmup={self.thermo_physical_guard_warmup_steps}"
            )
            print_flush(
                f"[Thermo Backtracking] enable={self.thermo_backtracking_enable}, "
                f"max_iter={self.thermo_backtracking_max_iter}, "
                f"shrink={self.thermo_backtracking_shrink:.2f}, "
                f"min_lambda={self.thermo_backtracking_min_lambda:.3e}"
            )
            print_flush(
                f"[Atlas Closure Monitor] enable={self.atlas_closure_monitor_enable}, "
                f"abs_tol={self.atlas_closure_overlap_abs_tol:.1e}, "
                f"spike_factor={self.atlas_closure_overlap_spike_factor:.2f}, "
                f"extra_passes_max={self.atlas_closure_extra_passes_max}"
            )
            if self.enable_thermo_diagnostics:
                print_flush("[Diagnostics] Thermodynamic rejection diagnostics enabled")

        # SNES/KSP
        self._snes_prefix = "zmag_"
        self.solver_dynamic = PETSc.SNES().create(comm=self.domain.comm)
        self.solver_dynamic.setOptionsPrefix(self._snes_prefix)
        self._snes_type_default = "newtonls"
        self._snes_type_retry = "newtontr"
        self._snes_current_type = self._snes_type_default
        self.solver_dynamic.setType(self._snes_type_default)
        # Keep residual-based convergence primary, but use a realistic step
        # tolerance (not near machine precision) to avoid over-constraining SNES.
        self.solver_dynamic.setTolerances(rtol=1e-4, atol=1e-5, stol=1e-8, max_it=50)

        ksp = self.solver_dynamic.getKSP()
        ksp.setType("gmres")
        ksp.setInitialGuessNonzero(False)
        ksp.setTolerances(rtol=1e-2, atol=1e-8, max_it=500)
        ksp.setGMRESRestart(500)
        ksp.setPCSide(PETSc.PC.Side.RIGHT)

        # Local index maps for diagnostics/convergence writers.
        V_qA_collapsed, qA_to_mixed_raw = self.V_mixed.sub(0).collapse()
        V_phi_collapsed, phi_to_mixed_raw = self.V_mixed.sub(2).collapse()
        qA_to_mixed = parent_dof_map_from_collapse(qA_to_mixed_raw)
        phi_to_mixed = parent_dof_map_from_collapse(phi_to_mixed_raw)
        n_q_owned = V_qA_collapsed.dofmap.index_map.size_local * V_qA_collapsed.dofmap.index_map_bs
        n_phi_owned = V_phi_collapsed.dofmap.index_map.size_local * V_phi_collapsed.dofmap.index_map_bs
        self._q_owned_local = np.sort(qA_to_mixed[:n_q_owned].copy())
        self._phi_owned_local = np.sort(phi_to_mixed[:n_phi_owned].copy())
        self._q_to_mixed_map = qA_to_mixed

        # Atlas fieldsplit preconditioner.
        self._configure_fieldsplit_qA_qB_phi()
        opts = PETSc.Options()
        prefix = self._snes_prefix
        opts[f"{prefix}snes_converged_reason"] = None
        opts[f"{prefix}ksp_monitor"] = None
        opts[f"{prefix}ksp_converged_reason"] = None

        # Matrix/vector preallocation for 3-field mixed system.
        du_qA_pa, du_qB_pa, du_phi_pa = ufl.split(self.du_mixed)
        v_qA_pa, v_qB_pa, v_phi_pa = ufl.split(self.v_mixed)
        jit_opts = self._jit_options if len(self._jit_options) > 0 else None
        F_prealloc = fem.form(
            ufl.inner(du_qA_pa, v_qA_pa) * self.dv
            + ufl.inner(du_qB_pa, v_qB_pa) * self.dv
            + du_phi_pa * v_phi_pa * self.dv
            + (du_qA_pa[0] + du_qA_pa[1] + du_qB_pa[0] + du_qB_pa[1]) * v_phi_pa * self.dv
            + du_phi_pa * (v_qA_pa[0] + v_qA_pa[1] + v_qB_pa[0] + v_qB_pa[1]) * self.dv,
            jit_options=jit_opts,
        )
        self.b = fem.petsc.create_vector(self.V_mixed)
        self.J = fem.petsc.create_matrix(F_prealloc)
        self.J_form = F_prealloc
        self.bcs = []
        self.heff = fem.Function(self.V_h, name="effective_field")

        if "demag" in self.energy_terms:
            self.mag_func_mixed = fem.Function(self.V_m)
            self.phi_computed_mixed = fem.Function(self.V_phi)
            if not getattr(self, "_atlas_checkpoint_restored", False):
                if not getattr(self, "_dual_charts_initialized", False):
                    self._sync_dual_charts_from_cartesian()
            # Always resolve magnetostatics with the *live* BEM. Checkpointed
            # phi (if restored into demag_helper) is only a warm start: remesh /
            # redistribution rebuilds a different BEM, and keeping the writer's
            # phi would poison the thermo baseline and the first post-restart
            # demag residual (energy jumps of either sign; only increases fail
            # the one-sided thermo gate).
            if (
                getattr(self, "_demag_phi_checkpoint_restored", False)
                and self.domain.comm.rank == 0
                and self.printout
            ):
                print_flush(
                    "[Demag Restart] Recomputing phi with current BEM "
                    "(checkpoint phi warm-start only; remesh may change the operator)"
                )
            mag_init_func = fem.Function(self.V_m)
            self._interpolate_m_cart_numpy(target=mag_init_func)
            self.demag_helper.compute_phi(mag_init_func)
            self.demag_helper.phi.x.scatter_forward()
            phi_init = self.u_mixed.sub(2)
            phi_init.interpolate(self.demag_helper.phi)
            phi_init.x.scatter_forward()
            self.u_mixed.x.scatter_forward()
            self.phi_computed_mixed.interpolate(self.demag_helper.phi)
            self.phi_computed_mixed.x.scatter_forward()
            phi_sol_func = self.u_mixed.sub(2)
            self.h_deg.interpolate(
                fem.Expression(-ufl.grad(phi_sol_func), self.V_h.element.interpolation_points)
            )
            self.h_deg.x.scatter_forward()

        self.solver_dynamic.setFromOptions()
        ksp.setFromOptions()
        ksp.getPC().setFromOptions()

        # Initialize mixed state from chart fields (preserve checkpoint history on restart).
        self._bootstrap_chart_state_for_time_loop()

        # Jacobian diagonal / per-step residual norm writers are omitted to keep output lean.

        self.V_scalar = fem.functionspace(self.domain, ("CG", 1))
        self.q_norm_grad = fem.Function(self.V_scalar)
        self.step_now = self.step_start
        self._xdmf_checkpoint_targets = args_init_dyn.get("xdmf_checkpoint_targets", [])
        self._xdmf_time_targets = set(args_init_dyn.get("xdmf_time_targets", []))
        self._xdmf_time_all_targets = ("all" in self._xdmf_time_targets) or ("*" in self._xdmf_time_targets)
        self._xdmf_time_writers = {}
        xdmf_explicit_targets = [
            tgt for tgt in args_init_dyn.get("xdmf_time_targets", []) if tgt not in ("all", "*")
        ]
        # Independent target lists:
        # - write_target drives BP streaming
        # - xdmf_time_targets drives XDMF time-series writes
        # The solver writes the union, preserving declaration order.
        self._time_write_targets = list(dict.fromkeys(list(self.write_target) + xdmf_explicit_targets))
        # Defer first mask update to the first solve step (cheaper init).
        self._atlas_masks_step_last = None
        # IMPORTANT: check_q_magnitude performs MPI allreduce, so all ranks
        # must call it. Printing remains rank-0 only.
        r_eff = self.check_q_magnitude(silent=True)
        if self.domain.comm.rank == 0 and self.printout:
            print_flush(f"Initialization complete in {time.time() - t0_init:.2f}s")

    def compute_energy(self):
        """
        Evaluate the atlas-weighted micromagnetic energy.

        Returns
        -------
        numpy.ndarray
        ``[exchange, uniaxial, cubic, demag, zeeman, total]`` (inactive terms are zero).

        """
        qA = self.qA
        qB = self.qB
        gA = ufl.grad(qA)
        gB = ufl.grad(qB)

        mA = self.stereo_to_cart_chart(qA, +1.0)
        mB = self.stereo_to_cart_chart(qB, -1.0)

        W_ex = PETSc.ScalarType(0.0)
        W_ua = PETSc.ScalarType(0.0)
        W_ca = PETSc.ScalarType(0.0)
        W_deg = PETSc.ScalarType(0.0)
        W_zee = PETSc.ScalarType(0.0)
        W = PETSc.ScalarType(0.0)

        if "exchange" in self.energy_terms:
            W_exA = 0.5 * self.Aex * ufl.tr(ufl.dot(ufl.dot(self.P_mu_gamma(qA), gA), gA.T))
            W_exB = 0.5 * self.Aex * ufl.tr(ufl.dot(ufl.dot(self.P_mu_gamma(qB), gB), gB.T))
            W_ex = self.wA * W_exA + self.wB * W_exB
            W += W_ex

        if "uniaxial_anisotropy" in self.energy_terms:
            W_ua_A = 0.5 * self.Q1 * self.K * (1 - ufl.dot(self.axis_uni_anisotropy, mA) ** 2)
            W_ua_B = 0.5 * self.Q1 * self.K * (1 - ufl.dot(self.axis_uni_anisotropy, mB) ** 2)
            W_ua = self.wA * W_ua_A + self.wB * W_ua_B
            W += W_ua

        if "cubic_anisotropy" in self.energy_terms:
            f1A = (
                mA[0] ** 2 * mA[1] ** 2
                + mA[1] ** 2 * mA[2] ** 2
                + mA[2] ** 2 * mA[0] ** 2
            )
            f2A = mA[0] ** 2 * mA[1] ** 2 * mA[2] ** 2
            f1B = (
                mB[0] ** 2 * mB[1] ** 2
                + mB[1] ** 2 * mB[2] ** 2
                + mB[2] ** 2 * mB[0] ** 2
            )
            f2B = mB[0] ** 2 * mB[1] ** 2 * mB[2] ** 2
            W_ca_A = 0.5 * self.Q1 * self.K * (f1A + self.K2K1_ratio * f2A)
            W_ca_B = 0.5 * self.Q1 * self.K * (f1B + self.K2K1_ratio * f2B)
            W_ca = self.wA * W_ca_A + self.wB * W_ca_B
            W += W_ca

        if "demag" in self.energy_terms:
            phi_sol_func = self.u_mixed.sub(2)
            h_deg_expr = -ufl.grad(phi_sol_func)
            # Thermo/model consistency: dynamic weak form scales demag coupling
            # by demag_ramp_constant, so split-energy gate must use the same
            # effective energy during continuation.
            W_deg = -0.5 * self.demag_ramp_constant * self.Ms * (
                self.wA * ufl.inner(h_deg_expr, mA) + self.wB * ufl.inner(h_deg_expr, mB)
            )
            W += W_deg

        if "zeeman" in self.energy_terms:
            W_zee = -self.Ms * (
                self.wA * ufl.inner(self.h_zee, mA) + self.wB * ufl.inner(self.h_zee, mB)
            )
            W += W_zee

        # Physics energy excludes atlas closure penalties: chart consistency is
        # enforced by post-accept reconstruction, not by implicit penalty terms.

        energy_ex_value_loc = fem.assemble_scalar(fem.form(W_ex * self.dv))
        energy_ua_value_loc = fem.assemble_scalar(fem.form(W_ua * self.dv))
        energy_ca_value_loc = fem.assemble_scalar(fem.form(W_ca * self.dv))
        energy_deg_value_loc = fem.assemble_scalar(fem.form(W_deg * self.dv))
        energy_zee_value_loc = fem.assemble_scalar(fem.form(W_zee * self.dv))
        energy_tot_value_loc = fem.assemble_scalar(fem.form(W * self.dv))

        energy_ex = self.domain.comm.allreduce(energy_ex_value_loc, op=MPI.SUM)
        energy_ua = self.domain.comm.allreduce(energy_ua_value_loc, op=MPI.SUM)
        energy_ca = self.domain.comm.allreduce(energy_ca_value_loc, op=MPI.SUM)
        energy_deg = self.domain.comm.allreduce(energy_deg_value_loc, op=MPI.SUM)
        energy_zee = self.domain.comm.allreduce(energy_zee_value_loc, op=MPI.SUM)
        energy_tot = self.domain.comm.allreduce(energy_tot_value_loc, op=MPI.SUM)

        if self.domain.comm.rank == 0 and self.printout:
            print_flush(
                f"\nExchange={energy_ex:.4e} | Uniaxial={energy_ua:.4e} | Cubic={energy_ca:.4e} | "
                f"Demag={energy_deg:.4e} | Zeeman={energy_zee:.4e} | Total={energy_tot:.4e}"
            )

        comp_by_name = {
            "exchange": energy_ex,
            "uniaxial_anisotropy": energy_ua,
            "cubic_anisotropy": energy_ca,
            "demag": energy_deg,
            "zeeman": energy_zee,
        }
        # Preserve user-provided energy_terms ordering so output columns remain
        # consistent with headers regardless of input order.
        energies = [comp_by_name[name] for name in self.energy_terms if name in comp_by_name]
        energies.append(energy_tot)
        return energies

    def _configure_fieldsplit_qA_qB_phi(self):
        """Configure PETSc fieldsplit preconditioner for qA/qB/phi blocks."""
        ksp = self.solver_dynamic.getKSP()
        pc = ksp.getPC()
        prefix = self._snes_prefix

        V_qA_collapsed, qA_to_mixed_raw = self.V_mixed.sub(0).collapse()
        V_qB_collapsed, qB_to_mixed_raw = self.V_mixed.sub(1).collapse()
        V_phi_collapsed, phi_to_mixed_raw = self.V_mixed.sub(2).collapse()
        qA_to_mixed = parent_dof_map_from_collapse(qA_to_mixed_raw)
        qB_to_mixed = parent_dof_map_from_collapse(qB_to_mixed_raw)
        phi_to_mixed = parent_dof_map_from_collapse(phi_to_mixed_raw)

        n_qA_owned = (
            V_qA_collapsed.dofmap.index_map.size_local * V_qA_collapsed.dofmap.index_map_bs
        )
        n_qB_owned = (
            V_qB_collapsed.dofmap.index_map.size_local * V_qB_collapsed.dofmap.index_map_bs
        )
        n_phi_owned = (
            V_phi_collapsed.dofmap.index_map.size_local * V_phi_collapsed.dofmap.index_map_bs
        )

        lo, _ = self.V_mixed.dofmap.index_map.local_range
        bs = self.V_mixed.dofmap.index_map_bs
        lo_global = lo * bs

        qA_global = (lo_global + np.sort(qA_to_mixed[:n_qA_owned].copy())).astype(np.int32)
        qB_global = (lo_global + np.sort(qB_to_mixed[:n_qB_owned].copy())).astype(np.int32)
        phi_global = (lo_global + np.sort(phi_to_mixed[:n_phi_owned].copy())).astype(np.int32)

        is_qA = PETSc.IS().createGeneral(qA_global, comm=self.domain.comm)
        is_qB = PETSc.IS().createGeneral(qB_global, comm=self.domain.comm)
        is_phi = PETSc.IS().createGeneral(phi_global, comm=self.domain.comm)

        pc.setType("fieldsplit")
        pc.setFieldSplitIS(("qA", is_qA), ("qB", is_qB), ("phi", is_phi))
        pc.setFieldSplitType(PETSc.PC.CompositeType.MULTIPLICATIVE)

        opts = PETSc.Options()
        opts[f"{prefix}ksp_type"] = "gmres"
        opts[f"{prefix}pc_type"] = "fieldsplit"
        opts[f"{prefix}ksp_diagonal_scale"] = True
        opts[f"{prefix}ksp_diagonal_scale_fix"] = True
        opts[f"{prefix}snes_linesearch_type"] = "bt"
        opts[f"{prefix}fieldsplit_qA_ksp_type"] = "preonly"
        opts[f"{prefix}fieldsplit_qA_pc_type"] = "bjacobi"
        opts[f"{prefix}fieldsplit_qA_sub_pc_type"] = "lu"
        opts[f"{prefix}fieldsplit_qA_sub_pc_factor_shift_type"] = "nonzero"
        opts[f"{prefix}fieldsplit_qB_ksp_type"] = "preonly"
        opts[f"{prefix}fieldsplit_qB_pc_type"] = "bjacobi"
        opts[f"{prefix}fieldsplit_qB_sub_pc_type"] = "lu"
        opts[f"{prefix}fieldsplit_qB_sub_pc_factor_shift_type"] = "nonzero"
        opts[f"{prefix}fieldsplit_phi_ksp_type"] = "preonly"
        opts[f"{prefix}fieldsplit_phi_pc_type"] = "bjacobi"
        opts[f"{prefix}fieldsplit_phi_sub_pc_type"] = "lu"
        opts[f"{prefix}fieldsplit_phi_sub_pc_factor_shift_type"] = "nonzero"
        self.solver_dynamic.setFromOptions()

    def _build_snes_forms(self):
        """
        Assemble SNES residual/Jacobian: BDF inertia, energies, and phi equation.

        Chart blocks use atlas weights ``wA/wB``. Demag couples through
        ``A_mu_i_chart`` and ``-grad(phi)``; phi subproblem uses blended ``m``.
        """
        qA_new, qB_new, phi_sol_sym = ufl.split(self.u_mixed)
        vA, vB, v_phi = self.v_qA, self.v_qB, self.v_phi

        qA_dot_expr = (
            self._bdf_c0 * qA_new + self._bdf_c1 * self.qA_prev + self._bdf_c2 * self.qA_prev_prev
        ) / self.dt
        qB_dot_expr = (
            self._bdf_c0 * qB_new + self._bdf_c1 * self.qB_prev + self._bdf_c2 * self.qB_prev_prev
        ) / self.dt

        h_demag_sym = -ufl.grad(phi_sol_sym)

        # Gyromagnetic-style inertia: Ms * w * L_mu^T * q_dot  (Landau-Lifshitz structure in chart form)
        FA = self.wA * self.Ms * ufl.dot(ufl.dot(qA_dot_expr, self.L_mu_gamma_chart(qA_new, +1.0).T), vA) * self.dv
        FB = self.wB * self.Ms * ufl.dot(ufl.dot(qB_dot_expr, self.L_mu_gamma_chart(qB_new, -1.0).T), vB) * self.dv

        # Anisotropy/Zeeman: Gateaux derivatives of energy densities w.r.t. mixed unknowns
        if "uniaxial_anisotropy" in self.energy_terms:
            magA = self.stereo_to_cart_chart(qA_new, +1.0)
            magB = self.stereo_to_cart_chart(qB_new, -1.0)
            H_ani_A = self.Q1 * self.K * (1 - ufl.dot(self.axis_uni_anisotropy, magA) ** 2)
            H_ani_B = self.Q1 * self.K * (1 - ufl.dot(self.axis_uni_anisotropy, magB) ** 2)
            FA += ufl.derivative(0.5 * self.wA * H_ani_A * self.dv, self.u_mixed, self.v_mixed)
            FB += ufl.derivative(0.5 * self.wB * H_ani_B * self.dv, self.u_mixed, self.v_mixed)

        if "cubic_anisotropy" in self.energy_terms:
            magA = self.stereo_to_cart_chart(qA_new, +1.0)
            magB = self.stereo_to_cart_chart(qB_new, -1.0)
            f1A = magA[0] ** 2 * magA[1] ** 2 + magA[1] ** 2 * magA[2] ** 2 + magA[2] ** 2 * magA[0] ** 2
            f2A = magA[0] ** 2 * magA[1] ** 2 * magA[2] ** 2
            f1B = magB[0] ** 2 * magB[1] ** 2 + magB[1] ** 2 * magB[2] ** 2 + magB[2] ** 2 * magB[0] ** 2
            f2B = magB[0] ** 2 * magB[1] ** 2 * magB[2] ** 2
            H_cA = self.Q1 * self.K * (f1A + self.K2K1_ratio * f2A)
            H_cB = self.Q1 * self.K * (f1B + self.K2K1_ratio * f2B)
            FA += ufl.derivative(0.5 * self.wA * H_cA * self.dv, self.u_mixed, self.v_mixed)
            FB += ufl.derivative(0.5 * self.wB * H_cB * self.dv, self.u_mixed, self.v_mixed)

        if "zeeman" in self.energy_terms:
            magA = self.stereo_to_cart_chart(qA_new, +1.0)
            magB = self.stereo_to_cart_chart(qB_new, -1.0)
            Z_A = -self.Ms * ufl.inner(magA, self.h_zee)
            Z_B = -self.Ms * ufl.inner(magB, self.h_zee)
            FA += ufl.derivative(self.wA * Z_A * self.dv, self.u_mixed, self.v_mixed)
            FB += ufl.derivative(self.wB * Z_B * self.dv, self.u_mixed, self.v_mixed)

        if "demag" in self.energy_terms:
            FA += -self.wA * self.demag_ramp_constant * self.Ms * ufl.dot(
                ufl.dot(self.A_mu_i_chart(qA_new, +1.0).T, h_demag_sym), vA
            ) * self.dv
            FB += -self.wB * self.demag_ramp_constant * self.Ms * ufl.dot(
                ufl.dot(self.A_mu_i_chart(qB_new, -1.0).T, h_demag_sym), vB
            ) * self.dv

        if "exchange" in self.energy_terms:
            gA = ufl.grad(qA_new)
            gB = ufl.grad(qB_new)
            FA += self.wA * (
                0.5 * self.Aex * ufl.dot(self.Q_mu(qA_new, gA), vA)
                + self.Aex * ufl.tr(ufl.dot(ufl.dot(self.P_mu_gamma(qA_new), gA), ufl.grad(vA).T))
            ) * self.dv
            FB += self.wB * (
                0.5 * self.Aex * ufl.dot(self.Q_mu(qB_new, gB), vB)
                + self.Aex * ufl.tr(ufl.dot(ufl.dot(self.P_mu_gamma(qB_new), gB), ufl.grad(vB).T))
            ) * self.dv
            if not self.NeumannBC:
                FA += -self.wA * self.Aex * ufl.dot(
                    ufl.dot(self.P_mu_gamma(qA_new), ufl.dot(gA, self.normal)), vA
                ) * self.ds
                FB += -self.wB * self.Aex * ufl.dot(
                    ufl.dot(self.P_mu_gamma(qB_new), ufl.dot(gB, self.normal)), vB
                ) * self.ds

        mA = self.stereo_to_cart_chart(qA_new, +1.0)
        mB = self.stereo_to_cart_chart(qB_new, -1.0)

        # Div-B=0-style constraint: phi solves magnetostatics for weighted m_blend
        if "demag" in self.energy_terms:
            w = self.wA + self.wB + PETSc.ScalarType(1e-12)
            self._mag_expr_for_phi = (self.wA * mA + self.wB * mB) / w
            F_phi_expr = self.demag_ramp_constant * ufl.dot(
                ufl.grad(phi_sol_sym - self.phi_computed_mixed), ufl.grad(v_phi)
            ) * self.dv
        else:
            self._mag_expr_for_phi = self.stereo_to_cart_chart(qA_new, +1.0)
            F_phi_expr = fem.Constant(self.domain, PETSc.ScalarType(0.0)) * v_phi * self.dv

        F_expr = FA + FB + F_phi_expr
        self._snes_problem = self.NonlinearSNESProblem_qAqB(self, F_expr)
        self.solver_dynamic.setFunction(self._snes_problem.F, self.b)
        self.solver_dynamic.setJacobian(self._snes_problem.J, self.J, self.J)

    class NonlinearSNESProblem_qAqB:
        """SNES nonlinear problem wrapper for qA/qB/phi system."""

        def __init__(self, sim, F_expr):
            self.sim = sim
            self.F_expr = F_expr
            jit_opts = self.sim._jit_options if len(getattr(self.sim, "_jit_options", {})) > 0 else None
            self.F_form = fem.form(self.F_expr, jit_options=jit_opts)
            self.J_expr = ufl.derivative(self.F_expr, self.sim.u_mixed, self.sim.du_mixed)
            self.J_form = fem.form(self.J_expr, jit_options=jit_opts)

        def F(self, snes, x, F):
            # Vec → Function (not assign(u, x), which overwrites the SNES trial).
            fem.petsc.assign(x, self.sim.u_mixed)
            self.sim.u_mixed.x.scatter_forward()
            with F.localForm() as f_local:
                f_local.set(0.0)
            fem.petsc.assemble_vector(F, self.F_form)
            if len(self.sim.bcs) > 0:
                fem.petsc.apply_lifting(F, [self.J_form], bcs=[self.sim.bcs], x0=[x], alpha=-1.0)
            F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
            fem.petsc.set_bc(F, self.sim.bcs, x, -1.0)
            F.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        def J(self, snes, x, J, P):
            fem.petsc.assign(x, self.sim.u_mixed)
            self.sim.u_mixed.x.scatter_forward()
            J.zeroEntries()
            fem.petsc.assemble_matrix(J, self.J_form, self.sim.bcs)
            J.assemble()

    def solve_weak_form(self, max_outer_override=None):
        """
        Advance one timestep: BDF shift, atlas update, outer split, and SNES on ``(q_A,q_B,phi)``.

        Called from :meth:`~Zmag_bins.Zmag_dynamic_stereo_base.Z_sim_dynamic_stereo_base.solve_problem`
        inside each retry attempt. Residual/thermo/closure gates run in the base class after
        this returns.

        Parameters
        ----------
        max_outer_override : int, optional
            Cap operator-split iterations.

        Returns
        -------
        dict
            Defects and convergence metadata (``iters``, ``q_defect``, ``phi_defect``, …).
        """
        # Shift BDF history to (n-1, n-2) using current accepted q at step start.
        self._shift_bdf_history_for_step()

        self._update_bdf_coefficients()

        if "demag" in self.energy_terms:
            ramp_val = self.demag_ramp_factor if self.demag_ramp_factor is not None else 1.0
            self.demag_ramp_constant.value = ramp_val

        if not hasattr(self, "_snes_problem"):
            self._build_snes_forms()

        max_outer = (
            max(1, int(max_outer_override))
            if max_outer_override is not None
            else max(1, int(self.outer_split_max_iter))
        )
        if self.outer_split_anderson_enable:
            # Need at least 3 fixed-point stages for Anderson to have an effect.
            max_outer = max(max_outer, 3)
        min_outer_to_accept = (
            max(1, int(self.outer_split_min_iter_with_accel))
            if self.outer_split_anderson_enable
            else 1
        )
        outer_q_defect = 0.0
        outer_q_defect_abs = 0.0
        outer_phi_defect = 0.0
        outer_phi_defect_abs = 0.0
        n_q_owned = self.V_q.dofmap.index_map.size_local * self.V_q.dofmap.index_map_bs
        n_phi_owned = self.V_phi.dofmap.index_map.size_local * self.V_phi.dofmap.index_map_bs
        x_hist = []
        g_hist = []
        anderson_used = 0

        for outer_it in range(max_outer):
            # Rebuild smooth atlas weights from current q state at each outer stage.
            self._atlas_update_masks_from_qA()
            self._atlas_masks_step_last = self.step_now

            # Freeze demag source for this Newton stage.
            if "demag" in self.energy_terms:
                self._refresh_demag_state_from_current_q()

            qA_outer_before = self.qA.x.array[:n_q_owned].copy()
            qB_outer_before = self.qB.x.array[:n_q_owned].copy()
            phi_outer_ref = None
            if "demag" in self.energy_terms:
                phi_outer_ref = self.phi_computed_mixed.x.array[:n_phi_owned].copy()

            uA = self.u_mixed.sub(0)
            uB = self.u_mixed.sub(1)
            uA.interpolate(self.qA)
            uB.interpolate(self.qB)
            self.u_mixed.x.scatter_forward()

            self.solver_dynamic.solve(None, self.u_mixed.x.petsc_vec)
            self.u_mixed.x.scatter_forward()

            qA_new_mixed = self.u_mixed.sub(0)
            qB_new_mixed = self.u_mixed.sub(1)
            self.qA.interpolate(qA_new_mixed)
            self.qB.interpolate(qB_new_mixed)
            self.qA.x.scatter_forward()
            self.qB.x.scatter_forward()

            self._update_q_dot_from_bdf_history()

            if "demag" in self.energy_terms:
                phi_sol_func = self.u_mixed.sub(2)
                self.h_deg.interpolate(
                    fem.Expression(-ufl.grad(phi_sol_func), self.V_h.element.interpolation_points)
                )
                self.h_deg.x.scatter_forward()

            # Outer fixed-point defects.
            dqA_local = self.qA.x.array[:n_q_owned] - qA_outer_before
            dqB_local = self.qB.x.array[:n_q_owned] - qB_outer_before
            q_before_local = qA_outer_before * qA_outer_before + qB_outer_before * qB_outer_before
            q_def_local = dqA_local * dqA_local + dqB_local * dqB_local
            num_local = float(np.sum(q_def_local))
            den_local = float(np.sum(q_before_local))
            num_global = self.domain.comm.allreduce(num_local, op=MPI.SUM)
            den_global = self.domain.comm.allreduce(den_local, op=MPI.SUM)
            outer_q_defect = float(np.sqrt(num_global / max(den_global, 1e-30)))
            n_q_global = self.domain.comm.allreduce(int(n_q_owned), op=MPI.SUM)
            outer_q_defect_abs = float(np.sqrt(num_global / max(float(n_q_global), 1.0)))

            outer_phi_defect = 0.0
            outer_phi_defect_abs = 0.0
            if "demag" in self.energy_terms and phi_outer_ref is not None:
                # Fixed-point defect for the split operator:
                # compare demag closures phi*(q^{k+1}) vs phi*(q^{k}).
                # Remove DOF-mean from Δφ so a residual Neumann gauge does not
                # reject steps when H_d=-∇φ is already consistent (Demag energy OK).
                self._refresh_demag_state_from_current_q()
                phi_new_ref = self.phi_computed_mixed.x.array[:n_phi_owned]
                dphi = np.asarray(phi_new_ref - phi_outer_ref, dtype=np.float64)
                n_phi_global = self.domain.comm.allreduce(int(n_phi_owned), op=MPI.SUM)
                mean_dphi = self.domain.comm.allreduce(float(np.sum(dphi)), op=MPI.SUM) / max(
                    float(n_phi_global), 1.0
                )
                dphi -= mean_dphi
                phi_ref_zm = np.asarray(phi_outer_ref, dtype=np.float64)
                mean_ref = self.domain.comm.allreduce(float(np.sum(phi_ref_zm)), op=MPI.SUM) / max(
                    float(n_phi_global), 1.0
                )
                phi_ref_zm = phi_ref_zm - mean_ref
                num_phi_local = float(np.dot(dphi, dphi))
                den_phi_local = float(np.dot(phi_ref_zm, phi_ref_zm))
                num_phi_global = self.domain.comm.allreduce(num_phi_local, op=MPI.SUM)
                den_phi_global = self.domain.comm.allreduce(den_phi_local, op=MPI.SUM)
                outer_phi_defect = float(np.sqrt(num_phi_global / max(den_phi_global, 1e-30)))
                outer_phi_defect_abs = float(np.sqrt(num_phi_global / max(float(n_phi_global), 1.0)))

            q_ok = (outer_q_defect <= self.outer_split_tol_q) or (
                outer_q_defect_abs <= self.outer_split_tol_q_abs
            )
            phi_ok = (outer_phi_defect <= self.outer_split_tol_phi) or (
                outer_phi_defect_abs <= self.outer_split_tol_phi_abs
            )
            if q_ok and phi_ok and int(outer_it + 1) >= min_outer_to_accept:
                break

            # Anderson acceleration on fixed-point map (qA,qB): x_k -> g(x_k).
            if self.outer_split_anderson_enable and outer_it >= int(self.outer_split_anderson_start_iter):
                xk = np.empty(2 * n_q_owned, dtype=np.float64)
                xk[:n_q_owned] = qA_outer_before
                xk[n_q_owned:] = qB_outer_before
                gk = self._outer_split_pack_state_owned(n_q_owned)
                x_hist.append(xk)
                g_hist.append(gk)
                mdepth = max(2, int(self.outer_split_anderson_depth))
                if len(x_hist) > mdepth:
                    x_hist = x_hist[-mdepth:]
                    g_hist = g_hist[-mdepth:]
                if len(x_hist) >= 2:
                    x_acc = self._outer_split_anderson_candidate(x_hist, g_hist)
                    omega = float(min(max(self.outer_split_anderson_damping, 0.0), 1.0))
                    x_mix = (1.0 - omega) * gk + omega * x_acc
                    if np.all(np.isfinite(x_mix)):
                        self._outer_split_unpack_state_owned(x_mix, n_q_owned)
                        anderson_used += 1

        outer_iters = int(outer_it + 1)
        q_ok = (outer_q_defect <= self.outer_split_tol_q) or (
            outer_q_defect_abs <= self.outer_split_tol_q_abs
        )
        phi_ok = (outer_phi_defect <= self.outer_split_tol_phi) or (
            outer_phi_defect_abs <= self.outer_split_tol_phi_abs
        )
        outer_converged = bool(q_ok and phi_ok)
        self.state_monitor["last_outer_split_iters"] = outer_iters
        self.state_monitor["last_outer_split_q_defect"] = float(outer_q_defect)
        self.state_monitor["last_outer_split_q_defect_abs"] = float(outer_q_defect_abs)
        self.state_monitor["last_outer_split_phi_defect"] = float(outer_phi_defect)
        self.state_monitor["last_outer_split_phi_defect_abs"] = float(outer_phi_defect_abs)
        self.state_monitor["last_outer_split_anderson_used"] = int(anderson_used)
        self.state_monitor["last_outer_split_converged"] = outer_converged
        return {
            "iters": outer_iters,
            "q_defect": float(outer_q_defect),
            "q_defect_abs": float(outer_q_defect_abs),
            "phi_defect": float(outer_phi_defect),
            "phi_defect_abs": float(outer_phi_defect_abs),
            "anderson_used": int(anderson_used),
            "converged": outer_converged,
        }

    def solve_magnetostatics(self, step):
        """
        Restart hook: optionally warm-start demag from checkpoint, then reconcile.

        Parameters
        ----------
        step : int
            Current step; triggers reload when ``step == step_start`` on restart.

        Notes
        -----
        Checkpoint ``phi`` matches the writer's BEM/partition. After remesh the
        live BEM differs, so any loaded ``phi`` is only a warm start and is
        immediately re-equilibrated via :meth:`_refresh_demag_state_from_current_q`.
        When ``init_dynamic`` already reconciled after atlas restore, this hook
        skips the redundant load.
        """
        if step == self.step_start and not self.first_run:
            checkpoint_dir = self.path_dirs["checkpoint_dir"]
            loading_dir = self.path_dirs["loading_dir"]
            if "demag" in self.energy_terms:
                ckpt_bp = loading_dir + "/" + f"checkpoint_timestep_{self.checkpoint_step}.bp"
                phi_already_restored = (
                    getattr(self, "_demag_phi_checkpoint_restored", False)
                    and getattr(self, "_demag_phi_checkpoint_base", None)
                    == _checkpoint_base_path(ckpt_bp)
                )
                if not phi_already_restored:
                    phi0, h_d0 = self.read_checkpoint_demag(ckpt_bp)
                    phi_init = self.u_mixed.sub(2)
                    phi_init.interpolate(phi0)
                    phi_init.x.scatter_forward()
                    self.u_mixed.x.scatter_forward()
                    self.h_deg.interpolate(h_d0)
                    self.h_deg.x.scatter_forward()
                    # Do not keep writer-partition phi; re-equilibrate with live BEM.
                    self._refresh_demag_state_from_current_q()
            self.write_xdmf_checkpoint("mag", checkpoint_dir + "/checkpoint_mag_start.xdmf")

    def _refresh_demag_state_from_current_q(self):
        """
        Recompute demag potential/state from the current accepted atlas state.

        Used after post-accept atlas closure (where qA/qB may change outside
        Newton) and after checkpoint restart when the live BEM differs from the
        writer's. Keeps ``(qA,qB,phi)`` consistent for the next step, thermo
        baseline, diagnostics, and output.
        """
        if "demag" not in self.energy_terms:
            return
        if not hasattr(self, "mag_func_mixed") or not hasattr(self, "phi_computed_mixed"):
            return

        # Blended atlas magnetization (same convention as weak-form demag RHS)
        self._interpolate_m_cart_numpy(target=self.mag_func_mixed)
        self.demag_helper.compute_phi(self.mag_func_mixed)
        self.demag_helper.phi.x.scatter_forward()

        # Keep both mixed phi and phi_computed_mixed aligned to the refreshed
        # magnetostatic solution. Source is already a Function → interpolate
        # directly (same pattern as init); Expression is only for UFL forms.
        phi_mixed = self.u_mixed.sub(2)
        phi_mixed.interpolate(self.demag_helper.phi)
        phi_mixed.x.scatter_forward()
        self.u_mixed.x.scatter_forward()
        self.phi_computed_mixed.interpolate(self.demag_helper.phi)
        self.phi_computed_mixed.x.scatter_forward()

        self.h_deg.interpolate(
            fem.Expression(-ufl.grad(phi_mixed), self.V_h.element.interpolation_points)
        )
        self.h_deg.x.scatter_forward()

    def open_output(self, target: str, filename: str):
        """
        Open BP (VTX) and optional XDMF writers for a named output field.

        Parameters
        ----------
        target : str
            One of ``mag``, ``qA``, ``qB``, ``phi``, ``demag``, ``heff``,
            atlas masks/weights, etc. (see FEATURE_MATRIX.md).
        filename : str
            Base path for ``.bp`` output files.
        """
        if target == "mag":
            self.vtx_mag = io.VTXWriter(self.domain.comm, filename + ".bp", [self.m_cart], engine="BP4")
        elif target == "phi":
            self._phi_out = fem.Function(self.V_phi, name="phi")
            self.vtx_phi = io.VTXWriter(self.domain.comm, filename + ".bp", [self._phi_out], engine="BP4")
        elif target == "phi1_boundary":
            if "demag" not in self.energy_terms:
                raise Exception("phi1_boundary requires demag energy term")
            self._phi1b_out = fem.Function(self.demag_helper.V_phi, name="phi1_boundary")
            self.vtx_phi1b = io.VTXWriter(self.domain.comm, filename + ".bp", [self._phi1b_out], engine="BP4")
        elif target == "qA":
            self.vtx_qA = io.VTXWriter(self.domain.comm, filename + ".bp", [self.qA], engine="BP4")
        elif target == "qB":
            self.vtx_qB = io.VTXWriter(self.domain.comm, filename + ".bp", [self.qB], engine="BP4")
        elif target == "chiA":
            self.vtx_chiA = io.VTXWriter(self.domain.comm, filename + ".bp", [self.chiA], engine="BP4")
        elif target == "chiB":
            self.vtx_chiB = io.VTXWriter(self.domain.comm, filename + ".bp", [self.chiB], engine="BP4")
        elif target == "chiAB":
            self.vtx_chiAB = io.VTXWriter(self.domain.comm, filename + ".bp", [self.chiAB], engine="BP4")
        elif target == "wA":
            self.vtx_wA = io.VTXWriter(self.domain.comm, filename + ".bp", [self.wA], engine="BP4")
        elif target == "wB":
            self.vtx_wB = io.VTXWriter(self.domain.comm, filename + ".bp", [self.wB], engine="BP4")
        elif target == "norm_q_dot":
            if not hasattr(self, "V_norm_q_dot"):
                self.V_norm_q_dot = fem.functionspace(self.domain, ("CG", 1))
                self.norm_q_dot = fem.Function(self.V_norm_q_dot)
            self.vtx_qdot = io.VTXWriter(self.domain.comm, filename + ".bp", [self.norm_q_dot], engine="BP4")
        elif target == "demag":
            self.vtx_demag = io.VTXWriter(self.domain.comm, filename + ".bp", [self.h_deg], engine="BP4")
        elif target == "heff":
            if not hasattr(self, "heff"):
                self.heff = fem.Function(self.V_h, name="effective_field")
            self.vtx_heff = io.VTXWriter(self.domain.comm, filename + ".bp", [self.heff], engine="BP4")
        elif target == "M_scaled":
            self.vtx_M_scaled = io.VTXWriter(self.domain.comm, filename + ".bp", [self.M_scaled], engine="BP4")
        elif target == "Hd_scaled":
            self.vtx_Hd_scaled = io.VTXWriter(self.domain.comm, filename + ".bp", [self.Hd_scaled], engine="BP4")
        elif target == "B_scaled":
            self.vtx_B_scaled = io.VTXWriter(self.domain.comm, filename + ".bp", [self.B_scaled], engine="BP4")
        else:
            raise Exception(
                "Options: mag, demag, heff, phi, phi1_boundary, qA, qB, chiA, chiB, chiAB, wA, wB, "
                "norm_q_dot, M_scaled, Hd_scaled, B_scaled"
            )
        self._open_xdmf_time_writer(target, filename)

    def close_output(self, target: str):
        """Close VTX/XDMF writers opened by :meth:`open_output` for ``target``."""
        target_map = {
            "mag": "vtx_mag",
            "phi": "vtx_phi",
            "phi1_boundary": "vtx_phi1b",
            "qA": "vtx_qA",
            "qB": "vtx_qB",
            "chiA": "vtx_chiA",
            "chiB": "vtx_chiB",
            "chiAB": "vtx_chiAB",
            "wA": "vtx_wA",
            "wB": "vtx_wB",
            "norm_q_dot": "vtx_qdot",
            "demag": "vtx_demag",
            "heff": "vtx_heff",
            "M_scaled": "vtx_M_scaled",
            "Hd_scaled": "vtx_Hd_scaled",
            "B_scaled": "vtx_B_scaled",
        }
        attr = target_map.get(target)
        if attr and hasattr(self, attr):
            getattr(self, attr).close()
            self._close_xdmf_time_writer(target)
            return
        raise Exception(
            "Options: mag, demag, heff, phi, phi1_boundary, qA, qB, chiA, chiB, chiAB, wA, wB, "
            "norm_q_dot, M_scaled, Hd_scaled, B_scaled"
        )

    def _fill_phi1_boundary_field(self, target_field: fem.Function):
        """Assemble and fill `target_field` with phi1 on the demag boundary.

        This centralizes the boundary solve/BEM map used by both:
        - streaming output (`write_output("phi1_boundary")`)
        - checkpoint export (`write_phi1_boundary`)
        """
        if "demag" not in self.energy_terms:
            raise Exception("phi1_boundary requires demag energy term")

        if not hasattr(self, "_phi1b_mag_work"):
            self._phi1b_mag_work = fem.Function(self.V_m, name="phi1b_mag_work")

        self._interpolate_m_cart_numpy(target=self._phi1b_mag_work)
        f_phi0 = self.demag_helper._build_neumann_rhs(self._phi1b_mag_work)
        if not hasattr(self.demag_helper, "_neumann_nullspace"):
            self.demag_helper._neumann_nullspace = PETSc.NullSpace().create(
                constant=True, comm=self.domain.comm
            )
        phi0_problem = fem.petsc.LinearProblem(
            self.demag_helper.a,
            f_phi0,
            bcs=[],
            u=self.demag_helper.phi0,
            petsc_options_prefix="zmag_phi0_bnd_",
            petsc_options=self.demag_helper.neumann_options,
        )
        phi0_problem.A.setNullSpace(self.demag_helper._neumann_nullspace)
        phi0_problem.A.setNearNullSpace(self.demag_helper._neumann_nullspace)
        phi0_func = phi0_problem.solve()
        phi0_func.x.scatter_forward()

        bnd_nodes = self.demag_helper.bnd_nodes
        n_bnd_nodes = self.demag_helper.n_bnd_nodes
        phi0_bnd_local = phi0_func.x.array[bnd_nodes[:n_bnd_nodes]]
        # IMPORTANT: use the same global ordering/distribution as the assembled BEM matrix.
        # This affects only `phi1_boundary` output/checkpoints.
        if isinstance(self.demag_helper.global_n_bnd_nodes, list):
            total_bnd_nodes = int(sum(self.demag_helper.global_n_bnd_nodes))
        else:
            total_bnd_nodes = int(np.sum(self.demag_helper.global_n_bnd_nodes))

        global_phi0_bnd_buffer = self.demag_helper.global_phi0_bnd_buffer
        if global_phi0_bnd_buffer.shape[0] != total_bnd_nodes:
            global_phi0_bnd_buffer = np.zeros(total_bnd_nodes, dtype=np.float64)

        global_phi0_bnd_buffer[:] = 0.0
        self.domain.comm.Allgatherv(
            sendbuf=phi0_bnd_local,
            recvbuf=[
                global_phi0_bnd_buffer,
                self.demag_helper.global_recvcounts,
                self.demag_helper.global_displs,
                MPI.DOUBLE,
            ],
        )

        global_phi1_bnd_buffer = np.dot(self.demag_helper.BEM, global_phi0_bnd_buffer)
        target_field.x.array[:] = 0.0
        target_field.x.array[bnd_nodes[:n_bnd_nodes]] = global_phi1_bnd_buffer
        target_field.x.scatter_forward()

    def write_output(self, target: str):
        """
        Update and write the field bundle for ``target`` at time ``self.t``.

        Populates derived fields (``mag``, ``demag``, atlas masks, etc.) then
        writes to BP/XDMF writers registered for ``target``.
        """
        if target == "mag":
            self._interpolate_m_cart_numpy()
            if hasattr(self, "vtx_mag"):
                self.vtx_mag.write(self.t)
            self._write_xdmf_time_output(target, self.m_cart)
            return

        if target == "phi":
            if not hasattr(self, "_phi_out"):
                self._phi_out = fem.Function(self.V_phi, name="phi")
            phi_mixed = self.u_mixed.sub(2)
            self._phi_out.interpolate(phi_mixed)
            self._phi_out.x.scatter_forward()
            if hasattr(self, "vtx_phi"):
                self.vtx_phi.write(self.t)
            self._write_xdmf_time_output(target, self._phi_out)
            return

        if target == "phi1_boundary":
            if not hasattr(self, "_phi1b_out"):
                self._phi1b_out = fem.Function(self.demag_helper.V_phi, name="phi1_boundary")
            self._fill_phi1_boundary_field(self._phi1b_out)
            if hasattr(self, "vtx_phi1b"):
                self.vtx_phi1b.write(self.t)
            self._write_xdmf_time_output(target, self._phi1b_out)
            return

        if target == "qA":
            self.qA.x.scatter_forward()
            if hasattr(self, "vtx_qA"):
                self.vtx_qA.write(self.t)
            self._write_xdmf_time_output(target, self.qA)
            return

        if target == "qB":
            self.qB.x.scatter_forward()
            if hasattr(self, "vtx_qB"):
                self.vtx_qB.write(self.t)
            self._write_xdmf_time_output(target, self.qB)
            return

        if target == "chiA":
            self.chiA.x.scatter_forward()
            if hasattr(self, "vtx_chiA"):
                self.vtx_chiA.write(self.t)
            self._write_xdmf_time_output(target, self.chiA)
            return

        if target == "chiB":
            self.chiB.x.scatter_forward()
            if hasattr(self, "vtx_chiB"):
                self.vtx_chiB.write(self.t)
            self._write_xdmf_time_output(target, self.chiB)
            return

        if target == "chiAB":
            self.chiAB.x.scatter_forward()
            if hasattr(self, "vtx_chiAB"):
                self.vtx_chiAB.write(self.t)
            self._write_xdmf_time_output(target, self.chiAB)
            return

        if target == "wA":
            self.wA.x.scatter_forward()
            if hasattr(self, "vtx_wA"):
                self.vtx_wA.write(self.t)
            self._write_xdmf_time_output(target, self.wA)
            return

        if target == "wB":
            self.wB.x.scatter_forward()
            if hasattr(self, "vtx_wB"):
                self.vtx_wB.write(self.t)
            self._write_xdmf_time_output(target, self.wB)
            return

        if target == "norm_q_dot":
            self._update_norm_q_dot_field()
            if hasattr(self, "vtx_qdot"):
                self.vtx_qdot.write(self.t)
            self._write_xdmf_time_output(target, self.norm_q_dot)
            return

        if target == "demag":
            phi_mixed = self.u_mixed.sub(2)
            self.h_deg.interpolate(fem.Expression(-ufl.grad(phi_mixed), self.V_h.element.interpolation_points))
            self.h_deg.x.scatter_forward()
            if hasattr(self, "vtx_demag"):
                self.vtx_demag.write(self.t)
            self._write_xdmf_time_output(target, self.h_deg)
            return

        if target == "heff":
            if not hasattr(self, "heff"):
                self.heff = fem.Function(self.V_h, name="effective_field")
            self.effective_field(target=self.heff)
            if hasattr(self, "vtx_heff"):
                self.vtx_heff.write(self.t)
            self._write_xdmf_time_output(target, self.heff)
            return

        if target == "M_scaled":
            self._interpolate_m_cart_numpy()
            n_own = self.V_m.dofmap.index_map.size_local
            Ms_val = float(self.Ms)
            self.M_scaled.x.array[: n_own * SPIN_DIM] = Ms_val * self.m_cart.x.array[: n_own * SPIN_DIM]
            self.M_scaled.x.scatter_forward()
            if hasattr(self, "vtx_M_scaled"):
                self.vtx_M_scaled.write(self.t)
            self._write_xdmf_time_output(target, self.M_scaled)
            return

        if target == "Hd_scaled":
            phi_mixed = self.u_mixed.sub(2)
            h_deg_expr = -ufl.grad(phi_mixed)
            self.Hd_scaled.interpolate(fem.Expression(self.Ms * h_deg_expr, self.V_scaled.element.interpolation_points))
            self.Hd_scaled.x.scatter_forward()
            if hasattr(self, "vtx_Hd_scaled"):
                self.vtx_Hd_scaled.write(self.t)
            self._write_xdmf_time_output(target, self.Hd_scaled)
            return

        if target == "B_scaled":
            self._interpolate_m_cart_numpy()
            phi_mixed = self.u_mixed.sub(2)
            h_deg_expr = -ufl.grad(phi_mixed)
            h_deg_func = fem.Function(self.V_m)
            h_deg_func.interpolate(fem.Expression(h_deg_expr, self.V_m.element.interpolation_points))
            h_deg_func.x.scatter_forward()
            n_own = self.V_m.dofmap.index_map.size_local
            Ms_val = float(self.Ms)
            self.B_scaled.x.array[: n_own * SPIN_DIM] = Ms_val * (
                h_deg_func.x.array[: n_own * SPIN_DIM] + self.m_cart.x.array[: n_own * SPIN_DIM]
            )
            self.B_scaled.x.scatter_forward()
            if hasattr(self, "vtx_B_scaled"):
                self.vtx_B_scaled.write(self.t)
            self._write_xdmf_time_output(target, self.B_scaled)
            return

        raise Exception(
            "Options: mag, demag, heff, phi, phi1_boundary, qA, qB, chiA, chiB, chiAB, wA, wB, "
            "norm_q_dot, M_scaled, Hd_scaled, B_scaled"
        )

    def write_checkpoint(self, filename):
        """
        Write ADIOS checkpoint plus npz chart histories.

        Parameters
        ----------
        filename : str
            Path ending in ``.bp``; npz companions use the same stem.
        """
        base = filename[:-3] if filename.endswith(".bp") else filename
        import adios4dolfinx as a4dx

        a4dx.write_mesh(filename, self.domain)
        a4dx.write_meshtags(filename, self.domain, self.cell_tags, meshtag_name="cell_tags")
        a4dx.write_meshtags(filename, self.domain, self.facet_tags, meshtag_name="facet_tags")

        save_checkpoint_atlas_state(self, base)
        save_checkpoint_time_state(self, base)

        if "demag" in self.energy_terms:
            phi_dump = fem.Function(self.V_phi, name="phi")
            phi_mixed = self.u_mixed.sub(2)
            phi_dump.interpolate(phi_mixed)
            phi_dump.x.scatter_forward()
            self._save_func_npz(phi_dump, base + "_phi.npz")

    def write_phi1_boundary(self, filename):
        """
        Export boundary potential ``phi1`` (Neumann data + BEM) to a BP checkpoint.

        Parameters
        ----------
        filename : str
        """
        phi1_boundary_func = fem.Function(self.demag_helper.V_phi, name="phi1_boundary")
        self._fill_phi1_boundary_field(phi1_boundary_func)
        a4dx.write_mesh(filename, self.domain)
        a4dx.write_function(filename, phi1_boundary_func)

    def write_xdmf_checkpoint(self, target: str, filename: str):
        """
        Export a single-time XDMF snapshot.

        Parameters
        ----------
        target : str
            ``mag``, ``phi``, ``demag``, ``heff``, ``chart_ind``, ``q_norm``, ``qA``, ``qB``, etc.
        filename : str
            Output ``.xdmf`` path.
        """
        if target == "mag":
            self._interpolate_m_cart_numpy()
            mag_arr = self.m_cart.x.array
            if not np.all(np.isfinite(mag_arr)):
                mag_arr[:] = np.nan_to_num(mag_arr, nan=0.0, posinf=0.0, neginf=0.0)
                self.m_cart.x.scatter_forward()
            with io.XDMFFile(self.domain.comm, filename, "w") as f:
                f.write_mesh(self.domain)
                f.write_function(self.m_cart)
        elif target == "demag":
            assert "demag" in self.energy_terms
            phi_mixed = self.u_mixed.sub(2)
            h_deg_snap = fem.Function(self.V_m, name="demag_field")
            h_deg_snap.interpolate(fem.Expression(-ufl.grad(phi_mixed), self.V_m.element.interpolation_points))
            h_deg_snap.x.scatter_forward()
            with io.XDMFFile(self.domain.comm, filename, "w") as f:
                f.write_mesh(self.domain)
                f.write_function(h_deg_snap)
        elif target == "phi":
            assert "demag" in self.energy_terms
            phi_snap = fem.Function(self.V_phi, name="phi")
            phi_mixed = self.u_mixed.sub(2)
            phi_snap.interpolate(phi_mixed)
            phi_snap.x.scatter_forward()
            with io.XDMFFile(self.domain.comm, filename, "w") as f:
                f.write_mesh(self.domain)
                f.write_function(phi_snap)
        elif target == "heff":
            heff_dg = self.effective_field()
            heff_snap = fem.Function(self.V_m, name="h_eff")
            heff_snap.interpolate(heff_dg)
            heff_snap.x.scatter_forward()
            with io.XDMFFile(self.domain.comm, filename, "w") as f:
                f.write_mesh(self.domain)
                f.write_function(heff_snap)
        elif target == "chart_ind":
            with io.XDMFFile(self.domain.comm, filename, "w") as f:
                f.write_mesh(self.domain)
                f.write_function(self.chart_ind)
        elif target == "q_norm":
            V_s = fem.functionspace(self.domain, ("CG", 1))
            q_norm_snap = fem.Function(V_s, name="q_norm")
            q_norm_snap.interpolate(
                fem.Expression(
                    ufl.sqrt(
                        self.wA * (self.qA[0] ** 2 + self.qA[1] ** 2)
                        + self.wB * (self.qB[0] ** 2 + self.qB[1] ** 2)
                    ),
                    V_s.element.interpolation_points,
                )
            )
            q_norm_snap.x.scatter_forward()
            with io.XDMFFile(self.domain.comm, filename, "w") as f:
                f.write_mesh(self.domain)
                f.write_function(q_norm_snap)
        elif target == "qA":
            qA_snap = fem.Function(self.V_q, name="qA")
            qA_snap.interpolate(self.qA)
            qA_snap.x.scatter_forward()
            with io.XDMFFile(self.domain.comm, filename, "w") as f:
                f.write_mesh(self.domain)
                f.write_function(qA_snap)
        elif target == "qB":
            qB_snap = fem.Function(self.V_q, name="qB")
            qB_snap.interpolate(self.qB)
            qB_snap.x.scatter_forward()
            with io.XDMFFile(self.domain.comm, filename, "w") as f:
                f.write_mesh(self.domain)
                f.write_function(qB_snap)
        elif target == "chiA":
            with io.XDMFFile(self.domain.comm, filename, "w") as f:
                f.write_mesh(self.domain)
                f.write_function(self.chiA)
        elif target == "chiB":
            with io.XDMFFile(self.domain.comm, filename, "w") as f:
                f.write_mesh(self.domain)
                f.write_function(self.chiB)
        elif target == "chiAB":
            with io.XDMFFile(self.domain.comm, filename, "w") as f:
                f.write_mesh(self.domain)
                f.write_function(self.chiAB)
        elif target == "wA":
            with io.XDMFFile(self.domain.comm, filename, "w") as f:
                f.write_mesh(self.domain)
                f.write_function(self.wA)
        elif target == "wB":
            with io.XDMFFile(self.domain.comm, filename, "w") as f:
                f.write_mesh(self.domain)
                f.write_function(self.wB)
        else:
            raise Exception("Options: mag, demag, phi, heff, chart_ind, q_norm, qA, qB, chiA, chiB, chiAB, wA, wB")

    def _restore_demag_phi_from_checkpoint(self, base: str) -> bool:
        """
        Load checkpoint demag potential into mixed state and helper.

        Returns True on success. The loaded ``phi`` matches the checkpoint
        writer's BEM/partition and is treated as a warm start only:
        :meth:`init_dynamic` re-equilibrates with the live BEM before the
        thermo baseline is seeded.
        """
        if "demag" not in self.energy_terms or not hasattr(self, "demag_helper"):
            return False
        try:
            phi_loaded = fem.Function(self.V_phi, name="phi_checkpoint")
            self._load_func_npz(phi_loaded, base + "_phi.npz")
            phi_loaded.x.scatter_forward()
            phi_mixed = self.u_mixed.sub(2)
            phi_mixed.interpolate(phi_loaded)
            phi_mixed.x.scatter_forward()
            self.u_mixed.x.scatter_forward()
            self.demag_helper.phi.x.array[:] = phi_loaded.x.array[:]
            self.demag_helper.phi.x.scatter_forward()
            if hasattr(self, "h_deg"):
                self.h_deg.interpolate(
                    fem.Expression(-ufl.grad(phi_loaded), self.V_h.element.interpolation_points)
                )
                self.h_deg.x.scatter_forward()
            self._demag_phi_checkpoint_base = base
            return True
        except (FileNotFoundError, OSError):
            return False

    def set_magnetostatics(self, interfaces: list[dict], surface_tag: int):
        r"""
        Wire the magnetostatic helper and interface facet measures.

        Parameters
        ----------
        interfaces : list[dict]
            Interface definitions with ``tag`` keys.
        surface_tag : int
            Exterior boundary tag for Dirichlet data on :math:`\varphi`.
        """
        if "zeeman" not in self.energy_terms:
            self.h_zee = ufl.as_vector([0.0, 0.0, 0.0])
        if "demag" in self.energy_terms:
            self.interfaces = interfaces
            self.bnd_mrk = surface_tag
            self.demag_helper = Z_sim_magnetostatic(
                self.domain, self.facet_tags, interfaces, surface_tag, self.dv, self.ds, self.h_zee, self.Ms
            )
            self.boundary_dofs = fem.locate_dofs_topological(
                self.V_phi, self.domain.topology.dim - 1, self.facet_tags.find(surface_tag)
            )
            self.interface_measures = {}
            for interface in interfaces:
                if "tag" not in interface:
                    raise Exception("Provide key tag in the interface dictionary")
                inter_tag = interface["tag"]
                dS_tag = ufl.Measure(
                    "dS", self.domain, subdomain_data=self.facet_tags, subdomain_id=inter_tag, metadata={"quadrature_degree": 4}
                )
                self.interface_measures[inter_tag] = dS_tag

    def set_BEMcpp(self, BEM_dir: str, cpp_backend: str | None = None):
        """
        Assemble the BEM operator for the Laplace subproblem.

        Parameters
        ----------
        BEM_dir : str
            BEM data directory for :class:`bem_assembler.bem_assembler.BEMAssembler`.
        cpp_backend : str, optional
            Optional backend preset.
        """
        if "demag" not in self.energy_terms:
            return
        assembler = BEMAssembler(
            domain=self.domain,
            facet_tags=self.facet_tags,
            surface_marker=self.bnd_mrk,
            verbose=True,
            cpp_backend=cpp_backend,
        )
        t0 = time.time()
        self.BEM = assembler.assemble()
        assembly_time = time.time() - t0
        # Same BEM layout on solver and demag_helper (for matvec + Allgatherv).
        self.n_bnd_nodes = assembler.global_n_bnd_nodes[self.domain.comm.rank]
        self.global_n_bnd_nodes = assembler.global_n_bnd_nodes
        self.bnd_nodes = assembler.bnd_nodes
        self.demag_helper.BEM = self.BEM
        self.demag_helper.n_bnd_nodes = self.n_bnd_nodes
        self.demag_helper.global_n_bnd_nodes = self.global_n_bnd_nodes
        self.demag_helper.bnd_nodes = self.bnd_nodes
        self.demag_helper.initialize_bem_cache()
        all_times = self.domain.comm.gather(assembly_time, root=0)
        if self.domain.comm.rank == 0:
            print_flush(f"Local matrix shape (rank 0): {self.BEM.shape}")
            print_flush(f"Assembly time (rank 0): {assembly_time:.4f} s")
            if all_times:
                print_flush(
                    f"Assembly time (all ranks): "
                    f"avg={np.mean(all_times):.4f}s, "
                    f"min={np.min(all_times):.4f}s, "
                    f"max={np.max(all_times):.4f}s"
                )
        self.domain.comm.Barrier()

    def read_checkpoint_demag(self, filename):
        """
        Restore magnetostatic potential and demagnetizing field from checkpoint.

        Parameters
        ----------
        filename : str

        Returns
        -------
        tuple[dolfinx.fem.Function, dolfinx.fem.Function]
            ``(phi, h_demag)`` with ``h_demag = -grad(phi)``.
        """
        base = filename[:-3] if filename.endswith(".bp") else filename
        ph = fem.Function(self.V_phi, name="phi")
        self._load_func_npz(ph, base + "_phi.npz")
        hd = fem.Function(self.V_h)
        hd.interpolate(fem.Expression(-ufl.grad(ph), self.V_h.element.interpolation_points))
        return ph, hd

    def read_checkpoint_mesh(self, mpi_comm, filename):
        """
        Restore distributed mesh and meshtags from an ADIOS BP file.

        Returns
        -------
        tuple
            ``(domain, boundary_facets, cell_tags, facet_tags)``.
        """
        domn = a4dx.read_mesh(filename, mpi_comm, "BP4", mesh.GhostMode.shared_facet)
        tdim = domn.topology.dim
        fdim = tdim - 1
        domn.topology.create_connectivity(fdim, tdim)
        boundary_facets = mesh.exterior_facet_indices(domn.topology)
        cell_tags = a4dx.read_meshtags(filename, domn, meshtag_name="cell_tags")
        facet_tags = a4dx.read_meshtags(filename, domn, meshtag_name="facet_tags")
        return domn, boundary_facets, cell_tags, facet_tags

    def read_checkpoint_time(self, run_dir, filename):
        """
        Parse energies log and locate latest ``checkpoint_timestep_*.bp``.

        Returns
        -------
        tuple[int, float, float]
            ``(step_start, t_start, dt)``.
        """
        comm = self.domain.comm if hasattr(self, "domain") else MPI.COMM_WORLD
        loaded_data = np.genfromtxt(fname=filename)
        if np.size(loaded_data) == 0:
            raise RuntimeError(f"Checkpoint energies file is empty: {filename}")
        loaded_data = np.atleast_2d(loaded_data)
        t_start = float(loaded_data[-1, 0])
        delta_t_start = float(loaded_data[-1, 1])
        if comm.rank == 0:
            try:
                filenames = os.listdir(run_dir)
                checkpoints = []
                for fn in filenames:
                    match = re.match(r"checkpoint_timestep_(\d+)\.bp$", fn)
                    if match:
                        checkpoints.append(int(match.group(1)))
                if not checkpoints:
                    status = (False, f"No checkpoint files found in {run_dir}", -1)
                else:
                    status = (True, "", max(checkpoints))
            except Exception as exc:
                status = (False, str(exc), -1)
        else:
            status = None
        ok, err_msg, timestep_start = comm.bcast(status, root=0)
        if not ok:
            raise RuntimeError(f"Failed to read checkpoint timestep from '{run_dir}': {err_msg}")
        return timestep_start, t_start, delta_t_start

