.. _configuration:

Configuration reference
=======================

Stereographic dynamic solvers are configured through Python dictionaries passed at
construction and initialization. Shared validation and defaults live in
:mod:`Zmag_bins.Zmag_dynamic_stereo_base`; model-specific keys are read by
:mod:`Zmag_bins.Zmag_dynamic_stereo_demag` and
:mod:`Zmag_bins.Zmag_dynamic_stereo_nodemag`.

The sections below follow the usual setup order: constructor options, time-integration
options, material and field data, magnetostatics (``demag`` only), then outputs and
checkpoints.


Constructor options
-------------------

Passed as ``args_init_sim`` to the solver constructor (or to
:func:`~Zmag_bins.Zmag_dynamic_stereo_solvers.create_stereo_solver`).

Required keys
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Key
     - Description
   * - ``dim``
     - Spatial dimension (typically ``3``).
   * - ``alpha``
     - Gilbert damping parameter in the chart matrix :math:`\mathcal{L}_\mu` (:doc:`theory_dynamics`).
   * - ``energy_terms``
     - Active energy contributions (see :ref:`config-energy-terms`).
   * - ``NeumannBC``
     - If ``True``, omit exchange boundary integrals (natural / Brown-type flux).
   * - ``printout``
     - Enable rank-0 diagnostic printing.
   * - ``path_dirs``
     - ``results_dir``, ``checkpoint_dir``; optional ``for_vacuum_dir`` for vacuum meshes.

Optional keys
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Key
     - Default / role
   * - ``projection_pole``
     - Unit vector for stereographic pole (default ``(0, -1, 0)``).
   * - ``sigma2_floor``
     - Lower bound on :math:`\sigma^2` in :math:`\mathcal{L}_\mu` for numerical stability (default ``0.25``).

.. _config-energy-terms:

Energy terms
~~~~~~~~~~~~

Values of ``energy_terms`` and which solver variants accept them:

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * - Term
     - ``demag`` model
     - ``nodemag`` model
   * - ``demag``
     - **required**
     - not allowed
   * - ``zeeman``
     - optional
     - optional
   * - ``exchange``
     - optional
     - optional
   * - ``uniaxial_anisotropy``
     - optional
     - optional
   * - ``cubic_anisotropy``
     - optional
     - optional


Time-integration options
------------------------

Passed as ``args_init_dyn`` to ``init_dynamic``.

Required keys
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Key
     - Description
   * - ``t_start``
     - Initial time :math:`t^\ast`.
   * - ``dt``
     - Initial non-dimensional timestep :math:`\Delta t^\ast`.
   * - ``step_start``
     - First step index (use the checkpoint value on restart).
   * - ``step_end``
     - Last step index (inclusive upper bound in the driver loop).
   * - ``first_run``
     - ``True`` for a new run; ``False`` when restarting from a checkpoint.
   * - ``checkpoint_every``
     - Write a checkpoint every this many accepted steps.
   * - ``write_target``
     - List of output field names (see :ref:`write-targets`).
   * - ``cfl_settings``
     - Adaptive timestep control (see :ref:`config-cfl-settings`).

.. _config-cfl-settings:

Adaptive timestep (``cfl_settings``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Sub-dictionary for the embedded BDF error estimate, CFL safety cap, and thermodynamic
retry budget. Keys omitted here are filled from
:data:`~Zmag_bins.Zmag_dynamic_stereo_base.DEFAULT_CFL_SETTINGS`.

.. list-table::
   :header-rows: 1
   :widths: 28 18 54

   * - Key
     - Default
     - Role
   * - ``CFL``
     - ``0.5``
     - Target CFL number for the precession limit.
   * - ``dt_min``, ``dt_max``
     - ``1e-7``, ``1e-2``
     - Hard bounds on :math:`\Delta t^\ast`.
   * - ``smooth_factor``
     - ``0.6``
     - Smoothing of timestep changes.
   * - ``max_growth``, ``min_shrink``
     - ``1.2``, ``0.5``
     - Maximum growth and minimum shrink factors per step.
   * - ``error_rel_tol``, ``error_abs_tol``, ``error_headroom``
     - ``1e-2``, ``1e-4``, ``0.90``
     - Tolerances for the embedded BDF temporal error estimate that drives
       :math:`\Delta t_{\mathrm{error}}` (see :doc:`theory_algorithms`).
   * - ``max_retry_thermo``
     - ``20``
     - Retries with thermodynamic backtracking before cutting ``dt``.

Optional keys
~~~~~~~~~~~~~

Solver and iteration control
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Key
     - Role
   * - ``max_retry``
     - Retries per timestep on SNES failure (default ``5``).
   * - ``time_order``
     - Target BDF order (``1`` or ``2``).
   * - ``use_mf``
     - Matrix-free SNES Jacobian when supported.
   * - ``jit_timeout_seconds``, ``jit_cache_dir``
     - FFCx JIT control (a per-rank cache is recommended on clusters).
   * - ``convergence_settings``
     - Steady-state stopping: ``physical_tol_mdot_rms``, ``physical_tol_edot``,
       stagnation counters, and related keys.

Atlas and chart control
^^^^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Key
     - Role
   * - ``chart_settings``
     - ``hysteresis_tau``, ``atlas_overlap_tau`` for DG0 masks (overridden by the
       top-level ``chart_hysteresis_tau`` when that key is set).
   * - ``chart_hysteresis_tau``
     - Hysteresis width for atlas chart switching (default from ``chart_settings``,
       otherwise ``0``).
   * - ``atlas_weight_relaxation``
     - Relaxation factor in ``[0, 1]`` for smooth ``wA`` / ``wB`` updates.
   * - ``atlas_qmax_guard``
     - Hard cap on the effective chart radius.

Continuation ramps
^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Key
     - Role
   * - ``demag_ramp_N``, ``demag_start_fraction``
     - Demagnetizing-field continuation (``demag`` model only).
   * - ``zeeman_ramp_N``, ``zeeman_start_fraction``
     - Zeeman-field continuation.

Checkpoint snapshots
^^^^^^^^^^^^^^^^^^^^

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Key
     - Role
   * - ``xdmf_checkpoint_targets``
     - List of XDMF snapshot names written on checkpoint steps (for example
       ``mag``, ``chart_ind``, ``q_norm``; ``demag`` also supports ``phi``,
       ``demag``, ``heff``, ``qA``, ``qB``, ``chiA``, ``chiB``).

Acceptance and outer-split gates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These keys control the algebraic, thermodynamic, and (for ``demag``) outer-split
acceptance logic. See :doc:`theory_algorithms` for the full workflow.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Key
     - Role
   * - ``thermo_use_split_energy``
     - If ``True`` (default), the thermo gate uses chart-weighted
       :math:`\mathcal{E}_{\mathrm{sol}}` from ``compute_energy``; if ``False``,
       the blended Cartesian stored energy is used (see :doc:`theory_algorithms`).
   * - ``thermo_enable_physical_guard``
     - Auxiliary Cartesian energy-spike detector (default ``True``).
   * - ``thermo_defect_ema_factor``, ``thermo_defect_ema_warmup_steps``
     - EMA budget on past thermo defects.
   * - ``thermo_physical_guard_ema_factor``, ``thermo_physical_guard_*_floor``
     - Adaptive floor for the physical guard.
   * - ``thermo_backtracking_enable``, ``thermo_backtracking_max_iter``, ``thermo_backtracking_shrink``, ``thermo_backtracking_min_lambda``
     - State-space line search before cutting ``dt`` on thermo failure.
   * - ``thermo_demag_energy_rel_floor``
     - Relative energy floor in the thermo budget (demag problems).
   * - ``nonlinear_residual_max_accept``, ``nonlinear_residual_ema_factor``, ``nonlinear_residual_gate_warmup_steps``, ``nonlinear_residual_gate_floor``, ``nonlinear_residual_dt_relax_max``
     - Algebraic residual gate after SNES.
   * - ``be_recovery_steps_on_thermo``
     - Force BDF1 for this many steps after thermo-related events.
   * - ``enforce_outer_split_acceptance``
     - Require outer-split :math:`(q,\varphi)` fixed-point convergence (``demag``).
   * - ``outer_split_max_iter``, ``outer_split_tol_q``, ``outer_split_tol_phi``, ``outer_split_tol_q_abs``, ``outer_split_tol_phi_abs``
     - Outer iteration cap and defect tolerances (``demag``; includes the
       :math:`\varphi` refresh).
   * - ``outer_split_anderson_enable``, ``outer_split_anderson_depth``, ``outer_split_anderson_damping``
     - Anderson acceleration on the outer split (``demag``).
   * - ``enable_thermo_diagnostics``, ``thermo_diag_log_accepted``
     - Verbose thermo logging.


Materials
---------

Passed as a list to ``set_materials``. Each entry describes one **cell-tag** region;
fields are interpolated to DG0.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Key
     - Description
   * - ``tag``
     - Integer cell marker (Gmsh physical group).
   * - ``Ms``
     - :math:`M_{\mathrm s}^\ast = M_{\mathrm{s}i}/M_{\mathrm{s}1}` (reference tag: ``1``).
   * - ``Aex``
     - :math:`A^\ast = A_i/A_1` (reference tag: ``1``).
   * - ``K``
     - :math:`K^\ast = K_{1i}/K_{11}` (reference tag: ``1``).
   * - ``Q1``
     - :math:`2K_{11}/(\mu_0 M_{\mathrm{s}1}^2)` of the **reference** material.
   * - ``K2K1_ratio``
     - :math:`K_{2i}/K_{1i}` for cubic anisotropy.
   * - ``anisotropy_axis``
     - Unit vector ``(x, y, z)`` for uniaxial anisotropy (required if that term is active).

See :doc:`theory_energy` and :doc:`theory_dynamics` for the scaling derivation.


External field
--------------

``set_external_field`` accepts either:

* A length-3 sequence or NumPy array — constant non-dimensional
  :math:`\mathbf{H}_{\mathrm{zee}}^\ast`.
* A coil dictionary with keys ``wire_path``, ``N_turns``, ``current``, ``l_ex``,
  ``Ms_ref``, ``size_scale_fac``, and optional ``write_Hzee_to_file``.


Magnetostatics (``demag`` only)
-------------------------------

``set_magnetostatics(interfaces, surface_tag)`` registers interface facet tags for
magnetostatic measures and the exterior Dirichlet tag for :math:`\varphi`.

``set_BEMcpp(BEM_dir)`` loads the C++ BEM assembler built as described in
:doc:`installation`.


.. _write-targets:

Output targets
--------------

Requested through ``write_target`` in ``args_init_dyn``.

Shared by both models
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Target
     - Content
   * - ``mag``
     - Cartesian :math:`\mathbf{m}` on nodes
   * - ``heff``
     - Effective-field diagnostic
   * - ``q``, ``qA``, ``qB``
     - Chart coordinates
   * - ``chiA``, ``chiB``, ``chiAB``
     - Atlas activity masks (DG0)
   * - ``wA``, ``wB``
     - Atlas weights (DG0)
   * - ``norm_q_dot``
     - Chart-velocity norm diagnostic
   * - ``M_scaled``
     - :math:`M_{\mathrm s}^\ast\,\mathbf{m}` (DG vector)

``demag`` only
~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Target
     - Content
   * - ``demag``
     - Demagnetizing field
   * - ``phi``
     - Total scalar potential
   * - ``phi1_boundary``
     - Boundary contribution :math:`\varphi_1`
   * - ``Hd_scaled``, ``B_scaled``
     - Scaled demagnetizing field and induction

Requesting a demag-only target with ``model="nodemag"`` raises ``ValueError``.


.. _checkpoint-format:

Checkpoints
-----------

Checkpoints are written automatically during ``solve_problem`` and are the
supported way to interrupt and resume a run. Each checkpoint uses three
kinds of files:

* **ADIOS2 BP files** (``.bp``) — distributed mesh and cell/facet tags,
  written through ``adios4dolfinx``. Mesh topology must be stored and reread
  under MPI in the layout DOLFINx expects; ADIOS2 is the FEniCSx mechanism for
  that.
* **NumPy archives** (``.npz``) — chart fields, BDF history, atlas metadata,
  controller state, and (for ``demag``) a warm-start ``phi``. These are dense
  arrays attached to an already reconstructed function space, so a lightweight
  archive is sufficient and easier to inspect than packing every field into the
  ADIOS file.
* **Optional XDMF** — visualization snapshots from
  ``xdmf_checkpoint_targets``. Not used for restart.

A restart needs the BP stem and its ``.npz`` companions for the same timestep.

Writing
~~~~~~~

Every ``checkpoint_every`` accepted steps the solver writes into
``path_dirs["checkpoint_dir"]``:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - File
     - Content
   * - ``checkpoint_timestep_{step}.bp``
     - Mesh and cell/facet tags (ADIOS2).
   * - ``checkpoint_timestep_{step}_qA.npz``, ``_qB.npz``
     - Current chart fields (required for restart).
   * - ``_qA_prev.npz``, ``_qA_prev_prev.npz``, ``_qB_prev.npz``, ``_qB_prev_prev.npz``
     - Optional BDF history. If missing on load, the current chart state is
       repeated so the run can continue at BDF1.
   * - ``_node_chart.npz``, ``_chart_ind.npz``
     - Atlas metadata used to rebuild chart masks and weights.
   * - ``_time_state.npz``
     - Adaptive timestep / BDF / thermo-gate controller state.
   * - ``_phi.npz``
     - Scalar potential (``demag`` only). Used as a warm start; after remesh the
       live BEM generally differs, so ``phi`` is re-equilibrated from the
       restored magnetization during ``init_dynamic``.
   * - ``checkpoint_energies.txt``
     - Energy log used by ``read_checkpoint_time`` to recover ``t_start`` and
       ``dt``.
   * - ``checkpoint_{target}_step_{step}.xdmf``
     - Optional visualization snapshots from ``xdmf_checkpoint_targets``.

Restarting
~~~~~~~~~~

Restart is driven by the user script, not by ``create_stereo_solver`` alone.
Three settings matter:

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Setting
     - Role on restart
   * - ``path_dirs["loading_dir"]``
     - Directory that contains the checkpoint to **load**.
   * - ``path_dirs["checkpoint_dir"]``
     - Directory where **new** checkpoints will be written (may differ from
       ``loading_dir``).
   * - ``first_run=False`` with ``step_start``, ``t_start``, ``dt``
     - ``step_start`` must be the checkpointed step index. The solver then
       begins the time loop at ``step_start + 1``.

Recommended sequence:

1. Create the solver with ``loading_dir`` set to the previous checkpoint
   directory.
2. Call ``read_checkpoint_time`` to recover the latest step, time, and
   timestep from that directory (it selects the newest
   ``checkpoint_timestep_*.bp`` file).
3. Rebuild the mesh from that BP file with ``read_checkpoint_mesh``, then call
   the usual setup methods (``set_mesh``, materials, and for ``demag``
   magnetostatics / BEM).
4. Restore chart and controller state with ``restore_checkpoint_atlas_state``.
5. Call ``init_dynamic`` with ``first_run=False`` and the recovered
   ``step_start`` / ``t_start`` / ``dt``, then ``solve_problem``.

For ``demag``, ``init_dynamic`` re-equilibrates the magnetostatic potential
with the **live** BEM assembled in step 3 before seeding the thermodynamic
baseline.  Restored ``phi`` is a warm start only; see :ref:`example-checkpoints`.

.. code-block:: python

   path_dirs["loading_dir"] = previous_checkpoint_dir
   sim = create_stereo_solver(args_init_sim, model="demag")

   step, t, dt = sim.read_checkpoint_time(
       previous_checkpoint_dir,
       previous_checkpoint_dir + "/checkpoint_energies.txt",
   )
   bp = f"{previous_checkpoint_dir}/checkpoint_timestep_{step}.bp"
   domain, _, cell_tags, facet_tags = sim.read_checkpoint_mesh(comm, bp)

   sim.set_mesh(domain, cell_tags, facet_tags, surface_tag)
   # ... materials, magnetostatics, BEM as usual ...
   sim.restore_checkpoint_atlas_state(bp)

   args_init_dyn.update(first_run=False, step_start=step, t_start=t, dt=dt)
   sim.init_dynamic(args_init_dyn)
   sim.solve_problem()

The full pattern is exercised by ``test_dynamic_checkpoints.py`` at the
repository root. See also :doc:`getting_started` and :doc:`api_common`.
