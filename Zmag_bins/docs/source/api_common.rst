Shared base module
==================

:mod:`Zmag_bins.Zmag_dynamic_stereo_base` is the **shared core** for dynamic
``demag`` and ``nodemag`` solvers.

Architecture
------------

.. code-block:: text

   Zmag_dynamic_stereo_solvers.create_stereo_solver(model=...)
        │
        ├── demag  → Z_sim_dynamic_stereo_demag  ──┐
        └── nodemag → Z_sim_dynamic_stereo_nodemag ─┤ subclass
                                                    ▼
                              Z_sim_dynamic_stereo_base
                              (time loop, atlas, BDF/CFL, gates, writers)

``solve_problem`` code map
~~~~~~~~~~~~~~~~~~~~~~~~~~

The acceptance labels (A)–(F) in :doc:`theory_algorithms` map to private methods on the base
class. One inner-loop attempt runs ``_run_accept_attempt``; failures call
``_handle_step_rejection``; successes call ``_finalize_accepted_step``.

.. list-table::
   :header-rows: 1
   :widths: 10 35 55

   * - Gate
     - Method
     - Role
   * - (A)
     - ``_run_nonlinear_solve``
     - Nonlinear solve / outer split → trial state
   * - (B)
     - ``_validate_algebraic_solve``
     - SNES/KSP and outer-split validity
   * - (C)
     - ``_refresh_attempt_energies``
     - Atlas radius guard + energy snapshot
   * - (D)
     - ``_apply_residual_gate``
     - Residual backtrack, merit, soft accept
   * - (E)
     - ``_apply_thermodynamic_gate``
     - Thermodynamic inequality; optional E1 backtrack
   * - (F)
     - ``_apply_closure_gate``
     - Atlas closure and jump gate

Module-level helpers (below) are called from ``init_dynamic`` on each subclass. Drivers
normally interact with the concrete solver or factory wrapper, not these functions directly.

Checkpoint files
----------------

Required on restart:

- ``<checkpoint_base>_qA.npz``, ``<checkpoint_base>_qB.npz``

Optional BDF history:

- ``_qA_prev.npz``, ``_qA_prev_prev.npz``, ``_qB_prev.npz``, ``_qB_prev_prev.npz``

Base class (selected API)
-------------------------

.. autoclass:: Zmag_bins.Zmag_dynamic_stereo_base.Z_sim_dynamic_stereo_base
   :members:
       check_and_save_mesh_partition,
       load_mesh_partition,
       set_mesh,
       set_materials,
       set_external_field,
       update_demag_ramp_factor,
       update_zeeman_ramp_factor,
       stereo_to_cart_chart,
       check_q_magnitude,
       solve_problem,
       effective_field,
       write_energies,
   :show-inheritance:

Module-level functions and constants
------------------------------------

.. automodule:: Zmag_bins.Zmag_dynamic_stereo_base
   :members:
   :exclude-members: Z_sim_dynamic_stereo_base
