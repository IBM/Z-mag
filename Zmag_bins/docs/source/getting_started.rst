Getting started
===============

This section describes a complete ``Z-mag`` simulation from mesh construction to
time-marched magnetization, explaining the purpose of each configuration step.
The prerequisite is a working installation (:doc:`installation`).

``Z-mag`` integrates the time-dependent **Landau–Lifshitz–Gilbert (LLG)** equation
on a finite-element mesh, with the magnetization :math:`\mathbf{m}` represented
through a two-chart stereographic atlas :math:`(\mathbf{q}_A, \mathbf{q}_B)`.
The unit-norm constraint :math:`|\mathbf{m}|=1` is satisfied by construction at
every node and every time step; no renormalization or penalty is required.
Two solver variants are available, distinguished by whether the demagnetizing field
is included:

* **demag** — full LLG with magnetostatics.  The scalar potential
  :math:`\varphi` is solved jointly with the chart fields via a hybrid FEM–BEM
  loop (:doc:`theory_magnetostatics`).
* **nodemag** — LLG without magnetostatics.  Appropriate when the demagnetizing
  contribution is negligible or intentionally excluded; computationally lighter
  than ``demag``.

Both variants are instantiated through the same factory function and share the
same setup API.  The steps below use ``demag`` and note where ``nodemag`` differs.

.. container:: solver-flow-figure-block

   .. raw:: html
      :file: _includes/getting_started_flow.html

**Figure 1.** Setup workflow for a ``Z-mag`` simulation.
The diamond at step 4 marks the only branch: ``demag`` requires an additional
magnetostatics configuration step; ``nodemag`` skips it and proceeds directly to
initialization.  The badge legend at the bottom refers to the solver's internal
timestep-acceptance outcomes described in :doc:`theory_algorithms`.


Step 1 — Mesh
--------------

``Z-mag`` requires a DOLFINx mesh together with two tag arrays:

* **cell tags** — one integer label per element, used to assign material regions.
* **facet tags** — one integer label per boundary facet, used to mark
  outer surfaces (for the open-boundary magnetostatic BEM) and internal
  interfaces between materials.

The standard workflow is to build the geometry in **Gmsh**, assign physical groups
with integer tags, and convert to DOLFINx with ``dolfinx.io.gmshio``:

.. code-block:: python

   import gmsh
   from mpi4py import MPI
   import dolfinx.io

   # Build geometry in Gmsh (not shown) ...
   gmsh.initialize()
   # ... add points, volumes, surfaces, physical groups ...

   domain, cell_tags, facet_tags = dolfinx.io.gmshio.model_to_mesh(
       gmsh.model, MPI.COMM_WORLD, rank=0, gdim=3
   )
   gmsh.finalize()

Alternatively, load a previously saved mesh from XDMF or ADIOS2 checkpoints.

.. note::
   Tag integers are arbitrary but must be consistent: the same integer assigned
   as a physical group in Gmsh must be the one passed to
   ``set_mesh``, ``set_magnetostatics``, and ``set_materials``.


Step 2 — Create solver
-----------------------

The solver is instantiated through the factory function ``create_stereo_solver``:

.. code-block:: python

   from Zmag_bins import create_stereo_solver

   args_init_sim = {
       "alpha": 0.02,                                     # Gilbert damping α
       "energy_terms": ["exchange", "uniaxial_anisotropy", "demag"],
       "NeumannBC": True,                                 # natural BC on ∂Ω_m
       "printout": MPI.COMM_WORLD.rank == 0,
       "path_dirs": {
           "results_dir": "./results",
           "checkpoint_dir": "./chk",
       },
   }

   sim = create_stereo_solver(args_init_sim, model="demag")

The ``model`` keyword selects the solver variant: ``"demag"`` activates the full
FEM–BEM solver, while ``"nodemag"`` selects the magnetostatics-free variant.

**Energy terms.**  The ``energy_terms`` list names the contributions included in
the variational residual.  Recognized keys are:

* ``"exchange"`` — Heisenberg exchange stiffness.
* ``"uniaxial_anisotropy"`` — uniaxial magnetocrystalline anisotropy.
* ``"cubic_anisotropy"`` — cubic magnetocrystalline anisotropy.
* ``"zeeman"`` — Zeeman coupling to an applied field.
* ``"demag"`` — demagnetizing energy via the FEM–BEM scalar potential.

If ``"demag"`` is omitted while using ``model="demag"``, the demagnetizing energy
is not included in the residual; conversely, listing ``"demag"`` with
``model="nodemag"`` will raise an error.

**Gilbert damping** :math:`\alpha` sets the strength of the dissipative torque
in the LLG equation (see :doc:`theory_llg`).  Typical physical values range
from 0.01 (lightly damped) to 0.5 (heavily overdamped).

**NeumannBC = True** applies the natural zero-flux boundary condition
:math:`\partial\mathbf{m}/\partial\mathbf{n}=0` on the outer surface of the
magnetic domain.  This is the correct micromagnetic Brown boundary condition
for a free surface (:doc:`theory_llg`).


Step 3 — Mesh and materials
----------------------------

.. code-block:: python

   surface_tag = 12   # integer label of the outer surface facets in your mesh

   sim.set_mesh(domain, cell_tags, facet_tags, surface_tag)

Then define materials.  At least one entry is required.  ``Z-mag`` operates in
**nondimensional units**: one material region is designated as the reference and
assigned :math:`M_s^\ast = A^\ast = K^\ast = 1`; all other regions supply
dimensionless ratios relative to this reference (see :doc:`theory_energy`).

.. code-block:: python

   materials = [
       {
           "tag": 1,                        # cell tag for this region
           "Ms":  1.0,                      # M_s* (1 = reference)
           "Aex": 1.0,                      # A* (exchange stiffness ratio)
           "K":   1.0,                      # K* (anisotropy constant ratio)
           "Q1":  0.1,                      # Q = 2K_1 / (μ_0 M_{s,ref}²)
           "K2K1_ratio": 0.0,               # K_2/K_1 (uniaxial second-order)
           "anisotropy_axis": (0.0, 0.0, 1.0),
       },
       # Additional dicts for multi-region geometries
   ]

   sim.set_materials(materials)

``Q1`` is the **quality factor** :math:`Q = 2K_1/(\mu_0 M_{\mathrm{s}}^2)` of
the reference material.  It sets the scale of anisotropy energy relative to
magnetostatic energy and is the only parameter that carries physical units
implicitly (:doc:`theory_energy`).

An external (applied) field is registered separately.  The Zeeman vector is
**nondimensional**, normalized by :math:`M_{\mathrm{s,ref}}`:

.. code-block:: python

   sim.set_external_field([0.0, 0.0, 0.01])   # H_zee* in units of M_{s,ref}


Step 4 — Magnetostatics (demag only)
--------------------------------------

This step is not required for ``nodemag`` and should be omitted entirely in that
case.

The Fredkin–Koehler FEM–BEM decomposition requires two categories of tagged
facets:

* **Outer surface** (:math:`\partial\Omega_\mathrm{m}`) — the boundary between
  the magnetic domain and the surrounding vacuum.  This is the same ``surface_tag``
  already passed to ``set_mesh`` in Step 3.  It is the surface over which the BEM
  integral is assembled to impose the open-boundary condition on the harmonic part
  :math:`\varphi_1` of the scalar potential.
* **Material interfaces** (:math:`\Gamma`) — internal facets separating two
  distinct material sub-regions.  At these facets the normal derivative of
  :math:`\varphi_0` is discontinuous, and the jump condition
  :math:`(\mathbf{M}^+ - \mathbf{M}^-)\cdot\mathbf{n}^+` must be enforced
  explicitly.  Each interface is identified by its own integer facet tag.

Both are passed to ``set_magnetostatics``, followed by the path to the compiled
BEM shared library:

.. code-block:: python

   interfaces = [
       {"name": "interface_layer", "tag": 10},   # facet tag of each internal interface Γ
   ]

   sim.set_magnetostatics(interfaces, surface_tag)
   sim.set_BEMcpp("/path/to/bem/build")          # directory containing bem_cpp.so

For a single-material geometry with no internal interfaces, the ``interfaces``
list is empty.  The BEM build is described in :doc:`installation`; the
underlying algorithm in :doc:`theory_magnetostatics`.


Step 5 — Initialize magnetization
-----------------------------------

Before time integration can begin, the two chart fields :math:`\mathbf{q}_A`
and :math:`\mathbf{q}_B` must be set to represent the desired initial
magnetization :math:`\mathbf{m}(\mathbf{x},0)`.  This is a one-time map from
a user-supplied field into the dual-chart atlas state; the full pipeline is
described in :ref:`dual-chart-initialization` (:doc:`theory_dynamics`).

Four entry points are available, depending on how the initial state is specified:

* :meth:`~Zmag_bins.Zmag_dynamic_stereo_base.Z_sim_dynamic_stereo_base.set_init_q_from_m_function`
  — the most general route: accepts a callable ``eval_m_init(x)`` that returns
  Cartesian :math:`(m_x, m_y, m_z)` arrays evaluated at node coordinates of
  shape :math:`(3, N)`.  Any unit or non-unit field is accepted; vectors are
  normalized internally.
* :meth:`~Zmag_bins.Zmag_dynamic_stereo_base.Z_sim_dynamic_stereo_base.set_init_q_from_theta_function`
  — accepts a callable that returns spatially varying spherical angles
  :math:`(\theta, \phi)` at each node.
* :meth:`~Zmag_bins.Zmag_dynamic_stereo_base.Z_sim_dynamic_stereo_base.set_init_q_from_angles`
  — initializes a spatially uniform direction from a single pair of spherical
  angles, with optional Gaussian noise and Helmholtz smoothing to seed
  domain-wall nucleation.
* :meth:`~Zmag_bins.Zmag_dynamic_stereo_base.Z_sim_dynamic_stereo_base.set_init_q_from_external_field`
  — aligns the initial magnetization with the local Zeeman field direction at
  each node; requires ``"zeeman"`` in ``energy_terms``.

The example below uses ``set_init_q_from_angles``:

.. code-block:: python

   sim.set_init_q_from_angles(
       theta0=0.1,      # polar angle at centre (radians)
       theta1=0.0,      # polar angle at boundary
       std_u=0.05,      # noise amplitude on q_u
       std_v=0.05,      # noise amplitude on q_v
       smooth_len=2,    # smoothing length (mesh cells)
       iters=4,         # smoothing iterations
   )


Step 6 — Time integration
--------------------------

``init_dynamic`` receives a second configuration dict that controls the
time-stepping and I/O, then ``solve_problem`` runs the integration:

.. code-block:: python

   args_init_dyn = {
       "t_start":          0.0,
       "dt":               1e-3,        # initial timestep (nondimensional)
       "step_start":       0,
       "step_end":         500,
       "first_run":        True,        # False to resume from a checkpoint
       "checkpoint_every": 50,
       "write_target":     ["mag", "qA", "qB"],
       "cfl_settings": {
           "CFL":            0.5,
           "dt_min":         1e-7,      # minimum allowed timestep
           "dt_max":         1e-2,      # maximum allowed timestep
           "smooth_factor":  0.2,
           "max_growth":     1.25,      # max dt growth per accepted step
           "min_shrink":     0.5,       # min dt shrink on rejection
           "error_rel_tol":  0.05,
           "error_abs_tol":  1e-6,
           "error_headroom": 1.2,
           "max_retry_thermo": 3,
       },
   }

   sim.init_dynamic(args_init_dyn)
   sim.solve_problem()

**Timestep and units.**  ``dt`` is nondimensional, normalized by the LLG
precession time scale :math:`(\gamma_0 \mu_0 M_{\mathrm{s,ref}})^{-1}` (see
:doc:`theory_dynamics`).  The ``cfl_settings`` sub-dict drives the adaptive
controller:

* ``dt_min``, ``dt_max`` — bounds on the timestep search range.
* ``max_growth``, ``min_shrink`` — limits on how aggressively the controller
  may adjust :math:`\Delta t^\ast` in a single step.
* ``error_rel_tol``, ``error_abs_tol`` — tolerances for the embedded
  error estimator.
* ``error_headroom`` — safety factor applied to the error-based step proposal.
* ``max_retry_thermo`` — maximum number of retries allowed at the
  thermodynamic gate before the step is unconditionally rejected.

The full acceptance logic is described in :doc:`theory_algorithms`.

**Output fields.**  ``write_target`` names the fields written to the results
directory at each output step.  Available keys include:

* ``"mag"`` — Cartesian magnetization :math:`\mathbf{m}`.
* ``"qA"``, ``"qB"`` — stereographic chart coordinates.
* ``"phi"`` — scalar magnetic potential (demag only).
* ``"Hd_scaled"`` — demagnetizing field (demag only).

The complete list of keys is in :ref:`configuration`.

**Checkpoints and restarts.**  Every ``checkpoint_every`` steps the full chart
state is written to ``path_dirs["checkpoint_dir"]``.  To resume a run, set
``first_run=False``; the solver will load the latest available checkpoint
automatically.


Nodemag variant
----------------

For the ``nodemag`` variant, replace the ``model`` argument, remove ``"demag"``
from ``energy_terms``, and omit Step 4 entirely:

.. code-block:: python

   args_init_sim["energy_terms"] = ["exchange", "uniaxial_anisotropy"]
   sim = create_stereo_solver(args_init_sim, model="nodemag")

   sim.set_mesh(domain, cell_tags, facet_tags, surface_tag)
   sim.set_materials(materials)
   sim.set_external_field([0.0, 0.0, 0.01])
   sim.set_init_q_from_angles(theta0=0.1, theta1=0.0,
                               std_u=0.05, std_v=0.05,
                               smooth_len=2, iters=4)
   sim.init_dynamic(args_init_dyn)
   sim.solve_problem()

All remaining configuration and time-integration steps are identical to those
described for the ``demag`` variant above.


Next steps
----------

* Physical background: :doc:`theory_llg`, :doc:`theory_energy`,
  :doc:`theory_magnetostatics`, :doc:`theory_dynamics`
* Timestep control, thermodynamic gates, outer split: :doc:`theory_algorithms`
* Validated benchmarks: :doc:`theory_benchmarks`
* Full list of configuration keys: :ref:`configuration`
* Class reference: :doc:`api_demag`, :doc:`api_nodemag`
