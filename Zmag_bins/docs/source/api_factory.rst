Dynamic solver factory
======================

:mod:`Zmag_bins.Zmag_dynamic_stereo_solvers` is the recommended entry point for
**time-dependent** stereographic LLG. It selects the implementation and exposes a uniform API
via attribute forwarding.

.. code-block:: python

   from Zmag_bins.Zmag_dynamic_stereo_solvers import create_stereo_solver

   sim = create_stereo_solver(args_init_sim, model="demag")  # or "nodemag"
   sim.set_mesh(...)
   sim.init_dynamic(args_init_dyn)
   sim.solve_problem()   # defined on Z_sim_dynamic_stereo_base

``sim.impl`` is the underlying :class:`~Zmag_bins.Zmag_dynamic_stereo_demag.Z_sim_dynamic_stereo_demag`
or :class:`~Zmag_bins.Zmag_dynamic_stereo_nodemag.Z_sim_dynamic_stereo_nodemag` instance.

Exports
-------

* :class:`StereoDemagSolver`, :class:`StereoNoDemagSolver` — validation + delegation wrappers
* :func:`create_stereo_solver` — factory
* :data:`Z_sim_dynamic_stereo_demag`, :data:`Z_sim_dynamic_stereo_nodemag` — concrete classes

Shared timestep control, checkpoints, and the main loop are documented in :doc:`api_common`.
Model-specific FEM spaces and SNES forms are in :doc:`api_demag` and :doc:`api_nodemag`.

.. automodule:: Zmag_bins.Zmag_dynamic_stereo_solvers
   :members:
   :show-inheritance:
