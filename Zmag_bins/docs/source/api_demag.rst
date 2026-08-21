Demag dynamic solver
====================

:class:`~Zmag_bins.Zmag_dynamic_stereo_demag.Z_sim_dynamic_stereo_demag`
subclasses :class:`~Zmag_bins.Zmag_dynamic_stereo_base.Z_sim_dynamic_stereo_base`
and adds mixed finite elements for ``(q_A, q_B, \varphi)`` plus magnetostatics/BEM.

Responsibilities
----------------

* Build mixed spaces and the weak form in :doc:`theory_dynamics`
* Couple to :class:`~Zmag_bins.Zmag_magnetostatic.Z_sim_magnetostatic` and C++ BEM (``set_BEMcpp``)
* Operator-split SNES per timestep on ``(q_A, q_B, \varphi)`` with field-split preconditioning
* ``init_dynamic``, ``write_checkpoint``, demag-specific output targets (see :ref:`write-targets`)

Typical call sequence
---------------------

``set_mesh`` → ``set_materials`` → ``set_external_field`` → ``set_magnetostatics`` →
``set_BEMcpp`` → initialize ``q_A``/``q_B`` → ``init_dynamic`` → ``solve_problem``.

Configuration keys: :ref:`configuration`. Workflow example: :doc:`getting_started`.

.. autoclass:: Zmag_bins.Zmag_dynamic_stereo_demag.Z_sim_dynamic_stereo_demag
   :members:
   :member-order: bysource
   :exclude-members: NonlinearSNESProblem_qAqB
   :special-members: __init__
   :show-inheritance:

.. seealso::

   Module overview: :mod:`Zmag_bins.Zmag_dynamic_stereo_demag`
