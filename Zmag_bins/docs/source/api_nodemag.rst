Nodemag dynamic solver
======================

:class:`~Zmag_bins.Zmag_dynamic_stereo_nodemag.Z_sim_dynamic_stereo_nodemag`
subclasses :class:`~Zmag_bins.Zmag_dynamic_stereo_base.Z_sim_dynamic_stereo_base`
with mixed unknowns ``(q_A, q_B)`` only. The ``demag`` term is stripped from ``energy_terms``;
``set_magnetostatics`` and ``set_BEMcpp`` are no-ops.

Use when shape demagnetization is negligible or when external Zeeman fields are prescribed.
Exchange, anisotropy, Zeeman, atlas logic, BDF stepping, and acceptance gates match the demag
variant. Allowed outputs are listed under :ref:`write-targets` (demag-only names raise
``ValueError``).

Typical call sequence
---------------------

``set_mesh`` → ``set_materials`` → ``set_external_field`` → initialize charts →
``init_dynamic`` → ``solve_problem``.

Configuration keys: :ref:`configuration`. Workflow example: :doc:`getting_started`.

.. autoclass:: Zmag_bins.Zmag_dynamic_stereo_nodemag.Z_sim_dynamic_stereo_nodemag
   :members:
   :member-order: bysource
   :exclude-members: NonlinearSNESProblem_qAqB
   :special-members: __init__
   :show-inheritance:

.. seealso::

   Module overview: :mod:`Zmag_bins.Zmag_dynamic_stereo_nodemag`
