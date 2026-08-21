About ``Z-mag``
===============

``Z-mag`` is an open-source finite-element library for micromagnetics, developed in the
`Tape Research group <https://research.ibm.com/projects/tape-research>`_ at IBM Research,
Zurich.  It is designed for large-scale,
MPI-parallel simulations of ferromagnetic systems and is built on top of
**DOLFINx / FEniCSx** for mesh management, parallel linear algebra, and
parallel input/output.


Stereographic parametrization
------------------------------

The magnetization at each point in space is a unit vector, so its evolution must
stay on the unit sphere. Standard ways of enforcing this constraint can involve
trade-offs that affect accuracy or robustness, especially in long simulations
(see :doc:`theory_llg`).

``Z-mag`` instead parametrizes the magnetization through a **two-chart stereographic
atlas**.  The unit-norm constraint is built into the parametrization, which removes
the need for any explicit enforcement mechanism.  The mathematical development is
in :doc:`theory_llg` and :doc:`theory_dynamics`.


Solver variants
---------------

Two variants are provided, distinguished by which energy contributions are active:

* **demag** — exchange, anisotropy, Zeeman, and demagnetizing energy.  The
  demagnetizing field is computed at every time step via the **Fredkin–Koehler
  hybrid FEM–BEM** method, which handles the open exterior domain exactly without
  domain truncation.  See :doc:`theory_magnetostatics` and :doc:`theory_algorithms`.

* **nodemag** — exchange, anisotropy, and Zeeman only, without magnetostatics.
  Appropriate for problems where the demagnetizing contribution is negligible or
  intentionally excluded; computationally lighter than ``demag``.

Both variants share the same implicit **BDF time integrator** with adaptive timestep
control and a multi-phase acceptance logic that combines algebraic, geometric, and
thermodynamic criteria.  The full acceptance algorithm is documented in
:doc:`theory_algorithms`.


Repository
----------

Source code, example drivers, and the C++ BEM assembler are published under the
MIT License at:

   `https://github.com/IBM/Z-mag <https://github.com/IBM/Z-mag>`_

Stable releases follow **Semantic Versioning** (``v1.0.0``, ``v1.1.0``, …) and are
described in ``CHANGELOG.md`` at the repository root.  The ``environment.yml`` shipped
with each tag pins the exact DOLFINx / FEniCSx stack required by that release.
Installation instructions are in :doc:`installation`.


License and contributors
------------------------

``Z-mag`` is distributed under the **MIT License** (copyright 2025–2026 ``Z-mag``
contributors).  The complete license text is in ``LICENSE`` at the repository root.

The library was developed by:

* Stella Brach, Staff Research Scientist — IBM Research, Zurich
* Kathrin Aebischer, Research Scientist — IBM Research, Zurich
* Federico Soldà, former intern
