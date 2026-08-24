Z-mag
=====

``Z-mag`` is an open-source finite-element library for micromagnetics, developed
at IBM Research, Zurich.  It integrates the time-dependent
Landau–Lifshitz–Gilbert equation on MPI-parallel unstructured meshes using
**DOLFINx / FEniCSx**, and represents the magnetization direction through a
**two-chart stereographic atlas** so the unit-norm constraint
:math:`|\mathbf{m}|=1` is satisfied exactly by construction—no renormalization,
penalty term, or saddle-point system is required.  Two solver variants are
provided: **demag**, with a self-consistent demagnetizing field computed via a
hybrid FEM–BEM operator-split scheme, and **nodemag**, without magnetostatics.
Both share the same adaptive implicit BDF time integrator with thermodynamic
acceptance gates.

The *Z* in the name reflects the project's origin at the IBM Zurich Research
Laboratory.  For the logo we chose the **Eiger**, **Mönch**, and **Jungfrau**—
three iconic summits of the Swiss Alps—because our solver also spends its life
climbing brutal energy gradients, balancing on unstable magnetization states,
and remaining firmly convinced that convergence is just over the next ridge.

.. figure:: images/mountains.jpeg
   :scale: 30%
   :align: center
   :alt: Eiger, Mönch, and Jungfrau

   Eiger, Mönch, and Jungfrau.
   `Photo © Adventure Peaks <https://www.adventurepeaks.com/site/assets/files/1442/eiger.jpg>`_

|br|

A full description of the design rationale, solver architecture, repository,
license, and contributors is in :doc:`overview`.

Recommended reading order
--------------------------

1. :doc:`overview` — about ``Z-mag``: design, repository, license, contributors
2. :doc:`installation` — dependencies, BEM build, and quick checks
3. :doc:`getting_started` — annotated minimal driver
4. :doc:`theory_llg` → :doc:`theory_energy` → :doc:`theory_magnetostatics` → :doc:`theory_dynamics`
5. :doc:`theory_algorithms` — timestep control, thermodynamic gates, outer split
6. :doc:`theory_benchmarks` — precession–damping, µMAG Standard Problem 3, checkpoint chain
7. :ref:`configuration` and API (:doc:`api_factory`, :doc:`api_common`, :doc:`api_demag`, :doc:`api_nodemag`)

.. note::
   ``Z-mag`` is under active development. API and defaults may change between
   releases. Please cite the paper :cite:`brach2026zmag` and the repository
   revision you use; see :doc:`overview`.

Contents
--------

Getting started
~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 3
   :caption: Getting started

   overview
   installation
   getting_started

Theory
~~~~~~

.. toctree::
   :maxdepth: 2
   :caption: Theory

   theory_llg
   theory_energy
   theory_magnetostatics
   theory_dynamics
   theory_algorithms
   theory_benchmarks

Configuration and API
~~~~~~~~~~~~~~~~~~~~~

.. toctree::
   :maxdepth: 2
   :caption: Solver documentation

   config
   api_factory
   api_common
   api_demag
   api_nodemag
   zreferences

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. |br| raw:: html

   <br />
