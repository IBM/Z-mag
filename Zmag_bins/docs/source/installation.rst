Installation
============

``Z-mag`` is distributed from the IBM GitHub repository
`https://github.com/IBM/Z-mag <https://github.com/IBM/Z-mag>`_ under the **MIT License**.
Releases are identified by **git tags** following Semantic Versioning
(``v1.0.0``, ``v1.1.0``, …).  The import name ``Zmag_bins`` is unchanged across
releases; selecting a tag fixes the software version of a given installation.

The procedure below installs from a clone of the repository.  To pin a specific
release, check out the corresponding git tag listed in :ref:`release-changelog`
before creating the conda environment.


.. _release-changelog:

Release history
---------------

The following table records what is in this tree.  After checking out a tag,
proceed with :ref:`install-latest` from that checkout.

.. list-table::
   :header-rows: 1
   :widths: 18 22 60

   * - Tag
     - Status
     - Summary
   * - ``main``
     - Current checkout
     - First public tree:

       - Stereographic dynamic ``demag`` / ``nodemag`` solvers
       - Hybrid FEM–BEM magnetostatics helpers
       - Sphinx manual under ``Zmag_bins/docs/``
       - Example drivers (Standard Problem 3, precession–damping, checkpoint / restart)
       - Runtime stack pinned in ``environment.yml`` (FEniCSx / DOLFINx, PETSc, Gmsh, MPI I/O)

       The first tagged release will be ``v1.0.0``.

Subsequent releases will be recorded here as new tags are published.  Detailed
release notes are maintained in ``CHANGELOG.md`` at the repository root.


.. _install-latest:

Install
-------

Prerequisites
~~~~~~~~~~~~~

- `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_ or Miniforge
- Git
- A C++17 compiler, CMake 3.15 or newer, and pybind11 (required to compile the BEM bindings)

1. Clone the repository
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: shell

   git clone https://github.com/IBM/Z-mag.git
   cd Z-mag

To install a tagged release instead of ``main``, run ``git fetch --tags`` and
``git checkout vX.Y.Z`` (see :ref:`release-changelog`).

2. Create the ``zmag`` environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

From the repository root, create and activate the conda environment named ``zmag``:

.. code-block:: shell

   conda env create -n zmag -f environment.yml
   conda activate zmag
   pip install "git+https://github.com/jorgensd/adios4dolfinx.git"

The file ``environment.yml`` specifies the FEniCSx / DOLFINx stack and companion
packages (MPI, PETSc, Gmsh, meshio, ADIOS2, and related dependencies) for the
checked-out release.  Package pins may differ between tags; the
``environment.yml`` shipped with the selected tag must be used.

If ``import adios4dolfinx`` raises ``No module named 'adios2'``, install the MPI
ADIOS2 binding:

.. code-block:: shell

   conda activate zmag
   conda install -y -c conda-forge "adios2=*=mpi_openmpi*"

3. Build the BEM bindings
~~~~~~~~~~~~~~~~~~~~~~~~~

Compile the C++ BEM accelerator provided in ``bem_assembler/cpp/``.  This step is
required for a complete installation and is not limited to ``demag`` simulations:

.. code-block:: shell

   cd bem_assembler/cpp
   ./install_python_bindings.sh --clean

If the bindings are not discovered automatically, prepend the build directory to
``PYTHONPATH``:

.. code-block:: shell

   export PYTHONPATH=/path/to/bem/build:${PYTHONPATH}

Confirm that the module loads:

.. code-block:: python

   import bem_cpp
   print(bem_cpp.__version__)

4. Configure the Python path
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Add the repository root to ``PYTHONPATH``:

.. code-block:: shell

   export PYTHONPATH="$(pwd):${PYTHONPATH}"

The factory API can then be imported as:

.. code-block:: python

   from Zmag_bins import create_stereo_solver

   solver = create_stereo_solver(model="demag")    # or model="nodemag"


Verification
------------

After installation, activate the environment and confirm that the principal
modules import successfully:

.. code-block:: shell

   conda activate zmag

.. code-block:: python

   import dolfinx
   from Zmag_bins import create_stereo_solver
   import bem_cpp

   print(dolfinx.__version__)
   print(bem_cpp.__version__)

An end-to-end check of the full stack can be performed from the repository root
by running one of the included example drivers, for example:

.. code-block:: shell

   mpirun -n 4 python test_dynamic_checkpoints.py


Build the manual
----------------

To rebuild this documentation from a source checkout:

.. code-block:: shell

   cd Zmag_bins/docs
   pip install -r requirements.txt
   make html

The generated HTML is written to ``Zmag_bins/docs/build/html/``.
