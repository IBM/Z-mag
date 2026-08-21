Micromagnetic energy
====================

This section defines the micromagnetic free energy functional and derives the two
quantities it determines: the effective field :math:`\mathbf{H}_\text{eff}`, which drives
the LLG dynamics introduced in :doc:`theory_llg`, and Brown's equations, which
characterize static energy-minimizing states.  It then presents the non-dimensional
form :math:`\mathcal{E}^\ast` used by ``Z-mag`` internally, and the stored energy
:math:`\mathcal{E}_\text{sol}` assembled at each time step.  The demagnetizing contribution
involves a scalar magnetostatic potential :math:`\varphi` whose computation is described
in :doc:`theory_magnetostatics`; the full weak form coupling energy and dynamics is
developed in :doc:`theory_dynamics`.

The magnetization :math:`\mathbf{m}:\Omega_\mathrm{m}\to\mathcal{S}^2` is the
unit-vector field introduced in :doc:`theory_llg`, defined on the ferromagnetic domain
:math:`\Omega_\mathrm{m}\subset\mathbb{R}^3`.  The physical magnetization is
:math:`\mathbf{M}=M_\mathrm{s}\mathbf{m}`, with saturation magnetization
:math:`M_\mathrm{s}(\mathbf{x})` [A/m].

Dimensional energy :math:`\mathcal{E}_\mathrm{SI}`
---------------------------------------------------

The total micromagnetic free energy is the sum of four terms
:cite:`brown1963micromagnetics,abert2019micromagnetics`:

.. math::
    :label: eq:energy_total

    \mathcal{E}_\mathrm{SI}(\mathbf{m})
    = \mathcal{E}_\text{ex}
    + \mathcal{E}_\text{ani}
    + \mathcal{E}_\text{zee}
    + \mathcal{E}_\text{d}
    \qquad\text{[J]}.

The subscript SI indicates that this energy is expressed in physical, dimensioned (SI)
units.  This dimensional form is the starting point from which the non-dimensional energy
:math:`\mathcal{E}^\ast` is derived below.  Each term is an integral over
:math:`\Omega_\mathrm{m}`; all material coefficients may vary piecewise over cell tags,
allowing modeling of heterogeneous media.

Exchange energy
~~~~~~~~~~~~~~~

The quantum-mechanical exchange interaction penalizes spatial variations of the
magnetization orientation.  In the micromagnetic continuum limit
:cite:`brown1963micromagnetics`, this is expressed as a penalty on the spatial gradient
of :math:`\mathbf{m}`:

.. math::
    :label: eq:energy_exchange

    \mathcal{E}_\text{ex}
    = \int_{\Omega_\mathrm{m}} A(\mathbf{x})\,|\nabla\mathbf{m}|^2 \, dV,

where :math:`A` [J/m] is the exchange stiffness.  The ratio of exchange to magnetostatic
energy defines a material length scale, the *exchange length*

.. math::
    :label: eq:ell_ex_def

    \ell_\mathrm{ex} = \sqrt{\frac{2A}{\mu_0 M_\mathrm{s}^2}},

where :math:`\mu_0 = 4\pi\times10^{-7}` [T·m/A] is the vacuum permeability.  On length
scales below :math:`\ell_\mathrm{ex}`, the exchange interaction dominates over
magnetocrystalline anisotropy and demagnetization.

Anisotropy energy
~~~~~~~~~~~~~~~~~

Magnetocrystalline anisotropy arises from the coupling between the electron spin and the
crystal lattice, which makes certain crystallographic directions energetically preferred
for the magnetization (*easy axes*) and others unfavorable (*hard axes*).

For a material with uniaxial symmetry, a single easy-axis unit vector
:math:`\mathbf{e}_\text{ua}` characterizes the preferred orientation.  The energy density
is minimal when :math:`\mathbf{m}\parallel\mathbf{e}_\text{ua}` and maximal when
:math:`\mathbf{m}\perp\mathbf{e}_\text{ua}`, and takes the form

.. math::
    :label: eq:energy_anisotropy

    \mathcal{E}_\text{ani}^\text{uni}
    = \int_{\Omega_\mathrm{m}} K_1(\mathbf{x})
    \bigl[1 - (\mathbf{m}\cdot\mathbf{e}_\text{ua})^2\bigr]\,dV,

where :math:`K_1` [J/m³] is the first-order uniaxial anisotropy constant.  It is
convenient to introduce the scalar shape function

.. math::
    :label: eq:f_ani

    f_\mathrm{ani} \equiv 1-(\mathbf{m}\cdot\mathbf{e}_\mathrm{ua})^2,

so that :math:`\mathcal{E}_\text{ani}^\text{uni}=\int_{\Omega_\mathrm{m}} K_1
f_\mathrm{ani}\,dV`.

For a material with cubic symmetry, there is no single preferred axis.  Denoting by
:math:`(m_1,m_2,m_3)` the direction cosines of :math:`\mathbf{m}` with respect to the
three orthogonal cubic axes, the anisotropy energy density to second order in these
cosines is

.. math::
    :label: eq:energy_anisotropy_cubic

    \mathcal{E}_\text{ani}^\text{cub}
    = \int_{\Omega_\mathrm{m}} \bigl[
      K_1\bigl(m_1^2 m_2^2 + m_2^2 m_3^2 + m_3^2 m_1^2\bigr)
    + K_2\,m_1^2 m_2^2 m_3^2
    \bigr]\,dV,

where :math:`K_1` and :math:`K_2` [J/m³] are the first- and second-order cubic
anisotropy constants.  ``Z-mag`` supports both symmetry classes.  The remainder of this
documentation employs the uniaxial form; the cubic case follows an analogous derivation.

Zeeman energy
~~~~~~~~~~~~~

The Zeeman energy quantifies the interaction between the magnetization and an externally
applied field :math:`\mathbf{H}_\text{zee}` [A/m].  It equals the negative of the work
done by the field on the magnetic moment as the magnetization rotates from a reference
state, and is given by :cite:`brown1963micromagnetics`

.. math::
    :label: eq:energy_zeeman

    \mathcal{E}_\text{zee}
    = -\mu_0 \int_{\Omega_\mathrm{m}}
    M_\mathrm{s}(\mathbf{x})\,\mathbf{m}\cdot\mathbf{H}_\text{zee}\,dV.

The negative sign ensures that configurations with :math:`\mathbf{m}` aligned with
:math:`\mathbf{H}_\text{zee}` have lower energy.

Demagnetization energy
~~~~~~~~~~~~~~~~~~~~~~

A magnetized body produces a magnetic field that extends throughout all of space,
including the interior of the body itself.  This internal field, called the *demagnetizing
field* :math:`\mathbf{H}_\text{d}` [A/m], is generated by the divergence of
:math:`\mathbf{M}` in the bulk and by the distribution of magnetic surface charges at
:math:`\partial\Omega_\mathrm{m}`, the boundary of the magnetic domain.  It is related to
a scalar magnetostatic potential :math:`\varphi` [A] by

.. math::
    :label: eq:Hd_def

    \mathbf{H}_\text{d} = -\nabla\varphi.

The determination of :math:`\varphi` from :math:`\mathbf{m}` constitutes the
magnetostatic subproblem described in :doc:`theory_magnetostatics`.  Because
:math:`\mathbf{H}_\text{d}` is itself determined by the magnetization distribution, the
demagnetization energy represents a self-interaction, penalizing configurations that
generate a large stray field.  The energy of this self-interaction is

.. math::
    :label: eq:energy_demagnetization

    \mathcal{E}_\text{d}
    = -\frac{\mu_0}{2}
    \int_{\Omega_\mathrm{m}} M_\mathrm{s}(\mathbf{x})\,\mathbf{m}\cdot\mathbf{H}_\text{d}\,dV.

The factor :math:`\tfrac{1}{2}` prevents double-counting: the magnetostatic interaction
between every pair of volume elements is counted once in :math:`\mathcal{E}_\text{d}`,
whereas the integrand :math:`\mathbf{m}\cdot\mathbf{H}_\text{d}` accumulated over
:math:`\Omega_\mathrm{m}` would otherwise count each pair twice.

Effective field and Brown's equations
--------------------------------------

The effective field :math:`\mathbf{H}_\text{eff}` [A/m] is defined as the negative
variational derivative of the total energy with respect to :math:`\mathbf{m}`,
normalized by :math:`\mu_0 M_\mathrm{s}`:

.. math::
    :label: eq:heff_definition

    \mathbf{H}_\text{eff}
    = -\frac{1}{\mu_0 M_\mathrm{s}}\,
    \frac{\delta\mathcal{E}_\mathrm{SI}}{\delta\mathbf{m}}.

This definition collects all torques acting on :math:`\mathbf{m}` into a single field
with units of A/m, providing a common driving term for both static equilibrium and dynamic
evolution.  Term-by-term variation of :eq:`eq:energy_total` yields
:cite:`brown1963micromagnetics,abert2019micromagnetics`:

.. math::
    :label: eq:heff_complete

    \mathbf{H}_\text{eff}
    = \mathbf{H}_\text{zee}
    + \frac{2}{\mu_0 M_\mathrm{s}}\nabla\cdot(A\nabla\mathbf{m})
    + \frac{2K_1}{\mu_0 M_\mathrm{s}}(\mathbf{m}\cdot\mathbf{e}_\text{ua})\mathbf{e}_\text{ua}
    - \nabla\varphi.

This is the quantity that enters the LLG equation :eq:`eq:llg_phys`; its role in time
evolution is developed in :doc:`theory_dynamics`.

At static equilibrium, the torque exerted by :math:`\mathbf{H}_\text{eff}` on
:math:`\mathbf{m}` must vanish everywhere.  Minimizing :math:`\mathcal{E}_\mathrm{SI}`
subject to :math:`|\mathbf{m}|=1` yields Brown's equations
:cite:`brown1963micromagnetics,di2020variational`:

.. math::
    :label: eq:brown_bulk

    \mathbf{m}\times\mathbf{H}_\text{eff} = \mathbf{0}
    \quad\text{in }\Omega_\mathrm{m},

.. math::
    :label: eq:brown_boundary

    A\,\frac{\partial\mathbf{m}}{\partial\mathbf{n}} = \mathbf{0}
    \quad\text{on }\partial\Omega_\mathrm{m},

where :math:`\mathbf{n}` is the outward unit normal to :math:`\partial\Omega_\mathrm{m}`.
Equation :eq:`eq:brown_bulk` states that :math:`\mathbf{H}_\text{eff}` and
:math:`\mathbf{m}` are parallel at every interior point; :eq:`eq:brown_boundary` is the
natural Neumann boundary condition that arises from the exchange energy term upon
integration by parts.

Non-dimensional energy :math:`\mathcal{E}^\ast`
------------------------------------------------

``Z-mag`` works internally with non-dimensional quantities.  The reference material
(cell tag 1) has saturation magnetization :math:`M_\mathrm{s1}`, exchange stiffness
:math:`A_1`, and uniaxial anisotropy constant :math:`K_{11}`.  Two derived quantities
characterize the reference material: the exchange length :math:`\ell_\mathrm{ex1}` and
the anisotropy quality factor :math:`Q_1`, defined by

.. math::
    :label: eq:ell_ex

    \ell_\mathrm{ex1}
    = \sqrt{\frac{2A_1}{\mu_0 M_\mathrm{s1}^2}},
    \qquad
    Q_1 = \frac{2K_{11}}{\mu_0 M_\mathrm{s1}^2},

so that :math:`K_{11}=\tfrac{1}{2}\mu_0 M_\mathrm{s1}^2 Q_1`.  Lengths are scaled by
:math:`\ell_\mathrm{ex1}`, fields by :math:`M_\mathrm{s1}`, and volumes by
:math:`\ell_\mathrm{ex1}^3`:

.. math::
    :label: eq:star_scales_energy

    \mathbf{x} = \ell_\mathrm{ex1}\,\mathbf{x}^\ast,
    \quad
    dV = \ell_\mathrm{ex1}^3\,dV^\ast,
    \quad
    \nabla = \ell_\mathrm{ex1}^{-1}\nabla^\ast,
    \quad
    \mathbf{H} = M_\mathrm{s1}\,\mathbf{H}^\ast.

For heterogeneous domains, the material coefficients of each region are expressed as
dimensionless ratios relative to the reference material:

.. math::
    :label: eq:energy_material_ratios

    M_\mathrm{s}^\ast = \frac{M_\mathrm{s}}{M_\mathrm{s1}},
    \quad
    A^\ast = \frac{A}{A_1},
    \quad
    K^\ast = \frac{K_1}{K_{11}}.

The non-dimensional energy is defined by dividing the physical energy by the reference
energy scale :math:`A_1\ell_\mathrm{ex1}`:

.. math::
    :label: eq:Estar_def

    \mathcal{E}^\ast \equiv \frac{\mathcal{E}_\mathrm{SI}}{A_1\ell_\mathrm{ex1}}.

This choice ensures that exchange gradients enter :math:`\mathcal{E}^\ast` with
:math:`\mathcal{O}(1)` coefficients when :math:`|\nabla^\ast\mathbf{m}|\sim 1`.
Combining the definition of :math:`\ell_\mathrm{ex1}` with the scaling relations
:eq:`eq:star_scales_energy` produces the useful identity

.. math::
    :label: eq:volume_factor

    \mu_0 M_\mathrm{s1}^2\,\ell_\mathrm{ex1}^3 = 2A_1\,\ell_\mathrm{ex1},

which is used in the term-by-term rescaling below to cancel factors of
:math:`\tfrac{1}{2}` that appear in the anisotropy and demagnetization prefactors.

Exchange energy
~~~~~~~~~~~~~~~

Substituting the scaling relations :eq:`eq:star_scales_energy` into
:eq:`eq:energy_exchange` and dividing by :math:`A_1\ell_\mathrm{ex1}` gives

.. math::
    :label: eq:Eex_star

    \mathcal{E}_\text{ex}^\ast
    = \int_{\Omega_\mathrm{m}} A^\ast\,|\nabla^\ast\mathbf{m}|^2\,dV^\ast.

Anisotropy energy
~~~~~~~~~~~~~~~~~

Writing :math:`K_1 = K^\ast K_{11}` and substituting
:math:`K_{11}=\tfrac{1}{2}\mu_0 M_\mathrm{s1}^2 Q_1` together with the identity
:eq:`eq:volume_factor` cancels the factor :math:`\tfrac{1}{2}`, yielding

.. math::
    :label: eq:Eani_star

    \mathcal{E}_\text{ani}^\ast
    = \int_{\Omega_\mathrm{m}} K^\ast Q_1\,f_\mathrm{ani}\,dV^\ast.

Zeeman energy
~~~~~~~~~~~~~

Writing :math:`\mathbf{H}_\text{zee} = M_\mathrm{s1}\mathbf{H}_\text{zee}^\ast` and
applying :eq:`eq:volume_factor` gives

.. math::
    :label: eq:Ezee_star

    \mathcal{E}_\text{zee}^\ast
    = -2\int_{\Omega_\mathrm{m}} M_\mathrm{s}^\ast\,
    \mathbf{m}\cdot\mathbf{H}_\text{zee}^\ast\,dV^\ast.

Demagnetization energy
~~~~~~~~~~~~~~~~~~~~~~

The SI prefactor :math:`\tfrac{\mu_0}{2}` in :eq:`eq:energy_demagnetization`, combined
with the identity :eq:`eq:volume_factor` applied to the volume element and
:math:`\mathbf{H}_\text{d} = M_\mathrm{s1}\mathbf{H}_\text{d}^\ast`, yields a unit
prefactor:

.. math::
    :label: eq:Ed_star

    \mathcal{E}_\text{d}^\ast
    = -\int_{\Omega_\mathrm{m}} M_\mathrm{s}^\ast\,
    \mathbf{m}\cdot\mathbf{H}_\text{d}^\ast\,dV^\ast.

Total non-dimensional energy
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Summing the four non-dimensional contributions gives

.. math::
    :label: eq:Estar_heterogeneous

    \mathcal{E}^\ast
    = \int_{\Omega_\mathrm{m}} \Bigl[
      A^\ast|\nabla^\ast\mathbf{m}|^2
    + K^\ast Q_1\,f_\mathrm{ani}
    - 2M_\mathrm{s}^\ast\,\mathbf{m}\cdot\mathbf{H}_\text{zee}^\ast
    - M_\mathrm{s}^\ast\,\mathbf{m}\cdot\mathbf{H}_\text{d}^\ast
    \Bigr]dV^\ast.

For the homogeneous reference material (:math:`M_\mathrm{s}^\ast=A^\ast=K^\ast=1`), this
reduces to

.. math::
    :label: eq:Estar_homogeneous

    \mathcal{E}^\ast
    = \int_{\Omega_\mathrm{m}}\Bigl[
    |\nabla^\ast\mathbf{m}|^2
    + Q_1\,f_\mathrm{ani}
    - 2\,\mathbf{m}\cdot\mathbf{H}_\text{zee}^\ast
    - \mathbf{m}\cdot\mathbf{H}_\text{d}^\ast
    \Bigr]dV^\ast.

Stored energy :math:`\mathcal{E}_\text{sol}`
--------------------------------------------

The non-dimensional energy :math:`\mathcal{E}^\ast` is the thermodynamically correct
measure of the system energy: its variational derivative with respect to :math:`\mathbf{m}`
yields :math:`\mathbf{H}_\text{eff}` via :eq:`eq:heff_definition`, and its decrease
characterizes relaxation toward equilibrium.  The implicit time integrator in ``Z-mag``,
however, does not assemble :math:`\mathcal{E}^\ast` as its primary working quantity.
Instead, the weak form of the LLG equation (derived in :doc:`theory_dynamics`) is built
from the Gateaux derivative of an auxiliary functional, the *stored energy*

.. math::
    :label: eq:Esol_def_energy

    \mathcal{E}_\text{sol}
    = \int_{\Omega_\mathrm{m}} W(\mathbf{m},\varphi)\,dV^\ast,

where :math:`W` is a non-dimensional volume density.  The density :math:`W` is determined
by the requirement that

.. math::
    :label: eq:Esol_variation_energy

    \delta\mathcal{E}_\text{sol}[\mathbf{v}]
    = \mathcal{G}[\mathbf{v}]
    \qquad\forall\,\mathbf{v},

where :math:`\mathcal{G}[\mathbf{v}]` is the energy torque functional entering the
chart residual (see :doc:`theory_dynamics`).

The notation :math:`\delta f[\mathbf{v}]` denotes the *Gateaux derivative* of a
functional :math:`f` at the current state :math:`\mathbf{m}` in the direction
:math:`\mathbf{v}`, defined as

.. math::
    :label: eq:gateaux_def

    \delta f[\mathbf{v}]
    \equiv \lim_{\varepsilon\to 0}
    \frac{f(\mathbf{m}+\varepsilon\mathbf{v}) - f(\mathbf{m})}{\varepsilon}
    = \left.\frac{d}{d\varepsilon}
    f(\mathbf{m}+\varepsilon\mathbf{v})\right|_{\varepsilon=0}.

The test directions :math:`\mathbf{v}` belong to the Sobolev space
:math:`H^1(\Omega_\mathrm{m})^3`, the space of vector fields on
:math:`\Omega_\mathrm{m}` whose components and their first-order partial derivatives are
square-integrable.  This regularity is the minimal requirement for the exchange energy
:math:`\int A|\nabla\mathbf{m}|^2\,dV` to be finite.

The following subsections compute :math:`\delta e_i[\mathbf{v}]` for each energy
contribution :math:`e_i` and identify the stored density :math:`W_i` such that
:math:`\delta W_i[\mathbf{v}] = \delta e_i[\mathbf{v}]`.

Exchange energy
~~~~~~~~~~~~~~~

The non-dimensional exchange integrand from :eq:`eq:Eex_star` is
:math:`e_\text{ex} = A^\ast|\nabla^\ast\mathbf{m}|^2`.  Applying the definition
:eq:`eq:gateaux_def` with :math:`f(\mathbf{m}) = A^\ast|\nabla^\ast\mathbf{m}|^2`:

.. math::

    \delta e_\text{ex}[\mathbf{v}]
    = \left.\frac{d}{d\varepsilon}
      A^\ast|\nabla^\ast(\mathbf{m}+\varepsilon\mathbf{v})|^2
      \right|_{\varepsilon=0}
    = \left.\frac{d}{d\varepsilon}
      A^\ast\bigl(|\nabla^\ast\mathbf{m}|^2
      + 2\varepsilon\,\nabla^\ast\mathbf{m}:\nabla^\ast\mathbf{v}
      + \varepsilon^2|\nabla^\ast\mathbf{v}|^2\bigr)
      \right|_{\varepsilon=0}
    = 2A^\ast\,\nabla^\ast\mathbf{m}:\nabla^\ast\mathbf{v}.

The stored density :math:`W_\text{ex}` satisfying
:math:`\delta W_\text{ex}[\mathbf{v}] = \delta e_\text{ex}[\mathbf{v}]` is

.. math::
    :label: eq:W_ex_energy

    W_\mathrm{ex} = \tfrac{1}{2}A^\ast\,|\nabla^\ast\mathbf{m}|^2.

Verification: :math:`\delta W_\text{ex}[\mathbf{v}]
= \left.\tfrac{d}{d\varepsilon}\tfrac{1}{2}A^\ast|\nabla^\ast(\mathbf{m}
+\varepsilon\mathbf{v})|^2\right|_{\varepsilon=0}
= A^\ast\nabla^\ast\mathbf{m}:\nabla^\ast\mathbf{v}
= \delta e_\text{ex}[\mathbf{v}]`. :math:`\checkmark`

Anisotropy energy
~~~~~~~~~~~~~~~~~

The non-dimensional anisotropy integrand from :eq:`eq:Eani_star` is
:math:`e_\text{ani} = K^\ast Q_1 f_\mathrm{ani}`, where
:math:`f_\mathrm{ani} = 1-(\mathbf{m}\cdot\mathbf{e}_\text{ua})^2`.
Applying :eq:`eq:gateaux_def`:

.. math::

    \delta e_\text{ani}[\mathbf{v}]
    = \left.\frac{d}{d\varepsilon}
      K^\ast Q_1\bigl[1-((\mathbf{m}+\varepsilon\mathbf{v})
      \cdot\mathbf{e}_\text{ua})^2\bigr]
      \right|_{\varepsilon=0}
    = \left.\frac{d}{d\varepsilon}
      K^\ast Q_1\bigl[1 - (\mathbf{m}\cdot\mathbf{e}_\text{ua})^2
      - 2\varepsilon(\mathbf{m}\cdot\mathbf{e}_\text{ua})(\mathbf{v}\cdot\mathbf{e}_\text{ua})
      - \varepsilon^2(\mathbf{v}\cdot\mathbf{e}_\text{ua})^2\bigr]
      \right|_{\varepsilon=0}
    = -2K^\ast Q_1\,(\mathbf{m}\cdot\mathbf{e}_\text{ua})(\mathbf{v}\cdot\mathbf{e}_\text{ua}).

The stored density satisfying
:math:`\delta W_\text{ani}[\mathbf{v}] = \delta e_\text{ani}[\mathbf{v}]` is

.. math::
    :label: eq:W_ani_energy

    W_\mathrm{ani} = \tfrac{1}{2}K^\ast Q_1\,f_\mathrm{ani}.

Verification: :math:`\delta W_\text{ani}[\mathbf{v}]
= \left.\tfrac{d}{d\varepsilon}\tfrac{1}{2}K^\ast Q_1
[1-((\mathbf{m}+\varepsilon\mathbf{v})\cdot\mathbf{e}_\text{ua})^2]
\right|_{\varepsilon=0}
= -K^\ast Q_1(\mathbf{m}\cdot\mathbf{e}_\text{ua})(\mathbf{v}\cdot\mathbf{e}_\text{ua})
= \delta e_\text{ani}[\mathbf{v}]`. :math:`\checkmark`

Zeeman energy
~~~~~~~~~~~~~

The non-dimensional Zeeman integrand from :eq:`eq:Ezee_star` is
:math:`e_\text{zee} = -2M_\mathrm{s}^\ast\,\mathbf{m}\cdot\mathbf{H}_\text{zee}^\ast`,
with :math:`\mathbf{H}_\text{zee}^\ast` a prescribed, :math:`\mathbf{m}`-independent
field.  Applying :eq:`eq:gateaux_def`:

.. math::

    \delta e_\text{zee}[\mathbf{v}]
    = \left.\frac{d}{d\varepsilon}
      \bigl[-2M_\mathrm{s}^\ast(\mathbf{m}+\varepsilon\mathbf{v})
      \cdot\mathbf{H}_\text{zee}^\ast\bigr]
      \right|_{\varepsilon=0}
    = -2M_\mathrm{s}^\ast\,\mathbf{v}\cdot\mathbf{H}_\text{zee}^\ast.

The stored density satisfying
:math:`\delta W_\text{zee}[\mathbf{v}] = \delta e_\text{zee}[\mathbf{v}]` is

.. math::
    :label: eq:W_zee_energy

    W_\mathrm{zee} = -M_\mathrm{s}^\ast\,\mathbf{m}\cdot\mathbf{H}_\text{zee}^\ast.

Verification: :math:`\delta W_\text{zee}[\mathbf{v}]
= \left.\tfrac{d}{d\varepsilon}[-M_\mathrm{s}^\ast
(\mathbf{m}+\varepsilon\mathbf{v})\cdot\mathbf{H}_\text{zee}^\ast]
\right|_{\varepsilon=0}
= -M_\mathrm{s}^\ast\,\mathbf{v}\cdot\mathbf{H}_\text{zee}^\ast
= \delta e_\text{zee}[\mathbf{v}]`. :math:`\checkmark`

The factor of two in :eq:`eq:Ezee_star` originates from the non-dimensionalization
scaling :eq:`eq:volume_factor`, not from a quadratic structure; the Zeeman functional is
linear in :math:`\mathbf{m}` and its Gateaux derivative yields the correct Zeeman
torque :math:`-M_\mathrm{s}^\ast\mathbf{H}_\text{zee}^\ast`, consistent with
:eq:`eq:heff_complete`.

Demagnetization energy
~~~~~~~~~~~~~~~~~~~~~~

The non-dimensional demagnetization integrand from :eq:`eq:Ed_star` is
:math:`e_\text{d} = -M_\mathrm{s}^\ast\,\mathbf{m}\cdot\mathbf{H}_\text{d}^\ast`.  In
the operator-splitting scheme implemented in ``Z-mag``, the scalar potential :math:`\varphi`
is treated as a field separate from :math:`\mathbf{m}` and held fixed during the
variation of the magnetization; accordingly, :math:`\mathbf{H}_\text{d}^\ast` is treated
as :math:`\mathbf{m}`-independent in this derivative.  Applying :eq:`eq:gateaux_def`:

.. math::

    \delta e_\text{d}[\mathbf{v}]
    = \left.\frac{d}{d\varepsilon}
      \bigl[-M_\mathrm{s}^\ast(\mathbf{m}+\varepsilon\mathbf{v})
      \cdot\mathbf{H}_\text{d}^\ast\bigr]
      \right|_{\varepsilon=0}
    = -M_\mathrm{s}^\ast\,\mathbf{v}\cdot\mathbf{H}_\text{d}^\ast.

The stored density satisfying
:math:`\delta W_\text{d}[\mathbf{v}] = \delta e_\text{d}[\mathbf{v}]` is

.. math::
    :label: eq:W_demag_energy

    W_\mathrm{d} = -\tfrac{1}{2}M_\mathrm{s}^\ast\,\mathbf{m}\cdot\mathbf{H}_\mathrm{d}^\ast.

Verification: :math:`\delta W_\text{d}[\mathbf{v}]
= \left.\tfrac{d}{d\varepsilon}[-\tfrac{1}{2}M_\mathrm{s}^\ast
(\mathbf{m}+\varepsilon\mathbf{v})\cdot\mathbf{H}_\text{d}^\ast]
\right|_{\varepsilon=0}
= -\tfrac{1}{2}M_\mathrm{s}^\ast\,\mathbf{v}\cdot\mathbf{H}_\text{d}^\ast
= \delta e_\text{d}[\mathbf{v}]`. :math:`\checkmark`

The factor :math:`\tfrac{1}{2}` in :math:`W_\text{d}` reflects the variational structure
of the operator-splitting scheme; it is distinct from the self-interaction factor
:math:`\tfrac{1}{2}` in the dimensional energy :eq:`eq:energy_demagnetization`, which
accounts for double-counting.

Relation between :math:`\mathcal{E}_\text{sol}` and :math:`\mathcal{E}^\ast`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Comparing the four stored densities
:math:`W = W_\mathrm{ex}+W_\mathrm{ani}+W_\mathrm{zee}+W_\mathrm{d}`
with the corresponding integrands of :math:`\mathcal{E}^\ast`, each component satisfies

.. math::

    W_i = \tfrac{1}{2}\,e_i,
    \qquad i \in \{\text{ex},\,\text{ani},\,\text{zee},\,\text{d}\},

where :math:`e_i` denotes the :math:`i`-th integrand of :math:`\mathcal{E}^\ast`.  For
the homogeneous reference material (:math:`M_\mathrm{s}^\ast=A^\ast=K^\ast=1`) this yields

.. math::
    :label: eq:Esol_vs_Estar

    \mathcal{E}^\ast = 2\,\mathcal{E}_\text{sol}.

This identity holds exactly at every configuration; it is a direct consequence of the
quadratic and bilinear structure of all four energy contributions and of the linear
relationship :math:`\delta W_i[\mathbf{v}] = \tfrac{1}{2}\delta e_i[\mathbf{v}]`
established term by term above.

In practice, ``Z-mag`` monitors
:math:`\mathcal{E}_\text{sol}` at each accepted time step, so the reported value must be
multiplied by two to recover the physical non-dimensional energy :math:`\mathcal{E}^\ast`.
The complete derivation of the weak form, the definition of the chart residual
:math:`\mathcal{G}`, and the role of :math:`\mathcal{E}_\text{sol}` in the two-chart
atlas are developed in :doc:`theory_dynamics`.
