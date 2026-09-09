---
title: 'Z-mag: MPI-parallel finite-element micromagnetics with a two-chart stereographic atlas'
tags:
  - micromagnetics
  - Landau-Lifshitz-Gilbert
  - stereographic projection
  - finite element method
  - FEniCSx
  - DOLFINx
  - PETSc
  - boundary element method
authors:
  - name: Stella Brach
    orcid: 0000-0003-4766-8131
    corresponding: true
    affiliation: 1
  - name: Federico Soldà
    affiliation: 1
  - name: Kathrin Aebischer
    orcid: 0000-0001-7825-3655
    affiliation: 1
  - name: Mark A. Lantz
    orcid: 0000-0003-3705-3352
    affiliation: 1
affiliations:
  - name: IBM Research, Säumerstrasse 4, 8803 Rüschlikon
    index: 1
date: 12 August 2026
bibliography: paper.bib
---

# Summary

`Z-mag` is an open-source Python library for time-dependent micromagnetic
simulation on unstructured tetrahedral meshes [@zmag_github]. In continuum
micromagnetics, the magnetization $\mathbf{m}$ of a ferromagnet is a unit-vector
field on the unit sphere $\mathcal{S}^2$, whose evolution is governed by the
Landau–Lifshitz–Gilbert (LLG) equation [@gilbert2004phenomenological]. Accurate
long-time integration of that equation requires the pointwise
constraint $|\mathbf{m}|=1$ to be maintained while resolving stiff dynamics
across a wide range of physical regimes. That capability is essential for interpreting switching,
relaxation, and field-driven behaviour in nanoscale devices used in spintronics,
magnetic sensing, and magnetic recording. `Z-mag` is intended for researchers
who need geometric flexibility—curved boundaries, multi-material junctions, and
local mesh refinement—together with reliable preservation of $|\mathbf{m}|=1$
over stiff LLG trajectories across the full range of Gilbert damping.

The library is built on DOLFINx/FEniCSx [@baratta2023dolfinx] and PETSc
[@petsc-web-page] and runs under the Message Passing Interface (MPI). Its
central numerical feature is a two-chart stereographic atlas: $\mathbf{m}$ is
represented by a pair of planar chart fields $q_A$ and $q_B$ whose inverse maps
return unit vectors for every finite coordinate value, so $|\mathbf{m}|=1$ is
intrinsic to the parametrization and needs no renormalization, penalty, or
Lagrange multiplier. A factory exposes two solver variants. The `demag` variant
couples LLG dynamics to open-boundary magnetostatics; the `nodemag` variant
omits magnetostatics for exchange- and anisotropy-dominated problems. Both
support exchange, uniaxial and cubic anisotropy, and Zeeman coupling, and both
write checkpoint bundles that survive arbitrary MPI repartitioning.

# Statement of need

Preserving $|\mathbf{m}|=1$ over long LLG integrations is a recurring difficulty
in computational micromagnetics [@abert2019micromagnetics; @kruzik2006recent].
Cartesian formulations typically restore the unit norm by post-step
renormalization, which can introduce directional drift; by penalty terms, which
add artificial stiffness; or by Lagrange multipliers, which lead to saddle-point
algebraic systems. Intrinsic coordinates avoid those mechanisms but bring their
own singularities: spherical (polar) angles fail at the poles, and a single
stereographic chart becomes poorly scaled near its projection pole
[@horley2009application; @alouges2012convergent].

`Z-mag` maintains two complementary stereographic charts with continuous
blending, so that at least one chart remains well conditioned for every
magnetization direction. It is intended for researchers who need
constraint-exact finite-element dynamics on geometrically complex domains. The
mathematical formulation, variational analysis, and numerical verification are
developed in a companion methods paper [@brach2026zmag]. The present
contribution releases that framework as reusable research software: a
documented, MPI-parallel implementation with regression drivers and a public
development history under an OSI-approved license.

# State of the field

Computational micromagnetics relies predominantly on two spatial
discretizations. Finite-difference methods, as in `OOMMF` [@donahue1999oommf],
`MuMax3` [@vansteenkiste2014design], and the `Ubermag` workflow [@beg2022ubermag],
combine regular grids with fast Fourier transform (FFT)-accelerated
magnetostatics and are efficient on rectilinear domains. Finite-element methods,
as in `magnum.fe` [@abert2013magnum], `FeeLLGood` [@kritsikis2014beyond], and
`tetmag` [@tetmag], support unstructured meshes, curved boundaries, and heterogeneous
media, at the cost of denser magnetostatic operators. In both classes the
magnetization is ordinarily represented by Cartesian components, with
$|\mathbf{m}|=1$ restored by auxiliary procedures after each step.

Stereographic coordinates have appeared in more specialized settings. Horley
*et al.* [-@horley2009application; -@horley2011numerical] used a single
stereographic chart with threshold-based pole switching for macrospin dynamics.
Rybakov *et al.* [-@rybakov2015bobber] employed adaptive chart redefinition
during energy minimization of chiral magnetic textures. Those constructions
demonstrate the usefulness of intrinsic charts, but they do not constitute a
general finite-element platform for time-dependent LLG dynamics with
open-boundary magnetostatics and MPI-parallel execution.

`Z-mag` supplies that platform as fully open research software under the MIT
license, with a public repository, pinned environments, and regression drivers
[@zmag_github]. Its dual-chart stereographic atlas with continuous blending
makes $|\mathbf{m}|=1$ intrinsic to the discrete state, while unstructured
tetrahedral meshes provide geometric flexibility beyond FFT-based finite-
difference grids. Alongside open packages, proprietary micromagnetic solvers
remain widely used in industrial practice; the present release makes a
constraint-exact, MPI-parallel finite-element LLG stack freely available for
inspection, reuse, and extension.

# Software design

The public interface is organized around a single factory,
`create_stereo_solver(..., model="demag"|"nodemag")`, which returns a thin
wrapper over a shared dynamic base class. Both variants share the same setup
path—mesh and material markers, selection of local energy contributions,
initialization of the chart fields from Cartesian or angular data, adaptive
time integration, and checkpointing—while magnetostatics is confined to
`demag`. Materials are specified in a nondimensional setting relative to a
reference region, which simplifies multi-material studies without changing the
discrete atlas representation.

The central architectural decision is to evolve both stereographic charts
simultaneously and to blend their contributions with cell-wise weights, rather
than switching charts abruptly when a pole is approached. Continuous blending
keeps the discrete operators smooth and avoids discontinuous jumps in the
residual. The chart fields $q_A$ and $q_B$ belong to continuous piecewise-linear
vector spaces $[\mathrm{CG}_1]^2$; the blending weights belong to a discontinuous
piecewise-constant $\mathrm{DG}_0$ space, so that inactive regions can drop out
of the residual at low cost. The magnetization that enters local energies is the
cell-wise blend of the two charts.

The `demag` and `nodemag` models differ in the finite-element unknown associated
with magnetostatics. In both cases the LLG residual is posed in the atlas
unknown $(q_A,q_B)$ on $[\mathrm{CG}_1]^2\times[\mathrm{CG}_1]^2$. When
open-boundary magnetostatics is required, the demagnetizing field is written
$\mathbf{H}_d=-\nabla\varphi$ in terms of a demagnetizing scalar potential
$\varphi$, which enters the effective field nonlocally. The `demag` variant
therefore enlarges the unknown to $(q_A,q_B,\varphi)$ on
$[\mathrm{CG}_1]^2\times[\mathrm{CG}_1]^2\times\mathrm{CG}_1$. The potential is
computed from the Fredkin–Koehler two-potential decomposition with a hybrid
finite-element/boundary-element (FEM–BEM) treatment of the unbounded exterior
[@fredkin1990hybrid]. Differentiating the dense BEM closure with respect to the
chart fields inside every nonlinear iteration would be prohibitively expensive,
so magnetization and potential updates are operator-split in an outer
fixed-point loop, optionally accelerated by Anderson mixing. Each inner solve
is a Newton–Krylov iteration on a sparse finite-element Jacobian in which
$\varphi$ is held at the frozen BEM reference for that outer iterate. The
`nodemag` variant is the restriction of this formulation when magnetostatic
self-interaction is negligible relative to exchange, anisotropy, or Zeeman
driving: $\varphi$ is omitted, the BEM operator is never assembled, and the
residual reduces to the atlas blocks alone. Selectable local energies are
chosen independently of this magnetostatic branch. In `demag`, the dense BEM
operator is assembled once per mesh partition by a compiled C++ kernel that
reads contiguous NumPy buffers without intermediate copies and writes the local
dense block in place; the result can be serialized for reuse on the same
partition.

Time integration uses variable-step backward differentiation formulae
(BDF1/BDF2). Each proposed step is subjected to a multi-phase acceptance
pipeline that combines algebraic solver checks, atlas consistency guards,
residual monitoring, and a thermodynamic gate based on work–dissipation balance
before the state is committed. This layered control is deliberate: LLG dynamics
are stiff, and local truncation-error estimates alone do not prevent
nonphysical energy jumps after difficult nonlinear solves. Checkpointing stores
mesh markers, chart history, and adaptive controller state so that long HPC
runs can restart after repartitioning; in `demag`, the BEM matrix is rebuilt
when ownership changes and the potential is recomputed before the thermodynamic
baseline is reseeded. \autoref{fig:workflow} summarizes the shared workflow and
the magnetostatic branch.

![Overview of the Z-mag workflow: mesh and materials, solver factory with selectable energy terms, demag versus nodemag magnetostatic paths, dual-chart atlas evolution, adaptive BDF time stepping with acceptance gates, and MPI-safe checkpointing. Blue highlights the solver factory (API and model selection); green highlights output and restart; white boxes follow the shared numerical path; soft gray marks the demag and nodemag alternatives. The stages shown here are explained in detail in the Sphinx documentation. \label{fig:workflow}](zmag_workflow.png)

Runtime dependencies are pinned per release tag in `environment.yml`
(DOLFINx/FEniCSx, PETSc, mpi4py, NumPy/SciPy, Gmsh/meshio, and ADIOS2 tooling).
Sphinx documentation covers installation, configuration, theory notes, and API
reference. Solver modules, the BEM assembler, and regression drivers occupy
separate top-level packages, so verification cases can evolve independently of
the factory-based public interface.

# Research impact statement

The mathematical formulation and numerical verification of the dual-chart atlas
are reported in a companion methods paper [@brach2026zmag]. The present
contribution releases the corresponding open-source implementation [@zmag_github]: an
MIT-licensed, MPI-parallel finite-element code with Sphinx documentation, pinned
runtime environments, and regression drivers exercised under MPI. Separating
the methods manuscript from the software artifact yields a citable algorithmic
reference together with a maintainable code base that can be installed and
extended independently of that paper. The intended research use is
time-dependent micromagnetics on geometrically complex and multi-material
domains, where intrinsic norm preservation and unstructured finite-element
discretization are both required.

# AI usage disclosure

IBM Bob was used to assist with proofreading this manuscript and with code
review of selected modules. Bob routes tasks across several foundation models;
the authors did not pin a single model version. All AI suggestions were
examined by the authors; scientific claims, software descriptions, citations,
and accepted code changes were verified by the authors, who accept full
responsibility for the content. The numerical methods, software architecture,
and core implementation of `Z-mag` were designed and developed by the authors.

# Acknowledgements

`Z-mag` was developed in the Tape Research group at IBM Research. We thank
colleagues who provided feedback on documentation, testing workflows, and early
prototype runs.

# References
