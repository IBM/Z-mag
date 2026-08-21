import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
import ufl
from dolfinx import fem, mesh, cpp
import dolfinx.fem.petsc


from functools import partial

from bem_assembler.bem_assembler import format_bytes

print_flush = partial(print, flush=True)

class Z_sim_magnetostatic:
    """
    Solve the magnetostatic problem via the finite element / boundary element method (FEM/BEM) by Fredkin & Koehler.
    
    This class computes the magnetostatic scalar potential `phi` given a magnetization field `mag` via the Fredkin-Koehler split on the magnetic mesh:

    - a Neumann (Poisson) subproblem for `phi0` in the magnetized region
    - a Laplace subproblem for `phi1` on the same mesh, with Dirichlet data on the exterior boundary supplied by the BEM matrix

    The exterior (vacuum) field is encoded by BEM; no vacuum FEM mesh is used here.

    The resulting scalar potential satisfies Δφ = ∇·M in Ω_m (weak form) with μ₀(M+H) constitutive law.       
    """

    def __init__(self, domain: mesh.Mesh, facet_tags: cpp.mesh.MeshTags_int32, interfaces: list[dict], surface_marker: int, dx: ufl.measure.Measure, ds: ufl.measure.Measure, h_zee: ufl.tensors.ListTensor, Ms: fem.Function):
        """ 
        Initialize the Demag object.

        Parameters
        ----------
        domain: mesh.Mesh
            Mesh of the magnetic domain
        facet_tags: cpp.mesh.MeshTags_int32
            Mesh tags for the boundary surface
        interfaces: list[dict]
            Interfaces between magnetic materials
        surface_marker: int
            Marker of the boundary of the magnetic material
        dx: ufl.measure.Measure
            Volume measure of the magnetic domain
        ds: ufl.measure.Measure
            External boundary of the magnetic domain
        h_zee: ufl.tensors.ListTensor
            Non-dimensional external magnetic field
        Ms: dolfinx.fem.Function
            Non-dimensional saturation magnetization
        """

        self.domain = domain
        self.facet_tags = facet_tags
        self.interfaces = interfaces
        self.bnd_mrk = surface_marker
        self.dv, self.ds = dx, ds
        self.normal = ufl.FacetNormal(self.domain)
        self.hzeeman = h_zee
        self.Ms = Ms
        
        # Create function space and functions for magnetostatic potentials
        self.V_phi = fem.functionspace(self.domain,("CG", 1))
        self.phi = fem.Function(self.V_phi, name="magnetic_potential")
        self.phi0 = fem.Function(self.V_phi)
        self.phi1 = fem.Function(self.V_phi)
        self.phi1_boundary = fem.Function(self.V_phi)
        self.phi_test, self.phi_trial = ufl.TestFunction(self.V_phi), ufl.TrialFunction(self.V_phi)
        
        # Identify local DOFs belonging to the boundary facets marked with 'surface_marker'
        self.boundary_dofs = fem.locate_dofs_topological(self.V_phi, self.domain.topology.dim-1, facet_tags.find(surface_marker))
        
        # Note that self.boundary_dofs might be different when the matrix is computed versus loaded
        # dof_coordinates = self.V_phi.tabulate_dof_coordinates()[self.boundary_dofs]
        # print('Rank = {}, self.boundary_dofs = {}\n\n'.format(self.domain.comm.rank, self.boundary_dofs))
        
        # Pre-assemble bilinear form for scalar potential
        self.a = ufl.dot(ufl.grad(self.phi_trial),ufl.grad(self.phi_test))*self.dv
        
        # Pre-compute zero RHS in Laplace problem
        zero_scalar_constant = fem.Constant(domain, PETSc.ScalarType(0.0))
        self.f_zero = zero_scalar_constant*self.phi_test*self.dv
                
        # Pre-compute interface measures
        self.interface_measures = {}
        self.interface_checks = {}
        for interface in self.interfaces:
            if 'tag' in interface:
                inter_tag = interface['tag']
                dS_tag = ufl.Measure('dS', self.domain, subdomain_data=self.facet_tags, subdomain_id=inter_tag, metadata={'quadrature_degree': 4})
                self.interface_measures[inter_tag] = dS_tag
                
                # Check surface tags, the integral is equal to 0 if they are not correctly assigned
                check_surface = self.domain.comm.allreduce(fem.assemble_scalar(fem.form(1*dS_tag)), op=MPI.SUM)
                self.interface_checks[inter_tag] = bool(check_surface)
                assert self.interface_checks[inter_tag], f'Surface tag {inter_tag} not assigned correctly'
            else:
                raise Exception('Provide key tag in the interface dictionary')
        
        # BEM MPI buffers will be initialized once after BEM is assembled
        self.bem_cache_initialized = False
        
        # Set PETSc solver options
        self.set_solvers_opts()
    
    
    def initialize_bem_cache(self):
        """Preallocate MPI buffers for ``BEM @ phi_boundary`` (after matrix assembly)."""
        assert all(hasattr(self, attr) for attr in ['global_n_bnd_nodes', 'BEM', 'bnd_nodes']), "BEM needs to be assembled before initializing cache"

        # MPI vector lengths for gathering boundary phi0 values across ranks.
        self.global_recvcounts = np.array(self.global_n_bnd_nodes, dtype='i')
        self.global_displs = np.zeros(self.domain.comm.size, dtype='i')
        if self.domain.comm.size > 1:
            self.global_displs[1:] = np.cumsum(self.global_recvcounts[:-1])

        # phi0: all global boundary values; phi1: this rank's portion of BEM * phi0.
        self.global_phi0_bnd_buffer = np.empty(sum(self.global_n_bnd_nodes), dtype=np.float64)
        self.global_phi1_bnd_buffer = np.empty(self.n_bnd_nodes, dtype=np.float64)

        # Flag initialization as completed and print required cache
        self.bem_cache_initialized = True
        
        # Print required cache
        bem_bytes = self.BEM.nbytes
        phi0_bytes = self.global_phi0_bnd_buffer.nbytes
        phi1_bytes = self.global_phi1_bnd_buffer.nbytes
        
        if self.domain.comm.rank == 0:
            total_bytes = bem_bytes + phi0_bytes + phi1_bytes
            print_flush("\n" + "=" * 65)
            print_flush(f"{' BEM cache initialized ':=^65}")
            print_flush(
                f"BEM matrix (rank 0 local block): {self.BEM.shape} = "
                f"{format_bytes(bem_bytes)}"
            )
            print_flush(
                f"phi0 buffer (global, all ranks): {self.global_phi0_bnd_buffer.shape} = "
                f"{format_bytes(phi0_bytes)}"
            )
            print_flush(
                f"phi1 buffer (rank 0 local):      {self.global_phi1_bnd_buffer.shape} = "
                f"{format_bytes(phi1_bytes)}"
            )
            print_flush(f"Total BEM workspace (rank 0):    {format_bytes(total_bytes)}")
            print_flush("=" * 65 + "\n")
    
    
    def set_solvers_opts(self):
        """
        Set Neumann and Dirichlet solvers options.
        """
        # LinearProblem default options
        # KSP: GMRES (restarted, restart=30)
        # PC:  ILU(0) preconditioner
        # RTOL: 1e-5 
        # Max iterations: 10000


        # Neumann solver
        self.neumann_options = {
            "ksp_type": "gmres",
            "ksp_rtol": 1e-6,
            "ksp_max_it": 2000,
            "pc_type": "bjacobi",
            "sub_pc_type": "ilu",
            "ksp_gmres_restart": 30
        }
        
        # Dirichlet solver
        self.dirichlet_options = {
            "ksp_type": "cg",
            "ksp_max_it": 100,
            "ksp_rtol": 1e-10,
            "pc_type": "hypre",
            "pc_hypre_type": "boomeramg"
        }
    
    

    def _build_neumann_rhs(self, mag):
        """
        Build RHS for Neumann problem.
        """
        # Update the linear form f for the Neumann problem on phi0
        # -) volume charge term
        # -) surface charge term on magnetic surface
        # -) external field contribution on surface
        f = -self.Ms*ufl.div(mag)*self.phi_test*self.dv
        f += self.Ms*ufl.dot(mag, self.normal)*self.phi_test*self.ds
        f += ufl.dot(self.hzeeman, self.normal)*self.phi_test*self.ds
        
        # Interface contributions inside the domain
        # ('+') is the outward normal on the interface dS
        # ('-') is the inward normal on the interface dS
        interface_terms = sum(
            (ufl.dot(self.Ms('+')*mag('+'), self.normal('+'))*self.phi_test('+') - ufl.dot(self.Ms('-')*mag('-'), self.normal('+'))*self.phi_test('-')) * dS
            for dS in self.interface_measures.values()
        )
        f += interface_terms
        return f
    
    


    """ 
    Solve the magnetostatic problem on the magnetic potential
    """
    def compute_phi(self, mag: fem.Function):
        r"""
        Compute magnetic scalar potential :math:`\phi` from magnetization :math:`\mathbf{m}`.

        Uses the Fredkin-Koehler FEM/BEM hybrid method.

        Parameters
        ----------
        mag : dolfinx.fem.Function
            The magnetization of the magnetic material.

        Returns
        -------
        phi : dolfinx.fem.Function
            The potential of the magnetization field.

        Notes
        -----
        The BEM matrix is stored in blocks of size (len(self.global_boundary_dofs), len(self.global_total_boundary_dofs)):

        - Each MPI rank stores a subset of the rows corresponding to the boundary DOFs that it owns.
        - Each subset of rows has a number of columns corresponding to the total number of global boundary DOFs.
        - Each MPI rank multiplies its own block of the BEM matrix with the full global boundary vector.
        - The result is distributed, in that each rank owns its portion of the output boundary values.
        - MPI_Allgatherv: assemble global boundary vector [rank0|rank1|...|rankN]

        Matrix layout visualization::

                        Global matrix               Global phi0 vector              Local output
                    |----------------------|
                    |  BEM_0  (Rank 0)     |                                    |------------------|
                    |----------------------|                                    | phi1_0  (Rank 0) |
            BEM   = |  BEM_1  (Rank 1)     |   x   global_phi0_boundary   =     | phi1_1  (Rank 1) |
                    |----------------------|                                    | phi1_2  (Rank 2) |
                    |  BEM_2  (Rank 2)     |                                    |------------------|
                    |----------------------|
        
        """
        # Check if MPI BEM buffers have been initialized
        assert self.bem_cache_initialized

        # --- Solve Neumann problem for phi0
        # Update the linear form f for the Neumann problem on phi0
        f = self._build_neumann_rhs(mag)

        # Pure Neumann Poisson has a constant nullspace. Without pinning it,
        # GMRES may return a different additive gauge on each call (especially
        # under more MPI partitions). BEM compensates only approximately, so
        # raw ||φ^{k+1}-φ^k|| can look O(0.1) while H_d=-∇φ stays fine.
        if not hasattr(self, "_neumann_nullspace"):
            self._neumann_nullspace = PETSc.NullSpace().create(
                constant=True, comm=self.domain.comm
            )

        # Solve Neumann subproblem phi0 (reuse Function for stable gauge)
        self.linear_problem = fem.petsc.LinearProblem(
            self.a,
            f,
            bcs=[],
            u=self.phi0,
            petsc_options_prefix="zmag_ms_neumann_",
            petsc_options=self.neumann_options,
        )
        self.linear_problem.A.setNullSpace(self._neumann_nullspace)
        self.linear_problem.A.setNearNullSpace(self._neumann_nullspace)
        self.phi0 = self.linear_problem.solve()
        self.phi0.x.scatter_forward()
        

        # --- BEM coupling
        # Extract locally-owned boundary DOFs, its length must match self.global_n_bnd_nodes[rank] from BEM assembly
        phi0_bnd_local = self.phi0.x.array[self.bnd_nodes[:self.n_bnd_nodes]]
        assert len(phi0_bnd_local) == self.global_n_bnd_nodes[self.domain.comm.rank], (f'Rank {self.domain.comm.rank}: phi0_bnd_local size = {len(phi0_bnd_local)} vs expected {self.global_n_bnd_nodes[self.domain.comm.rank]}')
        
        # Assemble the global phi0_boundary vector: each rank sends phi0_bnd_local, receives full global boundary vector
        self.domain.comm.Allgatherv(
            sendbuf=phi0_bnd_local, 
            recvbuf=[self.global_phi0_bnd_buffer, self.global_recvcounts, self.global_displs, MPI.DOUBLE]
        )


        # --- Solve Laplace problem for phi1
        # Compute Dirichlet boundary condition for phi
        self.global_phi1_bnd_buffer[:] = np.dot(self.BEM, self.global_phi0_bnd_buffer)
        self.phi1_boundary.x.array[self.bnd_nodes[:self.n_bnd_nodes]] = self.global_phi1_bnd_buffer
        self.phi1_boundary.x.scatter_forward()
                
        # Apply Dirichlet BC on phi1 from BEM boundary values
        dbc_phi1 = fem.dirichletbc(self.phi1_boundary, self.boundary_dofs)
        
        # Solve Laplace subproblem phi1 with Dirichlet BC
        self.linear_problem = fem.petsc.LinearProblem(
            self.a,
            self.f_zero,
            bcs=[dbc_phi1],
            u=self.phi1,
            petsc_options_prefix="zmag_ms_dirichlet_",
            petsc_options=self.dirichlet_options,
        )
        self.phi1 = self.linear_problem.solve()
        self.phi1.x.scatter_forward()


        # Compute magnetic potential phi = phi0 + phi1
        self.phi.x.array[:] = self.phi0.x.array + self.phi1.x.array
        self.phi.x.scatter_forward()