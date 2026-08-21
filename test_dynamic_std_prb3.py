import petsc4py
import os, sys, traceback
from functools import partial
import numpy as np
from mpi4py import MPI
from dolfinx import mesh
from Zmag_bins.tools import test_wrapper, dir_wrapper
from Zmag_bins.Zmag_dynamic_stereo_solvers import create_stereo_solver

petsc4py.init(sys.argv)
comm = MPI.COMM_WORLD
print_flush = partial(print, flush=True)


# =====================
# Simulation parameters
# =====================

# General options
compute_BEM = True
first_run = True
alpha = 0.5
energy_terms = ['exchange', 'uniaxial_anisotropy', 'demag']

# Magnetization initialization (SP3 branches)
initMag = 'F'                       # V = vortex branch, F = flower branch
std = 0.0                           # Deterministic validation setup
sp3_protocol = 'branch_energy'          # 'attractor' or 'branch_energy'
if sp3_protocol not in ('attractor', 'branch_energy'):
    raise ValueError("sp3_protocol must be 'attractor' or 'branch_energy'")

# =================
# Geometry and mesh
# =================
volume_tag = 1                       # Tag cells in the magnet volume
surface_tag = 10                     # Tag facets on the magnet boundary
L = 8.0                              # Non-dimensional length
mesh_size = 0.4                      # Non-dimensional mesh size, much smaller than 1
N = int(round(L/mesh_size))          # Mesh discretization
dim = 3                              # Dimension of the domain


# ===========
# Directories
# ===========
case_dir_name = f"{sp3_protocol}_{initMag}_L_{L}_xdmf_allfields_checkunified2"
output_dir_name = 'output_std_prb3_dynamic_stereo_qAqB_final'
output_dir = os.path.join(os.getcwd(), output_dir_name, case_dir_name)
results_file_name = 'results'

# Create directory tree
dir_tree = dir_wrapper.directory_tree(comm, compute_BEM, first_run, output_dir, energy_terms)
path_dirs = dir_tree.get_directory_tree_structure()
dir_tree.create_directory_tree()


# Init simulation
args_init_sim = {    'alpha': alpha,
    'energy_terms': energy_terms,
    'NeumannBC': True,
    'printout': True,
    'path_dirs': path_dirs
}
solver_model = "demag"


# ==========================================
# Physical constants and material properties
# ==========================================

# Compulsory keys: Q1, Ms, Aex, K, K2K1_ratio, anisotropy_axis if K!=0 and uniaxial_anisotropy
if initMag in ('F', 'V'):
    # SP3 reference/ubermag setup: easy axis along z.
    anisotropy_axis = (0.0, 0.0, 1.0)
else:
    raise ValueError(f"Unsupported initMag='{initMag}'. Use 'F' or 'V'.")

materials = [
    {'name': 'material0',
     'dim': dim,
     'tag': volume_tag,
     'Q1': 0.1,
     'Ms': 1,
     'Aex': 1,
     'K': 1,
     'anisotropy_axis': anisotropy_axis,
     'K2K1_ratio': 0}
]
interfaces = []


# =====================================
# Timestepping and convergence criteria
# =====================================

args_init_dyn = {
    # Basic simulation control parameters:
    't_start': 0.0,
    'dt': 1e-4,
    'step_start': 0,
    'step_end': 6000,
    'first_run': first_run,
    'checkpoint_every': 1000,
    'write_target': ['mag','qA','qB','chiA','chiB','chiAB','wA','wB'],
    'xdmf_time_targets': ['mag','qA','qB','chiA','chiB','chiAB','wA','wB'],
    # Time integration: 1 = Backward Euler, 2 = BDF2 (auto-switches after 2 BE steps)
    'time_order': 2,
    'max_retry': 5,
    'atlas_weight_relaxation': 0.35,
    'nonlinear_residual_max_accept': float("inf"),
    # Baseline stable profile for SP3 branch/crossover runs.
    'outer_split_max_iter': 4,
    'outer_split_tol_q': 5e-3,
    'outer_split_tol_phi': 1e-3,

    # Error-controlled timestep with CFL safety cap (faster profile)
    'cfl_settings': {
        'CFL': 0.6,
        'dt_min': 1e-7,
        'dt_max': 5e-3,
        'smooth_factor': 0.7,
        'max_growth': 1.05,
        'min_shrink': 0.5,
        'error_rel_tol': 1e-2,
        'error_abs_tol': 1e-4,
        'error_headroom': 0.90,
        'max_retry_thermo': 20,
    },

    # Atlas activity controls:
    # - hysteresis_tau: activity dead-band for stable cell mask decisions
    # - atlas_overlap_tau: overlap half-width used for smooth wA/wB blending
    'chart_settings': {
        'hysteresis_tau': 0.0,
        'atlas_overlap_tau': 0.05,
    },
    # Extra diagnostics for thermodynamic rejection analysis.
    # Logs detailed breakdown on rejected attempts; accepted-step diagnostics
    # can be enabled by setting 'thermo_diag_log_accepted': True.
    'diagnostic_settings': {
        'enable_thermo_diagnostics': True,
        'thermo_diag_log_accepted': False,
    },

    # Physical convergence settings
    'convergence_settings': {
        # Size-independent physical stop:
        # ||m_dot||_rms and normalized |dE/dt|/max(|E|,1) over accepted steps.
        'physical_tol_mdot_rms': 1e-4,
        'physical_tol_edot': 3e-5,
        'steady_steps': 5,
        'stagnation_tol': 1e-4,
        'stagnation_steps': 5,
    },
}

# Minimal startup stabilization for V branch:
# avoids step-0 thermo dead-on-arrival while preserving validation settings.
vortex_startup_stabilization = (initMag == 'V')
if vortex_startup_stabilization:
    args_init_dyn['dt'] = 5e-5
    args_init_dyn.update({
        'demag_ramp_N': 180,
        'demag_ramp_delay': 0,
        'demag_start_fraction': 0.02,
    })
    args_init_dyn['cfl_settings']['max_growth'] = 1.1

# Branch-energy protocol: keep F initialization in the flower basin during
# startup, then continue with the same physical acceptance criteria.
flower_startup_stabilization = (sp3_protocol == 'branch_energy' and initMag == 'F')
if flower_startup_stabilization:
    args_init_dyn['dt'] = 5e-5
    args_init_dyn.update({
        'demag_ramp_N': 300,
        'demag_ramp_delay': 0,
        'demag_start_fraction': 0.03,
    })
    args_init_dyn['cfl_settings']['max_growth'] = min(
        float(args_init_dyn['cfl_settings']['max_growth']),
        1.1,
    )


if comm.rank == 0:
    print_flush("\n" + "=" * 65)
    print_flush(f"{' Std Problem 3 — Stereographic Projection ':=^65}")
    print_flush(f" Projection pole    : -Y = (0, -1, 0)")
    print_flush(f" Init magnetization : {initMag}")
    print_flush(f" Mode               : SP3 {sp3_protocol}")
    print_flush(f" V startup stabilize: {'ON' if vortex_startup_stabilization else 'OFF'}")
    print_flush(f" F startup stabilize: {'ON' if flower_startup_stabilization else 'OFF'}")
    print_flush(f" Cube size (L)      : {L}")
    print_flush(f" Mesh size          : {mesh_size} ({N}x{N}x{N})")
    print_flush(f" Damping (alpha)    : {alpha}")
    print_flush(f" Energies           : {energy_terms}")
    print_flush(f" Anisotropy axis    : {anisotropy_axis}")
    print_flush(f" Time order         : {args_init_dyn['time_order']} ({'BE' if args_init_dyn['time_order']==1 else 'BE→BDF2'})")
    print_flush(f" Initial timestep   : {args_init_dyn['dt']}")
    if 'demag_ramp_N' in args_init_dyn:
        print_flush(
            f" Demag ramp         : N={args_init_dyn['demag_ramp_N']}, "
            f"start={args_init_dyn['demag_start_fraction']}, "
            f"delay={args_init_dyn['demag_ramp_delay']}"
        )
    print_flush(f" Init noise std     : {std}")
    print_flush(f" Output dir         : {output_dir_name}/{case_dir_name}")
    print_flush("=" * 65 + "\n")


"""
Initialize the micromagnetic problem
====================================
"""
if first_run:
    # Get directories
    checkpoint_dir, results_dir = path_dirs['checkpoint_dir'], path_dirs['results_dir']
    # Initialize simulation class
    std_pbm3 = test_wrapper.StandardProblem3(comm, L, initMag, std)
    sim = create_stereo_solver(args_init_sim, model=solver_model)
    # Create mesh
    domain = mesh.create_box(
        comm=comm,
        points=[[0.0, 0.0, 0.0], [L, L, L]],
        n=[N, N, N],
        cell_type=mesh.CellType.tetrahedron
    )
    tdim = domain.topology.dim
    fdim = tdim - 1
    # Build connectivity
    domain.topology.create_connectivity(fdim, tdim)
    domain.topology.create_connectivity(tdim, 0)
    domain.topology.create_connectivity(fdim, 0)
    # Tag entities
    boundary_facets = mesh.exterior_facet_indices(domain.topology)
    facet_tags = mesh.meshtags(domain, fdim, mesh.exterior_facet_indices(domain.topology), surface_tag)
    cell_tags = mesh.meshtags(domain, tdim, np.array(range(domain.topology.index_map(tdim).size_local)), volume_tag)
    # Check partition and save
    sim.check_and_save_mesh_partition(comm, domain, cell_tags, facet_tags, results_dir)
else:
    # Get directories
    checkpoint_dir, results_dir, loading_dir = path_dirs['checkpoint_dir'], path_dirs['results_dir'], path_dirs['loading_dir']
    # Initialize simulation class
    sim = create_stereo_solver(args_init_sim, model=solver_model)
    # Load time
    args_init_dyn['step_start'], args_init_dyn['t_start'], args_init_dyn['dt'] = \
        sim.read_checkpoint_time(loading_dir, loading_dir + '/checkpoint_energies.txt')
    # Load mesh and initial condition
    domain, boundary_facets, cell_tags, facet_tags = sim.read_checkpoint_mesh(
        comm,
        loading_dir + '/checkpoint_timestep_{}.bp'.format(args_init_dyn['step_start'])
    )


# Initialize the mesh
sim.set_mesh(domain, cell_tags, facet_tags, surface_tag)

# Initialize the materials
sim.set_materials(materials)

# Set magnetostatics
if "demag" in energy_terms:
    sim.set_magnetostatics(interfaces, surface_tag)
    sim.set_BEMcpp(path_dirs['BEM_dir'])

# Initialize the magnetization
if first_run:
    # Convert Cartesian SP3 initialization directly to stereographic coordinates.
    smoothing_len = 0.5
    smoothing_iters = 2
    sim.set_init_q_from_m_function(std_pbm3.eval_m_init, smoothing_len, smoothing_iters)
    if comm.rank == 0:
        print_flush(f"\n[Initialization] Magnetization set from '{initMag}' init (Cartesian SP3) → stereographic")

    # Report actual initialized Cartesian m statistics for verification.
    sim._interpolate_m_cart_numpy()
    m_arr = sim.m_cart.x.array
    mx_l = m_arr[0::3]
    my_l = m_arr[1::3]
    mz_l = m_arr[2::3]
    mx_min = comm.allreduce(float(np.min(mx_l)) if mx_l.size else 0.0, op=MPI.MIN)
    mx_max = comm.allreduce(float(np.max(mx_l)) if mx_l.size else 0.0, op=MPI.MAX)
    my_min = comm.allreduce(float(np.min(my_l)) if my_l.size else 0.0, op=MPI.MIN)
    my_max = comm.allreduce(float(np.max(my_l)) if my_l.size else 0.0, op=MPI.MAX)
    mz_min = comm.allreduce(float(np.min(mz_l)) if mz_l.size else 0.0, op=MPI.MIN)
    mz_max = comm.allreduce(float(np.max(mz_l)) if mz_l.size else 0.0, op=MPI.MAX)
    if comm.rank == 0:
        print_flush(
            "[Initialization stats] "
            f"mx:[{mx_min:.3e},{mx_max:.3e}] "
            f"my:[{my_min:.3e},{my_max:.3e}] "
            f"mz:[{mz_min:.3e},{mz_max:.3e}]"
        )
else:
    checkpoint_file = loading_dir + '/checkpoint_timestep_{}.bp'.format(args_init_dyn['step_start'])
    # Restore stereographic coordinates
    sim.restore_checkpoint_atlas_state(checkpoint_file)
    if comm.rank == 0:
        print_flush("[Checkpoint] Restored atlas chart state and BDF history")

# Save initial magnetization
sim.write_xdmf_checkpoint('mag', checkpoint_dir + '/checkpoint_mag_start.xdmf')


"""
Solve the problem
=================
"""
if comm.rank == 0:
    print_flush("\n~~~~ Running the dynamic solver (stereographic projection)")

comm.Barrier()
start_time = MPI.Wtime()

# Initialize dynamic solver
sim.init_dynamic(args_init_dyn)

# Open outputs collectively
comm.Barrier()
for target in args_init_dyn['write_target']:
    sim.open_output(target, results_dir + '/' + results_file_name + '_evol_' + target)
comm.Barrier()

run_failed = False
run_error = None

try:
    # Solve dynamic problem
    sim.solve_problem()

except Exception as e:
    run_failed = True
    run_error = e
    if comm.rank == 0:
        print("\nSimulation terminated prematurely with error:")
        print(f"Exception type: {type(e).__name__}")
        print(f"Exception message: {e}")
        traceback.print_exc(file=sys.stdout)
        sys.stdout.flush()

finally:
    comm.Barrier()
    end_time = MPI.Wtime()
    total_time = end_time - start_time
    if comm.rank == 0:
        print_flush("\nTime for solving dynamic problem: " + str(total_time))

    # Close outputs collectively
    comm.Barrier()
    for target in args_init_dyn['write_target']:
        sim.close_output(target)
    comm.Barrier()

    # Save final magnetization
    sim.write_xdmf_checkpoint('mag', checkpoint_dir + '/checkpoint_mag_end.xdmf')

    # End simulation
    sim.destroy_dynamic()
    if comm.rank == 0:
        print_flush('\n~~~~ Simulation ended')

# Raise after graceful collective cleanup so output files are valid.
failed_global = comm.allreduce(1 if run_failed else 0, op=MPI.MAX)
if failed_global:
    if run_error is not None:
        raise RuntimeError(f"Dynamic simulation failed: {run_error}")
    raise RuntimeError("Dynamic simulation failed on at least one MPI rank.")
