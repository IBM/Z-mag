"""
Checkpoint / Restart Chain Benchmark  (Stereographic formulation)
=================================================================

Multi-phase test for verifying checkpoint correctness with the two-chart
atlas stereographic solver (``create_stereo_solver``, model ``demag``):

  Phase 1    –  Continuous run steps 0..N-1  (reference)
  Restart 1  –  From Phase 1's checkpoint at step K,   run K+1..2K
  Restart 2  –  From Restart 1's checkpoint at step 2K, run 2K+1..3K
  Restart 3  –  From Restart 2's checkpoint at step 3K, run 3K+1..4K
  Restart 4  –  From Restart 3's checkpoint at step 4K, run 4K+1..N-1

This forms a checkpoint CHAIN — each restart loads from the PREVIOUS
restart's output, exercising the save/load round-trip repeatedly.

PASS criteria (HARD gate — checkpoint correctness):
  * Cell / facet tag counts match across all restarts.
  * Field norms at each restart point match to machine precision
    (rel diff < 1e-10).

Trajectory comparison (informational):
  Drift is expected because a4dx.read_mesh redistributes the mesh,
  giving a different BEM matrix and floating-point reduction order.
  Demag phi is re-equilibrated with the live BEM after restore so the
  thermo gate is not charged for that operator change on the first
  post-restart step.

Physics:
  Elongated box (4:1 aspect ratio) magnetized along the long axis.
  This keeps demag moderate (shape anisotropy favours the initial state),
  combined with uniaxial anisotropy along the same axis and moderate Q1,
  giving a well-conditioned problem with smooth relaxation dynamics.

Usage:
    mpirun -n 2 python test_dynamic_checkpoints.py
    python test_dynamic_checkpoints.py
"""

import petsc4py
import os, sys
import numpy as np
from dolfinx import fem, mesh
from mpi4py import MPI

from Zmag_bins.tools import dir_wrapper
from Zmag_bins.Zmag_dynamic_stereo_solvers import create_stereo_solver
from Zmag_bins.Zmag_dynamic_stereo_base import checkpoint_atlas_field_norms

petsc4py.init(sys.argv)
comm = MPI.COMM_WORLD


# =====================================================================
#  Helpers
# =====================================================================
def create_two_material_box(communicator, Lx, Ly, Lz, Nx, Ny, Nz,
                            vol_left, vol_right, surf_tag, iface_tag):
    """Create a box [0,Lx]x[0,Ly]x[0,Lz] split into two volumes at x = Lx/2."""
    domain = mesh.create_box(
        comm=communicator,
        points=[[0.0, 0.0, 0.0], [Lx, Ly, Lz]],
        n=[Nx, Ny, Nz],
        cell_type=mesh.CellType.tetrahedron,
        ghost_mode=mesh.GhostMode.shared_facet
    )
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    domain.topology.create_connectivity(tdim, 0)
    domain.topology.create_connectivity(fdim, 0)

    midplane = Lx / 2.0
    x_coords = domain.geometry.x
    cell_map = domain.topology.index_map(tdim)
    num_cells = cell_map.size_local
    cell_values = np.full(num_cells, vol_right, dtype=np.int32)
    cell_to_vertex = domain.topology.connectivity(tdim, 0)
    for c in range(num_cells):
        verts = cell_to_vertex.links(c)
        if np.mean(x_coords[verts, 0]) < midplane:
            cell_values[c] = vol_left
    cell_tags = mesh.meshtags(domain, tdim,
                              np.arange(num_cells, dtype=np.int32), cell_values)

    ext_facets = mesh.exterior_facet_indices(domain.topology)
    facet_to_vertex = domain.topology.connectivity(fdim, 0)
    num_facets_local = domain.topology.index_map(fdim).size_local
    tol = max(Lx / Nx, Ly / Ny, Lz / Nz) * 1.5
    iface_facets = []
    for f in range(num_facets_local):
        if f in ext_facets:
            continue
        cx = np.mean(x_coords[facet_to_vertex.links(f), 0])
        if abs(cx - midplane) < tol:
            iface_facets.append(f)
    iface_facets = np.array(iface_facets, dtype=np.int32)

    all_idx = np.concatenate([ext_facets.astype(np.int32), iface_facets])
    all_val = np.concatenate([
        np.full(len(ext_facets), surf_tag, dtype=np.int32),
        np.full(len(iface_facets), iface_tag, dtype=np.int32)])
    order = np.argsort(all_idx)
    facet_tags = mesh.meshtags(domain, fdim, all_idx[order], all_val[order])

    return domain, ext_facets, cell_tags, facet_tags


def owned_l2(func):
    """Global L2 norm, owned DOFs only (no ghost double-counting)."""
    imap = func.function_space.dofmap.index_map
    bs = func.function_space.dofmap.index_map_bs
    n = imap.size_local * bs
    return np.sqrt(comm.allreduce(np.sum(func.x.array[:n]**2), op=MPI.SUM))


def global_tag_count(tags, value, domain, edim):
    """Count entities with a given tag value, owned only."""
    n_owned = domain.topology.index_map(edim).size_local
    return comm.allreduce(int(np.sum((tags.indices < n_owned) & (tags.values == value))),
                          op=MPI.SUM)


def field_norms(sim):
    """Return dict of owned-L2 norms for the full atlas BDF state."""
    return {
        'qA': owned_l2(sim.qA),
        'qA_prev': owned_l2(sim.qA_prev),
        'qA_prev_prev': owned_l2(sim.qA_prev_prev),
        'qB': owned_l2(sim.qB),
        'qB_prev': owned_l2(sim.qB_prev),
        'qB_prev_prev': owned_l2(sim.qB_prev_prev),
    }


def make_dirs(out_dir):
    """Create a minimal directory tree for one phase."""
    if comm.rank == 0:
        os.makedirs(out_dir, exist_ok=True)
    comm.Barrier()
    tree = dir_wrapper.directory_tree(comm, True, True, out_dir, energy_terms)
    paths = tree.get_directory_tree_structure()
    tree.create_directory_tree()
    return paths


def make_args_sim(paths):
    return {'alpha': alpha, 'energy_terms': energy_terms,
            'NeumannBC': True, 'printout': (comm.rank == 0), 'path_dirs': paths,
            'projection_pole': projection_pole}


# =====================================================================
#  Parameters
# =====================================================================
dim = 3

# Elongated box (4:1 aspect ratio) — demag factor N_x ≈ 0.04 along
# the long axis keeps shape anisotropy moderate when m is along x.
L_x = 16.0;  L_y = 4.0;  L_z = 4.0
mesh_size = 1.0
N_x = int(round(L_x / mesh_size))   # 16
N_y = int(round(L_y / mesh_size))   # 4
N_z = int(round(L_z / mesh_size))   # 4

volume_tag_left  = 1     # material A  (x < L_x/2)
volume_tag_right = 2     # material B  (x > L_x/2)
surface_tag      = 10    # exterior boundary
interface_tag    = 20    # internal interface

alpha = 0.5              # high damping → smooth dynamics
energy_terms = ['exchange', 'uniaxial_anisotropy', 'demag']

# Materials — moderate Q1 so anisotropy competes with demag.
# Easy axis along x (long axis), aligned with shape anisotropy.
Q1_val = 0.5
materials = [
    {'name': 'material_A', 'dim': dim, 'tag': volume_tag_left,
     'Q1': Q1_val, 'Ms': 1.0, 'Aex': 1.0, 'K': 1.0,
     'anisotropy_axis': (1.0, 0.0, 0.0), 'K2K1_ratio': 0},
    {'name': 'material_B', 'dim': dim, 'tag': volume_tag_right,
     'Q1': Q1_val, 'Ms': 0.8, 'Aex': 0.9, 'K': 0.5,
     'anisotropy_axis': (1.0, 0.0, 0.0), 'K2K1_ratio': 0}
]
interfaces = [{'name': 'interface_AB', 'tag': interface_tag}]

# Stereographic projection — pole opposite to the initial m direction.
# m ≈ (1,0,0) maps to q ≈ (0,0) in chart A: maximally well-conditioned.
projection_pole = (-1, 0, 0)

# Timestepping — 50 steps, checkpoint every 10 → restarts at 10, 20, 30, 40
N_steps      = 50
checkpoint_K = 10
n_restarts   = 4

# Initialisation (stereographic noise)
# theta0 = pi/2, theta1 = 0 → m = (sin(pi/2)*cos(0), sin(pi/2)*sin(0), cos(pi/2)) = (1,0,0)
std_u = 0.05
std_v = 0.05
smoothing_len   = 2.0
smoothing_iters = 3

base_args_dyn = {
    't_start': 0.0,
    'dt': 1e-3,
    'step_start': 0,
    'step_end': N_steps,
    'first_run': True,
    'checkpoint_every': checkpoint_K,
    'write_target': ['mag'],
    'save_first_conv_metrics': False,
    'save_last_conv_metrics': False,

    # BDF time integration (1 = Backward Euler, 2 = BDF2)
    'time_order': 2,
    'max_retry': 5,

    'cfl_settings': {
        'CFL': 0.5,
        'dt_min': 1e-7,
        'dt_max': 2e-2,
        'smooth_factor': 0.7,
        'max_growth': 1.5,
        'min_shrink': 0.3,
    },
    'convergence_settings': {
        'physical_tol_mdot_rms': 1e-6,
        'physical_tol_edot': 1e-8,
        'steady_steps': 100,
        'stagnation_tol': 1e-8,
        'stagnation_steps': 100,
    },
}

# Directories
output_root = os.path.join(os.getcwd(), 'output_test_checkpoints_stereo')
compare_dir = os.path.join(output_root, 'comparison')
if comm.rank == 0:
    os.makedirs(output_root, exist_ok=True)
    os.makedirs(compare_dir, exist_ok=True)
comm.Barrier()


# =====================================================================
#  PHASE 1 — Continuous reference run (steps 0 .. N_steps-1)
# =====================================================================
if comm.rank == 0:
    print("\n" + "=" * 60)
    print(f"  PHASE 1: Continuous run (steps 0..{N_steps - 1}), "
          f"checkpoint every {checkpoint_K}")
    print("=" * 60, flush=True)

paths_ph1 = make_dirs(os.path.join(output_root, 'phase1_continuous'))
sim1 = create_stereo_solver(make_args_sim(paths_ph1), model="demag")

domain1, bf1, ct1, ft1 = create_two_material_box(
    comm, L_x, L_y, L_z, N_x, N_y, N_z,
    volume_tag_left, volume_tag_right,
    surface_tag, interface_tag)

sim1.set_mesh(domain1, ct1, ft1, surface_tag)
sim1.set_materials(materials)
sim1.set_magnetostatics(interfaces, surface_tag)
sim1.set_BEMcpp(paths_ph1['BEM_dir'])

# Initial magnetization along +x (long axis) with moderate noise
sim1.set_init_q_from_angles(np.pi / 2, 0.0,
                            std_u, std_v,
                            smoothing_len, smoothing_iters)

args_ph1 = base_args_dyn.copy()
args_ph1['first_run'] = True
sim1.init_dynamic(args_ph1)

for tgt in args_ph1['write_target']:
    sim1.open_output(tgt, paths_ph1['results_dir'] + '/results_evol_' + tgt)
    sim1.write_output(tgt)

sim1.solve_problem()

sim1.write_energies(paths_ph1['results_dir'] + '/results_energies.txt')
for tgt in args_ph1['write_target']:
    sim1.close_output(tgt)

ph1_energies    = np.array(sim1.energies).copy()
ph1_final_norms = field_norms(sim1)

# Tag counts (reference)
tdim_val = domain1.topology.dim
fdim_val = tdim_val - 1
ph1_tags = {
    'ct_A':   global_tag_count(ct1, volume_tag_left,  domain1, tdim_val),
    'ct_B':   global_tag_count(ct1, volume_tag_right, domain1, tdim_val),
    'ft_ext': global_tag_count(ft1, surface_tag,      domain1, fdim_val),
    'ft_if':  global_tag_count(ft1, interface_tag,     domain1, fdim_val),
}

# Ground-truth norms at the FIRST restart point (step K)
# Read the checkpoint back on Phase 1's own mesh (same partition → lossless).
first_ckpt = checkpoint_K
bp_gt = paths_ph1['checkpoint_dir'] + f'/checkpoint_timestep_{first_ckpt}.bp'
next_gt_norms = checkpoint_atlas_field_norms(sim1, bp_gt)

sim1.destroy_dynamic()


# =====================================================================
#  RESTART CHAIN  (4 restarts, each loading from previous phase)
# =====================================================================
prev_ckpt_dir = paths_ph1['checkpoint_dir']
restart_results = []          # collected per-restart diagnostics

for r in range(1, n_restarts + 1):
    ckpt_step = r * checkpoint_K
    # Each restart runs exactly K steps, EXCEPT the last one which
    # finishes Phase 1's step_end so all phases share the same final step.
    step_end_r = min((r + 1) * checkpoint_K + 1, N_steps)

    if comm.rank == 0:
        print("\n" + "=" * 60)
        print(f"  RESTART {r}/{n_restarts}: from step {ckpt_step}, "
              f"target step_end = {step_end_r - 1}")
        print("=" * 60, flush=True)

    out_dir_r = os.path.join(output_root, f'restart_{r}_from_step_{ckpt_step}')
    paths_r = make_dirs(out_dir_r)
    paths_r['loading_dir'] = prev_ckpt_dir
    sim_r = create_stereo_solver(make_args_sim(paths_r), model="demag")

    # --- Read checkpoint header ---
    args_r = base_args_dyn.copy()
    args_r['first_run'] = False
    args_r['step_end'] = step_end_r

    if r == 1:
        # Phase 1's checkpoint dir has MULTIPLE checkpoints (steps 0, K, 2K, ...),
        # so read_checkpoint_time would pick the LAST one (step 4K), not step K.
        # Extract step/t/dt manually from Phase 1's in-memory energies.
        args_r['step_start'] = ckpt_step
        args_r['t_start']    = float(ph1_energies[ckpt_step, 0])
        args_r['dt']         = float(ph1_energies[ckpt_step + 1, 1])
    else:
        # Previous restart produced exactly ONE checkpoint → read_checkpoint_time
        # returns the correct (and only) step with the CFL-patched dt.
        args_r['step_start'], args_r['t_start'], args_r['dt'] = \
            sim_r.read_checkpoint_time(prev_ckpt_dir,
                                       prev_ckpt_dir + '/checkpoint_energies.txt')

    # --- Read mesh + tags ---
    bp_path = prev_ckpt_dir + f'/checkpoint_timestep_{args_r["step_start"]}.bp'
    domain_r, bf_r, ct_r, ft_r = sim_r.read_checkpoint_mesh(comm, bp_path)

    sim_r.set_mesh(domain_r, ct_r, ft_r, surface_tag)
    sim_r.set_materials(materials)
    sim_r.set_magnetostatics(interfaces, surface_tag)
    sim_r.set_BEMcpp(paths_r['BEM_dir'])

    # --- Restore BDF2 state ---
    sim_r.restore_checkpoint_atlas_state(bp_path)

    # --- Checkpoint restoration diagnostic ---
    restored = field_norms(sim_r)
    gt_norms = next_gt_norms   # ground truth from the PREVIOUS phase's self-read
    restore_checks = []
    for fld in ['qA', 'qA_prev', 'qA_prev_prev', 'qB', 'qB_prev', 'qB_prev_prev']:
        rd = abs(restored[fld] - gt_norms[fld]) / gt_norms[fld] if gt_norms[fld] > 0 else 0.0
        ok = rd < 1e-10
        restore_checks.append((fld, gt_norms[fld], restored[fld], rd, ok))

    if comm.rank == 0:
        all_ok = all(c[4] for c in restore_checks)
        tag = "OK" if all_ok else "FAIL"
        print(f"  Restoration check [{tag}]:  ", end="")
        for fld, _, _, rd, ok in restore_checks:
            print(f"{fld} rd={rd:.1e}[{'OK' if ok else 'FAIL'}]  ", end="")
        print(flush=True)

    # Tag counts on reloaded mesh
    r_tags = {
        'ct_A':   global_tag_count(ct_r, volume_tag_left,  domain_r, tdim_val),
        'ct_B':   global_tag_count(ct_r, volume_tag_right, domain_r, tdim_val),
        'ft_ext': global_tag_count(ft_r, surface_tag,      domain_r, fdim_val),
        'ft_if':  global_tag_count(ft_r, interface_tag,     domain_r, fdim_val),
    }

    # --- Run dynamics ---
    sim_r.init_dynamic(args_r)
    for tgt in args_r['write_target']:
        sim_r.open_output(tgt, paths_r['results_dir'] + '/results_evol_' + tgt)
    sim_r.solve_problem()
    sim_r.write_energies(paths_r['results_dir'] + '/results_energies.txt')
    for tgt in args_r['write_target']:
        sim_r.close_output(tgt)

    if not hasattr(sim_r, 'energies'):
        raise RuntimeError(
            f"Restart {r}: solve_problem produced no steps "
            f"(step_start={args_r['step_start']}, step_end={args_r['step_end']})")
    r_energies   = np.array(sim_r.energies).copy()
    r_final_norms = field_norms(sim_r)

    # --- Ground truth for the NEXT restart ---
    if r < n_restarts:
        next_step = (r + 1) * checkpoint_K
        bp_next = paths_r['checkpoint_dir'] + f'/checkpoint_timestep_{next_step}.bp'
        next_gt_norms = checkpoint_atlas_field_norms(sim_r, bp_next)

    sim_r.destroy_dynamic()
    prev_ckpt_dir = paths_r['checkpoint_dir']

    restart_results.append({
        'r': r,
        'ckpt_step': ckpt_step,
        'step_end': step_end_r,
        'tags': r_tags,
        'restore_checks': restore_checks,
        'energies': r_energies,
        'final_norms': r_final_norms,
    })


# =====================================================================
#  COMPARISON
# =====================================================================
if comm.rank == 0:
    overall_pass = True
    os.makedirs(compare_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("  CHECKPOINT CHAIN BENCHMARK RESULTS  (Stereographic)")
    print("=" * 70)

    # ── A. Hard gate: checkpoint correctness ──
    print("\n  ── A. Checkpoint correctness (HARD gate) ──\n")

    # Tag counts
    print(f"  {'Test':<32s}", end="")
    print(f" {'Phase1':>7s}", end="")
    for res in restart_results:
        print(f" {'R' + str(res['r']):>7s}", end="")
    print(f" {'Result':>8s}")
    print("  " + "-" * (32 + 8 + 8 * len(restart_results) + 8))

    for key, label in [('ct_A', 'Cell: material_A'),
                        ('ct_B', 'Cell: material_B'),
                        ('ft_ext', 'Facet: exterior'),
                        ('ft_if', 'Facet: interface')]:
        ref = ph1_tags[key]
        vals = [res['tags'][key] for res in restart_results]
        ok = all(v == ref for v in vals)
        if not ok:
            overall_pass = False
        print(f"  {label:<32s}", end="")
        print(f" {ref:>7d}", end="")
        for v in vals:
            print(f" {v:>7d}", end="")
        print(f" {'PASS' if ok else 'FAIL':>8s}")

    # Restored field norms at each restart point
    print()
    print(f"  {'Restart':<10s} {'Field':<15s} {'Ground truth':>14s}"
          f" {'Restored':>14s} {'Rel diff':>12s} {'Result':>8s}")
    print("  " + "-" * 75)
    for res in restart_results:
        for fld, gt, rst, rd, ok in res['restore_checks']:
            if not ok:
                overall_pass = False
            print(f"  R{res['r']:<9d} {fld:<15s} {gt:>14.6e}"
                  f" {rst:>14.6e} {rd:>12.2e} {'PASS' if ok else 'FAIL':>8s}")

    # ── B. Trajectory comparison (informational) ──
    print("\n  ── B. Trajectory comparison (informational) ──")
    print("  (drift expected — mesh partition differs after each restart)\n")

    # Save energies for post-processing.
    np.savetxt(os.path.join(compare_dir, 'energies_phase1.txt'), ph1_energies,
               header='Phase 1 (continuous reference)')
    prev_restart_energies = None
    for res in restart_results:
        if res['r'] == 1:
            ckpt_row = ph1_energies[res['ckpt_step'], :]
        else:
            ckpt_row = prev_restart_energies[-1, :]
        augmented = np.vstack([ckpt_row.reshape(1, -1), res['energies']])
        np.savetxt(os.path.join(compare_dir, f'energies_restart{res["r"]}.txt'),
                   augmented,
                   header=f'Restart {res["r"]} from step {res["ckpt_step"]}')
        prev_restart_energies = res['energies']

    # Energy comparison — time-matched against Phase 1
    t_ph1 = ph1_energies[:, 0]
    print(f"  {'Restart':<10s} {'last t':>10s} {'E_restart':>14s}"
          f" {'E_ph1(t)':>14s} {'|diff|':>12s} {'rel':>10s}")
    print("  " + "-" * 72)
    for res in restart_results:
        t_r = res['energies'][-1, 0]
        e_r = res['energies'][-1, -1]
        idx = np.argmin(np.abs(t_ph1 - t_r))
        e_ph1 = ph1_energies[idx, -1]
        diff = abs(e_r - e_ph1)
        rel = diff / abs(e_ph1) if abs(e_ph1) > 0 else 0.0
        print(f"  R{res['r']:<9d} {t_r:>10.4e} {e_r:>14.8e}"
              f" {e_ph1:>14.8e} {diff:>12.4e} {rel:>10.4e}")

    # ── Overall verdict ──
    print("\n" + "=" * 70)
    if overall_pass:
        print("  *** PASS ***  Checkpoint chain preserves tags, fields, BDF2")
        print("                state across 4 restart cycles with MPI redistribution.")
    else:
        print("  *** FAIL ***  Checkpoint corruption — see FAIL entries above.")
    print("  Data saved to:", compare_dir)
    print("=" * 70 + "\n", flush=True)
