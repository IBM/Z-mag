import csv
import os
import sys
import traceback
from functools import partial

import numpy as np
import petsc4py
from dolfinx import mesh
from mpi4py import MPI

from Zmag_bins.tools import dir_wrapper
from Zmag_bins.Zmag_dynamic_stereo_solvers import create_stereo_solver


petsc4py.init(sys.argv)
comm = MPI.COMM_WORLD
print_flush = partial(print, flush=True)


def unit(v):
    arr = np.asarray(v, dtype=np.float64).reshape(-1)
    nrm = float(np.linalg.norm(arr))
    if nrm <= 1e-15:
        raise ValueError("Vector norm must be > 0.")
    return tuple(float(x) for x in (arr / nrm))


def uniform_m_init(vec):
    m0 = unit(vec)

    def _eval(x):
        n = x.shape[1]
        return (
            np.full(n, m0[0], dtype=np.float64),
            np.full(n, m0[1], dtype=np.float64),
            np.full(n, m0[2], dtype=np.float64),
        )

    return _eval


TESTS = {
    "A1": {
        "alpha": 0.05,
        "L": 1.0,
        "N": 6,
        "Ms": 1.0,
        "Aex": 0.0,
        "K": 0.0,
        "Q1": 1.0,
        "K2K1_ratio": 0.0,
        "anisotropy_axis": (0.0, 0.0, 1.0),
        "init_m": unit((0.92, -0.34, 0.18)),
        "zeeman": unit((0.0, 0.0, 1.0)),
        "energy_terms": ["zeeman"],
        "dt": 1.5e-4,
        "step_end": 60000,
    },
    "A2": {
        "alpha": 0.1,
        "L": 1.0,
        "N": 6,
        "Ms": 1.0,
        "Aex": 0.0,
        "K": 0.0,
        "Q1": 1.0,
        "K2K1_ratio": 0.0,
        "anisotropy_axis": (0.0, 0.0, 1.0),
        "init_m": unit((0.92, -0.34, 0.18)),
        "zeeman": unit((0.0, 0.0, 1.0)),
        "energy_terms": ["zeeman"],
        "dt": 1.5e-4,
        "step_end": 20000,
    },
    "A3": {
        "alpha": 0.5,
        "L": 1.0,
        "N": 6,
        "Ms": 1.0,
        "Aex": 0.0,
        "K": 0.0,
        "Q1": 1.0,
        "K2K1_ratio": 0.0,
        "anisotropy_axis": (0.0, 0.0, 1.0),
        "init_m": unit((0.92, -0.34, 0.18)),
        "zeeman": unit((0.0, 0.0, 1.0)),
        "energy_terms": ["zeeman"],
        "dt": 1.5e-4,
        "step_end": 20000,
    },
    "A4": {
        "alpha": 1.0,
        "L": 1.0,
        "N": 6,
        "Ms": 1.0,
        "Aex": 0.0,
        "K": 0.0,
        "Q1": 1.0,
        "K2K1_ratio": 0.0,
        "anisotropy_axis": (0.0, 0.0, 1.0),
        "init_m": unit((0.92, -0.34, 0.18)),
        "zeeman": unit((0.0, 0.0, 1.0)),
        "energy_terms": ["zeeman"],
        "dt": 1.5e-4,
        "step_end": 20000,
    },
}


selected_test = "A2"
solver_model = "nodemag"  # "demag" or "nodemag"
#output_root = "output_precession_damping_stereo_qAqB_checkunified"
output_root = "/proj/tape-research/stb/output_precession_damping_stereo_qAqB_checkunified_review"

use_field_ramps = False
write_target = [
    "mag",
    "phi",
    "demag",
    "heff",
    "qA",
    "qB",
    "chiA",
    "chiB",
    "chiAB",
    "wA",
    "wB",
    "norm_q_dot",
]
checkpoint_every = 5000


def _run_single_case(case_label: str, cfg: dict):
    volume_tag = 1
    surface_tag = 10
    dim = 3
    first_run = True
    compute_BEM = True

    energy_terms = list(cfg["energy_terms"])

    case_dir_name = f"alpha_{cfg['alpha']:.4f}_N_{cfg['N']}"
    output_dir = os.path.join(os.getcwd(), output_root, case_dir_name)
    results_file_name = "results"

    dir_tree = dir_wrapper.directory_tree(
        comm, compute_BEM, first_run, output_dir, energy_terms
    )
    path_dirs = dir_tree.get_directory_tree_structure()
    dir_tree.create_directory_tree()
    checkpoint_dir = path_dirs["checkpoint_dir"]
    results_dir = path_dirs["results_dir"]

    args_init_sim = {        "alpha": float(cfg["alpha"]),
        "energy_terms": energy_terms,
        "NeumannBC": True,
        "printout": True,
        "path_dirs": path_dirs,
    }

    materials = [
        {
            "name": "material0",
            "dim": dim,
            "tag": volume_tag,
            "Q1": float(cfg["Q1"]),
            "Ms": float(cfg["Ms"]),
            "Aex": float(cfg["Aex"]),
            "K": float(cfg["K"]),
            "anisotropy_axis": tuple(cfg["anisotropy_axis"]),
            "K2K1_ratio": float(cfg["K2K1_ratio"]),
        }
    ]
    interfaces = []

    active_write_target = list(write_target)
    checkpoint_targets = [
        "mag",
        "phi",
        "demag",
        "heff",
        "chart_ind",
        "q_norm",
        "qA",
        "qB",
        "chiA",
        "chiB",
        "chiAB",
        "wA",
        "wB",
    ]
    if solver_model.strip().lower() == "nodemag":
        active_write_target = [t for t in active_write_target if t not in {"demag", "phi", "phi1_boundary"}]
        checkpoint_targets = [t for t in checkpoint_targets if t not in {"demag", "phi", "phi1_boundary"}]

    args_init_dyn = {
        "t_start": 0.0,
        "dt": float(cfg["dt"]),
        "step_start": 0,
        "step_end": int(cfg["step_end"]),
        "first_run": first_run,
        "checkpoint_every": max(1, int(checkpoint_every)),
        "write_target": active_write_target,
        "xdmf_checkpoint_targets": checkpoint_targets,
        "time_order": 2,
        "max_retry": 5,
        "chart_settings": {
            "hysteresis_tau": 0.0,
            "atlas_overlap_tau": 0.05,
        },
        "atlas_weight_relaxation": 0.35,
        "cfl_settings": {
            "CFL": 0.5,
            "dt_min": 1e-7,
            "dt_max": 5e-3,
            "smooth_factor": 0.7,
            "max_growth": 1.1,
            "min_shrink": 0.5,
            "error_rel_tol": 2e-2,
            "error_abs_tol": 2e-4,
            "error_headroom": 0.90,
            "max_retry_thermo": 20,
        },
        "diagnostic_settings": {
            "enable_thermo_diagnostics": True,
            "thermo_diag_log_accepted": False,
        },
        "jit_timeout_seconds": 1800.0,
        "jit_cache_dir": os.path.join(output_dir, "jit_cache"),
        "convergence_settings": {
            "physical_tol_mdot_rms": 1e-4,
            "physical_tol_edot": 5e-5,
            "steady_steps": 5,
            "stagnation_tol": 1e-4,
            "stagnation_steps": 5,
        },
    }

    if use_field_ramps:
        args_init_dyn.update(
            {
                "demag_ramp_N": 180,
                "demag_ramp_delay": 0,
                "demag_start_fraction": 0.02,
                "zeeman_ramp_N": 180,
                "zeeman_ramp_delay": 0,
                "zeeman_start_fraction": 0.05,
            }
        )

    if comm.rank == 0:
        print_flush("\n" + "=" * 72)
        print_flush(f"{' Mixed Stereo qA/qB Precession-Damping ':=^72}")
        print_flush(f" Test label         : {case_label}")
        print_flush(f" Output dir         : {output_dir}")
        print_flush(f" alpha              : {cfg['alpha']}")
        print_flush(f" solver_model       : {solver_model}")
        print_flush(f" projection_pole    : (0, -1, 0)")
        print_flush(f" mesh               : box L={cfg['L']} with N={cfg['N']}")
        print_flush(f" init m             : {tuple(cfg['init_m'])}")
        print_flush(f" zeeman             : {tuple(cfg['zeeman'])}")
        print_flush(f" write_target       : {args_init_dyn['write_target']}")
        print_flush("=" * 72 + "\n")

    sim = create_stereo_solver(args_init_sim, model=solver_model)
    domain = mesh.create_box(
        comm=comm,
        points=[[0.0, 0.0, 0.0], [cfg["L"], cfg["L"], cfg["L"]]],
        n=[cfg["N"], cfg["N"], cfg["N"]],
        cell_type=mesh.CellType.tetrahedron,
    )
    tdim = domain.topology.dim
    fdim = tdim - 1
    domain.topology.create_connectivity(fdim, tdim)
    domain.topology.create_connectivity(tdim, 0)
    domain.topology.create_connectivity(fdim, 0)
    facet_tags = mesh.meshtags(
        domain,
        fdim,
        mesh.exterior_facet_indices(domain.topology),
        surface_tag,
    )
    cell_tags = mesh.meshtags(
        domain,
        tdim,
        np.array(range(domain.topology.index_map(tdim).size_local)),
        volume_tag,
    )

    sim.check_and_save_mesh_partition(comm, domain, cell_tags, facet_tags, results_dir)
    sim.set_mesh(domain, cell_tags, facet_tags, surface_tag)
    sim.set_materials(materials)
    sim.set_external_field(tuple(cfg["zeeman"]))
    sim.set_magnetostatics(interfaces, surface_tag)
    if "demag" in energy_terms and "BEM_dir" in path_dirs:
        sim.set_BEMcpp(path_dirs["BEM_dir"])
    sim.set_init_q_from_m_function(
        uniform_m_init(cfg["init_m"]),
        smoothing_len=0.15,
        smoothing_iters=1,
    )
    sim.write_xdmf_checkpoint("mag", checkpoint_dir + "/checkpoint_mag_start.xdmf")

    if comm.rank == 0:
        with open(os.path.join(results_dir, "run_config.tsv"), "w", newline="") as f:
            wr = csv.writer(f, delimiter="\t")
            wr.writerow(["key", "value"])
            wr.writerow(["test", case_label])
            for k in sorted(cfg.keys()):
                wr.writerow([k, cfg[k]])
            wr.writerow(["solver_model", solver_model])
            wr.writerow(["write_target", ",".join(args_init_dyn["write_target"])])

    if comm.rank == 0:
        print_flush("\n~~~~ Running dynamic solver (mixed stereo qA/qB)")

    comm.Barrier()
    start_time = MPI.Wtime()
    run_failed = False
    run_error = None

    try:
        sim.init_dynamic(args_init_dyn)
        for target in args_init_dyn["write_target"]:
            sim.open_output(target, results_dir + "/" + results_file_name + "_evol_" + target)
        sim.solve_problem()
    except Exception as exc:
        run_failed = True
        run_error = exc
        if comm.rank == 0:
            print_flush("\nSimulation terminated with error:")
            print_flush(f"{type(exc).__name__}: {exc}")
            traceback.print_exc(file=sys.stdout)
            sys.stdout.flush()
    finally:
        comm.Barrier()
        for target in args_init_dyn["write_target"]:
            try:
                sim.close_output(target)
            except Exception:
                pass
        sim.write_xdmf_checkpoint("mag", checkpoint_dir + "/checkpoint_mag_end.xdmf")
        sim.write_energies(results_dir + "/" + results_file_name + "_energies.txt")

        sim._interpolate_m_cart_numpy()
        m_arr = sim.m_cart.x.array
        n_local = m_arr.size // 3
        local_sum = np.array(
            [np.sum(m_arr[0::3]), np.sum(m_arr[1::3]), np.sum(m_arr[2::3]), float(n_local)],
            dtype=np.float64,
        )
        global_sum = np.zeros_like(local_sum)
        comm.Allreduce(local_sum, global_sum, op=MPI.SUM)
        if comm.rank == 0 and global_sum[3] > 0:
            m_mean = global_sum[:3] / global_sum[3]
            with open(os.path.join(results_dir, "final_mean_m.tsv"), "w", newline="") as f:
                wr = csv.writer(f, delimiter="\t")
                wr.writerow(["mx", "my", "mz"])
                wr.writerow([f"{m_mean[0]:.12e}", f"{m_mean[1]:.12e}", f"{m_mean[2]:.12e}"])

        sim.destroy_dynamic()
        elapsed = MPI.Wtime() - start_time
        if comm.rank == 0:
            print_flush(f"\nTime for solving dynamic problem: {elapsed:.4f} s")
            print_flush("\n~~~~ Simulation ended")

    failed_global = comm.allreduce(1 if run_failed else 0, op=MPI.MAX)
    if failed_global:
        if run_error is not None:
            raise RuntimeError(f"Dynamic simulation failed: {run_error}")
        raise RuntimeError("Dynamic simulation failed on at least one MPI rank.")


def main():
    if selected_test not in TESTS:
        valid = ", ".join(sorted(TESTS.keys()))
        raise ValueError(f"Unknown selected_test='{selected_test}'. Valid choices: {valid}")
    _run_single_case(selected_test, TESTS[selected_test])


if __name__ == "__main__":
    main()
