"""
Z-mag stereographic dynamic micromagnetic solvers.

Public entry point::

    from Zmag_bins import create_stereo_solver
"""

from Zmag_bins.Zmag_dynamic_stereo_solvers import (
    StereoDemagSolver,
    StereoNoDemagSolver,
    StereoSolverBase,
    Z_sim_dynamic_stereo_demag,
    Z_sim_dynamic_stereo_nodemag,
    create_stereo_solver,
)

__all__ = [
    "create_stereo_solver",
    "StereoSolverBase",
    "StereoDemagSolver",
    "StereoNoDemagSolver",
    "Z_sim_dynamic_stereo_demag",
    "Z_sim_dynamic_stereo_nodemag",
]

__version__ = "1.0.0"
