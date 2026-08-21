Dynamic stereographic tests
===========================

Run from the **repository root** with MPI where noted.

test_dynamic_std_prb3.py              Standard Problem 3 (flower / vortex)
test_dynamic_precession_damping.py    Precession / damping study
test_dynamic_checkpoints.py         Checkpoint / restart (`_qA` / `_qB` npz)

Example
-------
mpirun -n 4 python test_dynamic_checkpoints.py
