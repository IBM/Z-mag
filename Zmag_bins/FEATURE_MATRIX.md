# Stereographic dynamic solver feature matrix (DOLFINx 0.11)

This document summarizes the **time-dependent** stereographic solver variants shipped in
the public release:

- `Zmag_dynamic_stereo_base.py` — shared config, checkpoints, time loop (`Z_sim_dynamic_stereo_base`)
- `Zmag_dynamic_stereo_demag.py` / `Zmag_dynamic_stereo_nodemag.py` — model-specific FEM and SNES
- `Zmag_dynamic_stereo_solvers.py` — factory

Static equilibrium solvers (`Zmag_static_stereo_*`) are not part of this distribution.

## Model Variants

| Capability | `demag` variant | `nodemag` variant |
|---|---:|---:|
| Class | `Z_sim_dynamic_stereo_demag` | `Z_sim_dynamic_stereo_nodemag` |
| State unknowns | `(q_A, q_B, phi)` | `(q_A, q_B)` |
| Demagnetizing field term | yes | no |
| Magnetostatic helper/BEM operators | yes | no |
| Atlas masks/weights (`chiA/chiB/chiAB`, `wA/wB`) | yes | yes |
| Operator-split fixed-point solve | yes | yes |
| BDF/BE time integration | yes | yes |
| CFL/error/thermo timestep control | yes | yes |

## Energy Terms

| Energy term | `demag` variant | `nodemag` variant |
|---|---:|---:|
| `demag` | required | not allowed |
| `zeeman` | supported | supported |
| `exchange` | supported | supported |
| `uniaxial_anisotropy` | supported | supported |
| `cubic_anisotropy` | supported | supported |

## Output Targets

### Supported in both variants

- `mag`
- `heff`
- `q`
- `qA`
- `qB`
- `chiA`
- `chiB`
- `chiAB`
- `wA`
- `wB`
- `norm_q_dot`
- `M_scaled`

### Demag-only targets

- `demag`
- `phi`
- `phi1_boundary`
- `Hd_scaled`
- `B_scaled`

`nodemag` raises a `ValueError` when demag-only targets are requested.

## Checkpoint and Restart Behavior

| Capability | `demag` variant | `nodemag` variant |
|---|---:|---:|
| XDMF checkpoint for `mag/qA/qB/chi*/w*` | yes | yes |
| XDMF checkpoint for `demag/phi/phi1_boundary` | yes | no |
| Demag restart state (`phi`, demag field) | yes | no |
| qA/qB restart history | yes | yes |

## Dynamic configuration (`args_init_dyn`)

Required keys for `init_dynamic`: `t_start`, `dt`, `step_start`, `step_end`,
`first_run`, `checkpoint_every`, `write_target`, and `cfl_settings` (adaptive
timestep: BDF error estimate + CFL cap + thermo caps). See `docs/source/config.rst`.

Checkpoint npz files use `_qA`, `_qB`, and optional BDF history names only
(see `Zmag_dynamic_stereo_base.save_checkpoint_atlas_state`).

## Factory API

Use:

- `create_stereo_solver(args_init_sim, model="demag")`
- `create_stereo_solver(args_init_sim, model="nodemag")`

Import concrete classes from `Zmag_dynamic_stereo_solvers` (not from individual
implementation modules).

where `model` must be one of:

- `"demag"`
- `"nodemag"`

## Driver Guidance

For `model="nodemag"`:

- do not request demag-only outputs/checkpoint targets,
- do not interpret magnetostatic quantities (`phi`, demag field),
- use trajectory/atlas diagnostics based on `qA/qB` and `mag`.

