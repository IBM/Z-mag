# Z-mag

Stereographic dynamic micromagnetics with **DOLFINx / FEniCSx** (MPI-parallel).

Time-dependent Landau–Lifshitz–Gilbert on a two-chart stereographic atlas `(q_A, q_B)`, with optional demagnetizing field `(q_A, q_B, φ)` via hybrid FEM–BEM.

**Documentation:** https://ibm.github.io/Z-mag/

**Entry point:**

```python
from Zmag_bins import create_stereo_solver

solver = create_stereo_solver(model="demag")   # or model="nodemag"
```

## Requirements

- DOLFINx / FEniCSx **0.11** (see `environment.yml`)
- PETSc / petsc4py, mpi4py, NumPy, SciPy, Gmsh / meshio
- Optional: build the C++ BEM accelerator in `bem_assembler/cpp/`

Install from a clone of this repository: [installation](Zmag_bins/docs/source/installation.rst).

## Layout

| Path | Contents |
|------|----------|
| `Zmag_bins/` | Dynamic stereo solvers, magnetostatics helpers, `tools/`, Sphinx docs |
| `bem_assembler/` | Python BEM API + C++ sources |
| `test_dynamic_*.py` | Example / regression drivers (run from repo root) |

Manual sources: `Zmag_bins/docs/` (`make html`)
Config keys: `Zmag_bins/FEATURE_MATRIX.md`, `Zmag_bins/docs/source/config.rst`
Tests: see [README_tests.txt](README_tests.txt)

## How to cite

If you use Z-mag in published work, please cite:

> Stella Brach, Federico Soldà, Kathrin Aebischer, and Mark A. Lantz.
> Z-mag: A two-chart stereographic formulation for finite-element micromagnetic simulations.
> *Computer Methods in Applied Mechanics and Engineering* **462** (2026) 119290.
> https://doi.org/10.1016/j.cma.2026.119290

```bibtex
@article{brach2026zmag,
  title     = {{Z-mag}: A two-chart stereographic formulation for finite-element micromagnetic simulations},
  author    = {Brach, Stella and Sold{\`a}, Federico and Aebischer, Kathrin and Lantz, Mark A.},
  journal   = {Computer Methods in Applied Mechanics and Engineering},
  volume    = {462},
  pages     = {119290},
  year      = {2026},
  issn      = {0045-7825},
  doi       = {10.1016/j.cma.2026.119290},
  url       = {https://doi.org/10.1016/j.cma.2026.119290}
}
```

GitHub also exposes this metadata via [CITATION.cff](CITATION.cff) (**Cite this repository** in the sidebar). For a specific software snapshot, cite the git tag or commit as well.

## License

[MIT](LICENSE)

## Versioning

Software releases are git tags (`v1.0.0`, …) described in [CHANGELOG.md](CHANGELOG.md). The package import name `Zmag_bins` does not change between versions.
