# Contributing to Z-mag

This document covers how to report problems, ask for help, and propose changes.
The public tracker is [github.com/IBM/Z-mag](https://github.com/IBM/Z-mag).

## Report an issue

Open a [GitHub issue](https://github.com/IBM/Z-mag/issues).

Please include:

- operating system and how you installed the stack (`conda env create -f environment.yml` or otherwise)
- DOLFINx / FEniCSx version (`python -c "import dolfinx; print(dolfinx.__version__)"`)
- number of MPI ranks, if relevant
- a short description of what you ran and what went wrong (logs, traceback)

Search existing issues first.

## Seek support

Use the same [issue tracker](https://github.com/IBM/Z-mag/issues) for installation, configuration, and expected solver behaviour. There is no separate mailing list.

Installation and API notes are in the [Sphinx documentation](https://ibm.github.io/Z-mag/).

## Contribute code or documentation

1. For anything beyond a typo, open an issue first so the change can be discussed.
2. Fork the repository and branch from `main`.
3. Keep changes focused. Match the surrounding code style. If you change user-facing behaviour, add a note under the next version heading in `CHANGELOG.md`.
4. Run the regression drivers that touch your change. From the repository root, with the `zmag` environment active:

   ```bash
   mpirun -n 4 python test_dynamic_checkpoints.py
   mpirun -n 4 python test_dynamic_precession_damping.py
   mpirun -n 4 python test_dynamic_std_prb3.py
   ```

   See `README_tests.txt` for what each driver covers.
5. Open a pull request against `main` and describe the problem and the fix.

### Developer Certificate of Origin

This project uses the [Developer Certificate of Origin](https://developercertificate.org/). Sign off every commit:

```bash
git commit -s
```

That appends `Signed-off-by: Your Name <you@example.com>` using your git `user.name` and `user.email`.

## License

Contributions are accepted under the [MIT License](LICENSE) of this repository.
