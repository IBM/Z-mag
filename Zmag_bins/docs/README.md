# Z-mag documentation (Sphinx)

User and API documentation for the stereographic dynamic solvers in `Zmag_bins`.

The published HTML is at https://ibm.github.io/Z-mag/. To rebuild locally:

```bash
cd Zmag_bins/docs
pip install -r requirements.txt
make html
```

Built HTML is in `build/html/`. Interactive figures (solver flowcharts, etc.) are
inline SVG / HTML includes under `source/_includes/`.

## View over HTTP

MathJax and SVG figures do not work reliably from `file://` paths. Serve the
build over HTTP:

```bash
cd Zmag_bins/docs
pip install -r requirements.txt
make html
cd build/html && python -m http.server 8765 --bind 127.0.0.1
```

Then open http://127.0.0.1:8765/

### Live rebuild while you edit

```bash
cd Zmag_bins/docs
pip install -r requirements.txt
./serve.sh
```

`serve.sh` runs `sphinx-autobuild`: it rebuilds when RST or solver sources change
and serves at http://127.0.0.1:8765/. If the docs live on a remote machine, forward
the port (`ssh -L 8765:127.0.0.1:8765 USER@HOST`) or use an IDE port forward.

### Environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `DOCS_PORT` | `8765` | HTTP port |
| `DOCS_HOST` | `127.0.0.1` | Bind address |
| `DOCS_ONCE` | `0` | Set to `1` for `make html` only (no server) |

See [HOW_TO.txt](HOW_TO.txt) for the document map.

## Contents

| Section | Files |
|---------|--------|
| Getting started | `overview.rst`, `installation.rst`, `getting_started.rst` |
| Theory | `theory_llg.rst`, `theory_energy.rst`, `theory_magnetostatics.rst`, `theory_dynamics.rst`, `theory_algorithms.rst`, `theory_benchmarks.rst` |
| API | `config.rst`, `api_factory.rst`, `api_common.rst`, `api_demag.rst`, `api_nodemag.rst` |

Source lives under `source/`; built HTML under `build/html/`.
