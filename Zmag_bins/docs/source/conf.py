"""Sphinx configuration for Z-mag dynamic stereographic solver API docs."""
from __future__ import annotations

import os
import sys

# Repository root and Zmag_bins package (for autodoc imports).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

project = "Z-mag"
copyright = "2025, Stella Brach, IBM Research Zurich"
author = "Stella Brach"
release = "2025"

html_title = "Documentation"
html_short_title = "Documentation"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinxcontrib.bibtex",
]

_bib_dir = os.path.dirname(os.path.abspath(__file__))
bibtex_bibfiles = [os.path.join(_bib_dir, "refs.bib")]
bibtex_default_style = "plain"
bibtex_reference_style = "author_year"

numfig = True
numfig_format = {
    "figure": "Figure %s",
    "table": "Table %s",
    "code-block": "Listing %s",
    "section": "Section %s",
}

templates_path = ["_templates"]
exclude_patterns = ["_archive", "_includes"]
suppress_warnings = ["bibtex.duplicate_citation", "autosummary.import_cycle"]

# HTML theme (Furo)
html_theme = "furo"
html_static_path = ["_static", "images"]
html_css_files = ["solver_flow.css", "dual_chart_init.css", "bem_stepper.css"]
html_logo = "images/zmag_logo2.png"
html_favicon = "images/zmag_logo2.png"
htmlhelp_basename = "zmagdoc"

# MathJax 3 CDN (avoids Sphinx 9 default MathJax 4 breakage).
mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
mathjax3_config = {
    "tex": {
        "macros": {},
        "inlineMath": [["$", "$"], ["\\(", "\\)"]],
        "displayMath": [["$$", "$$"], ["\\[", "\\]"]],
        "packages": {"[+]": ["ams", "textmacros"]},
    },
    "options": {
        "renderActions": {
            "addMenu": [0, "", ""],
        },
    },
}

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "show-inheritance": True,
    "undoc-members": False,
    "private-members": False,
}
autodoc_mock_imports = [
    "geometry",
    "dolfinx",
    "dolfinx.fem",
    "dolfinx.io",
    "dolfinx.la",
    "dolfinx.mesh",
    "dolfinx.cpp",
    "dolfinx.cpp.mesh",
    "ufl",
    "basix",
    "basix.ufl",
    "mpi4py",
    "petsc4py",
    "PETSc",
    "adios4dolfinx",
    "numpy",
    "bem_assembler",
    "bem_assembler.bem_assembler",
]
autosummary_generate = True

napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True


def _skip_private_members(app, what, name, obj, skip, options):
    if skip:
        return skip
    if what in ("class", "exception", "module") and name.startswith("_"):
        return True
    if what in ("method", "function", "attribute") and name.startswith("_"):
        if name in ("__init__", "__call__"):
            return False
        return True
    return skip


def setup(app):
    app.connect("autodoc-skip-member", _skip_private_members)
