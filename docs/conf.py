# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime
import sys
import os

sys.path.insert(0, os.path.abspath("."))
sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------
about = {}
with open("../crest/__about__.py", "r") as fp:
    exec(fp.read(), about)

project = "CREST"
copyright = f"{datetime.datetime.now().year}, Craig Pelissier et. al."
release = about["__version__"]

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autosectionlabel",  # link from text to a heading using :ref:
      "sphinx.ext.autodoc",  # autodocument
      "sphinx.ext.napoleon",  # google and numpy doc string support
      "sphinx.ext.mathjax",  # latex rendering of equations using MathJax
      "myst_parser",
      "sphinxcontrib.mermaid",
      #"sphinx_autodoc_typehints",  # Disabled due to conflicts with mocked modules
      "nbsphinx",  # for direct embedding of jupyter notebooks into sphinx docs
      "nbsphinx_link",  # to be able to include notebooks from outside of the docs folder
  ]
    
autodoc_typehints = "description"
autodoc_typehints_format = "short"
typehints_use_signature = False
typehints_use_signature_return = False
always_use_bars_union = True
#    "sphinx.ext.viewcode",

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints", "examples/*.nblink"]

# The suffix of source filenames.
source_suffix = [".rst", ".md"]

# The master toctree document.
master_doc = "index"

suppress_warnings = ["autosectionlabel.*", "app.add_directive"]

# Allows to build the docs with a minimal environment without warnings about missing packages
autodoc_mock_imports = [
    "tensorflow",
    "pytest",
    "matplotlib",
    "tqdm",
    "pandas",
    "numpy",
    "numba",
    "fsspec",
    "xarray",
    "networkx",
    "dask",
    "bottleneck",
    "cloudpickle",
    "pyarrow",
    "sparse",
    "ipywidgets",
    "sklearn",
    "polars",
    "scipy",
    "zarr",
    "tiledb",
    "psutil",
    "seaborn",
    "tlz",
    "jsonpickle",
    "pyvis",
    "keras",
    "s3fs",
    "s3path",
    "rasterio",
    "geopandas",
    "cv2",
    "shapely",
    "dill",
    "prettytable",
]
    
  # -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/img/temp_logo.png"

# -- Napoleon autodoc options -------------------------------------------------
napoleon_numpy_docstring = True
napoleon_google_docstring = True
napoleon_use_ivar = True
napoleon_include_init_with_doc = True

nbsphinx_allow_errors = True
nbsphinx_kernel_name = "python3"
nbsphinx_execute = "never"
nbsphinx_timeout = -1  # No timeout
nbsphinx_require_js_path = ""  # Don't require JS
