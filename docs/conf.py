# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import datetime
import sys
import os

sys.path.insert(0, os.path.abspath('.'))
sys.path.insert(0, os.path.abspath('..'))

# -- Project information -----------------------------------------------------
about = {}
with open('../crest/__about__.py', "r") as fp:
    exec(fp.read(), about)
project = 'CREST'
copyright = f'{datetime.datetime.now().year}, Craig Pelissier et. al.'
release = about["__version__"]

# -- General configuration ---------------------------------------------------
extensions = [
    'sphinx.ext.autodoc',  # autodocument
    'sphinx.ext.napoleon',  # google and numpy doc string support
    'sphinx.ext.mathjax',  # latex rendering of equations using MathJax
    "sphinx.ext.doctest",
    "sphinx_autodoc_typehints",
    #'nbsphinx',  # for direct embedding of jupyter notebooks into sphinx docs
    #'nbsphinx_link'  # to be able to include notebooks from outside of the docs folder
]
#    "sphinx.ext.viewcode",

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The suffix of source filenames.
source_suffix = [".rst", ".md"]

# The master toctree document.
master_doc = 'index'

# Allows to build the docs with a minimal environment without warnings about missing packages
autodoc_mock_imports = [
    'tensorflow',
    'pytest',
    'matplotlib',
    'tqdm',
    'pandas',
    'numpy',
    'numba',
    'fsspec',
    'xarray',
    'networkx',
    'dask',
    'bottleneck',
    'cloudpickle',
    'pyarrow',
    'sparse',
    'ipywidgets',
    'sklearn',
    'polars',
    'scipy',
    'zarr',
    'psutil',
    'seaborn',
    'tlz',
    'jsonpickle',
]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']
html_logo = '_static/img/temp_logo.png'

# -- Napoleon autodoc options -------------------------------------------------
napoleon_numpy_docstring = True
napoleon_google_docstring = True
napoleon_use_ivar = True
napoleon_include_init_with_doc = True

nbsphinx_allow_errors = True
nbsphinx_kernel_name = 'python3'
nbsphinx_execute = 'never'
