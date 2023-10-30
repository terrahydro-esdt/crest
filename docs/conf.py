# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import sys
import os
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../crest'))

# -- Project information -----------------------------------------------------
project = 'CREST'
copyright = '2023, Craig Pelissier et. al.'
author = 'Craig Pelissier et. al.'
release = '0.1'

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx_autodoc_typehints",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    # for direct embedding of jupyter notebooks into sphinx docs
    'nbsphinx',
    # to be able to include notebooks from outside of the docs folder
    'nbsphinx_link',
    # google and numpy doc string support
    'sphinx.ext.napoleon',  
]

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'private-members': False,
    'undoc-members': False,
    'inherited-members': False,
    'show-inheritance': False,
#    'exclude-members': 'return_parser'
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The suffix of source filenames.
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

pygments_style = "sphinx"

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

