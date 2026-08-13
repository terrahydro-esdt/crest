.. TODO: Uncomment badges once the repository is public and DOI is registered

.. |docs| image:: https://readthedocs.org/projects/crest/badge/?version=latest
   :target: https://terrahydro-esdt.github.io/crest/
   :alt: Documentation Status

.. |license| image:: https://img.shields.io/badge/License-Apache%202.0-blue.svg
   :target: https://www.apache.org/licenses/LICENSE-2.0
   :alt: License: Apache 2.0

.. |python| image:: https://img.shields.io/badge/python-3.10+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.10+

.. |tf| image:: https://img.shields.io/badge/TensorFlow-2.18+-orange.svg
   :target: https://tensorflow.org/
   :alt: TensorFlow 2.18+

.. |style| image:: https://img.shields.io/badge/code%20style-black-000000.svg
   :target: https://github.com/psf/black
   :alt: Code style: black

|docs| |license| |python| |tf| |style|

CREST: Coupled Reusable Earth System Tensor Framework
=====================================================

*A tensor-based modeling framework for building AI-enabled Earth Information Systems*

Overview
--------

**CREST** is a tooling framework for developing data-driven Earth system models that run on tensor-based machine learning frameworks such as Keras, TensorFlow, and PyTorch. It enables users to efficiently combine, sample, and manipulate large, heterogeneous Earth datasets across dense, sparse, irregular, and polygon-based grids, creating flexible data pipelines for model training. CREST also provides intuitive graph-based APIs for constructing complex, coupled, hierarchical Earth system models, along with integrated tools for model explainability, evaluation, and reliability assessment.
 
Key capabilities
~~~~~~~~~~~~~~~~

- **Streamlined data pipelines**: Build scalable data pipelines that efficiently leverage large Earth datasets without extensive preprocessing or regridding.
- **Composable Earth system models**: Assemble, organize, and train complex coupled Earth system models from reusable subcomponents using intuitive graph-based abstractions.
- **Collaboration and interoperability**: Promote lightweight standardization of modeling interfaces and information exchange, making it easier for researchers and developers across disciplines to collaborate and integrate models.
- **Model evaluation and explainability**: Assess model performance, interpret predictions, and quantify reliability using built-in analysis and diagnostics tools.


Quick Start
-----------

Installation
~~~~~~~~~~~~

.. note::
   CREST will be available on PyPI at public release (planned August 2026).
   Until then, use the developer install below.

**Future public install:**

.. code-block:: bash

   pip install crest

**Developer install (current):**

Choose the environment file that matches your platform:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Platform
     - Environment file
   * - macOS Apple Silicon (M1/M2/M3)
     - ``cicd/environment_macos_arm64.yaml``
   * - Linux x86_64 (e.g. AWS cluster)
     - ``cicd/environment_cpu_x86_64.yaml``
   * - Linux aarch64
     - ``cicd/environment_cpu_aarch64.yaml``

.. code-block:: bash

   git clone https://github.com/terrahydro-esdt/crest.git
   cd crest
   conda env create -f cicd/environment_macos_arm64.yaml   # adjust for your platform
   conda activate OCETRA_cpu
   pip install -e .

Tutorials
~~~~~~~~~

See the `examples/ <examples/>`_ directory for full worked notebooks:

- `HTG Overview <examples/HTG_overview.ipynb>`_
- `HTG Algebra <examples/HTG_algebra.ipynb>`_
- `HTG Recurrent <examples/HTG_recurrent.ipynb>`_
- `Data Loader Demo <examples/data_loader_demo.ipynb>`_
- `Batcher Demo <examples/batcher_demo.ipynb>`_
- `Archiver Demo <examples/archiver_demo.ipynb>`_
- `Metrics Demo <examples/metrics_demo.ipynb>`_
- `MNIST Demo <examples/mnist_demo.ipynb>`_
- `CIFAR-10 Demo <examples/cifar10_demo.ipynb>`_

Documentation
-------------

Full documentation: https://terrahydro-esdt.github.io/crest/

- `Installation Guide <https://terrahydro-esdt.github.io/crest/quickstart.html>`_
- `Example Notebooks <https://terrahydro-esdt.github.io/crest/examples/index.html>`_
- `API Reference <https://terrahydro-esdt.github.io/crest/api/index.html>`_


Contributing
------------

We welcome contributions! See `CONTRIBUTING.md <CONTRIBUTING.md>`_ for full guidelines.

Quick checklist:

- Code follows style guide (PEP 8, ``black`` formatting)
- Tests added for new functionality
- Documentation updated (docstrings + tutorials if needed)
- All tests pass (``pytest tests/``)
- PR describes changes clearly

Community and Support
---------------------

.. TODO: Confirm all community channels before public release
- **Issues**: `GitHub Issues <https://github.com/terrahydro-esdt/crest/issues>`_
- **Discussions**: `GitHub Discussions <https://github.com/terrahydro-esdt/crest/discussions>`_
- **Email**: crest-dev@example.com

Citation
--------

If you use CREST in your research, please cite:

.. code-block:: bibtex

   @software{crest2024,
     title   = {CREST: Coupled Reusable Earth System Tensor Framework},
     author  = {Pelissier, Craig and {CREST Development Team}},
     year    = {2024},
     url     = {https://github.com/terrahydro-esdt/crest},
     version = {0.1.0}
   }

See `CITATION.cff <CITATION.cff>`_ for complete citation information.

License
-------

CREST is licensed under the Apache License 2.0. See `LICENSE <LICENSE>`_ for details.

Acknowledgments
---------------

.. TODO: Add full funding agency, institution, and collaborating organization names before public release
This project is supported by NASA.
