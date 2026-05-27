.. TODO: Uncomment badges once the repository is public and DOI is registered

.. |docs| image:: https://readthedocs.org/projects/crest/badge/?version=latest
   :target: https://crest.readthedocs.io/en/latest/
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

The **CREST framework** provides a flexible, scalable infrastructure for developing Earth
system models that seamlessly integrate machine learning with physics-based components.
Built on TensorFlow, CREST enables researchers to:

- **Build modular Earth system models** using hierarchical tensor graphs
- **Combine ML and physics-based processes** in a unified framework
- **Process large-scale gridded datasets** efficiently with optimized data pipelines
- **Deploy models** from research to operational systems
- **Share and reuse components** across the research community

Key Features
~~~~~~~~~~~~

- **Hierarchical Tensor Graphs (HTG)**: Directed acyclic graph structure for flexible model composition
- **Hybrid Modeling**: Seamlessly mix neural networks, physical equations, and empirical relationships
- **Scalable Data Pipeline**: Efficient batching and loading for large Earth science datasets
- **Model Serialization**: Save, version, and share complete model configurations
- **Distributed Computing**: Built-in support for multi-GPU and distributed training

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

   git clone https://gitlab.smce.nasa.gov/astg/terrahydro/development/crest   # TODO: change URL at release
   cd crest
   conda env create -f cicd/environment_macos_arm64.yaml   # adjust for your platform
   conda activate OCETRA_cpu
   pip install -e .

Your First Model
~~~~~~~~~~~~~~~~

.. code-block:: python

   from crest.model import HierarchalTensorGraph, Node, TensorSpec, Model

   # Node callables must accept a dict of tensors and return a dict of tensors.
   # TensorSpec takes positional args: TensorSpec(shape, dtype).
   def compute_pet(X):
       return {'pet': 0.5 * X['temperature'] + 2.0}

   # Define the node with explicit I/O specs
   pet_node = Node(
       node=compute_pet,
       inputs={'temperature': TensorSpec((None, 100), 'float32')},
       outputs={'pet': TensorSpec((None, 100), 'float32')},
       name='pet'  # potential evapotranspiration
   )

   # Wire the node into the graph with add_edge.
   # 'input' and 'output' are the HTG's external interface.
   htg = HierarchalTensorGraph(name='soil_moisture_model')
   htg.add_edge('input', pet_node)
   htg.add_edge(pet_node, 'output')

   # Build and train
   # training_data is a (inputs_dict, targets_dict) tuple matched to the HTG's I/O keys
   import numpy as np
   training_data = (
       {'temperature': np.random.rand(200, 100).astype('float32')},
       {'pet':         np.random.rand(200, 100).astype('float32')},
   )

   model = Model(htg)
   model.compile(optimizer='adam', loss='mse')
   model.fit(training_data, epochs=10)

See the `examples/ <examples/>`_ directory for full worked notebooks.

Documentation
-------------

.. TODO: change URL at release
Full documentation: https://astg.pages.smce.nasa.gov/terrahydro/development/crest  

- `Installation Guide <https://astg.pages.smce.nasa.gov/terrahydro/development/crest/quickstart.html>`_
- `Example Notebooks <https://astg.pages.smce.nasa.gov/terrahydro/development/crest/examples/index.html>`_
- `API Reference <https://astg.pages.smce.nasa.gov/terrahydro/development/crest/api/index.html>`_

Architecture
------------

CREST follows a modular, layered architecture:

.. code-block:: text

   ┌─────────────────────────────────────────────────┐
   │               Application Layer                 │
   │       (User Models, Workflows, Experiments)     │
   └─────────────────────────────────────────────────┘
                          ↓
   ┌─────────────────────────────────────────────────┐
   │                  Model Layer                    │
   │       HierarchalTensorGraph (HTG) · Model       │
   │  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
   │  │   Node   │  │Recurrent │  │ Prebuilt TF  │   │
   │  │(callable)│  │   Node   │  │Nodes (LSTM…) │   │
   │  └──────────┘  └──────────┘  └──────────────┘   │
   └─────────────────────────────────────────────────┘
                          ↓
   ┌─────────────────────────────────────────────────┐
   │                  Engine Layer                   │
   │      ModelEngine · DataServer · Archiver        │
   │   (training/inference orchestration and output) │
   └─────────────────────────────────────────────────┘
                          ↓
   ┌──────────────────────────────────────────────────┐
   │                   Data Layer                     │
   │   Dataset · Block · Blockset · StructuredDataset │
   │         Batcher · MultiBatcher · Transform       │
   └──────────────────────────────────────────────────┘
                          ↓
   ┌─────────────────────────────────────────────────┐
   │                 Storage Layer                   │
   │               Zarr | TileDB | NetCDF            │
   └─────────────────────────────────────────────────┘

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
- **Issues**: `GitHub Issues <https://github.com/terrahydro/crest/issues>`_
- **Discussions**: `GitHub Discussions <https://github.com/terrahydro/crest/discussions>`_
- **Email**: crest-dev@example.com

Citation
--------

If you use CREST in your research, please cite:

.. code-block:: bibtex

   @software{crest2024,
     title   = {CREST: Coupled Reusable Earth System Tensor Framework},
     author  = {Pelissier, Craig and {CREST Development Team}},
     year    = {2024},
     url     = {https://github.com/terrahydro/crest},
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
