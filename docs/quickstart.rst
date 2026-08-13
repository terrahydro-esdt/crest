Quick Start
============

Installation
------------

Public install (coming at release)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once CREST is published to PyPI, installation will be a single command:

.. code-block:: bash

   pip install crest

Developer install (current)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Clone the repository:

.. code-block:: bash

   git clone https://github.com/terrahydro-esdt/crest.git
   cd crest

Create the Python environment. Choose the file that matches your platform:

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

   conda env create -f cicd/environment_macos_arm64.yaml   # adjust for your platform
   conda activate OCETRA_cpu

Install CREST in development mode:

.. code-block:: bash

   pip install -e .

Verify your installation:

.. code-block:: bash

   pytest tests/ -v
