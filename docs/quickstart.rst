Quick Start
============

Installation from source
------------------------

For developers with SSH access:

.. code-block::

   git@ssh.gitlab.smce.nasa.gov:astg/terrahydro/development/crest.git

cd into the code repo:

.. code-block::

   cd crest
   
Create the Python environment:

.. code-block::

   conda env create -f cicd/environment.yaml

Once the installation has finished building, *activate* the installed environment by running:

.. code-block::

   conda activate crest_cpu

Install CREST in development mode:

.. code-block::

   pip install -e .


