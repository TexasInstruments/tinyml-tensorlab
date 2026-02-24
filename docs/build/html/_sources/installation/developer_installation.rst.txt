======================
Developer Installation
======================

This guide covers the full installation method for developers who want to
customize models, add features, or contribute to Tiny ML Tensorlab.

Overview
--------

Developer installation involves:

1. Cloning the repository
2. Creating a Python virtual environment
3. Installing all components in editable mode
4. Configuring environment variables

Step 1: Clone the Repository
----------------------------

.. code-block:: bash

   git clone https://github.com/TexasInstruments/tinyml-tensorlab.git
   cd tinyml-tensorlab

Step 2: Set Up Python Environment
---------------------------------

**Option A: Using setup_all.sh (Linux - Recommended)**

The easiest method on Linux:

.. code-block:: bash

   cd tinyml-modelmaker
   ./setup_all.sh

This script:

* Creates a virtual environment using pyenv
* Installs all dependencies
* Installs all components in editable mode

**Option B: Manual Installation**

If the script doesn't work or you're on Windows:

.. tabs::

   .. tab:: Linux

      .. code-block:: bash

         # Create virtual environment
         python -m venv venv
         source venv/bin/activate

         # Upgrade pip
         pip install --upgrade pip

         # Install components in order
         cd tinyml-modelmaker
         pip install -e .

         cd ../tinyml-tinyverse
         pip install -e .

         cd ../tinyml-modeloptimization/torchmodelopt
         pip install -e .

         cd ../../tinyml-modelzoo
         pip install -e .

   .. tab:: Windows

      .. code-block:: powershell

         # Create virtual environment
         python -m venv venv
         .\venv\Scripts\Activate.ps1

         # Upgrade pip
         python -m pip install --upgrade pip

         # Install components in order
         cd tinyml-modelmaker
         pip install -e .

         cd ..\tinyml-tinyverse
         pip install -e .

         cd ..\tinyml-modeloptimization\torchmodelopt
         pip install -e .

         cd ..\..\tinyml-modelzoo
         pip install -e .

Step 3: Verify Installation
---------------------------

Verify all components are installed:

.. code-block:: python

   import tinyml_modelmaker
   import tinyml_tinyverse
   import tinyml_modelzoo
   import tinyml_torchmodelopt

   # Check versions
   print(f"ModelMaker: {tinyml_modelmaker.__version__}")
   print(f"TinyVerse: {tinyml_tinyverse.__version__}")

Run the hello world example:

.. tabs::

   .. tab:: Linux

      .. code-block:: bash

         cd tinyml-modelzoo
         ./run_tinyml_modelzoo.sh examples/generic_timeseries_classification/config.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         run_tinyml_modelzoo.bat examples\generic_timeseries_classification\config.yaml

Directory Structure After Installation
--------------------------------------

.. code-block:: text

   tinyml-tensorlab/
   ├── tinyml-modelmaker/
   │   ├── tinyml_modelmaker/   # Source code (editable)
   │   ├── examples/
   │   ├── docs/
   │   └── data/                # Created on first run
   │       └── projects/        # Training outputs
   ├── tinyml-tinyverse/
   │   └── tinyml_tinyverse/    # Source code (editable)
   ├── tinyml-modeloptimization/
   │   └── torchmodelopt/
   │       └── tinyml_torchmodelopt/  # Source code (editable)
   ├── tinyml-modelzoo/
   │   ├── tinyml_modelzoo/     # Source code (editable)
   │   └── examples/            # Entry point configs
   └── venv/                    # Virtual environment

Updating
--------

**Using git_pull_all.sh (Recommended)**

The simplest way to keep all repositories up to date:

.. code-block:: bash

   ./git_pull_all.sh

This script pulls the latest changes for all sub-repositories in one step.

**Manual Update**

Alternatively, update each component manually:

.. code-block:: bash

   # Update from GitHub
   git pull origin main

   # Reinstall components if dependencies changed
   cd tinyml-modelmaker && pip install -e .
   cd ../tinyml-tinyverse && pip install -e .
   cd ../tinyml-modeloptimization/torchmodelopt && pip install -e .
   cd ../../tinyml-modelzoo && pip install -e .

Common Developer Tasks
----------------------

**Adding a New Model**

Edit files in ``tinyml-modelzoo/tinyml_modelzoo/models/``:

.. code-block:: python

   # tinyml_modelzoo/models/classification.py
   class MY_NEW_MODEL(GenericModelWithSpec):
       ...

**Modifying Training Scripts**

Edit files in ``tinyml-tinyverse/tinyml_tinyverse/references/``:

.. code-block:: bash

   # Example: classification training
   vim tinyml-tinyverse/tinyml_tinyverse/references/timeseries_classification/train.py

**Adding Custom Transforms**

Edit files in ``tinyml-tinyverse/tinyml_tinyverse/common/transforms/``.

Troubleshooting
---------------

**"ModuleNotFoundError" after installation**

Ensure you're in the correct virtual environment:

.. code-block:: bash

   which python  # Should point to your venv
   source venv/bin/activate  # Reactivate if needed

**Dependency conflicts**

Try reinstalling in a fresh environment:

.. code-block:: bash

   rm -rf venv
   python -m venv venv
   source venv/bin/activate
   # Reinstall all components

**Permission errors on Linux**

Make scripts executable:

.. code-block:: bash

   chmod +x tinyml-modelzoo/run_tinyml_modelzoo.sh
   chmod +x tinyml-modelmaker/run_tinyml_modelmaker.sh

Next Steps
----------

* :doc:`environment_variables` - Configure compilation tools
* :doc:`/getting_started/quickstart` - Train your first model
* :doc:`/byom/adding_models` - Add custom models
