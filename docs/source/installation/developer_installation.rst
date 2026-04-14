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
         pip install -e tinyml-modelmaker
         pip install -e tinyml-tinyverse
         pip install -e tinyml-modeloptimization/torchmodelopt
         pip install -e tinyml-modelzoo

   .. tab:: Windows

      .. code-block:: powershell

         # Create virtual environment
         python -m venv venv
         .\venv\Scripts\Activate.ps1

         # Upgrade pip
         python -m pip install --upgrade pip

         # Install components in order
         pip install -e tinyml-modelmaker
         pip install -e tinyml-tinyverse
         pip install -e tinyml-modeloptimization\torchmodelopt
         pip install -e tinyml-modelzoo

Step 3: Verify Installation
---------------------------

Verify all components are installed by importing the packages and checking versions:

.. code-block:: python

   import tinyml_modelmaker
   import tinyml_tinyverse
   import tinyml_torchmodelopt
   import tinyml_modelzoo

   print(f"TI Tiny ML ModelMaker: {tinyml_modelmaker.__version__}")
   print(f"TI Tiny ML Tinyverse: {tinyml_tinyverse.__version__}")
   print(f"TI Tiny ML Model Optimization toolkit: {tinyml_torchmodelopt.__version__}")
   print(f"TI Tiny ML Model Zoo: {tinyml_modelzoo.__version__}")

If all packages import without errors and versions are displayed, your installation is complete.

Step 4: Configure Environment Variables
----------------------------------------

.. warning::

   **IMPORTANT: Environment Variables Required for Model Compilation**

   For AI model compilation to work, you MUST set environment variables
   specific to your target device **before running examples**.

   The variables you need depend on which device you're targeting:

   * **C2000 devices (F28P55, F28P65, etc.)**: Set ``C2000_CG_ROOT`` and ``C2000WARE_ROOT``
   * **F29 devices (F29H85X, etc.)**: Set ``CG_TOOL_ROOT``
   * **MSPM0 devices**: Set ``ARM_LLVM_CGT_PATH``
   * **AM13E devices**: Set ``ARM_LLVM_CGT_PATH``
   * **AM26x devices**: Set ``ARM_LLVM_CGT_PATH``
   * **Connectivity devices (CC2755, CC1352, etc.)**: Set ``ARM_LLVM_CGT_PATH``

   See :doc:`environment_variables` for complete device-specific setup instructions.

Step 5: Run the hello world example:

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
   pip install -e tinyml-modelmaker
   pip install -e tinyml-tinyverse
   pip install -e tinyml-modeloptimization/torchmodelopt
   pip install -e tinyml-modelzoo

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

Speeding Up Installation
------------------------

The developer installation can take significant time due to downloading large dependencies (PyTorch, TensorFlow, etc.). Here are several ways to speed it up:

**Option 1: Use setup_all.sh with Parallel Installation (Linux)**

The ``setup_all.sh`` script automatically uses parallel builds:

.. code-block:: bash

   cd tinyml-modelmaker
   ./setup_all.sh

This is typically 2-3x faster than manual installation.

**Option 2: Enable pip Caching**

Create or update ``~/.pip/pip.conf`` to enable aggressive caching:

.. code-block:: ini

   [global]
   cache-dir = ~/.cache/pip
   no-cache-dir = False

This caches downloaded packages so reinstalls are faster.

**Option 3: Use Pre-built Wheels**

Ensure pip uses pre-built wheels instead of compiling from source:

.. code-block:: bash

   pip install --upgrade pip wheel
   pip install -e tinyml-modelmaker  # Will use wheels if available

**Option 4: Skip Optional GPU Dependencies (If Not Needed)**

If you don't need GPU support, you can use CPU-only PyTorch:

.. code-block:: bash

   # CPU-only PyTorch (faster to install)
   pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cpu

   # Then install components
   pip install -e tinyml-modelmaker
   pip install -e tinyml-tinyverse
   pip install -e tinyml-modeloptimization/torchmodelopt
   pip install -e tinyml-modelzoo

**Option 5: Use Faster Package Index**

Some regions may have mirror repositories. For example, in China:

.. code-block:: bash

   pip install -i https://pypi.tsinghua.edu.cn/simple -e tinyml-modelmaker

.. note::

   **Installation Times (Approximate)**

   * Full developer install (first time): 15-30 minutes
   * With caching enabled (subsequent installs): 5-10 minutes
   * Using setup_all.sh (Linux): 10-15 minutes

.. tip::

   **Checking Download Progress**

   To see real-time download progress:

   .. code-block:: bash

      pip install --verbose -e tinyml-modelmaker

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
