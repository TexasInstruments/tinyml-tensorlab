This guide explains how the tinyml-tensorlab repository is to be set-up once installed.
The guide assumes the tinyml-tensorlab repository is already installed at a particular path.
PLEASE VERIFY INSTALLATION OF THE REPOSITORY:
**Confirm the tinyml-tensorlab installation:**

Ask user: *"What is the full path to your tinyml-tensorlab directory?"*
(e.g., `/home/username/tinyml-tensorlab`)

```bash
TINYML_BASE_PATH=<user-provided path>

# Verify it
python3 $SCRIPTS_DIR/runner.py check_installation \
  "{\"tinyml_base_path\": \"$TINYML_BASE_PATH\"}"
```
The above is already done as part of step 0. But ensure you have this path before moving ahead with the below. IF YOU DONT HAVE IT, THEN ASK THE USER AGAIN. DO NOT PROMPT THE USER FOR IT IF YOU ALREADY HAVE THE PATH.

Before doing the below, make sure you are within the right directory:
```bash
cd $TINYML_BASE_PATH
```

Step 1: Set Up Python Environment
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

         # Upgrade pip and build tools
         pip install --upgrade pip setuptools wheel

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

         # Upgrade pip and build tools
         python -m pip install --upgrade pip setuptools wheel

         # Install components in order
         pip install -e tinyml-modelmaker
         pip install -e tinyml-tinyverse
         pip install -e tinyml-modeloptimization\torchmodelopt
         pip install -e tinyml-modelzoo

Step 2: Verify Installation
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

Step 3: Configure Environment Variables
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

   See :doc:`/references/environment_variables_guide.md` for complete device-specific setup instructions.

Step 4: Run the hello world example:

.. tabs::

   .. tab:: Linux

      .. code-block:: bash

         cd tinyml-modelzoo
         ./run_tinyml_modelzoo.sh examples/generic_timeseries_classification/config.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         run_tinyml_modelzoo.bat examples\generic_timeseries_classification\config.yaml

**REMEMBER TO DO THIS EXPLICITLY**: 
Once step 4 is complete, inform the user that repo setup is complete and that you can proceed with user requests now.
