=================
User Installation
=================

This guide covers the quick installation method for users who want to use
Tiny ML Tensorlab without modifying the source code.

.. note::
   This installation method provides read-only access to the toolchain.
   If you need to customize models or add new features, use
   :doc:`developer_installation` instead.

Quick Install
-------------

Install Tiny ML Tensorlab directly from GitHub:

.. code-block:: bash

   pip install git+https://github.com/TexasInstruments/tinyml-tensorlab.git@main#subdirectory=tinyml-modelmaker

This installs:

* ``tinyml-modelmaker`` - The main orchestration tool
* ``tinyml-tinyverse`` - Core training infrastructure
* ``tinyml-modelzoo`` - Model definitions
* ``tinyml-modeloptimization`` - Quantization toolkit

Running Your First Example
--------------------------

.. warning::

   **IMPORTANT: Environment Variables Required for Model Compilation**

   For AI model compilation to work, you MUST set environment variables
   specific to your target device **before running examples**.

   The variables you need depend on which device you're targeting:

   * **C2000 devices (F28P55, F28P65, etc.)**: Set ``C2000_CG_ROOT``
   * **F29 devices (F29H85X, etc.)**: Set ``CG_TOOL_ROOT``
   * **MSPM0 devices**: Set ``ARM_LLVM_CGT_PATH``
   * **AM13E devices**: Set ``ARM_LLVM_CGT_PATH``
   * **AM26x devices**: Set ``ARM_LLVM_CGT_PATH``
   * **Connectivity devices (CC2755, CC1352, etc.)**: Set ``ARM_LLVM_CGT_PATH``

   See :doc:`environment_variables` for complete device-specific setup instructions.

After setting environment variables for your target device, run the hello world example:

.. note::

   **Windows Users: Enable Long Path Support Before Cloning**

   If you are on Windows, run this command before cloning to enable support for long file paths:

   .. code-block:: powershell

      git config --global core.longpaths true

   This allows Git to handle the deeply nested directory structure in the repository. You only need to run this once per system.

.. code-block:: bash

   # Clone just the examples
   git clone --depth 1 https://github.com/TexasInstruments/tinyml-tensorlab.git
   cd tinyml-tensorlab/tinyml-modelzoo

   # Run the example (trains and compiles for the device specified in config.yaml)
   python -m tinyml_modelmaker examples/generic_timeseries_classification/config.yaml

Output will be saved to ``../tinyml-modelmaker/data/projects/``.

Verifying Installation
----------------------

Verify the installation by importing the packages and checking versions:

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

Updating
--------

To update to the latest version:

.. code-block:: bash

   pip install --upgrade git+https://github.com/TexasInstruments/tinyml-tensorlab.git@main#subdirectory=tinyml-modelmaker

Uninstalling
------------

To remove Tiny ML Tensorlab:

.. code-block:: bash

   pip uninstall tinyml-modelmaker tinyml-tinyverse tinyml-modelzoo tinyml-torchmodelopt

Limitations of User Install
---------------------------

The pip install method has some limitations:

* Cannot modify model architectures
* Cannot add custom feature extractors
* Cannot debug training scripts
* Updates require reinstallation

For full access, use :doc:`developer_installation`.

Troubleshooting
---------------

**"No module named tinyml_modelmaker"**

Ensure you're using the correct Python environment:

.. code-block:: bash

   which python  # Should point to your Python 3.10 installation
   python --version  # Should show 3.10.x

**Version conflicts**

If you have dependency conflicts, try installing in a virtual environment:

.. code-block:: bash

   python -m venv tensorlab_env
   source tensorlab_env/bin/activate  # Linux
   # or: tensorlab_env\Scripts\activate  # Windows

   pip install git+https://github.com/TexasInstruments/tinyml-tensorlab.git@main#subdirectory=tinyml-modelmaker

Next Steps
----------

* :doc:`/getting_started/quickstart` - Train your first model
* :doc:`environment_variables` - Configure compilation tools
* :doc:`developer_installation` - Full installation for customization
