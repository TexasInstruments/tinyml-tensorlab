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

After installation, you can run the hello world example:

.. code-block:: bash

   # Clone just the examples
   git clone --depth 1 https://github.com/TexasInstruments/tinyml-tensorlab.git
   cd tinyml-tensorlab/tinyml-modelzoo

   # Run the example
   python -m tinyml_modelmaker examples/generic_timeseries_classification/config.yaml

Output will be saved to ``../tinyml-modelmaker/data/projects/``.

Verifying Installation
----------------------

Verify the installation by importing the packages:

.. code-block:: python

   import tinyml_modelmaker
   import tinyml_tinyverse
   import tinyml_modelzoo
   import tinyml_torchmodelopt

   print("Installation successful!")

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
