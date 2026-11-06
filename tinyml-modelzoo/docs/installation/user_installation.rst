==================
Wheel Installation
==================

Use this path if you plan to make **no** model additions or changes. It is one
``pip install`` of the ``tinyml_modelzoo`` wheel.

Quick Install
-------------

Create and activate a virtual environment (see :doc:`prerequisites`), then run:

.. code-block:: bash

   pip install http://software-dl.ti.com/C2000/esd/mcu_ai/wheel/tinyml_modelzoo-1.5.0-py3-none-any.whl

The wheel declares these dependencies, so pip installs them for you:

* ``tinyml_modelmaker`` 1.5.0 (downloaded from the same TI server). It in turn pulls in
  ``tinyml_tinyverse`` (training), ``tinyml_torchmodelopt`` (quantization) and
  ``ti_mcu_nnc`` (the TI Neural Network Compiler).
* ``torch==2.11.0``, ``PyYAML==6.0.3`` and a pinned ``numpy``.

What Is in the Wheel
--------------------

The wheel contains only the Python package ``tinyml_modelzoo``: the model
definitions (``tinyml_modelzoo.models``), the model descriptions that map the
``training.model_name`` values in a config to models, and the custom layers.
It does **not** contain:

* the ``examples/`` folder with ready-to-run configs, and
* the ``run_tinyml_modelzoo.sh`` / ``.bat`` / ``.ps1`` wrapper scripts.

This is by design (it is a Python package), but it means a wheel-only user
supplies their own config file.

Running with the Wheel Install
------------------------------

Write a config YAML (see :doc:`/CONFIG_YAML_GUIDE`), or download one of the example
configs from the
`examples folder on GitHub <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples>`_
(for instance ``examples/generic_timeseries_classification/config.yaml``). Then call
the ModelMaker entry point directly. This is exactly what the wrapper scripts do:

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         python -m tinyml_modelmaker.run_tinyml_modelmaker config.yaml

   .. tab:: Windows

      .. code-block:: powershell

         python -m tinyml_modelmaker.run_tinyml_modelmaker config.yaml

Output is written to ``./data/projects/<project_name>/run/<date-time>/<model_name>/``
relative to the folder you run the command from; set ``common.projects_path`` in the
config to change it. Datasets named in the config are downloaded automatically.

.. warning::

   To compile for a device, set the compiler environment variable for your target
   **before** running. See :doc:`environment_variables`. Training and quantization work without it.

Verifying Installation
----------------------

.. code-block:: python

   import tinyml_modelmaker
   import tinyml_tinyverse
   import tinyml_torchmodelopt
   import tinyml_modelzoo

   print(f"TI Tiny ML ModelMaker: {tinyml_modelmaker.__version__}")
   print(f"TI Tiny ML Tinyverse: {tinyml_tinyverse.__version__}")
   print(f"TI Tiny ML Model Optimization toolkit: {tinyml_torchmodelopt.__version__}")
   print(f"TI Tiny ML Model Zoo: {tinyml_modelzoo.__version__}")

All four should print ``1.5.0``-series version numbers.

Updating and Uninstalling
-------------------------

To move to a newer release, install the newer wheel URL the same way
(add ``--upgrade``). To remove the toolchain:

.. code-block:: bash

   pip uninstall tinyml-modelzoo tinyml-modelmaker tinyml-tinyverse tinyml-torchmodelopt

Limitations of the Wheel Install
--------------------------------

* You cannot add or edit models without reinstalling (use :doc:`developer_installation`).
* The shipped ``examples/`` and wrapper scripts are not included.

Troubleshooting
---------------

**"No module named tinyml_modelmaker"**
   You are in a different Python environment than the one you installed into. Check
   ``which python`` (Linux/macOS) or ``where python`` (Windows) and ``python --version``.

**ti_mcu_nnc is not installed / compilation is unavailable**
   The TI Neural Network Compiler wheel is published for Python 3.10 only. On
   Python 3.11 or later training works, but compilation does not. See :doc:`prerequisites`.

**Version conflicts**
   Install into a fresh virtual environment:

   .. code-block:: bash

      python -m venv venv
      source venv/bin/activate      # Windows: venv\Scripts\activate
      pip install http://software-dl.ti.com/C2000/esd/mcu_ai/wheel/tinyml_modelzoo-1.5.0-py3-none-any.whl

Next Steps
----------

* :doc:`environment_variables` - Configure compiler tools
* :doc:`/getting_started/quickstart` - Train your first model
* :doc:`developer_installation` - Clone install for adding models
