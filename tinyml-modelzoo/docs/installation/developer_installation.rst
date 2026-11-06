==================
Clone Installation
==================

Use this path if you want the ``examples/`` folder and wrapper scripts, or you plan to
**add or change models** (including :doc:`/BRING_YOUR_OWN_MODEL`). You clone one
repository, ``tinyml-modelzoo``, and install it in editable mode.

Step 1: Clone the Repository
----------------------------

.. note::

   **Windows:** enable long paths once per system before cloning:
   ``git config --global core.longpaths true``

.. code-block:: bash

   git clone https://github.com/TexasInstruments/tinyml-modelzoo.git
   cd tinyml-modelzoo

Only this repository is needed. Do not clone ``tinyml-modelmaker``, ``tinyml-tinyverse``
or ``tinyml-modeloptimization``, and do not use ``tinyml-tensorlab`` (deprecated, its
source is private).

Step 2: Create a Virtual Environment and Install
------------------------------------------------

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         python3.10 -m venv venv
         source venv/bin/activate
         python -m pip install --upgrade pip setuptools wheel
         pip install -e .

   .. tab:: Windows (PowerShell)

      .. code-block:: powershell

         python -m venv venv
         .\venv\Scripts\Activate.ps1
         python -m pip install --upgrade pip setuptools wheel
         pip install -e .

``pip install -e .`` installs ``tinyml_modelzoo`` from your clone (edits to
``tinyml_modelzoo/models`` take effect immediately) and downloads ``tinyml_modelmaker``
as a prebuilt wheel from TI's server. ModelMaker brings in ``tinyml_tinyverse``,
``tinyml_torchmodelopt`` and ``ti_mcu_nnc``.

Step 3: Verify Installation
---------------------------

.. code-block:: python

   import tinyml_modelmaker, tinyml_tinyverse, tinyml_torchmodelopt, tinyml_modelzoo
   print(tinyml_modelmaker.__version__, tinyml_tinyverse.__version__,
         tinyml_torchmodelopt.__version__, tinyml_modelzoo.__version__)

Step 4: Configure the Compiler
------------------------------

Set the environment variable for your target device so ModelMaker can compile. See
:doc:`environment_variables`.

Step 5: Run the Hello World Example
-----------------------------------

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         ./run_tinyml_modelzoo.sh examples/generic_timeseries_classification/config.yaml

   .. tab:: Windows (cmd)

      .. code-block:: bat

         run_tinyml_modelzoo.bat examples\generic_timeseries_classification\config.yaml

   .. tab:: Windows (PowerShell)

      .. code-block:: powershell

         ./run_tinyml_modelzoo.ps1 examples/generic_timeseries_classification/config.yaml

The scripts only check that ``tinyml_modelmaker`` can be imported and then run
``python -m tinyml_modelmaker.run_tinyml_modelmaker <config>``. Output goes to
``./data/projects/`` relative to the directory you ran the script from.

Folder Layout
-------------

.. code-block:: text

   tinyml-modelzoo/
   ├── examples/                # ready-to-run configs
   ├── tinyml_modelzoo/
   │   ├── models/              # model definitions
   │   └── model_descriptions/  # model names used in configs
   ├── docs/                    # guides
   ├── run_tinyml_modelzoo.sh / .bat / .ps1
   ├── estimate_memory.py / .sh
   └── ADDING_NEW_MODELS.md

Common Developer Tasks
----------------------

* **Add a model**: see :doc:`/byom/adding_models` and :doc:`/BRING_YOUR_OWN_MODEL`;
  the model files live in ``tinyml_modelzoo/models/`` and are registered in
  ``tinyml_modelzoo/model_descriptions/``.
* **Add your own example**: copy an ``examples/<name>/config.yaml`` and adapt it
  (:doc:`/CONFIG_YAML_GUIDE`).

Updating
--------

.. code-block:: bash

   git pull origin main
   pip install -e .

Troubleshooting
---------------

**"Error: tinyml_modelmaker is not installed"** (from the run script)
   Activate your virtual environment and run ``pip install -e .`` inside ``tinyml-modelzoo``.

**Permission denied on the .sh script**
   ``chmod +x run_tinyml_modelzoo.sh``

**PowerShell blocks the .ps1 script**
   ``Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser``

Next Steps
----------

* :doc:`environment_variables`
* :doc:`/getting_started/quickstart`
