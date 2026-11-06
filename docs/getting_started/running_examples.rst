================
Running Examples
================

How to find, run and customize the example configurations in ``tinyml-modelzoo``.

Finding Examples
----------------

Examples are in the ``examples/`` folder of the repository (clone install), or browsable
on `GitHub <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples>`_
(wheel install). Each folder holds one or more config files and usually a ``readme.md``.

.. code-block:: bash

   ls examples/

Running the script without arguments lists every config it can find:

.. code-block:: bash

   ./run_tinyml_modelzoo.sh

See :doc:`/examples/index` for the guide to each example.

The Three Run Scripts
---------------------

Run them from the ``tinyml-modelzoo`` folder. They check that ``tinyml_modelmaker`` is
installed, resolve the config path (relative to the current folder or to the script
folder) and run ``python -m tinyml_modelmaker.run_tinyml_modelmaker <config> [extra args]``.

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Platform
     - Command
   * - Linux / macOS
     - ``./run_tinyml_modelzoo.sh examples/<example_name>/config.yaml``
   * - Windows cmd
     - ``run_tinyml_modelzoo.bat examples\<example_name>\config.yaml``
   * - Windows PowerShell
     - ``./run_tinyml_modelzoo.ps1 examples/<example_name>/config.yaml``
   * - Wheel install (no script)
     - ``python -m tinyml_modelmaker.run_tinyml_modelmaker config.yaml``

Examples:

.. code-block:: bash

   ./run_tinyml_modelzoo.sh examples/generic_timeseries_classification/config.yaml   # hello world
   ./run_tinyml_modelzoo.sh examples/dc_arc_fault/config_dsk.yaml                    # DC arc fault
   ./run_tinyml_modelzoo.sh examples/motor_bearing_fault/config.yaml                 # bearing faults
   ./run_tinyml_modelzoo.sh examples/forecasting_pmsm_rotor_temp/config.yaml         # forecasting

.. note::

   Config file names vary by example and device: ``config.yaml`` (default, usually F28P55),
   ``config_MSPM0.yaml``, ``config_CC1352.yaml`` and so on. Some folders have no plain
   ``config.yaml``. Check the folder.

Reading an Example Config
-------------------------

.. code-block:: yaml

   common:
     task_type: generic_timeseries_classification
     target_device: F28P55
   dataset:
     dataset_name: generic_timeseries_classification   # built-in preset, downloaded automatically
   data_processing_feature_extraction:
     feature_extraction_name: Generic_1024Input_FFTBIN_64Feature_8Frame
     variables: 1                                      # number of input channels
   training:
     model_name: CLS_1k_NPU
     training_epochs: 20

See :doc:`understanding_config` and the :doc:`/CONFIG_YAML_GUIDE`.

Customizing
-----------

.. code-block:: yaml

   common:
     target_device: MSPM0G3507         # another device

   training:
     model_name: CLS_4k_NPU            # another model
     training_epochs: 50

   compilation:
     enable: False                     # skip compilation for faster iteration

Output Location
---------------

All output goes to::

   ./data/projects/<project_name>/run/<date-time>/<model_name>/

relative to the directory you ran the script from. ``<project_name>`` is the
dataset name (for example ``dc_arc_fault_example_dsk``). The ``run_name`` default is
``{date-time}/{model_name}``. To write somewhere else, set ``projects_path`` in the config:

.. code-block:: yaml

   common:
     projects_path: './your/choice'     # or an absolute path

Downloaded datasets are stored under the project folder (``<project_name>/dataset``).
See :doc:`understanding_run_output` for what is in a run folder.

Troubleshooting
---------------

**Dataset download failed**
   Check your network or proxy. Download the dataset manually and point the config
   at it: ``dataset: {input_data_path: '/path/to/extracted/dataset'}``.

**Out of memory**
   Reduce ``training.batch_size``.

**Compilation failed**
   Check the compiler environment variable (:doc:`/installation/environment_variables`),
   or set ``compilation: {enable: False}`` to train only.

**tinyml_modelmaker is not installed**
   Activate your virtual environment and run ``pip install -e .`` in ``tinyml-modelzoo``.

Next Steps
----------

* :doc:`/byod/index` - use your own data
* :doc:`/examples/index` - example guides
* :doc:`/features/index` - advanced features
