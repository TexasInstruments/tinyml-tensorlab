==========
Quickstart
==========

From zero to a trained and compiled model in a few minutes.

Prerequisites
-------------

* Python 3.10 (needed for the compile step; see :doc:`/installation/prerequisites`)
* Tiny ML ModelZoo installed (see :doc:`/installation/index`)
* For the compile step: the compiler for your target device (see :doc:`/installation/environment_variables`).
  The hello world example targets the F28P55, so it needs ``C2000_CG_ROOT``
  (default ``~/ti/ti-cgt-c2000_25.11.1.LTS``).

Step 1: Get the Example Config
------------------------------

With the **clone install** the example is already in ``examples/``:

.. code-block:: bash

   cd tinyml-modelzoo

With the **wheel install** the ``examples/`` folder is not part of the package. Download
``examples/generic_timeseries_classification/config.yaml`` from the
`GitHub repository <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/generic_timeseries_classification>`_
into any folder.

Step 2: Run Hello World
-----------------------

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

   .. tab:: Wheel install

      .. code-block:: bash

         python -m tinyml_modelmaker.run_tinyml_modelmaker config.yaml

.. tip::

   For MSPM0 devices use ``config_MSPM0.yaml`` where it exists, for example
   ``examples/generic_timeseries_classification/config_MSPM0.yaml``. Connectivity devices have
   ``config_CC1352.yaml``, ``config_CC1354.yaml``, ``config_CC2755.yaml`` and ``config_CC35X1.yaml``.

The run:

1. Downloads the waveform dataset (sine, square, sawtooth)
2. Applies FFT-based feature extraction (``Generic_1024Input_FFTBIN_64Feature_8Frame``)
3. Trains ``CLS_1k_NPU`` (early stopping is on, with patience 5)
4. Quantizes the model for the MCU
5. Compiles for the F28P55 (NPU device)

Step 3: View Results
--------------------

Results are in::

   ./data/projects/generic_timeseries_classification/run/<date-time>/CLS_1k_NPU/

relative to the folder you ran the command from (change with ``common.projects_path``).
The layout is:

.. code-block:: text

   <date-time>/CLS_1k_NPU/
   ├── run.yaml, status.yaml              # resolved parameters and stage status
   ├── training/
   │   ├── base/                          # float model
   │   └── quantization/                  # quantized model (the one you deploy)
   └── compilation/
       ├── artifacts/mod.a, tvmgen_default.h
       └── <date-time>_CLS_1k_NPU_F28P55.zip

:doc:`understanding_run_output` explains every file.

Step 4: Read the Config
-----------------------

``examples/generic_timeseries_classification/config.yaml`` (comments omitted):

.. code-block:: yaml

   common:
     task_type: generic_timeseries_classification
     target_device: F28P55
   dataset:
     dataset_name: generic_timeseries_classification
   data_processing_feature_extraction:
     feature_extraction_name: Generic_1024Input_FFTBIN_64Feature_8Frame
     variables: 1
   training:
     model_name: CLS_1k_NPU
     batch_size: 256
     training_epochs: 20
     early_stopping: true
     early_stopping_patience: 5
     num_gpus: 0
   testing: {}
   compilation: {}

* ``task_type``: the ML task
* ``target_device``: the MCU to compile for
* ``dataset_name``: a built-in dataset preset, downloaded automatically
* ``feature_extraction_name``: how raw data is preprocessed
* ``model_name``: the model architecture from the model zoo

Step 5: Try Another Example
---------------------------

.. code-block:: bash

   ./run_tinyml_modelzoo.sh examples/dc_arc_fault/config_dsk.yaml

Note that some example folders have no plain ``config.yaml``; look at the files in the
folder (for ``dc_arc_fault`` there are ``config_dsk.yaml``, ``config_dsi.yaml`` and others).

Next Steps
----------

* :doc:`first_example` - Detailed walkthrough
* :doc:`understanding_run_output` - Read the outputs
* :doc:`understanding_config` and :doc:`/CONFIG_YAML_GUIDE` - Config reference
* :doc:`/DEPLOYING_TO_SDK_EXAMPLE` - Run the result on a board
