=================
Exporting Models
=================

This page describes how to take a model out of Model Composer for deployment on a TI MCU, and how to continue in the
command line with Tiny ML ModelZoo.

.. note::

   The export options below follow the 1.4.0 guide and were **not re-verified against the current web UI**. Check the
   **Export** page of the live application for the formats it offers.

Export options
--------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Format
     - Description
   * - CCS project
     - A Code Composer Studio project that you import and build.
   * - Artifacts only
     - Compiled model library and headers for use in an existing project.
   * - ONNX model
     - Float and/or quantized ONNX file for analysis or custom pipelines.
   * - Configuration
     - YAML file for the command line flow.

Importing a CCS project
-----------------------

1. Download the exported ZIP.
2. In Code Composer Studio choose **File > Import > CCS Projects** and select the archive.
3. Build and flash.

See :doc:`/deployment/ccs_integration`.

Using artifacts in an existing project
--------------------------------------

The compiled model is ``mod.a`` with the interface header ``tvmgen_default.h`` (as in the command line flow, see
:doc:`/DEPLOYING_TO_SDK_EXAMPLE`). Add the files to your project, set the include path, add the library to the linker
settings and call the model from your code. Use the feature extraction code and test vectors from the same export, because
files from different runs do not match.

Continuing in the command line
------------------------------

Use the exported configuration to reproduce or refine the run with Tiny ML ModelZoo 1.5.0:

1. Install Tiny ML ModelZoo as described in :doc:`/installation/index`. For quick use, install the wheel
   (``pip install http://software-dl.ti.com/C2000/esd/mcu_ai/wheel/tinyml_modelzoo-1.5.0-py3-none-any.whl``). To add
   or change models, clone ``https://github.com/TexasInstruments/tinyml-modelzoo`` and run ``pip install -e .`` there.
2. Copy the exported YAML into a working folder and check it against :doc:`/CONFIG_YAML_GUIDE`. Make sure that
   ``dataset.input_data_path`` points to your dataset, and that ``training.model_name`` is a model from
   :doc:`/appendix/model_zoo_reference`. A configuration exported from an older Model Composer version may use names
   that were removed.
3. Run it:

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         cd tinyml-modelzoo
         ./run_tinyml_modelzoo.sh /path/to/exported_config.yaml

   .. tab:: Windows (cmd)

      .. code-block:: bat

         cd tinyml-modelzoo
         run_tinyml_modelzoo.bat C:\path\to\exported_config.yaml

   .. tab:: Wheel install (no clone)

      .. code-block:: bash

         python -m tinyml_modelmaker.run_tinyml_modelmaker /path/to/exported_config.yaml

Results are written to ``./data/projects/<project_name>/run/<date-time>/<model_name>/``. See
:doc:`/getting_started/understanding_run_output`.

Example of an exported configuration (shape only; the values are from a 1.4.0 example):

.. code-block:: yaml

   common:
     task_type: 'generic_timeseries_classification'
     target_device: 'F28P55'
   dataset:
     dataset_name: 'dc_arc_fault_example_dsk'
   data_processing_feature_extraction:
     feature_extraction_name: 'FFT1024Input_256Feature_1Frame_Full_Bandwidth'
     variables: 1
   training:
     model_name: 'CLS_4k_NPU'
     training_epochs: 30
     batch_size: 256
   testing:
     enable: True
   compilation:
     enable: True

Export troubleshooting
----------------------

* Export is unavailable until training and compilation have finished.
* If a CCS import fails, check the CCS and SDK versions, or use the artifacts-only export.
* If the model does not behave on the device, check that the target device matches and compare against the golden
  vectors. See :doc:`/troubleshooting/common_errors`.

Next steps
----------

* :doc:`/deployment/ccs_integration`
* :doc:`/deployment/npu_device_deployment`
* :doc:`/deployment/non_npu_deployment`
