======================================
Generic Time Series Classification
======================================

This example is the "Hello World" of Tiny ML ModelZoo time series classification. It classifies
synthetic sine, square and sawtooth waveforms and shows the complete flow (data, feature
extraction, training, quantization, compilation) with the smallest possible configuration.

Example folder: ``examples/generic_timeseries_classification``
(`GitHub <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/generic_timeseries_classification>`_)

Overview
--------

* **Task**: ``generic_timeseries_classification`` (3 classes)
* **Dataset**: synthetic waveforms (sine, square, sawtooth), 1 variable
* **Model**: ``CLS_1k_NPU`` (about 1,000 parameters)
* **Default device**: F28P55 (NPU)
* **Training**: 20 epochs, batch size 256, early stopping enabled (patience 5)

Available Configurations
------------------------

One configuration file is provided per device family.

.. list-table::
   :header-rows: 1
   :widths: 25 15 20 40

   * - Config file
     - Target device
     - Model
     - Feature extraction preset
   * - ``config.yaml``
     - F28P55
     - ``CLS_1k_NPU``
     - ``Generic_1024Input_FFTBIN_64Feature_8Frame``
   * - ``config_MSPM0.yaml``
     - MSPM0G5187
     - ``CLS_13k_NPU``
     - ``FFT256Input_FE_RFFT_16Feature_8Frame_removeDC_Full_Bandwidth``
   * - ``config_CC1352.yaml``
     - CC1352
     - ``CLS_1k_NPU``
     - ``FFT256Input_FE_RFFT_16Feature_8Frame_removeDC_Full_Bandwidth``
   * - ``config_CC1354.yaml``
     - CC1354
     - ``CLS_1k_NPU``
     - ``FFT256Input_FE_RFFT_16Feature_8Frame_removeDC_Full_Bandwidth``
   * - ``config_CC2755.yaml``
     - CC2755
     - ``CLS_1k_NPU``
     - ``FFT256Input_FE_RFFT_16Feature_8Frame_removeDC_Full_Bandwidth``
   * - ``config_CC35X1.yaml``
     - CC35X1
     - ``CLS_1k_NPU``
     - ``FFT256Input_FE_RFFT_16Feature_8Frame_removeDC_Full_Bandwidth``

The non-C2000 configs use the dataset name ``hello_world_example_dsg`` with the input data path
``https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/generic_timeseries_classification.zip``.

Running the Example
-------------------

Install Tiny ML ModelZoo first (see :doc:`/installation/index` and :doc:`/getting_started/quickstart`),
then run from the ``tinyml-modelzoo`` folder:

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         ./run_tinyml_modelzoo.sh examples/generic_timeseries_classification/config.yaml

   .. tab:: Windows

      See :doc:`/getting_started/running_examples` for the ``.bat`` and ``.ps1`` launchers, for example:

      .. code-block:: bat

         run_tinyml_modelzoo.bat examples\generic_timeseries_classification\config.yaml

To target another device, pass its config file, for example
``examples/generic_timeseries_classification/config_MSPM0.yaml``.

Dataset Format
--------------

For classification tasks, Tiny ML ModelZoo expects the dataset in this structure:

.. code-block:: text

   {dataset_name}.zip/
   |
   |-- classes/
         |-- class_1/
         |     |-- file1.csv
         |     |-- file2.csv
         |
         |-- class_N/
               |-- ...

.. note::

   The folder names under ``classes/`` become the class labels. No annotation files are required.
   See :doc:`/byod/classification_format` for details.

Configuration (``config.yaml``)
-------------------------------

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

The ``Generic_1024Input_FFTBIN_64Feature_8Frame`` preset uses frames of 1024 samples, computes the
FFT, bins it into 64 features per frame and concatenates 8 frames (64 x 8 = 512 features). The config
file also lists, as comments, the other generic presets (``..._FFT_...``, ``..._RAW_...``) you can try.
For all keys see :doc:`/CONFIG_YAML_GUIDE`.

Output Location
---------------

Results are written to ``./data/projects/generic_timeseries_classification/run/<date-time>/CLS_1k_NPU/``
(relative to where you ran the script; configurable with ``common.projects_path``). The most
important artifacts are the trained model under ``training/``, the quantized ONNX model, and the
compiled artifacts under ``compilation/artifacts``. See :doc:`/getting_started/understanding_run_output`.

Expected Results
----------------

The example ``readme.md`` does not publish accuracy numbers. The waveforms are well separated, so
training converges quickly; check the printed test accuracy and confusion matrix at the end of your
own run.

Variations to Try
-----------------

* Raw time-domain input: ``feature_extraction_name: Generic_512Input_RAW_512Feature_1Frame``
* Other model sizes: ``CLS_100_NPU`` (smaller) or ``CLS_4k_NPU`` (larger)
* Another device: use one of the per-device configs above

Next Steps
----------

1. Try :doc:`arc_fault` for a real-world arc fault detection example.
2. Use your own data: :doc:`/byod/classification_format`.
3. Learn all configuration options: :doc:`/CONFIG_YAML_GUIDE`.
