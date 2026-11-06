===================================
NILM Appliance Usage Classification
===================================

Non-Intrusive Load Monitoring (NILM): identify which appliances are active from a single measurement point
(such as a smart meter), without a sensor on every appliance. This page covers both NILM examples in
the ModelZoo.

Example folders:

* ``examples/nilm_appliance_usage_classification`` (`GitHub <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/nilm_appliance_usage_classification>`__)
* ``examples/PLAID_nilm_classification`` (`GitHub <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/PLAID_nilm_classification>`__)

Overview
--------

NILM analyzes time series of electrical measurements to detect which appliances are running. Typical uses
are energy cost optimization in factories, power distribution tracking in electric vehicles, and appliance
usage analysis in households.

Both examples are time series classification (``generic_timeseries_classification``) on the F28P55 with the
``CLS_13k_NPU`` model, the same feature extraction preset ``Generic_32Input_FFTBIN_4Feature_8Frame`` and INT8
quantization (``quantization: 2``).

.. list-table::
   :header-rows: 1
   :widths: 28 36 36

   * -
     - ``nilm_appliance_usage_classification``
     - ``PLAID_nilm_classification``
   * - Dataset name
     - ``esda_nilm_2021_5_variable_4_class_dataset``
     - ``plaid_submetered``
   * - Dataset URL
     - ``.../mcu_ai/datasets/esda_nilm_2021_5_variable_4_class_dataset.zip``
     - ``.../mcu_ai/datasets/plaid_nilm_submetered_dataset.zip``
   * - Variables
     - 5 (active power, voltage, current, reactive power, phase)
     - 2 (voltage and current at 30 kHz)
   * - Classes
     - 4 appliance combinations
     - 11 appliance types
   * - Training
     - 15 epochs, batch size 256
     - 30 epochs, batch size 256, learning rate 0.04

Running the Examples
--------------------

Install Tiny ML ModelZoo first (:doc:`/installation/index`, :doc:`/getting_started/quickstart`), then run from the
``tinyml-modelzoo`` folder:

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         ./run_tinyml_modelzoo.sh examples/nilm_appliance_usage_classification/config.yaml
         ./run_tinyml_modelzoo.sh examples/PLAID_nilm_classification/config.yaml

   .. tab:: Windows

      See :doc:`/getting_started/running_examples` for all launchers, for example:

      .. code-block:: bat

         run_tinyml_modelzoo.bat examples\nilm_appliance_usage_classification\config.yaml
         run_tinyml_modelzoo.bat examples\PLAID_nilm_classification\config.yaml

Appliance Usage Classification (ESDA NILM 2021)
-----------------------------------------------

The data comes from the Kaggle `ESDA NILM Competition 2021 <https://www.kaggle.com/competitions/esda-nilm-2021>`_
(``vw_train.csv``). The original file has 28 classes of appliance combinations and 30 columns. Because many
classes have too little data, the example keeps 4 classes and 5 easily measured variables:

* Classes: ``+fridge``, ``+fridge+tumble_dryer+washer_dryer``, ``+fridge+washer_dryer``,
  ``+fridge+washing_machine+washer_dryer``
* Variables: active power, voltage, current, reactive power, phase

The prepared zip is downloaded automatically from the URL in the config.

.. code-block:: yaml

   common:
     task_type: generic_timeseries_classification
     target_device: F28P55
   dataset:
     dataset_name: esda_nilm_2021_5_variable_4_class_dataset
     input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/esda_nilm_2021_5_variable_4_class_dataset.zip
   data_processing_feature_extraction:
     feature_extraction_name: Generic_32Input_FFTBIN_4Feature_8Frame
     variables: 5
   training:
     model_name: CLS_13k_NPU
     batch_size: 256
     training_epochs: 15
     quantization: 2

The preset windows the data into 32-sample frames, applies FFT, bins to 4 features per frame, converts to
log scale and concatenates 8 frames. Compilation uses the default NPU preset with full INT8 quantization.

PLAID Appliance Identification
------------------------------

The PLAID dataset contains voltage and current measurements sampled at 30 kHz from 11 appliance types: air
conditioner, compact fluorescent lamp, fan, fridge, hairdryer, heater, incandescent light bulb, laptop,
microwave, vacuum and washing machine. The example uses a subset (``plaid_nilm_submetered_dataset.zip``).

.. code-block:: yaml

   common:
     task_type: generic_timeseries_classification
     target_device: F28P55
   dataset:
     dataset_name: plaid_submetered
     input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/plaid_nilm_submetered_dataset.zip
   data_processing_feature_extraction:
     feature_extraction_name: Generic_32Input_FFTBIN_4Feature_8Frame
     variables: 2
   training:
     model_name: CLS_13k_NPU
     batch_size: 256
     training_epochs: 30
     early_stopping: true
     early_stopping_patience: 5
     num_gpus: 0
     quantization: 2
     learning_rate: 0.04
   testing: {}
   compilation:
     keep_libc_files: true

Results
-------

Test-set results documented in the example readmes (measured with an earlier release) and in the automated 1.5.0
config test run:

.. list-table::
   :header-rows: 1
   :widths: 34 22 22 22

   * - Example
     - Example readme
     - 1.5.0 test run
     - AUC ROC (1.5.0 run)
   * - ``nilm_appliance_usage_classification``
     - 95.95%
     - 96.85%
     - 0.998
   * - ``PLAID_nilm_classification``
     - 97.72%
     - 97.82%
     - 0.989

On-Device Results and Performance
---------------------------------

Both readmes describe running the compiled model on a LAUNCHXL-F28P55X (C2000Ware 6.00, CCS 20.3.0) and checking that
the device output matches the golden vectors. The deployment steps are in
:doc:`/deploying_classification_models_from_modelmaker_to_device/readme` and :doc:`/DEPLOYING_TO_SDK_EXAMPLE`.

.. figure:: /_static/img/examples/nilm/ondevice_results_appliance.png
   :width: 550px
   :align: center
   :alt: On-device result for appliance usage classification

   Device output matching the golden vectors (appliance usage classification)

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Metric (F28P55x, NPU, from the example readmes)
     - Appliance usage
     - PLAID
   * - AI model cycles
     - 165,629
     - 190,571
   * - Inference time
     - 1104.19 us
     - 1270.47 us
   * - Model flash
     - 11,967 bytes
     - 13,032 bytes
   * - Model SRAM
     - 1,220 bytes
     - 1,004 bytes

See Also
--------

* :doc:`/byod/classification_format`: use your own classification data
* :doc:`/NPU_CONFIGURATION_GUIDELINES`
