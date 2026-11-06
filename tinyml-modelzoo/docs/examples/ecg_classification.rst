==================
ECG Classification
==================

Edge AI solution for real-time ECG signal classification on the MSPM0G5187 with NPU. It classifies
ECG signals into cardiac conditions for portable, low-power heart monitoring.

Example folder: ``examples/ecg_classification``
(`GitHub <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/ecg_classification>`_)

Overview
--------

Cardiovascular diseases are the leading cause of death globally (about 17.9 million deaths per year).
Continuous ECG monitoring helps early detection, but traditional analysis needs expensive equipment and
trained professionals. This example shows how a CNN running on the MSPM0G5187 NPU can classify ECG signals
at the point of care without cloud connectivity.

* **Application**: wearable health monitors, portable cardiac monitoring
* **Primary device**: MSPM0G5187 (with NPU); also AM13E2 and F28P55 (anomaly detection)

Targets of the reference solution: real-time on-device classification, low power consumption, more than 95%
classification accuracy, and LED feedback of the result.

System Components
-----------------

* Hardware: `MSPM0G5187 <https://www.ti.com/product/MSPM0G5187>`_ and the
  `AFE1594 analog front end <https://www.ti.com/drr/opn/AFE159RP4-DESIGN>`_ for ECG acquisition
* Software: Code Composer Studio 12.x or later, `MSPM0 SDK <https://www.ti.com/tool/MSPM0-SDK>`_ 2.08.00 or later

.. _ecg-available-configs:

Available Configurations
------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 15 20 35

   * - Config file
     - Device
     - Model
     - Mode and dataset
   * - ``config_MSPM0.yaml``
     - MSPM0G5187
     - ``ECG_55k_NPU``
     - ``task_type: ecg_classification``, dataset ``ecg_classification_4class``, preset ``ECG2500Input_Roundoff_1Frame``
   * - ``config.yaml``
     - AM13E2
     - ``CLS_4k_NPU``
     - ``generic_timeseries_classification``, dataset ``ecg`` (``ecg_classification_2class.zip``), 128-sample frames, 50 epochs
   * - ``config_anomaly_detection.yaml``
     - F28P55
     - ``AD_Linear``
     - ``generic_timeseries_anomalydetection``, dataset ``ecg_ad`` (``ecg_ad_dataset.zip``), 140-sample frames, 200 epochs

Running the Example
-------------------

Install Tiny ML ModelZoo first (:doc:`/installation/index`, :doc:`/getting_started/quickstart`), then run from the
``tinyml-modelzoo`` folder:

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         # MSPM0G5187 (primary target)
         ./run_tinyml_modelzoo.sh examples/ecg_classification/config_MSPM0.yaml
         # AM13E2
         ./run_tinyml_modelzoo.sh examples/ecg_classification/config.yaml
         # Anomaly detection
         ./run_tinyml_modelzoo.sh examples/ecg_classification/config_anomaly_detection.yaml

   .. tab:: Windows

      See :doc:`/getting_started/running_examples` for all launchers, for example:

      .. code-block:: bat

         run_tinyml_modelzoo.bat examples\ecg_classification\config_MSPM0.yaml

Dataset (MSPM0 config)
----------------------

``ecg_classification_4class`` contains labeled ECG recordings. The example ``readme.md`` lists these
categories:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Class
     - Description
   * - Normal
     - Healthy sinus rhythm
   * - Mild
     - Minor cardiac abnormalities
   * - Other
     - Other cardiac conditions requiring attention

Feature Extraction
------------------

The preset ``ECG2500Input_Roundoff_1Frame`` takes 2500 ECG samples per frame, normalizes the signal by
rounding off, and feeds a single frame to the model.

Model and Training (MSPM0 config)
---------------------------------

``ECG_55k_NPU`` is a CNN with about 55,000 parameters designed for the MSPM0G5187 NPU.

.. code-block:: yaml

   common:
     task_type: ecg_classification
     target_device: MSPM0G5187
   dataset:
     dataset_name: ecg_classification_4class
   data_processing_feature_extraction:
     feature_extraction_name: ECG2500Input_Roundoff_1Frame
   training:
     model_name: ECG_55k_NPU
     batch_size: 12
     learning_rate: 0.001
     weight_decay: 4e-5
     optimizer: adam
     training_epochs: 25

Running the config performs feature extraction, training, INT8 quantization and compilation with the
`TI Neural Network Compiler <https://software-dl.ti.com/mctools/nnc/mcu/users_guide/>`_, producing the model
library and interface headers for the MSPM0 SDK (see :doc:`/DEPLOYING_TO_SDK_EXAMPLE`).

Results
-------

The example ``readme.md`` states an accuracy of about 97% for the MSPM0 configuration. In the automated 1.5.0 config
test run:

.. list-table::
   :header-rows: 1
   :widths: 50 25 25

   * - Config
     - Test accuracy
     - AUC ROC
   * - ``config_MSPM0.yaml`` (``ECG_55k_NPU``)
     - 97.19%
     - 0.995
   * - ``config.yaml`` (AM13E2, ``CLS_4k_NPU``)
     - 97.21%
     - 0.995
   * - ``config_anomaly_detection.yaml`` (``AD_Linear``)
     - F1 score 98.02%
     - not applicable

Anomaly Detection Mode
----------------------

``config_anomaly_detection.yaml`` trains an autoencoder on normal ECG recordings only and flags
anomalies by reconstruction error. This is useful when labeled abnormal data is not available. It
compiles with ``compile_preset_name: forced_soft_npu_preset``.

References
----------

* `MSPM0G5187 product page <https://www.ti.com/product/MSPM0G5187>`_
* `AFE1594 design resources <https://www.ti.com/drr/opn/AFE159RP4-DESIGN>`_
* `MSPM0 SDK <https://www.ti.com/tool/MSPM0-SDK>`_
