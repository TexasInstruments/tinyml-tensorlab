==============================
Fan Blade Fault Classification
==============================

Detect faults in BLDC fans from 3-axis accelerometer data, from data collection to deployment on a
C2000 device. This is a full application example: it covers data capture with the Edge AI Studio
Model Composer, training with Tiny ML ModelZoo, and live preview on the F28P55x LaunchPad.

Example folder: ``examples/fan_blade_fault_classification``
(`GitHub <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/fan_blade_fault_classification>`_)

Overview
--------

* **Task**: classification of 4 conditions (``task_type: motor_fault``); an anomaly detection variant is also provided
* **Sensor**: ADXL355-PMDZ accelerometer (X, Y, Z), 3 variables, default 4 kHz sampling in the capture firmware
* **Dataset**: ``fan_blade_fault`` from ``https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/fan_blade_fault_dsi.zip``
* **Model**: ``CLS_4k_NPU`` for the main configs
* **Primary device**: F28P55

Demo Setup
----------

.. figure:: /_static/img/examples/fan_blade_fault/demo_setup.jpg
   :width: 600px
   :align: center
   :alt: Fan blade fault demo setup

   Hardware setup for the fan blade fault demo

Fault Types
-----------

.. list-table::
   :widths: 25 75

   * - .. figure:: /_static/img/examples/fan_blade_fault/normal.jpg
          :width: 200px
          :alt: Normal operation
     - **Normal**: fan running without faults

   * - .. figure:: /_static/img/examples/fan_blade_fault/blade_damage.jpg
          :width: 200px
          :alt: Blade damage
     - **Blade damage**: physical damage to the fan blades

   * - .. figure:: /_static/img/examples/fan_blade_fault/blade_imbalance.jpg
          :width: 200px
          :alt: Blade imbalance
     - **Blade imbalance**: uneven weight distribution

   * - .. figure:: /_static/img/examples/fan_blade_fault/blade_obstruction.jpg
          :width: 200px
          :alt: Blade obstruction
     - **Blade obstruction**: foreign object interfering with the blade

Available Configurations
------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 18 18 24

   * - Config file
     - Device
     - Model
     - Notes
   * - ``config.yaml``
     - F28P55
     - ``CLS_4k_NPU``
     - classification
   * - ``config_CC1312.yaml``, ``config_CC1314.yaml``, ``config_CC1352.yaml``, ``config_CC1354.yaml``, ``config_CC2755.yaml``, ``config_CC35X1.yaml``
     - CC1312, CC1314, CC1352, CC1354, CC2755, CC35X1
     - ``CLS_4k_NPU``
     - classification, same dataset and preset
   * - ``config_MSPM0.yaml``
     - MSPM0G5187
     - ``CLS_1k_NPU``
     - dataset ``fan_blower_imbalance_dataset``, preset ``MotorFault_256Input_FE_RFFT_16Feature_8Frame_3InputChannel_removeDC_2D1``, QAT with 4-bit weights
   * - ``config_anomaly_detection.yaml``
     - F28P55
     - ``AD_17k``
     - autoencoder, dataset ``fan_blade_ad_dataset.zip``, 200 epochs
   * - ``fan_blade_anomaly_detection_ondevice_training.yaml``
     - F29H85
     - ``Ondevice_Trainable_AD_Linear``
     - anomaly detection with on-device training; see :doc:`/ondevice_training/overview`

Running the Example
-------------------

Install Tiny ML ModelZoo first (:doc:`/installation/index`, :doc:`/getting_started/quickstart`), then run from the
``tinyml-modelzoo`` folder:

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         ./run_tinyml_modelzoo.sh examples/fan_blade_fault_classification/config.yaml

   .. tab:: Windows

      See :doc:`/getting_started/running_examples` for all launchers, for example:

      .. code-block:: bat

         run_tinyml_modelzoo.bat examples\fan_blade_fault_classification\config.yaml

Configuration (``config.yaml``)
-------------------------------

.. code-block:: yaml

   common:
     task_type: motor_fault
     target_device: F28P55
   dataset:
     dataset_name: fan_blade_fault
     input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/datasets/fan_blade_fault_dsi.zip
   data_processing_feature_extraction:
     data_proc_transforms: []
     feature_extraction_name: Input256_FFTBIN_16Feature_8Frame_3InputChannel_removeDC_2D1
     variables: 3
   training:
     model_name: CLS_4k_NPU
     batch_size: 256
     training_epochs: 20
   testing: {}
   compilation: {}

Collecting Your Own Data
------------------------

The example ``readme.md`` describes the full data capture flow:

#. Connect the ADXL355-PMDZ accelerometer to the F28P55 LaunchPad over SPI (chip select, MOSI, MISO, SCLK,
   data ready, ground and 3.3 V). The pin table and pinout pictures are in the example folder's ``readme.md``.
#. Load ``eAI_data_acq_dap_f28p55x.zip`` (in ``examples/fan_blade_fault_classification/assets``) from CCS onto the LaunchPad.
#. In Edge AI Studio Model Composer, select the serial port, choose *Capture & Display Live Data*, sensor
   *Vibration*, and capture each condition.
#. Train with Model Composer or with the config above.

See also :doc:`/model_composer/index` and :doc:`/byod/classification_format`.

Results
-------

The example ``readme.md`` states that ``config.yaml`` should give 100% accuracy. In the automated 1.5.0 config
test run, every classification config reached a test accuracy of 100.00% (AUC ROC 1.000): F28P55, CC1312,
CC1314, CC1352, CC1354, CC2755, CC35X1 (``CLS_4k_NPU``, 3,842 parameters) and MSPM0G5187
(``CLS_1k_NPU``, 994 parameters). The figures below are illustrative examples from the 1.4.0 version of this page.

.. figure:: /_static/img/examples/fan_blade_fault/One_vs_Rest_MultiClass_ROC_test.png
   :width: 600px
   :align: center
   :alt: ROC curve

   One-vs-rest ROC curves

.. figure:: /_static/img/examples/fan_blade_fault/Histogram_Class_Score_differences_test.png
   :width: 600px
   :align: center
   :alt: Class score histogram

   Class score differences between class pairs

.. figure:: /_static/img/examples/fan_blade_fault/pca_on_feature_extracted_train_data.png
   :width: 600px
   :align: center
   :alt: PCA of extracted features

   PCA of the extracted training features: clear cluster separation shows the classes are separable with this preset

Deploying and Live Preview
--------------------------

The compiled model is written to ``.../compilation/artifacts`` in the run folder, and feature extraction
golden vectors to ``.../training/quantization/golden_vectors``. Integrate them into a CCS project as
described in :doc:`/DEPLOYING_TO_SDK_EXAMPLE`. A single-sensor live preview project
(``eAI_mfd_eval_f28p55x.zip``) is included in the example's ``assets`` folder.

Anomaly Detection Variant
-------------------------

``config_anomaly_detection.yaml`` trains an ``AD_17k`` autoencoder on normal data only
(100 normal files); the 187 anomaly files (imbalance, damage, obstruction) are used only for testing. The
example's ``readme_anomaly_detection.md`` reports, at threshold ``mean + 4.0 * std`` (28.18): false positive
rate 0.15%, recall 100%, precision 99.98% and F1 99.99%. In the 1.5.0 test run this config produced an
F1 score of 99.99%, and the on-device training config an F1 score of 99.79%.

.. figure:: /_static/img/examples/fan_blade_fault/reconstruction_error_log_scale.png
   :width: 600px
   :align: center
   :alt: Reconstruction error

   Reconstruction error for normal vs. fault data (illustrative, from 1.4.0)

See Also
--------

* :doc:`blower_imbalance`: motor current based imbalance detection
* :doc:`motor_bearing_fault`
