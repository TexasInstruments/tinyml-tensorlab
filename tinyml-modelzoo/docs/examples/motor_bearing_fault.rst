===========================
Motor Bearing Fault
===========================

Motor bearing fault detection identifies bearing failure modes from 3-axis vibration data. The example
folder provides both a supervised classification example and an anomaly detection example (train on healthy
data only).

Example folder: ``examples/motor_bearing_fault``
(`GitHub <https://github.com/TexasInstruments/tinyml-modelzoo/tree/main/examples/motor_bearing_fault>`_)

Overview
--------

* **Task**: multi-class classification (``task_type: motor_fault``) and autoencoder anomaly detection
* **Application**: predictive maintenance for motors
* **Dataset**: 3-axis accelerometer vibration data from bearing experiments (``variables: 3``)
* **Models**: ``CLS_1k_NPU`` (default), ``MotorFault_model_1_t`` to ``MotorFault_model_3_t``; ``AD_Linear`` / ``AD_500_NPU`` for anomaly detection

Fault Classes
-------------

The classification dataset contains normal operation plus five fault types: contaminated lubricant,
erosion, flaking, no lubrication and localized defect (6 conditions in total).

Available Configurations
------------------------

.. list-table::
   :header-rows: 1
   :widths: 32 14 18 36

   * - Config file
     - Device
     - Model
     - Dataset / feature extraction
   * - ``config.yaml``
     - F28P55
     - ``CLS_1k_NPU``
     - ``motor_fault_example_dsk``; ``Input256_FFTBIN_16Feature_8Frame_3InputChannel_removeDC_2D1``
   * - ``config_MSPM0.yaml``
     - MSPM0G5187
     - ``CLS_1k_NPU``
     - ``motor_fault_example_dsk``; ``MotorFault_128Input_RAW_128Feature_1Frame_3InputChannel_removeDC_2D1`` (QAT, 4-bit weights, 8-bit activations)
   * - ``config_anomaly_detection.yaml``
     - F28P55
     - ``AD_Linear``
     - ``motor_fault_example_dsk_ad``; ``Input256_FFTBIN_16Feature_8Frame_3InputChannel_removeDC_2D1``
   * - ``config_anomaly_detection_MSPM0.yaml``
     - MSPM0G5187
     - ``AD_500_NPU``
     - ``motor_fault_example_dsk_ad``; ``MotorFault_128Input_RAW_128Feature_1Frame_3InputChannel_removeDC_2D1`` (QAT, 2-bit weights)

Running the Example
-------------------

Install Tiny ML ModelZoo first (:doc:`/installation/index`, :doc:`/getting_started/quickstart`), then run from the
``tinyml-modelzoo`` folder:

.. tabs::

   .. tab:: Linux / macOS

      .. code-block:: bash

         # F28P55
         ./run_tinyml_modelzoo.sh examples/motor_bearing_fault/config.yaml
         # MSPM0G5187
         ./run_tinyml_modelzoo.sh examples/motor_bearing_fault/config_MSPM0.yaml

   .. tab:: Windows

      See :doc:`/getting_started/running_examples` for all launchers, for example:

      .. code-block:: bat

         run_tinyml_modelzoo.bat examples\motor_bearing_fault\config.yaml
         run_tinyml_modelzoo.bat examples\motor_bearing_fault\config_MSPM0.yaml

Configuration (``config.yaml``)
-------------------------------

.. code-block:: yaml

   common:
     task_type: motor_fault
     target_device: F28P55
   dataset:
     dataset_name: motor_fault_example_dsk
   data_processing_feature_extraction:
     feature_extraction_name: Input256_FFTBIN_16Feature_8Frame_3InputChannel_removeDC_2D1
     variables: 3
   training:
     model_name: CLS_1k_NPU
     batch_size: 256
     training_epochs: 20
   testing: {}
   compilation: {}

Feature extraction takes 256-sample frames from each of the 3 axes, computes the FFT, removes DC, bins to 16
features per frame and concatenates 8 frames (384 features, laid out as 2D input per channel). Frequency
domain binning is well suited to bearing faults because the fault types have distinct frequency signatures.
The MSPM0 config instead uses raw 128-sample frames and downsamples the 313 kHz capture
(``sampling_rate: 313000`` to ``new_sr: 3130``).

Models
------

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Model
     - Approx. parameters
     - Description
   * - ``MotorFault_model_1_t``
     - 1,000
     - Baseline motor fault model
   * - ``MotorFault_model_2_t``
     - 2,000
     - Improved
   * - ``MotorFault_model_3_t``
     - 4,000
     - Highest accuracy of the three
   * - ``CLS_1k_NPU`` / ``CLS_2k_NPU`` / ``CLS_4k_NPU``
     - 1,000 / 2,000 / 4,000
     - Generic NPU models (``CLS_1k_NPU`` is the default in the configs)

Results from the 1.5.0 Test Run
-------------------------------

From the automated 1.5.0 config test logs (default settings):

.. list-table::
   :header-rows: 1
   :widths: 45 25 30

   * - Config
     - Test accuracy
     - AUC ROC
   * - ``config.yaml`` (F28P55, ``CLS_1k_NPU``)
     - 95.35%
     - 0.987
   * - ``config_MSPM0.yaml`` (MSPM0G5187, ``CLS_1k_NPU``, 4-bit weights)
     - 72.26%
     - 0.957
   * - ``config_anomaly_detection.yaml`` (``AD_Linear``)
     - F1 score 99.95%
     - not applicable

The lower accuracy of the MSPM0 config reflects the 4-bit weight quantization and raw-input features used by
that config; adjust the quantization bit widths, epochs or model size if you need more accuracy. The
``config_anomaly_detection_MSPM0.yaml`` config failed in that test run, so no result is reported for it.

Multi-Class Evaluation
----------------------

For the 6-class problem examine the confusion matrix (which faults are confused) and the one-vs-rest ROC
curves. The figures below come from the 1.4.0 version of this page and are illustrative only.

.. figure:: /_static/img/examples/motor_bearing_fault/One_vs_Rest_MultiClass_ROC_test.png
   :width: 600px
   :align: center
   :alt: ROC curves for motor bearing fault

   One-vs-rest multi-class ROC curves

.. figure:: /_static/img/examples/motor_bearing_fault/Histogram_Class_Score_differences_test.png
   :width: 600px
   :align: center
   :alt: Class score histogram

   Distribution of class score differences

Dataset Quality Analysis
------------------------

Set ``gof_test: True`` (with ``frame_size: 256``) in ``data_processing_feature_extraction`` to run the
Goodness of Fit analysis, which produces plots that show class separability for different transform
combinations.

Anomaly Detection Variant
-------------------------

If you only have healthy data, use ``config_anomaly_detection.yaml``
(``task_type: generic_timeseries_anomalydetection``). The model is an autoencoder trained on normal vibration
only; faults are detected as a high reconstruction error. The dataset has 36 normal files and 180 anomaly
files (216 files, three test motors). Normal data is split 60/30/10 into train, validation and test.

The anomaly example ``readme_anomaly_detection.md`` reports, at threshold ``mean + 4.5 * std`` (10.52):
false positive rate 0.00%, recall 100% (all 180 fault samples detected), precision 100% and F1 100%.

.. figure:: /_static/img/examples/motor_bearing_fault/reconstruction_error_log_scale.png
   :width: 600px
   :align: center
   :alt: Reconstruction error, log scale

   Reconstruction error: normal vs. fault conditions

Practical Considerations
------------------------

* Vibration patterns depend on sensor placement: train with the same mounting position as in deployment.
* Include different motor speeds, load conditions and temperatures.
* Early-stage faults have subtle signatures; include different severity levels if available.

Next Steps
----------

* Your own data: :doc:`/byod/classification_format`
* Deploy to a device: :doc:`/deployment/ccs_integration`, :doc:`/DEPLOYING_TO_SDK_EXAMPLE`
