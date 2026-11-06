===================================
Anomaly Detection Examples Overview
===================================

In 1.4.0 this page was a walkthrough of the DC arc fault anomaly detection example. In 1.5.0 the anomaly
detection configs are part of several example folders, so this page is an overview that lists all of them
and collects the common workflow: normal-only training, thresholds and how to read the outputs.

For a minimal first run use :doc:`generic_anomaly_detection`. For the theory and dataset format see
:doc:`/task_types/anomaly_detection` and :doc:`/byod/anomaly_detection_format`.

Anomaly Detection Configs in 1.5.0
----------------------------------

All of them use ``task_type: generic_timeseries_anomalydetection`` and are run from the ``tinyml-modelzoo``
folder with ``./run_tinyml_modelzoo.sh examples/<folder>/<config>.yaml`` (Windows: see
:doc:`/getting_started/running_examples`).

.. list-table::
   :header-rows: 1
   :widths: 38 14 18 30

   * - Config
     - Device
     - Model
     - Page
   * - ``generic_timeseries_anomalydetection/config.yaml``
     - F29H85
     - ``AD_17k``
     - :doc:`generic_anomaly_detection`
   * - ``generic_timeseries_anomalydetection/config_NPU.yaml``
     - F28P55
     - ``AD_11k_NPU``
     - :doc:`generic_anomaly_detection`
   * - ``dc_arc_fault/config_anomaly_detection_dsk.yaml``
     - F28P55
     - ``AD_17k``
     - :doc:`arc_fault`
   * - ``dc_arc_fault/config_anomaly_detection_dsi.yaml``
     - F28P55
     - ``AD_17k``
     - :doc:`arc_fault`
   * - ``motor_bearing_fault/config_anomaly_detection.yaml``
     - F28P55
     - ``AD_Linear``
     - :doc:`motor_bearing_fault`
   * - ``motor_bearing_fault/config_anomaly_detection_MSPM0.yaml``
     - MSPM0G5187
     - ``AD_500_NPU``
     - :doc:`motor_bearing_fault`
   * - ``fan_blade_fault_classification/config_anomaly_detection.yaml``
     - F28P55
     - ``AD_17k``
     - :doc:`fan_blade_fault_classification`
   * - ``ecg_classification/config_anomaly_detection.yaml``
     - F28P55
     - ``AD_Linear``
     - :doc:`ecg_classification`

In the automated 1.5.0 config test run the generic ``config.yaml``, the DC arc fault DSI config, the ECG,
fan blade and motor bearing (F28P55) configs completed. The DC arc fault DSK config stopped with a NaN/Inf
loss at epoch 0, the motor bearing MSPM0 config failed with a quantization scale/zero-point dimension error,
and the generic ``config_NPU.yaml`` failed with a CUDA device mismatch. Check the page of each example for
its status.

Workflow
--------

1. The autoencoder is trained on the ``Normal`` class only. The ``Anomaly`` class is used for testing and
   threshold evaluation.
2. After training, the mean and standard deviation of the reconstruction error on normal training data are
   computed and thresholds ``threshold = mean_train + k * std_train`` are generated for a range of ``k``.
3. At inference time a window with a reconstruction error above the threshold is an anomaly.

A typical anomaly detection config (the DC arc fault DSK variant keeps the same structure) has these
elements:

.. code-block:: yaml

   common:
     task_type: generic_timeseries_anomalydetection
     target_device: F28P55
   dataset:
     dataset_name: <name>
     input_data_path: <zip url or path>      # classes/Normal and classes/Anomaly
   data_processing_feature_extraction:
     data_proc_transforms: [...]             # or feature_extraction_name: <preset>
     variables: 1
   training:
     model_name: AD_17k
     quantization: 1
     output_int: false                       # must be false for anomaly detection
   testing:
     enable: True
   compilation:
     enable: True

Available Models
----------------

The anomaly detection models enabled in ``tinyml_modelzoo/model_descriptions/anomalydetection.py`` are:
``AD_500_NPU``, ``AD_1k``, ``AD_2k_NPU``, ``AD_4k``, ``AD_6k_NPU``, ``AD_8k_NPU``, ``AD_10k_NPU``,
``AD_11k_NPU``, ``AD_16k``, ``AD_17k``, ``AD_20k_NPU``, ``AD_Linear`` and
``Ondevice_Trainable_AD_Linear`` (used by the on-device learning example, see
:doc:`/ondevice_training/readme`). Models with the ``_NPU`` suffix are designed for devices with an NPU.
Start with a small model for simple, repetitive patterns and move to a larger one for complex signals.

Threshold Selection
-------------------

* Lower ``k`` detects more anomalies but raises more false alarms.
* Higher ``k`` reduces false alarms but may miss subtle anomalies.
* ``threshold_performance.csv`` in ``post_training_analysis/`` lists threshold, accuracy, precision, recall,
  F1 score and false positive rate for each ``k``. Pick the row that balances them for your application.
* ``reconstruction_error.png`` and ``reconstruction_error_log_scale.png`` in the same folder show the
  reconstruction error distributions of normal and anomaly data. Well separated distributions mean good
  detection capability; overlapping ones call for different features or a different model.

Anomaly Detection versus Classification
---------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Aspect
     - Classification
     - Anomaly detection
   * - Training data
     - All classes needed
     - Normal data only
   * - Unknown faults
     - Cannot be detected
     - Detected as anomalies
   * - Fault identification
     - Identifies the fault type
     - Only flags abnormality

Troubleshooting
---------------

* **High false positive rate**: increase ``k``, or add more diverse normal training data.
* **Missed anomalies**: decrease ``k``, increase model size, or use features that expose the anomaly.
* **Unstable reconstruction error**: train longer, check preprocessing, and compare with the float model.

See Also
--------

* :doc:`generic_anomaly_detection`
* :doc:`/features/feature_extraction`
* :doc:`/deploying_anomaly_detection_models_from_modelzoo_to_device/readme`
