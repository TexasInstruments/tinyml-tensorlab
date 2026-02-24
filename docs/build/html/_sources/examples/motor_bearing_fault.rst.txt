===========================
Motor Bearing Fault
===========================

Motor bearing fault classification detects and identifies different bearing
failure modes from vibration sensor data.

Overview
--------

* **Task**: Multi-class classification (6 fault types)
* **Application**: Predictive maintenance for motors
* **Dataset**: 3-axis vibration data from bearing experiments
* **Model**: MotorFault_model_1_t or CLS_4k_NPU

Fault Classes
-------------

The dataset includes 6 bearing conditions:

1. **Normal** - Healthy bearing operation
2. **Contaminated** - Foreign particles in lubricant
3. **Erosion** - Surface wear
4. **Flaking** - Material flaking from bearing surface
5. **No Lubrication** - Dry bearing operation
6. **Localized Fault** - Point defect on bearing

Running the Example
-------------------

.. tabs::

   .. tab:: Linux

      .. code-block:: bash

         cd tinyml-modelzoo
         ./run_tinyml_modelzoo.sh examples/motor_bearing_fault/config.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         run_tinyml_modelzoo.bat examples\motor_bearing_fault\config.yaml

Configuration
-------------

.. code-block:: yaml

   common:
     task_type: 'generic_timeseries_classification'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'motor_fault_classification_dsk'
     input_data_path: 'https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/motor_fault_classification_dsk.zip'

   data_processing_feature_extraction:
     feature_extraction_name: 'Input256_FFTBIN_16Feature_8Frame_3InputChannel_removeDC_2D1'
     variables: 3

   training:
     model_name: 'MotorFault_model_1_t'
     training_epochs: 20

Dataset Description
-------------------

* **Sensor**: 3-axis accelerometer mounted on motor bearing
* **Variables**: X, Y, Z acceleration (``variables: 3``)
* **Sampling**: Multiple sampling frequencies available

The vibration patterns differ between fault types due to:

* Different impact frequencies
* Varying severity levels
* Distinct frequency signatures

Feature Extraction Presets
--------------------------

Several presets are available for this dataset:

.. list-table::
   :header-rows: 1
   :widths: 50 25 25

   * - Preset
     - Features
     - Accuracy
   * - ``Input256_FFTBIN_16Feature_8Frame_3InputChannel_removeDC_1D``
     - 384
     - ~99.99%
   * - ``Input256_FFTBIN_16Feature_8Frame_3InputChannel_removeDC_2D1``
     - 384
     - ~100%
   * - ``Input256_FFT_128Feature_1Frame_3InputChannel_removeDC_2D1``
     - 384
     - ~98%
   * - ``Input128_RAW_128Feature_1Frame_3InputChannel_removeDC_2D1``
     - 384
     - ~92%

FFT-based binning performs best due to the frequency-domain nature of
bearing faults.

Available Models
----------------

.. list-table::
   :header-rows: 1
   :widths: 35 20 45

   * - Model
     - Parameters
     - Description
   * - ``MotorFault_model_1_t``
     - ~1,000
     - Baseline
   * - ``MotorFault_model_2_t``
     - ~2,000
     - Improved
   * - ``MotorFault_model_3_t``
     - ~4,000
     - Best accuracy
   * - ``CLS_4k_NPU``
     - ~4,000
     - Generic NPU model

Expected Results
----------------

With default settings:

.. code-block:: text

   Float32 Model:
   Accuracy: 99-100%
   F1-Score: ~1.0

   Quantized Model:
   Accuracy: 98-100%

Multi-Class Evaluation
----------------------

For 6-class classification, examine:

**Confusion Matrix**

Shows which fault types are confused with each other.
Common confusion pairs may indicate similar vibration signatures.

**Per-Class ROC Curves**

One-vs-Rest ROC shows how well each class separates from others.

.. figure:: /_static/img/examples/motor_bearing_fault/One_vs_Rest_MultiClass_ROC_test.png
   :width: 600px
   :align: center
   :alt: ROC Curves for Motor Bearing Fault

   One-vs-Rest Multi-class ROC curves for motor bearing fault detection showing excellent separation

**Class Score Distributions**

Histograms show classification confidence for each class.

.. figure:: /_static/img/examples/motor_bearing_fault/Histogram_Class_Score_differences_test.png
   :width: 600px
   :align: center
   :alt: Class Score Histogram

   Distribution of class score differences showing model confidence across different fault types

Dataset Quality Analysis
------------------------

Use the Goodness of Fit (GoF) test to visualize class separability:

.. code-block:: yaml

   data_processing_feature_extraction:
     gof_test: True
     frame_size: 256

   training:
     enable: True

This generates 8 plots showing cluster separation using different
transformation combinations.

Practical Considerations
------------------------

**Sensor Placement**

Vibration patterns depend heavily on sensor location.
Train with data from the same mounting position as deployment.

**Operating Conditions**

Include data from different:

* Motor speeds
* Load conditions
* Temperature ranges

**Fault Severity**

Early-stage faults have subtler signatures.
Include samples from different severity levels if available.

Anomaly Detection Alternative
-----------------------------

If you only have normal data, use anomaly detection instead:

.. code-block:: yaml

   common:
     task_type: 'generic_timeseries_anomalydetection'

   training:
     model_name: 'AD_4k_NPU'

This detects any deviation from normal without needing fault labels.

**Reconstruction Error Analysis:**

.. figure:: /_static/img/examples/motor_bearing_fault/reconstruction_error_log_scale.png
   :width: 600px
   :align: center
   :alt: Reconstruction Error Log Scale

   Reconstruction error distribution showing separation between normal and fault conditions

Next Steps
----------

* Try anomaly detection: :doc:`/task_types/anomaly_detection`
* Understand feature extraction: :doc:`/features/feature_extraction`
* Deploy to device: :doc:`/deployment/ccs_integration`
