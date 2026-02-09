========================================
Generic Time Series Anomaly Detection
========================================

This example serves as a **"Hello World" introduction** to time series anomaly
detection using the TinyML ModelZoo toolchain. It demonstrates how to use
autoencoder-based anomaly detection with our toolchain.

Overview
--------

* **Task**: Time series anomaly detection (autoencoder-based)
* **Dataset**: Synthetic sinusoidal signals with frequency/amplitude anomalies
* **Model**: AD_17k (~17,000 parameters)
* **Target**: F28P55 (NPU device)

Running the Example
-------------------

.. tabs::

   .. tab:: Linux

      .. code-block:: bash

         cd tinyml-modelzoo
         ./run_tinyml_modelzoo.sh examples/generic_timeseries_anomalydetection/config.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         run_tinyml_modelzoo.bat examples\generic_timeseries_anomalydetection\config.yaml

Understanding the Dataset
-------------------------

The dataset uses a synthetic sinusoidal pattern:

**Normal Pattern:**

* Signal: y = 1.2 sin(2pi*f*t) + 0.8 cos(2pi*f*t)
* Base frequency: f = 1.0 Hz
* Amplitude variation: +/- 10%
* Frequency variation: +/- 5%

**Anomaly Types:**

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Type
     - Description
   * - Frequency Faster
     - Signal frequency increases by 30-70%
   * - Frequency Slower
     - Signal frequency decreases by 20-50%
   * - Amplitude Higher
     - Signal amplitude increases by 60-120%
   * - Amplitude Lower
     - Signal amplitude decreases by 40-70%

**Dataset split:**

* Normal samples: 60 files (train: 50%, val: 10%, test: 40%)
* Anomaly samples: 32 files (all in test set)

Each file contains 5000 samples (50 seconds at 100 Hz).

Dataset Format
--------------

For anomaly detection, ModelZoo expects this folder structure:

.. code-block:: text

   dataset_name/
   |
   |-- classes/
         |-- Normal/
         |     |-- normal_0000.csv
         |     |-- normal_0001.csv
         |     |-- ...
         |
         |-- Anomaly/
               |-- freq_faster_0000.csv
               |-- freq_slower_0000.csv
               |-- amp_higher_0000.csv
               |-- amp_lower_0000.csv
               |-- ...

.. important::

   The autoencoder is trained **only on Normal data**. All Anomaly files are
   used exclusively for testing.

Configuration
-------------

**common section:**

.. code-block:: yaml

   common:
     target_module: 'timeseries'
     task_type: 'generic_timeseries_anomalydetection'
     target_device: 'F28P55'

**dataset section:**

.. code-block:: yaml

   dataset:
     enable: True
     dataset_name: generic_timeseries_anomalydetection
     input_data_path: https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/generic_timeseries_anomalydetection.zip
     split_factor: [0.5, 0.1, 0.4]

**data_processing_feature_extraction section:**

.. code-block:: yaml

   data_processing_feature_extraction:
     data_proc_transforms:
       - SimpleWindow
       - Downsample
     frame_size: 100
     sampling_rate: 100
     new_sr: 10
     variables: 1

**training section:**

.. code-block:: yaml

   training:
     enable: True
     model_name: 'AD_17k'
     batch_size: 64
     training_epochs: 50
     num_gpus: 1
     learning_rate: 0.001
     quantization: 1
     output_int: False

.. important::

   ``output_int`` must be ``False`` for anomaly detection since reconstruction
   error must be in float for threshold comparison.

How Autoencoder Detection Works
-------------------------------

**Training Phase (Normal Data Only):**

1. Autoencoder learns to compress and reconstruct normal patterns
2. For normal samples: Reconstruction error is LOW (~0.008)
3. Model learns: "This is what normal looks like"

**Testing Phase (Normal + Anomaly Data):**

.. code-block:: text

   Normal sample:
     Reconstruction Error: ~0.008 -> Correctly identified as normal

   Anomaly sample:
     Reconstruction Error: ~0.69 -> Flagged as anomaly (88x higher!)

**Detection Logic:**

.. code-block:: python

   if reconstruction_error > threshold:
       -> ANOMALY
   else:
       -> NORMAL

Why Frame Size Matters
----------------------

Anomaly detection requires **temporal context** - the model must see multiple
consecutive samples to understand the pattern.

.. list-table::
   :header-rows: 1
   :widths: 25 25 50

   * - Frame Size
     - Cycles Visible
     - Detection Quality
   * - 1 - 10
     - < 0.1 cycles
     - Insufficient
   * - 20 - 50
     - 0.2 - 0.5
     - Weak (partial cycle)
   * - 100
     - 1.0 cycle
     - Good (full cycle visible)
   * - >200
     - 2.0+ cycles
     - Excellent

**Rule of thumb:** Frame size should capture at least 1 complete cycle.

Expected Results
----------------

**Reconstruction Error Statistics:**

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Data Type
     - Mean
     - Std Dev
   * - Normal training data
     - 0.0078
     - 0.0022
   * - Normal test data
     - 0.0081
     - 0.0023
   * - Anomaly test data
     - 0.6898
     - 0.4616

**Performance at Threshold k=4.5:**

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Metric
     - Value
   * - Threshold
     - 0.0178
   * - Accuracy
     - 99.97%
   * - Precision
     - 99.95%
   * - Recall (Detection Rate)
     - 100.00%
   * - F1 Score
     - 99.97%
   * - False Positive Rate
     - 0.07%

**Confusion Matrix:**

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * -
     - Predicted Normal
     - Predicted Anomaly
   * - Ground Truth: Normal
     - 9,617 (TN)
     - 7 (FP)
   * - Ground Truth: Anomaly
     - 0 (FN)
     - 12,832 (TP)

The model successfully detects **all 4 anomaly types** with 100% recall!

Output Location
---------------

Results are saved to::

   ../tinyml-modelmaker/data/projects/{dataset_name}/run/{date-time}/{model_name}/

Key outputs:

* ``training/base/`` - Float model training results
* ``training/quantization/`` - Quantized model results
* ``training/quantization/post_training_analysis/`` - Threshold analysis
* ``training/quantization/post_training_analysis/threshold_performance.csv`` - Metrics table
* ``training/quantization/post_training_analysis/reconstruction_error_histogram.png`` - Error distribution
* ``compilation/artifacts/mod.a`` - Compiled for device

Next Steps
----------

After successfully running this generic example:

1. Try :doc:`anomaly_detection_example` - DC arc fault anomaly detection
2. Try :doc:`fan_blade_fault_classification` - Fan blade anomaly detection
3. Try :doc:`motor_bearing_fault` - Motor vibration anomaly detection
4. Read :doc:`/task_types/anomaly_detection` - Learn more about anomaly detection
