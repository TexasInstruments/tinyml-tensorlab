=====================================
Arc Fault Anomaly Detection Example
=====================================

This example demonstrates using autoencoder-based anomaly detection
to identify DC arc fault patterns in current waveform data.

Overview
--------

* **Task**: Anomaly detection (binary: normal vs anomaly)
* **Application**: DC arc fault detection
* **Model**: Autoencoder architecture (``AD_2k_NPU``)
* **Training**: Uses only normal data
* **Detection**: High reconstruction error indicates anomaly
* **Dataset**: DC arc fault current waveforms (DSK variant)

This example uses the same DC arc fault dataset as the classification example,
but approaches it as an anomaly detection problem. The autoencoder learns to
reconstruct normal current waveforms and flags arc fault patterns as anomalies
based on high reconstruction error.

Running the Example
-------------------

.. tabs::

   .. tab:: Linux

      .. code-block:: bash

         cd tinyml-modelzoo

         # DSK dataset variant
         ./run_tinyml_modelzoo.sh examples/dc_arc_fault/config_anomaly_detection_dsk.yaml

         # DSI dataset variant
         ./run_tinyml_modelzoo.sh examples/dc_arc_fault/config_anomaly_detection_dsi.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo

         # DSK dataset variant
         run_tinyml_modelzoo.bat examples\dc_arc_fault\config_anomaly_detection_dsk.yaml

         # DSI dataset variant
         run_tinyml_modelzoo.bat examples\dc_arc_fault\config_anomaly_detection_dsi.yaml

Configuration
-------------

.. code-block:: yaml

   common:
     task_type: 'generic_timeseries_anomalydetection'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'dc_arc_fault_example_dsk'
     input_data_path: 'https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/dc_arc_fault_example_dsk.zip'

   data_processing_feature_extraction:
     feature_extraction_name: 'FFT1024Input_256Feature_1Frame_Full_Bandwidth'
     variables: 1

   training:
     model_name: 'AD_2k_NPU'
     training_epochs: 200
     batch_size: 256

   testing:
     enable: True

   compilation:
     enable: True

Dataset Format
--------------

The dataset follows the anomaly detection folder structure:

.. code-block:: text

   dc_arc_fault_example_dsk/
   └── classes/
       ├── Normal/           # Normal current waveforms (training data)
       │   ├── file1.csv
       │   └── ...
       └── Anomaly/          # Arc fault waveforms (test-only data)
           ├── file1.csv
           └── ...

The model trains only on "Normal" class data. Anomaly data is used exclusively
for testing and threshold evaluation.

See :doc:`/byod/anomaly_detection_format` for full dataset format details.

Available Models
----------------

Anomaly detection models use autoencoder architecture:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Model
     - Parameters
     - Description
   * - ``AD_500_NPU``
     - ~500
     - Minimal, simple patterns
   * - ``AD_1k``
     - ~1,000
     - Compact autoencoder
   * - ``AD_2k_NPU``
     - ~2,000
     - Balanced (used in this example)
   * - ``AD_4k``
     - ~4,000
     - Complex patterns
   * - ``AD_6k_NPU``
     - ~6,000
     - Depthwise separable encoder
   * - ``AD_8k_NPU``
     - ~8,000
     - High complexity
   * - ``AD_Linear``
     - Varies
     - Deep linear autoencoder
   * - ``Ondevice_Trainable_AD_Linear``
     - Varies
     - On-device trainable variant

**Model Selection:**

* Simple, repetitive patterns: ``AD_500_NPU`` or ``AD_1k``
* Moderate complexity: ``AD_2k_NPU`` or ``AD_4k``
* Complex, variable patterns: ``AD_6k_NPU`` or ``AD_8k_NPU``

Expected Results
----------------

After training, you should see output similar to:

.. code-block:: text

   INFO: Best epoch: 39
   INFO: MSE: 1.773

   INFO: Reconstruction Error Statistics:
   INFO: Normal training data - Mean: 1.662490, Std: 1.968127
   INFO: Anomaly test data - Mean: 141.985321, Std: 112.756683
   INFO: Normal test data - Mean: 2.849831, Std: 1.343052

   INFO: Threshold for K = 4.5: 10.519060
   INFO: False positive rate: 0.00%
   INFO: Anomaly detection rate (recall): 100.00%

**Key indicators of good training:**

* Large gap between normal mean error and anomaly mean error
* Low false positive rate
* High recall (anomaly detection rate)

Threshold Selection
-------------------

The threshold determines the sensitivity:

.. code-block:: text

   threshold = mean_train + k * std_train

* **Lower k**: More anomalies detected, but more false alarms
* **Higher k**: Fewer false alarms, but may miss subtle anomalies
* **Typical starting point**: k=3 (covers ~99.7% of normal data)

Refer to the ``threshold_performance.csv`` output file to select the optimal
k value for your application. It contains precision, recall, F1, and false
positive rate for each k value from 0 to 4.5.

Interpreting Outputs
--------------------

After training, ModelMaker generates the following analysis outputs in the
``post_training_analysis/`` folder:

**Reconstruction Error Histogram** (``reconstruction_error_histogram.png``):

Shows distribution of reconstruction errors for normal vs anomaly data:

* Separated distributions = good detection capability
* Overlapping distributions = may need different features or model

**Threshold Performance CSV** (``threshold_performance.csv``):

Contains detection metrics for each k value:

.. code-block:: text

   k_value,threshold,accuracy,precision,recall,f1_score,false_positive_rate,...
   0.0,1.662,98.65,98.65,100.0,99.32,83.54,...
   1.0,3.631,99.71,99.70,100.0,99.85,18.13,...
   ...
   4.5,10.519,100.0,100.0,100.0,100.0,0.0,...

Use this file to select the threshold that best balances precision and recall
for your deployment requirements.

Advanced Configuration
----------------------

**Adjust Model Size:**

Smaller models compress more aggressively, which may miss subtle anomalies:

.. code-block:: yaml

   training:
     model_name: 'AD_500_NPU'   # Smaller, simpler patterns
     # model_name: 'AD_8k_NPU'  # Larger, complex patterns

**Feature Engineering:**

Better features improve detection:

.. code-block:: yaml

   data_processing_feature_extraction:
     # FFT captures frequency anomalies
     feature_extraction_name: 'Generic_1024Input_FFTBIN_64Feature_8Frame'

     # Raw captures waveform anomalies
     # feature_extraction_name: 'Generic_512Input_RAW_512Feature_1Frame'

Practical Applications
----------------------

**Vibration Monitoring:**

.. code-block:: yaml

   common:
     task_type: 'generic_timeseries_anomalydetection'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'vibration_normal_only_dsk'

   training:
     model_name: 'AD_4k_NPU'

**Current Waveform Monitoring:**

.. code-block:: yaml

   common:
     task_type: 'generic_timeseries_anomalydetection'

   data_processing_feature_extraction:
     feature_extraction_name: 'FFT1024Input_256Feature_1Frame_Full_Bandwidth'
     variables: 1

   training:
     model_name: 'AD_2k_NPU'

Comparison with Classification
------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Aspect
     - Classification
     - Anomaly Detection
   * - Training data
     - All classes needed
     - Only normal data
   * - Unknown faults
     - Cannot detect
     - Can detect (as anomaly)
   * - Fault identification
     - Identifies fault type
     - Only detects abnormality
   * - When to use
     - Known, labeled faults
     - Unknown or unlabeled faults

Troubleshooting
---------------

**High false positive rate:**

* Threshold too low -- increase k value
* Normal data not representative of all operating conditions
* Need more diverse normal training data

**Missing anomalies:**

* Threshold too high -- decrease k value
* Model too simple -- increase size
* Feature extraction missing relevant patterns

**Unstable reconstruction error:**

* Increase training epochs
* Try different learning rate
* Check for data preprocessing issues

Next Steps
----------

* Review the :doc:`/task_types/anomaly_detection` guide for threshold theory
* Learn about :doc:`/features/feature_extraction`
* Deploy to device: :doc:`/deployment/npu_device_deployment`
