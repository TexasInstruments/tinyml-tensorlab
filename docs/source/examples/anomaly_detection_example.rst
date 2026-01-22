============================
Anomaly Detection Example
============================

This example demonstrates using autoencoder-based anomaly detection
to identify abnormal patterns in time series data.

Overview
--------

* **Task**: Anomaly detection (binary: normal vs anomaly)
* **Model**: Autoencoder architecture
* **Training**: Uses only normal data
* **Detection**: High reconstruction error indicates anomaly

When to Use Anomaly Detection
-----------------------------

Anomaly detection is ideal when:

* You only have "normal" data for training
* Fault conditions are rare or unknown
* You want to detect any deviation from normal
* Labeling fault data is difficult or expensive

This contrasts with classification, which requires labeled examples
of each fault type.

Running the Example
-------------------

.. tabs::

   .. tab:: Linux

      .. code-block:: bash

         cd tinyml-modelzoo
         ./run_tinyml_modelzoo.sh examples/anomaly_detection/config.yaml

   .. tab:: Windows

      .. code-block:: powershell

         cd tinyml-modelzoo
         run_tinyml_modelzoo.bat examples\\anomaly_detection\\config.yaml

Configuration
-------------

.. code-block:: yaml

   common:
     task_type: 'generic_timeseries_anomalydetection'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'your_anomaly_dataset'
     input_data_path: '/path/to/dataset'

   data_processing_feature_extraction:
     feature_extraction_name: 'Generic_1024Input_FFTBIN_64Feature_8Frame'
     variables: 1

   training:
     model_name: 'AD_4k_NPU'
     training_epochs: 50
     batch_size: 256

   testing:
     enable: True

   compilation:
     enable: True

How Anomaly Detection Works
---------------------------

**Autoencoder Architecture:**

.. figure:: /_static/img/anomaly_detection/autoencoder_architecture.png
   :width: 700px
   :align: center
   :alt: Autoencoder Architecture

   Autoencoder architecture for anomaly detection

**Training Phase:**

1. Train autoencoder on normal data only
2. Encoder compresses input to latent representation
3. Decoder reconstructs original input
4. Model learns to accurately reconstruct normal patterns

**Inference Phase:**

1. Input new data sample
2. Autoencoder attempts reconstruction
3. Calculate reconstruction error
4. High error = anomaly (pattern not seen during training)

.. code-block:: text

   Normal data:     Input → Encoder → Decoder → Good reconstruction (low error)
   Anomaly data:    Input → Encoder → Decoder → Poor reconstruction (high error)

Dataset Format
--------------

For anomaly detection, your dataset should contain:

* **Normal samples**: Many examples of normal operation
* **Anomaly samples** (optional): For evaluation only

Directory structure:

.. code-block:: text

   my_anomaly_dataset/
   ├── annotations.yaml
   └── classes/
       ├── normal/           # Training data (many samples)
       │   ├── sample_001.csv
       │   ├── sample_002.csv
       │   └── ...
       └── anomaly/          # Test data only (few samples OK)
           ├── anomaly_001.csv
           └── ...

The model trains only on "normal" class but evaluates against both.

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
   * - ``AD_1k_NPU``
     - ~1,000
     - Small, common choice
   * - ``AD_2k_NPU``
     - ~2,000
     - Balanced
   * - ``AD_4k_NPU``
     - ~4,000
     - Complex patterns
   * - ``AD_8k_NPU``
     - ~8,000
     - High complexity

**Model Selection:**

* Simple, repetitive patterns: ``AD_500_NPU`` or ``AD_1k_NPU``
* Moderate complexity: ``AD_2k_NPU`` or ``AD_4k_NPU``
* Complex, variable patterns: ``AD_8k_NPU`` or larger

Expected Results
----------------

After training, you should see:

.. code-block:: text

   Training complete.

   Reconstruction Error Statistics:
   Normal data mean error: 0.05
   Normal data std error: 0.02

   Anomaly Detection Results:
   AUC-ROC: 0.95+
   Best threshold: 0.15

   At threshold 0.15:
   True Positive Rate: 0.92
   False Positive Rate: 0.05

Threshold Selection
-------------------

The threshold determines the sensitivity:

**Lower threshold:**

* More anomalies detected (higher TPR)
* More false alarms (higher FPR)
* Use when missing anomalies is costly

**Higher threshold:**

* Fewer false alarms (lower FPR)
* May miss subtle anomalies (lower TPR)
* Use when false alarms are disruptive

.. code-block:: yaml

   # Configure threshold in deployment
   # Use fpr_tpr_thresholds.csv to select optimal point
   threshold: 0.15  # Adjust based on ROC analysis

Interpreting Outputs
--------------------

**Reconstruction Error Histogram:**

Shows distribution of errors for normal vs anomaly data:

* Separated distributions = good detection
* Overlapping distributions = may need different features or model

**ROC Curve:**

* AUC close to 1.0 = excellent separation
* AUC around 0.5 = no better than random
* Use to select operating threshold

**Latent Space Visualization:**

* Normal samples should cluster tightly
* Anomalies should be outside normal cluster

Advanced Configuration
----------------------

**Adjust Bottleneck Size:**

Smaller bottleneck = stronger compression, may miss subtle anomalies:

.. code-block:: yaml

   training:
     model_name: 'AD_2k_NPU'
     # Model architecture defines bottleneck size

**Feature Engineering:**

Better features improve detection:

.. code-block:: yaml

   data_processing_feature_extraction:
     # FFT captures frequency anomalies
     feature_extraction_name: 'Generic_1024Input_FFTBIN_64Feature_8Frame'

     # Raw captures waveform anomalies
     # feature_extraction_name: 'Generic_512Input_RAW_512Feature_1Frame'

**Data Augmentation:**

Add variety to normal data:

.. code-block:: yaml

   data_processing_feature_extraction:
     augmentation:
       noise: True
       scale: True

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

**Temperature Profile Monitoring:**

.. code-block:: yaml

   common:
     task_type: 'generic_timeseries_anomalydetection'

   data_processing_feature_extraction:
     feature_extraction_name: 'Generic_256Input_RAW_256Feature_1Frame'
     variables: 3  # Multiple temperature sensors

   training:
     model_name: 'AD_4k_NPU'

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

* Threshold too low
* Normal data not representative
* Need more normal training data

**Missing anomalies:**

* Threshold too high
* Model too simple (increase size)
* Feature extraction missing relevant patterns

**Unstable reconstruction error:**

* Increase training epochs
* Try different learning rate
* Check for data preprocessing issues

Next Steps
----------

* Compare with :doc:`/task_types/anomaly_detection` guide
* Learn about :doc:`/features/feature_extraction`
* Deploy to device: :doc:`/deployment/npu_device_deployment`
