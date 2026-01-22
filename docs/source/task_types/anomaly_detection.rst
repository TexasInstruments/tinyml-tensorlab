=================
Anomaly Detection
=================

Anomaly detection identifies patterns that deviate from "normal" behavior using
autoencoder-based models trained only on normal data.

Overview
--------

**What it does**: Learns what "normal" looks like, then flags anything different.

**Key advantage**: Only needs normal data for training - no need to collect examples
of all possible faults.

**Use cases**:

* Equipment health monitoring
* Predictive maintenance
* Fault detection when fault examples are scarce
* Detecting novel/unknown failures

How It Works
------------

1. **Training**: Autoencoder learns to compress and reconstruct normal data
2. **Inference**: Input is compressed and reconstructed
3. **Decision**: High reconstruction error → Anomaly

.. code-block:: text

   Normal input → Low reconstruction error → "Normal"
   Anomaly input → High reconstruction error → "Anomaly"

Configuration
-------------

.. code-block:: yaml

   common:
     task_type: 'generic_timeseries_anomalydetection'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'my_anomaly_data'
     input_data_path: '/path/to/data'

   data_processing_feature_extraction:
     feature_extraction_name: 'Generic_256Input_RAW_256Feature_1Frame'
     variables: 3

   training:
     model_name: 'AD_4k_NPU'
     training_epochs: 50

   testing: {}
   compilation: {}

Dataset Format
--------------

Same as classification but with specific requirements:

.. code-block:: text

   my_dataset/
   └── classes/
       ├── normal/              # Training data
       │   ├── file1.csv
       │   └── ...
       └── anomaly/             # Test-only data
           ├── file1.csv
           └── ...

**Important**:

* Training uses **only normal class** data
* Anomaly class is only used for testing/validation
* You can have multiple anomaly types in separate folders for testing

Available Models
----------------

**NPU-Optimized Models**:

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Model Name
     - Parameters
     - Description
   * - ``AD_500_NPU``
     - ~500
     - Small autoencoder
   * - ``AD_1k_NPU``
     - ~1,000
     - Medium
   * - ``AD_4k_NPU``
     - ~4,000
     - Large
   * - ``AD_20k_NPU``
     - ~20,000
     - Very large

**Specialized Models**:

* ``AD_Linear`` - Linear autoencoder
* ``Ondevice_Trainable_AD_Linear`` - On-device training capable

Threshold Selection
-------------------

The threshold determines when reconstruction error indicates an anomaly:

.. code-block:: text

   threshold = mean_train + k × std_train

Where ``k`` controls sensitivity:

* ``k=1``: High sensitivity (catches more anomalies, more false alarms)
* ``k=3``: Balanced (typical choice)
* ``k=4``: High specificity (fewer false alarms, may miss subtle anomalies)

ModelMaker tests multiple k values and reports results.

Metrics
-------

* **Precision** - Of detected anomalies, how many are real
* **Recall** - Of real anomalies, how many are detected
* **F1-Score** - Balance of precision and recall
* **AUC-ROC** - Overall discrimination ability

Semi-Supervised vs Supervised
-----------------------------

**Use Anomaly Detection (Semi-Supervised) when**:

* You only have normal data available
* Faults are rare and hard to collect
* You want to detect unknown failure modes

**Use Classification (Supervised) when**:

* You have labeled examples of all fault types
* You need to identify specific fault types
* Faults are well-defined and documented

Example: Motor Bearing Anomaly
------------------------------

.. code-block:: yaml

   common:
     task_type: 'generic_timeseries_anomalydetection'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'motor_bearing_ad'
     input_data_path: '/path/to/bearing_data'

   data_processing_feature_extraction:
     feature_extraction_name: 'Generic_256Input_RAW_256Feature_1Frame'
     variables: 3   # 3-axis vibration

   training:
     model_name: 'AD_4k_NPU'
     training_epochs: 50

Tips
----

* **Include diverse normal conditions** in training data (different loads, speeds)
* **Feature extraction** can help or hurt - try both raw and FFT-based
* **Start with k=3** for threshold, adjust based on false positive rate
* **Monitor reconstruction error distribution** during testing

See Also
--------

* :doc:`timeseries_classification` - Alternative supervised approach
* :doc:`/features/post_training_analysis` - Understanding results
