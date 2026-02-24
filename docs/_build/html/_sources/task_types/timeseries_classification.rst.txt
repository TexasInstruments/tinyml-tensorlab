===========================
Time Series Classification
===========================

Time series classification assigns discrete labels to sequences of time-ordered data.
This is the most common task type in Tiny ML Tensorlab.

Overview
--------

**What it does**: Takes a sequence of sensor readings and outputs a class label.

**Example**: Current waveform → "Normal" or "Arc Fault"

**Use cases**:

* Arc fault detection (DC/AC)
* Motor bearing fault classification
* Human activity recognition
* ECG arrhythmia detection
* Vibration-based fault diagnosis

Configuration
-------------

**Minimal config:**

.. code-block:: yaml

   common:
     task_type: 'generic_timeseries_classification'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'my_classification_data'
     input_data_path: '/path/to/data'

   data_processing_feature_extraction:
     feature_extraction_name: 'Generic_1024Input_FFTBIN_64Feature_8Frame'
     variables: 1

   training:
     model_name: 'CLS_1k_NPU'
     training_epochs: 20

   testing: {}
   compilation: {}

Dataset Format
--------------

Classification datasets use a ``classes/`` folder structure:

.. code-block:: text

   my_dataset/
   └── classes/
       ├── class_normal/
       │   ├── file1.csv
       │   ├── file2.csv
       │   └── ...
       ├── class_fault_A/
       │   └── ...
       └── class_fault_B/
           └── ...

* Folder names become class labels
* Each file contains time series data
* Supports CSV, TXT, NPY, PKL formats

See :doc:`/byod/classification_format` for details.

Available Models
----------------

**NPU-Optimized Models** (for F28P55, AM13E2, MSPM0G5187):

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - Model Name
     - Parameters
     - Use Case
   * - ``CLS_100_NPU``
     - ~100
     - Ultra-minimal
   * - ``CLS_500_NPU``
     - ~500
     - Very small
   * - ``CLS_1k_NPU``
     - ~1,000
     - Small
   * - ``CLS_4k_NPU``
     - ~4,000
     - Medium
   * - ``CLS_13k_NPU``
     - ~13,000
     - Large
   * - ``CLS_55k_NPU``
     - ~55,000
     - Very large

**Application-Specific Models (Edge AI Studio only):**

.. note::

   The following application-specific models are only available in TI's
   Edge AI Studio (GUI) and are not included in Tensorlab. Use the
   generic ``CLS_*`` / ``CLS_*_NPU`` models above for equivalent or
   better performance.

* ``ArcFault_model_200_t`` to ``ArcFault_model_1400_t`` - Arc fault detection
* ``MotorFault_model_1_t`` to ``MotorFault_model_3_t`` - Motor bearing fault
* ``FanImbalance_model_1_t`` to ``FanImbalance_model_3_t`` - Fan imbalance

Feature Extraction
------------------

Common presets for classification:

.. list-table::
   :widths: 50 50

   * - ``Generic_1024Input_FFTBIN_64Feature_8Frame``
     - FFT + binning, good general choice
   * - ``Generic_512Input_FFT_256Feature_1Frame``
     - Full FFT spectrum
   * - ``Generic_256Input_RAW_256Feature_1Frame``
     - Raw time domain
   * - ``FFT1024Input_256Feature_1Frame_Full_Bandwidth``
     - Arc fault optimized

Metrics
-------

Classification models are evaluated using:

* **Accuracy** - Percentage of correct predictions
* **F1-Score** - Harmonic mean of precision and recall
* **AUC-ROC** - Area under the ROC curve
* **Confusion Matrix** - Detailed class-by-class results

Example: Arc Fault Detection
----------------------------

.. code-block:: yaml

   common:
     task_type: 'generic_timeseries_classification'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'dc_arc_fault'
     input_data_path: 'https://software-dl.ti.com/C2000/esd/mcu_ai/01_03_00/datasets/dc_arc_fault_example_dsk.zip'

   data_processing_feature_extraction:
     feature_extraction_name: 'FFT1024Input_256Feature_1Frame_Full_Bandwidth'
     variables: 1

   training:
     model_name: 'CLS_1k_NPU'
     training_epochs: 20

Run with:

.. code-block:: bash

   ./run_tinyml_modelzoo.sh examples/dc_arc_fault/config.yaml

Tips
----

* Start with a preset ``feature_extraction_name`` before customizing
* Use Goodness of Fit test to evaluate dataset quality
* Try FFT-based features first for vibration/current data
* Balance your classes for best results
* Use ROC curves to select operating thresholds

See Also
--------

* :doc:`/byod/classification_format` - Dataset format
* :doc:`/features/feature_extraction` - Feature extraction details
* :doc:`/examples/arc_fault` - Arc fault example
