===========================
Time Series Classification
===========================

Time series classification assigns discrete labels to sequences of time-ordered data.
This is the most common task type in Tiny ML ModelZoo.

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

Task Types and Related Settings
-------------------------------

.. list-table::
   :widths: 35 65

   * - ``task_type``
     - ``generic_timeseries_classification`` for your own data. The task types ``arc_fault``,
       ``motor_fault``, ``ecg_classification``, ``blower_imbalance`` and ``pir_detection`` are
       classification variants used by specific examples; they behave the same way.
   * - ``target_module``
     - ``timeseries`` (inferred from ``task_type`` if you leave it out)
   * - Data folder
     - ``classes/`` (``dataset.data_dir`` default)
   * - Annotations
     - Optional, generated from ``split_type`` / ``split_factor``
   * - Early stopping
     - On by default in 1.5.0 (``training.early_stopping``, ``early_stopping_patience: 5``);
       see :doc:`/features/early_stopping`

Available Models
----------------

The models are defined in ``tinyml_modelzoo/model_descriptions/classification.py``
(``enabled_models_list``). Pick the model by ``training.model_name``.

**Generic NPU-optimized models** (designed for NPU devices such as F28P55 and F28P65; see
:doc:`/devices/index` for device support):

``CLS_100_NPU``, ``CLS_500_NPU``, ``CLS_1k_NPU``, ``CLS_1.2k_NPU``, ``CLS_1.5k_NPU``, ``CLS_1.9k_NPU``,
``CLS_2k_NPU``, ``CLS_2.8k_NPU``, ``CLS_3.1k_NPU``, ``CLS_3.9k_NPU``, ``CLS_4k_NPU``, ``CLS_4.2k_NPU``,
``CLS_5k_NPU``, ``CLS_6k_NPU``, ``CLS_8k_NPU``, ``CLS_13k_NPU``, ``CLS_20k_NPU``, ``CLS_24k_NPU``,
``CLS_40k_NPU``, ``CLS_55k_NPU``

The number in the name is the approximate parameter count (for example ``CLS_1k_NPU`` has about 1,000
parameters). Smaller models fit smaller devices. ``estimate_memory.py`` in the tinyml-modelzoo folder
(see :doc:`/features/memory_estimation`) estimates whether a model fits your device.

**Residual models**: ``CLS_ResAdd_3k``, ``CLS_ResCat_3k``

**Application-specific models** (used by the corresponding examples):

* ``ArcFault_model_200_t``, ``ArcFault_model_300_t``, ``ArcFault_model_700_t``, ``ArcFault_model_1400_t`` - arc fault detection
* ``MotorFault_model_1_t`` to ``MotorFault_model_3_t`` - motor bearing fault
* ``FanImbalance_model_1_t`` to ``FanImbalance_model_3_t`` - fan / blower imbalance
* ``ECG_55k_NPU`` - ECG classification
* ``PIRDetection_model_1_t`` - PIR detection
* ``ElectricalFault_model_40k_t``, ``GearboxFault_model_1.2k_t``, ``GearboxFault_model_1.5k_t``
* ``WiFiPresDet_4.7k_CDE`` - Wi-Fi CSI presence detection (:doc:`/examples/wifi_csi_presence_detection`)

For your own architecture see :doc:`/BRING_YOUR_OWN_MODEL`.

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

* **Accuracy** - Percentage of correct predictions (logged as ``Acc@1``)
* **F1-Score** - Harmonic mean of precision and recall
* **Confusion Matrix** - Detailed class-by-class results
* **ROC / AUC** - Computed by the training script for the per-class ROC plots

These are reported for the float model (``FloatTrain``), the quantized model (``QuantTrain``), the
best epoch of each, and the held-out test data. See :doc:`/getting_started/understanding_run_output`.

Example: Arc Fault Detection
----------------------------

.. code-block:: yaml

   common:
     task_type: 'arc_fault'
     target_device: 'F28P55'

   dataset:
     dataset_name: 'dc_arc_fault_example_dsk'
     # input_data_path is resolved from dataset_name for the shipped example

   data_processing_feature_extraction:
     feature_extraction_name: 'FFT1024Input_256Feature_1Frame_Full_Bandwidth'
     variables: 1

   training:
     model_name: 'CLS_1k_NPU'
     training_epochs: 10

This is the content of ``examples/dc_arc_fault/config_dsk.yaml`` (abridged). Run with:

.. code-block:: bash

   ./run_tinyml_modelzoo.sh examples/dc_arc_fault/config_dsk.yaml

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
* :doc:`/examples/generic_classification` - Generic classification example
* :doc:`/examples/motor_bearing_fault` - Motor bearing fault example
