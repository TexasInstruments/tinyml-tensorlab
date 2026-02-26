====================================
Anomaly Detection Dataset Format
====================================

This page describes how to format your dataset for anomaly detection tasks
(``generic_timeseries_anomalydetection``).

Folder Structure
----------------

For anomaly detection tasks, ModelMaker expects the dataset to be organized with
separate folders for **Normal** and **Anomaly** data:

.. code-block:: text

   dataset_name/
   |
   +-- classes/
   |   +-- Normal/
   |   |   +-- file1.csv
   |   |   +-- file2.csv
   |   |   +-- ...
   |   +-- Anomaly/
   |       +-- file1.csv
   |       +-- file2.csv
   |       +-- ...
   |
   +-- annotations/                          # Optional
       +-- file_list.txt                     # List of all files
       +-- instances_train_list.txt          # Training files (Normal only)
       +-- instances_val_list.txt            # Validation files (Normal only)
       +-- instances_test_list.txt           # Test files (Normal + Anomaly)

.. warning::

   **Training uses ONLY normal data.** The ``Anomaly/`` folder is excluded
   from training entirely. Anomaly data is used **only for testing** to
   evaluate detection performance.

Key points:

* **Normal/ folder**: Contains all samples representing normal operating
  conditions. All training and validation data comes from this folder.
* **Anomaly/ folder**: Contains samples representing anomalous behavior
  (faults, failures, defects). Used ONLY for testing, never for training.
* **annotations/ folder**: Optional. If not provided, ModelMaker automatically
  generates the annotation files based on your ``split_type`` and
  ``split_factor`` settings.
* Multiple anomaly types (e.g., imbalance, damage, bearing wear) all go in
  the same ``Anomaly/`` folder. The model treats them all as "not normal" and
  does not distinguish between different anomaly types.

Concrete Example
----------------

.. code-block:: text

   fan_blade_ad_dataset.zip/
   |
   +-- classes/
   |   +-- Normal/
   |   |   +-- normal_001.csv
   |   |   +-- normal_002.csv
   |   |   +-- ...
   |   |   +-- normal_100.csv              # 100 normal samples
   |   |
   |   +-- Anomaly/
   |       +-- imbalance_001.csv
   |       +-- imbalance_002.csv
   |       +-- damage_001.csv
   |       +-- obstruction_001.csv
   |       +-- ...
   |       +-- obstruction_005.csv         # 20 anomaly samples (mixed types)
   |
   +-- annotations/                         # Optional - auto-generated if missing
       +-- file_list.txt
       +-- instances_train_list.txt
       +-- instances_val_list.txt
       +-- instances_test_list.txt

Data Splitting Strategy
-----------------------

Normal data is split into train, validation, and test sets. Anomaly data is
used **only** in the test set.

.. list-table::
   :header-rows: 1
   :widths: 20 25 25 30

   * - Split
     - Normal Data
     - Anomaly Data
     - Purpose
   * - **Training**
     - 60% of Normal files
     - **None**
     - Learn what "normal" looks like
   * - **Validation**
     - 10% of Normal files
     - **None**
     - Monitor overfitting on normal patterns
   * - **Test**
     - 30% of Normal files
     - **All Anomaly files**
     - Evaluate detection performance

Configure splitting in your YAML:

.. code-block:: yaml

   dataset:
     split_type: 'amongst_files'        # or 'within_files'
     split_factor: [0.6, 0.1, 0.3]      # [train, val, test]

Two split methods are available:

* **amongst_files** (default): Files are divided into train, validation, and
  test sets. For example, with 100 normal files and ``split_factor: [0.6, 0.1, 0.3]``,
  you get 60 files for training, 10 for validation, and 30 for testing.
* **within_files**: Each file is split internally. All files appear in all
  splits with different portions. For example, each file contributes its first
  60% of samples to training, next 10% to validation, and last 30% to testing.

.. note::

   If you do not provide an ``annotations/`` folder, ModelMaker automatically
   generates annotation files. For ``amongst_files``, it randomly splits
   normal files according to the split factor and adds all anomaly files to
   the test list. For ``within_files``, it creates new files for each split
   portion.

What If You Don't Have Anomaly Data?
-------------------------------------

It is common, especially early in a project, to have collected normal operating
data but no examples of anomalies.

**What You CAN Do Without Anomaly Data:**

* **Train the model**: Training only requires normal data.
* **Set a conservative threshold**: Use ``mean_train + 3 * std_train`` as a
  starting point.
* **Deploy and monitor**: Deploy the model and monitor reconstruction error
  in real time. Samples exceeding the threshold can be flagged for review.

**What You CANNOT Do Without Anomaly Data:**

* **Measure anomaly detection performance**: Without anomaly samples, you
  cannot calculate recall, precision, or F1 score.
* **Validate detection capability**: You can only verify the model does not
  produce false alarms on normal data.

.. note::

   Once deployed, you can collect anomaly samples as they naturally occur.
   These can then be added to the ``Anomaly/`` folder and used to re-run the
   testing step to get proper metrics and adjust the threshold ``k`` value.

Datafile Format (CSV)
---------------------

ModelMaker supports ``.csv``, ``.txt``, ``.npy``, and ``.pkl`` file formats.
Within CSV files, two layouts are accepted:

**Headerless format** (no header row, no index column):

.. code-block:: text

   2078
   2136
   2117
   2077
   2029

**Headered format** (with column names, optionally with a time/index column):

.. code-block:: text

   Time,Vibration_X,Vibration_Y,Vibration_Z
   0.0000,-2753,-558,64376
   0.0001,-2551,-468,63910
   0.0002,-424,-427,64032

.. warning::

   Any column with the text ``time`` in its header (case-insensitive) is
   **automatically dropped** by ModelMaker. If you have a useful column that
   contains "time" in its name, rename it before using the dataset.

For multi-channel data (e.g., 3-axis accelerometer), specify the number of
variables in your configuration:

.. code-block:: yaml

   data_processing_feature_extraction:
     variables: 3   # X, Y, Z axes

See Also
--------

* :doc:`/task_types/anomaly_detection` - Anomaly detection task type overview
* :doc:`classification_format` - Classification dataset format
* :doc:`data_splitting` - Data splitting strategies
